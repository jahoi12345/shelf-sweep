"""Chain-independent book count from a run's 2D panorama.

The live association counts books by following each spine from frame to frame; when that chain breaks (blur, fast moves) one book becomes several
groups. The 2D panorama (pipeline/panorama.py) stitches the frames it could register into planar pieces ("components") and shows each book once per
piece. This module counts spines on those pieces instead of along the chain:

1. every group's polygon is already projected into its component's panorama coordinates (panorama.json markers). Optionally the local OBB spine
   detector (providers/local_detector's book model) is run on the component images as a second, independent spine source;
2. pieces that show the same shelf area are registered to each other (SIFT + RANSAC homography on the rectified component images, verified by
   inlier spread, a near-similarity check and photometric agreement), so every spine can be placed in one shared frame per connected set of pieces;
3. spines of the same piece (or of registered pieces) that land on top of each other are one book; the merge evidence (group ids, box, pieces) is kept;
4. groups whose frames are not in any piece (tracking holds) are not placed anywhere: only those the track count already trusts (>= 2 sightings,
   tier confident/likely) are added, the single sightings are reported and withheld, exactly as in the track best guess.

Output `panorama_count.json` in the new `<run>_panorama_vN` directory next to panorama.json (never overwrites an existing file's evidence: the file is
rewritten atomically, the panorama itself is untouched). `attach` puts the result into claim_packet.json in place (pipeline/panorama_attach.py).

    python -m pipeline.panorama_count --run runs/<run> [--panorama runs/<run>_panorama_vN] [--detect] [--out file]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCHEMA = 1
REG = dict(side_px=1600, nfeatures=3000, ratio=0.75, ransac_px=6.0, min_inliers=20, min_spread=0.12, max_scale=6.0, min_scale=1 / 6.0, max_aniso=3.0,
           max_perspective=0.35, min_ncc=0.3, min_overlap_frac=0.04,
           # Piece-pair coverage: when the plain match fails, both pieces are re-featured at the scale that puts their median spine at target_spine_px
           # (a far piece enlarged, a close-up shrunk) so SIFT compares spines of one size; and time-consecutive pieces are registered THROUGH their boundary frames
           # (frame-to-frame SIFT/RANSAC composed with the frames' homographies into their pieces). Every link passes the same NCC / overlap / shape checks.
           target_spine_px=160, max_side_px=4200, boundary_frames=3, boundary_window_s=3.0)
MERGE = dict(iou_same_piece=0.35, iou_cross_piece=0.25)
MIN_COMPONENT_FRAMES = 3          # components made of fewer frames are never counted from the panorama (their groups stay with the track count)
MIN_REGISTRATION = 0.75           # the packet only switches to the panorama count when at least this share of frames is in a piece
DETECT_CONF = 0.4
COLLAPSED_SCALE = 0.25           # piece spine height / native spine height below this: the chain shrank the mosaic (observed 0.01-0.17 on collapsed pieces, >= 0.39 on the rest)
FAR_FRACTION = 0.5               # native spine height of a piece below this share of the run median: far overview piece (evidence only)
MIN_DETECTOR_SUPPORT = 0.6       # the detector must see at least this share of the confident track groups placed on the pieces, else the pieces are too coarse to count
MAX_TILES = 20                    # detector passes per piece (time budget: the whole count must add < 60 s to the post-sweep queue)
TARGET_SPINE_H_PX = 300           # spine height (px) at the detector input where the OBB model works (live frames: 270-330 px at imgsz 1280)


# ------------------------------------------------------------------------------------------ geometry
def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    if (a[:, 0].max() < b[:, 0].min() or b[:, 0].max() < a[:, 0].min() or a[:, 1].max() < b[:, 1].min() or b[:, 1].max() < a[:, 1].min()):
        return 0.0
    aa, bb = _poly_area(a), _poly_area(b)
    if aa < 1e-6 or bb < 1e-6:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    return float(inter) / (aa + bb - float(inter)) if inter > 0 else 0.0


def _hull(p: Any) -> np.ndarray:
    return cv2.convexHull(np.asarray(p, np.float32)).reshape(-1, 2)


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def spine_size(poly: Any) -> tuple[float, float]:
    p = np.asarray(poly, float)
    e = np.linalg.norm(np.roll(p, -1, 0) - p, axis=1)
    return float(e.max()), float(e.min())


# ------------------------------------------------------------------------------------------ registering pieces to each other
def piece_spine_px(comp: dict[str, Any]) -> float | None:
    """Median spine height (px, full-resolution piece coordinates) of the groups placed on the piece with >= 2 sightings; None when fewer than three."""
    hs = [spine_size(m['polygon'])[0] for m in comp.get('markers', []) if m.get('polygon') and (m.get('sightings_total') or 1) >= 2]
    return float(np.median(hs)) if len(hs) >= 3 else None


def component_features(pan: Path, comp: dict[str, Any], cfg: dict[str, Any] = REG, spine_px: float | None = None) -> dict[str, Any]:
    """SIFT features of a rectified piece. Default: the long side scaled to side_px. With ``spine_px`` (the piece's median spine height): the scale that puts that spine at
    cfg['target_spine_px'] (capped at max_side_px and never above 1.0), so two pieces filmed from different distances are compared at one spine size."""
    img = cv2.imread(str(pan / comp['image']), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {'pts': np.zeros((0, 2), np.float32), 'desc': None, 'gray': None, 'scale': 1.0}
    scale = cfg['side_px'] / max(img.shape)
    if spine_px:
        scale = min(1.0, cfg['target_spine_px'] / max(spine_px, 1.0), cfg['max_side_px'] / max(img.shape))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = None
    cov = cv2.imread(str(pan / comp['coverage_mask']), cv2.IMREAD_GRAYSCALE) if comp.get('coverage_mask') else None
    if cov is not None:
        mask = (cv2.resize(cov, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255
    sift = cv2.SIFT_create(nfeatures=cfg['nfeatures'])
    k, d = sift.detectAndCompute(small, mask)
    return {'pts': np.float32([p.pt for p in k]) if k else np.zeros((0, 2), np.float32), 'desc': d, 'gray': small, 'mask': mask, 'scale': scale}


def _ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    a, b = cv2.GaussianBlur(a, (0, 0), 3).astype(np.float32)[m], cv2.GaussianBlur(b, (0, 0), 3).astype(np.float32)[m]
    if len(a) < 500 or a.std() < 1 or b.std() < 1:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _shape_ok(H: np.ndarray, fb_gray: np.ndarray, cfg: dict[str, Any], composed: bool = False) -> bool:
    """Near-similarity, bounded zoom and perspective, no flip (H in small-image coordinates of the pair). A mapping COMPOSED through boundary frames carries the two
    pieces' own rectifications (each a known homography), so it is legitimately more projective than a direct match: only the zoom bounds, a looser anisotropy (2x) and
    the no-flip rule apply there; the photometric agreement and the frame registration's inlier coverage are the evidence."""
    A = H[:2, :2] / H[2, 2]
    sv = np.linalg.svd(A, compute_uv=False)
    aniso = cfg['max_aniso'] * (2.0 if composed else 1.0)
    if not (cfg['min_scale'] <= sv[1] and sv[0] <= cfg['max_scale'] and sv[0] / max(sv[1], 1e-9) <= aniso) or np.linalg.det(A) <= 0:
        return False
    if composed:
        return True
    hb, wb = fb_gray.shape
    return max(abs(H[2, 0]) * wb, abs(H[2, 1]) * hb) / abs(H[2, 2]) <= cfg['max_perspective']


def _photometric(fa: dict[str, Any], fb: dict[str, Any], H: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float] | None:
    """(ncc, overlap) of piece B warped into A by H (small-image coordinates), or None when the overlap or the agreement is below the gates."""
    ha, wa = fa['gray'].shape
    wb_img = cv2.warpPerspective(fb['gray'], H, (wa, ha))
    mb = cv2.warpPerspective(np.full(fb['gray'].shape, 255, np.uint8), H, (wa, ha)) > 0
    mk = mb & (fa['mask'] > 0 if fa.get('mask') is not None else True)
    overlap = float(mk.sum()) / (wa * ha)
    ncc = _ncc(fa['gray'], wb_img, mk)
    if overlap < cfg['min_overlap_frac'] or ncc < cfg['min_ncc']:
        return None
    return ncc, overlap


def _to_full(fa: dict[str, Any], fb: dict[str, Any], H: np.ndarray) -> list[list[float]]:
    """H maps B's small-image coordinates into A's small-image coordinates; convert to full-resolution component pixels."""
    Sa, Sb = np.diag([fa['scale'], fa['scale'], 1.0]), np.diag([fb['scale'], fb['scale'], 1.0])
    Hfull = np.linalg.inv(Sa) @ H @ Sb
    return (Hfull / Hfull[2, 2]).tolist()


def register_pair(fa: dict[str, Any], fb: dict[str, Any], cfg: dict[str, Any] = REG) -> dict[str, Any] | None:
    """Homography mapping component B's pixels (full-resolution component coordinates) into A's, with its evidence, or None when the pair is not verified."""
    if fa['desc'] is None or fb['desc'] is None or len(fa['pts']) < 30 or len(fb['pts']) < 30:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(fb['desc'], fa['desc'], k=2)
    good = [x for x, y in (z for z in m if len(z) == 2) if x.distance < cfg['ratio'] * y.distance]
    if len(good) < cfg['min_inliers']:
        return None
    pb = np.float32([fb['pts'][g.queryIdx] for g in good])
    pa = np.float32([fa['pts'][g.trainIdx] for g in good])
    H, inl = cv2.findHomography(pb, pa, cv2.RANSAC, cfg['ransac_px'])
    if H is None:
        return None
    inl = inl.ravel().astype(bool)
    n = int(inl.sum())
    if n < cfg['min_inliers'] or not _shape_ok(H, fb['gray'], cfg):
        return None
    for pts, g in ((pa[inl], fa['gray']), (pb[inl], fb['gray'])):         # the inliers must be spread over the picture, not one texture patch
        h, w = g.shape
        if min(np.ptp(pts[:, 0]) / w, np.ptp(pts[:, 1]) / h) < cfg['min_spread'] * 0.5 and max(np.ptp(pts[:, 0]) / w, np.ptp(pts[:, 1]) / h) < cfg['min_spread']:
            return None
    ph = _photometric(fa, fb, H, cfg)
    if ph is None:
        return None
    return {'H_b_to_a': _to_full(fa, fb, H), 'inliers': n, 'matches': len(good), 'ncc': round(ph[0], 3), 'overlap_frac': round(ph[1], 3)}


def register_via_boundary_frames(fa: dict[str, Any], fb: dict[str, Any], ca: dict[str, Any], cb: dict[str, Any], run_dir: Path, frames: dict[str, dict[str, Any]],
                                 cfg: dict[str, Any] = REG) -> dict[str, Any] | None:
    """Piece B -> piece A through their boundary frames: the last cfg['boundary_frames'] frames of the earlier piece and the first of the later one (within boundary_window_s)
    are registered frame-to-frame (pipeline.instances.register_frames: SIFT/RANSAC, inlier coverage, zoom bounds), the homography is composed with the frames' own
    homographies into their pieces, and the composed piece-to-piece mapping must pass the same shape and photometric checks as a direct match. Best inliers wins."""
    from pipeline.instances import CROSS, register_frames
    from pipeline.detect import resolve_evidence
    first, second = (ca, cb) if ca['source_frames'][0]['timestamp_s'] <= cb['source_frames'][0]['timestamp_s'] else (cb, ca)
    t_cut = second['source_frames'][0]['timestamp_s']
    tail = [f for f in first['source_frames'] if f['timestamp_s'] >= t_cut - cfg['boundary_window_s']][-cfg['boundary_frames']:]
    head = [f for f in second['source_frames'] if f['timestamp_s'] <= t_cut + cfg['boundary_window_s']][:cfg['boundary_frames']]
    if not tail or not head:
        return None
    sift = cv2.SIFT_create(nfeatures=CROSS['nfeatures'])
    cache: dict[str, tuple] = {}

    def feat(f):
        if f['frame_id'] not in cache:
            im = cv2.imread(str(resolve_evidence(run_dir, f['image_path'])), cv2.IMREAD_GRAYSCALE)
            if im is None:
                cache[f['frame_id']] = (None, None, 1.0)
            else:
                sc = min(0.5, CROSS['max_width_px'] / im.shape[1])
                g = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                cache[f['frame_id']] = (g, sift.detectAndCompute(g, None), sc)
        return cache[f['frame_id']]
    best = None
    for f1 in tail:
        for f2 in head:
            g1, k1, sc = feat(f1)
            g2, k2, _ = feat(f2)
            if g1 is None or g2 is None:
                continue
            reg = register_frames(g1, k1, g2, k2, sc, CROSS)           # frame f1 -> frame f2, full-resolution native pixels
            if reg is None:
                continue
            H1, H2 = np.asarray(f1['original_to_panorama_3x3'], float), np.asarray(f2['original_to_panorama_3x3'], float)
            H_second_to_first = H1 @ np.linalg.inv(reg['H']) @ np.linalg.inv(H2)          # second piece pixels -> first piece pixels
            Hfull = H_second_to_first if first is ca else np.linalg.inv(H_second_to_first)  # -> B pixels into A pixels
            Sa, Sb = np.diag([fa['scale'], fa['scale'], 1.0]), np.diag([fb['scale'], fb['scale'], 1.0])
            Hs = Sa @ Hfull @ np.linalg.inv(Sb)
            if not np.isfinite(Hs).all() or abs(Hs[2, 2]) < 1e-12:
                continue
            Hs = Hs / Hs[2, 2]
            if not _shape_ok(Hs, fb['gray'], cfg, composed=True):
                continue
            ph = _photometric(fa, fb, Hs, cfg)
            if ph is None:
                continue
            cand = {'H_b_to_a': (Hfull / Hfull[2, 2]).tolist(), 'inliers': reg['inliers'], 'matches': reg['matches'], 'ncc': round(ph[0], 3), 'overlap_frac': round(ph[1], 3),
                    'via': 'boundary_frames', 'frames': [f1['frame_id'], f2['frame_id']], 'frame_zoom': reg['scale']}
            if best is None or cand['inliers'] > best['inliers']:
                best = cand
    return best


def register_components(pan: Path, comps: list[dict[str, Any]], cfg: dict[str, Any] = REG, progress=None, run_dir: Path | None = None,
                        frames: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Pairwise registration of all pieces, then a maximum-inlier spanning forest. Returns {'links': [...], 'sheets': [[ids]], 'to_root': {id: (root, 3x3 or None)}}.
    Per pair, in order: the plain match; the spine-normalized match (both pieces re-featured at one spine size); for time-consecutive pieces, the match through
    their boundary frames (needs ``run_dir``). Each link records ``via``."""
    feats = {}
    for c in comps:
        feats[c['component_id']] = component_features(pan, c, cfg)
    by_id = {c['component_id']: c for c in comps}
    norm: dict[str, dict[str, Any]] = {}

    def normalized(cid):
        if cid not in norm:
            sp = piece_spine_px(by_id[cid])
            norm[cid] = component_features(pan, by_id[cid], cfg, spine_px=sp) if sp else feats[cid]
        return norm[cid]
    order = sorted(comps, key=lambda c: (c['source_frames'][0]['timestamp_s'] if c.get('source_frames') and isinstance(c['source_frames'][0], dict) else 0.0, c['component_id']))
    consecutive = {frozenset((order[k]['component_id'], order[k + 1]['component_id'])) for k in range(len(order) - 1)}
    ids = list(feats)
    links = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            r = register_pair(feats[a], feats[b], cfg)
            via = 'plain'
            if r is None:
                r = register_pair(normalized(a), normalized(b), cfg)
                via = 'spine_normalized'
            if r is None and run_dir is not None and frozenset((a, b)) in consecutive and by_id[a].get('source_frames') and isinstance(by_id[a]['source_frames'][0], dict):
                r = register_via_boundary_frames(feats[a], feats[b], by_id[a], by_id[b], Path(run_dir), frames or {}, cfg)
                if r is None:            # the photometric check is taken in A's frame: try the other orientation and invert the verified mapping
                    r = register_via_boundary_frames(feats[b], feats[a], by_id[b], by_id[a], Path(run_dir), frames or {}, cfg)
                    if r is not None:
                        Hinv = np.linalg.inv(np.asarray(r['H_b_to_a'], float))
                        r = {**r, 'H_b_to_a': (Hinv / Hinv[2, 2]).tolist(), 'verified_in': b}
                via = 'boundary_frames'
            if r:
                links.append({'a': a, 'b': b, 'via': r.get('via') or via, **{k: v for k, v in r.items() if k != 'via'}})
        if progress:
            progress(i + 1, len(ids))
    parent = {c: c for c in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    tree = []
    for lk in sorted(links, key=lambda l: -l['inliers']):
        ra, rb = find(lk['a']), find(lk['b'])
        if ra != rb:
            parent[ra] = rb
            tree.append(lk)
    adj: dict[str, list[tuple[str, np.ndarray]]] = {c: [] for c in ids}
    for lk in tree:
        H = np.asarray(lk['H_b_to_a'], float)
        adj[lk['a']].append((lk['b'], H))                # b -> a
        adj[lk['b']].append((lk['a'], np.linalg.inv(H)))  # a -> b
    to_root: dict[str, tuple[str, np.ndarray]] = {}
    sheets = []
    for c in ids:
        if c in to_root:
            continue
        to_root[c] = (c, np.eye(3))
        stack, members = [c], [c]
        while stack:
            u = stack.pop()
            for v, Hvu in adj[u]:                       # maps v pixels into u pixels
                if v not in to_root:
                    to_root[v] = (c, to_root[u][1] @ Hvu)
                    stack.append(v)
                    members.append(v)
        sheets.append(members)
    return {'links': links, 'tree': tree, 'sheets': sheets, 'to_root': to_root}


# ------------------------------------------------------------------------------------------ merging spines
def merge_spines(spines: list[dict[str, Any]], iou_same: float = MERGE['iou_same_piece'], iou_cross: float = MERGE['iou_cross_piece']) -> list[list[int]]:
    """Clusters of spine indices that sit on top of each other (same piece, or registered pieces in the shared frame). Spines carry 'poly' (shared frame) and 'comp'."""
    n = len(spines)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    hulls = [_hull(s['poly']) if len(s['poly']) >= 3 else None for s in spines]
    order = sorted(range(n), key=lambda i: hulls[i][:, 0].min() if hulls[i] is not None else 0)
    for oi, i in enumerate(order):
        if hulls[i] is None:
            continue
        xmax = hulls[i][:, 0].max()
        for j in order[oi + 1:]:
            if hulls[j] is None:
                continue
            if hulls[j][:, 0].min() > xmax:
                break
            thr = iou_same if spines[i]['comp'] == spines[j]['comp'] else iou_cross
            if find(i) != find(j) and _iou(hulls[i], hulls[j]) >= thr:
                parent[find(i)] = find(j)
    out: dict[int, list[int]] = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


# ------------------------------------------------------------------------------------------ spines detected on the pieces
def detection_plan(comp: dict[str, Any], width: int, height: int, ref_ratio: float | None) -> dict[str, Any]:
    """Scale and window for the OBB model on one piece. The model works on spines about TARGET_SPINE_H_PX tall at its input; the spine height of the
    piece comes from the run's own group polygons (projected into the piece), so a far view is enlarged and a close-up shrunk instead of both being
    resized to the same long side. A piece with no placed group uses the run's median spine-height/long-side ratio."""
    hs = [spine_size(m['polygon'])[0] for m in comp.get('markers', []) if m.get('polygon') and (m.get('sightings_total') or 1) >= 2]
    long_side = max(width, height)
    if len(hs) >= 3:
        h, basis = float(np.median(hs)), 'median_group_polygon_height'
    elif ref_ratio:
        h, basis = ref_ratio * long_side, 'run_median_ratio'
    else:
        h, basis = 0.22 * long_side, 'default_ratio'
    scale = min(3.0, max(0.25, TARGET_SPINE_H_PX / max(h, 1.0)))
    while scale > 0.25 and len(_tiles(width, height, int(math.ceil(1280 / scale)))) > MAX_TILES:       # time cap: enlarge the window (lower the scale) instead of tiling a far view into dozens of passes
        scale *= 0.93
    return {'spine_h_px': round(h, 1), 'basis': basis, 'scale': round(scale, 3), 'window_px': int(math.ceil(1280 / scale)), 'tiles': len(_tiles(width, height, int(math.ceil(1280 / scale))))}


def _tiles(w: int, h: int, size: int, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    if w <= size and h <= size:
        return [(0, 0, w, h)]

    def starts(length):
        final = max(0, length - size)
        return sorted(set(list(range(0, final + 1, max(1, int(size * (1 - overlap))))) + [final]))
    return [(x, y, min(size, w - x), min(size, h - y)) for y in starts(h) for x in starts(w)]


def detect_worker(job_path: str, out_path: str) -> None:
    """Runs in the optional detector interpreter (ultralytics + the book OBB weights). job: {weights_dir, conf, images: [{id, path, scale, window_px}]}."""
    import pathlib
    import torch
    import ultralytics.utils
    from ultralytics import YOLO
    job = json.loads(Path(job_path).read_text())
    wdir = pathlib.Path(job['weights_dir'])
    ultralytics.utils.WEIGHTS_DIR = wdir
    torch.set_num_threads(int(job.get('threads', 4)))
    model = YOLO(str(wdir / 'book-spine-best.pt')).to('cpu')
    out = {}
    for item in job['images']:
        img = cv2.imread(item['path'])
        if img is None:
            out[item['id']] = {'error': 'unreadable image', 'detections': []}
            continue
        H, W = img.shape[:2]
        s, dets, t0 = float(item['scale']), [], time.perf_counter()
        for (x, y, tw, th) in _tiles(W, H, int(item['window_px'])):
            imgsz = int(min(2560, max(320, round(max(tw, th) * s / 32) * 32)))
            r = model.predict(img[y:y + th, x:x + tw], imgsz=imgsz, conf=float(job['conf']), device='cpu', verbose=False)[0]
            if r.obb is None:
                continue
            for p, c in zip(r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2), r.obb.conf.cpu().numpy()):
                q = p + np.array([x, y], np.float32)
                lo, hi = q.min(axis=0), q.max(axis=0)
                inner = 4.0                           # a box touching a tile edge that is not the picture's edge is a fragment: the overlapping tile holds it whole
                cut = ((x > 0 and lo[0] < x + inner) or (y > 0 and lo[1] < y + inner) or (x + tw < W and hi[0] > x + tw - inner) or (y + th < H and hi[1] > y + th - inner))
                if not cut:
                    dets.append({'poly': np.round(q, 1).tolist(), 'conf': round(float(c), 4)})
        out[item['id']] = {'detections': dets, 'size': [W, H], 'seconds': round(time.perf_counter() - t0, 2)}
    Path(out_path).write_text(json.dumps(out))


def default_detector_runtime() -> tuple[Path, Path]:
    from providers.stages import DEFAULT_PYTHON, DEFAULT_WEIGHTS
    python = Path(os.environ.get('LIBRARY_DETECTOR_PYTHON') or DEFAULT_PYTHON)
    weights = Path(os.environ.get('LIBRARY_DETECTOR_WEIGHTS') or DEFAULT_WEIGHTS)
    if python.is_file() and (weights / 'book-spine-best.pt').is_file():
        return python, weights
    raise FileNotFoundError(f'local spine detector runtime ({python} + {weights}/book-spine-best.pt) not found: run scripts/setup_local_models.sh')


def detect_components(pan: Path, comps: list[dict[str, Any]], *, conf: float = 0.25, threads: int = 4, timeout_s: float = 240.0) -> dict[str, Any]:
    """{component_id: {'detections': [{poly, conf}], 'plan': {...}, 'size': [w, h], 'seconds': s}} from one detector subprocess (niced by the caller)."""
    python, weights = default_detector_runtime()
    sizes = {}
    for c in comps:
        sizes[c['component_id']] = (int(c.get('width_px') or 0), int(c.get('height_px') or 0))
    ratios = [spine_size(m['polygon'])[0] / max(sizes[c['component_id']]) for c in comps for m in c.get('markers', [])
              if m.get('polygon') and (m.get('sightings_total') or 1) >= 2 and max(sizes[c['component_id']]) > 0]
    ref = float(np.median(ratios)) if ratios else None
    plans = {c['component_id']: detection_plan(c, *sizes[c['component_id']], ref) for c in comps}
    job = {'weights_dir': str(weights), 'conf': conf, 'threads': threads,
           'images': [{'id': c['component_id'], 'path': str(pan / c['image']), 'scale': plans[c['component_id']]['scale'], 'window_px': plans[c['component_id']]['window_px']} for c in comps]}
    with tempfile.TemporaryDirectory() as tmp:
        jp, op = Path(tmp) / 'job.json', Path(tmp) / 'out.json'
        jp.write_text(json.dumps(job))
        env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in ('KEY', 'TOKEN', 'SECRET', 'CREDENTIAL', 'PASSWORD', 'AUTH'))}
        env.update(YOLO_CONFIG_DIR=str(Path(tmp) / 'yolo'), MPLCONFIGDIR=str(Path(tmp) / 'mpl'), YOLO_AUTOINSTALL='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        subprocess.run([str(python), '-u', '-m', 'pipeline.panorama_count', '--detect-worker', str(jp), str(op)], cwd=Path(__file__).resolve().parents[1], env=env,
                       check=True, capture_output=True, timeout=timeout_s)
        raw = json.loads(op.read_text())
    return {cid: {**raw.get(cid, {'detections': []}), 'plan': plans[cid]} for cid in plans}


# ------------------------------------------------------------------------------------------ the count
def latest_panorama(run_dir: Path) -> Path | None:
    cands = sorted(run_dir.parent.glob(run_dir.name + '_panorama_v*'), key=lambda p: int(p.name.rsplit('_v', 1)[1]) if p.name.rsplit('_v', 1)[1].isdigit() else -1)
    for p in reversed(cands):
        if (p / 'panorama.json').is_file():
            return p
    return None


def _box(poly: Any) -> list[float]:
    p = np.asarray(poly, float)
    return [round(float(p[:, 0].min()), 1), round(float(p[:, 1].min()), 1), round(float(p[:, 0].max()), 1), round(float(p[:, 1].max()), 1)]


def _center_in(poly: np.ndarray, pt: np.ndarray) -> bool:
    return cv2.pointPolygonTest(_hull(poly).astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def _fuse(det: dict[str, Any], comps: list[dict[str, Any]], reg: dict[str, Any], conf: float) -> list[dict[str, Any]]:
    """Fused panorama spines at one confidence: [{'cluster': [member idx], 'rep': idx}] plus the flat member list under key 'members'."""
    members = []
    for c in comps:
        cid = c['component_id']
        root, H = reg['to_root'][cid]
        for k, d in enumerate(det.get(cid, {}).get('detections', [])):
            if d['conf'] >= conf:
                members.append({'comp': cid, 'root': root, 'k': k, 'conf': d['conf'], 'poly_comp': d['poly'], 'poly': _apply(H, np.asarray(d['poly'], float)).tolist()})
    spines = []
    for root in sorted({m['root'] for m in members}):
        idx = [i for i, m in enumerate(members) if m['root'] == root]
        sub = [{'poly': members[i]['poly'], 'comp': members[i]['comp']} for i in idx]
        for cl in merge_spines(sub):
            ids = [idx[j] for j in cl]
            spines.append({'members': ids, 'rep': max(ids, key=lambda i: members[i]['conf'])})
    return members, spines


def classify_pieces(comps: list[dict[str, Any]], native_h: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Per piece: 'normal' (the detector can count it), 'collapsed' (its chain shrank the mosaic: spines are < COLLAPSED_SCALE of their native size, the detector sees nothing; its
    trusted track groups are the count) or 'far' (the frames show spines < FAR_FRACTION of the run's median native spine height: an overview that the close pieces repeat, evidence only)."""
    run_med = float(np.median(list(native_h.values()))) if native_h else 0.0
    out = {}
    for c in comps:
        ms = [m for m in c.get('markers', []) if m.get('polygon') and m['group_id'] in native_h and (m.get('sightings_total') or 1) >= 2]
        if len(ms) < 3 or run_med <= 0:
            out[c['component_id']] = {'class': 'normal', 'reason': 'too few confident groups to judge the piece scale', 'piece_scale': None, 'native_vs_run': None}
            continue
        ph = float(np.median([spine_size(m['polygon'])[0] for m in ms]))
        nh = float(np.median([native_h[m['group_id']] for m in ms]))
        scale, rel = ph / max(nh, 1e-6), nh / run_med
        if scale < COLLAPSED_SCALE:
            cls, why = 'collapsed', f'spines are {scale:.2f} of their native size in this piece (< {COLLAPSED_SCALE}): the stitching chain shrank the mosaic'
        elif rel < FAR_FRACTION:
            cls, why = 'far', f'spines in the frames are {rel:.2f} of the run median (< {FAR_FRACTION}): a far view, evidence only'
        else:
            cls, why = 'normal', None
        out[c['component_id']] = {'class': cls, 'reason': why, 'piece_scale': round(scale, 3), 'native_vs_run': round(rel, 3)}
    return out


def build_count(run_dir: str | Path, pan_dir: str | Path | None = None, *, detect: bool = True, conf: float = DETECT_CONF, progress=None,
                detections: dict[str, Any] | None = None, registration: dict[str, Any] | None = None, room_pan: bool | dict[str, Any] = True) -> dict[str, Any]:
    """The chain-independent count of one run. See the module docstring. Raises FileNotFoundError when there is no 2D panorama."""
    from concurrent.futures import ThreadPoolExecutor
    t_start = time.perf_counter()
    run_dir = Path(run_dir)
    pan = Path(pan_dir) if pan_dir else latest_panorama(run_dir)
    if pan is None or not (pan / 'panorama.json').is_file():
        raise FileNotFoundError(f'no 2D panorama for {run_dir.name}')
    P = json.loads((pan / 'panorama.json').read_text())
    inv = json.loads((run_dir / 'inventory.json').read_text())['data']
    tier = {b['instance_id']: b.get('count_tier') for b in inv['books']}
    sightings = {b['instance_id']: len(b.get('observation_ids') or []) for b in inv['books']}
    native_h = {}
    for b in inv['books']:
        if b.get('polygon_xy'):
            native_h[b['instance_id']] = spine_size(b['polygon_xy'])[0]
        elif b.get('bbox_xyxy'):
            x0, y0, x1, y1 = b['bbox_xyxy']
            native_h[b['instance_id']] = max(x1 - x0, y1 - y0)
    all_comps = P['components']
    comps = [c for c in all_comps if len(c['source_frames']) >= MIN_COMPONENT_FRAMES and c.get('image')]
    small = [c for c in all_comps if c not in comps]
    pcls = classify_pieces(comps, native_h)
    # Pieces made mostly of room-pan frames (pipeline.room_pan: the floor/ceiling/wall pan of the same sweep) are not counted, not detected, not used for merges
    from pipeline import room_pan as RP
    rp = (room_pan if isinstance(room_pan, dict) else RP.classify_run(run_dir)) if room_pan else None
    pan_frames = set((rp or {}).get('room_pan_frames') or [])
    room_pan_pieces = {}
    for c in comps:
        fids = [f.get('frame_id') if isinstance(f, dict) else f for f in c['source_frames']]
        share = sum(1 for f in fids if f in pan_frames) / len(fids) if fids else 0.0
        if pan_frames and share > RP.PIECE_SHARE:
            room_pan_pieces[c['component_id']] = round(share, 3)
            pcls[c['component_id']] = {**pcls[c['component_id']], 'class': 'room_pan', 'class_before_room_pan': pcls[c['component_id']]['class'],
                                       'reason': f'{RP.REASON}: {share:.0%} of its {len(fids)} frames are room-pan frames (> {RP.PIECE_SHARE:.0%}); not counted, evidence only'}
    normal = [c for c in comps if pcls[c['component_id']]['class'] == 'normal']
    n_frames = len(P['source_frames'])
    registered_ratio = round(1 - len(P.get('unregistered_frames') or []) / n_frames, 4) if n_frames else 0.0
    counted_frames = sum(len(c['source_frames']) for c in comps)
    if progress:
        progress('registering and detecting', 0.1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_reg = None if registration is not None else pool.submit(register_components, pan, normal, REG, None, run_dir, None)
        f_det = pool.submit(detect_components, pan, normal, conf=0.25) if detect and detections is None and normal else None
        reg = registration if registration is not None else f_reg.result()
        t_reg = time.perf_counter() - t_start
        det = detections if detections is not None else (f_det.result() if f_det else {c['component_id']: {'detections': [], 'plan': None} for c in normal})
    t_all = time.perf_counter() - t_start
    normal = [c for c in normal if c['component_id'] in reg['to_root']]
    sens = {}
    for cf in sorted({0.5, 0.4, 0.3, conf}, reverse=True):
        _, sp = _fuse(det, normal, reg, cf)
        sens[cf] = len(sp)
    members, spines = _fuse(det, normal, reg, conf)
    # groups of the track count placed in the pieces (marker polygons are in piece coordinates -> shared sheet frame)
    groups = []
    for c in normal:
        root, H = reg['to_root'][c['component_id']]
        for m in c.get('markers', []):
            if m.get('polygon'):
                groups.append({'id': m['group_id'], 'comp': c['component_id'], 'root': root, 'poly': _apply(H, np.asarray(m['polygon'], float)),
                               'sightings': int(m.get('sightings_total') or sightings.get(m['group_id'], 1)), 'tier': tier.get(m['group_id'])})
    by_spine: dict[int, list[dict[str, Any]]] = {}
    unsupported = []
    spine_polys = [[(_hull(members[i]['poly']), members[i]['root']) for i in sp['members']] for sp in spines]
    for g in groups:
        gh, best, best_iou = _hull(g['poly']), None, 0.0
        gc = g['poly'].mean(axis=0)
        for si, plist in enumerate(spine_polys):
            for ph, root in plist:
                if root != g['root'] or ph[:, 0].max() < gh[:, 0].min() or ph[:, 0].min() > gh[:, 0].max():
                    continue
                v = _iou(gh, ph)
                if v == 0.0 and _center_in(ph, gc):
                    v = 0.05
                if v > best_iou:
                    best, best_iou = si, v
        if best is not None and best_iou >= 0.05:
            by_spine.setdefault(best, []).append({'id': g['id'], 'sightings': g['sightings'], 'tier': g['tier'], 'iou': round(best_iou, 2)})
        else:
            unsupported.append(g)
    confident = [g for g in groups if g['sightings'] >= 2]
    support = (sum(1 for g in confident if g not in unsupported) / len(confident)) if confident else None
    merges = []
    for si, gl in sorted(by_spine.items()):
        if len(gl) >= 2:
            rep = members[spines[si]['rep']]
            merges.append({'panorama_spine': f'{rep["comp"]}-s{si:03d}', 'component': rep['comp'], 'box_xyxy': _box(rep['poly_comp']), 'confidence': round(rep['conf'], 3),
                           'groups': sorted(gl, key=lambda x: -x['sightings']), 'duplicates_removed': len(gl) - 1,
                           'evidence': f'{len(gl)} track groups ({", ".join(x["id"] for x in gl[:6])}{"..." if len(gl) > 6 else ""}) fall on one spine of {rep["comp"]} at {_box(rep["poly_comp"])}'})
    # every counted spine with its detector boxes in PIECE coordinates (the frame of panorama.json markers/overlays) and the groups the count placed on it: the instance layer's
    # reconciliation (pipeline.instances.reconcile) places single sightings on these spines; ids match merge_evidence ('<rep piece>-s<index>')
    spine_rows = []
    for si, sp in enumerate(spines):
        rep = members[sp['rep']]
        spine_rows.append({'panorama_spine': f'{rep["comp"]}-s{si:03d}', 'component': rep['comp'], 'confidence': round(rep['conf'], 3),
                           'members': [{'component': members[i]['comp'], 'conf': round(members[i]['conf'], 3), 'poly': np.round(np.asarray(members[i]['poly_comp'], float), 1).tolist()}
                                       for i in sp['members']],
                           'groups': sorted(({'id': g['id'], 'iou': g['iou']} for g in by_spine.get(si, [])), key=lambda x: -x['iou'])})
    # collapsed pieces: their trusted track groups are the count (the mosaic cannot resolve spines); far pieces: evidence only
    fallback_groups, far_groups = [], []
    for c in comps:
        k = pcls[c['component_id']]['class']
        rows = [{'id': m['group_id'], 'sightings': int(m.get('sightings_total') or 1), 'tier': tier.get(m['group_id'])} for m in c.get('markers', []) if m.get('polygon')]
        if k == 'collapsed':
            fallback_groups += [r for r in rows if r['tier'] in ('confident', 'likely')]
        elif k == 'far':
            far_groups += rows
    # groups of frames that are in no counted piece: placed nowhere, so only those the track count itself trusts are added
    placed_ids = {m['group_id'] for c in comps for m in c.get('markers', []) if m.get('polygon')}
    outside = [{'id': b['instance_id'], 'tier': b.get('count_tier'), 'sightings': len(b.get('observation_ids') or [])} for b in inv['books'] if b['instance_id'] not in placed_ids]
    pan_only = {b['instance_id'] for b in inv['books'] if pan_frames and b.get('observation_ids') and all(o.split('-o')[0] in pan_frames for o in b['observation_ids'])}
    trusted = [g for g in outside if g['tier'] in ('confident', 'likely') and g['id'] not in pan_only]   # a group seen only in the room pan is not added
    comp_rows = []
    for c in comps:
        cid = c['component_id']
        k = pcls[cid]
        mem = [i for i, m in enumerate(members) if m['comp'] == cid]
        sp_here = [si for si, sp in enumerate(spines) if members[sp['rep']]['comp'] == cid]
        d = det.get(cid, {})
        comp_rows.append({'component_id': cid, 'frames': len(c['source_frames']), 'size_px': [c.get('width_px'), c.get('height_px')], 'class': k['class'], 'class_reason': k['reason'],
                          'piece_scale': k['piece_scale'], 'native_spine_vs_run_median': k['native_vs_run'], 'sheet': reg['to_root'][cid][0] if cid in reg['to_root'] else None,
                          'plan': d.get('plan'), 'detect_seconds': d.get('seconds'), 'spines_detected': len(mem), 'spines_counted': len(sp_here) if k['class'] == 'normal' else 0,
                          'groups_placed': sum(1 for m in c.get('markers', []) if m.get('polygon')), 'coverage_fraction': c.get('coverage_fraction'),
                          'counted_from': {'normal': 'panorama detector', 'collapsed': 'trusted track groups', 'far': 'evidence only (not counted)',
                                           'room_pan': 'not counted (room-pan view only)'}[k['class']]})
    panorama_spines = len(spines)
    extra = len(trusted)
    fallback = len(fallback_groups)
    usable, why = True, None
    if not detect:
        usable, why = False, 'detector not run'
    elif not normal:
        usable, why = False, 'no piece the detector can count'
    elif support is not None and support < MIN_DETECTOR_SUPPORT:
        usable, why = False, f'the detector sees only {support:.0%} of the spines the track count trusts on these pieces (needs >= {MIN_DETECTOR_SUPPORT:.0%}): panorama resolution too low'
    total = panorama_spines + fallback + extra
    result = {
        'schema_version': SCHEMA, 'artifact_type': 'panorama_count', 'run_id': run_dir.name, 'panorama_run': pan.name,
        'estimator': 'panorama_spine_detector', 'computed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'method': ('Local OBB spine detector run on the stitched 2D panorama pieces at a per-piece scale (spine height from the run\'s own group polygons); boxes of the same spine fused '
                   'inside a piece; pieces registered to each other by SIFT+RANSAC (verified) and fused across; pieces whose mosaic collapsed are counted from their trusted track groups; '
                   'far overview pieces are evidence only; plus track groups of unregistered frames the track count trusts. Piece registration: plain SIFT, then spine-normalized '
                   'SIFT, then (time-consecutive pieces) composition through the boundary frames, every link verified by the same shape and photometric checks.'),
        'panorama_count': total, 'panorama_spines': panorama_spines, 'collapsed_piece_groups': fallback, 'extra_unregistered_groups': extra,
        'count_if_collapsed_pieces_not_counted': panorama_spines + extra,
        'count_if_confidence': {f'{k:.2f}': v + fallback + extra for k, v in sorted(sens.items(), reverse=True)},
        'range_panorama': [min(sens.values()) + fallback + extra, max(sens.values()) + fallback + extra],
        'usable': usable, 'unusable_reason': why, 'detector_support': None if support is None else round(support, 3),
        'registration': {'frames_total': n_frames, 'frames_unregistered': len(P.get('unregistered_frames') or []), 'registered_ratio': registered_ratio,
                         'frames_in_counted_pieces': counted_frames, 'pieces_total': len(all_comps), 'pieces_counted': len(comps), 'pieces_too_small': len(small),
                         'pieces_normal': len(normal), 'pieces_room_pan': len(room_pan_pieces), 'pieces_collapsed': sum(1 for v in pcls.values() if v['class'] == 'collapsed'), 'pieces_far': sum(1 for v in pcls.values() if v['class'] == 'far')},
        'parameters': {'detect_confidence': conf, 'merge_iou_same_piece': MERGE['iou_same_piece'], 'merge_iou_cross_piece': MERGE['iou_cross_piece'], 'min_piece_frames': MIN_COMPONENT_FRAMES,
                       'min_registration': MIN_REGISTRATION, 'target_spine_h_px': TARGET_SPINE_H_PX, 'max_tiles': MAX_TILES, 'min_detector_support': MIN_DETECTOR_SUPPORT,
                       'collapsed_scale': COLLAPSED_SCALE, 'far_fraction': FAR_FRACTION},
        'panorama_components': comp_rows,
        'sheets': [{'sheet': root, 'pieces': members_} for root, members_ in ((m[0], m) for m in reg['sheets'] if len(m) > 1)],
        'piece_links': [{k: v for k, v in l.items() if k != 'H_b_to_a'} for l in reg['tree']],
        'unregistered_groups': {'groups_in_no_counted_piece': len(outside), 'added_to_count': extra, 'added_ids': [g['id'] for g in trusted],
                                'withheld_single_sightings': len(outside) - extra,
                                'rule': 'a group whose frames are in no counted piece cannot be placed; it is added only when the track count itself trusts it (tier confident/likely), single sightings are withheld'},
        'collapsed_pieces': {'groups_counted': fallback, 'ids': [g['id'] for g in fallback_groups],
                             'by_piece': {c['component_id']: sum(1 for m in c.get('markers', []) if m['group_id'] in {g['id'] for g in fallback_groups}) for c in comps if pcls[c['component_id']]['class'] == 'collapsed'},
                             'rule': 'the mosaic of these pieces is too shrunken for the detector; their trusted track groups (tier confident/likely) are counted instead, without position merging'},
        'far_pieces': {'groups_not_counted': len(far_groups), 'ids': [g['id'] for g in far_groups]},
        'room_pan': None if rp is None else {**RP.summary(rp), 'pieces': room_pan_pieces,
                                             'groups_in_room_pan_pieces': sum(1 for c in comps if c['component_id'] in room_pan_pieces for m in c.get('markers', []) if m.get('polygon')),
                                             'unregistered_groups_withheld': sorted(g['id'] for g in outside if g['tier'] in ('confident', 'likely') and g['id'] in pan_only),
                                             'rule': f'a piece with more than {RP.PIECE_SHARE:.0%} room-pan frames is not counted (no detector pass, no merge evidence, no '
                                                     'collapsed-piece fallback); a track group outside the pieces whose every sighting is a room-pan frame is not added'},
        'reconciliation': {'track_groups_placed': len(groups), 'track_groups_on_a_spine': sum(len(v) for v in by_spine.values()), 'spines_with_several_groups': len(merges),
                           'track_duplicates_removed': sum(m['duplicates_removed'] for m in merges), 'track_groups_without_spine': len(unsupported),
                           'spines_without_track_group': panorama_spines - len(by_spine)},
        'merge_evidence': merges,
        'spines': spine_rows, 'spines_coordinates': 'members[].poly in the pixel frame of their own piece (panorama.json component image, markers and overlays)',
        'seconds': {'registration': round(t_reg, 1), 'total': round(t_all, 1)},
        'limits': ['Pieces that show the same shelf area but could not be registered to each other still count that area twice (usually only 2-3 piece pairs per run register).',
                   'Spines the detector misses (small, blurred, dark, thin) are not counted; the detector scale is set from the track groups, not from a per-spine check.',
                   'The detection confidence (0.4), the collapsed (< 0.25) and far (< 0.5) thresholds were chosen after seeing the live Yugo runs; `count_if_confidence` shows the count at other confidences.',
                   'Collapsed pieces fall back to the track groups and carry the track count\'s duplicates for those pieces.',
                   'No per-book truth exists: the hand count of 458 is a total, so offsetting errors are invisible.'],
    }
    return result


def write_count(result: dict[str, Any], pan_dir: str | Path, run_dir: str | Path | None = None) -> list[Path]:
    """panorama_count.json beside panorama.json (versioned dir) and a copy in the run dir (what the packet reads). Atomic; nothing else is touched."""
    out = []
    for d in [Path(pan_dir)] + ([Path(run_dir)] if run_dir else []):
        tmp = d / 'panorama_count.json.tmp'
        tmp.write_text(json.dumps(result, indent=1))
        tmp.replace(d / 'panorama_count.json')
        out.append(d / 'panorama_count.json')
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--detect-worker', nargs=2, metavar=('JOB', 'OUT'))
    ap.add_argument('--run', type=Path)
    ap.add_argument('--panorama', type=Path)
    ap.add_argument('--no-detect', action='store_true', help='skip the detector (registration and track-group reconciliation only)')
    ap.add_argument('--conf', type=float, default=DETECT_CONF)
    ap.add_argument('--no-room-pan', action='store_true', help='count room-pan pieces too (pipeline.room_pan; comparison only)')
    ap.add_argument('--attach', action='store_true', help='after writing, attach the count to the run\'s claim_packet.json (pipeline.panorama_attach)')
    ap.add_argument('--out', type=Path, help='write the result here instead of beside the panorama / in the run dir')
    a = ap.parse_args(argv)
    if a.detect_worker:
        detect_worker(*a.detect_worker)
        return 0
    if not a.run:
        ap.error('--run required')
    res = build_count(a.run, a.panorama, detect=not a.no_detect, conf=a.conf, room_pan=not a.no_room_pan)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    else:
        write_count(res, a.panorama or latest_panorama(a.run), a.run)
    if a.attach and not a.out:
        from pipeline import panorama_attach as PA
        res['attach'] = PA.attach(a.run)
    brief = {k: res[k] for k in ('run_id', 'panorama_run', 'panorama_count', 'range_panorama', 'count_if_confidence', 'usable', 'unusable_reason', 'detector_support', 'seconds')}
    brief['attach'] = res.get('attach')
    brief['registration'] = res['registration']
    brief['room_pan'] = {k: v for k, v in (res.get('room_pan') or {}).items() if k in ('room_pan_frame_count', 'pieces', 'groups_in_room_pan_pieces', 'unregistered_groups_withheld', 'uses_3d')}
    brief['reconciliation'] = res['reconciliation']
    brief['unregistered_groups'] = {k: v for k, v in res['unregistered_groups'].items() if k != 'added_ids'}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
