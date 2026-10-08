"""Sparse structure-from-motion of a saved run (pycolmap, BSD-3) + exactly one 3D marker per proposed book group.

Input: a saved run (manifest.json, frames/, optional inventory.json + candidates.json). Output: a NEW versioned directory
`runs/<run>_recon3d_vN` (never overwritten) holding recon3d.json (+ progress.json while running).

Frame of the output: SfM has an arbitrary pose, so the dominant plane of the cloud (the shelf face) is fitted by RANSAC and
everything (points, cameras, markers) is rotated into a shelf-aligned frame: that plane is z = 0, the camera side is +z,
spine "up" is +y (median long axis of the book polygons on the plane; camera up as a fallback), x = y cross z. Everything
stays UNSCALED: arbitrary units, no metric claim, no room or shelf dimensions.

Markers: one per BOOK group of inventory.json (counts.book_count = len(books)). Sightings are the group's observation_ids joined
to candidates.json. A marker is ray-triangulated when robust (>= 3 registered sightings, parallax >= 3 deg, point near the
plane), otherwise the sighting rays are cast onto the shelf plane ('plane_projected'). Groups with no registered sighting are
listed in `unplaced_groups`, never dropped: len(markers) + len(unplaced_groups) == book_count. No detector/OCR/hosted call. CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

SCHEMA = 'recon3d_v2'
MAX_POINTS = 200_000
MAX_SIGHTINGS = 12
LIMITS = [
    'UNSCALED: arbitrary scale and units; no metric distance, shelf size or room dimension is implied (orientation is aligned to the fitted shelf plane only)',
    'Sparse feature cloud from one handheld video; planar, repetitive or blurred stretches may not register',
    'Markers locate the proposed (unverified) groups from their detection polygons (triangulated when robust, otherwise projected onto the shelf plane); they locate, they do not count',
    'Only frames registered by SfM contribute; a group with no sighting in a registered frame is listed as unplaced']
OK_RATIO, PARTIAL_RATIO = 0.6, 0.3
FRAME_RULES = ('all', 'diet', 'keyframes')
DEFAULT_RULE = 'all'                 # 'diet' / 'keyframes' stay selectable: they keep registration on sparse captures (YT10) but shatter dense handheld pans (YT11)
IMAGE_SIZE, MAX_FEATURES, QUADRATIC = 1600, 8192, True   # SIFT image side cap, SIFT feature cap, COLMAP's quadratic extra pairs
DEFAULT_PROFILE = 'auto'
DEFAULT_MERGE = True                 # LIBRARY_RECON3D_MERGE=0|1: join the fragments of a split build (pipeline/recon3d_merge.py)
MERGE_MODES = ('auto', 'inline', 'deferred')   # LIBRARY_RECON3D_MERGE_MODE: where the costly merge steps run
DEFAULT_MERGE_MODE = 'deferred'       # gate: 'deferred' (first map published without the costly merge steps, then the merge pass --merge-from / post-sweep `merge3d` writes the joined map as the next version) passed on YT11/YT12/YT13/YT7 but lost ~9 points on YT5, so inline (one build) stays the default. auto = a step runs here only when predicted to end within MERGE_BUDGET_S
MERGE_BUDGET_S = 270.0               # LIBRARY_RECON3D_MERGE_BUDGET_S: the first map the user sees should be ready within 5 min (30 s margin for markers and a busy machine)
MERGE_PASS_RULE = 'full'             # bridge pairs + re-map rule of the merge pass (off the critical path: the first map is already published). 'full' = the full rule; 'cheap' (CHEAP_RULE) lost 9 points on YT5 in the gate
MERGE_CACHE = '_merge_cache'        # database + symlinks + first-pass models kept next to a map whose merge was deferred; the merge pass deletes it
AUTO_FAST_MIN_FRAMES = 100           # 'auto' = fast from this many frames on: below it the default build is already ~1 min and 'fast' split the 60-frame IESTest3 replay 2 times in 3
PROFILES = {                                              # LIBRARY_RECON3D_PROFILE: per-frame cost settings; explicit args / env vars still win
    'default': {},
    'fast': {'max_image_size': 1024, 'max_features': 2048, 'quadratic': True,
             'mapper_options': {'ba_global_frames_ratio': 1.5, 'ba_global_points_ratio': 1.5}},
}
BASE_OVERLAP, MIN_OVERLAP = 12, 6    # sequential matching window of the full build (12 frames at 5 fps = 2.4 s); the floor of a diet window
PER_INTERVAL = 2                     # diet: the keyframe plus its sharpest non-gated neighbour
MIN_KEPT = 24                        # a diet that keeps fewer frames than this (mostly blur-gated run) falls back to every frame
MIN_PLACED_FRACTION = 0.5            # below this share of groups placed the run is only 'partial'
TRI_MIN_VIEWS, TRI_MIN_ANGLE_DEG, TRI_TOL_DEG = 3, 3.0, 1.5
PLANE_NEAR_EXTENT = 0.04             # triangulated point must be within this fraction of the scene extent of the plane (or 3 rms)
MIN_PLANE_COS = 0.1                  # ray-plane hit rejected when the ray grazes the plane (cos to the normal below this)
REGION_MARGIN = 0.5                  # plane hits may fall this fraction of the cloud's in-plane span beyond the cloud
# Shelf-plane check: every shelf of these sweeps runs along ONE straight wall, so in a correct map every book lies
# on ONE plane. A book is 'on the plane' when its ray-triangulated position (no plane gate) is within PLANE_TOL_SPINES spine heights of it.
PLANE_TOL_SPINES = 0.5               # tolerance in spine heights (median long edge of the book polygons cast onto the plane)
PLANE_CONSISTENCY_OK = 0.75          # a map is never 'ok' when fewer of its triangulable books are on the plane (calibrated on development sweeps)
CAMERAS_ROOM_SIDE_OK = 0.98         # nor when fewer than this share of its cameras stand on the room side of the shelf plane (not inside the wall)
MIN_CONSISTENCY_BOOKS = 10           # fewer triangulable books: the share is reported but does not gate the status
SPINE_MIN_ASPECT = 1.4               # polygon long/short edge on the plane below this: no spine axis (same as pipeline.auto_scale)


def next_output(runs_dir, run_id):
    """First unused runs/<run>_recon3d_vN."""
    n = 1
    while (Path(runs_dir) / f'{run_id}_recon3d_v{n}').exists():
        n += 1
    return f'{run_id}_recon3d_v{n}'


def subsample(count, limit=MAX_POINTS, seed=0):
    """Indices (sorted) of at most `limit` of `count` points; deterministic."""
    if count <= limit:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, limit, replace=False))


# ---------------------------------------------------------------- rays and triangulation

def _solve(o, d):
    eye = np.eye(3)
    a = np.zeros((3, 3)); b = np.zeros(3)
    for oi, di in zip(o, d):
        m = eye - np.outer(di, di)
        a += m; b += m @ oi
    if abs(np.linalg.det(a)) < 1e-12:
        return None
    return np.linalg.solve(a, b)


def _angular_errors(p, o, d):
    """Per-ray angle (deg) between the ray and the direction to p; inf when p is behind the ray origin."""
    v = p - o
    depth = (v * d).sum(axis=1)
    perp = np.linalg.norm(v - depth[:, None] * d, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        err = np.degrees(np.arctan2(perp, depth))
    return np.where(depth > 0, err, np.inf), perp


def triangulate_rays(origins, directions, *, min_views=2, min_angle_deg=2.0, tol_deg=2.0, max_rays=24):
    """Robust nearest point to several world rays (pair-RANSAC, then least squares on the consensus set).

    origins/directions: (n,3). A candidate point from each pair of rays (in front of both, parallax >= min_angle_deg)
    is scored by how many rays pass within `tol_deg` of it. Returns None when no consensus of `min_views` rays with
    parallax >= min_angle_deg exists; else dict(xyz, views, residual, angle_deg): residual = median perpendicular
    distance of the consensus rays to xyz (SfM units), angle_deg = largest angle between consensus rays.
    """
    o = np.asarray(origins, float).reshape(-1, 3)
    d = np.asarray(directions, float).reshape(-1, 3)
    norm = np.linalg.norm(d, axis=1)
    keep = norm > 1e-12
    o, d = o[keep], d[keep] / norm[keep, None]
    if len(o) > max_rays:
        idx = np.linspace(0, len(o) - 1, max_rays).round().astype(int)
        o, d = o[idx], d[idx]
    n = len(o)
    if n < min_views:
        return None
    cos_min = np.cos(np.radians(min_angle_deg))
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            if float(d[i] @ d[j]) > cos_min:
                continue
            p = _solve(o[[i, j]], d[[i, j]])
            if p is None:
                continue
            err, _ = _angular_errors(p, o, d)
            inliers = np.flatnonzero(err <= tol_deg)
            if len(inliers) < min_views or not {i, j} <= set(inliers.tolist()):
                continue
            score = (len(inliers), -float(err[inliers].sum()))
            if best is None or score > best[0]:
                best = (score, inliers)
    if best is None:
        return None
    inliers = best[1]
    p = _solve(o[inliers], d[inliers])
    if p is None:
        return None
    err, perp = _angular_errors(p, o[inliers], d[inliers])
    if not np.isfinite(err).all():
        return None
    angle = float(np.degrees(np.arccos(np.clip((d[inliers] @ d[inliers].T).min(), -1, 1))))
    if angle < min_angle_deg:
        return None
    return {'xyz': p, 'views': int(len(inliers)), 'residual': float(np.median(perp)), 'angle_deg': angle}


class View:
    """A registered camera: centre and world-to-camera rotation in the current frame; `unproject(xy)` -> normalised (x, y)."""
    __slots__ = ('frame_id', 'centre', 'rotation', 'unproject')

    def __init__(self, frame_id, centre, rotation, unproject):
        self.frame_id = frame_id
        self.centre = np.asarray(centre, float).reshape(3)
        self.rotation = np.asarray(rotation, float).reshape(3, 3)
        self.unproject = unproject

    def ray(self, xy):
        cam = np.asarray(self.unproject(np.asarray(xy, float)), float).reshape(-1)[:2]
        return self.centre, self.rotation.T @ np.array([cam[0], cam[1], 1.0])

    def up(self):
        """World direction of the image's upward axis (image y points down)."""
        return -self.rotation.T @ np.array([0.0, 1.0, 0.0])

    def transformed(self, rot, origin):
        """The same camera after the rigid change of frame x' = rot (x - origin)."""
        return View(self.frame_id, rot @ (self.centre - origin), self.rotation @ rot.T, self.unproject)


def pinhole(focal, cx, cy):
    return lambda xy: ((xy[0] - cx) / focal, (xy[1] - cy) / focal)


# ---------------------------------------------------------------- shelf plane and aligned frame

def fit_plane(points, *, iterations=300, sample=30000, rel_threshold=0.015, seed=0):
    """Dominant plane of a cloud: 3-point RANSAC (inlier distance = rel_threshold * robust extent), then least squares
    on the inliers. Returns dict(normal, point, inlier_ratio, rms, threshold, extent) or None. `extent` is the diagonal of the
    5-95% percentile box of the cloud; `rms` the RMS distance of the inliers to the plane (SfM units)."""
    pts = np.asarray(points, float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 30:
        return None
    rng = np.random.default_rng(seed)
    sub = pts if len(pts) <= sample else pts[rng.choice(len(pts), sample, replace=False)]
    lo, hi = np.percentile(sub, [5, 95], axis=0)
    extent = float(np.linalg.norm(hi - lo))
    if extent < 1e-9:
        return None
    thr = rel_threshold * extent
    best = None
    for _ in range(iterations):
        i = rng.choice(len(sub), 3, replace=False)
        n = np.cross(sub[i[1]] - sub[i[0]], sub[i[2]] - sub[i[0]])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        count = int((np.abs((sub - sub[i[0]]) @ n) <= thr).sum())
        if best is None or count > best[0]:
            best = (count, n, sub[i[0]])
    if best is None:
        return None
    _, normal, point = best
    for _ in range(6):                                  # refine on all points
        mask = np.abs((pts - point) @ normal) <= thr
        if mask.sum() < 10:
            return None
        point = pts[mask].mean(axis=0)
        normal = np.linalg.svd(pts[mask] - point, full_matrices=False)[2][2]
    dist = (pts - point) @ normal
    mask = np.abs(dist) <= thr
    rms = float(np.sqrt(np.mean(dist[mask] ** 2)))
    return {'normal': normal / np.linalg.norm(normal), 'point': point, 'inlier_ratio': float(mask.mean()), 'rms': rms,
            'threshold': float(thr), 'extent': extent, 'inlier_mask': mask}


def orient_normal(normal, point, centres):
    """Flip the normal so the camera side is positive."""
    centres = np.asarray(centres, float).reshape(-1, 3)
    if len(centres) and float(np.median((centres - point) @ normal)) < 0:
        return -normal
    return normal


def plane_rotation(normal):
    """Rotation whose rows are an in-plane basis (e1, e2) and the normal: maps the normal to +z (in-plane axes arbitrary)."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(helper, n); e1 /= np.linalg.norm(e1)
    return np.array([e1, np.cross(n, e1), n])


def yaw_rotation(up_angle):
    """Rotation about z taking the in-plane direction at angle `up_angle` (from +x) to +y."""
    phi = np.pi / 2 - up_angle
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def hit_plane(origin, direction, min_cos=MIN_PLANE_COS):
    """Where a ray meets z = 0 in front of the camera (camera side z > 0); None when it grazes, is parallel or heads away."""
    norm = np.linalg.norm(direction)
    if norm < 1e-12 or origin[2] <= 0:
        return None
    d = direction / norm
    if d[2] > -min_cos:
        return None
    t = -origin[2] / d[2]
    return origin + t * d


def sighting_pixel(s):
    """Polygon centre (mean of vertices) of a sighting, else its bbox centre."""
    poly = s.get('polygon_xy')
    if poly is not None and len(poly) >= 3:
        arr = np.asarray(poly, float).reshape(-1, 2)
        if np.isfinite(arr).all():
            return arr.mean(axis=0)
    x0, y0, x1, y1 = s['bbox_xyxy']
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2], float)


def _long_axis_ends(poly, width, height, min_aspect=1.5, margin=3.0):
    """Midpoints of the two short edges of an oriented quad (its long axis), or None when it is near-square or clipped by the image."""
    p = np.asarray(poly, float).reshape(-1, 2)
    if len(p) != 4 or not np.isfinite(p).all():
        return None
    if (p[:, 0] < margin).any() or (p[:, 1] < margin).any() or (p[:, 0] > width - margin).any() or (p[:, 1] > height - margin).any():
        return None
    e0, e1 = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])
    if min(e0, e1) < 1e-6 or max(e0, e1) / min(e0, e1) < min_aspect:
        return None
    if e0 >= e1:
        return (p[1] + p[2]) / 2, (p[3] + p[0]) / 2
    return (p[0] + p[1]) / 2, (p[2] + p[3]) / 2


def estimate_up(views, sightings, *, min_samples=15):
    """In-plane 'up' direction angle (rad from +x of the plane frame, plane z = 0) for views already in the plane frame.

    Each book polygon's long axis is cast onto the plane and signed to agree with the camera's up vector; the circular median
    over polygons wins when >= min_samples exist, else the mean camera-up direction. Returns dict(angle, source, samples,
    spread_deg, camera_up_angle, up_vs_camera_up_deg) or None when nothing is usable."""
    cam_up = []
    for v in views.values():
        u = v.up()
        if np.hypot(u[0], u[1]) > 0.05:
            cam_up.append(np.arctan2(u[1], u[0]))
    cam_up_angle = _circ_mean(cam_up) if cam_up else None
    angles = []
    for s in sightings:
        poly = s.get('polygon_xy')
        v = views.get(s['frame_id'])
        if poly is None or v is None or s.get('width_px') is None:
            continue
        ends = _long_axis_ends(poly, s['width_px'], s['height_px'])
        if ends is None:
            continue
        hits = [hit_plane(*v.ray(e)) for e in ends]
        if hits[0] is None or hits[1] is None:
            continue
        a = hits[1][:2] - hits[0][:2]
        if np.linalg.norm(a) < 1e-12:
            continue
        u = v.up()
        if np.hypot(u[0], u[1]) > 0.05 and float(a @ u[:2]) < 0:
            a = -a
        angles.append(np.arctan2(a[1], a[0]))
    if len(angles) >= min_samples:
        mean = _circ_mean(angles)
        dev = np.angle(np.exp(1j * (np.asarray(angles) - mean)))
        med = float(np.median(dev))
        spread = float(np.degrees(np.median(np.abs(dev - med))))
        angle = float(mean + med)
        out = {'angle': angle, 'source': 'book_polygons', 'samples': len(angles), 'spread_deg': round(spread, 1)}
    elif cam_up_angle is not None:
        out = {'angle': float(cam_up_angle), 'source': 'camera_up', 'samples': len(cam_up), 'spread_deg': None}
    else:
        return None
    out['camera_up_angle'] = cam_up_angle
    out['up_vs_camera_up_deg'] = None if cam_up_angle is None else round(float(abs(np.degrees(np.angle(np.exp(1j * (out['angle'] - cam_up_angle)))))), 1)
    return out


def _circ_mean(angles):
    a = np.asarray(angles, float)
    return float(np.arctan2(np.sin(a).sum(), np.cos(a).sum()))


# ---------------------------------------------------------------- groups and marker placement

def load_groups(source):
    """Book and item groups of a run with their sightings (observation_ids joined to candidates.json).

    Returns (groups, book_count). Each group: group_id, kind 'book'|'item', category, sightings [{observation_id, frame_id,
    bbox_xyxy, polygon_xy, sharpness, confidence, width_px, height_px}]. Sightings are the group's observation_ids; the
    inventory's evidence_refs are only a fallback for an observation missing from candidates.json."""
    source = Path(source)
    path = source / 'inventory.json'
    if not path.is_file():
        return [], 0
    data = json.loads(path.read_text()).get('data', {})
    obs = {}
    cand = source / 'candidates.json'
    if cand.is_file():
        obs = {o['observation_id']: o for o in json.loads(cand.read_text()).get('data', {}).get('observations', [])}
    frame_info = {}
    manifest = source / 'manifest.json'
    if manifest.is_file():
        for f in json.loads(manifest.read_text()).get('data', {}).get('frames', []):
            frame_info[f['frame_id']] = {'sharpness': float((f.get('quality') or {}).get('sharpness') or 0.0),
                                         'width_px': f.get('width_px'), 'height_px': f.get('height_px')}
    groups = []
    for kind, key in (('book', 'books'), ('item', 'items')):
        for g in data.get(key, []):
            evidence = {e.get('observation_id'): e for e in g.get('evidence_refs', [])}
            sightings, seen = [], set()
            for oid in list(g.get('observation_ids') or []) + [k for k in evidence if k]:
                if oid in seen:
                    continue
                seen.add(oid)
                o, e = obs.get(oid), evidence.get(oid)
                frame_id = (o or e or {}).get('frame_id')
                bbox = (o or e or {}).get('bbox_xyxy')
                if frame_id is None or not bbox:
                    continue
                info = frame_info.get(frame_id, {})
                sightings.append({'observation_id': oid, 'frame_id': frame_id, 'bbox_xyxy': [float(v) for v in bbox],
                                  'polygon_xy': (o or {}).get('polygon_xy'), 'sharpness': info.get('sharpness', 0.0),
                                  'confidence': float((o or {}).get('confidence') or 0.0),
                                  'width_px': info.get('width_px'), 'height_px': info.get('height_px')})
            groups.append({'group_id': g['instance_id'], 'kind': kind, 'category': g.get('category') or g.get('kind'),
                           'sightings': sightings})
    book_count = len([g for g in groups if g['kind'] == 'book'])
    return groups, book_count


def plane_region(points_xy, margin=REGION_MARGIN):
    """In-plane box [(xmin, ymin), (xmax, ymax)] of the cloud (1-99%) widened by `margin` of its span."""
    lo, hi = np.percentile(np.asarray(points_xy, float).reshape(-1, 2), [1, 99], axis=0)
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


def _sighting_view(s):
    return {'frame_id': s['frame_id'], 'observation_id': s['observation_id'], 'bbox_xyxy': [round(v, 1) for v in s['bbox_xyxy']],
            **({'polygon_xy': [[round(float(x), 1), round(float(y), 1)] for x, y in s['polygon_xy']]} if s.get('polygon_xy') else {})}


def place_group(group, views, *, plane_tol, region=None):
    """(marker, None) or (None, {group_id, code, reason}) for one group; views are in the shelf-aligned frame (plane z = 0)."""
    allseen = group['sightings']
    gid = group['group_id']
    if not allseen:
        return None, {'group_id': gid, 'code': 'no_sightings', 'reason': 'the group has no sighting with a frame and box in candidates.json'}
    regs = [s for s in allseen if s['frame_id'] in views]
    if not regs:
        return None, {'group_id': gid, 'code': 'no_registered_sighting',
                      'reason': f'none of its {len(allseen)} sighting(s) is in a frame registered by SfM'}
    regs.sort(key=lambda s: (-s['sharpness'], -s['confidence'], s['observation_id']))     # best (sharpest frame) first
    rays = [views[s['frame_id']].ray(sighting_pixel(s)) for s in regs]
    base = {'group_id': gid, 'kind': group['kind'], 'category': group['category'], 'registered_sightings': len(regs),
            'sightings_total': len(allseen), 'sightings': [_sighting_view(s) for s in regs[:MAX_SIGHTINGS]]}

    def in_region(xy):
        return region is None or bool((xy >= region[0]).all() and (xy <= region[1]).all())

    if len(regs) >= TRI_MIN_VIEWS:
        fit = triangulate_rays([r[0] for r in rays], [r[1] for r in rays], min_views=TRI_MIN_VIEWS,
                               min_angle_deg=TRI_MIN_ANGLE_DEG, tol_deg=TRI_TOL_DEG)
        if fit is not None and abs(fit['xyz'][2]) <= plane_tol and in_region(fit['xyz'][:2]):
            return {**base, 'xyz': [round(float(v), 5) for v in fit['xyz']], 'method': 'triangulated', 'views': fit['views'],
                    'ray_residual': round(fit['residual'], 5), 'baseline_angle_deg': round(fit['angle_deg'], 2)}, None
    hits = [(s, h) for s, (o, d) in zip(regs, rays) for h in [hit_plane(o, d)] if h is not None]
    if not hits:
        return None, {'group_id': gid, 'code': 'no_plane_hit', 'reason': 'no sighting ray reaches the shelf plane in front of its camera'}
    xy = np.median(np.array([h[:2] for _, h in hits]), axis=0)      # median over sightings: robust to one bad box
    if not in_region(xy):
        return None, {'group_id': gid, 'code': 'off_shelf', 'reason': 'its rays meet the shelf plane far outside the reconstructed shelf region'}
    return {**base, 'xyz': [round(float(xy[0]), 5), round(float(xy[1]), 5), 0.0], 'method': 'plane_projected', 'views': len(hits),
            'ray_residual': None, 'baseline_angle_deg': None}, None


def place_markers(groups, views, *, plane_tol, region=None):
    """Place every group: returns (markers, unplaced). len(markers) + len(unplaced) == len(groups), order preserved."""
    markers, unplaced = [], []
    for g in groups:
        m, u = place_group(g, views, plane_tol=plane_tol, region=region)
        (markers if m else unplaced).append(m or u)
    return markers, unplaced


def hit_plane_at(origin, direction, normal, point, min_cos=MIN_PLANE_COS):
    """hit_plane for any plane (unit `normal` pointing to the camera side, through `point`): where the ray meets it in front of the
    camera, None when the camera is behind the plane or the ray grazes / heads away."""
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return None
    d = np.asarray(direction, float) / norm
    h = float((np.asarray(origin, float) - point) @ normal)
    c = float(d @ normal)
    if h <= 0 or c > -min_cos:
        return None
    return origin + (-h / c) * d


def spine_lengths(views, sightings, normal, point, *, margin=3.0, min_aspect=SPINE_MIN_ASPECT, limit=4000):
    """Long edge of every unclipped oriented book polygon cast onto a plane (units of the views' frame); the median is one spine height."""
    out = []
    for s in sightings:
        v, poly = views.get(s['frame_id']), s.get('polygon_xy')
        if v is None or poly is None or s.get('width_px') is None:
            continue
        p = np.asarray(poly, float).reshape(-1, 2)
        if len(p) != 4 or not np.isfinite(p).all() or (p < margin).any() or (p[:, 0] > s['width_px'] - margin).any() \
                or (p[:, 1] > s['height_px'] - margin).any():
            continue
        hits = [hit_plane_at(*v.ray(q), normal, point) for q in p]
        if any(h is None for h in hits):
            continue
        e = [float(np.linalg.norm(hits[(i + 1) % 4] - hits[i])) for i in range(4)]
        a, b = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
        if min(a, b) > 1e-12 and max(a, b) / min(a, b) >= min_aspect:
            out.append(max(a, b))
        if len(out) >= limit:
            break
    return out


def book_positions(books, views, normal=None, point=None):
    """Per book group with a sighting in `views`: its ray-triangulated position WITHOUT any plane gate ('triangulated': >= TRI_MIN_VIEWS rays
    with parallax), else, when a plane is given, the median of its rays' hits on that plane ('plane_projected'). [(group_id, xyz, method)]."""
    out = []
    for g in books:
        regs = sorted((s for s in g['sightings'] if s['frame_id'] in views), key=lambda s: (-s['sharpness'], -s['confidence'], s['observation_id']))
        if not regs:
            continue
        rays = [views[s['frame_id']].ray(sighting_pixel(s)) for s in regs]
        if len(rays) >= TRI_MIN_VIEWS:
            fit = triangulate_rays([r[0] for r in rays], [r[1] for r in rays], min_views=TRI_MIN_VIEWS, min_angle_deg=TRI_MIN_ANGLE_DEG, tol_deg=TRI_TOL_DEG)
            if fit is not None:
                out.append((g['group_id'], fit['xyz'], 'triangulated'))
                continue
        if normal is not None:
            hits = [h for h in (hit_plane_at(o, d, normal, point) for o, d in rays) if h is not None]
            if hits:
                out.append((g['group_id'], np.median(np.array(hits), axis=0), 'plane_projected'))
    return out


def plane_consistency(books, views, normal, point, *, tol_spines=PLANE_TOL_SPINES, min_books=MIN_CONSISTENCY_BOOKS, min_share=PLANE_CONSISTENCY_OK,
                      min_room_side=CAMERAS_ROOM_SIDE_OK):
    """Is the map ONE flat shelf face? Share of the triangulable books (book_positions, no plane gate) within tol_spines spine heights of the
    plane, and share of the cameras on its room side (no camera can stand behind the wall). The recon3d.json `plane_consistency` block;
    `consistent` is False when >= min_books books are evaluated and the share is below min_share, or the room-side share is below min_room_side."""
    sightings = [s for g in books for s in g['sightings']]
    lengths = spine_lengths(views, sightings, normal, point)
    tri = np.array([xyz for _, xyz, m in book_positions(books, views) if m == 'triangulated'], float).reshape(-1, 3)
    side = np.array([float((v.centre - point) @ normal) for v in views.values()])
    room = round(float((side > 0).mean()), 4) if len(side) else None
    out = {'rule': f'share of ray-triangulated book positions (no plane gate) within {tol_spines} spine heights of the shelf plane, and share of '
                   f"cameras on the room side of it; a map is not 'ok' when the first is below {min_share} (with >= {min_books} books evaluated) "
                   f'or the second below {min_room_side}',
           'tolerance_spines': tol_spines, 'min_share_for_ok': min_share, 'min_books': min_books, 'min_cameras_room_side_for_ok': min_room_side,
           'books_evaluated': int(len(tri)), 'spine_samples': len(lengths), 'spine_height': None, 'tolerance': None, 'share': None,
           'median_abs_offset_spines': None, 'evaluated': False, 'cameras_room_side': room}
    if lengths and len(tri):
        spine = float(np.median(lengths))
        d = np.abs((tri - point) @ normal)
        out.update(spine_height=round(spine, 5), tolerance=round(tol_spines * spine, 5), share=round(float((d <= tol_spines * spine).mean()), 4),
                   median_abs_offset_spines=round(float(np.median(d)) / spine, 3), p90_abs_offset_spines=round(float(np.percentile(d, 90)) / spine, 3),
                   evaluated=len(tri) >= min_books)
    else:
        out['note'] = 'no spine polygon on the plane' if not lengths else 'no triangulable book'
    out['consistent'] = not ((out['evaluated'] and out['share'] < min_share) or (room is not None and room < min_room_side))
    return out


def map_consistency(d):
    """'consistent' / 'inconsistent' (recon3d.json plane_consistency.consistent) or 'unchecked' (a map built before the shelf-plane check has no such block).
    Map pickers skip 'inconsistent' maps and prefer 'consistent' over 'unchecked' ones (old joined maps may be wrongly joined)."""
    pc = (d or {}).get('plane_consistency')
    if not isinstance(pc, dict) or 'consistent' not in pc:
        return 'unchecked'
    return 'consistent' if pc['consistent'] else 'inconsistent'


_CONSISTENCY_CACHE = {}


def map_consistency_at(path):
    """map_consistency of a recon3d.json file (or its dir), cached by modification time; 'unchecked' when unreadable."""
    f = Path(path)
    f = f / 'recon3d.json' if f.is_dir() else f
    try:
        key = (str(f), f.stat().st_mtime_ns)
    except OSError:
        return 'unchecked'
    if key not in _CONSISTENCY_CACHE:
        try:
            _CONSISTENCY_CACHE[key] = map_consistency(json.loads(f.read_text()))
        except (OSError, ValueError):
            _CONSISTENCY_CACHE[key] = 'unchecked'
    return _CONSISTENCY_CACHE[key]


def status_for(registered_ratio, markers, expects_markers=True, placed_fraction=1.0, plane_ok=True, plane_consistent=True):
    if registered_ratio < PARTIAL_RATIO:
        return 'failed'
    if registered_ratio < OK_RATIO or not plane_ok or not plane_consistent:
        return 'partial'
    if expects_markers and (not markers or placed_fraction < MIN_PLACED_FRACTION):
        return 'partial'
    return 'ok'


def _gated(f):
    return bool((f.get('quality') or {}).get('blur_gated'))


def _sharp(f):
    return float((f.get('quality') or {}).get('sharpness') or 0.0)


def _thin(frames, keep):
    """The `keep` sharpest of `frames`, back in time order (ties: earlier frame first, so the result is deterministic)."""
    ranked = sorted(range(len(frames)), key=lambda i: (-_sharp(frames[i]), i))[:keep]
    return [frames[i] for i in sorted(ranked)]


def select_frames(frames, rule=DEFAULT_RULE, max_frames=None):
    """Frames fed to SfM, in time order, and how they were chosen: (kept_frames, info).

    rule 'all': every frame (the pre-diet behaviour). 'diet': per pan interval of the manifest the keyframe (`selected`) and its
    sharpest other non-gated frame, never a blur-gated frame (<= 2 per interval; 5 fps frames overlap > 90%, so the rest only cost
    matching time). 'keyframes': the non-gated keyframes alone (the lower bound). A run whose manifest has no keyframe metadata
    (no `selected` flag on any frame) falls back to 'all'. `max_frames` thins whatever remains by sharpness, keeping time order.
    The info dict is stored in recon3d.json (`diet`)."""
    if rule not in FRAME_RULES:
        raise ValueError(f'unknown frame rule {rule!r}; choose one of {", ".join(FRAME_RULES)}')
    total = len(frames)
    info = {'rule': rule, 'frames_total': total, 'keyframes': sum(1 for f in frames if f.get('selected')),
            'blur_gated': sum(_gated(f) for f in frames)}
    used = rule
    if rule != 'all' and not info['keyframes']:
        used = 'all'
        info['fallback'] = 'no keyframe metadata in the manifest: all frames used'
    if used == 'all':
        kept = list(frames)
    else:
        groups = {}
        for i, f in enumerate(frames):
            groups.setdefault(f.get('interval_index'), []).append(i)
        take = set()
        info['gated_keyframes_dropped'] = 0
        info['neighbours_added'] = 0
        for key, members in groups.items():
            if key is None:                                   # outside any interval: only a keyframe can stand for it
                take.update(i for i in members if frames[i].get('selected') and not _gated(frames[i]))
                continue
            keys = [i for i in members if frames[i].get('selected')]
            clean = sorted((i for i in members if not _gated(frames[i])), key=lambda i: (-_sharp(frames[i]), i))
            info['gated_keyframes_dropped'] += sum(_gated(frames[i]) for i in keys)
            chosen = [i for i in keys if not _gated(frames[i])]
            if used == 'diet':
                chosen += [i for i in clean if i not in chosen][:max(0, PER_INTERVAL - len(chosen))]
                info['neighbours_added'] += max(0, len(chosen) - len([i for i in keys if not _gated(frames[i])]))
            take.update(chosen)
        kept = [frames[i] for i in sorted(take)]
        if len(kept) < min(MIN_KEPT, total):
            info['fallback'] = f'the {rule} rule kept only {len(kept)} frame(s) (blur-gated run): all frames used'
            used, kept = 'all', list(frames)
    if max_frames and len(kept) > max_frames:
        kept = _thin(kept, max_frames)
        info['capped_to'] = int(max_frames)
    info['frames_used'] = len(kept)
    info['applied'] = used
    return kept, info


def sequential_overlap(kept, total, base=BASE_OVERLAP, floor=MIN_OVERLAP):
    """Sequential-matching window in frames. The full build matches `base` frames ahead (2.4 s at 5 fps); with fewer, further apart
    frames the window must span the same time: base * kept / total, never below `floor` (and never above `base`)."""
    if total <= 0 or kept >= total:
        return base
    return max(min(floor, base), min(base, round(base * kept / total)))


def _env_int(name):
    try:
        v = int(os.environ.get(name, ''))
    except ValueError:
        return None
    return v if v > 0 else None


class Progress:
    def __init__(self, out):
        self.path = Path(out) / 'progress.json'
        self.started = time.time()

    def __call__(self, stage, **extra):
        data = {'stage': stage, 'elapsed_s': round(time.time() - self.started, 1), **extra}
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)


def _frame_table(source):
    manifest = json.loads((source / 'manifest.json').read_text())
    frames = manifest['data']['frames']
    if not frames:
        raise ValueError('Run has no frames')
    return frames


def _registered_images(rec):
    """frame name -> image, registered only (pycolmap 4.x exposes registered via the frame/image)."""
    out = {}
    for image in rec.images.values():
        try:
            registered = image.has_pose
        except AttributeError:
            registered = True
        if registered:
            out[image.name] = image
    return out


def merge_enabled():
    v = os.environ.get('LIBRARY_RECON3D_MERGE', '').strip().lower()
    return DEFAULT_MERGE if v == '' else v not in ('0', 'false', 'no', 'off')


def merge_mode(mode=None):
    m = (mode or os.environ.get('LIBRARY_RECON3D_MERGE_MODE') or DEFAULT_MERGE_MODE).strip().lower()
    if m not in MERGE_MODES:
        raise ValueError(f'unknown merge mode {m!r}; choose one of {", ".join(MERGE_MODES)}')
    return m


def merge_budget():
    try:
        return float(os.environ.get('LIBRARY_RECON3D_MERGE_BUDGET_S', '')) or MERGE_BUDGET_S
    except ValueError:
        return MERGE_BUDGET_S


def _load_models(sparse):
    """{model id: Reconstruction} of the models incremental_mapping wrote under `sparse` (sub-dirs 0, 1, ...)."""
    import pycolmap
    out = {}
    for d in sorted(Path(sparse).iterdir()) if Path(sparse).is_dir() else []:
        if d.is_dir() and d.name.isdigit() and (d / 'images.bin').exists():
            out[int(d.name)] = pycolmap.Reconstruction(str(d))
    return out


def pending_merge(source_run, runs_dir=None):
    """Newest `<run>_recon3d_vN` of a source run whose merge was deferred and whose cache is still there, else None."""
    source = Path(source_run)
    runs = Path(runs_dir) if runs_dir else source.parent
    found = []
    for p in runs.glob(f'{source.name}_recon3d_v*'):
        try:
            n = int(p.name.rsplit('_v', 1)[1])
            merge = json.loads((p / 'recon3d.json').read_text()).get('merge') or {}
        except (ValueError, OSError):
            continue
        if merge.get('deferred') and (p / MERGE_CACHE / 'merge_state.json').is_file():
            found.append((n, p))
    return max(found)[1] if found else None


def model_inputs(members, names_to_frame, max_error=2.0):
    """Cloud + cameras of one model or of several joined ones: members = [(Reconstruction, (s, R, t) into the main frame or None[, keep])];
    keep (shelf-plane check) = the image names used of that model (a re-mapped model trimmed to its parts on the shelf plane): only those cameras and the
    points they observe; None or absent = the whole model.

    Points: seen in >= 3 views with reprojection error <= max_error (the rest are mostly depth outliers), mapped into the main frame.
    A frame registered in more than one member keeps the pose of the first (the main model wins). Returns (xyz, colour, views,
    intrinsics, mean reprojection error over the members (weighted by points), points total, points kept by the filter)."""
    views, intrinsics, xyz, colour = {}, {}, [], []
    n_all = n_filtered = 0
    err_sum = err_n = 0.0
    for member in members:
        rec, sim, keep = member[0], member[1], (member[2] if len(member) > 2 else None)
        s, rot, t = sim if sim is not None else (1.0, np.eye(3), np.zeros(3))
        all_points = list(rec.points3D.values())
        if keep is not None:
            seen = {p.point3D_id for im in _registered_images(rec).values() if im.name in keep for p in im.points2D if p.has_point3D()}
            all_points = [p for pid, p in rec.points3D.items() if pid in seen]
        pts = [p for p in all_points if p.track.length() >= 3 and p.error <= max_error]
        n_all += len(all_points)
        n_filtered += len(pts)
        if pts:
            p = np.array([q.xyz for q in pts], float).reshape(-1, 3)
            xyz.append(s * p @ rot.T + t)
            colour.append(np.array([q.color for q in pts], int).reshape(-1, 3))
            err_sum += float(rec.compute_mean_reprojection_error()) * len(pts)
            err_n += len(pts)
        for image in _registered_images(rec).values():
            frame_id = names_to_frame.get(image.name)
            if frame_id is None or frame_id in views or (keep is not None and image.name not in keep):
                continue
            camera = rec.cameras[image.camera_id]
            v = _view_of(image, camera, frame_id)
            views[frame_id] = View(frame_id, s * rot @ v.centre + t, v.rotation @ rot.T, v.unproject)
            intrinsics[frame_id] = (float(camera.focal_length), int(camera.width), int(camera.height))
    xyz = np.concatenate(xyz) if xyz else np.zeros((0, 3))
    colour = np.concatenate(colour) if colour else np.zeros((0, 3), int)
    keep = subsample(len(xyz))
    return xyz[keep], colour[keep], views, intrinsics, (err_sum / err_n if err_n else None), n_all, n_filtered


def _view_of(image, camera, frame_id):
    pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
    rotation = np.asarray(pose.rotation.matrix(), float)
    centre = -rotation.T @ np.asarray(pose.translation, float)
    return View(frame_id, centre, rotation, lambda xy, c=camera: np.asarray(c.cam_from_img(np.asarray(xy, float)), float).reshape(-1)[:2])


def align_scene(xyz, views, sightings):
    """Fit the shelf plane and return (rot, origin, plane, alignment): x' = rot (x - origin) is the shelf-aligned frame.

    plane is None when no plane could be fitted (then rot = I, origin = 0, alignment.applied = False)."""
    plane = fit_plane(xyz)
    if plane is None:
        return np.eye(3), np.zeros(3), None, {'applied': False, 'reason': 'no dominant plane found in the cloud'}
    normal = orient_normal(plane['normal'], plane['point'], [v.centre for v in views.values()])
    plane['normal'] = normal
    rot1 = plane_rotation(normal)
    in_plane = {k: v.transformed(rot1, plane['point']) for k, v in views.items()}
    up = estimate_up(in_plane, sightings)
    yaw = yaw_rotation(up['angle']) if up else np.eye(3)
    rot = yaw @ rot1
    alignment = {'applied': True, 'frame': 'shelf_aligned', 'unscaled': True,
                 'description': 'z = 0 is the fitted shelf plane; +z points to the camera side; +y is spine-up '
                                + ('(median long axis of the detected book polygons on the plane)' if up and up['source'] == 'book_polygons'
                                   else '(mean camera up vector projected on the plane)' if up else '(no up estimate; in-plane axes arbitrary)')
                                + '; x = y cross z. Arbitrary units; no metric claim.',
                 'up_source': up['source'] if up else None, 'up_samples': up['samples'] if up else 0,
                 'up_spread_deg': up['spread_deg'] if up else None, 'up_vs_camera_up_deg': up['up_vs_camera_up_deg'] if up else None,
                 'rotation_aligned_from_sfm': [[round(float(v), 6) for v in row] for row in rot],
                 'origin_sfm': [round(float(v), 5) for v in plane['point']]}
    return rot, plane['point'], plane, alignment


def scene_result(xyz, colour, views, intrinsics, books, items, *, frame_ids, mean_error=None, points_total=None, points_filtered=None):
    """Everything that depends on the SfM model, as the recon3d.json fields: shelf alignment, points, cameras, one marker per book
    group (or an unplaced entry), registration numbers, status and reason. Pure numpy: views {frame_id: View} and intrinsics
    {frame_id: (focal_px, width_px, height_px)} are in the SfM frame; xyz (n,3) / colour (n,3) are the kept cloud."""
    book_count = len(books)
    all_sightings = [s for g in books for s in g['sightings']]
    rot, origin, plane, alignment = align_scene(xyz, views, all_sightings)
    xyz_a = (xyz - origin) @ rot.T if len(xyz) else xyz
    views_a = {k: v.transformed(rot, origin) for k, v in views.items()}
    plane_ok = plane is not None and plane['inlier_ratio'] >= 0.2
    scene_extent = float(np.linalg.norm(np.percentile(xyz_a, 95, axis=0) - np.percentile(xyz_a, 5, axis=0))) if len(xyz_a) else 0.0
    if plane is not None:
        region = plane_region(xyz_a[plane['inlier_mask']][:, :2])
        plane_tol = max(3 * plane['rms'], PLANE_NEAR_EXTENT * scene_extent)
    else:
        region, plane_tol = None, 0.0
    books_markers, books_unplaced = place_markers(books, views_a, plane_tol=plane_tol, region=region)
    item_markers, item_unplaced = place_markers(items, views_a, plane_tol=plane_tol, region=region)
    if plane is None:
        for u in books_unplaced + item_unplaced:
            u['reason'] += '; no shelf plane could be fitted for a plane projection'
    assert len(books_markers) + len(books_unplaced) == book_count, 'marker invariant broken'
    cameras = []
    for frame_id, v in sorted(views_a.items()):
        focal, width, height = intrinsics[frame_id]
        cameras.append({'frame_id': frame_id, 'center': [round(float(x), 5) for x in v.centre],
                        'rotation_world_to_cam': [[round(float(x), 6) for x in row] for row in v.rotation],
                        'focal_px': float(focal), 'width_px': int(width), 'height_px': int(height)})
    ratio = len(views) / len(frame_ids)
    placed_fraction = len(books_markers) / book_count if book_count else 1.0
    consistency = plane_consistency(books, views_a, np.array([0.0, 0.0, 1.0]), np.zeros(3)) if plane is not None else None
    consistent = consistency is None or consistency['consistent']
    status = status_for(ratio, books_markers, expects_markers=book_count > 0, placed_fraction=placed_fraction, plane_ok=plane_ok,
                        plane_consistent=consistent)
    reasons = []
    if status == 'failed':
        reasons.append(f"Only {len(views)}/{len(frame_ids)} frames registered ({ratio:.0%}); the shelf is not reconstructed")
    elif ratio < OK_RATIO:
        reasons.append(f"Only {ratio:.0%} of frames registered; the cloud covers part of the shelf")
    if status != 'failed' and not plane_ok:
        reasons.append('No dominant shelf plane in the cloud; the view is not shelf-aligned and plane projection is unavailable')
    if status != 'failed' and not consistent:
        c = consistency
        reasons.append('The map is not one flat shelf face (wrongly joined fragments?): '
                       + (f"{c['share']:.0%} of {c['books_evaluated']} triangulated books lie within {PLANE_TOL_SPINES} spine heights of the shelf plane "
                          f"(min {PLANE_CONSISTENCY_OK:.0%}); " if c['share'] is not None else '')
                       + f"{c['cameras_room_side']:.0%} of the cameras are on the room side of it (min {CAMERAS_ROOM_SIDE_OK:.0%})")
    if status != 'failed' and book_count and not books_markers:
        reasons.append('No group could be placed')
    elif status != 'failed' and book_count and placed_fraction < MIN_PLACED_FRACTION:
        reasons.append(f'Only {len(books_markers)}/{book_count} book groups could be placed')
    out = {'status': status, 'registered_frames': len(views), 'registered_ratio': round(ratio, 4),
           'unregistered_frames': sorted(set(frame_ids) - set(views)),
           'unregistered_reason': 'not registered: too few or inconsistent feature matches to the reconstructed model',
           'points_total': points_total, 'points_filtered': points_filtered, 'points_kept': int(len(xyz)),
           'mean_reprojection_error_px': mean_error,
           'points': [[round(float(v), 4) for v in p] for p in xyz_a], 'point_colors': [[int(v) for v in c] for c in colour],
           'cameras': cameras,
           'plane': None if plane is None else {
               'normal': [round(float(v), 6) for v in plane['normal']], 'point': [round(float(v), 5) for v in plane['point']],
               'frame': 'sfm_original', 'inlier_ratio': round(plane['inlier_ratio'], 4), 'rms': round(plane['rms'], 5),
               'threshold': round(plane['threshold'], 5), 'scene_extent': round(scene_extent, 4),
               'rms_over_extent': round(plane['rms'] / scene_extent, 4) if scene_extent else None,
               'in_aligned_frame': 'z = 0 (normal +z, camera side positive)'},
           'alignment': alignment, 'plane_tolerance': round(plane_tol, 5), 'plane_consistency': consistency,
           'markers': books_markers, 'markers_total': len(books_markers), 'unplaced_groups': books_unplaced,
           'group_markers': len(books_markers), 'groups_total': book_count,
           'marker_methods': {'triangulated': sum(m['method'] == 'triangulated' for m in books_markers),
                              'plane_projected': sum(m['method'] == 'plane_projected' for m in books_markers)},
           'item_markers': item_markers, 'item_markers_total': len(item_markers), 'unplaced_items': item_unplaced}
    if reasons:
        out['reason'] = '; '.join(reasons)
    return out


def build(source_run, out_run, *, max_image_size=None, overlap=None, progress=None, frames_rule=None, max_frames=None, max_features=None,
          quadratic=None, mapper_options=None, profile=None, merge=None, merge_mode_name=None):
    """SfM of a saved run into a NEW `out_run`. `frames_rule` ('all' | 'diet' | 'keyframes'; default env LIBRARY_RECON3D_FRAMES, else
    all) picks the frames fed to COLMAP (see select_frames); `max_frames` (default env LIBRARY_RECON3D_MAX_FRAMES) caps them by
    sharpness; `overlap` (default: scaled to the kept share, see sequential_overlap) is the sequential-matching window in frames.
    `profile` (default env LIBRARY_RECON3D_PROFILE, else 'auto' = 'fast' from AUTO_FAST_MIN_FRAMES frames on, else 'default'; see PROFILES)
    presets the per-frame cost settings below.
    `max_image_size` / `max_features` (SIFT cap) / `quadratic` (COLMAP's extra i+2^k pairs) default to the profile / module constants / env
    (LIBRARY_RECON3D_IMAGE_SIZE, LIBRARY_RECON3D_MAX_FEATURES, LIBRARY_RECON3D_QUADRATIC=0|1); `mapper_options` sets attributes of
    pycolmap.IncrementalPipelineOptions. `merge` (default env LIBRARY_RECON3D_MERGE, on) joins the fragments of a split build;
    `merge_mode_name` (default env LIBRARY_RECON3D_MERGE_MODE, else DEFAULT_MERGE_MODE) decides which merge steps run in this build and which are
    deferred to merge_pass (see MERGE_MODES / pipeline.recon3d_merge.MergePlan)."""
    import pycolmap
    requested = (profile or os.environ.get('LIBRARY_RECON3D_PROFILE') or DEFAULT_PROFILE).strip().lower()
    if requested != 'auto' and requested not in PROFILES:
        raise ValueError(f'unknown profile {requested!r}; choose one of auto, {", ".join(PROFILES)}')
    source, out = Path(source_run).resolve(), Path(out_run).resolve()
    if out.exists():
        raise ValueError('Refusing existing versioned output ' + str(out))
    all_frames = _frame_table(source)
    rule = (frames_rule or os.environ.get('LIBRARY_RECON3D_FRAMES') or DEFAULT_RULE).strip().lower()
    cap = max_frames if max_frames is not None else _env_int('LIBRARY_RECON3D_MAX_FRAMES')
    frames, diet = select_frames(all_frames, rule, cap)
    if overlap is None:
        overlap = sequential_overlap(len(frames), len(all_frames))
    diet['overlap'] = overlap
    profile_name = ('fast' if len(frames) >= AUTO_FAST_MIN_FRAMES else 'default') if requested == 'auto' else requested
    preset = PROFILES[profile_name]
    max_image_size = max_image_size or _env_int('LIBRARY_RECON3D_IMAGE_SIZE') or preset.get('max_image_size') or IMAGE_SIZE
    max_features = max_features or _env_int('LIBRARY_RECON3D_MAX_FEATURES_SIFT') or preset.get('max_features') or MAX_FEATURES
    quadratic = preset.get('quadratic', QUADRATIC) if quadratic is None else quadratic
    mapper_options = mapper_options if mapper_options is not None else preset.get('mapper_options')
    if os.environ.get('LIBRARY_RECON3D_QUADRATIC') in ('0', '1'):
        quadratic = os.environ['LIBRARY_RECON3D_QUADRATIC'] == '1'
    out.mkdir(parents=True)
    progress = progress or Progress(out)
    started = time.perf_counter()
    work = out / '_work'
    images = work / 'images'
    images.mkdir(parents=True)
    names_to_frame, ordered_links = {}, []
    ordered = [f"{f['frame_id']}{Path(f['image_path']).suffix.lower()}" for f in frames]
    prefix = ordered != sorted(ordered)       # COLMAP's sequential order is the sorted image name: keep time order if the ids do not sort
    for i, f in enumerate(frames):
        src = (source / f['image_path']).resolve()
        if not src.is_file():
            raise ValueError('Missing frame ' + f['image_path'])
        link = images / (f'{i:06d}_{ordered[i]}' if prefix else ordered[i])
        link.symlink_to(src)
        names_to_frame[link.name] = f['frame_id']
        ordered_links.append(link.name)
    database = work / 'database.db'
    t = {}

    progress('features', total=len(frames), frames_in_run=len(all_frames), overlap=overlap)
    t0, cpu0 = time.perf_counter(), time.process_time()
    cpu = {}
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = max_image_size
    extraction.sift.max_num_features = max_features
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = 'SIMPLE_RADIAL'
    pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=extraction, device=pycolmap.Device.cpu)
    t['features_s'], cpu['features_cpu_s'] = time.perf_counter() - t0, time.process_time() - cpu0

    progress('matching')
    t0, cpu0 = time.perf_counter(), time.process_time()
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pairing.quadratic_overlap = bool(quadratic)
    pairing.loop_detection = False
    pycolmap.match_sequential(database, pairing_options=pairing, device=pycolmap.Device.cpu)
    t['matching_s'], cpu['matching_cpu_s'] = time.perf_counter() - t0, time.process_time() - cpu0

    progress('mapping')
    t0, cpu0 = time.perf_counter(), time.process_time()
    sparse = work / 'sparse'
    sparse.mkdir()
    mapping = pycolmap.IncrementalPipelineOptions()
    for key, value in (mapper_options or {}).items():
        setattr(mapping, key, value)
    models = pycolmap.incremental_mapping(database, images, sparse, mapping)
    t['mapping_s'], cpu['mapping_cpu_s'] = time.perf_counter() - t0, time.process_time() - cpu0

    sizes = sorted(((len(_registered_images(r)), i) for i, r in models.items()), reverse=True)
    do_merge = merge_enabled() if merge is None else bool(merge)
    mode = merge_mode(merge_mode_name)
    members, merged_from = ([(models[sizes[0][1]], None)] if models else []), None
    if do_merge and len(models) > 1:
        from pipeline import recon3d_merge as RM
        if mode != 'inline' and sizes[0][0] / len(all_frames) >= RM.MERGE_SKIP_SHARE:
            merged_from = {'used': 'largest_only', 'skipped': f'the largest model already registers {sizes[0][0] / len(all_frames):.1%} of the run '
                                                              f'(>= {RM.MERGE_SKIP_SHARE:.0%}); no merge'}
        else:
            progress('merging', models=len(models))
            t0, cpu0 = time.perf_counter(), time.process_time()
            plan = RM.MergePlan(mode, merge_budget(), started=started, mapping_s=t['mapping_s'],
                                seconds_per_pair=t['matching_s'] / max(1, RM.sequential_pair_count(len(frames), overlap, quadratic)))
            try:
                members, merged_from = RM.merge(database, images, work, models, ordered_links, mapping, overlap=overlap,
                                                panorama_dir=RM.newest_panorama(source), progress=progress, plan=plan,
                                                sharpness=None if mode == 'inline' else [_sharp(f) for f in frames], total_frames=len(all_frames),
                                                shelf=shelf_check(source, names_to_frame))
            except Exception as exc:      # the merge is an add-on: a failure keeps the largest model and says why
                members = [(models[sizes[0][1]], None)]
                merged_from = {'used': 'largest_only', 'error': f'{type(exc).__name__}: {exc}', 'plan': plan.log}
            t['merge_s'], cpu['merge_cpu_s'] = time.perf_counter() - t0, time.process_time() - cpu0
    deferred = (merged_from or {}).get('deferred')
    result = {'schema': SCHEMA, 'source_run': str(source), 'unscaled': True, 'limits': LIMITS,
              'library': {'pycolmap': pycolmap.__version__, 'features': 'SIFT (CPU)', 'matcher': f'sequential overlap={overlap}' + (' +quadratic' if quadratic else ''),
                          'camera_model': 'SIMPLE_RADIAL (single shared camera)', 'max_image_size': max_image_size,
                          'max_features': max_features, 'profile': profile_name, 'profile_requested': requested, 'quadratic': bool(quadratic),
                          **({'mapper_options': mapper_options} if mapper_options else {})},
              'frames_total': len(all_frames), 'frames_used': len(frames), 'diet': diet, 'models_found': len(models), 'model_registered_sizes': [s for s, _ in sizes],
              'merge': _merge_header(do_merge, mode, merged_from, members, stage='first_map')}
    if deferred:                       # keep the database, symlinks and first-pass models for the merge pass (--merge-from); it deletes them
        (work / 'merge_state.json').write_text(json.dumps({
            'frame_ids': [f['frame_id'] for f in frames], 'names': ordered_links, 'overlap': overlap, 'quadratic': bool(quadratic),
            'mapper_options': mapper_options or {}, 'first_map_timings': {k: round(v, 2) for k, v in {**t, **cpu}.items()}}))
        work.rename(out / MERGE_CACHE)
    return _publish(out, source, result, frames, all_frames, models, members, merged_from, names_to_frame, t, cpu, started, progress,
                    work=None if deferred else work)


def shelf_check(source, names_to_frame):
    """The join check (recon3d_merge.ShelfCheck) for a run's book groups; None when LIBRARY_RECON3D_JOIN_CHECK=0 or the run has no groups."""
    from pipeline import recon3d_merge as RM
    if os.environ.get('LIBRARY_RECON3D_JOIN_CHECK', '1').strip().lower() in ('0', 'false', 'no', 'off'):
        return None
    books = [g for g in load_groups(source)[0] if g['kind'] == 'book']
    return RM.ShelfCheck(books, names_to_frame) if books else None


def _merge_header(enabled, mode, merged_from, members, stage):
    applied = bool(merged_from and (merged_from.get('used') == 'remapped' or len(members) > 1))
    deferred = (merged_from or {}).get('deferred')
    return {'enabled': enabled, 'mode': mode, 'stage': stage, 'applied': applied, 'model': 'merged' if applied else 'largest',
            'deferred': {'steps': deferred, 'cache': MERGE_CACHE,
                         'note': 'this map is published first; the merge pass (python -m pipeline.recon3d --merge-from <this dir>) writes the joined map '
                                 'as the next version'} if deferred else None}


def _publish(out, source, result, frames, all_frames, models, members, merged_from, names_to_frame, t, cpu, started, progress, work=None):
    """Markers + recon3d.json for the chosen model(s) (`members`), the timings, then the work dir is removed (when given)."""
    if merged_from is not None:
        result['merged_from'] = merged_from
    progress('markers')
    t0 = time.perf_counter()
    groups, book_count = load_groups(source)
    books = [g for g in groups if g['kind'] == 'book']
    items = [g for g in groups if g['kind'] == 'item']
    result.update(book_count=book_count, item_groups_total=len(items))
    if not models:
        result.update(status='failed', reason='Incremental mapping produced no model', registered_frames=0, registered_ratio=0.0,
                      registered_of_total=0.0, unregistered_frames=sorted(f['frame_id'] for f in frames),
                      unregistered_reason='no SfM model was produced', points=[], point_colors=[], cameras=[], markers=[],
                      unplaced_groups=[{'group_id': g['group_id'], 'code': 'no_model', 'reason': 'SfM produced no model'} for g in books],
                      markers_total=0, group_markers=0, groups_total=book_count, plane=None, alignment={'applied': False},
                      marker_methods={'triangulated': 0, 'plane_projected': 0}, item_markers=[], item_markers_total=0,
                      unplaced_items=[{'group_id': g['group_id'], 'code': 'no_model', 'reason': 'SfM produced no model'} for g in items],
                      points_kept=0, mean_reprojection_error_px=None)
    else:
        xyz, colour, views, intrinsics, error, n_all, n_filtered = model_inputs(members, names_to_frame)
        progress('align', points=int(len(xyz)))
        result.update(scene_result(xyz, colour, views, intrinsics, books, items, frame_ids=[f['frame_id'] for f in frames],
                                   mean_error=error, points_total=n_all, points_filtered=n_filtered))
        result['registered_of_total'] = round(len(views) / len(all_frames), 4)
    t['markers_s'] = time.perf_counter() - t0
    t['total_s'] = time.perf_counter() - started
    cpu['total_cpu_s'] = sum(v for k, v in cpu.items() if k != 'total_cpu_s')
    result['timings'] = {k: round(v, 2) for k, v in {**t, **cpu}.items()}      # wall seconds per stage + process CPU seconds (wall is inflated by a busy machine)
    progress('writing')
    tmp = out / 'recon3d.json.tmp'
    tmp.write_text(json.dumps(result))
    tmp.replace(out / 'recon3d.json')
    if work is not None:
        shutil.rmtree(work, ignore_errors=True)   # database/symlinks only; the source run is untouched
    progress('done')
    summary = {k: result.get(k) for k in ('status', 'registered_frames', 'frames_used', 'frames_total', 'registered_ratio', 'registered_of_total', 'points_total',
                                          'points_filtered', 'points_kept', 'mean_reprojection_error_px', 'book_count', 'markers_total',
                                          'marker_methods', 'timings')}
    summary['unplaced_groups'] = len(result.get('unplaced_groups') or [])
    summary['models_found'] = result.get('models_found')
    if result.get('merged_from'):
        mf = result['merged_from']
        summary['merge'] = {k: mf.get(k) for k in ('used', 'fragments_before', 'largest_before', 'registered', 'fragments_after', 'deferred', 'seconds')}
    print(json.dumps(summary), flush=True)
    return out / 'recon3d.json'


def merge_pass(first_run, out_run, *, progress=None):
    """Second build of a run whose first map deferred merge steps (MergePlan): reuses that build's database (features, sequential
    and any bridge matches), symlinks and first-pass models from `<first_run>/_merge_cache`, runs the merge with every step allowed,
    writes the joined map into the NEW `out_run` (the next version, so it becomes the newest map) and deletes the cache."""
    import pycolmap
    from pipeline import recon3d_merge as RM
    first, out = Path(first_run).resolve(), Path(out_run).resolve()
    cache = first / MERGE_CACHE
    if not (cache / 'merge_state.json').is_file() or not (first / 'recon3d.json').is_file():
        raise ValueError(f'No deferred merge to run in {first} (no {MERGE_CACHE}/merge_state.json)')
    if out.exists():
        raise ValueError('Refusing existing versioned output ' + str(out))
    prev = json.loads((first / 'recon3d.json').read_text())
    state = json.loads((cache / 'merge_state.json').read_text())
    source = Path(prev['source_run'])
    all_frames = _frame_table(source)
    by_id = {f['frame_id']: f for f in all_frames}
    frames = [by_id[i] for i in state['frame_ids']]
    names_to_frame = dict(zip(state['names'], state['frame_ids']))
    out.mkdir(parents=True)
    progress = progress or Progress(out)
    started = time.perf_counter()
    progress('merging', step='loading the first build', frames_in_run=len(all_frames))
    t, cpu = {}, {}
    t0, cpu0 = time.perf_counter(), time.process_time()
    models = _load_models(cache / 'sparse')
    mapping = pycolmap.IncrementalPipelineOptions()
    for key, value in (state.get('mapper_options') or {}).items():
        setattr(mapping, key, value)
    sizes = sorted(((len(_registered_images(r)), i) for i, r in models.items()), reverse=True)
    members = [(models[sizes[0][1]], None)] if models else []
    try:
        members, merged_from = RM.merge(cache / 'database.db', cache / 'images', cache, models, state['names'], mapping,
                                        overlap=state['overlap'], panorama_dir=RM.newest_panorama(source), progress=progress,
                                        plan=RM.MergePlan('inline'), total_frames=len(all_frames),
                                        sharpness=[_sharp(f) for f in frames] if MERGE_PASS_RULE == 'cheap' else None,
                                        shelf=shelf_check(source, names_to_frame))
    except Exception as exc:          # same as the first build: the largest model stands and the reason is recorded
        merged_from = {'used': 'largest_only', 'error': f'{type(exc).__name__}: {exc}'}
    t['merge_s'], cpu['merge_cpu_s'] = time.perf_counter() - t0, time.process_time() - cpu0
    keep = ('library', 'frames_total', 'frames_used', 'diet', 'models_found', 'model_registered_sizes')
    result = {'schema': SCHEMA, 'source_run': str(source), 'unscaled': True, 'limits': LIMITS, **{k: prev.get(k) for k in keep},
              'merge': {**_merge_header(True, 'inline', merged_from, members, stage='merge_pass'),
                        'first_map': {'run': first.name, 'registered_frames': prev.get('registered_frames'),
                                      'registered_ratio': prev.get('registered_ratio'), 'total_s': (prev.get('timings') or {}).get('total_s'),
                                      'deferred_steps': ((prev.get('merge') or {}).get('deferred') or {}).get('steps')}}}
    t['first_map_total_s'] = float((prev.get('timings') or {}).get('total_s') or 0.0)
    path = _publish(out, source, result, frames, all_frames, models, members, merged_from, names_to_frame, t, cpu, started, progress)
    shutil.rmtree(cache, ignore_errors=True)
    return path




def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-image-size', type=int, default=None, help=f'SIFT image side cap (default env LIBRARY_RECON3D_IMAGE_SIZE, else {IMAGE_SIZE})')
    p.add_argument('--max-features', type=int, default=None, help=f'SIFT feature cap per image (default env LIBRARY_RECON3D_MAX_FEATURES_SIFT, else {MAX_FEATURES})')
    p.add_argument('--no-quadratic', action='store_true', help='sequential matching without the extra i+2^k pairs')
    p.add_argument('--mapper', default=None, help='JSON of pycolmap.IncrementalPipelineOptions attributes')
    p.add_argument('--overlap', type=int, default=None, help='sequential-matching window in frames (default: 12 scaled by the kept share, min 6)')
    p.add_argument('--frames', choices=FRAME_RULES, default=None, help=f'frames fed to SfM (default: env LIBRARY_RECON3D_FRAMES, else {DEFAULT_RULE})')
    p.add_argument('--profile', choices=['auto', *sorted(PROFILES)], default=None, help=f'per-frame cost preset (default: env LIBRARY_RECON3D_PROFILE, else {DEFAULT_PROFILE})')
    p.add_argument('--max-frames', type=int, default=None, help='cap by sharpness, time order kept (default: env LIBRARY_RECON3D_MAX_FRAMES, else none)')
    p.add_argument('--merge', choices=['0', '1'], default=None, help='join the fragments of a split build (default: env LIBRARY_RECON3D_MERGE, else on)')
    p.add_argument('--merge-mode', choices=MERGE_MODES, default=None, help=f'where the costly merge steps run (default: env LIBRARY_RECON3D_MERGE_MODE, else {DEFAULT_MERGE_MODE})')
    p.add_argument('--merge-from', default=None, help="the merge pass: a <run>_recon3d_vN dir whose merge was deferred, or 'latest' (newest such dir of --source-run)")
    a = p.parse_args()
    if a.merge_from:
        first = pending_merge(a.source_run) if a.merge_from == 'latest' else Path(a.merge_from)
        if first is None:
            raise SystemExit(f'{a.source_run}: no 3D map with a deferred merge (recon3d.json merge.deferred + {MERGE_CACHE}/)')
        merge_pass(first, a.out)
        return
    build(a.source_run, a.out, max_image_size=a.max_image_size, overlap=a.overlap, frames_rule=a.frames, max_frames=a.max_frames,
          max_features=a.max_features, quadratic=False if a.no_quadratic else None, mapper_options=json.loads(a.mapper) if a.mapper else None, profile=a.profile,
          merge=None if a.merge is None else a.merge == '1', merge_mode_name=a.merge_mode)


if __name__ == '__main__':
    main()
