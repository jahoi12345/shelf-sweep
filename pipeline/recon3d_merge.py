"""Merge the fragments of an incremental SfM build into one model, with the evidence for every link.

A handheld sweep breaks COLMAP's incremental mapper into several models ("fragments"): sequential matching only pairs frames
up to 12 apart (+ the i+2^k quadratic pairs), and across a fast pan or a blurred stretch those pairs fail, so the mapper starts
a new model. Nearly every frame IS registered somewhere (YugoTest12: 13 models covering 506 of 509 frames), so the problem is
linking, not registration. Three steps, each explainable and recorded:

1. bridge_pairs: extra image pairs that sequential matching never tried: (a) around every fragment end, its last/first frames
   against the other fragments' frames within +-END_WINDOW frames (a pan that comes back, a stretch after a blur), (b) a
   subsample of every fragment against every other fragment (rows of a shelf filmed minutes apart overlap vertically), (c) the
   2D panorama's verified piece links when a panorama exists. They are matched + geometrically verified by COLMAP like any
   other pair (match_pairs); cross_links counts the verified inlier pairs between each two fragments (the evidence).
2. The incremental mapper runs again on the extended database (pipeline.recon3d does it).
3. chain_models: models that still come out separate but share registered frames are joined by a similarity transform
   estimated from the shared camera centres (RANSAC), accepted only when enough shared frames agree within a residual bound;
   frames/points of the joined model are mapped into the main model's frame. Models without such evidence stay separate and
   are reported; no geometry is invented.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

END_WINDOW = 60          # frames either side of a fragment end searched for a partner (12 s at 5 fps)
END_FRAMES = 8           # frames at each fragment end used as the bridge side
END_STEP = 2             # every 2nd end frame / partner frame (5 fps frames overlap > 90%)
PARTNER_STEP = 3
MAX_REPRESENTATIVES = 160  # all fragments together: subsample matched all-against-all across fragments (rows filmed apart)
MIN_LINK_INLIERS = 30    # a verified pair counts as a link between fragments from this many inliers (COLMAP's abs-pose floor)
HINT_PAIRS = 120         # cap of pairs per panorama piece link
CHAIN_MIN_SHARED = 4     # shared registered frames needed to join two models by a similarity
CHAIN_MAX_RESIDUAL = 0.02  # RMS of the shared camera centres after the similarity, relative to the target model's camera extent
CHAIN_MIN_INLIER_FRAC = 0.75
LINK_MIN_INLIERS = 40    # 3D-3D correspondences that must agree on a similarity between two models
LINK_THRESHOLD = 0.02    # correspondence agrees within this fraction of the target model's point extent
LINK_MIN_INLIER_FRAC = 0.35
HORIZON_MAX_TILT_DEG = 5.0       # a joined model may tilt the main model's level horizon by at most this (handheld roll is small)
HORIZON_MIN_CONDITIONING = 3.0   # the main model's cameras must pan enough for its horizontal plane to be defined
# Shelf-plane check: every shelf runs along ONE straight wall, so a correct join keeps every book on ONE plane.
# The horizon test above sees roll only; a join rotated about the vertical (yaw) passed it and fanned IESFULLTest1's shelf into crossing lines.
PLANE_MAX_ANGLE_DEG = 5.0        # the joined part's own shelf-face normal vs the reference shelf normal, any axis (incl. about the vertical)
PLANE_MAX_OFFSET_SPINES = 0.5    # median distance of the joined part's book positions to the reference plane, in spine heights
PLANE_MIN_BOOKS = 3              # book positions needed to verify a joined part at all (fewer: refused, nothing to check it against)
CAMERA_SIDE_MIN = 0.9            # share of the joined part's cameras that must be on the camera side of the reference plane (the room side)
FACE_MIN_COS = 0.5               # a shelf face is seen face-on: |normal . mean viewing direction| >= this (a floor / shelf board is edge-on)
FACE_TRIES = 3                   # dominant planes tried (each next one without the previous inliers) to find the face
PART_SPINE_LIMIT = 1500          # spine polygons cast per part (the median settles long before)

# Merge plan: the same merge in less time, and never holding back the first map.
MERGE_SKIP_SHARE = 0.80  # the first pass's largest model already registers this share of the run: no merge at all
REMAP_SKIP_SHARE = 0.75  # the similarity-joined fragments register this share: the costly re-map is skipped (it added 2.8 points on YT11)
REMAP_FACTOR = 2.0       # predicted re-map seconds = this x the first mapping's seconds (development builds: 1.2-2.4 x)
CHEAP_RULE = {'ends': END_FRAMES, 'end_keep': 2, 'partner_step': 3, 'reps': 80, 'rep_sharpest': True}   # bridge pairs with sharpness known


class MergePlan:
    """Where each costly merge step runs. 'inline': in this build (the full rule, and the merge pass). 'deferred': never here (left to the merge
    pass after the first map is published). 'auto': here only when this build is predicted to end within `budget_s` of `started`
    (elapsed + predicted seconds of the step); the predictions come from this build's own sequential matching rate and mapping time,
    so a busy machine defers more. Every decision is logged (`log`) into merged_from.plan."""

    def __init__(self, mode='inline', budget_s=None, started=None, seconds_per_pair=0.0, mapping_s=0.0, remap_factor=REMAP_FACTOR):
        self.mode, self.budget_s, self.started = mode, budget_s, started
        self.seconds_per_pair, self.mapping_s, self.remap_factor = float(seconds_per_pair), float(mapping_s), remap_factor
        self.log = []

    def allow(self, step, predicted_s):
        elapsed = time.perf_counter() - self.started if self.started is not None else 0.0
        ok = self.mode == 'inline' or (self.mode == 'auto' and self.budget_s is not None and elapsed + predicted_s <= self.budget_s)
        self.log.append({'step': step, 'mode': self.mode, 'elapsed_s': round(elapsed, 1), 'predicted_s': round(float(predicted_s), 1),
                         'budget_s': self.budget_s, 'decision': 'run in this build' if ok else 'deferred to the merge pass'})
        return ok


def sequential_pair_count(n, overlap, quadratic=True):
    """Image pairs COLMAP's sequential matcher tries: i + 1..overlap, plus i + 2^k (k < overlap) with quadratic overlap."""
    ds = set(range(1, overlap + 1)) | ({2 ** k for k in range(overlap)} if quadratic else set())
    return sum(max(0, n - d) for d in ds)


# ----------------------------------------------------------------------------------------------------- fragments

def _ranges(idx, gap=2):
    """Contiguous runs of sorted indices (a hole of <= gap frames does not split a run)."""
    out = []
    for i in idx:
        if out and i - out[-1][1] <= gap + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [tuple(r) for r in out]


def _reg_names(rec):
    out = []
    for image in rec.images.values():
        try:
            ok = image.has_pose
        except AttributeError:
            ok = True
        if ok:
            out.append(image.name)
    return out


def fragments_from_models(models, names):
    """[{model, size, frames (sorted run indices), ranges}] largest first. `names`: image name per run frame index."""
    index = {n: i for i, n in enumerate(names)}
    frags = []
    for key, rec in models.items():
        idx = sorted(index[n] for n in _reg_names(rec) if n in index)
        if idx:
            frags.append({'model': int(key), 'size': len(idx), 'frames': idx, 'ranges': _ranges(idx)})
    frags.sort(key=lambda f: (-f['size'], f['model']))
    return frags


def _sequential_pair(i, j, overlap):
    d = abs(i - j)
    return d <= overlap or (d & (d - 1) == 0 and d.bit_length() - 1 < overlap)     # i + 2^k quadratic pairs (k < overlap)


def _sharpest(idx, sharpness, keep):
    """The `keep` sharpest of the frame indices `idx` (ties: earlier first), in time order."""
    return sorted(sorted(idx, key=lambda i: (-sharpness[i], i))[:keep])


def _blocks(idx, size):
    return [idx[s:s + size] for s in range(0, len(idx), size)]


def bridge_pairs(frags, n_frames, *, overlap=12, window=END_WINDOW, ends=END_FRAMES, reps=MAX_REPRESENTATIVES, hints=(),
                 sharpness=None, end_keep=None, partner_step=PARTNER_STEP, rep_sharpest=False):
    """Extra (i, j) frame-index pairs (i < j) across fragments that sequential matching did not try, and counts by reason.

    Without `sharpness` (full rule): every END_STEP-th of the `ends` frames at a fragment end, every `partner_step`-th partner frame,
    `reps` representatives spread evenly over each fragment. With `sharpness` (per run frame index; cheap rule): the `end_keep`
    sharpest of the `ends` end frames, the sharpest frame of every block of `partner_step` partner frames, and (rep_sharpest) the
    sharpest frame of each representative slot instead of its middle, so blurred frames, which rarely verify, are not matched."""
    owner = {}
    for k, f in enumerate(frags):
        for i in f['frames']:
            owner.setdefault(i, set()).add(k)
    pairs, why = set(), {'ends': 0, 'representatives': 0, 'panorama_hints': 0}
    sharp = sharpness is not None

    def add(i, j, reason):
        if i == j or _sequential_pair(i, j, overlap):
            return
        a, b = min(i, j), max(i, j)
        if owner.get(a, set()) & owner.get(b, set()):
            return                                           # already in one model: nothing to link
        if (a, b) not in pairs:
            pairs.add((a, b))
            why[reason] += 1

    for k, f in enumerate(frags):                           # (a) fragment ends against nearby frames of other fragments / no fragment
        for lo, hi in f['ranges']:
            for end, side in ((lo, range(lo, min(hi, lo + ends - 1) + 1)), (hi, range(max(lo, hi - ends + 1), hi + 1))):
                near = [j for j in range(max(0, end - window), min(n_frames, end + window + 1)) if k not in owner.get(j, set())]
                if sharp:
                    mine = _sharpest(list(side), sharpness, end_keep or len(side))
                    partners = [_sharpest(b, sharpness, 1)[0] for b in _blocks(near, max(1, partner_step))]
                else:
                    mine = list(side)[::END_STEP]
                    partners = near[::partner_step]
                for i in mine:
                    for j in partners:
                        add(i, j, 'ends')
    total = sum(f['size'] for f in frags)                  # (b) representatives, quota proportional to fragment size
    if len(frags) > 1 and reps:
        chosen = []
        for k, f in enumerate(frags):
            quota = max(2, round(reps * f['size'] / total))
            if sharp and rep_sharpest:
                slots = np.array_split(np.asarray(f['frames']), min(quota, f['size']))
                chosen.append(sorted(_sharpest(s.tolist(), sharpness, 1)[0] for s in slots if len(s)))
                continue
            idx = np.linspace(0, f['size'] - 1, min(quota, f['size'])).round().astype(int)
            chosen.append([f['frames'][i] for i in sorted(set(idx.tolist()))])
        for a in range(len(chosen)):
            for b in range(a + 1, len(chosen)):
                for i in chosen[a]:
                    for j in chosen[b]:
                        add(i, j, 'representatives')
    for fa, fb in hints:                                    # (c) the 2D panorama's verified piece links
        sa, sb = fa[::max(1, len(fa) // 12)], fb[::max(1, len(fb) // 12)]
        for i in sa[:12]:
            for j in sb[:HINT_PAIRS // 12]:
                add(i, j, 'panorama_hints')
    return sorted(pairs), why


def panorama_hints(panorama_dir, names):
    """Frame-index lists of the two pieces of every verified piece link of a 2D panorama (panorama_count.json), [] when absent."""
    import json
    d = Path(panorama_dir)
    pc, pj = d / 'panorama_count.json', d / 'panorama.json'
    if not pc.is_file() or not pj.is_file():
        return []
    try:
        links = json.loads(pc.read_text()).get('piece_links') or []
        comps = {c['component_id']: c for c in json.loads(pj.read_text()).get('components', [])}
    except (ValueError, OSError):
        return []
    index = {Path(n).stem.split('_')[-1]: i for i, n in enumerate(names)}

    def frames_of(cid):
        out = []
        for s in (comps.get(cid) or {}).get('source_frames', []):
            fid = s if isinstance(s, str) else s.get('frame_id')
            if fid in index:
                out.append(index[fid])
        return sorted(out)
    return [(frames_of(link['a']), frames_of(link['b'])) for link in links if link.get('a') in comps and link.get('b') in comps]


# ----------------------------------------------------------------------------------------------------- matching + evidence

def match_pairs(database, pairs, names, pairs_file):
    """Match + verify the given pairs into the database (COLMAP's imported-pairs matcher). Returns counts and seconds."""
    import pycolmap
    pairs_file = Path(pairs_file)
    pairs_file.write_text(''.join(f'{names[i]} {names[j]}\n' for i, j in pairs))
    db = pycolmap.Database.open(str(database))
    before = db.num_verified_image_pairs()
    db.close()
    t0 = time.perf_counter()
    if pairs:
        opts = pycolmap.ImportedPairingOptions()
        opts.match_list_path = str(pairs_file)
        pycolmap.match_image_pairs(str(database), pairing_options=opts, device=pycolmap.Device.cpu)
    db = pycolmap.Database.open(str(database))
    after = db.num_verified_image_pairs()
    db.close()
    return {'pairs_tried': len(pairs), 'verified_pairs_added': int(after - before), 'seconds': round(time.perf_counter() - t0, 1)}


def _verified(database, names):
    """{(i, j): inliers} for every verified pair in the database (run frame indices, i < j)."""
    import pycolmap
    db = pycolmap.Database.open(str(database))
    index = {n: i for i, n in enumerate(names)}
    by_id = {im.image_id: index.get(im.name) for im in db.read_all_images()}
    pair_ids, inliers = db.read_two_view_geometry_num_inliers()
    db.close()
    out = {}
    for pid, n in zip(pair_ids, inliers):
        a, b = pycolmap.pair_id_to_image_pair(pid)
        i, j = by_id.get(a), by_id.get(b)
        if i is not None and j is not None and n > 0:
            out[(min(i, j), max(i, j))] = int(n)
    return out


def cross_links(frags, database, names, min_inliers=MIN_LINK_INLIERS):
    """Verified pairs with >= min_inliers between frames of two different fragments (frames registered in both excluded):
    [{a, b (model ids), pairs, max_inliers, best_pair}] sorted by pairs."""
    verified = _verified(database, names)
    owner = {}
    for f in frags:
        for i in f['frames']:
            owner.setdefault(i, set()).add(f['model'])
    acc = {}
    for (i, j), n in verified.items():
        if n < min_inliers:
            continue
        for a in owner.get(i, ()):
            for b in owner.get(j, ()):
                if a == b or b in owner.get(i, ()) or a in owner.get(j, ()):
                    continue
                key = (min(a, b), max(a, b))
                e = acc.setdefault(key, {'a': key[0], 'b': key[1], 'pairs': 0, 'max_inliers': 0, 'best_pair': None})
                e['pairs'] += 1
                if n > e['max_inliers']:
                    e['max_inliers'], e['best_pair'] = n, [names[i], names[j]]
    return sorted(acc.values(), key=lambda e: (-e['pairs'], e['a'], e['b']))


# ----------------------------------------------------------------------------------------------------- diagnosis

GEOMETRY = {0: 'undefined', 1: 'degenerate', 2: 'calibrated', 3: 'uncalibrated', 4: 'planar', 5: 'panoramic (rotation only)',
            6: 'planar or panoramic', 7: 'watermark', 8: 'multiple'}


def _geometry(database, names, pair):
    """COLMAP's two-view configuration of a verified pair (panoramic = pure rotation: nothing new can be triangulated)."""
    import pycolmap
    db = pycolmap.Database.open(str(database))
    try:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        g = db.read_two_view_geometry(ids[names[pair[0]]], ids[names[pair[1]]])
        return GEOMETRY.get(int(g.config), str(g.config))
    finally:
        db.close()


def diagnose(frags, frames, database, names, near=12):
    """Where the chain breaks: per fragment its ranges; per break in time order the frames between, their blur/sharpness/motion
    and the best verified inliers across it (from the database)."""
    n = len(frames)
    owner = {}
    for k, f in enumerate(frags):
        for i in f['frames']:
            owner.setdefault(i, k)          # largest fragment that holds the frame
    verified = _verified(database, names) if database and Path(database).exists() else {}
    sharp = np.array([float((f.get('quality') or {}).get('sharpness') or 0.0) for f in frames])
    med = float(np.median(sharp)) if len(sharp) else 0.0
    seq = [(i, owner.get(i)) for i in range(n)]
    breaks, last = [], None
    for i, k in seq:
        if k is None:
            continue
        if last is not None and k != last[1]:
            lo, hi = last[0], i
            between = list(range(lo + 1, hi))
            window = range(max(0, lo - 2), min(n, hi + 3))
            qa = [frames[j].get('quality') or {} for j in window]
            best, best_pair = 0, None
            for a in range(max(0, lo - near + 1), lo + 1):
                for b in range(hi, min(n, hi + near)):
                    if verified.get((a, b), 0) > best:
                        best, best_pair = verified[(a, b)], (a, b)
            breaks.append({'after_frame': frames[lo]['frame_id'], 'before_frame': frames[hi]['frame_id'],
                           'from_fragment': frags[last[1]]['model'], 'to_fragment': frags[k]['model'],
                           'frames_between_unowned': len(between),
                           't_s': [frames[lo].get('timestamp_s'), frames[hi].get('timestamp_s')],
                           'blur_gated_near': int(sum(bool(q.get('blur_gated')) for q in qa)),
                           'sharpness_near_median_rel': round(float(np.median([sharp[j] for j in window])) / med, 2) if med else None,
                           'motion_px_per_s_near_max': max([q.get('motion_px_per_s') or 0 for q in qa] or [0]),
                           'best_inliers_across_12': best,
                           'best_pair_geometry': _geometry(database, names, best_pair) if best_pair else None})
        last = (i, k)
    shared = []
    for a in range(len(frags)):
        for b in range(a + 1, len(frags)):
            s = set(frags[a]['frames']) & set(frags[b]['frames'])
            if s:
                shared.append({'a': frags[a]['model'], 'b': frags[b]['model'], 'shared_frames': len(s)})
    covered = set(owner)
    return {'frames': n, 'fragments': len(frags), 'registered_any': len(covered), 'unregistered': n - len(covered),
            'run_median_sharpness': round(med, 1), 'blur_gated_total': int(sum(bool((f.get('quality') or {}).get('blur_gated')) for f in frames)),
            'per_fragment': [{'model': f['model'], 'size': f['size'],
                              'ranges': [[frames[a]['frame_id'], frames[b]['frame_id']] for a, b in f['ranges']]} for f in frags],
            'shared_frames': shared, 'breaks': breaks}


# ----------------------------------------------------------------------------------------------------- similarity chaining

def _centres(rec):
    out = {}
    for image in rec.images.values():
        try:
            if not image.has_pose:
                continue
        except AttributeError:
            pass
        pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        r = np.asarray(pose.rotation.matrix(), float)
        out[image.name] = -r.T @ np.asarray(pose.translation, float)
    return out


def umeyama(src, tgt):
    """Similarity (s, R, t) minimising |tgt - (s R src + t)|^2 (Umeyama 1991). src/tgt (n,3), n >= 3."""
    src, tgt = np.asarray(src, float), np.asarray(tgt, float)
    ms, mt = src.mean(0), tgt.mean(0)
    a, b = src - ms, tgt - mt
    cov = b.T @ a / len(src)
    u, d, vt = np.linalg.svd(cov)
    sgn = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sgn[2, 2] = -1
    rot = u @ sgn @ vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(d) @ sgn) / var) if var > 1e-18 else 1.0
    return s, rot, mt - s * rot @ ms


def estimate_link(src_c, tgt_c, *, min_shared=CHAIN_MIN_SHARED, max_residual=CHAIN_MAX_RESIDUAL, iters=200, seed=0):
    """Similarity tgt_from_src from shared camera centres ({name: xyz} each). Returns (sim or None, evidence dict).

    RANSAC over 3-centre samples (inlier = centre error <= max_residual x target camera extent), Umeyama on the inliers; accepted
    when >= min_shared shared frames, inlier share >= CHAIN_MIN_INLIER_FRAC and the RMS of the inliers <= max_residual x extent."""
    shared = sorted(set(src_c) & set(tgt_c))
    ev = {'shared_frames': len(shared)}
    if len(shared) < min_shared:
        ev['decision'] = f'rejected: {len(shared)} shared registered frame(s), need {min_shared}'
        return None, ev
    s_xyz = np.array([src_c[n] for n in shared])
    t_xyz = np.array([tgt_c[n] for n in shared])
    allc = np.array(list(tgt_c.values()))
    extent = float(np.linalg.norm(np.percentile(allc, 95, 0) - np.percentile(allc, 5, 0))) or 1.0
    thr = max_residual * extent
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(shared), 3, replace=False)
        try:
            s, r, t = umeyama(s_xyz[idx], t_xyz[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm((s * (r @ s_xyz.T)).T + t - t_xyz, axis=1)
        inl = err <= thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        ev['decision'] = 'rejected: no consistent similarity'
        return None, ev
    s, r, t = umeyama(s_xyz[best], t_xyz[best])
    err = np.linalg.norm((s * (r @ s_xyz.T)).T + t - t_xyz, axis=1)
    inl = err <= thr
    rms = float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float('inf')
    ev.update(inliers=int(inl.sum()), inlier_fraction=round(float(inl.mean()), 3), rms_rel_extent=round(rms / extent, 5),
              scale=round(s, 5))
    if inl.sum() < min_shared or inl.mean() < CHAIN_MIN_INLIER_FRAC or rms > thr:
        ev['decision'] = 'rejected: shared frames disagree on the similarity'
        return None, ev
    ev['decision'] = 'accepted'
    return (s, r, t), ev


def _registered(rec):
    out = {}
    for image in rec.images.values():
        try:
            if not image.has_pose:
                continue
        except AttributeError:
            pass
        out[image.name] = image
    return out


def point_correspondences(rec_a, rec_b, verified, db_ids, database, *, min_pair_inliers=15, max_pairs=400):
    """3D-3D correspondences (xyz in B, xyz in A) between two models: the same keypoint of a frame registered in both, and the
    verified inlier matches of frame pairs (one frame registered in A, the other in B) where both keypoints are triangulated.
    Returns (src (n,3), tgt (n,3), pair_ids (n,), pairs_used [(name_a, name_b)])."""
    import pycolmap
    A, B = _registered(rec_a), _registered(rec_b)
    pa, pb = rec_a.points3D, rec_b.points3D
    corr = {}

    def add(img_a, ka, img_b, kb, pair):
        p, q = img_a.points2D[ka], img_b.points2D[kb]
        if p.has_point3D() and q.has_point3D():
            corr.setdefault((q.point3D_id, p.point3D_id), pair)

    pairs = []
    for name in sorted(set(A) & set(B)):
        pairs.append((name, name))
        for k in range(min(len(A[name].points2D), len(B[name].points2D))):
            add(A[name], k, B[name], k, len(pairs) - 1)
    cand = []
    for (x, y), n in verified.items():
        if n < min_pair_inliers:
            continue
        for na, nb in ((x, y), (y, x)):
            if na != nb and na in A and nb in B and not (na in B and nb in A):
                cand.append((n, na, nb))
    if cand:
        db = pycolmap.Database.open(str(database))
        try:
            for n, na, nb in sorted(cand, reverse=True)[:max_pairs]:
                m = np.asarray(db.read_two_view_geometry(db_ids[na], db_ids[nb]).inlier_matches).reshape(-1, 2)
                pairs.append((na, nb))
                for ka, kb in m:
                    if ka < len(A[na].points2D) and kb < len(B[nb].points2D):
                        add(A[na], int(ka), B[nb], int(kb), len(pairs) - 1)
        finally:
            db.close()
    keys = [k for k in corr if k[0] in pb and k[1] in pa]
    src = np.array([pb[k[0]].xyz for k in keys], float).reshape(-1, 3)
    tgt = np.array([pa[k[1]].xyz for k in keys], float).reshape(-1, 3)
    return src, tgt, np.array([corr[k] for k in keys], int), pairs


def robust_similarity(src, tgt, threshold, *, iters=500, seed=0):
    """RANSAC (3-point Umeyama) + refit on the inliers: (sim, inlier mask) or (None, None)."""
    n = len(src)
    if n < 3:
        return None, None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, r, t = umeyama(src[idx], tgt[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(s) or s <= 0:
            continue
        inl = np.linalg.norm(s * src @ r.T + t - tgt, axis=1) <= threshold
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        return None, None
    for _ in range(3):
        s, r, t = umeyama(src[best], tgt[best])
        best = np.linalg.norm(s * src @ r.T + t - tgt, axis=1) <= threshold
        if best.sum() < 3:
            return None, None
    return (s, r, t), best


def _extent(xyz):
    if len(xyz) < 2:
        return 1.0
    return float(np.linalg.norm(np.percentile(xyz, 95, 0) - np.percentile(xyz, 5, 0))) or 1.0


def _all_points(rec):
    return np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= 3], float).reshape(-1, 3)


def compose(outer, inner):
    """outer o inner for similarities (s, R, t): x -> s1 R1 (s2 R2 x + t2) + t1."""
    s1, r1, t1 = outer
    s2, r2, t2 = inner
    return s1 * s2, r1 @ r2, s1 * r1 @ t2 + t1


def link_models(rec_a, rec_b, verified, db_ids, database):
    """Similarity A_from_B from 3D-3D correspondences + its evidence. Accepted when >= LINK_MIN_INLIERS correspondences agree
    within LINK_THRESHOLD x A's point extent, they are >= LINK_MIN_INLIER_FRAC of all, and come from >= 2 frame pairs."""
    src, tgt, pid, pairs = point_correspondences(rec_a, rec_b, verified, db_ids, database)
    ev = {'correspondences': int(len(src)), 'frame_pairs': len(pairs), 'shared_frames': sum(a == b for a, b in pairs)}
    if len(src) < LINK_MIN_INLIERS:
        ev['decision'] = f'rejected: {len(src)} 3D-3D correspondences (need {LINK_MIN_INLIERS})'
        return None, ev
    ext = _extent(_all_points(rec_a))
    sim, inl = robust_similarity(src, tgt, LINK_THRESHOLD * ext)
    if sim is None:
        ev['decision'] = 'rejected: no consistent similarity'
        return None, ev
    err = np.linalg.norm(sim[0] * src @ sim[1].T + sim[2] - tgt, axis=1)
    used_pairs = sorted(set(pid[inl].tolist()))
    ev.update(inliers=int(inl.sum()), inlier_fraction=round(float(inl.mean()), 3),
              rms_rel_extent=round(float(np.sqrt(np.mean(err[inl] ** 2))) / ext, 5), scale=round(float(sim[0]), 5),
              inlier_frame_pairs=len(used_pairs), example_pairs=[list(pairs[i]) for i in used_pairs[:3]])
    if inl.sum() < LINK_MIN_INLIERS or inl.mean() < LINK_MIN_INLIER_FRAC or len(used_pairs) < 2:
        ev['decision'] = 'rejected: too few / too inconsistent correspondences'
        return None, ev
    ev['decision'] = 'accepted'
    return sim, ev


def _up(rights):
    """World up from (n,3) camera x axes (most perpendicular direction) and its conditioning (2nd / 3rd singular value)."""
    if len(rights) < 3:
        return None, 0.0
    sv = np.linalg.svd(rights, full_matrices=False)
    return sv[2][2], float(sv[1][1] / max(sv[1][2], 1e-9))


IDENTITY = (1.0, np.eye(3), np.zeros(3))


class ShelfCheck:
    """Shelf-plane check: does a joined part keep the shelf ONE flat face? A part = a model (or the frames of one first-pass fragment inside a model) mapped
    into the main frame by its similarity. Its own shelf-face plane must be within PLANE_MAX_ANGLE_DEG of the reference plane (the face of
    everything accepted so far), its book positions (ray-triangulated, else projected on its own face) within PLANE_MAX_OFFSET_SPINES spine
    heights of it, and its cameras on the room side. `books`: recon3d.load_groups book groups; `names_to_frame`: image name -> frame id."""

    def __init__(self, books, names_to_frame, *, max_angle_deg=PLANE_MAX_ANGLE_DEG, max_offset_spines=PLANE_MAX_OFFSET_SPINES,
                 min_books=PLANE_MIN_BOOKS, camera_side_min=CAMERA_SIDE_MIN):
        self.books, self.names_to_frame = books, dict(names_to_frame)
        self.max_angle_deg, self.max_offset_spines, self.min_books, self.camera_side_min = max_angle_deg, max_offset_spines, min_books, camera_side_min
        self.by_frame = {}
        for g in books:
            for s in g['sightings']:
                self.by_frame.setdefault(s['frame_id'], []).append(s)

    def unit(self, rec, sim=IDENTITY, keep=None, label=None):
        """The part's cameras, points, face plane, book positions and spine lengths, all in the main frame (cached per model, transform, frames)."""
        s, rot, t = sim
        key = (id(rec), float(s), np.asarray(rot, float).tobytes(), np.asarray(t, float).tobytes(), frozenset(keep) if keep is not None else None)
        cache = self.__dict__.setdefault('_cache', {})
        if key not in cache:
            cache[key] = self._unit(rec, sim, keep)
        return {**cache[key], 'label': label}

    def _unit(self, rec, sim, keep):
        from pipeline import recon3d as R
        s, rot, t = sim
        views = {}
        for image in _registered(rec).values():
            fid = self.names_to_frame.get(image.name)
            if fid is None or (keep is not None and image.name not in keep):
                continue
            v = R._view_of(image, rec.cameras[image.camera_id], fid)
            views[fid] = R.View(fid, s * rot @ v.centre + t, v.rotation @ rot.T, v.unproject)
        names = keep if keep is not None else [im.name for im in rec.images.values()]
        pts = _points_seen(rec, names)
        pts = s * pts @ rot.T + t if len(pts) else pts.reshape(-1, 3)
        u = {'frames': len(views), 'views': views, 'points': pts,
             'centres': np.array([v.centre for v in views.values()]).reshape(-1, 3),
             'forwards': np.array([v.rotation.T @ np.array([0.0, 0.0, 1.0]) for v in views.values()]).reshape(-1, 3)}
        u['plane'] = shelf_face(pts, u['forwards'], u['centres'])
        mine = [g for g in self.books if any(x['frame_id'] in views for x in g['sightings'])]
        if u['plane'] is None:
            u['books'], u['spines'] = [], []
        else:
            n, p = u['plane']['normal'], u['plane']['point']
            u['books'] = R.book_positions(mine, views, n, p)
            u['spines'] = R.spine_lengths(views, [x for f in views for x in self.by_frame.get(f, [])], n, p, limit=PART_SPINE_LIMIT)
        return u

    def reference(self, units):
        """The shelf face of everything accepted so far (points pooled) and its spine height (median over the units' polygons)."""
        pts = np.concatenate([u['points'] for u in units if len(u['points'])] or [np.zeros((0, 3))])
        fw = np.concatenate([u['forwards'] for u in units] or [np.zeros((0, 3))])
        cs = np.concatenate([u['centres'] for u in units] or [np.zeros((0, 3))])
        plane = shelf_face(pts, fw, cs)
        spines = [x for u in units for x in u['spines']]
        return {'plane': plane, 'spine': float(np.median(spines)) if spines else None, 'units': len(units)}

    def test(self, unit, ref, up=None):
        """(accepted, evidence) for one part against the reference."""
        ev = {'frames': unit['frames'], 'books': len(unit['books'])}
        rp = ref['plane']
        if rp is None or not ref['spine']:
            ev['reason'] = 'the reference (main model) has no shelf-face plane or no spine polygon to check against'
            return False, ev
        if unit['plane'] is None:
            ev['reason'] = 'the part has no shelf-face plane (too few points, or no plane facing its cameras)'
            return False, ev
        n, rn = unit['plane']['normal'], rp['normal']
        ev['angle_deg'] = round(_angle(n, rn), 1)
        if up is not None:                              # rotation about the vertical: the normals projected on the horizontal plane
            a, b = n - (n @ up) * up, rn - (rn @ up) * up
            if np.linalg.norm(a) > 1e-6 and np.linalg.norm(b) > 1e-6:
                ev['yaw_deg'] = round(_angle(a / np.linalg.norm(a), b / np.linalg.norm(b)), 1)
        side = (unit['centres'] - rp['point']) @ rn
        ev['cameras_room_side'] = round(float((side > 0).mean()), 3) if len(side) else None
        if unit['books']:
            d = np.abs((np.array([x for _, x, _ in unit['books']]) - rp['point']) @ rn) / ref['spine']
            ev.update(median_offset_spines=round(float(np.median(d)), 3), books_within=round(float((d <= self.max_offset_spines).mean()), 3),
                      books_triangulated=sum(m == 'triangulated' for _, _, m in unit['books']))
        if len(unit['books']) < self.min_books:
            ev['reason'] = f"{len(unit['books'])} book position(s) in the part (need {self.min_books}): it cannot be checked against the shelf plane"
        elif ev['angle_deg'] > self.max_angle_deg:
            ev['reason'] = (f"its shelf face is {ev['angle_deg']} deg off the main shelf plane"
                            + (f" ({ev['yaw_deg']} deg about the vertical)" if 'yaw_deg' in ev else '') + f" (max {self.max_angle_deg})")
        elif ev['cameras_room_side'] is not None and ev['cameras_room_side'] < self.camera_side_min:
            ev['reason'] = f"only {ev['cameras_room_side']:.0%} of its cameras are on the room side of the shelf plane (min {self.camera_side_min:.0%})"
        elif ev['median_offset_spines'] > self.max_offset_spines:
            ev['reason'] = f"its books lie {ev['median_offset_spines']} spine heights off the main shelf plane (median; max {self.max_offset_spines})"
        if 'reason' in ev:
            return False, ev
        ev['reason'] = 'on the main shelf plane'
        return True, ev


def _level_up(rec, names):
    """World up of the level horizon of these cameras (_up), signed like their image-up axes; None when they do not pan enough to define it."""
    up, cond = _up(_rights(rec, names))
    if up is None or cond < HORIZON_MIN_CONDITIONING:
        return None
    ups = [u for u, _ in _axes(rec, names).values()]
    return -up if ups and float(np.median(np.array(ups) @ up)) < 0 else up


def shelf_face(points, forwards, centres):
    """The dominant plane of `points` that the cameras look at face-on (|normal . mean viewing direction| >= FACE_MIN_COS), normal towards the
    cameras; tries up to FACE_TRIES planes (each without the previous inliers), None when none faces them."""
    from pipeline.recon3d import fit_plane, orient_normal
    pts = np.asarray(points, float).reshape(-1, 3)
    view = _mean_dir(np.asarray(forwards, float).reshape(-1, 3)) if len(forwards) else None
    for _ in range(FACE_TRIES):
        pl = fit_plane(pts)
        if pl is None:
            return None
        if view is None or abs(float(pl['normal'] @ view)) >= FACE_MIN_COS:
            pl['normal'] = orient_normal(pl['normal'], pl['point'], centres)
            return pl
        pts = pts[~pl['inlier_mask']]
    return None


def split_check(rec, frags, names, shelf, up=None):
    """Shelf-plane check, after the re-map: COLMAP's jointly adjusted model can itself glue first-pass fragments at wrong rotations. Its frames are split
    by the first-pass fragment that registered them (a frame no fragment had goes with its nearest owned frame in time); the part holding most
    of the first-pass main fragment is the anchor; every other part, largest first, must pass ShelfCheck.test against the parts kept so far.
    Returns (keep: image names of the kept parts, anchor units, evidence rows); keep is the whole model when every part passes."""
    index = {n: i for i, n in enumerate(names)}
    owner = {}
    for f in frags:                                     # largest fragment first: it owns a frame registered twice
        for i in f['frames']:
            owner.setdefault(i, f['model'])
    regs = [n for n in _reg_names(rec) if n in index]
    owned = sorted(owner)
    parts = {}
    for n in regs:
        i = index[n]
        k = owner.get(i)
        if k is None and owned:
            j = int(np.searchsorted(owned, i))
            near = [owned[x] for x in (j - 1, j) if 0 <= x < len(owned)]
            k = owner[min(near, key=lambda o: abs(o - i))]
        parts.setdefault(k, set()).add(n)
    main_frag = frags[0]['model'] if frags else None
    order = sorted(parts, key=lambda k: (-len(parts[k]), k != main_frag, -1 if k is None else k))
    anchor, ref, units = None, None, []
    for k in order:                                     # the largest part that has a shelf face and spines to check the others against
        units = [shelf.unit(rec, IDENTITY, parts[k], label=f'fragment {k}')]
        ref = shelf.reference(units)
        if ref['plane'] is not None and ref['spine']:
            anchor = k
            break
    if anchor is None:
        return set(), [], [{'fragment': None, 'frames': len(regs), 'decision': 'dropped: no part of the model has a shelf face with book spines'}]
    order = [k for k in order if k != anchor]
    if up is None:
        up = _level_up(rec, parts[anchor])
    rows = [{'fragment': int(anchor) if anchor is not None else None, 'frames': len(parts[anchor]),
             'decision': 'anchor (the largest part with a shelf face' + (', holds the first-pass main fragment)' if anchor == main_frag else ')')}]
    keep = set(parts[anchor])
    for k in order:
        u = shelf.unit(rec, IDENTITY, parts[k], label=f'fragment {k}')
        ok, ev = shelf.test(u, ref, up)
        rows.append({'fragment': int(k) if k is not None else None, **ev, 'decision': 'kept' if ok else 'dropped: ' + ev['reason']})
        if ok:
            units.append(u)
            keep |= parts[k]
            ref = shelf.reference(units)
    return (None if len(keep) == len(regs) else keep), units, rows


def final_check(models, members, shelf):
    """Shelf-plane check: the chosen join re-checked as a whole: every joined member (not the main one) against the shelf face of all the OTHER members
    (leave one out); the member failing worst (largest book offset) is dropped and the rest re-checked, until all agree: the largest
    consistent sub-join. Returns (members kept, evidence rows)."""
    mem, rows = list(members), []
    while len(mem) > 1:
        units = [shelf.unit(models[k], sim, keep, label=f'model {k}') for k, sim, keep in mem]
        fails = []
        for i in range(1, len(mem)):
            ok, ev = shelf.test(units[i], shelf.reference([u for j, u in enumerate(units) if j != i]))
            if not ok:
                fails.append((ev.get('median_offset_spines') or 0.0, ev.get('angle_deg') or 0.0, i, ev))
        if not fails:
            break
        _, _, i, ev = max(fails)
        rows.append({'model': int(mem[i][0]), **ev, 'decision': 'dropped from the join: ' + ev['reason']})
        mem.pop(i)
    rows.extend({'model': int(k), 'decision': 'kept' if n else 'main model'} for n, (k, _, _) in enumerate(mem))
    return mem, rows


def refused_joins(info):
    """Every join refused by a geometric check (horizon, shelf plane, re-mapped part, final check) with its reason: [{route, what, reason}]."""
    out = []
    for route, c in (info.get('chain') or {}).items():
        for ev in c.get('links') or []:
            d = ev.get('decision') or ''
            if d.startswith('rejected: joined model tilts') or 'plane_check' in ev and d.startswith('rejected'):
                out.append({'route': route, 'what': f"model {ev.get('b')} <-> {ev.get('a')} (similarity, {ev.get('inliers')} agreeing 3D correspondences)",
                            'reason': d[len('rejected: '):]})
        for row in c.get('final_check') or []:
            if row['decision'].startswith('dropped'):
                out.append({'route': route, 'what': f"model {row['model']} (final check)", 'reason': row['decision']})
    for row in ((info.get('remap') or {}).get('main_model_check') or {}).get('parts') or []:
        if row['decision'].startswith('dropped'):
            out.append({'route': 'remapped', 'what': f"first-pass fragment {row['fragment']} inside the re-mapped main model ({row['frames']} frames)",
                        'reason': row['decision'][len('dropped: '):]})
    return out


def chain_models(models, database=None, names=None, *, max_tilt_deg=None, shelf=None, anchor_units=None, main_keep=None):
    """Join models into the largest by similarities from 3D-3D correspondences (link_models), strongest accepted link first;
    a joined model must also keep the level horizon of the main model (median tilt of its camera x axes <= max_tilt_deg) and, with a
    ShelfCheck (`shelf`), keep every book on the main shelf plane (ShelfCheck.test against the parts joined so far; `anchor_units` /
    `main_keep`: the kept parts / image names of a main model that split_check trimmed).

    Returns (members, links): members = [(model_key, (s, R, t) into the main frame, keep)], main first with the identity (keep = the image
    names used of that model, None = all); links = one evidence row per attempted model pair."""
    max_tilt_deg = HORIZON_MAX_TILT_DEG if max_tilt_deg is None else max_tilt_deg
    keys = sorted(models, key=lambda k: -len(_reg_names(models[k])))
    if not keys:
        return [], []
    ident = IDENTITY
    if len(keys) == 1 or database is None or (main_keep is not None and not main_keep):   # (nothing of the main model kept: nothing to join to)
        return [(keys[0], ident, main_keep)], []
    import pycolmap
    db = pycolmap.Database.open(str(database))
    db_ids = {im.name: im.image_id for im in db.read_all_images()}
    db.close()
    verified = {(names[i], names[j]): n for (i, j), n in _verified(database, names).items()}
    regs = {k: set(_reg_names(models[k])) for k in keys}
    edges = []
    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            a, b = keys[x], keys[y]
            has = bool(regs[a] & regs[b]) or any(n >= 15 and ((p in regs[a] and q in regs[b]) or (q in regs[a] and p in regs[b]))
                                                 for (p, q), n in verified.items())
            if not has:
                continue
            sim_ab, ev = link_models(models[a], models[b], verified, db_ids, database)
            ev.update(a=int(a), b=int(b))
            edges.append((a, b, sim_ab, ev))
    up, cond = _up(_rights(models[keys[0]], main_keep if main_keep is not None else regs[keys[0]]))
    level_up = _level_up(models[keys[0]], main_keep if main_keep is not None else regs[keys[0]])
    ref_units = ref = None
    if shelf is not None:
        ref_units = list(anchor_units) if anchor_units else [shelf.unit(models[keys[0]], ident, main_keep, label=f'model {keys[0]}')]
        ref = shelf.reference(ref_units)
    joined = {keys[0]: ident}
    while True:
        best = None
        for a, b, sim_ab, ev in edges:
            if sim_ab is None or ev.get('decision') != 'accepted':
                continue
            if a in joined and b not in joined:
                cand = (b, compose(joined[a], sim_ab), ev)
            elif b in joined and a not in joined:
                s, r, t = sim_ab
                inv = (1.0 / s, r.T, -(r.T @ t) / s)
                cand = (a, compose(joined[b], inv), ev)
            else:
                continue
            if best is None or ev['inliers'] > best[2]['inliers']:
                best = cand
        if best is None:
            break
        k, sim, ev = best
        rights = _rights(models[k], regs[k]) @ sim[1].T
        if up is not None and cond >= HORIZON_MIN_CONDITIONING and len(rights):
            tilt = float(np.median(np.degrees(np.arcsin(np.clip(rights @ up, -1, 1)))))
            ev['horizon_tilt_deg'] = round(tilt, 1)
            if abs(tilt) > max_tilt_deg:
                ev['decision'] = f'rejected: joined model tilts the level horizon by {tilt:.1f} deg (max {max_tilt_deg})'
                continue
        else:
            ev['horizon_tilt_deg'] = None
            ev['horizon_note'] = 'main model cameras do not pan enough to fix the horizontal plane; horizon check skipped'
        if shelf is not None:
            unit = shelf.unit(models[k], sim, label=f'model {k}')
            ok, pev = shelf.test(unit, ref, level_up)
            ev['plane_check'] = pev
            if not ok:
                ev['decision'] = 'rejected: ' + pev['reason']
                continue
            ref_units.append(unit)
            ref = shelf.reference(ref_units)
        joined[k] = sim
        ev['joined'] = True
    links = [ev for _, _, _, ev in edges]
    return [(k, joined[k], main_keep if k == keys[0] else None) for k in keys if k in joined], links


def agreement(rec, fragments, *, max_residual=CHAIN_MAX_RESIDUAL):
    """Does a merged model reproduce the fragments the incremental mapper built independently? Per fragment: shared registered
    frames, and the similarity fragment -> merged estimated from the shared camera centres (estimate_link) with its RMS residual
    relative to the merged model's camera extent. A fragment the merged model contradicts (residual above the bound or the
    shared frames inconsistent) is flagged; a merge that contradicts any fragment of >= 20 frames is not trusted."""
    target = _centres(rec)
    out = []
    for key, frag in fragments.items():
        src = _centres(frag)
        sim, ev = estimate_link(src, target, max_residual=max_residual)
        out.append({'fragment': int(key), 'registered': len(src), 'shared_frames': ev['shared_frames'],
                    'agrees': sim is not None, 'rms_rel_extent': ev.get('rms_rel_extent'), 'inlier_fraction': ev.get('inlier_fraction'),
                    'decision': ev['decision']})
    return sorted(out, key=lambda e: -e['registered'])


def _points_seen(rec, image_names, max_error=2.0):
    names = set(image_names)
    ids = set()
    for image in rec.images.values():
        if image.name in names:
            try:
                if not image.has_pose:
                    continue
            except AttributeError:
                pass
            ids.update(p.point3D_id for p in image.points2D if p.has_point3D())
    pts = rec.points3D
    return np.array([pts[i].xyz for i in ids if i in pts and pts[i].track.length() >= 3 and pts[i].error <= max_error], float).reshape(-1, 3)


def plane_coherence(rec, fragments, min_points=200):
    """The shelf face must be ONE plane in a correctly merged model: the dominant plane of the whole cloud vs the points each
    original fragment's frames see. Per fragment: angle between its own dominant plane and the global one, and the median
    distance of its points to the global plane relative to the global plane's inlier threshold (1.5% of the scene extent)."""
    from pipeline.recon3d import fit_plane
    allp = _points_seen(rec, [im.name for im in rec.images.values()])
    g = fit_plane(allp)
    if g is None:
        return {'global_plane': None, 'fragments': []}
    out = []
    for key, frag in fragments.items():
        p = _points_seen(rec, _reg_names(frag))
        if len(p) < min_points:
            continue
        own = fit_plane(p)
        if own is None:
            continue
        angle = float(np.degrees(np.arccos(min(1.0, abs(float(own['normal'] @ g['normal']))))))
        d = np.abs((p - g['point']) @ g['normal'])
        out.append({'fragment': int(key), 'points': int(len(p)), 'angle_deg': round(angle, 1),
                    'median_dist_over_threshold': round(float(np.median(d)) / g['threshold'], 2),
                    'on_global_plane_fraction': round(float((d <= g['threshold']).mean()), 3)})
    return {'global_plane': {'inlier_ratio': round(g['inlier_ratio'], 3), 'points': int(len(allp))},
            'fragments': sorted(out, key=lambda e: -e['points'])}


def _axes(rec, names=None):
    """{name: (up, forward)} world directions of the image's up axis and optical axis of registered images."""
    out = {}
    want = set(names) if names is not None else None
    for image in rec.images.values():
        if want is not None and image.name not in want:
            continue
        try:
            if not image.has_pose:
                continue
        except AttributeError:
            pass
        pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        r = np.asarray(pose.rotation.matrix(), float)
        out[image.name] = (-r.T @ np.array([0.0, 1.0, 0.0]), r.T @ np.array([0.0, 0.0, 1.0]))
    return out


def _mean_dir(vs):
    m = np.sum(vs, axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 1e-12 else None


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def orientation_coherence(rec, fragments):
    """A handheld sweep keeps the phone upright and facing the shelf, so in a correctly joined model every original fragment's
    cameras share the same mean up and viewing direction (within the pans). A fragment glued in at a wrong rotation shows as a
    large angle. Per fragment: angle of its mean camera-up / mean optical axis to the largest fragment's, and its own spread."""
    rows, ref = [], None
    for key, frag in sorted(fragments.items(), key=lambda kv: -len(_reg_names(kv[1]))):
        ax = _axes(rec, _reg_names(frag))
        if len(ax) < 3:
            continue
        up = _mean_dir([u for u, _ in ax.values()])
        fw = _mean_dir([f for _, f in ax.values()])
        if up is None or fw is None:
            continue
        spread = float(np.median([_angle(u, up) for u, _ in ax.values()]))
        if ref is None:
            ref = (up, fw)
        rows.append({'fragment': int(key), 'frames': len(ax), 'up_deg': round(_angle(up, ref[0]), 1),
                     'view_deg': round(_angle(fw, ref[1]), 1), 'own_up_spread_deg': round(spread, 1)})
    return rows


def _rights(rec, names):
    want = set(names)
    out = []
    for image in rec.images.values():
        if image.name not in want:
            continue
        try:
            if not image.has_pose:
                continue
        except AttributeError:
            pass
        pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        out.append(np.asarray(pose.rotation.matrix(), float).T @ np.array([1.0, 0.0, 0.0]))
    return np.array(out).reshape(-1, 3)


def horizon_coherence(rec, fragments, reference=None):
    """Level-horizon test, independent of pan (yaw) and tilt (pitch): a handheld phone is held with little roll, so the image's
    x axis of every camera is (nearly) horizontal, i.e. perpendicular to ONE world up direction. World up = the direction most
    perpendicular to the x axes of the reference cameras (largest fragment, or `reference` names); per fragment the median
    tilt of its x axes out of that horizontal plane. A fragment joined at a wrong rotation tilts its horizon systematically."""
    frs = sorted(fragments.items(), key=lambda kv: -len(_reg_names(kv[1])))
    if not frs:
        return {'up': None, 'fragments': []}
    ref = _rights(rec, reference if reference is not None else _reg_names(frs[0][1]))
    if len(ref) < 3:
        return {'up': None, 'fragments': []}
    up, cond = _up(ref)
    rows = []
    for key, frag in frs:
        r = _rights(rec, _reg_names(frag))
        if len(r) < 3:
            continue
        tilt = np.degrees(np.arcsin(np.clip(r @ up, -1, 1)))
        m = _mean_dir(r)
        rows.append({'fragment': int(key), 'frames': len(r), 'median_tilt_deg': round(float(np.median(tilt)), 1),
                     'p90_abs_tilt_deg': round(float(np.percentile(np.abs(tilt), 90)), 1),
                     'own_pan_p90_deg': round(float(np.percentile([_angle(v, m) for v in r], 90)), 1) if m is not None else None})
    return {'up': [round(float(v), 4) for v in up], 'conditioning': round(cond, 1), 'fragments': rows}


def trajectory_jumps(rec, names, factor=8.0):
    """Consecutive registered frames (by run index) whose camera centres are more than `factor` x the median step apart:
    a handheld camera moves continuously, so a big jump between neighbours is a wrong link. Returns (median_step, jumps)."""
    index = {n: i for i, n in enumerate(names)}
    c = sorted((index[n], xyz) for n, xyz in _centres(rec).items() if n in index)
    steps = [(c[k][0], c[k + 1][0], float(np.linalg.norm(c[k + 1][1] - c[k][1])) / max(1, c[k + 1][0] - c[k][0])) for k in range(len(c) - 1)]
    if not steps:
        return 0.0, []
    med = float(np.median([s for _, _, s in steps])) or 1e-12
    return med, [{'from': names[a], 'to': names[b], 'step_over_median': round(s / med, 1)} for a, b, s in steps if s > factor * med and b - a <= 3]


# ----------------------------------------------------------------------------------------------------- the build step

def newest_panorama(source):
    """Newest <run>_panorama_vN next to a run that holds a panorama_count.json (its verified piece links are hints), else None."""
    source = Path(source)
    found = []
    for p in source.parent.glob(f'{source.name}_panorama_v*'):
        try:
            n = int(p.name.rsplit('_v', 1)[1])
        except ValueError:
            continue
        if (p / 'panorama_count.json').is_file():
            found.append((n, p))
    return max(found)[1] if found else None


def joined_centres(models, members):
    """{name: centre in the main frame} of every frame of the joined members (the main model wins a shared frame)."""
    out = {}
    for m in members:
        k, (s, r, t), keep = m[0], m[1], (m[2] if len(m) > 2 else None)
        for name, c in _centres(models[k]).items():
            if keep is None or name in keep:
                out.setdefault(name, s * r @ c + t)
    return out


def _union(models, members):
    names = set()
    for m in members:
        keep = m[2] if len(m) > 2 else None
        names.update(n for n in _reg_names(models[m[0]]) if keep is None or n in keep)
    return names


def _unmatched(database, pairs, names):
    """The pairs not matched yet in the database (a merge pass reuses the bridge matches of the first build)."""
    if not pairs:
        return []
    import pycolmap
    db = pycolmap.Database.open(str(database))
    try:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        return [(i, j) for i, j in pairs if not db.exists_two_view_geometry(ids[names[i]], ids[names[j]])]
    finally:
        db.close()


def merge(database, image_dir, work, models, names, mapping_options, *, overlap=12, panorama_dir=None, progress=None,
          plan=None, sharpness=None, total_frames=None, shelf=None):
    """After the first incremental mapping: bridge pairs -> match -> chain the original fragments by similarities -> (only when that
    joins less than REMAP_SKIP_SHARE of the run) map again and chain; keep whichever of the original / re-mapped model sets gives the
    larger joined model. `plan` (MergePlan, default inline) decides whether the bridge matching and the re-map run in this build or
    are deferred to the merge pass (`info['deferred']` lists them); `sharpness` (per name) switches to the cheaper CHEAP_RULE pairs.
    `shelf` (ShelfCheck): every join, and every first-pass fragment inside the re-mapped main model (split_check), must keep the books on
    the main shelf plane; the chosen join is re-checked part by part at the end (final_check) and shrinks to its consistent sub-join.
    Returns (members, info): members = [(Reconstruction, (s, R, t), keep)] main model first (identity; keep = image names used, None = all),
    info = the `merged_from` block."""
    import pycolmap
    plan = plan or MergePlan()
    t_all = time.perf_counter()
    total_frames = total_frames or len(names)
    frags = fragments_from_models(models, names)
    info = {'fragments_before': len(frags), 'fragment_sizes_before': [f['size'] for f in frags],
            'fragments': [{'model': f['model'], 'registered': f['size'],
                           'frame_ranges': [[names[a], names[b]] for a, b in f['ranges']]} for f in frags]}
    hints = panorama_hints(panorama_dir, names) if panorama_dir else []
    info['panorama_hints'] = {'panorama': Path(panorama_dir).name if panorama_dir else None, 'piece_links': len(hints)}
    candidates, deferred, bridged, links, secs = {'original': models}, [], False, [], {}
    if len(frags) > 1:
        rule = dict(CHEAP_RULE, sharpness=sharpness) if sharpness is not None else {}
        pairs, why = bridge_pairs(frags, len(names), overlap=overlap, hints=hints, **rule)
        todo = _unmatched(database, pairs, names)
        rule_info = {k: v for k, v in rule.items() if k != 'sharpness'} or 'full default'
        if plan.allow('bridge_matching', len(todo) * plan.seconds_per_pair):
            if progress:
                progress('merging', step='matching bridge pairs', fragments=len(frags))
            t0 = time.perf_counter()
            info['bridge_pairs'] = {'rule': rule_info, 'reasons': why, 'already_matched': len(pairs) - len(todo),
                                    **match_pairs(database, todo, names, Path(work) / 'bridge_pairs.txt')}
            secs['bridge_matching_s'] = time.perf_counter() - t0
            bridged = True
        else:
            deferred.append('bridge_matching')
            info['bridge_pairs'] = {'rule': rule_info, 'reasons': why, 'pairs_planned': len(todo), 'deferred': True}
        links = cross_links(frags, database, names)
        info['cross_fragment_links'] = links[:40]
    t0 = time.perf_counter()
    members_o, chain_o = chain_models(models, database, names, shelf=shelf)
    secs['chain_original_s'] = time.perf_counter() - t0
    joined = {'original': (models, members_o)}
    info['chain'] = {'original': {'joined_models': len(members_o), 'registered': len(_union(models, members_o)), 'links': chain_o}}
    share = info['chain']['original']['registered'] / total_frames
    if len(frags) > 1 and not bridged:
        deferred.append('re-map (when the joined fragments stay under %d%%)' % round(100 * REMAP_SKIP_SHARE))
    elif len(frags) > 1 and not links:
        info['remap'] = {'skipped': 'no verified pair with >= %d inliers links two fragments' % MIN_LINK_INLIERS}
    elif len(frags) > 1 and sharpness is not None and share >= REMAP_SKIP_SHARE:
        info['remap'] = {'skipped': f'the similarity-joined fragments already register {share:.1%} of the run (>= {REMAP_SKIP_SHARE:.0%})'}
    elif len(frags) > 1 and not plan.allow('remap', plan.remap_factor * plan.mapping_s):
        deferred.append('re-map')
        info['remap'] = {'deferred': True, 'joined_share_before': round(share, 4)}
    elif len(frags) > 1:
        if progress:
            progress('merging', step='re-mapping with the bridge pairs', links=len(links))
        out = Path(work) / 'sparse_merge'
        out.mkdir(exist_ok=True)
        t0 = time.perf_counter()
        remapped = pycolmap.incremental_mapping(database, image_dir, out, mapping_options)
        secs['remap_s'] = time.perf_counter() - t0
        info['remap'] = {'models': len(remapped), 'sizes': sorted((len(_reg_names(r)) for r in remapped.values()), reverse=True),
                         'seconds': round(secs['remap_s'], 1)}
        if remapped:
            candidates['remapped'] = remapped
            t0 = time.perf_counter()
            split = {}
            if shelf is not None:      # the re-mapped main model is checked part by part (first-pass fragments) before anything joins it
                main_r = max(remapped, key=lambda k: len(_reg_names(remapped[k])))
                keep, units, rows = split_check(remapped[main_r], frags, names, shelf)
                split = {'anchor_units': units, 'main_keep': keep}
                info['remap']['main_model_check'] = {'model': int(main_r), 'registered': len(_reg_names(remapped[main_r])),
                                                     'kept': len(keep) if keep is not None else len(_reg_names(remapped[main_r])), 'parts': rows}
            members_r, chain_r = chain_models(remapped, database, names, shelf=shelf, **split)
            secs['chain_remapped_s'] = time.perf_counter() - t0
            joined['remapped'] = (remapped, members_r)
            info['chain']['remapped'] = {'joined_models': len(members_r), 'registered': len(_union(remapped, members_r)), 'links': chain_r}
    if shelf is not None:              # each route re-checked as a whole: a member that disagrees with all the others is dropped
        for route, (ms_, mem_) in list(joined.items()):
            mem2, rows = final_check(ms_, mem_, shelf)
            info['chain'][route]['final_check'] = rows
            if len(mem2) != len(mem_):
                joined[route] = (ms_, mem2)
                info['chain'][route]['registered_before_final_check'] = info['chain'][route]['registered']
                info['chain'][route]['registered'] = len(_union(ms_, mem2))
                info['chain'][route]['joined_models'] = len(mem2)
    label = max(joined, key=lambda k: (info['chain'][k]['registered'], k == 'original'))
    ms, members = joined[label]
    if len(joined) == 2:          # the two routes are independent estimates: where they overlap they must agree
        a, b = (joined_centres(*joined[k]) for k in ('original', 'remapped'))
        sim, ev = estimate_link(a, b)
        info['cross_check'] = {'what': 'camera centres of the original-fragment chain vs the re-mapped model, one similarity over the shared frames',
                               **{k: ev.get(k) for k in ('shared_frames', 'inliers', 'inlier_fraction', 'rms_rel_extent', 'decision')}}
    final = _union(ms, members)
    for f, row in zip(frags, info['fragments']):
        row['frames_in_final_model'] = sum(names[i] in final for i in f['frames'])
    main_size = len(_reg_names(ms[members[0][0]])) if members else 0
    info.update(used=label, registered=len(final), fragments_after=len(ms) - len(members) + 1 if members else len(ms),
                largest_before=frags[0]['size'] if frags else 0, main_model_registered=main_size,
                unlinked_models=[{'model': int(k), 'registered': len(_reg_names(ms[k]))} for k in ms if k not in {m[0] for m in members}],
                members=[{'model': int(k), 'registered': len(_reg_names(ms[k])), **({'used_frames': len(keep)} if keep is not None else {}),
                          'transform': 'identity (main model)' if i == 0 else f'similarity (scale {sim[0]:.4f})'} for i, (k, sim, keep) in enumerate(members)],
                deferred=deferred or None, plan=plan.log, step_seconds={k: round(v, 1) for k, v in secs.items()},
                seconds=round(time.perf_counter() - t_all, 1))
    if shelf is not None:
        info['plane_check'] = {'rule': f'a joined part (model, or first-pass fragment inside the re-mapped model) is refused unless its shelf face is within '
                                       f'{shelf.max_angle_deg} deg of the main shelf plane (any axis, incl. about the vertical), the median distance of its book '
                                       f'positions to that plane is <= {shelf.max_offset_spines} spine heights, >= {shelf.camera_side_min:.0%} of its cameras are '
                                       f'on the room side, and it has >= {shelf.min_books} book positions',
                               'max_angle_deg': shelf.max_angle_deg, 'max_offset_spines': shelf.max_offset_spines,
                               'camera_side_min': shelf.camera_side_min, 'min_books': shelf.min_books, 'refused': refused_joins(info)}
    return [(ms[k], sim, keep) for k, sim, keep in members], info
