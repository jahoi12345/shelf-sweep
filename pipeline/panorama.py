"""Planar shelf-face panorama of ONE saved run, built from the run's own tracked frame motion.

Any run with a manifest and saved frames qualifies (site sweeps, browser replays, final runs, offline replays): no
registered recording, no re-detection and no fresh association are needed. The frame-to-frame motion is the LK
chain the association itself uses (pipeline/cross_view.chain_transforms: affine steps, homography only when clearly
better, tracking holds, new segment after a lost track); each connected segment becomes one component image. A tracking
hold or an implausible transform is never silently dropped: the frame is listed in `unregistered_frames` with the reason.
The chain lives in the first frame's perspective (a tall shelf looks like a trapezoid), so every component is rectified
to a fronto-parallel view automatically: the vertical vanishing point comes from the detected spine axes, the camera
rotation that makes it vertical is applied as a homography (focal length assumed, no human marks), then a small in-plane
roll is removed from the long horizontal/vertical edges of the result. Frames are blended with centre-weighted seams.

Markers: exactly one per BOOK group of the run's own inventory.json (counts.book_count), placed at the median of the group's
sightings projected into the component that holds most of them; groups with no sighting in a registered frame are listed in
`unplaced_groups`, never dropped. Non-book groups get separate `item_markers`.

No metric scale is implied. Output is a NEW versioned directory (`runs/<run>_panorama_vN`), never an existing one.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from pipeline import cross_view
from pipeline.detect import read_json, save_json, resolve_evidence

SCHEMA_VERSION = 3
LIMITS = dict(max_output_extent=3600, max_output_pixels=10_000_000, min_area_ratio=0.02, max_area_ratio=60.,
              max_frame_extent=20000, rectify_min_lines=25, rectify_min_inlier_fraction=0.12, rectify_tol_deg=2.5,
              default_focal_over_long_side=0.8, max_pitch_deg=70., yaw_min_lines=8, yaw_tol_deg=0.8, yaw_min_inlier_fraction=0.35)
CONVENTION = 'column vectors [x,y,1]; divide by w; native top-left pixel coordinates -> component output top-left pixel coordinates; x right, y down'
HOLD_REASON = {'hold:tracking_failed': 'tracking between this frame and the previous anchor failed (blur / too little texture / too fast)',
               'hold:no_image': 'frame image missing or unreadable'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def corners(frame):
    w, h = frame['width_px'], frame['height_px']
    return np.array([[0., 0.], [w, 0.], [w, h], [0., h]])


def project(points, matrix):
    p, H = np.asarray(points, float), np.asarray(matrix, float)
    if p.ndim != 2 or p.shape[1] != 2 or H.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(H).all():
        raise ValueError('Finite Nx2 points and 3x3 matrix required')
    q = np.c_[p, np.ones(len(p))] @ H.T
    if np.any(np.abs(q[:, 2]) < 1e-10):
        raise ValueError('Projective pole')
    q = q[:, :2] / q[:, 2:]
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite projection')
    return q


def safe_transform(frame, matrix, limits=None):
    """(ok, reason): the transform of one frame is finite, non-singular, not flipped, keeps the image off the projective
    pole and has a bounded footprint. Area ratio is relative to the native frame (zoom changes along a chain are legitimate)."""
    limits = {**LIMITS, **(limits or {})}
    H = np.asarray(matrix, float)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return False, 'Invalid/nonfinite homogeneous matrix'
    H = H / H[2, 2]
    if abs(np.linalg.det(H)) < 1e-10:
        return False, 'Singular transform'
    den = np.c_[corners(frame), np.ones(4)] @ H[2]
    if den.min() * den.max() <= 0:
        return False, 'Projective pole crosses native image'
    try:
        mapped = project(corners(frame), H)
    except ValueError as e:
        return False, str(e)
    area = cv2.contourArea(mapped.astype(np.float32), oriented=True) / (frame['width_px'] * frame['height_px'])
    if not limits['min_area_ratio'] < area < limits['max_area_ratio']:
        return False, 'Accumulated area distortion or flipped mapping'
    if np.ptp(mapped, axis=0).max() > limits['max_frame_extent']:
        return False, 'Accumulated frame extent limit'
    return True, None


def canvas_geometry(footprints, limits=None, unit_scale=1.0):
    """Translation/scale AFTER the frame transforms; bounded before any allocation. `unit_scale` is the preferred
    output pixels per plane pixel (1 = plane resolution); it is lowered if the canvas would exceed the caps."""
    limits = {**LIMITS, **(limits or {})}
    pts = np.vstack(footprints)
    if not np.isfinite(pts).all():
        raise ValueError('Non-finite footprint')
    lo, hi = np.floor(pts.min(axis=0)), np.ceil(pts.max(axis=0))
    w, h = hi - lo
    if min(w, h) <= 0:
        raise ValueError('Empty canvas')
    scale = min(unit_scale, limits['max_output_extent'] / max(w, h), math.sqrt(limits['max_output_pixels'] / (w * h)))
    width, height = int(math.ceil(w * scale)), int(math.ceil(h * scale))
    while width * height > limits['max_output_pixels'] or max(width, height) > limits['max_output_extent']:
        scale *= .999
        width, height = int(math.ceil(w * scale)), int(math.ceil(h * scale))
    return np.array([[scale, 0., -lo[0] * scale], [0., scale, -lo[1] * scale], [0., 0., 1.]]), width, height


# ------------------------------------------------------------------------------------------ motion
def homography_chain(order, image, T_assoc, status, scale=0.5):
    """Re-estimate the association's chain with FULL homography steps (frame -> segment plane, native pixels).

    Keeps the association's decisions (which frames are anchors, holds and segment starts, from `status`) and its
    anchor structure; only the per-step model changes: LK tracks anchor -> frame, RANSAC homography. The association's
    affine-first steps drop the perspective of a tilting handheld camera, which shows up as a tapered/sheared shelf; a
    homography step keeps it so the final rectification is a single projective correction. Where the homography step is not
    trustworthy (few inliers, extreme perspective terms) the association's own relative affine step is used instead.
    Returns ({frame_id: 3x3}, {frame_id: 'homography'|'association_affine'|'start'}, fallback_count)."""
    S = np.diag([scale, scale, 1.0])
    Si = np.diag([1 / scale, 1 / scale, 1.0])
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    out, how = {}, {}
    prev, prev_gray, fallbacks = None, None, 0
    for fid in order:
        st = status.get(fid)
        if st in ('start', 'segment_start'):
            out[fid], how[fid] = np.eye(3), 'start'
            img = image(fid)
            prev, prev_gray = fid, (cross_view._gray(img, scale) if img is not None else None)
            continue
        if st != 'ok' or prev is None:
            continue
        img = image(fid)
        if img is None:
            continue
        cur = cross_view._gray(img, scale)
        H, inliers = None, 0
        if prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1500, qualityLevel=0.005, minDistance=6, blockSize=5)
            if p0 is not None and len(p0) > 30:
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur, p0, None, winSize=(31, 31), maxLevel=4, criteria=crit)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev_gray, p1, None, winSize=(31, 31), maxLevel=4)
                good = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pb - p0, axis=2).ravel() < 1.0)
                if good.sum() >= 30:
                    Hh, mask = cv2.findHomography(p0[good].reshape(-1, 2), p1[good].reshape(-1, 2), cv2.RANSAC, 3.0)
                    if Hh is not None and mask is not None and np.isfinite(Hh).all():
                        inliers = int(mask.sum())
                        persp = float(np.abs(Hh[2, :2]).max()) / max(abs(Hh[2, 2]), 1e-12)
                        if inliers >= 40 and inliers >= 0.5 * int(good.sum()) and persp < 2e-3 and 0.5 < np.linalg.det(Hh[:2, :2]) / Hh[2, 2] ** 2 < 2.0:
                            H = Hh
        if H is not None:
            try:
                cand = out[prev] @ np.linalg.inv(Si @ H @ S)
                cand = cand / cand[2, 2]
                if np.isfinite(cand).all():
                    out[fid], how[fid] = cand, 'homography'
            except np.linalg.LinAlgError:
                pass
        if fid not in out:
            rel = np.linalg.inv(np.asarray(T_assoc[prev], float)) @ np.asarray(T_assoc[fid], float)
            out[fid], how[fid] = out[prev] @ rel, 'association_affine'
            fallbacks += 1
        prev, prev_gray = fid, cur
    return out, how, fallbacks


# ------------------------------------------------------------------------------------------ global alignment
MIN_COST_DROP = 0.85   # the refined solution must remove >= 15% of the robust transfer error, else the tracked chain is kept
ALIGN = dict(gaps=(1, 2, 4, 8), features=1800, ratio=0.85, min_inliers=30, min_inlier_ratio=0.3, ransac_px=3.0, iterations=12,
             huber_px=4.0, max_points_per_pair=150, gate_base_px=12.0, gate_per_gap_px=8.0, min_pairs_per_frame=1)


def _normalise(H, unit=1000.0):
    N = np.diag([1 / unit, 1 / unit, 1.0])
    return N @ H @ np.linalg.inv(N)


def _hom(delta):
    D = np.zeros((3, 3))
    D.flat[:8] = delta
    return np.eye(3) + D


def _transfer(Ha, Hb, xa, xb):
    """Residual of x_a mapped into frame b: pi(Hb^-1 Ha x_a) - x_b (normalised coordinates)."""
    M = np.linalg.solve(Hb, Ha)
    q = np.c_[xa, np.ones(len(xa))] @ M.T
    return q[:, :2] / q[:, 2:3] - xb


def collect_matches(order, T, image, gaps=None, scale=0.5, progress=None, cfg=None):
    """SIFT matches between frames up to max(gaps) apart in `order` (all in one tracking segment), gated by the chain's own
    prediction (so repeated spine texture cannot pair the wrong books) and verified by RANSAC homography.
    Returns [(i, j, xa, xb, inliers)] with coordinates in FULL-resolution pixels."""
    cfg = {**ALIGN, **(cfg or {})}
    gaps = gaps or cfg['gaps']
    sift = cv2.SIFT_create(nfeatures=cfg['features'])
    feats = []
    for n, fid in enumerate(order):
        img = image(fid)
        if img is None:
            feats.append(None)
            continue
        g = cross_view._gray(img, scale)
        k, d = sift.detectAndCompute(g, None)
        feats.append((np.float32([p.pt for p in k]), d) if d is not None and len(k) >= 40 else None)
        if progress and n % 10 == 0:
            progress('aligning: features', done=n, total=len(order))
    S = np.diag([scale, scale, 1.0])
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    for i in range(len(order)):
        for gap in gaps:
            j = i + gap
            if j >= len(order) or feats[i] is None or feats[j] is None:
                continue
            (pa, da), (pb, db) = feats[i], feats[j]
            M = S @ np.linalg.inv(T[order[j]]) @ T[order[i]] @ np.linalg.inv(S)                      # half-scale a -> half-scale b
            q = np.c_[pa, np.ones(len(pa))] @ M.T
            with np.errstate(divide='ignore', invalid='ignore'):
                q = q[:, :2] / q[:, 2:3]
            if not np.isfinite(q).all():
                continue
            radius = cfg['gate_base_px'] + cfg['gate_per_gap_px'] * gap
            dist2 = ((q[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
            mask = (dist2 < radius ** 2).astype(np.uint8)
            if mask.sum() < 40:
                continue
            knn = matcher.knnMatch(da, db, k=2, mask=mask)
            good = [m[0] for m in knn if len(m) == 2 and m[0].distance < cfg['ratio'] * m[1].distance] + [m[0] for m in knn if len(m) == 1]
            if len(good) < cfg['min_inliers']:
                continue
            xa = pa[[m.queryIdx for m in good]]
            xb = pb[[m.trainIdx for m in good]]
            H, inl = cv2.findHomography(xa, xb, cv2.RANSAC, cfg['ransac_px'], maxIters=2000)
            if H is None or inl is None or int(inl.sum()) < cfg['min_inliers'] or inl.mean() < cfg['min_inlier_ratio']:
                continue
            keep = inl.ravel() == 1
            xa, xb = xa[keep], xb[keep]
            if len(xa) > cfg['max_points_per_pair']:
                idx = np.random.default_rng(i * 1000 + j).choice(len(xa), cfg['max_points_per_pair'], replace=False)
                xa, xb = xa[idx], xb[idx]
            pairs.append((i, j, xa / scale, xb / scale, int(keep.sum())))
        if progress and i % 10 == 0:
            progress('aligning: matching', done=i, total=len(order))
    return pairs


def refine_homographies(order, T, pairs, cfg=None):
    """Robust (Huber) Levenberg-Marquardt over all frame->plane homographies of one segment so that every matched point
    pair agrees after transfer (error measured in frame pixels, so the plane scale cannot collapse). Frame 0 is the gauge.
    Returns (new T dict, report); T is returned unchanged when the problem is under-constrained or the cost did not drop."""
    cfg = {**ALIGN, **(cfg or {})}
    n = len(order)
    report = {'frames': n, 'pairs': len(pairs), 'correspondences': int(sum(len(p[2]) for p in pairs)), 'applied': False}
    degree = np.zeros(n, int)
    for i, j, *_ in pairs:
        degree[i] += 1
        degree[j] += 1
    if n < 3 or not pairs or (degree[1:] < cfg['min_pairs_per_frame']).any():
        report['reason'] = 'frames without any verified match pair' if pairs else 'no verified match pairs'
        return T, report
    unit = 1000.0
    A = [_normalise(np.asarray(T[f], float) / np.asarray(T[f], float)[2, 2], unit) for f in order]
    data = [(i, j, np.asarray(xa) / unit, np.asarray(xb) / unit) for i, j, xa, xb, _ in pairs]
    huber = cfg['huber_px'] / unit

    def cost_of(Alist):
        total, count = 0.0, 0
        for i, j, xa, xb in data:
            r = _transfer(Alist[i], Alist[j], xa, xb)
            e = np.linalg.norm(r, axis=1)
            total += float(np.where(e < huber, 0.5 * e ** 2, huber * (e - 0.5 * huber)).sum())
            count += len(e)
        return total, count

    def rms_px(Alist):
        e = np.concatenate([np.linalg.norm(_transfer(Alist[i], Alist[j], xa, xb), axis=1) for i, j, xa, xb in data])
        return float(np.sqrt((e ** 2).mean()) * unit), float(np.median(e) * unit)

    report['rms_px_before'], report['median_px_before'] = [round(v, 3) for v in rms_px(A)]
    lam, eps = 1e-3, 1e-6
    cost, _ = cost_of(A)
    cost_before = cost
    dim = 8 * (n - 1)
    for it in range(cfg['iterations']):
        JtJ = np.zeros((dim, dim))
        Jtr = np.zeros(dim)
        for i, j, xa, xb in data:
            r0 = _transfer(A[i], A[j], xa, xb)
            e = np.linalg.norm(r0, axis=1)
            w = np.where(e < huber, 1.0, huber / np.maximum(e, 1e-12))                 # IRLS weights of the Huber loss
            w_rep = np.repeat(np.sqrt(w), 2)
            cols, idx = [], []
            for frame, which in ((i, 0), (j, 1)):
                if frame == 0:
                    continue
                for k in range(8):
                    d = np.zeros(8)
                    d[k] = eps
                    Ap = [None, None]
                    Ap[0], Ap[1] = A[i], A[j]
                    Ap[which] = _hom(d) @ Ap[which]
                    cols.append(((_transfer(Ap[0], Ap[1], xa, xb) - r0) / eps).ravel() * w_rep)
                    idx.append(8 * (frame - 1) + k)
            if not cols:
                continue
            J = np.array(cols).T
            rv = r0.ravel() * w_rep
            idx = np.array(idx)
            JtJ[np.ix_(idx, idx)] += J.T @ J
            Jtr[idx] += J.T @ rv
        for attempt in range(6):
            damped = JtJ + lam * (np.diag(np.diag(JtJ)) + 1e-9 * np.eye(dim))
            try:
                step = -np.linalg.solve(damped, Jtr)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            trial = [A[0]] + [_hom(step[8 * k:8 * k + 8]) @ A[k + 1] for k in range(n - 1)]
            trial = [a / a[2, 2] for a in trial]
            new_cost, _ = cost_of(trial)
            if np.isfinite(new_cost) and new_cost < cost:
                improved = cost - new_cost
                A, cost, lam = trial, new_cost, max(lam / 5, 1e-7)
                break
            lam *= 8
        else:
            break
        if improved < 1e-9 * max(cost, 1e-12):
            break
    report['rms_px_after'], report['median_px_after'] = [round(v, 3) for v in rms_px(A)]
    report['iterations'] = it + 1
    new = {}
    N = np.diag([1 / unit, 1 / unit, 1.0])
    for f, a in zip(order, A):
        H = np.linalg.inv(N) @ a @ N
        new[f] = H / H[2, 2]
    report['robust_cost_before'], report['robust_cost_after'] = round(cost_before * unit ** 2, 3), round(cost * unit ** 2, 3)
    report['applied'] = cost < MIN_COST_DROP * cost_before
    if not report['applied']:
        report['reason'] = 'the optimisation removed less than 15% of the transfer error; the tracked chain is kept (it already agrees with the matches)'
        return T, report
    return new, report


# ------------------------------------------------------------------------------------------ rectification
def book_axis_lines(observations, T, native_vertical_deg=35.0):
    """Long-axis lines of (near-)upright detected spines in the component plane: [(homogeneous unit line, midpoint)]."""
    lines, mids = [], []
    for o in observations:
        if o.get('kind') != 'book' or o['frame_id'] not in T:
            continue
        poly = o.get('polygon_xy')
        if not poly or len(poly) != 4:
            continue
        P = np.asarray(poly, float)
        e = [np.linalg.norm(P[i] - P[(i + 1) % 4]) for i in range(4)]
        shorts = [0, 2] if e[0] < e[1] else [1, 3]
        m1 = (P[shorts[0]] + P[(shorts[0] + 1) % 4]) / 2
        m2 = (P[shorts[1]] + P[(shorts[1] + 1) % 4]) / 2
        length = float(np.linalg.norm(m1 - m2))
        if length < 2.2 * min(e) or length < 20:
            continue
        if abs((m2 - m1)[1]) / length < math.cos(math.radians(native_vertical_deg)):   # a lying / strongly leaning book
            continue
        try:
            q = project([m1, m2], T[o['frame_id']])
        except ValueError:
            continue
        l = np.cross(np.append(q[0], 1.), np.append(q[1], 1.))
        n = math.hypot(l[0], l[1])
        if n < 1e-9:
            continue
        lines.append(l / n)
        mids.append(q.mean(axis=0))
    return np.asarray(lines).reshape(-1, 3), np.asarray(mids).reshape(-1, 2)


def vanishing_point(lines, mids, iters=2500, tol_deg=1.0, seed=0):
    """RANSAC + least squares vanishing point of near-parallel lines. Returns (homogeneous unit point, inlier mask) or None."""
    n = len(lines)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    direction = np.c_[-lines[:, 1], lines[:, 0]]
    best = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        v = np.cross(lines[i], lines[j])
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            continue
        v = v / norm
        if abs(v[2]) > 1e-12:
            d = v[:2] / v[2] - mids
        else:
            d = np.tile(v[:2], (n, 1))
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        ang = np.degrees(np.arccos(np.clip(np.abs((d * direction).sum(1)), 0, 1)))
        inliers = ang < tol_deg
        if best is None or inliers.sum() > best[0]:
            best = (int(inliers.sum()), inliers, v)
    if best is None or best[0] < 3:
        return None
    vt = np.linalg.svd(lines[best[1]])[2]
    v = vt[-1]
    return v / np.linalg.norm(v), best[1]


def rectifying_homography(vertical_vp, width, height, focal):
    """K R K^-1: the rotation of a virtual camera (same centre) that makes the vertical vanishing point the image's
    vertical direction and the shelf face fronto-parallel; horizontal = image x projected orthogonal to vertical."""
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])
    Ki = np.linalg.inv(K)
    dv = Ki @ vertical_vp
    dv /= np.linalg.norm(dv)
    if dv[1] < 0:
        dv = -dv
    ex = np.array([1., 0, 0])
    dh = ex - (ex @ dv) * dv
    dh /= np.linalg.norm(dh)
    n = np.cross(dh, dv)
    if n[2] < 0:
        n, dh = -n, -dh
    return K @ np.vstack([dh, dv, n]) @ Ki


def edge_angles(gray, min_len=50, horizontal_window=35.0):
    """(horizontal deviation deg, weight) of long near-horizontal segments, same for near-vertical (deviation from vertical)."""
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = detector.detect(gray)[0]
    if found is None:
        return (np.zeros(0), np.zeros(0)), (np.zeros(0), np.zeros(0))
    seg = found.reshape(-1, 4)
    d = seg[:, 2:] - seg[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    ang = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 90) % 180 - 90        # -90..90, 0 = horizontal
    horizontal = (length > min_len) & (np.abs(ang) < horizontal_window)
    vertical = (length > min_len) & (np.abs(np.abs(ang) - 90) < 12)
    vdev = np.where(ang > 0, ang - 90, ang + 90)
    return (ang[horizontal], length[horizontal]), (vdev[vertical], length[vertical])


def weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def leveling_matrix(preview_gray):
    """Affine (about the preview origin, unit area) that makes the long near-horizontal and near-vertical edges of the
    rectified preview axis-aligned: removes the residual in-plane roll AND shear a yawed camera leaves after the vertical
    vanishing-point rectification (shelf boards sloping while spines are vertical). Returns (3x3, info)."""
    (ha, hw), (va, vw) = edge_angles(preview_gray, min_len=max(30, 0.06 * max(preview_gray.shape)))
    theta = weighted_median(ha, hw) if len(ha) >= 5 else None     # horizontals' tilt (deg)
    lean = weighted_median(va, vw) if len(va) >= 8 else None       # verticals' lean (deg)
    info = {'horizontal_edges': int(len(ha)), 'vertical_edges': int(len(va)), 'horizontal_tilt_deg': None if theta is None else round(theta, 2),
            'vertical_lean_deg': None if lean is None else round(lean, 2)}
    if theta is None and lean is None:
        return np.eye(3), info
    if theta is None:
        theta = lean
    if lean is None:
        lean = theta
    uh = np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta))])
    uv = np.array([math.sin(math.radians(lean)), math.cos(math.radians(lean))])   # vertical direction as measured; lean > 0: the top leans right (image y points down)
    base = np.c_[uh, uv]
    if abs(np.linalg.det(base)) < 0.5:       # the two directions are nearly parallel: not a plausible shelf frame
        return np.eye(3), info
    L = np.linalg.inv(base)
    L = L / math.sqrt(abs(np.linalg.det(L)))
    out = np.eye(3)
    out[:2, :2] = L
    return out, info


def horizontal_lines(gray, to_plane, min_len_frac=0.05, window=35.0):
    """Long near-horizontal segments (shelf boards, rows of book tops) of a preview image as unit lines in PLANE coordinates
    (to_plane: preview pixel -> plane pixel 3x3). Returns (lines, midpoints, lengths)."""
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = detector.detect(gray)[0]
    if found is None:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0)
    seg = found.reshape(-1, 4)
    d = seg[:, 2:] - seg[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    ang = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 90) % 180 - 90
    keep = (length > min_len_frac * max(gray.shape)) & (np.abs(ang) < window)
    lines, mids, lens = [], [], []
    for (x0, y0, x1, y1), L in zip(seg[keep], length[keep]):
        q = project([[x0, y0], [x1, y1]], to_plane)
        l = np.cross(np.append(q[0], 1.), np.append(q[1], 1.))
        n = math.hypot(l[0], l[1])
        if n < 1e-9:
            continue
        lines.append(l / n)
        mids.append(q.mean(axis=0))
        lens.append(L)
    return np.asarray(lines).reshape(-1, 3), np.asarray(mids).reshape(-1, 2), np.asarray(lens)


def yaw_homography(horizontal_vp, width, height, focal, max_yaw_deg=65.0):
    """K R K^-1 for the rotation about the VERTICAL axis of the (already pitch-rectified) virtual camera that sends the
    horizontal vanishing point to infinity along image x, i.e. makes the shelf boards parallel and level. None when the
    vanishing point is (nearly) at infinity already or the implied yaw is implausible."""
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])
    d = np.linalg.inv(K) @ horizontal_vp
    d[1] = 0.0
    norm = math.hypot(d[0], d[2])
    if norm < 1e-12:
        return None, 0.0
    x_axis = np.array([d[0], 0.0, d[2]]) / norm
    if x_axis[0] < 0:
        x_axis = -x_axis
    yaw = math.degrees(math.atan2(x_axis[2], x_axis[0]))
    if abs(yaw) < 0.5 or abs(yaw) > max_yaw_deg:
        return None, yaw
    y_axis = np.array([0., 1., 0.])
    z_axis = np.cross(x_axis, y_axis)
    return K @ np.vstack([x_axis, y_axis, z_axis]) @ np.linalg.inv(K), yaw


# ------------------------------------------------------------------------------------------ composition
def quality_weight(frame, fallback):
    s = (frame.get('quality') or {}).get('sharpness')
    s = fallback if s is None else s
    return s


def compose(members, M, images, size, progress=None, sigma=None):
    """Centre-weighted seam composition. members: frame records; M: {frame_id: frame -> canvas 3x3}; images(fid) -> BGR.
    Returns (canvas BGR uint8, coverage uint8 0/255, owner uint16 1-based index into `members`, per-frame valid fraction)."""
    width, height = size
    sharp = np.array([quality_weight(f, 0.) for f in members], float)
    median = np.median(sharp[sharp > 0]) if (sharp > 0).any() else 1.0
    q = np.clip((np.where(sharp > 0, sharp, median) / median) ** 0.5, 0.5, 1.6)
    best = np.zeros((height, width), np.float32)
    owner = np.zeros((height, width), np.uint16)
    rois = []
    for k, f in enumerate(members):
        H = M[f['frame_id']]
        fp = project(corners(f), H)
        x0, y0 = np.maximum(0, np.floor(fp.min(axis=0)).astype(int))
        x1, y1 = np.minimum([width, height], np.ceil(fp.max(axis=0)).astype(int))
        rois.append((x0, y0, x1, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        roi_H = np.array([[1., 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ H
        mask = cv2.warpPerspective(np.full((f['height_px'], f['width_px']), 255, np.uint8), roi_H, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        if dist.max() <= 0:
            continue
        score = dist / dist.max() * q[k]
        view = best[y0:y1, x0:x1]
        win = (score > view) & (mask > 0)
        view[win] = score[win]
        owner[y0:y1, x0:x1][win] = k + 1
        if progress and k % 10 == 0:
            progress('compositing seams', done=k, total=len(members))
    sigma = sigma or max(2.0, 0.0035 * max(width, height))
    accum = np.zeros((height, width, 3), np.float32)
    weight = np.zeros((height, width), np.float32)
    pad = int(math.ceil(3 * sigma))
    for k, f in enumerate(members):
        if progress and k % 10 == 0:
            progress('compositing blend', done=k, total=len(members))
        x0, y0, x1, y1 = rois[k]
        if x1 <= x0 or y1 <= y0:
            continue
        img = images(f['frame_id'])
        H = M[f['frame_id']]
        # pre-shrink strongly downscaled frames (area averaging) so the warp does not alias
        local = math.sqrt(abs(np.linalg.det(H[:2, :2])))
        shrink = int(1 / local) if local > 0 and 1 / local >= 2 else 1
        if shrink > 1:
            img = cv2.resize(img, None, fx=1 / shrink, fy=1 / shrink, interpolation=cv2.INTER_AREA)
            H = H @ np.diag([shrink, shrink, 1.])
        bx0, by0, bx1, by1 = max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)
        roi_H = np.array([[1., 0, -bx0], [0, 1, -by0], [0, 0, 1]]) @ H
        size_roi = (bx1 - bx0, by1 - by0)
        warped = cv2.warpPerspective(img, roi_H, size_roi, flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), roi_H, size_roi, flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((3, 3), np.uint8)) > 0
        mine = (owner[by0:by1, bx0:bx1] == k + 1).astype(np.float32)
        soft = cv2.GaussianBlur(mine, (0, 0), sigma) * valid
        accum[by0:by1, bx0:bx1] += warped.astype(np.float32) * soft[..., None]
        weight[by0:by1, bx0:bx1] += soft
    covered = weight > 1e-4
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[covered] = np.clip(accum[covered] / weight[covered, None], 0, 255).astype(np.uint8)
    # pixels owned by a frame always count as covered
    covered |= owner > 0
    coverage = np.where(covered, 255, 0).astype(np.uint8)
    return canvas, coverage, owner


# ------------------------------------------------------------------------------------------ build
class Progress:
    """progress.json for the build job UI: stage, label, done/total and an overall `fraction` 0..1 (stage weights measured on 60-100 frame runs)."""
    GLOBAL = {'tracking': .08, 'components': .85, 'markers': .03, 'writing': .04}
    PER = (('features', .13), ('matching', .58), ('rectifying', .16), ('seams', .04), ('blend', .09))     # shares of one component's work
    STAGE = {'aligning': ('features', 0.0), 'aligning: features': ('features', None), 'aligning: matching': ('matching', None), 'rectifying': ('rectifying', 0.0),
             'compositing': ('seams', 0.0), 'compositing seams': ('seams', None), 'compositing blend': ('blend', None)}
    LABEL = {'tracking': 'tracking frame motion', 'aligning': 'aligning frames', 'aligning: features': 'aligning frames: features', 'aligning: matching': 'aligning frames: matching',
             'rectifying': 'rectifying the shelf face', 'compositing': 'blending the panorama', 'compositing seams': 'blending the panorama: seams', 'compositing blend': 'blending the panorama',
             'markers': 'placing book markers', 'writing': 'writing the result', 'done': 'done'}

    def __init__(self, out):
        self.path = Path(out) / 'progress.json'
        self.started = time.time()
        self.stage = None
        self.sizes = None
        self.component = 0

    def set_components(self, sizes):
        total = float(sum(sizes)) or 1.0
        self.sizes = [s / total for s in sizes]

    def fraction(self, stage, extra):
        if stage == 'done':
            return 1.0
        if stage == 'tracking':
            return 0.0
        base = self.GLOBAL['tracking']
        if stage == 'markers':
            return base + self.GLOBAL['components']
        if stage == 'writing':
            return base + self.GLOBAL['components'] + self.GLOBAL['markers']
        if stage not in self.STAGE:
            return base
        if extra.get('component'):
            self.component = int(extra['component']) - 1
        name, fixed = self.STAGE[stage]
        sub = fixed if fixed is not None else (min(1.0, extra['done'] / extra['total']) if extra.get('total') else 0.0)
        within = sum(w for n, w in self.PER if n == name) * sub + sum(w for n, w in self.PER[:[n for n, _ in self.PER].index(name)])
        sizes = self.sizes or [1.0]
        before = sum(sizes[:self.component])
        weight = sizes[self.component] if self.component < len(sizes) else 0.0
        return base + self.GLOBAL['components'] * (before + weight * within)

    def __call__(self, stage, **extra):
        self.stage = stage
        label = self.LABEL.get(stage, stage)
        if extra.get('total'):
            label += f" {extra.get('done', 0)}/{extra['total']}"
        if self.sizes and len(self.sizes) > 1 and extra.get('component', self.component + 1):
            label += f" · section {int(extra.get('component', self.component + 1))}/{len(self.sizes)}"
        data = {'stage': stage, 'label': label, 'fraction': round(min(1.0, max(0.0, self.fraction(stage, extra))), 4),
                'elapsed_s': round(time.time() - self.started, 1), **extra}
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)


def next_output(runs_dir, run_id):
    """First unused runs/<run>_panorama_vN."""
    n = 1
    while (Path(runs_dir) / f'{run_id}_panorama_v{n}').exists():
        n += 1
    return f'{run_id}_panorama_v{n}'


def _polygon(o):
    poly = o.get('polygon_xy')
    if poly and len(poly) >= 3:
        return poly
    x1, y1, x2, y2 = o['bbox_xyxy']
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _inventory_groups(source):
    path = source / 'inventory.json'
    if not path.is_file():
        return None, []
    inventory = read_json(path)
    data = inventory.get('data', {})
    groups = [dict(g, _kind='book') for g in data.get('books', [])] + [dict(g, _kind='item') for g in data.get('items', [])]
    return inventory, groups


def _draw_marker_image(canvas, markers, polygons):
    out = canvas.copy()
    for poly in polygons:
        cv2.polylines(out, [np.rint(poly).astype(np.int32)], True, (60, 220, 40), 1, cv2.LINE_AA)
    for m in markers:
        c = (int(round(m['x'])), int(round(m['y'])))
        cv2.circle(out, c, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, c, 5, (40, 40, 220), 1, cv2.LINE_AA)
    return out


def build(source_run, output_run, scan_label=None, registry=None, *, progress=None, chain='association', align=True):
    started = time.perf_counter()
    root, out = Path(source_run).resolve(), Path(output_run).resolve()
    if out.exists():
        raise ValueError('Refusing existing versioned output ' + str(out))
    manifest = read_json(root / 'manifest.json')
    frames = sorted(manifest['data']['frames'], key=lambda f: (f['timestamp_s'], f['frame_id']))
    if len(frames) < 2:
        raise ValueError('Run has fewer than 2 saved frames')
    if len({f['frame_id'] for f in frames}) != len(frames):
        raise ValueError('Duplicate frame IDs')
    candidates = read_json(root / 'candidates.json') if (root / 'candidates.json').is_file() else None
    observations = (candidates or {}).get('data', {}).get('observations', [])
    inventory, groups = _inventory_groups(root)
    out.mkdir(parents=True)
    progress = progress or Progress(out)
    try:
        return _build(root, out, manifest, frames, candidates, observations, inventory, groups, scan_label, started, progress, chain, align)
    except BaseException as exc:
        shutil.rmtree(out, ignore_errors=True)      # never leave a half-written versioned dir behind
        if not isinstance(exc, Exception):
            raise
        raise RuntimeError(f'panorama failed while {getattr(progress, "stage", None) or "starting"}: {type(exc).__name__}: {exc}') from exc


def _build(root, out, manifest, frames, candidates, observations, inventory, groups, scan_label, started, progress, chain='association', align=True):
    by_frame = {f['frame_id']: f for f in frames}
    cache = {}

    def image(fid):
        if fid not in cache:
            f = by_frame[fid]
            img = cv2.imread(str(resolve_evidence(root, f['image_path'])))
            if img is None or img.shape[1::-1] != (f['width_px'], f['height_px']):
                return None
            cache[fid] = img
            while len(cache) > 6:
                cache.pop(next(iter(cache)))
        return cache[fid]

    # --- 1. the run's own tracked motion (same function and scale the association uses)
    progress('tracking', total=len(frames))
    t0 = time.perf_counter()
    order = [f['frame_id'] for f in frames]
    T, status = cross_view.chain_transforms(order, image, cross_view.DEFAULTS['track_scale'])
    segment = cross_view.segments_from_status(order, status)
    chain_mode = chain
    step_model = {}
    if chain == 'homography':
        T_h, step_model, fallbacks = homography_chain(order, image, T, status, cross_view.DEFAULTS['track_scale'])
        for fid, m in T_h.items():
            T[fid] = m
    tracking_s = time.perf_counter() - t0
    unregistered, edges = [], []
    prev = None
    for fid in order:
        st = status.get(fid, 'hold:no_image')
        ok = st in ('start', 'ok', 'segment_start')
        edges.append({'edge_id': f'step_{len(edges):05d}', 'from_frame': fid, 'to_frame': prev, 'direction': 'frame -> previous tracked anchor',
                      'status': 'accepted' if ok else 'rejected', 'method': 'lk_chain (pipeline.cross_view.chain_transforms)',
                      'tracking_status': st, 'composition_status': 'accepted' if ok else 'rejected',
                      'reason': None if ok else HOLD_REASON.get(st, st)})
        if not ok:
            unregistered.append({'frame_id': fid, 'reason': HOLD_REASON.get(st, st), 'stage': 'tracking'})
        else:
            prev = fid

    # --- 2. one component per tracking segment
    groups_by_segment = {}
    for fid in order:
        if status.get(fid) in ('start', 'ok', 'segment_start'):
            groups_by_segment.setdefault(segment[fid], []).append(fid)
    obs_by_frame = {}
    for o in observations:
        obs_by_frame.setdefault(o['frame_id'], []).append(o)
    group_by_obs = {oid: g for g in groups for oid in g.get('observation_ids', [])}

    progress.set_components([len(f) for _, f in sorted(groups_by_segment.items(), key=lambda kv: -len(kv[1]))] or [1])
    components, thumbs = [], []
    index_of = {fid: i + 1 for i, fid in enumerate(order)}
    render_total = len(groups_by_segment)
    for n, (seg_id, fids) in enumerate(sorted(groups_by_segment.items(), key=lambda kv: -len(kv[1]))):
        progress('aligning', component=n + 1, of=render_total, frames=len(fids))
        members = [by_frame[f] for f in fids]
        Tseg = {f: np.asarray(T[f], float) for f in fids}
        alignment = {'applied': False, 'reason': 'disabled' if not align else 'fewer than 3 frames'}
        refined = None
        if align and len(fids) >= 3:
            pairs = collect_matches(fids, Tseg, image, progress=progress)
            progress('rectifying', component=n + 1, of=render_total, frames=len(fids))
            refined, alignment = refine_homographies(fids, Tseg, pairs)
            if not alignment['applied']:
                refined = None
        # Express the plane in the MIDDLE frame's pixels: the chain starts at the segment's first frame, whose perspective
        # makes the far end of a long pan explode (scale 2-10x); the middle frame halves the foreshortening on both sides.
        mid = fids[len(fids) // 2]

        def rereference(Tm):
            R0 = np.linalg.inv(Tm[mid])
            return {f: (R0 @ m) / (R0 @ m)[2, 2] for f, m in Tm.items()}
        Tseg = rereference(Tseg)
        if refined is not None:
            candidate = rereference({f: np.asarray(m, float) for f, m in refined.items()})
            bad = [f for f in fids if not safe_transform(by_frame[f], candidate[f])[0]]
            if bad:
                alignment = {**alignment, 'applied': False, 'reason': f'refined transforms were implausible for {len(bad)} frame(s); the tracked chain is kept'}
            else:
                Tseg = candidate
        # sanity gate on the raw chain
        kept = []
        for f in members:
            ok, reason = safe_transform(f, Tseg[f['frame_id']])
            if ok:
                kept.append(f)
            else:
                unregistered.append({'frame_id': f['frame_id'], 'reason': reason, 'stage': 'composition'})
        members = kept
        if not members:
            continue
        ref = by_frame[mid] if mid in {f['frame_id'] for f in members} else members[len(members) // 2]
        rectification = {'applied': False, 'reason': 'single frame'}
        Hr = np.eye(3)
        if len(members) >= 2:
            lines, mids = book_axis_lines([o for f in members for o in obs_by_frame.get(f['frame_id'], [])], Tseg)
            rectification = {'applied': False, 'reason': f'only {len(lines)} usable spine axes (need {LIMITS["rectify_min_lines"]})', 'spine_axes': int(len(lines))}
            if len(lines) >= LIMITS['rectify_min_lines']:
                vp = vanishing_point(lines, mids, tol_deg=LIMITS['rectify_tol_deg'])
                if vp is not None:
                    v, inl = vp
                    frac = float(inl.mean())
                    focal = LIMITS['default_focal_over_long_side'] * max(ref['width_px'], ref['height_px'])
                    dv = np.linalg.inv(np.array([[focal, 0, ref['width_px'] / 2], [0, focal, ref['height_px'] / 2], [0, 0, 1.]])) @ v
                    pitch = math.degrees(math.asin(min(1.0, abs(dv[2]) / np.linalg.norm(dv))))   # camera pitch from horizontal (0 = level)
                    if frac >= LIMITS['rectify_min_inlier_fraction'] and pitch <= LIMITS['max_pitch_deg']:
                        Hr = rectifying_homography(v, ref['width_px'], ref['height_px'], focal)
                        rectification = {'applied': True, 'method': 'vertical vanishing point of detected spine axes -> fronto-parallel rotation homography',
                                         'spine_axes': int(len(lines)), 'inlier_fraction': round(frac, 3), 'assumed_focal_px': round(focal, 1),
                                         'camera_pitch_deg': round(pitch, 1), 'vertical_vp': [round(float(x), 6) for x in v]}
                    else:
                        rectification = {'applied': False, 'reason': f'vanishing point unreliable (inliers {frac:.0%}, pitch {pitch:.0f} deg)', 'spine_axes': int(len(lines))}
        Mseg = {f['frame_id']: Hr @ Tseg[f['frame_id']] for f in members}
        # preview composite -> in-plane roll correction -> final canvas
        locals_ = [math.sqrt(abs(np.linalg.det(Mseg[f['frame_id']][:2, :2] / Mseg[f['frame_id']][2, 2]))) for f in members]
        unit = 1.0 / float(np.median(locals_)) if np.median(locals_) > 0 else 1.0
        try:
            if rectification['applied']:
                Hy, yaw_info = _yaw_from_preview(members, Mseg, unit, image, ref, rectification['assumed_focal_px'])
                rectification['yaw'] = yaw_info
                if Hy is not None:
                    Mseg = {k: Hy @ v for k, v in Mseg.items()}
                Lm, level_info = _leveling_from_preview(members, Mseg, unit, image)
                rectification['leveling'] = level_info
                if not np.allclose(Lm, np.eye(3)):
                    Mseg = {k: Lm @ v for k, v in Mseg.items()}
            locals_ = [math.sqrt(abs(np.linalg.det(Mseg[f['frame_id']][:2, :2] / Mseg[f['frame_id']][2, 2]))) for f in members]
            unit = 1.0 / float(np.median(locals_)) if np.median(locals_) > 0 else 1.0
            A, width, height = canvas_geometry([project(corners(f), Mseg[f['frame_id']]) for f in members], unit_scale=unit)
        except ValueError as exc:
            for f in members:
                unregistered.append({'frame_id': f['frame_id'], 'reason': str(exc), 'stage': 'composition'})
            continue
        M = {k: (A @ v) / (A @ v)[2, 2] for k, v in Mseg.items()}
        progress('compositing', component=n + 1, of=render_total, frames=len(members), canvas=f'{width}x{height}')

        def img_of(fid):
            im = image(fid)
            if im is None:
                raise ValueError('Frame image became unreadable: ' + fid)
            return im
        canvas, coverage, owner = compose(members, M, img_of, (width, height), progress=progress)
        # owner indexes `members`; remap to the 1-based index into the run's frame list
        remap = np.zeros(len(members) + 1, np.uint16)
        for k, f in enumerate(members):
            remap[k + 1] = index_of[f['frame_id']]
        owner_global = remap[owner]
        stem = f'component_{len(components) + 1:03d}'
        records, overlays, polygons = [], [], []
        for k, f in enumerate(members):
            H = M[f['frame_id']]
            footprint = project(corners(f), H)
            rec = {**{k2: v for k2, v in f.items() if k2 != 'quality'}, 'quality': f.get('quality'), 'source_index': index_of[f['frame_id']],
                   'tracking_status': status[f['frame_id']], 'sha256': None, 'original_to_panorama_3x3': H.tolist(),
                   'panorama_to_original_3x3': np.linalg.inv(H).tolist(), 'footprint': footprint.tolist(), 'coordinate_system': CONVENTION}
            records.append(rec)
            for o in obs_by_frame.get(f['frame_id'], []):
                poly = _polygon(o)
                try:
                    pp = project(poly, H)
                except ValueError:
                    continue
                g = group_by_obs.get(o['observation_id'])
                overlays.append({**o, 'component_id': stem, 'plane_id': stem + '_unverified_plane', 'timestamp_s': f['timestamp_s'],
                                 'native_polygon_xy': poly, 'panorama_polygon_xy': pp.tolist(),
                                 'proposed_instance_id': g['instance_id'] if g else None,
                                 'uncertain_association': g.get('uncertain_association', True) if g else True})
                polygons.append(pp)
        components.append({'stem': stem, 'members': members, 'M': M, 'records': records, 'overlays': overlays, 'polygons': polygons,
                           'canvas': canvas, 'coverage': coverage, 'owner': owner_global, 'width': width, 'height': height,
                           'rectification': rectification, 'alignment': alignment, 'segment': seg_id,
                           'split_reason': 'First tracked segment' if n == 0 else 'Frame motion was lost (blur / no overlap); a new tracking segment starts a separate plane'})
        cache.clear()

    if not components:
        raise ValueError('No saved frame could be tracked and stitched (' + (unregistered[0]['reason'] if unregistered else 'no readable frames') + ')')

    # --- 3. markers: one per book group, placed in the component that holds most of its sightings
    progress('markers', groups=len(groups))
    sharp = {f['frame_id']: (f.get('quality') or {}).get('sharpness') or 0. for f in frames}
    obs_index = {o['observation_id']: o for o in observations}
    placed, unplaced = {}, []
    for g in groups:
        per_component = []
        for ci, comp in enumerate(components):
            frames_here = {f['frame_id'] for f in comp['members']}
            sightings = []
            for oid in g.get('observation_ids', []):
                o = obs_index.get(oid)
                if o is None or o['frame_id'] not in frames_here:
                    continue
                sightings.append(o)
            if sightings:
                per_component.append((len(sightings), sum(sharp[o['frame_id']] for o in sightings), ci, sightings))
        if not per_component and not observations and g.get('evidence_refs'):      # no candidates.json: only the representative sighting is known
            for ci, comp in enumerate(components):
                frames_here = {f['frame_id'] for f in comp['members']}
                s = [dict(e, kind=g['_kind'], observation_id=e.get('observation_id')) for e in g['evidence_refs'] if e['frame_id'] in frames_here and e.get('bbox_xyxy')]
                if s:
                    per_component.append((len(s), 0., ci, s))
        if not per_component:
            all_ids = g.get('observation_ids', [])
            known = [obs_index[i]['frame_id'] for i in all_ids if i in obs_index]
            reason = ('no sighting data (candidates.json missing or lacks its observations)' if not known
                      else 'every sighting is in a frame that could not be registered: ' + ', '.join(sorted(set(known))[:6]))
            unplaced.append({'group_id': g['instance_id'], 'kind': g['_kind'], 'reason': reason, 'sightings': len(all_ids)})
            continue
        per_component.sort(key=lambda t: (-t[0], -t[1]))
        count, _, ci, sightings = per_component[0]
        comp = components[ci]
        centres, rep = [], None
        for o in sightings:
            H = comp['M'][o['frame_id']]
            pp = project(_polygon(o), H)
            centres.append(pp.mean(axis=0))
            key = (sharp[o['frame_id']], cv2.contourArea(pp.astype(np.float32)))
            if rep is None or key > rep[0]:
                rep = (key, o, pp)
        centres = np.asarray(centres)
        median = np.median(centres, axis=0)
        spread = float(np.max(np.linalg.norm(centres - median, axis=1)))
        _, rep_o, rep_pp = rep
        other = sum(c for c, _, cj, _ in per_component[1:])
        marker = {'group_id': g['instance_id'], 'kind': g['_kind'], 'x': round(float(median[0]), 2), 'y': round(float(median[1]), 2),
                  'polygon': np.round(rep_pp, 1).tolist(), 'representative_observation_id': rep_o['observation_id'],
                  'sightings_in_component': int(count), 'sightings_total': len(g.get('observation_ids', [])), 'sightings_elsewhere': int(other),
                  'spread_px': round(spread, 1), 'frames': [o['frame_id'] for o in sightings][:12], 'component_id': comp['stem'],
                  'method': 'median of the group sightings projected through the tracked frame transforms'}
        placed.setdefault(ci, []).append(marker)

    # --- 4. write
    progress('writing', components=len(components))
    video_sha = manifest['data'].get('input', {}).get('sha256')
    source_frames_top = []
    for i, f in enumerate(frames):
        fid = f['frame_id']
        path = resolve_evidence(root, f['image_path'])
        digest = sha(path) if path.is_file() else None
        reg = next((c for c in components if fid in c['M']), None)
        reason = next((u['reason'] for u in unregistered if u['frame_id'] == fid), None)
        source_frames_top.append({**f, 'sha256': digest, 'source_image_path': str(path), 'source_run': root.name, 'source_index': i + 1,
                                  'registered': reg is not None, 'component_id': reg['stem'] if reg else None, 'unregistered_reason': None if reg else reason})
    digests = {s['frame_id']: s['sha256'] for s in source_frames_top}
    final_components, atlas_cards = [], []
    marker_total = item_total = 0
    for ci, comp in enumerate(components):
        stem = comp['stem']
        book_markers = [m for m in placed.get(ci, []) if m['kind'] == 'book']
        item_markers = [m for m in placed.get(ci, []) if m['kind'] == 'item']
        marker_total += len(book_markers)
        item_total += len(item_markers)
        for rec in comp['records']:
            rec['sha256'] = digests[rec['frame_id']]
        for o in comp['overlays']:
            o['frame_sha256'] = digests[o['frame_id']]
        detected = comp['canvas'].copy()
        for poly in comp['polygons']:
            cv2.polylines(detected, [np.rint(poly).astype(np.int32)], True, (60, 220, 40), 1, cv2.LINE_AA)
        marked = _draw_marker_image(comp['canvas'], book_markers, [m['polygon'] for m in book_markers])
        for suffix, img in [('.jpg', comp['canvas']), ('_detected.jpg', detected), ('_groups.jpg', marked), ('_coverage.png', comp['coverage']), ('_source_index.png', comp['owner'])]:
            ok = cv2.imwrite(str(out / (stem + suffix)), img, [cv2.IMWRITE_JPEG_QUALITY, 92] if suffix.endswith('.jpg') else [])
            if not ok:
                raise ValueError('Image write failed: ' + stem + suffix)
        files = {k: {'path': stem + s, 'sha256': sha(out / (stem + s))} for k, s in
                 [('image', '.jpg'), ('observations', '_detected.jpg'), ('groups', '_groups.jpg'), ('coverage', '_coverage.png'), ('pixel_source', '_source_index.png')]}
        group_overlays = []
        by_instance = {g['instance_id']: g for g in groups}
        for m in book_markers + item_markers:
            g = by_instance[m['group_id']]
            group_overlays.append({'proposed_instance_id': m['group_id'], 'kind': g['_kind'] if g['_kind'] == 'item' else 'book',
                                   'uncertain_association': g.get('uncertain_association', True), 'representative_observation_key': m['representative_observation_id'],
                                   'panorama_polygon_xy': m['polygon'], 'observation_keys': [], 'all_observation_ids': g.get('observation_ids', [])})
        final_components.append({'component_id': stem, 'plane_id': stem + '_unverified_plane', 'segment_id': comp['segment'],
                                 'status': 'single_frame' if len(comp['members']) == 1 else 'planar_diagnostic_stitch', 'plane_verified': False,
                                 'split_reason': comp['split_reason'], 'image': stem + '.jpg', 'overlay_image': stem + '_detected.jpg',
                                 'group_overlay_image': stem + '_groups.jpg', 'coverage_mask': stem + '_coverage.png', 'pixel_source_index': stem + '_source_index.png',
                                 'width_px': comp['width'], 'height_px': comp['height'], 'files': files, 'source_frames': comp['records'],
                                 'overlays': comp['overlays'], 'group_overlays': group_overlays, 'markers': book_markers, 'item_markers': item_markers,
                                 'rectification': comp['rectification'], 'alignment': comp['alignment'], 'coverage_fraction': float(np.count_nonzero(comp['coverage']) / comp['coverage'].size),
                                 'coordinate_system': CONVENTION,
                                 'coverage_provenance': 'uint16 source-index PNG: 0 uncovered, index maps to top-level source_frames; centre-weighted seam owner',
                                 'physical_count_verified': False})
        thumb_scale = min(1., 550 / comp['width'], 350 / comp['height'])
        thumb = cv2.resize(marked, (max(1, round(comp['width'] * thumb_scale)), max(1, round(comp['height'] * thumb_scale))))
        card = np.full((390, 570, 3), 25, np.uint8)
        card[35:35 + thumb.shape[0], 10:10 + thumb.shape[1]] = thumb
        cv2.putText(card, f'{stem}: {len(comp["members"])} views, {len(book_markers)} book markers', (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        atlas_cards.append(card)
    book_groups = [g for g in groups if g['_kind'] == 'book']
    item_groups = [g for g in groups if g['_kind'] == 'item']
    book_count = ((inventory or {}).get('data', {}).get('counts', {}) or {}).get('book_count', len(book_groups)) if inventory else None
    unplaced_books = [u for u in unplaced if u['kind'] == 'book']
    if inventory is not None and marker_total + len(unplaced_books) != len(book_groups):
        raise ValueError(f'Marker invariant broken: {marker_total} placed + {len(unplaced_books)} unplaced != {len(book_groups)} book groups')
    counts = {'raw_observations': len(observations), 'by_kind': dict(Counter(o['kind'] for o in observations)),
              'proposed_physical_groups': len(groups) if inventory else None, 'groups_by_kind': dict(Counter(g['_kind'] for g in groups)),
              'book_count': book_count, 'book_markers': marker_total, 'unplaced_book_groups': len(unplaced_books),
              'item_markers': item_total, 'unplaced_item_groups': len([u for u in unplaced if u['kind'] == 'item']),
              'registered_frames': sum(len(c['members']) for c in components), 'frames_total': len(frames),
              'projected_observations': sum(len(c['overlays']) for c in components), 'verified_physical_book_count': None}
    atlas_files = []
    for start in range(0, len(atlas_cards), 12):
        page = atlas_cards[start:start + 12]
        sheet = np.full((math.ceil(len(page) / 3) * 390, 1710, 3), 25, np.uint8)
        for j, card in enumerate(page):
            y, x = divmod(j, 3)
            sheet[y * 390:(y + 1) * 390, x * 570:(x + 1) * 570] = card
        name = f'atlas_{start // 12 + 1:02d}.jpg'
        cv2.imwrite(str(out / name), sheet)
        atlas_files.append(name)
    unstitched = sorted({u['frame_id'] for u in unregistered} | {r['frame_id'] for c in final_components if c['status'] == 'single_frame' for r in c['source_frames']})
    artifact = {'schema_version': SCHEMA_VERSION, 'artifact_type': 'diagnostic_panorama', 'run_id': out.name, 'scan_label': scan_label or manifest['data'].get('input', {}).get('scan_label'),
                'fixture': bool(manifest.get('fixture') or (candidates or {}).get('fixture')),
                'method': 'LK-chain frame motion shared with the association (affine steps, tracking segments) + automatic fronto-parallel rectification (spine vanishing point) + centre-weighted seam blending',
                'source_run': root.name, 'source_run_path': str(root), 'source_candidate_run_id': (candidates or {}).get('run_id'), 'source_video_sha256': video_sha,
                'source_manifest_sha256': sha(root / 'manifest.json'), 'source_candidates_sha256': sha(root / 'candidates.json') if candidates else None,
                'source_inventory_sha256': sha(root / 'inventory.json') if inventory else None,
                'status': 'partial', 'coordinate_system': CONVENTION, 'components': final_components,
                'source_frames': source_frames_top, 'registration_edges': edges, 'limits': LIMITS,
                'unregistered_frames': unregistered, 'unstitched_frame_ids': unstitched, 'atlas_images': atlas_files,
                'atlas_status': 'tiled_component_atlas_not_one_geometric_room',
                'book_count': book_count, 'markers_total': marker_total, 'unplaced_groups': unplaced, 'item_markers_total': item_total,
                'marker_policy': 'exactly one marker per book group of the run\'s inventory.json; placed groups + unplaced groups = book_count; items are separate',
                'raw_observation_count': len(observations), 'selected_frames': sum(f.get('selected', False) for f in frames), 'sampled_frames': len(frames), 'counts': counts,
                'association_evidence': {'inventory': 'inventory.json of the source run (not recomputed)', 'scope': 'the run\'s own groups; titles unused'},
                'metrics': {'elapsed_s': time.perf_counter() - started, 'tracking_s': tracking_s, 'fresh_detector_calls': 0, 'api_calls': 0, 'api_cost_usd': 0,
                            'registered_frames': counts['registered_frames'], 'frames_total': len(frames),
                            'registered_ratio': round(counts['registered_frames'] / len(frames), 4), 'components': len(final_components)},
                'limitations': ['Unverified planar sections: drift, parallax and seam blending remain; the shelf face is rectified with an ASSUMED focal length (no metric scale)',
                                'Markers show where the run\'s proposed groups were seen; they locate, they do not count or verify physical books',
                                'Unregistered frames (tracking holds) are listed and not drawn',
                                'No OCR/identity import by matching ID strings; source-run and content-hash join required']}
    save_json(out / 'panorama.json', artifact)
    save_json(out / 'diagnostics.json', {'status': 'partial', 'counts': counts, 'component_count': len(final_components), 'unregistered_frames': unregistered,
                                         'rectification': {c['component_id']: c['rectification'] for c in final_components}, 'metrics': artifact['metrics']})
    cards = ''.join(f'<h2>{c["component_id"]}: {len(c["source_frames"])} views / {c["status"]}</h2><a href="{c["group_overlay_image"]}"><img style="max-width:100%" src="{c["group_overlay_image"]}"></a>' for c in final_components)
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>panorama</title><body style="background:#171717;color:#eee;font:16px system-ui;padding:20px"><h1>'
                                    + html.escape(root.name) + f' - planar panorama, {marker_total} book markers (book_count {book_count})</h1><p>Unscaled; markers locate proposed groups, they do not verify counts.</p>' + cards + '</body>')
    print(json.dumps({'output': str(out), 'components': len(final_components), 'counts': counts, 'elapsed_s': round(artifact['metrics']['elapsed_s'], 2)}, indent=2), flush=True)
    progress('done')
    return out / 'panorama.json'


def _preview(members, Mseg, unit, image, long_side=900):
    """Low-resolution last-wins grey preview of a component: (canvas, plane->canvas 3x3, plane centre) or None."""
    fps = [project(corners(f), Mseg[f['frame_id']]) for f in members]
    pts = np.vstack(fps)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    s = min(long_side / max(span), unit)
    W, H = int(math.ceil(span[0] * s)), int(math.ceil(span[1] * s))
    if W < 20 or H < 20 or W * H > 4_000_000:
        return None
    A = np.array([[s, 0, -lo[0] * s], [0, s, -lo[1] * s], [0, 0, 1.]])
    canvas = np.zeros((H, W), np.uint8)
    for f in members[:: max(1, len(members) // 40)]:
        img = image(f['frame_id'])
        if img is None:
            continue
        g = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        Mc = A @ Mseg[f['frame_id']] @ np.diag([2., 2., 1.])
        warped = cv2.warpPerspective(g, Mc, (W, H))
        mask = cv2.warpPerspective(np.full(g.shape, 255, np.uint8), Mc, (W, H), flags=cv2.INTER_NEAREST) > 0
        canvas[mask] = warped[mask]
    return canvas, A, pts.mean(axis=0)


def _leveling_from_preview(members, Mseg, unit, image):
    """Levelling transform (3x3, plane pixels) of the rectified component from its long edges, plus info."""
    prev = _preview(members, Mseg, unit, image)
    if prev is None:
        return np.eye(3), {'skipped': 'preview size'}
    canvas, A, centre = prev
    L, info = leveling_matrix(canvas)
    C = np.array([[1., 0, centre[0]], [0, 1, centre[1]], [0, 0, 1]])
    return C @ L @ np.linalg.inv(C), info


def _yaw_from_preview(members, Mseg, unit, image, ref, focal):
    """Horizontal vanishing point of the shelf boards in the pitch-rectified component -> yaw homography (or None) + info."""
    prev = _preview(members, Mseg, unit, image)
    if prev is None:
        return None, {'skipped': 'preview size'}
    canvas, A, _ = prev
    lines, mids, lens = horizontal_lines(canvas, np.linalg.inv(A))
    info = {'horizontal_lines': int(len(lines))}
    if len(lines) < LIMITS['yaw_min_lines']:
        info['reason'] = f'only {len(lines)} long horizontal edges (need {LIMITS["yaw_min_lines"]})'
        return None, info
    vp = vanishing_point(lines, mids, tol_deg=LIMITS['yaw_tol_deg'])
    if vp is None:
        info['reason'] = 'no vanishing point'
        return None, info
    v, inl = vp
    info['inlier_fraction'] = round(float(inl.mean()), 3)
    # A far-away vanishing point also "explains" lines that are simply parallel (two noisy parallel lines always meet somewhere):
    # only correct the yaw when a finite vanishing point beats the plain parallel hypothesis clearly.
    angles = np.degrees(np.arctan2(lines[:, 0], -lines[:, 1]))
    angles = (angles + 90) % 180 - 90
    parallel = int((np.abs(angles - weighted_median(angles, lens)) < LIMITS['yaw_tol_deg']).sum())
    info['parallel_inliers'] = parallel
    info['vanishing_point_inliers'] = int(inl.sum())
    if parallel >= 0.85 * inl.sum():
        info['reason'] = 'the long horizontal edges are already parallel'
        return None, info
    if inl.sum() < LIMITS['yaw_min_lines'] or inl.mean() < LIMITS['yaw_min_inlier_fraction']:
        info['reason'] = 'horizontal vanishing point unreliable'
        return None, info
    H, yaw = yaw_homography(v, ref['width_px'], ref['height_px'], focal)
    info['yaw_deg'] = round(yaw, 2)
    if H is None:
        info['reason'] = 'boards already parallel (or implausible yaw)'
        return None, info
    return H, info


def audit(path):
    """Verify output file hashes, frame hashes and every saved forward/inverse transform."""
    started = time.perf_counter()
    path = Path(path).resolve()
    out = path.parent
    d = read_json(path)
    root = Path(d['source_run_path'])
    if sha(root / 'manifest.json') != d['source_manifest_sha256']:
        raise ValueError('Input artifact hash mismatch: manifest.json')
    maximum_error, mapped = 0., set()
    sources = {f['frame_id']: f for f in d['source_frames']}
    for c in d['components']:
        for file in c['files'].values():
            if sha(out / file['path']) != file['sha256']:
                raise ValueError('Derived output image hash mismatch')
        for f in c['source_frames']:
            if f['sha256'] != sources[f['frame_id']]['sha256'] or sha(resolve_evidence(root, f['image_path'])) != f['sha256']:
                raise ValueError('Native source hash mismatch')
            H, inv = np.array(f['original_to_panorama_3x3']), np.array(f['panorama_to_original_3x3'])
            if not np.allclose(project(corners(f), H), f['footprint'], atol=1e-5):
                raise ValueError('Saved native footprint projection mismatch')
            maximum_error = max(maximum_error, float(np.max(np.abs(project(project(corners(f), H), inv) - corners(f)))))
            mapped.add(f['frame_id'])
        for m in c.get('markers', []):
            if not (0 <= m['x'] <= c['width_px'] and 0 <= m['y'] <= c['height_px']):
                raise ValueError('Marker outside its component: ' + m['group_id'])
    if maximum_error > 1e-4:
        raise ValueError('Inverse/round-trip mismatch')
    ids = [m['group_id'] for c in d['components'] for m in c.get('markers', [])] + [u['group_id'] for u in d['unplaced_groups'] if u['kind'] == 'book']
    if len(ids) != len(set(ids)) or (d.get('book_count') is not None and len(ids) != d['book_count']):
        raise ValueError('Marker invariant violated: one marker (or unplaced entry) per book group')
    return {'status': 'verified_artifact_integrity', 'panorama_sha256': sha(path), 'run_id': d['run_id'], 'source_frames_checked': len(mapped),
            'max_roundtrip_error_px': maximum_error, 'markers_checked': len(ids), 'elapsed_s': time.perf_counter() - started}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--scan-label')
    p.add_argument('--registry', help='accepted for compatibility; not needed')
    p.add_argument('--audit', type=Path)
    p.add_argument('--chain', choices=['homography', 'association'], default='association')
    p.add_argument('--no-align', action='store_true', help='keep the tracked chain without the global alignment refinement')
    a = p.parse_args()
    if a.audit:
        print(json.dumps(audit(a.audit), indent=2))
        return
    if not (a.source_run and a.out):
        p.error('--source-run and --out required unless --audit supplied')
    build(a.source_run, a.out, a.scan_label, chain=a.chain, align=not a.no_align)


if __name__ == '__main__':
    main()
