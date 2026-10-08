"""Cross-view spine association by motion tracking + appearance verification.

Why not the old frame-pair feature registration: spines are repeated vertical texture, so
ORB/SIFT ratio-test matching rarely passes (YugoTest2: 260/350 attempts "insufficient
distinctive matches") and a single planar warp leaves a few pixels of parallax error, which the
strict appearance gate then turns into "uncertain" (YugoTest1: 642 rejected pairs).

Method (no learned parameters, nothing tuned against a count):
1. Track consecutive saved frames (pyramidal Lucas-Kanade + RANSAC homography) and chain them, so
   short-range motion is known even when feature matching between distant frames fails.
2. For detection-frame pairs that overlap in chain coordinates and are close in time, estimate the
   residual translation (parallax/drift) by voting over nearby spine centres.
3. Match one-to-one (greedy by appearance) within `tol_widths` spine widths after the residual shift,
   but only when the warped spine appearance agrees (`min_appearance`, set from the similarity of
   known-distinct adjacent spines, ~98th percentile).
4. Union with same-frame exclusion (two sightings from one frame are never one book). Titles are never
   used. Anything unmatched stays a separate, uncertain instance.
"""
from __future__ import annotations

import collections
import math

import cv2
import numpy as np

DEFAULTS = {
    "tol_widths": 0.5,            # match radius after residual shift, in spine widths
    "item_tol": 0.3,              # non-book match radius as a fraction of the object size
    "item_min_appearance": 0.4,
    "tol_across_min_length_frac": 0.08,
    "tol_length": 0.35,           # along-spine match radius as a fraction of spine length
    "min_appearance": 0.5,        # Lab patch correlation; ~98th pct of known-distinct adjacent spines
    "max_gap_s": 6.0,             # bound chain drift
    "min_overlap_frac": 0.10,     # footprint overlap (chain coords) required before pairing frames
    "min_votes": 4,               # residual-shift evidence
    "geometry_weight": 0.2,       # score = appearance - geometry_weight * normalised distance
    "keep_runner_ups": True,      # record one-to-one runner-ups as ineligible competitors (near-tie veto in dedup.associate)
    "near_miss_factor": 2.0,      # record (never merge) look-alike neighbours up to 2x the gate, so ambiguity is visible
    "near_gap_s": 0.45,           # short-gap rule below (measured tracked residual median 0.04 spine widths for gaps <0.5 s)
    "near_tol_widths": 0.35,      # tight geometric match radius for the short-gap rule
    "near_min_appearance": 0.2,   # blur/zoom can lower correlation; geometry carries the link
    "track_scale": 0.5,           # downscale for tracking
    "loop_closure": False,        # link revisits (>loop_min_gap_s apart) by appearance-gated constellation matching
    "loop_min_gap_s": 2.0,
    "loop_max_frames": 90,        # sharpest detection frames used for loop closure
    "loop_scale": 0.67,           # SIFT image scale
    "loop_tol_widths": 0.75,      # match radius after the (verified) homography, in spine widths
    "loop_min_obs": 6,
    "loop_min_inliers": 30,       # geometric inliers required for a revisit homography
    "loop_min_scale": 0.4,        # accepted revisit zoom (sqrt det of the homography), far views vs close-ups
    "loop_max_scale": 2.5,
    # Thin/flat boxes (lying or stacked books). Applied ONLY to pairs of boxes with short/long side < flat_ratio whose long axis is
    # >= flat_min_angle_deg away from the shelf "up" direction (dominant long-axis direction of all book boxes in chain coordinates),
    # and only to boxes the upright rule above left unmatched, so upright-spine behaviour is unchanged.
    "flat_rule": True,
    "flat_pair_ratio_max": 0.5,      # the SAME lying book gets a thicker rectangle in some frames (measured 0.12-0.44 across frames), so the partner of a thin box may be up to this
    "flat_ratio": 0.34,              # short/long side; = the aspect >= 3:1 admission of the flat supplement (first estimate ~0.3; measured flat boxes span 0.12-0.36)
    "flat_min_angle_deg": 50.0,
    "flat_min_along_overlap": 0.5,   # fraction of the shorter box's length that must overlap along the long axis after the residual shift
    "flat_across_length_frac": 0.12,  # across tolerance floor as a fraction of box LENGTH (thickness of a flat box is a poor scale)
    "flat_min_appearance": 0.3,      # thin flat patches correlate ~half as well as upright spines (measured on same-book pairs)
    "flat_near_min_appearance": 0.15,  # gap <= near_gap_s and tight geometry
    "stretch_appearance": True,      # upright spines: when whole-box appearance fails the gate, compare only the stretch of the spine both boxes cover
    "stretch_min_frac": 0.5,         # the common stretch must be at least this fraction of the shorter box
    # Border-clipped partial spines (post-association, clipped_partial_links): a group whose EVERY sighting touches the frame border is a stub of a
    # book whose full-length sightings sit in another group (the stub's centre and visible stretch differ from the full box, so pairwise gates miss it).
    "clipped_partials": True,
    "clip_margin_frac": 0.005,       # a box corner within this fraction of the shorter frame side of the border counts as clipped (detector boxes are cut AT the border)
    "stub_inside_min": 0.5,          # fraction of the motion-projected stub outline that must lie inside one sighting outline of the partner
    "stub_support_min": 0.5,         # that containment must hold in at least this fraction of the partner's evaluable sightings (chance alignment of a drifted chain is not consistent)
    "stub_margin": 0.25,             # the best partner must beat the runner-up by this much (near-tie veto, as in dedup.associate)
    "stub_length_max_frac": 0.85,    # a stub is clearly shorter than its partner (its longest sighting <= this fraction of the partner's longest unclipped one)
    # Singleton merge (post-association, singleton_merge_links): on blurred/jittery capture the appearance gate fails and most groups are single
    # sightings. A single sighting joins the group whose sighting it lands on after the tracked motion (geometry only, same segment, never co-visible).
    # Used for the UPPER end of the count range, not for the strict association.
    "singleton_merge": True,
    "singleton_min_iou": 0.35,       # IoU of the motion-projected singleton outline with the partner sighting (adjacent spines overlap ~0 when tracking holds)
    "singleton_margin": 0.15,        # best partner must beat the runner-up by this much (near-tie veto)
    # Tracking recovery (chain_transforms recover=True): live phone frames jump 200-500 px between saves; a failed LK step is retried from a
    # phase-correlation seed, and a segment break is re-joined to the previous segment's anchor when one of the next bridge_frames frames registers to it.
    "recover_tracking": True,
    # Appearance-guided residual (_guided_residual): at close range the tracked global motion mis-states the motion of protruding book
    # stacks (parallax; measured residual scale 0.77-0.93 on YugoTest6 close-ups), so true partners land 1-3 spine widths away. Spine pairs within
    # guided_radius_lengths spine lengths whose appearance passes min_appearance vote (RANSAC) for a residual affine; it replaces the translation only
    # when it puts MORE appearance-passing pairs inside the unchanged geometric gate.
    "guided_residual": True,
    "guided_radius_lengths": 0.6,
    "guided_min_inliers": 6,
    "guided_max_scale": 1.5,
    "bridge_frames": 8,
    "seed": 0,
}


def _gray(image, scale):
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _step_transform(p0, p1):
    """Per-step frame motion: affine by default (stable under JPEG re-encoding); a full homography only
    when it fits the shared inliers clearly better (>=30% lower reprojection error). Returns (H3x3, inliers)."""
    A, ma = cv2.estimateAffine2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None or ma is None or not np.isfinite(A).all() or abs(np.linalg.det(A[:, :2])) < 1e-3:
        return None, 0
    H, m = np.vstack([A, [0, 0, 1]]), ma
    Hh, mh = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    valid_h = (Hh is not None and mh is not None and np.asarray(Hh).shape == (3, 3) and np.isfinite(Hh).all()
               and abs(np.linalg.det(np.asarray(Hh, float))) > 1e-6 and np.asarray(mh).shape == np.asarray(ma).shape)
    if valid_h and mh.sum() >= 0.9 * ma.sum():
        sel = (ma.ravel() == 1) & (mh.ravel() == 1)
        if sel.sum() >= 25:
            P, Q = p0[sel].reshape(-1, 2), p1[sel].reshape(-1, 2)

            def err(M):
                q = np.hstack([P, np.ones((len(P), 1))]) @ M.T
                q = q[:, :2] / q[:, 2:3]
                return float(np.sqrt(((q - Q) ** 2).sum(1)).mean())
            if err(Hh) < 0.7 * err(H):
                H, m = Hh, mh
    return H, int(m.sum())


_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def _lk_step(prev, cur, shift=None):
    """prev -> cur motion from forward-backward-checked pyramidal LK on corners of prev; shift: optional initial
    displacement for every corner (from _coarse_shift). Returns (H3x3 in these pixels or None, inliers)."""
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=1500, qualityLevel=0.005, minDistance=6, blockSize=5)
    if p0 is None or len(p0) <= 30:
        return None, 0
    if shift is None:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(31, 31), maxLevel=4, criteria=_CRIT)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, winSize=(31, 31), maxLevel=4)
    else:
        init = (p0 + np.asarray(shift, np.float32)).astype(np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, init, winSize=(31, 31), maxLevel=4, criteria=_CRIT, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, p0.copy(), winSize=(31, 31), maxLevel=4, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pb - p0, axis=2).ravel() < 1.0)
    if good.sum() < 30:
        return None, 0
    return _step_transform(p0[good], p1[good])


def _coarse_shift(prev, cur, factor=4):
    """Global prev -> cur translation by phase correlation on a further `factor`x downscaled copy (in prev/cur pixels).
    Only a starting point for LK: the step is accepted by the same LK + RANSAC test as any other step."""
    a = cv2.resize(prev, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(cur, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA).astype(np.float32)
    if a.shape != b.shape or min(a.shape) < 8:
        return None
    (dx, dy), _ = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    if not (np.isfinite(dx) and np.isfinite(dy)) or abs(dx) > a.shape[1] or abs(dy) > a.shape[0]:
        return None
    return (dx * factor, dy * factor)


def _photometric_agreement(prev, cur, H):
    """Normalised correlation of prev warped by H with cur over their overlap (None when the overlap is under 5%). A recovered step or a bridge
    must pass this as well as the LK test: on smooth texture LK can lock onto an unrelated view, which correlates ~0 after warping."""
    h, w = cur.shape[:2]
    warped = cv2.warpPerspective(prev, H, (w, h))
    mask = cv2.warpPerspective(np.ones_like(prev), H, (w, h)) > 0
    if mask.mean() < 0.05:
        return None
    x, y = warped[mask].astype(np.float64), cur[mask].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else None


def _compose(T, H, S, Si):
    """Anchor plane transform T and anchor -> current step H (tracking pixels) -> current frame's plane transform, or None."""
    if H is None:
        return None
    try:
        cand = T @ np.linalg.inv(Si @ H @ S)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(cand).all() or abs(cand[2, 2]) <= 1e-9:
        return None
    cand = cand / cand[2, 2]
    return cand if np.isfinite(cand).all() else None


def chain_transforms(frame_ids, load_image, scale=0.5, max_hold_run=3, recover=False, bridge_frames=8, report=None, min_agreement=0.3):
    """frame_ids in time order -> ({frame_id: 3x3 frame->segment-plane}, {frame_id: status}); see segments_from_status.

    A failed step keeps the previous anchor frame (so one blurred frame can be bridged); after `max_hold_run`
    consecutive failures a NEW segment starts, because the motion across the gap is unknown and must not be
    treated as identity. Statuses: start | ok | hold:<reason> | segment_start.

    recover=True (the live phone stream moves 200-500 px between saved frames, beyond what LK finds from a zero start):
    1. a failed step is retried once with LK started from a phase-correlation estimate of the global shift; it is accepted by the same
       test as a normal step (>= 25 RANSAC inliers of forward-backward-checked tracks), so the estimate only seeds the search;
    2. after a segment break, each of the next `bridge_frames` tracked frames is registered (same seeded LK test) to the last anchor of the
       previous segment; on success the new segment is re-expressed in the old plane and the break disappears (its start becomes 'ok').
    Both must also pass a photometric check (_photometric_agreement >= min_agreement; ordinary tracked steps on YugoTest6 reach >= 0.45 at their
    1st percentile, an unrelated view ~0). Every recovered step and bridge is listed in `report` (dict, optional) with its inliers and agreement."""
    S = np.diag([scale, scale, 1.0])
    Si = np.diag([1 / scale, 1 / scale, 1.0])
    out, status, segment = {}, {}, {}
    prev, T, seg, run = None, np.eye(3), 0, 0
    pending_segment_start = False
    lost, seg_frames = None, []  # lost: (anchor image, anchor plane transform, segment id, frames left to try) of the segment that broke
    rec = {"recovered_steps": [], "bridges": [], "bridge_attempts": 0} if report is None else report.setdefault("recovery", {"recovered_steps": [], "bridges": [], "bridge_attempts": 0})

    def break_segment():
        nonlocal lost, seg_frames
        if recover and prev is not None:
            lost = (prev, T.copy(), seg, bridge_frames)
        seg_frames = []

    def try_bridge(fid, cur):
        nonlocal lost, T, seg
        rec["bridge_attempts"] += 1
        shift = _coarse_shift(lost[0], cur)
        H, n = _lk_step(lost[0], cur, shift) if shift is not None else (None, 0)
        if n < 25:
            H, n = _lk_step(lost[0], cur)
        agree = _photometric_agreement(lost[0], cur, H) if H is not None and n >= 25 else None
        old = _compose(lost[1], H, S, Si) if agree is not None and agree >= min_agreement else None
        if old is None:
            lost = None if lost[3] <= 1 else (*lost[:3], lost[3] - 1)
            return
        C = old @ np.linalg.inv(out[fid])
        for f in seg_frames:
            m = C @ out[f]
            out[f], segment[f] = m / m[2, 2], lost[2]
            if status[f] == "segment_start":
                status[f] = "ok"
        T = out[fid].copy()
        rec["bridges"].append({"frame_id": fid, "inliers": int(n), "agreement": round(agree, 3), "frames_rejoined": len(seg_frames)})
        seg, lost = lost[2], None

    for fid in frame_ids:
        image = load_image(fid)
        if image is None:
            run += 1
            if prev is not None and run >= max_hold_run:
                break_segment()
                seg += 1
                T, prev, run = np.eye(3), None, 0
                out[fid], status[fid], segment[fid] = T.copy(), "hold:no_image", seg
                pending_segment_start = True
            else:
                out[fid], status[fid], segment[fid] = T.copy(), "hold:no_image", seg
            seg_frames.append(fid)
            continue
        cur = _gray(image, scale)
        if prev is None:
            out[fid], status[fid], segment[fid] = T.copy(), "segment_start" if pending_segment_start else "start", seg
            pending_segment_start = False
            prev = cur
            seg_frames.append(fid)
            continue
        H, n = _lk_step(prev, cur)
        T_new = _compose(T, H, S, Si) if H is not None and n >= 25 else None
        if T_new is None and recover:
            shift = _coarse_shift(prev, cur)
            if shift is not None:
                H, n = _lk_step(prev, cur, shift)
                agree = _photometric_agreement(prev, cur, H) if H is not None and n >= 25 else None
                T_new = _compose(T, H, S, Si) if agree is not None and agree >= min_agreement else None
                if T_new is not None:
                    rec["recovered_steps"].append({"frame_id": fid, "inliers": int(n), "agreement": round(agree, 3),
                                                   "seed_shift_px": [round(shift[0] / scale, 1), round(shift[1] / scale, 1)]})
        if T_new is not None:
            T = T_new
            out[fid], status[fid], segment[fid] = T.copy(), "ok", seg
            prev, run = cur, 0
            seg_frames.append(fid)
            if lost is not None:
                try_bridge(fid, cur)
        else:
            run += 1
            if run >= max_hold_run:  # unknown motion: new segment, fresh plane
                break_segment()
                seg += 1
                T, prev, run = np.eye(3), cur, 0
                out[fid], status[fid], segment[fid] = T.copy(), "segment_start", seg
            else:  # keep the previous anchor; this frame is placed provisionally and flagged
                out[fid], status[fid], segment[fid] = T.copy(), "hold:tracking_failed", seg
            seg_frames.append(fid)
    return out, status


def segments_from_status(frame_ids, status):
    """Segment id per frame: increments at every 'segment_start' (unknown motion across a tracking gap)."""
    seg, out = 0, {}
    for fid in frame_ids:
        if status.get(fid) == "segment_start":
            seg += 1
        out[fid] = seg
    return out


def _proj(points, m):
    p = np.hstack([np.asarray(points, np.float64), np.ones((len(points), 1))]) @ m.T
    return p[:, :2] / p[:, 2:3]


def _patch(image, poly):
    """Spine quad warped to 24x96 Lab, long axis vertical."""
    p = np.asarray(poly, np.float32)
    e = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
    src = np.array([p[1], p[2], p[3], p[0]], np.float32) if e[0] >= e[1] else p
    if np.linalg.norm(src[0] - src[1]) > np.linalg.norm(src[1] - src[2]):
        src = np.array([src[1], src[2], src[3], src[0]], np.float32)
    M = cv2.getPerspectiveTransform(src, np.array([[0, 0], [23, 0], [23, 95], [0, 95]], np.float32))
    return cv2.cvtColor(cv2.warpPerspective(image, M, (24, 96), flags=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)


def appearance(a, b):
    """Mean per-channel normalised correlation; best of both 180-degree orientations."""
    best = -1.0
    for flip in (0, 1):
        bb = b[::-1, ::-1] if flip else b
        s = []
        for ch in range(3):
            x = a[:, :, ch].ravel() - a[:, :, ch].mean()
            y = bb[:, :, ch].ravel() - bb[:, :, ch].mean()
            d = np.linalg.norm(x) * np.linalg.norm(y)
            s.append(float(x @ y / d) if d > 1e-6 else 0.0)
        best = max(best, float(np.mean(s)))
    return best


def _sub_patch(image, quad, T, u, lo, hi, shift):
    """Lab patch (24x96, long axis vertical) of only the stretch [lo, hi] (chain coordinates along unit vector u) of a box.
    Boxes of one flat book cover different stretches of it in different frames, so comparing whole-box patches mixes different parts of the cover."""
    p = np.asarray(quad, np.float32)
    e = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
    src = np.array([p[1], p[2], p[3], p[0]], np.float32) if e[0] >= e[1] else p
    if np.linalg.norm(src[0] - src[1]) > np.linalg.norm(src[1] - src[2]):
        src = np.array([src[1], src[2], src[3], src[0]], np.float32)
    S = _proj(src, T) + shift
    start, end = (S[0] + S[1]) / 2 @ u, (S[3] + S[2]) / 2 @ u
    if abs(end - start) < 1e-6:
        return None
    f0, f1 = sorted([(lo - start) / (end - start), (hi - start) / (end - start)])
    f0, f1 = max(f0, 0.0), min(f1, 1.0)
    if f1 - f0 < 0.1:
        return None
    q = np.array([src[0] + f0 * (src[3] - src[0]), src[1] + f0 * (src[2] - src[1]), src[1] + f1 * (src[2] - src[1]), src[0] + f1 * (src[3] - src[0])], np.float32)
    M = cv2.getPerspectiveTransform(q, np.array([[0, 0], [23, 0], [23, 95], [0, 95]], np.float32))
    return cv2.cvtColor(cv2.warpPerspective(image, M, (24, 96), flags=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)


def _stretch_appearance(a, b, M, shift, image_a, image_b, min_frac):
    """Appearance over the stretch of an upright spine that both boxes cover (a, b: pair-geometry dicts; a is projected into b's pixels by M + shift).
    A box clipped by the frame border, or detected shorter, shows a different part of the book than the whole box of the other view, so whole-box
    patches mis-register along the spine and correlate poorly even for the same book. None when the boxes share too little of the spine."""
    u = b["v"]
    shift = shift - float(shift @ u) * u  # the voted shift is measured between box CENTRES, which clipping biases along the spine: align across the spine only
    sa, sb = (a["P"] + shift) @ u, b["P"] @ u
    lo, hi = max(sa.min(), sb.min()), min(sa.max(), sb.max())
    if hi - lo < min_frac * min(sa.max() - sa.min(), sb.max() - sb.min()):
        return None
    pa = _sub_patch(image_a, a["o"]["polygon_xy"], M, u, lo, hi, shift)
    pb = _sub_patch(image_b, b["o"]["polygon_xy"], np.eye(3), u, lo, hi, np.zeros(2))
    return None if pa is None or pb is None else appearance(pa, pb)


def _footprint(T, w, h):
    return _proj([[0, 0], [w, 0], [w, h], [0, h]], T)


def _overlap_frac(pa, pb):
    a = cv2.contourArea(pa.astype(np.float32))
    if a <= 0:
        return 0.0
    pa32, pb32 = pa.astype(np.float32), pb.astype(np.float32)
    if cv2.contourArea(pb32) <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(pa32, pb32)
    return float(inter) / min(a, cv2.contourArea(pb32))


def _loop_closure(det_frames, by_frame, info, patch, frames, cfg, load_image):
    """Revisit linking independent of the drifting chain: SIFT + RANSAC homography between
    sharp frames that are far apart in time, then appearance-verified one-to-one matching of spines.
    Spines repeat, so a pair only counts with many geometric inliers (wood/frames/signs carry them)."""
    cand = [f for f in det_frames if len(by_frame[f]) >= cfg["loop_min_obs"]]
    cand = sorted(cand, key=lambda f: -(frames[f].get("quality") or {}).get("sharpness", 0.0))[:cfg["loop_max_frames"]]
    cand.sort(key=lambda f: (frames[f]["timestamp_s"], f))
    sc = cfg["loop_scale"]
    S, Si = np.diag([sc, sc, 1.0]), np.diag([1 / sc, 1 / sc, 1.0])
    sift = cv2.SIFT_create(nfeatures=3000)
    feats = {}
    for f in cand:
        im = load_image(f)
        g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=sc, fy=sc)
        feats[f] = sift.detectAndCompute(g, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    edges, tested, accepted = [], 0, 0
    for i, fa in enumerate(cand):
        ka, da = feats[fa]
        if da is None or len(ka) < 30:
            continue
        for fb in cand[i + 1:]:
            if frames[fb]["timestamp_s"] - frames[fa]["timestamp_s"] < cfg["loop_min_gap_s"]:
                continue
            kb, db = feats[fb]
            if db is None or len(kb) < 30:
                continue
            tested += 1
            good = [m[0] for m in bf.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            if len(good) < 20:
                continue
            pa = np.float32([ka[m.queryIdx].pt for m in good]); pb = np.float32([kb[m.trainIdx].pt for m in good])
            H, mk = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
            if H is None or int(mk.sum()) < cfg["loop_min_inliers"]:
                continue
            H = Si @ H @ S  # to full-resolution pixels
            sdet = math.sqrt(abs(np.linalg.det(H[:2, :2])))
            if not cfg["loop_min_scale"] < sdet < cfg["loop_max_scale"]:
                continue
            A, B = by_frame[fa], by_frame[fb]
            ca = _proj([np.mean(info[k]["o"]["polygon_xy"], 0) for k in A], H)
            cb = np.array([np.mean(info[k]["o"]["polygon_xy"], 0) for k in B])
            wa = np.array([_geom_w(info[k]["o"]["polygon_xy"]) for k in A]) * sdet
            wb = np.array([_geom_w(info[k]["o"]["polygon_xy"]) for k in B])
            d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
            tol = cfg["loop_tol_widths"] * np.maximum(wa[:, None], wb[None, :])
            cands = []
            for x, y in zip(*np.nonzero(d <= tol)):
                a = appearance(patch(A[x]), patch(B[y]))
                if a >= cfg["min_appearance"]:
                    cands.append((a - 0.1 * d[x, y] / tol[x, y], x, y, a, d[x, y] / tol[x, y]))
            cands.sort(reverse=True)
            used_a, used_b, got = set(), set(), 0
            for _, x, y, a, nd in cands:
                if x in used_a or y in used_b:
                    continue
                used_a.add(x); used_b.add(y); got += 1
                edges.append({"observation_ids": [A[x], B[y]], "frame_ids": [fa, fb], "score": float(a) - 1.0,  # processed after short-range links
                              "eligible": True, "appearance_score": float(a), "center_distance_spine_widths": float(nd),
                              "geometry": "sift_homography_loop_closure", "pair_inliers": int(mk.sum()),
                              "reason": "Revisit linked by SIFT/RANSAC homography + warped-spine appearance; title not used"})
            if got:
                accepted += 1
    return edges, {"frames_used": len(cand), "pairs_tested": tested, "pairs_accepted": accepted, "edges": len(edges)}


def _geom_w(poly):
    p = np.asarray(poly, np.float64)
    e = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
    return min(e[0], e[1])


def _match_items(edges, IA, IB, iinfo, shift, fa, fb, frames, patch, cfg):
    """Non-book objects (plants, frames, signs...): same category, tracked-residual centre + box overlap + appearance."""
    short = frames[fb]["timestamp_s"] - frames[fa]["timestamp_s"] <= cfg["near_gap_s"]
    cand = []
    for i, a in enumerate(IA):
        for j, b in enumerate(IB):
            x, y = iinfo[a], iinfo[b]
            if x["key"] != y["key"]:
                continue
            size = max(x["s"], y["s"])
            nd = float(np.linalg.norm(y["c"] - x["c"] - shift)) / (cfg["item_tol"] * size)
            if nd > 1.0:
                continue
            ax = (x["box"][0] + shift[0], x["box"][1] + shift[1], x["box"][2] + shift[0], x["box"][3] + shift[1])
            iw = max(0.0, min(ax[2], y["box"][2]) - max(ax[0], y["box"][0])); ih = max(0.0, min(ax[3], y["box"][3]) - max(ax[1], y["box"][1]))
            inter = iw * ih
            union = (ax[2] - ax[0]) * (ax[3] - ax[1]) + (y["box"][2] - y["box"][0]) * (y["box"][3] - y["box"][1]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou < 0.3:
                continue
            ap = appearance(patch(a), patch(b))
            if ap >= (cfg["near_min_appearance"] if short and nd <= 0.6 else cfg["item_min_appearance"]):
                cand.append((ap + iou - 0.1 * nd, i, j, ap, nd))
    cand.sort(reverse=True)
    ua, ub = set(), set()
    for score, i, j, ap, nd in cand:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j)
        edges.append({"observation_ids": [IA[i], IB[j]], "frame_ids": [fa, fb], "score": float(score) - 0.5, "eligible": True,
                      "appearance_score": float(ap), "center_distance_spine_widths": float(nd), "geometry": "tracked_chain_item",
                      "reason": "Same-category object: tracked-motion overlap + box IoU + appearance; title/category text not used for merging"})


def _match_flat(edges, A, B, used_a, used_b, flat, thin, info, shift, fa, fb, frames, patch, cfg, load_image, images, T):
    """Thin/flat boxes (lying or stacked books): pairs of horizontal-ish boxes (>= flat_min_angle_deg from shelf-up, short/long < flat_pair_ratio_max) of which at least one is thin
    (short/long < flat_ratio); only boxes the upright rule left unmatched, one-to-one, same-frame exclusion is kept by dedup.associate.

    Differences from the upright rule, all because a flat box is long and thin:
      * across tolerance has a floor proportional to box LENGTH (flat_across_length_frac), not only to thickness;
      * along the long axis the boxes need only OVERLAP by flat_min_along_overlap of the shorter one (detections cover different stretches of one book);
      * appearance is the better of the whole-box patch and a patch of just the overlapping stretch, with a flat-specific gate."""
    ia = [i for i, k in enumerate(A) if k in flat and i not in used_a]
    ib = [j for j, k in enumerate(B) if k in flat and j not in used_b]
    if not ia or not ib:
        return 0
    gap_s = frames[fb]["timestamp_s"] - frames[fa]["timestamp_s"]
    short = gap_s <= cfg["near_gap_s"]
    cand = []
    for i in ia:
        a = info[A[i]]
        for j in ib:
            b = info[B[j]]
            u = b["v"]
            if A[i] not in thin and B[j] not in thin:  # at least one of the pair must be thin
                continue
            if abs(float(a["v"] @ u)) < 0.9:  # long axes within ~25 degrees
                continue
            d = b["c"] - a["c"] - shift
            dv = float(d @ u)
            du = float(np.sqrt(max(float(d @ d) - dv * dv, 0.0)))
            L = (a["L"] + b["L"]) / 2
            across_tol = max(cfg["tol_widths"] * max(a["w"], b["w"]), cfg["flat_across_length_frac"] * L)
            if du > across_tol:
                continue
            sa, sb = (a["P"] + shift) @ u, b["P"] @ u
            lo, hi = max(sa.min(), sb.min()), min(sa.max(), sb.max())
            if hi - lo < cfg["flat_min_along_overlap"] * min(sa.max() - sa.min(), sb.max() - sb.min()):
                continue
            s = appearance(patch(A[i]), patch(B[j]))
            if s < cfg["flat_min_appearance"]:
                for f_ in (fa, fb):
                    if f_ not in images:
                        images[f_] = load_image(f_)
                pa = _sub_patch(images[fa], info[A[i]]["o"]["polygon_xy"], T[fa], u, lo, hi, shift)
                pb = _sub_patch(images[fb], info[B[j]]["o"]["polygon_xy"], T[fb], u, lo, hi, np.zeros(2))
                if pa is not None and pb is not None:
                    s = max(s, appearance(pa, pb))
            nd = float(np.sqrt((du / across_tol) ** 2 + (abs(dv) / (cfg["tol_length"] * L)) ** 2))  # dimensionless (operator precedence fixed)
            tight = short and du <= 0.35 * across_tol
            if s >= (cfg["flat_near_min_appearance"] if tight else cfg["flat_min_appearance"]):
                cand.append((s - 0.1 * nd, i, j, s, nd))
    cand.sort(reverse=True)
    n = 0
    ua, ub_ = set(), set()
    for score, i, j, s, nd in cand:
        if i in ua or j in ub_:
            continue
        ua.add(i); ub_.add(j); n += 1
        edges.append({"observation_ids": [A[i], B[j]], "frame_ids": [fa, fb], "score": float(score), "eligible": True, "appearance_score": float(s),
                      "center_distance_spine_widths": float(nd), "geometry": "tracked_chain_flat",
                      "reason": "Thin flat box: tracked-motion overlap + length-scaled across tolerance + stretch overlap + warped-patch appearance; title not used"})
    return n


def _guided_residual(A, B, pinfo, shift, patch, cfg):
    """Residual affine (2x3, applied after the translation `shift`) for one frame pair from appearance-passing spine pairs, or None.
    Returns (R, inliers, gate_pairs_before, gate_pairs_after); accepted only when more appearance-passing pairs fall inside the gate."""
    ca = np.array([pinfo[k]["c"] for k in A]) + shift
    cb = np.array([pinfo[k]["c"] for k in B])
    rad = cfg["guided_radius_lengths"] * float(np.median([pinfo[k]["L"] for k in A + B]))
    cand = [(i, j) for i, j in zip(*np.nonzero(np.linalg.norm(cb[None] - ca[:, None], axis=2) <= rad))
            if appearance(patch(A[i]), patch(B[j])) >= cfg["min_appearance"]]
    if len(cand) < cfg["guided_min_inliers"]:
        return None
    ii, jj = np.array([c[0] for c in cand]), np.array([c[1] for c in cand])
    wmed = float(np.median([pinfo[k]["w"] for k in B]))
    R, inl = cv2.estimateAffine2D(ca[ii].astype(np.float32), cb[jj].astype(np.float32), method=cv2.RANSAC,
                                  ransacReprojThreshold=cfg["tol_widths"] * wmed, maxIters=2000, confidence=0.995)
    if R is None or inl is None or int(inl.sum()) < cfg["guided_min_inliers"] or not np.isfinite(R).all():
        return None
    sv = np.linalg.svd(R[:, :2], compute_uv=False)
    if sv.max() > cfg["guided_max_scale"] or sv.min() < 1 / cfg["guided_max_scale"]:
        return None

    def gate_pairs(pred):  # one-to-one appearance-passing pairs inside the usual anisotropic gate
        dd = cb[jj] - pred[ii]
        v = np.array([pinfo[B[j]]["v"] for j in jj])
        dv = np.abs((dd * v).sum(1))
        du = np.sqrt(np.maximum((dd ** 2).sum(1) - dv ** 2, 0))
        w = np.maximum([pinfo[A[i]]["w"] for i in ii], [pinfo[B[j]]["w"] for j in jj])
        L = (np.array([pinfo[A[i]]["L"] for i in ii]) + np.array([pinfo[B[j]]["L"] for j in jj])) / 2
        nd = np.sqrt((du / (cfg["tol_widths"] * w)) ** 2 + (dv / (cfg["tol_length"] * L)) ** 2)
        used_a, used_b, n = set(), set(), 0
        for k in np.argsort(nd):
            if nd[k] <= 1.0 and ii[k] not in used_a and jj[k] not in used_b:
                used_a.add(ii[k]); used_b.add(jj[k]); n += 1
        return n
    base, new = gate_pairs(ca), gate_pairs(ca @ R[:, :2].T + R[:, 2])
    return (R, int(inl.sum()), base, new) if new > base else None


def track_edges(observations, frames, load_image, config=None, transforms=None):
    """Candidate same-book edges between detection frames.

    observations: candidate observations (book kind only is used); frames: {frame_id: {timestamp_s,
    width_px, height_px}} for ALL saved frames (chain needs consecutive ones).
    Returns (edges, report) where edges are dedup.associate-compatible dicts.
    """
    cfg = {**DEFAULTS, **(config or {})}
    order = sorted(frames, key=lambda f: (frames[f]["timestamp_s"], f))
    chain_report = {}
    if transforms is not None:  # externally optimised (e.g. pose-graph) frame->plane transforms
        T, status = {f: np.asarray(m, np.float64) for f, m in transforms.items()}, {f: "external" for f in transforms}
        segment = {f: 0 for f in transforms}
    else:
        T, status = chain_transforms(order, load_image, cfg["track_scale"], recover=cfg["recover_tracking"], bridge_frames=cfg["bridge_frames"], report=chain_report)
        segment = segments_from_status(order, status)
    hold_frames = sorted(f for f, st in status.items() if st.startswith("hold") or st == "segment_start")
    # Pairing ban covers frames whose own motion is unknown ('hold:*'). A 'segment_start' frame is the NEW anchor of its
    # segment: later frames are tracked relative to it, so it may pair within its segment (never across segments).
    hold_set = {f for f in hold_frames if status.get(f, "").startswith("hold")}
    def quad(o):
        if len(o.get("polygon_xy") or []) == 4:
            return o["polygon_xy"]
        if o.get("bbox_xyxy"):  # hosted/axis-aligned detections carry only a box
            x1, y1, x2, y2 = o["bbox_xyxy"]
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return None
    books = []
    for o in observations:
        if o["kind"] == "book" and o["frame_id"] in T and quad(o) is not None:
            books.append(o if len(o.get("polygon_xy") or []) == 4 else {**o, "polygon_xy": quad(o)})
    def geom(P):
        P = np.asarray(P, np.float64)
        e = [np.linalg.norm(P[i] - P[(i + 1) % 4]) for i in range(4)]
        w = min(e[0], e[1])
        long_vec = (P[1] - P[0]) if e[0] >= e[1] else (P[2] - P[1])
        v = long_vec / (np.linalg.norm(long_vec) + 1e-9)
        return {"c": P.mean(0), "w": w, "L": max(e[0], e[1]), "v": v, "P": P}
    # native (per-frame pixel) geometry; pair geometry is computed in frame b's pixels via the RELATIVE transform
    # inv(T_b) @ T_a, never in the accumulated plane (which turns projectively degenerate on long handheld chains)
    info, by_frame, images = {}, collections.defaultdict(list), {}
    for o in books:
        g = geom(o["polygon_xy"])
        if g["w"] <= 1e-6:
            continue
        info[o["observation_id"]] = {"o": o, **g}
        by_frame[o["frame_id"]].append(o["observation_id"])
    # thin + flat boxes: short/long < flat_ratio and long axis far from the shelf "up" direction (doubled-angle mean of native long axes)
    flat, thin, lying = set(), set(), set()  # lying: orientation class regardless of the rule (the upright-only stretch comparison skips them)
    if info:
        vs = np.array([x["v"] for x in info.values() if x["L"] >= 2.0 * x["w"]])
        if len(vs):
            th = np.arctan2(vs[:, 1], vs[:, 0])
            up = 0.5 * np.arctan2(np.sin(2 * th).sum(), np.cos(2 * th).sum())
            for oid, x in info.items():
                a = abs((np.arctan2(x["v"][1], x["v"][0]) - up + np.pi / 2) % np.pi - np.pi / 2)  # axial angle to "up", 0..90 deg
                if x["w"] < cfg["flat_pair_ratio_max"] * x["L"] and np.degrees(a) >= cfg["flat_min_angle_deg"]:
                    lying.add(oid)
                    if cfg["flat_rule"]:
                        flat.add(oid)
                        if x["w"] < cfg["flat_ratio"] * x["L"]:
                            thin.add(oid)
    iinfo, items_by_frame = {}, collections.defaultdict(list)
    for o in observations:
        if o["kind"] == "book" or o["frame_id"] not in T or not o.get("bbox_xyxy"):
            continue
        x1, y1, x2, y2 = o["bbox_xyxy"]
        quad = o.get("polygon_xy") if len(o.get("polygon_xy") or []) == 4 else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        P = np.asarray(quad, np.float64)
        lo, hi = P.min(0), P.max(0)
        size = float(max(hi - lo))
        if size <= 1e-6:
            continue
        iinfo[o["observation_id"]] = {"o": o, "quad": quad, "c": P.mean(0), "s": size, "box": (lo[0], lo[1], hi[0], hi[1]),
                                      "key": o.get("category") if o["kind"] == "non_book" else "__ambiguous__"}
        items_by_frame[o["frame_id"]].append(o["observation_id"])
    patches = {}

    def patch(oid):
        if oid not in patches:
            if oid in iinfo:
                o = iinfo[oid]["o"]
                fid = o["frame_id"]
                if fid not in images:
                    images[fid] = load_image(fid)
                patches[oid] = _patch(images[fid], iinfo[oid]["quad"])
                return patches[oid]
            o = info[oid]["o"]
            fid = o["frame_id"]
            if fid not in images:
                images[fid] = load_image(fid)
            patches[oid] = _patch(images[fid], o["polygon_xy"])
        return patches[oid]

    def project_items(ids, M):
        out = {}
        for k in ids:
            P = _proj(iinfo[k]["quad"], M)
            lo, hi = P.min(0), P.max(0)
            out[k] = {**iinfo[k], "c": P.mean(0), "s": float(max(hi - lo)), "box": (lo[0], lo[1], hi[0], hi[1])}
        return out
    det_frames = sorted(set(by_frame) | set(items_by_frame), key=lambda f: (frames[f]["timestamp_s"], f))
    edges, pairs = [], []
    for i, fa in enumerate(det_frames):
        for fb in det_frames[i + 1:]:
            if frames[fb]["timestamp_s"] - frames[fa]["timestamp_s"] > cfg["max_gap_s"]:
                break
            if segment.get(fa) != segment.get(fb):  # motion across a tracking break is unknown: never pair through it
                continue
            if fa in hold_set or fb in hold_set:  # a held frame has no observed displacement: no identity claim (a constellation vote is not motion evidence)
                continue
            held = False
            try:
                M = np.linalg.inv(T[fb]) @ T[fa]  # fa pixels -> fb pixels
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(M).all() or abs(M[2, 2]) < 1e-9:
                continue
            M = M / M[2, 2]
            Wb, Hb = frames[fb]["width_px"], frames[fb]["height_px"]
            foot_a = _proj([[0, 0], [frames[fa]["width_px"], 0], [frames[fa]["width_px"], frames[fa]["height_px"]], [0, frames[fa]["height_px"]]], M)
            if not np.isfinite(foot_a).all() or _overlap_frac(foot_a, np.array([[0, 0], [Wb, 0], [Wb, Hb], [0, Hb]], float)) < cfg["min_overlap_frac"]:
                continue
            A, B = by_frame.get(fa, []), by_frame.get(fb, [])
            IA, IB = items_by_frame.get(fa, []), items_by_frame.get(fb, [])
            pinfo = {k: info[k] for k in B}
            for k in A:
                g = geom(_proj(info[k]["P"], M))
                pinfo[k] = {**info[k], **g}
            if not A or not B:
                if IA and IB and not held:
                    _match_items(edges, IA, IB, {**{k: iinfo[k] for k in IB}, **project_items(IA, M)}, np.zeros(2), fa, fb, frames, patch, cfg)
                continue
            ca = np.array([pinfo[k]["c"] for k in A]); cb = np.array([pinfo[k]["c"] for k in B])
            wa = np.array([pinfo[k]["w"] for k in A]); wb = np.array([pinfo[k]["w"] for k in B])
            med = float(np.median(wb))
            d = cb[None] - ca[:, None]
            near = np.linalg.norm(d, axis=2) < med * 4
            shift, votes = np.zeros(2), 0
            if near.sum() >= cfg["min_votes"]:
                diffs = d[near]
                g = max(med * 0.5, 1.0)
                cnt = collections.Counter(map(tuple, np.round(diffs / g).astype(int)))
                top = max(cnt.values())
                # regular spine spacing creates near-tied alternative shifts; trust the tracked prior
                # (smallest residual) among modes with comparable evidence
                (kx, ky) = min((k for k, v in cnt.items() if v >= 0.8 * top), key=lambda k: (k[0] ** 2 + k[1] ** 2, -cnt[k]))
                votes = cnt[(kx, ky)]
                if votes >= cfg["min_votes"]:
                    shift = np.array([kx, ky]) * g
                    close = diffs[np.linalg.norm(diffs - shift, axis=1) < g * 1.5]
                    shift = close.mean(0)
                else:
                    votes = 0
            if held and votes < cfg["min_votes"]:  # no observed displacement for a held frame: no identity claim
                pairs.append({"frame_ids": [fa, fb], "votes": int(votes), "candidates": 0, "matched": 0, "held": True, "shift_spine_widths": None,
                              "gap_s": round(frames[fb]["timestamp_s"] - frames[fa]["timestamp_s"], 3), "near_pairs": int(near.sum())})
                continue
            guided = _guided_residual(A, B, pinfo, shift, patch, cfg) if cfg["guided_residual"] else None
            if guided is not None:  # re-express fa in fb pixels through the residual affine; the gate below is unchanged
                R = np.vstack([guided[0], [0, 0, 1]])
                M = R @ np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]]) @ M
                M = M / M[2, 2]
                for k in A:
                    pinfo[k] = {**info[k], **geom(_proj(info[k]["P"], M))}
                ca = np.array([pinfo[k]["c"] for k in A]); wa = np.array([pinfo[k]["w"] for k in A])
                d = cb[None] - ca[:, None]
                shift = np.zeros(2)
            # anisotropic distance: across the spine in spine widths, along it in fractions of its length
            # (detector boxes of one spine often differ in vertical extent while the spine is the same)
            vb = np.array([pinfo[k]["v"] for k in B]); Lb = np.array([pinfo[k]["L"] for k in B])
            Lmean = (np.array([pinfo[k]["L"] for k in A])[:, None] + Lb[None]) / 2
            dd = d - shift
            dv = np.einsum("ijk,jk->ij", dd, vb)
            du = np.sqrt(np.maximum((dd ** 2).sum(2) - dv ** 2, 0))
            wmax = np.maximum(wa[:, None], wb[None])
            # positional noise scales with box size for lying books: floor on the across tolerance for FLAT-oriented thin boxes only
            flat_a = np.array([k in flat and pinfo[k]["w"] / pinfo[k]["L"] < 0.34 for k in A])
            flat_b = np.array([k in flat and pinfo[k]["w"] / pinfo[k]["L"] < 0.34 for k in B])
            thin_flat = flat_a[:, None] & flat_b[None]
            across_tol = np.where(thin_flat, np.maximum(cfg["tol_widths"] * wmax, cfg["tol_across_min_length_frac"] * Lmean), cfg["tol_widths"] * wmax)
            nd_all = np.sqrt((du / across_tol) ** 2 + (dv / (cfg["tol_length"] * Lmean)) ** 2)
            cand = []
            gap_s = frames[fb]["timestamp_s"] - frames[fa]["timestamp_s"]
            short = gap_s <= cfg["near_gap_s"]
            near_miss = []
            for ii, jj in zip(*np.nonzero(nd_all <= cfg["near_miss_factor"])):
                s = appearance(patch(A[ii]), patch(B[jj]))
                tight = short and nd_all[ii, jj] <= cfg["near_tol_widths"] / cfg["tol_widths"]
                need = cfg["near_min_appearance"] if tight and nd_all[ii, jj] <= 1.0 else cfg["min_appearance"]
                if s < need and cfg["stretch_appearance"] and A[ii] not in lying and B[jj] not in lying:  # lying boxes have their own stretch rule below
                    for f_ in (fa, fb):
                        if f_ not in images:
                            images[f_] = load_image(f_)
                    s2 = _stretch_appearance(pinfo[A[ii]], pinfo[B[jj]], M, shift, images[fa], images[fb], cfg["stretch_min_frac"])
                    s = s if s2 is None else max(s, s2)
                if nd_all[ii, jj] > 1.0:  # outside the gate but close: recorded so the ambiguity is visible, never merged
                    if s >= cfg["min_appearance"]:
                        near_miss.append((s - cfg["geometry_weight"] * nd_all[ii, jj], ii, jj, s, nd_all[ii, jj]))
                    continue
                if s >= need:
                    cand.append((s - cfg["geometry_weight"] * nd_all[ii, jj], ii, jj, s, nd_all[ii, jj]))
            cand.sort(reverse=True)
            used_a, used_b, n = set(), set(), 0
            for score, ii, jj, s, nd in cand:  # one-to-one, best first; runner-ups are kept as ineligible competitors
                selected = ii not in used_a and jj not in used_b
                if selected:
                    used_a.add(ii); used_b.add(jj); n += 1
                elif not cfg["keep_runner_ups"]:
                    continue
                edges.append({"observation_ids": [A[ii], B[jj]], "frame_ids": [fa, fb], "score": float(score),
                              "eligible": bool(selected), "appearance_score": float(s), "center_distance_spine_widths": float(nd),
                              "geometry": "tracked_relative_residual" if guided is None else "tracked_relative_guided_affine", "residual_shift_votes": int(votes),
                              **({} if guided is None else {"guided_residual_inliers": guided[1]}),
                              "reason": ("Tracked-motion overlap + residual alignment + warped-spine appearance; title not used" if guided is None else
                                         f"Tracked motion + residual affine voted by {guided[1]} appearance-matched spine pairs (gate pairs {guided[2]}->{guided[3]}) "
                                         "+ warped-spine appearance; title not used") if selected
                              else "Competing alternative not selected by one-to-one matching"})
            n_flat = 0
            if flat:
                n_flat = _match_flat(edges, A, B, used_a, used_b, flat, thin, pinfo, shift, fa, fb, frames, patch, cfg, load_image, images, {fa: M, fb: np.eye(3)})
            for score, ii, jj, s, nd in near_miss:
                edges.append({"observation_ids": [A[ii], B[jj]], "frame_ids": [fa, fb], "score": float(score), "eligible": False,
                              "appearance_score": float(s), "center_distance_spine_widths": float(nd), "geometry": "tracked_relative_residual",
                              "residual_shift_votes": int(votes), "reason": "Plausible neighbour outside the geometric gate; kept separate, flagged uncertain"})
            if IA and IB:
                _match_items(edges, IA, IB, {**{k: iinfo[k] for k in IB}, **project_items(IA, M)}, shift, fa, fb, frames, patch, cfg)
            pairs.append({"frame_ids": [fa, fb], "votes": int(votes), "candidates": len(cand), "matched": n, "flat_matched": n_flat,
                          "shift_spine_widths": round(float(np.linalg.norm(shift) / max(med, 1e-6)), 3),
                          "guided_residual": None if guided is None else {"inliers": guided[1], "gate_pairs_before": guided[2], "gate_pairs_after": guided[3]},
                          "gap_s": round(gap_s, 3), "near_pairs": int(near.sum())})
    loop_report = None
    if cfg["loop_closure"]:
        loop_edges, loop_report = _loop_closure(det_frames, by_frame, info, patch, frames, cfg, load_image)
        edges.extend(loop_edges)
    report = {"_transforms": {f: np.asarray(m).tolist() for f, m in T.items()}, "_segment_of": dict(segment), "hold_frames": hold_frames,
              "segments": int(max(segment.values()) + 1) if segment else 0, "step_transform": "affine, homography only when >=30% better fit",
              "method": "tracked_relative_residual_appearance", "loop_closure": loop_report, "tracking_recovery": chain_report.get("recovery"), "config": cfg, "tracking_failures": sum(s.startswith("hold") for s in status.values()),
              "frames_tracked": len(status), "detection_frames": len(det_frames), "pairs_considered": len(pairs), "edges": len(edges)}
    report["tracking_failures"] = len(hold_frames)
    report["held_frames"] = sorted(hold_set)  # frames whose own motion is unknown (no identity claim from them)
    return edges, report, pairs


def _clipped(poly, width, height, margin):
    p = np.asarray(poly, np.float64)
    return bool(p[:, 0].min() <= margin or p[:, 1].min() <= margin or p[:, 0].max() >= width - margin or p[:, 1].max() >= height - margin)


def _inside_fraction(inner, outer):
    """Fraction of the area of convex polygon `inner` that lies inside convex polygon `outer` (same pixel plane)."""
    a, b = cv2.convexHull(np.asarray(inner, np.float32)), cv2.convexHull(np.asarray(outer, np.float32))
    area = cv2.contourArea(a)
    if area <= 1e-6 or cv2.contourArea(b) <= 1e-6:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    return min(1.0, float(inter) / area)


def clipped_partial_links(groups, observations, frames, transforms, segments=None, held_frames=(), config=None):
    """Merge border-clipped STUB groups into the full-length group that contains them. Returns (new_groups, links).

    groups: lists of observation ids (the output of dedup.associate). A STUB is a book group whose EVERY sighting touches the frame border: in a fast pan the
    border cuts a spine, the detector box covers only the visible stretch, its centre is offset along the spine and the stretch differs from the full box, so
    the pairwise gates (centre distance, whole-box appearance) leave it as a separate group. A stub merges into a PARTNER group when
      * the partner has a sighting that is not clipped (its full extent was seen) and is clearly longer than the stub (stub_length_max_frac),
      * for every stub sighting that can be evaluated (same tracking segment, no held frame, within max_gap_s) the stub outline, projected into a partner
        sighting's pixels by the tracked RELATIVE transform, lies >= stub_inside_min inside that sighting's outline (best partner sighting per stub sighting;
        the weakest stub sighting scores the pair),
      * that containment is consistent: it holds in at least stub_support_min of the partner's evaluable sightings (a drifted chain or a neighbouring spine
        lands inside by chance in one frame, not in most),
      * the partner beats every other candidate by stub_margin (near-tie veto) and shares no frame with the stub (never co-visible: two sightings of one frame
        are two books), also not with stubs already merged into it.
    Titles and appearance are not used (blurred scans correlate poorly even for the same book). A stub without such a partner stays a separate counted
    group. General: no per-scan constants."""
    cfg = {**DEFAULTS, **(config or {})}
    if not cfg["clipped_partials"] or not groups:
        return [list(g) for g in groups], []
    segments = segments or {}
    held = set(held_frames)
    T = {f: np.asarray(m, np.float64) for f, m in transforms.items()}
    by_id = {o["observation_id"]: o for o in observations}

    def quad(o):
        if len(o.get("polygon_xy") or []) == 4:
            return np.asarray(o["polygon_xy"], np.float64)
        if o.get("bbox_xyxy"):
            x1, y1, x2, y2 = o["bbox_xyxy"]
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)
        return None

    sight = {}  # oid -> (frame_id, quad, clipped)
    for g in groups:
        for oid in g:
            o = by_id.get(oid)
            if o is None or o.get("kind") != "book" or o["frame_id"] not in frames:
                continue
            q = quad(o)
            if q is not None:
                f = frames[o["frame_id"]]
                sight[oid] = (o["frame_id"], q, _clipped(q, f["width_px"], f["height_px"], cfg["clip_margin_frac"] * min(f["width_px"], f["height_px"])))

    def length(oid):
        return float(max(np.linalg.norm(sight[oid][1][(i + 1) % 4] - sight[oid][1][i]) for i in range(4)))

    def partner_len(part):
        return max(length(o) for o in part if not sight[o][2])

    book_groups = [g for g in groups if g and all(oid in sight for oid in g)]
    stubs = [g for g in book_groups if all(sight[oid][2] for oid in g)]
    partners = [g for g in book_groups if any(not sight[oid][2] for oid in g)]
    group_frames = {tuple(g): {sight[oid][0] for oid in g} for g in book_groups}
    rel_cache = {}

    def relative(fa, fb):  # fa pixels -> fb pixels
        if (fa, fb) not in rel_cache:
            try:
                M = np.linalg.inv(T[fb]) @ T[fa]
                rel_cache[(fa, fb)] = M / M[2, 2] if np.isfinite(M).all() and abs(M[2, 2]) > 1e-9 else None
            except (np.linalg.LinAlgError, KeyError):
                rel_cache[(fa, fb)] = None
        return rel_cache[(fa, fb)]

    def usable(fa, fb):
        return (fa in T and fb in T and fa not in held and fb not in held and segments.get(fa) == segments.get(fb)
                and abs(frames[fb]["timestamp_s"] - frames[fa]["timestamp_s"]) <= cfg["max_gap_s"])

    def inside(sid, pid):  # fraction of the stub sighting that lies inside the partner sighting, in the partner sighting's pixels (None: not evaluable)
        (fa, qa, _), (fb, qb, _) = sight[sid], sight[pid]
        if not usable(fa, fb) or relative(fa, fb) is None:
            return None
        proj = _proj(qa, relative(fa, fb))
        return _inside_fraction(proj, qb) if np.isfinite(proj).all() else None

    ranked = []
    for stub in stubs:
        sf = group_frames[tuple(stub)]
        stub_len = max(length(o) for o in stub)
        scored = []
        for part in partners:
            if part is stub or sf & group_frames[tuple(part)]:
                continue
            if stub_len > cfg["stub_length_max_frac"] * partner_len(part):  # not clearly shorter: a near-full view the pairwise gates would have linked, or a tall group clipped throughout
                continue
            per_sighting, support = [], 1.0
            for sid in stub:
                vals = [(v, pid) for pid in part for v in [inside(sid, pid)] if v is not None]
                if vals:
                    per_sighting.append((max(vals)[0], sid, max(vals)[1]))
                    support = min(support, sum(v >= cfg["stub_inside_min"] for v, _ in vals) / len(vals))
            if per_sighting:
                scored.append((min(per_sighting)[0], part, min(per_sighting), support))
        scored.sort(key=lambda s: -s[0])
        if scored and scored[0][0] >= cfg["stub_inside_min"] and scored[0][3] >= cfg["stub_support_min"] and (len(scored) == 1 or scored[0][0] - scored[1][0] >= cfg["stub_margin"]):
            ranked.append((scored[0][0], scored[1][0] if len(scored) > 1 else 0.0, stub, scored[0][1], scored[0][2], scored[0][3]))
    ranked.sort(key=lambda r: (-r[0], r[2][0]))
    merged_into, links = {}, []
    for score, runner_up, stub, part, (_, sid, pid), support in ranked:
        key = tuple(part)
        frames_part = group_frames[key]
        if frames_part & group_frames[tuple(stub)]:  # an earlier stub merged into this partner already shows one of this stub's frames
            continue
        frames_part |= group_frames[tuple(stub)]
        merged_into.setdefault(key, []).append(stub)
        links.append({"observation_ids": [sid, pid], "frame_ids": [sight[sid][0], sight[pid][0]], "score": float(score), "eligible": True, "decision": "merged",
                      "appearance_score": None, "center_distance_spine_widths": None, "geometry": "clipped_partial_contained",
                      "inside_fraction": float(score), "runner_up_inside_fraction": float(runner_up), "partner_support_fraction": float(support),
                      "stub_sightings": len(stub), "stub_length_px": max(length(o) for o in stub), "partner_length_px": partner_len(part),
                      "reason": "Every sighting of this group is clipped by the frame border; its motion-projected outline lies inside a full-length group never seen in the same frame; title not used"})
    drop = {tuple(s) for stubs_ in merged_into.values() for s in stubs_}
    out = []
    for g in groups:
        if tuple(g) in drop:
            continue
        extra = [oid for s in merged_into.get(tuple(g), []) for oid in s]
        out.append(sorted(list(g) + extra) if extra else list(g))
    return sorted(out), links


def _quad(o):
    if len(o.get("polygon_xy") or []) == 4:
        return np.asarray(o["polygon_xy"], np.float64)
    if o.get("bbox_xyxy"):
        x1, y1, x2, y2 = o["bbox_xyxy"]
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)
    return None


def _iou(a, b):
    a, b = cv2.convexHull(np.asarray(a, np.float32)), cv2.convexHull(np.asarray(b, np.float32))
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 1e-6 else 0.0


def singleton_merge_links(groups, observations, frames, transforms, segments=None, held_frames=(), config=None, eligible=None):
    """Merge single-sighting book groups into the group they land on after the tracked motion. Returns (new_groups, links).

    A singleton S joins partner group G when, for some sighting of G in the same tracking segment, not held and within max_gap_s, the outline of S projected
    into that sighting's pixels by the tracked RELATIVE transform overlaps it with IoU >= singleton_min_iou; the best partner (max IoU over its sightings)
    must beat the runner-up by singleton_margin, and G must never be seen in S's frame (also after earlier merges). eligible: observation ids of the
    singletons that may merge (None = all). Geometry only: appearance is what
    failed on the blurred sightings, titles are never used. This is the relaxed pass behind the UPPER end of the count range."""
    cfg = {**DEFAULTS, **(config or {})}
    if not cfg["singleton_merge"] or not groups or not transforms:
        return [list(g) for g in groups], []
    segments, held = segments or {}, set(held_frames)
    T = {f: np.asarray(m, np.float64) for f, m in transforms.items()}
    by_id = {o["observation_id"]: o for o in observations}
    sight = {}
    for g in groups:
        for oid in g:
            o = by_id.get(oid)
            q = None if o is None or o.get("kind") != "book" or o["frame_id"] not in frames else _quad(o)
            if q is not None:
                sight[oid] = (o["frame_id"], q)
    book = [tuple(g) for g in groups if g and all(oid in sight for oid in g)]
    gid = {oid: i for i, g in enumerate(book) for oid in g}
    gframes = {i: {sight[oid][0] for oid in g} for i, g in enumerate(book)}

    def usable(fa, fb):
        return (fa in T and fb in T and fa not in held and fb not in held and segments.get(fa) == segments.get(fb)
                and abs(frames[fb]["timestamp_s"] - frames[fa]["timestamp_s"]) <= cfg["max_gap_s"])

    ranked, Tinv = [], {}
    for i, g in enumerate(book):
        if len(g) != 1 or (eligible is not None and g[0] not in eligible):
            continue
        fa, qa = sight[g[0]]
        best = {}
        for pid, (fb, qb) in sight.items():
            j = gid[pid]
            if j == i or fa in gframes[j] or not usable(fa, fb):
                continue
            if fb not in Tinv:
                Tinv[fb] = np.linalg.inv(T[fb])
            M = Tinv[fb] @ T[fa]
            if not np.isfinite(M).all() or abs(M[2, 2]) < 1e-9:
                continue
            proj = _proj(qa, M / M[2, 2])
            if np.isfinite(proj).all():
                v = _iou(proj, qb)
                if v > best.get(j, (0.0,))[0]:
                    best[j] = (v, pid)
        scored = sorted(((v, j, pid) for j, (v, pid) in best.items()), reverse=True)
        if scored and scored[0][0] >= cfg["singleton_min_iou"] and (len(scored) == 1 or scored[0][0] - scored[1][0] >= cfg["singleton_margin"]):
            ranked.append((scored[0][0], scored[1][0] if len(scored) > 1 else 0.0, i, scored[0][1], scored[0][2]))
    ranked.sort(key=lambda r: (-r[0], book[r[2]]))
    owner = list(range(len(book)))

    def root(k):
        while owner[k] != k:
            k = owner[k]
        return k
    links = []
    for score, runner, i, j, pid in ranked:
        ri, rj = root(i), root(j)
        if ri == rj or gframes[ri] & gframes[rj]:
            continue
        owner[ri] = rj
        gframes[rj] |= gframes[ri]
        sid = book[i][0]
        links.append({"observation_ids": [sid, pid], "frame_ids": [sight[sid][0], sight[pid][0]], "score": float(score), "eligible": True,
                      "decision": "merged", "appearance_score": None, "center_distance_spine_widths": None, "geometry": "singleton_tracked_position",
                      "iou": float(score), "runner_up_iou": float(runner),
                      "reason": "Single sighting whose motion-projected outline lands on a group never seen in the same frame; geometry only, title not used"})
    merged = collections.defaultdict(list)
    for i, g in enumerate(book):
        merged[root(i)].extend(g)
    inbook = set(book)
    out = [sorted(v) for v in merged.values()] + [list(g) for g in groups if tuple(g) not in inbook]
    return sorted(out), links


def merge(candidates_path, manifest_path, run_dir, config=None, readings_path=None, resolutions_path=None):
    """Stage entry point (same signature as dedup.merge) using tracked association."""
    from pipeline import dedup
    return dedup.merge(candidates_path, manifest_path, run_dir, {**(config or {}), "association": "track"}, readings_path, resolutions_path)
