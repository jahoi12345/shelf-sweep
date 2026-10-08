"""Tracking-by-detection for book spines (association='track_v2'), layered on pipeline/cross_view.py.

Why: on the close, blurry YugoTest2 scan, 174 of 640 cross_view groups start where another group ended shortly before at the same chain-plane position
(102 within 5 frames). Pairwise matching drops these links (blurred patches fail the appearance gate; flickering detections leave gaps). A tracker keeps
an identity alive through short gaps using the motion that is already known from the LK chain.

Method (no learned parameters; thresholds declared in DEFAULTS):
1. Motion: the LK chain (cross_view.chain_transforms) gives frame->plane homographies T. The last box of a track is predicted into the current frame
   with inv(T[current]) @ T[last_frame].
2. Assignment per detection frame: cost = 1 - IoU(predicted track polygon, detection polygon) (convex OBB polygons), Hungarian, accept IoU >= iou_min.
   Tracks unseen for up to max_gap_frames saved frames stay alive (gap bridging); older tracks are closed. A small age penalty prefers recent tracks.
3. Optional re-link (relink=True): a closed track end and a later track start within relink_max_gap_frames whose plane boxes overlap (IoU >= relink_iou)
   AND whose warped-spine appearance agrees (>= relink_min_appearance, cross_view.appearance) are linked, one-to-one, best first.
4. Edges: consecutive observations of a track become dedup.associate edges with scores above every cross_view edge, so they are processed first; the
   cross_view edges (appearance-gated, revisit/residual logic, non-book items, flat rule) are kept unchanged for everything else. Same-frame exclusion and the
   competing-edge veto in dedup.associate still apply. Titles are never used.
"""
from __future__ import annotations

import collections

import cv2
import numpy as np

DEFAULTS = {
    "iou_min": 0.3,               # predicted-vs-detected polygon IoU to continue a track
    "max_gap_frames": 5,          # bridge up to this many saved frames without a detection (1 s at 5 fps)
    "age_penalty": 0.02,          # per frame of age, added to the cost (prefer the most recent track)
    "relink": False,              # optional: re-link a track end to a later track start (appearance-gated)
    "relink_max_gap_frames": 25,
    "relink_iou": 0.5,
    "relink_min_appearance": 0.5,
    "cross_view": None,           # config passed through to cross_view.track_edges
}


def _quad(o):
    if len(o.get("polygon_xy") or []) == 4:
        return np.asarray(o["polygon_xy"], np.float64)
    x1, y1, x2, y2 = o["bbox_xyxy"]
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)


def _warp(P, H):
    q = np.hstack([P, np.ones((len(P), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]


def _poly_iou(a, b):
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    ha, hb = cv2.convexHull(a32), cv2.convexHull(b32)
    aa, ab = cv2.contourArea(ha), cv2.contourArea(hb)
    if aa <= 0 or ab <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(ha, hb)
    return float(inter) / (aa + ab - float(inter) + 1e-9)


def _assign(cost):
    """Exact min-cost assignment (Hungarian, potentials; O(n^3)) for a rectangular cost matrix; returns (rows, cols). No SciPy needed."""
    cost = np.asarray(cost, float)
    n, m = cost.shape
    transposed = n > m
    if transposed:
        cost, n, m = cost.T, m, n
    INF = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, int), np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j]:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return np.array(rows)[order], np.array(cols)[order]


def build_tracks(observations, frames, T, config=None, segment=None, unreliable=None):
    """Return (tracks, report): tracks = lists of observation_ids in time order (books only).

    segment: optional {frame_id: segment id}; transforms of different segments live in different planes, so a track never continues across a
    segment boundary. unreliable: optional set of frame_ids whose transform is provisional (tracking hold / segment start): no track continues
    into or out of such a frame (its motion is unknown), and detections there start new tracks."""
    cfg = {**DEFAULTS, **(config or {})}
    segment = segment or {}
    unreliable = set(unreliable or ())
    order = sorted(frames, key=lambda f: (frames[f]["timestamp_s"], f))
    idx = {f: k for k, f in enumerate(order)}
    by_frame = collections.defaultdict(list)
    for o in observations:
        if o["kind"] == "book" and o["frame_id"] in T and o["frame_id"] in idx:
            by_frame[o["frame_id"]].append(o)
    Tinv = {f: np.linalg.inv(np.asarray(T[f], np.float64)) for f in T}
    tracks, active = [], []  # active: list of track indices
    last = {}  # track index -> (frame_id, quad in its own frame)
    links = 0
    for f in order:
        dets = by_frame.get(f, [])
        k = idx[f]
        active = [t for t in active if k - idx[last[t][0]] <= cfg["max_gap_frames"] + 1]
        if not dets:
            continue
        if f in unreliable:
            usable = []
        else:
            usable = [t for t in active if segment.get(last[t][0]) == segment.get(f) and last[t][0] not in unreliable]
        if usable:
            preds = [_warp(last[t][1], Tinv[f] @ np.asarray(T[last[t][0]], np.float64)) for t in usable]
            quads = [_quad(o) for o in dets]
            iou = np.array([[_poly_iou(p, q) for q in quads] for p in preds])
            age = np.array([k - idx[last[t][0]] for t in usable], float)
            cost = 1.0 - iou + cfg["age_penalty"] * (age[:, None] - 1)
            cost[iou < cfg["iou_min"]] = 1e6
            rows, cols = _assign(cost)
            taken = set()
            for r, c in zip(rows, cols):
                if cost[r, c] >= 1e6:
                    continue
                t = usable[r]
                tracks[t].append(dets[c]["observation_id"])
                last[t] = (f, quads[c])
                taken.add(c)
                links += 1
            new = [c for c in range(len(dets)) if c not in taken]
        else:
            new = list(range(len(dets)))
        for c in new:
            tracks.append([dets[c]["observation_id"]])
            t = len(tracks) - 1
            last[t] = (f, _quad(dets[c]))
            active.append(t)
    report = {"tracks": len(tracks), "track_links": links, "singleton_tracks": sum(1 for t in tracks if len(t) == 1), "config": {k: v for k, v in cfg.items() if k != "cross_view"}}
    return tracks, report


def relink_tracks(tracks, observations, frames, T, load_image, config=None):
    """Optional appearance-gated re-link of track ends to later track starts. Returns list of (obs_end, obs_start, iou, appearance)."""
    from pipeline import cross_view

    cfg = {**DEFAULTS, **(config or {})}
    obs = {o["observation_id"]: o for o in observations}
    order = sorted(frames, key=lambda f: (frames[f]["timestamp_s"], f))
    idx = {f: k for k, f in enumerate(order)}

    def plane_box(o):
        P = _warp(_quad(o), np.asarray(T[o["frame_id"]], np.float64))
        return P

    ends = [(t[-1], idx[obs[t[-1]]["frame_id"]]) for t in tracks]
    starts = [(t[0], idx[obs[t[0]]["frame_id"]]) for t in tracks]
    frames_of = [{obs[i]["frame_id"] for i in t} for t in tracks]
    images, patches = {}, {}

    def patch(oid):
        if oid not in patches:
            f = obs[oid]["frame_id"]
            if f not in images:
                images[f] = load_image(f)
            patches[oid] = cross_view._patch(images[f], _quad(obs[oid]).tolist())
        return patches[oid]
    cand = []
    for a, (oe, ke) in enumerate(ends):
        pa = plane_box(obs[oe])
        for b, (os_, ks) in enumerate(starts):
            if a == b or not (cfg["max_gap_frames"] < ks - ke <= cfg["relink_max_gap_frames"]):
                continue
            if frames_of[a] & frames_of[b]:
                continue
            v = _poly_iou(pa, plane_box(obs[os_]))
            if v < cfg["relink_iou"]:
                continue
            app = cross_view.appearance(patch(oe), patch(os_))
            if app >= cfg["relink_min_appearance"]:
                cand.append((v + app, a, b, oe, os_, v, app))
    cand.sort(reverse=True)
    used_a, used_b, out = set(), set(), []
    for _, a, b, oe, os_, v, app in cand:
        if a in used_a or b in used_b:
            continue
        used_a.add(a); used_b.add(b)
        out.append((oe, os_, v, app))
    return out


def track_v2_edges(observations, frames, load_image, config=None, transforms=None):
    """Same contract as cross_view.track_edges: (edges, report, pairs). Tracker edges + cross_view edges; report['_transforms'] kept for dedup."""
    from pipeline import cross_view

    cfg = {**DEFAULTS, **(config or {})}
    edges, report, pairs = cross_view.track_edges(observations, frames, load_image, cfg.get("cross_view"), transforms=transforms)
    T = transforms if transforms is not None else report.get("_transforms")
    T = {f: np.asarray(m, np.float64) for f, m in T.items()}
    # the chain may be split into segments (separate planes) and contain provisional transforms at tracking holds: respect both
    segment, unreliable = {}, set(report.get("hold_frames") or ())
    if transforms is None and report.get("_segment_of"):
        # explicit segment ids from cross_view.track_edges; a held frame with an identity transform is not a segment start
        segment = {f: int(s) for f, s in report["_segment_of"].items()}
    elif transforms is None and report.get("hold_frames") is not None:
        order = sorted(frames, key=lambda f: (frames[f]["timestamp_s"], f))
        seg, starts = 0, set()
        for f in order:
            T0 = T.get(f)
            if T0 is not None and f in unreliable and np.allclose(T0, np.eye(3)) and f != order[0]:
                starts.add(f)
        for f in order:
            if f in starts:
                seg += 1
            segment[f] = seg
    tracks, trep = build_tracks(observations, frames, T, cfg, segment=segment, unreliable=unreliable)
    trep["respected_holds"] = len(unreliable)
    trep["segments_seen"] = (max(segment.values()) + 1) if segment else None
    obs = {o["observation_id"]: o for o in observations}
    tedges = []
    for t in tracks:
        for a, b in zip(t[:-1], t[1:]):
            tedges.append({"observation_ids": [a, b], "frame_ids": [obs[a]["frame_id"], obs[b]["frame_id"]], "score": 3.0, "eligible": True,
                           "appearance_score": None, "center_distance_spine_widths": None, "geometry": "tracker_iou",
                           "reason": "Motion-compensated IoU track (LK chain prediction, Hungarian per frame, gap bridging); title not used"})
    if cfg["relink"]:
        rl = relink_tracks(tracks, observations, frames, T, load_image, cfg)
        for oe, os_, v, app in rl:
            tedges.append({"observation_ids": [oe, os_], "frame_ids": [obs[oe]["frame_id"], obs[os_]["frame_id"]], "score": 2.0 + v, "eligible": True,
                           "appearance_score": float(app), "center_distance_spine_widths": None, "geometry": "tracker_relink",
                           "reason": "Track end re-linked to a later track start: plane overlap + warped-spine appearance; title not used"})
        trep["relinks"] = len(rl)
    report["tracker"] = trep
    report["method"] = "tracker_iou+tracked_chain_residual_appearance"
    return tedges + edges, report, pairs
