"""Automatic front end for ``pipeline.room``: wall quads from the sweep itself, no human taps.

``room.py`` turns *wall quads* (four image corners of a wall face between two vertical room corners, floor line to ceiling line,
TL,TR,BR,BL) plus ONE stated height into floor/wall area. Until now the quads were human-marked. This module produces the same
``room_evidence`` document automatically and abstains, with the exact words to ask the person to capture what is missing, when it cannot.

Two detectors, one validator, one output format:

* PRIMARY (hosted, optional): a vision model is asked ONLY for literal pixel geometry (wall quads) per keyframe; it never returns sizes or areas.
  Output is untrusted: it is validated geometrically (convex, ordered, near-vertical sides, plausible size) AND against the pixels (the claimed
  edges must lie on real colour edges perpendicular to them); anything that fails is dropped. Provenance ``auto_gemini``. A call counter hard-caps requests.
* FALLBACK (local, deterministic): line segments (LSD on grey + Lab chroma) -> full-height vertical corner candidates -> floor/ceiling lines between
  consecutive verticals -> quad, with corner confirmation by SLOPE CHANGE of the neighbouring plane's lines (a brightness step alone is not enough:
  bookcase uprights have one), a "corner stops at the ceiling and floor" test, and an interior-clutter penalty (a bookshelf is not a plain wall).
  Provenance ``auto_classical`` with a confidence in [0, 1].
* Scale is NEVER measured here: the height anchor is a user-stated ceiling height (``height_anchor_from_utterance``), exactly as ``room.py`` requires.

Everything is ``software_only_unvalidated``: accuracy against tape-measured rooms is not established. On a sweep that shows only the shelf wall
(YugoTest1/2) the expected and correct outcome is abstention.

    from pipeline import room_auto
    ev = room_auto.build_evidence(walls, height_anchor)             # room.py-compatible dict
    res = room_auto.estimate_from_frames(manifest, frames_dir, height_m=2.6, utterance="the ceiling is two point six metres")
    print(res["room"]["status"], res["capture_guidance"])
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

DEFAULTS = {
    "lsd_quant": 1.0,                # LSD gradient quantisation (default 2.0 misses 7-16 level corner steps)
    "work_width": 960,               # frames are resized to this width for line detection
    "min_seg_frac": 0.05,            # min line segment length / frame width
    "vertical_tol_deg": 7.0,         # near-vertical band around the dominant vertical direction
    "vertical_search_deg": 20.0,     # search band for the dominant vertical direction
    "min_vertical_span": 0.18,       # a corner edge spans >= this fraction of the frame height (a distant corner is short; later checks reject uprights/shelves)
    "min_face_width_frac": 0.20,     # a wall face is at least this wide between its vertical corners
    "min_face_height_frac": 0.45,    # at least ONE face of an adjacent pair must be this tall (mean of its two sides, share of frame height)
    "min_pair_face_height_frac": 0.15,  # every face of a pair must still be at least this tall (a distant wall is short)
    "single_face_min_slope_deg": 15.0,  # a single face (no adjacent face in the frame) needs the neighbouring walls' lines to differ by this much on BOTH sides (strong room-corner signature)
    "allow_single_face": False,      # OFF: admitted a real false positive (an empty blurred shelf compartment, YT2 f00224) and one face cannot give room.py an estimate by itself
    "require_adjacent_pair": True,   # emit walls only when two faces share a vertical corner: a single face cannot be told from a shelf compartment, and room.py needs a corner view for the focal length
    "neighbour_max_deg": 75.0,       # corner confirmation looks for the neighbouring wall's lines up to this angle from horizontal (steep for a near side wall)
    "horizontal_max_deg": 40.0,      # floor/ceiling lines may slope (perspective) up to this angle
    "min_line_coverage": 0.50,       # share of the face width covered by collinear segments for each of the top/bottom lines
    "slope_change_deg": 6.0,         # corner confirmation: the neighbouring plane's horizontal lines differ in direction by this much
    "brightness_step": 10.0,         # recorded only: a brightness step alone is NOT confirmation (shelf uprights have one)
    "corner_edge_overshoot": 0.20,   # a room corner stops at the ceiling and floor lines; a vertical continuing > this x face height beyond BOTH ends is an upright/door frame
    "frame_edge_margin": 0.06,       # a vertical this close to the image border cannot show its neighbouring wall; accepted unconfirmed if the other side is confirmed
    "min_confidence": 0.50,
    "max_interior_clutter": 0.55,    # vertical-edge density inside the quad above this => bookshelf-like, not a plain wall
    "max_interior_busy": 0.10,       # share of interior pixels with colour gradient >= busy_gradient: above this the face is not a plain wall (books, wood grain, pictures)
    "busy_gradient": 2.5,            # grey levels per pixel (Lab, sigma 1.5); painted walls ~0.00-0.03, shelf compartments in the YT1/YT2 videos 0.13-0.42 (dev observation)
    "edge_support_min": 0.55,        # validator: share of samples along each claimed edge that lie on a real image edge
    "edge_support_radius_px": 4,
    "edge_min_gradient": 1.0,        # grey levels per pixel (Lab, after sigma-1.5 smoothing); a 10-level corner step gives ~2.5, noise ~0.3
    "max_gemini_calls": 100,
}
PROVENANCE_ACTOR = "room_auto"


# ------------------------------------------------------------------------------------------------ geometry helpers
def _line_through(p, q):
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def _intersect(l1, l2):
    v = np.cross(l1, l2)
    if abs(v[2]) < 1e-9:
        return None
    return np.array([v[0] / v[2], v[1] / v[2]])


def _convex(quad) -> bool:
    q = np.asarray(quad, float)
    s = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        s.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(v > 0 for v in s) or all(v < 0 for v in s)


def _area(quad) -> float:
    q = np.asarray(quad, float)
    return float(abs(sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4))) / 2)


def validate_quad_geometry(quad, width: int, height: int, cfg: dict[str, Any] | None = None) -> list[str]:
    """Pure geometric checks of one wall quad (TL,TR,BR,BL) in image pixels; [] = acceptable."""
    c = {**DEFAULTS, **(cfg or {})}
    problems: list[str] = []
    try:
        q = np.asarray(quad, float).reshape(4, 2)
    except Exception:  # noqa: BLE001
        return ["quad is not 4 [x,y] pairs"]
    if not np.isfinite(q).all():
        return ["non-finite coordinates"]
    if not ((-0.1 * width <= q[:, 0]).all() and (q[:, 0] <= 1.1 * width).all() and (-0.1 * height <= q[:, 1]).all() and (q[:, 1] <= 1.1 * height).all()):
        problems.append("corner outside the image")
    if not _convex(q):
        problems.append("quad not convex / mis-ordered")
    if not (q[0][1] < q[3][1] and q[1][1] < q[2][1]):
        problems.append("top edge is not above the bottom edge")
    if not (q[0][0] < q[1][0] and q[3][0] < q[2][0]):
        problems.append("left edge is not left of the right edge")
    if _area(q) < 0.05 * width * height:
        problems.append("quad smaller than 5% of the frame")
    for (a, b, name) in ((q[0], q[3], "left"), (q[1], q[2], "right")):
        dx, dy = b[0] - a[0], b[1] - a[1]
        ang = abs(math.degrees(math.atan2(dx, dy)))  # 0 = vertical
        if min(ang, 180 - ang) > 15:
            problems.append(f"{name} side deviates {min(ang, 180 - ang):.0f} deg from vertical (room corners are vertical for a level camera)")
    if (q[2][0] - q[3][0]) < c["min_face_width_frac"] * width * 0.5 or max(q[3][1] - q[0][1], q[2][1] - q[1][1]) < c["min_pair_face_height_frac"] * height:
        problems.append("face too small to be a wall between two corners")
    return problems


def _gradient_field(img: np.ndarray):
    """Per-pixel colour gradient (grey levels per pixel) on Lab channels after light smoothing: gx, gy of shape (H, W, C)."""
    if img.ndim == 2:
        chans = [img.astype(np.float32)]
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        chans = [lab[:, :, i] for i in range(3)]
    gx = np.stack([cv2.Sobel(cv2.GaussianBlur(c, (0, 0), 1.5), cv2.CV_32F, 1, 0, ksize=3) / 8.0 for c in chans], -1)
    gy = np.stack([cv2.Sobel(cv2.GaussianBlur(c, (0, 0), 1.5), cv2.CV_32F, 0, 1, ksize=3) / 8.0 for c in chans], -1)
    return gx, gy


def interior_busy_fraction(img: np.ndarray, quad, cfg: dict[str, Any] | None = None) -> float:
    """Share of the quad interior (shrunk 3% of the frame width so the boundary edges themselves do not count) whose colour gradient is >= busy_gradient."""
    c = {**DEFAULTS, **(cfg or {})}
    gx, gy = _gradient_field(img)
    mag = np.sqrt((gx ** 2 + gy ** 2).sum(-1))
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(quad, float)).astype(np.int32)], 1)
    k = max(5, int(0.03 * img.shape[1]))
    m = cv2.erode(m, np.ones((k, k), np.uint8))
    return float((mag[m > 0] >= c["busy_gradient"]).mean()) if m.sum() > 0 else 1.0


def edge_support(img: np.ndarray, quad, cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """Pixel evidence for a claimed quad: for each edge, the fraction of sample points whose neighbourhood (+-radius px ACROSS the edge) contains a colour gradient
    PERPENDICULAR to the claimed edge of at least ``edge_min_gradient`` grey levels/px. A claimed line through flat or differently oriented texture gets ~0."""
    c = {**DEFAULTS, **(cfg or {})}
    gx, gy = _gradient_field(img)
    h, w = gx.shape[:2]
    r = int(c["edge_support_radius_px"])
    q = np.asarray(quad, float).reshape(4, 2)
    out = {}
    for name, a, b in (("top", q[0], q[1]), ("right", q[1], q[2]), ("bottom", q[3], q[2]), ("left", q[0], q[3])):
        d = b - a
        n_len = float(np.linalg.norm(d))
        if n_len < 1:
            out[name] = 0.0
            continue
        d = d / n_len
        nrm = np.array([-d[1], d[0]])
        ok = []
        for p in np.linspace(a, b, int(max(20, n_len / 4))):
            if not (0 <= p[0] < w and 0 <= p[1] < h):
                continue
            best = 0.0
            for k in range(-r, r + 1):
                x, y = int(round(p[0] + k * nrm[0])), int(round(p[1] + k * nrm[1]))
                if 0 <= x < w and 0 <= y < h:
                    best = max(best, float(np.linalg.norm(gx[y, x] * nrm[0] + gy[y, x] * nrm[1])))
            ok.append(best >= c["edge_min_gradient"])
        out[name] = float(np.mean(ok)) if ok else 0.0
    return out


# ------------------------------------------------------------------------------------------------ classical detector
def _segments(img: np.ndarray, cfg: dict[str, Any]):
    """LSD on grey AND on the Lab chroma channels (two walls of similar brightness but different colour still meet in a visible line); near-duplicate
    segments from different channels are merged."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, 0.8, 0.6, cfg["lsd_quant"])
    chans = [img] if img.ndim == 2 else [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)] + [c for c in cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))[1:]]
    allseg = []
    for ch in chans:
        lines = lsd.detect(cv2.GaussianBlur(ch, (3, 3), 0))[0]
        if lines is not None:
            allseg.append(np.asarray(lines, float).reshape(-1, 4))  # OpenCV 5 returns Nx4, OpenCV 4 returns Nx1x4
    if not allseg:
        return np.zeros((0, 4))
    s = np.concatenate(allseg)
    ln = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    s = s[ln >= cfg["min_seg_frac"] * chans[0].shape[1]]
    keep, seen = [], set()
    for i, r in enumerate(s):  # merge near-duplicates (same endpoints within ~6 px, either direction)
        pts = sorted([(r[0], r[1]), (r[2], r[3])])
        k = tuple(int(round(v / 6)) for p in pts for v in p)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return s[keep]


def _angle_deg(s):  # 0..180, 90 = vertical
    return np.degrees(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])) % 180.0


def _vertical_lines(seg, w, h, cfg):
    ang = _angle_deg(seg)
    ln = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    near = np.abs(ang - 90) <= cfg["vertical_search_deg"]
    if not near.any():
        return [], 90.0
    # dominant vertical direction = length-weighted circular mean of near-vertical orientations
    th = np.radians(ang[near] - 90) * 2
    dom = 90.0 + math.degrees(math.atan2((ln[near] * np.sin(th)).sum(), (ln[near] * np.cos(th)).sum()) / 2)
    vert = np.abs(ang - dom) <= cfg["vertical_tol_deg"]
    y0 = h / 2.0
    xs, items = [], []
    for s in seg[vert]:
        dx, dy = s[2] - s[0], s[3] - s[1]
        if abs(dy) < 1e-6:
            continue
        xm = s[0] + (y0 - s[1]) * dx / dy
        xs.append(xm)
        items.append((xm, min(s[1], s[3]), max(s[1], s[3]), math.hypot(dx, dy)))
    if not items:
        return [], dom
    order = np.argsort(xs)
    clusters, cur = [], [items[order[0]]]
    for i in order[1:]:
        if items[i][0] - cur[-1][0] <= 0.008 * w:
            cur.append(items[i])
        else:
            clusters.append(cur)
            cur = [items[i]]
    clusters.append(cur)
    lines = []
    for cl in clusters:
        ivs = sorted((c[1], c[2]) for c in cl)
        span, end = 0.0, -1e9
        for a, b in ivs:  # union of y-intervals
            a = max(a, end)
            if b > a:
                span += b - a
                end = b
        top, bot = min(c[1] for c in cl), max(c[2] for c in cl)
        wts = np.array([c[3] for c in cl])
        lines.append({"x": float(np.average([c[0] for c in cl], weights=wts)), "top": top, "bottom": bot, "span": span / h, "n": len(cl)})
    return [l for l in lines if l["span"] >= cfg["min_vertical_span"] and l["top"] <= 0.5 * h and l["bottom"] >= 0.5 * h], dom


def _best_line(seg, x0, x1, sel_y, cfg, max_deg=None):
    """Best near-horizontal line among segments whose midpoint is between x0..x1 and for which sel_y(mid_y) holds. Returns (line, coverage, angle) or None."""
    mid = np.stack([(seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2], 1)
    ang = _angle_deg(seg)
    hor = (np.minimum(ang, 180 - ang) <= (max_deg if max_deg is not None else cfg["horizontal_max_deg"])) & (mid[:, 0] > x0) & (mid[:, 0] < x1)
    cand = [i for i in np.nonzero(hor)[0] if sel_y(mid[i, 1])]
    if not cand:
        return None
    ln = np.hypot(seg[cand, 2] - seg[cand, 0], seg[cand, 3] - seg[cand, 1])
    best = None
    for k in np.argsort(-ln)[:6]:  # try the longest seeds; support = collinear segments
        s = seg[cand[k]]
        l = _line_through(s[:2], s[2:])
        nrm = np.hypot(l[0], l[1])
        support, ids = 0.0, []
        for j in cand:
            t = seg[j]
            d = max(abs(l[0] * t[0] + l[1] * t[1] + l[2]), abs(l[0] * t[2] + l[1] * t[3] + l[2])) / nrm
            dang = abs(((_angle_deg(seg[[j]])[0] - _angle_deg(seg[[cand[k]]])[0] + 90) % 180) - 90)
            if d <= 4.0 and dang <= 3.0:
                support += math.hypot(t[2] - t[0], t[3] - t[1])
                ids.append(j)
        if best is None or support > best[0]:
            best = (support, ids)
    support, ids = best
    pts = np.concatenate([seg[ids][:, :2], seg[ids][:, 2:]]).astype(np.float32)
    vx, vy, px, py = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    line = _line_through((px, py), (px + vx, py + vy))
    xs = np.concatenate([seg[ids][:, 0], seg[ids][:, 2]])
    cover = min(1.0, (xs.max() - xs.min()) / max(x1 - x0, 1.0)) if len(ids) else 0.0
    covered = min(1.0, support / max(x1 - x0, 1.0))
    return line, min(cover, covered * 1.2), float(math.degrees(math.atan2(-line[0], line[1])) % 180.0)


def detect_walls_classical(img_bgr: np.ndarray, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wall faces (both vertical corners + floor line + ceiling line visible) found with line geometry only. Abstains with reasons."""
    c = {**DEFAULTS, **(cfg or {})}
    src_h, src_w = img_bgr.shape[:2]
    sc = c["work_width"] / src_w if src_w > c["work_width"] else 1.0
    img = cv2.resize(img_bgr, (int(src_w * sc), int(src_h * sc)), interpolation=cv2.INTER_AREA) if sc != 1.0 else img_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    seg = _segments(img, c)
    out: dict[str, Any] = {"walls": [], "abstain": True, "reasons": [], "debug": {"segments": int(len(seg))}}
    if len(seg) < 4:
        out["reasons"].append("too_few_line_segments")
        return out
    verts, dom = _vertical_lines(seg, w, h, c)
    out["debug"].update(vertical_lines=[round(v["x"] / sc) for v in verts], vertical_direction_deg=round(dom, 1))
    if len(verts) < 2:
        out["reasons"].append("fewer_than_two_full_height_vertical_edges")
        return out
    verts.sort(key=lambda v: v["x"])
    ang_all = _angle_deg(seg)
    for a, b in zip(verts[:-1], verts[1:]):
        if (b["x"] - a["x"]) < c["min_face_width_frac"] * w:
            continue
        yc = 0.5 * (min(a["top"], b["top"]) + max(a["bottom"], b["bottom"]))
        top = _best_line(seg, a["x"], b["x"], lambda y, yc=yc: y < yc, c)
        bot = _best_line(seg, a["x"], b["x"], lambda y, yc=yc: y >= yc, c)
        if top is None or bot is None:
            out["reasons"].append("ceiling_or_floor_line_not_found_between_corners")
            continue
        (lt, covt, angt), (lb, covb, angb) = top, bot
        if min(covt, covb) < c["min_line_coverage"]:
            out["reasons"].append(f"floor_or_ceiling_line_coverage_{min(covt, covb):.2f}_below_{c['min_line_coverage']}")
            continue
        la = _line_through((a["x"], 0), (a["x"], h))
        lb_v = _line_through((b["x"], 0), (b["x"], h))
        pts = [_intersect(lt, la), _intersect(lt, lb_v), _intersect(lb, lb_v), _intersect(lb, la)]
        if any(p is None for p in pts):
            out["reasons"].append("degenerate_intersection")
            continue
        quad = [[float(p[0]), float(p[1])] for p in pts]
        prob = validate_quad_geometry(quad, w, h, c)
        height_frac = 0.5 * ((quad[3][1] - quad[0][1]) + (quad[2][1] - quad[1][1])) / h  # mean of both edges: an oblique face has one short edge
        if height_frac < c["min_pair_face_height_frac"]:
            prob.append("face_height_below_minimum")
        if prob:
            out["reasons"].append("geometry:" + "; ".join(prob))
            continue

        # corner confirmation: on each vertical, the neighbouring plane's horizontal lines point a different way (slope change). A grey-level step is recorded but is NOT
        # enough (bookcase uprights have one). A vertical within frame_edge_margin of the border cannot show its neighbour and is accepted only if the OTHER side is confirmed.
        def confirm(v, side):
            x = v["x"]
            lo, hi = (x - 0.30 * w, x - 0.01 * w) if side < 0 else (x + 0.01 * w, x + 0.30 * w)
            inner = (angt + angb) / 2
            other = _best_line(seg, max(lo, 0), min(hi, w), lambda y: True, c, max_deg=c["neighbour_max_deg"])  # a near side wall's floor/ceiling lines are steep
            sdiff = abs(((other[2] - inner + 90) % 180) - 90) if other is not None else 0.0
            slope = other is not None and sdiff >= c["slope_change_deg"]
            y0, y1 = int(max(v["top"], 0)), int(min(v["bottom"], h - 1))
            xl, xr = int(np.clip(x - 6, 0, w - 1)), int(np.clip(x + 6, 0, w - 1))
            step = abs(float(np.median(gray[y0:y1, max(xl - 8, 0):xl + 1])) - float(np.median(gray[y0:y1, xr:min(xr + 9, w)]))) if y1 > y0 else 0.0
            near_border = x < c["frame_edge_margin"] * w or x > (1 - c["frame_edge_margin"]) * w
            return bool(slope), bool(near_border), round(step, 1), round(float(sdiff), 1)
        cl, cr = confirm(a, -1), confirm(b, +1)
        confirmed = [bool(cl[0] or cl[1]), bool(cr[0] or cr[1])]
        if not (cl[0] or cr[0]):
            out["reasons"].append("no_corner_confirmed_by_slope_change_vertical_edges_may_be_furniture_or_shelves")
            continue
        if not all(confirmed):
            out["reasons"].append("one_vertical_edge_unconfirmed_and_not_at_the_image_border")
            continue
        # a real corner ends at the ceiling and floor lines: reject verticals that run on beyond BOTH ends of the face (shelf upright / door frame)
        fh = 0.5 * ((quad[3][1] - quad[0][1]) + (quad[2][1] - quad[1][1]))

        def overshoot(v):
            return (min(quad[0][1], quad[1][1]) - v["top"]) / max(fh, 1.0), (v["bottom"] - max(quad[2][1], quad[3][1])) / max(fh, 1.0)
        oa, ob = overshoot(a), overshoot(b)
        if (oa[0] > c["corner_edge_overshoot"] and oa[1] > c["corner_edge_overshoot"]) or (ob[0] > c["corner_edge_overshoot"] and ob[1] > c["corner_edge_overshoot"]):
            out["reasons"].append("vertical_edge_continues_beyond_ceiling_and_floor_lines_not_a_room_corner")
            continue
        # interior clutter: vertical edges inside the face
        x0, x1 = int(quad[0][0]), int(quad[1][0]); y0, y1 = int(min(quad[0][1], quad[1][1])), int(max(quad[2][1], quad[3][1]))
        inner_v = [s for s in seg[np.abs(ang_all - dom) <= c["vertical_tol_deg"]] if x0 + 0.04 * w < (s[0] + s[2]) / 2 < x1 - 0.04 * w and y0 <= (s[1] + s[3]) / 2 <= y1
                   and math.hypot(s[2] - s[0], s[3] - s[1]) >= 0.12 * (y1 - y0)]
        clutter = min(1.0, len(inner_v) / max(6.0, (x1 - x0) / (0.02 * w)))
        busy = interior_busy_fraction(img, quad, c)
        conf = float(min(covt, covb) * (1.0 if all(confirmed) else 0.75) * (1.0 - 0.6 * clutter) * max(0.5, 1.0 - 4.0 * busy))
        wall = {"_va": a["x"], "_vb": b["x"], "height_frac": round(height_frac, 3), "quad_px": [[round(p[0] / sc, 1), round(p[1] / sc, 1)] for p in quad], "confidence": round(conf, 3), "method": "lsd_vertical_pair_floor_ceiling",
                "corner_confirmed": confirmed, "corner_by_slope_change": [bool(cl[0]), bool(cr[0])], "neighbour_slope_diff_deg": [cl[3], cr[3]], "line_coverage": [round(covt, 2), round(covb, 2)], "interior_clutter": round(clutter, 2), "interior_busy": round(busy, 3)}
        if busy > c["max_interior_busy"]:
            out["reasons"].append(f"interior_busy_{busy:.2f}_not_a_plain_wall_books_wood_grain_or_pictures"); out["debug"].setdefault("rejected", []).append({k: v for k, v in wall.items() if not k.startswith("_")}); continue
        if clutter > c["max_interior_clutter"]:
            out["reasons"].append(f"interior_clutter_{clutter:.2f}_looks_like_shelving_not_a_plain_wall"); out["debug"].setdefault("rejected", []).append({k: v for k, v in wall.items() if not k.startswith("_")}); continue
        if conf < c["min_confidence"]:
            out["reasons"].append(f"confidence_{conf:.2f}_below_{c['min_confidence']}"); out["debug"].setdefault("rejected", []).append({k: v for k, v in wall.items() if not k.startswith("_")}); continue
        out["walls"].append(wall)
    if c["require_adjacent_pair"]:
        cands, keep = out["walls"], set()
        for i in range(len(cands) - 1):
            if abs(cands[i]["_vb"] - cands[i + 1]["_va"]) < 1e-6 and max(cands[i]["height_frac"], cands[i + 1]["height_frac"]) >= c["min_face_height_frac"]:
                keep.update((i, i + 1))
        singles = []
        if c["allow_single_face"]:
            for i, w in enumerate(cands):
                if i not in keep and w["height_frac"] >= c["min_face_height_frac"] and min(w["neighbour_slope_diff_deg"]) >= c["single_face_min_slope_deg"]:
                    singles.append(i)
        if cands and not keep and not singles:
            out["reasons"].append("single_face_or_short_faces_without_an_adjacent_tall_face_not_accepted_as_a_room_corner")
        out["walls"] = [dict(w, pair_confirmed=(i in keep), single_face_strong_corners=(i in singles)) for i, w in enumerate(cands) if i in keep or i in singles]
    for w in out["walls"]:
        w.pop("_va", None)
        w.pop("_vb", None)
    out["abstain"] = not out["walls"]
    return out


# ------------------------------------------------------------------------------------------------ hosted path (geometry only)
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "walls": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"},
            "quad_px": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "left_corner_visible": {"type": "boolean"}, "right_corner_visible": {"type": "boolean"},
            "floor_line_visible": {"type": "boolean"}, "ceiling_line_visible": {"type": "boolean"},
            "confidence": {"type": "number"}},
            "required": ["quad_px", "left_corner_visible", "right_corner_visible", "floor_line_visible", "ceiling_line_visible"]}},
        "notes": {"type": "string"}},
    "required": ["walls"]}


def gemini_prompt(width: int, height: int) -> str:
    return (f"This is one frame ({width}x{height} pixels) from a handheld sweep of a room. Report ONLY literal image geometry, no measurements, no sizes, no areas.\n"
            "For every WALL FACE for which ALL of the following are visible in this frame - the vertical corner where it meets the wall on its left, the vertical corner where it "
            "meets the wall on its right, the line where it meets the floor, and the line where it meets the ceiling - give its four corner points in image pixels "
            f"(x to the right 0..{width}, y downward 0..{height}) in the order top-left, top-right, bottom-right, bottom-left, where top = the wall/ceiling line and bottom = the wall/floor line. "
            "Skip any wall for which a corner, the floor line or the ceiling line is hidden, cropped by the frame edge, or hidden behind furniture or bookshelves. "
            "A bookcase, door frame or window frame is not a wall face. If no wall face qualifies, return an empty list. Do not guess hidden corners.")


class CallBudgetExceeded(RuntimeError):
    pass


class GeminiCornerClient:
    """Thin wrapper over ``providers.gemini`` with a HARD call cap and usage log. ``adapter`` is injectable for tests."""

    def __init__(self, run_dir: str | Path, max_calls: int = DEFAULTS["max_gemini_calls"], adapter: Callable | None = None):
        self.calls, self.max_calls, self.log = 0, int(max_calls), []
        if adapter is None:
            from providers import gemini
            self._gem = gemini
            self._usage = gemini.UsageLog()
            self.adapter = gemini.make_adapter(run_dir=run_dir, cache=gemini.get_cache(Path(run_dir) / "gemini_cache_room_auto"), usage=self._usage, default_stage="room_auto",
                                               config={"temperature": 0.0, "media_resolution": "high"})
        else:
            self._usage, self.adapter = None, adapter

    async def walls_for(self, image_path: str | Path, width: int, height: int, tag: str) -> dict[str, Any]:
        if self.calls >= self.max_calls:
            raise CallBudgetExceeded(f"hosted call cap {self.max_calls} reached")
        self.calls += 1
        res = await self.adapter(prompt=gemini_prompt(width, height), response_schema=RESPONSE_SCHEMA, image_paths=[str(image_path)], task="room_corners", request_id=tag)
        self.log.append({"tag": tag, "metrics": res.get("metrics")})
        data = res.get("data")
        return data if isinstance(data, dict) else {"walls": [], "notes": "unparseable response"}


def validate_gemini_walls(parsed: dict[str, Any], img: np.ndarray, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Untrusted model geometry -> accepted walls (with evidence checks) and rejected ones (with the reasons)."""
    c = {**DEFAULTS, **(cfg or {})}
    h, w = img.shape[:2]
    acc, rej = [], []
    for wl in (parsed or {}).get("walls", []) or []:
        q = wl.get("quad_px")
        why = []
        if not all(wl.get(k) is True for k in ("left_corner_visible", "right_corner_visible", "floor_line_visible", "ceiling_line_visible")):
            why.append("model itself reports a corner/floor/ceiling line not visible")
        why += validate_quad_geometry(q, w, h, c) if isinstance(q, list) else ["quad missing"]
        sup = None
        if not why:
            sup = edge_support(img, q, c)
            weak = [k for k, v in sup.items() if v < c["edge_support_min"]]
            if weak:
                why.append("claimed edges not on image edges: " + ", ".join(f"{k}={sup[k]:.2f}" for k in weak))
        (rej if why else acc).append({"quad_px": [[round(float(x), 1), round(float(y), 1)] for x, y in q] if isinstance(q, list) and len(q) == 4 else q,
                                      "label": wl.get("label"), "model_confidence": wl.get("confidence"), "edge_support": sup, "reasons": why})
    return {"accepted": acc, "rejected": rej}


# ------------------------------------------------------------------------------------------------ evidence + scale
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def wall_record(wall_id: str, frame_id: str, quad_px, source: str, confidence: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"wall_id": wall_id, "frame_id": frame_id, "quad_px": [[float(x), float(y)] for x, y in quad_px], "spans_floor_to_ceiling": True,
            "confidence": round(float(confidence), 3),
            "provenance": {"source": source, "actor": PROVENANCE_ACTOR, "supplied_at": now_iso(), **(extra or {})}}


_NUM = r"(\d+(?:[.,]\d+)?)"
_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}


def parse_height_utterance(text: str) -> float | None:
    """Ceiling height in metres from a spoken phrase ('2.6 metres', 'two point six meters', '8 feet 6 inches', '260 cm'); None if no height is stated."""
    t = text.lower().replace(",", ".")
    t = re.sub(r"\b(zero|one|two|three|four|five|six|seven|eight|nine)\b", lambda m: str(_WORDS[m.group(1)]), t)
    t = re.sub(r"(\d)\s*point\s*(\d)", r"\1.\2", t)
    m = re.search(_NUM + r"\s*(?:m|meter|meters|metre|metres)\b", t)
    if m:
        return float(m.group(1))
    m = re.search(_NUM + r"\s*(?:cm|centimeter|centimeters|centimetre|centimetres)\b", t)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(_NUM + r"\s*(?:ft|feet|foot|')\s*(?:" + _NUM + r"\s*(?:in|inch|inches|\"))?", t)
    if m:
        return (float(m.group(1)) * 12 + float(m.group(2) or 0)) * 0.0254
    return None


def height_anchor_from_utterance(utterance: str, actor: str = "user_voice", tolerance_pct: float = 3.0) -> dict[str, Any] | None:
    h = parse_height_utterance(utterance)
    if h is None:
        return None
    return {"kind": "user_supplied_ceiling_height", "height_m": round(h, 3),
            "provenance": {"actor": actor, "supplied_at": now_iso(), "utterance": utterance, "tolerance_pct": tolerance_pct, "independent_of_evaluation_truth": True}}


def build_evidence(walls: list[dict[str, Any]], height_anchor: dict[str, Any] | None, declared_shape: str = "rectangular") -> dict[str, Any]:
    """``room_evidence`` document in exactly the format of docs/templates/room_evidence_template.json (extra per-wall keys are ignored by room.py)."""
    ev: dict[str, Any] = {"schema_version": 1, "purpose": "room_evidence", "walls": walls,
                          "layout": {"declared_shape": declared_shape, "wall_order": [w["wall_id"] for w in walls]}, "openings": []}
    if height_anchor:
        ev["height_anchor"] = height_anchor
    return ev


CAPTURE_STEPS = [
    ("walls_visible", "Step back to the middle of the room and slowly turn on the spot. Hold each wall in view, with both of its corners, for two seconds."),
    ("corners", "Show me every corner where two walls meet, from the floor right up to the ceiling. Hold the phone steady for two seconds on each one."),
    ("floor_line", "Tilt down a little so I can see the line where each wall meets the floor, all the way along the wall."),
    ("ceiling_line", "Tilt up a little so I can see the line where each wall meets the ceiling."),
    ("height", "Pan slowly from the floor line up to the ceiling line along a wall, then do the same in each corner, so the room height can be worked out from the video."),
]


def capture_guidance(reasons: list[str], have_height: bool) -> list[str]:
    """The exact words the agent should say, chosen from what is missing."""
    r = " ".join(reasons)
    out = []
    if not have_height:
        out.append(CAPTURE_STEPS[4][1])
    if any(k in r for k in ("fewer_than_two", "no_wall", "no_walls", "too_few", "no_auto")):
        out += [CAPTURE_STEPS[0][1], CAPTURE_STEPS[1][1]]
    if "ceiling" in r:
        out.append(CAPTURE_STEPS[3][1])
    if "floor" in r:
        out.append(CAPTURE_STEPS[2][1])
    if "corner" in r and CAPTURE_STEPS[1][1] not in out:
        out.append(CAPTURE_STEPS[1][1])
    if not out:
        out = [CAPTURE_STEPS[0][1], CAPTURE_STEPS[1][1]]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


# ------------------------------------------------------------------------------------------------ orchestration
def _frames_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["frame_id"]: f for f in manifest["data"]["frames"]}


def select_candidate_frames(manifest: dict[str, Any], max_frames: int) -> list[str]:
    """Evenly spread keyframes, preferring sharp ones (no content test here: the detectors decide)."""
    fr = sorted(manifest["data"]["frames"], key=lambda f: (f["timestamp_s"], f["frame_id"]))
    sel = [f for f in fr if f.get("selected")] or fr
    if len(sel) <= max_frames:
        return [f["frame_id"] for f in sel]
    idx = np.linspace(0, len(sel) - 1, max_frames).round().astype(int)
    out = []
    for i in idx:  # within a small neighbourhood take the sharpest frame
        lo, hi = max(0, i - 1), min(len(sel), i + 2)
        out.append(max(sel[lo:hi], key=lambda f: (f.get("quality") or {}).get("sharpness") or 0)["frame_id"])
    return list(dict.fromkeys(out))


def detect_classical_over_frames(manifest: dict[str, Any], frames_dir: str | Path, frame_ids: list[str], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    walls, per_frame = [], {}
    fr = _frames_by_id(manifest)
    for fid in frame_ids:
        img = cv2.imread(str(Path(frames_dir) / f"{fid}.jpg"))
        if img is None:
            per_frame[fid] = {"error": "image unreadable"}; continue
        r = detect_walls_classical(img, cfg)
        per_frame[fid] = {"walls": len(r["walls"]), "reasons": r["reasons"][:3], "debug": r["debug"]}
        for k, w in enumerate(r["walls"]):
            walls.append(wall_record(f"{fid}_w{k}", fid, w["quad_px"], "auto_classical", w["confidence"], {"method": w["method"], "corner_confirmed": w["corner_confirmed"]}))
    return {"walls": walls, "per_frame": per_frame}


async def detect_gemini_over_frames(client: GeminiCornerClient, manifest: dict[str, Any], frames_dir: str | Path, frame_ids: list[str], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    walls, per_frame = [], {}
    for fid in frame_ids:
        p = Path(frames_dir) / f"{fid}.jpg"
        img = cv2.imread(str(p))
        if img is None:
            per_frame[fid] = {"error": "image unreadable"}; continue
        h, w = img.shape[:2]
        try:
            parsed = await client.walls_for(p, w, h, fid)
        except CallBudgetExceeded as exc:
            per_frame[fid] = {"error": str(exc)}; break
        v = validate_gemini_walls(parsed, img, cfg)
        per_frame[fid] = {"accepted": len(v["accepted"]), "rejected": [r["reasons"] for r in v["rejected"]][:3], "notes": (parsed or {}).get("notes")}
        for k, a in enumerate(v["accepted"]):
            walls.append(wall_record(f"{fid}_g{k}", fid, a["quad_px"], "auto_gemini", a["model_confidence"] if isinstance(a["model_confidence"], (int, float)) else 0.5,
                                     {"edge_support": a["edge_support"], "model": "gemini (geometry only)"}))
    return {"walls": walls, "per_frame": per_frame, "calls": client.calls}


def estimate_from_walls(walls: list[dict[str, Any]], manifest: dict[str, Any], height_anchor: dict[str, Any] | None, calibrations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run ``pipeline.room`` (unchanged) on automatic evidence; returns room data + guidance + the evidence document."""
    from pipeline import room
    ev = build_evidence(walls, height_anchor)
    data = room.estimate_data(ev, manifest, calibrations or [], None)
    reasons = list(data.get("abstention_reasons") or [])
    if not walls:
        reasons.append("no_auto_wall_evidence")
    return {"room": data, "evidence": ev, "abstention_reasons": reasons,
            "capture_guidance": [] if data["status"] == "estimated" else capture_guidance(reasons, bool(height_anchor)),
            "provenance_note": "wall quads detected automatically (see per-wall provenance.source); the ceiling height is user-stated, not measured"}


def estimate_from_frames(manifest: dict[str, Any], frames_dir: str | Path, *, height_m: float | None = None, utterance: str | None = None, max_frames: int = 24,
                         cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classical (local) path end to end: select frames -> detect -> evidence -> room.estimate_data."""
    anchor = height_anchor_from_utterance(utterance) if utterance else None
    if anchor is None and height_m is not None:
        anchor = {"kind": "user_supplied_ceiling_height", "height_m": float(height_m), "provenance": {"actor": "user_voice", "supplied_at": now_iso(), "utterance": utterance or f"{height_m} m",
                                                                                                     "tolerance_pct": 3.0, "independent_of_evaluation_truth": True}}
    ids = select_candidate_frames(manifest, max_frames)
    det = detect_classical_over_frames(manifest, frames_dir, ids, cfg)
    res = estimate_from_walls(det["walls"], manifest, anchor)
    res["detector"] = {"kind": "auto_classical", "frames_examined": len(ids), "frames_with_walls": sum(1 for v in det["per_frame"].values() if v.get("walls")), "per_frame": det["per_frame"]}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Automatic room evidence (no human taps). Local classical detector; hosted path is opt-in and capped.")
    ap.add_argument("--run", required=True, help="run dir containing manifest.json and frames/")
    ap.add_argument("--height-m", type=float); ap.add_argument("--utterance")
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--out", help="write the result JSON here (default: <run>/room_auto.json)")
    a = ap.parse_args(argv)
    run = Path(a.run)
    manifest = json.loads((run / "manifest.json").read_text())
    res = estimate_from_frames(manifest, run / "frames", height_m=a.height_m, utterance=a.utterance, max_frames=a.max_frames)
    out = Path(a.out) if a.out else run / "room_auto.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(res["room"]["status"], res["abstention_reasons"], str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
