"""Room geometry from the sweep alone: ceiling height, floor area, wall areas from the run's 3D reconstruction + the automatic metric scale.

    auto_room.estimate(recon3d_dict, scale_block, markers_sizes=None) -> room block        (called by pipeline.auto_scale.measure_data)

Nobody is asked to measure, state or tap anything. The reconstruction (pipeline/recon3d.py) is UNSCALED and aligned to the shelf wall: that plane is z = 0, the
room is on the +z side, +y is up (spine-up), x runs along the shelf. A rectangular room is a box in that frame, so the surfaces are axis-aligned planes:

* floor / ceiling: horizontal planes (constant y) below / above every camera centre;
* opposite wall: a plane z = const beyond every camera centre; side walls: planes x = const left / right of every camera centre; the shelf wall is z = 0.

Each plane is found as a peak of the point-cloud histogram along its axis that is a real SURFACE (its in-slab points fill a large share of the cloud's extent in the
other two axes), so a table top, a shelf board or a line of books is not mistaken for a floor. Lengths are plane-to-plane distances x cm/unit from
``pipeline.auto_scale`` (catalogue scale, or the labelled prior). Every quantity abstains with its own reason when its planes were not found (not filmed, not
reconstructed, or implausible); nothing is ever 0. A room that is not a box is NOT modelled: the result says so instead of forcing a rectangle.
Uncertainty is a stated sum of declared terms (scale, plane localisation); ``validation_state: software_only_unvalidated``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

FT2_PER_M2 = 10.763910416709722
DEFAULTS = {
    "bin_frac": 0.004,            # histogram bin width / robust cloud extent
    "slab_frac": 0.010,           # plane slab half-width / extent (reconstruction noise on a surface)
    "min_points": 80,
    "min_point_frac": 0.012,      # a surface holds at least this share of the cloud
    "peak_to_floor_ratio": 3.0,   # peak bin vs the mean occupied bin (a surface, not background)
    "min_fill": 0.22,             # share of the 10x10 grid cells over the cloud's extent that the slab points occupy
    "min_span_frac": 0.30,        # the slab points span at least this share of the cloud extent on both in-plane axes
    "camera_margin_frac": 0.01,   # a floor/ceiling/wall must clear the camera centres by this share of the extent
    "min_height_m": 1.8, "max_height_m": 6.0, "max_room_side_m": 25.0, "min_room_side_m": 1.5,
    "tilt_note_deg": 10.0,
}
LIMITS = [
    "Rectangular (box) rooms only: surfaces are assumed axis-aligned with the shelf wall; other shapes abstain, they are not forced into a rectangle.",
    "Planes are fitted to sparse structure-from-motion points; textureless walls, ceilings and floors may not reconstruct, in which case the quantity abstains.",
    "Wall area is the gross area of the four walls (doors, windows and other openings are not subtracted).",
    "Uncertainty is a stated sum of declared terms, not a statistical guarantee; accuracy against tape is NOT established.",
]


def _robust_extent(P: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(P, [1, 99], axis=0)
    return np.maximum(hi - lo, 1e-9)


def axis_planes(P: np.ndarray, axis: int, ext: np.ndarray, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Surfaces (constant coordinate on ``axis``) of the cloud, strongest first: [{c, n, rms, fill, spans:[fa, fb]}]."""
    E = float(ext.max())
    others = [i for i in range(3) if i != axis]
    c = P[:, axis]
    lo, hi = np.percentile(c, [0.2, 99.8])
    if hi - lo < 1e-9:
        return []
    w = cfg["bin_frac"] * E
    nb = int(max(8, min(4000, math.ceil((hi - lo) / w))))
    hist, edges = np.histogram(c, bins=nb, range=(lo, hi))
    k = np.ones(3) / 3.0
    sm = np.convolve(hist, k, mode="same")
    occupied = sm[sm > 0]
    if occupied.size == 0:
        return []
    thresh = max(cfg["min_points"], cfg["min_point_frac"] * len(P), cfg["peak_to_floor_ratio"] * float(occupied.mean()))
    peaks = [i for i in range(1, nb - 1) if sm[i] >= thresh and sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]]
    peaks.sort(key=lambda i: -sm[i])
    slab = cfg["slab_frac"] * E
    chosen, planes = [], []
    for i in peaks:
        centre = 0.5 * (edges[i] + edges[i + 1])
        if any(abs(centre - p) < 3 * slab for p in chosen):
            continue
        sel = np.abs(c - centre) <= slab
        n = int(sel.sum())
        if n < max(cfg["min_points"], cfg["min_point_frac"] * len(P)):
            continue
        c0 = float(np.median(c[sel]))
        Q = P[sel][:, others]
        spans = [float((np.percentile(Q[:, j], 95) - np.percentile(Q[:, j], 5)) / ext[others[j]]) for j in range(2)]
        grid = set()
        loA, loB = np.percentile(P[:, others[0]], 1), np.percentile(P[:, others[1]], 1)
        for a, b in Q:
            ia, ib = int(min(9, max(0, (a - loA) / ext[others[0]] * 10))), int(min(9, max(0, (b - loB) / ext[others[1]] * 10)))
            grid.add((ia, ib))
        fill = len(grid) / 100.0
        if fill < cfg["min_fill"] or min(spans) < cfg["min_span_frac"]:
            continue
        chosen.append(c0)
        planes.append({"c": c0, "n": n, "rms": float(np.std(c[sel])), "fill": round(fill, 2), "spans": [round(s, 2) for s in spans]})
    return planes


def _pick(planes: list[dict[str, Any]], side: str, bound: float, margin: float, strongest_ratio: float = 0.5) -> dict[str, Any] | None:
    """Among planes beyond ``bound`` (below it for side 'low', above it for 'high'), the outermost one holding >= half of the strongest candidate's points."""
    cand = [p for p in planes if (p["c"] < bound - margin if side == "low" else p["c"] > bound + margin)]
    if not cand:
        return None
    top = max(p["n"] for p in cand)
    ok = [p for p in cand if p["n"] >= strongest_ratio * top]
    return min(ok, key=lambda p: p["c"]) if side == "low" else max(ok, key=lambda p: p["c"])


def observing_frame(cameras: list[dict[str, Any]], axis: int, c: float) -> str | None:
    """The registered frame whose optical axis most squarely looks at the plane (coordinate ``c`` on ``axis``) from the camera side; evidence for 'this plane was filmed'."""
    best = None
    for cam in cameras:
        if cam.get("rotation_world_to_cam") is None or cam.get("frame_id") is None:
            continue
        f = np.asarray(cam["rotation_world_to_cam"], float).reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
        o = np.asarray(cam["center"], float)
        if abs(f[axis]) < 0.3 or (c - o[axis]) / f[axis] <= 0:
            continue
        if best is None or abs(f[axis]) > best[0]:
            best = (abs(f[axis]), cam["frame_id"])
    return best[1] if best else None


def _val(m: float | None, unc_pct: float | None, why: str | None, **extra: Any) -> dict[str, Any]:
    return {"value": None if m is None else round(m, 2), "relative_uncertainty_pct": None if unc_pct is None else round(unc_pct, 1), "reason": why, **extra}


def estimate(recon: dict[str, Any], scale: dict[str, Any], cfg: dict[str, Any] | None = None, shelf: dict[str, Any] | None = None) -> dict[str, Any]:
    """Room block from a recon3d dict and a scale block ({cm_per_unit, relative_uncertainty_pct, source}). ``shelf``: optional auto_scale shelf geometry (shelved-wall area and
    ``books_y_range_units`` = [lowest book bottom, highest book top]: a floor must lie below the lowest book and a ceiling above the highest, which rejects shelf boards)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    out: dict[str, Any] = {"method": "axis-aligned plane fit of the 3D reconstruction x automatic metric scale", "validation_state": "software_only_unvalidated",
                           "limits": LIMITS, "planes": {}, "abstention_reasons": []}
    s = scale.get("cm_per_unit")
    u_s = (scale.get("relative_uncertainty_pct") or 0.0) / 100.0
    why: list[str] = out["abstention_reasons"]
    pts = np.asarray(recon.get("points") or [], float).reshape(-1, 3)
    cams = np.asarray([c["center"] for c in recon.get("cameras") or []], float).reshape(-1, 3)
    if len(pts) < 200 or len(cams) < 3:
        why.append(f"reconstruction_too_sparse:{len(pts)}_points_{len(cams)}_cameras")
        return {**out, "status": "abstained", "dimensions": {}, "areas": {}, "scale": {k: scale.get(k) for k in ("source", "cm_per_unit")}}
    pts = pts[np.isfinite(pts).all(axis=1)]
    ext = _robust_extent(np.vstack([pts, cams]))
    E = float(ext.max())
    margin = cfg["camera_margin_frac"] * E
    px, py, pz = (axis_planes(pts, a, ext, cfg) for a in (0, 1, 2))
    out["planes"]["candidates_found"] = {"x": len(px), "y": len(py), "z": len(pz)}
    yr = (shelf or {}).get("books_y_range_units")
    floor_bound = float(cams[:, 1].min()) if not yr else min(float(cams[:, 1].min()), float(yr[0]))
    ceil_bound = float(cams[:, 1].max()) if not yr else max(float(cams[:, 1].max()), float(yr[1]))
    floor = _pick(py, "low", floor_bound, margin)
    ceil = _pick(py, "high", ceil_bound, margin)
    opp = _pick(pz, "high", float(cams[:, 2].max()), margin)
    left = _pick(px, "low", float(cams[:, 0].min()), margin)
    right = _pick(px, "high", float(cams[:, 0].max()), margin)
    camlist = recon.get("cameras") or []
    for plane, axis in ((floor, 1), (ceil, 1), (opp, 2), (left, 0), (right, 0)):
        if plane is not None:
            plane["frame_id"] = observing_frame(camlist, axis, plane["c"])
    out["planes"].update(floor=floor, ceiling=ceil, opposite_wall=opp, left_wall=left, right_wall=right,
                         shelf_wall={"c": 0.0, "frame_id": observing_frame(camlist, 2, 0.0), "note": "the fitted shelf plane z = 0"})

    def noted(name: str, plane: dict[str, Any] | None, label: str) -> None:
        if plane is None:
            why.append(f"{name}_not_found:no_surface_in_the_reconstruction_{label}")

    noted("floor", floor, "below_every_camera_and_the_lowest_book(floor_not_filmed_or_not_reconstructed)")
    noted("ceiling", ceil, "above_every_camera_and_the_highest_book(ceiling_not_filmed_or_not_reconstructed)")
    noted("opposite_wall", opp, "beyond_every_camera")
    noted("left_wall", left, "left_of_every_camera")
    noted("right_wall", right, "right_of_every_camera")
    dims: dict[str, Any] = {}
    areas: dict[str, Any] = {}
    out["scale"] = {k: scale.get(k) for k in ("source", "cm_per_unit", "relative_uncertainty_pct", "not_a_measurement", "reason")}
    if not s:
        why.append("no_metric_scale:" + str(scale.get("reason") or "unavailable"))
    plane_unc = lambda p: p["rms"] / max(E, 1e-9)                       # localisation noise relative to the cloud extent (declared term)

    def dist(a: dict[str, Any] | None, b_c: float, b_rms: float, name: str, lo_m: float, hi_m: float) -> dict[str, Any]:
        if a is None:
            return _val(None, None, f"{name}_planes_missing")
        if not s:
            return _val(None, None, "no_metric_scale")
        units = abs(a["c"] - b_c)
        m = units * s / 100.0
        if not (lo_m <= m <= hi_m):
            return _val(None, None, f"implausible_{name}:{m:.2f}m_outside_{lo_m}-{hi_m}m", rejected_value_m=round(m, 2))
        u = math.hypot(u_s, math.hypot(a["rms"], b_rms) / max(units, 1e-9))
        return _val(m, u * 100.0, None, units=round(units, 4))

    dims["height_m"] = dist(ceil, floor["c"], floor["rms"], "ceiling_height", cfg["min_height_m"], cfg["max_height_m"]) if (floor and ceil) else _val(
        None, None, "floor_or_ceiling_plane_missing")
    dims["depth_m"] = dist(opp, 0.0, 0.0, "room_depth", cfg["min_room_side_m"], cfg["max_room_side_m"]) if opp else _val(None, None, "opposite_wall_plane_missing")
    dims["width_m"] = dist(left, right["c"], right["rms"], "room_width", cfg["min_room_side_m"], cfg["max_room_side_m"]) if (left and right) else _val(
        None, None, "side_wall_planes_missing")
    out["dimensions"] = dims
    H, D, Wd = (dims[k]["value"] for k in ("height_m", "depth_m", "width_m"))
    uh, ud, uw = ((dims[k]["relative_uncertainty_pct"] or 0.0) / 100.0 for k in ("height_m", "depth_m", "width_m"))
    if Wd is not None and D is not None:
        # the scale error is common to both sides (area ~ scale^2); plane terms are independent
        pw, pd = (math.sqrt(max(u * u - u_s * u_s, 0.0)) for u in (uw, ud))
        u_floor = math.sqrt((2 * u_s) ** 2 + pw ** 2 + pd ** 2)
        areas["floor_area_m2"] = _val(Wd * D, u_floor * 100.0, None, ft2=round(Wd * D * FT2_PER_M2, 1), formula="width x depth (rectangle)")
    else:
        areas["floor_area_m2"] = _val(None, None, "needs_all_four_walls:" + ("," .join(k for k, v in (("opposite_wall", opp), ("left_wall", left), ("right_wall", right)) if v is None) or "scale"))
    if Wd is not None and D is not None and H is not None:
        gross = 2 * (Wd + D) * H
        ph = math.sqrt(max(uh * uh - u_s * u_s, 0.0))
        pw, pd = (math.sqrt(max(u * u - u_s * u_s, 0.0)) for u in (uw, ud))
        per = math.hypot(Wd * pw, D * pd) / max(Wd + D, 1e-9)
        u_wall = math.sqrt((2 * u_s) ** 2 + ph ** 2 + per ** 2)
        areas["wall_area_m2"] = _val(gross, u_wall * 100.0, None, ft2=round(gross * FT2_PER_M2, 1), formula="2 x (width + depth) x height, openings NOT subtracted")
    else:
        areas["wall_area_m2"] = _val(None, None, "needs_four_walls_and_floor_and_ceiling")
    sh = (shelf or {}).get("shelved_face_m2")
    areas["shelved_wall_area_m2"] = _val(sh, None if sh is None else math.hypot(2 * u_s, 0.05) * 100.0, None if sh is not None else "no_measured_books_or_no_scale",
                                         formula="bounding rectangle of the measured books on the shelf face (shelving larger than the books is not included)")
    out["areas"] = areas
    known = [k for k, v in dims.items() if v["value"] is not None]
    box_complete = all(dims[k]["value"] is not None for k in ("height_m", "depth_m", "width_m"))
    out["status"] = "estimated" if box_complete else "partial" if known else "abstained"
    out["shape"] = {"kind": "rectangular" if box_complete else None,
                    "assumption": "box room: four vertical walls at right angles to the shelf wall, horizontal floor and ceiling" if box_complete else "not established"}
    if not box_complete and not why:
        why.append("incomplete_room_geometry")
    out["capture_guidance"] = guidance(out)
    return out


def guidance(room: dict[str, Any]) -> list[str]:
    """What to FILM (never what to measure or say) for the missing pieces."""
    r = " ".join(room.get("abstention_reasons") or [])
    g = []
    if "floor_not_found" in r:
        g.append("Tilt down so the floor along the wall is in view, with the wall-floor line visible, then pan along it.")
    if "ceiling_not_found" in r:
        g.append("Tilt up so the ceiling and the wall-ceiling line are in view, then pan along it.")
    if any(k in r for k in ("opposite_wall_not_found", "left_wall_not_found", "right_wall_not_found")):
        g.append("Pan slowly to the opposite wall and to each room corner, from the floor up to the ceiling, so all four walls are in the video.")
    return g
