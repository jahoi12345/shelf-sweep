"""Stage ``room``: room dimensions and areas from observed wall evidence.

    room.estimate(manifest_path, run_dir, config) -> room_path     (room.json)
    python -m pipeline.room --run RUN --evidence EVIDENCE.json [--calibration CAL.json]

Two strictly separated parts:

1. **Deterministic arithmetic** (``polygon_area_m2``, ``rectangle_floor_area_m2``, ``wall_area_m2``, ``m2_to_ft2`` ...): exact,
   unit-tested, no estimation.
2. **Geometric estimation** (``estimate_data``) from human-marked/detected wall evidence. METHOD: each observed wall face is a
   rectangle whose four image corners (floor/ceiling x two vertical corner lines) are supplied as ``quad_px`` (TL,TR,BR,BL).
   Zhang-style rectangle rectification gives its metric ASPECT RATIO: with K = [[f,0,cx],[0,f,cy],[0,0,1]],
   width/height = |K^-1 h1| / |K^-1 h2| for the unit-square homography columns h1,h2. ``f`` is recovered in closed form from
   orthogonality of the rectangle axes (f^2 = -(a1.a2)/(h1z*h2z)), assuming square pixels, zero skew, principal point at the
   image centre (declared assumptions). A fronto-parallel wall needs no K. **Scale** comes from ONE height anchor shared by
   the walls: the floor-to-ceiling height, supplied as a user-stated ceiling height (labelled user_supplied) or from a
   floor-to-ceiling plane calibration. A single anchor does NOT give other axes or arbitrary depths: wall widths follow
   from each wall's measured aspect ratio, which needs consistent K across walls (>= 2 oblique walls agree within
   ``focal_consistency_tol``) or fronto-parallel views. Without observed walls, a height anchor and consistent intrinsics the
   stage ABSTAINS and lists exactly what to capture.

Rectangular rooms need two adjacent walls (opposite walls, if observed, must agree within ``opposite_wall_tol`` else the room
is reported inconsistent/unresolved, never forced into a rectangle). Non-rectangular rooms keep their measured wall segments
and an observed plan polygon only when explicitly provided with vertices; areas then use the shoelace formula.
Everything is ``software_only_unvalidated`` until compared with independent tape measurements.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import contracts as C
from pipeline.scale import (_dlt_homography, convex_quad, load_calibrations, oblique_index, polygon_area, validate_calibration,
                            DEFAULTS as SCALE_DEFAULTS)

FT2_PER_M2 = 10.763910416709722
VALIDATION_STATE = "software_only_unvalidated"
DEFAULTS = {"focal_consistency_tol": 0.10, "opposite_wall_tol": 0.05, "fronto_oblique_max": 1.03, "pixel_error_px": 2.0,
            "min_quad_area_px": 20000.0, "min_height_m": 1.5, "max_height_m": 6.0, "max_wall_m": 30.0}
LIMITS = [
    "Uncertainty is a stated sum of declared terms, not a statistical guarantee.",
    "Intrinsics assumptions (square pixels, zero skew, principal point at the image centre) are declared, not verified.",
    "Ceiling height, when user-supplied, is an input, not a camera measurement.",
    "Wall corner quads are human-marked or externally detected; their accuracy limits every dimension.",
    "Accuracy against tape-measured room dimensions is NOT established by this module or its synthetic tests.",
]


# ---------------------------------------------------------------------------------- deterministic arithmetic
def m2_to_ft2(m2: float | None) -> float | None:
    return None if m2 is None else m2 * FT2_PER_M2


def polygon_area_m2(vertices_m: list[list[float]]) -> float:
    """Shoelace area of a simple polygon given in metres (ordered, any winding)."""
    if len(vertices_m) < 3:
        raise ValueError("a polygon needs at least 3 vertices")
    return float(polygon_area(np.array(vertices_m, dtype=float)))


def rectangle_floor_area_m2(length_m: float, width_m: float) -> float:
    return length_m * width_m


def wall_area_m2(perimeter_m: float, height_m: float, openings_m2: float = 0.0) -> float:
    """Gross wall area minus declared openings (doors/windows). Ceiling/floor excluded."""
    return perimeter_m * height_m - openings_m2


def area_block(m2: float | None) -> dict[str, float | None]:
    return {"m2": None if m2 is None else round(m2, 4), "ft2": None if m2 is None else round(m2 * FT2_PER_M2, 4)}


def shelved_wall_area_m2(calibrations: list[dict[str, Any]]) -> tuple[float | None, list[dict[str, Any]]]:
    """Sum of user-supplied unit-face areas (plane_quad, role shelf_face, distinct plane ids). Labelled user-supplied."""
    seen, parts = set(), []
    for c in calibrations:
        if c.get("kind") != "plane_quad" or (c.get("plane") or {}).get("role") != "shelf_face" or c.get("status", "valid") != "valid":
            continue
        pid = c["plane"]["id"]
        if pid in seen:
            continue
        seen.add(pid)
        parts.append({"plane_id": pid, "calibration_id": c["calibration_id"],
                      "area_m2": round(c["metric"]["width_m"] * c["metric"]["height_m"], 4),
                      "provenance": "user-supplied unit width x height (not a camera measurement)"})
    return (round(sum(p["area_m2"] for p in parts), 4) if parts else None), parts


# --------------------------------------------------------------------------------------- geometry estimation
def quad_unit_homography(quad_px: list[list[float]]) -> np.ndarray:
    """Homography mapping the unit square (TL,TR,BR,BL) -> image pixels."""
    return _dlt_homography(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), np.array(quad_px, float))


def _vp(p1, p2, p3, p4) -> np.ndarray | None:
    """Intersection of lines (p1,p2) and (p3,p4) in pixels, or None if (near) parallel."""
    l1 = np.cross([*p1, 1.0], [*p2, 1.0]); l2 = np.cross([*p3, 1.0], [*p4, 1.0])
    v = np.cross(l1, l2)
    if abs(v[2]) < 1e-9 * max(np.linalg.norm(v[:2]), 1.0):
        return None
    return v[:2] / v[2]


def focal_from_quad(quad_px: list[list[float]], width_px: int, height_px: int, opts: dict[str, Any]) -> dict[str, Any]:
    """Closed-form focal length (px) from ONE rectangle's orthogonality, when both axes have finite vanishing points.

    A rectangle viewed with a level camera (vertical edges stay parallel) leaves f UNDETERMINED here (not zero, not guessed):
    ``f_px`` is None and ``degenerate`` is True; use a corner view (``focal_from_corner``) or a camera with some pitch.
    """
    q = np.array(quad_px, float)
    oi = oblique_index(q)
    fronto = oi <= opts["fronto_oblique_max"]
    if fronto:
        return {"f_px": None, "fronto_parallel": True, "oblique_index": round(oi, 4)}
    c = np.array([width_px / 2.0, height_px / 2.0])
    vx = _vp(q[0], q[1], q[3], q[2])       # horizontal edges (top/bottom)
    vy = _vp(q[0], q[3], q[1], q[2])       # vertical edges (left/right)
    diag = float(np.hypot(width_px, height_px))
    if vx is None or vy is None or np.linalg.norm(vx - c) > 50 * diag or np.linalg.norm(vy - c) > 50 * diag:
        return {"f_px": None, "fronto_parallel": False, "oblique_index": round(oi, 4), "degenerate": True,
                "problem": "one vanishing point at infinity (level camera): focal length not determined by this wall alone"}
    f2 = -float(np.dot(vx - c, vy - c))
    if f2 <= 0:
        return {"f_px": None, "fronto_parallel": False, "oblique_index": round(oi, 4), "problem": "orthogonality gives non-positive f^2 (bad corners?)"}
    return {"f_px": math.sqrt(f2), "fronto_parallel": False, "oblique_index": round(oi, 4), "source": "single_wall_orthogonality"}


def focal_from_corner(quad_a: list[list[float]], quad_b: list[list[float]], width_px: int, height_px: int) -> float | None:
    """Focal length from two ADJACENT walls seen in the SAME frame: their horizontal edge directions are orthogonal, so
    f^2 = -(vA-c).(vB-c) (square pixels, zero skew, principal point at centre). Works for a level camera."""
    c = np.array([width_px / 2.0, height_px / 2.0])
    qa, qb = np.array(quad_a, float), np.array(quad_b, float)
    va, vb = _vp(qa[0], qa[1], qa[3], qa[2]), _vp(qb[0], qb[1], qb[3], qb[2])
    if va is None or vb is None:
        return None
    f2 = -float(np.dot(va - c, vb - c))
    return math.sqrt(f2) if f2 > 0 else None


def aspect_from_quad(quad_px: list[list[float]], width_px: int, height_px: int, f_px: float | None) -> float:
    """Metric width/height of a rectangle from its image quad given focal length (None => fronto-parallel image aspect)."""
    H = quad_unit_homography(quad_px)
    cx, cy = width_px / 2.0, height_px / 2.0
    f = f_px if f_px else 1.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    if not f_px:  # fronto-parallel: K cancels in the ratio only if h_z terms vanish; use plain image-plane edge lengths
        q = np.array(quad_px, float)
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        return float(w / h)
    Ki = np.linalg.inv(K)
    return float(np.linalg.norm(Ki @ H[:, 0]) / np.linalg.norm(Ki @ H[:, 1]))


def validate_wall(w: dict[str, Any], frames: dict[str, dict[str, Any]], opts: dict[str, Any]) -> list[str]:
    wid, p = w.get("wall_id", "?"), []
    fr = frames.get(w.get("frame_id"))
    if fr is None:
        return [f"{wid}: frame_id {w.get('frame_id')!r} not in manifest"]
    q = w.get("quad_px")
    if not (isinstance(q, list) and len(q) == 4 and all(isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v) for v in q)):
        return [f"{wid}: quad_px must be 4 [x,y] pairs (TL,TR,BR,BL)"]
    if not all(-0.2 * fr["width_px"] <= x <= 1.2 * fr["width_px"] and -0.2 * fr["height_px"] <= y <= 1.2 * fr["height_px"] for x, y in q):
        p.append(f"{wid}: quad far outside the frame")
    a = np.array(q, float)
    if not convex_quad(a) or polygon_area(a) < opts["min_quad_area_px"]:
        p.append(f"{wid}: quad non-convex, mis-ordered or too small")
    if w.get("spans_floor_to_ceiling") is not True:
        p.append(f"{wid}: quad must be declared spans_floor_to_ceiling=true (corners on floor line and ceiling line)")
    pv = w.get("provenance") or {}
    if not pv.get("source") or not pv.get("actor") or not pv.get("supplied_at"):
        p.append(f"{wid}: provenance(source, actor, supplied_at) required")
    return p


def validate_anchor(anchor: dict[str, Any] | None, opts: dict[str, Any]) -> list[str]:
    if not anchor:
        return ["height_anchor missing"]
    p = []
    h = anchor.get("height_m")
    if not isinstance(h, (int, float)) or isinstance(h, bool) or not opts["min_height_m"] <= h <= opts["max_height_m"]:
        p.append(f"height_anchor.height_m must be within [{opts['min_height_m']}, {opts['max_height_m']}] m")
    if anchor.get("kind") not in {"user_supplied_ceiling_height", "calibration_vertical"}:
        p.append("height_anchor.kind must be user_supplied_ceiling_height or calibration_vertical (assumed heights are not anchors)")
    pv = anchor.get("provenance") or {}
    if not pv.get("actor") or not pv.get("supplied_at") or not (pv.get("utterance") or pv.get("note")):
        p.append("height_anchor.provenance(actor, supplied_at, utterance|note) required")
    if pv.get("independent_of_evaluation_truth") is not True:
        p.append("height_anchor must declare independent_of_evaluation_truth=true")
    return p


COVERAGE_ITEMS = [
    ("height_anchor", "Film one wall from the floor line up to the ceiling line (the height is worked out from the sweep; nobody is asked for a size)."),
    ("wall_faces_2_adjacent", "Show two adjacent walls floor-to-ceiling, each between two vertical corners."),
    ("floor_line_visible", "Tilt down so the floor line along each wall is visible."),
    ("ceiling_line_visible", "Tilt up so the ceiling line along each wall is visible."),
    ("opposite_wall_optional", "If possible show the opposite walls too (cross-check)."),
]


def coverage_status(evidence: dict[str, Any] | None, walls_ok: list[str], anchor_ok: bool) -> dict[str, Any]:
    ev = evidence or {}
    present = {"height_anchor": anchor_ok, "wall_faces_2_adjacent": len(walls_ok) >= 2,
               "floor_line_visible": bool(walls_ok), "ceiling_line_visible": bool(walls_ok),
               "opposite_wall_optional": len(walls_ok) >= 4}
    items = [{"item": k, "present": present[k], "optional": k == "opposite_wall_optional", "how": how} for k, how in COVERAGE_ITEMS]
    return {"items": items, "missing": [i["item"] for i in items if not i["present"] and not i["optional"]],
            "walls_with_valid_quads": walls_ok}


def estimate_data(evidence: dict[str, Any] | None, manifest: dict[str, Any], calibrations: list[dict[str, Any]],
                  config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pure estimation: evidence + manifest + calibrations -> room ``data`` (no I/O)."""
    opts = {**DEFAULTS, **{k: v for k, v in (config or {}).items() if k in DEFAULTS}}
    frames = {f["frame_id"]: f for f in manifest["data"]["frames"]}
    ev = evidence or {}
    if ev and ev.get("purpose") != "room_evidence":
        raise C.ContractError("room evidence must declare purpose='room_evidence' (evaluation truth must never be used here)")
    shelved, shelved_parts = shelved_wall_area_m2([{**c, "status": "valid"} for c in calibrations if not validate_calibration(c, frames, SCALE_DEFAULTS)])
    base = {"method": "rectangle_rectification_with_height_anchor", "options": opts, "limits": LIMITS,
            "validation": {"physical_truth_compared": False, "validation_state": VALIDATION_STATE},
            "shelved_wall": {"area": area_block(shelved), "parts": shelved_parts,
                             "provenance": "user-supplied unit dimensions" if shelved is not None else None}}
    walls_in = ev.get("walls", [])
    wall_problems, walls = {}, []
    for w in walls_in:
        pr = validate_wall(w, frames, opts)
        if pr:
            wall_problems[w.get("wall_id", "?")] = pr
        else:
            walls.append(w)
    anchor = ev.get("height_anchor")
    anchor_problems = validate_anchor(anchor, opts)
    cov = coverage_status(ev, [w["wall_id"] for w in walls], not anchor_problems)
    abst: list[str] = []
    if not walls_in:
        abst.append("no_wall_evidence_supplied")
    elif len(walls) < 2:
        abst.append("fewer_than_two_valid_walls")
    if anchor_problems:
        abst.append("height_anchor_missing_or_invalid")
    if wall_problems:
        abst.append("invalid_wall_evidence")
    result = {**base, "evidence_summary": {"walls_supplied": len(walls_in), "walls_valid": len(walls), "wall_problems": wall_problems,
                                          "anchor_problems": anchor_problems},
              "coverage": cov, "dimensions": {"length_m": None, "width_m": None, "height_m": None}, "areas": {
                  "floor": area_block(None), "wall": area_block(None), "shelved_wall": area_block(shelved)},
              "shape": {"kind": None, "rectangular_assumed": None, "walls": [], "plan_polygon_m": None},
              "uncertainty": None, "status": "abstained", "abstention_reasons": abst}
    if abst:
        return result
    # ---- focal-length candidates: single-wall orthogonality and corner views (two adjacent walls in one frame)
    per_wall, cands = [], []
    for w in walls:
        fr = frames[w["frame_id"]]
        fe = focal_from_quad(w["quad_px"], fr["width_px"], fr["height_px"], opts)
        per_wall.append({"wall_id": w["wall_id"], "frame_id": w["frame_id"], **fe})
        if fe["f_px"]:
            cands.append({"source": "single_wall_orthogonality", "walls": [w["wall_id"]], "f_px": fe["f_px"]})
    for i, wa in enumerate(walls):
        for wb in walls[i + 1:]:
            if wa["frame_id"] == wb["frame_id"]:
                fr = frames[wa["frame_id"]]
                f = focal_from_corner(wa["quad_px"], wb["quad_px"], fr["width_px"], fr["height_px"])
                if f:
                    cands.append({"source": "corner_two_adjacent_walls_one_frame", "walls": [wa["wall_id"], wb["wall_id"]], "f_px": f})
    oblique = [p for p in per_wall if not p["fronto_parallel"]]
    f_used, f_spread = None, None
    if oblique:
        if not cands:
            result["abstention_reasons"] = ["focal_length_not_determined_for_oblique_walls"]
            result["coverage"]["missing"].append("corner_view_two_walls_one_frame")
            result["coverage"]["items"].append({"item": "corner_view_two_walls_one_frame", "present": False, "optional": False,
                                                "how": "Show a vertical room corner so both adjacent walls are in one frame, or tilt the camera up/down a little."})
            result["shape"]["walls"] = per_wall
            return result
        fs = [c["f_px"] for c in cands]
        f_used = float(np.median(fs))
        f_spread = (max(fs) - min(fs)) / f_used if len(fs) > 1 else None
        result["intrinsics_candidates"] = [{**c, "f_px": round(c["f_px"], 1)} for c in cands]
        if f_spread is not None and f_spread > opts["focal_consistency_tol"]:
            result["abstention_reasons"] = ["focal_length_inconsistent_across_estimates"]
            result["shape"]["walls"] = per_wall
            result["intrinsics"] = {"f_px_values": [round(x, 1) for x in fs], "relative_spread": round(f_spread, 3)}
            return result
    h_m = float(anchor["height_m"])
    tol_anchor = float((anchor.get("provenance") or {}).get("tolerance_pct", anchor.get("tolerance_pct", 3.0))) / 100.0
    wall_out = []
    for w, pw in zip(walls, per_wall):
        fr = frames[w["frame_id"]]
        asp = aspect_from_quad(w["quad_px"], fr["width_px"], fr["height_px"], None if pw["fronto_parallel"] else f_used)
        q = np.array(w["quad_px"], float)
        min_edge = float(min(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)))
        rel = tol_anchor + 2 * opts["pixel_error_px"] / max(min_edge, 1.0) * 2 + (f_spread or 0.0)
        wall_out.append({"wall_id": w["wall_id"], "frame_id": w["frame_id"], "width_m": round(asp * h_m, 4), "height_m": h_m,
                         "aspect_w_over_h": round(asp, 4), "relative_uncertainty": round(rel, 4),
                         "method": "fronto_parallel_image_aspect" if pw["fronto_parallel"] else "k_inverse_rectification",
                         "provenance": w.get("provenance")})
    result["shape"]["walls"] = wall_out
    result["intrinsics"] = {"f_px": None if f_used is None else round(f_used, 1), "relative_spread": None if f_spread is None else round(f_spread, 4),
                            "assumptions": ["square pixels", "zero skew", "principal point at image centre"], "validated": False}
    layout = ev.get("layout") or {}
    order = layout.get("wall_order") or [w["wall_id"] for w in walls]
    by_id = {w["wall_id"]: w for w in wall_out}
    ordered = [by_id[i] for i in order if i in by_id]
    shape = layout.get("declared_shape", "rectangular")
    if shape == "polygon":
        pv = layout.get("plan_vertices_m")
        if not pv:
            result["abstention_reasons"] = ["polygon_room_requires_observed_plan_vertices"]
            result["shape"].update(kind="polygon_unresolved", rectangular_assumed=False)
            return result
        result["shape"].update(kind="polygon", rectangular_assumed=False, plan_polygon_m=pv)
        floor = polygon_area_m2(pv)
        perim = float(sum(math.dist(pv[i], pv[(i + 1) % len(pv)]) for i in range(len(pv))))
        result["dimensions"]["height_m"] = h_m
    else:
        if len(ordered) < 2:
            result["abstention_reasons"] = ["fewer_than_two_ordered_adjacent_walls"]
            return result
        a, b = ordered[0], ordered[1]
        for opp, adj in (((ordered[2] if len(ordered) > 2 else None), a), ((ordered[3] if len(ordered) > 3 else None), b)):
            if opp and abs(opp["width_m"] - adj["width_m"]) / adj["width_m"] > opts["opposite_wall_tol"]:
                result["abstention_reasons"] = ["opposite_walls_disagree_room_not_confirmed_rectangular"]
                result["shape"].update(kind="unresolved", rectangular_assumed=False)
                return result
        length, width = max(a["width_m"], b["width_m"]), min(a["width_m"], b["width_m"])
        result["shape"].update(kind="rectangular", rectangular_assumed=True,
                               assumption="right-angle corners assumed from rectangular wall evidence; opposite walls " +
                                          ("cross-checked" if len(ordered) >= 4 else "NOT observed"),
                               plan_polygon_m=[[0, 0], [length, 0], [length, width], [0, width]])
        result["dimensions"].update(length_m=round(length, 3), width_m=round(width, 3), height_m=h_m)
        floor = rectangle_floor_area_m2(length, width)
        perim = 2 * (length + width)
    openings = sum(float(o.get("width_m", 0)) * float(o.get("height_m", 0)) for o in ev.get("openings", [])
                   if o.get("provenance", {}).get("actor"))
    gross_wall = wall_area_m2(perim, h_m, openings)
    result["areas"].update(floor=area_block(floor), wall=area_block(gross_wall))
    result["areas"]["openings_subtracted_m2"] = round(openings, 4)
    result["areas"]["wall_basis"] = ("perimeter x height minus declared openings" if openings else
                                     "perimeter x height; no openings declared or subtracted")
    rels = [w["relative_uncertainty"] for w in wall_out]
    u = max(rels)
    result["uncertainty"] = {"relative_length_width": round(u, 4), "relative_floor_area": round(2 * u, 4), "relative_wall_area": round(u + tol_anchor, 4),
                             "note": "stated sum of declared terms, not a statistical guarantee"}
    result["height_provenance"] = {"kind": anchor["kind"], "label": "user-supplied input, not a camera measurement"
                                   if anchor["kind"] == "user_supplied_ceiling_height" else "from a calibrated vertical plane"}
    result["status"] = "estimated"
    result["abstention_reasons"] = []
    return result


def estimate(manifest_path: str | Path, run_dir: str | Path, config: dict[str, Any] | None = None) -> Path:
    t0 = time.perf_counter()
    config = config or {}
    run_dir = Path(run_dir)
    manifest = C.read_json(manifest_path)
    evp = config.get("room_evidence_path") or (run_dir / "room_evidence.json")
    evidence = C.read_json(evp) if evp and Path(evp).is_file() else None
    cals = config.get("calibrations")
    if cals is None:
        cals = load_calibrations(config.get("calibration_path") or (run_dir / "calibration.json"))
    data = estimate_data(evidence, manifest, cals, config.get("room"))
    issues = [{"code": r, "severity": "info", "detail": r} for r in data["abstention_reasons"]]
    env = C.envelope(config.get("run_id") or run_dir.name, "room", data, status="ok" if data["status"] == "estimated" else "partial",
                     issues=issues, metrics={"elapsed_s": round(time.perf_counter() - t0, 4), "request_count": 0, "cost": 0,
                                             "cost_provenance": "local geometry; no API calls"})
    return C.write_json(run_dir / C.ARTIFACTS["room"], env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--evidence"); ap.add_argument("--calibration")
    a = ap.parse_args(argv)
    run = Path(a.run)
    print(estimate(run / C.ARTIFACTS["frames"], run, {"room_evidence_path": a.evidence, "calibration_path": a.calibration}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
