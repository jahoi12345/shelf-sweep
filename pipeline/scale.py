"""Stage ``scale``: evidence-backed spine height/thickness and shelf run.

    scale.measure(inventory_path, manifest_path, run_dir, config) -> scale_path      (scale.json)
    python -m pipeline.scale --run RUN --calibration CAL.json                        (stage alone)
    python -m pipeline.scale --feasibility --video V --out DIR                        (bounded focal/vanishing-point check)

METHOD (one practical method for a continuous RGB sweep, no marker): **planar metric rectification from a
human-supplied calibration on the shelf plane**. A calibration is attributable human input (utterance/note, actor,
time) that gives known metric lengths for image points lying on one plane, in one saved keyframe:

* ``plane_quad``: four image corners (TL,TR,BR,BL) of a rectangle on the shelf face with known width and height.
  Four point correspondences determine a homography image->metric (no intrinsics needed), so *every axis and
  oblique view of that plane* is metric. This is the only kind that supports height AND thickness without further
  assumptions.
* ``segment``: two image points with a known length. One length fixes ONE direction only. It scales lengths parallel to
  that image axis (within ``axis_tolerance_deg``); the orthogonal axis abstains unless the human explicitly declares
  ``assume_fronto_parallel`` (then both axes are scaled with an extra stated uncertainty and the output is flagged).

Geometry limits, stated rather than hidden: spines sit a few centimetres behind the calibrated shelf face, so a
plane-offset uncertainty term (assumption, default 4 %) is added; one plane's scale is NEVER applied to objects outside
its region (``region_margin_frac``) or in unrelated planes; calibration is transferred to other keyframes only through
dedup's accepted frame-to-frame homographies, at most ``max_hops`` and with per-hop uncertainty. Spine axes come from the
detector's oriented polygon (so leaning and horizontal books use their own axes): long edge = spine height (length of
the spine), short edge = thickness; near-square footprints, truncated, heavily/partly occluded or off-plane
observations ABSTAIN with explicit reasons. Occlusion "unknown" is accepted with an added uncertainty and a flag.
Per-physical-instance values aggregate the instance's observations by median (never summing duplicates) and abstain when
they disagree by more than ``max_observation_spread``. ``shelf_run_m`` is reported only when every detected book in a
calibrated shelf-face region is measured and none has an uncertain association; otherwise it stays null and a partial
sum is reported separately, labelled not a shelf run.

Uncertainty is a stated sum of declared terms, NOT a statistical guarantee. Everything is ``validation_state:
software_only_unvalidated`` until compared with independently collected tape measurements (``eval/``; this module never
reads evaluation truth, and calibrations must declare ``independent_of_evaluation_truth: true``).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline import contracts as C

SCALE_VERSION = 1
DEFAULTS = {
    "plane_offset_pct": 4.0,          # ASSUMPTION: spine face recessed behind calibrated plane (unvalidated)
    "pixel_error_px": 2.0,            # per-edge localisation error assumed for detector polygons (unvalidated)
    "occlusion_unknown_pct": 3.0,     # added when the detector gives no occlusion state
    "hop_pct": 2.5,                   # added per registration hop when transferring a calibration (unvalidated)
    "max_hops": 3,
    "region_margin_frac": 0.15,       # allowed margin outside the calibrated quad (fraction of its diagonal)
    "min_axis_ratio": 1.5,            # long/short edge ratio below which axes are ambiguous -> abstain
    "max_observation_spread": 0.15,   # observation disagreement above which an instance abstains
    "axis_tolerance_deg": 20.0,
    "fronto_parallel_extra_pct": 8.0,
    "min_quad_area_px": 4000.0,
    "frame_edge_margin_px": 3,
    "min_calibration_m": 0.03, "max_calibration_m": 25.0,
    "min_registration_inlier_ratio": 0.25, "max_registration_reproj_px": 3.5,
    "run_requires_unique_associations": True,
}
VALIDATION_STATE = "software_only_unvalidated"
LIMITS = [
    "Uncertainty is a stated sum of declared terms, not a statistical guarantee.",
    "Plane-offset, pixel-error, per-hop and occlusion terms are assumptions that have not been validated against physical truth.",
    "A calibration scales only its own plane region; objects off the plane or outside the region abstain.",
    "Accuracy against tape measurements is NOT established by this module or its synthetic tests.",
]
REQUIRED_PROVENANCE = ("source", "actor", "supplied_at")


# ------------------------------------------------------------------------------------------------ geometry
def convex_quad(pts: np.ndarray) -> bool:
    """True if the 4 points (ordered) form a convex, non-self-intersecting quadrilateral."""
    signs = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def quad_homography(image_pts: list[list[float]], width_m: float, height_m: float) -> np.ndarray:
    """Homography mapping image pixels -> plane metres (origin TL, x right, y down) from a TL,TR,BR,BL quad."""
    src = np.array(image_pts, dtype=np.float64)
    dst = np.array([[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]], dtype=np.float64)
    return _dlt_homography(src, dst)


def _dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact 4-point DLT (float64) so tests can check metric accuracy without float32 rounding."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(A))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: Any) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = (H @ np.hstack([p, np.ones((len(p), 1))]).T).T
    return q[:, :2] / q[:, 2:3]


def oblique_index(image_pts: np.ndarray) -> float:
    """Ratio of the longer to shorter of each opposite-edge pair (1.0 = fronto-parallel); max over both pairs."""
    e = [np.linalg.norm(image_pts[(i + 1) % 4] - image_pts[i]) for i in range(4)]
    r1, r2 = max(e[0], e[2]) / max(min(e[0], e[2]), 1e-9), max(e[1], e[3]) / max(min(e[1], e[3]), 1e-9)
    return float(max(r1, r2))


# ------------------------------------------------------------------------------------------ calibration I/O
def validate_calibration(cal: dict[str, Any], frames: dict[str, dict[str, Any]], opts: dict[str, Any]) -> list[str]:
    """Return problems (empty = usable). Never trusts a calibration lacking provenance or independence."""
    problems: list[str] = []
    cid = cal.get("calibration_id", "?")
    prov = cal.get("provenance") or {}
    for k in REQUIRED_PROVENANCE:
        if not prov.get(k):
            problems.append(f"{cid}: provenance.{k} missing")
    if not (prov.get("utterance") or prov.get("note")):
        problems.append(f"{cid}: provenance needs the human's utterance or note")
    if prov.get("independent_of_evaluation_truth") is not True:
        problems.append(f"{cid}: must declare independent_of_evaluation_truth=true (calibration input may not be held-out truth)")
    if prov.get("source") not in {"user_supplied", "measured_by_user"}:
        problems.append(f"{cid}: source must be user_supplied or measured_by_user (assumed sizes are not calibrations)")
    kind = cal.get("kind")
    fr = frames.get(cal.get("frame_id"))
    if fr is None:
        return problems + [f"{cid}: frame_id {cal.get('frame_id')!r} not in manifest"]
    pts = cal.get("image_points_px")
    need = {"plane_quad": 4, "segment": 2}.get(kind)
    if need is None:
        return problems + [f"{cid}: unknown kind {kind!r}"]
    if not isinstance(pts, list) or len(pts) != need or not all(isinstance(p, (list, tuple)) and len(p) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p) for p in pts):
        return problems + [f"{cid}: image_points_px must be {need} [x,y] pairs"]
    if not all(0 <= x <= fr["width_px"] and 0 <= y <= fr["height_px"] for x, y in pts):
        problems.append(f"{cid}: points outside the {fr['width_px']}x{fr['height_px']} frame")
    m = cal.get("metric") or {}
    dims = ("width_m", "height_m") if kind == "plane_quad" else ("length_m",)
    for d in dims:
        v = m.get(d)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not opts["min_calibration_m"] <= v <= opts["max_calibration_m"]:
            problems.append(f"{cid}: metric.{d} must be a number within [{opts['min_calibration_m']}, {opts['max_calibration_m']}] m")
    tol = m.get("tolerance_pct")
    if not isinstance(tol, (int, float)) or isinstance(tol, bool) or not 0 <= tol <= 25:
        problems.append(f"{cid}: metric.tolerance_pct (how uncertain the human's number is) must be within [0, 25]")
    if kind == "plane_quad" and not problems:
        a = np.array(pts, float)
        if not convex_quad(a) or polygon_area(a) < opts["min_quad_area_px"]:
            problems.append(f"{cid}: quad is non-convex, mis-ordered (use TL,TR,BR,BL) or too small")
    if kind == "segment" and not problems:
        a = np.array(pts, float)
        if np.linalg.norm(a[1] - a[0]) < 40:
            problems.append(f"{cid}: segment shorter than 40 px")
    if not (cal.get("plane") or {}).get("id"):
        problems.append(f"{cid}: plane.id required (scope of the calibration)")
    return problems


# --------------------------------------------------------------------------- frame-to-frame transfer
def build_transfer_graph(registrations: list[dict[str, Any]], opts: dict[str, Any]) -> dict[str, list[tuple[str, np.ndarray, dict]]]:
    g: dict[str, list[tuple[str, np.ndarray, dict]]] = defaultdict(list)
    for r in registrations or []:
        if r.get("status") != "accepted" or not r.get("homography"):
            continue
        if (r.get("inlier_ratio") or 0) < opts["min_registration_inlier_ratio"] or (r.get("median_reprojection_px") or 99) > opts["max_registration_reproj_px"]:
            continue
        a, b = r["frame_ids"]
        H = np.array(r["homography"], dtype=np.float64)
        g[a].append((b, H, r))          # image(a) -> image(b)
        try:
            g[b].append((a, np.linalg.inv(H), r))
        except np.linalg.LinAlgError:
            continue
    return g


def transfer_paths(graph, source: str, max_hops: int) -> dict[str, tuple[np.ndarray, int]]:
    """BFS shortest-hop homography image(source)->image(target) for every frame within max_hops."""
    out = {source: (np.eye(3), 0)}
    q = deque([source])
    while q:
        cur = q.popleft()
        H, hops = out[cur]
        if hops >= max_hops:
            continue
        for nxt, Hn, _ in graph.get(cur, []):
            if nxt not in out:
                M = Hn @ H
                out[nxt] = (M / M[2, 2], hops + 1)
                q.append(nxt)
    return out


# -------------------------------------------------------------------------------------------- measurement
def _axis_angle(v: np.ndarray) -> float:
    return math.degrees(math.atan2(v[1], v[0])) % 180.0


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def measure_polygon(poly_px: np.ndarray, cal: dict[str, Any], H_cal_to_target: np.ndarray, hops: int,
                    opts: dict[str, Any], occlusion: str | None) -> dict[str, Any]:
    """Measure one oriented polygon (4 corners, image px of the TARGET frame) with one calibration.

    Returns {"ok": True, height_cm, thickness_cm, rel_uncertainty, edges_cm, ...} or {"ok": False, "reasons": [...]}.
    """
    reasons: list[str] = []
    cal_pts = np.array(cal["image_points_px"], float)
    # polygon corners expressed in the calibration frame's image coordinates
    try:
        p_cal = apply_h(np.linalg.inv(H_cal_to_target), poly_px) if hops else poly_px.astype(float)
    except np.linalg.LinAlgError:
        return {"ok": False, "reasons": ["transfer_homography_singular"]}
    kind = cal["kind"]
    m = cal["metric"]
    if kind == "plane_quad":
        quad = cal_pts
        diag = float(np.linalg.norm(quad[2] - quad[0]))
        margin = opts["region_margin_frac"] * diag
        inside = cv2.pointPolygonTest(quad.astype(np.float32).reshape(-1, 1, 2), tuple(p_cal.mean(axis=0).astype(float)), True)
        if inside < -margin:
            return {"ok": False, "reasons": ["outside_calibrated_plane_region"]}
        Hm = _dlt_homography(quad, np.array([[0, 0], [m["width_m"], 0], [m["width_m"], m["height_m"]], [0, m["height_m"]]], float))
        met = apply_h(Hm, p_cal)
        edges_m = [float(np.linalg.norm(met[(i + 1) % 4] - met[i])) for i in range(4)]
        extra_pct = 0.0
        scale_note = "plane_quad homography"
    else:  # segment
        a, b = cal_pts
        axis_vec = b - a
        axis_ang = _axis_angle(axis_vec)
        s = m["length_m"] / float(np.linalg.norm(axis_vec))          # metres per pixel along the anchor
        mid = (a + b) / 2
        reach = 4.0 * float(np.linalg.norm(axis_vec))
        if np.linalg.norm(p_cal.mean(axis=0) - mid) > reach:
            return {"ok": False, "reasons": ["outside_calibrated_plane_region"]}
        edges_px = [p_cal[(i + 1) % 4] - p_cal[i] for i in range(4)]
        fp = bool((cal.get("assumptions") or {}).get("assume_fronto_parallel"))
        edges_m = []
        for e in edges_px:
            along = _ang_diff(_axis_angle(e), axis_ang) <= opts["axis_tolerance_deg"]
            across = _ang_diff(_axis_angle(e), axis_ang + 90.0) <= opts["axis_tolerance_deg"]
            if along or (across and fp):
                edges_m.append(float(np.linalg.norm(e)) * s)
            else:
                edges_m.append(float("nan"))
        extra_pct = opts["fronto_parallel_extra_pct"] if fp else 0.0
        scale_note = "segment, fronto-parallel assumed (declared)" if fp else "segment, along anchor axis only"
    # long edges = spine length (height), short edges = thickness
    pairs = [(edges_m[0], edges_m[2]), (edges_m[1], edges_m[3])]
    means = []
    for p in pairs:
        means.append(float(np.nanmean(p)) if not all(math.isnan(v) for v in p) else float("nan"))
    px_edges = [float(np.linalg.norm(poly_px[(i + 1) % 4] - poly_px[i])) for i in range(4)]
    px_pairs = [(px_edges[0] + px_edges[2]) / 2, (px_edges[1] + px_edges[3]) / 2]
    long_i = 0 if px_pairs[0] >= px_pairs[1] else 1
    short_i = 1 - long_i
    if min(px_pairs) < 1 or max(px_pairs) / max(min(px_pairs), 1e-6) < opts["min_axis_ratio"]:
        return {"ok": False, "reasons": ["ambiguous_axes_near_square_footprint"]}
    height_m, thick_m = means[long_i], means[short_i]
    if math.isnan(height_m):
        reasons.append("single_segment_anchor_does_not_determine_spine_length_axis")
    if math.isnan(thick_m):
        reasons.append("single_segment_anchor_does_not_determine_thickness_axis")
    if reasons and kind == "segment":
        return {"ok": False, "reasons": reasons, "partial": {"height_cm": None if math.isnan(height_m) else height_m * 100,
                                                          "thickness_cm": None if math.isnan(thick_m) else thick_m * 100}}
    base_pct = (float(m.get("tolerance_pct", 0)) + opts["plane_offset_pct"] + extra_pct + opts["hop_pct"] * hops
                + (opts["occlusion_unknown_pct"] if occlusion in (None, "unknown") else 0.0))
    def rel(value_m: float, px_len: float) -> float:
        pix_pct = 100.0 * (2 * opts["pixel_error_px"]) / max(px_len, 1.0)  # both endpoints
        return (base_pct + pix_pct) / 100.0
    uh, ut = rel(height_m, px_pairs[long_i]), rel(thick_m, px_pairs[short_i])
    return {"ok": True, "height_cm": height_m * 100, "thickness_cm": thick_m * 100,
            "height_rel_uncertainty": uh, "thickness_rel_uncertainty": ut, "edges_cm": [round(e * 100, 2) for e in edges_m],
            "scale_note": scale_note, "hops": hops, "occlusion": occlusion or "unknown"}


def _best_calibration_for(frame_id: str, cands: list[tuple[dict, np.ndarray, int]]) -> list[tuple[dict, np.ndarray, int]]:
    return sorted(cands, key=lambda c: (c[2], c[0]["calibration_id"]))


def shelf_runs_for(scale_data: dict[str, Any], present_ids: set[str] | None) -> tuple[list[dict[str, Any]], float | None]:
    """Deterministic shelf run per calibrated shelf-face plane, optionally restricted to instances still present
    (user exclusions). Reported only when every present in-region book is measured and none has an uncertain association."""
    opts = {**DEFAULTS, **(scale_data.get("options") or {})}
    runs = []
    for reg in scale_data.get("shelf_regions", []):
        keep = (lambda i: present_ids is None or i in present_ids)
        in_region = [i for i in reg["in_region_ids"] if keep(i)]
        measured = {i: t for i, t in reg["measured_thickness_cm"].items() if keep(i)}
        unc = [i for i in reg["uncertain_ids"] if keep(i)]
        total_cm = sum(measured.values())
        complete = bool(in_region) and len(measured) == len(in_region) and (not unc or not opts["run_requires_unique_associations"])
        runs.append({"calibration_id": reg["calibration_id"], "plane_id": reg["plane_id"], "books_in_region": len(in_region),
                     "books_measured": len(measured), "books_with_uncertain_association": len(unc),
                     "partial_sum_thickness_m": round(total_cm / 100, 4), "shelf_run_m": round(total_cm / 100, 4) if complete else None,
                     "status": "complete" if complete else "withheld",
                     "withheld_reason": None if complete else (
                         "no detected books attributed to this calibrated plane" if not in_region else
                         "not every book in the calibrated region is measured" if len(measured) != len(in_region) else
                         "uncertain physical associations: provisional groups cannot support a verified shelf length"),
                     "note": "sum of per-instance spine thickness; excludes gaps between books and any uncounted books"})
    done = [r for r in runs if r["status"] == "complete"]
    total = round(sum(r["shelf_run_m"] for r in done), 4) if runs and len(done) == len(runs) else None
    return runs, total


def measure_data(inventory: dict[str, Any], manifest: dict[str, Any], observations: dict[str, dict[str, Any]],
                 calibrations: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pure function: inventory + manifest + calibrations -> scale ``data`` dict (no I/O)."""
    opts = {**DEFAULTS, **{k: v for k, v in (config or {}).items() if k in DEFAULTS}}
    frames = {f["frame_id"]: f for f in manifest["data"]["frames"]}
    inv = inventory.get("data") if "data" in inventory else inventory
    graph = build_transfer_graph(inv.get("registrations", []), opts)
    cal_report, usable = [], []
    for cal in calibrations:
        problems = validate_calibration(cal, frames, opts)
        entry = {"calibration_id": cal.get("calibration_id"), "kind": cal.get("kind"), "frame_id": cal.get("frame_id"),
                 "plane": cal.get("plane"), "metric": cal.get("metric"), "image_points_px": cal.get("image_points_px"),
                 "provenance": cal.get("provenance"), "assumptions": cal.get("assumptions") or {},
                 "status": "invalid" if problems else "valid", "problems": problems, "validation_state": VALIDATION_STATE}
        if not problems:
            if cal["kind"] == "plane_quad":
                entry["oblique_index"] = round(oblique_index(np.array(cal["image_points_px"], float)), 3)
            paths = transfer_paths(graph, cal["frame_id"], opts["max_hops"])
            entry["transfer_frames"] = {f: h for f, (_, h) in paths.items()}
            usable.append((cal, paths))
        cal_report.append(entry)

    books = inv.get("books", [])
    measurements = []
    for inst in books:
        rec: dict[str, Any] = {"instance_id": inst["instance_id"], "observation_ids": inst["observation_ids"],
                               "spine_height_cm": None, "spine_thickness_cm": None, "status": "abstained",
                               "abstention_reasons": [], "per_observation": [], "validation_state": VALIDATION_STATE,
                               "uncertain_association": bool(inst.get("uncertain_association"))}
        if not usable:
            rec["abstention_reasons"].append("no_valid_calibration")
            measurements.append(rec)
            continue
        hs, ts, hu, tu, used = [], [], [], [], []
        obs_reasons: list[str] = []
        for oid in inst["observation_ids"]:
            o = observations.get(oid)
            if not o or not o.get("polygon_xy") or len(o["polygon_xy"]) != 4:
                obs_reasons.append("no_oriented_polygon")
                continue
            fr = frames.get(o["frame_id"])
            poly = np.array(o["polygon_xy"], float)
            bb = o["bbox_xyxy"]
            m = opts["frame_edge_margin_px"]
            if fr and (bb[0] <= m or bb[1] <= m or bb[2] >= fr["width_px"] - m or bb[3] >= fr["height_px"] - m):
                obs_reasons.append("truncated_at_frame_edge")
                continue
            if o.get("occlusion") in ("partial", "heavy"):
                obs_reasons.append(f"occluded_{o['occlusion']}")
                continue
            if o.get("orientation") == "stacked":
                obs_reasons.append("stacked_orientation_axes_ambiguous")
                continue
            cands = [(cal, paths[o["frame_id"]][0], paths[o["frame_id"]][1]) for cal, paths in usable if o["frame_id"] in paths]
            if not cands:
                obs_reasons.append("no_calibration_transferable_to_this_frame")
                continue
            got = None
            tried: list[str] = []
            for cal, Hcal, hops in _best_calibration_for(o["frame_id"], cands):
                # Hcal maps calibration-frame image -> target-frame image
                r = measure_polygon(poly, cal, Hcal, hops, opts, o.get("occlusion"))
                if r["ok"]:
                    got = (cal, r)
                    break
                tried += r["reasons"]
            if not got:
                obs_reasons += tried or ["no_applicable_calibration"]
                continue
            cal, r = got
            hs.append(r["height_cm"]); ts.append(r["thickness_cm"]); hu.append(r["height_rel_uncertainty"]); tu.append(r["thickness_rel_uncertainty"])
            used.append({"observation_id": oid, "calibration_id": cal["calibration_id"], "plane_id": cal["plane"]["id"],
                         "frame_id": o["frame_id"], "hops": r["hops"], "edges_cm": r["edges_cm"], "scale_note": r["scale_note"],
                         "height_cm": round(r["height_cm"], 2), "thickness_cm": round(r["thickness_cm"], 2),
                         "orientation": o.get("orientation"), "occlusion": r["occlusion"],
                         "polygon_xy": o["polygon_xy"], "timestamp_s": fr["timestamp_s"] if fr else None})
        rec["per_observation"] = used
        if not used:
            rec["abstention_reasons"] = sorted(set(obs_reasons)) or ["no_measurable_observation"]
            measurements.append(rec)
            continue
        spread = lambda v: (max(v) - min(v)) / max(float(np.median(v)), 1e-9)
        if len(used) > 1 and (spread(hs) > opts["max_observation_spread"] or spread(ts) > opts["max_observation_spread"]):
            rec["abstention_reasons"] = ["observations_disagree_beyond_tolerance"]
            rec["observation_spread"] = {"height": round(spread(hs), 3), "thickness": round(spread(ts), 3)}
            measurements.append(rec)
            continue
        h, t = float(np.median(hs)), float(np.median(ts))
        lo_h, hi_h = min(v * (1 - u) for v, u in zip(hs, hu)), max(v * (1 + u) for v, u in zip(hs, hu))
        lo_t, hi_t = min(v * (1 - u) for v, u in zip(ts, tu)), max(v * (1 + u) for v, u in zip(ts, tu))
        rec.update(status="measured", spine_height_cm=round(h, 2), spine_thickness_cm=round(t, 2),
                   spine_height_interval_cm=[round(lo_h, 2), round(hi_h, 2)], spine_thickness_interval_cm=[round(lo_t, 2), round(hi_t, 2)],
                   calibration_ids=sorted({u["calibration_id"] for u in used}), plane_ids=sorted({u["plane_id"] for u in used}),
                   method="planar_rectification_from_user_calibration", observations_used=len(used),
                   provenance_label="derived from user-supplied calibration dimensions (not camera-only)",
                   interval_note="stated sum of declared uncertainty terms, not a statistical guarantee")
        if obs_reasons:
            rec["skipped_observations"] = sorted(set(obs_reasons))
        measurements.append(rec)

    # ---- shelf-run regions: which instances lie on each calibrated shelf-face plane
    regions = []
    by_inst = {m["instance_id"]: m for m in measurements}
    for cal, paths in usable:
        if (cal.get("plane") or {}).get("role") != "shelf_face" or cal["kind"] != "plane_quad":
            continue
        in_region, measured, unc = [], {}, []
        for inst in books:
            m = by_inst[inst["instance_id"]]
            # Region membership is decided conservatively: measured on this plane => in; abstained ONLY because it lies outside the
            # plane region => known outside; abstained for any other reason (not transferable, truncated, occluded, no polygon...)
            # => membership unknown, so it counts as an unmeasured book and withholds the shelf run.
            on_plane = any(u["calibration_id"] == cal["calibration_id"] for u in m["per_observation"]) or (
                not m["per_observation"] and bool(m["abstention_reasons"]) and set(m["abstention_reasons"]) != {"outside_calibrated_plane_region"})
            if on_plane:
                in_region.append(inst["instance_id"])
                if m["status"] == "measured":
                    measured[inst["instance_id"]] = m["spine_thickness_cm"]
                if m["uncertain_association"]:
                    unc.append(inst["instance_id"])
        regions.append({"calibration_id": cal["calibration_id"], "plane_id": cal["plane"]["id"], "in_region_ids": in_region,
                        "measured_thickness_cm": measured, "uncertain_ids": unc})
    shelf_runs, total_run = shelf_runs_for({"shelf_regions": regions, "options": opts}, None)
    reasons_count: dict[str, int] = defaultdict(int)
    for m in measurements:
        for r in m["abstention_reasons"]:
            reasons_count[r] += 1
    return {"method": "planar_rectification_from_user_calibration", "scale_version": SCALE_VERSION, "units": {"calibration": "m", "output": "cm", "shelf_run": "m"},
            "options": opts, "calibrations": cal_report, "measurements": measurements,
            "shelf_regions": regions, "shelf_runs": shelf_runs, "shelf_run_m": total_run,
            "summary": {"books": len(books), "measured": sum(m["status"] == "measured" for m in measurements),
                        "abstained": sum(m["status"] == "abstained" for m in measurements),
                        "valid_calibrations": len(usable), "invalid_calibrations": len(cal_report) - len(usable),
                        "abstention_reasons": dict(sorted(reasons_count.items(), key=lambda kv: -kv[1]))},
            "limits": LIMITS, "validation": {"physical_truth_compared": False, "validation_state": VALIDATION_STATE,
                                            "note": "no tape-measured comparison has been run; software behaviour only"}}


# ------------------------------------------------------------------------------------------------ stage I/O
def load_calibrations(path: str | Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).is_file():
        return []
    doc = C.read_json(path)
    if doc.get("purpose") != "calibration":
        raise C.ContractError("calibration file must declare purpose='calibration' (evaluation truth must never be used here)")
    return list(doc.get("calibrations", []))


def measure(inventory_path: str | Path, manifest_path: str | Path, run_dir: str | Path, config: dict[str, Any] | None = None) -> Path:
    """Stage entry point. Config: ``calibration_path`` | ``calibrations`` (list), ``scale`` option overrides."""
    t0 = time.perf_counter()
    config = config or {}
    run_dir = Path(run_dir)
    manifest = C.read_json(manifest_path)
    inv = C.read_json(inventory_path)
    cands_path = run_dir / C.ARTIFACTS["detect"]
    observations: dict[str, dict[str, Any]] = {}
    if cands_path.is_file():
        d = C.read_json(cands_path).get("data") or {}
        observations = {o["observation_id"]: o for o in (d.get("observations") if d.get("observations") is not None else d.get("detections")) or []}
    cals = config.get("calibrations")
    if cals is None:
        cals = load_calibrations(config.get("calibration_path") or (run_dir / "calibration.json"))
    data = measure_data(inv, manifest, observations, cals, config.get("scale"))
    data["inventory_sha256"] = __import__("providers.cache", fromlist=["x"]).sha256_file(inventory_path)
    issues = []
    if not cals:
        issues.append({"code": "no_calibration", "severity": "info", "detail": "no calibration supplied: all spine dimensions and shelf run stay unknown"})
    for c in data["calibrations"]:
        for p in c["problems"]:
            issues.append({"code": "invalid_calibration", "severity": "warning", "detail": p})
    status = "ok" if data["summary"]["valid_calibrations"] and data["summary"]["abstained"] == 0 else "partial"
    env = C.envelope(run_dir.name if not config.get("run_id") else config["run_id"], "scale", data, status=status, issues=issues,
                     metrics={"elapsed_s": round(time.perf_counter() - t0, 4), "request_count": 0, "cost": 0,
                              "cost_provenance": "local geometry; no API calls", "measured": data["summary"]["measured"],
                              "abstained": data["summary"]["abstained"], "valid_calibrations": data["summary"]["valid_calibrations"]})
    return C.write_json(run_dir / C.ARTIFACTS["scale"], env)


# ------------------------------------------------------------------ bounded feasibility: vanishing-point focal check
def _line_segments(gray: np.ndarray, min_len: float = 80.0) -> np.ndarray:
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    l = lines.reshape(-1, 4)
    ln = np.hypot(l[:, 2] - l[:, 0], l[:, 3] - l[:, 1])
    return l[ln >= min_len]


def _vp_ransac(lines: np.ndarray, iters: int = 400, seed: int = 0) -> tuple[np.ndarray | None, np.ndarray]:
    """Vanishing point (homogeneous) by RANSAC over line pairs; returns (vp, inlier mask)."""
    rng = np.random.default_rng(seed)
    L = np.cross(np.c_[lines[:, 0], lines[:, 1], np.ones(len(lines))], np.c_[lines[:, 2], lines[:, 3], np.ones(len(lines))])
    L = L / np.maximum(np.linalg.norm(L[:, :2], axis=1, keepdims=True), 1e-9)
    best, best_mask = None, np.zeros(len(lines), bool)
    mids = np.c_[(lines[:, 0] + lines[:, 2]) / 2, (lines[:, 1] + lines[:, 3]) / 2]
    for _ in range(iters):
        if len(lines) < 2:
            break
        i, j = rng.choice(len(lines), 2, replace=False)
        v = np.cross(L[i], L[j])
        if abs(v[2]) < 1e-9 and np.linalg.norm(v[:2]) < 1e-9:
            continue
        # angular residual: angle between each line and the direction from its midpoint to the vp
        vp = v[:2] / v[2] if abs(v[2]) > 1e-9 else None
        dirs = (np.c_[lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]])
        if vp is None:
            to = np.tile(v[:2], (len(lines), 1))
        else:
            to = vp[None, :] - mids
        cos = np.abs((dirs * to).sum(1)) / np.maximum(np.linalg.norm(dirs, axis=1) * np.linalg.norm(to, axis=1), 1e-9)
        mask = np.degrees(np.arccos(np.clip(cos, 0, 1))) < 2.0
        if mask.sum() > best_mask.sum():
            best, best_mask = v, mask
    return best, best_mask


def focal_from_orthogonal_vps(v1: np.ndarray, v2: np.ndarray, cx: float, cy: float) -> float | None:
    """f^2 = -(v1-c).(v2-c) for finite orthogonal vanishing points (square pixels, zero skew, principal point at c)."""
    if abs(v1[2]) < 1e-9 or abs(v2[2]) < 1e-9:
        return None
    a, b = v1[:2] / v1[2] - [cx, cy], v2[:2] / v2[2] - [cx, cy]
    f2 = -float(np.dot(a, b))
    return math.sqrt(f2) if f2 > 0 else None


def feasibility(video_or_run: str | Path, out_dir: str | Path, sample_frames: int = 10) -> Path:
    """Bounded feasibility attempt on REAL frames: do vanishing points give a stable focal length? (not a calibration)"""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = Path(video_or_run)
    man = C.read_json(run / C.ARTIFACTS["frames"])
    sel = [f for f in man["data"]["frames"] if f["selected"]][:sample_frames]
    per = []
    for f in sel:
        img = cv2.imread(str(run / f["image_path"]), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        lines = _line_segments(img)
        ang = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])) % 180 if len(lines) else np.array([])
        vert = lines[(np.abs(ang - 90) < 25)] if len(lines) else lines
        horiz = lines[(np.minimum(ang, 180 - ang) < 25)] if len(lines) else lines
        v_v, m_v = _vp_ransac(vert) if len(vert) > 5 else (None, np.zeros(0, bool))
        v_h, m_h = _vp_ransac(horiz) if len(horiz) > 5 else (None, np.zeros(0, bool))
        f_est = focal_from_orthogonal_vps(v_v, v_h, w / 2, h / 2) if v_v is not None and v_h is not None else None
        per.append({"frame_id": f["frame_id"], "lines": int(len(lines)), "vertical_inliers": int(m_v.sum()), "horizontal_inliers": int(m_h.sum()),
                    "vertical_vp_finite": bool(v_v is not None and abs(v_v[2]) > 1e-9), "horizontal_vp_finite": bool(v_h is not None and abs(v_h[2]) > 1e-9),
                    "focal_px_estimate": None if f_est is None else round(f_est, 1)})
    fs = [p["focal_px_estimate"] for p in per if p["focal_px_estimate"]]
    result = {"purpose": "bounded feasibility only; not a calibration and not a measurement", "frames": per,
              "frames_with_estimate": len(fs), "focal_px": {"median": round(float(np.median(fs)), 1) if fs else None,
                                                             "min": min(fs) if fs else None, "max": max(fs) if fs else None,
                                                             "relative_spread": round((max(fs) - min(fs)) / float(np.median(fs)), 3) if len(fs) > 1 else None},
              "undecoded_metadata_lead_px": 1712.08,
              "metadata_note": "Apple per-frame metadata float 1712.08 is an UNDECODED lead; shown only for comparison, never used as intrinsics.",
              "scale_note": "Even a stable focal length gives angles, not metric scale: no known length on any plane is visible in this recording.",
              "elapsed_s": round(time.perf_counter() - t0, 3), "request_count": 0}
    return C.write_json(out_dir / "feasibility.json", result)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run dir with manifest.json, candidates.json, inventory.json")
    ap.add_argument("--calibration", help="calibration JSON (purpose=calibration)")
    ap.add_argument("--feasibility", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.feasibility:
        print(feasibility(a.run, a.out or a.run))
        return 0
    run = Path(a.run)
    print(measure(run / C.ARTIFACTS["dedup"], run / C.ARTIFACTS["frames"], run, {"calibration_path": a.calibration}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
