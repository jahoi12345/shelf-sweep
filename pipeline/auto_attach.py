"""Attach an automatic measurement (pipeline.auto_scale: scale, spine sizes, shelf run, room) to a finished run's claim_packet.json.

    python -m pipeline.auto_attach --run runs/<run> --measure runs/<run>_measure_vN/auto_measure.json

The packet is built when the sweep ends, before the 3D map exists (the post-sweep queue builds it afterwards), so the measurement is attached afterwards,
IN PLACE: the packet's own verification (pipeline.packet.verify_packet) is re-run on the patched packet and nothing is written when it fails.

Every attach patches the CURRENT packet and rewrites only the measurement-owned fields (see ``attach``), so the titles, prices, panorama count and
instance lines attached since the End-sweep packet survive a newer measurement, in any order, and attaching twice gives the same packet. The End-sweep packet is
kept once as ``claim_packet.pre_auto_measure.json`` as the pristine copy only (read for the room fields a measurement without a complete room falls back to);
it is no longer the base every attach restarts from (that dropped every layer attached after it). The legacy calibration-based
scale/room blocks stay the source when a person volunteered calibrations; the automatic result only fills what they left unmeasured.

Every attached number carries its method, scale source and stated uncertainty; unsupported values stay null with the reason. A spine height whose thickness is
withheld (spine too narrow in pixels, cover view, implausible library median) is reported with ``measurement.thickness_reason`` and a null thickness.
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from pipeline import contracts as C
from pipeline import packet as P
from pipeline import room as RM
from pipeline.auto_scale import VALIDATION

CAL_ID, PLANE_ID = "auto-scale-1", "shelf-3d-1"
BACKUP = "claim_packet.pre_auto_measure.json"
DROP_CODES = {"no_metric_scale", "spine_dimensions_partial", "shelf_run_withheld", "spine_scale_from_prior"}     # review entries a measurement owns (room_not_measured apart)
NOTICE = " Spine sizes, shelf run and room geometry come from the automatic measurement (see measurement_evidence.auto); nobody was asked to measure."


def _label(scale: dict[str, Any]) -> str:
    if scale.get("source") == "catalogue_physical_dimensions":
        return f"automatic: catalogue physical dimensions of {scale.get('n_inliers')} identified books (Open Library equivalent editions) x 3D shelf-plane geometry; nobody was asked to measure"
    return "automatic: standard-format PRIOR scale (NOT a measurement; no identified book had catalogue dimensions) x 3D shelf-plane geometry; nobody was asked to measure"


def legacy_scale(auto: dict[str, Any], books: list[dict[str, Any]]) -> dict[str, Any]:
    """The packet's scale block (shape of pipeline.scale.measure_data) filled from an auto_measure result."""
    sc = auto["scale"]
    valid = bool(sc.get("cm_per_unit"))
    cal = {"calibration_id": CAL_ID, "kind": "auto_" + str(sc.get("source")), "frame_id": None, "plane": {"id": PLANE_ID, "role": "shelf_face"},
           "metric": {"cm_per_unit": None if not valid else round(sc["cm_per_unit"], 4), "relative_uncertainty_pct": sc.get("relative_uncertainty_pct"), "units": "cm per reconstruction unit"},
           "image_points_px": None, "assumptions": {}, "validation_state": VALIDATION,
           "provenance": {"source": "automatic", "actor": "pipeline.auto_scale", "supplied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": _label(sc) if valid else sc.get("reason"),
                          "independent_of_evaluation_truth": True, "user_supplied": False},
           "status": "valid" if valid else "invalid", "problems": [] if valid else [str(sc.get("reason"))]}
    rows = {s["group_id"]: s for s in auto["spines"]}
    measured, reasons = 0, {}
    meas = []
    for b in books:
        # a merged instance line (pipeline/instances.py) carries its member track groups: the measurement of any measured member (the best-seen first) applies to the line
        cands = [rows[g] for g in [b["id"]] + [m for m in (b.get("member_group_ids") or []) if m != b["id"]] if g in rows]
        r = next((x for x in sorted(cands, key=lambda x: -(x.get("sightings_used") or 0)) if x.get("status") == "measured"), cands[0] if cands else None)
        ok = r is not None and r.get("status") == "measured" and (r.get("spine_thickness_cm") is None or r["spine_height_cm"] >= r["spine_thickness_cm"] > 0)
        if ok:
            hu = r["height_relative_uncertainty_pct"] / 100.0
            h, t = r["spine_height_cm"], r.get("spine_thickness_cm")
            tu = (r["thickness_relative_uncertainty_pct"] / 100.0) if t is not None else None
            meas.append({"id": b["id"], "status": "measured", "spine_height_cm": h, "spine_thickness_cm": t, "method": "auto_3d_shelf_plane_x_" + str(sc.get("source")),
                         "calibration_ids": [CAL_ID], "plane_ids": [PLANE_ID], "spine_height_interval_cm": [round(h * (1 - hu), 2), round(h * (1 + hu), 2)],
                         "spine_thickness_interval_cm": None if t is None else [round(t * (1 - tu), 2), round(t * (1 + tu), 2)],
                         **({"thickness_reason": r.get("thickness_reason")} if t is None else {}), "observations_used": r["sightings_used"],
                         "validation_state": VALIDATION, "provenance_label": _label(sc), "interval_note": "stated sum of declared uncertainty terms, not a statistical guarantee",
                         "per_observation": r["per_observation"], "scale_source": sc.get("source")})
            measured += 1
        else:
            why = (r or {}).get("reason") or "not_in_reconstruction"
            reasons[str(why).split(":")[0]] = reasons.get(str(why).split(":")[0], 0) + 1
            meas.append({"id": b["id"], "status": "abstained", "reasons": [str(why)], "validation_state": VALIDATION})
    shelf = auto.get("shelf") or {}
    n = len(books)
    run_ok = valid and shelf.get("shelf_run_m") is not None and measured >= 0.5 * max(n, 1)
    runs = [{"calibration_id": CAL_ID, "plane_id": PLANE_ID, "books_in_region": n, "books_measured": measured, "books_with_uncertain_association": 0,
             "partial_sum_thickness_m": shelf.get("sum_measured_thickness_m"), "shelf_run_m": shelf.get("shelf_run_m") if run_ok else None,
             "status": "complete" if run_ok else "withheld",
             "withheld_reason": None if run_ok else ("fewer than half of the books measured" if valid else "no metric scale"),
             "note": "sum over shelf rows of the horizontal extent of the measured spines (gaps and unmeasured books inside the extent included); " + _label(sc)}] if valid else []
    total = round(sum(r["shelf_run_m"] for r in runs if r["shelf_run_m"] is not None), 4) if runs and all(r["status"] == "complete" for r in runs) else None
    return {"method": "auto_3d_shelf_plane_ray_casting", "units": {"scale": "cm per reconstruction unit", "output": "cm", "shelf_run": "m"}, "options": {"auto": True},
            "calibrations": [cal], "measurements": meas, "shelf_regions": [], "shelf_runs": runs, "shelf_run_m": total,
            "summary": {"books": n, "measured": measured, "abstained": n - measured, "valid_calibrations": 1 if valid else 0, "invalid_calibrations": 0 if valid else 1,
                        "abstention_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1]))},
            "limits": auto.get("limits") or [], "validation": {"physical_truth_compared": False, "validation_state": VALIDATION,
                                                              "note": "no tape-measured comparison has been run; software behaviour only"}}


def legacy_room(auto_room: dict[str, Any]) -> dict[str, Any] | None:
    """A packet room artifact (shape of pipeline.room.estimate_data) for a COMPLETE box room, else None (partial values stay in measurement_evidence.auto)."""
    if auto_room.get("status") != "estimated":
        return None
    d, a = auto_room["dimensions"], auto_room["areas"]
    W, D, H = (round(d[k]["value"], 2) for k in ("width_m", "depth_m", "height_m"))
    length, width = max(W, D), min(W, D)
    fl, wl = round(length * width, 2), round(a["wall_area_m2"]["value"], 2)
    walls = [{"wall_id": k, "frame_id": (auto_room["planes"].get(k) or {}).get("frame_id")} for k in ("shelf_wall", "opposite_wall", "left_wall", "right_wall")]
    sh = (a.get("shelved_wall_area_m2") or {}).get("value")
    unc = {"floor_area_pct": a["floor_area_m2"]["relative_uncertainty_pct"], "wall_area_pct": a["wall_area_m2"]["relative_uncertainty_pct"], "height_pct": d["height_m"]["relative_uncertainty_pct"],
           "note": "stated sum of declared terms (scale + plane localisation), not a statistical guarantee"}
    return {"status": "estimated", "method": "automatic plane fit of the 3D reconstruction x " + str((auto_room.get("scale") or {}).get("source")), "abstention_reasons": [],
            "dimensions": {"length_m": length, "width_m": width, "height_m": H}, "areas": {"floor": {"m2": fl, "ft2": round(fl * RM.FT2_PER_M2, 2)}, "wall": {"m2": wl, "ft2": round(wl * RM.FT2_PER_M2, 2)}},
            "shape": {"kind": "rectangular", "assumption": auto_room["shape"]["assumption"], "walls": walls},
            "height_provenance": {"kind": "auto_sfm_planes", "label": "measured from the reconstructed floor and ceiling planes x automatic scale (" + str((auto_room.get("scale") or {}).get("source")) + ")"},
            "uncertainty": unc, "validation": {"validation_state": VALIDATION}, "coverage": {"missing": []},
            "shelved_wall": {"area": {"m2": sh, "ft2": None if sh is None else round(sh * RM.FT2_PER_M2, 2)}} if sh is not None else None}


def _track_lines(p: dict[str, Any]) -> list[dict[str, Any]]:
    """Every line object that carries measurement fields: the book and unconfirmed lines and the members stored inside merged instance lines (the instance layer
    rebuilds a merged line from its members, so a member left with an older measurement would bring it back on the next instance apply)."""
    out = []
    for line in list(p.get("books") or []) + list(p.get("unconfirmed_books") or []):
        out.append(line)
        out.extend(line.get("merged_members") or [])
    return out


def _set_line(b: dict[str, Any], m: dict[str, Any]) -> None:
    if m["status"] == "measured":
        b["spine_height_cm"], b["spine_thickness_cm"] = m["spine_height_cm"], m["spine_thickness_cm"]
        b["measurement"] = {k: m[k] for k in m if k not in ("id", "spine_height_cm", "spine_thickness_cm")}
    else:
        b["spine_height_cm"] = b["spine_thickness_cm"] = None                  # an earlier measurement of this line does not survive a newer one that abstains
        b["measurement"] = {"status": "abstained", "reasons": m["reasons"], "validation_state": VALIDATION}


def attach(run_dir: str | Path, auto_path: str | Path, *, render: bool = True) -> dict[str, Any]:
    """Patch the CURRENT claim_packet.json with this measurement (composable with the identity / price / panorama / instance layers attached since).

    Only the measurement-owned fields are rewritten: per line ``spine_height_cm``/``spine_thickness_cm``/``measurement`` (book, unconfirmed and merged-member lines),
    ``totals.shelf_run_m``/``shelf_run_detail``/``spine_dimensions_measured``/``spine_dimensions_abstained``, ``measurement_evidence.scale/room/auto``, ``room``, the
    measurement review entries (MEASURE_CODES), ``completeness`` and the measurement sentence of ``provisional_notice``. Everything else (titles, prices, instance lines,
    count totals) is carried over untouched. The pristine End-sweep packet (``claim_packet.pre_auto_measure.json``, written once) is only read for the parts a
    measurement replaces wholesale: the room block / room evidence / room review entries when this measurement has no complete room, the notice, and the
    volunteered-calibration check. When the run has an instance layer (instances.json) it is re-applied, as pipeline.identity_attach / price_attach do, so the merged lines
    take their measurement from their members exactly as a later instance step would."""
    run_dir = Path(run_dir)
    pk = run_dir / C.ARTIFACTS["packet"]
    backup = run_dir / BACKUP
    if not pk.is_file():
        return {"attached": False, "reason": "no claim_packet.json"}
    if not backup.is_file():
        shutil.copy2(pk, backup)
    pristine = C.read_json(backup)["data"]
    env = C.read_json(pk)
    manifest = C.read_json(run_dir / C.ARTIFACTS["frames"])
    auto_env = C.read_json(auto_path)
    auto = auto_env["data"]
    p = env["data"]
    if p.get("fixture"):
        return {"attached": False, "reason": "fixture packet"}
    legacy = (pristine.get("measurement_evidence") or {}).get("scale") or {}
    if (legacy.get("summary") or {}).get("valid_calibrations"):
        return {"attached": False, "reason": "a person volunteered a valid calibration: the legacy scale stays the source of the spine sizes"}
    scale = legacy_scale(auto, p["books"])
    in_books = {id(b) for b in p["books"]}
    others = [x for x in _track_lines(p) if id(x) not in in_books]                   # unconfirmed lines and merged members (a member may share its line's id)
    for b, m in zip(p["books"], scale["measurements"]):
        _set_line(b, m)
    for x, m in zip(others, legacy_scale(auto, others)["measurements"]):
        _set_line(x, m)
    t = p["totals"]
    t["shelf_run_m"], t["shelf_run_detail"] = scale["shelf_run_m"], scale["shelf_runs"]
    me = p.setdefault("measurement_evidence", {})
    me["scale"] = {k: scale[k] for k in ("method", "units", "calibrations", "summary", "shelf_runs", "limits", "validation", "options")}
    arr = auto.get("room") or {}
    room_art = legacy_room(arr)
    reasons = list(arr.get("abstention_reasons") or [])
    pristine_room = pristine.get("room") or {}
    if room_art is not None:
        blk = P._room_block(room_art, scale)
        blk["shelved_wall_provenance"] = "bounding rectangle of the automatically measured books on the shelf face (not a shelving-unit measurement)" if blk.get("shelved_wall_area_m2") is not None else None
        p["room"] = blk
        me["room"] = room_art
    else:
        p["room"] = {**copy.deepcopy(pristine_room), "status": "abstained", "abstention_reasons": reasons or ["room_geometry_not_established"],
                     "coverage_missing": list(arr.get("capture_guidance") or []), "scale_method": None}
        pristine_me = (pristine.get("measurement_evidence") or {})
        if "room" in pristine_me:
            me["room"] = copy.deepcopy(pristine_me["room"])
        else:
            me.pop("room", None)
    me["auto"] = {"artifact": str(Path(auto_path).name), "source_measure_dir": Path(auto_path).parent.name, "status": auto.get("status"), "scale": {k: v for k, v in auto["scale"].items() if k != "sources"},
                  "scale_sources": auto["scale"].get("sources"), "summary": auto.get("summary"), "shelf": auto.get("shelf"), "room": arr, "validation": VALIDATION}
    # review queue: the current entries minus the measurement-owned ones (kept in place: room_not_measured while the room stays unmeasured); then this measurement's
    cur = P.flat_review(p.get("review_queue"))
    have_room_entry = any(x.get("code") == "room_not_measured" for x in cur)
    q = []
    for x in cur:
        if x.get("code") in DROP_CODES:
            continue
        if x.get("code") == "room_not_measured" and room_art is not None:
            continue
        q.append(x)
    if room_art is None and not have_room_entry:                                     # an earlier measurement estimated the room; this one does not: the End-sweep entry returns
        q += [copy.deepcopy(x) for x in P.flat_review(pristine.get("review_queue")) if x.get("code") == "room_not_measured"]
    sm = scale["summary"]
    if scale["summary"]["valid_calibrations"] == 0:
        q.append({"ref_id": "dimensions", "code": "no_metric_scale", "severity": "blocking", "reason": "Spine sizes unknown: " + str(auto["scale"].get("reason"))})
    else:
        if auto["scale"].get("source") != "catalogue_physical_dimensions":
            q.append({"ref_id": "dimensions", "code": "spine_scale_from_prior", "severity": "warning",
                      "reason": "Spine sizes use a standard-format PRIOR scale (not a measurement; stated uncertainty >= 15%) because no identified book had catalogue dimensions."})
        q.append({"ref_id": "dimensions", "code": "spine_dimensions_partial", "severity": "warning" if sm["measured"] else "blocking",
                  "reason": f"{sm['measured']} of {sm['books']} books measured; {sm['abstained']} abstained. Reasons: " + ", ".join(f"{k} x{v}" for k, v in list(sm["abstention_reasons"].items())[:6])})
        for r in scale["shelf_runs"]:
            if r["status"] != "complete":
                q.append({"ref_id": "dimensions", "code": "shelf_run_withheld", "severity": "warning", "reason": f"Shelf run withheld: {r['withheld_reason']}."})
    if room_art is None:
        for x in q:
            if x.get("code") == "room_not_measured":
                x["reason"] = "Room dimensions/areas not established from the sweep: " + ("; ".join(reasons[:5]) or "geometry incomplete") + (
                    ". Film: " + " ".join(arr.get("capture_guidance") or []) if arr.get("capture_guidance") else "") + "."
    p["review_queue"] = q
    p["provisional_notice"] = (pristine.get("provisional_notice") or "") + NOTICE
    from pipeline import instances as IN
    art = IN.load(run_dir)
    if art and any(b.get("member_group_ids") for b in p["books"]):                  # the lines are physical instances: rebuild the merged lines from the patched members
        IN.apply(p, art, (p.get("corrections") or {}).get("ledger"))
        for line in p["books"] + list(p.get("unconfirmed_books") or []):           # as pipeline.identity_attach / price_attach: carry the identity member's catalogue fields
            if line.get("status") == "identified" and line.get("merged_members"):
                src = next((x for x in line["merged_members"] if x.get("status") == "identified" and x.get("identity_source_url") == line.get("identity_source_url")), None)
                if src:
                    for k in ("catalogue", "catalogue_title", "identity_work", "catalogue_dimensions"):
                        if k in src:
                            line[k] = src.get(k)
    t["spine_dimensions_measured"] = sum(b.get("spine_height_cm") is not None for b in p["books"])
    t["spine_dimensions_abstained"] = sum((b.get("measurement") or {}).get("status") == "abstained" for b in p["books"])
    p["completeness"] = P.completeness(p["books"], p["items"], p["room"], p["totals"])
    P.finalize_contract(p, run_dir)
    problems = P.verify_packet(env, manifest, run_dir)
    if problems:
        return {"attached": False, "reason": "packet verification failed after the automatic measurement was attached", "problems": problems[:10]}
    C.write_json(pk, env)
    if render:
        from pipeline import report as R
        R.render(pk, run_dir)
        if (run_dir / "report.pdf").is_file():
            R.render_print_pdf(pk, run_dir)
    return {"attached": True, "spines_measured": sm["measured"], "books": sm["books"], "scale_source": auto["scale"].get("source"), "room_status": (p["room"] or {}).get("status"),
            "shelf_run_m": scale["shelf_run_m"], "base": "current packet (layers attached since the End-sweep packet are kept)"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--measure", required=True, help="path to auto_measure.json (from pipeline.auto_scale)")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    print(attach(a.run, a.measure, render=not a.no_render))
    return 0


if __name__ == "__main__":
    sys.exit(main())
