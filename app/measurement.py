"""Live measurement state: calibrations, room evidence, coverage guidance.

Holds attributable inputs a person MAY volunteer through the REST API (calibration dimensions, corner marks, a stated
ceiling height) and derives everything else by calling ``pipeline.scale`` / ``pipeline.room`` (the same code the recorded
pipeline uses). Calibration inputs are written to ``calibration.json`` / ``room_evidence.json`` (purpose fields keep them
separate from evaluation truth). Nothing here estimates a dimension itself.

HARD RULE: the person is NEVER asked to measure anything, state a dimension or tap/mark corners. Metric scale and room
geometry come from the sweep itself (``pipeline.auto_scale`` / ``pipeline.auto_room``); every prompt this module emits only
says what to FILM.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any

from pipeline import contracts as C
from pipeline import room as RM
from pipeline import scale as SC

UNIT_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "in": 0.0254}


class MeasurementError(ValueError):
    pass


def to_metres(value: float, unit: str) -> float:
    if unit not in UNIT_TO_M:
        raise MeasurementError(f"unit must be one of {sorted(UNIT_TO_M)}")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise MeasurementError("value must be a positive number")
    return float(value) * UNIT_TO_M[unit]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class MeasurementState:
    def __init__(self) -> None:
        self.calibrations: list[dict[str, Any]] = []
        self.walls: list[dict[str, Any]] = []
        self.height_anchor: dict[str, Any] | None = None
        self.layout: dict[str, Any] = {"declared_shape": "rectangular"}
        self.openings: list[dict[str, Any]] = []
        self.pending: dict[str, dict[str, Any]] = {}      # plane_id -> stated dimensions awaiting corner marks
        self.version = 0
        self._seq = {"cal": 0, "wall": 0, "plane": 0}
        self._cache: tuple[Any, Any] | None = None
        self._issued: dict[str, float] = {}
        self._last_guidance = -1e9

    # ---------------------------------------------------------------- inputs
    def _bump(self) -> None:
        self.version += 1
        self._cache = None

    def _prov(self, actor: str, utterance: str | None, note: str | None, source: str = "user_supplied") -> dict[str, Any]:
        if not (utterance or note):
            raise MeasurementError("an utterance or note is required: calibrations must be attributable to a human statement")
        return {"source": source, "actor": actor or "claimant", "supplied_at": _now(), "utterance": utterance, "note": note,
                "independent_of_evaluation_truth": True}

    def next_plane_id(self) -> str:
        self._seq["plane"] += 1
        return f"plane-{self._seq['plane']:02d}"

    def add_calibration(self, *, kind: str, frame_id: str, image_points_px: list[list[float]], metric: dict[str, Any],
                        plane_id: str | None = None, role: str = "shelf_face", actor: str = "claimant", utterance: str | None = None,
                        note: str | None = None, assume_fronto_parallel: bool = False, turn_id: str | None = None) -> dict[str, Any]:
        self._seq["cal"] += 1
        pid = plane_id or self.next_plane_id()
        cal = {"calibration_id": f"cal-{self._seq['cal']:03d}", "kind": kind, "frame_id": frame_id, "image_points_px": image_points_px,
               "metric": {**metric, "tolerance_pct": float(metric.get("tolerance_pct", 3.0))},
               "plane": {"id": pid, "role": role}, "provenance": {**self._prov(actor, utterance, note), "turn_id": turn_id}}
        if assume_fronto_parallel:
            cal["assumptions"] = {"assume_fronto_parallel": True}
        self.calibrations.append(cal)
        self.pending.pop(pid, None)
        self._bump()
        return cal

    def remove_calibration(self, calibration_id: str) -> None:
        n = len(self.calibrations)
        self.calibrations = [c for c in self.calibrations if c["calibration_id"] != calibration_id]
        if len(self.calibrations) == n:
            raise MeasurementError(f"unknown calibration {calibration_id}")
        self._bump()

    def set_pending_dimension(self, plane_id: str | None, what: str, metres: float, tolerance_pct: float, utterance: str | None, actor: str = "claimant") -> dict[str, Any]:
        """Record a stated unit width/height; the person must still mark the corners in the UI."""
        if what not in {"width", "height"}:
            raise MeasurementError("what must be width or height")
        pid = plane_id or next(iter(self.pending), None) or self.next_plane_id()
        slot = self.pending.setdefault(pid, {"plane_id": pid})
        slot[what + "_m"] = metres
        slot["tolerance_pct"] = max(slot.get("tolerance_pct", 0.0), tolerance_pct)
        slot["utterance"] = " | ".join(x for x in (slot.get("utterance"), utterance) if x)
        slot["actor"], slot["stated_at"] = actor, _now()
        self._bump()
        return slot

    def set_height_anchor(self, metres: float, tolerance_pct: float, utterance: str | None, note: str | None = None, actor: str = "claimant",
                          turn_id: str | None = None) -> dict[str, Any]:
        self.height_anchor = {"kind": "user_supplied_ceiling_height", "height_m": metres,
                              "provenance": {**self._prov(actor, utterance, note), "tolerance_pct": tolerance_pct, "turn_id": turn_id}}
        self._bump()
        return self.height_anchor

    def add_wall(self, *, frame_id: str, quad_px: list[list[float]], wall_id: str | None = None, actor: str = "claimant",
                 spans_floor_to_ceiling: bool = False, note: str | None = None) -> dict[str, Any]:
        self._seq["wall"] += 1
        wid = wall_id or chr(ord("A") + (self._seq["wall"] - 1) % 26)
        w = {"wall_id": wid, "frame_id": frame_id, "quad_px": quad_px, "spans_floor_to_ceiling": bool(spans_floor_to_ceiling),
             "provenance": {"source": "user_marked", "actor": actor, "supplied_at": _now(), "note": note}}
        self.walls = [x for x in self.walls if x["wall_id"] != wid] + [w]
        self._bump()
        return w

    def remove_wall(self, wall_id: str) -> None:
        n = len(self.walls)
        self.walls = [w for w in self.walls if w["wall_id"] != wall_id]
        if len(self.walls) == n:
            raise MeasurementError(f"unknown wall {wall_id}")
        self._bump()

    # ---------------------------------------------------------------- documents (versioned, human inputs only)
    def calibration_doc(self) -> dict[str, Any]:
        return {"schema_version": 1, "purpose": "calibration", "calibrations": copy.deepcopy(self.calibrations),
                "pending_dimensions": copy.deepcopy(list(self.pending.values())),
                "note": "human-supplied inputs; never evaluation truth"}

    def room_evidence_doc(self) -> dict[str, Any]:
        return {"schema_version": 1, "purpose": "room_evidence", "walls": copy.deepcopy(self.walls), "height_anchor": copy.deepcopy(self.height_anchor),
                "layout": self.layout, "openings": copy.deepcopy(self.openings)}

    def persist(self, run_dir: Path) -> None:
        C.write_json(run_dir / "calibration.json", self.calibration_doc())
        C.write_json(run_dir / "room_evidence.json", self.room_evidence_doc())

    # ---------------------------------------------------------------- derived (delegates to the shared stages)
    def compute(self, inventory_env: dict[str, Any] | None, manifest_env: dict[str, Any], observations: dict[str, dict[str, Any]],
                scale_config: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        key = (id(inventory_env), self.version, len(manifest_env["data"]["frames"]), len(observations))
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        scale = None
        if inventory_env is not None:
            scale = SC.measure_data(inventory_env, manifest_env, observations, self.calibrations, scale_config)
        evidence = self.room_evidence_doc() if (self.walls or self.height_anchor) else None
        room = RM.estimate_data(evidence, manifest_env, self.calibrations)
        out = (scale, room)
        self._cache = (key, out)
        return out

    # ---------------------------------------------------------------- coverage / guidance
    def status(self, scale: dict[str, Any] | None, room: dict[str, Any]) -> dict[str, Any]:
        valid_shelf = [c for c in (scale or {}).get("calibrations", []) if c["status"] == "valid" and (c.get("plane") or {}).get("role") == "shelf_face"]
        items = [
            {"item": "shelf_unit_dimensions", "present": bool(valid_shelf), "pending": bool(self.pending) and not valid_shelf, "optional": True,
             "how": "Film each shelving unit end to end at an even distance so the spines can be scaled from the books themselves."},
            *[{"item": i["item"], "present": i["present"], "optional": i["optional"], "how": i["how"]} for i in room["coverage"]["items"]
              if i["item"] != "height_anchor"],
        ]
        missing = [i for i in items if not i["present"] and not i.get("optional")]
        return {"items": items, "missing": [i["item"] for i in missing], "next_prompt": missing[0]["how"] if missing else None,
                "calibrations": [{"calibration_id": c["calibration_id"], "kind": c["kind"], "status": c["status"], "problems": c["problems"],
                                  "plane": c["plane"], "frame_id": c["frame_id"], "user_supplied": True} for c in (scale or {}).get("calibrations", [])],
                "pending_dimensions": list(self.pending.values()),
                "walls": [{"wall_id": w["wall_id"], "frame_id": w["frame_id"]} for w in self.walls],
                "height_anchor": None if not self.height_anchor else {"height_m": self.height_anchor["height_m"], "label": "user-supplied"},
                "scale": None if scale is None else {"summary": scale["summary"], "shelf_run_m": scale["shelf_run_m"],
                                                     "shelf_runs": scale["shelf_runs"]},
                "room": {"status": room["status"], "abstention_reasons": room["abstention_reasons"], "dimensions": room["dimensions"],
                         "areas": room["areas"], "uncertainty": room.get("uncertainty")},
                "validation": "software_only_unvalidated: no tape-measured comparison has been made"}

    def guidance_due(self, elapsed_s: float, keyframes: int, scale: dict[str, Any] | None, room: dict[str, Any]) -> list[dict[str, str]]:
        """At most one new calibration/coverage prompt per 25 s; each code issued once (re-issued only if the need returns)."""
        if elapsed_s - self._last_guidance < 25.0:
            return []
        walls_ok = room["evidence_summary"]["walls_valid"]
        candidates = []
        # Filming guidance only: the person is never asked for a size, a measurement or a tap (sizes come from the sweep).
        if elapsed_s >= 40 and walls_ok < 2:
            try:
                from pipeline.room_auto import CAPTURE_STEPS
                steps = dict(CAPTURE_STEPS)
                msg = ("When the shelves are done, for the room size: " + steps["walls_visible"] + " " + steps["corners"] + " " + steps["floor_line"] + " " + steps["ceiling_line"] +
                       " If the phone has a 0.5x wide lens, suggest switching to it. Do not ask them to stop the sweep.")
            except Exception:  # noqa: BLE001 - fall back to the original wording
                msg = ("When the shelves are done, ask the person to slowly show a vertical room corner from floor to ceiling so two walls are "
                       "in one frame, then the opposite walls. Do not ask them to stop the sweep.")
            candidates.append(("show_room_corner", msg))
        for code, msg in candidates:
            if code not in self._issued:
                self._issued[code] = elapsed_s
                self._last_guidance = elapsed_s
                return [{"code": code, "message": msg}]
        return []


def finalize_run(runs_dir: str | Path, run_id: str, recon_out_run: str, *, online: bool | None = None, budget_s: float = 60.0) -> dict[str, Any]:
    """The post-sweep measurement step (called by app.post_sweep once the 3D map is built): automatic metric scale (identified books' catalogue sizes, else the labelled
    standard-format prior), spine sizes, shelf run and room geometry from the run's reconstruction, written to a NEW ``<run>_measure_vN`` and attached to the finished
    packet in place (verified; never written when verification fails). Nobody is asked for anything. Free Open Library lookups only (cache first; LIBRARY_PRICE_ONLINE=0 = cache only)."""
    from pipeline import auto_attach as AA
    from pipeline import auto_scale as AS
    runs_dir = Path(runs_dir)
    if online is None:
        online = os.environ.get("LIBRARY_PRICE_ONLINE", "1") != "0"
    out = AS.next_output(runs_dir, run_id)
    path = AS.run(runs_dir / run_id, runs_dir / recon_out_run, out, online=online, config={"budget_s": budget_s})
    data = C.read_json(path)["data"]
    attach = AA.attach(runs_dir / run_id, path)
    sm = data.get("summary") or {}
    room = (data.get("room") or {}).get("status")
    summary = (f"{sm.get('measured', 0)}/{sm.get('book_groups', 0)} spines scaled ({data['scale'].get('source') or 'no scale'}); room {room}" +
               ("" if attach.get("attached") else f"; not attached to the packet: {attach.get('reason')}" +
                (f" ({attach['problems'][0]})" if attach.get("problems") else "")))
    return {"measure_dir": out.name, "status": data.get("status"), "scale_source": data["scale"].get("source") if data["scale"].get("cm_per_unit") else None,
            "attached": bool(attach.get("attached")), "summary": summary}


# ---------------------------------------------------------------------------------------------------------------- saved-run view + CLI
ROOM_REASON_WORDS = {
    "floor_not_found": "floor line not filmed (no floor surface in the 3D map)",
    "ceiling_not_found": "ceiling line not filmed (no ceiling surface in the 3D map)",
    "opposite_wall_not_found": "opposite wall not filmed",
    "left_wall_not_found": "left side wall not filmed",
    "right_wall_not_found": "right side wall not filmed",
    "no_metric_scale": "no metric scale",
    "floor_or_ceiling_plane_missing": "needs both the floor line and the ceiling line in the video",
    "opposite_wall_plane_missing": "needs the opposite wall in the video",
    "side_wall_planes_missing": "needs both side walls in the video",
    "needs_all_four_walls": "needs all four walls in the video",
    "needs_four_walls_and_floor_and_ceiling": "needs all four walls, the floor line and the ceiling line in the video",
    "room_geometry_not_established": "room geometry not established from the video",
}
SPINE_REASON_WORDS = {
    "polygon_clipped_by_frame_edge": "cut off by the frame edge in every view",
    "ray_grazes_or_misses_shelf_plane": "seen at too steep an angle",
    "degenerate_projected_quad": "outline unusable on the shelf plane",
    "no_oriented_polygon": "no spine outline",
    "single_usable_sighting_of_several": "only one usable view of several",
    "sightings_disagree_on_height": "views disagree on the height",
    "not_in_reconstruction": "not placed in the 3D map (its frames did not register)",
    "no_metric_scale": "no metric scale",
}


def _words(code: str | None, table: dict[str, str]) -> str | None:
    """A pipeline reason code ('floor_not_found:no_surface...', 'sightings_disagree_on_height:25%') in plain words."""
    if not code:
        return None
    head, _, tail = str(code).partition(":")
    if head == "no_usable_sighting":
        parts = [table.get(p, p.replace("_", " ")) for p in tail.split(",") if p]
        return "no usable view: " + ("; ".join(parts) or "no outline")
    if head in table:
        return table[head] + (f" ({tail})" if tail.endswith("%") else "")
    if head.startswith("implausible_"):
        return head.replace("_", " ") + " (rejected)"
    return head.replace("_", " ")


def _thickness_words(code: str | None) -> str | None:
    if not code:
        return None
    if code.startswith("library_median_thickness"):
        return "thickness withheld: spines too narrow in the video to measure a thickness"
    if code.startswith("spine_only_"):
        return "thickness withheld: " + code.split("_wide")[0].replace("_", " ") + " wide in the video"
    if code.startswith("polygon_aspect"):
        return "thickness withheld: seen face-on, not as a spine"
    return "thickness withheld: " + code.replace("_", " ")


def _pos(v: Any) -> float | None:
    """Zero is never a measurement: only positive numbers are shown."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None


def _version(p: Path, tag: str) -> int:
    try:
        return int(p.name.rsplit(tag, 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_recon3d(runs_dir: str | Path, run_id: str) -> str | None:
    """Newest FINISHED 3D map of a run (``<run>_recon3d_vN`` with recon3d.json) that is not marked inconsistent (plane_consistency), or None."""
    from pipeline.recon3d import map_consistency_at
    done = [p for p in Path(runs_dir).glob(f"{run_id}_recon3d_v*") if (p / "recon3d.json").is_file() and map_consistency_at(p) != "inconsistent"]
    return max(done, key=lambda p: _version(p, "_recon3d_v")).name if done else None


def _measure_map_ok(runs_dir: Path, measure: Path) -> bool:
    """A measurement taken on a 3D map marked inconsistent (not one flat shelf face) is not shown."""
    from pipeline.recon3d import map_consistency_at
    try:
        recon = (C.read_json(measure / "auto_measure.json").get("data") or {}).get("recon3d")
    except (OSError, ValueError, KeyError):
        return True
    return not recon or map_consistency_at(Path(runs_dir) / Path(str(recon)).name) != "inconsistent"


def latest_measure(runs_dir: str | Path, run_id: str) -> Path | None:
    done = [p for p in Path(runs_dir).glob(f"{run_id}_measure_v*") if (p / "auto_measure.json").is_file()]
    for p in sorted(done, key=lambda p: _version(p, "_measure_v"), reverse=True):
        if _measure_map_ok(Path(runs_dir), p):
            return p / "auto_measure.json"
    return None


def measured_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    out = []
    for run in sorted({p.name.rsplit("_measure_v", 1)[0] for p in Path(runs_dir).glob("*_measure_v*")}):
        path = latest_measure(runs_dir, run)
        if path is None or not (Path(runs_dir) / run).is_dir():
            continue
        try:
            d = C.read_json(path)["data"]
        except (OSError, ValueError, KeyError):
            continue
        sm = d.get("summary") or {}
        out.append({"run_id": run, "measure_dir": path.parent.name, "scale_source": (d.get("scale") or {}).get("source"),
                    "measured": sm.get("measured"), "book_groups": sm.get("book_groups"), "room_status": (d.get("room") or {}).get("status"),
                    "mtime": path.stat().st_mtime})
    return sorted(out, key=lambda r: -r["mtime"])


def auto_view(runs_dir: str | Path, run_id: str) -> dict[str, Any]:
    """What the Measurements tab shows for a saved run: per-book spine height/thickness with +-, the scale source (catalogue vs PRIOR), the shelf
    run and the room block with abstention reasons in plain words. Unknown values are null with a reason, never 0."""
    runs_dir = Path(runs_dir)
    path = latest_measure(runs_dir, run_id)
    recon = latest_recon3d(runs_dir, run_id)
    if path is None:
        return {"run_id": run_id, "available": False,
                "reason": ("3D map ready but not measured yet: python -m app.measurement --run " + run_id) if recon else
                          "no 3D map of this sweep yet (it is built automatically after End sweep)"}
    d = C.read_json(path)["data"]
    sc = d.get("scale") or {}
    src = sc.get("source") if _pos(sc.get("cm_per_unit")) else None
    try:
        packet = C.read_json(runs_dir / run_id / C.ARTIFACTS["packet"])["data"]
    except (OSError, ValueError, KeyError):
        packet = {}
    attached = ((packet.get("measurement_evidence") or {}).get("auto") or {}).get("source_measure_dir") == path.parent.name
    rows = {s["group_id"]: s for s in d.get("spines") or []}
    books = packet.get("books") or [{"id": g} for g in rows]
    out_books = []
    for b in books:
        s = rows.get(b["id"]) or {"status": "abstained", "reason": "not_in_reconstruction"}
        h, t = _pos(s.get("spine_height_cm")), _pos(s.get("spine_thickness_cm"))
        hp, tp = _pos(s.get("height_relative_uncertainty_pct")), _pos(s.get("thickness_relative_uncertainty_pct"))
        measured = s.get("status") == "measured" and h is not None
        out_books.append({
            "group_id": b["id"], "title": b.get("title") or None, "visible_text": (b.get("visible_text") or "")[:60] or None,
            "status": "measured" if measured else "abstained",
            "spine_height_cm": h if measured else None, "height_pm_cm": round(h * hp / 100, 1) if measured and hp else None, "height_pct": hp if measured else None,
            "spine_thickness_cm": t if measured else None, "thickness_pm_cm": round(t * tp / 100, 1) if measured and t and tp else None,
            "thickness_pct": tp if measured and t else None,
            "thickness_reason": (_thickness_words(s.get("thickness_reason")) or "thickness withheld") if measured and t is None else None,
            "reason": None if measured else _words(s.get("reason"), SPINE_REASON_WORDS), "sightings_used": s.get("sightings_used")})
    room = d.get("room") or {}
    def qty(block: dict[str, Any] | None, unit: str) -> dict[str, Any]:
        block = block or {}
        v = _pos(block.get("value"))
        return {"value": v, "unit": unit, "pct": _pos(block.get("relative_uncertainty_pct")) if v else None,
                "reason": None if v else (_words(block.get("reason"), ROOM_REASON_WORDS) or "not established")}
    dims, areas = room.get("dimensions") or {}, room.get("areas") or {}
    a_try = sc.get("scale_source_a_attempt") or {}
    shelf = d.get("shelf") or {}
    return {
        "run_id": run_id, "available": True, "measure_dir": path.parent.name, "recon3d": d.get("recon3d"),
        "attached_to_packet": attached, "status": d.get("status"),
        "scale": {"source": src, "label": {"catalogue_physical_dimensions": "catalogue (Open Library sizes of identified books)",
                                           "standard_format_prior": "PRIOR (assumed 22 cm median book; not a measurement)"}.get(src, "none"),
                  "not_a_measurement": bool(sc.get("not_a_measurement")), "relative_uncertainty_pct": _pos(sc.get("relative_uncertainty_pct")),
                  "catalogue_books_used": sc.get("n_inliers") if src == "catalogue_physical_dimensions" else 0,
                  "why_not_catalogue": None if src == "catalogue_physical_dimensions" else (a_try.get("reason") or sc.get("reason")),
                  "cross_check": sc.get("catalogue_cross_check"), "reason": None if src else (sc.get("reason") or "no metric scale")},
        "summary": {"books": len(out_books), "measured": sum(b["status"] == "measured" for b in out_books),
                    "with_thickness": sum(b["spine_thickness_cm"] is not None for b in out_books),
                    "abstained": sum(b["status"] != "measured" for b in out_books)},
        "books": out_books,
        "shelf": {"shelf_run_m": _pos(shelf.get("shelf_run_m")), "rows": shelf.get("rows"), "shelved_face_m2": _pos(shelf.get("shelved_face_m2")),
                  "note": shelf.get("note") or shelf.get("reason")},
        "room": {"status": room.get("status") or "abstained",
                 "dimensions": {k: qty(dims.get(k), "m") for k in ("width_m", "depth_m", "height_m")},
                 "areas": {k: qty(areas.get(k), "m2") for k in ("floor_area_m2", "wall_area_m2", "shelved_wall_area_m2")},
                 "reasons": [w for w in (_words(r, ROOM_REASON_WORDS) for r in room.get("abstention_reasons") or []) if w],
                 "film": list(room.get("capture_guidance") or [])},
        "validation": "software_only_unvalidated: not compared with a tape measure",
    }


def main(argv: list[str] | None = None) -> int:
    """python -m app.measurement --run <run>: (re)run the automatic measurement on the run's newest finished 3D map and attach it to its packet."""
    import argparse
    import json
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--run", required=True, help="run id or runs/<run> path")
    ap.add_argument("--runs-dir", default=None, help="default: the run's parent directory, else ./runs")
    ap.add_argument("--recon3d", default=None, help="a <run>_recon3d_vN name (default: newest finished)")
    ap.add_argument("--offline", action="store_true", help="Open Library cache only (no requests)")
    a = ap.parse_args(argv)
    run = Path(a.run)
    runs_dir = Path(a.runs_dir) if a.runs_dir else (run.parent if run.parent != Path(".") else Path("runs"))
    run_id = run.name
    if not (runs_dir / run_id).is_dir():
        print(f"no such run: {runs_dir / run_id}")
        return 2
    recon = a.recon3d or latest_recon3d(runs_dir, run_id)
    if not recon:
        print(f"{run_id}: no finished 3D map (<run>_recon3d_vN/recon3d.json); build it first (Debugger: Build 3D)")
        return 3
    res = finalize_run(runs_dir, run_id, recon, online=False if a.offline else None)
    state = runs_dir / "_server_state" / "post_sweep" / f"{run_id}.json"
    if state.is_file():                       # keep the Sweep page's build record in step (the server picks it up on its next start)
        try:
            rec = json.loads(state.read_text())
            rec["measure"] = {"state": "done", **res}
            state.write_text(json.dumps(rec, indent=2))
        except (OSError, ValueError):
            pass
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
