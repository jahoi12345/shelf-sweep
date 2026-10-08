"""Held-out tape-truth capture.

Entries are the human's own tape readings, saved as ``purpose: evaluation_truth`` under ``eval/truth_capture/<session>.json``,
OUTSIDE every run directory. Nothing in app/engine, pipeline/* or the packet bundle reads this module's files: pipeline.scale/room
refuse any file whose purpose is not calibration/room_evidence, the bundle lists explicit file names only, and a test asserts no
pipeline/engine source mentions the directory. Truth is separate from calibration inputs (docs/templates/tape_truth_template.json).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRUTH_DIR = Path(os.getenv("LIBRARY_TRUTH_DIR", ROOT / "eval" / "truth_capture"))
MAX_SPINES = 40
SID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
UNIT_CM = {"cm": 1.0, "mm": 0.1, "in": 2.54, "m": 100.0}
UNIT_M = {"m": 1.0, "cm": 0.01, "ft": 0.3048, "in": 0.0254, "mm": 0.001}


class TruthError(ValueError):
    pass


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        raise TruthError(f"{what} must be a number")
    return float(v)


def _path(session_id: str) -> Path:
    if not SID.fullmatch(session_id):
        raise TruthError("invalid session id")
    return TRUTH_DIR / f"{session_id}.json"


def load(session_id: str) -> dict[str, Any]:
    p = _path(session_id)
    if p.is_file():
        return json.loads(p.read_text())
    return {"schema_version": 1, "purpose": "evaluation_truth", "session_id": session_id, "annotator": None, "scan": None,
            "spine_samples": [], "room": None,
            "note": "HELD-OUT tape readings. Never given to pipeline.scale / pipeline.room / the app engine / the bundle."}


def _save(session_id: str, doc: dict[str, Any]) -> None:
    p = _path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def add_spine(session_id: str, *, frame: dict[str, Any], scan: dict[str, Any] | None, tap_px: list[float], height: float, thickness: float,
              unit: str = "cm", annotator: str, description: str = "", language: str | None = None) -> dict[str, Any]:
    if unit not in UNIT_CM:
        raise TruthError(f"unit must be one of {sorted(UNIT_CM)}")
    if not annotator or not str(annotator).strip():
        raise TruthError("annotator name is required (an attributable human)")
    if not (isinstance(tap_px, list) and len(tap_px) == 2):
        raise TruthError("tap one point on the book in the frame")
    x, y = _num(tap_px[0], "tap x"), _num(tap_px[1], "tap y")
    if not (0 <= x <= frame["width_px"] and 0 <= y <= frame["height_px"]):
        raise TruthError("tap is outside the frame")
    h, t = _num(height, "height") * UNIT_CM[unit], _num(thickness, "thickness") * UNIT_CM[unit]
    if not 5 <= h <= 60 or not 0.3 <= t <= 12:
        raise TruthError("implausible tape reading: height must be 5-60 cm and thickness 0.3-12 cm (check the unit)")
    if t >= h:
        raise TruthError("thickness must be smaller than height (the spine's long side is its height)")
    if language not in (None, "", "es", "en", "mixed"):
        raise TruthError("language must be es, en or mixed")
    doc = load(session_id)
    if len(doc["spine_samples"]) >= MAX_SPINES:
        raise TruthError(f"at most {MAX_SPINES} spine samples")
    doc["annotator"] = doc["annotator"] or str(annotator).strip()[:80]
    doc["scan"] = scan or doc.get("scan")
    sample = {"sample_id": f"s{len(doc['spine_samples']) + 1:02d}", "frame_id": frame["frame_id"], "frame_timestamp_s": frame.get("timestamp_s"),
              "tap_px": [round(x), round(y)], "description": str(description)[:200], "language": language or None,
              "height_cm": round(h, 2), "thickness_cm": round(t, 2), "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "annotator": doc["annotator"], "unit_entered": unit}
    # unique ids even after deletions
    used = {s["sample_id"] for s in doc["spine_samples"]}
    n = 1
    while f"s{n:02d}" in used:
        n += 1
    sample["sample_id"] = f"s{n:02d}"
    doc["spine_samples"].append(sample)
    _save(session_id, doc)
    return sample


def remove_spine(session_id: str, sample_id: str) -> None:
    doc = load(session_id)
    n = len(doc["spine_samples"])
    doc["spine_samples"] = [s for s in doc["spine_samples"] if s["sample_id"] != sample_id]
    if len(doc["spine_samples"]) == n:
        raise TruthError("unknown sample")
    _save(session_id, doc)


def set_room(session_id: str, *, length: float, width: float, height: float, unit: str, annotator: str, notes: str = "",
             floor_area_m2: float | None = None, wall_area_m2: float | None = None, scan: dict[str, Any] | None = None) -> dict[str, Any]:
    if unit not in UNIT_M:
        raise TruthError(f"unit must be one of {sorted(UNIT_M)}")
    if not annotator or not str(annotator).strip():
        raise TruthError("annotator name is required")
    l, w, h = (_num(v, n) * UNIT_M[unit] for v, n in ((length, "length"), (width, "width"), (height, "height")))
    if not (0.5 <= l <= 40 and 0.5 <= w <= 40 and 1.5 <= h <= 8):
        raise TruthError("implausible room dimensions (check the unit)")
    doc = load(session_id)
    doc["annotator"] = doc["annotator"] or str(annotator).strip()[:80]
    doc["scan"] = scan or doc.get("scan")
    room = {"length_m": round(l, 3), "width_m": round(w, 3), "height_m": round(h, 3), "notes": str(notes)[:400],
            "floor_area_m2": None if floor_area_m2 is None else _num(floor_area_m2, "floor area"),
            "wall_area_m2": None if wall_area_m2 is None else _num(wall_area_m2, "wall area"),
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "unit_entered": unit}
    doc["room"] = room
    _save(session_id, doc)
    return room
