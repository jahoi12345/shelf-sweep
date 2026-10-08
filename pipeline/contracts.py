"""Shared stage contracts, version 1.

Plain serialisable dicts are the wire format. This module holds the constants, ID helpers,
coordinate helpers and validators every stage uses. Conventions:

* Unknown scalar -> ``None``; no observations -> ``[]``; a genuine zero stays ``0``.
* Boxes are ``[x_min, y_min, x_max, y_max]`` in *original saved-frame pixels*.
* Paths inside artifacts are relative to the run directory.
* Every artifact is an envelope: schema_version, run_id, stage, status, data, issues, metrics.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

# Stable artifact file names, relative to the run directory.
ARTIFACTS = {
    "frames": "manifest.json",
    "detect": "candidates.json",
    "read": "readings.json",
    "resolve": "resolutions.json",
    "dedup": "inventory.json",
    "scale": "scale.json",
    "room": "room.json",
    "price": "prices.json",
    "packet": "claim_packet.json",
    "report": "report.html",
}
STAGE_ORDER = ["frames", "detect", "read", "resolve", "dedup", "scale", "room", "price", "packet", "report"]

STATUSES = {"ok", "complete", "partial", "failed", "not_implemented", "skipped"}  # "complete" == "ok" (alternate spelling)
OK_STATUSES = {"ok", "complete"}
# Accepted spellings of each stage name inside an artifact envelope (stage ids used by run.py are the keys).
STAGE_ALIASES = {"frames": {"frames", "extract"}, "detect": {"detect", "detection"},
                 "read": {"read", "read_spines"}, "resolve": {"resolve", "resolve_catalogue"},
                 "dedup": {"dedup", "merge", "inventory"}, "scale": {"scale"}, "room": {"room"}, "price": {"price"}, "packet": {"packet"}, "report": {"report"}}
DETECTION_KINDS = {"book", "non_book", "ambiguous"}
ORIENTATIONS = {"upright", "leaning", "flat", "stacked", "unknown"}
RESOLUTION_DECISIONS = {"accepted", "ambiguous", "no_match"}
BOOK_STATUSES = {"identified", "unidentified", "needs_appraisal"}
ITEM_STATUSES = {"priced", "range", "needs_appraisal"}


class ContractError(ValueError):
    """Raised when an artifact breaks the shared contract."""


# --------------------------------------------------------------------------- ids

def frame_id(index: int) -> str:
    """Deterministic frame id from the 0-based index in the sampled sequence."""
    return f"f{index:05d}"


def observation_id(fid: str, n: int) -> str:
    """One detection in one frame. ``n`` is the 0-based detection index within the frame."""
    return f"{fid}-o{n:03d}"


def instance_id(kind: str, n: int) -> str:
    """Legacy enumeration id of one physical item across frames (``bk-012``). ``kind``: 'bk' (book) or 'it' (non-book item).
    Enumeration ids are renumbered by every merge; new inventories use ``stable_instance_id``."""
    return f"{kind}-{n:03d}"


def stable_instance_id(kind: str, earliest_observation_id: str) -> str:
    """Stable id of one physical item across frames, derived from the EARLIEST observation of its group (``bk-f00012-o003``).
    A merge keeps the earliest member, so the id survives re-merges and sorts in capture order. ``kind``: 'bk' or 'it'."""
    if not _ID_RE["observation"].match(str(earliest_observation_id)):
        raise ContractError(f"not an observation id: {earliest_observation_id!r}")
    return f"{kind}-{earliest_observation_id}"


# ---------------------------------------------------------------------- coordinates

def normalized_box_to_pixels(box_yxyx_1000: Iterable[float], width_px: int, height_px: int) -> list[float]:
    """Convert a Gemini-style ``[ymin, xmin, ymax, xmax]`` 0-1000 box to ``[x0, y0, x1, y1]`` pixels.

    Values are clamped to the frame. Raises ContractError if the result is empty/degenerate.
    """
    try:
        y0, x0, y1, x1 = (float(v) for v in box_yxyx_1000)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"box must be four numbers, got {box_yxyx_1000!r}") from exc
    px = [x0 * width_px / 1000.0, y0 * height_px / 1000.0, x1 * width_px / 1000.0, y1 * height_px / 1000.0]
    px = [min(max(px[0], 0.0), width_px), min(max(px[1], 0.0), height_px),
          min(max(px[2], 0.0), width_px), min(max(px[3], 0.0), height_px)]
    if px[2] <= px[0] or px[3] <= px[1]:
        raise ContractError(f"degenerate box after conversion: {px}")
    return [round(v, 1) for v in px]


def check_bbox(bbox: Any, width_px: int, height_px: int, what: str = "bbox_xyxy") -> list[str]:
    """Return a list of problems with a pixel box (empty list = valid)."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [f"{what}: must be [x_min,y_min,x_max,y_max]"]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox):
        return [f"{what}: non-numeric value"]
    x0, y0, x1, y1 = bbox
    problems = []
    if not (x0 < x1 and y0 < y1):
        problems.append(f"{what}: not x_min<x_max and y_min<y_max ({bbox})")
    if x0 < 0 or y0 < 0 or x1 > width_px or y1 > height_px:
        problems.append(f"{what}: outside {width_px}x{height_px} frame ({bbox})")
    return problems


# ---------------------------------------------------------------------------- io

def envelope(run_id: str, stage: str, data: dict[str, Any], *, status: str = "ok",
             issues: list[dict[str, Any]] | None = None, metrics: dict[str, Any] | None = None,
             **extra: Any) -> dict[str, Any]:
    if status not in STATUSES:
        raise ContractError(f"unknown status {status!r}")
    env = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "stage": stage, "status": status,
           "data": data, "issues": issues or [], "metrics": metrics or {}}
    env.update(extra)
    return env


def write_json(path: str | Path, obj: Any) -> Path:
    """Atomic JSON write (UTF-8, stable key order is preserved as given)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rel_path(path: str | Path, run_dir: str | Path) -> str:
    """Run-relative POSIX path. Raises if ``path`` is outside the run directory."""
    return Path(path).resolve().relative_to(Path(run_dir).resolve()).as_posix()


def resolve_in_run(run_dir: str | Path, relative: str) -> Path:
    """Resolve a run-relative path, refusing absolute paths and escapes."""
    if os.path.isabs(relative) or ".." in Path(relative).parts:
        raise ContractError(f"path must be run-relative without '..': {relative!r}")
    return Path(run_dir) / relative


# ------------------------------------------------------------------- validators

_ID_RE = {
    "frame": re.compile(r"^f\d{5}$"),
    "observation": re.compile(r"^f\d{5}-o\d{3}$"),
    "book": re.compile(r"^bk-(\d{3,}|f\d{5}-o\d{3})$"),      # legacy enumeration (bk-012) or stable earliest-observation id (bk-f00012-o003)
    "item": re.compile(r"^it-(\d{3,}|f\d{5}-o\d{3})$"),
}


def _check_envelope(art: Any, stage: str | None) -> list[str]:
    if not isinstance(art, dict):
        return ["artifact is not an object"]
    problems = []
    for key in ("schema_version", "run_id", "stage", "status", "data", "issues", "metrics"):
        if key not in art:
            problems.append(f"missing top-level key {key!r}")
    if art.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if art.get("status") not in STATUSES:
        problems.append(f"bad status {art.get('status')!r}")
    if stage and art.get("stage") not in STAGE_ALIASES.get(stage, {stage}):
        problems.append(f"stage is {art.get('stage')!r}, expected one of {sorted(STAGE_ALIASES.get(stage, {stage}))}")
    return problems


def _check_conf(value: Any, what: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        return [f"{what}: confidence must be null or within [0,1], got {value!r}"]
    return []


def validate_manifest(art: dict[str, Any], run_dir: str | Path | None = None) -> list[str]:
    problems = _check_envelope(art, "frames")
    frames = (art.get("data") or {}).get("frames")
    if not isinstance(frames, list):
        return problems + ["data.frames must be a list"]
    seen: set[str] = set()
    last_t = -1.0
    for fr in frames:
        fid = fr.get("frame_id", "?")
        if not _ID_RE["frame"].match(str(fid)):
            problems.append(f"{fid}: bad frame_id")
        if fid in seen:
            problems.append(f"{fid}: duplicate frame_id")
        seen.add(fid)
        for key in ("timestamp_s", "image_path", "width_px", "height_px", "quality", "selected", "selection_reason"):
            if key not in fr:
                problems.append(f"{fid}: missing {key}")
        t = fr.get("timestamp_s")
        if isinstance(t, (int, float)):
            if t < last_t:
                problems.append(f"{fid}: timestamps not monotonic")
            last_t = t
        if run_dir is not None and fr.get("image_path"):
            try:
                if not resolve_in_run(run_dir, fr["image_path"]).is_file():
                    problems.append(f"{fid}: image_path does not exist: {fr['image_path']}")
            except ContractError as exc:
                problems.append(f"{fid}: {exc}")
    return problems


def validate_candidates(art: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    problems = _check_envelope(art, "detect")
    dims = {f["frame_id"]: (f["width_px"], f["height_px"]) for f in manifest["data"]["frames"]}
    seen: set[str] = set()
    data = art.get("data") or {}
    # Canonical key is data.observations; data.detections is accepted as an alias.
    for det in (data.get("observations") if data.get("observations") is not None else data.get("detections")) or []:
        oid = det.get("observation_id", "?")
        if not _ID_RE["observation"].match(str(oid)):
            problems.append(f"{oid}: bad observation_id")
        if oid in seen:
            problems.append(f"{oid}: duplicate observation_id")
        seen.add(oid)
        fid = det.get("frame_id")
        if fid not in dims:
            problems.append(f"{oid}: unknown frame_id {fid!r}")
            continue
        if not str(oid).startswith(f"{fid}-"):
            problems.append(f"{oid}: observation_id does not belong to frame {fid}")
        problems += [f"{oid}: {p}" for p in check_bbox(det.get("bbox_xyxy"), *dims[fid])]
        if det.get("polygon_xy"):
            w, h = dims[fid]
            for x, y in det["polygon_xy"]:
                if not (0 <= x <= w and 0 <= y <= h):
                    problems.append(f"{oid}: polygon vertex outside frame")
                    break
        if det.get("kind") not in DETECTION_KINDS:
            problems.append(f"{oid}: kind must be one of {sorted(DETECTION_KINDS)}")
        problems += [f"{oid}: {p}" for p in _check_conf(det.get("confidence"), "confidence")]
    return problems


def validate_readings(art: dict[str, Any], candidate_ids: set[str]) -> list[str]:
    problems = _check_envelope(art, "read")
    for rd in (art.get("data") or {}).get("readings", []) or []:
        oid = rd.get("observation_id", "?")
        if oid not in candidate_ids:
            problems.append(f"{oid}: reading for unknown observation")
        if "visible_text" not in rd:
            problems.append(f"{oid}: missing visible_text")
        problems += [f"{oid}: {p}" for p in _check_conf(rd.get("read_confidence"), "read_confidence")]
    return problems


def validate_resolutions(art: dict[str, Any], candidate_ids: set[str]) -> list[str]:
    problems = _check_envelope(art, "resolve")
    for rs in (art.get("data") or {}).get("resolutions", []) or []:
        oid = rs.get("observation_id", "?")
        if oid not in candidate_ids:
            problems.append(f"{oid}: resolution for unknown observation")
        if rs.get("decision") not in RESOLUTION_DECISIONS:
            problems.append(f"{oid}: decision must be one of {sorted(RESOLUTION_DECISIONS)}")
        if rs.get("decision") == "accepted" and not rs.get("source_url"):
            problems.append(f"{oid}: accepted resolution has no source_url")
    return problems


def validate_inventory(art: dict[str, Any], manifest: dict[str, Any], candidate_ids: set[str] | None = None) -> list[str]:
    problems = _check_envelope(art, "dedup")
    dims = {f["frame_id"]: (f["width_px"], f["height_px"]) for f in manifest["data"]["frames"]}
    data = art.get("data") or {}
    owner: dict[str, str] = {}
    for key, kind in (("books", "book"), ("items", "item")):
        seen: set[str] = set()
        for inst in data.get(key, []) or []:
            iid = inst.get("instance_id", "?")
            if not _ID_RE[kind].match(str(iid)):
                problems.append(f"{iid}: bad {kind} instance_id")
            if iid in seen:
                problems.append(f"{iid}: duplicate instance_id")
            seen.add(iid)
            obs = inst.get("observation_ids") or []
            if not obs:
                problems.append(f"{iid}: no observation_ids")
            for oid in obs:
                if candidate_ids is not None and oid not in candidate_ids:
                    problems.append(f"{iid}: unknown observation {oid}")
                if oid in owner and owner[oid] != iid:
                    problems.append(f"{oid}: assigned to both {owner[oid]} and {iid}")
                owner[oid] = iid
            ref = inst.get("frame_ref")
            if ref not in dims:
                problems.append(f"{iid}: frame_ref {ref!r} not in manifest")
            elif inst.get("bbox_xyxy") is not None:
                problems += [f"{iid}: {p}" for p in check_bbox(inst["bbox_xyxy"], *dims[ref])]
            problems += [f"{iid}: {p}" for p in _check_conf(inst.get("confidence"), "confidence")]
    return problems


def validate_artifact(stage: str, art: dict[str, Any], *, manifest: dict[str, Any] | None = None,
                      candidate_ids: set[str] | None = None, run_dir: str | Path | None = None) -> list[str]:
    """Dispatch to the right validator. Returns problems; empty list means valid."""
    if stage == "frames":
        return validate_manifest(art, run_dir)
    if manifest is None:
        return ["manifest required to validate this stage"]
    if stage == "detect":
        return validate_candidates(art, manifest)
    if stage == "read":
        return validate_readings(art, candidate_ids or set())
    if stage == "resolve":
        return validate_resolutions(art, candidate_ids or set())
    if stage == "dedup":
        return validate_inventory(art, manifest, candidate_ids)
    return _check_envelope(art, stage)
