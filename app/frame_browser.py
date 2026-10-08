"""Read-only indexes for saved frame evidence; live sessions keep their owner checks."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from app import run_visibility as RV

RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FRAME_ID = re.compile(r"^f\d{5}$")


def saved_run(root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    if not RUN_NAME.fullmatch(name) or name == "live":
        raise ValueError("No such saved run")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError("No such saved run")
    manifest = path / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("No such saved run")
    try:
        envelope = json.loads(manifest.read_text())
        if not isinstance(envelope.get("data", {}).get("frames"), list):
            raise ValueError("Invalid frame manifest")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("Invalid frame manifest") from exc
    return path, envelope


def evidence_path(run: Path, frame_id: str) -> Path:
    if not FRAME_ID.fullmatch(frame_id):
        raise ValueError("No such frame")
    path = run / "frames" / f"{frame_id}.jpg"
    if (not path.is_file() or path.is_symlink()
            or not path.resolve().is_relative_to(run.resolve())):
        raise ValueError("No such frame")
    return path


def index_frames(records: list[dict[str, Any]], image_prefix: str) -> list[dict[str, Any]]:
    frames = []
    for record in records:
        fid = record.get("frame_id", "")
        if not isinstance(fid, str) or not FRAME_ID.fullmatch(fid):
            continue
        frames.append({"frame_id": fid, "timestamp_s": record.get("timestamp_s"),
                       "width_px": record.get("width_px"), "height_px": record.get("height_px"),
                       "selected": bool(record.get("selected")),
                       "selection_reason": record.get("selection_reason") or "",
                       "quality": copy.deepcopy(record.get("quality") or {}),
                       "image_url": f"{image_prefix}/{fid}.jpg"})
    return frames


def saved_index(root: Path, name: str) -> dict[str, Any]:
    _, envelope = saved_run(root, name)
    data = envelope["data"]
    frames = index_frames(data["frames"], f"/api/frame-runs/{name}/frames")
    return {"run_id": name, "video_file": data.get("input", {}).get("file_name"),
            "fixture": bool(envelope.get("fixture")), "frames": frames,
            "frame_count": len(frames), "keyframe_count": sum(f["selected"] for f in frames)}


def list_saved_runs(root: Path, show_all: bool = True) -> list[dict[str, Any]]:
    """Featured runs (app.run_visibility) first, newest first; show_all=False hides the rest (still on disk)."""
    runs, pins, names = [], RV.pinned(root), RV.labels(root)
    for manifest in root.glob("*/manifest.json"):
        name = manifest.parent.name
        featured = RV.featured(name, root, pins)
        if not show_all and not featured:
            continue
        try:
            index = saved_index(root, name)
        except ValueError:
            continue
        if index["frames"]:
            runs.append({**{key: value for key, value in index.items() if key != "frames"},
                         "featured": featured, "created": RV.created(manifest), "label": names.get(name)})
    runs.sort(key=lambda r: r["created"] or "", reverse=True)
    return sorted(runs, key=lambda r: not r["featured"])
