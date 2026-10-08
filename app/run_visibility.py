"""Which saved runs the Frames and Debugger lists show by default.

Featured: site sweeps (sweep_*), final runs (final_*, not final_candidate_*), offline_baseline, any run named in
runs/_server_state/pinned_runs.json (a JSON list) and panorama / 3D outputs of those (<run>_panorama_v<N>, <run>_recon3d_v<N>). Everything else stays on disk untouched behind "Show all runs".
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

PINNED_FILE = Path("_server_state") / "pinned_runs.json"
LABELS_FILE = Path("_server_state") / "run_labels.json"   # {"sweep_x": "IESTest1"}: human names for runs


def pinned(root: Path) -> set[str]:
    try:
        return {str(n) for n in json.loads((Path(root) / PINNED_FILE).read_text())}
    except (OSError, ValueError, TypeError):
        return set()


PANORAMA_OUTPUT = re.compile(r"^(?P<source>.+)_(?:pano|panorama|recon3d)_v\d+$")   # runs/<run>_panorama_v<N> (legacy _pano_v<N>) / _recon3d_v<N>, written by app.build_jobs


def featured(name: str, root: Path, pins: set[str] | None = None) -> bool:
    # A panorama built from a featured run is featured too (its source is what the user is looking at).
    built = PANORAMA_OUTPUT.match(name)
    if built and featured(built["source"], root, pins):
        return True
    if name in (pins if pins is not None else pinned(root)) or name == "offline_baseline" or name in labels(root):
        return True
    return name.startswith("sweep_") or (name.startswith("final_") and not name.startswith("final_candidate"))


def created(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return None


def labels(root: Path) -> dict[str, str]:
    try:
        return {str(k): str(v) for k, v in json.loads((Path(root) / LABELS_FILE).read_text()).items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}
