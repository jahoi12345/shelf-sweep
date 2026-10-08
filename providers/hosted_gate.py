"""Hosted kill-switch. A server-side, file-persisted flag that blocks EVERY hosted (billable) call: the structured image
transport (reader/detector/room), the live voice connect, and anything else that asks ``check()``. Independent of LIBRARY_ALLOW_HOSTED (that
controls what `auto` may pick); this is an additional hard stop the user flips from the site header. Counters of hosted calls/tokens/cost are kept
here too (process-wide, persisted), so the indicator shows what was spent regardless of which session spent it."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = Path(os.environ.get("LIBRARY_HOSTED_GATE_FILE") or ROOT / "runs" / "_server_state" / "hosted_gate.json")
_lock = threading.Lock()


class HostedDisabledError(RuntimeError):
    """Raised at every hosted choke point while the switch is off."""

    status = "hosted disabled by user"


def _default() -> dict[str, Any]:
    return {"disabled": False, "changed_at": None, "changed_by": None, "calls": 0, "blocked": 0, "cache_hits": 0, "live_sessions_closed": 0,
            "tokens": {"prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "total_tokens": 0}, "cost_usd": 0.0, "by_kind": {}}


def _load() -> dict[str, Any]:
    try:
        d = json.loads(STATE_PATH.read_text())
        base = _default(); base.update({k: v for k, v in d.items() if k in base}); return base
    except (OSError, ValueError):
        return _default()


def _save(d: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, STATE_PATH)


def disabled() -> bool:
    with _lock:
        return bool(_load()["disabled"])


def set_disabled(value: bool, actor: str = "user") -> dict[str, Any]:
    with _lock:
        d = _load()
        d["disabled"] = bool(value); d["changed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); d["changed_by"] = actor
        _save(d)
        return dict(d)


def check(kind: str) -> None:
    """Call before any hosted request. Raises HostedDisabledError (and counts the blocked attempt) while the switch is off."""
    with _lock:
        d = _load()
        if d["disabled"]:
            d["blocked"] += 1
            _save(d)
            raise HostedDisabledError(f"{HostedDisabledError.status}: {kind} call blocked by the Disconnect Gemini switch")


def record_call(kind: str, usage: dict[str, Any] | None = None, cost_usd: float | None = None) -> None:
    with _lock:
        d = _load()
        d["calls"] += 1
        d["by_kind"][kind] = d["by_kind"].get(kind, 0) + 1
        for k in d["tokens"]:
            v = (usage or {}).get(k)
            if isinstance(v, (int, float)):
                d["tokens"][k] += int(v)
        if isinstance(cost_usd, (int, float)):
            d["cost_usd"] = round(d["cost_usd"] + float(cost_usd), 6)
        _save(d)


def record_cache_hit(kind: str) -> None:
    """A hosted response served from the shared cache: no request, no cost; counted so reruns show what they saved."""
    with _lock:
        d = _load(); d["cache_hits"] += 1; d["by_kind"]["cache_hit:" + kind] = d["by_kind"].get("cache_hit:" + kind, 0) + 1; _save(d)


def note_live_closed(n: int = 1) -> None:
    with _lock:
        d = _load(); d["live_sessions_closed"] += n; _save(d)


def summary() -> dict[str, Any]:
    with _lock:
        d = _load()
    return {**d, "status": HostedDisabledError.status if d["disabled"] else "hosted enabled", "state_file": str(STATE_PATH),
            "note": "Switch blocks the hosted transport, the live voice connect and the hosted reader/detector/room adapters; LIBRARY_ALLOW_HOSTED still governs what `auto` may pick."}
