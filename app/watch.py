"""Read-only laptop watch page data.

The phone's sweep runs in the :4183 process (scripts/serve_https.sh); the laptop opens http://127.0.0.1:4182/watch, a different process that
shares runs/. So the page reads only what is on disk:

  * ``<run>/watch.json`` — the engine's small status file (app.engine.SweepEngine.watch_status: status, frames, latest frame, live count,
    capture indicator, notices, end-sweep step and progress, first count). An engine in THIS process is read directly instead.
  * ``runs/_server_state/post_sweep/<run>.json`` via the post-sweep queue's ``status(run_id)`` (2D panorama, count, duplicate merging, 3D map,
    3D join, measurement, with build progress).
  * ``<run>/claim_packet.json`` totals (final count, range, reliability), cached by mtime; plus a compact viewer of the whole packet
    (status, issues, totals, books, items, pricing, room/measurement, stage_status) and the packet itself with absolute paths redacted.
  * ``<run>/live/transcript.jsonl``: the voice conversation, one line per finished turn, appended by app.server.append_turn.

A sweep saved before the watch page existed has no watch.json: its state is derived from the packet and the frames folder. Nothing here writes anything, and nothing
returned carries the owner cookie, the pairing key or an absolute path. The routes are served only on the loopback server (see app.server).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

RUN_RE = re.compile(r"^sweep_[0-9a-f]{16}$")
FRAME_RE = re.compile(r"^f\d{5}\.jpg$")
ENGINE_ACTIVE = {"capturing", "finalizing"}
POST_ACTIVE = {"queued", "waiting", "building_2d", "building_3d", "counting", "merging", "measuring", "joining_3d", "reading"}   # app.post_sweep.ACTIVE + "reading" (finishing the title reading; not a New-sweep blocker)
CROP_RE = re.compile(r"^[0-9a-f]{8,64}\.png$")
TOTAL_KEYS = ("book_count", "book_count_range", "book_count_lower", "book_count_upper", "count_reliability", "item_count", "books_identified",
              "headline_estimator", "book_lines", "book_lines_range", "book_count_range_label", "count_range_basis", "panorama_count",
              "books_replacement_cost", "books_used_value")
TRANSCRIPT_FILE = "transcript.jsonl"            # <run>/live/transcript.jsonl (written by app.server.append_turn, server sessions only)
TRANSCRIPT_MAX_TURNS = 2000
# absolute paths inside strings (the packet's provenance carries a few: OCR response files, a reader's venv); the watch page never shows them
ABS_PATH_RE = re.compile(r"(?<![\w.:/])/(?:Users|home|private|tmp|var|Volumes|opt|root|mnt|srv|etc|usr|Library|System)/[^\s\"'<>,;)\]]*")

_cache: dict[str, tuple[float, Any]] = {}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _cached(path: Path, fn: Callable[[Any], Any]) -> Any:
    """fn(parsed JSON) cached by (path, mtime): the packet of a 750-book sweep is several MB and the page polls every second."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key = f"{path}|{fn.__qualname__}"                 # the packet has two readers (totals, viewer)
    hit = _cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    raw = _read_json(path)
    value = fn(raw) if raw is not None else None
    if len(_cache) > 64:
        _cache.clear()
    _cache[key] = (mtime, value)
    return value


def _strip_path(m: re.Match) -> str:
    p = m.group(0)
    i = p.rfind("/runs/")
    if i >= 0:
        return p[i + len("/runs/"):] or "runs"
    return "…/" + p.rstrip("/").rsplit("/", 1)[-1]


def scrub(value: Any) -> Any:
    """Copy of value with every absolute path inside a string replaced by its run-relative tail (or …/basename)."""
    if isinstance(value, str):
        return ABS_PATH_RE.sub(_strip_path, value) if "/" in value else value
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def run_dir(runs_dir: Path, run_id: str) -> Path | None:
    if not RUN_RE.fullmatch(run_id or ""):
        return None
    d = Path(runs_dir) / run_id
    return d if d.is_dir() else None


def packet_totals(d: Path) -> dict[str, Any] | None:
    def pick(env: Any) -> dict[str, Any] | None:
        data = (env or {}).get("data") or {}
        t = data.get("totals") or {}
        sweep = data.get("sweep") or {}
        return {**{k: t.get(k) for k in TOTAL_KEYS}, "finalization_s": sweep.get("finalization_s"), "captured_at": sweep.get("captured_at"),
                "ended_at": sweep.get("ended_at"), "packet_status": (env or {}).get("status")}
    return _cached(d / "claim_packet.json", pick)


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _money(v: Any) -> dict[str, Any] | None:
    if not isinstance(v, dict) or v.get("amount") is None:
        return None
    return {k: v.get(k) for k in ("amount", "currency", "converted", "original_amount", "original_currency", "source", "condition_assumed", "url")}


def _crop_name(b: dict[str, Any]) -> str | None:
    for r in b.get("reading_evidence") or []:
        p = str((r or {}).get("crop_path") or "")
        if p.startswith("crops/") and CROP_RE.fullmatch(p[6:]):
            return p[6:]
    for m in b.get("merged_members") or []:
        c = _crop_name(m or {})
        if c:
            return c
    return None


def _book_row(b: dict[str, Any]) -> dict[str, Any]:
    nulls = b.get("pricing_null_reasons") or {}
    return {"i": b.get("display_index"), "id": b.get("id"), "title": b.get("title"), "author": b.get("author"), "publisher": b.get("publisher"),
            "status": b.get("status"), "decision": b.get("identity_decision"), "confidence": b.get("id_confidence"), "tier": b.get("count_tier"),
            "visible_text": (b.get("visible_text") or "")[:160] or None, "frame": b.get("frame_id"), "t_s": b.get("timestamp_s"),
            "sightings": b.get("sightings"), "members": len(b.get("member_group_ids") or []) or None, "language": b.get("language"),
            "replacement_cost": _money(b.get("replacement_cost")), "used_value": _money(b.get("used_value")),
            "price_note": None if (b.get("replacement_cost") or b.get("used_value")) else (nulls.get("used_value") or nulls.get("replacement_cost")),
            "isbn": b.get("isbn") or b.get("equivalent_isbn"), "source_url": b.get("identity_source_url"), "crop": _crop_name(b)}


def _item_row(it: dict[str, Any]) -> dict[str, Any]:
    return {"i": it.get("display_index"), "id": it.get("id"), "category": it.get("category"), "description": it.get("description"),
            "brand_model": it.get("brand_model"), "status": it.get("status"), "status_reason": it.get("status_reason"), "confidence": it.get("confidence"), "frame": it.get("frame_id"),
            "t_s": it.get("timestamp_s"), "replacement_cost": _money(it.get("replacement_cost")) if isinstance(it.get("replacement_cost"), dict) else it.get("replacement_cost")}


REVIEW_ROWS = 200   # review entries shown in the viewer (one per line/topic, most severe first); the raw JSON has all of them


def _second_country_view(sc: Any) -> dict[str, Any] | None:
    if not isinstance(sc, dict):
        return None
    return {"status": sc.get("status"), "country": sc.get("country"), "currency": sc.get("currency"), "source": sc.get("source"), "totals": sc.get("totals"),
            "books": [{"i": b.get("display_index"), "title": b.get("title"), "isbn": b.get("isbn"), "stale": b.get("stale"),
                       "us_new": _money((b.get("us") or {}).get("replacement_cost")), "us_used": _money((b.get("us") or {}).get("used_value")),
                       "new": _money((b.get("second") or {}).get("replacement_cost")), "used": _money((b.get("second") or {}).get("used_value")),
                       "issues": b.get("issues") or []} for b in sc.get("books") or [] if isinstance(b, dict)]}


def _packet_view(env: Any) -> dict[str, Any] | None:
    if not isinstance(env, dict):
        return None
    data = env.get("data") or {}
    totals = {k: v for k, v in (data.get("totals") or {}).items() if k != "shelf_run_detail"}
    pricing = {k: v for k, v in (data.get("pricing") or {}).items() if k != "prices"}
    meas = data.get("measurement_evidence") or {}
    auto = meas.get("auto") or {}
    count = data.get("count") or {}
    sweep = data.get("sweep") or {}
    view = {"status": env.get("status"), "issues": env.get("issues") or [], "metrics": env.get("metrics") or {},
            "completion_status": data.get("completion_status"), "provisional_notice": data.get("provisional_notice"),
            "totals": totals, "stage_status": data.get("stage_status") or {}, "pricing": pricing,
            "completeness": data.get("completeness") or {}, "language_breakdown": data.get("language_breakdown"),
            "room": data.get("room") or {}, "measurement": {"status": auto.get("status"), "scale": auto.get("scale"), "shelf": auto.get("shelf"),
                                                             "room": auto.get("room"), "validation": auto.get("validation"),
                                                             "scale_method": (meas.get("scale") or {}).get("method")},
            "count": {k: count.get(k) for k in ("best_guess", "book_groups", "uncertainty_band", "count_range_basis", "proposed_method", "confident_groups",
                                                 "singleton_groups", "unconfirmed_groups") if k in count},
            "sweep": {k: sweep.get(k) for k in ("captured_at", "ended_at", "duration_s", "country", "currency", "source", "finalization_s", "device")},
            "lists": {"books": len(data.get("books") or []), "items": len(data.get("items") or []), "unconfirmed_books": len(data.get("unconfirmed_books") or []),
                      "review_queue": len(data.get("review_queue") or []), "coverage_gaps": len(data.get("coverage_gaps") or []),
                      "review_details": sum(int(r.get("detail_count") or 1) for r in data.get("review_queue") or [] if isinstance(r, dict))},
            "review": [{"ref_id": r.get("ref_id"), "severity": r.get("severity"), "codes": r.get("codes") or [r.get("code")], "reason": r.get("reason"),
                        "details": r.get("detail_count") or 1} for r in (data.get("review_queue") or [])[:REVIEW_ROWS] if isinstance(r, dict)],
            "second_country": _second_country_view(data.get("second_country")),
            "books": [_book_row(b) for b in data.get("books") or [] if isinstance(b, dict)],
            "items": [_item_row(i) for i in data.get("items") or [] if isinstance(i, dict)]}
    return scrub(view)


def packet_view(d: Path) -> dict[str, Any] | None:
    """The claim packet for the watch page's viewer: everything a reader needs, books/items as compact rows, cached by mtime, paths redacted."""
    v = _cached(d / "claim_packet.json", _packet_view)
    return None if v is None else {**v, "packet_mtime": _mtime(d / "claim_packet.json")}


_raw_cache: dict[str, tuple[float, bytes]] = {}


def packet_raw(d: Path) -> bytes | None:
    """The whole claim packet as JSON with absolute paths redacted (raw view / download). One entry cached (the packet can be ~10 MB)."""
    path = d / "claim_packet.json"
    if path.is_symlink():
        return None
    mtime = _mtime(path)
    if mtime is None:
        return None
    hit = _raw_cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    raw = _read_json(path)
    if raw is None:
        return None
    body = json.dumps(scrub(raw), ensure_ascii=False).encode()
    _raw_cache.clear()
    _raw_cache[str(path)] = (mtime, body)
    return body


def transcript(d: Path) -> dict[str, Any]:
    """The voice conversation of a sweep (<run>/live/transcript.jsonl): speaker, text and time of every finished turn; no ids, no audio."""
    path = d / "live" / TRANSCRIPT_FILE
    if path.is_symlink() or not path.is_file():
        return {"persisted": False, "turns": []}
    turns = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines[-TRANSCRIPT_MAX_TURNS:]:
        try:
            t = json.loads(line)
        except ValueError:
            continue                                   # a line being written right now
        if isinstance(t, dict) and isinstance(t.get("text"), str):
            turns.append({"speaker": t.get("speaker"), "text": t["text"], "at_epoch": t.get("at_epoch"), "at_s": t.get("at_s")})
    return {"persisted": True, "turns": scrub(turns)}


def _legacy_duration(d: Path) -> float | None:
    def pick(env: Any) -> float | None:
        frames = ((env or {}).get("data") or {}).get("frames") or []
        return frames[-1].get("timestamp_s") if frames else None
    return _cached(d / "manifest.json", pick)


def _frames_on_disk(d: Path) -> tuple[int, str | None]:
    n, last = 0, None
    try:
        with os.scandir(d / "frames") as it:
            for e in it:
                if FRAME_RE.fullmatch(e.name):
                    n += 1
                    if last is None or e.name > last:
                        last = e.name
    except OSError:
        pass
    return n, (last[:-4] if last else None)


def engine_status(d: Path, live: dict[str, Any] | None) -> dict[str, Any]:
    """The engine part of the state: from a live engine in this process, else watch.json, else derived from the saved files."""
    w = live or _read_json(d / "watch.json")
    if isinstance(w, dict) and w.get("version") == 1:
        w = dict(w)
        status = w.get("status")
        if status in ENGINE_ACTIVE and (w.get("closed") or not pid_alive(w.get("pid"))):
            w["status"] = "stopped"                 # the server that ran it stopped (or the sweep was discarded) before End finished
            w["stopped_reason"] = "sweep closed before it was ended" if w.get("closed") else "the server running this sweep is no longer running"
        w.pop("pid", None)
        w["source_of_state"] = "live" if live else "watch.json"
        return w
    n, last = _frames_on_disk(d)
    finished = (d / "claim_packet.json").is_file()
    return {"version": 1, "run_id": d.name, "status": "finished" if finished else "incomplete", "frames": n, "latest_frame": last,
            "started_epoch": None, "ended_epoch": None, "duration_s": _legacy_duration(d), "live_count": None, "indicator": None,
            "notices": [], "finalize": None, "finalize_steps": [], "first_count": None, "source_of_state": "saved files (no watch.json: sweep predates the watch page)"}


def _activity(d: Path) -> float:
    best = 0.0
    for p in (d / "watch.json", d / "frames", d / "claim_packet.json", d):
        try:
            best = max(best, p.stat().st_mtime)
        except OSError:
            pass
    return best


def sweep_state(runs_dir: Path, run_id: str, *, live: dict[str, Any] | None = None,
                post_status: Callable[[str], dict[str, Any] | None] | None = None) -> dict[str, Any] | None:
    d = run_dir(runs_dir, run_id)
    if d is None:
        return None
    eng = engine_status(d, live)
    post = None
    if eng.get("status") == "finished" and post_status is not None:
        try:
            post = post_status(run_id)
        except Exception:  # noqa: BLE001 - the watch page never fails on a build-status read
            post = None
    post_active = bool(post) and post.get("state") in POST_ACTIVE
    status = eng.get("status")
    # a sweep page that is open but has not sent a frame yet (every page load creates a session) is "waiting", not recording
    phase = ("waiting" if status == "capturing" and not eng.get("frames") else "capturing" if status == "capturing" else "processing" if status == "finalizing" else "maps" if post_active
             else "done" if status == "finished" else status or "unknown")
    totals = packet_totals(d) if status == "finished" else None
    tp = d / "live" / TRANSCRIPT_FILE
    try:
        trev = None if tp.is_symlink() else tp.stat().st_size      # the page refetches the transcript / packet viewer only when these change
    except OSError:
        trev = None
    return {"run_id": run_id, "phase": phase, "active": status in ENGINE_ACTIVE or post_active, "engine": eng, "post_sweep": post,
            "packet": totals, "final": bool(totals) and not post_active, "report": (d / "report.html").is_file(),
            "packet_rev": _mtime(d / "claim_packet.json"), "transcript_rev": trev, "activity_epoch": _activity(d)}


def list_sweeps(runs_dir: Path, *, live: Callable[[str], dict[str, Any] | None] | None = None,
                post_status: Callable[[str], dict[str, Any] | None] | None = None, limit: int = 15) -> dict[str, Any]:
    """Recent sweeps (newest activity first) and the newest active one (the page follows it)."""
    root = Path(runs_dir)
    try:
        dirs = [p for p in root.iterdir() if RUN_RE.fullmatch(p.name) and p.is_dir()]
    except OSError:
        dirs = []
    dirs.sort(key=_activity, reverse=True)
    rows = []
    for d in dirs[:limit]:
        st = sweep_state(root, d.name, live=live(d.name) if live else None, post_status=post_status)
        if not st:
            continue
        eng = st["engine"]
        started = eng.get("started_epoch") or st["activity_epoch"]
        rows.append({"run_id": d.name, "phase": st["phase"], "active": st["active"], "frames": eng.get("frames"),
                     "started_epoch": started, "activity_epoch": st["activity_epoch"],
                     "book_count": (st["packet"] or {}).get("book_count") if st["packet"] else (eng.get("live_count") or {}).get("book_count")})
    active = [r for r in rows if r["active"]]
    # a sweep being recorded or ended wins over one whose maps are still building, which wins over an open page with no frames; then the most recently started
    newest = max(active, key=lambda r: (r["phase"] in ("capturing", "processing"), r["phase"] != "waiting", r["started_epoch"] or 0))["run_id"] if active else None
    return {"sweeps": rows, "newest_active": newest, "newest": rows[0]["run_id"] if rows else None}
