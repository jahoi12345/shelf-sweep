# Adapted from voice_ai_agents/insurance_claim_live_agent_team/live_demo/server.py
# (awesome-llm-apps @ 4bf51ab, Apache-2.0; see app/NOTICE.md). MODIFIED: the FastAPI app skeleton, local-only
# access middleware, owner-cookie sessions, transcript turn handling, WebSocket relay loops (client_to_gemini /
# gemini_to_client), task tracking, interruption handling, cleanup and ZIP download are kept; the insurance
# claim graph, policy lookup, evidence-photo verification, sketch generation and avatar code are removed and
# replaced by the incremental library-sweep engine (app/engine.py) and its tools.
"""FastAPI backend for the Library Contents Claim live sweep app.

Run:  .venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 4177
Open: http://127.0.0.1:4177

Works without a hosted-voice credential: frames are still saved and scored, nudges fire from code and local detection/OCR run when
configured. Hosted voice availability is reported independently of detector/reader availability (see per-session stage status).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import re
import secrets
import shutil
import sys
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import lan_access as LAN
from app import watch as WATCH
from app import ground_truth_page as GT
from app import live_tools as LT
from pipeline import sanitize as Z
from app import truth as TRUTH
from app import title_labels as TL
from app import pair_labels as PL
from providers import readers as READERS
from providers import stages as STG
from providers import hosted_gate as HG
from app import debug_server as DBG
from app.debug_registry import read_registry  # noqa: E402
from app import frame_browser as FB  # noqa: E402
from app.engine import StageFns, SweepEngine, SweepError
from app.measurement import auto_view, finalize_run, measured_runs
from app.post_sweep import PostSweepBuilds
from pipeline import panorama_attach as PANA
from app.measurement import MeasurementError  # noqa: E402
from app.replay import replay_into  # noqa: E402
from pipeline import corrections as K
from pipeline import contracts as C  # noqa: E402
from providers import gemini  # noqa: E402

gemini._load_repo_dotenv()   # <repo>/.env (never overrides real environment): credentials AND LIBRARY_* local-model settings
STATIC = Path(__file__).resolve().parent / "static"
RUNS_DIR = Path(os.getenv("LIBRARY_RUNS_DIR", REPO_ROOT / "runs"))   # session dirs are runs/sweep_<id> so the viewers discover them
# The ONE build queue (app/post_sweep.py, created in app/debug_server.py so the Debugger's Build buttons share it): after every finished sweep it builds the
# 2D panorama, then the 3D map; LIBRARY_POST_SWEEP_BUILD=0 turns the automatic part off (manual builds still queue).
post_sweep = DBG.build_queue
post_sweep.enabled = os.getenv("LIBRARY_POST_SWEEP_BUILD", "1") != "0"


def _measured(run_id, recon_out):
    out = finalize_run(RUNS_DIR, run_id, recon_out)     # automatic scale + spines + room attached to the packet (starts from its own backup of the packet) ...
    try:
        PANA.attach(RUNS_DIR / run_id)                  # ... so the panorama count is attached again on top (idempotent; no-op without panorama_count.json)
    except Exception:   # noqa: BLE001 - the measurement result stands
        logger.exception("re-attaching the panorama count after the measurement failed for %s", run_id)
    return out


post_sweep.on_built = _measured
if os.getenv("LIBRARY_PANORAMA_COUNT", "1") != "0":
    post_sweep.on_count = lambda run_id, pan_out: PANA.run_count_step(RUNS_DIR, run_id, pan_out)   # chain-independent count from the 2D panorama, before the 3D map
from pipeline import instances as INST
if os.getenv("LIBRARY_INSTANCES", "1") != "0":
    post_sweep.on_instances = lambda run_id, pan_out: INST.run_step(RUNS_DIR, run_id, pan_out)    # one packet line per physical book, after the count and after the measurement
if os.getenv("LIBRARY_FINISH_READING", "1") != "0":
    from app import finish_reading as FINISH
    # after the packet, read every attemptable group the packet's wall-clock budget left unread (same reader, hosted gate respected), resolve, price, update the packet
    post_sweep.finish_runner = lambda run_id, ctl: FINISH.run(RUNS_DIR / run_id, ctl)
post_sweep.finish_enabled = os.getenv("LIBRARY_FINISH_READING", "1") != "0"
REPLAY_DIR = Path(os.getenv("LIBRARY_REPLAY_DIR", REPO_ROOT / "sweeps"))
SAVED_RUNS_DIR = REPO_ROOT / "runs"
REPLAY_SUFFIXES = {".mov", ".mp4", ".m4v", ".webm"}
logger = logging.getLogger(__name__)

SESSION_TTL = 30 * 60
MAX_SESSIONS = 8
MAX_MESSAGE_BYTES = 14_000_000      # a 3840x2160 evidence frame is ~2-4 MB of JPEG (base64 x1.34)
MAX_EVIDENCE_BYTES = 10_000_000
GENAI_CLIENT = None
_live_client_factory = None  # tests inject a fake Live client


def _cors_origins() -> list[str]:
    raw = os.getenv("LIBRARY_CORS_ORIGINS", "")
    if raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://127.0.0.1:4177", "http://localhost:4177"]


def _has_api_key() -> bool:
    return gemini.credential_available()


def _live_client():
    """Real google-genai client for Gemini Live, or the test-injected fake."""
    global GENAI_CLIENT
    if _live_client_factory is not None:
        return _live_client_factory()
    if GENAI_CLIENT is None:
        from google import genai
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        GENAI_CLIENT = genai.Client(api_key=key)
    return GENAI_CLIENT


def set_live_client_factory(fn) -> None:
    global _live_client_factory
    _live_client_factory = fn


# ------------------------------------------------------------------------------------------- sessions
@dataclass
class SweepSession:
    session_id: str
    owner: str
    run_dir: Path
    engine: SweepEngine
    transcript: list[dict[str, str]] = field(default_factory=list)
    tool_activity: list[dict[str, Any]] = field(default_factory=list)
    camera_enabled: bool = False
    live_socket: Any = None
    tasks: set = field(default_factory=set)
    replay_task: Any = None
    updated_at: float = field(default_factory=time.monotonic)
    deleted: bool = False
    revision: int = 0
    live_model: str | None = None
    voice_enabled: bool = False


sessions: dict[str, SweepSession] = {}
stage_fns_factory = StageFns.real  # tests may replace


def append_turn(session: SweepSession, speaker: str, text: str, turn_id: str | None = None) -> str:
    text = text.strip()[:8000]
    turn_id = turn_id or uuid.uuid4().hex
    if any(t.get("id") == turn_id for t in session.transcript):
        return turn_id
    if len(session.transcript) >= 400 or sum(len(t["text"]) for t in session.transcript) + len(text) > 80000:
        raise ValueError("This sweep reached its conversation limit. Download the packet and start a new sweep.")
    session.transcript.append({"id": turn_id, "speaker": speaker, "text": text})
    session.updated_at = time.monotonic()
    _persist_turn(session, turn_id, speaker, text)
    return turn_id


def _persist_turn(session: SweepSession, turn_id: str, speaker: str, text: str) -> None:
    """Every finished turn of the conversation (what the person said, what the voice agent said) is appended to <run>/live/transcript.jsonl,
    so the loopback /watch page (another process) shows it live and it is kept after the sweep. Server sessions only (append_turn is the only
    writer); text only, never audio; the turn id is kept for de-duplication but the watch page does not return it."""
    try:
        live = session.run_dir / "live"
        live.mkdir(parents=True, exist_ok=True)
        now = time.time()
        started = getattr(session.engine, "started_epoch", None)
        rec = {"id": turn_id, "speaker": speaker, "text": text, "at_epoch": round(now, 3),
               "at_s": round(now - started, 2) if isinstance(started, (int, float)) else None}
        with open(live / WATCH.TRANSCRIPT_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - the saved transcript must never break the conversation
        logger.debug("transcript append failed", exc_info=True)


LIVE_LIST_CAP = 40   # rows per list in a state push while the camera is capturing (see _cap_live_lists)


def _cap_live_lists(snap: dict[str, Any]) -> dict[str, Any]:
    """While the camera is on, push only the newest rows of the long lists (+ their totals in ``list_caps``): the full snapshot of a 750-group sweep is ~3 MB,
    pushed up to 4x a second to a phone that is also capturing and uploading frames (YugoTest5/6: client capture fell from 4.3 to 1.5 fps as the list grew;
    the parse + re-render alone saturated the page's main thread). The full lists come back the moment the camera stops or the sweep ends."""
    caps: dict[str, Any] = {}
    snap = dict(snap)
    for key, newest in (("books", True), ("items", True), ("provisional_unmerged", True), ("review_queue", False)):
        rows = snap.get(key) or []
        caps[key] = {"shown": min(len(rows), LIVE_LIST_CAP), "total": len(rows)}
        if len(rows) > LIVE_LIST_CAP:
            snap[key] = rows[-LIVE_LIST_CAP:] if newest else rows[:LIVE_LIST_CAP]
        if key in ("books", "items"):   # dedup_reasons is two thirds of a row (~1.5 KB of 2.2 KB) and no live view reads it; it stays in the packet and the Debugger
            snap[key] = [{k: v for k, v in r.items() if k != "dedup_reasons"} for r in snap[key]]
    snap["list_caps"] = caps
    return snap


def _ui_state(session: SweepSession, snap: dict[str, Any] | None = None) -> dict[str, Any]:
    snap = snap if snap is not None else session.engine.snapshot()
    if snap.get("status") == "capturing" and session.camera_enabled:
        snap = _cap_live_lists(snap)
    return {**snap, "post_sweep": post_sweep.status(session.run_dir.name) if snap.get("status") == "finished" else None,
            "transcript": session.transcript, "tool_activity": session.tool_activity[-12:],
            "live_model": session.live_model, "voice_enabled": session.voice_enabled,
            "camera_enabled": session.camera_enabled}


app = FastAPI(title="Library Contents Claim Live Sweep API")
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins(), allow_methods=["*"], allow_headers=["*"])


def allowed_origin(origin: str | None, host: str, scheme="http") -> bool:
    if not origin:
        return False
    return origin in set(_cors_origins()) | {f"{scheme}://{host}"}


def local_host(host: str) -> bool:
    return LAN.host_ok(host)    # localhost / 127.0.0.1; plus the LAN names only when scripts/serve_https.sh set LIBRARY_LAN_MODE (app/lan_access.py)


def is_watch_path(path: str) -> bool:
    """Loopback-only pages: the watch page and the ground-truth labelling page."""
    return (path.rstrip("/") in ("/watch", "/watch.js", "/watch.css", "/truth", "/truth.js", "/truth.css")
            or path.startswith(("/api/watch", "/api/truth")))


def watch_allowed(request: Request) -> bool:
    """The laptop watch page exists only on the loopback server (:4182): never in LAN mode (:4183, even from the Mac itself), only for a
    loopback client and a loopback Host name. Everything else gets 404 (the page does not exist there)."""
    client = request.client.host if request.client else None
    return (not LAN.enabled() and client in LAN.LOOPBACK_CLIENTS
            and request.headers.get("host", "").split(":")[0].strip().lower() in LAN.LOOPBACK_HOSTS)


@app.middleware("http")
async def local_access(request: Request, call_next):
    # Local-only demo (kept from the reference): public serving needs a separate authenticated design. LAN mode (phone test, :4183) adds a Host
    # allowlist, private-range clients and a pairing key for every non-loopback client.
    if is_watch_path(request.url.path) and not watch_allowed(request):
        return Response("Not Found", status_code=404)
    if not local_host(request.headers.get("host", "")) or not LAN.client_ok(request.client.host):
        return Response("This demo accepts local connections only.", status_code=403)
    if LAN.needs_key(request.client.host) and request.url.path != LAN.CA_PATH:
        if LAN.key_ok(request.cookies.get(LAN.COOKIE)):
            pass
        elif request.method == "GET" and LAN.key_ok(request.query_params.get("k")):   # the pairing URL: set the cookie, drop the key from the address bar
            clean = "&".join(f"{k}={v}" for k, v in request.query_params.multi_items() if k != "k")
            response = RedirectResponse(request.url.path + (f"?{clean}" if clean else ""), status_code=303)
            response.set_cookie(LAN.COOKIE, request.query_params["k"], httponly=True, samesite="lax", max_age=30 * 24 * 3600, secure=request.url.scheme == "https")
            return response
        else:
            return Response("Open the pairing URL printed by scripts/serve_https.sh (it carries ?k=...).", status_code=403)
    origin = request.headers.get("origin")
    if (origin and not allowed_origin(origin, request.headers.get("host", ""), request.url.scheme)) or (
            request.method not in {"GET", "HEAD", "OPTIONS"} and not origin):
        return Response("Origin not allowed", status_code=403)
    try:
        length = int(request.headers.get("content-length", 0) or 0)
    except ValueError:
        return Response("Invalid Content-Length", status_code=400)
    if request.url.path.startswith("/api") and length > 16000:
        return Response("Request too large", status_code=413)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


async def discard_session(session: SweepSession, purge: bool = False) -> None:
    """Cancel tasks and release the engine. Evidence on disk is kept unless ``purge``."""
    session.deleted = True
    sessions.pop(session.session_id, None)
    if session.live_socket:
        with contextlib.suppress(Exception):
            await session.live_socket.close(code=1000)
    if session.replay_task:
        session.replay_task.cancel()
        await asyncio.gather(session.replay_task, return_exceptions=True)
    for task in list(session.tasks):
        task.cancel()
    if session.tasks:
        await asyncio.gather(*list(session.tasks), return_exceptions=True)
    await session.engine.aclose()
    if purge:
        shutil.rmtree(session.run_dir, ignore_errors=True)


detached: dict[str, dict[str, Any]] = {}     # sweeps the user moved on from (New sweep, idle timeout) while they were still recording/finalizing: they keep running until the packet is written


def sweep_processing(session: SweepSession) -> dict[str, Any]:
    """What is still running for this sweep: drives the New-sweep confirm and decides whether a delete DETACHES the sweep instead of closing it.
    {processing: False} means nothing would be lost; otherwise phase is capturing (frames saved, never ended), finalizing (packet being written) or maps (2D/3D builds)."""
    eng, run_id = session.engine, session.run_dir.name
    frames = len(eng.frames)
    base = {"processing": False, "run_id": run_id, "phase": eng.status, "frames": frames}
    keep = f"Nothing is deleted: it stays under runs/{run_id}. Start a new sweep?"
    if eng.status == "capturing" and frames:
        return {**base, "processing": True, "phase": "capturing", "detail": f"{frames} frames saved, the sweep has not been ended",
                "message": f"This sweep is still being recorded ({frames} frames saved, not ended). New sweep will end it and finish its packet and 2D/3D maps in the background. {keep}"}
    if eng.status == "finalizing":
        stage = eng.finalize_stage or "working"
        return {**base, "processing": True, "phase": "finalizing", "detail": f"packet in progress: {stage}",
                "message": f"This sweep's packet is still being written ({stage}). It will finish in the background and its 2D/3D maps will be built after it. {keep}"}
    if eng.status == "finished":
        ps = post_sweep.status(run_id)
        if ps and post_sweep.is_active(run_id):
            return {**base, "processing": True, "phase": "maps", "detail": ps.get("summary") or ps.get("note") or ps["state"],
                    "message": f"This sweep's maps are still building ({ps.get('summary') or ps['state']}). They keep building in the background. {keep}"}
    return base


async def detach_session(session: SweepSession) -> dict[str, Any]:
    """Let a recording/finalizing sweep finish without its browser session: the client side (Live socket, replay, relay tasks) is closed exactly like a discard, but the engine is NOT
    closed. A sweep that was never ended is ended here; the engine's own end_sweep task (shielded from caller cancellation) writes the packet and report and then enqueues the
    2D/3D builds through on_finished. The engine is closed only after that. Evidence on disk is never purged."""
    info = sweep_processing(session)
    session.deleted = True
    sessions.pop(session.session_id, None)
    if session.live_socket:
        with contextlib.suppress(Exception):
            await session.live_socket.close(code=1000)
    if session.replay_task:
        session.replay_task.cancel()
        await asyncio.gather(session.replay_task, return_exceptions=True)
    for task in list(session.tasks):
        task.cancel()
    if session.tasks:
        await asyncio.gather(*list(session.tasks), return_exceptions=True)
    eng, sid, run_id = session.engine, session.session_id, session.run_dir.name

    async def finish() -> None:
        try:
            if eng.status == "capturing":
                await eng.end_sweep()
            else:
                await eng.wait_ended()
        except Exception:  # noqa: BLE001 - recorded in the engine status (failed + finalize error); the frames stay under runs/
            logger.exception("background end of sweep %s failed", run_id)
        finally:
            detached.pop(sid, None)
            with contextlib.suppress(Exception):
                await eng.aclose()

    detached[sid] = {"session": session, "run_id": run_id, "since": time.time(), "task": asyncio.ensure_future(finish())}
    return {"run_id": run_id, "phase": info["phase"], "detail": info.get("detail")}


async def release_session(session: SweepSession, purge: bool = False) -> dict[str, Any]:
    """New sweep / idle timeout: never lose a sweep. Recording or finalizing -> detach (finishes in the background); maps still building -> the build queue runs on its own, the session
    is just closed; otherwise it is discarded as before. ``purge`` is honoured only when nothing is processing."""
    info = sweep_processing(session)
    if info["processing"] and info["phase"] in ("capturing", "finalizing"):
        return {"detached": True, "continues": info["phase"], **await detach_session(session)}
    await discard_session(session, purge=purge and not info["processing"])
    return {"detached": False, "continues": "maps" if info["processing"] else None, "run_id": info["run_id"]}


async def cleanup_sessions() -> None:
    for session in list(sessions.values()):
        if time.monotonic() - session.updated_at > SESSION_TTL:
            await release_session(session)


def owned_session(session_id: str, owner: str | None) -> SweepSession:
    session = sessions.get(session_id)
    if not session or session.deleted or not owner or not secrets.compare_digest(session.owner, owner):
        raise HTTPException(404, "Sweep not found or expired. Start a new sweep.")
    if time.monotonic() - session.updated_at > SESSION_TTL:
        raise HTTPException(410, "Sweep expired. Start a new sweep.")
    session.updated_at = time.monotonic()
    return session


def _new_engine(session_id: str, run_dir: Path, session_ref: list) -> SweepEngine:
    async def emit(event: dict[str, Any]) -> None:
        sess = session_ref[0] if session_ref else None
        if sess is None or sess.deleted:
            return
        hook = getattr(sess, "_emit_to_client", None)
        if hook:
            await hook(event)
    return SweepEngine(run_dir, session_id=session_id, stages=stage_fns_factory(), emit=emit,
                       on_finished=lambda finished_dir: post_sweep.enqueue(Path(finished_dir).name),
                       config={"run_id": f"sweep_{session_id}", "country": os.getenv("LIBRARY_COUNTRY", "US"),
                               "currency": os.getenv("LIBRARY_CURRENCY", "USD"),
                               "catalogue_online": os.getenv("LIBRARY_CATALOGUE_ONLINE", "") == "1",
                               "pricing_online": os.getenv("LIBRARY_PRICING_ONLINE", "") == "1",
                               "max_books": int(os.environ["LIBRARY_OCR_MAX_BOOKS"]) if "LIBRARY_OCR_MAX_BOOKS" in os.environ else None,
                               "cache_dir": str(RUNS_DIR / ".cache"), "watch_file": True})   # <run>/watch.json for the loopback /watch page


def voice_status() -> tuple[bool, str | None]:
    """Hosted voice availability and, when unavailable, the exact reason. Never guesses a model id."""
    if _live_client_factory is not None:
        return True, None     # tests inject a fake Live client
    if not _has_api_key():
        return False, "no hosted credential: set GOOGLE_API_KEY (or GEMINI_API_KEY) in the environment or the repo .env"
    if HG.disabled():
        return False, HG.HostedDisabledError.status + ": press Reconnect in the header to allow hosted calls again"
    if not LT.LIVE_MODEL_ID:
        return False, ("LIBRARY_LIVE_MODEL is not set; no Live model id is assumed. Set the model you have access to, "
                       "then verify it with `python -m app.smoke live`")
    return True, None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "has_api_key": _has_api_key(), "live_model": LT.LIVE_MODEL_ID, "tools": LT.TOOL_NAMES,
            "detect_model_default": gemini.default_model(), "detector_selection": os.getenv("LIBRARY_DETECTOR", "auto"),
            "reader_selection": os.getenv("LIBRARY_READER", "auto"), "reader_effective": READERS.effective_reader(), "reader_plugins": READERS.reader_names(), "hosted_allowed": STG.hosted_allowed(), "hosted_switch": {k: HG.summary()[k] for k in ("disabled", "status", "calls", "blocked", "tokens", "cost_usd")},
            "hosted_note": "auto never selects the billable Gemini adapter unless LIBRARY_ALLOW_HOSTED=1 or LIBRARY_DETECTOR/LIBRARY_READER=gemini explicitly", "market_default": {"country": "US", "currency": "USD"},
            "replay_videos": sorted((p.name for p in REPLAY_DIR.glob("*") if p.suffix.lower() in REPLAY_SUFFIXES), key=lambda n: (Path(n).suffix.lower() in {".mov"}, n)) if REPLAY_DIR.is_dir() else [],
            "voice_available": voice_status()[0], "voice_reason": voice_status()[1], "build": {k: v for k, v in build_info().items() if k in ("pid", "started_at", "backend_fingerprint_loaded", "backend_source_changed_since_start", "canonical_url")}, "voice_note": "hosted Gemini voice only; detector/reader availability is per session",
            "note": "Local detection and OCR are configured independently from Gemini voice. See per-session stage availability."}


@app.post("/api/sessions")
async def create_session(request: Request, response: Response) -> dict[str, Any]:
    await cleanup_sessions()
    owner = request.cookies.get("intake_owner") or secrets.token_urlsafe(32)
    if len(sessions) >= MAX_SESSIONS or sum(s.owner == owner for s in sessions.values()) >= 3:
        raise HTTPException(429, "Too many sweeps. Close or reset an existing sweep first.")
    sid = uuid.uuid4().hex[:16]
    run_dir = RUNS_DIR / f"sweep_{sid}"
    ref: list = []
    engine = _new_engine(sid, run_dir, ref)
    session = SweepSession(session_id=sid, owner=owner, run_dir=run_dir, engine=engine)
    ref.append(session)
    append_turn(session, "Agent", "Hi! I'll help document this room for your contents claim. We'll use the United States and US dollars unless you tell me otherwise. Please make one slow, continuous pass along every shelf and wall: keep the same distance, don't zoom, and try not to go back over areas you've already covered. I'll tell you what I'm logging.")
    sessions[sid] = session
    response.set_cookie("intake_owner", owner, httponly=True, samesite="strict", max_age=SESSION_TTL, secure=request.url.scheme == "https")
    return {"session_id": sid, "has_api_key": _has_api_key(), "state": _ui_state(session)}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    return {"session_id": session.session_id, "state": _ui_state(session), "has_api_key": _has_api_key()}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request, purge: int = 0):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    released = await release_session(session, purge=bool(purge))
    return {"deleted": True, "purged": bool(purge) and not released["detached"] and released["continues"] is None, **released}


@app.get("/api/sessions/{session_id}/processing")
def session_processing(session_id: str, request: Request):
    """Is this sweep still recording/finalizing/building maps? The New-sweep button asks before it leaves a sweep that is not done (it then finishes in the background)."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    return sweep_processing(session)


@app.get("/api/background-sweeps")
def background_sweeps():
    """Sweeps that were moved on from while still processing, with what they are doing now (they keep running; every sweep stays under runs/)."""
    out = []
    for entry in list(detached.values()):
        eng = entry["session"].engine
        out.append({"run_id": entry["run_id"], "status": eng.status, "stage": eng.finalize_stage, "error": eng.finalize_error, "since": entry["since"],
                    "post_sweep": post_sweep.status(entry["run_id"]) if eng.status == "finished" else None})
    return {"sweeps": out}


@app.post("/api/sessions/{session_id}/instances/split")
async def split_instance(session_id: str, request: Request):
    """Undo one merge of the physical-instance layer (pipeline.instances): the member track group named in the body becomes its own book line again. The merge
    only exists after the 2D panorama, i.e. on a finished sweep: the `split` entry goes into the corrections ledger (live state) and the saved packet is rebuilt in place."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    engine = session.engine
    body = await request.json()
    member = str((body or {}).get("member_group_id") or "").strip()
    if not member:
        raise HTTPException(400, "member_group_id required")
    try:
        obs = K.resolve_instance_targets((engine.inventory or {}).get("data") or {}, [member])
        engine._corr_seq += 1
        entry = K.new_entry(f"c{engine._corr_seq:03d}", "split", {"observation_ids": sorted(set(obs))}, {"member_group_id": member},
                            {"actor": "claimant", "utterance": str((body or {}).get("reason") or "")[:500]})
    except K.CorrectionError as exc:
        raise HTTPException(400, str(exc)) from exc
    engine.ledger.append(entry)
    engine._bump()
    attach = {"attached": False, "reason": "sweep not finished: the split applies once the instance layer exists"}
    if engine.status == "finished":
        attach = await asyncio.to_thread(INST.attach, engine.run_dir, None, extra_ledger=[entry])
    await engine._publish_state()
    return {"correction_id": entry["id"], "member_group_id": member, "attach": attach, "state": _ui_state(session)}


@app.get("/api/sessions/{session_id}/builds")
def session_builds(session_id: str, request: Request):
    """Status of the automatic post-sweep 2D panorama / 3D map builds of this sweep (None until the sweep has finished)."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    return {"run_id": session.run_dir.name, "post_sweep": post_sweep.status(session.run_dir.name), "enabled": post_sweep.enabled}


@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    try:
        summary = await session.engine.end_sweep()
    except SweepError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"summary": summary, "state": _ui_state(session)}


def _packet_files(session: SweepSession) -> list[tuple[str, Path]]:
    files = []
    for name in ("claim_packet.json", "report.html", "manifest.json", "candidates.json", "readings.json", "resolutions.json", "inventory.json", "prices.json",
                 "scale.json", "room.json", "calibration.json", "room_evidence.json", "evidence_index.json", "report.pdf"):
        p = session.run_dir / name
        if p.is_file():
            files.append((name, p))
    files += [(f"frames/{p.name}", p) for p in sorted((session.run_dir / "frames").glob("*.jpg"))]
    for folder in ('crops', 'detection_responses', 'pricing_evidence', 'ocr_responses', 'live', 'logs'):
        root = session.run_dir / folder
        if root.is_dir():
            files += [(str(p.relative_to(session.run_dir)), p) for p in root.rglob('*') if p.is_file() and not p.is_symlink()]
    return files


@app.get("/api/sessions/{session_id}/packet")
def download_packet(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    if session.engine.status != "finished":
        raise HTTPException(409, "End the sweep first: the packet is built when the sweep ends.")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        audit = []
        for arc, path in _packet_files(session):
            if path.suffix.lower() in (".jpg", ".jpeg"):   # shipped copies never carry EXIF/GPS; local originals stay untouched
                try:
                    data, rec = Z.sanitised_jpeg_for_bundle(path)
                except Z.SanitizeError as exc:
                    audit.append({"file": arc, "error": str(exc), "shipped": False})
                    continue
                archive.writestr(arc, data)
                audit.append({**rec, "file": arc})
            else:
                archive.write(path, arc)
        archive.writestr("privacy_audit.json", json.dumps({
            "policy": "no original video is ever bundled; JPEG frames are shipped with EXIF/XMP/IPTC/comment segments removed (lossless); "
                      "local originals are not modified", "video_files_bundled": 0, "images": audit,
            "images_with_metadata_removed": sum(bool(a.get("metadata_removed")) for a in audit),
            "images_still_with_metadata": sum(bool(a.get("metadata_after_any")) for a in audit)}, indent=2))
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="library-claim-packet.zip"'})


_FRAME_RE = re.compile(r"^f\d{5}\.jpg$")


@app.get("/api/sessions/{session_id}/frame-index")
def session_frame_index(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    frames = FB.index_frames(session.engine.frames, f"/api/sessions/{session_id}/frames")
    return {"run_id": session_id, "video_file": "Current sweep", "fixture": False,
            "frames": frames, "frame_count": len(frames),
            "keyframe_count": sum(f["selected"] for f in frames)}


ANNOTATION_DIR = Path(os.getenv("LIBRARY_ANNOTATION_DIR") or REPO_ROOT / "runs/annotations_http")   # a bundle written by eval.templates.http_bundle

@app.get("/annotations")
def annotations():
    path = ANNOTATION_DIR / 'index.html'
    if not path.is_file() or path.is_symlink():
        raise HTTPException(404, "Annotation artifact not published")
    content = path.read_text(encoding='utf-8').replace('annotation_payload.json', '/annotations/annotation_payload.json').replace('evidence/f', '/annotations/evidence/f')
    return Response(content, media_type='text/html', headers={'Cache-Control':'no-store'})


@app.get('/annotations/annotation_payload.json')
def annotation_payload():
    path=ANNOTATION_DIR/'annotation_payload.json'
    if not path.is_file() or path.is_symlink():raise HTTPException(404, 'No annotation payload')
    payload=json.loads(path.read_text())
    payload['images']={fid:'/annotations/evidence/'+Path(url).name for fid,url in payload.get('images',{}).items()}
    return payload


@app.get('/annotations/evidence/{name}')
def annotation_evidence(name: str):
    if not _FRAME_RE.fullmatch(name):raise HTTPException(404,'No such annotation frame')
    bundle=C.read_json(ANNOTATION_DIR/'bundle.json')
    entry=next((f for f in bundle.get('evidence_frames',[]) if f.get('url')=='evidence/'+name),None)
    path=ANNOTATION_DIR/'evidence'/name
    import hashlib
    if not entry or path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry.get('sha256'):
        raise HTTPException(404,'No validated annotation frame')
    return FileResponse(path, media_type='image/jpeg')


@app.get("/api/sessions/{session_id}/crops/{name}")
def get_crop(session_id: str, name: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    path = (session.run_dir / "crops" / name).resolve()
    allowed = {r.get('crop_path') for r in session.engine.readings}
    if path.parent != (session.run_dir / 'crops').resolve() or not path.is_file() or path.is_symlink() or str(path.relative_to(session.run_dir)) not in allowed:
        raise HTTPException(404, "No such evidence crop")
    return FileResponse(path, media_type="image/png")


@app.get("/api/frame-runs")
def saved_frame_runs(all: int = 0):   # all=1 also lists non-featured runs (app.run_visibility)
    return {"runs": FB.list_saved_runs(SAVED_RUNS_DIR, show_all=bool(all))}


@app.get("/api/frame-runs/{run_id}")
def saved_frame_index(run_id: str):
    try:
        return FB.saved_index(SAVED_RUNS_DIR, run_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/frame-runs/{run_id}/frames/{name}")
def saved_frame_image(run_id: str, name: str):
    if not _FRAME_RE.fullmatch(name):
        raise HTTPException(404, "No such frame")
    try:
        run, envelope = FB.saved_run(SAVED_RUNS_DIR, run_id)
        fid = name[:-4]
        if not any(f.get("frame_id") == fid and f.get("image_path") == f"frames/{name}"
                   for f in envelope["data"]["frames"]):
            raise ValueError("No such frame in the manifest")
        path = FB.evidence_path(run, fid)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/sessions/{session_id}/frames/{name}")
def get_frame(session_id: str, name: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    if not _FRAME_RE.match(name):
        raise HTTPException(404, "No such frame")
    path = session.run_dir / "frames" / name
    if not path.is_file():
        raise HTTPException(404, "No such frame")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/sessions/{session_id}/report")
def get_report(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    path = session.run_dir / "report.html"
    if not path.is_file():
        raise HTTPException(409, "Report is generated when the sweep ends.")
    return FileResponse(path, media_type="text/html")


# ------------------------------------------------------------------------------- measurement inputs (human-supplied)
async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "JSON body required") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON object required")
    return body


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")


@app.get("/api/auto-measurements")
def auto_measurement_runs():
    """Saved runs with an automatic measurement (newest first), for the Measurements tab."""
    return {"runs": measured_runs(RUNS_DIR)}


@app.get("/api/runs/{run_id}/auto-measurement")
def auto_measurement(run_id: str):
    """Automatic spine sizes (+-, scale source), shelf run and room block of a saved run; unknown values are null with a plain-words reason."""
    if not _RUN_ID_RE.match(run_id) or ".." in run_id:
        raise HTTPException(400, "bad run id")
    if not (RUNS_DIR / run_id).is_dir():
        raise HTTPException(404, "No such run")
    return auto_view(RUNS_DIR, run_id)


@app.get("/api/sessions/{session_id}/measurement")
def measurement_status(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    return session.engine.measurement_status()


@app.post("/api/sessions/{session_id}/calibrations")
async def post_calibration(session_id: str, request: Request):
    """Attributable calibration: image points the person tapped on a saved keyframe + the dimension(s) they stated."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    b = await _json_body(request)
    kind = b.get("kind")
    unit = str(b.get("unit", "m"))
    try:
        from app.measurement import to_metres
        metric: dict[str, Any] = {"tolerance_pct": float(b.get("tolerance_pct", 3.0))}
        if kind == "plane_quad":
            metric.update(width_m=to_metres(float(b["width"]), unit), height_m=to_metres(float(b["height"]), unit))
        elif kind == "segment":
            metric.update(length_m=to_metres(float(b["length"]), unit))
        else:
            raise MeasurementError("kind must be plane_quad or segment")
        res = await session.engine.add_calibration(
            kind=kind, frame_id=str(b.get("frame_id", "")), image_points_px=b.get("image_points_px"), metric=metric,
            plane_id=b.get("plane_id") or None, role=str(b.get("role", "shelf_face")), utterance=(str(b.get("utterance"))[:500] if b.get("utterance") else None),
            note=(str(b.get("note"))[:500] if b.get("note") else None), assume_fronto_parallel=bool(b.get("assume_fronto_parallel", False)))
    except (MeasurementError, SweepError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"calibration rejected: {exc}") from exc
    return {**res, "measurement": session.engine.measurement_status()}


@app.delete("/api/sessions/{session_id}/calibrations/{calibration_id}")
async def delete_calibration(session_id: str, calibration_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    try:
        await session.engine.remove_calibration(calibration_id)
    except MeasurementError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"deleted": calibration_id}


@app.post("/api/sessions/{session_id}/room-evidence/walls")
async def post_wall(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    b = await _json_body(request)
    try:
        w = await session.engine.add_wall(frame_id=str(b.get("frame_id", "")), quad_px=b.get("quad_px"), wall_id=(str(b["wall_id"])[:4] if b.get("wall_id") else None),
                                          spans_floor_to_ceiling=bool(b.get("spans_floor_to_ceiling", False)), note=(str(b.get("note"))[:300] if b.get("note") else None))
    except (MeasurementError, SweepError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"wall rejected: {exc}") from exc
    return {"wall": w, "measurement": session.engine.measurement_status()}


@app.delete("/api/sessions/{session_id}/room-evidence/walls/{wall_id}")
async def delete_wall(session_id: str, wall_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    try:
        await session.engine.remove_wall(wall_id)
    except MeasurementError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"deleted": wall_id}


@app.post("/api/sessions/{session_id}/room-evidence/height")
async def post_height(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    b = await _json_body(request)
    try:
        res = await session.engine.set_dimension("ceiling_height", float(b["value"]), str(b.get("unit", "m")), approximate=bool(b.get("approximate", True)),
                                                 utterance=(str(b.get("utterance"))[:500] if b.get("utterance") else None))
    except (MeasurementError, SweepError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"height rejected: {exc}") from exc
    return res


# ------------------------------------------------------------------------------- held-out tape truth (evaluation only)
def _truth_scan(session) -> dict[str, Any] | None:
    sc = session.engine.scan
    return {k: sc.get(k) for k in ("label", "sha256", "file")} if sc else None


@app.get("/api/sessions/{session_id}/truth")
def get_truth(session_id: str, request: Request):
    """The human's own entries for display/editing in the Measurements tab. Never read by the pipeline, engine or bundle."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    doc = TRUTH.load(session.session_id)
    return {"purpose": doc["purpose"], "annotator": doc["annotator"], "spine_samples": doc["spine_samples"], "room": doc["room"],
            "spine_target": 20, "saved_to": f"eval/truth_capture/{session.session_id}.json"}


@app.post("/api/sessions/{session_id}/truth/spines")
async def post_truth_spine(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    b = await _json_body(request)
    frame = next((f for f in session.engine.frames if f["frame_id"] == b.get("frame_id")), None)
    if frame is None:
        raise HTTPException(400, "unknown frame for this session")
    try:
        s = TRUTH.add_spine(session.session_id, frame=frame, scan=_truth_scan(session), tap_px=b.get("tap_px"), height=b.get("height"),
                            thickness=b.get("thickness"), unit=str(b.get("unit", "cm")), annotator=str(b.get("annotator", "")),
                            description=str(b.get("description", "")), language=b.get("language") or None)
    except TRUTH.TruthError as exc:
        raise HTTPException(400, f"tape reading rejected: {exc}") from exc
    return {"sample": s, "count": len(TRUTH.load(session.session_id)["spine_samples"])}


@app.delete("/api/sessions/{session_id}/truth/spines/{sample_id}")
def delete_truth_spine(session_id: str, sample_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    try:
        TRUTH.remove_spine(session.session_id, sample_id)
    except TRUTH.TruthError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"deleted": sample_id}


@app.post("/api/sessions/{session_id}/truth/room")
async def post_truth_room(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    b = await _json_body(request)
    try:
        room = TRUTH.set_room(session.session_id, length=b.get("length"), width=b.get("width"), height=b.get("height"), unit=str(b.get("unit", "m")),
                              annotator=str(b.get("annotator", "")), notes=str(b.get("notes", "")), scan=_truth_scan(session))
    except TRUTH.TruthError as exc:
        raise HTTPException(400, f"room reading rejected: {exc}") from exc
    return {"room": room}


# ------------------------------------------------------------------------------- title labelling (human truth for spine crops)
# ------------------------------------------------------------------------------- hosted kill-switch (Disconnect Gemini / Reconnect)
def _hosted_state() -> dict[str, Any]:
    live = [s.session_id for s in sessions.values() if s.live_socket is not None and not s.deleted]
    return {**HG.summary(), "live_sessions_open": live, "hosted_allowed_for_auto": STG.hosted_allowed()}


@app.get("/api/hosted")
def hosted_state():
    return _hosted_state()


@app.post("/api/hosted/disable")
async def hosted_disable(request: Request):
    # (a) close every live voice session now; (b) persisted server-side block; (c)+(d) state and counters live in the state file
    HG.set_disabled(True, "user:site")
    closed = 0
    for s in list(sessions.values()):
        ws = s.live_socket
        if ws is not None and not s.deleted:
            try:
                await ws.send_json({"type": "error", "message": "Hosted calls disconnected by the user (Disconnect Gemini). Local capture continues; voice is off until Reconnect."})
                await ws.close(code=1000)
            except Exception:  # noqa: BLE001 - the socket may already be gone
                pass
            s.live_socket = None
            closed += 1
    if closed:
        HG.note_live_closed(closed)
    return _hosted_state()


@app.post("/api/hosted/enable")
async def hosted_enable(request: Request):
    HG.set_disabled(False, "user:site")
    return _hosted_state()


@app.get("/api/label/runs")
def label_runs():
    return {"runs": TL.list_runs(), "defaults": TL.PREFERRED_DEFAULTS, "n_default": TL.DEFAULT_N, "seed_default": TL.DEFAULT_SEED}


@app.post("/api/label/session")
async def label_open(request: Request):
    b = await _json_body(request)
    try:
        return TL.open_session(str(b.get("run_id", "")), str(b.get("annotator", "")), int(b.get("n", TL.DEFAULT_N)), int(b.get("seed", TL.DEFAULT_SEED)))
    except (TL.LabelError, TypeError, ValueError, OSError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/label/{label_id}")
def label_state(label_id: str):
    try:
        return TL.state(label_id)
    except (TL.LabelError, OSError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/label/{label_id}/answer")
async def label_answer(label_id: str, request: Request):
    b = await _json_body(request)
    try:
        return TL.answer(label_id, str(b.get("observation_id", "")), str(b.get("status", "")), title=str(b.get("title", "")), author=str(b.get("author", "")),
                         language=b.get("language"), annotator=b.get("annotator"), rotation_seen=int(b.get("rotation_seen", 0) or 0),
                         prefill=b.get("prefill") if isinstance(b.get("prefill"), dict) else None)
    except (TL.LabelError, OSError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/label/{label_id}/undo")
def label_undo(label_id: str):
    try:
        return TL.undo(label_id)
    except (TL.LabelError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/label/{label_id}/crop/{index}")
def label_crop(label_id: str, index: int, view: str = "preview"):
    """view=preview (default): the crop as the readers see it (in-memory upscale+pad); view=native: the exact saved crop file."""
    try:
        if view == "native":
            return FileResponse(TL.crop_file(label_id, index), media_type="image/png")
        return Response(TL.crop_preview_png(label_id, index), media_type="image/png", headers={"Cache-Control": "no-store"})
    except (TL.LabelError, OSError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/label/{label_id}/context/{index}")
def label_context(label_id: str, index: int):
    try:
        return Response(TL.context_png(label_id, index), media_type="image/png", headers={"Cache-Control": "no-store"})
    except (TL.LabelError, OSError, KeyError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/label/{label_id}/sighting/{index}/{k}")
def label_sighting(label_id: str, index: int, k: int):
    try:
        return Response(TL.sighting_png(label_id, index, k), media_type="image/png", headers={"Cache-Control": "no-store"})
    except (TL.LabelError, OSError, KeyError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/labels.js")
def labels_script():
    return FileResponse(STATIC / "labels.js", media_type="text/javascript")


@app.get("/pairs.js")
def pairs_script():
    return FileResponse(STATIC / "pairs.js", media_type="text/javascript")


@app.get("/api/pairs/runs")
def pair_runs():
    return {"runs": PL.list_runs()}


@app.get("/api/pairs/prompt")
def pair_prompt():
    return FileResponse(REPO_ROOT / "docs/pair_adjudication_prompt.md", media_type="text/plain")


@app.get("/api/pairs/{run_id}/manifest")
def pair_manifest(run_id: str):
    try:
        return PL.blind_manifest(run_id)
    except (PL.PairError, OSError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/pairs/{run_id}/summary")
def pair_summary(run_id: str):
    try:
        return PL.summary(run_id)
    except (PL.PairError, OSError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/pairs/{run_id}/sheet/{pair_id}")
def pair_sheet(run_id: str, pair_id: str):
    try:
        return FileResponse(PL.image_file(run_id, pair_id=pair_id), media_type="image/png", headers={"Cache-Control": "no-store"})
    except (PL.PairError, OSError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/pairs/{run_id}/contact/{page}")
def pair_contact(run_id: str, page: int):
    try:
        return FileResponse(PL.image_file(run_id, page=page), media_type="image/png", headers={"Cache-Control": "no-store"})
    except (PL.PairError, OSError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/pairs/{run_id}/import")
async def pair_import(run_id: str, request: Request):
    b = await _json_body(request)
    try:
        answers = b.get("answers")
        if isinstance(answers, str):
            answers = PL.parse_jsonl(answers)
        return await asyncio.to_thread(PL.import_answers, run_id, answers, b.get("annotator"), b.get("kind"))
    except (PL.PairError, OSError, ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/replay/{name}")
def replay_video(name: str):
    p = (REPLAY_DIR / name).resolve()
    if p.parent != REPLAY_DIR.resolve() or p.suffix.lower() not in REPLAY_SUFFIXES or not p.is_file():
        raise HTTPException(404, "No such replay video")
    media = {".mov": "video/quicktime", ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm"}[p.suffix.lower()]
    return FileResponse(p, media_type=media)


@app.post("/api/sessions/{session_id}/replay")
async def start_headless_replay(session_id: str, request: Request):
    """Server-side replay of a file in sweeps/ through the SAME engine.ingest_frame path as the camera."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    body = await request.json()
    p = (REPLAY_DIR / str(body.get("video", ""))).resolve()
    if p.parent != REPLAY_DIR.resolve() or p.suffix.lower() not in REPLAY_SUFFIXES or not p.is_file():
        raise HTTPException(404, "No such replay video")
    if session.replay_task and not session.replay_task.done():
        raise HTTPException(409, "A replay is already running")
    speed = float(body.get("speed", 1.0))

    async def run():
        try:
            await replay_into(session.engine, p, fps=float(body.get("fps", 5.0)), speed=speed, should_stop=lambda: session.deleted)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("replay failed: %s", type(exc).__name__)
    session.replay_task = asyncio.ensure_future(run())
    return {"started": True, "video": p.name, "speed": speed}


# ------------------------------------------------------------------------------- scans, build info, mounted debugger
def registered_scans() -> list[dict[str, Any]]:
    """Exactly the canonical scans (YugoTest1/YugoTest2). Aliases and H.264 previews are NOT scans."""
    out = []
    for sc in read_registry(REPO_ROOT):
        out.append({"label": sc["label"], "sha256": sc["sha256"], "size_bytes": sc.get("size_bytes"), "file": sc["file"],
                    "status": sc.get("status"), "original_name": sc.get("original_name"), "aliases": sc.get("aliases", []),
                    "browser_preview": sc.get("browser_preview"), "replay_url": f"/replay/scan/{sc['label']}",
                    "preview_available": bool(sc.get("browser_preview") and (REPLAY_DIR / sc["browser_preview"]).is_file()),
                    "source_available": (REPLAY_DIR / sc["file"]).is_file()})
    return out


@app.get("/api/scans")
def api_scans():
    try:
        return {"scans": registered_scans(), "note": "Aliases and preview derivatives are not independent scans."}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(500, f"scan registry invalid: {exc}") from exc


@app.get("/replay/scan/{label}")
def replay_scan(label: str):
    sc = next((x for x in registered_scans() if x["label"] == label), None)
    if sc is None or not sc["preview_available"]:
        raise HTTPException(404, "No browser-playable preview for this scan")
    return FileResponse(REPLAY_DIR / sc["browser_preview"], media_type="video/mp4")


@app.post("/api/sessions/{session_id}/scan")
async def bind_scan(session_id: str, request: Request):
    """Bind the active session to a registered scan BEFORE its replay frames arrive (session identity stays explicit)."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    body = await request.json()
    label = body.get("label")
    sc = next((x for x in registered_scans() if x["label"] == label), None) if label else None
    if label and sc is None:
        raise HTTPException(404, "Unknown scan label")
    try:
        session.engine.set_scan(sc)
    except SweepError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"session_id": session_id, "scan": sc and {k: sc[k] for k in ("label", "sha256", "file")}}


_PROCESS_STARTED = time.time()


def _source_fingerprint(patterns: tuple[str, ...]) -> str:
    import hashlib
    h = hashlib.sha256()
    for pat in patterns:
        for path in sorted(REPO_ROOT.glob(pat)):
            if path.is_file():
                h.update(path.name.encode())
                h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()[:12]


_BACKEND_GLOBS = ("app/*.py", "pipeline/*.py", "providers/*.py")
_STATIC_GLOBS = ("app/static/*",)
_LOADED_BACKEND = _source_fingerprint(_BACKEND_GLOBS)   # what this process imported at start


def build_info() -> dict[str, Any]:
    disk = _source_fingerprint(_BACKEND_GLOBS)
    paths = {getattr(r, "path", "") for r in app.routes}
    return {"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(_PROCESS_STARTED)),
            "uptime_s": round(time.time() - _PROCESS_STARTED, 1),
            "backend_fingerprint_loaded": _LOADED_BACKEND, "backend_fingerprint_on_disk": disk,
            "backend_source_changed_since_start": disk != _LOADED_BACKEND,
            "static_fingerprint_current": _source_fingerprint(_STATIC_GLOBS),
            "note": "Static files are served from disk on every request; Python backend code is only what was imported at start. "
                    "A differing backend fingerprint means restart this server to load the new backend.",
            "routes": {"measurement": any("/calibrations" in p for p in paths), "scans": "/api/scans" in paths,
                       "debug_mount": any(p.startswith("/debug") for p in paths), "frames": "/frames" in paths},
            "active_sessions": [{"session_id": x.session_id, "status": x.engine.status, "frames": len(x.engine.frames),
                                 "scan": (x.engine.scan or {}).get("label")} for x in sessions.values()],
            "canonical_url": os.getenv("LIBRARY_CANONICAL_URL", "http://127.0.0.1:4182/")}


@app.get("/api/build")
def api_build():
    return build_info()


@app.get("/debug")
def debug_redirect():
    return RedirectResponse("/debug/", status_code=307)


app.mount("/debug", DBG.app)   # the existing debugger, same origin; its relative URLs resolve under /debug/


@app.on_event("startup")
async def start_cleanup():
    async def sweep():
        while True:
            await asyncio.sleep(60)
            await cleanup_sessions()
    app.state.cleanup_task = asyncio.create_task(sweep())
    if os.getenv("LIBRARY_POST_SWEEP_RESUME", "1") != "0":     # the :4183 phone server sets 0 while :4182 runs, so the two never re-queue the same builds
        post_sweep.resume()     # builds that were queued or running when the server last stopped


@app.on_event("shutdown")
async def stop_cleanup():
    post_sweep.stop()
    app.state.cleanup_task.cancel()
    await asyncio.gather(app.state.cleanup_task, return_exceptions=True)
    for session in list(sessions.values()):
        await discard_session(session)
    from providers.stages import close_runtimes
    await close_runtimes()


# ------------------------------------------------------------------------------------------- live relay
@app.websocket("/ws/live")
async def live_voice(websocket: WebSocket) -> None:
    host = websocket.headers.get("host", "")
    if not local_host(host) or not LAN.client_ok(websocket.client.host) or not allowed_origin(
            websocket.headers.get("origin"), host, "https" if websocket.url.scheme == "wss" else "http") or (
            LAN.needs_key(websocket.client.host) and not LAN.key_ok(websocket.cookies.get(LAN.COOKIE))):
        await websocket.close(code=1008)
        return
    try:
        session = owned_session(websocket.query_params.get("session_id", ""), websocket.cookies.get("intake_owner"))
    except HTTPException:
        await websocket.close(code=1008)
        return
    if session.live_socket is not None:
        await websocket.close(code=1008)
        return
    session.live_socket = websocket
    await websocket.accept()
    from google.genai import types
    send_lock = asyncio.Lock()
    tasks: set = set()
    tool_tasks: dict[str, asyncio.Task] = {}
    live_ref: dict[str, Any] = {"session": None}
    pending = {"Claimant": {"id": uuid.uuid4().hex, "text": ""}, "Agent": {"id": uuid.uuid4().hex, "text": ""}}
    state_tick = {"last": 0.0, "dirty": False, "task": None}

    async def send(payload):
        if session.deleted:
            return
        async with send_lock:
            await websocket.send_json({**payload, "session_id": session.session_id})

    socket_state = {"closed": False}      # the phone closed the socket (normal at the end of a sweep, a reload or a lock screen)

    def normal_end(error: BaseException) -> bool:
        """The end of a sweep is not a failure. The client loop ends with WebSocketDisconnect when the phone closes the socket; the voice relay then
        fails on the closed Live session (APIError); a second End after the first is a SweepError. None of these is logged as a failed task."""
        return (isinstance(error, (WebSocketDisconnect, SweepError)) or socket_state["closed"]
                or session.engine.status != "capturing")

    def track(task):
        tasks.add(task)
        session.tasks.add(task)

        def finished(done):
            tasks.discard(done)
            session.tasks.discard(done)
            if not done.cancelled():
                error = done.exception()
                if error and normal_end(error):
                    logger.debug("Sweep task ended with the connection or the sweep: %s", type(error).__name__)
                elif error:
                    logger.error("Background sweep task failed: %s", type(error).__name__)
        task.add_done_callback(finished)
        return task

    async def push_state(snap: dict[str, Any] | None = None):
        await send({"type": "state", "state": _ui_state(session, snap)})

    async def engine_event(event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "state":
            now = time.monotonic()
            if now - state_tick["last"] < 0.25:  # coalesce bursts of state events, but always flush the latest
                if state_tick["task"] is None or state_tick["task"].done():
                    async def delayed():
                        await asyncio.sleep(0.3)
                        state_tick["last"] = time.monotonic()
                        await push_state()
                    state_tick["task"] = track(asyncio.create_task(delayed()))
                return
            state_tick["last"] = now
            await push_state(event.get("state"))   # the engine already built this snapshot: do not build it twice per push
            return
        if kind == "nudge":
            await send(event)
            live = live_ref["session"]
            if live is not None and event.get("relay", True):     # relay False: the voice agent's own tool result already carries these words (room sweep)
                with contextlib.suppress(Exception):
                    notice = LT.guidance_notice(event["message"]) if event.get("kind") == "guidance" else LT.capture_notice(event["message"])
                    await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=notice)]),
                                                   turn_complete=True)
            return
        if kind in ("frame", "status", "finalize"):
            await send(event)

    session._emit_to_client = engine_event  # engine events reach this socket while it is connected

    async def finalize(speaker):
        turn = pending[speaker]
        if not turn["text"].strip():
            return
        append_turn(session, speaker, turn["text"], turn["id"])
        await send({"type": "transcript", "speaker": speaker, "text": turn["text"], "id": turn["id"], "final": True})
        pending[speaker] = {"id": uuid.uuid4().hex, "text": ""}

    async def publish_tool(entry):
        session.tool_activity = [i for i in session.tool_activity if i["id"] != entry["id"]] + [dict(entry)]
        session.tool_activity = session.tool_activity[-30:]
        await send({"type": "tool", **entry})

    async def run_tool(name: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        engine = session.engine
        if name == "get_inventory_status":
            return LT.status_for_voice(engine.snapshot()), False
        if name == "confirm_market":
            country, currency = str(args.get("country", "")).upper()[:2], str(args.get("currency", "")).upper()[:3]
            if len(country) != 2 or len(currency) != 3:
                return {"error": "country must be a 2-letter code and currency a 3-letter code"}, False
            engine.market = {"country": country, "currency": currency}
            engine._bump()
            return {"country": country, "currency": currency, "note": "Prices are not produced at this stage; this only sets the market."}, False
        if name == "record_correction":
            action, scope = args.get("action"), args.get("scope")
            value: dict[str, Any] = {k: args[k] for k in ("title", "author", "language", "category", "reason") if args.get(k)}
            if action == "reclassify":
                value["kind"] = args.get("new_kind")
            if action == "set_title" and not value.get("title"):
                return {"error": "set_title needs the title the person stated"}, False
            source = {"actor": "claimant", "utterance": str(args.get("utterance", ""))[:500],
                      "turn_id": next((t["id"] for t in reversed(session.transcript) if t["speaker"] == "Claimant"), None)}
            try:
                if scope == "current_view":
                    res = await engine.add_correction(action, time_window=engine.current_view_window(float(args.get("seconds") or 4)),
                                                      value=value, source=source)
                else:
                    ids = [str(i) for i in (args.get("instance_ids") or [])]
                    if not ids:
                        return {"error": "instance_ids required for scope instance_ids"}, False
                    res = await engine.add_correction(action, instance_ids=ids, value=value, source=source)
            except (K.CorrectionError, SweepError) as exc:
                return {"error": str(exc)}, False
            res["totals"] = {k: res["totals"][k] for k in ("book_count", "books_identified", "books_unidentified", "item_count",
                                                          "user_excluded_count", "books_needs_appraisal")}
            return res, False
        if name == "record_calibration_dimension":
            try:
                res = await engine.set_dimension(str(args.get("what")), float(args.get("value")), str(args.get("unit")),
                                                 approximate=bool(args.get("approximate", True)), utterance=str(args.get("utterance", ""))[:500] or None,
                                                 turn_id=next((t["id"] for t in reversed(session.transcript) if t["speaker"] == "Claimant"), None))
            except (MeasurementError, SweepError, TypeError, ValueError) as exc:
                return {"error": str(exc)}, False
            return res, False
        if name == "get_measurement_status":
            ms = engine.measurement_status()
            return {"provisional": True, "missing": ms["missing"], "next_question": ms["next_prompt"],
                    "calibrations_valid": sum(c["status"] == "valid" for c in ms["calibrations"]),
                    "pending_dimensions": ms["pending_dimensions"], "ceiling_height_recorded": ms["height_anchor"] is not None,
                    "spines_measured": (ms["scale"] or {}).get("summary", {}).get("measured"), "spines_abstained": (ms["scale"] or {}).get("summary", {}).get("abstained"),
                    "room_status": ms["room"]["status"], "room_abstention_reasons": ms["room"]["abstention_reasons"],
                    "how_to_use": "Say only, in one sentence, what to FILM next (next_question). Never ask for a size or a tap. Do not state any size, area or length yourself."}, False
        if name == "end_sweep":
            first = await engine.request_end(source="voice", camera=bool(session.camera_enabled), skip=bool(args.get("skip_room")))
            if first["action"] == "room_sweep":       # the shelf pass is over; the room pan comes before the end
                return {"provisional": True, "sweep_ended": False, "room_sweep_first": True, "say": first["message"],
                        "how_to_use": ("The sweep has NOT ended. Say the instruction in 'say' in one or two short sentences, nothing else, no summary. "
                                       "When the person says they are done again, or asks to skip the room, call end_sweep again with skip_room true.")}, False
            summary = await engine.end_sweep()
            t = summary.get("totals", {})
            return {"provisional": True, "finalization_s": summary.get("finalization_s"),
                    "book_count": t.get("book_count"), "book_count_range": t.get("book_count_range"), "count_reliability": t.get("count_reliability"),
                    "count_sentence": LT.count_sentence(t), "books_identified": t.get("books_identified"),
                    "books_unidentified": t.get("books_unidentified"), "non_book_items": t.get("item_count"),
                    "excluded_by_user": t.get("user_excluded_count"), "packet_status": summary.get("packet_status"),
                    "notes": "Unpriced and unmeasured: prices, spine sizes and room measurements are unknown, not zero."}, False
        return {"error": "Unknown tool"}, False

    async def execute_tool(fc, live_session):
        started = time.monotonic()
        name, args, call_id = str(fc.name or ""), dict(fc.args or {}), str(fc.id or uuid.uuid4().hex)
        entry = {"id": call_id, "name": name, "args": args, "phase": "running", "headline": LT.tool_headline(name, args, None), "model": LT.LIVE_MODEL_ID}
        await publish_tool(entry)
        urgent = False
        try:
            result, urgent = await run_tool(name, args)
        except asyncio.CancelledError:
            entry.update(phase="cancelled", headline="Cancelled")
            with contextlib.suppress(Exception):
                await publish_tool(entry)
            raise
        except Exception:
            logger.exception("Tool %s failed", name)
            result = {"error": f"{name} failed. Continue the conversation and retry if needed."}
        scheduling = LT.scheduling_for(urgent=urgent)
        entry.update(phase="error" if "error" in result else "done", headline=result.get("error") or LT.tool_headline(name, args, result),
                     duration_ms=int((time.monotonic() - started) * 1000), result=result, scheduling=scheduling.value if scheduling else None)
        await publish_tool(entry)
        await push_state()
        if live_session is not None:
            await live_session.send_tool_response(function_responses=[types.FunctionResponse(id=call_id, name=name, response=result, scheduling=scheduling)])

    async def launch_tool(fc, live_session):
        if str(fc.id) in tool_tasks:
            return
        if len(tool_tasks) >= 4:
            await live_session.send_tool_response(function_responses=[types.FunctionResponse(
                id=fc.id, name=fc.name, response={"error": "Busy. Wait for current tools to finish."})])
            return
        task = track(asyncio.create_task(execute_tool(fc, live_session)))
        tool_tasks[str(fc.id)] = task
        task.add_done_callback(lambda done, key=str(fc.id): tool_tasks.pop(key, None))

    async def handle_client_message(message: dict[str, Any], live_session, windows) -> bool:
        """Returns False to stop the receive loop."""
        kind = message.get("type")
        if kind == "close":
            await finalize("Claimant")
            await finalize("Agent")
            return False
        if kind not in windows:
            raise ValueError("Unknown input type")
        now = time.monotonic()
        window = windows[kind]
        # evidence frames: the phone sends 5/s; Wi-Fi delivers them in bursts, so the limit is over 2 s with room for the jitter (YugoTest14 hit 6/s)
        period, limit = {"text": (60, 20), "audio": (1, 100), "video": (1, 5), "evidence_frame": (2, 15), "camera_state": (1, 5),
                         "end_sweep": (5, 2), "end_request": (5, 3), "capture_info": (10, 5)}[kind]
        while window and now - window[0] >= period:
            window.popleft()
        if len(window) >= limit:
            raise ValueError("Input rate limit reached; pause and try again")
        window.append(now)
        session.updated_at = now
        if kind == "camera_state":
            enabled = message.get("enabled")
            if not isinstance(enabled, bool):
                raise ValueError("Camera state must be true or false")
            session.camera_enabled = enabled
            if not enabled:
                await push_state()   # camera off: the capped live lists give way to the full lists right away
            if live_session is not None:
                await live_session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=LT.camera_mode_instruction(enabled))]),
                                                       turn_complete=False)
        elif kind == "capture_info":      # device label / stream settings / client counters from the phone; stored with the sweep's capture stats
            session.engine.set_capture_info({k: v for k, v in message.items() if k != "type"})
        elif kind == "end_request":       # the End button during a camera sweep; the first press may start the room pan instead of ending
            res = await session.engine.request_end(source="button", camera=bool(session.camera_enabled))
            await send({"type": "end_ack", "action": res["action"], "message": res.get("message")})
        elif kind == "end_sweep":
            if session.engine.status == "capturing":      # a second End (button after the voice agent ended it, or a double tap) is a no-op, not an error
                await send({"type": "status", "status": "finalizing"})
                track(asyncio.create_task(session.engine.end_sweep()))
        elif kind == "text":
            text = message.get("text", "")
            if not isinstance(text, str) or not text.strip() or len(text) > 8000:
                raise ValueError("Text must contain 1–8000 characters")
            turn_id = message.get("id") or uuid.uuid4().hex
            if not isinstance(turn_id, str) or len(turn_id) > 100:
                raise ValueError("Invalid turn identifier")
            if any(t.get("id") == turn_id for t in session.transcript):
                return True
            append_turn(session, "Claimant", text, turn_id)
            await send({"type": "transcript", "speaker": "Claimant", "text": text, "id": turn_id, "final": True})
            if live_session is not None:
                await live_session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=True)
            else:
                await send({"type": "error", "message": "Voice agent unavailable (no model credential). Your note is saved in the transcript only."})
        else:
            encoded = message.get("data")
            if not isinstance(encoded, str):
                raise ValueError("Missing media data")
            data = base64.b64decode(encoded, validate=True)
            limit_bytes = {"video": 512000, "evidence_frame": MAX_EVIDENCE_BYTES}.get(kind, 128000)
            if not data or len(data) > limit_bytes:
                raise ValueError("Invalid media size")
            if kind in ("evidence_frame", "video") and session.engine.status != "capturing":
                # frames still in flight when End was pressed (Wi-Fi bursts, the voice agent's end_sweep while the camera runs) are dropped
                # silently; the sweep is over and the client is not told it failed.
                session.engine.counters["frames_after_end_dropped"] += 1
                return True
            if kind == "evidence_frame":
                if not data.startswith(b"\xff\xd8\xff"):
                    raise ValueError("Evidence frames must be JPEG images")
                src = message.get("source", "camera")
                if src not in {"camera", "replay"}:
                    raise ValueError("Unknown frame source")
                session.camera_enabled = True
                try:
                    client = {"t_s": message.get("client_t_s"), "cand": message.get("cand"), "sharp_small": message.get("sharp_small"), "seen": message.get("seen"),
                              "win_ms": message.get("win_ms"), "enc_ms": message.get("enc_ms"), "q": message.get("q")} if src == "camera" else None
                    await session.engine.ingest_frame(data, message.get("t_s") if src == "replay" else None, src, client)
                except SweepError as exc:
                    if session.engine.status != "capturing":     # End landed while this frame was in flight: dropped, not an error
                        session.engine.counters["frames_after_end_dropped"] += 1
                        return True
                    raise ValueError(str(exc)) from exc
            elif kind == "video":
                # Compressed voice-context frame only. It is NEVER used as OCR evidence.
                if not data.startswith(b"\xff\xd8\xff"):
                    raise ValueError("Camera frames must be JPEG images")
                if live_session is not None:
                    await live_session.send_realtime_input(video=types.Blob(data=data, mime_type="image/jpeg"))
            else:
                if len(data) % 2:
                    raise ValueError("Audio must be PCM16")
                if live_session is not None:
                    await live_session.send_realtime_input(audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000"))
        return True

    async def client_loop(live_session):
        windows = {k: deque() for k in ("text", "audio", "video", "evidence_frame", "camera_state", "end_sweep", "end_request", "capture_info")}
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                socket_state["closed"] = True
                raise
            if len(raw) > MAX_MESSAGE_BYTES:
                await send({"type": "error", "message": "Input exceeded the message size limit."})
                continue
            try:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("Expected a JSON object")
                if not await handle_client_message(message, live_session, windows):
                    return
            except (ValueError, TypeError) as exc:
                await send({"type": "error", "message": str(exc)})

    async def model_loop(live_session):
        while True:
            async for response in live_session.receive():
                if response.tool_call and response.tool_call.function_calls:
                    await finalize("Claimant")
                    for fc in response.tool_call.function_calls:
                        await launch_tool(fc, live_session)
                if response.tool_call_cancellation:
                    for call_id in response.tool_call_cancellation.ids or []:
                        task = tool_tasks.get(str(call_id))
                        if task:
                            task.cancel()
                content = response.server_content
                if not content:
                    continue
                # Clear queued voice before forwarding more content; an interrupted payload can still carry cancelled output.
                if content.interrupted:
                    await send({"type": "interrupted"})
                    await finalize("Agent")
                for speaker, chunk in (("Claimant", content.input_transcription), ("Agent", content.output_transcription)):
                    if speaker == "Agent" and content.interrupted:
                        continue
                    if chunk and chunk.text:
                        if speaker == "Agent":
                            await finalize("Claimant")
                        if chunk.text != pending[speaker]["text"] or speaker == "Claimant":
                            pending[speaker]["text"] += chunk.text
                        await send({"type": "transcript", "speaker": speaker, **pending[speaker], "final": False})
                    if chunk and getattr(chunk, "finished", False):
                        await finalize(speaker)
                if content.model_turn and not content.interrupted:
                    for part in content.model_turn.parts or []:
                        if part.inline_data and isinstance(part.inline_data.data, bytes) and (part.inline_data.mime_type or "").startswith("audio/pcm"):
                            await finalize("Claimant")
                            await send({"type": "audio", "data": base64.b64encode(part.inline_data.data).decode("ascii"),
                                        "mime_type": part.inline_data.mime_type})
                if getattr(content, "turn_complete", False):
                    await finalize("Claimant")
                    await finalize("Agent")
                    await send({"type": "turn_complete"})

    try:
        voice, voice_reason = voice_status()
        session.voice_enabled = voice
        if not voice:
            # Degraded mode: no credential. Frames/nudges/inventory still work; voice does not.
            av = session.engine.stage_availability
            def line(stage):
                a = av.get(stage) or {}
                return f"{stage}: {'available (' + str(a.get('selection') or 'configured') + ')' if a.get('available') else 'unavailable' + (' - ' + str(a.get('reason')) if a.get('reason') else '')}"
            await send({"type": "session", "model": None, "voice": False, "tools": [],
                        "stages": av, "voice_reason": voice_reason,
                        "notice": f"Hosted voice is unavailable: {voice_reason}. Frames, quality nudges, local detection/reading and the inventory still run; "
                                  "actual stage availability: " + "; ".join(line(k) for k in ("detect", "read")) + ". Replay is not a live microphone/voice test."})
            await push_state()
            await send({"type": "ready", "voice": False, "voice_reason": voice_reason})
            await client_loop(None)
            return
        history = [types.Content(role="user" if t["speaker"] == "Claimant" else "model", parts=[types.Part(text=t["text"])])
                   for t in session.transcript if t["speaker"] in {"Claimant", "Agent"}]
        config = LT.build_live_config(camera_enabled=session.camera_enabled, seed_history=bool(history))
        session.live_model = LT.LIVE_MODEL_ID
        HG.check("live_voice_connect")                  # user kill-switch: refuses to open a billable live session
        async with _live_client().aio.live.connect(model=LT.LIVE_MODEL_ID, config=config) as live_session:
            live_ref["session"] = live_session
            if history:
                await live_session.send_client_content(turns=history, turn_complete=True)
            await send({"type": "session", "model": LT.LIVE_MODEL_ID, "voice": True, "tools": LT.TOOL_NAMES})
            await push_state()
            await send({"type": "ready", "voice": True})
            pair = [track(asyncio.create_task(client_loop(live_session))), track(asyncio.create_task(model_loop(live_session)))]
            done, _ = await asyncio.wait(pair, timeout=20 * 60, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                await send({"type": "error", "message": "The live connection reached 20 minutes. Reconnect to continue this sweep."})
            for task in done:
                if not task.cancelled():  # a reset/discard cancels the relay tasks: that is a normal shutdown
                    task.result()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if normal_end(exc):       # the Live relay failing because the phone left or the sweep ended is the normal end of the connection
            logger.debug("Live session closed with the sweep: %s", type(exc).__name__)
            return
        logger.exception("Live session failed")
        with contextlib.suppress(Exception):
            await send({"type": "error", "message": "Live connection ended. Reconnect to continue this sweep."})
    finally:
        live_ref["session"] = None
        session._emit_to_client = None
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks), return_exceptions=True)
        for item in session.tool_activity:
            if item["phase"] == "running":
                item.update(phase="cancelled", headline="Connection ended")
        session.live_socket = None
        session.camera_enabled = False
        session.updated_at = time.monotonic()
        with contextlib.suppress(Exception):
            await websocket.close()


@app.get(LAN.CA_PATH)
def lan_ca() -> FileResponse:
    """The public CA certificate the phone installs once (scripts/serve_https.sh). Only in LAN mode; certificates are not secret."""
    path = LAN.ca_file() if LAN.enabled() else None
    if not path:
        raise HTTPException(404, "No LAN CA configured.")
    return FileResponse(path, media_type="application/x-x509-ca-cert", filename="library-lan-ca.crt")


# ------------------------------------------------------------------------------- laptop watch page: read-only, loopback server only
_watch_queue = None


def _watch_post_status(run_id: str) -> dict[str, Any] | None:
    """The post-sweep queue's status for RUNS_DIR: the server's own queue (repo runs/, the normal case), else a read-only reader of the same layout
    next to LIBRARY_RUNS_DIR (a test server on a scratch runs dir)."""
    global _watch_queue
    if RUNS_DIR.resolve() == (REPO_ROOT / "runs").resolve():
        return post_sweep.status(run_id)
    if RUNS_DIR.name != "runs":
        return None
    if _watch_queue is None:
        from app.build_jobs import BuildJobs
        _watch_queue = PostSweepBuilds(RUNS_DIR.parent, BuildJobs(RUNS_DIR.parent), enabled=False)
    return _watch_queue.status(run_id)


def _watch_live(run_id: str) -> dict[str, Any] | None:
    """watch_status of an engine running in THIS process (fresher than its watch.json)."""
    for s in list(sessions.values()) + [e["session"] for e in list(detached.values())]:
        if s.run_dir.name == run_id and not s.engine._closed:
            return s.engine.watch_status()
    return None


def _watch_guard(request: Request) -> None:
    if not watch_allowed(request):           # the middleware already answers 404; kept here so the routes never depend on middleware order
        raise HTTPException(404, "Not Found")


@app.get("/watch")
def watch_page(request: Request):
    _watch_guard(request)
    return FileResponse(STATIC / "watch.html")


@app.get("/watch.js")
def watch_script(request: Request):
    _watch_guard(request)
    return FileResponse(STATIC / "watch.js", media_type="text/javascript")


@app.get("/watch.css")
def watch_styles(request: Request):
    _watch_guard(request)
    return FileResponse(STATIC / "watch.css", media_type="text/css")


@app.get("/api/watch/sweeps")
def watch_sweeps(request: Request):
    _watch_guard(request)
    return WATCH.list_sweeps(RUNS_DIR, live=_watch_live, post_status=_watch_post_status)


@app.get("/api/watch/sweeps/{run_id}")
def watch_sweep(run_id: str, request: Request):
    _watch_guard(request)
    st = WATCH.sweep_state(RUNS_DIR, run_id, live=_watch_live(run_id) if WATCH.RUN_RE.fullmatch(run_id) else None, post_status=_watch_post_status)
    if st is None:
        raise HTTPException(404, "No such sweep")
    return st


@app.get("/api/watch/sweeps/{run_id}/frames/{name}")
def watch_frame(run_id: str, name: str, request: Request):
    _watch_guard(request)
    d = WATCH.run_dir(RUNS_DIR, run_id)
    path = d / "frames" / name if d is not None and WATCH.FRAME_RE.fullmatch(name) else None
    if path is None or path.is_symlink() or not path.is_file():
        raise HTTPException(404, "No such frame")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/watch/sweeps/{run_id}/packet")
def watch_packet(run_id: str, request: Request):
    """The claim packet for the viewer (status, issues, totals, books, items, pricing, room/measurement, stage_status), paths redacted."""
    _watch_guard(request)
    d = WATCH.run_dir(RUNS_DIR, run_id)
    view = WATCH.packet_view(d) if d is not None and not (d / "claim_packet.json").is_symlink() else None
    if view is None:
        raise HTTPException(404, "The claim packet is written when the sweep ends.")
    return view


@app.get("/api/watch/sweeps/{run_id}/packet.json")
def watch_packet_raw(run_id: str, request: Request, download: int = 0):
    """The whole claim packet (raw view and download), absolute paths redacted."""
    _watch_guard(request)
    d = WATCH.run_dir(RUNS_DIR, run_id)
    body = WATCH.packet_raw(d) if d is not None else None
    if body is None:
        raise HTTPException(404, "The claim packet is written when the sweep ends.")
    headers = {"Content-Disposition": f'attachment; filename="{run_id}_claim_packet.json"'} if download else {}
    return Response(body, media_type="application/json", headers=headers)


@app.get("/api/watch/sweeps/{run_id}/transcript")
def watch_transcript(run_id: str, request: Request):
    """The voice conversation of the sweep (person and voice agent, with times), from <run>/live/transcript.jsonl."""
    _watch_guard(request)
    d = WATCH.run_dir(RUNS_DIR, run_id)
    if d is None:
        raise HTTPException(404, "No such sweep")
    return WATCH.transcript(d)


@app.get("/api/watch/sweeps/{run_id}/crops/{name}")
def watch_crop(run_id: str, name: str, request: Request):
    """A book's spine crop (thumbnail in the packet viewer's books table)."""
    _watch_guard(request)
    d = WATCH.run_dir(RUNS_DIR, run_id)
    path = d / "crops" / name if d is not None and WATCH.CROP_RE.fullmatch(name) else None
    if path is None or path.is_symlink() or not path.is_file():
        raise HTTPException(404, "No such crop")
    return FileResponse(path, media_type="image/png")


@app.get("/api/watch/sweeps/{run_id}/report")
def watch_report(run_id: str, request: Request):
    _watch_guard(request)
    d = WATCH.run_dir(RUNS_DIR, run_id)
    path = d / "report.html" if d is not None else None
    if path is None or path.is_symlink() or not path.is_file():
        raise HTTPException(404, "The report is written when the sweep ends.")
    return FileResponse(path, media_type="text/html")


# ------------------------------------------------------------------------------- ground-truth labelling page: loopback server only
# Reads the sweep's packet, frames and crops (never writes under runs/); answers go to ground_truth/ground_truth_labels.json.
def _truth_run(run_id: str) -> Path:
    d = WATCH.run_dir(RUNS_DIR, run_id)
    if d is None or (d / "claim_packet.json").is_symlink() or not (d / "claim_packet.json").is_file():
        raise HTTPException(404, "No claim packet for that sweep")
    return d


def _truth_cards(run_id: str) -> dict[str, Any]:
    cards = GT.packet_cards(_truth_run(run_id))
    if cards is None:
        raise HTTPException(404, "The claim packet has no books or items")
    return cards


@app.get("/truth")
def truth_page(request: Request):
    _watch_guard(request)
    return FileResponse(STATIC / "truth.html")


@app.get("/truth.js")
def truth_script(request: Request):
    _watch_guard(request)
    return FileResponse(STATIC / "truth.js", media_type="text/javascript")


@app.get("/truth.css")
def truth_styles(request: Request):
    _watch_guard(request)
    return FileResponse(STATIC / "truth.css", media_type="text/css")


@app.get("/api/truth/default")
def truth_default(request: Request):
    _watch_guard(request)
    return {"run": GT.DEFAULT_RUN}


@app.get("/api/truth/{run_id}")
def truth_cards(run_id: str, request: Request):
    """Cards for the page (books, items, the 15 sampled prices, categories) + this sweep's saved answers."""
    _watch_guard(request)
    cards = _truth_cards(run_id)
    try:
        labels = GT.get_labels(run_id)
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from None
    return {**cards, "labels": labels, "labels_file": f"ground_truth/{GT.LABELS_NAME}", "sheet_file": GT.sheet_path(run_id).name,
            "excel_open": GT.excel_lock(run_id)}


@app.post("/api/truth/{run_id}/labels")
async def truth_save(run_id: str, request: Request):
    _watch_guard(request)
    cards = _truth_cards(run_id)
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "expected JSON") from None
    try:
        cur = await asyncio.to_thread(GT.save_labels, run_id, body, cards)
    except GT.LabelError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"saved": True, "updated_at": cur.get("updated_at")}


@app.post("/api/truth/{run_id}/sheet")
async def truth_write_sheet(run_id: str, request: Request, force: int = 0):
    """Rebuild the workbook (scripts/ground_truth_sheet.py) with this sweep's answers in the yellow cells; returns the Results preview."""
    _watch_guard(request)
    d = _truth_run(run_id)
    if GT.excel_lock(run_id) and not force:
        raise HTTPException(409, f"Excel has {GT.sheet_path(run_id).name} open. Save and close it first (values typed there are kept), "
                                 "or write anyway.")
    return await asyncio.to_thread(GT.write_sheet, run_id, d)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/index.html")
def index_alias():
    return FileResponse(STATIC / "index.html")


@app.get("/app.js")
def javascript():
    return FileResponse(STATIC / "app.js", media_type="text/javascript")


@app.get("/styles.css")
def styles():
    return FileResponse(STATIC / "styles.css", media_type="text/css")


@app.get("/frames")
def frame_browser_page():
    return FileResponse(STATIC / "frame-browser.html")


@app.get("/frame-browser.js")
def frame_browser_script():
    return FileResponse(STATIC / "frame-browser.js", media_type="text/javascript")


@app.get("/frame-browser.css")
def frame_browser_styles():
    return FileResponse(STATIC / "frame-browser.css", media_type="text/css")
