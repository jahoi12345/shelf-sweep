"""Incremental sweep engine: one ingestion path for live camera and recorded-video replay.

Frames arrive one at a time through ``SweepEngine.ingest_frame`` (from the browser camera, the browser
video-replay preview, or the headless ``app.replay`` driver; they are indistinguishable except for the
``source`` label). Each frame is saved UNCHANGED at the resolution received (high-resolution evidence;
the 512 px voice-context copy is a separate message and is never used for OCR), quality-scored with the
shared ``pipeline.quality`` code and fed to the shared ``pipeline.frames.PanIntervalSelector``. When a pan
interval closes, only that keyframe goes through the existing stage functions:

    detect(frame)  ->  read(new book observations)  ->  resolve(readings)  ->  (debounced) dedup.merge(all)

``merge`` is the only whole-run step; it is coalesced (at most one in flight, rerun if new data arrived)
instead of running per frame. Corrections are an overlay (``pipeline.corrections``) applied when a snapshot or
the final packet is built, so a stale merge can never undo them. Totals come from ``pipeline.packet``.
Stage functions are injected (``StageFns``) so tests can use fakes with the real signatures.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import copy
import logging
import itertools
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import cv2
import numpy as np

from pipeline import contracts as C
from app.measurement import MeasurementError, MeasurementState
from pipeline import corrections as K
from pipeline import packet as P
from pipeline import report as R
from pipeline.frames import PanIntervalSelector
from pipeline.quality import QualityConfig, frame_quality, reasons_for, sharpness_floor
from providers import gemini
from providers.stages import stage_configs
from providers import readers as plugin_hooks
from app import group_reader as GR
from app.capture_quality import MESSAGES as CQ_MESSAGES, CaptureQuality, capture_stats
from app.nudge_policy import NudgePolicy
from app.room_sweep import RoomSweep, long_side_px

log = logging.getLogger(__name__)

MAX_FRAMES = 1500          # hard cap on saved evidence frames per sweep
MIN_FRAME_BYTES = 1000
QUALITY_LONG_SIDE = int(os.getenv("LIBRARY_QUALITY_LONG_SIDE", "1920"))   # frame-quality signals (sharpness, motion, zoom, glare) are measured on a copy with at most this long side:
# the thresholds and every earlier run are 1080p-class numbers, and a 4K frame (3840x2160, ~2.5 MB) would cost ~4x the ingest time. The saved JPEG is always the native bytes.


@dataclass
class StageFns:
    """Synchronous callables with the stable stage signatures (run in worker threads)."""
    detect: Callable[..., Any]
    read: Callable[..., Any]
    resolve: Callable[..., Any]
    merge: Callable[..., Any]

    @classmethod
    def real(cls) -> "StageFns":
        from pipeline import dedup, detect, read_spines, resolve
        return cls(detect=detect.detect, read=read_spines.read, resolve=resolve.resolve, merge=dedup.merge)


class SweepError(RuntimeError):
    pass


class CaptureMonitor:
    """Turns recent frame/stage signals into capture-quality nudges (deterministic code, not the model). The detectors offer a code on every frame
    the condition holds; the shared app.nudge_policy.NudgePolicy decides whether it is spoken (once per bad stretch, re-arm, global gap)."""

    MESSAGES = {   # short and direct (user decision 2026-10-07)
        "blur": "Slow down. The picture is blurry.",
        "fast_pan": "Slow down. Keep the same distance.",
        "revisit": "Do not go back over shelves already filmed. Keep moving forward.",
        "glare": "Tilt the phone a little to avoid the glare.",
        "dark": "It is too dark. Turn on a light.",
        "overlap": "Slow down. Part of the shelf may have been skipped.",
        "closer": "Move a bit closer. The spines are too small to read.",
    }

    def __init__(self, policy: NudgePolicy | None = None) -> None:
        self.policy = policy or NudgePolicy()
        self.recent: collections.deque[list[str]] = collections.deque(maxlen=6)
        self.reads: collections.deque[bool] = collections.deque(maxlen=8)
        self._gaps_seen = 0
        self._pos = [0.0, 0.0]            # signed accumulated image motion, in frame widths / heights
        self._dir: int | None = None      # dominant horizontal direction of the current row
        self._best = 0.0                  # furthest progress along that direction
        self._row_y = 0.0

    def motion(self, dx_w: float, dy_h: float, now: float) -> list[dict[str, Any]]:
        """Flag a revisit: the camera retreats > 1 frame width against the row's travel direction (a new row, i.e. a vertical move > 0.4 of the
        frame height, starts a fresh baseline). Uses the same phase-correlation shift as frame selection; unreliable estimates are skipped."""
        self._pos[0] += dx_w
        self._pos[1] += dy_h
        if abs(self._pos[1] - self._row_y) > 0.4:
            self._row_y, self._dir, self._best = self._pos[1], None, 0.0
            self._base = self._pos[0]
        if self._dir is None:
            base = getattr(self, "_base", 0.0)
            if abs(self._pos[0] - base) > 1.5:
                self._dir = 1 if self._pos[0] > base else -1
                self._best = self._dir * self._pos[0]
            return []
        prog = self._dir * self._pos[0]
        self._best = max(self._best, prog)
        if self._best - prog > 1.0 and self._ok("revisit", now):
            return [{"code": "revisit", "message": self.MESSAGES["revisit"]}]
        return []

    def _ok(self, code: str, now: float) -> bool:
        return self.policy.offer(code, now)

    def frame(self, reasons: list[str], now: float, overlap_gaps: int) -> list[dict[str, Any]]:
        self.recent.append(reasons)
        out = []
        n = len(self.recent)
        count = lambda tag, window: sum(tag in r for r in list(self.recent)[-window:])
        if n >= 4 and count("blurry", 6) >= 4 and self._ok("blur", now):
            out.append("blur")
        if n >= 3 and count("fast_motion", 5) >= 3 and self._ok("fast_pan", now):
            out.append("fast_pan")
        if n >= 3 and count("glare", 5) >= 3 and self._ok("glare", now):
            out.append("glare")
        if n >= 4 and count("dark", 6) >= 4 and self._ok("dark", now):
            out.append("dark")
        if overlap_gaps > self._gaps_seen:
            self._gaps_seen = overlap_gaps
            if self._ok("overlap", now):
                out.append("overlap")
        return [{"code": c, "message": self.MESSAGES[c]} for c in out]

    def reading(self, readable: bool, now: float) -> list[dict[str, Any]]:
        self.reads.append(readable)
        if len(self.reads) >= 6 and sum(not r for r in self.reads) >= 5 and self._ok("closer", now):
            return [{"code": "closer", "message": self.MESSAGES["closer"]}]
        return []


def merged_ok(engine: "SweepEngine") -> bool:
    return bool(engine.inventory and (engine.inventory.get("data") or {}).get("books") is not None)


class SweepEngine:
    def __init__(self, run_dir: str | Path, *, config: dict[str, Any] | None = None, stages: StageFns | None = None,
                 emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None, session_id: str | None = None,
                 on_finished: Callable[[Path], None] | None = None):
        self.on_finished = on_finished   # called once with the run dir AFTER the packet and report are written (post-sweep 2D/3D builds)
        self.run_dir = Path(run_dir).resolve()
        (self.run_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "live").mkdir(exist_ok=True)
        self.config = dict(config or {})
        self.run_id = self.config.get("run_id") or session_id or self.run_dir.name
        self.config["run_id"] = self.run_id
        self.session_id = session_id or self.run_id
        self.stages = stages or StageFns.real()
        self.emit = emit
        self.market = {"country": self.config.get("country", "US"), "currency": self.config.get("currency", "USD")}
        self.qcfg = QualityConfig.from_dict(self.config.get("quality"))
        self.overlap_step = float(self.config.get("overlap_step", 0.25))
        self.usage = gemini.UsageLog(self.run_dir / "logs" / "usage.jsonl")
        # ONE shared hosted response cache across runs (key = image sha256 + prompt + model + generation config incl. thinking budget): a rerun of the
        # same video never pays twice. LIBRARY_HOSTED_CACHE_DIR (default runs/_hosted_cache); cache hits are counted in the hosted-switch counters.
        self.cache = gemini.get_cache(self.config.get("cache_dir") or os.environ.get("LIBRARY_HOSTED_CACHE_DIR") or (self.run_dir.parent / "_hosted_cache"),
                                      enabled=not self.config.get("no_cache"))
        self.stage_config = self._stage_config()
        self.frames: list[dict[str, Any]] = []
        self.selector: PanIntervalSelector | None = None
        self.nudge_policy = NudgePolicy(self.config.get("nudge_rearm_s"), self.config.get("nudge_gap_s"))   # ONE spoken-notice policy for both nudge sources
        self.monitor = CaptureMonitor(self.nudge_policy)
        self.room_sweep = RoomSweep(self.config.get("room_sweep_prompt"))   # the room instruction comes at the END of the shelf pass, never during it
        self.nudges: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []
        self.readings: list[dict[str, Any]] = []
        self.resolutions: list[dict[str, Any]] = []
        self.stage_log: dict[str, dict[str, Any]] = {}
        self.stage_issues: list[dict[str, Any]] = []
        self.ledger: list[dict[str, Any]] = []
        self.inventory: dict[str, Any] | None = None
        self.merged_observation_ids: set[str] = set()
        self.status = "capturing"
        self.revision = 0
        self.sources: set[str] = set()
        self.started = time.monotonic()
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.ended: float | None = None
        self.finalization: dict[str, Any] | None = None
        self.finalize_stage: str | None = None       # human label of the current end-sweep step (status indicator only)
        self.finalize_error: str | None = None
        self.final_paths: dict[str, str] = {}
        self.prices = None
        self.scan: dict[str, Any] | None = None
        self.meas = MeasurementState()
        self._last_measure: tuple[dict[str, Any] | None, dict[str, Any]] | None = None
        self._prev_small: np.ndarray | None = None
        self._prev_t: float | None = None
        self._prev_cap_t: float | None = None
        self._portrait_nudged = False          # orientation nudge spoken (once per upright stretch via the policy), camera source only
        self._sharp: collections.deque[float] = collections.deque(maxlen=40)
        self._ingest_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._detect_sem = asyncio.Semaphore(int(self.config.get("detect_concurrency", 2)))
        self._read_sem = asyncio.Semaphore(int(self.config.get("read_concurrency", 2)))
        self._merge_lock = asyncio.Lock()
        self._merge_task: asyncio.Task | None = None
        self._merge_dirty = False
        self._merge_error: str | None = None
        self.merge_debounce_s = float(self.config.get("merge_debounce_s", 0.5))
        self.merge_cost_factor = float(self.config.get("merge_cost_factor", 4.0))
        self.merge_live_budget_s = float(self.config.get("merge_live_budget_s", 20.0))
        self.dedup_max_gap_s = float(self.config.get("dedup_max_gap_s", os.environ.get("LIBRARY_DEDUP_MAX_GAP_S", 8.0)))
        self._last_merge_cost = 0.0
        self._last_merge_end = 0.0
        self._live_merges_stopped = False
        self._corr_seq = 0
        self._closed = False
        self._end_task: asyncio.Task | None = None   # the running end_sweep: owned here so a cancelled caller (socket closed, New sweep) cannot cancel the packet
        self.counters = collections.Counter()
        # overlap-aware detection cadence: add intermediate saved frames when consecutive detection frames would overlap < min_overlap
        self.min_overlap = float(self.config.get("detect_min_overlap", os.environ.get("LIBRARY_DETECT_MIN_OVERLAP", 0.30)))   # 0 disables
        self.extra_detect_max = int(self.config.get("extra_detect_max", os.environ.get("LIBRARY_EXTRA_DETECT_MAX", 60)))
        self.detect_policy = str(self.config.get("detect_policy", os.environ.get("LIBRARY_DETECT_POLICY", "dense")))   # dense (default) | keyframes
        if self.detect_policy not in ("keyframes", "dense"):
            raise ValueError("detect_policy must be keyframes or dense")
        # Defaults (design decision, 2026-10-06): uniform 2.5 fps detection (half the saved frames: what the local detector sustains), blurry frames NOT skipped
        # (uniform thinning keeps chain continuity; blur-skipping removed exactly the fast stretches). Measured frame-fraction curve: YT1 50% = 455 (-0.7%).
        self.detect_max_fps = float(self.config.get("detect_max_fps", os.environ.get("LIBRARY_DETECT_MAX_FPS", 2.5)))
        self.detect_dense_max = int(self.config.get("detect_dense_max", os.environ.get("LIBRARY_DETECT_DENSE_MAX", 480)))   # 2.5 det/s x 3 min sweep + margin
        self.detect_skip_blurry = str(self.config.get("detect_skip_blurry", os.environ.get("LIBRARY_DETECT_SKIP_BLURRY", "0"))).lower() not in ("0", "false", "no")
        # Blur gate: frames below the adaptive Laplacian floor (ratio x running median of the last 40 frames, never below an absolute floor)
        # are stored but not detected. "live" (default) = camera frames only; "all" = replays too (also what LIBRARY_DETECT_SKIP_BLURRY=1 means); "off".
        self.blur_gate = "all" if self.detect_skip_blurry else str(self.config.get("blur_gate", os.environ.get("LIBRARY_BLUR_GATE", "live"))).lower()
        if self.blur_gate not in ("live", "all", "off"):
            raise ValueError("blur_gate must be live, all or off")
        self.blur_gate_ratio = float(self.config.get("blur_gate_ratio", os.environ.get("LIBRARY_BLUR_GATE_RATIO", self.qcfg.blur_floor_ratio)))
        self.blur_gate_abs = float(self.config.get("blur_gate_abs", os.environ.get("LIBRARY_BLUR_GATE_ABS", self.qcfg.blur_floor_abs)))
        self.blur_gate_min_history = int(self.config.get("blur_gate_min_history", os.environ.get("LIBRARY_BLUR_GATE_MIN_HISTORY", 8)))
        self.capture_info: dict[str, Any] = {}
        self._detected_ids: set[str] = set()
        self._gated_ids: set[str] = set()
        self._gated_order: list[str] = []
        self._last_dense_t = -1e9
        self.dense = {"policy": self.detect_policy, "max_fps": self.detect_max_fps, "budget": self.detect_dense_max, "skip_blurry": self.detect_skip_blurry, "dense_frames": 0, "skipped_blurry": 0,
                      "skipped_rate": 0, "budget_exhausted": False, "gated_frames": 0}
        self.read_policy = str(self.config.get("read_policy", os.environ.get("LIBRARY_READ_POLICY", "group")))   # group | observation
        if self.read_policy not in ("group", "observation"):
            raise ValueError("read_policy must be group or observation")
        self.read_workers = int(self.config.get("read_workers", os.environ.get("LIBRARY_READ_WORKERS", 3)))
        # 400 s reads nearly all groups of a YT2-size run (~3 s per group-read call, 3 workers); packet time is reported honestly in finalization_timings.
        self.read_budget_s = float(self.config.get("read_budget_s", os.environ.get("LIBRARY_READ_BUDGET_S", 400)))
        self.packet_target_s = float(self.config.get("packet_target_s", os.environ.get("LIBRARY_PACKET_TARGET_S", 290)))   # adaptive: reading gets what is left of this
        self.read_report: dict[str, Any] = {"policy": self.read_policy, "workers": self.read_workers, "budget_s": self.read_budget_s, "status": "not_run"}
        self.cq = CaptureQuality(self.config, policy=self.nudge_policy)
        self.readings2: list[dict[str, Any]] = []
        self.alt_count: dict[str, Any] | None = None
        # the alternative association (track_v2) is a diagnostic only; by default it runs AFTER the packet is written and is patched into it,
        # so it is off the packet's critical path (LIBRARY_ALT_AFTER_PACKET=0 restores the old in-line step before the group reading).
        self.alt_after_packet = str(self.config.get("alt_after_packet", os.environ.get("LIBRARY_ALT_AFTER_PACKET", "1"))).lower() not in ("0", "false", "no")
        # the re-merge after the group reading only changes what pipeline.dedup._identity sees (association never reads readings/resolutions): it is replaced
        # by an in-place identity refresh when that is exact (_remerge_skippable; LIBRARY_REMERGE_SKIP=0 disables)
        self.remerge_skip = str(self.config.get("remerge_skip", os.environ.get("LIBRARY_REMERGE_SKIP", "1"))).lower() not in ("0", "false", "no")
        # honest read budget: reserve this many x the final merge's duration for the re-merge that follows the reading (LIBRARY_REMERGE_RESERVE)
        self.remerge_reserve = float(self.config.get("remerge_reserve", os.environ.get("LIBRARY_REMERGE_RESERVE", 1.0)))
        self._merged_accept_map: dict[str, tuple[str, Any]] | None = None
        self.remerge_info: dict[str, Any] = {}
        # provisional in-sweep reads: groups stable for >= stable_s with >= stable_min sightings get their best sighting read DURING capture
        self.provisional_reads = str(self.config.get("provisional_reads", os.environ.get("LIBRARY_PROVISIONAL_READS", "1"))).lower() not in ("0", "false", "no")
        self.stable_min = int(self.config.get("stable_min_sightings", os.environ.get("LIBRARY_STABLE_MIN_SIGHTINGS", 3)))
        self.stable_s = float(self.config.get("stable_s", os.environ.get("LIBRARY_STABLE_S", 10.0)))
        self._group_seen: dict[frozenset, float] = {}        # observation-id set -> first time seen unchanged
        self._prov_done: dict[str, dict[str, Any]] = {}     # observation_id -> reading made during capture
        self._prov_inflight: set[str] = set()
        self._prov_chunk = 0
        self.prov_report = {"enabled": self.provisional_reads, "groups_read": 0, "observations_read": 0, "chunks": 0, "status": "idle"}
        self._cum: list[float] = []            # cumulative estimated image path (in frame widths) per saved frame
        self._last_det_idx: int | None = None
        self.overlap_fill: dict[str, Any] = {"enabled": self.min_overlap > 0, "min_overlap": self.min_overlap, "budget": self.extra_detect_max,
                                             "extra_frames": [], "unfillable_gaps": [], "budget_exhausted": False}
        # read-only laptop watch page (/watch on the loopback server). The phone's sweep runs in the :4183 process, so the engine keeps a small
        # status file (<run>/watch.json, atomic, at most ~1/s plus every status/step/notice change) that the :4182 process reads. Off unless the server asks.
        self.started_epoch = time.time()
        self._ended_epoch: float | None = None
        self._watch_t = 0.0
        self._watch_key: tuple | None = None
        self._watch_nudges: list[dict[str, Any]] = []
        self._watch_steps: list[dict[str, Any]] = []
        self._watch_totals: dict[str, Any] | None = None
        self._watch_first: dict[str, Any] | None = None
        self._watch_closed = False
        self._watch_task: asyncio.Task | None = None
        self.watch_file = bool(self.config.get("watch_file"))
        self._watch_write(force=True)

    # ------------------------------------------------------------------ config / provenance
    def _stage_config(self) -> dict[str, Any]:
        self.stage_configs, self.stage_availability, self.stages, self.hook_notes = plugin_hooks.apply(
            self.config, self.stages, run_dir=self.run_dir, cache=self.cache, usage=self.usage, stage_configs_fn=stage_configs)
        self.provider_enabled = any(v['available'] for v in self.stage_availability.values())
        return self.stage_configs['dedup']

    def provenance(self) -> dict[str, Any]:
        s = self.usage.summary()
        local_calls = sum(v.get("detect", {}).get("local_inference_count", 0) + v.get("read", {}).get("local_inference_count", 0) for v in self.stage_log.values())
        if gemini.is_mocked():
            mode = "mocked"
        elif local_calls:
            mode = "real"
        elif not s["requests"] and not s["cache_hits"]:
            mode = "unavailable"
        else:
            mode = "real" if s["requests"] else ("cached" if s["cache_hits"] else "real_configured_no_calls_yet")
        return {"inference_mode": mode, "credential_available": gemini.credential_available(),
                "fresh_requests": s["requests"], "cache_hits": s["cache_hits"], "models": s["models"],
                "cost_usd": 0 if local_calls and not s['requests'] else s["cost_usd"],
                "cost_note": "Local CPU inference: zero API charge; hardware/energy costs not measured" if local_calls and not s['requests'] else s["cost_note"],
                "local_ocr_inferences": sum(v.get('read', {}).get('local_inference_count', 0) for v in self.stage_log.values()),
                "tokens": {k: s[k] for k in ("prompt_tokens", "output_tokens", "thought_tokens", "total_tokens")},
                "capture_sources": sorted(self.sources), "stages": self.stage_availability,
                "hooks": {k: (v() if callable(v) else v) for k, v in self.hook_notes.items()}, "provisional_reads": self.prov_report, "capture_quality": self.capture_quality(), "read_policy": {k: v for k, v in self.read_report.items() if k != "plan_groups"}, "detect_policy": {**self.dense, "backlog": self.counters["keyframes"] - sum(1 for v in self.stage_log.values() if isinstance(v, dict) and v.get("detect", {}).get("status") not in (None, "running"))}, "overlap_fill": {**self.overlap_fill, "extra_frames": len(self.overlap_fill["extra_frames"])}, "orchestration": {"dedup_max_gap_s": self.dedup_max_gap_s, "merge_cost_factor": self.merge_cost_factor, "merge_live_budget_s": self.merge_live_budget_s,
                                  "live_merges_stopped": self._live_merges_stopped, "ocr_max_books_per_keyframe": self.config.get("max_books")},
                "local_detection_calls": sum(v.get("detect", {}).get("local_inference_count", 0) for v in self.stage_log.values()),
                "note": "Detection and OCR availability are independent; local detection needs no Gemini key. Counts remain proposals."}

    # ------------------------------------------------------------------ events / tasks
    async def _emit(self, event: dict[str, Any]) -> None:
        if self.emit is not None:
            with contextlib.suppress(Exception):
                await self.emit(event)

    def _spawn(self, coro: Awaitable[Any], name: str) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)

        def done(t: asyncio.Task) -> None:
            self._tasks.discard(t)
            if not t.cancelled() and t.exception():
                log.error("task %s failed: %s", name, type(t.exception()).__name__)
        task.add_done_callback(done)
        return task

    def _bump(self) -> int:
        self.revision += 1
        self._watch_write()
        return self.revision

    async def _publish_state(self) -> None:
        snap = self.snapshot()
        self._watch_totals = snap.get("totals")
        await self._emit({"type": "state", "state": snap})

    # ------------------------------------------------------------------ watch page status (read by app.watch, usually in another process)
    WATCH_FILE = "watch.json"

    def _detect_progress(self) -> dict[str, int]:
        total = self.counters["keyframes"]
        done = sum(1 for v in self.stage_log.values() if isinstance(v, dict) and v.get("detect", {}).get("status") not in (None, "running"))
        return {"done": min(done, total), "total": total}

    def watch_status(self) -> dict[str, Any]:
        """Small, cheap, read-only view of the sweep for the laptop watch page: no snapshot build, no owner/session secrets, no absolute paths."""
        last = self.frames[-1] if self.frames else None
        t = self._watch_totals or {}
        return {"version": 1, "run_id": self.run_dir.name, "status": self.status, "closed": self._watch_closed, "pid": os.getpid(),
                "source": "+".join(sorted(self.sources)) or None, "started_epoch": round(self.started_epoch, 3),
                "ended_epoch": None if self._ended_epoch is None else round(self._ended_epoch, 3), "updated_epoch": round(time.time(), 3),
                "frames": len(self.frames), "latest_frame": last["frame_id"] if last else None,
                "latest_frame_t_s": last["timestamp_s"] if last else None,
                "frame_size": [last["width_px"], last["height_px"]] if last else None,
                "live_count": {k: t.get(k) for k in ("book_count", "item_count", "book_count_range", "count_reliability", "book_lines", "headline_estimator") if k in t} or None,
                "indicator": self.cq.indicator(), "distance_now": self.cq.distance_now, "room_phase": getattr(self.room_sweep, "phase", None),
                "notices": list(self._watch_nudges),
                "finalize": {**self._finalize_info(), "detect": self._detect_progress(),
                             "read_done": ({"attempted": self.read_report.get("attempted_observations", 0), "total": self.read_report.get("pass1_total"),
                                            "budget_exhausted": bool(self.read_report.get("budget_exhausted"))} if self.read_report.get("pass1_total") else None)},
                "finalize_steps": list(self._watch_steps), "first_count": self._watch_first,
                "finalization_s": (self.finalization or {}).get("finalization_s")}

    async def _watch_heartbeat(self) -> None:
        """~1/s refresh of watch.json while recording or ending (title reading, pricing ... do not bump the revision). Not in self._tasks: end_sweep never waits for it."""
        while not self._closed and self.status in ("capturing", "finalizing"):
            await asyncio.sleep(1.0)
            self._watch_write()
        self._watch_write(force=True)

    def _watch_write(self, force: bool = False) -> None:
        if not getattr(self, "watch_file", False):
            return
        key = (self.status, self.finalize_stage, len(self._watch_nudges), self._watch_closed, self._watch_first is not None)
        now = time.monotonic()
        if not force and key == self._watch_key and now - self._watch_t < 1.0:
            return
        self._watch_key, self._watch_t = key, now
        try:
            data = json.dumps(self.watch_status(), default=str)
            tmp = self.run_dir / (self.WATCH_FILE + ".tmp")
            tmp.write_text(data)
            tmp.replace(self.run_dir / self.WATCH_FILE)
        except Exception:  # noqa: BLE001 - the watch page must never affect the sweep
            logging.getLogger(__name__).debug("watch status write failed", exc_info=True)

    # ------------------------------------------------------------------ ingestion
    def blur_gate_floor(self, source: str) -> float | None:
        """Sharpness floor for the blur gate on a frame of ``source`` (None = gate not applied). Uses the history INCLUDING the current frame."""
        if self.blur_gate == "off" or (self.blur_gate == "live" and source != "camera"):
            return None
        if len(self._sharp) < self.blur_gate_min_history:
            return self.blur_gate_abs
        median = float(np.median(list(self._sharp)))
        return max(self.blur_gate_abs, self.blur_gate_ratio * median)

    def set_capture_info(self, info: dict[str, Any]) -> None:
        """Device/stream facts the phone reports (label, resolution, fps, orientation, client counters). Data only: strings are clipped, nothing is trusted for control."""
        clean: dict[str, Any] = {}
        for k, v in (info or {}).items():
            if k in ("type", "session_id"):
                continue
            if isinstance(v, str):
                clean[str(k)[:40]] = v[:200]
            elif isinstance(v, (bool, int, float)) or v is None:
                clean[str(k)[:40]] = v
            elif isinstance(v, dict):
                clean[str(k)[:40]] = {str(a)[:40]: (b[:200] if isinstance(b, str) else b) for a, b in list(v.items())[:60] if isinstance(b, (str, int, float, bool)) or b is None}
        self.capture_info.update(clean)
        self._bump()

    def capture_quality(self) -> dict[str, Any]:
        """The capture-quality summary plus comparable per-sweep capture stats (median sharpness, frame-gap p90, device label, blur-gate effect)."""
        gate = {"mode": self.blur_gate, "ratio": self.blur_gate_ratio, "abs_floor": self.blur_gate_abs, "min_history": self.blur_gate_min_history,
                "frames_gated": self.dense["gated_frames"], "frames_total": len(self.frames),
                "share_gated": round(self.dense["gated_frames"] / len(self.frames), 3) if self.frames else None,
                "gated_frame_ids": self._gated_order[:300]}
        return {**self.cq.summary(), **capture_stats(self.frames, self.capture_info, gate, self.sources)}

    async def ingest_frame(self, jpeg: bytes, timestamp_s: float | None = None, source: str = "camera", client: dict[str, Any] | None = None) -> dict[str, Any]:
        """The single frame-ingestion path. Returns {frame_id, timestamp_s, reasons, nudges, closed_keyframes}."""
        ingest_t0 = time.perf_counter()
        if self.status != "capturing":
            raise SweepError(f"sweep is {self.status}; not accepting frames")
        if not isinstance(jpeg, (bytes, bytearray)) or len(jpeg) < MIN_FRAME_BYTES or not jpeg.startswith(b"\xff\xd8\xff"):
            raise ValueError("evidence frames must be JPEG images")
        if len(self.frames) >= MAX_FRAMES:
            raise SweepError("evidence frame limit reached; end the sweep")
        async with self._ingest_lock:
            arr = np.frombuffer(bytes(jpeg), np.uint8)
            bgr = await asyncio.to_thread(cv2.imdecode, arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("JPEG could not be decoded")
            now_t = round(time.monotonic() - self.started, 3)
            t = round(float(timestamp_s), 3) if (source == "replay" and timestamp_s is not None) else now_t
            if self.frames and t <= self.frames[-1]["timestamp_s"]:
                t = round(self.frames[-1]["timestamp_s"] + 0.001, 3)
            h, w = bgr.shape[:2]
            qscale = QUALITY_LONG_SIDE / max(h, w) if max(h, w) > 1.05 * QUALITY_LONG_SIDE else 1.0
            qbgr = bgr if qscale >= 1.0 else await asyncio.to_thread(cv2.resize, bgr, (round(w * qscale), round(h * qscale)), interpolation=cv2.INTER_AREA)
            qh, qw = qbgr.shape[:2]                  # quality frame: shift px, sharpness and glare below are in these pixels (== native when the frame is <= 1920 long)
            if self.selector is None:
                self.selector = PanIntervalSelector(self.overlap_step, float(self.config.get("overlap_lost_step", 0.6)), qw)
            gray = await asyncio.to_thread(cv2.cvtColor, qbgr, cv2.COLOR_BGR2GRAY)
            dt = None if self._prev_t is None else t - self._prev_t
            client_clean = {k: client[k] for k in ("t_s", "cand", "sharp_small", "seen", "win_ms", "enc_ms", "q") if isinstance(client.get(k), (int, float)) and not isinstance(client.get(k), bool)
                            and math.isfinite(client[k])} if client else {}
            cap_t = client_clean.get("t_s")
            if cap_t is not None and self._prev_cap_t is not None and cap_t > self._prev_cap_t:
                dt = cap_t - self._prev_cap_t       # px/s and zoom/s use the phone's capture clock: best-of-window frames are not evenly spaced on the arrival clock
            self._prev_cap_t = cap_t
            q, small = await asyncio.to_thread(frame_quality, qbgr, gray, self.qcfg, self._prev_small, dt)
            self._prev_small, self._prev_t = small, t
            self._sharp.append(q["sharpness"])
            _, floor = sharpness_floor(list(self._sharp), self.qcfg)
            q["reasons"] = reasons_for(q, floor if len(self._sharp) >= 5 else self.qcfg.blur_floor_abs, self.qcfg)
            gate_floor = self.blur_gate_floor(source)
            if gate_floor is not None:
                q["blur_floor"] = round(gate_floor, 1)
                q["blur_gated"] = bool(q["sharpness"] < gate_floor)
            fid = C.frame_id(len(self.frames))
            path = self.run_dir / "frames" / f"{fid}.jpg"
            path.write_bytes(bytes(jpeg))  # exact bytes received: no re-encode, no downscale
            rec = {"frame_id": fid, "timestamp_s": t, "native_frame_index": len(self.frames),
                   "image_path": C.rel_path(path, self.run_dir), "width_px": w, "height_px": h, "quality": q,
                   "selected": False, "selection_reason": "", "source": source, "size_bytes": len(jpeg)}
            if client:
                rec["client"] = client_clean
            if qscale < 1.0:
                rec["quality_scale"] = round(qscale, 4)     # quality px = native px * quality_scale (see QUALITY_LONG_SIDE)
            self.frames.append(rec)
            self.sources.add(source)
            self.counters["frames"] += 1
            if q.get("shift_response") is not None and "motion_estimate_unreliable" not in q["reasons"] and self._cum:
                step = float(np.hypot(q["shift_dx_px"], q["shift_dy_px"])) / qw
            else:
                step = 0.0 if not self._cum else self.overlap_step * 0.5
            self._cum.append((self._cum[-1] if self._cum else 0.0) + step)
            closed = self.selector.feed(rec)
            overlap_gaps = sum(g["type"] == "overlap_lost" for g in self.selector.gaps)
            nudges = self.monitor.frame(q["reasons"], t, overlap_gaps)
            if q.get("shift_response") is not None and "motion_estimate_unreliable" not in q["reasons"] and len(self.frames) > 1:
                nudges += self.monitor.motion(q["shift_dx_px"] / qw, q["shift_dy_px"] / qh, t)    # capture clock (== now_t for camera frames)
            nudges += self.cq.add(t, q.get("motion_px_per_s"), q["sharpness"], q.get("zoom_rate_per_s"))
            if source == "camera" and h > w and len(self.frames) >= 10 and self.nudge_policy.offer("portrait", t):
                self._portrait_nudged = True       # the pipeline was tuned on landscape 1920x1080; upright 1080x1920 gives each book ~56% of the sightings
                nudges.append({"code": "portrait", "message": CQ_MESSAGES["portrait"]})
            self._start_closed(closed)
            self._dense_pick(rec)
            self._bump()
        if self.watch_file and self._watch_task is None:
            self._watch_task = asyncio.ensure_future(self._watch_heartbeat())
        for n in nudges:
            await self._add_nudge(n, fid)
        if self.counters["frames"] % 5 == 0:
            await self._guidance(t)
        await self._emit({"type": "frame", "frame_id": fid, "timestamp_s": t, "reasons": q["reasons"],
                          "keyframes_closed": [c["selected_frame_id"] for c in closed]})
        if closed:
            await self._publish_state()
        rec["ingest_ms"] = round(1000 * (time.perf_counter() - ingest_t0), 1)   # whole call incl. decode, quality, write, selector, nudges and any state build (the receive loop is serial)
        return {"frame_id": fid, "timestamp_s": t, "reasons": q["reasons"], "nudges": nudges,
                "closed_keyframes": [c["selected_frame_id"] for c in closed]}

    async def _add_nudge(self, nudge: dict[str, Any], frame_id: str | None) -> None:
        entry = {**nudge, "frame_id": frame_id, "at_s": round(time.monotonic() - self.started, 2)}
        self.nudges.append(entry)
        self.nudges = self.nudges[-20:]
        self._watch_nudges = (self._watch_nudges + [{k: entry.get(k) for k in ("code", "message", "kind", "at_s", "frame_id")}])[-100:]
        self._watch_write()
        await self._emit({"type": "nudge", **entry})

    # ------------------------------------------------------------------ keyframe processing
    def _start_closed(self, closed: list[dict[str, Any]]) -> None:
        for c in closed:
            self._fill_overlap(c["frame"])
            self._start_keyframe(c["frame"])
            self._last_det_idx = c["frame"]["native_frame_index"]

    def _fill_overlap(self, key: dict[str, Any]) -> None:
        """Detect extra saved frames between the previous detection frame and ``key`` so adjacent detection frames overlap >= min_overlap.
        Overlap is estimated from accumulated phase-correlation shift (1 - shift/width); greedy, prefers unflagged sharp frames, capped by budget.
        A single sampled-frame step that already exceeds the limit cannot be fixed by detecting more frames and is reported, not hidden."""
        if self.min_overlap <= 0 or self._last_det_idx is None:
            return
        k, limit = key["native_frame_index"], 1.0 - self.min_overlap
        pos, lo = self._last_det_idx, self._last_det_idx
        while self._cum[k] - self._cum[pos] > limit:
            cand = [i for i in range(pos + 1, k) if self._cum[i] - self._cum[pos] <= limit * 0.9]
            if not cand:      # next sampled frame is already further than the limit
                self.overlap_fill["unfillable_gaps"].append({"from": self.frames[pos]["frame_id"], "to": self.frames[min(pos + 1, k)]["frame_id"],
                                                             "shift_widths": round(self._cum[min(pos + 1, k)] - self._cum[pos], 2)})
                pos = min(pos + 1, k)
                if pos >= k: break
                continue
            far = [i for i in cand if self._cum[i] - self._cum[pos] >= limit * 0.5] or [cand[-1]]
            ok = [i for i in far if not set(self.frames[i]["quality"]["reasons"]) & {"blurry", "fast_motion", "glare"} and not self.frames[i]["quality"].get("blur_gated")] or far
            pick = max(ok, key=lambda i: self.frames[i]["quality"]["sharpness"])
            if len(self.overlap_fill["extra_frames"]) >= self.extra_detect_max:
                self.overlap_fill["budget_exhausted"] = True
                return
            f = self.frames[pick]
            f["selected"], f["selection_reason"] = True, f"added for detection overlap (>= {self.min_overlap:.0%} with neighbouring detection frames)"
            self.overlap_fill["extra_frames"].append(f["frame_id"])
            self.counters["extra_keyframes"] += 1
            self._start_keyframe(copy.deepcopy(f))
            pos = pick

    def _dense_pick(self, rec: dict[str, Any]) -> None:
        """Policy 'dense': detect every saved frame that is not clearly blurred, at most detect_max_fps per second of capture time and within a budget.
        Detection runs in background tasks (bounded by detect_concurrency), so capture is never blocked; the backlog is reported."""
        if self.detect_policy != "dense" or rec["frame_id"] in self._detected_ids:
            return
        d = self.dense
        if self._gated(rec):      # checked before the rate slot is used, so the next frame can take it
            return
        if rec["timestamp_s"] - self._last_dense_t < 1.0 / self.detect_max_fps - 1e-6:
            d["skipped_rate"] += 1; return
        if d["dense_frames"] >= self.detect_dense_max:
            d["budget_exhausted"] = True; return
        self._last_dense_t = rec["timestamp_s"]
        d["dense_frames"] += 1
        rec["selected"] = True
        rec["selection_reason"] = rec["selection_reason"] or "detected by dense policy (not blurry; rate-capped)"
        self._start_keyframe(copy.deepcopy(rec))

    def _gated(self, frame: dict[str, Any]) -> bool:
        """Blur gate: the frame stays saved (evidence, panorama) but is not sent to detection. Counted once per frame."""
        if not frame["quality"].get("blur_gated"):
            return False
        if frame["frame_id"] not in self._gated_ids:
            self._gated_ids.add(frame["frame_id"])
            self.dense["gated_frames"] += 1
            self.dense["skipped_blurry"] += 1
            self._gated_order.append(frame["frame_id"])
            frame_rec = self.frames[frame["native_frame_index"]]
            frame_rec["selection_reason"] = frame_rec["selection_reason"] or "stored, not detected: below the blur gate floor"
        return True

    def _start_keyframe(self, frame: dict[str, Any]) -> None:
        if frame["frame_id"] in self._detected_ids:     # already queued (dense policy and the pan-interval selector can pick the same frame)
            return
        if self._gated(frame):
            return
        self._detected_ids.add(frame["frame_id"])
        self.counters["keyframes"] += 1
        self._spawn(self._process_keyframe(copy.deepcopy(frame)), f"keyframe:{frame['frame_id']}")

    def _mini_manifest(self, frame: dict[str, Any], name: str) -> Path:
        env = C.envelope(self.run_id, "frames", {
            "input": {"file_name": "live-session", "sha256": None, "source": frame.get("source")},
            "frames": [{**frame, "selected": True}], "intervals": [], "capture_gaps": [], "audit": {}}, status="ok")
        return C.write_json(self.run_dir / "live" / "manifests" / f"{name}.json", env)

    async def _process_keyframe(self, frame: dict[str, Any]) -> None:
        fid = frame["frame_id"]
        rec = self.stage_log.setdefault(fid, {})
        try:
            mini = self._mini_manifest(frame, fid)
            (self.run_dir / "live" / "detect").mkdir(parents=True, exist_ok=True)
            async with self._detect_sem:
                rec["detect"] = {"status": "running"}
                cfg = {**self.stage_configs["detect"], "output_name": f"live/detect/{fid}.json"}
                t0 = time.perf_counter()
                path = await asyncio.to_thread(self.stages.detect, mini, self.run_dir, cfg)
                art = C.read_json(path)
            if self._closed: return
            new_obs = [o for o in (art.get("data") or {}).get("observations", []) if o["observation_id"] not in {x["observation_id"] for x in self.observations}]
            self.observations.extend(new_obs)
            wn = self.cq.add_widths(frame["timestamp_s"], [GR.spine_width_px(o) for o in new_obs if o.get("kind") == "book"])
            for n in wn:
                await self._add_nudge(n, fid)
            self.stage_issues.extend({**i, "stage": "detect"} for i in art.get("issues", []) if isinstance(i, dict))
            rec["detect"] = {"status": art.get("status"), "elapsed_s": round(time.perf_counter() - t0, 3),
                             "observations": len(new_obs), "fixture": bool(art.get("fixture")),
                             "local_inference_count": sum(r.get("local_inference_count", 0) for r in art.get("metrics", {}).get("requests", []))}
            self._bump()
            await self._publish_state()
            books = [o for o in new_obs if o.get("kind") == "book"]
            g = self.room_sweep.observe_detection(frame["timestamp_s"], fid, len(books), [long_side_px(o) for o in books])
            if g:
                await self._start_room_phase(g, frame["timestamp_s"], fid, relay=True)
            self._schedule_merge()
            if books and self.read_policy == "observation":
                await self._read_resolve(frame, books)
            self._schedule_merge()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal to the sweep
            rec["detect"] = {"status": "failed", "detail": f"{type(exc).__name__}: {str(exc)[:200]}"}
            self.stage_issues.append({"frame_id": fid, "stage": "detect", "reason": rec["detect"]["detail"]})
            self._bump()
            await self._publish_state()

    async def _read_resolve(self, frame: dict[str, Any], books: list[dict[str, Any]]) -> None:
        fid = frame["frame_id"]
        rec = self.stage_log.setdefault(fid, {})
        try:
            mini = self._mini_manifest(frame, fid)
            slice_path = C.write_json(self.run_dir / "live" / "read" / f"{fid}.candidates.json", C.envelope(
                self.run_id, "detect", {"observations": books}, status="ok"))
            async with self._read_sem:
                rec["read"] = {"status": "running"}
                t0 = time.perf_counter()
                cfg = {**self.stage_configs["read"], "session_revision": self.revision, "readings_filename": f"live/read/{fid}.readings.json"}
                rpath = await asyncio.to_thread(self.stages.read, slice_path, mini, self.run_dir, cfg)
                rart = C.read_json(rpath)
            if self._closed: return
            new_readings = (rart.get("data") or {}).get("readings", [])
            self.readings.extend(new_readings)
            rec["read"] = {"status": rart.get("status"), "elapsed_s": round(time.perf_counter() - t0, 3), "readings": len(new_readings),
                           "local_inference_count": sum(m.get("local_inference_count", 0) for m in rart.get("metrics", {}).get("provider_metrics", []))}
            self.stage_issues.extend({**i, "stage": "read"} for i in rart.get("issues", []) if isinstance(i, dict))
            for r in new_readings:
                for n in self.monitor.reading(bool(r.get("visible_text")), frame["timestamp_s"]):
                    await self._add_nudge(n, fid)
            t0 = time.perf_counter()
            rcfg = {**self.stage_configs["resolve"], "session_revision": self.revision, "resolutions_filename": f"live/read/{fid}.resolutions.json"}
            spath = await asyncio.to_thread(self.stages.resolve, rpath, self.run_dir, rcfg)
            sart = C.read_json(spath)
            if self._closed: return
            self.resolutions.extend((sart.get("data") or {}).get("resolutions", []))
            rec["resolve"] = {"status": sart.get("status"), "elapsed_s": round(time.perf_counter() - t0, 3)}
            self._bump()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            rec["read"] = {"status": "failed", "detail": f"{type(exc).__name__}: {str(exc)[:200]}"}
            self.stage_issues.append({"frame_id": fid, "stage": "read", "reason": rec["read"]["detail"]})

    # ------------------------------------------------------------------ group-level reading (after the final association)
    async def _read_groups(self) -> bool:
        """Read the best 1-2 sightings of every physical group, in parallel, within a wall-clock budget. Returns True if any reading was added."""
        books = (self.inventory or {}).get("data", {}).get("books", [])
        rr = self.read_report
        if not books:
            rr["status"] = "no_groups"; return False
        obs = {o["observation_id"]: o for o in self.observations}
        frames = {f["frame_id"]: f for f in self.frames}
        gray_cache: dict[str, Any] = {}
        def load_gray(fid: str):
            if fid not in gray_cache:
                gray_cache[fid] = cv2.imread(str(self.run_dir / frames[fid]["image_path"]), cv2.IMREAD_GRAYSCALE)
            return gray_cache[fid]
        t0 = time.monotonic()
        plan = await asyncio.to_thread(GR.plan, books, obs, frames, load_gray, per_group=3 if self.stage_configs.get("read2") else 2)
        gray_cache.clear()
        done_obs: dict[str, dict[str, Any]] = dict(self._prov_done)      # readings made during capture count as done
        self.prov_report["status"] = "done"
        rr["provisional"] = dict(self.prov_report)
        todo1 = [(g["instance_id"], g["sightings"][0]) for g in plan["groups"] if g["status"] == "attempt" and g["sightings"][0]["observation_id"] not in done_obs]
        rr["groups_already_read_during_capture"] = sum(1 for g in plan["groups"] if g["status"] == "attempt" and g["sightings"][0]["observation_id"] in done_obs)
        # expected-yield order: wide and sharp first (width x sharpness). Edge energy was tried as a factor but does not predict readability
        # on the dev set (AUC 0.39); it is only used to skip near-flat crops.
        todo1.sort(key=lambda t: t[1]["score"], reverse=True)
        todo2 = {g["instance_id"]: g["sightings"][1] for g in plan["groups"] if g["status"] == "attempt" and len(g["sightings"]) > 1}
        since_end = (time.monotonic() - self.ended) if self.ended else 0.0
        reserve = self._remerge_reserve_s()
        budget = max(30.0, min(self.read_budget_s, self.packet_target_s - since_end - reserve))     # adaptive: keep the packet under the target when possible
        rr.update(status="running", plan_counts=plan["counts"], thresholds=plan["thresholds"], pass1_total=len(todo1), pass2_candidates=len(todo2),
                  attempted_observations=0, queued=len(todo1), budget_exhausted=False, plan_s=round(time.monotonic() - t0, 2),
                  budget_s=round(budget, 1), remerge_reserve_s=round(reserve, 1),
                  budget_rule=f"clamp(packet_target {self.packet_target_s:.0f}s - {since_end:.0f}s already spent since sweep end - {reserve:.0f}s reserved for the re-merge "
                              f"(LIBRARY_REMERGE_RESERVE {self.remerge_reserve:g} x last merge), 30s, LIBRARY_READ_BUDGET_S {self.read_budget_s:.0f}s)")
        deadline = t0 + budget
        manifest = C.write_json(self.run_dir / C.ARTIFACTS["frames"], self._manifest_env())
        hosted_read = (self.stage_availability.get("read") or {}).get("selection") == "gemini"
        # hosted reads are I/O-bound: their own concurrency (LIBRARY_HOSTED_READ_CONCURRENCY, 16), never the CPU reader pool size
        workers = int(self.config.get("hosted_read_concurrency", os.environ.get("LIBRARY_HOSTED_READ_CONCURRENCY", 16))) if hosted_read else self.read_workers
        rr["workers"] = workers
        rr["hosted"] = hosted_read
        sem = asyncio.Semaphore(workers)
        # Catalogue lookups are paced process-wide (providers.openlibrary: ~1 request/s), so they are the scarce resource of the packet: resolution runs OUTSIDE the read
        # semaphore (reads keep their full concurrency) as a priority queue of single readings served by a small worker pool until the same wall-clock deadline. Readings that
        # carry a complete title and a visible author go first (they produced 84% of the accepts on the YT2 dev run from 30% of the readings); whatever the deadline does not
        # reach is counted as skipped, never guessed.
        resolve_workers = max(1, int(self.config.get("resolve_workers", os.environ.get("LIBRARY_RESOLVE_WORKERS", 2))))
        rv = rr["resolve"] = {"workers": resolve_workers, "queued": 0, "readings_resolved": 0, "readings_skipped_budget": 0, "failed": 0, "accepted": 0}
        resolve_q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        job_seq = itertools.count()
        chunk_no = 0

        # Work-level pricing of freshly identified works starts while reading continues (its HTTP is idle time otherwise): a one-worker prefetch warms the shared price cache
        # (price_config_from_env: runs/_price_cache), so the final pricing step finds those works already cached and spends its own budget on the rest.
        pf = rr["price_prefetch"] = {"enabled": False, "queued": 0, "works_priced": 0, "works_attempted": 0, "skipped_budget": 0, "errors": 0}
        pf_queue: asyncio.Queue = asyncio.Queue()
        pf_deadline = [deadline]
        pf_max = int(self.config.get("price_prefetch_max", os.environ.get("LIBRARY_PRICE_PREFETCH_MAX", 30)))
        pf_seen: set[str] = set()
        pf["enabled"] = pf_max > 0 and str(os.environ.get("LIBRARY_PRICE_ONLINE", "1")) != "0" and bool(self.config.get("catalogue_online"))

        def pf_feed(resolutions: list[dict[str, Any]]) -> None:
            if not pf["enabled"]:
                return
            for r in resolutions:
                work = r.get("work") or ""
                if r.get("decision") != "accepted" or not re.fullmatch(r"/works/OL\d+W", work) or pf["queued"] >= pf_max:
                    continue
                if work in pf_seen:
                    continue
                pf_seen.add(work); pf["queued"] += 1
                # language_code stays None exactly as in the final pricing step (the reading's language guess is not price evidence), so the cache keys line up
                pf_queue.put_nowait({"id": r["observation_id"], "work": work, "title": r.get("title"), "visible_text": r.get("visible_text"), "language_code": None})

        pf_state: dict[str, Any] = {}

        def pf_run(w: dict[str, Any]) -> None:      # blocking: runs in a worker thread, one work at a time, one client pair for the whole phase (pacing is per client)
            from pipeline.price import PublicBookPrices, price_config_from_env, price_work
            from providers.openlibrary import OpenLibrary
            if "cfg" not in pf_state:
                cfg = {**self.config, **self.market, **price_config_from_env(self.run_dir)}
                cache_dir = Path(cfg["price_cache_dir"])
                pf_state.update(cfg=cfg, catalogue=OpenLibrary(cache_dir / "openlibrary", online=bool(cfg.get("pricing_online"))),
                                provider=PublicBookPrices(self.run_dir / "pricing_evidence", online=bool(cfg.get("pricing_online")), cache_dir=cache_dir / "abebooks",
                                                          max_age_hours=cfg.get("max_age_hours", 24 * 30), min_interval_s=cfg.get("min_interval_s", 2.0)))
            if time.monotonic() >= pf_deadline[0]:
                pf["skipped_budget"] += 1; return
            pf["works_attempted"] += 1
            try:
                row = price_work(w, self.run_dir.resolve(), pf_state["catalogue"], pf_state["provider"], pf_state["cfg"], pf_deadline[0])
                pf["works_priced"] += bool(row.get("replacement_cost") or row.get("used_value"))
            except Exception:  # noqa: BLE001 - a prefetch failure only costs the warm cache
                pf["errors"] += 1

        async def pf_worker() -> None:
            try:
                while True:
                    w = await pf_queue.get()
                    if w is None:
                        return
                    await asyncio.to_thread(pf_run, w)
            finally:
                if "catalogue" in pf_state:
                    pf_state["catalogue"].close()

        pf_task = asyncio.create_task(pf_worker()) if pf["enabled"] else None

        def resolve_priority(reading: dict[str, Any]) -> tuple[int, int]:
            text = str(reading.get("visible_text") or "")
            complete = bool(reading.get("title_complete") and reading.get("title_text"))
            tier = 0 if complete and reading.get("author_text") else 1 if complete else 2 if len(text) >= 8 else 3
            return (tier, -len(text))

        def enqueue_resolve(readings: list[dict[str, Any]]) -> None:
            for r in readings:
                rv["queued"] += 1
                resolve_q.put_nowait((resolve_priority(r), next(job_seq), r))

        async def resolve_worker() -> None:
            while True:
                _prio, seq, reading = await resolve_q.get()
                if reading is None:
                    return
                if time.monotonic() > deadline:
                    rv["readings_skipped_budget"] += 1; continue
                art = C.write_json(self.run_dir / "live" / "read" / f"grp_res_{seq:05d}.readings.json", C.envelope(self.run_id, "read", {"readings": [reading]}, status="ok"))
                rcfg = {**self.stage_configs["resolve"], "session_revision": self.revision, "resolutions_filename": f"live/read/grp_res_{seq:05d}.resolutions.json",
                        "deadline_monotonic": deadline}
                try:
                    sart = C.read_json(await asyncio.to_thread(self.stages.resolve, art, self.run_dir, rcfg))
                except Exception as exc:  # noqa: BLE001 - recorded per reading
                    rv["failed"] += 1
                    self.stage_issues.append({"stage": "resolve", "reason": f"group resolve of {reading.get('observation_id')} failed: {type(exc).__name__}: {str(exc)[:160]}"})
                    continue
                rs = (sart.get("data") or {}).get("resolutions", [])
                for r in rs:
                    if "resolve_budget_exhausted" in " ".join(map(str, r.get("issues") or [])):
                        rv["readings_skipped_budget"] += 1
                    else:
                        rv["readings_resolved"] += 1
                        rv["accepted"] += r.get("decision") == "accepted"
                self.resolutions.extend(rs)
                pf_feed(rs)

        resolve_tasks = [asyncio.create_task(resolve_worker()) for _ in range(resolve_workers)]

        async def run_chunk(items: list[tuple[str, dict[str, Any]]], tag: str) -> None:
            nonlocal chunk_no
            async with sem:
                if time.monotonic() > deadline:
                    rr["budget_exhausted"] = True; return
                chunk_no += 1
                n = chunk_no
                ids = [it[1]["observation_id"] for it in items]
                slice_path = C.write_json(self.run_dir / "live" / "read" / f"grp_{tag}_{n:04d}.candidates.json", C.envelope(
                    self.run_id, "detect", {"observations": [obs[i] for i in ids]}, status="ok"))
                cfg = {**self.stage_configs["read"], "session_revision": self.revision, "readings_filename": f"live/read/grp_{tag}_{n:04d}.readings.json", "max_books": None}
                try:
                    rpath = await asyncio.to_thread(self.stages.read, slice_path, manifest, self.run_dir, cfg)
                    rart = C.read_json(rpath)
                except Exception as exc:  # noqa: BLE001 - recorded per chunk
                    self.stage_issues.append({"stage": "read", "reason": f"group read chunk {tag}{n} failed: {type(exc).__name__}: {str(exc)[:160]}"})
                    return
                for r in (rart.get("data") or {}).get("readings", []):
                    done_obs[r["observation_id"]] = r
                self.readings.extend((rart.get("data") or {}).get("readings", []))
                self.stage_issues.extend({**i, "stage": "read"} for i in rart.get("issues", []) if isinstance(i, dict))
                rr["attempted_observations"] += len(ids)
                await self._emit({"type": "finalize", **self._finalize_info()})
                rr["queued"] = max(0, rr["queued"] - len(ids))
                enqueue_resolve((rart.get("data") or {}).get("readings", []))
        try:
            size = 3
            await asyncio.gather(*[run_chunk(todo1[i:i + size], "a") for i in range(0, len(todo1), size)])
            # second sighting only where the first gave nothing useful (saves OCR time; the group keeps every raw reading)
            first = {g["instance_id"]: g["sightings"][0]["observation_id"] for g in plan["groups"] if g["status"] == "attempt"}
            weak = [(iid, sg) for iid, sg in todo2.items() if len(GR.reading_text(done_obs.get(first[iid], {}))) < 8]
            rr["pass2_total"] = len(weak)
            rr["queued"] = len(weak)
            await asyncio.gather(*[run_chunk(weak[i:i + size], "b") for i in range(0, len(weak), size)])
            await self._read_second(plan, books, done_obs, obs, manifest)
            for _ in resolve_tasks:                    # one stop signal per worker, sorted after every real job; jobs past the deadline are skipped in O(1)
                resolve_q.put_nowait(((99, 0), next(job_seq), None))
            await asyncio.gather(*resolve_tasks)
            if pf_task is not None:                    # reading is over: the prefetch gets a short grace period, then its remaining works are dropped (they price at finalize)
                pf_deadline[0] = min(pf_deadline[0], time.monotonic() + float(self.config.get("price_prefetch_grace_s", 15)))
                pf_queue.put_nowait(None)
                try:
                    await asyncio.wait_for(asyncio.shield(pf_task), timeout=pf_deadline[0] - time.monotonic() + 5)
                except asyncio.TimeoutError:
                    pf["skipped_budget"] += pf_queue.qsize()
        finally:                                   # an exception or cancellation (finalize timeout) must not leave idle workers behind
            for t in [*resolve_tasks, *([pf_task] if pf_task is not None else [])]:
                if not t.done():
                    t.cancel()
        attempted_groups = {g["instance_id"] for g in plan["groups"] if g["status"] == "attempt" and any(s["observation_id"] in done_obs for s in g["sightings"])}
        unread = [g for g in plan["groups"] if g["status"] == "attempt" and g["instance_id"] not in attempted_groups]
        for g in unread:
            g["status"], g["reason"] = "unattempted_budget", f"wall-clock read budget {budget:.0f}s reached before this group"
        self.stage_log["_group_read"] = {"read": {"status": "ok" if not unread else "partial", "elapsed_s": round(time.monotonic() - t0, 3), "readings": len(done_obs),
                                                   "local_inference_count": len(done_obs), "detail": f"group policy: {len(attempted_groups)} groups read, {len(unread)} unread (budget)"},
                                         "resolve": {"status": "ok" if not (rv["readings_skipped_budget"] or rv["failed"]) else "partial", "elapsed_s": round(time.monotonic() - t0, 3),
                                                     "detail": f"catalogue: {rv['readings_resolved']} readings resolved ({rv['accepted']} accepted), {rv['readings_skipped_budget']} not reached (budget), {rv['failed']} failed"}}
        rr.update(status="done", elapsed_s=round(time.monotonic() - t0, 2), queued=0, groups_total=len(books), groups_attempted=len(attempted_groups),
                  groups_unattempted={"too_small": plan["counts"]["unattempted_too_small"], "low_contrast": plan["counts"]["unattempted_low_contrast"],
                                      "no_sighting": plan["counts"]["unattempted_no_sighting"], "budget": len(unread)},
                  groups_with_text=sum(1 for g in books if GR.combine(g, done_obs)["best"]),
                  groups_with_text_ge8=sum(1 for g in books if len((GR.combine(g, done_obs)["best"] or {"text": ""})["text"]) >= 8))
        rr["yield_by_width_px"] = GR.yield_by_width(plan["groups"], books, done_obs)
        C.write_json(self.run_dir / "group_reading_plan.json", C.envelope(self.run_id, "read", {"plan": plan, "report": {k: v for k, v in rr.items()}}, status="ok"))
        return bool(done_obs)

    async def _read_second(self, plan: dict[str, Any], books: list[dict[str, Any]], done_obs: dict[str, dict[str, Any]], obs: dict[str, dict[str, Any]],
                           manifest: Path) -> None:
        """Optional second-pass reader (LIBRARY_READER_SECOND): reads up to 3 sightings of groups where the primary reader produced < 8 chars, within its
        own wall-clock budget. Second-pass readings are kept apart (self.readings2) and never replace the primary readings or feed catalogue resolution."""
        rr = self.read_report
        av = self.stage_availability.get("read2")
        cfg2 = self.stage_configs.get("read2")
        if av is None:
            rr["second_reader"] = {"status": "not_configured"}
            return
        if not av.get("available") or cfg2 is None:
            rr["second_reader"] = {"status": "unavailable", "name": av.get("selection"), "reason": av.get("reason")}
            return
        t0 = time.monotonic()
        budget = float(self.config.get("read_second_budget_s", os.environ.get("LIBRARY_READ_SECOND_BUDGET_S", 90)))   # background pass; packet time is reported honestly
        workers = int(self.config.get("florence_workers") or os.environ.get("LIBRARY_FLORENCE_WORKERS") or 2)
        by_group = {g["instance_id"]: g for g in plan["groups"]}
        weak = []
        for b in books:
            g = by_group.get(b["instance_id"])
            if not g or g["status"] != "attempt":
                continue
            best = GR.combine(b, done_obs)["best"]
            if not best or len(best["text"]) < 8:
                weak.append((g["sightings"][0]["score"], g))
        weak.sort(key=lambda t: t[0], reverse=True)
        stats = {"status": "running", "name": av["selection"], "groups_considered": len(weak), "groups_attempted": 0, "observations_read": 0, "budget_s": budget,
                 "budget_exhausted": False, "groups_new_text_ge8": 0, "label": "literal reading, unverified"}
        rr["second_reader"] = stats
        done2: dict[str, dict[str, Any]] = {}
        sem = asyncio.Semaphore(workers)
        counter = 0
        async def one(g: dict[str, Any], s: dict[str, Any], rank: int) -> None:
            nonlocal counter
            async with sem:
                if time.monotonic() - t0 > budget:
                    stats["budget_exhausted"] = True; return
                counter += 1
                n = counter
                oid = s["observation_id"]
                slice_path = C.write_json(self.run_dir / "live" / "read" / f"grp2_{n:04d}.candidates.json", C.envelope(self.run_id, "detect", {"observations": [obs[oid]]}, status="ok"))
                cfg = {**cfg2, "session_revision": self.revision, "readings_filename": f"live/read/grp2_{n:04d}.readings.json", "max_books": None}
                try:
                    rpath = await asyncio.to_thread(self.stages.read, slice_path, manifest, self.run_dir, cfg)
                    rd = (C.read_json(rpath).get("data") or {}).get("readings", [])
                except Exception as exc:  # noqa: BLE001
                    self.stage_issues.append({"stage": "read", "reason": f"second-pass read failed ({type(exc).__name__}: {str(exc)[:140]})"})
                    return
                for r in rd:
                    done2[r["observation_id"]] = r
                    self.readings2.append(r)
                stats["observations_read"] += len(rd)
        def wants_more(g: dict[str, Any]) -> bool:
            # Recipe: sightings 2-3 only when the best view touches the frame edge or is >= 15% shorter than another sighting
            best = g["sightings"][0]
            if best.get("edge_touch"):
                return True
            L = lambda s: max(1.0, (obs[s["observation_id"]]["bbox_xyxy"][3] - obs[s["observation_id"]]["bbox_xyxy"][1]), (obs[s["observation_id"]]["bbox_xyxy"][2] - obs[s["observation_id"]]["bbox_xyxy"][0]))
            return any(L(best) <= 0.85 * L(s) for s in g["sightings"][1:])
        for rank in range(3):                  # best view of every weak group first; sightings 2-3 only for cut-off best views (recipe), budget permitting
            todo = [(g, g["sightings"][rank]) for _, g in weak if len(g["sightings"]) > rank and (rank == 0 or wants_more(g))]
            await asyncio.gather(*[one(g, s, rank) for g, s in todo])
        stats["recipe"] = "best view; +2 sightings only if the best view is edge-touching or >= 15% shorter than another sighting (v3)"
        read_groups = {g["instance_id"] for _, g in weak if any(x["observation_id"] in done2 for x in g["sightings"])}
        stats.update(status="done", groups_attempted=len(read_groups), elapsed_s=round(time.monotonic() - t0, 2),
                     groups_new_text_ge8=sum(1 for _, g in weak if any(len(GR.reading_text(done2.get(x["observation_id"], {}))) >= 8 for x in g["sightings"])))
        if self.readings2:
            C.write_json(self.run_dir / "readings_second.json", C.envelope(self.run_id, "read", {"readings": copy.deepcopy(self.readings2)}, status="ok",
                         metrics={"reader": av["selection"], "note": "second-pass literal readings, unverified; kept apart from readings.json"}))

    async def _room_auto_pass(self, room: dict[str, Any], manifest: dict[str, Any], spent_s: float) -> tuple[dict[str, Any], dict[str, Any]]:
        """Automatic wall evidence (pipeline.room_auto; classical/local) when the human supplied no usable room evidence. Runs ahead of
        room.py's result only to REPLACE an abstention; a human-evidence estimate is never overridden. Any ceiling height is a volunteered
        user-supplied value (nobody is asked for one). Low-confidence walls go to the review queue; abstentions carry reasons + guidance."""
        info: dict[str, Any] = {"enabled": str(self.config.get("room_auto", os.environ.get("LIBRARY_ROOM_AUTO", "1"))).lower() not in ("0", "false", "no"),
                                "used": False, "status": "skipped", "method": "auto_classical"}
        if not info["enabled"] or room.get("status") == "estimated":
            info["reason"] = "disabled" if not info["enabled"] else "human room evidence already gives an estimate"
            return room, info
        if spent_s + 30.0 > self.packet_target_s:
            info["reason"] = f"would push the packet past LIBRARY_PACKET_TARGET_S={self.packet_target_s:.0f}s"
            return room, info
        try:
            from pipeline import room_auto as RA
            anchor = self.meas.height_anchor or {}
            t0 = time.monotonic()
            res = await asyncio.to_thread(RA.estimate_from_frames, manifest["data"], self.run_dir / "frames", height_m=anchor.get("height_m"),
                                          utterance=(anchor.get("provenance") or {}).get("utterance"), max_frames=int(self.config.get("room_auto_max_frames", 24)))
            walls = (res.get("evidence") or {}).get("walls") or []
            low = [w for w in walls if isinstance(w.get("confidence"), (int, float)) and w["confidence"] < 0.5]
            info.update(status=res["room"].get("status"), elapsed_s=round(time.monotonic() - t0, 2), walls=len(walls), low_confidence_walls=[w.get("wall_id") for w in low],
                        per_wall=[{"wall_id": w.get("wall_id"), "frame_id": w.get("frame_id"), "confidence": w.get("confidence"), "source": (w.get("provenance") or {}).get("source")} for w in walls],
                        abstention_reasons=res.get("abstention_reasons"), capture_guidance=res.get("capture_guidance"), provenance_note=res.get("provenance_note"),
                        height_anchor="user-stated" if anchor else None, validation="software_only_unvalidated: no tape-measured room truth exists")
            if res["room"].get("status") == "estimated":
                out = dict(res["room"])
                out["method"] = f"{out.get('method') or 'room'} + auto_classical walls"
                out["auto_evidence"] = {"method": "auto_classical", "walls": info["per_wall"], "low_confidence_walls": info["low_confidence_walls"],
                                        "note": res.get("provenance_note"), "validation": info["validation"]}
                info["used"] = True
                self.stage_log["_room_auto"] = {"status": "ok", "elapsed_s": info["elapsed_s"], "detail": f"{len(walls)} auto walls, {len(low)} low-confidence"}
                return out, info
            room = {**room, "auto_attempt": {k: info.get(k) for k in ("status", "walls", "abstention_reasons", "capture_guidance", "elapsed_s")}}
        except Exception as exc:  # noqa: BLE001 - never fatal; abstain cleanly
            info.update(status="failed", reason=f"{type(exc).__name__}: {str(exc)[:160]}")
        return room, info

    async def _alternative_count(self, spent_s: float) -> dict[str, Any]:
        """Honest method uncertainty: re-run the association with association='track_v2' on the SAME candidates (CPU, ~20 s, off the read path)
        and report it beside the proposed 'track' count. Skipped when it would push the packet past LIBRARY_PACKET_TARGET_S."""
        alt_assoc = str(self.config.get("alt_association", os.environ.get("LIBRARY_ALT_ASSOCIATION", "track_v2")))
        out: dict[str, Any] = {"method": alt_assoc, "status": "skipped", "book_groups": None, "elapsed_s": None,
                               "note": "the two methods differ in how aggressively they link blurred sightings; the truth is most likely between them"}
        if not alt_assoc or alt_assoc == "none" or not merged_ok(self):
            out["reason"] = "disabled or no primary inventory"
            return out
        if spent_s + 40.0 > self.packet_target_s:
            out["reason"] = f"would push the packet past LIBRARY_PACKET_TARGET_S={self.packet_target_s:.0f}s ({spent_s:.0f}s already spent)"
            return out
        try:
            t0 = time.monotonic()
            manifest, cand, _, _ = self._write_stage_artifacts()
            cfg = {**self.stage_config, "output_name": f"inventory_{alt_assoc}.json", "association": alt_assoc,
                   "dedup": {**(self.stage_config.get("dedup") or {}), "max_gap_s": self.dedup_max_gap_s, "association": alt_assoc}}
            from pipeline import dedup as _dd
            path = await asyncio.to_thread(_dd.merge, cand, manifest, self.run_dir, cfg)
            inv = C.read_json(path)
            c = inv["data"]["counts"]
            out.update(status="ok", book_groups=c.get("book_count"), count_if_near_ties_merged=c.get("count_if_near_ties_merged"), item_count=c.get("item_count"),
                       elapsed_s=round(time.monotonic() - t0, 2), inventory=C.rel_path(Path(path), self.run_dir))
        except Exception as exc:  # noqa: BLE001 - never fatal; reported
            out.update(status="failed", reason=f"{type(exc).__name__}: {str(exc)[:160]}")
        return out

    def _remerge_reserve_s(self) -> float:
        """Seconds the read budget leaves for the re-merge after it: LIBRARY_REMERGE_RESERVE x the last (final) merge's duration (the re-merge redoes the same
        association; measured 0.81-1.08x on the YT1/YT2/YT10-13 packets)."""
        last = float((self.stage_log.get("_merge") or {}).get("elapsed_s") or 0.0)
        return max(0.0, self.remerge_reserve * last)

    def _accepted_map(self) -> dict[str, tuple[str, Any]]:
        """Everything of readings/resolutions that pipeline.dedup.merge acts on beyond literal text: each observation's ACCEPTED catalogue resolution (last record
        wins, as dedup._stage_records reads it) and the read confidence that picks among them. The association itself never reads either file."""
        res = {r.get("observation_id"): r for r in self.resolutions}
        conf = {r.get("observation_id"): r.get("read_confidence") for r in self.readings}
        return {str(oid): (json.dumps(r, sort_keys=True, default=str), conf.get(oid)) for oid, r in res.items()
                if r.get("decision") == "accepted" and r.get("matching_evidence") and r.get("source_url")}

    def _remerge_skippable(self) -> str | None:
        """Why the re-merge after the group reading can be replaced by an in-place identity refresh (None = it cannot). The re-merge recomputes the SAME
        association (same observations; association never reads readings/resolutions) and differs only through dedup._identity: identity/status/issues of book
        groups, plus demotions (a demoted spine with an accepted identity is kept as a book). So it is skippable when every observation whose accepted resolution
        changed sits in a book group made only of book observations: such a group was not demoted, and the demotion rule is monotone under the near-tie unions
        (count_if_near_ties_merged), so no demotion or count can change. _refresh_literal_identities enforces the remaining rule (an accepted group stays accepted
        with the same work). Needs the real dedup stage (dedup.merge, or cross_view.merge which calls it)."""
        from pipeline import dedup as _dd
        if not self.remerge_skip:
            return None
        if getattr(self.stages.merge, "__module__", "") not in (_dd.__name__, "pipeline.cross_view") or not self.inventory:   # both end in dedup.merge
            return None
        if {o["observation_id"] for o in self.observations} != set(self.merged_observation_ids) or self._merged_accept_map is None:
            return None
        old, new = self._merged_accept_map, self._accepted_map()
        changed = {oid for oid in set(old) | set(new) if old.get(oid) != new.get(oid)}
        if not changed:
            return "no accepted catalogue resolution changed since the final merge; literal identities refreshed in place"
        book_of = {oid: g for g in self.inventory["data"].get("books", []) for oid in g.get("observation_ids", [])}
        kind = {o["observation_id"]: o.get("kind") for o in self.observations}
        for oid in changed:
            g = book_of.get(oid)
            if g is None or any(kind.get(o) != "book" for o in g["observation_ids"]):
                return None          # outside a pure book group: an acceptance could un-demote an ambiguous spine
        return f"{len(changed)} accepted catalogue resolution(s) changed, all on book groups (no demotion or count can change); identities refreshed in place"

    def _refresh_literal_identities(self) -> bool:
        """The re-merge's effect on book groups, replayed in place with dedup._identity on the current readings/resolutions: identity, status and the identity
        issue line. Returns False (caller re-merges) if a group that was accepted would lose or change its accepted work (that could demote it)."""
        from pipeline import dedup as _dd
        readings = {r["observation_id"]: r for r in self.readings}
        resolutions = {r["observation_id"]: r for r in self.resolutions}
        id_issues = set(_dd._identity([], {}, {})[1])
        _conf = {"decision": "accepted", "matching_evidence": ["x"], "source_url": "u"}
        id_issues |= set(_dd._identity([{"observation_id": "a"}, {"observation_id": "b"}], {},
                                       {"a": {**_conf, "work": "1"}, "b": {**_conf, "work": "2"}})[1])
        key = lambda i: (str(i.get("work") or ""), str(i.get("title") or "").casefold())
        books = self.inventory["data"]["books"]
        plan = []
        for g in books:
            old = g.get("identity") or {}
            n, issues = _dd._identity([{"observation_id": o} for o in g["observation_ids"]], readings, resolutions)
            if old.get("decision") == "accepted" and (n["decision"] != "accepted" or key(n) != key(old)):
                return False
            if n["decision"] != old.get("decision") and old.get("decision") != "accepted" and not (g.get("issues") and g["issues"][-1] in id_issues):
                return False         # the identity issue line is not where dedup puts it: do not guess
            plan.append((g, n, issues))
        for g, n, issues in plan:
            old_decision = (g.get("identity") or {}).get("decision")
            g["identity"] = n
            if n["decision"] != old_decision:
                g["status"] = "identified" if n["decision"] == "accepted" else "unidentified"
                if old_decision != "accepted":
                    g["issues"].pop()
                g["issues"].extend(issues)
        return True

    async def _alt_count_after_packet(self, packet_path: Path) -> None:
        """Alternative association (track_v2) after the packet is written: off the critical path, so the packet target no longer limits it. Patched into
        data.count.alternative_method of the packet already on disk. Never fails the sweep."""
        try:
            alt = await self._alternative_count(0.0)
            alt["ran_after_packet"] = True
            self.alt_count = alt
            env = C.read_json(packet_path)
            count = (env.get("data") or {}).get("count")
            if isinstance(count, dict):
                count["alternative_method"] = alt
                C.write_json(packet_path, env)
            if self.finalization is not None:
                self.finalization["alternative_count"] = {k: alt.get(k) for k in ("method", "status", "book_groups", "elapsed_s")}
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - diagnostic only
            logging.getLogger(__name__).exception("alternative association after the packet failed")

    def _apply_group_readings(self) -> None:
        """Write the best sighting reading into each group's identity.visible_text (all raw readings listed as evidence) and rewrite inventory.json."""
        if not self.inventory:
            return
        rmap = {r["observation_id"]: r for r in self.readings}
        rmap2 = {r["observation_id"]: r for r in self.readings2}
        n1 = (self.stage_availability.get("read") or {}).get("selection") or "primary"
        n2 = (self.stage_availability.get("read2") or {}).get("selection") or "second"
        for g in self.inventory["data"]["books"]:
            c = GR.combine(g, rmap, rmap2, (n1, n2))
            ident = g.setdefault("identity", {})
            if c["best"]:
                ident["visible_text"] = c["best"]["text"]
                ident["visible_text_observation_id"] = c["best"]["observation_id"]
                ident["visible_text_reader"] = c["best"]["reader"]
            if c["union_text"]:
                ident["reading_union_text"] = c["union_text"]
            ident["readings_considered"] = c["readings_considered"]
            ident["reading_status"] = "literal reading, unverified"
        C.write_json(self.run_dir / C.ARTIFACTS["dedup"], self.inventory)

    # ------------------------------------------------------------------ merge (the only whole-run step)
    def _schedule_merge(self) -> None:
        self._merge_dirty = True
        if self._merge_task is None or self._merge_task.done():
            self._merge_task = self._spawn(self._merge_loop(), "merge")

    async def _merge_loop(self) -> None:
        """Debounced live merges, throttled by their own cost so they cannot starve detection/OCR during capture.

        Next live merge waits ``merge_cost_factor`` x the previous merge duration; once one exceeds ``merge_live_budget_s`` live
        merging stops (unmerged detections stay visible as 'awaiting merge') and a single final merge runs at end_sweep.
        """
        while self._merge_dirty and not self._closed:
            self._merge_dirty = False
            await asyncio.sleep(self.merge_debounce_s)
            if self._live_merges_stopped:
                return
            wait = self._last_merge_end + self._last_merge_cost * self.merge_cost_factor - time.monotonic()
            if wait > 0:
                await asyncio.sleep(min(wait, 60))
            t0 = time.monotonic()
            await self._merge_once()
            self._last_merge_cost = time.monotonic() - t0
            self._last_merge_end = time.monotonic()
            if self._last_merge_cost > self.merge_live_budget_s and self.status == "capturing":
                self._live_merges_stopped = True
                self.stage_log["_live_merge_policy"] = {"status": "ok", "detail": f"live merging stopped after a {self._last_merge_cost:.0f}s merge (> {self.merge_live_budget_s:.0f}s budget); one final merge at end_sweep"}

    def _manifest_env(self) -> dict[str, Any]:
        sel = self.selector
        times = [f["timestamp_s"] for f in self.frames]
        scan = self.scan or {}
        data = {"input": {"file_name": scan.get("file") or ("live-session" if "camera" in self.sources else "replay"),
                          "sha256": scan.get("sha256"), "source": "+".join(sorted(self.sources)) or None,
                          "scan_label": scan.get("label"), "scan_note": ("registered scan replayed through the live ingestion path" if scan else
                                                                          "unregistered capture: not one of the canonical scans")},
                "sampling": {"mode": "incremental", "overlap_step": self.overlap_step, "sampled_frames": len(self.frames),
                             "selected_frames": sum(f["selected"] for f in self.frames)},
                "quality_config": self.qcfg.to_dict(), "frames": copy.deepcopy(self.frames),
                "intervals": list(sel.intervals) if sel else [], "capture_gaps": list(sel.gaps) if sel else [],
                "room_sweep": self.room_sweep.record(),
                "audit": {"fiducial_markers": None, "manual_review": None,
                          "scope_note": "Live capture: no room boundary or metric reference evidence evaluated."}}
        if times:
            data["input"]["duration_s"] = times[-1]
        return C.envelope(self.run_id, "frames", data, status="ok")

    def _write_stage_artifacts(self) -> tuple[Path, Path, Path | None, Path | None]:
        manifest = C.write_json(self.run_dir / C.ARTIFACTS["frames"], self._manifest_env())
        cand = C.write_json(self.run_dir / C.ARTIFACTS["detect"], C.envelope(
            self.run_id, "detect", {"observations": copy.deepcopy(self.observations)}, status="ok",
            issues=[i for i in self.stage_issues if i.get("stage") == "detect"], metrics={'stage_status': self._stage_status().get('detect')}))
        rd = rs = None
        if self.readings:
            rd = C.write_json(self.run_dir / C.ARTIFACTS["read"], C.envelope(self.run_id, "read", {"readings": copy.deepcopy(self.readings)}, status=self._stage_status()['read']['status'],
                metrics={'stage_status': self._stage_status()['read']}, issues=[i for i in self.stage_issues if i.get('stage')=='read']))
        if self.resolutions:
            rs = C.write_json(self.run_dir / C.ARTIFACTS["resolve"], C.envelope(self.run_id, "resolve", {"resolutions": copy.deepcopy(self.resolutions)}, status=self._stage_status()['resolve']['status']))
        return manifest, cand, rd, rs

    async def _merge_once(self) -> bool:
        if not self.observations:
            return False
        async with self._merge_lock:
            rev = self.revision
            ids = {o["observation_id"] for o in self.observations}
            manifest, cand, rd, rs = self._write_stage_artifacts()
            accept_map = self._accepted_map()       # what this merge's identity step sees (same instant as the files it reads)
            t0 = time.perf_counter()
            try:
                cfg = {**self.stage_config, "output_name": C.ARTIFACTS["dedup"],
                       "dedup": {**(self.stage_config.get("dedup") or {}), "max_gap_s": self.dedup_max_gap_s}}
                kw = {}
                if rd: kw["readings_path"] = rd
                if rs: kw["resolutions_path"] = rs
                path = await asyncio.to_thread(self.stages.merge, cand, manifest, self.run_dir, cfg, **kw)
                inv = C.read_json(path)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the previous inventory; surface the failure
                self._merge_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                self.stage_log["_merge"] = {"status": "failed", "detail": self._merge_error}
                await self._publish_state()
                return False
            if self._closed: return False
            self._merge_error = None
            # Revision guard: a result computed from fewer observations never replaces a newer one.
            if self.inventory is None or len(ids) >= len(self.merged_observation_ids):
                self.inventory, self.merged_observation_ids = inv, ids
                self._merged_accept_map = accept_map
            self.stage_log["_merge"] = {"status": inv.get("status"), "elapsed_s": round(time.perf_counter() - t0, 3),
                                        "based_on_revision": rev, "observations": len(ids)}
            self.counters["merges"] += 1
            self._bump()
        await self._publish_state()
        if self.status == "capturing" and self.read_policy == "group" and self.provisional_reads and self.stage_availability.get("read", {}).get("available"):
            self._spawn(self._provisional_read_pass(), "provisional-read")
        return True

    async def _provisional_read_pass(self) -> None:
        """Read the best sighting of groups that are stable (>= stable_min sightings, observation set unchanged for >= stable_s) while capture continues,
        so finalize only reads new groups. One small chunk per merge; raw readings are kept and reused by the final group read."""
        books = (self.inventory or {}).get("data", {}).get("books", [])
        now = time.monotonic()
        obs = {o["observation_id"]: o for o in self.observations}
        frames = {f["frame_id"]: f for f in self.frames}
        todo = []
        for g in books:
            key = frozenset(g.get("observation_ids", []))
            if len(key) < self.stable_min:
                continue
            first = self._group_seen.setdefault(key, now)
            if now - first < self.stable_s or any(o in self._prov_done or o in self._prov_inflight for o in key):
                continue
            seen = [o for o in key if o in obs and obs[o]["frame_id"] in frames]
            if not seen:
                continue
            best = max(seen, key=lambda o: GR.sighting_score(obs[o], frames[obs[o]["frame_id"]]))
            if GR.spine_width_px(obs[best]) < GR.MIN_WIDTH_PX:
                continue
            todo.append(best)
        todo = todo[:6]
        if not todo:
            return
        self._prov_inflight.update(todo)
        self._prov_chunk += 1
        n = self._prov_chunk
        try:
            async with self._read_sem:
                manifest = C.write_json(self.run_dir / "live" / "manifests" / f"prov_{n:04d}.json", self._manifest_env())
                slice_path = C.write_json(self.run_dir / "live" / "read" / f"prov_{n:04d}.candidates.json", C.envelope(self.run_id, "detect", {"observations": [obs[o] for o in todo]}, status="ok"))
                cfg = {**self.stage_configs["read"], "session_revision": self.revision, "readings_filename": f"live/read/prov_{n:04d}.readings.json", "max_books": None}
                rpath = await asyncio.to_thread(self.stages.read, slice_path, manifest, self.run_dir, cfg)
                rd = (C.read_json(rpath).get("data") or {}).get("readings", [])
            for r in rd:
                self._prov_done[r["observation_id"]] = r
            self.readings.extend(rd)
            self.prov_report.update(groups_read=self.prov_report["groups_read"] + len(todo), observations_read=self.prov_report["observations_read"] + len(rd), chunks=n, status="running")
            # the final group read treats these as done and never resolves them: resolve here (outside the read semaphore), or they stay unidentified for good
            try:
                rcfg = {**self.stage_configs["resolve"], "session_revision": self.revision, "resolutions_filename": f"live/read/prov_{n:04d}.resolutions.json"}
                sart = C.read_json(await asyncio.to_thread(self.stages.resolve, rpath, self.run_dir, rcfg))
                self.resolutions.extend((sart.get("data") or {}).get("resolutions", []))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.stage_issues.append({"stage": "resolve", "reason": f"provisional resolve chunk {n} failed: {type(exc).__name__}: {str(exc)[:140]}"})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.stage_issues.append({"stage": "read", "reason": f"provisional read chunk {n} failed: {type(exc).__name__}: {str(exc)[:140]}"})
        finally:
            self._prov_inflight.difference_update(todo)

    # ------------------------------------------------------------------ corrections
    def current_view_window(self, seconds: float = 4.0) -> dict[str, float]:
        if not self.frames:
            raise K.CorrectionError("no frames captured yet")
        end = self.frames[-1]["timestamp_s"]
        return {"start_s": max(0.0, round(end - seconds, 3)), "end_s": end}

    async def add_correction(self, kind: str, *, instance_ids: list[str] | None = None, observation_ids: list[str] | None = None,
                             time_window: dict[str, float] | None = None, value: dict[str, Any] | None = None,
                             source: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.status == "finished":
            raise SweepError("sweep already finished")
        obs_ids = list(observation_ids or [])
        if instance_ids:
            # ids are those shown on screen (latest merge); they are converted to stable observation ids now
            obs_ids += K.resolve_instance_targets((self.inventory or {}).get("data") or {}, instance_ids)
        target: dict[str, Any] = {}
        if obs_ids:
            target["observation_ids"] = sorted(set(obs_ids))
        if time_window:
            target["time_window"] = time_window
        self._corr_seq += 1
        entry = K.new_entry(f"c{self._corr_seq:03d}", kind, target, value, source)
        self.ledger.append(entry)
        self._bump()
        snap = self.snapshot()
        applied = next((a for a in snap["corrections"]["applied"] if a["correction_id"] == entry["id"]), {})
        await self._emit({"type": "state", "state": snap})
        return {"correction_id": entry["id"], "kind": kind, "matched_instances": applied.get("matched_instances", []),
                "matched_none": not applied.get("matched_instances"), "totals": snap["totals"]}

    def set_scan(self, scan: dict[str, Any] | None) -> None:
        """Bind this session to a registered scan (label + source hash) BEFORE frames arrive; never changes afterwards."""
        if self.frames and (self.scan or {}).get("sha256") != (scan or {}).get("sha256"):
            raise SweepError("this session already has frames from a different source; start a new sweep to switch scans")
        self.scan = dict(scan) if scan else None
        self._bump()

    # ------------------------------------------------------------------ measurement (human inputs -> shared scale/room stages)
    def _measure(self, manifest: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        manifest = manifest or self._manifest_env()
        obs = {o["observation_id"]: o for o in self.observations}
        out = self.meas.compute(self.inventory, manifest, obs, self.config.get("scale"))
        self._last_measure = out
        return out

    async def _meas_changed(self) -> None:
        self._bump()
        self.meas.persist(self.run_dir)
        await self._publish_state()

    def _frame_ok(self, frame_id: str) -> dict[str, Any]:
        fr = next((f for f in self.frames if f["frame_id"] == frame_id), None)
        if fr is None:
            raise MeasurementError(f"unknown frame {frame_id}")
        return fr

    async def add_calibration(self, **kw: Any) -> dict[str, Any]:
        if self.status == "finished":
            raise SweepError("sweep already finished")
        self._frame_ok(kw["frame_id"])
        cal = self.meas.add_calibration(**kw)
        from pipeline import scale as SC
        problems = SC.validate_calibration(cal, {f["frame_id"]: f for f in self.frames}, SC.DEFAULTS)
        if problems:  # reject unusable input at the door, with reasons; nothing is stored
            self.meas.remove_calibration(cal["calibration_id"])
            self._bump()
            raise MeasurementError("; ".join(problems))
        entry = {"status": "valid"}
        await self._meas_changed()
        return {"calibration": cal, "status": entry["status"]}

    async def remove_calibration(self, calibration_id: str) -> None:
        self.meas.remove_calibration(calibration_id)
        await self._meas_changed()

    async def set_dimension(self, what: str, value: float, unit: str, *, approximate: bool = True, plane_id: str | None = None,
                            utterance: str | None = None, turn_id: str | None = None) -> dict[str, Any]:
        """A dimension the person VOLUNTEERED (REST/tests; the live agent can no longer record or ask for one). Stored as a user-supplied cross-check; the
        automatic scale (pipeline.auto_scale) never depends on it."""
        m = __import__("app.measurement", fromlist=["to_metres"]).to_metres(value, unit)
        tol = 5.0 if approximate else 2.0
        if what == "ceiling_height":
            anchor = self.meas.set_height_anchor(m, tol, utterance, turn_id=turn_id)
            await self._meas_changed()
            return {"recorded": "ceiling_height_anchor", "height_m": m, "label": "user-supplied", "tolerance_pct": tol}
        if what not in {"shelf_unit_width", "shelf_unit_height"}:
            raise MeasurementError("what must be shelf_unit_width, shelf_unit_height or ceiling_height")
        slot = self.meas.set_pending_dimension(plane_id, what.split("_")[-1], m, tol, utterance)
        await self._meas_changed()
        have_both = "width_m" in slot and "height_m" in slot
        return {"recorded": what, "metres": round(m, 4), "plane_id": slot["plane_id"], "tolerance_pct": tol,
                "next": ("stored as a volunteered, user-supplied cross-check only; the app scales spines and rooms from the sweep and never asks the person for a size or a tap"
                         if have_both else "stored as a volunteered, user-supplied cross-check only; do not ask for anything else")}

    async def add_wall(self, **kw: Any) -> dict[str, Any]:
        self._frame_ok(kw["frame_id"])
        w = self.meas.add_wall(**kw)
        await self._meas_changed()
        return w

    async def remove_wall(self, wall_id: str) -> None:
        self.meas.remove_wall(wall_id)
        await self._meas_changed()

    def measurement_status(self) -> dict[str, Any]:
        scale, room = self._measure()
        return self.meas.status(scale, room)

    async def request_end(self, *, source: str, camera: bool = True, skip: bool = False) -> dict[str, Any]:
        """End requested by the person (End button or the voice agent's end_sweep). During a camera sweep the first request starts the room
        pan instead (app.room_sweep): the instruction is spoken once and recording goes on; the next request (or skip) ends. Does NOT end the
        sweep itself: the caller calls end_sweep when the action is 'end'."""
        if self.status != "capturing":
            return {"action": "end", "reason": self.status}
        last = self.frames[-1] if self.frames else None
        res = self.room_sweep.request_end(last["timestamp_s"] if last else None, last["frame_id"] if last else None,
                                          source=source, camera=camera and "camera" in self.sources, skip=skip)
        if res["action"] == "room_sweep":
            await self._start_room_phase({"code": "room_sweep", "message": res["message"]}, last["timestamp_s"] if last else None,
                                         last["frame_id"] if last else None, relay=source != "voice")   # the voice tool result carries the words itself
        return res

    async def _start_room_phase(self, g: dict[str, Any], t: float | None, fid: str | None, *, relay: bool) -> None:
        self.meas._issued.setdefault("show_room_corner", t or 0.0)     # the room instruction replaces the mid-sweep room-corner guidance
        self.nudge_policy.note_spoken("room_sweep", t)                  # said regardless of the gap: it answers the person's End / leaving the shelf
        self.nudge_policy.muted = True                                   # the room pan is far and off the shelf by design: no capture notices
        await self._add_nudge({"code": g["code"], "message": g["message"], "kind": "guidance", "relay": relay}, fid)

    async def _guidance(self, elapsed: float) -> None:
        if self.room_sweep.phase == "shelf":           # no room guidance during the shelf pass (YugoTest14: show_room_corner at 47 s, mid-shelf)
            return
        if not self.nudge_policy.gap_ok(elapsed):      # the global gap between spoken notices covers APP GUIDANCE too; guidance_due keeps it for later
            return
        scale, room = self._last_measure or self._measure()
        for g in self.meas.guidance_due(elapsed, self.counters["keyframes"], scale, room):
            self.nudge_policy.note_spoken(g["code"], elapsed)
            await self._add_nudge({"code": g["code"], "message": g["message"], "kind": "guidance"}, None)

    # ------------------------------------------------------------------ snapshot / finalization
    def _stage_status(self) -> dict[str, Any]:
        def agg(stage: str) -> dict[str, Any]:
            vals = [v[stage] for v in self.stage_log.values() if isinstance(v, dict) and stage in v]
            if not vals:
                return {"status": "skipped", "elapsed_s": None, "detail": "not run"}
            statuses = {v.get("status") for v in vals}
            bad = [v for v in vals if v.get("status") in ("failed",)]
            status = "failed" if bad and len(bad) == len(vals) else ("partial" if statuses - {"ok", "complete"} else "ok")
            return {"status": status, "elapsed_s": round(sum(v.get("elapsed_s") or 0 for v in vals), 3),
                    "detail": (f"{len(vals)} call(s)" + (f"; {bad[0].get('detail')}" if bad else ""))}
        out = {s: agg(s) for s in ("detect", "read", "resolve")}
        m = self.stage_log.get("_merge")
        out["dedup"] = ({"status": m.get("status") if m.get("status") != "complete" else "ok", "elapsed_s": m.get("elapsed_s"),
                         "detail": m.get("detail")} if m else {"status": "skipped", "elapsed_s": None, "detail": "no merge yet"})
        for stage, availability in self.stage_availability.items():
            if not availability['available']:
                out[stage] = {"status": "partial", "elapsed_s": out[stage].get("elapsed_s"),
                              "detail": availability['reason'] + "; unknown is not evidence of absence"}
        if '_price' in self.stage_log: out['price'] = self.stage_log['_price']
        return out

    def _ocr_coverage(self) -> dict[str, Any]:
        books = [o for o in self.observations if o.get("kind") == "book"]
        attempted = sum(1 for r in self.readings if r.get("crop_path") is not None and "reading_budget_limit" not in (r.get("issues") or []))
        budget_skipped = sum(1 for r in self.readings if "reading_budget_limit" in (r.get("issues") or []))
        reading_ids = {r.get("observation_id") for r in self.readings}
        queued = sum(1 for o in books if o["observation_id"] not in reading_ids)
        return {"book_observations": len(books), "ocr_attempted": attempted, "skipped_by_budget": budget_skipped, "queued_or_pending": queued,
                "note": "attempted != identified; OCR budget (LIBRARY_OCR_MAX_BOOKS) limits attempts per keyframe, it does not claim identity accuracy"}

    def snapshot(self) -> dict[str, Any]:
        manifest = self._manifest_env() if self.frames else C.envelope(self.run_id, "frames", {
            "input": {"file_name": "live-session", "sha256": None}, "frames": [], "intervals": [], "capture_gaps": [], "audit": {}})
        pk_cfg = {"run_id": self.run_id, "country": self.market["country"], "currency": self.market["currency"], "alternative_count": getattr(self, "alt_count", None),
                  "corrections": self.ledger, "stage_status": self._stage_status(), "provenance": self.provenance(),
                  "prices": self.prices, "readings": self.readings, "run_dir": str(self.run_dir),
                  "captured_at": self.started_at, "sweep_extra": {"source": "+".join(sorted(self.sources)) or "none"}}
        if not self.frames:
            manifest["data"]["frames"] = []
        measurement = None
        if self.frames:
            try:
                scale, room = self._measure(manifest)
                pk_cfg["scale"], pk_cfg["room"] = scale, room
                measurement = self.meas.status(scale, room)
            except Exception as exc:  # noqa: BLE001 - measurement must never break the inventory view
                measurement = {"error": f"measurement_failed:{type(exc).__name__}"}
        try:
            data = P.build_data(self.inventory, manifest, pk_cfg) if self.frames else None
        except Exception as exc:  # noqa: BLE001
            data = None
            self._merge_error = f"snapshot_failed:{type(exc).__name__}"
        pending = [{"observation_id": o["observation_id"], "kind": o["kind"], "category": o.get("category"),
                    "frame_id": o["frame_id"], "bbox_xyxy": o["bbox_xyxy"],
                    "evidence": {"frame_id": o["frame_id"], "bbox_xyxy": o["bbox_xyxy"],
                                 "frame_width_px": next((f["width_px"] for f in self.frames if f["frame_id"] == o["frame_id"]), None),
                                 "frame_height_px": next((f["height_px"] for f in self.frames if f["frame_id"] == o["frame_id"]), None)}}
                   for o in self.observations if o["observation_id"] not in self.merged_observation_ids]
        # an observation pending a merge belongs to a user exclusion window? keep visible but provisional
        in_flight = sum(1 for v in self.stage_log.values() if isinstance(v, dict) and any(
            isinstance(x, dict) and x.get("status") == "running" for x in v.values()))
        keyframes = self.counters["keyframes"]
        detected = sum(1 for v in self.stage_log.values() if isinstance(v, dict) and v.get("detect", {}).get("status") not in (None, "running"))
        return {
            "type_hint": "inventory_snapshot", "session_id": self.session_id, "revision": self.revision, "status": self.status,
            "market": self.market, "completion_status": "provisional",
            "totals": (data or {}).get("totals") or P.compute_totals([], [], {}),
            "books": (data or {}).get("books", []), "items": (data or {}).get("items", []),
            "excluded_by_user": (data or {}).get("excluded_by_user", []),
            "review_queue": (data or {}).get("review_queue", []), "coverage_gaps": (data or {}).get("coverage_gaps", []),
            "corrections": (data or {}).get("corrections", {"ledger": self.ledger, "applied": []}),
            "provisional_unmerged": pending,
            "progress": {"frames_saved": len(self.frames), "keyframes_selected": keyframes, "keyframes_detected": detected,
                         "stages_in_flight": in_flight, "merges": self.counters["merges"], "merge_error": self._merge_error},
            "nudges": self.nudges[-5:], "provenance": self.provenance(), "stage_status": pk_cfg["stage_status"],
            "measurement": measurement, "room": (data or {}).get("room"),
            "identity": {"session_id": self.session_id, "run_id": self.run_id, "run_dir_name": self.run_dir.name,
                         "scan": ({k: self.scan.get(k) for k in ("label", "sha256", "file")} if self.scan else None),
                         "source": "+".join(sorted(self.sources)) or None, "status": self.status, "frames_saved": len(self.frames)},
            "ocr_coverage": self._ocr_coverage(),
            "frames_index": [{"frame_id": f["frame_id"], "timestamp_s": f["timestamp_s"]} for f in self.frames if f.get("selected")][-80:] +
                            ([{"frame_id": self.frames[-1]["frame_id"], "timestamp_s": self.frames[-1]["timestamp_s"]}] if self.frames and not self.frames[-1].get("selected") else []),
            "finalization": self.finalization, "final_paths": self.final_paths, "finalize": self._finalize_info(),
        }

    def _finalize_info(self) -> dict[str, Any]:
        """Status-indicator view of end_sweep: current step, group-reading progress, failure reason. No decisions depend on it."""
        rr = self.read_report or {}
        total = rr.get("pass1_total")
        return {"stage": self.finalize_stage, "error": self.finalize_error,
                "read": ({"attempted": rr.get("attempted_observations", 0), "total": total, "budget_exhausted": bool(rr.get("budget_exhausted"))}
                         if self.finalize_stage == "reading titles" and total else None)}

    async def _finalize_step(self, name: str | None) -> None:
        self.finalize_stage = name
        if name:
            self._watch_steps.append({"stage": name, "at_s": round(time.monotonic() - (self.ended or self.started), 2)})
        self._watch_write()
        await self._emit({"type": "finalize", **self._finalize_info()})

    async def end_sweep(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Close the open interval, wait for in-flight stage work, merge, build packet+report. Idempotent.
        A failure is recorded (status 'failed', finalize.error) before it propagates; End can be pressed again.
        The work runs in a task the engine owns and the caller only awaits it (shielded): a caller that is cancelled (the Live socket closing, a New sweep, a closed
        browser tab) never aborts the packet. Only aclose() cancels it."""
        if self.status == "finished":
            return {"already_finished": True, **(self.finalization or {})}
        if self.status == "finalizing" or (self._end_task is not None and not self._end_task.done()):
            raise SweepError("already finalizing")
        self.finalize_error = None
        self._end_task = task = asyncio.ensure_future(self._end_sweep_recorded(timeout_s))
        task.add_done_callback(lambda t: t.cancelled() or t.exception())      # a failure is re-raised to the awaiting caller; with no caller it is already recorded in status/finalize_error
        return await asyncio.shield(task)

    async def wait_ended(self) -> None:
        """Wait for a running end_sweep (whoever started it) to finish. Never raises: a failure is in status/finalize_error."""
        task = self._end_task
        if task is not None and not task.done():
            await asyncio.wait([task])

    async def _end_sweep_recorded(self, timeout_s: float | None) -> dict[str, Any]:
        try:
            return await self._end_sweep(timeout_s=timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced to the status indicator, then re-raised
            self.finalize_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.finalize_stage = None
            self.status = "failed"
            self._bump()
            await self._publish_state()
            raise

    async def _end_sweep(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Close the open interval, wait for in-flight stage work, merge, build packet+report. Idempotent."""
        if self.status == "finished":
            return {"already_finished": True, **(self.finalization or {})}
        if self.status == "finalizing":
            raise SweepError("already finalizing")
        self.status = "finalizing"
        self.ended = time.monotonic()
        self._ended_epoch = time.time()
        self._watch_steps = []
        ended_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        await self._emit({"type": "status", "status": "finalizing"})
        await self._finalize_step("finishing detection and OCR")
        timeout = float(timeout_s if timeout_s is not None else self.config.get("final_timeout_s", 240))
        timed_out = False
        tm: dict[str, float] = {}
        _mark = time.monotonic()
        def lap(name: str) -> None:
            nonlocal _mark
            now = time.monotonic()
            tm[name] = round(now - _mark, 3)
            _mark = now
        async with self._ingest_lock:
            if self.selector:
                self._start_closed(self.selector.flush())
        try:
            deadline = time.monotonic() + timeout
            while True:
                pending = [t for t in self._tasks if t is not self._merge_task and not t.done()]
                if not pending:
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    timed_out = True
                    break
                await asyncio.wait(pending, timeout=min(left, 5))
        finally:
            for t in [t for t in self._tasks if not t.done() and t is not self._merge_task]:
                if timed_out:
                    t.cancel()
        lap("wait_for_inflight_detect_ocr_s")
        if self._merge_task:
            self._merge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._merge_task
        lap("cancel_pending_merge_s")
        await self._finalize_step("merging duplicates")
        merged = await self._merge_once() if self.observations else False
        lap("final_merge_dedup_s")
        if self.alt_after_packet:
            self.alt_count = {"method": str(self.config.get("alt_association", os.environ.get("LIBRARY_ALT_ASSOCIATION", "track_v2"))), "status": "pending", "book_groups": None,
                              "reason": "runs after the packet is written (diagnostic only, off the packet's critical path) and is patched into it when done",
                              "note": "the two methods differ in how aggressively they link blurred sightings; the truth is most likely between them"}
        else:
            await self._finalize_step("cross-checking the count")
            self.alt_count = await self._alternative_count(time.monotonic() - self.ended)
            lap("alternative_association_s")
        if self.read_policy == "group" and merged and self.stage_availability.get("read", {}).get("available"):
            await self._finalize_step("reading titles")
            if await self._read_groups():
                lap("group_reading_s")
                why = self._remerge_skippable()
                if why and self._refresh_literal_identities():
                    self.remerge_info = {"skipped": True, "reason": why}
                    self._bump()
                else:
                    self.remerge_info = {"skipped": False, "reason": ("accepted catalogue resolutions (or observations) changed since the final merge"
                                                                      if self.remerge_skip else "LIBRARY_REMERGE_SKIP=0")}
                    merged = await self._merge_once() or merged
                self._apply_group_readings()
                lap("remerge_and_combine_s")
        await self._finalize_step("pricing")
        manifest, cand, rd, rs = self._write_stage_artifacts()
        from pipeline.price import price_books_with_work_level, price_config_from_env
        before_price = P.build_data(self.inventory, C.read_json(manifest), {"run_id": self.run_id, "country": self.market['country'], "currency": self.market['currency'], "corrections": self.ledger})
        self.prices = await asyncio.to_thread(price_books_with_work_level, before_price['books'], self.run_dir, {**self.config, **self.market, **price_config_from_env(self.run_dir)})
        C.write_json(self.run_dir/'prices.json', self.prices)
        self.stage_log['_price'] = {"status": self.prices['status'], "elapsed_s": self.prices['metrics']['elapsed_s'], "detail": 'Source-backed pricing: exact edition by printed ISBN, else work-level equivalent edition (labelled, budgeted); unknown stays null'}
        lap("pricing_s")
        inv_path = self.run_dir / C.ARTIFACTS["dedup"]
        pk_cfg = {"run_id": self.run_id, "country": self.market["country"], "currency": self.market["currency"], "alternative_count": self.alt_count,
                  "corrections": self.ledger, "stage_status": self._stage_status(), "provenance": self.provenance(),
                  "prices": self.prices, "readings": self.readings, "run_dir": str(self.run_dir),
                  "captured_at": self.started_at,
                  "sweep_extra": {"source": "+".join(sorted(self.sources)) or "none", "ended_at": ended_at}}
        await self._finalize_step("room and scale")
        self.meas.persist(self.run_dir)
        scale, room = self._measure(C.read_json(manifest))
        if scale is not None:
            C.write_json(self.run_dir / C.ARTIFACTS["scale"], C.envelope(self.run_id, "scale", scale, status="ok" if scale["summary"]["valid_calibrations"] and not scale["summary"]["abstained"] else "partial",
                                                                          metrics={"request_count": 0, "cost": 0, "cost_provenance": "local geometry"}))
        C.write_json(self.run_dir / C.ARTIFACTS["room"], C.envelope(self.run_id, "room", room, status="ok" if room["status"] == "estimated" else "partial",
                                                                     metrics={"request_count": 0, "cost": 0, "cost_provenance": "local geometry"}))
        room, room_auto = await self._room_auto_pass(room, C.read_json(manifest), time.monotonic() - self.ended)
        if room_auto.get("used"):
            C.write_json(self.run_dir / C.ARTIFACTS["room"], C.envelope(self.run_id, "room", room, status="ok" if room["status"] == "estimated" else "partial",
                                                                         metrics={"request_count": 0, "cost": 0, "cost_provenance": "local geometry (automatic wall evidence)"}))
        pk_cfg["scale"], pk_cfg["room"], pk_cfg["room_auto"] = scale, room, room_auto
        lap("scale_room_artifacts_s")
        await self._finalize_step("building the packet")
        packet_path = P.build(inv_path if (merged or inv_path.is_file()) and self.inventory else None, manifest, self.run_dir, pk_cfg)
        lap("packet_build_s")
        await self._finalize_step("rendering the report")
        report_path = R.render(packet_path, self.run_dir)
        lap("report_render_s")
        elapsed = round(time.monotonic() - self.ended, 3)       # packet-ready time: JSON + HTML written (PDF is timed separately below)
        env = C.read_json(packet_path)
        env["data"]["sweep"]["finalization_timings"] = tm
        env["data"]["sweep"]["finalization_remerge"] = self.remerge_info or None
        env["data"]["sweep"]["capture_quality"] = self.capture_quality()
        env["data"]["sweep"]["room_sweep"] = self.room_sweep.record()
        env["data"]["sweep"]["finalization_s"] = elapsed
        env["data"]["sweep"]["finalization_note"] = ("from end_sweep call to claim_packet.json + report.html written; includes waiting for in-flight stage work and the group-reading "
                                                     f"budget (adaptive: {self.read_report.get('budget_s', 0)}s of LIBRARY_READ_BUDGET_S={self.read_budget_s:.0f}s so the packet lands near LIBRARY_PACKET_TARGET_S={self.packet_target_s:.0f}s; "
                                                     f"reading used {tm.get('group_reading_s', 0):.0f}s; groups attempted {self.read_report.get('groups_attempted')} / unattempted {self.read_report.get('groups_unattempted')}); the PDF is generated afterwards and timed separately")
        C.write_json(packet_path, env)
        report_path = R.render(packet_path, self.run_dir)
        pdf = {"status": "skipped", "reason": "report_pdf disabled"}
        if self.config.get("report_pdf", True):
            pdf = R.render_print_pdf(packet_path, self.run_dir)   # image-free print variant of the final report (evidence cited by path/hash)
        env = C.read_json(packet_path)
        env["data"]["evidence"]["report_pdf"] = pdf
        C.write_json(packet_path, env)
        self.final_paths = {"packet": C.rel_path(packet_path, self.run_dir), "report": C.rel_path(report_path, self.run_dir)}
        self.finalization = {"finalization_s": elapsed, "timings": tm, "report_pdf": pdf, "timed_out_waiting_for_stages": timed_out, "packet_status": env["status"],
                             "packet_issues": env["issues"]}
        tot = env["data"].get("totals") or {}
        self._watch_first = {k: tot.get(k) for k in ("book_count", "book_count_range", "book_count_lower", "book_count_upper", "count_reliability", "item_count",
                                                      "headline_estimator", "book_lines", "book_count_range_label")}   # headline + book lines on the watch page
        self.status = "finished"
        self.finalize_stage = None
        self._bump()
        # the post-sweep 2D/3D queue starts the moment the packet is on disk, before anything that talks to the phone or takes time. It used to
        # run after the alternative count (~50-90 s of CPU under load) and after the state push to a socket that is usually gone by then:
        # YugoTest14/15 were queued 50 s / 90 s after their packets, which looked like "never".
        if self.on_finished is not None:           # never delays or fails the packet: it is already on disk
            try:
                self.on_finished(self.run_dir)
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).exception("post-sweep hook failed")
        await self._publish_state()
        if self.alt_after_packet:                  # diagnostic only; patches count.alternative_method of the packet already on disk
            await self._alt_count_after_packet(packet_path)
        return {"finalization_s": elapsed, "timed_out": timed_out, "totals": env["data"]["totals"], "paths": self.final_paths,
                "capture_quality": self.capture_quality(), "count_reliability": self.cq.summary()["count_reliability"],
                "packet_status": env["status"]}

    async def aclose(self) -> None:
        """Cancel and await every background task (reset/disconnect). Evidence on disk is kept."""
        self._closed = True
        self._watch_closed = True
        self._watch_write(force=True)
        if self._watch_task is not None:
            self._watch_task.cancel()
        tasks = list(self._tasks)
        if self._end_task is not None and not self._end_task.done():
            tasks.append(self._end_task)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
