"""Headless recorded-video replay through the SAME ingestion path as the camera.

    python -m app.replay --video sweeps/IMG_2281.MOV --out runs/live_replay [--speed 0 | 1.0] [--fps 5]

Decodes the video, JPEG-encodes each sampled frame at native resolution and calls
``SweepEngine.ingest_frame(..., source="replay")``, exactly what the WebSocket handler does for camera
frames. ``--speed 0`` replays as fast as the stages allow; ``1.0`` is real time. Uses the real stage
functions; without a credential, detect/read are reported unavailable (never faked).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2

from app.engine import StageFns, SweepEngine


def iter_video_frames(video: str | Path, fps: float = 5.0, jpeg_quality: int = 95) -> "Iterator[tuple[float, bytes]]":
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    next_t = 0.0
    n = 0
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = pos / 1000.0 if pos else n / native_fps
        n += 1
        if t + 1e-6 >= next_t:
            ok2, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok2:
                yield round(t, 3), buf.tobytes()
            next_t += 1.0 / fps
            while next_t <= t:
                next_t += 1.0 / fps
    cap.release()


async def replay_into(engine: SweepEngine, video: str | Path, *, fps: float = 5.0, speed: float = 0.0,
                      on_frame: Callable[[dict[str, Any]], Any] | None = None, should_stop: Callable[[], bool] | None = None) -> int:
    """Feed ``video`` into ``engine`` frame by frame. Returns the number of frames ingested."""
    start = time.monotonic()
    count = 0
    it = iter_video_frames(video, fps)
    while True:
        item = await asyncio.to_thread(next, it, None)
        if item is None or (should_stop and should_stop()):
            break
        t, jpeg = item
        if speed > 0:
            delay = start + t / speed - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        res = await engine.ingest_frame(jpeg, t, source="replay")
        count += 1
        if on_frame:
            r = on_frame(res)
            if asyncio.iscoroutine(r):
                await r
    return count


def iter_run_frames(run_dir: str | Path) -> "Iterator[tuple[float, bytes]]":
    """Exact saved JPEG bytes and recorded timestamps of an earlier run (e.g. a live sweep), for re-running the pipeline on the same capture with another setting."""
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    for f in (manifest.get("data") or manifest)["frames"]:
        yield float(f["timestamp_s"]), (run / f["image_path"]).read_bytes()


async def replay_frames_into(engine: SweepEngine, items: "Iterator[tuple[float, bytes]]") -> int:
    count = 0
    for t, jpeg in items:
        await engine.ingest_frame(jpeg, t, source="replay")
        count += 1
    return count


async def _main(args: argparse.Namespace) -> int:
    from providers import gemini as _g
    _g._load_repo_dotenv()   # same .env behaviour as the server
    if Path(args.out).exists():
        raise ValueError('Replay refuses to overwrite an existing run; choose a new versioned output directory')
    state_log = Path(args.out)/'logs/live_states.jsonl'
    state_log.parent.mkdir(parents=True,exist_ok=True)
    async def emit(event):
        if event.get('type')=='state':
            state=event['state']
            with state_log.open('a') as target:
                target.write(json.dumps({'revision':state['revision'],'status':state['status'],'progress':state['progress'],'totals':state['totals'],
                                         'pending':len(state.get('provisional_unmerged',[])),
                                         'visible_text_lines':sum(bool(b.get('visible_text')) for b in state['books'])})+'\n')
    engine = SweepEngine(Path(args.out), config={"run_id": Path(args.out).name, "provider": "off" if args.no_provider else "auto",
                                                 "country": args.country, "currency": args.currency,
                                                 "catalogue_online": args.catalogue_online, "detector": args.detector, "reader": args.reader,
                                                 "max_books": args.max_books_per_keyframe, "rectify_polygons": True, "detector_weights": args.detector_weights, "detector_python": args.detector_python,
                                                 **({"blur_gate": args.blur_gate} if args.blur_gate else {}),
                                                 **({"blur_gate_ratio": args.blur_gate_ratio} if args.blur_gate_ratio is not None else {})}, stages=StageFns.real(), emit=emit)
    from providers.stages import close_runtimes
    try:   # bind the registered scan by file hash so the run is discoverable (labelling, debugger) and its identity is explicit
        if args.frames_from:
            raise LookupError("frames replay has no video file")
        import hashlib
        from app.debug_registry import read_registry
        digest = hashlib.sha256(Path(args.video).read_bytes()).hexdigest()
        sc = next((s for s in read_registry(Path(__file__).resolve().parents[1]) if s["sha256"] == digest), None)
        if sc:
            engine.set_scan({k: sc[k] for k in ("label", "sha256", "file")})
    except Exception:  # noqa: BLE001 - binding is advisory for ad-hoc videos
        pass
    t0 = time.perf_counter()
    try:
        n = (await replay_frames_into(engine, iter_run_frames(args.frames_from))) if args.frames_from else await replay_into(engine, args.video, fps=args.fps, speed=args.speed)
        ingest_s = time.perf_counter() - t0
        backlog = engine.provenance().get("detect_policy")      # detection still queued when capture stopped
        summary = await engine.end_sweep()
        result = {"frames_ingested": n, "ingest_wall_s": round(ingest_s, 2), "end_to_end_s": round(time.perf_counter()-t0, 2), "detect_backlog_at_end_of_sweep": backlog, **summary, "provenance": engine.provenance()}
        from pipeline import contracts as C
        C.write_json(engine.run_dir/'replay_summary.json', result)
        print(json.dumps(result, indent=2, default=str))
    finally:
        await engine.aclose()
        await close_runtimes()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video"); ap.add_argument("--out", required=True)
    ap.add_argument("--frames-from", help="re-run the pipeline on the saved frames and timestamps of an earlier run (e.g. a live sweep) instead of a video")
    ap.add_argument("--blur-gate", choices=["live", "all", "off"], help="override LIBRARY_BLUR_GATE for this replay ('all' applies the gate to replayed frames)")
    ap.add_argument("--blur-gate-ratio", type=float, help="floor = ratio x running median sharpness (default 0.5)")
    ap.add_argument("--fps", type=float, default=5.0); ap.add_argument("--speed", type=float, default=0.0)
    ap.add_argument("--no-provider", action="store_true"); ap.add_argument("--country", default="US"); ap.add_argument("--currency", default="USD")
    ap.add_argument("--catalogue-online", action="store_true")
    ap.add_argument("--detector", choices=["auto", "local", "gemini", "none"])
    ap.add_argument("--reader", choices=["auto", "apple", "easyocr", "gemini", "none"])
    ap.add_argument("--max-books-per-keyframe", type=int, help="bounded OCR budget; other crops remain unknown")
    ap.add_argument("--detector-weights"); ap.add_argument("--detector-python")
    args = ap.parse_args(argv)
    if bool(args.video) == bool(args.frames_from):
        ap.error("give exactly one of --video or --frames-from")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
