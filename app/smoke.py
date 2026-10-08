"""Minimal REAL-provider smoke checks. Skips explicitly when no credential is configured.

    .venv/bin/python -m app.smoke generate [--frame runs/offline_baseline/frames/f00014.jpg]   # one structured image request
    .venv/bin/python -m app.smoke live                                                           # connect to the Live model, 1 text turn
    .venv/bin/python -m app.smoke all

Purpose: verify the configured model IDs are accessible and the SDK/schema/config dialect is accepted, BEFORE any
full run. Each check makes at most one small request, prints a JSON result (never the key) and a clear verdict:
PASS (real call succeeded) | FAIL (call made, rejected: see error type) | SKIPPED (no credential).
Results are labelled real; nothing here is cached or mocked.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from providers import gemini

REPO = Path(__file__).resolve().parents[1]


async def check_generate(frame: Path) -> dict:
    from pipeline.detect import PROMPT, RESPONSE_SCHEMA
    if not gemini.credential_available():
        return {"check": "generate", "verdict": "SKIPPED", "reason": "no GOOGLE_API_KEY/GEMINI_API_KEY in environment or <repo>/.env"}
    usage = gemini.UsageLog()
    t0 = time.perf_counter()
    try:
        res = await gemini.structured_image_request(images=[frame], prompt=PROMPT, response_schema=RESPONSE_SCHEMA,
                                                    config={"temperature": 0.0, "media_resolution": "high"}, stage="smoke", usage=usage)
    except Exception as exc:  # noqa: BLE001 - message sanitised by the wrapper
        return {"check": "generate", "verdict": "FAIL", "model": gemini.default_model(), "error_type": type(exc).__name__,
                "error": str(exc)[:300], "elapsed_s": round(time.perf_counter() - t0, 2)}
    objs = (res.parsed or {}).get("objects", []) if isinstance(res.parsed, dict) else []
    return {"check": "generate", "verdict": "PASS", "model": res.model, "objects_returned": len(objs),
            "by_kind": {k: sum(o.get("kind") == k for o in objs) for k in ("book", "non_book", "ambiguous")},
            "usage": res.usage, "latency_s": res.latency_s, "cost_usd": res.cost_usd, "cost_note": res.cost_note,
            "note": "A real call. Counts here are one frame's raw model output, not an evaluated detection result."}


async def check_live() -> dict:
    from app import live_tools as LT
    if not gemini.credential_available():
        return {"check": "live", "verdict": "SKIPPED", "reason": "no credential"}
    if not LT.LIVE_MODEL_ID:
        return {"check": "live", "verdict": "SKIPPED", "reason": "LIBRARY_LIVE_MODEL is not set; no model id is assumed. Set it, then rerun."}
    import os
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    t0 = time.perf_counter()
    try:
        cfg = LT.build_live_config()
        async with client.aio.live.connect(model=LT.LIVE_MODEL_ID, config=cfg) as session:
            connected_s = time.perf_counter() - t0
            await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text="Say hello in five words.")]),
                                              turn_complete=True)
            out_chars, audio_bytes, tool_calls = 0, 0, 0
            async def collect():
                nonlocal out_chars, audio_bytes, tool_calls
                async for r in session.receive():
                    sc = r.server_content
                    if r.tool_call:
                        tool_calls += len(r.tool_call.function_calls or [])
                    if sc:
                        if sc.output_transcription and sc.output_transcription.text:
                            out_chars += len(sc.output_transcription.text)
                        for p in (sc.model_turn.parts if sc.model_turn else []) or []:
                            if p.inline_data and p.inline_data.data:
                                audio_bytes += len(p.inline_data.data)
                        if sc.turn_complete:
                            return
            await asyncio.wait_for(collect(), 30)
    except Exception as exc:  # noqa: BLE001
        return {"check": "live", "verdict": "FAIL", "model": LT.LIVE_MODEL_ID, "error_type": type(exc).__name__, "error": str(exc)[:300],
                "elapsed_s": round(time.perf_counter() - t0, 2)}
    return {"check": "live", "verdict": "PASS", "model": LT.LIVE_MODEL_ID, "connect_s": round(connected_s, 2),
            "reply_transcript_chars": out_chars, "reply_audio_bytes": audio_bytes, "tool_calls": tool_calls,
            "total_s": round(time.perf_counter() - t0, 2),
            "note": "Verifies model id, LiveConnectConfig + tool declarations accepted, audio reply. Does NOT verify microphone/camera devices."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["generate", "live", "all"])
    ap.add_argument("--frame", default=str(REPO / "runs/offline_baseline/frames/f00014.jpg"))
    a = ap.parse_args(argv)
    results = []
    if a.check in ("generate", "all"):
        results.append(asyncio.run(check_generate(Path(a.frame))))
    if a.check in ("live", "all"):
        results.append(asyncio.run(check_live()))
    print(json.dumps(results, indent=2))
    return 1 if any(r["verdict"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
