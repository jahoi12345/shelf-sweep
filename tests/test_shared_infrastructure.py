"""Focused tests for the shared infrastructure (no network, no credentials)."""
import asyncio
import json
from pathlib import Path

import pytest

from pipeline import contracts as C
from pipeline import packet as P
from pipeline.frames import select_frames
from providers import gemini
from providers.cache import DiskCache, make_key, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "sweeps" / "IMG_2281.MOV"


# ---- coordinates / evidence bounds
def test_normalized_box_conversion_and_clamp():
    assert C.normalized_box_to_pixels([100, 200, 500, 400], 1920, 1080) == [384.0, 108.0, 768.0, 540.0]
    # out-of-range provider values are clamped to the frame
    assert C.normalized_box_to_pixels([-50, -10, 1200, 1100], 1920, 1080) == [0.0, 0.0, 1920.0, 1080.0]
    with pytest.raises(C.ContractError):
        C.normalized_box_to_pixels([500, 500, 500, 600], 1920, 1080)  # zero height


def test_check_bbox_flags_out_of_bounds_and_inverted():
    assert C.check_bbox([0, 0, 10, 10], 100, 100) == []
    assert C.check_bbox([0, 0, 101, 10], 100, 100)
    assert C.check_bbox([10, 0, 5, 10], 100, 100)
    assert C.check_bbox([0, 0, 10], 100, 100)


def test_run_relative_paths_reject_escape():
    with pytest.raises(C.ContractError):
        C.resolve_in_run("/tmp/x", "../etc/passwd")
    with pytest.raises(C.ContractError):
        C.resolve_in_run("/tmp/x", "/abs/path")


# ---- cache identity
def test_cache_key_changes_with_every_input():
    base = dict(model="m", prompt="p", config={"t": 0}, schema=None, inputs=[{"sha256": "a"}])
    k = make_key("ns", **base)
    assert k == make_key("ns", **base)
    for change in ({"model": "m2"}, {"prompt": "p2"}, {"config": {"t": 1}}, {"schema": {"x": 1}}, {"inputs": [{"sha256": "b"}]}):
        assert make_key("ns", **{**base, **change}) != k


# ---- provider: cache hit, usage, bounded concurrency, structured-output failure
def _fake(calls, delay=0.01, text='{"a": 1}'):
    state = {"active": 0, "peak": 0}

    async def transport(model, images, prompt, config, schema):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(delay)
        state["active"] -= 1
        calls.append((model, prompt))
        return text, {"prompt_tokens": 10, "output_tokens": 5, "thought_tokens": 0, "total_tokens": 15}
    return transport, state


def test_provider_cache_hit_and_invalidation(tmp_path):
    calls = []
    transport, _ = _fake(calls)
    gemini.set_transport(transport)
    try:
        cache, usage = DiskCache(tmp_path / "c"), gemini.UsageLog(tmp_path / "u.jsonl")
        async def go(prompt="p", img=b"\xff\xd8\xffabc"):
            return await gemini.structured_image_request(images=[img], prompt=prompt, response_schema={"type": "object"},
                                                         model="m", cache=cache, usage=usage, stage="t")
        r1 = asyncio.run(go()); r2 = asyncio.run(go())
        assert (r1.cache_hit, r2.cache_hit) == (False, True) and r1.parsed == r2.parsed == {"a": 1}
        assert len(calls) == 1
        asyncio.run(go(prompt="other")); asyncio.run(go(img=b"\xff\xd8\xffxyz"))
        assert len(calls) == 3  # prompt change and image-byte change both miss
        s = usage.summary("t")
        assert s["requests"] == 3 and s["cache_hits"] == 1 and s["prompt_tokens"] == 30
        assert s["cost_usd"] is None and "unknown" in s["cost_note"]  # no assumed pricing
    finally:
        gemini.set_transport(None)


def test_provider_concurrency_bound_and_bad_json_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_CONCURRENCY", "2")
    calls = []
    transport, state = _fake(calls, delay=0.05)
    gemini.set_transport(transport)
    try:
        async def many():
            return await asyncio.gather(*[gemini.structured_image_request(
                images=[bytes([i])], prompt="p", response_schema={"type": "object"}, model="m") for i in range(8)])
        asyncio.run(many())
        assert state["peak"] <= 2
        bad, _ = _fake([], text="not json")
        gemini.set_transport(bad)
        cache = DiskCache(tmp_path / "c")
        with pytest.raises(gemini.StructuredOutputError):
            asyncio.run(gemini.structured_image_request(images=[b"x"], prompt="p", response_schema={"type": "object"},
                                                        model="m", cache=cache))
        assert cache.stats()["hits"] == 0 and not list((tmp_path / "c").rglob("*.json"))
    finally:
        gemini.set_transport(None)


def test_missing_credential_is_explicit(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False); monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini, "_dotenv_loaded", True)
    gemini.set_transport(None)
    with pytest.raises(gemini.MissingCredentialError):
        asyncio.run(gemini.structured_image_request(images=[b"x"], prompt="p", model="m"))


# ---- selection never silently drops a blurry stretch
def _f(i, t, sharp, dx=300.0, reasons=()):
    return {"frame_id": C.frame_id(i), "timestamp_s": t, "selected": False, "selection_reason": "",
            "quality": {"sharpness": sharp, "glare_fraction": 0.0, "shift_dx_px": dx, "shift_dy_px": 0.0,
                        "shift_response": 0.8, "reasons": list(reasons)}}


def test_selection_keeps_best_available_and_reports_gap():
    frames = [_f(i, i * 0.2, 300 if i < 5 else 20, reasons=[] if i < 5 else ["blurry"]) for i in range(12)]
    intervals, gaps = select_frames(frames, overlap_step=0.25, overlap_lost_step=0.6, width_px=1920)
    assert intervals and all(sum(f["selected"] for f in frames if f.get("interval_index") == iv["interval_index"]) == 1 for iv in intervals)
    assert gaps and gaps[0]["type"] == "no_clean_frame"  # blurry-only interval kept AND reported
    assert all(f["selection_reason"] for f in frames)


def test_overlap_lost_on_fast_pan():
    frames = [_f(0, 0.0, 200), _f(1, 0.2, 200, dx=1500.0)]
    _, gaps = select_frames(frames, 0.25, 0.6, 1920)
    assert any(g["type"] == "overlap_lost" for g in gaps)


# ---- totals / nulls / references (needs the video; builds a real manifest, fixture lines)
@pytest.mark.skipif(not VIDEO.exists(), reason="sweep video not present")
def test_fixture_packet_totals_and_unknowns(tmp_path):
    from pipeline.run import run
    out = tmp_path / "r"
    s = run(VIDEO, out, fixture=True, config={"cache_dir": str(tmp_path / "cache"), "no_provider": True})
    env = json.loads((out / "claim_packet.json").read_text())
    d = env["data"]
    assert d["completion_status"] == "provisional" and d["fixture"] is True
    manifest = json.loads((out / "manifest.json").read_text())
    assert P.verify_packet(env, manifest, out) == []
    t = d["totals"]
    assert t["book_count"] == len(d["books"]) == t["books_identified"] + t["books_unidentified"]
    assert t["shelf_run_m"] is None and t["books_replacement_cost"] is None and t["excluded_from_totals"] is None
    assert all(b["spine_height_cm"] is None and b["replacement_cost"] is None for b in d["books"])
    assert all(v is None for k, v in d["room"].items() if k.endswith(("_m", "_m2")))
    assert any(r["code"] == "fixture_data" for r in d["review_queue"])
    assert (out / "report.html").read_text().count("FIXTURE DATA") >= 1
    # every frame_ref opens
    assert all((out / b["frame_ref"]).is_file() for b in d["books"] + d["items"])
    # resume: second invocation skips frames
    s2 = run(VIDEO, out, fixture=True, config={"cache_dir": str(tmp_path / "cache"), "no_provider": True})
    assert s2["stages"]["frames"].get("resumed") is True
    # config change invalidates frames
    s3 = run(VIDEO, out, fixture=True, config={"cache_dir": str(tmp_path / "cache"), "no_provider": True, "frames": {"overlap_step": 0.4}})
    assert s3["stages"]["frames"].get("resumed") is False


def test_packet_verification_catches_tampering(tmp_path):
    from pipeline.run import run
    if not VIDEO.exists():
        pytest.skip("video missing")
    out = tmp_path / "r"
    run(VIDEO, out, fixture=True, config={"cache_dir": str(tmp_path / "c"), "no_provider": True})
    env = json.loads((out / "claim_packet.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    env["data"]["totals"]["book_count"] += 1
    env["data"]["books"][0]["replacement_cost"] = {"amount": 0}
    env["data"]["books"][1]["evidence"]["bbox_xyxy"] = [0, 0, 99999, 10]
    problems = P.verify_packet(env, manifest, out)
    assert any("book_count" in p for p in problems)
    assert any("replacement_cost must be null" in p for p in problems)
    assert any("outside" in p for p in problems)
