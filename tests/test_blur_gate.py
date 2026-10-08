"""Blur gate before detection + per-sweep capture stats. Synthetic frames and fake stages: plumbing only, not a quality claim."""
import pytest

from app.engine import SweepEngine
from tests.test_live_app import FAKES, make_frame

SHARP = make_frame(0, blur=0)
BLURRED = make_frame(0, blur=14)


async def engine(tmp_path, **config):
    return SweepEngine(tmp_path / "r", stages=FAKES, config={"cache_dir": str(tmp_path / "c"), "detect_max_fps": 5, **config}, session_id="gate")


async def feed(eng, pattern, source="replay"):
    for i, blurred in enumerate(pattern):
        await eng.ingest_frame(BLURRED if blurred else SHARP, i * 0.2, source=source)


@pytest.mark.asyncio
async def test_gate_defaults_to_live_only(tmp_path):
    eng = await engine(tmp_path)
    assert eng.blur_gate == "live"
    await feed(eng, [0] * 10 + [1])                                 # replay frames: never gated by default
    assert eng.dense["gated_frames"] == 0 and not eng.frames[-1]["quality"].get("blur_gated")
    await eng.aclose()


@pytest.mark.asyncio
async def test_live_blurred_frame_is_stored_but_not_detected(tmp_path):
    eng = await engine(tmp_path, blur_gate="all")
    await feed(eng, [0] * 10 + [1, 0])
    blurred, after = eng.frames[10], eng.frames[11]
    assert blurred["quality"]["blur_gated"] is True and blurred["quality"]["sharpness"] < blurred["quality"]["blur_floor"]
    assert (eng.run_dir / blurred["image_path"]).is_file()           # stored: evidence and panorama still have it
    assert blurred["frame_id"] not in eng._detected_ids and "below the blur gate" in blurred["selection_reason"]
    assert after["frame_id"] in eng._detected_ids                    # the gated frame did not consume the rate slot: the next sharp frame is detected
    assert eng.dense["gated_frames"] == 1 and eng.dense["skipped_blurry"] == 1
    cq = eng.capture_quality()
    assert cq["blur_gate"]["frames_gated"] == 1 and cq["blur_gate"]["frames_total"] == 12 and cq["blur_gate"]["gated_frame_ids"] == [blurred["frame_id"]]
    await eng.aclose()


@pytest.mark.asyncio
async def test_gate_modes(tmp_path):
    camera = await engine(tmp_path / "a")
    await feed(camera, [0] * 10 + [1], source="camera")              # "live" mode gates camera frames
    assert camera.dense["gated_frames"] == 1
    off = await engine(tmp_path / "b", blur_gate="off")
    await feed(off, [0] * 10 + [1], source="camera")
    assert off.dense["gated_frames"] == 0 and "blur_gated" not in off.frames[-1]["quality"]
    with pytest.raises(ValueError):
        await engine(tmp_path / "c", blur_gate="sometimes")
    await camera.aclose(); await off.aclose()


@pytest.mark.asyncio
async def test_gate_ratio_is_tunable_and_warmup_uses_the_absolute_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_BLUR_GATE", "all")
    monkeypatch.setenv("LIBRARY_BLUR_GATE_RATIO", "0.99")
    eng = await engine(tmp_path)
    assert eng.blur_gate == "all" and eng.blur_gate_ratio == 0.99
    await feed(eng, [1, 1, 1])                                       # fewer than min_history frames: only the absolute floor applies
    assert all(f["quality"]["blur_floor"] == eng.blur_gate_abs for f in eng.frames)
    await eng.aclose()


@pytest.mark.asyncio
async def test_capture_stats_land_in_the_packet_quality_block(tmp_path):
    eng = await engine(tmp_path, blur_gate="all")
    eng.set_capture_info({"label": "Back Camera", "width": 1920, "height": 1080, "frame_rate": 30, "capture_mode": "best_of_window", "type": "capture_info",
                          "client_stats": {"frames_sent": 11, "skipped_socket_backlog": 0}, "junk": {"nested": {"too": "deep"}}})
    await feed(eng, [0] * 10 + [1])
    cq = eng.capture_quality()
    assert cq["device_label"] == "Back Camera" and cq["device"]["frame_rate"] == 30 and cq["device"]["capture_mode"] == "best_of_window"
    assert cq["frames"] == 11 and cq["frame_size"] == "1920x1080" and cq["median_sharpness"] > 0
    assert cq["frame_gap_p50_s"] == 0.2 and cq["frame_gap_p90_s"] == 0.2 and cq["frame_gap_max_s"] == 0.2 and cq["fps_effective"] == 5.0
    assert cq["client_counters"]["frames_sent"] == 11 and "junk" not in cq["device"]
    assert "count_reliability" in cq and "bands" in cq                # the original summary keys are kept
    summary = await eng.end_sweep()
    from pipeline import contracts as C
    packet = C.read_json(eng.run_dir / "claim_packet.json")["data"]
    sweep_cq = packet["sweep"]["capture_quality"]
    assert sweep_cq["device_label"] == "Back Camera" and sweep_cq["median_sharpness"] == cq["median_sharpness"] and sweep_cq["blur_gate"]["frames_gated"] == 1
    assert summary["capture_quality"]["frame_gap_p90_s"] == 0.2
    await eng.aclose()


@pytest.mark.asyncio
async def test_client_capture_clock_gaps_are_reported_separately(tmp_path):
    eng = await engine(tmp_path)
    for i, t in enumerate([0.0, 0.2, 0.4, 1.0]):                      # arrival is regular, the phone's own capture clock is not
        await eng.ingest_frame(SHARP, i * 0.2, source="replay", client={"t_s": t, "cand": 6 + i, "sharp_small": 50.0 + i})
    cq = eng.capture_quality()
    assert cq["frame_gap_p90_s"] == 0.2 and cq["capture_clock_gap_p90_s"] == 0.6 and cq["client_window_candidates_median"] == 7.5
    assert eng.frames[-1]["client"] == {"t_s": 1.0, "cand": 9, "sharp_small": 53.0}
    await eng.ingest_frame(SHARP, 0.8 + 0.2, source="replay", client={"t_s": float("nan"), "cand": True})   # non-finite / boolean client values are dropped
    assert eng.frames[-1]["client"] == {}
    await eng.aclose()


@pytest.mark.asyncio
async def test_motion_rate_uses_the_phone_capture_clock_when_given(tmp_path):
    arrival, phone = await engine(tmp_path / "a"), await engine(tmp_path / "b")
    for eng, clock in ((arrival, None), (phone, [0.0, 0.4])):          # same pictures and arrival times; the phone says the two frames are 0.4 s apart
        for i, shift in enumerate([0, 120]):
            await eng.ingest_frame(make_frame(shift=shift, seed=7), i * 0.2, source="replay", client=None if clock is None else {"t_s": clock[i]})
    a, b = arrival.frames[1]["quality"]["motion_px_per_s"], phone.frames[1]["quality"]["motion_px_per_s"]
    assert a and b and abs(a / b - 2.0) < 0.05
    await arrival.aclose(); await phone.aclose()
