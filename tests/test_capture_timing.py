"""Live frame-rate timing stats and the capped live state push. Synthetic frames and fake stages: plumbing only, not a quality claim.
YugoTest5/6 (iPhone Safari, portrait 1080x1920) fell from ~4.3 fps to ~1.5 fps over the sweep; these tests pin the numbers that make that visible in the packet."""
import types

import pytest

from app import server as S
from app.capture_quality import _fps_series, capture_stats
from app.engine import SweepEngine
from tests.test_live_app import FAKES, make_frame


def frame(i, arrival, cap=None, size=(1080, 1920), ingest=None):
    f = {"frame_id": f"f{i:05d}", "timestamp_s": arrival, "width_px": size[0], "height_px": size[1], "quality": {"sharpness": 100.0 + i}}
    if cap is not None:
        f["client"] = {"t_s": cap, "cand": 4}
    if ingest is not None:
        f["ingest_ms"] = ingest
    return f


def sinking_sweep():
    """60 frames at 5 fps, then 60 at 2 fps; the phone's capture clock agrees with arrival except that the server falls 2 s behind over the second half."""
    frames, t, i = [], 0.0, 0
    for gap, n in ((0.2, 60), (0.5, 60)):
        for _ in range(n):
            lag = 4.0 if i < 60 else 4.0 + 2.0 * (i - 60) / 60
            frames.append(frame(i, t + lag, cap=t, ingest=80.0 + i))
            t += gap
            i += 1
    return frames


def test_rate_over_time_gaps_on_both_clocks_and_server_drift_are_in_the_stats():
    cq = capture_stats(sinking_sweep(), {}, None, {"camera"})
    ft = cq["frame_timing"]
    series = ft["fps_series_15s"]
    assert series[0] > 4.0 and series[-1] < 2.6 and series[0] > series[-1] + 2         # sinks from near the 5 fps tuning rate to near 2
    assert ft["fps_first_third"] == pytest.approx(5.0, abs=0.1) and ft["fps_last_third"] == pytest.approx(1.9, abs=0.1)
    assert cq["capture_clock_gap_p50_s"] in (0.2, 0.5) and cq["capture_clock_gap_p90_s"] == cq["capture_clock_gap_max_s"] == 0.5
    assert cq["frame_gap_p90_s"] == pytest.approx(0.53, abs=0.01)                          # arrival gaps are stretched by the server falling behind; the phone clock is not
    assert ft["arrival_clock"]["p50"] == cq["frame_gap_p50_s"] and ft["capture_clock"]["max"] == cq["capture_clock_gap_max_s"]
    assert cq["fps_effective"] == pytest.approx(cq["fps_capture_clock"], abs=0.3)
    assert ft["arrival_minus_capture_drift_s"] == pytest.approx(2.0, abs=0.3)                # the server fell ~2 s behind the phone
    assert ft["server_ingest_ms"]["p50"] == pytest.approx(140.0, abs=1.0) and ft["server_ingest_ms"]["max"] == 199.0


def test_gap_stats_never_come_back_null_when_frames_exist_and_use_the_documented_names():
    cq = capture_stats([frame(i, i * 0.4, cap=i * 0.4) for i in range(10)], {}, None, {"camera"})
    for key in ("frame_gap_p50_s", "frame_gap_p90_s", "frame_gap_max_s", "capture_clock_gap_p50_s", "capture_clock_gap_p90_s", "capture_clock_gap_max_s"):
        assert cq[key] == 0.4, key
    no_phone = capture_stats([frame(i, i * 0.4) for i in range(10)], {}, None, {"camera"})
    assert no_phone["frame_gap_p50_s"] == 0.4 and no_phone["capture_clock_gap_p50_s"] is None and no_phone["frame_timing"]["capture_clock"] is None
    assert no_phone["frame_timing"]["arrival_minus_capture_drift_s"] is None


def test_fps_series_scales_the_partial_last_slice_and_drops_slivers():
    times = [i * 0.2 for i in range(100)]                       # 19.8 s at 5 fps: one full 15 s slice and a 4.8 s slice
    assert _fps_series(times) == [pytest.approx(5.0, abs=0.1), pytest.approx(5.0, abs=0.3)]
    assert len(_fps_series([i * 0.2 for i in range(77)])) == 1  # 15.2 s: the 0.2 s sliver is dropped, not reported as a rate
    assert _fps_series([0.0]) == []


def test_orientation_comes_from_the_saved_pixels():
    portrait = capture_stats([frame(i, i * 0.2) for i in range(5)], {}, None, {"camera"})
    assert portrait["orientation"] == "portrait" and portrait["frame_aspect"] == 0.562 and portrait["orientation_share"] == {"portrait": 1.0}
    landscape = capture_stats([frame(i, i * 0.2, size=(1920, 1080)) for i in range(4)] + [frame(9, 1.0)], {}, None, {"camera"})   # the phone was turned for the last frame
    assert landscape["orientation"] == "landscape" and landscape["orientation_share"] == {"landscape": 0.8, "portrait": 0.2}
    assert capture_stats([], {}, None, set())["orientation"] is None


@pytest.mark.asyncio
async def test_engine_keeps_the_phone_timing_keys_and_times_its_own_ingest(tmp_path):
    eng = SweepEngine(tmp_path / "r", stages=FAKES, config={"cache_dir": str(tmp_path / "c"), "detect_max_fps": 5}, session_id="t21")
    client = {"t_s": 0.5, "cand": 5, "sharp_small": 40.0, "seen": 6, "win_ms": 201, "enc_ms": 90, "q": 0.9, "junk": 1, "nan": float("nan")}
    await eng.ingest_frame(make_frame(0, blur=0), 0.0, source="replay", client={k: v for k, v in client.items() if k != "nan"})
    assert eng.frames[0]["client"] == {"t_s": 0.5, "cand": 5, "sharp_small": 40.0, "seen": 6, "win_ms": 201, "enc_ms": 90, "q": 0.9}
    assert eng.frames[0]["ingest_ms"] > 0
    assert eng.capture_quality()["frame_timing"]["server_ingest_ms"]["p50"] == eng.frames[0]["ingest_ms"]
    await eng.aclose()


@pytest.mark.asyncio
async def test_client_stats_survive_with_all_their_flat_keys(tmp_path):
    eng = SweepEngine(tmp_path / "r", stages=FAKES, config={"cache_dir": str(tmp_path / "c")}, session_id="t21b")
    stats = {f"k{i}": i for i in range(45)}
    stats["sent_per_10s"] = ",".join(["48"] * 40) + ",ZZ" * 100                     # a long series is clipped, not rejected
    eng.set_capture_info({"type": "capture_info", "client_stats": stats})
    kept = eng.capture_info["client_stats"]
    assert len(kept) == 46 and kept["k44"] == 44 and len(kept["sent_per_10s"]) == 200
    await eng.aclose()


def books(n, size=1500):
    return [{"id": f"bk-{i:03d}", "dedup_reasons": [{"pad": "x" * size}], "evidence": {"pad": "y" * 400}} for i in range(n)]


def fake_session(camera_enabled):
    return types.SimpleNamespace(camera_enabled=camera_enabled, transcript=[], tool_activity=[], live_model="m", voice_enabled=False)


def big_snapshot(status="capturing"):
    return {"status": status, "totals": {"book_count": 751}, "books": books(751), "items": books(146), "review_queue": books(3262, 100), "provisional_unmerged": books(300, 0),
            "progress": {"frames_saved": 300}, "provenance": {"capture_quality": {"frames": 300}}}


def test_live_state_push_is_capped_while_the_camera_is_on_and_complete_otherwise():
    full = S._ui_state(fake_session(True), big_snapshot("finalizing"))
    assert len(full["books"]) == 751 and "list_caps" not in full                                    # not capturing: the whole list
    off = S._ui_state(fake_session(False), big_snapshot())
    assert len(off["books"]) == 751 and "list_caps" not in off                                      # capturing but the camera is off (stopped before End): the whole list
    live = S._ui_state(fake_session(True), big_snapshot())
    assert [b["id"] for b in live["books"]] == [f"bk-{i:03d}" for i in range(711, 751)]              # newest rows of books/items/unmerged
    assert len(live["items"]) == len(live["provisional_unmerged"]) == 40
    assert [r["id"] for r in live["review_queue"]] == [f"bk-{i:03d}" for i in range(40)]            # review keeps its head (highest severity first)
    assert live["list_caps"] == {"books": {"shown": 40, "total": 751}, "items": {"shown": 40, "total": 146}, "provisional_unmerged": {"shown": 40, "total": 300},
                                 "review_queue": {"shown": 40, "total": 3262}}
    assert live["totals"]["book_count"] == 751                                                       # counts are never capped
    assert all("dedup_reasons" not in r for r in live["books"] + live["items"]) and "dedup_reasons" in full["books"][0]   # the bulk of a row, read by no live view
    import json
    assert len(json.dumps(live)) < 0.05 * len(json.dumps(full))                                      # a ~3 MB push becomes ~100 KB (the phone also uploads frames over the same radio)


def test_short_lists_pass_through_untouched_but_still_report_their_totals():
    snap = {"status": "capturing", "books": books(5, 10), "items": [], "review_queue": [], "provisional_unmerged": []}
    out = S._ui_state(fake_session(True), snap)
    assert len(out["books"]) == 5 and out["list_caps"]["books"] == {"shown": 5, "total": 5}


def upright(jpeg: bytes) -> bytes:
    import cv2
    import numpy as np
    img = cv2.rotate(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), cv2.ROTATE_90_CLOCKWISE)
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.mark.asyncio
async def test_portrait_camera_gets_one_orientation_nudge_landscape_and_replays_none(tmp_path):
    portrait, landscape = upright(make_frame(0)), make_frame(0)
    codes = {}
    for name, jpeg, source in (("portrait_camera", portrait, "camera"), ("landscape_camera", landscape, "camera"), ("portrait_replay", portrait, "replay")):
        eng = SweepEngine(tmp_path / name, stages=FAKES, config={"cache_dir": str(tmp_path / "c"), "detect_max_fps": 5}, session_id=name)
        for i in range(14):
            await eng.ingest_frame(jpeg, i * 0.2, source=source)
        codes[name] = [n["code"] for n in eng.nudges if n["code"] == "portrait"]
        if name == "portrait_camera":
            assert eng.capture_quality()["orientation"] == "portrait" and eng.frames[0]["height_px"] > eng.frames[0]["width_px"]
            assert "landscape" in eng.nudges[[n["code"] for n in eng.nudges].index("portrait")]["message"]
        await eng.aclose()
    assert codes == {"portrait_camera": ["portrait"], "landscape_camera": [], "portrait_replay": []}   # once per sweep, camera source only


def test_frame_quality_motion_estimates_match_the_direct_functions_and_reuse_orb_features():
    """Ingest of a 1080x1920 frame was ~120 ms (two motion estimates = ~85 ms); the zoom now runs on a second thread and ORB features are reused. Same numbers."""
    import cv2
    import numpy as np
    from pipeline import quality as Q
    cfg = Q.QualityConfig()
    frames = [cv2.imdecode(np.frombuffer(make_frame(shift=s, seed=7), np.uint8), cv2.IMREAD_COLOR) for s in (0, 30, 70)]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    prev, results = None, []
    for bgr, gray in zip(frames, grays):
        q, small = Q.frame_quality(bgr, gray, cfg, prev, 0.2 if prev is not None else None)
        if prev is not None:
            dx, dy, resp = Q.estimate_shift(prev, small, gray.shape[1])
            assert (q["shift_dx_px"], q["shift_dy_px"], q["shift_response"]) == (round(dx, 1), round(dy, 1), round(resp, 3))
            scale, inliers = Q.estimate_zoom(prev, small)
            assert q["zoom_rate_per_s"] == (round(abs(float(np.log(scale))) / 0.2, 3) if inliers >= 12 and scale > 0 else None)
            assert id(small) in Q._FEATS and id(prev) in Q._FEATS                    # the previous frame's features were cached, not recomputed
        results.append(q)
        prev = small
    assert results[0]["motion_px_per_s"] is None and results[1]["motion_px_per_s"] > 0 and len(Q._FEATS) <= Q._FEATS_KEEP


@pytest.mark.asyncio
async def test_4k_frames_are_saved_native_but_measured_on_a_1920_long_side_copy(tmp_path):
    """A 3840x2160 evidence frame (?res=4k) must not change the 1080p-class thresholds (sharpness floor, px/s) nor cost 4x the ingest time."""
    import cv2
    import numpy as np
    jpegs4k, jpegs2k = [], []
    for shift in (0, 80):
        base = cv2.imdecode(np.frombuffer(make_frame(shift=shift // 4, seed=5), np.uint8), cv2.IMREAD_COLOR)
        big = cv2.resize(base, (3840, 2160), interpolation=cv2.INTER_CUBIC)
        jpegs4k.append(cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
        jpegs2k.append(cv2.imencode(".jpg", cv2.resize(big, (1920, 1080), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    engines = {}
    for name, jpegs in (("4k", jpegs4k), ("2k", jpegs2k)):
        eng = SweepEngine(tmp_path / name, stages=FAKES, config={"cache_dir": str(tmp_path / "c"), "detect_max_fps": 5}, session_id="q" + name)
        for i, j in enumerate(jpegs):
            await eng.ingest_frame(j, i * 0.2, source="replay")
        engines[name] = eng
    f4, f2 = engines["4k"].frames, engines["2k"].frames
    assert (f4[0]["width_px"], f4[0]["height_px"], f4[0]["quality_scale"]) == (3840, 2160, 0.5) and "quality_scale" not in f2[0]
    assert (engines["4k"].run_dir / f4[0]["image_path"]).read_bytes() == jpegs4k[0]                        # the saved evidence is the native JPEG, byte for byte
    assert f4[0]["quality"]["sharpness"] == pytest.approx(f2[0]["quality"]["sharpness"], rel=0.12)           # same scale as every earlier (1080p) run
    assert f4[1]["quality"]["shift_dx_px"] == pytest.approx(f2[1]["quality"]["shift_dx_px"], abs=6.0)        # shift in quality-frame px, not 2x
    assert engines["4k"].selector.width_px == 1920                                                              # the selector normalises shift by the quality frame
    for eng in engines.values():
        await eng.aclose()


def test_too_far_target_is_60_px_and_env_tunable(monkeypatch):
    from app.capture_quality import CaptureQuality
    assert CaptureQuality().t["min_spine_px"] == 60                                        # user decision 2026-10-08: the YugoTest12/13 close passes (was 120, 40 before)
    monkeypatch.setenv("LIBRARY_CQ_MIN_SPINE_PX", "120")
    assert CaptureQuality().t["min_spine_px"] == 120


def test_frame_quality_survives_a_picture_size_change_mid_sweep():
    """A phone that rotates mid-sweep sends a frame of another size: no motion estimate for that frame, no crash (it killed a live session on 2026-10-08)."""
    import numpy as np
    from pipeline.quality import QualityConfig, frame_quality
    cfg = QualityConfig()
    land = np.random.default_rng(0).integers(0, 255, (1080, 1920), dtype=np.uint8)
    _, prev = frame_quality(np.dstack([land] * 3), land, cfg, None, None)
    port = np.ascontiguousarray(land[:, :1080].T)
    q, small = frame_quality(np.dstack([port] * 3), port, cfg, prev, 0.2)
    assert q["motion_px_per_s"] is None and small.shape != prev.shape
