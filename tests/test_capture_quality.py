from app.capture_quality import CaptureQuality


def feed(cq, seconds, motion, sharp, zoom=0.0, fps=5, t0=0.0):
    out = []
    for i in range(int(seconds * fps)):
        out += cq.add(t0 + i / fps, motion, sharp, zoom)
    return out


def test_good_capture_is_strict_and_high_reliability():
    cq = CaptureQuality({})
    assert feed(cq, 20, 320, 130) == []
    s = cq.summary()
    assert s["share_strict"] == 1.0 and s["count_reliability"] == "high" and s["nudges_issued"] == {}


def test_slow_down_then_recapture_and_cooldown():
    cq = CaptureQuality({})
    n = feed(cq, 3, 520, 100)                       # > 450 px/s: slow down, not re-capture
    assert [x["code"] for x in n] == ["slow_down"]
    n = feed(cq, 20, 900, 100, t0=3.0)              # sustained > 700 px/s for > 1 s: re-capture (once the 15 s global gap since the slow-down clears)
    assert [x["code"] for x in n] == ["recapture"]
    assert cq.summary()["nudges_issued"] == {"slow_down": 1, "recapture": 1}   # once per bad stretch: no repeated slow-down spam


def test_sharpness_and_zoom_triggers_and_reliability_levels():
    assert [x["code"] for x in feed(CaptureQuality({}), 3, 300, 55)] == ["slow_down"]
    assert [x["code"] for x in feed(CaptureQuality({}), 3, 300, 100, zoom=0.7)] == ["slow_down"]
    cq = CaptureQuality({})
    feed(cq, 10, 300, 130)
    feed(cq, 20, 900, 130, t0=10.0)                 # 2/3 of seconds are bad -> low
    s = cq.summary()
    assert s["count_reliability"] == "low" and s["share_medium_or_better"] < 0.4


def test_env_tunable(monkeypatch):
    monkeypatch.setenv("LIBRARY_CQ_SLOW_MOTION", "250")
    assert [x["code"] for x in feed(CaptureQuality({}), 3, 300, 130)] == ["slow_down"]


def test_too_far_when_small_whatever_the_sharpness():
    cq = CaptureQuality({})
    feed(cq, 3, 300, 130)
    assert cq.now in ("strict", "medium")
    n = cq.add_widths(3.0, [26.0] * 12)
    assert [x["code"] for x in n] == ["too_far"] and cq.summary()["median_spine_px"] == 26.0
    assert cq.add_widths(3.2, [26.0] * 12) == []                         # global gap
    big = CaptureQuality({})
    feed(big, 3, 300, 130)
    assert big.add_widths(3.0, [62.0] * 12) == []                        # wide enough (target 60 px, user decision 2026-10-08)
    mid = CaptureQuality({})
    feed(mid, 3, 300, 130)
    assert [x['code'] for x in mid.add_widths(3.0, [58.0] * 12)] == ['too_far']
    fast = CaptureQuality({})
    feed(fast, 3, 600, 130)                                              # moving too fast AND too far: distance still speaks (after the slow-down, behind the gap)
    assert fast.add_widths(3.0, [26.0] * 12) == [] and fast.indicator() == "too_far"
    few = CaptureQuality({})
    feed(few, 3, 300, 130)
    assert few.add_widths(3.0, [26.0] * 3) == []                         # not enough detections to judge


def test_preprocess_strip_in_memory_only(tmp_path):
    import numpy as np
    from providers.readers import preprocess_strip
    img = np.full((40, 300, 3), (10, 200, 30), np.uint8)                 # horizontal strip, short side 40 < 60
    out, info = preprocess_strip(img)
    assert out.shape[0] >= 40 * 2 and info["pad_vertical_px"] == 2 * 0 + info["pad_vertical_px"] and info["border_median"] == [10, 200, 30]
    assert out[0, 0].tolist() == [10, 200, 30] and img.shape == (40, 300, 3)
    assert preprocess_strip(np.zeros((90, 300, 3), np.uint8))[1] is None


def test_scene_speed_is_informational_only(monkeypatch):
    cq = CaptureQuality({})
    for i in range(15):
        cq.add(i / 5, 300, 130, 0.0)
        cq.add_widths(i / 5, [25.0] * 3)         # 300 px/s at 25 px spines = 12 spine-widths/s
    s = cq.summary()
    sc = s["motion_scene_units"]
    assert sc["unit"] == "spine_widths_per_s" and abs(sc["p50"] - 12.0) < 0.01 and sc["informational_only"] and sc["reference_sw_s"] == 15
    assert s["bands"]["strict"] == s["seconds"] and not {"slow_down", "recapture"} & set(s["nudges_issued"])   # scene speed never bands or nudges (too_far may: 25 px spines)
    monkeypatch.setenv("LIBRARY_CQ_MAX_MOTION_PX_S", "250")             # the canonical env names are honoured
    assert CaptureQuality({}).t["slow_motion"] == 250.0
    assert CaptureQuality({}).t["slow_zoom"] == 0.3


def test_zoom_slow_down_at_0_3():
    assert [x["code"] for x in feed(CaptureQuality({}), 3, 300, 100, zoom=0.4)] == ["slow_down"]


def test_small_spines_cap_the_rating_at_medium():
    cq = CaptureQuality({})
    for i in range(100):                                                     # 20 s of sharp, slow capture: high by the bands alone
        cq.add(i / 5, 320, 130, 0.0)
        cq.add_widths(i / 5, [33.5] * 3)                                     # YugoTest14's median spine width: under the 60 px target
    s = cq.summary()
    assert s["count_reliability_bands"] == "high" and s["count_reliability"] == "medium" and "60" in s["count_reliability_capped"]
    close = CaptureQuality({})
    for i in range(100):
        close.add(i / 5, 320, 130, 0.0)
        close.add_widths(i / 5, [65.0] * 3)
    s = close.summary()
    assert s["count_reliability"] == "high" and s["count_reliability_capped"] is None
    few = CaptureQuality({})
    feed(few, 20, 320, 130)
    few.add_widths(1.0, [20.0] * 3)                                          # too few widths to judge the sweep: the bands decide
    assert few.summary()["count_reliability"] == "high"


def test_too_far_repeats_every_gap_until_fixed_and_the_indicator_is_red_meanwhile():
    from app.capture_quality import MESSAGES
    cq = CaptureQuality({})
    spoken, states = [], []
    for i in range(400):                                                     # 60 s too far (sharp and slow), then 20 s close
        t = i / 5
        spoken += [(x["code"], t) for x in cq.add(t, 320, 130, 0.0)]
        spoken += [(x["code"], t) for x in cq.add_widths(t, [33.0] * 6 if t < 60 else [70.0] * 6)]
        states.append(cq.indicator())
    times = [t for c, t in spoken if c == "too_far"]
    assert len(times) == 5 and all(b - a >= 15.0 for a, b in zip(times, times[1:])) and times[-1] <= 60 + 2   # every ~15 s while too far (2 s window), silent once fixed
    assert "Step closer" in MESSAGES["too_far"] and "arm's length" in MESSAGES["too_far"]
    assert set(states[5:295]) == {"too_far"} and states[-1] == "strict"     # red while too far although every frame is sharp; good again once close
    assert cq.summary()["share_frames_too_far"] > 0.7
