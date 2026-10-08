"""One deduplicated spoken-notice policy across both nudge sources (once per bad stretch, re-arm after good capture, global gap)."""
from app.capture_quality import CaptureQuality
from app.engine import CaptureMonitor
from app.nudge_policy import NudgePolicy


def test_once_per_stretch_then_rearm_after_good_capture():
    p = NudgePolicy(rearm_s=6, gap_s=15)
    spoken = [t / 5 for t in range(0, 300) if p.offer("slow_down", t / 5)]           # 60 s of continuous bad capture
    assert spoken == [0.0] and p.stretches["speed"] == 1                              # said once, silent while it continues
    assert not p.offer("slow_down", 65.0)                                            # 5 s of good capture: same stretch
    assert p.offer("slow_down", 72.0) and p.stretches["speed"] == 2                   # >= 6 s good: re-armed, next bad stretch spoken
    assert p.issued == {"slow_down": 2}


def test_same_meaning_codes_share_one_stretch_across_both_sources():
    p = NudgePolicy(rearm_s=6, gap_s=0)
    assert p.offer("blur", 1.0)
    assert not p.offer("fast_pan", 1.2) and not p.offer("slow_down", 1.4)            # speed family: one notice
    assert p.offer("too_far", 2.0) and not p.offer("closer", 2.5)                    # distance family: one notice
    assert p.offer("recapture", 3.0) and not p.offer("recapture", 4.0)               # recapture: own slot, at most once per stretch
    assert p.issued == {"blur": 1, "too_far": 1, "recapture": 1}


def test_global_gap_holds_later_notices_and_first_is_never_held():
    p = NudgePolicy(rearm_s=6, gap_s=15)
    assert p.offer("glare", 0.0)                                                      # first notice: no gap
    assert not p.offer("dark", 3.0)                                                   # another family inside the gap: held
    assert not p.offer("dark", 14.9)
    assert p.offer("dark", 15.0)                                                      # still dark when the gap clears: spoken once
    assert not p.offer("dark", 20.0)
    assert not p.offer("revisit", 25.0) and p.offer("revisit", 30.0)
    assert [s["code"] for s in p.spoken] == ["glare", "dark", "revisit"]
    gaps = [b["t_s"] - a["t_s"] for a, b in zip(p.spoken, p.spoken[1:])]
    assert min(gaps) >= 15


def test_held_stretch_that_ends_inside_the_gap_is_never_spoken():
    p = NudgePolicy(rearm_s=6, gap_s=15)
    assert p.offer("slow_down", 0.0)
    assert not p.offer("too_far", 2.0)                                                # held by the gap, then the condition ends
    assert p.offer("slow_down", 16.0) and p.stretches["speed"] == 2                  # a new speed stretch after the gap: spoken
    assert not p.offer("too_far", 17.0) and p.stretches["distance"] == 2             # a new distance stretch, held by the new gap
    assert p.issued == {"slow_down": 2}


def test_recapture_covers_slow_down_and_silences_the_expected_revisit():
    p = NudgePolicy(rearm_s=6, gap_s=0)
    assert p.offer("recapture", 1.0)
    assert not p.offer("slow_down", 1.2)                                              # "film it again, slowly" already said it
    assert not p.offer("revisit", 3.0)                                                # going back was asked for
    assert p.offer("revisit", 20.0)                                                   # a later, separate revisit is a problem again


def test_capture_quality_and_monitor_share_the_engine_policy():
    p = NudgePolicy(rearm_s=6, gap_s=15)
    cq, mon = CaptureQuality({}, policy=p), CaptureMonitor(p)
    out = []
    for i in range(150):                                                              # 30 s: blurry AND fast the whole time
        t = i / 5
        out += cq.add(t, 900, 30, 0.0)
        out += mon.frame(["blurry", "fast_motion"], t, 0)
    assert [n["code"] for n in out] == ["slow_down", "recapture"]                     # old policy: 3 slow_down + 3 blur + 3 fast_pan + 2 recapture
    s = cq.summary()
    assert s["nudges_issued"] == {"slow_down": 1, "recapture": 1} and s["nudge_policy"]["stretches"] == {"speed": 1}
    assert [x["t_s"] for x in s["nudge_policy"]["spoken"]] == [0.4, 15.4]             # recapture only once the 15 s gap clears
    assert cq.now == "bad"                                                            # the on-screen indicator keeps showing the state


def test_policy_is_tunable(monkeypatch):
    monkeypatch.setenv("LIBRARY_NUDGE_REARM_S", "3")
    monkeypatch.setenv("LIBRARY_NUDGE_GAP_S", "20")
    p = NudgePolicy()
    assert (p.rearm_s, p.gap_s) == (3.0, 20.0)
    assert NudgePolicy(rearm_s=6, gap_s=15).gap_s == 15.0


def test_messages_are_short_and_direct():
    from app.capture_quality import MESSAGES
    for msg in [*MESSAGES.values(), *CaptureMonitor.MESSAGES.values()]:
        assert len(msg.split()) <= 16 and "please" not in msg.lower(), msg
