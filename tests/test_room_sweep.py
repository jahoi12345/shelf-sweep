"""The room-sweep instruction comes at the END of the shelf pass (app/room_sweep.py), never during it; End then ends; frames after
the prompt are tagged as room pan (pipeline.room_pan rule f). Synthetic only."""
import json

import pytest

from app import live_tools as LT
from app.room_sweep import LEFT_MESSAGE, ROOM_MESSAGE, RoomSweep
from pipeline import room_pan as RP
from tests.test_live_app import Harness, feed_panning, tool_event
from tests.test_room_pan import _sweep


# ------------------------------------------------------------------ pure state machine
def test_first_end_starts_the_room_pan_and_the_second_end_or_skip_ends():
    r = RoomSweep(enabled=True)
    first = r.request_end(80.0, "f00400", source="button")
    assert first == {"action": "room_sweep", "message": ROOM_MESSAGE} and r.phase == "room" and r.muted
    assert r.record()["prompt"] == {"t_s": 80.0, "frame_id": "f00400", "trigger": "end_request_button"}
    assert r.request_end(95.0, "f00470", source="button")["action"] == "end"
    assert r.record()["prompt"]["t_s"] == 80.0                                            # the prompt time is not moved by the end
    for kw in ({"skip": True}, {"camera": False}):                                        # "skip the room", replay / no camera: end right away
        r2 = RoomSweep(enabled=True)
        assert r2.request_end(10.0, "f1", source="voice", **kw)["action"] == "end" and r2.phase == "shelf" and r2.prompt is None
    assert RoomSweep(enabled=False).request_end(10.0, "f1", source="button")["action"] == "end"
    for w in ("floor", "corner", "ceiling", "opposite wall", "End"):
        assert w in ROOM_MESSAGE and (w in LEFT_MESSAGE or w == "opposite wall" and "opposite wall" in LEFT_MESSAGE)


def _feed(r, items):
    out = []
    for t, books, spine in items:
        g = r.observe_detection(t, f"f{int(t * 10):05d}", books, [spine] * books)
        if g:
            out.append((t, g))
    return out


def test_left_the_shelf_trigger_fires_only_after_the_shelf_pass_and_a_real_room_pan():
    shelf = [(0.4 * i, 5, 300.0) for i in range(1, 150)]                                   # 60 s of close shelf frames
    r = RoomSweep(enabled=True)
    glance = [(20.0, 0, 0), (20.4, 0, 0), (20.8, 0, 0)]                                    # a glance away mid-shelf: 0.8 s
    early = [(10.1, 0, 0), (12.1, 0, 0), (14.1, 0, 0), (16.1, 0, 0)]                       # 6 s off the shelf, but only 10 s into the sweep
    assert _feed(r, sorted(shelf[:20] + early, key=lambda x: x[0])) == []
    assert _feed(r, sorted(shelf[20:] + glance, key=lambda x: x[0])) == [] and r.phase == "shelf"
    pan = [(61.0, 0, 0), (62.0, 2, 80.0), (63.0, 0, 0), (64.0, 0, 0), (65.5, 0, 0)]        # floor / far overview / ceiling, 4.5 s
    got = _feed(r, list(reversed(pan)))                                                     # detections finish out of order
    assert len(got) == 1 and got[0][1]["code"] == "room_sweep" and r.phase == "room" and r.prompt["trigger"] == "left_shelf"
    assert r.request_end(70.0, "f00700", source="button")["action"] == "end"             # End after the trigger ends right away


# ------------------------------------------------------------------ room-pan tagging after the prompt
def test_a_short_room_pan_after_the_prompt_is_tagged_and_close_shelf_frames_stay():
    kinds = 's.' * 30 + 'f.n.f.n' + 's.s.'                                                 # a 1.4 s far/off-shelf pan, then back at the shelf
    frames, obs, books, det = _sweep(kinds)
    assert RP.classify_frames(frames, obs, books, detection_frames=det)['room_pan_frames'] == []    # room-pan rule alone: too short
    t_prompt = frames[kinds.index('f')]['timestamp_s'] - 0.2
    res = RP.classify_frames(frames, obs, books, detection_frames=det, room_sweep_t=t_prompt)
    tagged = set(res['room_pan_frames'])
    last_shelf_before = max(i for i, k in enumerate(kinds) if k == 's' and frames[i]['timestamp_s'] < t_prompt)
    assert tagged == {f['frame_id'] for i, (f, k) in enumerate(zip(frames, kinds)) if i > last_shelf_before and k != 's'}   # every non-shelf frame from there on
    assert not tagged & {f['frame_id'] for f, k in zip(frames, kinds) if k == 's'} and res['segments'][0]['after_room_sweep_prompt']
    # YugoTest13-like: a pan at the START with no prompt is still found by the room-pan rule (rule f adds, never removes)
    start = 'n.n.n.n.ffff.n.n.n.n.ffff.n.n.n.n.n.n.n' + 's.s.s.s.s.s'
    fr, ob, bk, de = _sweep(start)
    assert RP.classify_frames(fr, ob, bk, detection_frames=de, room_sweep_t=fr[-1]['timestamp_s'] + 1)['room_pan_frames'] == \
        RP.classify_frames(fr, ob, bk, detection_frames=de)['room_pan_frames'] != []


def test_classify_run_reads_the_prompt_from_the_manifest(tmp_path):
    kinds = 's.' * 30 + 'f.n.f.n'
    frames, obs, books, det = _sweep(kinds)
    t_prompt = frames[60]['timestamp_s']
    env = lambda d: json.dumps({'data': d})
    (tmp_path / 'manifest.json').write_text(env({'frames': frames, 'room_sweep': {'prompt': {'t_s': t_prompt, 'frame_id': 'f00060', 'trigger': 'end_request_button'}}}))
    (tmp_path / 'candidates.json').write_text(env({'observations': obs}))
    (tmp_path / 'inventory.json').write_text(env({'books': books}))
    res = RP.classify_run(tmp_path, recon3d=False)
    assert res['room_sweep_prompt_t'] == t_prompt and res['room_pan_frame_count'] == 8 and RP.summary(res)['room_sweep_prompt_t'] == t_prompt


# ------------------------------------------------------------------ end-of-sweep flow through the live socket and the voice tool
@pytest.mark.asyncio
async def test_end_button_flow_room_pan_then_end(tmp_path):
    async with Harness(tmp_path) as h:
        eng = h.session.engine
        await h.client({"type": "camera_state", "enabled": True})
        await feed_panning(h, 6)
        await eng._guidance(60.0)
        assert not [m for m in h.sent if m["type"] == "nudge" and m.get("kind") == "guidance"]          # nothing about the room during the shelf pass
        await h.client({"type": "end_request"})
        await h.wait(lambda: any(m["type"] == "nudge" and m.get("code") == "room_sweep" for m in h.sent))
        await h.wait(lambda: any(m["type"] == "end_ack" for m in h.sent))
        assert eng.status == "capturing" and eng.nudge_policy.muted and [m["action"] for m in h.sent if m["type"] == "end_ack"] == ["room_sweep"]
        await h.wait(lambda: any(hasattr(c.get("turns"), "parts") and "APP GUIDANCE" in c["turns"].parts[0].text and "room" in c["turns"].parts[0].text
                                 for c in h.live.content))                                  # the agent says it (button path)
        assert eng.nudge_policy.offer("slow_down", 99.0) is False                           # capture notices muted during the room pan
        await h.client({"type": "end_request"})                                             # second press
        await h.wait(lambda: len([m for m in h.sent if m["type"] == "end_ack"]) == 2)
        assert [m for m in h.sent if m["type"] == "end_ack"][-1]["action"] == "end"
        await h.client({"type": "end_sweep"})
        await h.wait(lambda: eng.status == "finished", timeout=20)
        man = json.loads((eng.run_dir / "manifest.json").read_text())["data"]
        assert man["room_sweep"]["prompt"]["trigger"] == "end_request_button" and man["room_sweep"]["prompt"]["frame_id"] == eng.frames[-1]["frame_id"]


@pytest.mark.asyncio
async def test_voice_done_says_the_room_instruction_then_ends_on_skip(tmp_path):
    async with Harness(tmp_path) as h:
        eng = h.session.engine
        await h.client({"type": "camera_state", "enabled": True})
        await feed_panning(h, 6)
        n_content = len(h.live.content)
        await h.live.q.put(tool_event("e1", "end_sweep", {}))
        await h.wait(lambda: h.live.tool_responses, timeout=20)
        r = h.live.tool_responses[-1].response
        assert r["room_sweep_first"] and r["sweep_ended"] is False and r["say"] == ROOM_MESSAGE and eng.status == "capturing"
        assert not any(hasattr(c.get("turns"), "parts") and "APP GUIDANCE" in c["turns"].parts[0].text for c in h.live.content[n_content:])   # not said twice
        assert any(m["type"] == "nudge" and m.get("code") == "room_sweep" for m in h.sent)  # the screen shows it and relabels End
        await h.live.q.put(tool_event("e2", "end_sweep", {"skip_room": True}))
        await h.wait(lambda: len(h.live.tool_responses) >= 2, timeout=20)
        assert eng.status == "finished" and "book_count" in h.live.tool_responses[-1].response


def test_prompt_and_tool_say_room_only_after_the_shelf():
    low = LT.SYSTEM_INSTRUCTION.lower()
    assert "never mention the room" in low and "skip_room" in low and "room_sweep_first" in low
    end = next(f for t in LT.tool_declarations() for f in t.function_declarations if f.name == "end_sweep")
    assert "skip_room" in end.parameters.properties


# ------------------------------------------------------------------ end of sweep is not a failure (YugoTest14/15)
@pytest.mark.asyncio
async def test_frames_after_end_are_dropped_silently_and_a_closing_socket_still_queues_the_post_sweep_builds(tmp_path, caplog):
    from unittest.mock import patch
    from app import server as s
    from tests.test_live_app import make_frame
    import asyncio
    from app.engine import SweepEngine
    queued, alt = [], {"started": False, "done": False}

    async def slow_alt(self, packet_path):          # the alternative count after the packet: ~50-90 s of CPU on YugoTest14/15
        alt["started"] = True
        await asyncio.sleep(60)
        alt["done"] = True

    with patch.object(s.post_sweep, "enqueue", side_effect=lambda run_id: queued.append(run_id)), \
            patch.object(SweepEngine, "_alt_count_after_packet", slow_alt):
        async with Harness(tmp_path) as h:
            eng = h.session.engine
            eng.alt_after_packet = True
            await feed_panning(h, 6)
            await h.client({"type": "end_sweep"})
            await h.wait(lambda: eng.status != "capturing")
            for i in range(3):                                                             # Wi-Fi burst still in flight after End
                await h.frame(make_frame(shift=3000 + i * 500), t=10 + i * 0.2)
            await h.client({"type": "end_sweep"})                                          # a second End: no-op
            await h.wait(lambda: eng.counters["frames_after_end_dropped"] == 3)
            await h.client({"type": "close"})                                              # the phone leaves right away
            await h.wait(lambda: eng.status == "finished", timeout=30)
            await h.wait(lambda: queued, timeout=10)
            assert not [m for m in h.sent if m["type"] == "error"], [m for m in h.sent if m["type"] == "error"]
            assert len(eng.frames) == 6 and queued == [eng.run_dir.name]
            assert not alt["done"]                    # queued as soon as the packet is on disk, not after the 60 s alternative count
    assert not [r for r in caplog.records if "Background sweep task failed" in r.getMessage() or "Live session failed" in r.getMessage()]


def test_client_stops_frames_at_end_and_shows_processing_not_failed():
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "app" / "static" / "app.js").read_text()
    assert "return !sweepEnding && (cameraStream || replayActive)" in js                  # every frame path checks frameActive()
    assert "sweepEnding = true;                                  // the frame timers stop" in js
    assert 'word: "Sweep finished — processing"' in js
    assert 'if (state && state.status === "failed") endFailure = message.message;' in js
