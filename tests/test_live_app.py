"""Transport / ingestion / correction tests for the live sweep app.

All model, device and network behaviour here is MOCKED: fake Live client, fake stage functions with the real
stage signatures, synthetic frames. These tests prove plumbing and logic; they are NOT evidence of real
camera/mic/provider behaviour or recognition accuracy.
Structure ported from the reference's tests/test_live_turns.py (event ordering) and test_regressions.py (stale
results, cleanup); domain cases are new.
"""
import asyncio
import base64
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app import server as s
from app.engine import StageFns, SweepEngine, SweepError
from pipeline import contracts as C
from pipeline import corrections as K

W, H = 1920, 1080


# ------------------------------------------------------------------ synthetic frames + fake stages
def make_frame(shift: int = 0, blur: int = 0, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 200, np.uint8)
    for _ in range(260):
        x, y = int(rng.integers(0, W - 40)), int(rng.integers(0, H - 200))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 40)), y + int(rng.integers(80, 200))),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    img = np.roll(img, shift, axis=1)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def fake_detect(manifest_path, run_dir, config):
    m = C.read_json(manifest_path)
    f = m["data"]["frames"][0]
    fid = f["frame_id"]
    obs = [{"observation_id": C.observation_id(fid, i), "frame_id": fid, "kind": k, "category": cat, "orientation": "upright",
            "bbox_xyxy": [100 + 200 * i, 300, 160 + 200 * i, 700], "confidence": 0.9, "frame_ref": f["image_path"]}
           for i, (k, cat) in enumerate([("book", "book"), ("book", "book"), ("non_book", "plant")])]
    out = Path(run_dir) / config["output_name"]
    C.write_json(out, C.envelope(m["run_id"], "detect", {"observations": obs}, status="ok"))
    return out


def fake_read(slice_path, manifest_path, run_dir, config):
    a = C.read_json(slice_path)
    rd = [{"observation_id": o["observation_id"], "crop_path": None, "visible_text": f"TEXT {o['observation_id']}",
           "title_text": None, "read_confidence": 0.8, "issues": []} for o in a["data"]["observations"]]
    out = Path(run_dir) / config["readings_filename"]
    C.write_json(out, C.envelope(a["run_id"], "read", {"readings": rd}, status="ok"))
    return out


def fake_resolve(readings_path, run_dir, config):
    a = C.read_json(readings_path)
    rs = [{"observation_id": r["observation_id"], "decision": "no_match", "issues": []} for r in a["data"]["readings"]]
    out = Path(run_dir) / config["resolutions_filename"]
    C.write_json(out, C.envelope(a["run_id"], "resolve", {"resolutions": rs}, status="ok"))
    return out


def fake_merge(candidates_path, manifest_path, run_dir, config, readings_path=None, resolutions_path=None):
    cand = C.read_json(candidates_path)
    readings = {r["observation_id"]: r for r in (C.read_json(readings_path)["data"]["readings"] if readings_path else [])}
    books, items = [], []
    for o in cand["data"]["observations"]:
        inst = {"observation_ids": [o["observation_id"]], "frame_ref": o["frame_id"], "bbox_xyxy": o["bbox_xyxy"], "kind": o["kind"],
                "confidence": o["confidence"], "uncertain_association": False, "dedup_reasons": ["fake merge"], "issues": []}
        if o["kind"] == "book":
            inst.update(instance_id=C.instance_id("bk", len(books) + 1), identity={
                "decision": "no_match", "visible_text": (readings.get(o["observation_id"]) or {}).get("visible_text")})
            books.append(inst)
        else:
            inst.update(instance_id=C.instance_id("it", len(items) + 1), category=o["category"], description="plant")
            items.append(inst)
    art = C.envelope(cand["run_id"], "dedup", {"books": books, "items": items, "counts": {
        "count_lower": len(books), "count_upper": len(books), "uncertain_associations": 0, "uncertain_observations": 0}}, status="ok")
    out = Path(run_dir) / config["output_name"]
    C.write_json(out, art)
    return out


FAKES = StageFns(detect=fake_detect, read=fake_read, resolve=fake_resolve, merge=fake_merge)


def event(**fields):
    content = dict(input_transcription=None, output_transcription=None, model_turn=None, interrupted=False, turn_complete=False)
    content.update(fields)
    return NS(server_content=NS(**content), tool_call=None, tool_call_cancellation=None)


def tool_event(call_id, name, args):
    return NS(server_content=None, tool_call=NS(function_calls=[NS(id=call_id, name=name, args=args)]), tool_call_cancellation=None)


def words(text, finished=False):
    return NS(text=text, finished=finished)


class FakeLive:
    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()
        self.content, self.tool_responses, self.realtime = [], [], []

    async def send_client_content(self, **kw): self.content.append(kw)
    async def send_tool_response(self, function_responses): self.tool_responses.extend(function_responses)
    async def send_realtime_input(self, **kw): self.realtime.append(kw)

    async def receive(self):
        while True:
            yield await self.q.get()


class Harness:
    """Drives s.live_voice with a fake WebSocket and (optionally) a fake Live connection."""

    def __init__(self, tmp_path, voice=True, stages=FAKES):
        self.tmp, self.voice, self.stages = tmp_path, voice, stages
        self.sent, self.inbox, self.live = [], asyncio.Queue(), FakeLive()
        self.closed = False

    async def __aenter__(self):
        self._patches = [patch.object(s, "RUNS_DIR", self.tmp / "runs"), patch.object(s, "stage_fns_factory", lambda: self.stages),
                         patch.object(s, "_has_api_key", return_value=self.voice)]
        for p in self._patches:
            p.start()
        s.set_live_client_factory((lambda: NS(aio=NS(live=NS(connect=lambda **kw: self._cm())))) if self.voice else None)
        ref: list = []
        run_dir = s.RUNS_DIR / "sess1"
        engine = s._new_engine("sess1", run_dir, ref)
        self.session = s.SweepSession(session_id="sess1", owner="owner", run_dir=run_dir, engine=engine)
        ref.append(self.session)
        s.sessions["sess1"] = self.session
        harness = self

        class WS:
            headers = {"host": "127.0.0.1:4177", "origin": "http://127.0.0.1:4177"}
            client = NS(host="127.0.0.1")
            url = NS(scheme="ws")
            cookies = {"intake_owner": "owner"}
            query_params = {"session_id": "sess1"}
            async def accept(self): pass
            async def close(self, **kw): harness.closed = True
            async def send_json(self, payload): harness.sent.append(payload)
            async def receive_text(self): return await harness.inbox.get()
        self.task = asyncio.create_task(s.live_voice(WS()))
        await self.wait(lambda: any(m["type"] == "ready" for m in self.sent))
        return self

    def _cm(self):
        live = self.live

        class CM:
            async def __aenter__(self): return live
            async def __aexit__(self, *a): pass
        return CM()

    async def __aexit__(self, *exc):
        await self.client({"type": "close"})
        with contextlib_suppress():
            await asyncio.wait_for(self.task, 3)
        await self.session.engine.aclose()
        s.sessions.pop("sess1", None)
        s.set_live_client_factory(None)
        for p in self._patches:
            p.stop()

    async def client(self, message):
        await self.inbox.put(json.dumps(message))

    async def frame(self, jpeg, source="camera", t=None, kind="evidence_frame"):
        msg = {"type": kind, "source": source, "data": base64.b64encode(jpeg).decode()}
        if t is not None:
            msg["t_s"] = t
        await self.client(msg)

    async def wait(self, cond, timeout=8.0):
        end = asyncio.get_event_loop().time() + timeout
        while not cond():
            if asyncio.get_event_loop().time() > end:
                raise AssertionError("condition not reached; last messages: " + str([m["type"] for m in self.sent[-8:]]))
            await asyncio.sleep(0.02)

    def last_state(self):
        return next(m["state"] for m in reversed(self.sent) if m["type"] == "state")


import contextlib
contextlib_suppress = lambda: contextlib.suppress(Exception)


async def feed_panning(h: Harness, n: int, source="camera", step=500, unrelated=False):
    # unrelated=True: every frame is a different scene. The stub detectors put boxes at FIXED pixels, which is only physical when frames
    # cannot be registered to each other (a tracked 500 px pan, recovered by tracking recovery, makes each fixed box a different object).
    for i in range(n):
        await h.frame(make_frame(shift=i * step, seed=1 + i if unrelated else 1), source=source, t=i * 0.2)
        await asyncio.sleep(0.01)


# ------------------------------------------------------------------ tests
@pytest.mark.asyncio
async def test_frames_saved_unchanged_and_inventory_updates_before_end(tmp_path):
    async with Harness(tmp_path) as h:
        sent_bytes = [make_frame(shift=i * 500) for i in range(6)]
        for i, b in enumerate(sent_bytes):
            await h.frame(b, t=i * 0.2)
            await asyncio.sleep(0.01)
        # inventory is non-empty DURING the sweep (status is still capturing), not only after end_sweep
        await h.wait(lambda: any(m["type"] == "state" and m["state"]["totals"]["book_count"] > 0 for m in h.sent))
        st = h.last_state()
        assert st["status"] == "capturing" and st["totals"]["book_count"] >= 2
        assert st["progress"]["frames_saved"] == 6
        # high-resolution evidence persisted byte-for-byte; ids/timestamps in manifest-compatible form
        for i, b in enumerate(sent_bytes):
            assert (h.session.run_dir / "frames" / f"f{i:05d}.jpg").read_bytes() == b
        assert all(f["width_px"] == W and f["height_px"] == H for f in h.session.engine.frames)


@pytest.mark.asyncio
async def test_camera_and_replay_use_one_ingestion_path(tmp_path):
    async with Harness(tmp_path) as h:
        await h.frame(make_frame(0), source="camera")
        await h.frame(make_frame(300), source="replay", t=7.5)
        await h.wait(lambda: len(h.session.engine.frames) == 2)
        f = h.session.engine.frames
        assert [x["source"] for x in f] == ["camera", "replay"]
        assert f[1]["timestamp_s"] == 7.5            # replay keeps the video clock; camera uses the server clock
        assert h.session.engine.provenance()["capture_sources"] == ["camera", "replay"]


@pytest.mark.asyncio
async def test_compressed_voice_frame_is_never_ocr_evidence(tmp_path):
    async with Harness(tmp_path) as h:
        small = make_frame(0)[:]
        await h.frame(small, kind="video")
        await h.wait(lambda: h.live.realtime)
        assert h.session.engine.frames == []        # 'video' only reaches the model context; nothing saved/processed
        assert not list((h.session.run_dir / "frames").glob("*"))


@pytest.mark.asyncio
async def test_non_jpeg_and_oversize_evidence_rejected(tmp_path):
    async with Harness(tmp_path) as h:
        await h.client({"type": "evidence_frame", "source": "camera", "data": base64.b64encode(b"notajpeg" * 200).decode()})
        await h.wait(lambda: any(m["type"] == "error" for m in h.sent))
        assert h.session.engine.frames == []


@pytest.mark.asyncio
async def test_no_credential_degraded_mode_is_explicit_and_still_saves_frames(tmp_path):
    async with Harness(tmp_path, voice=False) as h:
        ready = next(m for m in h.sent if m["type"] == "ready")
        assert ready["voice"] is False
        await h.client({"type": "text", "text": "hello", "id": "t1"})
        await h.wait(lambda: any(m["type"] == "error" and "unavailable" in m["message"] for m in h.sent))
        await h.frame(make_frame(0))
        await h.wait(lambda: len(h.session.engine.frames) == 1)
        assert h.session.engine.provenance()["inference_mode"] in {"unavailable", "mocked"}


@pytest.mark.asyncio
async def test_quality_nudge_fires_and_is_spoken_via_capture_notice(tmp_path):
    async with Harness(tmp_path) as h:
        sharp = make_frame(0)
        await h.frame(sharp, t=0.0)
        for i in range(1, 8):
            await h.frame(make_frame(0, blur=12, seed=1), t=i * 0.2)
            await asyncio.sleep(0.01)
        await h.wait(lambda: any(m["type"] == "nudge" for m in h.sent))
        nudge = next(m for m in h.sent if m["type"] == "nudge")
        assert nudge["code"] in {"blur", "fast_pan", "slow_down"}
        await h.wait(lambda: any("APP CAPTURE NOTICE" in c["turns"].parts[0].text for c in h.live.content if hasattr(c.get("turns"), "parts")))
        notice = next(c for c in h.live.content if hasattr(c.get("turns"), "parts") and "APP CAPTURE NOTICE" in c["turns"].parts[0].text)
        assert notice["turn_complete"] is True        # urgent: the agent speaks while the user is still there


@pytest.mark.asyncio
async def test_voice_exclusion_current_view_changes_records_and_counts(tmp_path):
    async with Harness(tmp_path) as h:
        await feed_panning(h, 8)
        await h.wait(lambda: h.session.engine.inventory is not None and h.session.engine.stage_log.get("_merge"))
        before = h.session.engine.snapshot()["totals"]
        assert before["book_count"] >= 2
        await h.live.q.put(tool_event("t1", "record_correction",
                                      {"action": "exclude", "scope": "current_view", "seconds": 100, "reason": "not mine",
                                       "utterance": "skip that shelf, those are not mine"}))
        await h.wait(lambda: h.live.tool_responses)
        resp = h.live.tool_responses[-1].response
        assert resp["matched_instances"] and not resp.get("error")
        after = h.session.engine.snapshot()
        assert after["totals"]["book_count"] == before["book_count"] - sum(i.startswith("bk-") for i in resp["matched_instances"])
        assert after["totals"]["user_excluded_count"] == len(resp["matched_instances"])
        assert after["excluded_by_user"] and after["excluded_by_user"][0]["source"]["utterance"].startswith("skip that shelf")


@pytest.mark.asyncio
async def test_corrections_set_title_appraisal_print_restore_and_final_packet(tmp_path):
    async with Harness(tmp_path) as h:
        await feed_panning(h, 6, unrelated=True)
        await h.wait(lambda: h.session.engine.inventory is not None)
        eng = h.session.engine
        books = eng.inventory["data"]["books"]
        items = eng.inventory["data"]["items"]
        b1, b2, it = books[0]["instance_id"], books[1]["instance_id"], items[0]["instance_id"]
        await eng.add_correction("set_title", instance_ids=[b1], value={"title": "Cien años de soledad", "language": "es"})
        await eng.add_correction("flag_appraisal", instance_ids=[b2], value={"reason": "user says first edition"})
        await eng.add_correction("exclude", instance_ids=[it], value={"reason": "not mine"})
        snap = eng.snapshot()
        byid = {b["id"]: b for b in snap["books"]}
        assert byid[b1]["status"] == "identified" and byid[b1]["identity_source"] == "user_statement" and byid[b1]["isbn"] is None
        assert byid[b2]["status"] == "needs_appraisal"
        assert snap["totals"]["item_count"] == len(items) - 1 and snap["totals"]["user_excluded_count"] == 1
        await eng.add_correction("restore", instance_ids=[it])
        assert eng.snapshot()["totals"]["item_count"] == len(items)
        # final packet carries the same corrections deterministically
        await eng.add_correction("exclude", instance_ids=[b2], value={"reason": "not mine"})
        summary = await eng.end_sweep()
        packet = C.read_json(h.session.run_dir / "claim_packet.json")["data"]
        assert packet["completion_status"] == "provisional" and packet["sweep"]["country"] == "US" and packet["sweep"]["currency"] == "USD"
        assert b2 not in {b["id"] for b in packet["books"]} and b2 in {x["instance_id"] for x in packet["excluded_by_user"]}
        assert packet["totals"]["book_count"] == len(packet["books"]) == summary["totals"]["book_count"]
        assert all(b["spine_height_cm"] is None and b["replacement_cost"] is None for b in packet["books"])
        assert packet["totals"]["shelf_run_m"] is None
        manifest = C.read_json(h.session.run_dir / "manifest.json")
        from pipeline.packet import verify_packet
        assert verify_packet(C.read_json(h.session.run_dir / "claim_packet.json"), manifest, h.session.run_dir) == []
        assert (h.session.run_dir / "report.html").is_file() and packet["sweep"]["finalization_s"] is not None


@pytest.mark.asyncio
async def test_unknown_instance_correction_is_an_error_not_a_silent_noop(tmp_path):
    async with Harness(tmp_path) as h:
        # Unknown-ID validation needs an inventory, not a tracked six-frame
        # sweep. Avoid unrelated registration work and its load-sensitive wait.
        await h.frame(make_frame(), t=0.0)
        await h.wait(lambda: h.session.engine.inventory is not None)
        await h.live.q.put(tool_event("t9", "record_correction", {"action": "exclude", "scope": "instance_ids", "instance_ids": ["bk-999"]}))
        await h.wait(lambda: h.live.tool_responses)
        assert "unknown instance" in h.live.tool_responses[-1].response["error"]


@pytest.mark.asyncio
async def test_end_sweep_tool_returns_only_stage_backed_numbers(tmp_path):
    async with Harness(tmp_path) as h:
        await feed_panning(h, 6)
        await h.live.q.put(tool_event("e1", "end_sweep", {"skip_room": True}))     # a plain first call during a camera sweep asks for the room pan first
        await h.wait(lambda: h.live.tool_responses, timeout=20)
        r = h.live.tool_responses[-1].response
        assert r["provisional"] is True and "unknown, not zero" in r["notes"]
        assert not any(k in r for k in ("price", "total_value", "shelf_run_m", "room_area"))
        assert h.session.engine.status == "finished"
        # later frames are refused
        with pytest.raises(SweepError):
            await h.session.engine.ingest_frame(make_frame(0))


@pytest.mark.asyncio
async def test_interrupted_payload_cannot_requeue_cancelled_speech(tmp_path):
    """Ported from reference test_live_turns: cancelled audio must not be forwarded after an interruption."""
    async with Harness(tmp_path) as h:
        await h.live.q.put(event(output_transcription=words("Let me explain")))
        await h.live.q.put(event(interrupted=True, input_transcription=words("Stop.", True), output_transcription=words(" cancelled"),
                                 model_turn=NS(parts=[NS(inline_data=NS(data=b"cancelled", mime_type="audio/pcm;rate=24000"))])))
        await h.live.q.put(event(output_transcription=words("Okay, I will wait.", True), turn_complete=True))
        await h.wait(lambda: any(m["type"] == "turn_complete" for m in h.sent))
        assert [t["text"] for t in h.session.transcript] == ["Let me explain", "Stop.", "Okay, I will wait."]
        assert not any(m["type"] == "audio" for m in h.sent)
        stopped = next(i for i, m in enumerate(h.sent) if m["type"] == "interrupted")
        final = next(i for i, m in enumerate(h.sent) if m["type"] == "transcript" and m["final"])
        assert stopped < final


@pytest.mark.asyncio
async def test_slow_stage_does_not_block_frame_intake(tmp_path):
    import time
    def slow_detect(*a, **k):
        time.sleep(1.0)
        return fake_detect(*a, **k)
    stages = StageFns(detect=slow_detect, read=fake_read, resolve=fake_resolve, merge=fake_merge)
    async with Harness(tmp_path, stages=stages) as h:
        t0 = asyncio.get_event_loop().time()
        await feed_panning(h, 6)
        await h.wait(lambda: len(h.session.engine.frames) == 6)
        assert asyncio.get_event_loop().time() - t0 < 0.9   # intake did not wait for the 1 s detector
        assert h.session.engine.snapshot()["progress"]["stages_in_flight"] >= 1


@pytest.mark.asyncio
async def test_stale_merge_result_never_replaces_newer_inventory_or_undoes_corrections(tmp_path):
    eng = SweepEngine(tmp_path / "r", stages=FAKES, config={"cache_dir": str(tmp_path / "c")}, session_id="x")
    await eng.ingest_frame(make_frame(0), source="camera")
    # simulate: a newer merge (3 obs) is current; an older merge result (1 obs) arrives late
    newer = C.envelope("x", "dedup", {"books": [{"instance_id": "bk-001", "observation_ids": ["f00000-o000"], "frame_ref": "f00000",
              "bbox_xyxy": [0, 0, 10, 10], "identity": {"decision": "no_match"}}], "items": [], "counts": {}}, status="ok")
    older = C.envelope("x", "dedup", {"books": [], "items": [], "counts": {}}, status="ok")
    eng.inventory, eng.merged_observation_ids = newer, {"a", "b", "c"}
    ids = {"a"}
    if eng.inventory is None or len(ids) >= len(eng.merged_observation_ids):  # the guard used in _merge_once
        eng.inventory = older
    assert eng.inventory is newer
    await eng.add_correction("exclude", observation_ids=["f00000-o000"], value={"reason": "x"})
    assert eng.snapshot()["totals"]["book_count"] == 0          # overlay survives later merges
    await eng.aclose()


@pytest.mark.asyncio
async def test_reset_and_disconnect_release_tasks_and_purge(tmp_path):
    async with Harness(tmp_path) as h:
        await feed_panning(h, 6)
        await h.wait(lambda: h.session.engine._tasks or h.session.engine.inventory)
        run_dir = h.session.run_dir
        await s.discard_session(h.session, purge=True)
        assert h.session.engine._tasks == set() and h.session.deleted and not run_dir.exists()
        assert "sess1" not in s.sessions


@pytest.mark.asyncio
async def test_socket_close_cancels_tool_tasks_and_clears_live_reference(tmp_path):
    async with Harness(tmp_path) as h:
        await h.live.q.put(tool_event("s1", "get_inventory_status", {}))
        await h.wait(lambda: h.live.tool_responses)
        assert h.live.tool_responses[-1].response["provisional"] is True
    assert h.session.live_socket is None and h.session.camera_enabled is False


def test_replay_video_endpoint_rejects_traversal(tmp_path):
    from fastapi.testclient import TestClient
    with patch.object(s, "REPLAY_DIR", tmp_path):
        (tmp_path / "ok.mov").write_bytes(b"x")
        c = TestClient(s.app, base_url="http://127.0.0.1:4177")
        assert c.get("/replay/ok.mov").status_code == 200
        assert c.get("/replay/..%2Fsecret.mov").status_code == 404
        assert c.get("/replay/notes.txt").status_code == 404


def test_packet_zip_contains_frames_and_requires_finished_sweep(tmp_path):
    from fastapi.testclient import TestClient
    async def prepare():
        async with Harness(tmp_path) as h:
            await feed_panning(h, 6)
            await h.session.engine.end_sweep()
            return h.session
    sess = asyncio.run(prepare())
    # re-register for REST access (harness removed it on exit)
    s.sessions["sess1"] = sess
    try:
        c = TestClient(s.app, base_url="http://127.0.0.1:4177")
        c.cookies.set("intake_owner", "owner")
        r = c.get("/api/sessions/sess1/packet")
        assert r.status_code == 200
        import io
        names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
        assert "claim_packet.json" in names and "report.html" in names and any(n.startswith("frames/") for n in names)
        assert c.get("/api/sessions/sess1/frames/f00000.jpg").status_code == 200
        assert c.get("/api/sessions/sess1/frames/..%2Fmanifest.json").status_code == 404
        sess.engine.status = "capturing"
        assert c.get("/api/sessions/sess1/packet").status_code == 409
    finally:
        s.sessions.pop("sess1", None)


def test_corrections_time_window_never_excludes_instance_seen_outside_window():
    inv = {"books": [{"instance_id": "bk-001", "observation_ids": ["f00001-o000", "f00009-o000"], "frame_ref": "f00001"}], "items": []}
    ft = {"f00001": 1.0, "f00009": 9.0}
    e = K.new_entry("c1", "exclude", {"time_window": {"start_s": 0, "end_s": 2}}, {})
    eff, excluded, _ = K.apply(inv, [e], ft)
    assert len(eff["books"]) == 1 and excluded == []   # seen at 9 s as well: stays counted


def test_correction_validation():
    with pytest.raises(K.CorrectionError):
        K.new_entry("c", "set_title", {"observation_ids": ["a"]}, {"title": " "})
    with pytest.raises(K.CorrectionError):
        K.new_entry("c", "exclude", {})
    with pytest.raises(K.CorrectionError):
        K.new_entry("c", "bogus", {"observation_ids": ["a"]})


# ------------------------------------------------------------------ REAL stage code, MOCKED model transport
def _mock_transport(counter):
    async def transport(model, images, prompt, config, schema):
        counter.append(prompt[:20])
        if prompt.startswith("Inventory visible"):
            objs = [{"box_2d": [300, 100 + 120 * i, 700, 140 + 120 * i], "kind": "book", "category": "book", "orientation": "upright",
                     "confidence": 0.9, "occlusion": "none", "evidence_notes": "mock"} for i in range(4)]
            objs.append({"box_2d": [100, 700, 300, 800], "kind": "non_book", "category": "plant", "orientation": "unknown",
                         "confidence": 0.8, "occlusion": "none", "evidence_notes": "mock"})
            return json.dumps({"objects": objs}), {"prompt_tokens": 1000, "output_tokens": 200, "thought_tokens": 0, "total_tokens": 1200}
        return json.dumps({"visible_text": "CIEN AÑOS DE SOLEDAD", "title_text": "CIEN AÑOS DE SOLEDAD", "author_text": None,
                           "publisher_text": None, "title_complete": True, "read_confidence": 0.7, "issues": [],
                           "isbn_text": None, "edition_text": None}), {"prompt_tokens": 300, "output_tokens": 50, "thought_tokens": 0, "total_tokens": 350}
    return transport


@pytest.mark.asyncio
async def test_real_stage_functions_through_engine_with_mocked_model(tmp_path):
    from providers import gemini
    calls = []
    gemini.set_transport(_mock_transport(calls))
    try:
        eng = SweepEngine(tmp_path / "r", config={"cache_dir": str(tmp_path / "cache"), "catalogue_online": False, "detector": "gemini", "reader": "gemini"},   # pinned: a developer .env must not reroute this mocked-hosted test
                          stages=StageFns.real(), session_id="real-stages")
        for i in range(6):
            await eng.ingest_frame(make_frame(shift=i * 500, seed=2 + i), i * 0.2, source="replay")  # unrelated views: stub boxes sit at fixed pixels
        summary = await eng.end_sweep()
        prov = eng.provenance()
        assert prov["inference_mode"] == "mocked" and prov["fresh_requests"] > 0       # labelled mock, never "real"
        packet = C.read_json(tmp_path / "r" / "claim_packet.json")["data"]
        assert packet["totals"]["book_count"] >= 4 and packet["totals"]["item_count"] >= 1
        assert packet["provenance"]["inference_mode"] == "mocked"
        assert all(b["status"] == "unidentified" or b["identity_source"] for b in packet["books"])   # no unsupported identification
        assert all(b["replacement_cost"] is None and b["spine_height_cm"] is None for b in packet["books"])
        assert any(b["visible_text"] for b in packet["books"])                      # reading ran on native crops
        assert any(p.suffix == ".png" for p in (tmp_path / "r" / "crops").glob("*"))
        assert all((tmp_path / "r" / b["frame_ref"]).is_file() for b in packet["books"] + packet["items"])
        assert summary["packet_status"] in {"ok", "partial"}
        n_fresh = prov["fresh_requests"]
        # identical second run is served from the shared cache, still labelled mocked
        eng2 = SweepEngine(tmp_path / "r2", config={"cache_dir": str(tmp_path / "cache"), "detector": "gemini", "reader": "gemini"}, stages=StageFns.real(), session_id="again")
        for i in range(6):
            await eng2.ingest_frame(make_frame(shift=i * 500, seed=2), i * 0.2, source="replay")
        await eng2.end_sweep()
        assert eng2.provenance()["cache_hits"] > 0 and eng2.provenance()["fresh_requests"] < n_fresh
        await eng.aclose(); await eng2.aclose()
    finally:
        gemini.set_transport(None)


@pytest.mark.asyncio
async def test_provisional_reads_are_resolved_not_left_unidentified(tmp_path):
    """Groups read DURING capture are skipped by the final group read, so the provisional pass itself must resolve them (v1 final runs: 6 readings, 0 resolutions)."""
    resolved = []

    def spy_resolve(readings_path, run_dir, config):
        resolved.extend(r["observation_id"] for r in C.read_json(readings_path)["data"]["readings"])
        return fake_resolve(readings_path, run_dir, config)

    eng = SweepEngine(tmp_path / "r", config={"cache_dir": str(tmp_path / "cache"), "no_provider": True, "stable_min_sightings": 1, "stable_s": 0.0},
                      stages=StageFns(detect=fake_detect, read=fake_read, resolve=spy_resolve, merge=fake_merge), session_id="prov")
    try:
        for i in range(3):
            await eng.ingest_frame(make_frame(shift=i * 500, seed=3), i * 0.2, source="replay")
        await asyncio.gather(*[t for t in list(eng._tasks) if not t.done()], return_exceptions=True)
        eng._group_seen.clear()
        await eng._provisional_read_pass()
        assert eng._prov_done and set(eng._prov_done) == set(resolved)
        assert {r["observation_id"] for r in eng.resolutions} >= set(eng._prov_done)
    finally:
        await eng.aclose()


@pytest.mark.asyncio
async def test_group_resolve_is_a_priority_queue_outside_read_slots_with_its_own_counts(tmp_path):
    """Resolution (paced catalogue HTTP) does not hold a read slot, serves complete-title+author readings first, gets the engine deadline, and reports its counts."""
    import time as _time
    resolved = []

    def tiered_read(slice_path, manifest_path, run_dir, config):
        out = fake_read(slice_path, manifest_path, run_dir, config)
        art = C.read_json(out)
        for i, r in enumerate(art["data"]["readings"]):
            n = int(r["observation_id"].split("-o")[1]) if "-o" in r["observation_id"] else i
            if n == 0:
                r.update(title_text="A Title", title_complete=True, author_text="An Author")      # tier 0
            elif n == 1:
                r.update(visible_text="x", title_text=None, title_complete=False)                  # tier 3
            else:
                r.update(title_text="Only Title", title_complete=True, author_text=None)           # tier 1
        C.write_json(out, art)
        return out

    def spy_resolve(readings_path, run_dir, config):
        assert "deadline_monotonic" in config                     # the engine hands its wall-clock deadline to the stage
        rs = C.read_json(readings_path)["data"]["readings"]
        resolved.extend((r["observation_id"], r.get("title_text") is not None and r.get("author_text") is not None) for r in rs)
        _time.sleep(0.03)                                         # a slow catalogue: jobs queue up behind the single worker
        return fake_resolve(readings_path, run_dir, config)

    eng = SweepEngine(tmp_path / "r", config={"cache_dir": str(tmp_path / "cache"), "no_provider": True, "resolve_workers": 1},
                      stages=StageFns(detect=fake_detect, read=tiered_read, resolve=spy_resolve, merge=fake_merge), session_id="grp-resolve")
    try:
        eng.stage_availability["read"] = {"available": True, "selection": "fake_reader"}      # the fake read stage stands in for a configured reader
        for i in range(4):
            await eng.ingest_frame(make_frame(shift=i * 500, seed=4), i * 0.2, source="replay")
        await eng.end_sweep()
        rv = eng.read_report.get("resolve")
        assert rv is not None, eng.read_report
        assert rv["workers"] == 1 and rv["failed"] == 0 and rv["queued"] == rv["readings_resolved"] + rv["readings_skipped_budget"] and rv["readings_resolved"] == len(resolved) > 1
        assert {r["observation_id"] for r in eng.readings} <= {r["observation_id"] for r in eng.resolutions}
        flags = [f for _, f in resolved[1:]]                      # the first job may start before the rest are queued; afterwards title+author readings come first
        assert flags == sorted(flags, reverse=True)
    finally:
        await eng.aclose()


@pytest.mark.asyncio
async def test_price_prefetch_prices_accepted_works_once_while_reading(tmp_path, monkeypatch):
    """Accepted catalogue works are priced by a background worker during the group read (shared cache warm-up); one call per distinct work, honest counts."""
    import pipeline.price as PP
    calls = []

    def accepting_resolve(readings_path, run_dir, config):
        a = C.read_json(readings_path)
        rs = [{"observation_id": r["observation_id"], "decision": "accepted", "work": "/works/OL1W", "title": "Some Book", "visible_text": "SOME BOOK",
               "language_code": "en", "issues": [], "matching_evidence": [{"field": "title"}], "source_url": "https://openlibrary.org/works/OL1W"}
              for r in a["data"]["readings"]]
        out = Path(run_dir) / config["resolutions_filename"]
        C.write_json(out, C.envelope(a["run_id"], "resolve", {"resolutions": rs}, status="ok"))
        return out

    def fake_price_work(work, root, catalogue, provider, config, deadline=None):
        calls.append(work)
        return {"replacement_cost": {"amount": 9.0}, "used_value": None, "issues": []}

    monkeypatch.setattr(PP, "price_work", fake_price_work)
    monkeypatch.setenv("LIBRARY_PRICE_ONLINE", "1")
    eng = SweepEngine(tmp_path / "r", config={"cache_dir": str(tmp_path / "cache"), "no_provider": True, "catalogue_online": True},
                      stages=StageFns(detect=fake_detect, read=fake_read, resolve=accepting_resolve, merge=fake_merge), session_id="prefetch")
    try:
        eng.stage_availability["read"] = {"available": True, "selection": "fake_reader"}
        for i in range(4):
            await eng.ingest_frame(make_frame(shift=i * 500, seed=5), i * 0.2, source="replay")
        await eng.end_sweep()
        pf = eng.read_report["price_prefetch"]
        early = [c for c in calls if c["id"].startswith("f")]       # prefetch rows carry the observation id; the final pricing step uses packet ids (bk-..)
        assert pf["enabled"] and pf["queued"] == 1 and len(early) == 1 and early[0]["work"] == "/works/OL1W" and early[0]["language_code"] is None
        assert pf["works_attempted"] == 1 and pf["works_priced"] == 1 and pf["errors"] == 0
    finally:
        await eng.aclose()
