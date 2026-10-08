"""Laptop watch page: read-only, loopback server only (404 on the LAN server :4183), fed by the engine's <run>/watch.json."""
import json
import shutil

import pytest
from fastapi.testclient import TestClient

from app import server as s
from app import watch as W
from tests.test_lan_access import IP, KEY, lan, phone  # noqa: F401  (fixture)
from tests.test_live_app import Harness, feed_panning
from tests.test_live_measurement import STAGES

RID = "sweep_0123456789abcdef"
LOCAL = "http://127.0.0.1:4182"
PATHS = ["/watch", "/watch.js", "/watch.css", "/api/watch/sweeps", f"/api/watch/sweeps/{RID}", f"/api/watch/sweeps/{RID}/frames/f00000.jpg",
         f"/api/watch/sweeps/{RID}/report",
         # claim packet viewer, raw/download, voice transcript, spine crops
         f"/api/watch/sweeps/{RID}/packet", f"/api/watch/sweeps/{RID}/packet.json", f"/api/watch/sweeps/{RID}/packet.json?download=1",
         f"/api/watch/sweeps/{RID}/transcript", f"/api/watch/sweeps/{RID}/crops/0123456789abcdef0123.png"]


def _jpeg():
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".jpg", np.full((90, 160, 3), 120, np.uint8))
    return buf.tobytes()


@pytest.fixture()
def runs(tmp_path, monkeypatch):
    for k in ("LIBRARY_LAN_MODE", "LIBRARY_LAN_HOSTS", "LIBRARY_LAN_TOKEN", "LIBRARY_LAN_CA"):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "runs"
    d = root / RID
    (d / "frames").mkdir(parents=True)
    (d / "frames" / "f00000.jpg").write_bytes(_jpeg())
    (d / "frames" / "f00001.jpg").write_bytes(_jpeg())
    monkeypatch.setattr(s, "RUNS_DIR", root)
    monkeypatch.setattr(s, "_watch_post_status", lambda run_id: None)
    return root


def _watch(d, **kw):
    import os
    import time
    rec = {"version": 1, "run_id": d.name, "status": "capturing", "closed": False, "pid": os.getpid(), "started_epoch": time.time() - 30,
           "ended_epoch": None, "updated_epoch": time.time(), "frames": 2, "latest_frame": "f00001", "indicator": "too_far",
           "live_count": {"book_count": 12}, "notices": [{"code": "too_far", "message": "Too far. Step closer", "kind": None, "at_s": 12.5}],
           "finalize": {"stage": None, "error": None, "read": None, "detect": {"done": 3, "total": 4}}, "finalize_steps": [], "first_count": None}
    rec.update(kw)
    (d / "watch.json").write_text(json.dumps(rec))


def test_loopback_server_serves_the_page_and_the_api(runs):
    _watch(runs / RID)
    c = TestClient(s.app, base_url=LOCAL)
    for path in PATHS[:3]:
        assert c.get(path).status_code == 200, path
    lst = c.get("/api/watch/sweeps").json()
    assert lst["newest_active"] == RID and lst["sweeps"][0]["phase"] == "capturing"
    st = c.get(f"/api/watch/sweeps/{RID}").json()
    eng = st["engine"]
    assert st["phase"] == "capturing" and st["active"] and eng["frames"] == 2 and eng["indicator"] == "too_far" and eng["notices"][0]["code"] == "too_far"
    assert "pid" not in eng                                       # nothing about the serving process leaks into the page
    r = c.get(f"/api/watch/sweeps/{RID}/frames/f00001.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.headers["cache-control"] == "no-store"
    assert c.get(f"/api/watch/sweeps/{RID}/report").status_code == 404     # not written yet


def test_lan_server_answers_404_for_every_watch_path(runs, lan):  # noqa: F811
    _watch(runs / RID)
    _saved(runs / RID)                       # the packet, transcript and crop exist: the 404 is the LAN refusal, not a missing file
    paired = phone()
    assert paired.get(f"/?k={KEY}", follow_redirects=False).status_code == 303   # a phone holding the pairing key ...
    assert paired.get("/", follow_redirects=False).status_code == 200
    for path in PATHS:
        assert paired.get(path, follow_redirects=False).status_code == 404, path      # ... still gets 404
        assert phone().get(path, follow_redirects=False).status_code == 404, path     # without the key too (not 403: the page does not exist there)
        assert phone().get(path + f"?k={KEY}", follow_redirects=False).status_code == 404, path
        # the :4183 process itself, asked from the Mac (loopback client, loopback host): still 404 in LAN mode
        assert TestClient(s.app, base_url="https://127.0.0.1:4183").get(path, follow_redirects=False).status_code == 404, path
    assert TestClient(s.app, base_url=f"https://{IP}:4183").get("/watch/").status_code == 404


def test_only_loopback_host_and_client(runs):
    _watch(runs / RID)
    c = TestClient(s.app, base_url=LOCAL)
    assert c.get("/watch", headers={"host": "evil.example"}).status_code == 404     # DNS rebinding name
    assert TestClient(s.app, base_url=LOCAL, client=("192.168.1.9", 5000)).get("/api/watch/sweeps").status_code == 404
    assert c.get("/watch").status_code == 200


def test_read_only_routes_and_strict_names(runs):
    watch_routes = [r for r in s.app.routes if getattr(r, "path", "").startswith(("/watch", "/api/watch"))]
    assert watch_routes and all(r.methods <= {"GET", "HEAD"} for r in watch_routes)
    c = TestClient(s.app, base_url=LOCAL)
    _watch(runs / RID)
    for bad in [f"/api/watch/sweeps/{RID}/frames/..%2Fwatch.json", f"/api/watch/sweeps/{RID}/frames/f1.jpg", "/api/watch/sweeps/sess1",
                "/api/watch/sweeps/sweep_ZZ", f"/api/watch/sweeps/{RID}/frames/f00009.jpg"]:
        assert c.get(bad).status_code == 404, bad
    assert c.post("/api/watch/sweeps", headers={"origin": LOCAL}).status_code == 405
    (runs / RID / "frames" / "f00002.jpg").symlink_to(runs / RID / "frames" / "f00000.jpg")
    assert c.get(f"/api/watch/sweeps/{RID}/frames/f00002.jpg").status_code == 404


def test_dead_server_or_closed_sweep_is_not_active(runs):
    _watch(runs / RID, pid=999_999_9)
    st = W.sweep_state(runs, RID)
    assert st["phase"] == "stopped" and not st["active"]
    _watch(runs / RID, closed=True)
    assert W.sweep_state(runs, RID)["phase"] == "stopped"


def test_saved_sweep_without_watch_file_and_post_sweep_states(runs):
    d = runs / RID
    env = {"status": "ok", "data": {"totals": {"book_count": 755, "book_count_range": [745, 913], "count_reliability": "medium", "item_count": 99},
                                     "sweep": {"finalization_s": 210.4}}}
    (d / "claim_packet.json").write_text(json.dumps(env))
    (d / "report.html").write_text("<html>report</html>")
    st = W.sweep_state(runs, RID, post_status=lambda r: {"state": "building_3d", "steps": {}})
    assert st["phase"] == "maps" and st["active"] and not st["final"] and st["engine"]["frames"] == 2 and st["engine"]["latest_frame"] == "f00001"
    st = W.sweep_state(runs, RID, post_status=lambda r: {"state": "done", "steps": {}})
    assert st["phase"] == "done" and st["final"] and st["packet"]["book_count"] == 755 and st["packet"]["count_reliability"] == "medium"
    c = TestClient(s.app, base_url=LOCAL)
    assert c.get(f"/api/watch/sweeps/{RID}/report").text == "<html>report</html>"
    assert W.list_sweeps(runs)["newest_active"] is None


@pytest.mark.asyncio
async def test_engine_writes_the_watch_file_through_the_sweep(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        eng.config["report_pdf"] = False
        await feed_panning(h, 6)
        await h.wait(lambda: eng.inventory is not None)
        f = eng.run_dir / "watch.json"
        live = json.loads(f.read_text())
        assert live["status"] == "capturing" and live["frames"] >= 1 and live["latest_frame"].startswith("f") and "indicator" in live
        assert eng._watch_task is not None and not eng._watch_task.done()       # ~1/s heartbeat while recording/ending
        before = live["updated_epoch"]
        await h.wait(lambda: json.loads(f.read_text())["updated_epoch"] > before)
        await eng.end_sweep()
        done = json.loads(f.read_text())
        assert done["status"] == "finished" and done["ended_epoch"] and done["first_count"]["book_count"] is not None
        stages = [x["stage"] for x in done["finalize_steps"]]
        assert stages[0] == "finishing detection and OCR" and "rendering the report" in stages
        text = f.read_text()
        assert "owner" not in text and "session_id" not in text and "lan_key" not in text and str(tmp_path) not in text
        # the page reads exactly this file from another process
        other = tmp_path / "other" / RID
        other.mkdir(parents=True)
        shutil.copy(f, other / "watch.json")
        st = W.sweep_state(tmp_path / "other", RID)
        assert st["phase"] == "done" and st["engine"]["first_count"] == done["first_count"]


def test_engine_without_the_flag_writes_nothing(tmp_path):
    from app.engine import SweepEngine
    eng = SweepEngine(tmp_path / "r", stages=STAGES)
    eng._bump()
    assert not (tmp_path / "r" / "watch.json").exists()


def test_an_open_page_without_frames_is_waiting_and_never_followed_over_a_recording(runs):
    """Every sweep-page load creates a session (2026-10-08: a Mac tab reloading after a restart showed as "Recording" with 0 frames)."""
    import time
    _watch(runs / RID)
    empty = runs / "sweep_00000000000000e0"
    (empty / "frames").mkdir(parents=True)
    _watch(empty, frames=0, latest_frame=None, indicator=None, live_count=None, notices=[], started_epoch=time.time())
    c = TestClient(s.app, base_url=LOCAL)
    assert c.get(f"/api/watch/sweeps/{empty.name}").json()["phase"] == "waiting"
    assert c.get("/api/watch/sweeps").json()["newest_active"] == RID, "the sweep with frames is followed, not the newer empty page"


# ------------------------------------------------------------------ count banner fields, claim packet viewer, voice transcript
CROP = "0123456789abcdef0123.png"
ABS = "/Users/someone/shelf-sweep/runs"


def _png():
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".png", np.full((40, 12, 3), 90, np.uint8))
    return buf.tobytes()


def _packet(book_count=484, title=None):
    book = {"id": "bk-f00006-o000", "display_index": 1, "status": "identified" if title else "unidentified", "title": title, "author": "Chris Pavone" if title else None,
            "id_confidence": 0.95 if title else None, "count_tier": "confident", "frame_id": "f00027", "timestamp_s": 32.4, "visible_text": "TWO NIGHTS",
            "replacement_cost": {"amount": 31.78, "currency": "USD", "converted": False, "source": "AbeBooks seller"} if title else None,
            "used_value": {"amount": 4.41, "currency": "USD", "converted": True, "original_amount": 3.95, "original_currency": "EUR"} if title else None,
            "reading_evidence": [{"crop_path": f"crops/{CROP}", "provenance": {"request_ref": f"{ABS}/{RID}/ocr_responses/a.json"}}],
            "dedup_reasons": ["x" * 50], "evidence": {"bbox_xyxy": [1, 2, 3, 4]}}
    totals = {"book_count": book_count, "book_count_range": [461, 507], "book_count_lower": 461, "book_count_upper": 507, "count_reliability": "low",
              "headline_estimator": "panorama", "book_lines": 589, "book_lines_range": [418, 810], "book_count_range_label": "panorama range (may not contain the true count)",
              "books_identified": 1 if title else 0, "item_count": 1, "books_replacement_cost": 31.78 if title else None}
    return {"schema_version": 1, "run_id": RID, "stage": "packet", "status": "partial", "issues": [{"code": "stages_incomplete", "severity": "warning", "detail": "read"}],
            "metrics": {"book_lines": 589}, "data": {
                "completion_status": "provisional", "totals": totals, "books": [book],
                "items": [{"id": "it-1", "display_index": 1, "category": "plant", "description": "pot", "confidence": 0.8, "frame_id": "f00032", "timestamp_s": 33.5}],
                "stage_status": {"detect": {"status": "ok", "elapsed_s": 156.0, "detail": "535 call(s)"}, "read": {"status": "partial", "elapsed_s": 72.5}},
                "pricing": {"market": "US", "currency": "USD", "prices": [{"ref_id": "bk-f00006-o000"}]},
                "room": {"status": "abstained", "abstention_reasons": ["room_geometry_not_established"]},
                "measurement_evidence": {"auto": {"status": "abstained", "scale": {"cm_per_unit": None, "reason": "no_reconstruction"}}},
                "sweep": {"captured_at": "2026-10-08T03:11:16+0200", "currency": "USD"},
                "provenance": {"runtime": "/Users/someone/venv/bin/python", "note": f"see {ABS}/{RID}/ocr_responses/b.json and https://openlibrary.org/works/OL1W"}}}


def _saved(d, **kw):
    (d / "claim_packet.json").write_text(json.dumps(_packet(**kw)))
    (d / "crops").mkdir(exist_ok=True)
    (d / "crops" / CROP).write_bytes(_png())
    (d / "live").mkdir(exist_ok=True)
    (d / "live" / "transcript.jsonl").write_text(json.dumps({"id": "t1", "speaker": "Claimant", "text": "Hello", "at_epoch": 1.0, "at_s": 0.5}) + "\n")


def test_packet_viewer_raw_download_and_crops(runs):
    import os
    d = runs / RID
    _watch(d, status="finished", ended_epoch=1.0)
    _saved(d, title="Two Nights in Lisbon")
    c = TestClient(s.app, base_url=LOCAL)
    st = c.get(f"/api/watch/sweeps/{RID}").json()
    p = st["packet"]          # the count banner: headline + estimator, book lines, range + label, reliability
    assert (p["book_count"], p["headline_estimator"], p["book_lines"], p["book_count_range"], p["count_reliability"]) == (484, "panorama", 589, [461, 507], "low")
    assert p["book_count_range_label"].startswith("panorama range") and st["packet_rev"] and st["transcript_rev"]
    r = c.get(f"/api/watch/sweeps/{RID}/packet")
    v = r.json()
    assert r.status_code == 200 and v["status"] == "partial" and v["issues"][0]["code"] == "stages_incomplete" and v["stage_status"]["detect"]["status"] == "ok"
    b = v["books"][0]
    assert (b["title"], b["author"], b["confidence"], b["crop"], b["replacement_cost"]["amount"], b["used_value"]["converted"]) == (
        "Two Nights in Lisbon", "Chris Pavone", 0.95, CROP, 31.78, True)
    assert "dedup_reasons" not in b and "evidence" not in b and v["items"][0]["category"] == "plant" and "prices" not in v["pricing"]
    assert v["room"]["status"] == "abstained" and v["measurement"]["status"] == "abstained" and v["totals"]["headline_estimator"] == "panorama"
    raw = c.get(f"/api/watch/sweeps/{RID}/packet.json")
    assert raw.status_code == 200 and raw.headers["content-type"].startswith("application/json") and "attachment" not in raw.headers.get("content-disposition", "")
    for text in (r.text, raw.text):
        assert "/Users/" not in text and "someone" not in text                      # no absolute path leaks
    env = raw.json()
    assert env["data"]["provenance"]["runtime"] == "…/python" and f"{RID}/ocr_responses/b.json" in env["data"]["provenance"]["note"]
    assert "https://openlibrary.org/works/OL1W" in env["data"]["provenance"]["note"]   # URLs untouched
    assert env["data"]["books"][0]["reading_evidence"][0]["provenance"]["request_ref"] == f"{RID}/ocr_responses/a.json"
    dl = c.get(f"/api/watch/sweeps/{RID}/packet.json?download=1")
    assert dl.headers["content-disposition"] == f'attachment; filename="{RID}_claim_packet.json"' and dl.content == raw.content
    img = c.get(f"/api/watch/sweeps/{RID}/crops/{CROP}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    for bad in ["crops/..%2Fclaim_packet.json", "crops/abc.png", "crops/0123456789abcdef0123.jpg", "crops/ffffffffffffffffffff.png"]:
        assert c.get(f"/api/watch/sweeps/{RID}/{bad}").status_code == 404, bad
    (d / "crops" / "ffffffffffffffffffff.png").symlink_to(d / "crops" / CROP)
    assert c.get(f"/api/watch/sweeps/{RID}/crops/ffffffffffffffffffff.png").status_code == 404
    # the post-sweep steps rewrite the packet (finish reading fills titles, the count changes): the viewer and the banner follow
    (d / "claim_packet.json").write_text(json.dumps(_packet(book_count=470, title="Rayuela")))
    st0 = st["packet_rev"]
    os.utime(d / "claim_packet.json", (st0 + 5, st0 + 5))
    st2 = c.get(f"/api/watch/sweeps/{RID}").json()
    assert st2["packet_rev"] != st0 and st2["packet"]["book_count"] == 470
    assert c.get(f"/api/watch/sweeps/{RID}/packet").json()["books"][0]["title"] == "Rayuela"
    assert c.get(f"/api/watch/sweeps/{RID}/packet.json").json()["data"]["totals"]["book_count"] == 470
    # a sweep without a packet: 404 with a reason, and the state says so
    other = runs / "sweep_00000000000000aa"
    (other / "frames").mkdir(parents=True)
    assert c.get(f"/api/watch/sweeps/{other.name}/packet").status_code == 404 and c.get(f"/api/watch/sweeps/{other.name}/packet.json").status_code == 404
    assert c.get(f"/api/watch/sweeps/{other.name}").json()["packet_rev"] is None
    assert c.get(f"/api/watch/sweeps/{other.name}/transcript").json() == {"persisted": False, "turns": []}


def test_transcript_route_reads_the_saved_turns_without_ids(runs):
    d = runs / RID
    _watch(d)
    (d / "live").mkdir()
    lines = [{"id": "secret-turn-id", "speaker": "Agent", "text": "Hi! Make one slow pass.", "at_epoch": 100.0, "at_s": 0.1},
             {"id": "t2", "speaker": "Claimant", "text": f"my files are in {ABS}/x", "at_epoch": 112.0, "at_s": 12.0}]
    (d / "live" / "transcript.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines) + '{"id": "t3", "speak')   # a line being written
    c = TestClient(s.app, base_url=LOCAL)
    data = c.get(f"/api/watch/sweeps/{RID}/transcript").json()
    assert data["persisted"] and [t["speaker"] for t in data["turns"]] == ["Agent", "Claimant"] and data["turns"][1]["at_s"] == 12.0
    assert "secret-turn-id" not in json.dumps(data) and "/Users/" not in json.dumps(data)
    (d / "live" / "transcript.jsonl").unlink()
    (d / "live" / "transcript.jsonl").symlink_to(d / "watch.json")
    assert c.get(f"/api/watch/sweeps/{RID}/transcript").json() == {"persisted": False, "turns": []}


@pytest.mark.asyncio
async def test_live_voice_appends_every_finished_turn_to_the_run(tmp_path, monkeypatch):
    """A fake Live session (no hosted call): what the person said and what the voice agent said land in <run>/live/transcript.jsonl as they finish."""
    import asyncio
    from tests.test_live_app import event, words
    async with Harness(tmp_path) as h:
        await h.live.q.put(event(input_transcription=words("How many books so far?", True)))
        await h.live.q.put(event(output_transcription=words("About forty so far.", True), turn_complete=True))
        await h.client({"type": "text", "text": "Thanks", "id": "typed-1"})
        await h.wait(lambda: len(h.session.transcript) == 3)
        f = h.session.run_dir / "live" / "transcript.jsonl"
        rows = [json.loads(x) for x in f.read_text().splitlines()]
        assert sorted((r["speaker"], r["text"]) for r in rows) == [("Agent", "About forty so far."), ("Claimant", "How many books so far?"), ("Claimant", "Thanks")]
        assert all(isinstance(r["at_s"], float) and r["at_s"] >= 0 and r["at_epoch"] > 0 for r in rows)
        assert all(set(r) == {"id", "speaker", "text", "at_epoch", "at_s"} for r in rows)          # text only: no audio, no session/owner
        await h.client({"type": "text", "text": "Thanks", "id": "typed-1"})                          # a re-sent turn is not appended twice
        await asyncio.sleep(0.1)
        assert len(f.read_text().splitlines()) == 3
        assert "owner" not in f.read_text()
    # the watch page (another process, same runs dir) serves it, kept after the sweep
    monkeypatch.setattr(s, "RUNS_DIR", tmp_path / "w")
    monkeypatch.setattr(s, "_watch_post_status", lambda run_id: None)
    for k in ("LIBRARY_LAN_MODE", "LIBRARY_LAN_HOSTS", "LIBRARY_LAN_TOKEN", "LIBRARY_LAN_CA"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "w" / RID / "live").mkdir(parents=True)
    shutil.copy(f, tmp_path / "w" / RID / "live" / "transcript.jsonl")
    turns = TestClient(s.app, base_url=LOCAL).get(f"/api/watch/sweeps/{RID}/transcript").json()["turns"]
    assert sorted(t["text"] for t in turns) == ["About forty so far.", "How many books so far?", "Thanks"]


def test_append_turn_saves_the_greeting_and_never_breaks_the_conversation(tmp_path):
    """The greeting is added at session creation (no socket yet); an unwritable run dir never breaks the conversation."""
    from types import SimpleNamespace as NS
    sess = s.SweepSession(session_id="x", owner="o", run_dir=tmp_path / "r", engine=NS(started_epoch=None))
    s.append_turn(sess, "Agent", "Hi!")
    rec = json.loads((tmp_path / "r" / "live" / "transcript.jsonl").read_text())
    assert rec["speaker"] == "Agent" and rec["text"] == "Hi!" and rec["at_s"] is None
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a directory")
    sess2 = s.SweepSession(session_id="y", owner="o", run_dir=blocked, engine=NS(started_epoch=0.0))
    s.append_turn(sess2, "Claimant", "still works")
    assert sess2.transcript[-1]["text"] == "still works"
