"""Finish reading: the post-sweep step that reads the groups the packet's read budget left unread. Fake reader/resolver (no network, no Gemini)."""
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from app import finish_reading as FR
from app import post_sweep as PS
from pipeline import contracts as C
from tests.test_build_jobs import make_root, make_run, SHA1
from tests.test_live_app import Harness, feed_panning
from tests.test_live_measurement import STAGES


class FakeStages:
    """read: a completed reading per observation (text = FIRST/SECOND + id); resolve: accepted for every reading whose text is long enough."""

    def __init__(self):
        self.read_calls, self.resolve_calls = [], []

    def read(self, slice_path, manifest_path, run_dir, config):
        a = C.read_json(slice_path)
        ids = [o["observation_id"] for o in a["data"]["observations"]]
        self.read_calls.append(ids)
        rows = [{"observation_id": i, "frame_id": i.split("-")[0], "visible_text": f"THE BOOK {i}", "title_text": f"THE BOOK {i}", "title_complete": True, "author_text": "AUTHOR",
                 "read_confidence": 0.9, "ocr_attempt": "completed", "issues": []} for i in ids]
        out = Path(run_dir) / config["readings_filename"]
        C.write_json(out, C.envelope(a["run_id"], "read", {"readings": rows}, status="ok"))
        return out

    def resolve(self, readings_path, run_dir, config):
        a = C.read_json(readings_path)
        rows = [{"observation_id": r["observation_id"], "decision": "accepted", "work": "/works/OL1W", "title": "The Book", "author": "An Author", "edition": None, "isbn": None,
                 "publisher": None, "source_url": "https://openlibrary.org/works/OL1W", "matching_evidence": ["title"], "issues": [], "catalogue": "open_library"}
                for r in a["data"]["readings"]]
        self.resolve_calls.append([r["observation_id"] for r in rows])
        out = Path(run_dir) / config["resolutions_filename"]
        C.write_json(out, C.envelope(a["run_id"], "resolve", {"resolutions": rows}, status="ok"))
        return out


def reader(available=True, selection="gemini"):
    return ({"read": {"image_request": None}, "resolve": {}}, {"read": {"selection": selection, "available": available, "reason": "gate off"}})


async def finished_run(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        eng.config["report_pdf"] = False
        await feed_panning(h, 6, unrelated=True)
        await h.wait(lambda: eng.inventory is not None)
        await eng.end_sweep()
        return eng.run_dir


def skip_reading(run, n=2):
    """Mimic the budget: drop the readings (and resolutions) of the first n book groups' observations from the run files."""
    inv = json.loads((run / "inventory.json").read_text())
    drop = {o for g in inv["data"]["books"][:n] for o in g["observation_ids"]}
    for name, key in (("readings.json", "readings"), ("resolutions.json", "resolutions")):
        if (run / name).is_file():
            env = json.loads((run / name).read_text())
            env["data"][key] = [{**r, "ocr_attempt": "completed"} if key == "readings" else r for r in env["data"][key] if r["observation_id"] not in drop]
            (run / name).write_text(json.dumps(env))
    return drop


@pytest.mark.asyncio
async def test_unread_groups_are_read_resolved_attached_and_stage_status_ok(tmp_path, monkeypatch):
    run = await finished_run(tmp_path)
    if not (run / "readings.json").is_file():
        pytest.skip("fake engine wrote no readings.json")
    dropped = skip_reading(run, 2)
    before_pk = json.loads((run / "claim_packet.json").read_text())["data"]
    pl = FR.plan_unread(run)
    assert pl["todo1"], "dropping the readings must leave attemptable groups unread"
    unread_groups = len(pl["todo1"])
    seen = []
    ctl = FR.Ctl(progress=lambda phase, d, t, note="": seen.append((phase, d, t)))
    fake = FakeStages()
    out = FR.run(run, ctl, stages=fake, reader=reader(), price=False, render=False)
    assert out["state"] == "done" and out["groups_read"] == unread_groups, out
    read_ids = {i for call in fake.read_calls for i in call}
    assert read_ids and read_ids <= dropped | {o for o in pl["obs"]}     # only unread observations were read; none of the already-read ones
    pre = {r["observation_id"] for r in json.loads((run / "readings.pre_finish_reading.json").read_text())["data"]["readings"]}
    assert not (read_ids & pre)
    assert ("reading", unread_groups, unread_groups) in seen and any(p == "resolving" for p, *_ in seen)
    after = json.loads((run / "claim_packet.json").read_text())["data"]
    assert after["totals"]["books_identified"] > before_pk["totals"]["books_identified"]
    assert after["stage_status"]["read"]["status"] == "ok" and after["stage_status"]["resolve"]["status"] == "ok"
    assert after["sweep"]["finish_reading"]["groups_still_unread"] == 0
    assert any((b.get("visible_text") or "").startswith("THE BOOK") for b in after["books"])
    # a second run finds nothing left
    again = FR.run(run, FR.Ctl(), stages=FakeStages(), reader=reader(), price=False, render=False)
    assert again["state"] == "done" and "nothing left" in again["reason"]


@pytest.mark.asyncio
async def test_hosted_reader_off_skips_and_changes_nothing(tmp_path):
    run = await finished_run(tmp_path)
    if not (run / "readings.json").is_file():
        pytest.skip("fake engine wrote no readings.json")
    skip_reading(run, 2)
    before = (run / "claim_packet.json").read_bytes()
    out = FR.run(run, FR.Ctl(), stages=FakeStages(), reader=reader(available=False), price=False, render=False)
    assert out["state"] == "skipped" and out["reason"] == "hosted reader off"
    assert (run / "claim_packet.json").read_bytes() == before


@pytest.mark.asyncio
async def test_interrupted_step_resumes_from_finished_chunks(tmp_path):
    run = await finished_run(tmp_path)
    if not (run / "readings.json").is_file():
        pytest.skip("fake engine wrote no readings.json")
    skip_reading(run, 2)
    n_unread = len(FR.plan_unread(run)["todo1"])
    fake1 = FakeStages()
    stop_after = {"n": 0}

    def stop():
        return len(fake1.read_calls) >= 1
    out = FR.run(run, FR.Ctl(stop=stop), stages=fake1, reader=reader(), price=False, render=False)
    assert out["state"] == "interrupted"
    left = len(FR.plan_unread(run)["todo1"])            # the finished chunk files count as read
    assert left < n_unread
    fake2 = FakeStages()
    out2 = FR.run(run, FR.Ctl(), stages=fake2, reader=reader(), price=False, render=False)
    assert out2["state"] == "done"
    assert len({i for c in fake2.read_calls for i in c}) <= left * 2 + 3


def test_no_unread_and_no_unresolved_is_a_noop_helper():
    rows = [{"observation_id": "a", "ocr_attempt": "completed"}, {"observation_id": "b", "ocr_attempt": "completed"}, {"observation_id": "c", "ocr_attempt": "failed"}]
    res = [{"observation_id": "a", "issues": []}, {"observation_id": "b", "issues": ["resolve_budget_exhausted"]}]
    assert [r["observation_id"] for r in FR.unresolved_readings(rows, res)] == ["b"]


# ---------------------------------------------------------------------------------------------------------------- the queue
def wait_for(fn, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.03)
    raise AssertionError("condition not reached")


def queue(tmp_path, runner, **kw):
    root = make_root(tmp_path, ())
    make_run(root, "sweep_abc", SHA1, frames=9)
    psb = PS.PostSweepBuilds(root, NS(eligible=lambda *a: None, jobs=lambda: [], start=lambda *a, **k: (_ for _ in ()).throw(ValueError("no builds in this test"))), poll_s=0.03, finish_runner=runner, **kw)
    return root, psb


def test_queue_runs_the_step_records_progress_and_the_summary(tmp_path):
    seen = []

    def runner(run_id, ctl):
        ctl.progress("reading", 3, 10, "x")
        seen.append(ctl.settle())
        ctl.progress("resolving", 10, 10)
        return {"state": "done", "summary": "10 of 10 unread groups read", "groups_unread_at_start": 10, "before": {"books_identified": 5}, "after": {"books_identified": 150}}
    root, psb = queue(tmp_path, runner)
    st = psb.request_finish_reading("sweep_abc")
    assert st["state"] == "reading" and st["finish_reading"]["state"] in ("queued", "running")
    done = wait_for(lambda: (psb.status("sweep_abc") or {}).get("finish_reading", {}).get("state") == "done" and psb.status("sweep_abc"))
    psb.stop()
    f = done["finish_reading"]
    assert f["summary"].startswith("10 of 10") and f["after"]["books_identified"] == 150 and done["state"] != "reading"
    assert seen == [True]
    assert "reading: 10 of 10" in done["summary"]
    assert json.loads((root / "runs/_server_state/post_sweep/sweep_abc.json").read_text())["finish_reading"]["state"] == "done"


def test_skipped_and_failed_states_are_reported(tmp_path):
    root, psb = queue(tmp_path, lambda run_id, ctl: {"state": "skipped", "reason": "hosted reader off"})
    psb.request_finish_reading("sweep_abc")
    st = wait_for(lambda: (psb.status("sweep_abc") or {}).get("finish_reading", {}).get("state") == "skipped" and psb.status("sweep_abc"))
    assert "skipped: hosted reader off" in st["summary"]
    psb.stop()
    (tmp_path / "b").mkdir()
    root2, psb2 = queue(tmp_path / "b", lambda run_id, ctl: (_ for _ in ()).throw(RuntimeError("boom")))
    psb2.request_finish_reading("sweep_abc")
    st = wait_for(lambda: (psb2.status("sweep_abc") or {}).get("finish_reading", {}).get("state") == "failed" and psb2.status("sweep_abc"))
    assert "boom" in st["finish_reading"]["error"]
    psb2.stop()


def test_restart_requeues_a_running_step(tmp_path):
    root, psb = queue(tmp_path, lambda run_id, ctl: {"state": "done", "summary": "ok"})
    d = root / "runs/_server_state/post_sweep"
    d.mkdir(parents=True, exist_ok=True)
    rec = psb._blank("sweep_abc")
    rec["finish_reading"] = {"state": "running", "phase": "reading", "done": 4, "total": 9}
    (d / "sweep_abc.json").write_text(json.dumps(rec))
    assert psb.resume() >= 1
    wait_for(lambda: (psb.status("sweep_abc") or {}).get("finish_reading", {}).get("state") == "done")
    psb.stop()


def test_packet_lock_is_shared_and_the_chain_must_settle_first(tmp_path):
    root, psb = queue(tmp_path, lambda run_id, ctl: {"state": "done"})
    psb._records["sweep_abc"] = rec = psb._blank("sweep_abc")
    rec["steps"]["panorama"] = {"state": "running", "attempts": 1}
    assert psb._chain_settled("sweep_abc") is False
    rec["steps"]["panorama"] = {"state": "failed", "attempts": 3}
    assert psb._chain_settled("sweep_abc") is True
    psb.on_instances = lambda *a: {}
    out = root / "runs" / "sweep_abc_panorama_v1"
    out.mkdir(parents=True)
    (out / "panorama.json").write_text("{}")
    rec["steps"]["panorama"] = {"state": "done", "attempts": 1, "out_run": "sweep_abc_panorama_v1"}
    assert psb._chain_settled("sweep_abc") is False            # duplicate merge not yet recorded
    rec["instances"] = {"state": "done"}
    assert psb._chain_settled("sweep_abc") is True
    assert psb._packet_lock.acquire(blocking=False)
    psb._packet_lock.release()
    psb.stop()
