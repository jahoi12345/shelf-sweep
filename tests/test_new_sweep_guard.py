"""New sweep must never lose a sweep: a sweep that is recording, writing its packet or building its maps is detached (finishes in the background), never cancelled.
Mocked stages and a fake build queue; proves the lifecycle plumbing only."""
import asyncio

import httpx
import pytest

from app import server as s
from tests.test_live_app import Harness, feed_panning
from tests.test_live_measurement import STAGES
from tests.test_post_sweep import builds, terminal, wait_for


class FakeQueue:
    """Stands in for app.post_sweep.PostSweepBuilds: records the automatic enqueue after the packet."""
    def __init__(self):
        self.enqueued, self.active = [], set()

    def enqueue(self, run_id):
        self.enqueued.append(run_id)
        self.active.add(run_id)

    def status(self, run_id):
        return {"state": "building_3d", "summary": "building 3D map"} if run_id in self.active else None

    def is_active(self, run_id):
        return run_id in self.active


@pytest.fixture
def queue(monkeypatch):
    q = FakeQueue()
    monkeypatch.setattr(s, "post_sweep", q)
    return q


def api():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://127.0.0.1:4177", cookies={"intake_owner": "owner"},
                             headers={"host": "127.0.0.1:4177", "origin": "http://127.0.0.1:4177"})


async def recorded(h):
    h.session.engine.config["report_pdf"] = False
    await feed_panning(h, 6)
    await h.wait(lambda: h.session.engine.inventory is not None)
    return h.session.engine


async def settled(cond, timeout=10.0):
    end = asyncio.get_event_loop().time() + timeout
    while not cond():
        assert asyncio.get_event_loop().time() < end, "condition not reached"
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_processing_states_name_what_is_running(tmp_path, queue):
    async with Harness(tmp_path, stages=STAGES) as h:
        assert s.sweep_processing(h.session)["processing"] is False                    # nothing recorded yet: nothing to lose
        eng = await recorded(h)
        info = s.sweep_processing(h.session)
        assert info["processing"] and info["phase"] == "capturing" and info["frames"] == 6 and "has not been ended" in info["detail"] and f"runs/{eng.run_dir.name}" in info["message"]
        await eng.end_sweep()
        queue.active.add(eng.run_dir.name)
        info = s.sweep_processing(h.session)
        assert info["processing"] and info["phase"] == "maps" and "building 3D map" in info["message"]
        queue.active.clear()
        assert s.sweep_processing(h.session)["processing"] is False                    # finished and the maps are done


@pytest.mark.asyncio
async def test_a_cancelled_caller_does_not_cancel_the_packet(tmp_path, queue):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = await recorded(h)
        gate = asyncio.Event()
        original = eng._room_auto_pass          # a step before the packet (the alternative association now runs after it)

        async def slow(*a, **k):
            await gate.wait()
            return await original(*a, **k)
        eng._room_auto_pass = slow
        caller = asyncio.create_task(eng.end_sweep())                                    # e.g. the Live socket's end_sweep task, cancelled when the socket closes
        await settled(lambda: eng.status == "finalizing")
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        assert eng.status == "finalizing" and not (eng.run_dir / "claim_packet.json").exists()
        with pytest.raises(Exception, match="already finalizing"):
            await eng.end_sweep()                                                         # a second End while the first is running never starts a second packet
        gate.set()
        await eng.wait_ended()
        assert eng.status == "finished" and (eng.run_dir / "claim_packet.json").is_file() and queue.enqueued == [eng.run_dir.name]


@pytest.mark.asyncio
async def test_new_sweep_while_finalizing_lets_the_packet_and_maps_finish_in_the_background(tmp_path, queue):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = await recorded(h)
        gate = asyncio.Event()
        original = eng._room_auto_pass          # a step before the packet (the alternative association now runs after it)

        async def slow(*a, **k):
            await gate.wait()
            return await original(*a, **k)
        eng._room_auto_pass = slow
        caller = asyncio.create_task(eng.end_sweep())
        await settled(lambda: eng.status == "finalizing")
        async with api() as client:
            got = (await client.get("/api/sessions/sess1/processing")).json()
            assert got["processing"] and got["phase"] == "finalizing" and "packet is still being written" in got["message"]
            body = (await client.delete("/api/sessions/sess1")).json()
        assert body["deleted"] and body["detached"] is True and body["continues"] == "finalizing" and body["purged"] is False
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        assert "sess1" not in s.sessions and "sess1" in s.detached and eng.status == "finalizing" and not eng._closed        # not closed, not cancelled
        async with api() as client:
            listed = (await client.get("/api/background-sweeps")).json()["sweeps"]
        assert [(x["run_id"], x["status"]) for x in listed] == [("sess1", "finalizing")]
        gate.set()
        await settled(lambda: "sess1" not in s.detached)
        assert eng.status == "finished" and (eng.run_dir / "claim_packet.json").is_file() and (eng.run_dir / "report.html").is_file()
        assert queue.enqueued == ["sess1"] and eng.run_dir.is_dir() and eng._closed                                          # builds queued after the packet; engine closed after


@pytest.mark.asyncio
async def test_new_sweep_while_recording_ends_it_in_the_background_instead_of_dropping_it(tmp_path, queue):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = await recorded(h)
        frames = sorted((eng.run_dir / "frames").glob("*.jpg"))
        assert eng.status == "capturing" and len(frames) == 6
        async with api() as client:
            body = (await client.delete("/api/sessions/sess1?purge=1")).json()           # even an explicit purge never destroys a sweep that is processing
        assert body["detached"] is True and body["continues"] == "capturing" and body["purged"] is False
        await settled(lambda: "sess1" not in s.detached, timeout=30)
        assert eng.status == "finished" and (eng.run_dir / "claim_packet.json").is_file() and len(list((eng.run_dir / "frames").glob("*.jpg"))) == 6
        assert queue.enqueued == ["sess1"]


@pytest.mark.asyncio
async def test_new_sweep_while_maps_build_only_closes_the_session_and_keeps_the_run(tmp_path, queue):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = await recorded(h)
        await eng.end_sweep()
        assert queue.active == {"sess1"}
        out = await s.release_session(h.session, purge=True)
        assert out == {"detached": False, "continues": "maps", "run_id": "sess1"} and "sess1" not in s.sessions and "sess1" not in s.detached
        assert eng.run_dir.is_dir() and queue.active == {"sess1"}                         # purge ignored; the build queue is untouched


@pytest.mark.asyncio
async def test_an_idle_sweep_with_nothing_recorded_is_discarded_and_purged_as_before(tmp_path, queue):
    async with Harness(tmp_path, stages=STAGES) as h:
        run_dir = h.session.run_dir
        out = await s.release_session(h.session, purge=True)
        assert out["detached"] is False and out["continues"] is None and not run_dir.exists() and "sess1" not in s.sessions


@pytest.mark.asyncio
async def test_idle_timeout_detaches_a_sweep_that_is_still_recording(tmp_path, queue, monkeypatch):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = await recorded(h)
        h.session.updated_at -= s.SESSION_TTL + 5
        await s.cleanup_sessions()
        assert "sess1" not in s.sessions
        await settled(lambda: "sess1" not in s.detached, timeout=30)
        assert eng.status == "finished" and queue.enqueued == ["sess1"]


def test_post_sweep_reports_whether_a_run_still_has_maps_in_flight(tmp_path):
    root, jobs, psb = builds(tmp_path)
    assert psb.is_active("sweep_abc") is False
    psb.enqueue("sweep_abc")
    assert psb.is_active("sweep_abc") is True
    wait_for(terminal(psb, "sweep_abc"))
    assert psb.is_active("sweep_abc") is False
    psb.stop()
