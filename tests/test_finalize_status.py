"""End-sweep exposes its current step and a recorded failure (status indicator input); behaviour is otherwise unchanged."""
import pytest

from app import engine as E
from tests.test_live_app import Harness, feed_panning
from tests.test_live_measurement import STAGES


@pytest.mark.asyncio
async def test_finalize_steps_are_emitted_in_order_and_cleared_when_done(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        eng.config["report_pdf"] = False
        events = []
        original = eng._emit
        async def spy(event):
            if event.get("type") == "finalize":
                events.append(event["stage"])
            await original(event)
        eng._emit = spy
        await feed_panning(h, 6)
        await h.wait(lambda: eng.inventory is not None)
        await eng.end_sweep()
        order = [s for s in events if s]
        assert order[0] == "finishing detection and OCR"
        for earlier, later in [("merging duplicates", "pricing"), ("pricing", "building the packet"), ("building the packet", "rendering the report")]:
            assert order.index(earlier) < order.index(later)
        assert eng.status == "finished" and eng.snapshot()["finalize"]["stage"] is None


@pytest.mark.asyncio
async def test_failed_finalization_is_recorded_visible_and_retryable(tmp_path, monkeypatch):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        eng.config["report_pdf"] = False
        await feed_panning(h, 6)
        await h.wait(lambda: eng.inventory is not None)
        real_build = E.P.build
        monkeypatch.setattr(E.P, "build", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            await eng.end_sweep()
        snap = eng.snapshot()
        assert snap["status"] == "failed" and "disk full" in snap["finalize"]["error"]
        monkeypatch.setattr(E.P, "build", real_build)
        await eng.end_sweep()                      # End can be pressed again
        assert eng.status == "finished" and eng.snapshot()["finalize"]["error"] is None
