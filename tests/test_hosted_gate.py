"""The Disconnect Gemini switch must make every hosted adapter un-invokable while set (no transport call), persist on disk, and clear on Reconnect."""
import asyncio
import json
from pathlib import Path

import pytest

from providers import gemini, hosted_gate as HG, stages


@pytest.fixture()
def gate_file(tmp_path, monkeypatch):
    p = tmp_path / "hosted_gate.json"
    monkeypatch.setattr(HG, "STATE_PATH", p)
    yield p
    HG.set_disabled(False, "test")


def test_transport_is_never_called_while_disabled(gate_file, monkeypatch):
    calls = []
    async def fake(model, images, prompt, config, response_schema):
        calls.append(model); return "{}", {"total_tokens": 1}
    monkeypatch.setattr(gemini, "_real_transport", lambda *a, **k: fake(*a, **k))   # would be the network call
    HG.set_disabled(True, "test")
    assert HG.disabled() and json.loads(gate_file.read_text())["disabled"] is True        # persisted server-side
    with pytest.raises(HG.HostedDisabledError) as e:
        asyncio.run(gemini._real_transport.__wrapped__("m", [], "p", {}, None)) if hasattr(gemini._real_transport, "__wrapped__") else HG.check("structured_image_request:m")
    assert "hosted disabled by user" in str(e.value) and calls == [] and HG.summary()["blocked"] == 1
    HG.set_disabled(False, "test")
    HG.check("structured_image_request:m")                                               # passes again after Reconnect
    assert HG.summary()["blocked"] == 1 and HG.summary()["status"] == "hosted enabled"


def test_real_transport_checks_gate_before_any_network(gate_file, monkeypatch):
    HG.set_disabled(True, "test")
    monkeypatch.setenv("GOOGLE_API_KEY", "not-a-real-key")
    import importlib
    with pytest.raises(HG.HostedDisabledError):                                           # raised before the credential/client code runs
        asyncio.run(gemini._real_transport("gemini-x", [], "prompt", {}, None))


def test_stage_availability_and_voice_report_the_switch(gate_file, monkeypatch, tmp_path):
    monkeypatch.setattr(gemini, "credential_available", lambda: True)
    monkeypatch.setattr(gemini, "is_mocked", lambda: False)
    HG.set_disabled(True, "test")
    cfgs, avail = stages.stage_configs({"reader": "gemini", "detector": "gemini"}, run_dir=tmp_path, cache=None, usage=None)
    for st in ("read", "detect"):
        assert avail[st]["available"] is False and "hosted disabled by user" in avail[st]["reason"] and cfgs[st].get("image_request") is None
    import app.server as S
    ok, reason = S.voice_status()
    assert ok is False and "hosted disabled by user" in reason


def test_routes_toggle_persist_and_count(gate_file):
    from fastapi.testclient import TestClient
    import app.server as S
    c = TestClient(S.app, base_url="http://127.0.0.1:4182", headers={"Origin": "http://127.0.0.1:4182"})
    st = c.post("/api/hosted/disable", json={}).json()
    assert st["disabled"] is True and st["status"] == "hosted disabled by user"
    assert c.get("/api/health").json()["hosted_switch"]["disabled"] is True
    assert c.get("/api/hosted").json()["disabled"] is True                                 # survives a "reload": read from the file, not the page
    assert c.post("/api/hosted/enable", json={}).json()["disabled"] is False
    HG.record_call("structured_image_request:m", {"total_tokens": 120, "prompt_tokens": 100}, 0.001)
    s = c.get("/api/hosted").json()
    assert s["calls"] == 1 and s["tokens"]["total_tokens"] == 120 and s["cost_usd"] == 0.001


def test_cache_hits_are_counted_and_cost_nothing(gate_file, tmp_path, monkeypatch):
    import asyncio
    from providers import gemini
    from providers.cache import DiskCache
    calls = []
    async def fake(model, images, prompt, config, response_schema):
        calls.append(1); return '{"visible_text": "A"}', {"total_tokens": 10}
    gemini.set_transport(fake)
    try:
        cache = DiskCache(tmp_path / "_hosted_cache")
        for _ in range(2):
            asyncio.run(gemini.structured_image_request(images=[b"img"], prompt="p", response_schema=None, model="m", config={"thinking_config": {"thinking_budget": 0}}, cache=cache))
    finally:
        gemini.set_transport(None)
    assert len(calls) == 1 and HG.summary()["cache_hits"] == 1                           # second identical request served from the shared cache
