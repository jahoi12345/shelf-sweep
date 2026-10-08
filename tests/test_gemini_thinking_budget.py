"""Thinking-token cap for the Gemini image stages (providers/gemini.py). Mock transport only: no request reaches Google."""
import asyncio

import pytest

from providers import gemini


@pytest.fixture
def sent(monkeypatch):
    calls = []

    async def fake(model, images, prompt, config, schema):
        calls.append(dict(config))
        return '{"ok": true}', {"prompt_tokens": 10, "output_tokens": 2, "thought_tokens": 0, "total_tokens": 12}
    gemini.set_transport(fake)
    for k in ("GEMINI_THINKING_BUDGET", "GEMINI_THINKING_BUDGET_STAGES"):
        monkeypatch.delenv(k, raising=False)
    yield calls
    gemini.set_transport(None)


def ask(stage, config=None, cache=None):
    return asyncio.run(gemini.structured_image_request(images=[b"\xff\xd8x"], prompt="p", response_schema={"type": "object"}, config=config or {"temperature": 0.0},
                                                       stage=stage, cache=cache))


def test_read_and_detect_requests_carry_the_default_budget_and_other_stages_do_not(sent):
    ask("read"), ask("detect"), ask("voice")
    assert [c.get("thinking_config") for c in sent] == [{"thinking_budget": 512}, {"thinking_budget": 512}, None]


def test_the_environment_sets_or_removes_the_cap_and_a_callers_own_thinking_config_wins(sent, monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "256")
    ask("read")
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "off")
    ask("read")
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "0")
    monkeypatch.setenv("GEMINI_THINKING_BUDGET_STAGES", "detect")
    ask("read"), ask("detect")
    ask("read", {"temperature": 0.0, "thinking_config": {"thinking_budget": 2048}})
    assert [c.get("thinking_config") for c in sent] == [{"thinking_budget": 256}, None, None, {"thinking_budget": 0}, {"thinking_budget": 2048}]


def test_the_cache_key_is_the_callers_config_so_old_cached_readings_stay_hits(sent, tmp_path, monkeypatch):
    cache = gemini.get_cache(tmp_path / "c")
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "off")
    first = ask("read", cache=cache)                      # cached without a cap (as before the guard)
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "512")
    again = ask("read", cache=cache)
    assert not first.cache_hit and again.cache_hit and len(sent) == 1


def test_a_bad_value_falls_back_to_the_default():
    assert gemini.thinking_budget("read", {"GEMINI_THINKING_BUDGET": "lots"}) == gemini.DEFAULT_THINKING_BUDGET
    assert gemini.thinking_budget("read", {"GEMINI_THINKING_BUDGET": "-5"}) is None or gemini.thinking_budget("read", {"GEMINI_THINKING_BUDGET": "-5"}) == 0
