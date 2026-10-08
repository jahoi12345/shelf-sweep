import asyncio
import json
from pathlib import Path

import pytest

from providers import readers as R


def _adapter_factory(**kw):
    async def adapter(*, image_paths, prompt, response_schema, task, config=None):
        return {"data": {"visible_text": "HOLA", "title_text": "Hola", "author_text": None, "publisher_text": None, "edition_text": None, "isbn_text": None,
                         "title_complete": False, "read_confidence": 0.4, "issues": []}, "metrics": {"requests": []}}
    return adapter


def _fake_stage_configs(config, *, run_dir, cache, usage):
    assert config.get("reader") in (None, "none") or config["reader"] not in R.reader_names()
    return ({"read": {"no_provider": True}, "detect": {"no_provider": True}, "resolve": {}, "dedup": {}},
            {"read": {"selection": "none", "available": False, "reason": "x"}, "detect": {"selection": "none", "available": False, "reason": "x"}})


class S:
    def __init__(self, **kw): self.__dict__.update(kw)
    detect = read = resolve = merge = None


def test_register_validation_and_default_untouched(monkeypatch):
    for bad in ("easyocr", "Bad Name", "x"):
        with pytest.raises(R.PluginError):
            R.register_reader(bad, _adapter_factory)
    monkeypatch.delenv("LIBRARY_READER", raising=False)
    monkeypatch.delenv("LIBRARY_ALLOW_HOSTED", raising=False)
    cfgs, avail, stages, notes = R.apply({}, S(), run_dir="r", cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
    assert avail["read"]["selection"] != "gemini" and "assoc" in notes          # auto never hosted; the open reader (if installed) or none


def test_plugin_reader_is_selected_by_name(tmp_path):
    R.register_reader("fake_reader", _adapter_factory, description="test")
    try:
        cfgs, avail, _, notes = R.apply({"reader": "fake_reader"}, S(), run_dir=tmp_path, cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
        assert notes["reader_plugin"] == "fake_reader" and avail["read"]["available"] and cfgs["read"]["no_provider"] is False
        out = asyncio.run(cfgs["read"]["image_request"](image_paths=[], prompt="", response_schema={}, task="read"))
        assert out["data"]["title_text"] == "Hola"
    finally:
        R.unregister_reader("fake_reader")


def _ext(tmp_path, ts_shift=0.0):
    d = tmp_path / "ext"; d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"data": {"frames": [{"frame_id": "f00000", "timestamp_s": 1.0}, {"frame_id": "f00001", "timestamp_s": 2.0}]}}))
    obs = [{"observation_id": f"f00000-o00{i}", "frame_id": "f00000", "bbox_xyxy": [i, 0, i + 5, 50], "kind": "book"} for i in range(3)]
    (d / "candidates.json").write_text(json.dumps({"run_id": "ext", "data": {"observations": obs}}))
    return d / "candidates.json"


def test_external_detector_rekeys_by_timestamp_and_flags_gaps(tmp_path):
    ext = R.ExternalDetect(_ext(tmp_path))
    man = tmp_path / "mini.json"
    man.write_text(json.dumps({"schema_version": 1, "run_id": "live", "stage": "manifest", "status": "ok", "issues": [], "metrics": {},
                               "data": {"frames": [{"frame_id": "f00007", "timestamp_s": 1.03}, {"frame_id": "f00008", "timestamp_s": 5.0}]}}))
    run = tmp_path / "run"; run.mkdir()
    path = ext(man, run, {"output_name": "d.json"})
    doc = json.loads(Path(path).read_text())
    ids = [o["observation_id"] for o in doc["data"]["observations"]]
    assert ids == ["f00007-o000", "f00007-o001", "f00007-o002"]
    assert all(o["external_source"]["sha256"] and o["frame_id"] == "f00007" for o in doc["data"]["observations"])
    assert doc["status"] == "partial" and "f00008" in json.dumps(doc["issues"])


def test_apply_external_and_missing_cross_view(tmp_path, monkeypatch):
    p = _ext(tmp_path)
    monkeypatch.setenv("LIBRARY_ASSOC", "cross_view")
    cfgs, avail, stages, notes = R.apply({"detector": f"external:{p}"}, S(merge="dedup"), run_dir=tmp_path, cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
    assert avail["detect"]["selection"] == "external" and avail["detect"]["available"]
    assert callable(stages.detect)
    try:
        import pipeline.cross_view as cv
    except ImportError:          # optional dependency (scipy) absent in this venv: the hook must fall back and say so
        cv = None
    if cv is not None and hasattr(cv, "merge"):
        assert notes["assoc"] == "cross_view"
    else:
        assert "unavailable" in notes["assoc_error"] and stages.merge == "dedup"
    _, avail2, _, notes2 = R.apply({"detector": "external:/nope/candidates.json"}, S(), run_dir=tmp_path, cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
    assert not avail2["detect"]["available"] and "external_candidates_error" in notes2


def test_overlap_fill_adds_intermediate_detection_frames():
    from app.engine import SweepEngine
    e = SweepEngine.__new__(SweepEngine)
    e.min_overlap, e.extra_detect_max = 0.3, 10
    e.overlap_fill = {"extra_frames": [], "unfillable_gaps": [], "budget_exhausted": False}
    from collections import Counter
    e.counters = Counter()
    # 11 saved frames, 0.2 widths apart -> keyframes at 0 and 10 are 2.0 widths apart (no overlap)
    e.frames = [{"frame_id": f"f{i:05d}", "native_frame_index": i, "quality": {"reasons": [], "sharpness": 100 + i}} for i in range(11)]
    e._cum = [0.2 * i for i in range(11)]
    e._last_det_idx = 0
    started = []
    e._start_keyframe = lambda fr: started.append(fr["native_frame_index"])
    e._fill_overlap(e.frames[10])
    seq = [0] + started + [10]
    assert started and all(e._cum[b] - e._cum[a] <= 0.7 + 1e-9 for a, b in zip(seq, seq[1:]))
    assert e.overlap_fill["extra_frames"] and not e.overlap_fill["budget_exhausted"]


def test_rapidocr_registered_and_auto_policy(monkeypatch, tmp_path):
    assert "rapidocr" in R.reader_names()
    from providers import gemini
    monkeypatch.delenv("LIBRARY_READER", raising=False)
    monkeypatch.setattr(gemini, "credential_available", lambda: False)
    monkeypatch.setattr(gemini, "is_mocked", lambda: False)
    monkeypatch.setattr(R, "rapidocr_available", lambda: True)
    assert R.effective_reader({})["effective"] == "rapidocr"
    monkeypatch.setattr(R, "rapidocr_available", lambda: False)
    assert R.effective_reader({})["effective"] in ("easyocr", "none")         # never apple
    assert R.effective_reader({"reader": "apple"})["effective"] == "apple"    # only when explicitly requested
    monkeypatch.setattr(gemini, "credential_available", lambda: True)
    assert R.effective_reader({})["effective"] != "gemini"                    # a key alone never makes auto hosted
    monkeypatch.setenv("LIBRARY_ALLOW_HOSTED", "1")
    assert R.effective_reader({})["effective"] == "gemini"                    # hosted first only when explicitly allowed
    monkeypatch.delenv("LIBRARY_ALLOW_HOSTED", raising=False)
    monkeypatch.setattr(gemini, "credential_available", lambda: False)
    monkeypatch.setattr(R, "rapidocr_available", lambda: True)
    monkeypatch.setitem(R._READERS["rapidocr"], "factory", _adapter_factory)
    cfgs, avail, _, notes = R.apply({}, S(), run_dir=tmp_path, cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
    assert avail["read"]["selection"] == "rapidocr" and notes["reader_auto"]["effective"] == "rapidocr"
    _, avail, _, notes = R.apply({"provider": "off"}, S(), run_dir=tmp_path, cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
    assert avail["read"]["selection"] == "none"                                 # off disables auto-selection


def test_florence_optional_runtime_degrades_cleanly(monkeypatch, tmp_path):
    monkeypatch.setenv("LIBRARY_FLORENCE_PYTHON", str(tmp_path / "no_such_python"))
    assert "Florence runtime missing" in R.florence_unavailable_reason()
    cfgs, avail, _, notes = R.apply({"reader": "florence"}, S(), run_dir=tmp_path, cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
    assert avail["read"]["available"] is False and "unavailable: Florence runtime missing" in avail["read"]["reason"] and "reader_plugin_error" in notes
    R.register_reader("fake_primary", _adapter_factory)
    try:
        cfgs, avail, _, notes = R.apply({"reader": "fake_primary", "reader_second": "florence"}, S(), run_dir=tmp_path, cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
        assert avail["read"]["available"] and avail["read2"]["available"] is False and "read2" not in cfgs and "reader_second_error" in notes
        R.register_reader("fake_second", _adapter_factory)
        cfgs, avail, _, notes = R.apply({"reader": "fake_primary", "reader_second": "fake_second"}, S(), run_dir=tmp_path, cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
        assert avail["read2"]["selection"] == "fake_second" and cfgs["read2"]["image_request"] is not cfgs["read"]["image_request"] and notes["reader_second"] == "fake_second"
    finally:
        R.unregister_reader("fake_primary"); R.unregister_reader("fake_second")


def test_track_v2_hook_falls_back_until_tracker_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("LIBRARY_ASSOC", "track_v2")
    _, _, stages, notes = R.apply({}, S(merge="dedup"), run_dir=tmp_path, cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
    try:
        import pipeline.tracker as tr
    except ImportError:
        tr = None
    if tr is not None and hasattr(tr, "merge"):
        assert notes["assoc"] == "track_v2" and stages.merge is tr.merge
    else:
        assert "pipeline.tracker.merge unavailable" in notes["assoc_error"] and stages.merge == "dedup"
    monkeypatch.setenv("LIBRARY_ASSOC", "bogus")
    _, _, _, notes = R.apply({}, S(merge="dedup"), run_dir=tmp_path, cache=None, usage=None, stage_configs_fn=_fake_stage_configs)
    assert "unknown LIBRARY_ASSOC" in notes["assoc_error"]


def test_auto_never_hosted_without_allow_flag(monkeypatch, tmp_path):
    from providers import gemini, stages
    monkeypatch.setattr(gemini, "credential_available", lambda: True)
    monkeypatch.setattr(gemini, "is_mocked", lambda: False)
    monkeypatch.delenv("LIBRARY_ALLOW_HOSTED", raising=False)
    for var in ("LIBRARY_DETECTOR", "LIBRARY_READER"):
        monkeypatch.delenv(var, raising=False)
    cfgs, avail = stages.stage_configs({}, run_dir=tmp_path, cache=None, usage=None)
    assert avail["detect"]["selection"] != "gemini" and avail["read"]["selection"] != "gemini" and avail["read"]["hosted_allowed"] is False
    assert R.effective_reader({})["effective"] != "gemini" and "LIBRARY_ALLOW_HOSTED" in R.effective_reader({})["reason"]
    monkeypatch.setenv("LIBRARY_ALLOW_HOSTED", "1")
    assert R.effective_reader({})["effective"] == "gemini"
    monkeypatch.delenv("LIBRARY_ALLOW_HOSTED", raising=False)
    cfgs, avail = stages.stage_configs({"reader": "gemini"}, run_dir=tmp_path, cache=None, usage=None)   # explicit selection still works
    assert avail["read"]["selection"] == "gemini" and avail["read"]["hosted"] is True


def test_hosted_cap_falls_back_and_reports():
    import asyncio
    calls = {"hosted": 0, "fb": 0}
    async def hosted(**kw):
        calls["hosted"] += 1
        return {"data": {"visible_text": "HOSTED", "issues": []}, "metrics": {}}
    async def fb(**kw):
        calls["fb"] += 1
        return {"data": {"visible_text": "FREE", "issues": []}, "metrics": {}}
    c = R.HostedCapped(hosted, fb, 2)
    outs = [asyncio.run(c(image_paths=["x"], task="read")) for _ in range(4)]
    assert calls == {"hosted": 2, "fb": 2} and [o["data"]["visible_text"] for o in outs] == ["HOSTED", "HOSTED", "FREE", "FREE"]
    assert "hosted_cap_reached: read by the free fallback reader" in outs[3]["data"]["issues"] and outs[3]["metrics"]["hosted_cap_reached"]
    rep = c.report()
    assert rep["cap_reached"] and rep["status"] == "cap reached" and rep["hosted_calls"] == 2 and rep["fallback_calls"] == 2
    nofb = R.HostedCapped(hosted, None, 0)
    with pytest.raises(R.PluginError):
        asyncio.run(nofb(image_paths=["x"], task="read"))
    assert R.hosted_max_calls({}) == 800 and R.hosted_max_calls({"hosted_max_calls": "5"}) == 5


def test_hosted_reader_recipe_config_and_concurrency(monkeypatch, tmp_path):
    from providers import gemini, stages
    monkeypatch.setattr(gemini, "credential_available", lambda: True)
    monkeypatch.setattr(gemini, "is_mocked", lambda: False)
    monkeypatch.setattr(gemini, "make_adapter", lambda **kw: ("adapter", kw.get("config")))
    monkeypatch.delenv("LIBRARY_READER_THINKING_BUDGET", raising=False)
    cfgs, avail = stages.stage_configs({"reader": "gemini", "detector": "none"}, run_dir=tmp_path, cache=None, usage=None)
    gen = cfgs["read"]["hosted_generation"]
    assert gen["temperature"] == 0 and gen["thinking_config"] == {"thinking_budget": 0}       # literal text, minimal thinking by default
    monkeypatch.setenv("LIBRARY_READER_THINKING_BUDGET", "")
    cfgs, _ = stages.stage_configs({"reader": "gemini", "detector": "none"}, run_dir=tmp_path, cache=None, usage=None)
    assert "thinking_config" not in cfgs["read"]["hosted_generation"]                        # unset = SDK default
    from app.engine import SweepEngine
    e = SweepEngine(tmp_path / "r", config={"provider": "off"})
    assert int(e.config.get("hosted_read_concurrency", 16)) == 16 and e.read_workers == 3        # hosted concurrency is separate from the CPU pool
