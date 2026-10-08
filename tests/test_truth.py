"""Held-out truth capture: validation, isolation from every pipeline path, REST behaviour. Mocked stages."""
import asyncio
import io
import json
import re
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import server as s
from app import truth as T
from pipeline import contracts as C
from pipeline import scale as S
from tests.test_live_app import Harness, feed_panning
from tests.test_live_measurement import STAGES

ROOT = Path(__file__).resolve().parents[1]
H = {"origin": "http://127.0.0.1:4182"}
FRAME = {"frame_id": "f00003", "timestamp_s": 1.5, "width_px": 1920, "height_px": 1080}


@pytest.fixture(autouse=True)
def isolated_truth_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "TRUTH_DIR", tmp_path / "truth")
    yield


def add(**kw):
    base = dict(frame=FRAME, scan={"label": "YugoTest2"}, tap_px=[400, 300], height=23.5, thickness=3.1, unit="cm", annotator="Alex")
    base.update(kw)
    return T.add_spine("sess", **base)


def test_valid_entry_is_saved_as_evaluation_truth_outside_any_run_dir(tmp_path):
    smp = add(language="es", description="unit 2, shelf 3, 4th")
    doc = json.loads((tmp_path / "truth" / "sess.json").read_text())
    assert doc["purpose"] == "evaluation_truth" and doc["annotator"] == "Alex" and doc["scan"]["label"] == "YugoTest2"
    assert smp["height_cm"] == 23.5 and smp["thickness_cm"] == 3.1 and smp["frame_id"] == "f00003" and smp["tap_px"] == [400, 300]
    assert "runs" not in str((tmp_path / "truth" / "sess.json").relative_to(tmp_path)).split("/")


def test_unit_conversion_and_plausibility_and_attribution():
    assert add(height=9.3, thickness=1.2, unit="in")["height_cm"] == pytest.approx(23.62, abs=0.01)
    assert add(height=235, thickness=31, unit="mm")["height_cm"] == 23.5
    for bad in (dict(height=2.0), dict(height=23, thickness=40), dict(thickness=0.05), dict(height=23, thickness=23), dict(tap_px=[5000, 10]),
                dict(tap_px=None), dict(annotator=" "), dict(unit="yards"), dict(language="fr"), dict(height=float("nan")), dict(height=True)):
        with pytest.raises(T.TruthError):
            add(**bad)


def test_cap_delete_and_unique_ids():
    for i in range(T.MAX_SPINES):
        add(tap_px=[10 + i, 10])
    with pytest.raises(T.TruthError):
        add()
    T.remove_spine("sess", "s05")
    assert add()["sample_id"]                                             # a freed id may be reused, never duplicated
    ids = [x["sample_id"] for x in T.load("sess")["spine_samples"]]
    assert len(ids) == len(set(ids))
    with pytest.raises(T.TruthError):
        T.remove_spine("sess", "s99")


def test_room_reading_is_validated_and_unit_converted():
    r = T.set_room("sess", length=15, width=12, height=9, unit="ft", annotator="Alex", notes="door on west wall")
    assert r["length_m"] == pytest.approx(4.572) and r["height_m"] == pytest.approx(2.743, abs=1e-3)
    for bad in (dict(length=0.1), dict(height=20), dict(unit="mi"), dict(annotator="")):
        with pytest.raises(T.TruthError):
            T.set_room("sess", **{**dict(length=4, width=3, height=2.5, unit="m", annotator="Alex"), **bad})


def test_invalid_session_id_cannot_escape_the_truth_dir():
    with pytest.raises(T.TruthError):
        T.load("../../etc/passwd")


def test_no_pipeline_engine_or_provider_source_references_the_truth_store():
    for pattern in ("pipeline/*.py", "providers/*.py", "app/engine.py", "app/measurement.py", "app/replay.py"):
        for f in ROOT.glob(pattern):
            text = f.read_text()
            assert not re.search(r"truth_capture|app\.truth|from app import truth|evaluation_truth\.json", text), f"{f} must not touch held-out truth"


def test_stages_refuse_a_truth_document_as_input(tmp_path):
    add()
    p = tmp_path / "truth" / "sess.json"
    with pytest.raises(C.ContractError):
        S.load_calibrations(p)


def test_rest_saves_truth_but_it_never_reaches_the_bundle_or_the_packet(tmp_path):
    async def prep():
        async with Harness(tmp_path, stages=STAGES) as h:
            h.session.engine.config["report_pdf"] = False
            await feed_panning(h, 4)
            await h.wait(lambda: len(h.session.engine.frames) >= 4)
            return h.session
    sess = asyncio.run(prep())
    s.sessions["sess1"] = sess
    try:
        c = TestClient(s.app, base_url="http://127.0.0.1:4182"); c.cookies.set("intake_owner", "owner")
        fid = sess.engine.frames[1]["frame_id"]
        ok = {"frame_id": fid, "tap_px": [500, 400], "height": 22.0, "thickness": 2.8, "unit": "cm", "annotator": "Alex", "language": "en"}
        assert c.post("/api/sessions/sess1/truth/spines", json=ok, headers=H).status_code == 200
        assert c.post("/api/sessions/sess1/truth/spines", json={**ok, "frame_id": "f99999"}, headers=H).status_code == 400
        assert c.post("/api/sessions/sess1/truth/spines", json={**ok, "height": 200}, headers=H).status_code == 400
        assert c.post("/api/sessions/sess1/truth/room", json={"length": 4.2, "width": 3.1, "height": 2.5, "unit": "m", "annotator": "Alex"}, headers=H).status_code == 200
        got = c.get("/api/sessions/sess1/truth").json()
        assert got["purpose"] == "evaluation_truth" and len(got["spine_samples"]) == 1 and got["room"]["length_m"] == 4.2
        assert c2_status(s.app) == 404                                                  # no owner cookie: no access
        # finish the sweep; the packet, ZIP and every file in the run dir contain none of the truth values/names
        asyncio.run(sess.engine.end_sweep())
        z = zipfile.ZipFile(io.BytesIO(c.get("/api/sessions/sess1/packet").content))
        blob = b"".join(z.read(n) for n in z.namelist() if n.endswith((".json", ".html")))
        assert b"evaluation_truth" not in blob and b"truth_capture" not in blob and not any("truth" in n for n in z.namelist())
        assert not list(sess.run_dir.rglob("*truth*"))
    finally:
        s.sessions.pop("sess1", None)


def c2_status(app):
    return TestClient(app, base_url="http://127.0.0.1:4182").get("/api/sessions/sess1/truth").status_code
