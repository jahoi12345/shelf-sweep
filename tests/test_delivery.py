"""Delivery hardening checks: packet completeness, per-language fields, evidence index, bundle privacy, voice gating.
Mocked stages / synthetic inputs; they prove the behaviour they test, not recognition or accuracy."""
import asyncio
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import live_tools as LT
from app import server as s
from pipeline import contracts as C
from pipeline import packet as P
from tests.test_live_app import Harness, feed_panning, make_frame
from tests.test_live_measurement import STAGES
from tests.test_sanitize import jpeg_with_exif
from pipeline import sanitize as Z


def book(i, language=None, status="unidentified", text=None, title=None):
    return {"id": f"bk-{i:03d}", "status": status, "language": language, "visible_text": text, "title": title, "author": None, "edition": None,
            "isbn": None, "spine_height_cm": None, "spine_thickness_cm": None, "replacement_cost": None, "used_value": None, "measurement": {"status": "abstained"}}


def test_language_breakdown_counts_and_keeps_accuracy_denominators_unavailable():
    books = [book(1, "es", "identified", "CIEN AÑOS", "Cien años"), book(2, "es", text="LA SOMBRA"), book(3, "en", text="DUNE"), book(4, "mul"), book(5)]
    lb = P.language_breakdown(books)
    c = lb["counts"]
    assert (c["es"]["books"], c["es"]["identified"], c["es"]["with_visible_text"]) == (2, 1, 2)
    assert c["en"]["books"] == 1 and c["mixed"]["books"] == 1 and c["unknown"]["books"] == 1
    assert sum(g["books"] for g in c.values()) == len(books)
    for lang in ("es", "en"):
        assert lb["human_legible_spines"][lang]["status"] == "unavailable" and lb["title_accuracy"][lang]["status"] == "unavailable"
    assert "not a third language" in lb["note"]


def test_completeness_marks_every_unknown_explicitly_with_reasons():
    cp = P.completeness([book(1, "es", "identified", "X", "T"), book(2)], [{"category": "plant", "status": None, "replacement_cost": None}],
                        {"length_m": None, "abstention_reasons": ["no_wall_evidence_supplied"]}, {"shelf_run_m": None, "books_replacement_cost": None})
    assert cp["books"]["total"] == 2 and cp["books"]["title"] == {"known": 1, "null": 1} and cp["books"]["spine_height_cm"] == {"known": 0, "null": 2}
    assert cp["books"]["replacement_cost"]["null"] == 2 and cp["books"]["measurement_abstained"] == 2
    assert cp["items"]["dimensions_cm"]["known"] == 0 and "not measured" in cp["items"]["dimensions_cm"]["why"]
    assert cp["room"]["length_m"]["state"] == "null" and "no_wall_evidence_supplied" in cp["room"]["length_m"]["why"]
    assert cp["totals"]["shelf_run_m"]["state"] == "null"


@pytest.mark.asyncio
async def test_final_packet_has_evidence_index_language_completeness_and_pdf_status(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        eng.config["report_pdf"] = False
        await feed_panning(h, 6)
        await h.wait(lambda: eng.inventory is not None)
        await eng.end_sweep()
        run = h.session.run_dir
        env = C.read_json(run / "claim_packet.json")
        d = env["data"]
        assert {"language_breakdown", "completeness"} <= set(d) and d["evidence"]["index"] == "evidence_index.json"
        idx = C.read_json(run / "evidence_index.json")
        cited = {b["frame_ref"] for b in d["books"] + d["items"] if b.get("frame_ref")}
        listed = {f["path"] for f in idx["files"] if f["kind"] == "source_frame"}
        assert cited and cited == listed and not idx["missing"] and all(f["sha256"] for f in idx["files"] if f["exists"])
        assert d["evidence"]["report_pdf"]["status"] == "skipped" and "finalization_timings" in d["sweep"]
        assert set(d["sweep"]["finalization_timings"]) >= {"wait_for_inflight_detect_ocr_s", "final_merge_dedup_s", "packet_build_s", "report_render_s"}
        html = (run / "report.html").read_text()
        assert "Per-language breakdown" in html and "Completeness: known vs null" in html and "unavailable" in html


@pytest.mark.asyncio
async def test_missing_evidence_file_is_reported_not_hidden(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        eng.config["report_pdf"] = False
        await feed_panning(h, 6)
        await h.wait(lambda: eng.inventory is not None)
        victim = h.session.run_dir / "frames" / "f00000.jpg"
        await eng.end_sweep()
        d = C.read_json(h.session.run_dir / "claim_packet.json")["data"]
        victim.unlink()
        idx = P.build_evidence_index(h.session.run_dir, d)
        assert any(m.endswith("f00000.jpg") for m in idx["missing"]) or not any(b["frame_ref"].endswith("f00000.jpg") for b in d["books"] + d["items"])


def test_bundle_zip_ships_sanitised_frames_no_video_and_a_privacy_audit(tmp_path):
    async def prep():
        async with Harness(tmp_path, stages=STAGES) as h:
            eng = h.session.engine
            eng.config["report_pdf"] = False
            await feed_panning(h, 3)
            await h.wait(lambda: len(eng.frames) == 3)
            # replace one saved frame by a JPEG that carries EXIF + GPS tag + comment, as a raw phone photo would
            (h.session.run_dir / "frames" / "f00001.jpg").write_bytes(jpeg_with_exif(tmp_path))
            await eng.end_sweep()
            return h.session
    sess = asyncio.run(prep())
    s.sessions["sess1"] = sess
    try:
        c = TestClient(s.app, base_url="http://127.0.0.1:4182"); c.cookies.set("intake_owner", "owner")
        r = c.get("/api/sessions/sess1/packet")
        assert r.status_code == 200
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = z.namelist()
        assert not any(n.lower().endswith((".mov", ".mp4", ".m4v", ".webm")) for n in names) and "privacy_audit.json" in names
        assert "evidence_index.json" in names and "claim_packet.json" in names
        audit = json.loads(z.read("privacy_audit.json"))
        assert audit["video_files_bundled"] == 0 and audit["images_still_with_metadata"] == 0 and audit["images_with_metadata_removed"] >= 1
        assert not Z.jpeg_metadata(z.read("frames/f00001.jpg"))["any"]
        assert Z.jpeg_metadata((sess.run_dir / "frames" / "f00001.jpg").read_bytes())["exif"]      # local original untouched
    finally:
        s.sessions.pop("sess1", None)


def test_voice_status_never_assumes_a_model_id(monkeypatch):
    monkeypatch.setattr(s, "_live_client_factory", None)
    monkeypatch.setattr(s, "_has_api_key", lambda: False)
    ok, why = s.voice_status()
    assert not ok and "GOOGLE_API_KEY" in why
    monkeypatch.setattr(s, "_has_api_key", lambda: True)
    monkeypatch.setattr(LT, "LIVE_MODEL_ID", None)
    ok, why = s.voice_status()
    assert not ok and "LIBRARY_LIVE_MODEL" in why and "no Live model id is assumed" in why
    monkeypatch.setattr(LT, "LIVE_MODEL_ID", "some-model-the-user-verified")
    assert s.voice_status() == (True, None)


def test_no_default_live_model_id_in_source():
    src = (Path(__file__).resolve().parents[1] / "app" / "live_tools.py").read_text()
    assert "gemini-3.8-live" not in src.split("LIVE_MODEL_ID")[1][:400]


@pytest.mark.asyncio
async def test_degraded_session_reports_the_exact_voice_reason_and_health_agrees(tmp_path, monkeypatch):
    monkeypatch.setattr(LT, "LIVE_MODEL_ID", None)
    async with Harness(tmp_path, voice=False, stages=STAGES) as h:
        sess = next(m for m in h.sent if m["type"] == "session")
        ready = next(m for m in h.sent if m["type"] == "ready")
        assert ready["voice"] is False and ready["voice_reason"] and ready["voice_reason"] == sess["voice_reason"]
        assert ready["voice_reason"] in sess["notice"] and "no hosted credential" in ready["voice_reason"] or "LIBRARY_LIVE_MODEL" in ready["voice_reason"]
    h2 = s.health()
    assert h2["voice_available"] in (True, False) and (h2["voice_available"] or h2["voice_reason"])


def test_smoke_live_is_skipped_without_a_model_id(monkeypatch):
    from app import smoke
    monkeypatch.setattr(LT, "LIVE_MODEL_ID", None)
    monkeypatch.setattr(smoke.gemini, "credential_available", lambda: True)
    r = asyncio.run(smoke.check_live())
    assert r["verdict"] == "SKIPPED" and "LIBRARY_LIVE_MODEL" in r["reason"]
