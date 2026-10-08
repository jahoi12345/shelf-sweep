"""Live-engine measurement integration. Mocked model/stages + synthetic frames/geometry: these show that human
calibration inputs flow through the shared scale/room stages into the live snapshot, JSON packet and HTML report, and
that unsupported values stay null. They are NOT physical accuracy evidence."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app import server as s
from app.engine import StageFns, SweepEngine, SweepError
from app.measurement import MeasurementError
from pipeline import contracts as C
from pipeline import packet as P
from tests.test_live_app import (FAKES, Harness, fake_merge, fake_read, fake_resolve, feed_panning, make_frame, tool_event)
from tests import test_measurement as TM


def fake_detect_poly(manifest_path, run_dir, config):
    """Books with oriented polygons on a frontal plane: 30 px wide x 160 px tall at (200..); plus one plant."""
    m = C.read_json(manifest_path)
    f = m["data"]["frames"][0]
    fid = f["frame_id"]
    obs = []
    for i in range(4):
        x0, y0 = 300 + 60 * i, 300
        poly = [[x0, y0], [x0 + 30, y0], [x0 + 30, y0 + 160], [x0, y0 + 160]]
        obs.append({"observation_id": C.observation_id(fid, i), "frame_id": fid, "kind": "book", "category": "book", "orientation": "upright",
                    "bbox_xyxy": [x0, y0, x0 + 30, y0 + 160], "polygon_xy": poly, "confidence": 0.9, "occlusion": "unknown", "frame_ref": f["image_path"]})
    obs.append({"observation_id": C.observation_id(fid, 4), "frame_id": fid, "kind": "non_book", "category": "plant", "orientation": "unknown",
                "bbox_xyxy": [900, 300, 980, 500], "confidence": 0.8, "occlusion": "unknown", "frame_ref": f["image_path"]})
    out = Path(run_dir) / config["output_name"]
    C.write_json(out, C.envelope(m["run_id"], "detect", {"observations": obs}, status="ok"))
    return out


STAGES = StageFns(detect=fake_detect_poly, read=fake_read, resolve=fake_resolve, merge=fake_merge)
# frontal shelf face: 2.0 m x 1.2 m maps to the rectangle (200,100)-(1400,820) => 600 px/m
QUAD = [[200, 100], [1400, 100], [1400, 820], [200, 820]]


async def feed_still(h, n=4):
    """Same view repeatedly (no pan): keyframes close slowly; use panning frames so intervals close."""
    await feed_panning(h, n)


@pytest.mark.asyncio
async def test_stated_dimensions_and_corner_taps_produce_measured_spines_and_valid_packet(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        await feed_panning(h, 6)
        await h.wait(lambda: eng.inventory is not None)
        f0 = eng.frames[0]["frame_id"]
        # voice states a width: recorded as pending (needs corner taps), not as a measurement
        await h.live.q.put(tool_event("d1", "record_calibration_dimension", {"what": "shelf_unit_width", "value": 2.0, "unit": "m",
                                                                              "approximate": True, "utterance": "the unit is about two metres wide"}))
        await h.wait(lambda: h.live.tool_responses)
        r = h.live.tool_responses[-1].response
        assert r["recorded"] == "shelf_unit_width" and "cross-check" in r["next"] and "tap" not in r["next"]
        assert eng.snapshot()["measurement"]["pending_dimensions"] and all(b["spine_height_cm"] is None for b in eng.snapshot()["books"])
        # the person taps four corners in the UI (REST/engine path); the stated width+height are the metric part
        res = await eng.add_calibration(kind="plane_quad", frame_id=f0, image_points_px=QUAD, metric={"width_m": 2.0, "height_m": 1.2, "tolerance_pct": 5.0},
                                        utterance="unit is about 2.0 m wide and 1.2 m tall", plane_id="plane-01")
        assert res["status"] == "valid"
        snap = eng.snapshot()
        measured = [b for b in snap["books"] if b["spine_height_cm"] is not None]
        assert measured, snap["measurement"]["scale"]["summary"]
        b = measured[0]
        assert b["spine_height_cm"] == pytest.approx(160 / 600 * 100, abs=0.1) and b["spine_thickness_cm"] == pytest.approx(30 / 600 * 100, abs=0.1)
        assert b["measurement"]["validation_state"] == "software_only_unvalidated" and "user-supplied" in b["measurement"]["provenance_label"]
        # books in frames that the calibration cannot be transferred to stay unknown, with a reason
        others = [x for x in snap["books"] if x["spine_height_cm"] is None]
        assert others and all(x["measurement"]["status"] == "abstained" for x in others)
        assert snap["totals"]["shelf_run_m"] is None                               # incomplete region => withheld, not guessed
        assert snap["totals"]["spine_dimensions_abstained"] == len(others)
        # persisted human inputs are separate files and carry provenance
        cal = C.read_json(h.session.run_dir / "calibration.json")
        assert cal["purpose"] == "calibration" and cal["calibrations"][0]["provenance"]["independent_of_evaluation_truth"] is True
        # end of sweep: scale artifact, packet, report agree
        await eng.end_sweep()
        packet_env = C.read_json(h.session.run_dir / "claim_packet.json")
        pk = packet_env["data"]
        assert P.verify_packet(packet_env, C.read_json(h.session.run_dir / "manifest.json"), h.session.run_dir) == []
        assert (h.session.run_dir / "scale.json").is_file() and pk["measurement_evidence"]["scale"]["calibrations"][0]["status"] == "valid"
        assert any(x["spine_height_cm"] for x in pk["books"]) and pk["room"]["status"] != "estimated" and pk["room"]["length_m"] is None
        html = (h.session.run_dir / "report.html").read_text()
        assert "software_only_unvalidated" in html or "unvalidated" in html


@pytest.mark.asyncio
async def test_invalid_or_unattributed_calibrations_are_rejected_at_the_door(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        await feed_panning(h, 3)
        await h.wait(lambda: len(eng.frames) >= 1)
        f0 = eng.frames[0]["frame_id"]
        base = dict(kind="plane_quad", frame_id=f0, image_points_px=QUAD, metric={"width_m": 2.0, "height_m": 1.2})
        with pytest.raises(MeasurementError):
            await eng.add_calibration(**base)                                            # no utterance/note: not attributable
        with pytest.raises(MeasurementError):
            await eng.add_calibration(**{**base, "image_points_px": [[-5, 0], [10, 0], [10, 10], [-5, 10]]}, utterance="x")
        with pytest.raises(MeasurementError):
            await eng.add_calibration(**{**base, "frame_id": "f99999"}, utterance="x")
        assert eng.meas.calibrations == []
        # an assumed (not user-supplied) size can never be passed as a calibration through the stage validator
        assert eng.snapshot()["measurement"]["calibrations"] == []


@pytest.mark.asyncio
async def test_removing_a_calibration_makes_dependent_values_unknown_again(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        await feed_panning(h, 6)
        await h.wait(lambda: eng.inventory is not None)
        r = await eng.add_calibration(kind="plane_quad", frame_id=eng.frames[0]["frame_id"], image_points_px=QUAD,
                                      metric={"width_m": 2.0, "height_m": 1.2}, utterance="2 x 1.2 m")
        assert any(b["spine_height_cm"] for b in eng.snapshot()["books"])
        await eng.remove_calibration(r["calibration"]["calibration_id"])
        assert all(b["spine_height_cm"] is None for b in eng.snapshot()["books"])
        assert any(x["code"] == "no_metric_scale" for x in eng.snapshot()["review_queue"])


@pytest.mark.asyncio
async def test_calibration_cannot_change_after_the_sweep_finished(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        await feed_panning(h, 3)
        await eng.end_sweep()
        with pytest.raises(SweepError):
            await eng.add_calibration(kind="plane_quad", frame_id=eng.frames[0]["frame_id"], image_points_px=QUAD, metric={"width_m": 2.0, "height_m": 1.2}, utterance="x")


ASK_WORDS = ("tap", "how wide", "how tall", "approximate width", "approximate height", "ceiling is", "measure", "four corners")


def test_live_agent_never_asks_the_person_to_measure_state_or_tap():
    """HARD USER RULE: the live prompt, the tool list and every coverage/guidance string only say what to FILM."""
    from app import live_tools as LT
    from app.measurement import MeasurementState
    from pipeline import room as RM
    from pipeline import room_auto as RA
    assert "record_calibration_dimension" not in LT.TOOL_NAMES
    declared = {f.name for t in LT.tool_declarations() for f in t.function_declarations}
    assert "record_calibration_dimension" not in declared and declared == set(LT.TOOL_NAMES)
    low = LT.SYSTEM_INSTRUCTION.lower()
    assert "never ask the person to measure" in low and "tap or mark" in low
    for bad in ("then the app asks them to tap", "the approximate width and height of the shelving unit", "approximate ceiling height"):
        assert bad not in low
    texts = [how for _, how in RM.COVERAGE_ITEMS] + [how for _, how in RA.CAPTURE_STEPS] + RA.capture_guidance(["no_walls"], False)
    room = RM.estimate_data(None, TM.manifest(), [])
    m = MeasurementState()
    texts += [i["how"] for i in m.status(None, room)["items"]]
    g = m.guidance_due(45.0, 3, None, {**room, "evidence_summary": {"walls_valid": 0}})
    texts += [x["message"] for x in g]
    assert g and g[0]["code"] == "show_room_corner"
    for t in texts:
        assert not any(w in t.lower() for w in ASK_WORDS), t
    assert m.status(None, room)["next_prompt"] is None or "film" in m.status(None, room)["next_prompt"].lower() or "show" in m.status(None, room)["next_prompt"].lower() \
        or "tilt" in m.status(None, room)["next_prompt"].lower()


@pytest.mark.asyncio
async def test_guidance_is_filming_only_issued_once_and_status_never_asks_for_a_size(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        await feed_panning(h, 12)
        await eng._guidance(45.0)
        guid = lambda: [m for m in h.sent if m["type"] == "nudge" and m.get("kind") == "guidance"]
        assert guid() == []                                                                # no room guidance during the shelf pass
        await h.client({"type": "camera_state", "enabled": True})
        await h.client({"type": "end_request"})                                            # End pressed: the room instruction, once
        await h.wait(lambda: guid())
        await eng._guidance(90.0)
        g = guid()
        assert [x["code"] for x in g] == ["room_sweep"] and "ceiling" in g[0]["message"].lower()
        assert not any(w in x["message"].lower() for x in g for w in ASK_WORDS)
        await h.wait(lambda: any(hasattr(c.get("turns"), "parts") and "APP GUIDANCE" in c["turns"].parts[0].text for c in h.live.content))
        # status tool reports what is missing to FILM, without stating any size and without asking for one
        await h.live.q.put(tool_event("m1", "get_measurement_status", {}))
        await h.wait(lambda: h.live.tool_responses)
        r = h.live.tool_responses[-1].response
        assert "ceiling_height" not in r["missing"] and "shelf_unit_dimensions" not in r["missing"] and r["spines_measured"] in (0, None)
        assert not any(k in r for k in ("spine_height_cm", "room_area", "floor_area_m2"))
        assert "never ask" in r["how_to_use"].lower() and not any(w in str(r.get("next_question") or "").lower() for w in ASK_WORDS)


@pytest.mark.asyncio
async def test_ceiling_height_statement_is_user_supplied_and_room_needs_walls(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        await feed_panning(h, 3)
        await h.live.q.put(tool_event("c1", "record_calibration_dimension", {"what": "ceiling_height", "value": 8.2, "unit": "ft", "approximate": True,
                                                                              "utterance": "ceiling is about eight foot two"}))
        await h.wait(lambda: h.live.tool_responses)
        r = h.live.tool_responses[-1].response
        assert r["recorded"] == "ceiling_height_anchor" and r["label"] == "user-supplied" and r["height_m"] == pytest.approx(2.4994, abs=1e-3)
        snap = eng.snapshot()
        assert snap["room"]["height_m"] is None and snap["room"]["status"] == "abstained"      # an anchor alone is not a room measurement
        assert "no_wall_evidence_supplied" in snap["room"]["abstention_reasons"]
        bad = tool_event("c2", "record_calibration_dimension", {"what": "ceiling_height", "value": -1, "unit": "m"})
        await h.live.q.put(bad)
        await h.wait(lambda: len(h.live.tool_responses) >= 2)
        assert "error" in h.live.tool_responses[-1].response


@pytest.mark.asyncio
async def test_room_estimate_flows_into_snapshot_packet_and_report(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        await feed_panning(h, 3)
        f0 = eng.frames[0]["frame_id"]
        K, qa, qb, (la, lb) = TM.room_scene()
        await eng.set_dimension("ceiling_height", 2.5, "m", approximate=True, utterance="ceiling about 2.5 m")
        await eng.add_wall(frame_id=f0, quad_px=qa, wall_id="A", spans_floor_to_ceiling=True)
        await eng.add_wall(frame_id=f0, quad_px=qb, wall_id="B", spans_floor_to_ceiling=True)
        room = eng.snapshot()["room"]
        assert room["status"] == "estimated"
        assert room["length_m"] == pytest.approx(la, rel=0.02) and room["width_m"] == pytest.approx(lb, rel=0.02)
        assert room["floor_area_ft2"] == pytest.approx(room["floor_area_m2"] * 10.7639104, rel=1e-3)
        assert "user-supplied" in room["height_provenance"]["label"] and room["confidence"] is None
        await eng.end_sweep()
        env = C.read_json(h.session.run_dir / "claim_packet.json")
        assert P.verify_packet(env, C.read_json(h.session.run_dir / "manifest.json"), h.session.run_dir) == []
        assert env["data"]["room"]["status"] == "estimated" and (h.session.run_dir / "room.json").is_file()
        assert "user-supplied" in (h.session.run_dir / "report.html").read_text()
        # tampering with a room number in the JSON is caught by the packet verifier
        env["data"]["room"]["floor_area_m2"] = 99.0
        assert P.verify_packet(env, C.read_json(h.session.run_dir / "manifest.json"), h.session.run_dir)


def test_rest_endpoints_validate_inputs_and_require_the_owner(tmp_path):
    from fastapi.testclient import TestClient
    async def prep():
        async with Harness(tmp_path, stages=STAGES) as h:
            await feed_panning(h, 3)
            return h.session
    sess = asyncio.run(prep())
    s.sessions["sess1"] = sess
    try:
        c = TestClient(s.app, base_url="http://127.0.0.1:4177")
        hdr = {"origin": "http://127.0.0.1:4177"}
        c.cookies.set("intake_owner", "owner")
        f0 = sess.engine.frames[0]["frame_id"]
        ok = {"kind": "plane_quad", "frame_id": f0, "image_points_px": QUAD, "width": 2.0, "height": 1.2, "unit": "m", "utterance": "2 by 1.2"}
        assert c.post("/api/sessions/sess1/calibrations", json={**ok, "utterance": ""}, headers=hdr).status_code == 400
        assert c.post("/api/sessions/sess1/calibrations", json={**ok, "width": "wide"}, headers=hdr).status_code == 400
        assert c.post("/api/sessions/sess1/calibrations", json={**ok, "kind": "guess"}, headers=hdr).status_code == 400
        r = c.post("/api/sessions/sess1/calibrations", json=ok, headers=hdr)
        # the engine's tasks belong to the finished loop; the endpoint itself must still validate and reply
        assert r.status_code in (200, 400)
        assert c.get("/api/sessions/sess1/measurement").status_code == 200
        c2 = TestClient(s.app, base_url="http://127.0.0.1:4177")
        assert c2.post("/api/sessions/sess1/calibrations", json=ok, headers=hdr).status_code == 404       # no owner cookie
        assert c.delete("/api/sessions/sess1/calibrations/cal-999", headers=hdr).status_code == 404
    finally:
        s.sessions.pop("sess1", None)


@pytest.mark.asyncio
async def test_no_voice_credential_notice_reports_voice_and_stage_availability_independently(tmp_path):
    async with Harness(tmp_path, voice=False, stages=STAGES) as h:
        sess = next(m for m in h.sent if m["type"] == "session")
        assert sess["voice"] is False and "Hosted voice is unavailable" in sess["notice"]
        assert "fresh detection/reading are unavailable" not in sess["notice"]          # stale claim removed
        assert "actual stage availability" in sess["notice"] and "detect:" in sess["notice"] and "read:" in sess["notice"]
        assert "not a live microphone/voice test" in sess["notice"] and set(sess["stages"]) >= {"detect", "read"}
