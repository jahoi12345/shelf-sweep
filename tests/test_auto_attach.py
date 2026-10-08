"""pipeline.auto_attach: an automatic measurement is patched into a finished packet without breaking packet verification (synthetic)."""
import copy
import json

import numpy as np
import pytest

from pipeline import auto_attach as AA
from pipeline import auto_scale as AS
from pipeline import contracts as C
from pipeline import packet as P
from tests import test_measurement as TM
from tests.test_auto_room import D, HT, W, X0, Y0, face, room_cloud
from tests.test_auto_scale import FakeCatalogue, SCALE as TRUE_SCALE, edition, recon as spine_recon

N = 14
ROT = {"shelf": [[1, 0, 0], [0, -1, 0], [0, 0, -1]], "opp": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], "left": [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
       "right": [[0, 0, 1], [0, -1, 0], [1, 0, 0]], "floor": [[1, 0, 0], [0, 0, 1], [0, -1, 0]], "ceil": [[1, 0, 0], [0, 0, -1], [0, 1, 0]]}


def make_run(tmp_path, ceiling=True):
    """A finished synthetic run: packet WITHOUT any measurement (as built at the end of the sweep) + its manifest/frames."""
    ob = [TM.obs(f"f00000-o{i:03d}", TM.book_m(0.2 + 0.2 * i, 1.2, 0.24, 0.03, 0)) for i in range(N)]
    books = [{"instance_id": f"bk-{i + 1:03d}", "observation_ids": [o["observation_id"]], "frame_ref": "f00000", "bbox_xyxy": [max(5, min(1915, v)) for v in o["bbox_xyxy"]],
              "kind": "book", "confidence": 0.9, "uncertain_association": False, "identity": {"decision": "no_match"}} for i, o in enumerate(ob)]
    invd = {"data": {"books": books, "items": [], "registrations": [], "counts": {}}, "fixture": False}
    frames = ["f00000"] + [f"c{k}" for k in ROT]
    man = TM.manifest(frames)
    run = tmp_path / "run"
    (run / "frames").mkdir(parents=True)
    for f in frames:
        (run / "frames" / f"{f}.jpg").write_bytes(b"x")
    C.write_json(run / C.ARTIFACTS["frames"], man)
    data = P.build_data(invd, man, {"run_id": "run"})
    env = C.envelope("run", "packet", data, status="partial")
    assert P.verify_packet(env, man, run) == []
    C.write_json(run / C.ARTIFACTS["packet"], env)
    # the automatic measurement: books of 22 cm (+/- a few cm) seen by synthetic cameras, plus a box-room cloud
    rc = spine_recon({f"bk-{i + 1:03d}": (0.05 * i - 0.5, 0.0, (20.0 + (i % 5)) / 100.0, 3.0 / 100.0) for i in range(N)}, cam_z=2.0)      # 1 unit = 1 m
    cams = [{"frame_id": f"c{k}", "center": [0.0 + 0.2 * j, 0.2, 2.0], "rotation_world_to_cam": r, "focal_px": 1000.0, "width_px": 1920, "height_px": 1080} for j, (k, r) in enumerate(ROT.items())]
    rc["cameras"] = rc["cameras"] + cams
    pts = room_cloud()
    if not ceiling:
        pts = room_cloud(ceiling=False)
    rc["points"] = pts.tolist()
    # the room cameras must not become the spine sightings' cameras: sightings only reference f0000x frames of the base recon
    return run, man, rc


def run_auto(tmp_path, rc, identified=None, catalogue=None):
    d = AS.measure_data(rc, identified or [], catalogue)
    out = tmp_path / "run_measure_v1"
    out.mkdir(exist_ok=True)
    C.write_json(out / "auto_measure.json", C.envelope("run_measure_v1", "auto_measure", d, status="partial"))
    return out / "auto_measure.json", d


def test_attach_fills_spines_shelf_run_and_box_room_and_the_packet_still_verifies(tmp_path):
    run, man, rc = make_run(tmp_path)
    path, d = run_auto(tmp_path, rc)
    assert d["room"]["status"] == "estimated" and d["status"] == "prior_only"
    res = AA.attach(run, path, render=False)
    assert res["attached"] is True and res["room_status"] == "estimated", res
    env = C.read_json(run / C.ARTIFACTS["packet"])
    p = env["data"]
    assert P.verify_packet(env, man, run) == []
    measured = [b for b in p["books"] if b["spine_height_cm"] is not None]
    assert len(measured) == N and all(b["spine_thickness_cm"] and b["measurement"]["status"] == "measured" and b["measurement"]["calibration_ids"] == [AA.CAL_ID] for b in measured)
    assert all("PRIOR" in b["measurement"]["provenance_label"] and b["measurement"]["per_observation"] for b in measured)
    assert p["room"]["status"] == "estimated" and p["room"]["height_m"] == pytest.approx(HT, abs=0.1) and "auto_sfm_planes" in p["room"]["height_provenance"]["kind"]
    assert p["room"]["floor_area_m2"] == pytest.approx(W * D, rel=0.04) and p["room"]["wall_area_m2"] == pytest.approx(2 * (W + D) * HT, rel=0.04)
    assert p["totals"]["shelf_run_m"] and p["totals"]["shelf_run_m"] > 0 and p["totals"]["spine_dimensions_measured"] == N
    codes = {q["code"] for q in P.flat_review(p["review_queue"])}
    assert "spine_scale_from_prior" in codes and "room_not_measured" not in codes and "no_metric_scale" not in codes
    assert p["measurement_evidence"]["auto"]["status"] == "prior_only" and p["measurement_evidence"]["auto"]["scale"]["not_a_measurement"] is True
    assert (run / AA.BACKUP).is_file()


def test_attach_is_idempotent_and_starts_from_the_original_packet(tmp_path):
    run, man, rc = make_run(tmp_path)
    path, _ = run_auto(tmp_path, rc)
    AA.attach(run, path, render=False)
    first = C.read_json(run / C.ARTIFACTS["packet"])["data"]["books"][0]["spine_height_cm"]
    AA.attach(run, path, render=False)
    p = C.read_json(run / C.ARTIFACTS["packet"])["data"]
    assert p["books"][0]["spine_height_cm"] == first and len([q for q in P.flat_review(p["review_queue"]) if q["code"] == "spine_dimensions_partial"]) == 1
    assert p["provisional_notice"].count("automatic measurement") == 1


def test_missing_ceiling_leaves_the_packet_room_abstained_with_the_reasons_and_what_to_film(tmp_path):
    run, man, rc = make_run(tmp_path, ceiling=False)
    path, d = run_auto(tmp_path, rc)
    assert d["room"]["status"] in ("partial", "abstained")
    res = AA.attach(run, path, render=False)
    assert res["attached"] is True and res["room_status"] == "abstained"
    env = C.read_json(run / C.ARTIFACTS["packet"])
    p = env["data"]
    assert P.verify_packet(env, man, run) == [] and p["room"]["floor_area_m2"] is None and p["room"]["height_m"] is None
    assert any(x.startswith("ceiling_not_found") for x in p["room"]["abstention_reasons"])
    q = next(x for x in P.flat_review(p["review_queue"]) if x["code"] == "room_not_measured")
    assert "Film:" in q["reason"] and "ceiling" in q["reason"].lower() and "how tall" not in q["reason"].lower()
    assert p["measurement_evidence"]["auto"]["room"]["dimensions"]["depth_m"]["value"] == pytest.approx(D, abs=0.05)      # partial values stay in the evidence


def test_scale_a_flows_into_the_packet_label(tmp_path):
    run, man, rc = make_run(tmp_path)
    heights = {f"bk-{i + 1:03d}": 20.0 + (i % 5) for i in range(N)}
    works, ident = {}, []
    for i, (gid, h) in enumerate(list(heights.items())[:6]):
        w = f"/works/OL{i + 1}W"
        works[w] = [edition(f"/books/OL{i + 1}M", f"{i + 1:09d}", f"{h:.1f} x {h * 0.65:.1f} x 2 centimeters")]
        ident.append({"group_id": gid, "work": w, "title": f"T{i}", "visible_text": "THE OLD MAN AND THE SEA", "language_code": "en"})
    path, d = run_auto(tmp_path, rc, ident, FakeCatalogue(works))
    assert d["status"] == "ok" and d["scale"]["source"] == "catalogue_physical_dimensions"
    assert AA.attach(run, path, render=False)["attached"] is True
    p = C.read_json(run / C.ARTIFACTS["packet"])["data"]
    b = next(x for x in p["books"] if x["spine_height_cm"] is not None)
    assert "catalogue physical dimensions of 6 identified books" in b["measurement"]["provenance_label"]
    assert "spine_scale_from_prior" not in {q["code"] for q in P.flat_review(p["review_queue"])}
    assert b["spine_height_cm"] == pytest.approx(heights[b["id"]], abs=0.2)


def test_a_volunteered_valid_calibration_keeps_the_legacy_scale(tmp_path):
    run, man, rc = make_run(tmp_path)
    path, _ = run_auto(tmp_path, rc)
    env = C.read_json(run / C.ARTIFACTS["packet"])
    env["data"].setdefault("measurement_evidence", {})["scale"] = {"summary": {"valid_calibrations": 1}}
    C.write_json(run / C.ARTIFACTS["packet"], env)
    r = AA.attach(run, path, render=False)
    assert r["attached"] is False and "volunteered" in r["reason"]


def test_a_packet_that_would_fail_verification_is_never_written(tmp_path, monkeypatch):
    run, man, rc = make_run(tmp_path)
    path, _ = run_auto(tmp_path, rc)
    before = (run / C.ARTIFACTS["packet"]).read_text()
    monkeypatch.setattr(P, "verify_packet", lambda *a, **k: ["synthetic defect"])
    r = AA.attach(run, path, render=False)
    assert r["attached"] is False and r["problems"] == ["synthetic defect"] and (run / C.ARTIFACTS["packet"]).read_text() == before


def test_a_height_with_a_withheld_thickness_is_allowed_only_with_its_reason(tmp_path):
    run, man, rc = make_run(tmp_path)
    for m in rc["markers"][:3]:                                   # three books whose spines are far too narrow in pixels for a thickness
        for s in m["sightings"]:
            xs = [q[0] for q in s["polygon_xy"]]
            c = sum(xs) / 4
            s["polygon_xy"] = [[c + (q[0] - c) * 0.1, q[1]] for q in s["polygon_xy"]]
    path, d = run_auto(tmp_path, rc)
    assert sum(1 for s in d["spines"] if s["status"] == "measured" and s["spine_thickness_cm"] is None) >= 3
    assert AA.attach(run, path, render=False)["attached"] is True
    env = C.read_json(run / C.ARTIFACTS["packet"])
    assert P.verify_packet(env, man, run) == []
    b = next(x for x in env["data"]["books"] if x["spine_height_cm"] and x["spine_thickness_cm"] is None)
    assert b["measurement"]["thickness_reason"] and b["measurement"]["spine_thickness_interval_cm"] is None
    del b["measurement"]["thickness_reason"]                                                       # without its reason the packet contract still refuses it
    assert any("reported together" in x for x in P.verify_packet(env, man, run))


def _identify(run, man, line_id="bk-001", title="The Old Man and the Sea"):
    """What pipeline.identity_attach does to one line after the measurement: an accepted catalogue identity, totals recomputed."""
    env = C.read_json(run / C.ARTIFACTS["packet"])
    p = env["data"]
    b = next(x for x in p["books"] if x["id"] == line_id)
    b.update(status="identified", title=title, identity_source="catalogue_match", identity_source_url="https://openlibrary.org/works/OL1W", identity_decision="accepted",
             catalogue="open_library")
    fresh = P.compute_totals(p["books"], p["items"], {}, len(p.get("excluded_by_user") or []), p.get("pricing") is not None)
    for k in ("books_identified", "books_unidentified"):
        p["totals"][k] = fresh[k]
    assert P.verify_packet(env, man, run) == []
    C.write_json(run / C.ARTIFACTS["packet"], env)


def _save_measure(tmp_path, name, d):
    out = tmp_path / name
    out.mkdir()
    C.write_json(out / "auto_measure.json", C.envelope(name, "auto_measure", d, status="partial"))
    return out / "auto_measure.json"


def test_a_newer_measurement_patches_the_current_packet_and_keeps_the_layers_attached_since(tmp_path):
    """Re-attaching (a newer measurement) no longer restarts from the pre-measurement backup, which dropped the titles/prices attached after it."""
    run, man, rc = make_run(tmp_path)
    path, _ = run_auto(tmp_path, rc)
    assert AA.attach(run, path, render=False)["attached"] is True
    _identify(run, man)
    _, _, rc2 = make_run(tmp_path / "b", ceiling=False)
    path2 = _save_measure(tmp_path, "run_measure_v2", AS.measure_data(rc2, [], None))
    res = AA.attach(run, path2, render=False)
    assert res["attached"] is True and res["room_status"] == "abstained", res
    env = C.read_json(run / C.ARTIFACTS["packet"])
    p = env["data"]
    assert P.verify_packet(env, man, run) == []
    b = p["books"][0]
    assert b["status"] == "identified" and b["title"] == "The Old Man and the Sea" and p["totals"]["books_identified"] == 1      # the later layer survived
    assert p["measurement_evidence"]["auto"]["source_measure_dir"] == "run_measure_v2"
    pristine = C.read_json(run / AA.BACKUP)["data"]
    own = ("status", "abstention_reasons", "coverage_missing", "scale_method")
    assert p["room"]["floor_area_m2"] is None and p["room"]["height_m"] is None                    # the estimated room of v1 does not leak into the abstained room of v2
    assert {k: v for k, v in p["room"].items() if k not in own} == {k: v for k, v in pristine["room"].items() if k not in own}
    assert p["measurement_evidence"].get("room") == pristine.get("measurement_evidence", {}).get("room")
    codes = [q["code"] for q in P.flat_review(p["review_queue"])]
    assert codes.count("room_not_measured") == 1 and codes.count("spine_scale_from_prior") == 1 and codes.count("spine_dimensions_partial") == 1
    assert p["provisional_notice"].count("automatic measurement") == 1
    before = (run / C.ARTIFACTS["packet"]).read_bytes()
    AA.attach(run, path2, render=False)
    assert (run / C.ARTIFACTS["packet"]).read_bytes() == before                                     # idempotent on the current packet


def test_a_line_that_abstains_in_the_newer_measurement_loses_the_older_size_and_merged_members_are_patched(tmp_path):
    run, man, rc = make_run(tmp_path)
    path, d = run_auto(tmp_path, rc)
    assert AA.attach(run, path, render=False)["attached"] is True
    env = C.read_json(run / C.ARTIFACTS["packet"])
    p = env["data"]
    p["books"][1]["merged_members"] = [copy.deepcopy(p["books"][1]), copy.deepcopy(p["books"][2])]
    for m in p["books"][1]["merged_members"]:
        m["spine_height_cm"] = 99.0                                                               # a stale size an older measurement left in the member copies
    C.write_json(run / C.ARTIFACTS["packet"], env)
    d2 = copy.deepcopy(d)
    for s in d2["spines"]:
        if s["group_id"] == "bk-001":
            s.update(status="abstained", reason="not_in_reconstruction")
    assert AA.attach(run, _save_measure(tmp_path, "run_measure_v2", d2), render=False)["attached"] is True
    p = C.read_json(run / C.ARTIFACTS["packet"])["data"]
    assert p["books"][0]["spine_height_cm"] is None and p["books"][0]["measurement"]["status"] == "abstained"
    rows = {s["group_id"]: s for s in d2["spines"]}
    assert [m["spine_height_cm"] for m in p["books"][1]["merged_members"]] == [rows["bk-002"]["spine_height_cm"], rows["bk-003"]["spine_height_cm"]]
