import numpy as np

from app import group_reader as GR


def _obs(i, fid, bbox, poly=None):
    return {"observation_id": i, "frame_id": fid, "bbox_xyxy": bbox, "polygon_xy": poly}


FR = {"f1": {"width_px": 1000, "height_px": 800, "quality": {"sharpness": 100.0}},
      "f2": {"width_px": 1000, "height_px": 800, "quality": {"sharpness": 50.0}}}


def _img(textured=True):
    rng = np.random.default_rng(0)
    return (rng.integers(0, 255, (800, 1000)).astype(np.uint8) if textured else np.full((800, 1000), 200, np.uint8))


def test_plan_picks_best_sightings_and_never_skips_silently():
    obs = {"a": _obs("a", "f1", [100, 100, 130, 400]),          # 30 px wide, sharp
           "b": _obs("b", "f2", [100, 100, 160, 400]),          # 60 px wide, half sharpness -> score 3000 vs 3000 tie -> order stable
           "c": _obs("c", "f1", [0, 100, 40, 400]),             # touches the edge -> down-weighted
           "t": _obs("t", "f1", [500, 100, 510, 400]),          # 10 px: too small
           "blank": _obs("blank", "f1", [600, 100, 640, 400])}
    groups = [{"instance_id": "g1", "observation_ids": ["a", "b", "c"]}, {"instance_id": "g2", "observation_ids": ["t"]},
              {"instance_id": "g3", "observation_ids": ["blank"]}, {"instance_id": "g4", "observation_ids": ["missing"]}]
    flat = _img(False)
    def load(fid):
        g = _img(True)
        g[100:400, 600:640] = 200   # flat region for 'blank'
        return g
    p = GR.plan(groups, obs, FR, load)
    st = {g["instance_id"]: g for g in p["groups"]}
    assert st["g1"]["status"] == "attempt" and len(st["g1"]["sightings"]) == 2
    assert st["g1"]["sightings"][0]["observation_id"] in ("a", "b") and "c" not in [s["observation_id"] for s in st["g1"]["sightings"]]
    assert st["g2"]["status"] == "unattempted_too_small" and "px wide" in st["g2"]["reason"]
    assert st["g3"]["status"] == "unattempted_low_contrast"
    assert st["g4"]["status"] == "unattempted_no_sighting"
    assert sum(p["counts"][k] for k in ("attempt", "unattempted_too_small", "unattempted_low_contrast", "unattempted_no_sighting")) == p["counts"]["groups"] == 4


def test_combine_prefers_longest_confident_and_keeps_all_raw():
    g = {"observation_ids": ["x", "y", "z"]}
    rd = {"x": {"visible_text": "[?]"}, "y": {"visible_text": "PUERTAS CONNIG", "read_confidence": 0.9}, "z": {"title_text": "PUERT", "read_confidence": 0.9}}
    c = GR.combine(g, rd)
    assert c["best"]["observation_id"] == "y" and len(c["readings_considered"]) == 3
    assert GR.combine({"observation_ids": ["x"]}, rd)["best"] is None


def test_revisit_nudge_and_serpentine_rows():
    from app.engine import CaptureMonitor
    m = CaptureMonitor()
    t, out = 0.0, []
    for _ in range(12):                 # pan right 0.3 widths/frame = 3.6 widths
        t += 0.2; out += m.motion(0.3, 0.0, t)
    assert not out
    for _ in range(5):                  # come back 1.5 widths: revisit
        t += 0.2; out += m.motion(-0.3, 0.0, t)
    assert [n["code"] for n in out] == ["revisit"]
    m2, t, out = CaptureMonitor(), 0.0, []
    for _ in range(12):
        t += 0.2; out += m2.motion(0.3, 0.0, t)
    for _ in range(3):                  # step down to the next row (vertical move), then return along it: not a revisit
        t += 0.2; out += m2.motion(0.0, 0.2, t)
    for _ in range(12):
        t += 0.2; out += m2.motion(-0.3, 0.0, t)
    assert out == []


def test_dense_is_default_policy(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBRARY_DETECT_POLICY", raising=False)
    from app.engine import SweepEngine
    e = SweepEngine(tmp_path / "r", config={"provider": "off"})
    assert e.detect_policy == "dense" and e.detect_max_fps == 2.5 and e.detect_skip_blurry is False      # uniform 2.5/s, blurry included


def test_combine_tags_readers_and_unions_tokens():
    g = {"observation_ids": ["x", "y"]}
    c = GR.combine(g, {"x": {"visible_text": "SUM OF ALL"}}, {"y": {"visible_text": "ALL FEARS Clancy", "read_confidence": 0.6}}, ("rapidocr", "florence"))
    assert [r["reader"] for r in c["readings_considered"]] == ["rapidocr", "florence"]
    assert c["union_text"] == "sum of all fears clancy"
    assert GR.token_union(["Hábitos de la gente"]) == "habitos de la gente"


def test_yield_by_width_buckets():
    plan = [{"instance_id": "a", "status": "attempt", "sightings": [{"width_px": 20}]}, {"instance_id": "b", "status": "attempt", "sightings": [{"width_px": 45}]},
            {"instance_id": "c", "status": "unattempted_too_small", "sightings": [{"width_px": 10}]}]
    books = [{"instance_id": "a", "observation_ids": ["x"]}, {"instance_id": "b", "observation_ids": ["y"]}, {"instance_id": "c", "observation_ids": ["z"]}]
    y = GR.yield_by_width(plan, books, {"x": {"visible_text": "ABCDEFGHI"}, "y": {"visible_text": "[?]"}})
    assert y["<26"] == {"groups": 2, "attempted": 1, "with_text": 1, "text_ge8": 1} and y["40-60"]["attempted"] == 1 and y["40-60"]["with_text"] == 0


def test_alternative_count_is_reported_and_skipped_when_over_target(tmp_path, monkeypatch):
    import asyncio
    from app.engine import SweepEngine
    e = SweepEngine(tmp_path / "r", config={"provider": "off"})
    e.ended = 0.0
    assert asyncio.run(e._alternative_count(10.0))["status"] == "skipped"            # no primary inventory yet
    e.inventory = {"data": {"books": [], "counts": {"book_count": 0}}, "metrics": {}}
    out = asyncio.run(e._alternative_count(e.packet_target_s - 5))
    assert out["status"] == "skipped" and "past LIBRARY_PACKET_TARGET_S" in out["reason"] and out["method"] == "track_v2"
    assert "truth is most likely between them" in out["note"]


def test_remerge_is_skipped_only_when_no_accepted_resolution_changed(tmp_path):
    """Association never reads readings/resolutions, so the post-reading re-merge is replaced by an in-place literal refresh unless an accepted
    catalogue resolution changed (which can change identities, demotions and counts)."""
    from app.engine import SweepEngine
    e = SweepEngine(tmp_path / "r", config={"provider": "off"})
    e.observations = [{"observation_id": "o1"}, {"observation_id": "o2"}]
    e.merged_observation_ids = {"o1", "o2"}
    e.inventory = {"data": {"books": [{"instance_id": "bk1", "observation_ids": ["o1", "o2"], "identity": {"decision": "no_match", "visible_text": None}}]}}
    e._merged_accept_map = e._accepted_map()
    e.readings.append({"observation_id": "o2", "visible_text": "DUNE HERBERT", "read_confidence": 0.9})
    e.resolutions.append({"observation_id": "o2", "decision": "rejected"})
    assert e._remerge_skippable() and e._refresh_literal_identities()
    assert e.inventory["data"]["books"][0]["identity"]["visible_text"] == "DUNE HERBERT"
    unread = "Unreadable/unresolved spine; preserved as unidentified book"
    e.inventory["data"]["books"][0].update(status="unidentified", issues=["blurred", unread])
    e.resolutions.append({"observation_id": "o1", "decision": "accepted", "matching_evidence": ["title"], "source_url": "https://openlibrary.org/works/OL1W", "title": "Dune"})
    assert e._remerge_skippable() is None                                            # acceptance on a group with a non-book sighting: full re-merge
    for o in e.observations:
        o["kind"] = "book"
    assert e._remerge_skippable() and e._refresh_literal_identities()                # acceptance on a pure book group: identity replayed in place
    g = e.inventory["data"]["books"][0]
    assert g["status"] == "identified" and g["identity"]["title"] == "Dune" and g["issues"] == ["blurred"]
    e._merged_accept_map = e._accepted_map()
    e.resolutions.append({"observation_id": "o2", "decision": "accepted", "matching_evidence": ["title"], "source_url": "https://openlibrary.org/works/OL2W", "work": "OL2W", "title": "Emma"})
    assert e._remerge_skippable() and not e._refresh_literal_identities()            # an accepted group would turn ambiguous (could demote it): full re-merge
    e._merged_accept_map = e._accepted_map()
    e.observations.append({"observation_id": "o3"})
    assert e._remerge_skippable() is None                                            # observations changed: full re-merge
    assert SweepEngine(tmp_path / "s", config={"provider": "off", "remerge_skip": "0"})._remerge_skippable() is None


def test_alternative_association_runs_after_the_packet_and_is_patched_in(tmp_path):
    import asyncio
    from app.engine import SweepEngine
    from pipeline import contracts as C
    e = SweepEngine(tmp_path / "r", config={"provider": "off"})
    assert e.alt_after_packet
    pk = C.write_json(tmp_path / "claim_packet.json", {"data": {"count": {"alternative_method": {"status": "pending"}}}})
    e.finalization = {}

    async def fake(spent_s):
        assert spent_s == 0.0                                                         # off the critical path: the packet target no longer skips it
        return {"method": "track_v2", "status": "ok", "book_groups": 7, "elapsed_s": 1.0}
    e._alternative_count = fake
    asyncio.run(e._alt_count_after_packet(pk))
    alt = C.read_json(pk)["data"]["count"]["alternative_method"]
    assert alt["status"] == "ok" and alt["book_groups"] == 7 and alt["ran_after_packet"] is True
    assert e.finalization["alternative_count"]["book_groups"] == 7


def test_room_auto_pass_uses_estimate_and_flags_low_confidence_walls(tmp_path, monkeypatch):
    import asyncio
    from app.engine import SweepEngine
    from pipeline import room_auto as RA
    e = SweepEngine(tmp_path / "r", config={"provider": "off"})
    e.ended = 0.0
    abst = {"status": "abstained", "abstention_reasons": ["no_wall_evidence_supplied"], "dimensions": {}, "areas": {}}
    room, info = asyncio.run(e._room_auto_pass(abst, {"data": {"frames": []}}, e.packet_target_s))
    assert info["status"] == "skipped" and room is abst                                   # over target: abstain cleanly
    fake = {"room": {"status": "estimated", "method": "two_walls", "dimensions": {"length_m": 4.0}, "areas": {}, "abstention_reasons": []},
            "evidence": {"walls": [{"wall_id": "f1_w0", "frame_id": "f1", "confidence": 0.9, "provenance": {"source": "auto_classical"}},
                                   {"wall_id": "f2_w0", "frame_id": "f2", "confidence": 0.3, "provenance": {"source": "auto_classical"}}]},
            "abstention_reasons": [], "capture_guidance": [], "provenance_note": "auto"}
    monkeypatch.setattr(RA, "estimate_from_frames", lambda *a, **k: fake)
    room, info = asyncio.run(e._room_auto_pass(abst, {"data": {"frames": []}}, 0.0))
    assert info["used"] and room["status"] == "estimated" and "auto_classical" in room["method"] and info["low_confidence_walls"] == ["f2_w0"]
    assert room["auto_evidence"]["validation"].startswith("software_only_unvalidated")
    est = {"status": "estimated", "method": "human"}
    room2, info2 = asyncio.run(e._room_auto_pass(est, {"data": {"frames": []}}, 0.0))
    assert room2 is est and not info2["used"]                                             # human estimate never overridden
    fake_abst = {**fake, "room": {"status": "abstained", "abstention_reasons": ["fewer_than_two_walls"]}, "capture_guidance": [RA.CAPTURE_STEPS[0][1]]}
    monkeypatch.setattr(RA, "estimate_from_frames", lambda *a, **k: fake_abst)
    room3, info3 = asyncio.run(e._room_auto_pass(abst, {"data": {"frames": []}}, 0.0))
    assert not info3["used"] and room3["auto_attempt"]["capture_guidance"] and room3["status"] == "abstained"


def test_guidance_uses_room_auto_capture_steps():
    from app.measurement import MeasurementState
    m = MeasurementState()
    g = m.guidance_due(45.0, 3, None, {"evidence_summary": {"walls_valid": 0}, "status": "abstained", "abstention_reasons": [], "dimensions": {}, "areas": {}, "coverage": {"items": []}})
    codes = {x["code"]: x["message"] for x in g}
    assert set(codes) == {"show_room_corner"} and "ceiling" in codes["show_room_corner"].lower()
