"""Item over-trigger: consolidate_items_v2 rules and vetoes, context-only categories, min sightings, category floors, books untouched.

Synthetic mechanics only; the accuracy on the live Yugo sweeps was hand-checked separately.
"""
import json

import cv2
import numpy as np

from pipeline import items as I


def _o(oid, fid, box, cat="plant", conf=0.6, kind="non_book"):
    return {"observation_id": oid, "frame_id": fid, "bbox_xyxy": box, "category": cat, "confidence": conf, "kind": kind}


def _world(spec):
    obs, inst = {}, []
    for iid, cat, sightings in spec:
        ids = []
        for oid, fid, box in sightings:
            obs[oid] = _o(oid, fid, box, cat)
            ids.append(oid)
        inst.append({"instance_id": iid, "category": cat, "observation_ids": ids})
    return obs, inst


TS = {f"f{k}": float(k) for k in range(40)}


def test_leaves_above_their_pot_are_one_object_but_side_by_side_pots_are_two():
    obs, inst = _world([
        ("it-001", "plant", [(f"a{k}", f"f{k}", [100, 0, 200, 100]) for k in range(3)]),    # leaves
        ("it-002", "vase", [(f"b{k}", f"f{k}", [110, 95, 190, 200]) for k in range(3)]),    # pot right under them
        ("it-003", "vase", [(f"c{k}", f"f{k}", [400, 95, 480, 200]) for k in range(3)]),    # another pot beside, never overlapping
    ])
    groups, links = I.consolidate_items_v2(inst, obs, TS)
    assert groups == [["it-001", "it-002"], ["it-003"]]
    assert links[0]["rule"] == "same_frame_colocated"


def test_appearance_rules_need_no_shared_geometry_and_respect_the_cooccurrence_veto():
    rng = np.random.default_rng(0)
    green = np.zeros((300, 600, 3), np.uint8)
    green[:, :300] = (40, 160, 40)
    green[:, 300:] = (200, 60, 200)
    green = (green + rng.integers(0, 30, green.shape)).astype(np.uint8)
    load = lambda fid: green
    # same green object seen at t=1 and t=2 in different tracked segments -> merged; the pink one co-occurs apart from it -> never joined
    obs, inst = _world([
        ("it-001", "cup", [("a", "f1", [20, 20, 280, 280])]),
        ("it-002", "cup", [("b", "f2", [30, 25, 270, 285]), ("b2", "f2", [30, 25, 270, 285])]),
        ("it-003", "cup", [("c", "f2", [320, 20, 580, 280])]),
    ])
    tr = {f: np.eye(3) for f in ("f1", "f2")}
    groups, links = I.consolidate_items_v2(inst, obs, TS, load, tr, {"f1": 0, "f2": 1}, cache_key="t1")
    assert ["it-001", "it-002"] in groups and ["it-003"] in groups
    assert {l["rule"] for l in links} <= {"near_in_time_appearance", "time_appearance", "sift_inliers"}
    # in ONE tracked segment the plane decides: far apart on the plane -> no appearance merge
    tr2 = {"f1": np.eye(3), "f2": np.array([[1, 0, 5000], [0, 1, 0], [0, 0, 1]], float)}
    groups, _ = I.consolidate_items_v2(inst, obs, TS, load, tr2, {"f1": 0, "f2": 0}, cache_key="t1")
    assert ["it-001"] in groups


def test_direct_registration_merges_or_vetoes_same_family_groups():
    obs, inst = _world([
        ("it-001", "frame", [("a", "f1", [100, 100, 200, 200])]),
        ("it-002", "frame", [("b", "f2", [150, 100, 250, 200])]),      # same object after a 50 px pan
        ("it-003", "frame", [("c", "f2", [600, 100, 700, 200])]),      # a different frame further right
    ])
    pan = np.array([[1, 0, 50], [0, 1, 0], [0, 0, 1]], float)
    groups, links = I.consolidate_items_v2(inst, obs, TS, register=lambda a, b: pan if (a, b) == ("f1", "f2") else None)
    assert groups == [["it-001", "it-002"], ["it-003"]]
    assert links[0]["rule"] == "registered_position"
    trace = []
    I.consolidate_items_v2(inst, obs, TS, register=lambda a, b: np.eye(3) * 1.0 if a == "f1" else None, trace=trace)
    assert any(t[2] == "distinct_registered" for t in trace)       # identity: it-001 at x=100 vs it-003 at x=600 -> apart


def test_context_and_demoted_book_groups_are_never_consolidated():
    obs, inst = _world([
        ("it-001", "desk", [("a", "f1", [0, 0, 100, 100])]),
        ("it-002", "desk", [("b", "f1", [0, 0, 100, 100])]),
    ])
    inst.append({"instance_id": "it-003", "category": "book_or_isolated_object", "observation_ids": ["c"], "non_book_demotion": {"rule": "no_neighbour"}})
    obs["c"] = _o("c", "f1", [0, 0, 100, 100], "book", kind="book")
    groups, links = I.consolidate_items_v2(inst, obs, TS)
    assert groups == [["it-001"], ["it-002"], ["it-003"]] and links == []
    assert I.item_family("book_or_plant") == "potted" and I.item_family("cushion") == "cushion"


def _write_run(tmp_path, obs):
    from pipeline import contracts as C
    frames = {}
    for i, fid in enumerate(sorted({o["frame_id"] for o in obs})):
        cv2.imwrite(str(tmp_path / f"{fid}.png"), np.zeros((480, 1400, 3), np.uint8))
        frames[fid] = {"frame_id": fid, "timestamp_s": float(i), "width_px": 1400, "height_px": 480, "image_path": f"{fid}.png", "quality": {},
                       "selected": True, "selection_reason": "SYNTHETIC"}
    (tmp_path / "manifest.json").write_text(json.dumps(C.envelope("overtrigger", "frames", {"frames": list(frames.values())}, fixture=True)))
    (tmp_path / "candidates.json").write_text(json.dumps(C.envelope("overtrigger", "detect", {"observations": obs}, fixture=True)))


def test_merge_drops_structure_and_weak_lines_from_items_but_keeps_them_and_books_unchanged(tmp_path):
    from pipeline import dedup
    spine = lambda oid, fid, x: {"observation_id": oid, "frame_id": fid, "kind": "book", "category": "book", "bbox_xyxy": [x, 100, x + 30, 300],
                                 "polygon_xy": [[x, 100], [x + 30, 100], [x + 30, 300], [x, 300]], "confidence": .8, "orientation": "upright"}
    obs = [spine(f"f0000{f}-o00{i}", f"f0000{f}", 100 + 34 * i) for f in range(2) for i in range(4)]
    obs += [_o("f00000-o010", "f00000", [600, 100, 700, 300], "plant"), _o("f00001-o010", "f00001", [600, 100, 700, 300], "plant"),   # 2 sightings
            _o("f00000-o011", "f00000", [800, 100, 900, 300], "vase"),                                                            # 1 sighting
            _o("f00000-o012", "f00000", [0, 400, 1400, 480], "desk"), _o("f00001-o012", "f00001", [0, 400, 1400, 480], "desk"),   # shelf board
            _o("f00000-o013", "f00000", [1000, 100, 1100, 200], "box", conf=.31), _o("f00001-o013", "f00001", [1000, 100, 1100, 200], "box", conf=.33)]
    _write_run(tmp_path, obs)
    cfg = {"association": "feature_registration", "non_book_gate": False}
    off = json.loads(dedup.merge(tmp_path / "candidates.json", tmp_path / "manifest.json", tmp_path,
                                 {**cfg, "consolidate_items": False, "item_dedup": {"min_sightings": 1, "min_confidence": {}}, "output_name": "off.json"}).read_text())
    inv = json.loads(dedup.merge(tmp_path / "candidates.json", tmp_path / "manifest.json", tmp_path, cfg).read_text())
    d = inv["data"]
    assert [b["observation_ids"] for b in d["books"]] == [b["observation_ids"] for b in off["data"]["books"]]
    assert d["counts"]["book_count"] == off["data"]["counts"]["book_count"]
    assert [i["category"] for i in d["items"]] == ["plant"] and d["counts"]["item_count"] == 1
    assert sorted(i["category"] for i in d["item_candidates"]) == ["box", "vase"] and d["counts"]["item_candidates"] == 2
    assert all(i["status"] == "unconfirmed_candidate" for i in d["item_candidates"])
    assert any("category floor" in s for i in d["item_candidates"] for s in i["issues"])
    assert [i["category"] for i in d["context_objects"]] == ["desk"]
    assert inv["metrics"]["item_filter"]["context_only"] == {"desk": 1}


def test_merge_maps_labels_the_review_taxonomy_lacks_and_keeps_the_detector_label(tmp_path):
    """`chair` is not in the item review taxonomy (furniture is `other` there); the line says `other` and keeps `detector_category`."""
    from pipeline import dedup
    obs = [_o("f00000-o010", "f00000", [600, 100, 700, 300], "plant"), _o("f00001-o010", "f00001", [600, 100, 700, 300], "plant"),
           _o("f00000-o011", "f00000", [900, 100, 1300, 470], "chair"), _o("f00001-o011", "f00001", [900, 100, 1300, 470], "chair")]
    _write_run(tmp_path, obs)
    d = json.loads(dedup.merge(tmp_path / "candidates.json", tmp_path / "manifest.json", tmp_path,
                               {"association": "feature_registration", "non_book_gate": False}).read_text())["data"]
    assert sorted((i["category"], i.get("detector_category")) for i in d["items"]) == [("other", "chair"), ("plant", None)]
