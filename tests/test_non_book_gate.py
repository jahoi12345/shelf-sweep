"""Non-book robustness: scene gate, neighbour rule, desk/office conflicts, packet review entry.

Live sweeps start while the user aims the camera; the spine model then fires on laptop key rows, a face or wall-sign text. Those groups must become
`ambiguous` (kept, reviewed, never in book_count) while real books seen in a shelf row stay books.
"""
import json

import cv2
import numpy as np

from pipeline.items import non_book_demotion, spine_row_support


def _spine(oid, fid, x, y=100, w=30, h=200, kind="book"):
    return {"observation_id": oid, "frame_id": fid, "kind": kind, "category": "book", "bbox_xyxy": [x, y, x + w, y + h],
            "polygon_xy": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], "confidence": 0.8, "orientation": "upright", "occlusion": "none"}


def _flat(oid, fid, y, x=100, w=300, h=40):
    o = _spine(oid, fid, x, y, w, h)
    o["orientation"] = "flat"
    return o


def test_row_support_counts_side_by_side_parallel_spines_only():
    row = [_spine(f"a-o{i}", "a", 100 + 34 * i) for i in range(4)]
    lone = _spine("a-o9", "a", 1200, w=150, h=170)                     # square-ish, far away
    keys = [_flat("k-o0", "k", 100), _flat("k-o1", "k", 150)]           # two stacked key rows
    s = spine_row_support(row + [lone] + keys)
    assert all(s[o["observation_id"]] == {"row": 4, "frame_max_row": 4, "aspect": round(200 / 30, 2)} for o in row)
    assert s["a-o9"]["row"] == 1 and s["a-o9"]["frame_max_row"] == 4
    assert s["k-o0"]["row"] == 2 and s["k-o0"]["frame_max_row"] == 2
    # crossing (non-parallel) or vertically offset spines are not one row
    tilted = _spine("t-o1", "t", 140)
    tilted["polygon_xy"] = [[140, 100], [340, 130], [335, 160], [135, 130]]
    offset = _spine("t-o2", "t", 175, y=400)
    s = spine_row_support([_spine("t-o0", "t", 100), tilted, offset])
    assert s["t-o0"]["row"] == 1


def test_square_object_beside_a_spine_row_is_not_in_the_row_but_the_frame_stays_a_shelf_view():
    """The across gap is measured in widths of the NARROWER box. With the wider box a 400 px vase was "in a row" with every spine within 1200 px
    (YugoTest11 yellow vase, flower box, pop-up card). The scene gate (frame_max_row) keeps the wider box."""
    row = [_spine(f"a-o{i}", "a", 100 + 34 * i) for i in range(4)]
    thick = _spine("a-o8", "a", 236, w=60)                              # a real wider spine right next to the row: still a neighbour
    vase = _spine("a-o9", "a", 300, y=80, w=400, h=420)                 # touching that spine, square
    s = spine_row_support(row + [thick, vase])
    assert s["a-o9"]["row"] == 1 and s["a-o9"]["frame_max_row"] == 6
    assert s["a-o8"]["row"] == 5 and all(s[o["observation_id"]]["row"] == 5 for o in row)
    assert non_book_demotion(["a-o9"], s)["rule"] == "no_neighbour"
    assert spine_row_support(row + [thick, vase], {"row_gap_box": "wider"})["a-o9"]["row"] == 6   # the earlier wider-box rule, kept switchable
    # a thin and a 3.5x wider spine 120 px apart: not neighbours under the narrower box, but the scene gate still sees a row of 2
    sparse = [_spine("b-o0", "b", 100, w=30), _spine("b-o1", "b", 175, w=105)]
    sb = spine_row_support(sparse)
    assert sb["b-o0"]["row"] == 1 and sb["b-o0"]["frame_max_row"] == 2


def test_review_taxonomy_maps_only_labels_it_lacks_and_keeps_the_detector_label():
    from pipeline.items import ITEM_TAXONOMY
    assert ITEM_TAXONOMY["chair"] == "other" and ITEM_TAXONOMY["framed_sign"] == "sign" and ITEM_TAXONOMY["decorative_letters"] == "ornament"
    taxonomy = set("plant pot vase frame cup box basket lamp ornament sign record cushion sculpture shelf_part book other".split())
    assert not set(ITEM_TAXONOMY) & taxonomy and set(ITEM_TAXONOMY.values()) <= taxonomy   # never re-labels pot/vase/cup/...


def test_demotion_needs_every_sighting_to_fail():
    s = {"g1": {"row": 1, "frame_max_row": 2, "aspect": 6.0}, "g2": {"row": 2, "frame_max_row": 2, "aspect": 6.0},
         "shelf_thin": {"row": 1, "frame_max_row": 9, "aspect": 6.0}, "shelf_square": {"row": 1, "frame_max_row": 9, "aspect": 1.2},
         "in_row": {"row": 5, "frame_max_row": 9, "aspect": 6.0}}
    d = non_book_demotion(["g1", "g2"], s)
    assert d["rule"] == "scene_gate" and d["sightings"] == 2 and "not a shelf view" in d["reason"]
    assert non_book_demotion(["g1", "in_row"], s) is None                 # one proper shelf sighting keeps it a book
    assert non_book_demotion(["shelf_thin"], s) is None                   # a lone real book (thin spine) stays a book
    assert non_book_demotion(["shelf_square"], s)["rule"] == "no_neighbour"   # a lone near-square shape (stool, sign, record) does not
    assert non_book_demotion(["shelf_square", "unknown"], s) is None      # missing evidence never demotes
    assert non_book_demotion(["g1"], s, {"min_row": 2}) is None             # threshold is configurable


def _write_run(tmp_path, obs):
    from pipeline import contracts as C
    frames = {}
    for i, fid in enumerate(sorted({o["frame_id"] for o in obs})):
        cv2.imwrite(str(tmp_path / f"{fid}.png"), np.zeros((480, 1400, 3), np.uint8))
        frames[fid] = {"frame_id": fid, "timestamp_s": float(i), "width_px": 1400, "height_px": 480, "image_path": f"{fid}.png", "quality": {},
                       "selected": True, "selection_reason": "SYNTHETIC"}
    (tmp_path / "manifest.json").write_text(json.dumps(C.envelope("gate", "frames", {"frames": list(frames.values())}, fixture=True)))
    (tmp_path / "candidates.json").write_text(json.dumps(C.envelope("gate", "detect", {"observations": obs}, fixture=True)))


def test_merge_demotes_keyboard_and_lone_square_but_counts_the_shelf_row(tmp_path):
    from pipeline import dedup
    row = [_spine(f"f00000-o00{i}", "f00000", 100 + 34 * i) for i in range(4)]
    lone_square = _spine("f00000-o009", "f00000", 1000, w=150, h=170)
    keys = [_flat("f00001-o000", "f00001", 100), _flat("f00001-o001", "f00001", 150)]
    _write_run(tmp_path, row + [lone_square] + keys)
    cfg = {"association": "feature_registration", "non_book_gate": True}
    # Default: each demoted group here is seen once, so it is a review candidate (still ambiguous, still in count_upper), not an item line
    inv = json.loads(dedup.merge(tmp_path / "candidates.json", tmp_path / "manifest.json", tmp_path, {**cfg, "output_name": "inv_demoted.json"}).read_text())
    c = inv["data"]["counts"]
    assert c["book_count"] == 4 and c["ambiguous_instances"] == 0 and c["ambiguous_demoted_spines"] == 3 and c["count_upper"] == 7
    cands = [i for i in inv["data"]["item_candidates"] if i.get("non_book_demotion")]
    assert len(cands) == 3 and all(i["classification"] == "ambiguous" and i["status"] == "unconfirmed_candidate" for i in cands)
    assert not [i for i in inv["data"]["items"] if i.get("non_book_demotion")]
    cfg = {**cfg, "item_dedup": {"demoted_min_sightings": False}}   # Earlier behaviour: every demoted group is an ambiguous line
    inv = json.loads(dedup.merge(tmp_path / "candidates.json", tmp_path / "manifest.json", tmp_path, cfg).read_text())
    c, g = inv["data"]["counts"], inv["metrics"]["non_book_gate"]
    assert c["book_count"] == 4 and c["ambiguous_instances"] == 3 and c["ambiguous_demoted_spines"] == 3
    assert c["count_upper"] == 7 and c["count_lower"] <= 4
    assert g["frames_gated"] == ["f00001"] and g["observations_gated"] == 2 and g["groups_demoted"] == {"scene_gate": 2, "no_neighbour": 1}
    demoted = [i for i in inv["data"]["items"] if i.get("non_book_demotion")]
    assert len(demoted) == 3 and all(i["kind"] == "ambiguous" and i["classification"] == "ambiguous" for i in demoted)
    assert sorted(i["instance_id"] for i in demoted) == sorted(inv["data"]["ambiguous_instance_ids"])
    assert all(i["non_book_demotion"]["reason"] in i["issues"] for i in demoted)
    # switchable: without the gate every spine shape is a book again
    inv = json.loads(dedup.merge(tmp_path / "candidates.json", tmp_path / "manifest.json", tmp_path,
                                 {**cfg, "non_book_gate": False, "output_name": "inv_off.json"}).read_text())
    assert inv["data"]["counts"]["book_count"] == 7 and inv["metrics"]["non_book_gate"]["enabled"] is False


def test_keyboard_rows_conflict_by_containment_and_room_structure_is_not_an_item():
    from providers.local_detector import CONFLICT_ONLY_CATEGORIES, book_conflicts, fuse_predictions
    key_row = {"bbox_xyxy": [120, 100, 420, 140], "polygon_xy": [[120, 100], [420, 100], [420, 140], [120, 140]], "kind": "book", "category": "book",
               "confidence": .55, "spine_aspect_ratio": 7.5, "evidence_notes": "x", "detection_provenance": {"pass": "native", "source_tile": [0, 0, 1920, 1080]}}
    keyboard = {"bbox_xyxy": [80, 60, 900, 500], "kind": "non_book", "category": "keyboard", "confidence": .6, "evidence_notes": "y"}
    bookcase = {"bbox_xyxy": [80, 60, 900, 500], "kind": "non_book", "category": "shelving", "confidence": .6, "evidence_notes": "y"}
    door = {"bbox_xyxy": [1000, 0, 1400, 1000], "kind": "non_book", "category": "door", "confidence": .6, "evidence_notes": "y"}
    assert book_conflicts(key_row, [keyboard]) == [keyboard]          # IoU is tiny; containment decides
    assert book_conflicts(key_row, [bookcase]) == []                  # books inside a bookcase are the normal case
    vase = dict(keyboard, category="vase")
    assert book_conflicts(key_row, [vase]) == []                      # containment applies to small office objects only
    kept, _ = fuse_predictions([dict(key_row)], [dict(keyboard), dict(door)])
    book = next(o for o in kept if o.get("polygon_xy"))
    assert book["kind"] == "ambiguous" and book["category"] == "book_or_keyboard"
    assert "door" in CONFLICT_ONLY_CATEGORIES and not any(o["category"] == "door" for o in kept)


def test_packet_lists_ambiguous_lines_with_reason_and_keeps_them_out_of_book_count():
    from pipeline.packet import build_review_queue, compute_totals
    reason = "Every sighting (2) is in a frame with fewer than 3 co-aligned spines (not a shelf view)"
    items = [{"id": "it-001", "status": None, "uncertain_association": False, "confidence": .9, "category": "book_or_non_shelf_object",
              "classification": "ambiguous", "non_book_demotion": {"rule": "scene_gate", "reason": reason}, "issues": [reason]},
             {"id": "it-002", "status": None, "uncertain_association": False, "confidence": .9, "category": "keyboard",
              "classification": "non_book", "non_book_demotion": None, "issues": []}]
    q = build_review_queue([], items, {}, {})
    mine = [r for r in q if r["ref_id"] == "it-001"]
    assert [r["code"] for r in mine] == ["ambiguous_not_counted"] and mine[0]["reason"] == reason
    assert not [r for r in q if r["ref_id"] == "it-002"]
    t = compute_totals([], items, {})
    assert t["book_count"] == 0 and t["item_count"] == 2 and t["ambiguous_count"] == 1
