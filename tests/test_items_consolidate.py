"""Synthetic checks for pipeline.items.consolidate_item_instances (rules, veto, purity). They prove mechanics only, not accuracy on real footage."""
import copy

import numpy as np

from pipeline import items as I

EYE = np.eye(3)


def _obs(oid, frame, box, conf=0.5):
    return {"observation_id": oid, "frame_id": frame, "bbox_xyxy": box, "confidence": conf, "kind": "non_book"}


def _inst(iid, category, oids):
    return {"instance_id": iid, "category": category, "observation_ids": oids}


def _world(spec):
    obs, inst = {}, []
    for iid, cat, items in spec:
        ids = []
        for oid, fr, box in items:
            obs[oid] = _obs(oid, fr, box); ids.append(oid)
        inst.append(_inst(iid, cat, ids))
    return obs, inst


def test_vase_and_pot_boxes_of_one_object_in_the_same_frames_merge():
    obs, inst = _world([
        ("it-001", "vase", [(f"f{k}-a", f"f{k}", [100, 100, 200, 220]) for k in range(3)]),
        ("it-002", "pot", [(f"f{k}-b", f"f{k}", [102, 98, 199, 221]) for k in range(3)]),
    ])
    groups, links = I.consolidate_item_instances(inst, obs)
    assert groups == [["it-001", "it-002"]] and links[0]["rule"] == "same_frame_colocated"


def test_two_side_by_side_pots_never_merge_even_through_a_blurry_weak_group():
    # two distinct pots co-occur in frames, never overlapping; a weak one-sighting group (later frame) sits between them in the plane
    obs, inst = _world([
        ("it-001", "pot", [(f"f{k}-a", f"f{k}", [100, 100, 200, 200]) for k in range(4)]),
        ("it-002", "pot", [(f"f{k}-b", f"f{k}", [260, 100, 360, 200]) for k in range(4)]),
        ("it-003", "pot", [("f9-c", "f9", [150, 100, 310, 200])]),
    ])
    T = {f"f{k}": EYE for k in list(range(4)) + [9]}
    # it-003 overlaps both anchors' plane boxes; with the veto at most one of the two merges happens and it-001 / it-002 stay apart
    groups, _ = I.consolidate_item_instances(inst, obs, T, config={"plane_iou": 0.2, "plane_containment": 0.3})
    assert not any({"it-001", "it-002"} <= set(g) for g in groups)
    # mutation guard: without the veto the weak group bridges the two pots
    plain = copy.deepcopy(I.CONSOLIDATE_DEFAULTS)
    groups2, _ = I.consolidate_item_instances(inst, obs, T, config={"plane_iou": 0.2, "plane_containment": 0.3, "distinct_max_containment": -1.0})
    assert any({"it-001", "it-002"} <= set(g) for g in groups2)
    assert plain == I.CONSOLIDATE_DEFAULTS


def test_plane_rule_needs_an_anchor_and_transforms():
    obs, inst = _world([
        ("it-001", "frame", [(f"f{k}-a", f"f{k}", [100, 100, 200, 200]) for k in range(4)]),
        ("it-002", "frame", [("f20-b", "f20", [105, 102, 203, 199])]),   # re-detected later at the same plane position
    ])
    T = {f"f{k}": EYE for k in list(range(4)) + [20]}
    groups, links = I.consolidate_item_instances(inst, obs, T)
    assert groups == [["it-001", "it-002"]] and links[0]["rule"] == "same_plane_position"
    assert I.consolidate_item_instances(inst, obs)[0] == [["it-001"], ["it-002"]]                      # no transforms, no rule 2
    weak = [dict(inst[0], observation_ids=inst[0]["observation_ids"][:2]), inst[1]]
    assert I.consolidate_item_instances(weak, obs, T)[0] == [["it-001"], ["it-002"]]                    # no anchor with >= 3 sightings


def test_categories_outside_a_family_and_unrelated_families_do_not_merge_and_inputs_are_untouched():
    obs, inst = _world([
        ("it-001", "frame", [(f"f{k}-a", f"f{k}", [100, 100, 200, 220]) for k in range(3)]),
        ("it-002", "pot", [(f"f{k}-b", f"f{k}", [100, 100, 200, 220]) for k in range(3)]),          # identical boxes but a print is not a pot
        ("it-003", "sofa", [(f"f{k}-c", f"f{k}", [100, 100, 200, 220]) for k in range(3)]),
    ])
    before = copy.deepcopy((obs, inst))
    groups, links = I.consolidate_item_instances(inst, obs)
    assert groups == [["it-001"], ["it-002"], ["it-003"]] and links == []
    assert (obs, inst) == before
