"""Best guess + range under poor capture: count tiers, the singleton-merge pass, packet totals and the voice sentence.
Synthetic geometry and quality only; says nothing about accuracy on real footage."""
from pipeline import cross_view as cv, dedup as D, packet as P
from app import live_tools as LT
from tests.test_cross_view import FULL, _stub_world
from tests.test_measurement import manifest

SHARP = {"sharpness": 200.0, "motion_px_per_s": 100.0, "zoom_rate_per_s": 0.0}     # strict second
MEDIUM = {"sharpness": 70.0, "motion_px_per_s": 100.0, "zoom_rate_per_s": 0.0}     # medium second
BLUR = {"sharpness": 10.0, "motion_px_per_s": 900.0, "zoom_rate_per_s": 0.0}       # bad second


# ---- singleton merge (geometry only) ----

def test_singleton_landing_on_a_group_merges_and_a_neighbour_does_not():
    single = ("s", "f0", 0.0, 500, 0, 540, 530)          # projected 150 px lower: lies on FULL
    obs, frames, T = _stub_world([single, FULL, ("nb", "f0", 0.0, 600, 0, 640, 530)])
    groups, links = cv.singleton_merge_links([["s"], ["full"], ["nb"]], obs, frames, T)
    assert groups == [["full", "s"], ["nb"]]
    assert links[0]["geometry"] == "singleton_tracked_position" and links[0]["iou"] > 0.8


def test_singleton_merge_respects_eligible_segments_held_frames_and_shared_frames():
    single = ("s", "f0", 0.0, 500, 0, 540, 530)
    obs, frames, T = _stub_world([single, FULL])
    assert cv.singleton_merge_links([["s"], ["full"]], obs, frames, T, eligible=set())[1] == []
    assert cv.singleton_merge_links([["s"], ["full"]], obs, frames, T, segments={"f0": 0, "f1": 1})[1] == []
    assert cv.singleton_merge_links([["s"], ["full"]], obs, frames, T, held_frames=["f0"])[1] == []
    obs2, frames2, T2 = _stub_world([single, FULL, ("full0", "f0", 0.0, 900, 0, 940, 530)])
    assert cv.singleton_merge_links([["s"], ["full", "full0"]], obs2, frames2, T2)[1] == []   # the group is already seen in the singleton's frame


# ---- tiers and range ----

def _world():
    frames = {f: {"frame_id": f, "timestamp_s": t, "width_px": 1920, "height_px": 1080, "quality": q}
              for f, t, q in [("f0", 0.1, SHARP), ("f1", 1.1, BLUR), ("f2", 2.1, SHARP), ("f3", 3.1, MEDIUM), ("f4", 4.1, SHARP)]}
    o = lambda oid, f: {"observation_id": oid, "frame_id": f, "kind": "book", "bbox_xyxy": [0, 0, 10, 50]}
    obs = [o("a0", "f0"), o("a2", "f2"), o("c0", "f0"), o("d1", "f1"), o("b3", "f3"), o("b4", "f4"), o("m3", "f3"), o("u4", "f4"), o("x4", "f4"), o("y4", "f4")]
    # a: 2 sightings; c: strict + linked frame; d: bad second; m: medium + linked; u/x/y: f4 has only 1 of 4 sightings linked (unlinked frame)
    groups = [["a0", "a2"], ["c0"], ["d1"], ["b3", "b4"], ["m3"], ["u4"], ["x4"], ["y4"]]
    return groups, obs, frames


def test_tiers_lower_best_upper():
    groups, obs, frames = _world()
    tiers, c = D.count_range(groups, obs, frames)
    assert tiers[("c0",)] == "confident" and tiers[("d1",)] == "unconfirmed" and tiers[("m3",)] == "likely" and tiers[("u4",)] == "unconfirmed"
    assert (c["confident_groups"], c["count_best"], c["count_range"], c["singleton_groups"], c["all_groups"]) == (3, 4, [3, 8], 6, 8)
    held = D.count_range(groups, obs, frames, held_frames=["f0"])[0]
    assert held[("c0",)] == "unconfirmed"                                   # motion of a held frame is unknown: never confident


def test_upper_drops_unconfirmed_singletons_that_the_merge_pass_joins():
    groups, obs, frames = _world()
    seen = []
    c = D.count_range(groups, obs, frames, singleton_merge=lambda gs, ids: seen.append(set(ids)) or [g for g in gs if g != ["d1"]])[1]
    assert seen == [{"d1", "u4", "x4", "y4"}]                              # only unconfirmed singletons may merge
    assert c["count_range"] == [3, 7] and c["singleton_merges_for_upper"] == 1


# ---- packet + voice ----

def _inv(tiers):
    books = [{"instance_id": f"bk-{i:03d}", "observation_ids": [f"f00000-o{i:03d}"], "frame_ref": "f00000", "bbox_xyxy": [100, 100, 140, 400], "kind": "book",
              "confidence": 0.9, "uncertain_association": False, "identity": {"decision": "no_match"}, "count_tier": t} for i, t in enumerate(tiers, 1)]
    counts = {"count_range": [1, 3], "count_best": 2, "confident_groups": 1, "singleton_groups": 2, "unconfirmed_groups": 1, "count_reliability": "low",
              "count_lower": 3, "count_upper": 3}
    return {"data": {"books": books, "items": [], "registrations": [], "counts": counts}, "fixture": False}


def test_packet_book_count_is_the_best_guess_and_unconfirmed_lines_are_kept_apart():
    data = P.build_data(_inv(["confident", "likely", "unconfirmed"]), manifest(), {"run_id": "t"})
    t = data["totals"]
    assert t["book_count"] == len(data["books"]) == 2 and [b["id"] for b in data["unconfirmed_books"]] == ["bk-003"]
    assert t["book_count_range"] == [1, 3] and (t["book_count_lower"], t["book_count_upper"]) == (1, 3)
    assert t["confident_groups"] == 1 and t["singleton_groups"] == 2 and t["count_reliability"] == "low"
    assert data["count"]["best_guess"] == 2 and data["count"]["book_count_range"] == [1, 3]
    assert LT.count_sentence(t) == "about 2 books (range 1 to 3, count reliability low)"


def test_old_inventory_without_tiers_is_unchanged():
    inv = _inv([None, None])
    inv["data"]["counts"] = {}
    t = P.build_data(inv, manifest(), {"run_id": "t"})["totals"]
    assert t["book_count"] == 2 and "book_count_range" not in t and LT.count_sentence(t) == "about 2 books"
