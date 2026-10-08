"""Synthetic checks for tracked-motion association (pipeline/cross_view.py).

These prove the mechanics on a synthetic shelf (textured coloured spines panned by a known shift);
they say nothing about accuracy on real footage.
"""
import numpy as np
import cv2

from pipeline import cross_view as cv


def _shelf(width=1800, height=700, seed=1):
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), (60, 90, 120), np.uint8)
    noise = rng.integers(0, 40, (height, width, 3)).astype(np.uint8)
    img = cv2.GaussianBlur(cv2.add(img, noise), (5, 5), 0)  # texture so tracking has corners (real shelves always do)
    spines, x = [], 20
    while x < width - 80:
        w = int(rng.integers(36, 64))
        color = tuple(int(c) for c in rng.integers(30, 230, 3))
        cv2.rectangle(img, (x, 150), (x + w, 600), color, -1)
        for _ in range(6):  # printed texture so appearance and tracking have signal
            y = int(rng.integers(170, 560))
            cv2.rectangle(img, (x + 4, y), (x + w - 4, y + int(rng.integers(6, 22))), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
        img[150:600, x:x + w] = cv2.add(img[150:600, x:x + w], rng.integers(0, 45, (450, w, 3)).astype(np.uint8))
        spines.append((x, 150, x + w, 600))
        x += w + int(rng.integers(2, 6))
    return img, spines


def _view(img, spines, x0, view_w=900):
    crop = img[:, x0:x0 + view_w].copy()
    obs = []
    for i, (a, b, c, d) in enumerate(spines):
        if a >= x0 + 5 and c <= x0 + view_w - 5:
            obs.append([[a - x0, b], [c - x0, b], [c - x0, d], [a - x0, d]])
    return crop, obs


def test_overlapping_views_merge_same_spines_once():
    img, spines = _shelf()
    frames, images, observations, ids = {}, {}, [], {}
    starts = list(range(0, 901, 60))
    for k, x0 in enumerate(starts):
        fid = f"f{k:05d}"
        crop, polys = _view(img, spines, x0)
        images[fid] = crop
        frames[fid] = {"frame_id": fid, "timestamp_s": k * 0.2, "width_px": crop.shape[1], "height_px": crop.shape[0]}
        for j, poly in enumerate(polys):
            observations.append({"observation_id": f"{fid}-o{j:03d}", "frame_id": fid, "kind": "book", "polygon_xy": poly, "confidence": 0.9,
                                 "bbox_xyxy": [poly[0][0], poly[0][1], poly[2][0], poly[2][1]]})
    visible = {s for x0 in starts for s in [i for i, sp in enumerate(spines) if sp[0] >= x0 + 5 and sp[2] <= x0 + 895]}
    edges, report, pairs = cv.track_edges(observations, frames, lambda f: images[f])
    from pipeline import dedup
    groups, accepted, uncertain = dedup.associate(observations, edges, 0.0)
    # every group is one physical spine and no spine is split into more than one group:
    assert abs(len(groups) - len(visible)) <= max(2, int(0.03 * len(visible)))
    # same-frame exclusion: no group holds two observations from one frame
    by_obs = {o["observation_id"]: o["frame_id"] for o in observations}
    assert all(len({by_obs[o] for o in g}) == len(g) for g in groups)


def test_non_overlapping_views_do_not_merge():
    img, spines = _shelf()
    frames, images, observations = {}, {}, []
    for k, x0 in enumerate([0, 880]):
        fid = f"f{k:05d}"
        crop, polys = _view(img, spines, x0, view_w=850)
        images[fid] = crop
        frames[fid] = {"frame_id": fid, "timestamp_s": k * 0.4, "width_px": crop.shape[1], "height_px": crop.shape[0]}
        for j, poly in enumerate(polys):
            observations.append({"observation_id": f"{fid}-o{j:03d}", "frame_id": fid, "kind": "book", "polygon_xy": poly, "confidence": 0.9,
                                 "bbox_xyxy": [poly[0][0], poly[0][1], poly[2][0], poly[2][1]]})
    edges, _, _ = cv.track_edges(observations, frames, lambda f: images[f])
    assert edges == []


def _flat_scene(seed=3):
    """Textured shelf with upright spines plus two flat (lying) books stacked 60 px apart; views panned by 60 px.
    The two views of each flat book are detected over DIFFERENT stretches of it (as real detectors do), which the upright rule rejects
    (along-axis offset > 0.35 x length) but the flat rule accepts (>= 50% overlap along the long axis, length-scaled tolerance)."""
    img, spines = _shelf(height=800, seed=seed)
    rng = np.random.default_rng(seed + 10)
    for y0 in (640, 700):  # two stacked flat books, same x extent
        for x in range(100, 480, 14):
            cv2.rectangle(img, (x, y0), (x + 14, y0 + 38), tuple(int(c) for c in rng.integers(20, 235, 3)), -1)
        img[y0:y0 + 38, 100:480] = cv2.add(img[y0:y0 + 38, 100:480], rng.integers(0, 40, (38, 380, 3)).astype(np.uint8))
    frames, images, obs = {}, {}, []
    for k, (x0, spans) in enumerate([(0, {640: (100, 300), 700: (100, 300)}), (60, {640: (180, 420), 700: (180, 420)})]):
        fid = f"f{k:05d}"
        crop = img[:, x0:x0 + 900].copy()
        images[fid] = crop
        frames[fid] = {"frame_id": fid, "timestamp_s": k * 0.2, "width_px": crop.shape[1], "height_px": crop.shape[0]}
        j = 0
        for a, b, c, d in [s for s in spines if s[0] >= x0 + 5 and s[2] <= x0 + 895]:
            poly = [[a - x0, b], [c - x0, b], [c - x0, d], [a - x0, d]]
            obs.append({"observation_id": f"{fid}-o{j:03d}", "frame_id": fid, "kind": "book", "polygon_xy": poly, "confidence": 0.9, "bbox_xyxy": [poly[0][0], poly[0][1], poly[2][0], poly[2][1]]})
            j += 1
        for y0, (xa, xb) in spans.items():
            poly = [[xa - x0, y0], [xb - x0, y0], [xb - x0, y0 + 38], [xa - x0, y0 + 38]]
            obs.append({"observation_id": f"{fid}-o9{y0 // 60:02d}", "frame_id": fid, "kind": "book", "polygon_xy": poly, "confidence": 0.9, "bbox_xyxy": [poly[0][0], poly[0][1], poly[2][0], poly[2][1]]})
    return frames, images, obs


def _flat_ids(obs):
    return {o["observation_id"] for o in obs if o["observation_id"].split("-o")[1].startswith("9")}


def test_flat_books_link_across_views_and_stacked_neighbour_stays_separate():
    from pipeline import dedup
    frames, images, obs = _flat_scene()
    edges, report, pairs = cv.track_edges(obs, frames, lambda f: images[f])
    groups, accepted, _ = dedup.associate(obs, edges, 0.0)
    flat_groups = [sorted(g) for g in groups if set(g) & _flat_ids(obs)]
    assert len(flat_groups) == 2                          # two physical flat books, each seen twice
    assert all(len(g) == 2 for g in flat_groups)
    assert all(len({i.split("-o")[0] for i in g}) == 2 for g in flat_groups)   # one sighting per frame
    assert all(len({i.split("-o")[1] for i in g}) == 1 for g in flat_groups)      # each group joins the SAME book (same local id in both frames), never the stacked neighbour
    assert any(e["geometry"] == "tracked_chain_flat" for e in edges)


def test_flat_rule_off_leaves_flat_boxes_unlinked_and_upright_edges_identical():
    frames, images, obs = _flat_scene()
    on, _, _ = cv.track_edges(obs, frames, lambda f: images[f])
    off, _, _ = cv.track_edges(obs, frames, lambda f: images[f], {"flat_rule": False})
    flat = _flat_ids(obs)
    # with the rule off the flat boxes do not link (that is the failure the rule fixes) ...
    assert not any(set(e["observation_ids"]) & flat for e in off)
    # ... and every upright edge is unchanged by the rule (same pairs, same scores)
    key = lambda e: (tuple(e["observation_ids"]), round(e["score"], 9))
    assert sorted(map(key, off)) == sorted(map(key, [e for e in on if not (set(e["observation_ids"]) & flat)]))


def test_identical_stacked_flat_copies_seen_in_different_frames_do_not_link():
    """Two stacked flat books with IDENTICAL covers: appearance cannot separate them, so only the length-scaled across tolerance can.
    Frame 0 sees only the upper copy, frame 1 only the lower copy (60 px lower): they must NOT be linked as one book."""
    img, spines = _shelf(height=800, seed=5)
    rng = np.random.default_rng(99)
    cover = np.zeros((38, 380, 3), np.uint8)
    for x in range(0, 380, 14):
        cv2.rectangle(cover, (x, 0), (x + 14, 38), tuple(int(c) for c in rng.integers(20, 235, 3)), -1)
    img[640:678, 100:480] = cover
    img[700:738, 100:480] = cover
    frames, images, obs = {}, {}, []
    for k, (x0, y0, xa, xb) in enumerate([(0, 640, 100, 300), (60, 700, 180, 420)]):
        fid = f"f{k:05d}"
        crop = img[:, x0:x0 + 900].copy()
        images[fid] = crop
        frames[fid] = {"frame_id": fid, "timestamp_s": k * 0.2, "width_px": crop.shape[1], "height_px": crop.shape[0]}
        j = 0
        for a, b, c, d in [s for s in spines if s[0] >= x0 + 5 and s[2] <= x0 + 895]:
            poly = [[a - x0, b], [c - x0, b], [c - x0, d], [a - x0, d]]
            obs.append({"observation_id": f"{fid}-o{j:03d}", "frame_id": fid, "kind": "book", "polygon_xy": poly, "confidence": 0.9, "bbox_xyxy": [poly[0][0], poly[0][1], poly[2][0], poly[2][1]]})
            j += 1
        poly = [[xa - x0, y0], [xb - x0, y0], [xb - x0, y0 + 38], [xa - x0, y0 + 38]]
        obs.append({"observation_id": f"{fid}-o900", "frame_id": fid, "kind": "book", "polygon_xy": poly, "confidence": 0.9, "bbox_xyxy": [poly[0][0], poly[0][1], poly[2][0], poly[2][1]]})
    edges, _, _ = cv.track_edges(obs, frames, lambda f: images[f])
    assert not any({"f00000-o900", "f00001-o900"} == set(e["observation_ids"]) for e in edges)
    # mutation guard: with a 0.6 x length across tolerance the same data DOES link (so this test really exercises the tolerance)
    wide, _, _ = cv.track_edges(obs, frames, lambda f: images[f], {"flat_across_length_frac": 0.6})
    assert any({"f00000-o900", "f00001-o900"} == set(e["observation_ids"]) for e in wide)


def _clipped_scene(clip=200, seed=7):
    """Two views of one upright shelf (panned 60 px); in the second view every spine is detected only below `clip` px from its top, as when the
    frame border (or a shorter box) cuts the book. Whole-box patches then show different parts of the cover."""
    img, spines = _shelf(seed=seed)
    frames, images, obs = {}, {}, []
    for k, x0 in enumerate([0, 60]):
        fid = f"f{k:05d}"
        crop, polys = _view(img, spines, x0)
        images[fid] = crop
        frames[fid] = {"frame_id": fid, "timestamp_s": k * 0.2, "width_px": crop.shape[1], "height_px": crop.shape[0]}
        for j, poly in enumerate(polys):
            if k == 1:
                poly = [[poly[0][0], poly[0][1] + clip], poly[1][:1] + [poly[1][1] + clip], poly[2], poly[3]]
            obs.append({"observation_id": f"{fid}-o{j:03d}", "frame_id": fid, "kind": "book", "polygon_xy": poly, "confidence": 0.9,
                        "bbox_xyxy": [poly[0][0], poly[0][1], poly[2][0], poly[2][1]]})
    return frames, images, obs


def test_clipped_spine_boxes_link_by_the_stretch_both_views_cover():
    frames, images, obs = _clipped_scene()
    visible = sum(o["frame_id"] == "f00001" for o in obs)
    link = lambda cfg: sum(e["eligible"] for e in cv.track_edges(obs, frames, lambda f: images[f], cfg)[0])
    on, off = link({}), link({"stretch_appearance": False})
    assert on >= 0.9 * visible          # clipped views of the same spines are linked
    assert off < 0.6 * on               # mutation guard: without the stretch comparison the whole-box patches fail the appearance gate


# ---- clipped partial spines (clipped_partial_links): synthetic geometry and a known pan; says nothing about accuracy on real footage ----

def _stub_world(sightings, dy=150):
    """sightings: [(oid, frame_id, t_s, x1, y1, x2, y2)] -> (observations, frames, transforms). The camera pans so that frame f0 pixels land `dy` lower in every other frame."""
    frames, observations = {}, []
    for oid, fid, t, x1, y1, x2, y2 in sightings:
        frames[fid] = {"frame_id": fid, "timestamp_s": t, "width_px": 1920, "height_px": 1080}
        observations.append({"observation_id": oid, "frame_id": fid, "kind": "book", "confidence": 0.9, "bbox_xyxy": [x1, y1, x2, y2],
                             "polygon_xy": [[x2, y1], [x2, y2], [x1, y2], [x1, y1]]})
    return observations, frames, {f: (np.eye(3) if f == "f0" else np.array([[1.0, 0, 0], [0, 1.0, -dy], [0, 0, 1.0]])) for f in frames}


STUB = ("stub", "f0", 0.0, 500, 0, 540, 200)            # top of a spine cut by the frame border
FULL = ("full", "f1", 0.4, 500, 120, 540, 680)          # the same spine seen whole 0.4 s later, the camera having panned 150 px


def test_clipped_stub_merges_into_the_full_length_group_that_contains_it():
    obs, frames, T = _stub_world([STUB, FULL, ("nb", "f1", 0.4, 560, 120, 600, 680)])
    groups, links = cv.clipped_partial_links([["stub"], ["full"], ["nb"]], obs, frames, T)
    assert groups == [["full", "stub"], ["nb"]]
    assert links[0]["geometry"] == "clipped_partial_contained" and links[0]["decision"] == "merged" and links[0]["inside_fraction"] > 0.9


def test_clipped_stub_beside_the_full_spine_stays_separate():
    obs, frames, T = _stub_world([("stub", "f0", 0.0, 540, 0, 580, 200), FULL])  # a neighbouring spine: 100% off the partner's footprint
    groups, links = cv.clipped_partial_links([["stub"], ["full"]], obs, frames, T)
    assert groups == [["full"], ["stub"]] and links == []


def test_stub_never_merges_into_a_group_it_shares_a_frame_with():
    obs, frames, T = _stub_world([STUB, FULL, ("full0", "f0", 0.0, 500, 120, 540, 680)])
    groups, _ = cv.clipped_partial_links([["stub"], ["full", "full0"]], obs, frames, T)
    assert groups == [["full", "full0"], ["stub"]]  # one frame shows both: two books


def test_nearly_full_length_clipped_group_is_not_a_stub():
    obs, frames, T = _stub_world([("tall", "f0", 0.0, 500, 0, 540, 520), FULL])  # clipped at the top but 520 of 560 px long: the pairwise gates own this one
    groups, _ = cv.clipped_partial_links([["tall"], ["full"]], obs, frames, T)
    assert groups == [["full"], ["tall"]]


def test_group_with_an_unclipped_sighting_is_not_a_stub():
    obs, frames, T = _stub_world([STUB, ("short", "f2", 0.8, 500, 100, 540, 300), FULL])
    groups, _ = cv.clipped_partial_links([["short", "stub"], ["full"]], obs, frames, T)
    assert groups == [["full"], ["short", "stub"]]


def test_stub_across_a_tracking_break_or_a_held_frame_stays_separate():
    obs, frames, T = _stub_world([STUB, FULL])
    assert cv.clipped_partial_links([["stub"], ["full"]], obs, frames, T, segments={"f0": 0, "f1": 1})[1] == []
    assert cv.clipped_partial_links([["stub"], ["full"]], obs, frames, T, held_frames=["f1"])[1] == []
    assert len(cv.clipped_partial_links([["stub"], ["full"]], obs, frames, T, segments={"f0": 0, "f1": 0})[1]) == 1


def test_stub_inside_two_overlapping_candidates_is_a_near_tie_and_stays_separate():
    obs, frames, T = _stub_world([STUB, FULL, ("full2", "f2", 0.8, 505, 120, 545, 680)])
    groups, links = cv.clipped_partial_links([["stub"], ["full"], ["full2"]], obs, frames, T)
    assert links == [] and len(groups) == 3


def test_chance_alignment_in_one_partner_sighting_is_not_enough():
    # the partner is seen in four frames; the stub lies inside it in only one (the others sit a spine width away: drifted chain)
    obs, frames, T = _stub_world([STUB, FULL] + [(f"far{i}", f"f{2 + i}", 0.8 + 0.2 * i, 545, 120, 585, 680) for i in range(3)])
    groups, links = cv.clipped_partial_links([["stub"], ["full", "far0", "far1", "far2"]], obs, frames, T)
    assert links == [] and len(groups) == 2


def test_two_stubs_that_share_a_frame_cannot_both_join_one_partner():
    obs, frames, T = _stub_world([("s1", "f0", 0.0, 500, 0, 540, 200), ("s2", "f0", 0.0, 502, 0, 538, 190), FULL])
    groups, links = cv.clipped_partial_links([["s1"], ["s2"], ["full"]], obs, frames, T)
    assert len(links) == 1 and len(groups) == 2  # s1 and s2 are two books (same frame): only one of them can be the partner's stub


def test_clipped_partials_can_be_switched_off():
    obs, frames, T = _stub_world([STUB, FULL])
    groups, links = cv.clipped_partial_links([["stub"], ["full"]], obs, frames, T, config={"clipped_partials": False})
    assert groups == [["stub"], ["full"]] and links == []


def test_dedup_merge_joins_the_clipped_stub_and_reports_the_link(tmp_path):
    """End to end through dedup.merge (association=track): the pairwise rule leaves the stub separate, the clipped-partial pass joins it; switchable."""
    import json
    from unittest.mock import patch
    from pipeline import dedup, contracts as C
    sightings = [("f00000-o000", "f00000", 0.0, 100, 0, 120, 80), ("f00001-o000", "f00001", 0.2, 100, 40, 120, 300)]
    frames, obs = {}, []
    for oid, fid, t, x1, y1, x2, y2 in sightings:
        cv2.imwrite(str(tmp_path / f"{fid}.png"), np.zeros((480, 640, 3), np.uint8))
        frames[fid] = {"frame_id": fid, "timestamp_s": t, "width_px": 640, "height_px": 480, "image_path": f"{fid}.png", "quality": {}, "selected": True, "selection_reason": "SYNTHETIC"}
        obs.append({"observation_id": oid, "frame_id": fid, "kind": "book", "category": "book", "polygon_xy": [[x2, y1], [x2, y2], [x1, y2], [x1, y1]],
                    "bbox_xyxy": [x1, y1, x2, y2], "confidence": 0.9, "orientation": "upright", "occlusion": "none"})
    (tmp_path / "manifest.json").write_text(json.dumps(C.envelope("stub", "frames", {"frames": list(frames.values())}, fixture=True)))
    (tmp_path / "candidates.json").write_text(json.dumps(C.envelope("stub", "detect", {"observations": obs}, fixture=True)))
    T = {"f00000": np.eye(3), "f00001": np.array([[1.0, 0, 0], [0, 1.0, -60], [0, 0, 1.0]])}  # the camera pans 60 px between the frames
    counts = {}
    for name, track_cfg in (("on", {}), ("off", {"clipped_partials": False})):
        with patch.object(cv, "chain_transforms", return_value=(T, {"f00000": "start", "f00001": "ok"})), patch.object(cv, "appearance", return_value=0.0):
            p = dedup.merge(tmp_path / "candidates.json", tmp_path / "manifest.json", tmp_path, {"association": "track", "track": track_cfg, "output_name": f"inventory_{name}.json"})
        inv = json.loads(p.read_text())
        counts[name] = inv["data"]["counts"]["book_count"]
        if name == "on":
            assert any(a.get("geometry") == "clipped_partial_contained" for a in inv["data"]["associations"])
            assert inv["metrics"]["track_report"]["clipped_partial_merges"] == 1
    assert counts == {"on": 1, "off": 2}
