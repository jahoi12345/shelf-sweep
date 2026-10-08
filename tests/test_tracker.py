"""Synthetic checks for pipeline/tracker.py (mechanics only, not accuracy on real footage)."""
import numpy as np

from pipeline import tracker as TR

EYE = np.eye(3)


def _spine(fid, oid, x, w=30, y0=100, y1=400):
    return {"observation_id": f"{fid}-o{oid:03d}", "frame_id": fid, "kind": "book", "confidence": 0.9,
            "polygon_xy": [[x, y0], [x + w, y0], [x + w, y1], [x, y1]], "bbox_xyxy": [x, y0, x + w, y1]}


def _frames(n):
    return {f"f{k:05d}": {"timestamp_s": 0.2 * k, "width_px": 1920, "height_px": 1080} for k in range(n)}


def test_detection_gap_is_bridged_and_long_gap_is_not():
    fr = _frames(6)
    obs = [_spine(f, 0, 500) for f in ("f00000", "f00001", "f00004", "f00005")]   # missing in frames 2 and 3 (flicker)
    T = {f: EYE for f in fr}
    tracks, _ = TR.build_tracks(obs, fr, T)
    assert len(tracks) == 1 and len(tracks[0]) == 4
    tracks, _ = TR.build_tracks(obs, fr, T, {"max_gap_frames": 1})
    assert len(tracks) == 2


def test_motion_is_compensated_by_the_chain():
    fr = _frames(4)
    # camera pans: the spine moves 120 px left per frame in the image; the chain (frame -> plane) undoes it
    obs = [_spine(f"f{k:05d}", 0, 900 - 120 * k) for k in range(4)]
    T = {f"f{k:05d}": np.array([[1, 0, 120 * k], [0, 1, 0], [0, 0, 1]], float) for k in range(4)}
    tracks, _ = TR.build_tracks(obs, fr, T)
    assert len(tracks) == 1
    # without motion compensation the boxes (30 px wide, 120 px apart) do not overlap: four tracks
    tracks, _ = TR.build_tracks(obs, fr, {f: EYE for f in fr})
    assert len(tracks) == 4


def test_vanished_spine_track_is_not_taken_over_by_its_neighbour():
    fr = _frames(4)
    obs = [_spine("f00000", 0, 500), _spine("f00000", 1, 535), _spine("f00001", 0, 500), _spine("f00001", 1, 535),
           _spine("f00002", 1, 535), _spine("f00003", 1, 535)]          # left spine disappears after frame 1
    tracks, _ = TR.build_tracks(obs, fr, {f: EYE for f in fr})
    by_track = sorted(sorted(t) for t in tracks)
    assert by_track == [["f00000-o000", "f00001-o000"], ["f00000-o001", "f00001-o001", "f00002-o001", "f00003-o001"]]


def test_track_v2_edges_keep_cross_view_edges_and_add_tracker_edges():
    fr = _frames(3)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (1080, 1920, 3)).astype(np.uint8)
    obs = [_spine(f, 0, 500) for f in fr] + [_spine(f, 1, 600) for f in fr]
    edges, report, pairs = TR.track_v2_edges(obs, fr, lambda f: img, None, transforms={f: EYE for f in fr})
    geo = {e["geometry"] for e in edges}
    assert "tracker_iou" in geo and report["tracker"]["tracks"] == 2
    assert all(e["score"] > 1.0 for e in edges if e["geometry"] == "tracker_iou")


def test_track_does_not_continue_across_segment_boundary_or_through_hold_frames():
    fr = _frames(6)
    obs = [_spine(f"f{k:05d}", 0, 500) for k in range(6)]          # same pixel position in every frame
    T = {f: EYE for f in fr}
    # segments: frames 0-2 in segment 0, frames 3-5 in segment 1 (separate planes: identical pixels mean nothing across them)
    seg = {f"f{k:05d}": (0 if k < 3 else 1) for k in range(6)}
    tracks, _ = TR.build_tracks(obs, fr, T, segment=seg)
    assert sorted(len(t) for t in tracks) == [3, 3]
    # a provisional (hold) frame is never linked; the chain re-anchors the next good frame to the last good one, so the track may bridge over it
    tracks, _ = TR.build_tracks(obs, fr, T, unreliable={"f00002"})
    assert sorted(sorted(t) for t in tracks) == [["f00000-o000", "f00001-o000", "f00003-o000", "f00004-o000", "f00005-o000"], ["f00002-o000"]]


def test_identical_adjacent_spines_swap_only_beyond_half_a_pitch_of_position_error():
    """Documented limit of IoU tracking: a row of identical adjacent spines is tracked correctly while the predicted position is off by < 0.5 pitch
    and every link goes to the neighbour once it is off by > 0.5 pitch. Appearance cannot rescue it (the copies are identical)."""
    w, g, n = 30.0, 2.0, 8
    pitch = w + g
    fr = _frames(2)

    def wrong_links(err):
        obs = [_spine("f00000", i, i * pitch) for i in range(n)] + [_spine("f00001", i, i * pitch + err * pitch) for i in range(n)]
        for o in obs:
            o["observation_id"] = f"{o['frame_id']}-o{int(round((o['polygon_xy'][0][0] - (err * pitch if o['frame_id'] == 'f00001' else 0)) / pitch)):03d}"
        tr, _ = TR.build_tracks(obs, fr, {f: EYE for f in fr})
        return sum(1 for t in tr if len(t) == 2 and t[0].split("-o")[1] != t[1].split("-o")[1])
    assert wrong_links(0.3) == 0
    assert wrong_links(0.7) > 0
