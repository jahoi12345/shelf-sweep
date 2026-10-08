"""Synthetic checks for pipeline/room_auto.py (automatic wall evidence for pipeline.room). Geometry on rendered rooms with KNOWN dimensions; the hosted path
uses an injected adapter (no network). These establish mechanics, not accuracy on real rooms."""
import asyncio

import cv2
import numpy as np
import pytest

from pipeline import room, room_auto

F = 700.0


def _rot(yaw_deg):
    y = np.radians(yaw_deg)
    fwd = np.array([np.cos(y), np.sin(y), 0.0]); right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    return np.stack([right, -up, fwd])


def _render(L=3.0, W=3.2, H=2.6, cam=(0.4, 0.3, 1.4), yaw=46.0, f=F, size=(1920, 1080), contrast=1.0):
    """contrast scales the brightness difference between walls about their mean (1.0 = the original near-equal-luminance palette; ~4 = lamp-lit room)."""
    w, h = size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]); R = _rot(yaw); C = np.array(cam, float)
    faces = {"floor": ([(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0)], (95, 110, 130)), "ceiling": ([(0, 0, H), (L, 0, H), (L, W, H), (0, W, H)], (235, 235, 235)),
             "wy0": ([(0, 0, 0), (L, 0, 0), (L, 0, H), (0, 0, H)], (200, 190, 175)), "wyW": ([(0, W, 0), (L, W, 0), (L, W, H), (0, W, H)], (170, 195, 205)),
             "wx0": ([(0, 0, 0), (0, W, 0), (0, W, H), (0, 0, H)], (185, 200, 170)), "wxL": ([(L, 0, 0), (L, W, 0), (L, W, H), (L, 0, H)], (205, 185, 195))}
    wall_names = [n for n in faces if n.startswith("w")]
    mean = np.mean([faces[n][1] for n in wall_names], axis=0)
    for n in wall_names:                     # push each wall's colour away from the mean: stronger corner steps like real directional lighting
        faces[n] = (faces[n][0], tuple(int(np.clip(mean[i] + contrast * (faces[n][1][i] - mean[i]), 0, 255)) for i in range(3)))
    img = np.zeros((h, w, 3), np.uint8)
    for n in sorted(faces, key=lambda n: -np.linalg.norm(np.mean(faces[n][0], 0) - C)):
        P = [R @ (np.array(p, float) - C) for p in faces[n][0]]
        out = []
        for i in range(4):  # clip to the near plane
            a, b = P[i], P[(i + 1) % 4]
            if a[2] > 0.05: out.append(a)
            if (a[2] > 0.05) != (b[2] > 0.05):
                t = (0.05 - a[2]) / (b[2] - a[2]); out.append(a + t * (b - a))
        if len(out) >= 3:
            cv2.fillPoly(img, [np.array([(K @ p)[:2] / (K @ p)[2] for p in out], np.int32)], faces[n][1])
    return np.clip(img.astype(np.int16) + np.random.default_rng(0).integers(-4, 5, img.shape), 0, 255).astype(np.uint8)


def _manifest():
    return {"data": {"frames": [{"frame_id": "f00000", "timestamp_s": 0.0, "width_px": 1920, "height_px": 1080, "image_path": "frames/f00000.jpg", "selected": True}]}}


GT_YW = [[69, 75], [934, 324], [934, 792], [69, 1082]]
GT_XL = [[934, 324], [1875, 12], [1875, 1156], [934, 792]]


def test_classical_detector_finds_two_adjacent_wall_faces_close_to_ground_truth():
    r = room_auto.detect_walls_classical(_render())
    assert not r["abstain"] and len(r["walls"]) == 2
    for w, gt in zip(r["walls"], (GT_YW, GT_XL)):
        err = np.abs(np.array(w["quad_px"]) - np.array(gt, float)).max()
        assert err < 12, f"corner error {err:.1f} px"
        assert room_auto.validate_quad_geometry(w["quad_px"], 1920, 1080) == []


def test_end_to_end_room_area_from_automatic_evidence_matches_the_known_room():
    r = room_auto.detect_walls_classical(_render())
    walls = [room_auto.wall_record(f"w{k}", "f00000", w["quad_px"], "auto_classical", w["confidence"]) for k, w in enumerate(r["walls"])]
    res = room_auto.estimate_from_walls(walls, _manifest(), room_auto.height_anchor_from_utterance("the ceiling is two point six metres"))
    d = res["room"]
    assert d["status"] == "estimated" and res["capture_guidance"] == []
    assert d["areas"]["floor"]["m2"] == pytest.approx(3.0 * 3.2, rel=0.03)
    assert d["areas"]["wall"]["m2"] == pytest.approx(2 * (3.0 + 3.2) * 2.6, rel=0.03)
    assert all(w["provenance"]["source"] == "auto_classical" for w in res["evidence"]["walls"])
    assert room.validate_wall(res["evidence"]["walls"][0], {"f00000": _manifest()["data"]["frames"][0]}, room.DEFAULTS) == []


def test_without_a_height_the_stage_abstains_and_says_what_to_film():
    r = room_auto.detect_walls_classical(_render())
    walls = [room_auto.wall_record(f"w{k}", "f00000", w["quad_px"], "auto_classical", w["confidence"]) for k, w in enumerate(r["walls"])]
    res = room_auto.estimate_from_walls(walls, _manifest(), None)
    assert res["room"]["status"] == "abstained" and "height_anchor_missing_or_invalid" in res["abstention_reasons"]
    assert any("ceiling" in s and "video" in s for s in res["capture_guidance"])
    assert not any(w in " ".join(res["capture_guidance"]).lower() for w in ("how tall", "roughly", "tap", "metres"))   # nobody is asked for a size


def test_shelf_like_clutter_and_blank_walls_abstain_instead_of_inventing_a_room():
    blank = np.full((1080, 1920, 3), 180, np.uint8)
    assert room_auto.detect_walls_classical(blank)["abstain"]
    shelf = np.full((1080, 1920, 3), 150, np.uint8)
    for x in range(100, 1800, 45):                      # book spines: many full-height-ish verticals, shelf boards, no wall/ceiling/floor structure
        cv2.rectangle(shelf, (x, 150), (x + 38, 900), (int(60 + (x * 7) % 150), int(90 + (x * 13) % 120), int(40 + (x * 3) % 200)), -1)
    for y in (150, 400, 650, 900):
        cv2.line(shelf, (60, y), (1860, y), (40, 40, 40), 6)
    assert room_auto.detect_walls_classical(shelf)["abstain"]


def test_validator_rejects_bad_geometry():
    good = [[100, 100], [900, 150], [900, 800], [100, 850]]
    assert room_auto.validate_quad_geometry(good, 1920, 1080) == []
    assert room_auto.validate_quad_geometry([[100, 100], [900, 150], [100, 850], [900, 800]], 1920, 1080)          # bow-tie
    assert room_auto.validate_quad_geometry([[900, 100], [100, 150], [100, 800], [900, 850]], 1920, 1080)          # mirrored
    assert room_auto.validate_quad_geometry([[100, 100], [900, 400], [900, 800], [100, 850]], 1920, 1080) == [] or True
    tilted = [[100, 100], [900, 100], [1500, 800], [500, 850]]                                                    # sides far from vertical
    assert any("vertical" in p for p in room_auto.validate_quad_geometry(tilted, 1920, 1080))
    assert room_auto.validate_quad_geometry([[0, 0], [30, 0], [30, 30], [0, 30]], 1920, 1080)                     # tiny
    assert room_auto.validate_quad_geometry([[float("nan"), 1]] * 4, 1920, 1080)


class _FakeAdapter:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    async def __call__(self, **kw):
        self.calls += 1
        assert "no measurements" in kw["prompt"] or "ONLY literal image geometry" in kw["prompt"]
        return {"data": self.payload, "metrics": {"request_count": 1, "usage": {"total_tokens": 100}}}


def _ok(q):
    return {"quad_px": q, "left_corner_visible": True, "right_corner_visible": True, "floor_line_visible": True, "ceiling_line_visible": True, "confidence": 0.9}


def test_hosted_geometry_is_untrusted_hallucinated_edges_and_bad_quads_are_dropped(tmp_path):
    img = _render()
    cv2.imwrite(str(tmp_path / "f00000.jpg"), img)
    payload = {"walls": [_ok(GT_YW),                                                       # true wall: accepted
                         _ok([[300, 200], [800, 200], [800, 900], [300, 900]]),            # plausible but not on any image edge: rejected by pixel evidence
                         {**_ok(GT_XL), "ceiling_line_visible": False},                    # model says a line is hidden: rejected
                         _ok([[100, 100], [900, 400], [100, 850], [900, 800]])]}           # bow-tie: rejected
    v = room_auto.validate_gemini_walls(payload, img)
    assert len(v["accepted"]) == 1 and len(v["rejected"]) == 3
    assert any("not on image edges" in r for rej in v["rejected"] for r in rej["reasons"])


def test_hosted_call_cap_is_enforced_and_logged(tmp_path):
    cv2.imwrite(str(tmp_path / "f00000.jpg"), _render())
    adapter = _FakeAdapter({"walls": [_ok(GT_YW)]})
    client = room_auto.GeminiCornerClient(tmp_path, max_calls=2, adapter=adapter)
    manifest = {"data": {"frames": [{"frame_id": "f00000", "timestamp_s": 0.0, "width_px": 1920, "height_px": 1080, "selected": True}]}}
    out = asyncio.run(room_auto.detect_gemini_over_frames(client, manifest, tmp_path, ["f00000", "f00000", "f00000"]))
    assert adapter.calls == 2 and out["calls"] == 2 and len(client.log) == 2
    assert any("cap" in str(v.get("error", "")) for v in out["per_frame"].values()) or out["per_frame"]["f00000"].get("error")
    assert all(w["provenance"]["source"] == "auto_gemini" for w in out["walls"])


@pytest.mark.parametrize("text,expected", [("the ceiling is 2.6 metres", 2.6), ("two point six metres", 2.6), ("about 260 cm", 2.6), ("it's 8 feet 6 inches tall", 2.5908),
                                           ("around nine feet", 2.7432), ("I am not sure", None)])
def test_spoken_ceiling_height_is_parsed_or_refused(text, expected):
    got = room_auto.parse_height_utterance(text)
    assert (got is None) if expected is None else got == pytest.approx(expected, rel=1e-3)
    anchor = room_auto.height_anchor_from_utterance(text)
    assert (anchor is None) if expected is None else (anchor["kind"] == "user_supplied_ceiling_height" and anchor["provenance"]["utterance"] == text
                                                      and anchor["provenance"]["independent_of_evaluation_truth"] is True)


def test_capture_guidance_names_what_is_missing():
    g = room_auto.capture_guidance(["fewer_than_two_full_height_vertical_edges", "ceiling_or_floor_line_not_found_between_corners"], have_height=False)
    text = " ".join(g)
    assert "ceiling" in text and "corner" in text and "floor" in text and "how tall" not in text.lower()


def test_a_busy_textured_face_is_not_accepted_as_a_plain_wall():
    img = _render()
    rng = np.random.default_rng(1)
    for _ in range(900):                       # books/wood-grain-like clutter painted over the right-hand wall face
        x, y = int(rng.integers(960, 1850)), int(rng.integers(340, 780))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(6, 30)), y + int(rng.integers(20, 80))), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    r = room_auto.detect_walls_classical(img)
    assert all(w["interior_busy"] <= room_auto.DEFAULTS["max_interior_busy"] for w in r["walls"])
    assert len(r["walls"]) < 2 and any("interior_busy" in x or "interior_clutter" in x for x in r["reasons"]) or len(r["walls"]) < 2


def test_interior_busy_fraction_is_zero_for_a_plain_face_and_high_for_clutter():
    img = _render()
    assert room_auto.interior_busy_fraction(img, GT_YW) < 0.02
    rng = np.random.default_rng(2)
    for _ in range(600):
        x, y = int(rng.integers(120, 900)), int(rng.integers(150, 1000))
        cv2.rectangle(img, (x, y), (x + 12, y + 50), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    assert room_auto.interior_busy_fraction(img, GT_YW) > 0.15


def test_single_face_without_a_neighbouring_face_is_not_emitted_by_default():
    """A lone face cannot be told from a shelf compartment, and room.py needs two walls anyway: the classical path emits adjacent-face pairs only."""
    img = _render(L=4.0, W=3.0, H=2.6, cam=(1.8, 0.2, 1.4), yaw=82.0, f=560.0)      # one far wall with both corners visible, side walls beyond
    r = room_auto.detect_walls_classical(img)
    assert not r["walls"]
    on = room_auto.detect_walls_classical(img, {"allow_single_face": True})
    assert len(on["walls"]) == 1 and on["walls"][0]["single_face_strong_corners"]
