"""pipeline.auto_room on a synthetic box room reconstruction (known size), plus abstention when surfaces were not reconstructed.
Synthetic geometry only: not accuracy evidence against tape."""
import numpy as np
import pytest

from pipeline import auto_room as AR
from pipeline import auto_scale as AS

W, D, HT = 5.0, 4.0, 2.5                       # metres (1 reconstruction unit = 1 m here, cm_per_unit = 100)
X0, Y0 = -2.5, -1.2                            # room x in [X0, X0+W], floor y = Y0, ceiling y = Y0 + HT, shelf wall z = 0, opposite wall z = D
SCALE = {"source": "catalogue_physical_dimensions", "cm_per_unit": 100.0, "relative_uncertainty_pct": 4.0}


def face(rng, n, fixed_axis, fixed, ranges, noise=0.008):
    P = np.zeros((n, 3))
    for a, (lo, hi) in ranges.items():
        P[:, a] = rng.uniform(lo, hi, n)
    P[:, fixed_axis] = fixed + rng.normal(0, noise, n)
    return P


def room_cloud(seed=0, floor=True, ceiling=True, opposite=True, sides=True, clutter=True):
    rng = np.random.default_rng(seed)
    xr, yr, zr = (X0, X0 + W), (Y0, Y0 + HT), (0.0, D)
    parts = [face(rng, 3000, 2, 0.0, {0: xr, 1: yr})]                                  # shelf wall (z = 0): always reconstructed
    if floor:
        parts.append(face(rng, 1500, 1, Y0, {0: xr, 2: zr}))
    if ceiling:
        parts.append(face(rng, 1200, 1, Y0 + HT, {0: xr, 2: zr}))
    if opposite:
        parts.append(face(rng, 1200, 2, D, {0: xr, 1: yr}))
    if sides:
        parts.append(face(rng, 1000, 0, X0, {1: yr, 2: zr}))
        parts.append(face(rng, 1000, 0, X0 + W, {1: yr, 2: zr}))
    if clutter:
        parts.append(face(rng, 300, 1, -0.45, {0: (-0.5, 0.5), 2: (1.0, 1.8)}))       # a table top
        parts.append(face(rng, 200, 2, 1.4, {0: (0.2, 0.6), 1: (-0.8, 0.0)}))          # a standing object
    return np.vstack(parts)


def recon(points):
    rng = np.random.default_rng(1)
    cams = [{"frame_id": f"f{i}", "center": [rng.uniform(-1.5, 1.5), rng.uniform(0.1, 0.3), rng.uniform(1.5, 2.5)]} for i in range(30)]
    return {"points": points.tolist(), "cameras": cams}


def test_a_filmed_box_room_gives_height_depth_width_and_areas_with_uncertainty():
    r = AR.estimate(recon(room_cloud()), SCALE)
    assert r["status"] == "estimated" and r["shape"]["kind"] == "rectangular"
    d = r["dimensions"]
    assert d["height_m"]["value"] == pytest.approx(HT, abs=0.05) and d["depth_m"]["value"] == pytest.approx(D, abs=0.05) and d["width_m"]["value"] == pytest.approx(W, abs=0.05)
    assert r["areas"]["floor_area_m2"]["value"] == pytest.approx(W * D, rel=0.03)
    assert r["areas"]["wall_area_m2"]["value"] == pytest.approx(2 * (W + D) * HT, rel=0.03)
    assert r["areas"]["floor_area_m2"]["ft2"] == pytest.approx(W * D * AR.FT2_PER_M2, rel=0.03)
    for k in ("floor_area_m2", "wall_area_m2"):
        assert r["areas"][k]["relative_uncertainty_pct"] >= 2 * SCALE["relative_uncertainty_pct"] - 0.2          # area scales with scale^2
    assert r["validation_state"] == "software_only_unvalidated" and "openings NOT subtracted" in r["areas"]["wall_area_m2"]["formula"]


def test_a_table_top_and_a_standing_object_are_not_a_floor_or_a_wall():
    r = AR.estimate(recon(room_cloud()), SCALE)
    assert r["planes"]["floor"]["c"] == pytest.approx(Y0, abs=0.03)
    assert r["planes"]["opposite_wall"]["c"] == pytest.approx(D, abs=0.03)


@pytest.mark.parametrize("missing,reason,null_fields", [
    ({"ceiling": False}, "ceiling_not_found", ("height_m", "wall_area_m2")),
    ({"floor": False}, "floor_not_found", ("height_m", "wall_area_m2")),
    ({"opposite": False}, "opposite_wall_not_found", ("depth_m", "floor_area_m2", "wall_area_m2")),
    ({"sides": False}, "left_wall_not_found", ("width_m", "floor_area_m2", "wall_area_m2")),
])
def test_unfilmed_surfaces_abstain_with_a_reason_and_filming_guidance_never_zero(missing, reason, null_fields):
    r = AR.estimate(recon(room_cloud(**missing)), SCALE)
    assert r["status"] in ("partial", "abstained") and any(reason in x for x in r["abstention_reasons"])
    for k in null_fields:
        v = (r["dimensions"].get(k) or r["areas"].get(k))
        assert v["value"] is None and v["reason"]
    assert r["capture_guidance"] and not any(w in " ".join(r["capture_guidance"]).lower() for w in ("how tall", "measure", "tap", "tell me"))
    assert all(v.get("value") != 0 for v in list(r["dimensions"].values()) + list(r["areas"].values()))


def test_only_the_shelf_wall_reconstructed_abstains_everything():
    r = AR.estimate(recon(room_cloud(floor=False, ceiling=False, opposite=False, sides=False, clutter=False)), SCALE)
    assert r["status"] == "abstained" and r["areas"]["floor_area_m2"]["value"] is None and len(r["abstention_reasons"]) >= 5


def test_without_a_metric_scale_nothing_metric_is_reported():
    r = AR.estimate(recon(room_cloud()), {"cm_per_unit": None, "reason": "no_identified_books"})
    assert r["status"] == "abstained" and all(v["value"] is None for v in r["dimensions"].values()) and any(x.startswith("no_metric_scale") for x in r["abstention_reasons"])


def test_implausible_geometry_is_rejected_not_reported():
    r = AR.estimate(recon(room_cloud()), {"cm_per_unit": 400.0, "relative_uncertainty_pct": 4.0})        # absurd scale: 10 m ceiling
    assert r["dimensions"]["height_m"]["value"] is None and r["dimensions"]["height_m"]["reason"].startswith("implausible_ceiling_height")


def test_too_sparse_reconstruction_abstains():
    r = AR.estimate({"points": [[0, 0, 0]] * 10, "cameras": []}, SCALE)
    assert r["status"] == "abstained" and r["abstention_reasons"][0].startswith("reconstruction_too_sparse")


def test_measure_data_includes_the_room_block():
    from tests.test_auto_scale import recon as spine_recon
    rc = spine_recon({f"bk-{i}": (0.05 * i - 0.5, 0.0, (20.0 + (i % 5)) / 35.0, 3.0 / 35.0) for i in range(14)})
    rc["points"] = room_cloud().tolist()
    d = AS.measure_data(rc, [], None)
    assert d["room"]["status"] in ("estimated", "partial", "abstained") and "dimensions" in d["room"] and d["room"]["scale"]["source"] == "standard_format_prior"


def test_a_shelf_board_inside_the_books_is_not_mistaken_for_the_floor():
    rng = np.random.default_rng(3)
    board_y = -0.3                                                                           # bottom shelf board, books stand on it
    cloud = np.vstack([room_cloud(floor=False, clutter=False), face(rng, 900, 1, board_y, {0: (X0, X0 + W), 2: (0.0, 0.9 * D)})])
    shelf = {"books_y_range_units": [board_y, 0.9]}
    r = AR.estimate(recon(cloud), SCALE, None, shelf)
    assert r["planes"]["floor"] is None and any(x.startswith("floor_not_found") for x in r["abstention_reasons"])
    assert r["dimensions"]["height_m"]["value"] is None
    # with the true floor also present the floor below the lowest book is chosen
    both = np.vstack([cloud, face(rng, 1500, 1, Y0, {0: (X0, X0 + W), 2: (0.0, D)})])
    r2 = AR.estimate(recon(both), SCALE, None, shelf)
    assert r2["planes"]["floor"]["c"] == pytest.approx(Y0, abs=0.03) and r2["dimensions"]["height_m"]["value"] == pytest.approx(HT, abs=0.05)
