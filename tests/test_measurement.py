"""Geometry/measurement tests. SYNTHETIC: they show the software is mathematically consistent and abstains
correctly. They are NOT evidence of accuracy on real shelves or rooms (no physical tape truth is used or claimed)."""
import copy
import math
import re
from pathlib import Path

import numpy as np
import pytest

from pipeline import contracts as C
from pipeline import packet as P
from pipeline import room as RM
from pipeline import scale as S

ROOT = Path(__file__).resolve().parents[1]
FW, FH = 1920, 1080
PROV = {"source": "user_supplied", "actor": "claimant", "supplied_at": "2026-10-06T10:00:00+02:00",
        "utterance": "this shelving unit is 1.8 metres wide and 2 metres tall", "independent_of_evaluation_truth": True}
NO_UNC = {"plane_offset_pct": 0.0, "pixel_error_px": 0.0, "occlusion_unknown_pct": 0.0, "hop_pct": 0.0}

# an oblique plane: metres on the shelf face -> image pixels (known ground-truth geometry)
HPI = np.array([[480, -90, 300], [40, 330, 120], [0.00045, 0.00020, 1.0]], float)


def to_img(pts_m):
    return S.apply_h(HPI, pts_m)


def book_m(cx, cy, h, t, ang_deg):
    c, s = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
    pts = np.array([[-t / 2, h / 2], [t / 2, h / 2], [t / 2, -h / 2], [-t / 2, -h / 2]])
    return pts @ np.array([[c, -s], [s, c]]).T + [cx, cy]


def manifest(frame_ids=("f00000",)):
    frames = [{"frame_id": f, "timestamp_s": float(i), "image_path": f"frames/{f}.jpg", "width_px": FW, "height_px": FH,
               "quality": {}, "selected": True, "selection_reason": ""} for i, f in enumerate(frame_ids)]
    return C.envelope("t", "frames", {"input": {}, "frames": frames, "capture_gaps": [], "audit": {}}, status="ok")


def cal_quad(W=1.8, H=2.0, pts_m=None, tol=0.0, plane_role="shelf_face", frame="f00000", cid="cal-001", prov=None):
    corners = pts_m if pts_m is not None else [[0, 0], [W, 0], [W, H], [0, H]]
    return {"calibration_id": cid, "kind": "plane_quad", "frame_id": frame, "image_points_px": to_img(corners).tolist(),
            "metric": {"width_m": W, "height_m": H, "tolerance_pct": tol}, "plane": {"id": "plane-A", "role": plane_role},
            "provenance": prov or PROV}


def obs(oid, poly_m, frame="f00000", orientation="upright", occlusion="none", H=HPI):
    poly = S.apply_h(H, poly_m)
    bb = [float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())]
    return {"observation_id": oid, "frame_id": frame, "polygon_xy": poly.tolist(), "bbox_xyxy": bb,
            "orientation": orientation, "occlusion": occlusion, "kind": "book"}


def inv(instances, regs=None):
    books = [{"instance_id": f"bk-{i+1:03d}", "observation_ids": ids, "uncertain_association": u} for i, (ids, u) in enumerate(instances)]
    return {"data": {"books": books, "items": [], "registrations": regs or []}}


def measure(instances, observations, cals, cfg=NO_UNC, regs=None, frames=("f00000",)):
    return S.measure_data(inv(instances, regs), manifest(frames), {o["observation_id"]: o for o in observations}, cals, cfg)


# ------------------------------------------------------------------ plane rectification
def test_oblique_plane_recovers_metric_spine_size_exactly():
    o = obs("f00000-o000", book_m(0.9, 1.2, 0.24, 0.032, 0))
    d = measure([(["f00000-o000"], False)], [o], [cal_quad()])
    m = d["measurements"][0]
    assert m["status"] == "measured"
    assert m["spine_height_cm"] == pytest.approx(24.0, abs=0.01) and m["spine_thickness_cm"] == pytest.approx(3.2, abs=0.01)
    assert d["calibrations"][0]["oblique_index"] > 1.0 and d["validation"]["physical_truth_compared"] is False


@pytest.mark.parametrize("ang,orient", [(12, "leaning"), (-25, "leaning"), (90, "flat"), (84, "flat")])
def test_leaning_and_horizontal_books_use_their_own_axes(ang, orient):
    o = obs("f00000-o000", book_m(0.9, 1.0, 0.24, 0.035, ang), orientation=orient)
    m = measure([(["f00000-o000"], False)], [o], [cal_quad()])["measurements"][0]
    assert m["status"] == "measured"
    assert m["spine_height_cm"] == pytest.approx(24.0, abs=0.02) and m["spine_thickness_cm"] == pytest.approx(3.5, abs=0.02)


def test_uncertainty_interval_widens_with_declared_terms_and_contains_truth():
    o = obs("f00000-o000", book_m(0.9, 1.2, 0.24, 0.032, 0))
    narrow = measure([(["f00000-o000"], False)], [o], [cal_quad()], NO_UNC)["measurements"][0]
    wide = measure([(["f00000-o000"], False)], [o], [cal_quad(tol=5.0)], {})["measurements"][0]
    w = lambda m: m["spine_height_interval_cm"][1] - m["spine_height_interval_cm"][0]
    assert w(wide) > w(narrow)
    assert wide["spine_height_interval_cm"][0] <= 24.0 <= wide["spine_height_interval_cm"][1]
    assert "not a statistical guarantee" in wide["interval_note"]


def test_pixel_localisation_term_makes_small_spines_less_certain():
    big = obs("f00000-o000", book_m(0.9, 1.2, 0.30, 0.05, 0)); small = obs("f00000-o001", book_m(1.2, 1.2, 0.30, 0.012, 0))
    d = measure([(["f00000-o000"], False), (["f00000-o001"], False)], [big, small], [cal_quad()], {"pixel_error_px": 2.0})
    wt = lambda m: m["spine_thickness_interval_cm"][1] - m["spine_thickness_interval_cm"][0]
    assert wt(d["measurements"][1]) / d["measurements"][1]["spine_thickness_cm"] > wt(d["measurements"][0]) / d["measurements"][0]["spine_thickness_cm"]


# ------------------------------------------------------------------ abstention / scope
def test_no_calibration_means_every_dimension_unknown():
    o = obs("f00000-o000", book_m(0.9, 1.2, 0.24, 0.032, 0))
    d = measure([(["f00000-o000"], False)], [o], [])
    m = d["measurements"][0]
    assert m["spine_height_cm"] is None and m["abstention_reasons"] == ["no_valid_calibration"] and d["shelf_run_m"] is None


def test_scope_mismatch_object_outside_plane_region_abstains():
    far = obs("f00000-o000", book_m(2.5, 1.2, 0.24, 0.032, 0))      # 0.7 m beyond the right edge of a 1.8 m wide calibrated unit
    m = measure([(["f00000-o000"], False)], [far], [cal_quad()])["measurements"][0]
    assert m["status"] == "abstained" and "outside_calibrated_plane_region" in m["abstention_reasons"]


def test_invalid_calibrations_are_rejected_with_reasons():
    frames = {"f00000": manifest()["data"]["frames"][0]}
    good = cal_quad()
    assert S.validate_calibration(good, frames, S.DEFAULTS) == []
    bad_prov = copy.deepcopy(good); bad_prov["provenance"] = {"source": "assumed_standard_size", "actor": "x", "supplied_at": "t", "utterance": "u"}
    assert any("source must be" in p for p in S.validate_calibration(bad_prov, frames, S.DEFAULTS))
    not_indep = copy.deepcopy(good); not_indep["provenance"]["independent_of_evaluation_truth"] = False
    assert any("independent_of_evaluation_truth" in p for p in S.validate_calibration(not_indep, frames, S.DEFAULTS))
    no_utt = copy.deepcopy(good); no_utt["provenance"].pop("utterance")
    assert any("utterance or note" in p for p in S.validate_calibration(no_utt, frames, S.DEFAULTS))
    twisted = copy.deepcopy(good); twisted["image_points_px"] = [twisted["image_points_px"][i] for i in (0, 2, 1, 3)]
    assert any("non-convex" in p for p in S.validate_calibration(twisted, frames, S.DEFAULTS))
    zero = copy.deepcopy(good); zero["metric"]["width_m"] = 0
    assert any("metric.width_m" in p for p in S.validate_calibration(zero, frames, S.DEFAULTS))
    off = copy.deepcopy(good); off["image_points_px"][0] = [-50, 10]
    assert any("outside" in p for p in S.validate_calibration(off, frames, S.DEFAULTS))
    d = measure([(["f00000-o000"], False)], [obs("f00000-o000", book_m(0.9, 1.2, 0.24, 0.032, 0))], [bad_prov])
    assert d["summary"]["valid_calibrations"] == 0 and d["measurements"][0]["spine_height_cm"] is None


def test_single_segment_scales_only_its_own_axis_unless_fronto_parallel_declared():
    # frontal view (identity-like plane map) so axes are image-aligned
    Hf = np.array([[500.0, 0, 300], [0, 500.0, 100], [0, 0, 1]])
    seg = {"calibration_id": "cal-s", "kind": "segment", "frame_id": "f00000",
           "image_points_px": S.apply_h(Hf, [[0, 0.5], [1.8, 0.5]]).tolist(),     # horizontal 1.8 m width anchor
           "metric": {"length_m": 1.8, "tolerance_pct": 2.0}, "plane": {"id": "p", "role": "shelf_face"}, "provenance": PROV}
    o = obs("f00000-o000", book_m(0.9, 1.0, 0.24, 0.032, 0), H=Hf)
    m = measure([(["f00000-o000"], False)], [o], [seg])["measurements"][0]
    assert m["status"] == "abstained" and "single_segment_anchor_does_not_determine_spine_length_axis" in m["abstention_reasons"]
    fp = copy.deepcopy(seg); fp["assumptions"] = {"assume_fronto_parallel": True}
    m2 = measure([(["f00000-o000"], False)], [o], [fp])["measurements"][0]
    assert m2["status"] == "measured" and m2["spine_height_cm"] == pytest.approx(24.0, abs=0.05)
    assert m2["spine_height_interval_cm"][1] - m2["spine_height_cm"] > 0.05 * 24 * 0.5     # extra stated uncertainty


@pytest.mark.parametrize("kw,reason", [({"occlusion": "partial"}, "occluded_partial"), ({"orientation": "stacked"}, "stacked_orientation_axes_ambiguous")])
def test_occluded_or_stacked_observations_abstain(kw, reason):
    o = obs("f00000-o000", book_m(0.9, 1.2, 0.24, 0.032, 0), **kw)
    m = measure([(["f00000-o000"], False)], [o], [cal_quad()])["measurements"][0]
    assert m["status"] == "abstained" and reason in m["abstention_reasons"]


def test_truncated_and_near_square_footprints_abstain():
    o = obs("f00000-o000", book_m(0.9, 1.2, 0.24, 0.032, 0)); o["bbox_xyxy"][0] = 0.0
    assert "truncated_at_frame_edge" in measure([(["f00000-o000"], False)], [o], [cal_quad()])["measurements"][0]["abstention_reasons"]
    sq = obs("f00000-o001", book_m(0.9, 1.2, 0.05, 0.045, 0))
    assert "ambiguous_axes_near_square_footprint" in measure([(["f00000-o001"], False)], [sq], [cal_quad()])["measurements"][0]["abstention_reasons"]


# ------------------------------------------------------------------ transfer + aggregation
def _reg(a, b, H, inl=0.6, rp=1.0):
    return {"frame_ids": [a, b], "status": "accepted", "homography": H.tolist(), "inlier_ratio": inl, "median_reprojection_px": rp}


def test_calibration_transfers_through_registration_and_hops_add_uncertainty():
    # frame B sees the same plane shifted right by 240 px: image(a)->image(b) is a translation
    T = np.array([[1, 0, 240.0], [0, 1, 0], [0, 0, 1]])
    HPI_B = T @ HPI
    o_b = obs("f00001-o000", book_m(0.9, 1.2, 0.24, 0.032, 0), frame="f00001", H=HPI_B)
    d = measure([(["f00001-o000"], False)], [o_b], [cal_quad()], {}, regs=[_reg("f00000", "f00001", T)], frames=("f00000", "f00001"))
    m = d["measurements"][0]
    assert m["status"] == "measured" and m["spine_height_cm"] == pytest.approx(24.0, abs=0.02)
    assert m["per_observation"][0]["hops"] == 1
    direct = measure([(["f00000-o000"], False)], [obs("f00000-o000", book_m(0.9, 1.2, 0.24, 0.032, 0))], [cal_quad()], {})["measurements"][0]
    w = lambda x: x["spine_height_interval_cm"][1] - x["spine_height_interval_cm"][0]
    assert w(m) > w(direct)
    # an untrusted registration (low inlier ratio) must not be used
    bad = measure([(["f00001-o000"], False)], [o_b], [cal_quad()], {}, regs=[_reg("f00000", "f00001", T, inl=0.05)], frames=("f00000", "f00001"))
    assert bad["measurements"][0]["status"] == "abstained" and "no_calibration_transferable_to_this_frame" in bad["measurements"][0]["abstention_reasons"]
    # beyond max_hops: chain of 4
    ids = ["f00000", "f00001", "f00002", "f00003", "f00004"]
    regs = [_reg(a, b, T) for a, b in zip(ids, ids[1:])]
    far = obs("f00004-o000", book_m(0.9, 1.2, 0.24, 0.032, 0), frame="f00004", H=np.linalg.matrix_power(T, 4) @ HPI)
    r = measure([(["f00004-o000"], False)], [far], [cal_quad()], {}, regs=regs, frames=tuple(ids))
    assert r["measurements"][0]["status"] == "abstained"


def test_instance_observations_are_aggregated_not_summed_and_disagreement_abstains():
    T = np.array([[1, 0, 100.0], [0, 1, 0], [0, 0, 1]])
    a = obs("f00000-o000", book_m(0.9, 1.2, 0.24, 0.032, 0))
    b = obs("f00001-o000", book_m(0.9, 1.2, 0.24, 0.032, 0), frame="f00001", H=T @ HPI)
    d = measure([(["f00000-o000", "f00001-o000"], False)], [a, b], [cal_quad()], NO_UNC, regs=[_reg("f00000", "f00001", T)], frames=("f00000", "f00001"))
    m = d["measurements"][0]
    assert m["observations_used"] == 2 and m["spine_thickness_cm"] == pytest.approx(3.2, abs=0.02)
    # corrupt second observation by 40 %
    c = obs("f00001-o000", book_m(0.9, 1.2, 0.24, 0.045, 0), frame="f00001", H=T @ HPI)
    d2 = measure([(["f00000-o000", "f00001-o000"], False)], [a, c], [cal_quad()], NO_UNC, regs=[_reg("f00000", "f00001", T)], frames=("f00000", "f00001"))
    assert d2["measurements"][0]["abstention_reasons"] == ["observations_disagree_beyond_tolerance"]


def test_shelf_run_withheld_for_uncertain_associations_and_honours_exclusions():
    ob = [obs(f"f00000-o{i:03d}", book_m(0.2 + 0.2 * i, 1.2, 0.24, 0.03 + 0.002 * i, 0)) for i in range(5)]
    inst = [([o["observation_id"]], False) for o in ob]
    d = measure(inst, ob, [cal_quad()])
    assert d["shelf_run_m"] == pytest.approx(sum(m["spine_thickness_cm"] for m in d["measurements"]) / 100, abs=1e-4)
    # one instance exposed as uncertain -> run withheld, partial sum still reported separately
    inst_u = [([o["observation_id"]], i == 2) for i, o in enumerate(ob)]
    du = measure(inst_u, ob, [cal_quad()])
    assert du["shelf_run_m"] is None and "uncertain physical associations" in du["shelf_runs"][0]["withheld_reason"]
    assert du["shelf_runs"][0]["partial_sum_thickness_m"] > 0
    # excluding the uncertain book (user exclusion) makes the run reportable again, deterministically
    present = {m["instance_id"] for m in du["measurements"] if not m["uncertain_association"]}
    runs, total = S.shelf_runs_for(du, present)
    assert total is not None and total == pytest.approx(sum(m["spine_thickness_cm"] for m in du["measurements"] if m["instance_id"] in present) / 100, abs=1e-4)
    # an unmeasured book in the region (abstained: partly occluded) withholds the run
    ob2 = copy.deepcopy(ob); ob2[1]["occlusion"] = "partial"
    d3 = measure(inst, ob2, [cal_quad()])
    assert d3["shelf_run_m"] is None


# ------------------------------------------------------------------ independence of calibration and evaluation truth
def test_calibration_file_must_declare_calibration_purpose_and_module_never_reads_eval_truth(tmp_path):
    import json
    p = tmp_path / "x.json"; p.write_text(json.dumps({"purpose": "evaluation_truth", "calibrations": []}))
    with pytest.raises(C.ContractError):
        S.load_calibrations(p)
    for mod in ("scale.py", "room.py"):
        text = (ROOT / "pipeline" / mod).read_text()
        assert not re.search(r"ground_truth|eval/annotations|eval\.score", text.replace("evaluation truth", ""))


# ------------------------------------------------------------------ deterministic area arithmetic
def test_area_arithmetic_and_units():
    assert RM.rectangle_floor_area_m2(4.2, 3.1) == pytest.approx(13.02)
    assert RM.wall_area_m2(2 * (4.2 + 3.1), 2.5, 3.2) == pytest.approx(33.3)
    assert RM.m2_to_ft2(1.0) == pytest.approx(10.7639104167)
    assert RM.polygon_area_m2([[0, 0], [4, 0], [4, 3], [2, 3], [2, 5], [0, 5]]) == pytest.approx(16.0)       # L-shape
    assert RM.polygon_area_m2([[0, 5], [2, 5], [2, 3], [4, 3], [4, 0], [0, 0]]) == pytest.approx(16.0)       # winding independent
    with pytest.raises(ValueError):
        RM.polygon_area_m2([[0, 0], [1, 1]])
    assert RM.area_block(None) == {"m2": None, "ft2": None}


def test_shelved_wall_area_only_from_distinct_user_supplied_shelf_faces():
    c1, c2 = cal_quad(1.8, 2.0), cal_quad(1.2, 2.0, cid="cal-002")
    c2["plane"] = {"id": "plane-B", "role": "shelf_face"}
    dup = cal_quad(1.8, 2.0, cid="cal-003")  # same plane id as c1: not double counted
    other = cal_quad(1.0, 1.0, cid="cal-004", plane_role="wall"); other["plane"]["id"] = "plane-C"
    area, parts = RM.shelved_wall_area_m2([{**c, "status": "valid"} for c in (c1, c2, dup, other)])
    assert area == pytest.approx(1.8 * 2.0 + 1.2 * 2.0) and len(parts) == 2 and "user-supplied" in parts[0]["provenance"]


# ------------------------------------------------------------------ room geometry (synthetic ground truth)
def lookat(cam, target):
    z = np.array(target, float) - cam; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def proj(K, R, cam, P3):
    pc = R @ (np.array(P3, float) - cam)
    uv = K @ pc
    return (uv[:2] / uv[2]).tolist()


def room_scene(L=4.2, W=3.1, Hc=2.5, f=700.0):
    """Corner view: ONE level camera sees both adjacent walls A (y=0) and B (x=L). Known ground truth: L, W, Hc, f."""
    K = np.array([[f, 0, FW / 2], [0, f, FH / 2], [0, 0, 1.0]])
    cam = np.array([0.7, 2.6, 1.25]); R = lookat(cam, [L, 0.0, 1.25])
    wa = [[0.3, 0, Hc], [L, 0, Hc], [L, 0, 0], [0.3, 0, 0]]          # visible part of wall A between its left edge and the corner
    wb = [[L, 0, Hc], [L, 2.0, Hc], [L, 2.0, 0], [L, 0, 0]]
    return K, [proj(K, R, cam, c) for c in wa], [proj(K, R, cam, c) for c in wb], (L - 0.3, 2.0)


def pitched_wall(f, L=4.2, Hc=2.5):
    """A single wall seen by a camera with pitch (finite vertical vanishing point): focal length from ONE wall."""
    K = np.array([[f, 0, FW / 2], [0, f, FH / 2], [0, 0, 1.0]])
    cam = np.array([1.8, 3.4, 1.0]); R = lookat(cam, [2.0, 0.0, 1.9])
    return [proj(K, R, cam, c) for c in [[0, 0, Hc], [L, 0, Hc], [L, 0, 0], [0, 0, 0]]]


def evidence(qa, qb, height=2.5, kind="user_supplied_ceiling_height", extra=None):
    prov = {"source": "user_marked", "actor": "claimant", "supplied_at": "2026-10-06T10:00:00+02:00"}
    ev = {"purpose": "room_evidence", "walls": [
        {"wall_id": "A", "frame_id": "f00000", "quad_px": qa, "spans_floor_to_ceiling": True, "provenance": prov},
        {"wall_id": "B", "frame_id": "f00000", "quad_px": qb, "spans_floor_to_ceiling": True, "provenance": prov}],
        "height_anchor": {"kind": kind, "height_m": height, "provenance": {"actor": "claimant", "supplied_at": "t", "utterance": "ceiling is 2.5 m",
                                                                         "independent_of_evaluation_truth": True}},
        "layout": {"declared_shape": "rectangular", "wall_order": ["A", "B"]}}
    ev.update(extra or {})
    return ev


def test_room_dimensions_and_areas_recovered_from_synthetic_corner_view():
    K, qa, qb, (la, lb) = room_scene()
    r = RM.estimate_data(evidence(qa, qb), manifest(("f00000",)), [])
    assert r["status"] == "estimated", r["abstention_reasons"]
    d = r["dimensions"]
    # the quads cover wall A 0.3..4.2 (3.9 m) and wall B 0..2.0 (2.0 m): widths follow from aspect x anchored height
    assert d["length_m"] == pytest.approx(la, rel=0.01) and d["width_m"] == pytest.approx(lb, rel=0.01) and d["height_m"] == 2.5
    assert r["areas"]["floor"]["m2"] == pytest.approx(la * lb, rel=0.02)
    assert r["areas"]["wall"]["m2"] == pytest.approx(2 * (la + lb) * 2.5, rel=0.02)
    assert r["areas"]["floor"]["ft2"] == pytest.approx(r["areas"]["floor"]["m2"] * RM.FT2_PER_M2, rel=1e-4)
    assert r["intrinsics"]["f_px"] == pytest.approx(700, rel=0.01) and r["intrinsics"]["validated"] is False
    assert r["intrinsics_candidates"][0]["source"] == "corner_two_adjacent_walls_one_frame"
    assert r["shape"]["rectangular_assumed"] is True and "NOT observed" in r["shape"]["assumption"]
    assert "user-supplied" in r["height_provenance"]["label"]


def test_level_camera_single_wall_does_not_determine_focal_length():
    K, qa, qb, _ = room_scene()
    fe = RM.focal_from_quad(qa, FW, FH, RM.DEFAULTS)
    assert fe["f_px"] is None and fe.get("degenerate") is True and fe["fronto_parallel"] is False
    pw = RM.focal_from_quad(pitched_wall(1100.0), FW, FH, RM.DEFAULTS)
    assert pw["f_px"] == pytest.approx(1100.0, rel=1e-3)


def test_room_abstains_without_anchor_walls_or_consistent_intrinsics():
    K, qa, qb, _ = room_scene()
    m2 = manifest(("f00000", "f00001"))
    assert "no_wall_evidence_supplied" in RM.estimate_data(None, m2, [])["abstention_reasons"]
    ev = evidence(qa, qb); ev["height_anchor"]["kind"] = "assumed_standard_height"
    assert "height_anchor_missing_or_invalid" in RM.estimate_data(ev, m2, [])["abstention_reasons"]
    ev = evidence(qa, qb); ev["height_anchor"]["height_m"] = 9.0
    assert "height_anchor_missing_or_invalid" in RM.estimate_data(ev, m2, [])["abstention_reasons"]
    ev = evidence(qa, qb); ev["walls"] = ev["walls"][:1]
    r = RM.estimate_data(ev, m2, [])
    assert "fewer_than_two_valid_walls" in r["abstention_reasons"] and "wall_faces_2_adjacent" in r["coverage"]["missing"]
    # the same two walls but in DIFFERENT frames with a level camera: focal length undetermined => abstain, say what to capture
    ev = evidence(qa, qb); ev["walls"][1]["frame_id"] = "f00001"
    r = RM.estimate_data(ev, m2, [])
    assert r["abstention_reasons"] == ["focal_length_not_determined_for_oblique_walls"] and "corner_view_two_walls_one_frame" in r["coverage"]["missing"]
    assert r["dimensions"]["length_m"] is None
    # an independent single-wall estimate (f = 1100) contradicting the corner estimate (700) => abstain, never average
    ev = evidence(qa, qb)
    ev["walls"].append({"wall_id": "C", "frame_id": "f00001", "quad_px": pitched_wall(1100.0), "spans_floor_to_ceiling": True,
                        "provenance": ev["walls"][0]["provenance"]})
    r = RM.estimate_data(ev, m2, [])
    assert r["abstention_reasons"] == ["focal_length_inconsistent_across_estimates"] and r["dimensions"]["length_m"] is None
    # unmarked floor/ceiling declaration is rejected
    ev = evidence(qa, qb); ev["walls"][0]["spans_floor_to_ceiling"] = False
    assert "invalid_wall_evidence" in RM.estimate_data(ev, m2, [])["abstention_reasons"]


def _fronto(aspect, h_px=600, x0=200):
    return [[x0, 100], [x0 + aspect * h_px, 100], [x0 + aspect * h_px, 100 + h_px], [x0, 100 + h_px]]


def test_room_opposite_walls_must_agree_and_polygon_needs_vertices():
    K, qa, qb, (la, lb) = room_scene()
    m4 = manifest(("f00000", "f00001", "f00002"))
    prov = {"source": "user_marked", "actor": "claimant", "supplied_at": "t"}
    ev = evidence(qa, qb)
    ev["walls"] += [{"wall_id": "C", "frame_id": "f00001", "quad_px": _fronto(la / 2.5 * 1.19), "spans_floor_to_ceiling": True, "provenance": prov},
                    {"wall_id": "D", "frame_id": "f00002", "quad_px": _fronto(lb / 2.5), "spans_floor_to_ceiling": True, "provenance": prov}]
    ev["layout"]["wall_order"] = ["A", "B", "C", "D"]
    r = RM.estimate_data(ev, m4, [])
    assert r["abstention_reasons"] == ["opposite_walls_disagree_room_not_confirmed_rectangular"] and r["areas"]["floor"]["m2"] is None
    ev["walls"][2]["quad_px"] = _fronto(la / 2.5)          # C now agrees with A
    r_ok = RM.estimate_data(ev, m4, [])
    assert r_ok["status"] == "estimated" and "cross-checked" in r_ok["shape"]["assumption"]
    ev2 = evidence(qa, qb); ev2["layout"] = {"declared_shape": "polygon", "wall_order": ["A", "B"]}
    r2 = RM.estimate_data(ev2, manifest(("f00000",)), [])
    assert r2["abstention_reasons"] == ["polygon_room_requires_observed_plan_vertices"] and r2["shape"]["kind"] == "polygon_unresolved"
    ev3 = evidence(qa, qb); ev3["layout"] = {"declared_shape": "polygon", "wall_order": ["A", "B"], "plan_vertices_m": [[0, 0], [4, 0], [4, 3], [2, 3], [2, 5], [0, 5]]}
    r3 = RM.estimate_data(ev3, manifest(("f00000",)), [])
    assert r3["status"] == "estimated" and r3["areas"]["floor"]["m2"] == pytest.approx(16.0) and r3["shape"]["rectangular_assumed"] is False


def test_openings_subtract_from_wall_area_only_when_declared_with_provenance():
    K, qa, qb, _ = room_scene()
    ev = evidence(qa, qb, extra={"openings": [{"wall_id": "A", "type": "door", "width_m": 0.9, "height_m": 2.0, "provenance": {"actor": "claimant"}},
                                              {"wall_id": "B", "type": "window", "width_m": 5.0, "height_m": 5.0, "provenance": {}}]})
    r = RM.estimate_data(ev, manifest(("f00000",)), [])
    assert r["areas"]["openings_subtracted_m2"] == pytest.approx(1.8)          # the undeclared (no provenance) window is ignored
    assert r["areas"]["wall"]["m2"] == pytest.approx(2 * (r["dimensions"]["length_m"] + r["dimensions"]["width_m"]) * 2.5 - 1.8, abs=0.01)


def test_fronto_parallel_walls_need_no_intrinsics():
    # two frontal walls, image aspect == metric aspect
    qa = [[300, 100], [1500, 100], [1500, 900], [300, 900]]          # 1200 x 800 px
    qb = [[400, 100], [1200, 100], [1200, 900], [400, 900]]          # 800 x 800 px
    ev = evidence(qa, qb); ev["walls"][1]["frame_id"] = "f00001"
    r = RM.estimate_data(ev, manifest(("f00000", "f00001")), [])
    assert r["status"] == "estimated" and r["intrinsics"]["f_px"] is None
    assert r["dimensions"]["length_m"] == pytest.approx(3.75, rel=1e-3) and r["dimensions"]["width_m"] == pytest.approx(2.5, rel=1e-3)


# ------------------------------------------------------------------ packet gating
def _line(**kw):
    base = {"id": "bk-001", "spine_height_cm": None, "spine_thickness_cm": None, "measurement": None}
    base.update(kw)
    return base


def _pkt(books=None, scale_ev=None, room=None, room_ev=None, totals=None):
    return {"books": books or [], "measurement_evidence": {"scale": scale_ev, "room": room_ev}, "room": room or {},
            "totals": totals or {}}


def test_packet_rejects_unsupported_spine_and_room_numbers_and_accepts_evidence_backed():
    assert _only(P._verify_spine(_line(spine_height_cm=24.0, spine_thickness_cm=3.2), _pkt()))
    ok_meas = {"status": "measured", "calibration_ids": ["cal-001"], "validation_state": "software_only_unvalidated",
               "per_observation": [{"observation_id": "o"}], "spine_height_interval_cm": [22.0, 26.0], "spine_thickness_interval_cm": [3.0, 3.4]}
    sc = {"calibrations": [{"calibration_id": "cal-001", "status": "valid"}]}
    assert P._verify_spine(_line(spine_height_cm=24.0, spine_thickness_cm=3.2, measurement=ok_meas), _pkt(scale_ev=sc)) == []
    assert _only(P._verify_spine(_line(spine_height_cm=24.0, spine_thickness_cm=3.2, measurement={**ok_meas, "calibration_ids": ["cal-999"]}), _pkt(scale_ev=sc)))
    assert _only(P._verify_spine(_line(spine_height_cm=40.0, spine_thickness_cm=3.2, measurement=ok_meas), _pkt(scale_ev=sc)))      # outside interval
    assert _only(P._verify_spine(_line(spine_height_cm=2.0, spine_thickness_cm=3.2, measurement={**ok_meas, "spine_height_interval_cm": [1, 3]}), _pkt(scale_ev=sc)))  # h<t
    assert _only(P._verify_room(_pkt(room={"length_m": 4.2, "width_m": 3.1, "status": "not_measured"}), manifest(), "."))
    assert _only(P._verify_room(_pkt(room={"confidence": 0.9}), manifest(), "."))
    assert P._verify_room(_pkt(room={"length_m": None, "width_m": None}), manifest(), ".") == []
    assert _only(P._verify_shelf_run(_pkt(totals={"shelf_run_m": 2.0})))


def _only(problems):
    return problems if problems else None


def test_end_to_end_packet_with_measurements_validates_and_exclusions_change_shelf_run(tmp_path):
    from pipeline import corrections as K
    ob = [obs(f"f00000-o{i:03d}", book_m(0.2 + 0.2 * i, 1.2, 0.24, 0.03 + 0.002 * i, 0)) for i in range(4)]
    for o in ob:
        o["confidence"] = 0.9
    books = [{"instance_id": f"bk-{i+1:03d}", "observation_ids": [o["observation_id"]], "frame_ref": "f00000", "bbox_xyxy": o["bbox_xyxy"],
              "kind": "book", "confidence": 0.9, "uncertain_association": False, "identity": {"decision": "no_match"}} for i, o in enumerate(ob)]
    invd = {"data": {"books": books, "items": [], "registrations": [], "counts": {}}, "fixture": False}
    man = manifest(); (tmp_path / "frames").mkdir(); (tmp_path / "frames" / "f00000.jpg").write_bytes(b"x")
    # bboxes must be inside the frame for packet verification
    for b in books:
        b["bbox_xyxy"] = [max(5, min(FW - 5, v)) for v in b["bbox_xyxy"]]
    scale = S.measure_data(invd, man, {o["observation_id"]: o for o in ob}, [cal_quad()], {})
    assert scale["summary"]["measured"] == 4 and scale["shelf_run_m"] is not None
    cfg = {"run_id": "t", "scale": scale}
    data = P.build_data(invd, man, cfg)
    env = C.envelope("t", "packet", data, status="partial")
    assert P.verify_packet(env, man, tmp_path) == [], P.verify_packet(env, man, tmp_path)
    full = data["totals"]["shelf_run_m"]
    assert full == pytest.approx(scale["shelf_run_m"], abs=1e-4)
    assert all(b["measurement"]["status"] == "measured" for b in data["books"])
    assert any(r["code"] == "no_metric_scale" for r in data["review_queue"]) is False
    # user excludes one book -> run recomputed from remaining measured books only
    led = [K.new_entry("c001", "exclude", {"observation_ids": [ob[1]["observation_id"]]}, {"reason": "not mine"})]
    d2 = P.build_data(invd, man, {**cfg, "corrections": led})
    assert d2["totals"]["book_count"] == 3 and d2["totals"]["shelf_run_m"] == pytest.approx(
        sum(b["spine_thickness_cm"] for b in d2["books"]) / 100, abs=1e-4) and d2["totals"]["shelf_run_m"] < full
    assert P.verify_packet(C.envelope("t", "packet", d2, status="partial"), man, tmp_path) == []
    # tamper: a made-up dimension without evidence is rejected
    bad = copy.deepcopy(data); bad["books"][0]["measurement"]["calibration_ids"] = ["cal-404"]
    assert any("calibration" in p for p in P.verify_packet(C.envelope("t", "packet", bad, status="partial"), man, tmp_path))
    bad2 = copy.deepcopy(data); bad2["room"]["length_m"] = 4.0
    assert P.verify_packet(C.envelope("t", "packet", bad2, status="partial"), man, tmp_path)


def test_changed_calibration_invalidates_the_resume_digest(tmp_path):
    from pipeline.run import Run
    (tmp_path / "run").mkdir()
    r = Run(None, tmp_path / "run", {"no_provider": True})
    d0 = r.input_digest("scale"); d0r = r.input_digest("room"); dp = r.input_digest("packet")
    (tmp_path / "run" / "calibration.json").write_text('{"purpose": "calibration", "calibrations": []}')
    assert r.input_digest("scale") != d0 and r.input_digest("packet") != dp and r.input_digest("room") != d0r
    d1 = r.input_digest("scale")
    (tmp_path / "run" / "calibration.json").write_text('{"purpose": "calibration", "calibrations": [{"x": 1}]}')
    assert r.input_digest("scale") != d1


def test_strongly_oblique_camera_projection_still_recovers_metric_sizes():
    """Real perspective (camera yawed ~50 deg to the shelf face, pitched slightly): sizes must still come out metric."""
    f = 1000.0
    K = np.array([[f, 0, FW / 2], [0, f, FH / 2], [0, 0, 1.0]])
    cam = np.array([-1.6, 2.4, 1.2]); R = lookat(cam, [0.9, 0.0, 1.3])
    P3 = lambda xy: [proj(K, R, cam, [x, 0.0, z]) for x, z in xy]
    quad = P3([(0, 2.0), (1.8, 2.0), (1.8, 0.0), (0, 0.0)])               # TL,TR,BR,BL as seen (z up)
    assert S.oblique_index(np.array(quad)) > 1.15                          # genuinely keystoned
    cal = {"calibration_id": "cal-001", "kind": "plane_quad", "frame_id": "f00000", "image_points_px": quad,
           "metric": {"width_m": 1.8, "height_m": 2.0, "tolerance_pct": 0.0}, "plane": {"id": "p", "role": "shelf_face"}, "provenance": PROV}
    corners = lambda cx, cz, h, t: [(cx - t / 2, cz + h / 2), (cx + t / 2, cz + h / 2), (cx + t / 2, cz - h / 2), (cx - t / 2, cz - h / 2)]
    poly = np.array(P3(corners(1.1, 1.0, 0.26, 0.04)))
    bb = [*poly.min(0), *poly.max(0)]
    o = {"observation_id": "f00000-o000", "frame_id": "f00000", "polygon_xy": poly.tolist(), "bbox_xyxy": bb, "orientation": "upright", "occlusion": "none"}
    d = measure([(["f00000-o000"], False)], [o], [cal])
    m = d["measurements"][0]
    assert d["calibrations"][0]["status"] == "valid", d["calibrations"][0]["problems"]
    assert m["status"] == "measured"
    assert m["spine_height_cm"] == pytest.approx(26.0, abs=0.02) and m["spine_thickness_cm"] == pytest.approx(4.0, abs=0.02)
    # a naive uniform pixel scale from the unit's width would be badly wrong here (this is why a 4-point plane map is used)
    naive = 1.8 / np.linalg.norm(np.array(quad[1]) - np.array(quad[0]))
    naive_h = np.linalg.norm(poly[0] - poly[3]) * naive * 100
    assert abs(naive_h - 26.0) / 26.0 > 0.05


def test_stage_entry_points_write_versioned_artifacts_that_the_orchestrator_validates(tmp_path):
    ob = [obs(f"f00000-o{i:03d}", book_m(0.3 + 0.25 * i, 1.2, 0.24, 0.03, 0)) for i in range(3)]
    inv_env = C.envelope("t", "dedup", {"books": [{"instance_id": f"bk-{i+1:03d}", "observation_ids": [o["observation_id"]], "uncertain_association": False}
                                                   for i, o in enumerate(ob)], "items": [], "registrations": [], "counts": {}}, status="ok")
    C.write_json(tmp_path / "manifest.json", manifest()); C.write_json(tmp_path / "inventory.json", inv_env)
    C.write_json(tmp_path / "candidates.json", C.envelope("t", "detect", {"observations": ob}, status="ok"))
    C.write_json(tmp_path / "calibration.json", {"purpose": "calibration", "calibrations": [cal_quad()]})
    p = S.measure(tmp_path / "inventory.json", tmp_path / "manifest.json", tmp_path, {})
    env = C.read_json(p)
    assert p.name == "scale.json" and env["stage"] == "scale" and env["status"] == "ok" and env["metrics"]["request_count"] == 0
    assert env["data"]["summary"]["measured"] == 3 and len(env["data"]["inventory_sha256"]) == 64
    assert C.validate_artifact("scale", env, manifest=manifest()) == []
    # the room stage with no evidence is a valid, explicit abstention (never a fabricated room)
    rp = RM.estimate(tmp_path / "manifest.json", tmp_path, {})
    renv = C.read_json(rp)
    assert renv["stage"] == "room" and renv["status"] == "partial" and renv["data"]["dimensions"]["length_m"] is None
    assert C.validate_artifact("room", renv, manifest=manifest()) == []
