"""pipeline.auto_scale: automatic metric scale (source A catalogue dimensions, B standard-format prior) and spine sizes from a synthetic 3D reconstruction.
Synthetic geometry only: these show the method recovers a KNOWN scale and abstains with reasons; they are not accuracy evidence against tape."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pipeline import auto_scale as AS
from pipeline import contracts as C

W, H, F = 1920, 1080, 1000.0
SCALE = 35.0                                    # true cm per reconstruction unit used to build the synthetic scene


def isbn13(stem9: str) -> str:
    core = "978" + stem9
    return core + str((10 - sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(core)) % 10) % 10)


def project(p, cam_z=5.0):
    """Camera at (0,0,cam_z) looking at the plane z=0, image y down = world -y (world up = +y)."""
    return [F * p[0] / cam_z + W / 2, -F * p[1] / cam_z + H / 2]


def book_polygon(x, y, h, t, cam_z=5.0):
    pts = [(x - t / 2, y + h / 2), (x + t / 2, y + h / 2), (x + t / 2, y - h / 2), (x - t / 2, y - h / 2)]
    return [project(p, cam_z) for p in pts]


def recon(books, frames=3, status="ok", cam_z=5.0):
    cams = []
    for i in range(frames):
        cams.append({"frame_id": f"f{i:05d}", "center": [0.0, 0.0, cam_z], "rotation_world_to_cam": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                     "focal_px": F, "width_px": W, "height_px": H})
    markers = []
    for gid, (x, y, h, t) in books.items():
        poly = book_polygon(x, y, h, t, cam_z)
        markers.append({"group_id": gid, "kind": "book", "sightings_total": frames, "xyz": [x, y, 0.0],
                        "sightings": [{"frame_id": f"f{i:05d}", "observation_id": f"f{i:05d}-o{gid}", "bbox_xyxy": [0, 0, 1, 1], "polygon_xy": poly} for i in range(frames)]})
    return {"status": status, "cameras": cams, "markers": markers}


class FakeCatalogue:
    def __init__(self, works):
        self.works, self.requests = works, 0

    def editions(self, work, **_):
        if work not in self.works:
            raise RuntimeError("catalogue_offline_cache_miss")
        return {"entries": self.works[work], "size": len(self.works[work]), "urls": [f"https://openlibrary.org{work}/editions.json"], "retrieved_at": "2026-10-06T00:00:00+00:00"}

    def close(self):
        pass


def edition(key, stem, dims, fmt="Paperback", lang="eng", date="2015"):
    return {"key": key, "title": "T", "isbn_13": [isbn13(stem)], "languages": [{"key": f"/languages/{lang}"}], "physical_format": fmt, "publish_date": date,
            "physical_dimensions": dims, "publishers": ["P"]}


# ---------------------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("text,h,w,t", [
    ("9.3 x 5.6 x 0.4 inches", 23.62, 14.22, 1.02),
    ("23 x 15 x 4 centimeters", 23.0, 15.0, 4.0),
    ("23 x 15 x  centimeters", 23.0, 15.0, None),
    ("216 x 140 x 25 mm", 21.6, 14.0, 2.5),
    ("9 1/4 x 6 inches", 23.5, 15.24, None),
    ("5.6 x 9.3 x 0.4 in", 23.62, 14.22, 1.02),                       # catalogue order is not height-first
])
def test_parse_physical_dimensions(text, h, w, t):
    d = AS.parse_physical_dimensions(text)
    assert d["height_cm"] == pytest.approx(h, abs=0.02) and d["width_cm"] == pytest.approx(w, abs=0.02)
    assert (d["thickness_cm"] is None) if t is None else d["thickness_cm"] == pytest.approx(t, abs=0.02)
    assert d["raw"] == text.strip()


@pytest.mark.parametrize("text", [None, "", 12, "9 x 5", "tall", "9 inches", "0 x 5 x 1 cm", "1000 x 800 x 3 cm", "2 x 1 x 1 cm", "9 x 5 x 1 x 2 cm"])
def test_unusable_dimension_strings_are_none_never_zero(text):
    assert AS.parse_physical_dimensions(text) is None


# ---------------------------------------------------------------------------------------- catalogue height by the equivalent-edition rule
def test_catalogue_height_uses_the_ranked_equivalent_edition_and_reports_spread():
    entries = [edition("/books/OL1M", "000000001", "9 x 6 x 1 inches", date="2010"),
               edition("/books/OL2M", "000000002", "9.1 x 6 x 1 inches", date="2019"),                       # newest: rank 1
               edition("/books/OL3M", "000000003", "8 x 5 x 1 inches", fmt="Audio CD", date="2020"),         # excluded format
               edition("/books/OL4M", "000000004", "9 x 6 x 1 inches", lang="spa", date="2021"),            # other language: never switched
               edition("/books/OL5M", "000000005", None, date="2022")]                                       # newest but no dimensions: skipped for dims
    c = AS.catalogue_height(entries, "en", this_year=2026)
    assert c["height_cm"] == pytest.approx(np.median([9 * 2.54, 9.1 * 2.54]), abs=0.05)
    keys = [e["edition_key"] for e in c["editions"]]
    assert keys == ["/books/OL2M", "/books/OL1M"] and c["first_ranked_edition_has_dimensions"] is False and c["edition_spread_pct"] < 2
    assert all(e["raw"] and e["isbn"] for e in c["editions"])


def test_catalogue_height_rejects_works_whose_editions_disagree_and_works_without_dimensions():
    wide = [edition("/books/OL1M", "000000001", "7 x 4 x 1 inches"), edition("/books/OL2M", "000000002", "9.3 x 5.6 x 1 inches")]
    c = AS.catalogue_height(wide, "en", this_year=2026)
    assert c["height_cm"] is None and c["reason"].startswith("same_family_editions_disagree") and c["rejected_height_cm"] > 0
    none = AS.catalogue_height([edition("/books/OL1M", "000000001", None)], "en", this_year=2026)
    assert none["height_cm"] is None and "no_equivalent_edition_records_physical_dimensions" in none["reason"]
    assert AS.catalogue_height([], "en")["reason"] == "no_equivalent_edition_candidates"


# ---------------------------------------------------------------------------------------- geometry
def test_sighting_size_recovers_the_known_plane_size_and_abstains_when_clipped_or_grazing():
    rc = recon({"bk-1": (0.1, 0.2, 0.62, 0.075)})
    views, sizes = AS.views_from_recon(rc), AS.frame_sizes(rc)
    g = AS.group_sizes(rc["markers"], views, sizes, {**AS.DEFAULTS})["bk-1"]
    assert g["status"] == "ok" and g["h_units"] == pytest.approx(0.62, rel=1e-6) and g["t_units"] == pytest.approx(0.075, rel=1e-6)
    clipped = recon({"bk-2": (0.1, 0.2, 0.62, 0.075)})
    for s in clipped["markers"][0]["sightings"]:
        s["polygon_xy"][0][1] = -5.0                                                  # off the top of the frame
    g2 = AS.group_sizes(clipped["markers"], views, sizes, {**AS.DEFAULTS})["bk-2"]
    assert g2["status"] == "abstained" and "polygon_clipped_by_frame_edge" in g2["reason"]
    square = recon({"bk-3": (0.0, 0.0, 0.1, 0.09)})
    g3 = AS.group_sizes(square["markers"], views, sizes, {**AS.DEFAULTS})["bk-3"]
    assert g3["status"] == "abstained" and "near_square_polygon" in g3["reason"]


def test_sightings_that_disagree_abstain_with_the_spread():
    rc = recon({"bk-1": (0.0, 0.0, 0.6, 0.07)}, frames=3)
    rc["markers"][0]["sightings"][2]["polygon_xy"] = book_polygon(0.0, 0.0, 0.9, 0.07)
    g = AS.group_sizes(rc["markers"], AS.views_from_recon(rc), AS.frame_sizes(rc), {**AS.DEFAULTS})["bk-1"]
    assert g["status"] == "abstained" and g["reason"].startswith("sightings_disagree_on_height")


# ---------------------------------------------------------------------------------------- scale fit
def samples(ratios, spread=0.0):
    return [{"ratio": r, "h_units": 0.6, "catalogue_height_cm": r * 0.6, "edition_spread_pct": spread} for r in ratios]


def test_fit_scale_is_a_robust_median_that_rejects_an_outlier():
    fit = AS.fit_scale(samples([35.0, 34.0, 36.0, 35.5, 34.5, 60.0]), {**AS.DEFAULTS})
    assert fit["cm_per_unit"] == pytest.approx(35.0, rel=0.02) and fit["n_inliers"] == 5 and fit["inlier_mask"][-1] is False
    assert fit["relative_uncertainty_pct"] >= AS.DEFAULTS["catalogue_systematic_pct"] - 0.1
    assert fit["leave_one_out_catalogue_height_abs_error_pct"]["n"] == 5


def test_fit_scale_abstains_with_a_reason_when_too_few_or_inconsistent():
    few = AS.fit_scale(samples([35.0, 34.0]), {**AS.DEFAULTS})
    assert few["cm_per_unit"] is None and "only_2_identified_books" in few["reason"]
    bad = AS.fit_scale(samples([20.0, 35.0, 50.0, 70.0, 28.0, 90.0]), {**AS.DEFAULTS})
    assert bad["cm_per_unit"] is None and ("scatter" in bad["reason"] or "agree" in bad["reason"])


# ---------------------------------------------------------------------------------------- end to end
HEIGHTS_CM = {"bk-1": 24.0, "bk-2": 21.0, "bk-3": 23.0, "bk-4": 22.5, "bk-5": 19.0, "bk-6": 25.0}


def scene(n=6):
    books = {gid: (0.15 * i - 0.4, 0.9 * (i % 2), h / SCALE, 3.0 / SCALE) for i, (gid, h) in enumerate(list(HEIGHTS_CM.items())[:n])}
    return recon(books)


def works_for(heights, wrong=None):
    works, ident = {}, []
    for i, (gid, h) in enumerate(heights.items()):
        w = f"/works/OL{i + 1}W"
        cm = h if gid != wrong else h * 1.8                                           # a wrongly identified book (catalogue height of another book)
        works[w] = [edition(f"/books/OL{i + 1}M", f"{i + 1:09d}", f"{cm:.1f} x {cm * 0.65:.1f} x 2 centimeters")]
        ident.append({"group_id": gid, "work": w, "title": f"Title {i}", "visible_text": "THE OLD MAN AND THE SEA", "language_code": "en"})
    return FakeCatalogue(works), ident


def test_scale_source_a_recovers_the_scale_and_spine_heights_with_provenance():
    cat, ident = works_for(HEIGHTS_CM)
    d = AS.measure_data(scene(), ident, cat)
    assert d["status"] == "ok" and d["scale"]["source"] == "catalogue_physical_dimensions"
    assert d["scale"]["cm_per_unit"] == pytest.approx(SCALE, rel=1e-6)
    used = [s for s in d["scale"]["sources"] if s["used"]]
    assert len(used) == 6 and all(s["catalogue"]["editions"][0]["raw"] and s["catalogue"]["editions"][0]["isbn"] and s["catalogue_urls"] for s in used)
    sp = {s["group_id"]: s for s in d["spines"]}
    for gid, h in HEIGHTS_CM.items():
        assert sp[gid]["status"] == "measured" and sp[gid]["spine_height_cm"] == pytest.approx(h, abs=0.06)
        assert sp[gid]["spine_thickness_cm"] == pytest.approx(3.0, abs=0.06)
        assert sp[gid]["scale_source"] == "catalogue_physical_dimensions" and sp[gid]["height_relative_uncertainty_pct"] >= 4.0
        assert sp[gid]["validation_state"] == "software_only_unvalidated"
    assert d["summary"]["measured"] == 6 and d["shelf"]["shelf_run_m"] > 0 and d["shelf"]["rows"] == 2


def test_a_misidentified_book_is_rejected_as_an_outlier_not_averaged_in():
    cat, ident = works_for(HEIGHTS_CM, wrong="bk-3")
    d = AS.measure_data(scene(), ident, cat)
    assert d["scale"]["cm_per_unit"] == pytest.approx(SCALE, rel=0.01)
    bad = next(s for s in d["scale"]["sources"] if s["group_id"] == "bk-3")
    assert bad["used"] is False and bad["reason"] == "outlier_vs_median_scale"


def test_no_identified_books_falls_back_to_the_labelled_prior_with_wide_uncertainty():
    big = {f"bk-{i}": (0.05 * i - 0.5, 0.0, (20.0 + (i % 5)) / SCALE, 3.0 / SCALE) for i in range(14)}
    d = AS.measure_data(recon(big), [], None)
    assert d["status"] == "prior_only" and d["scale"]["source"] == "standard_format_prior" and d["scale"]["not_a_measurement"] is True
    assert d["scale"]["relative_uncertainty_pct"] >= 15.0 and d["scale"]["scale_source_a_attempt"]["reason"].startswith("no_identified_books")
    m = [s for s in d["spines"] if s["status"] == "measured"]
    assert m and all(s["scale_source"] == "standard_format_prior" and s["height_relative_uncertainty_pct"] >= 15.0 for s in m)
    off = AS.measure_data(recon(big), [], None, {"allow_prior": False})
    assert off["status"] == "abstained" and all(s["spine_height_cm"] is None and s["reason"].startswith("no_metric_scale") for s in off["spines"])


def test_unusable_reconstruction_abstains_everything_with_a_reason():
    d = AS.measure_data({"status": "failed", "cameras": [], "markers": []}, [], None)
    assert d["status"] == "abstained" and d["scale"]["cm_per_unit"] is None and d["spines"] == [] and "3d_reconstruction_unusable" in d["reason"]


def test_offline_catalogue_miss_is_a_reason_not_a_crash():
    cat, ident = works_for(HEIGHTS_CM)
    cat.works.pop("/works/OL1W")
    d = AS.measure_data(scene(), ident, cat)
    src = next(s for s in d["scale"]["sources"] if s["group_id"] == "bk-1")
    assert src["used"] is False and src["reason"].startswith("catalogue_unavailable") and d["scale"]["n_inliers"] == 5


def test_identified_books_only_accepts_catalogue_works():
    books = [{"id": "bk-1", "status": "identified", "identity_source": "catalogue_match", "identity_source_url": "https://openlibrary.org/works/OL1W", "title": "A", "language": "en"},
             {"id": "bk-2", "status": "identified", "identity_source": "user", "identity_source_url": None},
             {"id": "bk-3", "status": "unidentified"},
             {"id": "bk-4", "status": "identified", "identity_source": "catalogue_match", "identity_source_url": "https://openlibrary.org/books/OL1M"}]
    assert [b["group_id"] for b in AS.identified_from_books(books)] == ["bk-1"]


def test_run_writes_a_new_versioned_artifact_and_refuses_to_overwrite(tmp_path):
    rc = scene()
    rdir = tmp_path / "r_recon3d_v1"
    rdir.mkdir()
    (rdir / "recon3d.json").write_text(json.dumps(rc))
    src = tmp_path / "r"
    src.mkdir()
    out = tmp_path / "r_measure_v1"
    p = AS.run(src, rdir, out, identified=[], online=False)
    env = C.read_json(p)
    assert env["stage"] == "auto_measure" and env["data"]["schema"] == AS.SCHEMA and env["data"]["source_run"] == "r"
    with pytest.raises(SystemExit):
        AS.run(src, rdir, out, identified=[], online=False)
    assert AS.next_output(tmp_path, "r").name == "r_measure_v2"


def test_old_reconstructions_get_their_polygons_from_candidates_not_invented():
    rc = scene(2)
    poly = rc["markers"][0]["sightings"][0].pop("polygon_xy")
    for s in rc["markers"][0]["sightings"][1:]:
        s.pop("polygon_xy")
    cands = {"observations": [{"observation_id": rc["markers"][0]["sightings"][0]["observation_id"], "polygon_xy": poly}]}
    assert AS.enrich_polygons(rc, cands) == 1
    assert rc["markers"][0]["sightings"][0]["polygon_xy"] == poly and "polygon_xy" not in rc["markers"][0]["sightings"][1]
    assert AS.enrich_polygons(rc, None) == 0


def test_occluded_or_stacked_sightings_do_not_measure():
    rc = scene(2)
    for s in rc["markers"][0]["sightings"]:
        s["occlusion"] = "partial"
    for s in rc["markers"][1]["sightings"]:
        s["orientation"] = "stacked"
    g = AS.group_sizes(rc["markers"], AS.views_from_recon(rc), AS.frame_sizes(rc), {**AS.DEFAULTS})
    assert g["bk-1"]["status"] == "abstained" and "occluded_partial" in g["bk-1"]["reason"]
    assert g["bk-2"]["status"] == "abstained" and "stacked_orientation_axes_ambiguous" in g["bk-2"]["reason"]


def test_thickness_is_withheld_with_a_reason_when_spines_are_too_narrow_or_the_library_median_is_implausible():
    narrow = recon({f"bk-{i}": (0.05 * i - 0.5, 0.0, (20.0 + i % 5) / SCALE, 0.4 / SCALE) for i in range(14)})          # 0.4 cm thick: ~1.6 px wide at 5 units / f 1000
    d = AS.measure_data(narrow, [], None)
    m = [s for s in d["spines"] if s["status"] == "measured"]
    assert m and all(s["spine_height_cm"] and s["spine_thickness_cm"] is None and "thickness" in s["thickness_reason"] for s in m)
    thin_but_wide_enough = recon({f"bk-{i}": (0.05 * i - 0.5, 0.0, (20.0 + i % 5) / SCALE, 0.9 / SCALE) for i in range(14)}, cam_z=1.0)    # wide in pixels, library median 0.9 cm
    d2 = AS.measure_data(thin_but_wide_enough, [], None)
    m2 = [s for s in d2["spines"] if s["status"] == "measured"]
    assert m2 and all(s["spine_thickness_cm"] is None and "library_median_thickness" in s["thickness_reason"] for s in m2)


def test_too_few_catalogue_books_are_shown_as_a_cross_check_of_the_prior_not_used_as_the_scale():
    big = {f"bk-{i}": (0.05 * i - 0.5, 0.0, (20.0 + (i % 5)) / SCALE, 3.0 / SCALE) for i in range(14)}
    works = {"/works/OL1W": [edition("/books/OL1M", "000000001", "20 x 13 x 2 centimeters")], "/works/OL2W": [edition("/books/OL2M", "000000002", "21 x 14 x 2 centimeters")]}
    ident = [{"group_id": f"bk-{i}", "work": f"/works/OL{i + 1}W", "title": "T", "visible_text": "THE OLD MAN AND THE SEA", "language_code": "en"} for i in range(2)]
    d = AS.measure_data(recon(big), ident, FakeCatalogue(works))
    assert d["scale"]["source"] == "standard_format_prior" and d["scale"]["catalogue_cross_check"]["n_books"] == 2
    assert d["scale"]["scale_source_a_attempt"]["partial_cm_per_unit"] == pytest.approx(SCALE, rel=0.06)
    assert d["scale"]["not_a_measurement"] is True and d["scale"]["catalogue_cross_check"]["note"].startswith("too few books")
