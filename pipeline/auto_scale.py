"""Automatic metric scale and spine dimensions from the sweep alone. Nobody is ever asked to measure, state or tap anything.

    python -m pipeline.auto_scale --source-run runs/<run> --recon3d runs/<run>_recon3d_vN --out runs/<run>_measure_vN [--identified FILE] [--online]

SCALE SOURCE A (primary): identified books with a catalogue physical size. For every book resolved to an Open Library WORK the equivalent edition is chosen by
the SAME rule as pipeline/price.py (same language as the spine text, paperback family, most recent edition with a checksum-valid ISBN); the catalogue height
(largest of ``physical_dimensions``, cm) of that edition (else of the next ranked same-family editions that record one) is compared with the book's height in
the run's UNSCALED 3D reconstruction (recon3d.json). cm per unit = median over books of catalogue height / reconstructed height, outliers rejected by MAD,
every source cited (edition, ISBN, URL, retrieval time, raw string). Fewer than ``min_books`` consistent books -> no scale from A.

SCALE SOURCE B (fallback, off the books themselves): a standard-format PRIOR (median trade/mass-market/hardcover height). It is NOT a measurement: the
output labels it ``standard_format_prior`` with a deliberately wide stated uncertainty and never claims the 15% gate.

GEOMETRY: the 3D reconstruction (pipeline/recon3d.py: sparse SfM, one shared camera with estimated focal length, shelf plane fitted and aligned to z = 0, spine-up
= +y, UNSCALED) gives every book group's detection polygons in up to 12 registered frames. Each polygon corner is cast as a camera ray onto the shelf plane;
the long edge of the resulting quad is the spine height, the short edge the thickness; median over sightings. Spines are a few cm in front of the fitted
plane, so a plane-offset term is added. Cut-off (clipped) polygons, near-square or grazing-angle sightings abstain with a reason; values are never 0, only null.

Every number carries method, scale source, stated relative uncertainty (a sum of DECLARED terms, not a statistical guarantee) and
``validation_state: software_only_unvalidated`` until compared with tape. Evaluation truth is never read.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import contracts as C
from pipeline.recon3d import MIN_PLANE_COS, View, hit_plane, pinhole

SCHEMA = "auto_measure_v1"
VALIDATION = "software_only_unvalidated: no tape-measured comparison exists"
DEFAULTS = {
    "min_books": 3,                  # consistent catalogue-scaled books needed for scale source A
    "ratio_band": 0.18,              # inlier band on |log ratio - median| (the larger of this and 3 MAD)
    "max_scatter_pct": 25.0,         # robust scatter (MAD, %) of the inlier ratios above this: the books do not agree on a scale
    "min_inlier_fraction": 0.5,      # inliers / candidates below this -> the books disagree, abstain
    "catalogue_systematic_pct": 5.0,  # the owned edition may not be the catalogue's equivalent edition (declared, unvalidated)
    "max_edition_spread_pct": 20.0,  # a work whose same-family editions disagree on height by more is not used as a scale reference
    "top_editions": 5,
    "plane_offset_pct": 4.0,         # spines sit in front of the fitted plane (declared)
    "pixel_error_px": 2.0,           # polygon edge localisation (declared)
    "min_cos": 0.25,                 # ray/plane grazing limit for a measuring sighting
    "min_aspect": 1.4,               # long/short polygon edge below this: no reliable axis -> abstain
    "spine_aspect_min": 2.6,         # thickness only reported when the polygon looks like a spine, not a cover
    "max_thickness_pixel_err": 0.25,  # thickness needs the spine at least ~8 px wide (2 px edge error / width): narrower -> thickness abstains
    "library_thickness_cm": (1.5, 6.0),  # sanity band for the MEDIAN thickness of all measured spines (declared prior): outside -> thickness is withheld run-wide
    "max_sighting_spread": 0.20,     # relative spread of per-sighting heights above this -> the book abstains
    "border_margin_px": 3.0,
    "height_range_cm": (8.0, 50.0),
    "allow_prior": True,
    "budget_s": 60.0,                # wall-clock budget for catalogue lookups (cache first, 1 request/s when online); the rest abstain with a reason
    "max_fit_books": 30,             # stop once this many books carry a catalogue height: more adds nothing to a median
}
PRIOR = {"median_height_cm": 22.0, "relative_uncertainty_pct": 15.0,
         "basis": ("standard-format PRIOR: US/EU trade paperbacks 20-23 cm, hardcovers 22-25 cm, mass-market 17-18 cm (median ~22 cm). "
                   "An assumption about the whole library, not a measurement; the stated 15% is itself unvalidated")}
CM_PER = {"cm": 1.0, "centimeter": 1.0, "centimeters": 1.0, "centimetre": 1.0, "centimetres": 1.0, "mm": 0.1, "millimeter": 0.1, "millimeters": 0.1,
          "in": 2.54, "inch": 2.54, "inches": 2.54, '"': 2.54}
_UNIT = re.compile(r"(centimet(?:er|re)s?|millimet(?:er|re)s?|inches|inch|cm|mm|in\b|\")", re.I)
_NUM = re.compile(r"(?<![\d.])(\d+\s+\d+/\d+|\d+/\d+|\d+(?:[.,]\d+)?)")


# --------------------------------------------------------------------------------------------- catalogue dimensions
def _number(token: str) -> float:
    token = token.strip().replace(",", ".")
    if "/" in token:
        parts = token.split()
        whole, frac = (float(parts[0]), parts[1]) if len(parts) == 2 else (0.0, parts[0])
        a, b = frac.split("/")
        return whole + float(a) / float(b)
    return float(token)


def parse_physical_dimensions(text: Any) -> dict[str, Any] | None:
    """'9.3 x 5.6 x 0.4 inches' / '23 x 15 x  centimeters' / '216 x 140 mm' -> {height_cm, width_cm, thickness_cm|None, raw}; None when the text has no unit,
    fewer than two sizes, a non-positive size or a height outside the plausible range of a book. Catalogue order is inconsistent, so the largest size is the height."""
    if not isinstance(text, str):
        return None
    unit = _UNIT.search(text)
    if not unit:
        return None
    factor = CM_PER[unit.group(1).lower()]
    nums = [_number(t) for t in _NUM.findall(text[:unit.start()] if unit.start() > 3 else text)]
    nums = [n * factor for n in nums if n > 0]
    if len(nums) < 2 or len(nums) > 3:
        return None
    nums.sort(reverse=True)
    lo, hi = DEFAULTS["height_range_cm"]
    if not (lo <= nums[0] <= hi):
        return None
    return {"height_cm": round(nums[0], 2), "width_cm": round(nums[1], 2), "thickness_cm": round(nums[2], 2) if len(nums) == 3 else None, "raw": text.strip()}


def catalogue_height(entries: list[dict[str, Any]], language: str, *, family: str = "paperback", this_year: int | None = None,
                     top_editions: int = 5, max_spread_pct: float = 20.0) -> dict[str, Any]:
    """Catalogue height (cm) of one work through the work-level EQUIVALENT-edition rule of pipeline/price.py: rank edition records (same language, family, newest checksum-valid
    ISBN first); take the dimensions recorded by the best ranked editions (up to ``top_editions``), median height, spread reported. -> {height_cm|None, reason?, ...}."""
    from pipeline import price as PR
    cands, _rejected = PR.rank_equivalent_editions(entries, language, family, this_year, unrecorded_ok=True)
    by_key = {e.get("key"): e for e in entries}
    used = []
    for rank, c in enumerate(cands):
        dims = parse_physical_dimensions((by_key.get(c["edition_key"]) or {}).get("physical_dimensions"))
        if dims:
            used.append({"rank": rank, "edition_key": c["edition_key"], "isbn": c["isbn"], "publish_date": c.get("publish_date"), "physical_format": c.get("physical_format"),
                         "publisher": c.get("publisher"), **dims})
        if len(used) >= top_editions:
            break
    if not cands:
        return {"height_cm": None, "reason": "no_equivalent_edition_candidates"}
    if not used:
        return {"height_cm": None, "reason": "no_equivalent_edition_records_physical_dimensions", "editions_considered": len(cands)}
    heights = [u["height_cm"] for u in used]
    med = float(np.median(heights))
    spread = (max(heights) - min(heights)) / med * 100.0
    out = {"height_cm": round(med, 2), "edition_spread_pct": round(spread, 1), "editions_with_dimensions": len(used), "editions_considered": len(cands),
           "editions": used, "rule": PR.EQUIVALENT_RULE, "first_ranked_edition_has_dimensions": bool(used and used[0]["rank"] == 0)}
    if spread > max_spread_pct:
        out.update(height_cm=None, reason=f"same_family_editions_disagree_on_height:{spread:.0f}%", rejected_height_cm=round(med, 2))
    return out


# --------------------------------------------------------------------------------------------- reconstruction access
def views_from_recon(recon: dict[str, Any]) -> dict[str, View]:
    out = {}
    for c in recon.get("cameras") or []:
        out[c["frame_id"]] = View(c["frame_id"], c["center"], c["rotation_world_to_cam"], pinhole(c["focal_px"], c["width_px"] / 2.0, c["height_px"] / 2.0))
    return out


def frame_sizes(recon: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {c["frame_id"]: (int(c["width_px"]), int(c["height_px"])) for c in recon.get("cameras") or []}


def sighting_size(view: View, poly: Any, width: int, height: int, opts: dict[str, Any]) -> dict[str, Any]:
    """One polygon on the shelf plane: {h, t, aspect, ...} in reconstruction units, or {reason}."""
    p = np.asarray(poly, float).reshape(-1, 2) if poly is not None else None
    if p is None or len(p) != 4 or not np.isfinite(p).all():
        return {"reason": "no_oriented_polygon"}
    m = opts["border_margin_px"]
    if (p[:, 0] < m).any() or (p[:, 1] < m).any() or (p[:, 0] > width - m).any() or (p[:, 1] > height - m).any():
        return {"reason": "polygon_clipped_by_frame_edge"}
    pts = []
    for q in p:
        origin, direction = view.ray(q)
        hit = hit_plane(origin, direction, max(opts["min_cos"], MIN_PLANE_COS))
        if hit is None:
            return {"reason": "ray_grazes_or_misses_shelf_plane"}
        pts.append(hit[:2])
    pts = np.asarray(pts)
    e = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    a, b = (e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0
    if min(a, b) < 1e-9 or abs(e[0] - e[2]) / max(a, 1e-9) > 0.4 or abs(e[1] - e[3]) / max(b, 1e-9) > 0.4:
        return {"reason": "degenerate_projected_quad"}
    hh, tt = (a, b) if a >= b else (b, a)
    # pixel-edge localisation error relative to the shorter polygon side (declared term)
    side = sorted(float(np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4))
    return {"h": hh, "t": tt, "aspect": hh / tt, "pixel_rel_err_h": opts["pixel_error_px"] / max(side[-1], 1.0), "pixel_rel_err_t": opts["pixel_error_px"] / max(side[0], 1.0)}


def group_sizes(markers: list[dict[str, Any]], views: dict[str, View], sizes: dict[str, tuple[int, int]], opts: dict[str, Any],
                exclude_frames: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Per book group: median height/thickness (reconstruction units) over its usable sightings, with reasons for the rest. ``exclude_frames``: room-pan frames
    (pipeline.room_pan): their sightings are not measured (reason ``room_pan_view``), so the scale prior and the shelf run come from the shelf pass only."""
    out: dict[str, dict[str, Any]] = {}
    for m in markers:
        rec = {"group_id": m["group_id"], "sightings_total": m.get("sightings_total"), "sightings_used": 0, "rejected": {}}
        hs, ts, asp, perr_h, perr_t, per = [], [], [], [], [], []
        n_pan = 0
        for s in m.get("sightings") or []:
            if exclude_frames and s["frame_id"] in exclude_frames:
                n_pan += 1
                rec["rejected"]["room_pan_view"] = rec["rejected"].get("room_pan_view", 0) + 1
                continue
            view = views.get(s["frame_id"])
            if view is None:
                rec["rejected"]["frame_not_registered"] = rec["rejected"].get("frame_not_registered", 0) + 1
                continue
            if s.get("occlusion") in ("partial", "heavy") or s.get("orientation") == "stacked":
                key = "occluded_" + str(s["occlusion"]) if s.get("occlusion") in ("partial", "heavy") else "stacked_orientation_axes_ambiguous"
                rec["rejected"][key] = rec["rejected"].get(key, 0) + 1
                continue
            w, h = sizes[s["frame_id"]]
            r = sighting_size(view, s.get("polygon_xy"), w, h, opts)
            if "reason" in r:
                rec["rejected"][r["reason"]] = rec["rejected"].get(r["reason"], 0) + 1
                continue
            if r["aspect"] < opts["min_aspect"]:
                rec["rejected"]["near_square_polygon"] = rec["rejected"].get("near_square_polygon", 0) + 1
                continue
            hs.append(r["h"]); ts.append(r["t"]); asp.append(r["aspect"]); perr_h.append(r["pixel_rel_err_h"]); perr_t.append(r["pixel_rel_err_t"])
            per.append({"frame_id": s["frame_id"], "observation_id": s.get("observation_id"), "height_units": round(r["h"], 5), "thickness_units": round(r["t"], 5)})
        rec["sightings_used"] = len(hs)
        if not hs:
            rec.update(status="abstained", reason="no_usable_sighting:" + (",".join(sorted(rec["rejected"])) or "no_polygon"))
        else:
            med_h, med_t = float(np.median(hs)), float(np.median(ts))
            spread = (max(hs) - min(hs)) / med_h if len(hs) > 1 else None
            if len(hs) < 2 and (m.get("sightings_total") or 0) - n_pan > 1:
                rec.update(status="abstained", reason="single_usable_sighting_of_several")
            elif spread is not None and spread > opts["max_sighting_spread"]:
                rec.update(status="abstained", reason=f"sightings_disagree_on_height:{spread * 100:.0f}%")
            else:
                rec.update(status="ok", h_units=med_h, t_units=med_t, aspect=float(np.median(asp)), sighting_spread_pct=None if spread is None else round(spread * 100, 1),
                           per_sighting=per, pixel_rel_err_h=float(np.median(perr_h)), pixel_rel_err_t=float(np.median(perr_t)), xyz=m.get("xyz"))
        out[m["group_id"]] = rec
    return out


# --------------------------------------------------------------------------------------------- scale fit
def fit_scale(samples: list[dict[str, Any]], opts: dict[str, Any]) -> dict[str, Any]:
    """Robust cm/unit from per-book (catalogue height cm, reconstructed height units): median of log ratios, inliers within max(ratio_band, 3 MAD)."""
    n = len(samples)
    out: dict[str, Any] = {"n_candidates": n, "n_inliers": 0, "cm_per_unit": None}
    if n < opts["min_books"]:
        out["reason"] = f"only_{n}_identified_books_with_catalogue_dimensions_and_a_reconstructed_height(need_{opts['min_books']})"
        if n:
            out["partial_cm_per_unit"] = round(float(np.median([s["ratio"] for s in samples])), 4)     # NOT used as the scale: shown only as a cross-check of the prior
            out["partial_ratios_cm_per_unit"] = [round(s["ratio"], 2) for s in samples]
        return out
    lr = np.log([s["ratio"] for s in samples])
    keep = np.ones(n, bool)
    for _ in range(4):
        med = float(np.median(lr[keep]))
        mad = 1.4826 * float(np.median(np.abs(lr[keep] - med)))
        band = max(opts["ratio_band"], 3.0 * mad)
        new = np.abs(lr - med) <= band
        if new.sum() == keep.sum() and (new == keep).all():
            break
        keep = new
    med = float(np.median(lr[keep]))
    k = int(keep.sum())
    mad = 1.4826 * float(np.median(np.abs(lr[keep] - med))) if k > 1 else 0.0
    out.update(n_inliers=k, inlier_mask=[bool(x) for x in keep], scatter_pct=round(mad * 100, 1))
    if k < opts["min_books"]:
        out["reason"] = f"only_{k}_of_{n}_catalogue_scaled_books_agree(need_{opts['min_books']})"
        return out
    if k / n < opts["min_inlier_fraction"]:
        out["reason"] = f"catalogue_scaled_books_disagree:{k}_of_{n}_inliers"
        return out
    if mad * 100.0 > opts["max_scatter_pct"]:
        out["reason"] = f"catalogue_scaled_books_scatter_{mad * 100:.0f}%_above_{opts['max_scatter_pct']:.0f}%"
        return out
    out["cm_per_unit"] = float(math.exp(med))
    sample_err = (mad / math.sqrt(k)) * 1.25                     # standard error of the median
    edition_sys = max(opts["catalogue_systematic_pct"] / 100.0, float(np.median([s.get("edition_spread_pct") or 0.0 for s in samples])) / 200.0)
    out["relative_uncertainty_pct"] = round(math.hypot(sample_err, edition_sys) * 100.0, 1)
    out["uncertainty_terms_pct"] = {"median_standard_error": round(sample_err * 100, 1), "catalogue_edition_systematic": round(edition_sys * 100, 1)}
    # leave-one-out consistency of the catalogue heights with the scale (NOT an accuracy claim about the owned copies)
    loo = []
    idx = [i for i in range(n) if keep[i]]
    if k >= 3:
        for i in idx:
            others = [j for j in idx if j != i]
            sc = math.exp(float(np.median(lr[others])))
            pred = samples[i]["h_units"] * sc
            loo.append(abs(pred - samples[i]["catalogue_height_cm"]) / samples[i]["catalogue_height_cm"] * 100.0)
        out["leave_one_out_catalogue_height_abs_error_pct"] = {"median": round(float(np.median(loo)), 1), "max": round(max(loo), 1), "n": len(loo)}
    return out


def scale_from_catalogue(identified: list[dict[str, Any]], sizes: dict[str, dict[str, Any]], catalogue: Any, opts: dict[str, Any],
                         this_year: int | None = None) -> dict[str, Any]:
    """Scale source A. ``identified``: [{group_id, work, title, visible_text, language_code?}] (accepted catalogue WORKS only). ``catalogue``: providers.openlibrary.OpenLibrary."""
    from pipeline import price as PR
    sources, samples, memo = [], [], {}
    deadline = time.monotonic() + float(opts["budget_s"]) if opts.get("budget_s") else None
    ordered = sorted(identified, key=lambda b: -((sizes.get(b.get("group_id") or b.get("id")) or {}).get("sightings_used") or 0))     # best-seen books first
    for book in ordered:
        gid, work = book.get("group_id") or book.get("id"), book.get("work")
        src = {"group_id": gid, "work": work, "title": book.get("title"), "inventory_title": book.get("title"), "used": False}
        sources.append(src)
        if len(samples) >= opts["max_fit_books"]:
            src["reason"] = "not_needed:max_fit_books_reached"; continue
        if deadline is not None and time.monotonic() > deadline and work not in memo:
            src["reason"] = "catalogue_budget_exhausted"; continue
        size = sizes.get(gid)
        if not work:
            src["reason"] = "no_catalogue_work_id"; continue
        if size is None or size.get("status") != "ok":
            src["reason"] = "no_reconstructed_height:" + str((size or {}).get("reason") or "group_not_in_reconstruction"); continue
        dims = book.get("catalogue_dimensions")
        if dims and dims.get("height_cm"):                 # a Google Books volume with a recorded height: that volume is the edition (no ranking, spread 0)
            cat = {"height_cm": float(dims["height_cm"]), "edition_spread_pct": 0.0, "editions_with_dimensions": 1, "source": "google_books_volume",
                   "source_url": dims.get("source_url"), "retrieved_at": dims.get("retrieved_at")}
            src.update(catalogue=cat, catalogue_urls=[dims.get("source_url")], catalogue_retrieved_at=dims.get("retrieved_at"), h_units=size["h_units"],
                       catalogue_height_cm=cat["height_cm"], ratio=cat["height_cm"] / size["h_units"], edition_spread_pct=0.0)
            samples.append(src)
            continue
        try:
            if work not in memo:
                memo[work] = catalogue.editions(work)
            res = memo[work]
        except Exception as exc:                      # offline cache miss / network: this book contributes nothing
            src["reason"] = f"catalogue_unavailable:{type(exc).__name__}:{str(exc)[:60]}"; continue
        entries = res["entries"]
        language, basis = PR.resolve_language(book.get("visible_text"), book.get("language_code") if book.get("language_code") in PR._OL_LANG else None, book.get("title"), entries)
        if not language:
            src["reason"] = "language_not_determined:" + str(basis); continue
        cat = catalogue_height(entries, language, this_year=this_year, top_editions=opts["top_editions"], max_spread_pct=opts["max_edition_spread_pct"])
        src.update(language=language, language_basis=basis, catalogue=cat, catalogue_urls=res.get("urls"), catalogue_retrieved_at=res.get("retrieved_at"), h_units=size["h_units"])
        if cat.get("height_cm") is None:
            src["reason"] = cat.get("reason") or "no_catalogue_height"; continue
        src.update(catalogue_height_cm=cat["height_cm"], ratio=cat["height_cm"] / size["h_units"], edition_spread_pct=cat["edition_spread_pct"])
        samples.append(src)
    fit = fit_scale(samples, opts)
    mask = fit.pop("inlier_mask", None)
    if mask is not None:
        for s, keep in zip(samples, mask):
            s["used"] = bool(keep) and fit.get("cm_per_unit") is not None
            if not keep:
                s["reason"] = "outlier_vs_median_scale"
    return {"source": "catalogue_physical_dimensions", **fit, "sources": sources,
            "method": ("cm per reconstruction unit = median over identified books of (catalogue height of the work-level equivalent edition) / (the book's height on the "
                       "reconstructed shelf plane); MAD-rejected outliers; every source listed")}


def scale_from_prior(sizes: dict[str, dict[str, Any]], opts: dict[str, Any]) -> dict[str, Any]:
    hs = [s["h_units"] for s in sizes.values() if s.get("status") == "ok"]
    if len(hs) < 10:
        return {"source": "standard_format_prior", "cm_per_unit": None, "reason": f"only_{len(hs)}_measurable_spines(need_10_for_a_library_median)"}
    return {"source": "standard_format_prior", "cm_per_unit": PRIOR["median_height_cm"] / float(np.median(hs)), "relative_uncertainty_pct": PRIOR["relative_uncertainty_pct"],
            "n_spines": len(hs), "basis": PRIOR["basis"], "not_a_measurement": True,
            "method": "scale = prior median book height / median reconstructed spine height of the whole library"}


# --------------------------------------------------------------------------------------------- spines
def measure_spines(sizes: dict[str, dict[str, Any]], scale: dict[str, Any], opts: dict[str, Any]) -> list[dict[str, Any]]:
    s_cm = scale.get("cm_per_unit")
    s_unc = (scale.get("relative_uncertainty_pct") or 0.0) / 100.0
    out = []
    thick = [z["t_units"] * s_cm for z in sizes.values() if s_cm and z.get("status") == "ok" and z["aspect"] >= opts["spine_aspect_min"]]
    lib_med = float(np.median(thick)) if thick else None
    lo_t, hi_t = opts["library_thickness_cm"]
    library_ok = lib_med is None or lo_t <= lib_med <= hi_t
    for gid, size in sizes.items():
        row = {"group_id": gid, "sightings_used": size.get("sightings_used"), "validation_state": "software_only_unvalidated"}
        if size.get("status") != "ok":
            row.update(spine_height_cm=None, spine_thickness_cm=None, status="abstained", reason=size.get("reason"))
        elif not s_cm:
            row.update(spine_height_cm=None, spine_thickness_cm=None, status="abstained", reason="no_metric_scale:" + str(scale.get("reason") or "unavailable"))
        else:
            h_rel = math.sqrt(s_unc ** 2 + (opts["plane_offset_pct"] / 100.0) ** 2 + size["pixel_rel_err_h"] ** 2 + ((size["sighting_spread_pct"] or 0.0) / 100.0 / math.sqrt(max(size["sightings_used"], 1))) ** 2)
            row.update(status="measured", per_observation=[{**q, "height_cm": round(q["height_units"] * s_cm, 2), "thickness_cm": round(q["thickness_units"] * s_cm, 2)} for q in size["per_sighting"]],
                       spine_height_cm=round(size["h_units"] * s_cm, 1), height_relative_uncertainty_pct=round(h_rel * 100, 1),
                       method="3D shelf-plane ray casting of detection polygons x scale (" + str(scale.get("source")) + ")", scale_source=scale.get("source"))
            if not library_ok:
                row.update(spine_thickness_cm=None, thickness_reason=f"library_median_thickness_{lib_med:.1f}cm_outside_{lo_t}-{hi_t}cm(spines_too_narrow_in_pixels_for_a_thickness)")
            elif size["pixel_rel_err_t"] > opts["max_thickness_pixel_err"]:
                row.update(spine_thickness_cm=None, thickness_reason=f"spine_only_{opts['pixel_error_px'] / size['pixel_rel_err_t']:.0f}px_wide_too_narrow_for_a_thickness")
            elif size["aspect"] >= opts["spine_aspect_min"]:
                t_rel = math.sqrt(h_rel ** 2 + size["pixel_rel_err_t"] ** 2)
                row.update(spine_thickness_cm=round(size["t_units"] * s_cm, 1), thickness_relative_uncertainty_pct=round(t_rel * 100, 1))
            else:
                row.update(spine_thickness_cm=None, thickness_reason=f"polygon_aspect_{size['aspect']:.1f}_not_a_spine_view(cover_or_flat_face)")
        out.append(row)
    return out


def shelf_rows(sizes: dict[str, dict[str, Any]], s_cm: float | None) -> dict[str, Any]:
    """Shelf run: per shelf row (markers clustered by y) the x-extent covered by measured spines, summed; plus the sum of measured thicknesses. Both labelled, never one 0."""
    pts = [(s["xyz"][0], s["xyz"][1], s["t_units"], s["h_units"]) for s in sizes.values() if s.get("status") == "ok" and s.get("xyz")]
    if not pts or not s_cm:
        return {"shelf_run_m": None, "reason": "no_metric_scale_or_no_measured_spines"}
    pts.sort(key=lambda p: p[1])
    gap = 0.6 * float(np.median([p[3] for p in pts]))
    rows, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[1] - cur[-1][1] > gap:
            rows.append(cur); cur = [p]
        else:
            cur.append(p)
    rows.append(cur)
    spans = [(max(p[0] + p[2] / 2 for p in r) - min(p[0] - p[2] / 2 for p in r)) * s_cm / 100.0 for r in rows]
    xs0 = min(p[0] - p[2] / 2 for p in pts); xs1 = max(p[0] + p[2] / 2 for p in pts)
    ys0 = min(p[1] - p[3] / 2 for p in pts); ys1 = max(p[1] + p[3] / 2 for p in pts)
    return {"shelf_run_m": round(float(sum(spans)), 2), "rows": len(rows), "row_spans_m": [round(x, 2) for x in spans],
            "shelved_face_m2": round(float((xs1 - xs0) * (ys1 - ys0) * (s_cm / 100.0) ** 2), 2), "books_y_range_units": [round(float(ys0), 4), round(float(ys1), 4)],
            "books_x_range_units": [round(float(xs0), 4), round(float(xs1), 4)],
            "sum_measured_thickness_m": round(sum(p[2] for p in pts) * s_cm / 100.0, 2), "measured_spines": len(pts),
            "method": "sum over shelf rows (markers clustered by height) of the horizontal extent of the measured spines; includes gaps and unmeasured books inside the extent",
            "note": "a shelf run only for the measured part of the shelving; abstained books inside it are not added"}


# --------------------------------------------------------------------------------------------- orchestration
def identified_from_books(books: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Packet-style book records -> accepted catalogue works only (status identified, identity_source catalogue_match, a /works/OL..W source URL)."""
    from pipeline.price import work_key_of
    out = []
    for b in books:
        if b.get("status") != "identified" or b.get("identity_source") != "catalogue_match":
            continue
        work = work_key_of(b)
        if work:
            out.append({"group_id": b.get("id") or b.get("instance_id"), "work": work, "title": b.get("title"), "visible_text": b.get("visible_text"),
                        "language_code": b.get("language")})
        elif b.get("catalogue") == "google_books" and (b.get("catalogue_dimensions") or {}).get("height_cm"):
            # second catalogue: the Google Books VOLUME is the edition, and its recorded height (cited) is the catalogue height; no equivalent-edition ranking
            out.append({"group_id": b.get("id") or b.get("instance_id"), "work": b.get("identity_work") or b.get("identity_source_url"), "title": b.get("title"),
                        "visible_text": b.get("visible_text"), "language_code": b.get("language"), "catalogue_dimensions": b["catalogue_dimensions"]})
    return out


def measure_data(recon: dict[str, Any], identified: list[dict[str, Any]], catalogue: Any | None, config: dict[str, Any] | None = None,
                 exclude_frames: set[str] | None = None) -> dict[str, Any]:
    """Pure measurement from a loaded recon3d.json (+ identified works + a catalogue client). -> auto_measure data block. ``exclude_frames``: room-pan frames whose
    sightings are not measured (pipeline.room_pan); the room geometry still uses the whole reconstruction."""
    opts = {**DEFAULTS, **(config or {})}
    base: dict[str, Any] = {"schema": SCHEMA, "validation": VALIDATION,
                            "limits": ["Uncertainty is a stated sum of declared terms, not a statistical guarantee.",
                                       "Catalogue sizes belong to the equivalent edition, not necessarily the owned copy.",
                                       "Spines sit a few cm in front of the fitted shelf plane (declared 4% term).",
                                       "Evaluation truth is never an input; accuracy against tape is NOT established by this stage."]}
    if recon.get("status") not in ("ok", "partial") or not recon.get("markers"):
        return {**base, "status": "abstained", "reason": f"3d_reconstruction_unusable:{recon.get('status')}", "scale": {"cm_per_unit": None, "reason": "no_reconstruction"}, "spines": [],
                "shelf": {"shelf_run_m": None, "reason": "no_reconstruction"}}
    views, sizes_px = views_from_recon(recon), frame_sizes(recon)
    sizes = group_sizes([m for m in recon["markers"] if m.get("kind") == "book"], views, sizes_px, opts, exclude_frames)
    scale: dict[str, Any] = {"cm_per_unit": None, "reason": "no_identified_books_with_catalogue_dimensions"}
    if catalogue is not None and identified:
        scale = scale_from_catalogue(identified, sizes, catalogue, opts)
    elif not identified:
        scale = {"source": "catalogue_physical_dimensions", "cm_per_unit": None, "n_candidates": 0, "n_inliers": 0, "sources": [],
                 "reason": "no_identified_books(0_accepted_catalogue_works_in_this_run)"}
    tried_a = dict(scale)
    if not scale.get("cm_per_unit") and opts["allow_prior"]:
        prior = scale_from_prior(sizes, opts)
        if prior.get("cm_per_unit"):
            prior["scale_source_a_attempt"] = {k: tried_a.get(k) for k in ("reason", "n_candidates", "n_inliers", "sources", "partial_cm_per_unit", "partial_ratios_cm_per_unit")}
            if tried_a.get("partial_cm_per_unit"):
                prior["catalogue_cross_check"] = {"n_books": tried_a.get("n_candidates"), "catalogue_cm_per_unit": tried_a["partial_cm_per_unit"],
                                                  "prior_vs_catalogue_pct": round((prior["cm_per_unit"] / tried_a["partial_cm_per_unit"] - 1.0) * 100.0, 1),
                                                  "note": "too few books to replace the prior (need %d); shown only to cross-check it" % opts["min_books"]}
            scale = prior
    spines = measure_spines(sizes, scale, opts)
    measured = [s for s in spines if s["status"] == "measured"]
    reasons: dict[str, int] = {}
    for s in spines:
        if s["status"] != "measured":
            key = str(s.get("reason")).split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    status = "ok" if measured and scale.get("source") == "catalogue_physical_dimensions" else "prior_only" if measured else "abstained"
    shelf = shelf_rows(sizes, scale.get("cm_per_unit"))
    from pipeline import auto_room
    room = auto_room.estimate(recon, scale, config, shelf)
    return {**base, "status": status, "scale": scale, "spines": spines, "room": room,
            "summary": {"book_groups": len(spines), "measured": len(measured), "abstained": len(spines) - len(measured), "abstain_reasons": reasons,
                        "scale_source": scale.get("source") if scale.get("cm_per_unit") else None,
                        "room_pan_sightings_not_measured": sum(z["rejected"].get("room_pan_view", 0) for z in sizes.values()),
                        "room_pan_only_groups": sum(1 for z in sizes.values() if z["rejected"].get("room_pan_view") and not z.get("sightings_used") and set(z["rejected"]) == {"room_pan_view"}),
                        "median_spine_height_cm": round(float(np.median([s["spine_height_cm"] for s in measured])), 1) if measured else None},
            "shelf": shelf}


def enrich_polygons(recon: dict[str, Any], candidates: dict[str, Any] | None) -> int:
    """Older reconstructions (recon3d_v1) kept only boxes: take the detector's oriented polygon of each sighting from the source run's candidates.json (by observation id).
    -> number of sightings filled in. Nothing is invented: a sighting without a polygon stays without one."""
    obs = {o["observation_id"]: o for o in ((candidates or {}).get("observations") or [])}
    n = 0
    for m in recon.get("markers") or []:
        for s in m.get("sightings") or []:
            o = obs.get(s.get("observation_id"))
            if not o:
                continue
            if not s.get("polygon_xy") and o.get("polygon_xy"):
                s["polygon_xy"] = o["polygon_xy"]
                n += 1
            for k in ("occlusion", "orientation"):
                if o.get(k) and k not in s:
                    s[k] = o[k]
    return n


def next_output(runs_dir: str | Path, run_id: str) -> Path:
    n = 1
    while (Path(runs_dir) / f"{run_id}_measure_v{n}").exists():
        n += 1
    return Path(runs_dir) / f"{run_id}_measure_v{n}"


def run(source_run: str | Path, recon3d_dir: str | Path, out_dir: str | Path, *, identified: list[dict[str, Any]] | None = None, online: bool = False,
        cache_dir: str | Path | None = None, config: dict[str, Any] | None = None, room_pan: bool = True) -> Path:
    source_run, out_dir = Path(source_run), Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; never overwritten (pick a new directory)")
    t0 = time.monotonic()
    recon = json.loads((Path(recon3d_dir) / "recon3d.json").read_text())
    cand_path = source_run / "candidates.json"
    filled = enrich_polygons(recon, C.read_json(cand_path)["data"] if cand_path.is_file() else None)
    if identified is None:
        pk = source_run / "claim_packet.json"
        identified = identified_from_books(C.read_json(pk)["data"]["books"]) if pk.is_file() else []
    cache = Path(cache_dir) if cache_dir else source_run.resolve().parent / "_price_cache" / "openlibrary"
    catalogue = None
    if identified:
        from providers.openlibrary import OpenLibrary
        catalogue = OpenLibrary(cache, online=online)
    # The shelf is measured on the shelf pass only; frames of the room pan (floor / ceiling / walls of the same sweep) still build the room geometry
    from pipeline import room_pan as RP
    rp = RP.classify_run(source_run, recon3d=recon3d_dir) if room_pan else None
    try:
        data = measure_data(recon, identified, catalogue, config, exclude_frames=set((rp or {}).get("room_pan_frames") or []))
    finally:
        if catalogue is not None:
            catalogue.close()
    data["source_run"] = source_run.name
    data["recon3d"] = str(Path(recon3d_dir).name)
    data["polygons_filled_from_candidates"] = filled
    data["room_pan"] = None if rp is None else {**RP.summary(rp), "sightings_not_measured": (data.get("summary") or {}).get("room_pan_sightings_not_measured", 0),
                                                "groups_seen_only_in_room_pan": (data.get("summary") or {}).get("room_pan_only_groups", 0),
                                                "note": "sightings in room-pan frames are not measured (reason room_pan_view); the room estimate uses the whole 3D map"}
    data["catalogue_requests"] = getattr(catalogue, "requests", 0) if catalogue is not None else 0
    data["elapsed_s"] = round(time.monotonic() - t0, 2)
    out_dir.mkdir(parents=True)
    C.write_json(out_dir / "auto_measure.json", C.envelope(out_dir.name, "auto_measure", data, status="ok" if data["status"] == "ok" else "partial",
                                                           metrics={"request_count": data["catalogue_requests"], "cost": 0, "cost_provenance": "local geometry + free Open Library catalogue (cached)"}))
    return out_dir / "auto_measure.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--recon3d", required=True, help="a runs/<run>_recon3d_vN directory")
    ap.add_argument("--out", help="NEW output directory (default: next runs/<run>_measure_vN)")
    ap.add_argument("--identified", help="JSON list of {group_id, work, title, visible_text, language_code?}; default: accepted catalogue works of the run's claim_packet.json")
    ap.add_argument("--online", action="store_true", help="allow free Open Library requests on a cache miss (polite 1 request/s); default offline = cache only")
    ap.add_argument("--cache-dir")
    ap.add_argument("--no-room-pan", action="store_true", help="measure room-pan frames' sightings too (pipeline.room_pan; comparison only)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else next_output(Path(a.source_run).resolve().parent, Path(a.source_run).name)
    identified = json.loads(Path(a.identified).read_text()) if a.identified else None
    p = run(a.source_run, a.recon3d, out, identified=identified, online=a.online, cache_dir=a.cache_dir, room_pan=not a.no_room_pan)
    d = C.read_json(p)["data"]
    print(p, d["status"], json.dumps(d.get("summary")), (d["scale"] or {}).get("cm_per_unit"), (d["scale"] or {}).get("reason"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
