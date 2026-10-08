"""Stage ``packet``: provisional claim packet from the deduplicated inventory.

    packet.build(inventory_path, manifest_path, run_dir, config) -> packet_path   (claim_packet.json)

Keeps every required key of the claim packet's output contract. Prices, spine dimensions, shelf run and
room measurements are ``null`` (unknown) because those stages are not implemented in this
milestone; nothing is written as 0. Counts are computed here, in code, from the instance lines.
The packet is explicitly ``completion_status: "provisional"``.

Expected inventory shape (written by dedup; fields other than the first four are optional):
  data.books[] / data.items[]:
    instance_id, observation_ids[], frame_ref (= frame_id), bbox_xyxy,
    shelf, position, orientation, category, description, brand_model,
    identity{title, author, publisher, edition, isbn, decision, source_url, visible_text},
    confidence, issues[str|{code,detail}], dedup_reasons[], uncertain_association(bool)
  data.counts{uncertain_observations, uncertain_associations, count_lower, count_upper}  (optional)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Any

from pipeline import contracts as C
from pipeline import corrections as K
from pipeline import scale as SC
from pipeline import room as RM
from decimal import Decimal

DEFAULT_MARKET = {"country": "US", "currency": "USD"}  # configurable default, not inferred from the developer's location

PROVISIONAL_NOTICE = ("PROVISIONAL, INCOMPLETE: unpriced and unmeasured inventory from a recorded development video. "
                      "Not a finished claim packet. Prices, spine dimensions, shelf run and room measurements are "
                      "unknown (null), not zero.")
APPRAISAL_WORDS = ("frame", "art", "portrait", "painting", "framed", "print", "poster", "photograph", "picture")
LOW_CONFIDENCE = 0.5


def _issue_texts(inst: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for iss in inst.get("issues") or []:
        if isinstance(iss, dict):
            out.append((str(iss.get("code", "issue")), str(iss.get("detail") or iss.get("code") or "issue")))
        else:
            out.append(("issue", str(iss)))
    return out


def _evidence(inst: dict[str, Any], frames: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fr = frames.get(inst.get("frame_ref"))
    return {
        "frame_id": inst.get("frame_ref"),
        "frame_ref": fr["image_path"] if fr else None,
        "timestamp_s": fr["timestamp_s"] if fr else None,
        "bbox_xyxy": inst.get("bbox_xyxy"),
        "polygon_xy": inst.get("polygon_xy"),   # exact detected outline (rotated spine), for the highlight
        "frame_width_px": fr["width_px"] if fr else None,
        "frame_height_px": fr["height_px"] if fr else None,
        "observation_ids": inst.get("observation_ids", []),
    }


def _book_line(inst: dict[str, Any], frames: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ident = inst.get("identity") or {}
    accepted = ident.get("decision") in ("accepted", "user_stated") and bool(ident.get("title"))
    user_stated = ident.get("decision") == "user_stated"
    appraisal = inst.get("appraisal")
    ev = _evidence(inst, frames)
    return {
        "id": inst["instance_id"], "display_index": inst.get("display_index"), "shelf": inst.get("shelf"), "position": inst.get("position"),
        "frame_ref": ev["frame_ref"], "frame_id": ev["frame_id"], "timestamp_s": ev["timestamp_s"],
        "status": "needs_appraisal" if appraisal else ("identified" if accepted else "unidentified"),
        "identity_source": ("user_statement" if user_stated else "catalogue_match" if accepted else None),
        "language": ident.get("language") or ident.get("language_code"),
        "identity_matching_evidence": ident.get("matching_evidence") or [], "appraisal": appraisal,
        "title": ident.get("title") if accepted else None,
        "author": ident.get("author") if accepted else None,
        "edition": ident.get("edition") if accepted else None,
        "isbn": ident.get("isbn") if accepted else None,
        "publisher": ident.get("publisher") if accepted else None,
        "spine_height_cm": None, "spine_thickness_cm": None,
        "id_confidence": ident.get("id_confidence") if accepted else None,  # catalogue/reading confidence, not detection confidence
        "replacement_cost": None, "used_value": None,
        "visible_text": ident.get("visible_text"), "identity_decision": ident.get("decision"),
        "identity_source_url": ident.get("source_url") if accepted and not user_stated else None,
        "catalogue": (ident.get("catalogue") or "open_library") if accepted and not user_stated else None,
        "catalogue_title": ident.get("catalogue_title") if accepted and not user_stated else None,
        "identity_work": ident.get("work") if accepted and not user_stated else None,
        "catalogue_dimensions": ident.get("catalogue_dimensions") if accepted and not user_stated else None,
        "orientation": inst.get("orientation"), "detection_confidence": inst.get("confidence"),
        "evidence": ev, "dedup_reasons": inst.get("dedup_reasons", []),
        "uncertain_association": bool(inst.get("uncertain_association")),
        "issues": [d for _, d in _issue_texts(inst)], "user_corrections": inst.get("user_corrections", []),
        "count_tier": inst.get("count_tier"),
    }


WORK_LEVEL = 'work_level_equivalent'
WORK_LEVEL_NOTE = ('Work-level equivalent price: priced through one stated equivalent edition (the ISBN shown), NOT the exact edition on the shelf '
                   'and not a realized sale; US/USD asking prices (or, when no US seller offers it, a non-US asking price converted to USD and labelled converted), '
                   'shipping/tax excluded.')


WORK_LEVEL_EDITION_MATCH = 'work_level_equivalent_edition_not_exact_shelf_edition'


def _validate_price_line(offer: dict[str, Any], book: dict[str, Any], run_dir: Any, want: str | None, market: str = 'US') -> list[str]:
    """validate_offer for one priced line. A work-level line must declare itself as such (price_basis, equivalent ISBN, edition_match label); its raw-source
    check then runs with the source-level edition_match ('exact_isbn' = the listing is for exactly the EQUIVALENT ISBN), because the line's own label
    deliberately says it is not the shelf edition and so cannot equal the parsed source value."""
    from pipeline.price import validate_offer
    if want is None:
        return validate_offer(offer, book, run_dir)
    problems = []
    if offer.get('price_basis') != WORK_LEVEL or offer.get('equivalent_isbn') != want or offer.get('edition_match') != WORK_LEVEL_EDITION_MATCH:
        problems.append('work-level line must carry price_basis, the equivalent ISBN and the not-exact-edition label')
    return problems + validate_offer({**offer, 'edition_match': 'exact_isbn'}, book, run_dir, expected_isbn=want, market=market)


def _null_reason(book: dict[str, Any], key: str) -> str:
    """Why a priced-line value is null: the price row's own issue for that value, else its other issues, else a generic statement."""
    issues = [str(x) for x in (book.get('pricing_issues') or [])]
    own = [x for x in issues if x.startswith(key)]
    return "; ".join(own or issues) or 'no validated US/USD offer'


def _item_line(inst: dict[str, Any], frames: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ev = _evidence(inst, frames)
    line = {
        "id": inst["instance_id"], "display_index": inst.get("display_index"), "category": inst.get("category"), "detector_category": inst.get("detector_category"), "description": inst.get("description"),
        "brand_model": inst.get("brand_model"), "frame_ref": ev["frame_ref"], "frame_id": ev["frame_id"],
        "timestamp_s": ev["timestamp_s"], "dimensions_cm": {"w": None, "h": None, "d": None},
        "status": None, "status_basis": None, "status_reason": None,      # set by item_status below (priced | range | needs_appraisal, never null)
        "replacement_cost": inst.get("replacement_cost") if isinstance(inst.get("replacement_cost"), dict) else None, "confidence": inst.get("confidence"),
        "evidence": ev, "dedup_reasons": inst.get("dedup_reasons", []),
        "uncertain_association": bool(inst.get("uncertain_association")),
        "art_is_print": bool(inst.get("art_is_print")), "appraisal": inst.get("appraisal"),
        "issues": [d for _, d in _issue_texts(inst)], "user_corrections": inst.get("user_corrections", []),
        # 'ambiguous' = book/non-book undecided (spine shape outside a shelf row, or overlapping a non-book): reviewed, never in book_count.
        "classification": inst.get("classification"), "non_book_demotion": inst.get("non_book_demotion"),
    }
    set_item_status(line)
    return line


# ---------------------------------------------------------------------------------------------------------------------------------------------
# ITEM STATUS. The output contract allows only "priced | range | needs_appraisal" for an item. Art / portraits / framed pictures default to needs_appraisal
# unless the user said it is a print. An item with no SOURCED price or range (no legible brand/model and no price source queried) cannot honestly be "priced"
# or "range" (both claim a sourced figure), so it is needs_appraisal with status_basis "no_sourced_price": a person has to value it, nothing is estimated,
# it is excluded from the item totals and it is routed to the review queue (code item_no_sourced_price).
# ---------------------------------------------------------------------------------------------------------------------------------------------
ITEM_BASIS_REASONS = {
    "user_flagged_appraisal": "Flagged for human appraisal by the user; not auto-priced.",
    "art_default_needs_appraisal": "Art / portrait / framed item: defaults to needs appraisal unless the user states it is a print (the agent should ask); not auto-priced.",
    "no_sourced_price": ("No sourced price or range: brand/model not legible and no price source was queried for this item, so no value is given (never estimated). "
                         "Status needs_appraisal = a person must value it; excluded from the item totals."),
    "sourced_price": "Priced from the cited source (low = high).",
    "sourced_range": "Sourced price range (brand/model unknown).",
}


def _item_price(rc: Any) -> tuple[Decimal, Decimal] | None:
    """(low, high) of a SOURCED item price line, else None. A figure without source, url and retrieval date is not a price."""
    if not isinstance(rc, dict) or not (rc.get("source") and rc.get("url") and rc.get("retrieved_at")):
        return None
    lo, hi = rc.get("low"), rc.get("high")
    if lo is None and hi is None and rc.get("amount") is not None:
        lo = hi = rc["amount"]
    try:
        lo, hi = Decimal(str(lo)), Decimal(str(hi))
    except Exception:  # noqa: BLE001 - not a number -> not a price
        return None
    if not (lo.is_finite() and hi.is_finite()) or lo <= 0 or hi < lo:
        return None
    return lo, hi


def _is_art(line: dict[str, Any]) -> bool:
    cat, desc = str(line.get("category") or "").lower(), str(line.get("description") or "").lower()
    return any(w in cat or w in desc for w in APPRAISAL_WORDS)


def set_item_status(line: dict[str, Any]) -> dict[str, Any]:
    """status / status_basis / status_reason of one item line, from the line itself (idempotent; also fixes older packets with status null)."""
    price = _item_price(line.get("replacement_cost"))
    if line.get("appraisal"):
        status, basis = "needs_appraisal", "user_flagged_appraisal"
    elif _is_art(line) and not line.get("art_is_print"):
        status, basis = "needs_appraisal", "art_default_needs_appraisal"
    elif price:
        status, basis = ("priced", "sourced_price") if price[0] == price[1] else ("range", "sourced_range")
    else:
        status, basis = "needs_appraisal", "no_sourced_price"
    line["status"], line["status_basis"] = status, basis
    line["status_reason"] = (line.get("appraisal") or {}).get("reason") if basis == "user_flagged_appraisal" and isinstance(line.get("appraisal"), dict) else None
    line["status_reason"] = line["status_reason"] or ITEM_BASIS_REASONS[basis]
    return line


def item_review_entries(i: dict[str, Any]) -> list[dict[str, Any]]:
    """The status-owned review entries of an item line (recomputed on every finalize)."""
    basis = i.get("status_basis")
    if basis in ("art_default_needs_appraisal", "user_flagged_appraisal"):
        return [{"ref_id": i["id"], "code": "needs_appraisal", "severity": "warning",
                 "reason": "Art/portrait/framed item defaults to appraisal; ask whether it is a print." if basis == "art_default_needs_appraisal" else i["status_reason"]}]
    if basis == "no_sourced_price":
        return [{"ref_id": i["id"], "code": "item_no_sourced_price", "severity": "warning",
                 "reason": f"No sourced price or range for this {str(i.get('category') or 'item').replace('_', ' ')} (brand/model unknown); not estimated, "
                           "excluded from the item totals; needs a human valuation."}]
    return []


def item_totals(items: list[dict[str, Any]]) -> dict[str, Any]:
    """items_replacement_cost_low/high from the item lines: the sum of low / high over items whose status is priced (low = high) or range AND whose line carries a
    sourced figure; every other item (needs_appraisal: art default, user flag or no sourced price) is excluded and counted. null when no item line has a price."""
    priced = [(i, _item_price(i.get("replacement_cost"))) for i in items if i.get("status") in ("priced", "range")]
    priced = [(i, pr) for i, pr in priced if pr]
    q = lambda xs: float(sum(xs, Decimal(0)).quantize(Decimal(".01")))
    return {"items_replacement_cost_low": q([pr[0] for _, pr in priced]) if priced else None,
            "items_replacement_cost_high": q([pr[1] for _, pr in priced]) if priced else None,
            "items_priced": sum(i.get("status") == "priced" for i, _ in priced), "items_range": sum(i.get("status") == "range" for i, _ in priced),
            "items_no_sourced_price": sum(i.get("status_basis") == "no_sourced_price" for i in items),
            "items_unpriced": len(items) - len(priced)}


ITEM_TOTALS_NOTE = ("items_replacement_cost_low/high = sum of low / high over item lines with status priced (low = high) or range that carry a sourced figure; "
                    "needs_appraisal items (art default, user flag, or no sourced price: items_no_sourced_price) are excluded (items_unpriced counts them); "
                    "null when no item line has a sourced price. Nothing is estimated.")


CONVERTED_TOTALS_NOTE = ("The USD money totals include CONVERTED lines (converted: true): no US seller offered the equivalent edition, so a non-US seller's asking price "
                         "was converted to USD at the ECB reference rate cited on the line (native amount and currency kept). converted_lines counts them per total.")


def money_total(lines, key):
    values = [Decimal(str(line[key]['amount'])) for line in lines if isinstance(line.get(key), dict) and line[key].get('amount') is not None and line.get('status') != 'needs_appraisal']
    return float(sum(values).quantize(Decimal('.01'))) if values else None


def _second_market_totals(books: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Native-EUR second-market sums (Spain), kept apart from the US/USD totals; USD equivalents only when every contributing line carries FX evidence."""
    es = [b['second_market'] for b in books if isinstance(b.get('second_market'), dict) and (b['second_market'].get('replacement_cost') or b['second_market'].get('used_value'))]
    if not es:
        return None
    from decimal import Decimal
    tot = lambda key: float(sum((Decimal(str(m[key]['amount'])) for m in es if m.get(key)), Decimal(0)))
    usd = lambda key: (None if any(m.get(key) and m[key].get('usd_equivalent') is None for m in es)
                       else float(sum((Decimal(str(m[key]['usd_equivalent'])) for m in es if m.get(key)), Decimal(0))))
    return {"market": "ES", "currency": "EUR", "books_priced": len(es),
            "replacement_cost_eur": tot("replacement_cost"), "used_value_eur": tot("used_value"),
            "replacement_cost_usd_equivalent": usd("replacement_cost"), "used_value_usd_equivalent": usd("used_value"),
            "note": ("Second market (Spain): native EUR sums over the books that have a Spanish-seller line; the USD equivalents use the dated ECB reference rate "
                     "on each line (null if any line lacks FX evidence) and are NOT part of the US/USD totals above.")}


def _count_range_totals(books: list[dict[str, Any]], counts: dict[str, Any], user_excluded: int = 0) -> dict[str, Any]:
    """book_count_range / confident_groups / singleton_groups / count_reliability from the dedup count tiers (absent on older inventories)."""
    if not counts.get("count_range"):
        return {}
    lo, hi = counts["count_range"]
    n = len(books)
    lo, hi = min(max(0, lo - user_excluded), n), max(max(0, hi - user_excluded), n)
    return {"book_count_lower": lo, "book_count_upper": hi, "book_count_range": [lo, hi], "confident_groups": lo, "singleton_groups": counts.get("singleton_groups"),
            "unconfirmed_groups": counts.get("unconfirmed_groups"), "count_reliability": counts.get("count_reliability"),
            "count_range_basis": counts.get("count_range_basis")}


def compute_totals(books: list[dict[str, Any]], items: list[dict[str, Any]], counts: dict[str, Any],
                   user_excluded: int = 0, pricing_executed: bool = False) -> dict[str, Any]:
    """Counts derived from the lines. Unknown money/length totals stay null."""
    identified = sum(1 for b in books if b["status"] == "identified")  # needs_appraisal books are counted separately
    appraisal_items = sum(1 for i in items if i["status"] == "needs_appraisal")
    second = _second_market_totals(books)
    it_tot = item_totals(items)
    return {**({"second_market": second} if second else {}),
        "book_count": len(books), "books_identified": identified,
        "books_unidentified": sum(1 for b in books if b["status"] == "unidentified"),
        "books_needs_appraisal": sum(1 for b in books if b["status"] == "needs_appraisal"),
        "books_identified_by_user_statement": sum(1 for b in books if b.get("identity_source") == "user_statement"),
        "user_excluded_count": user_excluded,
        "shelf_run_m": None,
        "books_replacement_cost": money_total(books, "replacement_cost"), "books_used_value": money_total(books, "used_value"),
        **it_tot,
        "excluded_from_totals": (sum(not b.get("replacement_cost") or not b.get("used_value") for b in books) + it_tot["items_unpriced"]) if pricing_executed or any(b.get("pricing_attempted") for b in books) else None,
        "item_count": len(items), "items_needs_appraisal": appraisal_items,
        "ambiguous_count": sum(1 for i in items if i.get("classification") == "ambiguous"),
        "book_count_lower": (None if counts.get("count_lower") is None else max(0, counts["count_lower"] - user_excluded)),
        "book_count_upper": (None if counts.get("count_upper") is None else max(0, counts["count_upper"] - user_excluded)),
        **_count_range_totals(books, counts, user_excluded),
        "uncertain_observations": counts.get("uncertain_observations"),
        "uncertain_associations": counts.get("uncertain_associations"),
        "notes": {
            "money_and_length_totals": "Money sums include validated sourced lines only; null means no supported priced lines. Measurements remain null.",
            "excluded_from_totals": "Line count missing either book subtotal or item price after pricing runs (books without both values + items_unpriced); each monetary subtotal excludes its missing offers.",
            "book_count": ("best guess of physical books (confident + likely groups); unconfirmed single sightings are listed apart; human physical count unverified"
                           if counts.get("count_range") else "proposed deduplicated book groups; human physical count unverified; see lower/upper association bounds"),
            "ambiguous_count": "item lines whose book/non-book class is undecided (e.g. spine shapes never seen in a shelf row); in the review queue, never in book_count",
            "items_totals": ITEM_TOTALS_NOTE,
        },
    }


VALUATION_PARTIAL = ("Unknown prices excluded from each subtotal. Non-book items have no sourced prices (needs_appraisal, excluded from the item totals); "
                     "the second-country comparison of the same 10 books is in second_country.")


def build_review_queue(books, items, manifest_data, stage_status, fixture=False, scale=None, room=None) -> list[dict[str, Any]]:
    q: list[dict[str, Any]] = []

    def add(ref, code, severity, reason):
        q.append({"ref_id": ref, "code": code, "severity": severity, "reason": reason})

    for b in books:
        if b["status"] == "unidentified":
            why = "no accepted catalogue match" if b["visible_text"] else "spine text not read"
            add(b["id"], "unidentified_book", "info", f"Counted but unidentified: {why}. No title guessed.")
        if b.get("identity_source") == "user_statement":
            add(b["id"], "title_from_user_statement", "info", "Title comes from the user's statement, not verified from the spine or a catalogue; no ISBN/edition.")
        if b["status"] == "needs_appraisal":
            add(b["id"], "needs_appraisal", "warning", (b.get("appraisal") or {}).get("reason") or "Flagged for human appraisal; not auto-priced.")
        if b["identity_decision"] == "ambiguous":
            add(b["id"], "ambiguous_identity", "warning", "Catalogue candidates are ambiguous; not accepted.")
        if b["uncertain_association"]:
            add(b["id"], "uncertain_dedup", "warning", "Cross-frame association is uncertain; count may be over/under by this line.")
        if (b["detection_confidence"] is not None) and b["detection_confidence"] < LOW_CONFIDENCE:
            add(b["id"], "low_detection_confidence", "warning", f"Detection confidence {b['detection_confidence']:.2f} < {LOW_CONFIDENCE}.")
        if b.get('pricing_attempted'):
            for key in ('replacement_cost', 'used_value'):
                if b.get(key) is None:
                    add(b['id'], key+'_unknown', 'warning', 'No validated physical US/USD offer; excluded from this monetary subtotal.')
            for reason in b.get('pricing_issues', []):
                add(b['id'], 'pricing_exclusion', 'warning', str(reason))
        for text in b["issues"]:
            add(b["id"], "line_issue", "warning", text)
    for i in items:
        demoted = i.get("non_book_demotion") or {}
        if i.get("classification") == "ambiguous":
            add(i["id"], "ambiguous_not_counted", "warning", demoted.get("reason") or
                f"Book/non-book undecided ({i.get('category')}); kept for review, not counted as a book.")
        for e in item_review_entries(i):
            add(e["ref_id"], e["code"], e["severity"], e["reason"])
        if i["uncertain_association"]:
            add(i["id"], "uncertain_dedup", "warning", "Cross-frame association is uncertain.")
        if i["confidence"] is not None and i["confidence"] < LOW_CONFIDENCE:
            add(i["id"], "low_detection_confidence", "warning", f"Detection confidence {i['confidence']:.2f} < {LOW_CONFIDENCE}.")
        for text in i["issues"]:
            if text != demoted.get("reason"):
                add(i["id"], "line_issue", "warning", text)
    for g in manifest_data.get("capture_gaps", []):
        add("capture", g["type"], "warning", f"{g['start_s']:.1f}-{g['end_s']:.1f}s: {g['detail']}")
    if room is None or room.get("status") != "estimated":
        why = "; ".join((room or {}).get("abstention_reasons", [])) or "no room evidence supplied"
        miss = ", ".join(((room or {}).get("coverage") or {}).get("missing", []))
        add("room", "room_not_measured", "blocking", f"Room dimensions/areas not established ({why})" + (f"; still needed: {miss}" if miss else "") + ".")
    if not any(b.get('pricing_attempted') for b in books):
        add("prices", "prices_not_retrieved", "blocking", "No valuation stage run; replacement/used values and item ranges are null.")
    else:
        add('prices', 'valuation_partial', 'blocking', VALUATION_PARTIAL)
    if scale is None or scale["summary"]["valid_calibrations"] == 0:
        add("dimensions", "no_metric_scale", "blocking", "Spine height/thickness and shelf run are unknown: no valid user calibration on a shelf plane.")
    else:
        sm = scale["summary"]
        add("dimensions", "spine_dimensions_partial", "warning" if sm["measured"] else "blocking",
            f"{sm['measured']} of {sm['books']} books measured; {sm['abstained']} abstained. Reasons: " +
            ", ".join(f"{k} x{v}" for k, v in list(sm["abstention_reasons"].items())[:6]))
        withheld = [r for r in scale["shelf_runs"] if r["status"] != "complete"]
        for r in withheld:
            add("dimensions", "shelf_run_withheld", "warning", f"Shelf run for {r['plane_id']} withheld: {r['withheld_reason']}.")
    if fixture:
        add("fixture", "fixture_data", "blocking", "Lines come from test FIXTURES, not real detections; do not use for evaluation.")
    for stage, st in stage_status.items():
        if st["status"] not in ("ok", "skipped"):
            add(f"stage:{stage}", f"stage_{st['status']}", "blocking", st.get("detail") or f"Stage {stage} is {st['status']}.")
    return q


# ---------------------------------------------------------------------------------------------------------------------------------------------
# REVIEW QUEUE: ONE ENTRY PER ref_id. The stages add flat details ({ref_id, code, severity, reason}); the packet keeps one entry per ref_id with a
# short human ``reason``, the most severe detail's ``severity`` / ``code``, every code in ``codes`` and every detail in ``reasons`` (nothing dropped). Entries are
# ordered blocking -> warning -> info (stable: first appearance within a severity). flat_review() gives the details back, so layers that filter by code work on
# either shape; consolidate_review() is idempotent. Totals never read the queue.
# ---------------------------------------------------------------------------------------------------------------------------------------------
SEVERITY_RANK = {"blocking": 0, "warning": 1, "info": 2}
REVIEW_LABELS = {
    "unidentified_book": "unidentified (no title guessed)", "title_from_user_statement": "title from the user's statement",
    "needs_appraisal": "needs human appraisal", "ambiguous_identity": "ambiguous catalogue match", "uncertain_dedup": "uncertain duplicate link",
    "low_detection_confidence": "low detection confidence", "ambiguous_not_counted": "book/non-book undecided (not counted)",
    "item_no_sourced_price": "no sourced price or range (needs a human valuation)", "replacement_cost_unknown": "no sourced replacement price",
    "used_value_unknown": "no sourced used value", "no_clean_frame": "no clean frame", "room_auto_low_confidence_wall": "low-confidence wall",
    "count_headline": "why this headline count", "count_estimator_instances": "how the book lines were merged", "count_estimator_panorama": "panorama count note",
    "shelf_run_withheld": "shelf run withheld", "spine_scale_from_prior": "spine scale from a standard-format prior",
}
LINE_ISSUE_LABELS = (("Uncertain cross-frame", "uncertain duplicate link"), ("Unreadable/unresolved spine", "unidentified (no title guessed)"),
                     ("At least one view pair lacks trusted registration", "possible duplicate (unregistered views)"),
                     ("Object category uncertain", "category uncertain"))


def flat_review(queue: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Flat details of a review queue in either shape (consolidated entries expand to their ``reasons``; flat entries pass through)."""
    out = []
    for r in queue or []:
        if not isinstance(r, dict):
            continue
        det = r.get("reasons")
        if isinstance(det, list) and det and all(isinstance(x, dict) for x in det):
            out += [{"ref_id": r.get("ref_id"), "code": x.get("code"), "severity": x.get("severity"), "reason": x.get("reason")} for x in det]
        else:
            out.append({k: x for k, x in r.items() if k not in ("codes", "reasons", "detail_count")})
    return out


def _detail_label(r: dict[str, Any]) -> str:
    code, text = str(r.get("code") or ""), str(r.get("reason") or "")
    if code == "line_issue":
        return next((lab for pre, lab in LINE_ISSUE_LABELS if text.startswith(pre)), text if len(text) <= 70 else text[:67].rstrip() + "...")
    if code == "pricing_exclusion":
        if "_converted_from_" in text:
            return "price converted from " + text.rsplit("_", 1)[-1]
        return "no sourced price"
    return REVIEW_LABELS.get(code) or code.replace("_", " ")


def _short_reason(rows: list[dict[str, Any]]) -> str:
    """One short human sentence for a ref's details (the details themselves stay in ``reasons``)."""
    if len(rows) == 1:
        return str(rows[0].get("reason") or "")
    codes = {r.get("code") for r in rows}
    no_price = {"replacement_cost_unknown", "used_value_unknown"} <= codes
    labels: dict[str, Counter] = {}
    for r in rows:
        lab = _detail_label(r)
        if no_price and (r.get("code") in ("replacement_cost_unknown", "used_value_unknown") or (r.get("code") == "pricing_exclusion" and lab == "no sourced price")):
            lab = "no sourced price (excluded from money totals)"
        labels.setdefault(lab, Counter())[r.get("code")] += 1
    parts = [lab + (f" (x{max(c.values())})" if max(c.values()) > 1 else "") for lab, c in labels.items()]
    text = "; ".join(parts)
    return text[:1].upper() + text[1:]


def consolidate_review(queue: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """One entry per ref_id: {ref_id, severity, code, reason (short), codes, reasons (every detail), detail_count}; blocking first. Idempotent."""
    groups: dict[Any, list[dict[str, Any]]] = {}
    seen = set()
    for r in flat_review(queue):
        key = (r.get("ref_id"), r.get("code"), r.get("reason"))
        if key in seen:
            continue
        seen.add(key)
        groups.setdefault(r.get("ref_id"), []).append(r)
    out = []
    for ref, rows in groups.items():
        rows = sorted(rows, key=lambda x: SEVERITY_RANK.get(x.get("severity"), 3))
        out.append({"ref_id": ref, "severity": rows[0].get("severity"), "code": rows[0].get("code"), "reason": _short_reason(rows),
                    "codes": list(dict.fromkeys(x.get("code") for x in rows)),
                    "reasons": [{"code": x.get("code"), "severity": x.get("severity"), "reason": x.get("reason")} for x in rows], "detail_count": len(rows)})
    return sorted(out, key=lambda e: SEVERITY_RANK.get(e.get("severity"), 3))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# sweep.device: from the phone's capture_info (camera label, stream settings, user agent) recorded when the camera started; null only if nothing came.
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _ua_summary(ua: str) -> str | None:
    import re
    if not ua:
        return None
    m = re.search(r"\((iPhone|iPad|iPod)[^)]*?OS (\d+(?:_\d+)*)", ua)
    if m:
        dev, os_ = m.group(1), "iOS " + m.group(2).replace("_", ".")
    else:
        m = re.search(r"Android (\d+(?:\.\d+)*);\s*([^;)]+?)(?:\sBuild/[^;)]*)?[;)]", ua)
        if m:
            dev, os_ = m.group(2).strip(), "Android " + m.group(1)
        else:
            m = re.search(r"\((Macintosh|Windows NT [\d.]+|X11; Linux[^;)]*)", ua)
            dev, os_ = (m.group(1) if m else None), None
    browser = None
    b = re.search(r"(CriOS|FxiOS|EdgiOS|Edg|Chrome|Firefox)/(\d+(?:\.\d+)?)", ua)
    if b:
        browser = {"CriOS": "Chrome", "FxiOS": "Firefox", "EdgiOS": "Edge", "Edg": "Edge"}.get(b.group(1), b.group(1)) + " " + b.group(2)
    else:
        b = re.search(r"Version/(\d+(?:\.\d+)*).*Safari", ua)
        browser = "Safari " + b.group(1) if b else None
    bits = [x for x in (os_, browser) if x]
    if not (dev or bits):
        return None
    return (dev or "unknown device") + (f" ({', '.join(bits)})" if bits else "")


def device_from_capture(cq: dict[str, Any] | None) -> tuple[str | None, dict[str, Any] | None]:
    """(device string, device_detail) from a capture-quality record (``device_label`` + ``device`` settings, app.capture_quality) or a raw capture_info
    message (label, width, height, frame_rate, user_agent, ...). (None, None) when the phone reported nothing."""
    if not isinstance(cq, dict):
        return None, None
    st = cq.get("device") if isinstance(cq.get("device"), dict) else cq
    label = cq.get("device_label") if "device_label" in cq else cq.get("label")
    ua = str(st.get("user_agent") or "")
    w, h, fps = st.get("width"), st.get("height"), st.get("frame_rate")
    detail = {"camera_label": label or None, "user_agent": ua or None, "stream_width_px": w, "stream_height_px": h, "stream_fps": fps,
              "facing_mode": st.get("facing_mode"), "requested_resolution": st.get("requested_resolution"), "saved_frame_size": cq.get("frame_size"),
              "fps_effective": cq.get("fps_effective")}
    if not any(detail[k] is not None for k in ("camera_label", "user_agent", "stream_width_px", "stream_fps")):
        return None, None
    detail["source"] = "capture_info reported by the phone when the camera started (app/static/app.js reportCaptureInfo -> app.engine.set_capture_info)"
    parts = [_ua_summary(ua)]
    cam = ", ".join(x for x in (label or None, f"facing {st['facing_mode']}" if st.get("facing_mode") else None) if x)
    parts.append(f"camera: {cam}" if cam else None)
    if w and h:
        parts.append(f"stream {w}x{h}" + (f" @ {fps:g} fps" if isinstance(fps, (int, float)) else ""))
    elif isinstance(fps, (int, float)):
        parts.append(f"stream @ {fps:g} fps")
    if cq.get("frame_size"):
        parts.append(f"saved frames {cq['frame_size']}" + (f" at {cq['fps_effective']:g} fps" if isinstance(cq.get("fps_effective"), (int, float)) else ""))
    return "; ".join(x for x in parts if x) or None, {k: x for k, x in detail.items() if x is not None}


def finalize_contract(data: dict[str, Any], run_dir: str | Path | None = None) -> dict[str, Any]:
    """Make packet ``data`` meet the claim packet's output contract; idempotent, called by every packet writer (build and each attach layer) before verification.
      * items: status priced | range | needs_appraisal (never null), items_replacement_cost_low/high and excluded_from_totals recomputed from the lines;
      * review queue: status-owned item entries and the stage / prices entries refreshed, then one entry per ref_id (consolidate_review);
      * sweep.device from the phone's capture_info when the packet has none;
      * second_country: the same 10 priced books in the second country (pipeline.second_country), re-bound to the current lines."""
    items = data.get("items") or []
    for i in items:
        set_item_status(i)
    t = data.get("totals")
    if isinstance(t, dict):
        it = item_totals(items)
        t.update(it)
        t["items_needs_appraisal"] = sum(i.get("status") == "needs_appraisal" for i in items)
        t.setdefault("notes", {})["items_totals"] = ITEM_TOTALS_NOTE
        if t.get("excluded_from_totals") is not None:
            t["excluded_from_totals"] = sum(not b.get("replacement_cost") or not b.get("used_value") for b in data.get("books") or []) + it["items_unpriced"]
    flat = flat_review(data.get("review_queue"))
    item_ids = {i["id"] for i in items}
    flat = [r for r in flat if not (r.get("ref_id") in item_ids and r.get("code") in ("needs_appraisal", "item_no_sourced_price"))]
    for i in items:
        flat += item_review_entries(i)
    ss = data.get("stage_status")
    if isinstance(ss, dict) and ss:                      # stage entries follow the CURRENT stage_status (post-sweep steps update it after the packet was built)
        flat = [r for r in flat if not str(r.get("ref_id") or "").startswith("stage:")]
        flat += [{"ref_id": f"stage:{k}", "code": f"stage_{st['status']}", "severity": "blocking", "reason": st.get("detail") or f"Stage {k} is {st['status']}."}
                 for k, st in ss.items() if isinstance(st, dict) and st.get("status") not in ("ok", "skipped")]
    for r in flat:
        if r.get("ref_id") == "prices" and r.get("code") == "valuation_partial":
            r["reason"] = VALUATION_PARTIAL
    sw = data.get("sweep")
    if isinstance(sw, dict) and not sw.get("device"):
        prov = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
        dev, detail = device_from_capture(sw.get("capture_quality") or prov.get("capture_quality"))
        if dev:
            sw["device"], sw["device_detail"] = dev, detail
    if run_dir is not None and data.get("books") is not None:
        from pipeline import second_country as SCN
        SCN.apply(data, SCN.load(run_dir), run_dir)
    flat = [r for r in flat if r.get("ref_id") != "second_country"]
    if isinstance(data.get("second_country"), dict):
        from pipeline import second_country as SCN
        flat += SCN.review_entries(data["second_country"])
    data["review_queue"] = consolidate_review(flat)
    if isinstance(t, dict) and isinstance(data.get("room"), dict):
        data["completeness"] = completeness(data.get("books") or [], items, data["room"], t)
    return data


def _scale_evidence(scale: dict[str, Any] | None) -> dict[str, Any] | None:
    if scale is None:
        return None
    return {"method": scale["method"], "units": scale["units"], "calibrations": scale["calibrations"], "summary": scale["summary"],
            "shelf_runs": scale["shelf_runs"], "limits": scale["limits"], "validation": scale["validation"], "options": scale["options"]}


def _room_block(room: dict[str, Any] | None, scale: dict[str, Any] | None) -> dict[str, Any]:
    """The room block. Non-null values only from a room artifact with status 'estimated' (or user-supplied unit faces)."""
    blank = {"length_m": None, "width_m": None, "height_m": None, "floor_area_m2": None, "wall_area_m2": None,
             "shelved_wall_area_m2": None, "scale_method": None, "confidence": None, "status": "not_measured",
             "floor_area_ft2": None, "wall_area_ft2": None, "shelved_wall_area_ft2": None,
             "evidence_audit": "see coverage_gaps and measurement_evidence: no room boundaries or metric reference established"}
    if room is None:
        return blank
    out = dict(blank)
    out["abstention_reasons"] = room.get("abstention_reasons", [])
    out["coverage_missing"] = (room.get("coverage") or {}).get("missing", [])
    sh = (room.get("shelved_wall") or {}).get("area") or {}
    if sh.get("m2") is not None:
        out.update(shelved_wall_area_m2=sh["m2"], shelved_wall_area_ft2=sh["ft2"],
                   shelved_wall_provenance="user-supplied unit dimensions, not a camera measurement")
    if room.get("status") == "estimated":
        d, a = room["dimensions"], room["areas"]
        out.update(length_m=d["length_m"], width_m=d["width_m"], height_m=d["height_m"], floor_area_m2=a["floor"]["m2"],
                   wall_area_m2=a["wall"]["m2"], floor_area_ft2=a["floor"]["ft2"], wall_area_ft2=a["wall"]["ft2"],
                   scale_method=room["method"] + " + " + (room.get("height_provenance") or {}).get("kind", "height_anchor"),
                   status="estimated", shape=room["shape"].get("kind"), shape_assumption=room["shape"].get("assumption"),
                   height_provenance=room.get("height_provenance"), uncertainty=room.get("uncertainty"),
                   validation_state=room["validation"]["validation_state"],
                   confidence=None, confidence_note="no calibrated confidence score exists; see uncertainty (declared terms) and validation_state")
    else:
        out["status"] = "abstained"
    return out


LANG_LABELS = {"es": "es", "spa": "es", "spanish": "es", "en": "en", "eng": "en", "english": "en", "mul": "mixed", "mixed": "mixed"}


def language_breakdown(books: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-language (es/en) counts. Language is read from literal OCR/identity evidence only. Accuracy denominators stay
    'unavailable' until frozen human labels exist; nothing here is an accuracy figure."""
    groups: dict[str, dict[str, Any]] = {k: {"books": 0, "identified": 0, "with_visible_text": 0, "unidentified": 0, "needs_appraisal": 0}
                                          for k in ("es", "en", "mixed", "unknown")}
    for b in books:
        lang = LANG_LABELS.get(str(b.get("language") or "").strip().lower(), "unknown")
        g = groups[lang]
        g["books"] += 1
        g["identified"] += b["status"] == "identified"
        g["unidentified"] += b["status"] == "unidentified"
        g["needs_appraisal"] += b["status"] == "needs_appraisal"
        g["with_visible_text"] += bool(b.get("visible_text"))
    unavailable = {"status": "unavailable", "reason": "no frozen human labels: human-legible spine denominators and per-language title accuracy cannot be computed"}
    return {"counts": groups, "language_source": "literal OCR/identity language evidence or user statement; blank when unknown",
            "human_legible_spines": {"es": unavailable, "en": unavailable}, "title_accuracy": {"es": unavailable, "en": unavailable},
            "note": "counts describe provisional groups, not verified physical books; 'unknown' means no language evidence, not a third language"}


def completeness(books: list[dict[str, Any]], items: list[dict[str, Any]], room: dict[str, Any], totals: dict[str, Any]) -> dict[str, Any]:
    """What is known vs null/abstained, per field family, so a reader never has to infer absence from silence."""
    def split(lines, getter):
        known = sum(getter(x) is not None and getter(x) != "" for x in lines)
        return {"known": known, "null": len(lines) - known}
    money = lambda key: (lambda b: (b.get(key) or {}).get("amount"))

    def money_split(key):
        """known/null plus how many of the known lines are converted from a non-US market (converted: true) vs native US/USD."""
        out = split(books, money(key))
        priced = [b[key] for b in books if isinstance(b.get(key), dict) and b[key].get("amount") is not None and b.get("status") != "needs_appraisal"]
        out["converted"] = sum(o.get("converted") is True for o in priced)
        out["native_usd"] = len(priced) - out["converted"]
        out["converted_by_currency"] = {c: sum(o.get("original_currency") == c for o in priced if o.get("converted") is True)
                                        for c in sorted({o.get("original_currency") for o in priced if o.get("converted") is True})}
        return out
    room_fields = {}
    for k in ("length_m", "width_m", "height_m", "floor_area_m2", "wall_area_m2", "shelved_wall_area_m2"):
        v = room.get(k)
        room_fields[k] = {"value": v} if v is not None else {"value": None, "state": "null", "why": "; ".join(room.get("abstention_reasons") or []) or "not measured"}
    return {"books": {"total": len(books), "title": split(books, lambda b: b.get("title")), "author": split(books, lambda b: b.get("author")),
                      "edition": split(books, lambda b: b.get("edition")), "isbn": split(books, lambda b: b.get("isbn")),
                      "visible_text": split(books, lambda b: b.get("visible_text")),
                      "spine_height_cm": split(books, lambda b: b.get("spine_height_cm")), "spine_thickness_cm": split(books, lambda b: b.get("spine_thickness_cm")),
                      "replacement_cost": money_split("replacement_cost"), "used_value": money_split("used_value"),
                      "language": split(books, lambda b: b.get("language")),
                      "measurement_abstained": sum((b.get("measurement") or {}).get("status") == "abstained" for b in books)},
            "items": {"total": len(items), "category": split(items, lambda i: i.get("category")),
                      "dimensions_cm": {"known": 0, "null": len(items), "why": "item dimensions are not measured (no calibrated plane for items)"},
                      "replacement_cost": split(items, lambda i: (i.get("replacement_cost") or {}).get("amount") if isinstance(i.get("replacement_cost"), dict) else None),
                      "needs_appraisal": sum(i.get("status") == "needs_appraisal" for i in items)},
            "room": room_fields,
            "totals": {k: ({"value": v} if v is not None else {"value": None, "state": "null"}) for k, v in totals.items()
                       if k in ("shelf_run_m", "books_replacement_cost", "books_used_value", "items_replacement_cost_low", "items_replacement_cost_high", "excluded_from_totals")},
            # how many of the lines summed into books_replacement_cost / books_used_value are converted (recomputed from the lines on every attach path)
            "converted_lines": {"books_replacement_cost": money_split("replacement_cost")["converted"], "books_used_value": money_split("used_value")["converted"],
                                "note": CONVERTED_TOTALS_NOTE}}


def build_evidence_index(run_dir: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    """Index of every evidence file the packet's lines cite, with sha256 and size. Missing files are listed as missing."""
    import hashlib
    root = Path(run_dir)
    def entry(rel: str, kind: str, ref: str | None = None) -> dict[str, Any]:
        f = root / rel
        ok = f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(root.resolve())
        return {"kind": kind, "path": rel, "ref": ref, "exists": ok, "bytes": f.stat().st_size if ok else None,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest() if ok else None}
    frames: dict[str, str] = {}
    crops: dict[str, str] = {}
    for line in data["books"] + data["items"]:
        ev = line.get("evidence") or {}
        if line.get("frame_ref"):
            frames.setdefault(line["frame_ref"], line["id"])
        for r in line.get("reading_evidence") or []:
            if r.get("crop_path"):
                crops.setdefault(r["crop_path"], r.get("observation_id") or line["id"])
    arts = [C.ARTIFACTS[k] for k in ("frames", "detect", "read", "resolve", "dedup", "scale", "room", "price")] + ["calibration.json", "room_evidence.json"]
    files = [entry(a, "stage_artifact") for a in arts if (root / a).is_file()]
    files += [entry(f, "source_frame", ref) for f, ref in sorted(frames.items())]
    files += [entry(c, "ocr_crop", ref) for c, ref in sorted(crops.items())]
    dirs = {d: sum(1 for _ in (root / d).glob("*")) for d in ("detection_responses", "ocr_responses", "pricing_evidence") if (root / d).is_dir()}
    return {"schema_version": 1, "run_id": root.name, "files": files, "directories_not_enumerated": dirs,
            "missing": [f["path"] for f in files if not f["exists"]],
            "counts": {"source_frames_cited": len(frames), "ocr_crops_cited": len(crops), "stage_artifacts": sum(f["kind"] == "stage_artifact" for f in files)},
            "note": "every frame_ref in claim_packet.json appears here with its hash; sha256 of the packet itself is omitted to avoid circularity"}


def build_data(inv: dict[str, Any] | None, manifest: dict[str, Any], config: dict[str, Any] | None = None
               ) -> dict[str, Any]:
    """Packet ``data`` from an inventory envelope + manifest + optional corrections ledger.

    Shared by the offline CLI (``build``), the live snapshot and finalization, so there is exactly
    one place that turns instances into lines, totals and review entries. Corrections are applied as
    an overlay (``pipeline.corrections``); excluded instances are listed, never silently dropped.
    """
    config = config or {}
    stage_status = config.get("stage_status") or {}
    mdata = manifest["data"]
    frames = {f["frame_id"]: f for f in mdata["frames"]}
    inv_data = (inv or {}).get("data") or {}
    ledger = config.get("corrections") or []
    excluded: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    if ledger and inv_data:
        inv_data, excluded, applied = K.apply(inv_data, ledger, {fid: f["timestamp_s"] for fid, f in frames.items()})
    books = [_book_line(b, frames) for b in inv_data.get("books", [])]
    items = [_item_line(i, frames) for i in inv_data.get("items", [])]
    prices = config.get('prices')
    if prices is not None:
        from pipeline.price import identity_key
        price_rows = {r['ref_id']: r for r in prices.get('data', {}).get('prices', [])}
        for book in books:
            row = price_rows.get(book['id'])
            book['pricing_attempted'] = True
            wl = bool(row) and row.get('price_basis') == WORK_LEVEL
            # work-level rows are bound to the line by ref id + the accepted catalogue title (fail closed); legacy exact-edition rows by identity_key
            bound = bool(row) and ((book.get('title') is not None and row.get('title') == book['title']) if wl else row.get('identity_key') == identity_key(book))
            if bound and book['status'] != 'needs_appraisal':
                book['pricing_issues'] = row.get('issues', [])
                if row.get('needs_appraisal'):
                    book['status'] = 'needs_appraisal'
                    book['appraisal'] = {'reason': 'Price source exceeds configured appraisal threshold'}
                else:
                    want = ((row.get('equivalent_edition') or {}).get('isbn')) if wl else None
                    for key in ('replacement_cost', 'used_value'):
                        offer = row.get(key)
                        if not offer or offer.get('kind') != key: continue
                        if wl and not want: continue
                        if not _validate_price_line(offer, book, config.get('run_dir'), want):
                            book[key] = offer
                    if wl:
                        book['price_basis'], book['equivalent_isbn'] = WORK_LEVEL, want
                        book['equivalent_edition'] = row.get('equivalent_edition')
                        book['price_basis_note'] = WORK_LEVEL_NOTE
                        sm = row.get('second_market')
                        from pipeline.price import second_market_enabled
                        if second_market_enabled(config) and isinstance(sm, dict) and sm.get('market') == 'ES':
                            # second market (Spain, native EUR): validated under the ES/EUR rules, kept apart from the USD lines and totals
                            want_es = (sm.get('equivalent_edition') or {}).get('isbn')
                            lines = {key: sm[key] for key in ('replacement_cost', 'used_value')
                                     if want_es and isinstance(sm.get(key), dict) and sm[key].get('kind') == key
                                     and not _validate_price_line(sm[key], book, config.get('run_dir'), want_es, market='ES')}
                            book['second_market'] = {'market': 'ES', 'currency': 'EUR', 'source': sm.get('source'), 'note': sm.get('note'), 'fx': sm.get('fx'),
                                                     'equivalent_edition': sm.get('equivalent_edition'), 'issues': sm.get('issues', []), **lines}
            else:
                book['pricing_issues'] = ['price_row_missing_or_identity_changed_or_appraisal']
            nulls = {key: _null_reason(book, key) for key in ('replacement_cost', 'used_value') if book.get(key) is None}
            if nulls and book['status'] != 'needs_appraisal':
                book['pricing_null_reasons'] = nulls
    # Attach actual per-observation reading evidence without inventing instance identity.
    reading_rows = {r['observation_id']: r for r in config.get('readings', [])}
    for book in books:
        evidence_readings = [reading_rows[x] for x in book['evidence']['observation_ids'] if x in reading_rows]
        book['reading_evidence'] = evidence_readings
        if not book.get('visible_text'):
            best = max(evidence_readings, key=lambda r: len(r.get('visible_text') or ''), default={})
            book['visible_text'] = best.get('visible_text')
            book['language'] = book.get('language') or best.get('language_code')
    scale, room_data = config.get("scale"), config.get("room")
    shelf_runs, shelf_total = (None, None)
    if scale is not None:
        by_obs = {frozenset(m["observation_ids"]): m for m in scale.get("measurements", [])}
        for book in books:
            m = by_obs.get(frozenset(book["evidence"]["observation_ids"]))
            if m is None:
                book["measurement"] = {"status": "not_attempted", "reasons": ["instance not in the scale artifact's inventory"]}
            elif m["status"] == "measured":
                book["spine_height_cm"], book["spine_thickness_cm"] = m["spine_height_cm"], m["spine_thickness_cm"]
                book["measurement"] = {k: m[k] for k in ("status", "method", "calibration_ids", "plane_ids", "spine_height_interval_cm",
                                                         "spine_thickness_interval_cm", "observations_used", "validation_state",
                                                         "provenance_label", "interval_note", "per_observation") if k in m}
            else:
                book["measurement"] = {"status": "abstained", "reasons": m["abstention_reasons"], "validation_state": m["validation_state"]}
        # instance ids in scale artifact are keyed by that artifact's inventory; map through observation sets
        id_map = {m["instance_id"]: next((b["id"] for b in books if frozenset(b["evidence"]["observation_ids"]) == frozenset(m["observation_ids"])), None)
                  for m in scale.get("measurements", [])}
        present = {k for k, v in id_map.items() if v is not None}
        shelf_runs, shelf_total = SC.shelf_runs_for(scale, present)
    # Best-guess count: single sightings the capture cannot confirm (poor/bad second, untracked or unlinked frame) are kept as evidence in
    # unconfirmed_books but are not book lines, so book_count = the best guess. Lines with user corrections always stay.
    unconfirmed = [b for b in books if b.get("count_tier") == "unconfirmed" and not b.get("user_corrections")]
    if unconfirmed:
        drop = {b["id"] for b in unconfirmed}
        books = [b for b in books if b["id"] not in drop]
    totals = compute_totals(books, items, inv_data.get("counts") or {}, len(excluded), prices is not None)
    live_rel = ((config.get("provenance") or {}).get("capture_quality") or {}).get("count_reliability")
    if live_rel and "count_reliability" in totals:
        totals["count_reliability"] = live_rel  # the live monitor's reading of the same frames is canonical
    if any((b.get(k) or {}).get("converted") is True for b in books for k in ("replacement_cost", "used_value")):
        totals["notes"]["converted_lines"] = CONVERTED_TOTALS_NOTE + " Counts: completeness.converted_lines."
    if any(b.get("price_basis") == WORK_LEVEL for b in books):
        totals["notes"]["price_basis"] = ("Money totals sum WORK-LEVEL EQUIVALENT prices (one stated equivalent edition per work, ISBN on each line); "
                                          "they are not exact-edition prices of the shelf copies.")
    if scale is not None:
        totals["shelf_run_m"] = shelf_total
        totals["shelf_run_detail"] = shelf_runs
        totals["spine_dimensions_measured"] = sum(b.get("spine_height_cm") is not None for b in books)
        totals["spine_dimensions_abstained"] = sum((b.get("measurement") or {}).get("status") == "abstained" for b in books)
    fprobe = (mdata["input"].get("ffprobe") or {})
    market = {**DEFAULT_MARKET, **{k: v for k, v in (("country", config.get("country")), ("currency", config.get("currency"))) if v}}
    last_t = mdata["frames"][-1]["timestamp_s"] if mdata["frames"] else 0.0
    sweep = {"id": config.get("run_id") or "run", "captured_at": fprobe.get("creation_time") or config.get("captured_at"),
             "device": " ".join(x for x in (fprobe.get("device_make"), fprobe.get("device_model")) if x) or config.get("device"),
             "duration_s": fprobe.get("duration_s") or round(last_t, 2), **market,
             "source_video": mdata["input"].get("file_name"), "source_video_sha256": mdata["input"].get("sha256"),
             "source": mdata["input"].get("source", "recorded_video")}
    sweep.update(config.get("sweep_extra") or {})
    audit = mdata.get("audit") or {}
    inv_counts = ((inv or {}).get("data") or {}).get("counts") or {}
    data = {
        "completion_status": "provisional", "provisional_notice": ("PROVISIONAL, INCOMPLETE: source-backed prices where available; unknown prices excluded. Dimensions, shelf run and room geometry remain null." if prices is not None else PROVISIONAL_NOTICE),
        "pricing": prices.get('data') if prices is not None else None,
        "fixture": bool((inv or {}).get("fixture")),
        "sweep": sweep,
        "room": _room_block(room_data, scale),
        "books": books, "items": items, "totals": totals,
        "excluded_by_user": excluded,
        "unconfirmed_books": unconfirmed,
        "corrections": {"ledger": ledger, "applied": applied},
        "review_queue": build_review_queue(books, items, mdata, stage_status, bool((inv or {}).get("fixture")), scale, room_data) + [
            {"ref_id": w.get("wall_id"), "code": "room_auto_low_confidence_wall", "severity": "warning",
             "reason": f"automatic wall quad with confidence {w.get('confidence')} (frame {w.get('frame_id')}); verify or re-capture this wall"}
            for w in ((config.get("room_auto") or {}).get("per_wall") or []) if isinstance(w.get("confidence"), (int, float)) and w["confidence"] < 0.5] + [
            # Item detections not promoted to item lines (pipeline/items.py: one sighting, or best confidence under the category floor); never deleted,
            # listed in the inventory's data.item_candidates.
            {"ref_id": "item_candidates", "code": "item_candidates_unconfirmed", "severity": "info",
             "reason": f"{len(inv_data['item_candidates'])} item detection(s) not listed as items (one sighting or below the category confidence floor): "
                       + ", ".join(f"{k} x{v}" for k, v in sorted(Counter(c.get('category') for c in inv_data['item_candidates']).items()))}
            for _ in [0] if inv_data.get("item_candidates")],
        "coverage_gaps": [
            *({"type": g["type"], "start_s": g["start_s"], "end_s": g["end_s"], "detail": g["detail"]} for g in mdata.get("capture_gaps", [])),
            {"type": "room_boundaries_absent", "detail": ((audit.get("manual_review") or {}).get("room_boundaries") or {}).get(
                "detail", "Room boundaries not established from this capture (no manual review attached).")},
            {"type": "metric_reference_absent", "detail": ((audit.get("manual_review") or {}).get("metric_reference") or {}).get(
                "detail", "No metric reference established from this capture.")},
        ],
        "uncertainty": {"book_count_lower": totals["book_count_lower"], "book_count_upper": totals["book_count_upper"],
                        "uncertain_associations": totals["uncertain_associations"],
                        "unidentified_fraction": (round(totals["books_unidentified"] / totals["book_count"], 3)
                                                  if totals["book_count"] else None),
                        "basis": "from dedup stage if provided; null otherwise"},
        "count": {
            "book_groups": totals["book_count"], "book_count_lower": totals["book_count_lower"], "book_count_upper": totals["book_count_upper"],
            "count_if_near_ties_merged": (inv_counts or {}).get("count_if_near_ties_merged"),
            "proposed_method": (inv_counts or {}).get("association") or ((inv or {}).get("metrics") or {}).get("association") or "track",
            "alternative_method": config.get("alternative_count") or {"method": "track_v2", "status": "not_run", "book_groups": None,
                                                                       "note": "the two methods differ in how aggressively they link blurred sightings; the truth is most likely between them"},
            # one coherent band (low <= best <= high) from the totals; the near-tie count is a diagnostic (count_if_near_ties_merged), not an endpoint
            "uncertainty_band": _band(totals),
            "uncertain_observations": totals["uncertain_observations"], "uncertain_associations": totals["uncertain_associations"],
            "best_guess": totals["book_count"], "book_count_range": totals.get("book_count_range"), "confident_groups": totals.get("confident_groups"),
            "singleton_groups": totals.get("singleton_groups"), "unconfirmed_groups": totals.get("unconfirmed_groups"),
            "all_groups": (inv_counts or {}).get("all_groups"), "count_range_basis": totals.get("count_range_basis"),
            "capture_quality": (config.get("provenance") or {}).get("capture_quality"),
            "count_reliability": ((config.get("provenance") or {}).get("capture_quality") or {}).get("count_reliability") or totals.get("count_reliability"),
            "hand_count_comparison": "NOT computed inside this packet: any hand/ground-truth count stays outside the pipeline (evaluation truth is never read here). "
                                     "The packet states proposed groups, association bounds, uncertain links and capture quality; compare a hand count offline with eval/count_vs_truth.py.",
            "basis": "book_groups = proposed deduplicated physical-book groups; bounds are conditional on detected candidates (missed or merged spines are not in them)",
        },
        "stage_status": stage_status,
        "provenance": config.get("provenance") or {},
        "measurement_evidence": {"scale": _scale_evidence(scale), "room": room_data, "room_auto": config.get("room_auto")},
        "language_breakdown": language_breakdown(books),
        "completeness": completeness(books, items, _room_block(room_data, scale), totals),
        "evidence": {"manifest": C.ARTIFACTS["frames"], "inventory": C.ARTIFACTS["dedup"],
                     "frames_dir": "frames/", "selected_frames": [f["frame_id"] for f in mdata["frames"] if f.get("selected")]},
    }
    # The chain-independent panorama count (pipeline.panorama_count) replaces the track best guess when the capture is low/medium reliability and the panorama
    # registered >= 75% of the frames (pipeline.panorama_attach.decide); otherwise it is listed as a cross-check. Absent file = nothing changes.
    pc = config.get("panorama_count")
    if pc is None and config.get("run_dir"):
        from pipeline import panorama_attach as PA
        pc = PA.load(config["run_dir"])
    if pc:
        from pipeline import panorama_attach as PA
        PA.apply(data, pc)
    # One line per PHYSICAL book (pipeline.instances): track groups merged by panorama overlap and cross-segment registration, the count = the lines.
    # instances.json is written by the post-sweep step; absent file = the lines stay the track groups (every live sweep before the 2D build).
    art = config.get("instances")
    if art is None and config.get("run_dir"):
        from pipeline import instances as IN
        art = IN.load(config["run_dir"])
    if art:
        from pipeline import instances as IN
        IN.apply(data, art, ledger)       # ends with the headline choice (pipeline.panorama_attach.headline)
    return finalize_contract(data, config.get("run_dir"))


def _band(totals: dict[str, Any]) -> dict[str, Any]:
    lo, hi, best = totals.get("book_count_lower"), totals.get("book_count_upper"), totals.get("book_count")
    if lo is not None and best is not None:
        lo = min(lo, best)
    if hi is not None and best is not None:
        hi = max(hi, best)
    return {"low": lo, "best": best, "high": hi, "note": "coherent association band: low <= best guess <= high (totals.book_count_lower / book_count / book_count_upper); "
                                                         "covers association only, never spines the detector missed or fused"}


def build(inventory_path: str | Path | None, manifest_path: str | Path, run_dir: str | Path,
          config: dict[str, Any] | None = None) -> Path:
    config = config or {}
    run_dir = Path(run_dir)
    manifest = C.read_json(manifest_path)
    inv = C.read_json(inventory_path) if inventory_path and Path(inventory_path).is_file() else None
    config = {**config, "run_id": config.get("run_id") or run_dir.name, "run_dir": str(run_dir)}
    if 'prices' not in config and (run_dir/'prices.json').is_file(): config['prices'] = C.read_json(run_dir/'prices.json')
    if 'readings' not in config and (run_dir/'readings.json').is_file(): config['readings'] = C.read_json(run_dir/'readings.json').get('data', {}).get('readings', [])
    for key, fname in (("scale", C.ARTIFACTS["scale"]), ("room", C.ARTIFACTS["room"])):
        path = config.get(key + "_artifact")
        if key not in config and path and Path(path).is_file():
            config[key] = C.read_json(path).get("data")
    packet_data = build_data(inv, manifest, config)
    books, items = packet_data["books"], packet_data["items"]
    incomplete = [k for k, v in (config.get("stage_status") or {}).items() if v.get("status") not in ("ok", "complete", "skipped")]
    status = "ok" if inv is not None and not incomplete else "partial"
    issues = ([{"code": "stages_incomplete", "severity": "warning", "detail": "stage(s) not fully ok: " + ", ".join(incomplete)}] if incomplete else []) + ([] if inv is not None else [{"code": "no_inventory", "severity": "blocking",
                                          "detail": "inventory artifact missing; packet has no book/item lines"}])
    # Mandatory keys of the output contract are present by construction; verify and refuse to write a broken packet.
    env = C.envelope(config["run_id"], "packet", packet_data, status=status, issues=issues,
                     metrics={"book_lines": len(books), "item_lines": len(items), "review_entries": len(packet_data["review_queue"])})
    index = build_evidence_index(run_dir, packet_data)
    C.write_json(run_dir / "evidence_index.json", index)
    env["data"]["evidence"]["index"] = "evidence_index.json"
    env["data"]["evidence"]["index_summary"] = {**index["counts"], "missing": len(index["missing"])}
    if index["missing"]:
        env["issues"].append({"code": "evidence_missing", "severity": "blocking", "detail": f"{len(index['missing'])} cited evidence file(s) missing"})
    problems = verify_packet(env, manifest, run_dir)
    if problems:
        env["status"] = "partial"
        env["issues"] += [{"code": "packet_verification", "severity": "blocking", "detail": p} for p in problems]
    return C.write_json(run_dir / C.ARTIFACTS["packet"], env)


REQUIRED_TOP = ("sweep", "room", "books", "items", "totals", "review_queue")
REQUIRED_ROOM = ("length_m", "width_m", "height_m", "floor_area_m2", "wall_area_m2", "shelved_wall_area_m2",
                 "scale_method", "confidence")
REQUIRED_TOTALS = ("book_count", "books_identified", "books_unidentified", "shelf_run_m", "books_replacement_cost",
                   "books_used_value", "items_replacement_cost_low", "items_replacement_cost_high", "excluded_from_totals")


def _verify_spine(line: dict[str, Any], p: dict[str, Any]) -> list[str]:
    """A non-null spine dimension must carry measurement evidence from a valid calibration; otherwise it is a defect."""
    out = []
    h, t = line.get("spine_height_cm"), line.get("spine_thickness_cm")
    meas = line.get("measurement") or {}
    if h is None and t is None:
        if meas.get("status") == "measured":
            out.append(f"{line['id']}: measurement says measured but spine dimensions are null")
        return out
    height_only = h is not None and t is None and bool(meas.get("thickness_reason"))       # automatic measurement: a height whose thickness is withheld WITH a stated reason
    if (h is None or t is None) and not height_only:
        return [f"{line['id']}: spine height and thickness must be reported together"]
    if meas.get("status") != "measured":
        return [f"{line['id']}: spine dimensions need measurement evidence (status measured)"]
    cal_ids = meas.get("calibration_ids") or []
    scale_ev = (p.get("measurement_evidence") or {}).get("scale") or {}
    valid = {c["calibration_id"] for c in scale_ev.get("calibrations", []) if c.get("status") == "valid"}
    if not cal_ids or not set(cal_ids) <= valid:
        out.append(f"{line['id']}: measurement cites calibration(s) that are missing or invalid")
    if meas.get("validation_state") is None or not meas.get("per_observation"):
        out.append(f"{line['id']}: measurement lacks validation_state/per-observation evidence")
    for name, v, iv in (("height", h, meas.get("spine_height_interval_cm")),) + ((("thickness", t, meas.get("spine_thickness_interval_cm")),) if t is not None else ()):
        if not isinstance(v, (int, float)) or v <= 0 or v > 100:
            out.append(f"{line['id']}: implausible spine {name} {v!r} cm")
        if not iv or not (iv[0] <= v <= iv[1]):
            out.append(f"{line['id']}: spine {name} outside its stated interval")
    if isinstance(h, (int, float)) and isinstance(t, (int, float)) and h < t:
        out.append(f"{line['id']}: spine height smaller than thickness")
    return out


def _verify_shelf_run(p: dict[str, Any]) -> list[str]:
    t = p["totals"]
    v = t.get("shelf_run_m")
    if v is None:
        return []
    runs = ((p.get("measurement_evidence") or {}).get("scale") or {}).get("shelf_runs") or []
    if not runs or any(r.get("status") != "complete" for r in runs):
        return ["totals.shelf_run_m reported without complete shelf-run evidence"]
    auto = ((p.get("measurement_evidence") or {}).get("scale") or {}).get("method") == "auto_3d_shelf_plane_ray_casting"
    # the automatic shelf run is the extent of the measured spines on the shelf plane, not a sum of thicknesses: it needs measured heights
    measured = [b for b in p["books"] if b.get("spine_height_cm" if auto else "spine_thickness_cm") is not None]
    detail = t.get("shelf_run_detail") or []
    if detail and abs(sum(r["shelf_run_m"] for r in detail if r.get("shelf_run_m") is not None) - v) > 1e-3:
        return ["totals.shelf_run_m does not equal the sum of its shelf-run details"]
    if not measured:
        return ["totals.shelf_run_m reported but no book has a measured " + ("spine" if auto else "thickness")]
    return []


def _verify_room(p: dict[str, Any], manifest: dict[str, Any], run_dir: str | Path) -> list[str]:
    r = p["room"]
    geom = ("length_m", "width_m", "height_m", "floor_area_m2", "wall_area_m2")
    out = []
    set_keys = [k for k in geom if r.get(k) is not None]
    if r.get("shelved_wall_area_m2") is not None and not r.get("shelved_wall_provenance"):
        out.append("room.shelved_wall_area_m2 needs shelved_wall_provenance (user-supplied label)")
    if r.get("confidence") is not None:
        out.append("room.confidence must be null: no calibrated confidence score exists")
    if not set_keys:
        return out
    ev = (p.get("measurement_evidence") or {}).get("room") or {}
    if r.get("status") != "estimated" or ev.get("status") != "estimated":
        return out + ["room dimensions/areas reported without an estimated room artifact"]
    if len(set_keys) != len(geom):
        out.append("room dimensions and areas must be reported together")
    frames = {f["frame_id"] for f in manifest["data"]["frames"]}
    walls = (ev.get("shape") or {}).get("walls", [])
    if len(walls) < 2 or any(w.get("frame_id") not in frames for w in walls):
        out.append("room estimate lacks wall evidence that resolves to saved frames")
    if not (r.get("height_provenance") or {}).get("label"):
        out.append("room.height_m needs a provenance label (user-supplied vs measured)")
    if (ev.get("validation") or {}).get("validation_state") is None or not r.get("validation_state"):
        out.append("room estimate lacks validation_state")
    d, a = ev["dimensions"], ev["areas"]
    for k, src in (("length_m", d["length_m"]), ("width_m", d["width_m"]), ("height_m", d["height_m"]),
                   ("floor_area_m2", a["floor"]["m2"]), ("wall_area_m2", a["wall"]["m2"])):
        if r.get(k) != src:
            out.append(f"room.{k} differs from the room artifact")
    if ev["shape"].get("kind") == "rectangular" and r["length_m"] and abs(r["floor_area_m2"] - r["length_m"] * r["width_m"]) > 0.01 * r["floor_area_m2"] + 1e-3:
        out.append("room.floor_area_m2 != length x width")
    for m2k, ft2k in (("floor_area_m2", "floor_area_ft2"), ("wall_area_m2", "wall_area_ft2"), ("shelved_wall_area_m2", "shelved_wall_area_ft2")):
        if r.get(m2k) is not None and (r.get(ft2k) is None or abs(r[ft2k] - r[m2k] * RM.FT2_PER_M2) > 0.01 * r[m2k] * RM.FT2_PER_M2 + 1e-3):
            out.append(f"room.{ft2k} inconsistent with {m2k}")
    return out


def _price_row(p: dict[str, Any], book: dict[str, Any], offer: dict[str, Any], second: bool = False) -> dict[str, Any]:
    """The pricing row a line's offer came from: by the line id, or by a member group id of a merged instance line (pipeline/instances.py), preferring the row whose
    equivalent ISBN is the offer's."""
    ids = [book['id']] + [m for m in (book.get('member_group_ids') or []) if m != book['id']]
    rows = [r for r in ((p.get('pricing') or {}).get('prices') or []) if r.get('ref_id') in ids]
    want = (offer or {}).get('equivalent_isbn')
    for r in rows:
        ed = ((r.get('second_market') or {}).get('equivalent_edition') if second else r.get('equivalent_edition')) or {}
        if want and ed.get('isbn') == want:
            return r
    return rows[0] if rows else {}


def verify_packet(env: dict[str, Any], manifest: dict[str, Any], run_dir: str | Path) -> list[str]:
    """Consistency checks: required keys, evidence resolves, totals match lines, no fabricated numbers."""
    p = env.get("data") or {}
    problems = []
    for k in REQUIRED_TOP:
        if k not in p:
            problems.append(f"missing required key {k}")
    if problems:
        return problems
    problems += [f"room.{k} missing" for k in REQUIRED_ROOM if k not in p["room"]]
    problems += [f"totals.{k} missing" for k in REQUIRED_TOTALS if k not in p["totals"]]
    if p.get("completion_status") != "provisional":
        problems.append("completion_status must be 'provisional' in this milestone")
    dims = {f["frame_id"]: (f["width_px"], f["height_px"]) for f in manifest["data"]["frames"]}
    ids = set()
    for kind in ("books", "items"):
        for line in p[kind]:
            if line["id"] in ids:
                problems.append(f"{line['id']}: duplicate id")
            ids.add(line["id"])
            ev = line.get("evidence") or {}
            fid = ev.get("frame_id")
            if fid not in dims:
                problems.append(f"{line['id']}: frame {fid!r} not in manifest")
                continue
            if not line.get("frame_ref") or not (Path(run_dir) / line["frame_ref"]).is_file():
                problems.append(f"{line['id']}: frame_ref does not open ({line.get('frame_ref')})")
            if ev.get("bbox_xyxy") is not None:
                problems += [f"{line['id']}: {m}" for m in C.check_bbox(ev["bbox_xyxy"], *dims[fid])]
            if kind == "books":
                problems += _verify_spine(line, p)
                if (line["status"] == "identified" and line.get("identity_source") != "user_statement"
                        and not (line.get("title") and line.get("identity_source_url"))):
                    problems.append(f"{line['id']}: identified without title/source_url")
    t = p["totals"]
    b, i = p["books"], p["items"]
    line_ids = {x["id"] for x in b + i}
    for ex in p.get("excluded_by_user", []):
        if ex["instance_id"] in line_ids:
            problems.append(f"{ex['instance_id']}: listed both as a line and as user-excluded")
    if t.get("user_excluded_count") != len(p.get("excluded_by_user", [])):
        problems.append("totals.user_excluded_count mismatch")
    # The headline (totals.book_count) is the panorama count when headline_estimator == "panorama"; the line count is then totals.book_lines
    head_pan = t.get("headline_estimator") == "panorama"
    n_lines = t.get("book_lines") if head_pan else t["book_count"]
    if t.get("book_lines") is not None and t["book_lines"] != len(b):
        problems.append("totals.book_lines != number of book lines")
    if head_pan:
        if (p.get("count") or {}).get("best_guess") != t["book_count"]:
            problems.append("count.best_guess != totals.book_count")
        lo, hi = t.get("book_count_lower"), t.get("book_count_upper")
        if lo is None or hi is None or not lo <= t["book_count"] <= hi:
            problems.append("panorama headline outside its range")
    if t.get("count_estimator") == "instances":     # the lines are physical instances (pipeline/instances.py): the line count IS the instance count; members add up to the track groups
        if n_lines != len(b):
            problems.append("totals.book_count != number of book lines")
        members = sum(len(x.get("member_group_ids") or [x["id"]]) for x in b + list(p.get("unconfirmed_books") or []))
        if t.get("book_count_track") != members:
            problems.append("totals.book_count_track != track groups covered by the instance lines")
        if (p.get("count") or {}).get("best_guess") != t["book_count"]:
            problems.append("count.best_guess != totals.book_count")
        band = (p.get("count") or {}).get("uncertainty_band") or {}
        if not (band.get("low") is not None and band["low"] <= band.get("best", t["book_count"]) <= band.get("high", t["book_count"])):
            problems.append("count.uncertainty_band is not low <= best <= high")
    elif t.get("count_estimator") == "panorama":      # older packets: the best guess is the panorama count; the book lines are the track groups
        if t.get("book_count_track") != len(b):
            problems.append("totals.book_count_track != number of book lines")
        if (p.get("count") or {}).get("best_guess") != t["book_count"]:
            problems.append("count.best_guess != totals.book_count")
    elif n_lines != len(b):
        problems.append("totals.book_count != number of book lines")
    if t["books_identified"] != sum(x["status"] == "identified" for x in b):
        problems.append("totals.books_identified mismatch")
    if t["books_unidentified"] != sum(x["status"] == "unidentified" for x in b):
        problems.append("totals.books_unidentified mismatch")
    if t["item_count"] != len(i):
        problems.append("totals.item_count mismatch")
    for book in b:
        for key in ('replacement_cost', 'used_value'):
            offer = book.get(key)
            if offer:
                if not p.get('pricing'):
                    problems.append(book['id']+': '+key+' must be null without a pricing stage')
                want = None
                if offer.get('price_basis') == WORK_LEVEL:
                    row = _price_row(p, book, offer)
                    want = (row.get('equivalent_edition') or {}).get('isbn')
                    if not want or offer.get('equivalent_isbn') != want or book.get('equivalent_isbn') != want or book.get('price_basis') != WORK_LEVEL:
                        problems.append(book['id']+': work-level price line must carry price_basis and the equivalent ISBN of its price row')
                problems += [book['id']+': '+e for e in _validate_price_line(offer, book, run_dir, want)]
                if offer.get('kind') != key or book.get('status') == 'needs_appraisal': problems.append(book['id']+': invalid price kind/appraisal')
    for book in b:
        sm = book.get('second_market')
        if isinstance(sm, dict):
            row = _price_row(p, book, sm.get('replacement_cost') or sm.get('used_value') or {}, second=True)
            want_es = ((row.get('second_market') or {}).get('equivalent_edition') or {}).get('isbn')
            for key in ('replacement_cost', 'used_value'):
                offer = sm.get(key)
                if offer:
                    problems += [book['id']+': second market '+key+': '+e for e in _validate_price_line(offer, book, run_dir, want_es, market='ES')]
                    if offer.get('kind') != key or offer.get('currency') != 'EUR' or offer.get('equivalent_isbn') != want_es: problems.append(book['id']+': second market '+key+': invalid kind/currency/ISBN')
    expected = compute_totals(b, i, {}, len(p.get('excluded_by_user', [])), p.get("pricing") is not None)
    if t.get('second_market') != expected.get('second_market'): problems.append('totals.second_market mismatch')
    for key in ('books_replacement_cost','books_used_value','excluded_from_totals'):
        if t.get(key) != expected[key]: problems.append('totals.'+key+' mismatch')
    for k in ("items_replacement_cost_low", "items_replacement_cost_high"):
        if t[k] != expected[k]:
            problems.append(f"totals.{k} mismatch (sum of the sourced item lines)")
    for line in i:
        if line.get("status") not in C.ITEM_STATUSES:
            problems.append(f"{line['id']}: item status {line.get('status')!r} not in priced | range | needs_appraisal")
        elif line["status"] in ("priced", "range") and not _item_price(line.get("replacement_cost")):
            problems.append(f"{line['id']}: item {line['status']} without a sourced low/high (source, url, retrieved_at)")
    problems += _verify_shelf_run(p)
    problems += _verify_room(p, manifest, run_dir)
    problems += _verify_second_country(p, run_dir)
    refs = ids | {"capture", "room", "prices", "dimensions", "fixture", "item_candidates", "second_country"}
    seen_refs = set()
    for r in p["review_queue"]:
        if r["ref_id"] not in refs and not str(r["ref_id"]).startswith("stage:"):
            problems.append(f"review_queue refers to unknown {r['ref_id']}")
        if r["ref_id"] in seen_refs:
            problems.append(f"review_queue has more than one entry for {r['ref_id']}")
        seen_refs.add(r["ref_id"])
        if not r.get("reason") or not isinstance(r.get("reasons"), list) or not r["reasons"]:
            problems.append(f"review_queue entry {r['ref_id']} lacks reason / reasons")
    sev = [SEVERITY_RANK.get(r.get("severity"), 3) for r in p["review_queue"]]
    if sev != sorted(sev):
        problems.append("review_queue not ordered by severity (blocking, warning, info)")
    return problems


def _verify_second_country(p: dict[str, Any], run_dir: str | Path) -> list[str]:
    sc = p.get("second_country")
    if not isinstance(sc, dict):
        return []
    from pipeline import second_country as SCN
    return SCN.verify(sc, p, run_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build provisional claim_packet.json from an inventory artifact")
    ap.add_argument("--run", required=True, help="run directory containing manifest.json and inventory.json")
    a = ap.parse_args(argv)
    run = Path(a.run)
    print(build(run / C.ARTIFACTS["dedup"], run / C.ARTIFACTS["frames"], run, {}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
