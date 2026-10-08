"""The claim packet meets the claim packet's output contract on every new sweep.

* items: status is always priced | range | needs_appraisal (art default; no sourced price -> needs_appraisal + review entry, never an invented figure);
  items_replacement_cost_low/high computed from the lines (null when no line has a sourced price);
* review queue: one entry per ref_id (short reason + every detail in reasons/codes), blocking -> warning -> info, idempotent, layers can still filter by code;
* sweep.device from the phone's capture_info;
* second_country: the same (first 10) priced books priced for a second country, a setting; lines re-validated from raw bytes; totals from the lines.
Offline: synthetic AbeBooks / AbeBooks UK pages (tests.test_price_converted.web)."""
import copy

import pytest

from pipeline import contracts as C, packet as P, report as R, second_country as SCN
from tests.test_measurement import manifest
from tests.test_packet_work_level_price import ACCEPTED, _inv
from tests.test_price_converted import WORK, native, run, web  # noqa: F401 - web is a fixture
from tests.test_price_work_level import EN_NEW, entry, listing

IPHONE_UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.0.1 Mobile/15E148 Safari/604.1"


def _item(iid, category, **kw):
    return {"instance_id": iid, "observation_ids": [iid[3:]], "frame_ref": "f00000", "bbox_xyxy": [10, 10, 60, 60], "kind": "non_book", "category": category,
            "description": "Pretrained YOLO-World category prediction", "confidence": 0.8, "uncertain_association": False, **kw}


def _inv_items(*items):
    inv = _inv(ACCEPTED)
    inv["data"]["items"] = list(items)
    return inv


SOURCED = {"low": 40.0, "high": 120.0, "currency": "USD", "source": "Retailer listing", "url": "https://example.com/lamp", "retrieved_at": "2026-10-07T10:00:00+00:00"}


# ------------------------------------------------------------------------------------------------------------------------------------- items
def test_every_item_gets_a_contract_status_and_unpriced_items_are_routed_to_review_never_priced():
    inv = _inv_items(_item("it-f00000-o001", "lamp"), _item("it-f00000-o002", "frame"), _item("it-f00000-o003", "portrait", art_is_print=True),
                     _item("it-f00000-o004", "lamp", replacement_cost=dict(SOURCED)), _item("it-f00000-o005", "chair", replacement_cost={"low": 50, "high": 50}))
    d = P.build_data(inv, manifest(), {"run_id": "t"})
    st = {i["id"]: (i["status"], i["status_basis"]) for i in d["items"]}
    assert st == {"it-f00000-o001": ("needs_appraisal", "no_sourced_price"), "it-f00000-o002": ("needs_appraisal", "art_default_needs_appraisal"),
                  "it-f00000-o003": ("needs_appraisal", "no_sourced_price"),            # a print is not art-appraisal, but nothing priced it either
                  "it-f00000-o004": ("range", "sourced_range"),
                  "it-f00000-o005": ("needs_appraisal", "no_sourced_price")}            # a figure without source/url/date is not a price
    assert all(i["status"] in C.ITEM_STATUSES and i["status_reason"] for i in d["items"])
    t = d["totals"]
    assert (t["items_replacement_cost_low"], t["items_replacement_cost_high"], t["items_range"], t["items_unpriced"], t["items_no_sourced_price"]) == (40.0, 120.0, 1, 4, 3)
    codes = {r["ref_id"]: r["codes"] for r in d["review_queue"]}
    assert "item_no_sourced_price" in codes["it-f00000-o001"] and "needs_appraisal" in codes["it-f00000-o002"] and "it-f00000-o004" not in codes
    assert "excluded" in t["notes"]["items_totals"]


def test_item_totals_are_null_without_any_sourced_item_line_and_verify_rejects_a_bad_item():
    inv = _inv_items(_item("it-f00000-o001", "lamp"))
    d = P.build_data(inv, manifest(), {"run_id": "t"})
    assert d["totals"]["items_replacement_cost_low"] is None and d["totals"]["items_replacement_cost_high"] is None
    env = C.envelope("t", "packet", d, status="partial")
    bad = copy.deepcopy(env)
    bad["data"]["items"][0]["status"] = None
    assert any("not in priced | range | needs_appraisal" in x for x in P.verify_packet(bad, manifest(), "/nonexistent"))
    bad = copy.deepcopy(env)
    bad["data"]["items"][0]["status"] = "priced"
    assert any("without a sourced low/high" in x for x in P.verify_packet(bad, manifest(), "/nonexistent"))
    bad = copy.deepcopy(env)
    bad["data"]["totals"]["items_replacement_cost_low"] = 10.0
    assert any("items_replacement_cost_low mismatch" in x for x in P.verify_packet(bad, manifest(), "/nonexistent"))


def test_finalize_fixes_an_old_packet_with_null_item_statuses():
    d = P.build_data(_inv_items(_item("it-f00000-o001", "lamp"), _item("it-f00000-o002", "painting")), manifest(), {"run_id": "t"})
    for i in d["items"]:
        i["status"] = None if i["category"] == "lamp" else "needs_appraisal"
        i.pop("status_basis")
    d["review_queue"] = P.flat_review(d["review_queue"])
    P.finalize_contract(d)
    assert [i["status"] for i in d["items"]] == ["needs_appraisal", "needs_appraisal"] and d["items"][0]["status_basis"] == "no_sourced_price"


# ------------------------------------------------------------------------------------------------------------------------------------- review queue
FLAT = [{"ref_id": "bk-1", "code": "unidentified_book", "severity": "info", "reason": "Counted but unidentified."},
        {"ref_id": "bk-1", "code": "replacement_cost_unknown", "severity": "warning", "reason": "No validated physical US/USD offer."},
        {"ref_id": "bk-1", "code": "used_value_unknown", "severity": "warning", "reason": "No validated physical US/USD offer."},
        {"ref_id": "bk-1", "code": "pricing_exclusion", "severity": "warning", "reason": "no supported printed ISBN/physical edition"},
        {"ref_id": "capture", "code": "no_clean_frame", "severity": "warning", "reason": "1-2s: flagged"},
        {"ref_id": "capture", "code": "no_clean_frame", "severity": "warning", "reason": "3-4s: flagged"},
        {"ref_id": "room", "code": "room_not_measured", "severity": "blocking", "reason": "Room not established."},
        {"ref_id": "bk-1", "code": "unidentified_book", "severity": "info", "reason": "Counted but unidentified."}]       # exact duplicate


def test_one_entry_per_ref_with_a_short_reason_every_detail_kept_and_severity_order():
    q = P.consolidate_review(FLAT)
    assert [r["ref_id"] for r in q] == ["room", "bk-1", "capture"]
    bk = q[1]
    assert bk["severity"] == "warning" and bk["codes"] == ["replacement_cost_unknown", "used_value_unknown", "pricing_exclusion", "unidentified_book"]
    assert bk["reason"] == "No sourced price (excluded from money totals); unidentified (no title guessed)" and bk["detail_count"] == 4
    assert q[2]["reason"] == "No clean frame (x2)" and len(q[2]["reasons"]) == 2
    assert q[0]["reason"] == "Room not established."                                                     # a single detail keeps its own text
    assert P.consolidate_review(q) == q and len(P.flat_review(q)) == 7                                 # idempotent, nothing lost but the exact duplicate


def test_a_built_packet_has_one_review_entry_per_ref_and_layers_still_filter_by_code():
    inv = _inv_items(_item("it-f00000-o001", "lamp"))
    inv["data"]["books"][0]["identity"] = {"decision": "no_match", "visible_text": "XYZ"}
    inv["data"]["books"][0]["uncertain_association"] = True
    d = P.build_data(inv, manifest(), {"run_id": "t"})
    refs = [r["ref_id"] for r in d["review_queue"]]
    assert len(refs) == len(set(refs)) and "bk-001" in refs
    env = C.envelope("t", "packet", d, status="partial")
    assert not [x for x in P.verify_packet(env, manifest(), "/nonexistent") if "review_queue" in x]
    # a layer that drops one code (identity_attach drops unidentified_book of an identified line) works on the details
    q = P.consolidate_review([r for r in P.flat_review(d["review_queue"]) if r["code"] != "unidentified_book"])
    assert "unidentified_book" not in next(r for r in q if r["ref_id"] == "bk-001")["codes"]
    html = R.render_html(env, thumbs=False)
    assert "Review queue (" in html and "details" in html


def test_verify_rejects_two_entries_for_one_ref_and_a_severity_disorder():
    d = P.build_data(_inv(ACCEPTED), manifest(), {"run_id": "t"})
    env = C.envelope("t", "packet", d, status="partial")
    bad = copy.deepcopy(env)
    bad["data"]["review_queue"] = bad["data"]["review_queue"] + [dict(bad["data"]["review_queue"][0])]
    assert any("more than one entry" in x for x in P.verify_packet(bad, manifest(), "/nonexistent"))
    bad = copy.deepcopy(env)
    bad["data"]["review_queue"] = list(reversed(P.consolidate_review(FLAT)))
    assert any("ordered by severity" in x for x in P.verify_packet(bad, manifest(), "/nonexistent"))


def test_stage_entries_follow_the_current_stage_status():
    d = P.build_data(_inv(ACCEPTED), manifest(), {"run_id": "t", "stage_status": {"read": {"status": "partial", "detail": "120 of 964 read"}}})
    assert any(r["ref_id"] == "stage:read" for r in d["review_queue"])
    d["stage_status"]["read"] = {"status": "ok", "detail": "finish reading: done"}
    P.finalize_contract(d)
    assert not any(r["ref_id"] == "stage:read" for r in d["review_queue"])


# ------------------------------------------------------------------------------------------------------------------------------------- device
def test_device_comes_from_the_phone_capture_info():
    cq = {"device_label": "Back Camera", "frame_size": "1920x1080", "fps_effective": 4.94,
          "device": {"width": 1080, "height": 1920, "frame_rate": 60, "facing_mode": "environment", "user_agent": IPHONE_UA}}
    dev, detail = P.device_from_capture(cq)
    assert dev == "iPhone (iOS 18.7, Safari 27.0.1); camera: Back Camera, facing environment; stream 1080x1920 @ 60 fps; saved frames 1920x1080 at 4.94 fps"
    assert detail["user_agent"] == IPHONE_UA and detail["stream_fps"] == 60
    android = "Mozilla/5.0 (Linux; Android 14; Pixel 8 Build/AP1A) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Mobile Safari/537.36"
    assert P.device_from_capture({"label": "camera2 0, facing back", "width": 1920, "height": 1080, "frame_rate": 30, "user_agent": android})[0].startswith(
        "Pixel 8 (Android 14, Chrome 129.0)")
    assert P.device_from_capture({}) == (None, None) and P.device_from_capture(None) == (None, None)
    d = P.build_data(_inv(ACCEPTED), manifest(), {"run_id": "t", "provenance": {"capture_quality": cq}})
    assert d["sweep"]["device"] == dev and d["sweep"]["device_detail"]["camera_label"] == "Back Camera"
    assert P.build_data(_inv(ACCEPTED), manifest(), {"run_id": "t"})["sweep"]["device"] is None       # nothing reported -> null


# ------------------------------------------------------------------------------------------------------------------------------------- second country
def test_second_country_is_a_setting(monkeypatch):
    monkeypatch.delenv("LIBRARY_SECOND_COUNTRY", raising=False)
    assert SCN.setting() == ("GB", "default")
    monkeypatch.setenv("LIBRARY_SECOND_COUNTRY", "es")
    assert SCN.setting() == ("ES", "env LIBRARY_SECOND_COUNTRY")
    assert SCN.setting({"second_country": "off"})[0] is None and SCN.setting({"second_country": "JP"})[0] is None
    monkeypatch.setenv("LIBRARY_SECOND_COUNTRY_MAX_REQUESTS", "7")
    assert SCN.max_requests() == 7


def test_pick_books_takes_the_first_ten_identified_priced_lines_in_display_order():
    books = [{"id": f"bk-{i}", "display_index": 30 - i, "status": "identified", "replacement_cost": {"amount": 5.0, "isbn": f"97800000000{i:02d}"}} for i in range(14)]
    books += [{"id": "bk-x", "display_index": 0, "status": "unidentified", "replacement_cost": None}]
    picked = SCN.pick_books({"books": books})
    assert len(picked) == 10 and [b["display_index"] for b in picked] == sorted(b["display_index"] for b in picked) and picked[0]["id"] == "bk-13"


def _priced_packet(web, root):
    """A packet with one identified English book priced in the US (work-level line), written to root/claim_packet.json."""
    web[0][("us", EN_NEW)] = [listing(EN_NEW, 14.0, lang="ENG"), listing(EN_NEW, 6.0, "USED_GOOD", lang="ENG")]
    row = run(web, [entry("/books/OL3M", EN_NEW, lang=("eng",))], {**WORK, "visible_text": "Acosada", "language_code": "en"}, {"converted_fallback": False})
    row["title"], row["ref_id"] = "Acosada", "bk-001"
    prices = C.envelope("t", "price", {"prices": [row], "market": "US", "currency": "USD", "price_basis": "work_level_equivalent"})
    d = P.build_data(_inv(ACCEPTED), manifest(), {"run_id": "t", "prices": prices, "run_dir": str(root)})
    assert d["books"][0]["replacement_cost"]["amount"] == 14.0 and d["second_country"]["status"] in ("not_run", "off")
    env = C.envelope("t", "packet", d, status="partial")
    C.write_json(root / C.ARTIFACTS["packet"], env)
    return env


def test_the_same_books_are_priced_in_the_second_country_from_validated_listings_with_sources_and_dates(web, monkeypatch):
    monkeypatch.delenv("LIBRARY_SECOND_COUNTRY", raising=False)
    pages, urls, provider, root, _ = web
    _priced_packet(web, root)
    pages[("gb", EN_NEW)] = [native(EN_NEW, 9.5, "GBP", "GBR", lang="ENG"), native(EN_NEW, 3.25, "GBP", "GBR", "USED_GOOD", lang="ENG"),
                             native(EN_NEW, 1.0, "GBP", "USA", lang="ENG")]                                        # a US seller on the UK site is not a UK price
    n_before = len(urls)
    art = SCN.compute(root, country="GB", online=True, provider=provider, cap=10)
    assert all("abebooks.co.uk" in u for u in urls[n_before:]) and len(urls) - n_before <= 2
    b = art["data"]["books"][0]
    assert b["replacement_cost"]["amount"] == 9.5 and b["used_value"]["amount"] == 3.25 and b["replacement_cost"]["currency"] == "GBP"
    assert b["replacement_cost"]["url"].startswith("https://www.abebooks.co.uk/") and b["replacement_cost"]["retrieved_at"] and b["replacement_cost"]["evidence"]["sha256"]
    env = C.read_json(root / C.ARTIFACTS["packet"])
    P.finalize_contract(env["data"], root)
    sc = env["data"]["second_country"]
    assert sc["status"] == "ok" and sc["country"] == "GB" and sc["currency"] == "GBP"
    assert sc["totals"]["replacement_cost"] == 9.5 and sc["totals"]["used_value"] == 3.25 and sc["totals"]["us_replacement_cost_same_books_usd"] == 14.0
    assert env["data"]["totals"]["books_replacement_cost"] == 14.0                                            # never mixed into the US/USD totals
    assert SCN.verify(sc, env["data"], root) == []
    html = R.render_html(env, thumbs=False)
    assert "Second country: the same books priced in GB / GBP" in html and "9.5 GBP" in html
    bad = copy.deepcopy(sc)
    bad["books"][0]["second"]["replacement_cost"]["amount"] = 7.0                                               # a figure the raw page does not show
    assert any("raw source" in x for x in SCN.verify(bad, env["data"], root))
    bad = copy.deepcopy(sc)
    bad["totals"]["used_value"] = 99.0
    assert any("totals.used_value" in x for x in SCN.verify(bad, env["data"], root))


def test_no_listing_stays_null_with_the_reason_and_the_request_cap_holds(web):
    pages, urls, provider, root, _ = web
    _priced_packet(web, root)
    art = SCN.compute(root, country="GB", online=True, provider=provider, cap=10)                              # no UK page rows
    b = art["data"]["books"][0]
    assert b["replacement_cost"] is None and b["used_value"] is None and any("no_validated_GB_GBP_listing" in i for i in b["issues"])
    env = C.read_json(root / C.ARTIFACTS["packet"])
    P.finalize_contract(env["data"], root)
    assert env["data"]["second_country"]["status"] == "partial" and env["data"]["second_country"]["totals"]["replacement_cost"] is None
    entry_ = next(r for r in env["data"]["review_queue"] if r["ref_id"] == "second_country")
    assert "no validated GB listing" in entry_["reason"] and "fewer than the 10" in entry_["reason"]
    n = len(urls)
    capped = SCN.compute(root, country="GB", online=True, provider=provider, cap=0)
    assert len(urls) == n and any("request_cap" in i for i in capped["data"]["books"][0]["issues"])


def test_a_line_whose_us_isbn_changed_is_marked_stale_and_excluded(web):
    pages, urls, provider, root, _ = web
    _priced_packet(web, root)
    pages[("gb", EN_NEW)] = [native(EN_NEW, 9.5, "GBP", "GBR", lang="ENG")]
    SCN.compute(root, country="GB", online=True, provider=provider, cap=10)
    env = C.read_json(root / C.ARTIFACTS["packet"])
    env["data"]["books"][0]["replacement_cost"] = dict(env["data"]["books"][0]["replacement_cost"], isbn="9780000000001")
    env["data"]["books"][0]["used_value"] = None
    P.finalize_contract(env["data"], root)
    sc = env["data"]["second_country"]
    assert sc["books"][0]["stale"] and sc["totals"]["books"] == 0 and sc["totals"]["replacement_cost"] is None and sc["status"] == "partial"
