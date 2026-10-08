"""Packet JSON + report.html show price_basis and the chosen equivalent ISBN on every priced line; nulls show their reason (synthetic fixtures, no network)."""
import copy

import pytest

from pipeline import contracts as C, packet as P, report as R, price as PR
from tests.test_measurement import manifest
from tests.test_price_work_level import ES_NEW, abe, entry, listing, run  # noqa: F401  (abe is a fixture)


def _inv(identity):
    inst = {"instance_id": "bk-001", "observation_ids": ["f00000-o000"], "frame_ref": "f00000", "bbox_xyxy": [100, 100, 140, 400], "kind": "book", "confidence": 0.9,
            "uncertain_association": False, "identity": identity}
    return {"data": {"books": [inst], "items": [], "registrations": [], "counts": {}}, "fixture": False}


ACCEPTED = {"decision": "accepted", "title": "Acosada", "author": "Brian Freeman", "source_url": "https://openlibrary.org/works/OL1W", "language_code": "es"}


def _packet(tmp_path, row, **config):
    prices = C.envelope("t", "price", {"prices": [row], "market": "US", "currency": "USD", "price_basis": "work_level_equivalent"})
    data = P.build_data(_inv(ACCEPTED), manifest(), {"run_id": "t", "prices": prices, "run_dir": str(tmp_path), **config})
    return C.envelope("t", "packet", data, status="partial")


WORK = {"id": "bk-001", "work": "/works/OL1W", "title": "Acosada", "visible_text": "Acosada\nBrian Freeman", "language_code": "es"}


def _priced_row(abe):
    abe[0][ES_NEW] = [listing(ES_NEW, 14.5), listing(ES_NEW, 9.0, "USED_GOOD")]
    return run(abe, [entry("/books/OL2M", ES_NEW, date="Mar 19, 2019")], WORK)


def test_priced_lines_carry_basis_isbn_and_render_the_label(abe):
    row = _priced_row(abe)
    env = _packet(abe[3], row)
    b = env["data"]["books"][0]
    assert b["replacement_cost"]["amount"] == 14.5 and b["used_value"]["amount"] == 9.0
    assert b["price_basis"] == "work_level_equivalent" and b["equivalent_isbn"] == ES_NEW and b["equivalent_edition"]["isbn"] == ES_NEW
    assert "NOT the exact edition" in b["price_basis_note"] and "pricing_null_reasons" not in b
    for key in ("replacement_cost", "used_value"):
        assert b[key]["price_basis"] == "work_level_equivalent" and b[key]["equivalent_isbn"] == ES_NEW
    assert "WORK-LEVEL EQUIVALENT" in env["data"]["totals"]["notes"]["price_basis"]
    html = R.render_html(env, thumbs=False)
    assert html.count("work-level equivalent (not exact edition)") == 2 and html.count(f"equivalent ISBN {ES_NEW}") == 2
    assert "WORK-LEVEL EQUIVALENT" in html and "work-level equivalent, not exact-edition" in html
    assert not [x for x in P.verify_packet(env, manifest(), abe[3]) if "bk-001" in x and ("price" in x or "offer" in x or "work-level" in x or "raw" in x)]


def test_offer_with_a_different_isbn_than_its_row_is_not_accepted(abe):
    row = _priced_row(abe)
    bad = copy.deepcopy(row); bad["replacement_cost"]["equivalent_isbn"] = "9780143106654"
    b = _packet(abe[3], bad)["data"]["books"][0]
    assert b["replacement_cost"] is None and b["used_value"]["amount"] == 9.0
    assert "replacement_cost" in b["pricing_null_reasons"]


def test_unlabelled_work_level_line_fails_packet_verification(abe):
    env = _packet(abe[3], _priced_row(abe))
    env["data"]["books"][0].pop("equivalent_isbn")
    assert any("work-level price line must carry price_basis" in x for x in P.verify_packet(env, manifest(), abe[3]))


def test_null_lines_show_their_reason_in_json_and_html(abe):
    row = run(abe, [entry("/books/OL1M", "9788494326777")], WORK)                         # no checksum-valid ISBN -> null with an issue
    env = _packet(abe[3], row)
    b = env["data"]["books"][0]
    assert b["replacement_cost"] is None and b["used_value"] is None
    assert "no_isbn_bearing_edition" in b["pricing_null_reasons"]["replacement_cost"]
    html = R.render_html(env, thumbs=False)
    assert "no_isbn_bearing_edition" in html and "equivalent ISBN 9" not in html     # reason shown, no ISBN claimed


def test_row_for_a_different_title_is_not_bound_to_the_line(abe):
    row = _priced_row(abe); row["title"] = "Otro libro"
    b = _packet(abe[3], row)["data"]["books"][0]
    assert b["replacement_cost"] is None and b["pricing_issues"] == ["price_row_missing_or_identity_changed_or_appraisal"]


# ---- second market (Spain, native EUR): kept apart from the USD lines and totals ------------------------------------------------------------------------------------
def _es_row(abe):
    from tests.test_price_work_level import listing_es
    abe[0][ES_NEW] = [listing(ES_NEW, 14.5)]
    abe[0][("es", ES_NEW)] = [listing_es(ES_NEW, 11.0), listing_es(ES_NEW, 3.5, cond="USED_GOOD")]
    return run(abe, [entry("/books/OL2M", ES_NEW, date="Mar 19, 2019")], WORK, {"pricing_online": True, "second_market": True, "price_cache_dir": str(abe[3] / "pc")})


def test_second_market_lines_reach_the_packet_in_native_eur_with_fx_and_stay_out_of_the_usd_totals(abe):
    env = _packet(abe[3], _es_row(abe), second_market=True)
    b, t = env["data"]["books"][0], env["data"]["totals"]
    assert b["replacement_cost"]["amount"] == 14.5 and b["replacement_cost"]["currency"] == "USD"                    # the US/USD line is untouched
    sm = b["second_market"]
    assert sm["market"] == "ES" and sm["currency"] == "EUR" and sm["replacement_cost"]["amount"] == 11.0 and sm["used_value"]["amount"] == 3.5
    assert sm["replacement_cost"]["currency"] == "EUR" and sm["replacement_cost"]["usd_equivalent"] == 12.4 and sm["fx"]["rate_date"] == "2026-10-06"
    assert t["books_replacement_cost"] == 14.5 and t["second_market"]["replacement_cost_eur"] == 11.0 and t["second_market"]["used_value_eur"] == 3.5     # never summed together
    assert t["second_market"]["replacement_cost_usd_equivalent"] == 12.4 and "NOT part of the US/USD totals" in t["second_market"]["note"]
    html = R.render_html(env, thumbs=False)
    assert "Second market, Spain (native EUR" in html and "11.0 EUR" in html and "USD figure = EUR x ECB 1.1269 of 2026-10-06; not a US price" in html


def test_a_second_market_line_with_the_wrong_isbn_currency_or_label_is_dropped(abe):
    row = _es_row(abe)
    bad = copy.deepcopy(row)
    bad["second_market"]["replacement_cost"]["equivalent_isbn"] = "9780143106654"
    bad["second_market"]["used_value"]["currency"] = "USD"
    sm = _packet(abe[3], bad, second_market=True)["data"]["books"][0]["second_market"]
    assert "replacement_cost" not in sm and "used_value" not in sm
    unlabelled = copy.deepcopy(row); unlabelled["second_market"]["replacement_cost"]["edition_match"] = "exact_isbn"
    assert "replacement_cost" not in _packet(abe[3], unlabelled, second_market=True)["data"]["books"][0]["second_market"]


def test_packet_verification_accepts_a_valid_second_market_line_and_rejects_a_tampered_one(abe):
    env = _packet(abe[3], _es_row(abe), second_market=True)
    assert not [x for x in P.verify_packet(env, manifest(), abe[3]) if "second_market" in x or "bk-001" in x and ("price" in x or "offer" in x or "raw" in x)]
    env["data"]["books"][0]["second_market"]["replacement_cost"]["amount"] = 1.0           # edited amount no longer matches the raw bytes
    assert any("second" in x for x in P.verify_packet(env, manifest(), abe[3]))


def test_second_market_is_dropped_from_the_packet_lines_totals_and_report_unless_the_flag_is_on(abe, monkeypatch):
    monkeypatch.delenv("LIBRARY_SECOND_MARKET", raising=False)
    row = _es_row(abe)
    assert row["second_market"]["replacement_cost"]["amount"] == 11.0                    # the stale row carries an EUR line (e.g. a cached prices.json from before the decision)
    env = _packet(abe[3], row)
    b, t = env["data"]["books"][0], env["data"]["totals"]
    assert "second_market" not in b and "second_market" not in t and b["replacement_cost"]["amount"] == 14.5 and t["books_replacement_cost"] == 14.5
    html = R.render_html(env, thumbs=False)
    assert "EUR" not in html and "Iberlibro" not in html and "Second market" not in html
    assert not [x for x in P.verify_packet(env, manifest(), abe[3]) if "second" in x]
    monkeypatch.setenv("LIBRARY_SECOND_MARKET", "1")
    assert "second_market" in _packet(abe[3], row)["data"]["books"][0]
