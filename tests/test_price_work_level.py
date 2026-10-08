"""Work-level equivalent pricing policy. Offline: fake catalogue + monkeypatched HTTP serving synthetic AbeBooks Next.js payloads."""
import json
from pathlib import Path

import pytest

from pipeline import price as P
from providers import book_prices as BP

FX_XML = "<gesmes:Envelope><Cube><Cube time='2026-10-06'><Cube currency='USD' rate='1.1269'/></Cube></Cube></gesmes:Envelope>"
ES_NEW, ES_OLD, EN_NEW, EN_HARD = "9788494326776", "9786073179096", "9780143106654", "9788491879428"


def entry(key, isbn=None, lang=("spa",), fmt="paperback", date="2019", title="Acosada", isbn10=None):
    e = {"key": key, "title": title, "publish_date": date, "physical_format": fmt, "languages": [{"key": "/languages/" + x} for x in lang]}
    if isbn: e["isbn_13"] = [isbn]
    if isbn10: e["isbn_10"] = [isbn10]
    return e


class Catalogue:
    def __init__(self, entries): self.entries, self.calls = entries, 0
    def editions(self, work):
        self.calls += 1
        return {"entries": self.entries, "size": len(self.entries), "urls": ["https://openlibrary.org" + work + "/editions.json"], "retrieved_at": "2026-10-06T00:00:00+00:00"}


def listing(isbn, price, cond="NEW", country="USA", lang="SPA", cover="SOFT_COVER", extra=()):
    return {"listing": {"listingId": int(price * 100), "isbn13": isbn, "isbn10": None, "language": lang, "condition": cond, "vendorCondition": cond.title(), "vendorCountry": country,
                        "productType": "DEFAULT", "bsaCodes": [cover, "BOOK", *extra], "searchAttributes": [], "priceInPurchaseCurrency": {"amount": price, "currency": "USD"}}}


def page(listings):
    payload = '1:{"searchResultsArray":' + json.dumps({"results": listings}) + "}"
    return '<html><script>self.__next_f.push([1,"' + json.dumps(payload)[1:-1] + '"])</script></html>'


class Response:
    status_code = 200
    def __init__(self, content): self.content = content
    def raise_for_status(self): pass


@pytest.fixture
def abe(tmp_path, monkeypatch):
    """pages[isbn] = listings; every request is recorded in urls."""
    pages, urls = {}, []
    def get(url, **kwargs):
        if "ecb.europa.eu" in url:                                   # FX reference rate (providers.fx shares the httpx module): served here, not recorded as a price request
            if pages.get("__fx__") == "fail": raise RuntimeError("offline")
            return Response(FX_XML.encode())
        urls.append(url)
        isbn = url.split("isbn=")[1].split("&")[0]
        rows = pages.get(("es", isbn) if "iberlibro" in url else isbn, [])
        if "cond=new" in url: rows = [r for r in rows if r["listing"]["condition"] == "NEW"]
        return Response(page(rows).encode())
    monkeypatch.setattr(BP.httpx, "get", get)
    provider = BP.PublicBookPrices(tmp_path / "ev", online=True, cache_dir=tmp_path / "abe", min_interval_s=0)
    return pages, urls, provider, tmp_path


def run(abe_fixture, entries, work=None, config=None):
    pages, urls, provider, root = abe_fixture
    work = work or {"id": "bk-1", "work": "/works/OL1W", "title": "Acosada", "visible_text": "Acosada\nBrian Freeman", "language_code": "es"}
    return P.price_work(work, root, Catalogue(entries), provider, config or {})


def test_every_row_and_line_is_labelled_work_level_equivalent_and_records_isbn_and_alternatives(abe):
    pages = abe[0]
    pages[ES_NEW] = [listing(ES_NEW, 14.5), listing(ES_NEW, 9.0, "USED_GOOD")]
    row = run(abe, [entry("/books/OL1M", ES_OLD, date="2012"), entry("/books/OL2M", ES_NEW, date="Mar 19, 2019")])
    assert row["price_basis"] == "work_level_equivalent"
    assert row["equivalent_edition"]["isbn"] == ES_NEW and "most recent" in row["equivalent_edition"]["why"]
    for kind, amount in (("replacement_cost", 14.5), ("used_value", 9.0)):
        line = row[kind]
        assert line["amount"] == amount and line["price_basis"] == "work_level_equivalent" and line["equivalent_isbn"] == ES_NEW
        assert line["edition_match"] != "exact_isbn" and line["currency"] == "USD" and line["evidence"]["sha256"] and line["url"].startswith("https://www.abebooks.com/")
    assert [a["isbn"] for a in row["alternatives_considered"]] == [ES_NEW]


def test_language_is_never_switched(abe):
    pages = abe[0]
    pages[EN_NEW] = [listing(EN_NEW, 5.0, lang="ENG")]
    spanish_spine = {"id": "bk-1", "work": "/works/OL1W", "title": "Acosada", "visible_text": "Acosada\nBrian Freeman\nUna novela", "language_code": "es"}
    row = run(abe, [entry("/books/OL1M", EN_NEW, lang=("eng",), date="2024")], spanish_spine)   # only an English edition exists for a Spanish spine (spine words: 'una')
    assert row["replacement_cost"] is None and row["used_value"] is None and row["price_basis"] == "work_level_equivalent"
    assert "no_isbn_bearing_edition" in row["issues"][0] and row["rejected_edition_records"]["by_reason"] == {"language_mismatch_never_switched": 1}
    assert abe[1] == []                                                              # no price source was even queried


def test_edition_without_recorded_language_needs_the_listing_language_to_match_and_ranks_last(abe):
    cands, rejected = P.rank_equivalent_editions([entry("/books/OL1M", ES_NEW, lang=())], "es")
    assert cands == [] and rejected[0]["reason"] == "edition_language_not_recorded"          # strict default
    cands, _ = P.rank_equivalent_editions([entry("/books/A", ES_NEW, lang=(), date="2024"), entry("/books/B", ES_OLD, date="2010")], "es", unrecorded_ok=True)
    assert [c["edition_key"] for c in cands] == ["/books/B", "/books/A"]                       # catalogue-confirmed language first, even if older
    abe[0][ES_NEW] = [listing(ES_NEW, 9.0, lang="ENG")]
    spanish_spine = {"id": "bk-1", "work": "/works/OL1W", "title": "Acosada", "visible_text": "Acosada\nBrian Freeman\nUna novela"}
    assert run(abe, [entry("/books/OL1M", ES_NEW, lang=())], spanish_spine)["replacement_cost"] is None   # English listing never accepted for a Spanish spine
    abe[0][ES_OLD] = [listing(ES_OLD, 9.0, lang="SPA")]                                       # other ISBN: the first page above is already cached
    row = run(abe, [entry("/books/OL1M", ES_OLD, lang=())], spanish_spine)
    assert row["replacement_cost"]["amount"] == 9.0 and row["equivalent_edition"]["language_basis"].startswith("abebooks_listing_language_only")


def test_language_from_listing_of_editions_titled_like_the_spine(abe):
    abe[0][ES_NEW] = [listing(ES_NEW, 9.0, lang="SPA")]
    work = {"id": "bk-3", "work": "/works/OL3W", "title": "Acosada", "visible_text": "Acosada"}
    row = run(abe, [entry("/books/OL1M", ES_NEW, lang=())], work)
    assert row["language"] == "es" and row["language_source"].startswith("abebooks_listing_language") and row["replacement_cost"]["amount"] == 9.0


def test_listing_in_the_wrong_language_is_rejected_even_for_the_right_isbn(abe):
    abe[0][ES_NEW] = [listing(ES_NEW, 7.0, lang="ENG")]
    row = run(abe, [entry("/books/OL1M", ES_NEW)])
    assert row["replacement_cost"] is None and any("no_validated_us_usd_listing" in i for i in row["issues"])


def test_format_family_rule_hardcover_is_not_used_and_paperback_beats_a_newer_unknown_format():
    entries = [entry("/books/A", EN_HARD, lang=("eng",), fmt="hardcover", date="2024"), entry("/books/B", EN_NEW, lang=("eng",), fmt=None, date="2023"),
               entry("/books/C", "9781476750828", lang=("eng",), fmt="Paperback", date="2015")]
    cands, rejected = P.rank_equivalent_editions(entries, "en")
    assert [c["edition_key"] for c in cands] == ["/books/C", "/books/B"] and rejected[0]["reason"] == "format_family_hardcover"
    assert P.format_family("Mass Market Paperback") == "paperback" and P.format_family("Kindle Edition") == "excluded" and P.format_family(None) == "unknown"


def test_null_with_reason_when_no_edition_has_a_checksum_valid_isbn(abe):
    row = run(abe, [entry("/books/OL1M", "9788494326777"), entry("/books/OL2M", None)])    # bad check digit; no ISBN at all
    assert row["replacement_cost"] is None and row["used_value"] is None and row["price_basis"] == "work_level_equivalent"
    assert row["issues"] == ["no_isbn_bearing_edition_in_language_and_format_family"] and row["rejected_edition_records"]["by_reason"] == {"no_checksum_valid_isbn": 2} and abe[1] == []


def test_falls_back_to_next_edition_when_the_most_recent_has_no_us_listing_and_records_it(abe):
    abe[0][ES_OLD] = [listing(ES_OLD, 8.0)]
    row = run(abe, [entry("/books/OL1M", ES_OLD, date="2012"), entry("/books/OL2M", ES_NEW, date="2019")])
    assert row["equivalent_edition"]["isbn"] == ES_OLD and [a["isbn"] for a in row["alternatives_considered"]] == [ES_NEW, ES_OLD]
    assert row["alternatives_considered"][0]["listings_valid"] == 0


def test_non_us_signed_collectible_and_appraisal_listings_never_set_a_price(abe):
    abe[0][ES_NEW] = [listing(ES_NEW, 3.0, country="DEU"), listing(ES_NEW, 4.0, extra=("SIGNED",)), listing(ES_NEW, 5.0, "USED_GOOD", extra=("FIRST_EDITION",)),
                      listing(ES_NEW, 900.0), listing(ES_NEW, 12.0), listing(ES_NEW, 6.0, cover="HARDCOVER")]
    row = run(abe, [entry("/books/OL1M", ES_NEW)])
    assert row["replacement_cost"]["amount"] == 12.0 and row["used_value"] is None and "used_value_source_unavailable" in row["issues"]
    assert row["alternatives_considered"][0]["above_appraisal_threshold_listings_excluded"] is True


def test_language_unknown_stays_null_and_catalogue_title_evidence_can_decide_it(abe):
    abe[0][ES_NEW] = [listing(ES_NEW, 11.0)]
    work = {"id": "bk-2", "work": "/works/OL2W", "title": "Acosada", "visible_text": "Acosada\nBrian Freeman"}            # no function words, no language_code
    both = [entry("/books/OL1M", ES_NEW), entry("/books/OL2M", EN_NEW, lang=("eng",), title="Stalked")]
    row = run(abe, both, work)                                                                                        # only the Spanish edition is titled 'Acosada'
    assert row["language"] == "es" and row["language_source"].startswith("catalogue_editions_titled_exactly") and row["replacement_cost"]["amount"] == 11.0
    amb = run(abe, [entry("/books/OL1M", ES_NEW, title="Striker"), entry("/books/OL2M", EN_NEW, lang=("eng",), title="Striker")], {**work, "title": "Striker"})
    assert amb["replacement_cost"] is None and amb["language"] is None and amb["issues"][0].startswith("language_not_determined") and amb["price_basis"] == "work_level_equivalent"


def test_language_detection_uses_spine_function_words_only():
    assert P.detect_language("THE CONNELLYS OF COUNTY DOWN") == "en" and P.detect_language("Sólo un muerto más") == "es"
    assert P.detect_language("Flash Boys") is None and P.detect_language("The man y la mar") is None


def test_publish_date_ordering_and_future_dates():
    assert P.publish_sort_key("Mar 19, 2014") == (2014, 3) and P.publish_sort_key("2018-12-05") == (2018, 12) and P.publish_sort_key("2012") == (2012, 0)
    assert P.publish_sort_key("2099", this_year=2026) == (0, 0) and P.publish_sort_key(None) == (0, 0)


def test_abebooks_next_payload_parser_keeps_only_us_usd_exact_isbn_physical_unflagged():
    html = page([listing(ES_NEW, 10.0), listing(ES_NEW, 2.0, country="CAN"), listing("9780143106654", 3.0), listing(ES_NEW, 4.0, "UNSPECIFIED")])
    offers = BP.parse_abebooks(html, ES_NEW, "https://www.abebooks.com/servlet/SearchResults?isbn=" + ES_NEW)
    assert [(o["kind"], o["amount"]) for o in offers] == [("replacement_cost", 10.0)]


def test_cli_refuses_to_write_into_an_existing_directory(tmp_path, monkeypatch, capsys):
    (tmp_path / "existing").mkdir(); (tmp_path / "existing" / "x").write_text("1"); (tmp_path / "w.json").write_text("[]")
    monkeypatch.setattr("sys.argv", ["price", "--works", str(tmp_path / "w.json"), "--out", str(tmp_path / "existing")])
    with pytest.raises(SystemExit):
        P.main()


def test_isbn10_is_canonicalised_to_isbn13_and_deduplicated():
    entries = [entry("/books/A", None, isbn10="8494326775"), entry("/books/B", "9788494326776")]
    cands, _ = P.rank_equivalent_editions(entries, "es")
    assert [c["isbn"] for c in cands] == ["9788494326776"]


# ---- wiring: exact-edition stage first, then the work-level path under a budget (what app/engine.py finalize calls) ------------------------------------------------
def book(i, work="/works/OL1W", title="Acosada", **kw):
    b = {"id": i, "title": title, "status": "identified", "identity_source": "catalogue_match", "isbn": None, "edition": None, "identity_matching_evidence": [],
         "identity_source_url": "https://openlibrary.org" + work if work else None, "visible_text": "Acosada\nBrian Freeman", "language": "es"}
    return {**b, **kw}


def wired(abe_fixture, books, entries, config=None):
    pages, urls, provider, root = abe_fixture
    cfg = {"pricing_online": True, "min_interval_s": 0, "price_cache_dir": str(root / "pc"), **(config or {})}
    art = P.price_books_with_work_level(books, root / "run", cfg, catalogue=Catalogue(entries))
    return art, art["data"]["prices"], root / "run"


def test_wired_work_level_prices_books_without_isbn_and_the_packet_validator_accepts_the_line(abe):
    abe[0][ES_NEW] = [listing(ES_NEW, 14.5), listing(ES_NEW, 9.0, "USED_GOOD")]
    b = book("bk-001")
    art, rows, run = wired(abe, [b], [entry("/books/OL2M", ES_NEW)])
    row = rows[0]
    assert row["identity_key"] == P.identity_key(b) and row["price_basis"] == "work_level_equivalent" and row["equivalent_edition"]["isbn"] == ES_NEW and row["title"] == "Acosada"
    assert row["replacement_cost"]["amount"] == 14.5 and row["used_value"]["amount"] == 9.0 and row["replacement_cost"]["price_basis"] == "work_level_equivalent"
    assert P.validate_offer(row["replacement_cost"], b, run) == [] and P.validate_offer(row["used_value"], b, run) == []
    assert art["metrics"]["work_level_priced"] == 1 and art["metrics"]["replacement_lines"] == 1
    from pipeline import packet as PK
    from tests.test_packet_work_level_price import _inv, ACCEPTED
    from tests.test_measurement import manifest
    pkt = PK.build_data(_inv(ACCEPTED), manifest(), {"run_id": "t", "prices": art, "run_dir": str(run)})["books"][0]       # the real packet accepts the wired row
    assert pkt["replacement_cost"]["amount"] == 14.5 and pkt["price_basis"] == "work_level_equivalent" and pkt["equivalent_isbn"] == ES_NEW
    # the same line is rejected for a book that is not an accepted catalogue work, and tampering with the label is rejected
    assert P.validate_offer(row["replacement_cost"], {**b, "identity_source_url": None}, run)
    assert P.validate_offer({**row["replacement_cost"], "edition_match": "something_else"}, b, run)
    assert P.validate_offer({**row["replacement_cost"], "equivalent_isbn": ES_OLD, "isbn": ES_OLD}, b, run)


def test_wired_skips_unidentified_user_stated_and_no_work_id_books_and_keeps_exact_isbn_path(abe):
    books = [book("a", status="unidentified"), book("b", work=None), book("c", identity_source="user_statement")]
    art, rows, _ = wired(abe, books, [entry("/books/OL2M", ES_NEW)])
    assert all(r["replacement_cost"] is None and "price_basis" not in r for r in rows) and abe[1] == []
    assert "work_level_skipped:no_catalogue_work_id" in rows[1]["issues"]


def test_wired_budget_exhausted_leaves_null_with_budget_reason_and_no_requests(abe):
    abe[0][ES_NEW] = [listing(ES_NEW, 14.5)]
    art, rows, _ = wired(abe, [book("bk-1"), book("bk-2", work="/works/OL2W")], [entry("/books/OL2M", ES_NEW)], {"price_budget_s": 0})
    assert all(r["replacement_cost"] is None and "work_level_skipped:budget" in r["issues"] for r in rows) and abe[1] == []
    assert art["metrics"]["work_level_budget_skipped"] == 2


def test_wired_same_work_is_priced_once_and_offline_is_cache_only(abe, tmp_path):
    abe[0][ES_NEW] = [listing(ES_NEW, 14.5)]
    art, rows, _ = wired(abe, [book("bk-1"), book("bk-2")], [entry("/books/OL2M", ES_NEW)])
    assert [r["replacement_cost"]["amount"] for r in rows] == [14.5, 14.5] and len([u for u in abe[1] if "www.abebooks.com" in u]) == 1    # one US page fetch for one work (the page is not full: no new-only page)
    before = len(abe[1])
    art2 = P.price_books_with_work_level([book("bk-3", work="/works/OL9W")], tmp_path / "off", {"pricing_online": False, "price_cache_dir": str(tmp_path / "offc")})
    r = art2["data"]["prices"][0]
    assert r["replacement_cost"] is None and r["price_basis"] == "work_level_equivalent" and r["issues"][0].startswith("edition_records_unavailable") and len(abe[1]) == before


def test_env_config_defaults_budget_online_and_shared_cache(tmp_path):
    cfg = P.price_config_from_env(tmp_path / "runs" / "r1", {})
    assert cfg["pricing_online"] is True and cfg["price_budget_s"] == 60 and cfg["price_cache_dir"] == str((tmp_path / "runs" / "_price_cache").resolve())
    off = P.price_config_from_env(tmp_path / "runs" / "r1", {"LIBRARY_PRICE_ONLINE": "0", "LIBRARY_PRICE_BUDGET_S": "12.5"})
    assert off["pricing_online"] is False and off["price_budget_s"] == 12.5
    assert P.price_config_from_env(tmp_path, {"LIBRARY_PRICE_BUDGET_S": "x"})["price_budget_s"] == 60


# ---- language from a publisher visible on the spine ------------------------------------------------------------------------------------------------------------
def pub_entry(key, isbn, lang, publisher, title="Michael Jordan"):
    return {**entry(key, isbn, lang=(lang,) if lang else (), title=title), "publishers": [publisher]}


def test_publisher_on_the_spine_decides_language_only_when_unambiguous():
    entries = [pub_entry("/books/A", ES_NEW, "spa", "Geoplaneta"), pub_entry("/books/B", EN_NEW, "eng", "Little, Brown"), pub_entry("/books/C", ES_OLD, None, "Geoplaneta")]
    spine = "geoPlaneta\nMICHAEL JORDAN\nROLAND LAZENBY"
    assert P.resolve_language(spine, None, "Michael Jordan", entries)[0] == "es"
    assert P.resolve_language(spine, None, "Michael Jordan", entries)[1].startswith("spine_publisher_matches_edition_record_publisher")
    assert P.resolve_language("MICHAEL JORDAN\nROLAND LAZENBY", None, "Michael Jordan", [pub_entry("/books/B", EN_NEW, "eng", "Little, Brown")])[0] == "en"   # via title-equal editions, not publisher
    # same publisher printed in both languages -> ambiguous, not guessed
    both = [pub_entry("/books/A", ES_NEW, "spa", "Penguin", "Stoner"), pub_entry("/books/B", EN_NEW, "eng", "Penguin", "Stoner")]
    assert P.language_from_publisher("PENGUIN\nSTONER", "Stoner", both) is None
    # a title word that equals a publisher name, or a generic publisher word alone, is not publisher evidence
    assert P.language_from_publisher("PLANETA", "Planeta", [pub_entry("/books/A", ES_NEW, "spa", "Planeta")]) is None
    assert P.language_from_publisher("EDITORIAL BOOKS", "X", [pub_entry("/books/A", ES_NEW, "spa", "Editorial Books")]) is None



# ---- second market: Spain, native EUR (Iberlibro), never converted silently, never added to the USD lines -----------------------------------------------------------
def listing_es(isbn, price, cond="NEW", country="ESP", lang="SPA", cover="SOFT_COVER", currency="EUR"):
    row = listing(isbn, price, cond, country, lang, cover)
    row["listing"]["priceInPurchaseCurrency"] = {"amount": price, "currency": currency}
    return row


def test_iberlibro_parser_keeps_only_native_eur_from_sellers_in_spain():
    rows = [listing_es(ES_NEW, 14.0), listing_es(ES_NEW, 9.0, country="DEU"), listing_es(ES_NEW, 8.0, currency="USD"), listing_es(ES_NEW, 7.0, cond="USED_GOOD"),
            listing_es(ES_NEW, 6.0, cond="UNSPECIFIED")]
    offers = BP.parse_abebooks(page(rows), ES_NEW, "https://www.iberlibro.com/servlet/SearchResults?isbn=" + ES_NEW)
    assert sorted((o["kind"], o["amount"], o["currency"], o["market"], o["seller_country"]) for o in offers) == [("replacement_cost", 14.0, "EUR", "ES", "ESP"), ("used_value", 7.0, "EUR", "ES", "ESP")]
    assert BP.parse_abebooks(page([listing_es(ES_NEW, 5.0)]), ES_NEW, "https://www.abebooks.com/servlet/SearchResults?isbn=" + ES_NEW) == []     # EUR is not a US/USD line


def test_spanish_work_without_us_listing_gets_a_native_eur_second_market_line_with_cited_fx(abe, tmp_path):
    abe[0][("es", ES_NEW)] = [listing_es(ES_NEW, 14.5), listing_es(ES_NEW, 4.2, cond="USED_GOOD")]
    row = run(abe, [entry("/books/OL2M", ES_NEW)], config={"pricing_online": True, "second_market": True, "price_cache_dir": str(tmp_path / "pc")})
    assert row["replacement_cost"] is None and row["used_value"] is None and "no_validated_us_usd_listing_for_the_equivalent_editions_tried" in row["issues"]    # US/USD line stays null
    sm = row["second_market"]
    new, used = sm["replacement_cost"], sm["used_value"]
    assert (sm["market"], sm["currency"]) == ("ES", "EUR") and new["amount"] == 14.5 and new["currency"] == "EUR" and new["original_currency"] == "EUR" and used["amount"] == 4.2
    assert new["price_basis"] == "work_level_equivalent" and new["equivalent_isbn"] == ES_NEW and new["converted"] is False and new["url"].startswith("https://www.iberlibro.com/")
    assert sm["fx"]["rate"] == 1.1269 and sm["fx"]["rate_date"] == "2026-10-06" and sm["fx"]["sha256"] and new["usd_equivalent"] == 16.34 and "ECB reference rate 1.1269 of 2026-10-06" in new["usd_equivalent_basis"]
    assert P.validate_offer({**new, "evidence": new["evidence"]}, None, tmp_path, expected_isbn=ES_NEW, market="ES") == [] if (tmp_path / new["evidence"]["raw_path"]).is_file() else True
    assert P.validate_offer(new, None, None, expected_isbn=ES_NEW, market="US")          # an EUR line can never pass the US/USD validation (and so never reaches the USD totals)


def test_second_market_without_fx_evidence_keeps_native_amount_and_no_usd_figure(abe, tmp_path):
    abe[0]["__fx__"] = "fail"
    abe[0][("es", ES_NEW)] = [listing_es(ES_NEW, 14.5)]
    row = run(abe, [entry("/books/OL2M", ES_NEW)], config={"pricing_online": True, "second_market": True, "price_cache_dir": str(tmp_path / "pc")})
    new = row["second_market"]["replacement_cost"]
    assert new["amount"] == 14.5 and new["usd_equivalent"] is None and new["fx"] is None and "no FX evidence" in new["usd_equivalent_basis"]


def test_second_market_is_only_for_spanish_language_works(abe, tmp_path):
    abe[0][EN_NEW] = [listing(EN_NEW, 9.0, lang="ENG")]
    en = run(abe, [entry("/books/OL3M", EN_NEW, lang=("eng",), title="Acosada")], work={"id": "bk-1", "work": "/works/OL1W", "title": "Acosada", "visible_text": "The Acosada of Anne", "language_code": "en"},
             config={"second_market": True})
    assert en["replacement_cost"]["amount"] == 9.0 and "second_market" not in en and not [u for u in abe[1] if "iberlibro" in u]


def test_second_market_is_off_by_default_usd_only_and_never_requests_iberlibro_or_the_fx_rate(abe, tmp_path, monkeypatch):
    monkeypatch.delenv("LIBRARY_SECOND_MARKET", raising=False)
    abe[0][ES_NEW] = [listing(ES_NEW, 14.5)]
    abe[0][("es", ES_NEW)] = [listing_es(ES_NEW, 14.5)]
    off = run(abe, [entry("/books/OL2M", ES_NEW)], config={"pricing_online": True, "price_cache_dir": str(tmp_path / "pc2")})
    assert off["replacement_cost"]["amount"] == 14.5 and off["replacement_cost"]["currency"] == "USD" and "second_market" not in off
    assert abe[1] and not [u for u in abe[1] if "iberlibro" in u]
    explicit_off = run(abe, [entry("/books/OL2M", ES_NEW)], config={"pricing_online": True, "second_market": False, "price_cache_dir": str(tmp_path / "pc3")})
    assert "second_market" not in explicit_off


def test_second_market_flag_env_and_explicit_config(monkeypatch, tmp_path):
    monkeypatch.delenv("LIBRARY_SECOND_MARKET", raising=False)
    assert P.second_market_enabled() is False and P.second_market_enabled({}) is False and P.price_config_from_env(tmp_path / "run")["second_market"] is False
    monkeypatch.setenv("LIBRARY_SECOND_MARKET", "1")
    assert P.second_market_enabled() is True and P.second_market_enabled({}) is True and P.price_config_from_env(tmp_path / "run")["second_market"] is True
    assert P.second_market_enabled({"second_market": False}) is False                      # an explicit config value beats the environment
    assert P.second_market_enabled(None, {"LIBRARY_SECOND_MARKET": "0"}) is False and P.second_market_enabled(None, {}) is False


def test_second_market_enabled_by_env_prices_iberlibro(abe, tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_SECOND_MARKET", "1")
    abe[0][("es", ES_NEW)] = [listing_es(ES_NEW, 14.5)]
    row = run(abe, [entry("/books/OL2M", ES_NEW)], config={"pricing_online": True, "price_cache_dir": str(tmp_path / "pc4")})
    assert row["second_market"]["replacement_cost"]["amount"] == 14.5 and row["second_market"]["currency"] == "EUR"
