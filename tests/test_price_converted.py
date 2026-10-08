"""Converted non-US fallback lines (US wins; labelled; FX re-checked from raw bytes; totals count them), the reading-language fallback (never overrides the
catalogue), throttled (HTTP 202) answers are 'source unavailable' and never 'no listing', and price attach onto a finished packet (backup, idempotent, verified,
instance layer on top). Offline: synthetic AbeBooks / Iberlibro / AbeBooks UK pages and a synthetic ECB file."""
import copy
import json

import pytest

from pipeline import contracts as C, packet as P, price as PR, price_attach as PA, report as R
from providers import book_prices as BP, fx as FX
from tests.test_measurement import manifest
from tests.test_packet_work_level_price import _inv, ACCEPTED
from tests.test_price_work_level import Catalogue, ES_NEW, ES_OLD, EN_NEW, Response, entry, listing, page

FX_XML = "<gesmes:Envelope><Cube><Cube time='2026-10-06'><Cube currency='USD' rate='1.1269'/><Cube currency='GBP' rate='0.8650'/></Cube></Cube></gesmes:Envelope>"
WORK = {"id": "bk-1", "work": "/works/OL1W", "title": "Acosada", "visible_text": "Acosada\nBrian Freeman", "language_code": "es"}


def native(isbn, price, cur, country, cond="NEW", lang="SPA"):
    row = listing(isbn, price, cond, country, lang)
    row["listing"]["priceInPurchaseCurrency"] = {"amount": price, "currency": cur}
    return row


@pytest.fixture
def web(tmp_path, monkeypatch):
    """pages[(host, isbn)] = listings; host in us/es/gb. status[(host, isbn)] = HTTP status to answer (default 200). Every request is recorded."""
    pages, urls, status = {}, [], {}
    def get(url, **kwargs):
        if "ecb.europa.eu" in url:
            return Response(FX_XML.encode())
        urls.append(url)
        host = "es" if "iberlibro" in url else "gb" if "abebooks.co.uk" in url else "us"
        isbn = url.split("isbn=")[1].split("&")[0]
        code = status.get((host, isbn), 200)
        if isinstance(code, list):
            code = code.pop(0) if code else 200
        if code != 200:
            r = Response(b""); r.status_code = code
            return r
        rows = pages.get((host, isbn), [])
        if "cond=new" in url: rows = [r for r in rows if r["listing"]["condition"] == "NEW"]
        return Response(page(rows).encode())
    monkeypatch.setattr(BP.httpx, "get", get)
    monkeypatch.setattr(FX.httpx, "get", get)
    provider = BP.PublicBookPrices(tmp_path / "ev", online=True, cache_dir=tmp_path / "abe", min_interval_s=0)
    return pages, urls, provider, tmp_path, status


def run(web, entries, work=WORK, config=None):
    pages, urls, provider, root, _ = web
    cfg = {"pricing_online": True, "price_cache_dir": str(root / "pc"), **(config or {})}
    return PR.price_work(work, root, Catalogue(entries), provider, cfg)


# ---------------------------------------------------------------------------------------------------------------------- precedence + labelling
def test_a_us_offer_always_wins_and_no_foreign_market_is_even_asked(web):
    web[0][("us", ES_NEW)] = [listing(ES_NEW, 14.5), listing(ES_NEW, 9.0, "USED_GOOD"), native(ES_NEW, 5.0, "USD", "ESP")]   # a Spanish seller is listed on the US site
    web[0][("es", ES_NEW)] = [native(ES_NEW, 3.0, "EUR", "ESP"), native(ES_NEW, 1.0, "EUR", "ESP", "USED_GOOD")]
    row = run(web, [entry("/books/OL2M", ES_NEW)])
    assert row["replacement_cost"]["amount"] == 14.5 and row["used_value"]["amount"] == 9.0
    assert row["replacement_cost"]["converted"] is False and row["used_value"]["converted"] is False and "converted_fallback" not in row
    assert not [u for u in web[1] if "iberlibro" in u or "abebooks.co.uk" in u]


def test_no_us_seller_gives_a_converted_line_with_native_amount_rate_date_and_source(web):
    web[0][("us", ES_NEW)] = [native(ES_NEW, 13.0, "USD", "ESP")]                     # US page: only a Spanish seller (marketplace USD price, never used)
    web[0][("es", ES_NEW)] = [native(ES_NEW, 12.0, "EUR", "ESP"), native(ES_NEW, 4.0, "EUR", "ESP", "USED_GOOD")]
    row = run(web, [entry("/books/OL2M", ES_NEW)])
    new, used = row["replacement_cost"], row["used_value"]
    assert new["converted"] is True and new["currency"] == "USD" and new["original_currency"] == "EUR" and new["original_amount"] == 12.0 and new["market"] == "ES"
    assert new["amount"] == 13.52 and used["amount"] == 4.51                         # 12 x 1.1269 = 13.5228 ; 4 x 1.1269 = 4.5076 (half up, cents)
    fx = new["fx"]
    assert fx["rate_date"] == "2026-10-06" and fx["ecb_rates_per_eur"] == {"USD": "1.1269"} and fx["url"] == FX.ECB_URL and fx["sha256"] and fx["raw_path"].startswith("pricing_evidence/fx_")
    assert "EUR 12.00 x 1.1269 USD/EUR" in new["conversion"] and "= USD 13.52" in new["conversion"] and new["url"].startswith("https://www.iberlibro.com/")
    assert new["price_basis"] == "work_level_equivalent" and new["equivalent_isbn"] == ES_NEW and row["equivalent_edition"]["isbn"] == ES_NEW
    assert "no_validated_us_usd_listing_for_the_equivalent_editions_tried" in row["issues"] and "replacement_cost_converted_from_EUR" in row["issues"]
    assert "NO validated US listing" in row["equivalent_edition"]["why"] and row["converted_fallback"]["chosen"]["markets"] == ["ES"]
    root = web[3]
    assert PR.validate_offer(new, None, root, expected_isbn=ES_NEW) == []           # raw listing bytes + raw ECB bytes re-checked
    assert any("does not equal" in e for e in PR.validate_offer({**new, "amount": 20.0}, None, root, expected_isbn=ES_NEW))
    bad_fx = copy.deepcopy(new); bad_fx["fx"]["ecb_rates_per_eur"] = {"USD": "1.5"}
    assert any("fx rates/date" in e for e in PR.validate_offer(bad_fx, None, root, expected_isbn=ES_NEW))
    unlabelled = {**new, "converted": False}
    assert any("not an unconverted US/USD offer" in e for e in PR.validate_offer(unlabelled, None, root, expected_isbn=ES_NEW))   # never passes as a native US line


def test_gbp_offer_is_converted_through_the_ecb_cross_rate(web):
    en = [entry("/books/OL3M", EN_NEW, lang=("eng",), title="Acosada")]
    work = {**WORK, "visible_text": "The Acosada of Anne", "language_code": "en"}
    web[0][("us", EN_NEW)] = [native(EN_NEW, 9.0, "USD", "GBR", lang="ENG")]
    web[0][("gb", EN_NEW)] = [native(EN_NEW, 10.0, "GBP", "GBR", lang="ENG")]
    row = run(web, en, work)
    new = row["replacement_cost"]
    assert new["original_currency"] == "GBP" and new["market"] == "GB" and new["amount"] == 13.03       # 10 x 1.1269 / 0.8650 = 13.0277
    assert new["fx"]["ecb_rates_per_eur"] == {"USD": "1.1269", "GBP": "0.8650"} and "cross rate through EUR" in new["fx"]["method"]
    assert [u for u in web[1] if "abebooks.co.uk" in u][0].startswith("https://www.abebooks.co.uk/") and not [u for u in web[1] if "iberlibro" in u]   # English: UK first
    assert PR.validate_offer(new, None, web[3], expected_isbn=EN_NEW) == []


def test_a_used_only_us_pick_keeps_its_native_line_and_fills_new_from_abroad_for_the_same_isbn(web):
    web[0][("us", ES_NEW)] = [listing(ES_NEW, 7.0, "USED_GOOD"), native(ES_NEW, 20.0, "USD", "ESP")]
    web[0][("es", ES_NEW)] = [native(ES_NEW, 15.0, "EUR", "ESP")]
    web[0][("es", ES_OLD)] = [native(ES_OLD, 2.0, "EUR", "ESP")]                      # another edition abroad is never mixed in
    row = run(web, [entry("/books/OL2M", ES_NEW, date="2019"), entry("/books/OL1M", ES_OLD, date="2010")])
    assert row["used_value"]["amount"] == 7.0 and row["used_value"]["converted"] is False                    # native US line kept
    assert row["replacement_cost"]["converted"] is True and row["replacement_cost"]["original_amount"] == 15.0 and row["replacement_cost"]["isbn"] == ES_NEW
    assert row["equivalent_edition"]["isbn"] == ES_NEW and not [u for u in web[1] if ES_OLD in u and "iberlibro" in u]


def test_markets_without_a_seller_on_the_parsed_us_page_are_not_asked_and_the_reason_is_kept(web):
    web[0][("us", ES_NEW)] = [native(ES_NEW, 9.0, "USD", "DEU")]
    row = run(web, [entry("/books/OL2M", ES_NEW)])
    assert row["replacement_cost"] is None and not [u for u in web[1] if "iberlibro" in u or "abebooks.co.uk" in u]
    assert all("not_queried:no_" in t["skipped"] for t in row["converted_fallback"]["tried"])
    assert "no_validated_non_us_listing_for_the_equivalent_editions_tried" in row["issues"]


def test_the_converted_fallback_can_be_switched_off(web, monkeypatch):
    web[0][("us", ES_NEW)] = [native(ES_NEW, 13.0, "USD", "ESP")]
    web[0][("es", ES_NEW)] = [native(ES_NEW, 12.0, "EUR", "ESP")]
    assert run(web, [entry("/books/OL2M", ES_NEW)], config={"converted_fallback": False})["replacement_cost"] is None
    monkeypatch.setenv("LIBRARY_CONVERTED_FALLBACK", "0")
    assert PR.converted_fallback_enabled() is False and PR.price_config_from_env(web[3] / "r")["converted_fallback"] is False
    monkeypatch.delenv("LIBRARY_CONVERTED_FALLBACK")
    assert PR.converted_fallback_enabled() is True and PR.converted_markets() == ["ES", "GB"]


# ---------------------------------------------------------------------------------------------------------------------- totals + packet
def _packet(root, row):
    prices = C.envelope("t", "price", {"prices": [row], "market": "US", "currency": "USD", "price_basis": "work_level_equivalent"})
    return C.envelope("t", "packet", P.build_data(_inv(ACCEPTED), manifest(), {"run_id": "t", "prices": prices, "run_dir": str(root)}), status="partial")


def test_packet_totals_include_converted_lines_and_say_how_many_and_the_report_labels_them(web):
    web[0][("us", ES_NEW)] = [listing(ES_NEW, 7.0, "USED_GOOD"), native(ES_NEW, 20.0, "USD", "ESP")]
    web[0][("es", ES_NEW)] = [native(ES_NEW, 15.0, "EUR", "ESP")]
    row = run(web, [entry("/books/OL2M", ES_NEW)])
    row["title"] = "Acosada"; row["ref_id"] = "bk-001"
    env = _packet(web[3], row)
    b, t, cp = env["data"]["books"][0], env["data"]["totals"], env["data"]["completeness"]
    assert b["replacement_cost"]["converted"] is True and b["used_value"]["converted"] is False
    assert t["books_replacement_cost"] == 16.9 and t["books_used_value"] == 7.0                                  # 15 EUR x 1.1269 = 16.90 USD, in the USD total
    assert cp["converted_lines"]["books_replacement_cost"] == 1 and cp["converted_lines"]["books_used_value"] == 0
    assert cp["books"]["replacement_cost"]["converted"] == 1 and cp["books"]["used_value"]["native_usd"] == 1 and cp["books"]["replacement_cost"]["converted_by_currency"] == {"EUR": 1}
    assert "CONVERTED lines" in t["notes"]["converted_lines"]
    assert not [x for x in P.verify_packet(env, manifest(), web[3]) if any(w in x for w in ("price", "offer", "raw", "totals", "converted", "fx", "work-level"))]
    html = R.render_html(env, thumbs=False)
    assert "CONVERTED</b> from EUR 15.0" in html and "includes 1 line(s) converted from a non-US market" in html
    env["data"]["books"][0]["replacement_cost"]["amount"] = 15.0                       # the EUR figure passed off as USD
    assert any("bk-001" in x for x in P.verify_packet(env, manifest(), web[3]))


# ---------------------------------------------------------------------------------------------------------------------- language from the readings
def test_reading_language_decides_only_when_the_catalogue_does_not(web):
    both = [entry("/books/OL1M", ES_NEW, title="Striker"), entry("/books/OL2M", EN_NEW, lang=("eng",), title="Striker")]
    web[0][("us", ES_NEW)] = [listing(ES_NEW, 11.0)]
    readings = [{"observation_id": "a", "language_code": "es", "language_evidence": "del"}, {"observation_id": "b", "language_code": "es", "language_evidence": "los"},
                {"observation_id": "c", "language_code": "en", "language_evidence": "the"}, {"observation_id": "d", "language_code": None}]
    work = {"id": "bk-2", "work": "/works/OL2W", "title": "Striker", "visible_text": "Striker", "readings": readings}
    row = run(web, both, work)
    assert row["language"] == "es" and row["language_source"].startswith("spine_reading_language_majority:2_of_3") and row["replacement_cost"]["amount"] == 11.0
    assert row["language_reading_vote"]["evidence"][0] == {"observation_id": "a", "language_code": "es", "language_evidence": "del"}
    assert "both_languages" in row["language_undetermined_by_catalogue"]
    tie = run(web, both, {**work, "readings": readings[1:]})
    assert tie["language"] is None and tie["issues"][0].startswith("language_not_determined")
    off = run(web, both, work, {"reading_language_fallback": False})
    assert off["language"] is None


def test_a_catalogue_determined_language_is_never_overridden_by_the_readings(web):
    web[0][("us", ES_NEW)] = [listing(ES_NEW, 11.0)]
    english_votes = [{"observation_id": str(i), "language_code": "en", "language_evidence": "x"} for i in range(5)]
    row = run(web, [entry("/books/OL1M", ES_NEW)], {**WORK, "language_code": "en", "readings": english_votes})   # only a Spanish edition is titled 'Acosada'
    assert row["language"] == "es" and row["language_source"].startswith("catalogue_editions_titled_exactly") and "language_reading_vote" not in row
    assert PR.resolve_language("The man", "es", "X", [])[0] == "en"                                     # spine words beat the reading's code too
    assert PR.reading_language_vote([{"language_code": "mul"}]) is None


# ---------------------------------------------------------------------------------------------------------------------- throttled answers
def test_a_throttled_202_answer_is_source_unavailable_never_cached_and_never_no_listing(web):
    web[4][("us", ES_NEW)] = 202
    row = run(web, [entry("/books/OL2M", ES_NEW)], config={"converted_fallback": False})
    assert row["replacement_cost"] is None and row["issues"] == ["us_price_source_unavailable_http_for_every_edition_tried"]
    assert not list((web[3] / "abe").glob("*.html")) and "source_unavailable:http_202" in json.dumps(row["alternatives_considered"])
    provider = BP.PublicBookPrices(web[3] / "ev2", online=True, cache_dir=web[3] / "abe2", min_interval_s=0, throttle_retries=1, throttle_backoff_s=0)
    web[4][("us", ES_OLD)] = [202]                                                      # throttled once, then 200
    web[0][("us", ES_OLD)] = [listing(ES_OLD, 8.0)]
    found, failures = provider.offers(ES_OLD, [PR.ABE_URL.format(isbn=ES_OLD)])
    assert found[0]["amount"] == 8.0 and not failures and provider.throttle_events[0]["http_status"] == 202 and provider.http_attempts["www.abebooks.com"] == 2
    stuck = BP.PublicBookPrices(web[3] / "ev3", online=True, cache_dir=web[3] / "abe3", min_interval_s=0, throttle_retries=1, throttle_backoff_s=0)
    web[4][("us", EN_NEW)] = 202
    for _ in range(3):
        stuck.offers(EN_NEW, [PR.ABE_URL.format(isbn=EN_NEW)])
    before = stuck.http_attempts["www.abebooks.com"]
    _, fail = stuck.offers(EN_NEW, [PR.ABE_URL.format(isbn=EN_NEW)])
    assert "circuit_open" in fail[0]["reason"] and stuck.http_attempts["www.abebooks.com"] == before          # fails fast, no more requests


def test_a_page_with_fewer_than_a_full_page_of_listings_skips_the_new_only_page(web):
    web[0][("us", ES_NEW)] = [listing(ES_NEW, 14.5)]
    run(web, [entry("/books/OL2M", ES_NEW)])
    assert [u for u in web[1] if "abebooks.com" in u] == [PR.ABE_URL.format(isbn=ES_NEW)]
    full = [listing(EN_NEW, 30.0 + i, "USED_GOOD", lang="ENG") for i in range(PR.PAGE_SIZE)]
    web[0][("us", EN_NEW)] = full
    run(web, [entry("/books/OL3M", EN_NEW, lang=("eng",), title="Acosada")], {**WORK, "visible_text": "The Acosada", "language_code": "en"})
    assert PR.ABE_URL_NEW.format(isbn=EN_NEW) in web[1]


# ---------------------------------------------------------------------------------------------------------------------- attach onto a finished packet
def test_prices_attach_in_place_verified_idempotent_with_backup_and_the_instance_layer_on_top(web, tmp_path):
    from pipeline import instances as IN
    from tests.test_second_catalogue import IA, _files, _run_with_packet
    run_dir, man = _run_with_packet(tmp_path)
    ol = ("ACOSADA", "/works/OL1W", "https://openlibrary.org/works/OL1W", "open_library")
    rp, sp = _files(tmp_path, {"f00000-o000": ol, "f00001-o000": ol, "f00000-o001": ("STRIKER", "/works/OL2W", "https://openlibrary.org/works/OL2W", "open_library")})
    assert IA.attach(run_dir, rp, sp, render=False)["attached"]
    p = C.read_json(run_dir / C.ARTIFACTS["packet"])["data"]
    readings = {"f00000-o000": {"observation_id": "f00000-o000", "visible_text": "ACOSADA", "language_code": "es", "language_evidence": "acosada"},
                "f00000-o001": {"observation_id": "f00000-o001", "visible_text": "STRIKER", "language_code": None, "language_evidence": None}}   # undecided: stays null
    inputs = PA.price_inputs(p, readings)
    assert sorted(b["id"] for b in inputs) == ["bk-f00000-o000", "bk-f00000-o001", "bk-f00001-o000"]       # merged members are priced per track group
    assert next(b for b in inputs if b["id"] == "bk-f00000-o000")["reading_evidence"][0]["language_evidence"] == "acosada"
    web[0][("us", ES_NEW)] = [listing(ES_NEW, 7.0, "USED_GOOD"), native(ES_NEW, 20.0, "USD", "ESP")]
    web[0][("es", ES_NEW)] = [native(ES_NEW, 15.0, "EUR", "ESP")]
    cat = Catalogue([entry("/books/OL2M", ES_NEW, title="ACOSADA")])
    out = web[3]
    art = PR.price_books_with_work_level(inputs, out, {"pricing_online": True, "price_cache_dir": str(out / "pc"), "price_budget_s": 600}, catalogue=cat, wl_provider=web[2])
    C.write_json(out / "prices.json", art)
    res = PA.attach(run_dir, out / "prices.json", render=False)
    assert res["attached"], res
    env = C.read_json(run_dir / C.ARTIFACTS["packet"])
    p = env["data"]
    merged = p["books"][0]
    assert merged["member_group_ids"] == ["bk-f00000-o000", "bk-f00001-o000"] and merged["replacement_cost"]["converted"] is True and merged["used_value"]["amount"] == 7.0
    assert p["totals"]["books_replacement_cost"] == 16.9 and p["totals"]["books_used_value"] == 7.0 and p["completeness"]["converted_lines"]["books_replacement_cost"] == 1
    ev = merged["replacement_cost"]["evidence"]["raw_path"]
    assert ev == f"pricing_evidence/{merged['replacement_cost']['evidence']['sha256']}.html" and (run_dir / ev).is_file()
    assert (run_dir / merged["replacement_cost"]["fx"]["raw_path"]).is_file() and (run_dir / PA.BACKUP).is_file()
    assert P.verify_packet(env, man, run_dir) == []
    assert PA.attach(run_dir, out / "prices.json", render=False)["attached"] and C.read_json(run_dir / C.ARTIFACTS["packet"])["data"]["books"] == p["books"], "idempotent"
    assert IN.attach(run_dir, render=False)["attached"]                                                    # the instance layer re-applied on top keeps the prices
    again = C.read_json(run_dir / C.ARTIFACTS["packet"])["data"]
    assert again["totals"]["books_replacement_cost"] == 16.9 and again["books"][0]["replacement_cost"]["converted"] is True
    # zero loss: a later computation without a price for that group keeps the earlier validated line
    empty = copy.deepcopy(art)
    for r in empty["data"]["prices"]:
        r["replacement_cost"] = r["used_value"] = None
    C.write_json(out / "prices_empty.json", empty)
    assert PA.attach(run_dir, out / "prices_empty.json", render=False)["groups"].get("kept") == 2
    assert C.read_json(run_dir / C.ARTIFACTS["packet"])["data"]["totals"]["books_replacement_cost"] == 16.9


# ---------------------------------------------------------------------------------------------------------------------- Google Books identities
def test_a_google_books_identity_is_priced_through_its_saved_volume_record(web, tmp_path):
    res = {"observation_id": "o1", "decision": "accepted", "work": "gb:ZXRezQEACAAJ", "source_url": "https://books.google.com/books?id=ZXRezQEACAAJ",
           "catalogue_query": {"retrieved_at": "2026-10-07T09:31:06+00:00"},
           "google_books": {"volume_id": "ZXRezQEACAAJ", "title": "Mentalidad Mamba", "language": "es", "publishedDate": "2020-04-17", "publisher": "Planeta",
                            "industryIdentifiers": [{"type": "ISBN_13", "identifier": ES_NEW}]}}
    C.write_json(tmp_path / "res.json", C.envelope("x", "resolve", {"resolutions": [res]}))
    cat = PA.VolumeCatalogue(None, PA.volumes_from(tmp_path / "res.json"))
    eds = cat.editions("gb:ZXRezQEACAAJ")
    assert eds["entries"][0]["isbn_13"] == [ES_NEW] and eds["entries"][0]["languages"] == [{"key": "/languages/spa"}] and eds["urls"] == [res["source_url"]]
    book = {"id": "bk-9", "status": "identified", "identity_source": "catalogue_match", "identity_source_url": res["source_url"], "title": "Mentalidad Mamba"}
    assert PR.priced_work_key(book) == "gb:ZXRezQEACAAJ" and PR.work_key_of(book) is None                      # the Open Library work key stays OL-only (auto_scale)
    web[0][("us", ES_NEW)] = [listing(ES_NEW, 11.0)]
    row = PR.price_work({"id": "bk-9", "work": "gb:ZXRezQEACAAJ", "title": "Mentalidad Mamba", "visible_text": "MENTALIDAD MAMBA"}, web[3], cat, web[2],
                        {"pricing_online": True, "price_cache_dir": str(web[3] / "pc")})
    assert row["replacement_cost"]["amount"] == 11.0 and row["equivalent_edition"]["source_url"] == res["source_url"] and row["language"] == "es"
    assert PR.validate_offer(row["replacement_cost"], book, web[3]) == []
