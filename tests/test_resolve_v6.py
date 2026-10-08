"""Fragment resolution rule v6 in production: pipeline/resolve.py evaluate_candidates/resolve_text + providers/openlibrary.py search_fragments.

Regressions come from the reader benchmark (library samples A/B, YT1/YT2 dev sets; model-read labels, never human truth). Candidates are synthetic so the tests are offline.
"""
import json

import httpx
import pytest

from pipeline import resolve as R
from providers.openlibrary import OpenLibrary


def cand(titles, authors, work="/works/OL1W"):
    return {"work": work, "titles": titles if isinstance(titles, list) else [titles], "authors": authors, "source": "synthetic"}


def ev(text, cands, coverage=R.COVERAGE_FLOOR):
    return R.evaluate_candidates(text, cands, coverage)


def test_accepts_complete_title_with_independent_author_and_returns_catalogue_literal():
    decision, res = ev("THE CONNELLYS OF COUNTY DOWN\nJoan Dunn", [cand("The Connellys of County Down", ["Joan Dunn"])])
    assert decision == "accept" and res["title"] == "The Connellys of County Down" and res["independent_author_evidence"]


def test_title_tokens_are_deduplicated_libertad():
    # 'LIBERTAD LIBERTAD LIBERTAD' is ONE distinct token: without an author it cannot reach the >= 4 distinct-token bar
    decision, _ = ev("LIBERTAD LIBERTAD LIBERTAD", [cand("Libertad libertad libertad", ["Anon Writer"])])
    assert decision != "accept"


def test_author_evidence_required_for_short_titles_areilza():
    # biography of Areilza must not resolve a visible 'A lo largo del siglo' (title token 'siglo' + no independent author)
    decision, _ = ev("A lo largo del siglo", [cand("José María de Areilza: a lo largo del siglo", ["Fulano Pérez"])])
    assert decision != "accept"


def test_study_guide_pearl_must_not_win_without_author_evidence():
    decision, res = ev("THE PEARL", [cand("The Pearl by John Steinbeck", ["Study Guides Inc"]), cand("The Pearl", ["John Steinbeck"], "/works/OL2W")])
    assert decision != "accept", res


def test_one_word_title_needs_independent_author_evidence():
    assert ev("DUNE\nFrank Herbert", [cand("Dune", ["Frank Herbert"])])[0] == "accept"
    assert ev("LIBERTAD\nJuan Perez", [cand("Libertad", ["Juan Perez"])])[0] == "accept"
    assert ev("DUNE", [cand("Dune", ["Frank Herbert"])])[0] != "accept"                          # no author evidence
    assert ev("LIBERTAD", [cand("Libertad", ["Juan Perez"])])[0] != "accept"


def test_four_distinct_content_tokens_may_resolve_without_author():
    decision, res = ev("La casa de los espíritus verdes y azules", [cand("La casa de los espíritus verdes y azules", ["Z Z"])])
    assert decision == "accept" and not res["independent_author_evidence"]


def test_never_complete_unseen_title_tokens():
    assert ev("Alpha Beta\nAnne Miller", [cand("Alpha Beta Gamma", ["Anne Miller"])])[0] != "accept"


def test_coverage_floor_cannot_be_lowered_by_the_caller():
    assert ev("Alpha Beta\nAnne Miller", [cand("Alpha Beta Gamma", ["Anne Miller"])], coverage=0.3)[0] != "accept"


def test_spanish_text_never_replaced_by_english_title():
    text = "Harry Potter y el prisionero de Azkaban\nJ. K. Rowling"
    assert ev(text, [cand("Harry Potter and the Prisoner of Azkaban", ["J. K. Rowling"])])[0] != "accept"
    # the Spanish edition title of the same work is an acceptable (literal) answer
    decision, res = ev(text, [cand(["Harry Potter and the Prisoner of Azkaban", "Harry Potter y el prisionero de Azkaban"], ["J. K. Rowling"])])
    assert decision == "accept" and res["title"] == "Harry Potter y el prisionero de Azkaban"


def test_same_surname_different_given_name_prefers_visible_given_name_and_abstains_on_tie():
    two = [cand("Dune Messiah", ["John Herbert"]), cand("Dune Messiah", ["Frank Herbert"], "/works/OL2W")]
    decision, res = ev("Dune Messiah\nFrank Herbert", two)
    assert decision == "accept" and res["work"] == "/works/OL2W"
    assert ev("Dune Messiah\nHerbert", two)[0] == "abstain_ambiguous"


def test_distinct_works_with_equal_evidence_abstain():
    two = [cand("Sombras de Roma", ["Ana Ruiz"], "/works/OL1W"), cand("Sombras de Roma", ["Pedro Ruiz"], "/works/OL2W")]
    assert ev("Sombras de Roma\nRuiz", two)[0] == "abstain_ambiguous"


def test_same_work_listed_twice_is_not_ambiguous():
    two = [cand("Sombras de Roma", ["Ana Ruiz"], "/works/OL1W"), cand("Sombras de Roma", ["Ana Ruiz"], "/works/OL9W")]
    assert ev("Sombras de Roma\nAna Ruiz", two)[0] == "accept"


def test_fused_ocr_surname_counts_as_author_evidence():
    decision, _ = ev("CONNIGGULDEN\nLos hijos del imperio", [cand("Los hijos del imperio", ["Conn Iggulden"])])
    assert decision == "accept"


def test_text_too_short_abstains():
    assert ev("ab", [cand("Libertad", ["Juan Perez"])])[0] == "abstain_text_too_short"


# --- retrieval / catalogue behaviour ---------------------------------------------------------------------------------------------------------------------

def transport(counter):
    def handler(request):
        counter.append(str(request.url))
        q = request.url.params["q"]
        docs = [{"key": "/works/OL1W", "title": "Sombras de Roma", "author_name": ["Ana Ruiz"], "editions": {"docs": [{"key": "/books/OL1M", "title": "Sombras de Roma"}]}}]
        return httpx.Response(200, json={"numFound": len(docs), "docs": docs if "sombras" in q.lower() or "ruiz" in q.lower() else []})
    return httpx.MockTransport(handler)


def test_offline_cache_miss_abstains_with_reason(tmp_path):
    cat = OpenLibrary(tmp_path, online=False)
    out = R.resolve_text("Sombras de Roma\nAna Ruiz", cat)
    cat.close()
    assert out["decision"] == "abstain" and out["reason"].startswith("catalogue_unavailable") and "catalogue_offline_cache_miss" in out["reason"]
    assert cat.requests == 0


def test_online_resolves_then_second_run_is_served_from_cache_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_FRAGMENT_SINGLES", "always")
    sleeps, urls = [], []
    monkeypatch.setattr("providers.openlibrary.time.sleep", lambda s: sleeps.append(s))
    cat = OpenLibrary(tmp_path, online=True, transport=transport(urls))
    out = R.resolve_text("Sombras de Roma\nAna Ruiz", cat)
    cat.close()
    assert out["decision"] == "accepted" and out["title"] == "Sombras de Roma" and out["work"] == "/works/OL1W"
    assert len(urls) == cat.requests >= 2 and all("openlibrary.org/search.json" in u for u in urls)
    again = OpenLibrary(tmp_path, online=False)                  # no network needed: every query is cached
    out2 = R.resolve_text("Sombras de Roma\nAna Ruiz", again)
    again.close()
    assert out2["decision"] == "accepted" and again.requests == 0 and len(urls) == cat.requests


def test_polite_spacing_between_uncached_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_FRAGMENT_SINGLES", "always")
    sleeps = []
    monkeypatch.setattr("providers.openlibrary.time.sleep", lambda s: sleeps.append(s))
    cat = OpenLibrary(tmp_path, online=True, transport=transport([]))
    R.resolve_text("Sombras de Roma\nAna Ruiz", cat)
    cat.close()
    assert sleeps and all(0 < s <= 1.05 for s in sleeps)       # every uncached request after the first waits out the 1.05 s spacing


def test_catalogue_error_on_first_query_abstains_not_raises(tmp_path):
    cat = OpenLibrary(tmp_path, online=True, transport=httpx.MockTransport(lambda r: httpx.Response(429, json={})))
    out = R.resolve_text("Sombras de Roma\nAna Ruiz", cat)
    cat.close()
    assert out["decision"] == "abstain" and out["reason"].startswith("catalogue_unavailable")


def test_no_usable_terms_abstains(tmp_path):
    out = R.resolve_text("a de el", OpenLibrary(tmp_path, online=False))
    assert out["decision"] == "abstain" and out["reason"] in {"no_usable_query_terms"}


def test_resolve_one_fragment_fallback_is_opt_in_and_does_not_fill_unseen_author(tmp_path, monkeypatch):
    monkeypatch.setattr("providers.openlibrary.time.sleep", lambda s: None)
    reading = {"observation_id": "o1", "visible_text": "Sombras de Roma\nRuiz", "provenance": {}}
    cat = OpenLibrary(tmp_path, online=True, transport=transport([]))
    off = R.resolve_one(reading, cat)
    on = R.resolve_one(reading, cat, fragments=True)
    cat.close()
    assert off["decision"] == "no_match"
    assert on["decision"] == "accepted" and on["title"] == "Sombras de Roma" and on["author"] == "Ana Ruiz"
    assert any(i.startswith("resolved_by_fragment_rule_v6") for i in on["issues"]) and on["matching_evidence"][0]["coverage"] == 1.0


def test_resolve_one_offline_fragment_fallback_abstains_with_reason(tmp_path):
    reading = {"observation_id": "o1", "visible_text": "Sombras de Roma\nRuiz", "provenance": {}}
    cat = OpenLibrary(tmp_path, online=False)
    res = R.resolve_one(reading, cat, fragments=True)
    cat.close()
    assert res["decision"] == "no_match" and any("catalogue_offline_cache_miss" in i for i in res["issues"])


def test_shared_pacing_spaces_requests_across_instances(tmp_path, monkeypatch):
    """The live engine builds one client per resolve call and runs several in parallel: real clients share one process-wide clock."""
    import providers.openlibrary as OL
    now, sleeps = [100.0], []
    monkeypatch.setattr(OL.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(OL.time, "sleep", lambda s: (sleeps.append(round(s, 3)), now.__setitem__(0, now[0] + s)))
    monkeypatch.setattr(OL, "_PACE_LAST", [0.])
    monkeypatch.delenv("LIBRARY_OL_MIN_INTERVAL_S", raising=False)
    handler = httpx.MockTransport(lambda r: httpx.Response(200, json={"numFound": 0, "docs": []}))
    a, b = OpenLibrary(tmp_path / "a", online=True, transport=handler, shared_pacing=True), OpenLibrary(tmp_path / "b", online=True, transport=handler, shared_pacing=True)
    a.search("x"); b.search("y"); a.search("z")
    a.close(); b.close()
    assert sleeps == [1.05, 1.05]                    # first request starts at once (clock far past 0), then every start is >= 1.05 s after the previous one, whichever client
    assert a.requests == 2 and b.requests == 1


def test_pacing_interval_and_contact_come_from_the_environment(tmp_path, monkeypatch):
    import providers.openlibrary as OL
    monkeypatch.setenv("LIBRARY_OL_MIN_INTERVAL_S", "0.4")
    monkeypatch.setenv("LIBRARY_OL_CONTACT", "ops@example.org")
    cat = OpenLibrary(tmp_path, online=False)
    cat.close()
    assert cat.min_interval == 0.4 and "ops@example.org" in cat.client.headers["User-Agent"]
    monkeypatch.setenv("LIBRARY_OL_MIN_INTERVAL_S", "nope")
    assert OL.min_interval_s() == OL.DEFAULT_MIN_INTERVAL_S


def test_resolve_stage_skips_readings_after_the_engine_deadline_without_querying(tmp_path):
    import json, time
    reading = {"observation_id": "o1", "visible_text": "Sombras de Roma\nAna Ruiz", "provenance": {}}
    path = tmp_path / "readings.json"
    path.write_text(json.dumps({"schema_version": "1", "run_id": "r", "stage": "read", "status": "ok", "data": {"readings": [reading]}, "issues": [], "metrics": {}}))
    urls = []
    out = R.resolve(path, tmp_path / "run", {"catalogue": OpenLibrary(tmp_path / "cache", online=True, transport=transport(urls)), "deadline_monotonic": time.monotonic() - 1})
    res = json.loads(out.read_text())["data"]["resolutions"][0]
    assert res["decision"] == "no_match" and "resolve_budget_exhausted" in res["issues"][0] and urls == []


def test_single_term_fallbacks_are_skipped_when_the_all_terms_query_returned_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr("providers.openlibrary.time.sleep", lambda s: None)
    monkeypatch.delenv("LIBRARY_FRAGMENT_SINGLES", raising=False)
    urls = []
    cat = OpenLibrary(tmp_path, online=True, transport=transport(urls))
    out = R.resolve_text("Sombras de Roma\nAna Ruiz", cat)
    cat.close()
    assert out["decision"] == "accepted" and len(urls) == 1 and len(out["queries"]) == 1


def test_single_term_fallbacks_still_run_when_the_all_terms_query_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("providers.openlibrary.time.sleep", lambda s: None)
    monkeypatch.delenv("LIBRARY_FRAGMENT_SINGLES", raising=False)
    urls = []

    def handler(request):                       # an AND query with a truncated OCR word matches nothing; only the single surname finds the work
        urls.append(str(request.url))
        docs = [{"key": "/works/OL1W", "title": "Sombras de Roma", "author_name": ["Ana Ruiz"], "editions": {"docs": [{"key": "/books/OL1M", "title": "Sombras de Roma"}]}}]
        return httpx.Response(200, json={"numFound": 1, "docs": docs if request.url.params["q"] == "ruiz" else []})
    cat = OpenLibrary(tmp_path, online=True, transport=httpx.MockTransport(handler))
    out = R.resolve_text("Sombras de Romx\nAna Ruiz", cat)
    cat.close()
    assert len(urls) >= 2 and len(out["queries"]) >= 2          # the all-terms query was empty, so the single-term fallbacks ran


# ---- duplicate catalogue records of ONE book (Open Library keeps several works per book) ------------------------------------------------------------------------------
def test_article_and_punctuation_variants_of_one_title_are_one_book_not_ambiguous():
    two = [cand("The Bourne Identity", ["Robert Ludlum"], "/works/OL1W"), cand("Bourne Identity,the", ["Robert Ludlum"], "/works/OL2W")]
    decision, res = ev("THE BOURNE IDENTITY\nROBERT LUDLUM", two)
    assert decision == "accept" and res["work"] == "/works/OL1W"                              # first listed (relevance order); the variant spelling is not a rival


def test_different_authors_or_extra_title_words_still_abstain_as_distinct_books():
    assert ev("Dune Messiah\nFrank Herbert", [cand("Dune Messiah", ["Frank Herbert"], "/works/OL1W"), cand("Dune Messiah", ["Brian Herbert"], "/works/OL2W")])[0] == "accept"   # visible given name decides
    assert ev("Dune\nHerbert", [cand("Dune", ["Frank Herbert"], "/works/OL1W"), cand("Dune", ["Brian Herbert"], "/works/OL2W")])[0] == "abstain_ambiguous"


def _dup_transport(docs):
    return httpx.MockTransport(lambda request: httpx.Response(200, json={"numFound": len(docs), "docs": docs}))


def _doc(key, title, authors):
    return {"key": key, "title": title, "author_name": authors, "editions": {"docs": []}}


def test_complete_title_path_collapses_duplicate_works_with_the_same_title_and_authors(tmp_path, monkeypatch):
    monkeypatch.setattr("providers.openlibrary.time.sleep", lambda s: None)
    docs = [_doc("/works/OL1W", "The Things They Carried", ["Tim O'Brien"]), _doc("/works/OL2W", "The things they carried", ["Tim O'Brien"]),
            _doc("/works/OL3W", "The Things They Carried", ["Tim O'Brien"])]
    cat = OpenLibrary(tmp_path, online=True, transport=_dup_transport(docs))
    res = R.resolve_one({"observation_id": "o1", "visible_text": "THE THINGS THEY CARRIED\nTim O'Brien", "title_text": "THE THINGS THEY CARRIED", "author_text": "Tim O'Brien",
                         "title_complete": True, "provenance": {}}, cat)
    cat.close()
    assert res["decision"] == "accepted" and res["work"] == "/works/OL1W" and res["catalogue_title"] == "The Things They Carried"
    assert res["matching_evidence"][0]["duplicate_catalogue_works_same_title_and_authors"] == ["/works/OL2W", "/works/OL3W"]


def test_complete_title_path_keeps_ambiguity_when_author_names_differ(tmp_path, monkeypatch):
    monkeypatch.setattr("providers.openlibrary.time.sleep", lambda s: None)
    reading = {"observation_id": "o1", "visible_text": "PASSING\nNella Larsen", "title_text": "PASSING", "author_text": "Nella Larsen", "title_complete": True, "provenance": {}}
    docs = [_doc("/works/OL1W", "Passing", ["Nella Larsen"]), _doc("/works/OL2W", "Passing", ["Nella Larsen", "Tamar Mishmar"])]      # a co-author/translator differs: not provably one book
    cat = OpenLibrary(tmp_path, online=True, transport=_dup_transport(docs))
    res = R.resolve_one(reading, cat)
    cat.close()
    assert res["decision"] == "ambiguous" and res["work"] is None
