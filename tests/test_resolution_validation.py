"""Independent production catalogue review: synthetic evidence, never truth.

Only pure resolver calls and httpx MockTransport are used. No hosted requests,
OCR/model execution, downloads, real labels or run artifacts are written.
Red tests are review specifications; do not weaken them to bless a candidate.
"""
import httpx
import pytest

from pipeline import resolve as R
from providers.openlibrary import OpenLibrary


def candidate(title, authors=("Anne Miller",), work="/works/OL1W"):
    return {"work": work, "titles": title if isinstance(title, list) else [title],
            "authors": list(authors), "source": "synthetic", "url": None}


def must_abstain(text, candidates):
    decision, result = R.evaluate_candidates(text, candidates)
    assert decision != "accept", result


@pytest.mark.parametrize("suffix", ["2", "II"])
def test_series_volume_mismatch_is_not_erased_as_short_token(suffix):
    must_abstain("Chronicles of Amber 1\nRoger Zelazny",
                 [candidate("Chronicles of Amber " + suffix, ("Roger Zelazny",))])


def test_series_volume_missing_in_ocr_cannot_be_completed():
    must_abstain("Chronicles of Amber\nRoger Zelazny",
                 [candidate("Chronicles of Amber 2", ("Roger Zelazny",))])


def test_075_coverage_does_not_license_an_unseen_title_word():
    must_abstain("Alpha Beta Gamma\nAnne Miller", [candidate("Alpha Beta Gamma Delta")])


def test_subtitle_cannot_be_emitted_when_only_main_title_is_visible():
    must_abstain("Alpha Beta\nAnne Miller", [candidate("Alpha Beta : Hidden Sequel")])


def test_spanish_title_cannot_grow_an_unseen_english_subtitle():
    must_abstain("El mundo perdido\nArthur Conan Doyle",
                 [candidate("El mundo perdido : The Lost World", ("Arthur Conan Doyle",))])


def test_translation_is_rejected_even_when_names_and_author_overlap():
    must_abstain("Los Connellys de County Down\nJoan Dunn",
                 [candidate("The Connellys of County Down", ("Joan Dunn",))])


def test_native_spanish_alias_is_work_only_and_literal_text_is_unchanged():
    text = "La sombra del viento\nCarlos Ruiz Zafón"
    class Catalogue:
        def search_fragments(self, terms):
            return {"queries": [], "candidates": [candidate(
                ["The Shadow of the Wind", "La sombra del viento"], ("Carlos Ruiz Zafón",))]}
    reading = {"observation_id": "synthetic-es", "visible_text": text,
               "language_code": "es", "language_evidence": "La sombra del viento"}
    result = R.resolve_one(reading, Catalogue(), fragments=True)
    assert result["decision"] == "accepted"
    assert result["visible_text"] == text and result["title"] == "La sombra del viento"
    assert result["edition"] is None and result["isbn"] is None


def test_author_only_short_spine_cannot_become_a_title():
    must_abstain("Anne Miller", [candidate("Miller")])


def test_four_author_tokens_do_not_supply_independent_title_evidence():
    # Synthetic name repeated as a catalogue title: the >=4 exemption must not
    # turn an author-only line into independent title evidence.
    name = "George Raymond Richard Martin"
    must_abstain(name, [candidate(name, (name,))])


def test_same_title_different_authors_prefers_explicit_visible_given_name():
    candidates = [candidate("Dune Messiah", ("John Herbert",)),
                  candidate("Dune Messiah", ("Frank Herbert",), "/works/OL2W")]
    decision, result = R.evaluate_candidates("Dune Messiah\nFrank Herbert", candidates)
    assert decision == "accept" and result["work"] == "/works/OL2W"


def test_visible_given_name_contradiction_rejects_the_only_retrieved_work():
    must_abstain("Dune Messiah\nFrank Herbert", [candidate("Dune Messiah", ("John Herbert",))])


def test_long_title_exemption_does_not_override_visible_author_contradiction():
    must_abstain("Curious Copper Lantern Mountain\nAnne Miller",
                 [candidate("Curious Copper Lantern Mountain", ("John Wilson",))])


def test_same_title_with_only_shared_surname_is_ambiguous():
    must_abstain("Dune Messiah\nHerbert", [candidate("Dune Messiah", ("John Herbert",)),
                 candidate("Dune Messiah", ("Frank Herbert",), "/works/OL2W")])


def test_plausibly_fused_author_tokens_are_supported():
    decision, result = R.evaluate_candidates("CONNIGGULDEN\nLos hijos del imperio",
                                            [candidate("Los hijos del imperio", ("Conn Iggulden",))])
    assert decision == "accept" and result["independent_author_evidence"]


def test_different_surname_is_not_a_fused_match_just_because_it_contains_a_name():
    must_abstain("Dune Messiah\nFrank Herbertson", [candidate("Dune Messiah", ("Frank Herbert",))])


def test_fused_title_is_not_silently_repaired_to_catalogue_title():
    must_abstain("DUNEMESSIAH\nFrank Herbert", [candidate("Dune Messiah", ("Frank Herbert",))])


def test_repeated_title_tokens_do_not_create_the_four_token_exemption():
    must_abstain("Libertad Libertad Libertad Libertad", [candidate("Libertad Libertad Libertad Libertad")])


def test_complete_title_path_also_requires_author_for_fewer_than_four_content_tokens():
    class Catalogue:
        def search(self, *args, **kwargs):
            return {"payload": {"numFound": 1, "docs": [{"key": "/works/OL1W",
                    "title": "The Silver Lantern", "author_name": ["Anne Miller"]}]},
                    "url": "synthetic", "retrieved_at": "synthetic"}
    reading = {"observation_id": "synthetic-complete", "visible_text": "The Silver Lantern",
               "title_text": "The Silver Lantern", "title_complete": True}
    result = R.resolve_one(reading, Catalogue(), fragments=True)
    assert result["decision"] != "accepted", result


@pytest.mark.parametrize("work", [None, "/books/OL1M", "unvalidated-work-key"])
def test_fragment_accept_requires_a_valid_catalogue_work_key(work):
    must_abstain("Alpha Beta\nAnne Miller", [candidate("Alpha Beta", work=work)])


def test_offline_cache_miss_is_reasoned_abstention(tmp_path):
    cat = OpenLibrary(tmp_path / "cache", online=False)
    try:
        result = R.resolve_text("Alpha Beta\nAnne Miller", cat)
        assert result["decision"] == "abstain" and result["title"] is None
        assert "catalogue_offline_cache_miss" in result["reason"] and cat.requests == 0
    finally:
        cat.close()


def test_later_query_error_is_not_silently_converted_into_unique_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("providers.openlibrary.time.sleep", lambda _: None)
    monkeypatch.setenv("LIBRARY_FRAGMENT_SINGLES", "always")      # the later (single-term) queries exist only in this mode; the default skips them after a non-empty first answer
    urls = []
    def handle(request):
        urls.append(str(request.url))
        if len(urls) > 1:
            return httpx.Response(429, json={})
        return httpx.Response(200, json={"numFound": 1, "docs": [{
            "key": "/works/OL1W", "title": "Alpha Beta", "author_name": ["Anne Miller"]}]})
    cat = OpenLibrary(tmp_path / "cache", online=True, transport=httpx.MockTransport(handle))
    try:
        result = R.resolve_text("Alpha Beta\nAnne Miller", cat)
        assert result["decision"] == "abstain", result
        assert result["reason"] and len(urls) > 1
    finally:
        cat.close()


def test_truncated_results_cannot_establish_one_unambiguous_work(tmp_path, monkeypatch):
    monkeypatch.setattr("providers.openlibrary.time.sleep", lambda _: None)
    def handle(request):
        return httpx.Response(200, json={"numFound": 200, "docs": [{
            "key": "/works/OL1W", "title": "Alpha Beta", "author_name": ["Anne Miller"]}]})
    cat = OpenLibrary(tmp_path / "cache", online=True, transport=httpx.MockTransport(handle))
    try:
        result = R.resolve_text("Alpha Beta\nAnne Miller", cat)
        assert result["decision"] == "abstain", result
    finally:
        cat.close()
