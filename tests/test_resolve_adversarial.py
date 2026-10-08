"""Adversarial review of pipeline.resolve (post 008288c). Synthetic candidates only: no network, no OCR, no runs.

Cases cover Spanish/English, partial spines and look-alike works. Tests that currently FAIL against production are kept as strict xfail
(reason names the defect id) so the suite stays green but any fix flips them to XPASS(strict) = failure
until the marker is removed. They are real defects, not deferrals.
"""
import pytest

from pipeline import resolve as R


def cand(titles, authors=("Anne Miller",), work="/works/OL1W"):
    return {"work": work, "titles": titles if isinstance(titles, list) else [titles],
            "authors": list(authors), "source": "synthetic", "url": None}


def verdict(text, candidates):
    return R.evaluate_candidates(text, candidates)


def must_abstain(text, candidates):
    decision, result = verdict(text, candidates)
    assert decision != "accept", result


# ---- partial spines (Spanish): a truncated title never completes to the catalogue title -------------------------------------------------
@pytest.mark.parametrize("titles", [["La sombra del viento"], ["The Shadow of the Wind", "La sombra del viento"]])
def test_partial_spanish_spine_is_not_completed(titles):
    must_abstain("La sombra del\nCarlos Ruiz Zafón", [cand(titles, ("Carlos Ruiz Zafón",))])


# ---- look-alike works: sibling title that contains the other, same author ---------------------------------------------------------------
def test_sequel_spine_is_not_resolved_to_the_shorter_first_title():
    must_abstain("Dune Messiah\nFrank Herbert", [cand("Dune", ("Frank Herbert",))])


def test_sequel_spine_picks_the_sequel_when_both_are_retrieved():
    decision, result = verdict("Dune Messiah\nFrank Herbert",
                               [cand("Dune", ("Frank Herbert",)), cand("Dune Messiah", ("Frank Herbert",), "/works/OL2W")])
    assert decision == "accept" and result["work"] == "/works/OL2W"


def test_different_book_by_same_author_with_one_shared_title_word_abstains():
    must_abstain("Ficciones\nJorge Luis Borges", [cand("El Aleph", ("Jorge Luis Borges",)), cand("Ficciones y El Aleph", ("Jorge Luis Borges",), "/works/OL2W")])


# ---- Spanish/English: no cross-language completion even with shared proper names -------------------------------------------------------
@pytest.mark.parametrize("text,title", [
    ("The Shadow of the Wind\nCarlos Ruiz Zafón", "La sombra del viento"),
    ("La sombra del viento\nCarlos Ruiz Zafón", "The Shadow of the Wind"),
    ("Lost World\nArthur Conan Doyle", "El mundo perdido : Lost World"),
    ("El mundo perdido\nArthur Conan Doyle", "The Lost World"),
])
def test_cross_language_title_is_not_resolved(text, title):
    author = text.split("\n")[1]
    must_abstain(text, [cand(title, (author,))])


def test_unaccented_ocr_resolves_to_the_accented_catalogue_title_literally():
    decision, result = verdict("Pedro Paramo\nJuan Rulfo", [cand("Pedro Páramo", ("Juan Rulfo",))])
    assert decision == "accept" and result["title"] == "Pedro Páramo"


# ---- defects found by this review: B1/B2/B3 fixed (markers removed) ---------------------------------------
@pytest.mark.parametrize("catalogue_title", ["Chronicles of Amber I", "Chronicles of Amber"])
def test_visible_roman_volume_ii_never_resolves_to_another_volume(catalogue_title):
    must_abstain("Chronicles of Amber II\nRoger Zelazny", [cand(catalogue_title, ("Roger Zelazny",))])


def test_visible_author_initial_contradicting_given_name_abstains():
    must_abstain("Dune Messiah\nB. Herbert", [cand("Dune Messiah", ("Frank Herbert",))])


@pytest.mark.parametrize("text,title,author", [
    ("La amante\nMarguerite Duras", "El amante", "Marguerite Duras"),
    ("Los miserables\nVíctor Hugo", "Les Misérables", "Victor Hugo"),
])
def test_visible_article_that_differs_from_catalogue_article_abstains(text, title, author):
    must_abstain(text, [cand(title, (author,))])


# ---- recheck of the B1/B2 fixes (probes around the fix, not only the original repro) ----------------------------------------------------
@pytest.mark.parametrize("text,title", [
    ("Chronicles of Amber I\nRoger Zelazny", "Chronicles of Amber II"),
    ("Chronicles of Amber\nRoger Zelazny", "Chronicles of Amber II"),
    ("Tomo IV\nAnne Miller", "Tomo III"),
])
def test_volume_numbers_roman_and_arabic_must_agree(text, title):
    must_abstain(text, [cand(title, (text.split("\n")[1],))])


@pytest.mark.parametrize("text,title", [("Chronicles of Amber I\nAnne Miller", "Chronicles of Amber I"),
                                        ("Chronicles of Amber II\nAnne Miller", "Chronicles of Amber II")])
def test_matching_volume_still_resolves(text, title):
    assert verdict(text, [cand(title)])[0] == "accept"


@pytest.mark.parametrize("line,author,expect", [
    ("F. Herbert", "Frank Herbert", True), ("B. Herbert", "Frank Herbert", False), ("Herbert B.", "Frank Herbert", False),
    ("J.R.R. Tolkien", "John Ronald Reuel Tolkien", True), ("J.R.R. Tolkien", "Christopher Tolkien", False),
    ("G. García Márquez", "Gabriel García Márquez", True),
])
def test_author_initials_are_checked_against_the_catalogue_given_names(line, author, expect):
    decision, _ = verdict("The Silver Lantern\n" + line, [cand("The Silver Lantern", (author,))])
    assert (decision == "accept") == expect


def test_initial_selects_the_matching_author_when_two_same_surname_works_are_retrieved():
    decision, result = verdict("Dune Messiah\nB. Herbert", [cand("Dune Messiah", ("Frank Herbert",)), cand("Dune Messiah", ("Brian Herbert",), "/works/OL2W")])
    assert decision == "accept" and result["work"] == "/works/OL2W"


# ---- B3 fix: a printed article is compared with the catalogue's own; a MISSING printed article is not a conflict ----------------------------------------
@pytest.mark.parametrize("text,title,author", [
    ("Sombra del viento\nCarlos Ruiz Zafón", "La sombra del viento", "Carlos Ruiz Zafón"),       # spines often drop the article
    ("La amante\nMarguerite Duras", "La amante", "Marguerite Duras"),                             # same article
    ("L'amant\nMarguerite Duras", "L'amant", "Marguerite Duras"),
    ("La amante\nMarguerite Duras", "Amante", "Marguerite Duras"),                                # catalogue title has no article: nothing unseen is emitted
])
def test_article_rule_keeps_matching_or_missing_articles(text, title, author):
    assert verdict(text, [cand(title, (author,))])[0] == "accept"


@pytest.mark.parametrize("text,title,author", [
    ("Las amantes\nMarguerite Duras", "Los amantes", "Marguerite Duras"),
    ("The amante\nMarguerite Duras", "El amante", "Marguerite Duras"),
])
def test_article_rule_other_conflicts_abstain(text, title, author):
    must_abstain(text, [cand(title, (author,))])
