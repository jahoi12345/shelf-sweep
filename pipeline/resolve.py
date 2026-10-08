"""Conservative catalogue resolution of literal readings. Unknowns stay unknown."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

from pipeline.contracts import envelope, write_json
from pipeline.read_spines import load, records, sanitize_reading
from providers.openlibrary import OpenLibrary


def normalize(text):
    text = unicodedata.normalize("NFKD", text or "").casefold()
    return " ".join(re.findall(r"[^\W_]+", "".join(c for c in text if not unicodedata.combining(c))))


def normalize_title(text):
    """Compare literal native titles without losing Spanish diacritics or translating."""
    text = unicodedata.normalize("NFC", text or "").casefold()
    return " ".join(re.findall(r"[^\W_]+", text))


_GB_KEY = re.compile(r"gb:[A-Za-z0-9_-]{8,16}")
_LEAD_ARTICLE_WORDS = {"the", "a", "an", "el", "la", "los", "las", "un", "una", "lo", "les", "els", "l"}


def valid_work_key(key):
    """An Open Library work (/works/OL..W, or OL..W as search docs give it) or a Google Books volume (gb:<id>, the second catalogue)."""
    return bool(re.fullmatch(r"(?:/works/)?OL\d+W", key or "") or _GB_KEY.fullmatch(key or ""))


def title_matches(catalogue_title, wanted, folded=False):
    """The literal-title rule on normalised forms: case-insensitive equality (accent-insensitive when ``folded``; ``wanted`` must be normalised the same way), also when
    the catalogue title carries ONE leading article that the spine does not print (spines drop it: "VIDA DE ..." vs "La vida de ..."). Every visible word must still be the
    catalogue's, in order; nothing else may differ."""
    have = (normalize if folded else normalize_title)(catalogue_title)
    if not have or not wanted:
        return False
    if have == wanted:
        return True
    head, _, rest = have.partition(" ")
    return head in _LEAD_ARTICLE_WORDS and rest == wanted and wanted.split(" ", 1)[0] not in _LEAD_ARTICLE_WORDS


def catalogue_title_evidence(reading, candidate, folded=False):
    """folded=False: accents must agree (the production rule; 'Ébora' and 'Ebora' are different works). The complete-title path retries with folded=True only when no
    candidate matched with accents (OCR drops accents: 'ESPANA' for 'España')."""
    wanted = (normalize if folded else normalize_title)(reading.get("title_text"))
    if not wanted:
        return []
    evidence = []
    if title_matches(candidate.get("title"), wanted, folded):
        evidence.append({"title": candidate["title"], "source": "work_title", "edition_key": None, "language": None})
    expected = {"es": "spa", "en": "eng"}.get(reading.get("language_code"))
    for edition in (candidate.get("editions") or {}).get("docs", []):
        languages = edition.get("language", [])
        key = edition.get("key", "")
        if (title_matches(edition.get("title"), wanted, folded) and (re.fullmatch(r"/books/OL\d+M", key) or _GB_KEY.fullmatch(key))
                and (not expected or not languages or expected in languages)):
            evidence.append({"title": edition["title"], "source": "native_title_in_work_edition",
                             "edition_key": key, "language": languages})
    return evidence


def isbn_value(text):
    value = re.sub(r"[\s-]", "", (text or "").upper())
    if len(value) == 13 and value.isdigit() and value[:3] in ("978", "979"):
        return value if sum(int(c)*(1 if i % 2 == 0 else 3) for i, c in enumerate(value)) % 10 == 0 else None
    if len(value) == 10 and value[:9].isdigit() and (value[-1].isdigit() or value[-1] == "X"):
        digits = [10 if c == "X" else int(c) for c in value]
        return value if sum((10-i)*c for i, c in enumerate(digits)) % 11 == 0 else None
    return None


def supported_work(reading, candidate, folded=False):
    title = normalize_title(reading.get("title_text"))
    if not title or not reading.get("title_complete") or "[?]" in reading.get("title_text", ""):
        return False
    if not catalogue_title_evidence(reading, candidate, folded):
        return False
    author = normalize(reading.get("author_text"))
    names = candidate.get("author_name", [])
    if author:
        # Exact visible token containment handles printed surname without inventing initials.
        if len(author) < 4 or not any(set(author.split()) <= set(normalize(n).split()) for n in names):
            return False
    else:
        # Short/generic titles (e.g. Politics) require author support: same bar as the fragment rule (>= 4 distinct content tokens).
        if len(set(content_tokens(title.split()))) < 4:
            return False
    return valid_work_key(candidate.get("key", ""))


def edition_language_conflict(reading, edition):
    expected = {"es": "spa", "en": "eng"}.get(reading.get("language_code"))
    languages = {x.get("key", "").rsplit("/", 1)[-1] for x in edition.get("languages", []) if isinstance(x, dict)}
    return bool(expected and languages and expected not in languages)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Fragment resolution (rule "v6"): resolve literal OCR text (title/author/publisher lines in any order, fragments allowed) to a catalogue work
# ONLY when the catalogue title is covered by the visible tokens and independent author evidence (or a long enough title) supports it. Abstain by default.
# Evaluated offline on development runs (library samples A/B, YT1/YT2 dev sets, all model-read or catalogue-derived, never human truth).
# ---------------------------------------------------------------------------------------------------------------------------------------------
# Retrieval completeness: a first (all-terms) query that returned fewer works than it could have (returned < min(numFound, limit)) or reports no total is an incomplete/inconsistent
# answer -> abstain. A relevance-ranked answer that is merely capped by the request limit (numFound > limit == returned) is accepted by default because the rule already needs full
# title coverage + author evidence; set STRICT_TRUNCATION=True to also abstain there (offline YT2 replay: 7 of 24 accepts have capped first queries, see docs).
STRICT_TRUNCATION = False
COVERAGE_FLOOR = 0.7          # production policy: never accept below this share of the candidate title's content tokens
_STOP = set("""de del la las el los lo un una unos unas uns unes y e i o u en al a amb per pel pels com que dels des els les para con por sobre entre sin
the and for with from into of to in on at by an as is are or not its their his her our""".split())
_ARTICLES = {"the", "a", "an", "el", "la", "los", "las", "un", "una", "lo", "les", "els", "l"}
_LEAD_ARTICLES = _ARTICLES | {"le", "il", "gli", "os", "as", "um", "uma", "der", "die", "das"}   # leading articles compared against the printed one (es/en/fr/ca/it/pt/de)
# Function words that identify a language only (shared words such as "a", "i", "la", "de" are deliberately excluded).
_EN_ONLY = {"the", "and", "of", "to", "with", "from", "by", "for", "in", "on", "at", "an", "is", "are", "or"}
_ES_CA_ONLY = {"el", "los", "las", "y", "del", "para", "con", "por", "un", "una", "els", "les", "amb", "dels", "pel", "pels", "sobre", "entre"}


def fold(text):
    """Accent- and case-insensitive tokens for COMPARISON only (output titles are always returned literally from the catalogue)."""
    return normalize(text)


def main_title(title):
    head = re.split(r"\s[:/=]\s", title or "", maxsplit=1)[0].strip(" ,;:./")
    return head or (title or "").strip(" ,;:./")


def content_tokens(tokens):
    return [t for t in tokens if len(t) >= 3 and t not in _STOP]


def query_terms(text, k=8):
    toks = [t for t in fold(text).split() if len(t) >= 4 and not t.isdigit() and t not in _STOP]
    return sorted(dict.fromkeys(toks), key=lambda t: -len(t))[:k]


def _language_conflict(title_tokens, ocr_tokens):
    """True when the catalogue title and the visible text use function words of opposite languages (English vs Spanish/Catalan)."""
    t, o = set(title_tokens), set(ocr_tokens)
    t_en, t_es = bool(t & _EN_ONLY), bool(t & _ES_CA_ONLY)
    o_en, o_es = bool(o & _EN_ONLY), bool(o & _ES_CA_ONLY)
    return (t_en and not t_es and o_es and not o_en) or (t_es and not t_en and o_en and not o_es)


_ROMAN = {"ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12}
_WORK_KEY = re.compile(r"/works/OL\d+W")


def _volume_numbers(tokens, roman=True):
    """Volume/number tokens (1, 2, II ...) as ints. Short numeric tokens are title-specific (Chronicles of Amber 1 != 2), never noise."""
    out = set()
    for t in tokens:
        if t.isdigit() and len(t) <= 3:
            out.add(int(t))
        elif roman and t in _ROMAN:
            out.add(_ROMAN[t])
    return out


def _fused_parts(tok, ocr, names):
    """OCR tokens that are one surname glued to a given name of the SAME candidate (CONNIGGULDEN = conn + iggulden) -> {ocr_token: given-name part}.
    An arbitrary suffix/prefix (Herbert -> Herbertson) is a different surname, not a fused match."""
    found = {}
    if len(tok) < 6:
        return found
    for o in ocr:
        if o == tok or not (o.startswith(tok) or o.endswith(tok)):
            continue
        rem = o[len(tok):] if o.startswith(tok) else o[:-len(tok)]
        if rem in names and rem != tok:
            found[o] = rem
    return found


def _article_conflict(all_tokens, content, rows):
    """True when the spine prints a leading article right before the title's first content word and it is a DIFFERENT article than the catalogue title's own
    (La amante vs El amante, Los miserables vs Les Miserables): the emitted catalogue title would carry an unseen/contradicted word. A missing printed article
    (spines often drop it) or a catalogue title without one is not a conflict."""
    first = next((i for i, t in enumerate(all_tokens) if t in content), None)
    if not first or all_tokens[first - 1] not in _LEAD_ARTICLES:
        return False
    lead = all_tokens[first - 1]
    for row in rows:
        for j, t in enumerate(row):
            if t == all_tokens[first] and j > 0 and row[j - 1] in _LEAD_ARTICLES and row[j - 1] != lead:
                return True
    return False


def _author_evidence(authors, ocr_tokens, title_content, rows=None, title_tokens=()):
    """-> (ok, score, best_name). ok = a surname that is NOT one of the title's own tokens is visible (a surname of >= 6 letters may also sit inside one fused OCR token made of
    that surname and a given name of the same author, e.g. CONNIGGULDEN). When ``rows`` (the visible lines as token lists) are given, a line that shows the surname but also shows
    other words that belong neither to the title nor to the candidate's author names CONTRADICTS the candidate (Frank Herbert is visible, John Herbert is the only catalogue author) -> not ok.
    score = best share of an author's name tokens that are visible; it only ranks candidates that share a surname."""
    ocr = set(ocr_tokens)
    all_names = {t for n in authors or [] for t in fold(n).split()}
    ok, score, best_name = False, 0.0, ""
    for name in authors or []:
        sur_part = name.split(",")[0] if "," in name else (name.split()[-1] if name.split() else "")
        sur = {t for t in fold(sur_part).split() if len(t) >= 3} - set(title_content)
        toks = [t for t in fold(name).split() if len(t) >= 2]
        fused = {}
        for t in sur:
            fused.update(_fused_parts(t, ocr, set(toks)))
        seen_tok = ocr | set(fused.values())
        hit = {t for t in sur if t in ocr} | {t for t in sur if _fused_parts(t, ocr, set(toks))}
        name_ok = bool(hit)
        if name_ok and rows is not None:
            for row in rows:
                if not (hit & set(row) or set(fused) & set(row)):
                    continue
                stray = [t for t in row if len(t) >= 2 and t not in _STOP and t not in title_tokens and t not in all_names and t not in fused]
                # a visible initial must be the first letter of a given name of the candidate (B. Herbert != Frank Herbert)
                firsts = {n[0] for n in all_names}
                wrong_initial = [t for t in row if len(t) == 1 and t.isalpha() and t not in _STOP and t not in title_tokens and t not in firsts]
                if stray or wrong_initial:
                    name_ok = False
        if name_ok:
            ok = True
        if toks:
            share = sum(1 for t in toks if t in seen_tok) / len(toks)
            if share > score or not best_name:
                score, best_name = max(score, share), " ".join(sorted(toks))
    return ok, score, best_name


def _line_coverage(text, title_tokens, author_tokens):
    rows = [set(fold(r).split()) for r in text.split("\n")]
    shown = set()
    for row in rows:
        if row & title_tokens:
            shown |= {t for t in row if len(t) >= 3 and t not in _STOP and not any(ch.isdigit() for ch in t)}
    return (len(shown & (title_tokens | author_tokens)) / len(shown)) if shown else 0.0


def evaluate_candidates(text, candidates, coverage=COVERAGE_FLOOR):
    """Decide whether literal visible ``text`` identifies exactly one catalogue work. Returns (decision, result).

    candidates: [{"work": "/works/OL..W", "titles": [work title, edition titles...], "authors": [...], "source": "...", "url": "..."}]
    decision is "accept" or an "abstain_*" reason. The accepted title is returned LITERALLY as the catalogue gives it (never translated, never completed).
    Rules: valid /works/OL..W key; EVERY content token of the emitted catalogue title (main title AND any subtitle) and every volume number (1/2/II) is visible on the spine
    (``coverage`` is only a floor, the effective requirement is 1.0: a title with an unseen word is never emitted); independent author evidence that no visible line contradicts,
    unless the title has >= 4 distinct content tokens that are not author-name tokens and the spine shows nothing beyond title and author; no English-vs-Spanish/Catalan
    function-word conflict; line-coverage guard; exactly one work left among the candidates with equal author evidence (same title+surname duplicates count as one work). Abstain otherwise.
    """
    coverage = max(float(coverage), COVERAGE_FLOOR)
    otoks = fold(text or "").split(); ocr = set(otoks)
    if sum(len(t) for t in otoks) < 4:
        return "abstain_text_too_short", None
    rows = [fold(r).split() for r in (text or "").split("\n")]
    ocr_content = {t for t in otoks if len(t) >= 3 and t not in _STOP and not t.isdigit()}
    scored = []
    for cand in candidates or []:
        if not valid_work_key(cand.get("work") or ""):
            continue
        authors = cand.get("authors") or []
        author_tok = {t for a in authors for t in fold(a).split()}
        best = None
        for title in cand.get("titles") or []:
            all_tokens = fold(title).split()
            C = list(dict.fromkeys(content_tokens(all_tokens)))
            if not C:
                continue
            uncovered = [t for t in C if t not in ocr]
            cov = 1 - len(uncovered) / len(C)
            if uncovered or cov < coverage:
                continue
            vols = _volume_numbers(all_tokens)
            title_rows = [r for r in rows if set(r) & set(C)]
            seen_vols = _volume_numbers([t for r in title_rows for t in r])
            if not vols <= _volume_numbers(otoks) or (seen_vols - vols):
                continue
            if _article_conflict(all_tokens, set(C), rows):
                continue
            author_ok, author_score, author_name = _author_evidence(authors, otoks, C, rows, set(all_tokens))
            if not author_ok:
                independent = [t for t in C if t not in author_tok]
                if len(independent) < 4 or (ocr_content - set(all_tokens) - author_tok):
                    continue
            if _language_conflict(all_tokens, otoks):
                continue
            if _line_coverage(text, set(C), author_tok) < 0.75:
                continue
            rank = (cov, author_ok, author_score, len(C))
            if best is None or rank > best[:4]:
                best = (cov, author_ok, author_score, len(C), title, uncovered, author_name)
        if best:
            scored.append((best, cand))
    if not scored:
        return "abstain_no_eligible_candidate", None
    scored.sort(key=lambda x: x[0][:4], reverse=True)
    top = scored[0]

    def key(sc):
        names = sorted({t for a in sc[1].get("authors") or [] for t in fold(a.split(",")[0] if "," in a else (a.split()[-1] if a.split() else "")).split() if len(t) >= 3})
        # catalogue records of ONE book differ in article placement and punctuation ("The Bourne Identity" / "Bourne Identity,the"): compare the content tokens of the main title
        return (tuple(sorted(set(content_tokens(fold(main_title(sc[0][4])).split())))) or fold(main_title(sc[0][4])), names[0] if names else "")

    rivals = {key(sc) for sc in scored if sc[0][0] >= top[0][0] - 0.10 and sc[0][1] == top[0][1]}
    if len(rivals) > 1:
        return "abstain_ambiguous", None
    # same (title, surname) but different given names (John vs Frank Herbert): the work whose author is better visible wins; a tie on that is ambiguous
    same = [sc for sc in scored if key(sc) == key(top) and sc[0][0] >= top[0][0] - 0.10 and sc[0][1] == top[0][1] and abs(sc[0][2] - top[0][2]) < 1e-9]
    if len({sc[0][6] for sc in same}) > 1 and len({sc[1].get("work") for sc in same}) > 1:
        return "abstain_ambiguous", None
    best, cand = top
    return "accept", {"title": best[4], "authors": cand.get("authors") or [], "work": cand.get("work"), "source": cand.get("source"), "source_url": cand.get("url"),
                      "google_books": cand.get("google_books"),
                      "coverage": best[0], "independent_author_evidence": best[1], "author_name_score": best[2], "title_content_tokens": best[3], "uncovered_title_tokens": best[5],
                      "rule": "fragment_v6_literal_catalogue_title_no_translation_no_completion"}


def resolve_text(text, catalogue, coverage=COVERAGE_FLOOR):
    """Resolve literal OCR text through an OpenLibrary-like ``catalogue`` (needs ``search_fragments``). Offline or failing catalogue => abstain with a reason."""
    terms = query_terms(text)
    out = {"decision": "abstain", "reason": None, "title": None, "authors": [], "work": None, "source_url": None, "queries": [], "coverage_floor": COVERAGE_FLOOR}
    if not terms:
        out["reason"] = "no_usable_query_terms"
        return out
    try:
        found = catalogue.search_fragments(terms)
    except Exception as exc:
        out["reason"] = f"catalogue_unavailable:{type(exc).__name__}:{exc}"
        return out
    out["queries"] = found.get("queries", [])
    if found.get("errors"):
        out["reason"] = "catalogue_incomplete:" + "; ".join(f"{e.get('q')}:{e.get('error')}" for e in found["errors"])
        return out
    if "total" in found:
        total, returned = found.get("total"), found.get("returned") or 0
        if total is None or returned < min(total, found.get("limit") or total) or (STRICT_TRUNCATION and total > returned):
            out["reason"] = f"catalogue_results_truncated:{total}>{returned}"
            return out
    decision, result = evaluate_candidates(text, found.get("candidates", []), coverage)
    if decision != "accept":
        out["reason"] = decision
        return out
    out.update(decision="accepted", reason="fragment_v6_accepted", title=result["title"], authors=result["authors"], work=result["work"], source_url=result["source_url"],
               google_books=result.get("google_books"),
               evidence={k: result[k] for k in ("coverage", "independent_author_evidence", "author_name_score", "title_content_tokens", "uncovered_title_tokens", "rule")})
    return out


def resolve_one(reading, catalogue, fragments=False, second=None):
    """fragments=True adds the v6 fragment fallback (see resolve_text): used only when the complete-title path did not accept.
    ``second`` (providers.googlebooks.GoogleBooks or None): when Open Library yields NO eligible candidate on either path (never when it was ambiguous or unavailable),
    the same reading goes through the same two paths against the second catalogue, under the same evidence rule; an accepted result carries catalogue='google_books'."""
    result = _resolve_complete_title(reading, catalogue)
    if fragments and result["decision"] != "accepted" and hasattr(catalogue, "search_fragments"):
        _fragment_fallback(result, catalogue)
    result.setdefault("catalogue", "open_library" if result["decision"] == "accepted" else None)
    if second is not None and result["decision"] != "accepted" and _no_eligible_candidate(result) and not getattr(second, "disabled", False):
        quota = lambda r: any("quota" in str(i) for i in r["issues"])
        alt = _resolve_complete_title(reading, second)
        if not quota(alt) and fragments and alt["decision"] != "accepted":
            _fragment_fallback(alt, second)
        if quota(alt):
            second.disabled = True                                                   # quota exhausted: no more requests this run (each would fail the same way)
        if alt["decision"] == "accepted" and hasattr(second, "dimensions") and str(alt.get("work") or "").startswith("gb:"):
            try:                                                                     # dimensions live only in the volume record: one lookup per ACCEPTED volume
                alt.setdefault("google_books", {})["dimensions"] = second.dimensions(alt["work"][3:])
            except Exception as exc:
                alt["issues"].append(f"google_books_dimensions_unavailable:{type(exc).__name__}:{exc}")
        if alt["decision"] == "accepted":
            alt["catalogue"] = "google_books"
            alt["issues"].append("resolved_by_second_catalogue_google_books:open_library_had_no_eligible_candidate")
            alt["open_library_attempt"] = {"decision": result["decision"], "issues": result["issues"][-3:]}
            return alt
        result["second_catalogue"] = {"catalogue": "google_books", "decision": alt["decision"], "issues": alt["issues"][-3:]}
    return result


def _no_eligible_candidate(result):
    """Open Library answered and nothing was eligible (not ambiguous, not unavailable, not 'no complete title' without a fragment attempt)."""
    issues = [str(i) for i in result.get("issues") or []]
    if result.get("decision") == "ambiguous" or any(i.startswith(("catalogue_unavailable", "catalogue_incomplete", "catalogue_results_truncated")) for i in issues):
        return False
    frag = (result.get("fragment_resolution") or {}).get("reason")
    if frag is not None and frag != "abstain_no_eligible_candidate":
        return False
    return bool(result.get("catalogue_query") or frag)


def _fragment_fallback(result, catalogue):
    text = result.get("visible_text")
    if not text:
        return
    frag = resolve_text(text, catalogue)
    result["fragment_resolution"] = {k: frag[k] for k in ("decision", "reason", "queries", "coverage_floor")}
    if frag["decision"] != "accepted":
        result["issues"].append("fragment_resolution_abstained:" + str(frag["reason"]))
        return
    visible = fold(text).split()
    title_tokens = fold(frag["title"]).split()
    title_content = content_tokens(title_tokens)
    rows = [fold(r).split() for r in text.split("\n")]
    if frag.get("google_books"):
        result["google_books"] = frag["google_books"]
    result.update(work=frag["work"], title=frag["title"], catalogue_title=frag["title"], source_url=frag["source_url"], decision="accepted",
                  author=next((a for a in frag["authors"] if _author_evidence([a], visible, title_content, rows, set(title_tokens))[0]), None))
    result["matching_evidence"] = [{"field": "fragment_title_author", "visible": text, "catalogue": {"title": frag["title"], "authors": frag["authors"]},
                                    **frag["evidence"]}]
    result["issues"].append("resolved_by_fragment_rule_v6_title_not_read_completely")
    result["issues"].append("edition_and_isbn_unknown_no_verified_printed_isbn")


def _base_result(reading, literal):
    return {"observation_id": reading["observation_id"],
            "visible_text": literal["visible_text"], "title_text": literal["title_text"],
            "edition_text": literal["edition_text"], "language_code": literal["language_code"],
            "language_evidence": literal["language_evidence"], "language_provenance": literal["language_provenance"],
            "provenance": dict(reading.get("provenance", {})), "catalogue_title": None, "work": None, "title": None, "author": None,
            "edition": None, "isbn": None, "publisher": literal["publisher_text"], "source_url": None, "matching_evidence": [],
            "decision": "no_match", "issues": [], "candidates": []}


def _budget_exhausted(reading):
    """A reading the wall-clock budget did not reach: no catalogue query was made, so it stays unidentified (counted, never guessed)."""
    result = _base_result(reading, sanitize_reading(reading))
    result["issues"].append("resolve_budget_exhausted:no_catalogue_query_made")
    return result


def _resolve_complete_title(reading, catalogue):
    observation_id = reading["observation_id"]
    literal = sanitize_reading(reading)
    result = _base_result(reading, literal)
    if not literal["title_text"]:
        result["issues"].append("no_complete_visible_title_no_catalogue_query")
        return result
    variants = query_variants(literal["title_text"], literal["language_code"]) if getattr(catalogue, "use_query_variants", True) else [literal["title_text"]]
    for vi, query in enumerate(variants):
        try:
            search = (catalogue.search(query, literal["author_text"], language=literal["language_code"])
                      if literal["language_code"] in {"es", "en"} else catalogue.search(query, literal["author_text"]))
        except Exception as exc:
            result["issues"].append(f"catalogue_unavailable:{type(exc).__name__}:{exc}")
            return result
        payload = search["payload"]
        docs = payload.get("docs", [])
        for folded in (False, True):                  # accents must agree first; accent-insensitive only when no candidate matched with them
            unique = {}
            result["candidates"] = []
            for doc in docs:
                key = doc.get("key", "")
                if key.startswith("OL"):
                    key = "/works/"+key
                support = supported_work(literal, doc, folded)
                candidate = {"work": key, "title": doc.get("title"), "authors": doc.get("author_name", []),
                             "source_url": doc.get("url") or ("https://openlibrary.org"+key if re.fullmatch(r"/works/OL\d+W", key) else None),
                             "visible_text_support": support, "accent_insensitive": folded,
                             "title_evidence": catalogue_title_evidence(literal, doc, folded),
                             **({"google_books": doc["google_books"]} if doc.get("google_books") else {})}
                result["candidates"].append(candidate)
                if support:
                    unique[key] = candidate
            if unique:
                break
        total = payload.get("numFound", payload.get("num_found"))
        truncated = total is None or total > len(docs)
        result["catalogue_query"] = {"url": search["url"], "retrieved_at": search["retrieved_at"],
                                     "num_found": total, "returned": len(docs), "query": query, "query_variant": vi}
        if unique or vi == len(variants) - 1:
            break                                     # a later variant is asked only when this query produced no eligible candidate at all
    duplicates = []
    if len(unique) > 1 and not truncated:
        # Open Library keeps several work records for one book (same title, same authors). Records that agree on the normalized title AND the full author-name set are ONE
        # book: take the first listed (relevance order) and cite the others; any difference in title or author names (translators, co-authors) stays ambiguous.
        def signature(c):
            return (normalize_title(c["title"]), frozenset(normalize(a) for a in c["authors"]))
        signatures = {signature(c) for c in unique.values()}
        if len(signatures) == 1:
            ordered = [c for c in result["candidates"] if c["work"] in unique]
            duplicates = [c["work"] for c in ordered[1:]]
            unique = {ordered[0]["work"]: ordered[0]}
    if len(unique) != 1 or truncated:
        result["decision"] = "ambiguous" if docs else "no_match"
        result["issues"].append("multiple_or_unsupported_work_matches" if not truncated else "catalogue_results_truncated_or_total_unknown")
        return result
    match = next(iter(unique.values()))
    if match.get("google_books"):
        result["google_books"] = match["google_books"]
    result.update(work=match["work"], title=literal["title_text"], catalogue_title=match["title"],
                  # Do not fill unseen author merely because catalogue returned one.
                  author=literal["author_text"], source_url=match["source_url"], decision="accepted")
    result["matching_evidence"] = [{"field": "title", "visible": literal["title_text"],
                                     "catalogue": match["title_evidence"], "rule": "complete_native_literal_title_exact_no_translation",
                                     **({"query_variant": result["catalogue_query"]["query"]} if result["catalogue_query"].get("query_variant") else {}),
                                     **({"duplicate_catalogue_works_same_title_and_authors": duplicates} if duplicates else {})}]
    if literal["author_text"]:
        result["matching_evidence"].append({"field": "author", "visible": literal["author_text"],
                                            "catalogue": match["authors"], "rule": "visible_author_tokens_supported"})
    isbn = isbn_value(literal.get("isbn_text"))
    if isbn:
        try:
            edition = catalogue.edition_by_isbn(isbn)
            record = edition["payload"]
            record_isbns = record.get("isbn_10", []) + record.get("isbn_13", [])
            works = [w.get("key") for w in record.get("works", [])]
            key = record.get("key", "")
            if (isbn in record_isbns and result["work"] in works
                    and normalize_title(record.get("title")) == normalize_title(literal["title_text"])
                    and not edition_language_conflict(literal, record)
                    and re.fullmatch(r"/books/OL\d+M", key)):
                result.update(edition=key, isbn=isbn, edition_languages=record.get("languages", []))
                result["matching_evidence"].append({"field": "edition_isbn", "visible": literal["isbn_text"],
                                                    "catalogue": isbn, "edition_key": key,
                                                    "source_url": edition["url"], "retrieved_at": edition["retrieved_at"],
                                                    "rule": "printed_checksum_valid_isbn_exact_edition_work_title"})
            else:
                result["issues"].append("printed_isbn_not_verified_by_specific_edition")
        except Exception as exc:
            result["issues"].append(f"edition_lookup_unavailable:{type(exc).__name__}:{exc}")
    else:
        result["issues"].append("edition_and_isbn_unknown_no_verified_printed_isbn")
    return result


def second_catalogue(cache_dir, *, online, environ=None):
    """The second catalogue per LIBRARY_CATALOGUE_SECOND (googlebooks|none). Default: googlebooks when the catalogue is online AND GOOGLE_BOOKS_API_KEY is set (anonymous
    Google Books requests share an exhausted daily quota: every one would fail and cost its pacing interval); cache-only (offline) runs use it whenever its cache exists."""
    env = environ if environ is not None else os.environ
    choice = env.get("LIBRARY_CATALOGUE_SECOND", "").strip().lower()
    if choice == "none":
        return None
    if choice != "googlebooks" and not (online and env.get("GOOGLE_BOOKS_API_KEY", "").strip()) and not (not online and (Path(cache_dir) / "googlebooks").exists()):
        return None
    from providers.googlebooks import GoogleBooks
    return GoogleBooks(Path(cache_dir) / "googlebooks", online=online)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Query normaliser (pure Python, no model). Query VARIANTS only: the accepted title is always the literal reading, checked by the unchanged evidence rule.
# ---------------------------------------------------------------------------------------------------------------------------------------------
_OCR_JUNK = re.compile(r"[^\w\s'’\-]+", re.UNICODE)


def clean_query(text):
    """Strip OCR noise from a reading before it is used as a catalogue query: punctuation runs and bars/brackets, stray one-letter tokens at either end
    (a lone 'I' from a spine edge), underscores and repeated spaces. Never used as the accepted title."""
    t = _OCR_JUNK.sub(" ", unicodedata.normalize("NFC", text or "")).replace("_", " ")
    toks = t.split()
    keep_first = {"y", "a", "e", "o", "u"}            # Spanish one-letter words that can open a title ('A sangre fría'); a lone 'I' or 'l' is a spine edge
    while toks and len(toks[0]) == 1 and toks[0].lower() not in keep_first:
        toks.pop(0)
    while toks and len(toks[-1]) == 1:                   # a title never ends in a one-letter word worth querying
        toks.pop()
    return " ".join(toks)


def spanish_article(word):
    """Definite article a Spanish catalogue title would print before ``word`` (morphology only; a query variant, never evidence)."""
    w = normalize(word)
    if w.endswith(("as",)):
        return "las"
    if w.endswith(("os", "es")):
        return "los"
    if w.endswith(("a", "ion", "dad", "tad", "tud", "umbre", "ie")):
        return "la"
    return "el"


def query_variants(title, language=None):
    """[the reading as given, the OCR-cleaned reading (if different), the reading with a restored leading article (if it starts with a title word)].
    'VIDA DE PI' -> ['VIDA DE PI', 'la VIDA DE PI']; an English reading gets 'the'; an unknown language gets the Spanish article only (the shelves are mostly Spanish;
    an English title rarely starts with a word Open Library indexes with 'the' omitted on the spine)."""
    out = [title]
    clean = clean_query(title)
    if clean and clean != title:
        out.append(clean)
    base = clean or title
    toks = normalize(base).split()
    if len(toks) >= 2 and toks[0] not in _LEAD_ARTICLE_WORDS and not toks[0].isdigit():
        art = "the" if language == "en" else spanish_article(toks[0])
        out.append(f"{art} {base}")
    return list(dict.fromkeys(out))


def resolve(readings_path, run_dir, config=None):
    config = config or {}
    config = {**config, **config.get("resolve", {})}
    started = time.perf_counter()
    root = Path(run_dir).resolve()
    artifact = load(readings_path)
    own_catalogue = config.get("catalogue") is None
    catalogue = config.get("catalogue") or OpenLibrary(config.get("cache_dir", root / "cache"),
                                                       online=config.get("catalogue_online", False))
    second = config.get("second_catalogue", "env")
    own_second = second == "env"
    if own_second:
        second = second_catalogue(config.get("cache_dir", root / "cache"), online=config.get("catalogue_online", False))
    try:
        deadline = config.get("deadline_monotonic")          # engine wall-clock budget (time.monotonic value); readings after it are skipped, not queried
        resolutions = [_budget_exhausted(r) if deadline is not None and time.monotonic() > deadline
                       else resolve_one(r, catalogue, fragments=config.get("fragment_resolution", True), second=second) for r in records(artifact, "readings")]
        metrics = {"elapsed_s": time.perf_counter()-started, "provider": "Open Library" + (" + Google Books (second catalogue)" if second is not None else ""),
                   "requests": getattr(catalogue, "requests", None),
                   "request_details": getattr(catalogue, "metrics", []),
                   "second_catalogue": None if second is None else {"catalogue": "google_books", "requests": second.requests,
                                                                    "cache_hits": sum(1 for m in second.metrics if m.get("cache_hit")),
                                                                    "disabled_after_quota_error": bool(getattr(second, "disabled", False)),
                                                                    "accepted": sum(1 for r in resolutions if r.get("catalogue") == "google_books"),
                                                                    "latency_s": [round(m["latency_s"], 3) for m in second.metrics if not m.get("cache_hit")][:200]},
                   "cost": 0 if own_catalogue else None, "cost_provenance": "Public catalogue HTTP; no model calls",
                   "accepted": sum(r["decision"] == "accepted" for r in resolutions)}
    finally:
        if own_catalogue:
            catalogue.close()
        if own_second and second is not None:
            second.close()
    issues = [{"ref_id": r["observation_id"], "reason": i} for r in resolutions for i in r["issues"]]
    path = root / config.get("resolutions_filename", "resolutions.json")
    write_json(path, envelope(artifact["run_id"], "resolve", {"resolutions": resolutions},
                             status="partial" if issues else "ok", issues=issues, metrics=metrics,
                             inputs={'readings_sha256': __import__('hashlib').sha256(Path(readings_path).read_bytes()).hexdigest()},
                             context=dict(artifact.get("context", {})),
                             **({"fixture": True} if artifact.get("fixture") else {})))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-name", default="resolutions.json")
    parser.add_argument("--online", action="store_true", help="Allow catalogue HTTP; otherwise cache only")
    args = parser.parse_args()
    print(resolve(args.readings, args.run_dir, {"catalogue_online": args.online, "resolutions_filename": args.output_name}))


if __name__ == "__main__":
    main()
