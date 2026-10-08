"""Google Books as a second catalogue (providers/googlebooks.py) and the query normaliser (pipeline/resolve.py). Mocked HTTP only: nothing here reaches Google
or Open Library, and nothing here says how many real spines would be identified."""
import json

import httpx
import pytest

from pipeline import resolve as R
from providers import googlebooks as GB


def volume(vid, title, authors, lang="es", dims=None):
    vi = {"title": title, "authors": authors, "language": lang, "pageCount": 300}
    if dims:
        vi["dimensions"] = dims
    return {"id": vid, "volumeInfo": vi}


def gb_client(tmp_path, answers, seen=None, status=200, key=None, records=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": {"code": status}})
        if "/volumes/" in request.url.path:
            vid = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=(records or {}).get(vid, {"id": vid, "volumeInfo": {}}))
        q = request.url.params.get("q")
        items = next((v for k, v in answers.items() if k in q), [])
        return httpx.Response(200, json={"totalItems": len(items), "items": items})
    return GB.GoogleBooks(tmp_path / "gb", online=True, transport=httpx.MockTransport(handler), api_key=key)


class Empty:
    """An Open Library stand-in that answers and finds nothing (complete-title and fragment paths)."""
    requests = 0
    metrics = []

    def search(self, title, author=None, limit=20, language=None):
        return {"payload": {"docs": [], "numFound": 0}, "url": "https://openlibrary.org/search.json?title=" + title, "retrieved_at": "t"}

    def search_fragments(self, terms, **kw):
        return {"candidates": [], "queries": [{"q": " ".join(terms), "url": "u", "retrieved_at": "t"}], "errors": [], "total": 0, "returned": 0, "limit": 10}

    def edition_by_isbn(self, isbn):
        raise RuntimeError("no isbn")


def reading(text, title=None, author=None, lang="es", complete=True, oid="f00001-o001"):
    return {"observation_id": oid, "visible_text": text, "title_text": title, "author_text": author, "language_code": lang, "title_complete": complete,
            "provenance": {"reader": "test"}}


# ------------------------------------------------------------------------------------------ provider
def test_search_answers_in_open_library_shape_with_citations_and_dimensions(tmp_path):
    seen = []
    gb = gb_client(tmp_path, {"Crimen": [volume("AbCdEfGh1234", "Crimen y castigo", ["Fiódor Dostoievski"], dims={"height": "21.00 cm", "thickness": "3.1 cm"})]}, seen, key="K")
    res = gb.search("Crimen y castigo", "Dostoievski", language="es")
    (doc,) = res["payload"]["docs"]
    assert doc["key"] == "gb:AbCdEfGh1234" and doc["url"] == "https://books.google.com/books?id=AbCdEfGh1234" and doc["author_name"] == ["Fiódor Dostoievski"]
    assert doc["editions"]["docs"][0]["language"] == ["spa"] and doc["google_books"]["dimensions"]["height_cm"] == 21.0 and doc["google_books"]["dimensions"]["thickness_cm"] == 3.1
    assert res["payload"]["numFound"] == 1 and res["payload"]["total_estimate"] == 1 and res["retrieved_at"]
    q = seen[0].url.params
    assert q["q"] == "Crimen y castigo Dostoievski" and q["langRestrict"] == "es" and q["key"] == "K", "plain terms: field operators answer 0 on this key's project"
    assert "key=" not in res["url"], "the API key never appears in a stored URL"
    gb.search("Crimen y castigo", "Dostoievski", language="es")
    assert len(seen) == 1 and gb.metrics[-1]["cache_hit"], "the second identical query is served from the disk cache"


def test_quota_and_offline_errors_are_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_BOOKS_API_KEY", raising=False)   # the anonymous message must not depend on the developer's .env
    gb = gb_client(tmp_path, {}, status=429)
    with pytest.raises(RuntimeError, match="google_books_quota_exceeded_anonymous_set_GOOGLE_BOOKS_API_KEY"):
        gb.search("Rayuela")
    off = GB.GoogleBooks(tmp_path / "off", online=False)
    with pytest.raises(RuntimeError, match="catalogue_offline_cache_miss"):
        off.search("Rayuela")


def test_dimension_parsing_needs_a_height():
    assert GB.parse_dimensions({"height": "0.9 in"})["height_cm"] == 2.29
    assert GB.parse_dimensions({"width": "15 cm"}) is None and GB.parse_dimensions(None) is None


# ------------------------------------------------------------------------------------------ query normaliser
def test_clean_query_strips_ocr_noise_but_keeps_spanish_one_letter_words():
    assert R.clean_query("[arquitectura |") == "arquitectura"
    assert R.clean_query("I Crimen y castigo ]") == "Crimen y castigo"
    assert R.clean_query("Ic_ MARCA PAIS") == "Ic MARCA PAIS"


def test_query_variants_restore_the_leading_article_only_as_a_query():
    assert R.query_variants("VIDA DE PI", "es") == ["VIDA DE PI", "la VIDA DE PI"]
    assert R.query_variants("CRONICAS MARCIANAS", "es") == ["CRONICAS MARCIANAS", "las CRONICAS MARCIANAS"]
    assert R.query_variants("Little Women", "en") == ["Little Women", "the Little Women"]
    assert R.query_variants("La sombra del viento", "es") == ["La sombra del viento"], "a title that already starts with an article gets no variant"
    assert R.query_variants("Rayuela", "es") == ["Rayuela"], "one-word titles get no article variant"


def test_the_literal_title_rule_tolerates_only_one_unprinted_leading_article_and_folds_accents_only_on_request():
    assert R.title_matches("La vida de Pi", R.normalize_title("VIDA DE PI")) and R.title_matches("Un humanista en la España de Franco", R.normalize("UN HUMANISTA EN LA ESPANA DE FRANCO"), folded=True)
    assert not R.title_matches("Un humanista en la España de Franco", R.normalize_title("UN HUMANISTA EN LA ESPANA DE FRANCO")), "accents must agree unless folded"
    assert not R.title_matches("La vida de Pi", R.normalize_title("VIDA")), "every catalogue word except one leading article must be visible"
    assert not R.title_matches("Vida de Pi", R.normalize_title("LA VIDA DE PI")), "a printed article the catalogue lacks is not tolerated"
    assert not R.title_matches("El amante", R.normalize_title("LA AMANTE"))


def test_an_exact_accented_match_is_not_made_ambiguous_by_an_unaccented_other_work():
    class OL(Empty):
        def search(self, title, author=None, limit=20, language=None):
            docs = [{"key": "/works/OL1W", "title": "Ébora", "author_name": ["Ana Lopez"]}, {"key": "/works/OL2W", "title": "Ebora", "author_name": ["Bruno Lopez"]}]
            return {"payload": {"docs": docs, "numFound": 2}, "url": "u", "retrieved_at": "t"}
    exact = R._resolve_complete_title(reading("ÉBORA\nLOPEZ", "ÉBORA", "LOPEZ"), OL())
    assert exact["decision"] == "accepted" and exact["work"] == "/works/OL1W" and not exact["candidates"][0]["accent_insensitive"]
    folded = R._resolve_complete_title(reading("EBORA\nLOPEZ", "EBORA", "LOPEZ"), OL())
    assert folded["decision"] == "accepted" and folded["work"] == "/works/OL2W", "the unaccented reading matches the unaccented record exactly"
    ocr = R._resolve_complete_title(reading("ESPANA\nLOPEZ", "ESPANA", "LOPEZ"), type("O", (Empty,), {"search": lambda self, *a, **k: {"payload": {"docs": [
        {"key": "/works/OL3W", "title": "España", "author_name": ["Ana Lopez"]}], "numFound": 1}, "url": "u", "retrieved_at": "t"}})())
    assert ocr["decision"] == "accepted" and ocr["candidates"][0]["accent_insensitive"], "an OCR-dropped accent matches only when nothing matched with accents"


def test_the_known_vida_de_miss_resolves_through_the_article_variant_with_the_reading_as_the_title(tmp_path):
    class OL(Empty):
        def __init__(self):
            self.asked = []

        def search(self, title, author=None, limit=20, language=None):
            self.asked.append(title)
            docs = [{"key": "/works/OL123W", "title": "La vida de Pi", "author_name": ["Yann Martel"]}] if title.lower().startswith("la ") else []
            return {"payload": {"docs": docs, "numFound": len(docs)}, "url": "https://openlibrary.org/search.json?title=" + title, "retrieved_at": "t"}
    ol = OL()
    res = R.resolve_one(reading("VIDA DE PI\nYANN MARTEL", "VIDA DE PI", "MARTEL"), ol)
    assert ol.asked == ["VIDA DE PI", "la VIDA DE PI"]
    assert res["decision"] == "accepted" and res["title"] == "VIDA DE PI" and res["catalogue_title"] == "La vida de Pi" and res["work"] == "/works/OL123W"
    assert res["catalogue"] == "open_library" and res["matching_evidence"][0]["query_variant"] == "la VIDA DE PI"


# ------------------------------------------------------------------------------------------ the fallback
def test_google_books_is_asked_only_when_open_library_has_no_eligible_candidate_and_its_hit_passes_the_same_rule(tmp_path):
    gb = gb_client(tmp_path, {"humanista": [volume("HuMaNiSt1234", "Un humanista en la España de Franco", ["Javier Varela"])]},
                   records={"HuMaNiSt1234": volume("HuMaNiSt1234", "Un humanista en la España de Franco", ["Javier Varela"], dims={"height": "23 cm"})})
    res = R.resolve_one(reading("UN HUMANISTA EN LA ESPANA DE FRANCO\nVARELA", "UN HUMANISTA EN LA ESPANA DE FRANCO", "VARELA"), Empty(), fragments=True, second=gb)
    assert res["decision"] == "accepted" and res["catalogue"] == "google_books" and res["work"] == "gb:HuMaNiSt1234"
    assert res["source_url"] == "https://books.google.com/books?id=HuMaNiSt1234"
    assert res["google_books"]["dimensions"]["height_cm"] == 23.0 and res["google_books"]["dimensions"]["source_url"] == res["source_url"], "dimensions from the volume record, cited"
    assert any(i.startswith("resolved_by_second_catalogue_google_books") for i in res["issues"]) and res["open_library_attempt"]["decision"] == "no_match"


def test_a_google_books_title_with_an_unseen_word_is_refused_like_any_catalogue_title(tmp_path):
    gb = gb_client(tmp_path, {"humanista": [volume("HuMaNiSt1234", "Un humanista en la España de Franco y otros ensayos", ["Javier Varela"])]})
    res = R.resolve_one(reading("UN HUMANISTA EN LA ESPANA DE FRANCO\nVARELA", "UN HUMANISTA EN LA ESPANA DE FRANCO", "VARELA"), Empty(), fragments=True, second=gb)
    assert res["decision"] != "accepted" and res["second_catalogue"]["catalogue"] == "google_books"


def test_an_ambiguous_or_unavailable_open_library_answer_is_never_overridden_by_google_books(tmp_path):
    seen = []
    gb = gb_client(tmp_path, {"Rayuela": [volume("RaYuElA12345", "Rayuela", ["Julio Cortázar"])]}, seen)

    class Down(Empty):
        def search(self, *a, **k):
            raise RuntimeError("timeout")
    res = R.resolve_one(reading("RAYUELA CORTAZAR", "RAYUELA", "CORTAZAR"), Down(), fragments=True, second=gb)
    assert res["decision"] != "accepted" and seen == [], "an unavailable Open Library is not 'no eligible candidate'"


def test_a_quota_error_disables_the_second_catalogue_for_the_rest_of_the_run(tmp_path):
    seen = []
    gb = gb_client(tmp_path, {}, seen, status=429)
    for i in range(3):
        R.resolve_one(reading("UN HUMANISTA EN LA ESPANA", "UN HUMANISTA EN LA ESPANA", "VARELA", oid=f"f00001-o00{i}"), Empty(), fragments=True, second=gb)
    assert gb.disabled and len(seen) == 1


def test_the_second_catalogue_choice_follows_the_environment(tmp_path):
    assert R.second_catalogue(tmp_path, online=True, environ={}) is None, "no key: anonymous quota is exhausted, do not spend a request per reading"
    assert R.second_catalogue(tmp_path, online=True, environ={"GOOGLE_BOOKS_API_KEY": "k", "LIBRARY_CATALOGUE_SECOND": "none"}) is None
    gb = R.second_catalogue(tmp_path, online=True, environ={"GOOGLE_BOOKS_API_KEY": "k"})
    assert isinstance(gb, GB.GoogleBooks)
    gb.close()
    assert R.second_catalogue(tmp_path, online=False, environ={}) is None
    (tmp_path / "googlebooks").mkdir()
    off = R.second_catalogue(tmp_path, online=False, environ={})
    assert isinstance(off, GB.GoogleBooks) and off.online is False
    off.close()


# ------------------------------------------------------------------------------------------ identities attached after the sweep (pipeline/identity_attach.py)
from pipeline import contracts as C
from pipeline import identity_attach as IA
from pipeline import instances as IN
from pipeline import packet as P
from tests.test_instances import _art, _inventory, _manifest


def _run_with_packet(tmp_path, instances=True):
    man = _manifest(('f00000', 'f00001'))
    run = tmp_path / 'run'
    (run / 'frames').mkdir(parents=True)
    for f in ('f00000', 'f00001'):
        (run / 'frames' / f'{f}.jpg').write_bytes(b'x')
    C.write_json(run / C.ARTIFACTS['frames'], man)
    art = _art(['bk-f00000-o000', 'bk-f00001-o000'], run_id='run') if instances else None
    env = C.envelope('run', 'packet', P.build_data(_inventory(), man, {'run_id': 'run', **({'instances': art} if art else {})}), status='partial')
    C.write_json(run / C.ARTIFACTS['packet'], env)
    if art:
        IN.write(art, run)
    return run, man


def _files(tmp_path, accepted):
    reads = [{'observation_id': o, 'visible_text': t, 'read_confidence': 0.9} for o, t in accepted.items()]
    res = [{'observation_id': o, 'decision': 'accepted', 'title': t, 'catalogue_title': t.title(), 'author': None, 'work': w, 'source_url': u,
            'matching_evidence': [{'field': 'title'}], 'catalogue': c, 'issues': []}
           for o, (t, w, u, c) in {o: v for o, v in accepted.items()}.items()] if isinstance(next(iter(accepted.values())), tuple) else []
    rp, sp = tmp_path / 'readings.json', tmp_path / 'resolutions.json'
    C.write_json(rp, C.envelope('x', 'read', {'readings': [{'observation_id': o, 'visible_text': v[0], 'read_confidence': 0.9} for o, v in accepted.items()]}))
    C.write_json(sp, C.envelope('x', 'resolve', {'resolutions': res}))
    return rp, sp


def test_identities_attach_per_member_merge_into_the_instance_line_and_the_packet_still_verifies(tmp_path):
    run, man = _run_with_packet(tmp_path)
    ol = ('RAYUELA', '/works/OL1W', 'https://openlibrary.org/works/OL1W', 'open_library')
    rp, sp = _files(tmp_path, {'f00000-o000': ol, 'f00001-o000': ol, 'f00000-o001': ('FICCIONES', 'gb:FiCcIoNeS123', 'https://books.google.com/books?id=FiCcIoNeS123', 'google_books')})
    res = IA.attach(run, rp, sp, render=False)
    assert res['attached'] and res['books_identified_before'] == 0 and res['books_identified'] == 2 and res['groups']['identified'] == 3
    p = C.read_json(run / C.ARTIFACTS['packet'])['data']
    merged = p['books'][0]
    assert merged['member_group_ids'] == ['bk-f00000-o000', 'bk-f00001-o000'] and merged['status'] == 'identified' and merged['title'] == 'RAYUELA' and merged['catalogue'] == 'open_library'
    gb = p['books'][1]
    assert gb['status'] == 'identified' and gb['catalogue'] == 'google_books' and gb['identity_source_url'].startswith('https://books.google.com/')
    assert not [r for r in p['review_queue'] if r['code'] == 'unidentified_book' and r['ref_id'] in ('bk-f00000-o000', 'bk-f00000-o001')]
    assert P.verify_packet(C.read_json(run / C.ARTIFACTS['packet']), man, run) == [] and (run / IA.BACKUP).is_file()
    again = IA.attach(run, rp, sp, render=False)
    assert again['books_identified'] == 2 and C.read_json(run / C.ARTIFACTS['packet'])['data']['books'] == p['books'], 'idempotent'
    assert IN.attach(run, render=False)['attached'] and C.read_json(run / C.ARTIFACTS['packet'])['data']['totals']['books_identified'] == 2, 'the instance layer re-applied on top keeps them'


def test_members_with_different_works_leave_the_instance_unidentified_and_an_old_identity_is_never_lost(tmp_path):
    run, man = _run_with_packet(tmp_path)
    rp, sp = _files(tmp_path, {'f00000-o000': ('RAYUELA', '/works/OL1W', 'https://openlibrary.org/works/OL1W', 'open_library'),
                               'f00001-o000': ('FICCIONES', '/works/OL2W', 'https://openlibrary.org/works/OL2W', 'open_library')})
    IA.attach(run, rp, sp, render=False)
    merged = C.read_json(run / C.ARTIFACTS['packet'])['data']['books'][0]
    assert merged['status'] == 'unidentified' and merged['identity_decision'] == 'ambiguous'
    line = {'status': 'identified', 'title': 'X', 'identity_source_url': 'u'}
    assert IA.apply_identity(line, {'decision': 'no_match'}) == 'kept' and line['title'] == 'X'
    assert IA.apply_identity({'status': 'needs_appraisal'}, {'decision': 'accepted', 'title': 'Y', 'source_url': 'u'}) == 'skipped'
    assert IA.apply_identity({'status': 'unidentified', 'user_corrections': [{'kind': 'set_title'}]}, {'decision': 'accepted', 'title': 'Y', 'source_url': 'u'}) == 'skipped'


def test_a_complete_title_hit_in_a_capped_google_answer_is_accepted_when_unique(tmp_path):
    """Google answers every plain query with a large totalItems estimate; the returned volumes are the candidate set (else no complete-title hit could ever pass)."""
    def handler(request):
        if "/volumes/" in request.url.path:
            return httpx.Response(200, json={"id": "x", "volumeInfo": {}})
        items = [volume("RaYuElA12345", "Rayuela", ["Julio Cortázar"])] + [volume(f"OtHeR{i:07d}", f"Otro libro {i}", ["Nadie"]) for i in range(19)]
        return httpx.Response(200, json={"totalItems": 357, "items": items})
    gb = GB.GoogleBooks(tmp_path / "gb", online=True, transport=httpx.MockTransport(handler), api_key="K")
    res = R.resolve_one(reading("RAYUELA\nCORTAZAR", "RAYUELA", "CORTAZAR"), Empty(), fragments=True, second=gb)
    assert res["decision"] == "accepted" and res["work"] == "gb:RaYuElA12345" and res["catalogue_query"]["num_found"] == 20


def test_a_google_books_height_is_a_scale_a_source_and_cites_the_volume():
    from pipeline import auto_scale as AS
    books = [{"id": "bk-1", "status": "identified", "identity_source": "catalogue_match", "identity_source_url": "https://books.google.com/books?id=RaYuElA12345",
              "catalogue": "google_books", "identity_work": "gb:RaYuElA12345", "title": "RAYUELA",
              "catalogue_dimensions": {"height_cm": 21.0, "source_url": "https://books.google.com/books?id=RaYuElA12345", "retrieved_at": "t"}},
             {"id": "bk-2", "status": "identified", "identity_source": "catalogue_match", "identity_source_url": "https://books.google.com/books?id=FiCcIoNeS123",
              "catalogue": "google_books", "identity_work": "gb:FiCcIoNeS123", "title": "FICCIONES", "catalogue_dimensions": None}]
    ident = AS.identified_from_books(books)
    assert [b["group_id"] for b in ident] == ["bk-1"], "a Google Books volume without a recorded height contributes nothing"
    res = AS.scale_from_catalogue(ident, {"bk-1": {"status": "ok", "h_units": 0.42, "sightings_used": 3}}, catalogue=None, opts={**AS.DEFAULTS, "budget_s": None})
    (src,) = res["sources"]
    assert src["catalogue"]["source"] == "google_books_volume" and src["catalogue_height_cm"] == 21.0 and src["catalogue_urls"] == ["https://books.google.com/books?id=RaYuElA12345"]


def test_a_group_with_an_open_library_identity_ignores_a_google_books_identity_of_another_sighting():
    from pipeline.dedup import _identity
    ev = [{"field": "title"}]
    readings = {"o1": {"read_confidence": 0.9}, "o2": {"read_confidence": 0.9}}
    res = {"o1": {"observation_id": "o1", "decision": "accepted", "title": "The Story of Civilization I", "work": "/works/OL1073918W",
                  "source_url": "https://openlibrary.org/works/OL1073918W", "matching_evidence": ev, "catalogue": "open_library"},
           "o2": {"observation_id": "o2", "decision": "accepted", "title": "OUR ORIENTAL HERITAGE", "work": "gb:ru4LPyMAxxkC",
                  "source_url": "https://books.google.com/books?id=ru4LPyMAxxkC", "matching_evidence": ev, "catalogue": "google_books"}}
    ident, _ = _identity([{"observation_id": "o1"}, {"observation_id": "o2"}], readings, res)
    assert ident["decision"] == "accepted" and ident["work"] == "/works/OL1073918W" and ident["catalogue"] == "open_library"
    only_gb, _ = _identity([{"observation_id": "o2"}], readings, res)
    assert only_gb["decision"] == "accepted" and only_gb["catalogue"] == "google_books"
    two_ol = {**res, "o2": {**res["o1"], "observation_id": "o2", "work": "/works/OL2W", "title": "Other"}}
    assert _identity([{"observation_id": "o1"}, {"observation_id": "o2"}], readings, two_ol)[0]["decision"] == "ambiguous", "two Open Library works still conflict"
