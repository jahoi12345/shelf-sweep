"""Separate evidence-gated US/USD physical-book pricing; unknown stays null."""
from __future__ import annotations
import argparse
import copy
import os
import re
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
import time
from pipeline import contracts as C
from pipeline.resolve import isbn_value, normalize_title, fold, _EN_ONLY, _ES_CA_ONLY
from providers.book_prices import PublicBookPrices, positive_amount


def identity_key(book):
    return hashlib.sha256(json.dumps({k:book.get(k) for k in ['title','isbn','edition','identity_source']},sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def work_key_of(book):
    """Open Library work (/works/OL..W) of an accepted identity, else None."""
    m=_WORK_URL.fullmatch(book.get('identity_source_url') or '')
    return m.group(1) if m else None


def priced_work_key(book):
    """The work the pricing rule prices: an Open Library work, or a Google Books volume (gb:<id>, the second catalogue's accepted record) whose saved volume record
    serves as its edition record (pipeline.price_attach.VolumeCatalogue)."""
    g=_GB_URL.fullmatch(book.get('identity_source_url') or '')
    return work_key_of(book) or ('gb:'+g.group(1) if g else None)


def edition_url(edition_key):
    """Public URL of an edition record: Open Library /books/OL..M, or a Google Books volume (gb:<id>)."""
    key=str(edition_key or '')
    return 'https://books.google.com/books?id='+key[3:] if key.startswith('gb:') else 'https://openlibrary.org'+key


def supported_isbn(book):
    isbn=isbn_value(book.get('isbn'))
    evidence=book.get('identity_matching_evidence') or []
    if book.get('status')!='identified' or book.get('identity_source')!='catalogue_match' or not isbn or not book.get('edition'):
        return None
    valid=any(isinstance(e,dict) and e.get('field')=='edition_isbn' and isbn_value(e.get('visible'))==isbn and e.get('catalogue')==isbn
              and e.get('edition_key')==book['edition'] and e.get('source_url') and e.get('retrieved_at') for e in evidence)
    return isbn if valid else None


# market -> (currency, source hosts). US/USD lines feed the packet's money totals natively. ES/EUR and GB/GBP listings reach the totals ONLY as CONVERTED fallback lines
# (user decision 2026-10-07: no US-seller offer for the equivalent edition -> a non-US market offer converted to USD at the cited ECB rate, converted: true, native amount kept).
# The separate ES/EUR "second market" comparison line (native currency, never in the USD totals) is OFF unless LIBRARY_SECOND_MARKET=1 (see second_market_enabled).
MARKETS = {'US': ('USD', ('www.abebooks.com','www.penguinrandomhouse.com')), 'ES': ('EUR', ('www.iberlibro.com',)), 'GB': ('GBP', ('www.abebooks.co.uk',))}


def validate_offer(offer, book, run_dir=None, expected_isbn=None, market='US'):
    errors=[]
    if not isinstance(offer,dict): return ['offer must be an evidence-backed object']
    # a CONVERTED fallback line is validated where a US/USD line is (it is a USD money line of the packet), but against its own market's source and the cited FX evidence
    converted=market=='US' and offer.get('converted') is True
    if converted:
        src=offer.get('market') if offer.get('market') in CONVERTED_MARKETS else None
        currency,hosts=MARKETS[src] if src else ('?',())
        if not src: errors.append('converted line from an unsupported market')
        if offer.get('currency')!='USD' or offer.get('original_currency')!=currency or positive_amount(offer.get('original_amount')) is None or not isinstance(offer.get('fx'),dict):
            errors.append('converted line must carry the USD amount, the native amount and currency and the FX evidence')
    else:
        currency,hosts=MARKETS[market]
    if positive_amount(offer.get('amount')) is None: errors.append('amount is not positive finite money')
    for key in ('source','url','retrieved_at','condition_assumed','format','proxy_assumptions','edition_match'):
        if not offer.get(key): errors.append('missing '+key)
    if not converted and (offer.get('currency')!=currency or offer.get('market')!=market or offer.get('converted') is not False): errors.append(f'not an unconverted {market}/{currency} offer')
    if converted and not run_dir:
        from providers import fx as FX
        if FX.convert_to_usd(offer.get('original_amount'),offer.get('fx'))!=offer.get('amount'): errors.append('converted USD amount does not equal native amount x cited ECB rate')
    work_level=offer.get('price_basis')==PRICE_BASIS
    if work_level:
        # work-level line: the identity must be an accepted catalogue WORK; the ISBN is the stated equivalent edition's, never a printed one
        want=expected_isbn or offer.get('equivalent_isbn')
        # edition_match is the not-exact-shelf-edition label, or 'exact_isbn' when the caller already maps it to the source-level value (pipeline.packet does)
        if offer.get('edition_match') not in (_WL_EDITION_MATCH,'exact_isbn'): errors.append('work-level line must declare it is not an exact shelf edition')
        if book is not None and not (book.get('status')=='identified' and book.get('identity_source')=='catalogue_match' and priced_work_key(book)): errors.append('work-level price needs an accepted catalogue work identity')
    else: want=expected_isbn or supported_isbn(book)
    if not want or offer.get('isbn')!=want: errors.append('printed ISBN/edition unsupported or offer mismatched')
    if offer.get('format') not in __import__('providers.book_prices',fromlist=['PHYSICAL']).PHYSICAL: errors.append('not a physical format')
    parsed_url=urlparse(offer.get('url') or '')
    if parsed_url.scheme!='https' or parsed_url.hostname not in hosts: errors.append('unsupported source URL')
    try:
        retrieved=datetime.fromisoformat(offer.get('retrieved_at','').replace('Z','+00:00'))
        if retrieved.tzinfo is None or retrieved>datetime.now(timezone.utc):errors.append('invalid retrieval time')
    except (ValueError,TypeError):errors.append('invalid retrieval time')
    ev=offer.get('evidence') or {}
    if not ev.get('sha256') or not ev.get('raw_path') or ev.get('retrieved_at')!=offer.get('retrieved_at') or ev.get('url')!=offer.get('url') or ev.get('http_status')!=200: errors.append('missing consistent HTTP evidence')
    if run_dir and ev.get('raw_path'):
        path=(Path(run_dir)/ev['raw_path']).resolve()
        if not path.is_relative_to(Path(run_dir).resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=ev.get('sha256'): errors.append('raw evidence missing, escaped or hash mismatch')
        elif path.is_file():
            from providers.book_prices import parse_jsonld, parse_abebooks
            content=path.read_text()
            parsed=(parse_abebooks(content,offer.get('isbn'),offer.get('url','')) if urlparse(offer.get('url','')).hostname in ('www.abebooks.com','www.iberlibro.com','www.abebooks.co.uk') else parse_jsonld(content,offer.get('isbn'),offer.get('url','')))
            fields=('kind','isbn','format','amount','condition_assumed','edition_match','currency','market')
            compared=dict(offer,edition_match='exact_isbn') if work_level else offer   # the listing itself matches the equivalent ISBN exactly
            if converted: compared=dict(compared,amount=offer.get('original_amount'),currency=offer.get('original_currency'))   # the raw listing is in the seller's currency
            if not any(all(row.get(k)==compared.get(k) for k in fields) for row in parsed):errors.append('normalized offer does not match raw source bytes')
        if converted:
            from providers import fx as FX
            fxp=(Path(run_dir)/str((offer.get('fx') or {}).get('raw_path') or '')).resolve()
            ok=bool((offer.get('fx') or {}).get('raw_path')) and fxp.is_relative_to(Path(run_dir).resolve()) and fxp.is_file()
            errors+=FX.check_conversion(offer,fxp.read_bytes() if ok else None)
    return errors


def price_books(books, run_dir, config=None, provider=None):
    config=config or {}; root=Path(run_dir).resolve(); start=time.perf_counter()
    provider=provider or PublicBookPrices(root/'pricing_evidence',online=config.get('pricing_online',False),cache_dir=config.get('price_cache_dir'))
    rows=[]
    for book in books:
        row={'ref_id':book['id'],'identity_key':identity_key(book),'replacement_cost':None,'used_value':None,'issues':[]}
        isbn=supported_isbn(book)
        if book.get('status')=='needs_appraisal':row['issues'].append('appraisal_required_no_automatic_price')
        elif config.get('country','US')!='US' or config.get('currency','USD')!='USD':row['issues'].append('market_provider_unavailable; second-market comparison pending')
        elif not isbn:row['issues'].append('no supported printed ISBN/physical edition; work-level equivalent not justified')
        else:
            offers,failures=provider.offers(isbn,(config.get('price_source_urls') or {}).get(isbn))
            row['source_failures']=failures
            valid=[]
            for offer in offers:
                offer=dict(offer);offer['evidence']=dict(offer.get('evidence') or {})
                path=offer['evidence'].get('raw_path')
                if path and Path(path).is_absolute() and Path(path).is_relative_to(root):offer['evidence']['raw_path']=str(Path(path).relative_to(root))
                errors=validate_offer(offer,book,root)
                if errors:row['issues'].append({'rejected_offer':offer.get('url'),'reasons':errors})
                else:valid.append(offer)
            if any(Decimal(str(o['amount']))>=Decimal(str(config.get('appraisal_threshold_usd',250))) for o in valid):
                row['needs_appraisal']=True;row['issues'].append('source_offer_above_appraisal_threshold')
            else:
                for kind in ('replacement_cost','used_value'):
                    matching=[o for o in valid if o.get('kind')==kind]
                    if matching: row[kind]=min(matching,key=lambda x:Decimal(str(x['amount'])))
                    else:row['issues'].append(kind+'_source_unavailable')
                row['selection_policy']='lowest validated exact-ISBN asking offer per condition class; not a realized sale valuation'
        rows.append(row)
    return C.envelope(config.get('run_id',root.name),'price',{'prices':rows,'market':'US','currency':'USD',
        'second_market_comparison':{'status':'pending','required_books':10,'original_currencies_and_dated_fx_required':True}},
        status='partial' if any(r['issues'] for r in rows) or not rows else 'ok',
        metrics={'elapsed_s':round(time.perf_counter()-start,3),'replacement_lines':sum(r['replacement_cost'] is not None for r in rows),
                 'used_lines':sum(r['used_value'] is not None for r in rows),'actual_API_cost_usd':0,
                 'cost_provenance':'Public HTTP/cached evidence, no paid model/account'})


def price(inventory_path, manifest_path, run_dir, config=None):
    from pipeline.packet import build_data
    config=config or {}; inv=C.read_json(inventory_path) if inventory_path else None
    data=build_data(inv,C.read_json(manifest_path),config)
    artifact=price_books_with_work_level(data['books'],run_dir,config)
    return C.write_json(Path(run_dir)/'prices.json',artifact)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# WORK-LEVEL EQUIVALENT pricing (policy approved by the user; exact-edition ISBN cannot be read from spine video and ISBN scanning is out of scope).
# A resolved WORK (title+author, catalogue /works/OL..W) is priced through ONE stated equivalent edition. The result is NEVER an exact-edition price:
# every line carries price_basis="work_level_equivalent" and the chosen ISBN, the rule and the alternatives considered.
# ---------------------------------------------------------------------------------------------------------------------------------------------
PRICE_BASIS = 'work_level_equivalent'
_WL_EDITION_MATCH = 'work_level_equivalent_edition_not_exact_shelf_edition'
_WORK_URL = re.compile(r'https://openlibrary\.org(/works/OL\d+W)')
_GB_URL = re.compile(r'https://books\.google\.com/books\?id=([A-Za-z0-9_-]{6,20})')
EQUIVALENT_RULE = ('same language as the spine text (never switched; editions without a recorded language are not used) + same physical-format family (target paperback; '
                   'unknown format only if listings confirm it) + most recent publish date among Open Library edition records with a checksum-valid ISBN, preferring an edition '
                   'that has a NEW US listing (in print); US sellers, USD, shipping/tax excluded, asking prices; lowest new + lowest used')
_OL_LANG = {'es': 'spa', 'en': 'eng'}
_ABE_LANG = {'es': 'SPA', 'en': 'ENG'}
_FAMILY_TO_FORMAT = {'paperback': 'Paperback', 'hardcover': 'Hardcover'}
ABE_URL = 'https://www.abebooks.com/servlet/SearchResults?isbn={isbn}&sortby=2'
ABE_URL_NEW = 'https://www.abebooks.com/servlet/SearchResults?isbn={isbn}&cond=new&sortby=2'
_MONTHS = {m: i + 1 for i, m in enumerate('jan feb mar apr may jun jul aug sep oct nov dec'.split())}


def format_family(physical_format):
    f = (physical_format or '').casefold()
    if any(x in f for x in ('ebook', 'e-book', 'audio', 'cd', 'kindle', 'pdf', 'dvd', 'digital')): return 'excluded'
    if any(x in f for x in ('hardcover', 'hardback', 'hard cover', 'cloth', 'library binding')): return 'hardcover'
    if any(x in f for x in ('paper', 'softcover', 'soft cover', 'mass market', 'pocket', 'trade')): return 'paperback'
    return 'unknown'


def publish_sort_key(date_text, this_year=None):
    """(year, month) from Open Library free-text dates ('2018-12-05', 'Mar 19, 2014', '2012'); (0, 0) when unknown or in the future."""
    this_year = this_year or datetime.now(timezone.utc).year
    text = str(date_text or '')
    years = [int(y) for y in re.findall(r'(?<!\d)(1[4-9]\d\d|20\d\d)(?!\d)', text)]
    if not years or max(years) > this_year: return (0, 0)
    month = next((_MONTHS[w[:3]] for w in re.findall(r'[A-Za-z]+', text.casefold()) if w[:3] in _MONTHS), 0)
    iso = re.search(r'(?<!\d)\d{4}-(\d\d)', text)
    return (max(years), int(iso.group(1)) if iso else month)


def detect_language(text):
    """'es' / 'en' from function words of the visible spine text only (shared words are ignored); None when the text does not decide."""
    toks = set(fold(text or '').split())
    en, es = bool(toks & _EN_ONLY), bool(toks & _ES_CA_ONLY) or any(c in (text or '').casefold() for c in 'ñ¿¡')
    return 'en' if en and not es else 'es' if es and not en else None


def edition_languages(entry):
    return {(x.get('key') or '').rsplit('/', 1)[-1] for x in entry.get('languages') or [] if isinstance(x, dict)}


def isbn10_to_13(isbn10):
    core = '978' + isbn10[:9]
    return core + str((10 - sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(core)) % 10) % 10)


def edition_isbn(entry):
    """First checksum-valid ISBN of an edition record, canonicalised to ISBN-13 (an ISBN-10 is converted) so that two records of one book count once."""
    for field in ('isbn_13', 'isbn_10'):
        for raw in entry.get(field) or []:
            value = isbn_value(raw)
            if value: return (value if len(value) == 13 else isbn10_to_13(value)), field
    return None, None


_GENERIC_PUBLISHER = {'editorial', 'editorial', 'ediciones', 'edicions', 'edicion', 'books', 'book', 'press', 'publishing', 'publishers', 'publisher', 'group', 'house', 'inc', 'ltd', 'llc', 'sa', 'sl',
                      'the', 'de', 'del', 'la', 'el', 'los', 'las', 'and', 'y', 'co', 'company', 'libros', 'grupo', 'verlag'}


def language_from_publisher(visible_text, title, entries):
    """A publisher printed on the spine whose DISTINCTIVE name tokens all appear in the spine text and equal an edition record's publisher; the languages of the edition records of
    that publisher must be exactly one of spa/eng. -> (code, publisher name) or None. Title words never count as publisher evidence; ambiguity -> None (no guessing)."""
    visible, title_tokens = set(fold(visible_text or '').split()), set(fold(title or '').split())
    found = {}
    for e in entries:
        langs = edition_languages(e) & set(_OL_LANG.values())
        for name in e.get('publishers') or []:
            distinct = [t for t in fold(name).split() if t not in _GENERIC_PUBLISHER]
            if distinct and sum(map(len, distinct)) >= 4 and all(t in visible and t not in title_tokens for t in distinct):
                found.setdefault(name, set()).update(langs)
    langs = set().union(*found.values()) if found else set()
    if len(langs) != 1: return None
    return next(k for k, v in _OL_LANG.items() if v == next(iter(langs))), sorted(found)[0]


def resolve_language(visible_text, language_code, title, entries):
    """-> (code or None, source). Order: spine-text function words, a spine publisher matching edition records, the languages of the catalogue editions whose own title equals
    the resolved title (the printed title string is itself evidence of the language), and only then the reading's language_code (a model reading never overrides a
    spine- or catalogue-determined language); anything ambiguous or unknown -> (None, reason): the price stays null."""
    code = detect_language(visible_text)
    if code: return code, 'spine_text_function_words'
    code = language_from_publisher(visible_text, title, entries)
    if code: return code[0], 'spine_publisher_matches_edition_record_publisher:' + code[1]
    wanted = normalize_title(title)
    langs = set()
    for e in entries:
        if wanted and normalize_title(e.get('title')) == wanted: langs |= edition_languages(e) & set(_OL_LANG.values())
    if len(langs) == 1:
        return next(k for k, v in _OL_LANG.items() if v == next(iter(langs))), 'catalogue_editions_titled_exactly_like_the_spine_title'
    if language_code in _OL_LANG: return language_code, 'reading_language_code'
    return None, 'language_not_determined' + (':editions_with_that_title_in_both_languages' if len(langs) > 1 else '')


def reading_language_vote(readings):
    """Majority language of a group's spine readings (the reader's language_code + its quoted language_evidence) -> {code, votes, readings, evidence} or None.
    Only es/en count; a tie or no coded reading -> None. A model reading: used ONLY when the spine words and the catalogue do not decide (see price_work)."""
    rows = [r for r in readings or [] if isinstance(r, dict)]
    votes = {}
    for r in rows:
        if r.get('language_code') in _OL_LANG: votes[r['language_code']] = votes.get(r['language_code'], 0) + 1
    if not votes: return None
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]: return None
    code = ranked[0][0]
    evidence = [{'observation_id': r.get('observation_id'), 'language_code': code, 'language_evidence': r.get('language_evidence')} for r in rows if r.get('language_code') == code][:5]
    return {'code': code, 'votes': votes, 'readings': len(rows), 'evidence': evidence}


def listing_language_of_titled_editions(entries, title, provider, max_isbns=3):
    """Last resort when no spine function words and no recorded edition language decide: the AbeBooks listing language (ENG/SPA) of the checksum-valid editions whose own title equals
    the resolved spine title. Exactly one language across them -> that language, else None."""
    wanted, seen, langs = normalize_title(title), [], set()
    for e in entries:
        isbn, _ = edition_isbn(e)
        if wanted and isbn and normalize_title(e.get('title')) == wanted and isbn not in seen: seen.append(isbn)
    for isbn in seen[:max_isbns]:
        found, _ = provider.offers(isbn, [ABE_URL.format(isbn=isbn)])
        langs |= {o.get('language') for o in found if o.get('isbn') == isbn and o.get('language')}
    langs &= set(_ABE_LANG.values())
    return next((k for k, v in _ABE_LANG.items() if v in langs), None) if len(langs) == 1 else None


def rank_equivalent_editions(entries, language, target_family='paperback', this_year=None, unrecorded_ok=False):
    """-> (candidates best first, rejected). Pure rule, no network. Candidate = {edition_key, isbn, isbn_field, publish_date, family, languages, publisher, title}."""
    wanted, cands, rejected, seen = _OL_LANG[language], [], [], set()
    for e in entries:
        key, family = e.get('key'), format_family(e.get('physical_format'))
        info = {'edition_key': key, 'publish_date': e.get('publish_date'), 'physical_format': e.get('physical_format'), 'languages': sorted(edition_languages(e))}
        isbn, field = edition_isbn(e)
        if not isbn: rejected.append({**info, 'reason': 'no_checksum_valid_isbn'}); continue
        langs = edition_languages(e)
        if langs and wanted not in langs: rejected.append({**info, 'isbn': isbn, 'reason': 'language_mismatch_never_switched'}); continue
        if not langs and not unrecorded_ok: rejected.append({**info, 'isbn': isbn, 'reason': 'edition_language_not_recorded'}); continue
        if family == 'excluded' or (family not in (target_family, 'unknown')): rejected.append({**info, 'isbn': isbn, 'reason': 'format_family_' + family}); continue
        if isbn in seen: continue
        seen.add(isbn)
        cands.append({**info, 'isbn': isbn, 'isbn_field': field, 'family': family, 'publisher': (e.get('publishers') or [None])[0], 'title': e.get('title'),
                      'language_basis': 'catalogue_edition_record' if langs else 'abebooks_listing_language_only (edition language not recorded)',
                      '_rank': (bool(langs), family == target_family, publish_sort_key(e.get('publish_date'), this_year))})
    cands.sort(key=lambda c: c['_rank'], reverse=True)
    for c in cands: c.pop('_rank')
    return cands, rejected


def _line(offer, root, isbn):
    line = {k: v for k, v in offer.items() if k not in ('raw_offer',)}
    line['evidence'] = dict(offer.get('evidence') or {})
    path = line['evidence'].get('raw_path')
    if path and Path(path).is_absolute() and Path(path).is_relative_to(root): line['evidence']['raw_path'] = str(Path(path).relative_to(root))
    line.update(price_basis=PRICE_BASIS, equivalent_isbn=isbn, edition_match=_WL_EDITION_MATCH)
    return line


PAGE_SIZE = 30      # listings on one AbeBooks result page


def market_offers(provider, isbn, urls):
    """The all-conditions page first; the new-only page only when the first page was not parsed or was full (PAGE_SIZE listings): a page with fewer listings already
    shows every listing of the ISBN, new copies included, so the second request would add nothing."""
    found, failures = provider.offers(isbn, [urls[0]])
    seen = (getattr(provider, 'observed', None) or {}).get((urlparse(urls[0]).hostname, isbn))
    if len(urls) > 1 and not (seen and seen['listings'] < PAGE_SIZE):
        more, more_fail = provider.offers(isbn, list(urls[1:]))
        found, failures = found + more, failures + more_fail
    return found, failures


def _evaluate_isbn(cand, language, family, root, provider, config):
    """Fetch (cache first) the AbeBooks pages of one candidate ISBN and keep only validated, language/format-consistent, below-appraisal US/USD listings."""
    isbn = cand['isbn']
    found, failures = market_offers(provider, isbn, [ABE_URL.format(isbn=isbn), ABE_URL_NEW.format(isbn=isbn)])
    valid, rejects = [], []
    for offer in found:
        offer = dict(offer); offer['evidence'] = dict(offer.get('evidence') or {})
        path = offer['evidence'].get('raw_path')
        if path and Path(path).is_absolute() and Path(path).is_relative_to(root): offer['evidence']['raw_path'] = str(Path(path).relative_to(root))
        errors = validate_offer(offer, None, root, expected_isbn=isbn)
        if offer.get('language') != _ABE_LANG[language]: errors.append('listing_language_not_' + language)
        if offer.get('format') != _FAMILY_TO_FORMAT[family]: errors.append('listing_format_not_' + family)
        if errors: rejects.append({'listing_id': offer.get('listing_id'), 'reasons': errors})
        else: valid.append(offer)
    limit = Decimal(str(config.get('appraisal_threshold_usd', 250)))
    appraisal = any(Decimal(str(o['amount'])) >= limit for o in valid)
    valid = [o for o in valid if Decimal(str(o['amount'])) < limit]
    best = {kind: min((o for o in valid if o['kind'] == kind), key=lambda x: Decimal(str(x['amount'])), default=None) for kind in ('replacement_cost', 'used_value')}
    summary = {**cand, 'listings_valid': len(valid), 'listings_rejected': len(rejects), 'has_new': best['replacement_cost'] is not None, 'has_used': best['used_value'] is not None,
               'above_appraisal_threshold_listings_excluded': appraisal, 'source_failures': failures}
    return summary, best


# ---------------------------------------------------------------------------------------------------------------------------------------------
# SECOND MARKET (Spain, native EUR). A second-market comparison that keeps source currency and FX evidence. Spanish-language works are mostly not sold by
# US sellers, so their US/USD line stays null; the Spanish marketplace (Iberlibro = AbeBooks Spain) lists them from sellers in Spain in EUR. The line is the SAME work-level
# equivalent policy (equivalent Spanish edition ISBN, paperback family, lowest new / lowest used asking price, shipping excluded) but in the seller's own currency: the amount
# is NEVER converted into USD silently, never summed into the USD totals; a USD figure is shown only next to it with the dated ECB reference rate as cited evidence.
# ---------------------------------------------------------------------------------------------------------------------------------------------
SECOND_MARKET = 'ES'


def second_market_enabled(config=None, environ=None):
    """The Spain/EUR second market is OFF by default (user decision 2026-10-06: USD only). An explicit config['second_market'] wins, else LIBRARY_SECOND_MARKET=1 switches it on."""
    if config and 'second_market' in config: return bool(config['second_market'])
    return (os.environ if environ is None else environ).get('LIBRARY_SECOND_MARKET', '0') == '1'
ES_URL = 'https://www.iberlibro.com/servlet/SearchResults?isbn={isbn}&sortby=2'
ES_URL_NEW = 'https://www.iberlibro.com/servlet/SearchResults?isbn={isbn}&cond=new&sortby=2'
SECOND_MARKET_NOTE = ('Second market (Spain): native EUR asking prices from sellers in Spain on Iberlibro, work-level equivalent edition (ISBN shown), shipping/tax excluded. '
                      'The EUR amount is the price; the USD figure is a labelled conversion with the dated ECB reference rate, not a US price, and is never added to the USD totals.')


def _evaluate_isbn_es(cand, language, family, root, provider, config):
    isbn = cand['isbn']
    found, failures = provider.offers(isbn, [ES_URL.format(isbn=isbn), ES_URL_NEW.format(isbn=isbn)])
    valid, rejects = [], []
    for offer in found:
        offer = dict(offer); offer['evidence'] = dict(offer.get('evidence') or {})
        path = offer['evidence'].get('raw_path')
        if path and Path(path).is_absolute() and Path(path).is_relative_to(root): offer['evidence']['raw_path'] = str(Path(path).relative_to(root))
        errors = validate_offer(offer, None, root, expected_isbn=isbn, market='ES')
        if offer.get('language') != _ABE_LANG[language]: errors.append('listing_language_not_' + language)
        if offer.get('format') != _FAMILY_TO_FORMAT[family]: errors.append('listing_format_not_' + family)
        if errors: rejects.append({'listing_id': offer.get('listing_id'), 'reasons': errors})
        else: valid.append(offer)
    limit = Decimal(str(config.get('appraisal_threshold_usd', 250)))
    appraisal = any(Decimal(str(o['amount'])) >= limit for o in valid)
    valid = [o for o in valid if Decimal(str(o['amount'])) < limit]
    best = {kind: min((o for o in valid if o['kind'] == kind), key=lambda x: Decimal(str(x['amount'])), default=None) for kind in ('replacement_cost', 'used_value')}
    return {**cand, 'listings_valid': len(valid), 'listings_rejected': len(rejects), 'has_new': best['replacement_cost'] is not None, 'has_used': best['used_value'] is not None,
            'above_appraisal_threshold_listings_excluded': appraisal, 'source_failures': failures}, best


def price_second_market(row, cands, language, family, root, provider, config, deadline=None):
    """Attach row['second_market'] (ES / EUR native) for a Spanish-language work. Never touches the US lines. Uses the already ranked equivalent-edition candidates."""
    sm = {'market': SECOND_MARKET, 'currency': 'EUR', 'source': 'Iberlibro (AbeBooks Spain)', 'replacement_cost': None, 'used_value': None, 'issues': [], 'note': SECOND_MARKET_NOTE,
          'equivalent_edition': None, 'alternatives_considered': [], 'fx': None}
    row['second_market'] = sm
    if language != 'es': sm['issues'].append('second_market_only_for_spanish_language_works'); return sm
    tried = []
    for cand in cands[:config.get('max_isbn_tries_second_market', 4)]:
        if deadline is not None and time.monotonic() >= deadline: sm['issues'].append('budget'); break
        tried.append(_evaluate_isbn_es(cand, language, family, root, provider, config))
        if tried[-1][0]['has_new']: break
    sm['alternatives_considered'] = [t[0] for t in tried]
    pick = next((t for t in tried if t[0]['has_new']), None) or next((t for t in tried if t[0]['has_used']), None)
    if not pick: sm['issues'].append('no_validated_es_eur_listing_for_the_equivalent_editions_tried'); return sm
    summary, best = pick
    sm['equivalent_edition'] = {k: summary[k] for k in ('isbn', 'isbn_field', 'edition_key', 'publish_date', 'physical_format', 'family', 'languages', 'language_basis', 'publisher', 'title')}
    sm['equivalent_edition']['source_url'] = edition_url(summary['edition_key'])
    from providers import fx as FX
    fx = FX.eur_usd(root / 'pricing_evidence', online=config.get('pricing_online', False), cache_dir=Path(config.get('price_cache_dir') or root / 'cache') / 'fx')
    if fx: fx = {**fx, 'raw_path': 'pricing_evidence/' + fx['raw_path']}
    sm['fx'] = fx
    for kind in ('replacement_cost', 'used_value'):
        if not best[kind]: sm['issues'].append(kind + '_source_unavailable'); continue
        line = _line(best[kind], root, summary['isbn'])
        line['original_currency'], line['original_amount'] = 'EUR', line['amount']
        line['usd_equivalent'] = FX.to_usd(line['amount'], fx)
        line['usd_equivalent_basis'] = (f"native EUR x ECB reference rate {fx['rate']} of {fx['rate_date']} (not a US price)" if fx else 'no FX evidence retrieved: no USD figure')
        line['fx'] = fx
        sm[kind] = line
    return sm


# ---------------------------------------------------------------------------------------------------------------------------------------------
# CONVERTED FALLBACK (user decision 2026-10-07). US offers always take precedence. Only when no US seller offers the equivalent edition (no US pick at all, or the US pick
# lacks a condition class) is a NON-US market offer used: Iberlibro (sellers in Spain, EUR) and AbeBooks UK (sellers in the UK, GBP), the same equivalent-edition rule
# (language, format family, exact ISBN, lowest new / lowest used asking price, shipping excluded). The line's amount is USD at the ECB reference rate of the retrieval
# (rate, rate date, URL, sha256 of the raw bytes on the line), converted: true, and the native amount + currency stay on the line; totals count converted lines apart.
# ---------------------------------------------------------------------------------------------------------------------------------------------
UK_URL = 'https://www.abebooks.co.uk/servlet/SearchResults?isbn={isbn}&sortby=2'
UK_URL_NEW = 'https://www.abebooks.co.uk/servlet/SearchResults?isbn={isbn}&cond=new&sortby=2'
CONVERTED_MARKETS = {'ES': {'currency': 'EUR', 'country': 'ESP', 'urls': (ES_URL, ES_URL_NEW), 'name': 'Iberlibro (AbeBooks Spain), sellers in Spain'},
                     'GB': {'currency': 'GBP', 'country': 'GBR', 'urls': (UK_URL, UK_URL_NEW), 'name': 'AbeBooks UK, sellers in the United Kingdom'}}
CONVERTED_POLICY = ('Converted fallback: no US-seller USD offer was validated for the equivalent edition, so the lowest validated asking price of a seller in a non-US '
                    'market (native currency) is converted to USD at the ECB reference rate cited on the line (rate, rate date, source URL, sha256); converted: true, '
                    'native amount and currency kept. US offers always take precedence.')
def converted_fallback_enabled(config=None, environ=None):
    """ON by default (user decision 2026-10-07); config['converted_fallback'] wins, else LIBRARY_CONVERTED_FALLBACK=0 switches it off."""
    if config and 'converted_fallback' in config: return bool(config['converted_fallback'])
    return (os.environ if environ is None else environ).get('LIBRARY_CONVERTED_FALLBACK', '1') != '0'


def converted_markets(config=None, environ=None):
    raw = (config or {}).get('converted_markets') or (os.environ if environ is None else environ).get('LIBRARY_CONVERTED_MARKETS', 'ES,GB')
    out = [m.strip().upper() for m in (raw.split(',') if isinstance(raw, str) else raw)]
    return [m for m in out if m in CONVERTED_MARKETS]


def _market_order(language, markets):
    first = {'es': 'ES', 'en': 'GB'}.get(language)
    return sorted(markets, key=lambda m: m != first)


def _skip_market(provider, isbn, market, config):
    """A non-US market is not queried when the US result page of this ISBN was parsed, was not full, and lists no seller from that market's country (AbeBooks lists
    sellers of every country on the US site). No parsed US page -> never skipped."""
    if not config.get('fallback_prefilter', True): return None
    seen = (getattr(provider, 'observed', None) or {}).get(('www.abebooks.com', isbn))
    if not seen or seen['listings'] >= PAGE_SIZE: return None
    country = CONVERTED_MARKETS[market]['country']
    return None if seen['countries'].get(country) else f'not_queried:no_{country}_seller_listed_for_this_isbn_on_the_parsed_us_result_page'


def _evaluate_isbn_market(cand, language, family, root, provider, config, market):
    """Native-currency validated listings of one candidate ISBN in one non-US market (validated under that market's own rules; converted later)."""
    isbn, spec = cand['isbn'], CONVERTED_MARKETS[market]
    found, failures = market_offers(provider, isbn, [u.format(isbn=isbn) for u in spec['urls']])
    valid, rejects = [], []
    for offer in found:
        offer = dict(offer); offer['evidence'] = dict(offer.get('evidence') or {})
        path = offer['evidence'].get('raw_path')
        if path and Path(path).is_absolute() and Path(path).is_relative_to(root): offer['evidence']['raw_path'] = str(Path(path).relative_to(root))
        errors = validate_offer(offer, None, root, expected_isbn=isbn, market=market)
        if offer.get('language') != _ABE_LANG[language]: errors.append('listing_language_not_' + language)
        if offer.get('format') != _FAMILY_TO_FORMAT[family]: errors.append('listing_format_not_' + family)
        if errors: rejects.append({'listing_id': offer.get('listing_id'), 'reasons': errors})
        else: valid.append(offer)
    best = {kind: min((o for o in valid if o['kind'] == kind), key=lambda x: Decimal(str(x['amount'])), default=None) for kind in ('replacement_cost', 'used_value')}
    return {'isbn': isbn, 'edition_key': cand.get('edition_key'), 'market': market, 'currency': spec['currency'], 'listings_valid': len(valid), 'listings_rejected': len(rejects),
            'has_new': best['replacement_cost'] is not None, 'has_used': best['used_value'] is not None,
            'source_unavailable': bool(failures) and all(f.get('source_unavailable') for f in failures) and not found, 'source_failures': failures}, best


def _converted_line(offer, root, isbn, fx, config):
    from providers import fx as FX
    line = _line(offer, root, isbn)
    native, cur = offer['amount'], offer['currency']
    usd = FX.convert_to_usd(native, fx)
    line.update(amount=usd, currency='USD', converted=True, original_amount=native, original_currency=cur, fx={**fx, 'raw_path': 'pricing_evidence/' + fx['raw_path']},
                conversion=(f"{cur} {native:.2f} x {fx['rate']} USD/{cur} ({fx['method']}; ECB reference rate dated {fx['rate_date']}, retrieved {fx['retrieved_at'][:10]}) "
                            f"= USD {usd:.2f}"),
                conversion_policy=CONVERTED_POLICY)
    return line


def price_converted_fallback(row, us_pick, cands, language, family, root, provider, config, deadline=None):
    """Fill the condition classes the US market did not price, from non-US markets, converted to USD. Never replaces a US line. One equivalent edition per row: with a US
    pick, only that ISBN is looked up abroad; without one, the ranked candidates are. Returns the chosen (summary, lines) or None; records every market tried."""
    kinds = ('replacement_cost', 'used_value')
    need = [k for k in kinds if not (us_pick and us_pick[1][k])]
    markets = _market_order(language, converted_markets(config))
    if not need or not markets: return None
    fb = row['converted_fallback'] = {'policy': CONVERTED_POLICY, 'markets_in_order': markets, 'needed': need, 'tried': [], 'issues': []}
    pool = [next(c for c in cands if c['isbn'] == us_pick[0]['isbn'])] if us_pick else cands[:config.get('max_isbn_tries_converted', 3)]
    results = []
    for cand in pool:
        for market in markets:
            if deadline is not None and time.monotonic() >= deadline: fb['issues'].append('budget'); break
            skip = _skip_market(provider, cand['isbn'], market, config)
            if skip: fb['tried'].append({'isbn': cand['isbn'], 'market': market, 'skipped': skip}); continue
            summary, best = _evaluate_isbn_market(cand, language, family, root, provider, config, market)
            fb['tried'].append(summary)
            if any(best[k] for k in need): results.append((cand, summary, best))
        if any(all(r[2][k] for k in need) for r in results) or 'budget' in fb['issues']: break
    if not results:
        fb['issues'].append('no_validated_non_us_listing_for_the_equivalent_editions_tried'); return None
    chosen = (next((r for r in results if all(r[2][k] for k in need)), None) or next((r for r in results if r[2].get('replacement_cost') and 'replacement_cost' in need), None)
              or results[0])
    from providers import fx as FX
    fx_cache, lines = {}, {}
    limit = Decimal(str(config.get('appraisal_threshold_usd', 250)))
    for k in need:
        src = chosen if chosen[2][k] else next((r for r in results if r[0]['isbn'] == chosen[0]['isbn'] and r[2][k]), None)
        if not src: continue
        offer = src[2][k]
        cur = offer['currency']
        if cur not in fx_cache:
            fx_cache[cur] = FX.usd_rate(cur, root / 'pricing_evidence', online=config.get('pricing_online', False),
                                        cache_dir=Path(config.get('price_cache_dir') or root / 'cache') / 'fx')
        fx = fx_cache[cur]
        if not fx: fb['issues'].append(f'no_fx_evidence_for_{cur}'); continue
        line = _converted_line(offer, root, chosen[0]['isbn'], fx, config)
        if Decimal(str(line['amount'])) >= limit: fb['issues'].append('converted_offer_above_appraisal_threshold_excluded'); continue
        errors = validate_offer(line, None, root, expected_isbn=chosen[0]['isbn'])
        if errors: fb['issues'].append({'rejected_converted_line': line.get('url'), 'reasons': errors}); continue
        lines[k] = line
    if not lines: return None
    fb['chosen'] = {'isbn': chosen[0]['isbn'], 'markets': sorted({l['market'] for l in lines.values()})}
    return chosen, lines


def price_work(work, root, catalogue, provider, config, deadline=None):
    """work = {id, work, title, visible_text, language_code?, readings?}. Returns one price row (never an exact-edition claim)."""
    row = {'ref_id': work['id'], 'work': work.get('work'), 'title': work.get('title'), 'price_basis': PRICE_BASIS, 'replacement_cost': None, 'used_value': None, 'issues': [],
           'equivalent_edition': None, 'alternatives_considered': [], 'rule': EQUIVALENT_RULE}
    if work.get('status') == 'needs_appraisal': row['issues'].append('appraisal_required_no_automatic_price'); return row
    try:
        cat = catalogue.editions(work.get('work'))
    except Exception as exc:
        row['issues'].append(f'edition_records_unavailable:{type(exc).__name__}:{str(exc)[:120]}'); return row
    row['editions_seen'] = {'returned': len(cat['entries']), 'catalogue_total': cat['size'], 'urls': cat['urls'], 'retrieved_at': cat['retrieved_at']}
    language, source = resolve_language(work.get('visible_text'), None, work.get('title'), cat['entries'])
    if not language and not source.endswith('both_languages'):
        code = listing_language_of_titled_editions(cat['entries'], work.get('title'), provider)
        if code: language, source = code, 'abebooks_listing_language_of_editions_titled_exactly_like_the_spine_title'
    if not language and config.get('reading_language_fallback', True):
        # LAST resort (spine words, spine publisher, catalogue titles and listing languages did not decide): the majority language of the group's own spine readings,
        # with the reader's quoted evidence. Model-read: labelled as such on the row; never used when anything above decided.
        vote = reading_language_vote(work.get('readings')) if work.get('readings') else None
        if vote is None and work.get('language_code') in _OL_LANG:
            vote = {'code': work['language_code'], 'votes': {work['language_code']: 1}, 'readings': 1, 'evidence': []}
        if vote:
            row['language_undetermined_by_catalogue'] = source
            row['language_reading_vote'] = vote
            language = vote['code']
            source = (f"spine_reading_language_majority:{vote['votes'].get(language, 0)}_of_{sum(vote['votes'].values())}_coded_readings"
                      f" (model-read fallback; catalogue/spine did not decide: {row['language_undetermined_by_catalogue']})")
    row['language'], row['language_source'] = language, source
    if not language: row['issues'].append(source); return row
    family = config.get('target_family', 'paperback')
    cands, rejected = rank_equivalent_editions(cat['entries'], language, family, config.get('this_year'), unrecorded_ok=True)
    row['rejected_edition_records'] = {'count': len(rejected), 'by_reason': {r: sum(x['reason'] == r for x in rejected) for r in sorted({x['reason'] for x in rejected})}}
    if not cands: row['issues'].append('no_isbn_bearing_edition_in_language_and_format_family'); return row
    tried = []
    for cand in cands[:config.get('max_isbn_tries', 8)]:
        if deadline is not None and time.monotonic() >= deadline:
            row['issues'].append('budget'); break
        tried.append(_evaluate_isbn(cand, language, family, root, provider, config))
        if tried[-1][0]['has_new']: break
    row['alternatives_considered'] = [t[0] for t in tried]
    pick = next((t for t in tried if t[0]['has_new']), None) or next((t for t in tried if t[0]['has_used']), None)
    if second_market_enabled(config) and language == 'es' and config.get('pricing_online', True) is not None:
        try: price_second_market(row, cands, language, family, root, provider, config, deadline)
        except Exception as exc: row['second_market'] = {'market': SECOND_MARKET, 'currency': 'EUR', 'replacement_cost': None, 'used_value': None, 'issues': [f'second_market_failed:{type(exc).__name__}:{str(exc)[:120]}']}
    unavailable = [t[0]['isbn'] for t in tried if any(f.get('source_unavailable') for f in t[0].get('source_failures') or []) and not t[0]['listings_valid']]
    fallback = None
    if converted_fallback_enabled(config):
        try: fallback = price_converted_fallback(row, pick, cands, language, family, root, provider, config, deadline)
        except Exception as exc: row.setdefault('converted_fallback', {})['issues'] = [f'converted_fallback_failed:{type(exc).__name__}:{str(exc)[:120]}']
    if not pick and not fallback:
        if unavailable and len(unavailable) == len(tried): row['issues'].append('us_price_source_unavailable_http_for_every_edition_tried')
        else:
            row['issues'].append('no_validated_us_usd_listing_for_the_equivalent_editions_tried')
            if unavailable: row['issues'].append(f'us_price_source_unavailable_for_{len(unavailable)}_of_{len(tried)}_editions_tried')
        if row.get('converted_fallback'): row['issues'] += [i for i in row['converted_fallback']['issues'] if isinstance(i, str)]
        return row
    if pick:
        summary, best = pick
        why = ('most recent ' + family + '-family edition record in the spine language with a checksum-valid ISBN that has a validated US listing'
               + ('; a NEW US listing exists (in print)' if summary['has_new'] else '; ONLY used US listings exist (no new copy seen)'))
    else:
        cand, msum, best = fallback[0]
        summary = {**cand, **{k: msum[k] for k in ('has_new', 'has_used')}}
        why = ('most recent ' + family + '-family edition record in the spine language with a checksum-valid ISBN; NO validated US listing for any edition tried, '
               'so the offer is from a non-US market, converted to USD (converted: true)')
        row['issues'].append('no_validated_us_usd_listing_for_the_equivalent_editions_tried')
    row['equivalent_edition'] = {k: summary[k] for k in ('isbn', 'isbn_field', 'edition_key', 'publish_date', 'physical_format', 'family', 'languages', 'language_basis', 'publisher', 'title')}
    row['equivalent_edition'].update(source_url=edition_url(summary['edition_key']), language_source=source, why=why)
    for kind in ('replacement_cost', 'used_value'):
        if pick and best[kind]: row[kind] = _line(best[kind], root, summary['isbn'])
        elif fallback and kind in fallback[1]:
            row[kind] = fallback[1][kind]; row['issues'].append(kind + '_converted_from_' + fallback[1][kind]['original_currency'])
        else: row['issues'].append(kind + '_source_unavailable')
    row['selection_policy'] = ('lowest validated US/USD asking offer per condition class for the equivalent ISBN; when no US seller offers it, the lowest validated non-US '
                               'asking offer converted to USD (converted: true, ECB rate on the line); work-level equivalent, not an exact-edition or realized-sale price')
    return row


def price_works(works, run_dir, config=None, catalogue=None, provider=None):
    config = config or {}; root = Path(run_dir).resolve(); start = time.perf_counter()
    from providers.openlibrary import OpenLibrary
    cache_dir = config.get('price_cache_dir') or root / 'cache'
    own_cat = catalogue is None
    catalogue = catalogue or OpenLibrary(Path(cache_dir) / 'openlibrary', online=config.get('pricing_online', False))
    provider = provider or PublicBookPrices(root / 'pricing_evidence', online=config.get('pricing_online', False), cache_dir=Path(cache_dir) / 'abebooks',
                                            max_age_hours=config.get('max_age_hours', 24 * 30), min_interval_s=config.get('min_interval_s', 2.0))
    try:
        rows = [price_work(w, root, catalogue, provider, config) for w in works]
    finally:
        if own_cat: catalogue.close()
    priced = sum(r['replacement_cost'] is not None or r['used_value'] is not None for r in rows)
    return C.envelope(config.get('run_id', root.name), 'price', {'prices': rows, 'market': 'US', 'currency': 'USD', 'price_basis': PRICE_BASIS, 'equivalent_rule': EQUIVALENT_RULE},
                      status='partial' if any(r['issues'] for r in rows) or not rows else 'ok',
                      metrics={'elapsed_s': round(time.perf_counter() - start, 3), 'works': len(rows), 'priced_works': priced,
                               'replacement_lines': sum(r['replacement_cost'] is not None for r in rows), 'used_lines': sum(r['used_value'] is not None for r in rows),
                               'second_market_priced_works': sum(bool((r.get('second_market') or {}).get('replacement_cost') or (r.get('second_market') or {}).get('used_value')) for r in rows),
                               'actual_API_cost_usd': 0, 'cost_provenance': 'Public HTTP (Open Library, AbeBooks)/cached evidence, no paid model/account'})


def price_config_from_env(run_dir, environ=None):
    """Work-level pricing settings for the engine: LIBRARY_PRICE_ONLINE=0 -> cache only (no HTTP); LIBRARY_PRICE_BUDGET_S (default 60) -> wall-clock budget; shared cache runs/_price_cache;
    LIBRARY_SECOND_MARKET=1 -> also price the Spain/EUR second market (default off: USD only)."""
    env = os.environ if environ is None else environ
    try: budget = float(env.get('LIBRARY_PRICE_BUDGET_S', 60))
    except ValueError: budget = 60.0
    return {'pricing_online': env.get('LIBRARY_PRICE_ONLINE', '1') != '0', 'price_budget_s': budget, 'price_cache_dir': str(Path(run_dir).resolve().parent / '_price_cache'),
            'second_market': second_market_enabled(None, env), 'converted_fallback': converted_fallback_enabled(None, env), 'converted_markets': converted_markets(None, env)}


def price_books_with_work_level(books, run_dir, config=None, provider=None, catalogue=None, wl_provider=None):
    """Exact-edition pricing (price_books) first; every identified catalogue-match book that has a work id but no printed ISBN is then priced through the work-level
    equivalent policy under a wall-clock budget (config price_budget_s, default 60; works not reached -> null 'work_level_skipped:budget'). Never raises for a source failure.
    One price-source client (``wl_provider``, built once) serves every work, so the request spacing (min_interval_s) also holds between works."""
    config = config or {}; root = Path(run_dir).resolve()
    artifact = price_books(books, run_dir, config, provider)
    rows = artifact['data']['prices']
    deadline = time.monotonic() + float(config.get('price_budget_s', 60))
    cache_dir = config.get('price_cache_dir') or root / 'cache'
    own_cat = catalogue is None
    memo, attempted, skipped = {}, 0, 0
    try:
        for book, row in zip(books, rows):
            if row.get('replacement_cost') or row.get('used_value') or row.get('needs_appraisal') or book.get('status') != 'identified' or book.get('identity_source') != 'catalogue_match': continue
            if not any(str(i).startswith('no supported printed ISBN') for i in row['issues']): continue
            key = priced_work_key(book)
            if not key:
                row['issues'].append('work_level_skipped:no_catalogue_work_id'); continue
            if time.monotonic() >= deadline:
                row['issues'].append('work_level_skipped:budget'); skipped += 1; continue
            lang = book.get('language') if book.get('language') in _OL_LANG else None
            vote = reading_language_vote(book.get('reading_evidence'))
            memo_key = (key, book.get('visible_text'), lang, (vote or {}).get('code'))
            if memo_key not in memo:
                if catalogue is None:
                    from providers.openlibrary import OpenLibrary
                    catalogue = OpenLibrary(Path(cache_dir) / 'openlibrary', online=config.get('pricing_online', False))
                if wl_provider is None:
                    wl_provider = PublicBookPrices(root / 'pricing_evidence', online=config.get('pricing_online', False), cache_dir=Path(cache_dir) / 'abebooks',
                                                   max_age_hours=config.get('max_age_hours', 24 * 30), min_interval_s=config.get('min_interval_s', 2.0),
                                                   throttle_retries=config.get('throttle_retries', 0), throttle_backoff_s=config.get('throttle_backoff_s', 60.0))
                try:
                    memo[memo_key] = price_work({'id': book['id'], 'work': key, 'title': book.get('title'), 'visible_text': book.get('visible_text'), 'language_code': lang,
                                                 'readings': book.get('reading_evidence') or []},
                                                root, catalogue, wl_provider, config, deadline)
                except Exception as exc:
                    memo[memo_key] = {'replacement_cost': None, 'used_value': None, 'issues': [f'work_level_failed:{type(exc).__name__}:{str(exc)[:120]}']}
                attempted += 1
            wl = copy.deepcopy(memo[memo_key])
            row['issues'] = [i for i in row['issues'] if not str(i).startswith('no supported printed ISBN')] + wl['issues']
            # same shape as a price_work row (the packet binds work-level rows by ref_id + accepted title and reads equivalent_edition from the row)
            row.update(replacement_cost=wl['replacement_cost'], used_value=wl['used_value'], price_basis=PRICE_BASIS, title=book.get('title'), work=key,
                       **{k: wl[k] for k in ('language', 'language_source', 'language_reading_vote', 'equivalent_edition', 'alternatives_considered', 'editions_seen', 'rule', 'selection_policy',
                                             'second_market', 'converted_fallback') if wl.get(k) is not None})
            if any(i == 'budget' for i in wl['issues']): skipped += 1
    finally:
        if own_cat and catalogue is not None: catalogue.close()
    artifact['data']['work_level_equivalent'] = {'rule': EQUIVALENT_RULE, 'price_basis': PRICE_BASIS, 'budget_s': config.get('price_budget_s', 60), 'online': bool(config.get('pricing_online', False))}
    artifact['metrics'].update(replacement_lines=sum(r['replacement_cost'] is not None for r in rows), used_lines=sum(r['used_value'] is not None for r in rows),
                               second_market_priced=sum(bool((r.get('second_market') or {}).get('replacement_cost') or (r.get('second_market') or {}).get('used_value')) for r in rows),
                               work_level_attempted=attempted, work_level_priced=sum(r.get('price_basis') == PRICE_BASIS and (r['replacement_cost'] is not None or r['used_value'] is not None) for r in rows),
                               work_level_budget_skipped=skipped)
    artifact['status'] = 'partial' if any(r['issues'] for r in rows) or not rows else 'ok'
    return artifact


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run');parser.add_argument('--online',action='store_true');parser.add_argument('--source-config')
    parser.add_argument('--works',help='work-level equivalent pricing: JSON list of {id, work, title, visible_text, language_code?}; needs --out (a NEW directory)')
    parser.add_argument('--out');parser.add_argument('--cache-dir')
    args=parser.parse_args()
    if args.works:
        out=Path(args.out or '')
        if not args.out or (out.exists() and any(out.iterdir())): parser.error('--out must be a new (empty) directory; existing runs are never written into')
        out.mkdir(parents=True,exist_ok=True)
        cfg={'pricing_online':args.online,'run_id':out.name,**({'price_cache_dir':args.cache_dir} if args.cache_dir else {})}
        print(C.write_json(out/'prices.json',price_works(C.read_json(args.works),out,cfg)));return
    if not args.run: parser.error('--run or --works required')
    root=Path(args.run);cfg=C.read_json(args.source_config) if args.source_config else {}
    cfg['pricing_online']=args.online;print(price(root/'inventory.json',root/'manifest.json',root,cfg))

if __name__=='__main__':main()
