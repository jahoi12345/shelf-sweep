"""Integration and pricing safeguards. Synthetic offers are explicitly fixture evidence."""
import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from providers.stages import stage_configs
from providers import gemini
from providers.book_prices import parse_jsonld, parse_abebooks, PublicBookPrices
from pipeline.price import price_books, validate_offer, supported_isbn, identity_key
from pipeline.packet import compute_totals, build_data, verify_packet
from pipeline import contracts as C
from tests.test_live_app import make_frame
from app.engine import SweepEngine

ISBN='9780142410370'

@pytest.fixture
def annotation_bundle(tmp_path, monkeypatch):
    # Synthetic native evidence: test the HTTP/hash boundary without workspace runs.
    from app import server
    evidence=tmp_path/'evidence';evidence.mkdir()
    frame=make_frame();(evidence/'f00000.jpg').write_bytes(frame)
    (tmp_path/'index.html').write_text('Human reviewed <img src="evidence/f00000.jpg"><script src="annotation_payload.json"></script>')
    (tmp_path/'annotation_payload.json').write_text(json.dumps({'images':{'f00000':'evidence/f00000.jpg'},'annotations':{'provenance':{'review_status':'pending'}}}))
    (tmp_path/'bundle.json').write_text(json.dumps({'evidence_frames':[{'url':'evidence/f00000.jpg','sha256':hashlib.sha256(frame).hexdigest()}]}))
    monkeypatch.setattr(server,'ANNOTATION_DIR',tmp_path)
    return tmp_path


def supported_book(title='Matilda'):
    return {'id':'bk00001','title':title,'status':'identified','isbn':ISBN,'edition':'/books/OL1M',
            'identity_source':'catalogue_match','identity_matching_evidence':[{'field':'edition_isbn','visible':ISBN,
             'catalogue':ISBN,'edition_key':'/books/OL1M','source_url':'https://openlibrary.org/books/OL1M','retrieved_at':'2026-10-06T00:00:00Z'}]}


def fixture_page(fmt='Paperback', currency='USD', region='US', condition='NewCondition', price='12.34'):
    return '<script type="application/ld+json">'+json.dumps({'@type':'Book','isbn':ISBN,'bookFormat':'https://schema.org/'+fmt,
        'offers':{'@type':'Offer','price':price,'priceCurrency':currency,'eligibleRegion':region,
                  'itemCondition':'https://schema.org/'+condition,'availability':'https://schema.org/InStock'}})+'</script>'

class FixturePrices:
    def __init__(self, root):self.root=root
    def offers(self,isbn,urls=None):
        raw=self.root/'fixture_offer.html';raw.write_text(fixture_page()+fixture_page(condition='UsedCondition'))
        offers=[]
        for condition in ('NewCondition','UsedCondition'):
            offer=parse_jsonld(fixture_page(condition=condition),isbn,'https://www.penguinrandomhouse.com/fixture')[0]
            offer.update(source='SYNTHETIC fixture',url='https://www.penguinrandomhouse.com/fixture',retrieved_at='2026-10-06T00:00:00Z',converted=False,
                         evidence={'url':'https://www.penguinrandomhouse.com/fixture','retrieved_at':'2026-10-06T00:00:00Z','http_status':200,'raw_path':str(raw),
                                   'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'retrieval_provenance':'mocked_fixture'})
            offers.append(offer)
        return offers,[]

@pytest.mark.parametrize('fmt,currency,region,amount',[('EBook','USD','US','12'),('Paperback','EUR','US','12'),('Paperback','USD','ES','12'),('Paperback','USD','US','0'),('Paperback','USD','US','NaN')])
def test_reject_nonphysical_wrong_market_invalid_money(fmt,currency,region,amount):
    assert parse_jsonld(fixture_page(fmt,currency,region,price=amount),ISBN,'https://www.penguinrandomhouse.com/fixture')==[]


def test_same_physical_price_path_for_spanish_and_english(tmp_path):
    for title in ('Matilda','El jardín de los pájaros'):
        book=supported_book(title)
        assert supported_isbn(book)==ISBN
        result=price_books([book],tmp_path,provider=FixturePrices(tmp_path))
        row=result['data']['prices'][0]
        assert row['replacement_cost']['amount']==12.34 and row['used_value']['amount']==12.34
        assert validate_offer(row['used_value'],book,tmp_path)==[]
        assert row['identity_key']==identity_key(book)


def test_unsupported_and_appraisal_never_query(tmp_path):
    class NoCalls:
        def offers(self,*args):raise AssertionError('unsupported edition must not query')
    books=[{**supported_book(),'isbn':None},{**supported_book(),'status':'needs_appraisal'},
           {**supported_book(),'identity_source':'user_statement'},{**supported_book(),'identity_matching_evidence':[]}]
    result=price_books(books,tmp_path,provider=NoCalls())
    assert all(r['replacement_cost'] is None and r['used_value'] is None for r in result['data']['prices'])


def test_source_hash_and_edition_tampering_rejected(tmp_path):
    book=supported_book();row=price_books([book],tmp_path,provider=FixturePrices(tmp_path))['data']['prices'][0]
    (tmp_path/'fixture_offer.html').write_text('tampered')
    assert 'raw evidence missing, escaped or hash mismatch' in validate_offer(row['used_value'],book,tmp_path)
    assert validate_offer(row['used_value'],{**book,'isbn':None},tmp_path)


def test_decimal_totals_and_missing_exclusions():
    books=[{'status':'identified','pricing_attempted':True,'replacement_cost':{'amount':.1},'used_value':None},
           {'status':'unidentified','pricing_attempted':True,'replacement_cost':{'amount':.2},'used_value':None}]
    t=compute_totals(books,[],{})
    assert t['books_replacement_cost']==.3 and t['books_used_value'] is None and t['excluded_from_totals']==2


def test_cached_real_publisher_probe_is_separate_from_sweep(tmp_path):
    source=Path(__file__).resolve().parents[1]/'runs/worker3_phase2_pricing_probe'
    if not (source/'retrieval.json').is_file() or not (source/'publisher_page.html').is_file():
        pytest.skip('Optional saved publisher retrieval absent; synthetic source/edition fixtures run separately')
    meta=json.loads((source/'retrieval.json').read_text());html=(source/'publisher_page.html').read_text()
    assert hashlib.sha256((source/'publisher_page.html').read_bytes()).hexdigest()==meta['sha256']
    offers=parse_jsonld(html,ISBN,meta['url'])
    assert len(offers)==1 and offers[0]['amount']==9.99 and offers[0]['kind']=='replacement_cost'
    assert supported_isbn({'status':'unidentified','isbn':None}) is None


def test_abebooks_offer_scoped_to_own_isbn_condition_market():
    html=f'<div class="result">ISBN 13: {ISBN} Paperback Condition: Used - Good US$ 6.49 Ships within U.S.A.</div>'
    offers=parse_abebooks(html,ISBN,'https://www.abebooks.com/book-search/isbn/'+ISBN+'/')
    assert len(offers)==1 and offers[0]['kind']=='used_value'
    assert parse_abebooks(html.replace(ISBN,'9780142410363'),ISBN,'https://www.abebooks.com/')==[]


def test_stage_adapters_are_independent_without_gemini(tmp_path):
    with patch.object(gemini,'credential_available',return_value=False),patch.object(gemini,'is_mocked',return_value=False),\
         patch('providers.stages.Path.is_file',return_value=True):
        cfg,availability=stage_configs({'detector':'local','reader':'none'},run_dir=tmp_path,cache=None,usage=None)
    assert availability['detect']['available'] and not availability['read']['available']
    assert 'image_request' in cfg['detect'] and 'image_request' not in cfg['read']


def test_annotation_route_preserves_pending_human_labels(annotation_bundle):
    from app.server import annotations
    response=annotations()
    assert b'file://' not in response.body
    assert b'/annotations/evidence/f00000.jpg' in response.body
    from app.server import annotation_payload
    payload=annotation_payload()
    assert payload['images']['f00000']=='/annotations/evidence/f00000.jpg'
    assert payload['annotations']['provenance']['review_status']=='pending'
    assert b'Human reviewed' in response.body


def test_normalized_amount_tampering_rejected(tmp_path):
    book=supported_book();row=price_books([book],tmp_path,provider=FixturePrices(tmp_path))['data']['prices'][0]
    assert 'normalized offer does not match raw source bytes' in validate_offer({**row['replacement_cost'],'amount':99},book,tmp_path)


def test_dated_cache_never_fabricates_online_availability(tmp_path):
    provider=PublicBookPrices(tmp_path/'raw',online=False)
    offers,failures=provider.offers(ISBN)
    assert offers==[] and failures and 'online disabled' in failures[0]['reason']


def test_annotation_http_rejects_unlisted_and_keeps_hashes(annotation_bundle):
    from app.server import annotation_evidence
    from fastapi import HTTPException
    assert annotation_evidence('f00000.jpg').media_type=='image/jpeg'
    with pytest.raises(HTTPException):annotation_evidence('not-listed.jpg')
