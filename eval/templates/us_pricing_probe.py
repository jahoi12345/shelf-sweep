"""One US physical-book publisher offer probe, not production valuation.

Parse an ISBN-specific JSON-LD physical edition offer from an already retrieved
publisher page. This control cannot price a captured book without accepted printed
ISBN edition evidence in the supplied resolutions. No ebook substitution or FX.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from pipeline.contracts import write_json
from pipeline.read_spines import load, records
from pipeline.resolve import isbn_value


def walk(value):
    if isinstance(value,dict):
        yield value
        for item in value.values():yield from walk(item)
    elif isinstance(value,list):
        for item in value:yield from walk(item)


def parse_offer(html,isbn):
    if isbn_value(isbn) != isbn:
        raise ValueError("Probe ISBN must be checksum-valid")
    matches=[]
    for script in re.findall(r'<script[^>]*type=[\"\']application/ld\+json[\"\'][^>]*>([\s\S]*?)</script>',html):
        try:data=json.loads(script)
        except ValueError:continue
        for book in walk(data):
            if str(book.get('isbn','')) != isbn or book.get('@type') != 'Book':continue
            fmt=str(book.get('bookFormat','')).rsplit('/',1)[-1]
            if fmt not in {'Paperback','Hardcover'}:continue
            for offer in walk(book.get('potentialAction',{})):
                if offer.get('@type') != 'Offer' or offer.get('priceCurrency') != 'USD':continue
                region=offer.get('eligibleRegion',{})
                if not isinstance(region,dict) or region.get('name') != 'US':continue
                amount=offer.get('price',offer.get('Price'))
                if not isinstance(amount,(int,float)) or isinstance(amount,bool) or amount<0:continue
                matches.append({'isbn':isbn,'format':fmt,'amount':amount,'currency':'USD','market':'US',
                                'date_published':book.get('datePublished'),'availability':offer.get('availability'),
                                'offer_type':'publisher_list_price_proxy','condition_assumed':'new',
                                'condition_provenance':'publisher retail purchase offer; explicit condition not supplied',
                                'converted':False,'source_edition_url':book.get('sameAs')})
    if len({(m['format'],m['amount']) for m in matches}) != 1:
        raise ValueError('No unique ISBN-specific US/USD physical-book offer')
    return matches[0]


def probe(html_path,retrieval_path,isbn,out,resolutions_path=None):
    out=Path(out)
    if out.exists():raise ValueError('Preserve previous probe; use a new output path')
    evidence=load(retrieval_path)
    result={'probe_version':1,'production_valuation':False,'market':'US','currency':'USD',
            'source':evidence,'captured_book_supported':False,'control_only':True,
            'new':None,'used':None,'issues':[]}
    if hashlib.sha256(Path(html_path).read_bytes()).hexdigest() != evidence.get("sha256"):
        raise ValueError("Saved publisher evidence differs from retrieval hash")
    try:
        offer=parse_offer(Path(html_path).read_text(),isbn)
        result['new']={**offer,'source_url':evidence['final_url'],'retrieved_at':evidence['retrieved_at']}
        result['status']='control_offer_verified'
    except ValueError as exc:
        result['status']='unavailable';result['issues'].append(str(exc))
    supported=[]
    if resolutions_path:
        for rs in records(load(resolutions_path),'resolutions'):
            if rs.get('decision')=='accepted' and rs.get('isbn')==isbn and rs.get('edition') and any(e.get('field')=='edition_isbn' and isbn_value(e.get('visible'))==isbn and e.get('catalogue')==isbn and e.get('edition_key')==rs.get('edition') and e.get('source_url') and e.get('retrieved_at') for e in rs.get('matching_evidence',[])):
                supported.append(rs['observation_id'])
    if supported:
        result.update(captured_book_supported=True,control_only=False,supported_observation_ids=supported)
    else:
        result['issues'].append('No accepted captured edition/printed ISBN; do not attach control amount to inventory')
    result['issues'].append('Used-market source and condition-specific sold/asking comparable unavailable')
    write_json(out,result)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('html','retrieval','isbn','out'):p.add_argument('--'+arg,required=True)
    p.add_argument('--resolutions')
    a=p.parse_args();print(probe(a.html,a.retrieval,a.isbn,a.out,a.resolutions))

if __name__=='__main__':main()
