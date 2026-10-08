"""Public physical-book offers with raw-byte evidence and bounded disk caching.

No credentials/accounts, currency inference, ebook fallback or memory prices.
Publisher JSON-LD and AbeBooks ISBN search are separate price-source parsers.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from html import unescape
from html.parser import HTMLParser
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse
import httpx

PHYSICAL = {'Paperback', 'Hardcover', 'Hardback', 'Softcover', 'MassMarketPaperback', 'BoardBook'}


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values(): yield from walk(child)
    elif isinstance(value, list):
        for child in value: yield from walk(child)


def positive_amount(value):
    if isinstance(value, bool): return None
    try:
        amount = Decimal(str(value))
        return float(amount.quantize(Decimal('.01'))) if amount.is_finite() and amount > 0 else None
    except (InvalidOperation, ValueError, TypeError): return None


def parse_jsonld(html, isbn, url):
    offers = []
    for script in re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>([\s\S]*?)</script>', html, re.I):
        try: data = json.loads(unescape(script))
        except ValueError: continue
        for book in walk(data):
            if book.get('@type') not in ('Book', 'Product') or str(book.get('isbn') or book.get('gtin13')) != isbn: continue
            fmt = str(book.get('bookFormat') or book.get('format') or '').rsplit('/', 1)[-1]
            if fmt not in PHYSICAL: continue
            # Scope to this edition; never collect a sibling ebook/edition offer.
            for offer in walk([book.get('offers', {}), book.get('potentialAction', {})]):
                if offer.get('@type') != 'Offer': continue
                region = offer.get('eligibleRegion')
                if isinstance(region, dict): region = region.get('name') or region.get('addressCountry')
                if region not in ('US', 'USA', 'United States') or offer.get('priceCurrency') != 'USD': continue
                amount = positive_amount(offer.get('price', offer.get('Price')))
                if amount is None: continue
                condition = str(offer.get('itemCondition') or '').rsplit('/', 1)[-1]
                publisher = urlparse(url).hostname in ('www.penguinrandomhouse.com', 'penguinrandomhouse.com')
                if condition == 'NewCondition': kind, assumed, proxy = 'replacement_cost', 'new', 'retail asking offer'
                elif condition == 'UsedCondition': kind, assumed, proxy = 'used_value', 'used; grade unspecified', 'used asking offer; captured condition unknown'
                elif publisher: kind, assumed, proxy = 'replacement_cost', 'new assumed from publisher retail offer', 'publisher list price proxy'
                else: continue
                availability = str(offer.get('availability', ''))
                if any(x in availability for x in ('OutOfStock', 'Discontinued', 'SoldOut', 'PreOrder')): continue
                if not availability.endswith('InStock'): continue
                offers.append({'kind': kind, 'amount': amount, 'isbn': isbn, 'format': fmt,
                               'currency': 'USD', 'market': 'US', 'condition_assumed': assumed,
                               'proxy_assumptions': proxy, 'edition_match': 'exact_isbn',
                               'availability': availability, 'raw_offer': offer})
    return offers


class Listings(HTMLParser):
    """Capture individual AbeBooks result containers; ISBN/condition/price stay scoped."""
    def __init__(self):
        super().__init__(); self.depth = 0; self.active = None; self.rows = []; self.root_depth = 0
    def handle_starttag(self, tag, attrs):
        if tag not in ('img','br','input','meta','link','hr'): self.depth += 1
        a = dict(attrs); classes = a.get('class','').split()
        if self.active is None and ('result' in classes or a.get('data-cy') == 'listing-item'):
            self.active = []; self.root_depth = self.depth
    def handle_endtag(self, tag):
        if self.active is not None and self.depth == self.root_depth:
            self.rows.append(' '.join(self.active)); self.active = None
        self.depth = max(0, self.depth - 1)
    def handle_data(self, data):
        if self.active is not None and data.strip(): self.active.append(data.strip())


_EXCLUDED_FLAGS = ('SIGNED', 'FIRST_EDITION', 'COLLECT', 'ANTIQUAR', 'RARE', 'LIMITED_EDITION', 'INSCRIBED')


def abebooks_listings(html):
    """Listings embedded in the current (Next.js) AbeBooks result page: the React Flight payload carries ``searchResultsArray.results[].listing``. None if absent."""
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', html)
    try: text = ''.join(json.loads('"' + c + '"') for c in chunks)
    except ValueError: return None
    i = text.find('"searchResultsArray"')
    if i < 0: return None
    try: obj, _ = json.JSONDecoder().raw_decode(text[text.find('{', i + 10):])
    except ValueError: return None
    return [r['listing'] for r in obj.get('results', []) if isinstance(r, dict) and isinstance(r.get('listing'), dict)]


# marketplace host -> (market, price currency, seller country). A price is "native" only when the listing's currency is the currency of its seller's country: the US site for
# sellers in the USA (USD) and the Spanish site (iberlibro) for sellers in Spain (EUR). Every other combination is a marketplace conversion and is never used.
MARKETS = {'www.abebooks.com': ('US', 'USD', 'USA'), 'www.iberlibro.com': ('ES', 'EUR', 'ESP'), 'www.abebooks.co.uk': ('GB', 'GBP', 'GBR')}
SOURCE_NAMES = {'www.abebooks.com': 'AbeBooks seller', 'www.iberlibro.com': 'Iberlibro (AbeBooks Spain) seller in Spain',
                'www.abebooks.co.uk': 'AbeBooks UK seller in the United Kingdom'}
# A marketplace that answers with anything but 200 (AbeBooks answers 202 with an EMPTY body when it throttles a client) has not said "no listing": the page is
# unavailable, never parsed, never cached as evidence, and the caller reports the source as unavailable (not as "no validated listing").
THROTTLE_STATUS = {202, 403, 429, 503}


class SourceUnavailable(RuntimeError):
    pass


def parse_abebooks_listings(html, isbn, url):
    """Offers from the embedded listing JSON: only listings of this exact ISBN, physical, in the host's NATIVE currency from sellers of that market (abebooks.com: USD from US
    sellers; iberlibro.com: EUR from sellers in Spain), not signed/first/collectible/ebook."""
    market, currency, country = MARKETS.get(urlparse(url).hostname, MARKETS['www.abebooks.com'])
    out = []
    for l in abebooks_listings(html) or []:
        codes = set(l.get('bsaCodes') or []) | set(l.get('searchAttributes') or [])
        price = l.get('priceInPurchaseCurrency') or {}
        fmt = 'Hardcover' if 'HARDCOVER' in codes else 'Paperback' if 'SOFT_COVER' in codes else None
        cond = str(l.get('condition') or '')
        if isbn not in (l.get('isbn13'), l.get('isbn10')) or not fmt or price.get('currency') != currency or l.get('vendorCountry') != country: continue
        if any(f in c for c in codes for f in _EXCLUDED_FLAGS) or l.get('productType') not in (None, 'DEFAULT'): continue
        if cond == 'NEW': kind, assumed = 'replacement_cost', 'New'
        elif cond.startswith('USED'): kind, assumed = 'used_value', 'Used - ' + str(l.get('vendorCondition') or cond[5:].title())
        else: continue
        amount = positive_amount(price.get('amount'))
        if amount is None: continue
        out.append({'kind': kind, 'amount': amount, 'isbn': isbn, 'format': fmt, 'currency': currency, 'market': market, 'condition_assumed': assumed, 'edition_match': 'exact_isbn',
                    'language': l.get('language'), 'seller_country': l.get('vendorCountry'), 'listing_id': l.get('listingId'),
                    'proxy_assumptions': 'seller asking price, not completed sale; captured condition unknown; shipping/tax excluded',
                    'availability': 'listed offer; checkout not verified', 'raw_offer': l})
    return out


def parse_abebooks(html, isbn, url):
    if urlparse(url).hostname not in MARKETS: return []
    if abebooks_listings(html) is not None: return parse_abebooks_listings(html, isbn, url)
    if urlparse(url).hostname != 'www.abebooks.com': return []     # the legacy HTML fallback below only understands US$ rows from US sellers
    parser = Listings(); parser.feed(html); offers = []
    for row in parser.rows:
        if not re.search(r'ISBN\s*(?:13|10\s*/\s*ISBN\s*13)\s*:?\s*'+re.escape(isbn), row, re.I): continue
        fmt = next((f for f in PHYSICAL if re.search(r'\b'+f+r'\b', row, re.I)), None)
        if not fmt or re.search(r'\b(ebook|e-book|kindle|digital|signed|antiquarian)\b', row, re.I): continue
        if not re.search(r'(Ships within U\.S\.A\.|Ships within U\.S\.A|Ships to U\.S\.A)', row): continue
        cond = re.search(r'Condition\s*:?\s*(Used\s*-\s*(?:As New|Fine|Very Good|Good|Fair|Poor)|New)\b', row, re.I)
        prices = re.findall(r'US\$\s*([\d,]+\.\d{2})', row)
        if not cond or len(set(prices)) != 1: continue
        amount = positive_amount(prices[0].replace(',',''))
        if amount is None: continue
        condition = cond.group(1)
        offers.append({'kind': 'replacement_cost' if condition.lower() == 'new' else 'used_value',
                       'amount': amount, 'isbn': isbn, 'format': fmt, 'currency':'USD', 'market':'US',
                       'condition_assumed': condition, 'edition_match':'exact_isbn',
                       'proxy_assumptions':'seller asking price, not completed sale; captured condition unknown; shipping/tax excluded',
                       'availability':'listed offer; checkout not verified', 'raw_offer':row})
    return offers


class PublicBookPrices:
    def __init__(self, evidence_dir, *, online=False, cache_dir=None, max_age_hours=24, min_interval_s=0.0, throttle_retries=0, throttle_backoff_s=60.0):
        self.root=Path(evidence_dir); self.root.mkdir(parents=True,exist_ok=True)
        self.cache=Path(cache_dir or self.root); self.cache.mkdir(parents=True,exist_ok=True)
        self.online=online; self.max_age_hours=max_age_hours; self.retrievals=[]; self.min_interval_s=min_interval_s; self._last_fetch=0.0
        # throttled answers (202/403/429/503) are retried after throttle_backoff_s * 2**attempt, at most throttle_retries times; every HTTP attempt is counted per host
        self.throttle_retries=throttle_retries; self.throttle_backoff_s=throttle_backoff_s; self.throttle_events=[]
        self.http_attempts={}; self.observed={}
        # circuit breaker: after `throttle_circuit` consecutive fetches of one host that stayed throttled through every retry, that host fails fast for the rest of
        # this client's life (the caller reports the source unavailable; a later run with the warm cache fills the gap) instead of waiting minutes per page
        self.throttle_circuit=3; self._throttled_in_a_row={}

    def fetch(self, url):
        parsed=urlparse(url)
        # Bound request hosts to supported public sources; reject arbitrary/local targets.
        if parsed.scheme != 'https' or parsed.hostname not in ('www.abebooks.com','www.iberlibro.com','www.abebooks.co.uk','www.penguinrandomhouse.com'):
            raise ValueError('Unsupported public price-source URL')
        key=hashlib.sha256(url.encode()).hexdigest(); meta=self.cache/(key+'.json'); raw=self.cache/(key+'.html')
        cache_hit=False
        if meta.is_file() and raw.is_file():
            m=json.loads(meta.read_text()); age=(datetime.now(timezone.utc)-datetime.fromisoformat(m['retrieved_at'])).total_seconds()/3600
            cache_hit=m.get('url')==url and m.get('http_status')==200 and 0 <= age <= self.max_age_hours and hashlib.sha256(raw.read_bytes()).hexdigest()==m['sha256']
        if not cache_hit:
            if not self.online: raise RuntimeError('Price source not in dated cache; online disabled')
            import time
            if self.throttle_retries and self._throttled_in_a_row.get(parsed.hostname,0)>=self.throttle_circuit:
                raise SourceUnavailable('host_throttled_circuit_open')
            for attempt in range(self.throttle_retries+1):
                wait=self.min_interval_s-(time.monotonic()-self._last_fetch)
                if wait>0: time.sleep(wait)
                self._last_fetch=time.monotonic()
                self.http_attempts[parsed.hostname]=self.http_attempts.get(parsed.hostname,0)+1
                response=httpx.get(url,timeout=15,follow_redirects=False,headers={'User-Agent':'LibraryClaimSourceProbe/1.0'})
                if response.status_code==200:
                    self._throttled_in_a_row[parsed.hostname]=0; break
                self.throttle_events.append({'url':url,'http_status':response.status_code,'bytes':len(response.content or b''),'attempt':attempt,
                                             'at':datetime.now(timezone.utc).isoformat()})
                if response.status_code in THROTTLE_STATUS and attempt<self.throttle_retries:
                    time.sleep(self.throttle_backoff_s*(2**attempt)); continue
                if response.status_code in THROTTLE_STATUS:
                    self._throttled_in_a_row[parsed.hostname]=self._throttled_in_a_row.get(parsed.hostname,0)+1
                raise SourceUnavailable(f'http_{response.status_code}')
            m={'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),'http_status':response.status_code,
               'sha256':hashlib.sha256(response.content).hexdigest()}
            raw.write_bytes(response.content);meta.write_text(json.dumps(m,indent=2))
        destination=self.root/(key+'.html'); destination.write_bytes(raw.read_bytes())
        saved={**m,'cache_hit':cache_hit,'original_retrieved_at':m['retrieved_at'],'raw_path':str(destination),
               'retrieval_provenance':'cached_http_bytes' if cache_hit else 'fresh_public_http'}
        (self.root/(key+'.json')).write_text(json.dumps(saved,indent=2)); self.retrievals.append(saved)
        return destination.read_text(),saved

    def offers(self, isbn, urls=None):
        results=[]; failures=[]
        for url in urls or [f'https://www.abebooks.com/book-search/isbn/{isbn}/']:
            try:
                html,meta=self.fetch(url)
                host=urlparse(url).hostname
                if host in MARKETS:
                    listed=abebooks_listings(html)
                    if listed is not None:     # who sells this ISBN (any country), for the converted-fallback market pre-filter; never price evidence by itself
                        seen=self.observed.setdefault((host,isbn),{'listings':0,'countries':{}})
                        seen['listings']=max(seen['listings'],len(listed))
                        for l in listed:
                            if isbn in (l.get('isbn13'),l.get('isbn10')): seen['countries'][l.get('vendorCountry')]=seen['countries'].get(l.get('vendorCountry'),0)+1
                parsed=parse_abebooks(html,isbn,url) if host in MARKETS else parse_jsonld(html,isbn,url)
                for offer in parsed:
                    results.append({**offer,'source':SOURCE_NAMES.get(host,'Publisher'),
                                    'url':url,'retrieved_at':meta['retrieved_at'],'converted':False,
                                    'original_currency':offer.get('currency','USD'),'original_amount':offer['amount'], 'fx':None,
                                    'evidence':meta})
                if not parsed: failures.append({'url':url,'reason':'no validated exact-ISBN physical offer in the market currency'})
            except SourceUnavailable as exc: failures.append({'url':url,'reason':'source_unavailable:'+str(exc),'source_unavailable':True})
            except Exception as exc: failures.append({'url':url,'reason':f'{type(exc).__name__}: {str(exc)[:160]}','source_unavailable':True})
        return results,failures
