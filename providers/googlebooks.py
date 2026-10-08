"""Google Books Volumes API as a SECOND catalogue behind Open Library (titles only; never prices).

API: https://developers.google.com/books/docs/v1/reference/volumes/list (verified 2026-10-07). Anonymous requests share one daily quota that is usually exhausted
(HTTP 429 "Queries per day"); set GOOGLE_BOOKS_API_KEY to a Google Cloud key with the Books API enabled. Requests are paced (LIBRARY_GB_MIN_INTERVAL_S, default
1.0 s) and cached on disk like providers/openlibrary.py. Every hit carries its volume URL, retrieved_at and the raw title/authors/language/pageCount/dimensions.

The client answers in the SAME shapes as providers.openlibrary.OpenLibrary (``search`` -> {"payload": {"docs", "numFound"}, "url", "retrieved_at"}; ``search_fragments``
-> {"candidates", "queries", "errors", "total", "returned", "limit"}), so pipeline.resolve applies its unchanged evidence rule. Work keys are ``gb:<volume id>``.
"""
from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from providers.cache import DiskCache, make_key

BASE = "https://www.googleapis.com/books/v1/volumes"
FIELDS = "totalItems,items(id,volumeInfo(title,subtitle,authors,language,pageCount,dimensions,publishedDate,publisher,industryIdentifiers,printType))"
KEY_RE = re.compile(r"gb:[A-Za-z0-9_-]{8,16}")
LANG3 = {"es": "spa", "en": "eng", "fr": "fre", "ca": "cat", "it": "ita", "pt": "por", "de": "ger"}
DEFAULT_MIN_INTERVAL_S = 1.0
_PACE = threading.Lock()
_PACE_LAST = [0.0]


def min_interval_s(environ=None) -> float:
    try:
        return max(0.1, float((environ if environ is not None else os.environ).get("LIBRARY_GB_MIN_INTERVAL_S", DEFAULT_MIN_INTERVAL_S)))
    except ValueError:
        return DEFAULT_MIN_INTERVAL_S


def volume_url(volume_id: str) -> str:
    return f"https://books.google.com/books?id={volume_id}"


def parse_dimensions(dims: dict[str, Any] | None) -> dict[str, Any] | None:
    """Google's {"height": "23.00 cm", "width": "15.20 cm", "thickness": "2.10 cm"} -> {"height_cm", "width_cm", "thickness_cm", "raw"}; None when no height is recorded."""
    if not isinstance(dims, dict):
        return None
    out = {"raw": dims}
    for k in ("height", "width", "thickness"):
        m = re.fullmatch(r"\s*([\d.]+)\s*(cm|mm|in)?\s*", str(dims.get(k) or ""))
        if m:
            v = float(m.group(1)) * {"cm": 1.0, "mm": 0.1, "in": 2.54, None: 1.0}[m.group(2)]
            out[k + "_cm"] = round(v, 2)
    return out if out.get("height_cm") else None


def _doc(item: dict[str, Any]) -> dict[str, Any]:
    """One volume as an Open Library-shaped search doc (one 'edition' = the volume itself, with its language)."""
    vi = item.get("volumeInfo") or {}
    vid = item.get("id") or ""
    key = "gb:" + vid
    lang = LANG3.get(str(vi.get("language") or "").lower()[:2])
    return {"key": key, "title": vi.get("title"), "author_name": vi.get("authors") or [], "url": volume_url(vid),
            "editions": {"docs": [{"key": key, "title": vi.get("title"), "language": [lang] if lang else []}]},
            "google_books": {"volume_id": vid, "title": vi.get("title"), "subtitle": vi.get("subtitle"), "authors": vi.get("authors") or [],
                             "language": vi.get("language"), "pageCount": vi.get("pageCount"), "publisher": vi.get("publisher"),
                             "publishedDate": vi.get("publishedDate"), "industryIdentifiers": vi.get("industryIdentifiers") or [],
                             "printType": vi.get("printType"), "dimensions": parse_dimensions(vi.get("dimensions"))}}


class GoogleBooks:
    source = "google_books"
    use_query_variants = False          # plain-term search is already fuzzy: one query per reading on the complete-title path (quota)

    def __init__(self, cache_dir, *, online=False, transport=None, api_key=None, shared_pacing=None):
        self.cache = DiskCache(Path(cache_dir))
        self.online = online
        self.api_key = api_key if api_key is not None else os.environ.get("GOOGLE_BOOKS_API_KEY", "").strip() or None
        self.client = httpx.Client(timeout=20, follow_redirects=False, transport=transport, headers={"User-Agent": "shelf-sweep/0.1 (development catalogue lookup)"})
        self.requests = 0
        self.last_request = 0.0
        self.metrics: list[dict[str, Any]] = []
        self.shared_pacing = (transport is None) if shared_pacing is None else bool(shared_pacing)
        self.min_interval = min_interval_s()

    def close(self):
        self.client.close()

    def _pace(self):
        if self.shared_pacing:
            with _PACE:
                wait = max(0.0, self.min_interval - (time.monotonic() - _PACE_LAST[0]))
                if wait:
                    time.sleep(wait)
                _PACE_LAST[0] = self.last_request = time.monotonic()
            return
        wait = max(0.0, self.min_interval - (time.monotonic() - self.last_request))
        if wait:
            time.sleep(wait)
        self.last_request = time.monotonic()

    def _get(self, q: str, limit: int, language: str | None = None) -> dict[str, Any]:
        params = {"q": q, "maxResults": max(1, min(40, int(limit))), "printType": "books", "fields": FIELDS}
        if language in ("es", "en"):
            params["langRestrict"] = language
        key = make_key("googlebooks.http.v1", params=params)            # the API key is never part of the cache key or of any stored URL
        cached = self.cache.get(key)
        if cached is not None:
            self.metrics.append({"cache_hit": True, "url": cached["url"], "latency_s": 0})
            return cached
        if not self.online:
            raise RuntimeError("catalogue_offline_cache_miss")
        self._pace()
        started = time.perf_counter()
        self.requests += 1
        response = self.client.get(BASE, params={**params, **({"key": self.api_key} if self.api_key else {})})
        if response.status_code == 429:
            raise RuntimeError("google_books_quota_exceeded" + ("" if self.api_key else "_anonymous_set_GOOGLE_BOOKS_API_KEY"))
        response.raise_for_status()
        public = str(httpx.URL(BASE, params=params))
        value = {"payload": response.json(), "url": public, "retrieved_at": datetime.now(timezone.utc).isoformat()}
        self.cache.put(key, value)
        self.metrics.append({"cache_hit": False, "url": public, "latency_s": time.perf_counter() - started})
        return value

    def search(self, title, author=None, limit=20, language=None):
        """Title (+ author) search in Open Library's answer shape: payload.docs (one per volume), payload.numFound. Plain terms only: on the project behind the user's
        key the field operators (intitle:/inauthor:/isbn:) answer totalItems 0 or 503 (probed 2026-10-07), so retrieval is broad and the resolver's evidence rule filters."""
        q = re.sub(r'["\s]+', " ", f"{title} {author or ''}").strip()
        res = self._get(q, limit, language)
        items = res["payload"].get("items") or []
        # Google's totalItems is a loose estimate (hundreds for any plain query), so Open Library's completeness test (numFound > returned = truncated) would refuse
        # every answer. Like the fragment rule's default (a relevance-ranked answer capped by the request limit is accepted, because the rule still needs every title
        # word visible plus author evidence and uniqueness among what was returned), the returned volumes are the candidate set; the estimate is kept for the record.
        return {"payload": {"docs": [_doc(i) for i in items], "numFound": len(items), "total_estimate": res["payload"].get("totalItems")},
                "url": res["url"], "retrieved_at": res["retrieved_at"]}

    def volume(self, volume_id: str) -> dict[str, Any]:
        """One volume's record (dimensions are only in the volume record, never in search results). Cached; same pacing and quota handling."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,16}", volume_id or ""):
            raise ValueError("Invalid volume id")
        key = make_key("googlebooks.volume.v1", volume_id=volume_id)
        cached = self.cache.get(key)
        if cached is not None:
            self.metrics.append({"cache_hit": True, "url": cached["url"], "latency_s": 0})
            return cached
        if not self.online:
            raise RuntimeError("catalogue_offline_cache_miss")
        self._pace()
        started = time.perf_counter()
        self.requests += 1
        url = f"{BASE}/{volume_id}"
        response = self.client.get(url, params={"fields": "id,volumeInfo(title,authors,language,pageCount,dimensions,publisher,publishedDate,industryIdentifiers)",
                                                **({"key": self.api_key} if self.api_key else {})})
        if response.status_code == 429:
            raise RuntimeError("google_books_quota_exceeded")
        response.raise_for_status()
        value = {"payload": response.json(), "url": volume_url(volume_id), "retrieved_at": datetime.now(timezone.utc).isoformat()}
        self.cache.put(key, value)
        self.metrics.append({"cache_hit": False, "url": value["url"], "latency_s": time.perf_counter() - started})
        return value

    def dimensions(self, volume_id: str) -> dict[str, Any] | None:
        """Physical dimensions recorded for the volume (cm), with the citation; None when the volume records none."""
        rec = self.volume(volume_id)
        dims = parse_dimensions((rec["payload"].get("volumeInfo") or {}).get("dimensions"))
        return None if dims is None else {**dims, "source_url": rec["url"], "retrieved_at": rec["retrieved_at"]}

    def edition_by_isbn(self, isbn):
        raise NotImplementedError("google_books_isbn_edition_not_used_for_identity")

    def search_fragments(self, terms, limit_all=10, limit_single=40, singles="empty_first"):
        """Fragment retrieval with the same contract as OpenLibrary.search_fragments (all terms first, then the 3 longest terms alone when that returned nothing)."""
        terms = [t for t in terms if t]
        queries = list(dict.fromkeys([" ".join(terms)] + terms[:3]))
        candidates, seen, used, errors = [], set(), [], []
        total = returned = None
        for i, q in enumerate(queries):
            if i and singles == "empty_first" and returned:
                break
            try:
                res = self._get(q, limit_all if " " in q else limit_single)
            except Exception as exc:
                if i == 0:
                    raise
                errors.append({"q": q, "error": f"{type(exc).__name__}:{exc}"})
                continue
            items = res["payload"].get("items") or []
            if i == 0:
                total, returned = res["payload"].get("totalItems"), len(items)
            used.append({"q": q, "url": res["url"], "retrieved_at": res["retrieved_at"], "catalogue": "google_books"})
            for item in items:
                d = _doc(item)
                if d["key"] in seen or not KEY_RE.fullmatch(d["key"]):
                    continue
                seen.add(d["key"])
                titles = [t for t in [d["title"]] if t]
                candidates.append({"work": d["key"], "titles": titles, "authors": d["author_name"], "source": "google_books", "url": d["url"], "google_books": d["google_books"]})
        # Google Books totals are estimates far above what it returns; the resolver's completeness test is Open Library's, so report the returned count as the total
        # only when the first query returned fewer items than asked (then the answer is complete), else the estimate (the resolver then accepts only a capped answer).
        if total is not None and returned is not None and returned < (limit_all if " " in queries[0] else limit_single):
            total = returned
        return {"candidates": candidates, "queries": used, "errors": errors, "total": total, "returned": returned, "limit": limit_all}
