"""Retrievable catalogue metadata, using the shared provider cache. Never prices.

API: https://openlibrary.org/dev/docs/api/search (verified 2026-10-06).
Search results describe works; aggregated ISBN/publisher fields are NOT editions.
"""
from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from providers.cache import DiskCache, make_key


# Process-wide pacing for REAL network clients: the live engine builds one OpenLibrary per resolve call and runs several in parallel, so a per-instance
# 1.05 s spacing would multiply the request rate by the number of workers. Clients with an injected transport (tests, mocks) keep their own spacing.
_PACE = threading.Lock()
_PACE_LAST = [0.]
DEFAULT_MIN_INTERVAL_S = 1.05
# Fragment retrieval: "empty_first" asks the three single-term fallbacks only when the all-terms query returned nothing (-43% fragment requests on the YT2 Gemini readings,
# same accepted count: runs/w15_yt2_gemini_online_v1); "always" restores the older behaviour.
FRAGMENT_SINGLES = "empty_first"


def min_interval_s(environ=None):
    """Spacing between uncached catalogue requests. Open Library asks for <= 1 request/s from anonymous clients; LIBRARY_OL_MIN_INTERVAL_S overrides it
    (only set it below 1.05 for a client that identifies itself to Open Library with a contact in the User-Agent: LIBRARY_OL_CONTACT)."""
    try:
        return max(0.1, float((environ if environ is not None else os.environ).get("LIBRARY_OL_MIN_INTERVAL_S", DEFAULT_MIN_INTERVAL_S)))
    except ValueError:
        return DEFAULT_MIN_INTERVAL_S


class OpenLibrary:
    def __init__(self, cache_dir, *, online=False, transport=None, shared_pacing=None):
        self.cache = DiskCache(Path(cache_dir))
        self.online = online
        agent = "shelf-sweep/0.1 (development catalogue lookup"
        contact = os.environ.get("LIBRARY_OL_CONTACT", "").strip()
        self.client = httpx.Client(timeout=20, follow_redirects=False, transport=transport,
                                   headers={"User-Agent": agent + (f"; {contact}" if contact else "") + ")"})
        self.requests = 0
        self.last_request = 0.
        self.metrics = []
        self.shared_pacing = (transport is None) if shared_pacing is None else bool(shared_pacing)
        self.min_interval = min_interval_s()

    def _pace(self):
        """Sleep until this request may start: >= min_interval after the previous uncached request (process-wide for real clients, per instance otherwise)."""
        if self.shared_pacing:
            with _PACE:
                wait = max(0., self.min_interval - (time.monotonic() - _PACE_LAST[0]))
                if wait:
                    time.sleep(wait)
                _PACE_LAST[0] = self.last_request = time.monotonic()
            return
        wait = max(0., self.min_interval - (time.monotonic() - self.last_request))
        if wait:
            time.sleep(wait)
        self.last_request = time.monotonic()

    def close(self):
        self.client.close()

    def _get(self, route, params=None):
        key = make_key("openlibrary.http.v1", route=route, params=params or {})
        cached = self.cache.get(key)
        if cached is not None:
            self.metrics.append({"cache_hit": True, "url": cached["url"], "latency_s": 0})
            return cached
        if not self.online:
            raise RuntimeError("catalogue_offline_cache_miss")
        self._pace()
        started = time.perf_counter()
        self.requests += 1
        response = self.client.get("https://openlibrary.org"+route, params=params)
        response.raise_for_status()
        value = {"payload": response.json(), "url": str(response.url),
                 "retrieved_at": datetime.now(timezone.utc).isoformat()}
        self.cache.put(key, value)
        self.metrics.append({"cache_hit": False, "url": value["url"],
                             "latency_s": time.perf_counter()-started})
        return value

    def search(self, title, author=None, limit=20, language=None):
        params = {"title": title, "limit": limit, "fields": "key,title,author_name,editions,editions.key,editions.title,editions.language"}
        if author:
            params["author"] = author
        if language in {"es", "en"}:
            params["lang"] = language  # relevance preference only, never inferred edition identity
        return self._get("/search.json", params)

    def edition_by_isbn(self, isbn):
        # ISBN passed here has already been restricted to checksum-valid digits/X.
        if not isbn or any(c not in "0123456789X" for c in isbn):
            raise ValueError("Invalid ISBN URL component")
        return self._get(f"/isbn/{isbn}.json")

    def editions(self, work, limit=100, max_pages=3):
        """Edition records of one work (language, physical_format, publish_date, isbn_10/13). Cached/rate-limited like every request; bounded to ``max_pages`` pages.
        -> {"entries": [...], "size": total editions the catalogue reports, "urls": [...], "retrieved_at": ...}."""
        if not re.fullmatch(r"/works/OL\d+W", work or ""):
            raise ValueError("Invalid work key")
        entries, urls, size, retrieved = [], [], None, None
        for page in range(max_pages):
            res = self._get(f"{work}/editions.json", {"limit": limit, "offset": page * limit})
            payload = res["payload"]
            entries += payload.get("entries", [])
            urls.append(res["url"]); retrieved = retrieved or res["retrieved_at"]
            size = payload.get("size", len(entries))
            if len(entries) >= size or not payload.get("entries"):
                break
        return {"entries": entries, "size": size, "urls": urls, "retrieved_at": retrieved}

    def search_fragments(self, terms, limit_all=10, limit_single=40, singles=None):
        """Retrieval for fragment resolution. One query with all terms, then each of the 3 longest terms alone, because an OCR fragment that is a truncated
        word breaks an AND query while a rare surname alone still retrieves the author's works. Every request goes through ``_get`` (disk cache, 1.05 s spacing,
        offline => RuntimeError('catalogue_offline_cache_miss')). The first query must succeed (it raises otherwise); a later failure is NOT hidden: it is returned in
        ``errors`` and the caller must abstain. ``total``/``returned``/``limit`` describe the first (all-terms) query so the caller can judge whether uniqueness is knowable."""
        terms = [t for t in terms if t]
        singles = singles or os.environ.get("LIBRARY_FRAGMENT_SINGLES", FRAGMENT_SINGLES)
        queries = list(dict.fromkeys([" ".join(terms)] + terms[:3]))
        fields = "key,title,author_name,editions,editions.key,editions.title,editions.language"
        candidates, seen, used, errors = [], set(), [], []
        total = returned = None
        for i, q in enumerate(queries):
            if i and singles == "empty_first" and returned:
                break          # the all-terms query already returned candidates: the single-term fallbacks (for truncated OCR words that empty an AND query) are not needed
            try:
                res = self._get("/search.json", {"q": q, "limit": limit_all if " " in q else limit_single, "fields": fields})
            except Exception as exc:
                if i == 0:
                    raise
                errors.append({"q": q, "error": f"{type(exc).__name__}:{exc}"})
                continue
            if i == 0:
                total, returned = res["payload"].get("numFound", res["payload"].get("num_found")), len(res["payload"].get("docs", []))
            used.append({"q": q, "url": res["url"], "retrieved_at": res["retrieved_at"]})
            for doc in res["payload"].get("docs", []):
                key = doc.get("key", "")
                if key.startswith("OL"):
                    key = "/works/" + key
                if key in seen:
                    continue
                seen.add(key)
                titles = [doc.get("title")] + [e.get("title") for e in (doc.get("editions") or {}).get("docs", [])]
                candidates.append({"work": key, "titles": [t for t in titles if t], "authors": doc.get("author_name", []), "source": "openlibrary",
                                   "url": "https://openlibrary.org" + key if re.fullmatch(r"/works/OL\d+W", key) else None})
        return {"candidates": candidates, "queries": used, "errors": errors, "total": total, "returned": returned,
                "limit": limit_all}

