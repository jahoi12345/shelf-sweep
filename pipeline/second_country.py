"""The same 10 books priced for a second country (shows that the locale is a setting).

    compute(run_dir, ...)  -> <run>/second_country.json     (network: AbeBooks-family marketplace pages, cache first, capped; run by the post-sweep finish step)
    apply(packet_data, art) -> packet_data['second_country'] (re-bound to the current book lines; totals computed here from the lines)
    verify(section, packet_data, run_dir) -> problems         (every line re-validated against its raw source bytes)

Which books: the first 10 book lines (by display_index) that are identified AND carry a sourced price in the packet's own market (US/USD). For each one the SAME
ISBN as its US line is looked up on the second country's marketplace (sellers in that country, native currency): the lowest validated new listing is the
replacement cost, the lowest validated used listing the used value, exactly the rules of the US lines (pipeline.price.validate_offer for that market; listing
language and physical format must match the US line's). Nothing is converted, nothing comes from a language model; a book without a validated listing in the
second country stays null with the reason. Amounts are asking prices with their URL, retrieval date and raw-page sha256.

The country is a SETTING: ``config['second_country']`` or ``LIBRARY_SECOND_COUNTRY`` (default GB = AbeBooks UK, GBP; ES = Iberlibro, EUR; ``off`` disables).
Requests are capped (``LIBRARY_SECOND_COUNTRY_MAX_REQUESTS``, default 24 HTTP fetches; cache hits are free) and spaced like the pricing stage.
"""
from __future__ import annotations

import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from pipeline import contracts as C

ARTIFACT = "second_country.json"
N_BOOKS = 10
DEFAULT_COUNTRY = "GB"
DEFAULT_MAX_REQUESTS = 24
EVIDENCE_DIR = "pricing_evidence/second_country"
RULE = ("the first 10 book lines (by display_index) that are identified and carry a sourced US/USD price; the SAME ISBN as the US line, looked up on the second "
        "country's marketplace (sellers in that country, native currency); lowest validated new listing = replacement cost, lowest validated used listing = used "
        "value; listing language and physical format must match the US line; never converted, never estimated")
NOTE = ("Second-country comparison for the same books: native-currency asking prices from sellers in that country (shipping/tax excluded), NOT part of the "
        "US/USD totals. A null means no validated listing of that ISBN in that country (reason on the line), not zero.")


def setting(config: dict[str, Any] | None = None, environ: dict[str, str] | None = None) -> tuple[str | None, str]:
    """(country code or None when off, where the setting came from)."""
    from pipeline.price import CONVERTED_MARKETS
    env = os.environ if environ is None else environ
    if config and config.get("second_country") is not None:
        raw, src = config["second_country"], "config second_country"
    elif env.get("LIBRARY_SECOND_COUNTRY") is not None:
        raw, src = env["LIBRARY_SECOND_COUNTRY"], "env LIBRARY_SECOND_COUNTRY"
    else:
        raw, src = DEFAULT_COUNTRY, "default"
    code = str(raw).strip().upper()
    if code in ("", "0", "OFF", "NONE", "FALSE"):
        return None, src
    return (code if code in CONVERTED_MARKETS else None), src


def max_requests(config: dict[str, Any] | None = None, environ: dict[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw = (config or {}).get("second_country_max_requests", env.get("LIBRARY_SECOND_COUNTRY_MAX_REQUESTS", DEFAULT_MAX_REQUESTS))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_REQUESTS


def _us_offer(line: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("replacement_cost", "used_value"):
        o = line.get(key)
        if isinstance(o, dict) and o.get("amount") is not None and o.get("isbn"):
            return o
    return None


def pick_books(data: dict[str, Any], n: int = N_BOOKS) -> list[dict[str, Any]]:
    """The first n identified book lines with a sourced price in the packet's market (display order)."""
    lines = [b for b in data.get("books") or [] if b.get("status") == "identified" and _us_offer(b)]
    lines.sort(key=lambda b: (b.get("display_index") is None, b.get("display_index") or 0, str(b.get("id"))))
    return lines[:n]


def _price_row(data: dict[str, Any], line: dict[str, Any]) -> dict[str, Any]:
    ids = [line["id"]] + list(line.get("member_group_ids") or [])
    rows = {r.get("ref_id"): r for r in ((data.get("pricing") or {}).get("prices") or [])}
    return next((rows[i] for i in ids if i in rows), {})


def _money(o: Any) -> dict[str, Any] | None:
    if not isinstance(o, dict) or o.get("amount") is None:
        return None
    return {k: o.get(k) for k in ("amount", "currency", "converted", "original_amount", "original_currency", "source", "url", "retrieved_at", "condition_assumed", "isbn")}


def compute(run_dir: str | Path, *, country: str | None = None, online: bool = False, cache_dir: str | Path | None = None, cap: int = DEFAULT_MAX_REQUESTS,
            setting_source: str = "argument", provider: Any = None, n: int = N_BOOKS, min_interval_s: float = 2.0) -> dict[str, Any]:
    """Look up the picked books in the second country and write <run>/second_country.json. Reads claim_packet.json (never writes it)."""
    from pipeline.price import CONVERTED_MARKETS, _ABE_LANG, market_offers, validate_offer
    from providers.book_prices import PublicBookPrices
    root = Path(run_dir).resolve()
    data = C.read_json(root / C.ARTIFACTS["packet"])["data"]
    spec = CONVERTED_MARKETS[country]
    provider = provider or PublicBookPrices(root / EVIDENCE_DIR, online=online, cache_dir=Path(cache_dir or root / "cache") / "abebooks",
                                            max_age_hours=24 * 30, min_interval_s=min_interval_s)
    t0 = time.perf_counter()
    rows, memo = [], {}
    attempts = lambda: sum((getattr(provider, "http_attempts", None) or {}).values())
    for line in pick_books(data, n):
        us = _us_offer(line)
        isbn, fmt = us["isbn"], us.get("format")
        lang = _price_row(data, line).get("language") or (line.get("language") if line.get("language") in _ABE_LANG else None)
        row = {"ref_id": line["id"], "display_index": line.get("display_index"), "title": line.get("title"), "author": line.get("author"), "isbn": isbn,
               "format": fmt, "language": lang, "replacement_cost": None, "used_value": None, "issues": [], "lookup": None}
        if isbn not in memo:
            if online and attempts() + len(spec["urls"]) > cap:
                memo[isbn] = None
            else:
                found, failures = market_offers(provider, isbn, [u.format(isbn=isbn) for u in spec["urls"]])
                valid, rejects = [], []
                for offer in found:
                    offer = dict(offer)
                    offer["evidence"] = dict(offer.get("evidence") or {})
                    raw = offer["evidence"].get("raw_path")
                    if raw and Path(raw).is_absolute() and Path(raw).resolve().is_relative_to(root):
                        offer["evidence"]["raw_path"] = str(Path(raw).resolve().relative_to(root))
                    errors = validate_offer(offer, None, root, expected_isbn=isbn, market=country)
                    if lang in _ABE_LANG and offer.get("language") != _ABE_LANG[lang]:
                        errors.append("listing_language_not_" + lang)
                    if fmt and offer.get("format") != fmt:
                        errors.append("listing_format_not_" + str(fmt))
                    (rejects if errors else valid).append(offer if not errors else {"listing_id": offer.get("listing_id"), "reasons": errors})
                best = {k: min((o for o in valid if o["kind"] == k), key=lambda o: Decimal(str(o["amount"])), default=None) for k in ("replacement_cost", "used_value")}
                memo[isbn] = {"best": best, "listings_valid": len(valid), "listings_rejected": len(rejects), "failures": failures}
        got = memo[isbn]
        if got is None:
            row["issues"].append(f"not_queried:request_cap_{cap}_reached")
        else:
            row["lookup"] = {k: got[k] for k in ("listings_valid", "listings_rejected", "failures")}
            for k in ("replacement_cost", "used_value"):
                o = got["best"][k]
                if o:
                    row[k] = {**{x: o.get(x) for x in ("kind", "amount", "currency", "market", "isbn", "format", "condition_assumed", "edition_match", "source", "url",
                                                        "retrieved_at", "converted", "proxy_assumptions", "language", "seller_country", "listing_id", "evidence")},
                              "same_isbn_as_us_line": True}
                else:
                    unavailable = got["failures"] and all(f.get("source_unavailable") for f in got["failures"]) and not got["listings_valid"]
                    row["issues"].append(f"{k}:" + ("source_unavailable" if unavailable else f"no_validated_{country}_{spec['currency']}_listing_of_this_isbn"))
        rows.append(row)
    art = C.envelope(root.name, "second_country", {
        "country": country, "currency": spec["currency"], "source": spec["name"], "setting_source": setting_source, "n_books": n, "rule": RULE, "note": NOTE,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "online": bool(online), "books": rows,
        "requests": {"http_attempts": dict(getattr(provider, "http_attempts", None) or {}), "cap": cap,
                     "retrievals": len(getattr(provider, "retrievals", None) or []), "cache_hits": sum(1 for r in getattr(provider, "retrievals", None) or [] if r.get("cache_hit"))},
    }, status="ok" if rows and all(r["replacement_cost"] or r["used_value"] for r in rows) else "partial",
        metrics={"elapsed_s": round(time.perf_counter() - t0, 3), "books": len(rows), "priced": sum(bool(r["replacement_cost"] or r["used_value"]) for r in rows),
                 "actual_API_cost_usd": 0, "cost_provenance": "public marketplace pages (cache first), no model"})
    C.write_json(root / ARTIFACT, art)
    return art


def load(run_dir: str | Path) -> dict[str, Any] | None:
    f = Path(run_dir) / ARTIFACT
    try:
        return C.read_json(f) if f.is_file() else None
    except Exception:  # noqa: BLE001 - an unreadable artifact is treated as absent (the packet then says not_run)
        return None


def _sum(rows: list[dict[str, Any]], getter) -> float | None:
    vals = [Decimal(str(getter(r))) for r in rows if getter(r) is not None]
    return float(sum(vals, Decimal(0)).quantize(Decimal(".01"))) if vals else None


def apply(data: dict[str, Any], art: dict[str, Any] | None, run_dir: str | Path | None = None) -> dict[str, Any]:
    """Set data['second_country'] from the artifact, re-bound to the CURRENT book lines (a line whose id or US ISBN changed since is listed as stale and excluded)."""
    if not art:
        code, src = setting()
        if isinstance(data.get("second_country"), dict) and data["second_country"].get("status") not in (None, "not_run", "off"):
            return data["second_country"]            # computed earlier and attached; the artifact is not next to this packet (e.g. a copied packet)
        data["second_country"] = {"status": "off" if code is None else "not_run", "country": code, "setting_source": src, "books": [], "totals": None, "rule": RULE,
                                  "note": ("off by setting" if code is None else
                                           "computed after the sweep by the post-sweep finish step (app.finish_reading); not run for this packet yet")}
        return data["second_country"]
    a = art["data"]
    by_id = {}
    for b in data.get("books") or []:
        by_id.setdefault(b["id"], b)
        for m in b.get("member_group_ids") or []:
            by_id.setdefault(m, b)
    books = []
    for r in a.get("books") or []:
        line = by_id.get(r["ref_id"])
        us = _us_offer(line) if line else None
        stale = None if line and us and us.get("isbn") == r["isbn"] else ("book line no longer in the packet" if not line else "the line's US price / ISBN changed since")
        books.append({"ref_id": line["id"] if line else r["ref_id"], "display_index": (line or {}).get("display_index", r.get("display_index")),
                      "title": r.get("title"), "author": r.get("author"), "isbn": r["isbn"], "stale": stale,
                      "us": {"replacement_cost": _money((line or {}).get("replacement_cost")), "used_value": _money((line or {}).get("used_value"))},
                      "second": {"replacement_cost": r.get("replacement_cost"), "used_value": r.get("used_value")}, "issues": r.get("issues") or []})
    live = [b for b in books if not b["stale"]]
    cur = a["currency"]
    totals = {"books": len(live), "currency": cur,
              "replacement_cost": _sum(live, lambda b: (b["second"]["replacement_cost"] or {}).get("amount")),
              "used_value": _sum(live, lambda b: (b["second"]["used_value"] or {}).get("amount")),
              "priced_replacement": sum(bool(b["second"]["replacement_cost"]) for b in live), "priced_used": sum(bool(b["second"]["used_value"]) for b in live),
              "us_replacement_cost_same_books_usd": _sum(live, lambda b: (b["us"]["replacement_cost"] or {}).get("amount")),
              "us_used_value_same_books_usd": _sum(live, lambda b: (b["us"]["used_value"] or {}).get("amount")),
              "note": f"native {cur} sums over the listed (non-stale) books that have a {a['country']} listing; the USD sums are the same books' US lines; never mixed"}
    complete = len(live) == len(books) and len(live) > 0 and all(b["second"]["replacement_cost"] or b["second"]["used_value"] for b in live)
    data["second_country"] = {"status": "ok" if complete else "partial", "country": a["country"], "currency": cur, "source": a.get("source"),
                              "setting_source": a.get("setting_source"), "rule": a.get("rule"), "note": a.get("note"), "computed_at": a.get("computed_at"),
                              "requests": a.get("requests"), "artifact": ARTIFACT, "books": books, "totals": totals}
    return data["second_country"]


def review_entries(sc: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(sc, dict) or sc.get("status") in ("ok", "off"):
        return []
    if sc.get("status") == "not_run":
        return [{"ref_id": "second_country", "code": "second_country_not_run", "severity": "warning",
                 "reason": f"Second-country pricing ({sc.get('country')}) of the same 10 books has not run yet (post-sweep finish step)."}]
    books = sc.get("books") or []
    none = [b for b in books if not b.get("stale") and not (b["second"].get("replacement_cost") or b["second"].get("used_value"))]
    stale = [b for b in books if b.get("stale")]
    bits = ([f"{len(none)} of {len(books)} books have no validated {sc.get('country')} listing of the same ISBN (null, not zero)"] if none else []) + \
           ([f"{len(stale)} line(s) changed since the lookup (stale, excluded)"] if stale else []) + \
           ([f"only {len(books)} identified book(s) with a US price exist, fewer than the {N_BOOKS} asked for"] if 0 < len(books) < N_BOOKS else []) + \
           ([] if books else ["no identified, priced book to compare"])
    return [{"ref_id": "second_country", "code": "second_country_partial", "severity": "warning",
             "reason": f"Second-country comparison ({sc.get('country')}/{sc.get('currency')}) partial: " + "; ".join(bits) + "."}]


def verify(sc: dict[str, Any], data: dict[str, Any], run_dir: str | Path) -> list[str]:
    """Every second-country line validated against its raw source bytes under that country's rules; totals = sums of the lines; at most N_BOOKS books."""
    from pipeline.price import validate_offer
    out = []
    if sc.get("status") in ("not_run", "off"):
        return [] if not sc.get("books") else ["second_country: books listed although it did not run"]
    if len(sc.get("books") or []) > N_BOOKS:
        out.append("second_country: more than 10 books")
    for b in sc.get("books") or []:
        for k in ("replacement_cost", "used_value"):
            o = (b.get("second") or {}).get(k)
            if not o:
                continue
            errs = validate_offer(o, None, run_dir, expected_isbn=b["isbn"], market=sc["country"])
            if o.get("kind") != k or o.get("currency") != sc["currency"]:
                errs.append("kind/currency mismatch")
            out += [f"second_country {b['ref_id']} {k}: {e}" for e in errs]
    live = [b for b in sc.get("books") or [] if not b.get("stale")]
    t = sc.get("totals") or {}
    for k in ("replacement_cost", "used_value"):
        if t.get(k) != _sum(live, lambda b: ((b.get("second") or {}).get(k) or {}).get("amount")):
            out.append(f"second_country.totals.{k} != sum of its lines")
    return out


def run_step(run_dir: str | Path, *, lock: Any = None, config: dict[str, Any] | None = None, render: bool = False) -> dict[str, Any]:
    """The post-sweep step (app.finish_reading): compute (network, capped, outside the packet lock), then attach under the lock (finalize + verify + write)."""
    import contextlib
    from pipeline.price import price_config_from_env
    run = Path(run_dir).resolve()
    code, src = setting(config)
    if code is None:
        return {"state": "skipped", "reason": f"second country off or unsupported ({src})"}
    pc = price_config_from_env(run)
    cfg = config or {}
    art = compute(run, country=code, online=bool(cfg.get("pricing_online", pc["pricing_online"])), cache_dir=cfg.get("price_cache_dir", pc["price_cache_dir"]),
                  cap=max_requests(config), setting_source=src)
    with (lock if lock is not None else contextlib.nullcontext()):
        att = attach(run, render=render)
    return {"state": "done" if art["status"] == "ok" else "partial", "country": code, "priced": art["metrics"]["priced"], "books": art["metrics"]["books"],
            "requests": art["data"]["requests"], "attach": att}


def attach(run_dir: str | Path, *, render: bool = True) -> dict[str, Any]:
    from pipeline import packet as P
    run = Path(run_dir)
    pk = run / C.ARTIFACTS["packet"]
    if not pk.is_file():
        return {"attached": False, "reason": "no claim_packet.json"}
    env = C.read_json(pk)
    P.finalize_contract(env["data"], run)
    problems = P.verify_packet(env, C.read_json(run / C.ARTIFACTS["frames"]), run)
    if problems:
        return {"attached": False, "reason": "packet verification failed after the second-country section was attached", "problems": problems[:10]}
    C.write_json(pk, env)
    if render:
        from pipeline import report as R
        R.render(pk, run)
    sc = env["data"].get("second_country") or {}
    return {"attached": True, "status": sc.get("status"), "totals": sc.get("totals")}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path, help="a finished run directory (use a scratch copy to keep the original untouched)")
    ap.add_argument("--country", help="GB or ES (default: LIBRARY_SECOND_COUNTRY or GB)")
    ap.add_argument("--online", action="store_true", help="allow HTTP (capped); default cache only")
    ap.add_argument("--cache-dir")
    ap.add_argument("--max-requests", type=int)
    a = ap.parse_args(argv)
    cfg = {"pricing_online": a.online, **({"second_country": a.country} if a.country else {}), **({"price_cache_dir": a.cache_dir} if a.cache_dir else {}),
           **({"second_country_max_requests": a.max_requests} if a.max_requests is not None else {})}
    print(json.dumps(run_step(a.run, config=cfg, render=True), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
