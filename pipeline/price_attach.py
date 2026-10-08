"""Price the identified groups of a FINISHED run in a scratch directory, then attach the validated prices to the run's claim_packet.json in place.

    python -m pipeline.price_attach compute --run runs/<run> --readings <readings.json> --out runs/<new scratch dir> [--online] [--cache-dir DIR]
                                            [--no-reading-language] [--no-converted] [--markets ES,GB]
    python -m pipeline.price_attach attach  --run runs/<run> --prices runs/<scratch dir>/prices.json [--no-render]

``compute`` never writes into ``--run``: it reads the run's packet (every identified catalogue-work track group, including the members of merged instance lines and the
unconfirmed lines), takes each group's spine readings from ``--readings`` (the readings that produced the identity; their visible text and language_code /
language_evidence feed the language rule of pipeline.price as the LAST fallback), prices them through pipeline.price (work-level equivalent edition; US/USD first,
converted non-US fallback) and writes ``<out>/prices.json`` + the raw evidence under ``<out>/pricing_evidence``. Request counts and wall time are in its metrics.

``attach`` follows pipeline.identity_attach / pipeline.auto_attach: the original packet is kept once as ``claim_packet.pre_prices.json``; the raw evidence each line cites is
copied into the run (content-addressed under pricing_evidence/), each line is re-validated against those bytes, the instance layer (instances.json) is re-applied on top,
the totals are recomputed from the lines, and the patched packet must pass pipeline.packet.verify_packet (nothing is written otherwise). Attaching twice gives the same
packet. A group that already had a validated price keeps it when the new computation found none (zero loss). Lines flagged for appraisal or with a user title
correction are left alone. Prices are asking prices of WORK-LEVEL EQUIVALENT editions, not of the shelf copies.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

BACKUP = "claim_packet.pre_prices.json"
PRICE_FIELDS = ("replacement_cost", "used_value", "price_basis", "equivalent_isbn", "equivalent_edition", "price_basis_note", "pricing_issues", "pricing_null_reasons",
                "pricing_attempted", "second_market")
KINDS = ("replacement_cost", "used_value")


def _members(p: dict[str, Any]) -> list[dict[str, Any]]:
    """Track-group lines: plain lines and the members of merged instance lines, from books and unconfirmed_books (the objects themselves, for patching)."""
    return [m for line in list(p.get("books") or []) + list(p.get("unconfirmed_books") or []) for m in (line.get("merged_members") or [line])]


def _readings(path: str | Path) -> dict[str, dict[str, Any]]:
    doc = json.loads(Path(path).read_text())
    rows = (doc.get("data") or {}).get("readings") if isinstance(doc, dict) else doc
    return {r["observation_id"]: r for r in rows or [] if isinstance(r, dict) and r.get("observation_id")}


def price_inputs(p: dict[str, Any], readings: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One pricing input per identified catalogue-match track group: the identity fields of the line, its own spine readings (visible text = the longest reading's)."""
    out, seen = [], set()
    for m in _members(p):
        if m["id"] in seen or m.get("status") != "identified" or m.get("identity_source") != "catalogue_match":
            continue
        if any((c or {}).get("kind") == "set_title" for c in m.get("user_corrections") or []):
            continue
        seen.add(m["id"])
        obs = (m.get("evidence") or {}).get("observation_ids") or []
        rows = [readings[o] for o in obs if o in readings]
        text = max((r.get("visible_text") or "" for r in rows), key=len, default="") or m.get("visible_text")
        out.append({k: copy.deepcopy(m.get(k)) for k in ("id", "status", "identity_source", "identity_source_url", "title", "author", "isbn", "edition", "identity_matching_evidence")}
                   | {"visible_text": text, "language": None,
                      "reading_evidence": [{"observation_id": r.get("observation_id"), "language_code": r.get("language_code"), "language_evidence": r.get("language_evidence")}
                                           for r in rows]})
    return out


_LANG3 = {"es": "spa", "en": "eng"}


def volume_entry(gb: dict[str, Any]) -> dict[str, Any] | None:
    """A Google Books volume record (as saved on the resolution that accepted it) in Open Library edition shape: its ISBNs, language and date; format unknown
    (the listing must then confirm the paperback family)."""
    if not isinstance(gb, dict) or not gb.get("volume_id"):
        return None
    ids = {i.get("type"): i.get("identifier") for i in gb.get("industryIdentifiers") or [] if isinstance(i, dict)}
    lang = _LANG3.get(str(gb.get("language") or "")[:2])
    return {"key": "gb:" + gb["volume_id"], "title": gb.get("title"), "publish_date": gb.get("publishedDate"), "physical_format": None,
            "languages": [{"key": "/languages/" + lang}] if lang else [], "publishers": [gb["publisher"]] if gb.get("publisher") else [],
            **({"isbn_13": [ids["ISBN_13"]]} if ids.get("ISBN_13") else {}), **({"isbn_10": [ids["ISBN_10"]]} if ids.get("ISBN_10") else {})}


class VolumeCatalogue:
    """Edition records for Google-Books-identified works (gb:<id>) from the volume records saved on their resolutions; Open Library works go to ``ol``."""
    def __init__(self, ol, volumes: dict[str, dict[str, Any]]):
        self.ol, self.volumes = ol, volumes

    def editions(self, work):
        if str(work).startswith("gb:"):
            gb = self.volumes.get(work)
            entry = volume_entry(gb)
            if not entry:
                raise ValueError("no saved Google Books volume record for " + str(work))
            return {"entries": [entry], "size": 1, "urls": [gb.get("source_url") or "https://books.google.com/books?id=" + work[3:]], "retrieved_at": gb.get("retrieved_at")}
        return self.ol.editions(work)

    def close(self):
        self.ol.close()


def volumes_from(resolutions_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not resolutions_path:
        return {}
    doc = json.loads(Path(resolutions_path).read_text())
    rows = (doc.get("data") or {}).get("resolutions") if isinstance(doc, dict) else doc
    out = {}
    for r in rows or []:
        gb = r.get("google_books") if isinstance(r, dict) else None
        if r.get("decision") == "accepted" and str(r.get("work") or "").startswith("gb:") and isinstance(gb, dict):
            out.setdefault(r["work"], {**gb, "source_url": r.get("source_url"), "retrieved_at": (r.get("catalogue_query") or {}).get("retrieved_at")})
    return out


def compute(run_dir: str | Path, readings_path: str | Path, out_dir: str | Path, *, online: bool = False, cache_dir: str | Path | None = None,
            reading_language: bool = True, converted: bool = True, markets: str = "ES,GB", throttle_retries: int = 2, throttle_backoff_s: float = 60.0,
            min_interval_s: float = 2.0, resolutions: str | Path | None = None, only_google_books: bool = False) -> dict[str, Any]:
    from pipeline import contracts as C
    from pipeline import price as PR
    from providers.openlibrary import OpenLibrary
    run_dir, out = Path(run_dir), Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("--out must be a new (empty) directory; existing runs are never written into")
    out.mkdir(parents=True, exist_ok=True)
    p = C.read_json(run_dir / C.ARTIFACTS["packet"])["data"]
    books = price_inputs(p, _readings(readings_path))
    if only_google_books:
        books = [b for b in books if str(b.get("identity_source_url") or "").startswith("https://books.google.com/")]
    cache = Path(cache_dir) if cache_dir else out / "price_cache"
    cfg = {"run_id": out.name, "pricing_online": online, "price_budget_s": 10 ** 9, "price_cache_dir": str(cache), "reading_language_fallback": reading_language,
           "converted_fallback": converted, "converted_markets": markets, "second_market": False, "throttle_retries": throttle_retries,
           "throttle_backoff_s": throttle_backoff_s, "min_interval_s": min_interval_s, "max_age_hours": 24 * 30}
    catalogue = VolumeCatalogue(OpenLibrary(cache / "openlibrary", online=online), volumes_from(resolutions))
    provider = PR.PublicBookPrices(out.resolve() / "pricing_evidence", online=online, cache_dir=cache / "abebooks", max_age_hours=cfg["max_age_hours"],
                                   min_interval_s=min_interval_s, throttle_retries=throttle_retries, throttle_backoff_s=throttle_backoff_s)
    t0 = time.perf_counter()
    try:
        art = PR.price_books_with_work_level(books, out, cfg, catalogue=catalogue, wl_provider=provider)
    finally:
        catalogue.close()
    wall = round(time.perf_counter() - t0, 1)
    rows = art["data"]["prices"]
    for r in rows:                                                     # exact rows: the scratch packet binds work-level rows by id + title
        r.setdefault("title", next((b["title"] for b in books if b["id"] == r["ref_id"]), None))
    by_host = Counter()
    for x in provider.retrievals:
        by_host[(x["url"].split("/")[2], "cache" if x["cache_hit"] else "fresh")] += 1
    priced = [r for r in rows if r.get("replacement_cost") or r.get("used_value")]
    conv = lambda r: any((r.get(k) or {}).get("converted") is True for k in KINDS)
    misses = Counter(i.split("(")[0] for r in rows if not (r.get("replacement_cost") or r.get("used_value")) for i in r.get("issues") or [] if isinstance(i, str)
                     and not i.endswith("_source_unavailable"))
    art["data"].update(source_run=str(run_dir), readings=str(readings_path), config={k: v for k, v in cfg.items() if k != "price_budget_s"},
                       converted_policy=PR.CONVERTED_POLICY if converted else None)
    art["metrics"].update(
        wall_s=wall, identified_groups=len(books), distinct_works=len({PR.priced_work_key(b) for b in books}), priced_groups=len(priced),
        priced_groups_native_us_only=sum(1 for r in priced if not conv(r)), priced_groups_with_converted_line=sum(1 for r in priced if conv(r)),
        priced_groups_converted_only=sum(1 for r in priced if all((r.get(k) or {}).get("converted") is True for k in KINDS if r.get(k))),
        lines={k: {"native_usd": sum(1 for r in rows if r.get(k) and r[k].get("converted") is not True), "converted": sum(1 for r in rows if (r.get(k) or {}).get("converted") is True),
                   "converted_by_currency": dict(Counter(r[k]["original_currency"] for r in rows if (r.get(k) or {}).get("converted") is True))} for k in KINDS},
        language_from_reading_fallback=sum(1 for r in rows if str(r.get("language_source") or "").startswith("spine_reading_language_majority")),
        miss_reasons_unpriced_groups=dict(misses.most_common()),
        requests={"open_library_fresh": catalogue.ol.requests, "open_library_cache_hits": sum(1 for m in catalogue.ol.metrics if m.get("cache_hit")),
                  "price_pages": {f"{h}:{kind}": n for (h, kind), n in sorted(by_host.items())}, "price_http_attempts": dict(provider.http_attempts),
                  "price_throttled_answers": len(provider.throttle_events), "throttle_events": provider.throttle_events[:200],
                  "fx_ecb": "1 file per day (cached)"})
    C.write_json(out / "prices.json", art)
    return {k: art["metrics"][k] for k in ("wall_s", "identified_groups", "distinct_works", "priced_groups", "priced_groups_native_us_only", "priced_groups_with_converted_line",
                                             "priced_groups_converted_only", "lines", "language_from_reading_fallback", "miss_reasons_unpriced_groups", "requests")
            } | {"requests": {k: v for k, v in art["metrics"]["requests"].items() if k != "throttle_events"}}


# ------------------------------------------------------------------------------------------------------------------------------------------ attach
def _copy_evidence(row: dict[str, Any], src_root: Path, run_dir: Path) -> dict[str, Any]:
    """Copy the raw bytes the row's lines cite into the run (content-addressed: never overwrites other bytes) and point the lines at the copies."""
    row = copy.deepcopy(row)
    dest = run_dir / "pricing_evidence"
    for k in KINDS:
        line = row.get(k)
        if not isinstance(line, dict):
            continue
        ev = line.get("evidence") or {}
        src = src_root / str(ev.get("raw_path") or "")
        if ev.get("sha256") and src.is_file():
            name = f"pricing_evidence/{ev['sha256']}.html"
            dest.mkdir(parents=True, exist_ok=True)
            if not (run_dir / name).is_file():
                shutil.copy2(src, run_dir / name)
            ev["raw_path"] = name
        fx = line.get("fx") if line.get("converted") is True else None
        if isinstance(fx, dict) and fx.get("raw_path"):
            fsrc = src_root / fx["raw_path"]
            name = f"pricing_evidence/fx_{fx['sha256']}.xml"
            if fsrc.is_file() and not (run_dir / name).is_file():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fsrc, run_dir / name)
            fx["raw_path"] = name
    return row


def apply_row(line: dict[str, Any], row: dict[str, Any], run_dir: Path) -> str:
    """Set one track-group line's price fields from its row (validated against the run's copy of the raw bytes). -> 'priced' | 'kept' | 'null' | 'skipped' | 'unbound'."""
    from pipeline import packet as P
    if line.get("status") == "needs_appraisal" or any((c or {}).get("kind") == "set_title" for c in line.get("user_corrections") or []):
        return "skipped"
    if row.get("price_basis") == P.WORK_LEVEL and row.get("title") != line.get("title"):
        return "unbound"
    want = (row.get("equivalent_edition") or {}).get("isbn") if row.get("price_basis") == P.WORK_LEVEL else None
    fresh = {k: row[k] for k in KINDS if isinstance(row.get(k), dict) and row[k].get("kind") == k and (want or row.get("price_basis") != P.WORK_LEVEL)
             and not P._validate_price_line(row[k], line, str(run_dir), want)}
    had = any(isinstance(line.get(k), dict) for k in KINDS)
    if not fresh and had:
        return "kept"                                                   # zero loss: an earlier validated price stays when the new computation found none
    for k in PRICE_FIELDS:
        line.pop(k, None)
    line["pricing_attempted"] = True
    line["pricing_issues"] = row.get("issues", [])
    line.update(replacement_cost=fresh.get("replacement_cost"), used_value=fresh.get("used_value"))
    if row.get("price_basis") == P.WORK_LEVEL:
        line.update(price_basis=P.WORK_LEVEL, equivalent_isbn=want, equivalent_edition=row.get("equivalent_edition"), price_basis_note=P.WORK_LEVEL_NOTE)
    nulls = {k: P._null_reason(line, k) for k in KINDS if line.get(k) is None}
    if nulls:
        line["pricing_null_reasons"] = nulls
    return "priced" if fresh else "null"


def _refresh(p: dict[str, Any]) -> None:
    from pipeline import packet as P
    t = p["totals"]
    fresh = P.compute_totals(p["books"], p.get("items") or [], {}, len(p.get("excluded_by_user") or []), p.get("pricing") is not None)
    for k in ("books_identified", "books_unidentified", "books_needs_appraisal", "books_identified_by_user_statement", "books_replacement_cost", "books_used_value", "excluded_from_totals"):
        t[k] = fresh[k]
    notes = t.setdefault("notes", {})
    if any((b.get(k) or {}).get("converted") is True for b in p["books"] for k in KINDS):
        notes["converted_lines"] = P.CONVERTED_TOTALS_NOTE + " Counts: completeness.converted_lines."
    else:
        notes.pop("converted_lines", None)
    if any(b.get("price_basis") == P.WORK_LEVEL for b in p["books"]):
        notes["price_basis"] = ("Money totals sum WORK-LEVEL EQUIVALENT prices (one stated equivalent edition per work, ISBN on each line); "
                                "they are not exact-edition prices of the shelf copies.")
    p["language_breakdown"] = P.language_breakdown(p["books"])
    p["completeness"] = P.completeness(p["books"], p.get("items") or [], p.get("room") or {}, t)


def apply(p: dict[str, Any], prices: dict[str, Any], src_root: Path, run_dir: Path, instances_art: dict[str, Any] | None = None) -> dict[str, Any]:
    """Patch packet ``data`` in place from a compute() prices artifact. Returns counts per outcome (over track groups)."""
    new_rows = {r["ref_id"]: _copy_evidence(r, src_root, run_dir) for r in prices["data"]["prices"]}
    old_rows = {r["ref_id"]: r for r in ((p.get("pricing") or {}).get("prices") or [])}
    out = Counter()
    used_new = set()
    # a member of a merged instance line whose identity abstained (members disagree) or differs from the member's: its price would be carried onto a line that is not
    # that book (the merge copies the representative member's fields), so it is not priced; the instance line stays unpriced, as an unidentified line must
    abstained = {m["id"] for line in list(p.get("books") or []) + list(p.get("unconfirmed_books") or []) if line.get("merged_members")
                 for m in line["merged_members"] if line.get("status") != "identified" or str(line.get("title") or "").casefold() != str(m.get("title") or "").casefold()}
    for m in _members(p):
        row = new_rows.get(m["id"])
        if row is None:
            continue
        if m["id"] in abstained:
            out["skipped_instance_identity_abstained"] += 1
            continue
        res = apply_row(m, row, run_dir)
        out[res] += 1
        if res in ("priced", "null"):
            used_new.add(m["id"])
    rows = [new_rows[i] if i in used_new else r for i, r in old_rows.items()] + [new_rows[i] for i in used_new if i not in old_rows]
    p["pricing"] = {**{k: v for k, v in (p.get("pricing") or {}).items() if k != "prices"}, "prices": rows, "market": "US", "currency": "USD",
                    "price_basis": prices["data"].get("work_level_equivalent", {}).get("price_basis", "work_level_equivalent"),
                    "converted_fallback": {"enabled": bool((prices["data"].get("config") or {}).get("converted_fallback")), "policy": prices["data"].get("converted_policy"),
                                           "markets": (prices["data"].get("config") or {}).get("converted_markets")}}
    if instances_art:
        from pipeline import instances as IN
        IN.apply(p, instances_art, (p.get("corrections") or {}).get("ledger"))
        for line in p["books"] + list(p.get("unconfirmed_books") or []):     # as pipeline.identity_attach: the merge copies one member's identity; carry its catalogue fields
            if line.get("status") == "identified" and line.get("merged_members"):
                src = next((x for x in line["merged_members"] if x.get("status") == "identified" and x.get("identity_source_url") == line.get("identity_source_url")), None)
                if src:
                    for k in ("catalogue", "catalogue_title", "identity_work", "catalogue_dimensions"):
                        if k in src:
                            line[k] = src.get(k)
    _refresh(p)
    return dict(out)


def attach(run_dir: str | Path, prices_path: str | Path, *, render: bool = True) -> dict[str, Any]:
    from pipeline import contracts as C
    from pipeline import instances as IN
    from pipeline import packet as P
    run_dir, prices_path = Path(run_dir), Path(prices_path)
    pk = run_dir / C.ARTIFACTS["packet"]
    if not pk.is_file():
        return {"attached": False, "reason": "no claim_packet.json"}
    env = C.read_json(pk)
    if env["data"].get("fixture"):
        return {"attached": False, "reason": "fixture packet"}
    if run_dir.resolve() == prices_path.parent.resolve():
        return {"attached": False, "reason": "--prices must come from a scratch compute directory, not the run itself"}
    backup = run_dir / BACKUP
    if not backup.is_file():
        shutil.copy2(pk, backup)
    prices = C.read_json(prices_path)
    p = env["data"]
    t0 = dict(p["totals"])
    counts = apply(p, prices, prices_path.parent.resolve(), run_dir.resolve(), IN.load(run_dir))
    t = p["totals"]
    conv = (p.get("completeness") or {}).get("converted_lines") or {}
    p.setdefault("provenance", {})["price_attach"] = {"prices": str(prices_path), "groups": counts, "computed": prices.get("created_at") or prices.get("generated_at"),
                                                      "note": "asking prices of work-level equivalent editions (not the shelf copies); converted lines labelled; no human price check yet"}
    P.finalize_contract(p, run_dir)
    problems = P.verify_packet(env, C.read_json(run_dir / C.ARTIFACTS["frames"]), run_dir)
    if problems:
        return {"attached": False, "reason": "packet verification failed after the prices were attached", "problems": problems[:10]}
    C.write_json(pk, env)
    if render:
        from pipeline import report as R
        R.render(pk, run_dir)
        if (run_dir / "report.pdf").is_file():
            R.render_print_pdf(pk, run_dir)
    priced = [b for b in p["books"] if b.get("replacement_cost") or b.get("used_value")]
    return {"attached": True, "groups": counts, "book_lines": t.get("book_lines", t["book_count"]), "books_identified": t["books_identified"], "lines_priced": len(priced),
            "lines_priced_with_converted": sum(1 for b in priced if any((b.get(k) or {}).get("converted") is True for k in KINDS)),
            "books_replacement_cost": t["books_replacement_cost"], "books_used_value": t["books_used_value"], "converted_lines": {k: v for k, v in conv.items() if k != "note"},
            "before": {k: t0.get(k) for k in ("books_replacement_cost", "books_used_value", "excluded_from_totals")}, "excluded_from_totals": t["excluded_from_totals"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--run", required=True); c.add_argument("--readings", required=True); c.add_argument("--out", required=True)
    c.add_argument("--online", action="store_true"); c.add_argument("--cache-dir")
    c.add_argument("--no-reading-language", action="store_true"); c.add_argument("--no-converted", action="store_true"); c.add_argument("--markets", default="ES,GB")
    c.add_argument("--resolutions", help="resolutions.json whose accepted Google Books volumes (ISBNs, language) serve as edition records of gb: identities")
    c.add_argument("--only-google-books", action="store_true", help="price only the Google-Books-identified groups (an incremental pass)")
    c.add_argument("--throttle-retries", type=int, default=2); c.add_argument("--throttle-backoff-s", type=float, default=60.0); c.add_argument("--min-interval-s", type=float, default=2.0)
    a = sub.add_parser("attach")
    a.add_argument("--run", required=True); a.add_argument("--prices", required=True); a.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "compute":
        res = compute(args.run, args.readings, args.out, online=args.online, cache_dir=args.cache_dir, reading_language=not args.no_reading_language,
                      converted=not args.no_converted, markets=args.markets, throttle_retries=args.throttle_retries, throttle_backoff_s=args.throttle_backoff_s,
                      min_interval_s=args.min_interval_s, resolutions=args.resolutions, only_google_books=args.only_google_books)
    else:
        res = attach(args.run, args.prices, render=not args.no_render)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
