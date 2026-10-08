"""Re-resolve (and optionally re-price) the EXISTING readings of a finished run into a NEW directory: no detection, no reader, no Gemini call.

    python scripts/reresolve_run.py --source runs/final_yt2_gemini_v1 --out runs/<new dir> [--online] [--seed-cache DIR ...] [--price] [--price-budget-s 600]

Reads <source>/readings.json + inventory.json + manifest.json, never writes into <source> (``--out`` must not exist). Resolution is the production rule
(pipeline.resolve, complete-title path + fragment rule v6); identity per physical group is chosen exactly like pipeline.dedup (``_identity``), so the
"identified" count is the packet's definition. With ``--price`` every identified catalogue-work group is priced through the work-level equivalent policy
(pipeline.price) in a copy of the inventory. Writes <out>/identified.json (counts, per-group identity, the reasons readings were not accepted) and, with --price,
<out>/prices.json. A model-read reading is never truth: this reports how many groups the pipeline identified, not how many are right.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import contracts as C  # noqa: E402
from pipeline.dedup import _identity, _stage_records  # noqa: E402
from pipeline.resolve import resolve  # noqa: E402


def group_identities(inventory, readings, resolutions):
    """-> (books with identity/status rewritten like dedup.merge, per-group summary rows)."""
    rows = []
    for book in inventory["data"]["books"]:
        identity, issues = _identity([{"observation_id": o} for o in book["observation_ids"]], readings, resolutions)
        # keep the reading fields the packet needs; only the catalogue identity is replaced
        old = book.get("identity") or {}
        for key in ("visible_text", "visible_text_observation_id", "visible_text_reader", "reading_union_text", "readings_considered", "reading_status"):
            if key in old:
                identity[key] = old[key]
        book["identity"] = identity
        book["status"] = "identified" if identity["decision"] == "accepted" else "unidentified"
        rows.append({"instance_id": book["instance_id"], "decision": identity["decision"], "title": identity.get("title"), "author": identity.get("author"),
                     "work": identity.get("work"), "source_url": identity.get("source_url"), "visible_text": identity.get("visible_text")})
    return inventory["data"]["books"], rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="finished run directory (read only)")
    ap.add_argument("--out", required=True, help="NEW directory (refuses to reuse an existing one)")
    ap.add_argument("--online", action="store_true", help="catalogue (and, with --price, price source) HTTP; otherwise cache only")
    ap.add_argument("--seed-cache", action="append", default=[], help="existing Open Library cache directory to copy into the new run's cache (repeatable)")
    ap.add_argument("--price", action="store_true", help="work-level equivalent pricing of the identified groups")
    ap.add_argument("--price-budget-s", type=float, default=60.0)
    ap.add_argument("--price-cache-dir", help="price cache root (default: <out>/price_cache)")
    ap.add_argument("--readings", help="readings file name inside --source (default readings.json)", default="readings.json")
    args = ap.parse_args(argv)
    src, out = Path(args.source).resolve(), Path(args.out).resolve()
    if out.exists():
        ap.error(f"{out} exists; a new directory is required (existing runs are never overwritten)")
    (out / "cache").mkdir(parents=True)
    for seed in args.seed_cache:
        shutil.copytree(seed, out / "cache", dirs_exist_ok=True)
    readings_path = out / "readings.json"
    shutil.copyfile(src / args.readings, readings_path)
    started = time.perf_counter()
    res_path = resolve(readings_path, out, {"catalogue_online": args.online})
    resolve_s = time.perf_counter() - started
    res_art = C.read_json(res_path)
    inventory = C.read_json(src / "inventory.json")
    readings, resolutions = _stage_records(readings_path, "readings"), _stage_records(res_path, "resolutions")
    books, rows = group_identities(inventory, readings, resolutions)
    reasons = Counter()
    for r in resolutions.values():
        if r["decision"] != "accepted":
            reasons[(r["issues"][-1] if r.get("issues") else "no_issue")[:70]] += 1
    identified = [r for r in rows if r["decision"] == "accepted"]
    summary = {"source": str(src), "readings": len(readings), "groups": len(books), "groups_identified": len(identified),
               "distinct_works": len({r["work"] for r in identified}), "groups_ambiguous": sum(r["decision"] == "ambiguous" for r in rows),
               "resolutions_accepted": sum(r["decision"] == "accepted" for r in resolutions.values()), "resolve_wall_s": round(resolve_s, 1),
               "catalogue_requests": res_art["metrics"].get("requests"), "catalogue_cache_hits": sum(1 for m in res_art["metrics"].get("request_details", []) if m.get("cache_hit")),
               "not_accepted_by_last_issue": reasons.most_common(12), "online": args.online}
    C.write_json(out / "identified.json", {"summary": summary, "groups": rows})
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if args.price:
        from pipeline.packet import build_data
        from pipeline.price import price_books_with_work_level
        C.write_json(out / "inventory.json", inventory)
        manifest = C.read_json(src / "manifest.json")
        packet_books = build_data(inventory, manifest, {"run_id": out.name, "country": "US", "currency": "USD"})["books"]
        cfg = {"country": "US", "currency": "USD", "pricing_online": args.online, "price_budget_s": args.price_budget_s,
               "price_cache_dir": args.price_cache_dir or str(out / "price_cache")}
        t0 = time.perf_counter()
        prices = price_books_with_work_level(packet_books, out, cfg)
        C.write_json(out / "prices.json", prices)
        m = prices["metrics"]
        priced = sum(1 for r in prices["data"]["prices"] if r.get("replacement_cost") or r.get("used_value"))
        print(json.dumps({"price_wall_s": round(time.perf_counter() - t0, 1), "books_priced": priced, "replacement_lines": m.get("replacement_lines"),
                          "used_lines": m.get("used_lines"), "work_level_attempted": m.get("work_level_attempted"),
                          "work_level_budget_skipped": m.get("work_level_budget_skipped"), "budget_s": args.price_budget_s}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
