"""Independent audit of a finished run directory (read-only; never writes into runs/). Not collected by pytest (no test_ prefix).

    .venv/bin/python -m tests.packet_audit runs/final_yt1_v1 [more runs] [--json out.json]
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

HAND_COUNT = 458
BOOK_NULLABLE = ("title", "author", "edition", "isbn", "publisher", "language", "replacement_cost", "used_value", "spine_height_cm", "spine_thickness_cm")


def _fold(t):
    t = unicodedata.normalize("NFKD", (t or "").casefold())
    return " ".join(re.findall(r"[^\W_]+", "".join(c for c in t if not unicodedata.combining(c))))


def audit(run_dir):
    run = Path(run_dir)
    pk = json.loads((run / "claim_packet.json").read_text())
    inv = json.loads((run / "inventory.json").read_text())
    d = pk["data"]
    books, items = d["books"], d["items"]
    res = {"run": run.name, "violations": collections.defaultdict(list), "facts": {}}
    V = res["violations"]; F = res["facts"]

    # --- count consistency (one number everywhere) -------------------------------------------------------------------------------------
    c, t, ic = d["count"], d["totals"], inv["data"]["counts"]
    counts = {"len(books)": len(books), "count.book_groups": c["book_groups"], "totals.book_count": t["book_count"], "inventory.counts.book_count": ic["book_count"],
              "inventory.books": len(inv["data"].get("books", books))}
    F["count_sources"] = counts
    if len(set(counts.values())) != 1:
        V["count_disagreement"].append(counts)
    ids = [b["id"] for b in books]
    if len(set(ids)) != len(ids):
        V["duplicate_book_ids"].append(len(ids) - len(set(ids)))
    obs = collections.Counter(o for b in books for o in (b["evidence"].get("observation_ids") or []))
    dup = [o for o, n in obs.items() if n > 1]
    if dup:
        V["observation_in_two_book_groups"].append(dup[:5])
    lo, hi, prop = c["book_count_lower"], c["book_count_upper"], c["book_groups"]
    F["band"] = [lo, prop, hi, c.get("count_if_near_ties_merged")]
    if not (lo <= prop <= hi):
        V["band_does_not_contain_proposed"].append([lo, prop, hi])
    alt = c.get("alternative_method") or {}
    F["alternative_method"] = {k: alt.get(k) for k in ("method", "status", "book_groups")}
    err = (prop - HAND_COUNT) / HAND_COUNT
    F["count_vs_hand_458"] = {"proposed": prop, "rel_err": round(err, 4), "pass_pm5": abs(err) <= 0.05}
    if alt.get("book_groups") is not None:
        ae = (alt["book_groups"] - HAND_COUNT) / HAND_COUNT
        F["alt_vs_hand_458"] = {"alt": alt["book_groups"], "rel_err": round(ae, 4), "pass_pm5": abs(ae) <= 0.05}
    if "hand_count" in json.dumps(d.get("count", {})) and "NOT computed" not in json.dumps(d["count"].get("hand_count_comparison", "")):
        V["hand_count_leaked_into_packet"].append(True)

    # --- every line cites evidence that exists -------------------------------------------------------------------------------------------
    missing = collections.Counter()
    for kind, rows in (("book", books), ("item", items)):
        for r in rows:
            ev = r.get("evidence") or {}
            if not ev.get("observation_ids"):
                V["line_without_observation"].append(r["id"])
            fr = ev.get("frame_ref")
            if not fr or not (run / fr).is_file():
                missing[kind + "_frame"] += 1
            bb = ev.get("bbox_xyxy")
            if not bb or not (0 <= bb[0] < bb[2] <= ev["frame_width_px"] + 1 and 0 <= bb[1] < bb[3] <= ev["frame_height_px"] + 1):
                V["bbox_outside_frame"].append(r["id"])
    for b in books:
        for re_ in b.get("reading_evidence") or []:
            cp = re_.get("crop_path")
            if cp and not (run / cp).is_file():
                missing["crop"] += 1
    if missing:
        V["missing_evidence_files"].append(dict(missing))
    F["books_with_reading_crop"] = sum(1 for b in books if b.get("reading_evidence"))
    # evidence index hashes (bytes re-hashed)
    idx = json.loads((run / "evidence_index.json").read_text())
    bad_hash, checked = [], 0
    for f in idx["files"]:
        p = run / f["path"]
        if not f.get("exists"):
            V["evidence_index_marks_missing"].append(f["path"]); continue
        if not p.is_file():
            bad_hash.append(f["path"] + " (absent)"); continue
        checked += 1
        if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
            bad_hash.append(f["path"])
    F["evidence_index_files_rehashed"] = checked
    if bad_hash:
        V["evidence_hash_mismatch"].append(bad_hash[:5])
    if idx.get("missing"):
        V["evidence_index_missing"].append(idx["missing"][:5])

    # --- unknowns explicit: every null has a stated reason ------------------------------------------------------------------------------
    nulls = collections.Counter(); unexplained = collections.Counter()
    comp = d["completeness"]["books"]
    for b in books:
        for k in BOOK_NULLABLE:
            if b.get(k) is None:
                nulls[k] += 1
                if k in ("replacement_cost", "used_value"):
                    if not (b.get("pricing_null_reasons") or {}).get(k):
                        unexplained[k] += 1
                elif k in ("title", "author", "edition", "isbn", "publisher", "language"):
                    if not (b.get("status") == "unidentified" and (b.get("identity_decision") or b.get("issues"))):
                        unexplained[k] += 1
                else:  # spine dimensions: abstention must be stated somewhere reachable from the line or the packet
                    pass
    F["book_nulls"] = dict(nulls)
    if unexplained:
        V["null_without_reason"].append(dict(unexplained))
    if comp["measurement_abstained"] != len(books) and any(b.get("spine_height_cm") is None for b in books):
        n_null = sum(1 for b in books if b.get("spine_height_cm") is None)
        if comp["measurement_abstained"] != n_null:
            V["measurement_abstention_count_mismatch"].append([comp["measurement_abstained"], n_null])
    # packet completeness block agrees with recount
    for k in ("title", "author", "edition", "isbn", "visible_text", "replacement_cost", "used_value"):
        known = sum(1 for b in books if b.get(k) not in (None, ""))
        if comp[k]["known"] != known:
            V["completeness_block_disagrees"].append([k, comp[k]["known"], known])
    itm = d["completeness"]["items"]
    if itm["total"] != len(items) or itm["category"]["known"] != sum(1 for i in items if i.get("category")):
        V["item_completeness_disagrees"].append(itm["total"])
    # room nulls carry reasons
    room = d["completeness"]["room"]
    for k, v in room.items():
        if isinstance(v, dict) and v.get("value") is None and not v.get("why"):
            V["room_null_without_reason"].append(k)

    # --- titles: nothing invented; every identified line is literal-evidence backed ----------------------------------------------------
    identified = [b for b in books if b.get("status") == "identified" or b.get("title")]
    F["identified_books"] = len(identified)
    for b in identified:
        vis = set(_fold(" ".join(r.get("visible_text") or "" for r in (b.get("reading_evidence") or []))).split())
        title_toks = [x for x in _fold(b.get("title")).split() if len(x) >= 3]
        unseen = [x for x in title_toks if x not in vis]
        if unseen and b.get("identity_source") != "user_statement":
            V["identified_title_with_unseen_words"].append([b["id"], unseen])
        if not b.get("identity_matching_evidence") or not b.get("identity_source_url"):
            V["identified_without_matching_evidence"].append(b["id"])
    # resolution honesty: offline catalogue misses must not be hidden behind "unreadable"
    resf = run / "resolutions.json"
    if resf.is_file():
        rr = json.loads(resf.read_text())["data"]["resolutions"]
        why = collections.Counter()
        for r in rr:
            for i in r.get("issues") or []:
                why[re.sub(r"[:\d].*", "", i)[:60] if not i.startswith("catalogue_unavailable") else "catalogue_unavailable_offline_cache_miss"] += 1
        F["resolution_decisions"] = dict(collections.Counter(r.get("decision") for r in rr))
        off = sum(1 for r in rr if "catalogue_offline_cache_miss" in json.dumps(r.get("issues")))
        F["resolutions_blocked_by_offline_catalogue"] = off
        F["resolutions_with_title_text"] = sum(1 for r in rr if r.get("title_text"))
        if off:
            st = d["stage_status"]["resolve"]["status"]
            hidden = sum(1 for b in books if "catalogue" not in json.dumps(b.get("issues") or []) + json.dumps(b.get("pricing_issues") or []))
            V["offline_catalogue_not_disclosed_per_line"].append({"resolutions_offline": off, "resolve_stage_status": st, "book_lines_not_mentioning_catalogue": hidden})
    # --- prices -------------------------------------------------------------------------------------------------------------------------
    priced = [(b, k) for b in books for k in ("replacement_cost", "used_value") if b.get(k)]
    F["priced_book_lines"] = len(priced)
    for b, k in priced:
        o = b[k]
        for need in ("amount", "currency", "market", "url", "retrieved_at", "evidence", "price_basis"):
            if not o.get(need):
                V["priced_line_missing_" + need].append(b["id"])
        if o.get("price_basis") not in ("exact_edition", "work_level_equivalent"):
            V["price_basis_unlabelled"].append(b["id"])
        if o.get("price_basis") == "work_level_equivalent" and not o.get("equivalent_isbn"):
            V["work_level_without_equivalent_isbn"].append(b["id"])
        if o.get("currency") != "USD" or o.get("market") != "US" or o.get("converted"):
            V["not_us_usd_unconverted"].append(b["id"])
        ev = o.get("evidence") or {}
        p = run / ev.get("raw_path", "")
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != ev.get("sha256"):
            V["price_evidence_hash"].append(b["id"])
    for k, tk in (("replacement_cost", "books_replacement_cost"), ("used_value", "books_used_value")):
        s = round(sum(b[k]["amount"] for b in books if b.get(k)), 2)
        tv = t.get(tk)
        if (tv or 0) and abs((tv if not isinstance(tv, dict) else tv.get("amount", 0)) - s) > 0.005 or (tv is None and s):
            V["price_total_mismatch"].append([tk, tv, s])
    # --- timings ------------------------------------------------------------------------------------------------------------------------
    F["finalization_s"] = d["sweep"].get("finalization_s")
    F["packet_under_300s"] = (F["finalization_s"] or 1e9) < 300
    F["reading"] = {k: d["provenance"]["read_policy"].get(k) for k in ("groups_total", "groups_attempted", "groups_with_text_ge8", "groups_unattempted")}
    F["hosted"] = d["provenance"]["stages"]["read"] if "stages" in d["provenance"] else None
    F["room_status"] = (d["room"] or {}).get("status")
    F["room_auto"] = d["measurement_evidence"]["room_auto"]
    # --- report numbers equal packet numbers -------------------------------------------------------------------------------------------
    rep = run / "report.html"
    if rep.is_file():
        html = rep.read_text(errors="replace")
        text = re.sub(r"<[^>]+>", " ", html)
        for label, v in (("book_groups", prop), ("band_low", lo), ("band_high", hi)):
            if not re.search(r"(?<!\d)" + str(v) + r"(?!\d)", text):
                V["report_html_missing_number"].append([label, v])
        F["report_pdf_bytes"] = (run / "report.pdf").stat().st_size if (run / "report.pdf").is_file() else None
        if F["report_pdf_bytes"] is None:
            V["report_pdf_missing"].append(True)
    else:
        V["report_html_missing"].append(True)
    res["violations"] = {k: v for k, v in V.items()}
    return res


if __name__ == "__main__":
    out, args = None, sys.argv[1:]
    if "--json" in args:
        i = args.index("--json"); out = args[i + 1]; del args[i:i + 2]
    allr = [audit(a) for a in args]
    for r in allr:
        print("==", r["run"]); print(" facts:", json.dumps(r["facts"], default=str)[:1800]); print(" violations:", json.dumps({k: (v[:3] if isinstance(v, list) else v) for k, v in r["violations"].items()}, default=str)[:2500] or "none")
    if out:
        Path(out).write_text(json.dumps(allr, indent=1, default=str))
