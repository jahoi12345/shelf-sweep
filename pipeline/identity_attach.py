"""Attach catalogue identities resolved AFTER the sweep (e.g. a hosted re-read + re-resolution in another run dir) to a finished run's claim_packet.json, in place.

    python -m pipeline.identity_attach --run runs/<run> --readings <readings.json> --resolutions <resolutions.json> [--no-render]

Same pattern as pipeline.auto_attach / pipeline.panorama_attach / pipeline.instances.attach: the original packet is kept once as ``claim_packet.pre_identities.json``, the
patched packet must pass pipeline.packet.verify_packet (nothing is written otherwise), the report is re-rendered, and attaching twice gives the same packet.

Per TRACK GROUP (a plain line, or each member of a merged instance line) the identity is chosen exactly like pipeline.dedup (``_identity``: an accepted resolution with
matching evidence and a source URL; conflicting works -> ambiguous, never a guess). A group that already had an accepted identity keeps it when the new resolutions have
none (zero loss). Lines with a user title correction or flagged for appraisal are left alone. When the run has an instance layer (instances.json), it is re-applied on top
(pipeline.instances.apply), which merges the members' identities into each merged line (one resolution; disagreeing works abstain) and recomputes the totals.
Identified != correct: the readings are model-read and no human title labels exist.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

BACKUP = "claim_packet.pre_identities.json"
IDENTITY_KEYS = ("title", "author", "edition", "isbn", "publisher", "id_confidence", "identity_source", "identity_source_url", "identity_matching_evidence",
                 "identity_decision", "catalogue", "catalogue_title", "identity_work", "catalogue_dimensions")


def _resolution_rows(path: str | Path, key: str) -> dict[str, dict[str, Any]]:
    from pipeline.dedup import _stage_records
    return _stage_records(str(path), key)


def identity_for(observation_ids: list[str], readings: dict[str, Any], resolutions: dict[str, Any]) -> dict[str, Any]:
    """dedup._identity over the observations, plus the catalogue that accepted it and the catalogue's own title."""
    from pipeline.dedup import _identity
    identity, _ = _identity([{"observation_id": o} for o in observation_ids], readings, resolutions)
    res = resolutions.get(identity.get("resolution_observation_id") or "") or {}
    identity["catalogue"] = res.get("catalogue") or ("open_library" if identity.get("decision") == "accepted" else None)
    identity["catalogue_title"] = res.get("catalogue_title")
    identity["catalogue_dimensions"] = (res.get("google_books") or {}).get("dimensions")
    return identity


def apply_identity(line: dict[str, Any], identity: dict[str, Any]) -> str:
    """Set one line's identity fields. Returns 'identified' | 'kept' | 'ambiguous' | 'skipped' | 'unchanged'."""
    if line.get("status") == "needs_appraisal" or any((c or {}).get("kind") == "set_title" for c in line.get("user_corrections") or []):
        return "skipped"
    if identity.get("decision") == "accepted" and identity.get("title") and identity.get("source_url"):
        line.update(status="identified", identity_source="catalogue_match", title=identity["title"], author=identity.get("author"), edition=identity.get("edition"),
                    isbn=identity.get("isbn"), publisher=identity.get("publisher"), id_confidence=identity.get("id_confidence"),
                    identity_source_url=identity["source_url"], identity_matching_evidence=identity.get("matching_evidence") or [], identity_decision="accepted",
                    catalogue=identity.get("catalogue"), catalogue_title=identity.get("catalogue_title"), identity_work=identity.get("work"),
                    catalogue_dimensions=identity.get("catalogue_dimensions"))
        line["issues"] = [i for i in line.get("issues") or [] if "Unreadable/unresolved spine" not in str(i)]
        return "identified"
    if line.get("status") == "identified":
        if not line.get("catalogue") and str(line.get("identity_source_url") or "").startswith("https://openlibrary.org/"):
            line["catalogue"] = "open_library"                            # identified before the catalogue field existed: the cited URL names it
        return "kept"                                                     # zero loss: an earlier accepted identity stays when the new resolutions have none
    if identity.get("decision") == "ambiguous":
        line["identity_decision"] = "ambiguous"
        return "ambiguous"
    return "unchanged"


def _refresh_totals(p: dict[str, Any]) -> None:
    from pipeline import packet as P
    t = p["totals"]
    fresh = P.compute_totals(p["books"], p.get("items") or [], {}, len(p.get("excluded_by_user") or []), p.get("pricing") is not None)
    for k in ("books_identified", "books_unidentified", "books_needs_appraisal", "books_identified_by_user_statement"):
        t[k] = fresh[k]
    p["language_breakdown"] = P.language_breakdown(p["books"])
    p["completeness"] = P.completeness(p["books"], p.get("items") or [], p.get("room") or {}, t)


def apply(p: dict[str, Any], readings: dict[str, Any], resolutions: dict[str, Any], instances_art: dict[str, Any] | None = None) -> dict[str, Any]:
    """Patch packet ``data`` in place. Returns counts per outcome (over track groups)."""
    from collections import Counter
    out = Counter()
    for lst in ("books", "unconfirmed_books"):
        for line in p.get(lst) or []:
            targets = line.get("merged_members") or [line]
            for t in targets:
                obs = (t.get("evidence") or {}).get("observation_ids") or []
                out[apply_identity(t, identity_for(obs, readings, resolutions))] += 1
            if line.get("merged_members"):                               # a merged line without the instance layer: identified only when its members agree on one work
                ids = {(m.get("identity_work"), m.get("title")) for m in line["merged_members"] if m.get("status") == "identified"}
                if len(ids) == 1:
                    src = next(m for m in line["merged_members"] if m.get("status") == "identified")
                    for k in IDENTITY_KEYS + ("status",):
                        line[k] = copy.deepcopy(src.get(k))
    from pipeline import packet as P
    identified = {b["id"] for b in p.get("books") or [] if b.get("status") == "identified"}
    p["review_queue"] = P.consolidate_review([r for r in P.flat_review(p.get("review_queue")) if not (r.get("code") == "unidentified_book" and r.get("ref_id") in identified)])
    if instances_art:
        from pipeline import instances as IN
        IN.apply(p, instances_art, (p.get("corrections") or {}).get("ledger"))
        identified = {b["id"] for b in p["books"] if b.get("status") == "identified"}
        p["review_queue"] = P.consolidate_review([r for r in P.flat_review(p["review_queue"]) if not (r.get("code") == "unidentified_book" and r.get("ref_id") in identified)])
        for line in p["books"] + list(p.get("unconfirmed_books") or []):     # the merge copies the identity fields of one member: carry its catalogue and work too
            if line.get("status") == "identified" and line.get("merged_members"):
                src = next((m for m in line["merged_members"] if m.get("status") == "identified" and m.get("identity_source_url") == line.get("identity_source_url")), None)
                if src:
                    for k in ("catalogue", "catalogue_title", "identity_work", "catalogue_dimensions"):
                        line[k] = src.get(k)
    _refresh_totals(p)
    return dict(out)


def attach(run_dir: str | Path, readings_path: str | Path, resolutions_path: str | Path, *, render: bool = True) -> dict[str, Any]:
    from pipeline import contracts as C
    from pipeline import instances as IN
    from pipeline import packet as P
    run_dir = Path(run_dir)
    pk = run_dir / C.ARTIFACTS["packet"]
    if not pk.is_file():
        return {"attached": False, "reason": "no claim_packet.json"}
    env = C.read_json(pk)
    if env["data"].get("fixture"):
        return {"attached": False, "reason": "fixture packet"}
    backup = run_dir / BACKUP
    if not backup.is_file():
        shutil.copy2(pk, backup)
    readings, resolutions = _resolution_rows(readings_path, "readings"), _resolution_rows(resolutions_path, "resolutions")
    before = env["data"]["totals"].get("books_identified")
    counts = apply(env["data"], readings, resolutions, IN.load(run_dir))
    p = env["data"]
    p.setdefault("provenance", {})["identity_attach"] = {"readings": str(readings_path), "resolutions": str(resolutions_path), "groups": counts,
                                                         "note": "identities resolved after the sweep from these files; identified != correct (model-read readings, no human labels)"}
    P.finalize_contract(p, run_dir)
    problems = P.verify_packet(env, C.read_json(run_dir / C.ARTIFACTS["frames"]), run_dir)
    if problems:
        return {"attached": False, "reason": "packet verification failed after the identities were attached", "problems": problems[:10]}
    C.write_json(pk, env)
    if render:
        from pipeline import report as R
        R.render(pk, run_dir)
        if (run_dir / "report.pdf").is_file():
            R.render_print_pdf(pk, run_dir)
    t = p["totals"]
    return {"attached": True, "books_identified_before": before, "books_identified": t["books_identified"], "book_lines": t.get("book_lines", t["book_count"]),
            "groups": counts, "catalogues": dict(__import__("collections").Counter(b.get("catalogue") for b in p["books"] if b.get("status") == "identified"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--readings", required=True)
    ap.add_argument("--resolutions", required=True)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(attach(a.run, a.readings, a.resolutions, render=not a.no_render), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
