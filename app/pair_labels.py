"""Held-out pair adjudication imports and reporting, separate from production.

Model answers are model-read with provenance, never truth. Only explicit human
labels may support agreement/accuracy statements about the checked subset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from pipeline import pair_sheets as PS

RUNS = PS.RUNS
ANSWERS = {"same", "different", "cannot_tell"}
KINDS = {"model", "human"}
ANNOTATOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")

PairError = PS.PairError


def sheet_dir(run_id):
    PS.run_dir(run_id, RUNS)
    p = RUNS / "pair_sheets" / run_id
    if p.is_symlink() or not p.is_dir() or not p.resolve().is_relative_to(RUNS.resolve()):
        raise PairError("pair sheets have not been generated for this run")
    return p


def _manifest(run_id):
    return PS.read(PS.safe_file(sheet_dir(run_id), "pairs.json"))


def list_runs():
    out = []
    for p in sorted((RUNS / "pair_sheets").glob("*/pairs.json")):
        try:
            m = _manifest(p.parent.name)
            out.append({"run_id": p.parent.name, "pairs": len(m["pairs"])})
        except (PairError, OSError, ValueError):
            continue
    return sorted(out, key=lambda r: (r["run_id"] != PS.DEFAULT_RUN, r["run_id"]))


def blind_manifest(run_id):
    """No proposed groups, source kinds, controls, or model answers in this view."""
    m = _manifest(run_id)
    return {"run_id": run_id, "pairs": [{"pair_id": p["pair_id"],
             "sightings": [{k: s[k] for k in ("frame_id", "timestamp_s")} for s in p["sightings"]]} for p in m["pairs"]],
            "contact_pages": len(m["contact_sheets"])}


def image_file(run_id, *, pair_id=None, page=None):
    m = _manifest(run_id)
    if pair_id is not None:
        item = next((p for p in m["pairs"] if p["pair_id"] == pair_id), None)
        if item is None:
            raise PairError("unknown pair id")
        relative, digest = item["sheet_path"], item["sheet_sha256"]
    else:
        if not isinstance(page, int) or not 1 <= page <= len(m["contact_sheets"]):
            raise PairError("unknown contact page")
        item = m["contact_sheets"][page - 1]
        relative, digest = item["path"], item["sha256"]
    p = PS.safe_file(sheet_dir(run_id), relative)
    if PS.sha(p) != digest:
        raise PairError("sheet changed since generation")
    return p


def parse_jsonl(text):
    answers = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise PairError(f"invalid JSON on line {n}") from exc
        if not isinstance(entry, dict):
            raise PairError(f"line {n} must be an object")
        answers.append(entry)
    return answers


def _atomic_json(path, doc):
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def import_answers(run_id, answers, annotator, kind, *, source_name="API", source_sha256=None):
    if not isinstance(annotator, str) or not ANNOTATOR.fullmatch(annotator):
        raise PairError("annotator must be 1-80 safe filename characters (letters, digits, dot, dash, underscore)")
    if kind not in KINDS:
        raise PairError("kind must be model or human")
    if not isinstance(answers, list) or not answers:
        raise PairError("answers must be a nonempty list")
    root = sheet_dir(run_id)
    manifest = _manifest(run_id)
    pairs = {p["pair_id"]: p for p in manifest["pairs"]}
    clean = {}
    for a in answers:
        if not isinstance(a, dict) or not isinstance(a.get("pair_id"), str) or a["pair_id"] not in pairs:
            raise PairError("answer contains an unknown pair id")
        pid = a["pair_id"]
        if pid in clean:
            raise PairError(f"duplicate answer for {pid}")
        if not isinstance(a.get("answer"), str) or a["answer"] not in ANSWERS:
            raise PairError("answer must be same, different or cannot_tell")
        reason = a.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000 or "\n" in reason or "\r" in reason:
            raise PairError("reason must be one nonempty line of at most 1000 characters")
        # Bind to the displayed bytes without reparsing the whole manifest for
        # each answer in a large contact-sheet batch.
        sheet = PS.safe_file(root, pairs[pid]["sheet_path"])
        if PS.sha(sheet) != pairs[pid]["sheet_sha256"]:
            raise PairError("sheet changed since generation")
        clean[pid] = {"answer": a["answer"], "reason": reason.strip(), "sheet_sha256": pairs[pid]["sheet_sha256"]}
    directory = PS.run_dir(run_id, RUNS) / "pair_labels"
    if directory.is_symlink():
        raise PairError("unsafe label directory")
    directory.mkdir(exist_ok=True)
    manifest_sha, key_sha = PS.sha(root / "pairs.json"), PS.sha(PS.safe_file(root, "pairs_key.json"))
    lock = directory / ".import.lock"
    if lock.is_symlink():
        raise PairError("unsafe label lock")
    with lock.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        path = directory / f"{annotator}.json"
        if path.is_symlink():
            raise PairError("unsafe label file")
        timestamp = datetime.now(timezone.utc).isoformat()
        doc = PS.read(path) if path.is_file() else {
            "schema_version": 1, "purpose": "held_out_pair_evaluation", "run_id": run_id,
            "annotator": annotator, "kind": kind, "label_status": "model-read" if kind == "model" else "human-spot-check",
            "note": "Model labels are model-read, never truth; never consumed by the inventory, packet or association pipeline.",
            "manifest_sha256": manifest_sha, "key_sha256": key_sha, "created_at": timestamp, "labels": {}, "imports": []}
        if doc["kind"] != kind:
            raise PairError("annotator already has another kind; use a distinct annotator name")
        if doc["manifest_sha256"] != manifest_sha or doc["key_sha256"] != key_sha:
            raise PairError("pair evidence/key changed since the previous import")
        for pid, entry in clean.items():
            previous = doc["labels"].get(pid)
            if previous and any(previous[k] != entry[k] for k in entry):
                raise PairError(f"conflicting reimport for {pid}; use a new annotator/version to preserve provenance")
        new_ids = sorted(set(clean) - doc["labels"].keys())
        for pid in new_ids:
            doc["labels"][pid] = {**clean[pid], "labeled_at": timestamp, "human_confirmed": kind == "human"}
        if new_ids:
            payload_sha = source_sha256 or hashlib.sha256(json.dumps(answers, sort_keys=True).encode()).hexdigest()
            doc["imports"].append({"imported_at": timestamp, "source_name": Path(source_name).name,
                                   "source_sha256": payload_sha, "pair_ids": new_ids, "kind": kind, "annotator": annotator})
            _atomic_json(path, doc)
        result = summary(run_id)
        write_summary(run_id, result)
    return {"imported": len(new_ids), "total_for_annotator": len(doc["labels"]), "kind": kind,
            "saved_to": str(path), "summary": result}


def _agreement(rows):
    # Abstentions stay in the answered denominator; decisive rate is separate.
    n = len(rows)
    decided = [(a, b) for a, b in rows if a != "cannot_tell" and b != "cannot_tell"]
    agree = sum(a == b for a, b in decided)
    return {"answered": n, "decisive": len(decided), "agree": agree,
            "abstentions": n - len(decided), "agreement_of_answered": agree / n if n else None,
            "agreement_of_decisive": agree / len(decided) if decided else None}


def summary(run_id):
    root = sheet_dir(run_id)
    m, key = _manifest(run_id), PS.read(PS.safe_file(root, "pairs_key.json"))
    hashes = PS.sha(root / "pairs.json"), PS.sha(root / "pairs_key.json")
    pairs = {p["pair_id"]: p for p in m["pairs"]}
    ids = set(pairs)
    directory = PS.run_dir(run_id, RUNS) / "pair_labels"
    if directory.is_symlink():
        raise PairError("unsafe label directory")
    docs = []
    for path in sorted(directory.glob("*.json")):
        d = PS.read(PS.safe_file(directory, path.name))
        if (d.get("manifest_sha256"), d.get("key_sha256")) != hashes or d.get("run_id") != run_id:
            raise PairError(f"stale or mismatched evidence in {path.name}")
        if d.get("kind") not in KINDS or not set(d["labels"]).issubset(ids):
            raise PairError(f"invalid label provenance in {path.name}")
        expected_status = "human-spot-check" if d["kind"] == "human" else "model-read"
        if d.get("label_status") != expected_status:
            raise PairError(f"invalid label status in {path.name}")
        for pid, label in d["labels"].items():
            if (not isinstance(label.get("answer"), str) or label["answer"] not in ANSWERS
                    or label.get("human_confirmed") is not (d["kind"] == "human")
                    or label.get("sheet_sha256") != pairs[pid]["sheet_sha256"]):
                raise PairError(f"invalid label provenance in {path.name}: {pid}")
        docs.append(d)
    model = [d for d in docs if d["kind"] == "model"]
    human = [d for d in docs if d["kind"] == "human"]
    model_ids = set().union(*(set(d["labels"]) for d in model)) if model else set()
    human_ids = set().union(*(set(d["labels"]) for d in human)) if human else set()
    consensus, conflicts = {}, []
    for pid in sorted(human_ids):
        votes = {d["labels"][pid]["answer"] for d in human if pid in d["labels"] and d["labels"][pid]["answer"] != "cannot_tell"}
        if len(votes) == 1:
            consensus[pid] = votes.pop()
        elif len(votes) > 1:
            conflicts.append(pid)
    per_annotator = []
    for d in docs:
        labels = d["labels"]
        control_rows = [(l["answer"], key["pairs"][pid]["expected_answer"]) for pid, l in labels.items()
                        if key["pairs"][pid]["control_kind"]]
        labelled_sources = Counter(s for p in m["pairs"] if p["pair_id"] in labels for s in p["source_kinds"])
        entry = {"annotator": d["annotator"], "kind": d["kind"], "label_status": d["label_status"], "labeled": len(labels),
                 "answers": dict(Counter(l["answer"] for l in labels.values())), "counts_per_source": dict(labelled_sources),
                 "control_agreement": _agreement(control_rows)}
        if d["kind"] == "model":
            checked = set(labels) & human_ids
            entry["human_overlap"] = {"spot_checked": len(checked), "of_model_labeled": len(labels),
                "human_consensus_reference": _agreement([(l["answer"], consensus[pid]) for pid, l in labels.items() if pid in consensus]),
                "human_conflicts": len(set(labels) & set(conflicts)),
                "per_human": [{"annotator": h["annotator"], **_agreement([(labels[pid]["answer"], h["labels"][pid]["answer"])
                               for pid in sorted(set(labels) & set(h["labels"]))])} for h in human]}
        per_annotator.append(entry)
    overlap = model_ids & human_ids
    return {"run_id": run_id, "pairs": len(ids), "counts_per_source": m["counts_per_source"],
            "source_counts_overlap": True, "controls": dict(Counter(v["control_kind"] for v in key["pairs"].values() if v["control_kind"])),
            "annotators": per_annotator, "model_labeled_pairs": len(model_ids), "human_labeled_pairs": len(human_ids),
            "human_spot_checked_model_pairs": len(overlap), "human_consensus_pairs": len(consensus), "human_conflict_pairs": conflicts,
            "honesty": f"Model labels are model-read; human spot-check {len(overlap)} of {len(model_ids)} model-labeled pairs ({len(human_ids)} human-labeled pairs of {len(ids)} sheets).",
            "limits": "Control agreement is against automatic proxy expectations, not human truth. Human-reference results apply only to the checked subset; abstentions/conflicts cannot support an accuracy claim. No count correction or whole-run accuracy is proven."}


def write_summary(run_id, result=None):
    s = result or summary(run_id)
    lines = [f"# Pair labels: {run_id}", "", s["honesty"], "", f"Sheets: {s['pairs']}. Controls: {json.dumps(s['controls'], sort_keys=True)}.",
             f"Sources (overlap allowed): {json.dumps(s['counts_per_source'], sort_keys=True)}.", "",
             "| Annotator | Kind | Labels | Control agreement (agree / answered) | Human checked (of model labels) | Human-reference agreement (agree / answered) |",
             "|---|---|---:|---:|---:|---:|"]
    for a in s["annotators"]:
        c = a["control_agreement"]
        h = a.get("human_overlap")
        hc = h["human_consensus_reference"] if h else None
        lines.append(f"| {a['annotator']} | {a['label_status']} | {a['labeled']} | {c['agree']} / {c['answered']} | "
                     + (f"{h['spot_checked']} / {h['of_model_labeled']} | {hc['agree']} / {hc['answered']} |" if h else "— | — |"))
    if not s["annotators"]:
        lines.append("| No answers imported | — | 0 | — | — | — |")
    lines += ["", f"Human conflict pairs: {len(s['human_conflict_pairs'])}. Model abstentions stay in the answered denominator; decisive agreement is reported separately in the API.", "", s["limits"], ""]
    p = PS.run_dir(run_id, RUNS) / "pair_labels_summary.md"
    if p.is_symlink():
        raise PairError("unsafe summary file")
    # Atomic text output, so readers never see half a report.
    fd, temp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines))
    os.replace(temp, p)
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import")
    imp.add_argument("--run", required=True)
    imp.add_argument("--answers", type=Path, required=True)
    imp.add_argument("--annotator", required=True)
    imp.add_argument("--kind", choices=sorted(KINDS), required=True)
    summ = commands.add_parser("summary")
    summ.add_argument("--run", required=True)
    args = parser.parse_args()
    try:
        if args.command == "import":
            payload = args.answers.read_bytes()
            result = import_answers(args.run, parse_jsonl(payload.decode("utf-8")), args.annotator, args.kind,
                                    source_name=args.answers.name, source_sha256=hashlib.sha256(payload).hexdigest())
        else:
            result = summary(args.run)
            write_summary(args.run, result)
        print(json.dumps(result, indent=2))
    except (PairError, OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
