"""Score baseline/correction or reader comparison using one immutable human label file.

Entries describe actual artifacts and executed/blocked/not_runnable states. A blocked
reader has no inferred output and no fabricated accuracy. No agreement-as-truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from eval.score import score, unavailable_languages, metric
from pipeline.contracts import write_json
from pipeline.read_spines import load


def compare(labels_path, entries_path, output_dir):
    root = Path(output_dir)
    if root.exists():
        raise ValueError("Comparison output exists; choose a new version")
    labels = load(labels_path)
    label_bytes_hash = hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()
    entries = load(entries_path)
    if isinstance(entries, dict):
        entries = entries["runs"]
    results = []
    for entry in entries:
        name = entry["name"]
        if not name or Path(name).name != name or name in {r["name"] for r in results}:
            raise ValueError("Comparison entry names must be unique simple names")
        status = entry["execution_status"]
        if status not in {"executed", "blocked", "not_runnable"}:
            raise ValueError("Execution status must be executed, blocked or not_runnable")
        run_dir = Path(entry["run_dir"])
        def artifact(key):
            filename = entry.get(key)
            return load(run_dir/filename) if filename else None
        if status != "executed":
            reason = entry.get("reason", "reader did not execute")
            bars = {"book_count_error": .05, "title_correctness": .70, "confident_wrong": .03, "non_book_correct_category_coverage": .80}
            result = {"status": "unavailable", "issues": [reason],
                      "metrics": {k: metric(reason=reason, eligible=False, threshold=bars.get(k)) for k in
                                  ("book_count_error", "book_detection_precision", "book_detection_recall", "title_correctness", "confident_wrong", "non_book_correct_category_coverage", "duplicate_instances", "split_instances", "merged_instances", "clutter_false_positive_rate", "duplicate_predictions")},
                      "clutter_false_positives": {"available": False, "counts_by_category": None,
                                                 "denominator_all_book_predictions": None,
                                                 "total_false_book_predictions": None, "unavailable_reason": reason},
                      "language_scores": {k: unavailable_languages(reason) for k in
                                           ("literal_observations", "catalogue_observations", "inventory_instances")}}
        else:
            candidates = artifact("candidates")
            if candidates is None:
                raise ValueError("Executed comparison requires a real candidate artifact")
            result = score(labels, candidates, readings=artifact("readings"), resolutions=artifact("resolutions"),
                           inventory=artifact("inventory"), manifest=artifact("manifest"), run_dir=run_dir)
        execution_report = artifact("execution_report")
        record = {"name": name, "execution_status": status,
                  "component_execution": execution_report,
                  "input_provenance": entry.get("input_provenance", "unknown"),
                  "run_dir": str(run_dir), "labels_file_sha256": label_bytes_hash,
                  "evaluation": result}
        results.append(record)
    # Prevent unnoticed label edits while a potentially slow comparison was running.
    if hashlib.sha256(Path(labels_path).read_bytes()).hexdigest() != label_bytes_hash:
        raise ValueError("Human labels changed during comparison; freeze before scoring")
    root.mkdir(parents=True)
    for record in results:
        write_json(root/(record["name"]+".json"), record)
    summary = {"comparison_version": 1, "labels_path": str(labels_path), "labels_file_sha256": label_bytes_hash,
               "runs": results, "accuracy_claim": "Only frozen human labels supply truth; model agreement is not ground truth"}
    write_json(root/"comparison.json", summary)
    return root/"comparison.json"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations",required=True);p.add_argument("--runs",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();print(compare(a.annotations,a.runs,a.out))

if __name__=="__main__":main()
