"""Compare a run's physical-book count with the user's hand count (eval/shelf_facts.json).

    python -m eval.count_vs_truth runs/<run>/inventory.json [more inventories] [--truth 458] [--tol 0.05]

The hand count is a total, not per-book labels: a matching total can still hide offsetting errors
(missed books vs duplicate/false-positive groups). The gate is |count - truth| / truth <= tol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_truth(path: str = "eval/shelf_facts.json") -> int:
    return int(json.loads(Path(path).read_text())["claims"]["physical_book_count"]["value"])


def compare(inventory_path: str, truth: int, tol: float = 0.05) -> dict:
    inv = json.loads(Path(inventory_path).read_text())
    counts = inv["data"]["counts"]
    books = int(counts["book_count"])
    if "books" in inv["data"] and len(inv["data"]["books"]) != books:
        raise ValueError(f"inventory disagreement: counts.book_count={books} but {len(inv['data']['books'])} book instances")
    err = (books - truth) / truth
    return {
        "inventory": inventory_path,
        "run_id": inv.get("run_id"),
        "association": (inv.get("metrics") or {}).get("association", "feature_registration"),
        "book_groups": books,
        "hand_count": truth,
        "relative_error": round(err, 4),
        "gate_within": tol,
        "gate_pass": abs(err) <= tol,
        "count_bounds": [counts.get("count_lower"), counts.get("count_upper")],
        "best_guess": counts.get("count_best"),
        "best_guess_relative_error": None if counts.get("count_best") is None else round((counts["count_best"] - truth) / truth, 4),
        "best_guess_gate_pass": None if counts.get("count_best") is None else abs(counts["count_best"] - truth) / truth <= tol,
        "count_range": counts.get("count_range"),
        "truth_in_range": None if not counts.get("count_range") else counts["count_range"][0] <= truth <= counts["count_range"][1],
        "count_reliability": counts.get("count_reliability"),
        "uncertain_observations": counts.get("uncertain_observations"),
        "caveat": "Hand count is a total, not per-book truth; offsetting errors (misses vs duplicates/false positives) are not separable.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventories", nargs="+")
    p.add_argument("--truth", type=int)
    p.add_argument("--tol", type=float, default=0.05)
    a = p.parse_args()
    truth = a.truth if a.truth is not None else load_truth()
    for path in a.inventories:
        r = compare(path, truth, a.tol)
        print(f"{r['run_id']}: {r['book_groups']} books vs hand count {r['hand_count']} -> {r['relative_error']:+.1%} "
              f"({'PASS' if r['gate_pass'] else 'FAIL'} at ±{a.tol:.0%}; assoc={r['association']})")
        if r["best_guess"] is not None:
            print(f"  best guess {r['best_guess']} ({r['best_guess_relative_error']:+.1%}, {'PASS' if r['best_guess_gate_pass'] else 'FAIL'}), "
                  f"range {r['count_range'][0]}-{r['count_range'][1]} (truth {'inside' if r['truth_in_range'] else 'OUTSIDE'}), reliability {r['count_reliability']}")
        print(json.dumps(r))


if __name__ == "__main__":
    main()
