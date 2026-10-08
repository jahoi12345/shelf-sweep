"""Score a run's readings/resolutions against human title truth.

    python -m app.score_titles --truth eval/truth_capture/titles_<...>.json [--truth ...] --run runs/<run_id> [--out runs/titles_score_v1]

Truth = ``purpose: evaluation_truth`` files written by the Label-titles view. Only ``labeled`` rows are scored; unreadable / not_a_book /
duplicate_of_previous are excluded from denominators and reported. No labels => every metric is unavailable (never zero). Rules reuse
``eval.score.metric`` (denominator/availability semantics) and ``pipeline.resolve.normalize_title``; neither is modified.
A prediction is matched to a truth crop by (label artifact, observation id) when the run is the labelled run (same candidates hash), else by same scan sha,
frame timestamp within 0.06 s and bbox IoU >= 0.5 (one-to-one, best IoU first).

Metrics per language (es/en/other) and overall, at two layers:
  resolution layer: accepted catalogue title if the resolver accepted one, else abstained
  reading layer:    ASSERTED title_text only (an OCR block without title_text counts as abstained: it asserts no title)
  literal layer:    visible_text as the reader produced it (informational: how much of the title the literal OCR recovered; never "confident")
    exact_title      normalized prediction == normalized truth
    title_contained  one normalized title contains the other (prediction >= 4 chars)
    recall_ge_0_5    >= half of the truth's tokens appear in the prediction
    abstained        no prediction text
    confidently_wrong  an ASSERTED full title (accepted catalogue identity, or title_complete with read_confidence >= 0.85 = eval/score.py cutoff)
                       whose normalized text differs from the truth; containment or partial overlap does not rescue it
    unmatched        no run observation overlaps the labelled crop (reported, counted in the denominator as a miss, never as abstained)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.score import CONFIDENT_THRESHOLD, metric
from pipeline.resolve import normalize_title

LANGS = ("es", "en", "other")
IOU_MIN, TS_TOL, CONF_MIN = 0.5, 0.06, CONFIDENT_THRESHOLD      # confidence cutoff aligned with eval/score.py (0.85)
ROOT = Path(__file__).resolve().parents[1]


def _read(p: Path) -> Any:
    return json.loads(p.read_text())


def _iou(a: list[float], b: list[float]) -> float:
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = x * y
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _tokens(s: str) -> list[str]:
    return normalize_title(s).split()


def _contiguous(needle: list[str], hay: list[str]) -> bool:
    n = len(needle)
    return bool(needle) and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def compare(truth: str, pred: str) -> dict[str, bool]:
    """Containment is on TOKEN boundaries (the full truth title as a contiguous token run inside the prediction, or a >= 4-char prediction
    inside the truth); substrings like 'Dune' in 'Sandune' never count."""
    t, p = normalize_title(truth), normalize_title(pred)
    tl, pl = t.split(), p.split()
    tt, pt = set(tl), set(pl)
    return {"exact": bool(p) and p == t,
            "contained": bool(p) and (_contiguous(tl, pl) or (len(p) >= 4 and _contiguous(pl, tl))),
            "recall": bool(tt) and len(tt & pt) / len(tt) >= 0.5}


def _sha(path: Path) -> str:
    from hashlib import sha256
    return sha256(path.read_bytes()).hexdigest()


def truth_key(r: dict[str, Any]) -> str:
    """A label is identified by its ARTIFACT (labelled run + candidates.json hash) AND observation id: bare observation ids repeat across runs."""
    return r.get("truth_key") or f"{r.get('label_run')}|{r.get('candidates_sha256')}|{r['observation_id']}"


def load_truth(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, info = [], {"files": [], "excluded": {"unreadable": 0, "not_a_book": 0, "duplicate_of_previous": 0, "partial": 0}, "unlabeled": 0, "annotators": set(),
                      "sha_verified": 0, "sha_mismatch": 0, "crop_missing": 0, "crop_unverifiable": 0, "duplicate_labels": 0, "integrity_failures": []}
    seen: dict[str, str] = {}
    for p in paths:
        d = _read(p)
        if d.get("purpose") != "evaluation_truth" or d.get("kind") != "title_labels":
            raise ValueError(f"{p}: not a title_labels evaluation_truth file")
        prov = d.get("provenance") or {}
        if d.get("fixture") or prov.get("source_type") not in (None, "human") or prov.get("review_status") == "pending" or not d.get("annotator"):
            raise ValueError(f"{p}: labels are not reviewed human truth (fixture / model-read / pending / no annotator); use --dev-smoke for model-read labels")
        if not (d.get("source_hashes") or {}).get("candidates_sha256"):
            raise ValueError(f"{p}: label file has no candidates_sha256; labels cannot be joined to a run without the source hash")
        info["files"].append(str(p)); info["annotators"].add(d.get("annotator"))
        lrun = ROOT / "runs" / str(d.get("run_id", ""))
        cand_sha = (d.get("source_hashes") or {}).get("candidates_sha256")
        for s in d["sample"]:
            lab = d["labels"].get(s["observation_id"])
            if not lab:
                info["unlabeled"] += 1
            elif lab["status"] == "labeled":
                # The labelled pixels must still be the labelled pixels. A missing / changed / undeclared source crop makes truth integrity
                # UNAVAILABLE for the whole file set (reported, never silently dropped): the row can neither be trusted nor ignored.
                oid, crop_rel, want = s["observation_id"], s.get("crop_path"), s.get("crop_sha256")
                crop = lrun / str(crop_rel or "")
                if not crop_rel or not want:
                    info["crop_unverifiable"] += 1; info["integrity_failures"].append({"observation_id": oid, "label_run": d.get("run_id"), "problem": "no crop_path/crop_sha256 declared"}); continue
                if not crop.is_file() or crop.is_symlink() or not crop.resolve().is_relative_to(lrun.resolve()):
                    info["crop_missing"] += 1; info["integrity_failures"].append({"observation_id": oid, "label_run": d.get("run_id"), "problem": "source crop missing"}); continue
                if _sha(crop) != want:
                    info["sha_mismatch"] += 1; info["integrity_failures"].append({"observation_id": oid, "label_run": d.get("run_id"), "problem": "source crop changed (sha256 differs)"}); continue
                info["sha_verified"] += 1
                row = {**s, "truth_title": lab["title"], "language": lab["language"], "label_run": d["run_id"], "label_scan_sha": (d.get("scan") or {}).get("sha256"),
                       "candidates_sha256": cand_sha}
                k = truth_key(row); row["truth_key"] = k
                if k in seen:
                    if seen[k] != lab["title"]:
                        raise ValueError(f"{p}: observation {oid} of artifact {d['run_id']}/{cand_sha[:12]} is labelled twice with different titles")
                    info["duplicate_labels"] += 1; continue                 # same label repeated across files: one physical crop, counted once
                seen[k] = lab["title"]; rows.append(row)
            else:
                info["excluded"][lab["status"]] = info["excluded"].get(lab["status"], 0) + 1     # partial/unreadable/not_a_book/duplicate: not full-title truth
    info["annotators"] = sorted(a for a in info["annotators"] if a)
    bad = info["sha_mismatch"] + info["crop_missing"] + info["crop_unverifiable"]
    info["integrity"] = {"status": "unavailable" if bad else "verified", "verified": info["sha_verified"], "failed": bad}
    return rows, info


def load_run(run: Path) -> dict[str, Any]:
    cand = _read(run / "candidates.json")["data"]["observations"]
    man = _read(run / "manifest.json")["data"]
    readings = {r["observation_id"]: r for r in _read(run / "readings.json")["data"]["readings"]} if (run / "readings.json").is_file() else {}
    res = {r["observation_id"]: r for r in _read(run / "resolutions.json")["data"]["resolutions"]} if (run / "resolutions.json").is_file() else {}
    from hashlib import sha256
    return {"cands": [o for o in cand if o.get("kind") == "book"], "frames": {f["frame_id"]: f for f in man["frames"]}, "readings": readings, "res": res,
            "candidates_sha256": sha256((run / "candidates.json").read_bytes()).hexdigest(), "scan_sha": (man.get("input") or {}).get("sha256"),
            "run_id": run.name}


def match(rows: list[dict[str, Any]], R: dict[str, Any]) -> dict[str, str | None]:
    """truth key (label artifact + observation id, see truth_key) -> run observation_id (or None). Never keyed by a bare observation id."""
    out: dict[str, str | None] = {}
    ids = {o["observation_id"] for o in R["cands"]}
    # observation ids are trusted only when the label file's candidates.json hash equals this run's (same artifact); never by bare id
    identical = [r for r in rows if r.get("candidates_sha256") and r["candidates_sha256"] == R["candidates_sha256"] and r["observation_id"] in ids]
    for r in identical:
        out[truth_key(r)] = r["observation_id"]
    # cross-run geometric matching: only when BOTH files carry the same scan hash (an unknown hash can never establish identity);
    # per frame-time group, maximum-cardinality assignment (eval.score.match_boxes, Hungarian), never greedy best-IoU
    from eval.score import match_boxes
    rest = [r for r in rows if truth_key(r) not in out and r.get("label_scan_sha") and R["scan_sha"] and r["label_scan_sha"] == R["scan_sha"]]
    used_o = set(out.values())
    by_t: dict[float, list[dict[str, Any]]] = {}
    for r in rest:
        by_t.setdefault(round(r["timestamp_s"] / TS_TOL) * TS_TOL, []).append(r)
    for t, group in by_t.items():
        preds = [o for o in R["cands"] if o["observation_id"] not in used_o and R["frames"].get(o["frame_id"]) and abs(R["frames"][o["frame_id"]]["timestamp_s"] - t) <= TS_TOL]
        for ti, pi, v in match_boxes(group, preds, IOU_MIN):
            if v >= IOU_MIN:
                out[truth_key(group[ti])] = preds[pi]["observation_id"]; used_o.add(preds[pi]["observation_id"])
    return {truth_key(r): out.get(truth_key(r)) for r in rows}


def predictions(R: dict[str, Any], oid: str | None) -> dict[str, dict[str, Any]]:
    if oid is None:
        return {"resolution": {"matched": False}, "reading": {"matched": False}, "literal": {"matched": False}}
    rd, rs = R["readings"].get(oid) or {}, R["res"].get(oid) or {}
    accepted = rs.get("decision") == "accepted" and rs.get("title")
    conf = rd.get("read_confidence")
    asserted = rd.get("title_text") or ""                 # an OCR block (visible_text) is NOT an asserted title: it scores only in the literal layer
    # An accepted catalogue title is a confident assertion only when the evidence it was resolved from reaches the same 0.85 cutoff as the
    # reading layer (the resolution's own confidence if it carries one, else the matched reading's); below it, or unknown, it is scored wrong-not-confident.
    rconf = next((v for v in (rs.get("read_confidence"), rs.get("confidence"), conf) if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
    return {"resolution": {"matched": True, "text": rs["title"] if accepted else "", "accepted": bool(accepted), "confidence": rconf,
                           "confident": bool(accepted and rconf is not None and rconf >= CONF_MIN)},
            "reading": {"matched": True, "text": asserted, "confident": bool(asserted and isinstance(conf, (int, float)) and conf >= CONF_MIN and rd.get("title_complete"))},
            "literal": {"matched": True, "text": rd.get("visible_text") or asserted, "confident": False}}


def score(rows: list[dict[str, Any]], R: dict[str, Any], truth_info: dict[str, Any]) -> dict[str, Any]:
    m = match(rows, R)
    out: dict[str, Any] = {"layers": {}, "matching": {"matched": sum(v is not None for v in m.values()), "labeled": len(rows)}}
    for layer in ("resolution", "reading", "literal"):
        groups = {g: {"n": 0, "exact": 0, "contained": 0, "recall": 0, "abstained": 0, "confidently_wrong": 0, "unmatched": 0} for g in (*LANGS, "all")}
        for r in rows:
            p = predictions(R, m[truth_key(r)])[layer]
            for g in ([r["language"]] if r["language"] in LANGS else []) + ["all"]:   # dev labels carry no language: "all" only
                c = groups[g]; c["n"] += 1
                if not p["matched"]:
                    c["unmatched"] += 1; continue
                if not p["text"].strip():
                    c["abstained"] += 1; continue
                k = compare(r["truth_title"], p["text"])
                c["exact"] += k["exact"]; c["contained"] += k["contained"]; c["recall"] += k["recall"]
                # STRICT (eval/score.py rule): an ASSERTED full title (accepted catalogue identity or title_complete reading, each with confidence >= 0.85) that
                # differs from the truth is confidently wrong; containment / partial overlap does not rescue it (Dune vs Dune Messiah).
                if not k["exact"] and p.get("confident"):
                    c["confidently_wrong"] += 1
        out["layers"][layer] = {g: {"counts": c, **{name: metric(c[key], c["n"], reason=None if c["n"] else "no human-labelled crops in this language", eligible=False)
                                                   for name, key in (("exact_title", "exact"), ("title_contained", "contained"), ("recall_ge_0_5", "recall"), ("abstained", "abstained"),
                                                                     ("confidently_wrong", "confidently_wrong"), ("unmatched", "unmatched"))}} for g, c in groups.items()}
    return out


def load_dev_labels(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """SMOKE ONLY: the model-read dev labels (eval/dev_labels/*.json, purpose 'DEV tuning labels ONLY'). Rows get frame/bbox from the labelled
    run so matching works; language is unknown (scored under 'all' only). Never a reported accuracy."""
    d = _read(path)
    if d.get("purpose") == "evaluation_truth" or "labels" not in d or not isinstance(d["labels"], list):
        raise ValueError(f"{path}: not a dev-label file")
    src = ROOT / d["run"]
    R = load_run(src)
    cands = {o["observation_id"]: o for o in R["cands"]}
    rows = []
    for l in d["labels"]:
        o = cands.get(l["observation_id"])
        if o is None or not (l.get("title") or "").strip():
            continue
        f = R["frames"][o["frame_id"]]
        rows.append({"observation_id": o["observation_id"], "frame_id": o["frame_id"], "timestamp_s": f["timestamp_s"], "bbox_xyxy": o["bbox_xyxy"],
                     "truth_title": l["title"], "language": None, "label_run": d["run"], "label_scan_sha": R["scan_sha"], "candidates_sha256": R["candidates_sha256"]})
    info = {"files": [str(path)], "excluded": {}, "unlabeled": 0, "annotators": [str((d.get("provenance") or {}).get("annotator"))], "dev_smoke": True,
            "warning": "DEV model-read labels (NOT human truth): smoke test of the scorer only; numbers are relative and must never be reported as accuracy"}
    return rows, info


def build(truth_paths: list[Path], run: Path, dev_smoke: Path | None = None) -> dict[str, Any]:
    if dev_smoke is not None:
        rows, info = load_dev_labels(dev_smoke)
        base = {"schema_version": 1, "kind": "title_score_DEV_SMOKE", "run_id": run.name, "truth": {**info, "labeled": len(rows)},
                "validation": "DEV SMOKE ONLY: model-read labels, not truth; do not report"}
        if not rows:
            return {**base, "status": "unavailable", "reason": "no dev labels matched the run", "layers": {}}
        return {**base, "status": "dev_smoke", **score(rows, load_run(run), info)}
    rows, info = load_truth(truth_paths)
    base = {"schema_version": 1, "kind": "title_score", "run_id": run.name, "truth": {**info, "labeled": len(rows)},
            "rules": __doc__.split("Metrics per language")[0].strip().splitlines()[-1], "validation": "scored against held-out human title labels; labels never used by the pipeline"}
    if info["integrity"]["status"] != "verified":
        reason = (f"truth integrity unavailable: {info['integrity']['failed']} labelled source crop(s) missing, changed or unverifiable "
                  f"(see truth.integrity_failures); scores withheld, nothing dropped silently")
        un = {k: metric(reason=reason, eligible=False) for k in ("exact_title", "title_contained", "recall_ge_0_5", "abstained", "confidently_wrong", "unmatched")}
        return {**base, "status": "unavailable", "reason": reason, "layers": {l: {g: un for g in (*LANGS, "all")} for l in ("resolution", "reading", "literal")}}
    if not rows:
        reason = "no human title labels yet: label crops in the Label titles view first"
        un = {k: metric(reason=reason, eligible=False) for k in ("exact_title", "title_contained", "recall_ge_0_5", "abstained", "confidently_wrong", "unmatched")}
        return {**base, "status": "unavailable", "reason": reason, "layers": {l: {g: un for g in (*LANGS, "all")} for l in ("resolution", "reading", "literal")}}
    try:
        R = load_run(run)
    except FileNotFoundError as e:
        reason = f"run artifacts unavailable: {Path(e.filename).name if e.filename else e} not found in {run.name}; nothing scored"
        un = {k: metric(reason=reason, eligible=False) for k in ("exact_title", "title_contained", "recall_ge_0_5", "abstained", "confidently_wrong", "unmatched")}
        return {**base, "status": "unavailable", "reason": reason, "layers": {l: {g: un for g in (*LANGS, "all")} for l in ("resolution", "reading", "literal")}}
    return {**base, "status": "scored", **score(rows, R, info)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--truth", action="append", type=Path, default=[])
    ap.add_argument("--dev-smoke", type=Path, help="SMOKE ONLY: a model-read dev-label file (eval/dev_labels/); output is marked DEV, never truth")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="directory for title_score_<run>.json (default: print only)")
    a = ap.parse_args(argv)
    if not a.truth and a.dev_smoke is None:
        ap.error("--truth <evaluation_truth file> is required (or --dev-smoke <dev-label file> for a smoke test)")
    doc = build(a.truth, a.run, a.dev_smoke)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        name = f"title_score_{'DEV_SMOKE_' if a.dev_smoke else ''}{a.run.name}.json"
        (a.out / name).write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    if a.dev_smoke:
        print("DEV SMOKE ONLY (model-read labels, not truth; never report these numbers)")
    for layer, groups in doc["layers"].items():
        for g, v in groups.items():
            n = v.get("counts", {}).get("n", 0) if "counts" in v else 0
            f = lambda k: "n/a" if not v[k]["available"] else f"{v[k]['value']:.2f}"
            print(f"{layer:10s} {g:5s} n={n:3d} exact={f('exact_title')} contained={f('title_contained')} recall>=.5={f('recall_ge_0_5')} abstained={f('abstained')} conf_wrong={f('confidently_wrong')} unmatched={f('unmatched')}")
    print(doc["status"], "-", doc.get("reason", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
