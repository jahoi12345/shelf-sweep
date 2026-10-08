"""Score attributable reviewed human labels with explicit coverage and denominators.

No label inference. Pending/partial truth cannot become a passing assignment bar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.contracts import check_bbox, validate_manifest, validate_candidates, validate_inventory, validate_readings, validate_resolutions, write_json
from pipeline.read_spines import inside, load, records
from pipeline.resolve import normalize_title

normalize = normalize_title

CONFIDENT_THRESHOLD = 0.85  # declared before tuning; run-specific override recorded
IOU_THRESHOLD = 0.5


def metric(numerator=None, denominator=None, *, reason=None, threshold=None, direction="min", eligible=True):
    available = numerator is not None and denominator is not None and denominator > 0 and reason is None
    value = numerator / denominator if available else None
    if not available and reason is None:
        reason = "zero denominator" if denominator == 0 else "numerator or denominator unavailable"
    passed = None
    if available and eligible and threshold is not None:
        passed = value >= threshold if direction == "min" else value <= threshold
    return {"numerator": numerator, "denominator": denominator, "value": value,
            "available": available, "unavailable_reason": reason if not available else None,
            "threshold": threshold, "pass": passed,
            "assignment_eligible": bool(eligible and available)}


def iou(a, b):
    x = max(0., min(a[2], b[2])-max(a[0], b[0]))
    y = max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = x*y
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union > 0 else 0.


def match_boxes(truth, predictions, threshold=IOU_THRESHOLD):
    """Maximum-cardinality, then maximum-total-IoU assignment (Hungarian).

    Match spatially before class; a pot predicted as a book remains a clutter error.
    Each truth/prediction participates at most once. Dummies represent misses.
    """
    n, p = len(truth), len(predictions)
    if not n or not p:
        return []
    m = p+n
    overlaps = [[iou(t["bbox_xyxy"], d["bbox_xyxy"]) for d in predictions] for t in truth]
    # Cardinality advantage exceeds the sum of all possible IoU gains.
    costs = [[-(n+1)-v if v >= threshold else 1. for v in row]+[0.]*n for row in overlaps]
    u, v, owner, way = [0.]*(n+1), [0.]*(m+1), [0]*(m+1), [0]*(m+1)
    for row in range(1, n+1):
        owner[0] = row
        j0 = 0
        minv, used = [math.inf]*(m+1), [False]*(m+1)
        while True:
            used[j0] = True
            i0, delta, j1 = owner[j0], math.inf, 0
            for j in range(1, m+1):
                if not used[j]:
                    cur = costs[i0-1][j-1]-u[i0]-v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m+1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while True:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(owner[j]-1, j-1, overlaps[owner[j]-1][j-1]) for j in range(1, p+1)
            if owner[j] and overlaps[owner[j]-1][j-1] >= threshold]


def annotation_problems(annotations):
    problems = []
    if annotations.get("annotation_version") != 1:
        problems.append("annotation_version must be 1")
    prov = annotations.get("provenance", {})
    if prov.get("source_type") != "human":
        problems.append("labels are not declared human ground truth")
    mode = prov.get("review_mode")
    # Legacy independently reviewed labels remain usable; self-review is explicit.
    if mode is None and prov.get("reviewer") and prov.get("reviewer") != prov.get("annotator"):
        mode = "independent"
    if not prov.get("annotator"):
        problems.append("named human annotator required")
    if mode == "self_reviewed":
        if prov.get("reviewer") not in (None, "", prov.get("annotator")):
            problems.append("self-review must be attributable to the annotator")
    elif mode == "independent":
        if not prov.get("reviewer") or prov.get("reviewer") == prov.get("annotator"):
            problems.append("claimed independent review requires a distinct named human")
    else:
        problems.append("explicit self_reviewed or attributable independent review required")
    if prov.get("review_status") != "approved" or not prov.get("frozen_at"):
        problems.append("human review is pending or labels are not frozen")
    frames = annotations.get("frames", [])
    if not frames:
        problems.append("no annotated frames")
    frame_ids, annotation_ids = set(), set()
    for frame in frames:
        fid = frame.get("frame_id")
        if not fid or fid in frame_ids:
            problems.append("missing/duplicate annotation frame ID")
        frame_ids.add(fid)
        if not frame.get("exhaustive"):
            continue
        if not frame.get("reviewed"):
            problems.append(f"{fid}: exhaustive labels not human reviewed")
        instance_ids = [o.get("instance_id") for o in frame.get("objects", []) if o.get("instance_id")]
        if len(instance_ids) != len(set(instance_ids)):
            problems.append(f"{fid}: two same-frame objects cannot share a physical instance ID")
        for obj in frame.get("objects", []):
            aid = obj.get("annotation_id")
            if not aid or aid in annotation_ids:
                problems.append(f"{fid}: missing/duplicate annotation ID")
            annotation_ids.add(aid)
            if obj.get("language_code") not in {None, "es", "en", "mul"}:
                problems.append(f"{fid}/{aid}: language_code must be es/en/mul/null")
            if obj.get("kind") not in {"book", "non_book", "ambiguous"}:
                problems.append(f"{fid}/{aid}: invalid kind")
            if not isinstance(obj.get("accepted_titles", []), list) or any(not isinstance(t, str) or not t.strip() for t in obj.get("accepted_titles", [])):
                problems.append(f"{fid}/{aid}: accepted_titles must be nonempty strings")
            problems += [f"{fid}/{aid}: {p}" for p in check_bbox(obj.get("bbox_xyxy"), frame["width_px"], frame["height_px"])]
            if obj.get("kind") == "book" and obj.get("title_legible") is True and not obj.get("accepted_titles"):
                problems.append(f"{fid}/{aid}: legible title lacks human accepted title")
    kinds_by_instance = defaultdict(set)
    for f in frames:
        for o in f.get("objects", []):
            if o.get("instance_id"):
                kinds_by_instance[o["instance_id"]].add(o.get("kind"))
    if any(len(kinds) > 1 for kinds in kinds_by_instance.values()):
        problems.append("Human physical instance has conflicting kinds across views")
    return problems


def _title_counts(truths, predicted, confidence_threshold):
    counts = Counter(correct=0, incorrect=0, confident_wrong=0, abstained=0, missed=0)
    for key, accepted in truths.items():
        prediction = predicted.get(key)
        if prediction is None:
            counts["missed"] += 1
        elif not prediction.get("title"):
            counts["abstained"] += 1
        elif normalize(prediction["title"]) in {normalize(t) for t in accepted}:
            counts["correct"] += 1
        else:
            counts["incorrect"] += 1
            conf = prediction.get("confidence")
            if isinstance(conf, (int, float)) and conf >= confidence_threshold:
                counts["confident_wrong"] += 1
    return dict(counts)


def language_scores(truths, predictions, human_languages, threshold):
    groups = {}
    for language in ("es", "en", "mul", "unknown"):
        selected = {k: titles for k, titles in truths.items()
                    if (human_languages.get(k) or "unknown") == language}
        counts = _title_counts(selected, predictions, threshold)
        n = len(selected)
        reason = None if n else "no human-legible labelled sample in this language"
        groups[language] = {"human_legible_sample_size": n, "counts": counts,
                           "correctness": metric(counts["correct"], n, reason=reason, eligible=False),
                           "abstention_rate": metric(counts["abstained"], n, reason=reason, eligible=False),
                           "miss_rate": metric(counts["missed"], n, reason=reason, eligible=False),
                           "confident_wrong_rate": metric(counts["confident_wrong"], n, reason=reason, eligible=False)}
    return groups


def unavailable_languages(reason):
    return {language: {"human_legible_sample_size": None,
                       **{key: metric(reason=reason, eligible=False) for key in
                          ("correctness", "abstention_rate", "miss_rate", "confident_wrong_rate")}}
            for language in ("es", "en", "mul", "unknown")}


def score(annotations, candidates, *, readings=None, resolutions=None, inventory=None,
          manifest=None, run_dir=None, confidence_threshold=CONFIDENT_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    label_hash = hashlib.sha256(json.dumps(annotations, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    result = {"evaluation_version": 1, "annotation_sha256": label_hash,
              "label_provenance": dict(annotations.get("provenance", {})),
              "settings": {"iou_threshold": iou_threshold, "confident_threshold": confidence_threshold,
                           "title_normalization": "NFC; casefold; punctuation/whitespace normalized; accents preserved; exact full title only",
                           "matching": "one-to-one maximum cardinality then total IoU, class-independent"},
              "coverage": dict(annotations.get("coverage", {})), "issues": [], "metrics": {},
              "stage_metrics": {a.get("stage", "unknown"): a.get("metrics", {})
                                for a in (candidates, readings, resolutions, inventory, manifest) if a}}
    problems = annotation_problems(annotations)
    exhaustive = [f for f in annotations.get("frames", []) if f.get("exhaustive") and f.get("reviewed")]
    result["coverage"].update(annotated_frame_count=len(annotations.get("frames", [])),
                                scored_frame_ids=[f["frame_id"] for f in exhaustive],
                                whole_video=annotations.get("coverage", {}).get("whole_video", False))
    eligible = bool(annotations.get("coverage", {}).get("whole_video") and annotations.get("coverage", {}).get("physical_inventory_complete"))
    fixture_inputs = any(a.get("fixture") for a in (candidates, readings, resolutions, inventory, manifest) if a)
    result["fixture_inputs"] = fixture_inputs
    if fixture_inputs:
        eligible = False
        result["issues"].append("Synthetic fixture inputs; these are not measured real-room results")
    # Evidence validity is independently checkable even while human labels are pending.
    if manifest is not None:
        prediction_ids = {d.get("observation_id") for d in records(candidates, "observations")}
        independent_errors = validate_manifest(manifest, run_dir) + validate_candidates(candidates, manifest)
        if readings is not None:
            independent_errors += validate_readings(readings, prediction_ids)
        if resolutions is not None:
            independent_errors += validate_resolutions(resolutions, prediction_ids)
        if inventory is not None:
            independent_errors += validate_inventory(inventory, manifest, prediction_ids)
        for rd in records(readings or {}, "readings"):
            if run_dir is not None and rd.get("crop_path"):
                try:
                    if not inside(run_dir, rd["crop_path"]).is_file():
                        independent_errors.append(f"{rd['observation_id']}: missing crop evidence")
                except ValueError:
                    independent_errors.append(f"{rd['observation_id']}: crop path escapes run")
        result["evidence_validation"] = {"available": True, "problems": independent_errors,
                                          "candidate_denominator": len(records(candidates, "observations")),
                                          "frame_denominator": len(records(manifest, "frames"))}
    else:
        result["evidence_validation"] = {"available": False, "problems": [],
                                          "unavailable_reason": "manifest missing"}
    if problems or not exhaustive:
        reason = "; ".join(problems or ["no exhaustively labelled reviewed frames"])
        result.update(status="unavailable", issues=problems)
        bars = {"book_count_error": (0.05, "max"), "title_correctness": (0.70, "min"), "confident_wrong": (0.03, "max"), "non_book_correct_category_coverage": (0.80, "min")}
        result["metrics"] = {k: metric(reason=reason, eligible=False, threshold=bars.get(k, (None, "min"))[0], direction=bars.get(k, (None, "min"))[1]) for k in (
            "book_count_error", "book_detection_precision", "book_detection_recall", "title_correctness",
            "confident_wrong", "non_book_correct_category_coverage", "duplicate_instances", "split_instances", "merged_instances",
            "clutter_false_positive_rate", "duplicate_predictions")}
        result["clutter_false_positives"] = {"available": False, "counts_by_category": None,
                                            "denominator_all_book_predictions": None,
                                            "total_false_book_predictions": None,
                                            "unavailable_reason": reason}
        result["language_scores"] = {"literal_observations": unavailable_languages(reason),
                                      "catalogue_observations": unavailable_languages(reason),
                                      "inventory_instances": unavailable_languages(reason)}
        return result
    result["status"] = "scored" if eligible else "development_subset"
    detections = records(candidates, "observations")
    reading_by_id = {r["observation_id"]: r for r in records(readings or {}, "readings")}
    resolution_by_id = {r["observation_id"]: r for r in records(resolutions or {}, "resolutions")}
    matched_truth, matched_predictions, observation_to_truth = {}, {}, {}
    details, matches, invalid, duplicates = [], [], [], []
    all_ids = [d.get("observation_id") for d in detections]
    duplicates = sorted(k for k, n in Counter(all_ids).items() if n > 1)
    scored_frame_ids = {f["frame_id"] for f in exhaustive}
    excluded_frames = sorted({d["frame_id"] for d in detections} - scored_frame_ids)
    result["coverage"]["unannotated_prediction_frames"] = excluded_frames
    if excluded_frames:
        eligible = False
        result["issues"].append("Predictions outside annotated coverage; whole-video assignment scores withheld")
    evidence_errors = []
    if manifest is not None:
        evidence_errors += validate_manifest(manifest, run_dir)
        manifest_frames = {f["frame_id"]: f for f in records(manifest, "frames")}
    else:
        manifest_frames = {}
    if manifest is not None:
        missing_selected = sorted({f["frame_id"] for f in records(manifest, "frames") if f.get("selected")} - scored_frame_ids)
        result["coverage"]["unannotated_selected_frames"] = missing_selected
        if missing_selected:
            eligible = False
        evidence_errors += validate_candidates(candidates, manifest)
        candidate_ids = set(all_ids)
        if inventory is not None:
            evidence_errors += validate_inventory(inventory, manifest, candidate_ids)
        if readings is not None:
            evidence_errors += validate_readings(readings, candidate_ids)
        if resolutions is not None:
            evidence_errors += validate_resolutions(resolutions, candidate_ids)
            from pipeline.read_spines import sanitize_reading
            from pipeline.resolve import isbn_value
            for rs in records(resolutions, "resolutions"):
                if rs.get("decision") != "accepted":
                    continue
                rd = reading_by_id.get(rs["observation_id"])
                if not rd or not sanitize_reading(rd)["title_text"] or normalize(rd.get("title_text")) != normalize(rs.get("title")):
                    evidence_errors.append(f"{rs['observation_id']}: accepted work lacks complete literal title support")
                if rs.get("edition") or rs.get("isbn"):
                    printed = isbn_value(sanitize_reading(rd or {}).get("isbn_text"))
                    edition_evidence = [e for e in rs.get("matching_evidence", []) if e.get("field") == "edition_isbn" and e.get("source_url")]
                    if not printed or printed != rs.get("isbn") or not edition_evidence:
                        evidence_errors.append(f"{rs['observation_id']}: unsupported edition/ISBN")
        for rd in records(readings or {}, "readings"):
            if run_dir is not None and rd.get("crop_path"):
                try:
                    if not inside(run_dir, rd["crop_path"]).is_file():
                        evidence_errors.append(f"{rd['observation_id']}: missing crop evidence")
                except ValueError:
                    evidence_errors.append(f"{rd['observation_id']}: crop path escapes run")
    book_tp = book_gt = book_pred = non_gt = non_tp = non_category_tp = 0
    clutter_fp, duplicate_predictions = Counter(), 0
    legible_truth, visible_predictions, resolved_predictions, human_languages = {}, {}, {}, {}
    unsupported_unreadable = 0
    scoped_detections = []
    for frame in exhaustive:
        fid, truth = frame["frame_id"], frame.get("objects", [])
        if manifest is not None:
            mf = manifest_frames.get(fid)
            if mf is None or (mf["width_px"], mf["height_px"]) != (frame["width_px"], frame["height_px"]):
                evidence_errors.append(f"{fid}: annotation/manifest frame missing or dimensions differ")
            elif run_dir is not None and frame.get("image_sha256"):
                if hashlib.sha256(inside(run_dir, mf["image_path"]).read_bytes()).hexdigest() != frame["image_sha256"]:
                    evidence_errors.append(f"{fid}: original annotated frame content changed")
        predictions = []
        for detection in [d for d in detections if d.get("frame_id") == fid]:
            scoped_detections.append(detection)
            box_problems = check_bbox(detection.get("bbox_xyxy"), frame["width_px"], frame["height_px"])
            if box_problems:
                invalid.append(detection.get("observation_id"))
                evidence_errors += [f"{detection.get('observation_id')}: {p}" for p in box_problems]
            elif detection.get("observation_id") not in duplicates:
                predictions.append(detection)
        # Invalid predictions still enter precision's denominator.
        book_gt += sum(t.get("kind") == "book" for t in truth)
        book_pred += sum(d.get("kind") == "book" for d in scoped_detections if d["frame_id"] == fid)
        non_gt += sum(t.get("kind") == "non_book" for t in truth)
        assignment = match_boxes(truth, predictions, iou_threshold)
        frame_matches = {ti: pi for ti, pi, _ in assignment}
        used = {pi for _, pi, _ in assignment}
        for ti, pi, overlap in assignment:
            t, d = truth[ti], predictions[pi]
            oid = d["observation_id"]
            matched_truth[t["annotation_id"]] = d
            matched_predictions[oid] = t
            observation_to_truth[oid] = t
            matches.append({"annotation_id": t["annotation_id"], "observation_id": oid, "iou": overlap})
            if d.get("kind") == "book":
                if t.get("kind") == "book":
                    book_tp += 1
                else:
                    clutter_fp[t.get("category") or t.get("kind", "unknown")] += 1
            if t.get("kind") == "non_book" and d.get("kind") == "non_book":
                non_tp += 1
                if t.get("category") and t["category"] == d.get("category"):
                    non_category_tp += 1
        for pi, d in enumerate(predictions):
            if pi not in used and d.get("kind") == "book":
                overlapping = [t for t in truth if iou(t["bbox_xyxy"], d["bbox_xyxy"]) >= iou_threshold]
                if any(t.get("kind") == "book" for t in overlapping):
                    duplicate_predictions += 1
                    clutter_fp["duplicate_book_detection"] += 1
                elif overlapping:
                    clutter_fp[overlapping[0].get("category") or "ambiguous"] += 1
                else:
                    clutter_fp["unmatched_background_or_localization"] += 1
        for t in truth:
            if t.get("kind") != "book":
                continue
            aid = t["annotation_id"]
            detection = matched_truth.get(aid)
            if t.get("title_legible") is True:
                legible_truth[aid] = t["accepted_titles"]
                human_languages[aid] = t.get("language_code") if t.get("language_code") in {"es", "en", "mul"} else None
                if detection is not None and detection.get("kind") == "book":
                    oid = detection["observation_id"]
                    rd, rs = reading_by_id.get(oid, {}), resolution_by_id.get(oid, {})
                    visible_predictions[aid] = {"title": rd.get("title_text"), "confidence": rd.get("read_confidence")}
                    resolved_predictions[aid] = {"title": rs.get("title") if rs.get("decision") == "accepted" else None,
                                                "confidence": rd.get("read_confidence")}
            elif t.get("title_legible") is False and detection:
                rs = resolution_by_id.get(detection["observation_id"], {})
                unsupported_unreadable += int(rs.get("decision") == "accepted" and bool(rs.get("title")))
    m = result["metrics"]
    m["book_detection_precision"] = metric(book_tp, book_pred, eligible=False)
    m["book_detection_recall"] = metric(book_tp, book_gt, eligible=False)
    m["non_book_detection_coverage"] = metric(non_tp, non_gt, eligible=False)
    m["non_book_correct_category_coverage_observations"] = metric(non_category_tp, non_gt, eligible=False)
    result["language_scores"] = {"literal_observations": language_scores(legible_truth, visible_predictions, human_languages, confidence_threshold),
                                  "catalogue_observations": language_scores(legible_truth, resolved_predictions, human_languages, confidence_threshold)}
    if readings is None:
        result["language_scores"]["literal_observations"] = unavailable_languages("reading artifact missing")
    if resolutions is None:
        result["language_scores"]["catalogue_observations"] = unavailable_languages("resolution artifact missing")
    counts = _title_counts(legible_truth, visible_predictions, confidence_threshold)
    resolved_counts = _title_counts(legible_truth, resolved_predictions, confidence_threshold)
    result["legibility_counts"] = {"human_legible_observations": len(legible_truth),
                                   "human_unreadable_observations": sum(o.get("title_legible") is False for f in exhaustive for o in f.get("objects", []) if o["kind"] == "book"),
                                   "unknown_legibility_observations": sum(o.get("title_legible") is None for f in exhaustive for o in f.get("objects", []) if o["kind"] == "book")}
    result["title_observations"] = {"literal": counts, "catalogue": resolved_counts, "human_legible_denominator": len(legible_truth)}
    m["literal_title_correctness_observations"] = metric(counts["correct"], len(legible_truth), eligible=False)
    m["catalogue_title_correctness_observations"] = metric(resolved_counts["correct"], len(legible_truth), eligible=False)
    m["confident_wrong_observations"] = metric(counts["confident_wrong"], len(legible_truth), eligible=False)
    if readings is None:
        m["literal_title_correctness_observations"] = metric(reason="reading artifact missing", eligible=False)
        m["confident_wrong_observations"] = metric(reason="reading artifact missing", eligible=False)
    if resolutions is None:
        m["catalogue_title_correctness_observations"] = metric(reason="resolution artifact missing", eligible=False)
    clutter_fp["invalid_box_or_duplicate_id"] += sum(d.get("kind") == "book" and (d.get("observation_id") in invalid or d.get("observation_id") in duplicates) for d in scoped_detections)
    clutter_fp += Counter()  # suppress zero categories
    result["clutter_false_positives"] = {"available": True, "counts_by_category": dict(clutter_fp), "denominator_all_book_predictions": book_pred,
                                        "total_false_book_predictions": book_pred-book_tp}
    m["clutter_false_positive_rate"] = metric(book_pred-book_tp, book_pred, eligible=False)
    m["duplicate_predictions"] = metric(duplicate_predictions, book_pred, eligible=False)
    result["unsupported_titles_on_human_unreadable_observations"] = unsupported_unreadable
    # Physical truth: IDs explicitly assigned by humans across the exhaustive frames.
    truth_instances = defaultdict(list)
    for frame in exhaustive:
        for obj in frame.get("objects", []):
            if obj.get("instance_id"):
                truth_instances[obj["instance_id"]].append(obj)
    books = [t for t in truth_instances if truth_instances[t][0]["kind"] == "book"]
    items = [t for t in truth_instances if truth_instances[t][0]["kind"] == "non_book"]
    instance_gap = any(not t.get("instance_id") for f in exhaustive for t in f.get("objects", []) if t["kind"] in {"book", "non_book"})
    physical_ready = annotations.get("coverage", {}).get("physical_instances_complete") and not instance_gap
    if inventory is None or not physical_ready:
        reason = "inventory missing" if inventory is None else "human physical-instance association incomplete"
        result["language_scores"]["inventory_instances"] = unavailable_languages(reason)
        for k in ("book_count_error", "title_correctness", "confident_wrong", "non_book_correct_category_coverage",
                  "duplicate_instances", "split_instances", "merged_instances"):
            bars = {"book_count_error": (0.05, "max"), "title_correctness": (0.70, "min"), "confident_wrong": (0.03, "max"), "non_book_correct_category_coverage": (0.80, "min")}
            m[k] = metric(reason=reason, eligible=False, threshold=bars.get(k, (None, "min"))[0], direction=bars.get(k, (None, "min"))[1])
    else:
        predicted_books = records(inventory, "books")
        predicted_items = records(inventory, "items")
        scoped_ids = {d["observation_id"] for d in scoped_detections}
        pbooks = [p for p in predicted_books if set(p.get("observation_ids", [])) & scoped_ids]
        pitems = [p for p in predicted_items if set(p.get("observation_ids", [])) & scoped_ids]
        assignment_by_truth, predicted_truth, oid_owners = defaultdict(set), {}, defaultdict(set)
        for p in pbooks+pitems:
            iid = p["instance_id"]
            tids = set()
            for oid in p.get("observation_ids", []):
                oid_owners[oid].add(iid)
                obj = observation_to_truth.get(oid)
                if obj and obj.get("instance_id"):
                    tids.add(obj["instance_id"])
            predicted_truth[iid] = tids
            for tid in tids:
                assignment_by_truth[tid].add(iid)
        count_error = abs(len(pbooks)-len(books))
        m["book_count_error"] = metric(count_error, len(books), threshold=0.05, direction="max", eligible=eligible)
        result["book_counts"] = {"predicted": len(pbooks), "human": len(books), "signed_error": len(pbooks)-len(books)}
        splits = sum(len(ps) > 1 for ps in assignment_by_truth.values())
        extra_instances = sum(max(0, len(ps)-1) for ps in assignment_by_truth.values())
        merges = sum(len(ts) > 1 for ts in predicted_truth.values())
        m["split_instances"] = metric(splits, len(truth_instances), eligible=False)
        m["duplicate_instances"] = metric(extra_instances, len(pbooks)+len(pitems), eligible=False)
        m["merged_instances"] = metric(merges, len(pbooks)+len(pitems), eligible=False)
        result["association_errors"] = {"truth_to_predicted": {t: sorted(ps) for t, ps in assignment_by_truth.items()},
                                         "predicted_to_truth": {p: sorted(ts) for p, ts in predicted_truth.items()},
                                         "observations_in_multiple_instances": {o: sorted(ps) for o, ps in oid_owners.items() if len(ps) > 1}}
        title_truth, title_predictions, instance_languages = {}, {}, {}
        p_by_id = {p["instance_id"]: p for p in pbooks}
        for tid in books:
            accepted = {title for obj in truth_instances[tid] if obj.get("title_legible") is True for title in obj.get("accepted_titles", [])}
            if not accepted:
                continue
            title_truth[tid] = sorted(accepted)
            langs = {o.get("language_code") for o in truth_instances[tid] if o.get("language_code") in {"es", "en", "mul"}}
            instance_languages[tid] = next(iter(langs)) if len(langs) == 1 else ("mul" if langs else None)
            ps = assignment_by_truth.get(tid, set()) & set(p_by_id)
            if not ps:
                continue
            # A split counts once; conflicting identities count incorrect, never pick a convenient title.
            titles = {(p_by_id[p].get("identity") or p_by_id[p]).get("title") for p in ps if (p_by_id[p].get("identity") or p_by_id[p]).get("title")}
            confidence = max([(p_by_id[p].get("identity") or p_by_id[p]).get("id_confidence", p_by_id[p].get("confidence")) or 0 for p in ps], default=0)
            title_predictions[tid] = {"title": next(iter(titles)) if len(titles) == 1 else ("[conflicting instance titles]" if titles else None),
                                      "confidence": confidence}
        result["language_scores"]["inventory_instances"] = language_scores(title_truth, title_predictions, instance_languages, confidence_threshold)
        physical_counts = _title_counts(title_truth, title_predictions, confidence_threshold)
        result["title_instances"] = {**physical_counts, "human_legible_denominator": len(title_truth)}
        unknown_legibility = sum(not any(o.get("title_legible") is True for o in truth_instances[t]) and not all(o.get("title_legible") is False for o in truth_instances[t]) for t in books)
        result["title_instances"]["unknown_legibility_instances"] = unknown_legibility
        title_eligible = eligible and unknown_legibility == 0
        m["title_correctness"] = metric(physical_counts["correct"], len(title_truth), threshold=0.70, eligible=title_eligible)
        m["confident_wrong"] = metric(physical_counts["confident_wrong"], len(title_truth), threshold=0.03, direction="max", eligible=title_eligible)
        non_correct = 0
        for tid in items:
            categories = {t.get("category") for t in truth_instances[tid] if t.get("category")}
            non_correct += int(any(p["instance_id"] in assignment_by_truth.get(tid, set()) and p.get("category") in categories for p in pitems))
        category_unknown = sum(not any(t.get("category") for t in truth_instances[tid]) for tid in items)
        m["non_book_correct_category_coverage"] = metric(non_correct, len(items), threshold=0.80, eligible=eligible and category_unknown == 0)
        result["unknown_non_book_category_instances"] = category_unknown
    result["status"] = "scored" if eligible else "development_subset"
    if fixture_inputs:
        result["status"] = "synthetic_fixture"
    result["matches"] = matches
    result["issues"] += evidence_errors
    result["evidence_validation"] = {"invalid_detection_ids": invalid, "duplicate_observation_ids": duplicates,
                                      "problems": evidence_errors}
    if duplicates or evidence_errors:
        result["status"] = "invalid_evidence"
        for value in m.values():
            value["assignment_eligible"] = False
            value["pass"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--candidates", required=True)
    for name in ("readings", "resolutions", "inventory", "manifest", "run-dir"):
        parser.add_argument("--"+name)
    parser.add_argument("--out", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENT_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error("Preserve baseline: --out already exists; choose a new results path")
    kwargs = {name: load(getattr(args, name)) if getattr(args, name) else None
              for name in ("readings", "resolutions", "inventory", "manifest")}
    result = score(load(args.annotations), load(args.candidates), run_dir=args.run_dir,
                   confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold, **kwargs)
    write_json(out, result)
    print(json.dumps({"status": result["status"], "out": str(out), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
