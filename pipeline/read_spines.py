"""Native evidence crops and literal transcription; no catalogue inference here."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline.contracts import envelope, write_json

PROMPT = """Transcribe ONLY text you can actually see on this one candidate book spine.
Read vertical, rotated, leaning or horizontal text. Do not infer from appearance,
prior knowledge, neighboring books or a catalogue. Preserve literal spelling and
uncertain characters as [?]. visible_text is the literal transcription, or null
when unreadable. title_text, author_text and publisher_text must be literal
substrings of that transcription, with no completion or correction. A fragment
is allowed as visible_text, but set title_text to null unless the entire title is
readable. Return title_complete as a boolean. isbn_text must be printed and fully
readable; edition_text must be explicit printed edition wording. Use null for
unknowns. Confidence is reading confidence, not confidence in a familiar title.
Include concrete issues (blur, tiny_text, occlusion, partial_title, not_a_book).
Spanish and English have equal priority. Preserve accents, ñ, original language,
printed edition wording and literal ISBN. Do not translate or anglicize text.
language_code describes readable text (es, en, mul or null), not proven edition
language. language_evidence must quote a visible substring supporting that text
language. Use null if too little text supports it. Never derive language from an
author name alone. No prices, normalized title, catalogue match or guessed author."""

READ_SCHEMA = {
    "type": "object", "properties": {
        **{k: {"type": "string", "nullable": True} for k in (
            "visible_text", "title_text", "author_text", "publisher_text",
            "edition_text", "isbn_text", "language_evidence")},
        "language_code": {"type": "string", "nullable": True, "enum": ["es", "en", "mul"]},
        "title_complete": {"type": "boolean"},
        "read_confidence": {"type": "number", "nullable": True, "minimum": 0, "maximum": 1},
        "issues": {"type": "array", "items": {"type": "string"}},
    }, "required": ["visible_text", "title_text", "author_text", "publisher_text",
                      "title_complete", "read_confidence", "issues"],
}


def load(path):
    return json.loads(Path(path).read_text())


def records(artifact, key):
    data = artifact.get("data", artifact)
    if isinstance(data, list):
        return data
    return data.get(key, data.get("detections", []) if key == "observations" else [])


def inside(run_dir, relative):
    """Reject evidence escaping the run directory, including symlinks."""
    root = Path(run_dir).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence path must remain inside run directory")
    return path


def ordered_quad(points):
    """Order four convex geometric corners; no claim about printed text direction."""
    q = np.asarray(points, dtype=np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all() or len({tuple(p) for p in q}) != 4:
        raise ValueError("Polygon is not four distinct finite corners")
    center = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:, 1]-center[1], q[:, 0]-center[0]))]
    if not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < 4:
        raise ValueError("Polygon is not a convex quadrilateral")
    # A tie in the upper-left score is ambiguous; preserve bbox instead of guessing.
    scores = q[:, 0]+q[:, 1]
    if np.count_nonzero(np.isclose(scores, scores.min(), atol=.001, rtol=0)) != 1:
        raise ValueError("Polygon corner order is geometrically ambiguous")
    q = np.roll(q, -int(np.argmin(scores)), axis=0)
    if cv2.contourArea(q, oriented=True) < 0:
        q = q[[0, 3, 2, 1]]
    return q.tolist()


def native_crop(image, observation):
    """Return pixels and a 3x3 crop->native-frame transform. No upsampling.

    Optional rectification requires explicitly ordered TL,TR,BR,BL corners;
    an unordered segmentation polygon alone is deliberately not rectified.
    """
    height, width = image.shape[:2]
    box = observation["bbox_xyxy"]
    if len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
        raise ValueError("Invalid native bbox")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Box outside native frame or empty")
    x0, y0, x1, y1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)
    crop = image[y0:y1, x0:x1].copy()
    transform = np.array([[1., 0, x0], [0, 1., y0], [0, 0, 1.]])
    method = "bbox_native"
    quad = observation.get("rectify_quad_xy")
    if quad is not None:
        q = np.asarray(quad, dtype=np.float32)
        if q.shape != (4, 2) or not np.isfinite(q).all():
            raise ValueError("Rectification requires four finite ordered corners")
        if (q[:, 0] < x0).any() or (q[:, 0] > x1).any() or (q[:, 1] < y0).any() or (q[:, 1] > y1).any():
            raise ValueError("Rectification corners outside detection box")
        if not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < 4:
            raise ValueError("Degenerate/non-convex rectification quad")
        cw = max(2, math.floor(min(np.linalg.norm(q[1]-q[0]), np.linalg.norm(q[2]-q[3]))))
        ch = max(2, math.floor(min(np.linalg.norm(q[3]-q[0]), np.linalg.norm(q[2]-q[1]))))
        target = np.float32([[0, 0], [cw-1, 0], [cw-1, ch-1], [0, ch-1]])
        forward = cv2.getPerspectiveTransform(q, target)
        crop = cv2.warpPerspective(image, forward, (cw, ch))
        transform = np.linalg.inv(forward)
        method = "ordered_quad_rectified_no_upsampling"
    # Rotation is explicit text geometry, not inferred from shelf orientation.
    rotation = observation.get("text_rotation_deg", 0)
    h, w = crop.shape[:2]
    if rotation == 90:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        inverse = np.array([[0., 1, 0], [-1, 0, h-1], [0, 0, 1.]])
    elif rotation == 180:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
        inverse = np.array([[-1., 0, w-1], [0, -1, h-1], [0, 0, 1.]])
    elif rotation == 270:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        inverse = np.array([[0., -1, w-1], [1, 0, 0], [0, 0, 1.]])
    elif rotation == 0:
        inverse = np.eye(3)
    else:
        raise ValueError("text_rotation_deg must be 0, 90, 180 or 270")
    return crop, {"method": method, "bbox_xyxy": [x0, y0, x1, y1],
                  "rotation_deg_clockwise": rotation,
                  "crop_to_frame_3x3": (transform @ inverse).tolist(),
                  "crop_width_px": crop.shape[1], "crop_height_px": crop.shape[0],
                  "resampled": quad is not None, "upscaled": False}


def sanitize_reading(payload):
    """Reject nonliteral extracted fields and incomplete titles independently of confidence."""
    if not isinstance(payload, dict):
        raise ValueError("Reader response must be an object")
    result = {}
    issues = [str(x) for x in payload.get("issues", [])]
    for key in ("visible_text", "title_text", "author_text", "publisher_text", "edition_text", "isbn_text"):
        value = payload.get(key)
        result[key] = value.strip() if isinstance(value, str) and value.strip() else None
    visible = result["visible_text"]
    for key in ("title_text", "author_text", "publisher_text", "edition_text", "isbn_text"):
        if result[key] and (not visible or result[key] not in visible or "[?]" in result[key]):
            result[key] = None
            issues.append(f"{key}_not_literal_or_uncertain")
    result["title_complete"] = payload.get("title_complete") is True
    if not result["title_complete"]:
        result["title_text"] = None
        issues.append("title_not_fully_readable")
    confidence = payload.get("read_confidence")
    result["read_confidence"] = (float(confidence) if isinstance(confidence, (int, float))
                                 and not isinstance(confidence, bool) and math.isfinite(confidence)
                                 and 0 <= confidence <= 1 else None)
    if visible is None:
        for key in ("title_text", "author_text", "publisher_text", "edition_text", "isbn_text"):
            result[key] = None
        result["title_complete"] = False
        issues.append("unreadable_spine")
    evidence = payload.get("language_evidence")
    language = payload.get("language_code")
    if language in {"es", "en", "mul"} and isinstance(evidence, str) and evidence.strip() and visible and evidence.strip() in visible and "[?]" not in evidence:
        result["language_code"] = language
        result["language_evidence"] = evidence.strip()
        result["language_provenance"] = "model_observation_of_visible_text_not_edition_or_human_label"
    else:
        result["language_code"] = result["language_evidence"] = result["language_provenance"] = None
        if language is not None:
            issues.append("language_unsupported_by_literal_text")
    result["issues"] = sorted(set(issues))
    return result


async def _read(candidates_path, manifest_path, run_dir, config):
    started = time.perf_counter()
    config = {**config, **config.get("read", {})}
    owned = None
    if not config.get("no_provider") and not config.get("image_request") and config.get("provider") == "local_vision":
        from providers.local_ocr import LocalOCR
        owned = await asyncio.to_thread(LocalOCR, config.get("ocr_runtime_dir", Path(run_dir)/"ocr_runtime"))
        config = {**config, "image_request": owned}
    elif not config.get("no_provider") and not config.get("image_request") and config.get("provider") == "local_easyocr":
        from providers.easyocr_reader import EasyOCRAdapter
        if not config.get("ocr_model_dir"):
            raise ValueError("local_easyocr requires ocr_model_dir with existing pretrained latin_g2.pth")
        owned = await asyncio.to_thread(EasyOCRAdapter, config["ocr_model_dir"],
                                       evidence_dir=config.get("ocr_evidence_dir", Path(run_dir)/"easyocr_responses"))
        config = {**config, "image_request": owned}
    try:
        path = await _read_with_adapter(candidates_path, manifest_path, run_dir, config)
        adapter = config.get("image_request")
        if adapter is not None and getattr(adapter, "runtime", None):
            artifact = load(path)
            artifact["metrics"].update(adapter_runtime=adapter.runtime,
                                       adapter_setup_s=adapter.setup_s if owned is not None else 0,
                                       adapter_lifecycle="created_by_stage" if owned is not None else "caller_owned_reused",
                                       elapsed_s_including_setup=time.perf_counter()-started)
            write_json(path, artifact)
        return path
    finally:
        if owned is not None:
            owned.close()


async def _read_with_adapter(candidates_path, manifest_path, run_dir, config):
    started = time.perf_counter()
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidates, manifest = load(candidates_path), load(manifest_path)
    if candidates.get("run_id") != manifest.get("run_id"):
        raise ValueError("Candidate/manifest run IDs differ; do not mix evidence runs")
    candidate_hash = hashlib.sha256(Path(candidates_path).read_bytes()).hexdigest()
    frames = {f["frame_id"]: f for f in records(manifest, "frames")}
    observations = records(candidates, "observations")
    ids = [o["observation_id"] for o in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate observation IDs")
    adapter = None if config.get("no_provider") else config.get("image_request")
    shared_usage = None
    if adapter is None and not config.get("no_provider") and config.get("provider") == "gemini":
        from providers.gemini import make_adapter, get_cache, UsageLog
        shared_usage = config.get("usage") or UsageLog(root / "reading_usage.jsonl")
        cache = config.get("cache") or get_cache(config.get("cache_dir", root / "cache"))
        adapter = make_adapter(run_dir=root, cache=cache, usage=shared_usage,
                               model=config.get("model"), default_stage="read",
                               config=config.get("generation_config", {"temperature": 0.0}))
    readings, stage_issues, usage = [], [], []
    max_books = config.get("max_books")
    if max_books is not None and (not isinstance(max_books, int) or isinstance(max_books, bool) or max_books < 0):
        raise ValueError("max_books must be a nonnegative integer")
    selected_ids = config.get("attempt_observation_ids")
    if selected_ids is not None:
        selected_ids = set(selected_ids)
        if not selected_ids.issubset({o['observation_id'] for o in observations if o.get('kind') == 'book'}):
            raise ValueError('OCR selection contains unknown/non-book observations')
    book_index = 0
    provider_attempts = 0
    for observation in observations:
        if observation.get("kind") != "book":
            continue
        current_index = book_index
        book_index += 1
        base = {"observation_id": observation["observation_id"], "frame_id": observation["frame_id"],
                "crop_path": None, "crop_transform": None, "visible_text": None,
                "title_text": None, "author_text": None, "publisher_text": None,
                "edition_text": None, "isbn_text": None, "title_complete": False,
                "read_confidence": None, "language_code": None, "language_evidence": None,
                "language_provenance": None, "issues": [], "ocr_attempt": "not_attempted",
                "provenance": {"run_id": candidates.get("run_id"), "candidate_artifact_sha256": candidate_hash,
                    "source_turn_id": observation.get("source_turn_id", config.get("source_turn_id")),
                    "session_revision": observation.get("session_revision", config.get("session_revision")),
                    "source_type": "candidate_native_crop", "provider_evidence": None}}
        try:
            frame = frames[observation["frame_id"]]
            source = inside(root, frame["image_path"])
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Missing/unreadable source image")
            if image.shape[:2] != (frame["height_px"], frame["width_px"]):
                raise ValueError("Saved frame dimensions differ from manifest")
            geometry = dict(observation)
            if config.get("rectify_polygons") and observation.get("polygon_xy") and not observation.get("rectify_quad_xy"):
                try:
                    geometry["rectify_quad_xy"] = ordered_quad(observation["polygon_xy"])
                except ValueError as exc:
                    base["issues"].append(f"polygon_rectification_abstained:{exc}")
            crop, transform = native_crop(image, geometry)
            frame_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            token = hashlib.sha256(json.dumps({"observation_id": observation["observation_id"],
                                                "frame_sha256": frame_hash, "transform": transform}, sort_keys=True).encode()).hexdigest()[:20]
            base["source_polygon_xy"] = observation.get("polygon_xy")
            if geometry.get("rectify_quad_xy"):
                transform["ordered_source_quad_xy"] = geometry["rectify_quad_xy"]
            relative = f"{config.get('crop_subdir', 'crops')}/{token}.png"
            inside(root, relative).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(inside(root, relative)), crop):
                raise ValueError("Could not save crop")
            base["provenance"].update(frame_sha256=frame_hash,
                                      crop_sha256=hashlib.sha256(inside(root, relative).read_bytes()).hexdigest())
            base.update(crop_path=relative, crop_transform=transform,
                        frame_ref=frame["frame_id"], frame_image_path=frame["image_path"], timestamp_s=frame["timestamp_s"])
            if adapter is None:
                base["issues"].append("provider_unavailable_crop_only")
            elif selected_ids is not None and observation['observation_id'] not in selected_ids:
                base['issues'].append('outside_bounded_ocr_selection')
            elif max_books is not None and provider_attempts >= max_books:
                base["issues"].append("reading_budget_limit")
            else:
                provider_attempts += 1
                base['ocr_attempt'] = 'failed'
                response = adapter(image_paths=[inside(root, relative)], prompt=PROMPT,
                                   response_schema=READ_SCHEMA, task="read", config={**config, **config.get("generation_config", {})})
                if inspect.isawaitable(response):
                    response = await response
                # Shared adapter returns payload+metrics; fixtures can return a raw payload.
                payload = response.get("data", response)
                normalized = sanitize_reading(payload)
                normalized["issues"] = sorted(set(base["issues"] + normalized["issues"]))
                base.update(normalized)
                base['ocr_attempt'] = 'completed'
                provider_metrics = response.get("metrics", {})
                base["provenance"]["provider_evidence"] = provider_metrics
                base["provenance"]["request_ref"] = response.get("request_ref")
                if response.get("ocr_evidence"):
                    base["ocr_evidence"] = response["ocr_evidence"]
                usage.append(provider_metrics)
        except Exception as exc:
            base["issues"].append(f"reading_failed:{type(exc).__name__}:{exc}")
            stage_issues.append({"ref_id": observation["observation_id"], "reason": base["issues"][-1]})
        readings.append(base)
    if shared_usage is not None:
        metrics_summary = shared_usage.summary(stage="read")
    else:
        metrics_summary = None
    if adapter is None:
        stage_issues.append({"reason": "No shared image provider configured; crops saved, text unknown"})
    local_only = bool(usage) and all(u.get("request_count") == 0 and u.get("actual_cost") == 0 and str(u.get("provider", "")).startswith("local_") for u in usage)
    metrics = {"elapsed_s": time.perf_counter()-started, "book_observations": len(readings),
               "crops_saved": sum(r["crop_path"] is not None for r in readings),
               "observed_text_language_counts": {lang: sum((r.get("language_code") or "unknown") == lang for r in readings) for lang in ("es", "en", "mul", "unknown")},
               "provider_attempts": provider_attempts, "provider_metrics": usage, "request_count": sum(u.get("request_count", 0) for u in usage),
               "local_inference_count": sum(u.get("local_inference_count", 0) for u in usage),
               "actual_cost": 0 if provider_attempts == 0 or local_only else None,
               "cost_provenance": "Local OCR; no API calls" if local_only else ("Local crop-only; no API calls" if provider_attempts == 0 else "Shared provider records; failed request cost not assumed")}
    if metrics_summary is not None:
        metrics["shared_usage"] = metrics_summary
    artifact = envelope(candidates.get("run_id", root.name), "read", {"readings": readings},
                        issues=stage_issues, metrics=metrics, status="partial" if stage_issues or any(r["issues"] for r in readings) else "ok")
    artifact["context"] = {"source_turn_id": config.get("source_turn_id"),
                           "session_revision": config.get("session_revision")}
    artifact['inputs'] = {'candidates_sha256': candidate_hash,
                          'manifest_sha256': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}
    if candidates.get("fixture") or manifest.get("fixture"):
        artifact["fixture"] = True
    path = root / config.get("readings_filename", "readings.json")
    write_json(path, artifact)
    return path


def read(candidates_path, manifest_path, run_dir, config=None):
    config = config or {}
    return asyncio.run(_read(candidates_path, manifest_path, run_dir, {**config, **config.get("read", {})}))


read_async = _read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--provider", choices=["none", "gemini", "local_vision", "local_easyocr"], default="none")
    parser.add_argument("--ocr-runtime-dir")
    parser.add_argument("--ocr-model-dir")
    parser.add_argument("--model")
    parser.add_argument("--max-books", type=int)
    parser.add_argument("--output-name", default="readings.json")
    parser.add_argument("--rectify-polygons", action="store_true")
    args = parser.parse_args()
    config = {"provider": args.provider, "model": args.model, "max_books": args.max_books, "readings_filename": args.output_name, "rectify_polygons": args.rectify_polygons}
    if args.ocr_runtime_dir:
        config["ocr_runtime_dir"] = args.ocr_runtime_dir
    if args.ocr_model_dir:
        config["ocr_model_dir"] = args.ocr_model_dir
    print(read(args.candidates, args.manifest, args.run_dir, config))


if __name__ == "__main__":
    main()
