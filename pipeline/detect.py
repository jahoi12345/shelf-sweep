"""Mixed-content observations in native frame pixels; independent API/replay CLI."""
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
from pipeline import contracts as C

NON_BOOK_CATEGORIES = {"plant","pot","planter","vase","framed_art","framed_sign",
    "decorative_letters","shelving","frame","decoration","lamp","furniture","electronics"}

PROMPT = """Inventory visible physical objects in this shelf/room image.
Spanish and English books require equal detection support; never filter by text language.
One tight box per individual book spine: upright, leaning, horizontal, stacked,
partly occluded and unreadable. Each layer in a stack is a separate book. Do not
merge adjacent spines or guess books fully hidden. Text is NOT book evidence:
check binding/spine/page edges. Separately detect non-books: shelving, plants,
pots, vases, framed art, framed signs, decorative letters, furniture, lamps,
electronics and decor. Separate plant and pot when separately visible; do not
also emit a combined potted-plant duplicate. Shelving is a unit, not each plank.
Use ambiguous for uncertain objects. Preserve occlusion and shape evidence;
flag boxes covering inseparable multiple spines. Art is appraisal-sensitive.
Do not transcribe titles or identify works. box_2d is [ymin,xmin,ymax,xmax]
normalized 0..1000 for THIS image. Return {"objects": [{"box_2d": [...],
"kind": "book|non_book|ambiguous", "category": "book|plant|pot|vase|framed_art|framed_sign|decorative_letters|shelving|...",
"orientation": "upright|leaning|horizontal|stacked|unknown", "confidence": 0..1,
"occlusion": "none|partial|heavy|unknown", "evidence_notes": "visible shape evidence and uncertainty"}]}.
"""

RESPONSE_SCHEMA = {
    "type": "object", "properties": {"objects": {"type": "array", "items": {
        "type": "object", "properties": {
            "box_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            "kind": {"type": "string", "enum": ["book", "non_book", "ambiguous"]},
            "category": {"type": "string"}, "orientation": {"type": "string"},
            "confidence": {"type": "number"}, "occlusion": {"type": "string"},
            "evidence_notes": {"type": "string"}},
        "required": ["box_2d", "kind", "category", "orientation", "confidence", "occlusion", "evidence_notes"]}}},
    "required": ["objects"]}


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
    return path


def manifest_frames(manifest):
    data = manifest.get("data", manifest)
    return data["frames"] if isinstance(data, dict) else data


def resolve_evidence(run_dir, path):
    root = Path(run_dir).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Evidence path must stay in run directory: {path}")
    return result


def native_box(box, fmt, tile, width, height):
    """Convert provider coordinates, reject invalid geometry, allow rounding only."""
    if len(box) != 4 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in box):
        raise ValueError("Box must have four finite numeric coordinates")
    tx, ty, tw, th = tile
    if fmt == "yxyx_1000":
        if any(v < 0 or v > 1000 for v in box):
            raise ValueError("Normalized coordinates outside 0..1000")
        y1, x1, y2, x2 = box
        values = [tx + x1 * tw / 1000, ty + y1 * th / 1000,
                  tx + x2 * tw / 1000, ty + y2 * th / 1000]
    elif fmt == "xyxy_pixels":
        if not (0 <= box[0] < box[2] <= tw and 0 <= box[1] < box[3] <= th):
            raise ValueError("Pixel coordinates outside source image")
        values = [tx + box[0], ty + box[1], tx + box[2], ty + box[3]]
    else:
        raise ValueError(f"Unknown coordinate format: {fmt}")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Inverted/empty box")
    return [max(0, math.floor(x1)), max(0, math.floor(y1)),
            min(width, math.ceil(x2)), min(height, math.ceil(y2))]


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    return intersection / max(aa+ab-intersection, 1)


def tiles(width, height, size, overlap_fraction=.2):
    if not size or (width <= size and height <= size):
        return [(0, 0, width, height)]
    if size < 128 or not 0 <= overlap_fraction < .75:
        raise ValueError("Tile size >=128 and overlap in [0,.75) required")
    def starts(length):
        final = max(0, length-size)
        return sorted(set(list(range(0, final+1, max(1, int(size*(1-overlap_fraction))))) + [final]))
    return [(x,y,min(size,width-x),min(size,height-y)) for y in starts(height) for x in starts(width)]


def fuse_tiles(observations, iou_threshold=.88):
    """Suppress only very close same-kind/category duplicates from different tiles.

    Raw responses and aliases survive. Same-tile overlapping/merged spines and
    weaker tile duplicates are retained with review evidence.
    """
    result, possible = [], []
    for obs in sorted(observations, key=lambda o: (-(o.get("confidence") or 0), o["observation_id"])):
        duplicate = None
        for kept in result:
            if obs["kind"] != kept["kind"] or obs["category"] != kept["category"]:
                continue
            score = overlap(obs["bbox_xyxy"], kept["bbox_xyxy"])
            same_tile = obs["source_tile"] == kept["source_tile"]
            if not same_tile and score >= iou_threshold:
                duplicate = kept
                break
            if score > .55:
                possible.append({"observation_ids": [kept["observation_id"], obs["observation_id"]],
                    "reason": "Overlapping same-tile detections; possible merged/duplicate spines" if same_tile else "Uncertain tile duplicate; retained both",
                    "iou": score})
        if duplicate:
            duplicate.setdefault("tile_aliases", []).append({"observation_id": obs["observation_id"],
                "bbox_xyxy": obs["bbox_xyxy"], "response_ref": obs["response_ref"], "source_tile": obs["source_tile"],
                "reason": "Same category and native IoU >= tile threshold"})
        else:
            result.append(obs)
    return sorted(result, key=lambda o: o["observation_id"]), possible


async def _detect(manifest_path, run_dir, config):
    from providers.detector import structured_detection
    start = time.perf_counter()
    run_dir = Path(run_dir).resolve()
    manifest = read_json(manifest_path)
    problems = C.validate_manifest(manifest,run_dir)
    if problems:
        raise ValueError("Invalid evidence manifest: " + "; ".join(problems[:5]))
    replay = read_json(config["responses"]) if config.get("responses") else {}
    observations, issues, frame_results, uncertain_tiles, calls = [], [], [], [], []
    attempts = 0
    for frame in manifest_frames(manifest):
        if not frame.get("selected", True):
            frame_results.append({"frame_id": frame["frame_id"], "status": "not_selected"})
            continue
        fid = frame["frame_id"]
        if config.get("frame_ids") is not None and fid not in config["frame_ids"]:
            frame_results.append({"frame_id":fid,"status":"unprocessed","reason":"Outside explicitly requested detection scope"})
            issues.append({"frame_id":fid,"reason":"Selected frame outside caller scope; not an empty view"})
            continue
        try:
            path = resolve_evidence(run_dir, frame["image_path"])
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError("Frame cannot be opened")
            height, width = img.shape[:2]
            if (width,height) != (frame["width_px"],frame["height_px"]):
                raise ValueError("Manifest resolution differs from saved evidence")
            spec = replay.get(fid)
            if config.get("responses") and spec is None:
                frame_results.append({"frame_id": fid, "status": "unprocessed", "reason": "No saved response"})
                issues.append({"frame_id": fid, "reason": "Selected frame lacks cached predictions; not an empty shelf"})
                continue
            frame_obs, failed_tiles, object_offset = [], 0, 0
            regions = [(0,0,width,height)] if spec else tiles(width,height,config.get("tile_size",0),config.get("tile_overlap",.2))
            for ti, tile in enumerate(regions):
                tx,ty,tw,th = tile
                request_id = f"{fid}_tile{ti:03d}"
                try:
                    if spec:
                        raw_path = resolve_evidence(run_dir, spec["response_path"])
                        raw = read_json(raw_path)
                        response_ref = raw_path.relative_to(run_dir).as_posix()
                        coordinate_format = spec.get("coordinate_format","yxyx_1000")
                        calls.append({"provider": spec.get("provider", "saved_response"), "model": spec.get("model"),
                            "request_count": 0, "replay_hits": 1, "cost": 0,
                            "cost_provenance": "No new API call; historical billed cost unknown"})
                    else:
                        if config.get("no_provider"):
                            raise RuntimeError("Provider disabled by caller")
                        tile_path = path
                        if len(regions) > 1:
                            tile_path = run_dir / "detection_tiles" / config.get("response_namespace","") / f"{request_id}.png"
                            tile_path.parent.mkdir(parents=True, exist_ok=True)
                            if not cv2.imwrite(str(tile_path),img[ty:ty+th,tx:tx+tw]):
                                raise ValueError("Cannot save native tile")
                        attempts += 1
                        result = config.get("image_request",config.get("adapter",structured_detection))(image_path=tile_path, prompt=PROMPT,
                            response_schema=RESPONSE_SCHEMA, model=config.get("model"), run_dir=run_dir, stage="detect", request_id=request_id)
                        if inspect.isawaitable(result):
                            result = await result
                        raw = result["data"]
                        coordinate_format = result.get("coordinate_format","yxyx_1000")
                        calls.append(result.get("metrics",{}))
                        issues.extend(result.get("issues",[]))
                        raw_path = save_json(run_dir / "detection_responses" / config.get("response_namespace","") / f"{request_id}.json",raw)
                        response_ref = raw_path.relative_to(run_dir).as_posix()
                    objects = raw if isinstance(raw,list) else raw["objects"]
                    if not isinstance(objects,list):
                        raise ValueError("Provider objects must be a list")
                    legacy = bool(spec and spec.get("legacy_book_only"))
                    for oi,obj in enumerate(objects):
                        oid = C.observation_id(fid,object_offset+oi)
                        try:
                            kind = "book" if legacy else obj["kind"]
                            if kind not in {"book","non_book","ambiguous"}:
                                raise ValueError("Invalid kind")
                            fmt = coordinate_format
                            box = native_box(obj.get("bbox_xyxy",obj.get("box_2d",[])),fmt,tile,width,height)
                            conf = None if legacy else obj.get("confidence")
                            if conf is not None and (isinstance(conf,bool) or not isinstance(conf,(int,float)) or not math.isfinite(conf) or not 0 <= conf <= 1):
                                raise ValueError("Invalid confidence")
                            notes = obj.get("evidence_notes", "")
                            object_issues = []
                            if not legacy and kind == "book" and obj.get("category","").lower() in NON_BOOK_CATEGORIES:
                                kind = "ambiguous"
                                object_issues.append("Provider kind/category conflict; text-bearing decor is not accepted as a book")
                            if legacy:
                                notes = "Historical pretrained book-only spike; detection confidence/occlusion unavailable; not human truth"
                            obs = {"observation_id": oid,"frame_id":fid,"bbox_xyxy":box,"kind":kind,
                                "category":"book" if kind == "book" else obj.get("category","unknown"),
                                "orientation":"flat" if obj.get("orientation") == "horizontal" else obj.get("orientation","unknown"),"confidence":conf,
                                "shelf_hint":obj.get("shelf_hint"),"occlusion":obj.get("occlusion","unknown"),
                                "evidence_notes":notes,"frame_ref":frame["image_path"],"source_tile":list(tile),
                                "response_ref":response_ref,"issues":["Historical book-only detector"] if legacy else object_issues}
                            if obj.get("polygon_xy") is not None:
                                polygon = obj["polygon_xy"]
                                if fmt != "xyxy_pixels" or len(polygon) < 3:
                                    raise ValueError("polygon_xy requires >=3 source-pixel points")
                                if any(len(p) != 2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in p)
                                    or not 0 <= p[0] <= tw or not 0 <= p[1] <= th for p in polygon):
                                    raise ValueError("Invalid polygon")
                                obs["polygon_xy"] = [[p[0]+tx,p[1]+ty] for p in polygon]
                            for field in ['detection_provenance','detection_aliases']:
                                if field in obj:
                                    obs[field]=obj[field]
                            frame_obs.append(obs)
                        except (ValueError,KeyError,TypeError) as e:
                            issues.append({"ref_id":oid,"frame_id":fid,"reason":f"Rejected invalid object: {e}","response_ref":response_ref})
                    if legacy:
                        issues.append({"frame_id":fid,"reason":"Historical response omitted non-books; non-book inventory incomplete"})
                    object_offset += len(objects)
                except Exception as e:
                    # Provider exceptions may embed credentials: save the type only.
                    failed_tiles += 1
                    reason = "Provider explicitly disabled; detection not run" if config.get("no_provider") else f"Detection unavailable ({type(e).__name__}); no empty-scene inference"
                    issues.append({"frame_id":fid,"tile":list(tile),"reason":reason})
            fused,uncertain = fuse_tiles(frame_obs,config.get("tile_merge_iou",.88))
            observations.extend(fused)
            uncertain_tiles.extend(uncertain)
            frame_results.append({"frame_id":fid,"status":"failed" if failed_tiles == len(regions) else "partial" if failed_tiles else "processed",
                "tile_count":len(regions),"failed_tiles":failed_tiles,"observation_count":len(fused)})
        except (ValueError,KeyError) as e:
            issues.append({"frame_id":fid,"reason":str(e)})
            frame_results.append({"frame_id":fid,"status":"failed"})
    unprocessed = sum(f["status"] in {"failed","unprocessed","partial"} for f in frame_results)
    artifact = {"schema_version":1,"run_id":manifest.get("run_id",run_dir.name),"stage":"detect",
        "status":"partial" if issues or unprocessed else "complete",
        "data":{"observations":observations,"frames":frame_results,"uncertain_tile_associations":uncertain_tiles},
        "issues":issues,"metrics":{"elapsed_s":round(time.perf_counter()-start,4),
            "observation_count":len(observations),"by_kind":{k:sum(o["kind"] == k for o in observations) for k in ["book","non_book","ambiguous"]},
            "requests":calls,"request_count":sum(c.get("request_count",0) for c in calls),
            "local_inference_count":sum(c.get("local_inference_count",0) for c in calls),
            "provider_attempts":attempts,
            "replay_hits":sum(c.get("replay_hits",0) for c in calls),"billed_cost":None,
            "cost_provenance":"Per-request provenance in requests; unavailable billed costs remain null",
            "input_sha256":hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}}
    if manifest.get("fixture") or any(c.get("fixture") for c in calls) or any(s.get("fixture") for s in replay.values()):
        artifact["fixture"] = True
    return save_json(run_dir / config.get("output_name","candidates.json"),artifact)


def detect(manifest_path,run_dir,config=None):
    """Sync entrypoint. Async orchestration may await detect_async instead."""
    config = config or {}
    return asyncio.run(_detect(manifest_path,run_dir,{**config,**config.get("detect",{})}))


detect_async = _detect


async def detect_frame_async(manifest_path,run_dir,frame_id,*,revision,current_revision,config=None):
    """One exact frozen-frame result for the session/revision task system.

    Caller owns inventory mutation, voice and cancellation. A stale revision can
    save diagnostic evidence but never returns a publishable observation update.
    Snapshot and raw response paths are revision-scoped, so concurrent frames do
    not replace each other's stage artifacts. Cancellation propagates normally.
    """
    if current_revision() != revision:
        return {"status":"discarded","reason":"stale_revision_before_inference","revision":revision}
    root = Path(run_dir).resolve()
    manifest = read_json(manifest_path)
    frames = [dict(f) for f in manifest_frames(manifest) if f["frame_id"] == frame_id]
    if len(frames) != 1:
        raise ValueError("Exactly one known frame_id required")
    frames[0]["selected"] = True
    frames[0]["selection_reason"] = "Caller selected exact frozen live frame"
    # Validate ID before it becomes a filename.
    if C.validate_manifest({**manifest,"data":{"frames":frames}},root):
        raise ValueError("Invalid incremental frame evidence")
    mini = {**manifest,"data":{**manifest["data"],"frames":frames}}
    namespace = f"live/detect/{frame_id}.r{int(revision):06d}"
    mini_path = save_json(root/(namespace+".manifest.json"),mini)
    cfg = config or {}
    cfg = {**cfg,**cfg.get("detect",{}),"output_name":namespace+".candidates.json",
           "response_namespace":namespace,"frame_ids":[frame_id]}
    path = await _detect(mini_path,root,cfg)
    if current_revision() != revision:
        return {"status":"discarded","reason":"stale_revision_after_inference","revision":revision,
                "artifact_path":path.relative_to(root).as_posix()}
    artifact = read_json(path)
    return {"status":"ready","revision":revision,"frame_id":frame_id,
            "artifact_path":path.relative_to(root).as_posix(),"observations":artifact["data"]["observations"],
            "issues":artifact["issues"],"metrics":artifact["metrics"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--out",required=True,help="Run directory containing evidence")
    parser.add_argument("--responses",help="Frame ID -> saved response specs for offline replay")
    parser.add_argument("--model")
    parser.add_argument("--tile-size",type=int,default=0,help="0: full frame; otherwise native tiles")
    parser.add_argument("--tile-overlap",type=float,default=.2)
    parser.add_argument("--allow-hosted",action="store_true",help="Permit billable hosted detection calls (default: refuse; use providers.local_detector for local detection)")
    args = parser.parse_args()
    if not args.responses and not args.allow_hosted:
        parser.error("this CLI calls the hosted detector (billable) unless --responses is given; pass --allow-hosted to confirm, or use `python -m providers.local_detector` for local detection")
    print(detect(args.manifest,args.out,vars(args)))


if __name__ == "__main__":
    main()
