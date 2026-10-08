"""Associate physical objects using registered frame geometry and crop appearance.

Titles NEVER create links. Same-frame observations cannot share an instance.
Uncertain links remain separate and retain their evidence for review. ORB/RANSAC
is a geometry component, not a trained book detector; detection is a separate stage.
"""
from __future__ import annotations

import argparse
import itertools
import math
import time
from collections import Counter, OrderedDict
from pathlib import Path

import os

import cv2
import numpy as np

from pipeline import contracts as C
from pipeline.detect import read_json, save_json, manifest_frames, resolve_evidence, overlap
from pipeline.items import ITEM_CONTEXT_ONLY, ITEM_DEDUP_DEFAULTS, ITEM_TAXONOMY, consolidate_items_v2, detection_attributes, non_book_demotion, spine_row_support


DEFAULTS = {"min_inliers":18,"min_inlier_ratio":.45,"max_reprojection_px":3.0,
    "min_spatial_coverage":.06,"min_iou":.55,"max_center_distance":.55,
    "min_appearance":.72,"ambiguity_margin":.07,"min_pair_score":.70,
    "max_gap_s":30.0,"registration_max_size":1600,"feature_method":"orb",
    "bridge_registration":False,"max_bridge_gap_s":3.5,"max_bridge_steps":8,
    "max_bridge_step_s":.65,"association_geometry":"bbox","appearance_mode":"direct","association":"track",
    # Non-book robustness (pipeline/items.py): book groups never seen in a shelf-like row are demoted to ambiguous, not deleted.
    # None = LIBRARY_NON_BOOK_GATE decides (default on; tests/conftest.py turns it off for legacy synthetic scenes of lone boxes).
    "non_book_gate":None,"scene_gate_min_row":3}


def registration(image_a,image_b,config):
    """A -> B planar homography from reciprocal distinctive visual landmarks."""
    if image_a.shape == image_b.shape and np.array_equal(image_a,image_b):
        return {"status":"accepted","method":"identical_saved_pixels","homography":np.eye(3).tolist(),
            "inliers":None,"inlier_ratio":1.0,"median_reprojection_px":0.0,"spatial_coverage":1.0}
    def features(image):
        h,w = image.shape[:2]
        scale = min(1.0,config["registration_max_size"]/max(w,h))
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if scale < 1:
            gray = cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        method = config.get("feature_method","orb")
        if method == "orb":
            detector = cv2.ORB_create(nfeatures=5000,fastThreshold=12)
        elif method == "sift":
            detector = cv2.SIFT_create(nfeatures=5000,contrastThreshold=.03)
        else:
            raise ValueError("feature_method must be orb or sift")
        kp,desc = detector.detectAndCompute(gray,None)
        return kp,desc,scale
    ka,da,sa = features(image_a)
    kb,db,sb = features(image_b)
    method = config.get("feature_method","orb")
    method_name = method.upper()+"_reciprocal_RANSAC"
    failure = {"status":"unavailable","method":method_name,"homography":None}
    if da is None or db is None or min(len(da),len(db)) < 2:
        return {**failure,"reason":"Insufficient distinctive visual features"}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2)
    def distinctive(query,train):
        return {pair[0].queryIdx:pair[0] for pair in matcher.knnMatch(query,train,k=2)
            if len(pair) == 2 and pair[0].distance < .7*pair[1].distance}
    ab,ba = distinctive(da,db),distinctive(db,da)
    matches = [m for i,m in ab.items() if m.trainIdx in ba and ba[m.trainIdx].trainIdx == i]
    if len(matches) < config["min_inliers"]:
        return {**failure,"reason":"Insufficient reciprocal distinctive matches","matches":len(matches)}
    src = np.float32([ka[m.queryIdx].pt for m in matches])/sa
    dst = np.float32([kb[m.trainIdx].pt for m in matches])/sb
    H,mask = cv2.findHomography(src,dst,cv2.RANSAC,config["max_reprojection_px"])
    if H is None or mask is None or not np.isfinite(H).all():
        return {**failure,"reason":"No valid projective fit"}
    inlier = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(src.reshape(-1,1,2),H).reshape(-1,2)
    error = float(np.median(np.linalg.norm(projected[inlier]-dst[inlier],axis=1)))
    coverage = min(cv2.contourArea(cv2.convexHull(points[inlier]))/(img.shape[0]*img.shape[1])
        for points,img in [(src,image_a),(dst,image_b)])
    ratio = float(inlier.mean())
    stats = {"method":method_name,"matches":len(matches),"inliers":int(inlier.sum()),
        "inlier_ratio":ratio,"median_reprojection_px":error,"spatial_coverage":float(coverage)}
    if inlier.sum() < config["min_inliers"] or ratio < config["min_inlier_ratio"] or error > config["max_reprojection_px"] or coverage < config["min_spatial_coverage"]:
        return {**failure,**stats,"reason":"Registration quality gate failed; keep instances separate"}
    # Reject flips, singularities and explosive extrapolation (parallax stays a review risk).
    h,w = image_a.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners,H).reshape(-1,2)
    denominators = corners.reshape(-1,2)@H[2,:2]+H[2,2]
    area_ratio = cv2.contourArea(warped,oriented=True)/(w*h)
    if not np.isfinite(warped).all() or np.min(denominators)*np.max(denominators) <= 0 or not .15 < area_ratio < 6:
        return {**failure,**stats,"reason":"Unstable/flipped projective mapping"}
    return {"status":"accepted",**stats,"homography":H.tolist()}


def projected_box(box,H):
    x1,y1,x2,y2 = box
    points = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(points,H).reshape(-1,2)
    return [float(warped[:,0].min()),float(warped[:,1].min()),
            float(warped[:,0].max()),float(warped[:,1].max())]


def bridge_registration(fa,fb,frames,run_dir,images,config,pair_cache):
    """Compose trusted short-overlap transforms through unchanged saved evidence.

    Intermediate frames support geometry only; they do not add counted detections.
    Every edge uses the unchanged direct registration gates. This avoids relaxing
    thresholds for blurry endpoint views or repeated identical book bindings.
    """
    ta,tb=frames[fa]["timestamp_s"],frames[fb]["timestamp_s"]
    if tb-ta > config["max_bridge_gap_s"]:
        return None
    pool=sorted([f for f in frames.values() if ta<=f["timestamp_s"]<=tb],key=lambda f:f["timestamp_s"])
    pool=[f for f in pool if f["frame_id"] in {fa,fb} or not f.get("quality",{}).get("reasons")
          or "blurry" not in f.get("quality",{}).get("reasons",[])]
    def image(fid):
        if fid not in images:
            img=cv2.imread(str(resolve_evidence(run_dir,frames[fid]["image_path"])))
            if img is not None and img.shape[1::-1] == (frames[fid]["width_px"],frames[fid]["height_px"]):
                images[fid]=img
        return images.get(fid)
    # A temporal DAG guarantees no revisits/cycles. Prefer fewer accurate steps.
    paths={fa:(0.0,np.eye(3),[])}
    for b in pool[1:]:
        fid=b["frame_id"]
        choices=[]
        for a in pool:
            aid=a["frame_id"]
            if aid not in paths or not 0<b["timestamp_s"]-a["timestamp_s"]<=config["max_bridge_step_s"]:
                continue
            cost,H,steps=paths[aid]
            if len(steps)>=config["max_bridge_steps"]:
                continue
            if image(aid) is None or image(fid) is None:
                continue
            key=(aid,fid)
            if key not in pair_cache:
                pair_cache[key]=registration(images[aid],images[fid],config)
            reg=pair_cache[key]
            if reg["status"] != "accepted":
                continue
            step={"frame_ids":[aid,fid],**reg,"evidence_refs":[frames[aid]["image_path"],frames[fid]["image_path"]]}
            choices.append((cost+1+reg["median_reprojection_px"]/3,np.array(reg["homography"])@H,steps+[step]))
        if choices:
            paths[fid]=min(choices,key=lambda x:x[0])
    if fb not in paths:
        return None
    _,H,steps=paths[fb]
    if len(steps)<2 or not np.isfinite(H).all() or abs(H[2,2])<1e-12:
        return None
    H=H/H[2,2]
    h,w=images[fa].shape[:2]
    corners=np.float32([[0,0],[w,0],[w,h],[0,h]])
    denominator=corners@H[2,:2]+H[2,2]
    warped=cv2.perspectiveTransform(corners.reshape(-1,1,2),H).reshape(-1,2)
    area=cv2.contourArea(warped,oriented=True)/(w*h)
    if np.min(denominator)*np.max(denominator)<=0 or not .15<area<6:
        return None
    return {"status":"accepted","method":"trusted_intermediate_frame_chain","homography":H.tolist(),
        "bridge_steps":steps,"bridge_steps_count":len(steps),"inliers":min(s["inliers"] or 0 for s in steps),
        "inlier_ratio":min(s["inlier_ratio"] for s in steps),
        "median_reprojection_px":max(s["median_reprojection_px"] for s in steps),
        "spatial_coverage":min(s["spatial_coverage"] for s in steps),
        "assumption":"Common static shelf plane; local crop appearance gate must still pass; composed drift not human-validated"}


def appearance_score(a,b):
    """Comparable registered crops: texture correlation plus color histogram."""
    if not a.size or not b.size:
        return 0.0
    a = cv2.resize(a,(32,96),interpolation=cv2.INTER_AREA)
    b = cv2.resize(b,(32,96),interpolation=cv2.INTER_AREA)
    ga = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(float).ravel()
    gb = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(float).ravel()
    if min(ga.std(),gb.std()) < 3:
        texture = max(0.0,1.0-float(np.mean(np.abs(ga-gb)))/64)
    else:
        texture = max(0.0,float(np.corrcoef(ga,gb)[0,1]))
    def hist(image):
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256])
        return cv2.normalize(h,h,alpha=1,norm_type=cv2.NORM_L1)
    color = 1.0-float(cv2.compareHist(hist(a),hist(b),cv2.HISTCMP_BHATTACHARYYA))
    return .65*texture+.35*max(0.0,color)


def polygon_iou(a,b):
    """Convex source-pixel shapes; axis-aligned bounds can overlap adjacent spines."""
    a,b=np.asarray(a,np.float32),np.asarray(b,np.float32)
    intersection,_=cv2.intersectConvexConvex(a,b)
    union=cv2.contourArea(a)+cv2.contourArea(b)-intersection
    return float(intersection/union) if union>0 else 0.0


def spine_patch(image,polygon):
    """Rectify one observed OBB, excluding neighbors inside its loose bbox."""
    points=np.asarray(polygon,np.float32)
    if points.shape!=(4,2):
        raise ValueError('Four spine corners required')
    lengths=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
    if lengths[0]>lengths[1]:
        points=np.roll(points,-1,axis=0)
    H=cv2.getPerspectiveTransform(points,np.float32([[0,0],[31,0],[31,95],[0,95]]))
    return cv2.warpPerspective(image,H,(32,96))


def aligned_spine_appearance(warped,image,polygon):
    """Bounded local registration residual, without shifting to another spine.

    RANSAC allows <=3px median reprojection error; direct patch correlation was
    treating a two-pixel residual as different identity. Keep global polygon/
    center gates unchanged and compare only within +/-2 native source pixels.
    """
    target=spine_patch(image,polygon)
    points=np.asarray(polygon,np.float32)
    choices=[]
    for dx in [-2,0,2]:
        for dy in [-2,0,2]:
            choices.append((appearance_score(spine_patch(warped,points+[dx,dy]),target),dx,dy))
    best=max(choices,key=lambda x:(x[0],-abs(x[1])-abs(x[2])))
    return best[0],[best[1],best[2]]


def pair_evidence(a,b,H,warped_image,image_b,config):
    if a["kind"] != b["kind"] or (a["kind"] != "book" and a.get("category") != b.get("category")):
        return None
    projected = projected_box(a["bbox_xyxy"],H)
    box = b["bbox_xyxy"]
    bbox_iou = overlap(projected,box)
    if bbox_iou < .20:
        return None
    pw,ph = projected[2]-projected[0],projected[3]-projected[1]
    bw,bh = box[2]-box[0],box[3]-box[1]
    if min(pw,ph,bw,bh) <= 0 or not .5 <= (pw*ph)/(bw*bh) <= 2:
        return None
    distance = math.hypot((projected[0]+projected[2]-box[0]-box[2])/2,
        (projected[1]+projected[3]-box[1]-box[3])/2)/max(1,min(pw,ph,bw,bh))
    if distance > 1.5:
        return None
    x1,y1,x2,y2 = map(int,box)
    geometry='bbox'
    iou=bbox_iou
    projected_polygon=None
    alignment_offset=[0,0]
    if config.get('association_geometry')=='polygon' and len(a.get('polygon_xy',[]))==4 and len(b.get('polygon_xy',[]))==4:
        projected_polygon=cv2.perspectiveTransform(np.asarray(a['polygon_xy'],np.float32).reshape(-1,1,2),H).reshape(-1,2).tolist()
        iou=polygon_iou(projected_polygon,b['polygon_xy'])
        app=appearance_score(spine_patch(warped_image,b['polygon_xy']),spine_patch(image_b,b['polygon_xy']))
        direct_app=app
        if config.get('appearance_mode')=='aligned_polygon':
            app,alignment_offset=aligned_spine_appearance(warped_image,image_b,b['polygon_xy'])
        geometry='polygon'
    else:
        app = appearance_score(warped_image[y1:y2,x1:x2],image_b[y1:y2,x1:x2])
        direct_app=app
    score = .55*iou+.35*app+.10*max(0,1-distance)
    accepted = iou >= config["min_iou"] and distance <= config["max_center_distance"] and app >= config["min_appearance"] and score >= config["min_pair_score"]
    return {"observation_ids":[a["observation_id"],b["observation_id"]],"frame_ids":[a["frame_id"],b["frame_id"]],
        "projected_bbox_xyxy":projected,"iou":iou,"center_distance_spine_widths":distance,
        "appearance_score":app,"score":score,"eligible":bool(accepted),
        "geometry":geometry,"bbox_iou":bbox_iou,"projected_polygon_xy":projected_polygon,
        "direct_appearance_score":direct_app,"alignment_offset_native_px":alignment_offset,
        "reason":"Registered spatial overlap + local texture/color; title not used" if accepted else "Plausible overlap fails strict spatial/appearance gate"}


def associate(observations,pair_candidates,margin):
    """Globally one-to-one unions with same-frame exclusion and competing-edge veto."""
    owner = {o["observation_id"]:o["observation_id"] for o in observations}
    groups = {o["observation_id"]:{o["observation_id"]} for o in observations}
    frames = {o["observation_id"]:{o["frame_id"]} for o in observations}
    def root(oid):
        while owner[oid] != oid:
            owner[oid] = owner[owner[oid]]
            oid = owner[oid]
        return oid
    accepted,uncertain = [],[]
    by_pair = {}
    for edge in pair_candidates:
        key = tuple(edge["frame_ids"])
        by_pair.setdefault(key,[]).append(edge)
    for edge in sorted(pair_candidates,key=lambda e:(-e["score"],e["observation_ids"])):
        a,b = edge["observation_ids"]
        ra,rb = root(a),root(b)
        if ra == rb:
            continue
        reason = None
        if not edge["eligible"]:
            reason = edge["reason"]
        else:
            competing = [e["score"] for e in by_pair[tuple(edge["frame_ids"])]
                if e is not edge and set(e["observation_ids"]) & {a,b}]
            if competing and edge["score"]-max(competing) < margin:
                reason = "Near-tie competing physical instance; identical copies/adjacent spines stay separate"
            elif frames[ra] & frames[rb]:
                reason = "Same-frame exclusion blocks transitive merge of separate objects"
        if reason:
            uncertain.append({**edge,"reason":reason,"decision":"uncertain"})
        else:
            keep,drop = sorted([ra,rb])
            owner[drop] = keep
            groups[keep] |= groups.pop(drop)
            frames[keep] |= frames.pop(drop)
            accepted.append({**edge,"decision":"merged"})
    return sorted([sorted(v) for v in groups.values()]),accepted,uncertain


def _stage_records(path,key):
    if not path:
        return {}
    art = read_json(path)
    return {r["observation_id"]:r for r in art.get("data",{}).get(key,[])}


def _identity(obs,readings,resolutions):
    literal = [readings[o["observation_id"]] for o in obs if o["observation_id"] in readings]
    best = max(literal,key=lambda r:r.get("read_confidence") or 0,default={})
    identity = {k:None for k in ["work","title","author","publisher","edition","isbn","source_url","id_confidence"]}
    identity.update(decision="no_match",visible_text=best.get("visible_text"))
    accepted = [resolutions[o["observation_id"]] for o in obs if o["observation_id"] in resolutions
        and resolutions[o["observation_id"]].get("decision") == "accepted"
        and resolutions[o["observation_id"]].get("matching_evidence") and resolutions[o["observation_id"]].get("source_url")]
    if not accepted:
        return identity,["Unreadable/unresolved spine; preserved as unidentified book"]
    # Google Books is the SECOND catalogue: a group with any Open Library acceptance uses those only (the two catalogues file one book under different works,
    # e.g. "The Story of Civilization I" vs "Our Oriental Heritage", which would otherwise read as a conflict and abstain)
    primary = [r for r in accepted if (r.get("catalogue") or "open_library") == "open_library"]
    accepted = primary or accepted
    keys = {(str(r.get("work") or ""),str(r.get("title") or "").casefold()) for r in accepted}
    if len(keys) > 1:
        identity["decision"] = "ambiguous"
        return identity,["Conflicting catalogue work evidence across observations; identity abstained"]
    chosen = max(accepted,key=lambda r:readings.get(r["observation_id"],{}).get("read_confidence") or 0)
    for key in identity:
        if key in chosen:
            identity[key] = chosen[key]
    identity["decision"] = "accepted"
    identity["id_confidence"] = readings.get(chosen["observation_id"],{}).get("read_confidence")
    identity["resolution_observation_id"] = chosen["observation_id"]
    identity["matching_evidence"] = chosen["matching_evidence"]
    identity["catalogue"] = chosen.get("catalogue") or "open_library"          # second catalogue (Google Books) when Open Library had no eligible candidate
    identity["catalogue_title"] = chosen.get("catalogue_title")
    identity["catalogue_dimensions"] = (chosen.get("google_books") or {}).get("dimensions")
    # Edition/ISBN are never extended from a different catalogue candidate.
    return identity,[]


def second_bands(frames):
    """Capture-quality band per whole second (strict/medium/poor/bad) and the count reliability, recomputed from the saved
    frame quality exactly as the live monitor does (app/capture_quality.py)."""
    from app.capture_quality import CaptureQuality
    cq = CaptureQuality()
    for f in sorted(frames.values(),key=lambda f:f["timestamp_s"]):
        q = f.get("quality") or {}
        if q.get("sharpness") is not None:
            cq.add(f["timestamp_s"],q.get("motion_px_per_s"),q["sharpness"],q.get("zoom_rate_per_s"))
    bands = {sec:cq.band(cq._med([x[0] for x in v]),cq._med([x[1] for x in v]) or 0.0,cq._med([x[2] for x in v])) for sec,v in cq.seconds.items()}
    return bands,cq.summary()["count_reliability"]


COUNT_RANGE = {"linked_frame_min":0.3}


def count_range(groups,observations,frames,held_frames=(),singleton_merge=None,config=None):
    """Best guess + range for poor capture. Every book group gets a tier:
      confident   >= 2 sightings, or 1 sighting on a STRICT capture second in a LINKED frame
      likely      1 sighting on a MEDIUM second in a linked frame (or a frame without quality data)
      unconfirmed 1 sighting on a poor/bad second, or in a frame whose motion was not tracked (held), or in an unlinked frame
    (with under 5 s of capture-quality data, reliability "unknown", every singleton is "likely").
    A frame is linked when >= linked_frame_min of its book sightings belong to multi-sighting groups: a frame where nothing links (a blurred
    re-view after the track chain broke) mostly re-shows books already counted elsewhere.
    lower = confident; best guess = confident + likely; upper = every group after the unconfirmed singletons that land on another group by tracked
    position have been merged (singleton_merge(groups, eligible_ids) -> merged groups). Range covers association only, never missed/merged spines."""
    cfg = {**COUNT_RANGE,**(config or {})}
    by_id = {o["observation_id"]:o for o in observations}
    book = [g for g in groups if by_id[g[0]]["kind"] == "book"]
    bands,reliability = second_bands(frames)
    tot,linked = Counter(),Counter()
    for g in book:
        for oid in g:
            tot[by_id[oid]["frame_id"]] += 1
            linked[by_id[oid]["frame_id"]] += len(g) > 1
    held = set(held_frames)
    tiers = {}
    for g in book:
        if len(g) > 1:
            tiers[tuple(g)] = "confident"
            continue
        fid = by_id[g[0]]["frame_id"]
        band = bands.get(int(frames[fid]["timestamp_s"])) if fid in frames else None
        ok = fid not in held and linked[fid] >= cfg["linked_frame_min"]*tot[fid]
        if reliability == "unknown":  # under 5 s of quality data the capture cannot be judged: no singleton is demoted
            tiers[tuple(g)] = "likely"
            continue
        tiers[tuple(g)] = ("confident" if ok and band == "strict" else "likely" if ok and band in ("medium",None) else "unconfirmed")
    n = Counter(tiers.values())
    unconfirmed = {g[0] for g in book if tiers[tuple(g)] == "unconfirmed"}
    merged = len(book)
    if singleton_merge and unconfirmed:
        merged = sum(1 for g in singleton_merge(book,unconfirmed) if by_id[g[0]]["kind"] == "book")
    lower,best = n["confident"],n["confident"]+n["likely"]
    return tiers,{"count_best":best,"count_range":[lower,max(best,merged)],"confident_groups":lower,"likely_groups":n["likely"],
        "unconfirmed_groups":n["unconfirmed"],"singleton_groups":sum(len(g) == 1 for g in book),"all_groups":len(book),
        "count_reliability":reliability,"singleton_merges_for_upper":len(book)-merged,
        "count_range_basis":("best guess = confident + likely groups; lower = confident groups (>=2 sightings, or 1 sighting on a strict capture second in a "
            "tracked frame where >= 30% of sightings link); upper = all groups after unconfirmed single sightings that land on another group by tracked "
            "position are merged. Covers association only: spines the detector missed or fused are outside the range.")}


_ITEM_REGISTRATIONS = OrderedDict()   # (run root, frame a, frame b) -> homography or None; live merges re-ask the same pairs


def _item_registration(key,fa,fb,load_image,config):
    """Direct feature registration of two frames for item dedup across a broken tracked chain (pipeline/items.consolidate_items_v2)."""
    k = (key,fa,fb)
    if k not in _ITEM_REGISTRATIONS:
        a,b = load_image(fa),load_image(fb)
        reg = registration(a,b,config) if a is not None and b is not None else {"status":"unavailable"}
        _ITEM_REGISTRATIONS[k] = np.array(reg["homography"]) if reg["status"] == "accepted" else None
        while len(_ITEM_REGISTRATIONS) > 5000:
            _ITEM_REGISTRATIONS.popitem(last=False)
    return _ITEM_REGISTRATIONS[k]


def merge(candidates_path,manifest_path,run_dir,config=None,readings_path=None,resolutions_path=None):
    start = time.perf_counter()
    config = config or {}
    config = {**DEFAULTS,**config,**config.get("dedup",{})}
    root = Path(run_dir).resolve()
    candidates,manifest = read_json(candidates_path),read_json(manifest_path)
    observations = candidates.get("data",{}).get("observations",candidates.get("data",{}).get("detections",[]))
    errors = C.validate_candidates(candidates,manifest)
    if errors:
        raise ValueError("Invalid candidate contract: " + "; ".join(errors[:5]))
    by_id = {o["observation_id"]:o for o in observations}
    frames = {f["frame_id"]:f for f in manifest_frames(manifest)}
    by_frame = {fid:[o for o in observations if o["frame_id"] == fid] for fid in frames}
    def load_image(fid):
        image = cv2.imread(str(resolve_evidence(root,frames[fid]["image_path"])))
        if image is None or image.shape[1::-1] != (frames[fid]["width_px"],frames[fid]["height_px"]):
            return None
        return image
    class BoundedImages(OrderedDict):
        """Opt-in native cache; evicted bridge endpoints can be reloaded unchanged."""
        def __missing__(self,fid):
            image = load_image(fid)
            if image is None:
                raise ValueError("Previously validated association frame became unreadable: "+fid)
            self[fid] = image
            return image
        def __getitem__(self,fid):
            result = super().__getitem__(fid)
            self.move_to_end(fid)
            return result
        def __setitem__(self,fid,image):
            super().__setitem__(fid,image)
            self.move_to_end(fid)
            while len(self)>cache_limit:
                self.popitem(last=False)
    cache_limit = config.get("max_image_cache")
    if cache_limit is not None and (not isinstance(cache_limit,int) or isinstance(cache_limit,bool) or cache_limit<2):
        raise ValueError("max_image_cache must be an integer >=2")
    images = BoundedImages() if cache_limit else {}
    issues = list(candidates.get("issues",[]))
    valid_frames = []
    for fid,obs in by_frame.items():
        if not obs:
            continue
        image = load_image(fid)
        if image is None:
            issues.append({"frame_id":fid,"reason":"Association image missing or resolution mismatch; retain separate instances"})
        else:
            images[fid] = image
            valid_frames.append(fid)
    cv2.setRNGSeed(0)
    registrations,edges,pair_cache = [],[],{}
    ordered = sorted(valid_frames,key=lambda fid:(frames[fid]["timestamp_s"],fid))
    track_report = None
    track_transforms = None
    hold_frames = set()
    single_merge = None
    if config.get("association") in ("track","track_v2"):
        # Tracked-motion chain + residual alignment + appearance verification (pipeline/cross_view.py).
        # Replaces per-pair feature registration, which fails on repeated spine texture.
        # track_v2 adds motion-compensated IoU tracks (pipeline/tracker.py) on top of the same cross_view edges.
        from pipeline import cross_view
        if config.get("association") == "track_v2":
            from pipeline import tracker
            edges,track_report,track_pairs = tracker.track_v2_edges(observations,frames,load_image,{**(config.get("tracker") or {}),"cross_view":config.get("track")})
        else:
            edges,track_report,track_pairs = cross_view.track_edges(observations,frames,load_image,config.get("track"))
        track_transforms = track_report.pop("_transforms",None)
        track_report["_segments"] = track_report.pop("_segment_of",None)
        hold_frames = set(track_report.get("hold_frames") or [])
        for fid in sorted(hold_frames):
            issues.append({"frame_id":fid,"reason":"Frame motion could not be tracked; its sightings are placed provisionally and stay uncertain"})
        seen = set()
        for tp in track_pairs:
            registrations.append({"frame_ids":tp["frame_ids"],"status":"accepted","method":"tracked_chain_residual",
                "homography":None,"matched_pairs":tp["matched"],"residual_votes":tp["votes"],
                **({"guided_residual":tp["guided_residual"]} if tp.get("guided_residual") else {})})
            seen.update(tp["frame_ids"])
        for fid in ordered:
            if by_frame[fid] and fid not in seen:
                registrations.append({"frame_ids":[fid,fid],"status":"unavailable","method":"tracked_chain_residual","homography":None,
                    "reason":"No overlapping neighbouring detection frame within the time gap; duplicates with other views possible"})
                issues.append({"frame_ids":[fid],"reason":"No tracked overlap with a neighbouring view; repeated views may remain duplicate instances"})
        ordered = []
    for fa,fb in itertools.combinations(ordered,2):
        if abs(frames[fb]["timestamp_s"]-frames[fa]["timestamp_s"]) > config["max_gap_s"]:
            registrations.append({"frame_ids":[fa,fb],"status":"not_attempted","homography":None,
                "reason":"Pair exceeds configured time gap; revisits may remain duplicated"})
            continue
        direct=registration(images[fa],images[fb],config)
        pair_cache[(fa,fb)]=direct
        reg = {"frame_ids":[fa,fb],**direct}
        if reg["status"] != "accepted" and config["bridge_registration"]:
            bridge=bridge_registration(fa,fb,frames,root,images,config,pair_cache)
            if bridge:
                reg={"frame_ids":[fa,fb],**bridge,"direct_attempt":direct}
        registrations.append(reg)
        if reg["status"] != "accepted":
            issues.append({"frame_ids":[fa,fb],"reason":"No trusted registration; repeated views may remain duplicate instances"})
            continue
        H = np.array(reg["homography"])
        h,w = images[fb].shape[:2]
        warped = cv2.warpPerspective(images[fa],H,(w,h))
        for a,b in itertools.product(by_frame[fa],by_frame[fb]):
            ev = pair_evidence(a,b,H,warped,images[fb],config)
            if ev:
                ev["registration_ref"] = len(registrations)-1
                edges.append(ev)
    groups,accepted,uncertain = associate(observations,edges,config["ambiguity_margin"])
    # Precision-safe count above; the count if every near-tie were merged (margin 0) is the lower bound.
    groups_ties_merged = associate(observations,edges,0.0)[0] if edges else groups
    if track_transforms and config.get("association") in ("track","track_v2"):
        # Border-clipped stub groups join the full-length group that contains them (pipeline/cross_view.clipped_partial_links).
        seg_map,held = track_report.get("_segments") or {},track_report.get("held_frames") or []
        groups,stub_links = cross_view.clipped_partial_links(groups,observations,frames,track_transforms,seg_map,held,config.get("track"))
        accepted.extend(stub_links)
        groups_ties_merged = cross_view.clipped_partial_links(groups_ties_merged,observations,frames,track_transforms,seg_map,held,config.get("track"))[0]
        track_report["clipped_partial_merges"] = len(stub_links)
        single_merge = lambda gs,ids: cross_view.singleton_merge_links(gs,observations,frames,track_transforms,seg_map,held,config.get("track"),eligible=ids)[0]
    readings,resolutions = _stage_records(readings_path,"readings"),_stage_records(resolutions_path,"resolutions")
    gate_cfg = {"min_row":int(config.get("scene_gate_min_row",3)),"row_gap_box":config.get("row_gap_box","narrower")}
    gate_on = config.get("non_book_gate")
    gate_on = os.environ.get("LIBRARY_NON_BOOK_GATE","1") != "0" if gate_on is None else bool(gate_on)
    row_support = spine_row_support(observations,gate_cfg) if gate_on else {}
    gate_report = {"enabled":gate_on,**gate_cfg,"frames_with_books":len({o["frame_id"] for o in observations if o["kind"] == "book"}),
        "frames_gated":sorted({o["frame_id"] for o in observations if row_support.get(o["observation_id"],{}).get("frame_max_row",99) < gate_cfg["min_row"]}),
        "observations_gated":sum(1 for r in row_support.values() if r["frame_max_row"] < gate_cfg["min_row"]),
        "observations_isolated":sum(1 for r in row_support.values() if r["row"] < 2),
        "groups_demoted":{"scene_gate":0,"no_neighbour":0},"kept_by_accepted_identity":0}

    def demotion_of(group,count=False):
        if not row_support or by_id[group[0]]["kind"] != "book":
            return None
        d = non_book_demotion(group,row_support,gate_cfg)
        if d and _identity([by_id[oid] for oid in group],readings,resolutions)[0]["decision"] == "accepted":
            if count:
                gate_report["kept_by_accepted_identity"] += 1
            return None  # a catalogue-matched spine is a book whatever the frame looked like
        if d and count:
            gate_report["groups_demoted"][d["rule"]] += 1
        return d
    books_if_ties_merged = sum(1 for g in groups_ties_merged if by_id[g[0]]["kind"] == "book" and not demotion_of(g))
    # Demoted (ambiguous) groups are outside the count range as well as book_count.
    tiers,count_tiers = count_range([g for g in groups if not demotion_of(g)],observations,frames,(track_report or {}).get("held_frames") or [],single_merge,config.get("count_range"))
    uncertain_ids = {oid for e in uncertain for oid in e["observation_ids"]}
    uncertain_tiles = candidates.get("data",{}).get("uncertain_tile_associations",[])
    uncertain_ids |= {oid for e in uncertain_tiles for oid in e["observation_ids"] if oid in by_id}
    unregistered_frames = {fid for r in registrations if r["status"] != "accepted" for fid in r["frame_ids"]}
    unregistered_ids = {o["observation_id"] for o in observations if o["frame_id"] in unregistered_frames}
    # Eviction does not mean native evidence is missing.
    missing_evidence_ids = {o["observation_id"] for o in observations if o["frame_id"] not in valid_frames}
    uncertain_ids |= unregistered_ids | missing_evidence_ids
    if hold_frames:
        gap = float((config.get("track") or {}).get("max_gap_s", 6.0))
        hold_times = [frames[f]["timestamp_s"] for f in hold_frames if f in frames]
        near_hold = {fid for fid in frames if any(abs(frames[fid]["timestamp_s"]-t) <= gap for t in hold_times)}
        uncertain_ids |= {o["observation_id"] for o in observations if o["frame_id"] in near_hold}
    books,items,ambiguous = [],[],[]
    for group in groups:
        obs = [by_id[oid] for oid in group]
        # Best evidence: sharp large view; no identity dependency.
        best = max(obs,key=lambda o:((o.get("confidence") or 0),
            (o["bbox_xyxy"][2]-o["bbox_xyxy"][0])*(o["bbox_xyxy"][3]-o["bbox_xyxy"][1])))
        kind = best["kind"]
        demotion = demotion_of(group,count=True) if kind == "book" else None
        if demotion:
            kind = "ambiguous"
        # Stable id = the group's earliest observation (a merge keeps the earliest member, so the id survives re-merges and sorts in capture
        # order); display_index is the human enumeration of this inventory and changes with every merge.
        instance = {"instance_id":C.stable_instance_id("bk" if kind == "book" else "it",min(group)),"display_index":len(books)+1 if kind == "book" else len(items)+1,
            "observation_ids":sorted(group),"frame_ref":best["frame_id"],"bbox_xyxy":best["bbox_xyxy"],"polygon_xy":best.get("polygon_xy"),
            "kind":kind,"orientation":best.get("orientation","unknown"),"confidence":best.get("confidence"),
            "shelf":best.get("shelf_hint"),"position":None,"uncertain_association":bool(set(group)&uncertain_ids),
            "evidence_refs":[{"observation_id":o["observation_id"],"frame_id":o["frame_id"],
                "image_path":frames[o["frame_id"]]["image_path"],"timestamp_s":frames[o["frame_id"]]["timestamp_s"],
                "bbox_xyxy":o["bbox_xyxy"],"response_ref":o.get("response_ref")} for o in obs],
            "dedup_reasons":[e for e in accepted if set(e["observation_ids"]).issubset(group)] or ["Single observation; no unsupported title-based merge"],
            "issues":[i for o in obs for i in o.get("issues",[])]}
        if instance["uncertain_association"]:
            instance["issues"].append("Uncertain cross-frame physical association; retained separate instances")
        if set(group) & (unregistered_ids | missing_evidence_ids):
            instance["issues"].append("At least one view pair lacks trusted registration/evidence; duplicate instances possible")
        if kind == "book":
            identity,identity_issues = _identity(obs,readings,resolutions)
            instance.update(identity=identity,status="identified" if identity["decision"] == "accepted" else "unidentified",
                spine_height_cm=None,spine_thickness_cm=None,count_tier=tiers.get(tuple(group),"confident"))
            instance["issues"].extend(identity_issues)
            books.append(instance)
        else:
            attributes = detection_attributes(obs)
            if demotion:
                attributes.update(category=demotion["category"],classification="ambiguous",
                    issues=[demotion["reason"]]+[i for i in attributes["issues"] if "excluded from definite book count" not in i])
                instance["non_book_demotion"] = demotion
            instance["issues"].extend(attributes.pop("issues"))
            instance.update(attributes)
            items.append(instance)
            if kind == "ambiguous":
                ambiguous.append(instance["instance_id"])
    consolidation = None
    # Room structure (shelf board read as `desk`, the bookcase as `shelving`) is conflict evidence for spines, never an item line (pipeline/items.py).
    context_only = [i for i in items if i["category"] in ITEM_CONTEXT_ONLY and not i.get("non_book_demotion")]
    items[:] = [i for i in items if i not in context_only]
    ambiguous[:] = [a for a in ambiguous if a not in {i["instance_id"] for i in context_only}]
    if config.get("consolidate_items",True) and items:
        seg_of = (track_report or {}).get("_segments") or {}
        cached_image = lambda fid: images[fid] if fid in images else load_image(fid)   # frames already decoded for association
        groups_,links_ = consolidate_items_v2(items,observations,{f:frames[f]["timestamp_s"] for f in frames},cached_image,
            track_transforms if config.get("association") in ("track","track_v2") else None,seg_of,config.get("item_dedup"),cache_key=str(root),
            register=lambda fa,fb: _item_registration(str(root),fa,fb,cached_image,config))
        merged_ids = set()
        by_inst = {i["instance_id"]:i for i in items}
        for grp in groups_:
            if len(grp) < 2:
                continue
            members = [by_inst[g] for g in grp]
            keep = max(members,key=lambda i:(len(i["observation_ids"]),i.get("confidence") or 0))
            for other in members:
                if other is keep:
                    continue
                keep["observation_ids"] = sorted(set(keep["observation_ids"]) | set(other["observation_ids"]))
                keep["uncertain_association"] = bool(keep.get("uncertain_association") or other.get("uncertain_association"))
                keep["evidence_refs"] = keep["evidence_refs"] + other["evidence_refs"]
                keep["issues"] = keep["issues"] + [i for i in other["issues"] if i not in keep["issues"]]
                merged_ids.add(other["instance_id"])
            keep["dedup_reasons"] = [r for r in keep["dedup_reasons"] if not isinstance(r,str)] + [l for l in links_ if set(l["instances"]) <= set(grp)] or keep["dedup_reasons"]
        items[:] = [i for i in items if i["instance_id"] not in merged_ids]
        for i in items:   # a consolidated item keeps the id of its earliest observation (the kept member was the best-seen one, not the earliest)
            i["instance_id"] = C.stable_instance_id("it",min(i["observation_ids"]))
        ambiguous[:] = [i["instance_id"] for i in items if i["kind"] == "ambiguous"]
        consolidation = {"merged_instances":len(merged_ids),"links":len(links_),"rule":"pipeline.items.consolidate_items_v2",
            "links_by_rule":dict(Counter(l["rule"] for l in links_))}
    # One sighting is not enough for an item line (Yugo hand check: most were blurred re-views or half-visible fragments). Kept as review candidates.
    item_cfg = {**ITEM_DEDUP_DEFAULTS,**(config.get("item_dedup") or {})}
    min_sightings,floors = int(item_cfg["min_sightings"]),item_cfg["min_confidence"]
    def unconfirmed(i):
        if i.get("non_book_demotion"):
            # A demoted spine group seen once is a review candidate like any one-sighting item (model-read item review: 22 of YugoTest5's
            # 185 lines were one-sighting `book_or_isolated_object` lines showing a book). Still ambiguous, never in book_count, still in count_upper.
            if len(i["observation_ids"]) < min_sightings and item_cfg.get("demoted_min_sightings", True):
                return f"Demoted spine group seen in fewer than {min_sightings} frames; review candidate, not an inventory line"
            return None
        if len(i["observation_ids"]) < min_sightings:
            return f"Seen in fewer than {min_sightings} frames; review candidate, not an inventory line"
        best = max((by_id[o].get("confidence") or 0) for o in i["observation_ids"])
        if best < floors.get(i["category"],0):
            return f"Best {i['category']} confidence {best:.2f} below the category floor {floors[i['category']]}; review candidate, not an inventory line"
        return None
    item_candidates = [i for i in items if unconfirmed(i)]
    for i in item_candidates:
        i["issues"].append(unconfirmed(i))
        i["status"] = "unconfirmed_candidate"
    items[:] = [i for i in items if i not in item_candidates]
    ambiguous[:] = [a for a in ambiguous if a not in {i["instance_id"] for i in item_candidates}]
    item_filter = {"context_only":dict(Counter(i["category"] for i in context_only)),"unconfirmed_candidates":len(item_candidates),"min_sightings":min_sightings,"min_confidence":floors}
    if track_report is not None:
        seg_of = track_report.pop("_segments", None) or {}  # internal; the saved artifact keeps only where each tracking segment starts
        by_time = sorted(seg_of,key=lambda f:(frames[f]["timestamp_s"],f) if f in frames else (0.0,f))
        track_report["segment_starts"] = [f for i,f in enumerate(by_time) if i and seg_of[f] != seg_of[by_time[i-1]]]
    for i in items + item_candidates:   # review taxonomy for labels it lacks (pipeline/items.ITEM_TAXONOMY); the detector's label is kept beside it
        mapped = ITEM_TAXONOMY.get(i.get("category"))
        if mapped:
            i["detector_category"],i["category"] = i["category"],mapped
    demoted_candidates = sum(1 for i in item_candidates if i.get("non_book_demotion"))
    for n,i in enumerate(items,1):
        i["display_index"] = n
    for n,i in enumerate(item_candidates,1):
        i["display_index"] = n
    max_in_frame = max(Counter(o["frame_id"] for o in observations if o["kind"] == "book").values(),default=0)
    has_uncertainty = bool(uncertain_ids)
    counts = {"book_count":len(books),"item_count":len(items),"item_candidates":len(item_candidates),"ambiguous_instances":len(ambiguous),
        "ambiguous_demoted_spines":sum(gate_report["groups_demoted"].values()),
        "uncertain_observations":len(uncertain_ids | {o["observation_id"] for o in observations if o["kind"] == "ambiguous"}),
        "uncertain_associations":len(uncertain)+len(uncertain_tiles),
        "count_lower":None if uncertain_tiles else min(books_if_ties_merged, len(books)),
        "count_if_near_ties_merged":books_if_ties_merged,"count_lower_note":"Lower bound = groups if every near-tie competing match were merged (ambiguity margin 0); per-frame maximum is not a bound on physical books.",
        "count_upper":len(books)+len(ambiguous)+demoted_candidates,
        "unregistered_frame_pairs":sum(r["status"] != "accepted" for r in registrations),
        "unknown_detection_confidence":sum(o.get("confidence") is None for o in observations),
        **count_tiers,
        "bounds_scope":"Detected candidates only, conditional on per-frame boxes/classification; excludes missed/merged spines and uncaptured items"}
    counts["physical_inventory_complete"] = False
    counts["missed_books"] = None
    counts["clutter_false_positives"] = None
    counts["duplicate_instance_errors"] = None
    counts["merged_instance_errors"] = None
    artifact = {"schema_version":1,"run_id":manifest.get("run_id",root.name),"stage":"dedup",
        "status":"partial" if issues or has_uncertainty or ambiguous else "complete",
        "data":{"books":books,"items":items,"item_candidates":item_candidates,"context_objects":context_only,"ambiguous_instance_ids":ambiguous,"counts":counts,
            "registrations":registrations,"associations":accepted,"uncertain_associations":uncertain,
            "uncertain_tile_associations":uncertain_tiles,
            "coverage_gaps":manifest.get("data",{}).get("coverage_gaps",[])},
        "issues":issues,"metrics":{"elapsed_s":round(time.perf_counter()-start,4),"provider":"OpenCV "+config["feature_method"].upper()+"/RANSAC + registered texture/color",
            "request_count":0,"actual_cost":0,"cost_provenance":"Local deterministic association; no API requests",
            "observations":len(observations),"merged_links":len(accepted),"uncertain_links":len(uncertain),
            "registration_pairs":len(registrations),"native_image_cache_limit":cache_limit,"association":config.get("association","feature_registration"),"track_report":track_report,"item_consolidation":consolidation,"item_filter":item_filter,"non_book_gate":gate_report,
            "association_thresholds":{k:config[k] for k in DEFAULTS}}}
    if manifest.get("fixture") or candidates.get("fixture"):
        artifact["fixture"] = True
    errors = C.validate_inventory(artifact,manifest,set(by_id))
    if errors:
        raise ValueError("Invalid inventory contract: " + "; ".join(errors[:5]))
    return save_json(root/config.get("output_name","inventory.json"),artifact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates",required=True)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--readings")
    parser.add_argument("--resolutions")
    parser.add_argument("--output-name",default="inventory.json",help="Use a new filename to preserve a published inventory")
    parser.add_argument("--bridge-registration",action="store_true",help="Use unchanged intermediate evidence when direct registration fails")
    parser.add_argument("--feature-method",choices=["orb","sift"],default="orb")
    parser.add_argument("--association-geometry",choices=['bbox','polygon'],default='bbox')
    parser.add_argument("--appearance-mode",choices=['direct','aligned_polygon'],default='direct')
    parser.add_argument("--association",choices=['feature_registration','track','track_v2'],default='feature_registration',help="track = tracked-motion chain + residual alignment + appearance (pipeline/cross_view.py); track_v2 = track + motion-compensated IoU tracker (pipeline/tracker.py)")
    args = parser.parse_args()
    print(merge(args.candidates,args.manifest,args.out,{"bridge_registration":args.bridge_registration,"feature_method":args.feature_method,"output_name":args.output_name,"association_geometry":args.association_geometry,"appearance_mode":args.appearance_mode,"association":args.association},
        readings_path=args.readings,resolutions_path=args.resolutions))


if __name__ == "__main__":
    main()
