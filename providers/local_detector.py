"""Optional local pretrained OBB spines + YOLO-World non-book candidates.

This is a credential-free fallback, not a Gemini/cache result or ground truth.
Run with the isolated comparison Python runtime. No hosted OCR is imported.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import threading
import time
import sys
from pathlib import Path

from pipeline import contracts as C
from pipeline.detect import _detect, read_json, save_json, overlap

NON_BOOK_PROMPTS = [
    ("plant","plant"),("flower pot","pot"),("vase","vase"),
    ("framed sign","framed_sign"),("picture frame","frame"),
    ("bookcase","shelving"),("sofa","sofa"),("chair","chair"),
    ("table","table"),("lamp","lamp"),("decorative letter","decorative_letters"),
    ("electronic appliance","electronics")]

# Keep the predicted object label; families are review metadata, not permission
# to merge different containers, plants, furniture or adjacent objects.
FAST_ITEMS_PROMPTS = NON_BOOK_PROMPTS + [
    ('storage box', 'box'), ('basket', 'basket'), ('mug', 'cup'),
    ('glass jar', 'jar'), ('sculpture', 'sculpture'), ('cushion', 'cushion'),
    ('curtain', 'curtain'), ('vinyl record', 'record'),
    ('decorative ornament', 'ornament'), ('small wooden sign', 'sign'),
    # Plant synonyms: YOLO-World scores 'plant' low on small potted/flower objects.
    ('potted plant', 'plant'), ('houseplant', 'plant'), ('flowers', 'plant'), ('green leaves', 'plant')]
# Desk/office/room objects seen while the user aims the camera after Start (21:14 live sweep: two rows of laptop keys read as flat spines). A spine box overlapping
# one becomes `ambiguous` (kept for review, never counted). Room structure is conflict evidence only: kept in raw_predictions, never emitted as an inventory object.
# Faces/sign text (IESTest1) are left to the scene gate in pipeline/items.py: YOLO-World finds no 'face', and a 'person' box also covers books beside the head.
DESK_OFFICE_PROMPTS = [
    ('computer keyboard', 'keyboard'), ('computer monitor', 'monitor'), ('laptop', 'laptop'), ('computer mouse', 'mouse'),
    ('cable', 'cable'), ('mobile phone', 'phone'), ('desk', 'desk'), ('floor', 'floor'), ('door', 'door'), ('window', 'window'),
    ('radiator', 'radiator')]
FAST_ITEMS_PROMPTS = FAST_ITEMS_PROMPTS + DESK_OFFICE_PROMPTS
CONFLICT_ONLY_CATEGORIES = {'cable', 'floor', 'door', 'window', 'radiator'}
# A thin false spine (one row of keys) lies INSIDE the keyboard box, far below IoU .5: for these small office objects containment of the spine box also conflicts.
CONTAINER_CONFLICT_CATEGORIES = {'keyboard', 'laptop', 'monitor', 'phone', 'mouse'}
CONTAINMENT_CONFLICT = .8
ITEM_REVIEW_FAMILIES = {
    **dict.fromkeys(['plant', 'pot', 'vase'], 'potted'),
    **dict.fromkeys(['framed_sign', 'frame', 'record', 'sign'], 'framed'),
    **dict.fromkeys(['box', 'basket', 'cup', 'jar'], 'container'),
    **dict.fromkeys(['lamp', 'decorative_letters', 'electronics', 'sculpture', 'ornament'], 'decor'),
    **dict.fromkeys(['shelving', 'sofa', 'chair', 'table', 'cushion', 'curtain'], 'furniture/structure'),
    **dict.fromkeys(['keyboard', 'monitor', 'laptop', 'mouse', 'phone', 'cable'], 'electronics'),
    **dict.fromkeys(['desk', 'floor', 'door', 'window', 'radiator'], 'furniture/structure')}


def contained_fraction(inner, outer):
    """Share of the inner box's area inside the outer box (0..1)."""
    w = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return w * h / max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-6)


def book_conflicts(book, nonbooks, excluded=frozenset({'shelving', 'sofa', 'table', 'chair'})):
    """Non-book predictions that make a spine box ambiguous: same-region IoU >= .5, or the spine lies inside a small office object."""
    return [o for o in nonbooks if o['category'] not in excluded and (overlap(book['bbox_xyxy'], o['bbox_xyxy']) >= .5 or
            (o['category'] in CONTAINER_CONFLICT_CATEGORIES and contained_fraction(book['bbox_xyxy'], o['bbox_xyxy']) >= CONTAINMENT_CONFLICT))]

REVIEW_PROMPTS=NON_BOOK_PROMPTS+[("decorative sign","framed_sign"),
    ("arrow sign","framed_sign"),("decorative letters","decorative_letters"),
    ("potted plant","plant")]

STACK_PROMPTS=[('horizontal book spine','possible_horizontal_spine'),('stack of books','book_stack')]
PROFILES = ['baseline', 'review_v1', 'review_v2', 'review_v3', 'review_v4', 'fast_960', 'fast_1024', 'fast_1280', 'fast_1280_512', 'fast_1280_512_n5', 'fast_items']


def frame_pass_plan(profile, frame_index, image_size=None):
    """Sparse item inference never transports boxes into different source pixels."""
    if profile not in PROFILES:
        raise ValueError('Unknown local detector profile')
    fast = profile.startswith('fast_')
    size = image_size if image_size is not None else int(profile.split('_')[1]) if fast and profile != 'fast_items' else 1280
    stride = 1 if profile == 'fast_items' else 5 if profile.endswith('_n5') else 3 if fast else 1
    return {'image_size': size, 'item_stride': stride,
            'run_items': frame_index % stride == 0}


def same_spine_prediction(a,b):
    """Cross-pass repeats in the same saved pixels, preserving source exclusion."""
    import numpy as np
    from pipeline.dedup import polygon_iou
    iou=polygon_iou(a['polygon_xy'],b['polygon_xy'])
    if iou>=.85:return True
    def source(o):
        p=o.get('detection_provenance',{})
        return (p.get('pass'),tuple(p.get('source_tile',[])))
    sources={source(a)}|{source(x) for x in a.get('detection_aliases',[]) if 'polygon_xy' in x}
    if source(b) in sources or iou<.75:return False
    pa,pb=np.asarray(a['polygon_xy']),np.asarray(b['polygon_xy'])
    lengths=[np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1) for p in [pa,pb]]
    areas=[float(x.min()*x.max()) for x in lengths]
    short=min(x.min() for x in lengths)
    distance=float(np.linalg.norm(pa.mean(axis=0)-pb.mean(axis=0)))/max(1,short)
    return distance<=.55 and .67<=areas[0]/max(1e-6,areas[1])<=1.5


def native_prediction_polygon(points,tile,pass_name,width,height):
    import numpy as np
    points=np.asarray(points,float).reshape(-1,2).copy()
    tx,ty,tw,th=tile
    if 'rotated_clockwise' in pass_name:
        points=np.column_stack([points[:,1],th-points[:,0]])
    points+=np.array([tx,ty])
    points[:,0]=np.clip(points[:,0],0,width);points[:,1]=np.clip(points[:,1],0,height)
    return points


def object_polygon_iou(a,b):
    from pipeline.dedup import polygon_iou
    return polygon_iou(a['polygon_xy'],b['polygon_xy'])


def fuse_predictions(books,nonbooks):
    """One region for strong duplicate predictions; retain raw alternatives.

    Plant and pot are distinct objects even when overlapping. Adjacent leaning
    spines are not duplicates just because axis-aligned rectangles overlap.
    """
    kept=[];suppressed=[]
    for obj in sorted(books,key=lambda o:-o['confidence']):
        duplicate=next((x for x in kept if same_spine_prediction(x,obj)),None)
        if duplicate:
            duplicate.setdefault('detection_aliases',[]).append(obj)
            suppressed.append({'reason':'Near-identical oriented spine prediction','prediction':obj})
        else:kept.append(obj)
    items=[]
    for obj in sorted(nonbooks,key=lambda o:-o['confidence']):
        duplicate=next((x for x in items if overlap(x['bbox_xyxy'],obj['bbox_xyxy'])>=.85 and
            (x['category']==obj['category'] or set([x['category'],obj['category']])<=set(['pot','vase','pot_or_vase']))),None)
        if duplicate:
            duplicate.setdefault('detection_aliases',[]).append(obj)
            if duplicate['category']!=obj['category']:
                duplicate.update(kind='ambiguous',category='pot_or_vase')
                duplicate['evidence_notes']+='; competing container categories at same region retained'
            suppressed.append({'reason':'Same-region non-book duplicate/category alternative','prediction':obj})
        else:items.append(obj)
    consumed=set()
    for book in kept:
        conflicts=[(i,o) for i,o in enumerate(items) if o in book_conflicts(book,items,frozenset({'shelving','sofa','table','chair','plant'}))]
        if conflicts:
            _,best=max(conflicts,key=lambda x:overlap(book['bbox_xyxy'],x[1]['bbox_xyxy']))
            book.update(kind='ambiguous',category='book_or_'+best['category'])
            book['evidence_notes']+='; conflicting non-book shape/category, not a confirmed book'
            for i,o in conflicts:
                if overlap(book['bbox_xyxy'],o['bbox_xyxy'])>=.8:
                    consumed.add(i);book.setdefault('detection_aliases',[]).append(o)
                    suppressed.append({'reason':'Same-region book/non-book disagreement represented once as ambiguous','prediction':o})
        elif book.get('spine_aspect_ratio',2)<1.8:
            book.update(kind='ambiguous',category='book_or_decoration')
            book['evidence_notes']+='; near-square shape lacks distinctive narrow-spine geometry; preserve for review'
    return kept+[o for i,o in enumerate(items) if i not in consumed and o['category'] not in CONFLICT_ONLY_CATEGORIES],suppressed


def prepare_run(source_run,output_run):
    source=Path(source_run).resolve();output=Path(output_run).resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite versioned run")
    manifest=read_json(source/"manifest.json")
    problems=C.validate_manifest(manifest,source)
    if problems:
        raise ValueError("Invalid source evidence: "+"; ".join(problems[:5]))
    output.mkdir(parents=True)
    hashes=[]
    for frame in manifest["data"]["frames"]:
        src=source/frame["image_path"];dst=output/frame["image_path"]
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        source_sha=hashlib.sha256(src.read_bytes()).hexdigest()
        assert source_sha == hashlib.sha256(dst.read_bytes()).hexdigest()
        hashes.append({"frame_id":frame["frame_id"],"timestamp_s":frame["timestamp_s"],"image_path":frame["image_path"],"sha256":source_sha})
    manifest["run_id"]=output.name
    save_json(output/"manifest.json",manifest)
    save_json(output/"evidence_copy.json",{"source_run":source.name,"source_manifest_sha256":hashlib.sha256((source/"manifest.json").read_bytes()).hexdigest(),
        "frames":hashes,"frame_ids_timestamps_selections_preserved":True,"frame_bytes_preserved":True})
    return output/"manifest.json"


class LocalDetector:
    """Same async adapter boundary as the shared provider; local weights only."""
    def __init__(self,weights_dir,*,image_size=None,book_conf=.5,item_conf=.25,profile='baseline'):
        plan = frame_pass_plan(profile, 0, image_size)
        import torch
        import ultralytics.utils
        from ultralytics import YOLO,YOLOWorld
        self.weights_dir=Path(weights_dir).resolve()
        torch.set_num_threads(4)
        ultralytics.utils.WEIGHTS_DIR=self.weights_dir
        t=time.perf_counter()
        self.books=YOLO(str(self.weights_dir/"book-spine-best.pt")).to("cpu")
        self.items=YOLOWorld(str(self.weights_dir/"yolov8m-worldv2.pt")).to("cpu")
        self.nonbook_prompts=(REVIEW_PROMPTS+STACK_PROMPTS if profile in {'review_v3','review_v4'} else REVIEW_PROMPTS) if profile.startswith('review_') else NON_BOOK_PROMPTS
        if profile == 'fast_items':
            self.nonbook_prompts = FAST_ITEMS_PROMPTS
        self.items.set_classes([p for p,c in self.nonbook_prompts])
        self.setup_s=time.perf_counter()-t
        self._inference_lock=threading.Lock()
        self.image_size,self.book_conf,self.item_conf=plan['image_size'],book_conf,item_conf
        self._frame_index = 0
        self.profile=profile
        self.horizontal_conf=.15 if profile=='review_v4' else book_conf

    async def __call__(self,*,image_path,**kwargs):
        if kwargs.get('stage') not in (None,'detect'):
            raise ValueError('Local detector can only serve the detect stage, never OCR')
        # Inference runs in a thread when called by an async live task; cancellation
        # can discard results at the session boundary without blocking voice.
        return await asyncio.to_thread(self._infer,image_path)

    def _infer(self,image_path):
        # Ultralytics predictors carry mutable state. Serialize this adapter's
        # model access while the caller remains free to cancel/discard the caller.
        with self._inference_lock:
            return self._infer_locked(image_path)

    def _infer_locked(self,image_path):
        import cv2
        import numpy as np
        t=time.perf_counter()
        image=cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Cannot open local evidence")
        height,width=image.shape[:2]
        frame_index = self._frame_index
        plan = frame_pass_plan(self.profile, frame_index, self.image_size)
        self._frame_index += 1
        results=[('native',(0,0,width,height),self.books.predict(image,imgsz=self.image_size,conf=self.book_conf,device="cpu",verbose=False)[0])]
        if self.profile.startswith('review_'):
            rotated=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
            results.append(('rotated_clockwise',(0,0,width,height),self.books.predict(rotated,imgsz=self.image_size,conf=self.horizontal_conf,device="cpu",verbose=False)[0]))
        if self.profile in {'review_v2','review_v3','review_v4'}:
            from pipeline.detect import tiles
            for tile in tiles(width,height,960,.2):
                tx,ty,tw,th=tile
                crop=image[ty:ty+th,tx:tx+tw]
                for name,source in [('tile_native',crop),('tile_rotated_clockwise',cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE))]:
                    confidence=self.horizontal_conf if 'rotated' in name else self.book_conf
                    results.append((name,tile,self.books.predict(source,imgsz=self.image_size,conf=confidence,device='cpu',verbose=False)[0]))
        objects=[]
        for pass_name,tile,result in results:
          if result.obb is not None:
            if self.profile.startswith('fast_'):
                predictions=zip(result.obb.xyxyxyxy.cpu().numpy().reshape(-1,4,2),result.obb.conf.cpu().numpy())
            else:
                predictions=((obb.xyxyxyxy.cpu().numpy().reshape(-1,2),float(obb.conf[0])) for obb in result.obb)
            for points,confidence in predictions:
                points=native_prediction_polygon(points,tile,pass_name,width,height)
                lo=points.min(axis=0);hi=points.max(axis=0)
                edges=np.roll(points,-1,axis=0)-points
                lengths=np.linalg.norm(edges,axis=1)
                long=edges[int(np.argmax(lengths))]
                angle=float(np.degrees(np.arctan2(long[1],long[0])))%180
                orientation="flat" if min(angle,180-angle)<20 else "upright" if abs(angle-90)<12 else "leaning"
                if 'rotated_clockwise' in pass_name and orientation!='flat':
                    continue
                objects.append({"bbox_xyxy":[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],
                    "polygon_xy":points.tolist(),"kind":"book","category":"book","orientation":orientation,
                    "confidence":float(confidence),"occlusion":"unknown",
                    "spine_aspect_ratio":float(max(lengths)/max(1e-6,min(lengths))),
                    "detection_provenance":{"pass":pass_name,"source_tile":list(tile),"model":"custom_book_OBB","coordinate_space":"original_native_pixels"},
                    "evidence_notes":"Pretrained oriented spine-shape prediction; no title/language filtering; possibly merged or occluded spines require review"})
                if 'rotated' in pass_name and float(confidence)<self.book_conf:
                    objects[-1].update(kind='ambiguous',category='possible_horizontal_spine')
                    objects[-1]['evidence_notes']+='; lower-confidence rotated horizontal proposal, not a confirmed book'
        item_size=512 if self.profile.startswith('fast_1280_512') else 640 if self.profile in {'fast_1280','fast_items'} else self.image_size
        nonbook_result=self.items.predict(image,imgsz=item_size,conf=self.item_conf,device="cpu",verbose=False)[0] if plan['run_items'] else None
        nonbooks=[]
        if nonbook_result is not None and nonbook_result.boxes is not None:
            for box in nonbook_result.boxes:
                category=self.nonbook_prompts[int(box.cls[0])][1]
                coords=box.xyxy[0].cpu().tolist()
                coords=[max(0,coords[0]),max(0,coords[1]),min(width,coords[2]),min(height,coords[3])]
                nonbooks.append({"bbox_xyxy":coords,"kind":"non_book","category":category,"orientation":"unknown",
                    "confidence":float(box.conf[0]),"occlusion":"unknown",
                    "detection_provenance":{"pass":"native","model":"YOLOWorld_m_v2","prompt":self.nonbook_prompts[int(box.cls[0])][0],"coordinate_space":"original_native_pixels"},
                    "evidence_notes":"Pretrained YOLO-World category prediction; words do not establish a book; plant/pot boxes may overlap and material/model unknown"})
                if self.profile == 'fast_items':
                    nonbooks[-1]['detection_provenance']['item_family'] = ITEM_REVIEW_FAMILIES[category]
                if category in {'possible_horizontal_spine','book_stack'}:
                    nonbooks[-1].update(kind='ambiguous',orientation='flat',
                        evidence_notes='Pretrained horizontal/stack shape region; individual layer count unverified, do not count a whole stack as one confirmed book')
        # A direct spatial kind disagreement is preserved, never silently counted
        # as a confirmed book merely because one model emitted a book box.
        for book in objects if self.profile=='baseline' or self.profile.startswith('fast_') else []:
            conflicts=book_conflicts(book,nonbooks)
            if conflicts:
                book["kind"]="ambiguous"
                book["category"]="book_or_"+conflicts[0]["category"]
                book["evidence_notes"] += "; conflicts with non-book model category at same location"
        suppressed=[]
        raw=objects+nonbooks
        if self.profile.startswith('review_'):
            import copy
            raw=copy.deepcopy(raw)
            objects,suppressed=fuse_predictions(objects,nonbooks)
        else:
            objects=[o for o in raw if o.get('category') not in CONFLICT_ONLY_CATEGORIES]
        issues = [] if plan['run_items'] else [{'stage':'non_book_detection','status':'unprocessed','frame_ref':str(image_path),
            'reason':f'Profile samples non-books every {plan["item_stride"]} inference frames; no boxes reused from another frame'}]
        if not plan['run_items']:
            for book in objects:
                book['evidence_notes'] += '; non-book conflict pass not scheduled on this frame'
        return {"data":{"objects":objects,"raw_predictions":raw,"suppressed_predictions":suppressed},"coordinate_format":"xyxy_pixels","issues":issues,
            "metrics":{"provider":"local_pretrained","model":"custom_book_OBB+YOLOWorld_m_v2","request_count":0,
                "local_inference_count":1,"cache_hits":0,"latency_s":time.perf_counter()-t,"actual_cost":0,
                "cost_provenance":"Local CPU inference; no API requests","image_size":self.image_size,
                "book_conf":self.book_conf,"item_conf":self.item_conf,
                "horizontal_conf":self.horizontal_conf,
                "source_image_sha256":hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
                "source_width_px":width,"source_height_px":height,"profile":self.profile,
                "model_forward_count":len(results)+int(plan['run_items']),"suppressed_prediction_count":len(suppressed),
                "non_book_pass_status":"ok" if plan['run_items'] else "unprocessed",
                "non_book_stride":plan['item_stride'],'non_book_image_size':item_size,'sampling_frame_index':frame_index}}


async def run(args):
    manifest=prepare_run(args.source_run,args.out)
    root=Path(args.out).resolve()
    adapter=LocalDetector(args.weights_dir,profile=args.profile)
    cfg={"image_request":adapter,"model":"local_pretrained","frame_ids":args.frame_ids.split(",") if args.frame_ids else None}
    path=await _detect(manifest,root,cfg)
    settings={"backend":"local_pretrained_fallback","fresh_local_inference":True,"hosted_provider_smoke":"MissingCredentialError in separate phase2_detection_baseline run",
        "image_size":adapter.image_size,"book_conf":adapter.book_conf,"item_conf":adapter.item_conf,"non_book_prompts":adapter.nonbook_prompts,"profile":adapter.profile,
        "setup_s":adapter.setup_s,"weights":[{"file":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in [adapter.weights_dir/"book-spine-best.pt",adapter.weights_dir/"yolov8m-worldv2.pt",adapter.weights_dir/"clip"/"ViT-B-32.pt"]],
        "reading":"No title/author/edition reading; the reader owns literal reading","accuracy":"Unavailable until frozen human labels; candidate counts are predictions"}
    save_json(root/"local_provider_settings.json",settings)
    return path


def serve(args):
    """Private persistent detection worker. Protocol stdout never carries model logs."""
    protocol=sys.stdout
    def emit(data):
        protocol.write(json.dumps(data,allow_nan=False)+'\n');protocol.flush()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            adapter=LocalDetector(args.weights_dir,profile=args.profile)
            emit({'status':'ready','setup_s':adapter.setup_s,'worker_pid':os.getpid(),
                'profile':args.profile,'provider':'local_pretrained','actual_API_requests':0})
        except Exception as exc:
            emit({'error_type':type(exc).__name__});return
        for line in sys.stdin:
            try:
                request=json.loads(line)
                result=adapter._infer(request['image_path'])
                result['metrics']['request_ref']=request.get('request_id')
                emit(result)
            except Exception as exc:
                emit({'error_type':type(exc).__name__})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run")
    p.add_argument("--out")
    p.add_argument("--weights-dir",required=True)
    p.add_argument("--frame-ids",help="Small smoke scope; omitted selected frames are unprocessed")
    p.add_argument("--serve",action="store_true")
    p.add_argument("--profile",choices=PROFILES,default='fast_items')
    args=p.parse_args()
    if args.serve:
        serve(args);return
    if not args.source_run or not args.out:
        p.error('--source-run and --out required unless --serve')
    print(asyncio.run(run(args)))


if __name__=="__main__":
    main()
