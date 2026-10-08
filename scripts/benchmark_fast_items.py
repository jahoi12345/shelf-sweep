"""Paired local inference, track association, and relative model-read item coverage.

No human labels, hosted calls, invented panorama coordinates or overwriting runs.
Both persistent processes initialize before timing; invocation order alternates.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
from zoneinfo import ZoneInfo

from pipeline import contracts as C
from pipeline.detect import _detect, save_json
from providers.local_detector import FAST_ITEMS_PROMPTS, ITEM_REVIEW_FAMILIES, NON_BOOK_PROMPTS
from providers.local_detector_process import LocalDetectorProcess
from scripts.benchmark_local_detector import agreement


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def project_box(box, matrix):
    import numpy as np
    x0, y0, x1, y1 = box
    points = np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]], float) @ np.asarray(matrix, float).T
    if not np.isfinite(points).all() or (abs(points[:,2]) < 1e-9).any():
        raise ValueError('Invalid panorama projection')
    points = points[:,:2] / points[:,2:3]
    return [float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]


def family(category):
    return ITEM_REVIEW_FAMILIES.get(category.removeprefix('book_or_'), 'unknown')


def visual_coverage(observations, visual, transforms):
    """The item consolidation's any-category overlap proxy plus a separate family check.

    Multiple observations can hit one visual object; no identity/precision claim.
    Containers and potted objects share the vase_pot visual class.
    """
    compatible = {'plant':{'potted'}, 'vase_pot':{'potted','container'},
                  'sign_frame':{'framed'}, 'decor':{'decor'}, 'furniture':{'furniture/structure'}}
    hits = defaultdict(list)
    for o in observations:
        if o['kind'] == 'book' or o['frame_id'] not in transforms:
            continue
        projected = project_box(o['bbox_xyxy'], transforms[o['frame_id']])
        for v in visual['objects']:
            iou = box_iou(projected, v['box'])
            if iou >= .3:
                hits[v['id']].append({'observation_id':o['observation_id'],'frame_id':o['frame_id'],
                    'category':o['category'],'family':family(o['category']),'iou':round(iou,4)})
    rows=[]
    for v in visual['objects']:
        matches=hits[v['id']]
        rows.append({'visual_id':v['id'],'name':v['name'],'class':v['class'],'certainty':v['certainty'],
            'hit_any_category':bool(matches),'hit_compatible_family':any(m['family'] in compatible[v['class']] for m in matches),
            'matches':matches})
    by_class={}
    for cls in sorted({v['class'] for v in rows}):
        selected=[v for v in rows if v['class']==cls]
        by_class[cls]={'total':len(selected),'hit_any_category':sum(v['hit_any_category'] for v in selected),
                       'hit_compatible_family':sum(v['hit_compatible_family'] for v in selected)}
    return {'status':'available','label':visual['label'],'total':len(rows),'iou_threshold':.3,
            'hit_any_category':sum(v['hit_any_category'] for v in rows),
            'hit_compatible_family':sum(v['hit_compatible_family'] for v in rows),'by_class':by_class,'objects':rows,
            'limits':'Relative to a model-read development list; any overlap, not one-to-one matching or human recall. Family agreement does not validate an exact category.'}


async def materialize(source, out, manifest, payloads):
    selected=json.loads(json.dumps(manifest))
    selected['run_id']=out.name
    for f in selected['data']['frames']:
        f['selected']=True
        destination=out/f['image_path'];destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/f['image_path'],destination)
    selected['data']['sampling'].update(sampled_frames=len(selected['data']['frames']),selected_frames=len(selected['data']['frames']))
    save_json(out/'manifest.json',selected)
    by_sha={p['metrics']['source_image_sha256']:p for p in payloads}
    async def replay(*,image_path,**kwargs):
        value=json.loads(json.dumps(by_sha[sha(image_path)]))
        for issue in value.get('issues',[]):
            issue['frame_ref']=str(image_path)
        return value
    await _detect(out/'manifest.json',out,{'image_request':replay,'model':'local_pretrained'})
    # A separate process makes association cancellation bounded too.
    process=await asyncio.create_subprocess_exec(sys.executable,'-m','pipeline.dedup',
        '--candidates',str(out/'candidates.json'),'--manifest',str(out/'manifest.json'),
        '--out',str(out),'--association','track',stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
    try:
        if await process.wait():raise RuntimeError('Track association failed; inspect saved candidates')
    finally:
        if process.returncode is None:
            process.kill();await process.wait()


async def run(args):
    source=Path(args.source).resolve();out=Path(args.out).resolve()
    manifest=json.loads((source/'manifest.json').read_text())
    if C.validate_manifest(manifest,source):raise ValueError('Invalid source manifest')
    frames=manifest['data']['frames'][:args.frames]
    if len(frames)!=args.frames:raise ValueError('Not enough source frames')
    manifest['data']['frames']=frames
    out.mkdir(parents=True,exist_ok=False)
    code_files=['providers/local_detector.py','providers/local_detector_process.py','pipeline/detect.py',
                'pipeline/dedup.py','pipeline/cross_view.py','pipeline/items.py',__file__]
    provenance={'source_run':str(source),'source_manifest_sha256':sha(source/'manifest.json'),
        'code_sha256':{str(p):sha(p) for p in code_files},
        'weights_sha256':{name:sha(Path(args.weights_dir)/name) for name in ['book-spine-best.pt','yolov8m-worldv2.pt','clip/ViT-B-32.pt']},
        'prompts':{'baseline':NON_BOOK_PROMPTS,'fast_items':FAST_ITEMS_PROMPTS},
        'method':'Alternate inference order each frame after both persistent adapters initialize; original JPEGs, native book polygons; CPU, 4 Torch threads; setup/association excluded from s/frame.',
        'frames':[{'frame_id':f['frame_id'],'timestamp_s':f['timestamp_s'],'sha256':sha(source/f['image_path'])} for f in frames],
        'actual_hosted_calls':0,'heavy_job_cutoff':args.cutoff}
    save_json(out/'provenance.json',provenance)
    records={'baseline':[],'fast_items':[]};payloads={p:[] for p in records};adapters={};setups={}
    try:
        for p in records:
            adapters[p]=LocalDetectorProcess(python=args.python,weights_dir=args.weights_dir,work_dir=out/p/'process',profile=p,timeout_s=120)
            setups[p]=await adapters[p].start()
        for i,f in enumerate(frames):
            for p in (['baseline','fast_items'] if i%2==0 else ['fast_items','baseline']):
                start=time.perf_counter();response=await adapters[p](image_path=source/f['image_path']);elapsed=time.perf_counter()-start
                payloads[p].append(response)
                save_json(out/p/'raw_responses'/f"{f['frame_id']}.json",response)
                records[p].append({'frame_id':f['frame_id'],'wall_s':elapsed,'metrics':response['metrics']})
            save_json(out/'progress.json',{'status':'running','stage':'inference','completed_pairs':i+1,'required_pairs':len(frames)})
            if (i+1)%10==0:print(f'Completed {i+1}/{len(frames)} paired frames',flush=True)
    finally:
        for adapter in adapters.values():await adapter.aclose()
    report={'status':'running','provenance':provenance,'profiles':{}}
    for p in records:
        save_json(out/'progress.json',{'status':'running','stage':'normalization_and_track','profile':p})
        started=time.perf_counter();await materialize(source,out/p,manifest,payloads[p])
        candidates=json.loads((out/p/'candidates.json').read_text());inventory=json.loads((out/p/'inventory.json').read_text())
        report['profiles'][p]={'mean_s_per_frame':statistics.mean(r['wall_s'] for r in records[p]),
            'median_s_per_frame':statistics.median(r['wall_s'] for r in records[p]),'setup':setups[p],
            'normalization_and_association_s':time.perf_counter()-started,'timed_frames':records[p],
            'candidates_status':candidates['status'],'counts':inventory['data']['counts'],
            'item_groups_by_category':dict(Counter(it['category'] for it in inventory['data']['items'])),
            'item_groups_by_family':dict(Counter(family(it['category']) for it in inventory['data']['items'])),
            'item_consolidation':inventory['metrics'].get('item_consolidation'),
            'visual_coverage':{'status':'unavailable','reason':'No visual inventory supplied; group counts are not coverage.'}}
        save_json(out/'benchmark.json',report)
    baseline=json.loads((out/'baseline/candidates.json').read_text())['data']['observations']
    candidate=json.loads((out/'fast_items/candidates.json').read_text())['data']['observations']
    report['book_agreement']=agreement(baseline,candidate)
    report['speedup']=report['profiles']['baseline']['mean_s_per_frame']/report['profiles']['fast_items']['mean_s_per_frame']
    if args.visual_inventory:
        visual=json.loads(Path(args.visual_inventory).read_text());pano=json.loads(Path(args.panorama).read_text());transforms={}
        # The visual list was drawn on component_001. Other walls are not joined.
        comp=next(c for c in pano['components'] if c['component_id']=='component_001')
        for f in comp['source_frames']:
            path=source/f['image_path']
            if path.is_file() and sha(path)==f['sha256']:
                transforms[f['frame_id']]=f['original_to_panorama_3x3']
            else:raise ValueError('Panorama source pixels mismatch; no visual scoring permitted')
        report['visual_provenance']={'inventory_sha256':sha(args.visual_inventory),'panorama_sha256':sha(args.panorama),
            'component':'component_001','hash_verified_transform_frames':len(transforms),
            'unprojectable_measured_frame_ids':[f['frame_id'] for f in frames if f['frame_id'] not in transforms]}
        for p,obs in [('baseline',baseline),('fast_items',candidate)]:
            coverage=visual_coverage(obs,visual,transforms)
            save_json(out/p/'visual_coverage.json',coverage)
            report['profiles'][p]['visual_coverage']={k:v for k,v in coverage.items() if k!='objects'}
    report['gates']={'speed_at_least_2x':report['speedup']>=2,
        'book_agreement_at_least_98pct':report['book_agreement']['baseline_prediction_recall_proxy']>=.98,
        'coverage_at_least_baseline':None}
    if args.visual_inventory:
        b=report['profiles']['baseline']['visual_coverage'];f=report['profiles']['fast_items']['visual_coverage']
        report['gates']['coverage_at_least_baseline']=f['hit_any_category']>=b['hit_any_category'] and f['hit_compatible_family']>=b['hit_compatible_family']
    report['status']='complete';save_json(out/'benchmark.json',report);save_json(out/'progress.json',{'status':'complete'})
    print(json.dumps({'speedup':report['speedup'],'book_agreement':report['book_agreement'],'gates':report['gates'],
        'profiles':{p:{k:v for k,v in value.items() if k not in ['timed_frames','setup']} for p,value in report['profiles'].items()}},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--python',required=True);parser.add_argument('--weights-dir',required=True)
    parser.add_argument('--frames',type=int,required=True);parser.add_argument('--visual-inventory');parser.add_argument('--panorama')
    parser.add_argument('--cutoff',default='2026-10-06T19:45:00+02:00')
    args=parser.parse_args()
    if not 1<=args.frames<=97:parser.error('--frames must be 1..97')
    if bool(args.visual_inventory)!=bool(args.panorama):parser.error('Visual inventory and panorama must be supplied together')
    cutoff=datetime.fromisoformat(args.cutoff)
    if cutoff.tzinfo is None:parser.error('Cutoff requires timezone')
    remaining=cutoff.timestamp()-datetime.now(ZoneInfo('Europe/Madrid')).timestamp()
    if remaining<=0:parser.error('Heavy-job cutoff has passed')
    try:asyncio.run(asyncio.wait_for(run(args),min(900,remaining)))
    except BaseException as exc:
        out=Path(args.out)
        # Do not clobber a previously completed output if mkdir rejected it.
        if out.exists() and (out/'progress.json').is_file():
            progress=json.loads((out/'progress.json').read_text())
            if progress.get('status')!='complete':
                save_json(out/'progress.json',{**progress,'status':'partial','failure_type':type(exc).__name__})
        raise


if __name__=='__main__':main()
