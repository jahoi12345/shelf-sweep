"""Compare exact saved YT2 frames; counts/IoU agreement are not human recall."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import time

from pipeline import contracts as C
from pipeline.detect import _detect, read_json, save_json, resolve_evidence
from pipeline.dedup import merge, polygon_iou
from providers.local_detector_process import LocalDetectorProcess
from providers.local_detector import PROFILES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def agreement(reference, proposed):
    """One-to-one native polygon agreement with the frozen baseline predictions."""
    baseline = [o for o in reference if o['kind'] == 'book']
    candidate = [o for o in proposed if o['kind'] == 'book']
    edges=[]
    for a in baseline:
        for b in candidate:
            if a['frame_id'] != b['frame_id']:continue
            iou=polygon_iou(a['polygon_xy'],b['polygon_xy'])
            if iou>=.5:edges.append((iou,a['observation_id'],b['observation_id']))
    left=set();right=set()
    for iou,a,b in sorted(edges,reverse=True):
        if a not in left and b not in right:left.add(a);right.add(b)
    return {'baseline_books':len(baseline),'proposed_books':len(candidate),'matched_at_iou_05':len(left),
            'baseline_prediction_recall_proxy':len(left)/len(baseline) if baseline else None,
            'observation_count_ratio':len(candidate)/len(baseline) if baseline else None,
            'note':'Agreement with model predictions only; no human recall/physical-count assertion'}


async def run(args):
    source=Path(args.source).resolve();root=Path(args.out).resolve()
    if root.exists():raise ValueError('Refusing to overwrite benchmark version')
    original=read_json(source/'manifest.json');frozen=read_json(source/'candidates.json')
    errors=C.validate_manifest(original,source)
    if errors:raise ValueError('Invalid source manifest: '+'; '.join(errors[:3]))
    frames=original['data']['frames'][:args.frames];ids={f['frame_id'] for f in frames}
    if len(frames)!=args.frames:raise ValueError('Source has fewer requested frames')
    root.mkdir(parents=True)
    manifest=json.loads(json.dumps(original));manifest['data']['frames']=frames
    manifest['data']['sampling'].update(sampled_frames=len(frames),selected_frames=len(frames))
    for f in frames:f['selected']=True
    ref=[o for o in frozen['data']['observations'] if o['frame_id'] in ids]
    result={'status':'running','source_run':str(source),'source_manifest_sha256':sha(source/'manifest.json'),
            'source_candidates_sha256':sha(source/'candidates.json'),'frames':[],
            'scope':'First 60 contiguous YT2 frames at original sample rate; partial view, not entire-room count',
            'association':'track','profiles':{},'actual_hosted_calls':0}
    for f in frames:
        path=resolve_evidence(source,f['image_path'])
        result['frames'].append({'frame_id':f['frame_id'],'timestamp_s':f['timestamp_s'],'sha256':sha(path)})
    for profile in args.profiles.split(','):
        out=root/profile;out.mkdir();m=json.loads(json.dumps(manifest));m['run_id']=out.name
        for f in frames:
            destination=out/f['image_path'];destination.parent.mkdir(exist_ok=True,parents=True)
            shutil.copyfile(resolve_evidence(source,f['image_path']),destination)
            assert sha(destination)==next(r['sha256'] for r in result['frames'] if r['frame_id']==f['frame_id'])
        save_json(out/'manifest.json',m)
        if profile=='saved_baseline':
            for response in {o['response_ref'] for o in ref}:
                destination=out/response;destination.parent.mkdir(exist_ok=True,parents=True)
                shutil.copyfile(resolve_evidence(source,response),destination)
            c=json.loads(json.dumps(frozen));c['run_id']=out.name;c['data']['observations']=ref
            c['data']['frames']=[f for f in c['data']['frames'] if f['frame_id'] in ids]
            c['data']['uncertain_tile_associations']=[]
            save_json(out/'candidates.json',c);timing=None;setup=None
        else:
            adapter=LocalDetectorProcess(python=args.python,weights_dir=args.weights_dir,work_dir=out/'process',profile=profile,timeout_s=180)
            try:
                setup=await adapter.start();start=time.perf_counter()
                await _detect(out/'manifest.json',out,{'image_request':adapter,'model':'local_pretrained'})
                timing=time.perf_counter()-start
            finally:await adapter.aclose()
        c=read_json(out/'candidates.json')
        path=merge(out/'candidates.json',out/'manifest.json',out,{'association':'track'})
        inventory=read_json(path)
        result['profiles'][profile]={'detection_wall_s':timing,'seconds_per_frame':timing/len(frames) if timing is not None else None,
            'setup':setup,'counts':inventory['data']['counts'],'agreement':agreement(ref,c['data']['observations']),
            'books_per_frame':{fid:sum(o['kind']=='book' and o['frame_id']==fid for o in c['data']['observations']) for fid in sorted(ids)}}
        baseline=result['profiles'].get('baseline',{}).get('seconds_per_frame')
        if timing and baseline:result['profiles'][profile]['speedup_vs_fresh_baseline']=baseline/(timing/len(frames))
        save_json(root/'benchmark.json',result)
        print(json.dumps({'profile':profile,**result['profiles'][profile]},default=str),flush=True)
    result['status']='complete';save_json(root/'benchmark.json',result)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--python',required=True);p.add_argument('--weights-dir',required=True)
    p.add_argument('--frames',type=int,default=60)
    p.add_argument('--profiles',default='saved_baseline,baseline,fast_960,fast_1024')
    a=p.parse_args()
    if not 1<=a.frames<=60:p.error('--frames must be 1..60')
    profiles=a.profiles.split(',')
    if len(profiles)!=len(set(profiles)) or any(x not in ['saved_baseline',*PROFILES] for x in profiles):p.error('Invalid or duplicate profile')
    try:code=asyncio.run(run(a))
    except BaseException as exc:
        path=Path(a.out)/'benchmark.json'
        if path.is_file():
            partial=read_json(path);partial.update(status='partial',failure_type=type(exc).__name__)
            save_json(path,partial)
        raise
    raise SystemExit(code)


if __name__=='__main__':main()
