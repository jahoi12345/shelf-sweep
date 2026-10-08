"""Bounded timing control; versioned output, local models, no hosted calls."""
import argparse,asyncio,json,time,statistics,hashlib
from pathlib import Path
from providers.local_detector_process import LocalDetectorProcess
async def run(args):
 source=Path(args.source).resolve();out=Path(args.out).resolve()
 out.mkdir(exist_ok=False);manifest=json.loads((source/'manifest.json').read_text());frames=manifest['data']['frames'][:60];adapters={};records={p:[] for p in ['baseline',args.profile]}
 if len(frames)!=60:raise ValueError('Exactly 60 source frames required')
 try:
  for p in records:
   a=LocalDetectorProcess(python=args.python,weights_dir=args.weights_dir,work_dir=out/p/'process',profile=p,timeout_s=120);adapters[p]=a;await a.start()
  for i,f in enumerate(frames):
   order=['baseline',args.profile] if i%2==0 else [args.profile,'baseline']
   for p in order:
    path=source/f['image_path'];start=time.perf_counter();r=await adapters[p](image_path=path);elapsed=time.perf_counter()-start
    records[p].append({'frame_id':f['frame_id'],'timestamp_s':f['timestamp_s'],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'wall_s':elapsed,'metrics':r['metrics'],'books':sum(o['kind']=='book' for o in r['data']['objects'])})
   (out/'progress.json').write_text(json.dumps({'status':'running','completed_pairs':i+1,'frames':records},indent=2))
   if (i+1)%10==0:print(f'Completed {i+1}/60 paired frames',flush=True)
  result={'status':'complete','source_run':str(source),'method':'Alternate order each frame after both persistent workers warm up; same 60 original frames; detection only, no human recall assertion','profiles':{p:{'mean_s_per_frame':statistics.mean(r['wall_s'] for r in rows),'median_s_per_frame':statistics.median(r['wall_s'] for r in rows),'book_observations':sum(r['books'] for r in rows)} for p,rows in records.items()},'frames':records}
  result['speedup']=result['profiles']['baseline']['mean_s_per_frame']/result['profiles'][args.profile]['mean_s_per_frame'];(out/'paired_timing.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))
 except BaseException as exc:
  (out/'progress.json').write_text(json.dumps({'status':'partial','failure_type':type(exc).__name__,'frames':records},indent=2))
  raise
 finally:
  for a in adapters.values():await a.aclose()
def main():
 from providers.local_detector import PROFILES
 p=argparse.ArgumentParser(description='Alternate baseline/experimental inference order on 60 original YT2 frames; comparison to model predictions, not human recall.')
 p.add_argument('--source',required=True);p.add_argument('--out',required=True)
 p.add_argument('--python',required=True);p.add_argument('--weights-dir',required=True)
 p.add_argument('--profile',choices=[x for x in PROFILES if x.startswith('fast_')],default='fast_1280_512')
 args=p.parse_args()
 asyncio.run(asyncio.wait_for(run(args),900))

if __name__=='__main__':main()
