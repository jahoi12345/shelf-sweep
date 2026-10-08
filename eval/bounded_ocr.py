"""Fresh bounded native-crop OCR, selected by frame coverage/geometry before OCR.

Selection is diagnostic, not a human-legible or representative language sample.
Existing candidates/frames are never modified. New output names are mandatory.
"""
import argparse, json
from pathlib import Path
from pipeline.read_spines import read, load, records
from pipeline.resolve import resolve
from eval.debug_index import sha


def select(observations, limit):
    by_frame={}
    for o in observations:
        if o.get('kind')=='book': by_frame.setdefault(o['frame_id'],[]).append(o)
    frames=list(by_frame); n=min(len(frames),max(1,(limit+1)//2))
    chosen=[frames[round(i*(len(frames)-1)/max(1,n-1))] for i in range(n)] if frames else []
    selected=[]
    def area(o):
        a,b,c,d=o['bbox_xyxy'];return (c-a)*(d-b)
    for fid in chosen:
        books=sorted(by_frame[fid],key=lambda o:(-area(o),o['observation_id']))
        first=books[0]; selected.append(first)
        second=next((o for o in books[1:] if o.get('orientation')!=first.get('orientation')), None)
        if second is None and len(books)>1: second=books[1]
        if second: selected.append(second)
    return selected[:limit]


def execute(run_dir,out,limit=24,runtime_dir='.runtime/work/bounded_ocr',suffix='panorama_v1'):
    root=Path(run_dir).resolve(); dest=Path(out).resolve()
    paths=[root/('readings_'+suffix+'.json'),root/('resolutions_'+suffix+'.json')]
    if dest.exists() or any(p.exists() for p in paths): raise ValueError('Refusing to overwrite published OCR artifacts')
    m=root/'manifest.json'; c=root/'candidates.json'; before={str(p):sha(p) for p in (m,c)}
    observations=select(records(load(c),'observations'),limit)
    dest.mkdir(parents=True)
    selection={'source_run':str(root),'run_id':load(c)['run_id'],'candidate_sha256':sha(c),'manifest_sha256':sha(m),
               'selection_policy':'Uniformly spaced candidate-bearing frames; largest proposed book bbox, then largest different orientation (else second largest). Before OCR; no human/title/language inputs.',
               'human_truth':False,'observations':observations}
    (dest/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    reading=read(c,m,root,{'provider':'local_vision','ocr_runtime_dir':str(Path(runtime_dir).resolve()),
                         'ocr_evidence_dir':str(dest/'ocr_responses'),'attempt_observation_ids':[o['observation_id'] for o in observations],
                         'max_books':limit,'rectify_polygons':True,'crop_subdir':'bounded_'+suffix+'_crops',
                         'readings_filename':paths[0].name})
    resolution=resolve(reading,root,{'resolutions_filename':paths[1].name})
    if before!={str(p):sha(p) for p in (m,c)}: raise ValueError('Inputs changed during OCR; do not publish joined evidence')
    a=load(reading);rows=records(a,'readings'); attempted=[r for r in rows if r.get('ocr_attempt')!='not_attempted']
    summary={'readings':str(reading),'readings_sha256':sha(reading),'resolutions':str(resolution),'resolutions_sha256':sha(resolution),
             'source_inputs':before,'selection_count':len(observations),'attempted':len(attempted),
             'attempted_frames':len({r['frame_id'] for r in attempted}),
             'nonempty_literal_observations':sum(bool(r.get('visible_text')) for r in attempted),
             'empty_recognition':sum(r.get('ocr_attempt')=='completed' and not r.get('visible_text') for r in attempted),
             'failed_attempts':sum(r.get('ocr_attempt')=='failed' for r in attempted),'metrics':a['metrics'],
             'accepted_work_identities':load(resolution)['metrics']['accepted'],
             'human_accuracy':None,'human_accuracy_reason':'No attributable frozen human labels; non-empty OCR is not accuracy'}
    (dest/'execution.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--limit',type=int,default=24);p.add_argument('--runtime-dir',default='.runtime/work/bounded_ocr');p.add_argument('--suffix',default='panorama_v1')
    a=p.parse_args();s=execute(**vars(a));print(json.dumps({k:v for k,v in s.items() if k!='metrics'}))


if __name__=='__main__':main()
