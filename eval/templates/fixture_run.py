"""Generate labelled software fixtures and exercise the shared provider/cache adapter.

Synthetic image/boxes/transcripts; no human annotation, no real accuracy or API calls.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.contracts import envelope, write_json
from pipeline.read_spines import read
from pipeline.resolve import resolve
from providers.gemini import get_cache, make_adapter, set_transport, UsageLog


def run(root):
    root=Path(root)
    if root.exists():
        raise ValueError("Choose a new fixture run directory; preserving earlier artifacts")
    root.mkdir(parents=True)
    (root/'frames').mkdir()
    image=np.full((200,600,3),230,np.uint8)
    cv2.rectangle(image,(10,15),(550,90),(20,20,20),-1)
    cv2.putText(image,'FIXTURE ATLAS | A EXAMPLE',(20,65),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2)
    cv2.rectangle(image,(10,110),(550,180),(80,80,80),-1)
    cv2.imwrite(str(root/'frames/f00000.png'),image)
    manifest=envelope(root.name,'frames',{'frames':[dict(frame_id='f00000',timestamp_s=0,image_path='frames/f00000.png',width_px=600,height_px=200,selected=True,selection_reason='synthetic software fixture',quality={})]},fixture=True)
    observations=[dict(observation_id=f'f00000-o{i:03}',frame_id='f00000',kind='book',category='book',orientation='flat',confidence=None,bbox_xyxy=box,evidence_notes='Synthetic labelled software box; never human ground truth') for i,box in enumerate(([10,15,550,90],[10,110,550,180]))]
    candidates=envelope(root.name,'detect',{'observations':observations},fixture=True)
    write_json(root/'manifest.json',manifest);write_json(root/'candidates.json',candidates)
    usage=UsageLog(root/'fixture_provider_usage.jsonl')
    count=0
    async def transport(model,images,prompt,config,schema):
        nonlocal count
        count+=1
        payload={'visible_text':'FIXTURE ATLAS | A EXAMPLE' if count==1 else None,
                 'title_text':'FIXTURE ATLAS' if count==1 else None,'author_text':'A EXAMPLE' if count==1 else None,
                 'publisher_text':None,'title_complete':count==1,'read_confidence':None,
                 'issues':['synthetic_transport_no_model_call']}
        return json.dumps(payload),{'prompt_tokens':0,'output_tokens':0,'thought_tokens':0,'total_tokens':0}
    adapter=make_adapter(run_dir=root,cache=get_cache(root/'cache'),usage=usage,model='synthetic_transport_no_model')
    set_transport(transport)
    try:
        first=read(root/'candidates.json',root/'manifest.json',root,{'image_request':adapter,'readings_filename':'readings_first.json'})
        second=read(root/'candidates.json',root/'manifest.json',root,{'image_request':adapter,'readings_filename':'readings_cached.json'})
        resolutions=resolve(first,root)
        summary={'fixture':True,'provenance':'Synthetic software exercise, not human ground truth or real model inference',
                 'transport_calls':count,'actual_api_requests':0,'actual_cost':0,'cache_summary':usage.summary(),
                 'artifacts':[p.name for p in (first,second,resolutions)]}
        write_json(root/'fixture_summary.json',summary)
        print(json.dumps(summary,indent=2))
    finally:
        set_transport(None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    run(p.parse_args().out)

if __name__=='__main__':main()
