"""Native review overlays and evidence diagnostics, never human ground truth."""
from __future__ import annotations
import argparse
import html
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from pipeline.detect import read_json,save_json,overlap
from pipeline.dedup import polygon_iou


def diagnostics(run_dir,out_dir):
    root=Path(run_dir).resolve();out=Path(out_dir).resolve()
    if out.exists():raise ValueError('Choose a new diagnostic version')
    out.mkdir(parents=True)
    c=read_json(root/'candidates.json');m=read_json(root/'manifest.json');inv=read_json(root/'inventory.json')
    observations=c['data']['observations'];by_id={o['observation_id']:o for o in observations}
    by_frame=defaultdict(list)
    for o in observations:by_frame[o['frame_id']].append(o)
    groups={o:book['instance_id'] for book in inv['data']['books']+inv['data']['items'] for o in book['observation_ids']}
    frames=[f for f in m['data']['frames'] if f.get('selected')];canvas=np.full(((len(frames)+2)//3*270,3*480,3),245,np.uint8)
    cards=[];same_view=[]
    for i,f in enumerate(frames):
        image=cv2.imread(str(root/f['image_path']))
        if image is None:raise ValueError('Missing native evidence')
        for o in by_frame[f['frame_id']]:
            color={'book':(40,210,40),'non_book':(220,40,220),'ambiguous':(0,150,255)}[o['kind']]
            x1,y1,x2,y2=o['bbox_xyxy']
            if 'polygon_xy' in o:cv2.polylines(image,[np.array(o['polygon_xy'],np.int32)],True,color,2)
            else:cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
            cv2.putText(image,o['observation_id'].split('-')[-1]+' '+groups.get(o['observation_id'],'?'),(x1,max(13,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1)
        cv2.putText(image,f['frame_id']+' '+str(f['timestamp_s'])+'s: proposals, not verified counts',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3)
        filename=f['frame_id']+'.jpg';cv2.imwrite(str(out/filename),image)
        thumb=cv2.resize(image,(480,270));y=i//3*270;x=i%3*480;canvas[y:y+270,x:x+480]=thumb
        cards.append('<article><h2>'+html.escape(f['frame_id'])+' '+str(f['timestamp_s'])+'s</h2><a href="'+filename+'"><img src="'+filename+'" alt="Native proposal overlay"></a><p><a href="'+html.escape((root/f['image_path']).as_uri())+'">Original unchanged evidence</a></p></article>')
        local=by_frame[f['frame_id']]
        for j,a in enumerate(local):
            for b in local[j+1:]:
                bbox=overlap(a['bbox_xyxy'],b['bbox_xyxy'])
                if bbox<=.55:continue
                poly=polygon_iou(a['polygon_xy'],b['polygon_xy']) if len(a.get('polygon_xy',[]))==len(b.get('polygon_xy',[]))==4 else None
                same_view.append({'observation_ids':[a['observation_id'],b['observation_id']],'frame_id':f['frame_id'],'kinds':[a['kind'],b['kind']],
                    'categories':[a['category'],b['category']],'bbox_iou':bbox,'polygon_iou':poly,
                    'interpretation':'Loose boxes overlap but actual polygons largely disjoint' if poly is not None and poly<.1 else 'Potential duplicate, merged-spine or category disagreement; review required'})
    cv2.imwrite(str(out/'pan_contact.jpg'),canvas)
    pairs=[]
    for a,b in zip(frames,frames[1:]):
        reg=next((r for r in inv['data']['registrations'] if r['frame_ids']==[a['frame_id'],b['frame_id']]),None)
        edges=[e for e in inv['data']['associations']+inv['data']['uncertain_associations'] if e['frame_ids']==[a['frame_id'],b['frame_id']]]
        pairs.append({'frame_ids':[a['frame_id'],b['frame_id']],'registration':reg,'accepted_link_proposals':sum(e.get('decision')=='merged' for e in edges),
            'uncertain_link_hypotheses':sum(e.get('decision')=='uncertain' for e in edges),
            'gate_rejections':dict(Counter(g for e in edges if not e['eligible'] for g,failed in [('iou',e['iou']<.55),('center',e['center_distance_spine_widths']>.55),('appearance',e['appearance_score']<.72),('aggregate',e['score']<.7)] if failed))})
    hypotheses=[e for e in inv['data']['uncertain_associations'] if e['iou']>.65 and e['appearance_score']<.72]
    summary={'provenance':'Real saved evidence with algorithmic proposals and agent diagnostics, not reviewed human truth','run_id':m['run_id'],
        'observations_by_kind':dict(Counter(o['kind'] for o in observations)),'proposed_counts':inv['data']['counts'],
        'same_view_overlaps':same_view,'same_view_bbox_overlaps_polygon_lt_point1':sum(e['polygon_iou'] is not None and e['polygon_iou']<.1 for e in same_view),
        'adjacent_pairs':pairs,'spatial_overlap_appearance_rejection_hypotheses':hypotheses,
        'all_uncertain_reasons':dict(Counter(e['reason'] for e in inv['data']['uncertain_associations'])),
        'metrics':{'physical_count_error':None,'precision':None,'recall':None,'clutter_false_positives':None,'missed_books':None,'merge_errors':None,'duplicate_errors':None,
            'reason':'No independent exhaustive frozen human labels; proposal statistics are not accuracy'}}
    save_json(out/'diagnostics.json',summary)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Detection evidence review</title><style>body{font-family:system-ui;margin:24px;background:#fafafa}main{display:grid;grid-template-columns:repeat(2,1fr);gap:24px}img{max-width:100%}article{background:white;padding:12px}</style><h1>Native frame proposal review</h1><p>Green book proposals, magenta non-books, orange ambiguous. IDs label provisional groups. No human truth or verified counts. Click each image for full native resolution.</p><a href="pan_contact.jpg">Whole selected-pan contact sheet</a><main>'+''.join(cards)+'</main>')
    return out/'diagnostics.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(diagnostics(a.run_dir,a.out))

if __name__=='__main__':main()
