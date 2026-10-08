"""Read-only join for live run artifacts, including incremental reading slices.

A reading's candidate hash can name a single-group slice rather than the final
candidate file. Verify that slice and its exact native observation before joining.
This adapter never resolves identities or creates panorama coordinates.
"""
from pathlib import Path
import copy
from .debug_views import digest, load


def adapt_readings(root, candidate, manifest, add):
    root=Path(root).resolve();run=candidate.get('run_id');candidate_hash=digest(root/'candidates.json')
    observations={o['observation_id']:o for o in candidate.get('data',{}).get('observations',[])}
    frames={f['frame_id']:f for f in manifest.get('data',{}).get('frames',[])}
    slices={}
    paths=sorted((root/'live/read').glob('*.candidates.json'))[:2000]
    for p in paths:
        if not p.resolve().is_relative_to(root): continue
        a=load(p)
        if a and a.get('run_id')==run:
            slices[digest(p)]={o['observation_id']:o for o in a.get('data',{}).get('observations',[])}
    rows={};provenance=[]
    for name,label in [('readings.json','primary'),('readings_second.json','second')]:
        p=root/name;a=load(p)
        if not a:continue
        provenance.append({'artifact':name,'sha256':digest(p),'run_id':a.get('run_id'),'reader_pass':label})
        for r in a.get('data',{}).get('readings',[])[:5000]:
            oid=r.get('observation_id');o=observations.get(oid);f=frames.get(r.get('frame_id'));pv=r.get('provenance',{})
            errors=[]
            if not o or not f or a.get('run_id')!=run or pv.get('run_id')!=run:errors.append('run/observation/frame mismatch')
            binding=pv.get('candidate_artifact_sha256')
            if binding!=candidate_hash and (not o or slices.get(binding,{}).get(oid)!=o):errors.append('candidate slice hash or exact observation mismatch')
            if o and (r.get('source_polygon_xy')!=o.get('polygon_xy') or r.get('crop_transform',{}).get('bbox_xyxy')!=o.get('bbox_xyxy')):errors.append('native geometry mismatch')
            if f and (r.get('timestamp_s')!=f.get('timestamp_s') or r.get('frame_image_path')!=f.get('image_path')):errors.append('source timestamp/path mismatch')
            frame_path=(root/(f or {}).get('image_path','')).resolve()
            if not frame_path.is_relative_to(root) or not frame_path.is_file() or digest(frame_path)!=pv.get('frame_sha256'):errors.append('native frame hash mismatch')
            crop_path=(root/r.get('crop_path','')).resolve()
            crop_token=add(root,r.get('crop_path'),pv.get('crop_sha256')) if pv.get('crop_sha256') else None
            if not crop_token:errors.append('crop hash missing or mismatch')
            raw_ref=r.get('ocr_evidence') or {};raw_path=Path(raw_ref.get('path',''))
            if not raw_path.is_absolute():raw_path=root/raw_path
            raw_path=raw_path.resolve();raw=None
            if raw_ref:
                if raw_path.is_relative_to(root) and raw_path.is_file() and digest(raw_path)==raw_ref.get('sha256'):
                    raw=load(raw_path)
                    if raw.get('input_sha256')!=pv.get('crop_sha256'):errors.append('raw OCR input crop mismatch')
                else:errors.append('raw OCR hash/path mismatch')
            result={'reader_pass':label,'reading_join':'rejected_mismatch' if errors else 'verified_run_slice_frame_crop_geometry','ocr_state':'unavailable' if errors else ('not_attempted' if r.get('ocr_attempt')!='completed' else ('uncertain_transcription' if r.get('visible_text') else 'empty_recognition')),
                    'literal_text':None if errors else r.get('visible_text'),'reading':None if errors else copy.deepcopy(r),'raw_ocr':None if errors else raw,'crop_asset':None if errors else crop_token,
                    'supported_work':None,'supported_edition':None,'work_state':'unresolved_work','edition_state':'unsupported_physical_edition',
                    'review_reasons':errors+r.get('issues',[])+['Literal OCR unverified; no identity inferred by the debugger'], 'candidate_binding':'full candidate hash' if binding==candidate_hash else 'verified incremental slice and exact observation'}
            previous=rows.get(oid)
            if previous:previous.setdefault('reader_passes',[copy.deepcopy(previous)]).append(result)
            else:rows[oid]=result
    return rows,provenance
