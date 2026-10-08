"""Read-only, exact-run OCR/identity evidence export. No inference or human labels.

Paths are repository-relative allowlisted evidence, not arbitrary serving routes.
The viewer must verify source_run + candidate SHA before joining observation IDs.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from pipeline.read_spines import load, records, inside, native_crop, ordered_quad
from pipeline.resolve import isbn_value, supported_work, edition_language_conflict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def unique(rows, key):
    result={}
    for row in rows:
        if row[key] in result: raise ValueError('Duplicate '+key)
        result[row[key]]=row
    return result


def attempt_state(reading):
    """No attempted recognition means no evidence about human readability."""
    if not reading: return 'not_attempted'
    issues=reading.get('issues',[])
    explicit=reading.get('ocr_attempt')
    if explicit=='not_attempted': return 'not_attempted'
    if explicit=='failed': return 'failed_attempt'
    if explicit!='completed':
        if any(str(i).startswith('reading_failed:') for i in issues): return 'failed_attempt'
        if any(i in issues for i in ('reading_budget_limit','provider_unavailable_crop_only','outside_bounded_ocr_selection')): return 'not_attempted'
        if not reading.get('provenance',{}).get('provider_evidence'): return 'attempt_unknown'
    text=reading.get('visible_text')
    if text is None or text=='': return 'empty_recognition'
    return 'uncertain_transcription'  # Engine output is unreviewed even at a high score.


def export_index(run_dir, out, *, readings=None, resolutions=None, inventory=None, panorama=None, scan_label=None, repository=None):
    root=Path(run_dir).resolve(); repo=Path(repository or Path(__file__).resolve().parents[1]).resolve()
    target=Path(out)
    if target.exists(): raise ValueError('Refusing to overwrite versioned debug index')
    artifacts={}; allowed={}; digest_cache={}
    def reference(path, expected=None):
        p=Path(path).resolve()
        if not p.is_relative_to(repo/'runs') or not p.is_file(): raise ValueError('Missing/out-of-scope evidence: '+str(p))
        key=str(p.relative_to(repo)); value=digest_cache.setdefault(key,sha(p)) if key not in digest_cache else digest_cache[key]
        if expected and expected!=value: raise ValueError('SHA mismatch: '+key)
        ref={'path':key,'sha256':value}
        allowed[key]=ref
        return ref
    for name,path in [('manifest',root/'manifest.json'),('candidates',root/'candidates.json'),
                      ('readings',readings),('resolutions',resolutions),('inventory',inventory),('panorama',panorama)]:
        if path: artifacts[name]=reference(path)
    manifest=load(root/'manifest.json'); candidates=load(root/'candidates.json'); run=candidates['run_id']
    if manifest['run_id']!=run: raise ValueError('Manifest/candidate run mismatch')
    frames=unique(records(manifest,'frames'),'frame_id'); observations=unique(records(candidates,'observations'),'observation_id')
    reading_art=load(readings) if readings else {}; resolution_art=load(resolutions) if resolutions else {}
    rrows=unique(records(reading_art,'readings'),'observation_id')
    srows=unique(records(resolution_art,'resolutions'),'observation_id')
    rejected=[]; rows=[]; frame_refs={}; blank_images={}
    for o in observations.values():
        oid=o['observation_id']; f=frames[o['frame_id']]
        if f['frame_id'] not in frame_refs: frame_refs[f['frame_id']]=reference(inside(root,f['image_path']),f.get('image_sha256'))
        source=frame_refs[f['frame_id']]
        row={'observation_id':oid,'join_key':{'run_id':run,'candidate_sha256':artifacts['candidates']['sha256'],
                'source_video_sha256':manifest.get('data',{}).get('input',{}).get('sha256'),
                'frame_id':f['frame_id'],'frame_sha256':source['sha256'],'observation_id':oid,'observation_sha256':object_sha(o)},
             'detection':o,'source_frame':{**source,'frame_id':f['frame_id'],'timestamp_s':f['timestamp_s'],
                'width_px':f['width_px'],'height_px':f['height_px']},'crop':None,'crop_transform':None,
             'reading_join':'absent','ocr_state':'not_attempted','literal_text':None,'reading':None,
             'raw_ocr':None,'language':{'code':None,'evidence':None,'status':'unknown_not_human_label_or_edition'},
             'semantic_state':'not_attempted','resolution_join':'absent','catalogue':None,
             'work_state':'unresolved_work','edition_state':'unsupported_physical_edition',
             'supported_work':None,'supported_edition':None,'pricing_eligible':False,'review_reasons':[],
             'accepted_identity_evidence':[],'rejected_identity_evidence':[]}
        r=rrows.get(oid)
        if r:
            try:
                p=r.get('provenance',{})
                if reading_art.get('run_id')!=run or p.get('run_id')!=run: raise ValueError('reading_run_mismatch')
                if p.get('candidate_artifact_sha256')!=artifacts['candidates']['sha256']: raise ValueError('reading_candidate_hash_mismatch')
                inputs=reading_art.get('inputs',{})
                if inputs.get('manifest_sha256') and inputs['manifest_sha256']!=artifacts['manifest']['sha256']: raise ValueError('reading_manifest_hash_mismatch')
                if r.get('frame_id')!=f['frame_id'] or p.get('frame_sha256')!=source['sha256']: raise ValueError('reading_source_frame_mismatch')
                if r.get('timestamp_s')!=f['timestamp_s'] or r.get('frame_image_path')!=f['image_path']: raise ValueError('reading_source_metadata_mismatch')
                if r.get('source_polygon_xy')!=o.get('polygon_xy'): raise ValueError('reading_polygon_mismatch')
                t=r.get('crop_transform') or {}; geometry=dict(o)
                if t.get('ordered_source_quad_xy'):
                    q=ordered_quad(o['polygon_xy'])
                    if not np.allclose(q,t['ordered_source_quad_xy'],atol=.0001,rtol=0): raise ValueError('reading_rectification_mismatch')
                    geometry['rectify_quad_xy']=q
                shape=(f['height_px'],f['width_px'])
                if shape not in blank_images: blank_images[shape]=np.zeros((*shape,3),np.uint8)
                _,expected=native_crop(blank_images[shape],geometry)
                for key in ('method','bbox_xyxy','rotation_deg_clockwise','crop_width_px','crop_height_px','resampled','upscaled'):
                    if t.get(key)!=expected[key]: raise ValueError('reading_transform_mismatch:'+key)
                if not np.allclose(t.get('crop_to_frame_3x3'),expected['crop_to_frame_3x3'],atol=1e-6,rtol=1e-6): raise ValueError('reading_transform_matrix_mismatch')
                crop=reference(inside(root,r['crop_path']),p['crop_sha256'])
                image=cv2.imread(str(inside(root,r['crop_path'])))
                if image is None or image.shape[:2]!=(t['crop_height_px'],t['crop_width_px']): raise ValueError('reading_crop_dimensions_mismatch')
                raw=None; e=r.get('ocr_evidence')
                if e:
                    ref=reference(e['path'],e['sha256']); rawdata=load(repo/ref['path'])
                    if rawdata.get('input_sha256')!=crop['sha256']: raise ValueError('raw_ocr_crop_hash_mismatch')
                    if Path(rawdata.get('input_path','')).resolve()!=inside(root,r['crop_path']): raise ValueError('raw_ocr_input_path_mismatch')
                    if rawdata.get('data',{}).get('visible_text')!=r.get('visible_text'): raise ValueError('raw_ocr_literal_mismatch')
                    if rawdata.get('selected_rotation_deg_clockwise')!=e.get('selected_rotation_deg_clockwise'): raise ValueError('raw_ocr_rotation_mismatch')
                    from providers.local_ocr import rotation_to_crop
                    selected_rotation=e['selected_rotation_deg_clockwise']
                    if e.get('oriented_pixels_to_native_crop_3x3')!=rotation_to_crop(selected_rotation,t['crop_width_px'],t['crop_height_px']): raise ValueError('selected_rotation_source_mapping_mismatch')
                    if selected_rotation not in [v.get('rotation_deg_clockwise') for v in rawdata.get('variants',[])]: raise ValueError('selected_rotation_not_in_raw_variants')
                    variants=[]
                    for v in rawdata.get('variants',[]):
                        rot=v['rotation_deg_clockwise']; matrix=rotation_to_crop(rot,t['crop_width_px'],t['crop_height_px'])
                        if v.get('oriented_pixels_to_native_crop_3x3')!=matrix: raise ValueError('raw_rotation_source_mapping_mismatch')
                        w,h=(t['crop_height_px'],t['crop_width_px']) if rot in (90,270) else (t['crop_width_px'],t['crop_height_px'])
                        if (v.get('oriented_width_px'),v.get('oriented_height_px'))!=(w,h): raise ValueError('raw_rotation_dimensions_mismatch')
                        variants.append({**v,'oriented_pixels_to_frame_3x3':(np.array(t['crop_to_frame_3x3'])@np.array(matrix)).tolist()})
                    raw={**ref,'variants':variants,'selected_rotation_deg_clockwise':rawdata.get('selected_rotation_deg_clockwise'),
                         'uncertain_line_indices':rawdata.get('uncertain_line_indices'), 'selection_policy':rawdata.get('selection_policy'),
                         'engine_score_note':'Uncalibrated OCR engine score; never identity confidence',
                         'coordinate_mapping':e,'runtime':rawdata.get('runtime')}
                row.update(reading_join='verified_exact_run_frame_crop_geometry', reading=r, crop=crop,crop_transform=t,
                           literal_text=r.get('visible_text'),raw_ocr=raw,ocr_state=attempt_state(r))
                row['join_key']['crop_sha256']=crop['sha256']
                row['language'].update(code=r.get('language_code'),evidence=r.get('language_evidence'),provenance=r.get('language_provenance'))
                row['semantic_state']='not_extracted' if 'semantic_title_author_edition_not_extracted' in r.get('issues',[]) else ('extracted_literal_fields' if r.get('title_text') else 'incomplete_or_unresolved')
                row['review_reasons'].extend(r.get('issues',[]))
            except (ValueError,KeyError,TypeError) as error:
                row.update(reading_join='rejected_mismatch',ocr_state='not_attempted_for_this_verified_geometry')
                row['review_reasons'].append(str(error)); rejected.append({'observation_id':oid,'artifact':'readings','reason':str(error),'untrusted_record':r})
        else: row['review_reasons'].append('No OCR record for this exact run/geometry')
        row['ocr_applicability']='proposed_book' if o.get('kind')=='book' else 'not_in_book_reader_scope'
        if o.get('kind')!='book':
            row.update(work_state='not_assessed_kind_'+str(o.get('kind')),edition_state='not_assessed_kind_'+str(o.get('kind')))
            row['review_reasons'].append('Book OCR was not requested for this non-book/ambiguous classification')
        s=srows.get(oid)
        if s:
            binding=resolution_art.get('inputs',{}).get('readings_sha256')
            if (row['reading_join'].startswith('verified') and resolution_art.get('run_id')==run
                and s.get('provenance')==r.get('provenance') and s.get('visible_text')==r.get('visible_text')
                and (not binding or binding==artifacts.get('readings',{}).get('sha256'))):
                row.update(catalogue=s,resolution_join='verified_reading_hash' if binding else 'verified_exact_row_provenance_legacy_artifact')
                row['review_reasons'].extend(s.get('issues',[]))
                accepted=s.get('decision')=='accepted'
                literal_title=r.get('title_text'); literal_author=r.get('author_text'); visible=r.get('visible_text') or ''
                valid_candidates=[]
                for candidate in s.get('candidates',[]):
                    doc={'key':candidate.get('work') or '', 'title':candidate.get('title'),'author_name':candidate.get('authors',[]),
                         'editions':{'docs':[{'key':e.get('edition_key'),'title':e.get('title'),'language':e.get('language') or []}
                                            for e in candidate.get('title_evidence',[]) if e.get('edition_key')]}}
                    if candidate.get('visible_text_support') and supported_work(r,doc): valid_candidates.append(candidate)
                    else: row['rejected_identity_evidence'].append({'candidate':candidate,'reason':'Native complete literal title/author/language gate unsupported','artifact':artifacts.get('resolutions')})
                supported=(accepted and r.get('title_complete') is True and literal_title and literal_title in visible
                           and (not literal_author or literal_author in visible) and s.get('author')==literal_author and s.get('title')==literal_title
                           and any(c.get('work')==s.get('work') for c in valid_candidates)
                           and any(e.get('field')=='title' and e.get('visible')==literal_title for e in s.get('matching_evidence',[])))
                if supported:
                    row.update(work_state='evidence_supported_work',supported_work={'work':s['work'],'title':s['title'],'author':s.get('author')})
                    row['accepted_identity_evidence']=[{'evidence':e,'artifact':artifacts.get('resolutions')} for e in s.get('matching_evidence',[]) if e.get('field') in {'title','author'}]
                    isbn=isbn_value(r.get('isbn_text'))
                    exact=(isbn and s.get('isbn')==isbn and s.get('edition') and not edition_language_conflict(r,{'languages':s.get('edition_languages',[])})
                           and 'printed_isbn_not_verified_by_specific_edition' not in s.get('issues',[])
                           and any(e.get('field')=='edition_isbn' and e.get('visible')==r.get('isbn_text') and e.get('catalogue')==isbn and e.get('edition_key')==s['edition']
                                   and e.get('rule')=='printed_checksum_valid_isbn_exact_edition_work_title' for e in s.get('matching_evidence',[])))
                    if exact:
                        row.update(edition_state='evidence_supported_exact_isbn_edition', supported_edition={'isbn':isbn,'edition':s['edition']},pricing_eligible=True)
                        row['accepted_identity_evidence'].extend({'evidence':e,'artifact':artifacts.get('resolutions')} for e in s.get('matching_evidence',[]) if e.get('field')=='edition_isbn')
                    elif s.get('edition') or s.get('isbn'): row['rejected_identity_evidence'].append({'reason':'Edition lacks exact verified printed ISBN or conflicts with visible language','proposed_edition':s.get('edition'),'proposed_isbn':s.get('isbn'),'artifact':artifacts.get('resolutions')})
                elif accepted: row['review_reasons'].append('Accepted catalogue identity rejected by debug visible-evidence gate')
            else:
                row['resolution_join']='rejected_mismatch'; row['review_reasons'].append('Resolution not bound to this exact reading')
                rejected.append({'observation_id':oid,'artifact':'resolutions','reason':'resolution_reading_provenance_mismatch','untrusted_record':s})
        if row['ocr_state']=='not_attempted': row['review_reasons'].append('Unattempted OCR is not evidence that a human cannot read this spine')
        if row['edition_state']=='unsupported_physical_edition': row['review_reasons'].append('Physical edition/ISBN unsupported; excluded from exact-edition pricing')
        rows.append(row)
    for kind,data in [('readings',rrows),('resolutions',srows)]:
        rejected.extend({'observation_id':k,'artifact':kind,'reason':'observation_absent_in_target_candidates','untrusted_record':v} for k,v in data.items() if k not in observations)
    groups=[]
    if inventory:
        inv=load(inventory); binding_errors=[]; binding='legacy_geometry_checked_not_candidate_hash_bound'
        provenance_path=Path(inventory).with_name('inventory_provenance.json')
        if provenance_path.exists():
            artifacts['inventory_provenance']=reference(provenance_path); ip=load(provenance_path)
            for field,value in [('source_candidate_run_id',run),('source_candidate_sha256',artifacts['candidates']['sha256']),
                                ('source_manifest_sha256',artifacts['manifest']['sha256']),('inventory_sha256',artifacts['inventory']['sha256'])]:
                if ip.get(field)!=value: binding_errors.append('inventory_provenance_mismatch:'+field)
            binding='verified_candidate_manifest_inventory_hashes' if not binding_errors else 'rejected_hash_mismatch'
        row_map={x['observation_id']:x for x in rows}
        for g in records(inv,'books')+records(inv,'items'):
            members=g.get('observation_ids',[]); errors=list(binding_errors)
            if inv.get('run_id')!=run: errors.append('inventory_run_mismatch')
            if not members or any(m not in observations for m in members): errors.append('inventory_member_missing')
            refs=g.get('evidence_refs',[])
            if set(e.get('observation_id') for e in refs)!=set(members): errors.append('inventory_member_evidence_incomplete')
            for e in refs:
                o=observations.get(e.get('observation_id'),{})
                if any(e.get(k)!=o.get(k) for k in ('frame_id','bbox_xyxy')): errors.append('inventory_member_geometry_mismatch')
                f=frames.get(o.get('frame_id'),{})
                if e.get('timestamp_s')!=f.get('timestamp_s') or e.get('image_path')!=f.get('image_path'): errors.append('inventory_member_source_metadata_mismatch')
            groups.append({'instance_id':g.get('instance_id',g.get('item_id')),'status':'rejected_mismatch' if errors else 'proposed_unverified_physical_group',
                           'observation_ids':members,'join_run_id':run,'candidate_sha256':artifacts['candidates']['sha256'],
                           'binding':binding, 'association_evidence':g,
                           'member_readings':[{'observation_id':m,'ocr_state':row_map[m]['ocr_state'],'literal_text':row_map[m]['literal_text'],
                                               'work_state':row_map[m]['work_state'],'edition_state':row_map[m]['edition_state'],
                                               'review_reasons':row_map[m]['review_reasons']} for m in members if m in row_map],
                           'review_reasons':errors+g.get('issues',[]),'identity_policy':'Inspect member resolutions; group is not human truth or automatic identity evidence'})
    panorama_binding={'status':'not_supplied','accepted_occurrences':0,'rejected_occurrences':0}
    if panorama:
        pano=load(panorama); errors=[]
        for field,value in [('source_candidate_run_id',run),('source_candidates_sha256',artifacts['candidates']['sha256']),
                            ('source_manifest_sha256',artifacts['manifest']['sha256']),
                            ('source_video_sha256',manifest.get('data',{}).get('input',{}).get('sha256'))]:
            if pano.get(field)!=value: errors.append('panorama_source_mismatch:'+field)
        panorama_binding.update(status='rejected_mismatch' if errors else 'verified_exact_source_hashes',reasons=errors)
        row_map={x['observation_id']:x for x in rows}
        for component in pano.get('components',[]):
            source_frames={f['frame_id']:f for f in component.get('source_frames',[])}
            for overlay in component.get('overlays',[]):
                row=row_map.get(overlay.get('observation_id')); reasons=list(errors)
                if not row: reasons.append('panorama_observation_absent')
                else:
                    key=row['join_key']
                    original=observations[key['observation_id']]
                    if any(overlay.get(k)!=v for k,v in original.items()): reasons.append('panorama_native_observation_geometry_or_fields_mismatch')
                    poly=original.get('polygon_xy')
                    if not poly:
                        x0,y0,x1,y1=original['bbox_xyxy'];poly=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
                    if overlay.get('native_polygon_xy')!=poly: reasons.append('panorama_native_polygon_mismatch')
                    for field,value in [('source_candidate_run_id',run),('source_candidates_sha256',key['candidate_sha256']),
                                        ('observation_sha256',key['observation_sha256']),('frame_sha256',key['frame_sha256']),
                                        ('timestamp_s',row['source_frame']['timestamp_s']),('frame_id',key['frame_id'])]:
                        if overlay.get(field)!=value: reasons.append('panorama_overlay_mismatch:'+field)
                    sf=source_frames.get(key['frame_id'],{})
                    if sf.get('sha256')!=key['frame_sha256']: reasons.append('panorama_transform_source_frame_mismatch')
                    if any(sf.get(k)!=row['source_frame'].get(k) for k in ('timestamp_s','width_px','height_px')): reasons.append('panorama_transform_source_metadata_mismatch')
                if reasons:
                    panorama_binding['rejected_occurrences']+=1
                    rejected.append({'artifact':'panorama','observation_id':overlay.get('observation_id'),'component_id':component['component_id'],'reasons':reasons})
                else:
                    panorama_binding['accepted_occurrences']+=1
                    row.setdefault('panorama_occurrences',[]).append({'component_id':component['component_id'],'plane_id':component.get('plane_id'),
                        'native_polygon_xy':overlay.get('native_polygon_xy'),'panorama_polygon_xy':overlay.get('panorama_polygon_xy'),
                        'original_to_panorama_3x3':sf.get('original_to_panorama_3x3'),'proposed_instance_id':overlay.get('proposed_instance_id'),
                        'provenance':artifacts['panorama'],'coordinate_system':pano.get('coordinate_system')})
    input_video=manifest.get('data',{}).get('input',{})
    result={'debug_index_version':1,'fixture':bool(manifest.get('fixture') or candidates.get('fixture')),'scan_label':scan_label,'run_id':run,'source_run':str(root.relative_to(repo)),
            'source_video_sha256':input_video.get('sha256'),'input_artifacts':artifacts,'observations':rows,'proposed_groups':groups,
            'rejected_evidence':rejected,'panorama_binding':panorama_binding,'evidence_allowlist':list(allowed.values()),'usage':reading_art.get('metrics',{}),
            'summary':{'observations':len(rows),'ocr_states':dict(Counter(x['ocr_state'] for x in rows)),
                       'reading_joins':dict(Counter(x['reading_join'] for x in rows)), 'rejected_records':len(rejected),
                       'supported_works':sum(x['supported_work'] is not None for x in rows),'supported_editions':sum(x['supported_edition'] is not None for x in rows)},
            'evaluation':{'human_labels':'pending','accuracy':None,'es':{'sample_size':None,'accuracy':None,'abstentions':None,'confident_errors':None},
                          'en':{'sample_size':None,'accuracy':None,'abstentions':None,'confident_errors':None}},
            'limits':['No human accuracy inferred from OCR yield','Same-room scans are not independent scene splits',
                      'Engine scores are uncalibrated; observed text language is not physical edition language',
                      'No panoramic geometry is created or modified; viewer must verify its own source-run and hashes',
                      'Asset routes must allow only evidence_allowlist entries; paths are not unrestricted filesystem routes'],
            'market':{'country':'US','currency':'USD','pricing_requires_supported_physical_edition':True}}
    books=[x for x in rows if x['detection'].get('kind')=='book']
    result['summary']['book_ocr_coverage']={'proposed_book_observations':len(books),'crops_verified':sum(x['crop'] is not None for x in books),
            'attempted':sum(x['ocr_state'] in {'failed_attempt','empty_recognition','uncertain_transcription'} for x in books),
            'states':dict(Counter(x['ocr_state'] for x in books)),'sampled_native_frames':len(frames),
            'candidate_bearing_frames':len({x['source_frame']['frame_id'] for x in rows}),
            'attempted_frames':len({x['source_frame']['frame_id'] for x in books if x['ocr_state'] in {'failed_attempt','empty_recognition','uncertain_transcription'}})}
    target.parent.mkdir(parents=True,exist_ok=True)
    for ref in artifacts.values():
        if sha(repo/ref['path'])!=ref['sha256']: raise ValueError('Input artifact changed during export: '+ref['path'])
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir');p.add_argument('--out',required=True);p.add_argument('--scan-label');p.add_argument('--preview-of')
    for key in ('readings','resolutions','inventory','panorama'): p.add_argument('--'+key)
    a=p.parse_args();args=vars(a);preview=args.pop('preview_of')
    if preview: result=export_preview(preview,args['out'])
    else:
        if not args['run_dir']:p.error('--run-dir is required unless --preview-of is used')
        result=export_index(**args)
    print(json.dumps(result['summary']))


def export_preview(index_path,out,limit=24):
    """Small early viewer payload. Omitted IDs are unindexed, never unreadable."""
    p=Path(out)
    if p.exists():raise ValueError('Refusing to overwrite debug preview')
    source=load(index_path);rows=source['observations']
    selected=[r for r in rows if r['ocr_state'] in {'uncertain_transcription','empty_recognition','failed_attempt'}][:limit]
    if len(selected)<limit:
        selected += [r for r in rows if r['ocr_state']=='not_attempted'][:limit-len(selected)]
    ids={r['observation_id'] for r in selected}
    result={**source,'observations':selected,'proposed_groups':[g for g in source['proposed_groups'] if ids.intersection(g['observation_ids'])],
            'index_scope':'partial_preview_not_complete_scan','full_index':{'path':str(Path(index_path).resolve()),'sha256':sha(index_path)},
            'omitted_observations':len(rows)-len(selected),'omission_policy':'IDs absent from this preview are unindexed here; consult full_index, do not infer unreadability/no attempt',
            'summary':{**source['summary'],'indexed_preview_observations':len(selected)}}
    # Complete allowlist remains hash-bound; compact JSON avoids duplicated pretty layout.
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    return result


if __name__=='__main__':main()
