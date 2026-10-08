"""Synthetic contracts only: no generated human approvals, labels or OCR accuracy."""
import asyncio, copy, json, tempfile, unittest
from pathlib import Path
import cv2
import numpy as np
from pipeline.contracts import envelope
from pipeline.read_spines import read_async
from providers.local_ocr import rotation_to_crop
from eval.debug_index import export_index, sha, attempt_state, object_sha
from eval.templates.http_bundle import annotation_payload


class DebugIndex(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.repo=Path(self.temp.name);self.run=self.repo/'runs'/'test';self.run.mkdir(parents=True)
        (self.run/'frames').mkdir();cv2.imwrite(str(self.run/'frames'/'f.png'),np.zeros((30,40,3),np.uint8))
        self.frame={'frame_id':'f','image_path':'frames/f.png','width_px':40,'height_px':30,'timestamp_s':.2}
        self.obs={'observation_id':'same-id','frame_id':'f','bbox_xyxy':[3,5,18,25],'kind':'book'}
        self.manifest=envelope('run-a','frames',{'input':{'video_path':'scan2.mov','sha256':'video-a'},'frames':[self.frame]},fixture=True)
        self.candidates=envelope('run-a','detect',{'observations':[self.obs]},fixture=True)
        self.save('manifest.json',self.manifest);self.save('candidates.json',self.candidates)
        async def fake(**kw):
            crop=kw['image_paths'][0];w,h=15,20;text='El niño y la niña — García Márquez'
            variants=[{'rotation_deg_clockwise':rot,'oriented_width_px':h if rot in (90,270) else w,
                       'oriented_height_px':w if rot in (90,270) else h,
                       'oriented_pixels_to_native_crop_3x3':rotation_to_crop(rot,w,h),
                       'lines':[{'candidates':[{'text':text,'confidence':.8}]}]} for rot in (0,90,180,270)]
            payload={'visible_text':text,'title_complete':False,'issues':['semantic_title_author_edition_not_extracted']}
            raw={'input_path':str(crop),'input_sha256':sha(crop),'data':payload,'variants':variants,'selected_rotation_deg_clockwise':90}
            rp=self.run/'raw.json';rp.write_text(json.dumps(raw,ensure_ascii=False))
            return {'data':payload,'request_ref':str(rp),'ocr_evidence':{'path':str(rp),'sha256':sha(rp),'selected_rotation_deg_clockwise':90,'oriented_pixels_to_native_crop_3x3':rotation_to_crop(90,w,h)},
                    'metrics':{'provider':'local_test','request_count':0,'actual_cost':0,'local_inference_count':4}}
        asyncio.run(read_async(self.run/'candidates.json',self.run/'manifest.json',self.run,{'image_request':fake}))
        self.readings=json.loads((self.run/'readings.json').read_text())
    def tearDown(self):self.temp.cleanup()
    def save(self,name,data):
        p=self.run/name;p.write_text(json.dumps(data,ensure_ascii=False));return p
    def export(self,name='out.json',**kw):
        return export_index(self.run,self.repo/'runs'/name,readings=self.run/'readings.json',repository=self.repo,**kw)
    def test_exact_unicode_rotation_and_evidence_references_readonly(self):
        original=sha(self.run/'readings.json');x=self.export();r=x['observations'][0]
        self.assertTrue(x['fixture']);self.assertEqual(r['literal_text'],'El niño y la niña — García Márquez')
        self.assertEqual(r['ocr_state'],'uncertain_transcription');self.assertEqual(r['semantic_state'],'not_extracted')
        self.assertEqual(r['work_state'],'unresolved_work');self.assertFalse(r['pricing_eligible'])
        v=r['raw_ocr']['variants'][1];matrix=np.array(v['oriented_pixels_to_frame_3x3'])
        self.assertEqual((matrix@np.array([0,0,1])).tolist(),[3,24,1])
        for ref in x['evidence_allowlist']:self.assertEqual(sha(self.repo/ref['path']),ref['sha256'])
        self.assertEqual(original,sha(self.run/'readings.json'))
        with self.assertRaises(ValueError):self.export()
    def test_same_observation_id_other_run_and_stale_candidate_are_rejected(self):
        for i,field,value in [(0,'run_id','run-b'),(1,'candidate_artifact_sha256','stale')]:
            a=copy.deepcopy(self.readings);a['data']['readings'][0]['provenance'][field]=value;self.save('readings.json',a)
            x=self.export('bad'+str(i)+'.json');self.assertEqual(x['observations'][0]['reading_join'],'rejected_mismatch');self.assertIsNone(x['observations'][0]['literal_text'])
    def test_stale_source_and_crop_hashes(self):
        a=copy.deepcopy(self.readings);a['data']['readings'][0]['provenance']['frame_sha256']='stale';self.save('readings.json',a)
        self.assertEqual(self.export()['summary']['rejected_records'],1)
        self.save('readings.json',self.readings);crop=self.run/self.readings['data']['readings'][0]['crop_path'];crop.write_bytes(crop.read_bytes()+b'tamper')
        x=self.export('crop.json');self.assertIn('SHA mismatch',x['observations'][0]['review_reasons'][0])
    def test_stale_manifest_and_geometry_matrix(self):
        a=copy.deepcopy(self.readings);a['inputs']['manifest_sha256']='stale';self.save('readings.json',a)
        self.assertIn('manifest_hash',self.export()['observations'][0]['review_reasons'][0])
        a=copy.deepcopy(self.readings);a['data']['readings'][0]['crop_transform']['crop_to_frame_3x3'][0][2]+=1;self.save('readings.json',a)
        self.assertIn('matrix_mismatch',self.export('geom.json')['observations'][0]['review_reasons'][0])
    def test_raw_tampering_and_source_rotation(self):
        rp=self.run/'raw.json';raw=json.loads(rp.read_text());raw['variants'][1]['oriented_pixels_to_native_crop_3x3'][0][2]=500;rp.write_text(json.dumps(raw))
        self.assertEqual(self.export()['observations'][0]['reading_join'],'rejected_mismatch')
        a=copy.deepcopy(self.readings);a['data']['readings'][0]['ocr_evidence']['sha256']=sha(rp);self.save('readings.json',a)
        x=self.export('rotation.json');self.assertIn('source_mapping_mismatch',x['observations'][0]['review_reasons'][0])
    def test_attempt_states_never_equate_unattempted_with_unreadable(self):
        self.assertEqual(attempt_state(None),'not_attempted')
        self.assertEqual(attempt_state({'ocr_attempt':'not_attempted','issues':['unreadable_spine']}),'not_attempted')
        self.assertEqual(attempt_state({'ocr_attempt':'failed'}),'failed_attempt')
        self.assertEqual(attempt_state({'ocr_attempt':'completed','visible_text':None}),'empty_recognition')
        self.assertEqual(attempt_state({'issues':['provider_unavailable_crop_only']}),'not_attempted')
        self.assertEqual(attempt_state({'visible_text':None}),'attempt_unknown')
    def test_resolution_must_bind_reading_hash_and_provenance(self):
        r=self.readings['data']['readings'][0]
        s=envelope('run-a','resolve',{'resolutions':[{'observation_id':'same-id','visible_text':r['visible_text'],'provenance':r['provenance'],'decision':'no_match','candidates':[],'issues':[]}]},inputs={'readings_sha256':sha(self.run/'readings.json')})
        p=self.save('resolutions.json',s);self.assertEqual(self.export(resolutions=p)['observations'][0]['resolution_join'],'verified_reading_hash')
        s['inputs']['readings_sha256']='stale';self.save('resolutions.json',s)
        self.assertEqual(self.export('resolution.json',resolutions=p)['observations'][0]['resolution_join'],'rejected_mismatch')
    def test_panorama_scoped_native_geometry_join(self):
        ref=sha(self.run/'frames/f.png');poly=[[3,5],[18,5],[18,25],[3,25]]
        sf={**self.frame,'sha256':ref,'original_to_panorama_3x3':np.eye(3).tolist()}
        overlay={**self.obs,'source_candidate_run_id':'run-a','source_candidates_sha256':sha(self.run/'candidates.json'),'observation_sha256':object_sha(self.obs),'frame_sha256':ref,'timestamp_s':.2,'native_polygon_xy':poly,'panorama_polygon_xy':poly}
        p={'source_candidate_run_id':'run-a','source_candidates_sha256':sha(self.run/'candidates.json'),'source_manifest_sha256':sha(self.run/'manifest.json'),'source_video_sha256':'video-a','components':[{'component_id':'c','source_frames':[sf],'overlays':[overlay]}]}
        path=self.save('panorama.json',p);self.assertEqual(self.export(panorama=path)['panorama_binding']['accepted_occurrences'],1)
        p['components'][0]['overlays'][0]['native_polygon_xy']=[[0,0]];self.save('panorama.json',p)
        x=self.export('pano_bad.json',panorama=path);self.assertEqual(x['panorama_binding']['rejected_occurrences'],1)
        self.assertNotIn('panorama_occurrences',x['observations'][0])
    def test_blank_annotation_context_no_invented_approval(self):
        p=annotation_payload(self.run/'manifest.json',self.run,lambda f:'/frames/'+f['frame_id']);a=p['annotations']
        self.assertEqual(a['coverage']['video_path'],'scan2.mov');self.assertEqual(a['source_scan']['video_sha256'],'video-a')
        self.assertEqual(a['frames'][0]['objects'],[]);self.assertIsNone(a['provenance']['frozen_at']);self.assertIsNone(a['provenance']['annotator'])
    def test_supported_native_work_still_cannot_claim_edition_without_isbn(self):
        a=copy.deepcopy(self.readings);r=a['data']['readings'][0]
        r.update(title_text='El niño y la niña',author_text='García Márquez',title_complete=True)
        self.save('readings.json',a)
        candidate={'work':'/works/OL1W','title':'El niño y la niña','authors':['García Márquez'],'visible_text_support':True,'title_evidence':[]}
        record={'observation_id':'same-id','visible_text':r['visible_text'],'provenance':r['provenance'],'title':r['title_text'],'author':r['author_text'],
                'decision':'accepted','work':'/works/OL1W','edition':'/books/OL2M','isbn':None,'candidates':[candidate],
                'matching_evidence':[{'field':'title','visible':r['title_text']}],'issues':[]}
        s=envelope('run-a','resolve',{'resolutions':[record]},inputs={'readings_sha256':sha(self.run/'readings.json')})
        p=self.save('resolutions.json',s);x=self.export(resolutions=p)['observations'][0]
        self.assertEqual(x['work_state'],'evidence_supported_work');self.assertEqual(x['edition_state'],'unsupported_physical_edition');self.assertFalse(x['pricing_eligible'])
        record['title']='The boy and the girl';self.save('resolutions.json',s)
        x=self.export('translation.json',resolutions=p)['observations'][0];self.assertEqual(x['work_state'],'unresolved_work')
        self.assertEqual(x['literal_text'],r['visible_text'])


if __name__=='__main__':unittest.main()
