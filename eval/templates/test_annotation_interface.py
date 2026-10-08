"""Synthetic interface checks. No fabricated human labels or OCR quality scores."""
import asyncio,copy,hashlib,json,tempfile,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
from urllib.request import urlopen
import numpy as np,cv2
from providers.local_ocr import reading_payload,rotation_to_crop,LocalOCR
from providers.easyocr_reader import EasyOCRAdapter
from pipeline.read_spines import read_async
from eval.templates.http_bundle import annotation_payload,render_viewer,render_http_viewer,prepare_http_bundle
from pipeline.contracts import envelope

class OCRInterfaces(unittest.TestCase):
    def test_literal_unicode_and_no_semantic_completion(self):
        text='El niño y la niña\nGarcía Márquez'
        payload,rotation,uncertain=reading_payload([{'rotation_deg_clockwise':90,'lines':[{'candidates':[{'text':text,'confidence':.7}]}]}])
        self.assertEqual(payload['visible_text'],text);self.assertEqual(rotation,90)
        for key in ('title_text','author_text','isbn_text','edition_text','language_code'):self.assertIsNone(payload[key])
        self.assertFalse(payload['title_complete']);self.assertIsNone(payload['read_confidence'])
    def test_empty_low_confidence_and_unexpected_scripts_are_explicit(self):
        p,_,_=reading_payload([]);self.assertIsNone(p['visible_text'])
        p,_,uncertain=reading_payload([{'rotation_deg_clockwise':0,'lines':[{'candidates':[{'text':'unknown','confidence':.1}]}]}])
        self.assertEqual(p['visible_text'],'[?]');self.assertEqual(uncertain,[0])
        p,_,_=reading_payload([{'rotation_deg_clockwise':0,'lines':[{'candidates':[{'text':'PESИT','confidence':.8}]}]}])
        self.assertIn('unexpected_script_for_es_en_request_raw_text_unverified',p['issues'])
    def test_rotation_inverse_maps_every_native_corner(self):
        image=np.arange(4*7).reshape(4,7)
        for degrees,flag in [(0,None),(90,cv2.ROTATE_90_CLOCKWISE),(180,cv2.ROTATE_180),(270,cv2.ROTATE_90_COUNTERCLOCKWISE)]:
            rot=image if flag is None else cv2.rotate(image.astype('uint8'),flag)
            matrix=np.array(rotation_to_crop(degrees,7,4))
            for y,x in [(0,0),(0,rot.shape[1]-1),(rot.shape[0]-1,0),(rot.shape[0]-1,rot.shape[1]-1)]:
                px,py,_=matrix@np.array([x,y,1]);self.assertEqual(rot[y,x],image[int(py),int(px)])
    def test_both_adapters_reject_detector_tasks_before_inference(self):
        for cls in (LocalOCR,EasyOCRAdapter):
            obj=object.__new__(cls)
            with self.assertRaises(ValueError):asyncio.run(obj(task='detect',image_paths=['unused']))
    def test_stage_preserves_raw_ocr_provenance_cost_and_caller_context(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'frames').mkdir();cv2.imwrite(str(root/'frames/f00000.png'),np.zeros((30,40,3),np.uint8))
            m=envelope('test','frames',{'frames':[{'frame_id':'f00000','image_path':'frames/f00000.png','width_px':40,'height_px':30,'timestamp_s':0}]})
            c=envelope('test','detect',{'observations':[{'observation_id':'f00000-o000','frame_id':'f00000','bbox_xyxy':[0,0,20,30],'kind':'book'}]})
            for n,a in [('manifest',m),('candidates',c)]:(root/(n+'.json')).write_text(json.dumps(a))
            async def fake(**kwargs):
                self.assertEqual(kwargs['task'],'read');self.assertEqual(kwargs['config']['session_revision'],5)
                return {'data':{'visible_text':'El niño','title_complete':False,'issues':[]},'request_ref':'raw.json',
                        'ocr_evidence':{'path':'raw.json','selected_rotation_deg_clockwise':90},
                        'metrics':{'provider':'local_test','request_count':0,'actual_cost':0,'local_inference_count':4}}
            out=asyncio.run(read_async(root/'candidates.json',root/'manifest.json',root,{'read':{'image_request':fake},'source_turn_id':'t1','session_revision':5}))
            a=json.loads(out.read_text());r=a['data']['readings'][0]
            self.assertEqual(r['visible_text'],'El niño');self.assertEqual(r['provenance']['session_revision'],5)
            self.assertEqual(r['ocr_evidence']['selected_rotation_deg_clockwise'],90)
            self.assertEqual(a['metrics']['actual_cost'],0);self.assertEqual(a['metrics']['local_inference_count'],4)

class HTTPAnnotation(unittest.TestCase):
    def test_prediction_free_bundle_serves_native_bytes_without_review_approval(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir();(run/'frames').mkdir();image=run/'frames/f00000.png';cv2.imwrite(str(image),np.zeros((12,16,3),np.uint8))
            m=envelope('test','frames',{'frames':[{'frame_id':'f00000','image_path':'frames/f00000.png','width_px':16,'height_px':12,'timestamp_s':0}]})
            (run/'manifest.json').write_text(json.dumps(m))
            bundle=root/'http';prepare_http_bundle(run/'manifest.json',run,bundle)
            a=json.loads((bundle/'annotations_pending.json').read_text());self.assertEqual(a['frames'][0]['objects'],[])
            self.assertIsNone(a['provenance']['frozen_at']);self.assertEqual(a['provenance']['review_status'],'pending')
            class Quiet(SimpleHTTPRequestHandler):
                def log_message(self,*args):pass
            server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(bundle)));t=threading.Thread(target=server.serve_forever);t.start()
            try:
                base='http://127.0.0.1:'+str(server.server_port)
                self.assertNotIn(b'file://',urlopen(base+'/index.html').read())
                self.assertIn(b'annotation_payload.json',urlopen(base+'/index.html').read())
                served=json.loads(urlopen(base+'/annotation_payload.json').read())
                self.assertEqual(served['annotations']['frames'][0]['objects'],[])
                self.assertEqual(hashlib.sha256(urlopen(base+'/evidence/f00000.png').read()).hexdigest(),hashlib.sha256(image.read_bytes()).hexdigest())
            finally:server.shutdown();server.server_close();t.join()
            with self.assertRaises(ValueError):prepare_http_bundle(run/'manifest.json',run,bundle)
    def test_http_viewer_rejects_file_urls_and_escapes_embedded_payload(self):
        with self.assertRaises(ValueError):render_viewer({'images':{'x':'file:///private/file'},'annotations':{}})
        html=render_viewer({'images':{'x':'evidence/x.png'},'annotations':{'notes':'</script>'}})
        self.assertIn('\\u003c/script>',html)

if __name__=='__main__':unittest.main()
