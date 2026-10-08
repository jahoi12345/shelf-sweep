"""Focused software regressions; synthetic fixtures are not shelf accuracy."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np
from pipeline.dedup import pair_evidence,DEFAULTS,polygon_iou
from providers.local_detector import LocalDetector,fuse_predictions,native_prediction_polygon
from providers.local_detector_process import LocalDetectorProcess


def spine(poly,source='native',tile=(0,0,640,480),ratio=10):
    points=np.array(poly,float);lo=points.min(axis=0);hi=points.max(axis=0)
    return dict(polygon_xy=points.tolist(),bbox_xyxy=[*lo,*hi],kind='book',category='book',
        confidence=.9,orientation='leaning',spine_aspect_ratio=ratio,evidence_notes='SYNTHETIC shape',
        detection_provenance={'pass':source,'source_tile':list(tile)})


class GeometryTests(unittest.TestCase):
    def test_original_pixels_after_rotated_native_tile(self):
        original=np.array([[10,20],[50,20],[50,220],[10,220]],float)
        rotated=np.column_stack([960-original[:,1],original[:,0]])
        native=native_prediction_polygon(rotated,[100,200,960,960],'tile_rotated_clockwise',1920,1080)
        np.testing.assert_array_equal(native,original+[100,200])

    def test_cross_pass_repeat_alias_and_same_frame_copies(self):
        p=[[100,50],[120,50],[120,250],[100,250]]
        a=spine(p);repeat=spine(np.array(p)+[1,0],'tile_native');other=spine(np.array(p)+[25,0])
        merged,suppressed=fuse_predictions([a,repeat,other],[])
        self.assertEqual(len(merged),2);self.assertEqual(len(suppressed),1)
        self.assertEqual(len(merged[0]['detection_aliases']),1)

    def test_loose_leaning_boxes_do_not_merge_disjoint_spines(self):
        p=[[20,20],[30,20],[130,220],[120,220]];q=(np.array(p)+[20,0]).tolist()
        a,b=spine(p),spine(q);self.assertEqual(polygon_iou(p,q),0)
        self.assertEqual(len(fuse_predictions([a,b],[])[0]),2)
        a.update(observation_id='f00000-o000',frame_id='f00000');b.update(observation_id='f00001-o000',frame_id='f00001')
        image=np.full((480,640,3),80,np.uint8)
        bbox=pair_evidence(a,b,np.eye(3),image,image,DEFAULTS)
        polygon=pair_evidence(a,b,np.eye(3),image,image,{**DEFAULTS,'association_geometry':'polygon'})
        self.assertTrue(bbox['eligible']);self.assertFalse(polygon['eligible'])
        self.assertGreater(polygon['bbox_iou'],.55);self.assertEqual(polygon['iou'],0)

    def test_pot_vase_alternative_once_and_plant_separate(self):
        def item(cat):return dict(bbox_xyxy=[100,100,200,200],kind='non_book',category=cat,confidence=.9,evidence_notes='SYNTHETIC')
        objects,suppressed=fuse_predictions([],[item('pot'),item('vase'),item('plant')])
        self.assertEqual(len(objects),2);self.assertEqual(len(suppressed),1)
        self.assertEqual({o['category'] for o in objects},{'pot_or_vase','plant'})
        self.assertEqual(next(o for o in objects if o['category']=='pot_or_vase')['kind'],'ambiguous')

    def test_near_square_shape_abstains_without_using_words(self):
        obj=spine([[0,0],[100,0],[100,70],[0,70]],ratio=100/70)
        objects,_=fuse_predictions([obj],[])
        self.assertEqual(objects[0]['kind'],'ambiguous')
        self.assertEqual(objects[0]['category'],'book_or_decoration')

    def test_bounded_alignment_recovers_two_pixel_registration_residual(self):
        rng=np.random.default_rng(32);a_img=rng.integers(0,255,(480,640,3),dtype=np.uint8)
        b_img=cv2.warpAffine(a_img,np.float32([[1,0,2],[0,1,0]]),(640,480))
        p=[[200,100],[260,100],[260,400],[200,400]]
        a,b=spine(p),spine(np.array(p)+[2,0]);a.update(observation_id='f00000-o000',frame_id='f00000');b.update(observation_id='f00001-o000',frame_id='f00001')
        cfg={**DEFAULTS,'association_geometry':'polygon'}
        direct=pair_evidence(a,b,np.eye(3),a_img,b_img,cfg)
        aligned=pair_evidence(a,b,np.eye(3),a_img,b_img,{**cfg,'appearance_mode':'aligned_polygon'})
        self.assertFalse(direct['eligible']);self.assertTrue(aligned['eligible'])
        self.assertEqual(aligned['alignment_offset_native_px'],[-2,0])
        self.assertEqual(direct['iou'],aligned['iou'])


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_detection_adapters_reject_ocr_stage_before_inference(self):
        for adapter in [LocalDetector.__new__(LocalDetector),LocalDetectorProcess()]:
            with self.assertRaises(ValueError):
                await adapter(image_path='unused',stage='read')

    async def test_venv_symlink_invocation_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base-python';base.write_text('fixture');venv=root/'venv/bin';venv.mkdir(parents=True);link=venv/'python';link.symlink_to(base)
            adapter=LocalDetectorProcess(python=link)
            self.assertEqual(adapter.python,link.absolute());self.assertNotEqual(adapter.python,base)

    async def test_direct_predictors_are_serialized_off_event_loop(self):
        adapter=LocalDetector.__new__(LocalDetector);adapter._inference_lock=threading.Lock()
        state={'active':0,'max':0}
        def fake(image_path):
            state['active']+=1;state['max']=max(state['max'],state['active']);time.sleep(.02);state['active']-=1
            return {'fixture':True,'image_path':image_path}
        adapter._infer_locked=fake
        results=await asyncio.gather(adapter(image_path='a'),adapter(image_path='b'))
        self.assertEqual(state['max'],1);self.assertEqual(len(results),2)

    async def test_cancelled_request_is_drained_before_next_frame(self):
        # Real subprocess transport; the inference payload is explicitly synthetic.
        script='''import sys,json,time\nprint(json.dumps({'status':'ready','setup_s':0,'fixture':True}),flush=True)\nfor line in sys.stdin:\n r=json.loads(line)\n time.sleep(.04)\n print(json.dumps({'data':{'objects':[]},'fixture':True,'metrics':{'request_ref':r['request_id'],'fixture':True}}),flush=True)\n'''
        class FakeWorker(LocalDetectorProcess):
            async def _start_locked(self):
                if self._closed:raise RuntimeError('Closed')
                if self._process is not None:return
                self._process=await asyncio.create_subprocess_exec(sys.executable,'-u','-c',script,
                    stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                self._stderr_task=asyncio.create_task(self._stderr(self._process.stderr))
                self.setup_metrics=await self._line()
        adapter=FakeWorker()
        try:
            await adapter.start();pid=adapter._process.pid
            first=asyncio.create_task(adapter(image_path='a',request_id='A'))
            await asyncio.sleep(.01);first.cancel()
            with self.assertRaises(asyncio.CancelledError):await first
            second=await adapter(image_path='b',request_id='B')
            self.assertEqual(second['metrics']['request_ref'],'B')
            self.assertEqual(second['metrics']['worker_pid'],pid)
            self.assertEqual([x['request_ref'] for x in adapter.drain_usage()],['A','B'])
            self.assertEqual(adapter.drain_usage(),[])
        finally:await adapter.aclose()
        self.assertIsNone(adapter._process)


if __name__=='__main__':unittest.main()
