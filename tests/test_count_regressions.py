"""Synthetic regressions, NOT real-shelf ground truth or accuracy measurements."""
import json
import asyncio
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from pipeline import contracts as C
from pipeline.detect import detect, detect_frame_async, native_box, fuse_tiles, tiles, save_json, read_json
from pipeline.dedup import merge, associate, registration, bridge_registration, DEFAULTS
from providers.local_detector import prepare_run


def observation(fid,index,box,kind="book",category="book",orientation="upright"):
    return {"observation_id":C.observation_id(fid,index),"frame_id":fid,"bbox_xyxy":box,
        "kind":kind,"category":category,"orientation":orientation,"confidence":.9,
        "occlusion":"partial","evidence_notes":"SYNTHETIC fixture",
        "source_tile":[0,0,640,480],"response_ref":"response.json"}


class DetectionTests(unittest.TestCase):
    def test_native_coordinates_and_bounds(self):
        self.assertEqual(native_box([100,200,900,800],"yxyx_1000",[100,200,500,300],1920,1080),[200,230,500,470])
        for box in [[900,20,100,40],[0,-1,20,40],[0,20,float("nan"),40]]:
            with self.assertRaises(ValueError):
                native_box(box,"yxyx_1000",[0,0,1920,1080],1920,1080)
        regions = tiles(1920,1080,960)
        self.assertTrue(all(x+w<=1920 and y+h<=1080 for x,y,w,h in regions))
        self.assertEqual(max(x+w for x,y,w,h in regions),1920)

    def test_only_cross_tile_duplicates_suppressed(self):
        a = observation("f00000",0,[10,10,30,100])
        b = observation("f00000",1,[10,10,30,100])
        c = observation("f00000",2,[11,10,31,100])
        c["source_tile"] = [5,0,640,480]
        kept,possible = fuse_tiles([a,b,c])
        self.assertEqual(len(kept),2)  # same-frame same-tile duplicates remain auditable
        self.assertTrue(possible)
        self.assertEqual(sum(len(o.get("tile_aliases",[])) for o in kept),1)
        d = observation("f00000",3,[12,10,32,100],"non_book","framed_sign")
        d["source_tile"] = c["source_tile"]
        self.assertEqual(len(fuse_tiles([a,d])[0]),2)  # text decor cannot be suppressed into a book

    def test_mixed_api_injection_and_missing_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = np.zeros((480,640,3),np.uint8)
            cv2.imwrite(str(root/"frame.png"),img)
            frames = [{"frame_id":f"f{i:05d}","timestamp_s":i,"image_path":"frame.png",
                "width_px":640,"height_px":480,"quality":{},"selected":True,"selection_reason":"SYNTHETIC"} for i in range(2)]
            manifest = C.envelope("synthetic","frames",{"frames":frames},fixture=True)
            save_json(root/"manifest.json",manifest)
            raw = {"objects":[dict(box_2d=[100,100,600,180],kind="book",category="book",orientation="leaning",confidence=.8,occlusion="partial",evidence_notes="binding visible; unreadable"),
                dict(box_2d=[650,100,700,500],kind="book",category="book",orientation="horizontal",confidence=.8,occlusion="none",evidence_notes="stack layer"),
                dict(box_2d=[720,100,770,500],kind="book",category="book",orientation="stacked",confidence=.8,occlusion="none",evidence_notes="second stack layer"),
                dict(box_2d=[100,600,400,900],kind="non_book",category="framed_sign",orientation="upright",confidence=.9,occlusion="none",evidence_notes="words on framed sign; no binding") ]}
            async def adapter(**kwargs):
                return {"data":raw,"metrics":{"request_count":0,"fixture":True}}
            path = detect(root/"manifest.json",root,{"image_request":adapter})
            artifact = read_json(path)
            self.assertTrue(artifact["fixture"])
            self.assertEqual(artifact["metrics"]["by_kind"],{"book":6,"non_book":2,"ambiguous":0})
            self.assertEqual(C.validate_candidates(artifact,manifest),[])
            self.assertEqual(artifact["data"]["observations"][1]["orientation"],"flat")
            save_json(root/"response.json",raw)
            save_json(root/"replay.json",{"f00000":{"response_path":"response.json"}})
            partial = read_json(detect(root/"manifest.json",root,{"responses":root/"replay.json"}))
            self.assertEqual(partial["status"],"partial")
            self.assertEqual(partial["data"]["frames"][1]["status"],"unprocessed")


class AssociationTests(unittest.TestCase):
    def _edge(self,a,b,score=1.0,eligible=True):
        return {"observation_ids":[a["observation_id"],b["observation_id"]],
            "frame_ids":[a["frame_id"],b["frame_id"]],"score":score,"eligible":eligible,"reason":"SYNTHETIC"}

    def test_identical_copies_and_transitive_same_frame_exclusion(self):
        a1,a2 = [observation("f00000",i,[i*30,10,i*30+20,100]) for i in range(2)]
        b = observation("f00001",0,[0,10,20,100])
        c = observation("f00002",0,[30,10,50,100])
        groups,accepted,uncertain = associate([a1,a2,b,c],
            [self._edge(a1,b),self._edge(a2,c),self._edge(b,c,.9)],.07)
        self.assertEqual(len(groups),2)
        self.assertEqual(len(accepted),2)
        self.assertEqual(len(uncertain),1)
        self.assertIn("Same-frame",uncertain[0]["reason"])

    def test_competing_identical_copies_abstain(self):
        a = observation("f00000",0,[0,0,20,100])
        b,c = [observation("f00001",i,[i*20,0,i*20+20,100]) for i in range(2)]
        groups,accepted,uncertain = associate([a,b,c],[self._edge(a,b,.90),self._edge(a,c,.88)],.07)
        self.assertEqual(len(groups),3)
        self.assertEqual(len(accepted),0)
        self.assertEqual(len(uncertain),2)

    def test_unreadable_and_identical_title_copies_remain_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(12)
            image = rng.integers(0,255,(480,640,3),dtype=np.uint8)
            frames,obs = [],[]
            for i in range(2):
                fid=f"f{i:05d}"
                cv2.imwrite(str(root/f"{fid}.png"),image)
                frames.append(dict(frame_id=fid,timestamp_s=i,image_path=f"{fid}.png",width_px=640,height_px=480,quality={},selected=True,selection_reason="SYNTHETIC"))
                obs += [observation(fid,0,[100,100,140,300]),observation(fid,1,[140,100,180,300]),
                    observation(fid,2,[350,100,440,260],"non_book","framed_sign"),
                    observation(fid,3,[350,300,440,390],"ambiguous","unknown")]
            save_json(root/"manifest.json",C.envelope("synthetic","frames",{"frames":frames},fixture=True))
            save_json(root/"candidates.json",C.envelope("synthetic","detect",{"observations":obs},fixture=True))
            # Equal titles deliberately cannot create/alter association links.
            readings = [{"observation_id":o["observation_id"],"visible_text":"SAME TITLE","title_text":"SAME TITLE","read_confidence":.9} for o in obs if o["kind"] == "book"]
            save_json(root/"readings.json",C.envelope("synthetic","read",{"readings":readings},fixture=True))
            inventory = read_json(merge(root/"candidates.json",root/"manifest.json",root,readings_path=root/"readings.json"))
            self.assertTrue(inventory["fixture"])
            self.assertEqual(len(inventory["data"]["books"]),2)
            self.assertEqual(len(inventory["data"]["items"]),2)
            self.assertEqual(len(inventory["data"]["ambiguous_instance_ids"]),1)
            for book in inventory["data"]["books"]:
                self.assertEqual(len(book["observation_ids"]),2)
                self.assertEqual(book["status"],"unidentified")
                self.assertIsNone(book["identity"]["title"])
                self.assertEqual(len(book["evidence_refs"]),2)

    def test_orb_translation_and_unrelated_views(self):
        rng=np.random.default_rng(42)
        image=rng.integers(0,255,(600,800,3),dtype=np.uint8)
        shifted=cv2.warpPerspective(image,np.array([[1,0,35],[0,1,12],[0,0,1]],dtype=float),(800,600))
        reg=registration(image,shifted,DEFAULTS)
        self.assertEqual(reg["status"],"accepted")
        np.testing.assert_allclose(np.array(reg["homography"])[:2,2],[35,12],atol=1)
        unrelated=rng.integers(0,255,(600,800,3),dtype=np.uint8)
        self.assertEqual(registration(image,unrelated,DEFAULTS)["status"],"unavailable")

    def test_bridge_composes_only_trusted_steps_and_retains_evidence(self):
        frames={f"f{i:05d}":dict(frame_id=f"f{i:05d}",timestamp_s=i*.4,
            image_path=f"frames/{i}.png",width_px=640,height_px=480,quality={}) for i in range(3)}
        images={fid:np.zeros((480,640,3),np.uint8) for fid in frames}
        H=np.array([[1,0,10],[0,1,5],[0,0,1]],float)
        trusted=dict(status="accepted",homography=H.tolist(),inliers=30,inlier_ratio=.8,
            median_reprojection_px=1.,spatial_coverage=.2)
        with patch("pipeline.dedup.registration",return_value=trusted):
            result=bridge_registration("f00000","f00002",frames,Path("."),images,DEFAULTS,{})
        self.assertEqual(result["bridge_steps_count"],2)
        np.testing.assert_allclose(result["homography"],H@H)
        self.assertEqual(result["bridge_steps"][0]["evidence_refs"],["frames/0.png","frames/1.png"])
        cache={("f00000","f00001"):{"status":"unavailable","homography":None}}
        with patch("pipeline.dedup.registration",side_effect=AssertionError("No trusted route exists")):
            self.assertIsNone(bridge_registration("f00000","f00002",frames,Path("."),images,DEFAULTS,cache))


class IncrementalTests(unittest.TestCase):
    def _manifest(self,root):
        cv2.imwrite(str(root/"frame.png"),np.zeros((480,640,3),np.uint8))
        frames=[dict(frame_id=f"f{i:05d}",timestamp_s=i,image_path="frame.png",width_px=640,
            height_px=480,quality={},selected=True,selection_reason="SYNTHETIC") for i in range(2)]
        return save_json(root/"manifest.json",C.envelope("synthetic","frames",{"frames":frames},fixture=True))

    def test_exact_frame_revision_and_pixel_polygon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self._manifest(root)
            revision=[1]
            async def adapter(**kwargs):
                return {"coordinate_format":"xyxy_pixels","data":{"objects":[{
                    "bbox_xyxy":[10,20,30,100],"polygon_xy":[[10,20],[30,20],[30,100],[10,100]],
                    "kind":"book","category":"book","orientation":"leaning","confidence":.8,
                    "evidence_notes":"SYNTHETIC unreadable spine","occlusion":"partial"}]},
                    "metrics":{"request_count":0,"local_inference_count":1,"fixture":True}}
            result=asyncio.run(detect_frame_async(manifest,root,"f00001",revision=1,
                current_revision=lambda:revision[0],config={"image_request":adapter}))
            self.assertEqual(result["status"],"ready")
            self.assertEqual(result["observations"][0]["observation_id"],"f00001-o000")
            self.assertEqual(result["observations"][0]["bbox_xyxy"],[10,20,30,100])
            self.assertEqual(result["observations"][0]["polygon_xy"][0],[10,20])
            self.assertEqual(result["metrics"]["local_inference_count"],1)
            frozen=(root/result["artifact_path"]).read_bytes()
            revision[0]=2
            second=asyncio.run(detect_frame_async(manifest,root,"f00001",revision=2,
                current_revision=lambda:revision[0],config={"image_request":adapter}))
            self.assertNotEqual(second["artifact_path"],result["artifact_path"])
            self.assertEqual(frozen,(root/result["artifact_path"]).read_bytes())

    def test_stale_and_cancelled_tasks_never_publish_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self._manifest(root);revision=[2];calls=[]
            async def stale_adapter(**kwargs):
                calls.append(1);revision[0]=3
                return {"data":{"objects":[]},"metrics":{"fixture":True}}
            before=asyncio.run(detect_frame_async(manifest,root,"f00000",revision=1,
                current_revision=lambda:revision[0],config={"image_request":stale_adapter}))
            self.assertEqual(before["status"],"discarded");self.assertEqual(calls,[])
            after=asyncio.run(detect_frame_async(manifest,root,"f00000",revision=2,
                current_revision=lambda:revision[0],config={"image_request":stale_adapter}))
            self.assertEqual(after["status"],"discarded");self.assertNotIn("observations",after)
            async def cancellation():
                entered=asyncio.Event()
                async def wait_adapter(**kwargs):
                    entered.set();await asyncio.Event().wait()
                task=asyncio.create_task(detect_frame_async(manifest,root,"f00000",revision=3,
                    current_revision=lambda:revision[0],config={"image_request":wait_adapter}))
                await entered.wait();task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            asyncio.run(cancellation())
            self.assertFalse((root/"live/detect/f00000.r000003.candidates.json").exists())

    def test_versioned_copy_preserves_exact_bytes_ids_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source";source.mkdir();self._manifest(source)
            output=root/"new"
            prepare_run(source,output)
            self.assertEqual((source/"frame.png").read_bytes(),(output/"frame.png").read_bytes())
            self.assertEqual(read_json(source/"manifest.json")["data"],read_json(output/"manifest.json")["data"])
            with self.assertRaises(ValueError):
                prepare_run(source,output)


if __name__ == "__main__":
    unittest.main()
