"""Independent regressions from the first count review; rerun against later fixes.
No production fixes. Synthetic copy labels are declared by construction, not room truth.
"""
import numpy as np
import pytest
from unittest.mock import patch
from pipeline import cross_view as cv, dedup
from pipeline.items import consolidate_item_instances


def o(fid,j,x,y,w=10,h=250):
 poly=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
 return {'observation_id':f'{fid}-o{j:03d}','frame_id':fid,'kind':'book','category':'book',
         'polygon_xy':poly,'bbox_xyxy':[x,y,x+w,y+h],'confidence':.9,'orientation':'upright','occlusion':'none'}


def test_neighbouring_identical_thin_copies_do_not_merge_under_perfect_registration():
 frames={f'f{i:05d}':{'timestamp_s':i*.2,'width_px':640,'height_px':480} for i in range(2)}
 obs=[]
 for fid in frames:
  obs += [o(fid,j,x,50) for j,x in enumerate([40,220,350,500])]
 # Perfectly registered scene contains TWO identical thin books at x=100 and112.
 # Detector flickers: frame0 sees only copy A, frame1 only copy B.
 obs += [o('f00000',4,100,50),o('f00001',4,112,50)]
 with patch.object(cv,'appearance',return_value=1.):
  edges,_,_=cv.track_edges(obs,frames,lambda f:np.zeros((480,640,3),np.uint8),transforms={f:np.eye(3) for f in frames})
 assert not any(set(e['observation_ids'])=={'f00000-o004','f00001-o004'} for e in edges)


def test_equal_matching_alternatives_preserve_uncertainty_before_greedy_pruning():
 frames={f'f{i:05d}':{'timestamp_s':i*.2,'width_px':640,'height_px':480} for i in range(2)}
 obs=[o('f00000',0,106,50),o('f00001',0,100,50),o('f00001',1,112,50)]
 with patch.object(cv,'appearance',return_value=1.):
  edges,_,_=cv.track_edges(obs,frames,lambda f:np.zeros((480,640,3),np.uint8),transforms={f:np.eye(3) for f in frames})
 groups,accepted,uncertain=dedup.associate(obs,edges,.07)
 assert not accepted and uncertain, 'Two equally plausible adjacent copies must not be confidently joined'


def test_failed_tracking_hold_is_not_used_as_valid_identity_registration():
 frames={f'f{i:05d}':{'timestamp_s':i*.2,'width_px':640,'height_px':480} for i in range(2)}
 obs=[o('f00000',0,100,50),o('f00001',0,100,50)]
 # Low-texture tracking failure provides NO observed camera displacement.
 with patch.object(cv,'appearance',return_value=1.):
  edges,report,_=cv.track_edges(obs,frames,lambda f:np.zeros((480,640,3),np.uint8))
 assert report['tracking_failures']==1
 assert edges==[], 'A hold cannot establish identity placement of another native view'


def test_singular_tracking_fit_is_rejected_without_crashing():
 image=np.random.default_rng(8).integers(0,255,(120,160,3),dtype=np.uint8)
 # The active fitter is affine-first; preserve rejection on that path.
 with patch.object(cv2_module(),'estimateAffine2D',return_value=(np.zeros((2,3)),np.ones((100,1),np.uint8))):
  T,status=cv.chain_transforms(['a','b'],lambda f:image)
 assert status['b'].startswith('hold') and np.isfinite(T['b']).all()


def test_singular_homography_does_not_override_valid_affine():
 image=np.random.default_rng(8).integers(0,255,(120,160,3),dtype=np.uint8)
 with patch.object(cv2_module(),'findHomography',return_value=(np.zeros((3,3)),np.ones((100,1),np.uint8))):
  T,status=cv.chain_transforms(['a','b'],lambda f:image)
 assert status['b']=='ok' and np.isfinite(T['b']).all()
 assert abs(np.linalg.det(T['b']))>1e-6


def cv2_module():
 import cv2
 return cv2


def test_item_plane_rule_is_restricted_to_time_disjoint_sightings():
 obs={};instances=[]
 for j,x in enumerate([0,20]):
  ids=[]
  for k in range(3):
   fid=f"f{k:05d}";oid=f"{fid}-o{j:03d}";ids.append(oid)
   obs[oid]={"observation_id":oid,"frame_id":fid,"bbox_xyxy":[x,0,x+100,100],"confidence":.9}
  instances.append({"instance_id":str(j),"category":"frame","observation_ids":ids})
 groups,links=consolidate_item_instances(instances,obs,{f"f{k:05d}":np.eye(3) for k in range(3)})
 assert len(groups)==2, "Time-overlapping moderate-overlap objects must not use the documented time-disjoint plane rule"


def test_distinct_item_view_veto_is_not_erased_by_one_bad_colocated_view():
 obs={};instances=[]
 for j in range(2):
  ids=[]
  for k in range(3):
   x=130 if j==1 and k==0 else 0
   fid=f"f{k:05d}";oid=f"{fid}-o{j:03d}";ids.append(oid)
   obs[oid]={"observation_id":oid,"frame_id":fid,"bbox_xyxy":[x,0,x+100,100],"confidence":.9}
  instances.append({"instance_id":str(j),"category":"frame","observation_ids":ids})
 groups,_=consolidate_item_instances(instances,obs)
 assert len(groups)==2, "Two objects demonstrably disjoint in f0 cannot become one because boxes collide later"


def test_flat_distance_and_score_are_invariant_to_plane_coordinate_scale():
 def run(scale):
  A=["a"];B=["b"]
  # Matching spans overlap120px of200, but centre displacement80 exceeds
  # the upright length gate. Flat matching must compare dimensionless scores.
  def info(x):
   P=np.array([[x,100],[x+200,100],[x+200,110],[x,110]],float)*scale
   return {"P":P,"c":P.mean(0),"w":10*scale,"L":200*scale,"v":np.array([1.,0.]),"o":{"polygon_xy":P.tolist()}}
  edges=[]
  with patch.object(cv,"appearance",return_value=.9):
   cv._match_flat(edges,A,B,set(),set(),{"a","b"},{"a","b"},{"a":info(0),"b":info(80)},np.zeros(2),"f0","f1",{"f0":{"timestamp_s":0},"f1":{"timestamp_s":1}},lambda i:np.zeros((96,24,3)),cv.DEFAULTS,lambda f:None,{}, {})
  return edges[0]
 a,b=run(1),run(2)
 assert a["center_distance_spine_widths"]==pytest.approx(b["center_distance_spine_widths"])
 assert a["score"]==pytest.approx(b["score"])


def test_unmatched_sightings_after_failed_tracking_remain_uncertain(tmp_path):
 import cv2,json
 from pipeline import contracts as C
 frames=[];obs=[]
 for i,x in enumerate([100,300]):
  fid=f"f{i:05d}";cv2.imwrite(str(tmp_path/f"{fid}.png"),np.zeros((480,640,3),np.uint8))
  frames.append({"frame_id":fid,"image_path":f"{fid}.png","timestamp_s":i*.2,"width_px":640,"height_px":480,"quality":{},"selected":True,"selection_reason":"SYNTHETIC"})
  obs.append(o(fid,0,x,50))
 (tmp_path/"manifest.json").write_text(json.dumps(C.envelope("synthetic","frames",{"frames":frames},fixture=True)))
 (tmp_path/"candidates.json").write_text(json.dumps(C.envelope("synthetic","detect",{"observations":obs},fixture=True)))
 p=dedup.merge(tmp_path/"candidates.json",tmp_path/"manifest.json",tmp_path,{"association":"track"})
 d=json.loads(p.read_text())
 assert d["metrics"]["track_report"]["tracking_failures"]==1
 assert all(g["uncertain_association"] for g in d["data"]["books"])
 assert d["status"]=="partial"


def test_item_consolidation_preserves_member_uncertainty(tmp_path):
 import cv2,json
 from pipeline import contracts as C
 fid='f00000';cv2.imwrite(str(tmp_path/'frame.png'),np.zeros((480,640,3),np.uint8))
 frames=[{'frame_id':fid,'image_path':'frame.png','timestamp_s':0.,'width_px':640,'height_px':480,'quality':{},'selected':True,'selection_reason':'SYNTHETIC'}, {'frame_id':'f00001','image_path':'frame.png','timestamp_s':.2,'width_px':640,'height_px':480,'quality':{},'selected':True,'selection_reason':'SYNTHETIC'}]
 obs=[{**o(fid,0,100,50), 'kind':'non_book','category':'pot','confidence':.9},
      {**o(fid,1,100,50), 'kind':'non_book','category':'vase','confidence':.8}, o(fid,2,400,50), o('f00001',0,400,50)]
 data={'observations':obs,'uncertain_tile_associations':[{'observation_ids':[obs[1]['observation_id']]}]}
 (tmp_path/'manifest.json').write_text(json.dumps(C.envelope('synthetic','frames',{'frames':frames},fixture=True)))
 (tmp_path/'candidates.json').write_text(json.dumps(C.envelope('synthetic','detect',data,fixture=True)))
 p=dedup.merge(tmp_path/'candidates.json',tmp_path/'manifest.json',tmp_path,{'association':'track'})
 d=json.loads(p.read_text());assert len(d['data']['items'])==1
 assert d['data']['items'][0]['uncertain_association'], 'The retained representative must carry uncertainty of all merged members'


def test_count_comparison_rejects_count_array_disagreement(tmp_path):
 import json
 from eval.count_vs_truth import compare
 p=tmp_path/'inventory.json'
 p.write_text(json.dumps({'run_id':'synthetic','data':{'books':[], 'counts':{'book_count':458}}}))
 with pytest.raises((ValueError,KeyError)):
  compare(str(p),458)
