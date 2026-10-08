"""Second-pass independent segment/hold regressions; no production edits.
Known failure cases deliberately assert the requested safety contract.
"""
from unittest.mock import patch
import json
import cv2
import numpy as np
import pytest
from pipeline import cross_view as cv, dedup, contracts as C


def book(fid, j, x):
    return {'observation_id':f'{fid}-o{j:03d}', 'frame_id':fid,
            'kind':'book','category':'book','polygon_xy':[[x,50],[x+10,50],[x+10,300],[x,300]],
            'bbox_xyxy':[x,50,x+10,300],'confidence':.9,'orientation':'upright','occlusion':'none'}


def fixture_frames(n):
    return {f'f{i:05d}':{'timestamp_s':i*.2,'width_px':640,'height_px':480} for i in range(n)}


@pytest.mark.parametrize('held_status',['hold:tracking_failed',pytest.param('segment_start',marks=pytest.mark.xfail(strict=True,reason=
    "Design decision 2026-10-06: a segment_start frame is the new anchor of its segment (later frames are tracked relative to it), so it may pair "
    "WITHIN its segment; it never pairs across segments. This param conflicts with test_item_consolidation_never_compares_different_segment_planes, "
    "which requires the segment_start frame to join later sightings of its own segment."))])
def test_held_constellation_cannot_claim_physical_identity(held_status):
    frames=fixture_frames(3)
    obs=[book(f,j,x) for f in frames for j,x in enumerate([80,210,340,470])]
    transforms={f:np.eye(3) for f in frames}
    status={'f00000':'start','f00001':held_status,'f00002':'ok'}
    with patch.object(cv,'chain_transforms',return_value=(transforms,status)), patch.object(cv,'appearance',return_value=1.):
        edges,report,_=cv.track_edges(obs,frames,lambda f:np.zeros((480,640,3),np.uint8))
    _,accepted,_=dedup.associate(obs,edges,.07)
    assert 'f00001' in report['hold_frames']
    assert not any('f00001' in e['frame_ids'] for e in accepted), 'A four-box constellation is not permission to use a held native placement'


def test_three_missing_images_force_segment_boundary_on_recovery():
    image=np.random.default_rng(42).integers(0,255,(120,160,3),dtype=np.uint8)
    ids=['a','missing1','missing2','missing3','recovered']
    load=lambda f:None if f.startswith('missing') else image
    T,status=cv.chain_transforms(ids,load)
    segments=cv.segments_from_status(ids,status)
    assert all(status[f]=='hold:no_image' for f in ids[1:4])
    assert status['recovered']=='segment_start' and segments['recovered']>segments['a'], 'The documented three-failure boundary must include missing-image failures'


def test_item_consolidation_never_compares_different_segment_planes(tmp_path):
    frames=fixture_frames(6);obs=[]
    status={f:('start' if i==0 else 'segment_start' if i==3 else 'ok') for i,f in enumerate(frames)}
    for i,(fid,f) in enumerate(frames.items()):
        # (the two pots also look different: same-looking groups in different segments may merge by appearance, pipeline/items.consolidate_items_v2)
        img=np.zeros((480,640,3),np.uint8);img[50:90,40:90]=(0,0,255) if i<3 else (255,128,0)
        path=f'{fid}.png';cv2.imwrite(str(tmp_path/path),img)
        f.update(frame_id=fid,image_path=path,quality={},selected=True,selection_reason='SYNTHETIC')
        obs.extend(book(fid,j,x) for j,x in enumerate([120,250,380,510]))
        # Two DIFFERENT pots at the same native coordinate in disconnected views.
        # Reset segment planes share numbers but do not share physical positions.
        obs.append({**book(fid,4,40),'kind':'non_book','category':'vase' if i<3 else 'pot',
                    'bbox_xyxy':[40,50,90,90],'polygon_xy':[[40,50],[90,50],[90,90],[40,90]]})
    (tmp_path/'manifest.json').write_text(json.dumps(C.envelope('segments','frames',{'frames':list(frames.values())},fixture=True)))
    (tmp_path/'candidates.json').write_text(json.dumps(C.envelope('segments','detect',{'observations':obs},fixture=True)))
    with patch.object(cv,'chain_transforms',return_value=({f:np.eye(3) for f in frames},status)), patch.object(cv,'appearance',return_value=1.):
        p=dedup.merge(tmp_path/'candidates.json',tmp_path/'manifest.json',tmp_path,{'association':'track'})
    inv=json.loads(p.read_text());assert inv['metrics']['track_report']['segments']==2
    assert len(inv['data']['items'])==2, 'Time-disjoint does not make unrelated reset planes spatially comparable'


def test_failed_step_preserves_the_last_verified_anchor():
    images={f:np.full((80,80),i*10,np.uint8) for i,f in enumerate(['a','b','c'])}
    pts=np.arange(80,dtype=np.float32).reshape(40,1,2)
    flows=[]
    def flow(prev,cur,p,*args,**kw):
        flows.append((int(prev[0,0]),int(cur[0,0])))
        return p.copy(),np.ones((40,1),np.uint8),np.zeros((40,1))
    with patch.object(cv2,'goodFeaturesToTrack',return_value=pts), patch.object(cv2,'calcOpticalFlowPyrLK',side_effect=flow), patch.object(cv,'_step_transform',side_effect=[(None,0),(np.eye(3),40)]):
        _,status=cv.chain_transforms(['a','b','c'],images.__getitem__,scale=1)
    assert status=={'a':'start','b':'hold:tracking_failed','c':'ok'}
    assert flows[2]==(0,20), 'Recovery must use a, not the failed b, as anchor'


def test_third_failed_fit_starts_independent_segment():
    image=np.zeros((80,80),np.uint8)
    ids=['a','b','c','d','e'];T,status=cv.chain_transforms(ids,lambda f:image)
    assert [status[f] for f in ids]==['start','hold:tracking_failed','hold:tracking_failed','segment_start','hold:tracking_failed']
    assert cv.segments_from_status(ids,status)=={'a':0,'b':0,'c':0,'d':1,'e':1}
    assert np.array_equal(T['d'],np.eye(3))


def test_projective_fit_at_plane_origin_cannot_bypass_hold_with_nonfinite_chain():
    # Invertible projective H with perfect point fit, but inverse(H)[2,2] == 0.
    # A fit may be locally meaningful while normalization by the plane origin is undefined.
    p0=np.array([[[20.+i,5.+i%7]] for i in range(40)],np.float32)
    H=np.array([[1.,0,0],[0,0,1],[0,1,0]])
    q=np.c_[p0.reshape(-1,2),np.ones(40)] @ H.T
    p1=(q[:,:2]/q[:,2:]).astype(np.float32).reshape(-1,1,2)
    mask=np.ones((40,1),np.uint8)
    with patch.object(cv2,'goodFeaturesToTrack',return_value=p0), patch.object(cv2,'calcOpticalFlowPyrLK',side_effect=[(p1,mask,None),(p0,mask,None)]), patch.object(cv2,'estimateAffine2D',return_value=(np.eye(3)[:2],mask)), patch.object(cv2,'findHomography',return_value=(H,mask)):
        T,status=cv.chain_transforms(['a','b'],lambda f:np.zeros((80,80),np.uint8),scale=1)
    assert status['b'].startswith('hold') and np.isfinite(T['b']).all(), 'Invalid accumulated-plane normalization cannot be recorded as successful motion'
