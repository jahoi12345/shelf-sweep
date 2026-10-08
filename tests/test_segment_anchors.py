"""Final reviewer contracts: approved segment anchors and tracker segment propagation."""
from unittest.mock import patch
import numpy as np
from pipeline import cross_view as cv,dedup,tracker


def fixture():
    frames={f'f{i:05d}':{'timestamp_s':i*.2,'width_px':640,'height_px':480} for i in range(4)}
    obs=[]
    for f in frames:
        for j,x in enumerate([80,210,340,470]):
            obs.append({'observation_id':f'{f}-o{j:03d}','frame_id':f,'kind':'book','category':'book','confidence':.9,
                        'bbox_xyxy':[x,50,x+10,300],'polygon_xy':[[x,50],[x+10,50],[x+10,300],[x,300]],'orientation':'upright'})
    return frames,obs


def test_segment_start_pairs_inside_its_verified_segment_only():
    frames,obs=fixture();status={'f00000':'start','f00001':'segment_start','f00002':'ok','f00003':'ok'}
    with patch.object(cv,'chain_transforms',return_value=({f:np.eye(3) for f in frames},status)),patch.object(cv,'appearance',return_value=1.):
        edges,report,_=cv.track_edges(obs,frames,lambda f:np.zeros((480,640,3),np.uint8))
    _,accepted,_=dedup.associate(obs,edges,.07)
    assert any(e['frame_ids']==['f00001','f00002'] for e in accepted)
    segment=cv.segments_from_status(list(frames),status)
    assert all(segment[e['frame_ids'][0]]==segment[e['frame_ids'][1]] for e in accepted)


def test_tracker_uses_reported_segments_when_recovering_from_identity_holds():
    frames,obs=fixture();obs=[o for o in obs if o['frame_id'] in {'f00000','f00003'} and o['observation_id'].endswith('o000')]
    # Last anchor survives two holds. Recovery from that anchor is verified in
    # the SAME segment; identity-valued held transforms are not new segments.
    report={'_transforms':{f:np.eye(3).tolist() for f in frames},'_segment_of':{f:0 for f in frames},
            'hold_frames':['f00001','f00002'],'segments':1}
    with patch.object(cv,'track_edges',return_value=([],report,[])):
        edges,rep,_=tracker.track_v2_edges(obs,frames,lambda f:np.zeros((480,640,3),np.uint8))
    assert rep['tracker']['tracks']==1,'Tracker must use explicit segment IDs, not guess a boundary from a held identity transform'
    assert any(e['geometry']=='tracker_iou' for e in edges)
