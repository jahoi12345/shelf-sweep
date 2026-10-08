"""Debugger security/join regressions; real artifact checks are separate from truth."""
import copy
import json
from pathlib import Path
import struct
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.debug_registry import read_registry
from app.debug_views import DebugStore, digest
from app.debug_server import app, image_size

ROOT=Path(__file__).resolve().parents[1]
saved_runs = pytest.mark.skipif(not (ROOT/'runs/yugotest1_panorama_worker2_v1/panorama.json').is_file(), reason='Optional real-artifact integration check: workspace panorama runs are not distributed')



def test_canonical_registry_aliases_not_scans():
    scans=read_registry(ROOT)
    assert [s['label'] for s in scans]==['IESTest3','YugoTest1','YugoTest2']   # sorted by label
    assert 'IMG_2281.MOV' in next(s for s in scans if s['label']=='YugoTest1')['aliases']
    assert not any('preview' in s['label'] for s in scans)


@pytest.mark.parametrize('run',['../sweeps','/etc','a/b','..','x%2fy'])
def test_invalid_run_ids(run):
    with pytest.raises(ValueError): DebugStore(ROOT).run(run)


@saved_runs
def test_actual_panorama_native_projection_and_separate_association():
    store=DebugStore(ROOT)
    v=store.view('YugoTest1','yugotest1_panorama_worker2_v1')
    pan=json.loads((ROOT/'runs/yugotest1_panorama_worker2_v1/panorama.json').read_text())
    assert v['counts']['verified'] is False
    assert v['counts']['proposed_book_groups']==pan['counts']['groups_by_kind']['book']
    for c in v['components']:
        for o in c['overlays']:
            f=next(f for f in c['source_frames'] if f['frame_id']==o['frame_id'])
            points=np.column_stack([o['native_polygon_xy'],np.ones(len(o['native_polygon_xy']))])
            p=points@np.asarray(f['original_to_panorama_3x3']).T
            assert np.allclose(p[:,:2]/p[:,2,None],o['panorama_polygon_xy'],atol=.02)


@saved_runs
def test_wrong_scan_never_joins():
    with pytest.raises(ValueError,match='hash mismatch'):
        DebugStore(ROOT).view('YugoTest2','yugotest1_panorama_worker2_v1')


@saved_runs
def test_actual_http_assets_crops_downloads_and_limits():
    with TestClient(app) as client:
        v=client.get('/api/view/YugoTest1/yugotest1_panorama_worker2_v1').json()
        asset=v['components'][0]['image_asset']
        r=client.get('/api/assets/yugotest1_panorama_worker2_v1/'+asset)
        assert r.status_code==200 and r.headers['content-type']=='image/jpeg'
        assert client.get('/api/assets/yugotest1_panorama_worker2_v1/not-listed').status_code==404
        r=client.get('/api/crop/YugoTest1/yugotest1_panorama_worker2_v1/f00000-o000')
        assert r.status_code==200 and r.content.startswith(b'\x89PNG')
        assert client.get('/static/app.js').status_code==404
        token=v['downloads']['panorama.json']
        assert 'attachment' in client.get('/api/assets/yugotest1_panorama_worker2_v1/'+token+'?download=true').headers['content-disposition']
        assert "script-src 'self'" in r.headers['content-security-policy']
        c=v['components'][0]['component_id']
        assert client.get(f'/api/inspect/yugotest1_panorama_worker2_v1/{c}?x=-1&y=5').status_code==422
        assert client.get(f'/api/mask/yugotest1_panorama_worker2_v1/{c}/seams').status_code==200
        assert client.get(f'/api/mask/yugotest1_panorama_worker2_v1/{c}/gaps').status_code==200
        preview=client.get('/api/preview/YugoTest1',headers={'Range':'bytes=0-255'})
        assert preview.status_code==206 and len(preview.content)==256


def test_bounded_image_headers(tmp_path):
    p=tmp_path/'bomb.png';p.write_bytes(b'\x89PNG\r\n\x1a\n'+b'\0\0\0\rIHDR'+struct.pack('>II',100000,100000))
    with pytest.raises(HTTPException) as e: image_size(p)
    assert e.value.status_code==413


def test_index_candidate_mismatch_rejected(tmp_path):
    # Copy only tiny synthetic artifacts; no baseline or real index is changed.
    root=tmp_path; (root/'sweeps').mkdir(); (root/'runs/r').mkdir(parents=True)
    registry=json.loads((ROOT/'sweeps/scan_registry.json').read_text());(root/'sweeps/scan_registry.json').write_text(json.dumps(registry))
    sha=registry['scans'][0]['sha256'];manifest={'run_id':'r','status':'ok','data':{'input':{'sha256':sha},'frames':[]}}
    candidates={'run_id':'r','status':'ok','data':{'observations':[],'frames':[]}}
    for name,a in [('manifest.json',manifest),('candidates.json',candidates)]: (root/'runs/r'/name).write_text(json.dumps(a))
    index={'source_run':'runs/r','run_id':'r','source_video_sha256':sha,'input_artifacts':{'candidates':{'sha256':'wrong'},'manifest':{'sha256':digest(root/'runs/r/manifest.json')}},'observations':[]}
    (root/'runs/r/debug_index.json').write_text(json.dumps(index))
    v=DebugStore(root).view('YugoTest1','r')
    assert v['ocr_records']=={} and 'mismatched' in v['ocr_index_status']
    assert v['stages']['registration/stitching']['status']=='unprocessed'
    assert v['counts']['proposed_book_groups'] is None


def test_untrusted_ocr_never_html():
    js=(ROOT/'app/static/debug-app.js').read_text()
    assert 'innerHTML' not in js and 'textContent' in js
    assert 'supported_work' in js and 'supported_edition' in js


@saved_runs
def test_both_actual_scans_keep_versions_and_unstitched_views():
    store=DebugStore(ROOT)
    scans=store.catalogue()
    for scan in [s for s in scans if s['label'] in ('YugoTest1','YugoTest2')]:   # only the Yugo scans have panorama runs
        run=f"{scan['label'].lower()}_panorama_worker2_v1"
        v=store.view(scan['label'],run)
        assert len(v['frames'])==({'YugoTest1':97,'YugoTest2':350}[scan['label']])
        assert any(c['status']=='single_frame' for c in v['components'])
        assert v['ocr_records'] and 'available' in v['ocr_index_status']
        assert not any(r.get('supported_edition') for r in v['ocr_records'].values())
        assert any(r.get('ocr_state')=='not_attempted' for r in v['ocr_records'].values())
        token=v['downloads']['inventory.json']
        assert store.asset_records(run)[6][token].parent.name==run


@saved_runs
def test_actual_pixel_supplier_inverse_mapping():
    from app.debug_server import inspect_pixel
    from app.debug_views import DebugStore
    store=DebugStore(ROOT);run='yugotest1_panorama_worker2_v1'
    _,_,pan,_,components,_,records,_=store.asset_records(run)
    c=components[0]
    import cv2
    indices=cv2.imread(str(records[c['pixel_source_index_asset']]),cv2.IMREAD_UNCHANGED)
    y,x=np.argwhere(indices>0)[len(np.argwhere(indices>0))//2]
    result=inspect_pixel(run,c['component_id'],float(x),float(y))
    f=next(f for f in c['source_frames'] if f['frame_id']==result['frame_id'])
    p=np.asarray(f['original_to_panorama_3x3'])@np.asarray([*result['native_xy'],1])
    assert result['status']=='source_pixel' and np.allclose(p[:2]/p[2],[x,y],atol=.01)


def test_run_reading_adapter_verifies_incremental_slices_and_second_pass(tmp_path):
    import cv2,hashlib
    from app.debug_run_adapter import adapt_readings
    p=tmp_path; (p/'frames').mkdir();(p/'crops').mkdir();(p/'live/read').mkdir(parents=True);(p/'ocr_responses').mkdir()
    im=np.zeros((10,10,3),dtype=np.uint8);cv2.imwrite(str(p/'frames/f.jpg'),im);cv2.imwrite(str(p/'crops/c.png'),im)
    o={'observation_id':'o','frame_id':'f','bbox_xyxy':[0,0,10,10],'polygon_xy':[[0,0],[10,0],[10,10],[0,10]]}
    c={'run_id':'r','data':{'observations':[o]}};m={'run_id':'r','data':{'frames':[{'frame_id':'f','image_path':'frames/f.jpg','timestamp_s':1}]}}
    (p/'candidates.json').write_text(json.dumps(c));slice_path=p/'live/read/g.candidates.json';slice_path.write_text(json.dumps(c)+'\n')
    raw=p/'ocr_responses/raw.json';raw.write_text(json.dumps({'input_sha256':digest(p/'crops/c.png'),'variants':[{'text':'á <script>'}]}))
    r={'observation_id':'o','frame_id':'f','frame_image_path':'frames/f.jpg','timestamp_s':1,'source_polygon_xy':o['polygon_xy'],'crop_path':'crops/c.png','crop_transform':{'bbox_xyxy':o['bbox_xyxy']},'ocr_attempt':'completed','visible_text':'á <script>', 'provenance':{'run_id':'r','candidate_artifact_sha256':digest(slice_path),'frame_sha256':digest(p/'frames/f.jpg'),'crop_sha256':digest(p/'crops/c.png')},'ocr_evidence':{'path':str(raw),'sha256':digest(raw)}}
    for name in ['readings.json','readings_second.json']:(p/name).write_text(json.dumps({'run_id':'r','data':{'readings':[r]}}))
    def add(base,name,expected):return 'asset' if digest(base/name)==expected else None
    rows,binding=adapt_readings(p,c,m,add)
    assert rows['o']['reading_join'].startswith('verified') and len(rows['o']['reader_passes'])==2
    assert rows['o']['literal_text']=='á <script>' and rows['o']['supported_work'] is None
    r['provenance']['candidate_artifact_sha256']='wrong';(p/'readings.json').write_text(json.dumps({'run_id':'r','data':{'readings':[r]}}))
    rows,_=adapt_readings(p,c,m,add)
    assert rows['o']['reading_join']=='rejected_mismatch' and rows['o']['literal_text'] is None


def test_registry_catalogue_without_runs(tmp_path):
    (tmp_path/'sweeps').mkdir();(tmp_path/'sweeps/scan_registry.json').write_text((ROOT/'sweeps/scan_registry.json').read_text())
    assert all(s['runs']==[] for s in DebugStore(tmp_path).catalogue())
