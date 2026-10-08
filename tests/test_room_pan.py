"""Room-pan frames (pipeline/room_pan.py) and their exclusion from the count layers (instances, panorama count, shelf run).
Synthetic sweeps only: nothing here says what the rule does on real sweeps."""
import json

import numpy as np

from pipeline import auto_scale as AS
from pipeline import instances as IN
from pipeline import panorama_count as PC
from pipeline import room_pan as RP
from tests import test_auto_scale as TAS
from tests import test_panorama_count as TPC

FPS = 5.0


def _frames(n):
    return [{'frame_id': f'f{i:05d}', 'timestamp_s': i / FPS} for i in range(n)]


def _book_obs(fid, k, long_px):
    return {'observation_id': f'{fid}-o{k:03d}', 'frame_id': fid, 'kind': 'book', 'polygon_xy': [[0, 0], [30, 0], [30, long_px], [0, long_px]]}


def _sweep(kinds):
    """kinds: one letter per frame. 's' shelf frame (close spines), 'f' far spines, 'n' detector ran, no book, '.' no detection."""
    frames = _frames(len(kinds))
    obs, det = [], set()
    for f, k in zip(frames, kinds):
        fid = f['frame_id']
        if k in 'sfn':
            det.add(fid)
        if k == 's':
            obs += [_book_obs(fid, j, 400) for j in range(3)]
        elif k == 'f':
            obs += [_book_obs(fid, j, 120) for j in range(3)]
        elif k == 'n':
            obs.append({'observation_id': f'{fid}-o000', 'frame_id': fid, 'kind': 'non_book', 'category': 'sofa'})
    books = [{'instance_id': f'bk-{i}', 'polygon_xy': [[0, 0], [30, 0], [30, 400], [0, 400]]} for i in range(20)]   # run median native spine 400 px
    return frames, obs, books, det


def test_a_long_pan_off_the_shelf_is_a_room_pan_and_every_frame_of_it_is_tagged_with_the_far_re_views_inside():
    # floor/ceiling frames (no book) over 8 s with far overviews of the shelf in between, then the close shelf pass
    kinds = 'n.n.n.n.ffff.n.n.n.n.ffff.n.n.n.n.n.n.n' + 's.s.s.s.s.s.s.s.s.s.s.s.s'
    frames, obs, books, det = _sweep(kinds)
    res = RP.classify_frames(frames, obs, books, detection_frames=det)
    first_shelf = kinds.index('s')
    assert res['room_pan_frames'] == [f['frame_id'] for f in frames[:first_shelf]]
    (seg,) = res['segments']
    assert seg['room_pan'] and seg['off_shelf_frames'] >= RP.MIN_OFFSHELF_FRAMES and seg['off_shelf_span_s'] >= RP.MIN_OFFSHELF_SPAN_S
    assert res['signals']['c_detected_no_books'] == kinds.count('n') and res['signals']['b_far_spines'] == kinds.count('f') and not res['uses_3d']


def test_a_walk_up_a_glance_away_and_detector_misses_are_not_a_room_pan():
    walk_up = 'f.f.f.f.f.f.f.f.f.f.f.f' + 's.s.s.s.s.s'                 # far frames facing the shelf, no off-shelf frame
    glance = 's.s.s.s.n.n.n.s.s.s.s'                                    # three frames off the shelf over 0.8 s
    misses = 's.s.n.s.s.n.s.s.n.s.s.n.s'                                # single empty detection frames between shelf frames
    for kinds in (walk_up, glance, misses):
        frames, obs, books, det = _sweep(kinds)
        res = RP.classify_frames(frames, obs, books, detection_frames=det)
        assert res['room_pan_frames'] == [] and res['segments'] == [], kinds


def test_the_3d_pose_adds_oblique_frames_but_never_overrules_a_frame_with_close_spines():
    kinds = '.' * 30 + 's.s.s.s.s.s'
    frames, obs, books, det = _sweep(kinds)
    assert RP.classify_frames(frames, obs, books, detection_frames=det)['room_pan_frames'] == []       # 2D alone: no evidence at all in the first 6 s
    oblique = {f'f{i:05d}': {'z': 3.0, 'angle_deg': 85.0} for i in range(30)}                         # the camera looks along the plane (floor, ceiling)
    close = {f['frame_id']: {'z': 1.0, 'angle_deg': 5.0} for f in frames[30:]}
    res = RP.classify_frames(frames, obs, books, detection_frames=det, poses={**oblique, **close})
    assert len(res['room_pan_frames']) == 30 and res['uses_3d'] and res['signals']['a_oblique_3d'] == 30
    # a joined 3D fragment with wrong poses: close spines in every other frame keep the shelf pass, whatever the 3D says
    kinds2 = 's.' * 20
    frames2, obs2, books2, det2 = _sweep(kinds2)
    wrong = {f['frame_id']: {'z': 1.0, 'angle_deg': 80.0} for f in frames2}
    assert RP.classify_frames(frames2, obs2, books2, detection_frames=det2, poses=wrong)['room_pan_frames'] == []


def test_camera_pose_signals_read_the_shelf_aligned_cameras():
    rec = {'alignment': {'applied': True}, 'cameras': [
        {'frame_id': 'f00000', 'center': [0, 0, 1.5], 'rotation_world_to_cam': [[1, 0, 0], [0, -1, 0], [0, 0, -1]]},     # looks straight at the plane
        {'frame_id': 'f00001', 'center': [0, 0, 1.5], 'rotation_world_to_cam': [[1, 0, 0], [0, 0, -1], [0, 1, 0]]}]}      # optical axis along +y: the ceiling
    p = RP.camera_pose_signals(rec)
    assert p['f00000']['angle_deg'] < 1 and p['f00001']['angle_deg'] > 89 and p['f00000']['z'] == 1.5
    assert RP.camera_pose_signals({'cameras': rec['cameras'], 'alignment': {'applied': False}}) == {}


def test_classify_run_reads_the_run_files_and_the_newest_reconstruction(tmp_path):
    kinds = 'n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n' + 's.s.s.s.s'
    frames, obs, books, det = _sweep(kinds)
    run = tmp_path / 'runs' / 'r1'
    (run / 'detection_responses').mkdir(parents=True)
    for fid in det:
        (run / 'detection_responses' / f'{fid}_tile000.json').write_text('{}')
    (run / 'manifest.json').write_text(json.dumps({'data': {'frames': frames}}))
    (run / 'candidates.json').write_text(json.dumps({'data': {'observations': [o for o in obs if o['kind'] == 'book']}}))    # empty frames leave no observation
    (run / 'inventory.json').write_text(json.dumps({'data': {'books': books}}))
    res = RP.classify_run(run)
    assert res['room_pan_frame_count'] == kinds.index('s') and res['recon3d'] is None
    for v in (1, 3):
        (tmp_path / 'runs' / f'r1_recon3d_v{v}').mkdir()
        (tmp_path / 'runs' / f'r1_recon3d_v{v}' / 'recon3d.json').write_text(json.dumps({'alignment': {'applied': True}, 'cameras': []}))
    assert RP.latest_recon3d(run).parent.name == 'r1_recon3d_v3'
    assert RP.classify_run(tmp_path / 'runs' / 'missing')['room_pan_frame_count'] == 0


def _group(gid, obs, tier='confident'):
    return {'instance_id': gid, 'observation_ids': sorted(obs), 'count_tier': tier, 'frame_ref': obs[0].split('-o')[0]}


def test_an_instance_seen_only_in_the_room_pan_is_unconfirmed_with_its_reason_and_one_shelf_sighting_keeps_a_line():
    frames = {f'f{i:05d}': {'timestamp_s': i * 0.2, 'quality': {'sharpness': 100.0}} for i in range(8)}
    segs = {f: 0 for f in frames}
    books = [_group('bk-f00000-o000', ['f00000-o000', 'f00001-o000']), _group('bk-f00001-o001', ['f00001-o001', 'f00005-o001']),
             _group('bk-f00005-o000', ['f00005-o000', 'f00006-o000'])]
    rp = {'room_pan_frames': ['f00000', 'f00001', 'f00002'], 'room_pan_frame_count': 3, 'frames_total': 8, 'segments': [], 'uses_3d': False}
    art = IN.build({'books': books, 'counts': {'count_range': [3, 3]}}, frames, segs, room_pan=rp)
    inst = {i['instance_id']: i for i in art['instances']}
    assert inst['bk-f00000-o000']['tier'] == 'unconfirmed' and inst['bk-f00000-o000']['room_pan_only'] and inst['bk-f00000-o000']['tier_reason'].startswith(IN.ROOM_PAN_REASON)
    assert inst['bk-f00001-o001']['tier'] == 'confident' and not inst['bk-f00001-o001']['room_pan_only']
    assert art['count']['best'] == 2 and art['stats']['room_pan_demoted'] == 1 and art['room_pan']['demoted_instance_ids'] == ['bk-f00000-o000']
    assert IN.build({'books': books, 'counts': {'count_range': [3, 3]}}, frames, segs)['count']['best'] == 3       # no room pan: nothing changes
    assert IN.ROOM_PAN_REASON == RP.REASON


def test_a_piece_made_mostly_of_room_pan_frames_is_not_counted_and_gives_no_collapsed_fallback(tmp_path):
    run, pan, reg = TPC.make_run(tmp_path, TPC.DETS)
    P = json.loads((pan / 'panorama.json').read_text())
    for c in P['components']:
        c['source_frames'] = [{'frame_id': f'{c["component_id"]}-f{i}'} for i in range(len(c['source_frames']))]
    (pan / 'panorama.json').write_text(json.dumps(P))
    before = PC.build_count(run, pan, detections=TPC.DETS, registration=reg)
    pan_frames = [f'component_002-f{i}' for i in range(4)] + ['component_001-f0']          # 4 of 5 frames of the collapsed piece, 1 of 5 of the normal one
    res = PC.build_count(run, pan, detections=TPC.DETS, registration=reg, room_pan={'room_pan_frames': pan_frames, 'room_pan_frame_count': 5})
    cls = {c['component_id']: c for c in res['panorama_components']}
    assert cls['component_002']['class'] == 'room_pan' and cls['component_001']['class'] == 'normal' and RP.REASON in cls['component_002']['class_reason']
    assert res['collapsed_piece_groups'] == 0 and res['panorama_count'] == before['panorama_count'] - before['collapsed_piece_groups']
    assert res['room_pan']['pieces'] == {'component_002': 0.8} and res['registration']['pieces_room_pan'] == 1
    assert res['panorama_spines'] == before['panorama_spines'] and res['merge_evidence'] == before['merge_evidence']


def test_the_shelf_run_and_the_scale_prior_skip_room_pan_sightings_but_keep_the_rest():
    big = {f'bk-{i}': (0.05 * i - 0.5, 0.0, (20.0 + (i % 5)) / TAS.SCALE, 3.0 / TAS.SCALE) for i in range(14)}
    rc = TAS.recon(big, frames=3)
    all_frames = AS.measure_data(rc, [], None)
    skip_one = AS.measure_data(rc, [], None, exclude_frames={'f00000'})
    assert skip_one['summary']['measured'] == all_frames['summary']['measured'] == 14
    assert skip_one['summary']['room_pan_sightings_not_measured'] == 14 and skip_one['summary']['room_pan_only_groups'] == 0
    none_left = AS.measure_data(rc, [], None, exclude_frames={'f00000', 'f00001', 'f00002'})
    assert none_left['summary']['measured'] == 0 and none_left['summary']['room_pan_only_groups'] == 14
    assert none_left['shelf']['shelf_run_m'] is None and all(s['reason'].startswith('no_usable_sighting:room_pan_view') for s in none_left['spines'])
    assert np.isclose(skip_one['shelf']['shelf_run_m'], all_frames['shelf']['shelf_run_m'])
