"""Stable instance ids (pipeline/dedup.py, pipeline/contracts.py) and the physical-instance layer (pipeline/instances.py).
Synthetic frames and groups only (the accuracy of the merges on real shelves was measured on live sweeps)."""
import copy
import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pipeline import contracts as C
from pipeline import cross_view as cv
from pipeline import dedup
from pipeline import instances as IN
from tests.test_count_segments import book, fixture_frames


# ------------------------------------------------------------------------------------------ step 1: stable ids
def _segmented_run(tmp_path, break_at=3, n=6, pots=True):
    """Six frames of four upright spines (+ one pot); the tracking chain breaks at ``break_at`` so every book becomes two groups."""
    frames = fixture_frames(n)
    obs = []
    status = {f: ('start' if i == 0 else 'segment_start' if i == break_at else 'ok') for i, f in enumerate(frames)}
    for i, (fid, f) in enumerate(frames.items()):
        img = np.zeros((480, 640, 3), np.uint8)
        img[50:90, 40:90] = (0, 0, 255)
        path = f'{fid}.png'
        cv2.imwrite(str(tmp_path / path), img)
        f.update(frame_id=fid, image_path=path, quality={'sharpness': 100.0 + i}, selected=True, selection_reason='SYNTHETIC')
        obs.extend(book(fid, j, x) for j, x in enumerate([120, 250, 380, 510]))
        if pots:
            obs.append({**book(fid, 4, 40), 'kind': 'non_book', 'category': 'vase', 'bbox_xyxy': [40, 50, 90, 90], 'polygon_xy': [[40, 50], [90, 50], [90, 90], [40, 90]]})
    (tmp_path / 'manifest.json').write_text(json.dumps(C.envelope('segments', 'frames', {'frames': list(frames.values())}, fixture=True)))
    (tmp_path / 'candidates.json').write_text(json.dumps(C.envelope('segments', 'detect', {'observations': obs}, fixture=True)))
    return frames, obs, status


def _merge(tmp_path, frames, status):
    with patch.object(cv, 'chain_transforms', return_value=({f: np.eye(3) for f in frames}, status)), patch.object(cv, 'appearance', return_value=1.):
        p = dedup.merge(tmp_path / 'candidates.json', tmp_path / 'manifest.json', tmp_path, {'association': 'track'})
    return json.loads(p.read_text())


def test_ids_come_from_the_earliest_observation_and_survive_a_re_merge(tmp_path):
    frames, obs, status = _segmented_run(tmp_path)
    inv = _merge(tmp_path, frames, status)
    books = inv['data']['books']
    assert [b['instance_id'] for b in books] == [f'bk-f00000-o{j:03d}' for j in range(4)] + [f'bk-f00003-o{j:03d}' for j in range(4)]
    assert [b['display_index'] for b in books] == list(range(1, 9))
    assert all(b['instance_id'] == 'bk-' + min(b['observation_ids']) for b in books)
    assert inv['metrics']['track_report']['segment_starts'] == ['f00003'] and inv['metrics']['track_report']['segments'] == 2
    assert C.validate_inventory(inv, json.loads((tmp_path / 'manifest.json').read_text())) == []
    # the chain holds: the second-segment groups fold into the first-segment ones and the surviving ids are unchanged
    joined = _merge(tmp_path, frames, {f: ('start' if i == 0 else 'ok') for i, f in enumerate(frames)})
    assert [b['instance_id'] for b in joined['data']['books']] == [f'bk-f00000-o{j:03d}' for j in range(4)]
    assert all(len(b['observation_ids']) == 6 for b in joined['data']['books'])


def test_items_keep_the_earliest_observation_id_after_consolidation(tmp_path):
    frames, obs, status = _segmented_run(tmp_path, break_at=99)
    inv = _merge(tmp_path, frames, status)
    (item,) = inv['data']['items']
    assert item['instance_id'] == 'it-f00000-o004' and item['display_index'] == 1 and len(item['observation_ids']) == 6


def test_both_id_forms_validate_and_the_stable_form_needs_an_observation_id():
    assert C.stable_instance_id('bk', 'f00012-o003') == 'bk-f00012-o003' and C.instance_id('bk', 12) == 'bk-012'
    with pytest.raises(C.ContractError):
        C.stable_instance_id('bk', 'bk-012')
    for ok in ('bk-012', 'bk-1234', 'bk-f00012-o003'):
        assert C._ID_RE['book'].match(ok)
    assert not C._ID_RE['book'].match('bk-f12-o3') and not C._ID_RE['item'].match('bk-f00012-o003')


def test_segment_map_reads_segment_starts_or_reconstructs_them_from_hold_and_held_frames():
    frames = {f: {'timestamp_s': i * 0.2} for i, f in enumerate(['f00000', 'f00001', 'f00002', 'f00003', 'f00004'])}
    assert IN.segment_map(frames, {'segment_starts': ['f00003']}) == {'f00000': 0, 'f00001': 0, 'f00002': 0, 'f00003': 1, 'f00004': 1}
    legacy = {'hold_frames': ['f00001', 'f00002', 'f00003'], 'held_frames': ['f00001', 'f00002']}       # the third consecutive failed step starts a segment
    assert IN.segment_map(frames, legacy) == {'f00000': 0, 'f00001': 0, 'f00002': 0, 'f00003': 1, 'f00004': 1}
    assert set(IN.segment_map(frames, None).values()) == {0}


# ------------------------------------------------------------------------------------------ step 2: the two evidence kinds and the same-frame exclusion
def _group(gid, obs, tier='confident', frame=None):
    return {'instance_id': gid, 'observation_ids': sorted(obs), 'count_tier': tier, 'frame_ref': frame or obs[0].split('-o')[0]}


FRAMES = {f'f{i:05d}': {'timestamp_s': i * 0.2, 'quality': {'sharpness': 100.0}} for i in range(8)}
SEGS = {f: (0 if i < 4 else 1) for i, f in enumerate(FRAMES)}


def test_panorama_overlap_merges_groups_on_one_spine_and_refuses_co_visible_ones():
    books = [_group('bk-f00000-o000', ['f00000-o000', 'f00001-o000']), _group('bk-f00005-o000', ['f00005-o000', 'f00006-o000']),
             _group('bk-f00001-o003', ['f00001-o003']), _group('bk-f00007-o001', ['f00007-o001'], tier='unconfirmed')]
    pc = {'panorama_run': 'r_panorama_v1', 'merge_evidence': [
        {'panorama_spine': 'component_002-s001', 'component': 'component_002', 'box_xyxy': [1, 2, 3, 4], 'confidence': 0.8,
         'groups': [{'id': 'bk-f00000-o000', 'iou': 0.8}, {'id': 'bk-f00005-o000', 'iou': 0.7}, {'id': 'bk-f00001-o003', 'iou': 0.4}, {'id': 'bk-f00007-o001', 'iou': 0.5}]}]}
    art = IN.build({'books': books, 'counts': {'count_range': [2, 4]}}, FRAMES, SEGS, panorama_count=pc, panorama_eligible=True)
    inst = {i['instance_id']: i for i in art['instances']}
    merged = inst['bk-f00000-o000']
    assert merged['member_group_ids'] == ['bk-f00000-o000', 'bk-f00005-o000', 'bk-f00007-o001'] and merged['tier'] == 'confident' and merged['segments'] == [0, 1]
    assert [e['kind'] for e in merged['evidence']] == [IN.KIND_PANORAMA] * 2 and merged['evidence'][0]['panorama_spine'] == 'component_002-s001'
    assert merged['evidence'][0]['iou'] == {'bk-f00000-o000': 0.8, 'bk-f00005-o000': 0.7}
    assert 'bk-f00001-o003' in inst and inst['bk-f00001-o003']['member_group_ids'] == ['bk-f00001-o003']          # co-visible with the anchor in f00001: refused
    (ref,) = art['rejected_unions']
    assert ref['groups'] == ['bk-f00000-o000', 'bk-f00001-o003'] and 'same-frame exclusion' in ref['reason'] and 'f00001' in ref['reason']
    assert {k: art['stats'][k] for k in ('track_groups', 'instances', 'merged_groups', 'merged_instances', 'merged_by', 'unconfirmed_instances')} == {
        'track_groups': 4, 'instances': 2, 'merged_groups': 2, 'merged_instances': 1, 'merged_by': {IN.KIND_PANORAMA: 2}, 'unconfirmed_instances': 0}
    assert art['count'] == {'lower': 1, 'best': 2, 'upper': 2, 'track_upper': 4, 'basis': art['count']['basis']}


def _obs(oid, long_px):
    f = oid.split('-o')[0]
    return {'observation_id': oid, 'frame_id': f, 'kind': 'book', 'polygon_xy': [[0, 0], [20, 0], [20, long_px], [0, long_px]], 'bbox_xyxy': [0, 0, 20, long_px]}


def test_far_view_only_instances_are_unconfirmed_unless_merged_with_a_close_sighting():
    # run median native spine = 250 px (the groups' best polygons: 400, 400, 100, 100); sightings under 125 px are far views (pipeline.panorama_count.FAR_FRACTION)
    books = [dict(_group('bk-f00000-o000', ['f00000-o000', 'f00001-o000']), polygon_xy=_obs('x', 400)['polygon_xy']),
             dict(_group('bk-f00002-o000', ['f00002-o000', 'f00003-o000']), polygon_xy=_obs('x', 400)['polygon_xy']),
             dict(_group('bk-f00004-o000', ['f00004-o000', 'f00005-o000']), polygon_xy=_obs('x', 100)['polygon_xy']),       # the walk-up: every sighting far
             dict(_group('bk-f00006-o000', ['f00006-o000']), polygon_xy=_obs('x', 100)['polygon_xy'])]                    # far, but merged with a close group below
    obs = [_obs('f00000-o000', 400), _obs('f00001-o000', 410), _obs('f00002-o000', 390), _obs('f00003-o000', 400), _obs('f00004-o000', 100), _obs('f00005-o000', 110), _obs('f00006-o000', 100)]
    link = {'groups': ['bk-f00002-o000', 'bk-f00006-o000'], 'observation_ids': ['f00003-o000', 'f00006-o000'], 'frame_ids': ['f00003', 'f00006'], 'segments': [0, 1], 'gap_s': 0.6,
            'inliers': 40, 'appearance_score': 0.7, 'center_distance_spine_widths': 0.1, 'score': 0.69, 'kind': IN.KIND_CROSS, 'transition': 'f00004'}
    art = IN.build({'books': books, 'counts': {'count_range': [3, 4]}}, FRAMES, SEGS, cross={'links': [link], 'rejected': [], 'transitions_total': 1, 'transitions_registered': 1, 'transitions_with_links': 1},
                   observations=obs)
    inst = {i['instance_id']: i for i in art['instances']}
    assert inst['bk-f00004-o000']['tier'] == 'unconfirmed' and inst['bk-f00004-o000']['far_only'] and 'far view only' in inst['bk-f00004-o000']['tier_reason']
    assert inst['bk-f00002-o000']['member_group_ids'] == ['bk-f00002-o000', 'bk-f00006-o000'] and inst['bk-f00002-o000']['tier'] == 'confident' and not inst['bk-f00002-o000']['far_only']
    assert inst['bk-f00000-o000']['tier'] == 'confident'
    assert art['stats']['far_only_instances'] == 1 and art['stats']['far_only_demoted'] == 1 and art['stats']['median_native_spine_px'] == 250.0
    assert art['count']['best'] == 2 and art['count']['lower'] == 1                          # the merged instance supports the lower endpoint; the two-sighting single-segment one does not
    man = _manifest(tuple(FRAMES))
    inv = {'data': {'books': [{**b, 'frame_ref': b['frame_ref'], 'bbox_xyxy': [100, 100, 140, 400], 'kind': 'book', 'confidence': 0.9, 'uncertain_association': False,
                               'identity': {'decision': 'no_match'}} for b in books], 'items': [], 'registrations': [], 'counts': {'count_range': [3, 4]}}, 'fixture': False}
    data = P.build_data(inv, man, {'run_id': 't', 'instances': art})
    assert [b['id'] for b in data['unconfirmed_books']] == ['bk-f00004-o000'] and any('far view only' in i for i in data['unconfirmed_books'][0]['issues'])
    assert data['totals']['book_count'] == 2 and data['count']['instances']['far_only_unconfirmed'] == 1


def _rect(oid, w, h):
    return {'observation_id': oid, 'frame_id': oid.split('-o')[0], 'kind': 'book', 'polygon_xy': [[0, 0], [w, 0], [w, h], [0, h]], 'bbox_xyxy': [0, 0, w, h]}


def test_cover_view_only_instances_are_unconfirmed_unless_one_sighting_is_a_spine_view():
    # Cover-view rule: polygon long/short under COVER_VIEW_MAX_ASPECT (2.0) is a cover/flat view. Same long side everywhere, so the far rule never fires here.
    books = [dict(_group('bk-f00000-o000', ['f00000-o000', 'f00001-o000']), polygon_xy=_rect('x', 40, 400)['polygon_xy']),        # spine views: a line
             dict(_group('bk-f00002-o000', ['f00002-o000', 'f00003-o000']), polygon_xy=_rect('x', 330, 400)['polygon_xy']),       # every sighting a cover (vase): unconfirmed
             dict(_group('bk-f00004-o000', ['f00004-o000', 'f00005-o000']), polygon_xy=_rect('x', 330, 400)['polygon_xy']),       # one cover + one spine view: a line
             dict(_group('bk-f00006-o000', ['f00006-o000']), polygon_xy=_rect('x', 330, 400)['polygon_xy']),                       # cover only, merged with the spine group below
             dict(_group('bk-f00007-o000', ['f00007-o000']), polygon_xy=_rect('x', 30, 400)['polygon_xy'])]
    obs = [_rect('f00000-o000', 40, 400), _rect('f00001-o000', 45, 400), _rect('f00002-o000', 330, 400), _rect('f00003-o000', 300, 400), _rect('f00004-o000', 330, 400),
           _rect('f00005-o000', 60, 400), _rect('f00006-o000', 380, 400), _rect('f00007-o000', 30, 400)]
    assert IN.cover_view_sightings(obs) == {'f00002-o000', 'f00003-o000', 'f00004-o000', 'f00006-o000'}
    assert IN.cover_view_sightings([{**obs[2], 'polygon_xy': None}]) == set()                   # no polygon: never a cover view (no demotion on missing evidence)
    link = {'groups': ['bk-f00006-o000', 'bk-f00007-o000'], 'observation_ids': ['f00006-o000', 'f00007-o000'], 'frame_ids': ['f00006', 'f00007'], 'segments': [1, 1], 'gap_s': 0.2,
            'inliers': 40, 'appearance_score': 0.7, 'center_distance_spine_widths': 0.1, 'score': 0.69, 'kind': IN.KIND_CROSS, 'transition': 'f00006'}
    art = IN.build({'books': books, 'counts': {'count_range': [4, 5]}}, FRAMES, SEGS, cross={'links': [link], 'rejected': [], 'transitions_total': 1, 'transitions_registered': 1,
                   'transitions_with_links': 1}, observations=obs)
    inst = {i['instance_id']: i for i in art['instances']}
    vase = inst['bk-f00002-o000']
    assert vase['tier'] == 'unconfirmed' and vase['cover_only'] and not vase['far_only'] and vase['tier_reason'].startswith('cover or flat view only') and '2.0:1' in vase['tier_reason']
    assert inst['bk-f00004-o000']['tier'] == 'confident' and not inst['bk-f00004-o000']['cover_only']
    assert inst['bk-f00006-o000']['member_group_ids'] == ['bk-f00006-o000', 'bk-f00007-o000'] and inst['bk-f00006-o000']['tier'] == 'confident'
    assert inst['bk-f00000-o000']['tier'] == 'confident'
    assert {k: art['stats'][k] for k in ('cover_only_instances', 'cover_only_demoted', 'far_only_demoted', 'cover_view_max_aspect')} == {
        'cover_only_instances': 1, 'cover_only_demoted': 1, 'far_only_demoted': 0, 'cover_view_max_aspect': 2.0}
    assert art['count']['best'] == 3
    man = _manifest(tuple(FRAMES))
    inv = {'data': {'books': [{**b, 'frame_ref': b['frame_ref'], 'bbox_xyxy': [100, 100, 140, 400], 'kind': 'book', 'confidence': 0.9, 'uncertain_association': False,
                               'identity': {'decision': 'no_match'}} for b in books], 'items': [], 'registrations': [], 'counts': {'count_range': [4, 5]}}, 'fixture': False}
    data = P.build_data(inv, man, {'run_id': 't', 'instances': art})
    assert [b['id'] for b in data['unconfirmed_books']] == ['bk-f00002-o000'] and any(i.startswith('cover or flat view only') for i in data['unconfirmed_books'][0]['issues'])
    assert data['totals']['book_count'] == 3 and data['count']['instances']['cover_view_unconfirmed'] == 1 and data['count']['instances']['far_only_unconfirmed'] == 0


def test_the_lower_endpoint_counts_instances_placed_on_a_normal_panorama_piece_only_when_the_panorama_is_eligible():
    books = [_group('bk-f00000-o000', ['f00000-o000', 'f00001-o000']), _group('bk-f00002-o000', ['f00002-o000', 'f00003-o000']), _group('bk-f00005-o000', ['f00005-o000'])]
    placed = {'bk-f00000-o000', 'bk-f00005-o000'}
    on = IN.build({'books': books}, FRAMES, SEGS, panorama_count={'merge_evidence': []}, panorama_eligible=True, placed_in_normal_piece=placed)
    off = IN.build({'books': books}, FRAMES, SEGS, panorama_count={'merge_evidence': []}, panorama_eligible=False, placed_in_normal_piece=placed)
    assert on['count']['lower'] == 1 and off['count']['lower'] == 0 and on['stats']['in_normal_piece'] == 2 and off['stats']['in_normal_piece'] == 0
    assert {i['instance_id']: i['lower_support'] for i in on['instances']} == {'bk-f00000-o000': True, 'bk-f00002-o000': False, 'bk-f00005-o000': False}   # one sighting is never enough


def test_panorama_evidence_is_ignored_when_the_panorama_estimator_is_not_eligible():
    books = [_group('bk-f00000-o000', ['f00000-o000']), _group('bk-f00005-o000', ['f00005-o000'])]
    pc = {'merge_evidence': [{'panorama_spine': 's', 'groups': [{'id': 'bk-f00000-o000', 'iou': 0.8}, {'id': 'bk-f00005-o000', 'iou': 0.7}]}]}
    art = IN.build({'books': books}, FRAMES, SEGS, panorama_count=pc, panorama_eligible=False, panorama_reason='count reliability is high')
    assert art['stats']['merged_groups'] == 0 and art['panorama'] == {'eligible': False, 'reason': 'count reliability is high', 'panorama_run': None, 'spines_with_several_groups': 0, 'unions': 0}


def test_cross_segment_links_merge_with_their_evidence_line_and_every_merge_keeps_the_earliest_id():
    books = [_group('bk-f00002-o000', ['f00002-o000', 'f00003-o000']), _group('bk-f00004-o000', ['f00004-o000', 'f00005-o000'])]
    link = {'groups': ['bk-f00004-o000', 'bk-f00002-o000'], 'observation_ids': ['f00003-o000', 'f00004-o000'], 'frame_ids': ['f00003', 'f00004'], 'segments': [0, 1], 'gap_s': 0.2,
            'inliers': 66, 'appearance_score': 0.71, 'center_distance_spine_widths': 0.1, 'score': 0.69, 'kind': IN.KIND_CROSS, 'transition': 'f00004'}
    art = IN.build({'books': books}, FRAMES, SEGS, cross={'links': [link], 'rejected': [], 'transitions_total': 1, 'transitions_registered': 1, 'transitions_with_links': 1})
    (inst,) = art['instances']
    assert inst['instance_id'] == 'bk-f00002-o000' and inst['member_group_ids'] == ['bk-f00002-o000', 'bk-f00004-o000'] and inst['sightings'] == 4
    (ev,) = inst['evidence']
    assert ev['kind'] == IN.KIND_CROSS and ev['inliers'] == 66 and ev['groups'] == ['bk-f00004-o000', 'bk-f00002-o000'] and ev['observation_ids'] == ['f00003-o000', 'f00004-o000']
    assert art['fragmentation']['association_reliability'] == 'high' and art['fragmentation']['duplicate_links_found'] == 1


def test_association_reliability_follows_the_share_of_registered_transitions():
    assert IN._reliability_of(0, 0) == 'high' and IN._reliability_of(10, 8) == 'high' and IN._reliability_of(10, 5) == 'medium' and IN._reliability_of(11, 2) == 'low'
    assert IN.weaker('high', 'low') == 'low' and IN.weaker('medium', 'high') == 'medium' and IN.weaker('unknown', 'high') == 'unknown' and IN.weaker(None, 'low') == 'low'


def _shelf_texture(seed=5):
    rng = np.random.default_rng(seed)
    big = cv2.cvtColor(cv2.GaussianBlur(rng.integers(0, 255, (700, 1700), dtype=np.uint8), (0, 0), 2.5), cv2.COLOR_GRAY2BGR)
    big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
    spines = {}
    for k, x in enumerate([220, 320, 420, 520, 620, 1180]):          # six upright spines on the shelf; the last one is far from the rest
        col = tuple(int(v) for v in rng.integers(30, 225, 3))
        cv2.rectangle(big, (x, 150), (x + 46, 520), col, -1)
        for _ in range(14):                                          # cover texture, different per spine
            cx, cy = int(rng.integers(x + 4, x + 42)), int(rng.integers(160, 510))
            cv2.circle(big, (cx, cy), int(rng.integers(3, 9)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        spines[k] = (x, 150, x + 46, 520)
    return big, spines


def _two_views(tmp_path, big, spines, shift=180, extra=None):
    """Frame f00003 (segment 0) and f00004 (segment 1): crops of the same shelf shifted by ``shift`` px; every spine is its own group in each frame."""
    frames, obs, groups = {}, [], []
    for i, (fid, dx) in enumerate([('f00003', 0), ('f00004', shift)]):
        crop = big[60:660, 100 + dx:1100 + dx]
        cv2.imwrite(str(tmp_path / f'{fid}.png'), crop)
        frames[fid] = {'frame_id': fid, 'timestamp_s': 0.6 + 0.2 * i, 'image_path': f'{fid}.png', 'width_px': 1000, 'height_px': 600, 'quality': {'sharpness': 150.0}}
        for k, (x0, y0, x1, y1) in spines.items():
            x0, x1 = x0 - 100 - dx, x1 - 100 - dx
            if x0 < 0 or x1 > 1000:
                continue
            oid = f'{fid}-o{k:03d}'
            obs.append({'observation_id': oid, 'frame_id': fid, 'kind': 'book', 'bbox_xyxy': [x0, y0 - 60, x1, y1 - 60],
                        'polygon_xy': [[x0, y0 - 60], [x1, y0 - 60], [x1, y1 - 60], [x0, y1 - 60]], 'confidence': 0.9})
            groups.append(_group(f'bk-{oid}', [oid]))
    return frames, obs, groups


def test_cross_segment_registration_links_the_same_spines_across_a_transition_and_nothing_else(tmp_path):
    big, spines = _shelf_texture()
    frames, obs, groups = _two_views(tmp_path, big, spines)
    segments = {'f00003': 0, 'f00004': 1}
    res = IN.cross_segment_links(tmp_path, frames, obs, groups, segments)
    assert res['transitions_total'] == 1 and res['pairs_tested'] == 1 and res['pairs_registered'] == 1 and res['frames_used'] == 2
    pairs = {tuple(l['observation_ids']) for l in res['links']}
    assert pairs == {(f'f00003-o{k:03d}', f'f00004-o{k:03d}') for k in range(1, 5)}, 'the four spines visible in both views link one-to-one; spines 0 and 5 are each outside one view'
    for l in res['links']:
        assert l['kind'] == IN.KIND_CROSS and l['inliers'] >= IN.CROSS['min_inliers'] and l['appearance_score'] >= IN.CROSS['min_appearance'] and l['segments'] == [0, 1]
    assert res['rejected'] == [] and res['seconds'] < 10


def test_identical_adjacent_spines_are_vetoed_as_a_near_tie(tmp_path):
    big, spines = _shelf_texture()
    big[140:530, 414:472] = big[140:530, 314:372]                         # spine 2 becomes a pixel copy of spine 1 (with its margin): two identical copies side by side, both in both views
    frames, obs, groups = _two_views(tmp_path, big, spines)
    tight = IN.cross_segment_links(tmp_path, frames, obs, groups, {'f00003': 0, 'f00004': 1})
    assert len(tight['links']) == 4 and tight['rejected'] == [], 'with the production gate (0.5 spine widths) the registered position alone tells the two copies apart'
    res = IN.cross_segment_links(tmp_path, frames, obs, groups, {'f00003': 0, 'f00004': 1}, {'tol_widths': 2.5, 'geometry_weight': 0.0})   # both copies inside each other's gate, geometry silent
    linked = {o for l in res['links'] for o in l['observation_ids']}
    assert not ({'f00003-o001', 'f00003-o002', 'f00004-o001', 'f00004-o002'} & linked), 'identical copies inside one gate never link'
    assert res['rejected'] and all('near-tie' in r['reason'] for r in res['rejected']) and {tuple(l['observation_ids']) for l in res['links']} == {('f00003-o003', 'f00004-o003'), ('f00003-o004', 'f00004-o004')}


def test_no_transition_means_no_registration_work(tmp_path):
    big, spines = _shelf_texture()
    frames, obs, groups = _two_views(tmp_path, big, spines)
    res = IN.cross_segment_links(tmp_path, frames, obs, groups, {'f00003': 0, 'f00004': 0})
    assert res['transitions_total'] == 0 and res['pairs_tested'] == 0 and res['links'] == [] and res['frames_used'] == 0



def test_short_gap_registration_links_a_book_split_inside_one_segment_and_respects_its_gates(tmp_path):
    """Short-gap registration: two frames 0.2 s apart in ONE segment, every spine its own group (the tracker lost them): the short-gap evidence links the four shared
    spines one-to-one; the cross-segment probe never looks (no transition). Frames further apart than max_gap_s are not compared; groups that already share
    a frame are never linked; build() merges each pair with one short_gap_registration evidence line."""
    big, spines = _shelf_texture()
    frames, obs, groups = _two_views(tmp_path, big, spines)
    segments = {'f00003': 0, 'f00004': 0}
    assert IN.cross_segment_links(tmp_path, frames, obs, groups, segments)['links'] == []
    res = IN.short_gap_links(tmp_path, frames, obs, groups, segments)
    assert res['pairs_in_window'] == 1 and res['pairs_registered'] == 1 and res['rejected'] == []
    assert {tuple(l['observation_ids']) for l in res['links']} == {(f'f00003-o{k:03d}', f'f00004-o{k:03d}') for k in range(1, 5)}
    assert all(l['kind'] == IN.KIND_SHORT_GAP and l['segments'] == [0, 0] and l['gap_s'] <= IN.SHORT_GAP['max_gap_s'] for l in res['links'])
    inv = {'books': groups, 'counts': {}}
    art = IN.build(inv, frames, segments, short_gap=res, run_id='t')
    assert art['stats']['merged_by'] == {IN.KIND_SHORT_GAP: 4} and art['short_gap']['unions'] == 4 and len(art['instances']) == len(groups) - 4
    merged = [i for i in art['instances'] if len(i['member_group_ids']) == 2]
    assert len(merged) == 4 and all(len(i['evidence']) == 1 and i['evidence'][0]['kind'] == IN.KIND_SHORT_GAP for i in merged)
    far = {f: {**v, 'timestamp_s': v['timestamp_s'] + (5.0 if f == 'f00004' else 0.0)} for f, v in frames.items()}
    assert IN.short_gap_links(tmp_path, far, obs, groups, segments)['pairs_in_window'] == 0, 'frames 5 s apart are never compared (no loop closure)'
    together = [g for g in groups if g['instance_id'] not in ('bk-f00003-o001', 'bk-f00004-o001')] + [_group('bk-f00003-o001', ['f00003-o001', 'f00004-o001'])]
    obs2 = obs + [{**o, 'observation_id': 'f00004-o099'} for o in obs if o['observation_id'] == 'f00004-o002']
    together = [g if g['instance_id'] != 'bk-f00003-o002' else _group('bk-f00003-o002', ['f00003-o002', 'f00004-o099']) for g in together]
    res2 = IN.short_gap_links(tmp_path, frames, obs2, together, segments)
    linked = {g for l in res2['links'] for g in l['groups']}
    assert 'bk-f00003-o002' not in linked and 'bk-f00004-o002' not in linked, 'two groups seen together in f00004 are two books: never linked'


def test_revisit_registration_links_a_book_seen_again_and_refuses_a_registration_whose_aligned_spines_differ(tmp_path):
    """Revisit linking: the same two views 10 s apart (a pass that comes back over the shelf) are not short-gap pairs; the revisit evidence registers them and links
    the four shared spines with one revisit_registration line each (gap in seconds, layout share). When most of the spines the registration aligns look
    different (a look-alike cubby: the frame registers on its surroundings, the books are not the same), the layout check drops the whole frame pair.
    Pairs within 3 s and beyond 20 s are never revisit pairs; the flag turns the evidence off."""
    big, spines = _shelf_texture()
    frames, obs, groups = _two_views(tmp_path, big, spines)
    segments = {'f00003': 0, 'f00004': 0}
    assert IN.revisit_links(tmp_path, frames, obs, groups, segments)['pairs_in_window'] == 0, 'frames 0.2 s apart are the short-gap evidence, not a revisit'
    later = {f: {**v, 'timestamp_s': v['timestamp_s'] + (10.0 if f == 'f00004' else 0.0)} for f, v in frames.items()}
    assert IN.short_gap_links(tmp_path, later, obs, groups, segments)['pairs_in_window'] == 0
    res = IN.revisit_links(tmp_path, later, obs, groups, segments, workers=2)
    assert res['pairs_in_window'] == 1 and res['pairs_registered'] == 1 and res['rejected'] == [] and res['layout_rejected'] == []
    assert {tuple(l['observation_ids']) for l in res['links']} == {(f'f00003-o{k:03d}', f'f00004-o{k:03d}') for k in range(1, 5)}
    assert all(l['kind'] == IN.KIND_REVISIT and l['gap_s'] == 10.2 and l['layout'] == [4, 4] and l['appearance_score'] >= IN.CROSS['min_appearance'] for l in res['links'])
    art = IN.build({'books': groups, 'counts': {}}, later, segments, revisit=res, run_id='t')
    assert art['stats']['merged_by'] == {IN.KIND_REVISIT: 4} and art['revisit']['unions'] == 4 and len(art['instances']) == len(groups) - 4
    too_far = {f: {**v, 'timestamp_s': v['timestamp_s'] + (25.0 if f == 'f00004' else 0.0)} for f, v in frames.items()}
    assert IN.revisit_links(tmp_path, too_far, obs, groups, segments)['pairs_in_window'] == 0, 'beyond 20 s: never compared (no loop closure)'
    rng = np.random.default_rng(11)
    other = big.copy()
    for k in (2, 3, 4):                                                    # the same shelf around them, but three of the four shared spines are other books
        x0, y0, x1, y1 = spines[k]
        other[y0:y1, x0:x1] = tuple(int(v) for v in rng.integers(30, 225, 3))
        for _ in range(14):
            cv2.circle(other, (int(rng.integers(x0 + 4, x1 - 4)), int(rng.integers(y0 + 10, y1 - 10))), int(rng.integers(3, 9)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    cv2.imwrite(str(tmp_path / 'f00004.png'), other[60:660, 280:1280])
    res2 = IN.revisit_links(tmp_path, later, obs, groups, segments, workers=2)
    assert res2['pairs_registered'] == 1 and res2['links'] == [], 'one alike spine out of four aligned: the registration is not trusted'
    assert [(r['alike_spines'], r['aligned_spines']) for r in res2['layout_rejected']] == [(1, 4)]
    assert IN.revisit_links(tmp_path, later, obs, groups, segments, {'min_layout_share': 0.25})['links'][0]['observation_ids'] == ['f00003-o001', 'f00004-o001']
    with patch.dict('os.environ', {IN.REVISIT_ENV: '0'}):
        assert not IN.revisit_enabled() and IN.revisit_enabled(True)
    with patch.dict('os.environ', {}, clear=True):
        assert IN.revisit_enabled(), 'on by default (the user, 2026-10-08)'


def test_stretch_appearance_compares_only_the_common_stretch_and_is_off_by_default(monkeypatch):
    """Common-stretch appearance: a box truncated to the top half of a spine fails the whole-box appearance against the full box (different stretches of the cover) but
    passes over the common stretch; a different spine still fails; frames are aligned (H = identity). The second chance is off unless the flag is set."""
    rng = np.random.default_rng(49)
    img = np.zeros((500, 400, 3), np.uint8)
    img[50:450, 100:140] = np.repeat(rng.integers(0, 255, (400, 1, 3), dtype=np.uint8), 40, axis=1)    # spine 1: random colour bands along the spine
    img[50:450, 160:200] = np.repeat(rng.integers(0, 255, (400, 1, 3), dtype=np.uint8), 40, axis=1)    # spine 2: another book
    full = [[100, 50], [140, 50], [140, 450], [100, 450]]
    top = [[100, 50], [140, 50], [140, 250], [100, 250]]
    other = [[160, 50], [200, 50], [200, 250], [160, 250]]
    whole = cv.appearance(cv._patch(img, full), cv._patch(img, top))
    st = IN.stretch_appearance(img, full, img, top, np.eye(3))
    assert whole < 0.5 <= st and st > 0.95
    assert IN.stretch_appearance(img, full, img, other, np.eye(3)) < 0.5
    assert IN.stretch_appearance(img, full, img, [[100, 600], [140, 600], [140, 700], [100, 700]], np.eye(3)) is None, 'no common stretch: no score'
    monkeypatch.delenv(IN.SHORT_GAP_STRETCH_ENV, raising=False)
    assert not IN.short_gap_stretch_enabled()
    monkeypatch.setenv(IN.SHORT_GAP_STRETCH_ENV, '1')
    assert IN.short_gap_stretch_enabled()

YT11 = pytest.importorskip('pathlib').Path(__file__).resolve().parents[1] / 'runs/sweep_3af346a2f3f14f72'


@pytest.mark.skipif(not (YT11 / 'candidates.json').is_file() or not (YT11 / 'frames' / 'f00633.jpg').is_file(), reason='YugoTest11 live run not on this machine')
def test_yugotest11_confirmed_cross_segment_pairs_become_one_instance_each():
    """The audit's two visually confirmed duplicates (a live count audit): frames f00627 (segment 10) and f00633 (segment 11).
    Anchored on observation ids; nothing in the run is written (the artifact is built in memory)."""
    inv = json.loads((YT11 / 'inventory.json').read_text())
    manifest = json.loads((YT11 / 'manifest.json').read_text())['data']
    frames = {f['frame_id']: f for f in manifest['frames']}
    segments = IN.segment_map(frames, inv['metrics']['track_report'])
    assert segments['f00627'] + 1 == segments['f00633']
    books = inv['data']['books']
    by_obs = {o: b['instance_id'] for b in books for o in b['observation_ids']}
    pairs = [('f00627-o006', 'f00633-o000'), ('f00627-o001', 'f00633-o008')]
    assert all(by_obs[a] != by_obs[b] for a, b in pairs), 'precondition: the track association keeps each pair as two groups'
    obs = json.loads((YT11 / 'candidates.json').read_text())['data']['observations']
    res = IN.cross_segment_links(YT11, frames, obs, books, segments)
    art = IN.build(inv['data'], frames, segments, cross=res, run_id=YT11.name)
    member_of = {m: i['instance_id'] for i in art['instances'] for m in i['member_group_ids']}
    inst = {i['instance_id']: i for i in art['instances']}
    for a, b in pairs:
        assert member_of[by_obs[a]] == member_of[by_obs[b]], (a, b)
        (ev,) = [e for e in inst[member_of[by_obs[a]]]['evidence'] if set(e['groups']) == {by_obs[a], by_obs[b]}]
        assert ev['kind'] == IN.KIND_CROSS and ev['segments'] == [segments['f00627'], segments['f00633']] and ev['inliers'] >= IN.CROSS['min_inliers'] and ev['appearance_score'] >= 0.5
    assert res['seconds'] < 60


# ------------------------------------------------------------------------------------------ step 3: packet lines = instances
from pipeline import packet as P
from app import post_sweep as PS
from app.build_jobs import BuildJobs
from pipeline import panorama_attach as PA
from tests.test_measurement import manifest as _manifest


def _inventory(tiers=('confident', 'confident', 'likely', 'unconfirmed'), identities=None):
    ids = ['f00000-o000', 'f00000-o001', 'f00001-o000', 'f00001-o001']
    books = []
    for i, (oid, t) in enumerate(zip(ids, tiers)):
        ident = (identities or {}).get(oid) or {'decision': 'no_match', 'visible_text': f'text {i}'}
        books.append({'instance_id': 'bk-' + oid, 'display_index': i + 1, 'observation_ids': [oid], 'frame_ref': oid.split('-o')[0], 'bbox_xyxy': [100 + 50 * i, 100, 140 + 50 * i, 400],
                      'kind': 'book', 'confidence': 0.9, 'uncertain_association': False, 'identity': ident, 'count_tier': t})
    counts = {'count_range': [2, 4], 'count_best': 3, 'confident_groups': 2, 'singleton_groups': 4, 'unconfirmed_groups': 1, 'count_reliability': 'medium', 'count_lower': 4, 'count_upper': 4,
              'count_if_near_ties_merged': 3}
    return {'data': {'books': books, 'items': [], 'registrations': [], 'counts': counts, 'item_candidates': []}, 'fixture': False}


def _art(members, run_id='t', **kw):
    """An instances artifact whose first instance merges ``members`` (group ids); every other group is its own instance."""
    all_ids = ['bk-f00000-o000', 'bk-f00000-o001', 'bk-f00001-o000', 'bk-f00001-o001']
    inst = [{'instance_id': members[0], 'member_group_ids': list(members), 'observation_ids': [m[3:] for m in members], 'observation_segments': [int(m[4:9]) for m in members],
             'sightings': len(members), 'segments': sorted({int(m[4:9]) for m in members}), 'tier': 'confident', 'representative_group_id': members[0],
             'evidence': [{'kind': IN.KIND_CROSS, 'groups': list(members), 'inliers': 40, 'appearance_score': 0.8, 'frame_ids': ['f00000', 'f00001'], 'segments': [0, 1]}], 'evidence_kinds': [IN.KIND_CROSS]}]
    for g in all_ids:
        if g not in members:
            inst.append({'instance_id': g, 'member_group_ids': [g], 'observation_ids': [g[3:]], 'observation_segments': [int(g[4:9])], 'sightings': 1, 'segments': [int(g[4:9])],
                         'tier': 'unconfirmed' if g == 'bk-f00001-o001' else 'confident', 'representative_group_id': g, 'evidence': [], 'evidence_kinds': []})
    return {'schema_version': 1, 'artifact_type': 'instances', 'run_id': run_id, 'instances': inst, 'rejected_unions': [],
            'stats': {'track_groups': 4, 'instances': len(inst), 'merged_groups': 4 - len(inst), 'merged_by': {IN.KIND_CROSS: len(members) - 1}},
            'count': {'lower': 1, 'best': len(inst) - 1, 'upper': 3, 'track_upper': 4, 'basis': 'test basis'},
            'fragmentation': {'segments': 2, 'transitions': 1, 'transitions_registered': 1, 'association_reliability': 'high'}, 'panorama': {'eligible': False}, **kw}


def test_packet_lines_are_instances_with_members_evidence_and_coherent_totals():
    man = _manifest(('f00000', 'f00001'))
    data = P.build_data(_inventory(), man, {'run_id': 't', 'instances': _art(['bk-f00000-o000', 'bk-f00001-o000'])})
    t, c = data['totals'], data['count']
    assert [b['id'] for b in data['books']] == ['bk-f00000-o000', 'bk-f00000-o001'] and [b['id'] for b in data['unconfirmed_books']] == ['bk-f00001-o001']
    merged = data['books'][0]
    assert merged['member_group_ids'] == ['bk-f00000-o000', 'bk-f00001-o000'] and merged['sightings'] == 2 and merged['count_tier'] == 'confident'
    assert merged['evidence']['observation_ids'] == ['f00000-o000', 'f00001-o000'] and merged['merge_evidence'][0]['kind'] == IN.KIND_CROSS
    assert [m['id'] for m in merged['merged_members']] == ['bk-f00000-o000', 'bk-f00001-o000'] and any('Merged from 2 track groups' in i for i in merged['issues'])
    assert (t['book_count'], t['book_count_track'], t['count_estimator'], t['book_count_range']) == (2, 4, 'instances', [1, 3])      # upper = track upper 4 - 1 merged group (the unconfirmed singleton stays inside the upper)
    assert t['count_reliability'] == 'medium' and t['count_reliability_association'] == 'high' and t['instances_merged_groups'] == 1
    assert c['uncertainty_band'] == {'low': 1, 'best': 2, 'high': 3, 'note': c['uncertainty_band']['note']} and c['estimator'] == 'instances' and c['best_guess'] == 2
    assert 'no per-book truth' in c['instances']['note'] and any(r['code'] == 'count_estimator_instances' for r in data['review_queue'])
    env = {'data': data}
    assert [x for x in P.verify_packet(env, man, '/nonexistent') if 'frame_ref' not in x] == []


def test_apply_is_idempotent_and_an_exclusion_of_one_member_excludes_the_instance():
    man = _manifest(('f00000', 'f00001'))
    art = _art(['bk-f00000-o000', 'bk-f00001-o000'])
    data = P.build_data(_inventory(), man, {'run_id': 't', 'instances': art})
    before = json.dumps(data['books'], sort_keys=True)
    IN.apply(data, art)
    assert json.dumps(data['books'], sort_keys=True) == before and data['totals']['book_count'] == 2
    ledger = [{'id': 'c001', 'kind': 'exclude', 'target': {'observation_ids': ['f00001-o000']}, 'value': {'reason': 'not mine'}, 'source': {}, 'active': True}]
    data = P.build_data(_inventory(), man, {'run_id': 't', 'instances': art, 'corrections': ledger})
    assert [b['id'] for b in data['books']] == ['bk-f00000-o001'] and sorted(x['instance_id'] for x in data['excluded_by_user']) == ['bk-f00000-o000', 'bk-f00001-o000']
    assert data['totals']['user_excluded_count'] == 2 and any('member of excluded instance' in x['reason'] for x in data['excluded_by_user'])
    restored = P.build_data(_inventory(), man, {'run_id': 't', 'instances': art, 'corrections': ledger + [{'id': 'c002', 'kind': 'restore', 'target': {'observation_ids': ['f00001-o000']}, 'value': {}, 'source': {}, 'active': True}]})
    assert [b['id'] for b in restored['books']] == ['bk-f00000-o000', 'bk-f00000-o001'] and restored['totals']['user_excluded_count'] == 0


def test_identity_is_the_members_single_accepted_work_or_abstains_when_they_disagree():
    man = _manifest(('f00000', 'f00001'))
    ok = {'decision': 'accepted', 'title': 'Rayuela', 'author': 'Cortázar', 'source_url': 'https://openlibrary.org/works/OL1W', 'matching_evidence': ['x'], 'id_confidence': 0.9}
    other = {**ok, 'title': 'Ficciones', 'source_url': 'https://openlibrary.org/works/OL2W'}
    agree = P.build_data(_inventory(identities={'f00001-o000': ok}), man, {'run_id': 't', 'instances': _art(['bk-f00000-o000', 'bk-f00001-o000'])})
    line = agree['books'][0]
    assert line['status'] == 'identified' and line['title'] == 'Rayuela' and line['identity_source_url'] == ok['source_url'] and agree['totals']['books_identified'] == 1
    clash = P.build_data(_inventory(identities={'f00000-o000': ok, 'f00001-o000': other}), man, {'run_id': 't', 'instances': _art(['bk-f00000-o000', 'bk-f00001-o000'])})
    line = clash['books'][0]
    assert line['status'] == 'unidentified' and line['title'] is None and line['identity_decision'] == 'ambiguous' and any('different catalogue works' in i for i in line['issues'])


def test_panorama_attach_on_an_instance_packet_only_adds_the_cross_check_when_the_panorama_is_not_usable():
    man = _manifest(('f00000', 'f00001'))
    data = P.build_data(_inventory(), man, {'run_id': 't', 'instances': _art(['bk-f00000-o000', 'bk-f00001-o000'])})
    pc = {'artifact_type': 'panorama_count', 'run_id': 't', 'panorama_count': 99, 'usable': False, 'unusable_reason': 'detector too weak', 'registration': {'registered_ratio': 0.95}, 'merge_evidence': []}
    res = PA.apply(data, pc)
    t = data['totals']
    assert res['used'] is False and t['panorama_count'] == 99 and t['book_count'] == t['book_lines'] == 2 and t['count_estimator'] == 'instances' and t['book_count_range'] == [1, 3]
    assert t['headline_estimator'] == 'lines' and data['count']['panorama']['estimator_used'] == 'instances' and data['count']['best_guess'] == 2


def test_a_usable_panorama_is_the_headline_on_an_instance_packet_and_survives_a_rebuild_of_the_lines():
    # Headline = the panorama count, the instance lines stay the lines (book_lines, count.lines); re-applying either layer keeps both
    man = _manifest(('f00000', 'f00001'))
    pc = {'artifact_type': 'panorama_count', 'run_id': 't', 'panorama_count': 3, 'range_panorama': [2, 4], 'usable': True, 'detector_support': 0.9,
          'registration': {'registered_ratio': 0.95}, 'merge_evidence': []}
    art = _art(['bk-f00000-o000', 'bk-f00001-o000'])
    data = P.build_data(_inventory(), man, {'run_id': 't', 'instances': art, 'panorama_count': pc})
    t, c = data['totals'], data['count']
    assert (t['headline_estimator'], t['book_count'], t['book_count_range'], t['book_lines'], t['book_lines_range'], t['count_estimator']) == ('panorama', 3, [2, 4], 2, [1, 3], 'instances')
    assert c['best_guess'] == 3 and c['uncertainty_band']['best'] == 3 and c['lines']['count'] == 2 and c['panorama']['estimator_used'] == 'panorama'
    assert 'detector saw 90%' in c['headline']['reason'] and 'Headline 3 books = the 2D panorama count' in c['headline']['reason'] and "inside the book lines' own range [1, 3]" in c['headline']['reason']
    once = copy.deepcopy(data)
    PA.apply(data, pc)
    assert data == once
    IN.apply(data, art, [])
    assert data == once
    far = {**pc, 'panorama_count': 99, 'range_panorama': [90, 120]}                 # the reconciliation guard: outside the lines' own range -> lines
    PA.apply(data, far)
    assert data['totals']['headline_estimator'] == 'lines' and data['totals']['book_count'] == 2 and 'outside the book lines\' own range [1, 3]' in data['count']['headline']['reason']


def test_the_packet_band_is_coherent_without_an_instance_layer_too():
    data = P.build_data(_inventory(), _manifest(('f00000', 'f00001')), {'run_id': 't'})
    band = data['count']['uncertainty_band']
    assert band['low'] <= band['best'] <= band['high'] and band['best'] == data['totals']['book_count'] == 3 and (band['low'], band['high']) == (2, 4)


# ------------------------------------------------------------------------------------------ step 4: the post-sweep step
from tests.test_post_sweep import builds, fake_command, terminal, wait_for


def test_the_instance_step_runs_after_the_count_and_again_after_the_measurement(tmp_path):
    log = tmp_path / 'order.log'

    def note(name, result):
        def hook(run_id, out):
            with open(log, 'a') as f:
                f.write(f'{name}:{out}\n')
            return result
        return hook
    root, jobs, psb = builds(tmp_path, command=fake_command(log=log), on_count=note('count', {'panorama_count': 5, 'estimator': 'track'}),
                             on_instances=note('instances', {'book_count': 4, 'summary': '4 books (1 of 5 track groups merged: panorama 1, cross-segment 0)'}),
                             on_built=note('measure', {'summary': 'measured'}))
    psb.enqueue('sweep_abc')
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert log.read_text().split() == ['panorama:start', 'count:sweep_abc_panorama_v1', 'instances:sweep_abc_panorama_v1', 'recon3d:start', 'measure:sweep_abc_recon3d_v1', 'instances:sweep_abc_panorama_v1']
    assert st['state'] == 'done' and st['instances']['state'] == 'done' and 'books: 4 books (1 of 5 track groups merged' in st['summary']


def test_a_failing_instance_step_is_recorded_and_never_fails_the_maps(tmp_path):
    def boom(run_id, out):
        raise RuntimeError('instance step failed: no inventory')
    root, jobs, psb = builds(tmp_path, command=fake_command(), on_count=lambda r, p: {'panorama_count': 5, 'estimator': 'track'}, on_instances=boom)
    psb.enqueue('sweep_abc')
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['state'] == 'done' and st['steps']['recon3d']['state'] == 'done' and st['instances']['state'] == 'failed' and 'no inventory' in st['instances']['error']
    assert 'books: failed' in st['summary']


def test_a_restart_runs_the_instance_step_of_a_counted_panorama_that_was_never_merged(tmp_path):
    seen = []
    hooks = dict(on_count=lambda r, p: {'panorama_count': 5, 'estimator': 'track'}, on_instances=lambda r, p: seen.append((r, p)) or {'book_count': 4})
    root, jobs, psb = builds(tmp_path, command=fake_command(), **hooks)
    psb.enqueue('sweep_abc')
    wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    seen.clear()
    rec = json.loads((psb.dir / 'sweep_abc.json').read_text())
    rec.pop('instances')
    (psb.dir / 'sweep_abc.json').write_text(json.dumps(rec))
    psb2 = PS.PostSweepBuilds(root, BuildJobs(root, fake_command()), poll_s=0.03, **hooks)
    assert psb2.resume() >= 1
    wait_for(lambda: (psb2.status('sweep_abc') or {}).get('instances', {}).get('state') == 'done')
    psb2.stop()
    assert seen == [('sweep_abc', 'sweep_abc_panorama_v1')]


def test_without_an_instance_hook_the_queue_behaves_as_before(tmp_path):
    root, jobs, psb = builds(tmp_path, command=fake_command(), on_count=lambda r, p: {'panorama_count': 5, 'estimator': 'track'})
    psb.enqueue('sweep_abc')
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['state'] == 'done' and 'instances' not in st


# ------------------------------------------------------------------------------------------ step 5: split (undo a merge)
from pipeline import corrections as K


def test_a_split_correction_makes_the_member_its_own_line_again_and_the_ledger_logs_it():
    man = _manifest(('f00000', 'f00001'))
    art = _art(['bk-f00000-o000', 'bk-f00001-o000'])
    entry = K.new_entry('c001', 'split', {'observation_ids': ['f00001-o000']}, {'member_group_id': 'bk-f00001-o000'})
    inv, excluded, log = K.apply(_inventory()['data'], [entry], {'f00000': 0.0, 'f00001': 1.0})
    assert len(inv['books']) == 4 and excluded == [] and log == [{'correction_id': 'c001', 'kind': 'split', 'matched_instances': ['bk-f00001-o000']}]
    data = P.build_data(_inventory(), man, {'run_id': 't', 'instances': art, 'corrections': [entry]})
    assert [b['id'] for b in data['books']] == ['bk-f00000-o000', 'bk-f00000-o001', 'bk-f00001-o000'] and all(len(b['member_group_ids']) == 1 for b in data['books'])
    t = data['totals']
    assert (t['book_count'], t['book_count_track'], t['instances_merged_groups']) == (3, 4, 0) and data['count']['instances']['splits'] == [{'instance_id': 'bk-f00000-o000', 'member_group_id': 'bk-f00001-o000'}]
    assert [x for x in P.verify_packet({'data': data}, man, '/nonexistent') if 'frame_ref' not in x] == []
    with pytest.raises(K.CorrectionError):
        K.new_entry('c002', 'split', {}, {})


def _run_on_disk(tmp_path):
    man = _manifest(('f00000', 'f00001'))
    run = tmp_path / 'run'
    (run / 'frames').mkdir(parents=True)
    for f in ('f00000', 'f00001'):
        (run / 'frames' / f'{f}.jpg').write_bytes(b'x')
    C.write_json(run / C.ARTIFACTS['frames'], man)
    env = C.envelope('run', 'packet', P.build_data(_inventory(), man, {'run_id': 'run'}), status='partial')
    assert P.verify_packet(env, man, run) == []
    C.write_json(run / C.ARTIFACTS['packet'], env)
    IN.write(_art(['bk-f00000-o000', 'bk-f00001-o000'], run_id='run'), run)
    return run, man


def test_attach_in_place_is_verified_idempotent_and_a_split_entry_persists_in_the_packet_ledger(tmp_path):
    run, man = _run_on_disk(tmp_path)
    assert IN.load(run)['run_id'] == 'run'
    res = IN.attach(run, render=False)
    assert res['attached'] and res['book_count'] == 2 and (run / IN.BACKUP).is_file()
    p = C.read_json(run / C.ARTIFACTS['packet'])['data']
    assert [b['id'] for b in p['books']] == ['bk-f00000-o000', 'bk-f00000-o001'] and p['totals']['count_estimator'] == 'instances'
    again = IN.attach(run, render=False)
    assert again['attached'] and C.read_json(run / C.ARTIFACTS['packet'])['data']['books'] == p['books']
    entry = K.new_entry('c001', 'split', {'observation_ids': ['f00001-o000']}, {'member_group_id': 'bk-f00001-o000'})
    split = IN.attach(run, render=False, extra_ledger=[entry])
    assert split['attached'] and split['book_count'] == 3 and split['splits'] == 1
    p = C.read_json(run / C.ARTIFACTS['packet'])['data']
    assert [b['id'] for b in p['books']] == ['bk-f00000-o000', 'bk-f00000-o001', 'bk-f00001-o000'] and [e['id'] for e in p['corrections']['ledger']] == ['c001']
    assert IN.attach(run, render=False)['book_count'] == 3, 'the split stays: the ledger lives in the packet'
    assert P.verify_packet(C.read_json(run / C.ARTIFACTS['packet']), man, run) == []


def test_a_packet_that_would_fail_verification_is_never_written(tmp_path, monkeypatch):
    run, man = _run_on_disk(tmp_path)
    before = (run / C.ARTIFACTS['packet']).read_bytes()
    monkeypatch.setattr(P, 'verify_packet', lambda *a, **k: ['boom'])
    res = IN.attach(run, render=False)
    assert res['attached'] is False and res['problems'] == ['boom'] and (run / C.ARTIFACTS['packet']).read_bytes() == before


# ------------------------------------------------------------------------------------------ reading-contradiction veto (precision guard; titles never create a link)
def test_text_normalisation_and_overlap():
    assert IN.text_tokens('Crimen y Castigo — Dostoïevski') == {'crimen', 'castigo', 'dostoievski'}
    assert IN.text_overlap('CRIMEN Y CASTIGO', 'crimen castigo, Dostoievski') == 1.0
    assert IN.text_overlap('Un humanista en la Espana de Franco', 'Little Women Louisa') < 0.3
    assert IN.text_overlap('', 'abc def') is None and IN.text_overlap('x', 'abcd') is None
    assert IN.text_overlap('arquitectu', 'arquitectura ]') == 1.0 and IN.text_overlap('_ NIVds=', 'LNIVdsF=') == 1.0, 'OCR truncation/garble of one title is agreement'
    assert IN.text_overlap('2 NADA PERSONAL; SOLO NEGOCIOS', 'Ic_ MARCA PAIS') < 0.3


def test_contradicting_readings_veto_a_merge_and_agreeing_or_missing_readings_do_not():
    books = [_group('bk-f00000-o000', ['f00000-o000', 'f00001-o000']), _group('bk-f00005-o000', ['f00005-o000', 'f00006-o000']),
             _group('bk-f00002-o000', ['f00002-o000']), _group('bk-f00007-o000', ['f00007-o000'])]
    link = lambda a, b, oa, ob: {'groups': [a, b], 'observation_ids': [oa, ob], 'frame_ids': [oa.split('-o')[0], ob.split('-o')[0]], 'segments': [0, 1], 'gap_s': 1.0,
                                 'inliers': 50, 'appearance_score': 0.7, 'center_distance_spine_widths': 0.1, 'score': 0.69, 'kind': IN.KIND_CROSS, 'transition': 'f00004'}
    cross = {'links': [link('bk-f00000-o000', 'bk-f00005-o000', 'f00001-o000', 'f00005-o000'), link('bk-f00002-o000', 'bk-f00007-o000', 'f00002-o000', 'f00007-o000')],
             'rejected': [], 'transitions_total': 1, 'transitions_registered': 1, 'transitions_with_links': 1}
    readings = [{'observation_id': 'f00000-o000', 'visible_text': 'Crimen y castigo', 'read_confidence': 0.8}, {'observation_id': 'f00001-o000', 'visible_text': 'short', 'read_confidence': 0.9},
                {'observation_id': 'f00006-o000', 'visible_text': 'Un humanista en la Espana de Franco', 'read_confidence': 0.7},
                {'observation_id': 'f00002-o000', 'visible_text': 'LITTLE WOMEN Louisa May Alcott', 'read_confidence': 0.6}, {'observation_id': 'f00007-o000', 'visible_text': 'Little Women', 'read_confidence': 0.5}]
    art = IN.build({'books': books}, FRAMES, SEGS, cross=cross, readings=readings)
    inst = {i['instance_id']: i for i in art['instances']}
    assert 'bk-f00005-o000' in inst and inst['bk-f00000-o000']['member_group_ids'] == ['bk-f00000-o000'], 'different titles on both sides: kept apart'
    (veto,) = [r for r in art['rejected_unions'] if 'reading contradiction' in r['reason']]
    assert veto['groups'] == ['bk-f00000-o000', 'bk-f00005-o000'] and veto['texts'] == ['Crimen y castigo', 'Un humanista en la Espana de Franco'] and veto['token_overlap'] == 0.0
    assert inst['bk-f00002-o000']['member_group_ids'] == ['bk-f00002-o000', 'bk-f00007-o000'], 'agreeing readings (the short one inside the long one) merge'
    assert art['stats']['reading_vetoes'] == 1 and art['stats']['groups_with_readable_text'] == 4
    silent = IN.build({'books': books}, FRAMES, SEGS, cross=cross, readings=[readings[0]])
    assert {i['instance_id'] for i in silent['instances']} == {'bk-f00000-o000', 'bk-f00002-o000'}, 'a reading on one side only never vetoes (and never merges)'


# ------------------------------------------------------------------------------------------ acceptance rule for panorama_overlap unions
def _spine(groups):
    return {'panorama_run': 'r_panorama_v1', 'merge_evidence': [{'panorama_spine': 'component_002-s001', 'component': 'component_002', 'box_xyxy': [1, 2, 3, 4], 'confidence': 0.8,
                                                                 'groups': [{'id': g, 'iou': v} for g, v in groups]}]}


def test_a_loose_panorama_placement_is_rejected_with_its_reason_and_an_accepted_union_records_its_features():
    books = [_group('bk-f00000-o000', ['f00000-o000']), _group('bk-f00005-o000', ['f00005-o000']), _group('bk-f00006-o000', ['f00006-o000'])]
    pc = _spine([('bk-f00000-o000', 0.9), ('bk-f00005-o000', 0.6), ('bk-f00006-o000', 0.3)])
    art = IN.build({'books': books}, FRAMES, SEGS, panorama_count=pc, panorama_eligible=True)
    inst = {i['instance_id']: i for i in art['instances']}
    assert inst['bk-f00000-o000']['member_group_ids'] == ['bk-f00000-o000', 'bk-f00005-o000'] and inst['bk-f00006-o000']['member_group_ids'] == ['bk-f00006-o000']
    (ev,) = inst['bk-f00000-o000']['evidence']
    assert ev['acceptance'] == {'min_spine_iou': 0.6, 'frame_check': 'not run', 'rule': 'registration_decides'}
    (rej,) = art['rejected_unions']
    assert rej['groups'] == ['bk-f00000-o000', 'bk-f00006-o000'] and rej['reason'].startswith('panorama placement too loose') and '0.30' in rej['reason']
    assert art['stats']['panorama_acceptance']['outcomes'] == {'accepted_frame_not run': 1, 'placement_too_loose': 1}
    old = IN.build({'books': books}, FRAMES, SEGS, panorama_count=pc, panorama_eligible=True, panorama_rule=False)
    assert len(old['instances']) == 1 and old['rejected_unions'] == [], 'with the rule off every placement on the spine is a union (the behaviour without an acceptance rule)'


def test_a_frame_check_that_contradicts_the_panorama_refuses_the_union_and_an_agreeing_one_is_kept_as_evidence():
    books = [_group('bk-f00000-o000', ['f00000-o000']), _group('bk-f00005-o000', ['f00005-o000']), _group('bk-f00006-o000', ['f00006-o000'])]
    pc = _spine([('bk-f00000-o000', 0.9), ('bk-f00005-o000', 0.6), ('bk-f00006-o000', 0.5)])
    agree = {'outcome': 'agree', 'observation_ids': ['f00000-o000', 'f00005-o000'], 'frame_ids': ['f00000', 'f00005'], 'inliers': 120, 'across_overlap': 0.95,
             'best_spine': 'f00005-o000', 'best_overlap': 0.95}
    contra = {'outcome': 'contradicts', 'observation_ids': ['f00000-o000', 'f00006-o000'], 'frame_ids': ['f00000', 'f00006'], 'inliers': 80, 'across_overlap': 0.1,
              'best_spine': 'f00006-o004', 'best_overlap': 0.9}
    art = IN.build({'books': books}, FRAMES, SEGS, panorama_count=pc, panorama_eligible=True,
                   pan_checks={'bk-f00000-o000|bk-f00005-o000': agree, 'bk-f00000-o000|bk-f00006-o000': contra})
    inst = {i['instance_id']: i for i in art['instances']}
    assert inst['bk-f00000-o000']['member_group_ids'] == ['bk-f00000-o000', 'bk-f00005-o000'] and 'bk-f00006-o000' in inst
    assert inst['bk-f00000-o000']['evidence'][0]['frame_check']['inliers'] == 120 and inst['bk-f00000-o000']['evidence'][0]['acceptance']['frame_check'] == 'agree'
    (rej,) = art['rejected_unions']
    assert rej['reason'].startswith('frame check contradicts the panorama') and 'f00006-o004' in rej['reason'] and rej['frame_check'] == contra


def test_an_agreeing_registration_decides_over_a_loose_placement_and_the_placement_decides_only_without_one():
    books = [_group('bk-f00000-o000', ['f00000-o000']), _group('bk-f00005-o000', ['f00005-o000']), _group('bk-f00006-o000', ['f00006-o000']),
             _group('bk-f00007-o000', ['f00007-o000'])]
    pc = _spine([('bk-f00000-o000', 0.9), ('bk-f00005-o000', 0.2), ('bk-f00006-o000', 0.6), ('bk-f00007-o000', 0.25)])
    agree = {'outcome': 'agree', 'observation_ids': ['f00000-o000', 'f00005-o000'], 'frame_ids': ['f00000', 'f00005'], 'inliers': 90, 'across_overlap': 1.0,
             'best_spine': 'f00005-o000', 'best_overlap': 1.0}
    contra = {'outcome': 'contradicts', 'observation_ids': ['f00000-o000', 'f00006-o000'], 'frame_ids': ['f00000', 'f00006'], 'inliers': 80, 'across_overlap': 0.0,
              'best_spine': 'f00006-o004', 'best_overlap': 0.9}
    checks = {'bk-f00000-o000|bk-f00005-o000': agree, 'bk-f00000-o000|bk-f00006-o000': contra, 'bk-f00000-o000|bk-f00007-o000': {'outcome': 'unregistered', 'pairs_tried': 3}}
    art = IN.build({'books': books}, FRAMES, SEGS, panorama_count=pc, panorama_eligible=True, pan_checks=checks)
    inst = {i['instance_id']: i for i in art['instances']}
    assert inst['bk-f00000-o000']['member_group_ids'] == ['bk-f00000-o000', 'bk-f00005-o000'], 'placement 0.2 but the registered frame pair agrees: one book'
    assert inst['bk-f00000-o000']['evidence'][0]['acceptance']['placement'].startswith('below 0.35')
    reasons = {r['groups'][1]: r['reason'] for r in art['rejected_unions']}
    assert reasons['bk-f00006-o000'].startswith('frame check contradicts') and reasons['bk-f00007-o000'].startswith('panorama placement too loose')
    assert art['stats']['panorama_acceptance']['rule'] == 'registration_decides'
    old = IN.build({'books': books}, FRAMES, SEGS, panorama_count=pc, panorama_eligible=True, pan_checks=checks, panorama_rule='placement_and_frame')
    assert len(old['instances']) == 4, 'placement_and_frame (comparison): the loose placement is refused before the frame check is read'


def test_across_overlap_is_robust_to_a_truncated_detection_and_zero_for_the_neighbour():
    full = np.array([[100, 0], [140, 0], [140, 400], [100, 400]], float)
    half = np.array([[102, 200], [138, 200], [138, 400], [102, 400]], float)      # the same spine, detected on its lower half only
    neighbour = np.array([[145, 0], [185, 0], [185, 400], [145, 400]], float)
    wide = np.array([[100, 0], [185, 0], [185, 400], [100, 400]], float)          # a box spanning both spines
    assert IN.across_overlap(half, full) == 1.0 and IN.across_overlap(full, neighbour) == 0.0
    assert IN.across_overlap(wide, full) == 1.0 and IN.across_overlap(wide, neighbour) == 1.0, 'a two-spine box covers both: the rule then needs B to be the best-covered spine'
    assert IN.across_overlap(np.array([[100, 500], [140, 500], [140, 600], [100, 600]], float), full) == 0.0, 'no overlap along the spine: not the same spine'


def test_panorama_frame_checks_agree_on_the_same_spine_and_contradict_on_the_neighbour(tmp_path):
    big, spines = _shelf_texture()
    frames, obs, groups = _two_views(tmp_path, big, spines)
    pairs = [('bk-f00003-o002', 'bk-f00004-o002'), ('bk-f00003-o002', 'bk-f00004-o003'), ('bk-f00003-o001', 'bk-f00003-o002')]
    res = IN.panorama_frame_checks(tmp_path, frames, obs, groups, pairs)
    same, neighbour, co_visible = (res[f'{a}|{b}'] for a, b in pairs)
    assert same['outcome'] == 'agree' and same['inliers'] >= IN.CROSS['min_inliers'] and same['across_overlap'] >= 0.9 and same['frame_ids'] == ['f00003', 'f00004']
    assert neighbour['outcome'] == 'contradicts' and neighbour['best_spine'] == 'f00004-o002' and neighbour['across_overlap'] == 0.0
    assert co_visible == {'outcome': 'unregistered', 'pairs_tried': 0}, 'two sightings in one frame are never a frame check (the same-frame exclusion refuses them)'


def test_panorama_frame_check_is_ambiguous_not_contradicting_when_another_spine_ties_with_b(tmp_path):
    """A's projected spine covering B AND a second detected spine equally (a narrow detection inside B's span) was 'contradicts' by name order; it is 'ambiguous'."""
    big, spines = _shelf_texture()
    frames, obs, groups = _two_views(tmp_path, big, spines)
    o = next(x for x in obs if x['observation_id'] == 'f00004-o002')
    (x0, y0), (x1, y1) = o['polygon_xy'][0], o['polygon_xy'][2]
    obs.append({'observation_id': 'f00004-o000', 'frame_id': 'f00004', 'kind': 'book', 'bbox_xyxy': [x0 + 10, y0, x1 - 10, y1],
                'polygon_xy': [[x0 + 10, y0], [x1 - 10, y0], [x1 - 10, y1], [x0 + 10, y1]], 'confidence': 0.9})   # sorts before f00004-o002
    groups.append(_group('bk-f00004-o000', ['f00004-o000']))
    res = IN.panorama_frame_checks(tmp_path, frames, obs, groups, [('bk-f00003-o002', 'bk-f00004-o002')])['bk-f00003-o002|bk-f00004-o002']
    assert res['outcome'] == 'ambiguous' and res['tied_spines'] == ['f00004-o000', 'f00004-o002'] and res['across_overlap'] >= 0.9


# ------------------------------------------------------------------------------------------ reconciliation with the panorama count
def _box(x0, x1, y0=0, y1=400):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _reconcile_case(panorama_count=3, eligible=True, frame_check=None):
    """Piece component_001 with five detected spines; one counted group, six unconfirmed single sightings (one of them a cover view)."""
    books = [_group('bk-f00000-o000', ['f00000-o000', 'f00001-o000'])] + [_group(f'bk-f0000{i}-o001', [f'f0000{i}-o001'], tier='unconfirmed') for i in range(2, 8)]
    spines = [{'panorama_spine': f'component_001-s00{k}', 'component': 'component_001', 'confidence': 0.8,
               'members': [{'component': 'component_001', 'conf': 0.8, 'poly': _box(50 * k, 50 * k + 40)}], 'groups': []} for k in range(5)]
    sight = lambda oid, g, poly: {'component': 'component_001', 'observation_id': oid, 'group_id': g, 'poly': poly}
    sightings = [sight('f00000-o000', 'bk-f00000-o000', _box(0, 40)), sight('f00001-o000', 'bk-f00000-o000', _box(2, 42)),   # the counted line on s000
                 sight('f00002-o001', 'bk-f00002-o001', _box(50, 90)),          # s001: free -> promoted
                 sight('f00003-o001', 'bk-f00003-o001', _box(0, 40)),           # s000: carries the counted line -> not promoted
                 sight('f00004-o001', 'bk-f00004-o001', _box(100, 140, 0, 120)),   # s002 at IoU 0.30 -> too loose
                 sight('f00005-o001', 'bk-f00005-o001', _box(150, 190)),        # s003: free, IoU 1.0 -> promoted
                 sight('f00006-o001', 'bk-f00006-o001', _box(155, 195)),        # s003 too, lower IoU -> one line per spine
                 sight('f00007-o001', 'bk-f00007-o001', _box(200, 240))]        # s004: free, but a cover view -> never a candidate
    obs = [{'observation_id': 'f00007-o001', 'frame_id': 'f00007', 'kind': 'book', 'polygon_xy': _box(0, 300), 'bbox_xyxy': [0, 0, 300, 400]}]
    pc = {'panorama_run': 'r_panorama_v1', 'panorama_count': panorama_count, 'merge_evidence': [], 'spines': spines}
    return IN.build({'books': books, 'counts': {'count_range': [1, 10]}}, FRAMES, SEGS, panorama_count=pc, panorama_eligible=eligible, panorama_reason='test reason',
                    observations=obs, piece_sightings=sightings, reconcile_frame_check=frame_check)


def test_reconciliation_moves_toward_the_panorama_never_past_it_and_a_counted_line_in_a_neighbour_frame_blocks_a_promotion():
    one = _reconcile_case(panorama_count=2)                             # room for one line: the stronger placement (s003 at IoU 1.0 before s001 at 1.0 by observation id)
    rec = one['reconciliation']
    assert rec['room'] == 1 and rec['promotions'] == 1 and one['count']['best'] == 2 and rec['outcomes']['no_room_left'] == 1
    assert any('leaves room for 1 more line' in n['reason'] for n in rec['not_promoted'])
    none = _reconcile_case(panorama_count=1)                            # the lines already reach the panorama count: nothing to reconcile
    assert none['reconciliation']['room'] == 0 and none['reconciliation']['promotions'] == 0 and none['count']['best'] == 1
    seen = []

    def check(oid, counted):
        seen.append(oid)
        assert set(counted) == {'f00000-o000', 'f00001-o000'}            # the counted lines' sightings, before any promotion
        if oid == 'f00002-o001':
            return {'outcome': 'occupied', 'frame_id': 'f00001', 'inliers': 80, 'observation_id': 'f00001-o000', 'instance_id': 'bk-f00000-o000', 'across_overlap': 0.95}
        return {'outcome': 'free', 'registered': [{'frame_id': 'f00004', 'inliers': 60, 'covers': None}]}
    art = _reconcile_case(frame_check=check)
    rec = art['reconciliation']
    assert sorted(seen) == ['f00002-o001', 'f00005-o001'] and rec['promotions'] == 1 and rec['outcomes']['counted_in_a_neighbour_frame'] == 1
    assert [p['instance_id'] for p in rec['promoted']] == ['bk-f00005-o001'] and 'no counted line there in 1 registered neighbour frame(s)' in rec['promoted'][0]['evidence']
    assert any('registering f00002 -> f00001 (80 inliers) puts it on counted line bk-f00000-o000' in n['reason'] for n in rec['not_promoted'])


def test_reconciliation_promotes_an_unconfirmed_single_sighting_onto_a_free_detected_spine_with_its_evidence():
    art = _reconcile_case()
    inst = {i['instance_id']: i for i in art['instances']}
    rec = art['reconciliation']
    assert rec['eligible'] and rec['range_before'] == [0, 10] and rec['lines_before'] == 1 and rec['promotions'] == 2 and rec['lines_after'] == 3
    assert rec['abs_diff_before'] == 2 and rec['abs_diff_after'] == 0 and art['count']['best'] == 3
    p = inst['bk-f00002-o001']
    assert p['tier'] == IN.PROMOTED_TIER and p['promotion']['panorama_spine'] == 'component_001-s001'
    assert p['tier_reason'].startswith('promoted: distinct detected spine component_001-s001 in piece component_001, no counted line on it')
    assert inst['bk-f00005-o001']['tier'] == IN.PROMOTED_TIER and inst['bk-f00006-o001']['tier'] == 'unconfirmed'     # one line per spine: the better placement
    assert inst['bk-f00003-o001']['tier'] == 'unconfirmed' and inst['bk-f00004-o001']['tier'] == 'unconfirmed'
    assert inst['bk-f00007-o001']['tier'] == 'unconfirmed' and inst['bk-f00007-o001']['cover_only']                   # cover-view-only is never promoted
    assert rec['outcomes'] == {'promoted': 2, 'spine_carries_a_counted_line': 1, 'placement_too_loose': 1, 'spine_already_promoted': 1, 'far_or_cover_view_only': 1} and rec['candidates'] == 5
    why = {n['instance_id']: n['reason'] for n in rec['not_promoted']}
    assert 'carries counted line(s) bk-f00000-o000' in why['bk-f00003-o001'] and 'IoU 0.30 < 0.35' in why['bk-f00004-o001'] and 'already claimed by bk-f00005-o001' in why['bk-f00006-o001']


def test_reconciliation_guard_needs_an_eligible_panorama_whose_count_lies_inside_the_instance_range():
    outside = _reconcile_case(panorama_count=11)                        # above the instance upper bound (10): the two estimators disagree
    assert outside['reconciliation']['eligible'] is False and 'outside the instance layer' in outside['reconciliation']['reason'] and outside['count']['best'] == 1
    assert all(i['tier'] == 'unconfirmed' for i in outside['instances'] if i['instance_id'] != 'bk-f00000-o000')
    off = _reconcile_case(eligible=False)
    assert off['reconciliation']['eligible'] is False and off['reconciliation']['reason'] == 'panorama not eligible: test reason' and off['count']['best'] == 1
    old = IN.build({'books': [_group('bk-f00002-o001', ['f00002-o001'], tier='unconfirmed')], 'counts': {'count_range': [0, 3]}}, FRAMES, SEGS,
                   panorama_count={'panorama_count': 1, 'merge_evidence': []}, panorama_eligible=True, piece_sightings=[])
    assert 'predates the per-spine list' in old['reconciliation']['reason'] and old['count']['best'] == 0


def test_a_promoted_line_is_counted_with_its_evidence_and_reverts_when_the_artifact_no_longer_promotes_it():
    man = _manifest(('f00000', 'f00001'))
    art = _art(['bk-f00000-o000', 'bk-f00001-o000'])
    ev = 'promoted: distinct detected spine component_001-s007 in piece component_001, no counted line on it (sighting f00001-o001, placement IoU 0.80)'
    promoted = json.loads(json.dumps(art))
    u = next(i for i in promoted['instances'] if i['instance_id'] == 'bk-f00001-o001')
    u.update(tier=IN.PROMOTED_TIER, tier_reason=ev, promotion={'panorama_spine': 'component_001-s007', 'component': 'component_001', 'observation_id': 'f00001-o001', 'iou': 0.8, 'evidence': ev})
    promoted['reconciliation'] = {'eligible': True, 'promotions': 1, 'lines_before': 2, 'lines_after': 3, 'promoted': [{'instance_id': 'bk-f00001-o001'}], 'not_promoted': [{'instance_id': 'x'}]}
    data = P.build_data(_inventory(), man, {'run_id': 't', 'instances': promoted})
    line = next(b for b in data['books'] if b['id'] == 'bk-f00001-o001')
    assert line['count_tier'] == IN.PROMOTED_TIER and line['issues'][-1] == ev and line['promotion']['tier_before'] == 'unconfirmed' and data['totals']['book_count'] == 3
    assert data['count']['instances']['reconciliation']['promotions'] == 1 and 'not_promoted' not in data['count']['instances']['reconciliation']
    IN.apply(data, promoted)                                                                    # idempotent
    assert data['totals']['book_count'] == 3 and next(b for b in data['books'] if b['id'] == 'bk-f00001-o001')['issues'].count(ev) == 1
    IN.apply(data, art)                                                                         # an artifact without the promotion: back to unconfirmed
    assert [b['id'] for b in data['unconfirmed_books']] == ['bk-f00001-o001'] and data['totals']['book_count'] == 2
    assert not any(str(i).startswith('promoted: ') for i in data['unconfirmed_books'][0]['issues']) and 'promotion' not in data['unconfirmed_books'][0]


# ------------------------------------------------------------------------------------------ shelf_plane_position (merged 3D map)
def _cam(fid, cx=0.0):
    """A camera 5 units in front of the shelf plane (z = 0 of the aligned frame), looking straight at it, image y down = world -y."""
    return {'frame_id': fid, 'center': [cx, 0.0, 5.0], 'rotation_world_to_cam': [[1, 0, 0], [0, -1, 0], [0, 0, -1]], 'focal_px': 1000.0, 'width_px': 1000, 'height_px': 1000}


def _plane_marker(gid, oid, x, w=0.2, h=1.0, y=0.0):
    u = lambda X: 1000.0 * X / 5.0 + 500.0
    v = lambda Y: -1000.0 * Y / 5.0 + 500.0
    poly = [[u(x - w / 2), v(y + h / 2)], [u(x + w / 2), v(y + h / 2)], [u(x + w / 2), v(y - h / 2)], [u(x - w / 2), v(y - h / 2)]]
    return {'group_id': gid, 'xyz': [x, y, 0.0], 'method': 'plane_projected', 'sightings': [{'frame_id': oid.split('-o')[0], 'observation_id': oid, 'polygon_xy': poly}]}


def _plane_case():
    specs = [('bk-f00001-o000', 0.0, 1.0), ('bk-f00005-o000', 0.02, 1.0),        # one book, seen in segments 0 and 1
             ('bk-f00005-o001', 0.4, 1.0),                                        # the neighbour, 2 spine widths away: no link
             ('bk-f00006-o000', -0.01, 0.5),                                      # same position, half the height: no link
             ('bk-f00002-o000', 1.0, 1.0), ('bk-f00006-o001', 1.0, 1.0), ('bk-f00007-o000', 1.003, 1.0)]   # two equally good candidates in segment 1
    books = [_group(g, [g[3:]]) for g, _, _ in specs]
    recon = {'cameras': [_cam(f) for f in FRAMES], 'markers': [_plane_marker(g, g[3:], x, h=h) for g, x, h in specs]}
    return books, recon


def test_shelf_plane_links_join_one_spine_position_across_segments_and_nothing_else():
    books, recon = _plane_case()
    out = IN.shelf_plane_links(recon, books, SEGS, model_id='r_recon3d_v1:merged')
    assert out['groups_placed'] == 7
    (lk,) = out['links']
    assert lk['groups'] == ['bk-f00001-o000', 'bk-f00005-o000'] and lk['kind'] == IN.KIND_PLANE and lk['segments'] == [0, 1] and lk['model'] == 'r_recon3d_v1:merged'
    assert lk['across_spine_widths'] == pytest.approx(0.1, abs=1e-3) and lk['along_overlap'] == pytest.approx(1.0) and lk['spine_width_plane'] == [pytest.approx(0.2), pytest.approx(0.2)]
    assert len(out['rejected']) == 2 and all('bk-f00002-o000' in r['groups'] and 'near-tie' in r['reason'] for r in out['rejected'])
    art = IN.build({'books': books}, FRAMES, SEGS, plane=out)
    inst = {i['instance_id']: i for i in art['instances']}
    assert inst['bk-f00001-o000']['member_group_ids'] == ['bk-f00001-o000', 'bk-f00005-o000'] and inst['bk-f00001-o000']['evidence_kinds'] == [IN.KIND_PLANE]
    assert art['stats']['merged_by'] == {IN.KIND_PLANE: 1} and art['shelf_plane']['unions'] == 1 and len(art['instances']) == 6


def test_a_plane_link_between_co_visible_groups_is_refused_and_one_segment_or_no_map_links_nothing(tmp_path):
    books, recon = _plane_case()
    out = IN.shelf_plane_links(recon, books, SEGS, model_id='m')
    books[1]['observation_ids'] = ['f00001-o009', 'f00005-o000']                # the segment-1 group is also seen in f00001, beside the first: two books
    art = IN.build({'books': books}, FRAMES, SEGS, plane=out)
    assert art['shelf_plane']['unions'] == 0 and any('same-frame exclusion' in r['reason'] for r in art['rejected_unions'])
    assert IN.shelf_plane_links(recon, books, SEGS)['links'] == []              # a group spanning two segments is not a candidate
    assert IN.shelf_plane_links(recon, books, {f: 0 for f in FRAMES})['links'] == []
    assert IN.plane_links_for(tmp_path / 'run', books, SEGS)['reason'].startswith('no usable 3D map')
    assert IN.plane_links_for(tmp_path / 'run', books, {f: 0 for f in FRAMES})['reason'].startswith('single tracking segment')


def test_the_best_3d_map_is_the_usable_build_with_the_largest_registered_share(tmp_path, monkeypatch):
    run = tmp_path / 'sweep_x'
    run.mkdir()
    for v, d in {1: {'status': 'ok', 'registered_ratio': 0.5}, 2: {'status': 'failed', 'registered_ratio': 0.9}, 3: {'status': 'partial', 'registered_ratio': 0.8},
                 4: {'status': 'ok', 'registered_ratio': 0.95, 'alignment': {'applied': False}}}.items():
        (tmp_path / f'sweep_x_recon3d_v{v}').mkdir()
        (tmp_path / f'sweep_x_recon3d_v{v}' / 'recon3d.json').write_text(json.dumps({'alignment': {'applied': True}, 'markers': [{}], 'cameras': [{}], **d}))
    assert IN.best_recon3d(run).parent.name == 'sweep_x_recon3d_v3'
    monkeypatch.delenv(IN.PLANE_ENV, raising=False)
    assert IN.plane_merge_enabled() is (IN.PLANE_DEFAULT == '1')
    monkeypatch.setenv(IN.PLANE_ENV, '1')
    assert IN.plane_merge_enabled() and not IN.plane_merge_enabled(False)


def test_round2_the_appearance_gate_and_a_disagreeing_second_build_refuse_a_plane_link():
    books, recon = _plane_case()
    assert IN.shelf_plane_links(recon, books, SEGS, appearance=lambda a, b: 0.3)['links'] == []            # below the production gate 0.5
    out = IN.shelf_plane_links(recon, books, SEGS, appearance=lambda a, b: 0.8)
    assert [lk['groups'] for lk in out['links']] == [['bk-f00001-o000', 'bk-f00005-o000']] and out['links'][0]['appearance_score'] == 0.8
    assert IN.shelf_plane_links(recon, books, SEGS, appearance=lambda a, b: None)['candidates_dropped'].get('appearance_unavailable')
    moved = json.loads(json.dumps(recon))
    moved['markers'][1]['xyz'][0] = 0.3                                                                    # the second build puts the segment-1 sighting 1.5 widths over
    out = IN.shelf_plane_links(recon, books, SEGS, agree_with=[('v3', moved)])
    assert out['links'] == [] and out['agreement_models'] == ['v3']
    assert [lk['groups'] for lk in IN.shelf_plane_links(recon, books, SEGS, agree_with=[('v3', recon)])['links']] == [['bk-f00001-o000', 'bk-f00005-o000']]


def test_the_first_pass_judges_the_panorama_on_the_instance_layers_own_reliability():
    from pipeline import panorama_attach as PA
    pc = {'usable': True, 'panorama_count': 545, 'registration': {'registered_ratio': 0.97}}
    totals = {'count_reliability': 'high'}
    assert PA.decide(totals, pc)[0] is False                                         # the packet's capture reading alone refuses it (YugoTest14 first pass)
    unbridged = {'transitions_total': 5, 'transitions_registered': 0}
    own = IN.own_reliability_totals(totals, unbridged)
    assert own['count_reliability'] == 'low' and totals == {'count_reliability': 'high'}   # the input is not modified
    used, why = PA.decide(own, pc)
    assert used and 'association low (0 of 5 tracking breaks registered across)' in why and 'packet high' in why
    bridged = {'transitions_total': 5, 'transitions_registered': 5}
    assert IN.own_reliability_totals(totals, bridged) is totals                          # nothing weaker: the packet's reading stands, the reason text is unchanged
    assert IN.own_reliability_totals(totals, None) is totals
    assert IN.own_reliability_totals({'count_reliability': 'low'}, unbridged) == {'count_reliability': 'low'}
