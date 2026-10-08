"""The chain-independent panorama count (pipeline.panorama_count), its packet attach (pipeline.panorama_attach) and the post-sweep hook.
Synthetic pieces, polygons and inventories only: nothing here says anything about accuracy on real shelves."""
import copy
import json

import cv2
import numpy as np
import pytest

from app import post_sweep as PS
from app.build_jobs import BuildJobs
from pipeline import contracts as C
from pipeline import packet as P
from pipeline import panorama_attach as PA
from pipeline import panorama_count as PC
from tests import test_auto_attach as TAA
from tests import test_count_range as TCR
from tests.test_post_sweep import POLL, builds, fake_command, terminal, wait_for


def rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ------------------------------------------------------------------------------------------ geometry and merging
def test_spines_on_top_of_each_other_merge_and_neighbours_do_not():
    spines = [{'poly': rect(100, 0, 140, 400), 'comp': 'a'}, {'poly': rect(104, 10, 142, 396), 'comp': 'a'},     # the same spine twice
              {'poly': rect(150, 0, 190, 400), 'comp': 'a'}]                                                         # its neighbour
    assert sorted(len(c) for c in PC.merge_spines(spines)) == [1, 2]


def test_cross_piece_threshold_is_looser_than_inside_a_piece():
    a, b = rect(100, 0, 140, 400), rect(120, 0, 160, 400)                     # IoU 1/3
    same = [{'poly': a, 'comp': 'a'}, {'poly': b, 'comp': 'a'}]
    cross = [{'poly': a, 'comp': 'a'}, {'poly': b, 'comp': 'b'}]
    assert len(PC.merge_spines(same)) == 2 and len(PC.merge_spines(cross)) == 1


# ------------------------------------------------------------------------------------------ piece registration
def texture(seed=0, h=600, w=1100):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), 2.0)
    for _ in range(160):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), int(rng.integers(4, 22)), int(rng.integers(0, 255)), -1)
        cv2.rectangle(img, (x, y), (x + int(rng.integers(5, 40)), y + int(rng.integers(5, 60))), int(rng.integers(0, 255)), 2)
    return img


def write_pieces(tmp_path, images):
    comps = []
    for i, img in enumerate(images):
        name = f'piece_{i}.png'
        cv2.imwrite(str(tmp_path / name), img)
        comps.append({'component_id': f'component_{i + 1:03d}', 'image': name})
    return comps


def test_overlapping_pieces_register_with_the_right_shift_and_unrelated_ones_do_not(tmp_path):
    big = texture(1)
    comps = write_pieces(tmp_path, [big[:, :700], big[:, 400:], texture(2, 600, 700)])
    reg = PC.register_components(tmp_path, comps)
    assert [sorted(s) for s in reg['sheets'] if len(s) > 1] == [['component_001', 'component_002']]
    assert reg['to_root']['component_003'][0] == 'component_003'
    (link,) = reg['tree']
    H = np.asarray(link['H_b_to_a'])                                           # B's pixels -> A's pixels: B's x = 0 is A's x = 400
    pt = PC._apply(H, np.array([[100.0, 100.0]]))[0]
    assert abs(pt[0] - 500) < 12 and abs(pt[1] - 100) < 12
    assert link['inliers'] >= PC.REG['min_inliers'] and link['ncc'] >= PC.REG['min_ncc']
    root, Hb = reg['to_root']['component_002']
    assert root in ('component_001', 'component_002')


def test_a_flipped_or_wildly_scaled_match_is_not_accepted(tmp_path):
    big = texture(3)
    flipped = cv2.flip(big[:, 400:], 1)
    comps = write_pieces(tmp_path, [big[:, :700], flipped])
    assert PC.register_components(tmp_path, comps)['links'] == []


# ------------------------------------------------------------------------------------------ piece classes and detector plan
def piece(cid, polys, frames=5, n_sight=3):
    return {'component_id': cid, 'image': cid + '.jpg', 'source_frames': [{}] * frames, 'width_px': 3000, 'height_px': 2000,
            'markers': [{'group_id': f'{cid}-g{i}', 'polygon': p, 'sightings_total': n_sight} for i, p in enumerate(polys)]}


def tall(h, n=4, w=None):
    w = w or h / 8
    return [rect(100 + 3 * w * i, 100, 100 + 3 * w * i + w, 100 + h) for i in range(n)]


def test_pieces_are_classified_normal_collapsed_or_far():
    comps = [piece('n', tall(400)), piece('c', tall(3)), piece('f', tall(150)), piece('few', tall(400, n=2))]
    native = {f'{c["component_id"]}-g{i}': {'n': 450, 'c': 450, 'f': 160, 'few': 450}[c['component_id']] for c in comps for i in range(4)}
    for i in range(20):
        native[f'ref{i}'] = 450.0                                              # the rest of the run: the median native spine is 450 px
    cls = PC.classify_pieces(comps, native)
    assert {k: v['class'] for k, v in cls.items()} == {'n': 'normal', 'c': 'collapsed', 'f': 'far', 'few': 'normal'}
    assert cls['c']['piece_scale'] < PC.COLLAPSED_SCALE and cls['f']['native_vs_run'] < PC.FAR_FRACTION and 'too few' in cls['few']['reason']


def test_detector_scale_follows_the_spine_height_of_the_piece_and_tiles_are_capped():
    close = PC.detection_plan(piece('a', tall(900)), 3600, 2000, None)
    far = PC.detection_plan(piece('b', tall(120)), 3600, 2000, None)
    assert close['scale'] < 0.5 < far['scale'] and close['basis'] == 'median_group_polygon_height'
    assert far['tiles'] <= PC.MAX_TILES and far['window_px'] * far['scale'] == pytest.approx(1280, rel=0.02)
    assert PC.detection_plan({'markers': []}, 3000, 2000, 0.2)['basis'] == 'run_median_ratio'


# ------------------------------------------------------------------------------------------ the count of a synthetic run
def make_run(tmp_path, detections, *, tiers=None):
    """runs/r1 + runs/r1_panorama_v1 with four pieces: a normal one (two track groups on ONE spine, one on another), a collapsed one, a far one and a 2-frame one."""
    runs = tmp_path / 'runs'
    run, pan = runs / 'r1', runs / 'r1_panorama_v1'
    run.mkdir(parents=True)
    pan.mkdir()
    normal = piece('component_001', [rect(100, 100, 140, 500), rect(103, 105, 141, 498), rect(300, 100, 340, 500)])
    collapsed = piece('component_002', tall(3, n=4))
    far = piece('component_003', tall(150, n=3))
    small = piece('component_004', [rect(10, 10, 20, 90)], frames=2)
    books = []
    for comp, nat in ((normal, 450), (collapsed, 450), (far, 160), (small, 450)):
        for m in comp['markers']:
            books.append({'instance_id': m['group_id'], 'observation_ids': ['o'] * m['sightings_total'], 'polygon_xy': rect(0, 0, 50, nat), 'count_tier': 'confident'})
    for i in range(30):
        books.append({'instance_id': f'ref{i}', 'observation_ids': ['o'], 'polygon_xy': rect(0, 0, 50, 450), 'count_tier': 'unconfirmed'})      # only here to fix the run's median native spine
    books.append({'instance_id': 'lost-1', 'observation_ids': ['o', 'o'], 'polygon_xy': rect(0, 0, 50, 450), 'count_tier': 'confident'})       # frames in no piece, trusted
    books.append({'instance_id': 'lost-2', 'observation_ids': ['o'], 'polygon_xy': rect(0, 0, 50, 450), 'count_tier': 'unconfirmed'})         # frames in no piece, single sighting
    collapsed['markers'][3]['sightings_total'] = 1
    books[[b['instance_id'] for b in books].index('component_002-g3')]['count_tier'] = 'unconfirmed'
    for b in books:
        if tiers and b['instance_id'] in tiers:
            b['count_tier'] = tiers[b['instance_id']]
    for c in (normal, collapsed, far, small):
        for m in c['markers']:
            m['polygon'] = m['polygon']
    (pan / 'panorama.json').write_text(json.dumps({'components': [normal, collapsed, far, small], 'source_frames': [{}] * 20, 'unregistered_frames': [{}, {}]}))
    (run / 'inventory.json').write_text(json.dumps({'data': {'books': books}}))
    reg = {'to_root': {c['component_id']: (c['component_id'], np.eye(3)) for c in (normal, collapsed, far)}, 'sheets': [[c['component_id']] for c in (normal, collapsed, far)], 'links': [], 'tree': []}
    return run, pan, reg


DETS = {'component_001': {'detections': [{'poly': rect(98, 98, 142, 502), 'conf': 0.9}, {'poly': rect(298, 98, 342, 502), 'conf': 0.8}, {'poly': rect(600, 98, 640, 502), 'conf': 0.3}],
                          'plan': {'scale': 1.0}, 'seconds': 0.1}}


def test_count_merges_duplicates_adds_trusted_groups_and_keeps_far_and_withheld_groups_out(tmp_path):
    run, pan, reg = make_run(tmp_path, DETS)
    res = PC.build_count(run, pan, detections=DETS, registration=reg)
    cls = {c['component_id']: c['class'] for c in res['panorama_components']}
    assert cls == {'component_001': 'normal', 'component_002': 'collapsed', 'component_003': 'far'}       # the 2-frame piece is never counted from the panorama
    assert res['panorama_spines'] == 2                                            # the 0.3 box is under the 0.4 confidence
    assert res['collapsed_piece_groups'] == 3                                     # 4 groups in the collapsed piece, one single sighting is not trusted
    assert res['extra_unregistered_groups'] == 2 and set(res['unregistered_groups']['added_ids']) == {'component_004-g0', 'lost-1'}
    assert res['unregistered_groups']['withheld_single_sightings'] == 31          # lost-2 and the 30 reference singles
    assert res['panorama_count'] == 2 + 3 + 2 and res['far_pieces']['groups_not_counted'] == 3
    assert res['count_if_collapsed_pieces_not_counted'] == 2 + 2 and res['collapsed_pieces']['by_piece'] == {'component_002': 3}
    assert res['count_if_confidence']['0.30'] == res['panorama_count'] + 1 and res['range_panorama'] == [res['panorama_count'] - 0, res['panorama_count'] + 1]
    (line,) = res['merge_evidence']                                               # two track groups, one panorama spine
    assert [g['id'] for g in line['groups']] == ['component_001-g0', 'component_001-g1'] and line['duplicates_removed'] == 1 and line['box_xyxy'] == [98.0, 98.0, 142.0, 502.0]
    assert res['reconciliation']['track_duplicates_removed'] == 1 and res['reconciliation']['track_groups_without_spine'] == 0
    assert res['usable'] and res['detector_support'] == 1.0 and res['registration']['registered_ratio'] == 0.9
    # every counted spine is kept with its boxes in piece coordinates (the instance layer's reconciliation places single sightings on them)
    assert len(res['spines']) == res['panorama_spines'] and line['panorama_spine'] in {s['panorama_spine'] for s in res['spines']}
    sp = next(s for s in res['spines'] if s['panorama_spine'] == line['panorama_spine'])
    assert sp['members'][0]['component'] == 'component_001' and len(sp['members'][0]['poly']) == 4 and [g['id'] for g in sp['groups']] == ['component_001-g0', 'component_001-g1']


def test_a_detector_that_misses_the_trusted_track_groups_makes_the_count_unusable(tmp_path):
    run, pan, reg = make_run(tmp_path, {})
    res = PC.build_count(run, pan, detections={'component_001': {'detections': [], 'plan': None}}, registration=reg)
    assert res['usable'] is False and 'sees only 0%' in res['unusable_reason'] and res['detector_support'] == 0.0


def test_write_count_is_atomic_and_lands_beside_the_panorama_and_in_the_run(tmp_path):
    run, pan, reg = make_run(tmp_path, DETS)
    res = PC.build_count(run, pan, detections=DETS, registration=reg)
    paths = PC.write_count(res, pan, run)
    assert [p.parent.name for p in paths] == ['r1_panorama_v1', 'r1'] and all(json.loads(p.read_text())['panorama_count'] == res['panorama_count'] for p in paths)
    assert not list(run.glob('*.tmp')) and not list(pan.glob('*.tmp'))
    assert PC.latest_panorama(run) == pan


# ------------------------------------------------------------------------------------------ the packet
def pc_for(run_id, count=500, ratio=0.8, usable=True, **kw):
    return {'artifact_type': 'panorama_count', 'run_id': run_id, 'panorama_run': run_id + '_panorama_v1', 'panorama_count': count, 'range_panorama': [count - 40, count + 40], 'usable': usable,
            'unusable_reason': None if usable else 'detector too weak', 'registration': {'registered_ratio': ratio}, 'merge_evidence': [{'panorama_spine': 'c-s000', 'groups': []}] * 3,
            'panorama_components': [{'component_id': 'component_001', 'class': 'normal'}], **kw}


def test_a_usable_panorama_makes_the_panorama_count_the_headline_and_keeps_the_line_count():
    # User decision 2026-10-08: headline = the panorama count whenever the panorama is usable; the line count is always kept beside it
    data = P.build_data(TCR._inv(['confident', 'likely', 'unconfirmed']), TCR.manifest(), {'run_id': 't', 'panorama_count': pc_for('t', 3)})
    t, c = data['totals'], data['count']
    assert t['headline_estimator'] == 'panorama' and t['book_count'] == 3 and t['book_lines'] == len(data['books']) == 2 and t['panorama_count'] == 3
    assert t['count_estimator'] == 'track' and 'book_count_track' not in t                       # count_estimator keeps naming the line layer
    assert t['book_count_range'] == [0, 43] and t['book_count_range_label'] == 'panorama range (may not contain the true count)'   # pc_for: count -/+ 40, floored at 0
    assert t['book_lines_range'] == [1, 3] and c['lines']['count'] == 2 and c['lines']['estimator'] == 'track'
    assert c['best_guess'] == 3 and c['book_groups'] == 2 and c['panorama']['estimator_used'] == 'panorama' and c['panorama']['used'] and c['panorama']['merge_evidence_total'] == 3
    assert c['uncertainty_band']['low'] <= 3 <= c['uncertainty_band']['high']
    h = c['headline']
    assert h['estimator'] == 'panorama' and h['book_lines'] == 2 and 'usability check' in h['reason'] and '2 book lines are listed (some may be duplicates)' in h['reason']
    assert 'may not contain the true count' in h['reason'] and [q['reason'] for q in data['review_queue'] if q['code'] == 'count_headline'] == [h['reason']]
    from app import live_tools as LT
    assert LT.count_sentence(t) == ('about 3 books (counted from the 2D panorama; panorama range 0 to 43, which may not contain the true count; '
                                    '2 book lines are listed, some may be duplicates)')


@pytest.mark.parametrize('reliability', ['high', 'medium', 'low', None])
def test_the_count_reliability_does_not_gate_the_headline(reliability):
    inv = TCR._inv(['confident', 'confident', 'likely', 'unconfirmed'])
    inv['data']['counts'].update(count_range=[2, 9], confident_groups=2, count_reliability=reliability)
    t = P.build_data(inv, TCR.manifest(), {'run_id': 't', 'panorama_count': pc_for('t', 5)})['totals']
    assert t['headline_estimator'] == 'panorama' and t['book_count'] == 5 and t['book_count_range'] == [0, 45] and t['confident_groups'] == 2
    assert t['book_lines'] == 3 and t['book_lines_range'] == [2, 9]


@pytest.mark.parametrize('why,kw', [('low registration', {'ratio': 0.5}), ('unusable', {'usable': False}), ('outside the line range', {})])
def test_the_line_count_stays_the_headline_when_the_panorama_is_not_usable_and_the_panorama_is_a_cross_check(why, kw):
    inv = TCR._inv(['confident', 'likely'])
    data = P.build_data(inv, TCR.manifest(), {'run_id': 't', 'panorama_count': pc_for('t', 5, **kw)})                # 5 lies outside the lines' range [1, 3]
    t = data['totals']
    assert t['headline_estimator'] == 'lines' and t['count_estimator'] == 'track' and t['book_count'] == t['book_lines'] == len(data['books']) == 2 and 'book_count_track' not in t
    assert t['panorama_count'] == 5 and t['book_count_range'] == t['book_lines_range'] == [1, 3] and t['book_count_range_label'].startswith('range of the book lines')
    c = data['count']
    assert c['panorama']['used'] is False and 'usability check' in c['panorama']['reason'] and c['best_guess'] == 2
    assert c['headline']['estimator'] == 'lines' and 'cross-check' in c['headline']['reason']
    from app import live_tools as LT
    assert LT.count_sentence(t) == 'about 2 books (range 1 to 3, count reliability low)'


def test_apply_is_idempotent_and_can_hand_the_headline_back_to_the_line_count():
    data = P.build_data(TCR._inv(['confident', 'likely']), TCR.manifest(), {'run_id': 't'})
    plain = copy.deepcopy(data['totals'])
    plain_band = copy.deepcopy(data['count']['uncertainty_band'])
    PA.apply(data, pc_for('t', 3))
    once = copy.deepcopy(data)
    PA.apply(data, pc_for('t', 3))
    assert data == once and data['totals']['book_count'] == 3
    PA.apply(data, pc_for('t', 7, ratio=0.3))                                    # a later, worse panorama: the line count comes back
    t = data['totals']
    assert t['book_count'] == 2 and t['book_count_range'] == plain['book_count_range'] and t['book_count_lower'] == plain['book_count_lower'] and t['count_estimator'] == 'track'
    assert t['headline_estimator'] == 'lines' and data['count']['uncertainty_band'] == plain_band


def test_a_count_of_another_run_or_a_corrupt_file_is_ignored(tmp_path):
    run = tmp_path / 'r1'
    run.mkdir()
    (run / 'panorama_count.json').write_text(json.dumps(pc_for('another_run')))
    assert PA.load(run) is None
    (run / 'panorama_count.json').write_text('{not json')
    assert PA.load(run) is None
    (run / 'panorama_count.json').write_text(json.dumps(pc_for('r1')))
    assert PA.load(run)['panorama_count'] == 500


# ------------------------------------------------------------------------------------------ attaching to a finished packet
def finished_run(tmp_path, reliability='low'):
    run, man, rc = TAA.make_run(tmp_path)                                         # run dir 'run': packet built at the end of a sweep, 14 lines, no tiers
    env = C.read_json(run / C.ARTIFACTS['packet'])
    env['data']['totals']['count_reliability'] = reliability
    C.write_json(run / C.ARTIFACTS['packet'], env)
    (run / 'panorama_count.json').write_text(json.dumps(pc_for('run', 11)))
    return run, man


def test_attach_patches_the_packet_in_place_keeps_the_original_and_still_verifies(tmp_path):
    run, man = finished_run(tmp_path)
    before = (run / C.ARTIFACTS['packet']).read_bytes()
    res = PA.attach(run, render=False)
    assert res['attached'] and res['estimator'] == 'panorama' and res['headline_estimator'] == 'panorama' and res['book_count'] == 11 and res['book_lines'] == res['track_count'] == 14
    env = C.read_json(run / C.ARTIFACTS['packet'])
    assert P.verify_packet(env, man, run) == [] and env['data']['totals']['book_count'] == 11 and env['data']['totals']['book_lines'] == 14
    assert (run / PA.BACKUP).read_bytes() == before


def test_attach_twice_gives_the_same_packet_and_does_not_undo_a_measurement(tmp_path):
    run, man = finished_run(tmp_path)
    PA.attach(run, render=False)
    first = (run / C.ARTIFACTS['packet']).read_bytes()
    env = C.read_json(run / C.ARTIFACTS['packet'])
    env['data']['totals']['spine_dimensions_measured'] = 3                        # what pipeline.auto_attach adds later
    C.write_json(run / C.ARTIFACTS['packet'], env)
    PA.attach(run, render=False)
    again = C.read_json(run / C.ARTIFACTS['packet'])
    assert again['data']['totals']['spine_dimensions_measured'] == 3 and again['data']['totals']['book_count'] == 11
    again['data']['totals'].pop('spine_dimensions_measured')
    assert json.dumps(again, sort_keys=True) == json.dumps(json.loads(first), sort_keys=True)


def test_nothing_is_written_when_the_patched_packet_does_not_verify(tmp_path, monkeypatch):
    run, man = finished_run(tmp_path)
    monkeypatch.setattr(P, 'verify_packet', lambda *a, **k: ['totals mismatch (synthetic)'])
    before = (run / C.ARTIFACTS['packet']).read_bytes()
    res = PA.attach(run, render=False)
    assert res['attached'] is False and 'verification failed' in res['reason'] and (run / C.ARTIFACTS['packet']).read_bytes() == before


def test_attach_without_a_count_or_a_packet_is_a_no_op(tmp_path):
    assert PA.attach(tmp_path, render=False) == {'attached': False, 'reason': 'no claim_packet.json'}
    run, _ = finished_run(tmp_path)
    (run / 'panorama_count.json').unlink()
    assert PA.attach(run, render=False)['reason'] == 'no panorama_count.json for this run'


# ------------------------------------------------------------------------------------------ the post-sweep hook
def test_the_count_runs_after_the_panorama_and_before_the_3d_map_and_is_recorded(tmp_path):
    log = tmp_path / 'order.log'

    def on_count(run_id, pan_out):
        with open(log, 'a') as f:
            f.write(f'count:{run_id}:{pan_out}\n')
        return {'panorama_count': 321, 'estimator': 'panorama', 'range_panorama': [300, 340]}
    root, jobs, psb = builds(tmp_path, command=fake_command(log=log), on_count=on_count)
    psb.enqueue('sweep_abc')
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert log.read_text().split() == ['panorama:start', 'count:sweep_abc:sweep_abc_panorama_v1', 'recon3d:start']
    assert st['state'] == 'done' and st['count']['state'] == 'done' and 'count: panorama count 321 (panorama)' in st['summary']


def test_a_failing_count_is_recorded_and_never_fails_the_maps(tmp_path):
    def boom(run_id, pan_out):
        raise RuntimeError('detector runtime missing')
    root, jobs, psb = builds(tmp_path, command=fake_command(), on_count=boom)
    psb.enqueue('sweep_abc')
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['state'] == 'done' and st['steps']['panorama']['state'] == 'done' and st['steps']['recon3d']['state'] == 'done'
    assert st['count']['state'] == 'failed' and 'detector runtime missing' in st['count']['error'] and 'count: failed' in st['summary']


def test_without_an_on_count_hook_the_queue_behaves_as_before(tmp_path):
    root, jobs, psb = builds(tmp_path, command=fake_command())
    psb.enqueue('sweep_abc')
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['state'] == 'done' and 'count' not in st


def test_a_restart_counts_a_finished_panorama_that_was_never_counted(tmp_path):
    seen = []
    root, jobs, psb = builds(tmp_path, command=fake_command(), on_count=lambda r, p: seen.append((r, p)) or {'panorama_count': 5, 'estimator': 'track'})
    psb.enqueue('sweep_abc')
    wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    seen.clear()
    rec = json.loads((psb.dir / 'sweep_abc.json').read_text())
    rec.pop('count')
    (psb.dir / 'sweep_abc.json').write_text(json.dumps(rec))
    psb2 = PS.PostSweepBuilds(root, BuildJobs(root, fake_command()), poll_s=POLL, on_count=lambda r, p: seen.append((r, p)) or {'panorama_count': 5, 'estimator': 'track'})
    assert psb2.resume() >= 1
    wait_for(lambda: (psb2.status('sweep_abc') or {}).get('count', {}).get('state') == 'done')
    psb2.stop()
    assert seen == [('sweep_abc', 'sweep_abc_panorama_v1')]


# ------------------------------------------------------------------------------------------ piece-pair coverage (spine-normalized SIFT, boundary frames)
def test_spine_normalized_features_put_the_piece_spine_at_the_target_height(tmp_path):
    big = texture(7)
    comps = write_pieces(tmp_path, [big])
    plain = PC.component_features(tmp_path, comps[0])
    norm = PC.component_features(tmp_path, comps[0], spine_px=640.0)                       # spines 640 px tall in the piece -> target 160 px: scale 0.25
    assert plain['scale'] == pytest.approx(PC.REG['side_px'] / 1100) and norm['scale'] == pytest.approx(0.25)
    assert PC.component_features(tmp_path, comps[0], spine_px=40.0)['scale'] == 1.0, 'a piece is never enlarged beyond its own pixels'
    assert PC.piece_spine_px(piece('p', tall(400))) == pytest.approx(400.0) and PC.piece_spine_px(piece('q', tall(400, n=2))) is None


def test_a_far_piece_registers_to_a_close_one_through_spine_normalization(tmp_path):
    big = texture(11, 900, 1500)
    close = big[:, :1000]
    far = cv2.resize(big[:, 500:], None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)        # the same shelf area filmed 3.3x further away
    comps = write_pieces(tmp_path, [close, far])
    comps[0]['markers'] = piece('a', tall(300))['markers']
    comps[1]['markers'] = piece('b', tall(90))['markers']
    for c in comps:
        c['source_frames'] = []
    reg = PC.register_components(tmp_path, comps)
    assert [sorted(s) for s in reg['sheets'] if len(s) > 1] == [['component_001', 'component_002']]
    (link,) = reg['tree']
    assert link['via'] in ('plain', 'spine_normalized') and link['inliers'] >= PC.REG['min_inliers']
    pt = PC._apply(np.asarray(link['H_b_to_a']), np.array([[30.0, 30.0]]))[0]                  # far (0.3x) pixel (30, 30) is close pixel (500 + 100, 100)
    assert abs(pt[0] - 600) < 15 and abs(pt[1] - 100) < 15


def test_consecutive_pieces_register_through_their_boundary_frames_when_the_direct_match_fails(tmp_path):
    big = cv2.cvtColor(texture(13, 900, 1700), cv2.COLOR_GRAY2BGR)
    run = tmp_path / 'run'
    (run / 'frames').mkdir(parents=True)
    pan = tmp_path / 'pan'
    pan.mkdir()
    piece_a, piece_b = big[100:700, 0:900], cv2.GaussianBlur(big[100:700, 700:1700], (0, 0), 7)   # B is too blurred for a direct piece match
    cv2.imwrite(str(pan / 'a.png'), piece_a)
    cv2.imwrite(str(pan / 'b.png'), piece_b)
    fa, fb = big[200:600, 500:1100], big[200:600, 650:1250]                                           # sharp boundary frames: fa in A (x 500..), fb in B (x 650.. = B's 0 + -50?)
    cv2.imwrite(str(run / 'frames' / 'f00010.jpg'), fa)
    cv2.imwrite(str(run / 'frames' / 'f00011.jpg'), fb)
    T = lambda dx, dy: [[1, 0, dx], [0, 1, dy], [0, 0, 1]]
    frame_a = {'frame_id': 'f00010', 'timestamp_s': 10.0, 'image_path': 'frames/f00010.jpg', 'width_px': 600, 'height_px': 400, 'original_to_panorama_3x3': T(500, 100)}
    frame_b = {'frame_id': 'f00011', 'timestamp_s': 10.4, 'image_path': 'frames/f00011.jpg', 'width_px': 600, 'height_px': 400, 'original_to_panorama_3x3': T(650 - 700, 100)}
    comps = [{'component_id': 'component_001', 'image': 'a.png', 'source_frames': [frame_a], 'markers': []},
             {'component_id': 'component_002', 'image': 'b.png', 'source_frames': [frame_b], 'markers': []}]
    direct = PC.register_pair(PC.component_features(pan, comps[0]), PC.component_features(pan, comps[1]))
    reg = PC.register_components(pan, comps, PC.REG, None, run, None)
    assert direct is None or reg['tree'][0]['via'] != 'boundary_frames', 'precondition: the blurred piece defeats the direct match'
    (link,) = reg['tree']
    assert link['via'] == 'boundary_frames' and link['frames'] == ['f00010', 'f00011'] and link['ncc'] >= PC.REG['min_ncc']
    pt = PC._apply(np.asarray(link['H_b_to_a']), np.array([[100.0, 100.0]]))[0]                   # B pixel (100, 100) = big (800, 200) = A pixel (800, 100)
    assert abs(pt[0] - 800) < 10 and abs(pt[1] - 100) < 10
    assert PC.register_components(pan, comps)['tree'] == [] or PC.register_components(pan, comps)['tree'][0]['via'] != 'boundary_frames', 'without the run dir there is no boundary path'
