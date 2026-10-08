"""Fragment merging of a split SfM build (pipeline/recon3d_merge.py + recon3d.model_inputs). No pycolmap: fake models."""
import types

import numpy as np
import pytest

from pipeline import recon3d as R
from pipeline import recon3d_merge as M


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    if axis == 'x':
        return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


class _Pose:
    def __init__(self, r, t):
        self.rotation = types.SimpleNamespace(matrix=lambda r=r: r)
        self.translation = t


def _image(name, r_wc, centre, camera_id=1):
    """A registered image: world-to-camera rotation r_wc, camera centre in world."""
    t = -r_wc @ np.asarray(centre, float)
    return types.SimpleNamespace(name=name, has_pose=True, camera_id=camera_id, cam_from_world=_Pose(r_wc, t), points2D=[])


def _point(xyz, track=3, error=0.5):
    return types.SimpleNamespace(xyz=np.asarray(xyz, float), color=np.array([1, 2, 3]), error=error,
                                 track=types.SimpleNamespace(length=lambda n=track: n))


class _Rec:
    def __init__(self, images, points):
        self.images = {i + 1: im for i, im in enumerate(images)}
        self.points3D = {i + 1: p for i, p in enumerate(points)}
        self.cameras = {1: types.SimpleNamespace(focal_length=1000.0, width=1920, height=1080,
                                                 cam_from_img=lambda xy: (np.asarray(xy, float) - [960, 540]) / 1000.0)}

    def compute_mean_reprojection_error(self):
        return 0.5


def _level_camera(yaw_deg, pitch_deg=0.0):
    """Upright handheld camera (no roll) facing -z of the world rotated by yaw about world up (+y) and pitched."""
    # camera axes in world: x right, y down, z forward; start facing -z with up = +y
    base = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])         # rows: world->camera for yaw 0
    return base @ _rot('x', pitch_deg).T @ _rot('y', yaw_deg).T


# ------------------------------------------------------------------------------------------- pairs

def test_ranges_merge_small_holes_and_split_big_ones():
    assert M._ranges([0, 1, 2, 5, 6, 20]) == [(0, 6), (20, 20)]
    assert M._ranges([]) == []


def test_sequential_pairs_are_the_ones_colmap_already_matched():
    assert M._sequential_pair(10, 22, 12) and M._sequential_pair(10, 42, 12)       # window, then i + 2^5
    assert not M._sequential_pair(10, 23, 12) and not M._sequential_pair(10, 50, 12)


def test_bridge_pairs_only_cross_fragments_and_skip_sequential_pairs():
    frags = [{'model': 0, 'size': 60, 'frames': list(range(0, 60)), 'ranges': [(0, 59)]},
             {'model': 1, 'size': 40, 'frames': list(range(60, 100)), 'ranges': [(60, 99)]}]
    pairs, why = M.bridge_pairs(frags, 100, overlap=12, window=30, ends=8, reps=20)
    owner = {i: 0 for i in range(60)} | {i: 1 for i in range(60, 100)}
    assert pairs and all(owner[i] != owner[j] for i, j in pairs)
    assert not any(M._sequential_pair(i, j, 12) for i, j in pairs)
    assert why['ends'] > 0 and why['representatives'] > 0 and sum(why.values()) == len(pairs) == len(set(pairs))
    assert all(i < j for i, j in pairs)


def test_a_single_fragment_needs_no_representatives():
    frags = [{'model': 0, 'size': 50, 'frames': list(range(50)), 'ranges': [(0, 49)]}]
    pairs, why = M.bridge_pairs(frags, 80, overlap=12, window=30, ends=8, reps=20)
    assert why['representatives'] == 0 and all(j >= 50 for _, j in pairs)        # only frames no model registered


# ------------------------------------------------------------------------------------------- similarity

def test_umeyama_and_robust_similarity_recover_a_similarity_despite_outliers():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(80, 3))
    s, r, t = 2.5, _rot('z', 30) @ _rot('x', 10), np.array([1.0, -2.0, 0.5])
    tgt = s * src @ r.T + t
    tgt[:15] += rng.normal(scale=3.0, size=(15, 3))                               # 19% wrong correspondences
    sim, inl = M.robust_similarity(src, tgt, 0.05)
    assert sim is not None and inl[15:].all() and inl[:15].sum() <= 1
    assert sim[0] == pytest.approx(s, rel=1e-6) and np.allclose(sim[1], r, atol=1e-6) and np.allclose(sim[2], t, atol=1e-6)


def test_compose_is_function_composition():
    a = (2.0, _rot('z', 20), np.array([1.0, 0, 0]))
    b = (0.5, _rot('x', 40), np.array([0, 3.0, 0]))
    x = np.array([0.3, -0.2, 1.1])
    s, r, t = M.compose(a, b)
    inner = b[0] * b[1] @ x + b[2]
    assert np.allclose(s * r @ x + t, a[0] * a[1] @ inner + a[2])


def test_estimate_link_needs_enough_consistent_shared_frames():
    rng = np.random.default_rng(2)
    tgt = {f'f{i}': rng.normal(size=3) for i in range(10)}
    s, r, t = 0.7, _rot('y', 25), np.array([0.2, 0.1, -1.0])
    src = {k: (v - t) @ r / s for k, v in tgt.items()}                         # src = inverse similarity of tgt
    sim, ev = M.estimate_link(src, tgt)
    assert sim is not None and ev['decision'] == 'accepted' and ev['shared_frames'] == 10 and ev['scale'] == pytest.approx(s, rel=1e-4)
    few = {k: src[k] for k in list(src)[:3]}
    sim, ev = M.estimate_link(few, tgt)
    assert sim is None and ev['decision'].startswith('rejected: 3 shared')
    noisy = {k: v + rng.normal(scale=5.0, size=3) for k, v in src.items()}      # frames disagree
    assert M.estimate_link(noisy, tgt)[0] is None


def test_up_is_the_direction_perpendicular_to_every_level_camera_x_axis():
    rights = np.array([_level_camera(y, p).T @ [1.0, 0, 0] for y in (-40, -10, 0, 25, 50) for p in (-20, 0, 15)])
    up, cond = M._up(rights)
    assert abs(abs(float(up @ [0, 1.0, 0])) - 1.0) < 1e-9 and cond > M.HORIZON_MIN_CONDITIONING
    still = np.array([_level_camera(0, p).T @ [1.0, 0, 0] for p in (-20, 0, 15)])    # pitch only: horizontal plane undefined
    assert M._up(still)[1] < M.HORIZON_MIN_CONDITIONING


def test_horizon_coherence_flags_a_fragment_joined_with_a_roll():
    yaws = (-40, -20, 0, 20, 40)
    good = _Rec([_image(f'a{y}', _level_camera(y), [y / 10, 0, 5]) for y in yaws], [])
    tilted = _Rec([_image(f'b{y}', _level_camera(y) @ _rot('z', 15).T, [y / 10, 0, 5]) for y in yaws], [])
    merged = _Rec(list(good.images.values()) + list(tilted.images.values()), [])
    out = M.horizon_coherence(merged, {0: good, 1: tilted})
    rows = {r['fragment']: r for r in out['fragments']}
    assert abs(rows[0]['median_tilt_deg']) < 0.5 and abs(rows[1]['median_tilt_deg']) > 5


# ------------------------------------------------------------------------------------------- joined scene

def test_model_inputs_maps_joined_members_into_the_main_frame_and_keeps_the_main_pose_of_shared_frames():
    main = _Rec([_image('f0.jpg', np.eye(3), [0, 0, 0]), _image('f1.jpg', np.eye(3), [1, 0, 0])],
                [_point([0, 0, 5]), _point([1, 1, 5]), _point([9, 9, 9], track=2)])
    sim = (2.0, _rot('z', 90), np.array([10.0, 0, 0]))
    other = _Rec([_image('f1.jpg', np.eye(3), [7, 7, 7]), _image('f2.jpg', np.eye(3), [1, 0, 0])], [_point([0, 0, 1])])
    xyz, colour, views, intr, err, n_all, n_kept = R.model_inputs([(main, None), (other, sim)], {'f0.jpg': 'f0', 'f1.jpg': 'f1', 'f2.jpg': 'f2'})
    assert set(views) == {'f0', 'f1', 'f2'} and n_all == 4 and n_kept == 3 and len(xyz) == 3 and err == pytest.approx(0.5)
    assert np.allclose(views['f1'].centre, [1, 0, 0])                              # shared frame: the main model's pose wins
    assert np.allclose(views['f2'].centre, 2.0 * _rot('z', 90) @ [1, 0, 0] + [10, 0, 0])
    assert np.allclose(views['f2'].rotation, _rot('z', 90).T)                      # world-to-camera of the joined frame
    assert any(np.allclose(p, 2.0 * _rot('z', 90) @ [0, 0, 1] + [10, 0, 0]) for p in xyz)
    o, d = views['f2'].ray([960, 540])                                            # the optical axis follows the similarity
    assert np.allclose(d / np.linalg.norm(d), _rot('z', 90) @ [0, 0, 1])


def test_merge_flag_from_the_environment(monkeypatch):
    monkeypatch.delenv('LIBRARY_RECON3D_MERGE', raising=False)
    assert R.merge_enabled() is R.DEFAULT_MERGE
    monkeypatch.setenv('LIBRARY_RECON3D_MERGE', '0')
    assert R.merge_enabled() is False
    monkeypatch.setenv('LIBRARY_RECON3D_MERGE', '1')
    assert R.merge_enabled() is True


def test_merge_mode_from_the_environment_and_the_plan_decisions(monkeypatch):
    monkeypatch.delenv('LIBRARY_RECON3D_MERGE_MODE', raising=False)
    assert R.merge_mode() == R.DEFAULT_MERGE_MODE
    monkeypatch.setenv('LIBRARY_RECON3D_MERGE_MODE', 'Inline')
    assert R.merge_mode() == 'inline' and R.merge_mode('deferred') == 'deferred'
    monkeypatch.setenv('LIBRARY_RECON3D_MERGE_MODE', 'later')
    with pytest.raises(ValueError):
        R.merge_mode()
    assert M.MergePlan('inline').allow('remap', 1e6) is True                    # full-rule behaviour, and the merge pass itself
    deferred = M.MergePlan('deferred', 270, started=None)
    assert deferred.allow('bridge_pairs', 0.0) is False and deferred.log[0]['decision'] == 'deferred to the merge pass'
    auto = M.MergePlan('auto', 270, started=None)                                # elapsed 0 without a start time
    assert auto.allow('bridge_pairs', 200) is True and auto.allow('remap', 300) is False
    assert [e['decision'] for e in auto.log] == ['run in this build', 'deferred to the merge pass']


def _fake_build(tmp_path, monkeypatch, models, merge_impl):
    import json
    import sys
    run = tmp_path / 'run'
    (run / 'frames').mkdir(parents=True)
    frames = [{'frame_id': f'f{i:04d}', 'image_path': f'frames/f{i:04d}.jpg'} for i in range(10)]
    for f in frames:
        (run / f['image_path']).write_bytes(b'x')
    (run / 'manifest.json').write_text(json.dumps({'data': {'frames': frames}}))

    class Opts:
        def __init__(self):
            self.sift = types.SimpleNamespace()

    fake = types.SimpleNamespace(
        __version__='0', FeatureExtractionOptions=Opts, ImageReaderOptions=Opts, SequentialPairingOptions=Opts, IncrementalPipelineOptions=Opts,
        CameraMode=types.SimpleNamespace(SINGLE=0), Device=types.SimpleNamespace(cpu=0),
        extract_features=lambda *a, **k: None, match_sequential=lambda *a, **k: None, incremental_mapping=lambda *a, **k: models)
    monkeypatch.setitem(sys.modules, 'pycolmap', fake)
    monkeypatch.setattr(M, 'merge', merge_impl)
    monkeypatch.setattr(M, 'newest_panorama', lambda source: None)
    return json.loads(R.build(run, tmp_path / 'run_recon3d_v1', profile='default', merge=True).read_text())


def _cams(names):
    return [_image(f'{n}.jpg', np.eye(3), [i * 0.1, 0, 0]) for i, n in enumerate(names)]


def test_build_uses_the_joined_members_and_records_merged_from(tmp_path, monkeypatch):
    big = _Rec(_cams(['f0000', 'f0001', 'f0002', 'f0003']), [_point([0, 0, 5])])
    small = _Rec(_cams(['f0006', 'f0007']), [_point([0, 0, 5])])
    seen = {}

    def fake_merge(database, images, work, models, names, mapping, **k):
        seen['names'] = names
        return [(big, None), (small, (1.0, np.eye(3), np.array([5.0, 0, 0])))], {'used': 'original', 'registered': 6}
    out = _fake_build(tmp_path, monkeypatch, {0: big, 1: small}, fake_merge)
    assert seen['names'][:2] == ['f0000.jpg', 'f0001.jpg'] and len(seen['names']) == 10          # run order, every frame
    assert out['registered_frames'] == 6 and out['models_found'] == 2 and out['model_registered_sizes'] == [4, 2]
    assert {k: out['merge'][k] for k in ('enabled', 'applied', 'model', 'stage', 'deferred')} == \
        {'enabled': True, 'applied': True, 'model': 'merged', 'stage': 'first_map', 'deferred': None} and out['merged_from']['used'] == 'original'
    assert 'merge_s' in out['timings']


def test_a_failing_merge_keeps_the_largest_model_and_says_why(tmp_path, monkeypatch):
    big = _Rec(_cams(['f0000', 'f0001', 'f0002']), [_point([0, 0, 5])])
    small = _Rec(_cams(['f0006', 'f0007']), [])

    def broken(*a, **k):
        raise RuntimeError('boom')
    out = _fake_build(tmp_path, monkeypatch, {0: small, 1: big}, broken)
    assert out['registered_frames'] == 3 and out['merge']['model'] == 'largest' and out['merge']['applied'] is False
    assert out['merged_from']['used'] == 'largest_only' and out['merged_from']['error'] == 'RuntimeError: boom'


def test_newest_panorama_needs_a_count_file(tmp_path):
    run = tmp_path / 'sweep_x'
    run.mkdir()
    for n in (1, 2, 10):
        (tmp_path / f'sweep_x_panorama_v{n}').mkdir()
    (tmp_path / 'sweep_x_panorama_v2' / 'panorama_count.json').write_text('{}')
    assert M.newest_panorama(run).name == 'sweep_x_panorama_v2'
    (tmp_path / 'sweep_x_panorama_v10' / 'panorama_count.json').write_text('{}')
    assert M.newest_panorama(run).name == 'sweep_x_panorama_v10'
