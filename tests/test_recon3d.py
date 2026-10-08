"""pipeline.recon3d pure parts: triangulation, plane fit + shelf alignment, marker placement and its invariant, naming, status.
SfM itself (pycolmap) is checked on real runs; everything here runs on synthetic data."""
import json

import numpy as np
import pytest

from pipeline import recon3d as R


def rays(point, centres):
    point = np.asarray(point, float)
    centres = np.asarray(centres, float)
    return centres, point - centres


def test_triangulates_exact_point_from_three_views():
    o, d = rays([1.0, 2.0, 30.0], [[0, 0, 0], [3, 0, 0], [-3, 1, 0]])
    fit = R.triangulate_rays(o, d)
    assert np.allclose(fit['xyz'], [1, 2, 30], atol=1e-6)
    assert fit['views'] == 3 and fit['residual'] < 1e-9 and fit['angle_deg'] > 5


def test_rejects_parallel_or_single_views():
    o, d = rays([0, 0, 50], [[0, 0, 0], [0.01, 0, 0]])           # almost no baseline
    assert R.triangulate_rays(o, d) is None
    assert R.triangulate_rays([[0, 0, 0]], [[0, 0, 1]]) is None


def test_one_bad_ray_is_dropped():
    o, d = rays([0.0, 0.0, 20.0], [[-4, 0, 0], [4, 0, 0], [0, 4, 0], [0, -4, 0]])
    d = d.copy()
    d[3] = [1.0, 0.8, 1.0]                                        # wrong sighting
    fit = R.triangulate_rays(o, d)
    assert fit['views'] == 3 and np.allclose(fit['xyz'], [0, 0, 20], atol=1e-6)


def test_points_behind_cameras_are_rejected():
    o = np.array([[0, 0, 0], [4, 0, 0]], float)
    d = np.array([[0, 0, 1], [1, 0, 1.0]])                       # the rays only meet at z=-4, behind both cameras
    assert R.triangulate_rays(o, d) is None


def test_subsample_is_bounded_deterministic_and_sorted():
    assert list(R.subsample(5, 10)) == [0, 1, 2, 3, 4]
    a, b = R.subsample(1000, 100), R.subsample(1000, 100)
    assert len(a) == 100 and (a == b).all() and (np.diff(a) > 0).all()
    assert R.MAX_POINTS == 200_000


def test_next_output_never_reuses_a_directory(tmp_path):
    assert R.next_output(tmp_path, 'final_yt1_v1') == 'final_yt1_v1_recon3d_v1'
    (tmp_path / 'final_yt1_v1_recon3d_v1').mkdir()
    (tmp_path / 'final_yt1_v1_recon3d_v3').mkdir()
    assert R.next_output(tmp_path, 'final_yt1_v1') == 'final_yt1_v1_recon3d_v2'


@pytest.mark.parametrize('ratio,markers,kwargs,expected', [
    (0.9, [1], {}, 'ok'), (0.9, [], {}, 'partial'), (0.45, [1], {}, 'partial'), (0.1, [1], {}, 'failed'),
    (0.9, [], {'expects_markers': False}, 'ok'),                  # no inventory: nothing to place
    (0.9, [1], {'placed_fraction': 0.3}, 'partial'),              # most groups unplaced
    (0.9, [1], {'plane_ok': False}, 'partial')])                  # no shelf plane: not shelf-aligned
def test_status_thresholds(ratio, markers, kwargs, expected):
    assert R.status_for(ratio, markers, **kwargs) == expected


def test_refuses_existing_output(tmp_path):
    (tmp_path / 'm').mkdir()
    (tmp_path / 'm/manifest.json').write_text('{"data": {"frames": [{"frame_id": "f0", "image_path": "frames/f0.jpg"}]}}')
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.raises(ValueError, match='existing'):
        R.build(tmp_path / 'm', out)


# ---------------------------------------------------------------- synthetic shelf: ideal frame is plane z=0, x right, y up

W, H, F = 1920, 1080, 1500.0
R_FRONT = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])    # world->cam for a camera on +z looking at -z, image up = +y


def project(view, point):
    xc = view.rotation @ (np.asarray(point, float) - view.centre)
    return np.array([xc[0] / xc[2] * F + W / 2, xc[1] / xc[2] * F + H / 2])


def front_view(frame_id, x, y=0.0, z=6.0, yaw=0.0):
    """Camera at (x, y, z) in the ideal frame looking at the shelf (optionally yawed about the vertical)."""
    c, s = np.cos(yaw), np.sin(yaw)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R.View(frame_id, [x, y, z], R_FRONT @ ry, R.pinhole(F, W / 2, H / 2))


def rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def spine(view, centre_xy, w=0.25, h=1.2):
    """Detection polygon (image px, OBB order) of a vertical spine on the plane as seen from `view`."""
    x, y = centre_xy
    corners = [(x + w / 2, y - h / 2, 0.0), (x + w / 2, y + h / 2, 0.0), (x - w / 2, y + h / 2, 0.0), (x - w / 2, y - h / 2, 0.0)]
    return [project(view, c).tolist() for c in corners]


def sighting(view, centre_xy, oid, sharp=1.0, **extra):
    poly = spine(view, centre_xy)
    arr = np.array(poly)
    return {'observation_id': oid, 'frame_id': view.frame_id, 'bbox_xyxy': [*arr.min(axis=0), *arr.max(axis=0)], 'polygon_xy': poly,
            'sharpness': sharp, 'confidence': 0.9, 'width_px': W, 'height_px': H, **extra}


def test_fit_plane_recovers_noisy_plane_among_outliers():
    rng = np.random.default_rng(1)
    plane = np.c_[rng.uniform(-5, 5, 3000), rng.uniform(-1.5, 1.5, 3000), rng.normal(0, 0.01, 3000)]
    junk = rng.uniform(-6, 6, (600, 3))
    fit = R.fit_plane(np.vstack([plane, junk]))
    assert abs(fit['normal'][2]) > 0.999 and fit['rms'] < 0.03 and 0.7 < fit['inlier_ratio'] < 0.95
    assert fit['rms'] / fit['extent'] < 0.01
    assert R.fit_plane(np.zeros((5, 3))) is None


def test_orient_normal_points_to_the_camera_side():
    n = R.orient_normal(np.array([0, 0, -1.0]), np.zeros(3), [[0, 0, 5], [1, 0, 6]])
    assert n[2] == 1.0
    assert R.orient_normal(np.array([0, 0, 1.0]), np.zeros(3), [[0, 0, 5]])[2] == 1.0


def synthetic_sfm(seed, with_polygons=True):
    """The ideal shelf scene moved by a random rigid motion (an arbitrary SfM frame): cloud, views, sightings and the motion."""
    rng = np.random.default_rng(seed)
    m, b = rotation(seed), rng.normal(size=3) * 3
    cloud = np.c_[rng.uniform(-5, 5, 4000), rng.uniform(-1.5, 1.5, 4000), rng.normal(0, 0.02, 4000)]
    ideal = [front_view(f'f{i:03d}', x, y=0.05 * (i % 3), yaw=0.03 * np.sin(i)) for i, x in enumerate(np.linspace(-4, 4, 17))]
    sightings = []
    for k, (x, y) in enumerate([(-3.0, 0.0), (-1.2, 0.3), (0.4, -0.2), (2.5, 0.1), (3.6, 0.5)]):
        for v in ideal:
            if abs(v.centre[0] - x) < 3.2 and with_polygons:
                sightings.append(sighting(v, (x, y), f'{v.frame_id}-o{k}'))
    sfm_cloud = cloud @ m.T + b
    sfm_views = {v.frame_id: R.View(v.frame_id, m @ v.centre + b, v.rotation @ m.T, v.unproject) for v in ideal}
    return sfm_cloud, sfm_views, sightings, m, b, ideal


@pytest.mark.parametrize('seed', [3, 11, 29])
def test_alignment_makes_plane_z0_camera_side_up_y_and_is_a_proper_rotation(seed):
    cloud, views, sightings, m, b, ideal = synthetic_sfm(seed)
    rot, origin, plane, alignment = R.align_scene(cloud, views, sightings)
    assert alignment['applied'] and alignment['up_source'] == 'book_polygons' and alignment['up_samples'] >= 15
    assert np.isclose(np.linalg.det(rot), 1.0) and np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    aligned = (cloud - origin) @ rot.T
    assert abs(np.median(aligned[:, 2])) < 0.02 and np.abs(aligned[:, 2]).mean() < 0.05                  # shelf plane = z 0
    av = {k: v.transformed(rot, origin) for k, v in views.items()}
    assert all(v.centre[2] > 4 for v in av.values())                                                      # camera side = +z
    assert np.allclose(rot @ m, np.eye(3), atol=0.02)                                                     # back to x right, y up, z out
    assert all(v.up()[1] > 0.99 for v in av.values())                                                     # cameras are upright
    assert plane['inlier_ratio'] > 0.9 and plane['rms'] / plane['extent'] < 0.01
    assert alignment['up_spread_deg'] < 1.0 and alignment['up_vs_camera_up_deg'] < 5.0
    assert 'no metric claim' in alignment['description'] and 'z = 0' in alignment['description']


def test_alignment_up_from_camera_when_no_polygons():
    cloud, views, _, m, b, _ = synthetic_sfm(5, with_polygons=False)
    rot, origin, plane, alignment = R.align_scene(cloud, views, [])
    assert alignment['up_source'] == 'camera_up'
    assert np.allclose(rot @ m, np.eye(3), atol=0.05)


def test_alignment_follows_the_spines_not_the_cameras_when_the_camera_is_rolled():
    cloud, views, sightings, m, b, ideal = synthetic_sfm(7)
    roll = np.radians(25)                                         # camera rolled 25 deg: image up is no longer shelf up
    rr = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    rolled = {}
    polys = []
    for v in ideal:
        rv = R.View(v.frame_id, v.centre, rr @ v.rotation, v.unproject)
        rolled[v.frame_id] = R.View(v.frame_id, m @ rv.centre + b, rv.rotation @ m.T, rv.unproject)
    for k, (x, y) in enumerate([(-3.0, 0.0), (-1.2, 0.3), (0.4, -0.2), (2.5, 0.1), (3.6, 0.5)]):
        for v in ideal:
            if abs(v.centre[0] - x) < 3.2:
                rv = R.View(v.frame_id, v.centre, rr @ v.rotation, v.unproject)
                polys.append(sighting(rv, (x, y), f'{v.frame_id}-o{k}'))
    rot, origin, _, alignment = R.align_scene(cloud, rolled, polys)
    assert alignment['up_source'] == 'book_polygons' and 20 < alignment['up_vs_camera_up_deg'] < 30
    assert np.allclose(rot @ m, np.eye(3), atol=0.03)             # shelf up stays +y although the camera was rolled


def test_alignment_without_a_plane_is_identity():
    rot, origin, plane, alignment = R.align_scene(np.zeros((3, 3)), {}, [])
    assert plane is None and not alignment['applied'] and np.allclose(rot, np.eye(3))


# ---------------------------------------------------------------- marker placement (aligned frame)

def views_row(n=9):
    return {f'f{i}': front_view(f'f{i}', x) for i, x in enumerate(np.linspace(-4, 4, n))}


def group(gid, views, centre_xy, frames, kind='book', **kw):
    return {'group_id': gid, 'kind': kind, 'category': 'book',
            'sightings': [sighting(views[f], centre_xy, f'{f}-{gid}', **kw) for f in frames]}


def test_triangulates_when_robust_and_projects_otherwise():
    views = views_row()
    tri = group('bk-001', views, (1.0, 0.3), ['f3', 'f4', 'f5', 'f6'])
    one = group('bk-002', views, (-2.0, -0.4), ['f2'])
    marker, none = R.place_group(tri, views, plane_tol=0.2)
    assert none is None and marker['method'] == 'triangulated' and marker['views'] >= 3
    assert np.allclose(marker['xyz'], [1.0, 0.3, 0.0], atol=0.02) and marker['baseline_angle_deg'] >= R.TRI_MIN_ANGLE_DEG
    marker, none = R.place_group(one, views, plane_tol=0.2)
    assert none is None and marker['method'] == 'plane_projected' and marker['views'] == 1
    assert np.allclose(marker['xyz'], [-2.0, -0.4, 0.0], atol=0.02) and marker['ray_residual'] is None


def test_low_parallax_group_is_plane_projected_not_triangulated():
    views = {f'f{i}': front_view(f'f{i}', 0.01 * i, z=40.0) for i in range(4)}      # far away, tiny baseline
    g = group('bk-003', views, (0.5, 0.0), list(views))
    marker, _ = R.place_group(g, views, plane_tol=0.5)
    assert marker['method'] == 'plane_projected' and marker['views'] == 4 and np.allclose(marker['xyz'], [0.5, 0.0, 0.0], atol=0.05)


def test_far_off_plane_triangulation_falls_back_to_the_plane():
    views = views_row()
    g = {'group_id': 'bk-004', 'kind': 'book', 'category': 'book', 'sightings': []}
    # a point 2.5 units in front of the plane, seen from four cameras: it triangulates, but not near the shelf plane
    for f in ['f2', 'f3', 'f4', 'f5']:
        v = views[f]
        px = project(v, (0.5, 0.2, 2.5))
        g['sightings'].append({'observation_id': f'{f}-x', 'frame_id': f, 'bbox_xyxy': [px[0] - 20, px[1] - 30, px[0] + 20, px[1] + 30],
                               'polygon_xy': None, 'sharpness': 1.0, 'confidence': 0.9})
    marker, _ = R.place_group(g, views, plane_tol=0.2)
    assert marker['method'] == 'plane_projected' and marker['xyz'][2] == 0.0


def test_unregistered_and_empty_groups_are_unplaced_with_a_reason():
    views = views_row(3)
    ghost = {'group_id': 'bk-005', 'kind': 'book', 'category': 'book', 'sightings': [sighting(front_view('fX', 0.0), (0, 0), 'fX-o0')]}
    marker, un = R.place_group(ghost, views, plane_tol=0.2)
    assert marker is None and un['group_id'] == 'bk-005' and un['code'] == 'no_registered_sighting' and '1 sighting' in un['reason']
    marker, un = R.place_group({'group_id': 'bk-006', 'kind': 'book', 'category': None, 'sightings': []}, views, plane_tol=0.2)
    assert marker is None and un['code'] == 'no_sightings'


def test_ray_that_misses_the_shelf_region_is_unplaced():
    views = views_row()
    g = group('bk-007', views, (30.0, 0.0), ['f4'])
    region = (np.array([-6.0, -3.0]), np.array([6.0, 3.0]))
    marker, un = R.place_group(g, views, plane_tol=0.2, region=region)
    assert marker is None and un['code'] == 'off_shelf'


def test_every_group_is_either_a_marker_or_unplaced_in_order():
    views = views_row()
    groups = [group('bk-001', views, (1.0, 0.3), ['f3', 'f4', 'f5']), group('bk-002', views, (-2.0, 0.0), ['f1']),
              {'group_id': 'bk-003', 'kind': 'book', 'category': None, 'sightings': [sighting(front_view('fX', 0.0), (0, 0), 'fX-o0')]},
              group('bk-004', views, (3.0, -0.2), ['f7', 'f8'])]
    markers, unplaced = R.place_markers(groups, views, plane_tol=0.2)
    assert len(markers) + len(unplaced) == len(groups) == 4
    assert [m['group_id'] for m in markers] == ['bk-001', 'bk-002', 'bk-004'] and [u['group_id'] for u in unplaced] == ['bk-003']
    assert {m['method'] for m in markers} <= {'triangulated', 'plane_projected'}


def test_marker_sightings_are_capped_sharpest_first_and_registered_only():
    views = views_row(9)
    g = group('bk-008', views, (0.0, 0.0), list(views))
    for i, s in enumerate(g['sightings']):
        s['sharpness'] = float(i)                                  # f8 is the sharpest
    g['sightings'].append(sighting(front_view('fX', 0.0), (0, 0), 'fX-o0', sharp=999.0))
    for i in range(12):                                            # more sightings than the cap
        g['sightings'].append(sighting(views['f0'], (0, 0), f'f0-extra{i}', sharp=0.5))
    marker, _ = R.place_group(g, views, plane_tol=0.2)
    got = marker['sightings']
    assert len(got) == R.MAX_SIGHTINGS and got[0]['frame_id'] == 'f8' and all(s['frame_id'] != 'fX' for s in got)
    assert set(got[0]) >= {'frame_id', 'observation_id', 'bbox_xyxy'}
    assert marker['registered_sightings'] == 9 + 12 and marker['sightings_total'] == 9 + 1 + 12


def test_marker_uses_the_polygon_centre_not_the_bbox_centre():
    v = front_view('f0', 0.0)
    skewed = {'bbox_xyxy': [0, 0, 100, 100], 'polygon_xy': [[10, 10], [30, 10], [30, 30], [10, 30]]}
    assert np.allclose(R.sighting_pixel(skewed), [20, 20])
    assert np.allclose(R.sighting_pixel({'bbox_xyxy': [0, 0, 100, 60], 'polygon_xy': None}), [50, 30])
    assert np.allclose(v.ray(R.sighting_pixel({'bbox_xyxy': [0, 0, W, H]}))[1] / np.linalg.norm(v.ray([W / 2, H / 2])[1]), [0, 0, -1])


def test_view_transform_moves_rays_consistently():
    m, b = rotation(2), np.array([1.0, -2.0, 3.0])
    v = front_view('f0', 1.0, y=0.5)
    o, d = v.ray([400, 300])
    o2, d2 = v.transformed(m, b).ray([400, 300])
    assert np.allclose(o2, m @ (o - b)) and np.allclose(d2, m @ d)
    assert np.allclose(v.transformed(m, b).up(), m @ v.up())


# ---------------------------------------------------------------- groups come from observation_ids, not the one evidence ref

def write_run(tmp_path, with_candidates=True):
    run = tmp_path / 'run'
    run.mkdir()
    box = [10, 20, 30, 80]
    obs = [{'observation_id': f'f{i:05d}-o000', 'frame_id': f'f{i:05d}', 'bbox_xyxy': box, 'polygon_xy': [[10, 20], [30, 20], [30, 80], [10, 80]],
            'confidence': 0.8 + i / 100} for i in range(5)]
    book = {'instance_id': 'bk-001', 'observation_ids': [o['observation_id'] for o in obs[:4]], 'kind': 'book',
            'evidence_refs': [{'observation_id': obs[0]['observation_id'], 'frame_id': 'f00000', 'bbox_xyxy': box}]}
    book2 = {'instance_id': 'bk-002', 'observation_ids': [obs[4]['observation_id']], 'kind': 'book', 'evidence_refs': []}
    item = {'instance_id': 'it-001', 'observation_ids': [obs[4]['observation_id']], 'kind': 'non_book', 'category': 'box',
            'evidence_refs': [{'observation_id': obs[4]['observation_id'], 'frame_id': 'f00004', 'bbox_xyxy': box}]}
    (run / 'inventory.json').write_text(json.dumps({'data': {'books': [book, book2], 'items': [item], 'counts': {'book_count': 2}}}))
    if with_candidates:
        (run / 'candidates.json').write_text(json.dumps({'data': {'observations': obs}}))
    (run / 'manifest.json').write_text(json.dumps({'data': {'frames': [
        {'frame_id': f'f{i:05d}', 'width_px': 1920, 'height_px': 1080, 'quality': {'sharpness': 10.0 * i}} for i in range(5)]}}))
    return run


def test_load_groups_joins_observation_ids_to_candidates(tmp_path):
    groups, book_count = R.load_groups(write_run(tmp_path))
    books = [g for g in groups if g['kind'] == 'book']
    assert book_count == 2 and len(books) == 2 and [g['group_id'] for g in groups] == ['bk-001', 'bk-002', 'it-001']
    first = books[0]
    assert [s['frame_id'] for s in first['sightings']] == ['f00000', 'f00001', 'f00002', 'f00003']      # not just the evidence ref
    assert first['sightings'][2]['sharpness'] == 20.0 and first['sightings'][2]['polygon_xy'][2] == [30, 80] and first['sightings'][2]['width_px'] == 1920
    assert groups[2]['kind'] == 'item' and groups[2]['category'] == 'box'                                # items are not books


def test_load_groups_falls_back_to_evidence_refs_without_candidates(tmp_path):
    groups, book_count = R.load_groups(write_run(tmp_path, with_candidates=False))
    assert book_count == 2 and [len(g['sightings']) for g in groups] == [1, 0, 1]


def test_load_groups_without_inventory_is_empty(tmp_path):
    assert R.load_groups(tmp_path) == ([], 0)


# ---------------------------------------------------------------- the whole model-dependent result (no pycolmap)

def test_scene_result_one_marker_per_book_group_in_a_shelf_aligned_frame():
    cloud, sfm_views, sightings, m, b, ideal = synthetic_sfm(13)
    ideal_views = {v.frame_id: v for v in ideal}
    books = []
    for k, (x, y) in enumerate([(-3.0, 0.0), (-1.2, 0.3), (0.4, -0.2), (2.5, 0.1), (3.6, 0.5), (1.3, 0.4)]):
        frames = [v.frame_id for v in ideal if abs(v.centre[0] - x) < 3.2]
        books.append(group(f'bk-{k + 1:03d}', ideal_views, (x, y), frames if k < 4 else frames[:1]))
    # sightings are in pixels, so they do not depend on the frame; the views are the arbitrary SfM ones
    books.append({'group_id': 'bk-007', 'kind': 'book', 'category': 'book', 'sightings': [sighting(front_view('fGone', 0.0), (0, 0), 'fGone-o0')]})
    items = [group('it-001', ideal_views, (0.0, -1.0), [ideal[8].frame_id])]
    items[0]['kind'] = 'item'
    intrinsics = {k: (F, W, H) for k in sfm_views}
    colour = np.full((len(cloud), 3), 128)
    frame_ids = [v.frame_id for v in ideal] + ['fGone', 'fAlsoGone']
    out = R.scene_result(cloud, colour, sfm_views, intrinsics, books, items, frame_ids=frame_ids, mean_error=0.7, points_total=5000, points_filtered=4000)
    json.dumps(out)                                                                   # serialisable
    assert out['status'] == 'ok' and out['registered_frames'] == 17 and out['unregistered_frames'] == ['fAlsoGone', 'fGone']
    assert out['groups_total'] == 7 and out['markers_total'] == 6 and len(out['markers']) + len(out['unplaced_groups']) == 7
    assert [u['group_id'] for u in out['unplaced_groups']] == ['bk-007'] and out['unplaced_groups'][0]['reason']
    assert out['marker_methods']['triangulated'] >= 3 and out['marker_methods']['plane_projected'] >= 1
    assert sum(out['marker_methods'].values()) == out['markers_total']
    assert len(out['item_markers']) == 1 and out['item_markers'][0]['kind'] == 'item'     # items stay out of the book count
    assert out['plane']['rms_over_extent'] < 0.01 and out['alignment']['applied']
    by_id = {mk['group_id']: mk for mk in out['markers']}
    for gid, (x, y) in {'bk-001': (-3.0, 0.0), 'bk-002': (-1.2, 0.3), 'bk-003': (0.4, -0.2), 'bk-004': (2.5, 0.1), 'bk-006': (1.3, 0.4)}.items():
        # aligned frame = ideal frame up to the (mirror-free) in-plane ambiguity of the plane fit, which up-alignment removes
        assert np.allclose(by_id[gid]['xyz'], [x, y, 0.0], atol=0.12), gid
    assert all(c['center'][2] > 4 for c in out['cameras']) and np.abs(np.array(out['points'])[:, 2]).mean() < 0.05
    assert out['mean_reprojection_error_px'] == 0.7 and out['points_kept'] == len(cloud)


def test_scene_result_without_a_plane_is_partial_and_keeps_every_group():
    views = {f'f{i}': front_view(f'f{i}', x) for i, x in enumerate([-2.0, 0.0, 2.0])}
    g = group('bk-001', views, (0.0, 0.0), list(views))
    out = R.scene_result(np.zeros((5, 3)), np.zeros((5, 3), int), views, {k: (F, W, H) for k in views}, [g], [], frame_ids=list(views))
    assert out['plane'] is None and not out['alignment']['applied']
    assert out['status'] == 'partial' and 'plane' in out['reason'].lower()
    assert len(out['markers']) + len(out['unplaced_groups']) == out['groups_total'] == 1


def test_scene_result_with_few_placeable_groups_is_only_partial():
    cloud, sfm_views, _, m, b, ideal = synthetic_sfm(17)
    books = [{'group_id': f'bk-{i:03d}', 'kind': 'book', 'category': 'book', 'sightings': []} for i in range(1, 5)]
    out = R.scene_result(cloud, np.zeros((len(cloud), 3), int), sfm_views, {k: (F, W, H) for k in sfm_views}, books, [],
                         frame_ids=list(sfm_views))
    assert out['status'] == 'partial' and out['markers'] == [] and len(out['unplaced_groups']) == 4
    assert out['unplaced_groups'][0]['reason'] and 'could be placed' in out['reason']


# ---------------------------------------------------------------- frame diet

def _frame(i, interval, sharp, *, selected=False, gated=False):
    return {'frame_id': f'f{i:05d}', 'image_path': f'frames/f{i:05d}.jpg', 'interval_index': interval, 'selected': selected,
            'quality': {'sharpness': sharp, 'blur_gated': gated}}


def _manifest_frames(intervals=40, per=5):
    """`intervals` pan intervals of `per` frames; the keyframe is the sharpest of its interval; every 7th interval is blurry (all gated)."""
    frames = []
    for k in range(intervals):
        blurry = k % 7 == 3
        sharp = [20.0 + j if blurry else 100.0 + (j * 13) % 50 for j in range(per)]
        best = max(range(per), key=lambda j: sharp[j])
        for j in range(per):
            frames.append(_frame(len(frames), k, sharp[j], selected=j == best, gated=blurry))
    return frames


def test_all_rule_keeps_every_frame_including_gated():
    frames = _manifest_frames()
    kept, info = R.select_frames(frames, 'all')
    assert [f['frame_id'] for f in kept] == [f['frame_id'] for f in frames]
    assert info['frames_total'] == info['frames_used'] == len(frames) and info['applied'] == 'all'


def test_diet_is_keyframe_plus_sharpest_non_gated_neighbour_per_interval_in_time_order():
    frames = _manifest_frames()
    kept, info = R.select_frames(frames, 'diet')
    ids = [f['frame_id'] for f in kept]
    assert ids == sorted(ids)                                          # time order
    assert not any(f['quality']['blur_gated'] for f in kept)           # never a gated frame
    by_interval = {}
    for f in kept:
        by_interval.setdefault(f['interval_index'], []).append(f)
    blurry = {k for k in range(40) if k % 7 == 3}
    assert set(by_interval) == set(range(40)) - blurry                 # an all-gated interval contributes nothing
    for k, fs in by_interval.items():
        members = [f for f in frames if f['interval_index'] == k]
        key = next(f for f in members if f['selected'])
        assert key in fs and len(fs) == 2
        other = max((f for f in members if f is not key), key=lambda f: f['quality']['sharpness'])
        assert other in fs                                             # the sharpest other frame
    assert info['frames_used'] == len(kept) == 2 * len(by_interval)
    assert info['gated_keyframes_dropped'] == len(blurry) and info['neighbours_added'] == len(by_interval)


def test_keyframes_rule_is_the_non_gated_keyframes_only():
    frames = _manifest_frames()
    kept, info = R.select_frames(frames, 'keyframes')
    assert all(f['selected'] and not f['quality']['blur_gated'] for f in kept)
    assert len(kept) == sum(1 for f in frames if f['selected'] and not f['quality']['blur_gated'])
    assert info['neighbours_added'] == 0


def test_interval_with_only_the_keyframe_adds_no_neighbour():
    frames = [_frame(i, i, 100.0 + i, selected=True) for i in range(30)]
    kept, info = R.select_frames(frames, 'diet')
    assert len(kept) == 30 and info['neighbours_added'] == 0


def test_cap_thins_by_sharpness_and_keeps_time_order():
    frames = _manifest_frames()
    kept, info = R.select_frames(frames, 'diet', max_frames=30)
    assert len(kept) == 30 and info['capped_to'] == 30 and [f['frame_id'] for f in kept] == sorted(f['frame_id'] for f in kept)
    full, _ = R.select_frames(frames, 'diet')
    floor = sorted((f['quality']['sharpness'] for f in full), reverse=True)[29]
    assert min(f['quality']['sharpness'] for f in kept) >= floor
    again, _ = R.select_frames(frames, 'diet', max_frames=30)
    assert [f['frame_id'] for f in again] == [f['frame_id'] for f in kept]


def test_cap_applies_to_the_all_rule_too():
    kept, info = R.select_frames(_manifest_frames(), 'all', max_frames=50)
    assert len(kept) == 50 and info['applied'] == 'all'


def test_manifest_without_keyframe_metadata_falls_back_to_all_frames():
    frames = [{'frame_id': f'f{i:05d}', 'image_path': f'frames/f{i:05d}.jpg', 'quality': {'sharpness': 50.0}} for i in range(40)]
    kept, info = R.select_frames(frames, 'diet')
    assert len(kept) == 40 and 'fallback' in info


def test_mostly_blurry_run_falls_back_instead_of_starving_sfm():
    frames = [_frame(i, i // 4, 10.0, selected=i % 4 == 0, gated=True) for i in range(80)]
    kept, info = R.select_frames(frames, 'diet')
    assert len(kept) == 80 and 'fallback' in info and info['applied'] == 'all'


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        R.select_frames(_manifest_frames(), 'sparse')


@pytest.mark.parametrize('kept,total,expected', [
    (643, 643, 12), (300, 643, 6), (400, 643, 7), (500, 643, 9), (44, 157, 6), (150, 157, 11), (10, 1000, 6), (5, 0, 12)])
def test_sequential_overlap_spans_the_same_time_as_twelve_full_frames(kept, total, expected):
    assert R.sequential_overlap(kept, total) == expected


def test_build_reports_frames_total_used_rule_and_timings(tmp_path, monkeypatch):
    """The recon3d.json header fields, with pycolmap replaced by a stub that registers nothing."""
    import sys
    import types
    run = tmp_path / 'run'
    (run / 'frames').mkdir(parents=True)
    frames = _manifest_frames(intervals=20)
    for f in frames:
        (run / f['image_path']).write_bytes(b'x')
    (run / 'manifest.json').write_text(json.dumps({'data': {'frames': frames}}))
    seen = {}

    class Opts:
        def __init__(self):
            self.sift = types.SimpleNamespace()

    fake = types.SimpleNamespace(
        __version__='0', FeatureExtractionOptions=Opts, ImageReaderOptions=Opts, SequentialPairingOptions=Opts, IncrementalPipelineOptions=Opts,
        CameraMode=types.SimpleNamespace(SINGLE=0), Device=types.SimpleNamespace(cpu=0),
        extract_features=lambda db, images, extraction_options=None, **k: seen.update(
            images=sorted(p.name for p in images.iterdir()), size=extraction_options.max_image_size, features=extraction_options.sift.max_num_features),
        match_sequential=lambda db, pairing_options=None, **k: seen.update(overlap=pairing_options.overlap, quadratic=pairing_options.quadratic_overlap),
        incremental_mapping=lambda *a, **k: {})
    monkeypatch.setitem(sys.modules, 'pycolmap', fake)
    monkeypatch.delenv('LIBRARY_RECON3D_FRAMES', raising=False)
    monkeypatch.delenv('LIBRARY_RECON3D_PROFILE', raising=False)
    default = json.loads(R.build(run, tmp_path / 'run_recon3d_v0').read_text())
    assert default['frames_used'] == 100 and default['diet']['applied'] == 'all' and seen['overlap'] == 12    # density is the default
    preset = R.PROFILES['fast']                                                                               # 100 frames: auto = fast
    assert default['library']['profile'] == 'fast' and default['library']['profile_requested'] == 'auto'
    assert (seen['size'], seen['features'], seen['quadratic']) == (preset['max_image_size'], preset['max_features'], preset['quadratic'])
    default_run = tmp_path / 'small'
    (default_run / 'frames').mkdir(parents=True)
    small = _manifest_frames(intervals=11)[:R.AUTO_FAST_MIN_FRAMES - 1]
    for f in small:
        (default_run / f['image_path']).write_bytes(b'x')
    (default_run / 'manifest.json').write_text(json.dumps({'data': {'frames': small}}))
    below = json.loads(R.build(default_run, tmp_path / 'small_recon3d_v1').read_text())
    assert below['library']['profile'] == 'default' and (seen['size'], seen['features'], seen['quadratic']) == (R.IMAGE_SIZE, R.MAX_FEATURES, R.QUADRATIC)
    monkeypatch.setenv('LIBRARY_RECON3D_FRAMES', 'diet')
    path = R.build(run, tmp_path / 'run_recon3d_v1', profile='default')
    out = json.loads(path.read_text())
    assert out['frames_total'] == 100 and out['frames_used'] == len(seen['images']) < 100
    assert out['diet']['rule'] == 'diet' and out['diet']['frames_used'] == out['frames_used']
    assert seen['overlap'] == out['diet']['overlap'] == R.sequential_overlap(out['frames_used'], 100)
    assert {'features_s', 'matching_s', 'mapping_s', 'markers_s', 'total_s', 'total_cpu_s'} <= set(out['timings'])
    assert (seen['size'], seen['features'], seen['quadratic']) == (R.IMAGE_SIZE, R.MAX_FEATURES, R.QUADRATIC)
    monkeypatch.setenv('LIBRARY_RECON3D_FRAMES', 'all')
    monkeypatch.setenv('LIBRARY_RECON3D_PROFILE', 'fast')
    fast = json.loads(R.build(run, tmp_path / 'run_recon3d_v2').read_text())
    assert fast['frames_used'] == 100 and seen['overlap'] == 12 and fast['diet']['applied'] == 'all'
    preset = R.PROFILES['fast']
    assert (seen['size'], seen['features'], seen['quadratic']) == (preset['max_image_size'], preset['max_features'], preset['quadratic'])
    assert fast['library']['profile'] == 'fast'
    monkeypatch.setenv('LIBRARY_RECON3D_PROFILE', 'nope')
    with pytest.raises(ValueError):
        R.build(run, tmp_path / 'run_recon3d_v3')
