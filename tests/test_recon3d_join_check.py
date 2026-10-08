"""A joined 3D map must keep every book on ONE shelf plane (recon3d_merge.ShelfCheck / split_check / final_check, recon3d.plane_consistency).
Fake models only (no pycolmap). World: the shelf face is z = 0, cameras stand 3 units in front of it (+z) facing it, world up is +y."""
import types

import numpy as np
import pytest

from pipeline import recon3d as R
from pipeline import recon3d_merge as M
from tests.test_recon3d_merge import _Rec, _image, _point, _rot

_FACE = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])        # world->camera of a camera facing the shelf from +z, image up = +y
UP = np.array([0.0, 1.0, 0.0])


def _project(r_wc, centre, x):
    c = r_wc @ (np.asarray(x, float) - centre)
    return np.array([960 + 1000 * c[0] / c[2], 540 + 1000 * c[1] / c[2]])


def _shelf_part(xs, first, books_at, sim=None):
    """Cameras at x in `xs` (frames f<first>...), a grid of points on the plane, book spines (0.05 wide, 0.25 tall) at `books_at`.
    With `sim` = (s, R, t) the model is expressed in its own frame: world = s R own + t."""
    s, rot, t = sim if sim is not None else (1.0, np.eye(3), np.zeros(3))

    def own(x):
        return rot.T @ (np.asarray(x, float) - t) / s
    pts = [_point(own([x, y, 0.0])) for x in np.linspace(xs[0] - 1, xs[-1] + 1, 25) for y in np.linspace(-1, 1, 9)]
    pts += [_point(own([x, y, -0.4])) for x in np.linspace(xs[0] - 1, xs[-1] + 1, 6) for y in (-1.5, 1.5)]   # a little clutter
    ids = list(range(1, len(pts) + 1))
    images, sightings = [], {}
    for i, x in enumerate(xs):
        name, centre = f'f{first + i}', np.array([x, 0.0, 3.0])
        im = _image(f'{name}.jpg', _FACE @ rot, own(centre))
        im.points2D = [types.SimpleNamespace(has_point3D=lambda: True, point3D_id=k) for k in ids]
        images.append(im)
        for b, (bx, by) in enumerate(books_at):
            corners = [[bx - .025, by - .125, 0], [bx + .025, by - .125, 0], [bx + .025, by + .125, 0], [bx - .025, by + .125, 0]]
            poly = [_project(_FACE, centre, c).tolist() for c in corners]
            if all(0 < p[0] < 1920 and 0 < p[1] < 1080 for p in poly):
                xy = np.array(poly)
                sightings.setdefault(f'b{bx:.2f}_{by:.2f}', []).append(
                    {'observation_id': f'o{first + i}_{b}', 'frame_id': name, 'bbox_xyxy': [*xy.min(0), *xy.max(0)], 'polygon_xy': poly,
                     'sharpness': 1.0, 'confidence': 0.9, 'width_px': 1920, 'height_px': 1080})
    return _Rec(images, pts), sightings


def _books(*sighting_maps):
    merged = {}
    for m in sighting_maps:
        for gid, ss in m.items():
            merged.setdefault(gid, []).extend(ss)
    return [{'group_id': gid, 'kind': 'book', 'category': 'book', 'sightings': ss} for gid, ss in merged.items()]


BOOKS_A = [(x, y) for x in np.arange(0.2, 2.6, 0.3) for y in (-0.5, 0.5)]
BOOKS_B = [(x, y) for x in np.arange(3.4, 5.8, 0.3) for y in (-0.5, 0.5)]
PIVOT = np.array([3.0, 0.0, 0.0])


def _swing(deg):
    """A rotation about world up through the gap between the two parts (the IESFULLTest1 failure: level horizon, wrong yaw)."""
    r = _rot('y', deg)
    return 1.0, r, PIVOT - r @ PIVOT


def _two_parts(wrong_yaw_deg=0.0):
    true = (0.5, _rot('y', 30), np.array([1.0, 0.2, -0.3]))
    a, sa = _shelf_part(np.arange(0, 3, 0.3), 0, BOOKS_A)
    b, sb = _shelf_part(np.arange(3.0, 6.0, 0.3), 10, BOOKS_B, sim=true)
    names = {f'f{i}.jpg': f'f{i}' for i in range(20)}
    return a, b, M.compose(_swing(wrong_yaw_deg), true), M.ShelfCheck(_books(sa, sb), names)


def test_a_correct_join_keeps_every_book_on_the_shelf_plane():
    a, b, sim, shelf = _two_parts(0.0)
    ref = shelf.reference([shelf.unit(a)])
    assert ref['plane'] is not None and ref['spine'] == pytest.approx(0.25, rel=0.05)
    ok, ev = shelf.test(shelf.unit(b, sim), ref, up=UP)
    assert ok, ev
    assert ev['angle_deg'] < 1 and ev['median_offset_spines'] < 0.05 and ev['cameras_room_side'] == 1.0 and ev['books'] >= 10


def test_a_join_rotated_about_the_vertical_is_refused_with_the_reason():
    a, b, sim, shelf = _two_parts(20.0)                   # the horizon stays level: the tilt check alone accepts this
    ok, ev = shelf.test(shelf.unit(b, sim), shelf.reference([shelf.unit(a)]), up=UP)
    assert not ok and ev['angle_deg'] == pytest.approx(20, abs=1) and ev['yaw_deg'] == pytest.approx(20, abs=1)
    assert 'about the vertical' in ev['reason']


def test_a_join_shifted_off_the_plane_is_refused_on_the_book_offset():
    a, b, (s, r, t), shelf = _two_parts(0.0)
    ok, ev = shelf.test(shelf.unit(b, (s, r, t + np.array([0, 0, -0.5]))), shelf.reference([shelf.unit(a)]))   # 2 spine heights behind
    assert not ok and ev['median_offset_spines'] == pytest.approx(2.0, abs=0.1) and 'spine heights' in ev['reason']


def test_a_part_without_books_cannot_be_verified_and_is_refused():
    a, b, sim, shelf = _two_parts(0.0)
    bare = M.ShelfCheck([g for g in shelf.books if all(int(s['frame_id'][1:]) < 10 for s in g['sightings'])], shelf.names_to_frame)
    ok, ev = bare.test(bare.unit(b, sim), bare.reference([bare.unit(a)]))
    assert not ok and ev['books'] == 0 and 'cannot be checked' in ev['reason']


def _merged_rec(a, b_world):
    """One model holding both parts (as a re-map would)."""
    rec = _Rec([], [])
    n = len(a.points3D)
    for im in b_world.images.values():
        im.points2D = [types.SimpleNamespace(has_point3D=lambda: True, point3D_id=p.point3D_id + n) for p in im.points2D]
    rec.images = {i + 1: im for i, im in enumerate(list(a.images.values()) + list(b_world.images.values()))}
    rec.points3D = {i + 1: p for i, p in enumerate(list(a.points3D.values()) + list(b_world.points3D.values()))}
    return rec


@pytest.mark.parametrize('yaw, dropped', [(0.0, False), (20.0, True)])
def test_split_check_drops_a_first_pass_fragment_glued_at_a_wrong_yaw_inside_a_remapped_model(yaw, dropped):
    a, _, _, shelf = _two_parts()
    s, r, t = _swing(yaw)
    b, _ = _shelf_part(np.arange(3.0, 6.0, 0.3), 10, BOOKS_B, sim=(1.0, r.T, -r.T @ t))     # B swung by `yaw` inside the one model
    names = [f'f{i}.jpg' for i in range(20)]
    frags = [{'model': 0, 'size': 10, 'frames': list(range(10))}, {'model': 1, 'size': 10, 'frames': list(range(10, 20))}]
    keep, units, rows = M.split_check(_merged_rec(a, b), frags, names, shelf)
    if not dropped:
        assert keep is None and rows[1]['decision'] == 'kept', rows
        return
    assert keep == {f'f{i}.jpg' for i in range(10)} and rows[1]['decision'].startswith('dropped') and rows[1]['angle_deg'] > 15
    refused = M.refused_joins({'chain': {}, 'remap': {'main_model_check': {'parts': rows}}})
    assert refused[0]['what'].startswith('first-pass fragment 1') and 'deg off the main shelf plane' in refused[0]['reason']


def test_final_check_keeps_the_largest_consistent_sub_join():
    a, b, good, shelf = _two_parts(0.0)
    bad = _two_parts(25.0)[2]
    kept, rows = M.final_check({0: a, 1: b, 2: b}, [(0, M.IDENTITY, None), (1, good, None), (2, bad, None)], shelf)
    assert [k for k, _, _ in kept] == [0, 1]
    assert rows[0]['model'] == 2 and rows[0]['decision'].startswith('dropped from the join')


def test_model_inputs_uses_only_the_kept_frames_and_their_points():
    a, _, _, _ = _two_parts(0.0)
    rec = _merged_rec(a, _shelf_part(np.arange(3.0, 6.0, 0.3), 10, [])[0])
    names = {f'f{i}.jpg': f'f{i}' for i in range(20)}
    xyz, _, views, _, _, n_all, _ = R.model_inputs([(rec, None, {f'f{i}.jpg' for i in range(10)})], names)
    assert set(views) == {f'f{i}' for i in range(10)} and n_all == len(a.points3D)


def _views(rec, names):
    out = {}
    for im in rec.images.values():
        out[names[im.name]] = R._view_of(im, rec.cameras[1], names[im.name])
    return out


def test_plane_consistency_counts_the_books_on_the_plane_in_spine_heights():
    a, sa = _shelf_part(np.arange(0, 3, 0.3), 0, BOOKS_A)
    names = {f'f{i}.jpg': f'f{i}' for i in range(10)}
    views = _views(a, names)
    pc = R.plane_consistency(_books(sa), views, np.array([0.0, 0.0, 1.0]), np.zeros(3), min_books=5)
    assert pc['evaluated'] and pc['share'] == 1.0 and pc['spine_height'] == pytest.approx(0.25, rel=0.05) and pc['books_evaluated'] >= 10
    off = R.plane_consistency(_books(sa), views, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.3]), min_books=5)
    # 0.3 off; the spines cast on that nearer plane measure 0.25 x 2.7 / 3 = 0.225, so 1.33 spine heights
    assert off['share'] == 0.0 and off['median_abs_offset_spines'] == pytest.approx(0.3 / 0.225, abs=0.03)


def _map(root, name, ratio, consistent=None):
    import json
    d = root / name
    d.mkdir()
    body = {'status': 'ok', 'registered_ratio': ratio, 'alignment': {'applied': True}, 'markers': [{'group_id': 'g'}], 'cameras': [{'frame_id': 'f'}]}
    if consistent is not None:
        body['plane_consistency'] = {'consistent': consistent}
    (d / 'recon3d.json').write_text(json.dumps(body))


def test_map_pickers_skip_inconsistent_maps_and_prefer_checked_ones(tmp_path):
    import json
    from app import measurement as MS
    from pipeline import instances as I
    from pipeline import room_pan as RP
    (tmp_path / 'sweep_x').mkdir()
    _map(tmp_path, 'sweep_x_recon3d_v1', 0.30, None)       # first map built before the plane check
    _map(tmp_path, 'sweep_x_recon3d_v2', 0.75, None)       # joined map built before the plane check (unchecked, may be wrongly joined)
    _map(tmp_path, 'sweep_x_recon3d_v3', 0.34, True)       # plane-checked map, one flat shelf face
    _map(tmp_path, 'sweep_x_recon3d_v4', 0.80, False)      # plane-checked map marked inconsistent
    assert R.map_consistency_at(tmp_path / 'sweep_x_recon3d_v4') == 'inconsistent'
    assert I.best_recon3d(tmp_path / 'sweep_x').parent.name == 'sweep_x_recon3d_v3'
    assert MS.latest_recon3d(tmp_path, 'sweep_x') == 'sweep_x_recon3d_v3'
    assert RP.latest_recon3d(tmp_path / 'sweep_x').parent.name == 'sweep_x_recon3d_v3'
    for n, recon in ((1, 'sweep_x_recon3d_v3'), (2, 'sweep_x_recon3d_v4')):
        (tmp_path / f'sweep_x_measure_v{n}').mkdir()
        (tmp_path / f'sweep_x_measure_v{n}' / 'auto_measure.json').write_text(json.dumps({'data': {'recon3d': recon}}))
    assert MS.latest_measure(tmp_path, 'sweep_x').parent.name == 'sweep_x_measure_v1'     # v2 measured the inconsistent map


def test_a_map_whose_books_are_off_its_plane_is_not_ok():
    assert R.status_for(0.9, [1], plane_consistent=False) == 'partial'
    assert R.status_for(0.9, [1], plane_consistent=True) == 'ok'
