"""Synthetic geometry / provenance / marker tests for pipeline.panorama (not shelf accuracy labels)."""
import json
import math

import cv2
import numpy as np
import pytest

from pipeline import panorama as P
from pipeline.panorama import (project, safe_transform, corners, canvas_geometry, LIMITS, build, sha, audit, next_output,
                               vanishing_point, rectifying_homography, leveling_matrix, book_axis_lines)


def frame(w=1920, h=1080):
    return {'width_px': w, 'height_px': h}


def test_projection_direction_inverse_and_roundtrip():
    H = np.array([[.8, .02, 110.], [-.01, .7, 40.], [.00002, .00003, 1.]])
    points = np.array([[0, 0], [5, 9], [123, 456], [1919, 1079]], float)
    q = project(points, H)
    assert np.allclose(q[0], [110, 40])
    assert np.allclose(project(q, np.linalg.inv(H)), points, atol=1e-9)
    A = np.array([[1, 0, 300], [0, 1, 20], [0, 0, 1.]])
    B = np.array([[1, 0, -100], [0, 1, 5], [0, 0, 1.]])
    assert np.allclose(project([[10, 10]], A @ B), [[210, 35]])
    assert not np.allclose(project([[10, 10]], H @ A), project([[10, 10]], A @ H))


def test_differing_native_resolutions_are_not_normalized_implicitly():
    large, small = frame(), frame(960, 540)
    H = np.diag([2., 2., 1.])
    assert np.allclose(project(corners(small), H), corners(large))
    assert safe_transform(small, H)[0]
    T, w, h = canvas_geometry([corners(large), project(corners(small), H)])
    assert (w, h) == (1920, 1080)
    assert np.allclose(project(corners(small), T @ H), corners(large))


@pytest.mark.parametrize('H', [
    np.zeros((3, 3)), np.diag([-1, 1, 1]), np.diag([20, 20, 1]),
    np.array([[1., 0, 0], [0, 1, 0], [-.001, 0, 1]]),
    np.array([[1., 0, 0], [0, 1, 0], [0, 0, float('nan')]])])
def test_invalid_or_explosive_transform_rejected(H):
    assert not safe_transform(frame(), H)[0]


def test_zoom_along_a_chain_is_legitimate_but_bounded():
    assert safe_transform(frame(), np.diag([2.5, 2.5, 1]))[0]      # the tilt-up pan of a tall shelf grows 2.5x
    assert not safe_transform(frame(), np.diag([0.05, 0.05, 1]))[0]


def test_canvas_is_bounded_before_allocation_and_rescaled_not_rejected():
    T, w, h = canvas_geometry([corners(frame()), corners(frame()) + [15000, 0]])
    assert max(w, h) <= LIMITS['max_output_extent'] and w * h <= LIMITS['max_output_pixels'] and T[0, 0] < 1
    T, w, h = canvas_geometry([corners(frame(5000, 2000))])
    assert max(w, h) <= LIMITS['max_output_extent'] and T[0, 0] < 1
    assert np.allclose(project(project([[230, 510]], T), np.linalg.inv(T)), [[230, 510]])
    T, w, h = canvas_geometry([corners(frame(400, 300))], unit_scale=0.5)
    assert (w, h) == (200, 150)


def test_next_output_is_versioned_and_never_reuses(tmp_path):
    assert next_output(tmp_path, 'sweep_x') == 'sweep_x_panorama_v1'
    (tmp_path / 'sweep_x_panorama_v1').mkdir()
    (tmp_path / 'sweep_x_panorama_v3').mkdir()
    assert next_output(tmp_path, 'sweep_x') == 'sweep_x_panorama_v2'


# ------------------------------------------------------------------------------------------ rectification
def camera_lines(pitch_deg, n=60, focal=1500., size=(1920, 1080), seed=1):
    """Image lines of vertical world lines seen by a camera pitched down by pitch_deg (world 'up' = -Y_world)."""
    rng = np.random.default_rng(seed)
    pitch = math.radians(pitch_deg)
    R = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])   # world -> camera
    K = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1.]])
    lines, mids = [], []
    for _ in range(n):
        x, z = rng.uniform(-0.8, 0.8), rng.uniform(1.5, 2.5)
        pts = np.array([[x, y, z] for y in (-0.4, 0.4)])
        cam = pts @ R.T
        img = (K @ cam.T).T
        q = img[:, :2] / img[:, 2:]
        l = np.cross(np.append(q[0], 1), np.append(q[1], 1))
        lines.append(l / math.hypot(l[0], l[1]))
        mids.append(q.mean(0))
    return np.array(lines), np.array(mids), K


def test_vanishing_point_and_rectification_make_verticals_vertical():
    lines, mids, K = camera_lines(25)
    vp, inliers = vanishing_point(lines, mids, tol_deg=0.5)
    assert inliers.mean() > 0.95
    Hr = rectifying_homography(vp, 1920, 1080, 1500.)
    for l, m in zip(lines[:20], mids[:20]):
        d = np.array([-l[1], l[0]])
        a = project([m, m + d * 100], Hr)
        v = a[1] - a[0]
        assert abs(v[0]) / np.linalg.norm(v) < 0.02          # vertical after rectification


def test_vanishing_point_ignores_outlier_lines_and_needs_two_lines():
    lines, mids, _ = camera_lines(15, n=50)
    noise = np.random.default_rng(3).normal(size=(20, 3))
    noise[:, :2] /= np.linalg.norm(noise[:, :2], axis=1, keepdims=True)
    vp, inliers = vanishing_point(np.vstack([lines, noise]), np.vstack([mids, np.random.default_rng(4).uniform(0, 1000, (20, 2))]), tol_deg=0.5)
    assert inliers[:50].mean() > 0.9 and inliers[50:].mean() < 0.3
    assert vanishing_point(lines[:1], mids[:1]) is None


def test_leveling_removes_sloping_boards_and_leaning_verticals():
    img = np.zeros((600, 500), np.uint8)
    for y in (100, 300, 500):                                    # boards sloping 12 degrees
        cv2.line(img, (20, y), (480, y + int(460 * math.tan(math.radians(12)))), 255, 2)
    for x in (60, 250, 440):                                     # verticals
        cv2.line(img, (x, 20), (x, 580), 255, 2)
    L, info = leveling_matrix(img)
    assert abs(info['horizontal_tilt_deg'] - 12) < 1.5 and abs(info['vertical_lean_deg']) < 1.5
    out = project([[0, 0], [100, 100 * math.tan(math.radians(12))]], L)
    assert abs(out[1][1] - out[0][1]) < 2.5                      # the sloping direction is level now
    assert abs(np.linalg.det(L[:2, :2]) - 1) < 1e-6              # unit area: no scale change
    assert np.allclose(leveling_matrix(np.zeros((200, 200), np.uint8))[0], np.eye(3))


def test_book_axis_lines_use_only_upright_spines():
    def box(cx, cy, w, h, angle=0.):
        a = math.radians(angle)
        c = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return ((c @ r.T) + [cx, cy]).tolist()
    obs = [{'kind': 'book', 'frame_id': 'f0', 'polygon_xy': box(100, 200, 30, 160)},
           {'kind': 'book', 'frame_id': 'f0', 'polygon_xy': box(300, 200, 30, 160, 20)},
           {'kind': 'book', 'frame_id': 'f0', 'polygon_xy': box(500, 200, 160, 30)},          # lying
           {'kind': 'non_book', 'frame_id': 'f0', 'polygon_xy': box(100, 200, 30, 160)},
           {'kind': 'book', 'frame_id': 'f9', 'polygon_xy': box(100, 200, 30, 160)}]           # frame not tracked
    lines, mids = book_axis_lines(obs, {'f0': np.eye(3)})
    assert len(lines) == 2 and np.allclose(mids[0], [100, 200], atol=1e-6)


# ------------------------------------------------------------------------------------------ build on a synthetic run
STEP = 35


def synthetic_run(root, name='run', frames=14, blank=(), groups_only_in_blank=False, with_inventory=True, w=640, h=360):
    """A textured wall panned to the right; books are coloured spines with known polygons (the 'ground truth')."""
    rng = np.random.default_rng(0)
    wall = np.full((h + 40, w * 3, 3), 128, np.uint8)
    for _ in range(900):                                    # many shapes of varied size: trackable corners at several scales
        x0, y0 = int(rng.integers(0, w * 3)), int(rng.integers(0, h + 40))
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < .5:
            cv2.rectangle(wall, (x0, y0), (x0 + int(rng.integers(6, 60)), y0 + int(rng.integers(6, 60))), colour, -1)
        else:
            cv2.circle(wall, (x0, y0), int(rng.integers(3, 25)), colour, -1)
    wall = cv2.GaussianBlur(wall, (0, 0), 0.8)
    book_boxes = []
    for i in range(12):
        x0 = 120 + i * 130
        book_boxes.append((x0, 80, x0 + 40, 280))
        cv2.rectangle(wall, (x0, 80), (x0 + 40, 280), tuple(int(v) for v in rng.integers(40, 255, 3)), -1)
        cv2.putText(wall, str(i), (x0 + 5, 180), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    d = root / name
    (d / 'frames').mkdir(parents=True)
    step = STEP
    manifest_frames, observations, seen = [], [], {}
    for k in range(frames):
        fid = f'f{k:05d}'
        x = k * step
        img = wall[0:h, x:x + w].copy()
        if k in blank:
            img = np.full_like(img, 128)
        cv2.imwrite(str(d / 'frames' / f'{fid}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        manifest_frames.append({'frame_id': fid, 'timestamp_s': round(k * .2, 3), 'native_frame_index': k, 'image_path': f'frames/{fid}.jpg',
                                'width_px': w, 'height_px': h, 'quality': {'sharpness': 100.0 + k}})
        for b, (x0, y0, x1, y1) in enumerate(book_boxes):
            if x <= x0 and x1 <= x + w:
                oid = f'{fid}-o{b:03d}'
                poly = [[x0 - x, y0], [x1 - x, y0], [x1 - x, y1], [x0 - x, y1]]
                observations.append({'observation_id': oid, 'frame_id': fid, 'kind': 'book', 'category': 'book', 'confidence': .9,
                                     'bbox_xyxy': [x0 - x, y0, x1 - x, y1], 'polygon_xy': poly})
                seen.setdefault(b, []).append((oid, fid))
    (d / 'manifest.json').write_text(json.dumps({'run_id': name, 'data': {'input': {'sha256': 'c' * 64, 'scan_label': None}, 'frames': manifest_frames}}))
    (d / 'candidates.json').write_text(json.dumps({'run_id': name, 'data': {'observations': observations}}))
    if with_inventory:
        books = [{'instance_id': f'bk-{b + 1:03d}', 'observation_ids': [o for o, _ in sightings], 'uncertain_association': False}
                 for b, sightings in sorted(seen.items())]
        (d / 'inventory.json').write_text(json.dumps({'run_id': name, 'data': {'books': books, 'items': [], 'counts': {'book_count': len(books)}}}))
    return d, book_boxes, seen


def test_build_stitches_all_frames_into_one_component_with_one_marker_per_book(tmp_path):
    src, boxes, seen = synthetic_run(tmp_path)
    out = tmp_path / 'run_panorama_v1'
    path = build(src, out)
    d = json.loads(path.read_text())
    assert d['schema_version'] == 3 and d['status'] == 'partial' and d['source_run'] == 'run'
    assert len(d['components']) == 1 and d['unregistered_frames'] == [] and d['metrics']['registered_ratio'] == 1.0
    c = d['components'][0]
    assert len(c['source_frames']) == 14 and c['status'] == 'planar_diagnostic_stitch'
    book_count = d['book_count']
    assert book_count == len(seen) == len(c['markers']) == d['markers_total'] and d['unplaced_groups'] == []
    ids = [m['group_id'] for m in c['markers']]
    assert len(set(ids)) == len(ids)                                   # exactly one marker per group
    # Markers sit on the right spine: x spacing of consecutive books is ~130 px in the plane (canvas scale 1).
    xs = sorted(m['x'] for m in c['markers'])
    assert np.allclose(np.diff(xs), 130, atol=6)
    assert max(m['spread_px'] for m in c['markers']) < 6               # all sightings of a group agree
    assert c['rectification']['applied'] and c['rectification']['camera_pitch_deg'] < 1   # fronto-parallel pan: the rectification is the identity
    for name in ('image', 'overlay_image', 'group_overlay_image', 'coverage_mask', 'pixel_source_index'):
        assert (out / c[name]).is_file()
    assert audit(path)['markers_checked'] == book_count


def test_stitch_geometry_matches_the_known_pan(tmp_path):
    src, _, _ = synthetic_run(tmp_path)
    d = json.loads(build(src, tmp_path / 'run_panorama_v1').read_text())
    c = d['components'][0]
    Hs = [np.array(f['original_to_panorama_3x3']) for f in c['source_frames']]
    shifts = [project([[0, 0]], H)[0][0] for H in Hs]
    assert np.allclose(np.diff(shifts), STEP, atol=1.5)                # the pan advances STEP px per frame
    im = cv2.imread(str(tmp_path / 'run_panorama_v1' / c['image']))
    assert im.shape[1] > 640 + 13 * STEP - 20 and np.count_nonzero(cv2.imread(str(tmp_path / 'run_panorama_v1' / c['coverage_mask']), 0)) / (im.shape[0] * im.shape[1]) > .98


def test_untracked_frame_is_listed_and_its_only_group_is_unplaced_not_dropped(tmp_path):
    src, _, _ = synthetic_run(tmp_path, blank=(5,))
    d = json.loads(build(src, tmp_path / 'run_panorama_v1').read_text())
    listed = {u['frame_id']: u for u in d['unregistered_frames']}
    assert 'f00005' in listed and listed['f00005']['stage'] == 'tracking'
    assert d['components'][0]['source_frames'] and 'f00005' not in {f['frame_id'] for f in d['components'][0]['source_frames']}
    assert d['markers_total'] + len([u for u in d['unplaced_groups'] if u['kind'] == 'book']) == d['book_count']
    assert any(not f['registered'] and f['frame_id'] == 'f00005' and f['unregistered_reason'] for f in d['source_frames'])
    assert audit(tmp_path / 'run_panorama_v1' / 'panorama.json')['status'] == 'verified_artifact_integrity'


def test_group_seen_only_in_unregistered_frames_goes_to_unplaced_with_reason(tmp_path):
    src, _, seen = synthetic_run(tmp_path)
    inv = json.loads((src / 'inventory.json').read_text())
    inv['data']['books'].append({'instance_id': 'bk-900', 'observation_ids': ['f99999-o000'], 'uncertain_association': True})
    inv['data']['counts']['book_count'] += 1
    (src / 'inventory.json').write_text(json.dumps(inv))
    cand = json.loads((src / 'candidates.json').read_text())
    cand['data']['observations'].append({'observation_id': 'f99999-o000', 'frame_id': 'f99999', 'kind': 'book', 'category': 'book', 'confidence': .5,
                                         'bbox_xyxy': [10, 10, 40, 100], 'polygon_xy': [[10, 10], [40, 10], [40, 100], [10, 100]]})   # a frame that is in no component
    (src / 'candidates.json').write_text(json.dumps(cand))
    d = json.loads(build(src, tmp_path / 'run_panorama_v1').read_text())
    un = {u['group_id']: u for u in d['unplaced_groups']}
    assert set(un) == {'bk-900'} and 'f99999' in un['bk-900']['reason']
    assert d['markers_total'] + 1 == d['book_count'] == len(seen) + 1


def test_run_without_inventory_stitches_but_has_no_markers(tmp_path):
    src, _, _ = synthetic_run(tmp_path, with_inventory=False)
    d = json.loads(build(src, tmp_path / 'run_panorama_v1').read_text())
    assert d['book_count'] is None and d['markers_total'] == 0 and d['components'][0]['markers'] == []


def test_existing_output_is_refused_and_a_failed_build_leaves_no_directory(tmp_path):
    src, _, _ = synthetic_run(tmp_path)
    taken = tmp_path / 'run_panorama_v1'
    taken.mkdir()
    (taken / 'keep.txt').write_text('x')
    with pytest.raises(ValueError, match='Refusing existing'):
        build(src, taken)
    assert (taken / 'keep.txt').read_text() == 'x'
    (src / 'frames' / 'f00003.jpg').write_bytes(b'not an image')
    out = tmp_path / 'run_panorama_v2'
    d = json.loads(build(src, out).read_text())                          # an unreadable frame is listed, not fatal
    assert any(u['frame_id'] == 'f00003' for u in d['unregistered_frames'])
    for p in (src / 'frames').glob('*.jpg'):
        p.unlink()
    with pytest.raises(RuntimeError, match='panorama failed while'):
        build(src, tmp_path / 'run_panorama_v3')
    assert not (tmp_path / 'run_panorama_v3').exists()


def test_too_few_frames_is_a_clear_error(tmp_path):
    src, _, _ = synthetic_run(tmp_path, frames=1)
    with pytest.raises(ValueError, match='fewer than 2'):
        build(src, tmp_path / 'run_panorama_v1')
    assert not (tmp_path / 'run_panorama_v1').exists()


def test_source_hashes_are_recorded_and_audit_detects_tampering(tmp_path):
    src, _, _ = synthetic_run(tmp_path)
    path = build(src, tmp_path / 'run_panorama_v1')
    d = json.loads(path.read_text())
    assert d['source_manifest_sha256'] == sha(src / 'manifest.json') and d['source_inventory_sha256'] == sha(src / 'inventory.json')
    assert d['counts']['book_count'] == d['book_count'] and d['counts']['verified_physical_book_count'] is None
    (tmp_path / 'run_panorama_v1' / d['components'][0]['image']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        audit(path)


# ------------------------------------------------------------------------------------------ global alignment
def synthetic_matches(n=8, per_pair=60, seed=0):
    """Frames panning over a plane with true frame->plane homographies; pair correspondences are exact (+0.3 px noise)."""
    rng = np.random.default_rng(seed)
    true = []
    for i in range(n):
        H = np.array([[1 + 0.02 * i, 0.01 * i, 120. * i], [-0.005 * i, 1 + 0.015 * i, 40. * i], [1e-5 * i, 2e-5 * i, 1.]])
        true.append(H / H[2, 2])
    pairs = []
    for i in range(n):
        for gap in (1, 2, 3):
            j = i + gap
            if j >= n:
                continue
            plane = np.c_[rng.uniform(200, 900, per_pair) + 120 * i, rng.uniform(100, 700, per_pair) + 40 * i]
            xa = project(plane, np.linalg.inv(true[i]))
            xb = project(plane, np.linalg.inv(true[j])) + rng.normal(0, 0.3, (per_pair, 2))
            ok = (xa[:, 0] > 0) & (xa[:, 0] < 1920) & (xa[:, 1] > 0) & (xa[:, 1] < 1080) & (xb[:, 0] > 0) & (xb[:, 0] < 1920) & (xb[:, 1] > 0) & (xb[:, 1] < 1080)
            if ok.sum() >= 12:
                pairs.append((i, j, xa[ok], xb[ok], int(ok.sum())))
    return true, pairs


def test_refinement_removes_accumulated_scale_drift_of_a_chain():
    true, pairs = synthetic_matches()
    order = [f'f{i}' for i in range(len(true))]
    drift = {f'f{i}': np.diag([1.03 ** i, 1.03 ** i, 1.]) @ true[i] for i in range(len(true))}      # a chain with biased scale steps
    drift = {k: v / v[2, 2] for k, v in drift.items()}
    refined, report = P.refine_homographies(order, drift, pairs)
    assert report['applied'] and report['rms_px_after'] < 0.3 * report['rms_px_before'] and report['rms_px_after'] < 1.0
    for i in range(1, len(true) - 1):                                      # relative geometry between frames matches the truth
        want = np.linalg.inv(true[i + 1]) @ true[i]
        got = np.linalg.inv(refined[f'f{i + 1}']) @ refined[f'f{i}']
        assert np.abs(project(corners(frame()), want) - project(corners(frame()), got)).max() < 3.0


def test_refinement_keeps_the_chain_when_nothing_constrains_a_frame():
    true, pairs = synthetic_matches()
    order = [f'f{i}' for i in range(len(true))]
    chain = {f'f{i}': true[i] for i in range(len(true))}
    no_last = [p for p in pairs if p[1] != len(true) - 1 and p[0] != len(true) - 1]
    out, report = P.refine_homographies(order, chain, no_last)
    assert not report['applied'] and 'without any verified match' in report['reason'] and out is chain
    assert P.refine_homographies(order, chain, [])[1]['reason'] == 'no verified match pairs'


def test_collect_matches_is_gated_by_the_chain_prediction_and_returns_full_resolution_points(tmp_path):
    src, _, _ = synthetic_run(tmp_path)
    frames = json.loads((src / 'manifest.json').read_text())['data']['frames']
    order = [f['frame_id'] for f in frames]
    T = {f: np.array([[1, 0, STEP * k], [0, 1, 0], [0, 0, 1.]]) for k, f in enumerate(order)}        # the exact pan
    image = lambda fid: cv2.imread(str(src / 'frames' / f'{fid}.jpg'))
    pairs = P.collect_matches(order, T, image, gaps=(1, 2))
    assert len(pairs) >= len(order) - 2
    i, j, xa, xb, n = pairs[0]
    assert np.abs((xa - xb) - [STEP * (j - i), 0]).max() < 2.5 and xa.max() > 100          # full-resolution pixels
    wrong = {f: np.array([[1, 0, 400. * k], [0, 1, 300.], [0, 0, 1.]]) for k, f in enumerate(order)}   # a chain that predicts the wrong place: no pairs
    assert P.collect_matches(order, wrong, image, gaps=(1,)) == []


def test_build_reports_the_alignment_and_can_skip_it(tmp_path):
    src, _, _ = synthetic_run(tmp_path)
    d = json.loads(build(src, tmp_path / 'run_panorama_v1').read_text())
    a = d['components'][0]['alignment']
    assert a['pairs'] > 5 and 'rms_px_before' in a
    d2 = json.loads(build(src, tmp_path / 'run_panorama_v2', align=False).read_text())
    assert d2['components'][0]['alignment'] == {'applied': False, 'reason': 'disabled'}
    assert d2['markers_total'] == d['markers_total'] == d['book_count']


def test_yaw_homography_makes_converging_shelf_boards_parallel_and_level():
    focal, size = 1500., (1920, 1080)
    K = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1.]])
    yaw = math.radians(22)
    R = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])      # camera turned 22 deg about the vertical axis
    lines, mids = [], []
    for y in np.linspace(-0.5, 0.5, 9):                        # horizontal boards on a plane 2 m away
        pts = np.array([[x, y, 2.0] for x in (-0.8, 0.8)])
        img = (K @ (pts @ R.T).T).T
        q = img[:, :2] / img[:, 2:]
        l = np.cross(np.append(q[0], 1), np.append(q[1], 1))
        lines.append(l / math.hypot(l[0], l[1]))
        mids.append(q.mean(0))
    lines, mids = np.array(lines), np.array(mids)
    vp, inl = vanishing_point(lines, mids, tol_deg=0.3)
    assert inl.all()
    H, found = P.yaw_homography(vp, *size, focal)
    assert H is not None and abs(abs(found) - 22) < 1.0
    for l, m in zip(lines, mids):
        d = np.array([-l[1], l[0]])
        a = project([m, m + d * 50], H)
        v = a[1] - a[0]
        assert abs(v[1]) / np.linalg.norm(v) < 0.02            # level after the yaw correction
    assert P.yaw_homography(np.array([1., 0, 0]), *size, focal)[0] is None          # already at infinity along x: nothing to do


def test_horizontal_lines_are_returned_in_plane_coordinates():
    img = np.zeros((400, 600), np.uint8)
    cv2.line(img, (20, 100), (580, 140), 255, 3)
    cv2.line(img, (20, 300), (580, 300), 255, 3)
    cv2.line(img, (300, 20), (300, 380), 255, 3)               # vertical: not a board
    to_plane = np.diag([2., 2., 1.])
    lines, mids, lens = P.horizontal_lines(img, to_plane)
    assert len(lines) >= 2 and lens.min() > 200                       # thick strokes give a pair of edges each; the vertical stroke is not a board
    assert min(abs(mids[:, 1] - 240)) < 60 and min(abs(mids[:, 1] - 600)) < 60
    angles = (np.degrees(np.arctan2(lines[:, 0], -lines[:, 1])) + 90) % 180 - 90
    assert np.all(np.abs(angles) < 35)                                  # every returned line is near-horizontal


# ------------------------------------------------------------------------------------------ progress for the build job UI
def test_progress_fraction_is_monotone_labelled_and_ends_at_one(tmp_path, monkeypatch):
    seen = []
    real = P.Progress.__call__

    def spy(self, stage, **extra):
        real(self, stage, **extra)
        data = json.loads(self.path.read_text())
        seen.append((data['stage'], data['fraction'], data['label']))
    monkeypatch.setattr(P.Progress, '__call__', spy)
    src, _, _ = synthetic_run(tmp_path)
    build(src, tmp_path / 'run_panorama_v1')
    fractions = [f for _, f, _ in seen]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    stages = [st for st, _, _ in seen]
    for expected in ('tracking', 'aligning: features', 'aligning: matching', 'rectifying', 'compositing', 'compositing seams', 'compositing blend', 'markers', 'writing', 'done'):
        assert expected in stages, expected
    assert any('aligning frames: matching' in label and '/' in label for _, _, label in seen)       # done/total in the label
    assert 0.0 < max(f for st, f, _ in seen if st == 'aligning: matching') < 0.9


def test_progress_weights_components_by_frame_count():
    p = P.Progress.__new__(P.Progress)
    p.sizes, p.component = None, 0
    p.set_components([30, 10])
    assert abs(sum(p.sizes) - 1) < 1e-9 and p.sizes[0] == 0.75
    first = p.fraction('compositing blend', {'component': 1, 'done': 15, 'total': 30})
    second_start = p.fraction('aligning', {'component': 2})
    assert first < second_start < p.fraction('markers', {})                       # the small second section only adds its share
    assert p.fraction('done', {}) == 1.0 and p.fraction('tracking', {}) == 0.0
