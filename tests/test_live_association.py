"""Live-stream association (tracking recovery, segment bridge, appearance-guided residual)."""
import cv2
import numpy as np
from unittest.mock import patch

from pipeline import cross_view as cv


def _texture():
    rng = np.random.default_rng(3)
    big = cv2.GaussianBlur(rng.integers(0, 255, (900, 1600), dtype=np.uint8), (0, 0), 3)
    return cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)


def _crop(big, dx, dy=0):
    return cv2.cvtColor(big[100 + dy:700 + dy, 100 + dx:900 + dx], cv2.COLOR_GRAY2BGR)


def test_large_step_is_recovered_from_a_phase_correlation_seed_and_reported():
    big = _texture()
    ims = {'a': _crop(big, 0), 'b': _crop(big, 150)}
    _, plain = cv.chain_transforms(['a', 'b'], ims.get)
    assert plain['b'].startswith('hold'), 'precondition: plain LK from a zero start loses a 150 px step on this texture'
    rep = {}
    T, status = cv.chain_transforms(['a', 'b'], ims.get, recover=True, report=rep)
    assert status['b'] == 'ok'
    assert np.allclose(T['b'][:2, 2], [150, 0], atol=2), 'the seeded step must be the true motion, accepted by the normal LK+RANSAC test'
    steps = rep['recovery']['recovered_steps']
    assert [s['frame_id'] for s in steps] == ['b'] and steps[0]['inliers'] >= 25


def test_recovery_is_off_by_default_and_leaves_tracked_steps_unchanged():
    big = _texture()
    ims = {'a': _crop(big, 0), 'b': _crop(big, 12)}
    T0, s0 = cv.chain_transforms(['a', 'b'], ims.get)
    rep = {}
    T1, s1 = cv.chain_transforms(['a', 'b'], ims.get, recover=True, report=rep)
    assert s0 == s1 == {'a': 'start', 'b': 'ok'} and np.allclose(T0['b'], T1['b'])
    assert rep['recovery']['recovered_steps'] == [] and rep['recovery']['bridges'] == []


def test_segment_break_is_rejoined_when_a_later_frame_registers_to_the_old_anchor():
    big = _texture()
    ims = {'a': _crop(big, 0), 'c': _crop(big, 40), 'd': _crop(big, 60)}
    ids = ['a', 'm1', 'm2', 'm3', 'c', 'd']
    load = ims.get
    _, s_plain = cv.chain_transforms(ids, load)
    assert s_plain['c'] == 'segment_start'
    rep = {}
    T, status = cv.chain_transforms(ids, load, recover=True, report=rep)
    seg = cv.segments_from_status(ids, status)
    assert status['c'] == 'ok' and seg['c'] == seg['d'] == seg['a'], 'a verified bridge removes the break'
    assert np.allclose(T['d'][:2, 2], [60, 0], atol=2) and np.allclose(T['c'][:2, 2], [40, 0], atol=2)
    assert rep['recovery']['bridges'] and rep['recovery']['bridges'][0]['frame_id'] == 'd' and rep['recovery']['bridges'][0]['agreement'] > 0.9


def test_unrelated_view_after_a_break_stays_a_separate_segment():
    big = _texture()
    other = cv2.GaussianBlur(np.random.default_rng(99).integers(0, 255, (900, 1600), dtype=np.uint8), (0, 0), 3)
    other = cv2.normalize(other, None, 0, 255, cv2.NORM_MINMAX)
    ims = {'a': _crop(big, 0), 'c': _crop(other, 0), 'd': _crop(other, 20)}
    ids = ['a', 'm1', 'm2', 'm3', 'c', 'd']
    _, status = cv.chain_transforms(ids, ims.get, recover=True, bridge_frames=8)
    seg = cv.segments_from_status(ids, status)
    assert status['c'] == 'segment_start' and seg['d'] != seg['a'], 'no shared content: the break must stay'


def _pinfo(centres, w=10.0, L=150.0):
    out = {}
    for k, c in centres.items():
        c = np.asarray(c, float)
        P = np.array([[c[0] - w / 2, c[1] - L / 2], [c[0] + w / 2, c[1] - L / 2], [c[0] + w / 2, c[1] + L / 2], [c[0] - w / 2, c[1] + L / 2]])
        out[k] = {"c": c, "w": w, "L": L, "v": np.array([0.0, 1.0]), "P": P}
    return out


def test_guided_residual_recovers_a_zoom_the_tracked_motion_missed():
    # two rows of 6 upright spines 12 px apart; frame b sees them 15% smaller around the origin (the chain reported no zoom), so most true partners sit
    # outside the 0.5-width gate. Appearance identifies the true partners; the residual affine must restore them.
    xs = np.r_[np.arange(6) * 12.0 + 200, np.arange(6) * 12.0 + 200]
    ys = np.r_[np.full(6, 300.0), np.full(6, 400.0)]  # two shelf rows
    A = [f'a{i}' for i in range(12)]
    B = [f'b{i}' for i in range(12)]
    pinfo = {**_pinfo({a: (x, y) for a, x, y in zip(A, xs, ys)}), **_pinfo({b: (0.85 * x, 0.85 * y) for b, x, y in zip(B, xs, ys)})}
    same = lambda p, q: 1.0 if p == q else 0.0
    with patch.object(cv, 'appearance', side_effect=same), patch.object(cv, '_patch', side_effect=lambda *a: None):
        res = cv._guided_residual(A, B, pinfo, np.zeros(2), lambda k: k[1:], {**cv.DEFAULTS})
    assert res is not None
    R, inliers, before, after = res
    assert after == 12 and before < after and inliers >= 6
    assert abs(R[0, 0] - 0.85) < 0.02 and abs(R[1, 1] - 0.85) < 0.02


def test_guided_residual_keeps_the_translation_when_it_already_matches():
    xs = np.arange(12) * 12.0 + 200
    A = [f'a{i}' for i in range(12)]
    B = [f'b{i}' for i in range(12)]
    pinfo = {**_pinfo({a: (x, 300) for a, x in zip(A, xs)}), **_pinfo({b: (x + 1, 300) for b, x in zip(B, xs)})}
    with patch.object(cv, 'appearance', side_effect=lambda p, q: 1.0 if p == q else 0.0):
        assert cv._guided_residual(A, B, pinfo, np.zeros(2), lambda k: k[1:], {**cv.DEFAULTS}) is None


def test_guided_residual_needs_appearance_evidence():
    xs = np.arange(12) * 12.0 + 200
    A = [f'a{i}' for i in range(12)]
    B = [f'b{i}' for i in range(12)]
    pinfo = {**_pinfo({a: (x, 300) for a, x in zip(A, xs)}), **_pinfo({b: (0.85 * x, 300) for b, x in zip(B, xs)})}
    with patch.object(cv, 'appearance', return_value=0.0):
        assert cv._guided_residual(A, B, pinfo, np.zeros(2), lambda k: k, {**cv.DEFAULTS}) is None
