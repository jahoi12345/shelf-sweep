"""Physical-instance identity: one packet line per PHYSICAL book.

The live association (pipeline/dedup.py + pipeline/cross_view.py) follows each spine from frame to frame; every tracking break starts a new
segment and nothing pairs across it, so one book becomes several track groups and the count tiers promote each group alone. This module
merges track groups into *instances* with exactly two kinds of evidence, nothing else (no OCR/row merging, no appearance-only links, no
global loop closure):

  1. ``panorama_overlap``: panorama_count's ``merge_evidence`` (pipeline/panorama_count.py): several track groups fall on one spine detected on a
     stitched 2D piece, or on pieces registered to each other. Used only when the packet would also trust the panorama estimator
     (pipeline.panorama_attach.decide: count reliability low/medium, >= 75% of the frames registered, detector support). The reliability judged is the
     instance layer's own (the weaker of the packet's and this pass's association reliability), so the first pass already admits the panorama.
     Acceptance rule (a blinded model adjudication (model-read) judged 30-46% of these unions to join two different books): each anchor-other union needs
     (a) both groups placed on the spine with IoU >= 0.35 (panorama_count's own same-spine fusion IoU) and (b) where the two groups' closest sightings' frames register
     (cross-segment gates), A's spine projected into B's frame must cover B, and no other detected spine better (>= 0.5 across the spine axis). A union that
     fails is listed in ``rejected_unions`` with the reason; an accepted one carries ``acceptance`` and ``frame_check`` with the values.
     Registration decides (checked against the user's own same/different answers): the registered frame pair DECIDES when there
     is one ('agree' accepts even a placement below 0.35, 'contradicts' refuses); the placement IoU decides only when no frame pair registers or the
     registration is ambiguous (A's spine covers B and another detected spine equally).
  2. ``cross_segment_registration``: bounded SIFT/RANSAC registration between the sharpest detection frames on BOTH sides of EVERY
     tracking-segment transition (a few seconds around it; ~50 frames per run), then warped-spine overlap + appearance with the production
     gate (0.5), one-to-one with the near-tie veto. Only pairs that straddle a transition are tested: this is not the global loop closure the
     user declined (that scans all frame pairs).
  3. ``short_gap_registration``: the same registration and matcher on EVERY pair of detection
     frames at most 3 s apart (CROSS['window_s']), within or across segments: one book split inside a segment by a fast close-range pan. Frames
     further apart are never compared (still no loop closure). ``revisit_registration`` (revisit linking) extends the
     same matcher to frames 3-20 s apart (a pass that comes back over a shelf section) with a layout check (>= half of the spines a registration aligns must
     look alike); env LIBRARY_REVISIT_MERGE=0 / --revisit 0 turns it off. Later evidence kinds (shelf_plane_position, the opt-in oracle) follow their own sections.

Same-frame exclusion is kept at every union: two groups seen in one frame are two books whatever the evidence says. Every merge carries its
evidence line. Instances are the unit for the count tiers (sightings = union), the packet lines, prices, measurement and titles.

Artifacts: ``instances.json`` in the run directory (and a copy beside the panorama count). ``apply`` patches a packet's ``data`` in place
and is idempotent (a merged line stores its member lines, so a re-apply first expands them); ``attach`` does that for a saved
claim_packet.json with verification, like pipeline.auto_attach / pipeline.panorama_attach. ``run_step`` is the post-sweep step
(app/post_sweep.py), run after the panorama count and again after the measurement, in a niced subprocess.

    python -m pipeline.instances --run runs/<run> [--panorama-count file] [--out file] [--attach] [--no-cross-segment]
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCHEMA = 1
ARTIFACT = 'instances.json'
BACKUP = 'claim_packet.pre_instances.json'
KIND_PANORAMA = 'panorama_overlap'
KIND_CROSS = 'cross_segment_registration'
KIND_ORACLE = 'hosted_pair_adjudication'   # pipeline/pair_oracle.py: MODEL-READ, opt-in (LIBRARY_PAIR_ORACLE=1 or --pair-oracle), never on by default
ORACLE_LABEL = 'merged by hosted pair adjudication (model-read)'
FAR_FRACTION = 0.5         # pipeline.panorama_count.FAR_FRACTION: a sighting whose spine (polygon long side) is < this share of the run's median native spine is a far view
# Cover-view rule: a sighting whose polygon is under this long/short side ratio (native pixels) is a cover or flat view, not a spine view (the class
# pipeline.auto_scale reports as 'polygon_aspect_<x>_not_a_spine_view(cover_or_flat_face)'). Real spines seen whole are 3:1 and more (YugoTest11: p5 of all
# book sightings 3.6). An instance with NO spine-view sighting is a cover face, a second box on a book already counted from its spine, or a decor object.
COVER_VIEW_MAX_ASPECT = 2.0
ROOM_PAN_REASON = 'room-pan view only'   # = pipeline.room_pan.REASON: the reason string of the room-pan rule
READING_VETO = {'min_chars': 8, 'min_overlap': 0.3}   # precision guard: two members with readable spine texts that disagree are never one instance (titles still never CREATE a link)
CROSS = {
    'window_s': 3.0,          # frames this close to a transition, on either side, are candidates
    'frames_per_side': 3,     # sharpest detection frames per side (the audit's probe used 3 + 3 per transition)
    'min_obs': 4,             # a candidate frame shows at least this many book sightings
    'max_width_px': 960,      # SIFT runs at min(0.5, max_width_px / frame width)
    'nfeatures': 3000,
    'ratio': 0.75,
    'ransac_px': 4.0,
    'min_inliers': 30,        # geometric inliers for the pair homography (the audit's break probe: 30)
    'min_coverage': 0.04,     # inlier hull area over the frame area (one texture patch is not a registration)
    'min_scale': 0.4, 'max_scale': 2.5,    # sqrt(det) of the homography: zoom between the two views
    'tol_widths': 0.5,        # match radius after the homography, in spine widths (the production gate, cross_view.DEFAULTS)
    'min_appearance': 0.5,    # warped-spine Lab correlation (the production gate)
    'geometry_weight': 0.1,
    'margin': 0.07,           # near-tie veto: a competing candidate for either spine within this score is ambiguous (dedup.DEFAULTS ambiguity_margin)
}
# Acceptance rule for panorama_overlap unions (precision). Both constants are existing production gates, not fitted:
PANORAMA_ACCEPT = {
    'min_spine_iou': 0.35,        # = pipeline.panorama_count.MERGE['iou_same_piece']: EACH group's placement must overlap the detected spine as much as two detector boxes
                                  #   must overlap to be fused into one spine (panorama_count places a group on a spine from IoU 0.05 or its centre inside the box)
    'min_across_overlap': 0.5,    # = CROSS['tol_widths']: in a direct frame registration, A's spine must cover >= half of the narrower spine across the spine axis of B
}
PANORAMA_FRAME_PAIRS = 3          # = CROSS['frames_per_side']: closest-in-time sighting pairs tried for that registration (the first that registers decides)
# Which evidence decides (same two constants, nothing fitted). 'registration_decides' (default): a registered frame pair decides
# (agree -> union, contradicts -> refused), the placement IoU only where none registered or the registration is ambiguous;
# 'placement_and_frame': placement AND frame check (comparison only).
PANORAMA_RULES = ('registration_decides', 'placement_and_frame')
# Reconciliation with the panorama count. An unconfirmed single sighting (not far-view-only, not cover-view-only, not a
# demoted non-book) becomes a line when its sighting, projected into a normal (detector-counted) panorama piece, lies on a detected spine (the acceptance rule's placement IoU 0.35)
# that no sighting of a counted line occupies (panorama_count's placement: each sighting on its best-overlapping spine, IoU >= 0.05 or centre inside). One line per spine.
# Used only when the panorama is eligible (pipeline.panorama_attach.decide) AND its count lies inside the instance layer's own [lower, upper]; at most (panorama count -
# lines) promotions, strongest placement first: the reconciliation moves the lines toward the panorama count and never past it. No new constant.
RECONCILE = {'min_spine_iou': PANORAMA_ACCEPT['min_spine_iou'], 'occupy_min_iou': 0.05}   # 0.05 = pipeline.panorama_count's placement floor (centre inside counts as 0.05)
PROMOTED_TIER = 'likely'
NOTE = ('{lines} lines merged from {groups} track groups by panorama overlap ({pan}) and cross-segment registration ({cross}); '
        'duplicates across unregistered pieces and unbridged transitions remain possible; no per-book truth exists')
_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}


# ------------------------------------------------------------------------------------------ segments of the saved run
def segment_map(frames: dict[str, dict[str, Any]], track_report: dict[str, Any] | None) -> dict[str, int]:
    """{frame_id: tracking segment index} from the saved track report: ``segment_starts`` (written by dedup since the stable-id change) or, for older
    inventories, the hold frames that are not held frames (a third consecutive failed step starts a new segment; cross_view.chain_transforms)."""
    tr = track_report or {}
    order = sorted(frames, key=lambda f: (frames[f]['timestamp_s'], f))
    starts = tr.get('segment_starts')
    if starts is None:
        starts = sorted(set(tr.get('hold_frames') or []) - set(tr.get('held_frames') or []))
    starts = set(starts)
    out, seg = {}, 0
    for f in order:
        if f in starts:
            seg += 1
        out[f] = seg
    return out


def _quad(o: dict[str, Any]) -> list[list[float]] | None:
    if len(o.get('polygon_xy') or []) == 4:
        return o['polygon_xy']
    if o.get('bbox_xyxy'):
        x1, y1, x2, y2 = o['bbox_xyxy']
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return None


def _width(poly: Any) -> float:
    p = np.asarray(poly, np.float64)
    e = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
    return float(min(e[0], e[1]))


def register_frames(ga: np.ndarray, fa_feat: tuple, gb: np.ndarray, fb_feat: tuple, scale: float, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Homography A -> B (full-resolution pixels) from SIFT features of the two downscaled grey frames, or None with the gates of ``cfg``."""
    ka, da = fa_feat
    kb, db = fb_feat
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x for x, y in (z for z in m if len(z) == 2) if x.distance < cfg['ratio'] * y.distance]
    if len(good) < cfg['min_inliers']:
        return None
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, mk = cv2.findHomography(pa, pb, cv2.RANSAC, cfg['ransac_px'])
    if H is None or mk is None:
        return None
    inl = mk.ravel().astype(bool)
    n = int(inl.sum())
    if n < cfg['min_inliers']:
        return None
    h, w = ga.shape[:2]
    cov = min(cv2.contourArea(cv2.convexHull(pa[inl])) / (h * w), cv2.contourArea(cv2.convexHull(pb[inl])) / (gb.shape[0] * gb.shape[1]))
    if cov < cfg['min_coverage']:
        return None
    S, Si = np.diag([scale, scale, 1.0]), np.diag([1 / scale, 1 / scale, 1.0])
    H = Si @ H @ S
    sdet = math.sqrt(abs(np.linalg.det(H[:2, :2])))
    if not cfg['min_scale'] < sdet < cfg['max_scale'] or not np.isfinite(H).all():
        return None
    return {'H': H, 'inliers': n, 'matches': len(good), 'coverage': round(float(cov), 3), 'scale': round(sdet, 3)}


class _FrameStore:
    """Lazy per-run cache of frame images, SIFT features (downscaled grey) and spine patches, shared by the registration evidences."""

    def __init__(self, root: Path, frames: dict[str, dict[str, Any]], obs: dict[str, dict[str, Any]], scale: float, cfg: dict[str, Any]):
        self.root, self.frames, self.obs, self.scale = root, frames, obs, scale
        self.images: dict[str, np.ndarray | None] = {}
        self.feats: dict[str, tuple] = {}
        self.greys: dict[str, np.ndarray | None] = {}
        self.patches: dict[str, np.ndarray] = {}
        self.sift = cv2.SIFT_create(nfeatures=cfg['nfeatures'])
        cv2.setRNGSeed(0)

    def image(self, f):
        from pipeline.detect import resolve_evidence
        if f not in self.images:
            im = cv2.imread(str(resolve_evidence(self.root, self.frames[f]['image_path'])))
            if im is None or im.shape[1::-1] != (self.frames[f]['width_px'], self.frames[f]['height_px']):
                im = None
            self.images[f] = im
        return self.images[f]

    def feat(self, f):
        if f not in self.feats:
            im = self.image(f)
            if im is None:
                self.feats[f], self.greys[f] = (None, None), None
            else:
                g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
                self.greys[f] = g
                self.feats[f] = self.sift.detectAndCompute(g, None)
        return self.feats[f]

    def patch(self, oid):
        from pipeline import cross_view as CV
        if oid not in self.patches:
            self.patches[oid] = CV._patch(self.image(self.obs[oid]['frame_id']), _quad(self.obs[oid]))
        return self.patches[oid]


def _registered_matches(reg: dict[str, Any], A: list[str], B: list[str], obs: dict[str, dict[str, Any]], store: _FrameStore, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The production spine matcher on a registered frame pair A -> B: centres within tol_widths spine widths after the homography, appearance >= min_appearance,
    one-to-one by score; a match whose spine has a competing candidate within ``margin`` is returned with ``runner_up`` set (near-tie veto, the caller refuses it)."""
    from pipeline import cross_view as CV
    H, sdet = reg['H'], reg['scale']
    ca = CV._proj([np.mean(_quad(obs[k]), 0) for k in A], H)
    cb = np.array([np.mean(_quad(obs[k]), 0) for k in B])
    wa = np.array([_width(_quad(obs[k])) for k in A]) * sdet
    wb = np.array([_width(_quad(obs[k])) for k in B])
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
    tol = cfg['tol_widths'] * np.maximum(wa[:, None], wb[None, :])
    cands = []
    for x, y in zip(*np.nonzero(d <= tol)):
        a = CV.appearance(store.patch(A[x]), store.patch(B[y]))
        if a >= cfg['min_appearance']:
            cands.append((float(a - cfg['geometry_weight'] * d[x, y] / tol[x, y]), int(x), int(y), float(a), float(d[x, y] / tol[x, y])))
    cands.sort(reverse=True)
    used_a, used_b, out = set(), set(), []
    for score, x, y, a, nd in cands:
        if x in used_a or y in used_b:
            continue
        used_a.add(x)
        used_b.add(y)
        competing = [s for s, x2, y2, _, _ in cands if (x2 == x) != (y2 == y)]      # another candidate for exactly one of the two spines
        runner = max(competing) if competing and score - max(competing) < cfg['margin'] else None
        out.append({'a': A[x], 'b': B[y], 'appearance': a, 'nd': nd, 'score': score, 'runner_up': runner})
    return out


def cross_segment_links(run_dir: str | Path, frames: dict[str, dict[str, Any]], observations: list[dict[str, Any]], groups: list[dict[str, Any]],
                        segments: dict[str, int], config: dict[str, Any] | None = None, progress=None) -> dict[str, Any]:
    """Candidate same-book links between track groups across every tracking-segment transition (see the module docstring).
    Returns {'links': [accepted], 'rejected': [near-tie vetoes], 'transitions': [...], 'frames_used', 'pairs_tested', 'pairs_registered', 'seconds'}.
    Links are group-level ({'groups': [ga, gb], 'observation_ids', 'frame_ids', ...}); the same-frame exclusion is applied when instances are built."""
    cfg = {**CROSS, **(config or {})}
    t0 = time.perf_counter()
    root = Path(run_dir)
    group_of = {o: g['instance_id'] for g in groups for o in g['observation_ids']}
    obs = {o['observation_id']: o for o in observations if o['kind'] == 'book' and o['observation_id'] in group_of and _quad(o) is not None}
    by_frame: dict[str, list[str]] = collections.defaultdict(list)
    for oid, o in obs.items():
        by_frame[o['frame_id']].append(oid)
    order = sorted(frames, key=lambda f: (frames[f]['timestamp_s'], f))
    starts = [f for i, f in enumerate(order) if i and segments.get(f, 0) != segments.get(order[i - 1], 0)]
    sharp = lambda f: float((frames[f].get('quality') or {}).get('sharpness') or 0.0)
    det = [f for f in order if len(by_frame.get(f, [])) >= cfg['min_obs']]
    transitions, wanted = [], set()
    for fs in starts:
        t, seg = frames[fs]['timestamp_s'], segments[fs]
        before = [f for f in det if segments[f] == seg - 1 and t - cfg['window_s'] <= frames[f]['timestamp_s'] < t]
        after = [f for f in det if segments[f] == seg and t <= frames[f]['timestamp_s'] <= t + cfg['window_s']]
        before = sorted(sorted(before, key=sharp, reverse=True)[:cfg['frames_per_side']])
        after = sorted(sorted(after, key=sharp, reverse=True)[:cfg['frames_per_side']])
        transitions.append({'at': fs, 'timestamp_s': t, 'segment': seg, 'frames_before': before, 'frames_after': after, 'pairs_tested': 0, 'pairs_registered': 0, 'links': 0, 'vetoed': 0})
        wanted.update(before + after)
    scale = 1.0
    if wanted:
        scale = min(0.5, cfg['max_width_px'] / max(frames[f]['width_px'] for f in wanted))
    store = _FrameStore(root, frames, obs, scale, cfg)
    links, rejected = [], []
    tested = registered = 0
    for ti, tr in enumerate(transitions):
        if progress:
            progress(ti, len(transitions))
        for fa in tr['frames_before']:
            for fb in tr['frames_after']:
                if store.feat(fa)[1] is None or store.feat(fb)[1] is None:
                    continue
                tested += 1
                tr['pairs_tested'] += 1
                reg = register_frames(store.greys[fa], store.feat(fa), store.greys[fb], store.feat(fb), scale, cfg)
                if reg is None:
                    continue
                registered += 1
                tr['pairs_registered'] += 1
                for m in _registered_matches(reg, by_frame[fa], by_frame[fb], obs, store, cfg):
                    ga, gb = group_of[m['a']], group_of[m['b']]
                    row = {'groups': [ga, gb], 'observation_ids': [m['a'], m['b']], 'frame_ids': [fa, fb], 'segments': [segments[fa], segments[fb]],
                           'gap_s': round(frames[fb]['timestamp_s'] - frames[fa]['timestamp_s'], 3), 'inliers': reg['inliers'], 'registration_coverage': reg['coverage'],
                           'view_scale': reg['scale'], 'appearance_score': round(m['appearance'], 4), 'center_distance_spine_widths': round(m['nd'] * cfg['tol_widths'], 4),
                           'score': round(m['score'], 4), 'kind': KIND_CROSS, 'transition': tr['at']}
                    if m['runner_up'] is not None:
                        tr['vetoed'] += 1
                        rejected.append({**row, 'reason': 'near-tie competing spine (adjacent/identical copies stay separate)', 'runner_up_score': round(m['runner_up'], 4)})
                        continue
                    if ga == gb:
                        continue                                                                   # already one group: no new information
                    tr['links'] += 1
                    links.append(row)
    if progress:
        progress(len(transitions), len(transitions))
    return {'links': links, 'rejected': rejected, 'transitions': transitions, 'frames_used': len(wanted), 'pairs_tested': tested, 'pairs_registered': registered,
            'transitions_total': len(transitions), 'transitions_registered': sum(1 for t in transitions if t['pairs_registered']),
            'transitions_with_links': sum(1 for t in transitions if t['links']), 'scale': scale, 'config': {k: cfg[k] for k in CROSS},
            'seconds': round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------------------------------ short_gap_registration
# On IESFULLTest1 most duplicates are ONE book split into two track groups 0.5-1.5 s apart INSIDE one tracking
# segment (fast close-range pans: the chained LK motion is 1-1.7 spine widths off, against the 0.5 gate), so the cross-segment probe never looks at them,
# although a direct SIFT/RANSAC registration of the two frames puts the spine on its partner. This evidence registers EVERY pair of detection frames at most
# CROSS['window_s'] apart (within or across segments). Spines are matched by the panorama acceptance frame check (the registered frame pair decides; checked against the
# user's same/different answers): A's warped spine must cover B by >= 0.5 across the spine axis (across_overlap, robust to a box that drifted or is truncated,
# where the centre distance is not), B's back-warped spine must cover A the same way, each must be the other's best with no competitor within the near-tie
# margin, and the cross_view appearance must reach the production 0.5. It is the transition probe without the transition, not loop closure: frames further
# apart than window_s are never compared. Constants: existing production gates, none new, none fitted to a hand count.
KIND_SHORT_GAP = 'short_gap_registration'
SHORT_GAP = {'max_gap_s': CROSS['window_s'],                                  # 3.0 s: the cross-segment probe's own window
             'min_across_overlap': PANORAMA_ACCEPT['min_across_overlap']}    # 0.5: the panorama acceptance frame-check overlap, both directions
# every other gate is CROSS (30 inliers, 4% coverage, zoom 0.4-2.5, appearance 0.5, near-tie margin 0.07); same-frame exclusion and the reading veto apply at the union
SHORT_GAP_ENV = 'LIBRARY_SHORT_GAP_MERGE'
SHORT_GAP_DEFAULT = '1'
# The whole-box appearance of a short-gap pair is low mostly because the two boxes cover DIFFERENT stretches of
# the spine (one truncated by the frame edge or detected shorter, a 1.5-3x zoom change), not because of blur (blur-matching the sharper crop moved the score by
# ~0.01). Common-stretch appearance: the same Lab correlation and the same 0.5 gate, measured only over the stretch ALONG the spine that both boxes cover once A
# is projected into B by the registration. Used only as a second chance for a pair under the gate (pair OR stretch). OFF by default: it removes correct-looking
# duplicates on YugoTest12 too (435 -> 427), which the "YugoTest12 must not drop" rule forbids.
SHORT_GAP_STRETCH_ENV = 'LIBRARY_SHORT_GAP_STRETCH'
SHORT_GAP_STRETCH_DEFAULT = '0'
STRETCH_MIN_FRAC = 0.3   # the common stretch must cover >= 30% of the shorter box along the spine (below that the comparison falls back to the whole-box score)


def _upright(quad: Any) -> np.ndarray:
    """Quad reordered as cross_view._patch does: [top-left, top-right, bottom-right, bottom-left] of an upright spine (short edge first)."""
    p = np.asarray(quad, np.float32)
    e = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
    src = np.array([p[1], p[2], p[3], p[0]], np.float32) if e[0] >= e[1] else p
    if np.linalg.norm(src[0] - src[1]) > np.linalg.norm(src[1] - src[2]):
        src = np.array([src[1], src[2], src[3], src[0]], np.float32)
    return src


def stretch_appearance(image_a: np.ndarray, quad_a: Any, image_b: np.ndarray, quad_b: Any, H: np.ndarray, min_frac: float = STRETCH_MIN_FRAC) -> float | None:
    """cross_view.appearance (24x96 Lab correlation, both orientations) of two spine boxes over the stretch along B's spine axis that both cover, A projected
    into B's pixels by the registration H (A -> B). Both sub-boxes are rectified to one grid (the smaller native size per axis) before the 24x96 patch.
    None when the boxes share less than ``min_frac`` of the shorter one along the spine."""
    from pipeline import cross_view as CV
    sa, sb = _upright(quad_a), _upright(quad_b)
    u = (sb[3] + sb[2]) / 2 - (sb[0] + sb[1]) / 2
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        return None
    u = u / n
    SA = CV._proj(sa, H)
    pa, pb = SA @ u, sb @ u
    lo, hi = max(pa.min(), pb.min()), min(pa.max(), pb.max())
    if hi - lo < min_frac * min(np.ptp(pa), np.ptp(pb)):
        return None

    def sub(src, S):
        st, en = (S[0] + S[1]) / 2 @ u, (S[3] + S[2]) / 2 @ u
        if abs(en - st) < 1e-6:
            return None
        f0, f1 = sorted([(lo - st) / (en - st), (hi - st) / (en - st)])
        f0, f1 = max(f0, 0.0), min(f1, 1.0)
        if f1 - f0 < 0.1:
            return None
        return np.array([src[0] + f0 * (src[3] - src[0]), src[1] + f0 * (src[2] - src[1]), src[1] + f1 * (src[2] - src[1]), src[0] + f1 * (src[3] - src[0])], np.float32)
    qa, qb = sub(sa, SA), sub(sb, sb)
    if qa is None or qb is None:
        return None
    w = max(4, int(round(min(np.linalg.norm(qa[0] - qa[1]), np.linalg.norm(qb[0] - qb[1])))))
    h = max(8, int(round(min(np.linalg.norm(qa[1] - qa[2]), np.linalg.norm(qb[1] - qb[2])))))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)

    def patch(img, q):
        r = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (w, h), flags=cv2.INTER_AREA)
        return cv2.cvtColor(cv2.resize(r, (24, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(CV.appearance(patch(image_a, qa), patch(image_b, qb)))


def short_gap_stretch_enabled(flag: bool | None = None) -> bool:
    return bool(flag) if flag is not None else os.environ.get(SHORT_GAP_STRETCH_ENV, SHORT_GAP_STRETCH_DEFAULT) == '1'


def _overlap_matches(reg: dict[str, Any], A: list[str], B: list[str], obs: dict[str, dict[str, Any]], store: _FrameStore, size_a: tuple, size_b: tuple,
                     cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Mutual-best spine matches of a registered frame pair by warped-spine across-overlap (both directions, >= min_across_overlap), with the near-tie veto (a
    second spine covered within ``margin`` of the best, in either direction: ``runner_up`` set) and the appearance gate (pairs under it are dropped)."""
    from pipeline import cross_view as CV
    Hi = np.linalg.inv(reg['H'])

    def best_of(src, dst, H, size):
        out = {}
        for s_ in src:
            q = _clip_to_frame(CV._proj(_quad(obs[s_]), H), *size)
            if q is None:
                continue
            ov = {d: across_overlap(q, np.asarray(_quad(obs[d]), np.float64)) for d in dst}
            ov = {d: v for d, v in ov.items() if v > 0}
            if not ov:
                continue
            ranked = sorted(ov, key=lambda d: (-ov[d], d))
            out[s_] = (ranked[0], ov[ranked[0]], ov[ranked[1]] if len(ranked) > 1 else None)
        return out
    ab, ba = best_of(A, B, reg['H'], size_b), best_of(B, A, Hi, size_a)
    res = []
    for a, (b, ov_ab, second_ab) in ab.items():
        if b not in ba or ba[b][0] != a:
            continue
        ov_ba, second_ba = ba[b][1], ba[b][2]
        if min(ov_ab, ov_ba) < cfg['min_across_overlap']:
            continue
        app = CV.appearance(store.patch(a), store.patch(b))
        app_stretch = None
        if app < cfg['min_appearance'] and cfg.get('stretch_appearance'):       # second chance: the same gate over the common stretch of the spine
            app_stretch = stretch_appearance(store.image(obs[a]['frame_id']), _quad(obs[a]), store.image(obs[b]['frame_id']), _quad(obs[b]), reg['H'])
        if app < cfg['min_appearance'] and (app_stretch is None or app_stretch < cfg['min_appearance']):
            continue
        runner = max([x for x, top in ((second_ab, ov_ab), (second_ba, ov_ba)) if x is not None and top - x < cfg['margin']], default=None)
        res.append({'a': a, 'b': b, 'appearance': app, 'appearance_stretch': app_stretch, 'across_overlap': [round(ov_ab, 3), round(ov_ba, 3)], 'runner_up': runner})
    return res


def short_gap_enabled(flag: bool | None = None) -> bool:
    return bool(flag) if flag is not None else os.environ.get(SHORT_GAP_ENV, SHORT_GAP_DEFAULT) == '1'


def short_gap_links(run_dir: str | Path, frames: dict[str, dict[str, Any]], observations: list[dict[str, Any]], groups: list[dict[str, Any]],
                    segments: dict[str, int], config: dict[str, Any] | None = None, progress=None) -> dict[str, Any]:
    """Same-book links between track groups seen in two detection frames at most ``max_gap_s`` apart. A frame pair is registered only when each frame shows a
    group the other does not (otherwise no link could pass the same-frame exclusion); a match whose two groups already share a frame is not reported (the
    union would refuse it). One evidence line per union (build). ``config['stretch_appearance']`` (default: LIBRARY_SHORT_GAP_STRETCH, off) turns on the common-stretch
    common-stretch second chance for pairs under the appearance gate."""
    cfg = {**CROSS, **SHORT_GAP, 'stretch_appearance': short_gap_stretch_enabled(), **(config or {})}
    t0 = time.perf_counter()
    root = Path(run_dir)
    group_of = {o: g['instance_id'] for g in groups for o in g['observation_ids']}
    frames_of = collections.defaultdict(set)
    for o in group_of:
        frames_of[group_of[o]].add(o.split('-o')[0])
    obs = {o['observation_id']: o for o in observations if o['kind'] == 'book' and o['observation_id'] in group_of and _quad(o) is not None}
    by_frame: dict[str, list[str]] = collections.defaultdict(list)
    for oid, o in obs.items():
        by_frame[o['frame_id']].append(oid)
    det = sorted((f for f in frames if len(by_frame.get(f, [])) >= cfg['min_obs']), key=lambda f: (frames[f]['timestamp_s'], f))
    gset = {f: {group_of[o] for o in by_frame[f]} for f in det}
    pairs = []
    for i, fa in enumerate(det):
        for fb in det[i + 1:]:
            if frames[fb]['timestamp_s'] - frames[fa]['timestamp_s'] > cfg['max_gap_s']:
                break
            if gset[fa] - gset[fb] and gset[fb] - gset[fa]:
                pairs.append((fa, fb))
    scale = min(0.5, cfg['max_width_px'] / max(frames[f]['width_px'] for f in det)) if det else 1.0
    store = _FrameStore(root, frames, obs, scale, cfg)
    links, rejected, seen = [], [], set()
    tested = registered = same_frame = 0
    for pi, (fa, fb) in enumerate(pairs):
        if progress and pi % 50 == 0:
            progress(pi, len(pairs))
        if store.feat(fa)[1] is None or store.feat(fb)[1] is None:
            continue
        tested += 1
        reg = register_frames(store.greys[fa], store.feat(fa), store.greys[fb], store.feat(fb), scale, cfg)
        if reg is None:
            continue
        registered += 1
        size = lambda f: (frames[f]['width_px'], frames[f]['height_px'])
        for m in _overlap_matches(reg, by_frame[fa], by_frame[fb], obs, store, size(fa), size(fb), cfg):
            ga, gb = group_of[m['a']], group_of[m['b']]
            if ga == gb:
                continue
            if frames_of[ga] & frames_of[gb]:
                same_frame += 1                                                                 # the two groups are seen together: two books, whatever the match says
                continue
            row = {'groups': [ga, gb], 'observation_ids': [m['a'], m['b']], 'frame_ids': [fa, fb], 'segments': [segments.get(fa, 0), segments.get(fb, 0)],
                   'gap_s': round(frames[fb]['timestamp_s'] - frames[fa]['timestamp_s'], 3), 'inliers': reg['inliers'], 'registration_coverage': reg['coverage'],
                   'view_scale': reg['scale'], 'appearance_score': round(m['appearance'], 4), 'across_overlap': m['across_overlap'], 'kind': KIND_SHORT_GAP}
            if m.get('appearance_stretch') is not None:
                row['appearance_stretch'] = round(m['appearance_stretch'], 4)
            if m['runner_up'] is not None:
                rejected.append({**row, 'reason': 'near-tie: the warped spine covers another spine within the margin (adjacent/identical copies stay separate)',
                                 'runner_up_overlap': round(m['runner_up'], 3)})
                continue
            key = tuple(sorted((ga, gb)))
            if key in seen:
                continue                                                                        # the first (earliest) registered frame pair of a group pair is its evidence
            seen.add(key)
            links.append(row)
    if progress:
        progress(len(pairs), len(pairs))
    return {'links': links, 'rejected': rejected, 'frames': len(det), 'pairs_in_window': len(pairs), 'pairs_tested': tested, 'pairs_registered': registered,
            'same_frame_matches_skipped': same_frame, 'scale': scale, 'config': {**{k: cfg[k] for k in (*SHORT_GAP, *CROSS) if k not in ('window_s', 'frames_per_side', 'tol_widths', 'geometry_weight')},
                                       'stretch_appearance': bool(cfg['stretch_appearance'])},
            'seconds': round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------------------------------ revisit_registration
# On YugoTest13 most remaining duplicates are the SAME book seen again 10-20 s later (the
# pass comes back over a shelf section), outside the 3 s short-gap window. Registering those two detection frames directly and applying the short-gap matcher
# unchanged (mutual-best warped-spine across-overlap >= 0.5 both ways, near-tie veto, appearance 0.5 on the registered pair) separated the labelled revisit
# duplicates from neighbours (23/28 vs 0/18). One extra guard for these longer gaps, the LAYOUT CHECK: a true registration aligns spines that look alike, a
# false one between two look-alike cubbies (identical series spines) aligns mostly different spines, so a frame pair where fewer than half of the mutual-best
# aligned spines pass the appearance gate is dropped whole (0.5 = a majority, not fitted). Only frame pairs more than SHORT_GAP['max_gap_s'] and at most
# max_gap_s apart are registered here (pairs within 3 s stay the short-gap evidence); still no loop closure across the whole sweep. The 20 s window is the
# user's choice (2026-10-08), not a production constant. Same-frame exclusion and the reading veto apply at the union (build).
KIND_REVISIT = 'revisit_registration'
REVISIT = {'min_gap_s': SHORT_GAP['max_gap_s'],     # 3.0 s: shorter gaps are the short-gap evidence
           'max_gap_s': 20.0,                       # the user's window (YT13's revisit bursts are 10.8-20.0 s apart)
           'min_layout_share': 0.5}                 # layout check: share of a registration's aligned spines that must pass the appearance gate
REVISIT_ENV = 'LIBRARY_REVISIT_MERGE'
REVISIT_DEFAULT = '1'


def revisit_enabled(flag: bool | None = None) -> bool:
    return bool(flag) if flag is not None else os.environ.get(REVISIT_ENV, REVISIT_DEFAULT) == '1'


def revisit_links(run_dir: str | Path, frames: dict[str, dict[str, Any]], observations: list[dict[str, Any]], groups: list[dict[str, Any]],
                  segments: dict[str, int], config: dict[str, Any] | None = None, progress=None, workers: int | None = None) -> dict[str, Any]:
    """Same-book links between track groups seen in two detection frames more than ``min_gap_s`` and at most ``max_gap_s`` apart (a revisit): the short-gap
    matcher plus the layout check. Registrations run in a thread pool (OpenCV releases the GIL; each task reseeds the RANSAC RNG, so the result does not depend
    on scheduling). One evidence line per union (build)."""
    from concurrent.futures import ThreadPoolExecutor
    cfg = {**CROSS, **SHORT_GAP, **REVISIT, **(config or {})}
    t0 = time.perf_counter()
    root = Path(run_dir)
    group_of = {o: g['instance_id'] for g in groups for o in g['observation_ids']}
    frames_of = collections.defaultdict(set)
    for o in group_of:
        frames_of[group_of[o]].add(o.split('-o')[0])
    obs = {o['observation_id']: o for o in observations if o['kind'] == 'book' and o['observation_id'] in group_of and _quad(o) is not None}
    by_frame: dict[str, list[str]] = collections.defaultdict(list)
    for oid, o in obs.items():
        by_frame[o['frame_id']].append(oid)
    det = sorted((f for f in frames if len(by_frame.get(f, [])) >= cfg['min_obs']), key=lambda f: (frames[f]['timestamp_s'], f))
    gset = {f: {group_of[o] for o in by_frame[f]} for f in det}
    pairs = []
    for i, fa in enumerate(det):
        for fb in det[i + 1:]:
            gap = frames[fb]['timestamp_s'] - frames[fa]['timestamp_s']
            if gap > cfg['max_gap_s']:
                break
            if gap > cfg['min_gap_s'] and gset[fa] - gset[fb] and gset[fb] - gset[fa]:
                pairs.append((fa, fb))
    scale = min(0.5, cfg['max_width_px'] / max(frames[f]['width_px'] for f in det)) if det else 1.0
    store = _FrameStore(root, frames, obs, scale, cfg)
    for f in sorted({f for p in pairs for f in p}, key=lambda f: (frames[f]['timestamp_s'], f)):
        store.feat(f)                                                                           # SIFT once per frame, serially (one detector object)
    todo = [(fa, fb) for fa, fb in pairs if store.feats[fa][1] is not None and store.feats[fb][1] is not None]

    def register(pair):
        fa, fb = pair
        cv2.setRNGSeed(0)                                                                       # thread-local RANSAC RNG: same result whatever thread runs the pair
        return register_frames(store.greys[fa], store.feats[fa], store.greys[fb], store.feats[fb], scale, cfg)
    n_workers = workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    regs = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for k, reg in enumerate(pool.map(register, todo)):
            regs.append(reg)
            if progress and k % 500 == 0:
                progress(k, len(todo))
    loose = {**cfg, 'min_appearance': -math.inf}                                                # every aligned spine with its appearance (the layout check counts them)
    size = lambda f: (frames[f]['width_px'], frames[f]['height_px'])
    links, rejected, layout_rejected, seen = [], [], [], set()
    registered = same_frame = 0
    for (fa, fb), reg in zip(todo, regs):
        if reg is None:
            continue
        registered += 1
        aligned = _overlap_matches(reg, by_frame[fa], by_frame[fb], obs, store, size(fa), size(fb), loose)
        alike = [m for m in aligned if m['appearance'] >= cfg['min_appearance']]
        if not alike:
            continue
        gap = round(frames[fb]['timestamp_s'] - frames[fa]['timestamp_s'], 3)
        if len(alike) < cfg['min_layout_share'] * len(aligned):
            layout_rejected.append({'frame_ids': [fa, fb], 'gap_s': gap, 'inliers': reg['inliers'], 'aligned_spines': len(aligned), 'alike_spines': len(alike)})
            continue
        for m in alike:
            ga, gb = group_of[m['a']], group_of[m['b']]
            if ga == gb:
                continue
            if frames_of[ga] & frames_of[gb]:
                same_frame += 1                                                                 # the two groups are seen together: two books
                continue
            row = {'groups': [ga, gb], 'observation_ids': [m['a'], m['b']], 'frame_ids': [fa, fb], 'segments': [segments.get(fa, 0), segments.get(fb, 0)],
                   'gap_s': gap, 'inliers': reg['inliers'], 'registration_coverage': reg['coverage'], 'view_scale': reg['scale'],
                   'appearance_score': round(m['appearance'], 4), 'across_overlap': m['across_overlap'], 'layout': [len(alike), len(aligned)], 'kind': KIND_REVISIT}
            if m['runner_up'] is not None:
                rejected.append({**row, 'reason': 'near-tie: the warped spine covers another spine within the margin (adjacent/identical copies stay separate)',
                                 'runner_up_overlap': round(m['runner_up'], 3)})
                continue
            key = tuple(sorted((ga, gb)))
            if key in seen:
                continue                                                                        # the first registered frame pair of a group pair is its evidence
            seen.add(key)
            links.append(row)
    if progress:
        progress(len(todo), len(todo))
    return {'links': links, 'rejected': rejected, 'layout_rejected': layout_rejected, 'frames': len(det), 'pairs_in_window': len(pairs), 'pairs_tested': len(todo),
            'pairs_registered': registered, 'same_frame_matches_skipped': same_frame, 'scale': scale, 'workers': n_workers,
            'config': {k: cfg[k] for k in (*REVISIT, 'min_across_overlap', 'min_appearance', 'margin', 'min_inliers', 'min_coverage', 'min_scale', 'max_scale')},
            'seconds': round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------------------------------ shelf_plane_position
# A tracking break that no frame pair registers across (2D) leaves the same book as two groups in two segments. The merged 3D map (pipeline.recon3d +
# recon3d_merge) registers most frames of a sweep in ONE model and places each group's marker on the fitted shelf plane in that common frame
# (recon3d.json: alignment 'shelf_aligned', z = 0 the plane, +y spine-up). Two groups of DIFFERENT segments whose markers lie on one spine position of
# that plane are one book. Constants declared before any count was seen; every one is an existing production gate or a declared tolerance, none fitted to 458:
KIND_PLANE = 'shelf_plane_position'
PLANE = {
    'tol_widths': CROSS['tol_widths'],      # 0.5: marker centres at most half a spine width apart ACROSS the spine axis (width = the larger of the two groups' own
                                            #   polygon widths projected to the plane; = the production match radius)
    'min_along_overlap': PANORAMA_ACCEPT['min_across_overlap'],   # 0.5: the two spines' extents ALONG the spine axis overlap by >= half the shorter one
    'min_height_ratio': 0.75, 'max_height_ratio': 1.33,           # similar spine height on the plane (shorter / longer and its inverse)
    'margin': CROSS['margin'],              # 0.07 near-tie veto: a competing candidate for either group within this score -> ambiguous, no link
    'min_appearance': CROSS['min_appearance'],   # 0.5 (round 2): cross_view appearance of the two groups' best sightings (sharpest frame), the production gate
    'geometry_weight': CROSS['geometry_weight'], # 0.1: score = appearance - 0.1 x (distance / tolerance), the cross-segment score
}
PLANE_ENV = 'LIBRARY_PLANE_MERGE'          # default set by PLANE_DEFAULT
PLANE_DEFAULT = '1'


def plane_merge_enabled(flag: bool | None = None) -> bool:
    return bool(flag) if flag is not None else os.environ.get(PLANE_ENV, PLANE_DEFAULT) == '1'


def usable_recon3d(d: dict[str, Any]) -> bool:
    return d.get('status') != 'failed' and bool((d.get('alignment') or {}).get('applied')) and bool(d.get('markers')) and bool(d.get('cameras'))


def best_recon3d(run_dir: str | Path) -> Path | None:
    """The ``<run>_recon3d_vN/recon3d.json`` beside the run with the largest registered share among builds that are not 'failed', are shelf-aligned and
    carry group markers (ties: the newest). Several builds of one run exist (profiles, merge on/off); the one model that holds the most frames is the common frame."""
    run_dir = Path(run_dir)
    best = None
    for p in run_dir.parent.glob(f'{run_dir.name}_recon3d_v*'):
        f = p / 'recon3d.json'
        tail = p.name[len(run_dir.name) + len('_recon3d_v'):]
        if not tail.isdigit() or not f.is_file():
            continue
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not usable_recon3d(d):
            continue
        from pipeline.recon3d import map_consistency        # shelf-plane check: a map that is not one flat shelf face is never the common frame;
        state = map_consistency(d)                         # a checked consistent map beats an unchecked (pre-plane-check) one of higher registration
        if state == 'inconsistent':
            continue
        key = (state == 'consistent', float(d.get('registered_ratio') or 0.0), int(tail))
        if best is None or key > best[0]:
            best = (key, f)
    return None if best is None else best[1]


def _plane_footprint(sightings: list[dict[str, Any]], cams: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Median spine width / height and long-axis direction of a group's polygons cast onto the shelf plane (z = 0 of the aligned frame) from its registered cameras."""
    ws, hs, axes = [], [], []
    for s in sightings:
        c = cams.get(s.get('frame_id'))
        q = _quad(s)
        if c is None or q is None:
            continue
        R, C, f = np.asarray(c['rotation_world_to_cam'], float), np.asarray(c['center'], float), float(c['focal_px'])
        q = np.asarray(q, float)
        rays = np.c_[(q[:, 0] - c['width_px'] / 2) / f, (q[:, 1] - c['height_px'] / 2) / f, np.ones(4)] @ R      # = (R.T @ d) for each corner
        if np.any(np.abs(rays[:, 2]) < 1e-9):
            continue
        t = -C[2] / rays[:, 2]
        if np.any(t <= 0):
            continue
        P = C[None, :2] + t[:, None] * rays[:, :2]
        e = [P[(i + 1) % 4] - P[i] for i in range(4)]
        n = [float(np.linalg.norm(v)) for v in e]
        long_i = 0 if n[0] + n[2] >= n[1] + n[3] else 1
        a = e[long_i] - e[long_i + 2]                         # opposite sides point opposite ways around the quad
        if np.linalg.norm(a) < 1e-12:
            continue
        a = a / np.linalg.norm(a)
        if a[1] < 0 or (a[1] == 0 and a[0] < 0):
            a = -a
        hs.append((n[long_i] + n[long_i + 2]) / 2)
        ws.append((n[1 - long_i] + n[3 - long_i]) / 2)
        axes.append(a)
    if not ws:
        return None
    ax = np.mean(axes, 0)
    return {'width': float(np.median(ws)), 'height': float(np.median(hs)), 'axis': ax / (np.linalg.norm(ax) or 1.0), 'views': len(ws)}


def _plane_candidates(recon: dict[str, Any], known: dict[str, Any], segs_of: dict[str, set[int]], cfg: dict[str, Any]) -> tuple[list[tuple], list[tuple]]:
    """Placed groups [(gid, segment, xy, footprint, method)] and the geometric candidates [(distance/tolerance, i, j, across in widths, along overlap, height ratio)]."""
    cams = {c['frame_id']: c for c in recon.get('cameras') or []}
    geo = []
    for m in recon.get('markers') or []:
        g = m.get('group_id')
        if g not in known or m.get('xyz') is None or len(segs_of[g]) != 1:
            continue
        fp = _plane_footprint(m.get('sightings') or [], cams)
        if fp is None or fp['width'] <= 0 or fp['height'] <= 0:
            continue
        geo.append((g, next(iter(segs_of[g])), np.asarray(m['xyz'][:2], float), fp, m.get('method')))
    raw = []
    if geo:
        P = np.array([x[2] for x in geo])
        W = np.array([x[3]['width'] for x in geo])
        Hh = np.array([x[3]['height'] for x in geo])
        A = np.array([x[3]['axis'] for x in geo])
        S = np.array([x[1] for x in geo])
        for i in range(len(geo)):
            j = np.arange(i + 1, len(geo))
            j = j[S[j] != S[i]]
            if not len(j):
                continue
            sgn = np.sign(A[j] @ A[i]); sgn[sgn == 0] = 1
            u = A[i][None, :] + A[j] * sgn[:, None]
            u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
            v = np.c_[-u[:, 1], u[:, 0]]
            dv = P[j] - P[i][None, :]
            across = np.abs(np.sum(dv * v, 1))
            tol = cfg['tol_widths'] * np.maximum(W[i], W[j])
            along_c = np.sum(dv * u, 1)                                       # centre of j minus centre of i along the axis
            lo = np.maximum(-Hh[i] / 2, along_c - Hh[j] / 2)
            hi = np.minimum(Hh[i] / 2, along_c + Hh[j] / 2)
            ov = np.maximum(0.0, hi - lo) / np.minimum(Hh[i], Hh[j])
            hr = Hh[i] / Hh[j]
            ok = (across <= tol) & (ov >= cfg['min_along_overlap']) & (hr >= cfg['min_height_ratio']) & (hr <= cfg['max_height_ratio'])
            for k in np.nonzero(ok)[0]:
                jj = int(j[k])
                raw.append((float(across[k] / tol[k]), i, jj, float(across[k] / max(W[i], W[jj])), float(ov[k]), float(hr[k])))
    return geo, raw


def shelf_plane_links(recon: dict[str, Any], books: list[dict[str, Any]], segments: dict[str, int], *, model_id: str | None = None,
                      config: dict[str, Any] | None = None, appearance=None, agree_with: list[tuple[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Candidate unions between groups of DIFFERENT tracking segments whose recon3d markers (one model, both placed) sit on one spine position of the shelf
    plane: across distance <= tol_widths x the larger projected spine width, along-axis overlap >= min_along_overlap of the shorter spine, height ratio in
    [min, max]; one-to-one per segment pair, strongest first, with the near-tie veto (another candidate for exactly one of the two within ``margin``).
    ``appearance(ga, gb)`` (round 2): the candidate also needs the production appearance gate (score = appearance - 0.1 x distance/tolerance);
    ``agree_with`` [(model id, recon)]: the pair must be within the same tolerances in every other build too (two-build agreement).
    Same-frame exclusion and the reading veto follow in build()'s union()."""
    cfg = {**PLANE, **(config or {})}
    t0 = time.perf_counter()
    known = {b['instance_id']: b for b in books}
    frames_of = {g: {o.split('-o')[0] for o in b['observation_ids']} for g, b in known.items()}
    segs_of = {g: {segments.get(f, 0) for f in fs} for g, fs in frames_of.items()}
    geo, raw = _plane_candidates(recon, known, segs_of, cfg)
    others = [(mid, {frozenset((g[x[1]][0], g[x[2]][0])) for x in r}) for mid, (g, r) in
              ((mid, _plane_candidates(rc, known, segs_of, cfg)) for mid, rc in (agree_with or []))]
    by_pair: dict[tuple[int, int], list[tuple]] = collections.defaultdict(list)
    dropped = collections.Counter()
    for nd, i, jj, aw, ov, hr in raw:
        pair = frozenset((geo[i][0], geo[jj][0]))
        miss = [mid for mid, cs in others if pair not in cs]
        if miss:
            dropped['not_within_tolerance_in_' + miss[0]] += 1
            continue
        a = None
        if appearance is not None:
            a = appearance(geo[i][0], geo[jj][0])
            if a is None or a < cfg['min_appearance']:
                dropped['appearance_below_gate' if a is not None else 'appearance_unavailable'] += 1
                continue
        score = (a - cfg['geometry_weight'] * nd) if a is not None else 1.0 - nd
        key = (min(geo[i][1], geo[jj][1]), max(geo[i][1], geo[jj][1]))
        by_pair[key].append((score, i, jj, aw, ov, hr, a))
    links, rejected = [], []
    for key, cands in sorted(by_pair.items()):
        cands.sort(reverse=True)
        used = set()
        for score, i, j, aw, ov, hr, a in cands:
            if i in used or j in used:
                continue
            gi, gj = geo[i], geo[j]
            row = {'kind': KIND_PLANE, 'groups': [gi[0], gj[0]], 'segments': [int(gi[1]), int(gj[1])], 'model': model_id,
                   'plane_xy': [[round(float(c), 4) for c in gi[2]], [round(float(c), 4) for c in gj[2]]],
                   'spine_width_plane': [round(gi[3]['width'], 4), round(gj[3]['width'], 4)], 'spine_height_plane': [round(gi[3]['height'], 4), round(gj[3]['height'], 4)],
                   'across_spine_widths': round(aw, 4), 'along_overlap': round(ov, 3), 'height_ratio': round(hr, 3), 'marker_methods': [gi[4], gj[4]],
                   'views': [gi[3]['views'], gj[3]['views']], 'score': round(score, 4), 'appearance_score': None if a is None else round(a, 4),
                   'agreed_models': [mid for mid, _ in others]}
            competing = [s for s, a, b, *_ in cands if (a == i or b == i) != (a == j or b == j)]
            if competing and score - max(competing) < cfg['margin']:
                rejected.append({**row, 'reason': 'near-tie competing spine position on the plane (adjacent/identical copies stay separate)', 'runner_up_score': round(max(competing), 4)})
                continue
            used.update((i, j))
            links.append(row)
    return {'model': model_id, 'groups_placed': len(geo), 'groups_total': len(known), 'candidates': len(raw), 'candidates_dropped': dict(dropped),
            'candidates_kept': sum(len(c) for c in by_pair.values()), 'agreement_models': [mid for mid, _ in others], 'appearance_gate': appearance is not None,
            'links': links, 'rejected': rejected,
            'config': {k: cfg[k] for k in PLANE}, 'seconds': round(time.perf_counter() - t0, 2)}


def group_appearance(root: Path, frames: dict[str, dict[str, Any]], observations: list[dict[str, Any]], books: list[dict[str, Any]]):
    """(ga, gb) -> cross_view appearance of the two groups' best sightings (sharpest frame, then the longest spine polygon), or None without an image."""
    from pipeline import cross_view as CV
    from pipeline.detect import resolve_evidence
    obs = {o['observation_id']: o for o in observations}
    by_id = {b['instance_id']: b for b in books}
    patches: dict[str, Any] = {}

    def best(g):
        oids = [o for o in by_id[g]['observation_ids'] if o in obs and _quad(obs[o]) is not None and obs[o].get('frame_id') in frames]
        if not oids:
            return None
        return max(oids, key=lambda o: (float((frames[obs[o]['frame_id']].get('quality') or {}).get('sharpness') or 0.0), spine_long_side(_quad(obs[o]))))

    def patch(g):
        if g not in patches:
            o = best(g)
            im = None if o is None else cv2.imread(str(resolve_evidence(root, frames[obs[o]['frame_id']]['image_path'])))
            patches[g] = None if im is None else CV._patch(im, _quad(obs[o]))
        return patches[g]

    def appearance(ga, gb):
        pa, pb = patch(ga), patch(gb)
        return None if pa is None or pb is None else float(CV.appearance(pa, pb))
    return appearance


def _load_recon(path: Path) -> tuple[str, dict[str, Any]] | None:
    d = json.loads(path.read_text())
    return (f'{path.parent.name}:{(d.get("merge") or {}).get("model") or "largest"}', d) if usable_recon3d(d) else None


def plane_links_for(root: Path, books: list[dict[str, Any]], segments: dict[str, int], recon3d: str | Path | None = None, *, appearance=None,
                    agree: list[str | Path] | None = None) -> dict[str, Any]:
    """Load the run's best merged 3D map (or ``recon3d``: a dir or a recon3d.json) and return shelf_plane_links, or a row saying why there is none."""
    path = None
    if recon3d:
        path = Path(recon3d) / 'recon3d.json' if Path(recon3d).is_dir() else Path(recon3d)
    else:
        path = best_recon3d(root)
    if not any(segments.values()):
        return {'model': None, 'links': [], 'rejected': [], 'reason': 'single tracking segment: nothing to link across'}
    if path is None or not path.is_file():
        return {'model': None, 'links': [], 'rejected': [], 'reason': 'no usable 3D map (no recon3d build that is shelf-aligned, not failed and has markers)'}
    d = json.loads(path.read_text())
    if not usable_recon3d(d):
        return {'model': None, 'links': [], 'rejected': [], 'recon3d': str(path), 'status': d.get('status'),
                'reason': f'3D map not usable (status {d.get("status")}, needs: not failed, shelf-aligned, group markers)'}
    model_id = f'{path.parent.name}:{(d.get("merge") or {}).get("model") or "largest"}'
    others = []
    for a in agree or []:
        ap = Path(a) / 'recon3d.json' if Path(a).is_dir() else Path(a)
        got = _load_recon(ap) if ap.is_file() else None
        if got is None:
            return {'model': model_id, 'links': [], 'rejected': [], 'reason': f'agreement build not usable: {ap}'}
        others.append(got)
    out = shelf_plane_links(d, books, segments, model_id=model_id, appearance=appearance, agree_with=others)
    out.update(recon3d=str(path), registered_ratio=d.get('registered_ratio'), status=d.get('status'))
    return out


# ------------------------------------------------------------------------------------------ panorama_overlap acceptance
def panorama_pairs(pan_lines: list[dict[str, Any]], known: set[str] | dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
    """The candidate unions of panorama_count's merge_evidence: on each spine, the group with the highest IoU (the anchor) with every other group. [(spine, anchor, other)]."""
    out = []
    for spine in pan_lines:
        ids = [g['id'] for g in spine.get('groups') or [] if g['id'] in known]
        if len(ids) < 2:
            continue
        iou = {g['id']: g.get('iou') for g in spine['groups']}
        ids.sort(key=lambda i: -(iou.get(i) or 0))
        out += [(spine, ids[0], other) for other in ids[1:]]
    return out


def across_overlap(qa: np.ndarray, qo: np.ndarray) -> float:
    """Share of the narrower of two spine quads covered ACROSS the long axis of ``qo`` (0 when they do not also overlap along it). A truncated or partial detection of
    the same spine keeps its across extent, so this is robust where a centre distance is not (one quad stops at the frame edge or covers half the spine)."""
    e = [qo[(i + 1) % 4] - qo[i] for i in range(len(qo))]
    long_ = max(e, key=lambda v: float(np.linalg.norm(v)))
    u = long_ / max(float(np.linalg.norm(long_)), 1e-9)
    n = np.array([-u[1], u[0]])
    pa, po = qa @ n, qo @ n
    inter = min(pa.max(), po.max()) - max(pa.min(), po.min())
    if inter <= 0:
        return 0.0
    la, lo = qa @ u, qo @ u
    if min(la.max(), lo.max()) - max(la.min(), lo.min()) <= 0:
        return 0.0
    return float(inter / max(min(pa.max() - pa.min(), po.max() - po.min()), 1e-9))


def _clip_to_frame(q: np.ndarray, w: float, h: float) -> np.ndarray | None:
    rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    area, inter = cv2.intersectConvexConvex(cv2.convexHull(np.float32(q)).reshape(-1, 2), rect)
    return None if inter is None or area <= 1.0 else inter.reshape(-1, 2).astype(np.float64)


def panorama_frame_checks(run_dir: str | Path, frames: dict[str, dict[str, Any]], observations: list[dict[str, Any]], groups: list[dict[str, Any]],
                          pairs: list[tuple[str, str]], config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Direct frame check of panorama_overlap candidates: for the two groups' closest-in-time sightings (up to PANORAMA_FRAME_PAIRS frame pairs, distinct frames), register the
    two frames with the cross-segment gates (SIFT/RANSAC, 30 inliers, coverage, zoom), project A's spine into B's frame and ask which detected spine it covers.
    'agree': B is the detected spine A covers most, by >= PANORAMA_ACCEPT['min_across_overlap']; 'contradicts': A lands on another detected spine or on none;
    'ambiguous': A covers B by >= that but another detected spine as much, within CROSS['margin'] (a projected quad spanning two spines; formerly 'contradicts' by name order);
    'unregistered': no frame pair registered (the decision then rests on the placement alone). Keyed 'ga|gb'."""
    from pipeline import cross_view as CV
    from pipeline.detect import resolve_evidence
    cfg = {**CROSS, **(config or {})}
    root = Path(run_dir)
    obs = {o['observation_id']: o for o in observations if o.get('kind', 'book') == 'book' and _quad(o) is not None and o.get('frame_id') in frames}
    by_frame: dict[str, list[str]] = collections.defaultdict(list)
    for oid, o in obs.items():
        by_frame[o['frame_id']].append(oid)
    members = {g['instance_id']: [o for o in g['observation_ids'] if o in obs] for g in groups}
    if not pairs or not frames:
        return {}
    scale = min(0.5, cfg['max_width_px'] / max(f.get('width_px') or 1 for f in frames.values()))
    sift = cv2.SIFT_create(nfeatures=cfg['nfeatures'])
    cv2.setRNGSeed(0)
    feats: dict[str, tuple] = {}

    def feat(f):
        if f not in feats:
            im = cv2.imread(str(resolve_evidence(root, frames[f]['image_path'])), cv2.IMREAD_GRAYSCALE)
            if im is None:
                feats[f] = (None, (None, None))
            else:
                g = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                feats[f] = (g, sift.detectAndCompute(g, None))
        return feats[f]
    ts = lambda o: frames[obs[o]['frame_id']]['timestamp_s']
    out = {}
    for ga, gb in pairs:
        A, B = members.get(ga) or [], members.get(gb) or []
        cands = sorted(((abs(ts(a) - ts(b)), a, b) for a in A for b in B), key=lambda x: (x[0], x[1], x[2]))
        used, tried, res = set(), 0, None
        for dt, a, b in cands:
            fa, fb = obs[a]['frame_id'], obs[b]['frame_id']
            if fa == fb or fa in used or fb in used:
                continue
            used |= {fa, fb}
            tried += 1
            (gra, fta), (grb, ftb) = feat(fa), feat(fb)
            reg = register_frames(gra, fta, grb, ftb, scale, cfg) if gra is not None and grb is not None else None
            if reg is not None:
                qa = _clip_to_frame(CV._proj(_quad(obs[a]), reg['H']), frames[fb]['width_px'], frames[fb]['height_px'])
                ov = {o: across_overlap(qa, np.asarray(_quad(obs[o]), np.float64)) for o in by_frame[fb]} if qa is not None else {}
                best = max(sorted(ov), key=lambda o: ov[o]) if ov else None
                tied = sorted(o for o in ov if ov[o] >= ov[best] - cfg['margin']) if best else []   # near tie: the production ambiguity margin (0.07)
                covers = ov.get(b, 0.0) >= PANORAMA_ACCEPT['min_across_overlap'] and b in tied
                if covers:
                    best = b
                res = {'outcome': ('agree' if len(tied) == 1 else 'ambiguous') if covers else 'contradicts', **({'tied_spines': tied} if covers and len(tied) > 1 else {}), 'observation_ids': [a, b], 'frame_ids': [fa, fb], 'gap_s': round(dt, 3), 'inliers': reg['inliers'],
                       'registration_coverage': reg['coverage'], 'view_scale': reg['scale'], 'across_overlap': round(ov.get(b, 0.0), 3),
                       'best_spine': best, 'best_overlap': round(ov.get(best, 0.0), 3) if best else 0.0, 'pairs_tried': tried}
                break
            if tried >= PANORAMA_FRAME_PAIRS:
                break
        out[f'{ga}|{gb}'] = res or {'outcome': 'unregistered', 'pairs_tried': tried}
    return out


# ------------------------------------------------------------------------------------------ instances
def _reliability_of(transitions_total: int, transitions_registered: int) -> str:
    """Association (fragmentation) reliability, declared rule, not fitted: how many of the tracking breaks could be bridged by registration."""
    if transitions_total == 0:
        return 'high'
    share = transitions_registered / transitions_total
    return 'high' if share >= 0.8 else 'medium' if share >= 0.5 else 'low'


ORDER = {'unknown': 0, 'low': 1, 'medium': 2, 'high': 3}


def weaker(a: str | None, b: str | None) -> str | None:
    """The weaker of two reliability labels (unknown stays unknown)."""
    if a is None:
        return b
    if b is None:
        return a
    return a if ORDER.get(a, 0) <= ORDER.get(b, 0) else b


def own_reliability_totals(totals: dict[str, Any], cross: dict[str, Any] | None) -> dict[str, Any]:
    """The totals pipeline.panorama_attach.decide judges, with the count reliability the instance layer itself will report (the weaker of the
    packet's reading and this pass's association reliability, the same rule ``apply`` writes into the packet). Before, the FIRST instance pass judged the
    panorama on the capture reading alone (YugoTest14: capture "high", 0 of 5 tracking breaks registered) and showed a count the second pass replaced."""
    if cross is None:
        return totals
    rel = totals.get('count_reliability')
    assoc = _reliability_of(cross.get('transitions_total', 0), cross.get('transitions_registered', 0))
    own = weaker(rel, assoc)
    if own == rel:
        return totals
    return {**totals, 'count_reliability': own,
            'count_reliability_basis': (f"the instance layer's own reading: packet {rel or 'unknown'}, association {assoc} "
                                        f"({cross.get('transitions_registered', 0)} of {cross.get('transitions_total', 0)} tracking breaks registered across)")}


def text_tokens(text: str) -> set[str]:
    """Normalised tokens of a spine reading: lower case, accents stripped, alphanumeric runs of >= 2 characters."""
    import re
    import unicodedata
    flat = unicodedata.normalize('NFKD', str(text or '')).encode('ascii', 'ignore').decode().lower()
    return {t for t in re.findall(r'[a-z0-9]+', flat) if len(t) >= 2}


def _trigrams(text: str) -> set[str]:
    import re
    import unicodedata
    flat = re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', str(text or '')).encode('ascii', 'ignore').decode().lower())
    return {flat[i:i + 3] for i in range(len(flat) - 2)}


def text_overlap(a: str, b: str) -> float | None:
    """Agreement of two spine readings beyond OCR noise: the larger of the token overlap and the character-trigram overlap, each as the share of the shorter reading
    found in the other (so a truncated or partly garbled read of the same title, 'arquitectu' vs 'arquitectura', agrees). None when either has fewer than 5 letters."""
    ta, tb = text_tokens(a), text_tokens(b)
    ga, gb = _trigrams(a), _trigrams(b)
    if len(ga) < 3 or len(gb) < 3:
        return None
    tok = len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0
    return max(tok, len(ga & gb) / min(len(ga), len(gb)))


def group_texts(books: list[dict[str, Any]], readings: list[dict[str, Any]] | None, min_chars: int = READING_VETO['min_chars']) -> dict[str, dict[str, Any]]:
    """Per track group, its best readable spine text (>= min_chars characters; highest read confidence, then longest) from the run's readings."""
    rows: dict[str, dict[str, Any]] = {}
    for r in readings or []:
        t = str(r.get('visible_text') or '').strip()
        if len(t) >= min_chars and r.get('observation_id'):
            rows[r['observation_id']] = r
    out = {}
    for b in books:
        cands = [rows[o] for o in b['observation_ids'] if o in rows]
        if cands:
            best = max(cands, key=lambda r: (r.get('read_confidence') or 0.0, len(str(r.get('visible_text') or ''))))
            out[b['instance_id']] = {'text': str(best['visible_text']).strip(), 'observation_id': best['observation_id'], 'read_confidence': best.get('read_confidence')}
    return out


def spine_long_side(poly: Any) -> float:
    p = np.asarray(poly, float)
    return float(np.linalg.norm(np.roll(p, -1, 0) - p, axis=1).max())


def far_sightings(books: list[dict[str, Any]], observations: list[dict[str, Any]] | None, fraction: float = FAR_FRACTION) -> tuple[set[str], float | None]:
    """Observation ids whose spine (polygon long side, native pixels) is under ``fraction`` of the run's median native spine (median over the groups' best polygons, exactly
    pipeline.panorama_count's 'far' rule). Returns (far observation ids, run median px)."""
    native = [spine_long_side(b['polygon_xy']) for b in books if len(b.get('polygon_xy') or []) == 4]
    if not native or not observations:
        return set(), None
    med = float(np.median(native))
    far = {o['observation_id'] for o in observations if o.get('kind') == 'book' and len(o.get('polygon_xy') or []) == 4 and spine_long_side(o['polygon_xy']) < fraction * med}
    return far, med


def cover_view_sightings(observations: list[dict[str, Any]] | None, max_aspect: float = COVER_VIEW_MAX_ASPECT) -> set[str]:
    """Observation ids of book sightings whose polygon (native pixels) is under ``max_aspect`` long/short: cover or flat views. Sightings without a polygon are never in
    the set (no demotion on missing evidence)."""
    out = set()
    for o in observations or []:
        q = o.get('polygon_xy') or []
        if o.get('kind') == 'book' and len(q) == 4:
            e = np.linalg.norm(np.roll(np.asarray(q, float), -1, 0) - np.asarray(q, float), axis=1)
            if e.max() < max_aspect * max(float(e.min()), 1e-6):
                out.add(o['observation_id'])
    return out


def piece_sightings(run_dir: str | Path, pc: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Every book sighting projected into a NORMAL (detector-counted) piece, in that piece's pixel frame: panorama.json ``overlays`` (one per observation of the piece's
    frames, ``panorama_polygon_xy``, track group ``proposed_instance_id``). None when the panorama is not there."""
    if not pc:
        return None
    pan = Path(run_dir).parent / str(pc.get('panorama_run') or '')
    try:
        comps = json.loads((pan / 'panorama.json').read_text())['components']
    except (OSError, ValueError, KeyError):
        return None
    normal = {c['component_id'] for c in (pc.get('panorama_components') or []) if c.get('class') == 'normal'}
    return [{'component': c['component_id'], 'observation_id': o['observation_id'], 'group_id': o['proposed_instance_id'], 'poly': o['panorama_polygon_xy']}
            for c in comps if c.get('component_id') in normal for o in c.get('overlays') or []
            if o.get('kind', 'book') == 'book' and o.get('proposed_instance_id') and len(o.get('panorama_polygon_xy') or []) >= 3]


def place_on_spines(sightings: list[dict[str, Any]], spines: list[dict[str, Any]]) -> dict[str, tuple[str, str, float]]:
    """{observation_id: (fused spine id, piece, IoU)}: each sighting on the detected spine box of ITS piece it overlaps most (pipeline.panorama_count's placement:
    IoU, or 0.05 when the sighting's centre lies inside a box it does not otherwise overlap; below 0.05 the sighting is on no spine)."""
    from pipeline.panorama_count import _center_in, _hull, _iou
    boxes: dict[str, list[tuple[np.ndarray, str]]] = collections.defaultdict(list)
    for sp in spines:
        for m in sp.get('members') or []:
            if len(m.get('poly') or []) >= 3:
                boxes[m['component']].append((_hull(m['poly']), sp['panorama_spine']))
    out = {}
    for s in sightings:
        h = _hull(s['poly'])
        c = h.mean(axis=0)
        best, biou = None, 0.0
        for bh, sid in boxes.get(s['component'], []):
            if bh[:, 0].max() < h[:, 0].min() or bh[:, 0].min() > h[:, 0].max():
                continue
            v = _iou(h, bh)
            if v == 0.0 and _center_in(bh, c):
                v = RECONCILE['occupy_min_iou']
            if v > biou:
                best, biou = sid, v
        if best is not None and biou >= RECONCILE['occupy_min_iou']:
            out[s['observation_id']] = (best, s['component'], round(biou, 3))
    return out


def neighbour_frame_check(run_dir: str | Path, frames: dict[str, dict[str, Any]], observations: list[dict[str, Any]], config: dict[str, Any] | None = None):
    """The second half of "no counted line on it": a counted line seen only in frames of ANOTHER piece (one that did not register to this one) has no sighting projected
    into this piece. Returns ``check(observation_id, counted) -> dict``: the candidate's frame is registered (cross-segment gates: SIFT/RANSAC, 30 inliers, coverage,
    zoom) with the nearest detection frames within CROSS['window_s'] on each side (CROSS['frames_per_side'] per side), its spine is projected there and the detected
    sighting it covers best is taken (across overlap >= 0.5, near ties within CROSS['margin'] kept). ``counted``: {observation_id: counted instance id}.
    Outcome 'occupied' (a counted line's sighting is that spine in a registered neighbour), 'free' (registered neighbours, none shows a counted line there) or
    'unregistered' (no neighbour frame registered: the piece evidence alone decides)."""
    from pipeline import cross_view as CV
    from pipeline.detect import resolve_evidence
    cfg = {**CROSS, **(config or {})}
    root = Path(run_dir)
    obs = {o['observation_id']: o for o in observations if o.get('kind', 'book') == 'book' and _quad(o) is not None and o.get('frame_id') in frames}
    by_frame: dict[str, list[str]] = collections.defaultdict(list)
    for oid, o in obs.items():
        by_frame[o['frame_id']].append(oid)
    det = sorted(by_frame, key=lambda f: (frames[f]['timestamp_s'], f))
    pos = {f: k for k, f in enumerate(det)}
    scale = min(0.5, cfg['max_width_px'] / max((f.get('width_px') or 1) for f in frames.values())) if frames else 0.5
    sift = cv2.SIFT_create(nfeatures=cfg['nfeatures'])
    feats: dict[str, tuple] = {}

    def feat(f):
        if f not in feats:
            im = cv2.imread(str(resolve_evidence(root, frames[f]['image_path'])), cv2.IMREAD_GRAYSCALE)
            if im is None:
                feats[f] = (None, (None, None))
            else:
                g = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                feats[f] = (g, sift.detectAndCompute(g, None))
        return feats[f]

    def check(oid: str, counted: dict[str, str]) -> dict[str, Any]:
        cv2.setRNGSeed(0)
        fa = obs[oid]['frame_id'] if oid in obs else None
        if fa is None:
            return {'outcome': 'unregistered', 'frames_tried': []}
        k, ta = pos[fa], frames[fa]['timestamp_s']
        side = lambda rng: [det[j] for j in rng if abs(frames[det[j]]['timestamp_s'] - ta) <= cfg['window_s']][:cfg['frames_per_side']]
        nb = side(range(k - 1, -1, -1)) + side(range(k + 1, len(det)))
        nb.sort(key=lambda f: (abs(frames[f]['timestamp_s'] - ta), f))
        tried, registered = [], []
        for fb in nb:
            tried.append(fb)
            (gra, fta), (grb, ftb) = feat(fa), feat(fb)
            reg = register_frames(gra, fta, grb, ftb, scale, cfg) if gra is not None and grb is not None else None
            if reg is None:
                continue
            qa = _clip_to_frame(CV._proj(_quad(obs[oid]), reg['H']), frames[fb]['width_px'], frames[fb]['height_px'])
            ov = {o: across_overlap(qa, np.asarray(_quad(obs[o]), np.float64)) for o in by_frame[fb]} if qa is not None else {}
            best = max(sorted(ov), key=lambda o: ov[o]) if ov else None
            if best is None or ov[best] < PANORAMA_ACCEPT['min_across_overlap']:
                registered.append({'frame_id': fb, 'inliers': reg['inliers'], 'covers': None})
                continue
            tied = sorted(o for o in ov if ov[o] >= ov[best] - cfg['margin'])
            hit = [o for o in tied if o in counted]
            registered.append({'frame_id': fb, 'inliers': reg['inliers'], 'covers': tied, 'overlap': round(ov[best], 3)})
            if hit:
                return {'outcome': 'occupied', 'frame_id': fb, 'inliers': reg['inliers'], 'observation_id': hit[0], 'instance_id': counted[hit[0]],
                        'across_overlap': round(ov[hit[0]], 3), 'frames_tried': tried}
        return {'outcome': 'free' if registered else 'unregistered', 'registered': registered, 'frames_tried': tried}
    return check


def reconcile(out: list[dict[str, Any]], books_by_id: dict[str, dict[str, Any]], panorama_count: dict[str, Any] | None, sightings: list[dict[str, Any]] | None, *,
              panorama_eligible: bool, panorama_reason: str | None, lower: int, upper: int, lines_before: int, frame_check=None) -> dict[str, Any]:
    """Promote unconfirmed single sightings that the panorama detector shows as a distinct, unoccupied spine (in place on ``out``). Returns the record for the artifact.
    Guard: the panorama must be eligible (pipeline.panorama_attach.decide: usable, detector support, reliability, registration) and its count must lie inside the
    instance layer's own [lower, upper] computed before any promotion."""
    pc = panorama_count or {}
    pan_n = pc.get('panorama_count')
    diff = None if pan_n is None else abs(lines_before - pan_n)
    rec: dict[str, Any] = {'rule': ('promote an unconfirmed single sighting (not far-view-only, not cover-view-only, not a demoted non-book) whose sighting lies on a detected spine '
                                    f'of a normal panorama piece (placement IoU >= {RECONCILE["min_spine_iou"]}) that no sighting of a counted line occupies (panorama_count placement, '
                                    f'IoU >= {RECONCILE["occupy_min_iou"]} or centre inside) and that no counted line covers in a registered neighbour detection frame (+-3 s, cross-segment gates, '
                                    'across overlap >= 0.5); one promotion per spine; at most (panorama count - lines) promotions, strongest '
                                    'placement first (toward the panorama, never past it); only when the panorama is eligible and its count lies inside the instance range '
                                    '[lower, upper]'),
                           'constants': dict(RECONCILE), 'panorama_count': pan_n, 'range_before': [lower, upper], 'lines_before': lines_before, 'promotions': 0,
                           'lines_after': lines_before, 'abs_diff_before': diff, 'abs_diff_after': diff, 'eligible': False, 'reason': None, 'candidates': 0, 'outcomes': {},
                           'promoted': [], 'not_promoted': []}
    if not pc:
        rec['reason'] = 'no panorama count for this run'
        return rec
    if not panorama_eligible:
        rec['reason'] = f'panorama not eligible: {panorama_reason}'
        return rec
    if pan_n is None or not (lower <= pan_n <= upper):
        rec['reason'] = (f"panorama count {pan_n} lies outside the instance layer's own range [{lower}, {upper}]: the two estimators disagree beyond the association "
                         'uncertainty, so the panorama decides no line')
        return rec
    spines = pc.get('spines')
    if spines is None:
        rec['reason'] = 'the panorama count predates the per-spine list (re-run pipeline.panorama_count on this run to enable the reconciliation)'
        return rec
    if sightings is None:
        rec['reason'] = 'panorama.json (overlays in piece coordinates) not found next to the run'
        return rec
    rec['eligible'] = True
    rec['reason'] = f'panorama eligible ({panorama_reason}) and its count {pan_n} lies inside the instance range [{lower}, {upper}]'
    placed = place_on_spines(sightings, spines)
    inst_of = {g: i for i in out for g in i['member_group_ids']}
    occupied: dict[str, set[str]] = collections.defaultdict(set)
    for s in sightings:
        inst = inst_of.get(s['group_id'])
        if inst is not None and inst['tier'] != 'unconfirmed' and s['observation_id'] in placed:
            occupied[placed[s['observation_id']][0]].add(inst['instance_id'])
    outcomes: collections.Counter = collections.Counter()
    claims: dict[str, list[tuple[float, str, dict[str, Any], str]]] = collections.defaultdict(list)
    for inst in out:
        if inst['tier'] != 'unconfirmed' or len(inst['member_group_ids']) != 1:
            continue                                  # merged instances are never unconfirmed
        if inst.get('far_only') or inst.get('cover_only') or inst.get('room_pan_only') or inst['tier_reason']:
            outcomes['far_or_cover_view_only'] += 1   # a single sighting that is already unconfirmed carries the flag, not a tier_reason
            continue
        b = books_by_id.get(inst['member_group_ids'][0]) or {}
        if b.get('non_book_demotion') or b.get('classification') == 'ambiguous' or b.get('kind', 'book') != 'book':
            outcomes['demoted_non_book'] += 1
            continue
        rec['candidates'] += 1
        pl = [placed[o] + (o,) for o in inst['observation_ids'] if o in placed]
        if not pl:
            outcomes['not_on_a_spine_of_a_normal_piece'] += 1
            continue
        sid, comp, iou, oid = max(pl, key=lambda x: (x[2], x[3]))
        if iou < RECONCILE['min_spine_iou']:
            outcomes['placement_too_loose'] += 1
            rec['not_promoted'].append({'instance_id': inst['instance_id'], 'panorama_spine': sid, 'component': comp, 'iou': iou,
                                        'reason': f'placement on {sid} IoU {iou:.2f} < {RECONCILE["min_spine_iou"]}'})
            continue
        if occupied.get(sid):
            outcomes['spine_carries_a_counted_line'] += 1
            rec['not_promoted'].append({'instance_id': inst['instance_id'], 'panorama_spine': sid, 'component': comp, 'iou': iou,
                                        'reason': f'spine {sid} carries counted line(s) ' + ', '.join(sorted(occupied[sid])[:3])})
            continue
        claims[sid].append((iou, oid, inst, comp))
    winners = []
    for sid, cl in sorted(claims.items()):
        cl.sort(key=lambda x: (-x[0], x[1]))
        winners.append((sid, cl[0]))
        for iou2, _, other, comp in cl[1:]:
            outcomes['spine_already_promoted'] += 1
            rec['not_promoted'].append({'instance_id': other['instance_id'], 'panorama_spine': sid, 'component': comp, 'iou': iou2,
                                        'reason': f'spine {sid} already claimed by {cl[0][2]["instance_id"]} (one line per spine)'})
    # toward the panorama, never past it: the panorama shows room = panorama count - lines spines more than the lines; the strongest placements are promoted first.
    # Each one is also checked in the nearest detection frames (neighbour_frame_check): a counted line seen there on the same spine keeps it unconfirmed.
    room = max(0, pan_n - lines_before)
    rec['room'] = room
    winners.sort(key=lambda w: (-w[1][0], w[1][1]))
    counted_obs = {o: i['instance_id'] for i in out if i['tier'] != 'unconfirmed' for o in i['observation_ids']}
    for sid, (iou, oid, inst, comp) in winners:
        if len(rec['promoted']) >= room:
            outcomes['no_room_left'] += 1
            rec['not_promoted'].append({'instance_id': inst['instance_id'], 'panorama_spine': sid, 'component': comp, 'iou': iou,
                                        'reason': (f'free spine {sid} (placement IoU {iou:.2f}), but the panorama count {pan_n} leaves room for {room} more line(s) than the '
                                                   f'{lines_before} counted and {room} stronger placements were promoted')})
            continue
        fc = frame_check(oid, counted_obs) if frame_check else None
        if fc and fc['outcome'] == 'occupied':
            outcomes['counted_in_a_neighbour_frame'] += 1
            rec['not_promoted'].append({'instance_id': inst['instance_id'], 'panorama_spine': sid, 'component': comp, 'iou': iou, 'frame_check': fc,
                                        'reason': (f'free spine {sid} in piece {comp}, but registering {oid.split("-o")[0]} -> {fc["frame_id"]} ({fc["inliers"]} inliers) puts it on '
                                                   f'counted line {fc["instance_id"]} (across overlap {fc["across_overlap"]:.2f})')})
            continue
        nb = '' if fc is None else ('; no counted line there in ' + str(len(fc.get('registered') or [])) + ' registered neighbour frame(s)' if fc['outcome'] == 'free'
                                    else '; no neighbour frame registered')
        ev = f'promoted: distinct detected spine {sid} in piece {comp}, no counted line on it (sighting {oid}, placement IoU {iou:.2f}{nb})'
        inst['tier'], inst['tier_reason'] = PROMOTED_TIER, ev
        inst['promotion'] = {'panorama_spine': sid, 'component': comp, 'observation_id': oid, 'iou': iou, 'evidence': ev, **({'frame_check': fc} if fc else {})}
        rec['promoted'].append({'instance_id': inst['instance_id'], **inst['promotion']})
        outcomes['promoted'] += 1
    n = len(rec['promoted'])
    rec.update(promotions=n, lines_after=lines_before + n, abs_diff_after=abs(lines_before + n - pan_n), outcomes=dict(outcomes), spines_total=len(spines),
               spines_occupied_by_counted_lines=len(occupied), sightings_in_normal_pieces=len(sightings), sightings_on_a_spine=len(placed))
    return rec


def build(inventory_data: dict[str, Any], frames: dict[str, dict[str, Any]], segments: dict[str, int], *, panorama_count: dict[str, Any] | None = None,
          panorama_eligible: bool = False, panorama_reason: str | None = None, cross: dict[str, Any] | None = None, run_id: str | None = None,
          inventory_sha256: str | None = None, observations: list[dict[str, Any]] | None = None, placed_in_normal_piece: set[str] | None = None,
          readings: list[dict[str, Any]] | None = None, pan_checks: dict[str, dict[str, Any]] | None = None, panorama_rule: bool | str = True,
          oracle: list[dict[str, Any]] | None = None, piece_sightings: list[dict[str, Any]] | None = None, reconcile_with_panorama: bool = True,
          reconcile_frame_check=None, room_pan: dict[str, Any] | None = None, plane: dict[str, Any] | None = None, short_gap: dict[str, Any] | None = None,
          revisit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pure: track groups + the two evidence kinds (+ the opt-in ``oracle`` links of pipeline.pair_oracle, model-read) -> the instances artifact. Every union passes the same-frame exclusion; refused unions are listed.
    ``observations`` (candidates) enable the far-view rule: an instance whose EVERY sighting is a far view (pipeline.panorama_count.FAR_FRACTION of the run's median
    native spine) is tier 'unconfirmed' unless it is merged with a close sighting. ``placed_in_normal_piece``: group ids placed on a normal (detector-counted) panorama
    piece, the lower endpoint's panorama clause. ``room_pan``: pipeline.room_pan.classify_run of the run; an instance whose every sighting is in a
    room-pan frame is tier 'unconfirmed' (reason 'room-pan view only')."""
    books = inventory_data.get('books') or []
    pan_frames = set((room_pan or {}).get('room_pan_frames') or [])
    by_id = {b['instance_id']: b for b in books}
    far_obs, median_px = far_sightings(books, observations)
    cover_obs = cover_view_sightings(observations)
    placed = placed_in_normal_piece if (placed_in_normal_piece and panorama_eligible) else set()
    texts = group_texts(books, readings)
    reading_vetoes = 0
    frames_of = {b['instance_id']: {o.split('-o')[0] for o in b['observation_ids']} for b in books}
    parent = {b['instance_id']: b['instance_id'] for b in books}
    members: dict[str, set[str]] = {b['instance_id']: {b['instance_id']} for b in books}
    evidence: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    rejected: list[dict[str, Any]] = []
    merged_by = collections.Counter()

    nonlocal_counter = {'reading_vetoes': 0}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(ga, gb, ev):
        ra, rb = find(ga), find(gb)
        if ra == rb:
            return True
        fa = set().union(*(frames_of[m] for m in members[ra]))
        fb = set().union(*(frames_of[m] for m in members[rb]))
        if fa & fb:
            rejected.append({**ev, 'groups': [ga, gb], 'reason': 'same-frame exclusion: the two groups are seen together in ' + ', '.join(sorted(fa & fb)[:3])})
            return False
        # reading-contradiction veto: readable texts on both sides that disagree beyond OCR noise keep the groups apart (a title never creates a link, it can only refuse one)
        ta = [texts[m] for m in members[ra] if m in texts]
        tb = [texts[m] for m in members[rb] if m in texts]
        if ta and tb:
            best_a = max(ta, key=lambda t: (t['read_confidence'] or 0.0, len(t['text'])))
            best_b = max(tb, key=lambda t: (t['read_confidence'] or 0.0, len(t['text'])))
            ov = text_overlap(best_a['text'], best_b['text'])
            if ov is not None and ov < READING_VETO['min_overlap']:
                nonlocal_counter['reading_vetoes'] += 1
                rejected.append({**ev, 'groups': [ga, gb], 'reason': f'reading contradiction: the two sides read differently (token overlap {ov:.2f} < {READING_VETO["min_overlap"]})',
                                 'texts': [best_a['text'], best_b['text']], 'text_observation_ids': [best_a['observation_id'], best_b['observation_id']], 'token_overlap': round(ov, 3)})
                return False
        keep, drop = sorted([ra, rb], key=lambda r: min(by_id[m]['observation_ids'][0] for m in members[r]))
        parent[drop] = keep
        members[keep] |= members.pop(drop)
        evidence[keep].extend(evidence.pop(drop, []))
        evidence[keep].append({**ev, 'groups': [ga, gb]})
        merged_by[ev['kind']] += 1
        return True
    pan_lines = (panorama_count or {}).get('merge_evidence') or [] if panorama_eligible else []
    pan_used = 0
    pan_outcomes = collections.Counter()
    rule = ('registration_decides' if panorama_rule is True else panorama_rule) if panorama_rule else None
    assert rule is None or rule in PANORAMA_RULES, panorama_rule
    for spine, anchor, other in panorama_pairs(pan_lines, by_id):
        iou = {g['id']: g.get('iou') for g in spine['groups']}
        ev = {'kind': KIND_PANORAMA, 'panorama_spine': spine.get('panorama_spine'), 'component': spine.get('component'), 'box_xyxy': spine.get('box_xyxy'),
              'confidence': spine.get('confidence'), 'iou': {anchor: iou.get(anchor), other: iou.get(other)}}
        # Acceptance rule: (1) both placements overlap the detected spine by >= min_spine_iou; (2) a direct registration of the two groups' closest sightings,
        # when one registers, must put A's spine on B and on no other detected spine. Same-frame exclusion and the reading veto follow in union().
        low = min(iou.get(anchor) or 0.0, iou.get(other) or 0.0)
        if not rule:
            pan_used += union(anchor, other, ev)
            continue
        fc = None if pan_checks is None else pan_checks.get(f'{anchor}|{other}')
        # Registration decides: a registered frame pair looks at the two sightings directly, the placement IoU is a proxy on the stitched piece: an agreeing registration
        # overrules a loose placement (the user's answers: 3 of 3 such pairs the same book), and the placement decides only where no frame pair registered.
        decided = rule == 'registration_decides' and fc is not None and fc.get('outcome') in ('agree', 'contradicts')
        if low < PANORAMA_ACCEPT['min_spine_iou'] and not decided:
            pan_outcomes['placement_too_loose'] += 1
            rejected.append({**ev, 'groups': [anchor, other], 'reason': f'panorama placement too loose: a group overlaps the detected spine with IoU {low:.2f} '
                             f'(< {PANORAMA_ACCEPT["min_spine_iou"]}, the IoU at which two detector boxes are one spine)'})
            continue
        if fc is not None:
            ev['frame_check'] = fc
            if fc['outcome'] == 'contradicts':
                pan_outcomes['frame_contradicts'] += 1
                where = 'on no detected spine' if not fc.get('best_spine') or fc['best_overlap'] < PANORAMA_ACCEPT['min_across_overlap'] else f'on another detected spine ({fc["best_spine"]}, overlap {fc["best_overlap"]:.2f})'
                rejected.append({**ev, 'groups': [anchor, other], 'reason': f'frame check contradicts the panorama: registering {fc["frame_ids"][0]} -> {fc["frame_ids"][1]} ({fc["inliers"]} inliers) '
                                 f'puts {anchor}\'s spine {where}; it covers {other}\'s by {fc["across_overlap"]:.2f} (< {PANORAMA_ACCEPT["min_across_overlap"]} or not the best)'})
                continue
        ev['acceptance'] = {'min_spine_iou': round(low, 3), 'frame_check': (fc or {}).get('outcome', 'not run'), 'rule': rule}
        if low < PANORAMA_ACCEPT['min_spine_iou']:
            ev['acceptance']['placement'] = f'below {PANORAMA_ACCEPT["min_spine_iou"]}: the registered frame pair decides'
        ok = union(anchor, other, ev)
        pan_used += ok
        pan_outcomes[(('accepted_frame_' + ev['acceptance']['frame_check']) + ('_loose_placement' if low < PANORAMA_ACCEPT['min_spine_iou'] else '')) if ok else 'refused_in_union'] += 1
    for lk in (cross or {}).get('links') or []:
        ga, gb = lk['groups']
        if ga not in by_id or gb not in by_id:
            continue
        ev = {k: lk[k] for k in ('kind', 'observation_ids', 'frame_ids', 'segments', 'gap_s', 'inliers', 'appearance_score', 'center_distance_spine_widths', 'score', 'transition') if k in lk}
        union(ga, gb, ev)
    short_used = 0
    for lk in (short_gap or {}).get('links') or []:   # short_gap_registration (two detection frames <= 3 s apart, any segment); same unions and vetoes
        ga, gb = lk['groups']
        if ga in by_id and gb in by_id and find(ga) != find(gb):
            ok = union(ga, gb, {k: v for k, v in lk.items() if k != 'groups'})
            lk['outcome'] = 'union' if ok else 'refused_in_union'
            short_used += ok
        elif ga in by_id and gb in by_id:
            lk['outcome'] = 'already_one_instance'
    revisit_used = 0
    for lk in (revisit or {}).get('links') or []:     # revisit_registration (detection frames 3-20 s apart + layout check); same unions and vetoes
        ga, gb = lk['groups']
        if ga in by_id and gb in by_id and find(ga) != find(gb):
            ok = union(ga, gb, {k: v for k, v in lk.items() if k != 'groups'})
            lk['outcome'] = 'union' if ok else 'refused_in_union'
            revisit_used += ok
        elif ga in by_id and gb in by_id:
            lk['outcome'] = 'already_one_instance'
    plane_used = 0
    for lk in (plane or {}).get('links') or []:   # shelf_plane_position (merged 3D map); same unions: same-frame exclusion and the reading veto apply
        ga, gb = lk['groups']
        if ga in by_id and gb in by_id and find(ga) != find(gb):
            ok = union(ga, gb, {k: v for k, v in lk.items() if k != 'groups'})
            lk['outcome'] = 'union' if ok else 'refused_in_union'
            plane_used += ok
        elif ga in by_id and gb in by_id:
            lk['outcome'] = 'already_one_instance'
    for lk in oracle or []:   # opt-in hosted pair adjudication: same unions, so the same-frame exclusion and the reading veto apply
        ga, gb = lk['groups']
        if ga in by_id and gb in by_id:
            union(ga, gb, {k: v for k, v in lk.items() if k != 'groups'})
    sharp = lambda f: float((frames.get(f, {}).get('quality') or {}).get('sharpness') or 0.0)
    out = []
    for root, mem in members.items():
        ms = sorted(mem, key=lambda m: by_id[m]['observation_ids'][0])
        obs = sorted({o for m in ms for o in by_id[m]['observation_ids']})
        segs = sorted({segments.get(o.split('-o')[0], 0) for o in obs})
        rep = max(ms, key=lambda m: (sharp(by_id[m].get('frame_ref')), len(by_id[m]['observation_ids'])))
        tier = 'confident' if len(ms) > 1 or len(obs) > 1 else by_id[ms[0]].get('count_tier') or 'confident'
        far_only = bool(far_obs) and all(o in far_obs for o in obs)
        tier_reason = None
        if far_only and tier != 'unconfirmed':
            tier, tier_reason = 'unconfirmed', f'far view only: every sighting shows the spine under {FAR_FRACTION:.0%} of the run median native spine ({median_px:.0f} px); the close pass counts it'
        # Cover-view rule: EVERY sighting (all members) is a cover/flat view -> listed, not a line; one spine-view sighting anywhere in the instance keeps it
        cover_only = bool(cover_obs) and all(o in cover_obs for o in obs)
        if cover_only and tier != 'unconfirmed':
            tier, tier_reason = 'unconfirmed', (f'cover or flat view only: every sighting\'s polygon is under {COVER_VIEW_MAX_ASPECT:.1f}:1 long/short (a spine seen whole is longer); '
                                                'a cover face, a second box on a book counted from its spine, or a decor object; listed, not counted')
        # Room-pan rule: EVERY sighting is in a frame of the room pan (floor / ceiling / side-wall pan of the same sweep, pipeline.room_pan): not a line
        room_pan_only = bool(pan_frames) and all(o.split('-o')[0] in pan_frames for o in obs)
        if room_pan_only and tier != 'unconfirmed':
            tier, tier_reason = 'unconfirmed', (f'{ROOM_PAN_REASON}: every sighting is in a frame of the room pan (frames the sweep spent on the floor, ceiling and walls, '
                                                'pipeline.room_pan); the close shelf pass counts the shelf; listed, not counted')
        in_pan = any(m in placed for m in ms)
        out.append({'instance_id': ms[0], 'member_group_ids': ms, 'observation_ids': obs, 'observation_segments': [segments.get(o.split('-o')[0], 0) for o in obs],
                    'sightings': len(obs), 'segments': segs, 'tier': tier, 'tier_reason': tier_reason, 'far_only': far_only, 'cover_only': cover_only, 'room_pan_only': room_pan_only, 'in_normal_piece': in_pan,
                    'lower_support': len(ms) > 1 or len(segs) > 1 or (in_pan and len(obs) > 1),
                    'representative_group_id': rep, 'evidence': evidence.get(root, []), 'evidence_kinds': sorted({e['kind'] for e in evidence.get(root, [])})})
    out.sort(key=lambda i: i['observation_ids'][0])
    n_groups, n_inst = len(books), len(out)
    counts = inventory_data.get('counts') or {}
    track_upper = (counts.get('count_range') or [None, None])[1]
    lines = [i for i in out if i['tier'] != 'unconfirmed']
    lower = sum(1 for i in lines if i['lower_support'])
    best = len(lines)
    upper = max(best, (track_upper if track_upper is not None else n_groups) - (n_groups - n_inst))
    if reconcile_with_panorama:
        recon = reconcile(out, by_id, panorama_count, piece_sightings, panorama_eligible=panorama_eligible, panorama_reason=panorama_reason, lower=lower, upper=upper, lines_before=best,
                          frame_check=reconcile_frame_check)
        best = sum(1 for i in out if i['tier'] != 'unconfirmed')
        upper = max(best, upper)
    else:
        recon = {'eligible': False, 'reason': 'reconciliation switched off (--no-reconcile)', 'promotions': 0, 'lines_before': best, 'lines_after': best, 'promoted': [], 'not_promoted': []}
    cx = cross or {}
    frag = {'segments': (max(segments.values()) + 1) if segments else 1, 'transitions': cx.get('transitions_total', 0), 'transitions_registered': cx.get('transitions_registered', 0),
            'transitions_with_links': cx.get('transitions_with_links', 0), 'duplicate_links_found': len(cx.get('links') or []), 'near_tie_vetoes': len(cx.get('rejected') or []),
            'same_frame_refusals': len(rejected), 'association_reliability': _reliability_of(cx.get('transitions_total', 0), cx.get('transitions_registered', 0)),
            'rule': 'association reliability: high when no tracking break or >= 80% of the breaks registered across (SIFT/RANSAC), medium >= 50%, else low; the count reliability is the weaker of the capture and association readings'}
    return {'schema_version': SCHEMA, 'artifact_type': 'instances', 'run_id': run_id, 'inventory_sha256': inventory_sha256, 'computed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'method': ('track groups merged into physical instances by (1) panorama overlap (panorama_count merge_evidence, when the panorama estimator is eligible) and (2) bounded '
                       'cross-segment frame registration around every tracking-segment transition, (3) short-gap frame registration (any two detection frames <= 3 s apart), (4) revisit frame registration (3-20 s apart, with the layout check, when enabled); same-frame exclusion at every union; no OCR/row/appearance-only/loop-closure merging'),
            'panorama': {'eligible': bool(panorama_eligible), 'reason': panorama_reason, 'panorama_run': (panorama_count or {}).get('panorama_run'), 'spines_with_several_groups': len(pan_lines),
                         'unions': pan_used},
            'cross_segment': {k: cx.get(k) for k in ('frames_used', 'pairs_tested', 'pairs_registered', 'transitions_total', 'transitions_registered', 'transitions_with_links', 'scale', 'config', 'seconds')} | {
                'links': cx.get('links') or [], 'rejected': cx.get('rejected') or [], 'transitions': cx.get('transitions') or []},
            'short_gap': None if short_gap is None else {**{k: v for k, v in short_gap.items() if k not in ('links', 'rejected')}, 'unions': short_used,
                                                         'links': short_gap.get('links') or [], 'rejected': short_gap.get('rejected') or []},
            'revisit': None if revisit is None else {**{k: v for k, v in revisit.items() if k not in ('links', 'rejected')}, 'unions': revisit_used,
                                                     'links': revisit.get('links') or [], 'rejected': revisit.get('rejected') or []},
            'shelf_plane': None if plane is None else {**{k: v for k, v in plane.items() if k not in ('links', 'rejected')}, 'unions': plane_used,
                                                        'links': plane.get('links') or [], 'rejected': plane.get('rejected') or []},
            'instances': out, 'rejected_unions': rejected, 'reconciliation': recon,
            'stats': {'track_groups': n_groups, 'instances': n_inst, 'merged_groups': n_groups - n_inst, 'merged_instances': sum(1 for i in out if len(i['member_group_ids']) > 1),
                      'merged_by': dict(merged_by), 'unconfirmed_instances': n_inst - best, 'far_only_instances': sum(1 for i in out if i['far_only']),
                      'far_only_demoted': sum(1 for i in out if str(i['tier_reason'] or '').startswith('far view only')),
                      'cover_only_instances': sum(1 for i in out if i['cover_only']), 'cover_only_demoted': sum(1 for i in out if str(i['tier_reason'] or '').startswith('cover or flat view only')),
                      'cover_view_max_aspect': COVER_VIEW_MAX_ASPECT,
                      'room_pan_only_instances': sum(1 for i in out if i['room_pan_only']), 'room_pan_demoted': sum(1 for i in out if str(i['tier_reason'] or '').startswith(ROOM_PAN_REASON)),
                      'room_pan_frames': len(pan_frames), 'in_normal_piece': sum(1 for i in out if i['in_normal_piece']),
                      'panorama_acceptance': {'rule': rule or 'off (no acceptance rule: every placement on the spine is a union)', 'outcomes': dict(pan_outcomes),
                                              'constants': {**PANORAMA_ACCEPT, 'frame_pairs': PANORAMA_FRAME_PAIRS}, 'frame_checks_run': pan_checks is not None},
                      'reading_vetoes': nonlocal_counter['reading_vetoes'], 'groups_with_readable_text': len(texts), 'reading_veto': READING_VETO,
                      'median_native_spine_px': None if median_px is None else round(median_px, 1), 'far_fraction': FAR_FRACTION,
                      'promoted_by_reconciliation': recon.get('promotions', 0)},
            'count': {'lower': lower, 'best': best, 'upper': upper, 'track_upper': track_upper,
                      'basis': ('best guess = instances that are not unconfirmed (unconfirmed = a single sighting the capture cannot confirm, or an instance seen only in far views: every '
                                f'sighting under {FAR_FRACTION:.0%} of the run median native spine, the close pass counts it; or an instance with no spine-view sighting: every polygon under '
                                f'{COVER_VIEW_MAX_ASPECT:.1f}:1 long/short, a cover face or a non-book); lower = instances with sightings in >= 2 tracking segments, '
                                'or merged from >= 2 track groups, or placed with >= 2 sightings on a normal (detector-counted) piece of the 2D panorama when the panorama estimator is '
                                'eligible (a single-segment multi-sighting group alone is NOT evidence of a globally unique book); upper = the track upper bound (all groups after unconfirmed '
                                'single sightings that land on another group by tracked position are merged) minus the groups merged here. Covers association only: spines the detector missed '
                                'or fused are outside the range. Reconciliation: an unconfirmed single sighting on a detected panorama spine that no counted line '
                                'occupies is a line (tier likely, evidence in its issues), only when the panorama count lies inside [lower, upper] computed before it.')},
            'fragmentation': frag,
            'room_pan': None if room_pan is None else {**{k: room_pan.get(k) for k in ('room_pan_frame_count', 'frames_total', 'segments', 'signals', 'uses_3d', 'recon3d', 'rule')},
                                                        'demoted_instance_ids': [i['instance_id'] for i in out if str(i['tier_reason'] or '').startswith(ROOM_PAN_REASON)]},
            'limits': ['Duplicates across pieces that did not register to each other, and across transitions where no frame pair registered, are not merged.',
                       'Panorama overlap evidence inherits panorama_count\'s placement (best spine per group, IoU >= 0.05 or centre inside); no per-book truth exists.',
                       'The gates (0.5 appearance, 0.5 spine widths, 30 inliers, 0.07 near-tie margin) are the production association gates, not fitted to a hand count.',
                       'The far-view rule reuses pipeline.panorama_count.FAR_FRACTION (0.5 of the run median native spine); a shelf filmed only from far away is listed apart, not counted.',
                       f'The cover-view rule lists apart an instance whose every sighting is under {COVER_VIEW_MAX_ASPECT:.1f}:1 long/short (cover or flat view); a real '
                       'spine seen only through a gap (occluded by a plant) or a book lying face-up is lost from the count with it; a cover seen obliquely looks spine-shaped and is not caught.',
                       'Room-pan rule: an instance seen only in frames of the room pan (pipeline.room_pan: a run of non-shelf frames with >= 3 frames '
                       'off the shelf over >= 4 s) is listed apart; a book seen ONLY during the pan (never in the close pass) is lost from the count with it.',
                       'Spine readings only VETO a merge (both sides readable, token overlap < 0.3); they never create one. Unreadable spines get no protection from this guard.',
                       'Reconciliation: a promoted single sighting rests on one detector box in one piece; a book counted only from another piece that '
                       'did not register to this one is not seen on the spine, so a duplicate across unregistered pieces can be promoted (the panorama count has the same limit).',
                       'Panorama overlap unions: where a frame pair of the two groups registers it decides (A\'s spine must land on B); otherwise both placements must '
                       'overlap the detected spine by IoU >= 0.35. Checked against 40 same/different answers of one human, '
                       'not against truth; it was also measured against a blinded model adjudicator (model-read).']}


def sha256_of(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def load(run_dir: str | Path) -> dict[str, Any] | None:
    """The run's instances.json when it is there and belongs to this run (cached by mtime: the Sweep page rebuilds the packet data on every poll)."""
    path = Path(run_dir) / ARTIFACT
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _CACHE.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        art = json.loads(path.read_text())
        ok = isinstance(art, dict) and art.get('artifact_type') == 'instances' and art.get('run_id') == Path(run_dir).name and isinstance(art.get('instances'), list)
    except (OSError, ValueError):
        art, ok = None, False
    _CACHE[str(path)] = (mtime, art if ok else None)
    return art if ok else None


# ------------------------------------------------------------------------------------------ packet lines = instances
PRICE_KEYS = ('replacement_cost', 'used_value', 'price_basis', 'equivalent_isbn', 'equivalent_edition', 'price_basis_note', 'pricing_issues', 'pricing_null_reasons',
              'pricing_attempted', 'second_market')
IDENTITY_KEYS = ('title', 'author', 'edition', 'isbn', 'publisher', 'id_confidence', 'identity_source', 'identity_source_url', 'identity_matching_evidence', 'language',
                 'identity_decision', 'visible_text')


def expand(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merged lines back to their member lines (a merged line stores them in ``merged_members``); plain lines pass through. Makes ``apply`` idempotent."""
    out = []
    for line in lines:
        if line.get('merged_members'):
            out.extend(copy.deepcopy(m) for m in line['merged_members'])
        else:
            out.append(line)
    return out


def merge_lines(members: list[dict[str, Any]], inst: dict[str, Any]) -> dict[str, Any]:
    """One packet line from the member lines of an instance (reading union, one resolution; prices and measurement from the member that has them)."""
    by_id = {m['id']: m for m in members}
    rep = by_id.get(inst.get('representative_group_id')) or max(members, key=lambda m: len((m.get('evidence') or {}).get('observation_ids') or []))
    out = copy.deepcopy(rep)
    out.pop('merged_members', None)
    out['id'] = inst['instance_id']
    out['member_group_ids'] = [m['id'] for m in members]
    out['merged_members'] = [copy.deepcopy({k: v for k, v in m.items() if k != 'merged_members'}) for m in members]
    obs = sorted({o for m in members for o in ((m.get('evidence') or {}).get('observation_ids') or [])})
    out.setdefault('evidence', {})['observation_ids'] = obs
    out['sightings'] = len(obs)
    out['merge_evidence'] = inst.get('evidence') or []
    out['merge_evidence_kinds'] = inst.get('evidence_kinds') or sorted({e.get('kind') for e in out['merge_evidence']})
    out['count_tier'] = inst.get('tier') or ('confident' if len(members) > 1 or len(obs) > 1 else rep.get('count_tier'))
    out['uncertain_association'] = any(m.get('uncertain_association') for m in members)
    out['user_corrections'] = [c for m in members for c in (m.get('user_corrections') or [])]
    out['reading_evidence'] = [r for m in members for r in (m.get('reading_evidence') or [])]
    issues = []
    for m in members:
        for i in m.get('issues') or []:
            if i not in issues:
                issues.append(i)
    # identity: the accepted catalogue works of the members must agree; otherwise the identity is abstained (never a guess)
    accepted = [m for m in members if m.get('status') == 'identified' and m.get('title')]
    titles = {(str(m.get('title') or '').casefold(), m.get('identity_source')) for m in accepted}
    if rep.get('status') == 'needs_appraisal':
        pass
    elif len(titles) == 1:
        src = accepted[0] if rep not in accepted else rep
        for k in IDENTITY_KEYS:
            out[k] = src.get(k)
        out['status'] = 'identified'
    elif len(titles) > 1:
        for k in IDENTITY_KEYS:
            out[k] = None
        for k in PRICE_KEYS:   # a price belongs to an identity; an abstained identity carries no member's price
            out.pop(k, None)
        out['replacement_cost'] = out['used_value'] = None
        out['identity_decision'] = 'ambiguous'
        out['status'] = 'unidentified'
        issues.append('Members of this instance resolved to different catalogue works (' + '; '.join(sorted(str(m.get('title')) for m in accepted)[:3]) + '); identity abstained')
    else:
        longest = max(members, key=lambda m: len(m.get('visible_text') or ''))
        out['visible_text'] = longest.get('visible_text')
        out['language'] = out.get('language') or longest.get('language')
        out['status'] = 'unidentified' if rep.get('status') != 'needs_appraisal' else rep['status']
    if out.get('replacement_cost') is None and out.get('used_value') is None and out.get('status') != 'needs_appraisal':
        donor = next((m for m in members if (m.get('replacement_cost') or m.get('used_value')) and m.get('status') != 'needs_appraisal'
                      and (str(m.get('title') or '').casefold() == str(out.get('title') or '').casefold())), None)
        if donor is not None:
            for k in PRICE_KEYS:
                if k in donor:
                    out[k] = copy.deepcopy(donor[k])
    if out.get('spine_height_cm') is None:
        donor = next((m for m in members if m.get('spine_height_cm') is not None), None)
        if donor is not None:
            out['spine_height_cm'], out['spine_thickness_cm'], out['measurement'] = donor['spine_height_cm'], donor.get('spine_thickness_cm'), copy.deepcopy(donor.get('measurement'))
    if KIND_ORACLE in out['merge_evidence_kinds'] and ORACLE_LABEL not in issues:
        issues.append(ORACLE_LABEL)
    if len(members) > 1:
        issues.append(f"Merged from {len(members)} track groups by {', '.join(out['merge_evidence_kinds']).replace('_', ' ')} ({', '.join(out['member_group_ids'][:6])}{'...' if len(members) > 6 else ''})")
    if inst.get('tier') == 'unconfirmed' and inst.get('tier_reason') and inst['tier_reason'] not in issues:
        issues.append(inst['tier_reason'])
    out['issues'] = issues
    return out


def apply(p: dict[str, Any], art: dict[str, Any] | None, ledger: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Patch packet ``data`` in place so that the book lines are physical instances (idempotent). ``ledger``: the corrections ledger (``split`` entries undo
    a merge for the member group they target; an ``exclude`` of any member excludes the whole instance). Returns a summary."""
    if not art:
        return {'applied': False, 'reason': 'no instances artifact'}
    from pipeline import packet as P
    t, c = p['totals'], p.setdefault('count', {})
    lines = expand(list(p.get('books') or [])) + expand(list(p.get('unconfirmed_books') or []))
    by_id = {ln['id']: ln for ln in lines}
    split_obs = {o for e in (ledger or []) if e.get('active', True) and e.get('kind') == 'split' for o in ((e.get('target') or {}).get('observation_ids') or [])}
    excluded = list(p.get('excluded_by_user') or [])
    excluded_ids = {x['instance_id'] for x in excluded}
    instances: list[dict[str, Any]] = []
    split_log = []
    for inst in art['instances']:
        mem = [m for m in inst['member_group_ids'] if m in by_id or m in excluded_ids]
        if not mem:
            continue
        if split_obs:
            keep, apart = [], []
            for m in mem:
                obs = set((by_id.get(m) or {}).get('evidence', {}).get('observation_ids') or []) or {o for x in excluded if x['instance_id'] == m for o in x.get('observation_ids') or []}
                (apart if obs & split_obs else keep).append(m)
            for m in apart:
                split_log.append({'instance_id': inst['instance_id'], 'member_group_id': m})
                instances.append({**inst, 'instance_id': m, 'member_group_ids': [m], 'representative_group_id': m, 'evidence': [], 'evidence_kinds': [], 'tier': (by_id.get(m) or {}).get('count_tier') or 'confident'})
            if not keep:
                continue
            mem = keep
            inst = {**inst, 'member_group_ids': mem, 'instance_id': min(mem) if inst['instance_id'] not in mem else inst['instance_id'],
                    'representative_group_id': inst.get('representative_group_id') if inst.get('representative_group_id') in mem else mem[0],
                    'evidence': [e for e in inst.get('evidence') or [] if set(e.get('groups') or []) <= set(mem)], 'evidence_kinds': sorted({e['kind'] for e in inst.get('evidence') or [] if set(e.get('groups') or []) <= set(mem)}),
                    'tier': inst['tier'] if len(mem) > 1 else (by_id.get(mem[0]) or {}).get('count_tier') or inst['tier']}
        instances.append(inst)
    covered = {m for inst in instances for m in inst['member_group_ids']}
    for ln in lines:                                                                    # groups unknown to the artifact (older artifact, re-merged inventory): their own instance
        if ln['id'] not in covered:
            instances.append({'instance_id': ln['id'], 'member_group_ids': [ln['id']], 'representative_group_id': ln['id'], 'evidence': [], 'evidence_kinds': [], 'tier': ln.get('count_tier') or 'confident'})
    # an exclusion of any member excludes the instance (observation-id targets resolve to one group; the siblings follow it)
    sibling_excluded = 0
    for inst in instances:
        hit = [x for x in excluded if x['instance_id'] in inst['member_group_ids']]
        if not hit:
            continue
        for m in inst['member_group_ids']:
            ln = by_id.get(m)
            if ln is None or m in excluded_ids:
                continue
            excluded.append({'instance_id': m, 'kind_before': 'book', 'observation_ids': (ln.get('evidence') or {}).get('observation_ids') or [], 'frame_ref': ln.get('frame_ref'),
                             'bbox_xyxy': (ln.get('evidence') or {}).get('bbox_xyxy'), 'correction_id': hit[0].get('correction_id'), 'source': hit[0].get('source'),
                             'reason': f"member of excluded instance {inst['instance_id']}: " + str(hit[0].get('reason') or 'excluded by user')})
            excluded_ids.add(m)
            sibling_excluded += 1
    books, unconfirmed = [], []
    merged_line_ids: dict[str, str] = {}
    for inst in instances:
        mem = [by_id[m] for m in inst['member_group_ids'] if m in by_id and m not in excluded_ids]
        if not mem:
            continue
        if len(mem) == 1 and len(inst['member_group_ids']) == 1:
            line = copy.deepcopy(mem[0])
            line.pop('merged_members', None)
            line['member_group_ids'] = [mem[0]['id']]
            line['merge_evidence'], line['merge_evidence_kinds'] = [], []
            line['sightings'] = len((line.get('evidence') or {}).get('observation_ids') or [])
            prom = inst.get('promotion')
            if line.get('promotion') and not prom:          # an earlier attach promoted it, this artifact does not: back to its own tier
                line['count_tier'] = line.pop('promotion').get('tier_before') or 'unconfirmed'
                line['issues'] = [i for i in line.get('issues') or [] if not str(i).startswith('promoted: ')]
            if prom:
                line['promotion'] = {**prom, 'tier_before': (line.get('promotion') or {}).get('tier_before') or line.get('count_tier')}
                line['count_tier'] = inst.get('tier') or PROMOTED_TIER
                line['issues'] = [i for i in line.get('issues') or [] if not str(i).startswith('promoted: ')] + [prom['evidence']]
            if inst.get('tier_reason') and inst.get('tier') == 'unconfirmed':
                line['count_tier'] = 'unconfirmed'
                line['issues'] = [i for i in line.get('issues') or [] if not str(i).startswith(('far view only', 'cover or flat view only', ROOM_PAN_REASON))] + [inst['tier_reason']]
        else:
            line = merge_lines(mem, inst)
        line['lower_support'] = bool(inst.get('lower_support', len(line['member_group_ids']) > 1))
        for m in mem:
            merged_line_ids[m['id']] = line['id']
        (unconfirmed if line.get('count_tier') == 'unconfirmed' and not line.get('user_corrections') else books).append(line)
    books.sort(key=lambda b: (b.get('evidence') or {}).get('observation_ids', [b['id']])[0] if (b.get('evidence') or {}).get('observation_ids') else b['id'])
    p['books'], p['unconfirmed_books'], p['excluded_by_user'] = books, unconfirmed, excluded
    items = p.get('items') or []
    fresh = P.compute_totals(books, items, {}, len(excluded), p.get('pricing') is not None)
    for k in ('book_count', 'books_identified', 'books_unidentified', 'books_needs_appraisal', 'books_identified_by_user_statement', 'user_excluded_count', 'books_replacement_cost',
              'books_used_value', 'excluded_from_totals', 'item_count', 'items_needs_appraisal', 'ambiguous_count'):
        t[k] = fresh[k]
    if 'second_market' in fresh:
        t['second_market'] = fresh['second_market']
    else:
        t.pop('second_market', None)
    if 'spine_dimensions_measured' in t:
        t['spine_dimensions_measured'] = sum(b.get('spine_height_cm') is not None for b in books)
        t['spine_dimensions_abstained'] = sum((b.get('measurement') or {}).get('status') == 'abstained' for b in books)
    track_groups = sum(len(b['member_group_ids']) for b in books + unconfirmed)
    st, cnt, frag = art.get('stats') or {}, art.get('count') or {}, art.get('fragmentation') or {}
    n = len(books)
    merged_groups = track_groups - len(books) - len(unconfirmed)
    lower = sum(1 for b in books if b.get('lower_support') or len(b['member_group_ids']) >= 2 or len(_segments_of(b, art)) >= 2)
    track_upper = cnt.get('track_upper')
    base_upper = (track_upper if track_upper is not None else st.get('track_groups') or track_groups)
    upper = max(n, base_upper - merged_groups - len(excluded))
    lower = min(lower, n)
    capture_rel = ((p.get('provenance') or {}).get('capture_quality') or {}).get('count_reliability') or t.get('count_reliability')
    rel = weaker(capture_rel, frag.get('association_reliability')) if frag else capture_rel
    merged_by = st.get('merged_by') or {}
    note = NOTE.format(lines=n, groups=track_groups, pan=merged_by.get(KIND_PANORAMA, 0), cross=merged_by.get(KIND_CROSS, 0))
    if merged_by.get(KIND_PLANE):
        note += f'; {merged_by[KIND_PLANE]} more by shelf-plane position in the merged 3D map'
    t.update(book_count=n, book_count_track=track_groups, count_estimator='instances', book_count_lower=lower, book_count_upper=upper, book_count_range=[lower, upper],
             confident_groups=lower, count_reliability=rel, count_reliability_capture=capture_rel, count_reliability_association=frag.get('association_reliability'),
             instances_merged_groups=merged_groups, instances_merged_by=merged_by, count_range_basis=cnt.get('basis'))
    t.setdefault('notes', {})['book_count'] = 'best guess of physical books = instances (track groups merged by panorama overlap and cross-segment registration); ' + note
    c.update(best_guess=n, book_groups=n, estimator='instances', book_count_lower=lower, book_count_upper=upper, book_count_range=[lower, upper], count_range_basis=cnt.get('basis'),
             count_reliability=rel, confident_groups=lower,
             uncertainty_band={'low': lower, 'best': n, 'high': upper, 'note': 'coherent low <= best <= high from the instance layer (count.instances); the near-tie and tiered bounds are diagnostics only'},
             instances={'lines': n, 'unconfirmed_lines': len(unconfirmed), 'track_groups': track_groups, 'merged_groups': merged_groups, 'merged_by': merged_by, 'merged_lines': sum(1 for b in books if len(b['member_group_ids']) > 1),
                        'panorama': art.get('panorama'), 'cross_segment': {k: (art.get('cross_segment') or {}).get(k) for k in ('frames_used', 'pairs_tested', 'pairs_registered', 'transitions_total', 'transitions_registered', 'transitions_with_links', 'seconds')},
                        'fragmentation': frag, 'rejected_unions': len(art.get('rejected_unions') or []), 'splits': split_log, 'sibling_exclusions': sibling_excluded, 'artifact': ARTIFACT, 'note': note,
                        'far_only_unconfirmed': sum(1 for b in unconfirmed if any(str(i).startswith('far view only') for i in b.get('issues') or [])), 'far_fraction': st.get('far_fraction'),
                        'cover_view_unconfirmed': sum(1 for b in unconfirmed if any(str(i).startswith('cover or flat view only') for i in b.get('issues') or [])),
                        'cover_view_max_aspect': st.get('cover_view_max_aspect'),
                        'room_pan_unconfirmed': sum(1 for b in unconfirmed if any(str(i).startswith(ROOM_PAN_REASON) for i in b.get('issues') or [])),
                        'room_pan': {k: v for k, v in (art.get('room_pan') or {}).items() if k != 'demoted_instance_ids'} or None,
                        'shelf_plane': {k: v for k, v in (art.get('shelf_plane') or {}).items() if k not in ('links', 'rejected')} or None,
                        'reading_vetoes': st.get('reading_vetoes'), 'groups_with_readable_text': st.get('groups_with_readable_text'),
                        'reconciliation': {k: v for k, v in (art.get('reconciliation') or {}).items() if k != 'not_promoted'},
                        'median_native_spine_px': st.get('median_native_spine_px'), 'lower_supported_by_panorama_piece': sum(1 for b in books if b.get('lower_support') and len(b['member_group_ids']) == 1 and len(_segments_of(b, art)) < 2),
                        'limits': art.get('limits')})
    if c.get('panorama'):
        c['panorama'].update(used=False, estimator_used='instances', reason='the book lines are physical instances (count.instances); the panorama count is a cross-check')
    p.setdefault('uncertainty', {}).update(book_count_lower=lower, book_count_upper=upper)
    # review entries of merged-away members point at their instance; a split/unmerge restores them on the next apply (the refs are recomputed from the lines)
    q = []
    seen = set()
    for r in P.flat_review(p.get('review_queue')):
        ref = merged_line_ids.get(r.get('ref_id'), r.get('ref_id'))
        if ref != r.get('ref_id') and ref in {b['id'] for b in books} and r.get('code') in ('unidentified_book', 'line_issue', 'uncertain_dedup', 'low_detection_confidence'):
            continue                                                                        # the merged line carries its own entries
        key = (ref, r.get('code'), r.get('reason'))
        if key in seen:
            continue
        seen.add(key)
        q.append({**r, 'ref_id': ref} if ref != r.get('ref_id') else r)
    line_ids = {b['id'] for b in books} | {i['id'] for i in items}
    q = [r for r in q if r.get('code') != 'count_estimator_instances' and (not str(r.get('ref_id', '')).startswith(('bk-', 'it-')) or r['ref_id'] in line_ids)]
    q.append({'ref_id': 'capture', 'code': 'count_estimator_instances', 'severity': 'info', 'reason': note + f"; association reliability {frag.get('association_reliability') or 'n/a'} "
              f"({frag.get('transitions_registered', 0)} of {frag.get('transitions', 0)} tracking breaks registered across)."})
    p['review_queue'] = P.consolidate_review(q)             # one entry per ref_id
    p['language_breakdown'] = P.language_breakdown(books)
    p['completeness'] = P.completeness(books, items, p.get('room') or {}, t)
    # The headline count (the panorama count when the panorama is usable, else these lines); the line figures stay in totals.book_lines / count.lines
    from pipeline import panorama_attach as PA
    hl = PA.headline(p, fresh=True)
    return {'applied': True, 'book_count': t['book_count'], 'book_lines': n, 'headline_estimator': hl['estimator'], 'track_groups': track_groups, 'merged_groups': merged_groups, 'range': [lower, upper], 'reliability': rel, 'splits': len(split_log)}


def _segments_of(line: dict[str, Any], art: dict[str, Any]) -> set[int]:
    """Tracking segments of a line's sightings, from the artifact's per-observation segments (cached on the artifact object)."""
    seg = art.get('_segment_of_obs')
    if seg is None:
        seg = art['_segment_of_obs'] = {o: s for inst in art.get('instances') or [] for o, s in zip(inst.get('observation_ids') or [], inst.get('observation_segments') or [])}
    return {seg[o] for o in (line.get('evidence') or {}).get('observation_ids') or [] if o in seg}


# ------------------------------------------------------------------------------------------ attach in place, the post-sweep step, CLI
def compute(run_dir: str | Path, *, panorama_count: dict[str, Any] | None = None, cross_segment: bool = True, cross_config: dict[str, Any] | None = None,
            packet_totals: dict[str, Any] | None = None, progress=None, readings_path: str | Path | None = None, panorama_rule: bool | str = True,
            pair_oracle: str | Path | bool | None = None, reconcile_with_panorama: bool = True, room_pan: bool = True, plane_merge: bool | None = None,
            recon3d: str | Path | None = None, recon3d_agree: list[str | Path] | None = None, plane_appearance: bool = True, short_gap: bool | None = None,
            revisit: bool | None = None) -> dict[str, Any]:
    """Build the instances artifact of a run from its saved inventory, manifest, candidates, panorama count and readings (nothing in the run is modified).
    ``readings_path``: a readings.json to use for the reading-contradiction veto instead of the run's own (e.g. a hosted re-read kept in another run dir)."""
    from pipeline import panorama_attach as PA
    root = Path(run_dir)
    inv_env = json.loads((root / 'inventory.json').read_text())
    inv = inv_env['data']
    manifest = json.loads((root / 'manifest.json').read_text())['data']
    frames = {f['frame_id']: f for f in manifest['frames']}
    tr = (inv_env.get('metrics') or {}).get('track_report') or {}
    segments = segment_map(frames, tr)
    pc = panorama_count if panorama_count is not None else PA.load(root)
    cross = None
    cands = json.loads((root / 'candidates.json').read_text())['data']
    observations = cands.get('observations', cands.get('detections', []))
    if cross_segment and any(s for s in segments.values()):
        cross = cross_segment_links(root, frames, observations, inv.get('books') or [], segments, cross_config, progress=progress)
    elif cross_segment:
        cross = {'links': [], 'rejected': [], 'transitions': [], 'frames_used': 0, 'pairs_tested': 0, 'pairs_registered': 0, 'transitions_total': 0, 'transitions_registered': 0,
                 'transitions_with_links': 0, 'scale': None, 'config': {k: (cross_config or {}).get(k, CROSS[k]) for k in CROSS}, 'seconds': 0.0}
    eligible, why = False, 'no panorama count for this run'
    if pc:
        totals = packet_totals
        if totals is None:
            try:
                totals = json.loads((root / 'claim_packet.json').read_text())['data']['totals']
            except (OSError, ValueError, KeyError):
                totals = {'count_reliability': (inv.get('counts') or {}).get('count_reliability')}
        eligible, why = PA.decide(own_reliability_totals(totals, cross), pc)
    placed = placed_in_normal_pieces(root, pc) if pc and eligible else set()
    sg = short_gap_links(root, frames, observations, inv.get('books') or [], segments, progress=progress) if short_gap_enabled(short_gap) else None
    rv = revisit_links(root, frames, observations, inv.get('books') or [], segments, progress=progress) if revisit_enabled(revisit) else None
    pan_checks = None
    if pc and eligible and panorama_rule:
        books = inv.get('books') or []
        frames_of = {b['instance_id']: {o.split('-o')[0] for o in b['observation_ids']} for b in books}
        todo = []
        for spine, a, b in panorama_pairs(pc.get('merge_evidence') or [], frames_of):
            iou = {g['id']: g.get('iou') or 0.0 for g in spine['groups']}
            # registration_decides: every candidate is checked (an agreeing registration decides even below the placement IoU);
            # placement_and_frame checks only the placements >= 0.35
            if (panorama_rule != 'placement_and_frame' or min(iou[a], iou[b]) >= PANORAMA_ACCEPT['min_spine_iou']) and not frames_of[a] & frames_of[b]:
                todo.append((a, b))
        pan_checks = panorama_frame_checks(root, frames, observations, books, todo)
    rpath = Path(readings_path) if readings_path else root / 'readings.json'
    try:
        readings = json.loads(rpath.read_text()).get('data', {}).get('readings', [])
    except (OSError, ValueError):
        readings = []
    oracle_links, oracle_info = oracle_links_for(root, pair_oracle)
    from pipeline import room_pan as RP
    rp = RP.classify_run(root) if room_pan else None
    plane = (plane_links_for(root, inv.get('books') or [], segments, recon3d, agree=recon3d_agree,
                             appearance=group_appearance(root, frames, observations, inv.get('books') or []) if plane_appearance else None)
             if plane_merge_enabled(plane_merge) else None)
    art = build(inv, frames, segments, panorama_count=pc, panorama_eligible=eligible, panorama_reason=why, cross=cross, run_id=root.name, inventory_sha256=sha256_of(root / 'inventory.json'),
                observations=observations, placed_in_normal_piece=placed, readings=readings, pan_checks=pan_checks, panorama_rule=panorama_rule, oracle=oracle_links,
                piece_sightings=piece_sightings(root, pc) if (pc and eligible and reconcile_with_panorama) else None, reconcile_with_panorama=reconcile_with_panorama,
                reconcile_frame_check=neighbour_frame_check(root, frames, observations) if (pc and eligible and reconcile_with_panorama) else None, room_pan=rp, plane=plane,
                short_gap=sg, revisit=rv)
    art['readings_source'] = str(rpath) if readings else None
    if oracle_info:
        art['pair_oracle'] = {**oracle_info, 'unions': (art['stats']['merged_by'] or {}).get(KIND_ORACLE, 0)}
    return art


def oracle_links_for(root: Path, pair_oracle: str | Path | bool | None) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """The opt-in hosted pair adjudication links (pipeline.pair_oracle). ``pair_oracle``: None -> the env flag LIBRARY_PAIR_ORACLE=1 (default OFF); True -> the run's
    pair_oracle.json; a path -> that file. A missing file is recorded, never an error."""
    if pair_oracle is None:
        pair_oracle = os.environ.get('LIBRARY_PAIR_ORACLE') == '1'
    if pair_oracle is False:
        return None, None
    from pipeline import pair_oracle as PO
    path = root / PO.ARTIFACT if pair_oracle is True else Path(pair_oracle)
    art = PO.load(path)
    if not art:
        return None, {'enabled': True, 'source': str(path), 'links': 0, 'reason': 'no pair_oracle.json'}
    links, refused = PO.links(art)
    return links, {'enabled': True, 'source': str(path), 'model': art.get('model'), 'links': len(links), 'refused_by_geometry': len(refused), 'label': ORACLE_LABEL,
                   'note': 'model-read: answer same + relative spine height ratio + not the same frame; two models agreeing is not truth'}


def placed_in_normal_pieces(run_dir: Path, pc: dict[str, Any]) -> set[str]:
    """Group ids whose panorama marker lies in a 'normal' (detector-counted) piece of the panorama the count was made from (panorama.json markers + panorama_count's piece classes)."""
    pan = run_dir.parent / str(pc.get('panorama_run') or '')
    try:
        comps = json.loads((pan / 'panorama.json').read_text())['components']
    except (OSError, ValueError, KeyError):
        return set()
    normal = {c['component_id'] for c in (pc.get('panorama_components') or []) if c.get('class') == 'normal'}
    return {m['group_id'] for c in comps if c.get('component_id') in normal for m in c.get('markers') or [] if m.get('polygon')}


def write(art: dict[str, Any], run_dir: str | Path, pan_dir: str | Path | None = None) -> list[Path]:
    out = []
    for d in [Path(run_dir)] + ([Path(pan_dir)] if pan_dir else []):
        if not d.is_dir():
            continue
        tmp = d / (ARTIFACT + '.tmp')
        tmp.write_text(json.dumps(art, indent=1))
        tmp.replace(d / ARTIFACT)
        out.append(d / ARTIFACT)
    return out


def attach(run_dir: str | Path, art: dict[str, Any] | None = None, *, render: bool = True, extra_ledger: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Patch the run's claim_packet.json in place (verified; nothing written when the verification fails). ``extra_ledger`` entries (e.g. a ``split``) are appended to
    the packet's corrections ledger first. Idempotent: the merged lines carry their members."""
    from pipeline import contracts as C
    from pipeline import packet as P
    from pipeline import panorama_attach as PA
    run_dir = Path(run_dir)
    pk = run_dir / C.ARTIFACTS['packet']
    if not pk.is_file():
        return {'attached': False, 'reason': 'no claim_packet.json'}
    art = art if art is not None else load(run_dir)
    if not art:
        return {'attached': False, 'reason': f'no {ARTIFACT} for this run'}
    env = C.read_json(pk)
    if env['data'].get('fixture'):
        return {'attached': False, 'reason': 'fixture packet'}
    backup = run_dir / BACKUP
    if not backup.is_file():
        shutil.copy2(pk, backup)
    manifest = C.read_json(run_dir / C.ARTIFACTS['frames'])
    data = env['data']
    corr = data.setdefault('corrections', {'ledger': [], 'applied': []})
    for e in extra_ledger or []:
        if e.get('id') not in {x.get('id') for x in corr.get('ledger') or []}:
            corr.setdefault('ledger', []).append(e)
    pc = PA.load(run_dir)
    if pc:
        PA.apply(data, pc)
    res = apply(data, art, corr.get('ledger'))
    P.finalize_contract(data, run_dir)
    problems = P.verify_packet(env, manifest, run_dir)
    if problems:
        return {'attached': False, 'reason': 'packet verification failed after the instance layer was attached', 'problems': problems[:10]}
    C.write_json(pk, env)
    if render:
        from pipeline import report as R
        R.render(pk, run_dir)
        if (run_dir / 'report.pdf').is_file():
            R.render_print_pdf(pk, run_dir)
    return {'attached': True, **res}


def run_step(runs_dir: str | Path, run_id: str, panorama_out_run: str | None = None, *, timeout_s: float = 300.0) -> dict[str, Any]:
    """The post-sweep step (app.post_sweep): compute the instances of a run in a niced subprocess (OpenCV must not stall the server), write instances.json and attach it to
    the packet in place. Returns the summary recorded in the queue state. Raises on failure (recorded; never fails the maps)."""
    runs = Path(runs_dir)
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, '-m', 'pipeline.instances', '--run', str(runs / run_id), '--attach']
    if panorama_out_run:
        cmd += ['--panorama-dir', str(runs / panorama_out_run)]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=timeout_s, preexec_fn=lambda: os.nice(10))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-1:] or ['exited with code ' + str(proc.returncode)]
        raise RuntimeError(f'instance step failed: {tail[0][:200]}')
    brief = json.loads(proc.stdout[proc.stdout.index('{'):])
    att = brief.get('attach') or {}
    st = brief.get('stats') or {}
    out = {'instances': st.get('instances'), 'track_groups': st.get('track_groups'), 'merged_groups': st.get('merged_groups'), 'merged_by': st.get('merged_by'),
           'book_count': att.get('book_count'), 'book_lines': att.get('book_lines'), 'headline_estimator': att.get('headline_estimator'),
           'range': att.get('range'), 'reliability': att.get('reliability'), 'attached': bool(att.get('attached')),
           'seconds': (brief.get('cross_segment') or {}).get('seconds'), 'panorama_eligible': (brief.get('panorama') or {}).get('eligible')}
    head = (f"{att['book_count']} books (2D panorama count), {att.get('book_lines')} book lines " if att.get('headline_estimator') == 'panorama'
            else f"{att.get('book_count', st.get('instances'))} books ")
    out['summary'] = (head + f"({st.get('merged_groups', 0)} of {st.get('track_groups', 0)} track groups merged: "
                      f"panorama {(st.get('merged_by') or {}).get(KIND_PANORAMA, 0)}, cross-segment {(st.get('merged_by') or {}).get(KIND_CROSS, 0)})"
                      + ('' if att.get('attached') else f" (not attached: {att.get('reason')})"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--panorama-count', type=Path, help='panorama_count.json to use (default: the run\'s own)')
    ap.add_argument('--panorama-dir', type=Path, help='the panorama output dir that gets a copy of instances.json')
    ap.add_argument('--no-cross-segment', action='store_true')
    ap.add_argument('--no-panorama-rule', action='store_true', help='accept every panorama_overlap placement (the pre-Worker-28 behaviour; for comparison only)')
    ap.add_argument('--panorama-rule', choices=PANORAMA_RULES, default='registration_decides', help='registration_decides (default): a registered frame pair decides, the placement IoU only where none '
                    'registered; placement_and_frame: placement AND frame check (comparison only)')
    ap.add_argument('--readings', type=Path, help='readings.json for the reading-contradiction veto (default: the run\'s own)')
    ap.add_argument('--pair-oracle', nargs='?', const=True, default=None, help='use the hosted pair adjudication links (pipeline.pair_oracle; model-read): the run\'s '
                    'pair_oracle.json, or the given file (default: the env flag LIBRARY_PAIR_ORACLE=1, OFF)')
    ap.add_argument('--no-reconcile', action='store_true', help='do not promote unconfirmed single sightings onto unoccupied panorama spines (comparison only)')
    ap.add_argument('--plane-merge', choices=('0', '1'), help=f'shelf_plane_position unions from the merged 3D map (default: env {PLANE_ENV}, else {PLANE_DEFAULT})')
    ap.add_argument('--recon3d', type=Path, help='recon3d dir or recon3d.json for --plane-merge (default: the run\'s best <run>_recon3d_vN)')
    ap.add_argument('--recon3d-agree', type=Path, action='append', help='another recon3d build that must also place the pair within tolerance (two-build agreement; repeatable)')
    ap.add_argument('--no-plane-appearance', action='store_true', help='plane merge without the appearance gate (round 1 behaviour; comparison only)')
    ap.add_argument('--short-gap', choices=('0', '1'), help=f'short_gap_registration unions: detection frames <= {SHORT_GAP["max_gap_s"]:.0f} s apart, any segment ('
                    f'default: env {SHORT_GAP_ENV}, else {SHORT_GAP_DEFAULT})')
    ap.add_argument('--revisit', choices=('0', '1'), help=f'revisit_registration unions: detection frames {REVISIT["min_gap_s"]:.0f}-{REVISIT["max_gap_s"]:.0f} s apart with the '
                    f'layout check (default: env {REVISIT_ENV}, else {REVISIT_DEFAULT})')
    ap.add_argument('--no-room-pan', action='store_true', help='do not exclude room-pan frames (pipeline.room_pan) from counting (comparison only)')
    ap.add_argument('--out', type=Path, help='write the artifact here instead of into the run dir (nothing in the run is touched)')
    ap.add_argument('--attach', action='store_true', help='after writing, patch the run\'s claim_packet.json in place')
    ap.add_argument('--no-render', action='store_true')
    a = ap.parse_args(argv)
    if isinstance(a.pair_oracle, str) and not Path(a.pair_oracle).is_file():   # a word like '--readings x' (an unsplit shell variable) must not become the oracle path silently
        ap.error(f'--pair-oracle: no such file: {a.pair_oracle!r}')
    pc = json.loads(a.panorama_count.read_text()) if a.panorama_count else None
    art = compute(a.run, panorama_count=pc, cross_segment=not a.no_cross_segment, readings_path=a.readings, panorama_rule=False if a.no_panorama_rule else a.panorama_rule,
                  pair_oracle=a.pair_oracle, reconcile_with_panorama=not a.no_reconcile, room_pan=not a.no_room_pan,
                  plane_merge=None if a.plane_merge is None else a.plane_merge == '1', recon3d=a.recon3d,
                  recon3d_agree=a.recon3d_agree, plane_appearance=not a.no_plane_appearance, short_gap=None if a.short_gap is None else a.short_gap == '1',
                  revisit=None if a.revisit is None else a.revisit == '1')
    if a.out:
        a.out.write_text(json.dumps(art, indent=1))
    else:
        write(art, a.run, a.panorama_dir)
    brief = {k: art[k] for k in ('run_id', 'stats', 'count', 'fragmentation', 'panorama')} | ({'pair_oracle': art['pair_oracle']} if art.get('pair_oracle') else {})
    brief['cross_segment'] = {k: v for k, v in art['cross_segment'].items() if k not in ('links', 'rejected', 'transitions')}
    if art.get('short_gap') is not None:
        brief['short_gap'] = {k: v for k, v in art['short_gap'].items() if k not in ('links', 'rejected', 'config')} | {'links': len(art['short_gap']['links']), 'near_tie_vetoes': len(art['short_gap']['rejected'])}
    if art.get('revisit') is not None:
        brief['revisit'] = {k: v for k, v in art['revisit'].items() if k not in ('links', 'rejected', 'layout_rejected', 'config')} | {
            'links': len(art['revisit']['links']), 'near_tie_vetoes': len(art['revisit']['rejected']), 'layout_rejected_pairs': len(art['revisit']['layout_rejected'])}
    if art.get('shelf_plane') is not None:
        brief['shelf_plane'] = {k: v for k, v in art['shelf_plane'].items() if k not in ('links', 'rejected')} | {'links': len(art['shelf_plane']['links']), 'near_tie_vetoes': len(art['shelf_plane']['rejected'])}
    brief['reconciliation'] = {k: v for k, v in art['reconciliation'].items() if k not in ('promoted', 'not_promoted', 'rule')}
    if a.attach and not a.out:
        brief['attach'] = attach(a.run, art, render=not a.no_render)
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
