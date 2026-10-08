"""Room-pan frames: tag the frames of a sweep that are NOT the close shelf pass, so the count layers can leave them out.

The user films ONE sweep: the close shelf pass plus a pan to the floor line, the ceiling corners and the side walls (for the room size). Those pan
frames still feed the 3D map / room geometry, but their book sightings (far, oblique re-views of the shelf) must not become book lines, panorama
spines or shelf run. No manual marks: every tag comes from signals already in the run.

Per frame (manifest order), from the run's own files:
  (b) far spines   - a detection frame whose book polygons have a median long side < FAR_FRACTION of the run's median native spine
                     (pipeline.panorama_count.FAR_FRACTION, the panorama count's far-view rule);
  (c) no shelf     - a frame the detector processed (detection_responses/ or candidates.json) with ZERO book detections: floor, ceiling, walls;
  (a) 3D pose      - when a recon3d.json exists (newest ``<run>_recon3d_vN``): the camera's optical axis against the fitted shelf-plane normal
                     (> OBLIQUE_DEG: the camera looks along the shelf plane, at the floor, ceiling or a side wall) and its distance to the plane against
                     the median distance of the shelf frames (reported; see below);
  (e) panorama     - membership in a 2D panorama piece (reported per piece by pipeline.panorama_count, which uses the tags).
A frame is SHELF when it is a detection frame with book polygons whose median spine is >= FAR_FRACTION of the run median. 2D shelf evidence always
wins over the 3D pose: a joined 3D fragment can carry wrong poses (YugoTest5 f203-231: close shelf frames whose joined fragment reads 'oblique').
A frame is OFF-SHELF when it is not shelf and (c) or (a: oblique) holds.

(d) time contiguity: a ROOM PAN is a maximal run of consecutive non-shelf frames (bounded by shelf frames or the sweep's ends) that holds at least
MIN_OFFSHELF_FRAMES off-shelf frames spread over at least MIN_OFFSHELF_SPAN_S seconds (first to last off-shelf frame). Every frame of such a run is a
room-pan frame (the far re-views of the shelf inside the pan with it). A walk-up (far frames facing the shelf, no off-shelf frames), a glance away of a
second or two and a detector miss on a few frames are not a room pan. Without a 3D map the rule runs on (b)+(c)+(d) alone.
(f) app room-sweep prompt: when the live app told the person the shelf pass is over and to film the room (manifest
``room_sweep.prompt.t_s``: the End press / "done", or the live left-the-shelf trigger), every run of non-shelf frames that reaches past
that time is a room pan, whatever its length (a short or far-only room pan after the prompt is still not the shelf pass). Close shelf
frames after the prompt stay shelf frames.

    python -m pipeline.room_pan --run runs/<run> [--recon3d runs/<run>_recon3d_vN | --no-3d] [--json]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

FAR_FRACTION = 0.5            # = pipeline.panorama_count.FAR_FRACTION / pipeline.instances.FAR_FRACTION (the panorama count's far constant, reused)
OBLIQUE_DEG = 60.0            # optical axis more than 60 deg from the shelf-plane normal: the view runs along the plane (floor, ceiling, side wall), not at the shelf
FAR_DISTANCE_RATIO = 2.0      # reported only: camera farther than 2x the median shelf-frame distance from the plane
MIN_OFFSHELF_FRAMES = 3       # a room pan holds at least 3 off-shelf frames ...
MIN_OFFSHELF_SPAN_S = 4.0     # ... spread over at least 4 s: a slow pan to the floor/ceiling/walls (<= ~1000 px/s) takes seconds; a glance away or a pre-roll does not
PIECE_SHARE = 0.5             # a panorama piece is a room-pan piece when more than half of its frames are room-pan frames
REASON = 'room-pan view only'


def _long_side(poly: Any) -> float:
    p = np.asarray(poly, float)
    return float(np.linalg.norm(np.roll(p, -1, 0) - p, axis=1).max())


def latest_recon3d(run_dir: str | Path) -> Path | None:
    """Newest ``<run>_recon3d_vN/recon3d.json`` beside the run that has camera poses and a fitted plane, else None."""
    from pipeline.recon3d import map_consistency_at      # a map marked inconsistent (not one flat shelf face) is skipped
    run_dir = Path(run_dir)
    best = None
    for p in run_dir.parent.glob(f'{run_dir.name}_recon3d_v*'):
        m = re.fullmatch(re.escape(run_dir.name) + r'_recon3d_v(\d+)', p.name)
        if m and (p / 'recon3d.json').is_file() and (best is None or int(m.group(1)) > best[0]) and map_consistency_at(p) != 'inconsistent':
            best = (int(m.group(1)), p / 'recon3d.json')
    return best[1] if best else None


def camera_pose_signals(recon: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    """frame_id -> {'z': distance to the shelf plane (aligned frame, units), 'angle_deg': optical axis vs the plane normal}. Empty without an aligned plane."""
    if not recon or not recon.get('cameras') or not (recon.get('alignment') or {}).get('applied'):
        return {}
    out = {}
    for c in recon['cameras']:
        axis = np.asarray(c['rotation_world_to_cam'], float)[2]       # camera +z (optical axis) in the shelf-aligned world frame; the plane normal is +z, camera side
        out[c['frame_id']] = {'z': float(c['center'][2]), 'angle_deg': float(np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0))))}
    return out


def classify_frames(frames: list[dict[str, Any]], observations: list[dict[str, Any]], books: list[dict[str, Any]], *, detection_frames: set[str] | None = None,
                    poses: dict[str, dict[str, float]] | None = None, room_sweep_t: float | None = None) -> dict[str, Any]:
    """Pure. ``frames``: manifest frames (frame_id, timestamp_s) in capture order; ``observations``: candidates; ``books``: inventory track groups (their best
    polygons give the run median native spine); ``detection_frames``: frames the detector processed (default: frames with any observation); ``poses``:
    camera_pose_signals of a recon3d.json (None/empty -> the 2D fallback b+c+d); ``room_sweep_t``: capture time of the app's room-sweep prompt (f)."""
    native = [_long_side(b['polygon_xy']) for b in books if len(b.get('polygon_xy') or []) == 4]
    med = float(np.median(native)) if native else None
    spines: dict[str, list[float]] = collections.defaultdict(list)
    n_books: collections.Counter = collections.Counter()
    seen = set()
    for o in observations:
        seen.add(o['frame_id'])
        if o.get('kind') == 'book':
            n_books[o['frame_id']] += 1
            if len(o.get('polygon_xy') or []) == 4:
                spines[o['frame_id']].append(_long_side(o['polygon_xy']))
    det = (set(detection_frames) if detection_frames is not None else set()) | seen
    poses = poses or {}
    rows = []
    for f in frames:
        fid = f['frame_id']
        sp = spines.get(fid) or []
        rel = (float(np.median(sp)) / med) if (sp and med) else None
        rows.append({'frame_id': fid, 't': float(f.get('timestamp_s') or 0.0), 'detected': fid in det, 'books': n_books[fid],
                     'spine_vs_run': None if rel is None else round(rel, 3), 'pose': poses.get(fid)})
    shelf_z = [r['pose']['z'] for r in rows if r['pose'] and r['spine_vs_run'] is not None and r['spine_vs_run'] >= FAR_FRACTION]
    z_ref = float(np.median(shelf_z)) if shelf_z else None
    sig = collections.Counter()
    for r in rows:
        far_2d = r['spine_vs_run'] is not None and r['spine_vs_run'] < FAR_FRACTION
        no_books = r['detected'] and r['books'] == 0
        shelf = r['books'] > 0 and r['spine_vs_run'] is not None and not far_2d
        p = r['pose']
        oblique = bool(p) and p['angle_deg'] > OBLIQUE_DEG
        far_3d = bool(p) and z_ref is not None and z_ref > 0 and p['z'] > FAR_DISTANCE_RATIO * z_ref
        sig['b_far_spines'] += far_2d
        sig['c_detected_no_books'] += no_books
        sig['a_oblique_3d'] += oblique
        sig['a_far_3d'] += far_3d
        r['signals'] = [k for k, v in (('far_spines', far_2d), ('no_books', no_books), ('oblique_3d', oblique), ('far_3d', far_3d)) if v]
        r['label'] = 'shelf' if shelf else ('off_shelf' if (no_books or oblique) else ('far' if (far_2d or far_3d) else 'none'))
    segments, cur = [], []
    for i, r in enumerate(rows + [None]):
        if r is not None and r['label'] != 'shelf':
            cur.append(i)
            continue
        if cur:
            off = [j for j in cur if rows[j]['label'] == 'off_shelf']
            span = (rows[off[-1]]['t'] - rows[off[0]]['t']) if off else 0.0
            after_prompt = room_sweep_t is not None and rows[cur[-1]]['t'] >= room_sweep_t
            is_pan = (len(off) >= MIN_OFFSHELF_FRAMES and span >= MIN_OFFSHELF_SPAN_S) or after_prompt
            segments.append({'first': rows[cur[0]]['frame_id'], 'last': rows[cur[-1]]['frame_id'], 'frames': len(cur), 'off_shelf_frames': len(off),
                             'off_shelf_span_s': round(span, 2), 'far_frames': sum(1 for j in cur if rows[j]['label'] == 'far'),
                             'seconds': round(rows[cur[-1]]['t'] - rows[cur[0]]['t'], 2), 'room_pan': is_pan, 'after_room_sweep_prompt': after_prompt, '_idx': cur})
            cur = []
    tagged = []
    for s in segments:
        if s['room_pan']:
            tagged += [rows[j]['frame_id'] for j in s['_idx']]
    pans = [{k: v for k, v in s.items() if k != '_idx'} for s in segments if s['room_pan']]
    near_misses = sorted(({k: v for k, v in s.items() if k != '_idx'} for s in segments if not s['room_pan'] and s['off_shelf_frames']),
                         key=lambda s: (-s['off_shelf_span_s'], -s['off_shelf_frames']))[:5]
    return {'frames_total': len(rows), 'room_pan_frames': tagged, 'room_pan_frame_count': len(tagged), 'segments': pans, 'largest_rejected_segments': near_misses,
            'signals': dict(sig), 'labels': dict(collections.Counter(r['label'] for r in rows)), 'run_median_spine_px': None if med is None else round(med, 1),
            'shelf_plane_distance_ref': None if z_ref is None else round(z_ref, 4), 'uses_3d': bool(poses), 'room_sweep_prompt_t': room_sweep_t,
            'constants': {'far_fraction': FAR_FRACTION, 'oblique_deg': OBLIQUE_DEG, 'far_distance_ratio_reported': FAR_DISTANCE_RATIO,
                          'min_offshelf_frames': MIN_OFFSHELF_FRAMES, 'min_offshelf_span_s': MIN_OFFSHELF_SPAN_S, 'piece_share': PIECE_SHARE},
            'rule': ('room pan = a run of consecutive non-shelf frames with >= %d off-shelf frames (detector saw no book, or the 3D camera looks > %.0f deg off the shelf '
                     'normal) spread over >= %.0f s; every frame of that run is excluded from counting (still used for 3D/room); a shelf frame = book polygons with '
                     'median spine >= %.0f%% of the run median (2D shelf evidence wins over the 3D pose)') % (MIN_OFFSHELF_FRAMES, OBLIQUE_DEG, MIN_OFFSHELF_SPAN_S, FAR_FRACTION * 100),
            'per_frame': [{k: r[k] for k in ('frame_id', 't', 'label', 'signals', 'books', 'spine_vs_run')} for r in rows]}


def classify_run(run_dir: str | Path, *, recon3d: str | Path | bool | None = None) -> dict[str, Any]:
    """Load a run's manifest / candidates / inventory (+ detection_responses frame list, + the newest recon3d.json unless ``recon3d`` is False or a path) and
    classify. Never raises on a missing optional input; a run without a manifest gets an empty result."""
    root = Path(run_dir)
    try:
        mdata = json.loads((root / 'manifest.json').read_text())['data']
        frames = mdata['frames']
        cands = json.loads((root / 'candidates.json').read_text())['data']
        books = json.loads((root / 'inventory.json').read_text())['data'].get('books') or []
    except (OSError, ValueError, KeyError):
        return {'frames_total': 0, 'room_pan_frames': [], 'room_pan_frame_count': 0, 'segments': [], 'uses_3d': False, 'reason': 'run files missing'}
    observations = cands.get('observations', cands.get('detections', []))
    dr = root / 'detection_responses'
    det = {p.name[:6] for p in dr.glob('f*.json')} if dr.is_dir() else None
    path = None if recon3d is False else (Path(recon3d) / 'recon3d.json' if isinstance(recon3d, (str, Path)) and Path(recon3d).is_dir() else
                                          Path(recon3d) if isinstance(recon3d, (str, Path)) else latest_recon3d(root))
    poses, src = {}, None
    if path is not None:
        try:
            poses, src = camera_pose_signals(json.loads(Path(path).read_text())), str(Path(path).parent.name)
        except (OSError, ValueError):
            poses = {}
    prompt = ((mdata.get('room_sweep') or {}).get('prompt') or {}) if isinstance(mdata.get('room_sweep'), dict) else {}
    out = classify_frames(frames, observations, books, detection_frames=det, poses=poses,
                          room_sweep_t=float(prompt['t_s']) if prompt.get('t_s') is not None else None)
    out['recon3d'] = src if poses else None
    return out


def summary(res: dict[str, Any]) -> dict[str, Any]:
    """The compact record the count layers store (no per-frame list)."""
    return {k: res.get(k) for k in ('room_pan_frame_count', 'frames_total', 'segments', 'signals', 'labels', 'uses_3d', 'recon3d', 'rule', 'constants',
                                    'room_sweep_prompt_t')}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--recon3d', type=Path)
    ap.add_argument('--no-3d', action='store_true')
    ap.add_argument('--json', action='store_true', help='print the full result (per frame)')
    a = ap.parse_args(argv)
    res = classify_run(a.run, recon3d=False if a.no_3d else a.recon3d)
    print(json.dumps(res if a.json else {**summary(res), 'largest_rejected_segments': res.get('largest_rejected_segments')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
