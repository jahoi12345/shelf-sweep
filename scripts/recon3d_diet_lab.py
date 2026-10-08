"""Frame-diet experiments for pipeline.recon3d without paying SIFT extraction per trial (dev tool; never writes into runs/).

  extract: SIFT features of EVERY frame of a saved run into a scratch cache (one time per run).
  trial:   pick frames by a rule (or an id list), copy the cache database, match only the sequential pairs of that selection
           (same pair rule as COLMAP's sequential matcher, incl. the quadratic neighbours unless --no-quadratic), run the incremental
           mapper on those images, print registration numbers and the matching/mapping seconds. Extraction seconds of a real build
           are estimated as frames_used x the per-frame cost measured by `extract`.

  .venv/bin/python scripts/recon3d_diet_lab.py extract --run runs/<run> --cache <dir>
  .venv/bin/python scripts/recon3d_diet_lab.py trial --run runs/<run> --cache <dir> --rule diet --overlap 6 [--bridge-gap 8] [--no-quadratic]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import recon3d as R  # noqa: E402


def _frames(run):
    return json.loads((Path(run) / 'manifest.json').read_text())['data']['frames']


def extract(run, cache, max_image_size=1600, max_features=None, frame_range=None):
    import pycolmap
    run, cache = Path(run).resolve(), Path(cache)
    images = cache / 'images'
    images.mkdir(parents=True, exist_ok=True)
    frames = _frames(run)
    if frame_range:
        frames = frames[frame_range[0]:frame_range[1] + 1]
    for f in frames:
        link = images / f"{f['frame_id']}{Path(f['image_path']).suffix.lower()}"
        if not link.exists():
            link.symlink_to((run / f['image_path']).resolve())
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = max_image_size
    if max_features:
        ext.sift.max_num_features = max_features
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = 'SIMPLE_RADIAL'
    t0, c0 = time.perf_counter(), time.process_time()
    pycolmap.extract_features(cache / 'database.db', images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext, device=pycolmap.Device.cpu)
    dt, cpu = time.perf_counter() - t0, time.process_time() - c0
    info = {'frames': len(frames), 'seconds': round(dt, 1), 'per_frame_s': round(dt / len(frames), 3), 'cpu_s': round(cpu, 1),
            'cpu_per_frame_s': round(cpu / len(frames), 3), 'max_image_size': max_image_size, 'max_features': max_features or 8192}
    (cache / 'extract.json').write_text(json.dumps(info))
    print(json.dumps(info))


def bridged(frames, kept, gap):
    """`kept` plus, wherever two consecutive kept frames are more than `gap` run frames apart, evenly spaced frames from the gap
    (the sharpest frame of each slot, blur-gated or not) so the sequence never jumps more than `gap` frames."""
    index = {f['frame_id']: i for i, f in enumerate(frames)}
    keep = sorted(index[f['frame_id']] for f in kept)
    add = set()
    for a, b in zip(keep, keep[1:]):
        if b - a > gap:
            slots = -(-(b - a) // gap)                       # ceil: pieces of <= gap
            for s in range(1, slots):
                lo, hi = a + round((b - a) * (s - 0.5) / slots), a + round((b - a) * (s + 0.5) / slots)
                best = max(range(max(a + 1, lo), min(b, hi + 1)), key=lambda i: (R._sharp(frames[i]), -i), default=None)
                if best is not None:
                    add.add(best)
    return [frames[i] for i in sorted(set(keep) | add)]


def pairs_of(names, overlap, quadratic=True):
    out = set()
    n = len(names)
    for i in range(n):
        for k in range(overlap):
            for j in ((i + k + 1),) + (((i + (1 << k)),) if quadratic else ()):
                if i < j < n:
                    out.add((names[i], names[j]))
    return sorted(out)


def trial(run, cache, rule, overlap, *, max_frames=None, bridge_gap=None, quadratic=True, ids_file=None, keep_work=False, tag='trial', frame_range=None, mapper_opts=None):
    import pycolmap
    cache = Path(cache)
    all_frames = _frames(run)
    if ids_file:
        want = set(Path(ids_file).read_text().split())
        kept, info = [f for f in all_frames if f['frame_id'] in want], {'rule': 'ids'}
    elif rule.startswith('every') or rule == 'nongated':
        if rule == 'nongated':
            kept = [f for f in all_frames if not R._gated(f)]
        else:
            kept = all_frames[::int(rule[5:])]
        info = {'rule': rule}
    else:
        kept, info = R.select_frames(all_frames, rule, max_frames)
    if frame_range:
        lo, hi = frame_range
        keep_ids = {f['frame_id'] for f in all_frames[lo:hi + 1]}
        kept = [f for f in kept if f['frame_id'] in keep_ids]
    if bridge_gap:
        kept = bridged(all_frames, kept, bridge_gap)
    names = [f"{f['frame_id']}.jpg" for f in kept]
    if overlap is None:
        overlap = R.sequential_overlap(len(kept), len(all_frames))
    work = cache / f'_{tag}'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    db = work / 'database.db'
    shutil.copy(cache / 'database.db', db)
    pairs = pairs_of(names, overlap, quadratic)
    (work / 'pairs.txt').write_text('\n'.join(f'{a} {b}' for a, b in pairs))
    t0, c0 = time.perf_counter(), time.process_time()
    pycolmap.match_image_pairs(db, pairing_options=_imported(pycolmap, work / 'pairs.txt'), device=pycolmap.Device.cpu)
    t_match, c_match = time.perf_counter() - t0, time.process_time() - c0
    t0, c0 = time.perf_counter(), time.process_time()
    sparse = work / 'sparse'
    sparse.mkdir()
    opts = pycolmap.IncrementalPipelineOptions()
    opts.image_names = names
    for k, v in (mapper_opts or {}).items():
        setattr(opts, k, v)
    models = pycolmap.incremental_mapping(db, cache / 'images', sparse, opts)
    t_map, c_map = time.perf_counter() - t0, time.process_time() - c0
    sizes = sorted((len(R._registered_images(m)) for m in models.values()), reverse=True)
    per_frame = json.loads((cache / 'extract.json').read_text())['per_frame_s'] if (cache / 'extract.json').is_file() else None
    out = {'tag': tag, 'rule': rule if not ids_file else 'ids', 'bridge_gap': bridge_gap, 'quadratic': quadratic, 'overlap': overlap,
           'frames_used': len(kept), 'frames_total': len(all_frames), 'pairs': len(pairs),
           'registered_largest': sizes[0] if sizes else 0, 'registered_largest_ratio': round(sizes[0] / len(kept), 3) if sizes else 0.0,
           'registered_of_total': round(sizes[0] / len(all_frames), 3) if sizes else 0.0, 'fragments': sizes[:10], 'models': len(sizes),
           'matching_s': round(t_match, 1), 'mapping_s': round(t_map, 1), 'matching_cpu_s': round(c_match, 1), 'mapping_cpu_s': round(c_map, 1),
           'mapper_opts': mapper_opts or None,
           'est_extract_s': None if per_frame is None else round(per_frame * len(kept), 1),
           'est_total_s': None if per_frame is None else round(per_frame * len(kept) + t_match + t_map, 1)}
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(out), flush=True)
    return out


def _imported(pycolmap, path):
    o = pycolmap.ImportedPairingOptions()
    o.match_list_path = str(path)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--run', required=True)
    e.add_argument('--cache', required=True)
    e.add_argument('--max-image-size', type=int, default=1600)
    e.add_argument('--max-features', type=int, default=None)
    e.add_argument('--range', default=None)
    t = sub.add_parser('trial')
    t.add_argument('--run', required=True)
    t.add_argument('--cache', required=True)
    t.add_argument('--rule', default='diet', help='all|diet|keyframes|nongated|everyN')
    t.add_argument('--overlap', type=int, default=None)
    t.add_argument('--max-frames', type=int, default=None)
    t.add_argument('--bridge-gap', type=int, default=None, help='insert frames wherever consecutive kept frames are > N run frames apart')
    t.add_argument('--no-quadratic', action='store_true')
    t.add_argument('--ids-file')
    t.add_argument('--tag', default='trial')
    t.add_argument('--mapper', default=None, help='JSON of IncrementalPipelineOptions attributes, e.g. {"ba_global_frames_ratio": 1.5}')
    t.add_argument('--range', default=None, help='a:b, keep only run frame indices a..b of the selection (compare densities on one segment)')
    a = ap.parse_args()
    if a.cmd == 'extract':
        extract(a.run, a.cache, a.max_image_size, a.max_features, tuple(int(v) for v in a.range.split(':')) if a.range else None)
    else:
        trial(a.run, a.cache, a.rule, a.overlap, max_frames=a.max_frames, bridge_gap=a.bridge_gap, quadratic=not a.no_quadratic,
              ids_file=a.ids_file, tag=a.tag, frame_range=tuple(int(v) for v in a.range.split(':')) if a.range else None,
              mapper_opts=json.loads(a.mapper) if a.mapper else None)


if __name__ == '__main__':
    main()
