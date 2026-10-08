"""Fragment-merge experiments for pipeline.recon3d (dev tool; never writes into runs/).

  base:  the production `fast` build (features + sequential matching + incremental mapping) into a scratch cache, KEEPING the
         database and every model (production keeps only the largest and deletes its work dir).
  diag:  per model the frame ranges it registered, overlaps between models, the frames between consecutive fragments with their
         blur / sharpness / motion, and the best verified two-view inlier count across each break.
  trial: copy the sequential database, add cross-fragment pairs (pipeline.recon3d_merge.bridge_pairs), match them, then map again
         (incremental from scratch | incremental continuing the largest model | global); prints fragments, sizes, seconds.

  .venv/bin/python scripts/recon3d_merge_lab.py base  --run runs/<run> --cache <dir>
  .venv/bin/python scripts/recon3d_merge_lab.py diag  --run runs/<run> --cache <dir>
  .venv/bin/python scripts/recon3d_merge_lab.py trial --run runs/<run> --cache <dir> --tag t1 --mapper incremental|continue|global
"""
from __future__ import annotations

import argparse
import numpy as np
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import recon3d as R  # noqa: E402
from pipeline import recon3d_merge as M  # noqa: E402


def _frames(run):
    return json.loads((Path(run) / 'manifest.json').read_text())['data']['frames']


def _names(frames):
    ordered = [f"{f['frame_id']}{Path(f['image_path']).suffix.lower()}" for f in frames]
    prefix = ordered != sorted(ordered)
    return [f'{i:06d}_{n}' if prefix else n for i, n in enumerate(ordered)]


def _mapping_options():
    import pycolmap
    opts = pycolmap.IncrementalPipelineOptions()
    for k, v in R.PROFILES['fast']['mapper_options'].items():
        setattr(opts, k, v)
    return opts


def base(run, cache):
    import pycolmap
    run, cache = Path(run).resolve(), Path(cache)
    images = cache / 'images'
    images.mkdir(parents=True, exist_ok=True)
    frames = _frames(run)
    for f, name in zip(frames, _names(frames)):
        link = images / name
        if not link.exists():
            link.symlink_to((run / f['image_path']).resolve())
    db = cache / 'database_seq.db'
    t = {}
    t0 = time.perf_counter()
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = 1024
    ext.sift.max_num_features = 2048
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = 'SIMPLE_RADIAL'
    pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=ext,
                              device=pycolmap.Device.cpu)
    t['features_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = R.BASE_OVERLAP
    pairing.quadratic_overlap = True
    pairing.loop_detection = False
    pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cpu)
    t['matching_s'] = time.perf_counter() - t0
    shutil.copy(db, cache / 'database_seq_pristine.db')
    t0 = time.perf_counter()
    sparse = cache / 'sparse_seq'
    sparse.mkdir(exist_ok=True)
    models = pycolmap.incremental_mapping(db, images, sparse, _mapping_options())
    t['mapping_s'] = time.perf_counter() - t0
    shutil.copy(cache / 'database_seq_pristine.db', db)
    info = {'frames': len(frames), 'models': {str(k): r.num_reg_images() for k, r in models.items()},
            **{k: round(v, 1) for k, v in t.items()}}
    (cache / 'base.json').write_text(json.dumps(info))
    print(json.dumps(info))


def load_models(path):
    import pycolmap
    out = {}
    for d in sorted(Path(path).iterdir()):
        if d.is_dir() and (d / 'images.bin').exists():
            out[int(d.name)] = pycolmap.Reconstruction(str(d))
    return out


def diag(run, cache, sparse='sparse_seq', db_name='database_seq.db'):
    frames = _frames(run)
    names = _names(frames)
    models = load_models(Path(cache) / sparse)
    frags = M.fragments_from_models(models, names)
    d = M.diagnose(frags, frames, Path(cache) / db_name, names)
    print(json.dumps(d, indent=1))
    (Path(cache) / f'diag_{sparse}.json').write_text(json.dumps(d))


def trial(run, cache, tag, mapper, window, ends, reps, panorama=None, reuse=None, mapper_json=None):
    import pycolmap
    cache = Path(cache)
    frames = _frames(run)
    names = _names(frames)
    models = load_models(cache / 'sparse_seq')
    frags = M.fragments_from_models(models, names)
    db = cache / f'database_{tag}.db'
    shutil.copy(cache / (f'database_{reuse}.db' if reuse else 'database_seq_pristine.db'), db)
    hints = M.panorama_hints(panorama, names) if panorama else []
    t = {}
    t0 = time.perf_counter()
    if reuse:
        pairs, why, stats = [], 'reused database_%s' % reuse, {}
    else:
        pairs, why = M.bridge_pairs(frags, len(names), overlap=R.BASE_OVERLAP, window=window, ends=ends, reps=reps, hints=hints)
        stats = M.match_pairs(db, pairs, names, cache / f'pairs_{tag}.txt')
    t['bridge_match_s'] = time.perf_counter() - t0
    links = M.cross_links(frags, db, names)
    out = cache / f'sparse_{tag}'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    t0 = time.perf_counter()
    if mapper == 'global':
        models2 = pycolmap.global_mapping(db, cache / 'images', out, pycolmap.GlobalPipelineOptions())
    elif mapper == 'continue':
        best = max(models, key=lambda k: models[k].num_reg_images())
        models2 = pycolmap.incremental_mapping(db, cache / 'images', out, _mapping_options(), input_path=str(cache / 'sparse_seq' / str(best)))
    else:
        opts = _mapping_options()
        for k, v in json.loads(mapper_json or '{}').items():
            if k.startswith('mapper.'):
                setattr(opts.mapper, k[7:], v)
            elif k.startswith('triangulation.'):
                setattr(opts.triangulation, k[14:], v)
            else:
                setattr(opts, k, v)
        models2 = pycolmap.incremental_mapping(db, cache / 'images', out, opts)
    t['mapping_s'] = time.perf_counter() - t0
    sizes = sorted((r.num_reg_images() for r in models2.values()), reverse=True)
    res = {'tag': tag, 'mapper': mapper, 'mapper_options': mapper_json, 'pairs': len(pairs), 'pair_reasons': why, 'match': stats, 'links': links,
           'models': len(models2), 'sizes': sizes, 'largest_ratio': round(sizes[0] / len(names), 4) if sizes else 0,
           **{k: round(v, 1) for k, v in t.items()}}
    (cache / f'trial_{tag}.json').write_text(json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if k != 'links'}))
    print('links', json.dumps(links)[:3000])


def check(run, cache, tag):
    """Quality of a trial's models: reprojection error, track length, agreement with every sequential fragment, trajectory jumps."""
    cache = Path(cache)
    names = _names(_frames(run))
    frags = load_models(cache / 'sparse_seq')
    merged = load_models(cache / f'sparse_{tag}')
    out = []
    for k, rec in sorted(merged.items(), key=lambda kv: -kv[1].num_reg_images()):
        ag = [a for a in M.agreement(rec, frags) if a['shared_frames']]
        med, jumps = M.trajectory_jumps(rec, names)
        row = {'model': k, 'registered': rec.num_reg_images(), 'points': rec.num_points3D(),
               'reproj_px': round(rec.compute_mean_reprojection_error(), 3), 'track': round(rec.compute_mean_track_length(), 2),
               'fragments_agree': sum(a['agrees'] for a in ag), 'fragments_shared': len(ag), 'jumps': len(jumps)}
        out.append(row)
        print(json.dumps(row))
        for a in ag:
            print('   ', json.dumps(a))
        print('    jumps', json.dumps(jumps[:10]))
        oc = M.orientation_coherence(rec, {k: f for k, f in frags.items() if any(a['fragment'] == k for a in ag)})
        row['orientation'] = oc
        for e in oc:
            print('    orient', json.dumps(e))
        hc = M.horizon_coherence(rec, {k: f for k, f in frags.items() if any(a['fragment'] == k for a in ag)})
        row['horizon'] = hc
        print('    horizon conditioning', hc.get('conditioning'))
        for e in hc['fragments']:
            print('    horizon', json.dumps(e))
        pc = M.plane_coherence(rec, {k: f for k, f in frags.items() if any(a['fragment'] == k for a in ag)})
        row['plane'] = pc
        print('    plane', json.dumps(pc['global_plane']))
        for e in pc['fragments']:
            print('     ', json.dumps(e))
    (cache / f'check_{tag}.json').write_text(json.dumps(out))


def align(run, cache, sparse):
    """Pairwise pycolmap.align_reconstructions_via_reprojections between the models of `sparse` that share registered images."""
    import pycolmap
    models = load_models(Path(cache) / sparse)
    keys = sorted(models, key=lambda k: -models[k].num_reg_images())
    for a_i, a in enumerate(keys):
        na = set(M._reg_names(models[a]))
        for b in keys[a_i + 1:]:
            shared = na & set(M._reg_names(models[b]))
            if not shared:
                continue
            sim = pycolmap.align_reconstructions_via_reprojections(models[b], models[a], min_inlier_observations=0.3, max_reproj_error=8.0)
            print(json.dumps({'tgt': a, 'src': b, 'shared': len(shared), 'sim': None if sim is None else {'scale': round(sim.scale, 4)}}))


def chain(run, cache, sparse, db_name, link_min=None):
    """chain_models on the models of `sparse` with the database `db_name`; prints every link and the joined set."""
    cache = Path(cache)
    names = _names(_frames(run))
    models = load_models(cache / sparse)
    t0 = time.perf_counter()
    if link_min:
        M.LINK_MIN_INLIERS = link_min
    members, links = M.chain_models(models, cache / db_name, names)
    secs = time.perf_counter() - t0
    for ev in links:
        print(json.dumps(ev))
    union = M._union(models, members)
    res = {'sparse': sparse, 'models': len(models), 'joined': [int(k) for k, _ in members], 'registered_union': len(union),
           'largest': max(m.num_reg_images() for m in models.values()), 'seconds': round(secs, 1)}
    print(json.dumps(res))
    (cache / f'chain_{sparse}.json').write_text(json.dumps({'result': res, 'links': links}))


def cross(run, cache, db_name, sparse_a='sparse_seq', sparse_b='sparse_t1'):
    """The two routes (chain of the original fragments / re-mapped model + its chain) compared on their shared frames."""
    cache = Path(cache)
    names = _names(_frames(run))
    out = {}
    for s in (sparse_a, sparse_b):
        models = load_models(cache / s)
        members, links = M.chain_models(models, cache / db_name, names)
        out[s] = M.joined_centres(models, members)
        print(s, 'joined', [int(k) for k, _ in members], len(out[s]))
    sim, ev = M.estimate_link(out[sparse_a], out[sparse_b])
    print(json.dumps(ev))


def chainplane(run, cache, sparse, db_name):
    """Shelf-plane coherence of a chained (similarity-joined) set: per member the share of its points on the joint plane."""
    from pipeline.recon3d import fit_plane
    cache = Path(cache)
    names = _names(_frames(run))
    models = load_models(cache / sparse)
    members, _ = M.chain_models(models, cache / db_name, names)
    clouds = {}
    for k, (s, r, t) in members:
        p = M._all_points(models[k])
        clouds[k] = s * p @ r.T + t
    allp = np.concatenate(list(clouds.values()))
    g = fit_plane(allp)
    print('joint plane inlier ratio', round(g['inlier_ratio'], 3), 'points', len(allp))
    main = fit_plane(clouds[members[0][0]])
    for k, p in clouds.items():
        d = np.abs((p - g['point']) @ g['normal'])
        dm = np.abs((p - main['point']) @ main['normal'])
        print(json.dumps({'model': int(k), 'points': len(p), 'on_joint_plane': round(float((d <= g['threshold']).mean()), 3),
                          'on_main_plane': round(float((dm <= main['threshold']).mean()), 3)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cmd', choices=['base', 'diag', 'trial', 'check', 'align', 'chain', 'cross', 'chainplane'])
    p.add_argument('--run', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--tag', default='t1')
    p.add_argument('--mapper', default='incremental', choices=['incremental', 'continue', 'global'])
    p.add_argument('--window', type=int, default=M.END_WINDOW)
    p.add_argument('--ends', type=int, default=M.END_FRAMES)
    p.add_argument('--reps', type=int, default=M.MAX_REPRESENTATIVES)
    p.add_argument('--panorama', default=None)
    p.add_argument('--sparse', default='sparse_seq')
    p.add_argument('--link-min', type=int, default=None)
    p.add_argument('--reuse', default=None, help='tag of a trial database to start from (no new bridge pairs)')
    p.add_argument('--mapper-json', default=None, help='IncrementalPipelineOptions overrides; mapper.* / triangulation.* for nested')
    p.add_argument('--db', default='database_seq.db')
    a = p.parse_args()
    if a.cmd == 'base':
        base(a.run, a.cache)
    elif a.cmd == 'chain':
        chain(a.run, a.cache, a.sparse, a.db, a.link_min)
    elif a.cmd == 'chainplane':
        chainplane(a.run, a.cache, a.sparse, a.db)
    elif a.cmd == 'cross':
        cross(a.run, a.cache, a.db)
    elif a.cmd == 'align':
        align(a.run, a.cache, a.sparse)
    elif a.cmd == 'check':
        check(a.run, a.cache, a.tag)
    elif a.cmd == 'diag':
        diag(a.run, a.cache, a.sparse, a.db)
    else:
        trial(a.run, a.cache, a.tag, a.mapper, a.window, a.ends, a.reps, a.panorama, a.reuse, a.mapper_json)


if __name__ == '__main__':
    main()
