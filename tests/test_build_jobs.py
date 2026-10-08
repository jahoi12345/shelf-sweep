"""Debugger 'Build 3D' / 'Build panorama': eligibility, versioned output, background lifecycle, featured visibility."""
import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import debug_server, run_visibility as RV
from app.debug_views import DebugStore, UNREGISTERED
from app.build_jobs import BuildJobs

SHA1, SHA2 = 'a' * 64, 'b' * 64


def make_root(tmp_path, runs=('sweep_abc',)):
    (tmp_path / 'sweeps').mkdir()
    (tmp_path / 'sweeps/scan_registry.json').write_text(json.dumps({'scans': [
        {'label': 'YugoTest1', 'file': 'YugoTest1.MOV', 'sha256': SHA1, 'original_name': 'x', 'aliases': []},
        {'label': 'YugoTest2', 'file': 'YugoTest2.MOV', 'sha256': SHA2, 'original_name': 'y', 'aliases': []}]}))
    for label in ('YugoTest1', 'YugoTest2'):
        (tmp_path / 'sweeps' / f'{label}.MOV').write_bytes(b'x')
    for name in runs:
        make_run(tmp_path, name, SHA1)
    return tmp_path


def make_run(root, name, sha, frames=3):
    """A saved run with a manifest, saved frame images and candidates; files are 'old' so the live-sweep quiet period passes."""
    d = root / 'runs' / name
    (d / 'frames').mkdir(parents=True)
    listed = []
    for i in range(frames):
        (d / 'frames' / f'f{i:05d}.jpg').write_bytes(b'x')
        listed.append({'frame_id': f'f{i:05d}', 'image_path': f'frames/f{i:05d}.jpg', 'timestamp_s': i * .2, 'width_px': 64, 'height_px': 48})
    (d / 'manifest.json').write_text(json.dumps({'data': {'input': {'sha256': sha}, 'frames': listed}}))
    (d / 'candidates.json').write_text('{}')
    old = time.time() - 3600
    for f in d.iterdir():
        if f.is_file():
            os.utime(f, (old, old))
    return d


def fake(code, text='working'):
    """argv that exits `code`; on success writes panorama.json into the requested (new) output dir."""
    def command(kind, source, out, label, registry):
        script = ("import sys, pathlib, json; out = pathlib.Path(sys.argv[1]); print(sys.argv[3]);"
                  "(out.mkdir(parents=True) or (out / 'panorama.json').write_text('{}')) if sys.argv[2] == '0' else None;"
                  "sys.exit(int(sys.argv[2]))")
        return [sys.executable, '-c', script, str(out), str(code), f'{text} {label} {source.name}']
    return command


def wait(jobs, state='running', timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        listing = jobs.jobs()
        if listing and listing[0]['state'] != state:
            return listing[0]
        time.sleep(0.05)
    raise AssertionError('job did not finish')


def test_successful_build_writes_new_versioned_dir_and_never_reuses_names(tmp_path):
    root = make_root(tmp_path)
    jobs = BuildJobs(root, fake(0))
    first = jobs.start('sweep_abc')
    assert first['out_run'] == 'sweep_abc_panorama_v1' and first['scan'] == 'YugoTest1'
    done = wait(jobs)
    assert done['state'] == 'done' and (root / 'runs/sweep_abc_panorama_v1/panorama.json').is_file()
    assert 'YugoTest1 sweep_abc' in done['log_tail']
    assert jobs.start('sweep_abc')['out_run'] == 'sweep_abc_panorama_v2'   # v1 is never overwritten
    wait(jobs)


def test_failure_is_reported_with_log_tail(tmp_path):
    root = make_root(tmp_path)
    jobs = BuildJobs(root, fake(3, 'registration exploded'))
    jobs.start('sweep_abc')
    job = wait(jobs)
    assert job['state'] == 'failed' and job['exit_code'] == 3
    assert 'registration exploded' in job['error']
    assert not (root / 'runs/sweep_abc_panorama_v1/panorama.json').exists()


def test_only_one_build_at_a_time(tmp_path):
    root = make_root(tmp_path, ('sweep_abc', 'sweep_def'))
    slow = lambda kind, source, out, label, registry: [sys.executable, '-c', 'import time; time.sleep(30)']
    jobs = BuildJobs(root, slow)
    job = jobs.start('sweep_abc')
    with pytest.raises(ValueError, match='already running'):
        jobs.start('sweep_def')
    import os, signal
    os.killpg(job['pid'], signal.SIGKILL)
    assert wait(jobs)['state'] == 'failed'


@pytest.mark.parametrize('run_id', ['../sweeps', 'a/b', '', None, 'missing_run'])
def test_rejects_bad_run_ids(tmp_path, run_id):
    with pytest.raises(ValueError):
        BuildJobs(make_root(tmp_path), fake(0)).start(run_id)


def test_eligibility_reasons(tmp_path):
    root = make_root(tmp_path, ('sweep_abc', 'sweep_pano', 'sweep_fresh'))
    make_run(root, 'sweep_live', 'c' * 64)                          # live camera sweep: no registered recording
    make_run(root, 'sweep_one', SHA1, frames=1)
    (root / 'runs/sweep_pano/panorama.json').write_text('{}')
    (root / 'runs/sweep_fresh/manifest.json').touch()               # written just now: a sweep still processing
    jobs = BuildJobs(root, fake(0))
    assert jobs.eligible('sweep_abc')[1] == 'YugoTest1'
    assert jobs.eligible('sweep_live') == (root / 'runs/sweep_live', None)      # no registry hash needed
    (root / 'sweeps/YugoTest1.MOV').unlink()
    assert jobs.eligible('sweep_abc')[1] == 'YugoTest1'            # the recording itself is not needed either
    for run, text in [('sweep_one', 'only 1 saved frame'), ('sweep_pano', 'itself a panorama'), ('sweep_fresh', 'still processing')]:
        with pytest.raises(ValueError, match=text):
            jobs.eligible(run)
    with pytest.raises(ValueError, match='at least 8'):
        jobs.eligible('sweep_abc', 'recon3d')
    (root / 'runs/sweep_abc/manifest.json').write_text('not json')
    with pytest.raises(ValueError, match='manifest'):
        jobs.eligible('sweep_abc')


def test_failure_reason_names_the_stage_and_the_exception(tmp_path):
    root = make_root(tmp_path)
    jobs = BuildJobs(root, fake(1, 'RuntimeError: panorama failed while tracking: ValueError: boom'))
    jobs.start('sweep_abc')
    job = wait(jobs)
    assert job['state'] == 'failed' and 'boom' in job['error']


def patched_queue(tmp_path, monkeypatch, runs=('sweep_abc',), command=None, **kw):
    from app.post_sweep import PostSweepBuilds
    root = make_root(tmp_path, ())
    for name in runs:
        make_run(root, name, SHA1, frames=9)
    jobs = BuildJobs(root, command or fake(0))
    queue = PostSweepBuilds(root, jobs, poll_s=0.03, **kw)
    monkeypatch.setattr(debug_server, 'build_jobs', jobs)
    monkeypatch.setattr(debug_server, 'build_queue', queue)
    return root, jobs, queue


def test_http_endpoints_queue_builds_and_report_status(tmp_path, monkeypatch):
    root, jobs, queue = patched_queue(tmp_path, monkeypatch)
    client = TestClient(debug_server.app)
    assert client.get('/api/panorama/jobs').json() == {'jobs': []}
    assert client.get('/api/builds/sweep_abc').json()['status'] is None
    assert client.get('/api/builds/..%2Fx').status_code in (404, 422)
    assert client.post('/api/panorama/build', data='run_id=sweep_abc').status_code == 422   # JSON body only
    refused = client.post('/api/panorama/build', json={'run_id': 'nope'})
    assert refused.status_code == 409 and 'Unknown run' in refused.json()['detail']
    started = client.post('/api/panorama/build', json={'run_id': 'sweep_abc'})
    body = started.json()
    assert started.status_code == 200 and body['queued'] is True and body['status']['steps']['panorama']['state'] in ('queued', 'waiting', 'running')
    deadline = time.time() + 15
    while time.time() < deadline and client.get('/api/builds/sweep_abc').json()['status']['state'] != 'done':
        time.sleep(0.05)
    status = client.get('/api/builds/sweep_abc').json()['status']
    assert status['state'] == 'done' and status['steps']['panorama']['out_run'] == 'sweep_abc_panorama_v1' and status['steps']['recon3d']['state'] == 'none'
    assert client.get('/api/panorama/jobs').json()['jobs'][0]['state'] == 'done'
    queue.stop()


def test_pressing_build_while_another_build_runs_is_queued_not_a_409(tmp_path, monkeypatch):
    """Reported live: Build 3D, then Build 2D on the same run -> the second press used to fail with 409 'already running' and nothing showed."""
    root, jobs, queue = patched_queue(tmp_path, monkeypatch, command=fake_slow(0.6))
    client = TestClient(debug_server.app)
    first = client.post('/api/panorama/build', json={'run_id': 'sweep_abc', 'kind': 'recon3d'})
    assert first.status_code == 200
    deadline = time.time() + 10
    while time.time() < deadline and client.get('/api/builds/sweep_abc').json()['status']['steps']['recon3d']['state'] != 'running':
        time.sleep(0.03)
    second = client.post('/api/panorama/build', json={'run_id': 'sweep_abc', 'kind': 'panorama'})
    assert second.status_code == 200 and second.json()['status']['steps']['panorama']['state'] == 'queued'
    assert second.json()['status']['steps']['panorama']['position'] == 1
    assert second.json()['queue']['running'] == ['sweep_abc', 'recon3d'] and 'queued #1' in second.json()['status']['summary']
    deadline = time.time() + 15
    while time.time() < deadline and client.get('/api/builds/sweep_abc').json()['status']['state'] != 'done':
        time.sleep(0.05)
    done = client.get('/api/builds/sweep_abc').json()['status']
    assert done['steps']['recon3d']['state'] == 'done' and done['steps']['panorama']['state'] == 'done'
    queue.stop()


def fake_slow(delay):
    def command(kind, source, out, label, registry):
        script = ("import sys, time, pathlib, json; kind, out, delay = sys.argv[1], pathlib.Path(sys.argv[2]), float(sys.argv[3]); time.sleep(delay);"
                  "out.mkdir(parents=True); (out / ('recon3d.json' if kind == 'recon3d' else 'panorama.json')).write_text(json.dumps({'book_count': 3, 'markers_total': 3}))")
        return [sys.executable, '-c', script, kind, str(out), str(delay)]
    return command


def test_panorama_outputs_of_featured_runs_are_visible_by_default(tmp_path):
    for name in ['sweep_abc_panorama_v1', 'sweep_abc_pano_v1', 'final_yt1_v1_panorama_v2', 'offline_baseline_panorama_v1', 'final_yt1_v1_recon3d_v1', 'sweep_abc_recon3d_v12']:
        assert RV.featured(name, tmp_path, set()), name
    assert not RV.featured('final_candidate_yt1_v1_panorama_v1', tmp_path, set())
    assert not RV.featured('dev_experiment_panorama_v1', tmp_path, set())
    assert not RV.featured('final_candidate_yt1_v1_recon3d_v1', tmp_path, set())
    assert RV.featured('my_pinned_panorama_v3', tmp_path, {'my_pinned'})


def test_catalogue_lists_new_panorama_run_and_attaches_it_to_its_source_run(tmp_path):
    root = make_root(tmp_path)
    for n in (1, 2):
        out = root / f'runs/sweep_abc_panorama_v{n}'
        out.mkdir()
        (out / 'panorama.json').write_text(json.dumps({'status': 'partial', 'source_video_sha256': SHA1, 'source_run': 'sweep_abc', 'components': []}))
    runs = {r['run_id']: r for s in DebugStore(root).catalogue(show_all=False) if s['label'] == 'YugoTest1' for r in s['runs']}
    assert runs['sweep_abc_panorama_v2']['has_panorama'] and runs['sweep_abc_panorama_v2']['source_run'] == 'sweep_abc'
    assert runs['sweep_abc']['panorama_run'] == 'sweep_abc_panorama_v2' and runs['sweep_abc']['has_panorama']   # newest version
    assert not runs['sweep_abc']['is_panorama_output'] and runs['sweep_abc_panorama_v1']['is_panorama_output']
    assert DebugStore(root).latest_panorama('sweep_abc').name == 'sweep_abc_panorama_v2'


def test_unregistered_live_runs_are_listed_and_viewable(tmp_path):
    root = make_root(tmp_path)
    make_run(root, 'sweep_live', 'f' * 64)
    store = DebugStore(root)
    other = next(s for s in store.catalogue(show_all=False) if s['label'] == UNREGISTERED)
    assert [r['run_id'] for r in other['runs']] == ['sweep_live'] and other['browser_preview'] is None
    view = store.view(UNREGISTERED, 'sweep_live')
    assert view['scan']['label'] == UNREGISTERED and view['panorama_run'] is None
    with pytest.raises(ValueError, match='registered scan'):
        store.view(UNREGISTERED, 'sweep_abc')


def test_recon_list_puts_finished_builds_first_so_the_default_is_the_newest_finished_one(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.setattr(debug_server, 'ROOT', root)
    done = root / 'runs/sweep_abc_recon3d_v1'
    done.mkdir()
    (done / 'recon3d.json').write_text(json.dumps({'status': 'ok', 'registered_frames': 5, 'frames_total': 5, 'registered_ratio': 1.0}))
    building = root / 'runs/sweep_abc_recon3d_v2'                    # newer, still building: no recon3d.json yet
    building.mkdir()
    client = TestClient(debug_server.app)
    ids = [r['recon_id'] for r in client.get('/api/recon3d/sweep_abc').json()['recon']]
    assert ids == ['sweep_abc_recon3d_v1', 'sweep_abc_recon3d_v2']


# ------------------------------------------------------------------------------------------ progress of running builds
def running_job(jobs, root, kind, log_lines, stage, elapsed=30.0, progress_extra=None):
    out = 'sweep_abc_recon3d_v1' if kind == 'recon3d' else 'sweep_abc_panorama_v1'
    (root / 'runs' / out).mkdir(parents=True)
    (root / 'runs' / out / 'progress.json').write_text(json.dumps({'stage': stage, **(progress_extra or {})}))
    job = {'job_id': 'jobx', 'kind': kind, 'source_run': 'sweep_abc', 'out_run': out, 'state': 'running', 'started_at': time.time() - elapsed, 'pid': 1}
    jobs.state.mkdir(parents=True, exist_ok=True)
    (jobs.state / 'jobx.log').write_text('\n'.join(log_lines) + '\n')
    return job


def test_3d_progress_comes_from_the_stage_and_the_pycolmap_log(tmp_path):
    root = make_root(tmp_path, ())
    jobs = BuildJobs(root, fake(0))
    features = [f'I2026 feature_extraction.cc:270] Processed file [{i}/60]' for i in range(1, 31)]
    job = running_job(jobs, root, 'recon3d', features, 'features')
    p = jobs.progress(job)
    assert 0.0 < p['fraction'] < 0.14 and p['label'] == 'extracting features 30/60' and (p['done'], p['total']) == (30, 60) and p['eta_s'] > 0
    matching = features + ['I2026 feature_matching.cc:196] === Feature matching & geometric verification ==='] + ['I2026 feature_matching.cc:218] in 0.04s'] * 30
    (jobs.state / 'jobx.log').write_text('\n'.join(matching) + '\n')
    (root / 'runs/sweep_abc_recon3d_v1/progress.json').write_text(json.dumps({'stage': 'matching'}))
    p2 = jobs.progress(job)                                                      # incremental: only the new lines are read
    assert p2['label'] == 'matching frames 30/60' and abs(p2['fraction'] - (0.14 + 0.085)) < 1e-6 and p2['fraction'] > p['fraction']
    mapping = matching + [f'I2026 incremental_pipeline.cc:620] Registering image #{k} (num_reg_frames={k})' for k in range(2, 46)]
    (jobs.state / 'jobx.log').write_text('\n'.join(mapping) + '\n')
    (root / 'runs/sweep_abc_recon3d_v1/progress.json').write_text(json.dumps({'stage': 'mapping'}))
    p3 = jobs.progress(job)
    assert p3['label'] == 'registering frames (SfM) 45/60' and p3['fraction'] > 0.31 + 0.41 * 0.7 and p3['fraction'] < 0.72 and p3['eta_s'] >= 2
    (root / 'runs/sweep_abc_recon3d_v1/progress.json').write_text(json.dumps({'stage': 'merging'}))     # only when the map splits
    p4 = jobs.progress(job)
    assert p4['label'] == 'joining 3D fragments' and p4['fraction'] == pytest.approx(0.72)
    (root / 'runs/sweep_abc_recon3d_v1/progress.json').write_text(json.dumps({'stage': 'markers'}))
    assert jobs.progress(job)['fraction'] == pytest.approx(0.92)
    assert jobs.progress(dict(job, state='done'))['fraction'] == 1.0 and jobs.progress(dict(job, state='done'))['eta_s'] == 0


def test_panorama_progress_uses_the_fraction_the_build_writes_and_an_eta(tmp_path):
    root = make_root(tmp_path, ())
    jobs = BuildJobs(root, fake(0))
    job = running_job(jobs, root, 'panorama', [], 'aligning: matching', elapsed=10.0, progress_extra={'fraction': 0.4, 'label': 'aligning frames: matching 30/60', 'done': 30, 'total': 60})
    p = jobs.progress(job)
    assert p['fraction'] == 0.4 and p['label'].startswith('aligning frames') and 10 <= p['eta_s'] <= 20          # 10 s elapsed at 40%  ->  ~15 s left


def test_status_carries_step_progress_and_an_overall_bar(tmp_path):
    from app.post_sweep import PostSweepBuilds
    root = make_root(tmp_path, ())
    make_run(root, 'sweep_abc', SHA1, frames=9)
    jobs = BuildJobs(root, fake_slow(1.5))
    queue = PostSweepBuilds(root, jobs, poll_s=0.03)
    queue.request('sweep_abc', ['panorama', 'recon3d'])
    deadline = time.time() + 10
    while time.time() < deadline and queue.status('sweep_abc')['steps']['panorama']['state'] != 'running':
        time.sleep(0.03)
    st = queue.status('sweep_abc')
    assert 'progress' in st['steps']['panorama'] and st['steps']['panorama']['progress']['eta_s'] >= 2
    assert st['steps']['recon3d']['state'] == 'queued' and st['steps']['recon3d']['position'] == 1
    assert 0.0 <= st['overall']['fraction'] < 0.25 and st['overall']['eta_s'] > 0                                # the panorama is ~1/4 of the work
    deadline = time.time() + 20
    while time.time() < deadline and queue.status('sweep_abc')['state'] != 'done':
        time.sleep(0.05)
    done = queue.status('sweep_abc')
    queue.stop()
    assert done['overall'] == {'fraction': 1.0, 'eta_s': 0}
