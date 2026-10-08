"""Automatic 2D panorama + 3D map after every finished sweep: order, waiting, failures, persistence/resume, engine hook, server status."""
import json
import os
import sys
import time

import pytest

from app import post_sweep as PS
from app.build_jobs import BuildJobs
from tests.test_build_jobs import make_root, make_run, SHA1

POLL = 0.03


def fake_command(failures=(), delay=0.0, log=None):
    """argv factory: writes the requested product (panorama.json / recon3d.json) into the new output dir, or fails with a reason."""
    def command(kind, source, out, label, registry):
        fail = kind in failures
        script = ("import sys, time, pathlib, json; kind, out, fail, delay, log = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3] == '1', float(sys.argv[4]), sys.argv[5];"
                  "time.sleep(delay);"
                  "open(log, 'a').write(kind + ':start\\n') if log else None;"
                  "print('RuntimeError: panorama failed while tracking: boom') if fail else None;"
                  "sys.exit(3) if fail else None;"
                  "out.mkdir(parents=True);"
                  "(out / ('recon3d.json' if kind == 'recon3d' else 'panorama.json')).write_text(json.dumps({'book_count': 7, 'markers_total': 7, 'registered_ratio': 1.0, 'metrics': {'registered_ratio': 1.0}}))")
        return [sys.executable, '-c', script, kind, str(out), '1' if fail else '0', str(delay), str(log or '')]
    return command


def wait_for(fn, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.03)
    raise AssertionError('condition not reached')


def builds(tmp_path, runs=('sweep_abc',), command=None, **kw):
    root = make_root(tmp_path, ())
    for name in runs:
        make_run(root, name, SHA1, frames=9)                                   # 3D needs >= 8 saved frames
    jobs = BuildJobs(root, command or fake_command())
    psb = PS.PostSweepBuilds(root, jobs, poll_s=POLL, **kw)
    return root, jobs, psb


def terminal(psb, run):
    return lambda: (psb.status(run) or {}).get('state') in ('done', 'failed') and psb.status(run)


def test_panorama_first_then_3d_each_a_versioned_niced_job_and_the_summary_names_the_marker_count(tmp_path):
    log = tmp_path / 'order.log'
    root, jobs, psb = builds(tmp_path, command=fake_command(log=log))
    rec = psb.enqueue('sweep_abc')
    assert rec['state'] in ('queued', 'waiting') and rec['scan_label'] == 'YugoTest1'
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['state'] == 'done' and st['note'] == '2D panorama and 3D map ready'
    assert log.read_text().split() == ['panorama:start', 'recon3d:start']
    assert st['steps']['panorama']['out_run'] == 'sweep_abc_panorama_v1' and st['steps']['recon3d']['out_run'] == 'sweep_abc_recon3d_v1'
    assert (root / 'runs/sweep_abc_panorama_v1/panorama.json').is_file() and (root / 'runs/sweep_abc_recon3d_v1/recon3d.json').is_file()
    assert st['steps']['panorama']['markers'] == st['steps']['panorama']['book_count'] == 7
    assert '2D panorama ready (7 markers = book_count 7)' in st['summary'] and '3D map ready' in st['summary']


def test_a_run_written_a_moment_ago_is_waited_for_not_refused(tmp_path):
    root, jobs, psb = builds(tmp_path)
    (root / 'runs/sweep_abc/inventory.json').write_text('{}')          # just written: the sweep is still settling
    psb.enqueue('sweep_abc')
    wait_for(lambda: (psb.status('sweep_abc') or {}).get('state') == 'waiting')
    assert 'settle' in psb.status('sweep_abc')['note'] and not (root / 'runs/sweep_abc_panorama_v1').exists()
    old = time.time() - 3600
    os.utime(root / 'runs/sweep_abc/inventory.json', (old, old))
    assert wait_for(terminal(psb, 'sweep_abc'))['state'] == 'done'
    psb.stop()


def test_it_waits_for_a_build_started_from_the_debugger_and_never_overlaps(tmp_path):
    root, jobs, psb = builds(tmp_path, runs=('sweep_abc', 'sweep_manual'), command=fake_command(delay=0.4))
    manual = jobs.start('sweep_manual', 'panorama')                    # the Debugger button
    psb.enqueue('sweep_abc')
    wait_for(lambda: 'running build' in (psb.status('sweep_abc') or {}).get('note', ''))
    first_start = {}
    for _ in range(300):
        running = [j for j in jobs.jobs() if j['state'] == 'running']
        assert len(running) <= 1                                       # one build at a time, always
        for j in jobs.jobs():
            first_start.setdefault(j['job_id'], j['started_at'])
        if (psb.status('sweep_abc') or {}).get('state') in ('done', 'failed'):
            break
        time.sleep(0.03)
    psb.stop()
    assert psb.status('sweep_abc')['state'] == 'done'
    assert first_start[manual['job_id']] < first_start[psb.status('sweep_abc')['steps']['panorama']['job_id']]


def test_a_failed_step_records_the_real_reason_and_does_not_stop_the_other_step(tmp_path):
    root, jobs, psb = builds(tmp_path, command=fake_command(failures=('panorama',)))
    psb.enqueue('sweep_abc')
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['state'] == 'failed'
    assert st['steps']['panorama']['state'] == 'failed' and 'boom' in st['steps']['panorama']['error']
    assert st['steps']['recon3d']['state'] == 'done'
    assert '2D panorama failed' in st['summary'] and '3D map ready' in st['summary'] and 'boom' in st['note']


def test_a_run_that_cannot_be_built_fails_with_the_eligibility_reason(tmp_path):
    root = make_root(tmp_path, ())
    make_run(root, 'sweep_tiny', SHA1, frames=1)
    psb = PS.PostSweepBuilds(root, BuildJobs(root, fake_command()), poll_s=POLL)
    psb.enqueue('sweep_tiny')
    st = wait_for(terminal(psb, 'sweep_tiny'))
    psb.stop()
    assert st['state'] == 'failed' and 'only 1 saved frame' in st['steps']['panorama']['error'] and 'at least 8' in st['steps']['recon3d']['error']


def test_enqueue_is_idempotent_while_active_and_rebuilds_after_a_finished_run(tmp_path):
    root, jobs, psb = builds(tmp_path, command=fake_command(delay=0.2))
    first = psb.enqueue('sweep_abc')
    again = psb.enqueue('sweep_abc')
    assert again['queued_at'] == first['queued_at']
    wait_for(terminal(psb, 'sweep_abc'))
    assert len([p for p in (root / 'runs').glob('sweep_abc_panorama_v*')]) == 1
    second = psb.enqueue('sweep_abc')                                   # End sweep pressed again after a failure/finish: a fresh record
    assert second['queued_at'] > first['queued_at']
    wait_for(lambda: (psb.status('sweep_abc') or {}).get('state') == 'done' and psb.status('sweep_abc')['queued_at'] == second['queued_at'])
    psb.stop()
    assert (root / 'runs/sweep_abc_panorama_v2/panorama.json').is_file()


def test_state_is_persisted_and_unfinished_runs_resume_after_a_restart(tmp_path):
    root, jobs, psb = builds(tmp_path)
    rec = psb._new_record('sweep_abc')
    rec.update(state='building_2d', note='building 2D panorama')
    rec['steps']['panorama'].update(state='running', job_id='lost', out_run='sweep_abc_panorama_v1', attempts=1)
    psb.dir.mkdir(parents=True)
    (psb.dir / 'sweep_abc.json').write_text(json.dumps(rec))
    fresh = PS.PostSweepBuilds(root, BuildJobs(root, fake_command()), poll_s=POLL)       # a new server process
    assert fresh.resume() == 2                                                           # both tasks of the run: the lost panorama job and the queued 3D map
    st = wait_for(terminal(fresh, 'sweep_abc'))
    fresh.stop()
    assert st['state'] == 'done' and st['note'] == '2D panorama and 3D map ready'
    assert json.loads((fresh.dir / 'sweep_abc.json').read_text())['state'] == 'done'      # history on disk
    assert PS.PostSweepBuilds(root, BuildJobs(root, fake_command()), poll_s=POLL).resume() == 0      # done records are not re-run


def test_a_build_that_keeps_getting_interrupted_stops_after_two_attempts(tmp_path):
    root, jobs, psb = builds(tmp_path)
    rec = psb._new_record('sweep_abc')
    rec['steps']['panorama'].update(state='pending', attempts=PS.MAX_ATTEMPTS)
    psb.dir.mkdir(parents=True)
    (psb.dir / 'sweep_abc.json').write_text(json.dumps(rec))
    assert psb.resume() == 2
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['steps']['panorama']['state'] == 'failed' and 'interrupted' in st['steps']['panorama']['error'] and st['steps']['recon3d']['state'] == 'done'


def test_disabled_queue_does_nothing_and_unknown_runs_have_no_status(tmp_path):
    root, jobs, psb = builds(tmp_path, enabled=False)
    assert psb.enqueue('sweep_abc') is None and psb.status('sweep_abc') is None
    assert not (root / 'runs/_server_state/post_sweep').exists()


def test_a_deleted_run_fails_with_a_reason_instead_of_hanging(tmp_path):
    root, jobs, psb = builds(tmp_path)
    psb.enqueue('sweep_gone')
    st = wait_for(terminal(psb, 'sweep_gone'))
    psb.stop()
    assert st['state'] == 'failed' and 'Unknown run' in st['steps']['panorama']['error']


# ------------------------------------------------------------------------------------------ engine hook + server
from tests.test_live_app import Harness, feed_panning          # noqa: E402
from tests.test_live_measurement import STAGES                  # noqa: E402


@pytest.mark.asyncio
async def test_engine_calls_the_hook_once_after_the_packet_and_report_exist(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        eng.config["report_pdf"] = False
        calls = []

        def hook(run_dir):
            calls.append((run_dir, (run_dir / 'claim_packet.json').is_file(), (run_dir / 'report.html').is_file(), eng.status))
        eng.on_finished = hook
        await feed_panning(h, 6)
        await h.wait(lambda: eng.inventory is not None)
        await eng.end_sweep()
        await eng.end_sweep()                                         # idempotent: no second trigger
        assert len(calls) == 1 and calls[0][1:] == (True, True, 'finished') and calls[0][0] == eng.run_dir


@pytest.mark.asyncio
async def test_a_failing_hook_never_fails_the_sweep(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        eng.config["report_pdf"] = False
        eng.on_finished = lambda run_dir: (_ for _ in ()).throw(RuntimeError('queue exploded'))
        await feed_panning(h, 6)
        await h.wait(lambda: eng.inventory is not None)
        summary = await eng.end_sweep()
        assert eng.status == 'finished' and summary['paths']['packet']


@pytest.mark.asyncio
async def test_server_wires_the_hook_and_exposes_the_status(tmp_path, monkeypatch):
    from app import server as s
    (tmp_path / 'site').mkdir()
    root = make_root(tmp_path / 'site', ())
    jobs = BuildJobs(root, fake_command())
    psb = PS.PostSweepBuilds(root, jobs, poll_s=POLL)
    monkeypatch.setattr(s, 'post_sweep', psb)
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        eng.config["report_pdf"] = False
        assert s._ui_state(h.session)['post_sweep'] is None            # not finished yet
        await feed_panning(h, 6)
        await h.wait(lambda: eng.inventory is not None)
        make_run(root, 'sess1', SHA1, frames=9)                       # the engine's run (the Harness names it sess1) as a saved run under the queue's root
        await eng.end_sweep()
        queued = psb.status('sess1')
        assert queued is not None and queued['scan_label'] == 'YugoTest1'
        wait_for(terminal(psb, 'sess1'))
        state = s._ui_state(h.session)
        assert state['status'] == 'finished' and state['post_sweep']['state'] == 'done' and state['post_sweep']['steps']['recon3d']['out_run'] == 'sess1_recon3d_v1'
        from fastapi.testclient import TestClient
        client = TestClient(s.app, headers={'host': '127.0.0.1:4177', 'origin': 'http://127.0.0.1:4177'}, base_url='http://127.0.0.1:4177')
        client.cookies.set('intake_owner', 'owner')
        body = client.get('/api/sessions/sess1/builds')
        assert body.status_code == 200 and body.json()['post_sweep']['state'] == 'done' and body.json()['enabled'] is True
    psb.stop()


# ------------------------------------------------------------------------------------------ manual builds share the queue
def test_a_second_manual_build_is_queued_behind_the_running_one_not_refused(tmp_path):
    """The reported bug: Build 3D, then Build 2D while it runs -> the 2D click was refused with 409 and nothing showed."""
    log = tmp_path / 'order.log'
    root, jobs, psb = builds(tmp_path, command=fake_command(delay=0.5, log=log))
    first = psb.request('sweep_abc', ['recon3d'])
    wait_for(lambda: (psb.status('sweep_abc') or {}).get('steps', {}).get('recon3d', {}).get('state') == 'running')
    second = psb.request('sweep_abc', ['panorama'])                                  # pressed while the 3D map is building
    assert second['steps']['panorama']['state'] == 'queued' and second['steps']['panorama']['position'] == 1
    assert second['steps']['panorama']['ahead'] == [{'run_id': 'sweep_abc', 'kind': 'recon3d'}]
    assert 'queued #1' in second['summary'] and 'building 3D map' in second['summary']
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['state'] == 'done' and st['steps']['recon3d']['state'] == 'done' and st['steps']['panorama']['state'] == 'done'
    assert log.read_text().split() == ['recon3d:start', 'panorama:start']             # FIFO: the 3D map first, then the panorama
    assert (root / 'runs/sweep_abc_panorama_v1/panorama.json').is_file() and (root / 'runs/sweep_abc_recon3d_v1/recon3d.json').is_file()


def test_requests_for_different_runs_are_served_in_order_with_positions(tmp_path):
    root, jobs, psb = builds(tmp_path, runs=('sweep_a', 'sweep_b', 'sweep_c'), command=fake_command(delay=0.3))
    psb.request('sweep_a', ['panorama'])
    wait_for(lambda: (psb.status('sweep_a') or {}).get('state') == 'building_2d')
    psb.request('sweep_b', ['recon3d'])
    c = psb.request('sweep_c', ['panorama'])
    assert psb.status('sweep_b')['steps']['recon3d']['position'] == 1 and c['steps']['panorama']['position'] == 2
    assert [a['run_id'] for a in c['steps']['panorama']['ahead']] == ['sweep_a', 'sweep_b']
    assert psb.queue_overview() == {'running': ['sweep_a', 'panorama'], 'queued': [['sweep_b', 'recon3d'], ['sweep_c', 'panorama']]}
    for run in ('sweep_a', 'sweep_b', 'sweep_c'):
        assert wait_for(terminal(psb, run))['state'] == 'done'
    psb.stop()


def test_a_manual_request_for_one_kind_leaves_the_other_alone_and_is_not_queued_twice(tmp_path):
    root, jobs, psb = builds(tmp_path, command=fake_command(delay=0.2))
    psb.request('sweep_abc', ['panorama'])
    again = psb.request('sweep_abc', ['panorama'])                                   # double click
    assert again['steps']['recon3d']['state'] == 'none'
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['steps']['panorama']['state'] == 'done' and st['steps']['recon3d']['state'] == 'none'
    assert len(list((root / 'runs').glob('sweep_abc_panorama_v*'))) == 1 and not list((root / 'runs').glob('sweep_abc_recon3d_v*'))
    assert st['summary'] == '2D panorama ready (7 markers = book_count 7)' and st['note'] == '2D panorama ready'


def test_a_rebuild_request_after_a_finished_build_makes_a_new_version(tmp_path):
    root, jobs, psb = builds(tmp_path)
    psb.request('sweep_abc', ['panorama'])
    wait_for(terminal(psb, 'sweep_abc'))
    psb.request('sweep_abc', ['panorama'])
    wait_for(lambda: (root / 'runs/sweep_abc_panorama_v2/panorama.json').is_file())
    psb.stop()


def test_a_run_that_can_never_be_built_is_refused_with_the_reason_and_nothing_is_queued(tmp_path):
    root, jobs, psb = builds(tmp_path)
    make_run(root, 'sweep_tiny', SHA1, frames=1)
    with pytest.raises(ValueError, match='only 1 saved frame'):
        psb.request('sweep_tiny', ['panorama'])
    with pytest.raises(ValueError, match='at least 8'):
        psb.request('sweep_tiny', ['recon3d'])
    with pytest.raises(ValueError, match='Unknown run'):
        psb.request('sweep_missing', ['panorama'])
    with pytest.raises(ValueError, match='build kind'):
        psb.request('sweep_abc', ['video'])
    assert psb.status('sweep_tiny') is None and psb.queue_overview() == {'running': None, 'queued': []}
    psb.stop()


def test_a_run_still_being_written_is_accepted_and_waited_for(tmp_path):
    root, jobs, psb = builds(tmp_path)
    (root / 'runs/sweep_abc/inventory.json').write_text('{}')
    st = psb.request('sweep_abc', ['panorama'])                                      # not refused
    assert st['steps']['panorama']['state'] in ('queued', 'waiting')
    wait_for(lambda: psb.status('sweep_abc')['steps']['panorama']['state'] == 'waiting')
    old = time.time() - 3600
    os.utime(root / 'runs/sweep_abc/inventory.json', (old, old))
    assert wait_for(terminal(psb, 'sweep_abc'))['state'] == 'done'
    psb.stop()


def test_manual_requests_work_even_when_the_automatic_post_sweep_build_is_off(tmp_path):
    root, jobs, psb = builds(tmp_path, enabled=False)
    assert psb.enqueue('sweep_abc') is None
    psb.request('sweep_abc', ['panorama'])
    assert wait_for(terminal(psb, 'sweep_abc'))['state'] == 'done'
    psb.stop()


def deferred_command(delay=0.0):
    """argv factory for the two-stage 3D map: the first 3D build publishes a map whose merge was deferred (merge.deferred +
    _merge_cache/merge_state.json); the merge pass (--merge-from) writes the joined map as the next version and deletes the cache."""
    def command(kind, source, out, label, registry, merge_from=None):
        script = ("import sys, time, pathlib, json, shutil; kind, out, first, delay = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]);"
                  "time.sleep(delay);"
                  "out.mkdir(parents=True);"
                  "deferred = kind == 'recon3d' and not first;"
                  "ratio = 0.8 if first else 0.3;"
                  "(out / ('recon3d.json' if kind == 'recon3d' else 'panorama.json')).write_text(json.dumps({'book_count': 7, 'markers_total': 7, 'registered_ratio': ratio, 'metrics': {'registered_ratio': ratio},"
                  " 'merge': {'deferred': {'steps': ['bridge_pairs'], 'cache': '_merge_cache'} if deferred else None, 'stage': 'merge_pass' if first else 'first_map'}}));"
                  "(out / '_merge_cache').mkdir() if deferred else None;"
                  "(out / '_merge_cache' / 'merge_state.json').write_text('{}') if deferred else None;"
                  "shutil.rmtree(pathlib.Path(first) / '_merge_cache') if first else None")
        return [sys.executable, '-c', script, kind, str(out), str(merge_from or ''), str(delay)]
    return command


def test_a_deferred_first_map_is_measured_then_joined_by_a_merge_pass_that_is_measured_again(tmp_path):
    measured, seen = [], set()
    root, jobs, psb = builds(tmp_path, command=deferred_command(delay=0.3), on_built=lambda run, out: measured.append(out) or {'summary': out})

    def done():
        st = psb.status('sweep_abc') or {}
        seen.add(st.get('state'))
        return st.get('state') == 'done' and (st.get('merge3d') or {}).get('state') == 'done' and st
    psb.enqueue('sweep_abc')
    st = wait_for(done)
    psb.stop()
    assert 'joining_3d' in seen                                         # the client labels this state (app.js POST_SWEEP_ACTIVE)
    assert measured == ['sweep_abc_recon3d_v1', 'sweep_abc_recon3d_v2']    # the first map is measured at once, the joined map after the pass
    assert st['merge3d']['first_map'] == 'sweep_abc_recon3d_v1' and st['merge3d']['out_run'] == 'sweep_abc_recon3d_v2'
    assert st['steps']['recon3d']['out_run'] == 'sweep_abc_recon3d_v2' and st['steps']['recon3d']['first_map_out_run'] == 'sweep_abc_recon3d_v1'
    assert not (root / 'runs/sweep_abc_recon3d_v1/_merge_cache').exists() and (root / 'runs/sweep_abc_recon3d_v1/recon3d.json').is_file()
    assert 'joined map ready (sweep_abc_recon3d_v2' in st['summary'] and st['overall']['fraction'] == 1.0
