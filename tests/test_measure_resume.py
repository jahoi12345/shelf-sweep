"""A server restart never costs a build attempt, a map that finished meanwhile is adopted (never rebuilt) and measured, the
`python -m app.measurement --run` CLI, and the Measurements-tab view (plain-words reasons, null never 0)."""
import json

from app import measurement as M
from app import post_sweep as PS
from app.build_jobs import BuildJobs
from pipeline import contracts as C
from tests.test_auto_attach import make_run as make_packet_run
from tests.test_post_sweep import POLL, builds, fake_command, terminal, wait_for


def _write(psb, rec):
    psb.dir.mkdir(parents=True, exist_ok=True)
    (psb.dir / f"{rec['run_id']}.json").write_text(json.dumps(rec))


def _map(root, name):
    out = root / 'runs' / name
    out.mkdir(parents=True)
    (out / 'recon3d.json').write_text(json.dumps({'book_count': 7, 'markers_total': 7, 'registered_ratio': 1.0}))


def test_a_3d_map_that_finished_during_restarts_is_adopted_and_measured_not_failed_or_rebuilt(tmp_path):
    """The 22:08 case: the step was marked failed after restarts although its recon3d.json was written; measurement 'skipped'."""
    calls = []
    root, jobs, psb = builds(tmp_path, on_built=lambda run_id, recon: calls.append((run_id, recon)) or {'summary': 'measured'})
    _map(root, 'sweep_abc_recon3d_v2')
    rec = psb._new_record('sweep_abc')
    rec['steps']['panorama'] = {'state': 'done', 'attempts': 1, 'out_run': 'sweep_abc_panorama_v1'}
    rec['steps']['recon3d'] = {'state': 'failed', 'attempts': 3, 'job_id': 'gone', 'out_run': 'sweep_abc_recon3d_v2',
                               'error': 'the build was interrupted 2 times (server restarts); press Build again'}
    rec['measure'] = {'state': 'skipped', 'reason': 'no finished 3D map to measure'}
    rec['state'] = 'failed'
    _write(psb, rec)
    assert psb.resume() == 1                                            # only the measurement is queued
    st = wait_for(lambda: (psb.status('sweep_abc') or {}).get('measure', {}).get('state') == 'done' and psb.status('sweep_abc'))
    psb.stop()
    assert calls == [('sweep_abc', 'sweep_abc_recon3d_v2')]
    assert st['state'] == 'done' and st['steps']['recon3d']['state'] == 'done' and st['steps']['recon3d']['error'] is None
    assert not (root / 'runs/sweep_abc_recon3d_v3').exists()            # never rebuilt


def test_a_build_lost_with_its_process_is_started_again_and_only_the_new_start_counts(tmp_path):
    root, jobs, psb = builds(tmp_path, on_built=lambda *a: {'summary': 'ok'})
    rec = psb._new_record('sweep_abc')
    rec['steps']['panorama'] = {'state': 'done', 'attempts': 1, 'out_run': 'sweep_abc_panorama_v1'}
    rec['steps']['recon3d'].update(state='running', job_id='lost', out_run='sweep_abc_recon3d_v1', attempts=1)
    _write(psb, rec)
    assert psb.resume() == 1
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['state'] == 'done' and st['steps']['recon3d']['attempts'] == 2 and st['measure']['state'] == 'done'


def test_a_build_that_outlives_the_server_is_followed_without_a_new_attempt(tmp_path):
    root, jobs, psb = builds(tmp_path, command=fake_command(delay=0.6), on_built=lambda *a: {'summary': 'ok'})
    job = jobs.start('sweep_abc', 'recon3d')                            # started by the previous server process
    rec = psb._new_record('sweep_abc')
    rec['steps']['panorama'] = {'state': 'done', 'attempts': 1}
    rec['steps']['recon3d'].update(state='running', job_id=job['job_id'], out_run=job['out_run'], attempts=1)
    _write(psb, rec)
    assert psb.resume() == 1
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['state'] == 'done' and st['steps']['recon3d']['attempts'] == 1 and st['steps']['recon3d']['out_run'] == 'sweep_abc_recon3d_v1'
    assert st['measure']['state'] == 'done' and not (root / 'runs/sweep_abc_recon3d_v2').exists()


def test_an_old_finished_map_that_was_never_measured_is_measured_on_resume_once(tmp_path):
    calls = []
    root, jobs, psb = builds(tmp_path, on_built=lambda run_id, recon: calls.append(recon) or {'summary': 'ok'})
    _map(root, 'sweep_abc_recon3d_v1')
    rec = psb._new_record('sweep_abc')
    for k in PS.STEPS:
        rec['steps'][k] = {'state': 'done', 'attempts': 1, 'out_run': f'sweep_abc_{"panorama" if k == "panorama" else "recon3d"}_v1'}
    psb._recompute(rec)
    _write(psb, rec)
    assert psb.resume() == 1
    wait_for(lambda: (psb.status('sweep_abc') or {}).get('measure', {}).get('state') == 'done')
    psb.stop()
    again = PS.PostSweepBuilds(root, BuildJobs(root, fake_command()), poll_s=POLL, on_built=lambda *a: calls.append('again'))
    assert again.resume() == 0 and calls == ['sweep_abc_recon3d_v1']


def _packet_run(tmp_path):
    run, man, rc = make_packet_run(tmp_path)
    runs = run.parent
    for v in (1, 2):
        (runs / f"run_recon3d_v{v}").mkdir()
    (runs / "run_recon3d_v2" / "recon3d.json").write_text(json.dumps(rc))      # v2 finished; v1 interrupted (no recon3d.json)
    return runs


def test_cli_measures_the_newest_finished_map_and_the_view_shows_plain_words(tmp_path):
    runs = _packet_run(tmp_path)
    assert M.latest_recon3d(runs, 'run') == 'run_recon3d_v2'
    view = M.auto_view(runs, 'run')
    assert view['available'] is False and 'python -m app.measurement --run run' in view['reason']
    assert M.main(['--run', str(runs / 'run'), '--offline']) == 0
    view = M.auto_view(runs, 'run')
    assert view['available'] and view['measure_dir'] == 'run_measure_v1' and view['attached_to_packet'] is True
    assert view['scale']['source'] == 'standard_format_prior' and 'PRIOR' in view['scale']['label']
    measured = [b for b in view['books'] if b['status'] == 'measured']
    assert measured and all(b['spine_height_cm'] > 0 and b['height_pm_cm'] > 0 for b in measured)
    for b in view['books']:
        assert b['spine_height_cm'] != 0 and b['spine_thickness_cm'] != 0
        if b['status'] != 'measured':
            assert b['spine_height_cm'] is None and b['reason']
        if b['status'] == 'measured' and b['spine_thickness_cm'] is None:
            assert b['thickness_reason']
    room = view['room']
    for q in [*room['dimensions'].values(), *room['areas'].values()]:
        assert (q['value'] is None) == bool(q['reason']) and q['value'] != 0
    assert [r['run_id'] for r in M.measured_runs(runs)] == ['run']


def test_plain_words_for_room_and_spine_reasons():
    assert M._words('floor_not_found:no_surface_below(floor_not_filmed)', M.ROOM_REASON_WORDS).startswith('floor line not filmed')
    assert M._words('ceiling_not_found:x', M.ROOM_REASON_WORDS).startswith('ceiling line not filmed')
    assert M._words('sightings_disagree_on_height:25%', M.SPINE_REASON_WORDS) == 'views disagree on the height (25%)'
    assert M._words('no_usable_sighting:polygon_clipped_by_frame_edge', M.SPINE_REASON_WORDS) == 'no usable view: cut off by the frame edge in every view'
    assert 'too narrow' in M._thickness_words('library_median_thickness_1.2cm_outside_1.5-6.0cm(spines_too_narrow_in_pixels_for_a_thickness)')
    assert M._pos(0) is None and M._pos(0.0) is None and M._pos(-1) is None and M._pos(2.5) == 2.5


def test_cli_without_a_finished_map_says_so(tmp_path, capsys):
    run, man, rc = make_packet_run(tmp_path)
    assert M.main(['--run', str(run)]) == 3 and 'no finished 3D map' in capsys.readouterr().out


def test_auto_measurement_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app import server as S
    runs = _packet_run(tmp_path)
    M.main(['--run', str(runs / 'run'), '--offline'])
    monkeypatch.setattr(S, 'RUNS_DIR', runs)
    client = TestClient(S.app, base_url="http://127.0.0.1:4177")
    assert client.get('/api/auto-measurements').json()['runs'][0]['run_id'] == 'run'
    body = client.get('/api/runs/run/auto-measurement').json()
    assert body['available'] and body['scale']['source'] == 'standard_format_prior'
    assert client.get('/api/runs/nope/auto-measurement').status_code == 404
    assert client.get('/api/runs/..%2Fetc/auto-measurement').status_code in (400, 404)


def test_attach_succeeds_when_thickness_is_withheld_run_wide(tmp_path):
    """Webcam sweeps (the 22:08 live run): spines too narrow for a thickness; the extent-based shelf run must still verify and attach."""
    from pipeline import auto_attach as AA
    from pipeline import auto_scale as AS
    runs = _packet_run(tmp_path)
    path = AS.run(runs / 'run', runs / 'run_recon3d_v2', runs / 'run_measure_v1', online=False)
    env = C.read_json(path)
    for s in env['data']['spines']:
        if s.get('status') == 'measured':
            s['spine_thickness_cm'] = None
            s['thickness_reason'] = 'library_median_thickness_1.2cm_outside_1.5-6.0cm(spines_too_narrow_in_pixels_for_a_thickness)'
    C.write_json(path, env)
    res = AA.attach(runs / 'run', path, render=False)
    assert res['attached'] is True, res
    books = C.read_json(runs / 'run' / 'claim_packet.json')['data']['books']
    assert any(b['spine_height_cm'] for b in books) and not any(b['spine_thickness_cm'] for b in books)
    assert all(b['measurement'].get('thickness_reason') for b in books if b['spine_height_cm'] is not None)
