"""The post-sweep measurement step: runs after the 3D map, never fails the queue, records its outcome; app.measurement.finalize_run on a synthetic finished run."""
import json

import pytest

from app import post_sweep as PS
from app.build_jobs import BuildJobs
from app.measurement import finalize_run
from pipeline import contracts as C
from tests.test_auto_attach import make_run as make_packet_run
from tests.test_post_sweep import POLL, builds, fake_command, terminal, wait_for


def test_the_measurement_step_runs_after_the_3d_map_and_is_recorded(tmp_path):
    calls = []
    root, jobs, psb = builds(tmp_path, on_built=lambda run_id, recon: calls.append((run_id, recon)) or {"summary": "13/14 spines scaled (standard_format_prior); room abstained"})
    psb.enqueue('sweep_abc')
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert calls == [('sweep_abc', 'sweep_abc_recon3d_v1')] and st['state'] == 'done'
    assert st['measure']['state'] == 'done' and 'measurement: 13/14 spines scaled' in st['summary'] and '13/14' in st['note']


def test_a_failing_measurement_is_reported_and_does_not_fail_the_maps(tmp_path):
    def boom(run_id, recon):
        raise RuntimeError('catalogue exploded')
    root, jobs, psb = builds(tmp_path, on_built=boom)
    psb.enqueue('sweep_abc')
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert st['state'] == 'done' and st['measure']['state'] == 'failed' and 'catalogue exploded' in st['measure']['error']
    assert 'measurement: failed' in st['summary']


def test_no_3d_map_means_no_measurement(tmp_path):
    calls = []
    root, jobs, psb = builds(tmp_path, command=fake_command(failures=('recon3d',)), on_built=lambda *a: calls.append(a) or {})
    psb.enqueue('sweep_abc')
    st = wait_for(terminal(psb, 'sweep_abc'))
    psb.stop()
    assert calls == [] and st['state'] == 'failed' and st['measure']['state'] == 'skipped'


def test_finalize_run_writes_a_new_measure_dir_and_patches_the_packet(tmp_path):
    run, man, rc = make_packet_run(tmp_path)
    runs = run.parent
    recon_dir = runs / "run_recon3d_v1"
    recon_dir.mkdir()
    (recon_dir / "recon3d.json").write_text(json.dumps(rc))
    res = finalize_run(runs, "run", "run_recon3d_v1", online=False)
    assert res["attached"] is True and res["measure_dir"] == "run_measure_v1" and res["scale_source"] == "standard_format_prior" and "spines scaled" in res["summary"]
    assert (runs / "run_measure_v1" / "auto_measure.json").is_file()
    assert C.read_json(run / "claim_packet.json")["data"]["books"][0]["spine_height_cm"] is not None
    res2 = finalize_run(runs, "run", "run_recon3d_v1", online=False)                 # a second pass makes run_measure_v2, never overwrites v1
    assert res2["measure_dir"] == "run_measure_v2" and (runs / "run_measure_v1" / "auto_measure.json").is_file()
