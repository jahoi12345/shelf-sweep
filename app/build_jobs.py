"""Background "Build 3D" / "Build panorama" jobs for the debugger.

pipeline/recon3d.py (SfM) and pipeline/panorama.py are offline steps; saved site sweeps and final runs never get one
on their own. A job runs one as a niced subprocess for ONE saved run (any run with a manifest and saved frames: site sweeps, browser
replays, final runs, offline replays; no registered recording is needed) and writes a NEW versioned dir
(`runs/<run>_recon3d_v<N>` / `runs/<run>_panorama_v<N>`, never an existing one). Job records live in
runs/_server_state/panorama_jobs/ so they survive a page refresh. Nothing here edits pipeline/ or any saved run.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .debug_registry import read_registry

ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
STATE = Path('_server_state') / 'panorama_jobs'
TAIL_LINES = 12
# Progress of a RUNNING build: pipeline.panorama writes its own `fraction` into progress.json; pycolmap has no callbacks, so the 3D map's
# fraction comes from its stage plus counters parsed from the job log (files processed, image batches matched, frames registered).
WEIGHTS_3D = (('features', .14), ('matching', .17), ('mapping', .41), ('merging', .20), ('markers', .05), ('writing', .03))     # 60-100 frame runs; 'merging' only runs when the map splits
# The merge pass (`--merge-from`, after the first map is published) has no counters: its fraction comes from the merge step it reports.
MERGE_PASS_STEPS = {'loading the first build': .05, 'matching bridge pairs': .15, 're-mapping with the bridge pairs': .35}
TYPICAL_S = {'panorama': (5.0, 0.3), 'recon3d': (10.0, 1.1)}      # rough total seconds = base + per frame, for an ETA before the fraction is informative
LOG_FILES = re.compile(r'Processed file \[(\d+)/(\d+)\]')
LOG_REGISTERED = re.compile(r'num_reg_frames=(\d+)')
QUIET_S = 20   # a run whose manifest/candidates/inventory changed this recently is still being written by a live sweep


def _nice():
    try:
        os.nice(10)
    except OSError:
        pass


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunStillWriting(ValueError):
    """The run's manifest/candidates/inventory changed a moment ago (a sweep is still processing): wait, do not give up."""


SUFFIX = {'panorama': '_panorama_v', 'recon3d': '_recon3d_v'}
LEGACY_SUFFIX = {'panorama': '_pano_v'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


class BuildJobs:
    def __init__(self, root, command=None):
        self.root = Path(root).resolve()
        self.runs = self.root / 'runs'
        self.state = self.runs / STATE
        self._procs = {}
        self._logs = {}      # job_id -> incremental log counters (running 3D builds only)
        self._lock = threading.RLock()   # one start()/refresh at a time: the Debugger buttons, status polls and the post-sweep queue share this object
        # command(kind, source_dir, out_dir, scan_label, registry_path) -> argv; injectable so tests never run a real build.
        self._command = command or self._default_command

    def _default_command(self, kind, source, out, label, registry, merge_from=None):
        if kind == 'recon3d':
            return [sys.executable, '-m', 'pipeline.recon3d', '--source-run', str(source), '--out', str(out)] + \
                (['--merge-from', str(merge_from)] if merge_from else [])
        argv = [sys.executable, '-m', 'pipeline.panorama', '--source-run', str(source), '--out', str(out)]
        return argv + (['--scan-label', label] if label else [])

    # ---------------------------------------------------------------- eligibility
    def eligible(self, run_id, kind='panorama'):
        """Return (source_dir, scan_label or None). Raises ValueError with the reason a run cannot be built."""
        if kind not in SUFFIX:
            raise ValueError('Unknown build kind')
        if not isinstance(run_id, str) or not ID.fullmatch(run_id):
            raise ValueError('Invalid run ID')
        source = (self.runs / run_id).resolve()
        if not source.is_relative_to(self.runs.resolve()) or not source.is_dir():
            raise ValueError('Unknown run')
        if (source / 'panorama.json').is_file() or (source / 'recon3d.json').is_file():
            raise ValueError('This run is itself a panorama/3D output; build from its source run')
        try:
            manifest = json.loads((source / 'manifest.json').read_text())
            frames = manifest['data']['frames']
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError('Run has no readable manifest.json with a frame list')
        shown = [f for f in frames if isinstance(f, dict) and isinstance(f.get('image_path'), str)]
        present = [f for f in shown if (source / f['image_path']).is_file() and Path(f['image_path']).suffix.lower() in IMAGE_SUFFIXES]
        need = 8 if kind == 'recon3d' else 2
        if len(present) < need:
            raise ValueError(f'Run has only {len(present)} saved frame image(s) of {len(frames)} listed; need at least {need}')
        newest = max((q.stat().st_mtime for q in (source / n for n in ('manifest.json', 'candidates.json', 'inventory.json')) if q.is_file()), default=0)
        if time.time() - newest < QUIET_S:
            raise RunStillWriting(f'Run files were written {time.time() - newest:.0f} s ago: the sweep is probably still processing; try again when it has finished')
        sha = (manifest['data'].get('input') or {}).get('sha256')
        scan = next((s for s in read_registry(self.root) if s.get('sha256') == sha), None) if sha else None
        return source, (scan['label'] if scan else None)

    def _next_output(self, run_id, kind):
        taken = {j['out_run'] for j in self._records()}
        n = 1
        while (self.runs / f'{run_id}{SUFFIX[kind]}{n}').exists() or f'{run_id}{SUFFIX[kind]}{n}' in taken:
            n += 1
        return f'{run_id}{SUFFIX[kind]}{n}'

    # ---------------------------------------------------------------- records
    def _records(self):
        out = []
        for path in sorted(self.state.glob('*.json')) if self.state.is_dir() else []:
            try:
                out.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
        return out

    def _save(self, job):
        self.state.mkdir(parents=True, exist_ok=True)
        tmp = self.state / f"{job['job_id']}.json.tmp"
        tmp.write_text(json.dumps(job, indent=2))
        tmp.replace(self.state / f"{job['job_id']}.json")

    def _refresh(self, job):
        if job['state'] != 'running':
            return job
        proc = self._procs.get(job['job_id'])
        code = proc.poll() if proc is not None else (None if _alive(job['pid']) else -1)
        product = self.runs / job['out_run'] / ('recon3d.json' if job.get('kind') == 'recon3d' else 'panorama.json')
        if proc is None and code == -1 and product.is_file():   # the build finished while the server was restarting
            job.update(state='done', finished_at=job.get('finished_at') or time.time())
        elif proc is None and code == -1:
            job.update(state='interrupted', finished_at=time.time(), error='Server restarted or the build process vanished; nothing was overwritten')
        elif code is not None:
            if code == 0 and product.is_file():
                job.update(state='done', finished_at=time.time())
            else:
                job.update(state='failed', finished_at=time.time(), exit_code=code,
                           error=self.failure_reason(job, code))
            self._procs.pop(job['job_id'], None)
        else:
            return job
        self._save(job)
        return job

    def tail(self, job):
        try:
            lines = (self.state / f"{job['job_id']}.log").read_text(errors='replace').strip().splitlines()
        except OSError:
            return ''
        return '\n'.join(lines[-TAIL_LINES:])

    def failure_reason(self, job, code):
        """The real reason a build failed: the last non-empty log line (the exception message), prefixed with the stage it was in."""
        lines = [l for l in self.tail(job).splitlines() if l.strip()]
        last = lines[-1] if lines else f"{'pipeline.recon3d' if job.get('kind') == 'recon3d' else 'pipeline.panorama'} exited with code {code} and printed nothing"
        stage = (self.progress(job) or {}).get('stage')
        return f'[{stage}] {last}' if stage and stage != 'done' else last

    def _log_counters(self, job):
        """Counters from the new bytes of a running job's log since the last call: files processed i/N, image batches matched, frames registered."""
        c = self._logs.setdefault(job['job_id'], {'offset': 0, 'files': 0, 'total': 0, 'matched': 0, 'registered': 0, 'matching': False})
        try:
            with (self.state / f"{job['job_id']}.log").open('rb') as f:
                f.seek(c['offset'])
                data = f.read(4 * 1024 * 1024)
                c['offset'] += len(data)
        except OSError:
            return c
        for line in data.decode('utf-8', 'replace').splitlines():
            m = LOG_FILES.search(line)
            if m:
                c['files'], c['total'] = int(m.group(1)), int(m.group(2))
            elif 'feature_matching.cc:196' in line:
                c['matching'] = True
            elif c['matching'] and 'feature_matching.cc:218' in line:
                c['matched'] += 1
            else:
                m = LOG_REGISTERED.search(line)
                if m:
                    c['registered'] = int(m.group(1))
        return c

    def _fraction_3d(self, job, raw):
        """(fraction, label, done, total) of a running 3D build from its stage and the log counters."""
        if job.get('merge_from'):
            stage = raw.get('stage') or 'merging'
            fraction = {'markers': .9, 'align': .92, 'writing': .97, 'done': 1.0}.get(stage, MERGE_PASS_STEPS.get(raw.get('step'), .05))
            return fraction, 'joining 3D fragments' + (f" ({raw['step']})" if stage == 'merging' and raw.get('step') else ''), 0, 0
        c = self._log_counters(job)
        stage = raw.get('stage') or 'features'
        n = c['total'] or raw.get('total') or 0
        sub = {'features': (c['files'], n), 'matching': (c['matched'], n), 'mapping': (c['registered'], n)}.get(stage)
        done, total = sub if sub and sub[1] else (0, 0)
        within = min(1.0, done / total) if total else (1.0 if stage in ('writing', 'done') else 0.0)
        names = [w[0] for w in WEIGHTS_3D]
        if stage not in names:
            return (1.0 if stage == 'done' else 0.0), stage, 0, 0
        fraction = sum(w for k, w in WEIGHTS_3D[:names.index(stage)]) + dict(WEIGHTS_3D)[stage] * within
        label = {'features': 'extracting features', 'matching': 'matching frames', 'mapping': 'registering frames (SfM)', 'merging': 'joining 3D fragments', 'markers': 'placing book markers', 'writing': 'writing the result'}[stage]
        return fraction, label + (f' {done}/{total}' if total else ''), done, total

    def progress(self, job):
        """progress.json of the build plus {label, fraction 0..1, done, total, eta_s}; the fraction/ETA are only computed for a running job."""
        try:
            raw = json.loads((self.runs / job['out_run'] / 'progress.json').read_text())
        except (OSError, ValueError):
            raw = None
        if job.get('state') == 'done':
            return dict(raw or {}, fraction=1.0, eta_s=0)
        if job.get('state') != 'running':
            self._logs.pop(job['job_id'], None)
            return raw
        raw = dict(raw or {'stage': None})
        elapsed = time.time() - job['started_at']
        raw['elapsed_s'] = round(elapsed, 1)
        if job.get('kind') == 'recon3d':
            fraction, label, done, total = self._fraction_3d(job, raw)
            raw.update(fraction=round(fraction, 4), label=label, done=done, total=total)
        raw.setdefault('fraction', 0.0)
        raw.setdefault('label', raw.get('stage') or 'starting')
        base, per_frame = TYPICAL_S.get(job.get('kind'), (10.0, 0.5))
        typical = base + per_frame * (raw.get('total') or raw.get('frames') or 60)
        f = raw['fraction']
        raw['eta_s'] = round(max(2.0, elapsed * (1 - f) / f), 0) if f >= 0.15 and elapsed > 4 else round(max(5.0, typical - elapsed), 0)
        return raw

    def jobs(self):
        with self._lock:
            return self._jobs_locked()

    def _jobs_locked(self):
        out = []
        for job in self._records():
            job = self._refresh(job)
            end = job.get('finished_at') or time.time()
            out.append(dict(job, elapsed_s=round(end - job['started_at'], 1), log_tail=self.tail(job),
                            progress=self.progress(job)))
        out.sort(key=lambda j: j['started_at'], reverse=True)
        return out

    # ---------------------------------------------------------------- start
    def start(self, run_id, kind='panorama', merge_from=None):
        """Start one build. merge_from (3D only): a `<run>_recon3d_vN` whose merge was deferred; the job is then the merge pass,
        which reuses that build's database and writes the joined map as the next version."""
        with self._lock:
            return self._start(run_id, kind, merge_from)

    def _start(self, run_id, kind, merge_from=None):
        if merge_from is not None and (kind != 'recon3d' or not ID.fullmatch(str(merge_from)) or not str(merge_from).startswith(f'{run_id}_recon3d_v')):
            raise ValueError('Invalid merge source')
        source, label = self.eligible(run_id, kind)
        if any(j['state'] == 'running' for j in self.jobs()):
            raise ValueError('Another panorama build is already running; wait for it to finish')
        out_run = self._next_output(run_id, kind)
        registry = self.root / 'sweeps' / 'scan_registry.json'
        job = {'job_id': uuid.uuid4().hex[:12], 'kind': kind, 'source_run': run_id, 'out_run': out_run, 'scan': label,
               'state': 'running', 'started_at': time.time(), 'pid': None, **({'merge_from': merge_from} if merge_from else {})}
        self.state.mkdir(parents=True, exist_ok=True)
        log = (self.state / f"{job['job_id']}.log").open('wb')
        argv = self._command(kind, source, self.runs / out_run, label, registry, **({'merge_from': self.runs / merge_from} if merge_from else {}))
        try:
            proc = subprocess.Popen(argv, cwd=self.root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, preexec_fn=_nice)
        finally:
            log.close()
        job['pid'] = proc.pid
        self._procs[job['job_id']] = proc
        self._save(job)
        return dict(job, elapsed_s=0.0, log_tail='', progress=None)
