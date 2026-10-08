"""The one build queue for 2D panoramas and 3D maps: automatic after every finished sweep AND the Debugger's Build buttons.

* After a sweep (End sweep, live camera or browser replay) the engine calls `enqueue(run_id)` only AFTER claim_packet.json / report.html are
  written, so the packet and its time budget are never delayed. It queues the 2D panorama first, then the 3D map.
* The Debugger buttons call `request(run_id, kinds)`. A press while something else is building is QUEUED (FIFO, never refused or lost);
  the status says "queued #n". Only a run that can never be built (unknown run, too few frames) is refused, and that reason is returned so it can
  be shown next to the button.
* One worker thread takes one task at a time; each task is an ordinary niced `app.build_jobs` job, and it also waits for a build that was started
  elsewhere in the same state directory (a CLI run is not controllable, but it is waited for). A run whose files were written a moment ago is
  waited for, not refused. A failed step records the real reason and never blocks the other steps.
* After a 2D panorama finishes, `on_count(run_id, panorama_out_run)` (pipeline.panorama_attach.run_count_step: spines counted on the panorama pieces and attached to the packet,
  before the 3D map starts) runs; its outcome is recorded in `count` and it can never fail the maps.
* After a 3D map finishes, `on_built(run_id, recon3d_out_run)` (the cheap local measurement step, app.measurement.finalize_run) runs; its outcome
  is recorded in `measure` and it can never fail the maps.
* After the count and again after the measurement, `on_instances(run_id, panorama_out_run)` (pipeline.instances.run_step: track groups merged into physical
  instances, the packet's lines rebuilt in place) runs; its outcome is recorded in `instances` and it can never fail the maps.

State per run lives in `runs/_server_state/post_sweep/<run>.json`, so it survives the browser closing and a server restart. A server restart is
NOT an attempt: on resume a build whose process outlived the server is followed, a build whose output finished meanwhile is taken as done (never
rebuilt), and only a build whose process vanished without output is started again (at most MAX_ATTEMPTS starts). A finished 3D map whose
measurement never ran (interrupted, or the map predates the step) is measured on resume. Nothing here edits a saved run's frames/detections: outputs are the usual new versioned `<run>_panorama_vN` /
`<run>_recon3d_vN` directories.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .build_jobs import RunStillWriting

STATE = Path('_server_state') / 'post_sweep'
STEPS = ('panorama', 'recon3d')
ACTIVE_STEP = {'queued', 'waiting', 'running', 'pending'}      # 'pending' = records written by the first post-sweep version
ACTIVE = {'queued', 'waiting', 'building_2d', 'building_3d', 'counting', 'merging', 'measuring', 'joining_3d'}
ACTIVE_FINISH = {'queued', 'running'}   # rec['finish_reading']: the post-sweep title reading of the groups the packet budget left unread; overall state 'reading'
ACTIVE_MERGE = {'queued', 'running'}     # rec['merge3d']: the 3D merge pass after a first map whose fragment merge was deferred
LABEL = {'panorama': '2D panorama', 'recon3d': '3D map'}
MAX_WAIT_S = 300          # how long a run may keep changing before the queue gives up on it
MAX_ATTEMPTS = 3          # build STARTS per step; following a build across a server restart is not a start
PRODUCT = {'panorama': 'panorama.json', 'recon3d': 'recon3d.json'}
MERGE_CACHE = '_merge_cache'     # pipeline.recon3d.MERGE_CACHE (not imported: the server must not load pycolmap-side modules for this)


class PostSweepBuilds:
    def __init__(self, root, jobs, *, enabled=True, poll_s=2.0, on_built=None, on_count=None, on_instances=None, finish_runner=None):
        """jobs: an app.build_jobs.BuildJobs, or a zero-argument callable returning one (the Debugger may swap it).
        enabled only gates the AUTOMATIC post-sweep `enqueue`; manual `request` always works."""
        self.root = Path(root).resolve()
        self.runs = self.root / 'runs'
        self.dir = self.runs / STATE
        self._jobs = jobs
        self.on_built = on_built      # on_built(run_id, recon3d_out_run) -> dict: the cheap local measurement step after the 3D map (app.measurement.finalize_run)
        self.on_count = on_count      # on_count(run_id, panorama_out_run) -> dict: the chain-independent count from the 2D panorama (pipeline.panorama_attach.run_count_step)
        self.on_instances = on_instances  # on_instances(run_id, panorama_out_run) -> dict: physical instances + packet lines in place (pipeline.instances.run_step), after the count and after the measurement
        self.finish_runner = finish_runner  # finish_runner(run_id, ctl) -> dict: app.finish_reading.run on the run (reads the groups the packet budget skipped, resolves, prices, updates the packet)
        self.finish_enabled = True      # gates the AUTOMATIC finish step in enqueue (LIBRARY_FINISH_READING=0 turns it off); request_finish_reading always works
        self.enabled = enabled
        self.poll_s = poll_s
        self._records = {}
        self._queue = []          # [(run_id, kind)] FIFO of tasks not started yet
        self._current = None      # (run_id, kind) being processed
        self._lock = threading.RLock()
        self._packet_lock = threading.RLock()   # ONE writer on a run's packet at a time: the build worker's count / duplicate merge / measurement and the finish-reading thread's attach phases
        self._finish_queue = []
        self._finish_current = None
        self._finish_wake = threading.Event()
        self._finish_thread = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ plumbing
    @property
    def jobs(self):
        return self._jobs() if callable(self._jobs) else self._jobs

    def _save(self, rec):
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / f"{rec['run_id']}.json.tmp"
            tmp.write_text(json.dumps(rec, indent=2))
            tmp.replace(self.dir / f"{rec['run_id']}.json")
        except OSError:
            pass

    def _scan_label(self, run_id):
        from .debug_registry import read_registry
        from .debug_views import UNREGISTERED
        try:
            sha = json.loads((self.runs / run_id / 'manifest.json').read_text())['data'].get('input', {}).get('sha256')
            scan = next((s for s in read_registry(self.root) if s.get('sha256') == sha), None) if sha else None
            return scan['label'] if scan else UNREGISTERED
        except (OSError, ValueError, KeyError, TypeError):
            return UNREGISTERED

    def _blank(self, run_id):
        try:
            frames = len(json.loads((self.runs / run_id / 'manifest.json').read_text())['data']['frames'])
        except (OSError, ValueError, KeyError, TypeError):
            frames = 60
        return {'run_id': run_id, 'scan_label': self._scan_label(run_id), 'state': 'none', 'queued_at': time.time(), 'note': '', 'frames': frames,
                'steps': {k: {'state': 'none', 'attempts': 0} for k in STEPS}}

    def _new_record(self, run_id):
        """Kept for tests/tools: a record with both steps queued, as after a sweep."""
        rec = self._blank(run_id)
        for k in STEPS:
            rec['steps'][k]['state'] = 'queued'
        self._recompute(rec)
        return rec

    @staticmethod
    def _measure_text(rec):
        m = rec.get('measure')
        if not m:
            return ''
        return str(m.get('summary')) if m.get('state') == 'done' and m.get('summary') else f"{m.get('state')}: {m.get('error') or m.get('reason') or ''}".strip()

    @staticmethod
    def _instances_text(rec):
        c = rec.get('instances') or {}
        if c.get('state') == 'done':
            return c.get('summary') or f"{c.get('book_count')} books"
        return f"{c.get('state')}: {c.get('error') or c.get('reason') or ''}".strip()

    @staticmethod
    def _finish_note(f):
        n = f" {f['done']} of {f['total']}" if f.get('total') else ''
        return f"finishing the title reading: {f.get('phase') or f.get('state')}{n}" if f.get('state') == 'running' else 'finishing the title reading (queued)'

    @staticmethod
    def _finish_text(rec):
        f = rec.get('finish_reading') or {}
        if f.get('state') == 'done':
            return f.get('summary') or 'done'
        if f.get('state') in ACTIVE_FINISH:
            return PostSweepBuilds._finish_note(f)
        return f"{f.get('state')}: {f.get('error') or f.get('reason') or ''}".strip()

    @staticmethod
    def _count_text(rec):
        c = rec.get('count') or {}
        if c.get('state') == 'done':
            return c.get('summary') or f"panorama count {c.get('panorama_count')} ({c.get('estimator')})"
        return f"{c.get('state')}: {c.get('error') or c.get('reason') or ''}".strip()

    def _recompute(self, rec):
        """Overall state/note of a record from its steps (and the measurement step)."""
        steps = rec['steps']
        running = [k for k in STEPS if steps[k]['state'] == 'running']
        active = [k for k in STEPS if steps[k]['state'] in ACTIVE_STEP]
        if running:
            rec['state'] = 'building_2d' if running[0] == 'panorama' else 'building_3d'
            rec['note'] = f'building {LABEL[running[0]]}'
        elif (rec.get('count') or {}).get('state') == 'running':
            rec['state'] = 'counting'
            rec['note'] = 'counting spines on the 2D panorama'
        elif (rec.get('instances') or {}).get('state') == 'running':
            rec['state'] = 'merging'
            rec['note'] = 'merging track groups into physical books'
        elif (rec.get('measure') or {}).get('state') == 'running':
            rec['state'] = 'measuring'
            rec['note'] = 'scaling the 3D map (spines, shelf run, room)'
        elif (rec.get('merge3d') or {}).get('state') in ACTIVE_MERGE and not active:
            rec['state'] = 'joining_3d'
            rec['note'] = 'the first 3D map is ready; joining its fragments into one map' + (' (queued)' if rec['merge3d']['state'] == 'queued' else '')
        elif any(steps[k]['state'] == 'waiting' for k in STEPS):
            rec['state'] = 'waiting'
            rec['note'] = steps[[k for k in STEPS if steps[k]['state'] == 'waiting'][0]].get('note') or 'waiting for the sweep files to settle'
        elif active:
            rec['state'] = 'queued'
            rec['note'] = 'queued'
        elif (rec.get('finish_reading') or {}).get('state') in ACTIVE_FINISH:
            rec['state'] = 'reading'
            rec['note'] = self._finish_note(rec['finish_reading'])
        else:
            bad = [k for k in STEPS if steps[k]['state'] == 'failed']
            asked = [k for k in STEPS if steps[k]['state'] in ('done', 'failed')]
            rec['state'] = 'none' if not asked else ('failed' if bad else 'done')
            if bad:
                rec['note'] = '; '.join(f"{LABEL[k]}: {steps[k].get('error') or 'failed'}" for k in bad)
            elif asked:
                extra = self._measure_text(rec)
                rec['note'] = ' and '.join(LABEL[k] for k in asked) + ' ready' + (f' ({extra})' if extra else '')
            else:
                rec['note'] = ''
            if asked:
                rec['finished_at'] = time.time()

    # ------------------------------------------------------------------ public API
    def enqueue(self, run_id):
        """After a finished sweep: queue the 2D panorama, then the 3D map (automatic; honours `enabled`)."""
        if not self.enabled:
            return None
        out = self._add(run_id, STEPS)
        if self.finish_enabled and self.finish_runner is not None:
            out = self.request_finish_reading(run_id)
        return out

    def request(self, run_id, kinds):
        """The Debugger buttons. Raises ValueError (with the reason) only for a run that can never be built; otherwise the build is
        queued behind whatever is running/queued. A kind that is already queued or running is not queued twice."""
        kinds = tuple(k for k in STEPS if k in tuple(kinds))
        if not kinds:
            raise ValueError('Unknown build kind')
        for kind in kinds:
            try:
                self.jobs.eligible(run_id, kind)
            except RunStillWriting:
                pass                                   # waited for by the worker
        return self._add(run_id, kinds)

    def _add(self, run_id, kinds):
        with self._lock:
            rec = self._records.get(run_id) or self._load(run_id) or self._blank(run_id)
            if rec.get('state') not in ACTIVE:         # a finished record starts a new round (its old results stay in the output dirs)
                rec['queued_at'] = time.time()
                rec.pop('measure', None)
                rec.pop('count', None)
                rec.pop('instances', None)
                rec.pop('merge3d', None)
            for kind in kinds:
                step = rec['steps'][kind]
                if step['state'] in ACTIVE_STEP:
                    continue
                rec['steps'][kind] = {'state': 'queued', 'attempts': 0}
                self._queue.append((run_id, kind))
            self._records[run_id] = rec
            self._recompute(rec)
            self._save(rec)
        self.start()
        self._wake.set()
        return self.status(run_id)

    def status(self, run_id):
        """Copy of the run's record with queue positions and the running step's live progress, or None when nothing was ever queued."""
        with self._lock:
            rec = self._records.get(run_id) or self._load(run_id)
            if rec is None:
                return None
            out = json.loads(json.dumps(rec))
            queue = list(self._queue)
            current = self._current
        for key, step in out['steps'].items():
            if step.get('state') == 'queued' and (run_id, key) in queue:
                step['position'] = queue.index((run_id, key)) + 1
                step['ahead'] = ([{'run_id': current[0], 'kind': current[1]}] if current else []) + \
                    [{'run_id': r, 'kind': k} for r, k in queue[:step['position'] - 1]]
            if step.get('state') == 'running' and step.get('job_id'):
                job = next((j for j in self._safe_jobs() if j['job_id'] == step['job_id']), None)
                if job:
                    pr = job.get('progress') or {}
                    step['stage'] = pr.get('stage')
                    step['elapsed_s'] = job.get('elapsed_s')
                    step['progress'] = {k: pr.get(k) for k in ('label', 'fraction', 'done', 'total', 'eta_s') if pr.get(k) is not None}
        m = out.get('merge3d') or {}
        if m.get('state') == 'running' and m.get('job_id'):
            job = next((j for j in self._safe_jobs() if j['job_id'] == m['job_id']), None)
            if job:
                pr = job.get('progress') or {}
                m['elapsed_s'] = job.get('elapsed_s')
                m['progress'] = {k: pr.get(k) for k in ('label', 'fraction', 'eta_s') if pr.get(k) is not None}
        self._recompute(out)
        out['summary'] = self._summary(out)
        out['overall'] = self._overall(out)
        if out.get('count'):
            out['summary'] += ' · count: ' + self._count_text(out)
        if out.get('instances'):
            out['summary'] += ' · books: ' + self._instances_text(out)
        if out.get('finish_reading'):
            out['summary'] += ' · reading: ' + self._finish_text(out)
        if out.get('measure'):
            out['summary'] += ' · measurement: ' + self._measure_text(out)
        if out.get('merge3d'):
            out['summary'] += ' · 3D fragments: ' + self._merge3d_text(out)
        return out

    @staticmethod
    def _merge3d_text(rec):
        m = rec.get('merge3d') or {}
        if m.get('state') == 'done':
            ratio = m.get('registered_ratio')
            return f"joined map ready ({m.get('out_run')}" + (f", {ratio:.0%} of frames registered" if isinstance(ratio, (int, float)) else '') + ')'
        if m.get('state') in ACTIVE_MERGE:
            return f"joining after the first map ({m['state']})"
        return f"{m.get('state')}: {m.get('error') or m.get('reason') or ''}".strip()

    @staticmethod
    def _overall(rec):
        """Fraction 0..1 and a rough ETA (seconds) over the steps asked for in this round (the panorama is ~1/4 of the work, the 3D map ~3/4)."""
        from .build_jobs import TYPICAL_S
        weights = {'panorama': .25, 'recon3d': .75}
        num = den = eta = 0.0
        frames = rec.get('frames') or 60
        for kind in STEPS:
            st = rec['steps'][kind]
            if st['state'] == 'none':
                continue
            den += weights[kind]
            pr = st.get('progress') or {}
            if st['state'] in ('done', 'failed'):
                num += weights[kind]
            elif st['state'] == 'running':
                num += weights[kind] * float(pr.get('fraction') or 0.0)
                eta += float(pr.get('eta_s') or 0.0)
            else:
                base, per = TYPICAL_S[kind]
                eta += base + per * frames
        m = rec.get('merge3d') or {}
        if m.get('state'):                      # the merge pass after the first 3D map: about 0.4 of a 3D build
            den += .3
            pr = m.get('progress') or {}
            if m['state'] in ACTIVE_MERGE:
                num += .3 * float(pr.get('fraction') or 0.0)
                eta += float(pr.get('eta_s') or 0.4 * (TYPICAL_S['recon3d'][0] + TYPICAL_S['recon3d'][1] * frames))
            else:
                num += .3
        if not den:
            return None
        return {'fraction': round(num / den, 4), 'eta_s': round(eta)}

    def is_active(self, run_id):
        """True while this run still has maps queued, waiting, building or being measured (the New-sweep guard: those finish in the background, nothing cancels them)."""
        st = self.status(run_id)
        return bool(st) and st['state'] in ACTIVE

    def queue_overview(self):
        with self._lock:
            return {'running': list(self._current) if self._current else None, 'queued': [list(t) for t in self._queue]}

    def _safe_jobs(self):
        try:
            return self.jobs.jobs()
        except Exception:   # noqa: BLE001 - status must never break the Sweep tab or the Debugger
            return []

    @staticmethod
    def _summary(rec):
        parts = []
        for key in STEPS:
            st = rec['steps'][key]
            name = LABEL[key]
            if st['state'] == 'none':
                continue
            if st['state'] == 'done':
                extra = f" ({st['markers']} markers = book_count {st['book_count']})" if st.get('book_count') is not None and st.get('markers') is not None else ''
                parts.append(f'{name} ready{extra}')
            elif st['state'] == 'failed':
                parts.append(f"{name} failed: {st.get('error') or 'unknown reason'}")
            elif st['state'] == 'running':
                parts.append(f"building {name}" + (f" · {st['stage']}" if st.get('stage') else ''))
            elif st['state'] == 'waiting':
                parts.append(f"{name}: {st.get('note') or 'waiting'}")
            else:
                parts.append(f"{name} queued" + (f" #{st['position']}" if st.get('position') else ''))
        return ' · '.join(parts)

    # ------------------------------------------------------------------ worker
    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name='build-queue', daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        self._finish_wake.set()

    def resume(self):
        """Server start: re-queue tasks that were queued, waiting or running when the server stopped (done/failed records are history)."""
        if not self.dir.is_dir():
            return 0
        n = 0
        for path in sorted(self.dir.glob('*.json')):
            try:
                rec = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(rec.get('steps'), dict):
                continue
            self._adopt_finished(rec)
            measure = self._needs_measure(rec)
            count = self._needs_count(rec)
            inst = self._needs_instances(rec)
            merge3d = self._needs_merge3d(rec)
            fin = (rec.get('finish_reading') or {}).get('state') in ACTIVE_FINISH and self.finish_runner is not None
            if not measure and not count and not inst and not merge3d and not fin and not any(rec['steps'].get(k, {}).get('state') in ACTIVE_STEP for k in STEPS):
                continue
            with self._lock:
                if rec['run_id'] in self._records:
                    continue
                job_by_id = {j['job_id']: j for j in self._safe_jobs()}
                for kind in STEPS:
                    step = rec['steps'].setdefault(kind, {'state': 'none', 'attempts': 0})
                    if step['state'] not in ACTIVE_STEP:
                        continue
                    job = job_by_id.get(step.get('job_id')) if step.get('job_id') else None
                    if step['state'] == 'running' and job and job['state'] == 'done':
                        step['state'] = 'done'
                        continue
                    step['state'] = 'running' if (step['state'] == 'running' and job and job['state'] == 'running') else 'queued'
                    self._queue.append((rec['run_id'], kind))
                    n += 1
                if (rec.get('measure') or {}).get('state') == 'running':
                    rec.pop('measure')
                if (rec.get('count') or {}).get('state') == 'running':
                    rec.pop('count')
                if (rec.get('instances') or {}).get('state') == 'running':
                    rec.pop('instances')
                if count and rec['steps']['panorama']['state'] == 'done':
                    self._queue.insert(0, (rec['run_id'], 'count'))
                    n += 1
                elif inst and rec['steps']['panorama']['state'] == 'done':
                    self._queue.insert(0, (rec['run_id'], 'instances'))
                    n += 1
                if measure and rec['steps']['recon3d']['state'] == 'done':
                    self._queue.append((rec['run_id'], 'measure'))
                    n += 1
                if fin:
                    rec['finish_reading']['state'] = 'queued'
                    self._finish_queue.append(rec['run_id'])
                    n += 1
                if merge3d:
                    m = rec.get('merge3d') or {}
                    if m.get('state') not in ACTIVE_MERGE:
                        rec['merge3d'] = {'state': 'queued', 'first_map': merge3d, 'attempts': 0}
                    self._queue.append((rec['run_id'], 'merge3d'))
                    n += 1
                self._records[rec['run_id']] = rec
                self._recompute(rec)
                self._save(rec)
        if n:
            self.start()
            self._wake.set()
            self._start_finish()
        return n

    def _finished(self, out_run, kind):
        """The build's output exists (both builders write it atomically as their last step)."""
        return bool(out_run) and (self.runs / out_run / PRODUCT[kind]).is_file()

    def _adopt_finished(self, rec):
        """A step whose build finished while the server was down (or was wrongly marked failed after restarts) is done, not rebuilt."""
        changed = False
        for kind in STEPS:
            step = rec['steps'].get(kind) or {}
            if step.get('state') in ACTIVE_STEP | {'failed'} and self._finished(step.get('out_run'), kind):
                if step.get('state') == 'running' and step.get('job_id'):
                    job = next((j for j in self._safe_jobs() if j['job_id'] == step['job_id']), None)
                    if job and job['state'] == 'running':
                        continue                              # still writing; followed by the worker
                step.update(state='done', error=None, note=None, finished_at=step.get('finished_at') or time.time(),
                            **self._result_summary(step['out_run'], kind))
                changed = True
        if changed:
            self._recompute(rec)
        return changed

    def _needs_measure(self, rec):
        """A finished 3D map that was never measured (or whose measurement was cut by a restart)."""
        if self.on_built is None:
            return False
        step = rec['steps'].get('recon3d') or {}
        m = rec.get('measure') or {}
        return (step.get('state') == 'done' and self._finished(step.get('out_run'), 'recon3d')
                and m.get('state') in (None, 'running', 'skipped'))

    def _deferred_first_map(self, out_run):
        """True when this 3D map deferred its fragment merge and the cache the merge pass needs is still there."""
        if not out_run:
            return False
        try:
            merge = json.loads((self.runs / out_run / PRODUCT['recon3d']).read_text()).get('merge') or {}
        except (OSError, ValueError):
            return False
        return bool(merge.get('deferred')) and (self.runs / out_run / MERGE_CACHE / 'merge_state.json').is_file()

    def _needs_merge3d(self, rec):
        """The first map name when a merge pass is due: queued/running when the server stopped, or a finished first map whose deferred
        merge was never queued (the server stopped in between). None otherwise."""
        m = rec.get('merge3d') or {}
        if m.get('state') in ACTIVE_MERGE and m.get('first_map'):
            return m['first_map']
        step = rec['steps'].get('recon3d') or {}
        if not m and step.get('state') == 'done' and self._deferred_first_map(step.get('out_run')):
            return step['out_run']
        return None

    def _needs_count(self, rec):
        """A finished 2D panorama whose count never ran (interrupted, or the panorama predates the step) and that has no panorama_count.json for that panorama."""
        if self.on_count is None:
            return False
        step = rec['steps'].get('panorama') or {}
        c = rec.get('count') or {}
        if not (step.get('state') == 'done' and self._finished(step.get('out_run'), 'panorama') and c.get('state') in (None, 'running')):
            return False
        try:
            have = json.loads((self.runs / rec['run_id'] / 'panorama_count.json').read_text()).get('panorama_run')
        except (OSError, ValueError):
            have = None
        return have != step.get('out_run')

    def _needs_instances(self, rec):
        """A counted 2D panorama whose instance step never ran (interrupted, or the packet predates the step) and that has no instances.json yet."""
        if self.on_instances is None:
            return False
        step = rec['steps'].get('panorama') or {}
        c, i = rec.get('count') or {}, rec.get('instances') or {}
        if not (step.get('state') == 'done' and self._finished(step.get('out_run'), 'panorama') and c.get('state') == 'done' and i.get('state') in (None, 'running')):
            return False
        return not (self.runs / rec['run_id'] / 'instances.json').is_file()

    def _load(self, run_id):
        try:
            return json.loads((self.dir / f'{run_id}.json').read_text())
        except (OSError, ValueError):
            return None

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                task = self._queue.pop(0) if self._queue else None
                self._current = task
            if task is None:
                self._wake.wait(timeout=5)
                self._wake.clear()
                continue
            try:
                self._process(*task)
            except Exception as exc:   # noqa: BLE001 - a bug here must not kill the queue
                if task[1] == 'measure':
                    self._set_measure(task[0], {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
                elif task[1] == 'instances':
                    self._set_instances(task[0], {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
                elif task[1] == 'merge3d':
                    self._set_merge3d(task[0], {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
                else:
                    self._step(task[0], task[1], state='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                with self._lock:
                    self._current = None

    def _step(self, run_id, kind, **fields):
        with self._lock:
            rec = self._records[run_id]
            rec['steps'][kind].update(fields)
            self._recompute(rec)
            self._save(rec)

    def _process(self, run_id, kind):
        if kind == 'measure':
            self._measure(run_id)
            return
        if kind == 'count':
            self._count(run_id)
            return
        if kind == 'instances':
            self._instances(run_id)
            return
        if kind == 'merge3d':
            self._merge3d(run_id)
            return
        with self._lock:
            step = self._records[run_id]['steps'][kind]
            if step['state'] not in ACTIVE_STEP:
                return
            resumed = step['state'] == 'running' and step.get('job_id')
        if not resumed:
            self._step(run_id, kind, state='waiting', note='waiting for the sweep files to settle')
        self._step(run_id, kind, **self._build(run_id, kind))
        if kind == 'panorama':
            self._count(run_id)
        if kind == 'recon3d':
            self._measure(run_id)
            self._queue_merge3d(run_id)

    # ------------------------------------------------------------------ finish reading
    def request_finish_reading(self, run_id):
        """Queue the post-sweep "finish reading" step for a run (automatic after a sweep; also callable for an older run). One run at a time in its own thread
        (it is network-bound, so it overlaps the 2D / 3D builds); its packet-writing phases take the same packet lock as the build worker's hooks."""
        if self.finish_runner is None:
            raise ValueError('finish reading is not configured')
        with self._lock:
            rec = self._records.get(run_id) or self._load(run_id) or self._blank(run_id)
            f = rec.get('finish_reading') or {}
            if f.get('state') in ACTIVE_FINISH:
                self._records.setdefault(run_id, rec)
                return self.status(run_id)
            rec['finish_reading'] = {'state': 'queued', 'queued_at': time.time(), 'phase': 'queued'}
            self._records[run_id] = rec
            self._finish_queue.append(run_id)
            self._recompute(rec)
            self._save(rec)
        self._start_finish()
        self._finish_wake.set()
        return self.status(run_id)

    def _start_finish(self):
        with self._lock:
            if self._finish_thread and self._finish_thread.is_alive():
                return
            self._stop.clear()
            self._finish_thread = threading.Thread(target=self._finish_loop, name='finish-reading', daemon=True)
            self._finish_thread.start()

    def _finish_loop(self):
        while not self._stop.is_set():
            with self._lock:
                run_id = self._finish_queue.pop(0) if self._finish_queue else None
                self._finish_current = run_id
            if run_id is None:
                self._finish_wake.wait(timeout=5)
                self._finish_wake.clear()
                continue
            try:
                self._finish(run_id)
            except Exception as exc:   # noqa: BLE001 - a bug here must not kill the thread
                self._set_finish(run_id, {'state': 'failed', 'error': f'{type(exc).__name__}: {str(exc)[:200]}', 'finished_at': time.time()}, keep=True)
            finally:
                with self._lock:
                    self._finish_current = None

    def _set_finish(self, run_id, fields, *, keep=True):
        with self._lock:
            rec = self._records[run_id]
            rec['finish_reading'] = {**(rec.get('finish_reading') or {}), **fields} if keep else fields
            self._recompute(rec)
            self._save(rec)

    def _chain_settled(self, run_id):
        """True when the run's panorama / count / duplicate-merge chain has no more work to do (the finish step writes the packet only then)."""
        with self._lock:
            rec = self._records.get(run_id)
            if rec is None:
                return True
            pan = rec['steps'].get('panorama') or {}
            if pan.get('state') in ACTIVE_STEP:
                return False
            if (run_id, 'count') in self._queue or (run_id, 'instances') in self._queue or self._current in ((run_id, 'count'), (run_id, 'instances')):
                return False
            if pan.get('state') == 'done' and self._finished(pan.get('out_run'), 'panorama'):
                if self.on_count is not None and (rec.get('count') or {}).get('state') not in ('done', 'failed', 'skipped'):
                    return False
                if self.on_instances is not None and (rec.get('instances') or {}).get('state') not in ('done', 'failed', 'skipped'):
                    return False
            return True

    def _finish(self, run_id):
        from .finish_reading import Ctl
        state = {'t': 0.0}

        def progress(phase, done=0, total=0, note=''):
            now = time.time()
            if now - state['t'] < 0.5 and done not in (0, total):
                return
            state['t'] = now
            self._set_finish(run_id, {'state': 'running', 'phase': phase, 'done': done, 'total': total, 'note': note})

        def instances():
            self._instances(run_id)
            return (self._records[run_id].get('instances') or {})

        self._set_finish(run_id, {'state': 'running', 'phase': 'starting', 'done': 0, 'total': 0, 'started_at': time.time(), 'error': None, 'reason': None})
        ctl = Ctl(progress=progress, stop=self._stop.is_set, packet_lock=self._packet_lock, settle=lambda: self._chain_settled(run_id),
                  instances=instances if self.on_instances is not None else None)
        out = self.finish_runner(run_id, ctl) or {}
        if out.get('state') == 'interrupted' or self._stop.is_set():
            self._set_finish(run_id, {'state': 'queued', 'phase': 'queued'})        # server stopping: resume() runs it again (finished chunks are kept)
            return
        keep = {k: out.get(k) for k in ('state', 'reason', 'summary', 'groups_attemptable', 'groups_unread_at_start', 'groups_read', 'readings_new', 'readings_unresolved_at_start',
                                        'resolved', 'accepted_new', 'before', 'after', 'usage', 'elapsed_s') if k in out}
        self._set_finish(run_id, {**keep, 'phase': 'finished', 'finished_at': time.time(), 'done': out.get('groups_unread_at_start') or 0, 'total': out.get('groups_unread_at_start') or 0})

    # ------------------------------------------------------------------ the 3D merge pass
    def _set_merge3d(self, run_id, merge3d, *, keep=True):
        with self._lock:
            rec = self._records[run_id]
            rec['merge3d'] = {**(rec.get('merge3d') or {}), **merge3d} if keep else merge3d
            self._recompute(rec)
            self._save(rec)

    def _queue_merge3d(self, run_id):
        """After a first 3D map (built, measured) whose fragment merge was deferred to keep it within ~5 min: queue the merge pass. The
        first map stays published and measured meanwhile; the joined map replaces it as the newest version when the pass finishes."""
        with self._lock:
            rec = self._records[run_id]
            step = rec['steps'].get('recon3d') or {}
            if step.get('state') != 'done' or not self._deferred_first_map(step.get('out_run')):
                return
            m = rec.get('merge3d') or {}
            if m.get('state') in ACTIVE_MERGE and m.get('first_map') == step['out_run']:
                return
            rec['merge3d'] = {'state': 'queued', 'first_map': step['out_run'], 'attempts': 0, 'queued_at': time.time()}
            self._queue.append((run_id, 'merge3d'))
            self._recompute(rec)
            self._save(rec)
        self._wake.set()

    def _merge3d(self, run_id):
        """The merge pass: one ordinary niced 3D job (`--merge-from <first map>`), then the measurement of the joined map. A failure
        is recorded and the first map stands; it never fails the maps."""
        with self._lock:
            m = dict(self._records[run_id].get('merge3d') or {})
        if m.get('state') not in ACTIVE_MERGE or not m.get('first_map'):
            return
        job_id = m.get('job_id') if m.get('state') == 'running' else None
        if job_id:
            job = next((j for j in self._safe_jobs() if j['job_id'] == job_id), None)
            if job is None or job['state'] not in ('running', 'done'):
                job_id = None                                   # the pass vanished with a server restart: start it again
        if job_id is None and not self._deferred_first_map(m['first_map']):
            self._set_merge3d(run_id, {'state': 'skipped', 'reason': f"{m['first_map']} has no deferred merge (cache gone)"})
            return
        started = time.time()
        while job_id is None:
            if self._stop.is_set():
                return
            if m.get('attempts', 0) >= MAX_ATTEMPTS:
                self._set_merge3d(run_id, {'state': 'failed', 'error': f"the merge pass was interrupted {m['attempts']} times; the first map stands"})
                return
            try:
                job = self.jobs.start(run_id, 'recon3d', merge_from=m['first_map'])
                job_id = job['job_id']
                m['attempts'] = m.get('attempts', 0) + 1
                self._set_merge3d(run_id, {'state': 'running', 'job_id': job_id, 'out_run': job['out_run'], 'started_at': time.time(),
                                           'attempts': m['attempts'], 'error': None})
            except RunStillWriting as exc:
                if time.time() - started > MAX_WAIT_S:
                    self._set_merge3d(run_id, {'state': 'failed', 'error': f'the run kept changing for {MAX_WAIT_S} s: {exc}'})
                    return
                self._stop.wait(self.poll_s)
            except ValueError as exc:
                if 'already running' in str(exc):
                    self._stop.wait(self.poll_s)
                    continue
                self._set_merge3d(run_id, {'state': 'failed', 'error': str(exc)})
                return
            except OSError as exc:
                self._set_merge3d(run_id, {'state': 'failed', 'error': f'could not start the merge pass: {exc}'})
                return
        while not self._stop.is_set():
            job = next((j for j in self._safe_jobs() if j['job_id'] == job_id), None)
            if job is None:
                self._set_merge3d(run_id, {'state': 'failed', 'error': 'the merge pass job record disappeared'})
                return
            if job['state'] != 'running':
                break
            self._stop.wait(self.poll_s)
        else:
            return                                              # server stopping: resumed (followed or restarted) on the next start
        if job['state'] != 'done':
            self._set_merge3d(run_id, {'state': 'failed', 'out_run': job['out_run'], 'finished_at': time.time(),
                                       'error': (job.get('error') or f"merge pass {job['state']}") + '; the first map stands'})
            return
        summary = self._result_summary(job['out_run'], 'recon3d')
        with self._lock:
            rec = self._records[run_id]
            rec['steps']['recon3d'].update(out_run=job['out_run'], first_map_out_run=m['first_map'], **summary)
        self._set_merge3d(run_id, {'state': 'done', 'out_run': job['out_run'], 'finished_at': time.time(), **summary})
        self._measure(run_id)                                   # the joined map is measured and attached like the first one

    def _count(self, run_id):
        """After a 2D panorama: count the spines on its pieces and attach the result to the packet (the best guess switches only for low/medium reliability and >= 75% registration).
        Runs BEFORE the 3D map (the queue is FIFO), so the measurement attach that follows starts from a packet that already has the count. Never fails the queue."""
        if self.on_count is None:
            return
        with self._lock:
            step = dict(self._records[run_id]['steps']['panorama'])
        if step.get('state') != 'done' or not step.get('out_run'):
            self._set_count(run_id, {'state': 'skipped', 'reason': 'no finished 2D panorama to count from'})
            return
        self._set_count(run_id, {'state': 'running'})
        try:
            with self._packet_lock:
                outcome = {'state': 'done', **(self.on_count(run_id, step['out_run']) or {})}
        except Exception as exc:   # noqa: BLE001 - the maps are done; a count failure is reported, never fatal
            outcome = {'state': 'failed', 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
        self._set_count(run_id, outcome)
        self._instances(run_id)

    def _instances(self, run_id):
        """After the panorama count (and again after the measurement, which rebuilds the packet from its own backup): merge track groups into physical instances and rebuild
        the packet's book lines in place (pipeline.instances.run_step). Never fails the queue; the outcome is recorded."""
        if self.on_instances is None:
            return
        with self._lock:
            step = dict(self._records[run_id]['steps']['panorama'])
        pan_out = step.get('out_run') if step.get('state') == 'done' else None
        self._set_instances(run_id, {'state': 'running'})
        try:
            with self._packet_lock:
                outcome = {'state': 'done', **(self.on_instances(run_id, pan_out) or {})}
        except Exception as exc:   # noqa: BLE001 - the maps are done; an instance failure is reported, never fatal
            outcome = {'state': 'failed', 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
        self._set_instances(run_id, outcome)

    def _set_instances(self, run_id, instances):
        with self._lock:
            rec = self._records[run_id]
            rec['instances'] = instances
            self._recompute(rec)
            self._save(rec)

    def _set_count(self, run_id, count):
        with self._lock:
            rec = self._records[run_id]
            rec['count'] = count
            self._recompute(rec)
            self._save(rec)

    def _measure(self, run_id):
        """After a 3D map: automatic metric scale, spine sizes and room geometry from it, attached to the packet. Never fails the queue; the outcome is recorded."""
        if self.on_built is None:
            return
        with self._lock:
            rec = self._records[run_id]
            step = dict(rec['steps']['recon3d'])
        if step.get('state') != 'done' or not step.get('out_run'):
            self._set_measure(run_id, {'state': 'skipped', 'reason': 'no finished 3D map to measure'})
            return
        self._set_measure(run_id, {'state': 'running'})
        try:
            with self._packet_lock:
                outcome = {'state': 'done', **(self.on_built(run_id, step['out_run']) or {})}
        except Exception as exc:   # noqa: BLE001 - the maps are done; a measurement failure is reported, never fatal
            outcome = {'state': 'failed', 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
        self._set_measure(run_id, outcome)
        if outcome.get('state') == 'done' and (self._records[run_id].get('instances') or {}).get('state') in ('done', 'failed'):
            self._instances(run_id)       # the measurement attach starts from its own packet backup: put the instance lines back on top (idempotent)

    def _set_measure(self, run_id, measure):
        with self._lock:
            rec = self._records[run_id]
            rec['measure'] = measure
            self._recompute(rec)
            self._save(rec)

    def _build(self, run_id, kind):
        """Start one build job as soon as the run is quiet and no other build is running; follow it to the end."""
        started = time.time()
        with self._lock:
            old = dict(self._records[run_id]['steps'][kind])
            job_id = old.get('job_id') if old.get('state') == 'running' else None        # a job that outlived a server restart
        if job_id is None and self._finished(old.get('out_run'), kind):                 # finished while the server was down: never rebuilt
            return {'state': 'done', 'error': None, 'note': None, 'finished_at': time.time(), **self._result_summary(old['out_run'], kind)}
        if job_id:
            self._step(run_id, kind, state='running')
        elif old.get('attempts', 0) >= MAX_ATTEMPTS:
            return {'state': 'failed', 'error': f'the build was interrupted {old["attempts"]} times before it finished; press Build again'}
        while job_id is None:
            if self._stop.is_set():
                return {'state': 'queued', 'note': 'server stopping'}
            try:
                job = self.jobs.start(run_id, kind)
                job_id = job['job_id']
                self._step(run_id, kind, state='running', job_id=job_id, out_run=job['out_run'], started_at=time.time(), error=None,
                           attempts=old.get('attempts', 0) + 1)
            except RunStillWriting as exc:
                if time.time() - started > MAX_WAIT_S:
                    return {'state': 'failed', 'error': f'the run kept changing for {MAX_WAIT_S} s: {exc}'}
                self._step(run_id, kind, state='waiting', note='waiting for the sweep files to settle')
                self._stop.wait(self.poll_s)
            except ValueError as exc:
                if 'already running' in str(exc):                # another build (a CLI run or the previous task): wait our turn
                    self._step(run_id, kind, state='waiting', note='waiting for the running build to finish')
                    self._stop.wait(self.poll_s)
                    continue
                return {'state': 'failed', 'error': str(exc)}
            except OSError as exc:
                return {'state': 'failed', 'error': f'could not start the build: {exc}'}
        while not self._stop.is_set():
            job = next((j for j in self._safe_jobs() if j['job_id'] == job_id), None)
            if job is None:
                return {'state': 'failed', 'error': 'the build job record disappeared'}
            if job['state'] != 'running':
                break
            self._stop.wait(self.poll_s)
        else:
            return {'state': 'queued', 'note': 'server stopping'}
        if job['state'] == 'done':
            return {'state': 'done', 'job_id': job_id, 'out_run': job['out_run'], 'error': None, 'finished_at': time.time(), **self._result_summary(job['out_run'], kind)}
        return {'state': 'failed', 'job_id': job_id, 'out_run': job['out_run'], 'error': job.get('error') or f"build {job['state']}", 'finished_at': time.time()}

    def _result_summary(self, out_run, kind):
        try:
            d = json.loads((self.runs / out_run / ('recon3d.json' if kind == 'recon3d' else 'panorama.json')).read_text())
        except (OSError, ValueError):
            return {}
        if kind == 'recon3d':
            return {'book_count': d.get('book_count'), 'markers': d.get('markers_total'), 'registered_ratio': d.get('registered_ratio')}
        return {'book_count': d.get('book_count'), 'markers': d.get('markers_total'), 'registered_ratio': (d.get('metrics') or {}).get('registered_ratio')}
