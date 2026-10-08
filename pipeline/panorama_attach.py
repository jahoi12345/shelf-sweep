"""Attach the panorama count (pipeline.panorama_count) to a run's claim_packet.json.

    python -m pipeline.panorama_attach --run runs/<run> [--count runs/<run>/panorama_count.json]

Same in-place pattern as pipeline.auto_attach: the packet is patched, its own verification (pipeline.packet.verify_packet) is re-run and nothing is written when it
fails, the original packet is kept once as ``claim_packet.pre_panorama_count.json``, the report is re-rendered, and attaching twice gives the same packet (the track
estimate is stored inside the packet's `count.panorama.track_estimate`, so a later attach does not depend on the backup and does not undo the 3D measurement that
pipeline.auto_attach put in the same file). pipeline.packet.build_data applies the same patch to every packet it builds from a run directory that holds
panorama_count.json, so a rebuild after a user correction keeps it.

Which estimator is the HEADLINE (`totals.book_count`), user decision 2026-10-08:
  * the PANORAMA count whenever the panorama passes its usability check (`usable`: the detector saw >= 60% of the spines the track count trusts on the pieces) and
    >= 75% of the frames registered in the panorama AND its count lies inside the line count's own range [book_count_lower, book_count_upper] (the
    instance-layer reconciliation's guard); its range is the panorama's detector-confidence range, labelled "panorama range (may not contain the true count)"
    (on the held-out sweeps the panorama's mean absolute error was 10% vs 18% for the lines, but its range missed the hand count on 5 of 8 usable sweeps);
  * otherwise the LINE count (the book lines of the line layer: physical instances, or track groups without the instance layer) and its association range.
The line count is always kept beside it: `totals.book_lines` (= number of book lines), `book_lines_lower/upper/range`, and `count.lines` (the line layer's own figures).
`totals.headline_estimator` names the estimator ('panorama' | 'lines'); `totals.count_estimator` keeps naming the LINE layer ('instances' | 'track'; 'panorama' only on
older packets). `count.headline` carries the reason text. `decide` (count reliability low/medium + registration + usable) is the older, stricter
rule; it now only gates the instance layer's panorama-overlap merges and reconciliation (pipeline/instances.py).
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

BACKUP = 'claim_packet.pre_panorama_count.json'
ARTIFACT = 'panorama_count.json'
MIN_REGISTRATION = 0.75
ELIGIBLE_RELIABILITY = ('low', 'medium')
TRACK_KEYS = ('book_count_lower', 'book_count_upper', 'book_count_range', 'confident_groups', 'count_range_basis')
_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}


def load(run_dir: str | Path) -> dict[str, Any] | None:
    """The run's panorama_count.json when it is there and belongs to this run (cached by mtime: the Sweep page rebuilds the packet data on every poll)."""
    path = Path(run_dir) / ARTIFACT
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _CACHE.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        pc = json.loads(path.read_text())
        ok = isinstance(pc, dict) and pc.get('artifact_type') == 'panorama_count' and pc.get('run_id') == Path(run_dir).name and isinstance(pc.get('panorama_count'), int)
    except (OSError, ValueError):
        pc, ok = None, False
    _CACHE[str(path)] = (mtime, pc if ok else None)
    return pc if ok else None


def decide(totals: dict[str, Any], pc: dict[str, Any]) -> tuple[bool, str]:
    """(use the panorama count?, reason). The reliability is the packet's canonical one (the live monitor's reading of the same frames)."""
    rel = totals.get('count_reliability')
    basis = totals.get('count_reliability_basis')      # set by pipeline.instances when its own reading is weaker than the packet's
    rel_text = f"{rel} ({basis})" if basis else rel
    ratio = (pc.get('registration') or {}).get('registered_ratio') or 0.0
    if not pc.get('usable'):
        return False, f"the panorama count is not usable here: {pc.get('unusable_reason') or 'unspecified'}; the track count stays the best guess"
    if rel not in ELIGIBLE_RELIABILITY:
        return False, (f"count reliability is {rel or 'unknown'} (the panorama estimator replaces the track count only when it is low or medium): the track count stays the best guess; "
                       f"the panorama count {pc['panorama_count']} is a cross-check")
    if ratio < MIN_REGISTRATION:
        return False, f"only {ratio:.0%} of the frames are in the panorama (needs >= {MIN_REGISTRATION:.0%}): the track count stays the best guess"
    return True, (f"count reliability {rel_text} and {ratio:.0%} of the frames registered in the panorama: the best guess is the panorama-detector count "
                  f"(the track count is kept as `book_count_track`)")


def _compact(pc: dict[str, Any], used: bool, why: str, track: dict[str, Any]) -> dict[str, Any]:
    evidence = pc.get('merge_evidence') or []
    return {
        'estimator_used': 'panorama' if used else 'track', 'used': used, 'reason': why, 'artifact': ARTIFACT, 'panorama_run': pc.get('panorama_run'),
        'panorama_count': pc.get('panorama_count'), 'range_panorama': pc.get('range_panorama'), 'count_if_confidence': pc.get('count_if_confidence'),
        'count_if_collapsed_pieces_not_counted': pc.get('count_if_collapsed_pieces_not_counted'),
        'panorama_spines': pc.get('panorama_spines'), 'collapsed_piece_groups': pc.get('collapsed_piece_groups'), 'extra_unregistered_groups': pc.get('extra_unregistered_groups'),
        'usable': pc.get('usable'), 'unusable_reason': pc.get('unusable_reason'), 'detector_support': pc.get('detector_support'),
        'registration': pc.get('registration'), 'parameters': pc.get('parameters'), 'seconds': pc.get('seconds'),
        'panorama_components': [{k: c.get(k) for k in ('component_id', 'frames', 'class', 'class_reason', 'counted_from', 'spines_detected', 'spines_counted', 'groups_placed', 'sheet')}
                                for c in (pc.get('panorama_components') or [])],
        'sheets': pc.get('sheets'), 'piece_links': pc.get('piece_links'),
        'unregistered_groups': {k: v for k, v in (pc.get('unregistered_groups') or {}).items() if k != 'added_ids'} | {'added_ids': (pc.get('unregistered_groups') or {}).get('added_ids', [])[:50]},
        'collapsed_pieces': {k: v for k, v in (pc.get('collapsed_pieces') or {}).items() if k != 'ids'}, 'far_pieces': {'groups_not_counted': (pc.get('far_pieces') or {}).get('groups_not_counted')},
        'reconciliation': pc.get('reconciliation'),
        'merge_evidence_total': len(evidence), 'merge_evidence': evidence[:40],
        'merge_evidence_note': f'first {min(40, len(evidence))} of {len(evidence)} lines; all of them are in {ARTIFACT}',
        'track_estimate': track, 'limits': pc.get('limits'),
    }


def apply(p: dict[str, Any], pc: dict[str, Any]) -> dict[str, Any]:
    """Patch packet `data` in place (idempotent): record the panorama count (`count.panorama`) and choose the headline (`headline`). Returns {'used': bool, 'reason': str}."""
    t, c = p['totals'], p['count']
    lines = len(p['books'])
    track = (c.get('panorama') or {}).get('track_estimate') or {k: copy.deepcopy(t[k]) for k in TRACK_KEYS if k in t}
    t['panorama_count'] = pc['panorama_count']
    from pipeline import packet as P
    p['review_queue'] = [x for x in P.flat_review(p.get('review_queue')) if x.get('code') != 'count_estimator_panorama']
    if c.get('estimator') == 'instances' or t.get('count_estimator') == 'instances':
        # The lines are physical instances (pipeline/instances.py); their figures are already in the packet (count.lines once a headline was chosen)
        c['panorama'] = _compact(pc, False, 'the book lines are physical instances (count.instances)', track)
        c['panorama']['estimator_used'] = 'instances'
        res = headline(p, fresh=False)
        return {'used': res['estimator'] == 'panorama', 'reason': res['reason']}
    # the line layer is the track groups: always start from the track estimate (an older packet may still hold the old panorama best guess)
    for k in TRACK_KEYS:
        if k in track:
            t[k] = copy.deepcopy(track[k])
        else:
            t.pop(k, None)
    t.pop('book_count_track', None)
    t['count_estimator'] = 'track'
    t['book_count'] = lines
    t.setdefault('notes', {})['book_count'] = ("best guess of physical books (confident + likely groups); unconfirmed single sightings are listed apart; human physical count unverified"
                                               if track.get('book_count_range') else "proposed deduplicated book groups; human physical count unverified; see lower/upper association bounds")
    c.update(best_guess=lines, book_groups=lines, estimator='track', **{k: track.get(k) for k in ('book_count_range', 'book_count_lower', 'book_count_upper', 'count_range_basis')
                                                                      if k in track and track.get(k) is not None})
    p.setdefault('uncertainty', {}).update(book_count_lower=t.get('book_count_lower'), book_count_upper=t.get('book_count_upper'))
    from pipeline.packet import _band
    c['uncertainty_band'] = _band(t)                                   # the track band (pipeline.packet.build_data), not a panorama band left by an earlier attach
    c['panorama'] = _compact(pc, False, 'the book lines are track groups', track)
    res = headline(p, fresh=True)
    return {'used': res['estimator'] == 'panorama', 'reason': res['reason']}


# ------------------------------------------------------------------------------------------ the headline count (user decision 2026-10-08)
HEADLINE_RANGE_LABEL = {'panorama': 'panorama range (may not contain the true count)', 'lines': 'range of the book lines (association bounds)'}
LINES_LABEL = 'book lines listed (some may be duplicates)'
HELD_OUT_EVIDENCE = ('on the held-out sweeps (YugoTest12/13/14/16, IESFULLTest1) the panorama count was off by 10% on average against the hand counts, the line count by 18%; '
                     'the panorama range missed the hand count on 5 of 8 usable sweeps')


def headline_decision(pan: dict[str, Any] | None) -> tuple[bool, str]:
    """(headline = the panorama count?, reason). `pan`: panorama_count.json or its compact copy in the packet (`count.panorama`)."""
    if not pan or pan.get('panorama_count') is None:
        return False, 'there is no 2D panorama count for this sweep'
    n = pan['panorama_count']
    if not pan.get('usable'):
        return False, f"the panorama count {n} failed its usability check ({pan.get('unusable_reason') or 'unspecified'})"
    ratio = (pan.get('registration') or {}).get('registered_ratio') or 0.0
    if ratio < MIN_REGISTRATION:
        return False, f"the panorama count {n} failed its usability check: only {ratio:.0%} of the frames registered in the panorama (needs >= {MIN_REGISTRATION:.0%})"
    sup = pan.get('detector_support')
    sup_text = f"the spine detector saw {sup:.0%} of the trusted track groups (needs >= 60%)" if isinstance(sup, (int, float)) else 'the spine detector support check passed'
    return True, f"the panorama passed its usability check: {sup_text} and {ratio:.0%} of the frames registered (needs >= {MIN_REGISTRATION:.0%})"


def headline(p: dict[str, Any], *, fresh: bool = True) -> dict[str, Any]:
    """Choose the headline book count (idempotent). The line layer's figures are the totals as they stand when ``fresh`` (the line layer has just written them:
    pipeline.instances.apply, the track path of ``apply``); otherwise they come back from ``count.lines``. The panorama facts come from ``count.panorama``.
    Returns {'estimator': 'panorama'|'lines', 'book_count', 'book_lines', 'reason'}."""
    t, c = p['totals'], p.setdefault('count', {})
    if fresh or not c.get('lines'):
        c['lines'] = {'estimator': t.get('count_estimator') or 'track', 'count': t.get('book_count'), 'lower': t.get('book_count_lower'), 'upper': t.get('book_count_upper'),
                      'range': copy.deepcopy(t.get('book_count_range')), 'basis': t.get('count_range_basis'), 'uncertainty_band': copy.deepcopy(c.get('uncertainty_band')),
                      'note': (t.get('notes') or {}).get('book_count'), 'label': LINES_LABEL}
    ln = c['lines']
    pan = c.get('panorama') or None
    use, why = headline_decision(pan)
    n_lines = ln['count']
    if use and ln['lower'] is not None and ln['upper'] is not None:
        # the instance-layer reconciliation's guard (pipeline/instances.py RECONCILE): the two estimators must agree within the lines' own association range
        n_pan = max(0, int(pan['panorama_count']) - int(t.get('user_excluded_count') or 0))
        if not ln['lower'] <= n_pan <= ln['upper']:
            use, why = False, (f"{why}, but the panorama count {n_pan} lies outside the book lines' own range [{ln['lower']}, {ln['upper']}]: "
                               f"the two estimators disagree beyond the association uncertainty")
        else:
            why += f", and the panorama count {n_pan} lies inside the book lines' own range [{ln['lower']}, {ln['upper']}]"
    t.update(book_lines=n_lines, book_lines_lower=ln['lower'], book_lines_upper=ln['upper'], book_lines_range=copy.deepcopy(ln['range']))
    if use:
        excl = int(t.get('user_excluded_count') or 0)
        n = max(0, int(pan['panorama_count']) - excl)
        rp = pan.get('range_panorama') or [n, n]
        lo, hi = min(max(0, int(rp[0]) - excl), n), max(max(0, int(rp[1]) - excl), n)
        conf = ', '.join(f'{k}: {v}' for k, v in (pan.get('count_if_confidence') or {}).items())
        basis = (f"panorama range: lowest and highest panorama count over the detector confidences tried ({conf or 'n/a'})"
                 + (f", minus {excl} line(s) excluded by the user" if excl else '') + '; it may not contain the true count')
        reason = (f"Headline {n} books = the 2D panorama count, because {why}. {n_lines} book lines are listed (some may be duplicates). The range {lo}-{hi} is the panorama "
                  f"range and may not contain the true count. Why the panorama: {HELD_OUT_EVIDENCE}.")
        t.update(book_count=n, book_count_lower=lo, book_count_upper=hi, book_count_range=[lo, hi], count_range_basis=basis)
        t.setdefault('notes', {})['book_count'] = ('headline = the 2D panorama count (spines detected on the stitched pieces, fused across registered pieces); '
                                                   'book_lines = the book lines listed (some may be duplicates); human physical count unverified')
        c.update(best_guess=n, book_count_lower=lo, book_count_upper=hi, book_count_range=[lo, hi], count_range_basis=basis,
                 uncertainty_band={'low': lo, 'best': n, 'high': hi, 'note': 'panorama range (detector confidence); may not contain the true count'})
    else:
        n, lo, hi = n_lines, ln['lower'], ln['upper']
        reason = (f"Headline {n} books = the book line count, because {why}. "
                  + (f"The panorama count {pan['panorama_count']} is only a cross-check. " if pan and pan.get('panorama_count') is not None else '')
                  + 'The range is the association range of the book lines.')
        t.update(book_count=n, book_count_lower=lo, book_count_upper=hi)
        for k, v in (('book_count_range', ln['range']), ('count_range_basis', ln['basis'])):
            if v is not None:
                t[k] = copy.deepcopy(v)
                c[k] = copy.deepcopy(v)
            else:
                t.pop(k, None)
        if ln.get('note'):
            t.setdefault('notes', {})['book_count'] = ln['note']
        c.update(best_guess=n, book_count_lower=lo, book_count_upper=hi)
        if ln.get('uncertainty_band') is not None:
            c['uncertainty_band'] = copy.deepcopy(ln['uncertainty_band'])
        else:
            c.pop('uncertainty_band', None)
    est = 'panorama' if use else 'lines'
    t['headline_estimator'] = est
    t['book_count_range_label'] = HEADLINE_RANGE_LABEL[est]
    t.setdefault('notes', {})['book_lines'] = f'number of book lines in the packet ({LINES_LABEL}); equals book_count when headline_estimator is "lines"'
    c['headline'] = {'estimator': est, 'book_count': n, 'range': [lo, hi] if lo is not None and hi is not None else None, 'range_label': HEADLINE_RANGE_LABEL[est],
                     'book_lines': n_lines, 'book_lines_label': LINES_LABEL, 'line_estimator': ln['estimator'], 'panorama_count': (pan or {}).get('panorama_count'),
                     'reason': reason,
                     'rule': 'panorama count when the panorama passes its usability check (detector support >= 60%, >= 75% of frames registered), otherwise the line count (user decision 2026-10-08)'}
    if pan:
        pan.update(used=use, estimator_used='panorama' if use else ln['estimator'], reason=why)
    p.setdefault('uncertainty', {}).update(book_count_lower=lo, book_count_upper=hi)
    from pipeline import packet as P
    q = [x for x in P.flat_review(p.get('review_queue')) if x.get('code') not in ('count_headline', 'count_estimator_panorama')]
    q.append({'ref_id': 'capture', 'code': 'count_headline', 'severity': 'info', 'reason': reason})
    p['review_queue'] = P.consolidate_review(q)             # one entry per ref_id (P imported above)
    return {'estimator': est, 'book_count': n, 'book_lines': n_lines, 'reason': reason}


def attach(run_dir: str | Path, count_path: str | Path | None = None, *, render: bool = True) -> dict[str, Any]:
    from pipeline import contracts as C
    from pipeline import packet as P
    run_dir = Path(run_dir)
    pk = run_dir / C.ARTIFACTS['packet']
    if not pk.is_file():
        return {'attached': False, 'reason': 'no claim_packet.json'}
    pc = json.loads(Path(count_path).read_text()) if count_path else load(run_dir)
    if not pc:
        return {'attached': False, 'reason': f'no {ARTIFACT} for this run'}
    backup = run_dir / BACKUP
    if not backup.is_file():
        shutil.copy2(pk, backup)
    env = C.read_json(pk)
    if env['data'].get('fixture'):
        return {'attached': False, 'reason': 'fixture packet'}
    manifest = C.read_json(run_dir / C.ARTIFACTS['frames'])
    res = apply(env['data'], pc)
    P.finalize_contract(env['data'], run_dir)
    problems = P.verify_packet(env, manifest, run_dir)
    if problems:
        return {'attached': False, 'reason': 'packet verification failed after the panorama count was attached', 'problems': problems[:10]}
    C.write_json(pk, env)
    if render:
        from pipeline import report as R
        R.render(pk, run_dir)
        if (run_dir / 'report.pdf').is_file():
            R.render_print_pdf(pk, run_dir)
    t = env['data']['totals']
    return {'attached': True, 'estimator': 'panorama' if res['used'] else (t.get('count_estimator') or 'track'), 'headline_estimator': t.get('headline_estimator'),
            'book_count': t['book_count'], 'book_lines': t.get('book_lines'), 'panorama_count': pc['panorama_count'], 'track_count': len(env['data']['books']),
            'book_count_range': t.get('book_count_range'), 'reason': res['reason']}


def run_count_step(runs_dir: str | Path, run_id: str, panorama_out_run: str, *, timeout_s: float = 400.0) -> dict[str, Any]:
    """The post-sweep step (app.post_sweep): count the spines on `runs/<panorama_out_run>` in a niced subprocess (the detector and OpenCV must not stall the server),
    write panorama_count.json and attach it to the packet. Returns the summary recorded in the queue state. Raises on failure (the queue records it; the maps are unaffected)."""
    import os
    import subprocess
    runs = Path(runs_dir)
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, '-m', 'pipeline.panorama_count', '--run', str(runs / run_id), '--panorama', str(runs / panorama_out_run), '--attach']
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=timeout_s, preexec_fn=lambda: os.nice(10))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-1:] or ['exited with code ' + str(proc.returncode)]
        raise RuntimeError(f'panorama count failed: {tail[0][:200]}')
    brief = json.loads(proc.stdout[proc.stdout.index('{'):])
    att = brief.get('attach') or {}
    est = att.get('estimator') or 'track'
    out = {'panorama_count': brief['panorama_count'], 'range_panorama': brief['range_panorama'], 'usable': brief['usable'], 'estimator': est, 'attached': bool(att.get('attached')),
           'seconds': (brief.get('seconds') or {}).get('total'), 'book_count': att.get('book_count')}
    out['summary'] = (f"panorama count {brief['panorama_count']} (range {brief['range_panorama'][0]}-{brief['range_panorama'][1]}), best guess from the {est} estimator"
                      + ('' if att.get('attached') else f" (not attached: {att.get('reason')})"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--count')
    ap.add_argument('--no-render', action='store_true')
    a = ap.parse_args(argv)
    print(json.dumps(attach(a.run, a.count, render=not a.no_render), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
