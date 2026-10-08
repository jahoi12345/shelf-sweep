"""The hosted pair oracle (pipeline/pair_oracle.py) as an opt-in, model-read merge evidence kind of the instance layer.
Synthetic groups and a fake transport only: nothing here says how the real answers agree with a blinded review."""
import asyncio
import json

import cv2
import numpy as np

from pipeline import contracts as C
from pipeline import instances as IN
from pipeline import pair_oracle as PO
from providers import gemini as G

FRAMES = {f'f{i:05d}': {'timestamp_s': i * 0.2, 'quality': {'sharpness': 100.0}} for i in range(8)}
SEGS = {f: (0 if i < 4 else 1) for i, f in enumerate(FRAMES)}


def _group(gid, obs):
    return {'instance_id': gid, 'observation_ids': sorted(obs), 'count_tier': 'confident', 'frame_ref': obs[0].split('-o')[0]}


def _pair(groups, oids, answer, ratio=1.0, **kw):
    return {'groups': groups, 'observation_ids': oids, 'sources': ['uncertain_association'], 'pair_key': 'k' + ''.join(groups), 'height_ratio': ratio,
            'oracle': {'answer': answer, 'reason': 'same blue spine', 'request_key': 'rq1', 'model': 'm'}, **kw}


def test_links_need_same_and_the_geometric_checks_and_never_use_validation_pairs():
    art = {'pairs': [_pair(['a', 'b'], ['f00001-o000', 'f00005-o000'], 'same'),
                     _pair(['a', 'c'], ['f00001-o000', 'f00006-o000'], 'different'),
                     _pair(['a', 'd'], ['f00001-o000', 'f00007-o000'], 'cannot_tell'),
                     _pair(['b', 'e'], ['f00005-o000', 'f00002-o000'], 'same', ratio=1.6),
                     _pair(['b', 'f'], ['f00005-o000', 'f00005-o001'], 'same'),
                     _pair(['c', 'g'], ['f00006-o000', 'f00003-o000'], 'same', validation_only=True)]}
    links, refused = PO.links(art)
    assert [lk['groups'] for lk in links] == [['a', 'b']]
    lk = links[0]
    assert lk['kind'] == PO.KIND == IN.KIND_ORACLE and lk['label'] == 'merged by hosted pair adjudication (model-read)' and lk['request_key'] == 'rq1'
    assert sorted(r['refused'].split(' ')[0] for r in refused) == ['height', 'same']


def test_oracle_links_merge_through_the_same_union_rules_and_the_line_says_model_read():
    books = [_group('bk-f00000-o000', ['f00000-o000', 'f00001-o000']), _group('bk-f00005-o000', ['f00005-o000', 'f00006-o000']),
             _group('bk-f00001-o003', ['f00001-o003']), _group('bk-f00002-o000', ['f00002-o000']), _group('bk-f00007-o000', ['f00007-o000'])]
    ev = lambda a, b, oa, ob: {'groups': [a, b], 'kind': IN.KIND_ORACLE, 'label': IN.ORACLE_LABEL, 'observation_ids': [oa, ob], 'frame_ids': [oa[:6], ob[:6]], 'answer': 'same'}
    oracle = [ev('bk-f00000-o000', 'bk-f00005-o000', 'f00000-o000', 'f00005-o000'),
              ev('bk-f00005-o000', 'bk-f00001-o003', 'f00005-o000', 'f00001-o003'),       # bk-f00001-o003 shares f00001 with the merged instance: refused
              ev('bk-f00002-o000', 'bk-f00007-o000', 'f00002-o000', 'f00007-o000')]       # contradicting readings: vetoed
    readings = [{'observation_id': 'f00002-o000', 'visible_text': 'Crimen y castigo', 'read_confidence': 0.8},
                {'observation_id': 'f00007-o000', 'visible_text': 'Un humanista en la Espana de Franco', 'read_confidence': 0.8}]
    off = IN.build({'books': books}, FRAMES, SEGS, readings=readings)
    assert off['stats']['instances'] == 5, 'no oracle links: nothing merged (the default)'
    art = IN.build({'books': books}, FRAMES, SEGS, readings=readings, oracle=oracle)
    inst = {i['instance_id']: i for i in art['instances']}
    assert inst['bk-f00000-o000']['member_group_ids'] == ['bk-f00000-o000', 'bk-f00005-o000'] and inst['bk-f00000-o000']['evidence_kinds'] == [IN.KIND_ORACLE]
    assert 'bk-f00001-o003' in inst and 'bk-f00002-o000' in inst and 'bk-f00007-o000' in inst
    reasons = sorted(r['reason'].split(':')[0] for r in art['rejected_unions'])
    assert reasons == ['reading contradiction', 'same-frame exclusion'] and art['stats']['merged_by'] == {IN.KIND_ORACLE: 1}
    members = [{'id': m, 'evidence': {'observation_ids': next(b['observation_ids'] for b in books if b['instance_id'] == m)}, 'issues': []}
               for m in inst['bk-f00000-o000']['member_group_ids']]
    line = IN.merge_lines(members, inst['bk-f00000-o000'])
    assert 'merged by hosted pair adjudication (model-read)' in line['issues'] and line['merge_evidence_kinds'] == [IN.KIND_ORACLE]


def test_the_oracle_is_off_by_default_and_the_flag_or_cli_reads_the_runs_file(tmp_path, monkeypatch):
    monkeypatch.delenv('LIBRARY_PAIR_ORACLE', raising=False)
    assert IN.oracle_links_for(tmp_path, None) == (None, None)
    monkeypatch.setenv('LIBRARY_PAIR_ORACLE', '1')
    links, info = IN.oracle_links_for(tmp_path, None)
    assert links is None and info['reason'] == 'no pair_oracle.json'
    (tmp_path / PO.ARTIFACT).write_text(json.dumps({'model': 'm', 'pairs': [_pair(['a', 'b'], ['f00001-o000', 'f00005-o000'], 'same')]}))
    links, info = IN.oracle_links_for(tmp_path, None)
    assert [lk['groups'] for lk in links] == [['a', 'b']] and info['links'] == 1 and 'model-read' in info['label']
    monkeypatch.setenv('LIBRARY_PAIR_ORACLE', '0')
    assert IN.oracle_links_for(tmp_path, None) == (None, None) and IN.oracle_links_for(tmp_path, True)[1]['links'] == 1


def _run(tmp_path):
    """Two frames, three upright spines each; groups: spine 0 seen in both frames as two groups (a near-tie edge), spine 1 one group, spine 2 two groups."""
    frames = []
    obs = []
    for i in range(2):
        fid = f'f{i:05d}'
        img = np.full((480, 640, 3), 200, np.uint8)
        for j, x in enumerate([100, 250, 400]):
            img[60:420, x:x + 60] = (40 * (j + 1), 80, 160)
            obs.append({'observation_id': f'{fid}-o{j:03d}', 'frame_id': fid, 'kind': 'book', 'bbox_xyxy': [x, 60, x + 60, 420],
                        'polygon_xy': [[x, 60], [x + 60, 60], [x + 60, 420], [x, 420]]})
        cv2.imwrite(str(tmp_path / f'{fid}.png'), img)
        frames.append({'frame_id': fid, 'image_path': f'{fid}.png', 'timestamp_s': i * 0.5, 'width_px': 640, 'height_px': 480, 'quality': {'sharpness': 100.0}})
    books = [_group('bk-f00000-o000', ['f00000-o000']), _group('bk-f00001-o000', ['f00001-o000']), _group('bk-f00000-o001', ['f00000-o001', 'f00001-o001']),
             _group('bk-f00000-o002', ['f00000-o002']), _group('bk-f00001-o002', ['f00001-o002'])]
    unc = [{'observation_ids': ['f00000-o000', 'f00001-o000'], 'score': 0.9}, {'observation_ids': ['f00000-o000', 'f00000-o001'], 'score': 0.5}]
    (tmp_path / 'manifest.json').write_text(json.dumps(C.envelope('t', 'frames', {'frames': frames}, fixture=True)))
    (tmp_path / 'candidates.json').write_text(json.dumps(C.envelope('t', 'detect', {'observations': obs}, fixture=True)))
    (tmp_path / 'inventory.json').write_text(json.dumps({'data': {'books': books, 'uncertain_associations': unc}}))
    (tmp_path / 'readings.json').write_text(json.dumps({'data': {'readings': [
        {'observation_id': 'f00000-o002', 'visible_text': 'LITTLE WOMEN Alcott', 'read_confidence': 0.6},
        {'observation_id': 'f00001-o002', 'visible_text': 'Little Women', 'read_confidence': 0.6}]}}))
    return tmp_path


def test_proposals_render_blinded_sheets_ask_in_batches_and_store_every_answer(tmp_path):
    run = _run(tmp_path / 'run') if (tmp_path / 'run').mkdir() is None else None
    seen = []

    async def fake(model, images, prompt, config, schema):
        seen.append((len(images), prompt))
        n = len(images)
        return json.dumps({'answers': [{'sheet': k, 'answer': 'same', 'reason': 'identical spine'} for k in range(1, n + 1)]}), {'total_tokens': 10}
    G.set_transport(fake)
    try:
        art = PO.run(run, tmp_path / 'out', panorama_dir=tmp_path / 'none')
    finally:
        G.set_transport(None)
    groups = [p['groups'] for p in art['pairs']]
    assert groups == [['bk-f00000-o000', 'bk-f00001-o000'], ['bk-f00000-o002', 'bk-f00001-o002']], 'near-tie edge + same text; same-frame edge excluded'
    assert art['proposals'] == {'uncertain_association': 1, 'same_text': 1} and art['skipped'] == {'uncertain_association:same_frame': 1}
    assert seen == [(2, PO.PROMPT.format(n=2))] and art['hosted']['calls'] == 1
    assert all(p['oracle']['answer'] == 'same' and p['oracle']['request_key'] for p in art['pairs']) and art['links_accepted'] == 2
    assert all((tmp_path / 'out' / 'sheets' / f"{p['pair_key']}.png").is_file() for p in art['pairs'])
    on_disk = json.loads((tmp_path / 'out' / PO.ARTIFACT).read_text())
    assert on_disk['evidence_kind'] == 'hosted_pair_adjudication' and 'model-read' in on_disk['label'] and 'not truth' in on_disk['note']
    # second run: served from the cache, no request
    G.set_transport(fake)
    try:
        again = PO.run(run, tmp_path / 'out', panorama_dir=tmp_path / 'none')
    finally:
        G.set_transport(None)
    assert len(seen) == 1 and again['hosted']['cache_hits'] == 1


def test_the_call_cap_leaves_pairs_unasked_and_unasked_pairs_never_merge(tmp_path):
    run = _run(tmp_path / 'run') if (tmp_path / 'run').mkdir() is None else None

    async def fake(model, images, prompt, config, schema):
        return json.dumps({'answers': [{'sheet': 1, 'answer': 'same', 'reason': 'x'}]}), {}
    G.set_transport(fake)
    try:
        rd = PO.RunData(run, panorama_dir=tmp_path / 'none')
        prop = PO.propose(rd)
        res = asyncio.run(PO.adjudicate(rd, prop['pairs'], tmp_path / 'out', cfg={'batch': 1}, calls_left=1))
    finally:
        G.set_transport(None)
    answers = [res['answers'][p['pair_key']] for p in prop['pairs']]
    assert answers[0]['answer'] == 'same' and answers[1] == {'answer': None, 'error': 'call cap reached; not asked'} and res['not_asked_batches'] == 1
    # the cap counts NEW requests only: with no calls left, the cached batch is still served and the uncached one is not asked
    G.set_transport(fake)
    try:
        again = asyncio.run(PO.adjudicate(rd, prop['pairs'], tmp_path / 'out', cfg={'batch': 1}, calls_left=0))
    finally:
        G.set_transport(None)
    assert again['answers'][prop['pairs'][0]['pair_key']]['cache_hit'] is True and again['calls'] == 0 and again['not_asked_batches'] == 1
