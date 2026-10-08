"""Hosted pair oracle: a model-read adjudication of track-group pairs that the pipeline did NOT merge.

Evidence kind ``hosted_pair_adjudication`` for the instance layer (pipeline/instances.py). It is MODEL-READ, not truth: two models
agreeing is not a verified merge; only a person looking at the sheets is.

1. ``propose``: candidate pairs of track groups per run, from three sources (all must pass the same-frame exclusion and must not be
   in one instance already):
     (a) ``uncertain_association``: dedup's rejected near-tie edges (``inventory.uncertain_associations``) between different groups;
     (b) ``piece_boundary``: groups in the last frames of a stitched piece and the first frames of the time-adjacent next piece
         (gap <= ``boundary_gap_s``) whose in-frame spine centres are within ``boundary_spine_widths`` spine widths (no registration is
         stored between pieces, so the boundary frames' own positions are the only geometry; a camera move between them is not caught);
     (c) ``same_text``: groups whose best readable spine texts (>= 8 characters) agree (pipeline.instances.text_overlap >= ``text_agree``),
         at most ``text_partners`` partners per group. Text only PROPOSES; it never merges.
2. ``adjudicate``: each candidate is rendered as a blinded two-crop sheet (pipeline/pair_sheets.py conventions: native upright spine crop
   + outlined frame context per side, A/B, frame id and time, no source label) and Gemini Flash is asked "same physical book spine?
   same | different | cannot_tell + reason" (providers.gemini.structured_image_request: hosted kill-switch, disk cache, usage).
   Several sheets per call (``batch``), a hard call cap (``max_calls``). Every answer is stored with its request key.
3. ``links``: the merge links the instance layer consumes: only answer 'same' AND the geometric sanity checks: the relative spine
   height (long side / median book long side in that frame) of the two sightings within ``height_ratio`` and not the same frame. The
   instance layer adds the group-level same-frame exclusion and the reading-contradiction veto at the union.

All thresholds are declared here, before any count was seen (CONFIG). Nothing is tuned to a hand count.

    python -m pipeline.pair_oracle --run runs/<run> --out <dir> [--readings file] [--max-calls N] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

KIND = 'hosted_pair_adjudication'
LABEL = 'merged by hosted pair adjudication (model-read)'
ARTIFACT = 'pair_oracle.json'
SCHEMA = 1
CONFIG = {
    'boundary_frames': 3,            # frames at the end / start of two time-adjacent pieces
    'boundary_gap_s': 3.0,           # pieces are time-adjacent when the later one starts within this gap
    'boundary_spine_widths': 2.0,    # in-frame centre distance, in median spine widths of the two frames
    'text_min_chars': 8,
    'text_agree': 0.8,               # pipeline.instances.text_overlap of the two best readings
    'text_partners': 3,
    'height_ratio': [0.75, 1.33],    # relative spine height of the two adjudicated sightings
    'batch': 6,                      # sheets per hosted call
    'max_calls': 1500,
}
SHEET = (1080, 420)
ANSWERS = ('same', 'different', 'cannot_tell')
PROMPT = (
    'You will see {n} images, in order, labelled "Sheet 1" to "Sheet {n}" at the top. Each sheet shows two sightings, A (left) and B (right), of a book spine '
    'on a bookshelf, taken from a video at different moments. On each side: an upright crop of the spine, and the frame context with that spine outlined. '
    'For every sheet, decide whether A and B show the SAME PHYSICAL BOOK SPINE (the same object on the shelf). Two copies or volumes of one series that differ '
    'in any printed detail (number, title, colour, label) are different. Judge from the printed text, colours, wear, labels and the neighbouring spines. '
    'Answer "same", "different" or "cannot_tell" (when the crops are too blurred, small or ambiguous), with a short reason naming the visual evidence. '
    'Return exactly one entry per sheet, with its sheet number.')
RESPONSE_SCHEMA = {'type': 'object', 'properties': {'answers': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'sheet': {'type': 'integer'}, 'answer': {'type': 'string', 'enum': list(ANSWERS)}, 'reason': {'type': 'string'}},
    'required': ['sheet', 'answer', 'reason']}}}, 'required': ['answers']}


def _data(path: Path) -> Any:
    d = json.loads(Path(path).read_text())
    return d.get('data', d) if isinstance(d, dict) else d


def _long(poly_or_box: Any) -> float:
    from pipeline.instances import spine_long_side
    if poly_or_box and len(poly_or_box) == 4 and isinstance(poly_or_box[0], (list, tuple)):
        return spine_long_side(poly_or_box)
    x0, y0, x1, y1 = poly_or_box
    return float(max(x1 - x0, y1 - y0))


def _short(o: dict[str, Any]) -> float:
    x0, y0, x1, y1 = o['bbox_xyxy']
    return float(min(x1 - x0, y1 - y0))


def _centre(o: dict[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = o['bbox_xyxy']
    return (x0 + x1) / 2, (y0 + y1) / 2


class RunData:
    """Read-only view of a run: track groups, book observations, frames, current instances, panorama pieces, readings."""

    def __init__(self, run_dir: str | Path, *, readings_path: str | Path | None = None, instances_path: str | Path | None = None,
                 panorama_dir: str | Path | None = None):
        self.root = Path(run_dir)
        self.frames = {f['frame_id']: f for f in _data(self.root / 'manifest.json')['frames']}
        cands = _data(self.root / 'candidates.json')
        self.obs = {o['observation_id']: o for o in cands.get('observations', cands.get('detections', []))
                    if o.get('kind') == 'book' and o.get('frame_id') in self.frames}
        self.inv = _data(self.root / 'inventory.json')
        self.groups = {b['instance_id']: b for b in self.inv.get('books') or []}
        self.group_of = {o: g for g, b in self.groups.items() for o in b['observation_ids']}
        self.frames_of = {g: {o.split('-o')[0] for o in b['observation_ids']} for g, b in self.groups.items()}
        self.by_frame: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for o in self.obs.values():
            self.by_frame[o['frame_id']].append(o)
        ip = Path(instances_path) if instances_path else self.root / 'instances.json'
        self.instance_of: dict[str, str] = {}
        try:
            for inst in json.loads(ip.read_text()).get('instances') or []:
                for m in inst.get('member_group_ids') or []:
                    self.instance_of[m] = inst['instance_id']
        except (OSError, ValueError):
            pass
        rp = Path(readings_path) if readings_path else self.root / 'readings.json'
        try:
            self.readings = _data(rp).get('readings', [])
            self.readings_source = str(rp)
        except (OSError, ValueError, AttributeError):
            self.readings, self.readings_source = [], None
        pd = Path(panorama_dir) if panorama_dir else self.root.parent / f'{self.root.name}_panorama_v1'
        try:
            self.pieces = json.loads((pd / 'panorama.json').read_text())['components']
        except (OSError, ValueError, KeyError):
            self.pieces = []

    def instance(self, g: str) -> str:
        return self.instance_of.get(g, g)

    def same_frame(self, ga: str, gb: str) -> bool:
        return bool(self.frames_of.get(ga, set()) & self.frames_of.get(gb, set()))

    def rank(self, oid: str) -> float:
        from pipeline.pair_sheets import _rank
        return _rank(self.obs[oid], self.frames)

    def rel_height(self, oid: str) -> float | None:
        """Spine long side relative to the median book long side in its frame (scale-free across zoom and distance)."""
        o = self.obs.get(oid)
        if o is None:
            return None
        sides = [_long(x.get('polygon_xy') or x['bbox_xyxy']) for x in self.by_frame[o['frame_id']]]
        med = float(np.median(sides)) if sides else 0.0
        return _long(o.get('polygon_xy') or o['bbox_xyxy']) / med if med > 0 else None

    def best_pair(self, ga: str, gb: str) -> list[str] | None:
        """The sharpest, least-cropped sightings of the two groups in different frames."""
        ra = sorted((o for o in self.groups[ga]['observation_ids'] if o in self.obs), key=lambda o: (-self.rank(o), o))[:5]
        rb = sorted((o for o in self.groups[gb]['observation_ids'] if o in self.obs), key=lambda o: (-self.rank(o), o))[:5]
        best = None
        for a in ra:
            for b in rb:
                if self.obs[a]['frame_id'] != self.obs[b]['frame_id']:
                    s = self.rank(a) * self.rank(b)
                    if best is None or s > best[0]:
                        best = (s, [a, b])
        return best[1] if best else None


# --------------------------------------------------------------------------------------------------------------- proposals
def propose(rd: RunData, cfg: dict[str, Any] | None = None, *, preferred: dict[tuple[str, str], list[str]] | None = None) -> dict[str, Any]:
    """Candidate pairs (unordered group pairs) with their provenance. ``preferred``: group pair -> observation ids to show (e.g. the ids of an
    existing blinded sheet, so an earlier adjudication can be compared on the same sightings)."""
    cfg = {**CONFIG, **(cfg or {})}
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    skipped = collections.Counter()

    def add(ga: str, gb: str, source: str, prov: dict[str, Any], oids: list[str] | None = None):
        if ga not in rd.groups or gb not in rd.groups or ga == gb:
            skipped[f'{source}:unknown_or_same_group'] += 1
            return
        if rd.instance(ga) == rd.instance(gb):
            skipped[f'{source}:already_one_instance'] += 1
            return
        if rd.same_frame(ga, gb):
            skipped[f'{source}:same_frame'] += 1
            return
        key = tuple(sorted((ga, gb)))
        p = pairs.get(key)
        if p is None:
            ids = (preferred or {}).get(key)
            if not ids:
                ids = oids if oids and all(o in rd.obs for o in oids) and rd.obs[oids[0]]['frame_id'] != rd.obs[oids[1]]['frame_id'] else rd.best_pair(ga, gb)
            if not ids:
                skipped[f'{source}:no_sightings'] += 1
                return
            ids = sorted(ids, key=lambda o: 0 if rd.group_of.get(o) == key[0] else 1)
            p = pairs[key] = {'groups': list(key), 'observation_ids': ids, 'sources': [], 'provenance': []}
        if source not in p['sources']:
            p['sources'].append(source)
        p['provenance'].append({'source': source, **prov})

    # (a) dedup's rejected near-tie edges between different groups
    for e in rd.inv.get('uncertain_associations') or []:
        ids = e.get('observation_ids') or []
        if len(ids) != 2 or any(o not in rd.group_of for o in ids):
            skipped['uncertain_association:unknown'] += 1
            continue
        ga, gb = rd.group_of[ids[0]], rd.group_of[ids[1]]
        add(ga, gb, 'uncertain_association', {'observation_ids': ids, 'score': e.get('score'), 'appearance_score': e.get('appearance_score'),
                                             'reason': e.get('reason')}, oids=list(ids))

    # (b) time-adjacent stitched pieces: boundary frames, in-frame centre distance in spine widths
    def times(c):
        return sorted((float(f.get('timestamp_s') or 0.0), f['frame_id']) for f in c.get('source_frames') or [] if f.get('frame_id') in rd.frames)
    pieces = [(c['component_id'], times(c)) for c in rd.pieces]
    pieces = sorted((p for p in pieces if p[1]), key=lambda p: p[1][0][0])
    for (ca, ta), (cb, tb) in zip(pieces, pieces[1:]):
        gap = tb[0][0] - ta[-1][0]
        if gap > cfg['boundary_gap_s'] or gap < -cfg['boundary_gap_s']:
            continue
        fa = [f for _, f in ta[-cfg['boundary_frames']:] if rd.by_frame.get(f)]
        fb = [f for _, f in tb[:cfg['boundary_frames']] if rd.by_frame.get(f)]
        for x in fa:
            for y in fb:
                if x == y:
                    continue
                width = float(np.median([_short(o) for o in rd.by_frame[x] + rd.by_frame[y]]))
                for oa in rd.by_frame[x]:
                    ca_ = _centre(oa)
                    for ob in rd.by_frame[y]:
                        cb_ = _centre(ob)
                        d = math.hypot(ca_[0] - cb_[0], ca_[1] - cb_[1]) / max(width, 1.0)
                        if d <= cfg['boundary_spine_widths'] and oa['observation_id'] in rd.group_of and ob['observation_id'] in rd.group_of:
                            add(rd.group_of[oa['observation_id']], rd.group_of[ob['observation_id']], 'piece_boundary',
                                {'pieces': [ca, cb], 'frames': [x, y], 'gap_s': round(gap, 3), 'centre_distance_spine_widths': round(d, 3)},
                                oids=[oa['observation_id'], ob['observation_id']])

    # (c) the same readable title text (proposes only)
    from pipeline.instances import group_texts, text_overlap
    texts = group_texts(list(rd.groups.values()), rd.readings, cfg['text_min_chars'])
    ids = sorted(texts)
    partners: dict[str, list[tuple[float, str]]] = collections.defaultdict(list)
    for i, ga in enumerate(ids):
        for gb in ids[i + 1:]:
            ov = text_overlap(texts[ga]['text'], texts[gb]['text'])
            if ov is not None and ov >= cfg['text_agree']:
                partners[ga].append((ov, gb))
                partners[gb].append((ov, ga))
    seen = set()
    for ga in ids:
        for ov, gb in sorted(partners[ga], reverse=True)[:cfg['text_partners']]:
            key = tuple(sorted((ga, gb)))
            if key in seen:
                continue
            seen.add(key)
            add(ga, gb, 'same_text', {'texts': [texts[key[0]]['text'], texts[key[1]]['text']], 'text_overlap': round(ov, 3),
                                      'text_observation_ids': [texts[key[0]]['observation_id'], texts[key[1]]['observation_id']]},
                oids=[texts[key[0]]['observation_id'], texts[key[1]]['observation_id']])
    out = sorted(pairs.values(), key=lambda p: p['groups'])
    for p in out:
        p['pair_key'] = hashlib.sha256(('|'.join(p['groups']) + '|' + '|'.join(p['observation_ids'])).encode()).hexdigest()[:16]
    return {'pairs': out, 'skipped': dict(skipped), 'counts': dict(collections.Counter(s for p in out for s in p['sources'])),
            'readings_source': rd.readings_source, 'config': cfg}


# --------------------------------------------------------------------------------------------------------------- blinded sheets
def render_sheet(rd: RunData, oids: list[str], label: str) -> np.ndarray:
    """A blinded two-crop sheet with pipeline/pair_sheets.py's layout (no source, group or hypothesis on it)."""
    from pipeline.pair_sheets import _context, _fit, _text
    from pipeline.read_spines import native_crop
    sheet = np.full((SHEET[1], SHEET[0], 3), 246, np.uint8)
    _text(sheet, label + '  |  Same physical book?', (20, 27), .7)
    sheet[40:402, 539:541] = 210
    for side, oid in enumerate(oids):
        o = rd.obs[oid]
        img = _frame(str(rd.root), rd.frames[o['frame_id']]['image_path'])
        crop, _ = native_crop(img, o)
        if crop.shape[1] > crop.shape[0]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        context, _ = _context(img, o, rd.by_frame[o['frame_id']])
        left = side * 540
        _text(sheet, f"{'A' if side == 0 else 'B'}  {o['frame_id']}  {float(rd.frames[o['frame_id']].get('timestamp_s') or 0):.3f} s", (left + 16, 62))
        _fit(sheet, crop, (left + 12, 78, 130, 300))
        _fit(sheet, context, (left + 150, 78, 375, 300))
        _text(sheet, 'Spine', (left + 45, 398), .45)
        _text(sheet, 'Frame context (outlined spine)', (left + 160, 398), .45)
    return sheet


@lru_cache(maxsize=24)
def _frame(root: str, rel: str) -> np.ndarray:
    from pipeline.pair_sheets import safe_file
    img = cv2.imread(str(safe_file(Path(root), rel)))
    if img is None:
        raise ValueError(f'unreadable frame {rel}')
    return img


# --------------------------------------------------------------------------------------------------------------- hosted adjudication
async def adjudicate(rd: RunData, pairs: list[dict[str, Any]], out_dir: str | Path, *, cfg: dict[str, Any] | None = None, model: str | None = None,
                     calls_left: int | None = None) -> dict[str, Any]:
    """Ask the hosted model about every pair (blinded sheets, batched). Writes the sheets under ``out_dir/sheets``. Returns answers keyed by pair_key."""
    from providers import gemini as G
    cfg = {**CONFIG, **(cfg or {})}
    out = Path(out_dir)
    (out / 'sheets').mkdir(parents=True, exist_ok=True)
    cache = G.get_cache(out.parent / '_pair_oracle_cache')
    usage = G.UsageLog() if hasattr(G, 'UsageLog') else None
    batches = [pairs[i:i + cfg['batch']] for i in range(0, len(pairs), cfg['batch'])]
    cap = cfg['max_calls'] if calls_left is None else min(cfg['max_calls'], calls_left)
    answers: dict[str, dict[str, Any]] = {}
    stats = collections.Counter()
    tokens = collections.Counter()

    from providers.cache import make_key, sha256_bytes
    req_model = model or G.default_model()
    req_config = {'temperature': 0.0}

    def images_of(batch: list[dict[str, Any]]) -> list[Any]:
        images = []
        for k, p in enumerate(batch, 1):
            sheet = render_sheet(rd, p['observation_ids'], f'Sheet {k}')
            path = out / 'sheets' / f"{p['pair_key']}.png"
            if not path.exists():
                cv2.imwrite(str(path), render_sheet(rd, p['observation_ids'], 'Pair'))
            ok, buf = cv2.imencode('.png', sheet)
            images.append(G.ImageInput(data=buf.tobytes(), mime_type='image/png'))
        return images

    def cached(batch: list[dict[str, Any]], images: list[Any]) -> bool:
        """True when this exact request is in the cache (same key as providers.gemini.structured_image_request), so asking it costs no call."""
        loaded = [im.load() for im in images]
        key = make_key('gemini.structured_image', model=req_model, prompt=PROMPT.format(n=len(batch)), config=req_config,
                       schema=G._schema_fingerprint(RESPONSE_SCHEMA), inputs=[{'sha256': sha256_bytes(d), 'mime': m} for d, m in loaded])
        return cache.get(key) is not None

    async def one(bi: int, batch: list[dict[str, Any]], images: list[Any]):
        try:
            res = await G.structured_image_request(images=images, prompt=PROMPT.format(n=len(batch)), response_schema=RESPONSE_SCHEMA, model=req_model,
                                                   config=dict(req_config), stage='pair_oracle', tag=f'{rd.root.name}:{bi}', cache=cache, usage=usage)
        except Exception as exc:  # noqa: BLE001 - recorded per pair, never a merge
            stats['failed_calls'] += 1
            code = getattr(getattr(exc, '__cause__', None), 'code', None) or getattr(exc, 'code', None)
            for p in batch:
                answers[p['pair_key']] = {'answer': None, 'error': f'{type(exc).__name__}: {str(exc)[:160]}' + (f' (code {code})' if code else '')}
            return
        stats['cache_hits' if res.cache_hit else 'calls'] += 1
        for k, v in (res.usage or {}).items():
            if isinstance(v, int):
                tokens[k] += v
        got = {int(a.get('sheet', -1)): a for a in ((res.parsed or {}).get('answers') or []) if isinstance(a, dict)}
        for k, p in enumerate(batch, 1):
            a = got.get(k)
            ans = a.get('answer') if a else None
            answers[p['pair_key']] = {'answer': ans if ans in ANSWERS else None, 'reason': (a or {}).get('reason'), 'request_key': res.request_key,
                                      'model': res.model, 'cache_hit': res.cache_hit, 'sheet_index': k, 'batch': bi}

    # cache hits are free: every batch whose request is cached is served, and at most ``cap`` NEW requests are made
    sem = asyncio.Semaphore(max(1, int(os.environ.get('GEMINI_MAX_CONCURRENCY', '4'))))
    started = 0

    async def guarded(bi, batch, images):
        async with sem:
            await one(bi, batch, images)
    tasks = []
    for bi, batch in enumerate(batches):
        images = images_of(batch)
        if not cached(batch, images):
            if started >= cap:
                for p in batch:
                    answers[p['pair_key']] = {'answer': None, 'error': 'call cap reached; not asked'}
                stats['not_asked_batches'] += 1
                continue
            started += 1
        tasks.append(guarded(bi, batch, images))
    await asyncio.gather(*tasks)
    return {'answers': answers, 'calls': stats['calls'], 'cache_hits': stats['cache_hits'], 'failed_calls': stats['failed_calls'],
            'not_asked_batches': stats['not_asked_batches'], 'tokens': dict(tokens), 'model': req_model}


# --------------------------------------------------------------------------------------------------------------- merge links
def links(art: dict[str, Any], rd: RunData | None = None, cfg: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The oracle's merge links: answer 'same' AND relative spine height ratio within the declared bounds AND not the same frame. The
    instance layer then applies the group-level same-frame exclusion and the reading-contradiction veto at the union."""
    cfg = {**CONFIG, **(art.get('config') or {}), **(cfg or {})}
    lo, hi = cfg['height_ratio']
    out, refused = [], []
    for p in art.get('pairs') or []:
        a = p.get('oracle') or {}
        if a.get('answer') != 'same' or p.get('validation_only'):
            continue
        oids = p['observation_ids']
        frames = [o.split('-o')[0] for o in oids]
        ratio = p.get('height_ratio')
        if ratio is None and rd is not None:
            ha, hb = rd.rel_height(oids[0]), rd.rel_height(oids[1])
            ratio = None if not ha or not hb else ha / hb
        ev = {'kind': KIND, 'label': LABEL, 'observation_ids': oids, 'frame_ids': frames, 'sources': p.get('sources'), 'answer': 'same',
              'reason': a.get('reason'), 'model': a.get('model'), 'request_key': a.get('request_key'), 'pair_key': p.get('pair_key'),
              'height_ratio': None if ratio is None else round(ratio, 3)}
        if frames[0] == frames[1]:
            refused.append({**ev, 'groups': p['groups'], 'refused': 'same frame'})
        elif ratio is None or not (lo <= ratio <= hi):
            refused.append({**ev, 'groups': p['groups'], 'refused': f'height ratio {ratio if ratio is None else round(ratio, 3)} outside {lo}-{hi}'})
        else:
            out.append({**ev, 'groups': list(p['groups'])})
    return out, refused


def load(path: str | Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def run(run_dir: str | Path, out_dir: str | Path, *, readings_path: str | Path | None = None, instances_path: str | Path | None = None,
        panorama_dir: str | Path | None = None, max_calls: int | None = None, dry_run: bool = False, model: str | None = None,
        preferred: dict[tuple[str, str], list[str]] | None = None, validation: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """``validation``: extra pairs ({groups, observation_ids, source}) asked only to compare with another adjudicator; they never become links."""
    t0 = time.time()
    rd = RunData(run_dir, readings_path=readings_path, instances_path=instances_path, panorama_dir=panorama_dir)
    cfg = dict(CONFIG)
    if max_calls is not None:
        cfg['max_calls'] = max_calls
    prop = propose(rd, cfg, preferred=preferred)
    have = {tuple(p['groups']) for p in prop['pairs']}
    for v in validation or []:
        key = tuple(sorted(v['groups']))
        if key in have or any(o not in rd.obs for o in v['observation_ids']):
            continue
        have.add(key)
        ids = list(v['observation_ids'])
        prop['pairs'].append({'groups': list(key), 'observation_ids': ids, 'sources': [v.get('source') or 'validation'], 'provenance': [],
                              'validation_only': True,
                              'pair_key': hashlib.sha256(('|'.join(key) + '|' + '|'.join(ids)).encode()).hexdigest()[:16]})
    for p in prop['pairs']:
        ha, hb = rd.rel_height(p['observation_ids'][0]), rd.rel_height(p['observation_ids'][1])
        p['height_ratio'] = None if not ha or not hb else round(ha / hb, 4)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    adj = {'answers': {}, 'calls': 0, 'cache_hits': 0, 'failed_calls': 0, 'not_asked_batches': 0, 'tokens': {}, 'model': model}
    if not dry_run:
        adj = asyncio.run(adjudicate(rd, prop['pairs'], out, cfg=cfg, model=model))
    for p in prop['pairs']:
        p['oracle'] = adj['answers'].get(p['pair_key'])
    art = {'schema_version': SCHEMA, 'artifact_type': 'pair_oracle', 'evidence_kind': KIND, 'label': LABEL, 'run_id': rd.root.name,
           'computed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'config': cfg, 'readings_source': prop['readings_source'],
           'model': adj.get('model'), 'prompt': PROMPT, 'response_schema': RESPONSE_SCHEMA,
           'proposals': prop['counts'], 'skipped': prop['skipped'], 'pairs': prop['pairs'],
           'answers': dict(collections.Counter(str((p.get('oracle') or {}).get('answer')) for p in prop['pairs'])),
           'hosted': {k: adj.get(k) for k in ('calls', 'cache_hits', 'failed_calls', 'not_asked_batches', 'tokens')}, 'seconds': round(time.time() - t0, 1),
           'note': 'MODEL-READ evidence. A merge needs answer same + relative spine height ratio within bounds + not the same frame; the instance layer adds '
                   'the same-frame exclusion and the reading-contradiction veto. Two models agreeing is not truth; a person checking the sheets is.'}
    ok, refused = links(art, rd)
    art['links_accepted'] = len(ok)
    art['links_refused'] = refused
    (out / ARTIFACT).write_text(json.dumps(art, indent=1))
    return art


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path, help='output dir for pair_oracle.json and the sheets (never the live run dir unless intended)')
    ap.add_argument('--readings', type=Path)
    ap.add_argument('--instances', type=Path, help='instances.json of the run (pairs already in one instance are not asked)')
    ap.add_argument('--panorama-dir', type=Path)
    ap.add_argument('--max-calls', type=int)
    ap.add_argument('--model')
    ap.add_argument('--dry-run', action='store_true', help='propose only, no hosted call')
    a = ap.parse_args(argv)
    art = run(a.run, a.out, readings_path=a.readings, instances_path=a.instances, panorama_dir=a.panorama_dir, max_calls=a.max_calls,
              dry_run=a.dry_run, model=a.model)
    print(json.dumps({k: art[k] for k in ('run_id', 'proposals', 'skipped', 'answers', 'hosted', 'links_accepted', 'seconds')} | {'pairs': len(art['pairs'])}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
