"""Read-only artifact adapter. No inference, approval, or shared pipeline mutations."""
from pathlib import Path
import hashlib
import json
import re
import copy
from functools import lru_cache
from .debug_registry import read_registry

ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
MAX_JSON = 24 * 1024 * 1024
UNREGISTERED = 'Other runs (no canonical recording)'   # runs whose input hash is not a registered scan (live camera sweeps, uploads)
PANORAMA_DIR = re.compile(r'^(?P<source>.+)_(?:panorama|pano)_v(?P<n>\d+)$')


@lru_cache(maxsize=8192)
def _digest(path, mtime, size):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def digest(path):
    stat=path.stat()
    return _digest(str(path.resolve()),stat.st_mtime_ns,stat.st_size)


@lru_cache(maxsize=16)
def _json(path, mtime, size):
    return json.loads(Path(path).read_text())


def load(path):
    if not path.is_file(): return None
    stat = path.stat()
    if stat.st_size > MAX_JSON: raise ValueError('Artifact exceeds 24 MiB')
    return _json(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


class DebugStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.runs = self.root/'runs'

    def run(self, run_id):
        if not ID.fullmatch(run_id): raise ValueError('Invalid run ID')
        p = (self.runs/run_id).resolve()
        if not p.is_relative_to(self.runs.resolve()) or not p.is_dir(): raise ValueError('Unknown run')
        return p

    def source(self, p, panorama):
        raw = panorama.get('source_run', p.name)
        q = Path(raw)
        if not q.is_absolute(): q = self.root/q if str(q).startswith('runs/') else self.runs/q
        q = q.resolve()
        if not q.is_relative_to(self.runs.resolve()) or not q.is_dir(): raise ValueError('Invalid source run')
        return q

    def artifact(self, p, name):
        q = (p/name).resolve()
        if not q.is_relative_to(p.resolve()): raise ValueError('Artifact outside run')
        return load(q)

    def latest_panorama(self, run_id):
        """Newest runs/<run>_panorama_vN (or legacy _pano_vN) directory with a panorama.json, or None."""
        best = None
        for q in self.runs.glob(run_id + '_*_v*'):
            m = PANORAMA_DIR.match(q.name)
            if m and m['source'] == run_id and q.is_dir() and (q / 'panorama.json').is_file():
                if best is None or int(m['n']) > best[0]:
                    best = (int(m['n']), q)
        return best[1] if best else None

    def catalogue(self, show_all=True):
        from app import run_visibility as RV
        pins = RV.pinned(self.runs)
        scans = read_registry(self.root)
        by_hash = {s['sha256']: s for s in scans}
        for s in scans: s['runs'] = []
        other = {'label': UNREGISTERED, 'file': None, 'original_name': 'runs without a registered recording', 'sha256': None, 'aliases': [],
                 'browser_preview': None, 'runs': [], 'synthetic': True}
        newest = {}   # source run -> (N, panorama dir name)
        for p in sorted(self.runs.iterdir()) if self.runs.is_dir() else []:
            if not p.is_dir() or not ID.fullmatch(p.name) or not p.resolve().is_relative_to(self.runs.resolve()): continue
            if not show_all and not RV.featured(p.name, self.runs, pins): continue
            try:
                pan = self.artifact(p, 'panorama.json')
                source = self.source(p, pan) if pan else p
                manifest = self.artifact(source, 'manifest.json')
                if manifest is None or not manifest.get('data', {}).get('frames'): continue
                sha = (pan or {}).get('source_video_sha256') or manifest.get('data', {}).get('input', {}).get('sha256')
                scan = by_hash.get(sha) or other
                candidate = self.artifact(source, 'candidates.json')
                reqs = (candidate or {}).get('metrics', {}).get('requests', [])
                profile = reqs[0].get('profile', 'undeclared') if reqs else 'undeclared'
                m = PANORAMA_DIR.match(p.name) if pan else None
                if m and m['source'] == source.name and (source.name not in newest or int(m['n']) > newest[source.name][0]): newest[source.name] = (int(m['n']), p.name)
                scan['runs'].append({'run_id':p.name,'source_run':source.name,'profile':profile,'fixture':bool((candidate or {}).get('fixture') or (manifest or {}).get('fixture')),'panorama_status':(pan or {}).get('status','unprocessed'),'has_panorama': bool(pan),'is_panorama_output': bool(pan and source.name != p.name),'created':RV.created(p),'label':RV.labels(self.runs).get(p.name)})
            except (ValueError, OSError, KeyError): continue
        scans = scans + ([other] if other['runs'] else [])
        for s in scans:
            for r in s['runs']:   # a source run shows the newest panorama built from it
                if not r['is_panorama_output'] and r['run_id'] in newest:
                    r['panorama_run'] = newest[r['run_id']][1]; r['has_panorama'] = True; r['panorama_status'] = 'panorama ' + newest[r['run_id']][1].rsplit('_', 1)[-1]
            s['runs'].sort(key=lambda r:r['created'] or '', reverse=True)   # newest first
        return scans

    def indexes(self, source):
        """Only explicit panorama debug export locations, never a recursive filesystem API."""
        paths = [source/'debug_index.json']
        paths += sorted(self.runs.glob('worker3_panorama_debug*/*/index.json'), reverse=True)[:64]
        results=[]
        for path in paths:
            if not path.is_file() or not path.resolve().is_relative_to(self.runs.resolve()): continue
            try:
                a=load(path)
                if a.get('source_run') == str(source.relative_to(self.root)):
                    results.append((path,a))
            except (ValueError,OSError): continue
        return results

    def asset_records(self, run_id):
        p = self.run(run_id); pan = self.artifact(p,'panorama.json') or {}
        if not pan:   # a source run shows the newest panorama built from it; assets resolve inside that directory
            attached = self.latest_panorama(run_id)
            if attached is not None: p = attached; pan = self.artifact(p,'panorama.json') or {}
        source = self.source(p,pan) if pan else p
        manifest = self.artifact(source,'manifest.json') or {}
        records = {}
        def add(base, name, expected=None):
            if not isinstance(name,str): return None
            q = (base/name).resolve()
            if not q.is_relative_to(base.resolve()) or not q.is_file(): return None
            if q.suffix.lower() not in {'.jpg','.jpeg','.png','.json'}: return None
            if q.stat().st_size > (MAX_JSON if q.suffix=='.json' else 20*1024*1024): return None
            if expected and digest(q)!=expected: return None
            token = hashlib.sha256(str(q).encode()).hexdigest()[:24];records[token] = q;return token
        components=[]
        for c in pan.get('components',[])[:512]:
            out = copy.deepcopy(c)
            for key in ['image','coverage_mask','overlay_image','pixel_source_index']:
                file_ref=next((f for f in c.get('files',{}).values() if f.get('path')==c.get(key)),{})
                out[key+'_asset'] = add(p,c.get(key),file_ref.get('sha256'))
            components.append(out)
        frames=[]
        candidate_data=(self.artifact(source,'candidates.json') or {}).get('data',{})
        frame_states={f['frame_id']:f.get('status','unknown') for f in candidate_data.get('frames',[])}
        for observation in candidate_data.get('observations',[]):
            frame_states.setdefault(observation.get('frame_id'),'observations available')
        provenance_frames={f['frame_id']:f for c in components for f in c.get('source_frames',[])}
        for f in manifest.get('data',{}).get('frames',[])[:5000]:
            provenance=provenance_frames.get(f['frame_id'],{})
            sha=provenance.get('sha256') or f.get('image_sha256')
            token=add(source,f.get('image_path'),sha)
            frames.append(dict(f,sha256=sha or (digest(records[token]) if token else None),asset=token, detection_status=frame_states.get(f['frame_id'],'unprocessed')))
        downloads = {}
        inventory_base=p if (p/'inventory.json').exists() else source
        for base,name in [(p,'panorama.json'),(p,'diagnostics.json'),(source,'candidates.json'),(inventory_base,'inventory.json'),(source,'debug_index.json'),(source,'readings.json'),(source,'readings_second.json'),(source,'group_reading_plan.json')]:
            token=add(base,name)
            if token: downloads[name]=token
        for index_path,index in self.indexes(source):
            token=add(index_path.parent,index_path.name)
            if token: downloads[str(index_path.relative_to(self.runs))]=token
            for ref in index.get('evidence_allowlist',[])[:12000]:
                path=(self.root/ref.get('path','')).resolve()
                if path.is_relative_to(self.runs.resolve()) and path.is_file() and path.suffix.lower() in {'.png','.jpg','.jpeg'}:
                    if path.stat().st_size<=20*1024*1024 and digest(path)==ref.get('sha256'): add(self.runs,str(path.relative_to(self.runs)))
        for name in ('readings.json','readings_second.json'):
            a=self.artifact(source,name) or {}
            for row in a.get('data',{}).get('readings',[])[:5000]:
                expected=row.get('provenance',{}).get('crop_sha256')
                if expected:add(source,row.get('crop_path'),expected)
        return p, source, pan, manifest, components, frames, records, downloads

    def view(self, scan_label, run_id):
        registry = read_registry(self.root)
        scan = next((s for s in registry if s['label']==scan_label),None)
        if scan is None and scan_label == UNREGISTERED:
            scan = {'label': UNREGISTERED, 'file': None, 'original_name': 'runs without a registered recording', 'sha256': None, 'aliases': [], 'browser_preview': None, 'synthetic': True}
        if scan is None: raise ValueError('Unknown scan')
        p,source,pan,manifest,components,frames,records,downloads = self.asset_records(run_id)
        sha = pan.get('source_video_sha256') or manifest.get('data',{}).get('input',{}).get('sha256')
        if scan.get('synthetic'):
            if sha in {s['sha256'] for s in registry}: raise ValueError('Run belongs to a registered scan')
        elif sha != scan['sha256']: raise ValueError('Run/scan hash mismatch')
        for key,name in [('source_candidates_sha256','candidates.json'),('source_manifest_sha256','manifest.json')]:
            if pan.get(key) and pan[key]!=digest(source/name): raise ValueError(f'{name} of the source run changed after this panorama was built; press Build 2D panorama for a new version')
        stages={}
        for label,name in [('decoding/extraction','manifest.json'),('detection','candidates.json'),('association','inventory.json'),('OCR','readings.json'),('identity','resolutions.json')]:
            a=self.artifact(source,name)
            stages[label]={'status':a.get('status','unknown') if a else 'unprocessed','elapsed_s':(a.get('metrics',{}).get('elapsed_s') or a.get('metrics',{}).get('stage_status',{}).get('elapsed_s')) if a else None,'issues':a.get('issues',[])[:100] if a else ['Artifact unavailable'], 'sha256':digest(source/name) if a else None}
            if label=='OCR' and a and a.get('data',{}).get('readings') and a.get('status')=='skipped':
                stages[label]['status']='partial: saved readings, header says skipped'
                stages[label]['issues'].append('Artifact header contradicts nonempty reading rows; no success inferred')
        stages['registration/stitching']={'status':pan.get('status','unprocessed'),'elapsed_s':pan.get('metrics',{}).get('elapsed_s'),'issues':pan.get('limitations',[])[:100] or ([] if pan else ['panorama.json unavailable']), 'sha256':digest(p/'panorama.json') if pan else None}
        # Panorama association is a separately versioned stage; do not substitute
        # the source baseline's different group IDs/counts into the panorama association output.
        inv=self.artifact(p,'inventory.json') or self.artifact(source,'inventory.json') or {};data=inv.get('data',{})
        if pan.get('source_inventory_sha256') and (p/'inventory.json').exists() and pan['source_inventory_sha256']!=digest(p/'inventory.json'):
            raise ValueError('Panorama association hash mismatch')
        if inv and (p/'inventory.json').exists():
            stages['association']={'status':inv.get('status','unknown'),'elapsed_s':inv.get('metrics',{}).get('elapsed_s'),'issues':inv.get('issues',[])[:100],'sha256':digest(p/'inventory.json')}
        groups=data.get('books',[])+data.get('items',[])
        group_by_obs={o:g['instance_id'] for g in groups for o in g.get('observation_ids',[])}
        for c in components:
            for o in c.get('overlays',[]): o['proposed_group_id']=o.get('proposed_instance_id') or group_by_obs.get(o['observation_id'])
        candidate=self.artifact(source,'candidates.json') or {};obs=copy.deepcopy(candidate.get('data',{}).get('observations',[]))
        for o in obs: o['proposed_group_id']=group_by_obs.get(o['observation_id'])
        ocr_records={};index_status='unavailable: exact-run debug index pending';index_provenance=[]
        candidate_sha=digest(source/'candidates.json') if candidate else None
        observations={o['observation_id']:o for o in candidate.get('data',{}).get('observations',[])}
        frame_by_id={f['frame_id']:f for f in frames}
        for index_path,index in self.indexes(source):
            valid=(index.get('run_id')==candidate.get('run_id') and index.get('source_video_sha256')==sha and index.get('input_artifacts',{}).get('candidates',{}).get('sha256')==candidate_sha and index.get('input_artifacts',{}).get('manifest',{}).get('sha256')==digest(source/'manifest.json'))
            index_provenance.append({'path':str(index_path.relative_to(self.root)),'sha256':digest(index_path),'binding':'verified exact run and input hashes' if valid else 'REJECTED mismatch','summary':index.get('summary')})
            if not valid:
                if not ocr_records: index_status='mismatched: debug index rejected'
                continue
            for row in index.get('observations',[]):
                oid=row.get('observation_id');original=observations.get(oid);key=row.get('join_key',{});frame=frame_by_id.get(key.get('frame_id'))
                frame_path=records.get((frame or {}).get('asset'))
                row_valid=bool(original and row.get('detection')==original and key.get('run_id')==candidate.get('run_id') and key.get('candidate_sha256')==candidate_sha and frame_path and digest(frame_path)==key.get('frame_sha256'))
                if not row_valid:
                    ocr_records[oid]={'reading_join':'REJECTED mismatch','ocr_state':'unavailable','review_reasons':['Debug index row/frame/native geometry mismatch']};continue
                out=copy.deepcopy(row)
                crop_ref=row.get('crop') or {}
                crop_path=(self.root/crop_ref.get('path','')).resolve()
                token=hashlib.sha256(str(crop_path).encode()).hexdigest()[:24]
                if row.get('reading_join','').startswith('verified') and (token not in records or crop_ref.get('sha256')!=digest(records[token])):
                    out={'reading_join':'REJECTED changed crop','ocr_state':'unavailable','review_reasons':['Crop missing or changed since index export']}
                elif token in records: out['crop_asset']=token
                ocr_records[oid]=out
            for label,key in [('OCR','readings'),('identity','resolutions')]:
                ref=index.get('input_artifacts',{}).get(key,{})
                artifact_path=(self.root/ref.get('path','')).resolve()
                if artifact_path.is_relative_to(self.runs.resolve()) and artifact_path.is_file() and digest(artifact_path)==ref.get('sha256'):
                    artifact=load(artifact_path)
                    stages[label]={'status':artifact.get('status','unknown'),'elapsed_s':artifact.get('metrics',{}).get('elapsed_s'),'issues':artifact.get('issues',[])[:100], 'sha256':ref['sha256'],'artifact':ref['path']}
            index_status='available: exact run/frame/candidate/crop hashes verified';break
        stages['OCR/index']={'status':'available' if ocr_records else 'unavailable','elapsed_s':None,'issues':[index_status],'provenance':index_provenance}
        if not ocr_records and candidate:
            from .debug_run_adapter import adapt_readings
            def add_crop(base,name,expected):
                if not isinstance(name,str):return None
                path=(base/name).resolve();token=hashlib.sha256(str(path).encode()).hexdigest()[:24]
                return token if path.is_relative_to(source.resolve()) and token in records and digest(path)==expected else None
            ocr_records,bindings=adapt_readings(source,candidate,manifest,add_crop)
            if bindings:
                index_status='run adapter: verified incremental/full candidate, frame, crop and raw OCR hashes; rejected rows stay explicit'
                index_provenance=bindings
                stages['OCR/index']={'status':'partial' if any(r['reading_join']=='rejected_mismatch' for r in ocr_records.values()) else 'available','elapsed_s':None,'issues':[index_status],'provenance':bindings}
        packet=self.artifact(source,'claim_packet.json') or {}
        summary=self.artifact(source,'replay_summary.json') or {}
        packet_data=packet.get('data',{})
        capture_quality=packet_data.get('count',{}).get('capture_quality') or summary.get('capture_quality') or {}
        reading_plan=self.artifact(source,'group_reading_plan.json') or {}
        logs=[]
        logdir=source/'logs'
        if logdir.is_dir():
            for f in sorted(logdir.glob('*.jsonl'))[:8]:
                if not f.resolve().is_relative_to(source.resolve()):continue
                with f.open('rb') as stream:
                    stream.seek(max(0,f.stat().st_size-16384));lines=stream.read(16384).decode('utf-8','replace').splitlines()[-40:]
                logs.append({'file':f.name,'tail':lines,'truncated':True})
        return {'association':inv.get('metrics',{}).get('association') or 'undeclared','association_metrics':inv.get('metrics',{}),'capture_quality':capture_quality,'group_reading_plan':reading_plan.get('data',{}),'count_detail':packet_data.get('count',{}),'read_policy':summary.get('provenance',{}).get('read_policy',{}),'scan':scan,'run_id':run_id,'source_run':source.name,'panorama_run':p.name if pan else None,'panorama_summary':{k:pan.get(k) for k in ('book_count','markers_total','item_markers_total','unplaced_groups','unregistered_frames','counts','metrics','method')} if pan else None,'components':components,'frames':frames,'observations':obs,'groups':groups,'registration_edges':pan.get('registration_edges',[]),'association_evidence':{k:data.get(k,[]) for k in ['associations','uncertain_associations','uncertain_tile_associations']},'stages':stages,'downloads':downloads,'logs':logs,'method':pan.get('method'),'limits':pan.get('limitations',[]),'ocr_index_status':index_status,'ocr_records':ocr_records,'index_provenance':index_provenance,'fixture':bool(candidate.get('fixture') or manifest.get('fixture')),'counts':{'raw_observations':len(obs),'raw_books':sum(o.get('kind')=='book' for o in obs),'proposed_book_groups':len(data['books']) if 'books'in data else None,'proposed_item_groups':len(data['items']) if 'items'in data else None,'verified':False},'capture_gaps':manifest.get('data',{}).get('capture_gaps',[])}
