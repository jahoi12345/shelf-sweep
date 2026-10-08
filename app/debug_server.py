"""Isolated read-only stitched evidence viewer; run on port 4180."""
from pathlib import Path
import json
import re
import cv2
import numpy as np
import struct
from functools import lru_cache
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from .debug_views import DebugStore, ID
from .build_jobs import BuildJobs
from .post_sweep import PostSweepBuilds

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent/'static'
store = DebugStore(ROOT)
build_jobs = BuildJobs(ROOT)
# ONE build queue for the Debugger's Build buttons and (app/server.py enables it on this same object) the automatic builds after every sweep.
build_queue = PostSweepBuilds(ROOT, lambda: build_jobs, enabled=False)
app = FastAPI(title='Yugo stitched evidence debugger')


@app.middleware('http')
async def security_headers(request, call_next):
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self'; media-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'self'"
    return response


def checked(fn):
    try: return fn()
    except (ValueError, OSError, KeyError) as exc: raise HTTPException(404,str(exc)) from exc


def image_size(path):
    """Read bounded PNG/JPEG headers before allocating any raster buffer."""
    with path.open('rb') as stream: header=stream.read(256*1024)
    if header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR':
        width,height=struct.unpack('>II',header[16:24])
    elif header[:2]==b'\xff\xd8':
        pos=2; width=height=0
        while pos+4<=len(header):
            if header[pos]!=255: raise HTTPException(422,'Invalid JPEG header')
            while pos<len(header) and header[pos]==255: pos+=1
            marker=header[pos];pos+=1
            if marker in {0xd8,0xd9} or 0xd0<=marker<=0xd7: continue
            length=int.from_bytes(header[pos:pos+2],'big')
            if length<2: break
            if marker in {0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf}:
                height,width=struct.unpack('>HH',header[pos+3:pos+7]);break
            pos+=length
        if not width or not height: raise HTTPException(422,'JPEG size unavailable in bounded header')
    else: raise HTTPException(422,'Unsupported image header')
    if width*height>32_000_000 or not width or not height: raise HTTPException(413,'Image pixel limit')
    return width,height


@app.get('/')
def index(): return FileResponse(STATIC/'debug-index.html')


@app.get('/static/vendor/{name}')
def vendor(name: str):
    if name not in {'three.module.min.js','THREE-LICENSE.txt'}: raise HTTPException(404)
    return FileResponse(STATIC/'vendor'/name)


@app.get('/static/{name}')
def static(name: str):
    if name not in {'debug-app.js','debug-style.css','recon-view.js'}: raise HTTPException(404)
    return FileResponse(STATIC/name)


@app.get('/api/scans')
def scans(all: int = 0): return checked(lambda: store.catalogue(show_all=bool(all)))   # all=1: non-featured runs too


@app.get('/api/panorama/jobs')
def build_job_list(): return {'jobs': build_jobs.jobs()}


@app.post('/api/panorama/build')
def build_start(payload: dict = Body(...)):
    """Queue pipeline/panorama.py (kind=panorama) or pipeline/recon3d.py (kind=recon3d) for one saved run into a NEW versioned dir (JSON body only).
    Never refused because another build is running: the request waits its turn ("queued #n"). 409 only for a run that can never be built, with the reason."""
    try: status = build_queue.request(payload.get('run_id'), [payload.get('kind', 'panorama')])
    except ValueError as exc: raise HTTPException(409, str(exc)) from exc
    return {'queued': True, 'status': status, 'queue': build_queue.queue_overview()}


@app.get('/api/builds/{run_id}')
def build_status(run_id: str):
    """Queue status of the builds of one run (queued #n / building / done / failed + reason) and what the queue is doing overall."""
    if not ID.fullmatch(run_id): raise HTTPException(404, 'Invalid run ID')
    return {'status': build_queue.status(run_id), 'queue': build_queue.queue_overview()}


@app.on_event('startup')
def resume_builds(): build_queue.resume()   # standalone debugger; app.server resumes the same queue itself when it mounts this app


RECON_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}_recon3d_v\d+$')


@app.get('/api/recon3d/{run_id}')
def recon_list(run_id: str):
    """Versioned 3D reconstructions built from this run (newest first); headers only, no point arrays."""
    def go():
        if not ID.fullmatch(run_id): raise ValueError('Invalid run ID')
        runs = ROOT/'runs'; found = []
        for p in sorted(runs.glob(f'{run_id}_recon3d_v*'), key=lambda p: ((p/'recon3d.json').is_file(), p.stat().st_mtime), reverse=True):   # finished ones first: the newest FINISHED build is the default
            if not RECON_ID.fullmatch(p.name) or not p.is_dir(): continue
            entry = {'recon_id': p.name, 'status': 'building'}
            try:
                data = _recon_json(str(p/'recon3d.json'), (p/'recon3d.json').stat().st_mtime_ns)
                entry.update({k: data.get(k) for k in ('status', 'reason', 'registered_frames', 'frames_total', 'frames_used', 'registered_ratio', 'registered_of_total', 'points_kept',
                                                       'mean_reprojection_error_px', 'group_markers', 'groups_total', 'unscaled')})
                from pipeline.recon3d import map_consistency
                entry['map_consistency'] = map_consistency(data)
            except (OSError, ValueError): pass
            found.append(entry)
        found.sort(key=lambda e: e.get('map_consistency') == 'inconsistent')   # a map that is not one flat shelf face is never the default (stable sort)
        return {'run_id': run_id, 'recon': found}
    return checked(go)


@lru_cache(maxsize=4)
def _recon_json(path, mtime):
    return json.loads(Path(path).read_text())


@app.get('/api/recon3d/{run_id}/{recon_id}')
def recon_file(run_id: str, recon_id: str):
    def go():
        if not ID.fullmatch(run_id) or not RECON_ID.fullmatch(recon_id) or not recon_id.startswith(run_id+'_recon3d_v'): raise ValueError('Invalid reconstruction ID')
        path = (ROOT/'runs'/recon_id/'recon3d.json').resolve()
        if not path.is_relative_to((ROOT/'runs').resolve()) or not path.is_file(): raise ValueError('Reconstruction unavailable')
        return path
    return FileResponse(checked(go), media_type='application/json')


@app.get('/api/frame/{run_id}/{frame_id}')
def run_frame(run_id: str, frame_id: str):
    """Original frame JPEG of a saved run (only frames listed in its manifest)."""
    def go():
        run = store.run(run_id)
        manifest = json.loads((run/'manifest.json').read_text())
        frame = next((f for f in manifest['data']['frames'] if f['frame_id'] == frame_id), None)
        if frame is None: raise ValueError('Unknown frame')
        path = (run/frame['image_path']).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}: raise ValueError('Frame unavailable')
        return path
    return FileResponse(checked(go))


@app.get('/api/view/{scan}/{run_id}')
def view(scan: str, run_id: str): return checked(lambda:store.view(scan,run_id))


@app.get('/api/assets/{run_id}/{token}')
def asset(run_id: str, token: str, download: bool = False):
    records=checked(lambda:store.asset_records(run_id)[6])
    p=records.get(token)
    if not p: raise HTTPException(404,'Not a listed evidence asset')
    if p.suffix.lower() in {'.jpg','.jpeg','.png'}:
        image_size(p)
    return FileResponse(p,filename=p.name if download else None)


@app.get('/api/crop/{scan}/{run_id}/{observation_id}')
def crop(scan: str, run_id: str, observation_id: str):
    if not ID.fullmatch(observation_id): raise HTTPException(404)
    v=checked(lambda:store.view(scan,run_id))
    o=next((o for o in v['observations'] if o['observation_id']==observation_id),None)
    f=next((f for f in v['frames'] if o and f['frame_id']==o['frame_id']),None)
    if not f or not f['asset']: raise HTTPException(404,'Native frame unavailable')
    p=checked(lambda:store.asset_records(run_id)[6][f['asset']])
    width,height=image_size(p);b=o.get('bbox_xyxy')
    if not b or len(b)!=4 or not(0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height): raise HTTPException(422,'Invalid native crop')
    im=cv2.imread(str(p));x0,y0,x1,y1=map(int,b)
    if im is None: raise HTTPException(422)
    out=cv2.imencode('.png',im[y0:y1,x0:x1])[1].tobytes()
    return Response(out,media_type='image/png',headers={'X-Crop-Provenance':'Native frame axis-aligned bounds; no OCR rectification'})


@app.get('/api/preview/{scan}')
def preview(scan: str):
    from .debug_registry import read_registry
    s=next((s for s in read_registry(ROOT) if s['label']==scan),None)
    if not s: raise HTTPException(404)
    p=ROOT/'sweeps'/s['browser_preview']
    if not p.is_file() or p.stat().st_size>200*1024*1024: raise HTTPException(404)
    return FileResponse(p,media_type='video/mp4')


def component_asset(run_id, component_id):
    if not ID.fullmatch(component_id): raise HTTPException(404)
    p,source,pan,manifest,components,frames,records,downloads=checked(lambda:store.asset_records(run_id))
    c=next((c for c in components if c['component_id']==component_id),None)
    if not c or c['width_px']*c['height_px']>10_000_000: raise HTTPException(404)
    return c,pan,records


@app.get('/api/inspect/{run_id}/{component_id}')
def inspect_pixel(run_id: str, component_id: str, x: float, y: float):
    c,pan,records=component_asset(run_id,component_id)
    if not np.isfinite([x,y]).all() or not (0<=x<c['width_px'] and 0<=y<c['height_px']): raise HTTPException(422)
    path=records.get(c.get('pixel_source_index_asset'))
    if not path: return {'status':'unavailable','reason':'No per-pixel source provenance in this panorama version'}
    if image_size(path)!=(c['width_px'],c['height_px']): raise HTTPException(422)
    im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim!=2: raise HTTPException(422)
    index=int(im[int(y),int(x)])
    if not index: return {'status':'uncovered','reason':'No source pixels here. No content inferred in this gap.'}
    frames=pan.get('source_frames',[])
    if index>len(frames): raise HTTPException(422)
    f=frames[index-1];member=next((m for m in c['source_frames'] if m['frame_id']==f['frame_id']),None)
    if not member or not member.get('panorama_to_original_3x3'): return {'status':'unavailable','reason':'Inverse transform unavailable'}
    point=np.asarray(member['panorama_to_original_3x3'])@np.asarray([x,y,1.])
    return {'status':'source_pixel','frame_id':f['frame_id'],'frame_sha256':f['sha256'],'timestamp_s':f['timestamp_s'],'native_xy':(point[:2]/point[2]).tolist(),'component_id':component_id,'panorama_xy':[x,y]}


@lru_cache(maxsize=4)
def diagnostic_mask(path, mtime, mode):
    image_size(Path(path))
    indices=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    if indices is None or indices.ndim!=2 or indices.size>10_000_000: raise ValueError('Invalid coverage raster')
    rgba=np.zeros((*indices.shape,4),dtype=np.uint8)
    if mode=='seams':
        edges=np.zeros(indices.shape,dtype=bool)
        edges[:,1:]|=(indices[:,1:]!=indices[:,:-1])&(indices[:,1:]>0)&(indices[:,:-1]>0)
        edges[1:,:]|=(indices[1:,:]!=indices[:-1,:])&(indices[1:,:]>0)&(indices[:-1,:]>0)
        edges=cv2.dilate(edges.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        rgba[edges]=[0,220,255,220]
    else: rgba[indices==0]=[180,70,230,180]
    return cv2.imencode('.png',rgba)[1].tobytes()


@app.get('/api/mask/{run_id}/{component_id}/{mode}')
def mask(run_id: str, component_id: str, mode: str):
    if mode not in {'seams','gaps'}: raise HTTPException(404)
    c,pan,records=component_asset(run_id,component_id)
    path=records.get(c.get('pixel_source_index_asset')) if mode=='seams' else records.get(c.get('coverage_mask_asset'))
    if not path: raise HTTPException(404,'Coverage/seam provenance unavailable')
    data=checked(lambda:diagnostic_mask(str(path),path.stat().st_mtime_ns,mode))
    return Response(data,media_type='image/png')
