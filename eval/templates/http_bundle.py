"""Prediction-free HTTP annotation payload/static bundle for the app frontend.

No server startup, app edits or review approval. A human exports frozen JSON.
The frontend can render_viewer(payload) and supply its own authenticated frame URLs.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
from pipeline.read_spines import inside,load,records
from pipeline.contracts import write_json


def annotation_payload(manifest_path,run_dir,image_url_for_frame):
    manifest=load(manifest_path);labels=load(Path(__file__).with_name('annotation_v1.json'))
    labels['dataset_id']=manifest['run_id'];labels['frames']=[];images={}
    source=manifest.get('data',{}).get('input',{})
    labels['coverage']['video_path']=source.get('video_path')
    labels['source_scan']={'video_sha256':source.get('sha256'), 'video_path':source.get('video_path'),
                           'manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}
    for f in records(manifest,'frames'):
        path=inside(run_dir,f['image_path'])
        labels['frames'].append({**{k:f[k] for k in ('frame_id','timestamp_s','image_path','width_px','height_px')},
                                 'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'exhaustive':False,'reviewed':False,'objects':[],'notes':None})
        images[f['frame_id']]=image_url_for_frame(f)
    return {'annotations':labels,'images':images}


def render_viewer(payload, payload_url=None):
    urls=payload['images'].values()
    if any(str(url).startswith('file:') for url in urls):raise ValueError('HTTP viewer needs same-origin/HTTP evidence URLs, not file://')
    data=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    html=Path(__file__).with_name('review.html').read_text().replace('__ANNOTATION_PAYLOAD__',data)
    if payload_url:
        from html import escape
        html=html.replace('<title>', '<link rel="alternate" type="application/json" href="'+escape(payload_url,quote=True)+'"><title>',1)
    return html


def render_http_viewer(payload_url='annotation_payload.json'):
    """Fetch blank evidence separately; the frontend can rewrite the relative route."""
    template=Path(__file__).with_name('review.html').read_text()
    start="<script>\nconst payload=__ANNOTATION_PAYLOAD__;"
    replacement="<script>\n(async()=>{\nconst response=await fetch("+json.dumps(payload_url)+",{cache:'no-store'});\nif(!response.ok)throw Error('Annotation payload could not load');\nconst payload=await response.json();"
    if start not in template:raise ValueError('Annotation template script marker changed')
    template=template.replace(start,replacement)
    return template.replace('</script>',"})().catch(error=>{document.getElementById('status').textContent=error.message;});\n</script>")


def prepare_http_bundle(manifest_path,run_dir,out):
    root=Path(out)
    if root.exists():raise ValueError('Refusing to overwrite HTTP annotation bundle')
    payload=annotation_payload(manifest_path,run_dir,lambda f:'evidence/'+f['frame_id']+Path(f['image_path']).suffix)
    root.mkdir(parents=True);(root/'evidence').mkdir()
    verified=[]
    for frame in payload['annotations']['frames']:
        src=inside(run_dir,frame['image_path']);dst=root/payload['images'][frame['frame_id']]
        shutil.copy2(src,dst);assert hashlib.sha256(dst.read_bytes()).hexdigest()==frame['image_sha256']
        verified.append({'frame_id':frame['frame_id'],'sha256':frame['image_sha256'],'url':payload['images'][frame['frame_id']]})
    write_json(root/'annotations_pending.json',payload['annotations']);write_json(root/'annotation_payload.json',payload)
    (root/'index.html').write_text(render_viewer(payload, 'annotation_payload.json'))
    (root/'index_fetch.html').write_text(render_http_viewer())
    write_json(root/'bundle.json',{'source_run':str(run_dir),'source_manifest':str(manifest_path),
                                'evidence_frames':verified,'prediction_free':True,'labels_status':'blank_pending_human',
                                'serving':'Mount this directory on the HTTP frontend; no services started by this tool'})
    return root/'index.html'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(prepare_http_bundle(a.manifest,a.run_dir,a.out))

if __name__=='__main__':main()
