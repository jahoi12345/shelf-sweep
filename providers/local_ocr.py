"""Native macOS text-only OCR adapter; no catalogue/title completion.

App runtime: initialize LocalOCR once in a background thread, pass image_request=ocr
into read_async. Each call is thread-offloaded and serialized. App owns stale
revision/cancellation handling; cancelling a coroutine cannot stop native inference.
This Apple Vision fallback is NOT an open-source upstream reader workflow.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import shutil
import subprocess
import threading
import time
import unicodedata
import uuid


def rotation_to_crop(rotation, width, height):
    """Rotated top-left pixel coordinates -> unrotated native crop coordinates."""
    return {0:[[1,0,0],[0,1,0],[0,0,1]],
            90:[[0,1,0],[-1,0,height-1],[0,0,1]],
            180:[[-1,0,width-1],[0,-1,height-1],[0,0,1]],
            270:[[0,-1,width-1],[1,0,0],[0,0,1]]}[rotation]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reading_payload(variants, uncertain_threshold=.5):
    """Select one rotation by OCR evidence only. No semantic field extraction.

    Rotation confidence is an engine score, not calibrated title correctness.
    Other rotations and all low-confidence alternatives remain in raw evidence.
    """
    def rank(v):
        rows=v.get('lines',[])
        good=[r['candidates'][0] for r in rows if r.get('candidates')]
        return sum(min(40,sum(c.isalnum() for c in x['text']))*float(x['confidence'])**2 for x in good)
    selected=max(variants,key=rank) if variants else {'lines':[],'rotation_deg_clockwise':0}
    fragments=[];uncertain=[]
    for index,row in enumerate(selected.get('lines',[])):
        if not row.get('candidates'):continue
        candidate=row['candidates'][0];text=candidate['text'].strip()
        if not text or not any(c.isalnum() for c in text):continue
        if float(candidate['confidence'])<.3:
            fragments.append('[?]');uncertain.append(index)
        else:
            fragments.append(text)
            if float(candidate['confidence'])<uncertain_threshold:uncertain.append(index)
    text='\n'.join(fragments) or None
    issues=['local_ocr_unreviewed','semantic_title_author_edition_not_extracted']
    if text and any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in text):
        issues.append('unexpected_script_for_es_en_request_raw_text_unverified')
    if uncertain:issues.append('low_confidence_ocr_lines_see_raw_evidence')
    if not text:issues.append('unreadable_spine')
    payload={'visible_text':text,'title_text':None,'author_text':None,'publisher_text':None,
             'edition_text':None,'isbn_text':None,'title_complete':False,'read_confidence':None,
             'language_code':None,'language_evidence':None,'issues':issues}
    return payload,selected['rotation_deg_clockwise'],uncertain


class LocalOCR:
    def __init__(self,runtime_dir,*,rotations=(0,90,180,270),timeout_s=45,compile_if_missing=True):
        if platform.system()!='Darwin':raise RuntimeError('Apple Vision OCR requires macOS; no substitute output')
        self.runtime_dir=Path(runtime_dir).resolve();self.runtime_dir.mkdir(parents=True,exist_ok=True)
        self.source=Path(__file__).with_name('vision_ocr.swift')
        self.binary=self.runtime_dir/'vision_ocr';stamp=self.runtime_dir/'vision_ocr.source.sha256'
        self.rotations=tuple(rotations)
        if not self.rotations or any(x not in (0,90,180,270) for x in self.rotations):raise ValueError('OCR rotations must be 0/90/180/270')
        self.timeout_s=timeout_s;self._lock=threading.Lock();self.process=None
        start=time.perf_counter();source_hash=digest(self.source)
        if not self.binary.exists() or not stamp.exists() or stamp.read_text().strip()!=source_hash:
            if not compile_if_missing:raise RuntimeError('Native OCR binary missing/stale; compile once during setup')
            swift=shutil.which('swiftc')
            if not swift:raise RuntimeError('swiftc unavailable; install the native runtime')
            result=subprocess.run([swift,'-module-cache-path',str(self.runtime_dir/'module_cache'),str(self.source),'-o',str(self.binary)],capture_output=True,text=True,timeout=120)
            if result.returncode:raise RuntimeError('Swift OCR compile failed: '+result.stderr[-2000:])
            stamp.write_text(source_hash+'\n')
        self.process=subprocess.Popen([str(self.binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
        try:
            self.runtime=self._receive()
            if not self.runtime.get('ready'):raise RuntimeError('OCR runtime not ready: '+str(self.runtime))
        except BaseException:self.close();raise
        self.setup_s=time.perf_counter()-start
        self.runtime.update(source_sha256=source_hash,binary_sha256=digest(self.binary),
                            adapter_sha256=digest(__file__),
                            engine_license='Apple proprietary system framework; not open source',
                            model_weights='OS-bundled; standalone hash/version unavailable')
    def _receive(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout,selectors.EVENT_READ)
            if not selector.select(self.timeout_s):
                self.close();raise TimeoutError('Local OCR worker timed out and was stopped; reinitialize adapter')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Local OCR worker exited before response')
        result=json.loads(line)
        if result.get('error'):raise RuntimeError('Local OCR failure: '+result['error'])
        return result
    async def __call__(self,*,image_paths=None,image_path=None,task='read',config=None,**kwargs):
        if task!='read':raise ValueError('LocalOCR is a reading adapter, never a detector')
        paths=image_paths if image_paths is not None else ([image_path] if image_path else [])
        if len(paths)!=1:raise ValueError('LocalOCR requires one saved native crop')
        return await asyncio.to_thread(self._infer,Path(paths[0]).resolve(),config or {})
    def _infer(self,path,config):
        with self._lock:
            if self.process is None or self.process.poll() is not None:raise RuntimeError('OCR adapter closed; initialize again')
            import cv2
            image=cv2.imread(str(path))
            if image is None:raise ValueError('Cannot decode native OCR crop')
            height,width=image.shape[:2]
            start=time.perf_counter();variants=[]
            for rotation in self.rotations:
                self.process.stdin.write(json.dumps({'image_path':str(path),'rotation_deg_clockwise':rotation})+'\n');self.process.stdin.flush()
                result=self._receive()
                result['oriented_width_px']=height if rotation in (90,270) else width
                result['oriented_height_px']=width if rotation in (90,270) else height
                result['oriented_pixels_to_native_crop_3x3']=rotation_to_crop(rotation,width,height)
                variants.append(result)
            payload,rotation,uncertain=reading_payload(variants)
            root=Path(config.get('ocr_evidence_dir',self.runtime_dir.parent/'ocr_responses')).resolve();root.mkdir(parents=True,exist_ok=True)
            key=hashlib.sha256(json.dumps({'image':digest(path),'source':self.runtime['source_sha256'],'runtime':self.runtime['os_version'],'rotations':self.rotations,'turn':config.get('source_turn_id'),'revision':config.get('session_revision')},sort_keys=True).encode()).hexdigest()
            evidence={'input_path':str(path),'input_sha256':digest(path),'runtime':self.runtime,'variants':variants,
                      'selected_rotation_deg_clockwise':rotation,'uncertain_line_indices':uncertain,
                      'selection_policy':'max confidence-squared weighted alphanumeric length; no catalogue/label input',
                      'confidence_note':'Raw engine line scores are not calibrated correctness or identity confidence',
                      'data':payload,'source_turn_id':config.get('source_turn_id'),'session_revision':config.get('session_revision')}
            target=root/(key+'.'+uuid.uuid4().hex+'.json');target.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
            return {'data':payload,'request_ref':str(target),'ocr_evidence':{'path':str(target),'sha256':digest(target),'selected_rotation_deg_clockwise':rotation,
                    'oriented_pixels_to_native_crop_3x3':rotation_to_crop(rotation,width,height),
                    'bbox_convention':'Raw Vision boxes are normalized bottom-left in the oriented crop; transform uses top-left pixel coordinates'},
                    'metrics':{'provider':'local_apple_vision','model':'VNRecognizeTextRequest_revision_'+str(self.runtime['revision']),
                               'request_count':0,'local_inference_count':len(variants),'latency_s':time.perf_counter()-start,
                               'actual_cost':0,'cost_provenance':'Native local OCR; no API calls','cache_hit':False}}
    def close(self):
        p=self.process
        if p is not None:
            self.process=None
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=3)
            for stream in (p.stdin,p.stdout):
                if stream:stream.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def make_adapter(*,runtime_dir,**kwargs):
    return LocalOCR(runtime_dir,**kwargs)
