"""providers/florence_reader.py (adopted unchanged from the reader comparison). Florence-2 <OCR> small-crop reader behind the SAME async adapter contract as
LocalOCR / EasyOCRAdapter / RapidOCRAdapter (call with image_paths=[crop], task='read' -> {'data', 'request_ref', 'ocr_evidence', 'metrics'}); text only, no title/author claim.
Runs the model in a persistent CHILD PROCESS with its own interpreter (the app's .venv is 3.12 without torch), like providers.local_detector_process.
Measured on a 41-crop YT1 dev set (model-read labels; relative only): base 0.380 token recall / 13 of 41 books >=50% at ~1.8 s/crop; large 0.455 / 18 at ~14 s/crop (loaded machine).
Weights: florence-community/Florence-2-base (MIT, 463 MB) or -large (MIT, 1.55 GB); pre-seed the model dir; no network at runtime. Florence NEVER returns empty text: treat output as a fragment, not a title."""
from __future__ import annotations
import asyncio, hashlib, json, subprocess, threading, time, uuid, selectors
from pathlib import Path
from providers.local_ocr import digest, reading_payload, rotation_to_crop


class FlorenceAdapter:
    def __init__(self, evidence_dir, *, python, model_dir, worker_script=None, rotations=(90, 270), preprocess=None, threads=4, timeout_s=120):
        self.evidence_dir = Path(evidence_dir).resolve(); self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.rotations = tuple(rotations); self.preprocess = preprocess if preprocess is not None else {'below': 60, 'target': 108}
        self.timeout_s = timeout_s; self._lock = threading.Lock()
        script = worker_script or str(Path(__file__).with_name('florence_worker.py'))
        t = time.perf_counter()
        self.process = subprocess.Popen([str(python), script, str(model_dir), str(threads)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        first = self._receive(); self.runtime = first['runtime']; self.setup_s = time.perf_counter()-t
        self.runtime.update(adapter_sha256=digest(__file__), worker_sha256=digest(script), license='MIT (model weights florence-community/Florence-2-*)',
                            decoder='greedy; no dictionary or catalogue completion', preprocess=self.preprocess)

    def _receive(self):
        with selectors.DefaultSelector() as sel:
            sel.register(self.process.stdout, selectors.EVENT_READ)
            if not sel.select(self.timeout_s): self.close(); raise TimeoutError('Florence worker timed out and was stopped; reinitialize adapter')
        line = self.process.stdout.readline()
        if not line: raise RuntimeError('Florence worker exited')
        out = json.loads(line)
        if out.get('error'): raise RuntimeError(out['error'])
        return out

    async def __call__(self, *, image_paths=None, image_path=None, task='read', config=None, **kwargs):
        if task != 'read': raise ValueError('FlorenceAdapter is never a book detector')
        paths = image_paths if image_paths is not None else ([image_path] if image_path else [])
        if len(paths) != 1: raise ValueError('Requires one native crop')
        return await asyncio.to_thread(self._infer, Path(paths[0]).resolve(), config or {})

    def _infer(self, path, config):
        import cv2
        with self._lock:
            start = time.perf_counter(); h, w = cv2.imread(str(path)).shape[:2]
            self.process.stdin.write(json.dumps({'image_path': str(path), 'rotations': list(self.rotations), 'preprocess': self.preprocess})+'\n'); self.process.stdin.flush()
            out = self._receive(); variants = out['variants']
            for v in variants: v['oriented_pixels_to_native_crop_3x3'] = rotation_to_crop(v['rotation_deg_clockwise'], w, h)
            payload, rotation, uncertain = reading_payload(variants)
            key = hashlib.sha256(json.dumps({'crop': digest(path), 'engine': 'florence2', 'rot': self.rotations, 'turn': config.get('source_turn_id'), 'rev': config.get('session_revision')}, sort_keys=True).encode()).hexdigest()
            target = self.evidence_dir/(key+'.'+uuid.uuid4().hex+'.json')
            record = {'runtime': self.runtime, 'input_path': str(path), 'input_sha256': digest(path), 'variants': variants, 'selected_rotation_deg_clockwise': rotation, 'uncertain_line_indices': uncertain,
                      'data': payload, 'confidence_note': 'Florence score = exp(mean token log-prob); NOT calibrated correctness; the model emits text even for unreadable crops',
                      'source_turn_id': config.get('source_turn_id'), 'session_revision': config.get('session_revision'), 'saved_crop_modified': False}
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
            return {'data': payload, 'request_ref': str(target), 'ocr_evidence': {'path': str(target), 'sha256': digest(target), 'selected_rotation_deg_clockwise': rotation,
                    'oriented_pixels_to_native_crop_3x3': rotation_to_crop(rotation, w, h)},
                    'metrics': {'provider': 'local_florence2', 'model': 'Florence-2 <OCR> greedy letterbox', 'request_count': 0, 'local_inference_count': len(variants), 'latency_s': time.perf_counter()-start,
                                'actual_cost': 0, 'cost_provenance': 'Local CPU model; no API calls', 'cache_hit': False}}

    def close(self):
        p, self.process = getattr(self, 'process', None), None
        if p is not None and p.poll() is None:
            p.terminate()
            try: p.wait(timeout=3)
            except subprocess.TimeoutExpired: p.kill()
