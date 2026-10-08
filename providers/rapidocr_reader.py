"""providers/rapidocr_reader.py (adopted unchanged from the reader comparison). Cross-platform open-source spine-crop reader:
RapidOCR (ONNX Runtime, PP-OCRv5 text detector + Latin PP-OCRv5 recogniser, CPU) with the same 4 explicit rotations and the same
literal-text/`[?]`/rotation-selection contract as providers.local_ocr.LocalOCR and providers.easyocr_reader.EasyOCRAdapter:
no dictionary/catalogue completion, accents preserved, raw per-rotation evidence saved, engine score is NOT calibrated correctness.
Install (any OS):  pip install rapidocr onnxruntime opencv-python-headless numpy   (first run downloads two ONNX models, ~12.7 MB, from RapidOCR's
ModelScope repo; set `model_dir` to pre-seed an offline copy). Same constructor/async-call/close() interface as EasyOCRAdapter."""
from __future__ import annotations
import asyncio, hashlib, json, threading, time, uuid
from pathlib import Path
from providers.local_ocr import digest, reading_payload, rotation_to_crop


def _merge_rows(boxes, texts, scores):
    """Group detected word/line boxes into text rows by vertical overlap; rows top-to-bottom, words left-to-right."""
    items = []
    for b, t, sc in zip(boxes, texts, scores):
        if not t or not str(t).strip(): continue
        xs = [float(p[0]) for p in b]; ys = [float(p[1]) for p in b]
        items.append({'t': str(t).strip(), 's': float(sc), 'cy': (min(ys)+max(ys))/2, 'h': max(ys)-min(ys), 'x': min(xs)})
    items.sort(key=lambda i: i['cy']); rows = []
    for it in items:
        if rows and abs(it['cy']-rows[-1]['cy']) < 0.5*max(it['h'], rows[-1]['h']):
            rows[-1]['items'].append(it)
            rows[-1]['cy'] = sum(i['cy'] for i in rows[-1]['items'])/len(rows[-1]['items']); rows[-1]['h'] = max(rows[-1]['h'], it['h'])
        else: rows.append({'cy': it['cy'], 'h': it['h'], 'items': [it]})
    out = []
    for r in rows:
        its = sorted(r['items'], key=lambda i: i['x']); w = [max(1, len(i['t'])) for i in its]
        out.append((' '.join(i['t'] for i in its), sum(i['s']*wi for i, wi in zip(its, w))/sum(w), [[int(p[0]), int(p[1])] for p in its[0]['box']] if 'box' in its[0] else None))
    return out


class RapidOCRAdapter:
    def __init__(self, evidence_dir, *, rotations=(0, 90, 180, 270)):
        import cv2, rapidocr, onnxruntime
        from rapidocr import RapidOCR, LangRec, OCRVersion, ModelType
        self.evidence_dir = Path(evidence_dir).resolve(); self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.rotations = tuple(rotations); self._lock = threading.Lock()
        if not self.rotations or any(x not in (0, 90, 180, 270) for x in self.rotations): raise ValueError('Invalid OCR rotations')
        t = time.perf_counter()
        self.engine = RapidOCR(params={'Rec.lang_type': LangRec.LATIN, 'Rec.ocr_version': OCRVersion.PPOCRV5, 'Rec.model_type': ModelType.MOBILE,
                                       'Det.ocr_version': OCRVersion.PPOCRV5, 'Det.model_type': ModelType.MOBILE})
        self.setup_s = time.perf_counter()-t
        self.runtime = {'engine': 'RapidOCR', 'onnxruntime': onnxruntime.__version__, 'device': 'cpu', 'languages': 'latin (es/en/ca accents; PP-OCRv5 latin dictionary)',
                        'text_detection': 'PP-OCRv5 mobile DB detector', 'adapter_sha256': digest(__file__), 'license': 'Apache-2.0 (RapidOCR + PaddleOCR models)',
                        'decoder': 'CTC greedy; no title dictionary or catalogue words'}
        self._cv2 = cv2

    async def __call__(self, *, image_paths=None, image_path=None, task='read', config=None, **kwargs):
        if task != 'read': raise ValueError('RapidOCRAdapter is never a book detector')
        paths = image_paths if image_paths is not None else ([image_path] if image_path else [])
        if len(paths) != 1: raise ValueError('Requires one native crop')
        return await asyncio.to_thread(self._infer, Path(paths[0]).resolve(), config or {})

    def _infer(self, path, config):
        cv2 = self._cv2
        with self._lock:
            start = time.perf_counter(); image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None: raise ValueError('Cannot open crop')
            h, w = image.shape[:2]; variants = []
            for rotation in self.rotations:
                oriented = image if rotation == 0 else cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
                oh, ow = oriented.shape[:2]; t = time.perf_counter()
                res = self.engine(oriented, use_det=True, use_cls=False, use_rec=True)
                rows = _merge_rows(res.boxes, res.txts, res.scores) if (res is not None and res.txts is not None and res.boxes is not None) else []
                variants.append({'rotation_deg_clockwise': rotation, 'lines': [{'bbox_oriented_pixels': None, 'candidates': [{'text': tx, 'confidence': float(cf)}]} for tx, cf, _ in rows],
                                 'latency_s': time.perf_counter()-t, 'oriented_width_px': ow, 'oriented_height_px': oh, 'oriented_pixels_to_native_crop_3x3': rotation_to_crop(rotation, w, h)})
            payload, rotation, uncertain = reading_payload(variants)
            key = hashlib.sha256(json.dumps({'crop': digest(path), 'engine': 'rapidocr_latin_v5', 'rotations': self.rotations, 'turn': config.get('source_turn_id'),
                                             'revision': config.get('session_revision')}, sort_keys=True).encode()).hexdigest()
            target = self.evidence_dir/(key+'.'+uuid.uuid4().hex+'.json')
            record = {'runtime': self.runtime, 'input_path': str(path), 'input_sha256': digest(path), 'variants': variants, 'selected_rotation_deg_clockwise': rotation,
                      'uncertain_line_indices': uncertain, 'data': payload, 'confidence_note': 'OCR confidence is not calibrated identity or correctness',
                      'source_turn_id': config.get('source_turn_id'), 'session_revision': config.get('session_revision')}
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
            return {'data': payload, 'request_ref': str(target), 'ocr_evidence': {'path': str(target), 'sha256': digest(target), 'selected_rotation_deg_clockwise': rotation,
                    'oriented_pixels_to_native_crop_3x3': rotation_to_crop(rotation, w, h)},
                    'metrics': {'provider': 'local_rapidocr', 'model': 'ppocrv5_det_mobile+latin_ppocrv5_rec_mobile', 'request_count': 0, 'local_inference_count': len(variants),
                                'latency_s': time.perf_counter()-start, 'actual_cost': 0, 'cost_provenance': 'Local CPU OCR; no API calls', 'cache_hit': False}}

    def close(self): pass
