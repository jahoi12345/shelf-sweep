"""providers/florence_worker.py (adopted from the reader comparison). Persistent JSONL worker for Florence-2 <OCR> (run with an interpreter that has torch + transformers>=5, e.g. the isolated py3.13 runtime).
stdin: one JSON per line {"image_path": "...", "rotations": [90,270], "preprocess": {"below":60,"target":108}|null}
stdout: one JSON per line {"variants":[{"rotation_deg_clockwise":r,"lines":[{"candidates":[{"text":..,"confidence":..}]}],"latency_s":..,"preprocess":{..}|null}], "latency_s":..}
First line out = {"ready":true,"setup_s":..,"runtime":{..}}.  The saved crop is only READ; letterboxing/upscaling/padding are in memory.  No network (HF_HUB_OFFLINE)."""
import json, os, sys, time
import cv2, numpy as np, torch
from PIL import Image
model_dir = sys.argv[1]; threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.environ.setdefault('HF_HUB_OFFLINE', '1'); torch.set_num_threads(threads)
from transformers import AutoProcessor, Florence2ForConditionalGeneration
protocol = sys.stdout; sys.stdout = sys.stderr                # model/library chatter must never corrupt the JSONL protocol
t0 = time.perf_counter(); proc = AutoProcessor.from_pretrained(model_dir); model = Florence2ForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
def emit(o): protocol.write(json.dumps(o, ensure_ascii=False)+'\n'); protocol.flush()
emit({'ready': True, 'setup_s': time.perf_counter()-t0, 'runtime': {'engine': 'Florence-2', 'model_dir': model_dir, 'torch': torch.__version__, 'device': 'cpu', 'task': '<OCR>', 'decoding': 'greedy', 'input': 'aspect-preserving letterbox to square, constant median colour'}})
ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
def preprocess_strip(img, below=60, target=108):          # same recipe as providers/readers.py preprocess_strip
    h, w = img.shape[:2]; short = min(h, w)
    if short >= below or short < 4: return img, None
    k = target/short; up = cv2.resize(img, (max(1, round(w*k)), max(1, round(h*k))), interpolation=cv2.INTER_CUBIC); s = min(up.shape[:2])
    border = np.concatenate([up[0], up[-1], up[:, 0], up[:, -1]]).reshape(-1, 3); med = [int(v) for v in np.median(border, axis=0)]
    return cv2.copyMakeBorder(up, s, s, s//2, s//2, cv2.BORDER_CONSTANT, value=med), {'scale': round(k, 2), 'pad_vertical_px': int(s), 'pad_horizontal_px': int(s//2), 'border_median': med}
def letterbox(img):
    h, w = img.shape[:2]; s = max(h, w); border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).reshape(-1, 3); med = [int(v) for v in np.median(border, axis=0)]
    top = (s-h)//2; left = (s-w)//2; return cv2.copyMakeBorder(img, top, s-h-top, left, s-w-left, cv2.BORDER_CONSTANT, value=med)
def read(img):
    pil = Image.fromarray(cv2.cvtColor(letterbox(img), cv2.COLOR_BGR2RGB)); inp = proc(text='<OCR>', images=pil, return_tensors='pt')
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=64, num_beams=1, do_sample=False, output_scores=True, return_dict_in_generate=True)
    text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip(); lp = [torch.log_softmax(sc[0], -1).max().item() for sc in out.scores]
    return text, float(np.exp(np.mean(lp))) if lp else 0.5
for line in sys.stdin:
    try:
        req = json.loads(line); img = cv2.imread(req['image_path'])
        if img is None: raise ValueError('cannot read image')
        t = time.perf_counter(); variants = []
        for r in req.get('rotations', [90, 270]):
            tv = time.perf_counter(); x = img if ROT[r] is None else cv2.rotate(img, ROT[r]); pp = None
            if req.get('preprocess'): x, pp = preprocess_strip(x, **req['preprocess'])
            text, conf = read(x)
            variants.append({'rotation_deg_clockwise': r, 'lines': [{'candidates': [{'text': text, 'confidence': conf}]}] if text else [], 'latency_s': time.perf_counter()-tv, 'preprocess': pp})
        emit({'variants': variants, 'latency_s': time.perf_counter()-t})
    except Exception as exc:
        emit({'error': f'{type(exc).__name__}: {exc}'})
