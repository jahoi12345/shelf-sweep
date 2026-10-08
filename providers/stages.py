"""Independent detection and OCR selection; adapters never cross stage boundaries.

Local models are lazy process-wide singletons. Startup and frame intake don't load
weights; the first stage call runs initialization in its worker thread.
"""
from __future__ import annotations
import asyncio
import os
import threading
from pathlib import Path
from providers import gemini

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'                                    # written by scripts/setup_local_models.sh
LEGACY_DETECTOR_ROOT = ROOT / 'runs/phase2_reader_comparison_v1'  # older layout, used only when .runtime has no detector
DETECTOR_ROOT = RUNTIME if (RUNTIME / 'venv/bin/python').is_file() or not (LEGACY_DETECTOR_ROOT / 'venv/bin/python').is_file() else LEGACY_DETECTOR_ROOT
DEFAULT_WEIGHTS = DETECTOR_ROOT / 'models'
DEFAULT_PYTHON = DETECTOR_ROOT / 'venv/bin/python'
_lock = threading.Lock()
_detectors = {}
_readers = {}

class LazyEasyOCR:
    def __init__(self, runtime, model_dir, package_dir):
        self.python, self.model_dir, self.package_dir = runtime, model_dir, package_dir
    def _get(self):
        key = ('easyocr', str(self.python), str(self.model_dir))
        with _lock:
            if key not in _readers:
                from providers.runtime import ReaderProcess
                _readers[key] = ReaderProcess(self.python, self.model_dir, self.package_dir, RUNTIME/'work/easyocr')
            return _readers[key]
    async def __call__(self, **kwargs):
        adapter = await asyncio.to_thread(self._get)
        return await adapter(**kwargs)


class LazyLocalOCR:
    def __init__(self, runtime_dir):
        self.runtime_dir = Path(runtime_dir)

    def _get(self):
        key = str(self.runtime_dir)
        with _lock:
            if key not in _readers:
                from providers.local_ocr import LocalOCR
                _readers[key] = LocalOCR(self.runtime_dir)
            return _readers[key]

    async def __call__(self, **kwargs):
        adapter = await asyncio.to_thread(self._get)
        return await adapter(**kwargs)


class LazyLocalDetector:
    def __init__(self, weights, runtime):
        self.weights, self.runtime = str(weights), str(runtime)

    def _get(self):
        key = (self.weights, self.runtime)
        with _lock:
            if key not in _detectors:
                # The detector stage owns this persistent runtime bridge and model implementation.
                from providers.local_detector_process import LocalDetectorProcess
                loop = asyncio.new_event_loop()
                thread = threading.Thread(target=loop.run_forever, name='library-local-detector', daemon=True)
                thread.start()
                adapter = LocalDetectorProcess(python=self.runtime, weights_dir=self.weights, work_dir=RUNTIME/'work/detector')
                _detectors[key] = (adapter, loop)
            return _detectors[key]

    async def __call__(self, **kwargs):
        adapter, loop = await asyncio.to_thread(self._get)
        # Stage functions create their own event loops. The model process always
        # stays on one dedicated loop so locks/pipes survive across frames/sessions.
        future = asyncio.run_coroutine_threadsafe(adapter(**kwargs), loop)
        return await asyncio.wrap_future(future)


async def close_runtimes():
    for adapter, loop in list(_detectors.values()):
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(adapter.aclose(), loop))
        loop.call_soon_threadsafe(loop.stop)
    _detectors.clear()
    for adapter in list(_readers.values()):
        await asyncio.to_thread(adapter.close)
    _readers.clear()


def hosted_allowed(config=None):
    """True only with LIBRARY_ALLOW_HOSTED=1 (or config allow_hosted). Without it `auto` stays local even when a key is configured."""
    v = (config or {}).get('allow_hosted')
    if v is None:
        v = os.getenv('LIBRARY_ALLOW_HOSTED', '0')
    return str(v).lower() in ('1', 'true', 'yes')


def stage_configs(config, *, run_dir, cache, usage):
    common = {k: v for k, v in config.items() if k not in ('image_request', 'detector_request', 'reader_request')}
    # Backward-compatible provider=off/no_provider disables both unless explicit selections override.
    off = config.get('provider') == 'off' or config.get('no_provider')
    detector = config.get('detector') or os.getenv('LIBRARY_DETECTOR', 'none' if off else 'auto')
    reader = config.get('reader') or os.getenv('LIBRARY_READER', 'none' if off else 'auto')
    weights = Path(config.get('detector_weights') or os.getenv('LIBRARY_DETECTOR_WEIGHTS', str(DEFAULT_WEIGHTS))).resolve()
    runtime = Path(config.get('detector_python') or os.getenv('LIBRARY_DETECTOR_PYTHON', str(DEFAULT_PYTHON))).absolute()
    hosted = gemini.credential_available() or gemini.is_mocked()
    # Billing guard: `auto` NEVER selects a hosted (billable) adapter unless LIBRARY_ALLOW_HOSTED=1 (config allow_hosted) or the stage is set to gemini explicitly.
    auto_hosted = hosted and hosted_allowed(config)
    if detector == 'auto':
        detector = 'gemini' if auto_hosted else ('local' if runtime.is_file() and (weights/'book-spine-best.pt').is_file() else 'none')
    if reader == 'auto':
        # auto: (hosted only if allowed) > easyocr weights > none. RapidOCR is preferred earlier by providers.readers.apply; Apple Vision only when LIBRARY_READER=apple.
        reader = 'gemini' if auto_hosted else ('easyocr' if (RUNTIME/'easyocr_models/latin_g2.pth').is_file() else 'none')
    if detector not in ('none', 'gemini', 'local'):
        raise ValueError('detector must be auto, none, gemini or local')
    if reader not in ('none', 'gemini', 'apple', 'easyocr'):
        raise ValueError('reader must be auto, none, gemini, easyocr or apple')
    configs, availability = {}, {}
    for stage, selection in [('detect', detector), ('read', reader)]:
        cfg = {**common, 'no_provider': True, 'provider': 'none'}
        custom = config.get('detector_request' if stage == 'detect' else 'reader_request')
        available = False
        reason = 'stage disabled; observations/text remain unknown'
        if custom:
            cfg.update(image_request=custom, no_provider=False)
            available, reason = True, 'injected stage adapter'
        elif selection == 'gemini' and hosted:
            gen = dict(config.get('generation') or {})
            if stage == 'read':
                # production title-reading recipe: literal text, temperature 0, minimal thinking budget (thinking was 45% of tokens).
                # LIBRARY_READER_THINKING_BUDGET: 0 = minimal/off where the model allows it; unset = SDK default. Not yet exercised against the live API.
                gen.setdefault('temperature', float(os.getenv('LIBRARY_READER_TEMPERATURE', '0')))
                tb = os.getenv('LIBRARY_READER_THINKING_BUDGET', '0')
                if tb != '' and 'thinking_config' not in gen:
                    try:
                        gen['thinking_config'] = {'thinking_budget': int(tb)}
                    except ValueError:
                        pass
            pre = None
            if stage == 'read':
                from providers.readers import hosted_preprocess    # same in-memory upscale+pad recipe as the open readers (files untouched)
                pre = hosted_preprocess
            cfg.update(image_request=gemini.make_adapter(run_dir=run_dir, cache=cache, usage=usage,
                       model=config.get('model'), config=gen, image_preprocess=pre), no_provider=False, provider='gemini', hosted_generation=gen,
                       hosted_preprocess='in-memory upscale(short side<60px -> ~108) + constant median pad 1.0x/0.5x, PNG' if pre else None)
            available, reason = True, 'shared hosted adapter; authentication/real calls reported separately'
        elif stage == 'detect' and selection == 'local':
            available = runtime.is_file() and all((weights/p).is_file() for p in ['book-spine-best.pt','yolov8m-worldv2.pt','clip/ViT-B-32.pt'])
            reason = 'lazy persistent local YOLO runtime' if available else 'local runtime/weights missing'
            if available:
                cfg.update(image_request=LazyLocalDetector(weights, runtime), no_provider=False, model='local_pretrained')
        elif stage == 'read' and selection == 'easyocr':
            model_dir = Path(config.get('ocr_model_dir') or os.getenv('LIBRARY_OCR_MODEL_DIR', str(RUNTIME/'easyocr_models')))
            package_dir = Path(config.get('ocr_package_dir') or os.getenv('LIBRARY_OCR_PACKAGE_DIR', str(RUNTIME/'easyocr_packages')))
            cfg.update(image_request=LazyEasyOCR(runtime, model_dir, package_dir), no_provider=False, ocr_evidence_dir=str(Path(run_dir)/'ocr_responses'))
            available, reason = True, 'persistent local Spanish/English EasyOCR; unresolved literal text only'
        elif stage == 'read' and selection == 'apple':
            runtime_dir = config.get('ocr_runtime_dir') or os.getenv('LIBRARY_OCR_RUNTIME_DIR', str(RUNTIME/'work/ocr'))
            cfg.update(image_request=LazyLocalOCR(runtime_dir), no_provider=False, ocr_evidence_dir=str(Path(run_dir)/'ocr_responses'))
            available, reason = True, 'Apple Vision local literal OCR; no hosted credential'
        elif selection == 'gemini':
            reason = 'Gemini credential unavailable'
        configs[stage] = cfg
        if selection == 'gemini':
            from providers import hosted_gate
            if hosted_gate.disabled():
                available, reason = False, hosted_gate.HostedDisabledError.status + ' (Disconnect Gemini switch is on)'
                cfg.update(image_request=None, no_provider=True, provider='none')
        availability[stage] = {'selection': selection, 'available': available, 'reason': reason, 'hosted': selection == 'gemini', 'hosted_allowed': hosted_allowed(config),
                              'runtime': str(runtime) if stage == 'detect' and selection == 'local' else None,
                              'weights_dir': str(weights) if stage == 'detect' and selection == 'local' else None}
    configs['resolve'] = {**common, 'no_provider': True}
    configs['read'].setdefault('rectify_polygons', True)
    configs['dedup'] = {**common, 'no_provider': True}
    return configs, availability
