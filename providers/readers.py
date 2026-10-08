"""Pluggable readers + integration hooks for the engine. Core stays untouched; the default reader is unchanged.

Reader plugin = a factory registered under a name, selected with ``LIBRARY_READER=<name>`` (or ``config["reader"]``)::

    # my_reader.py
    from providers.readers import register_reader
    def factory(*, run_dir, config, cache, usage):
        async def adapter(*, image_paths, prompt, response_schema, task, config=None):    # reader adapter protocol
            return {"data": {"visible_text": "...", "title_text": "...", "author_text": None, "publisher_text": None, "edition_text": None,
                             "isbn_text": None, "title_complete": False, "read_confidence": 0.5, "issues": []},
                    "metrics": {"requests": [{"provider": "my_reader", "local_inference_count": 1, "actual_cost": 0}]}}
        return adapter
    register_reader("my_reader", factory, description="what it is")

Load plugin modules with ``LIBRARY_READER_PLUGINS=my_reader,other_mod`` (importable module names, comma separated) or
``config["reader_plugins"]``. ``sanitize_reading`` still enforces literalness on whatever a plugin returns.
Also here: ``LIBRARY_DETECTOR=external:/path/candidates.json`` and ``LIBRARY_ASSOC=cross_view`` (see ``apply``).
"""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from pipeline import contracts as C

_READERS: dict[str, dict[str, Any]] = {}
BUILTIN = ("none", "auto", "gemini", "apple", "easyocr")
NAME = re.compile(r"[a-z][a-z0-9_\-]{1,40}")
TS_TOL = 0.06


class PluginError(ValueError):
    pass


def register_reader(name: str, factory: Callable[..., Any], *, description: str = "") -> None:
    if not NAME.fullmatch(name or "") or name in BUILTIN:
        raise PluginError(f"reader name must match {NAME.pattern} and not be a built-in ({', '.join(BUILTIN)})")
    if not callable(factory):
        raise PluginError("factory must be callable: factory(*, run_dir, config, cache, usage) -> adapter")
    _READERS[name] = {"factory": factory, "description": description}


def unregister_reader(name: str) -> None:
    _READERS.pop(name, None)


def reader_names() -> list[str]:
    return sorted(_READERS)


def load_plugins(config: dict[str, Any]) -> list[str]:
    mods = config.get("reader_plugins") or os.environ.get("LIBRARY_READER_PLUGINS", "")
    names = [m.strip() for m in (mods.split(",") if isinstance(mods, str) else mods) if m and m.strip()]
    for m in names:
        importlib.import_module(m)
    return names


# ---------------------------------------------------------------------------------------------- built-in optional reader: RapidOCR (ONNX, CPU, any OS)
_rapid: dict[str, Any] = {}


def rapidocr_available() -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("rapidocr", "onnxruntime"))


def _clean(r: dict[str, Any]) -> str:
    t = ((r.get("data") or {}).get("visible_text") or "").strip()
    return "" if t in ("", "[?]") else t


def preprocess_strip(img, *, below: int = 60, target: int = 108):
    """In-memory only. For a crop whose short side < ``below`` px: bicubic upscale to short side ~``target`` and CONSTANT-colour padding (median of the
    border pixels) of 1.0x the upscaled short side vertically and 0.5x horizontally. Bench on YT1 best-view crops (22-60 px): >=8-char yield 1/40
    unpadded -> 11/40; replicate padding does not help. Returns (image, info|None); the native crop file is never modified."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    short = min(h, w)
    if short >= below or short < 4:
        return img, None
    k = target / short
    up = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_CUBIC)
    uh, uw = up.shape[:2]
    border = np.concatenate([up[0], up[-1], up[:, 0], up[:, -1]]).reshape(-1, up.shape[2] if up.ndim == 3 else 1)
    med = [int(v) for v in np.median(border, axis=0)]
    s = min(uh, uw)
    out = cv2.copyMakeBorder(up, int(s), int(s), int(s // 2), int(s // 2), cv2.BORDER_CONSTANT, value=med if up.ndim == 3 else med[0])
    return out, {"scale": round(k, 2), "pad_vertical_px": int(s), "pad_horizontal_px": int(s // 2), "border_median": med}


class _PreprocEngine:
    """Wraps the RapidOCR engine call so small crops are upscaled/padded in memory before text detection + recognition."""

    def __init__(self, engine, stats, below, target):
        self.engine, self.stats, self.below, self.target = engine, stats, below, target

    def __call__(self, img, **kw):
        img2, info = preprocess_strip(img, below=self.below, target=self.target)
        self.stats["preprocessed" if info else "native"] += 1
        return self.engine(img2, **kw)


class _RapidPool:
    """N RapidOCR adapters behind one callable so group-level reading runs in parallel. Each instance gets cores//N ONNX intra-op threads
    (no CPU oversubscription). Rotations are two-stage: spine crops are long-axis-vertical, so (90, 270) is tried first and (0, 180) only when
    both are empty (about half the cost). Instances load lazily."""

    def __init__(self, evidence_dir: Path, size: int):
        import queue
        self.dir, self.size, self.made = Path(evidence_dir), max(1, size), 0
        self.threads = max(1, (os.cpu_count() or 4) // self.size)
        self.free: "queue.Queue[Any]" = queue.Queue()
        self.lock = __import__("threading").Lock()
        self.stats = {"calls": 0, "second_stage": 0, "preprocessed": 0, "native": 0}
        self.below = int(os.environ.get("LIBRARY_READ_PREPROC_BELOW", 60))      # 0 disables
        self.target = int(os.environ.get("LIBRARY_READ_PREPROC_TARGET", 108))

    def _make(self):
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        from providers.rapidocr_reader import RapidOCRAdapter
        a = RapidOCRAdapter(self.dir, rotations=(90, 270))
        a.engine = RapidOCR(params={"Rec.lang_type": LangRec.LATIN, "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
                                    "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
                                    "EngineConfig.onnxruntime.intra_op_num_threads": self.threads, "EngineConfig.onnxruntime.inter_op_num_threads": 1})
        a.runtime["intra_op_threads"] = self.threads
        if self.below > 0:
            a.engine = _PreprocEngine(a.engine, self.stats, self.below, self.target)
            a.runtime["preprocessing"] = (f"in-memory bicubic upscale to short side ~{self.target}px + constant median-border padding (1.0x short side vertical, 0.5x horizontal) "
                                          f"for crops with short side < {self.below}px; native crop and saved evidence crop unchanged")
        else:
            a.runtime["preprocessing"] = "none (native crop)"
        return a

    def _acquire(self):
        import queue
        with self.lock:
            try:
                return self.free.get_nowait()
            except queue.Empty:
                if self.made < self.size:
                    self.made += 1
                    return self._make()
        return self.free.get()

    async def __call__(self, **kwargs):
        inst = await __import__("asyncio").to_thread(self._acquire)
        try:
            inst.rotations = (90, 270)
            out = await inst(**kwargs)
            self.stats["calls"] += 1
            if not _clean(out):
                inst.rotations = (0, 180)
                first = out
                second = await inst(**kwargs)
                self.stats["second_stage"] += 1
                m1, m2 = first.get("metrics", {}), second.get("metrics", {})
                out = second if _clean(second) else first
                for k in ("local_inference_count", "latency_s"):
                    if k in m1 and k in m2:
                        out.setdefault("metrics", {})[k] = m1[k] + m2[k]
                # every rotation attempt stays linked from the returned reading (evidence provenance), whichever stage was selected
                refs = [x for x in (first.get("request_ref"), second.get("request_ref")) if x]
                out.setdefault("ocr_evidence", {})["attempts"] = [{"rotations": [90, 270], "evidence": (first.get("ocr_evidence") or {}).get("path") or first.get("request_ref")},
                                                                 {"rotations": [0, 180], "evidence": (second.get("ocr_evidence") or {}).get("path") or second.get("request_ref")}]
                out["request_refs"] = refs
            return out
        finally:
            self.free.put(inst)


def _rapidocr_factory(*, run_dir, config, cache, usage):
    """One pool per run (models load lazily, once per instance); evidence is written under the run's ocr_responses/."""
    key = str(Path(run_dir).resolve())
    if key not in _rapid:
        policy = (config or {}).get("read_policy") or os.environ.get("LIBRARY_READ_POLICY", "group")
        n = int((config or {}).get("read_workers") or os.environ.get("LIBRARY_READ_WORKERS") or 3)
        _rapid.clear()
        _rapid[key] = _RapidPool(Path(run_dir) / "ocr_responses", n if policy == "group" else 1)
    return _rapid[key]


def florence_paths() -> tuple[Path, Path | None]:
    """(python interpreter with torch + transformers>=5, Florence-2 model snapshot dir). Env: LIBRARY_FLORENCE_PYTHON, LIBRARY_FLORENCE_MODEL_DIR; defaults are
    .runtime/florence/venv and the Hugging Face cache under .runtime/florence/models (not installed by the setup script). Optional: nothing here is needed unless the florence reader is selected."""
    root = Path(__file__).resolve().parents[1]
    py = Path(os.environ.get("LIBRARY_FLORENCE_PYTHON") or root / ".runtime/florence/venv/bin/python")
    md = os.environ.get("LIBRARY_FLORENCE_MODEL_DIR")
    size = os.environ.get("LIBRARY_FLORENCE_MODEL", "large").lower()      # large (the best reader on YugoTest2-like crops) | base (3x faster)
    if md:
        model = Path(md)
    else:
        import glob
        g = sorted(glob.glob(str(root / f".runtime/florence/models/hf/hub/models--florence-community--Florence-2-{'base' if size == 'base' else 'large'}/snapshots/*")))
        model = Path(g[0]) if g else None
    return py, model


def florence_unavailable_reason() -> str | None:
    py, model = florence_paths()
    if not py.is_file():
        return f"unavailable: Florence runtime missing (python not found: {py}); set LIBRARY_FLORENCE_PYTHON"
    if model is None or not (model / "config.json").is_file():
        return "unavailable: Florence-2 weights missing; set LIBRARY_FLORENCE_MODEL_DIR to a florence-community/Florence-2-base snapshot"
    return None


class _FlorencePool:
    """N persistent Florence-2 child processes (own interpreter with torch; the app venv has none), each with cores//N torch threads."""

    def __init__(self, evidence_dir: Path, size: int):
        import queue
        self.dir, self.size, self.made = Path(evidence_dir), max(1, size), 0
        self.threads = max(1, (os.cpu_count() or 4) // self.size)
        self.free: "queue.Queue[Any]" = queue.Queue()
        self.lock = __import__("threading").Lock()
        self.adapters: list[Any] = []

    def _acquire(self):
        import queue
        with self.lock:
            try:
                return self.free.get_nowait()
            except queue.Empty:
                if self.made < self.size:
                    from providers.florence_reader import FlorenceAdapter
                    py, model = florence_paths()
                    self.made += 1
                    a = FlorenceAdapter(self.dir, python=py, model_dir=model, threads=self.threads)
                    self.adapters.append(a)
                    return a
        return self.free.get()

    async def __call__(self, **kwargs):
        inst = await __import__("asyncio").to_thread(self._acquire)
        try:
            out = await inst(**kwargs)
            out.setdefault("data", {})["issues"] = list(dict.fromkeys([*(out["data"].get("issues") or []), "literal_reading_unverified", "florence_may_hallucinate_plausible_text"]))
            return out
        finally:
            self.free.put(inst)

    def close(self):
        for a in self.adapters:
            a.close()


_florence: dict[str, Any] = {}


def _florence_factory(*, run_dir, config, cache, usage):
    reason = florence_unavailable_reason()
    if reason:
        raise PluginError(reason)
    key = str(Path(run_dir).resolve())
    if key not in _florence:
        for old in _florence.values():
            old.close()
        _florence.clear()
        n = int((config or {}).get("florence_workers") or os.environ.get("LIBRARY_FLORENCE_WORKERS") or 2)
        _florence[key] = _FlorencePool(Path(run_dir) / "ocr_responses_florence", n)
    return _florence[key]


register_reader("florence", _florence_factory, description="Florence-2 <OCR> (large by default, LIBRARY_FLORENCE_MODEL=base for the 3x faster model; MIT, CPU, optional isolated torch runtime; output is a literal, unverified fragment)")
register_reader("rapidocr", _rapidocr_factory, description="RapidOCR (ONNX Runtime, PP-OCRv5 detector + Latin recogniser, CPU, 4 rotations; any OS)")


def effective_reader(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """What LIBRARY_READER=auto will use here, without loading anything: hosted > rapidocr (if installed) > easyocr (if weights) > none.
    Apple Vision is used only when explicitly requested (LIBRARY_READER=apple); it is never auto-selected."""
    config = config or {}
    sel = config.get("reader") or os.environ.get("LIBRARY_READER") or "auto"
    if sel != "auto":
        return {"requested": sel, "effective": sel, "reason": "explicit selection"}
    from providers import gemini
    from providers.stages import hosted_allowed
    root = Path(__file__).resolve().parents[1]
    note = ""
    if gemini.credential_available() or gemini.is_mocked():
        if hosted_allowed(config):
            return {"requested": "auto", "effective": "gemini", "reason": "hosted credential configured and LIBRARY_ALLOW_HOSTED=1"}
        note = "; a hosted credential exists but LIBRARY_ALLOW_HOSTED is not set, so auto stays local"
    if rapidocr_available():
        return {"requested": "auto", "effective": "rapidocr", "reason": "open cross-platform reader installed" + note}
    if (root / ".runtime/easyocr_models/latin_g2.pth").is_file():
        return {"requested": "auto", "effective": "easyocr", "reason": "local EasyOCR weights present (RapidOCR not installed)" + note}
    return {"requested": "auto", "effective": "none", "reason": "no reader available: install the rapidocr extra (see docs/configuration.md)" + note}


def hosted_preprocess(path):
    """Hosted read path: the same in-memory recipe the open readers get (upscale short side < 60 px to ~108, constant median-border pad 1.0x/0.5x),
    encoded as PNG bytes for the request. The 0.90 recall was measured on preprocessed crops. The native crop file is never modified;
    crops that need no preprocessing are sent as the original file."""
    import cv2
    from providers.gemini import ImageInput
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return p
    out, info = preprocess_strip(img)
    if info is None:
        return p
    ok, buf = cv2.imencode(".png", out)
    return ImageInput(data=bytes(buf), mime_type="image/png") if ok else p


# ---------------------------------------------------------------------------------------------- hosted hard cap + free fallback
class HostedCapped:
    """Wraps a hosted (billable) read adapter: at most LIBRARY_HOSTED_MAX_CALLS fresh calls per run; after that every call goes to the free fallback
    adapter (RapidOCR) and the reading carries issue 'hosted_cap_reached'. Counts, tokens and cost are reported in ``report()`` for provenance."""

    def __init__(self, hosted, fallback, max_calls: int, usage=None, name: str = "gemini", fallback_name: str = "rapidocr"):
        self.hosted, self.fallback, self.max_calls, self.usage = hosted, fallback, max(0, int(max_calls)), usage
        self.name, self.fallback_name = name, fallback_name
        self.calls = self.fallback_calls = self.errors = 0
        self.lock = __import__("threading").Lock()

    def cap_reached(self) -> bool:
        return self.calls >= self.max_calls

    async def __call__(self, **kwargs):
        with self.lock:
            use_hosted = self.calls < self.max_calls
            if use_hosted:
                self.calls += 1
        if use_hosted:
            try:
                out = await self.hosted(**kwargs)
                (out.setdefault("metrics", {})).update(hosted_calls_so_far=self.calls, hosted_cap=self.max_calls)
                return out
            except Exception as exc:  # noqa: BLE001 - a hosted failure also falls back; it is recorded
                self.errors += 1
                err = f"{type(exc).__name__}: {str(exc)[:120]}"
                if self.fallback is None:
                    raise
                out = await self.fallback(**kwargs)
                out.setdefault("data", {}).setdefault("issues", []).append(f"hosted_error_fallback_{self.fallback_name}: {err}")
                return out
        if self.fallback is None:
            raise PluginError(f"hosted call cap reached ({self.max_calls}) and no fallback reader is configured")
        with self.lock:
            self.fallback_calls += 1
        out = await self.fallback(**kwargs)
        out.setdefault("data", {}).setdefault("issues", []).append("hosted_cap_reached: read by the free fallback reader")
        out.setdefault("metrics", {}).update(hosted_cap_reached=True, hosted_cap=self.max_calls)
        return out

    def report(self) -> dict[str, Any]:
        u = self.usage.summary() if self.usage is not None and hasattr(self.usage, "summary") else {}
        return {"hosted": self.name, "fallback": self.fallback_name, "max_calls": self.max_calls, "hosted_calls": self.calls, "fallback_calls": self.fallback_calls,
                "hosted_errors": self.errors, "cap_reached": self.cap_reached(), "status": "cap reached" if self.cap_reached() else ("under cap" if self.calls else "no hosted calls yet"),
                "tokens": {k: u.get(k) for k in ("prompt_tokens", "output_tokens", "thought_tokens", "total_tokens")}, "cost_usd": u.get("cost_usd"), "cost_note": u.get("cost_note")}


def hosted_max_calls(config: dict[str, Any] | None = None) -> int:
    v = (config or {}).get("hosted_max_calls", os.environ.get("LIBRARY_HOSTED_MAX_CALLS", 800))
    try:
        return int(v)
    except (TypeError, ValueError):
        return 800


# ---------------------------------------------------------------------------------------------- external candidates (replay a detector's output)
class ExternalDetect:
    """``StageFns.detect``-compatible callable that serves candidates from a finished candidates.json instead of running a detector.

    Frames map by timestamp (+-0.06 s) using the sibling manifest.json; observation ids are re-keyed to the engine's frame id
    (``<frame>-oNNN``) and each carries ``external_source`` (file, sha256, original id). Frames with no external frame are reported as an issue,
    never guessed.
    """

    def __init__(self, path: str | Path):
        p = Path(path).resolve()
        if not p.is_file():
            raise PluginError(f"external candidates not found: {p}")
        man = p.with_name("manifest.json")
        if not man.is_file():
            raise PluginError(f"external candidates need a sibling manifest.json (frame timestamps): {man}")
        self.path, self.sha = p, __import__("hashlib").sha256(p.read_bytes()).hexdigest()
        doc = json.loads(p.read_text())
        self.frames = {f["frame_id"]: f for f in json.loads(man.read_text())["data"]["frames"]}
        self.by_frame: dict[str, list[dict[str, Any]]] = {}
        for o in (doc.get("data") or {}).get("observations", []):
            self.by_frame.setdefault(o["frame_id"], []).append(o)
        self.run_id = doc.get("run_id")

    def frame_for(self, t: float) -> str | None:
        best = min(self.frames.values(), key=lambda f: abs(f["timestamp_s"] - t), default=None)
        return best["frame_id"] if best and abs(best["timestamp_s"] - t) <= TS_TOL else None

    def __call__(self, manifest_path, run_dir, config=None):
        config = config or {}
        man = C.read_json(manifest_path)
        frames = man["data"]["frames"]
        out_obs, issues = [], []
        for f in frames:
            src = self.frame_for(f["timestamp_s"])
            if src is None:
                issues.append({"frame_id": f["frame_id"], "reason": "no external candidate frame within 0.06 s of this timestamp; nothing was guessed"})
                continue
            for i, o in enumerate(sorted(self.by_frame.get(src, []), key=lambda x: x["observation_id"])):
                no = {**o, "observation_id": f"{f['frame_id']}-o{i:03d}", "frame_id": f["frame_id"], "frame_ref": f.get("image_path") or f"frames/{f['frame_id']}.jpg",
                      "external_source": {"file": str(self.path), "sha256": self.sha, "run_id": self.run_id, "observation_id": o["observation_id"], "frame_id": src}}
                out_obs.append(no)
        env = C.envelope(Path(run_dir).name, "detect", {"observations": out_obs}, status="ok" if not issues else "partial", issues=issues,
                         metrics={"external_candidates": str(self.path), "external_sha256": self.sha, "requests": []})
        return C.write_json(Path(run_dir) / config.get("output_name", "candidates.json"), env)


def apply(config: dict[str, Any], stages: Any, *, run_dir, cache, usage, stage_configs_fn):
    """Return (stage_configs, availability, stages, notes). Resolves plugin readers / external detector / cross_view association, otherwise
    delegates to ``stage_configs_fn`` unchanged. Never raises for a missing optional hook: it reports the fallback in ``notes``."""
    notes: dict[str, Any] = {}
    cfg = dict(config)
    load_plugins(cfg)
    reader = cfg.get("reader") or os.environ.get("LIBRARY_READER")
    detector = cfg.get("detector") or os.environ.get("LIBRARY_DETECTOR")
    off = cfg.get("provider") == "off" or cfg.get("no_provider")
    if (reader in (None, "auto")) and not off:           # auto prefers an installed open reader over the local-weights fallback (hosted still wins)
        eff = effective_reader(cfg)
        if eff["effective"] == "rapidocr":
            reader = "rapidocr"
            notes["reader_auto"] = eff
    plugin = reader if reader in _READERS else None
    external = detector[len("external:"):] if isinstance(detector, str) and detector.startswith("external:") else None
    if plugin:
        cfg["reader"] = "none"
    if external:
        cfg["detector"] = "none"
    configs, avail = stage_configs_fn(cfg, run_dir=run_dir, cache=cache, usage=usage)
    if plugin:
        try:
            adapter = _READERS[plugin]["factory"](run_dir=run_dir, config=config, cache=cache, usage=usage)
            cfg_read = configs["read"]
            cfg_read.update(image_request=adapter, no_provider=False, ocr_evidence_dir=str(Path(run_dir) / "ocr_responses"))
            avail["read"] = {"selection": plugin, "available": True, "reason": f"plugin reader: {_READERS[plugin]['description'] or plugin}", "runtime": None, "weights_dir": None}
            notes["reader_plugin"] = plugin
        except Exception as exc:  # noqa: BLE001 - an optional runtime that is missing must degrade to 'unavailable', never crash the session
            avail["read"] = {"selection": plugin, "available": False, "reason": str(exc)[:240], "runtime": None, "weights_dir": None}
            notes["reader_plugin_error"] = str(exc)[:240]
    # hosted PRIMARY reader (explicit LIBRARY_READER=gemini): hard call cap + free fallback (RapidOCR when installed)
    if avail.get("read", {}).get("selection") == "gemini" and avail["read"].get("available") and configs["read"].get("image_request") is not None:
        fallback = None
        if rapidocr_available():
            try:
                fallback = _READERS["rapidocr"]["factory"](run_dir=Path(run_dir) / "fallback", config=config, cache=cache, usage=usage)
            except Exception as exc:  # noqa: BLE001
                notes["hosted_fallback_error"] = str(exc)[:200]
        capped = HostedCapped(configs["read"]["image_request"], fallback, hosted_max_calls(cfg), usage)
        configs["read"]["image_request"] = capped
        configs["read"]["ocr_evidence_dir"] = str(Path(run_dir) / "ocr_responses")
        avail["read"]["reason"] += f"; hard cap {capped.max_calls} hosted calls per run, then {'rapidocr fallback' if fallback else 'NO fallback (reads stop)'}"
        avail["read"]["hosted_cap"] = capped.max_calls
        notes["hosted_cap"] = capped.report
    second = cfg.get("reader_second") or os.environ.get("LIBRARY_READER_SECOND")
    if second:
        if second not in _READERS:
            notes["reader_second_error"] = f"unknown second reader {second!r}; registered: {', '.join(reader_names())}"
        else:
            try:
                adapter2 = _READERS[second]["factory"](run_dir=Path(run_dir) / "second", config=config, cache=cache, usage=usage)
                configs["read2"] = {**configs["read"], "image_request": adapter2, "no_provider": False, "ocr_evidence_dir": str(Path(run_dir) / f"ocr_responses_{second}")}
                avail["read2"] = {"selection": second, "available": True, "reason": f"second-pass reader: {_READERS[second]['description'] or second}", "runtime": None, "weights_dir": None}
                notes["reader_second"] = second
            except Exception as exc:  # noqa: BLE001
                avail["read2"] = {"selection": second, "available": False, "reason": str(exc)[:240], "runtime": None, "weights_dir": None}
                notes["reader_second_error"] = str(exc)[:240]
    if external:
        try:
            ext = ExternalDetect(external)
            stages = type(stages)(detect=ext, read=stages.read, resolve=stages.resolve, merge=stages.merge)
            configs["detect"].update(no_provider=False)
            avail["detect"] = {"selection": "external", "available": True, "reason": f"external candidates {ext.path.name} sha256 {ext.sha[:12]}", "runtime": None, "weights_dir": None}
            notes["external_candidates"] = {"path": str(ext.path), "sha256": ext.sha}
        except (PluginError, OSError, KeyError, ValueError) as exc:
            avail["detect"] = {"selection": "external", "available": False, "reason": f"external candidates unusable: {exc}", "runtime": None, "weights_dir": None}
            notes["external_candidates_error"] = str(exc)
    assoc = cfg.get("assoc") or os.environ.get("LIBRARY_ASSOC") or "cross_view"     # default association; LIBRARY_ASSOC=dedup keeps the legacy merge
    ASSOC_MODULES = {"cross_view": "pipeline.cross_view", "track_v2": "pipeline.tracker"}   # track_v2 = the tracker (module may not exist yet)
    if assoc != "dedup":
        if assoc not in ASSOC_MODULES:
            notes["assoc_error"] = f"unknown LIBRARY_ASSOC {assoc!r}; supported: cross_view (default), track_v2, dedup"
        else:
            try:
                mod = importlib.import_module(ASSOC_MODULES[assoc])
                merge = getattr(mod, "merge")        # same signature as dedup.merge(candidates_path, manifest_path, run_dir, config, readings_path, resolutions_path)
                stages = type(stages)(detect=stages.detect, read=stages.read, resolve=stages.resolve, merge=merge)
                notes["assoc"] = assoc
            except (ImportError, AttributeError) as exc:
                notes["assoc_error"] = f"{ASSOC_MODULES[assoc]}.merge unavailable ({type(exc).__name__}); using the default dedup.merge"
    return configs, avail, stages, notes
