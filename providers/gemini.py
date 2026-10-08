"""Shared async structured-image model interface.

Stages call ``structured_image_request`` and never load keys, cache or count usage themselves.

    from providers.gemini import structured_image_request, ImageInput, UsageLog, get_cache

    result = await structured_image_request(
        images=[ImageInput(path="runs/x/frames/f00012.jpg")],   # or ImageInput(data=jpeg_bytes)
        prompt="...",
        response_schema={...JSON schema...} | PydanticModelClass | None,
        model=None,                      # default: env GEMINI_MODEL
        config={"temperature": 0.0, "media_resolution": "high"},   # optional passthrough
        stage="detect", tag="f00012",    # for usage accounting only (not part of the cache key)
        cache=get_cache(cache_dir),      # optional; None disables caching
        usage=usage_log,                 # optional UsageLog
    )
    result.parsed      # parsed JSON (dict/list) when response_schema given or JSON mime used
    result.text        # raw model text
    result.cache_hit   # True if served from cache (no request made, usage tokens = 0)
    result.usage       # {"prompt_tokens","output_tokens","thought_tokens","total_tokens"}  (ints or None)
    result.latency_s   # wall time of this call (0.0 for cache hit)

Configuration is by environment only: GOOGLE_API_KEY or GEMINI_API_KEY, GEMINI_MODEL (default
below), GEMINI_MAX_CONCURRENCY (default 4), GEMINI_MAX_RETRIES (default 3), GEMINI_THINKING_BUDGET (default 512 thinking
tokens per request for the image stages in GEMINI_THINKING_BUDGET_STAGES, default "read,detect"; "off" = the model's own
dynamic thinking; see ``thinking_budget``). A ``<repo>/.env`` file
is read if present (setdefault; never overrides real environment). Billed cost is NOT assumed:
``cost_usd`` is None unless GEMINI_PRICE_IN_PER_MTOK and GEMINI_PRICE_OUT_PER_MTOK are set by the
user, in which case it is labelled an estimate from those user-supplied rates.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from providers.cache import DiskCache, make_key, sha256_bytes

DEFAULT_MODEL = "gemini-3.8-flash"  # configurable default only; not a claim about availability/price
REPO_ROOT = Path(__file__).resolve().parents[1]

_dotenv_loaded = False


def _load_repo_dotenv() -> None:
    """Load <repo>/.env (only) into os.environ without overriding existing variables."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    env = REPO_ROOT / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class ProviderError(RuntimeError):
    """Request failed after retries (message never contains credentials)."""


class StructuredOutputError(ProviderError):
    """Model returned text that is not valid JSON for a structured request."""


class MissingCredentialError(ProviderError):
    pass


@dataclass
class ImageInput:
    """One image: a file path or raw bytes. Bytes are sent as-is (native resolution, no resize)."""
    path: str | Path | None = None
    data: bytes | None = None
    mime_type: str | None = None

    def load(self) -> tuple[bytes, str]:
        if self.data is None:
            if self.path is None:
                raise ValueError("ImageInput needs path or data")
            self.data = Path(self.path).read_bytes()
        mime = self.mime_type or (mimetypes.guess_type(str(self.path))[0] if self.path else None) or "image/jpeg"
        return self.data, mime


@dataclass
class GeminiResult:
    parsed: Any
    text: str
    model: str
    cache_hit: bool
    usage: dict[str, int | None]
    latency_s: float
    request_key: str
    cost_usd: float | None = None
    cost_note: str = "billed cost unknown: provider pricing not verified or not configured"


@dataclass
class UsageLog:
    """Thread/async-safe accumulator of every request; optionally mirrored to a JSONL file."""
    path: Path | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self.records.append(rec)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def summary(self, stage: str | None = None) -> dict[str, Any]:
        with self._lock:
            recs = [r for r in self.records if stage in (None, r.get("stage"))]
        def tot(key):
            vals = [r["usage"].get(key) for r in recs if not r["cache_hit"]]
            vals = [v for v in vals if v is not None]
            return sum(vals) if vals else None
        costs = [r["cost_usd"] for r in recs if r.get("cost_usd") is not None]
        return {
            "requests": sum(1 for r in recs if not r["cache_hit"]),
            "cache_hits": sum(1 for r in recs if r["cache_hit"]),
            "prompt_tokens": tot("prompt_tokens"), "output_tokens": tot("output_tokens"),
            "thought_tokens": tot("thought_tokens"), "total_tokens": tot("total_tokens"),
            "request_latency_s": round(sum(r["latency_s"] for r in recs if not r["cache_hit"]), 3),
            "models": sorted({r["model"] for r in recs}),
            "cost_usd": round(sum(costs), 6) if costs else None,
            "cost_note": ("estimated from user-configured GEMINI_PRICE_* rates, not billed cost" if costs
                          else "billed cost unknown: provider pricing not verified or not configured"),
        }


Transport = Callable[[str, list[tuple[bytes, str]], str, dict[str, Any], Any], Awaitable[tuple[str, dict[str, int | None]]]]
_transport: Transport | None = None  # tests/fixtures may inject; real calls use google-genai
_caches: dict[str, DiskCache] = {}
_semaphores: dict[tuple[int, int], asyncio.Semaphore] = {}


def get_cache(cache_dir: str | Path, enabled: bool = True) -> DiskCache:
    """Process-wide cache per directory so hit/miss counters aggregate across stages."""
    key = f"{Path(cache_dir).resolve()}|{enabled}"
    if key not in _caches:
        _caches[key] = DiskCache(cache_dir, enabled=enabled)
    return _caches[key]


def set_transport(fn: Transport | None) -> None:
    """Inject a fake transport (tests/fixtures). ``None`` restores the real google-genai transport."""
    global _transport
    _transport = fn


def default_model() -> str:
    _load_repo_dotenv()
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _semaphore() -> asyncio.Semaphore:
    _load_repo_dotenv()
    limit = max(1, int(os.environ.get("GEMINI_MAX_CONCURRENCY", "4")))
    key = (id(asyncio.get_running_loop()), limit)
    if key not in _semaphores:
        _semaphores[key] = asyncio.Semaphore(limit)
    return _semaphores[key]


async def _real_transport(model: str, images: list[tuple[bytes, str]], prompt: str,
                          config: dict[str, Any], response_schema: Any) -> tuple[str, dict[str, int | None]]:
    from providers import hosted_gate
    hosted_gate.check(f"structured_image_request:{model}")      # user kill-switch: raises HostedDisabledError, no request is made
    _load_repo_dotenv()
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise MissingCredentialError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in the environment or <repo>/.env")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    cfg = dict(config)
    media_res = cfg.pop("media_resolution", None)
    system = cfg.pop("system_instruction", None)
    gen = types.GenerateContentConfig(system_instruction=system, **cfg)
    if response_schema is not None:
        gen.response_mime_type = "application/json"
        gen.response_schema = response_schema
    if media_res:
        gen.media_resolution = getattr(types.MediaResolution, f"MEDIA_RESOLUTION_{str(media_res).upper()}")
    parts = [types.Part.from_bytes(data=d, mime_type=m) for d, m in images] + [types.Part(text=prompt)]
    resp = await client.aio.models.generate_content(model=model, contents=parts, config=gen)
    u = resp.usage_metadata
    usage = {
        "prompt_tokens": getattr(u, "prompt_token_count", None),
        "output_tokens": getattr(u, "candidates_token_count", None),
        "thought_tokens": getattr(u, "thoughts_token_count", None),
        "total_tokens": getattr(u, "total_token_count", None),
    }
    hosted_gate.record_call(f"structured_image_request:{model}", usage, _estimate_cost(usage))
    return resp.text or "", usage


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (MissingCredentialError, StructuredOutputError)):
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code == 429 or code >= 500
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, TimeoutError))


def _estimate_cost(usage: dict[str, int | None]) -> float | None:
    try:
        p_in = float(os.environ["GEMINI_PRICE_IN_PER_MTOK"])
        p_out = float(os.environ["GEMINI_PRICE_OUT_PER_MTOK"])
    except (KeyError, ValueError):
        return None
    if usage.get("prompt_tokens") is None or usage.get("output_tokens") is None:
        return None
    out = (usage["output_tokens"] or 0) + (usage.get("thought_tokens") or 0)
    return (usage["prompt_tokens"] * p_in + out * p_out) / 1e6


def _schema_fingerprint(schema: Any) -> Any:
    if schema is None or isinstance(schema, (dict, list, str)):
        return schema
    for attr in ("model_json_schema",):
        if hasattr(schema, attr):
            return getattr(schema, attr)()
    return repr(schema)


DEFAULT_THINKING_BUDGET = 512
DEFAULT_BUDGET_STAGES = ("read", "detect")


def thinking_budget(stage: str, environ: dict[str, str] | None = None) -> int | None:
    """Thinking-token cap for one request of ``stage`` (None = no cap: the model's dynamic thinking). Reading a spine crop or listing boxes needs little reasoning;
    on 2026-10-07 the YugoTest7 reader spent up to ~19,000 thinking tokens and ~3 min on single crops (median ~1,300 on YugoTest11), which drained the prepaid
    credits. GEMINI_THINKING_BUDGET: an integer >= 0, or 'off'; GEMINI_THINKING_BUDGET_STAGES: comma-separated stage names it applies to."""
    env = environ if environ is not None else os.environ
    stages = {x.strip() for x in env.get("GEMINI_THINKING_BUDGET_STAGES", ",".join(DEFAULT_BUDGET_STAGES)).split(",") if x.strip()}
    if (stage or "") not in stages:
        return None
    raw = str(env.get("GEMINI_THINKING_BUDGET", DEFAULT_THINKING_BUDGET)).strip().lower()
    if raw in ("off", "none", "dynamic", "-1", ""):
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_THINKING_BUDGET


def _with_thinking_budget(config: dict[str, Any], stage: str) -> dict[str, Any]:
    """The config SENT for this request: the caller's own thinking_config always wins; otherwise the stage budget is added. The cache key stays the caller's
    config, so readings cached before the guard existed remain valid cache hits."""
    budget = thinking_budget(stage)
    if budget is None or "thinking_config" in config:
        return config
    return {**config, "thinking_config": {"thinking_budget": budget}}


async def structured_image_request(
    *, images: list[ImageInput | bytes | str | Path], prompt: str, response_schema: Any = None,
    model: str | None = None, config: dict[str, Any] | None = None, stage: str = "", tag: str = "",
    cache: DiskCache | None = None, usage: UsageLog | None = None,
) -> GeminiResult:
    """One structured request over zero or more images. See module docstring for semantics."""
    model = model or default_model()
    config = dict(config or {})
    loaded: list[tuple[bytes, str]] = []
    for img in images:
        if isinstance(img, (str, Path)):
            img = ImageInput(path=img)
        elif isinstance(img, (bytes, bytearray)):
            img = ImageInput(data=bytes(img))
        loaded.append(img.load())

    key = make_key("gemini.structured_image", model=model, prompt=prompt, config=config,
                   schema=_schema_fingerprint(response_schema),
                   inputs=[{"sha256": sha256_bytes(d), "mime": m} for d, m in loaded])

    cached = cache.get(key) if cache is not None else None
    if cached is not None:
        try:
            from providers import hosted_gate
            hosted_gate.record_cache_hit(f"structured_image_request:{model}")
        except Exception:  # noqa: BLE001 - counters are advisory
            pass
        res = GeminiResult(parsed=cached["parsed"], text=cached["text"], model=model, cache_hit=True,
                           usage={"prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "total_tokens": 0},
                           latency_s=0.0, request_key=key, cost_usd=None,
                           cost_note="served from cache; no request made")
        if usage is not None:
            usage.add({"stage": stage, "tag": tag, "model": model, "cache_hit": True, "usage": res.usage,
                       "latency_s": 0.0, "cost_usd": None, "request_key": key})
        return res

    transport = _transport or _real_transport
    send_config = _with_thinking_budget(config, stage)
    retries = max(0, int(os.environ.get("GEMINI_MAX_RETRIES", "3")))
    last: Exception | None = None
    async with _semaphore():
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                text, use = await transport(model, loaded, prompt, send_config, response_schema)
                latency = time.perf_counter() - t0
                parsed: Any = None
                if response_schema is not None or config.get("response_mime_type") == "application/json":
                    try:
                        parsed = json.loads(text)
                    except ValueError as exc:
                        raise StructuredOutputError(f"model output is not valid JSON ({len(text)} chars)") from exc
                break
            except Exception as exc:  # noqa: BLE001 - classified below, message sanitised
                last = exc
                if attempt >= retries or not _retryable(exc):
                    raise (exc if isinstance(exc, ProviderError) else ProviderError(
                        f"{type(exc).__name__} from model request")) from None
                await asyncio.sleep(min(2 ** attempt, 8))
        else:  # pragma: no cover
            raise ProviderError(f"request failed: {type(last).__name__}")

    cost = _estimate_cost(use)
    if cache is not None:
        cache.put(key, {"parsed": parsed, "text": text},
                  meta={"model": model, "stage": stage, "tag": tag, "usage": use})
    res = GeminiResult(parsed=parsed, text=text, model=model, cache_hit=False, usage=use,
                       latency_s=round(latency, 3), request_key=key, cost_usd=cost,
                       cost_note=("estimated from user-configured GEMINI_PRICE_* rates, not billed cost"
                                  if cost is not None else GeminiResult.cost_note))
    if usage is not None:
        usage.add({"stage": stage, "tag": tag, "model": model, "cache_hit": False, "usage": use,
                   "latency_s": res.latency_s, "cost_usd": cost, "request_key": key})
    return res


def make_adapter(*, run_dir: str | Path, cache: DiskCache | None = None, usage: UsageLog | None = None,
                 model: str | None = None, default_stage: str = "", config: dict[str, Any] | None = None,
                 image_preprocess: Any = None):
    """``image_preprocess(path) -> ImageInput | Path`` (optional) transforms each image in memory before encoding; files are never modified."""
    """Canonical adapter for stages. Accepts BOTH calling styles agreed so far:

    reader:    await adapter(image_paths=[...], prompt=..., response_schema=..., task="read_spine", config={...})
    detector:  await adapter(image_path=..., prompt=..., response_schema=..., model=..., run_dir=...,
                             stage="detect", request_id="f00012")
    Returns ``{"data": parsed_json, "metrics": {...}, "issues": []}``. Image paths may be absolute or
    relative to ``run_dir``. Failures raise ProviderError/MissingCredentialError (message has no secrets).
    """
    run_dir = Path(run_dir)
    base_config = dict(config or {})

    async def adapter(*, prompt: str, response_schema: Any = None, image_paths: list[Any] | None = None,
                      image_path: Any = None, model: str | None = model, stage: str | None = None,
                      task: str | None = None, request_id: str | None = None, run_dir: Any = None,
                      config: dict[str, Any] | None = None, **_ignored: Any) -> dict[str, Any]:
        paths = list(image_paths or ([] if image_path is None else [image_path]))
        resolved = [Path(p) if Path(p).is_absolute() else Path(_run_dir_for(run_dir)) / p for p in paths]
        images: list[Any] = [image_preprocess(p) for p in resolved] if image_preprocess else resolved   # in-memory only; files untouched
        gen_cfg = {**base_config, **{k: v for k, v in (config or {}).items() if k in _GEN_KEYS}}
        res = await structured_image_request(
            images=images, prompt=prompt, response_schema=response_schema, model=model, config=gen_cfg,
            stage=stage or task or default_stage, tag=request_id or task or "", cache=cache, usage=usage)
        return {"data": res.parsed if res.parsed is not None else res.text, "issues": [],
                "metrics": {"request_count": 0 if res.cache_hit else 1, "cache_hits": 1 if res.cache_hit else 0,
                            "model": res.model, "latency_s": res.latency_s, "usage": res.usage,
                            "actual_cost": res.cost_usd, "cost_provenance": res.cost_note}}

    _holder = {"run_dir": run_dir}

    def _run_dir_for(override: Any) -> Path:
        return Path(override) if override else _holder["run_dir"]

    return adapter


_GEN_KEYS = {"temperature", "top_p", "top_k", "max_output_tokens", "media_resolution", "system_instruction",
             "thinking_config", "seed"}


def is_mocked() -> bool:
    """True when a fake transport is injected (tests/fixtures). Used to label results as mock, never fresh."""
    return _transport is not None


def credential_available() -> bool:
    _load_repo_dotenv()
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
