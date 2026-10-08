"""Recorded-video pipeline orchestrator.

    python -m pipeline.run --video sweeps/IMG_2281.MOV --out runs/offline_baseline

Stages (artifact in run dir): frames -> manifest.json, detect -> candidates.json,
read -> readings.json, resolve -> resolutions.json, dedup -> inventory.json,
packet -> claim_packet.json, report -> report.html.

* Every stage is also runnable alone through its own module CLI.
* Resume: a stage is skipped when its artifact exists and the digest of (upstream artifacts + stage
  config + video hash) is unchanged. ``--force`` reruns all, ``--from-stage X`` reruns X onward,
  ``--only a,b`` runs just those stages (upstream artifacts must already exist).
* A stage owned by someone else that is missing/unimplemented is recorded as ``not_implemented``;
  the run continues and the provisional packet shows the gap. Nothing is fabricated.
* ``--fixture`` substitutes clearly marked FIXTURE detections (never real detections) so the
  integration path can be exercised before real detectors exist. Use a different ``--out``.
* Logs: logs/stages.json (per-stage status/latency), logs/usage.jsonl + logs/run_summary.json
  (requests, tokens, cache hits, cost provenance). Country/currency are never inferred.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from pipeline import contracts as C
from providers.cache import canonical_json, sha256_bytes, sha256_file
from providers import gemini
from providers.stages import stage_configs

STAGE_VERSION = "w1-1"  # bump to invalidate resumed artifacts when orchestration semantics change
UPSTREAM = {"frames": [], "detect": ["frames"], "read": ["frames", "detect"], "resolve": ["read"],
            "dedup": ["frames", "detect", "read", "resolve"], "scale": ["frames", "detect", "dedup"], "room": ["frames"],
            "price": ["frames", "dedup"], "packet": ["frames", "dedup", "scale", "room", "price"], "report": ["packet"]}
REQUIRED = {"detect": ["frames"], "read": ["detect"], "resolve": ["read"], "dedup": ["detect"], "scale": ["dedup"], "room": ["frames"],
            "price": ["dedup"], "packet": ["frames"], "report": ["packet"]}  # hard prerequisites (others are optional inputs)


STAGE_MODULES = {"frames": ["frames", "quality"], "detect": ["detect"], "read": ["read_spines"], "resolve": ["resolve"],
                 "dedup": ["dedup"], "scale": ["scale"], "room": ["room"], "price": ["price"], "packet": ["packet"], "report": ["report"]}


def _code_hash(stage: str) -> str:
    """Hash of the stage's own source + shared contracts so code edits invalidate resumed artifacts."""
    root = Path(__file__).resolve().parent
    h = []
    for name in STAGE_MODULES[stage] + ["contracts"]:
        f = root / f"{name}.py"
        h.append(sha256_file(f) if f.is_file() else f"nofile:{name}")
    return _digest(h)


def _digest(parts: list[str]) -> str:
    return sha256_bytes(canonical_json(parts).encode())


IMPORTABLE_DIRS = ("frames", "crops", "pricing_evidence", "detection_responses", "ocr_responses")


def import_run(src: str | Path, out: str | Path, stages: list[str] | None = None) -> dict[str, Any]:
    """Deliberately import EXISTING real stage artifacts (and their evidence dirs) into a new run directory.

    The source is only read. Each imported stage is recorded in run_state.json as ``imported_from`` (not rerun,
    not fresh inference) with no input digest, so a later invocation reruns it unless explicitly skipped.
    """
    import shutil
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "run_state.json"
    state = C.read_json(state_path) if state_path.is_file() else {"stages": {}}
    for stage in (stages or ["frames", "detect", "read", "resolve", "dedup", "price"]):
        f = src / C.ARTIFACTS[stage]
        if not f.is_file():
            continue
        shutil.copy2(f, out / C.ARTIFACTS[stage])
        art = C.read_json(f)
        state["stages"][stage] = {"status": art.get("status", "ok"), "elapsed_s": None, "imported_from": str(src), "fixture": bool(art.get("fixture")),
                                  "detail": f"imported existing artifact from {src.name}; not rerun, not fresh inference"}
    for d in IMPORTABLE_DIRS:
        if (src / d).is_dir() and not (out / d).exists():
            shutil.copytree(src / d, out / d)
    C.write_json(state_path, state)
    return state


class Run:
    def __init__(self, video: Path | None, out: Path, config: dict[str, Any]):
        self.video, self.out, self.config = video, out.resolve(), config
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "logs").mkdir(exist_ok=True)
        self.run_id = config.get("run_id") or self.out.name
        self.config["run_id"] = self.run_id
        self.state_path = self.out / "run_state.json"
        self.state: dict[str, Any] = C.read_json(self.state_path) if self.state_path.is_file() else {"stages": {}}
        self.usage = gemini.UsageLog(self.out / "logs" / "usage.jsonl")
        self.cache = gemini.get_cache(config.get("cache_dir") or (self.out.parent / ".cache"), enabled=not config.get("no_cache"))
        self.has_credential = False
        _ = gemini.default_model()  # triggers repo .env load (<repo>/.env only)
        if not config.get("no_provider") and (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            self.has_credential = True
            config["image_request"] = gemini.make_adapter(
                run_dir=self.out, cache=self.cache, usage=self.usage, model=config.get("model"),
                config=config.get("generation"))
        config.setdefault("cache_dir", str(self.cache.root))
        self.stage_configs, self.stage_availability = stage_configs(config, run_dir=self.out, cache=self.cache, usage=self.usage)

    # ---- paths
    def artifact(self, stage: str) -> Path:
        return self.out / C.ARTIFACTS[stage]

    def exists(self, stage: str) -> bool:
        st = self.state["stages"].get(stage, {})
        return self.artifact(stage).is_file() and st.get("status") in ("ok", "complete", "partial")

    # ---- digests
    def input_digest(self, stage: str) -> str:
        parts = [STAGE_VERSION, stage, _code_hash(stage), canonical_json(self.config.get(stage) or {}),
                 canonical_json({k: self.config.get(k) for k in ("model", "generation", "country", "currency", "detector", "reader", "detector_weights", "detector_python", "pricing_online", "price_source_urls")})]
        if stage == "frames":
            parts += [sha256_file(self.video), canonical_json(self.config.get("frames") or {}),
                      canonical_json(self.config.get("quality") or {}), canonical_json(self.config.get("audit_notes") and C.read_json(self.config["audit_notes"]))]
        if stage == "detect" or stage == "read":
            parts += [canonical_json(self.stage_availability)]
        if stage in ("scale", "room", "packet"):
            for key in ("calibration_path", "room_evidence_path"):
                f = self.config.get(key) or self.out / ("calibration.json" if key == "calibration_path" else "room_evidence.json")
                parts.append(f"{key}={sha256_file(f) if Path(f).is_file() else 'absent'}")
            parts.append(canonical_json(self.config.get("scale") or {}))
        for up in UPSTREAM[stage]:
            a = self.artifact(up)
            parts.append(sha256_file(a) if a.is_file() else f"absent:{up}")
        if stage == "packet":
            parts.append(canonical_json({k: v["status"] for k, v in self.state["stages"].items() if k != "packet" and k != "report"}))
        return _digest(parts)

    def save_state(self) -> None:
        C.write_json(self.state_path, self.state)

    # ---- stage callers
    def _call(self, stage: str) -> Path | None:
        c, o, m = self.config, self.out, self.artifact("frames")
        def mod(name):
            return importlib.import_module(f"pipeline.{name}")
        if stage == "frames":
            return mod("frames").extract(self.video, o, c)
        if stage == "detect":
            fn = getattr(mod("detect"), "detect", None)
            return fn(m, o, self.stage_configs["detect"]) if fn else None
        if stage == "read":
            fn = getattr(mod("read_spines"), "read", None)
            return fn(self.artifact("detect"), m, o, self.stage_configs["read"]) if fn else None
        if stage == "resolve":
            fn = getattr(mod("resolve"), "resolve", None)
            return fn(self.artifact("read"), o, c) if fn else None
        if stage == "dedup":
            fn = getattr(mod("dedup"), "merge", None)
            if not fn:
                return None
            kw = {}
            if self.exists("read"):
                kw["readings_path"] = self.artifact("read")
            if self.exists("resolve"):
                kw["resolutions_path"] = self.artifact("resolve")
            return fn(self.artifact("detect"), m, o, c, **kw)
        if stage == "scale":
            return mod("scale").measure(self.artifact("dedup"), m, o, c)
        if stage == "room":
            return mod("room").estimate(m, o, c)
        if stage == "price":
            return mod('price').price(self.artifact('dedup') if self.exists('dedup') else None, m, o, c)
        if stage == "packet":
            c = {**c, "stage_status": self.stage_summary()}
            for k in ("scale", "room"):
                if self.exists(k):
                    c[k + "_artifact"] = str(self.artifact(k))
            inv = self.artifact("dedup") if self.exists("dedup") else None
            return mod("packet").build(inv, m, o, c)
        if stage == "report":
            return mod("report").render(self.artifact("packet"), o, c)
        raise KeyError(stage)

    def stage_summary(self) -> dict[str, Any]:
        out = {}
        for s in C.STAGE_ORDER:
            if s in ("packet", "report"):
                continue
            st = self.state["stages"].get(s)
            out[s] = ({"status": st["status"], "elapsed_s": st.get("elapsed_s"), "detail": st.get("detail"),
                       "fixture": st.get("fixture", False)} if st else {"status": "skipped", "elapsed_s": None, "detail": "not run"})
        return out

    def validate(self, stage: str) -> list[str]:
        if stage == "report":
            return [] if self.artifact("report").is_file() else ["report.html missing"]
        art = C.read_json(self.artifact(stage))
        if stage == "frames":
            return C.validate_manifest(art, self.out)
        manifest = C.read_json(self.artifact("frames"))
        cand_ids: set[str] = set()
        if self.artifact("detect").is_file():
            d = C.read_json(self.artifact("detect")).get("data") or {}
            cand_ids = {o["observation_id"] for o in (d.get("observations") if d.get("observations") is not None else d.get("detections")) or []}
        if stage == "packet":
            return C.validate_artifact("packet", art, manifest=manifest) + (
                __import__("pipeline.packet", fromlist=["x"]).verify_packet(art, manifest, self.out))
        return C.validate_artifact(stage, art, manifest=manifest, candidate_ids=cand_ids, run_dir=self.out)

    def run_stage(self, stage: str, force: bool) -> None:
        st = self.state["stages"].get(stage)
        for req in REQUIRED.get(stage, []):
            rs = self.state["stages"].get(req, {}).get("status")
            if rs not in ("ok", "complete", "partial") or not self.artifact(req).is_file():
                self.state["stages"][stage] = {"status": "skipped", "elapsed_s": 0.0,
                                               "detail": f"prerequisite stage '{req}' is {rs or 'not run'}"}
                self.save_state()
                return
        digest = self.input_digest(stage)
        if (not force and st and st.get("input_digest") == digest and self.artifact(stage).is_file()
                and st["status"] in ("ok", "complete", "partial")):
            st["resumed"] = True
            print(f"[{stage}] skipped (artifact up to date)")
            self.save_state()
            return
        print(f"[{stage}] running...", flush=True)
        t0 = time.perf_counter()
        before = self.usage.summary()
        rec: dict[str, Any] = {"input_digest": digest, "resumed": False, "fixture": False}
        try:
            result = self._call(stage)
            if result is None:
                rec.update(status="not_implemented", detail=f"pipeline.{ {'read':'read_spines'}.get(stage, stage) } has no entry point yet")
            else:
                problems = self.validate(stage)
                art_status = (C.read_json(self.artifact(stage)).get("status") if stage != "report" else "ok")
                rec.update(artifact=C.rel_path(result, self.out), status=art_status if art_status in C.STATUSES else "ok")
                if stage != "report" and art_status == "partial":
                    iss = C.read_json(self.artifact(stage)).get("issues") or []
                    texts = [str(i.get("reason") or i.get("detail") or i) if isinstance(i, dict) else str(i) for i in iss]
                    n = len(texts)
                    rec["detail"] = f"{n} issue(s); first: " + " | ".join(t[:160] for t in texts[:2]) if n else "partial"
                if problems:
                    rec.update(status="failed", detail="artifact failed validation: " + "; ".join(problems[:6]),
                               validation_problems=problems)
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            rec.update(status="failed", detail=f"{type(exc).__name__}: {str(exc)[:300]}",
                       traceback_tail=traceback.format_exc().splitlines()[-4:])
        rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
        after = self.usage.summary()
        rec["requests"] = after["requests"] - before["requests"]
        rec["cache_hits"] = after["cache_hits"] - before["cache_hits"]
        self.state["stages"][stage] = rec
        self.save_state()
        print(f"[{stage}] {rec['status']} in {rec['elapsed_s']}s" + (f" - {rec.get('detail')}" if rec.get("detail") else ""))

    def finish(self, t_start: float) -> dict[str, Any]:
        summary = {
            "run_id": self.run_id, "end_to_end_elapsed_s": round(time.perf_counter() - t_start, 3),
            "stages": {s: {k: v for k, v in rec.items() if k in ("status", "elapsed_s", "requests", "cache_hits", "resumed", "detail", "fixture")}
                       for s, rec in self.state["stages"].items()},
            "provider": {"stage_availability": self.stage_availability, "credential_available": self.has_credential, "model_default": gemini.default_model(),
                         **self.usage.summary(), "cache_dir": str(self.cache.root),
                         "cache_process_stats": self.cache.stats()},
            "note": "elapsed_s of resumed stages is the original run's time; end_to_end_elapsed_s is this invocation only",
        }
        C.write_json(self.out / "logs" / "run_summary.json", summary)
        C.write_json(self.out / "logs" / "stages.json", self.state["stages"])
        return summary


def run(video: str | Path | None, out: str | Path, *, stages: list[str] | None = None, from_stage: str | None = None,
        force: bool = False, config: dict[str, Any] | None = None, fixture: bool = False) -> dict[str, Any]:
    t_start = time.perf_counter()
    r = Run(Path(video) if video else None, Path(out), dict(config or {}))
    order = list(stages) if stages else list(C.STAGE_ORDER)
    for s in order:
        if s not in UPSTREAM:
            raise SystemExit(f"unknown stage {s!r}; choose from {C.STAGE_ORDER}")
    if from_stage:
        i = C.STAGE_ORDER.index(from_stage)
        order = [s for s in C.STAGE_ORDER[i:] if s in order]
        force_from = set(order)
    else:
        force_from = set()
    if r.video is None and "frames" in order and not r.artifact("frames").is_file():
        raise SystemExit("--video is required to run the frames stage")
    if r.video is None:
        order = [s for s in order if s != "frames"]
    for s in order:
        if fixture and s == 'price' and not r.config.get('pricing_fixture'):
            # Preserve the legacy unpriced fixture contract; price tests opt in.
            r.state['stages']['price'] = {'status':'skipped','elapsed_s':0,'detail':'legacy fixture has no sourced valuation'}
            continue
        if fixture and s in ("detect", "read", "resolve", "dedup"):
            from tests.fixtures import make_fixture_stage  # test-only module, never used for real runs
            make_fixture_stage(r, s)
            continue
        r.run_stage(s, force or s in force_from)
    summary = r.finish(t_start)
    import asyncio
    from providers.stages import close_runtimes
    asyncio.run(close_runtimes())
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video"); ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma-separated stages to run"); ap.add_argument("--from-stage", choices=C.STAGE_ORDER)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample-fps", type=float); ap.add_argument("--overlap-step", type=float)
    ap.add_argument("--audit-notes", help="JSON manual visual review to embed in the manifest")
    ap.add_argument("--model", help="override GEMINI_MODEL for this run"); ap.add_argument("--max-concurrency", type=int)
    ap.add_argument("--cache-dir"); ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-provider", action="store_true", help="never create a model adapter (crop-only / unknown text)")
    ap.add_argument("--country"); ap.add_argument("--currency")
    ap.add_argument('--detector', choices=['auto','local','gemini','none'])
    ap.add_argument('--reader', choices=['auto','apple','easyocr','gemini','none'])
    ap.add_argument('--detector-weights'); ap.add_argument('--detector-python')
    ap.add_argument('--pricing-online', action='store_true')
    ap.add_argument('--price-source-config', help='JSON with ISBN-keyed price_source_urls; no memory prices')
    ap.add_argument("--import-run", help="import existing real artifacts (read-only source) into --out; not rerun, not fresh")
    ap.add_argument("--calibration", help="calibration JSON (purpose=calibration): user-supplied known dimensions on a shelf plane")
    ap.add_argument("--room-evidence", help="room evidence JSON (purpose=room_evidence): wall quads + height anchor")
    ap.add_argument("--fixture", action="store_true", help="use FIXTURE detections (clearly marked; not real)")
    a = ap.parse_args(argv)
    if a.max_concurrency:
        os.environ["GEMINI_MAX_CONCURRENCY"] = str(a.max_concurrency)
    cfg: dict[str, Any] = {"frames": {}}
    if a.sample_fps: cfg["frames"]["sample_fps"] = a.sample_fps
    if a.overlap_step: cfg["frames"]["overlap_step"] = a.overlap_step
    for k, v in (("audit_notes", a.audit_notes), ("model", a.model), ("cache_dir", a.cache_dir), ("country", a.country),
                 ("currency", a.currency)):
        if v: cfg[k] = v
    for key in ('detector', 'reader', 'detector_weights', 'detector_python', 'pricing_online'):
        if getattr(a, key) is not None: cfg[key] = getattr(a, key)
    if a.price_source_config: cfg.update(C.read_json(a.price_source_config))
    if a.calibration: cfg["calibration_path"] = a.calibration
    if a.room_evidence: cfg["room_evidence_path"] = a.room_evidence
    cfg["no_cache"], cfg["no_provider"] = a.no_cache, a.no_provider
    if a.import_run:
        import_run(a.import_run, a.out)
        cfg["imported_from"] = a.import_run
    summary = run(a.video, a.out, stages=a.only.split(",") if a.only else None, from_stage=a.from_stage,
                  force=a.force, config=cfg, fixture=a.fixture)
    print(json.dumps({k: summary[k] for k in ("run_id", "end_to_end_elapsed_s")}, indent=2))
    bad = [s for s, v in summary["stages"].items() if v["status"] in ("failed",)]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
