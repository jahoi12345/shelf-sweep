"""Post-sweep "finish reading" step: read every attemptable spine group the packet's wall-clock budget left unread, with no budget.

At End sweep the group reading is clamped to what is left of the packet target (LIBRARY_PACKET_TARGET_S), which can be the 30 s minimum: YugoTest16 read 120 of
964 groups and nothing read the rest later. This step runs in the build queue (app.post_sweep) AFTER the packet exists and:

  1. plans the run's groups again with the engine's own policy (app.group_reader.plan: best 1-2 sightings per group, too-small / low-contrast groups stay
     unattempted), and picks the attemptable groups that have no completed reading yet (plus a second sighting where the first read < 8 characters);
  2. reads them with the SAME reader the engine used (providers.stages / providers.readers.apply: Gemini when LIBRARY_READER=gemini and the hosted gate is
     connected, the shared hosted cache, the same preprocessing), 16 requests at a time for a hosted reader; "skipped: hosted reader off" when the gate is off;
  3. resolves the new readings (and any reading the sweep's deadline left unresolved) against the catalogues, one reading per call exactly like the engine;
  4. under the queue's packet lock, once the 2D panorama / count / duplicate-merge chain has settled: merges readings.json / resolutions.json / inventory.json,
     re-runs the instance layer (its reading-contradiction veto sees the new readings), attaches the identities (pipeline.identity_attach);
  5. prices the identified works (pipeline.price_attach, no wall-clock budget, shared price cache) and attaches the prices under the lock;
  6. sets stage_status read / resolve / price (ok when nothing is left), updates the group-reading plan / provenance and re-renders the report.

Everything it writes is either a new file under ``<run>/finish_reading/`` or an atomic replacement (a ``*.pre_finish_reading.json`` copy of the old file is
kept once). State survives a restart: finished chunks are files, so a restarted step re-uses them (the hosted cache makes a repeated call free anyway).
Nothing here edits frames or detections. Model readings are never truth: identified != correct.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import itertools
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2

from app import group_reader as GR
from pipeline import contracts as C

DIR = "finish_reading"
CHUNK = 3                                   # observations per read call, as the engine
SUFFIXES = ("readings", "resolutions", "inventory")


class Ctl:
    """What the build queue hands to the step: progress sink, stop test, packet lock, 'chain settled' wait, instance step. All optional (CLI / tests)."""

    def __init__(self, progress: Callable[..., None] | None = None, stop: Callable[[], bool] | None = None, packet_lock: Any = None,
                 settle: Callable[[], bool] | None = None, instances: Callable[[], dict[str, Any]] | None = None):
        self.progress = progress or (lambda *a, **k: None)
        self.stopped = stop or (lambda: False)
        self.packet_lock = packet_lock if packet_lock is not None else contextlib.nullcontext()
        self.settle = settle or (lambda: True)
        self.instances = instances


def _load(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _keep_once(path: Path) -> None:
    dst = path.with_name(path.stem + ".pre_finish_reading" + path.suffix)
    if path.is_file() and not dst.is_file():
        shutil.copy2(path, dst)


# ------------------------------------------------------------------------------------------------------------------------------ planning
def plan_unread(run_dir: str | Path) -> dict[str, Any]:
    """The engine's group plan on the saved run, and what is still unread. -> {plan, todo1: [(instance_id, sighting)], done, readings, resolutions, ...}."""
    run = Path(run_dir)
    inv = _load(run / "inventory.json")
    cand = _load(run / "candidates.json")["data"]
    man = _load(run / "manifest.json")
    obs = {o["observation_id"]: o for o in cand.get("observations", cand.get("detections", []))}
    frames = {f["frame_id"]: f for f in man["data"]["frames"]}
    books = inv["data"]["books"]
    cache: dict[str, Any] = {}

    def load_gray(fid: str):
        if fid not in cache:
            cache[fid] = cv2.imread(str(run / frames[fid]["image_path"]), cv2.IMREAD_GRAYSCALE)
        return cache[fid]
    plan = GR.plan(books, obs, frames, load_gray, per_group=2)
    cache.clear()
    readings = _load(run / "readings.json")["data"]["readings"] if (run / "readings.json").is_file() else []
    done = {r["observation_id"] for r in readings if r.get("ocr_attempt") == "completed"}
    for f in sorted((run / DIR).glob("grp_*.readings.json")):          # chunks of an interrupted earlier run of this step
        with contextlib.suppress(OSError, ValueError, KeyError):
            done |= {r["observation_id"] for r in _load(f)["data"]["readings"] if r.get("ocr_attempt") == "completed"}
    todo1 = [(g["instance_id"], g["sightings"][0]) for g in plan["groups"]
             if g["status"] == "attempt" and not any(s["observation_id"] in done for s in g["sightings"])]
    todo1.sort(key=lambda t: t[1]["score"], reverse=True)
    return {"plan": plan, "todo1": todo1, "done": done, "obs": obs, "books": books, "manifest_run_id": man["run_id"]}


def _resolve_priority(reading: dict[str, Any]) -> tuple[int, int]:
    text = str(reading.get("visible_text") or "")
    complete = bool(reading.get("title_complete") and reading.get("title_text"))
    tier = 0 if complete and reading.get("author_text") else 1 if complete else 2 if len(text) >= 8 else 3
    return (tier, -len(text))


def unresolved_readings(readings: list[dict[str, Any]], resolutions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Completed readings that have no resolution row, or whose resolution was cut by the sweep's deadline."""
    res = {r.get("observation_id"): r for r in resolutions}
    out = []
    for r in readings:
        if r.get("ocr_attempt") != "completed":
            continue
        x = res.get(r["observation_id"])
        if x is None or any("resolve_budget_exhausted" in str(i) for i in x.get("issues") or []):
            out.append(r)
    return out


# ------------------------------------------------------------------------------------------------------------------------------ stages
class _Stages:
    def __init__(self):
        from pipeline import read_spines, resolve
        self.read, self.resolve = read_spines.read, resolve.resolve


def _config(run: Path, overrides: dict[str, Any] | None) -> dict[str, Any]:
    cfg = {"run_id": run.name, "country": os.getenv("LIBRARY_COUNTRY", "US"), "currency": os.getenv("LIBRARY_CURRENCY", "USD"),
           "catalogue_online": os.getenv("LIBRARY_CATALOGUE_ONLINE", "") == "1", "pricing_online": os.getenv("LIBRARY_PRICING_ONLINE", "") == "1",
           "max_books": None, "cache_dir": str(run.parent / ".cache")}       # the same cache the server gives the engine (RUNS_DIR/.cache)
    cfg.update(overrides or {})
    return cfg


def _pool(fn: Callable[[Any], Any], jobs: list[Any], workers: int, stop: Callable[[], bool], tick: Callable[[Any], None]) -> None:
    if not jobs:
        return
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        pending = set()
        it = iter(jobs)
        for job in itertools.islice(it, workers * 2):
            pending.add(ex.submit(fn, job))
        while pending:
            done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in done:
                tick(f.result())
                if not stop():
                    nxt = next(it, None)
                    if nxt is not None:
                        pending.add(ex.submit(fn, nxt))


# ------------------------------------------------------------------------------------------------------------------------------ merge into the run
def refresh_inventory(inv_env: dict[str, Any], readings: dict[str, dict[str, Any]], resolutions: dict[str, dict[str, Any]]) -> dict[str, int]:
    """dedup._identity on the current readings/resolutions for every group + the best visible text, in place (the engine's _refresh_literal_identities and
    _apply_group_readings). A group that was accepted keeps its identity unless the new one is the same work (zero loss)."""
    from pipeline import dedup as _dd
    id_issues = set(_dd._identity([], {}, {})[1])
    _conf = {"decision": "accepted", "matching_evidence": ["x"], "source_url": "u"}
    id_issues |= set(_dd._identity([{"observation_id": "a"}, {"observation_id": "b"}], {}, {"a": {**_conf, "work": "1"}, "b": {**_conf, "work": "2"}})[1])
    key = lambda i: (str(i.get("work") or ""), str(i.get("title") or "").casefold())
    out = {"groups": 0, "newly_accepted": 0, "kept": 0, "with_text": 0}
    for g in inv_env["data"]["books"]:
        out["groups"] += 1
        old = g.get("identity") or {}
        n, issues = _dd._identity([{"observation_id": o} for o in g["observation_ids"]], readings, resolutions)
        if old.get("decision") == "accepted" and (n["decision"] != "accepted" or key(n) != key(old)):
            out["kept"] += 1
        else:
            if n["decision"] != old.get("decision") and old.get("decision") != "accepted" and not (g.get("issues") and g["issues"][-1] in id_issues):
                pass                       # the identity issue line is not where dedup puts it: do not guess, keep the identity as it was
            else:
                decision_before = old.get("decision")
                visible = {k: old[k] for k in ("visible_text", "visible_text_observation_id", "visible_text_reader", "reading_union_text", "readings_considered", "reading_status") if k in old}
                g["identity"] = {**n, **visible}
                if n["decision"] != decision_before:
                    g["status"] = "identified" if n["decision"] == "accepted" else "unidentified"
                    if decision_before != "accepted" and g.get("issues"):
                        g["issues"].pop()
                    g.setdefault("issues", []).extend(issues)
                    out["newly_accepted"] += n["decision"] == "accepted"
        c = GR.combine(g, readings)
        ident = g.setdefault("identity", {})
        if c["best"]:
            ident["visible_text"] = c["best"]["text"]
            ident["visible_text_observation_id"] = c["best"]["observation_id"]
            ident["visible_text_reader"] = c["best"]["reader"]
            out["with_text"] += 1
        if c["union_text"]:
            ident["reading_union_text"] = c["union_text"]
        ident["readings_considered"] = c["readings_considered"]
        ident["reading_status"] = "literal reading, unverified"
    return out


def _member_lines(p: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for line in list(p.get("books") or []) + list(p.get("unconfirmed_books") or []) for m in (line.get("merged_members") or [line])]


def patch_visible_text(packet_data: dict[str, Any], readings: dict[str, dict[str, Any]]) -> int:
    """The best reading of every track-group line (plain lines and members of merged instance lines) as its visible_text, like the engine's group combine."""
    n = 0
    for m in _member_lines(packet_data):
        if any((c or {}).get("kind") == "set_title" for c in m.get("user_corrections") or []):
            continue
        best = GR.combine({"observation_ids": (m.get("evidence") or {}).get("observation_ids") or []}, readings)["best"]
        if best and best["text"] != m.get("visible_text"):
            m["visible_text"] = best["text"]
            n += 1
    return n


# ------------------------------------------------------------------------------------------------------------------------------ the step
def run(run_dir: str | Path, ctl: Ctl | None = None, *, stages: Any = None, config: dict[str, Any] | None = None, reader: tuple | None = None,
        price: bool = True, render: bool = True) -> dict[str, Any]:
    """Run the whole step on one finished run. Returns {'state': 'done'|'partial'|'skipped'|'interrupted', ...} with before/after counts.
    ``stages``: object with .read / .resolve (tests inject a fake reader); ``reader``: (stage_configs, availability) to skip the provider setup."""
    from providers import gemini
    gemini._load_repo_dotenv()
    run = Path(run_dir).resolve()
    ctl = ctl or Ctl()
    t_start = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    fr = run / DIR
    fr.mkdir(exist_ok=True)
    cfg = _config(run, config)
    before = snapshot(run)
    pl = plan_unread(run)
    todo1, plan = pl["todo1"], pl["plan"]
    readings_rows = _load(run / "readings.json")["data"]["readings"]
    res_rows = _load(run / "resolutions.json")["data"]["resolutions"] if (run / "resolutions.json").is_file() else []
    need_res = unresolved_readings(readings_rows, res_rows)
    summary: dict[str, Any] = {"state": "done", "before": before, "groups_attemptable": plan["counts"]["attempt"], "groups_unread_at_start": len(todo1),
                               "readings_unresolved_at_start": len(need_res), "started_at": t_start}
    if not todo1 and not need_res and not before.get("price_budget_skipped"):
        sc_out = second_country_step(run, ctl, render=render) if price else {"state": "skipped", "reason": "pricing off for this step"}
        summary.update(reason="nothing left to read, resolve or price", after=before, second_country=sc_out, finished_at=time.time())
        return summary
    # ---- the reader (the same selection, gate and cache as the engine)
    usage = gemini.UsageLog(run / "logs" / "usage.jsonl")
    usage_before = usage_summary(run)
    new_readings: list[dict[str, Any]] = []
    flags = {"gate_off": False, "read_failed": 0}
    if stages is None:
        stages = _Stages()
    if todo1:
        if reader is None:
            from providers import readers as PH
            from providers.stages import stage_configs
            cfg_cap = {**cfg, "hosted_max_calls": 2 * len(todo1) + 20}      # the per-run hosted cap (800) is a runaway guard; this step's own bound is its plan
            cache = gemini.get_cache(cfg["cache_dir"])
            sc, avail, _stages_unused, _notes = PH.apply(cfg_cap, stages, run_dir=run, cache=cache, usage=usage, stage_configs_fn=stage_configs)
        else:
            sc, avail = reader
        ra = avail.get("read") or {}
        if not ra.get("available"):
            why = "hosted reader off" if ra.get("selection") == "gemini" else str(ra.get("reason") or "no reader available")
            summary.update(state="skipped", reason=why, after=before, finished_at=time.time())
            return summary
        hosted = ra.get("selection") == "gemini"
        workers = int(os.environ.get("LIBRARY_HOSTED_READ_CONCURRENCY", 16)) if hosted else int(os.environ.get("LIBRARY_READ_WORKERS", 3))
        manifest = run / "manifest.json"
        obs, run_id = pl["obs"], pl["manifest_run_id"]
        seq = itertools.count(1)
        total = len(todo1)
        counters = {"done": 0, "total": total}

        def read_chunk(job: tuple[str, list[tuple[str, dict[str, Any]]]]) -> int:
            tag, items = job
            if ctl.stopped():
                return 0
            if hosted:
                from providers import hosted_gate as HG
                if HG.disabled():
                    flags["gate_off"] = True
                    return 0
            n = next(seq)
            ids = [it[1]["observation_id"] for it in items]
            slice_path = C.write_json(fr / f"grp_{tag}_{stamp}_{n:04d}.candidates.json", C.envelope(run_id, "detect", {"observations": [obs[i] for i in ids]}, status="ok"))
            rcfg = {**sc["read"], "session_revision": None, "readings_filename": f"{DIR}/grp_{tag}_{stamp}_{n:04d}.readings.json", "max_books": None}
            try:
                rpath = stages.read(slice_path, manifest, run, rcfg)
                rows = C.read_json(rpath)["data"]["readings"]
            except Exception as exc:  # noqa: BLE001 - a chunk that fails is counted, the others go on
                flags["read_failed"] += len(ids)
                return len(ids)
            new_readings.extend(rows)
            flags["read_failed"] += sum(r.get("ocr_attempt") != "completed" for r in rows)
            if hosted and any("HostedDisabledError" in " ".join(r.get("issues") or []) for r in rows):
                flags["gate_off"] = True
            return len(ids)

        def tick_pass1(n: int) -> None:
            counters["done"] += n
            ctl.progress("reading", counters["done"], counters["total"], "reading spine groups")
        ctl.progress("reading", 0, total, "reading spine groups")
        _pool(read_chunk, [("a", todo1[i:i + CHUNK]) for i in range(0, total, CHUNK)], workers, lambda: ctl.stopped() or flags["gate_off"], tick_pass1)
        if ctl.stopped():
            return {**summary, "state": "interrupted"}
        # second sighting where the first read gave < 8 characters (the engine's pass 2), only for groups this step read
        first_text = {r["observation_id"]: GR.reading_text(r) for r in new_readings}
        have_obs = {r["observation_id"] for r in new_readings}
        weak = []
        for g in plan["groups"]:
            if g["status"] != "attempt" or len(g["sightings"]) < 2:
                continue
            f, sg = g["sightings"][0]["observation_id"], g["sightings"][1]
            if f in first_text and len(first_text[f]) < 8 and sg["observation_id"] not in pl["done"] and sg["observation_id"] not in have_obs:
                weak.append((g["instance_id"], sg))
        counters.update(done=0, total=len(weak))
        ctl.progress("reading second sightings", 0, len(weak), "second sightings of weak first reads")
        _pool(read_chunk, [("b", weak[i:i + CHUNK]) for i in range(0, len(weak), CHUNK)], workers, lambda: ctl.stopped() or flags["gate_off"], tick_pass1)
        if ctl.stopped():
            return {**summary, "state": "interrupted"}
    done_now = {r["observation_id"] for r in new_readings if r.get("ocr_attempt") == "completed"}
    # ---- resolve: the new readings and whatever the sweep's deadline left unresolved
    todo_res = sorted([r for r in new_readings if r.get("ocr_attempt") == "completed"] + need_res, key=lambda r: (_resolve_priority(r), r["observation_id"]))
    seen_ids: set[str] = set()
    todo_res = [r for r in todo_res if not (r["observation_id"] in seen_ids or seen_ids.add(r["observation_id"]))]
    (fr / "res").mkdir(exist_ok=True)
    rcounters = {"done": 0, "total": len(todo_res), "accepted": 0, "failed": 0}
    resolve_cfg = _resolve_config(stages, reader, cfg, run)
    run_id = pl["manifest_run_id"]

    def resolve_one(reading: dict[str, Any]) -> list[dict[str, Any]]:
        if ctl.stopped():
            return []
        oid = re.sub(r"[^A-Za-z0-9_.-]", "_", reading["observation_id"])
        out_path = fr / "res" / f"{oid}.resolutions.json"
        if out_path.is_file():
            with contextlib.suppress(OSError, ValueError, KeyError):
                return _load(out_path)["data"]["resolutions"]
        art = C.write_json(fr / "res" / f"{oid}.readings.json", C.envelope(run_id, "read", {"readings": [reading]}, status="ok"))
        rcfg = {**resolve_cfg, "session_revision": None, "resolutions_filename": f"{DIR}/res/{oid}.resolutions.json"}
        try:
            return C.read_json(stages.resolve(art, run, rcfg))["data"]["resolutions"]
        except Exception:  # noqa: BLE001 - recorded as failed; the reading stays unresolved
            rcounters["failed"] += 1
            return []
    new_res: list[dict[str, Any]] = []

    def tick_res(rows: list[dict[str, Any]]) -> None:
        new_res.extend(rows)
        rcounters["done"] += 1
        rcounters["accepted"] += sum(r.get("decision") == "accepted" for r in rows)
        ctl.progress("resolving", rcounters["done"], rcounters["total"], "matching readings against the catalogues")
    ctl.progress("resolving", 0, len(todo_res), "matching readings against the catalogues")
    _pool(resolve_one, todo_res, int(cfg.get("resolve_workers", os.environ.get("LIBRARY_RESOLVE_WORKERS", 2))), ctl.stopped, tick_res)
    if ctl.stopped():
        return {**summary, "state": "interrupted"}
    # ---- merge into the run and the packet (under the queue's packet lock, after the panorama / count / duplicate-merge chain settled)
    ctl.progress("waiting", 0, 0, "waiting for the panorama chain before updating the packet")
    while not ctl.settle():
        if ctl.stopped():
            return {**summary, "state": "interrupted"}
        time.sleep(2.0)
    from pipeline import identity_attach as IA
    attach_out: dict[str, Any] = {}
    with ctl.packet_lock:
        ctl.progress("updating the packet", 0, 0, "merging readings, identities and duplicate links")
        for name in ("readings.json", "resolutions.json", "inventory.json", "claim_packet.json"):
            _keep_once(run / name)
        rd = _load(run / "readings.json")
        have = {r["observation_id"] for r in new_readings}
        rd["data"]["readings"] = [r for r in rd["data"]["readings"] if r["observation_id"] not in have] + new_readings
        C.write_json(run / "readings.json", rd)
        rs = _load(run / "resolutions.json") if (run / "resolutions.json").is_file() else C.envelope(run_id, "resolve", {"resolutions": []}, status="ok")
        got = {r["observation_id"] for r in new_res}
        rs["data"]["resolutions"] = [r for r in rs["data"]["resolutions"] if r.get("observation_id") not in got] + new_res
        C.write_json(run / "resolutions.json", rs)
        rmap = {r["observation_id"]: r for r in rd["data"]["readings"]}
        res_map = {r["observation_id"]: r for r in rs["data"]["resolutions"]}
        inv = _load(run / "inventory.json")
        inv_stats = refresh_inventory(inv, rmap, res_map)
        C.write_json(run / "inventory.json", inv)
        pk = C.read_json(run / "claim_packet.json")
        summary["visible_text_updated"] = patch_visible_text(pk["data"], rmap)
        C.write_json(run / "claim_packet.json", pk)
        inst = None
        if ctl.instances is not None:
            try:
                inst = ctl.instances()
            except Exception as exc:  # noqa: BLE001 - the instance layer already on disk stands
                inst = {"state": "failed", "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        attach_out["identities"] = IA.attach(run, run / "readings.json", run / "resolutions.json", render=False)
        attach_out["instances"] = inst
    # ---- price the identified works (network; no packet writes), then attach under the lock
    price_out: dict[str, Any] = {}
    if price:
        from pipeline import price_attach as PAt
        from pipeline.price import price_config_from_env
        pc = price_config_from_env(run)
        ctl.progress("pricing", 0, 0, "pricing identified works")
        out_dir = fr / f"prices_{stamp}"
        try:
            price_out["compute"] = PAt.compute(run, run / "readings.json", out_dir, online=pc["pricing_online"], cache_dir=pc["price_cache_dir"], resolutions=run / "resolutions.json",
                                               converted=bool(pc.get("converted_fallback")), markets=",".join(pc.get("converted_markets") or []) or "ES,GB")
            with ctl.packet_lock:
                price_out["attach"] = PAt.attach(run, out_dir / "prices.json", render=False)
        except Exception as exc:  # noqa: BLE001 - identities stand; the price failure is reported
            price_out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        price_out["second_country"] = second_country_step(run, ctl)
    # ---- bookkeeping: stage_status, plan, provenance, report
    ctl.progress("finishing", 0, 0, "updating the report")
    left = len(plan_unread(run)["todo1"]) if todo1 else 0
    res_after = _load(run / "resolutions.json")["data"]["resolutions"]
    unresolved_after = len(unresolved_readings(_load(run / "readings.json")["data"]["readings"], res_after))
    after_usage = usage_summary(run)
    with ctl.packet_lock:
        after = update_bookkeeping(run, plan, todo1, left, unresolved_after, rcounters, flags, price_out, usage_before, after_usage, time.time() - t_start, attach_out)
        if render:
            from pipeline import report as R
            pk_path = run / "claim_packet.json"
            R.render(pk_path, run)
            if (run / "report.pdf").is_file():
                R.render_print_pdf(pk_path, run)
    unread_ids = {iid for iid, _ in todo1}
    groups_read = sum(1 for g in plan["groups"] if g["instance_id"] in unread_ids and any(x["observation_id"] in done_now for x in g["sightings"]))
    fallback = sum(1 for r in new_readings if any("hosted_error_fallback" in str(i) or "hosted_cap_reached" in str(i) for i in r.get("issues") or []))
    ident = attach_out.get("identities") or {}
    problems = []
    if flags["gate_off"]:
        problems.append("hosted reader switched off during the step")
    if left:
        problems.append(f"{left} attemptable groups still unread")
    if unresolved_after or rcounters["failed"]:
        problems.append(f"{unresolved_after} readings unresolved, {rcounters['failed']} resolve calls failed")
    if flags["read_failed"]:
        problems.append(f"{flags['read_failed']} read calls failed")
    if fallback:
        problems.append(f"{fallback} readings came from the free fallback reader (hosted error or cap)")
    if ident.get("attached") is False:
        problems.append("identities not attached: " + str(ident.get("reason")))
    if price_out.get("error") or (price and (price_out.get("attach") or {}).get("attached") is False):
        problems.append("prices not attached: " + str(price_out.get("error") or (price_out.get("attach") or {}).get("reason")))
    state = "partial" if problems else "done"
    reason = "; ".join(problems) or None
    summary.update(state=state, reason=reason, after=after, groups_read=groups_read, readings_new=len(new_readings), readings_failed=flags["read_failed"],
                   resolved=len(new_res), accepted_new=rcounters["accepted"], resolve_failed=rcounters["failed"], inventory=inv_stats, attach=attach_out, price=price_out,
                   usage={"fresh_requests": (after_usage["requests"] - usage_before["requests"]), "cache_hits": after_usage["cache_hits"] - usage_before["cache_hits"],
                          "tokens": {k: (after_usage[k] or 0) - (usage_before[k] or 0) for k in ("prompt_tokens", "output_tokens", "thought_tokens", "total_tokens")}},
                   finished_at=time.time(), elapsed_s=round(time.time() - t_start, 1))
    summary["summary"] = (f"{groups_read} of {len(todo1)} unread groups read, {rcounters['accepted']} new catalogue matches; "
                          f"books identified {before.get('books_identified')} -> {after.get('books_identified')}, priced {before.get('lines_priced')} -> {after.get('lines_priced')}")
    C.write_json(fr / "finish_reading.json", summary)
    return summary


def second_country_step(run: Path, ctl: Ctl, *, render: bool = False) -> dict[str, Any]:
    """The same 10 priced books priced for the second country (pipeline.second_country; LIBRARY_SECOND_COUNTRY, default GB). Network (marketplace pages, cache
    first) is capped (LIBRARY_SECOND_COUNTRY_MAX_REQUESTS) and runs outside the packet lock; the attach (finalize + verify + write) runs under it. Never raises."""
    from pipeline import second_country as SCN
    ctl.progress("second country", 0, 0, "pricing the same 10 books for the second country")
    try:
        return SCN.run_step(run, lock=ctl.packet_lock, render=render)
    except Exception as exc:  # noqa: BLE001 - the packet without the section stands; the failure is reported
        return {"state": "failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def _resolve_config(stages: Any, reader: tuple | None, cfg: dict[str, Any], run: Path) -> dict[str, Any]:
    if reader is not None:
        return {**reader[0].get("resolve", {}), **{k: cfg[k] for k in ("catalogue_online", "cache_dir") if k in cfg}}
    return {**cfg, "no_provider": True}


# ------------------------------------------------------------------------------------------------------------------------------ measurement of the run
def usage_summary(run: Path) -> dict[str, Any]:
    rows = []
    with contextlib.suppress(OSError):
        for line in (run / "logs" / "usage.jsonl").read_text().splitlines():
            with contextlib.suppress(ValueError):
                rows.append(json.loads(line))
    fresh = [r for r in rows if not r.get("cache_hit")]
    tok = lambda k: sum((r.get("usage") or {}).get(k) or 0 for r in fresh)
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    return {"requests": len(fresh), "cache_hits": len(rows) - len(fresh), "prompt_tokens": tok("prompt_tokens"), "output_tokens": tok("output_tokens"),
            "thought_tokens": tok("thought_tokens"), "total_tokens": tok("total_tokens"), "models": sorted({r.get("model") for r in rows if r.get("model")}),
            "cost_usd": round(sum(costs), 6) if costs else None}


def snapshot(run: Path) -> dict[str, Any]:
    """Identified / priced counts of the packet and the budget leftovers, for before/after."""
    try:
        d = _load(run / "claim_packet.json")["data"]
    except (OSError, ValueError, KeyError):
        return {}
    books = d.get("books") or []
    t = d.get("totals") or {}
    rows = (d.get("pricing") or {}).get("prices") or []
    return {"book_lines": t.get("book_count"), "books_identified": t.get("books_identified"), "lines_priced": sum(1 for b in books if b.get("replacement_cost") or b.get("used_value")),
            "books_replacement_cost": t.get("books_replacement_cost"), "books_used_value": t.get("books_used_value"),
            "visible_text_known": sum(1 for b in books if b.get("visible_text")),
            "price_budget_skipped": sum(1 for r in rows if any("budget" in str(i) for i in r.get("issues") or [])),
            "stage_status": {k: (d.get("stage_status") or {}).get(k, {}).get("status") for k in ("read", "resolve", "price")}}


def update_bookkeeping(run: Path, plan: dict[str, Any], todo1: list, left: int, unresolved: int, rc: dict[str, Any], flags: dict[str, Any], price_out: dict[str, Any],
                       usage_before: dict[str, Any], usage_after: dict[str, Any], elapsed: float, attach_out: dict[str, Any]) -> dict[str, Any]:
    """stage_status read / resolve / price (ok when nothing is left), provenance, the group-reading plan file; returns the packet's after-snapshot."""
    pk = C.read_json(run / "claim_packet.json")
    d = pk["data"]
    ss = d.setdefault("stage_status", {})
    prev = lambda k: float((ss.get(k) or {}).get("elapsed_s") or 0.0)
    snap = snapshot(run)
    read_ok = left == 0 and not flags["gate_off"] and not flags["read_failed"]
    ss["read"] = {"status": "ok" if read_ok else "partial", "elapsed_s": round(prev("read") + elapsed, 3),
                  "detail": (f"finish reading: {len(todo1)} groups the packet budget left unread were read after the sweep"
                             + ("" if read_ok else f"; {left} still unread, {flags['read_failed']} read calls failed" + ("; hosted reader switched off" if flags["gate_off"] else "")))}
    res_ok = unresolved == 0 and not rc["failed"]
    ss["resolve"] = {"status": "ok" if res_ok else "partial", "elapsed_s": round(prev("resolve"), 3),
                     "detail": f"finish reading: {rc['done']} readings resolved after the sweep ({rc['accepted']} accepted), {unresolved} still unresolved, {rc['failed']} failed"}
    price_skipped = snap.get("price_budget_skipped", 0)
    price_ok = not price_out.get("error") and price_skipped == 0 and "attach" in price_out and (price_out["attach"] or {}).get("attached", False)
    ss["price"] = {**(ss.get("price") or {}), "status": "ok" if price_ok else "partial",
                   "detail": ("Source-backed pricing: exact edition by printed ISBN, else work-level equivalent edition (labelled); finish reading priced the works found after the sweep"
                              + ("" if price_ok else f" (not complete: {price_out.get('error') or str((price_out.get('attach') or {}).get('reason') or price_skipped)})"))}
    d.setdefault("sweep", {})["finish_reading"] = {"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "groups_read": len(todo1) - left, "groups_still_unread": left,
                                                    "readings_unresolved": unresolved, "elapsed_s": round(elapsed, 1),
                                                    "note": "titles, identities and prices of groups the packet's read budget skipped, added after the packet by the post-sweep step"}
    prov = d.setdefault("provenance", {})
    if isinstance(prov, dict) and "fresh_requests" in prov:
        prov.update(fresh_requests=usage_after["requests"], cache_hits=usage_after["cache_hits"], models=usage_after["models"],
                    tokens={k: usage_after[k] for k in ("prompt_tokens", "output_tokens", "thought_tokens", "total_tokens")})
        rp = prov.get("read_policy")
        if isinstance(rp, dict):
            rp["finish_reading"] = {"groups_read": len(todo1) - left, "groups_still_unread": left}
            rp["groups_attempted"] = int(rp.get("groups_attempted") or 0) + len(todo1) - left
            if isinstance(rp.get("groups_unattempted"), dict):
                rp["groups_unattempted"]["budget"] = left
    from pipeline import packet as P
    P.finalize_contract(d, run)                  # the review queue's stage entries follow the stage_status set here (one entry per ref_id)
    C.write_json(run / "claim_packet.json", pk)
    plan_path = run / "group_reading_plan.json"
    if plan_path.is_file():
        with contextlib.suppress(OSError, ValueError, KeyError):
            pe = _load(plan_path)
            unread_now = {iid for iid, _ in plan_unread(run)["todo1"]}
            for g in pe["data"]["plan"]["groups"]:
                if g["status"] == "unattempted_budget" and g["instance_id"] not in unread_now:
                    g["status"], g["reason"] = "attempt", "read after the packet by the finish-reading step"
            rep = pe["data"]["report"]
            rep["finish_reading"] = {"groups_read": len(todo1) - left, "groups_still_unread": left, "readings_resolved": rc["done"]}
            if isinstance(rep.get("groups_unattempted"), dict):
                rep["groups_unattempted"]["budget"] = left
            rep["groups_attempted"] = sum(1 for g in pe["data"]["plan"]["groups"] if g["status"] == "attempt")
            C.write_json(plan_path, pe)
    return snapshot(run)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path, help="a finished run directory (use a scratch runs dir to keep the original untouched)")
    ap.add_argument("--dry-run", action="store_true", help="only count what is left to read / resolve")
    ap.add_argument("--no-price", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run:
        pl = plan_unread(a.run)
        print(json.dumps({"attemptable": pl["plan"]["counts"], "unread_groups": len(pl["todo1"]), "before": snapshot(a.run.resolve())}, indent=1))
        return 0
    from pipeline import instances as INST
    runs = a.run.resolve().parent
    pans = sorted(p.name for p in runs.glob(a.run.resolve().name + "_panorama_v*") if (p / "panorama.json").is_file())
    ctl = Ctl(progress=lambda phase, d, t, note="": print(f"[{phase}] {d}/{t} {note}", flush=True),
              instances=lambda: INST.run_step(runs, a.run.resolve().name, pans[-1] if pans else None))
    out = run(a.run, ctl, price=not a.no_price)
    print(json.dumps({k: v for k, v in out.items() if k != "attach"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
