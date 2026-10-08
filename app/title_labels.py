"""Human title labelling for spine crops of a finished run.

Samples spine crops (default 100, fixed seed, stratified by time bin x shelf-row band) from the EXACT run's own crops and stores the human's
literal title/author/language as ``purpose: evaluation_truth`` under ``eval/truth_capture/`` (outside run dirs). The pipeline, engine and
packet never read these files. The model's reading is deliberately NOT shown while labelling (no anchoring). Each label stores the crop's
sha256; resume and undo are supported; ``not_a_book`` / ``duplicate_of_previous`` marks feed a later count truth.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from app import truth as TR

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DEFAULT_N, DEFAULT_SEED = 100, 20261006
STATUSES = {"labeled", "partial", "unreadable", "not_a_book", "duplicate_of_previous"}   # partial = cut-off/fragment: literal text kept, never a full title
LANGS = {"es", "en", "other"}
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
TIME_BINS, ROW_BANDS = 6, 3
PREFERRED_DEFAULTS = {"YugoTest2": "final_candidate_yt2_v1", "YugoTest1": "final_candidate_yt1_v2"}   # YugoTest2 is the PRIMARY scan (closer, legible)


class LabelError(ValueError):
    pass


def _read(p: Path) -> Any:
    return json.loads(p.read_text())


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def registry_by_sha() -> dict[str, dict[str, Any]]:
    try:
        from app.debug_registry import read_registry
        return {s["sha256"]: s for s in read_registry(ROOT)}
    except Exception:  # noqa: BLE001 - registry advisory here; runs just show no scan label
        return {}


def run_dir(run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id or ""):
        raise LabelError("invalid run id")
    p = (RUNS / run_id).resolve()
    if p.parent != RUNS.resolve() or not p.is_dir() or p.is_symlink():
        raise LabelError("unknown run")
    return p


def list_runs() -> list[dict[str, Any]]:
    """Finished runs that have exact crops, with the scan they belong to. One default per scan (most crops, then newest)."""
    reg = registry_by_sha()
    out = []
    for p in sorted(RUNS.iterdir()):
        try:
            if not (p.is_dir() and (p / "readings.json").is_file() and (p / "manifest.json").is_file() and (p / "candidates.json").is_file() and (p / "crops").is_dir()):
                continue
            m = _read(p / "manifest.json")
            sha = (m.get("data", {}).get("input") or {}).get("sha256")
            r = _read(p / "readings.json")["data"]["readings"]
            crops = sum(1 for x in r if x.get("crop_path"))
            if not crops:
                continue
            scan, inferred = (reg.get(sha) or {}).get("label"), False
            if scan is None and sha is None:   # older replays did not bind a scan: infer from video duration, and say so
                dur = max((f["timestamp_s"] for f in m["data"]["frames"]), default=0)
                scan = "YugoTest1" if 18.5 <= dur <= 19.6 else "YugoTest2" if 68.5 <= dur <= 70.5 else None
                inferred = scan is not None
            out.append({"run_id": p.name, "scan": scan, "scan_inferred_from_duration": inferred, "sha256": sha, "crops": crops,
                        "frames": len(m["data"]["frames"]), "mtime": p.stat().st_mtime})
        except (OSError, ValueError, KeyError):
            continue
    best: dict[str | None, dict[str, Any]] = {}
    for r in out:
        cur = best.get(r["scan"])
        if any(w in r["run_id"] for w in ("review", "diagnostic", "smoke", "sample", "negative")):
            continue   # never auto-elected as the default run
        if cur is None or (r["crops"], r["mtime"]) > (cur["crops"], cur["mtime"]):
            best[r["scan"]] = r
    for scan, rid in PREFERRED_DEFAULTS.items():       # explicit agreed defaults win when the run exists
        pick = next((r for r in out if r["run_id"] == rid and r["scan"] == scan), None)
        if pick:
            best[scan] = pick
    for r in out:
        r["default_for_scan"] = best.get(r["scan"]) is r and r["scan"] is not None
        r.pop("mtime")
    return sorted(out, key=lambda r: (r["scan"] or "~", not r["default_for_scan"], r["run_id"]))


def sample_run(run_id: str, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Deterministic stratified sample of distinct proposed-group representatives that have an exact crop."""
    p = run_dir(run_id)
    if not 1 <= n <= 400:
        raise LabelError("n must be 1-400")
    man = _read(p / "manifest.json")["data"]
    frames = {f["frame_id"]: f for f in man["frames"]}
    readings = {x["observation_id"]: x for x in _read(p / "readings.json")["data"]["readings"] if x.get("crop_path")} if (p / "readings.json").is_file() else {}
    cands = {o["observation_id"]: o for o in _read(p / "candidates.json")["data"]["observations"] if o.get("kind") == "book"}
    inv = _read(p / "inventory.json")["data"] if (p / "inventory.json").is_file() else {"books": []}
    groups = inv.get("books", [])
    pool = []
    others: dict[str, list[dict[str, Any]]] = {}
    if groups:
        from app.group_reader import sighting_score      # best-view sighting = widest x sharpest (what the readers get), not the first one
        for g in groups:
            seen = [o for o in g["observation_ids"] if o in cands and cands[o]["frame_id"] in frames]
            if seen:
                ranked = sorted(seen, key=lambda o: sighting_score(cands[o], frames[cands[o]["frame_id"]]), reverse=True)
                best = ranked[0]
                others[best] = [{"observation_id": o, "frame_id": cands[o]["frame_id"], "bbox_xyxy": cands[o]["bbox_xyxy"],
                                 "frame_sha256": frames[cands[o]["frame_id"]].get("sha256") or (_sha(p / frames[cands[o]["frame_id"]]["image_path"]) if (p / frames[cands[o]["frame_id"]]["image_path"]).is_file() else None)}
                                for o in ranked[1:3]]
                if best not in readings:                   # unread group (reading budget): extract its exact native bbox crop once, no OCR
                    extracted = extract_native_crop(p, cands[best], frames[cands[best]["frame_id"]])
                    if extracted is None:
                        continue
                    readings[best] = {"observation_id": best, "crop_path": extracted, "crop_source": "label_extract_native_bbox"}
                pool.append((best, g["instance_id"]))
    else:  # no association artifact: every observation is its own representative
        pool = [(o, None) for o in cands if o in readings]
    if not pool:
        raise LabelError("this run has no book crops to label")
    duration = max((f["timestamp_s"] for f in frames.values()), default=1.0) or 1.0
    strata: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for oid, gid in sorted(pool):
        o, f = cands[oid], frames[cands[oid]["frame_id"]]
        tb = min(TIME_BINS - 1, int(TIME_BINS * f["timestamp_s"] / (duration + 1e-9)))
        cy = (o["bbox_xyxy"][1] + o["bbox_xyxy"][3]) / 2 / f["height_px"]
        rb = min(ROW_BANDS - 1, int(ROW_BANDS * cy))
        crop = p / readings[oid]["crop_path"]
        if not crop.is_file() or crop.is_symlink() or not crop.resolve().is_relative_to(p.resolve()):
            continue
        strata.setdefault((tb, rb), []).append({"observation_id": oid, "group_id": gid, "frame_id": o["frame_id"], "timestamp_s": f["timestamp_s"],
                                                "bbox_xyxy": o["bbox_xyxy"], "crop_path": readings[oid]["crop_path"], "stratum": f"t{tb}r{rb}",
                                                "frame_sha256": f.get("sha256") or (_sha(p / f["image_path"]) if (p / f["image_path"]).is_file() else None),
                                                "crop_source": readings[oid].get("crop_source", "reader_crop"), "other_sightings": others.get(oid, [])})
    rng = random.Random(seed)
    for k in strata:
        rng.shuffle(strata[k])
    order, keys = [], sorted(strata)
    while len(order) < n and any(strata[k] for k in keys):          # round-robin across strata: equal share, then remainder
        for k in keys:
            if strata[k] and len(order) < n:
                order.append(strata[k].pop())
    rng.shuffle(order)
    for i, it in enumerate(order):
        it["index"] = i
        it["crop_sha256"] = _sha(p / it["crop_path"])
    return {"items": order, "strata_sizes": {f"t{a}r{b}": 0 for a, b in keys}, "pool": len(pool),
            "representative": "best-view sighting per proposed group: max(spine_width_px x frame sharpness, edge-touching x0.25)"}


def extract_native_crop(run: Path, obs: dict[str, Any], frame: dict[str, Any]) -> str | None:
    """Exact pixel crop of the observation's bbox from the saved native frame (no resampling, no OCR), written once to
    <run>/crops_labeling/<observation_id>.png so it can be hashed like a reader crop. Returns the run-relative path, or None if the frame is unreadable."""
    import cv2
    out = run / "crops_labeling" / f"{obs['observation_id']}.png"
    if out.is_file():
        return str(out.relative_to(run))
    img = cv2.imread(str(run / frame["image_path"]), cv2.IMREAD_COLOR)
    if img is None:
        return None
    x0, y0, x1, y1 = (int(round(v)) for v in obs["bbox_xyxy"])
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), img[y0:y1, x0:x1])
    return str(out.relative_to(run))


def label_id(run_id: str, n: int, seed: int) -> str:
    return f"titles_{run_id}_n{n}_seed{seed}"


def _path(lid: str) -> Path:
    if not re.fullmatch(r"titles_[A-Za-z0-9_.-]{1,160}", lid or ""):
        raise LabelError("invalid label id")
    return TR.TRUTH_DIR / f"{lid}.json"


def _save(lid: str, doc: dict[str, Any]) -> None:
    p = _path(lid)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def open_session(run_id: str, annotator: str, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Create or resume the labelling file for (run, n, seed)."""
    if not annotator or not annotator.strip():
        raise LabelError("annotator name is required")
    lid = label_id(run_id, n, seed)
    p = _path(lid)
    if p.is_file():
        doc = _read(p)
        # resume only if the sampled crops are still byte-identical
        run = run_dir(run_id)
        changed = [i["observation_id"] for i in doc["sample"] if not (run / i["crop_path"]).is_file() or _sha(run / i["crop_path"]) != i["crop_sha256"]]
        if changed:
            raise LabelError(f"{len(changed)} sampled crop(s) changed since labelling started; start a new sample (different seed)")
        return state(lid)
    s = sample_run(run_id, n, seed)
    reg = {r["run_id"]: r for r in list_runs()}.get(run_id, {})
    p_run = run_dir(run_id)
    doc = {"schema_version": 1, "purpose": "evaluation_truth", "kind": "title_labels", "label_id": lid, "run_id": run_id,
           "scan": {"label": reg.get("scan"), "sha256": reg.get("sha256")}, "annotator": annotator.strip()[:80], "seed": seed, "n_requested": n,
           "n_sampled": len(s["items"]), "stratification": f"{TIME_BINS} time bins x {ROW_BANDS} shelf-row bands, round-robin, seeded",
           "source_hashes": {"candidates_sha256": _sha(p_run / "candidates.json"), "readings_sha256": _sha(p_run / "readings.json"),
                             "manifest_sha256": _sha(p_run / "manifest.json")},
           "representative": s.get("representative"), "preview": "labelling preview = in-memory x~108px-short-side upscale + median-border pad (same as the readers get); hashes are of the NATIVE crop",
           "note": "HELD-OUT human title truth. Never read by the pipeline/engine/packet. Model readings are not shown while labelling.",
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sample": s["items"], "labels": {}, "ops": []}
    _save(lid, doc)
    return state(lid)


_PREFILL_CACHE: dict[str, dict[str, Any]] = {}


def prefill_map(run_id: str) -> dict[str, dict[str, Any]]:
    """observation_id -> {title, author, source} from the run's own readings (primary, second-pass, group union) and any model-read DEV label files
    (eval/dev_labels/*.json) whose run matches. Model text, shown to the annotator as a starting point; the saved label records prefilled/edited."""
    if run_id in _PREFILL_CACHE:
        return _PREFILL_CACHE[run_id]
    p = run_dir(run_id)
    out: dict[str, dict[str, Any]] = {}
    def put(oid: str, title: str | None, author: str | None, source: str, prio: int) -> None:
        t = " ".join(str(title or "").split())          # keep "[?]" and other uncertainty markers visible: the annotator must see them
        if not t or t == "[?]":
            return
        cur = out.get(oid)
        if cur is None or prio > cur["_prio"] or (prio == cur["_prio"] and len(t) > len(cur["title"])):
            out[oid] = {"title": t[:200], "author": (" ".join(str(author or "").split()) or None), "source": source, "_prio": prio}
    for name, src, prio in (("readings.json", "reader", 1), ("readings_second.json", "second_reader", 2)):
        f = p / name
        if f.is_file():
            for r in _read(f)["data"].get("readings", []):
                put(r["observation_id"], r.get("title_text") or r.get("visible_text"), r.get("author_text"), src, prio)
    if (p / "inventory.json").is_file():
        for g in _read(p / "inventory.json")["data"].get("books", []):
            ident = g.get("identity") or {}
            for oid in g.get("observation_ids", []):
                if oid not in out:
                    put(oid, ident.get("visible_text"), None, "group_best_reading", 0)
    dev_dir = ROOT / "eval" / "dev_labels"
    for f in sorted(dev_dir.glob("*.json")) if dev_dir.is_dir() else []:
        try:
            d = _read(f)
        except (OSError, ValueError):
            continue
        if d.get("purpose") == "evaluation_truth" or str(d.get("run", "")).rstrip("/").split("/")[-1] != run_id:
            continue
        for l in d.get("labels", []):
            put(l.get("observation_id", ""), l.get("title"), l.get("author"), f"dev_label:{f.name} (model-read, not truth)", 3)
    for v in out.values():
        v.pop("_prio", None)
    _PREFILL_CACHE[run_id] = out
    return out


def state(lid: str) -> dict[str, Any]:
    doc = _read(_path(lid))
    labels = doc["labels"]
    pre = prefill_map(doc["run_id"])
    def item_prefill(i: dict[str, Any]) -> dict[str, Any] | None:
        for oid in (i["observation_id"], *[o["observation_id"] for o in i.get("other_sightings", [])]):
            if oid in pre:
                return pre[oid]
        return None
    items = [{**{k: i[k] for k in ("index", "observation_id", "group_id", "frame_id", "stratum")}, "other_sightings": len(i.get("other_sightings", [])),
              "prefill": item_prefill(i), "label": labels.get(i["observation_id"])} for i in doc["sample"]]
    counts = {k: sum(1 for l in labels.values() if l["status"] == k) for k in STATUSES}
    done = len(labels)
    nxt = next((it["index"] for it in items if it["label"] is None), None)
    return {"label_id": lid, "run_id": doc["run_id"], "scan": doc["scan"], "annotator": doc["annotator"], "seed": doc["seed"], "n": len(items),
            "done": done, "remaining": len(items) - done, "next_index": nxt, "counts": counts, "items": items,
            "saved_to": f"eval/truth_capture/{lid}.json", "can_undo": bool(doc["ops"])}


def answer(lid: str, observation_id: str, status: str, *, title: str = "", author: str = "", language: str | None = None,
           annotator: str | None = None, rotation_seen: int = 0, prefill: dict[str, Any] | None = None) -> dict[str, Any]:
    doc = _read(_path(lid))
    _refuse_if_frozen(doc)
    if observation_id not in {i["observation_id"] for i in doc["sample"]}:
        raise LabelError("not in this sample")
    if status not in STATUSES:
        raise LabelError(f"status must be one of {sorted(STATUSES)}")
    entry: dict[str, Any] = {"status": status, "annotator": (annotator or doc["annotator"]).strip()[:80], "rotation_seen_deg": int(rotation_seen) % 360,
                             "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if status in ("labeled", "partial"):
        t = " ".join(str(title).split())
        if not t:
            raise LabelError("type the title exactly as printed on the spine (or the visible fragment for partial), or mark it unreadable")
        if len(t) > 200 or len(str(author)) > 120:
            raise LabelError("title/author too long")
        if language not in LANGS:
            raise LabelError("choose a language: es, en or other")
        entry.update(title=t, author=" ".join(str(author).split()) or None, language=language, human_confirmed=True,
                     full_title=(status == "labeled"))
        if status == "partial":
            entry["partial_note"] = "cut-off / fragment: literal visible text only, NOT a full title; excluded from full-title denominators"
        if prefill and prefill.get("title"):
            entry.update(prefilled=True, prefill_source=str(prefill.get("source", ""))[:120], prefill_title=str(prefill["title"])[:200],
                         edited=(t != " ".join(str(prefill["title"]).split())) or ((entry["author"] or "") != " ".join(str(prefill.get("author") or "").split())),
                         confirmed_by=entry["annotator"], confirmed_at=entry["labeled_at"])
        else:
            entry.update(prefilled=False, edited=True)
    doc["ops"].append({"observation_id": observation_id, "previous": doc["labels"].get(observation_id)})
    doc["labels"][observation_id] = entry
    _save(lid, doc)
    return state(lid)


def _refuse_if_frozen(doc: dict[str, Any]) -> None:
    if doc.get("frozen") or (doc.get("provenance") or {}).get("frozen_at"):
        raise LabelError("this label file is frozen (a scored version); it cannot be edited. Start a new sample (different seed) to label again")


def undo(lid: str) -> dict[str, Any]:
    doc = _read(_path(lid))
    _refuse_if_frozen(doc)
    if not doc["ops"]:
        raise LabelError("nothing to undo")
    op = doc["ops"].pop()
    if op["previous"] is None:
        doc["labels"].pop(op["observation_id"], None)
    else:
        doc["labels"][op["observation_id"]] = op["previous"]
    _save(lid, doc)
    return {**state(lid), "undone_observation_id": op["observation_id"]}


def crop_file(lid: str, index: int) -> Path:
    doc = _read(_path(lid))
    item = next((i for i in doc["sample"] if i["index"] == index), None)
    if item is None:
        raise LabelError("no such crop")
    run = run_dir(doc["run_id"])
    f = run / item["crop_path"]
    if not f.is_file() or f.is_symlink() or not f.resolve().is_relative_to(run.resolve()) or _sha(f) != item["crop_sha256"]:
        raise LabelError("crop missing or changed since sampling")
    return f


def crop_preview_png(lid: str, index: int) -> bytes:
    """The crop as the readers see it (in memory; the native file is untouched): same preprocess_strip recipe, encoded as PNG."""
    import cv2
    from providers.readers import preprocess_strip
    img = cv2.imread(str(crop_file(lid, index)), cv2.IMREAD_COLOR)
    if img is None:
        raise LabelError("crop unreadable")
    out, _ = preprocess_strip(img)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise LabelError("preview encoding failed")
    return bytes(buf)


def _verify_frame(run: Path, frame: dict[str, Any], expected: str | None) -> None:
    """Context/other-sighting pixels come from the saved frame: refuse if it changed since the sample pinned it (or since the manifest recorded it)."""
    f = run / frame["image_path"]
    if not f.is_file():
        raise LabelError("frame missing")
    want = expected or frame.get("sha256")
    if want and _sha(f) != want:
        raise LabelError("frame changed since labelling started: context cannot be served from unverified pixels")


def _frame_crop(run: Path, frame: dict[str, Any], bbox: list[float], widen_along: float = 0.0):
    """In-memory crop of a bbox from the saved native frame; widen_along grows the LONG axis of the box by that fraction (both ends) for context."""
    import cv2
    img = cv2.imread(str(run / frame["image_path"]), cv2.IMREAD_COLOR)
    if img is None:
        raise LabelError("frame unreadable")
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if widen_along:
        if (y1 - y0) >= (x1 - x0):
            d = (y1 - y0) * widen_along; y0 -= d; y1 += d
        else:
            d = (x1 - x0) * widen_along; x0 -= d; x1 += d
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise LabelError("degenerate crop")
    return img[y0:y1, x0:x1]


def _item(lid: str, index: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    doc = _read(_path(lid))
    item = next((i for i in doc["sample"] if i["index"] == index), None)
    if item is None:
        raise LabelError("no such crop")
    return doc, item, run_dir(doc["run_id"])


def _png(img) -> bytes:
    import cv2
    from providers.readers import preprocess_strip
    out, _ = preprocess_strip(img)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise LabelError("encoding failed")
    return bytes(buf)


def context_png(lid: str, index: int, widen: float = 0.3) -> bytes:
    """Best-view sighting widened 30% along the spine (shows title text the detector box cut off), reader-style preview."""
    doc, item, run = _item(lid, index)
    frames = {f["frame_id"]: f for f in _read(run / "manifest.json")["data"]["frames"]}
    _verify_frame(run, frames[item["frame_id"]], item.get("frame_sha256"))
    return _png(_frame_crop(run, frames[item["frame_id"]], item["bbox_xyxy"], widen))


def sighting_png(lid: str, index: int, k: int) -> bytes:
    """k-th other top sighting of the same group (0-based), reader-style preview, in memory."""
    doc, item, run = _item(lid, index)
    others = item.get("other_sightings", [])
    if not 0 <= k < len(others):
        raise LabelError("no such sighting")
    frames = {f["frame_id"]: f for f in _read(run / "manifest.json")["data"]["frames"]}
    o = others[k]
    _verify_frame(run, frames[o["frame_id"]], o.get("frame_sha256"))
    return _png(_frame_crop(run, frames[o["frame_id"]], o["bbox_xyxy"], 0.3))
