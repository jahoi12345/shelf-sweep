"""Offline, blinded duplicate-pair evidence. Never consumes adjudication answers.

Inputs are saved run artifacts; outputs live only in runs/pair_sheets/<run>.
One pair per unordered inventory group pair, with all proposal provenance kept.
Control expectations are association/exclusion checks, never human truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DEFAULT_RUN = "sweep_3af346a2f3f14f72"
DEFAULT_RUNS = (DEFAULT_RUN, "sweep_92ea122a48394471", "sweep_059fa9973f7e4612")
DEFAULT_SEED = 20261007
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
CONFIRMED_ANCHORS = (("f00627-o006", "f00633-o000"), ("f00627-o001", "f00633-o008"))
SIZE = (1080, 420)


class PairError(ValueError):
    pass


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_dir(run_id: str, runs: Path | None = None) -> Path:
    runs = runs or RUNS
    if not RUN_ID.fullmatch(run_id or ""):
        raise PairError("invalid run id")
    p = runs / run_id
    if p.is_symlink() or not p.is_dir() or p.resolve().parent != runs.resolve():
        raise PairError("unknown run")
    return p


def safe_file(root: Path, relative: str) -> Path:
    p = root / relative
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root.resolve()):
        raise PairError("missing or unsafe evidence file")
    return p


def _data(path: Path):
    d = read(path)
    return d.get("data", d) if isinstance(d, dict) else d


def _rank(o, frames):
    x0, y0, x1, y1 = o["bbox_xyxy"]
    f = frames[o["frame_id"]]
    score = min(x1 - x0, y1 - y0) * float((f.get("quality") or {}).get("sharpness", 1))
    if x0 < 2 or y0 < 2 or x1 > f["width_px"] - 2 or y1 > f["height_px"] - 2:
        score *= .25
    return score


def candidates(run_id: str, *, runs: Path | None = None, seed: int = DEFAULT_SEED):
    """Seeded union + ~20% different / ~10% same controls (of final set)."""
    runs = runs or RUNS
    run = run_dir(run_id, runs)
    frames = {f["frame_id"]: f for f in _data(run / "manifest.json")["frames"]}
    obs = {o["observation_id"]: o for o in _data(run / "candidates.json")["observations"]
           if o.get("kind") == "book" and o["frame_id"] in frames}
    inv = _data(run / "inventory.json")
    groups = {g["instance_id"]: g for g in inv["books"]}
    group_of = {oid: gid for gid, g in groups.items() for oid in g["observation_ids"] if oid in obs}
    ranked = {gid: sorted((oid for oid in g["observation_ids"] if oid in obs),
                          key=lambda oid: (-_rank(obs[oid], frames), oid)) for gid, g in groups.items()}
    pairs, skipped = {}, Counter()
    inputs = [run / n for n in ("manifest.json", "candidates.json", "inventory.json")]

    def add(ids, source, evidence):
        if len(ids) != 2 or any(oid not in group_of for oid in ids):
            skipped[source] += 1
            return
        gids = [group_of[oid] for oid in ids]
        if gids[0] == gids[1]:
            skipped[source] += 1
            return
        key = tuple(sorted(gids))
        # Audit anchors are added first, so their exact observation ids survive.
        p = pairs.setdefault(key, {"observation_ids": list(ids), "groups": gids, "source_kinds": [], "provenance": []})
        if source not in p["source_kinds"]:
            p["source_kinds"].append(source)
        p["provenance"].append({"source_kind": source, **evidence})

    audit = runs / "live_count_diagnosis_2026-10-07" / "YugoTest11_break_duplicate_edges.json"
    if run_id == DEFAULT_RUN and audit.is_file():
        inputs.append(audit)
        edges = _data(audit)
        anchors = {frozenset(x) for x in CONFIRMED_ANCHORS}
        edges = sorted(edges, key=lambda e: frozenset(e["observation_ids"]) not in anchors)
        for e in edges:
            add(e["observation_ids"], "audit_break", {"file": str(audit.relative_to(runs)), "observation_ids": e["observation_ids"]})
    for pan in sorted(runs.glob(f"{run_id}_panorama_v*/panorama_count.json")):
        if pan.is_symlink() or not pan.resolve().is_relative_to(runs.resolve()):
            raise PairError("unsafe panorama file")
        inputs.append(pan)
        for e in _data(pan).get("merge_evidence", []):
            gids = sorted(set(g["id"] if isinstance(g, dict) else g for g in e["groups"]))
            for a, b in itertools.combinations(gids, 2):
                if not ranked.get(a) or not ranked.get(b):
                    skipped["panorama_merge"] += 1
                    continue
                add([ranked[a][0], ranked[b][0]], "panorama_merge",
                    {"file": str(pan.relative_to(runs)), "panorama_spine": e.get("panorama_spine"), "groups": [a, b]})
    for e in inv.get("uncertain_associations", []):
        add(e.get("observation_ids", []), "uncertain_association",
            {"file": f"{run_id}/inventory.json", "observation_ids": e.get("observation_ids"), "reason": e.get("reason")})

    rng = random.Random(seed)
    real = list(pairs.values())
    # N = R/.7, so .2N different controls and .1N same controls.
    n_different, n_same = round(len(real) * 2 / 7), round(len(real) / 7)
    by_frame = defaultdict(list)
    for oid, o in obs.items():
        if oid in group_of:
            by_frame[o["frame_id"]].append(oid)
    different = {}
    for fid, ids in sorted(by_frame.items()):
        # Same-row, adjacent and non-overlapping detections. Exclusion is a
        # structural control; it is not an independently human-verified label.
        ids.sort(key=lambda oid: sum(obs[oid]["bbox_xyxy"][::2]))
        for a, b in zip(ids, ids[1:]):
            ga, gb = group_of[a], group_of[b]
            if ga == gb:
                continue
            ax0, ay0, ax1, ay1 = obs[a]["bbox_xyxy"]
            bx0, by0, bx1, by1 = obs[b]["bbox_xyxy"]
            overlap = min(ay1, by1) - max(ay0, by0)
            if ax1 > bx0 or overlap < .6 * min(ay1 - ay0, by1 - by0):
                continue
            key = tuple(sorted((ga, gb)))
            if key in pairs or key in different:
                continue
            # Pick separate-frame sightings, if possible, to avoid advertising
            # a control through identical frame ids on the image.
            choices = [(x, y) for x in ranked[ga][:5] for y in ranked[gb][:5]
                       if obs[x]["frame_id"] != obs[y]["frame_id"]]
            if not choices:
                continue
            x, y = choices[0]
            different[key] = {"observation_ids": [x, y], "groups": [ga, gb], "source_kinds": ["control"],
                              "control_kind": "different", "provenance": [{"source_kind": "control", "basis": "adjacent_same_frame_exclusion",
                                                                           "witness_observation_ids": [a, b], "witness_frame_id": fid}]}
    same = []
    for gid, ids in sorted(ranked.items()):
        if len(ids) < 2:
            continue
        a = ids[0]
        b = next((oid for oid in ids[1:] if obs[oid]["frame_id"] != obs[a]["frame_id"]), None)
        if b:
            same.append({"observation_ids": [a, b], "groups": [gid, gid], "source_kinds": ["control"], "control_kind": "same",
                         "provenance": [{"source_kind": "control", "basis": "two_sightings_of_one_proposed_group"}]})
    different = list(different.values())
    rng.shuffle(different)
    rng.shuffle(same)
    selected = real + different[:n_different] + same[:n_same]
    # Random order and side, common id space and rendering for every source.
    rng.shuffle(selected)
    for i, p in enumerate(selected, 1):
        if rng.random() < .5:
            p["observation_ids"].reverse()
            p["groups"].reverse()
        p["pair_id"] = f"pair-{i:05d}"
        p["source_kind"] = p["source_kinds"][0]
    return {"pairs": selected, "observations": obs, "frames": frames, "source_files": inputs,
            "skipped": dict(skipped), "controls_requested": {"different": n_different, "same": n_same},
            "controls_available": {"different": len(different), "same": len(same)}}


def _context(image, o, frame_obs):
    x0, y0, x1, y1 = o["bbox_xyxy"]
    vertical = y1 - y0 >= x1 - x0
    along = 0 if vertical else 1
    cross = 1 - along
    bbox = o["bbox_xyxy"]
    neighbours = []
    for other in frame_obs:
        b = other["bbox_xyxy"]
        overlap = min(b[cross + 2], bbox[cross + 2]) - max(b[cross], bbox[cross])
        if overlap >= .5 * min(b[cross + 2] - b[cross], bbox[cross + 2] - bbox[cross]):
            neighbours.append(other)
    neighbours.sort(key=lambda it: it["bbox_xyxy"][along] + it["bbox_xyxy"][along + 2])
    idx = next(i for i, it in enumerate(neighbours) if it["observation_id"] == o["observation_id"])
    chosen = neighbours[max(0, idx - 3):idx + 4]
    boxes = np.asarray([n["bbox_xyxy"] for n in chosen])
    pad = max(4, int(min(x1 - x0, y1 - y0) * .3))
    h, w = image.shape[:2]
    left, top = max(0, int(boxes[:, 0].min()) - pad), max(0, int(boxes[:, 1].min()) - pad)
    right, bottom = min(w, math.ceil(boxes[:, 2].max()) + pad), min(h, math.ceil(boxes[:, 3].max()) + pad)
    out = image[top:bottom, left:right].copy()
    polygon = o.get("polygon_xy") or [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    points = np.rint(np.asarray(polygon) - [left, top]).astype(np.int32)
    cv2.polylines(out, [points], True, (0, 230, 255), max(2, round(out.shape[0] / 180)))
    return out, [left, top, right, bottom]


def _fit(canvas, img, box):
    x, y, w, h = box
    # Never invent resolution in a small source crop.
    s = min(1., w / img.shape[1], h / img.shape[0])
    out = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    yy, xx = y + (h - out.shape[0]) // 2, x + (w - out.shape[1]) // 2
    canvas[yy:yy + out.shape[0], xx:xx + out.shape[1]] = out


def _text(canvas, text, xy, size=.6):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, (35, 35, 35), 1, cv2.LINE_AA)


def generate(run_id: str = DEFAULT_RUN, *, runs: Path | None = None, seed: int = DEFAULT_SEED):
    runs = runs or RUNS
    run = run_dir(run_id, runs)
    dest = runs / "pair_sheets" / run_id
    if dest.exists():
        raise PairError("sheets already exist; preserve them and their labels (do not regenerate in place)")
    if dest.parent.is_symlink():
        raise PairError("unsafe output directory")
    selection = candidates(run_id, runs=runs, seed=seed)
    if not selection["pairs"]:
        raise PairError("no candidate pairs found")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=dest.parent))
    try:
        for sub in ("sheets", "native_crops", "contact_sheets"):
            (staging / sub).mkdir()
        obs, frames = selection["observations"], selection["frames"]
        by_frame = defaultdict(list)
        for o in obs.values():
            by_frame[o["frame_id"]].append(o)

        @lru_cache(maxsize=16)
        def load(fid):
            path = safe_file(run, frames[fid]["image_path"])
            image = cv2.imread(str(path))
            if image is None:
                raise PairError(f"unreadable frame: {fid}")
            return image, sha(path)

        @lru_cache(maxsize=128)
        def sighting(oid):
            from pipeline.read_spines import native_crop
            o = obs[oid]
            img, digest = load(o["frame_id"])
            crop, _ = native_crop(img, o)
            if crop.size == 0:
                raise PairError(f"empty crop: {oid}")
            rotation = 0
            if crop.shape[1] > crop.shape[0]:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                rotation = 90
            path = f"native_crops/{oid}.png"
            if not cv2.imwrite(str(staging / path), crop):
                raise PairError("native crop write failed")
            context, context_box = _context(img, o, by_frame[o["frame_id"]])
            return crop, context, {"observation_id": oid, "frame_id": o["frame_id"], "timestamp_s": frames[o["frame_id"]]["timestamp_s"],
                                   "bbox_xyxy": o["bbox_xyxy"], "frame_sha256": digest, "crop_path": path,
                                   "crop_sha256": sha(staging / path), "rotation_deg": rotation,
                                   "native_crop_size": [crop.shape[1], crop.shape[0]], "context_bbox_xyxy": context_box}

        public, key, page_ids, page = [], {}, [], None
        contacts = []
        for i, p in enumerate(selection["pairs"]):
            sheet = np.full((SIZE[1], SIZE[0], 3), 246, np.uint8)
            _text(sheet, p["pair_id"] + "  |  Same physical book?", (20, 27), .7)
            sheet[40:402, 539:541] = 210
            sightings = []
            for side, oid in enumerate(p["observation_ids"]):
                crop, context, meta = sighting(oid)
                sightings.append(meta)
                left = side * 540
                _text(sheet, f"{'A' if side == 0 else 'B'}  {meta['frame_id']}  {meta['timestamp_s']:.3f} s", (left + 16, 62))
                _fit(sheet, crop, (left + 12, 78, 130, 300))
                _fit(sheet, context, (left + 150, 78, 375, 300))
                _text(sheet, "Spine", (left + 45, 398), .45)
                _text(sheet, "Frame context (outlined spine)", (left + 160, 398), .45)
            sheet_path = f"sheets/{p['pair_id']}.png"
            if not cv2.imwrite(str(staging / sheet_path), sheet):
                raise PairError("sheet write failed")
            public.append({k: p[k] for k in ("pair_id", "observation_ids", "groups", "source_kind", "source_kinds")} |
                          {"sheet_path": sheet_path, "sheet_sha256": sha(staging / sheet_path), "sightings": sightings})
            key[p["pair_id"]] = {"control_kind": p.get("control_kind"), "expected_answer": p.get("control_kind"), "provenance": p["provenance"]}
            if i % 12 == 0:
                page = np.full((SIZE[1] * 4, SIZE[0] * 3, 3), 246, np.uint8)
                page_ids = []
            row, col = divmod(i % 12, 3)
            page[row * SIZE[1]:(row + 1) * SIZE[1], col * SIZE[0]:(col + 1) * SIZE[0]] = sheet
            page_ids.append(p["pair_id"])
            if i % 12 == 11 or i == len(selection["pairs"]) - 1:
                path = f"contact_sheets/page-{i // 12 + 1:03d}.png"
                if not cv2.imwrite(str(staging / path), page):
                    raise PairError("contact sheet write failed")
                contacts.append({"path": path, "pair_ids": page_ids, "sha256": sha(staging / path)})
        source_counts = Counter(kind for p in public for kind in p["source_kinds"])
        manifest = {"schema_version": 1, "run_id": run_id, "seed": seed, "purpose": "blinded_pair_adjudication",
                    "note": "Send only sheets/contact_sheets to the adjudicator; metadata and key reveal hypotheses. Native-source crops are upright, displayed at <=300 px height without upscaling.",
                    "source_hashes": {str(p.relative_to(runs)): sha(p) for p in selection["source_files"]},
                    "pairs": public, "contact_sheets": contacts, "counts_per_source": dict(source_counts),
                    "skipped_source_edges": selection["skipped"]}
        hidden = {"schema_version": 1, "run_id": run_id, "pairs": key,
                  "note": "Automatic controls are proxy expectations from proposed groups / same-frame exclusion; not independent human truth.",
                  "controls_requested": selection["controls_requested"], "controls_available": selection["controls_available"]}
        (staging / "pairs.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "pairs_key.json").write_text(json.dumps(hidden, indent=2) + "\n")
        staging.rename(dest)
        return {"run_id": run_id, "pairs": len(public), "counts_per_source": dict(source_counts), "output": str(dest),
                "controls": dict(Counter(v["control_kind"] for v in key.values() if v["control_kind"]))}
    except Exception:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--all-defaults", action="store_true", help="Generate YugoTest11, YugoTest7 and YugoTest5")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    for rid in DEFAULT_RUNS if args.all_defaults else (args.run,):
        print(json.dumps(generate(rid, seed=args.seed)))


if __name__ == "__main__":
    main()
