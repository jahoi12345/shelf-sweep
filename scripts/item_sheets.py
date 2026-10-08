"""Saved-packet item evidence for a blinded visual review; no pipeline imports.

Generate: .venv/bin/python scripts/item_sheets.py generate
Score frozen answers: .venv/bin/python scripts/item_sheets.py score
Keys are written separately and are only read by the score command.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    "sweep_3af346a2f3f14f72", "sweep_92ea122a48394471",
    "sweep_059fa9973f7e4612", "sweep_75a1c0ad5a564bc9",
)
CATEGORIES = set("plant pot vase frame cup box basket lamp ornament sign record cushion sculpture shelf_part book other".split())
FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
SEED = 20261007


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def safe_file(run, ref):
    p = run / ref
    if not p.resolve().is_relative_to(run.resolve()) or not p.is_file():
        raise ValueError(f"Missing or unsafe frame: {ref}")
    return p


def fit(img, size):
    img = img.copy()
    img.thumbnail(size, Image.Resampling.LANCZOS)
    return img


def generate(run_id, output):
    run = ROOT / "runs" / run_id
    dest = output / run_id
    # Immutable sets: never replace evidence associated with existing answers.
    dest.mkdir(parents=True, exist_ok=False)
    for sub in ("sheets", "native_crops", "contexts", "contacts"):
        (dest / sub).mkdir()
    packet = read(run / "claim_packet.json")["data"]
    inventory = read(run / "inventory.json")["data"]
    manifest = read(run / "manifest.json")["data"]
    frames = {f["frame_id"]: f for f in manifest["frames"]}
    refs = {f["image_path"]: f for f in manifest["frames"]}
    rows = [(i, "item") for i in packet.get("items", [])]
    # These packet versions omit item_candidates. Preserve that fact and review
    # the saved inventory candidates as a separately scored supplement.
    candidates = packet.get("item_candidates")
    candidate_source = "packet" if candidates is not None else "inventory"
    if candidates is None:
        candidates = inventory.get("item_candidates", [])
    rows += [(i, "candidate") for i in candidates]
    random.Random(SEED).shuffle(rows)
    key, blind, tiles = {}, [], []
    rotate = 90 if run_id == RUNS[-1] else 0  # fixed camera quarter-turn
    for index, (line, source) in enumerate(rows, 1):
        item_id = line.get("id") or line["instance_id"]
        sheet_id = item_id if source == "item" else "candidate-" + item_id
        if sheet_id in key:
            raise ValueError("Duplicate sheet id")
        ev = line.get("evidence") or line
        frame_ref = ev.get("frame_ref") or line["frame_ref"]
        if frame_ref in frames:
            frame_ref = frames[frame_ref]["image_path"]
        fid = ev.get("frame_id") or line.get("frame_id") or refs[frame_ref]["frame_id"]
        timestamp = frames[fid]["timestamp_s"]
        image_path = safe_file(run, frame_ref)
        with Image.open(image_path) as im:
            img = im.convert("RGB")
        b = ev.get("bbox_xyxy") or line.get("bbox_xyxy")
        if not b or len(b) != 4 or any(not math.isfinite(float(v)) for v in b):
            raise ValueError(f"Invalid bbox: {sheet_id}")
        x0, y0, x1, y1 = b
        box = (max(0, math.floor(x0)), max(0, math.floor(y0)),
               min(img.width, math.ceil(x1)), min(img.height, math.ceil(y1)))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Empty bbox: {sheet_id}")
        crop = img.crop(box)
        pad = max(160, round(max(box[2]-box[0], box[3]-box[1]) * .55))
        cb = (max(0, box[0]-pad), max(0, box[1]-pad),
              min(img.width, box[2]+pad), min(img.height, box[3]+pad))
        context = img.crop(cb)
        ImageDraw.Draw(context).rectangle(
            (box[0]-cb[0], box[1]-cb[1], box[2]-cb[0]-1, box[3]-cb[1]-1),
            outline="#fff000", width=4)
        if rotate:
            crop = crop.transpose(Image.Transpose.ROTATE_90)
            context = context.transpose(Image.Transpose.ROTATE_90)
        crop.save(dest / "native_crops" / f"{sheet_id}.png")
        context.save(dest / "contexts" / f"{sheet_id}.png")
        # Individual sheets show the crop at native pixels and context at <=1x.
        ctx = fit(context, (800, 800))
        sheet = Image.new("RGB", (crop.width + ctx.width + 32, max(crop.height, ctx.height)+65), "#202020")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f"{sheet_id}   {fid}   {timestamp:.3f}s", font=FONT, fill="white")
        sheet.paste(crop, (8, 50))
        sheet.paste(ctx, (crop.width+24, 50))
        sheet.save(dest / "sheets" / f"{sheet_id}.png")
        tile = Image.new("RGB", (600, 335), "#202020")
        ImageDraw.Draw(tile).text((8, 8), f"{sheet_id} | {fid} | {timestamp:.1f}s", font=FONT, fill="white")
        a, c = fit(crop, (210, 280)), fit(context, (370, 280))
        tile.paste(a, (8+(210-a.width)//2, 45+(280-a.height)//2))
        tile.paste(c, (225+(370-c.width)//2, 45+(280-c.height)//2))
        tiles.append(tile)
        key[sheet_id] = {"item_id": item_id, "category": line.get("category"),
                         "classification": line.get("classification"), "source": source}
        blind.append({"sheet_id": sheet_id, "frame_id": fid, "frame_ref": frame_ref,
                      "timestamp_s": timestamp, "bbox_xyxy": list(b), "crop_box": list(box),
                      "context_box": list(cb), "rotation_ccw": rotate,
                      "source_frame_sha256": sha(image_path),
                      "sheet_sha256": sha(dest / "sheets" / f"{sheet_id}.png")})
    for start in range(0, len(tiles), 12):
        page = Image.new("RGB", (1800, 1340), "#404040")
        for j, tile in enumerate(tiles[start:start+12]):
            page.paste(tile, ((j % 3)*600, (j//3)*335))
        page.save(dest / "contacts" / f"page-{start//12+1:03d}.png")
    write(dest / "items_key.json", key)
    write(dest / "sheets_manifest.json", {
        "run_id": run_id, "seed": SEED, "items": len(packet.get("items", [])),
        "packet_candidates_present": "item_candidates" in packet,
        "candidate_source": candidate_source, "candidates": len(candidates),
        "source_hashes": {name: sha(run / name) for name in ("claim_packet.json", "inventory.json", "manifest.json")},
        "sheets": blind,
    })
    print(json.dumps({"run": run_id, "sheets": len(rows), "pages": math.ceil(len(rows)/12)}))


def score(run_id, output, answers_dir):
    dest = output / run_id
    manifest = read(dest / "sheets_manifest.json")
    answers_path = answers_dir / f"{run_id}_answers.jsonl"
    answers = [json.loads(line) for line in answers_path.read_text().splitlines() if line.strip()]
    freeze_path = answers_dir / "review_freeze.json"
    if freeze_path.exists():
        frozen = read(freeze_path)["answer_sha256"].get(answers_path.name)
        if frozen != sha(answers_path):
            raise ValueError("Answers changed after the blind review was frozen")
    key = read(dest / "items_key.json")
    ids = [a["item_id"] for a in answers]
    if len(ids) != len(set(ids)) or set(ids) != set(key):
        raise ValueError(f"Incomplete or duplicated answers for {run_id}")
    seen = set()
    for a in answers:
        if set(a) != {"item_id", "real_object", "your_category", "duplicate_of", "reason"}:
            raise ValueError("Unexpected answer fields")
        if a["real_object"] not in {"yes", "no", "cannot_tell"} or a["your_category"] not in CATEGORIES:
            raise ValueError("Invalid judgement")
        dup = a["duplicate_of"]
        if dup is not None and (dup not in seen or a["real_object"] != "yes"):
            raise ValueError("Duplicates must refer to an earlier real object")
        if dup and next(x for x in answers if x["item_id"] == dup)["real_object"] != "yes":
            raise ValueError("Duplicate target is not real")
        seen.add(a["item_id"])
    for row in manifest["sheets"]:
        if sha(dest / "sheets" / f"{row['sheet_id']}.png") != row["sheet_sha256"]:
            raise ValueError("Sheet bytes changed")
        if sha(safe_file(ROOT / "runs" / run_id, row["frame_ref"])) != row["source_frame_sha256"]:
            raise ValueError("Source frame bytes changed")
    current_hashes = {name: sha(ROOT / "runs" / run_id / name) for name in manifest["source_hashes"]}
    drift = [name for name, value in manifest["source_hashes"].items() if current_hashes[name] != value]
    # Completed packets can acquire measurement attachments during review. The
    # frozen key + sheet manifest define this review's captured item population;
    # never substitute newly saved lines or silently claim the files are unchanged.
    current_items = {i["id"]: i for i in read(ROOT / "runs" / run_id / "claim_packet.json")["data"]["items"]}
    item_drift = []
    for row in manifest["sheets"]:
        k = key[row["sheet_id"]]
        if k["source"] != "item":
            continue
        i = current_items.get(k["item_id"])
        ev = (i or {}).get("evidence") or i or {}
        if (not i or i.get("category") != k["category"] or i.get("classification") != k["classification"]
                or ev.get("frame_id") != row["frame_id"] or ev.get("bbox_xyxy") != row["bbox_xyxy"]):
            item_drift.append(row["sheet_id"])
    # Exact labels and a narrow, disclosed taxonomy map. Ambiguous book_or_*
    # labels are never mapped to an accepted non-book category.
    aliases = {"potted_plant": "plant", "framed_sign": "sign", "picture_frame": "frame",
               "jar": "pot", "shelving": "shelf_part", "desk": "shelf_part"}
    by_id = {a["item_id"]: a for a in answers}
    def root(i):
        while by_id[i]["duplicate_of"]:
            i = by_id[i]["duplicate_of"]
        return i
    results = {}
    populations = {
        "item": lambda k: k["source"] == "item",
        "candidate": lambda k: k["source"] == "candidate",
        "emitted_non_book": lambda k: k["source"] == "item" and k["classification"] == "non_book",
        "emitted_ambiguous": lambda k: k["source"] == "item" and k["classification"] == "ambiguous",
    }
    for source, include in populations.items():
        rows = [a for a in answers if include(key[a["item_id"]])]
        n = len(rows)
        real = [a for a in rows if a["real_object"] == "yes"]
        exact = sum(key[a["item_id"]]["category"] == a["your_category"] for a in rows)
        mapped = sum(aliases.get(key[a["item_id"]]["category"], key[a["item_id"]]["category"]) == a["your_category"] for a in rows)
        real_match = [a for a in real if aliases.get(key[a["item_id"]]["category"], key[a["item_id"]]["category"]) == a["your_category"]]
        real_exact = [a for a in real if key[a["item_id"]]["category"] == a["your_category"]]
        clusters = {root(a["item_id"]) for a in real}
        correct_clusters = {root(a["item_id"]) for a in real_match}
        results[source] = {
            "judged": n, "real": len(real), "not_item": sum(a["real_object"] == "no" for a in rows),
            "cannot_tell": sum(a["real_object"] == "cannot_tell" for a in rows),
            "real_pct": round(100*len(real)/n, 2) if n else None,
            "category_exact": exact, "category_exact_pct": round(100*exact/n, 2) if n else None,
            "category_mapped": mapped, "category_mapped_pct": round(100*mapped/n, 2) if n else None,
            "real_category_match": len(real_match),
            "real_category_exact": len(real_exact),
            "real_category_exact_pct_all": round(100*len(real_exact)/n, 2) if n else None,
            "real_category_exact_pct_real": round(100*len(real_exact)/len(real), 2) if real else None,
            "duplicates": sum(a["duplicate_of"] is not None for a in rows),
            "duplicate_lines_within_population": len(real) - len(clusters),
            "distinct_visible_real_clusters": len(clusters),
            "distinct_exact_category_clusters": len({root(a["item_id"]) for a in real_exact}),
            "distinct_exact_per_line_pct": round(100*len({root(a["item_id"]) for a in real_exact})/n, 2) if n else None,
            "distinct_correctly_categorized_clusters": len(correct_clusters),
            "distinct_correct_per_line_pct": round(100*len(correct_clusters)/n, 2) if n else None,
            "visual_categories": dict(Counter(a["your_category"] for a in rows)),
        }
    result = {"run_id": run_id, "kind": "model_read_not_truth", "answers_sha256": sha(answers_path),
              "original_source_hashes": manifest["source_hashes"], "current_source_hashes": current_hashes,
              "source_files_changed_since_generation": drift,
              "current_packet_item_metadata_changed": item_drift,
              "current_packet_item_count": len(current_items),
              "key_sha256": sha(dest / "items_key.json"), "aliases": aliases, "results": results}
    write(answers_dir / f"{run_id}_score.json", result)
    print(json.dumps(result, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("generate", "score"))
    ap.add_argument("--run", choices=RUNS, action="append")
    ap.add_argument("--output", type=Path, default=ROOT / "runs" / "item_sheets")
    ap.add_argument("--answers", type=Path, default=ROOT / "runs" / "item_sheets" / "answers")
    args = ap.parse_args()
    for run_id in args.run or RUNS:
        if args.command == "generate":
            generate(run_id, args.output)
        else:
            score(run_id, args.output, args.answers)


if __name__ == "__main__":
    main()
