"""Group-level reading policy: read per PHYSICAL BOOK GROUP after the final association, not per observation.

plan():     for each group choose its best 1-2 sightings by  score = spine_width_px * frame_sharpness  (sightings touching the frame edge are
            down-weighted), then classify each group as attempt / unattempted (too small, low contrast, no usable sighting).
combine():  per group pick the best reading among its sightings (longest non-empty string, weighted by confidence) and keep every raw reading.
Pure functions: no I/O except reading the frame image region needed for the contrast check. Nothing is skipped silently: every group gets a status.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

MIN_WIDTH_PX = 16.0          # narrower sightings are almost surely unreadable at native resolution
MIN_EDGE_ENERGY = 3.0        # mean |Sobel|/8 inside the box; blank side-on spines are ~flat
MIN_STD = 12.0
EDGE_MARGIN = 0.015          # fraction of frame size
EDGE_PENALTY = 0.25


def spine_width_px(o: dict[str, Any]) -> float:
    p = o.get("polygon_xy")
    if p and len(p) == 4:
        a = np.asarray(p, float)
        s1 = (np.linalg.norm(a[0] - a[1]) + np.linalg.norm(a[2] - a[3])) / 2
        s2 = (np.linalg.norm(a[1] - a[2]) + np.linalg.norm(a[3] - a[0])) / 2
        return float(min(s1, s2))
    x = o["bbox_xyxy"]
    return float(min(x[2] - x[0], x[3] - x[1]))


def touches_edge(o: dict[str, Any], w: int, h: int) -> bool:
    x0, y0, x1, y1 = o["bbox_xyxy"]
    return x0 <= EDGE_MARGIN * w or y0 <= EDGE_MARGIN * h or x1 >= (1 - EDGE_MARGIN) * w or y1 >= (1 - EDGE_MARGIN) * h


def sighting_score(o: dict[str, Any], frame: dict[str, Any]) -> float:
    s = spine_width_px(o) * float(frame["quality"]["sharpness"])
    return s * (EDGE_PENALTY if touches_edge(o, frame["width_px"], frame["height_px"]) else 1.0)


def contrast(img_gray: np.ndarray, bbox: list[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    reg = img_gray[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if reg.size < 16 or min(reg.shape) < 3:
        return 0.0, 0.0
    gx, gy = cv2.Sobel(reg, cv2.CV_32F, 1, 0), cv2.Sobel(reg, cv2.CV_32F, 0, 1)
    return float(np.mean(np.abs(gx) + np.abs(gy)) / 8.0), float(reg.std())


def plan(groups: list[dict[str, Any]], obs: dict[str, dict[str, Any]], frames: dict[str, dict[str, Any]], load_gray, *, per_group: int = 2) -> dict[str, Any]:
    """Return {"groups": [{instance_id, status, reason, sightings:[{observation_id, score, width_px, edge_touch, edge_energy, std}]}], "counts": {...}}.
    ``load_gray(frame_id)`` returns the native grayscale frame (cached by the caller)."""
    out, counts = [], {"groups": 0, "attempt": 0, "unattempted_too_small": 0, "unattempted_low_contrast": 0, "unattempted_no_sighting": 0}
    for g in groups:
        counts["groups"] += 1
        sights = []
        for oid in g.get("observation_ids", []):
            o = obs.get(oid)
            f = frames.get(o["frame_id"]) if o else None
            if o is None or f is None:
                continue
            sights.append({"observation_id": oid, "frame_id": o["frame_id"], "score": round(sighting_score(o, f), 2), "width_px": round(spine_width_px(o), 1),
                           "edge_touch": touches_edge(o, f["width_px"], f["height_px"])})
        sights.sort(key=lambda s: s["score"], reverse=True)
        if not sights:
            out.append({"instance_id": g["instance_id"], "status": "unattempted_no_sighting", "reason": "no usable sighting (observation or frame missing)", "sightings": []})
            counts["unattempted_no_sighting"] += 1
            continue
        usable = [s for s in sights if s["width_px"] >= MIN_WIDTH_PX]     # a narrow-but-sharp sighting must never hide a usable wider one
        sights = usable + [s for s in sights if s["width_px"] < MIN_WIDTH_PX]
        top = sights[0]
        if top["width_px"] < MIN_WIDTH_PX:
            out.append({"instance_id": g["instance_id"], "status": "unattempted_too_small", "sightings": sights[:per_group],
                        "reason": f"best sighting {top['width_px']} px wide (< {MIN_WIDTH_PX:.0f} px): unreadable at native resolution"})
            counts["unattempted_too_small"] += 1
            continue
        chosen = []
        for s in sights:
            if len(chosen) >= per_group or s["width_px"] < MIN_WIDTH_PX:
                continue
            ee, sd = contrast(load_gray(s["frame_id"]), obs[s["observation_id"]]["bbox_xyxy"])
            s["edge_energy"], s["std"] = round(ee, 2), round(sd, 1)
            if ee < MIN_EDGE_ENERGY and sd < MIN_STD:
                s["skipped"] = "low_contrast"
                continue
            chosen.append(s)
        if not chosen:
            out.append({"instance_id": g["instance_id"], "status": "unattempted_low_contrast", "sightings": sights[:per_group],
                        "reason": "all candidate sightings are almost flat (blank/low-contrast spine)"})
            counts["unattempted_low_contrast"] += 1
            continue
        out.append({"instance_id": g["instance_id"], "status": "attempt", "reason": None, "sightings": chosen})
        counts["attempt"] += 1
    return {"groups": out, "counts": counts, "thresholds": {"min_width_px": MIN_WIDTH_PX, "min_edge_energy": MIN_EDGE_ENERGY, "min_std": MIN_STD,
                                                                  "edge_margin": EDGE_MARGIN, "edge_penalty": EDGE_PENALTY, "per_group": per_group,
                                                                  "score": "spine_width_px * frame_sharpness (edge-touching x0.25)"}}


def reading_text(r: dict[str, Any]) -> str:
    t = (r.get("title_text") or r.get("visible_text") or "").replace("\n", " ").strip()
    return "" if t in ("", "[?]") else t


def combine(group: dict[str, Any], readings: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]] | None = None,
            names: tuple[str, str] = ("primary", "second")) -> dict[str, Any]:
    """Best reading among the group's sightings: longest non-empty string, weighted by confidence (None counts as 0.5). All raw readings are kept
    (each tagged with the reader that produced it); ``union_text`` is the unique tokens over every reading (for catalogue queries; noisy)."""
    seen = []
    for tag, rd in ((names[0], readings), (names[1], second or {})):
        for oid in group.get("observation_ids", []):
            r = rd.get(oid)
            if r is not None:
                seen.append({"observation_id": oid, "reader": tag, "text": reading_text(r), "read_confidence": r.get("read_confidence"),
                             "evidence": (r.get("ocr_evidence") or {}).get("path")})
    cand = [s for s in seen if s["text"]]
    best = max(cand, key=lambda s: len(s["text"]) * (0.5 + (s["read_confidence"] if isinstance(s["read_confidence"], (int, float)) else 0.5)), default=None)
    return {"best": best, "readings_considered": seen, "union_text": token_union([s["text"] for s in cand])}


def fold_tokens(text: str) -> list[str]:
    import re
    import unicodedata
    t = unicodedata.normalize("NFD", unicodedata.normalize("NFC", text or "").casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [x for x in re.findall(r"[^\W_]+", t) if len(x) >= 2]


def token_union(texts: list[str]) -> str:
    seen: dict[str, None] = {}
    for t in texts:
        for tok in fold_tokens(t):
            seen.setdefault(tok, None)
    return " ".join(seen)


BUCKETS = ((0, 26, "<26"), (26, 40, "26-40"), (40, 60, "40-60"), (60, 1e9, ">=60"))


def yield_by_width(plan_groups: list[dict[str, Any]], books: list[dict[str, Any]], readings: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Where does reading actually work? Per best-sighting width bucket (px): groups, attempted, with any text, with >= 8 chars."""
    by_id = {b["instance_id"]: b for b in books}
    out = {name: {"groups": 0, "attempted": 0, "with_text": 0, "text_ge8": 0} for _, _, name in BUCKETS}
    for g in plan_groups:
        if not g["sightings"]:
            continue
        w = max(s["width_px"] for s in g["sightings"])
        name = next(n for lo, hi, n in BUCKETS if lo <= w < hi)
        o = out[name]
        o["groups"] += 1
        if g["status"].startswith("unattempted"):
            continue
        o["attempted"] += 1
        best = combine(by_id[g["instance_id"]], readings)["best"] if g["instance_id"] in by_id else None
        if best:
            o["with_text"] += 1
            o["text_ge8"] += len(best["text"]) >= 8
    return out
