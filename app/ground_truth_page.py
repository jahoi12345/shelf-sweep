"""Ground-truth labelling page data: http://127.0.0.1:4182/truth, loopback server only (same gating as /watch).

Reads a sweep's claim_packet.json (never writes anything under runs/) and keeps the human's answers in ONE file outside every run directory:
``ground_truth/ground_truth_labels.json`` (git-ignored) = {"version": 1, "sweeps": {"<sweep id>": {...}}}. Per sweep:

  titles          {line id: {"hand_title": str, "correct": "Y"|"N"|null, "legible": bool|null}}
  missed_titles   [{"title", "frame", "notes"}]            legible spines the system did not list
  items           {item id: {"real": "Y"|"N"|null, "true_category": str, "category_correct": "Y"|"N"|null}}
  missed_items    [{"category", "frame", "notes"}]
  true_item_count int|null
  prices          {line id: {"price": float|null, "url": str, "date": "YYYY-MM-DD"}}   only the 15 sampled lines
  position        {"titles": line id, "items": item id}    resume where the person left off
  updated_at      ISO time

"Write to sheet" rebuilds the workbook with scripts/ground_truth_sheet.py (the price sample comes from the same code, so the page and the
sheet always show the same 15 books). No hosted call anywhere.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from app import watch as WATCH

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "sweep_4ad69753649948dc"          # the demo-video sweep
SHEET_NAME = "ground_truth_and_results.xlsx"
LABELS_NAME = "ground_truth_labels.json"
YN = {"Y", "N"}
FRAME_REF_RE = re.compile(r"^frames/(f\d{5}\.jpg)$")
FREE_FRAME_RE = re.compile(r"^(f?\d{1,5}(\.jpg)?)?$")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})?$")
MAX_TEXT = 300
MAX_LIST = 500
_lock = threading.Lock()


def ground_truth_dir() -> Path:
    return Path(os.getenv("LIBRARY_GROUND_TRUTH_DIR") or ROOT / "ground_truth")


def labels_path() -> Path:
    return ground_truth_dir() / LABELS_NAME


def sheet_path(run_id: str) -> Path:
    """The default sweep keeps the plain workbook name; another sweep gets its own file so it never overwrites it."""
    return ground_truth_dir() / (SHEET_NAME if run_id == DEFAULT_RUN else f"ground_truth_and_results_{run_id}.xlsx")


def sheet_module():
    spec = importlib.util.spec_from_file_location("ground_truth_sheet", ROOT / "scripts" / "ground_truth_sheet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SHEET = None


def _sheet():
    global _SHEET
    if _SHEET is None:
        _SHEET = sheet_module()
    return _SHEET


# ------------------------------------------------------------------------------------------------ packet -> cards
def _frame(ref: Any) -> str | None:
    m = FRAME_REF_RE.fullmatch(str(ref or ""))
    return m.group(1) if m else None


def _box(ev: dict[str, Any]) -> dict[str, Any]:
    poly = ev.get("polygon_xy")
    bbox = ev.get("bbox_xyxy")
    if isinstance(poly, list) and len(poly) >= 3:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    return {"bbox": [round(float(v), 1) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None,
            "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in poly] if isinstance(poly, list) and len(poly) >= 3 else None,
            "frame_w": ev.get("frame_width_px"), "frame_h": ev.get("frame_height_px")}


def _view(env: Any) -> dict[str, Any] | None:
    data = (env or {}).get("data") or {}
    if not data.get("books") and not data.get("items"):
        return None
    mod = _sheet()
    books = mod.sorted_books(data)
    rows = []
    for b in books:
        ev = b.get("evidence") or {}
        rows.append({"id": b.get("id"), "i": b.get("display_index"), "title": b.get("title"), "author": b.get("author"), "status": b.get("status"),
                     "confidence": b.get("id_confidence"), "visible_text": (b.get("visible_text") or "")[:200] or None,
                     "frame": _frame(ev.get("frame_ref") or b.get("frame_ref")), "t_s": b.get("timestamp_s"),
                     "crop": WATCH._crop_name(b), **_box(ev)})
    items = []
    for it in data.get("items") or []:
        ev = it.get("evidence") or {}
        items.append({"id": it.get("id"), "i": it.get("display_index"), "category": it.get("category"), "status": it.get("status"),
                      "confidence": it.get("confidence"), "description": (it.get("description") or "")[:200] or None,
                      "frame": _frame(ev.get("frame_ref") or it.get("frame_ref")), "t_s": it.get("timestamp_s"), **_box(ev)})
    pp, n_priced = mod.price_sample(books)
    prices = []
    for b in pp:
        rc = b.get("replacement_cost") or {}
        url = rc.get("url") if str(rc.get("url") or "").startswith(("https://", "http://")) else None
        prices.append({"id": b.get("id"), "i": b.get("display_index"), "title": b.get("title"), "author": b.get("author"),
                       "isbn": rc.get("isbn") or b.get("isbn"), "amount": rc.get("amount"), "currency": rc.get("currency"),
                       "converted": bool(rc.get("converted")), "original": [rc.get("original_amount"), rc.get("original_currency")] if rc.get("converted") else None,
                       "source": rc.get("source"), "condition": rc.get("condition_assumed"), "url": url, "retrieved_at": rc.get("retrieved_at")})
    sweep = data.get("sweep") or {}
    cats = sorted({str(x["category"]) for x in items if x.get("category")})
    return {"sweep": {"id": sweep.get("id"), "captured_at": sweep.get("captured_at"), "duration_s": sweep.get("duration_s")},
            "books": rows, "items": items, "categories": cats, "prices": prices, "priced": n_priced,
            "system_book_count": (data.get("totals") or {}).get("book_count")}


def packet_cards(d: Path) -> dict[str, Any] | None:
    return WATCH._cached(d / "claim_packet.json", _view)


# ------------------------------------------------------------------------------------------------ labels file
def _read_doc() -> dict[str, Any]:
    p = labels_path()
    try:
        doc = json.loads(p.read_text())
    except FileNotFoundError:
        return {"version": 1, "sweeps": {}}
    if not isinstance(doc, dict) or not isinstance(doc.get("sweeps"), dict):
        raise ValueError(f"{p.name} is not a labels file")
    return doc


def _write_doc(doc: dict[str, Any]) -> None:
    p = labels_path(); p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        os.chmod(tmp, 0o644)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def get_labels(run_id: str) -> dict[str, Any]:
    return (_read_doc()["sweeps"].get(run_id)) or {}


class LabelError(ValueError):
    pass


def _text(v: Any, what: str, n: int = MAX_TEXT) -> str:
    if v is None:
        return ""
    if not isinstance(v, str) or len(v) > n:
        raise LabelError(f"{what}: text up to {n} characters")
    return v.strip()


def _yn(v: Any, what: str) -> str | None:
    if v in (None, ""):
        return None
    if v not in YN:
        raise LabelError(f"{what}: Y, N or empty")
    return v


def _frame_text(v: Any) -> str:
    s = _text(v, "frame", 20)
    if not FREE_FRAME_RE.fullmatch(s):
        raise LabelError("frame: like f00123")
    return s


def _clean(patch: dict[str, Any], cards: dict[str, Any]) -> dict[str, Any]:
    """Validate a patch from the page against this sweep's packet (ids must exist; the price ids must be the sampled 15)."""
    if not isinstance(patch, dict):
        raise LabelError("expected an object")
    book_ids = {b["id"] for b in cards["books"]}; item_ids = {x["id"] for x in cards["items"]}; price_ids = {p["id"] for p in cards["prices"]}
    out: dict[str, Any] = {}
    for key in patch:
        if key not in {"titles", "items", "prices", "missed_titles", "missed_items", "true_item_count", "position"}:
            raise LabelError(f"unknown field {key}")
    for k, v in (patch.get("titles") or {}).items():
        if k not in book_ids:
            raise LabelError(f"no book line {k}")
        if v is None:
            out.setdefault("titles", {})[k] = None; continue
        legible = v.get("legible")
        if legible not in (None, True, False):
            raise LabelError("legible: true, false or null")
        out.setdefault("titles", {})[k] = {"hand_title": _text(v.get("hand_title"), "hand_title"), "correct": _yn(v.get("correct"), "correct"),
                                           "legible": legible}
    for k, v in (patch.get("items") or {}).items():
        if k not in item_ids:
            raise LabelError(f"no item {k}")
        if v is None:
            out.setdefault("items", {})[k] = None; continue
        out.setdefault("items", {})[k] = {"real": _yn(v.get("real"), "real"), "true_category": _text(v.get("true_category"), "true_category", 80),
                                          "category_correct": _yn(v.get("category_correct"), "category_correct")}
    for k, v in (patch.get("prices") or {}).items():
        if k not in price_ids:
            raise LabelError(f"{k} is not one of the 15 sampled price lines")
        if v is None:
            out.setdefault("prices", {})[k] = None; continue
        price = v.get("price")
        if price in ("", None):
            price = None
        else:
            try:
                price = round(float(price), 2)
            except (TypeError, ValueError):
                raise LabelError("price: a number") from None
            if not 0 <= price < 100000:
                raise LabelError("price: 0 to 99999")
        url = _text(v.get("url"), "url", 2000)
        if url and not url.startswith(("https://", "http://")):
            raise LabelError("url: http(s) only")
        date = _text(v.get("date"), "date", 10)
        if not DATE_RE.fullmatch(date):
            raise LabelError("date: YYYY-MM-DD")
        out.setdefault("prices", {})[k] = {"price": price, "url": url, "date": date}
    if "missed_titles" in patch:
        lst = patch["missed_titles"]
        if not isinstance(lst, list) or len(lst) > MAX_LIST:
            raise LabelError("missed_titles: a list")
        out["missed_titles"] = [{"title": _text(m.get("title"), "title"), "frame": _frame_text(m.get("frame")), "notes": _text(m.get("notes"), "notes")}
                                for m in lst if isinstance(m, dict)]
    if "missed_items" in patch:
        lst = patch["missed_items"]
        if not isinstance(lst, list) or len(lst) > MAX_LIST:
            raise LabelError("missed_items: a list")
        out["missed_items"] = [{"category": _text(m.get("category"), "category", 80), "frame": _frame_text(m.get("frame")),
                                "notes": _text(m.get("notes"), "notes")} for m in lst if isinstance(m, dict)]
    if "true_item_count" in patch:
        n = patch["true_item_count"]
        if n in (None, ""):
            out["true_item_count"] = None
        elif isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 100000:
            raise LabelError("true_item_count: a whole number")
        else:
            out["true_item_count"] = n
    if "position" in patch:
        pos = patch["position"] or {}
        out["position"] = {"titles": pos.get("titles") if pos.get("titles") in book_ids else None,
                           "items": pos.get("items") if pos.get("items") in item_ids else None}
    return out


def save_labels(run_id: str, patch: dict[str, Any], cards: dict[str, Any]) -> dict[str, Any]:
    """Merge a validated patch into this sweep's entry (per line / per item / per price; lists and counts replace) and write the file
    atomically. A null entry removes that line's answer."""
    clean = _clean(patch, cards)
    with _lock:
        doc = _read_doc()
        cur = doc["sweeps"].setdefault(run_id, {})
        for sec in ("titles", "items", "prices"):
            for k, v in (clean.get(sec) or {}).items():
                if v is None:
                    cur.setdefault(sec, {}).pop(k, None)
                else:
                    cur.setdefault(sec, {})[k] = v
        for k in ("missed_titles", "missed_items", "true_item_count", "position"):
            if k in clean:
                cur[k] = clean[k]
        cur["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_doc(doc)
        return cur


def excel_lock(run_id: str) -> bool:
    """Excel keeps ~$<name> next to a workbook it has open; replacing the file then loses whatever Excel saves next."""
    p = sheet_path(run_id)
    return (p.parent / f"~${p.name}").exists()


def write_sheet(run_id: str, d: Path) -> dict[str, Any]:
    with _lock:
        labels = (_read_doc()["sweeps"].get(run_id)) or {}
        out = sheet_path(run_id)
        res = _sheet().build(d / "claim_packet.json", out, labels)
        res["out"] = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else out.name
        res["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        doc = _read_doc()                      # the page shows the last write's preview again on the next visit
        doc["sweeps"].setdefault(run_id, {})["last_sheet"] = {k: res[k] for k in ("out", "written_at", "results")}
        _write_doc(doc)
    return res
