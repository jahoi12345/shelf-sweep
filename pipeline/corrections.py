"""Voice/user corrections and ownership exclusions as a deterministic overlay.

Corrections are NEVER baked into stage artifacts. They are an append-only ledger applied on top of
the latest ``inventory.json`` every time a snapshot/packet is built, so a late or stale merge result
cannot undo a correction, and offline replays can re-apply the same ledger.

Targets are matched through stable ``observation_id``s (``f00012-o003``). Instance ids are derived from the
group's earliest observation (``bk-f00012-o003``) and survive re-merges, but a merge can still fold one id
into another, so instance ids supplied by a caller are resolved to observation ids at call time
(``resolve_instance_targets``).

Ledger entry:
  {"id": "c001", "kind": <KINDS>, "target": {...}, "value": {...}, "source": {"turn_id", "utterance", "actor"},
   "created_at": iso, "active": bool}
Target forms:  {"observation_ids": [...]}  or  {"time_window": {"start_s": x, "end_s": y}}
Kinds: exclude | restore | reclassify (value.kind book|non_book, value.category) | set_title (value.title/author/language)
       | flag_appraisal | mark_print (art is a print, not an original) | clear_flag
       | split (undo a merge of the physical-instance layer: the member track group targeted becomes its own line again;
         applied by pipeline.instances.apply, nothing changes at the inventory level; value.member_group_id for display)
"""

from __future__ import annotations

import copy
import time
from typing import Any

KINDS = {"exclude", "restore", "reclassify", "set_title", "flag_appraisal", "mark_print", "clear_flag", "split"}


class CorrectionError(ValueError):
    pass


def new_entry(entry_id: str, kind: str, target: dict[str, Any], value: dict[str, Any] | None = None,
              source: dict[str, Any] | None = None) -> dict[str, Any]:
    if kind not in KINDS:
        raise CorrectionError(f"unknown correction kind {kind!r}; choose from {sorted(KINDS)}")
    t = target or {}
    if not (t.get("observation_ids") or t.get("time_window")):
        raise CorrectionError("target needs observation_ids or time_window")
    if kind == "set_title" and not str((value or {}).get("title") or "").strip():
        raise CorrectionError("set_title needs a non-empty title")
    if kind == "reclassify" and (value or {}).get("kind") not in {"book", "non_book"}:
        raise CorrectionError("reclassify needs value.kind of 'book' or 'non_book'")
    if t.get("time_window"):
        w = t["time_window"]
        if not all(isinstance(w.get(k), (int, float)) for k in ("start_s", "end_s")) or w["end_s"] < w["start_s"]:
            raise CorrectionError("time_window needs numeric start_s <= end_s")
    return {"id": entry_id, "kind": kind, "target": t, "value": value or {}, "source": source or {},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "active": True}


def resolve_instance_targets(inventory_data: dict[str, Any], instance_ids: list[str]) -> list[str]:
    """Instance ids as shown to the user -> stable observation ids. Unknown ids raise."""
    index = {i["instance_id"]: i for i in (inventory_data.get("books") or []) + (inventory_data.get("items") or [])}
    missing = [i for i in instance_ids if i not in index]
    if missing:
        raise CorrectionError(f"unknown instance id(s): {', '.join(missing)}")
    return [oid for i in instance_ids for oid in index[i]["observation_ids"]]


def _matches(entry: dict[str, Any], inst: dict[str, Any], frame_t: dict[str, float]) -> bool:
    t = entry["target"]
    obs = set(inst["observation_ids"])
    if t.get("observation_ids"):
        return bool(obs & set(t["observation_ids"]))
    w = t["time_window"]
    times = [frame_t[o.split("-o")[0]] for o in obs if o.split("-o")[0] in frame_t]
    # an instance is in-window only if EVERY observation lies inside it (never exclude a book seen elsewhere)
    return bool(times) and all(w["start_s"] <= x <= w["end_s"] for x in times)


def apply(inventory_data: dict[str, Any], ledger: list[dict[str, Any]], frame_times: dict[str, float]
          ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (effective_inventory_data, excluded_lines, applied_log). Input is not mutated.

    Order matters and is deterministic: entries apply in ledger order; ``restore`` undoes an earlier
    ``exclude`` for the same target; later entries of the same kind win.
    """
    inv = copy.deepcopy(inventory_data)
    lines = [(("books", i)) for i in inv.get("books", [])] + [("items", i) for i in inv.get("items", [])]
    state: dict[str, dict[str, Any]] = {i["instance_id"]: {"excluded": None, "title": None, "kind": None,
                                                           "appraisal": None, "print": False} for _, i in lines}
    log: list[dict[str, Any]] = []
    for e in (x for x in ledger if x.get("active", True)):
        hits = [i for _, i in lines if _matches(e, i, frame_times)]
        log.append({"correction_id": e["id"], "kind": e["kind"], "matched_instances": [i["instance_id"] for i in hits]})
        for i in hits:
            st = state[i["instance_id"]]
            k, v = e["kind"], e["value"]
            if k == "exclude":
                st["excluded"] = {"correction_id": e["id"], "reason": v.get("reason") or "excluded by user", "source": e["source"]}
            elif k == "restore":
                st["excluded"] = None
            elif k == "reclassify":
                st["kind"] = {"kind": v["kind"], "category": v.get("category"), "correction_id": e["id"]}
            elif k == "set_title":
                st["title"] = {"title": v["title"].strip(), "author": v.get("author"), "language": v.get("language"),
                               "correction_id": e["id"], "source": e["source"]}
            elif k == "flag_appraisal":
                st["appraisal"] = {"reason": v.get("reason") or "user flagged signed/rare/first edition", "correction_id": e["id"]}
            elif k == "clear_flag":
                st["appraisal"] = None
            elif k == "mark_print":
                st["print"] = True
            # "split" changes nothing here: the instance layer (pipeline.instances.apply) reads it from the ledger; the match is logged for the audit trail
    out_books: list[dict[str, Any]] = []
    out_items: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for key, inst in lines:
        st = state[inst["instance_id"]]
        inst.setdefault("user_corrections", [])
        if st["excluded"]:
            excluded.append({"instance_id": inst["instance_id"], "kind_before": key[:-1] if key.endswith("s") else key,
                             "observation_ids": inst["observation_ids"], "frame_ref": inst.get("frame_ref"),
                             "bbox_xyxy": inst.get("bbox_xyxy"), **st["excluded"]})
            continue
        target_list = key
        if st["kind"]:
            target_list = "books" if st["kind"]["kind"] == "book" else "items"
            inst["kind"] = st["kind"]["kind"]
            if st["kind"]["kind"] == "non_book":
                inst["category"] = st["kind"].get("category") or inst.get("category") or "unspecified"
                inst["description"] = inst.get("description") or "reclassified from book by user"
            inst["user_corrections"].append({"kind": "reclassify", "to": st["kind"]["kind"], "correction_id": st["kind"]["correction_id"]})
            if target_list == "books" and "identity" not in inst:
                inst["identity"] = {"decision": "no_match", "visible_text": None}
        if st["title"] and target_list == "books":
            ident = inst.setdefault("identity", {})
            ident.update(title=st["title"]["title"], author=st["title"]["author"], decision="user_stated",
                         source="user_statement", language=st["title"]["language"], source_url=None,
                         user_correction_id=st["title"]["correction_id"])
            inst["user_corrections"].append({"kind": "set_title", "correction_id": st["title"]["correction_id"]})
        if st["appraisal"]:
            inst["appraisal"] = st["appraisal"]
            inst["user_corrections"].append({"kind": "flag_appraisal", "correction_id": st["appraisal"]["correction_id"]})
        if st["print"]:
            inst["art_is_print"] = True
            inst["user_corrections"].append({"kind": "mark_print"})
        (out_books if target_list == "books" else out_items).append(inst)
    inv["books"], inv["items"] = out_books, out_items
    counts = dict(inv.get("counts") or {})
    counts.update(book_count=len(out_books), item_count=len(out_items))
    # association bounds from dedup describe the pre-correction candidate set; keep but mark scope
    counts["user_excluded"] = len(excluded)
    inv["counts"] = counts
    return inv, excluded, log
