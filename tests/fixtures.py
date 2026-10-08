"""FIXTURE stage outputs for integration checks only. Everything here is fake and marked ``fixture: true``.

Never use these for evaluation or pass off as detections. Selected by ``pipeline.run --fixture``.
"""
from __future__ import annotations

from typing import Any

from pipeline import contracts as C


def _frames(manifest: dict[str, Any], n: int = 3):
    sel = [f for f in manifest["data"]["frames"] if f["selected"]][:n]
    return sel


def make_fixture_stage(run, stage: str) -> None:
    """Write fixture artifact for ``stage`` into run.out and record it in run state (status 'ok', fixture)."""
    manifest = C.read_json(run.artifact("frames"))
    rid, out = run.run_id, run.out
    fr = _frames(manifest)
    obs = []
    for k, f in enumerate(fr):
        w, h = f["width_px"], f["height_px"]
        for j, (kind, cat) in enumerate([("book", "book"), ("book", "book"), ("non_book", "plant"), ("non_book", "framed picture"), ("ambiguous", "decor")]):
            x0 = int(w * (0.1 + 0.15 * j)); y0 = int(h * (0.3 + 0.05 * k))
            obs.append({"observation_id": C.observation_id(f["frame_id"], j), "frame_id": f["frame_id"],
                        "bbox_xyxy": [x0, y0, x0 + int(w * 0.04), y0 + int(h * 0.3)], "kind": kind, "category": cat,
                        "orientation": "upright", "confidence": 0.9 if j else 0.4, "evidence_notes": "FIXTURE"})
    if stage == "detect":
        art = C.envelope(rid, "detect", {"observations": obs}, fixture=True)
        path = C.write_json(out / C.ARTIFACTS["detect"], art)
    elif stage == "read":
        rd = [{"observation_id": o["observation_id"], "crop_path": None, "visible_text": "FIXTURE TEXT" if i == 0 else None,
               "title_text": None, "read_confidence": 0.5 if i == 0 else None, "issues": ["fixture"]}
              for i, o in enumerate(x for x in obs if x["kind"] == "book")]
        path = C.write_json(out / C.ARTIFACTS["read"], C.envelope(rid, "read", {"readings": rd}, fixture=True))
    elif stage == "resolve":
        books = [o for o in obs if o["kind"] == "book"]
        rs = [{"observation_id": books[0]["observation_id"], "title": "FIXTURE BOOK", "author": "FIXTURE AUTHOR",
               "source_url": "fixture://not-a-real-source", "decision": "accepted", "matching_evidence": ["fixture"], "issues": []}]
        path = C.write_json(out / C.ARTIFACTS["resolve"], C.envelope(rid, "resolve", {"resolutions": rs}, fixture=True))
    else:  # dedup
        books = [o for o in obs if o["kind"] == "book"]
        inv_books = []
        for n, o in enumerate(books):
            inv_books.append({"instance_id": C.instance_id("bk", n + 1), "observation_ids": [o["observation_id"]], "frame_ref": o["frame_id"],
                              "bbox_xyxy": o["bbox_xyxy"], "orientation": "upright", "confidence": o["confidence"],
                              "identity": ({"title": "FIXTURE BOOK", "author": "FIXTURE AUTHOR", "decision": "accepted",
                                            "source_url": "fixture://not-a-real-source", "visible_text": "FIXTURE TEXT"} if n == 0 else
                                           {"decision": "no_match", "visible_text": None}),
                              "dedup_reasons": ["fixture"], "uncertain_association": n == 1, "issues": []})
        items = [{"instance_id": C.instance_id("it", n + 1), "observation_ids": [o["observation_id"]], "frame_ref": o["frame_id"],
                  "bbox_xyxy": o["bbox_xyxy"], "category": o["category"], "description": "FIXTURE " + o["category"],
                  "confidence": o["confidence"], "dedup_reasons": ["fixture"], "issues": []}
                 for n, o in enumerate(x for x in obs if x["kind"] != "book")]
        art = C.envelope(rid, "dedup", {"books": inv_books, "items": items,
                         "counts": {"count_lower": len(inv_books) - 1, "count_upper": len(inv_books), "uncertain_associations": 1,
                                    "uncertain_observations": 1}}, fixture=True)
        path = C.write_json(out / C.ARTIFACTS["dedup"], art)
    run.state["stages"][stage] = {"status": "ok", "elapsed_s": 0.0, "artifact": C.rel_path(path, out), "fixture": True,
                                  "detail": "FIXTURE data, not a real detection/reading/resolution/merge"}
    run.save_state()
