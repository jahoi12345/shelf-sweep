"""Protocol review: synthetic sessions only, never human truth.

Intentional red specifications describe gaps; they do not fix the application.
All session, frame and label writes are inside pytest's temporary directory.
"""
import hashlib
import json

import cv2
import numpy as np
import pytest

from app import title_labels as TL


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(TL, "ROOT", tmp_path)
    monkeypatch.setattr(TL, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(TL.TR, "TRUTH_DIR", tmp_path / "eval/truth_capture")
    monkeypatch.setattr(TL, "_PREFILL_CACHE", {})
    monkeypatch.setattr(TL, "list_runs", lambda: [])
    run = tmp_path / "runs/synthetic_protocol_fixture"
    (run / "frames").mkdir(parents=True)
    (run / "crops").mkdir()
    image = np.random.default_rng(6).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(run / "frames/f.png"), image)
    cv2.imwrite(str(run / "crops/o.png"), image[20:160, 30:60])
    docs = {
        "manifest.json": {"data": {"frames": [{"frame_id": "f", "timestamp_s": 0,
            "image_path": "frames/f.png", "height_px": 200, "width_px": 200,
            "quality": {"sharpness": 20},
            "sha256": hashlib.sha256((run / "frames/f.png").read_bytes()).hexdigest()}]}},
        "candidates.json": {"data": {"observations": [{"observation_id": "o", "frame_id": "f",
            "bbox_xyxy": [30, 20, 60, 160], "kind": "book"}]}},
        "readings.json": {"data": {"readings": [{"observation_id": "o", "crop_path": "crops/o.png",
            "visible_text": "Cien años de soledad [?]", "title_text": None}]}},
        "inventory.json": {"data": {"books": [{"instance_id": "synthetic-g", "observation_ids": ["o"]}]}},
    }
    for name, doc in docs.items():
        (run / name).write_text(json.dumps(doc, ensure_ascii=False))
    state = TL.open_session(run.name, "SYNTHETIC fixture actor", n=1, seed=6)
    path = TL.TR.TRUTH_DIR / (state["label_id"] + ".json")
    doc = json.loads(path.read_text())
    doc.update(fixture=True, note="SYNTHETIC TEST ONLY; no human annotation or approval")
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return state, run, path


def test_control_literal_unicode_and_native_preview_are_preserved(session):
    state, run, path = session
    doc = json.loads(path.read_text())
    assert doc["purpose"] == "evaluation_truth" and doc["fixture"] is True
    assert not path.is_relative_to(run)
    assert "años" in state["items"][0]["prefill"]["title"]
    before = hashlib.sha256((run / "crops/o.png").read_bytes()).hexdigest()
    assert TL.crop_preview_png(state["label_id"], 0).startswith(b"\x89PNG")
    assert hashlib.sha256((run / "crops/o.png").read_bytes()).hexdigest() == before


def test_partial_title_has_a_distinct_non_full_title_status(session):
    state, _, _ = session
    result = TL.answer(state["label_id"], "o", "partial", title="Cien años", language="es")
    assert result["items"][0]["label"]["status"] == "partial"
    assert result["items"][0]["label"]["title"] == "Cien años"


def test_prefill_does_not_silently_remove_visible_uncertainty(session):
    state, _, _ = session
    assert "[?]" in state["items"][0]["prefill"]["title"]


def test_confirmed_prefill_records_an_explicit_human_action(session):
    state, _, _ = session
    prefill = state["items"][0]["prefill"]
    result = TL.answer(state["label_id"], "o", "labeled", title=prefill["title"],
                       language="es", prefill=prefill)
    label = result["items"][0]["label"]
    assert label["prefilled"] is True
    assert label.get("human_confirmed") is True
    assert label["annotator"] == "SYNTHETIC fixture actor"


def test_changed_frame_cannot_supply_unverified_context(session):
    state, run, _ = session
    # Frame drift while the previously labelled primary crop remains unchanged.
    cv2.imwrite(str(run / "frames/f.png"), np.zeros((200, 200, 3), dtype=np.uint8))
    with pytest.raises(TL.LabelError):
        TL.context_png(state["label_id"], 0)


def test_frozen_synthetic_version_cannot_be_silently_overwritten(session):
    state, _, path = session
    doc = json.loads(path.read_text())
    # Fictional metadata exercises the guard; this is NOT an actual freeze/approval.
    doc["frozen"] = True
    doc["provenance"] = {"source_type": "synthetic", "frozen_at": "synthetic-only"}
    path.write_text(json.dumps(doc, ensure_ascii=False))
    before = path.read_bytes()
    with pytest.raises(TL.LabelError):
        TL.answer(state["label_id"], "o", "labeled", title="Synthetic title", language="en")
    assert path.read_bytes() == before
