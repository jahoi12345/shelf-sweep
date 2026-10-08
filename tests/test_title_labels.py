import json
import pytest
from pathlib import Path

from app import title_labels as TL

RUN = "worker1_delivery_v3"
pytestmark = pytest.mark.skipif(not (Path("runs") / RUN / "crops").is_dir(), reason="workspace run artifacts absent")


@pytest.fixture()
def tdir(tmp_path, monkeypatch):
    monkeypatch.setattr(TL.TR, "TRUTH_DIR", tmp_path)
    return tmp_path


def test_sampling_is_deterministic_and_stratified():
    a, b = TL.sample_run(RUN, 40, 7)["items"], TL.sample_run(RUN, 40, 7)["items"]
    assert [x["observation_id"] for x in a] == [x["observation_id"] for x in b]
    assert len({x["observation_id"] for x in a}) == 40
    assert len({x["stratum"] for x in a}) >= 8
    assert [x["observation_id"] for x in TL.sample_run(RUN, 40, 8)["items"]] != [x["observation_id"] for x in a]


def test_defaults_are_per_scan():
    runs = TL.list_runs()
    d = {r["scan"]: r["run_id"] for r in runs if r.get("default_for_scan")}
    assert "YugoTest2" in d and "YugoTest1" in d
    for scan, rid in TL.PREFERRED_DEFAULTS.items():       # the agreed run wins whenever it exists
        if any(r["run_id"] == rid for r in runs):
            assert d[scan] == rid


def test_representative_is_best_view_and_preview_is_in_memory(tdir):
    s = TL.open_session(RUN, "t", 4, 5)
    doc = json.loads((tdir / f"{s['label_id']}.json").read_text())
    assert "best-view" in doc["representative"] and "NATIVE" in doc["preview"]
    png = TL.crop_preview_png(s["label_id"], 0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    native = TL.crop_file(s["label_id"], 0)
    assert TL._sha(native) == doc["sample"][0]["crop_sha256"]       # the file on disk is untouched by the preview


def test_session_answer_resume_undo_and_purpose(tdir):
    s = TL.open_session(RUN, "tester", 6, 1)
    lid = s["label_id"]
    doc = json.loads((tdir / f"{lid}.json").read_text())
    assert doc["purpose"] == "evaluation_truth" and doc["annotator"] == "tester"
    assert all(len(i["crop_sha256"]) == 64 for i in doc["sample"])
    o = s["items"][0]["observation_id"]
    with pytest.raises(TL.LabelError):
        TL.answer(lid, o, "labeled", title="  ", language="es")
    with pytest.raises(TL.LabelError):
        TL.answer(lid, o, "labeled", title="x", language="fr")
    with pytest.raises(TL.LabelError):
        TL.answer(lid, o, "bogus")
    st = TL.answer(lid, o, "labeled", title="El Quijote", author="Cervantes", language="es", rotation_seen=90)
    assert st["done"] == 1
    st = TL.answer(lid, s["items"][1]["observation_id"], "not_a_book")
    assert st["counts"]["not_a_book"] == 1
    again = TL.open_session(RUN, "tester", 6, 1)             # resume
    assert again["done"] == 2 and again["next_index"] == 2
    assert TL.undo(lid)["done"] == 1
    assert TL.undo(lid)["done"] == 0
    with pytest.raises(TL.LabelError):
        TL.undo(lid)


def test_annotator_required(tdir):
    with pytest.raises(TL.LabelError):
        TL.open_session(RUN, " ", 3, 1)


def test_changed_crop_blocks_resume_and_serving(tdir):
    s = TL.open_session(RUN, "t", 3, 2)
    p = tdir / f"{s['label_id']}.json"
    doc = json.loads(p.read_text())
    doc["sample"][0]["crop_sha256"] = "0" * 64
    p.write_text(json.dumps(doc))
    with pytest.raises(TL.LabelError):
        TL.crop_file(s["label_id"], 0)
    with pytest.raises(TL.LabelError):
        TL.open_session(RUN, "t", 3, 2)


def test_pipeline_does_not_read_truth_capture():
    import pipeline, pathlib
    for p in pathlib.Path(pipeline.__file__).parent.glob("*.py"):
        assert "title_labels" not in p.read_text() and "truth_capture" not in p.read_text().replace("evaluation_truth", "") or p.name in {"scale.py", "room.py"}, p


def test_unread_groups_get_native_bbox_crops(tdir):
    run = "final_candidate_yt2_v1"
    if not (Path("runs") / run / "inventory.json").is_file():
        pytest.skip("workspace run absent")
    s = TL.sample_run(run, 60, 3)
    assert s["pool"] > 600                                               # all groups, not only the reader-attempted ones
    srcs = {i["crop_source"] for i in s["items"]}
    assert "label_extract_native_bbox" in srcs
    it = next(i for i in s["items"] if i["crop_source"] == "label_extract_native_bbox")
    p = Path("runs") / run / it["crop_path"]
    assert p.is_file() and p.parent.name == "crops_labeling" and len(it["crop_sha256"]) == 64
    import cv2
    img = cv2.imread(str(p)); x0, y0, x1, y1 = it["bbox_xyxy"]
    assert img is not None and abs(img.shape[1] - (round(x1) - round(x0))) <= 1 and abs(img.shape[0] - (round(y1) - round(y0))) <= 1   # exact bbox, no resampling


def test_prefill_and_edited_flags_and_context_crops(tdir):
    run = "final_candidate_yt2_v1"
    if not (Path("runs") / run / "inventory.json").is_file():
        pytest.skip("workspace run absent")
    s = TL.open_session(run, "t", 6, 7)
    assert any(i["prefill"] for i in s["items"]) and all("other_sightings" in i for i in s["items"])
    it = next(i for i in s["items"] if i["prefill"])
    pre = it["prefill"]
    st = TL.answer(s["label_id"], it["observation_id"], "labeled", title=pre["title"], author=pre["author"] or "", language="es", prefill=pre)
    lab = st["items"][it["index"]]["label"]
    assert lab["prefilled"] is True and lab["edited"] is False and lab["prefill_source"] == pre["source"]
    st = TL.answer(s["label_id"], it["observation_id"], "labeled", title=pre["title"] + " corrected", language="es", prefill=pre)
    assert st["items"][it["index"]]["label"]["edited"] is True
    it2 = next(i for i in s["items"] if not i["prefill"]) if any(not i["prefill"] for i in s["items"]) else None
    if it2:
        st = TL.answer(s["label_id"], it2["observation_id"], "labeled", title="Typed", language="en")
        assert st["items"][it2["index"]]["label"]["prefilled"] is False
    png = TL.context_png(s["label_id"], it["index"])
    assert png[:4] == b"\x89PNG"
    if it["other_sightings"]:
        assert TL.sighting_png(s["label_id"], it["index"], 0)[:4] == b"\x89PNG"
    with pytest.raises(TL.LabelError):
        TL.sighting_png(s["label_id"], it["index"], 5)
