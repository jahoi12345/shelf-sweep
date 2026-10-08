"""Blinding, provenance and held-out scoring checks using synthetic saved pixels."""
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import pair_labels as PL
from pipeline import pair_sheets as PS


@pytest.fixture()
def saved_run(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run = runs / "fixture"
    (run / "frames").mkdir(parents=True)
    frames, obs, groups = [], [], []
    for k in range(3):
        fid = f"f{k:05d}"
        image = np.full((380, 540, 3), 240, np.uint8)
        for i in range(12):
            x = 15 + i * 43
            cv2.rectangle(image, (x, 35), (x + 28, 335), (20 + i * 15, 140, 200 - i * 10), -1)
            obs.append({"observation_id": f"{fid}-o{i:03d}", "frame_id": fid, "kind": "book", "bbox_xyxy": [x, 35, x + 28, 335]})
        cv2.imwrite(str(run / "frames" / f"{fid}.png"), image)
        frames.append({"frame_id": fid, "timestamp_s": k * 1.2, "width_px": 540, "height_px": 380,
                       "image_path": f"frames/{fid}.png", "quality": {"sharpness": 100 - k}})
    for i in range(12):
        groups.append({"instance_id": f"bk-{i:03d}", "observation_ids": [f"f{k:05d}-o{i:03d}" for k in range(3)]})
    uncertain = [{"observation_ids": [f"f00000-o{i:03d}", f"f00001-o{i + 2:03d}"], "reason": "Near-tie competing physical instance"} for i in range(10)]
    uncertain += [{"observation_ids": ["f00000-o000", "f00001-o000"], "reason": "same group"},
                  {"observation_ids": ["missing-item", "f00001-o000"], "reason": "item excluded"}]
    for name, data in (("manifest", {"frames": frames}), ("candidates", {"observations": obs}),
                       ("inventory", {"books": groups, "uncertain_associations": uncertain})):
        (run / f"{name}.json").write_text(json.dumps({"data": data}))
    pan = runs / "fixture_panorama_v1"
    pan.mkdir()
    (pan / "panorama_count.json").write_text(json.dumps({"merge_evidence": [{"panorama_spine": "spine-1", "groups": [{"id": "bk-000"}, {"id": "bk-002"}]}]}))
    monkeypatch.setattr(PL, "RUNS", runs)
    monkeypatch.setattr(PS, "RUNS", runs)
    return runs, run


@pytest.fixture()
def sheets(saved_run):
    runs, run = saved_run
    result = PS.generate("fixture", runs=runs)
    PL.write_summary("fixture")
    return runs, run, result


def test_union_deterministic_controls_and_all_provenance(saved_run):
    runs, _ = saved_run
    a = PS.candidates("fixture", runs=runs)
    b = PS.candidates("fixture", runs=runs)
    assert a["pairs"] == b["pairs"]
    assert a["pairs"] != PS.candidates("fixture", runs=runs, seed=3)["pairs"]
    assert len(a["pairs"]) == 14
    assert Counter(p.get("control_kind") for p in a["pairs"]) == {None: 10, "different": 3, "same": 1}
    combo = next(p for p in a["pairs"] if set(p["groups"]) == {"bk-000", "bk-002"})
    assert set(combo["source_kinds"]) == {"panorama_merge", "uncertain_association"}
    assert len(combo["provenance"]) == 2
    assert a["skipped"]["uncertain_association"] == 2
    # Different controls show different frames but retain the co-visible witness.
    for p in a["pairs"]:
        if p.get("control_kind") == "different":
            assert p["observation_ids"][0].split("-o")[0] != p["observation_ids"][1].split("-o")[0]
            witness = p["provenance"][0]["witness_observation_ids"]
            assert witness[0].split("-o")[0] == witness[1].split("-o")[0]


def test_native_pixels_contact_pages_blinding_and_preserve_run(sheets):
    runs, run, result = sheets
    root = runs / "pair_sheets/fixture"
    m, key = PS.read(root / "pairs.json"), PS.read(root / "pairs_key.json")
    assert result["pairs"] == 14
    assert len(m["contact_sheets"]) == 2
    assert len(m["contact_sheets"][0]["pair_ids"]) == 12
    assert len(m["contact_sheets"][1]["pair_ids"]) == 2
    assert "human spot-check 0 of 0" in (run / "pair_labels_summary.md").read_text()
    for p in m["pairs"]:
        assert "control_kind" not in p and "expected_answer" not in p
        for s in p["sightings"]:
            crop = cv2.imread(str(root / s["crop_path"]))
            frame = cv2.imread(str(run / f"frames/{s['frame_id']}.png"))
            x0, y0, x1, y1 = s["bbox_xyxy"]
            assert np.array_equal(crop, frame[y0:y1, x0:x1])
        image = cv2.imread(str(root / p["sheet_path"]))
        assert image.shape[:2] == (PS.SIZE[1], PS.SIZE[0])
        assert PS.sha(PL.image_file("fixture", pair_id=p["pair_id"])) == p["sheet_sha256"]
    assert any(v["control_kind"] for v in key["pairs"].values())
    blind = json.dumps(PL.blind_manifest("fixture"))
    assert all(word not in blind for word in ("bk-", "control", "source_kind", "reason", "answer"))
    assert not (run / "pair_labels").exists()
    before = PS.sha(run / "inventory.json")
    with pytest.raises(PS.PairError, match="already exist"):
        PS.generate("fixture", runs=runs)
    assert before == PS.sha(run / "inventory.json")


def _answers(pid, answer="same", reason="Visible markings agree."):
    return [{"pair_id": pid, "answer": answer, "reason": reason}]


def test_model_is_not_truth_human_subset_abstention_and_conflict(sheets):
    runs, run, _ = sheets
    key = PS.read(runs / "pair_sheets/fixture/pairs_key.json")["pairs"]
    controls = [pid for pid, k in key.items() if k["control_kind"]]
    a, b = controls[:2]
    before = PS.sha(run / "inventory.json")
    model_answers = _answers(a, key[a]["expected_answer"]) + _answers(b, "cannot_tell", "Too blurry.")
    result = PL.import_answers("fixture", model_answers, "chatgpt-test", "model")
    doc = PS.read(run / "pair_labels/chatgpt-test.json")
    assert doc["label_status"] == "model-read" and doc["purpose"] != "evaluation_truth"
    assert not doc["labels"][a]["human_confirmed"]
    assert len(doc["imports"][0]["source_sha256"]) == 64
    assert result["summary"]["human_labeled_pairs"] == 0
    c = result["summary"]["annotators"][0]["control_agreement"]
    assert c == {"answered": 2, "decisive": 1, "agree": 1, "abstentions": 1, "agreement_of_answered": .5, "agreement_of_decisive": 1.}
    PL.import_answers("fixture", _answers(a, key[a]["expected_answer"]), "human-one", "human")
    s = PL.summary("fixture")
    assert s["human_spot_checked_model_pairs"] == 1 and s["model_labeled_pairs"] == 2
    m = next(x for x in s["annotators"] if x["kind"] == "model")
    assert m["human_overlap"]["human_consensus_reference"]["agree"] == 1
    opposite = "same" if key[a]["expected_answer"] == "different" else "different"
    PL.import_answers("fixture", _answers(a, opposite), "human-two", "human")
    s = PL.summary("fixture")
    assert s["human_conflict_pairs"] == [a] and s["human_consensus_pairs"] == 0
    assert "human spot-check 1 of 2" in (run / "pair_labels_summary.md").read_text()
    assert before == PS.sha(run / "inventory.json")


def test_idempotency_conflicting_versions_changed_evidence_and_validation(sheets):
    runs, run, _ = sheets
    pid = PL.blind_manifest("fixture")["pairs"][0]["pair_id"]
    result = PL.import_answers("fixture", _answers(pid), "model-one", "model")
    label_path = run / "pair_labels/model-one.json"
    digest = PS.sha(label_path)
    assert PL.import_answers("fixture", _answers(pid), "model-one", "model")["imported"] == 0
    assert PS.sha(label_path) == digest
    with pytest.raises(PS.PairError, match="another kind"):
        PL.import_answers("fixture", _answers(pid), "model-one", "human")
    with pytest.raises(PS.PairError, match="conflicting"):
        PL.import_answers("fixture", _answers(pid, "different"), "model-one", "model")
    for annotator in ("../escape", "", "a/b", ".."):
        with pytest.raises(PS.PairError):
            PL.import_answers("fixture", _answers(pid), annotator, "model")
    for answers in (_answers("unknown"), _answers(pid, "yes"), _answers(pid, reason=""), _answers(pid, reason="one\ntwo"), _answers(pid) * 2):
        with pytest.raises(PS.PairError):
            PL.import_answers("fixture", answers, "invalid", "model")
    assert not (run / "pair_labels/invalid.json").exists()
    image = PL.image_file("fixture", pair_id=pid)
    image.write_bytes(b"changed")
    with pytest.raises(PS.PairError, match="changed"):
        PL.import_answers("fixture", _answers(pid), "model-other", "model")
    m = runs / "pair_sheets/fixture/pairs.json"
    m.write_text(m.read_text() + " ")
    with pytest.raises(PS.PairError, match="stale"):
        PL.summary("fixture")


def test_api_blinded_manifest_import_summary_and_no_key(sheets):
    from fastapi.testclient import TestClient
    from app import server
    client = TestClient(server.app, base_url="http://127.0.0.1:4182", headers={"Origin": "http://127.0.0.1:4182"})
    assert client.get("/api/pairs/runs").json()["runs"] == [{"run_id": "fixture", "pairs": 14}]
    manifest = client.get("/api/pairs/fixture/manifest").json()
    pid = manifest["pairs"][0]["pair_id"]
    assert "control" not in json.dumps(manifest)
    assert client.get(f"/api/pairs/fixture/sheet/{pid}").content.startswith(b"\x89PNG")
    assert client.get("/api/pairs/fixture/contact/1").status_code == 200
    assert client.get("/api/pairs/fixture/contact/0").status_code == 404
    assert client.get("/api/pairs/fixture/key").status_code == 404
    assert client.post("/api/pairs/fixture/import", json={"answers": _answers(pid), "annotator": "api-model"}).status_code == 400
    r = client.post("/api/pairs/fixture/import", json={"answers": json.dumps(_answers(pid)[0]), "annotator": "api-model", "kind": "model"})
    assert r.status_code == 200 and r.json()["imported"] == 1
    assert client.get("/api/pairs/fixture/summary").json()["human_labeled_pairs"] == 0
    assert client.get("/pairs.js").status_code == 200


def test_jsonl_and_paths(saved_run):
    runs, run = saved_run
    for text in ('[]', '{bad}', '"text"'):
        with pytest.raises(PS.PairError):
            PL.parse_jsonl(text)
    assert PL.parse_jsonl('\n{"pair_id":"pair-00001"}\n') == [{"pair_id": "pair-00001"}]
    with pytest.raises(PS.PairError):
        PS.run_dir("../fixture", runs)
    alias = runs / "alias"
    alias.symlink_to(run, target_is_directory=True)
    with pytest.raises(PS.PairError):
        PS.run_dir("alias", runs)
    (run / "frames/link.png").symlink_to(run / "frames/f00000.png")
    with pytest.raises(PS.PairError):
        PS.safe_file(run, "frames/link.png")


def test_cli_import_and_tampered_provenance(sheets, monkeypatch, capsys, tmp_path):
    import hashlib
    import sys
    _, run, _ = sheets
    pid = PL.blind_manifest("fixture")["pairs"][0]["pair_id"]
    answers = tmp_path / "answers.jsonl"
    payload = json.dumps(_answers(pid)[0]).encode() + b"\n"
    answers.write_bytes(payload)
    monkeypatch.setattr(sys, "argv", ["pair_labels", "import", "--run", "fixture", "--answers", str(answers),
                                      "--annotator", "cli-model", "--kind", "model"])
    PL.main()
    assert json.loads(capsys.readouterr().out)["imported"] == 1
    path = run / "pair_labels/cli-model.json"
    doc = PS.read(path)
    assert doc["imports"][0]["source_sha256"] == hashlib.sha256(payload).hexdigest()
    doc["labels"][pid]["human_confirmed"] = True
    path.write_text(json.dumps(doc))
    with pytest.raises(PS.PairError, match="provenance"):
        PL.summary("fixture")


def test_confirmed_live_pairs_are_anchored():
    if not (Path("runs") / PS.DEFAULT_RUN / "inventory.json").is_file():
        pytest.skip("workspace live artifacts absent")
    # Explicit runs prevents the synthetic fixtures from affecting the live gate.
    selected = PS.candidates(PS.DEFAULT_RUN, runs=Path("runs"))
    actual = {frozenset(p["observation_ids"]) for p in selected["pairs"]}
    assert all(frozenset(ids) in actual for ids in PS.CONFIRMED_ANCHORS)
    for ids, gids in zip(PS.CONFIRMED_ANCHORS, ({"bk-687", "bk-713"}, {"bk-682", "bk-719"})):
        p = next(p for p in selected["pairs"] if set(p["observation_ids"]) == set(ids))
        assert set(p["groups"]) == gids and "audit_break" in p["source_kinds"]


def test_production_does_not_consume_pair_labels():
    for folder in ("pipeline", "app"):
        for path in Path(folder).glob("*.py"):
            # These app modules expose review-only import/report tools. The
            # pipeline and live engine must never consume pair labels.
            if folder == "app" and path.name in {"pair_labels.py", "server.py"}:
                continue
            text = path.read_text()
            assert "import pair_labels" not in text and ' / "pair_labels"' not in text, path
    script = Path("app/static/pairs.js").read_text()
    assert len(script.splitlines()) < 60
    assert "innerHTML" not in script
