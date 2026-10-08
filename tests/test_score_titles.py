import json
from pathlib import Path

import pytest

from app import score_titles as ST


def test_compare_rules_keep_accents_and_partial():
    assert ST.compare("Cien años de soledad", "CIEN AÑOS DE SOLEDAD")["exact"]
    assert not ST.compare("Cien años de soledad", "Cien anos de soledad")["exact"]      # no diacritic folding: literal
    k = ST.compare("Don Quijote de la Mancha", "Quijote de la Mancha")
    assert k["recall"] and not k["exact"]
    assert ST.compare("Dune", "Dune Messiah")["contained"]
    assert not ST.compare("Dune", "xyz")["contained"]


@pytest.fixture(autouse=True)
def _runs_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "ROOT", tmp_path)          # label crops are verified under ROOT/runs/<label run>; never touch the real runs/


def _truth(tmp, rows, labels, run_id="r", cands_sha="x", name="t.json", write_crops=True):
    """Label file whose samples carry real crop files + sha256 under tmp/runs/<run_id> (synthetic fixtures, never human truth)."""
    from hashlib import sha256
    crops = tmp / "runs" / run_id / "crops"; crops.mkdir(parents=True, exist_ok=True)
    sample = []
    for r in rows:
        data = f"crop-{run_id}-{r['observation_id']}".encode()
        if write_crops:
            (crops / f"{r['observation_id']}.png").write_bytes(data)
        sample.append({**r, "crop_path": f"crops/{r['observation_id']}.png", "crop_sha256": sha256(data).hexdigest()})
    p = tmp / name
    p.write_text(json.dumps({"purpose": "evaluation_truth", "kind": "title_labels", "run_id": run_id, "annotator": "a", "scan": {"sha256": "scan-x"},
                             "source_hashes": {"candidates_sha256": cands_sha}, "sample": sample, "labels": labels}))
    return p


def _run(tmp, texts):
    r = tmp / "run"; r.mkdir()
    obs = [{"observation_id": f"f1-o{i}", "kind": "book", "frame_id": "f1", "bbox_xyxy": [10 * i, 0, 10 * i + 8, 100]} for i in range(len(texts))]
    (r / "candidates.json").write_text(json.dumps({"data": {"observations": obs}}))
    (r / "manifest.json").write_text(json.dumps({"data": {"frames": [{"frame_id": "f1", "timestamp_s": 1.0}], "input": {"sha256": "scan-x"}}}))
    (r / "readings.json").write_text(json.dumps({"data": {"readings": [{"observation_id": f"f1-o{i}", "title_text": t, "read_confidence": 0.9, "title_complete": True} for i, t in enumerate(texts)]}}))
    (r / "resolutions.json").write_text(json.dumps({"data": {"resolutions": [{"observation_id": f"f1-o{i}", "decision": "no_match", "title": None} for i in range(len(texts))]}}))
    return r


def test_scoring_matches_by_geometry_and_counts_confident_wrong(tmp_path):
    rows = [{"observation_id": f"x-o{i}", "frame_id": "f9", "timestamp_s": 1.02, "bbox_xyxy": [10 * i, 0, 10 * i + 8, 100]} for i in range(4)]
    labels = {r["observation_id"]: {"status": "labeled", "title": t, "language": "es"} for r, t in zip(rows, ["Dune", "Rayuela", "Ficciones", "Pedro Páramo"])}
    labels["x-o3"]["language"] = "en"
    run = _run(tmp_path, ["DUNE", "", "Otra cosa", "Pedro Páramo"])
    doc = ST.build([_truth(tmp_path, rows, labels)], run)
    es, en = doc["layers"]["reading"]["es"], doc["layers"]["reading"]["en"]
    assert es["counts"] == {"n": 3, "exact": 1, "contained": 1, "recall": 1, "abstained": 1, "confidently_wrong": 1, "unmatched": 0}
    assert en["exact_title"]["value"] == 1.0 and doc["layers"]["reading"]["other"]["exact_title"]["available"] is False
    assert doc["layers"]["resolution"]["es"]["counts"]["abstained"] == 3     # no_match => abstained


def test_no_labels_is_unavailable_not_zero(tmp_path):
    p = _truth(tmp_path, [{"observation_id": "x"}], {})
    doc = ST.build([p], _run(tmp_path, ["a"]))
    assert doc["status"] == "unavailable" and doc["layers"]["reading"]["all"]["exact_title"]["value"] is None


def test_rejects_non_truth_file(tmp_path):
    p = tmp_path / "x.json"; p.write_text(json.dumps({"purpose": "calibration"}))
    with pytest.raises(ValueError):
        ST.load_truth([p])


def test_dev_smoke_is_marked_dev_and_never_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "ROOT", Path(__file__).resolve().parents[1])             # real workspace runs/ (the autouse fixture points ROOT at tmp)
    dev = Path("eval/dev_labels/model_read_yt1_v1.json")
    run = Path("runs/final_candidate_yt1_v2")
    if not dev.is_file() or not (run / "readings.json").is_file():
        pytest.skip("workspace dev labels / run absent")
    doc = ST.build([], run, dev_smoke=dev)
    assert doc["status"] == "dev_smoke" and doc["kind"] == "title_score_DEV_SMOKE" and "NOT human truth" in doc["truth"]["warning"]
    assert doc["layers"]["reading"]["es"]["exact_title"]["available"] is False          # dev labels carry no language
    assert doc["layers"]["reading"]["all"]["counts"]["n"] > 0
    with pytest.raises(ValueError):
        ST.load_dev_labels(Path("eval/dev_labels/../shelf_facts.json"))


def test_strict_confidently_wrong_and_sha_rules(tmp_path):
    rows = [{"observation_id": "x-o0", "frame_id": "f9", "timestamp_s": 1.02, "bbox_xyxy": [0, 0, 8, 100]}]
    labels = {"x-o0": {"status": "labeled", "title": "Dune", "language": "en"}}
    run = _run(tmp_path, ["Dune Messiah"])                                  # asserted full title, confidence 0.9 >= 0.85, contains the truth
    doc = ST.build([_truth(tmp_path, rows, labels)], run)
    en = doc["layers"]["reading"]["en"]["counts"]
    assert en["contained"] == 1 and en["exact"] == 0 and en["confidently_wrong"] == 1   # containment does not rescue a wrong asserted title
    low = tmp_path / "low"; low.mkdir()
    for f in ("candidates.json", "manifest.json", "resolutions.json"):
        (low / f).write_text((run / f).read_text())
    (low / "readings.json").write_text(json.dumps({"data": {"readings": [{"observation_id": "f1-o0", "title_text": "Dune Messiah", "read_confidence": 0.6, "title_complete": True}]}}))
    assert ST.build([_truth(tmp_path, rows, labels)], low)["layers"]["reading"]["en"]["counts"]["confidently_wrong"] == 0   # below the 0.85 cutoff: wrong, not confident
    p = tmp_path / "nohash.json"
    p.write_text(json.dumps({"purpose": "evaluation_truth", "kind": "title_labels", "run_id": "r", "annotator": "a", "scan": {}, "source_hashes": {}, "sample": rows, "labels": labels}))
    with pytest.raises(ValueError):
        ST.load_truth([p])
    assert ST.CONF_MIN == 0.85


def _run_with(tmp, texts, res_extra=None, conf=0.9, resolutions=None):
    r = _run(tmp, texts)
    rd = json.loads((r / "readings.json").read_text())
    for x in rd["data"]["readings"]:
        x["read_confidence"] = conf
    (r / "readings.json").write_text(json.dumps(rd))
    if resolutions is not None:
        (r / "resolutions.json").write_text(json.dumps({"data": {"resolutions": resolutions}}))
    return r


def _one_row(oid="x-o0", title="Dune"):
    return ([{"observation_id": oid, "frame_id": "f9", "timestamp_s": 1.02, "bbox_xyxy": [0, 0, 8, 100]}], {oid: {"status": "labeled", "title": title, "language": "en"}})


def test_accepted_catalogue_title_needs_the_085_confidence_threshold(tmp_path):
    rows, labels = _one_row()
    acc = [{"observation_id": "f1-o0", "decision": "accepted", "title": "Dune Messiah"}]
    hi = ST.build([_truth(tmp_path, rows, labels)], _run_with(tmp_path, ["x"], resolutions=acc, conf=0.9))
    assert hi["layers"]["resolution"]["en"]["counts"]["confidently_wrong"] == 1          # accepted + confident + wrong
    lo_dir = tmp_path / "lo"; lo_dir.mkdir()
    lo = ST.build([_truth(tmp_path, rows, labels)], _run_with(lo_dir, ["x"], resolutions=acc, conf=0.84))
    c = lo["layers"]["resolution"]["en"]["counts"]
    assert c["confidently_wrong"] == 0 and c["exact"] == 0 and c["abstained"] == 0        # wrong, but below the cutoff: not "confident"
    edge_dir = tmp_path / "edge"; edge_dir.mkdir()
    edge = ST.build([_truth(tmp_path, rows, labels)], _run_with(edge_dir, ["x"], resolutions=acc, conf=0.85))
    assert edge["layers"]["resolution"]["en"]["counts"]["confidently_wrong"] == 1        # the cutoff itself is inclusive
    nc_dir = tmp_path / "nc"; nc_dir.mkdir()
    nc = ST.build([_truth(tmp_path, rows, labels)], _run_with(nc_dir, ["x"], resolutions=[{**acc[0], "confidence": 0.5}], conf=0.95))
    assert nc["layers"]["resolution"]["en"]["counts"]["confidently_wrong"] == 0          # the resolution's own lower confidence wins


def test_matching_is_keyed_by_artifact_and_observation_id_not_bare_id(tmp_path):
    # two label artifacts reuse the SAME observation id; both must be scored independently against the run they share a hash with
    run = _run_with(tmp_path, ["Dune", "Rayuela"])
    from hashlib import sha256
    sha = sha256((run / "candidates.json").read_bytes()).hexdigest()
    ra, la = _one_row("f1-o0", "Dune"); rb, lb = _one_row("f1-o0", "Dune")
    a = _truth(tmp_path, ra, la, run_id="ra", cands_sha=sha, name="a.json")
    b = _truth(tmp_path, rb, lb, run_id="rb", cands_sha="other-artifact", name="b.json")
    rows, info = ST.load_truth([a, b])
    assert len(rows) == 2 and len({ST.truth_key(r) for r in rows}) == 2
    m = ST.match(rows, ST.load_run(run))
    by_art = {r["label_run"]: m[ST.truth_key(r)] for r in rows}
    assert by_art["ra"] == "f1-o0"                                    # identical artifact hash: id join
    assert by_art["rb"] is None                                       # different artifact hash and no shared scan evidence: never joined by bare id
    doc = ST.build([a, b], run)
    assert doc["layers"]["reading"]["all"]["counts"]["n"] == 2
    assert doc["matching"] == {"matched": sum(v is not None for v in m.values()), "labeled": 2}


def test_bare_observation_id_collision_does_not_overwrite(tmp_path):
    run = _run_with(tmp_path, ["Dune", "Rayuela"])
    from hashlib import sha256
    sha = sha256((run / "candidates.json").read_bytes()).hexdigest()
    ra, la = _one_row("f1-o0", "Dune"); rb, lb = _one_row("f1-o0", "Ficciones")
    a = _truth(tmp_path, ra, la, run_id="ra", cands_sha=sha, name="a.json")
    b = _truth(tmp_path, rb, lb, run_id="rb", cands_sha=sha, name="b.json")       # same observation id, same candidates hash, different run label: conflicting duplicate
    # same artifact hash but different label_run => distinct keys (artifact = run + hash); both rows survive and each keeps its own truth title
    rows, _ = ST.load_truth([a, b])
    assert sorted(r["truth_title"] for r in rows) == ["Dune", "Ficciones"]
    m = ST.match(rows, ST.load_run(run))
    assert len(m) == 2 and all(v == "f1-o0" for v in m.values())


def test_same_artifact_conflicting_duplicate_label_is_rejected(tmp_path):
    ra, la = _one_row("o1", "Dune"); rb, lb = _one_row("o1", "Ficciones")
    a = _truth(tmp_path, ra, la, name="a.json"); b = _truth(tmp_path, rb, lb, name="b.json")
    with pytest.raises(ValueError):
        ST.load_truth([a, b])


def test_missing_source_crop_makes_truth_integrity_unavailable(tmp_path):
    rows, labels = _one_row()
    doc = ST.build([_truth(tmp_path, rows, labels, write_crops=False)], _run_with(tmp_path, ["Dune"]))
    assert doc["status"] == "unavailable" and "integrity" in doc["reason"]
    assert doc["truth"]["integrity"] == {"status": "unavailable", "verified": 0, "failed": 1}
    assert doc["truth"]["crop_missing"] == 1 and doc["truth"]["integrity_failures"][0]["problem"] == "source crop missing"
    assert doc["layers"]["reading"]["all"]["exact_title"]["value"] is None


def test_changed_source_crop_is_unavailable_not_silently_dropped(tmp_path):
    rows, labels = _one_row()
    p = _truth(tmp_path, rows, labels)
    (tmp_path / "runs" / "r" / "crops" / "x-o0.png").write_bytes(b"tampered")
    loaded, info = ST.load_truth([p])
    assert loaded == [] and info["sha_mismatch"] == 1 and info["integrity"]["status"] == "unavailable"
    doc = ST.build([p], _run_with(tmp_path, ["Dune"]))
    assert doc["status"] == "unavailable" and doc["truth"]["sha_mismatch"] == 1
    assert "layers" in doc and doc["layers"]["reading"]["all"]["exact_title"]["available"] is False


def test_crop_without_declared_hash_is_unverifiable(tmp_path):
    rows, labels = _one_row()
    p = _truth(tmp_path, rows, labels)
    d = json.loads(p.read_text()); [s.pop("crop_sha256") for s in d["sample"]]; p.write_text(json.dumps(d))
    _, info = ST.load_truth([p])
    assert info["crop_unverifiable"] == 1 and info["integrity"]["status"] == "unavailable"


def test_verified_crops_report_integrity_verified(tmp_path):
    rows, labels = _one_row()
    doc = ST.build([_truth(tmp_path, rows, labels)], _run_with(tmp_path, ["Dune"]))
    assert doc["status"] == "scored" and doc["truth"]["integrity"] == {"status": "verified", "verified": 1, "failed": 0}


def test_missing_candidates_json_is_unavailable_not_a_crash(tmp_path):
    rows, labels = _one_row()
    run = tmp_path / "emptyrun"; run.mkdir()
    doc = ST.build([_truth(tmp_path, rows, labels)], run)
    assert doc["status"] == "unavailable" and "candidates.json" in doc["reason"]
    assert doc["layers"]["reading"]["all"]["exact_title"]["value"] is None
