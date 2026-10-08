"""Ground-truth labelling page: loopback only (404 on the LAN server), answers saved outside runs/, and the answers land in the
yellow cells of the ground truth and results workbook (scripts/ground_truth_sheet.py) with a Results preview that matches its formulas."""
import json
import os

import pytest
from fastapi.testclient import TestClient

openpyxl = pytest.importorskip("openpyxl")

from app import ground_truth_page as GT  # noqa: E402
from app import server as s  # noqa: E402
from tests.test_lan_access import IP, KEY, lan, phone  # noqa: E402,F401  (fixture)

RID = "sweep_0123456789abcdef"
LOCAL = "http://127.0.0.1:4182"
HDR = {"origin": LOCAL}


def _packet(rid=RID, n=40):
    books = []
    for i in range(n):
        b = {"id": f"bk-{i:03d}", "display_index": i + 1, "status": "identified" if i % 2 == 0 else "unidentified",
             "title": f"Title {i}" if i % 2 == 0 else None, "author": "A" if i % 2 == 0 else None, "id_confidence": 0.9 if i % 2 == 0 else None,
             "frame_ref": "frames/f00000.jpg", "timestamp_s": float(i),
             "evidence": {"frame_ref": "frames/f00000.jpg", "bbox_xyxy": [10, 10, 30, 80], "polygon_xy": [[10, 10], [30, 10], [30, 80], [10, 80]],
                          "frame_width_px": 160, "frame_height_px": 90},
             "spine_height_cm": 20.0 + i % 3, "spine_thickness_cm": 2.0}
        if i % 2 == 0:
            b["replacement_cost"] = {"amount": 10.0 + i, "currency": "USD", "source": "AbeBooks seller", "url": f"https://www.abebooks.com/x?isbn={i}",
                                     "retrieved_at": "2026-10-08T03:00:00+00:00", "isbn": f"978000000{i:04d}", "converted": False}
        books.append(b)
    items = [{"id": f"it-{j}", "display_index": j + 1, "category": ["plant", "frame", "vase"][j % 3], "status": "needs_appraisal", "confidence": 0.5,
              "frame_ref": "frames/f00000.jpg", "evidence": {"frame_ref": "frames/f00000.jpg", "bbox_xyxy": [40, 20, 90, 70], "polygon_xy": None}}
             for j in range(6)]
    return {"status": "ok", "data": {
        "sweep": {"id": rid, "captured_at": "2026-10-08T05:00:00Z", "duration_s": 300, "country": "US", "currency": "USD",
                  "finalization_timings": {"packet": 100.0, "report": 20.5}},
        "books": books, "items": items, "room": {"status": "abstained"},
        "totals": {"book_count": 42, "book_lines": n, "books_identified": n // 2, "books_unidentified": n - n // 2}}}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for k in ("LIBRARY_LAN_MODE", "LIBRARY_LAN_HOSTS", "LIBRARY_LAN_TOKEN", "LIBRARY_LAN_CA"):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "runs"
    d = root / RID
    (d / "frames").mkdir(parents=True)
    (d / "frames" / "f00000.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (d / "claim_packet.json").write_text(json.dumps(_packet()))
    sub = tmp_path / "ground_truth"
    monkeypatch.setattr(s, "RUNS_DIR", root)
    monkeypatch.setenv("LIBRARY_GROUND_TRUTH_DIR", str(sub))
    monkeypatch.setattr(GT, "DEFAULT_RUN", RID)
    return {"runs": root, "run": d, "sub": sub}


def _snapshot(d):
    return sorted((str(p.relative_to(d)), p.stat().st_mtime_ns, p.stat().st_size) for p in d.rglob("*"))


def test_page_and_cards_on_the_loopback_server(env):
    c = TestClient(s.app, base_url=LOCAL)
    for path in ("/truth", "/truth.js", "/truth.css"):
        assert c.get(path).status_code == 200, path
    assert c.get("/api/truth/default").json() == {"run": RID}
    r = c.get(f"/api/truth/{RID}").json()
    assert len(r["books"]) == 40 and len(r["items"]) == 6 and r["categories"] == ["frame", "plant", "vase"] and r["labels"] == {}
    mod = GT.sheet_module()
    sample, n = mod.price_sample(mod.sorted_books(_packet()["data"]))
    assert [p["id"] for p in r["prices"]] == [b["id"] for b in sample] and len(sample) == 15 and r["priced"] == n == 20
    b0 = r["books"][0]
    assert b0["frame"] == "f00000.jpg" and b0["bbox"] == [10.0, 10.0, 30.0, 80.0] and len(b0["polygon"]) == 4
    assert r["sheet_file"] == "ground_truth_and_results.xlsx" and r["excel_open"] is False
    assert c.get("/api/truth/sweep_ffffffffffffffff").status_code == 404 and c.get("/api/truth/../etc").status_code == 404


def test_lan_server_answers_404_for_every_truth_path(env, lan):  # noqa: F811
    paired = phone()
    assert paired.get(f"/?k={KEY}", follow_redirects=False).status_code == 303
    gets = ["/truth", "/truth.js", "/truth.css", "/api/truth/default", f"/api/truth/{RID}"]
    posts = [f"/api/truth/{RID}/labels", f"/api/truth/{RID}/sheet", f"/api/truth/{RID}/sheet?force=1"]
    for client in (paired, phone(), TestClient(s.app, base_url="https://127.0.0.1:4183")):
        for path in gets:
            assert client.get(path, follow_redirects=False).status_code == 404, path
        for path in posts:
            assert client.post(path, json={"titles": {}}, headers={"origin": f"https://{IP}:4183"}).status_code == 404, path
    assert not (env["sub"] / GT.LABELS_NAME).exists()


def test_only_loopback_host_and_client(env):
    c = TestClient(s.app, base_url=LOCAL)
    assert c.get("/truth", headers={"host": "evil.example"}).status_code == 404
    assert TestClient(s.app, base_url=LOCAL, client=("192.168.1.9", 5000)).post(f"/api/truth/{RID}/labels", json={}, headers=HDR).status_code == 404
    assert c.post(f"/api/truth/{RID}/labels", json={"true_item_count": 3}).status_code == 403      # no Origin: refused like every other POST


def test_labels_are_validated_and_saved_outside_runs(env):
    c = TestClient(s.app, base_url=LOCAL)
    before = _snapshot(env["run"])
    url = f"/api/truth/{RID}/labels"
    for bad in ({"titles": {"bk-nope": {"hand_title": "x"}}}, {"titles": {"bk-000": {"correct": "maybe"}}},
                {"prices": {"bk-001": {"price": 3}}}, {"prices": {"bk-000": {"price": "abc"}}}, {"prices": {"bk-000": {"url": "javascript:alert(1)"}}},
                {"items": {"it-0": {"real": "Y", "true_category": "x" * 81}}}, {"true_item_count": -1}, {"missed_titles": [{"title": "t", "frame": "../x"}]},
                {"nonsense": 1}):
        assert c.post(url, json=bad, headers=HDR).status_code == 400, bad
    assert c.post(url, json={"titles": {"bk-000": {"hand_title": "Title 0", "correct": "Y", "legible": True}}}, headers=HDR).json()["saved"]
    assert c.post(url, json={"titles": {"bk-002": {"hand_title": "Real", "correct": "N", "legible": True}}, "position": {"titles": "bk-002"}},
                  headers=HDR).status_code == 200
    doc = json.loads((env["sub"] / GT.LABELS_NAME).read_text())
    cur = doc["sweeps"][RID]
    assert set(cur["titles"]) == {"bk-000", "bk-002"} and cur["position"] == {"titles": "bk-002", "items": None}       # per-line merge
    assert c.post(url, json={"titles": {"bk-002": None}}, headers=HDR).status_code == 200                               # null removes an answer
    assert set(c.get(f"/api/truth/{RID}").json()["labels"]["titles"]) == {"bk-000"}
    assert not list(env["sub"].glob(".*.tmp")) and oct((env["sub"] / GT.LABELS_NAME).stat().st_mode & 0o777) == "0o644"
    assert _snapshot(env["run"]) == before                                                                              # nothing under runs/ changed


def test_labels_round_trip_into_the_workbook(env):
    c = TestClient(s.app, base_url=LOCAL)
    before = _snapshot(env["run"])
    sheet = env["sub"] / GT.SHEET_NAME
    # the workbook already exists with values typed straight into it (spines, room tape): a rebuild keeps them
    GT.sheet_module().build(env["run"] / "claim_packet.json", sheet)
    wb = openpyxl.load_workbook(sheet)
    wb["Spines"]["G2"] = 25.0; wb["Spines"]["H2"] = 2.0
    wb["Room"]["B2"] = 4.0; wb["Room"]["B3"] = 3.0; wb["Room"]["B4"] = 2.5
    wb["Titles"]["H5"] = "Typed in Excel"; wb["Titles"]["I5"] = "Y"           # bk-003: the page has no answer for it
    wb.save(sheet)
    titles = {"bk-000": {"hand_title": "Title 0", "correct": "Y", "legible": True},          # identified, correct
              "bk-002": {"hand_title": "Other", "correct": "N", "legible": True},            # identified, confidently wrong
              "bk-001": {"hand_title": "Read by hand", "correct": "N", "legible": True},     # unidentified line, legible
              "bk-004": {"hand_title": "", "correct": None, "legible": False}}              # not legible
    mod = GT.sheet_module()
    sample = [b["id"] for b in mod.price_sample(mod.sorted_books(_packet()["data"]))[0]]
    patch = {"titles": titles,
             "missed_titles": [{"title": "Missed One", "frame": "f00012", "notes": "top shelf"}],
             "items": {"it-0": {"real": "Y", "true_category": "plant", "category_correct": "Y"},
                       "it-1": {"real": "Y", "true_category": "mirror", "category_correct": "N"},
                       "it-2": {"real": "N", "true_category": "", "category_correct": None}},
             "missed_items": [{"category": "lamp", "frame": "", "notes": ""}], "true_item_count": 4,
             "prices": {sample[0]: {"price": 12.5, "url": "https://example.org/a", "date": "2026-10-08"}}}
    assert c.post(f"/api/truth/{RID}/labels", json=patch, headers=HDR).status_code == 200
    r = c.post(f"/api/truth/{RID}/sheet", headers=HDR)
    assert r.status_code == 200, r.text
    res = {x["measure"]: x for x in r.json()["results"]}
    wb = openpyxl.load_workbook(sheet)
    ti = wb["Titles"]
    rows = {ti.cell(row=k, column=2).value: k for k in range(2, 42)}
    assert ti.cell(rows["bk-000"], 8).value == "Title 0" and ti.cell(rows["bk-000"], 9).value == "Y"
    assert ti.cell(rows["bk-002"], 9).value == "N" and ti.cell(rows["bk-004"], 8).value is None
    assert ti.cell(rows["bk-003"], 8).value == "Typed in Excel"                                     # carried over
    missed_row = next(k for k in range(42, ti.max_row + 1) if ti.cell(k, 8).value == "Missed One")
    assert ti.cell(missed_row, 4).value == "missed" and ti.cell(missed_row, 9).value == "N" and ti.cell(missed_row, 12).value == "top shelf"
    assert wb["Spines"]["G2"].value == 25.0 and wb["Room"]["B2"].value == 4.0
    pr = wb["Prices"]
    assert pr["B2"].value == sample[0] and pr["J2"].value == 12.5 and pr["K2"].value == "https://example.org/a" and pr["L2"].value == "2026-10-08"
    im = wb["Items"]
    assert im["E1"].value == 4 and [im.cell(4, c).value for c in (7, 8, 9)] == ["Y", "plant", "Y"] and im.cell(6, 7).value == "N"
    assert any(im.cell(k, 8).value == "lamp" for k in range(10, im.max_row + 1))
    # Results preview == the formulas' arithmetic on the same cells
    legible = 5                                       # Title 0, Other, Read by hand, Typed in Excel, Missed One
    assert res["Title identification"]["ground_truth"] == legible and res["Title identification"]["system"] == 2
    assert res["Title identification"]["score"] == pytest.approx(2 / 5) and res["Title identification"]["result"] == "FAIL"
    assert res["Title identification"]["note"].startswith("confidently wrong 1")
    assert res["Book count"]["score"] == pytest.approx(abs(42 - 458) / 458) and res["Book count"]["result"] == "FAIL"
    sys_h = wb["Spines"]["E2"].value
    assert res["Spine dimensions"]["score"] == pytest.approx(max(abs(sys_h - 25) / 25, 0.0)) and res["Spine dimensions"]["ground_truth"] == 1
    assert res["Book prices"]["score"] == pytest.approx(abs(pr["E2"].value - 12.5) / 12.5) and res["Book prices"]["ground_truth"] == 1
    assert res["Non-book items"]["system"] == 1 and res["Non-book items"]["score"] == pytest.approx(0.25) and res["Non-book items"]["result"] == "FAIL"
    assert res["Floor area"]["ground_truth"] == pytest.approx(12.0) and res["Floor area"]["result"] == "FAIL (system abstained)"
    assert res["Wall area"]["ground_truth"] == pytest.approx(35.0)
    assert res["Time to packet"]["system"] == 120.5 and res["Time to packet"]["result"] == "PASS"
    assert wb["Results"]["C3"].value.startswith("=COUNTA(Titles!$H$2:$H$")
    # rebuilding again changes nothing (no duplicated missed rows, carried values stay)
    assert c.post(f"/api/truth/{RID}/sheet", headers=HDR).json()["results"] == r.json()["results"]
    assert c.get(f"/api/truth/{RID}").json()["labels"]["last_sheet"]["results"] == r.json()["results"]      # shown again on the next visit
    wb2 = openpyxl.load_workbook(sheet)
    assert sum(1 for k in range(2, wb2["Titles"].max_row + 1) if wb2["Titles"].cell(k, 8).value == "Missed One") == 1
    assert oct(sheet.stat().st_mode & 0o777) == "0o644" and not list(env["sub"].glob(".*.tmp"))
    assert _snapshot(env["run"]) == before


def test_excel_lock_refuses_unless_forced_and_other_sweeps_get_their_own_file(env, monkeypatch):
    c = TestClient(s.app, base_url=LOCAL)
    env["sub"].mkdir(parents=True, exist_ok=True)
    (env["sub"] / f"~${GT.SHEET_NAME}").write_text("lock")
    r = c.post(f"/api/truth/{RID}/sheet", headers=HDR)
    assert r.status_code == 409 and "Excel" in r.json()["detail"] and not (env["sub"] / GT.SHEET_NAME).exists()
    assert c.get(f"/api/truth/{RID}").json()["excel_open"] is True
    assert c.post(f"/api/truth/{RID}/sheet?force=1", headers=HDR).status_code == 200 and (env["sub"] / GT.SHEET_NAME).exists()
    monkeypatch.setattr(GT, "DEFAULT_RUN", "sweep_ffffffffffffffff")                 # not the demo sweep: never overwrites the deliverable
    assert GT.sheet_path(RID).name == f"ground_truth_and_results_{RID}.xlsx"


def test_script_cli_reads_the_labels_file(env, tmp_path):
    from scripts import ground_truth_sheet as mod
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"version": 1, "sweeps": {RID: {"titles": {"bk-000": {"hand_title": "Title 0", "correct": "Y", "legible": True}},
                                                               "true_item_count": 7}, "sweep_other": {"true_item_count": 99}}}))
    out = tmp_path / "out.xlsx"
    mod.main([str(env["run"] / "claim_packet.json"), str(out), "--labels", str(labels)])
    wb = openpyxl.load_workbook(out)
    assert wb["Items"]["E1"].value == 7 and wb["Titles"]["H2"].value == "Title 0" and wb["Titles"]["I2"].value == "Y"
    assert os.path.getsize(out) > 0
