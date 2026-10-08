"""Unified site plumbing: scan registry, mounted debugger, session/scan identity, build info. Mocked stages."""
import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import frame_browser as FB
from app import server as s
from pipeline import contracts as C
from tests.test_live_app import Harness, feed_panning, make_frame
from tests.test_live_measurement import STAGES

H = {"origin": "http://127.0.0.1:4182"}


def client():
    return TestClient(s.app, base_url="http://127.0.0.1:4182")


def test_scan_registry_is_exactly_the_two_canonical_scans_without_aliases_or_previews():
    d = client().get("/api/scans").json()
    labels = [x["label"] for x in d["scans"]]
    assert labels == ["IESTest3", "YugoTest1", "YugoTest2"]
    assert all(len(x["sha256"]) == 64 for x in d["scans"])
    blob = json.dumps([{k: v for k, v in x.items() if k not in ("aliases", "original_name", "browser_preview", "file")} for x in d["scans"]])
    assert "IMG_" not in blob and "h264" not in blob          # aliases / derivatives are metadata of a scan, never listed as scans


needs_previews = pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "sweeps" / "YugoTest2_preview_h264.mp4").is_file(),
                                    reason="workspace video previews are not part of a clean checkout")
needs_saved_runs = pytest.mark.skipif(not list((Path(__file__).resolve().parents[1] / "runs").glob("yugotest*panorama*/panorama.json")),
                                      reason="saved panorama runs are not part of a clean checkout")


@needs_previews
def test_registered_preview_is_served_and_unknown_label_is_404():
    c = client()
    ok = c.get("/replay/scan/YugoTest2")
    assert ok.status_code in (200, 206) and ok.headers["content-type"].startswith("video/")
    assert c.get("/replay/scan/IMG_2281").status_code == 404 and c.get("/replay/scan/..%2Fsecret").status_code in (404, 422)


@needs_saved_runs
def test_debugger_is_mounted_same_origin_with_working_static_and_api_urls():
    c = client()
    assert c.get("/debug", follow_redirects=False).headers["location"] == "/debug/"
    page = c.get("/debug/")
    assert page.status_code == 200 and 'href="static/debug-style.css"' in page.text and 'src="static/debug-app.js"' in page.text   # relative: resolves under /debug/
    assert c.get("/debug/static/debug-app.js").status_code == 200 and c.get("/debug/static/debug-style.css").status_code == 200
    registered = lambda rows: [x["label"] for x in rows if not x.get("synthetic")]   # runs without a registered recording are listed under one extra synthetic group
    assert registered(c.get("/debug/api/scans").json()) == ["IESTest3", "YugoTest1", "YugoTest2"]
    scans = c.get("/debug/api/scans?all=1").json()   # default lists featured runs only (app.run_visibility); panorama runs are experiments
    assert registered(scans) == ["IESTest3", "YugoTest1", "YugoTest2"]
    # an asset URL under the prefix for a real saved run resolves (listed evidence only)
    run = next(r for x in scans for r in x["runs"] if r.get("has_panorama"))
    sc = next(x for x in scans if run in x["runs"])
    view = c.get(f"/debug/api/view/{sc['label']}/{run['run_id']}").json()
    token = next(f["asset"] for f in view["frames"] if f.get("asset"))
    assert c.get(f"/debug/api/assets/{run['run_id']}/{token}").status_code == 200
    assert c.get("/debug/api/assets/%2e%2e/etc").status_code in (404, 422)


def test_build_info_distinguishes_loaded_backend_from_static_files():
    b = client().get("/api/build").json()
    assert b["pid"] > 0 and b["started_at"] and len(b["backend_fingerprint_loaded"]) == 12 and len(b["static_fingerprint_current"]) == 12
    assert b["routes"] == {"measurement": True, "scans": True, "debug_mount": True, "frames": True}
    assert b["backend_source_changed_since_start"] in (True, False) and "restart" in b["note"]
    assert client().get("/api/health").json()["build"]["pid"] == b["pid"]


@pytest.mark.asyncio
async def test_session_is_bound_to_a_scan_and_manifest_carries_its_hash_and_label(tmp_path):
    async with Harness(tmp_path, stages=STAGES) as h:
        eng = h.session.engine
        sc = next(x for x in s.registered_scans() if x["label"] == "YugoTest2")
        eng.set_scan(sc)
        await feed_panning(h, 3)
        await h.wait(lambda: len(eng.frames) == 3)
        m = eng._manifest_env()["data"]["input"]
        assert m["sha256"] == sc["sha256"] and m["scan_label"] == "YugoTest2"
        snap = eng.snapshot()["identity"]
        assert snap["scan"]["label"] == "YugoTest2" and snap["session_id"] == "sess1" and snap["run_id"].startswith("sweep_")
        other = next(x for x in s.registered_scans() if x["label"] == "YugoTest1")
        with pytest.raises(Exception):
            eng.set_scan(other)                      # a session never silently switches scan after frames arrived
        await eng.end_sweep()
        packet = C.read_json(h.session.run_dir / "claim_packet.json")["data"]
        assert packet["sweep"]["source_video_sha256"] == sc["sha256"]


def test_bind_endpoint_validates_labels_and_owner(tmp_path):
    async def prep():
        async with Harness(tmp_path, stages=STAGES) as h:
            return h.session
    sess = asyncio.run(prep())
    s.sessions["sess1"] = sess
    try:
        c = client(); c.cookies.set("intake_owner", "owner")
        assert c.post("/api/sessions/sess1/scan", json={"label": "IMG_2281"}, headers=H).status_code == 404
        ok = c.post("/api/sessions/sess1/scan", json={"label": "YugoTest1"}, headers=H)
        assert ok.status_code == 200 and ok.json()["scan"]["label"] == "YugoTest1"
        c2 = client()
        assert c2.post("/api/sessions/sess1/scan", json={"label": "YugoTest1"}, headers=H).status_code == 404
    finally:
        s.sessions.pop("sess1", None)


def test_session_runs_live_under_runs_root_so_frames_and_debugger_can_discover_them(tmp_path, monkeypatch):
    assert s.RUNS_DIR.name == "runs" and s.SAVED_RUNS_DIR == s.RUNS_DIR
    run = tmp_path / "sweep_abc"
    (run / "frames").mkdir(parents=True)
    (run / "frames" / "f00000.jpg").write_bytes(b"x")
    C.write_json(run / "manifest.json", C.envelope("sweep_abc", "frames", {"input": {"file_name": "YugoTest1.MOV"}, "frames": [
        {"frame_id": "f00000", "timestamp_s": 0.0, "image_path": "frames/f00000.jpg", "width_px": 10, "height_px": 10, "quality": {}, "selected": True, "selection_reason": ""}]}))
    assert [r["run_id"] for r in FB.list_saved_runs(tmp_path)] == ["sweep_abc"]        # finished sweeps appear without restarting the server
