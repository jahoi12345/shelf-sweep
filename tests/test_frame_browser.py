"""Frame evidence indexes and file-serving boundaries, without model calls."""
import json
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server


def make_run(root, name="offline_baseline"):
    run = root / name
    (run / "frames").mkdir(parents=True)
    frames = []
    for index, selected in enumerate((True, False, True)):
        fid = f"f{index:05d}"
        (run / "frames" / f"{fid}.jpg").write_bytes(b"\xff\xd8test-image\xff\xd9")
        frames.append({"frame_id": fid, "timestamp_s": index * 0.2, "selected": selected,
                       "image_path": f"frames/{fid}.jpg", "width_px": 1920, "height_px": 1080,
                       "quality": {"reasons": ["blurry"] if index == 1 else []}})
    (run / "manifest.json").write_text(json.dumps({"data": {"input": {"file_name": "sample.MOV"}, "frames": frames}}))
    return run, frames


def test_saved_frames_preserve_selection_and_serve_original_bytes(tmp_path):
    run, _ = make_run(tmp_path)
    make_run(tmp_path, "live")  # Sessions must use their own owner-protected endpoint.
    with patch.object(server, "SAVED_RUNS_DIR", tmp_path):
        client = TestClient(server.app, base_url="http://127.0.0.1:4177")
        assert [r["run_id"] for r in client.get("/api/frame-runs").json()["runs"]] == ["offline_baseline"]
        index = client.get("/api/frame-runs/offline_baseline").json()
        assert (index["frame_count"], index["keyframe_count"]) == (3, 2)
        assert index["frames"][1]["quality"]["reasons"] == ["blurry"]
        assert index["frames"][1]["timestamp_s"] == 0.2
        response = client.get(index["frames"][1]["image_url"])
        assert response.status_code == 200
        assert response.content == (run / "frames/f00001.jpg").read_bytes()


def test_frame_routes_reject_unlisted_files_and_external_symlinks(tmp_path):
    root = tmp_path / "runs"
    run, _ = make_run(root)
    secret = tmp_path / "private.jpg"
    secret.write_bytes(b"private content")
    (run / "frames/f00001.jpg").unlink()
    (run / "frames/f00001.jpg").symlink_to(secret)
    (root / "outside").symlink_to(tmp_path)
    (run / "frames/f00099.jpg").write_bytes(b"unlisted image")
    with patch.object(server, "SAVED_RUNS_DIR", root):
        client = TestClient(server.app, base_url="http://127.0.0.1:4177")
        for path in ("/api/frame-runs/outside", "/api/frame-runs/live",
                     "/api/frame-runs/offline_baseline/frames/f00001.jpg",
                     "/api/frame-runs/offline_baseline/frames/f00099.jpg",
                     "/api/frame-runs/offline_baseline/frames/manifest.json"):
            assert client.get(path).status_code == 404


def test_live_frame_index_requires_session_owner(tmp_path):
    _, frames = make_run(tmp_path)
    engine = SimpleNamespace(frames=frames)
    session = server.SweepSession(session_id="frame-test", owner="correct-owner", run_dir=tmp_path, engine=engine)
    with patch.dict(server.sessions, {"frame-test": session}):
        client = TestClient(server.app, base_url="http://127.0.0.1:4177")
        assert client.get("/api/sessions/frame-test/frame-index").status_code == 404
        client.cookies.set("intake_owner", "wrong-owner")
        assert client.get("/api/sessions/frame-test/frame-index").status_code == 404
        client.cookies.set("intake_owner", "correct-owner")
        response = client.get("/api/sessions/frame-test/frame-index")
        assert response.status_code == 200
        assert response.json()["keyframe_count"] == 2
        assert response.json()["frames"][0]["image_url"] == "/api/sessions/frame-test/frames/f00000.jpg"
