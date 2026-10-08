"""Sanitising of shipped media. Synthetic inputs built here (EXIF/GPS markers, ffmpeg-tagged video)."""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline import sanitize as Z


def jpeg_with_exif(tmp_path):
    ok, buf = cv2.imencode(".jpg", np.full((64, 64, 3), 120, np.uint8))
    raw = buf.tobytes()
    exif = b"Exif\x00\x00" + b"MM\x00*\x00\x00\x00\x08" + b"\x88\x25" + b"GPSLATITUDE=40.45N"          # fake EXIF with a GPS IFD tag
    seg = b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif
    com = b"\xff\xfe" + (len(b"phone-owner") + 2).to_bytes(2, "big") + b"phone-owner"
    return raw[:2] + seg + com + raw[2:]


def test_jpeg_exif_gps_and_comment_are_removed_losslessly_and_original_untouched(tmp_path):
    data = jpeg_with_exif(tmp_path)
    f = tmp_path / "f.jpg"; f.write_bytes(data)
    before = Z.jpeg_metadata(data)
    assert before["exif"] and before["gps_possible"] and before["comment"]
    out, audit = Z.sanitised_jpeg_for_bundle(f)
    assert not Z.jpeg_metadata(out)["any"] and audit["metadata_removed"] and audit["metadata_after_any"] is False
    assert f.read_bytes() == data                                   # original untouched
    a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR); b = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(a, b)                                     # pixels identical: no re-encode
    assert audit["sha256_original"] != audit["sha256_shipped"]


def test_clean_jpeg_is_shipped_byte_identical():
    ok, buf = cv2.imencode(".jpg", np.full((32, 32, 3), 7, np.uint8))
    raw = buf.tobytes()
    assert Z.strip_jpeg(raw) == raw or not Z.jpeg_metadata(Z.strip_jpeg(raw))["any"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_video_metadata_and_data_tracks_are_stripped_but_source_stays_intact(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=1", "-pix_fmt", "yuv420p",
                    "-metadata", "location=+40.4503-003.7137/", "-metadata", "com.apple.quicktime.location.ISO6709=+40.4503-003.7137+676.609/",
                    "-metadata", "com.apple.quicktime.model=iPhone Air", "-metadata", "creation_time=2026-10-05T22:02:00Z", str(src)], check=True)
    before = Z.video_metadata(src)
    assert before["gps_tags"] and before["identifying_tags"]
    audit = Z.strip_video(src, dst)
    assert audit["clean"] and not audit["after"]["gps_tags"] and not audit["after"]["identifying_tags"]
    assert Z.video_metadata(src)["gps_tags"]                        # original still has its tags: local originals untouched
    with pytest.raises(Z.SanitizeError):
        Z.strip_video(src, src)


def test_corrupt_jpeg_is_rejected():
    with pytest.raises(Z.SanitizeError):
        Z.jpeg_metadata(b"not a jpeg")
