"""Privacy sanitising for anything shipped in a bundle.

Local originals are NEVER modified: functions read a source and return/write a sanitised COPY.
  * JPEG frames: lossless removal of EXIF/XMP/IPTC/comment segments (no re-encode; pixels unchanged).
  * Video: ffmpeg stream copy with global/stream metadata, chapters and data tracks (Apple location/mebx) dropped.
  * Audit: reports what metadata (GPS/EXIF/make/model/creation time) was found before/after.

    python -m pipeline.sanitize video IN.MOV OUT.mp4      # sanitised copy + audit JSON on stdout
    python -m pipeline.sanitize audit FILE...
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

GPS_KEYS = ("location", "iso6709", "gps")
IDENTIFYING_KEYS = ("make", "model", "software", "creationdate", "creation_time", "serial", "author", "artist", "copyright", "comment", "title")


class SanitizeError(RuntimeError):
    pass


def jpeg_segments(data: bytes) -> list[tuple[int, int, int]]:
    """(marker, start, end) of each segment before the entropy-coded data."""
    if not data.startswith(b"\xff\xd8"):
        raise SanitizeError("not a JPEG")
    out, pos = [], 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            raise SanitizeError("corrupt JPEG segment header")
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker == 0xDA:      # start of scan: the rest is image data
            out.append((marker, pos, len(data)))
            return out
        length = int.from_bytes(data[pos + 2:pos + 4], "big")
        if length < 2:
            raise SanitizeError("corrupt JPEG segment length")
        out.append((marker, pos, pos + 2 + length))
        pos += 2 + length
    return out


def jpeg_metadata(data: bytes) -> dict[str, Any]:
    found: dict[str, Any] = {"exif": False, "xmp": False, "iptc": False, "comment": False, "gps_possible": False}
    for marker, a, b in jpeg_segments(data):
        body = data[a + 4:b]
        if marker == 0xE1 and body.startswith(b"Exif\x00\x00"):
            found["exif"] = True
            found["gps_possible"] = found["gps_possible"] or b"\x25\x88" in body or b"\x88\x25" in body or b"GPS" in body
        elif marker == 0xE1 and (b"xmp" in body[:40].lower() or b"adobe" in body[:40].lower()):
            found["xmp"] = True
            found["gps_possible"] = found["gps_possible"] or b"GPS" in body
        elif marker == 0xED:
            found["iptc"] = True
        elif marker == 0xFE:
            found["comment"] = True
    found["any"] = any(found[k] for k in ("exif", "xmp", "iptc", "comment"))
    return found


def strip_jpeg(data: bytes) -> bytes:
    """Remove every APPn (except JFIF APP0) and COM segment without touching the compressed image data."""
    segs = jpeg_segments(data)
    out = bytearray(b"\xff\xd8")
    for marker, a, b in segs:
        if marker == 0xDA:
            out += data[a:]
            break
        keep = marker == 0xE0 or not (0xE1 <= marker <= 0xEF or marker == 0xFE)
        if keep:
            out += data[a:b]
    return bytes(out)


def _probe(path: Path) -> dict[str, Any]:
    if not shutil.which("ffprobe"):
        return {"error": "ffprobe not installed"}
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return {"error": r.stderr.strip()[:200]}
    return json.loads(r.stdout)


def video_metadata(path: Path) -> dict[str, Any]:
    p = _probe(path)
    if "error" in p:
        return {"error": p["error"]}
    tags: dict[str, str] = {k.lower(): str(v) for k, v in ((p.get("format") or {}).get("tags") or {}).items()}
    for st in p.get("streams", []):
        tags.update({f"stream{st.get('index')}.{k.lower()}": str(v) for k, v in (st.get("tags") or {}).items() if k.lower() not in ("handler_name", "vendor_id", "language", "encoder")})
    data_streams = [st for st in p.get("streams", []) if st.get("codec_type") == "data"]
    gps = [k for k in tags if any(g in k for g in GPS_KEYS)]
    ident = [k for k in tags if any(i in k for i in IDENTIFYING_KEYS)]
    return {"gps_tags": gps, "identifying_tags": ident, "data_streams": len(data_streams), "tag_count": len(tags),
            "any": bool(gps or ident or data_streams)}


def strip_video(src: Path, dst: Path) -> dict[str, Any]:
    """Write a sanitised stream-copy of ``src`` to ``dst`` (src untouched); return the before/after audit."""
    src, dst = Path(src), Path(dst)
    if not shutil.which("ffmpeg"):
        raise SanitizeError("ffmpeg is required to sanitise video")
    if src.resolve() == dst.resolve():
        raise SanitizeError("refusing to overwrite the original; write to a different path")
    before = video_metadata(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", "-dn", "-sn", "-map_metadata", "-1",
           "-map_chapters", "-1", "-c", "copy", "-movflags", "+faststart", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise SanitizeError(f"ffmpeg failed: {r.stderr.strip()[:300]}")
    after = video_metadata(dst)
    return {"source": str(src), "output": str(dst), "before": before, "after": after, "clean": not after.get("any") and "error" not in after,
            "note": "stream copy: pixels/audio unchanged; original file not modified"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sanitised_jpeg_for_bundle(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = Path(path).read_bytes()
    before = jpeg_metadata(raw)
    out = strip_jpeg(raw) if before["any"] else raw
    after = jpeg_metadata(out)
    return out, {"file": path.name, "metadata_before": {k: v for k, v in before.items() if k != "any"}, "metadata_removed": before["any"],
                 "metadata_after_any": after["any"], "sha256_original": sha256_bytes(raw), "sha256_shipped": sha256_bytes(out)}


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) >= 3 and a[0] == "video":
        print(json.dumps(strip_video(Path(a[1]), Path(a[2])), indent=2))
        return 0
    if a and a[0] == "audit":
        for f in a[1:]:
            p = Path(f)
            print(json.dumps({"file": f, **(jpeg_metadata(p.read_bytes()) if p.suffix.lower() in (".jpg", ".jpeg") else video_metadata(p))}, indent=2))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
