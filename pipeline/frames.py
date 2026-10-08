"""Stage ``frames``: timestamped native-resolution frames + quality/coverage manifest.

    python -m pipeline.frames --video sweeps/IMG_2281.MOV --out runs/offline_baseline

    extract(video_path, run_dir, config) -> manifest_path   (run_dir/manifest.json)

Every sampled frame is saved at the video's native resolution (JPEG, quality configurable) so
every frame_ref opens. ``selected`` marks a documented subset for downstream detection: the
sharpest frame per pan interval (intervals close when the estimated accumulated image shift
reaches ``overlap_step`` of the frame width, so consecutive selected frames overlap). Intervals
whose best frame is still blurry are KEPT (best available) and reported as capture gaps.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline import contracts as C
from pipeline.quality import QualityConfig, add_reasons, frame_quality
from providers.cache import sha256_file

DEFAULTS = {"sample_fps": 5.0, "jpeg_quality": 95, "overlap_step": 0.25, "overlap_lost_step": 0.6}


def probe_video(path: Path) -> dict[str, Any]:
    """Container/stream metadata via ffprobe when available, else OpenCV properties."""
    info: dict[str, Any] = {"ffprobe": None}
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, timeout=60, check=True).stdout
            raw = json.loads(out)
            v = next((s for s in raw.get("streams", []) if s.get("codec_type") == "video"), {})
            a = next((s for s in raw.get("streams", []) if s.get("codec_type") == "audio"), None)
            tags = (raw.get("format") or {}).get("tags") or {}
            info["ffprobe"] = {
                "container": (raw.get("format") or {}).get("format_name"),
                "duration_s": float((raw.get("format") or {}).get("duration") or 0) or None,
                "video_codec": v.get("codec_name"), "width_px": v.get("width"), "height_px": v.get("height"),
                "r_frame_rate": v.get("r_frame_rate"), "avg_frame_rate": v.get("avg_frame_rate"),
                "nb_frames": v.get("nb_frames"), "pix_fmt": v.get("pix_fmt"),
                "rotation_side_data": [sd for sd in v.get("side_data_list", []) if "rotation" in sd] or None,
                "has_audio": a is not None, "audio_codec": a.get("codec_name") if a else None,
                "creation_time": tags.get("creation_time"),
                "device_make": tags.get("com.apple.quicktime.make"), "device_model": tags.get("com.apple.quicktime.model"),
            }
        except (subprocess.SubprocessError, ValueError, OSError):
            info["ffprobe"] = None
    return info


def _aruco_audit(frames_dir: Path, frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Automated check for fiducial markers on the SELECTED frames. Absence != proof of absence."""
    dicts = ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50", "DICT_ARUCO_ORIGINAL",
             "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11"]
    detectors = {}
    for name in dicts:
        if hasattr(cv2.aruco, name):
            detectors[name] = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), cv2.aruco.DetectorParameters())
    hits: list[dict[str, Any]] = []
    checked = 0
    for fr in frames:
        if not fr["selected"]:
            continue
        gray = cv2.imread(str(frames_dir / Path(fr["image_path"]).name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        checked += 1
        for name, det in detectors.items():
            _, ids, _ = det.detectMarkers(gray)
            if ids is not None and len(ids):
                hits.append({"frame_id": fr["frame_id"], "dictionary": name, "ids": [int(i) for i in ids.flatten()]})
    return {"method": "cv2.aruco/AprilTag detection on selected frames", "dictionaries": list(detectors),
            "frames_checked": checked, "detections": hits, "marker_found": bool(hits),
            "caveat": "A detection could be a false positive; no detection does not prove absence "
                      "(small/blurred/oblique markers can be missed)."}


def _contact_sheet(run_dir: Path, frames: list[dict[str, Any]], cols: int = 4, thumb_w: int = 480) -> str:
    sel = [f for f in frames if f["selected"]]
    if not sel:
        return ""
    thumbs = []
    for fr in sel:
        img = cv2.imread(str(run_dir / fr["image_path"]))
        h = round(img.shape[0] * thumb_w / img.shape[1])
        t = cv2.resize(img, (thumb_w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (thumb_w, 22), (0, 0, 0), -1)
        cv2.putText(t, f"{fr['frame_id']} t={fr['timestamp_s']:.2f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
    h = max(t.shape[0] for t in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.zeros((rows * h, cols * thumb_w, 3), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * h:r * h + t.shape[0], c * thumb_w:(c + 1) * thumb_w] = t
    out = run_dir / "audit" / "contact_sheet_selected.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return C.rel_path(out, run_dir)


class PanIntervalSelector:
    """Incremental pan-interval keyframe selection shared by recorded replay and live capture.

    ``feed(frame)`` returns the intervals that CLOSED because of this frame (each with the frame
    chosen as the sharpest in its interval); ``flush()`` closes the open interval. Marks
    ``selected``/``selection_reason``/``interval_index`` on the frame dicts in place.
    """

    def __init__(self, overlap_step: float, overlap_lost_step: float, width_px: int):
        self.overlap_step, self.overlap_lost_step, self.width_px = overlap_step, overlap_lost_step, width_px
        self.intervals: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self._cur: list[dict[str, Any]] = []
        self._path = 0.0
        self._prev: dict[str, Any] | None = None

    def _close(self) -> dict[str, Any] | None:
        cur = self._cur
        if not cur:
            return None
        best = max(cur, key=lambda f: f["quality"]["sharpness"] * (1 - min(f["quality"]["glare_fraction"] * 5, 0.8)))
        idx = len(self.intervals)
        for f in cur:
            f["interval_index"] = idx
            f["selected"] = f is best
            f["selection_reason"] = ("sharpest in pan interval" if f is best else
                                     f"not selected: {best['frame_id']} is the sharpest frame in the same pan interval")
        flags = best["quality"]["reasons"]
        iv = {"interval_index": idx, "start_s": cur[0]["timestamp_s"], "end_s": cur[-1]["timestamp_s"],
              "frame_ids": [f["frame_id"] for f in cur], "selected_frame_id": best["frame_id"],
              "selected_quality_reasons": flags}
        self.intervals.append(iv)
        if "blurry" in flags or "glare" in flags or "fast_motion" in flags:
            best["selection_reason"] += f" (best available; flags: {', '.join(flags)})"
            self.gaps.append({"type": "no_clean_frame", "start_s": cur[0]["timestamp_s"], "end_s": cur[-1]["timestamp_s"],
                              "frame_ids": [f["frame_id"] for f in cur], "flags": flags,
                              "detail": "Best frame in this pan interval is flagged; text/boundaries here may be unreadable."})
        self._cur, self._path = [], 0.0
        return {**iv, "frame": best}

    def feed(self, f: dict[str, Any]) -> list[dict[str, Any]]:
        closed = []
        q = f["quality"]
        if self._prev is not None:
            if q["shift_response"] is not None and "motion_estimate_unreliable" not in q["reasons"]:
                step = math.hypot(q["shift_dx_px"], q["shift_dy_px"]) / self.width_px
            else:
                step = self.overlap_step * 0.5  # unknown motion: be conservative, keep more frames
            if step >= self.overlap_lost_step:
                self.gaps.append({"type": "overlap_lost", "start_s": self._prev["timestamp_s"], "end_s": f["timestamp_s"],
                                  "frame_ids": [self._prev["frame_id"], f["frame_id"]], "flags": ["fast_pan"],
                                  "detail": f"Estimated shift {step:.2f} of frame width between consecutive sampled frames: "
                                            "content between them may not be covered at this sample rate."})
            if self._cur and self._path + step > self.overlap_step:
                c = self._close()
                if c:
                    closed.append(c)
            self._path += step
        self._cur.append(f)
        self._prev = f
        return closed

    def flush(self) -> list[dict[str, Any]]:
        c = self._close()
        return [c] if c else []


def select_frames(frames: list[dict[str, Any]], overlap_step: float, overlap_lost_step: float,
                  width_px: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mark ``selected`` in place; return (intervals, capture_gaps). Batch wrapper over PanIntervalSelector."""
    sel = PanIntervalSelector(overlap_step, overlap_lost_step, width_px)
    for f in frames:
        sel.feed(f)
    sel.flush()
    return sel.intervals, sel.gaps


def extract(video_path: str | Path, run_dir: str | Path, config: dict[str, Any] | None = None) -> Path:
    t_start = time.perf_counter()
    video_path, run_dir = Path(video_path), Path(run_dir)
    config = config or {}
    opts = {**DEFAULTS, **(config.get("frames") or {})}
    qcfg = QualityConfig.from_dict(config.get("quality"))
    run_id = config.get("run_id") or run_dir.name
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise C.ContractError(f"cannot open video {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width_px, height_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    period = 1.0 / float(opts["sample_fps"])
    next_t = 0.0
    frames: list[dict[str, Any]] = []
    prev_small, prev_t = None, None
    native_idx, last_t = 0, -1.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = pos_ms / 1000.0 if pos_ms and pos_ms / 1000.0 > last_t else native_idx / native_fps
        last_t = t
        if t + 1e-6 >= next_t:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            q, small = frame_quality(bgr, gray, qcfg, prev_small, None if prev_t is None else t - prev_t)
            fid = C.frame_id(len(frames))
            path = frames_dir / f"{fid}.jpg"
            cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(opts["jpeg_quality"])])
            frames.append({"frame_id": fid, "timestamp_s": round(t, 3), "native_frame_index": native_idx,
                           "image_path": C.rel_path(path, run_dir), "width_px": bgr.shape[1], "height_px": bgr.shape[0],
                           "quality": q, "selected": False, "selection_reason": ""})
            prev_small, prev_t = small, t
            next_t += period
            while next_t <= t:  # never fall behind after a long gap
                next_t += period
        native_idx += 1
    cap.release()
    if not frames:
        raise C.ContractError("no frames decoded")

    stats = add_reasons(frames, qcfg)
    intervals, gaps = select_frames(frames, float(opts["overlap_step"]), float(opts["overlap_lost_step"]), width_px)
    sheet = _contact_sheet(run_dir, frames)
    audit = {
        "fiducial_markers": _aruco_audit(frames_dir, frames),
        "contact_sheet": sheet,
        "manual_review": None,
        "scope_note": ("Automated checks cannot establish room boundaries or metric references. "
                       "Attach a human/worker visual review with --audit-notes; until then these are UNKNOWN, "
                       "not absent."),
    }
    notes = config.get("audit_notes")
    if notes:
        audit["manual_review"] = C.read_json(notes)

    probe = probe_video(video_path)
    sel_ids = [f["frame_id"] for f in frames if f["selected"]]
    data = {
        "input": {"video_path": str(video_path), "file_name": video_path.name, "size_bytes": video_path.stat().st_size,
                  "sha256": sha256_file(video_path), "opencv": {"fps": native_fps, "frame_count": n_native,
                  "width_px": width_px, "height_px": height_px}, **probe},
        "sampling": {**opts, "timestamp_source": "cv2.CAP_PROP_POS_MSEC (fallback index/fps)",
                     "jpeg_note": "frames are re-encoded JPEG at native resolution (lossy but not downscaled)",
                     "sampled_frames": len(frames), "selected_frames": len(sel_ids)},
        "quality_config": qcfg.to_dict(), "quality_stats": stats,
        "frames": frames, "intervals": intervals, "capture_gaps": gaps, "audit": audit,
    }
    issues = []
    if gaps:
        issues.append({"code": "capture_gaps", "severity": "warning", "detail": f"{len(gaps)} capture gap(s); see data.capture_gaps"})
    env = C.envelope(run_id, "frames", data, status="ok", issues=issues,
                     metrics={"elapsed_s": round(time.perf_counter() - t_start, 3), "frames_written": len(frames)})
    problems = C.validate_manifest(env, run_dir)
    if problems:
        raise C.ContractError("manifest failed validation: " + "; ".join(problems[:5]))
    return C.write_json(run_dir / C.ARTIFACTS["frames"], env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract timestamped native-resolution frames + quality manifest")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--sample-fps", type=float, default=None)
    ap.add_argument("--audit-notes", default=None, help="JSON file with a manual visual review to embed")
    a = ap.parse_args(argv)
    cfg: dict[str, Any] = {"frames": {}}
    if a.sample_fps:
        cfg["frames"]["sample_fps"] = a.sample_fps
    if a.audit_notes:
        cfg["audit_notes"] = a.audit_notes
    path = extract(a.video, a.out, cfg)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
