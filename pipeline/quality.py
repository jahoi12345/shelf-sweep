"""Per-frame capture-quality signals.

Heuristic, unvalidated signals computed on the NATIVE-resolution frame (motion on a downscaled
copy). They guide frame selection and capture-gap reporting; they are not accuracy claims.
Thresholds are configuration, not ground truth.
"""

from __future__ import annotations

import collections
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


@dataclass
class QualityConfig:
    blur_floor_ratio: float = 0.5      # frame is "blurry" if sharpness < ratio * run median
    blur_floor_abs: float = 30.0       # ...or below this absolute Laplacian variance
    glare_fraction: float = 0.04       # fraction of near-white, low-saturation pixels flagged as glare
    dark_mean_luma: float = 50.0       # mean luma below this -> "dark"
    fast_motion_px_s: float = 1200.0   # estimated image shift speed above this -> "fast_motion"
    motion_response_min: float = 0.05  # phase-correlation peak below this -> motion estimate unreliable
    motion_scale_width: int = 960      # motion is estimated on a copy resized to this width

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "QualityConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian at native resolution (higher = sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tile_sharpness(gray: np.ndarray, rows: int = 3, cols: int = 4) -> list[list[float]]:
    """Laplacian variance per tile; lets the manifest show which regions are readable."""
    h, w = gray.shape
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            tile = gray[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols]
            row.append(round(sharpness(tile), 1))
        out.append(row)
    return out


def glare_fraction(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 2] >= 250) & (hsv[..., 1] <= 40)
    return float(mask.mean())


def small_gray(gray: np.ndarray, width: int) -> np.ndarray:
    h, w = gray.shape
    if w <= width:
        return gray
    return cv2.resize(gray, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def estimate_shift(prev_small: np.ndarray, cur_small: np.ndarray, full_width: int) -> tuple[float, float, float]:
    """Image shift (dx, dy) in NATIVE pixels between two frames and the correlation response.

    Phase correlation assumes a mostly global translation; with parallax and repetitive shelves
    it is approximate, so the response is reported and low responses are treated as unreliable.
    """
    a = np.float32(prev_small)
    b = np.float32(cur_small)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, win)
    scale = full_width / a.shape[1]
    return float(dx * scale), float(dy * scale), float(response)


# ORB features of a frame are computed once and reused when that frame is the "previous" one of the next call (each frame was detected twice before);
# keyed by array identity (the entry keeps the array alive, so an id is never reused while cached). One ORB object per thread: a detector instance is not safe to share.
_TLS = threading.local()
_FEATS: "collections.OrderedDict[int, tuple[np.ndarray, Any, Any]]" = collections.OrderedDict()
_FEATS_LOCK = threading.Lock()
_FEATS_KEEP = 8


def _orb_features(small: np.ndarray) -> tuple[Any, Any]:
    with _FEATS_LOCK:
        hit = _FEATS.get(id(small))
        if hit is not None and hit[0] is small:
            return hit[1], hit[2]
    orb = getattr(_TLS, "orb", None)
    if orb is None:
        orb = _TLS.orb = cv2.ORB_create(1500)
    kps, desc = orb.detectAndCompute(small, None)
    with _FEATS_LOCK:
        _FEATS[id(small)] = (small, kps, desc)
        while len(_FEATS) > _FEATS_KEEP:
            _FEATS.popitem(last=False)
    return kps, desc


def estimate_zoom(prev_small: np.ndarray, cur_small: np.ndarray, min_inliers: int = 12) -> tuple[float, int]:
    """Frame-to-frame magnification, defined as the tracker's zoom: sqrt|det| of the 2x2 part of the homography (ORB + RANSAC) between consecutive
    frames. >1 means the later frame is magnified. Returns (scale, inlier_count); inlier_count < min_inliers means unreliable. The determinant
    includes perspective/anisotropic change, so values run higher than a pure similarity scale (YT1 p90 0.22/s vs 0.14/s similarity)."""
    k1, d1 = _orb_features(prev_small)
    k2, d2 = _orb_features(cur_small)
    if d1 is None or d2 is None:
        return 1.0, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(m) < min_inliers:
        return 1.0, 0
    p1 = np.float32([k1[x.queryIdx].pt for x in m])
    p2 = np.float32([k2[x.trainIdx].pt for x in m])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if H is None or inl is None or int(inl.sum()) < min_inliers or abs(H[2, 2]) < 1e-9:
        return 1.0, 0
    H = H / H[2, 2]
    return float(np.sqrt(abs(np.linalg.det(H[:2, :2])))), int(inl.sum())


# The zoom estimate (ORB + RANSAC) and everything else in frame_quality are independent given the two small frames: run the zoom on a second thread (cv2 releases
# the GIL). Same numbers as before; on a 1080x1920 frame the live ingest was ~120 ms per frame under load, of which ~85 ms were these two motion estimates.
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frame-quality")


def _zoom_or_none(prev_small: np.ndarray, small: np.ndarray) -> tuple[float, int] | None:
    try:
        return estimate_zoom(prev_small, small)
    except cv2.error:
        return None


def frame_quality(bgr: np.ndarray, gray: np.ndarray, cfg: QualityConfig,
                  prev_small: np.ndarray | None, dt_s: float | None) -> tuple[dict[str, Any], np.ndarray]:
    """Return (quality dict without final reasons, small gray for the next frame's motion estimate)."""
    h, w = gray.shape
    small = small_gray(gray, cfg.motion_scale_width)
    if prev_small is not None and prev_small.shape != small.shape:
        prev_small = None   # the picture size changed mid-sweep (phone rotated, camera switched): no motion estimate for this frame
    zoom_future = _POOL.submit(_zoom_or_none, prev_small, small) if prev_small is not None and dt_s else None
    q: dict[str, Any] = {
        "sharpness": round(sharpness(gray), 2),
        "tile_sharpness_3x4": tile_sharpness(gray),
        "glare_fraction": round(glare_fraction(bgr), 4),
        "mean_luma": round(float(gray.mean()), 1),
        "shift_dx_px": None, "shift_dy_px": None, "shift_response": None, "motion_px_per_s": None, "zoom_rate_per_s": None,
    }
    if prev_small is not None and dt_s:
        dx, dy, resp = estimate_shift(prev_small, small, w)
        q.update(shift_dx_px=round(dx, 1), shift_dy_px=round(dy, 1), shift_response=round(resp, 3),
                 motion_px_per_s=round(float(np.hypot(dx, dy)) / dt_s, 1))
        zoom = zoom_future.result() if zoom_future is not None else None
        if zoom is not None and zoom[1] >= 12 and zoom[0] > 0:
            q["zoom_rate_per_s"] = round(abs(float(np.log(zoom[0]))) / dt_s, 3)
    return q, small


def reasons_for(q: dict[str, Any], floor: float, cfg: QualityConfig) -> list[str]:
    """Quality flags for one frame given a sharpness floor (run-relative or running)."""
    reasons = []
    if q["sharpness"] < floor:
        reasons.append("blurry")
    if q["glare_fraction"] >= cfg.glare_fraction:
        reasons.append("glare")
    if q["mean_luma"] < cfg.dark_mean_luma:
        reasons.append("dark")
    if q["motion_px_per_s"] is not None and q["motion_px_per_s"] > cfg.fast_motion_px_s:
        reasons.append("fast_motion")
    if q["shift_response"] is not None and q["shift_response"] < cfg.motion_response_min:
        reasons.append("motion_estimate_unreliable")
    return reasons


def sharpness_floor(sharpness_values: list[float], cfg: QualityConfig) -> tuple[float, float]:
    median = float(np.median(sharpness_values)) if len(sharpness_values) else 0.0
    return median, max(cfg.blur_floor_abs, cfg.blur_floor_ratio * median)


def add_reasons(frames: list[dict[str, Any]], cfg: QualityConfig) -> dict[str, Any]:
    """Fill ``quality.reasons`` for every frame using run-relative sharpness. Returns run stats."""
    median, floor = sharpness_floor([f["quality"]["sharpness"] for f in frames], cfg)
    for f in frames:
        f["quality"]["reasons"] = reasons_for(f["quality"], floor, cfg)
    return {"sharpness_median": round(median, 2), "blur_floor": round(floor, 2)}
