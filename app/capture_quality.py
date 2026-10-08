"""Live capture-quality rules. Thresholds come from a YT2 analysis (correlational, one video) and are env-tunable.

Per second of capture we take the median motion (native px/s), sharpness and zoom rate, and place the second in a band:
  strict  motion <= 400, sharpness >= 90, zoom <= 0.3/s
  medium  within the slow-down limits (motion <= 450, sharpness >= 60, zoom <= 0.5/s)
  poor    beyond a slow-down limit;  bad = poor with motion > 700 or sharpness < 40
Nudges: SLOW-DOWN when the last second's median breaks a slow-down limit; RE-CAPTURE when EVERY frame over > 1 s has motion > 700 or sharpness < 40.
Whether a nudge is SPOKEN is decided by app.nudge_policy (once per bad stretch, re-arm after good capture, global gap), shared with the engine's monitor.
count reliability: high if >= 70% of seconds are medium-or-better, medium if >= 40%, else low ("unknown" under 5 seconds). The 40% cut is ours.
DISTANCE (user decisions 2026-10-08): the target spine width is TARGET_SPINE_PX = 60 px (the close passes of YugoTest12/13; env LIBRARY_CQ_MIN_SPINE_PX).
  - while the median spine width over the last 2 s is under it, `distance_now` = "too_far": the on-screen indicator turns red ("Too far - move closer"),
    whatever the sharpness band says, and the voice repeats the move-closer command every global gap (~15 s) until it is fixed;
  - a sweep whose median spine width is under it is rated at most medium. Frames are still recorded and counted: it only warns.
Zoom rate = |d ln scale| / dt with scale = sqrt|det| of the 2x2 part of the frame-to-frame homography (the YT2 analysis' definition), estimated
here from ORB+RANSAC between consecutive frames (pipeline.quality.estimate_zoom). That analysis chains Lucas-Kanade instead; on YT1 the p90 is ~0.22/s here
vs their 0.29, so our zoom values run ~25% lower and the 0.3/0.5 limits are slightly more lenient than in their analysis.
"""

from __future__ import annotations

import collections
import os
import statistics
from typing import Any

from app.nudge_policy import NudgePolicy

TARGET_SPINE_PX = 60.0   # user decision 2026-10-08: spines about as wide as in the YugoTest12/13 close passes (was 120 before, 40 earlier)

MESSAGES = {   # short and direct (user decision 2026-10-07); spoken once per bad stretch via app.nudge_policy
    "slow_down": "Slow down and hold the phone steady.",
    "too_far": "Too far. Step closer: about one arm's length, two or three shelves on screen.",
    "recapture": "That stretch was too blurry. Go back to where it started and film it again, slowly.",
    "portrait": "Turn the phone sideways (landscape).",
}


def _opt(c: dict[str, Any], key: str, env: str) -> float | None:
    v = c.get(key, os.environ.get(env))
    try:
        return None if v in (None, "") else float(v)
    except ValueError:
        return None


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class CaptureQuality:
    def __init__(self, config: dict[str, Any] | None = None, policy: NudgePolicy | None = None) -> None:
        c = config or {}
        self.policy = policy or NudgePolicy(c.get("nudge_rearm_s"), c.get("nudge_gap_s"))   # the engine passes its one shared policy
        def g(key, env, d, alias=None):          # alias = the canonical env name (takes precedence when set)
            v = c.get(key)
            if v is None and alias and os.environ.get(alias) not in (None, ""):
                v = os.environ[alias]
            return float(v) if v is not None else _f(env, d)
        self.t = {"strict_motion": g("cq_strict_motion", "LIBRARY_CQ_STRICT_MOTION", 400), "strict_sharp": g("cq_strict_sharp", "LIBRARY_CQ_STRICT_SHARP", 90),
                  "strict_zoom": g("cq_strict_zoom", "LIBRARY_CQ_STRICT_ZOOM", 0.3), "slow_motion": g("cq_slow_motion", "LIBRARY_CQ_SLOW_MOTION", 450, "LIBRARY_CQ_MAX_MOTION_PX_S"),
                  "slow_sharp": g("cq_slow_sharp", "LIBRARY_CQ_SLOW_SHARP", 60, "LIBRARY_CQ_MIN_SHARPNESS"), "slow_zoom": g("cq_slow_zoom", "LIBRARY_CQ_SLOW_ZOOM", 0.3, "LIBRARY_CQ_MAX_ZOOM_RATE"),
                  "recap_motion": g("cq_recap_motion", "LIBRARY_CQ_RECAP_MOTION", 700), "recap_sharp": g("cq_recap_sharp", "LIBRARY_CQ_RECAP_SHARP", 40, "LIBRARY_CQ_RECAPTURE_SHARPNESS"),
                  "recap_seconds": g("cq_recap_seconds", "LIBRARY_CQ_RECAP_SECONDS", 1.0), "high_share": g("cq_high_share", "LIBRARY_CQ_HIGH_SHARE", 0.70),
                  "medium_share": g("cq_medium_share", "LIBRARY_CQ_MEDIUM_SHARE", 0.40),
                  "min_spine_px": g("cq_min_spine_px", "LIBRARY_CQ_MIN_SPINE_PX", TARGET_SPINE_PX), "width_window_s": g("cq_width_window_s", "LIBRARY_CQ_WIDTH_WINDOW_S", 2.0),
                  "min_width_dets": g("cq_min_width_dets", "LIBRARY_CQ_MIN_WIDTH_DETS", 10),
                  # SCENE speed (spine-widths/s = px/s / median detected spine width in the same 1-s window) is INFORMATIONAL ONLY: the YT2 analysis measured
                  # that it does not predict link failure (pooled AUC 0.52; pixel motion 0.61 and sharpness do). No nudge, no band; reported with a reference.
                  "scene_speed_ref_sw_s": g("cq_max_scene_speed_sw_s", "LIBRARY_CQ_MAX_SCENE_SPEED_SW_S", 15)}
        self.sec_widths: dict[int, list[float]] = {}
        self.widths: collections.deque[tuple[float, float]] = collections.deque()
        self.all_widths: list[float] = []
        self.recent_widths: collections.deque[tuple[float, float]] = collections.deque(maxlen=max(1, int(self.t["min_width_dets"])))
        self.samples: collections.deque[tuple[float, float | None, float, float | None]] = collections.deque()
        self.seconds: dict[int, list[tuple[float | None, float, float | None]]] = {}
        self.now = "unknown"
        self.distance_now = "unknown"          # "too_far" / "ok" from the last width window with enough detections
        self.indicator_frames: collections.Counter[str] = collections.Counter()   # indicator state per captured frame (for the summary)
        self.now_scene_sw: float | None = None

    # ---------------------------------------------------------------- classification
    def _med(self, vals: list[float | None]) -> float | None:
        v = [x for x in vals if x is not None]
        return statistics.median(v) if v else None

    def band(self, motion: float | None, sharp: float, zoom: float | None) -> str:
        """Pixel motion (px/s), sharpness and zoom rate are the rule (the two scans agree in pixels); scene speed never enters."""
        t = self.t
        slow = (motion is not None and motion > t["slow_motion"]) or sharp < t["slow_sharp"] or (zoom is not None and zoom > t["slow_zoom"])
        if (motion is not None and motion > t["recap_motion"]) or sharp < t["recap_sharp"]:
            return "bad"
        if slow:
            return "poor"
        strict = (motion is None or motion <= t["strict_motion"]) and sharp >= t["strict_sharp"] and (zoom is None or zoom <= t["strict_zoom"])
        return "strict" if strict else "medium"

    def _sw(self, sec: int, motion: float | None) -> float | None:
        """px/s -> spine-widths/s using the median detected spine width of that 1-s window (None without detections)."""
        w = self.sec_widths.get(sec)
        return None if motion is None or not w else motion / max(1.0, statistics.median(w))

    # ---------------------------------------------------------------- feed
    def add(self, t_s: float, motion: float | None, sharp: float, zoom: float | None) -> list[dict[str, str]]:
        self.samples.append((t_s, motion, sharp, zoom))
        while self.samples and t_s - self.samples[0][0] > max(1.0, self.t["recap_seconds"]) + 0.05:
            self.samples.popleft()
        self.seconds.setdefault(int(t_s), []).append((motion, sharp, zoom))
        recent = list(self.samples)
        out: list[str] = []
        if len(recent) >= 3:
            m, s, z = (self._med([r[i] for r in recent]) for i in (1, 2, 3))
            self.now = self.band(m, s if s is not None else 0.0, z)
            self.now_scene_sw = self._sw(int(t_s), m)
            span = recent[-1][0] - recent[0][0]
            bad_all = all((r[1] is not None and r[1] > self.t["recap_motion"]) or r[2] < self.t["recap_sharp"] for r in recent)
            if span >= self.t["recap_seconds"] and bad_all and self.policy.offer("recapture", t_s):
                out.append("recapture")
            elif self.now in ("poor", "bad") and self.policy.offer("slow_down", t_s):   # offered every bad frame: keeps the stretch open
                out.append("slow_down")
        self.indicator_frames[self.indicator()] += 1
        return [{"code": c, "message": MESSAGES[c]} for c in out]

    def add_widths(self, t_s: float, widths: list[float]) -> list[dict[str, str]]:
        """Detected book spine widths (px, native) for a frame. When the median width over the last window is below the target (TARGET_SPINE_PX,
        LIBRARY_CQ_MIN_SPINE_PX) the distance state is "too_far" whatever the sharpness band (distance overrides sharpness, user decision 2026-10-08)
        and the `too_far` command is offered; app.nudge_policy repeats it every global gap while it holds. Row changes do not matter here."""
        for w in widths:
            self.widths.append((t_s, float(w)))
            self.recent_widths.append((t_s, float(w)))
            self.all_widths.append(float(w))
            self.sec_widths.setdefault(int(t_s), []).append(float(w))
        while self.widths and t_s - self.widths[0][0] > self.t["width_window_s"]:
            self.widths.popleft()
        recent = [w for _, w in self.widths]
        # few spines in the 2-s window (sparse detection frames): fall back to the last min_width_dets widths when they are all from the last
        # NudgePolicy.rearm_s (6 s); none that recent (a room pan, a gap, the start): no distance reading, the sharpness band shows
        n = int(self.t["min_width_dets"])
        if len(recent) < n:
            tail = list(self.recent_widths)[-n:]
            recent = [w for _, w in tail] if len(tail) == n and t_s - tail[0][0] <= self.policy.rearm_s else []
        self.distance_now = ("unknown" if len(recent) < n else "too_far" if statistics.median(recent) < self.t["min_spine_px"] else "ok")
        if self.distance_now == "too_far" and self.policy.offer("too_far", t_s):
            return [{"code": "too_far", "message": MESSAGES["too_far"]}]
        return []

    def indicator(self) -> str:
        """The on-screen state: "too_far" while the spines are under the target (never good/ok then), else the sharpness band."""
        return "too_far" if self.distance_now == "too_far" else self.now

    def spine_too_small(self) -> bool:
        """The sweep's spines are too small in frame: the median detected spine width over the whole sweep is under the `too_far` target
        (LIBRARY_CQ_MIN_SPINE_PX), with at least LIBRARY_CQ_MIN_WIDTH_DETS widths seen."""
        return len(self.all_widths) >= self.t["min_width_dets"] and statistics.median(self.all_widths) < self.t["min_spine_px"]

    def median_spine_px(self) -> float | None:
        recent = [w for _, w in self.widths]
        return round(statistics.median(recent), 1) if recent else None

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        bands = collections.Counter()
        sw_rates = []
        for sec, v in sorted(self.seconds.items()):
            m = self._med([x[0] for x in v])
            sw = self._sw(sec, m)
            if sw is not None:
                sw_rates.append(sw)
            bands[self.band(m, self._med([x[1] for x in v]) or 0.0, self._med([x[2] for x in v]))] += 1
        scene = None
        if sw_rates:
            q = sorted(sw_rates)
            ref = self.t["scene_speed_ref_sw_s"]
            scene = {"unit": "spine_widths_per_s", "informational_only": True, "reference_sw_s": ref, "seconds_with_detections": len(q),
                     "p50": round(q[len(q) // 2], 2), "p90": round(q[min(len(q) - 1, int(0.9 * len(q)))], 2), "max": round(q[-1], 2),
                     "share_above_reference": round(sum(1 for x in q if x > ref) / len(q), 3), "now": None if self.now_scene_sw is None else round(self.now_scene_sw, 2),
                     "note": "scene speed does not predict link failure (pooled AUC 0.52); it drives no nudge or band"}
        n = sum(bands.values())
        ok = (bands["strict"] + bands["medium"]) / n if n else None
        reliability = "unknown" if n < 5 else "high" if ok >= self.t["high_share"] else "medium" if ok >= self.t["medium_share"] else "low"
        by_bands, capped = reliability, None
        if reliability == "high" and self.spine_too_small():      # sharp, slow seconds of spines too small in frame do not make a reliable count
            reliability = "medium"
            capped = (f"capped at medium: median detected spine width over the sweep {statistics.median(self.all_widths):.1f} px is under the "
                      f"target {self.t['min_spine_px']:.0f} px (LIBRARY_CQ_MIN_SPINE_PX)")
        return {"seconds": n, "count_reliability_bands": by_bands, "count_reliability_capped": capped, "bands": {k: bands[k] for k in ("strict", "medium", "poor", "bad")},
                "share_strict": None if not n else round(bands["strict"] / n, 3), "share_medium_or_better": None if ok is None else round(ok, 3),
                "now": self.now, "distance_now": self.distance_now, "indicator": self.indicator(), "target_spine_px": self.t["min_spine_px"],
                "share_frames_too_far": round(self.indicator_frames["too_far"] / sum(self.indicator_frames.values()), 3) if self.indicator_frames else None,
                "median_spine_px": self.median_spine_px(), "motion_scene_units": scene,
                "median_spine_px_overall": round(statistics.median(self.all_widths), 1) if self.all_widths else None, "count_reliability": reliability, "nudges_issued": dict(self.policy.issued), "nudge_policy": self.policy.summary(), "thresholds": self.t,
                "note": "correlational thresholds from one analysed video (YT2); explains count reliability, does not correct the count"}


# ---------------------------------------------------------------- per-sweep capture stats
def _pctl(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def _gap_stats(times: list[float]) -> dict[str, float | None]:
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    r = lambda v: None if v is None else round(v, 3)  # noqa: E731
    return {"p50": r(_pctl(gaps, 0.5)), "p90": r(_pctl(gaps, 0.9)), "max": r(gaps[-1] if gaps else None)}


def _fps_series(times: list[float], bucket_s: float = 15.0, limit: int = 40) -> list[float]:
    """Frames per second in consecutive ``bucket_s`` slices of the sweep (the last partial slice is scaled to its length and dropped when under 3 s).
    A live sweep whose rate sinks over time (page main thread saturated by the growing state) shows here as a falling series."""
    if len(times) < 2:
        return []
    t0, span = times[0], times[-1] - times[0]
    counts: dict[int, int] = collections.Counter(int((t - t0) // bucket_s) for t in times)
    out = []
    for b in range(int(span // bucket_s) + 1):
        length = min(bucket_s, span - b * bucket_s)
        if length < 3.0 and b > 0:
            continue
        out.append(round(counts.get(b, 0) / max(length, 1e-6), 2))
    return out[:limit]


def _orientation(sizes: collections.Counter) -> dict[str, Any]:
    """Portrait/landscape of the saved frames (from the pixels the pipeline actually got, not from what the phone says): the pipeline was tuned on 1920x1080
    landscape; a phone held upright delivers 1080x1920 and the page cannot force landscape (the camera follows the device orientation)."""
    total = sum(sizes.values())
    kinds: collections.Counter = collections.Counter()
    for key, n in sizes.items():
        try:
            w, h = (int(x) for x in str(key).split("x"))
        except ValueError:
            continue
        kinds["portrait" if h > w else "landscape" if w > h else "square"] += n
    if not total or not kinds:
        return {"orientation": None, "orientation_share": None, "frame_aspect": None}
    top = kinds.most_common(1)[0][0]
    w, h = (int(x) for x in str(sizes.most_common(1)[0][0]).split("x"))
    return {"orientation": top, "orientation_share": {k: round(v / total, 3) for k, v in kinds.items()}, "frame_aspect": round(w / h, 3)}


def capture_stats(frames: list[dict[str, Any]], info: dict[str, Any] | None, gate: dict[str, Any] | None, sources: set[str] | list[str]) -> dict[str, Any]:
    """Comparable per-sweep capture numbers: native-resolution Laplacian sharpness, frame gaps on BOTH clocks (server arrival and the phone's capture clock when the
    client sent it), fps over time, device, orientation and the blur gate's effect. Descriptive only (no accuracy claim); IESTest4 is the reference live run.
    Key names: ``frame_gap_p50_s / _p90_s / _max_s`` (arrival), ``capture_clock_gap_p50_s / _p90_s / _max_s`` (phone clock), the same two under ``frame_timing``."""
    info = info or {}
    n = len(frames)
    sharp = sorted(float(f["quality"]["sharpness"]) for f in frames)
    ts = [float(f["timestamp_s"]) for f in frames]
    arr = _gap_stats(ts)
    cap_rows = [(float(f["client"]["t_s"]), float(f["timestamp_s"])) for f in frames if isinstance(f.get("client"), dict) and isinstance(f["client"].get("t_s"), (int, float))]
    cap_t = [c for c, _ in cap_rows]
    cap = _gap_stats(cap_t) if len(cap_t) >= 2 else None
    span = (ts[-1] - ts[0]) if n >= 2 else 0.0
    cap_span = (cap_t[-1] - cap_t[0]) if len(cap_t) >= 2 else 0.0
    sizes = collections.Counter(f"{f.get('width_px')}x{f.get('height_px')}" for f in frames)
    cands = [f["client"]["cand"] for f in frames if isinstance(f.get("client"), dict) and isinstance(f["client"].get("cand"), int)]
    series = _fps_series(ts)
    third = max(1, len(ts) // 3)
    lag = [a - c for c, a in cap_rows]
    k = max(5, len(lag) // 10)
    drift = round(statistics.median(lag[-k:]) - statistics.median(lag[:k]), 2) if len(lag) >= 20 else None   # >0: the server is falling behind the phone
    ingest = sorted(float(f["ingest_ms"]) for f in frames if isinstance(f.get("ingest_ms"), (int, float)))
    ing = None if not ingest else {"p50": round(_pctl(ingest, 0.5), 1), "p90": round(_pctl(ingest, 0.9), 1), "max": round(ingest[-1], 1)}
    timing = {"arrival_clock": arr, "capture_clock": cap, "fps_series_15s": series,
              "fps_first_third": round((third - 1) / (ts[third - 1] - ts[0]), 2) if n >= 6 and third > 1 and ts[third - 1] > ts[0] else None,
              "fps_last_third": round((third - 1) / (ts[-1] - ts[-third]), 2) if n >= 6 and third > 1 and ts[-1] > ts[-third] else None,
              "arrival_minus_capture_drift_s": drift, "server_ingest_ms": ing}
    return {
        "frames": n, "span_s": round(span, 2), "fps_effective": round((n - 1) / span, 2) if span > 0 else None,
        "fps_capture_clock": round((len(cap_t) - 1) / cap_span, 2) if cap_span > 0 else None,
        "capture_sources": sorted(sources), "timestamp_basis": "video clock (replay)" if set(sources) == {"replay"} else "server arrival clock",
        "median_sharpness": round(statistics.median(sharp), 1) if sharp else None,
        "sharpness_p10": round(_pctl(sharp, 0.1), 1) if sharp else None,
        "frame_gap_p50_s": arr["p50"], "frame_gap_p90_s": arr["p90"], "frame_gap_max_s": arr["max"],
        "capture_clock_gap_p50_s": None if cap is None else cap["p50"],
        "capture_clock_gap_p90_s": None if cap is None else cap["p90"], "capture_clock_gap_max_s": None if cap is None else cap["max"],
        "frame_timing": timing,
        "frame_size": sizes.most_common(1)[0][0] if sizes else None, **_orientation(sizes),
        "device_label": info.get("label"),
        "device": {k: info.get(k) for k in ("width", "height", "frame_rate", "facing_mode", "focus_mode", "orientation", "secure_context", "capture_mode",
                                              "jpeg_quality", "user_agent", "constraint_set", "requested_resolution", "zoom", "zoom_requested", "zoom_max") if info.get(k) is not None},
        "client_window_candidates_median": statistics.median(cands) if cands else None,
        "client_counters": info.get("client_stats") or None,
        "blur_gate": gate,
    }
