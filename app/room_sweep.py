"""Room sweep at the END of the shelf pass (user request 2026-10-08).

One sweep = the close shelf pass, then a pan of the room for its size. Room prompts must not interrupt the shelf pass (YugoTest14: a
"show_room_corner" notice fired mid-shelf at 47 s), so the room instruction is said once, when the shelf pass is over:

  * the person says they are done (voice end_sweep) or presses End  -> the app says ROOM_MESSAGE and the sweep keeps recording;
    End (a second press, "done" again, or "skip") then ends the sweep;
  * or the shelf pass looks finished: the camera has left the shelf (the room-pan rule of pipeline.room_pan run live on the detection frames:
    >= 3 frames with no book over >= 4 s, after >= MIN_SHELF_S of shelf pass) -> the app says LEFT_MESSAGE; End then ends right away.

Phases: shelf -> room -> (the engine ends). During the shelf phase no room guidance is spoken; during the room phase capture notices
(blur, too far, ...) are muted, because the room pan is far and off the shelf by design. The prompt time is written to the manifest
(``room_sweep``) so pipeline.room_pan tags every non-shelf frame after it as room pan (not counted as books).
Replays and sweeps without the camera end at the first request (nothing to pan). LIBRARY_ROOM_SWEEP_PROMPT=0 restores End = end now.
"""

from __future__ import annotations

import os
from typing import Any

ROOM_MESSAGE = ("Shelf done. Now film the room for its size: tilt down to the line where the floor meets the wall and pan along it, "
                "show each room corner from floor to ceiling, pan along the ceiling line, then the opposite wall. Press End when finished.")
LEFT_MESSAGE = ("If the shelf is finished, film the room for its size: the floor-to-wall line, each corner from floor to ceiling, "
                "the ceiling line, then the opposite wall. Press End when finished.")
MIN_SHELF_S = 45.0          # the live left-the-shelf trigger needs this much shelf pass first (first shelf detection frame to the first off-shelf frame)
FAR_FRACTION = 0.5          # = pipeline.room_pan.FAR_FRACTION: a detection frame whose median spine is < half the running median is not a shelf frame
MIN_OFF_FRAMES = 3          # = pipeline.room_pan.MIN_OFFSHELF_FRAMES
MIN_OFF_SPAN_S = 4.0        # = pipeline.room_pan.MIN_OFFSHELF_SPAN_S


def long_side_px(o: dict[str, Any]) -> float | None:
    """Spine length (the long side of the book polygon, else of the box), as pipeline.room_pan measures it."""
    p = o.get("polygon_xy")
    if p and len(p) == 4:
        return max(((p[i][0] - p[i - 1][0]) ** 2 + (p[i][1] - p[i - 1][1]) ** 2) ** 0.5 for i in range(4))
    b = o.get("bbox_xyxy")
    return max(b[2] - b[0], b[3] - b[1]) if b else None


def enabled_default() -> bool:
    return os.environ.get("LIBRARY_ROOM_SWEEP_PROMPT", "1").lower() not in ("0", "false", "no", "off")


class RoomSweep:
    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = enabled_default() if enabled is None else bool(enabled)
        self.phase = "shelf"
        self.prompt: dict[str, Any] | None = None     # {t_s, frame_id, trigger}
        self._det: dict[str, tuple[float, int, float | None]] = {}   # frame_id -> (t, books, median spine px)
        self._spines: list[float] = []

    # ------------------------------------------------------------------ end requests (button / voice)
    def request_end(self, t: float | None, frame_id: str | None, *, source: str, camera: bool = True, skip: bool = False) -> dict[str, Any]:
        """'room_sweep' = say ``message`` and keep recording; 'end' = end the sweep now."""
        if skip or not self.enabled or not camera or self.phase != "shelf":
            return {"action": "end", "reason": "skip" if skip else ("room phase: second request" if self.phase == "room" else
                                                                     "disabled" if not self.enabled else "no camera" if not camera else self.phase)}
        self._enter(t, frame_id, f"end_request_{source}")
        return {"action": "room_sweep", "message": ROOM_MESSAGE}

    def _enter(self, t: float | None, frame_id: str | None, trigger: str) -> None:
        self.phase = "room"
        self.prompt = {"t_s": None if t is None else round(float(t), 3), "frame_id": frame_id, "trigger": trigger}

    # ------------------------------------------------------------------ live left-the-shelf trigger
    def observe_detection(self, t: float, frame_id: str, books: int, spines_px: list[float]) -> dict[str, Any] | None:
        """One finished detection frame (they finish out of order). Returns the guidance to speak when the camera has left the shelf."""
        if not self.enabled or self.phase != "shelf":
            return None
        spines = sorted(float(s) for s in spines_px if s and s > 0)
        self._det[frame_id] = (float(t), int(books), spines[len(spines) // 2] if spines else None)
        self._spines.extend(spines)
        if len(self._spines) > 4000:
            self._spines = self._spines[-4000:]
        hit = self._left_shelf()
        if hit is None:
            return None
        self._enter(hit[0], hit[1], "left_shelf")
        return {"code": "room_sweep", "message": LEFT_MESSAGE}

    def _left_shelf(self) -> tuple[float, str] | None:
        if not self._spines:
            return None
        run_med = sorted(self._spines)[len(self._spines) // 2]
        first_shelf, off = None, []
        for fid, (t, n, med) in sorted(self._det.items(), key=lambda kv: kv[1][0]):
            shelf = n > 0 and med is not None and med >= FAR_FRACTION * run_med
            if shelf:
                first_shelf = t if first_shelf is None else first_shelf
                off = []
            elif n == 0:
                off.append((t, fid))
            # far frames (books, small spines) neither break nor extend the off-shelf run, as in pipeline.room_pan
        if first_shelf is None or len(off) < MIN_OFF_FRAMES or off[0][0] - first_shelf < MIN_SHELF_S:
            return None
        return off[-1] if off[-1][0] - off[0][0] >= MIN_OFF_SPAN_S else None

    # ------------------------------------------------------------------ record
    @property
    def muted(self) -> bool:
        return self.phase == "room"

    def record(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "phase": self.phase, "prompt": dict(self.prompt) if self.prompt else None,
                "note": "frames after prompt.t_s that are not close shelf frames are room pan (pipeline.room_pan); capture notices muted after it"}
