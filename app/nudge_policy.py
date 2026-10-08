"""One spoken-notice policy for every capture nudge (user decision 2026-10-07: "say it once per bad stretch, then stay silent").

Both nudge sources (app.capture_quality.CaptureQuality and app.engine.CaptureMonitor) detect conditions; they no longer decide whether to speak.
Every frame on which a condition holds they call ``offer(code, t)``; this class answers whether the notice is spoken now.

  family    codes with the same meaning share one stretch: speed = slow_down, blur, fast_pan (+ recapture in its own slot);
            distance = too_far, closer; revisit; overlap; glare; dark; portrait.
  repeat    REPEATING codes (too_far, user decision 2026-10-08) are the exception to "once": while the condition holds they are spoken
            again whenever the global gap allows (about every GAP_S seconds) until it is fixed.
  stretch   starts at the first offer of a family; it continues while offers keep arriving, and ends (re-arms) once no offer of that family
            has arrived for REARM_S seconds of capture (default 6 s, LIBRARY_NUDGE_REARM_S): the condition was gone for that long.
  once      each slot (main; recapture) is spoken at most once per stretch. A spoken recapture ("film it again, slowly") also covers the
            main speed slot, and opens a silent revisit stretch (the person was told to go back, so the going back is not a problem).
  gap       at least GAP_S seconds (default 15 s, LIBRARY_NUDGE_GAP_S) between ANY two spoken notices; the first notice is never held.
            A stretch whose notice is held by the gap is spoken on its first offer after the gap clears, if it is still going on.

Time is the capture clock (frame timestamps): late detection/reading evidence carries its frame's time, and the gap uses the newest time seen.
The on-screen quality indicator is separate (CaptureQuality.now) and keeps showing the state while the voice stays silent.
``issued`` counts spoken notices only; ``stretches`` counts bad stretches per family (spoken or not).
"""

from __future__ import annotations

import collections
import os
from typing import Any

FAMILY = {"slow_down": "speed", "blur": "speed", "fast_pan": "speed", "recapture": "speed",
          "too_far": "distance", "closer": "distance"}           # every other code is its own family
SLOT = {"recapture": "recapture"}                              # every other code speaks in the family's "main" slot
REPEATING = {"too_far"}                                        # spoken again every gap while the condition holds (user decision 2026-10-08)
REARM_S = 6.0
GAP_S = 15.0


def _env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class NudgePolicy:
    def __init__(self, rearm_s: float | None = None, gap_s: float | None = None) -> None:
        self.rearm_s = float(rearm_s) if rearm_s is not None else _env("LIBRARY_NUDGE_REARM_S", REARM_S)
        self.gap_s = float(gap_s) if gap_s is not None else _env("LIBRARY_NUDGE_GAP_S", GAP_S)
        self.clock = float("-inf")
        self.last_spoken: float | None = None
        self._st: dict[str, dict[str, Any]] = {}
        self.issued: collections.Counter[str] = collections.Counter()
        self.stretches: collections.Counter[str] = collections.Counter()
        self.spoken: list[dict[str, Any]] = []
        self.muted = False      # the room pan after the shelf pass (app.room_sweep) is far and off the shelf by design: no capture notices

    def gap_ok(self, t: float | None = None) -> bool:
        if t is not None:
            self.clock = max(self.clock, float(t))
        return self.last_spoken is None or self.clock - self.last_spoken >= self.gap_s

    def _state(self, fam: str, t: float) -> dict[str, Any]:
        st = self._st.get(fam)
        if st is None or t - st["last_bad"] >= self.rearm_s:
            st = self._st[fam] = {"start": t, "last_bad": t, "said": set()}
            self.stretches[fam] += 1
        st["last_bad"] = max(st["last_bad"], t)
        return st

    def offer(self, code: str, t: float) -> bool:
        """The condition behind ``code`` holds at capture time ``t``. True = speak it now (and it is counted)."""
        self.clock = max(self.clock, float(t))
        if self.muted:
            return False
        fam, slot = FAMILY.get(code, code), SLOT.get(code, "main")
        st = self._state(fam, float(t))
        if (slot in st["said"] and code not in REPEATING) or not self.gap_ok():
            return False
        st["said"].add(slot)
        if code == "recapture":
            st["said"].add("main")
            rv = self._st.get("revisit")
            if rv is None or t - rv["last_bad"] >= self.rearm_s:
                self._st["revisit"] = {"start": float(t), "last_bad": float(t), "said": {"main"}}   # not counted as a bad stretch
            else:
                rv["said"].add("main")
        self.note_spoken(code)
        return True

    def note_spoken(self, code: str, t: float | None = None) -> None:
        """Record a notice spoken outside the stretch logic (APP GUIDANCE), so the global gap covers it too."""
        if t is not None:
            self.clock = max(self.clock, float(t))
        self.last_spoken = self.clock
        self.issued[code] += 1
        self.spoken.append({"code": code, "t_s": round(self.clock, 2)})

    def summary(self) -> dict[str, Any]:
        return {"rearm_s": self.rearm_s, "gap_s": self.gap_s, "muted_room_pan": self.muted, "stretches": dict(self.stretches), "spoken": self.spoken[-40:]}
