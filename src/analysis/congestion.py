"""Traffic congestion - the worked example of adding a use case.

This file is the acceptance test for the analyzer seam: it was added with NO
edit to pipeline.py, no new storage column, and no change to any other
analyzer. It is frame-level rather than object-level, so it also demonstrates
that one protocol covers both shapes.

Level is derived from three cheap signals, all already on the context:
  density    how many on-road objects are present
  occupancy  what fraction of the frame their boxes cover
  motion     mean centroid displacement, from the previous positions the
             tracker store already keeps

Motion is what separates "busy but flowing" from "jammed": a full frame of
stationary vehicles is congestion, a full frame of moving ones is not.
A level must PERSIST for `min_frames` before it is reported, so one noisy
frame does not raise an alert.
"""

import cv2

from ..detectors.classes import ON_ROAD
from .base import register

FREE, BUSY, HEAVY, JAMMED = "free", "busy", "heavy", "jammed"
_ORDER = {FREE: 0, BUSY: 1, HEAVY: 2, JAMMED: 3}
_COLOR = {FREE: (0, 200, 0), BUSY: (0, 200, 200),
          HEAVY: (0, 140, 255), JAMMED: (0, 0, 255)}


class CongestionAnalyzer:
    name = "congestion"

    def __init__(self, cfg: dict):
        c = (cfg or {}).get("congestion", {}) or {}
        self.busy_n = int(c.get("busy_count", 6))
        self.heavy_n = int(c.get("heavy_count", 12))
        self.jam_occupancy = float(c.get("jam_occupancy", 0.28))
        self.slow_px = float(c.get("slow_px_per_frame", 2.0))
        self.min_frames = int(c.get("min_frames", 15))
        self.draw_overlay = bool(c.get("draw", True))
        self.level = FREE
        self._candidate = FREE
        self._streak = 0
        self._frames_at = {FREE: 0, BUSY: 0, HEAVY: 0, JAMMED: 0}
        self._peak = FREE
        self._fps = 30.0

    def setup(self, source, cfg: dict):
        src_fps = float(getattr(source, "fps", 30.0) or 30.0)
        # seconds_at must tick on ANALYSED frames, not source frames: at
        # analyse_fps=5 on a 30fps source each analysed frame is 6 source
        # frames, so dividing by source fps would inflate seconds ~6x.
        try:
            target = (cfg or {}).get("processing", {}).get("analyse_fps")
            self._fps = min(src_fps, float(target)) if target else src_fps
        except (TypeError, ValueError):
            self._fps = src_fps
        self._area = max(1, int(getattr(source, "width", 1)) *
                         int(getattr(source, "height", 1)))

    def _measure(self, ctx):
        on_road = [d for d in ctx.detections if d.group in ON_ROAD]
        occupancy = sum(d.area for d in on_road) / self._area
        moved, n = 0.0, 0
        for d in on_road:
            if d.track_id is None:
                continue
            # Pipeline-captured previous centroid (see lanes.py for why).
            prev = d.extra.get("prev_xy")
            if not prev or prev[0] is None:
                continue
            cx, cy = d.centroid
            moved += ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
            n += 1
        return len(on_road), occupancy, (moved / n if n else None)

    def _classify(self, count, occupancy, motion) -> str:
        if count < self.busy_n:
            return FREE
        stalled = motion is not None and motion < self.slow_px
        if occupancy >= self.jam_occupancy and stalled:
            return JAMMED
        if count >= self.heavy_n:
            return HEAVY if not stalled else JAMMED
        return BUSY

    def process(self, ctx):
        count, occupancy, motion = self._measure(ctx)
        candidate = self._classify(count, occupancy, motion)
        # Require persistence before changing the reported level.
        if candidate == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = candidate, 1
        if self._streak >= self.min_frames and candidate != self.level:
            self.level = candidate
            if _ORDER[candidate] > _ORDER[self._peak]:
                self._peak = candidate
            ctx.emit("congestion", {"level": candidate, "count": count,
                                    "occupancy": round(occupancy, 4),
                                    "motion_px": None if motion is None
                                    else round(motion, 2)})
        self._frames_at[self.level] = self._frames_at.get(self.level, 0) + 1
        if self.draw_overlay:
            cv2.putText(ctx.annotated,
                        f"traffic: {self.level} (n={count} occ={occupancy:.0%})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        _COLOR[self.level], 2)

    def summary(self) -> dict:
        return {"level_final": self.level, "level_peak": self._peak,
                "seconds_at": {k: round(v / self._fps, 1)
                               for k, v in self._frames_at.items() if v}}


register("congestion", CongestionAnalyzer)
