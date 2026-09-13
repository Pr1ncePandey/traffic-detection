"""Self-test for the congestion analyzer (src/analysis/congestion.py).

    python tools/test_congestion.py

Verifies the level logic with
fabricated detections (no video, no models):

  free    few objects no matter the motion
  busy    enough objects, flowing
  heavy   many objects, flowing
  jammed  many objects AND stalled (or dense + stalled)
  persist a level must hold `min_frames` before it flips or emits
  payload the congestion event carries {level,count,occupancy,motion_px}
  summary seconds math honors analyse fps, not source fps
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.analysis.congestion import CongestionAnalyzer
from src.runtime.context import AnalysisView, Detection, TrackedBox

_passed, _failed = 0, []


def check(label, condition, detail=""):
    global _passed
    if condition:
        _passed += 1
        print(f"  ok   {label}")
    else:
        _failed.append(label)
        print(f"  FAIL {label}{('  <- ' + detail) if detail else ''}")


def section(title):
    print(f"\n{title}")


class FakeSource:
    fps = 30.0
    width = 1280
    height = 720


class FakeCtx:
    """Mutable frame stand-in: collects emitted events, owns annotated."""

    def __init__(self):
        self.events = []
        self.annotated = np.zeros((720, 1280, 3), dtype=np.uint8)

    def emit(self, kind, detail, track_id=None):
        self.events.append({"kind": kind, "detail": detail,
                            "track_id": track_id})


def box(i, cx, cy, w=120, h=80, track_id=None, group="vehicle"):
    """One immutable snapshot box with a previous position for motion."""
    d = Detection(cls_id=2, cls_name="car", group=group, conf=0.9,
                  bbox=(cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2),
                  track_id=track_id if track_id is not None else 1000 + i)
    return d


def frame(n, motion, w=120, h=80, start=(640, 360), step=40):
    """n boxes in a row; each carries prev_xy `motion` px behind."""
    boxes = []
    for i in range(n):
        cx = start[0] + (i - n // 2) * step
        d = box(i, cx, start[1], w, h)
        d.extra["prev_xy"] = (cx - motion, start[1])
        boxes.append(TrackedBox.of(d))
    return AnalysisView(frame_no=0, timestamp=0.0, boxes=tuple(boxes),
                        width=1280, height=720)


def fresh(**kw):
    cfg = {"congestion": {"busy_count": 6, "heavy_count": 12,
                          "jam_occupancy": 0.28, "slow_px_per_frame": 2.0,
                          "min_frames": 15, "draw": False}}
    cfg["congestion"].update(kw)
    a = CongestionAnalyzer(cfg)
    a.setup(FakeSource(), cfg)
    return a


def run_frames(a, n, motion, **kw):
    """compute() + replay queued events through ctx.emit, like the scheduler."""
    events = []
    for _ in range(n):
        ctx = FakeCtx()
        found = a.compute(frame(kw.get("count", 12), motion,
                                w=kw.get("w", 120), h=kw.get("h", 80)))
        for kind, detail, tid in found.events:
            ctx.emit(kind, detail, tid)
        events.extend(ctx.events)
    return events


section("levels")
a = fresh()
run_frames(a, 20, motion=0.5)
check("12 stalled boxes -> jammed", a.level == "jammed", a.level)

a = fresh()
run_frames(a, 20, motion=10.0)
check("12 moving boxes -> heavy (dense but flowing)", a.level == "heavy", a.level)

a = fresh()
run_frames(a, 20, motion=10.0, count=8)
check("8 moving boxes -> busy", a.level == "busy", a.level)

a = fresh()
run_frames(a, 20, motion=0.0, count=3)
check("3 stalled boxes -> free (too few)", a.level == "free", a.level)

a = fresh()
run_frames(a, 20, motion=0.5, count=8, w=400, h=300)
check("8 huge stalled boxes -> jammed via occupancy", a.level == "jammed", a.level)

section("persistence gate")
a = fresh()
events = []
for _ in range(14):  # one short of min_frames
    ctx = FakeCtx()
    found = a.compute(frame(12, motion=0.5))
    for kind, detail, tid in found.events:
        ctx.emit(kind, detail, tid)
    events.extend(ctx.events)
check("14 jammed frames -> no event yet",
      a.level == "free" and not [e for e in events if e["kind"] == "congestion"],
      f"{a.level} {len(events)} events")
ctx = FakeCtx()
found = a.compute(frame(12, motion=0.5))
for kind, detail, tid in found.events:
    ctx.emit(kind, detail, tid)
events.extend(ctx.events)
check("15th jammed frame -> flips + emits",
      a.level == "jammed" and len([e for e in events if e["kind"] == "congestion"]) == 1,
      f"{a.level} {len(events)} events")

section("event payload")
ev = [e for e in events if e["kind"] == "congestion"][0]["detail"]
check("payload has level/count/occupancy/motion_px",
      set(ev) == {"level", "count", "occupancy", "motion_px"}, str(sorted(ev)))
check("payload values sane",
      ev["level"] == "jammed" and ev["count"] == 12 and ev["motion_px"] == 0.5,
      str(ev))

section("summary clock")
# Note: the flip lands ON frame 15, so 40 frames -> 14 free + 26 busy.
a = fresh()
run_frames(a, 40, motion=10.0, count=8)  # busy motion at 30fps source
s = a.summary()
check("seconds_at uses source fps when no analyse_fps (busy=26/30s)",
      s["seconds_at"].get("busy") == round(26 / 30, 1), str(s["seconds_at"]))
b = CongestionAnalyzer({"congestion": {}, "processing": {"analyse_fps": 5}})
b.setup(FakeSource(), {"processing": {"analyse_fps": 5}})
for _ in range(40):
    b.compute(frame(8, motion=10.0))
s = b.summary()
check("seconds_at honors analyse_fps=5 (26 frames = 5.2s, not 0.9s)",
      s["seconds_at"].get("busy") == 5.2, str(s["seconds_at"]))

section("overlay")
a = fresh(draw=True)
ctx = FakeCtx()
found = a.compute(frame(12, motion=0.5))
a.draw(ctx, found)
check("overlay paints (annotated non-blank)", ctx.annotated.any(), "blank frame")

print(f"\n{_passed} passed, {len(_failed)} failed")
sys.exit(1 if _failed else 0)
