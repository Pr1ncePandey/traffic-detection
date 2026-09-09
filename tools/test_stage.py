"""Self-test for the analysis scheduler (src/analysis/stage.py).

    python tools/test_stage.py

Checks the properties the three-phase split is supposed to buy, because none
of them are visible in a normal run: the shipped analyzers are one staged
(lanes) plus three legacy ones, so the parallel path never engages and a
regression in it would go unnoticed until the first analysis is migrated.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.analysis.base import Findings, is_concurrent, is_staged
from src.analysis.stage import AnalysisStage
from src.runtime.context import Detection, FrameContext
from src.trackers.store import TrackStore

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


class Staged:
    """A staged analyzer that records which thread each phase ran on."""

    def __init__(self, name, concurrent=True, work=0.0, boom=False, order=None):
        self.name = name
        self.concurrent = concurrent
        self._work = work
        self._boom = boom
        self._order = order if order is not None else []
        self.compute_threads = []
        self.applied = 0
        self.drawn = 0

    def setup(self, source, cfg):
        pass

    def compute(self, view):
        self.compute_threads.append(threading.current_thread().name)
        if self._boom:
            raise RuntimeError(f"{self.name} exploded")
        if self._work:
            time.sleep(self._work)
        found = Findings()
        for box in view.tracked():
            found.set(box.track_id, lane_id=f"{self.name}-lane",
                      extra={f"{self.name}_seen": True})
        found.event("tick", {"analyzer": self.name})
        found.frame["n"] = len(view.boxes)
        found.overlay = {"name": self.name}
        return found

    def apply(self, ctx, findings):
        self._order.append(f"apply:{self.name}")
        self.applied += 1

    def draw(self, ctx, findings):
        self._order.append(f"draw:{self.name}")
        self.drawn += 1

    def summary(self):
        return {"applied": self.applied, "drawn": self.drawn}

    def forget(self, tid):
        self._order.append(f"forget:{self.name}:{tid}")


class Legacy:
    """The old single-phase shape, which must keep working untouched."""

    def __init__(self, name, order=None, boom=False):
        self.name = name
        self._order = order if order is not None else []
        self._boom = boom
        self.calls = 0
        self.saw_lane = None

    def setup(self, source, cfg):
        pass

    def process(self, ctx):
        self._order.append(f"process:{self.name}")
        self.calls += 1
        if self._boom:
            raise RuntimeError("legacy exploded")
        # Reads what a staged analyzer wrote, to prove apply is in order.
        self.saw_lane = ctx.detections[0].lane_id if ctx.detections else None

    def summary(self):
        return {"calls": self.calls}


def make_ctx(n=3):
    raw = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = [Detection(cls_id=2, cls_name="car", group="vehicle", conf=0.9,
                      bbox=(100 + 50 * i, 300, 160 + 50 * i, 400), track_id=i + 1)
            for i in range(n)]
    return FrameContext(frame_no=1, timestamp=1.0, raw=raw,
                        annotated=raw.copy(), detections=dets,
                        store=TrackStore(), source=None)


section("1. the shapes are told apart correctly")
check("a staged analyzer is detected", is_staged(Staged("a")))
check("and is concurrent when it says so", is_concurrent(Staged("a")))
check("a staged analyzer that opts out is not concurrent",
      is_staged(Staged("b", concurrent=False))
      and not is_concurrent(Staged("b", concurrent=False)))
check("a legacy analyzer is neither",
      not is_staged(Legacy("c")) and not is_concurrent(Legacy("c")))

section("2. the read-only view is a real snapshot")
ctx = make_ctx()
view = ctx.view()
check("boxes match the detections", len(view.boxes) == 3)
check("view carries frame geometry", (view.width, view.height) == (1280, 720))
check("ground point is bottom-centre, not the centroid",
      view.boxes[0].ground_point == (130, 400)
      and view.boxes[0].centroid == (130, 350))
check("a TrackedBox cannot be written to (frozen tuple)",
      isinstance(view.boxes[0], tuple)
      and not hasattr(view.boxes[0], "__dict__"))
try:
    view.boxes[0].bbox = (0, 0, 1, 1)
    mutated = True
except AttributeError:
    mutated = False
check("assigning to a snapshot field raises", not mutated)

section("3. compute really does run on worker threads when parallel")
a, b = Staged("a", work=0.05), Staged("b", work=0.05)
stage = AnalysisStage([a, b], parallel=True)
check("the stage engages parallelism with 2 concurrent analyzers",
      stage.parallel, stage.describe())
ctx = make_ctx()
t0 = time.perf_counter()
stage.run(ctx)
elapsed = time.perf_counter() - t0
check("two 50ms computes overlap (wall clock < serial sum)",
      elapsed < 0.09, f"{elapsed*1000:.0f}ms")
main = threading.current_thread().name
check("compute ran off the main thread",
      all(t != main for t in a.compute_threads + b.compute_threads),
      str(a.compute_threads + b.compute_threads))
stage.close()

section("4. one concurrent analyzer means no pool (nothing to overlap)")
solo = AnalysisStage([Staged("only")], parallel=True)
check("parallelism is declined", not solo.parallel, solo.describe())
solo.close()

section("5. findings are written back onto the detections")
one = Staged("x")
stage = AnalysisStage([one])
ctx = make_ctx()
stage.run(ctx)
check("a known field lands on the Detection attribute",
      all(d.lane_id == "x-lane" for d in ctx.detections))
check("an `extra` dict is merged, not replaced",
      all(d.extra.get("x_seen") for d in ctx.detections))
check("events are replayed through ctx.emit",
      len(ctx.events) == 1 and ctx.events[0]["kind"] == "tick")
stage.close()

section("6. apply and draw run serially, in config order")
order = []
first, second = Staged("first", order=order), Staged("second", order=order)
legacy = Legacy("mid", order=order)
stage = AnalysisStage([first, legacy, second], parallel=True)
ctx = make_ctx()
stage.run(ctx)
check("apply order follows the config list",
      order.index("apply:first") < order.index("apply:second"), str(order))
check("a legacy process() runs in the apply pass, in its listed position",
      order.index("apply:first") < order.index("process:mid")
      < order.index("apply:second"), str(order))
check("all draws happen after all applies",
      max(i for i, s in enumerate(order) if s.startswith("apply")
          or s.startswith("process"))
      < min(i for i, s in enumerate(order) if s.startswith("draw")), str(order))
check("a legacy analyzer sees what an earlier staged one wrote",
      legacy.saw_lane == "first-lane", str(legacy.saw_lane))
stage.close()

section("7. one broken analyzer does not take down the frame")
good, bad = Staged("good"), Staged("bad", boom=True)
badlegacy = Legacy("badlegacy", boom=True)
stage = AnalysisStage([bad, good, badlegacy], parallel=True)
ctx = make_ctx()
stage.run(ctx)
check("the healthy analyzer still applied", good.applied == 1)
check("the healthy analyzer still drew", good.drawn == 1)
check("the broken compute is not applied", bad.applied == 0)
check("both failures are counted, not printed per frame",
      len(stage.errors) == 2, str(stage.errors))
summaries = stage.summaries()
check("failures surface in the summary", "_analysis_errors" in summaries)
check("timing surfaces in the summary",
      summaries["_analysis_timing"]["parallel"] is True)
for _ in range(5):
    stage.run(make_ctx())
check("a repeatedly broken analyzer accumulates a count",
      stage.errors["bad"]["count"] == 6, str(stage.errors))
stage.close()

section("8. retirement reaches every analyzer that wants it")
order = []
s1 = Staged("s1", order=order)
stage = AnalysisStage([s1, Legacy("l1", order=order)])
stage.forget(42)
check("forget() is forwarded to analyzers that define it",
      "forget:s1:42" in order, str(order))
stage.forget(43)   # legacy has no forget(); must not raise
check("and is safely skipped for those that do not", True)
stage.close()

section("9. an analyzer returning nothing is not an error")
class Silent(Staged):
    def compute(self, view):
        return None

silent = Silent("silent")
stage = AnalysisStage([silent])
stage.run(make_ctx())
check("a None result is treated as empty findings", not stage.errors,
      str(stage.errors))
stage.close()

print(f"\n{'=' * 62}")
print(f"{_passed} passed, {len(_failed)} failed")
for name in _failed:
    print(f"  FAILED: {name}")
sys.exit(1 if _failed else 0)
