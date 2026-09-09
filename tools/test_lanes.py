"""Self-test for the wrong-side rule. No video, no model, no pytest.

    python tools/test_lanes.py

Every case below is a bug that was live in the previous implementation, or a
guarantee that replaced it. It runs in milliseconds and needs nothing but the
standard library, so there is no excuse for not running it before trusting a
lane change.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.geometry import (Line, clip_to_halfplane, ensure_simple,
                                   frame_ring, is_simple, point_in_polygon,
                                   scale_points, to_ratios)
from src.analysis.lane_calibration import MotionSurvey
from src.analysis.lane_model import (EDGE_CLIPPED, OK, TRACK_JUMP, WRONG_LANE,
                                     WRONG_WAY, LaneModel, Thresholds)

W, H = 1280, 720
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


def lane(name, ring, flow, allowed=None):
    return {"name": name, "polygon": ring, "flow": flow,
            "allowed": allowed or [], "units": "pixel"}


def model_with(*lanes, divider=None, rules=None):
    return LaneModel.from_config(list(lanes), W, H, divider_cfg=divider,
                                 rules=rules or {}, units="pixel",
                                 verbose=False)


def drive(model, tid, start, step, frames, box_h=60, cls_name="car",
          box_w=None, warmup=False):
    """Walk a synthetic track and return the last verdict."""
    box_w = box_w or box_h
    x, y = float(start[0]), float(start[1])
    verdict = None
    for i in range(frames):
        bbox = (int(x - box_w / 2), int(y - box_h), int(x + box_w / 2), int(y))
        verdict = model.evaluate(tid, bbox, cls_name, frame_no=i, warmup=warmup)
        x += step[0]
        y += step[1]
    return verdict


# --------------------------------------------------------------------------
section("1. polygon corner order (tools/draw_lanes.py wrote click order)")
# Clicking a rectangle's corners in reading order: TL, TR, BL, BR.
clicks = [[100, 100], [500, 100], [100, 400], [500, 400]]
check("a reading-order click sequence really is self-intersecting",
      not is_simple(clicks))
check("the interior of that bowtie tests as OUTSIDE (the old bug)",
      not point_in_polygon(150, 150, clicks))
repaired, was_repaired = ensure_simple(clicks, "rect", verbose=False)
check("ensure_simple repairs it", was_repaired and is_simple(repaired))
check("the whole interior now tests as inside",
      all(point_in_polygon(x, y, repaired)
          for x, y in [(150, 150), (450, 350), (300, 250), (110, 390)]))
check("a point outside stays outside",
      not point_in_polygon(600, 250, repaired))
already = [[0, 0], [435, 0], [64, 720], [0, 720]]
kept, touched = ensure_simple(already, "wedge", verbose=False)
check("an already-simple ring is left byte-identical",
      kept == already and not touched)

section("2. ratio vs pixel units (the (1,1) explosion)")
check("ratio polygon scales to pixels",
      scale_points([[0, 0], [0.5, 0], [0.5, 1]], W, H) == [[0, 0], [640, 0], [640, 720]])
check("a pixel polygon containing (1,1) is NOT scaled",
      scale_points([[1, 1], [435, 0], [64, 720]], W, H) == [[1, 1], [435, 0], [64, 720]])
check("an explicit unit beats the guess",
      scale_points([[1, 1]], W, H, units="ratio") == [[1280, 720]])
check("ratios round-trip", to_ratios([[640, 360]], W, H) == [[0.5, 0.5]])

section("3. the divider is a line, and both sides are well defined")
div = Line((0.356 * W, 0), (0.072 * W, H))
check("opposite sides get opposite signs",
      div.side(1000, 360) == -div.side(10, 360) != 0)
check("a point on the line is neither side (with a margin)",
      div.side(div.x_at_y(360), 360, margin=2.0) == 0)
check("distance grows with offset",
      div.distance(1000, 360) > div.distance(300, 360))
ends = div.clipped_to_frame(W, H)
check("the divider spans the whole frame for drawing",
      ends is not None and {ends[0][1], ends[1][1]} == {0, H - 1}, str(ends))
left = clip_to_halfplane(frame_ring(W, H), div, div.side(10, 360))
right = clip_to_halfplane(frame_ring(W, H), div, -div.side(10, 360))
check("splitting the frame by the divider tiles it with no gap",
      point_in_polygon(10, 360, left) and point_in_polygon(1000, 360, right)
      and not point_in_polygon(1000, 360, left))

section("4. a vehicle going the wrong way IS flagged")
up = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
v = drive(up, 1, (640, 300), (0, 12), 20)          # travelling DOWN
check("wrong_way after the confirmation window", v.flag == WRONG_WAY, repr(v))
check("the verdict carries its evidence",
      v.alignment is not None and v.alignment < -0.9 and v.travel_px > 0,
      repr(v.detail()))

section("5. a vehicle going the right way is NOT flagged")
ok = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
v = drive(ok, 2, (640, 600), (0, -12), 20)
check("stays ok", v.flag == OK, repr(v))

section("6. far, slow traffic gets a verdict (the flat 4px/frame gate did not)")
# Measured on samples/plate_test.mp4: far objects move a median of 1 px per
# analysed frame, so the old `dy > 4` test could never fire up there.
far = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
v = drive(far, 3, (640, 200), (0, 1.2), 30, box_h=30)   # 1.2 px/frame, downward
check("1.2 px/frame over the window is enough to judge",
      v.flag == WRONG_WAY, repr(v))
jitter = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
for i in range(30):
    y = 300 + (2 if i % 2 else -2)          # oscillating, going nowhere
    jitter.evaluate(4, (620, y - 60, 660, y), "car", frame_no=i)
v = jitter.evaluate(4, (620, 240, 660, 300), "car", frame_no=99)
check("a jittering stationary box is still not flagged",
      v.flag == OK, repr(v))

section("7. a diagonal road (no cardinal direction describes it)")
# Flow mostly to the RIGHT and slightly up: the old rule could only be given
# `direction: up`, and would then read any vehicle with dy > 0 as opposing.
diag = model_with(lane("road", frame_ring(W, H), [0.98, -0.2]))
v = drive(diag, 5, (200, 400), (11.0, 0.6), 20)   # correct, but drifting DOWN
check("travelling along a diagonal lane is ok even with dy > 0",
      v.flag == OK, repr(v))
v = drive(diag, 6, (1000, 400), (-11.0, -0.6), 20)
check("travelling back along it is wrong_way", v.flag == WRONG_WAY, repr(v))

section("8. no verdict where the position cannot be trusted")
clipped = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
v = clipped.evaluate(7, (600, 500, 700, H - 1), "car", frame_no=1)
check("a box touching the bottom edge is skipped entirely",
      v.flag == OK and v.reason == EDGE_CLIPPED, repr(v))
side = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
v = side.evaluate(8, (0, 300, 90, 400), "car", frame_no=1)
check("so is one against the left edge", v.reason == EDGE_CLIPPED, repr(v))

section("9. crossing a lane is not driving the wrong way")
cross = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
v = drive(cross, 9, (300, 400), (12, 0), 20)      # pure sideways
check("sideways travel yields no wrong_way", v.flag == OK, repr(v))

section("10. confirmation tolerates gaps (the old streak reset on one frame)")
# 5 opposing samples out of 8, with correct-direction frames interleaved.
gappy = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]),
                   rules={"confirm_window": 8, "confirm_count": 5,
                          "history": 2, "min_travel_px": 4.0,
                          "min_travel_rel": 0.0})
y, flagged = 400.0, False
for i in range(24):
    step = 10.0 if i % 4 != 3 else -10.0      # 3 of every 4 frames wrong-way
    y += step
    v = gappy.evaluate(10, (620, int(y) - 60, 660, int(y)), "car", frame_no=i)
    flagged = flagged or v.flag == WRONG_WAY
check("an interrupted wrong-way run is still confirmed", flagged)

section("10b. a lost-and-refound track is not differenced across the gap")
# Regression for the ONLY false positive in a full 812-frame run of
# samples/plate_test.mp4. Track 349's ground point was last seen at (118, 487),
# the track vanished for 26 frames, and it was re-acquired at (77, 705) - 218 px
# LOWER. The velocity window spanned the gap and read that as 190 px of
# downward travel on a road that flows up (alignment -0.99), while the vehicle
# actually drove correctly up the frame for the next 40 frames. Its class also
# flipped truck -> motorcycle across the gap, i.e. the id had been reused.
gap = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
for i in range(8):
    gap.evaluate(349, (60, 267 - i * 2, 176, 487 - i * 2), "truck",
                 frame_no=323 + i)
v = None
for i in range(40):
    y = 705 - i * 7
    v = gap.evaluate(349, (20, y - 150, 134, y), "truck", frame_no=350 + i)
check("the re-acquired track is never flagged wrong_way", v.flag == OK, repr(v))

fresh = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
for i in range(8):
    fresh.evaluate(349, (60, 267 - i * 2, 176, 487 - i * 2), "truck",
                   frame_no=323 + i)
first = fresh.evaluate(349, (20, 555, 134, 705), "truck", frame_no=350)
check("the first sample after a 26-frame gap yields no heading",
      first.reason == TRACK_JUMP, repr(first))

jump = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
for i in range(6):
    jump.evaluate(20, (600, 340 + i, 700, 400 + i), "car", frame_no=i)
v = jump.evaluate(20, (600, 590, 700, 650), "car", frame_no=7)
check("an implausible one-frame jump also resets the history",
      v.reason == TRACK_JUMP, repr(v))

after = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
v = drive(after, 21, (640, 200), (0, 12), 20)
check("a genuine wrong-way track is still caught with the guard in place",
      v.flag == WRONG_WAY, repr(v))

section("11. warmup suppresses alerts but still measures")
warm = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
v = drive(warm, 11, (640, 200), (0, 12), 20, warmup=True)
check("no flag is raised during warmup", v.flag == OK, repr(v))
# Same track, next frame, still travelling the wrong way.
y = 200 + 20 * 12
v = warm.evaluate(11, (610, y - 60, 670, y), "car", frame_no=20)
check("and the evidence gathered during it counts immediately after",
      v.flag == WRONG_WAY, repr(v))

section("12. restricted lanes are a separate rule from wrong-way")
bus = model_with(lane("buslane", frame_ring(W, H), [0.0, -1.0], allowed=["bus"]))
v = drive(bus, 12, (640, 600), (0, -12), 20, cls_name="truck")
check("a truck driving correctly in a bus lane is wrong_lane, not wrong_way",
      v.flag == WRONG_LANE, repr(v))
v = drive(bus, 13, (640, 600), (0, -12), 20, cls_name="bus")
check("a bus in the bus lane is ok", v.flag == OK, repr(v))

section("13. lane membership near the divider is withheld, not guessed")
d = Line((640, 0), (640, H))
two = model_with(lane("west", clip_to_halfplane(frame_ring(W, H), d, d.side(10, 360)),
                      [0.0, -1.0]),
                 lane("east", clip_to_halfplane(frame_ring(W, H), d, -d.side(10, 360)),
                      [0.0, 1.0]),
                 divider={"points": [[640, 0], [640, H]], "units": "pixel"})
found, confident = two.lane_at(300, 400, box_h=60)
check("well inside a lane is confident", found.name == "west" and confident)
found, confident = two.lane_at(641, 400, box_h=60)
check("right on the divider is NOT confident", not confident)
v = drive(two, 14, (642, 400), (0, -12), 20)
check("a vehicle straddling the divider is not flagged",
      v.flag == OK, repr(v))

section("14. calibration: one-way traffic does not invent a divider")
one = MotionSurvey(W, H)
for tid in range(8):
    for i in range(12):
        x = 500 + tid * 40
        y = 600 - i * 30
        one.observe(tid, (x - 30, y - 60, x + 30, y))
s = one.suggest()
check("one flow -> one lane", len(s.lanes) == 1, s.note)
check("one flow -> NO divider", s.divider is None and not s.two_way)
check("and its flow points the way traffic went", s.lanes[0]["flow"][1] < -0.9,
      str(s.lanes[0]["flow"]))

section("15. calibration: real two-way traffic gets a fitted divider")
two_way = MotionSurvey(W, H)
for tid in range(8):                                    # right side, going up
    for i in range(12):
        x, y = 900 + tid * 20, 650 - i * 40
        two_way.observe(tid, (x - 30, y - 60, x + 30, y))
for tid in range(100, 106):                             # left side, coming down
    for i in range(12):
        x, y = 200 + (tid - 100) * 20, 60 + i * 40
        two_way.observe(tid, (x - 30, y - 60, x + 30, y))
s = two_way.suggest()
check("two flows -> two lanes", len(s.lanes) == 2 and s.two_way, s.note)
check("a divider was fitted", s.divider is not None)
if s.divider is not None:
    mid = s.divider.x_at_y(H / 2)
    check("it sits between the two flows", mid is not None and 300 < mid < 900,
          f"x at mid-height = {mid}")
check("the two lanes flow in opposite directions",
      s.lanes[0]["flow"][1] * s.lanes[1]["flow"][1] < 0)

section("16. verification catches a reversed configured flow")
# This is the samples/plate_test.mp4 failure, in miniature: traffic goes UP,
# the config says the lane flows DOWN.
survey = MotionSurvey(W, H)
for tid in range(6):
    for i in range(12):
        x, y = 400 + tid * 30, 640 - i * 40
        survey.observe(tid, (x - 30, y - 60, x + 30, y))
wrong = model_with(lane("road", frame_ring(W, H), [0.0, 1.0]))
report = survey.verify(wrong)
check("the mismatch is detected", not report.ok, report.report())
check("and it names the lane and both directions",
      report.opposing[0]["lane"] == "road"
      and report.opposing[0]["alignment"] < -0.9)
right = model_with(lane("road", frame_ring(W, H), [0.0, -1.0]))
check("a correct config verifies clean", survey.verify(right).ok)
check("a lane with no observed traffic is 'no-data', not 'agrees'",
      survey.verify(model_with(
          lane("empty", [[0, 0], [50, 0], [50, 50], [0, 50]], [0.0, -1.0])
      )).rows[0]["verdict"] == "no-data")

section("17. thresholds really do scale with depth")
t = Thresholds({})
check("a near object needs more travel than a far one",
      t.travel_gate(300) > t.travel_gate(30))
check("the far gate never drops below the floor",
      t.travel_gate(1) == t.min_travel_px)
check("the divider keep-out scales too",
      t.divider_margin(300) > t.divider_margin(30) >= t.divider_margin_min_px)
check("the jump limit scales too", t.step_limit(300) > t.step_limit(30))

print(f"\n{'=' * 62}")
print(f"{_passed} passed, {len(_failed)} failed")
for name in _failed:
    print(f"  FAILED: {name}")
sys.exit(1 if _failed else 0)
