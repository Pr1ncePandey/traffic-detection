"""Self-test for the human behaviour analysis (src/analysis/behaviour.py).

    python tools/test_behaviour.py

Fabricated boxes, no video, no models. Behaviour rules fail quietly - an
intrusion per wobble along a zone edge, a rider counted as a jaywalker, a
tracker swap read as a sprint, a crowd that flickers - so each is checked,
along with the incidents the rules raise.
"""

import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.base import build, enabled_names  # noqa: E402
from src.analysis.behaviour import BehaviourAnalyzer  # noqa: E402
from src.config import load_for_camera  # noqa: E402
from src.incidents import NEVER, IncidentPolicy  # noqa: E402
from src.runtime.context import AnalysisView, Detection, TrackedBox  # noqa: E402

_passed, _failed = 0, []


def check(label, condition, detail=""):
    global _passed
    if condition:
        _passed += 1
        print(f"  ok   {label}")
    else:
        _failed.append(label)
        print(f"  FAIL {label}{('  <- ' + str(detail)) if detail else ''}")


class Source:
    fps, width, height = 10.0, 1280, 720


class Ctx:
    def __init__(self):
        self.annotated = np.full((720, 1280, 3), 100, np.uint8)


def person(gx, gy, tid=1, h=200):
    """A person standing with their feet at (gx, gy)."""
    w = h // 3
    return TrackedBox.of(Detection(0, "person", "person", 0.9,
                                   (gx - w // 2, gy - h, gx + w // 2, gy), track_id=tid))


def vehicle(x1, y1, x2, y2, cls="motorcycle", tid=900):
    return TrackedBox.of(Detection(3, cls, "vehicle", 0.9, (x1, y1, x2, y2), track_id=tid))


def view(boxes, ts):
    return AnalysisView(frame_no=int(ts * 10), timestamp=ts, boxes=tuple(boxes),
                        width=1280, height=720)


def fresh(zones=None, **kw):
    cfg = {"analyses": {"behaviour": {"enabled": True, "zones": zones or [], **kw}}}
    b = BehaviourAnalyzer(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        b.setup(Source(), cfg)
    return b


def ticks(t0, t1, step=0.1):
    return [round(t0 + i * step, 3) for i in range(int(round((t1 - t0) / step)) + 1)]


def run(b, frames):
    """frames: [(ts, boxes)] -> (all events, last Findings)"""
    events, last = [], None
    for ts, boxes in frames:
        last = b.compute(view(boxes, ts))
        events += last.events
    return events, last


def of(events, kind):
    return [e for e in events if e[0] == kind]


ROAD = {"name": "road", "polygon": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]],
        "restricted": True}          # x < 640, y < 360 in pixels

print("\nzones")
b = fresh([ROAD, {"name": "all", "loitering_s": 5},
           {"name": "bad", "polygon": [[0, 0], [1, 1]]}, {"name": "road"}])
check("ratio polygon becomes pixels", b.zones[0].ring == [[0, 0], [640, 0], [640, 360], [0, 360]],
      b.zones[0].ring)
check("a zone with no polygon is the whole frame", b.zones[1].whole_frame and b.zones[1].contains(1270, 710))
check("a zone with under 3 corners and a duplicate name are skipped",
      [z.name for z in b.zones] == ["road", "all"], [z.name for z in b.zones])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    BehaviourAnalyzer({"analyses": {"behaviour": {"zones": ["oops"]}}}).setup(Source(), {})
check("a zone that is not a mapping says so", "not a mapping" in buf.getvalue())

print("\nintrusion")
b = fresh([ROAD])
ev, _ = run(b, [(t, [person(300, 300)]) for t in ticks(0, 0.8)])
check("no intrusion before intrusion_s", not of(ev, "intrusion"))
ev, f = run(b, [(t, [person(300, 300)]) for t in ticks(0.9, 3.0)])
check("intrusion fires ONCE after 1 s inside", len(of(ev, "intrusion")) == 1, len(of(ev, "intrusion")))
kind, detail, tid = of(ev, "intrusion")[0]
check("event names the zone, the track and the dwell",
      detail["zone"] == "road" and tid == 1 and detail["dwell_s"] >= 0.99, detail)
check("the person is tagged and boxed for the video",
      f.per_track[1]["extra"]["behaviour"] == ["INTRUSION road"] and f.overlay["flags"], f.per_track)
b = fresh([ROAD])
ev, _ = run(b, [(t, [person(300, 400)]) for t in ticks(0, 3)])
check("feet decide: box middle over the zone, feet outside = no intrusion", not of(ev, "intrusion"))
b = fresh([ROAD])
ev, _ = run(b, [(t, [person(300, 350 if i % 2 == 0 else 370)]) for i, t in enumerate(ticks(0, 1.5))])
check("a wobble along the edge counts only time INSIDE (not yet 1 s)", not of(ev, "intrusion"))
check("...and a wobble is not a zone exit", not of(ev, "zone_exit"))
ev, _ = run(b, [(t, [person(300, 350 if i % 2 == 0 else 370)]) for i, t in enumerate(ticks(1.6, 3.0))])
check("...but enough inside time adds up to one intrusion", len(of(ev, "intrusion")) == 1)
b = fresh([dict(ROAD, restricted=False)])
ev, _ = run(b, [(t, [person(300, 300)]) for t in ticks(0, 3)])
check("an unrestricted zone never raises intrusion", not of(ev, "intrusion"))

print("\nleaving")
b = fresh([ROAD])
ev, _ = run(b, [(t, [person(300, 300)]) for t in ticks(0, 1.0)] +
            [(t, [person(900, 600)]) for t in ticks(1.1, 4.0)])
exits = of(ev, "zone_exit")
check("walking out and staying out = one zone_exit", len(exits) == 1, len(exits))
check("...after the grace period, carrying the dwell",
      exits and abs(exits[0][1]["dwell_s"] - 1.0) < 0.01 and exits[0][1]["intrusion"], exits)
b = fresh([ROAD])
ev, _ = run(b, [(t, [person(300, 300)]) for t in ticks(0, 1.0)] + [(t, []) for t in ticks(1.1, 4.0)])
check("disappearing from view also closes the zone", len(of(ev, "zone_exit")) == 1)
s = b.summary()["zones"]["road"]
check("summary: entries, exits, dwell", s["entries"] == 1 and s["exits"] == 1 and s["mean_dwell_s"] == 1.0, s)
ev, _ = run(b, [(t, [person(300, 300)]) for t in ticks(4.1, 6.0)])
check("the same track coming back is NOT a second intrusion", not of(ev, "intrusion"))
b = fresh([ROAD], exit_events=False)
ev, _ = run(b, [(t, [person(300, 300)]) for t in ticks(0, 1.0)] + [(t, []) for t in ticks(1.1, 4.0)])
check("exit_events: false keeps dwell stats but emits nothing",
      not of(ev, "zone_exit") and b.summary()["zones"]["road"]["exits"] == 1)

print("\nloitering")
b = fresh([{"name": "atm", "loitering_s": 5}])
ev, _ = run(b, [(t, [person(300, 300)]) for t in ticks(0, 4.8)])
check("no loitering before loitering_s", not of(ev, "loitering"))
ev, f = run(b, [(t, [person(300, 300)]) for t in ticks(4.9, 8.0)])
check("loitering fires once at loitering_s", len(of(ev, "loitering")) == 1, len(of(ev, "loitering")))
check("loitering in an unrestricted zone is not an intrusion", not of(ev, "intrusion"))
check("the label shows how long", f.overlay["flags"][0][1] == "LOITERING atm 8s", f.overlay["flags"])
b = fresh([{"name": "atm", "loitering_s": 5}])
ev, _ = run(b, [(t, [person(300, 300)]) for t in ticks(0, 3)] + [(t, []) for t in ticks(3.1, 4.0)] +
            [(t, [person(300, 300)]) for t in ticks(4.1, 5.5)])
check("a short gap (under exit_grace_s) does not restart the clock", len(of(ev, "loitering")) == 1)

print("\ncrowd")
SQUARE = [{"name": "square", "crowd_people": 3}]
three = [person(100, 300, 1), person(400, 300, 2), person(700, 300, 3)]
b = fresh(SQUARE, crowd_hold_s=2)
ev, _ = run(b, [(t, three) for t in ticks(0, 1.5)])
check("no crowd before it has held", not of(ev, "crowd"))
ev, f = run(b, [(t, three) for t in ticks(1.6, 2.5)])
check("crowded once it holds", [e[1]["state"] for e in of(ev, "crowd")] == ["crowded"], of(ev, "crowd"))
check("frame findings say so", f.frame["crowded"]["square"] and f.frame["occupancy"]["square"] == 3)
ev, _ = run(b, [(2.6, three[:2])] + [(t, three) for t in ticks(2.7, 5.0)])
check("one frame with a missed person does not clear it", not of(ev, "crowd"))
ev, _ = run(b, [(t, three[:2]) for t in ticks(5.1, 6.0)])
check("fewer people must also hold before clear", not of(ev, "crowd"))
ev, _ = run(b, [(t, three[:2]) for t in ticks(6.1, 7.5)])
check("then clear, once", [e[1]["state"] for e in of(ev, "crowd")] == ["clear"], of(ev, "crowd"))
s = b.summary()["zones"]["square"]
check("summary: peak, events, crowded seconds",
      s["peak_people"] == 3 and s["crowd_events"] == 1 and 4.0 <= s["crowded_s"] <= 5.5, s)
b = fresh(SQUARE, crowd_hold_s=2)
ev, _ = run(b, [(t, three if i % 6 else three[:2]) for i, t in enumerate(ticks(0, 4))])
check("a crowd with a detector miss every 6th frame still counts (an unbroken hold never would)",
      [e[1]["state"] for e in of(ev, "crowd")] == ["crowded"], of(ev, "crowd"))
b = fresh(SQUARE, crowd_hold_s=2)
ev, _ = run(b, [(t, three if i % 2 else three[:2]) for i, t in enumerate(ticks(0, 4))])
check("over the threshold only half the time is not a crowd", not of(ev, "crowd"))

print("\nriders")
b = fresh([ROAD])
bike = vehicle(250, 200, 350, 320)
ev, f = run(b, [(t, [bike, person(300, 310, h=150)]) for t in ticks(0, 3)])
check("a motorcycle rider on the road is not an intrusion", not of(ev, "intrusion"))
check("...and not counted in the zone", f.frame["occupancy"]["road"] == 0)
check("...and counted as ignored", b.summary()["rider_boxes_ignored"] == 31)
b = fresh([ROAD])
ev, _ = run(b, [(t, [bike, person(500, 300, tid=2)]) for t in ticks(0, 3)])
check("a pedestrian beside the bike still is", len(of(ev, "intrusion")) == 1)
b = fresh([ROAD])
ev, _ = run(b, [(t, [vehicle(0, 0, 640, 360, cls="bus"), person(300, 300, h=100)]) for t in ticks(0, 3)])
check("a passenger seen through a bus window is not", not of(ev, "intrusion"))
b = fresh([ROAD])
ev, _ = run(b, [(t, [vehicle(200, 250, 500, 360, cls="car"), person(300, 340)]) for t in ticks(0, 3)])
check("a pedestrian BEHIND a parked car (legs hidden, 45% covered) still counts",
      len(of(ev, "intrusion")) == 1 and b.rider_boxes == 0)
b = fresh([ROAD])
ev, _ = run(b, [(t, [vehicle(250, 200, 350, 320, cls="car"), person(300, 310, h=150)]) for t in ticks(0, 3)])
check("the two-wheeler rule does not apply to cars (73% covered, not 90%)", len(of(ev, "intrusion")) == 1)
b = fresh([ROAD], ignore_riders=False)
ev, _ = run(b, [(t, [bike, person(300, 310, h=150)]) for t in ticks(0, 3)])
check("ignore_riders: false counts riders", len(of(ev, "intrusion")) == 1)

print("\nexclude masks")
POLE = [[0.2, 0.3], [0.3, 0.3], [0.3, 0.45], [0.2, 0.45]]   # x 256-384, y 216-324
b = fresh([ROAD], exclude=[POLE])
ev, f = run(b, [(t, [person(300, 300)]) for t in ticks(0, 3)])
check("a 'person' standing in an exclude mask never fires", not of(ev, "intrusion"))
check("...is not counted in the zone, and is counted as masked",
      f.frame["occupancy"]["road"] == 0 and b.summary()["masked_boxes_ignored"] == 31)
b = fresh([ROAD], exclude=[{"polygon": POLE}])
ev, _ = run(b, [(t, [person(500, 300, tid=2)]) for t in ticks(0, 3)])
check("a real person outside the mask still fires (mask given as a mapping)", len(of(ev, "intrusion")) == 1)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    bad = BehaviourAnalyzer({"analyses": {"behaviour": {"exclude": [[[0, 0], [1, 1]]]}}})
    bad.setup(Source(), {})
check("a mask with under 3 corners is skipped with a message", not bad.masks and "exclude #1" in buf.getvalue())

print("\nrunning")


def moving(speed_hps, t0=0.0, t1=3.0, h=200, y=600, tid=1, x0=100):
    return [(t, [person(int(x0 + speed_hps * h * (t - t0)), y, tid=tid, h=h)]) for t in ticks(t0, t1)]


b = fresh()
ev, _ = run(b, moving(0.8))
check("walking (0.8 h/s) is not running", not of(ev, "running"))
b = fresh()
ev, f = run(b, moving(2.5, t1=2.0))
check("2.5 h/s held for over 1 s = one running event", len(of(ev, "running")) == 1, len(of(ev, "running")))
check("...with the measured speed", of(ev, "running") and abs(of(ev, "running")[0][1]["speed_hps"] - 2.5) < 0.1,
      of(ev, "running"))
check("the box is labelled", f.overlay["flags"] and f.overlay["flags"][0][1].startswith("RUNNING"))
b = fresh()
ev, _ = run(b, moving(2.5, t1=1.4))
check("a burst shorter than window + min_s is not", not of(ev, "running"))
b = fresh()
ev, _ = run(b, moving(2.5, h=60, y=300, t1=2.0) + [])
check("speed is in body heights: a small far person at 2.5 h/s still counts", len(of(ev, "running")) == 1)
b = fresh()
ev, _ = run(b, [(t, [person(100, 600)]) for t in ticks(0, 1)] + [(t, [person(1100, 600)]) for t in ticks(1.1, 3)])
check("a tracker swap (a jump across the frame) is not running", not of(ev, "running"))
b = fresh()
ev, _ = run(b, moving(2.5, y=720, t1=2.0))
check("a box cut by the frame edge is skipped", not of(ev, "running"))
b = fresh()
ev, _ = run(b, moving(2.5, h=30, y=300, t1=2.0))
check("a box under min_height_px is skipped", not of(ev, "running"))
b = fresh(running={"enabled": False})
ev, _ = run(b, moving(2.5, t1=2.0))
check("running: enabled false", not of(ev, "running") and b.summary()["running"] is None)
b = fresh()
run(b, moving(0.8))
p = b.summary()["speed_hps"]
check("speed percentiles are kept for tuning", p.get("samples", 0) > 0 and 0.7 <= p["p50"] <= 0.9, p)

print("\nlifecycle")
b = fresh([ROAD])
ev, _ = run(b, [(t, [person(300, 300)]) for t in ticks(0, 3)] + [(t, [person(300, 300)]) for t in ticks(0, 1.5)])
check("a clock going backwards (clip restarted) starts fresh", len(of(ev, "intrusion")) == 2)
b.forget(1)
check("forget frees the track, keeps its dwell in the stats",
      1 not in b._tracks and b.summary()["zones"]["road"]["exits"] == 1, b.summary()["zones"]["road"])
b = fresh([ROAD])
run(b, [(t, [person(300, 300, tid=5)]) for t in ticks(0, 1)])
run(b, [(40.0, [])])
check("a track unseen for long is forgotten even without forget()", 5 not in b._tracks)

print("\ndrawing")
b = fresh([ROAD, {"name": "all"}])
_, f = run(b, [(t, [person(300, 300)]) for t in ticks(0, 1.5)])
ctx = Ctx()
b.draw(ctx, f)
check("zone outline is drawn", (ctx.annotated[0:2, 100:600] != 100).any())
x1, y1, x2, y2 = person(300, 300).bbox
check("the intruder is boxed in red, 4 px OUTSIDE its box (the pipeline draws on the box itself later)",
      tuple(ctx.annotated[(y1 + y2) // 2, x1 - 4]) == (0, 0, 255)
      and tuple(ctx.annotated[(y1 + y2) // 2, x1]) == (100, 100, 100),
      (ctx.annotated[(y1 + y2) // 2, x1 - 4], ctx.annotated[(y1 + y2) // 2, x1]))
check("the flag label sits above the box, clear of the pipeline's own label (y1 - 8)",
      (ctx.annotated[y1 - 40:y1 - 26, x1:x1 + 40] != 100).any()
      and (ctx.annotated[y1 - 20:y1 - 6, x1 + 60:x1 + 80] == 100).all())
check("far from any zone line or flag the video is untouched", (ctx.annotated[500:700, 700:1200] == 100).all())
b = fresh([ROAD], draw=False)
_, f = run(b, [(0.0, [person(300, 300)])])
ctx = Ctx()
b.draw(ctx, f)
check("draw: false leaves the video alone", (ctx.annotated == 100).all())

print("\nincidents")
pol = IncidentPolicy({"incidents": {"enabled": True, "base_url": "http://h:8000"}}, None,
                     {"cam": {"name": "Cam", "lat": 1.0, "lon": 2.0}})
DET = {"zone": "road", "group": "person", "cls_name": "person", "dwell_s": 1.0, "bbox": [1, 2, 3, 4]}
out = pol.handle({"kind": "intrusion", "track_id": 7, "ts": 12.5, "detail": DET},
                 {"camera": "cam", "object_id": 42})
check("intrusion raises immediately, not at retirement", len(out) == 1)
body = out[0]["payload"]
check("id is camera-object-zone-kind", body["incident_id"] == "cam-42-road-intrusion", body["incident_id"])
check("payload: zone, subject, sighting, state ongoing",
      body["zone"] == "road" and body["subject"] == {"group": "person", "cls": "person"}
      and body["sighting"] == {"object_id": 42} and body["state"] == "ongoing", body)
check("no vehicle block and no name on a behaviour incident", "vehicle" not in body and "person" not in body)
check("image_url points at the crop", body["image_url"] == "http://h:8000/crops/42.jpg")
check("detail keeps the evidence, not the projected fields",
      body["detail"].get("dwell_s") == 1.0 and "group" not in body["detail"], body["detail"])
check("loitering is on by default", len(pol.handle({"kind": "loitering", "track_id": 7, "ts": 30.0,
                                                    "detail": DET}, {"camera": "cam", "object_id": 42})) == 1)
run_ev = {"kind": "running", "track_id": 7, "ts": 5.0, "detail": dict(DET, zone=None, speed_hps=2.4)}
check("running is OFF by default (noisy)", pol.handle(run_ev, {"camera": "cam", "object_id": 42}) == [])
pol_run = IncidentPolicy({"incidents": {"kinds": {"running": True}}}, None, {})
out = pol_run.handle(run_ev, {"camera": "cam", "object_id": 42})
check("...and fires when enabled, id without a zone", out and out[0]["id"] == "cam-42-running",
      out and out[0]["id"])
out = pol.handle({"kind": "crowd", "ts": 35.0, "detail": {"zone": "square", "state": "crowded", "people": 6,
                                                          "crowd_people": 5, "since": 30.0, "held_s": 5.0}},
                 {"camera": "cam", "object_id": None})
body = out[0]["payload"] if out else {}
check("crowd raises with its state and zone, no subject",
      body.get("state") == "crowded" and body.get("zone") == "square" and "subject" not in body, body)
check("crowd id is zone + onset time", body.get("incident_id") == "cam-square-t30-crowd", body.get("incident_id"))
check("zone_exit can never be an incident", "zone_exit" in NEVER and
      pol.handle({"kind": "zone_exit", "track_id": 7, "ts": 1.0, "detail": {}}, {"camera": "cam"}) == [])

print("\nwiring")
cfg = load_for_camera(None)
check("behaviour is OFF by default", "behaviour" not in enabled_names(cfg))
check("background search embeddings stay OFF in config.yaml",
      cfg["server"]["search"]["embed_in_background"] is False)
demo = load_for_camera("behaviour_demo")
check("behaviour_demo camera switches it on with 3 zones",
      "behaviour" in enabled_names(demo) and len(demo["analyses"]["behaviour"]["zones"]) == 3)
with contextlib.redirect_stdout(io.StringIO()):
    built = build(demo, Source())
beh = [a for a in built if a.name == "behaviour"]
check("...and it is built like any analysis", beh and [z.name for z in beh[0].zones] == ["road", "shopfront", "pavement"])

print(f"\n{'=' * 60}\n{_passed} passed, {len(_failed)} failed")
for name in _failed:
    print(f"  FAILED: {name}")
sys.exit(1 if _failed else 0)
