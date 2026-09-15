"""Human behaviour - named zones, and what people do in them.

Tier 1 of the behaviour work (docs/human-behaviour.md): rules over the tracks
the pipeline already has. No model and no new dependency - a few float
operations per person per frame.

    zones       named polygons per camera: occupancy, entries, dwell time
    intrusion   a person inside a `restricted` zone for `intrusion_s`
    loitering   a person inside a zone for that zone's `loitering_s`
    crowd       `crowd_people` or more inside a zone in `crowd_fraction` of
                the frames of the last `crowd_hold_s`; reported on both edges
                (crowded, clear), like congestion
    running     faster than `min_speed_hps` body heights per second for
                `min_s`, anywhere in the frame

Positions are GROUND POINTS (bottom-centre of the box): where a person stands
is what "inside the zone" means. A tall box whose middle is over the road
while the feet are on the pavement is on the pavement.

WHY EVERY RULE HAS A HOLD TIME

A box's ground point wobbles by a few pixels every frame, so a person walking
along a zone edge is in, out, in, out. Two separate clocks absorb that:
intrusion counts only the time actually INSIDE, so a wobble in and out is not
an intrusion; and a person has to be OUT for `exit_grace_s` before they have
left, so a wobble does not end a dwell and restart the loitering clock.

WHY SPEED IS IN BODY HEIGHTS

Pixels per second means nothing without calibrating the floor: a walker near
the camera crosses more pixels than a sprinter far away. Dividing by the
person's own box height (a person is ~1.65 m) makes the number roughly
independent of depth with no calibration at all. It is still a 2-D
projection: running straight at the camera barely moves the ground point, so
running is UNDER-reported, never invented.

WHY RIDERS ARE IGNORED

On Indian roads most fast "people" are on two-wheelers, and the detector
boxes the rider as a person. Two tests, because two kinds of vehicle hide a
person differently:

  two-wheeler  feet inside its box and 30% of the person covered: a rider
               (the bike covers roughly the rider's lower half)
  enclosed     90% of the person inside a car/bus/truck box: a passenger
               seen through a window

The enclosed test is strict on purpose. The first version used the 30% rule
for every vehicle, and on the demo clip all 23 "riders" it removed were
pedestrians walking behind a parked taxi - legs hidden, head and shoulders
above the roof. Only a person almost entirely inside the vehicle's box is in
it. Riders are not running, not jaywalking, not part of a crowd.

EVERY RULE FIRES ONCE PER TRACK (per zone). A person who leaves the frame and
comes back is usually a new track and can fire again; that is a tracker
limit, and the incident id still dedupes a retry of the same object.
"""

import math
from collections import deque

import cv2
import numpy as np

from ..runtime.plugin import Findings
from .base import block, register
from .geometry import ensure_simple, frame_ring, point_in_polygon, scale_points

COLOR_ZONE = (255, 200, 0)          # BGR
COLOR_RESTRICTED = (0, 0, 255)
COLOR_CROWD = (0, 140, 255)
COLOR_LOITER = (0, 200, 255)
COLOR_RUN = (255, 0, 255)
COLOR_MASK = (128, 128, 128)

SPEED_BIN = 0.1                     # body heights per second per bin
SPEED_BINS = 61                     # 0.0 - 6.0 h/s, last bin is "faster"
STALE_S = 30.0                      # forget a track not seen this long
EPS = 1e-6                          # 0.1 added ten times is 0.9999999
# Same lists as config.py's defaults, so a config built by hand without them
# (a test, a tool) does not silently stop ignoring riders.
RIDERS_IN = ("bicycle", "motorcycle")
PASSENGERS_IN = ("car", "bus", "truck", "train")


class Zone:
    """One named polygon and the rules attached to it."""

    def __init__(self, name: str, ring: list, spec: dict, whole_frame=False):
        self.name = name
        self.ring = ring
        self.whole_frame = whole_frame
        self.restricted = bool(spec.get("restricted", False))
        self.loitering_s = max(0.0, float(spec.get("loitering_s") or 0))
        self.crowd_people = max(0, int(spec.get("crowd_people") or 0))
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        self.bounds = (min(xs), min(ys), max(xs), max(ys))
        self.label_at = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    def contains(self, x: float, y: float) -> bool:
        if self.whole_frame:
            return True
        x1, y1, x2, y2 = self.bounds
        if x < x1 or x > x2 or y < y1 or y > y2:
            return False
        return point_in_polygon(x, y, self.ring)


def _covered(box, others) -> float:
    """Largest fraction of `box` covered by one box in `others`, counted only
    for boxes that also contain its ground point (the rider test)."""
    x1, y1, x2, y2 = box
    gx, gy = (x1 + x2) / 2, y2
    area = max(1, (x2 - x1) * (y2 - y1))
    best = 0.0
    for ox1, oy1, ox2, oy2 in others:
        if not (ox1 <= gx <= ox2 and oy1 <= gy <= oy2):
            continue
        iw = min(x2, ox2) - max(x1, ox1)
        ih = min(y2, oy2) - max(y1, oy1)
        if iw > 0 and ih > 0:
            best = max(best, iw * ih / area)
    return best


class BehaviourAnalyzer:
    name = "behaviour"
    # compute() reads the view and writes only this plugin's own track and zone
    # state; draw() only draws. Nothing shared, so it may run on a worker.
    concurrent = True

    def __init__(self, cfg: dict):
        c = block(cfg, self.name)
        self.groups = {str(g) for g in (c.get("groups") or ["person"])}
        self.units = str(c.get("units", "auto"))
        self.zone_specs = list(c.get("zones") or [])
        self.exclude_specs = list(c.get("exclude") or [])
        self.intrusion_s = max(0.0, float(c.get("intrusion_s", 1.0)))
        self.exit_grace_s = max(0.0, float(c.get("exit_grace_s", 2.0)))
        self.crowd_hold_s = max(0.0, float(c.get("crowd_hold_s", 5.0)))
        self.crowd_fraction = min(1.0, max(0.5, float(c.get("crowd_fraction", 0.8))))
        self.ignore_riders = bool(c.get("ignore_riders", True))
        self.rider_overlap = float(c.get("rider_overlap", 0.3))
        self.riders_in = {str(n).lower()
                          for n in (c.get("riders_in") or RIDERS_IN)}
        self.passenger_overlap = float(c.get("passenger_overlap", 0.9))
        self.passengers_in = {str(n).lower()
                              for n in (c.get("passengers_in") or PASSENGERS_IN)}
        r = c.get("running") or {}
        self.run_on = bool(r.get("enabled", True))
        self.run_speed = float(r.get("min_speed_hps", 1.6))
        self.run_min_s = max(0.0, float(r.get("min_s", 1.0)))
        self.run_window_s = max(0.2, float(r.get("window_s", 1.0)))
        self.run_min_h = int(r.get("min_height_px", 40))
        self.max_jump = float(r.get("max_jump_hps", 6.0))
        self.edge_px = int(r.get("edge_px", 2))
        self.exit_events = bool(c.get("exit_events", True))
        self.draw_overlay = bool(c.get("draw", True))

        self.width = self.height = 0
        self.zones: list = []
        self.masks: list = []
        self._tracks: dict = {}
        self._crowd: dict = {}
        self._stats: dict = {}
        self._last_ts = None
        self._speeds = np.zeros(SPEED_BINS, np.int64)
        self.frames = 0
        self.running_count = 0
        self.rider_boxes = 0
        self.masked_boxes = 0

    def setup(self, source, cfg: dict):
        self.width = max(1, int(getattr(source, "width", 0) or 1))
        self.height = max(1, int(getattr(source, "height", 0) or 1))
        self.zones = []
        for i, spec in enumerate(self.zone_specs):
            if not isinstance(spec, dict):
                print(f"[behaviour] zone #{i + 1} is not a mapping; skipping")
                continue
            name = str(spec.get("name") or f"zone{i + 1}")
            if name in self._stats:
                print(f"[behaviour] duplicate zone name {name!r}; skipping the second")
                continue
            points = spec.get("polygon")
            if points:
                ring = scale_points(points, self.width, self.height,
                                    spec.get("units", self.units))
                if len(ring) < 3:
                    print(f"[behaviour] zone {name!r} needs 3+ corners; skipping")
                    continue
                ring, _ = ensure_simple(ring, name)
                zone = Zone(name, ring, spec)
            else:
                zone = Zone(name, frame_ring(self.width, self.height), spec,
                            whole_frame=True)
            self.zones.append(zone)
            self._stats[name] = {"entries": 0, "exits": 0, "dwell_total": 0.0,
                                 "dwell_max": 0.0, "people_now": 0,
                                 "peak_people": 0, "intrusions": 0,
                                 "loitering": 0, "crowd_events": 0,
                                 "crowded_s": 0.0}
            self._crowd[name] = {"crowded": False, "samples": deque()}
        # Exclude masks: spots where the detector keeps seeing a "person" that
        # is not one (a sign pole, a bollard, a poster). Measured on the demo
        # clip, those reach 0.6 confidence - as high as real people - so no
        # confidence floor can remove them, and a mask is the honest fix.
        self.masks = []
        for i, points in enumerate(self.exclude_specs):
            if isinstance(points, dict):
                points = points.get("polygon")
            ring = (scale_points(points, self.width, self.height, self.units)
                    if points else [])
            if len(ring) < 3:
                print(f"[behaviour] exclude #{i + 1} needs 3+ corners; skipping")
                continue
            ring, _ = ensure_simple(ring, f"exclude #{i + 1}")
            self.masks.append(Zone(f"exclude{i + 1}", ring, {}))
        if not self.zones and not self.run_on:
            print("[behaviour] no zones and running is off: nothing to watch")

    # --- per frame ---------------------------------------------------------
    def compute(self, view) -> Findings:
        found = Findings(analyzer=self.name)
        now = float(view.timestamp)
        if self._last_ts is not None and now < self._last_ts - 1.0:
            # The clock went backwards: a file source started over. Dwell
            # clocks from the previous run would be nonsense on this one.
            self._tracks.clear()
            for state in self._crowd.values():
                state["crowded"] = False
                state["samples"].clear()
        dt = 0.0 if self._last_ts is None else min(1.0, max(0.0, now - self._last_ts))
        self._last_ts = now
        self.frames += 1

        vehicles = self.vehicles(view.boxes)
        occupancy = {z.name: 0 for z in self.zones}
        flags = []
        for box in view.boxes:
            if box.group not in self.groups:
                continue
            x, y = box.ground_point
            if self.masks and any(m.contains(x, y) for m in self.masks):
                self.masked_boxes += 1
                continue
            if box.group == "person" and self.on_vehicle(box.bbox, vehicles):
                self.rider_boxes += 1
                continue
            inside = [z for z in self.zones if z.contains(x, y)]
            for z in inside:
                occupancy[z.name] += 1
            if box.track_id is None:
                continue
            track = self._tracks.get(box.track_id)
            if track is None:
                track = self._tracks[box.track_id] = {
                    "zones": {}, "fired": set(), "hist": deque(),
                    "run_since": None, "last": None,
                    "group": box.group, "cls": box.cls_name}
            step = 0.0 if track["last"] is None else min(1.0, max(0.0, now - track["last"]))
            track["last"] = now
            labels = self._zones(found, box, track, inside, now, step)
            if self.run_on:
                label = self._running(found, box, track, inside, now)
                if label:
                    labels.append(label)
            # One set() call: Findings.set replaces a field, so two calls
            # each passing extra= would keep only the second.
            extra = {}
            if inside:
                extra["behaviour_zones"] = [z.name for z in inside]
            if labels:
                extra["behaviour"] = [t for t, _ in labels]
                flags.append((box.bbox, " | ".join(t for t, _ in labels),
                              labels[0][1]))
            if extra:
                found.set(box.track_id, extra=extra)

        self._sweep(found, now)
        crowded = self._crowds(found, occupancy, now, dt)
        found.frame = {"occupancy": dict(occupancy), "crowded": crowded}
        found.overlay = {"occupancy": occupancy, "crowded": crowded,
                         "flags": flags}
        return found

    def vehicles(self, boxes):
        """(two-wheeler boxes, enclosed-vehicle boxes) in this frame."""
        if not self.ignore_riders:
            return [], []
        names = [(str(b.cls_name).lower(), b.bbox) for b in boxes]
        return ([bbox for n, bbox in names if n in self.riders_in],
                [bbox for n, bbox in names if n in self.passengers_in])

    def on_vehicle(self, bbox, vehicles) -> bool:
        """A rider on a two-wheeler, or a passenger inside a vehicle."""
        riders, cabins = vehicles
        return bool((riders and _covered(bbox, riders) >= self.rider_overlap)
                    or (cabins and _covered(bbox, cabins) >= self.passenger_overlap))

    def _zones(self, found, box, track, inside, now, step) -> list:
        """Entry, dwell, intrusion and loitering for one tracked person."""
        labels = []
        names_in = set()
        for z in inside:
            names_in.add(z.name)
            zs = track["zones"].get(z.name)
            if zs is None:
                zs = track["zones"][z.name] = {"entered": now, "last_in": now,
                                               "inside_s": 0.0}
                self._stats[z.name]["entries"] += 1
            else:
                zs["inside_s"] += step
            zs["last_in"] = now
            dwell = now - zs["entered"]
            if (z.restricted and ("intrusion", z.name) not in track["fired"]
                    and zs["inside_s"] >= self.intrusion_s - EPS):
                track["fired"].add(("intrusion", z.name))
                self._stats[z.name]["intrusions"] += 1
                found.event("intrusion",
                            self._detail(box, z.name, dwell_s=zs["inside_s"]),
                            box.track_id)
            if (z.loitering_s and ("loitering", z.name) not in track["fired"]
                    and dwell >= z.loitering_s - EPS):
                track["fired"].add(("loitering", z.name))
                self._stats[z.name]["loitering"] += 1
                found.event("loitering",
                            self._detail(box, z.name, dwell_s=dwell,
                                         loitering_s=z.loitering_s),
                            box.track_id)
            if ("intrusion", z.name) in track["fired"]:
                labels.append((f"INTRUSION {z.name}", COLOR_RESTRICTED))
            if ("loitering", z.name) in track["fired"]:
                labels.append((f"LOITERING {z.name} {dwell:.0f}s", COLOR_LOITER))
        for name, zs in list(track["zones"].items()):
            if name not in names_in and now - zs["last_in"] > self.exit_grace_s:
                self._exit(found, box.track_id, track, name)
        return labels

    def _running(self, found, box, track, inside, now):
        """Speed over the last `window_s`, in body heights per second."""
        x, y = box.ground_point
        h = box.height
        _x1, y1, _x2, y2 = box.bbox
        if (h < self.run_min_h or y1 <= self.edge_px
                or y2 >= self.height - self.edge_px):
            # A tiny box is all noise, and a box cut by the frame edge has the
            # wrong height and the wrong feet. Skip the sample, keep history.
            return None
        hist = track["hist"]
        if hist:
            pt, px, py, ph = hist[-1]
            gap = now - pt
            if gap > self.run_window_s or (
                    gap > 0 and math.hypot(x - px, y - py) / max(h, ph) / gap
                    > self.max_jump):
                # A long gap, or a jump nobody can run: the tracker swapped
                # people. Start over rather than measure the swap.
                hist.clear()
                track["run_since"] = None
        hist.append((now, x, y, h))
        while now - hist[0][0] > self.run_window_s + EPS:
            hist.popleft()
        t0, x0, y0, _h0 = hist[0]
        span = now - t0
        if span < 0.6 * self.run_window_s:
            return None
        heights = sorted(s[3] for s in hist)
        speed = math.hypot(x - x0, y - y0) / max(1, heights[len(heights) // 2]) / span
        self._speeds[min(SPEED_BINS - 1, int(speed / SPEED_BIN))] += 1
        if speed < self.run_speed:
            track["run_since"] = None
            return None
        if track["run_since"] is None:
            track["run_since"] = now
        if now - track["run_since"] < self.run_min_s - EPS:
            return None
        if ("running", None) not in track["fired"]:
            track["fired"].add(("running", None))
            self.running_count += 1
            found.event("running",
                        self._detail(box, inside[0].name if inside else None,
                                     speed_hps=speed,
                                     zones=[z.name for z in inside]),
                        box.track_id)
        return (f"RUNNING {speed:.1f} h/s", COLOR_RUN)

    def _sweep(self, found, now):
        """Close the zones of people who are gone; forget long-gone tracks."""
        for tid, track in list(self._tracks.items()):
            idle = now - (track["last"] if track["last"] is not None else now)
            if idle > self.exit_grace_s:
                for name in list(track["zones"]):
                    self._exit(found, tid, track, name)
            if idle > STALE_S:
                del self._tracks[tid]

    def _exit(self, found, tid, track, name, emit=True):
        zs = track["zones"].pop(name)
        dwell = max(0.0, zs["last_in"] - zs["entered"])
        st = self._stats.get(name)
        if st is not None:
            st["exits"] += 1
            st["dwell_total"] += dwell
            st["dwell_max"] = max(st["dwell_max"], dwell)
        if emit and self.exit_events:
            found.event("zone_exit",
                        {"zone": name, "group": track["group"],
                         "dwell_s": round(dwell, 2),
                         "intrusion": ("intrusion", name) in track["fired"],
                         "loitering": ("loitering", name) in track["fired"]},
                        tid)

    def _crowds(self, found, occupancy, now, dt) -> dict:
        """Per-zone crowd state. Both edges are events, each after a hold.

        The hold is a FRACTION of a window, not an unbroken run. The first
        version needed the count over the threshold on every frame for
        crowd_hold_s, and on the demo clip it never fired: in 4-second spans
        averaging 5.0 people the longest unbroken run at 5+ was 2.0 s,
        because the detector drops one person for a frame every second or
        so. A crowd is still a crowd while one of them is behind an umbrella.
        """
        crowded = {}
        for z in self.zones:
            st = self._stats[z.name]
            n = occupancy[z.name]
            st["people_now"] = n
            st["peak_people"] = max(st["peak_people"], n)
            state = self._crowd[z.name]
            if z.crowd_people:
                if state["crowded"]:
                    st["crowded_s"] += dt
                samples = state["samples"]
                samples.append((now, n >= z.crowd_people))
                while now - samples[0][0] > self.crowd_hold_s + EPS:
                    samples.popleft()
                start = samples[0][0]
                if now - start >= self.crowd_hold_s - EPS:
                    over = sum(1 for _, o in samples if o) / len(samples)
                    flip = ((not state["crowded"] and over >= self.crowd_fraction)
                            or (state["crowded"] and 1.0 - over >= self.crowd_fraction))
                    if flip:
                        state["crowded"] = not state["crowded"]
                        if state["crowded"]:
                            st["crowd_events"] += 1
                        found.event("crowd",
                                    {"zone": z.name,
                                     "state": "crowded" if state["crowded"] else "clear",
                                     "people": n, "crowd_people": z.crowd_people,
                                     "over_fraction": round(over, 2),
                                     "since": round(start, 3),
                                     "held_s": round(now - start, 2)})
            crowded[z.name] = state["crowded"]
        return crowded

    @staticmethod
    def _detail(box, zone, **fields) -> dict:
        detail = {"zone": zone, "group": box.group, "cls_name": box.cls_name,
                  "bbox": [int(v) for v in box.bbox]}
        for key, value in fields.items():
            detail[key] = round(value, 2) if isinstance(value, float) else value
        return detail

    def apply(self, ctx, findings: Findings):
        """Nothing shared to write: per-track tags and events were queued by
        compute and the scheduler has already applied and replayed them."""

    # --- drawing -----------------------------------------------------------
    def draw(self, ctx, findings: Findings):
        if not self.draw_overlay or not findings.overlay:
            return
        frame = ctx.annotated
        occupancy = findings.overlay.get("occupancy", {})
        crowded = findings.overlay.get("crowded", {})
        for m in self.masks:
            cv2.polylines(frame, [np.array(m.ring, np.int32)], True, COLOR_MASK, 1)
        corner = 0
        for z in self.zones:
            colour = (COLOR_CROWD if crowded.get(z.name)
                      else COLOR_RESTRICTED if z.restricted else COLOR_ZONE)
            text = f"{z.name}: {occupancy.get(z.name, 0)}"
            if crowded.get(z.name):
                text += " CROWD"
            if z.whole_frame:
                _label(frame, text, (10, 140 + 26 * corner), colour)
                corner += 1
                continue
            cv2.polylines(frame, [np.array(z.ring, np.int32)], True, colour, 2)
            _label(frame, text, z.label_at, colour)
        # The pipeline draws every object's own box and label AFTER the
        # analyses (pipeline.py), on the same box and at y1 - 8. So the flag
        # box sits 4 px outside it and the flag label above that label, or
        # both are painted over - which is what the first real run showed.
        for bbox, text, colour in findings.overlay.get("flags", []):
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), colour, 3)
            _label(frame, text, (x1, y1 - 30), colour)

    # --- lifecycle ---------------------------------------------------------
    def forget(self, tid):
        """The track retired. Its dwell still counts; it just cannot emit."""
        track = self._tracks.pop(tid, None)
        if track is None:
            return
        for name in list(track["zones"]):
            self._exit(None, tid, track, name, emit=False)

    def summary(self) -> dict:
        zones = {}
        for z in self.zones:
            st = self._stats[z.name]
            zones[z.name] = {
                "restricted": z.restricted,
                "loitering_s": z.loitering_s or None,
                "crowd_people": z.crowd_people or None,
                "entries": st["entries"], "exits": st["exits"],
                "peak_people": st["peak_people"],
                "mean_dwell_s": (round(st["dwell_total"] / st["exits"], 1)
                                 if st["exits"] else None),
                "max_dwell_s": round(st["dwell_max"], 1),
                "intrusions": st["intrusions"], "loitering": st["loitering"],
                "crowd_events": st["crowd_events"],
                "crowded_s": round(st["crowded_s"], 1)}
        return {"frames": self.frames, "zones": zones,
                "running": self.running_count if self.run_on else None,
                "rider_boxes_ignored": self.rider_boxes,
                "masked_boxes_ignored": self.masked_boxes,
                "speed_hps": self.speed_percentiles()}

    def speed_percentiles(self) -> dict:
        """p50/p90/p99 of every speed sample measured - what a threshold is
        tuned from. Bin upper edges, so accurate to 0.1 h/s."""
        total = int(self._speeds.sum())
        if not total:
            return {}
        cumulative = np.cumsum(self._speeds)
        out = {"samples": total}
        for p in (50, 90, 99):
            i = int(np.searchsorted(cumulative, total * p / 100.0))
            out[f"p{p}"] = round((i + 1) * SPEED_BIN, 1)
        return out


def _label(frame, text, origin, colour, scale=0.55):
    """Text on a black backing box, kept inside the frame."""
    x, y = int(origin[0]), int(origin[1])
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    h, w = frame.shape[:2]
    x = max(2, min(x, w - tw - 4))
    y = max(th + 4, min(y, h - 4))
    cv2.rectangle(frame, (x - 3, y - th - 4), (x + tw + 3, y + base), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


register("behaviour", BehaviourAnalyzer)
