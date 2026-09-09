"""What lane is this object in, and is it facing the way that lane flows?

This is the wrong-side rule, and it is deliberately a pure function of numbers:
a LaneModel takes points and velocities and returns a verdict. It never touches
a frame, the tracker store, or storage, so it is testable without a video
(tools/test_lanes.py) and safe to call from a worker thread.

WHAT CHANGED, AND WHY

The previous rule was `direction == "up" and dy > 4`, on the bbox CENTROID, for
5 consecutive frames. Measured on samples/plate_test.mp4 that produced two
wrong-way alerts, both on vehicles travelling correctly, and would have missed
a real offender:

  * only the SIGN OF dy was tested, against one of four cardinal directions.
    A camera on a pole sees the carriageway running diagonally, so a vehicle
    correctly heading away moves up AND sideways, and near the vanishing point
    the sideways component is the larger one. Now: a per-lane flow VECTOR and
    the cosine between it and the track's velocity.

  * the 4 px gate was absolute. Measured: 96% of far-object frames and 68% of
    near-object frames moved less than 4 px, so the test was blind in the top
    half of the frame while still firing on jitter at the bottom. Now: the gate
    scales with the object's own bbox height, which is the cheapest available
    proxy for depth, and it is applied to displacement over a WINDOW rather
    than one frame.

  * the bbox centroid drifts UP for an approaching vehicle whose box is
    clipped by the bottom edge: the box stops growing downward while its top
    keeps rising. Both false positives above were on edge-clipped boxes. Now:
    the ground contact point (bottom-centre), and no verdict at all while the
    box touches a frame edge.

  * 5 CONSECUTIVE opposing frames, with the streak reset by any frame under
    the gate. A vehicle crawling the wrong way down a jammed road resets the
    counter constantly. Now: k opposing samples out of the last n.
"""

from collections import deque

from .geometry import (Line, direction_vector, distance_to_edge, ensure_simple,
                       point_in_polygon, scale_points)

# Verdicts for one object in one frame.
OK = "ok"
WRONG_WAY = "wrong_way"
WRONG_LANE = "wrong_lane"
WRONG_WAY_AND_LANE = "wrong_way+wrong_lane"

# Why no verdict was reached - kept so "nothing was flagged" can be explained.
NO_LANE = "no_lane"
NO_MOTION = "too_slow"
EDGE_CLIPPED = "edge_clipped"
NEAR_DIVIDER = "near_divider"
AMBIGUOUS = "ambiguous_heading"
WARMUP = "warmup"
UNVERIFIED = "unverified_geometry"
TRACK_JUMP = "track_discontinuity"

SEVERITY = {OK: 0, WRONG_LANE: 1, WRONG_WAY: 2, WRONG_WAY_AND_LANE: 3}


class Thresholds:
    """The tunables, in one place, all named in the units they are measured in.

    Defaults come from the two sample clips; every one of them is a
    `lanes_rules:` key in config.yaml so a new camera is a config change.
    """

    def __init__(self, cfg: dict | None = None):
        c = dict(cfg or {})
        # Velocity is estimated over this many samples of history. More =
        # steadier heading, slower to react. 8 analysed frames at 5 fps is
        # ~1.6 s of travel.
        self.history = max(2, int(c.get("history", 8)))
        # Minimum travel over the window before a heading is trusted: the
        # larger of an absolute floor and a fraction of the object's own
        # height. The fraction is what makes it work at both depths.
        self.min_travel_px = float(c.get("min_travel_px", 6.0))
        self.min_travel_rel = float(c.get("min_travel_rel", 0.15))
        # Cosine between travel and lane flow. Beyond +/- this it is a call;
        # inside it the heading is sideways and no verdict is given (a lane
        # change should not read as wrong-way).
        self.align_cos = float(c.get("align_cos", 0.5))
        # Ignore an object whose box is within this many px of a frame edge:
        # its ground point is not where the vehicle is.
        self.edge_px = max(0, int(c.get("edge_px", 2)))
        # Keep-out band around the divider, as a fraction of the object's own
        # height. Replaces a flat 14 px, which was metres of road at the top
        # of the frame and centimetres at the bottom.
        self.divider_margin_rel = float(c.get("divider_margin_rel", 0.35))
        self.divider_margin_min_px = float(c.get("divider_margin_min_px", 6.0))
        # Same idea for the lane polygon boundary.
        self.lane_margin_rel = float(c.get("lane_margin_rel", 0.15))
        self.lane_margin_min_px = float(c.get("lane_margin_min_px", 4.0))
        # A track the detector lost and re-acquired is not the same
        # observation. MEASURED on a full run of samples/plate_test.mp4: track
        # 349 vanished for 26 frames and came back 218 px lower down the
        # frame; the velocity window spanned the gap, read it as 190 px of
        # DOWNWARD travel, and raised the run's only wrong-way alert on a
        # vehicle driving correctly. (Its class also flipped truck ->
        # motorcycle across the gap, i.e. ByteTrack had reused the id for a
        # different object.) Position history older than this many analysed
        # frames is therefore discarded rather than differenced.
        self.max_gap_frames = max(1, int(c.get("max_gap_frames", 5)))
        # Same idea within a small gap: a single step longer than this many of
        # the object's own heights is a detector jump or an identity switch,
        # not motion. Real motion peaks around 0.2 box-heights per frame.
        self.max_step_rel = float(c.get("max_step_rel", 1.5))
        # Confirm wrong-way on k opposing samples out of the last n. NOT
        # consecutive - see the module docstring.
        self.confirm_window = max(1, int(c.get("confirm_window", 8)))
        self.confirm_count = max(1, int(c.get("confirm_count", 5)))
        if self.confirm_count > self.confirm_window:
            self.confirm_count = self.confirm_window

    def travel_gate(self, box_h: float) -> float:
        return max(self.min_travel_px, self.min_travel_rel * float(box_h))

    def step_limit(self, box_h: float) -> float:
        return max(self.min_travel_px, self.max_step_rel * float(box_h))

    def divider_margin(self, box_h: float) -> float:
        return max(self.divider_margin_min_px,
                   self.divider_margin_rel * float(box_h))

    def lane_margin(self, box_h: float) -> float:
        return max(self.lane_margin_min_px, self.lane_margin_rel * float(box_h))

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


class Lane:
    """One carriageway: an area, the direction traffic flows in it, and who
    is allowed in it."""

    def __init__(self, name: str, polygon, flow, allowed=None, side=None,
                 flow_source: str = "config"):
        self.name = str(name)
        self.polygon = polygon              # pixels, perimeter order
        self.flow = flow                    # unit vector, image coords
        self.allowed = [str(a) for a in (allowed or [])]
        self.side = side                    # +1/-1 relative to the divider
        self.flow_source = flow_source      # config | learned | cardinal
        # Cleared when observed motion contradicts this lane's configured
        # flow. Membership and restricted-class checks continue; only the
        # wrong-way verdict is withheld, and only for THIS lane. Emitting
        # alerts from geometry already known to be wrong is worse than
        # emitting none - it is what made the previous version untrustworthy.
        self.judge = True

    @property
    def has_flow(self) -> bool:
        return abs(self.flow[0]) > 1e-9 or abs(self.flow[1]) > 1e-9

    def contains(self, px, py) -> bool:
        return point_in_polygon(px, py, self.polygon)

    def __repr__(self):
        return f"Lane({self.name!r} flow={self.flow} side={self.side})"


class Verdict:
    """One object, one frame: the answer plus the evidence for it.

    The evidence is not decoration. A wrong-way alert that cannot be explained
    cannot be trusted, and the previous version latched a flag permanently
    into the database with nothing recorded about why.
    """

    __slots__ = ("lane", "flag", "reason", "alignment", "travel_px",
                 "opposing", "confirmed", "heading")

    def __init__(self, lane="", flag=OK, reason="", alignment=None,
                 travel_px=None, opposing=False, confirmed=False, heading=None):
        self.lane = lane
        self.flag = flag
        self.reason = reason
        self.alignment = alignment
        self.travel_px = travel_px
        self.opposing = opposing
        self.confirmed = confirmed
        self.heading = heading

    def detail(self) -> dict:
        """Compact, JSON-safe evidence for the events table."""
        out = {"lane": self.lane, "flag": self.flag}
        if self.reason:
            out["reason"] = self.reason
        if self.alignment is not None:
            out["alignment"] = round(float(self.alignment), 3)
        if self.travel_px is not None:
            out["travel_px"] = round(float(self.travel_px), 1)
        if self.heading is not None:
            out["heading"] = [round(float(self.heading[0]), 3),
                              round(float(self.heading[1]), 3)]
        return out

    def __repr__(self):
        return (f"Verdict(lane={self.lane!r} flag={self.flag!r} "
                f"reason={self.reason!r} align={self.alignment})")


class _TrackState:
    """Per-track history. Lives in the model, NOT in the shared TrackStore.

    That matters for the parallel story: the compute phase of an analyzer must
    not read or write state another analyzer also touches, or it cannot be
    moved off the main thread. The previous code depended on
    `det.extra["prev_xy"]`, which the pipeline had to pre-compute for it.
    """

    __slots__ = ("points", "verdicts")

    def __init__(self, history: int, window: int):
        self.points = deque(maxlen=history)     # (x, y, frame_no)
        self.verdicts = deque(maxlen=window)    # bool: opposing?


class LaneModel:
    """The lanes of one camera, plus the wrong-side rule over them."""

    def __init__(self, lanes, divider=None, thresholds=None,
                 width: int = 0, height: int = 0, note: str = ""):
        self.lanes = list(lanes or [])
        self.divider = divider              # geometry.Line or None
        self.t = thresholds or Thresholds()
        self.width = int(width)
        self.height = int(height)
        self.note = note                    # how this model was arrived at
        self._tracks: dict = {}

    # --- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, lanes_cfg, width, height, divider_cfg=None,
                    rules=None, units="auto", verbose=True):
        """Build from the `lanes:` / `divider:` config blocks.

        Every polygon is passed through ensure_simple, so a bowtie saved by an
        older tools/draw_lanes.py is repaired here rather than silently
        inverting containment for the rest of the run.
        """
        thresholds = Thresholds(rules)
        lanes = []
        for entry in (lanes_cfg or []):
            entry = dict(entry or {})
            raw = entry.get("polygon", []) or []
            if len(raw) < 3:
                if verbose:
                    print(f"[lanes] lane {entry.get('name')!r} has "
                          f"{len(raw)} points, needs 3+; skipped")
                continue
            ring = scale_points(raw, width, height,
                                entry.get("units", units))
            ring, _ = ensure_simple(ring, entry.get("name", "lane"), verbose)
            # `flow` (a vector) wins over `direction` (a cardinal name),
            # because a diagonal road cannot be described by the latter.
            if entry.get("flow") is not None:
                flow = direction_vector(entry["flow"])
                source = "config"
            else:
                flow = direction_vector(entry.get("direction"))
                source = "cardinal"
            if not (abs(flow[0]) > 1e-9 or abs(flow[1]) > 1e-9) and verbose:
                print(f"[lanes] lane {entry.get('name')!r} has no usable "
                      f"direction/flow; membership only, no wrong-way check")
            lanes.append(Lane(entry.get("name", f"lane{len(lanes) + 1}"),
                              ring, flow, entry.get("allowed"),
                              side=entry.get("side"), flow_source=source))
        divider = None
        dcfg = divider_cfg or {}
        pts = dcfg.get("points") if isinstance(dcfg, dict) else dcfg
        if pts and len(pts) >= 2:
            scaled = scale_points(pts, width, height,
                                  (dcfg.get("units", units)
                                   if isinstance(dcfg, dict) else units))
            divider = Line(scaled[0], scaled[1])
            if not divider.valid:
                if verbose:
                    print("[lanes] divider points are identical; ignoring")
                divider = None
        model = cls(lanes, divider, thresholds, width, height,
                    note="from config")
        model.assign_sides()
        return model

    def assign_sides(self):
        """Label each lane +1/-1 by which side of the divider it sits on.

        Uses the polygon centroid, so it survives lanes that share the divider
        as an edge - which is exactly the case polygon containment got wrong.
        """
        if self.divider is None:
            return
        for lane in self.lanes:
            if lane.side is not None:
                continue
            n = len(lane.polygon) or 1
            cx = sum(p[0] for p in lane.polygon) / n
            cy = sum(p[1] for p in lane.polygon) / n
            lane.side = self.divider.side(cx, cy)

    # --- per-frame state ---------------------------------------------------

    def _state(self, tid) -> _TrackState:
        st = self._tracks.get(tid)
        if st is None:
            st = _TrackState(self.t.history, self.t.confirm_window)
            self._tracks[tid] = st
        return st

    def _break_history(self, st, px, py, box_h, frame_no) -> bool:
        """Drop history across a detection gap or an implausible jump.

        Differencing positions either side of a gap measures where the tracker
        re-found the object, not where the object went.
        """
        if not st.points:
            return False
        lx, ly, last_frame = st.points[-1]
        gap = frame_no - last_frame
        step = ((px - lx) ** 2 + (py - ly) ** 2) ** 0.5
        if gap > self.t.max_gap_frames or step > self.t.step_limit(box_h):
            st.points.clear()
            st.verdicts.clear()
            return True
        return False

    def forget(self, tid):
        self._tracks.pop(tid, None)

    def track_count(self) -> int:
        return len(self._tracks)

    def is_clipped(self, bbox) -> bool:
        """Box touching a frame edge -> its ground point is not the vehicle."""
        if not (self.width and self.height):
            return False
        x1, y1, x2, y2 = bbox
        m = self.t.edge_px
        return (x1 <= m or y1 <= m
                or x2 >= self.width - 1 - m or y2 >= self.height - 1 - m)

    def lane_at(self, px, py, box_h=0.0):
        """Which lane contains this point, and is it safely inside it?

        Returns (lane, confident). `confident` is False when the point is
        within a depth-scaled margin of the lane boundary or of the divider,
        which is where membership is a coin flip and the old code guessed by
        list order.
        """
        found = None
        for lane in self.lanes:
            if lane.contains(px, py):
                found = lane
                break
        if found is None:
            return (None, False)
        confident = distance_to_edge(px, py, found.polygon) >= self.t.lane_margin(box_h)
        if confident and self.divider is not None and self.divider.valid:
            if self.divider.distance(px, py) < self.t.divider_margin(box_h):
                confident = False
        return (found, confident)

    def heading(self, tid, box_h=0.0):
        """Estimated (unit heading, travel px) over the track's window.

        Displacement between the oldest and newest samples, rather than
        frame-to-frame: at 1-2 px per frame - which is what the far half of
        these clips actually moves - a single-frame difference is noise.
        """
        st = self._tracks.get(tid)
        if st is None or len(st.points) < 2:
            return (None, 0.0)
        (x0, y0, _), (x1, y1, _) = st.points[0], st.points[-1]
        dx, dy = x1 - x0, y1 - y0
        travel = (dx * dx + dy * dy) ** 0.5
        if travel < self.t.travel_gate(box_h):
            return (None, travel)
        return ((dx / travel, dy / travel), travel)

    def evaluate(self, tid, bbox, cls_name="", frame_no=0, warmup=False) -> Verdict:
        """The wrong-side rule for one tracked object in one frame."""
        x1, y1, x2, y2 = bbox
        box_h = max(1.0, float(y2 - y1))
        px, py = (x1 + x2) / 2.0, float(y2)   # ground contact point

        if self.is_clipped(bbox):
            # Do not feed a clipped box into the history either: a box pinned
            # against the bottom edge reports a ground point that stops moving
            # while the vehicle keeps coming.
            return Verdict(reason=EDGE_CLIPPED)

        st = self._state(tid)
        jumped = self._break_history(st, px, py, box_h, frame_no)
        st.points.append((px, py, frame_no))

        lane, confident = self.lane_at(px, py, box_h)
        if lane is None:
            st.verdicts.clear()
            return Verdict(reason=NO_LANE)

        flag, reason = OK, ""
        alignment = None
        heading, travel = self.heading(tid, box_h)
        if jumped:
            # One sample of history: no heading, and nothing to confirm from.
            return Verdict(lane=lane.name, reason=TRACK_JUMP)

        if not lane.judge:
            reason = UNVERIFIED
        elif not lane.has_flow:
            reason = NO_MOTION if heading is None else ""
        elif heading is None:
            reason = NO_MOTION
            st.verdicts.clear()
        elif not confident:
            reason = NEAR_DIVIDER
        else:
            alignment = heading[0] * lane.flow[0] + heading[1] * lane.flow[1]
            opposing = alignment <= -self.t.align_cos
            if not opposing and alignment < self.t.align_cos:
                reason = AMBIGUOUS       # crossing the lane, not opposing it
            st.verdicts.append(bool(opposing))
            if warmup:
                reason = WARMUP
            elif sum(st.verdicts) >= self.t.confirm_count:
                flag = WRONG_WAY
                reason = ""

        # A restricted lane (bus-only, no-trucks) is a separate rule: it can
        # apply to a vehicle travelling perfectly correctly.
        if lane.allowed and cls_name and cls_name not in lane.allowed:
            flag = WRONG_LANE if flag == OK else WRONG_WAY_AND_LANE

        return Verdict(lane=lane.name, flag=flag, reason=reason,
                       alignment=alignment, travel_px=travel,
                       opposing=bool(st.verdicts and st.verdicts[-1]),
                       confirmed=(flag in (WRONG_WAY, WRONG_WAY_AND_LANE)),
                       heading=heading)

    # --- reporting ---------------------------------------------------------

    def describe(self) -> str:
        from .geometry import describe_vector
        if not self.lanes:
            return "no lanes"
        parts = [f"{L.name}[{describe_vector(L.flow)}/{L.flow_source}"
                 f"{'' if L.judge else '/NOT-JUDGED'}]"
                 for L in self.lanes]
        div = "divider yes" if (self.divider and self.divider.valid) else "no divider"
        return f"{len(self.lanes)} lanes: {', '.join(parts)} | {div} | {self.note}"

    def as_config(self, width=None, height=None) -> dict:
        """Round-trip back to a `lanes:`/`divider:` block, in ratio coords."""
        from .geometry import to_ratios
        w = int(width or self.width or 1)
        h = int(height or self.height or 1)
        out = {"lanes": [{"name": L.name,
                          "polygon": to_ratios(L.polygon, w, h),
                          "flow": [round(L.flow[0], 4), round(L.flow[1], 4)],
                          "allowed": list(L.allowed)}
                         for L in self.lanes]}
        if self.divider is not None and self.divider.valid:
            out["divider"] = {"points": to_ratios(self.divider.as_points(), w, h)}
        return out
