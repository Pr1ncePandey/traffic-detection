"""Learn the coming/going divider from motion, and check a config against it.

WHY THIS EXISTS

The divider was a hand-typed constant. `config.yaml` and `cameras/demo.yaml`
both carried the same two polygons, tuned by eye on samples/input.mp4, and any
other source silently inherited them. Run on samples/plate_test.mp4 - which is
what outputs/ was last built from - that divider cuts diagonally through the
middle of ONE one-way carriageway, labels its left half "coming", and every
vehicle correctly driving up that half is a wrong-way alert. Measured: 2 of 90
vehicle tracks flagged, both travelling correctly, 0 true positives.

Nothing checked the geometry against the video. This module is that check, and
it is the same code path that can propose the geometry in the first place:

  MotionSurvey   watch ground points for a while, group tracks by heading
  suggest()      -> LaneSuggestion: a divider and one lane per direction
  verify()       -> LaneVerification: does an EXISTING config match reality?

HONESTY RULE

If only one direction of travel is observed, there is no evidence of a
divider, and this module says so instead of inventing one. Both sample clips
are that case - `input.mp4` has 8 clear tracks, all moving up; `plate_test.mp4`
has 28, all moving up. The old auto mode split the frame in half and
manufactured an oncoming lane out of nothing, which is where the false alerts
came from. A single carriageway with a known flow direction is a complete,
useful answer: it still detects a vehicle coming the other way.
"""

from .geometry import (clamp_ring, clip_to_halfplane, convex_hull,
                       describe_vector, dilate_ring, fit_line_through,
                       frame_ring, unit)


class Trajectory:
    """One track's motion over the survey window."""

    __slots__ = ("tid", "points", "box_h", "cls_name")

    def __init__(self, tid, cls_name=""):
        self.tid = tid
        self.points = []        # (x, y) ground points
        self.box_h = 0.0        # mean bbox height, the depth proxy
        self.cls_name = cls_name

    def add(self, x, y, box_h):
        n = len(self.points)
        self.points.append((float(x), float(y)))
        self.box_h = (self.box_h * n + float(box_h)) / (n + 1)

    @property
    def travel(self) -> float:
        if len(self.points) < 2:
            return 0.0
        (x0, y0), (x1, y1) = self.points[0], self.points[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    @property
    def heading(self):
        if len(self.points) < 2:
            return (0.0, 0.0)
        (x0, y0), (x1, y1) = self.points[0], self.points[-1]
        return unit(x1 - x0, y1 - y0)

    @property
    def mean_x(self) -> float:
        return sum(x for x, _ in self.points) / max(1, len(self.points))


class LaneSuggestion:
    """Proposed geometry, plus how confident the evidence behind it is."""

    def __init__(self, lanes, divider=None, note="", tracks_used=0,
                 groups=(0, 0), headings=((0.0, 0.0), (0.0, 0.0))):
        self.lanes = lanes            # list of dicts: name/polygon/flow
        self.divider = divider        # geometry.Line or None
        self.note = note
        self.tracks_used = tracks_used
        self.groups = groups          # (n_dominant, n_opposing)
        self.headings = headings

    @property
    def two_way(self) -> bool:
        return self.divider is not None and len(self.lanes) > 1

    def report(self) -> str:
        lines = [f"[calibrate] {self.note}",
                 f"[calibrate] tracks with clear motion: {self.tracks_used} "
                 f"(dominant {self.groups[0]}, opposing {self.groups[1]})"]
        for lane in self.lanes:
            lines.append(f"[calibrate]   lane {lane['name']!r} flow "
                         f"{describe_vector(lane['flow'])}")
        if self.divider is not None:
            lines.append(f"[calibrate]   divider {self.divider}")
        return "\n".join(lines)


class LaneVerification:
    """Per-lane comparison of configured flow against observed flow."""

    AGREES = "agrees"
    OPPOSES = "OPPOSES"
    # Observed flow has MORE against the configured flow than with it, but is
    # not fully antiparallel. Measured need for this: on samples/plate_test.mp4
    # with a 120-frame warmup, the misconfigured lane collected only 4 tracks
    # and scored -0.45 - just inside a -0.5 antiparallel threshold - so the
    # mismatch went unreported and 105 false wrong-way rows followed. Anything
    # pointing against the configured direction is enough to distrust it.
    CONTRADICTS = "CONTRADICTS"
    SIDEWAYS = "sideways"
    NO_DATA = "no-data"

    def __init__(self, rows, tracks_used=0):
        self.rows = rows              # list of dicts
        self.tracks_used = tracks_used

    @property
    def opposing(self) -> list:
        """Lanes whose configured flow the evidence argues against."""
        return [r for r in self.rows
                if r["verdict"] in (self.OPPOSES, self.CONTRADICTS)]

    @property
    def ok(self) -> bool:
        return not self.opposing

    def report(self) -> str:
        if not self.rows:
            return "[verify] no lanes to check"
        out = [f"[verify] checked {self.tracks_used} tracks with clear motion "
               f"against {len(self.rows)} configured lane(s)"]
        for r in self.rows:
            if r["verdict"] == self.NO_DATA:
                out.append(f"[verify]   {r['lane']}: no traffic observed "
                           f"({r['n']} tracks) - direction unverified")
                continue
            out.append(f"[verify]   {r['lane']}: configured "
                       f"{describe_vector(r['configured'])}, observed "
                       f"{describe_vector(r['observed'])} over {r['n']} tracks "
                       f"-> {r['verdict']} (alignment {r['alignment']:+.2f})")
        for r in self.opposing:
            out.append(f"[verify] MISMATCH ({r['verdict']}): lane {r['lane']!r} "
                       f"is configured to flow {describe_vector(r['configured'])} "
                       f"but its {r['n']} observed track(s) flow "
                       f"{describe_vector(r['observed'])}. Correctly-driving "
                       f"vehicles in it would be reported wrong-way.")
        return "\n".join(out)


class MotionSurvey:
    """Accumulate ground-point trajectories, then draw conclusions from them.

    Fed the same detections the analyzers see, so calibration and runtime can
    never disagree about what a "ground point" or a "clear motion track" is.
    """

    def __init__(self, width, height, min_samples=6, min_travel_px=25.0,
                 min_travel_rel=0.5, opposing_min_tracks=3,
                 opposing_min_frac=0.12, align_cos=0.5, bands=8):
        self.width = int(width)
        self.height = int(height)
        self.min_samples = int(min_samples)
        self.min_travel_px = float(min_travel_px)
        self.min_travel_rel = float(min_travel_rel)
        self.opposing_min_tracks = int(opposing_min_tracks)
        self.opposing_min_frac = float(opposing_min_frac)
        self.align_cos = float(align_cos)
        self.bands = max(2, int(bands))
        self.tracks: dict = {}
        self.frames = 0

    # --- collection --------------------------------------------------------

    def observe_frame(self):
        self.frames += 1

    def observe(self, tid, bbox, cls_name=""):
        """Feed one tracked box. Ground point + bbox height, nothing else."""
        x1, y1, x2, y2 = bbox
        box_h = max(1.0, float(y2 - y1))
        traj = self.tracks.get(tid)
        if traj is None:
            traj = Trajectory(tid, cls_name)
            self.tracks[tid] = traj
        traj.add((x1 + x2) / 2.0, float(y2), box_h)

    def clear(self):
        self.tracks.clear()
        self.frames = 0

    # --- analysis ----------------------------------------------------------

    def moving(self) -> list:
        """Tracks that travelled far enough to have a trustworthy heading.

        Depth-scaled, like the runtime gate: a far vehicle covering 30 px has
        crossed much more road than a near one covering the same 30 px.
        """
        out = []
        for traj in self.tracks.values():
            if len(traj.points) < self.min_samples:
                continue
            gate = max(self.min_travel_px, self.min_travel_rel * traj.box_h)
            if traj.travel >= gate:
                out.append(traj)
        return out

    def _split_by_heading(self, trajs):
        """Two antipodal groups: with the dominant heading, and against it.

        Seeded from the longest-travelling track, then the group means are
        recomputed twice. Averaging headings without splitting first is what
        you must not do - two opposing flows average to nothing.
        """
        if not trajs:
            return [], [], (0.0, 0.0), (0.0, 0.0)
        seed = max(trajs, key=lambda t: t.travel).heading
        group_a, group_b = trajs, []
        for _ in range(3):
            group_a, group_b = [], []
            for t in trajs:
                h = t.heading
                (group_a if (h[0] * seed[0] + h[1] * seed[1]) >= 0 else
                 group_b).append(t)
            if not group_a:
                break
            seed = self._mean_heading(group_a)
        return group_a, group_b, self._mean_heading(group_a), self._mean_heading(group_b)

    @staticmethod
    def _mean_heading(trajs):
        """Travel-weighted mean heading. Long tracks are better evidence."""
        if not trajs:
            return (0.0, 0.0)
        sx = sum(t.heading[0] * t.travel for t in trajs)
        sy = sum(t.heading[1] * t.travel for t in trajs)
        return unit(sx, sy)

    def _fit_divider(self, left, right):
        """Divider through the midpoints between the two flows, per y-band.

        Per-band rather than one global midpoint because the gap between the
        carriageways narrows with distance; a single split would be wrong at
        both ends of the frame.
        """
        band_h = self.height / self.bands
        mids = []
        for i in range(self.bands):
            lo, hi = i * band_h, (i + 1) * band_h
            lx = [x for t in left for (x, y) in t.points if lo <= y < hi]
            rx = [x for t in right for (x, y) in t.points if lo <= y < hi]
            if not lx or not rx:
                continue
            gap_lo, gap_hi = max(lx), min(rx)
            if gap_hi < gap_lo:
                # The two flows overlap in x in this band: they are not
                # separated left/right here, so this band says nothing.
                continue
            mids.append(((gap_lo + gap_hi) / 2.0, (lo + hi) / 2.0))
        if len(mids) < 2:
            return None, mids
        return fit_line_through(mids), mids

    def suggest(self) -> LaneSuggestion:
        """Propose lanes (and a divider, if the evidence supports one)."""
        trajs = self.moving()
        if not trajs:
            return LaneSuggestion(
                [], None,
                note=(f"no tracks with clear motion in {self.frames} frames; "
                      f"cannot infer lanes. Run longer, or set lanes "
                      f"explicitly with tools/draw_lanes.py."),
                tracks_used=0)

        group_a, group_b, head_a, head_b = self._split_by_heading(trajs)
        n_total = len(trajs)
        need = max(self.opposing_min_tracks,
                   int(round(self.opposing_min_frac * n_total)))
        two_way = len(group_b) >= need

        if not two_way:
            hull = convex_hull([p for t in group_a for p in t.points])
            ring = clamp_ring(dilate_ring(hull, 0.25 * max(
                1.0, sum(t.box_h for t in group_a) / len(group_a))),
                self.width, self.height)
            note = (f"ONE direction of travel observed ({len(group_a)} tracks "
                    f"{describe_vector(head_a)}, {len(group_b)} opposing - "
                    f"needed {need}). No divider inferred: there is no evidence "
                    f"of a second carriageway in view. A vehicle heading the "
                    f"other way through this area will still be flagged.")
            return LaneSuggestion(
                [{"name": "carriageway", "polygon": ring or frame_ring(
                    self.width, self.height), "flow": head_a, "allowed": []}],
                None, note, n_total, (len(group_a), len(group_b)),
                (head_a, head_b))

        # Two flows. Which group is on the left of the frame decides the sign.
        mean_a = sum(t.mean_x for t in group_a) / len(group_a)
        mean_b = sum(t.mean_x for t in group_b) / len(group_b)
        left, right = ((group_a, group_b) if mean_a <= mean_b
                       else (group_b, group_a))
        head_left = self._mean_heading(left)
        head_right = self._mean_heading(right)
        divider, mids = self._fit_divider(left, right)
        if divider is None:
            hull = clamp_ring(convex_hull([p for t in trajs for p in t.points]),
                              self.width, self.height)
            note = (f"two opposing flows found ({len(left)} vs {len(right)} "
                    f"tracks) but they overlap in x, so no left/right divider "
                    f"could be fitted (only {len(mids)} usable bands). The "
                    f"roads may be split top/bottom rather than left/right - "
                    f"set the divider by hand with tools/draw_lanes.py.")
            return LaneSuggestion(
                [{"name": "carriageway", "polygon": hull or frame_ring(
                    self.width, self.height), "flow": head_left, "allowed": []}],
                None, note, n_total, (len(group_a), len(group_b)),
                (head_a, head_b))

        frame = frame_ring(self.width, self.height)
        # Sample points decide which sign of the divider each side is on,
        # rather than assuming a winding order.
        probe_left = (sum(t.mean_x for t in left) / len(left),
                      self.height / 2.0)
        side_left = divider.side(probe_left[0], probe_left[1]) or 1
        ring_left = clip_to_halfplane(frame, divider, side_left)
        ring_right = clip_to_halfplane(frame, divider, -side_left)
        note = (f"two carriageways found: {len(left)} tracks "
                f"{describe_vector(head_left)} on one side, {len(right)} "
                f"{describe_vector(head_right)} on the other. Divider fitted "
                f"through {len(mids)} y-bands.")
        return LaneSuggestion(
            [{"name": "left_side", "polygon": ring_left, "flow": head_left,
              "allowed": []},
             {"name": "right_side", "polygon": ring_right, "flow": head_right,
              "allowed": []}],
            divider, note, n_total, (len(group_a), len(group_b)),
            (head_a, head_b))

    def verify(self, model, min_tracks: int = 3) -> LaneVerification:
        """Compare an existing LaneModel's configured flow against reality.

        A track is attributed to the lane that contains most of its ground
        points, so a vehicle that clips a neighbouring polygon for a frame or
        two does not vote in both.
        """
        trajs = self.moving()
        per_lane: dict = {L.name: [] for L in model.lanes}
        for traj in trajs:
            counts: dict = {}
            for (x, y) in traj.points:
                lane, _ = model.lane_at(x, y, traj.box_h)
                if lane is not None:
                    counts[lane.name] = counts.get(lane.name, 0) + 1
            if not counts:
                continue
            best = max(counts, key=counts.get)
            per_lane.setdefault(best, []).append(traj)

        rows = []
        for lane in model.lanes:
            group = per_lane.get(lane.name, [])
            observed = self._mean_heading(group)
            row = {"lane": lane.name, "configured": lane.flow,
                   "observed": observed, "n": len(group), "alignment": 0.0,
                   "verdict": LaneVerification.NO_DATA}
            if len(group) >= min_tracks and lane.has_flow:
                align = observed[0] * lane.flow[0] + observed[1] * lane.flow[1]
                row["alignment"] = align
                if align >= self.align_cos:
                    row["verdict"] = LaneVerification.AGREES
                elif align <= -self.align_cos:
                    row["verdict"] = LaneVerification.OPPOSES
                elif align < 0.0:
                    row["verdict"] = LaneVerification.CONTRADICTS
                else:
                    row["verdict"] = LaneVerification.SIDEWAYS
            rows.append(row)
        return LaneVerification(rows, len(trajs))
