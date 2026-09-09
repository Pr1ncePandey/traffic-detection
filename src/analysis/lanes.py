"""Wrong-side / wrong-lane detection, as a staged analyzer.

This runs AFTER detection, classification and tracking, on the objects those
produced, and it is independent of every other analysis: its compute phase
reads an immutable AnalysisView and its own state, so it can sit next to
congestion or ANPR on another thread (see analysis/stage.py). The rule itself
lives in lane_model.py and the geometry in geometry.py, both free of frames,
config and storage - which is how tools/test_lanes.py can check the rule
without a video.

    geometry.py          points, rings, lines. No domain.
    lane_model.py        the wrong-side rule. No I/O.
    lane_calibration.py  learn the divider from motion; verify a config.
    lanes.py (here)      config -> model, per-frame plumbing, overlay.

THE BUG THIS FIXES

The divider was a hand-typed constant tuned on one clip and inherited by every
other source unchecked. On samples/plate_test.mp4 it cut diagonally through
the middle of a single one-way carriageway: 2 of 90 vehicle tracks were
reported wrong-way, both driving correctly, 0 real offenders. So the geometry
is now either learned from motion (`lanes_mode: auto`) or CHECKED against
motion (`lanes_mode: explicit`), during a warmup window in which no alert is
raised. A configured lane whose traffic demonstrably flows the other way is
now reported at startup instead of generating a false alert per vehicle for
the rest of the run.
"""

import cv2
import numpy as np

from ..detectors.classes import ON_ROAD, VEHICLE
from .base import Findings, register
from .geometry import describe_vector, unit
from .lane_calibration import MotionSurvey
from .lane_model import (OK, WRONG_LANE, WRONG_WAY, WRONG_WAY_AND_LANE,
                         LaneModel, Thresholds)

COLOR_DIVIDER = (0, 0, 255)
COLOR_LANE = (255, 255, 0)
COLOR_WARMUP = (0, 200, 255)
COLOR_FLOW = (0, 255, 255)


class LaneAnalyzer:
    """Lane membership + wrong-way / wrong-lane flags.

    `concurrent = True` is a claim about compute(): it reads the view, mutates
    only this instance's own model/survey/counters, and returns a Findings.
    One frame is computed at a time per analyzer, so that state needs no lock.
    Nothing shared - not the TrackStore, not the annotated canvas, not another
    analyzer's output - is touched before apply()/draw(), which are serial.
    """

    name = "lanes"
    concurrent = True

    def __init__(self, cfg: dict):
        self._cfg = cfg or {}
        self.model = None
        self.survey = None
        self.mode = "explicit"
        self.enabled = False
        self.width = self.height = 0
        # Warmup: measure before judging. Also gives every track enough
        # history for a heading, which is why no alert can fire on frame 1.
        self.warmup_frames = 0
        self.frames_seen = 0
        self.calibrated = False
        self.on_mismatch = "suppress"
        self.verification = None
        self.suggestion = None
        self.notes: list = []
        # Per-track bookkeeping. Own state -> safe inside compute().
        # These two are DEDUPE sets and are cleared by forget(), so they stay
        # bounded on a 24/7 feed...
        self._alerted_way: set = set()
        self._alerted_lane: set = set()
        self._lane_seen: dict = {}
        # ...which is exactly why the run totals cannot be read off them. The
        # pipeline calls forget() for every track it evicts and for every
        # track still alive at shutdown, so by the time summary() runs those
        # sets are empty - the first version of this file reported
        # `wrong_way_ids: []` for a run that had flagged 105 detections.
        self._wrong_way_total = 0
        self._wrong_lane_total = 0
        self._wrong_way_ids: list = []
        self._wrong_lane_ids: list = []
        self._lane_totals: dict = {}
        self._reasons: dict = {}
        self.evaluated = 0
        self.skipped = 0

    # --- setup -------------------------------------------------------------

    def setup(self, source, cfg: dict):
        cfg = cfg or {}
        self.width = int(getattr(source, "width", 0) or 0)
        self.height = int(getattr(source, "height", 0) or 0)
        self.mode = str(cfg.get("lanes_mode", "explicit")).lower()
        rules = dict(cfg.get("lanes_rules", {}) or {})
        self.on_mismatch = str(rules.get("on_mismatch", "suppress")).lower()
        self.warmup_frames = max(0, int(rules.get("warmup_frames", 120)))
        thresholds = Thresholds(rules)

        if self.mode == "off":
            self.enabled = False
            self._note("lanes_mode: off - no lane or wrong-way analysis")
            return

        lanes_cfg = cfg.get("lanes", []) or []
        if self.mode == "explicit" and not lanes_cfg:
            # Falling back to `auto` beats running with no lanes at all, and
            # beats the old behaviour of silently using another clip's numbers.
            self._note("lanes_mode: explicit but no lanes configured; "
                       "measuring from motion instead (lanes_mode: auto)")
            self.mode = "auto"

        if self.mode == "auto":
            self.model = LaneModel([], None, thresholds, self.width,
                                   self.height, note="awaiting calibration")
        else:
            self.model = LaneModel.from_config(
                lanes_cfg, self.width, self.height,
                divider_cfg=cfg.get("divider"), rules=rules,
                units=cfg.get("lanes_units", "auto"))

        if self.warmup_frames > 0:
            self.survey = MotionSurvey(
                self.width, self.height,
                min_samples=int(rules.get("survey_min_samples", 6)),
                min_travel_px=float(rules.get("survey_min_travel_px", 25.0)),
                min_travel_rel=float(rules.get("survey_min_travel_rel", 0.5)),
                opposing_min_tracks=int(rules.get("opposing_min_tracks", 3)),
                opposing_min_frac=float(rules.get("opposing_min_frac", 0.12)),
                align_cos=thresholds.align_cos)
        elif self.mode == "auto":
            self._note("lanes_mode: auto needs warmup_frames > 0 to measure "
                       "anything; lane analysis disabled")
            self.enabled = False
            return

        self.enabled = True
        what = ("learning lanes from motion" if self.mode == "auto"
                else f"checking {len(self.model.lanes)} configured lane(s) "
                     f"against motion")
        print(f"[lanes] {what} over the first {self.warmup_frames} analysed "
              f"frames; no wrong-way alert until then")
        if self.mode != "auto":
            print(f"[lanes] {self.model.describe()}")

    # --- compute (pure; may run off the main thread) ------------------------

    def compute(self, view) -> Findings:
        found = Findings(analyzer=self.name)
        if not self.enabled or self.model is None:
            return found
        self.frames_seen += 1
        warming = self.frames_seen <= self.warmup_frames

        if self.survey is not None and warming:
            self.survey.observe_frame()
            for box in view.tracked():
                # Vehicles only: a pedestrian crossing the carriageway would
                # otherwise vote in the flow estimate for the road.
                if box.group == VEHICLE and not self.model.is_clipped(box.bbox):
                    self.survey.observe(box.track_id, box.bbox, box.cls_name)

        if self.survey is not None and not warming and not self.calibrated:
            self._finish_warmup(found)

        for box in view.tracked():
            if box.group not in ON_ROAD:
                continue
            verdict = self.model.evaluate(box.track_id, box.bbox, box.cls_name,
                                          frame_no=view.frame_no,
                                          warmup=warming)
            if verdict.lane:
                self.evaluated += 1
            else:
                self.skipped += 1
            if verdict.reason:
                self._reasons[verdict.reason] = self._reasons.get(verdict.reason, 0) + 1
            found.set(box.track_id, lane_id=verdict.lane, lane_flag=verdict.flag)
            if verdict.lane:
                seen = self._lane_seen.setdefault(verdict.lane, set())
                if box.track_id not in seen:
                    seen.add(box.track_id)
                    self._lane_totals[verdict.lane] = \
                        self._lane_totals.get(verdict.lane, 0) + 1
            self._collect_alerts(found, box, verdict)

        found.overlay = {"warming": warming,
                         "remaining": max(0, self.warmup_frames - self.frames_seen),
                         "counts": dict(self._lane_totals)}
        return found

    def _collect_alerts(self, found, box, verdict):
        """One event per track per kind, with the evidence attached."""
        tid = box.track_id
        if verdict.flag in (WRONG_WAY, WRONG_WAY_AND_LANE) and tid not in self._alerted_way:
            self._alerted_way.add(tid)
            self._wrong_way_total += 1
            self._remember(self._wrong_way_ids, tid)
            found.event("wrong_way", {**verdict.detail(), "cls": box.cls_name}, tid)
        if verdict.flag in (WRONG_LANE, WRONG_WAY_AND_LANE) and tid not in self._alerted_lane:
            self._alerted_lane.add(tid)
            self._wrong_lane_total += 1
            self._remember(self._wrong_lane_ids, tid)
            found.event("wrong_lane", {**verdict.detail(), "cls": box.cls_name}, tid)

    def _finish_warmup(self, found):
        """Learn or check the geometry, once, at the end of the warmup window."""
        self.calibrated = True
        if self.mode == "auto":
            self._apply_suggestion(found)
        else:
            self._apply_verification(found)
        # The survey has served its purpose; drop the trajectories it holds so
        # a long-running feed does not carry them forever.
        self.survey.clear()

    def _apply_suggestion(self, found):
        suggestion = self.survey.suggest()
        self.suggestion = suggestion
        print(suggestion.report())
        self._note(suggestion.note)
        if not suggestion.lanes:
            self.enabled = False
            print("[lanes] no lanes could be measured; lane analysis disabled "
                  "for this run (set lanes explicitly with tools/draw_lanes.py)")
            found.event("lanes_calibrated",
                        {"result": "failed", "note": suggestion.note})
            return
        self.model = LaneModel.from_config(
            suggestion.lanes, self.width, self.height,
            divider_cfg=({"points": suggestion.divider.as_points(),
                          "units": "pixel"} if suggestion.divider else None),
            rules=self._cfg.get("lanes_rules", {}) or {}, units="pixel",
            verbose=False)
        self.model.note = "learned from motion"
        for lane in self.model.lanes:
            # from_config saw an explicit `flow` and labelled it "config";
            # it did not come from a config file, it came from the road.
            lane.flow_source = "learned"
        print(f"[lanes] {self.model.describe()}")
        print("[lanes] save this geometry with "
              "`python tools/calibrate.py --suggest-lanes --out cameras/<name>.yaml` "
              "to skip the warmup next run")
        found.event("lanes_calibrated",
                    {"result": "two_way" if suggestion.two_way else "one_way",
                     "lanes": [L["name"] for L in suggestion.lanes],
                     "tracks_used": suggestion.tracks_used,
                     "note": suggestion.note})

    def _apply_verification(self, found):
        verification = self.survey.verify(self.model)
        self.verification = verification
        print(verification.report())
        for row in verification.rows:
            if row["verdict"] == "no-data":
                self._note(f"lane {row['lane']!r}: direction never verified "
                           f"(no traffic observed during warmup)")
        if verification.ok:
            found.event("lanes_verified",
                        {"result": "ok", "tracks": verification.tracks_used})
            return
        names = [r["lane"] for r in verification.opposing]
        found.event("lanes_verified",
                    {"result": "mismatch", "lanes": names,
                     "tracks": verification.tracks_used})
        if self.on_mismatch == "suppress":
            # Keep the camera working: lane membership, counts and restricted-
            # class checks all continue. Only the wrong-way verdict, and only
            # on the lanes the evidence contradicts, is withheld.
            for row in verification.opposing:
                for lane in self.model.lanes:
                    if lane.name == row["lane"]:
                        lane.judge = False
            print(f"[lanes] on_mismatch=suppress: wrong-way checking disabled "
                  f"for {', '.join(names)}; every other lane keeps working. "
                  f"Fix the geometry (tools/calibrate.py --verify) or set "
                  f"lanes_rules.on_mismatch: flip to trust the measurement.")
            self._note(f"wrong-way checking suppressed on {', '.join(names)}: "
                       f"the configured flow contradicts observed motion")
        elif self.on_mismatch == "flip":
            for row in verification.opposing:
                for lane in self.model.lanes:
                    if lane.name == row["lane"]:
                        lane.flow = unit(*row["observed"])
                        lane.flow_source = "corrected-from-motion"
                        self.model._tracks.clear()
                print(f"[lanes] on_mismatch=flip: lane {row['lane']!r} flow "
                      f"corrected to {describe_vector(row['observed'])}")
            self._note(f"corrected the configured flow of {', '.join(names)} "
                       f"from observed motion (lanes_rules.on_mismatch: flip)")
        elif self.on_mismatch == "off":
            self.enabled = False
            print("[lanes] on_mismatch=off: lane analysis disabled rather than "
                  "reporting wrong-way against geometry that does not match")
            self._note("disabled: configured lanes contradict observed motion")
        else:
            self._note(f"WARNING: {', '.join(names)} contradict observed "
                       f"motion; wrong-way alerts from them are unreliable. "
                       f"Fix the geometry, or set lanes_rules.on_mismatch: "
                       f"flip to trust the measurement instead.")

    # --- apply / draw (serial, main thread) --------------------------------

    def apply(self, ctx, findings: Findings):
        """Only the shared-state writes. Detection fields the stage wrote."""
        store = ctx.store
        for tid, fields in findings.per_track.items():
            store.set_lane(tid, fields.get("lane_id", ""),
                           fields.get("lane_flag", OK))

    def draw(self, ctx, findings: Findings):
        """The overlay exists to make a wrong divider obvious, so it draws the
        boundary in use AND the direction each side is expected to flow.

        Labels sit at each lane's own centroid rather than at the top of its
        bounding box: two lanes that both start at y=0 - which is the normal
        case for a left/right split - printed their names on top of each
        other there, and an unreadable label is how you end up believing the
        geometry is fine.
        """
        frame = ctx.annotated
        overlay = findings.overlay or {}
        counts = overlay.get("counts", {})
        if self.model is not None:
            for lane in self.model.lanes:
                cv2.polylines(frame, [np.array(lane.polygon, np.int32)], True,
                              COLOR_LANE, 2)
            if self.model.divider is not None and self.model.divider.valid:
                ends = self.model.divider.clipped_to_frame(self.width, self.height)
                if ends:
                    cv2.line(frame, ends[0], ends[1], COLOR_DIVIDER, 3)
                    mid = ((ends[0][0] + ends[1][0]) // 2,
                           (ends[0][1] + ends[1][1]) // 2)
                    self._label(frame, "divider", (mid[0] + 10, mid[1]),
                                COLOR_DIVIDER)
            # Arrows and labels last, so the divider line cannot cover them.
            for lane in self.model.lanes:
                self._draw_lane_label(frame, lane, counts.get(lane.name, 0))
        if not self.enabled:
            self._label(frame, "lanes: disabled", (10, 112), COLOR_WARMUP)
        elif overlay.get("warming"):
            self._label(frame, f"lanes: measuring flow "
                               f"({overlay.get('remaining', 0)} frames left, "
                               f"no alerts yet)", (10, 112), COLOR_WARMUP)

    @staticmethod
    def _label(frame, text, origin, color, scale=0.55):
        """Text on a filled backing box - road markings are white, and white
        text on white paint is not a label."""
        x, y = int(origin[0]), int(origin[1])
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        h, w = frame.shape[:2]
        x = max(2, min(x, w - tw - 4))
        y = max(th + 4, min(y, h - 4))
        cv2.rectangle(frame, (x - 3, y - th - 4), (x + tw + 3, y + base),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 2)

    def _draw_lane_label(self, frame, lane, count):
        """Name, unique-vehicle count and expected flow, at the lane centroid."""
        if not lane.polygon:
            return
        n = len(lane.polygon)
        cx = int(sum(p[0] for p in lane.polygon) / n)
        cy = int(sum(p[1] for p in lane.polygon) / n)
        if lane.has_flow:
            length = max(30, int(0.08 * max(self.width, self.height)))
            tip = (int(cx + lane.flow[0] * length),
                   int(cy + lane.flow[1] * length))
            cv2.arrowedLine(frame, (cx, cy), tip, COLOR_FLOW, 3, tipLength=0.35)
        text = f"{lane.name} n={count}"
        if not lane.judge:
            # The operator must be able to see that this lane is NOT being
            # checked, or an absence of alerts reads as an absence of problems.
            text += " [flow unverified: not checked]"
        self._label(frame, text, (cx + 12, cy + 22),
                    COLOR_LANE if lane.judge else COLOR_WARMUP)

    # --- housekeeping ------------------------------------------------------

    def forget(self, tid):
        """Free per-track state for a retired track. Totals are NOT touched."""
        if self.model is not None:
            self.model.forget(tid)
        self._alerted_way.discard(tid)
        self._alerted_lane.discard(tid)
        for seen in self._lane_seen.values():
            seen.discard(tid)

    # Keep a bounded sample of offending ids for the report: the full list on
    # a busy 24/7 feed is unbounded, and a count plus the first few is what a
    # summary is actually read for.
    ID_SAMPLE = 500

    def _remember(self, bucket: list, tid):
        if len(bucket) < self.ID_SAMPLE:
            bucket.append(tid)

    def _note(self, text: str):
        if text and text not in self.notes:
            self.notes.append(text)

    def summary(self) -> dict:
        lanes = []
        if self.model is not None:
            lanes = [{"name": L.name,
                      "flow": describe_vector(L.flow),
                      "flow_source": L.flow_source,
                      "judged": L.judge,
                      "unique_ids": self._lane_totals.get(L.name, 0)}
                     for L in self.model.lanes]
        out = {"mode": self.mode, "enabled": self.enabled, "lanes": lanes,
               "divider": bool(self.model and self.model.divider
                               and self.model.divider.valid),
               "wrong_way_tracks": self._wrong_way_total,
               "wrong_lane_tracks": self._wrong_lane_total,
               "wrong_way_ids": sorted(self._wrong_way_ids),
               "wrong_lane_ids": sorted(self._wrong_lane_ids),
               "judged": self.evaluated, "off_lane": self.skipped,
               # Why verdicts were withheld. "nothing was flagged" is only
               # good news if it is not actually "nothing was measurable".
               "no_verdict_reasons": dict(sorted(self._reasons.items(),
                                                 key=lambda kv: -kv[1])),
               "notes": list(self.notes)}
        if self.verification is not None:
            out["verified"] = [
                {"lane": r["lane"], "verdict": r["verdict"], "tracks": r["n"],
                 "alignment": round(r["alignment"], 3),
                 "observed": describe_vector(r["observed"])}
                for r in self.verification.rows]
        if self.suggestion is not None:
            out["calibration"] = {"tracks_used": self.suggestion.tracks_used,
                                  "groups": list(self.suggestion.groups),
                                  "two_way": self.suggestion.two_way}
        return out

    def as_config(self) -> dict:
        """The geometry actually in use, as a pasteable config block."""
        if self.model is None:
            return {}
        return self.model.as_config(self.width, self.height)


register("lanes", LaneAnalyzer)
