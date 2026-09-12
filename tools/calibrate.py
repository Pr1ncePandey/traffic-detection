"""Calibrate a camera from motion: lane geometry, flow direction, zone lines.

Usage:
  python tools/calibrate.py                                  # config.yaml source
  python tools/calibrate.py --source other.mp4 --max-frames 600
  python tools/calibrate.py --suggest-lanes --out cameras/new.yaml
  python tools/calibrate.py --verify --camera demo           # check, don't guess

How it works:
  1. Runs the same YOLO + ByteTrack the pipeline runs.
  2. Records each track's GROUND POINT (bottom-centre of the box, where the
     vehicle meets the road) - not the box centroid, which drifts upward for a
     vehicle whose box is clipped by the bottom edge.
  3. Groups tracks into at most two opposing flows and, if there really are
     two, fits the divider between them per y-band.

This shares src/analysis/lane_calibration.py with the runtime, so what this
tool prints and what `analyses.lanes.mode: auto` does cannot drift apart.

--verify is the one to reach for when wrong-way alerts look wrong: it takes
the lanes you already have and reports, per lane, the direction traffic
actually flows in it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from src.analysis.geometry import describe_vector, to_ratios
from src.analysis.lane_calibration import MotionSurvey
from src.analysis.lane_model import LaneModel
from src.config import load_for_camera
from src.detectors.classes import VEHICLE
from src.detectors.yolo import Detector


def lane_block(cfg: dict) -> dict:
    """The lanes analysis config, at its one canonical path."""
    return (cfg or {}).get("analyses", {}).get("lanes", {}) or {}


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure lane geometry, flow direction and zone lines")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--out", default=None, help="write cameras/<name>.yaml")
    p.add_argument("--suggest-lanes", action="store_true",
                   help="propose lane polygons + flow vectors from motion")
    p.add_argument("--verify", action="store_true",
                   help="check the CONFIGURED lanes against observed motion")
    p.add_argument("--camera", default=None,
                   help="with --verify: check cameras/<name>.yaml")
    return p.parse_args()


def survey_source(cfg, source, max_frames):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = Detector(cfg["perception"]["model"])
    rules = lane_block(cfg).get("rules", {}) or {}
    survey = MotionSurvey(
        width, height,
        min_samples=int(rules.get("survey_min_samples", 6)),
        min_travel_px=float(rules.get("survey_min_travel_px", 25.0)),
        min_travel_rel=float(rules.get("survey_min_travel_rel", 0.5)),
        opposing_min_tracks=int(rules.get("opposing_min_tracks", 3)),
        opposing_min_frac=float(rules.get("opposing_min_frac", 0.12)),
        align_cos=float(rules.get("align_cos", 0.5)))
    seen = 0
    while seen < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        seen += 1
        survey.observe_frame()
        for det in detector.detect(frame, cfg["perception"]["tracker"]):
            # Vehicles only: a pedestrian crossing the carriageway must not
            # vote on which way the road flows.
            if det.track_id is not None and det.group == VEHICLE:
                survey.observe(det.track_id, det.bbox, det.cls_name)
    cap.release()
    print(f"[calibrate] {seen} frames, {len(survey.tracks)} tracks, "
          f"{len(survey.moving())} with clear motion")
    return survey, width, height, seen


def zone_snippet(survey, width, height):
    """The A/B counting lines, from the dominant flow."""
    trajs = survey.moving()
    ups = sum(1 for t in trajs if t.heading[1] < 0)
    downs = sum(1 for t in trajs if t.heading[1] > 0)
    if ups >= downs:
        label_a, label_b = "bottom (entry)", "top (exit)"
        flow = "bottom -> top"
    else:
        label_a, label_b = "top (entry)", "bottom (exit)"
        flow = "top -> bottom"
    print(f"[calibrate] dominant vertical flow: {flow} "
          f"(up={ups} down={downs} of {len(trajs)})")
    return (f'camera:\n  name: "new_camera"\n\n'
            f'video:\n  source: "CHANGE_ME"\n\n'
            f'analyses:\n  counting:\n    zones:\n'
            f'      line_a_ratio: 0.35\n      line_b_ratio: 0.65\n'
            f'      label_a: "{label_a}"\n      label_b: "{label_b}"\n')


def lanes_snippet(suggestion, width, height):
    print(suggestion.report())
    if not suggestion.lanes:
        return ""
    # Continues the `analyses:` mapping zone_snippet opened, so the two
    # halves concatenate into one valid camera file.
    lines = ['\n  lanes:', '    mode: "explicit"']
    if suggestion.divider is not None:
        lines += ["    divider:",
                  f"      points: {to_ratios(suggestion.divider.as_points(), width, height, 4)}"]
    lines.append("    lanes:")
    for lane in suggestion.lanes:
        lines.append(f'      - name: "{lane["name"]}"')
        lines.append(f"        polygon: {to_ratios(lane['polygon'], width, height, 4)}")
        lines.append(f"        flow: [{lane['flow'][0]:.4f}, {lane['flow'][1]:.4f}]")
        lines.append("        allowed: []")
    if not suggestion.two_way:
        lines.append("      # ONE carriageway: only one direction of travel was")
        lines.append("      # observed, so no divider was invented. If a second")
        lines.append("      # road IS in view but was empty, add it by hand")
        lines.append("      # with tools/draw_lanes.py, with the opposite flow.")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    # The SAME loader the pipeline uses, so --verify cannot check a config
    # that differs from the one that will actually run.
    cfg = load_for_camera(args.camera, args.config)
    source = args.source or cfg["video"]["source"]
    print(f"[calibrate] source {source}, up to {args.max_frames} frames")
    survey, width, height, seen = survey_source(cfg, source, args.max_frames)

    if not survey.moving():
        print("[calibrate] no tracks with clear motion. Try --max-frames 1200, "
              "or check that the detector is finding vehicles at all.")
        return

    if args.verify:
        conf = lane_block(cfg)
        lanes_cfg = conf.get("lanes") or []
        if not lanes_cfg:
            print("[calibrate] --verify needs configured lanes; none found. "
                  "Run with --suggest-lanes instead.")
            return
        model = LaneModel.from_config(lanes_cfg, width, height,
                                      divider_cfg=conf.get("divider"),
                                      rules=conf.get("rules", {}))
        print(f"[calibrate] {model.describe()}")
        verification = survey.verify(model)
        print(verification.report())
        if verification.ok:
            print("[calibrate] configured lanes agree with observed motion.")
        else:
            print("[calibrate] FIX THE GEOMETRY before trusting wrong-way "
                  "alerts, or set analyses.lanes.rules.on_mismatch: flip to "
                  "trust the "
                  "measurement at runtime.")
        return

    snippet = zone_snippet(survey, width, height)
    if args.suggest_lanes:
        snippet += lanes_snippet(survey.suggest(), width, height)
    else:
        head = survey._mean_heading(survey.moving())
        print(f"[calibrate] mean flow {describe_vector(head)}. "
              f"Add --suggest-lanes for lane polygons and a divider.")
    print("--- suggested cameras/<name>.yaml ---\n" + snippet)
    print("Tip: lines at 0.35/0.65 cut most lanes. Move them if a lane sits "
          "outside the band.")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"# Measured by tools/calibrate.py from {source} "
                    f"({seen} frames) - confirm once per CCTV.\n" + snippet)
        name = os.path.splitext(os.path.basename(args.out))[0]
        print(f"Saved to {args.out}. Run: python main.py --camera {name}")


if __name__ == "__main__":
    main()
