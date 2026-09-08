"""Thin entry point. All settings live in config.yaml; CLI flags override them.

Run:
  python main.py
  python main.py --conf 0.4 --max-frames 100 --no-line
"""

import argparse
import copy

from src.config import load
from src.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Traffic tracking prototype (YOLO + ByteTrack)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--model", default=None, help="e.g. yolov8s.pt or custom.pt")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu or cuda / 0")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--frame-skip", type=int, default=None)
    p.add_argument("--no-line", action="store_true", help="hide the counting zones")
    p.add_argument("--line-a", type=float, default=None, help="upper zone ratio, e.g. 0.35")
    p.add_argument("--line-b", type=float, default=None, help="lower zone ratio, e.g. 0.65")
    p.add_argument("--camera", default=None, help="camera name -> cameras/<name>.yaml (one file per CCTV)")
    p.add_argument("--lanes", default=None, choices=["explicit", "auto", "off"],
                   help="explicit = yaml polygons (saved per camera); auto = halves guess ignoring yaml (new clip); off = no lanes")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load(args.config)
    cfg = copy.deepcopy(cfg)
    if args.source: cfg["video"]["source"] = args.source
    if args.target: cfg["video"]["target"] = args.target
    if args.csv: cfg["video"]["csv"] = args.csv
    if args.model: cfg["model"]["name"] = args.model
    if args.conf is not None: cfg["model"]["conf"] = args.conf
    if args.iou is not None: cfg["model"]["iou"] = args.iou
    if args.imgsz is not None: cfg["model"]["imgsz"] = args.imgsz
    if args.device: cfg["model"]["device"] = args.device
    if args.max_frames is not None: cfg["processing"]["max_frames"] = args.max_frames
    if args.frame_skip is not None: cfg["processing"]["frame_skip"] = args.frame_skip
    if args.no_line:
        cfg.setdefault("counting_line", {})["enabled"] = False
        cfg.setdefault("camera", {})["enabled"] = False
    if args.line_a is not None:
        cfg.setdefault("camera", {})["line_a_ratio"] = args.line_a
    if args.line_b is not None:
        cfg.setdefault("camera", {})["line_b_ratio"] = args.line_b
    if args.camera:
        import os as _os
        from src.config import load as _load
        alt = _os.path.join("cameras", f"{args.camera}.yaml")
        if _os.path.exists(alt):
            cam_cfg = _load(alt)
            cfg["camera"] = cam_cfg.get("camera", cfg.get("camera", {}))
            cfg["camera"]["name"] = args.camera
            if cam_cfg.get("lanes"):
                cfg["lanes"] = cam_cfg["lanes"]
            if cam_cfg.get("lanes_mode"):
                cfg["lanes_mode"] = cam_cfg["lanes_mode"]
        else:
            print(f"[WARN] cameras/{args.camera}.yaml not found, using config.yaml camera block")
    if args.lanes:
        cfg["lanes_mode"] = args.lanes
    run_pipeline(cfg)
    print("Next: python report.py | python query.py --list")


if __name__ == "__main__":
    main()
