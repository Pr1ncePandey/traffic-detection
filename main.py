"""Thin entry point. All settings live in config.yaml; CLI flags override them.

Run:
  python main.py
  python main.py --source rtsp://cam/stream --analyse-fps 5
  python main.py --source 0                       # webcam
  python main.py --analyzers counting,lanes,anpr,congestion
"""

import argparse
import copy
import os

from src.config import load
from src.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Traffic detection (YOLO + ByteTrack)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None,
                   help="video file, rtsp:// or http:// URL, or a webcam index like 0")
    p.add_argument("--target", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--db", default=None, help="SQLite path (default outputs/traffic.db)")
    p.add_argument("--model", default=None, help="e.g. yolov8s.pt or custom.pt")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu or cuda / 0")
    p.add_argument("--classes", default=None,
                   help='"all", a group ("vehicle"/"on_road"), or a comma list of names/ids')
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--analyse-fps", type=float, default=None,
                   help="frames per second to analyse, independent of source rate")
    p.add_argument("--frame-skip", type=int, default=None, help="deprecated; use --analyse-fps")
    p.add_argument("--analyzers", default=None,
                   help="comma list, in run order: counting,lanes,anpr,congestion")
    p.add_argument("--backpressure", default=None, choices=["drop_oldest", "buffer_all"],
                   help="live: drop stale frames (flat latency) or keep all (growing lag)")
    p.add_argument("--frames-format", default=None, choices=["jpeg", "segments"])
    p.add_argument("--no-frames", action="store_true", help="do not store frame images")
    p.add_argument("--max-disk-gb", type=float, default=None, help="frame retention cap")
    p.add_argument("--max-age-hours", type=float, default=None, help="frame retention age")
    p.add_argument("--no-line", action="store_true", help="hide the counting zones")
    p.add_argument("--line-a", type=float, default=None, help="upper zone ratio, e.g. 0.35")
    p.add_argument("--line-b", type=float, default=None, help="lower zone ratio, e.g. 0.65")
    p.add_argument("--camera", default=None,
                   help="camera name -> cameras/<name>.yaml (one file per CCTV)")
    p.add_argument("--ocr-backend", default=None,
                   choices=["rapidocr", "fast_plate", "paddle_anpr"])
    p.add_argument("--ocr-model", default=None, help="backend-specific OCR model name")
    p.add_argument("--lanes", default=None, choices=["explicit", "auto", "off"])
    return p.parse_args()


def _classes(spec: str):
    """--classes accepts a keyword, a group, or a comma list of names/ids."""
    text = spec.strip()
    if text.lower() in ("all", "none", ""):
        return None
    if "," not in text:
        return text.lower() if not text.isdigit() else [int(text)]
    return [int(x) if x.strip().isdigit() else x.strip() for x in text.split(",")]


# Keys a cameras/<name>.yaml may override. Everything geometric belongs here:
# the divider and the per-camera rule tuning are as much a property of one
# CCTV as the lane polygons are, and leaving them out of this list is how a
# camera file ends up half-applied.
_CAMERA_KEYS = ("camera", "lanes", "lanes_mode", "lanes_units", "divider",
                "lanes_rules", "counting_line")


def _apply_camera(cfg: dict, name: str):
    """Overlay cameras/<name>.yaml onto cfg, using ONLY the keys it declares.

    Read raw rather than through load(): load() fills in every default, so a
    camera file that simply does not mention `lanes_mode` used to overwrite
    the one in config.yaml with the default "auto". Absent must mean absent.
    """
    path = os.path.join("cameras", f"{name}.yaml")
    if not os.path.exists(path):
        print(f"[WARN] cameras/{name}.yaml not found, using config.yaml camera block")
        return
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] could not read cameras/{name}.yaml ({e}); "
              f"using config.yaml camera block")
        return
    for key in _CAMERA_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}   # per-key, so partial files work
        else:
            cfg[key] = value
    cfg.setdefault("camera", {})["name"] = name
    # Backpressure is a per-camera property: a stream wants drop_oldest.
    bp = (raw.get("source") or {}).get("backpressure")
    if bp:
        cfg.setdefault("source", {})["backpressure"] = bp
    applied = [k for k in _CAMERA_KEYS if k in raw]
    print(f"[camera] {path}: applied {', '.join(applied) or 'nothing'}")


def main():
    args = parse_args()
    cfg = copy.deepcopy(load(args.config))
    if args.source: cfg["video"]["source"] = args.source
    if args.target: cfg["video"]["target"] = args.target
    if args.csv: cfg["video"]["csv"] = args.csv
    if args.db: cfg.setdefault("storage", {})["path"] = args.db
    if args.model: cfg["model"]["name"] = args.model
    if args.conf is not None: cfg["model"]["conf"] = args.conf
    if args.iou is not None: cfg["model"]["iou"] = args.iou
    if args.imgsz is not None: cfg["model"]["imgsz"] = args.imgsz
    if args.device: cfg["model"]["device"] = args.device
    if args.classes is not None: cfg["model"]["classes"] = _classes(args.classes)
    if args.max_frames is not None: cfg["processing"]["max_frames"] = args.max_frames
    if args.analyse_fps is not None: cfg["processing"]["analyse_fps"] = args.analyse_fps
    if args.frame_skip is not None: cfg["processing"]["frame_skip"] = args.frame_skip
    if args.analyzers is not None:
        cfg["analyzers"] = [a.strip() for a in args.analyzers.split(",") if a.strip()]
    if args.backpressure: cfg.setdefault("source", {})["backpressure"] = args.backpressure
    if args.frames_format: cfg.setdefault("frames", {})["format"] = args.frames_format
    if args.no_frames: cfg.setdefault("frames", {})["enabled"] = False
    if args.max_disk_gb is not None:
        cfg.setdefault("frames", {}).setdefault("retention", {})["max_disk_gb"] = args.max_disk_gb
    if args.max_age_hours is not None:
        cfg.setdefault("frames", {}).setdefault("retention", {})["max_age_hours"] = args.max_age_hours
    if args.ocr_backend: cfg.setdefault("plate", {})["ocr_backend"] = args.ocr_backend
    if args.ocr_model: cfg.setdefault("plate", {})["ocr_model"] = args.ocr_model
    if args.no_line:
        cfg.setdefault("counting_line", {})["enabled"] = False
        cfg.setdefault("camera", {})["enabled"] = False
    if args.line_a is not None: cfg.setdefault("camera", {})["line_a_ratio"] = args.line_a
    if args.line_b is not None: cfg.setdefault("camera", {})["line_b_ratio"] = args.line_b
    if args.camera:
        _apply_camera(cfg, args.camera)
    if args.lanes: cfg["lanes_mode"] = args.lanes
    run_pipeline(cfg)
    print("Next: python report.py | python query.py --list")


if __name__ == "__main__":
    main()
