"""Thin entry point. Global settings live in config.yaml, per-camera settings
in cameras/<id>.yaml, and CLI flags override both.

Run:
  python main.py --camera demo
  python main.py --camera junction_7 --analyse-fps 5
  python main.py --source rtsp://cam/stream --camera junction_7
  python main.py --camera demo --enable congestion --disable anpr
  python main.py --camera person_test --behaviour behaviour/person_test.yaml
"""

import argparse
import copy

from src.config import apply_behaviour, load_for_camera
from src.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Traffic detection (YOLO + ByteTrack)")
    p.add_argument("--config", default="config.yaml", help="fleet-wide defaults")
    p.add_argument("--camera", default=None,
                   help="camera id -> cameras/<id>.yaml (one file per CCTV). "
                        "The camera normally declares its own source.")
    p.add_argument("--source", default=None,
                   help="override the camera's source: video file, rtsp:// or "
                        "http:// URL, or a webcam index like 0")
    p.add_argument("--recorded-at", default=None,
                   help="when the FOOTAGE starts: unix epoch or ISO-8601 "
                        "(2026-09-12T08:30:00). File sources only - without it "
                        "this run's timestamps are clip-relative and cannot be "
                        "ordered against another camera. Live sources are "
                        "already on wall-clock and ignore it.")
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
    p.add_argument("--enable", default=None,
                   help="comma list of analyses to switch on for this run")
    p.add_argument("--behaviour", default=None, metavar="FILE",
                   help="run the human behaviour analysis with zones from FILE "
                        "(behaviour/<name>.yaml). Not a camera setting: this "
                        "flag is the only way to switch it on")
    p.add_argument("--disable", default=None,
                   help="comma list of analyses or attributes to switch off")
    p.add_argument("--parallel", action="store_true",
                   help="run concurrent-safe compute phases on worker threads")
    p.add_argument("--backpressure", default=None, choices=["drop_oldest", "buffer_all"],
                   help="live: drop stale frames (flat latency) or keep all (growing lag)")
    p.add_argument("--frames-format", default=None, choices=["jpeg", "segments"])
    p.add_argument("--no-frames", action="store_true", help="do not store frame images")
    p.add_argument("--max-disk-gb", type=float, default=None, help="frame retention cap")
    p.add_argument("--max-age-hours", type=float, default=None, help="frame retention age")
    p.add_argument("--no-line", action="store_true", help="hide the counting zones")
    p.add_argument("--line-a", type=float, default=None, help="upper zone ratio, e.g. 0.35")
    p.add_argument("--line-b", type=float, default=None, help="lower zone ratio, e.g. 0.65")
    p.add_argument("--no-reid", action="store_true",
                   help="disable plate-keyed vehicle re-identification "
                        "(a re-entering vehicle then gets a new id)")
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


def _names(spec: str) -> list:
    return [n.strip() for n in (spec or "").split(",") if n.strip()]


def _toggle(cfg: dict, names: list, on: bool):
    """Switch an analysis or an attribute enricher on/off for this run.

    One flag covers both stages because from the operator's side "turn plates
    off for this camera" is one intent, and which stage `plate` lives in is an
    implementation detail they should not have to know.
    """
    analyses = cfg.setdefault("analyses", {})
    attributes = cfg.setdefault("perception", {}).setdefault("attributes", {})
    for name in names:
        if name in analyses:
            analyses[name]["enabled"] = on
        elif name in attributes:
            attributes[name]["enabled"] = on
        else:
            known = ", ".join(sorted(set(analyses) | set(attributes)))
            print(f"[main] unknown analysis/attribute {name!r}; known: {known}")


def main():
    args = parse_args()
    # Camera first, CLI second: a flag typed at the prompt must beat the file.
    # The old order applied the camera file AFTER the flags, so --source was
    # silently discarded whenever --camera was also given.
    cfg = copy.deepcopy(load_for_camera(args.camera, args.config))

    model = cfg["perception"]["model"]
    if args.source: cfg["video"]["source"] = args.source
    if args.recorded_at: cfg["video"]["recorded_at"] = args.recorded_at
    if args.target: cfg["video"]["target"] = args.target
    if args.csv: cfg["video"]["csv"] = args.csv
    if args.db: cfg.setdefault("storage", {})["path"] = args.db
    if args.model: model["name"] = args.model
    if args.conf is not None: model["conf"] = args.conf
    if args.iou is not None: model["iou"] = args.iou
    if args.imgsz is not None: model["imgsz"] = args.imgsz
    if args.device: model["device"] = args.device
    if args.classes is not None: model["classes"] = _classes(args.classes)
    if args.max_frames is not None: cfg["processing"]["max_frames"] = args.max_frames
    if args.analyse_fps is not None: cfg["processing"]["analyse_fps"] = args.analyse_fps
    if args.parallel: cfg.setdefault("analysis", {})["parallel"] = True
    if args.behaviour: apply_behaviour(cfg, args.behaviour)
    if args.enable: _toggle(cfg, _names(args.enable), True)
    if args.disable: _toggle(cfg, _names(args.disable), False)
    if args.backpressure: cfg.setdefault("source", {})["backpressure"] = args.backpressure
    if args.frames_format: cfg.setdefault("frames", {})["format"] = args.frames_format
    if args.no_frames: cfg.setdefault("frames", {})["enabled"] = False
    if args.max_disk_gb is not None:
        cfg["frames"].setdefault("retention", {})["max_disk_gb"] = args.max_disk_gb
    if args.max_age_hours is not None:
        cfg["frames"].setdefault("retention", {})["max_age_hours"] = args.max_age_hours
    if args.no_reid: cfg.setdefault("reid", {})["enabled"] = False

    plate = cfg["perception"]["attributes"].setdefault("plate", {})
    if args.ocr_backend: plate["ocr_backend"] = args.ocr_backend
    if args.ocr_model: plate["ocr_model"] = args.ocr_model

    zones = cfg["analyses"].setdefault("counting", {}).setdefault("zones", {})
    if args.no_line: zones["enabled"] = False
    if args.line_a is not None: zones["line_a_ratio"] = args.line_a
    if args.line_b is not None: zones["line_b_ratio"] = args.line_b
    if args.lanes: cfg["analyses"].setdefault("lanes", {})["mode"] = args.lanes

    run_pipeline(cfg)
    print("Next: python report.py | python query.py --list")


if __name__ == "__main__":
    main()
