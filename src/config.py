"""Loads config.yaml into a dict with defaults. No other module reads YAML directly."""

import os

try:
    import yaml
except ImportError:
    yaml = None

DEFAULTS = {
    "model": {"name": "yolov8n.pt", "conf": 0.3, "iou": 0.5, "imgsz": 640,
              "device": "cpu", "classes": [2, 3, 5, 7]},
    "tracker": {"name": "bytetrack.yaml", "persist": True},
    "video": {"source": "samples/input.mp4", "target": "outputs/annotated.mp4",
              "csv": "outputs/tracks.csv", "summary": "outputs/summary.txt"},
    "processing": {"max_frames": -1, "frame_skip": 1},
    "counting_line": {"enabled": True, "y_ratio": 0.6, "color": [0, 255, 255]},
    "camera": {"name": "demo", "line_a_ratio": 0.35, "line_b_ratio": 0.65,
               "label_a": "bottom (entry)", "label_b": "top (exit)",
               "color_a": [255, 0, 0], "color_b": [0, 255, 255]},
    "lanes": [],
    "lanes_mode": "auto",  # explicit = use lanes list; auto = split halves guess; off = no lanes
    "outputs": {"save_crops": True, "crop_min_conf": 0.5, "save_raw_frames": False,
                "save_frame_every": 30, "frame_dir": "outputs/frames"},
    "attributes": {"enabled": ["type"]},
    "plate": {"det_conf": 0.3, "min_conf": 0.5, "det_imgsz": 480},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str = "config.yaml") -> dict:
    cfg = DEFAULTS
    if os.path.exists(path):
        if yaml is None:
            print("[WARN] pyyaml not installed, using defaults. pip install pyyaml")
        else:
            with open(path, encoding="utf-8") as f:
                cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    return cfg
