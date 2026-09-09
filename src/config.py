"""Loads config.yaml into a dict with defaults. No other module reads YAML directly."""

import os

try:
    import yaml
except ImportError:
    yaml = None

DEFAULTS = {
    # classes: null = every COCO class ("anything on road"). Still accepts the
    # old [2,3,5,7], a group name like "vehicle"/"on_road", or class names.
    "model": {"name": "yolov8n.pt", "conf": 0.3, "iou": 0.5, "imgsz": 640,
              "device": "cpu", "classes": None},
    "tracker": {"name": "bytetrack.yaml", "persist": True, "track_buffer": 30},
    "video": {"source": "samples/input.mp4", "target": "outputs/annotated.mp4",
              "csv": "outputs/tracks.csv", "summary": "outputs/summary.txt",
              "write_video": True},
    # source: how to read the input. Only used for live sources.
    "source": {"reconnect": True, "max_retries": 0, "backpressure": None,
               "queue_size": None},
    # analyse_fps decouples analysis rate from source rate. null = every frame.
    # frame_skip is the deprecated integer form and is translated on load.
    "processing": {"max_frames": -1, "frame_skip": 1, "analyse_fps": None,
                   "evict_every": 150},
    "counting_line": {"enabled": True, "y_ratio": 0.6, "color": [0, 255, 255]},
    "camera": {"name": "demo", "line_a_ratio": 0.35, "line_b_ratio": 0.65,
               "label_a": "bottom (entry)", "label_b": "top (exit)",
               "color_a": [255, 0, 0], "color_b": [0, 255, 255]},
    "lanes": [],
    # auto  = measure the lanes from motion during a warmup window
    # explicit = use `lanes:` below, and CHECK them against motion
    # off   = no lane or wrong-way analysis
    "lanes_mode": "auto",
    "lanes_units": "auto",       # ratio | pixel | auto (per-lane override too)
    "divider": None,             # {"points": [[x, y], [x, y]]} - the road divider
    # The wrong-side rule. Every threshold here is depth-scaled or windowed;
    # see analysis/lane_model.py for what each one is measured in and why the
    # previous absolute values misfired.
    "lanes_rules": {
        "warmup_frames": 300, "on_mismatch": "suppress",
        "history": 8, "min_travel_px": 6.0, "min_travel_rel": 0.15,
        "align_cos": 0.5, "edge_px": 2,
        "max_gap_frames": 5, "max_step_rel": 1.5,
        "divider_margin_rel": 0.35, "divider_margin_min_px": 6.0,
        "lane_margin_rel": 0.15, "lane_margin_min_px": 4.0,
        "confirm_window": 8, "confirm_count": 5,
        "survey_min_samples": 6, "survey_min_travel_px": 25.0,
        "survey_min_travel_rel": 0.5,
        "opposing_min_tracks": 3, "opposing_min_frac": 0.12,
    },
    # How the analyses are scheduled. Analyses are independent by contract;
    # `parallel` only moves the compute phase of analyzers that declare
    # themselves concurrent onto worker threads. See analysis/stage.py.
    "analysis": {"parallel": False, "workers": None},
    # Use cases, in run order. Add "congestion" to enable it - no code change.
    "analyzers": ["counting", "lanes", "anpr"],
    "storage": {"backend": "sqlite", "path": "outputs/traffic.db",
                "batch_rows": 500, "commit_interval": 2.0, "csv_export": True},
    # Every analysed frame is stored raw AND annotated.
    #   format: jpeg     one file per frame per variant. Simple, and what a
    #                    finite clip wants (~614 MB for the 812-frame sample).
    #           segments rolling H.264. Same every-frame guarantee at ~5% of
    #                    the bytes - measured, 1080p traffic encodes to ~378 KB
    #                    per JPEG at q85, i.e. ~2 TB/day for raw+annotated at
    #                    30 fps, against ~130 GB/day as segments. Switch to
    #                    segments before pointing this at a permanent stream.
    "frames": {"enabled": True, "format": "jpeg", "dir": "outputs/frames",
               "quality": 85, "encode_workers": 3, "segment_seconds": 300,
               "retention": {"max_age_hours": 0, "max_disk_gb": 0,
                             "check_interval": 60}},
    "objects": {"save_crops": True, "crop_min_conf": 0.5, "crop_dir": "outputs/crops"},
    "outputs": {"save_crops": True, "crop_min_conf": 0.5},   # legacy aliases
    "attributes": {"enabled": ["type"]},
    "congestion": {"busy_count": 6, "heavy_count": 12, "jam_occupancy": 0.28,
                   "slow_px_per_frame": 2.0, "min_frames": 15, "draw": True},
    "plate": {"det_conf": 0.3, "min_conf": 0.5, "det_imgsz": 480,
              "ocr_backend": "paddle_anpr", "ocr_model": "",
              "format_correction": True, "join_rows": True,
              "read_every": 3, "max_reads": 8,
              "vote": True, "vote_min_conf": 0.8},
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
    return _reconcile(cfg)


def _reconcile(cfg: dict) -> dict:
    """Bridge legacy keys so old configs keep working.

    attributes.enabled containing "plate" is the old way to turn ANPR on; it
    now maps onto the analyzers list, so neither spelling silently does nothing.
    """
    analyzers = list(cfg.get("analyzers") or [])
    enabled_attrs = [str(a).lower() for a in (cfg.get("attributes", {}).get("enabled") or [])]
    if "plate" in enabled_attrs and "anpr" not in analyzers:
        analyzers.append("anpr")
    if "plate" not in enabled_attrs and "anpr" in analyzers:
        # Being in `analyzers` is intent enough; keep attributes consistent.
        cfg.setdefault("attributes", {}).setdefault("enabled", []).append("plate")
    cfg["analyzers"] = analyzers
    # objects.* is the new home for crop settings; honour outputs.* if set.
    legacy = cfg.get("outputs", {}) or {}
    obj = cfg.setdefault("objects", {})
    if "save_crops" in legacy:
        obj.setdefault("save_crops", legacy["save_crops"])
    if "crop_min_conf" in legacy:
        obj.setdefault("crop_min_conf", legacy["crop_min_conf"])
    return cfg
