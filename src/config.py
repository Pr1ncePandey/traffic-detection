"""Loads the global config plus one camera's overrides into a single dict.

TWO LAYERS, ONE MERGE

    config.yaml            fleet-wide defaults. Applies to every stream.
    cameras/<id>.yaml      that camera's overrides, deep-merged on top.

`load_for_camera()` is the only way to build a runnable config, and it is the
only merge implementation. That matters: main.py and tools/calibrate.py each
used to carry their OWN partial overlay with a hardcoded key whitelist, and the
two lists had drifted - calibrate.py omitted `counting_line`, so `--verify`
validated a different config than the one that actually ran.

WHAT A CAMERA MAY OVERRIDE

Anything, except the keys in LOCKED_KEYS. The old whitelist had the failure
mode backwards: it named the seven keys a camera COULD set, so a camera could
not change its own detector threshold, analysis rate, or which analyses ran -
the knobs that actually differ between a highway camera and a junction. The
locked list instead names the handful of keys that must stay fleet-wide,
because at 50 cameras on a shared database a per-camera `storage.path` writes
to a different DB while looking like it worked, and a per-camera
`perception.model.name` makes cross-camera counts incomparable with nothing
saying so.

Keys in NOTED_KEYS are legitimately per-camera but are printed when overridden,
so drift in detection sensitivity across a fleet is visible in the logs.
"""

import os

try:
    import yaml
except ImportError:
    yaml = None

# Fleet-wide only. A camera file setting one of these is warned about and
# ignored - see _strip_locked(). Dotted paths; a path names a whole subtree.
LOCKED_KEYS = (
    "storage",                    # shared DB: one path for the whole fleet
    "frames.format",              # jpeg vs segments is a disk-budget decision
    "frames.retention",           # ...and so is the retention window
    "perception.model.name",      # one model, or counts are incomparable
    "perception.model.device",
    "analysis",                   # scheduling/threading is a host property
)

# Overridable, but say so out loud.
NOTED_KEYS = (
    "perception.model.conf",
    "perception.model.iou",
    "perception.model.imgsz",
    "perception.model.classes",
    "processing.analyse_fps",
)

# Paths templated per camera so 50 processes cannot overwrite each other.
# storage.path is deliberately NOT here: the database is shared by design.
_PATH_KEYS = ("video.target", "video.csv", "video.summary",
              "frames.dir", "objects.crop_dir")

DEFAULTS = {
    # Identity. `id` is the cameras/<id>.yaml stem and the {camera_id} used in
    # output paths; `name` is what a human reads in the report.
    "camera": {"id": "demo", "name": "demo"},

    # A camera declares its OWN source, so geometry and footage cannot be
    # mismatched - pairing one camera's lanes with another's video produces
    # plausible-looking but wrong lane flags, which is worse than an error.
    "video": {"source": "samples/input.mp4",
              "target": "outputs/{camera_id}/annotated.mp4",
              "csv": "outputs/{camera_id}/tracks.csv",
              "summary": "outputs/{camera_id}/summary.txt",
              "write_video": True},

    # Only used for live sources (rtsp:// / http:// / webcam index).
    "source": {"reconnect": True, "max_retries": 0, "backpressure": None,
               "queue_size": None},

    # analyse_fps decouples analysis rate from source rate. null = every frame.
    "processing": {"max_frames": -1, "analyse_fps": None, "evict_every": 150},

    # THE COMMON STAGE: detect -> track -> attributes. Everything here runs
    # once per frame and produces the objects every analysis then reads.
    "perception": {
        "model": {"name": "yolov8n.pt", "conf": 0.3, "iou": 0.5, "imgsz": 640,
                  "device": "cpu", "classes": None},
        "tracker": {"name": "bytetrack.yaml", "persist": True,
                    "track_buffer": 30},
        # Enrichers, in run order. Each adds attributes to tracked objects
        # BEFORE any analysis sees them, which is what lets an analysis filter
        # on an attribute. Disable one per camera with enabled: false.
        "attributes": {
            "color": {"enabled": True},
            "plate": {"enabled": True,
                      "det_conf": 0.3, "min_conf": 0.5, "det_imgsz": 480,
                      "ocr_backend": "paddle_anpr", "ocr_model": "",
                      "format_correction": True, "join_rows": True,
                      "read_every": 3, "max_reads": 8,
                      "vote": True, "vote_min_conf": 0.8},
        },
    },

    # INDEPENDENT ANALYSES over the objects perception produced. Dict order is
    # run order for the serial phases; each has its own `enabled` so a camera
    # can switch one off without touching the others.
    "analyses": {
        "counting": {
            "enabled": True,
            "vehicles_only": True,
            "zones": {"enabled": True,
                      "line_a_ratio": 0.35, "line_b_ratio": 0.65,
                      "label_a": "bottom (entry)", "label_b": "top (exit)",
                      "color_a": [255, 0, 0], "color_b": [0, 255, 255]},
        },
        "lanes": {
            "enabled": True,
            # auto     = measure the lanes from motion during a warmup window
            # explicit = use `lanes:` below, and CHECK it against motion
            # off      = no lane or wrong-way analysis
            "mode": "auto",
            "units": "auto",          # ratio | pixel | auto (per-lane too)
            "lanes": [],
            "divider": None,          # {"points": [[x, y], [x, y]]}
            # Every threshold here is depth-scaled or windowed; see
            # analysis/lane_model.py for what each is measured in.
            "rules": {
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
        },
        "congestion": {
            "enabled": False,
            "busy_count": 6, "heavy_count": 12, "jam_occupancy": 0.28,
            "slow_px_per_frame": 2.0, "min_frames": 15, "draw": True,
        },
    },

    # How the stages are SCHEDULED, not what they do. Host property, locked.
    "analysis": {"parallel": False, "workers": None},

    "storage": {"backend": "sqlite", "path": "outputs/traffic.db",
                "batch_rows": 500, "commit_interval": 2.0, "csv_export": True},

    # Every analysed frame is stored raw AND annotated. jpeg is right for a
    # finite clip; switch to segments before pointing this at a live stream
    # (measured ~2 TB/day vs ~130 GB/day at 1080p30).
    "frames": {"enabled": True, "format": "jpeg",
               "dir": "outputs/{camera_id}/frames",
               "quality": 85, "encode_workers": 3, "segment_seconds": 300,
               "retention": {"max_age_hours": 0, "max_disk_gb": 0,
                             "check_interval": 60}},

    "objects": {"save_crops": True, "crop_min_conf": 0.5,
                "crop_dir": "outputs/{camera_id}/crops"},

    # Durable identity. ByteTrack gives a re-entering vehicle a NEW track id;
    # this keys on the plate so it gets its original id back.
    "reid": {"enabled": True, "min_conf": 0.7, "require_format": True,
             "fuzzy_distance": 0, "cache_size": 4096},
}


def _merge(base: dict, over: dict) -> dict:
    """Recursive overlay. Absent means absent: a key the override does not
    mention keeps the base value rather than reverting to a default."""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _walk(cfg: dict, prefix: str = ""):
    """Yield (dotted_path, value) for every key, descending only into dicts."""
    for key, value in (cfg or {}).items():
        path = f"{prefix}.{key}" if prefix else str(key)
        yield path, value
        if isinstance(value, dict):
            yield from _walk(value, path)


def _pop(cfg: dict, dotted: str):
    """Remove a dotted path if present. Returns True when something went."""
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return False
    return node.pop(parts[-1], _MISSING) is not _MISSING


_MISSING = object()


def _strip_locked(raw: dict, camera_id: str) -> dict:
    """Drop fleet-wide keys a camera file must not set, loudly.

    Silence here would be the worst option: a camera that believes it set its
    own storage.path and did not would write its rows somewhere unexpected and
    look fine doing it.
    """
    cleaned = dict(raw or {})
    for locked in LOCKED_KEYS:
        if _pop(cleaned, locked):
            print(f"[config] cameras/{camera_id}.yaml sets {locked!r}, which is "
                  f"fleet-wide and cannot be overridden per camera; "
                  f"keeping the global value")
    return cleaned


def _known_paths() -> set:
    """Dotted paths DEFAULTS declares.

    Only dict values are descended into, so free-form subtrees - lane
    polygons, divider points, a model class list - are not policed.
    """
    return {path for path, _ in _walk(DEFAULTS)}


def _warn_unknown(raw: dict, camera_id: str):
    """A typo in a camera file used to vanish silently. Now it says so."""
    known = _known_paths()
    for path, _value in _walk(raw or {}):
        head = path.split(".")[0]
        if head not in DEFAULTS:
            print(f"[config] cameras/{camera_id}.yaml: unknown key {path!r} "
                  f"(no such setting); ignoring")
            continue
        if path in known:
            continue
        # Inside a free-form subtree (a lane entry, divider points, ...)?
        parent = path.rsplit(".", 1)[0]
        if parent in known and not isinstance(_lookup(DEFAULTS, parent), dict):
            continue
        if any(p in known and not isinstance(_lookup(DEFAULTS, p), dict)
               for p in _ancestors(path)):
            continue
        print(f"[config] cameras/{camera_id}.yaml: unknown key {path!r}; ignoring")


def _ancestors(dotted: str):
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        yield ".".join(parts[:i])


def _lookup(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _note_overrides(raw: dict, camera_id: str):
    touched = [p for p, _ in _walk(raw or {}) if p in NOTED_KEYS]
    if touched:
        print(f"[config] cameras/{camera_id}.yaml overrides {', '.join(touched)} "
              f"- detection behaviour differs from the fleet default")


def _expand_paths(cfg: dict, camera_id: str) -> dict:
    """Substitute {camera_id} in output paths.

    Without this every camera writes outputs/frames and outputs/tracks.csv,
    and 50 processes silently overwrite one another.
    """
    for dotted in _PATH_KEYS:
        value = _lookup(cfg, dotted)
        if isinstance(value, str) and "{camera_id}" in value:
            parts = dotted.split(".")
            node = cfg
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value.format(camera_id=camera_id)
    return cfg


def _read_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    if yaml is None:
        print("[config] pyyaml not installed, using defaults. pip install pyyaml")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[config] could not read {path} ({e}); ignoring it")
        return {}


def load(path: str = "config.yaml") -> dict:
    """The fleet-wide config: DEFAULTS with config.yaml overlaid."""
    return _expand_paths(_merge(DEFAULTS, _read_yaml(path)),
                         _lookup(DEFAULTS, "camera.id") or "default")


def load_for_camera(camera_id: str | None = None,
                    path: str = "config.yaml") -> dict:
    """The runnable config for one camera: DEFAULTS <- config.yaml <- camera.

    The single merge implementation. Pass camera_id=None for the global config
    alone, which is what a bare `python main.py` and the calibration tools use
    when no camera is named.
    """
    cfg = _merge(DEFAULTS, _read_yaml(path))
    if camera_id:
        cam_path = os.path.join("cameras", f"{camera_id}.yaml")
        if not os.path.exists(cam_path):
            print(f"[config] cameras/{camera_id}.yaml not found; "
                  f"running on config.yaml alone")
        else:
            raw = _read_yaml(cam_path)
            _warn_unknown(raw, camera_id)
            _note_overrides(raw, camera_id)
            cfg = _merge(cfg, _strip_locked(raw, camera_id))
            applied = sorted({p.split(".")[0] for p, _ in _walk(raw)})
            print(f"[config] {cam_path}: applied {', '.join(applied) or 'nothing'}")
        cfg.setdefault("camera", {})["id"] = camera_id
        cfg["camera"].setdefault("name", camera_id)
    return _expand_paths(cfg, cfg.get("camera", {}).get("id", "default"))
