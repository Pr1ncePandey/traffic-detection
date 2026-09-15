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
    "server",                     # one listen address for the whole service;
                                  # a camera file must not be able to move it
    "retention",                  # one disk budget over one shared database
    "incidents.subscriptions",    # who gets told is an operator decision, not
                                  # a per-camera one. Per-camera FILTERING is
                                  # still available via a subscription's
                                  # `cameras` list, which is the right place
                                  # for it: routing stays in one file.
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
              "frames.dir", "objects.crop_dir",
              "perception.attributes.face.snapshot_dir",
              "analyses.heatmap.dir")

DEFAULTS = {
    # Identity. `id` is the cameras/<id>.yaml stem and the {camera_id} used in
    # output paths; `name` is what a human reads in the report.
    # `location` is where this camera physically is. Declared here in DEFAULTS
    # rather than only in the yaml because that is what makes it a KNOWN key:
    # without an entry, _warn_unknown() prints "no such setting; ignoring" and
    # the value is silently dropped. null/null means "location unknown", which
    # journeys report as an unmeasurable hop rather than a distance of zero.
    "camera": {"id": "demo", "name": "demo",
               "location": {"lat": None, "lon": None}},

    # A camera declares its OWN source, so geometry and footage cannot be
    # mismatched - pairing one camera's lanes with another's video produces
    # plausible-looking but wrong lane flags, which is worse than an error.
    #
    # recorded_at: when the FOOTAGE starts, as a unix epoch or an ISO-8601
    # string. Only meaningful for file sources, where frame timestamps are
    # seconds-from-clip-start and therefore not comparable across cameras until
    # anchored to something real. See src/timebase.py for why this cannot be
    # inferred. Live sources ignore it - they are already on wall-clock.
    "video": {"source": "samples/short/input.mp4",
              "target": "outputs/{camera_id}/annotated.mp4",
              "csv": "outputs/{camera_id}/tracks.csv",
              "summary": "outputs/{camera_id}/summary.txt",
              "write_video": True,
              "recorded_at": None},

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
                      "ocr_backend": "fast_plate", "ocr_model": "",
                      "format_correction": True, "join_rows": True,
                      "read_every": 3, "max_reads": 8,
                      "vote": True, "vote_min_conf": 0.8},
            "person": {"enabled": True,
                       "model": "models/person_attr/person_attr.onnx",
                       "min_height": 80, "min_width": 30,
                       "edge_margin": 4, "min_det_conf": 0.35,
                       "max_umbrella_cover": 0.25,
                       "read_every": 3, "max_reads": 8, "min_reads": 3,
                       "threads": 1, "thresholds": {"Glasses": 0.6},
                       "keys": ["facing", "sleeves", "lower", "bag", "hat",
                                "glasses"]},
            "garments": {"enabled": False,
                         "model": "models/clothes_seg/segformer_b2_clothes.onnx",
                         "min_height": 120, "min_width": 40,
                         "read_every": 6, "max_reads": 3, "min_reads": 2,
                         "threads": 4},
            "age_gender": {"enabled": False,
                           "min_height": 100, "min_width": 35,
                           "read_every": 6, "max_reads": 3, "min_reads": 2,
                           "threads": 4},
            # Face recognition against the people database (src/faces/).
            # OFF fleet-wide; a camera switches it on. `people` empty = search
            # for everyone enabled in the database, else only these names.
            "face": {"enabled": False,
                     "models_dir": "models/faces", "people": [],
                     "detect_every": 2, "detect_width": 1280,
                     "min_det_score": 0.8, "min_eye_px": 28,
                     "min_sharpness": 40, "face_top": 0.6,
                     "threshold": 0.45, "margin": 0.05, "min_votes": 2,
                     "max_reads": 6, "read_gap_s": 0.3, "recheck_s": 3.0,
                     "lost_below": 0.25, "crossing_overlap": 0.3,
                     "threads": 4, "batch": 8,
                     "max_pending": 16, "embed_async": "auto",
                     "reload_s": 10.0,
                     "snapshot_dir": "outputs/{camera_id}/faces"},
        },
    },

    # INDEPENDENT ANALYSES over the objects perception produced. Dict order is
    # run order for the serial phases; each has its own `enabled` so a camera
    # can switch one off without touching the others.
    "analyses": {
        # First on purpose: analyses draw in declaration order, so the heatmap
        # goes under the counting lines and lane overlays drawn after it.
        "heatmap": {
            "enabled": False,
            "groups": ["person", "vehicle"], "show": "all",
            "grid_w": 96, "blur": 1.5, "half_life_s": 0,
            "alpha": 0.45, "min_level": 0.08, "scale": "log", "colormap": "jet",
            "redraw_every": 5, "draw": True, "save": True,
            "background_every": 150, "dir": "outputs/{camera_id}/heatmap",
        },
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
        # Human behaviour (analysis/behaviour.py): named zones and the rules
        # that watch them - intrusion, loitering, crowding, running. No model:
        # it reads the same tracks every other analysis does. OFF, and NOT a
        # camera setting: a camera file's block is ignored (_strip_behaviour).
        # It runs only when asked, with a zones file kept outside cameras/:
        #   python main.py --camera X --behaviour behaviour/<name>.yaml
        # (apply_behaviour). The keys below are that file's. Polygons are ratios
        # or pixels like lanes; a zone with no polygon is the whole frame.
        # Per-zone rules: restricted, loitering_s, crowd_people (0/false = off).
        # See docs/human-behaviour.md for why each threshold is what it is.
        "behaviour": {
            "enabled": False,
            "groups": ["person"],
            "units": "auto",
            "zones": [],
            # Polygons where a detected "person" is ignored by every rule: a
            # sign pole or bollard the detector keeps mistaking for one.
            "exclude": [],
            "intrusion_s": 1.0,        # inside a restricted zone this long
            "exit_grace_s": 2.0,       # outside this long before "left"
            "crowd_hold_s": 5.0,       # crowd must hold before it is reported:
            "crowd_fraction": 0.8,     # ...in this share of that window's frames
            # Riders and passengers are not pedestrians. A two-wheeler covers
            # ~half its rider (feet inside + 30%); a passenger is almost wholly
            # inside a car/bus box (90%). 90% and not 30% for enclosed
            # vehicles: a pedestrian behind a parked car is 30-60% covered.
            "ignore_riders": True,
            "rider_overlap": 0.3, "riders_in": ["bicycle", "motorcycle"],
            "passenger_overlap": 0.9,
            "passengers_in": ["car", "bus", "truck", "train"],
            # Speed in BODY HEIGHTS PER SECOND, so it means the same near and
            # far from the camera. Walking is ~0.8, jogging ~1.6, running 2+.
            "running": {"enabled": True, "min_speed_hps": 1.6, "min_s": 1.0,
                        "window_s": 1.0, "min_height_px": 40,
                        "max_jump_hps": 6.0, "edge_px": 2},
            "exit_events": True,
            "draw": True,
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

    # Multi-camera journey reconstruction (offline; see src/journeys.py).
    #   max_gap_s     idle time that ends one trip and starts another. A car at
    #                 camera A in the morning and at A again at night made two
    #                 trips; it did not loop.
    #   max_speed_kmh implied-speed ceiling per hop, from haversine distance
    #                 over elapsed time. Above this the two sightings are an OCR
    #                 collision or a cloned plate, not a journey. Hops are
    #                 FLAGGED, never dropped - a bad plate merge should be
    #                 visible rather than invisible.
    "journey": {"max_gap_s": 1800, "max_speed_kmh": 150},

    # THE SERVICE (serve.py). Host property, so it is locked fleet-wide: a
    # camera file cannot move the listen address out from under the operator.
    #   host      127.0.0.1 and nothing else by default. There is NO
    #             application-level auth; /crops serves plate imagery, /live
    #             streams plate strings and /stream serves LIVE ROAD FOOTAGE,
    #             so binding wider publishes all three.
    #   push_hz   WebSocket rate, independent of analyse_fps. Nobody reads 30
    #             updates a second and a file replaying at 3x would flood it.
    #   video     the MJPEG stream the dashboard shows under its boxes.
    #             enabled=False keeps the metadata dashboard and serves no
    #             pixels at all, which is the pre-video behaviour.
    #             fps/quality/max_width are the bandwidth dial: 8 Hz at 70%
    #             and 960 px is roughly 1-3 Mbit/s per viewer. Encoding is
    #             skipped entirely when no tab is watching, dropping to
    #             snapshot_fps so /snapshot stays fresh for thumbnails.
    #   search    open-vocabulary search over embedded crops. `model` is the
    #             ONE place the embedding space is named - the dashboard
    #             queries it and the background embedder fills it, so they
    #             cannot disagree (they used to, and search silently returned
    #             nothing). embed_in_background keeps the index current on a
    #             duty cycle; see src/server/embedder.py for the arithmetic.
    "server": {"host": "127.0.0.1", "port": 8000, "push_hz": 8.0,
               "video": {"enabled": True, "fps": 8, "quality": 70,
                         "max_width": 960, "snapshot_fps": 1},
               "search": {"model": "clip-vit-l14",
                          "embed_in_background": True,
                          "batch": 8, "pause_s": 3.0, "idle_poll_s": 30,
                          "min_px": 48, "min_conf": 0.5, "max_area": 0.33}},

    # INCIDENT WEBHOOKS. Policy, not code: which kinds fire is config.
    "incidents": {
        "enabled": True,
        # A per-vehicle incident fires at track retirement, where the voted
        # plate and chosen crop exist. A track that never retires (a parked
        # car - exactly the stationary-obstruction case worth alerting on)
        # force-fires after this many seconds of continuous flagging, then is
        # suppressed permanently so "fire once" stays true.
        "max_dwell_s": 120,
        # Congestion fires on clear->congested and back, but only once the new
        # state has HELD this long. Without the dwell, a metric oscillating
        # around its threshold emits hundreds of webhooks a minute.
        "congestion_dwell_s": 30,
        "congested_at": "heavy",     # free | busy | heavy | jammed
        # Prefix for image_url in the payload. A crop's local path under
        # outputs/ is unreadable to a remote consumer, so the payload points at
        # this server's /crops endpoint instead. Empty = send null.
        "base_url": "",
        # crossing is deliberately false and would be wrong to enable: it fires
        # for EVERY vehicle, so it is a counter rather than an incident.
        "kinds": {"wrong_way": True, "congestion": True,
                  "wrong_lane": False, "crossing": False,
                  "face_match": True,
                  # Behaviour (analysis/behaviour.py) fires AT DETECTION: the
                  # analysis already held each rule for its own hold time.
                  # running is off: children and joggers make it noisy.
                  "intrusion": True, "loitering": True, "crowd": True,
                  "running": False},
        # A recognised person fires AT CONFIRMATION, not at retirement: the
        # name is settled then, and "who just walked in" is worth nothing
        # five seconds late. The same person on the same camera again within
        # this many seconds (a track split by an occlusion, walking out and
        # straight back) is recorded as an event but not re-sent.
        "face_match_cooldown_s": 60,
        # One incident fans out to every matching subscription, each with its
        # own deliveries row - so retries and dead-lettering are per subscriber
        # and one broken consumer cannot delay another. Absent or empty filters
        # mean "everything". The secret is read from the ENVIRONMENT, never
        # from this file.
        "subscriptions": [],
    },

    # RETENTION beyond frames. 0 = unlimited everywhere, matching
    # frames.retention exactly so there is one retention idiom, not two.
    # Nothing is deleted that an operator did not ask to have deleted.
    "retention": {
        "check_interval": 60,
        # Crops were never reaped at all before this: one JPEG per object
        # accumulated forever. A crop an incident references is PINNED and
        # survives until that incident is reaped, so a delivered image_url
        # stays valid exactly as long as its incident.
        "crops": {"max_age_hours": 0, "max_disk_gb": 0},
        # events is the high-volume table - `crossing` fires per vehicle per
        # line - so it is the one most likely to need a real cap in practice.
        "events": {"max_age_hours": 0},
        "deliveries": {"max_age_hours": 0},
        "incidents": {"max_age_hours": 0},
    },
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


def _strip_behaviour(raw: dict, camera_id: str) -> dict:
    """Behaviour is never set up in a camera file.

    The owner does not want the behaviour analysis attached to cameras, so no
    camera - and therefore no serve.py worker - runs it on its own. It runs
    only when an operator asks for it with --behaviour and a zones file (see
    apply_behaviour). A camera file carrying the block is told so and the block
    is dropped, rather than half-working or silently vanishing.
    """
    cleaned = dict(raw or {})
    if _pop(cleaned, "analyses.behaviour"):
        print(f"[config] cameras/{camera_id}.yaml sets 'analyses.behaviour', but "
              f"behaviour is not a camera setting; ignoring it. Run it with "
              f"--behaviour behaviour/<name>.yaml")
    return cleaned


def apply_behaviour(cfg: dict, path: str) -> dict:
    """Switch the behaviour analysis on for one run, with zones from `path`.

        python main.py --camera person_test --behaviour behaviour/person_test.yaml

    The file holds what would otherwise be an `analyses.behaviour` block -
    zones, exclude masks, any threshold overrides - at its top level. Kept in
    behaviour/, not cameras/, because behaviour is not a camera setting
    (_strip_behaviour). Changes `cfg` in place and returns it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"behaviour file not found: {path}")
    raw = _read_yaml(path) or {}
    known = DEFAULTS["analyses"]["behaviour"]
    for key in raw:
        if key not in known:
            print(f"[config] {path}: unknown behaviour key {key!r} "
                  f"(no such setting); ignoring")
    analyses = cfg.setdefault("analyses", {})
    block = _merge(analyses.get("behaviour") or {},
                   {k: v for k, v in raw.items() if k in known})
    block["enabled"] = True
    analyses["behaviour"] = block
    print(f"[config] {path}: behaviour on, {len(block.get('zones') or [])} zone(s), "
          f"{len(block.get('exclude') or [])} exclude mask(s)")
    return cfg


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
            cfg = _merge(cfg, _strip_locked(_strip_behaviour(raw, camera_id),
                                            camera_id))
            applied = sorted({p.split(".")[0] for p, _ in _walk(raw)})
            print(f"[config] {cam_path}: applied {', '.join(applied) or 'nothing'}")
        cfg.setdefault("camera", {})["id"] = camera_id
        cfg["camera"].setdefault("name", camera_id)
    return _expand_paths(cfg, cfg.get("camera", {}).get("id", "default"))
