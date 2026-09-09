"""Orchestrator: source -> sample -> detect/track -> analyzers -> storage.

Deliberately thin. It owns the loop and the order of operations, and knows
nothing about any individual use case - analyzers are looked up by name from
config. Adding congestion, or anything else, does not touch this file.

Order matters and is load-bearing:
  1. detect/track                  ids assigned
  2. register tracks in the store  so analyzers can read per-object state
  3. capture previous positions    BEFORE observe() overwrites them, or every
                                   direction vector reads as zero
  4. analysis stage                compute (parallelisable) -> apply -> draw,
                                   apply/draw in config order
  5. draw boxes, persist, evict

Step 4 is one call into analysis/stage.py. The pipeline does not know how many
analyses there are, whether any of them ran on another thread, or what they
concluded - which is the property that lets wrong-side detection, congestion
and ANPR be independent analyses over the same tracked objects.
"""

import os
import time

import cv2
from tqdm import tqdm

from .analysis import build as build_analyzers
from .analysis.stage import AnalysisStage
from .attributes.plate import crop_score
from .detectors.classes import VEHICLE
from .detectors.yolo import Detector
from .runtime.context import FrameContext
from .runtime.reader import FrameReader
from .runtime.sampler import from_config as sampler_from_config
from .runtime.source import open_source
from .storage.csv_store import CsvStore
from .storage.frames import Reaper, build_writer
from .storage.sqlite_store import SqliteStore
from .trackers.store import TrackStore, ttl_for

BOX_OK = (0, 255, 0)
BOX_WRONG_WAY = (0, 0, 255)
BOX_WRONG_LANE = (0, 165, 255)
BOX_BY_GROUP = {"person": (255, 200, 0), "animal": (255, 0, 255),
                "obstacle": (0, 0, 255), "infrastructure": (160, 160, 160)}


def _box_color(det):
    if "wrong_way" in det.lane_flag:
        return BOX_WRONG_WAY
    if "wrong_lane" in det.lane_flag:
        return BOX_WRONG_LANE
    if det.group == VEHICLE:
        return BOX_OK
    return BOX_BY_GROUP.get(det.group, (200, 200, 200))


def _label(det):
    tid = "?" if det.track_id is None else det.track_id
    tag = f"#{tid} {det.cls_name} {det.conf:.2f}"
    if det.lane_id:
        tag += f" {det.lane_id}:{det.lane_flag}"
    plate = det.extra.get("plate_number")
    if plate:
        tag += f" {plate}"
    return tag


def run_pipeline(cfg: dict) -> dict:
    source = open_source(cfg["video"]["source"], cfg.get("source", {}))
    info = source.info
    W, H = info.width, info.height
    sampler = sampler_from_config(cfg.get("processing", {}), info.fps, info.is_live)
    reader = FrameReader(source,
                         backpressure=cfg.get("source", {}).get("backpressure"),
                         queue_size=cfg.get("source", {}).get("queue_size"))

    detector = Detector(cfg["model"])
    store = TrackStore()
    ttl = ttl_for(sampler.effective_fps, cfg.get("tracker", {}).get("track_buffer", 30))
    evict_every = max(1, int(cfg.get("processing", {}).get("evict_every", 150)))

    st_cfg = cfg.get("storage", {})
    storage = SqliteStore(st_cfg.get("path", "outputs/traffic.db"),
                          batch_rows=st_cfg.get("batch_rows"),
                          commit_interval=st_cfg.get("commit_interval"))
    storage.start_run({"source": str(cfg["video"]["source"]),
                       "camera": cfg.get("camera", {}).get("name", ""),
                       "fps": info.fps, "width": W, "height": H,
                       "analyse_fps": sampler.effective_fps, "config": cfg})

    fr_cfg = cfg.get("frames", {}) or {}
    save_frames = bool(fr_cfg.get("enabled", True))
    writer = build_writer(fr_cfg, sampler.effective_fps, (W, H)) if save_frames else None
    ret = fr_cfg.get("retention", {}) or {}
    reaper = Reaper(storage, max_age_hours=ret.get("max_age_hours", 0),
                    max_disk_gb=ret.get("max_disk_gb", 0),
                    interval=ret.get("check_interval", 60),
                    usage_fn=(lambda: writer.bytes_written) if writer else None)
    if reaper.enabled:
        reaper.reconcile()
        reaper.start()

    obj_cfg = cfg.get("objects", {}) or {}
    crop_dir = obj_cfg.get("crop_dir", "outputs/crops")
    save_crops = bool(obj_cfg.get("save_crops", True))
    crop_min_conf = float(obj_cfg.get("crop_min_conf", 0.5))
    if save_crops:
        os.makedirs(crop_dir, exist_ok=True)

    analyzers = build_analyzers(cfg.get("analyzers", []), cfg, info)
    an_cfg = cfg.get("analysis", {}) or {}
    # The stage owns phase order and threading; see analysis/stage.py for why
    # `parallel` is opt-in rather than the default.
    stage = AnalysisStage(analyzers,
                          parallel=bool(an_cfg.get("parallel", False)),
                          workers=an_cfg.get("workers"))

    out = None
    if cfg["video"].get("write_video", True):
        os.makedirs(os.path.dirname(cfg["video"]["target"]) or ".", exist_ok=True)
        out = cv2.VideoWriter(cfg["video"]["target"], cv2.VideoWriter_fourcc(*"mp4v"),
                              sampler.effective_fps, (W, H))
    csv = CsvStore(cfg["video"]["csv"]) if st_cfg.get("csv_export", True) else None

    print(f"[pipeline] {info.label}")
    print(f"[pipeline] {detector.describe()}")
    print(f"[pipeline] sampling {sampler.describe()} | reader {reader.describe()}")
    print(f"[pipeline] analysis: {stage.describe()}")
    print(f"[pipeline] frames={fr_cfg.get('format') if save_frames else 'off'} "
          f"| db={st_cfg.get('path')} | track ttl={ttl:.1f}s")

    max_frames = int(cfg.get("processing", {}).get("max_frames", -1))
    total = None
    if info.total_frames and not info.is_live:
        total = int(info.total_frames / max(1.0, info.fps / sampler.effective_fps))
        if max_frames > 0:
            total = min(total, max_frames)

    start = time.time()
    read_n = analysed = 0
    pbar = tqdm(total=total, desc="Processing")
    reader.start()
    try:
        for index, raw in reader.frames():
            read_n = index + 1
            if not sampler.should_process(index):
                continue
            if max_frames > 0 and analysed >= max_frames:
                break
            analysed += 1
            timestamp = (time.time() if info.is_live
                         else (index / info.fps if info.fps else index))
            annotated = raw.copy()
            detections = detector.detect(raw, cfg["tracker"])

            for det in detections:
                if det.track_id is None:
                    continue
                store.touch(det.track_id, det.cls_name, timestamp,
                            group=det.group, conf=det.conf)
                if det.track_id not in store.object_ids:
                    store.object_ids[det.track_id] = storage.next_object_id()
                cx, cy = det.centroid
                # Capture the PREVIOUS position before observe() replaces it.
                det.extra["prev_xy"] = store.observe(det.track_id, cx, cy)

            ctx = FrameContext(frame_no=analysed, timestamp=timestamp, raw=raw,
                               annotated=annotated, detections=detections,
                               store=store, source=info)
            stage.run(ctx)

            for det in detections:
                color = _box_color(det)
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, _label(det), (x1, max(12, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            frame_id = storage.next_frame_id()
            ctx.frame_id = frame_id
            meta = ({"raw_path": None, "annotated_path": None, "segment_id": None,
                     "frame_offset": None, "bytes": None} if writer is None
                    else writer.write_pair(analysed, timestamp, raw, annotated))
            storage.put_frame({"id": frame_id, "frame_no": analysed,
                               "ts": timestamp, **meta})

            for det in detections:
                oid = store.object_ids.get(det.track_id)
                storage.put_detection({
                    "frame_id": frame_id, "object_id": oid,
                    "x1": det.bbox[0], "y1": det.bbox[1],
                    "x2": det.bbox[2], "y2": det.bbox[3],
                    "conf": round(det.conf, 3), "cls_name": det.cls_name,
                    "event": det.event or "", "lane_id": det.lane_id or "",
                    "lane_flag": det.lane_flag or ""})
                if csv is not None:
                    csv.put({"frame": analysed, "time_s": round(timestamp, 2),
                             "object_id": det.track_id, "vehicle_class": det.cls_name,
                             "cls_group": det.group, "confidence": round(det.conf, 3),
                             "x1": det.bbox[0], "y1": det.bbox[1],
                             "x2": det.bbox[2], "y2": det.bbox[3],
                             "event": det.event or "", "lane_id": det.lane_id or "",
                             "lane_flag": det.lane_flag or "",
                             "plate_number": det.extra.get("plate_number", ""),
                             "plate_conf": det.extra.get("plate_conf", 0.0)})
                if det.track_id is None or oid is None:
                    continue
                _keep_best_crop(store, det, raw, crop_dir, save_crops, crop_min_conf)

            for ev in ctx.events:
                storage.put_event({"frame_id": frame_id,
                                   "object_id": store.object_ids.get(ev["track_id"]),
                                   "kind": ev["kind"], "detail_json": ev["detail"],
                                   "ts": ev["ts"]})

            if analysed % evict_every == 0:
                store.evict_stale(timestamp, ttl,
                                  on_evict=lambda t, v: _finalize(storage, store,
                                                                  stage, t, v))
            if out is not None:
                out.write(annotated)
            pbar.update(1)
    except KeyboardInterrupt:
        print("\n[pipeline] interrupted; flushing what has been analysed")
    finally:
        pbar.close()
        reader.stop()
        source.release()
        # Everything still tracked at shutdown must be written, or the last
        # objects of a run would exist only in memory.
        for tid, vehicle in list(store.vehicles.items()):
            _finalize(storage, store, stage, tid, vehicle)
        stage.close()
        if writer is not None:
            writer.close()
        reaper.stop()
        if out is not None:
            out.release()
        storage.flush()
        df = csv.close() if csv is not None else None

    elapsed = max(1e-6, time.time() - start)
    summaries = stage.summaries()
    _write_summary(cfg, info, store, elapsed, read_n, analysed, summaries, writer)
    storage.close()

    print(f"[done] {elapsed:.1f}s | read={read_n} analysed={analysed} "
          f"({analysed/elapsed:.1f} fps) | tracks={len(store.object_ids) + store.evicted} "
          f"| A->B={store.a_to_b} B->A={store.b_to_a} "
          f"| dropped={reader.frames_dropped} | rows={storage.rows_written}")
    return {"frames": analysed, "read": read_n, "df": df, "store": store,
            "db": st_cfg.get("path"), "summaries": summaries}


def _keep_best_crop(store, det, raw, crop_dir, save_crops, min_conf):
    """One image per object, the sharpest seen rather than the first.

    The original kept whichever crop appeared first above the threshold and
    latched it forever, which for an approaching vehicle is its smallest and
    blurriest view.
    """
    if not save_crops:
        return
    x1, y1, x2, y2 = det.bbox
    crop = raw[y1:y2, x1:x2]
    if crop.size == 0:
        return
    vehicle = store.vehicles.get(det.track_id)
    if vehicle is None:
        return
    have = vehicle.get("_crop_path") is not None
    # Below the confidence bar we still take a first image (so every object has
    # one) but never overwrite an image taken above it.
    if have and det.conf < min_conf:
        return
    score = crop_score(crop)
    if have and score <= float(vehicle.get("_crop_score", 0.0)):
        return
    vehicle["_crop_score"] = score if det.conf >= min_conf else 0.0
    path = os.path.join(crop_dir, f"object_{store.object_ids[det.track_id]}.jpg")
    try:
        cv2.imwrite(path, crop)
        vehicle["_crop_path"] = path
    except Exception as e:
        print(f"[pipeline] crop write failed {path}: {e}")


# attrs uses a shorter confidence key than the value key for plates
# ("plate_number" pairs with "plate_conf", not "plate_number_conf").
_CONF_KEY = {"plate_number": "plate_conf"}


def _attr_conf(attrs: dict, key: str) -> float:
    for candidate in (_CONF_KEY.get(key), f"{key}_conf"):
        if candidate and candidate in attrs:
            try:
                return float(attrs[candidate] or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _finalize(storage, store, stage, tid, vehicle):
    """Persist an object row plus its attributes, then let analyzers forget it."""
    oid = store.object_ids.get(tid)
    if oid is None:
        return
    storage.upsert_object({
        "id": oid, "track_id": tid,
        "cls_name": vehicle.get("vehicle_class", ""),
        "cls_group": vehicle.get("cls_group", ""),
        "first_seen_s": vehicle.get("first_seen_s", 0.0),
        "last_seen_s": vehicle.get("last_seen_s", 0.0),
        "frames_seen": vehicle.get("frames_seen", 0),
        "best_conf": vehicle.get("best_conf", 0.0),
        "crop_path": vehicle.get("_crop_path"),
        "lane_id": store.lane_of.get(tid),
        "lane_flag": store.lane_flag_of.get(tid)})
    attrs = vehicle.get("attrs", {}) or {}
    ts = vehicle.get("last_seen_s", 0.0)
    for key, value in attrs.items():
        if key.endswith("_conf") or value in ("", None):
            continue
        storage.put_attribute(oid, key, str(value), _attr_conf(attrs, key), ts)
    stage.forget(tid)


def _write_summary(cfg, info, store, elapsed, read_n, analysed, summaries, writer):
    path = cfg["video"]["summary"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cam = cfg.get("camera", {})
    tracks = len(store.object_ids) + store.evicted
    with open(path, "w") as f:
        f.write("TRAFFIC REPORT\n==============\n")
        f.write(f"Source        : {info.label}\n")
        f.write(f"Camera        : {cam.get('name', 'demo')} "
                f"({cam.get('label_a', 'entry')} -> {cam.get('label_b', 'exit')})\n")
        f.write(f"Frames        : read {read_n}, analysed {analysed} in "
                f"{elapsed:.1f}s ({analysed/elapsed:.1f} fps)\n")
        # Named precisely: ByteTrack mints a new id for a reappearing object,
        # so this counts TRACKS, which is an upper bound on distinct objects.
        f.write(f"Distinct track IDs: {tracks} "
                f"(upper bound on real objects - a re-entering object gets a new ID)\n")
        f.write(f"A->B          : {store.a_to_b}\nB->A          : {store.b_to_a}\n")
        if writer is not None:
            f.write(f"Frames stored : {writer.frames_written} files, "
                    f"{writer.bytes_written/1e6:.1f} MB ({writer.format})\n")
        f.write("\nPer-class track counts:\n")
        for k, v in sorted(store.per_class_counts().items(), key=lambda kv: -kv[1]):
            f.write(f"  {k}: {v}\n")
        f.write("\nAnalyzer summaries:\n")
        for name, data in summaries.items():
            f.write(f"  {name}: {data}\n")
