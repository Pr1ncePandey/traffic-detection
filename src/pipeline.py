"""Orchestrator: source -> sample -> detect/track -> attributes -> analyses.

Deliberately thin. It owns the loop and the order of operations, and knows
nothing about any individual use case - plugins are looked up by name from
config. Adding congestion, or anything else, does not touch this file.

Order matters and is load-bearing:
  1. detect/track                  ids assigned
  2. register tracks in the store  so plugins can read per-object state
  3. capture previous positions    BEFORE observe() overwrites them, or every
                                   direction vector reads as zero
  4. PERCEPTION stage              attribute enrichers: plate, colour. THE
                                   COMMON POINT - everything after this sees
                                   objects that already carry their attributes
  5. ANALYSIS stage                congestion, wrong-way, counting. Independent
                                   consumers of what step 4 produced; they run
                                   side by side and know nothing of each other
  6. bind durable identity         plate -> vehicle id, BEFORE the labels are
                                   drawn so a re-entering vehicle shows its
                                   original id in the same frame it is read
  7. draw boxes, persist, evict

Steps 4 and 5 are one call each, into the same scheduler (runtime/plugin.py)
with two instances. The pipeline does not know how many plugins there are,
whether any ran on another thread, or what they concluded.

Why the two stages are separate rather than one list: an enricher states a FACT
about an object and an analysis draws a CONCLUSION from it, so the enrichers
must run first. They used to share one hand-ordered list in which `anpr` and
`color` were listed last, which meant no analysis could read an attribute at
all.

Step 5 lives here rather than inside the ANPR analyzer on purpose. Identity is
the same concern as the object ids assigned in step 2, and analysis/base.py is
explicit that an analyzer must not see the storage backend. So the pipeline
reads whatever plate an analyzer left on det.extra and does the binding - which
also means any future plate source gets re-identification for free.
"""

import os
import time

import cv2
from tqdm import tqdm

from .analysis import build as build_analyzers
from .analysis.stage import AnalysisStage
from .attributes.plate import crop_score
from .attributes.registry import build as build_enrichers
from .attributes.stage import AttributeStage
from .attributes.plate_format import fits_template
from .detectors.classes import VEHICLE
from .detectors.yolo import Detector
from .runtime.context import FrameContext
from .runtime.reader import FrameReader
from .runtime.sampler import from_config as sampler_from_config
from .runtime.source import open_source
from .storage.csv_store import CsvStore
from .storage.frames import Reaper, build_writer
from .storage.sqlite_store import SqliteStore
from .trackers.identity import from_config as identity_from_config
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


def _label(det, store=None):
    """Box caption. Prefers the durable vehicle id over the track id.

    This is the visible half of re-identification: ByteTrack gives a returning
    vehicle a new track id, so showing "#31" for a car last seen as "#4" is
    what makes the system look like it has forgotten. V7 is the same car.
    """
    tid = "?" if det.track_id is None else det.track_id
    vid = None if store is None else store.vehicle_of.get(det.track_id)
    tag = (f"V{vid} {det.cls_name} {det.conf:.2f}" if vid is not None
           else f"#{tid} {det.cls_name} {det.conf:.2f}")
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

    perception = cfg["perception"]
    detector = Detector(perception["model"])
    store = TrackStore()
    ttl = ttl_for(sampler.effective_fps,
                  perception.get("tracker", {}).get("track_buffer", 30))
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

    reid_cfg = cfg.get("reid", {}) or {}
    identity = identity_from_config(reid_cfg, storage)
    reid_min_conf = float(reid_cfg.get("min_conf", 0.7))
    reid_require_format = bool(reid_cfg.get("require_format", True))

    an_cfg = cfg.get("analysis", {}) or {}
    parallel = bool(an_cfg.get("parallel", False))
    workers = an_cfg.get("workers")
    # Two stages, two pools. Perception is where parallelism pays (plate OCR is
    # tens of ms of ONNX per frame); the analyses are cheap by comparison.
    perception_stage = AttributeStage(build_enrichers(cfg, info),
                                      parallel=parallel, workers=workers)
    stage = AnalysisStage(build_analyzers(cfg, info),
                          parallel=parallel, workers=workers)
    stages = (perception_stage, stage)

    out = None
    if cfg["video"].get("write_video", True):
        os.makedirs(os.path.dirname(cfg["video"]["target"]) or ".", exist_ok=True)
        out = cv2.VideoWriter(cfg["video"]["target"], cv2.VideoWriter_fourcc(*"mp4v"),
                              sampler.effective_fps, (W, H))
    csv = CsvStore(cfg["video"]["csv"]) if st_cfg.get("csv_export", True) else None

    print(f"[pipeline] {info.label}")
    print(f"[pipeline] {detector.describe()}")
    print(f"[pipeline] sampling {sampler.describe()} | reader {reader.describe()}")
    print(f"[pipeline] perception: {perception_stage.describe()}")
    print(f"[pipeline] analysis: {stage.describe()}")
    print(f"[pipeline] frames={fr_cfg.get('format') if save_frames else 'off'} "
          f"| db={st_cfg.get('path')} | track ttl={ttl:.1f}s")
    print(f"[pipeline] re-id: "
          + ("off (track ids only; a re-entering vehicle is a new object)"
             if identity is None else
             f"plate-keyed, min_conf={reid_min_conf} "
             f"format={'required' if reid_require_format else 'optional'}"))

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
            detections = detector.detect(raw, perception["tracker"])

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
            # Attributes FIRST, so every analysis below sees objects that
            # already carry their plate and colour on this same frame.
            perception_stage.run(ctx)
            stage.run(ctx)

            if identity is not None:
                _bind_identities(identity, ctx, reid_min_conf, reid_require_format)

            for det in detections:
                color = _box_color(det)
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, _label(det, store), (x1, max(12, y1 - 8)),
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
                             "object_id": det.track_id,
                             "vehicle_id": store.vehicle_of.get(det.track_id),
                             "vehicle_class": det.cls_name,
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
                                  on_evict=lambda t, v: _finalize(
                                      storage, store, stages, t, v, identity,
                                      reid_min_conf, reid_require_format))
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
            _finalize(storage, store, stages, tid, vehicle, identity,
                      reid_min_conf, reid_require_format)
        for st in stages:
            st.close()
        if writer is not None:
            writer.close()
        reaper.stop()
        if out is not None:
            out.release()
        storage.flush()
        df = csv.close() if csv is not None else None

    elapsed = max(1e-6, time.time() - start)
    summaries = {**perception_stage.summaries(), **stage.summaries()}
    if identity is not None:
        summaries["_identity"] = identity.stats()
    _write_summary(cfg, info, store, elapsed, read_n, analysed, summaries, writer,
                   identity)
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


def _plate_is_bindable(plate: str, conf: float, min_conf: float,
                       require_format: bool) -> bool:
    """Is this read good enough to key a durable identity on?

    Deliberately stricter than the bar for DISPLAYING a plate. plate.min_conf
    (0.5) decides whether a read is worth showing; getting that wrong costs a
    wrong caption on one box. This decides whether two sightings are the same
    car; getting it wrong merges two vehicles' histories, which no later frame
    undoes. Hence a higher confidence floor and, by default, a plate that
    actually fits a registration template rather than merely looking plate-ish.
    """
    if not plate:
        return False
    if float(conf or 0.0) < float(min_conf):
        return False
    return fits_template(plate) if require_format else True


def _bind_identities(identity, ctx, min_conf: float, require_format: bool):
    """Attach a durable vehicle id to any track whose plate is now readable.

    Runs every frame, after the analyzers and before the labels are drawn.
    Provisional by nature: plate._consensus() keeps voting as more reads
    accumulate, so the string can still change. Rebinding on change is cheap
    and self-correcting - only the caption and objects.vehicle_id depend on it,
    and _finalize() re-resolves from the FINAL voted plate. No detection row
    ever references a vehicle id, so nothing needs rewriting.
    """
    store = ctx.store
    for det in ctx.detections:
        tid = det.track_id
        if tid is None:
            continue
        plate = det.extra.get("plate_number") or ""
        if not plate or plate == store.plate_of.get(tid):
            continue          # unchanged since the last bind: nothing to do
        if not _plate_is_bindable(plate, det.extra.get("plate_conf", 0.0),
                                  min_conf, require_format):
            continue
        vehicle_id = identity.resolve(plate, ctx.timestamp)
        if vehicle_id is None:
            continue
        previous = store.vehicle_of.get(tid)
        store.plate_of[tid] = plate
        store.vehicle_of[tid] = vehicle_id
        if vehicle_id != previous:
            ctx.emit("vehicle_identified",
                     {"vehicle_id": vehicle_id, "plate": plate,
                      "conf": det.extra.get("plate_conf", 0.0),
                      "rebound_from": previous},
                     track_id=tid)


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


def _settled_vehicle_id(storage, identity, store, tid, vehicle, attrs,
                        min_conf: float, require_format: bool):
    """The vehicle id to persist for a retiring track.

    Prefers a fresh resolve of the FINAL voted plate over whatever was bound
    mid-track: _bind_identities works from a running consensus that is still
    being voted, so an early binding can be superseded. Falls back to the
    mid-track binding, then to None.
    """
    bound = store.vehicle_of.get(tid)
    if identity is None:
        return bound
    plate = attrs.get("plate_number", "") or ""
    if not _plate_is_bindable(plate, _attr_conf(attrs, "plate_number"),
                              min_conf, require_format):
        return bound
    resolved = identity.resolve(plate, vehicle.get("last_seen_s", 0.0))
    if resolved is None:
        return bound
    # Submit the sighting's real bounds. resolve() only ever knows the instant
    # a plate became legible, which is part-way through the track; the upsert
    # takes MIN/MAX, so this widens the vehicle's window to cover when the car
    # was actually visible rather than when its plate happened to be readable.
    touch = getattr(storage, "touch_vehicle", None)
    if touch is not None:
        try:
            touch(resolved, plate, vehicle.get("first_seen_s", 0.0),
                  vehicle.get("last_seen_s", 0.0))
        except Exception as e:
            print(f"[pipeline] vehicle touch failed for {resolved}: {e}")
    return resolved


def _finalize(storage, store, stages, tid, vehicle, identity=None,
              min_conf: float = 0.7, require_format: bool = True):
    """Persist an object row plus its attributes, then let plugins forget it.

    This is where identity becomes authoritative. Any binding made mid-track
    used a running consensus; by the time a track is retired the vote is
    complete, so the final plate is re-resolved and that is the vehicle_id
    written to the row.
    """
    oid = store.object_ids.get(tid)
    if oid is None:
        return
    attrs = vehicle.get("attrs", {}) or {}
    vehicle_id = _settled_vehicle_id(storage, identity, store, tid, vehicle,
                                     attrs, min_conf, require_format)
    storage.upsert_object({
        "id": oid, "track_id": tid, "vehicle_id": vehicle_id,
        "cls_name": vehicle.get("vehicle_class", ""),
        "cls_group": vehicle.get("cls_group", ""),
        "first_seen_s": vehicle.get("first_seen_s", 0.0),
        "last_seen_s": vehicle.get("last_seen_s", 0.0),
        "frames_seen": vehicle.get("frames_seen", 0),
        "best_conf": vehicle.get("best_conf", 0.0),
        "crop_path": vehicle.get("_crop_path"),
        "lane_id": store.lane_of.get(tid),
        "lane_flag": store.lane_flag_of.get(tid)})
    ts = vehicle.get("last_seen_s", 0.0)
    for key, value in attrs.items():
        if key.endswith("_conf") or value in ("", None):
            continue
        storage.put_attribute(oid, key, str(value), _attr_conf(attrs, key), ts)
    # BOTH stages: an enricher keeps per-track read state too, and a leak there
    # is unbounded on a 24/7 feed.
    for st in stages:
        st.forget(tid)


def _write_summary(cfg, info, store, elapsed, read_n, analysed, summaries, writer,
                   identity=None):
    path = cfg["video"]["summary"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cam = cfg.get("camera", {})
    zones = (cfg.get("analyses", {}).get("counting", {}).get("zones", {}) or {})
    tracks = len(store.object_ids) + store.evicted
    with open(path, "w") as f:
        f.write("TRAFFIC REPORT\n==============\n")
        f.write(f"Source        : {info.label}\n")
        f.write(f"Camera        : {cam.get('name', 'demo')} "
                f"({zones.get('label_a', 'entry')} -> "
                f"{zones.get('label_b', 'exit')})\n")
        f.write(f"Frames        : read {read_n}, analysed {analysed} in "
                f"{elapsed:.1f}s ({analysed/elapsed:.1f} fps)\n")
        # Named precisely: ByteTrack mints a new id for a reappearing object,
        # so this counts SIGHTINGS. The vehicle line below is the de-duplicated
        # figure, for the vehicles whose plate was actually read.
        f.write(f"Distinct track IDs: {tracks} "
                f"(one per sighting - a re-entering object gets a new ID)\n")
        if identity is not None:
            st = identity.stats()
            # The honest count, and the gap between the two lines is the point:
            # tracks over-counts, this does not - for vehicles whose plate was
            # actually read. Vehicles with no readable plate are in neither.
            # "identities created", not "vehicles seen", and deliberately NOT
            # a re-entry count: this process cannot tell a second sighting from
            # the same track re-resolving at finalize. The number of cars that
            # actually came back is a question about objects.vehicle_id, which
            # report.py answers exactly.
            f.write(f"Vehicle identities created: {st['vehicles_created']} "
                    f"(plates that bound to a durable id)\n")
            f.write("  Re-entries (one vehicle, several sightings): "
                    "run `python report.py`\n")
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
