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

import datetime
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
from .timebase import time_base_for
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


def _safely(callback, *args, what: str = "callback"):
    """Invoke a subscriber, absorbing whatever it raises.

    Error containment, per Layer B: failures in the one-shot pipeline could
    print and continue because a crash ended a finite run anyway. In a service
    a camera thread that dies takes its feed offline, and a camera that is
    silently dead is the failure mode worth designing against - so a dashboard
    subscriber or a webhook policy raising must cost that frame's notification
    and nothing more.
    """
    if callback is None:
        return None
    try:
        return callback(*args)
    except Exception as e:
        print(f"[pipeline] {what} raised {type(e).__name__}: {e}")
        return None


def _camera_location(cam_cfg: dict):
    """(lat, lon) for this camera, or (None, None) if it has not said.

    Snapshotted onto the run row by the caller. Absent is a first-class answer:
    a journey through a camera with no location reports the hop as
    distance-unknown, which is honest, rather than as a 0 km hop at 0 km/h,
    which would silently pass the implied-speed check that exists to catch
    plate collisions.
    """
    loc = cam_cfg.get("location") or {}
    if not isinstance(loc, dict):
        print(f"[pipeline] camera.location should be a mapping with lat/lon, "
              f"got {type(loc).__name__}; treating the location as unknown")
        return None, None
    out = []
    for key in ("lat", "lon"):
        value = loc.get(key)
        if value is None or value == "":
            return None, None
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            print(f"[pipeline] camera.location.{key}={value!r} is not a number; "
                  f"treating the location as unknown")
            return None, None
    lat, lon = out
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        print(f"[pipeline] camera.location ({lat}, {lon}) is outside valid "
              f"lat/lon range; treating the location as unknown")
        return None, None
    return lat, lon


def _recorded_at(cfg: dict, info) -> float | None:
    """When the footage starts, as an epoch. None for live, or if unsupplied.

    A file run without this is unorderable against any other camera, and that
    is worth one line of output at start-up: it is invisible otherwise until a
    journey query quietly reports fewer hops than expected.
    """
    if info.is_live:
        return None
    raw = (cfg.get("video", {}) or {}).get("recorded_at")
    if raw in (None, ""):
        print("[pipeline] file source with no video.recorded_at: this run's "
              "timestamps are clip-relative and cannot be ordered against "
              "another camera (pass --recorded-at to anchor them)")
        return None
    value = parse_recorded_at(raw)
    if value is None:
        print(f"[pipeline] could not read video.recorded_at={raw!r} as an epoch "
              f"or ISO-8601 time; this run stays unorderable")
    return value


def parse_recorded_at(raw) -> float | None:
    """Epoch seconds from a number or an ISO-8601 string. None if unreadable.

    A naive ISO string (no offset) is read in LOCAL time, matching how an
    operator reading a timestamp off a CCTV file would mean it.
    """
    if raw in (None, ""):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, (datetime.datetime, datetime.date)):
        moment = (raw if isinstance(raw, datetime.datetime)
                  else datetime.datetime(raw.year, raw.month, raw.day))
        return moment.timestamp()
    text = str(raw).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def run_pipeline(cfg: dict, stop=None, on_frame=None, on_event=None,
                 on_ready=None, storage=None, progress: bool = True) -> dict:
    """Run one camera to completion, or until `stop` is set.

    The four optional arguments are what let a long-running server drive this
    loop without the loop knowing a server exists. All default to None, so a
    one-shot `python main.py` takes exactly the path it always did.

      stop      threading.Event checked once per iteration. A camera has to be
                stoppable cleanly - mid-frame is not a safe place to stop,
                because the object row for the track in flight would be lost.
      on_frame  called after the stages run, with a metadata dict. This is how
                the WebSocket hub gets frames without pipeline.py importing it.
                Metadata only: the annotated frame stays here.
      on_event  called per emitted event, with the event and its context. The
                incident layer hooks this rather than the analyzers, so no
                detector needs to know incidents exist.
      on_ready  called once with the run's own facts (run_id, source info)
                after start-up succeeds, so a supervisor can mark the camera
                healthy and show its resolution and fps.

    Callbacks are invoked inside the frame loop, so a slow one slows the camera.
    Each is wrapped: a subscriber raising must not kill the run - see _safely().

    `storage` is the SHARED-STORE hatch, and it is what makes N camera workers
    in one process legal. SqliteStore is built around exactly one writer thread
    on one connection; if every camera constructed its own, N writer threads
    would contend for one file and the invariant the storage layer is designed
    around would be gone. So the server builds one store and passes it here.
    When it does, this function does not close it - the owner does - and every
    row it writes carries an explicit run_id, because a shared store's own
    `run_id` attribute holds whichever run started last.

    `progress` turns off the tqdm bar. A service wants it off: N cameras each
    redrawing a bar into the same log is unreadable, and a camera the
    supervisor restarts would emit a fresh one every time.
    """
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
    owns_storage = storage is None
    if owns_storage:
        storage = SqliteStore(st_cfg.get("path", "outputs/traffic.db"),
                              batch_rows=st_cfg.get("batch_rows"),
                              commit_interval=st_cfg.get("commit_interval"),
                              # Live: shed high-volume rows rather than stall
                              # the frame loop, since dropped wall-clock time
                              # on a camera is unrecoverable. A file can wait.
                              shed_when_full=info.is_live)
    cam_cfg = cfg.get("camera", {}) or {}
    lat, lon = _camera_location(cam_cfg)
    recorded_at = _recorded_at(cfg, info)
    run_id = storage.start_run({"source": str(cfg["video"]["source"]),
                                "camera": cam_cfg.get("id")
                                          or cam_cfg.get("name", ""),
                                "fps": info.fps, "width": W, "height": H,
                                "analyse_fps": sampler.effective_fps,
                                "config": cfg,
                                "time_base": time_base_for(info.is_live),
                                "recorded_at": recorded_at,
                                "cam_lat": lat, "cam_lon": lon})

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
    print(f"[pipeline] clock: {time_base_for(info.is_live)}"
          + (f", anchored at {datetime.datetime.fromtimestamp(recorded_at):%Y-%m-%d %H:%M:%S}"
             if recorded_at else "")
          + f" | location: "
          + (f"{lat:.5f},{lon:.5f}" if lat is not None else "not set"))
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
    pbar = tqdm(total=total, desc="Processing", disable=not progress)
    reader.start()
    camera_id = cam_cfg.get("id") or cam_cfg.get("name") or "camera"
    _safely(on_ready, {"camera": camera_id, "run_id": run_id,
                       "source": info.label, "width": W, "height": H,
                       "fps": info.fps, "analyse_fps": sampler.effective_fps,
                       "is_live": info.is_live},
            what="on_ready")
    try:
        for index, raw in reader.frames():
            # Checked before any work on this frame, so a stop lands between
            # frames rather than half way through one.
            if stop is not None and stop.is_set():
                print(f"[pipeline] stop requested after {analysed} frames")
                break
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
            storage.put_frame({"id": frame_id, "run_id": run_id,
                               "frame_no": analysed,
                               "ts": timestamp, **meta})

            for det in detections:
                oid = store.object_ids.get(det.track_id)
                storage.put_detection({
                    "run_id": run_id,
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
                object_id = store.object_ids.get(ev["track_id"])
                storage.put_event({"run_id": run_id,
                                   "frame_id": frame_id,
                                   "object_id": object_id,
                                   "kind": ev["kind"], "detail_json": ev["detail"],
                                   "ts": ev["ts"]})
                # The incident layer hooks the event DRAIN, not the analyzers:
                # lanes.py, counting.py and congestion.py already emit
                # everything needed, so no detector has to learn what an
                # incident is.
                _safely(on_event, ev, {"camera": camera_id, "object_id": object_id,
                                       "frame_no": analysed, "frame_id": frame_id,
                                       "timestamp": timestamp, "store": store},
                        what="on_event")

            if on_frame is not None:
                _safely(on_frame, _frame_meta(camera_id, analysed, timestamp,
                                              ctx, store, storage, reader),
                        what="on_frame")

            if analysed % evict_every == 0:
                store.evict_stale(timestamp, ttl,
                                  on_evict=lambda t, v: _finalize(
                                      storage, store, stages, t, v, identity,
                                      reid_min_conf, reid_require_format,
                                      on_retire=on_event, camera=camera_id,
                                      run_id=run_id))
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
                      reid_min_conf, reid_require_format,
                      on_retire=on_event, camera=camera_id, run_id=run_id)
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
    # A shared store belongs to the server and outlives this camera, so close
    # only what this call created. The run row still gets its ended_at, named
    # explicitly because the store's own run_id has moved on to another camera.
    if owns_storage:
        storage.close()
    else:
        storage.end_run(run_id)

    print(f"[done] {elapsed:.1f}s | read={read_n} analysed={analysed} "
          f"({analysed/elapsed:.1f} fps) | tracks={len(store.object_ids) + store.evicted} "
          f"| A->B={store.a_to_b} B->A={store.b_to_a} "
          f"| dropped={reader.frames_dropped} | rows={storage.rows_written}")
    # Shed and failed rows are silent otherwise, which is exactly the failure
    # mode Layer A exists to remove. Printed only when non-zero so a clean run
    # stays quiet.
    if storage.rows_dropped:
        print(f"[done] write queue shed {storage.rows_dropped} rows "
              f"({storage.dropped_by_table}) - the writer could not keep up")
    if storage.rows_failed:
        print(f"[done] WARNING: {storage.rows_failed} rows were LOST to "
              f"{storage.write_failures} failed write(s). "
              f"Last error: {storage.last_write_error}")
    return {"frames": analysed, "read": read_n, "df": df, "store": store,
            "db": st_cfg.get("path"), "run_id": run_id,
            "summaries": summaries}


def _frame_meta(camera: str, frame_no: int, timestamp: float, ctx, store,
                storage, reader) -> dict:
    """The live-view payload for one frame. METADATA ONLY.

    No pixels: the dashboard draws these boxes client-side. ~30 boxes at ~120
    bytes is about 4 KB, so at the hub's 8 Hz push rate a viewer costs ~32 KB/s
    - which is the whole reason the metadata option was chosen over streaming
    video. ctx.annotated still holds the drawn frame if an MJPEG endpoint is
    ever wanted, so that stays a small addition rather than a rewrite.

    `health` is here rather than on a separate endpoint because these are the
    numbers that go stale fastest, and a feed silently degrading is the thing
    an operator most needs to see next to the boxes.
    """
    boxes = []
    for det in ctx.detections:
        if det.track_id is None:
            continue
        boxes.append({"track_id": det.track_id,
                      "object_id": store.object_ids.get(det.track_id),
                      "vehicle_id": store.vehicle_of.get(det.track_id),
                      "cls": det.cls_name, "group": det.group,
                      "conf": round(float(det.conf), 3),
                      "xyxy": [int(v) for v in det.bbox],
                      "lane_id": det.lane_id or None,
                      "lane_flag": det.lane_flag or None,
                      "plate": det.extra.get("plate_number") or None})
    return {"camera": camera, "frame_no": frame_no,
            "ts": round(float(timestamp), 3), "boxes": boxes,
            "counts": dict(store.lane_counts),
            "crossings": {"a_to_b": store.a_to_b, "b_to_a": store.b_to_a},
            "flagged": {"wrong_way": len(store.wrong_way_ids),
                        "wrong_lane": len(store.wrong_lane_ids)},
            "congestion": store.__dict__.get("congestion_state"),
            "health": {"queue_depth": storage._q.qsize(),
                       "rows_dropped": storage.rows_dropped,
                       "rows_failed": storage.rows_failed,
                       "write_alarm": storage.write_alarm,
                       "frames_dropped": getattr(reader, "frames_dropped", 0),
                       "tracked": len(store.vehicles),
                       "state_size": store.state_size()}}


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
              min_conf: float = 0.7, require_format: bool = True,
              on_retire=None, camera: str = "", run_id=None):
    """Persist an object row plus its attributes, then let plugins forget it.

    This is where identity becomes authoritative. Any binding made mid-track
    used a running consensus; by the time a track is retired the vote is
    complete, so the final plate is re-resolved and that is the vehicle_id
    written to the row.

    It is also the incident layer's firing point, which is why `on_retire`
    exists. Everything a complete payload needs - the voted plate, its
    confidence, the chosen crop, the sighting's real bounds - is settled HERE
    and nowhere earlier. The cost is latency: a track is not retired until
    ttl_for() seconds after it was last seen, so a webhook lands roughly
    time-in-frame + 5s after the event. That is the price of a payload that
    carries a plate rather than a null, and the dashboard already serves
    anyone who needs to know sooner.
    """
    oid = store.object_ids.get(tid)
    if oid is None:
        return
    attrs = vehicle.get("attrs", {}) or {}
    vehicle_id = _settled_vehicle_id(storage, identity, store, tid, vehicle,
                                     attrs, min_conf, require_format)
    storage.upsert_object({
        "id": oid, "track_id": tid, "vehicle_id": vehicle_id,
        **({} if run_id is None else {"run_id": run_id}),
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
    if on_retire is not None:
        # A synthetic event rather than a second callback: the incident layer
        # already has to handle ctx.events, and a retirement is just one more
        # kind of thing that happened. detail carries the settled facts so the
        # subscriber never has to reach back into TrackStore, which is about to
        # be emptied of this track.
        _safely(on_retire,
                {"kind": "track_retired", "track_id": tid, "ts": ts,
                 "detail": {
                     "object_id": oid, "vehicle_id": vehicle_id,
                     "plate": attrs.get("plate_number") or None,
                     "plate_conf": _attr_conf(attrs, "plate_number"),
                     "cls_name": vehicle.get("vehicle_class", ""),
                     "colour": attrs.get("color") or None,
                     "first_seen_s": vehicle.get("first_seen_s", 0.0),
                     "last_seen_s": vehicle.get("last_seen_s", 0.0),
                     "frames_seen": vehicle.get("frames_seen", 0),
                     "crop_path": vehicle.get("_crop_path"),
                     "lane_id": store.lane_of.get(tid),
                     "lane_flag": store.lane_flag_of.get(tid)}},
                {"camera": camera, "object_id": oid, "timestamp": ts},
                what="on_retire")
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
