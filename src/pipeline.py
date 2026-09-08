"""Main pipeline: video -> detect -> track -> attributes -> analysis -> storage.

Parallelism: vehicle rows go to CsvStore via a background-thread queue while
frame analytics (counts, congestion stub) runs inline per frame and frame
snapshots (raw + annotated) are saved every N frames. Same output as before.
"""

import os
import time

import cv2
import pandas as pd
from tqdm import tqdm

from .analysis.lanes import LaneChecker
from .analysis.line_counter import LineCounter, ZoneCounter, check_lanes
from .attributes.base import configure_plate, run_attributes
from .detectors.yolo import VehicleDetector
from .storage.csv_store import CsvStore
from .trackers.store import TrackStore


def run_pipeline(cfg: dict) -> dict:
    src = cfg["video"]["source"]
    if not os.path.exists(src):
        raise FileNotFoundError(f"Input video not found: {src}")

    os.makedirs(os.path.dirname(cfg["video"]["target"]) or ".", exist_ok=True)
    os.makedirs("outputs/crops", exist_ok=True)

    detector = VehicleDetector(cfg["model"])
    store = TrackStore()
    pcfg = cfg.get("plate", {})
    configure_plate(det_conf=float(pcfg.get("det_conf", 0.4)),
                    min_conf=float(pcfg.get("min_conf", 0.5)),
                    det_imgsz=int(pcfg.get("det_imgsz", 480)))
    out_cfg = cfg["outputs"]
    frame_dir = out_cfg.get("frame_dir", "outputs/frames")
    save_frames = bool(out_cfg.get("save_raw_frames", False))
    every = int(out_cfg.get("save_frame_every", 30))
    if save_frames:
        os.makedirs(f"{frame_dir}/raw", exist_ok=True)
        os.makedirs(f"{frame_dir}/annotated", exist_ok=True)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(cfg["processing"].get("max_frames", -1))
    frame_skip = max(1, int(cfg["processing"].get("frame_skip", 1)))
    if max_frames > 0:
        total = min(total, max_frames // frame_skip + 1)

    line = LineCounter(cfg.get("counting_line", {}), H)
    zone = ZoneCounter(cfg.get("camera", {}), H, legacy=cfg.get("counting_line", {}))
    # --no-line / enabled:false hides both lines.
    if not line.enabled:
        zone.enabled = False
    lanes_mode = str(cfg.get("lanes_mode", "explicit")).lower()
    checker = LaneChecker(cfg.get("lanes", []), W, H, mode=lanes_mode)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(cfg["video"]["target"], fourcc, fps, (W, H))
    csv = CsvStore(cfg["video"]["csv"])

    cam = cfg.get("camera", {})
    print(f"[pipeline] {W}x{H} @ {fps:.1f}fps | model={cfg['model']['name']} "
          f"conf={cfg['model']['conf']} device={cfg['model'].get('device', 'cpu')} | "
          f"zones A@{zone.b_ratio:.2f}({cam.get('label_a', 'entry')}) "
          f"B@{zone.a_ratio:.2f}({cam.get('label_b', 'exit')}) "
          f"{'on' if zone.enabled else 'off'} | "
          f"lanes={lanes_mode}({len(checker.lanes)}): " +
          ", ".join(f"{L['name']}:{L['direction']}" for L in checker.lanes))

    start = time.time()
    frame_no, seen = 0, 0
    pbar = tqdm(total=total if total > 0 else None, desc="Processing")
    while True:
        ret, raw = cap.read()
        if ret:
            frame_no += 1
        if not ret:
            break
        if max_frames > 0 and frame_no > max_frames:
            break
        if (frame_no - 1) % frame_skip != 0:
            continue
        seen += 1
        timestamp = frame_no / fps
        frame = raw.copy()

        result = detector.track(frame, cfg["tracker"])
        frame = checker.draw(frame, dict(store.lane_counts))
        frame = zone.draw(frame, store.a_to_b, store.b_to_a)

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            clss = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for box, tid, cls, conf in zip(boxes, ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                name = VehicleDetector.class_name(cls)
                prev_cx, prev_cy = store.observe(tid, cx, cy)
                event = store.zone_crossing(tid, cy, zone.line_a_y, zone.line_b_y, zone.enabled)
                store.touch(tid, name, timestamp)
                lane_id, lane_flag = checker.check(cx, cy, prev_cx, prev_cy, name, tid)
                store.set_lane(tid, lane_id, lane_flag)

                crop = raw[max(0, y1):y2, max(0, x1):x2]
                vehicle = store.vehicles[tid]
                if crop.size > 0:
                    run_attributes(cfg["attributes"].get("enabled", []), crop, vehicle)
                    if (out_cfg.get("save_crops", True) and tid not in store.saved_crops
                            and conf > float(out_cfg.get("crop_min_conf", 0.5))):
                        cv2.imwrite(f"outputs/crops/vehicle_{tid}.jpg", crop)
                        store.saved_crops.add(tid)

                csv.put({"frame": frame_no, "time_s": round(timestamp, 2), "object_id": tid,
                         "vehicle_class": name, "confidence": round(float(conf), 3),
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2, "event": event,
                         "lane_id": lane_id, "lane_flag": lane_flag,
                         "plate_number": vehicle["attrs"].get("plate_number", ""),
                         "plate_conf": vehicle["attrs"].get("plate_conf", 0.0),
                         "color": vehicle["attrs"].get("color", "")})
                box_color = (0, 255, 0)
                if "wrong_way" in lane_flag:
                    box_color = (0, 0, 255)      # red = going opposite
                elif "wrong_lane" in lane_flag:
                    box_color = (0, 165, 255)    # orange = wrong lane type
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                tag = f"#{tid} {name} {conf:.2f}"
                if lane_id:
                    tag += f" {lane_id}:{lane_flag}"
                if vehicle["attrs"].get("plate_number"):
                    tag += f" {vehicle['attrs']['plate_number']}"
                cv2.putText(frame, tag, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        if save_frames and frame_no % every == 0:
            cv2.imwrite(f"{frame_dir}/raw/frame_{frame_no:06d}.jpg", raw)
            cv2.imwrite(f"{frame_dir}/annotated/frame_{frame_no:06d}.jpg", frame)
        out.write(frame)
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()
    df = csv.close()

    elapsed = time.time() - start
    cam = cfg.get("camera", {})
    with open(cfg["video"]["summary"], "w") as f:
        f.write("TRAFFIC PROTOTYPE REPORT\n======================\n")
        f.write(f"Source video : {src}\n")
        f.write(f"Camera       : {cam.get('name', 'demo')} "
                f"({cam.get('label_a', 'entry')} -> {cam.get('label_b', 'exit')})\n")
        f.write(f"Frames       : {frame_no} in {elapsed:.1f}s ({frame_no/elapsed:.1f} fps)\n")
        f.write(f"Unique vehicles (object IDs): {len(store.vehicles)}\n")
        f.write(f"A->B ({cam.get('label_a', 'entry')} to {cam.get('label_b', 'exit')}): {store.a_to_b}\n")
        f.write(f"B->A ({cam.get('label_b', 'exit')} to {cam.get('label_a', 'entry')}): {store.b_to_a}\n")
        f.write(f"Legacy IN: {store.in_count}, OUT: {store.out_count}\n")
        f.write(f"\nPer-lane unique vehicles:\n")
        for L in checker.lanes:
            f.write(f"  {L['name']} ({L['direction']}): {store.lane_counts.get(L['name'], 0)}\n")
        f.write(f"\nWrong-WAY IDs (opposite direction): {sorted(store.wrong_way_ids) or 'none'}\n")
        f.write(f"Wrong-LANE IDs (wrong vehicle type): {sorted(store.wrong_lane_ids) or 'none'}\n")
        f.write(f"\nPer-class count:\n")
        for k, v in store.per_class_counts().items():
            f.write(f"  {k}: {v}\n")

    print(f"[done] {elapsed:.1f}s | unique={len(store.vehicles)} "
          f"A->B={store.a_to_b} B->A={store.b_to_a} (IN={store.in_count} OUT={store.out_count}) "
          f"| lanes={dict(store.lane_counts)} "
          f"| wrong_way={sorted(store.wrong_way_ids) or 'none'} "
          f"| crops={len(store.saved_crops)}")
    return {"frames": frame_no, "unique": len(store.vehicles), "df": df, "store": store}
