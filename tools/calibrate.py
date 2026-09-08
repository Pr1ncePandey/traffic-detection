"""Calibrate a new camera: suggest zones + lane split + dominant direction from motion.

Usage:
  python tools/calibrate.py                              # uses config.yaml video source
  python tools/calibrate.py --source some.mp4 --max-frames 600 --out cameras/new.yaml
  python tools/calibrate.py --suggest-lanes --source other.mp4   # + lane polygons

How it works (simple English):
  1. Runs the same YOLO + ByteTrack on the first N frames.
  2. For each vehicle ID, records start centroid -> end centroid + mean x.
  3. dy < 0 = moved up (bottom->top). dy > 0 = moved down.
  4. Majority vote = dominant flow. With --suggest-lanes: splits IDs by mean x
     (left half vs right half), votes direction per half, prints two lane blocks.
     Works for 2 separate roads AND 1 road with 2 ways (same format).

No code change per video: save the printed block as cameras/<name>.yaml,
then run: python main.py --camera <name>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from src.config import load
from src.detectors.yolo import VehicleDetector


def parse_args():
    p = argparse.ArgumentParser(description="Suggest camera zones + direction from motion")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--out", default=None, help="write suggested cameras/<name>.yaml")
    p.add_argument("--suggest-lanes", action="store_true",
                   help="also suggest left/right lane polygons + directions from motion")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load(args.config)
    src = args.source or cfg["video"]["source"]
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = VehicleDetector(cfg["model"])
    tracks = {}  # tid -> [first_y, last_y, first_x, last_x, n]
    n = 0
    while n < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        res = detector.track(frame, cfg["tracker"])
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if tid not in tracks:
                    tracks[tid] = [cy, cy, cx, cx, 1]
                else:
                    tracks[tid][1] = cy
                    tracks[tid][3] = cx
                    tracks[tid][4] += 1
    cap.release()

    ups = downs = 0
    for tid, (y0, y1, x0, x1, frames) in tracks.items():
        if frames < 5:
            continue  # ignore flicker tracks
        if y1 < y0 - 20:
            ups += 1
        elif y1 > y0 + 20:
            downs += 1
    total = ups + downs
    print(f"[calibrate] {len(tracks)} IDs, {total} with clear motion: up(bottom->top)={ups} down={downs}")
    if total == 0:
        print("Not enough motion. Try --max-frames 1200.")
        return
    if ups >= downs:
        dom, la, lb = "bottom -> top", "bottom (entry)", "top (exit)"
    else:
        dom, la, lb = "top -> bottom", "top (entry)", "bottom (exit)"
    print(f"[calibrate] dominant flow: {dom} ({max(ups, downs)}/{total})")
    snippet = (f"camera:\n  name: \"new_camera\"\n  line_a_ratio: 0.35\n"
               f"  line_b_ratio: 0.65\n  label_a: \"{la}\"\n  label_b: \"{lb}\"\n"
               f"  color_a: [255, 0, 0]\n  color_b: [0, 255, 255]\n")
    print("--- suggested cameras/<name>.yaml ---\n" + snippet)
    if args.suggest_lanes:
        left_up = left_down = right_up = right_down = 0
        for tid, (y0, y1, x0, x1, frames) in tracks.items():
            if frames < 5:
                continue
            xm = (x0 + x1) / 2 / max(W, 1)
            moved_up = y1 < y0 - 20
            moved_down = y1 > y0 + 20
            if xm < 0.5:
                if moved_up:
                    left_up += 1
                elif moved_down:
                    left_down += 1
            else:
                if moved_up:
                    right_up += 1
                elif moved_down:
                    right_down += 1
        left_dir = "down" if left_down >= left_up else "up"
        right_dir = "down" if right_down >= right_up else "up"
        print(f"[lanes] left half: up={left_up} down={left_down} -> {left_dir} | "
              f"right half: up={right_up} down={right_down} -> {right_dir}")
        lane_block = (
            f"\nlanes_mode: \"explicit\"\nlanes:\n"
            f"  - name: \"left\"\n    polygon: [[0, 0], [0.48, 0], [0.48, 1], [0, 1]]\n"
            f"    direction: \"{left_dir}\"\n    allowed: []\n"
            f"  - name: \"right\"\n    polygon: [[0.52, 0], [1.0, 0], [1.0, 1], [0.52, 1]]\n"
            f"    direction: \"{right_dir}\"\n    allowed: []\n")
        print("--- suggested lanes (paste under camera block) ---" + lane_block)
        print("Note: halves guess (0.48/0.52). If the divider is DIAGONAL in perspective "
              "(like our demo fence), use tools/draw_lanes.py to click the real corners.")
        snippet += lane_block
    print("Tip: lines at 0.35/0.65 cut most lanes. Move them if a lane sits outside the band.")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Auto-suggested by tools/calibrate.py - confirm once per CCTV.\n" + snippet)
        print(f"Saved to {args.out}. Run: python main.py --camera <name>")


if __name__ == "__main__":
    main()
