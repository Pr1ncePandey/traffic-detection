"""Eyeball test for the person-attribute model on real footage.

    python tools/eval_person_attr.py
    python tools/eval_person_attr.py --video samples/short/person_test.mp4 \
        --csv outputs/person_attr_eval_person_test/tracks.csv \
        --out outputs/person_attr_eval_person_test

Takes the boxes a previous run wrote to a tracks CSV (people, plus umbrellas
when present), cuts the crops out of the video and runs the model with the
SAME crop rules, minimum reads and stored keys as the `person` enricher - read
from config.yaml, so this measures what actually runs. Writes:

    <out>/sheet.jpg     each described person: sharpest crop + stored labels
    <out>/results.csv   one row per track, label + conf
    <out>/timing.txt    ms per crop on this machine, and why crops were skipped

--all-keys shows every decoded attribute, including the ones not stored.
There is no ground truth, so accuracy is judged by looking.
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attributes.enrichers.person import (KEYS, UMBRELLA, CropRules,  # noqa: E402
                                             PersonAttrModel, decode)
from src.attributes.registry import block  # noqa: E402
from src.config import load_for_camera  # noqa: E402

SHORT = {"gender": "", "age_group": "age ", "facing": "", "sleeves": "sleeve ",
         "lower": "", "bag": "bag ", "hat": "hat ", "glasses": "glasses ",
         "holding": "holding ", "long_coat": "coat ", "upper_pattern": "top ",
         "lower_pattern": "bottom ", "boots": "boots "}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="samples/short/indian_road.mp4")
    ap.add_argument("--csv", default="outputs/tracks_v2.csv")
    ap.add_argument("--out", default="outputs/person_attr_eval")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--all-keys", action="store_true",
                    help="show every decoded attribute, not just the stored ones")
    args = ap.parse_args()

    conf = block(load_for_camera(None, args.config), "person")
    rules = CropRules(conf)
    max_reads = int(conf.get("max_reads", 8))
    read_every = max(1, int(conf.get("read_every", 3)))
    min_reads = int(conf.get("min_reads", 3))
    thresholds = dict(conf.get("thresholds") or {})
    keys = KEYS if args.all_keys else tuple(conf.get("keys") or KEYS)
    model = PersonAttrModel(conf.get("model", "models/person_attr/person_attr.onnx"))

    df = pd.read_csv(args.csv)
    name = df["cls_name"] if "cls_name" in df else df["cls_group"]
    people = df[df.cls_group == "person"]
    umbrellas = df[name == UMBRELLA]
    by_frame = {f: g for f, g in people.groupby("frame")}
    umb_by_frame = {f: [tuple(int(v) for v in r) for r in
                        g[["x1", "y1", "x2", "y2"]].itertuples(index=False)]
                    for f, g in umbrellas.groupby("frame")}
    print(f"{people.object_id.nunique()} person tracks, "
          f"{len(umbrellas)} umbrella boxes")

    cap = cv2.VideoCapture(args.video)
    fw, fh = int(cap.get(3)), int(cap.get(4))
    sums, counts, seen, best, skipped, times = {}, {}, {}, {}, {}, []
    frame_no, last = 0, max(by_frame) if by_frame else -1
    while frame_no <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for r in (by_frame[frame_no].itertuples() if frame_no in by_frame else ()):
            oid = int(r.object_id)
            if counts.get(oid, 0) >= max_reads:
                continue
            bbox = (int(r.x1), int(r.y1), int(r.x2), int(r.y2))
            why = rules.reject(bbox, float(r.confidence), fw, fh,
                               umb_by_frame.get(frame_no, ()))
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            seen[oid] = seen.get(oid, 0) + 1
            if (seen[oid] - 1) % read_every:
                continue
            crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if crop.size == 0:
                continue
            t = time.perf_counter()
            p = model.probs(crop)
            times.append((time.perf_counter() - t) * 1000)
            sums[oid] = sums.get(oid, 0.0) + p
            counts[oid] = counts.get(oid, 0) + 1
            h = bbox[3] - bbox[1]
            if oid not in best or h > best[oid][0]:
                best[oid] = (h, crop.copy())
        frame_no += 1
    cap.release()

    described = [o for o in sums if counts[o] >= min_reads]
    os.makedirs(args.out, exist_ok=True)
    rows, tiles = [], []
    for oid in sorted(described, key=lambda o: -best[o][0]):
        read = decode(sums[oid] / counts[oid], thresholds)
        rows.append({"object_id": oid, "reads": counts[oid], "max_h": int(best[oid][0]),
                     **{k: read[k][0] for k in keys},
                     **{k + "_conf": round(read[k][1], 2) for k in keys}})
        tiles.append(tile(oid, best[oid][1], read, counts[oid], keys))

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["object_id"])
        w.writeheader()
        w.writerows(rows)
    if tiles:
        cols = 6
        tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
        sheet = np.vstack([np.hstack(tiles[i:i + cols])
                           for i in range(0, len(tiles), cols)])
        cv2.imwrite(os.path.join(args.out, "sheet.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    ms = np.array(times) if times else np.array([0.0])
    report = (f"crops {len(times)}  mean {ms.mean():.2f} ms  median "
              f"{np.median(ms):.2f} ms  p95 {np.percentile(ms, 95):.2f} ms (1 thread)\n"
              f"tracks read {len(sums)}  described (>= {min_reads} reads) {len(described)}\n"
              f"crops skipped: {dict(sorted(skipped.items()))}")
    open(os.path.join(args.out, "timing.txt"), "w").write(report + "\n")
    print(report)
    print(f"wrote {args.out}/sheet.jpg and results.csv ({len(rows)} people)")


def tile(oid, crop, read, n, keys, w=230, h=330):
    """Crop on the left, labels on the right, fixed size for the grid."""
    canvas = np.full((h, w, 3), 255, np.uint8)
    ch, cw = crop.shape[:2]
    scale = min(96 / cw, (h - 30) / ch)
    small = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
    canvas[24:24 + small.shape[0], 4:4 + small.shape[1]] = small
    cv2.putText(canvas, f"#{oid}  n={n}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 1, cv2.LINE_AA)
    y = 36
    for key in keys:
        value, conf = read[key]
        if value in ("no", "plain", "none") and key not in ("facing", "sleeves", "lower"):
            continue                                  # only show what is present
        cv2.putText(canvas, f"{SHORT[key]}{value} {conf:.2f}", (104, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 200) if conf < 0.6 else (20, 20, 20), 1, cv2.LINE_AA)
        y += 17
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (200, 200, 200), 1)
    return canvas


if __name__ == "__main__":
    main()
