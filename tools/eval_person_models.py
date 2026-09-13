"""Side-by-side test: PP-Human vs SegFormer garments vs MiVOLO age/gender.

    python tools/eval_person_models.py --video samples/person_test2.mp4 \
        --csv outputs/person_attr_eval_person_test2/tracks_umb.csv \
        --out outputs/person_models_person_test2

Uses the tracks CSV of an earlier run (people, plus umbrellas when present),
the crop rules from config.yaml (perception.attributes.person), and only people
with >= min_reads usable crops - so all three models describe the SAME people
from the SAME crops:

    PP-Human   every kept crop (up to max_reads), as the `person` enricher does
    SegFormer  3 crops spread along the track
    MiVOLO     the same 3 crops, body only

Writes sheet.jpg (crop | garment mask | three label blocks), results.csv and
timing.txt. No ground truth: judge by looking.
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

from src.attributes.enrichers import age_gender as ag  # noqa: E402
from src.attributes.enrichers import garments as gm  # noqa: E402
from src.attributes.enrichers import person as pp  # noqa: E402
from src.attributes.mivolo_loader import load_mivolo, predict  # noqa: E402
from src.attributes.registry import block  # noqa: E402
from src.config import load_for_camera  # noqa: E402

# Colours for the garment mask (BGR), one per ATR class.
PALETTE = np.array([
    [235, 235, 235], [0, 140, 255], [60, 60, 60], [255, 0, 255], [200, 90, 30],
    [0, 200, 200], [120, 40, 160], [0, 180, 0], [0, 0, 0], [90, 90, 200],
    [90, 90, 200], [150, 190, 240], [120, 170, 220], [120, 170, 220],
    [130, 180, 230], [130, 180, 230], [0, 0, 220], [0, 220, 120]], np.uint8)


def spread(items, k):
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heavy-reads", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    conf = block(load_for_camera(None), "person")
    rules = pp.CropRules(conf)
    max_reads, read_every = int(conf.get("max_reads", 8)), int(conf.get("read_every", 3))
    min_reads = int(conf.get("min_reads", 3))
    keys = tuple(conf.get("keys") or pp.DEFAULT_STORE)
    thresholds = dict(conf.get("thresholds") or {})

    df = pd.read_csv(args.csv)
    people = df[df.cls_group == "person"]
    umb = df[df.vehicle_class == "umbrella"]
    by_frame = {f: g for f, g in people.groupby("frame")}
    umb_by = {f: [tuple(int(v) for v in r) for r in g[["x1", "y1", "x2", "y2"]]
                  .itertuples(index=False)] for f, g in umb.groupby("frame")}

    # One pass over the video: keep the crops the person enricher would read.
    cap = cv2.VideoCapture(args.video)
    fw, fh = int(cap.get(3)), int(cap.get(4))
    crops, seen, frame_no, last = {}, {}, 0, max(by_frame)
    while frame_no <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for r in (by_frame[frame_no].itertuples() if frame_no in by_frame else ()):
            oid = int(r.object_id)
            if len(crops.get(oid, ())) >= max_reads:
                continue
            bbox = (int(r.x1), int(r.y1), int(r.x2), int(r.y2))
            if rules.reject(bbox, float(r.confidence), fw, fh, umb_by.get(frame_no, ())):
                continue
            seen[oid] = seen.get(oid, 0) + 1
            if (seen[oid] - 1) % read_every:
                continue
            crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if crop.size:
                crops.setdefault(oid, []).append(crop.copy())
        frame_no += 1
    cap.release()
    people_ids = [o for o, c in crops.items() if len(c) >= min_reads]
    people_ids.sort(key=lambda o: -max(c.shape[0] for c in crops[o]))
    print(f"{len(people_ids)} people with >= {min_reads} usable crops")

    t_pp, t_seg, t_mv = [], [], []
    paddle = pp.PersonAttrModel(conf.get("model", "models/person_attr/person_attr.onnx"))
    seg = gm.GarmentSegModel("models/clothes_seg/segformer_b2_clothes.onnx", args.threads)
    model, proc, mcfg = load_mivolo(args.threads)

    rows, tiles = [], []
    for n, oid in enumerate(people_ids, 1):
        cs = crops[oid]
        probs = []
        for c in cs:
            t = time.perf_counter(); probs.append(paddle.probs(c))
            t_pp.append((time.perf_counter() - t) * 1000)
        p_read = pp.decode(np.mean(probs, axis=0), thresholds)

        heavy = spread(cs, args.heavy_reads)
        evs, best_map = [], None
        for c in heavy:
            t = time.perf_counter(); lm = seg.label_map(c)
            t_seg.append((time.perf_counter() - t) * 1000)
            evs.append(gm.measure(lm))
            if best_map is None or c.shape[0] >= best_map[0].shape[0]:
                best_map = (c, lm)
        g_read = gm.decode(np.mean(evs, axis=0))

        t = time.perf_counter()
        m_read = ag.summarise(predict(model, proc, mcfg, heavy))
        t_mv.append((time.perf_counter() - t) * 1000 / len(heavy))

        rows.append({"object_id": oid, "crops": len(cs), "max_h": max(c.shape[0] for c in cs),
                     **{f"paddle_{k}": p_read[k][0] for k in keys},
                     **{f"seg_{k}": v for k, (v, _) in g_read.items()},
                     **{f"seg_{k}_conf": round(float(c), 2) for k, (_, c) in g_read.items()},
                     **{f"mivolo_{k}": v for k, v in m_read.items()}})
        tiles.append(tile(oid, *best_map, p_read, keys, g_read, m_read))
        if n % 20 == 0:
            print(f"  {n}/{len(people_ids)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    cols = 4
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(os.path.join(args.out, "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def ms(v):
        return f"median {np.median(v):.0f} ms, p95 {np.percentile(v, 95):.0f} ms"
    report = (f"people {len(rows)}\n"
              f"PP-Human  per crop: {ms(t_pp)} (1 thread)\n"
              f"SegFormer per crop: {ms(t_seg)} ({args.threads} threads)\n"
              f"MiVOLO    per crop: {ms(t_mv)} ({args.threads} threads, batched per person)")
    open(os.path.join(args.out, "timing.txt"), "w").write(report + "\n")
    print(report)


def tile(oid, crop, label_map, p_read, keys, g_read, m_read, w=420, h=300):
    canvas = np.full((h, w, 3), 255, np.uint8)
    ch, cw = crop.shape[:2]
    s = min(96 / cw, (h - 28) / ch)
    size = (max(1, int(cw * s)), max(1, int(ch * s)))
    canvas[22:22 + size[1], 4:4 + size[0]] = cv2.resize(crop, size)
    mask = cv2.resize(PALETTE[label_map], size, interpolation=cv2.INTER_NEAREST)
    canvas[22:22 + size[1], 104:104 + size[0]] = mask
    font, x = cv2.FONT_HERSHEY_SIMPLEX, 206
    cv2.putText(canvas, f"#{oid}", (4, 15), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    def block_(title, lines, y, colour):
        cv2.putText(canvas, title, (x, y), font, 0.4, colour, 1, cv2.LINE_AA)
        y += 15
        for text, weak in lines:
            cv2.putText(canvas, text, (x + 4, y), font, 0.36,
                        (0, 0, 200) if weak else (25, 25, 25), 1, cv2.LINE_AA)
            y += 14
        return y + 6

    y = block_("MiVOLO", [(f"{m_read['gender']} {m_read['gender_conf']:.2f}",
                           m_read["gender_conf"] < 0.6),
                          (f"age {m_read['age_years']}  +/-{m_read['age_spread']}", False)],
               16, (120, 60, 0))
    y = block_("SegFormer", [(f"{k} {v} {c:.2f}", c < 0.6) for k, (v, c) in g_read.items()
                             if not (k in ("bag", "sunglasses") and v in ("no", "unknown"))],
               y, (0, 110, 0))
    block_("PP-Human", [(f"{k} {p_read[k][0]} {p_read[k][1]:.2f}", p_read[k][1] < 0.6)
                        for k in keys if p_read[k][0] not in ("no", "none")],
           y, (140, 0, 140))
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (200, 200, 200), 1)
    return canvas


if __name__ == "__main__":
    main()
