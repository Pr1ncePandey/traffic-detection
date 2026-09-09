"""Compare OCR backends on the same plate crops, against hand-labelled truth.

Two steps, because this repo ships no ground truth and none can be invented:

  # 1. cut plate crops out of the vehicle crops and write a label template
  python tools/bench_ocr.py --make-gt

  #    ...now open samples/plate_gt.csv and type what you can read in each
  #    image under outputs/plates/. Leave a row blank to skip it.

  # 2. score every installed backend against those labels
  python tools/bench_ocr.py

Benchmarking saved plate crops (rather than re-running the video) keeps this
loop fast and isolates OCR quality from detection quality. It also sidesteps
base.py's max-3-tries-per-vehicle rate limit, which would otherwise cap how
much OCR a single run can exercise.
"""

import argparse
import csv
import glob
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attributes import ocr_engines, plate_format  # noqa: E402
from src.attributes.plate import configure, detect_plate_box, read_plate_image  # noqa: E402
from src.config import load  # noqa: E402

GT_PATH = os.path.join("samples", "plate_gt.csv")
# Plate crops live beside their labels, not in disposable outputs/, so the
# fixture stays self-contained and re-runnable after outputs/ is wiped.
PLATE_DIR = os.path.join("samples", "plates")


def parse_args():
    p = argparse.ArgumentParser("Compare OCR backends on plate crops")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--make-gt", action="store_true",
                   help="cut plate crops from vehicle crops and write a label template")
    p.add_argument("--crops", default="outputs/crops/vehicle_*.jpg",
                   help="glob of vehicle crops to cut plates from")
    p.add_argument("--gt", default=GT_PATH, help="labelled CSV (crop,plate)")
    p.add_argument("--backends", default=None,
                   help="comma-separated subset, e.g. rapidocr,fast_plate")
    p.add_argument("--limit", type=int, default=0, help="cap crops processed")
    return p.parse_args()


def make_gt(args, cfg):
    """Detect the plate inside each vehicle crop and save it for labelling."""
    configure(**cfg.get("plate", {}))
    os.makedirs(PLATE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(GT_PATH) or ".", exist_ok=True)
    crops = sorted(glob.glob(args.crops))
    if not crops:
        print(f"No crops matched {args.crops!r}. Run main.py first with "
              f"attributes.enabled including \"plate\".")
        return
    if args.limit:
        crops = crops[:args.limit]
    rows, kept = [], 0
    for path in crops:
        img = cv2.imread(path)
        if img is None:
            continue
        plate = detect_plate_box(img)
        if plate is None:
            continue
        name = f"plate_{kept:03d}.jpg"
        cv2.imwrite(os.path.join(PLATE_DIR, name), plate)
        rows.append({"crop": name, "plate": ""})
        kept += 1
    with open(GT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "plate"])
        w.writeheader()
        w.writerows(rows)
    print(f"{kept} plate crop(s) -> {PLATE_DIR}/")
    print(f"Label template       -> {GT_PATH}")
    print("Fill in the 'plate' column, then re-run without --make-gt.")


def _levenshtein(a: str, b: str) -> int:
    """Edit distance. Inline to avoid a dependency for ~12 lines."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(samples):
    """Score the CURRENTLY configured backend via the real read path."""
    exact = edits = gt_chars = 0
    cer_each, confs, times = [], [], []
    for img, truth in samples:
        t0 = time.perf_counter()
        text, conf = read_plate_image(img)
        times.append((time.perf_counter() - t0) * 1000.0)
        d = _levenshtein(text, truth)
        exact += int(text == truth)
        edits += d
        gt_chars += len(truth)
        cer_each.append(d / max(1, len(truth)))
        confs.append(conf)
    n = len(samples)
    return {
        "n": n,
        "exact": exact,
        "exact_pct": 100.0 * exact / n if n else 0.0,
        "cer": edits / gt_chars if gt_chars else 0.0,
        "cer_macro": sum(cer_each) / n if n else 0.0,
        "conf": sum(confs) / n if n else 0.0,
        "ms": sorted(times)[n // 2] if n else 0.0,
    }


def main():
    args = parse_args()
    cfg = load(args.config)
    if args.make_gt:
        make_gt(args, cfg)
        return

    if not os.path.exists(args.gt):
        print(f"No ground truth at {args.gt}. Create it with --make-gt first.")
        return
    samples = []
    with open(args.gt, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth = plate_format.normalize(row.get("plate", ""))
            if not truth:
                continue  # unlabelled or unreadable -> not scoreable
            img = cv2.imread(os.path.join(PLATE_DIR, row["crop"]))
            if img is not None:
                samples.append((img, truth))
    if not samples:
        print(f"{args.gt} has no labelled rows yet. Fill in the 'plate' column.")
        return

    wanted = ([b.strip() for b in args.backends.split(",")]
              if args.backends else list(ocr_engines.BACKENDS))
    pcfg = cfg.get("plate", {})

    print(f"\n{len(samples)} labelled plate(s) from {args.gt}")
    print("NOTE: published accuracy figures for these backends come from "
          "different datasets and are\nNOT comparable to each other or to the "
          "numbers below. Only this table is comparable,\nand only on this "
          "ground truth. n is small - one flip moves exact% a lot.\n")
    header = f"{'backend':<14}{'correct':<9}{'n':>4}{'exact':>7}{'exact%':>8}" \
             f"{'CER':>8}{'CER_mac':>9}{'conf':>7}{'ms':>8}"
    print(header)
    print("-" * len(header))
    for name in wanted:
        if ocr_engines.get_engine(name, pcfg.get("ocr_model", ""),
                                  pcfg.get("join_rows", True)) is None:
            print(f"{name:<14}(skipped: not installed / unavailable)")
            continue
        for correct in (False, True):
            opts = dict(pcfg)
            opts.update(ocr_backend=name, format_correction=correct)
            configure(**opts)
            r = score(samples)
            print(f"{name:<14}{str(correct):<9}{r['n']:>4}{r['exact']:>7}"
                  f"{r['exact_pct']:>7.1f}%{r['cer']:>8.3f}"
                  f"{r['cer_macro']:>9.3f}{r['conf']:>7.2f}{r['ms']:>7.1f}ms")
    print()


if __name__ == "__main__":
    main()
