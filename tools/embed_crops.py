"""Embed object crops for open-vocabulary search. Offline, resumable.

    python tools/embed_crops.py --db outputs/traffic.db
    python tools/embed_crops.py --camera indian_road --limit 200   # smoke test
    python tools/embed_crops.py --dry-run                          # count only

NEVER RUNS INLINE IN THE PIPELINE. No real-time decision needs an embedding,
and keeping inference out of pipeline.py means swapping the model is a re-run
of this script rather than a change to the pipeline. It writes through the
existing SqliteStore queue, so the single-writer invariant holds.

RESUMABLE AND RE-RUNNABLE. It selects objects with a crop but no row in
`embeddings` for this model, so an interrupted run continues where it stopped.
`--force` re-embeds rows that already exist (for a re-exported model or
changed preprocessing); the upsert replaces rather than duplicating.

THREE FILTERS, EACH ONE A MEASURED FAILURE MODE

  --min-px 48    short side. 24-51% of person crops fall below 48 px, and
                 CLIP upscales them into 224 regardless.
  --min-conf 0.5 detection confidence OF THE CROP'S OWN FRAME. 35-51% of
                 crops come from sub-threshold detections.
  --max-area .33 box area as a fraction of frame. A near-full-frame box is a
                 picture of the whole scene, not an object: on indian_road two
                 such boxes were labelled `giraffe` at conf 0.51-0.55, so a
                 confidence floor does NOT catch them, and they would embed as
                 confident garbage matching almost any query. The largest
                 genuine object measured 24.8% of frame, so 33% sits in a
                 clean gap - RE-CHECK IT per camera, because a camera mounted
                 closer to the road has legitimately larger objects.

Every skip is counted and printed. A silent skip would hide the thing these
filters exist to measure.

WHERE crop_conf COMES FROM. Nothing stores it: `_keep_best_crop` never records
which frame the kept crop came from, and `objects.best_conf` is the track
MAXIMUM. It is recovered by geometry - `src/detectors/yolo.py` clips boxes
before storing them, so a saved crop's on-disk (w, h) equals some detections
row's (x2-x1, y2-y1) for that object. Measured over 195 crops: 0 failed to
match, 88% matched exactly one confidence, 12% matched several sharing that
box size (the max is taken).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.query.clip_onnx import DEFAULT_MODEL, SPECS, ClipOnnx
from src.query.embed import (BATCH, MAX_AREA, MIN_CONF, MIN_PX,
                             CropEmbedder, candidates, new_skips)  # noqa: E402
from src.storage.sqlite_store import SqliteStore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default="outputs/traffic.db")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SPECS))
    p.add_argument("--camera", default=None, help="only this camera's runs")
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--min-px", type=int, default=48,
                   help="skip crops whose short side is below this")
    p.add_argument("--min-conf", type=float, default=0.5,
                   help="skip crops from detections below this confidence")
    p.add_argument("--max-area", type=float, default=0.33,
                   help="skip boxes covering more than this fraction of frame")
    p.add_argument("--force", action="store_true",
                   help="re-embed objects that already have a vector")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be embedded, write nothing")
    return p.parse_args()


def main():
    a = parse_args()
    import cv2

    enc = ClipOnnx(a.model)
    missing = enc.missing()
    if missing and not a.dry_run:
        print(f"[embed] model files missing for {a.model}:")
        for m in missing:
            print(f"          {m}")
        print(f"[embed] fetch them with: python tools/fetch_clip.py "
              f"--model {a.model}")
        return 2

    if not os.path.exists(a.db):
        print(f"[embed] no database at {a.db}")
        return 2

    store = None if a.dry_run else SqliteStore(a.db)
    conn = (store.connect_ro() if store else __import__("sqlite3").connect(a.db))
    conn.row_factory = __import__("sqlite3").Row

    rows = candidates(conn, a.model, a.camera, a.force, a.limit)
    if not rows:
        print(f"[embed] nothing to do: every crop already has a {a.model} "
              f"vector (use --force to re-embed)")
        if store:
            store.close(end_run=False)
        return 0

    # The eligibility rules and the batching live in src/query/embed.py, so
    # this tool and the server's background embedder cannot drift apart about
    # what belongs in an index.
    worker = CropEmbedder(enc, min_px=a.min_px, min_conf=a.min_conf,
                          max_area=a.max_area, batch=a.batch)
    skip = new_skips()
    t0 = time.time()
    result = worker.run(conn, rows, put=(store.put_embedding if store else None),
                        dry_run=a.dry_run, skips=skip)
    embedded = result["embedded"]
    t_infer = result["infer_s"]
    if store:
        store.flush()

    wall = time.time() - t0
    total_skipped = sum(skip.values())
    verb = "would embed" if a.dry_run else "embedded"
    print(f"[embed] {verb} {embedded} crops with {a.model} "
          f"(dim {enc.spec.dim}) from {len(rows)} candidates")
    if total_skipped:
        detail = ", ".join(f"{k}={v}" for k, v in skip.items() if v)
        print(f"[embed] skipped {total_skipped}: {detail}")
    if embedded and not a.dry_run:
        print(f"[embed] {1000 * t_infer / embedded:.1f} ms/crop inference, "
              f"{wall:.1f}s wall, "
              f"{embedded / wall:.1f} crops/s end to end")
    if store:
        h = store.health()
        if h["rows_failed"] or h["write_alarm"]:
            print(f"[embed] WRITE FAILURES: {h['rows_failed']} rows lost, "
                  f"last error: {h['last_write_error']}")
        store.close(end_run=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
