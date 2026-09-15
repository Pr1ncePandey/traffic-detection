"""Replay a finished run's detections through the behaviour analysis.

    python tools/replay_behaviour.py --db outputs/person_test/behaviour.db \
        --camera person_test --behaviour behaviour/person_test.yaml \
        [--sheet outputs/person_test/replay]

Detection on CPU is the slow part of a run (7 minutes for a 63 s clip); the
behaviour rules cost 0.6 ms a frame. Drawing a zone or a mask by eye takes a
few tries, so re-running YOLO for each try is the wrong loop. This reads the
boxes a run already stored (detections + frames tables) and feeds them through
BehaviourAnalyzer with the CURRENT zones file, in seconds.

What it cannot replay: boxes the run did not store, and anything that depends
on pixels. Track ids are the run's object ids, which is what the rules key on.

--sheet writes a contact sheet per event kind (and one for ignored riders)
from the source video, to check each decision by eye.
"""

import argparse
import collections
import contextlib
import io
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.behaviour import BehaviourAnalyzer  # noqa: E402
from src.config import apply_behaviour, load_for_camera  # noqa: E402
from src.detectors.classes import group_of  # noqa: E402
from src.runtime.context import AnalysisView, Detection, TrackedBox  # noqa: E402


class Source:
    def __init__(self, width, height, fps):
        self.width, self.height, self.fps = width, height, fps


def load(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    run = con.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    frames = con.execute("SELECT id, ts FROM frames WHERE run_id=? ORDER BY frame_no",
                         (run["id"],)).fetchall()
    boxes = collections.defaultdict(list)
    for r in con.execute("SELECT frame_id, object_id, x1, y1, x2, y2, conf, cls_name "
                         "FROM detections WHERE run_id=?", (run["id"],)):
        boxes[r["frame_id"]].append(r)
    return run, frames, boxes


def replay(cfg, run, frames, boxes):
    analyzer = BehaviourAnalyzer(cfg)
    analyzer.setup(Source(run["width"], run["height"], run["fps"]), cfg)
    events, riders = [], {}
    for frame in frames:
        tracked = []
        for r in boxes.get(frame["id"], ()):
            det = Detection(0, r["cls_name"], group_of(r["cls_name"]), float(r["conf"]),
                            (r["x1"], r["y1"], r["x2"], r["y2"]), track_id=r["object_id"])
            tracked.append(TrackedBox.of(det))
        found = analyzer.compute(AnalysisView(frame_no=frame["id"], timestamp=float(frame["ts"]),
                                              boxes=tuple(tracked), width=run["width"],
                                              height=run["height"]))
        events += [(k, d, t, float(frame["ts"])) for k, d, t in found.events]
        vehicles = analyzer.vehicles(tracked)
        for b in tracked:
            if (b.group == "person" and b.track_id not in riders
                    and analyzer.on_vehicle(b.bbox, vehicles)):
                riders[b.track_id] = (float(frame["ts"]), b.bbox)
    for tid in list(analyzer._tracks):
        analyzer.forget(tid)
    return analyzer, events, riders


def sheet(video, items, path, per_row=8):
    """items: [(label, ts, bbox)] -> one image of crops."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video)
    tiles = []
    for label, ts, (x1, y1, x2, y2) in items:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        pad = 80
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        tile = cv2.resize(crop, (200, 260))
        cv2.putText(tile, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(tile)
    if not tiles:
        return False
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    return cv2.imwrite(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--camera", default=None, help="camera the run used (optional)")
    ap.add_argument("--behaviour", required=True, metavar="FILE",
                    help="zones file, behaviour/<name>.yaml")
    ap.add_argument("--video", default=None, help="default: the run's source")
    ap.add_argument("--sheet", default=None, help="folder for contact sheets")
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        cfg = load_for_camera(args.camera)
    apply_behaviour(cfg, args.behaviour)
    run, frames, boxes = load(args.db)
    print(f"run {run['id']}: {run['source']} {run['width']}x{run['height']}, "
          f"{len(frames)} frames, {sum(len(v) for v in boxes.values())} boxes")
    analyzer, events, riders = replay(cfg, run, frames, boxes)

    counts = collections.Counter(k for k, *_ in events)
    print("events:", dict(counts))
    for kind, detail, tid, ts in events:
        if kind == "zone_exit":
            continue
        extra = {k: v for k, v in detail.items() if k not in ("bbox", "group", "cls_name")}
        print(f"  {ts:7.2f}s {kind:10s} track={tid} {json.dumps(extra)}")
    summary = analyzer.summary()
    for name, z in summary["zones"].items():
        print(f"  zone {name}: {json.dumps(z)}")
    print(f"  running={summary['running']} riders_ignored={summary['rider_boxes_ignored']} "
          f"(tracks={len(riders)}) masked={summary['masked_boxes_ignored']} "
          f"speed={summary['speed_hps']}")

    if args.sheet:
        os.makedirs(args.sheet, exist_ok=True)
        video = args.video or run["source"]
        by_kind = collections.defaultdict(list)
        for kind, detail, tid, ts in events:
            if kind in ("intrusion", "loitering", "running") and detail.get("bbox"):
                by_kind[kind].append((f"{tid} {ts:.1f}s", ts, detail["bbox"]))
        by_kind["riders"] = [(f"{tid} {ts:.1f}s", ts, bbox)
                             for tid, (ts, bbox) in sorted(riders.items())]
        for kind, items in by_kind.items():
            path = os.path.join(args.sheet, f"{kind}.jpg")
            if sheet(video, items[:48], path):
                print(f"  sheet: {path} ({min(len(items), 48)} of {len(items)})")


if __name__ == "__main__":
    main()
