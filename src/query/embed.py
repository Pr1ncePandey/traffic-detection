"""Turn saved crops into vectors. ONE implementation, two callers.

`tools/embed_crops.py` (the CLI, run by hand) and `src/server/embedder.py`
(the background thread inside the service) both embed crops, and the rules for
WHICH crops are eligible are load-bearing measurement decisions rather than
taste - a resolution floor, a confidence floor, and a frame-area ceiling, each
of which exists because ignoring it produced confident garbage. Two copies of
those rules would drift, and the drift would be invisible: the CLI and the
service would quietly build different indexes and nothing would compare them.

So the rules live here, once.

THE THREE SKIPS, AND WHY EACH ONE EXISTS

  small      a crop whose short side is under `min_px` carries too few pixels
             to embed meaningfully. Measured: person crops on the sample
             footage run ~47 px, below the person attribute model's useful
             range as well.
  low_conf   `crop_conf` is the detection confidence of the frame the crop came
             from, NOT objects.best_conf (the track maximum). 35-51% of crops
             come from sub-threshold detections and must be excludable.
  big_area   a near-frame-sized box is a picture of the whole scene, not an
             object, and embeds as something that matches almost any query.

`no_geometry` is a fourth outcome and is NOT a filter: it means no detection
row matches the crop's exact box size, so the confidence filter cannot be
applied honestly. Those crops are skipped and counted rather than embedded
with a guessed confidence.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .clip_onnx import preprocess

# Defaults shared by both callers, so the CLI's documented numbers and the
# service's index are built to the same standard.
MIN_PX = 48
MIN_CONF = 0.5
MAX_AREA = 0.33
BATCH = 16


def candidates(conn, model: str, camera: str | None = None,
               force: bool = False, limit: int = 0,
               after_id: int = 0) -> list:
    """Crop rows eligible for embedding, cheapest filters first (SQL only).

    `after_id` exists for the INCREMENTAL caller. "Not embedded" is not the
    same as "embeddable": a crop below the size or confidence floor is never
    written, so it stays un-embedded forever. A batched caller that always
    takes the first N un-embedded rows therefore re-reads the same ineligible
    crops on every pass and never advances - measured, the server's background
    embedder span 37 passes over one batch of 8 permanently-skipped crops and
    wrote nothing. Paging by id is what makes progress monotonic.

    The one-shot CLI does not need it: it sweeps every candidate in one pass.
    """
    sql = ["SELECT o.id, o.run_id, o.cls_name, o.crop_path, r.camera",
           "FROM objects o JOIN runs r ON r.id = o.run_id",
           "WHERE o.crop_path IS NOT NULL"]
    params: list = []
    if after_id:
        sql.append("AND o.id > ?")
        params.append(int(after_id))
    if not force:
        # The resumability contract: skip what this model already has. Scoped
        # by model, so a second embedding space does not look "already done".
        sql.append("AND NOT EXISTS (SELECT 1 FROM embeddings e"
                   " WHERE e.object_id = o.id AND e.model = ?)")
        params.append(model)
    if camera:
        sql.append("AND r.camera = ?")
        params.append(camera)
    sql.append("ORDER BY o.id")
    if limit:
        sql.append("LIMIT ?")
        params.append(int(limit))
    return conn.execute(" ".join(sql), params).fetchall()


def crop_conf(conn, object_id: int, w: int, h: int):
    """Detection confidence of the frame this crop came from, or None.

    Matched on exact box size, because nothing stores it: `_keep_best_crop`
    never records the confidence of the frame it chose. `MAX` because 12% of
    crops match several detections sharing that box size; None when nothing
    matches, which is reported rather than assumed benign.
    """
    row = conn.execute(
        "SELECT MAX(conf) FROM detections"
        " WHERE object_id = ? AND (x2-x1) = ? AND (y2-y1) = ?",
        (object_id, w, h)).fetchone()
    return None if row is None else row[0]


def frame_size(conn, run_id: int):
    """Frame dimensions for one run, inferred from the largest box in it.

    Boxes are clipped to the frame before storage (src/detectors/yolo.py), so
    the maximum x2/y2 across a run IS the frame size - which avoids adding a
    column for something already implied. Per RUN, not per camera: the same
    camera can be re-run at a different resolution.
    """
    row = conn.execute(
        "SELECT MAX(x2), MAX(y2) FROM detections d"
        " JOIN objects o ON o.id = d.object_id WHERE o.run_id = ?",
        (run_id,)).fetchone()
    if not row or not row[0] or not row[1]:
        return None
    return int(row[0]), int(row[1])


def new_skips() -> dict:
    return {"missing_file": 0, "unreadable": 0, "small": 0, "low_conf": 0,
            "big_area": 0, "no_geometry": 0}


class CropEmbedder:
    """Embeds crop rows in batches. Storage is the caller's business.

    `put` is called as put(object_id, model, vec, crop_w=, crop_h=,
    crop_conf=, crop_area=) - the CLI hands it a fresh SqliteStore, the server
    hands it the one shared store whose single writer thread it must not
    bypass. Passing a callback rather than a store keeps this module free of
    any opinion about who owns the database.
    """

    def __init__(self, encoder, min_px: int = MIN_PX,
                 min_conf: float = MIN_CONF, max_area: float = MAX_AREA,
                 batch: int = BATCH):
        self.enc = encoder
        self.min_px = int(min_px)
        self.min_conf = float(min_conf)
        self.max_area = float(max_area)
        self.batch = max(1, int(batch))
        self._frames: dict[int, tuple] = {}      # run_id -> (w, h), cached

    def run(self, conn, rows, put=None, dry_run: bool = False,
            skips: dict | None = None, should_stop=None) -> dict:
        """Embed `rows`. Returns {embedded, skipped, infer_s}.

        `should_stop` is checked between batches so a service can shut down
        without waiting for a long backlog to finish.
        """
        import cv2

        skip = skips if skips is not None else new_skips()
        pend_x, pend_meta = [], []
        embedded = 0
        infer_s = 0.0

        written: list = []

        def flush():
            nonlocal embedded, infer_s
            if not pend_x:
                return
            t = time.time()
            vecs = self.enc.encode_images(np.stack(pend_x))
            infer_s += time.time() - t
            if put is not None:
                for v, m in zip(vecs, pend_meta):
                    put(m["id"], self.enc.model, v, crop_w=m["w"],
                        crop_h=m["h"], crop_conf=m["conf"],
                        crop_area=m["area"])
            written.extend(m["id"] for m in pend_meta)
            embedded += len(pend_x)
            pend_x.clear()
            pend_meta.clear()

        for r in rows:
            if should_stop is not None and should_stop():
                break
            path = r["crop_path"]
            if not os.path.exists(path):
                skip["missing_file"] += 1
                continue
            img = cv2.imread(path)
            if img is None:
                skip["unreadable"] += 1
                continue
            h, w = img.shape[:2]
            if min(w, h) < self.min_px:
                skip["small"] += 1
                continue

            conf = crop_conf(conn, r["id"], w, h)
            if conf is None:
                # No detection row matches this crop's geometry. Do not guess a
                # confidence: without it the filter cannot be applied honestly.
                skip["no_geometry"] += 1
                continue
            if conf < self.min_conf:
                skip["low_conf"] += 1
                continue

            run_id = r["run_id"]
            if run_id not in self._frames:
                self._frames[run_id] = frame_size(conn, run_id) or (0, 0)
            fw, fh = self._frames[run_id]
            area = None
            if fw and fh:
                area = (w * h) / float(fw * fh)
                if area > self.max_area:
                    skip["big_area"] += 1
                    continue

            if dry_run:
                embedded += 1
                continue
            x = preprocess(img, self.enc.spec)
            if x is None:
                skip["unreadable"] += 1
                continue
            pend_x.append(x)
            pend_meta.append({"id": r["id"], "w": w, "h": h, "conf": conf,
                              "area": area})
            if len(pend_x) >= self.batch:
                flush()
        flush()
        # `embedded_ids` lets an incremental caller work out which rows it
        # considered and did NOT write, which is the only way to avoid
        # re-reading them next pass. The writer is asynchronous, so asking the
        # database instead would race the queue.
        return {"embedded": embedded, "skipped": skip, "infer_s": infer_s,
                "embedded_ids": written}
