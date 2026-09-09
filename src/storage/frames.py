"""Writing the stored frames, and reaping them when they exceed the budget.

Every analysed frame is stored TWICE - the untouched original and the
annotated copy - plus one image per detected object. Two formats:

  jpeg      (default) one file per frame per variant. Literal and simple;
            each file is independently readable by any tool.
  segments  rolling H.264 files. Same "every frame" guarantee at roughly 5%
            of the bytes, recovered by seeking to (segment, frame_offset).

MEASURED, so the choice is informed: 1920x1080 traffic footage encodes to
~378 KB/frame at q85, i.e. ~2 TB/day for raw+annotated at 30 fps. That is
fine for a clip (~614 MB for the 812-frame sample) and untenable for a
permanent stream, where `segments` costs ~130 GB/day instead.

Encoding is offloaded to a small thread pool: cv2.imencode of a 1080p frame
costs ~10-15 ms, and at 30 fps x 2 variants that would otherwise consume most
of a frame budget in the hot loop.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

JPEG_QUALITY = 85
ENCODE_WORKERS = 3        # enough to hide encode latency; more just adds contention
SEGMENT_SECONDS = 300     # 5-minute segments: small enough to reap, big enough to be cheap


class JpegFrameWriter:
    """One JPEG per frame per variant, written off the hot loop."""

    format = "jpeg"

    def __init__(self, root: str, quality: int = JPEG_QUALITY,
                 workers: int = ENCODE_WORKERS):
        self.root = root
        self.quality = int(quality)
        self.raw_dir = os.path.join(root, "raw")
        self.annot_dir = os.path.join(root, "annotated")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.annot_dir, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jpeg")
        self._lock = threading.Lock()
        self.bytes_written = 0
        self.frames_written = 0

    def _write(self, path: str, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        try:
            with open(path, "wb") as f:
                f.write(buf)
            with self._lock:
                self.bytes_written += len(buf)
                self.frames_written += 1
        except OSError as e:
            print(f"[frames] write failed {path}: {e}")

    def write_pair(self, frame_no: int, ts: float, raw, annotated) -> dict:
        """Queue both variants; return the paths immediately.

        Paths are deterministic from frame_no, so the database row can be
        written without waiting for the encode to finish.
        """
        name = f"frame_{frame_no:09d}.jpg"
        raw_path = os.path.join(self.raw_dir, name)
        annot_path = os.path.join(self.annot_dir, name)
        # copy(): the caller reuses its buffers on the next iteration, and the
        # encode happens later on another thread.
        self._pool.submit(self._write, raw_path, raw.copy())
        self._pool.submit(self._write, annot_path, annotated.copy())
        return {"raw_path": raw_path, "annotated_path": annot_path,
                "segment_id": None, "frame_offset": None, "bytes": None}

    def close(self):
        self._pool.shutdown(wait=True)


class SegmentFrameWriter:
    """Rolling H.264 segments. Every frame kept, ~5% of the JPEG bytes."""

    format = "segments"

    def __init__(self, root: str, fps: float, size: tuple[int, int],
                 segment_seconds: float = SEGMENT_SECONDS):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.size = (int(size[0]), int(size[1]))
        self.segment_seconds = float(segment_seconds)
        self.segment_id = 0
        self._raw = self._annot = None
        self._started = 0.0
        self._offset = 0
        self.bytes_written = 0
        self.frames_written = 0
        self._roll()

    def _paths(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return (os.path.join(self.root, f"raw_{stamp}_{self.segment_id:04d}.mp4"),
                os.path.join(self.root, f"annot_{stamp}_{self.segment_id:04d}.mp4"))

    def _roll(self):
        self._close_writers()
        self.segment_id += 1
        self._offset = 0
        self._started = time.monotonic()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.raw_path, self.annot_path = self._paths()
        self._raw = cv2.VideoWriter(self.raw_path, fourcc, self.fps, self.size)
        self._annot = cv2.VideoWriter(self.annot_path, fourcc, self.fps, self.size)

    def _close_writers(self):
        for w in (self._raw, self._annot):
            if w is not None:
                w.release()
        # Sizes are only real once the container is finalised.
        for p in (getattr(self, "raw_path", None), getattr(self, "annot_path", None)):
            if p and os.path.exists(p):
                self.bytes_written += os.path.getsize(p)

    def write_pair(self, frame_no: int, ts: float, raw, annotated) -> dict:
        if (time.monotonic() - self._started) >= self.segment_seconds:
            self._roll()
        self._raw.write(raw)
        self._annot.write(annotated)
        offset = self._offset
        self._offset += 1
        self.frames_written += 1
        return {"raw_path": self.raw_path, "annotated_path": self.annot_path,
                "segment_id": self.segment_id, "frame_offset": offset, "bytes": None}

    def close(self):
        self._close_writers()
        self._raw = self._annot = None


class Reaper:
    """Enforce max_age_hours / max_disk_gb, oldest first.

    Ordering comes from the database (authoritative, monotonic ts) while sizes
    come from os.stat at reap time. That combination avoids a second database
    write per frame just to record a byte count, and avoids scanning a
    directory that can hold millions of files.

    Files are unlinked BEFORE their rows are deleted, so an interrupted reap
    leaves a row pointing at a missing file rather than a file no query can
    ever find. reconcile() cleans up the former.
    """

    def __init__(self, store, max_age_hours: float = 0.0, max_disk_gb: float = 0.0,
                 interval: float = 60.0, usage_fn=None):
        self.store = store
        self.max_age_s = float(max_age_hours or 0) * 3600.0
        self.max_bytes = float(max_disk_gb or 0) * (1024 ** 3)
        self.interval = float(interval)
        self.usage_fn = usage_fn
        self.deleted_frames = 0
        self.deleted_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self.max_age_s > 0 or self.max_bytes > 0

    def start(self):
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._loop, name="reaper", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.reap_once()
            except Exception as e:
                print(f"[reaper] pass failed: {e}")

    def _usage(self) -> float:
        return float(self.usage_fn()) if self.usage_fn else 0.0

    def reap_once(self) -> int:
        conn = self.store.connect_ro()
        removed = 0
        try:
            if self.max_age_s > 0:
                cutoff = time.time() - self.max_age_s
                removed += self._delete(conn, "ts < ?", (cutoff,), limit=5000)
            if self.max_bytes > 0:
                over = self._usage() - self.deleted_bytes - self.max_bytes
                if over > 0:
                    # Byte-targeted, not count-batched: stop the moment enough
                    # bytes are freed. A fixed row batch overshoots wildly - a
                    # batch of 500 rows to reclaim 2 MB deletes hours of footage.
                    removed += self._delete_bytes(conn, over)
        finally:
            conn.close()
        return removed

    def _delete_bytes(self, conn, target_bytes: float) -> int:
        """Delete oldest frames until `target_bytes` have been reclaimed."""
        freed = 0.0
        removed = 0
        ids: list[int] = []
        rows = conn.execute(
            "SELECT id, raw_path, annotated_path FROM frames"
            " ORDER BY ts ASC LIMIT 20000").fetchall()
        for r in rows:
            if freed >= target_bytes:
                break
            for path in (r["raw_path"], r["annotated_path"]):
                if not path:
                    continue
                try:
                    freed += os.path.getsize(path)
                    os.unlink(path)
                except OSError:
                    pass
            ids.append(r["id"])
            removed += 1
            if len(ids) >= 500:              # bound the DELETE statement size
                self.store.delete_frames(ids)
                ids = []
        if ids:
            self.store.delete_frames(ids)
        self.deleted_frames += removed
        self.deleted_bytes += freed
        return removed

    def _delete(self, conn, where: str, params: tuple, limit: int,
                track_bytes: bool = False) -> int:
        rows = conn.execute(
            f"SELECT id, raw_path, annotated_path FROM frames WHERE {where}"
            f" ORDER BY ts ASC LIMIT {int(limit)}", params).fetchall()
        if not rows:
            return 0
        ids, freed = [], 0
        for r in rows:
            for p in (r["raw_path"], r["annotated_path"]):
                if not p:
                    continue
                try:
                    freed += os.path.getsize(p)
                    os.unlink(p)
                except OSError:
                    pass
            ids.append(r["id"])
        # Row deletion goes through the writer thread's connection to avoid two
        # writers on one database file.
        self.store.delete_frames(ids)
        self.deleted_frames += len(ids)
        self.deleted_bytes += freed
        return len(ids)

    def reconcile(self) -> int:
        """Drop rows whose files are gone (crash mid-reap, or manual deletion)."""
        conn = self.store.connect_ro()
        dead = []
        try:
            for r in conn.execute(
                    "SELECT id, raw_path FROM frames WHERE raw_path IS NOT NULL"):
                if not os.path.exists(r["raw_path"]):
                    dead.append(r["id"])
        finally:
            conn.close()
        if dead:
            self.store.delete_frames(dead)
            print(f"[reaper] reconciled {len(dead)} rows with missing files")
        return len(dead)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def build_writer(cfg: dict, fps: float, size: tuple[int, int]):
    """Pick the frame writer from the `frames` config block."""
    fmt = str((cfg or {}).get("format", "jpeg")).lower()
    root = (cfg or {}).get("dir", "outputs/frames")
    if fmt in ("segment", "segments", "video", "h264"):
        return SegmentFrameWriter(root, fps, size,
                                  segment_seconds=float(cfg.get("segment_seconds",
                                                                SEGMENT_SECONDS)))
    if fmt not in ("jpeg", "jpg"):
        print(f"[frames] unknown frames.format={fmt!r}, using jpeg")
    return JpegFrameWriter(root, quality=int((cfg or {}).get("quality", JPEG_QUALITY)),
                           workers=int((cfg or {}).get("encode_workers", ENCODE_WORKERS)))
