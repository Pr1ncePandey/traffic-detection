"""SQLite storage: object-level record with an extensible attribute table.

Replaces CsvStore for the primary write path. The bug being fixed: CsvStore
accumulated every row in a Python list and wrote to disk only in close(), so a
live stream (which never reaches close()) persisted nothing and grew until the
process died. Here a writer thread commits continuously.

Schema shape worth knowing:
  objects     one row per track, NOT per frame - the thing that was missing
  detections  one row per object per frame (what tracks.csv used to be)
  attributes  TALL (object_id, key, value) rather than wide columns, so a new
              attribute - colour, brand, speed - needs no migration. The
              (key, value) index is what keeps plate lookup fast.
"""

import json
import os
import queue
import sqlite3
import threading
import time

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, started_at REAL, ended_at REAL,
  source TEXT, camera TEXT, fps REAL, width INT, height INT,
  analyse_fps REAL, config_json TEXT);

CREATE TABLE IF NOT EXISTS frames(
  id INTEGER PRIMARY KEY, run_id INT, frame_no INT, ts REAL,
  raw_path TEXT, annotated_path TEXT,
  segment_id INT, frame_offset INT, bytes INT);
CREATE INDEX IF NOT EXISTS idx_frames_run_no ON frames(run_id, frame_no);
CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames(ts);

CREATE TABLE IF NOT EXISTS objects(
  id INTEGER PRIMARY KEY, run_id INT, track_id INT,
  cls_name TEXT, cls_group TEXT,
  first_seen_s REAL, last_seen_s REAL, frames_seen INT,
  best_conf REAL, crop_path TEXT,
  lane_id TEXT, lane_flag TEXT,
  UNIQUE(run_id, track_id));
CREATE INDEX IF NOT EXISTS idx_objects_class ON objects(cls_name);

CREATE TABLE IF NOT EXISTS detections(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INT, frame_id INT, object_id INT,
  x1 INT, y1 INT, x2 INT, y2 INT, conf REAL,
  cls_name TEXT, event TEXT, lane_id TEXT, lane_flag TEXT);
CREATE INDEX IF NOT EXISTS idx_det_object ON detections(object_id);
CREATE INDEX IF NOT EXISTS idx_det_frame ON detections(frame_id);

CREATE TABLE IF NOT EXISTS attributes(
  object_id INT, key TEXT, value TEXT, conf REAL, updated_s REAL,
  PRIMARY KEY(object_id, key));
CREATE INDEX IF NOT EXISTS idx_attr_kv ON attributes(key, value);

CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INT, frame_id INT, object_id INT,
  kind TEXT, detail_json TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
"""

# INSERT templates per table. Keys are the queue's routing labels.
_SQL = {
    "frames": ("INSERT OR REPLACE INTO frames"
               "(id,run_id,frame_no,ts,raw_path,annotated_path,segment_id,frame_offset,bytes)"
               " VALUES(?,?,?,?,?,?,?,?,?)"),
    "detections": ("INSERT INTO detections"
                   "(run_id,frame_id,object_id,x1,y1,x2,y2,conf,cls_name,event,lane_id,lane_flag)"
                   " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"),
    # Repeated upserts are expected (first sight, then finalize), so conflicts
    # update rather than fail. best_conf and frames_seen only ever grow.
    "objects": ("INSERT INTO objects"
                "(id,run_id,track_id,cls_name,cls_group,first_seen_s,last_seen_s,"
                "frames_seen,best_conf,crop_path,lane_id,lane_flag)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(run_id,track_id) DO UPDATE SET"
                " cls_name=excluded.cls_name, cls_group=excluded.cls_group,"
                " last_seen_s=MAX(objects.last_seen_s, excluded.last_seen_s),"
                " frames_seen=MAX(objects.frames_seen, excluded.frames_seen),"
                " best_conf=MAX(objects.best_conf, excluded.best_conf),"
                " crop_path=COALESCE(excluded.crop_path, objects.crop_path),"
                " lane_id=COALESCE(excluded.lane_id, objects.lane_id),"
                " lane_flag=COALESCE(excluded.lane_flag, objects.lane_flag)"),
    # Keep the most CONFIDENT read, not the most recent - a late, worse OCR
    # result must not overwrite a good one.
    "attributes": ("INSERT INTO attributes(object_id,key,value,conf,updated_s)"
                   " VALUES(?,?,?,?,?)"
                   " ON CONFLICT(object_id,key) DO UPDATE SET"
                   " value=excluded.value, conf=excluded.conf, updated_s=excluded.updated_s"
                   " WHERE excluded.conf >= attributes.conf"),
    "events": ("INSERT INTO events(run_id,frame_id,object_id,kind,detail_json,ts)"
               " VALUES(?,?,?,?,?,?)"),
}

_COLS = {
    "frames": ("id", "run_id", "frame_no", "ts", "raw_path", "annotated_path",
               "segment_id", "frame_offset", "bytes"),
    "detections": ("run_id", "frame_id", "object_id", "x1", "y1", "x2", "y2",
                   "conf", "cls_name", "event", "lane_id", "lane_flag"),
    "objects": ("id", "run_id", "track_id", "cls_name", "cls_group", "first_seen_s",
                "last_seen_s", "frames_seen", "best_conf", "crop_path",
                "lane_id", "lane_flag"),
    "events": ("run_id", "frame_id", "object_id", "kind", "detail_json", "ts"),
}


class SqliteStore:
    BATCH_ROWS = 500          # flush once this many rows are pending...
    COMMIT_INTERVAL = 2.0     # ...or this many seconds pass. Worst-case loss on
                              # a hard kill is therefore ~2s of rows.
    QUEUE_MAX = 20000         # backstop: if the writer somehow cannot keep up we
                              # block rather than grow memory without bound.

    def __init__(self, path: str, batch_rows: int | None = None,
                 commit_interval: float | None = None):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.batch_rows = int(batch_rows or self.BATCH_ROWS)
        self.commit_interval = float(commit_interval or self.COMMIT_INTERVAL)
        # check_same_thread=False: built here, written only by the writer thread.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self.run_id = 0
        self._q: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAX)
        self._flushed = threading.Event()
        self._id_lock = threading.Lock()
        self._frame_id = self._max_id("frames")
        self._object_id = self._max_id("objects")
        self._thread = threading.Thread(target=self._writer, name="sqlite-writer", daemon=True)
        self._thread.start()
        self.rows_written = 0

    def _max_id(self, table: str) -> int:
        cur = self._conn.execute(f"SELECT COALESCE(MAX(id),0) FROM {table}")
        return int(cur.fetchone()[0])

    # --- ids: synchronous, no database round-trip ---------------------------
    def next_frame_id(self) -> int:
        with self._id_lock:
            self._frame_id += 1
            return self._frame_id

    def next_object_id(self) -> int:
        with self._id_lock:
            self._object_id += 1
            return self._object_id

    # --- run lifecycle ------------------------------------------------------
    def start_run(self, meta: dict) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs(started_at,source,camera,fps,width,height,analyse_fps,config_json)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (time.time(), str(meta.get("source", "")), str(meta.get("camera", "")),
             float(meta.get("fps", 0) or 0), int(meta.get("width", 0) or 0),
             int(meta.get("height", 0) or 0), float(meta.get("analyse_fps", 0) or 0),
             json.dumps(meta.get("config", {}), default=str)))
        self._conn.commit()
        self.run_id = int(cur.lastrowid)
        return self.run_id

    def end_run(self):
        self._conn.execute("UPDATE runs SET ended_at=? WHERE id=?", (time.time(), self.run_id))
        self._conn.commit()

    # --- writes: queued ----------------------------------------------------
    def _put(self, table: str, row: dict):
        cols = _COLS[table]
        self._q.put((table, tuple(row.get(c) for c in cols)))

    def put_frame(self, row: dict):
        row.setdefault("run_id", self.run_id)
        self._put("frames", row)

    def put_detection(self, row: dict):
        row.setdefault("run_id", self.run_id)
        self._put("detections", row)

    def upsert_object(self, row: dict):
        row.setdefault("run_id", self.run_id)
        self._put("objects", row)

    def put_attribute(self, object_id: int, key: str, value: str,
                      conf: float = 0.0, ts: float = 0.0):
        self._q.put(("attributes", (int(object_id), str(key), str(value),
                                    float(conf), float(ts))))

    def delete_frames(self, frame_ids) -> None:
        """Queue frame-row deletion (used by the Reaper).

        Routed through the writer thread on purpose: two connections writing
        one SQLite file is how you earn "database is locked" under load.
        """
        ids = [int(i) for i in frame_ids]
        if ids:
            self._q.put(("__delete_frames__", ids))

    def put_event(self, row: dict):
        row.setdefault("run_id", self.run_id)
        if isinstance(row.get("detail_json"), (dict, list)):
            row["detail_json"] = json.dumps(row["detail_json"], default=str)
        self._put("events", row)

    # --- writer thread -----------------------------------------------------
    def _writer(self):
        pending: dict[str, list] = {}
        count = 0
        last = time.monotonic()
        while True:
            timeout = max(0.05, self.commit_interval - (time.monotonic() - last))
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                item = None
            if item is not None:
                if item[0] == "__flush__":
                    count += self._commit(pending)
                    last = time.monotonic()
                    self._flushed.set()
                    continue
                if item[0] == "__stop__":
                    self._commit(pending)
                    self._flushed.set()
                    return
                if item[0] == "__delete_frames__":
                    # Commit first so a pending INSERT for these ids cannot
                    # land after the DELETE and resurrect the row.
                    count += self._commit(pending)
                    self._delete_frames(item[1])
                    last = time.monotonic()
                    continue
                table, values = item
                pending.setdefault(table, []).append(values)
                count += 1
            due = count >= self.batch_rows or (time.monotonic() - last) >= self.commit_interval
            if pending and due:
                self._commit(pending)
                count = 0
                last = time.monotonic()

    def _commit(self, pending: dict) -> int:
        if not pending:
            return 0
        n = 0
        try:
            for table, rows in pending.items():
                if rows:
                    self._conn.executemany(_SQL[table], rows)
                    n += len(rows)
            self._conn.commit()
            self.rows_written += n
        except Exception as e:
            print(f"[sqlite] write failed, dropping {n} rows: {e}")
        pending.clear()
        return n

    def _delete_frames(self, ids: list):
        try:
            self._conn.executemany("DELETE FROM frames WHERE id=?",
                                   [(i,) for i in ids])
            self._conn.commit()
        except Exception as e:
            print(f"[sqlite] frame delete failed: {e}")

    def flush(self):
        """Block until everything queued so far is committed."""
        self._flushed.clear()
        self._q.put(("__flush__", None))
        self._flushed.wait(timeout=10.0)

    def close(self):
        self._flushed.clear()
        self._q.put(("__stop__", None))
        self._thread.join(timeout=10.0)
        try:
            self.end_run()
        except Exception:
            pass
        self._conn.close()

    # --- reads (reports/queries) -------------------------------------------
    def connect_ro(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn
