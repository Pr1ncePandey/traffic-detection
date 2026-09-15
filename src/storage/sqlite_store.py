"""SQLite storage: object-level record with an extensible attribute table.

Replaces CsvStore for the primary write path. The bug being fixed: CsvStore
accumulated every row in a Python list and wrote to disk only in close(), so a
live stream (which never reaches close()) persisted nothing and grew until the
process died. Here a writer thread commits continuously.

Schema shape worth knowing:
  objects     one row per track, NOT per frame - the thing that was missing
  detections  one row per object per frame (what tracks.csv used to be)
  attributes  TALL (object_id, key, value) rather than wide columns, so a new
              attribute - colour, speed - needs no migration. The (key, value)
              index is what keeps plate lookup fast.
  identities  one row per DURABLE IDENTITY, not per track - the thing a track
              id cannot provide, because ByteTrack mints a new id every time an
              object reappears. objects.identity_id points here, so several
              sightings of one entity share it while each keeps its own objects
              row. See trackers/identity.py for why they are split.

              CLASS-AGNOSTIC, which the former `vehicles` table was not: it
              was keyed `plate TEXT NOT NULL UNIQUE`, so only vehicles could
              ever have a durable identity and no person could be followed
              across cameras. `kind` discriminates instead, exactly as
              objects.cls_group does for sightings.
"""

import json
import os
import queue
import sqlite3
import threading
import time
from collections import OrderedDict

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
-- One writer thread means contention is largely designed out, but the frame
-- reaper, external readers, backups and anyone opening the file with a CLI can
-- still collide. One pragma removes a whole class of intermittent
-- "database is locked" failures.
PRAGMA busy_timeout=5000;

-- time_base says which CLOCK first_seen_s/last_seen_s/ts are on, because the
-- pipeline writes two incompatible kinds of number into one column: a
-- wall-clock epoch (~1.7e9) for live sources, seconds-from-start-of-clip
-- (~12.4) for files. Without this, one fleet-wide DB holding both makes any
-- ORDER BY time return silent nonsense. recorded_at is the footage's real
-- start time, which is what makes a file run's clip-seconds absolute.
-- cam_lat/cam_lon are SNAPSHOT here rather than read from yaml at query time:
-- editing a camera file months later must not retroactively move where a
-- historical sighting happened.
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, started_at REAL, ended_at REAL,
  source TEXT, camera TEXT, fps REAL, width INT, height INT,
  analyse_fps REAL, config_json TEXT,
  time_base TEXT, recorded_at REAL, cam_lat REAL, cam_lon REAL);

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
  identity_id INT,
  UNIQUE(run_id, track_id));
CREATE INDEX IF NOT EXISTS idx_objects_class ON objects(cls_name);
CREATE INDEX IF NOT EXISTS idx_objects_identity ON objects(identity_id);

-- Durable identity, one row per entity. `kind` is the identity AXIS and `key`
-- the value on it: a licence plate is kind='plate', key='HR26DK8337'. An
-- appearance cluster from the embeddings table will be kind='person_reid' with
-- a synthetic cluster id. A new axis is a new VALUE, never a new table.
--
-- UNIQUE(kind, key) is both the correctness constraint - one entity per key
-- within its axis - and the lookup index, so no second index is declared for
-- it: `WHERE kind=? AND key=?` is a leftmost-prefix hit. Bind too loosely and
-- two entities' histories merge, which no later frame undoes and no exception
-- reports, so this constraint is the thing that catches it.
--
-- NO sightings counter, NO cached centroid, NO member count, on purpose: how
-- many times an entity was seen is exactly COUNT(*) over objects.identity_id,
-- which idx_objects_identity already makes cheap, and a centroid is the mean
-- of its members' vectors in `embeddings`. A stored aggregate would be bumped
-- by every mid-track rebind as well as by finalize, so it would drift from the
-- rows it claims to summarise - and a number that is quietly wrong is worse
-- than a join.
--
-- NO `model` column either. The encoder is EVIDENCE for an identity, not part
-- of it: keying on it would mint a fresh identity universe on every encoder
-- swap and orphan every existing objects.identity_id. The embedding space is
-- recorded once, in embeddings.model/dim, where the vectors are.
CREATE TABLE IF NOT EXISTS identities(
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  key  TEXT NOT NULL,
  first_seen_at REAL, last_seen_at REAL,
  UNIQUE(kind, key));

-- No `event` column. counting.py wrote every zone crossing twice - once here
-- and once as an events row - and the two copies did not even have equal
-- durability: detections is _SHEDDABLE and events is not, so under queue
-- pressure THIS was the copy that vanished. events was already canonical
-- (query.py, report.py and /events all read it), nothing ever read this
-- column, and it was empty on all but a handful of rows in the highest-volume
-- table in the schema. Detection.event still exists in memory and still
-- reaches tracks.csv; it is only not duplicated into SQLite.
CREATE TABLE IF NOT EXISTS detections(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INT, frame_id INT, object_id INT,
  x1 INT, y1 INT, x2 INT, y2 INT, conf REAL,
  cls_name TEXT, lane_id TEXT, lane_flag TEXT);
CREATE INDEX IF NOT EXISTS idx_det_object ON detections(object_id);
CREATE INDEX IF NOT EXISTS idx_det_frame ON detections(frame_id);

CREATE TABLE IF NOT EXISTS attributes(
  object_id INT, key TEXT, value TEXT, conf REAL, updated_s REAL,
  PRIMARY KEY(object_id, key));
CREATE INDEX IF NOT EXISTS idx_attr_kv ON attributes(key, value);

-- Open-vocabulary search vectors, one per object per embedding space.
-- Written OFFLINE by tools/embed_crops.py, never by the pipeline: no
-- real-time decision needs an embedding, and keeping inference out of
-- pipeline.py makes a model swap a re-run of a script.
--
-- Every non-obvious column is a known failure mode made queryable:
--   model      cosine is only meaningful WITHIN one space, so two spaces must
--              never be ranked against each other. Present from the first
--              insert; adding it later, with an index already built, is the
--              expensive version of this.
--   dim        varies by model (512 for ViT-B/16, 768 for L/14), so the
--              reader cannot assume a width.
--   vec        float32 little-endian, L2-NORMALISED ON WRITE. That is what
--              lets search be a plain dot product, so the query path needs no
--              division and cannot disagree with itself about the metric.
--   crop_w/h   the crop's real pixel size, so the resolution floor is a WHERE
--              clause rather than an anecdote.
--   crop_conf  detection confidence of the frame the crop came from. NOT
--              objects.best_conf, which is the track maximum: 35-51% of crops
--              come from sub-threshold detections and must be excludable.
--   crop_area  box area as a fraction of frame. A near-full-frame box is a
--              picture of the whole scene, not an object crop, and embeds as
--              confident garbage that matches almost any query.
-- PRIMARY KEY(object_id, model) matches attributes' one-row-per-thing shape
-- and makes the retention delete a plain DELETE ... WHERE object_id IN (...).
CREATE TABLE IF NOT EXISTS embeddings(
  object_id  INTEGER NOT NULL,
  model      TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vec        BLOB NOT NULL,
  crop_w     INTEGER,
  crop_h     INTEGER,
  crop_conf  REAL,
  crop_area  REAL,
  created_at REAL,
  PRIMARY KEY(object_id, model));
CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model);

CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INT, frame_id INT, object_id INT,
  kind TEXT, detail_json TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

-- THE OUTBOX. An incident is the durable record of something worth telling
-- someone about; a delivery is one attempt to tell one subscriber.
--
-- id is a TEXT natural key ("demo-1841-wrong_way"), not an autoincrement, and
-- that is the idempotency contract: delivery is at-least-once, so a consumer
-- that has already seen this id must be able to recognise it. INSERT OR IGNORE
-- on the primary key also makes a double fire locally harmless.
CREATE TABLE IF NOT EXISTS incidents(
  id TEXT PRIMARY KEY, camera TEXT, kind TEXT, object_id INT, identity_id INT,
  payload_json TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at);
CREATE INDEX IF NOT EXISTS idx_incidents_kind ON incidents(kind);
CREATE INDEX IF NOT EXISTS idx_incidents_crop ON incidents(object_id);

-- One row per (incident, subscriber). endpoint lives HERE rather than only in
-- config because retries and dead-lettering are per subscriber: one broken
-- consumer must not delay another, which a single shared attempt counter
-- could not express.
--   status: pending | sent | dead
-- next_attempt_at is the backoff clock; the dispatcher polls on it.
CREATE TABLE IF NOT EXISTS deliveries(
  id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id TEXT, endpoint TEXT,
  attempts INT DEFAULT 0, next_attempt_at REAL,
  status TEXT DEFAULT 'pending', last_error TEXT,
  created_at REAL, sent_at REAL,
  UNIQUE(incident_id, endpoint));
CREATE INDEX IF NOT EXISTS idx_deliveries_due
  ON deliveries(status, next_attempt_at);
"""

# INSERT templates per table. Keys are the queue's routing labels.
_SQL = {
    "frames": ("INSERT OR REPLACE INTO frames"
               "(id,run_id,frame_no,ts,raw_path,annotated_path,segment_id,frame_offset,bytes)"
               " VALUES(?,?,?,?,?,?,?,?,?)"),
    "detections": ("INSERT INTO detections"
                   "(run_id,frame_id,object_id,x1,y1,x2,y2,conf,cls_name,lane_id,lane_flag)"
                   " VALUES(?,?,?,?,?,?,?,?,?,?,?)"),
    # Repeated upserts are expected (first sight, then finalize), so conflicts
    # update rather than fail. best_conf and frames_seen only ever grow.
    "objects": ("INSERT INTO objects"
                "(id,run_id,track_id,cls_name,cls_group,first_seen_s,last_seen_s,"
                "frames_seen,best_conf,crop_path,lane_id,lane_flag,identity_id)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(run_id,track_id) DO UPDATE SET"
                " cls_name=excluded.cls_name, cls_group=excluded.cls_group,"
                " last_seen_s=MAX(objects.last_seen_s, excluded.last_seen_s),"
                " frames_seen=MAX(objects.frames_seen, excluded.frames_seen),"
                " best_conf=MAX(objects.best_conf, excluded.best_conf),"
                " crop_path=COALESCE(excluded.crop_path, objects.crop_path),"
                " lane_id=COALESCE(excluded.lane_id, objects.lane_id),"
                " lane_flag=COALESCE(excluded.lane_flag, objects.lane_flag),"
                # COALESCE, not excluded.*: an identity is resolved part-way
                # through a track, so an earlier upsert legitimately has no
                # identity_id and must not blank one that is already known.
                " identity_id=COALESCE(excluded.identity_id, objects.identity_id)"),
    # ON CONFLICT rather than a bare INSERT is load-bearing, not tidiness:
    # _commit() catches per BATCH, so one UNIQUE(kind,key) violation would drop
    # up to batch_rows (500) unrelated rows with it.
    # Idempotent: MIN/MAX rather than assignment, so re-binding a key mid
    # track (the consensus is still being voted) cannot drag first_seen_at
    # forward or last_seen_at backward.
    "identities": ("INSERT INTO identities(id,kind,key,first_seen_at,last_seen_at)"
                   " VALUES(?,?,?,?,?)"
                   " ON CONFLICT(kind,key) DO UPDATE SET"
                   " first_seen_at=MIN(identities.first_seen_at, excluded.first_seen_at),"
                   " last_seen_at=MAX(identities.last_seen_at, excluded.last_seen_at)"),
    # Keep the most CONFIDENT read, not the most recent - a late, worse OCR
    # result must not overwrite a good one.
    "attributes": ("INSERT INTO attributes(object_id,key,value,conf,updated_s)"
                   " VALUES(?,?,?,?,?)"
                   " ON CONFLICT(object_id,key) DO UPDATE SET"
                   " value=excluded.value, conf=excluded.conf, updated_s=excluded.updated_s"
                   " WHERE excluded.conf >= attributes.conf"),
    "events": ("INSERT INTO events(run_id,frame_id,object_id,kind,detail_json,ts)"
               " VALUES(?,?,?,?,?,?)"),
    # Re-embedding the same object in the same space replaces the vector
    # rather than failing, so the embedder is resumable AND re-runnable: a
    # model re-export or a changed preprocessing step is a plain re-run.
    # Conflicts are on (object_id, model), so a second embedding space lands
    # alongside the first instead of overwriting it.
    "embeddings": ("INSERT INTO embeddings"
                   "(object_id,model,dim,vec,crop_w,crop_h,crop_conf,crop_area,created_at)"
                   " VALUES(?,?,?,?,?,?,?,?,?)"
                   " ON CONFLICT(object_id,model) DO UPDATE SET"
                   " dim=excluded.dim, vec=excluded.vec,"
                   " crop_w=excluded.crop_w, crop_h=excluded.crop_h,"
                   " crop_conf=excluded.crop_conf, crop_area=excluded.crop_area,"
                   " created_at=excluded.created_at"),
}

_COLS = {
    "frames": ("id", "run_id", "frame_no", "ts", "raw_path", "annotated_path",
               "segment_id", "frame_offset", "bytes"),
    "detections": ("run_id", "frame_id", "object_id", "x1", "y1", "x2", "y2",
                   "conf", "cls_name", "lane_id", "lane_flag"),
    "objects": ("id", "run_id", "track_id", "cls_name", "cls_group", "first_seen_s",
                "last_seen_s", "frames_seen", "best_conf", "crop_path",
                "lane_id", "lane_flag", "identity_id"),
    "identities": ("id", "kind", "key", "first_seen_at", "last_seen_at"),
    "events": ("run_id", "frame_id", "object_id", "kind", "detail_json", "ts"),
}

# (table, column, what it was added for) - checked before the schema script.
# Add an entry whenever a column joins a table that already ships in the wild.
_REQUIRED_COLS = (
    ("objects", "identity_id", "class-agnostic durable identity"),
    ("runs", "time_base", "comparable cross-camera timestamps"),
)

# Which tables may be DROPPED rather than blocked on when the write queue is
# full. The split is by volume, not importance: frames and detections are
# written once per frame and are reconstructible-ish, so shedding them lets the
# queue drain. objects/identities/attributes/events are ~once per track or per
# incident, so they are the durable record and always wait their turn - and
# because the droppable tables are the ones filling the queue, that wait ends.
_SHEDDABLE = frozenset({"frames", "detections"})


def _opt_float(value):
    """None stays None. Load-bearing: recorded_at NULL means 'unorderable',
    which 0.0 would silently turn into 1970."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_str(value):
    return None if value in (None, "") else str(value)


class SqliteStore:
    BATCH_ROWS = 500          # flush once this many rows are pending...
    COMMIT_INTERVAL = 2.0     # ...or this many seconds pass. Worst-case loss on
                              # a hard kill is therefore ~2s of rows.
    QUEUE_MAX = 20000         # backstop: if the writer somehow cannot keep up we
                              # block rather than grow memory without bound.
    MINTED_CAP = 1024         # plate->id entries held across the commit window

    FAILURE_ALARM_AFTER = 3   # consecutive failed commits before escalating

    def __init__(self, path: str, batch_rows: int | None = None,
                 commit_interval: float | None = None,
                 shed_when_full: bool = False):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.batch_rows = int(batch_rows or self.BATCH_ROWS)
        self.commit_interval = float(commit_interval or self.COMMIT_INTERVAL)
        # check_same_thread=False: built here, written only by the writer thread.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        # BEFORE executescript, not after: SCHEMA declares an index ON
        # objects(identity_id), which on a pre-change file fails with a bare
        # "no such column: identity_id" and aborts the rest of the script (so
        # `identities` is never created either). Checking first turns that into
        # an error that says what to do.
        self._assert_schema_current()
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        # Identity lookups are reads on the main thread while the writer thread
        # owns _conn. WAL (set above) lets a separate reader run without ever
        # blocking the writer or seeing "database is locked" - which sharing one
        # connection across two threads would risk. connect_ro() is the same idea.
        self._rconn = sqlite3.connect(path, check_same_thread=False)
        self.run_id = 0
        self._q: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAX)
        self._flushed = threading.Event()
        self._id_lock = threading.Lock()
        self._frame_id = self._max_id("frames")
        self._object_id = self._max_id("objects")
        self._identity_id = self._max_id("identities")
        # (kind, key) -> id for identities minted but not yet committed.
        # Consulted before the DB so a lookup during the commit window cannot
        # mint a second id for a plate that already has one. Capped: anything
        # this many mints old has certainly been committed (the writer flushes
        # every batch_rows rows or commit_interval seconds).
        self._minted: OrderedDict[tuple[str, str], int] = OrderedDict()
        # Live sources shed instead of applying backpressure: a blocking put
        # stalls inference, and on a camera that means dropping real-world time
        # that can never be recovered. A file is merely slow, so it keeps the
        # backpressure and loses nothing.
        self.shed_when_full = bool(shed_when_full)
        self.rows_written = 0
        self.rows_dropped = 0          # shed at the queue, by table
        self.dropped_by_table: dict[str, int] = {}
        self.rows_failed = 0           # lost to a failing commit
        self.write_failures = 0        # total failed commits
        self._consecutive_failures = 0
        self.last_write_error: str | None = None
        self.write_alarm = False       # sticky: a persistent failure happened
        self._thread = threading.Thread(target=self._writer, name="sqlite-writer", daemon=True)
        self._thread.start()

    def _assert_schema_current(self):
        """Fail loudly on a database predating a column this code now writes.

        CREATE TABLE IF NOT EXISTS cannot add a column, so an older file keeps
        its narrower table while _SQL supplies more values. For `objects` that
        raises inside _commit(), which catches per table per batch - so the run
        would appear to work while silently discarding up to batch_rows object
        rows at a time. A hard error naming the fix is worth more than a
        migration here.

        Called before the schema script for the reason given at the call site.
        A brand-new file has none of these tables yet, so the empty-cols case is
        a pass, not a failure.
        """
        for table, column, why in _REQUIRED_COLS:
            cols = {r[1] for r in
                    self._conn.execute(f"PRAGMA table_info({table})")}
            if cols and column not in cols:
                raise RuntimeError(
                    f"{self.path} predates {why} ({table} has no {column} "
                    f"column). Delete it, or point --db at a new path; rows "
                    f"would otherwise be dropped silently.")

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

    def next_identity_id(self) -> int:
        with self._id_lock:
            self._identity_id += 1
            return self._identity_id

    # --- run lifecycle ------------------------------------------------------
    def start_run(self, meta: dict) -> int:
        """Open a run row.

        `time_base` and `recorded_at` are what make this run's timestamps
        comparable with another camera's - see the runs DDL and src/timebase.py.
        `cam_lat`/`cam_lon` are snapshotted rather than referenced so a later
        yaml edit cannot move a historical sighting.
        """
        cur = self._conn.execute(
            "INSERT INTO runs(started_at,source,camera,fps,width,height,"
            "analyse_fps,config_json,time_base,recorded_at,cam_lat,cam_lon)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), str(meta.get("source", "")), str(meta.get("camera", "")),
             float(meta.get("fps", 0) or 0), int(meta.get("width", 0) or 0),
             int(meta.get("height", 0) or 0), float(meta.get("analyse_fps", 0) or 0),
             json.dumps(meta.get("config", {}), default=str),
             _opt_str(meta.get("time_base")), _opt_float(meta.get("recorded_at")),
             _opt_float(meta.get("cam_lat")), _opt_float(meta.get("cam_lon"))))
        self._conn.commit()
        self.run_id = int(cur.lastrowid)
        return self.run_id

    def end_run(self, run_id: int | None = None):
        """Close a run row. Pass run_id when the store is SHARED.

        `self.run_id` is only meaningful for a single-camera process. When N
        camera workers share one store (the server's process model) it holds
        whichever run started last, so a caller that owns a run must name it.
        """
        target = self.run_id if run_id is None else int(run_id)
        if not target:
            return
        self._conn.execute("UPDATE runs SET ended_at=? WHERE id=?",
                           (time.time(), target))
        self._conn.commit()

    # --- writes: queued ----------------------------------------------------
    def _enqueue(self, table: str, values):
        """Queue one row, shedding rather than blocking where that is allowed.

        The old unconditional blocking put meant a writer falling behind stalled
        the frame loop. On a file that is merely slow; on a live camera it drops
        real-world time that cannot be recovered, so the loop must keep running
        and the loss must be COUNTED - a silently degrading feed is worse than
        an obviously broken one. See _SHEDDABLE for what may go.
        """
        if self.shed_when_full and table in _SHEDDABLE:
            try:
                self._q.put_nowait((table, values))
            except queue.Full:
                self.rows_dropped += 1
                self.dropped_by_table[table] = self.dropped_by_table.get(table, 0) + 1
            return
        self._q.put((table, values))

    def _put(self, table: str, row: dict):
        cols = _COLS[table]
        self._enqueue(table, tuple(row.get(c) for c in cols))

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
        self._enqueue("attributes", (int(object_id), str(key), str(value),
                                     float(conf), float(ts)))

    def put_embedding(self, object_id: int, model: str, vec, crop_w=None,
                      crop_h=None, crop_conf=None, crop_area=None,
                      ts: float = 0.0):
        """Queue one object's search vector, through the same writer thread as
        everything else.

        Positional like put_attribute rather than going through _put/_COLS:
        the row has a fixed shape and a BLOB, so a dict round-trip would buy
        nothing.

        `vec` is expected ALREADY L2-normalised - see the DDL. It is converted
        to little-endian float32 here so the on-disk layout does not depend on
        the writer's architecture, which is what lets a database move between
        machines and still be searchable.
        """
        import numpy as np
        arr = np.ascontiguousarray(vec, dtype="<f4").ravel()
        self._enqueue("embeddings", (
            int(object_id), str(model), int(arr.size), arr.tobytes(),
            None if crop_w is None else int(crop_w),
            None if crop_h is None else int(crop_h),
            None if crop_conf is None else float(crop_conf),
            None if crop_area is None else float(crop_area),
            float(ts) if ts else time.time()))

    # --- durable identity (see trackers/identity.py) ------------------------
    # kind is the identity AXIS ('plate', later 'person_reid'), key the value on
    # it. These three are generic so a second axis needs no new storage method;
    # trackers/identity.py binds kind='plate' at the wiring point.
    def lookup_identity(self, kind: str, key: str):
        """Identity id for (kind, key), or None. Synchronous: the caller needs
        the id now, to label this frame.

        One indexed hit on UNIQUE(kind, key) - a leftmost-prefix match, so no
        separate index - on the dedicated read connection. Only reached on a
        key's first confident read in this process; the PlateIdentity LRU
        absorbs the rest, so it is not on the hot path.
        """
        k, v = str(kind), str(key)
        pending = self._minted.get((k, v))
        if pending is not None:
            return int(pending)
        row = self._rconn.execute(
            "SELECT id FROM identities WHERE kind=? AND key=?", (k, v)).fetchone()
        return None if row is None else int(row[0])

    def create_identity(self, kind: str, key: str, ts: float = 0.0) -> int:
        """Mint an identity id for a new (kind, key) and queue the row.

        The id comes from the in-memory counter, like next_object_id, so there
        is no round trip; the INSERT goes through the writer thread like every
        other write. The two can disagree only if a key is evicted from the
        identity cache before its INSERT commits, and ON CONFLICT(kind,key)
        makes that harmless: the first row wins and keeps its id.
        """
        k, v = str(kind), str(key)
        existing = self.lookup_identity(k, v)
        if existing is not None:
            # Already known (committed, or minted moments ago). Still queue the
            # row so last_seen_at advances via ON CONFLICT.
            self._put("identities", {"id": existing, "kind": k, "key": v,
                                     "first_seen_at": float(ts or 0.0),
                                     "last_seen_at": float(ts or 0.0)})
            return existing
        vid = self.next_identity_id()
        self._minted[(k, v)] = vid
        while len(self._minted) > self.MINTED_CAP:
            self._minted.popitem(last=False)
        self._put("identities", {"id": vid, "kind": k, "key": v,
                                 "first_seen_at": float(ts or 0.0),
                                 "last_seen_at": float(ts or 0.0)})
        return vid

    def touch_identity(self, identity_id: int, kind: str, key: str,
                       ts: float = 0.0, last_ts: float | None = None) -> None:
        """Widen a known identity's seen window. Idempotent (MIN/MAX upsert).

        Needed because the common case - a key already in the identity cache or
        already in the table - never reaches create_identity, so without this an
        identity's last_seen_at would be frozen at whenever it was first read.

        Pass `last_ts` to submit a whole interval rather than one instant:
        _finalize does that with the sighting's own bounds, so the window covers
        when the ENTITY was visible, not merely when its key happened to be
        legible.

        KNOWN LIMITATION, unchanged by the identities rename: `ts` arrives as
        the run's own first_seen_s, which is a wall-clock epoch for a live
        source and clip-seconds for a file (see src/timebase.py). In one DB
        holding both, MIN() therefore always prefers the clip-seconds value.
        Fixing it means normalising through runs.time_base here and accepting
        NULL for an unorderable file run - a semantic change to the column, so
        it is deliberately NOT bundled into a rename.
        """
        first = float(ts or 0.0)
        last = first if last_ts is None else float(last_ts)
        self._put("identities", {"id": int(identity_id), "kind": str(kind),
                                 "key": str(key),
                                 "first_seen_at": min(first, last),
                                 "last_seen_at": max(first, last)})

    def delete_frames(self, frame_ids) -> None:
        """Queue frame-row deletion (used by the Reaper).

        Routed through the writer thread on purpose: two connections writing
        one SQLite file is how you earn "database is locked" under load.
        """
        ids = [int(i) for i in frame_ids]
        if ids:
            self._q.put(("__delete_frames__", ids))

    # --- the outbox (see src/incidents.py) ---------------------------------
    def put_incident(self, incident: dict, endpoints) -> None:
        """Queue one incident plus a delivery row per subscriber, ATOMICALLY.

        Routed as a single queue item rather than as N ordinary row writes for
        two reasons. First, the dispatcher reads `deliveries` and joins to
        `incidents` for the payload, so a delivery committed before its
        incident is a row pointing at nothing. Second, both tables must appear
        together or not at all - a committed incident with no deliveries is an
        alert that will never be sent and nothing says so.

        An incident with no matching subscriber still gets its row: the record
        is worth keeping (and the dashboard reads it) even when nobody asked to
        be told.
        """
        self._q.put(("__incident__", (dict(incident), list(endpoints))))

    def _write_incident(self, incident: dict, endpoints: list) -> None:
        now = time.time()
        try:
            # OR IGNORE, not a bare INSERT: the id is a natural key, so a
            # double fire for one track is a no-op rather than an exception
            # that would take the batch down with it.
            self._conn.execute(
                "INSERT OR IGNORE INTO incidents"
                "(id,camera,kind,object_id,identity_id,payload_json,created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (str(incident["id"]), incident.get("camera"), incident.get("kind"),
                 incident.get("object_id"), incident.get("identity_id"),
                 json.dumps(incident.get("payload", {}), default=str),
                 float(incident.get("created_at") or now)))
            for endpoint in endpoints:
                self._conn.execute(
                    "INSERT OR IGNORE INTO deliveries"
                    "(incident_id,endpoint,attempts,next_attempt_at,status,created_at)"
                    " VALUES(?,?,0,?,'pending',?)",
                    (str(incident["id"]), str(endpoint), now, now))
            self._conn.commit()
            self.rows_written += 1 + len(endpoints)
        except Exception as e:
            self.rows_failed += 1
            self.write_failures += 1
            self.last_write_error = f"incident: {e}"
            print(f"[sqlite] incident write failed for "
                  f"{incident.get('id')!r}: {e}")

    def due_deliveries(self, limit: int = 50) -> list:
        """Pending deliveries whose backoff has elapsed, oldest first."""
        rows = self._rconn.execute(
            "SELECT d.id, d.incident_id, d.endpoint, d.attempts,"
            "       i.payload_json, i.kind, i.camera"
            " FROM deliveries d JOIN incidents i ON i.id = d.incident_id"
            " WHERE d.status='pending' AND d.next_attempt_at <= ?"
            " ORDER BY d.next_attempt_at ASC LIMIT ?",
            (time.time(), int(limit))).fetchall()
        cols = ("id", "incident_id", "endpoint", "attempts", "payload_json",
                "kind", "camera")
        return [dict(zip(cols, r)) for r in rows]

    def mark_delivery(self, delivery_id: int, status: str, attempts: int,
                      next_attempt_at: float | None = None,
                      error: str | None = None) -> None:
        """Record the outcome of one delivery attempt. Queued, like every write."""
        self._q.put(("__delivery__", (int(delivery_id), str(status), int(attempts),
                                      next_attempt_at, error)))

    def _write_delivery(self, delivery_id, status, attempts, next_at, error):
        try:
            self._conn.execute(
                "UPDATE deliveries SET status=?, attempts=?, next_attempt_at=?,"
                " last_error=?, sent_at=? WHERE id=?",
                (status, attempts, next_at, error,
                 time.time() if status == "sent" else None, delivery_id))
            self._conn.commit()
        except Exception as e:
            print(f"[sqlite] delivery update failed for {delivery_id}: {e}")

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
                if item[0] == "__incident__":
                    # Committed immediately rather than batched: the dispatcher
                    # is polling for these, and holding an alert back for up to
                    # commit_interval seconds to save one fsync is the wrong
                    # trade for something a human is waiting on.
                    count += self._commit(pending)
                    self._write_incident(*item[1])
                    last = time.monotonic()
                    continue
                if item[0] == "__delivery__":
                    count += self._commit(pending)
                    self._write_delivery(*item[1])
                    last = time.monotonic()
                    continue
                if item[0] == "__prune__":
                    count += self._commit(pending)
                    self._run_prune(item[1])
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
        """Write one batch. Per-table isolation, and failures are counted.

        Two deliberate changes from the naive version:

        Per TABLE, not per batch. A schema mismatch on one table used to
        discard every other table's rows in the same batch - up to batch_rows
        (500) unrelated rows per collision. Now one bad table loses only its
        own rows.

        Counted, and escalating. A one-shot run could tolerate a printed line;
        a service running for weeks cannot, because the loss is unbounded and
        invisible. rows_failed/write_failures/last_write_error are readable by
        a health endpoint, and FAILURE_ALARM_AFTER consecutive failures latches
        write_alarm - which is the signal that this is persistent (a schema
        mismatch, a full disk) rather than one transient lock.
        """
        if not pending:
            return 0
        n = failed = 0
        error = None
        for table, rows in pending.items():
            if not rows:
                continue
            try:
                self._conn.executemany(_SQL[table], rows)
                n += len(rows)
            except Exception as e:
                failed += len(rows)
                error = f"{table}: {e}"
                print(f"[sqlite] write failed, dropping {len(rows)} "
                      f"{table} rows: {e}")
        try:
            self._conn.commit()
        except Exception as e:
            failed += n
            n = 0
            error = f"commit: {e}"
            print(f"[sqlite] commit failed, dropping {failed} rows: {e}")
        pending.clear()
        if failed:
            self.rows_failed += failed
            self.write_failures += 1
            self._consecutive_failures += 1
            self.last_write_error = error
            if self._consecutive_failures == self.FAILURE_ALARM_AFTER:
                self.write_alarm = True
                print(f"[sqlite] ALARM: {self._consecutive_failures} consecutive "
                      f"failed writes ({self.rows_failed} rows lost so far). "
                      f"This is not transient - last error: {error}")
        else:
            self._consecutive_failures = 0
        self.rows_written += n
        return n

    def health(self) -> dict:
        """Writer-side counters, for a status endpoint or an end-of-run line."""
        return {"rows_written": self.rows_written,
                "queue_depth": self._q.qsize(),
                "rows_dropped": self.rows_dropped,
                "dropped_by_table": dict(self.dropped_by_table),
                "rows_failed": self.rows_failed,
                "write_failures": self.write_failures,
                "write_alarm": self.write_alarm,
                "last_write_error": self.last_write_error}

    def prune(self, statements: list) -> None:
        """Queue a list of (sql, params) row deletions for the writer thread.

        Kept generic so retention policy lives in storage/retention.py rather
        than here; this end only guarantees the statements run IN ORDER on the
        writer's connection, which is what makes the caller's ordering
        contract (deliveries before incidents, because a delivery references
        an incident and crops are pinned by one) actually hold.
        """
        if statements:
            self._q.put(("__prune__", list(statements)))

    def _run_prune(self, statements: list) -> None:
        for sql, params in statements:
            try:
                self._conn.execute(sql, params)
            except Exception as e:
                print(f"[sqlite] prune failed ({sql.split()[0:3]}): {e}")
        try:
            self._conn.commit()
        except Exception as e:
            print(f"[sqlite] prune commit failed: {e}")

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

    def close(self, end_run: bool = True):
        self._flushed.clear()
        self._q.put(("__stop__", None))
        self._thread.join(timeout=10.0)
        if end_run:
            try:
                self.end_run()
            except Exception:
                pass
        self._conn.close()
        try:
            self._rconn.close()
        except Exception:
            pass

    # --- reads (reports/queries) -------------------------------------------
    def connect_ro(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn
