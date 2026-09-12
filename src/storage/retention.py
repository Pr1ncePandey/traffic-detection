"""Retention for crops and for rows. The half the frame Reaper never covered.

TWO BUGS THIS FIXES, BOTH INVISIBLE UNTIL SOMETHING RUNS FOR A WEEK

**Crops were never reaped at all.** `frames.Reaper` applies its age and byte
budgets to the `frames` table and nothing else - the module contains no
reference to crops. So `outputs/{camera_id}/crops/object_N.jpg` accumulated one
JPEG per object forever. Correct for a finite clip; an unbounded disk leak in a
service, and the same bug the frame reaper exists to prevent.

**Rows were never capped.** `events` in particular is high volume - `crossing`
fires per vehicle per counting line - and had no budget of any kind.

THE CROP POLICY: KEEP WHAT AN INCIDENT REFERENCES, REAP THE REST

An ordinary crop ages out on a normal age/disk budget. A crop referenced by a
retained incident survives until that incident is itself reaped, so
`image_url` in a delivered payload is valid for exactly as long as the incident
it belongs to.

The alternative - copying the image into an incident-owned directory when it
fires - stores the bytes twice, and a stale 404 in a payload someone already
received is a worse failure than a reaper consulting one more table.

Two consequences of that join, both deliberate:

  - A crop's eligibility is no longer a function of its own age, so the
    byte-targeted pass cannot assume the oldest file is always deletable. It
    SKIPS pinned crops and keeps going rather than stopping at the first one.
  - With incident retention unlimited by default, a long deployment pins a crop
    for every incident ever raised. That is intended, but it means incident
    volume sets a disk floor the crop budget cannot reclaim - so `stats()`
    reports pinned bytes separately from reclaimable ones, and the dashboard
    shows both. The floor should be visible before it is a full disk.

ROW RETENTION: UNLIMITED BY DEFAULT

`0` means unlimited, matching `frames.retention` exactly so there is one
retention idiom in the codebase rather than two. Nothing is deleted that an
operator did not ask to have deleted.

DELETE ORDER IS LOAD-BEARING: deliveries reference an incident, and crops are
pinned by one, so an incident must be the LAST thing to go. Deleting incidents
first would orphan delivery rows and unpin crops that a still-retryable
delivery is about to advertise.
"""

import os
import threading
import time

# Tables this module can cap, in SAFE DELETE ORDER. `deliveries` before
# `incidents` because a delivery row references one; see the note above.
ROW_TABLES = ("events", "deliveries", "incidents")

_AGE_COLUMN = {"events": "ts", "deliveries": "created_at",
               "incidents": "created_at"}


class CropReaper:
    """Age/byte budget over object crops, skipping those an incident pins."""

    def __init__(self, store, max_age_hours: float = 0.0,
                 max_disk_gb: float = 0.0):
        self.store = store
        self.max_age_s = float(max_age_hours or 0) * 3600.0
        self.max_bytes = float(max_disk_gb or 0) * (1024 ** 3)
        self.deleted = 0
        self.deleted_bytes = 0
        self.pinned_skipped = 0

    @property
    def enabled(self) -> bool:
        return self.max_age_s > 0 or self.max_bytes > 0

    def _pinned(self, conn) -> set:
        """object_ids referenced by a surviving incident.

        One query rather than a per-file lookup: the set is small (one entry
        per incident) and a join per candidate crop would make a byte-targeted
        pass over thousands of files quadratic in database round trips.
        """
        return {r[0] for r in conn.execute(
            "SELECT DISTINCT object_id FROM incidents"
            " WHERE object_id IS NOT NULL")}

    def reap_once(self) -> int:
        if not self.enabled:
            return 0
        conn = self.store.connect_ro()
        try:
            pinned = self._pinned(conn)
            candidates = conn.execute(
                "SELECT id, crop_path, last_seen_s FROM objects"
                " WHERE crop_path IS NOT NULL ORDER BY id ASC").fetchall()
        finally:
            conn.close()

        removed = 0
        freed = 0.0
        cutoff = time.time() - self.max_age_s if self.max_age_s > 0 else None
        total = self._total_bytes(candidates)
        over = (total - self.max_bytes) if self.max_bytes > 0 else 0.0
        cleared: list = []

        for row in candidates:
            path = row["crop_path"]
            if not path or not os.path.exists(path):
                continue
            if row["id"] in pinned:
                self.pinned_skipped += 1
                continue          # keep going: a later crop may be eligible
            try:
                stat = os.stat(path)
            except OSError:
                continue
            aged_out = cutoff is not None and stat.st_mtime < cutoff
            need_bytes = over > 0 and freed < over
            if not (aged_out or need_bytes):
                continue
            try:
                os.unlink(path)
            except OSError as e:
                print(f"[crops] could not delete {path}: {e}")
                continue
            freed += stat.st_size
            removed += 1
            cleared.append(row["id"])

        if cleared:
            # Null the column so a query does not advertise a file that is
            # gone. Batched to bound statement size.
            self.store.prune([
                ("UPDATE objects SET crop_path=NULL WHERE id IN ("
                 + ",".join("?" * len(chunk)) + ")", tuple(chunk))
                for chunk in _chunks(cleared, 400)])
        self.deleted += removed
        self.deleted_bytes += freed
        return removed

    @staticmethod
    def _total_bytes(rows) -> float:
        total = 0.0
        for r in rows:
            try:
                total += os.path.getsize(r["crop_path"])
            except (OSError, TypeError):
                pass
        return total

    def stats(self, conn=None) -> dict:
        """Pinned vs reclaimable bytes, kept apart on purpose.

        One number for "crop bytes on disk" would hide the floor that incident
        retention sets: an operator seeing 40 GB of crops needs to know how
        much of it the byte budget is even allowed to touch.
        """
        owned = conn is None
        conn = conn or self.store.connect_ro()
        try:
            pinned = self._pinned(conn)
            rows = conn.execute("SELECT id, crop_path FROM objects"
                                " WHERE crop_path IS NOT NULL").fetchall()
        finally:
            if owned:
                conn.close()
        pinned_bytes = free_bytes = 0
        pinned_n = free_n = 0
        for r in rows:
            try:
                size = os.path.getsize(r["crop_path"])
            except (OSError, TypeError):
                continue
            if r["id"] in pinned:
                pinned_bytes += size
                pinned_n += 1
            else:
                free_bytes += size
                free_n += 1
        return {"enabled": self.enabled,
                "pinned_crops": pinned_n, "pinned_mb": round(pinned_bytes / 1e6, 1),
                "reclaimable_crops": free_n,
                "reclaimable_mb": round(free_bytes / 1e6, 1),
                "deleted": self.deleted,
                "deleted_mb": round(self.deleted_bytes / 1e6, 1),
                "pinned_skipped": self.pinned_skipped}


class RowReaper:
    """Age caps over events / deliveries / incidents. 0 = unlimited."""

    def __init__(self, store, policy: dict | None = None):
        self.store = store
        self.policy = {t: dict((policy or {}).get(t, {}) or {}) for t in ROW_TABLES}
        self.deleted: dict = {t: 0 for t in ROW_TABLES}

    @property
    def enabled(self) -> bool:
        return any(float(p.get("max_age_hours", 0) or 0) > 0
                   for p in self.policy.values())

    def reap_once(self) -> int:
        statements = []
        counted = 0
        conn = self.store.connect_ro()
        try:
            # ROW_TABLES order IS the delete order. Do not sort this.
            for table in ROW_TABLES:
                hours = float(self.policy.get(table, {}).get("max_age_hours", 0) or 0)
                if hours <= 0:
                    continue
                cutoff = time.time() - hours * 3600.0
                column = _AGE_COLUMN[table]
                # Never delete an incident whose delivery is still pending:
                # that would drop the payload out from under a retry and unpin
                # a crop the payload advertises.
                guard = ("" if table != "incidents" else
                         " AND id NOT IN (SELECT incident_id FROM deliveries"
                         " WHERE status='pending')")
                # The COUNT carries the SAME guard as the DELETE. Counting
                # without it reports rows as deleted that the guard then
                # spares, so the retention numbers on the dashboard would
                # drift upward from reality on every pass.
                n = conn.execute(f"SELECT COUNT(*) FROM {table}"
                                 f" WHERE {column} < ?{guard}",
                                 (cutoff,)).fetchone()[0]
                if not n:
                    continue
                statements.append(
                    (f"DELETE FROM {table} WHERE {column} < ?{guard}", (cutoff,)))
                self.deleted[table] += n
                counted += n
        finally:
            conn.close()
        if statements:
            self.store.prune(statements)
        return counted

    def stats(self) -> dict:
        return {"enabled": self.enabled, "deleted": dict(self.deleted),
                "policy": {t: p for t, p in self.policy.items() if p}}


class RetentionService:
    """One thread running both reapers on an interval."""

    def __init__(self, store, crop_reaper: CropReaper, row_reaper: RowReaper,
                 interval: float = 60.0):
        self.store = store
        self.crops = crop_reaper
        self.rows = row_reaper
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self.crops.enabled or self.rows.enabled

    def start(self):
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._loop, name="retention",
                                        daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.crops.reap_once()
                self.rows.reap_once()
            except Exception as e:
                print(f"[retention] pass failed: {e}")

    def stats(self) -> dict:
        return {"crops": self.crops.stats(), "rows": self.rows.stats()}

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def from_config(cfg: dict, store) -> RetentionService:
    """Build the retention service from the `retention` config block."""
    ret = (cfg or {}).get("retention", {}) or {}
    crops = ret.get("crops", {}) or {}
    return RetentionService(
        store,
        CropReaper(store, max_age_hours=crops.get("max_age_hours", 0),
                   max_disk_gb=crops.get("max_disk_gb", 0)),
        RowReaper(store, {t: ret.get(t, {}) for t in ROW_TABLES}),
        interval=float(ret.get("check_interval", 60)))
