"""Durable vehicle identity: plate -> stable vehicle id.

The gap this fills: ByteTrack matches on Kalman-predicted motion and IoU only.
It holds a lost track for `track_buffer` frames and then deletes it, so a
vehicle that leaves and re-enters gets a fresh, higher id and is counted as a
new object. The plate is the only identity the system reads that survives an
absence, and until now it was used only as a post-hoc join key in query.py.

TWO LEVELS, AND THE DIFFERENCE IS THE POINT

  track / object   one SIGHTING. Still one `objects` row per track, exactly as
                   before, so throughput counting is untouched: a vehicle that
                   legitimately passes twice really did pass twice.
  vehicle          one ENTITY, keyed on the plate. Several `objects` rows point
                   at it via objects.vehicle_id.

Splitting them rather than re-using the object id is what makes this cheap.
The plate is NOT known when a track is born - with read_every=3/max_reads=8 it
takes ~24 frames with a readable plate box, and plate._consensus() only votes
once there are three reads above vote_min_conf. Re-using the object id would
mean buffering every detection row until the plate resolved, or rewriting rows
afterwards. Attaching a vehicle id instead means detections.object_id is never
rewritten - identity lands at finalize time, when voting is already done.

MEMORY: the cache is an LRU with a hard cap. A busy road produces unboundedly
many plates, and preloading the table (or never evicting) is the same bug this
project already fixed once in TrackStore.evict_stale. A cache miss costs one
indexed SELECT on a UNIQUE column, and only ever happens on a plate's first
confident read in this process.
"""

from collections import OrderedDict

from ..attributes.plate_format import normalize

CACHE_SIZE = 4096


class PlateIdentity:
    """Resolves a plate string to a stable vehicle id.

    Storage-agnostic on purpose: it takes a `lookup(plate) -> id | None` and a
    `create(plate, ts) -> id` callable rather than importing sqlite3, so
    tools/test_reid.py can drive it with plain dicts and no database.

    Not thread-safe, and does not need to be. The only caller is the pipeline
    loop, on the main thread, between the analysis stage and the draw pass.
    Binding lives there rather than in the ANPR analyzer because analysis/base.py
    is explicit that an analyzer must not see the storage backend - and because
    identity is the same concern as the object ids assigned just above it.
    """

    def __init__(self, lookup=None, create=None, touch=None,
                 cache_size: int = CACHE_SIZE, fuzzy_distance: int = 0):
        self._lookup = lookup
        self._create = create
        self._touch = touch
        self.cache_size = max(1, int(cache_size or CACHE_SIZE))
        # 0 = exact match only, and that is the right default. Real plates
        # genuinely differ by one character (...8337 and ...8338 are two cars),
        # so a Hamming-1 merge does not repair OCR noise, it fuses strangers.
        # Raise it only for footage where you have measured that it helps.
        self.fuzzy_distance = max(0, int(fuzzy_distance or 0))
        self._cache: OrderedDict[str, int] = OrderedDict()
        self.created = 0       # plate never seen -> new vehicle minted
        # Resolves that found an existing id. NOT a re-entry count: the same
        # track resolves repeatedly (once when its plate first reads, again
        # when it retires), so this counts lookups, not distinct vehicles
        # returning. The exact re-entry figure needs the objects table -
        # report.py computes it as vehicles having >1 objects row.
        self.resolve_hits = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0

    # --- cache -------------------------------------------------------------
    def _remember(self, plate: str, vehicle_id: int):
        self._cache[plate] = vehicle_id
        self._cache.move_to_end(plate)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _cached(self, plate: str):
        vid = self._cache.get(plate)
        if vid is not None:
            self._cache.move_to_end(plate)
            return vid
        if self.fuzzy_distance:
            return self._fuzzy(plate)
        return None

    def _fuzzy(self, plate: str):
        """Nearest cached plate within fuzzy_distance, or None if ambiguous.

        Same-length only - a length difference is a dropped or hallucinated
        character, not a substitution, and treating it as one is how 'MH12AB123'
        becomes some other car. An ambiguous match (two candidates equally
        close) is declined rather than guessed at, mirroring plate_format's
        window rule.
        """
        best, best_d, ties = None, self.fuzzy_distance + 1, 0
        for known, vid in self._cache.items():
            if len(known) != len(plate):
                continue
            d = sum(a != b for a, b in zip(known, plate))
            if d < best_d:
                best, best_d, ties = vid, d, 1
            elif d == best_d:
                ties += 1
        if best is None or best_d > self.fuzzy_distance or ties > 1:
            return None
        return best

    # --- resolution --------------------------------------------------------
    def resolve(self, plate: str, ts: float = 0.0):
        """plate -> vehicle id. None when the plate is unusable.

        Callers gate on confidence and plate_format.fits_template BEFORE
        calling; this decides identity, not readability.

        A failing lookup/create is swallowed and counted: a storage hiccup must
        cost the vehicle its durable id for this sighting, not abort the run.
        """
        key = normalize(plate)
        if not key:
            return None
        vid = self._cached(key)
        if vid is not None:
            self.cache_hits += 1
            self.resolve_hits += 1
            self._touched(key, vid, ts)
            return vid
        self.cache_misses += 1
        try:
            vid = self._lookup(key) if self._lookup is not None else None
        except Exception as e:
            self.errors += 1
            print(f"[identity] lookup failed for {key}: {e}")
            return None
        if vid is not None:
            self.resolve_hits += 1
            self._remember(key, int(vid))
            self._touched(key, int(vid), ts)
            return int(vid)
        if self._create is None:
            return None
        try:
            vid = self._create(key, ts)
        except Exception as e:
            self.errors += 1
            print(f"[identity] create failed for {key}: {e}")
            return None
        if vid is None:
            return None
        self.created += 1
        self._remember(key, int(vid))
        return int(vid)

    def _touched(self, plate: str, vehicle_id: int, ts: float):
        """Let storage advance the vehicle's last-seen window on a reuse.

        Reuse is the whole point of this class and it never calls create(), so
        without this hook a vehicle's last_seen_at would stay pinned to the
        first time it was ever read. Best-effort: a failure here loses a
        timestamp, which must not cost the caller its identity.
        """
        if self._touch is None:
            return
        try:
            self._touch(vehicle_id, plate, ts)
        except Exception as e:
            self.errors += 1
            print(f"[identity] touch failed for {plate}: {e}")

    # --- introspection -----------------------------------------------------
    def state_size(self) -> int:
        """Cached entries. TrackStore.state_size()'s counterpart: the point is
        to be able to assert in a long run that this really is bounded."""
        return len(self._cache)

    def stats(self) -> dict:
        return {"vehicles_created": self.created,
                "resolve_hits": self.resolve_hits,
                "cache_entries": len(self._cache), "cache_cap": self.cache_size,
                "cache_hits": self.cache_hits, "cache_misses": self.cache_misses,
                "errors": self.errors}


def from_config(cfg: dict, storage=None) -> "PlateIdentity | None":
    """Build from the `reid:` block, wired to a SqliteStore. None when off.

    Returning None rather than a disabled object is deliberate: the pipeline
    then has one `if identity is None` check instead of every call site having
    to know about an enabled flag.
    """
    rc = dict(cfg or {})
    if not rc.get("enabled", True):
        return None
    lookup = create = touch = None
    if storage is not None:
        lookup = getattr(storage, "lookup_vehicle", None)
        create = getattr(storage, "create_vehicle", None)
        touch = getattr(storage, "touch_vehicle", None)
        if lookup is None or create is None:
            print("[identity] storage cannot persist vehicles; re-id disabled")
            return None
    return PlateIdentity(lookup=lookup, create=create, touch=touch,
                         cache_size=int(rc.get("cache_size") or CACHE_SIZE),
                         fuzzy_distance=int(rc.get("fuzzy_distance") or 0))
