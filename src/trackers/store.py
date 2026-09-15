"""Per-TRACK memory: lifetime, crossings, attribute dict that grows.

One entry per tracked object of ANY class - car, person, dog - keyed by track
id, with `cls_group` telling them apart exactly as objects.cls_group does in
the database. It was called `vehicles` and documented as per-vehicle memory,
which was false the whole time: pipeline.py calls touch() for every tracked
detection with no class filter, and the person enrichers (person, garments,
age_gender) have always written here. The name also collided with the
`identities` table's former name.

Attributes plug in as enrichers that fill track['attrs'][name]; search
("red Maruti", "man with a backpack") reads this dict later. `attrs` starts
EMPTY on purpose - it used to be seeded with plate_number/plate_conf/color/
brand, so every pedestrian carried four vehicle-only fields. They stayed out
of the database only because _finalize skips empty values; one non-empty
default would have written plate_number='...' rows onto people.
"""

from collections import defaultdict

# ByteTrack keeps a lost track alive for track_buffer frames (30 by default)
# and can still revive it. Evicting inside that window would throw away state
# the tracker is about to reuse, so the TTL is a multiple of it.
TTL_SAFETY = 3.0
TTL_MIN_S = 5.0


def ttl_for(fps: float, track_buffer: int = 30) -> float:
    """Seconds of absence before a track may be retired."""
    fps = float(fps) if fps and fps > 0 else 30.0
    return max(TTL_MIN_S, (float(track_buffer) / fps) * TTL_SAFETY)


class TrackStore:
    def __init__(self):
        self.tracks = {}            # tid -> {cls_name, cls_group, first_seen_s, last_seen_s, frames_seen, attrs{}}
        self.prev_y = {}            # tid -> previous centroid y (for legacy single-line crossing)
        self.prev_xy = {}           # tid -> (prev_cx, prev_cy) for lane direction vectors
        self.in_count = 0
        self.out_count = 0
        self.a_to_b = 0             # zone A (below line_b) -> zone B (above line_a)
        self.b_to_a = 0             # zone B -> zone A
        self.saved_crops = set()
        self._seen_below = set()    # tids ever seen below line_b (side A)
        self._seen_above = set()    # tids ever seen above line_a (side B)
        self._counted = set()       # (tid, direction) already counted
        self.lane_of = {}           # tid -> lane name (latest)
        self.lane_flag_of = {}      # tid -> worst flag latched (wrong_way sticks)
        self._lane_seen = set()     # (lane, tid) pairs -> lane_counts counts unique IDs
        self.lane_counts = defaultdict(int)  # lane name -> unique IDs seen
        self.wrong_way_ids = set()
        self.wrong_lane_ids = set()
        self.object_ids = {}        # tid -> storage object id (survives in the DB)
        # Durable identity. identity_of is the answer to "is this the same car
        # as before"; plate_of remembers which plate string produced it, so a
        # consensus that changes as more frames are voted triggers a rebind
        # instead of silently keeping the first guess. See trackers/identity.py.
        self.identity_of = {}       # tid -> identities.id (cross-run)
        self.plate_of = {}          # tid -> plate string that was bound
        self.evicted = 0           # tracks retired by evict_stale
        # Cumulative, incremented once per track on first sight. self.tracks
        # cannot be used for totals because evict_stale removes from it.
        self.class_totals = defaultdict(int)

    def crossing(self, tid: int, cy: int, line_y: int, line_enabled: bool) -> str:
        prev = self.prev_y.get(tid)
        self.prev_y[tid] = cy
        if not line_enabled or prev is None:
            return ""
        if prev < line_y <= cy:
            self.in_count += 1
            return "IN"
        if prev > line_y >= cy:
            self.out_count += 1
            return "OUT"
        return ""

    def zone_crossing(self, tid: int, cy: int, line_a_y: int, line_b_y: int,
                      enabled: bool) -> str:
        """Two-zone counting. Robust to 'appeared below the line' problem.

        Side A = below line_b (entry, bottom). Side B = above line_a (exit, top).
        A vehicle counts A->B only after being seen on BOTH sides in order.
        Each direction counted once per ID.
        """
        if cy > line_b_y:
            self._seen_below.add(tid)
        if cy < line_a_y:
            self._seen_above.add(tid)
        if not enabled:
            return ""
        if tid in self._seen_below and cy < line_a_y and (tid, "A_TO_B") not in self._counted:
            self._counted.add((tid, "A_TO_B"))
            self.a_to_b += 1
            self.out_count += 1  # alias: upward = legacy OUT
            return "A_TO_B"
        if tid in self._seen_above and cy > line_b_y and (tid, "B_TO_A") not in self._counted:
            self._counted.add((tid, "B_TO_A"))
            self.b_to_a += 1
            self.in_count += 1  # alias: downward = legacy IN
            return "B_TO_A"
        return ""

    def observe(self, tid: int, cx: int, cy: int):
        """Return previous (cx, cy) or (None, None); store current for next frame."""
        prev = self.prev_xy.get(tid, (None, None))
        self.prev_xy[tid] = (cx, cy)
        return prev

    def set_lane(self, tid: int, lane_id: str, flag: str):
        """Latch lane + worst flag (wrong_way sticks even if later frames read ok)."""
        if lane_id:
            if (lane_id, tid) not in self._lane_seen:
                self._lane_seen.add((lane_id, tid))
                self.lane_counts[lane_id] += 1
            self.lane_of[tid] = lane_id
        prev = self.lane_flag_of.get(tid, "ok")
        order = {"ok": 0, "wrong_lane": 1, "wrong_way": 2, "wrong_way+wrong_lane": 3}
        if order.get(flag, 0) >= order.get(prev, 0):
            self.lane_flag_of[tid] = flag
        if "wrong_way" in flag:
            self.wrong_way_ids.add(tid)
        if "wrong_lane" in flag:
            self.wrong_lane_ids.add(tid)

    def touch(self, tid: int, cls_name: str, timestamp: float,
              group: str = "", conf: float = 0.0):
        v = self.tracks.get(tid)
        if v is None:
            self.class_totals[cls_name] += 1
            # attrs is EMPTY: each enricher owns its own keys. See the module
            # docstring for why the old vehicle-only seed was wrong.
            self.tracks[tid] = {"cls_name": cls_name, "cls_group": group,
                                "first_seen_s": round(timestamp, 2),
                                "last_seen_s": round(timestamp, 2), "frames_seen": 1,
                                "best_conf": round(float(conf), 3),
                                "attrs": {}}
        else:
            v["last_seen_s"] = round(timestamp, 2)
            v["frames_seen"] += 1
            if conf > v.get("best_conf", 0.0):
                v["best_conf"] = round(float(conf), 3)
            if group and not v.get("cls_group"):
                v["cls_group"] = group

    def evict_stale(self, now_s: float, ttl_s: float, on_evict=None) -> int:
        """Drop tracks unseen for ttl_s, after handing them to on_evict.

        Why this exists: every container below is keyed by track id and the
        original code never removed anything, so a 24/7 feed grew until the
        process died. ByteTrack mints a NEW id for a reappearing object rather
        than reusing the old one, so ids only ever accumulate.

        ttl_s MUST exceed ByteTrack's own track_buffer (30 frames by default)
        converted to seconds, or we would evict a track the tracker can still
        revive - the caller computes that, see ttl_for().

        on_evict(tid, track) is called BEFORE the state is dropped, so the
        final object row and voted plate can be persisted. Anything it raises
        is swallowed: a storage hiccup must not abort the sweep and leak.
        """
        stale = [tid for tid, v in self.tracks.items()
                 if (now_s - v.get("last_seen_s", 0.0)) > ttl_s]
        for tid in stale:
            track = self.tracks.get(tid)
            if on_evict is not None and track is not None:
                try:
                    on_evict(tid, track)
                except Exception as e:
                    print(f"[store] evict handler failed for track {tid}: {e}")
            self.tracks.pop(tid, None)
            self.prev_y.pop(tid, None)
            self.prev_xy.pop(tid, None)
            self.lane_of.pop(tid, None)
            self.lane_flag_of.pop(tid, None)
            self.object_ids.pop(tid, None)
            self.identity_of.pop(tid, None)
            self.plate_of.pop(tid, None)
            self.saved_crops.discard(tid)
            self._seen_below.discard(tid)
            self._seen_above.discard(tid)
            self._counted.discard((tid, "A_TO_B"))
            self._counted.discard((tid, "B_TO_A"))
            self.wrong_way_ids.discard(tid)
            self.wrong_lane_ids.discard(tid)
        if stale:
            # _lane_seen is keyed (lane, tid), so it needs a scan not a pop.
            dead = set(stale)
            self._lane_seen = {(ln, t) for (ln, t) in self._lane_seen if t not in dead}
        self.evicted += len(stale)
        return len(stale)

    def state_size(self) -> int:
        """Total tracked entries - used to assert memory really is bounded."""
        return (len(self.tracks) + len(self.prev_y) + len(self.prev_xy)
                + len(self.lane_of) + len(self.lane_flag_of) + len(self.object_ids)
                + len(self.identity_of) + len(self.plate_of)
                + len(self.saved_crops) + len(self._seen_below) + len(self._seen_above)
                + len(self._counted) + len(self._lane_seen)
                + len(self.wrong_way_ids) + len(self.wrong_lane_ids))

    def per_class_counts(self) -> dict:
        """Tracks per class over the WHOLE run, including evicted ones."""
        return dict(self.class_totals)
