"""Per-vehicle memory: lifetime, crossings, attribute dict that grows.

Future attributes (plate, color, brand) plug in here as functions that fill
vehicle['attrs'][name]. Search ("red Maruti") reads this dict later.
"""

from collections import defaultdict


class TrackStore:
    def __init__(self):
        self.vehicles = {}          # tid -> {vehicle_class, first_seen_s, last_seen_s, frames_seen, attrs{}}
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

    def touch(self, tid: int, vclass: str, timestamp: float):
        v = self.vehicles.get(tid)
        if v is None:
            self.vehicles[tid] = {"vehicle_class": vclass, "first_seen_s": round(timestamp, 2),
                                   "last_seen_s": round(timestamp, 2), "frames_seen": 1,
                                   "attrs": {"plate_number": "", "plate_conf": 0.0,
                                             "color": "", "brand": ""}}
        else:
            v["last_seen_s"] = round(timestamp, 2)
            v["frames_seen"] += 1

    def per_class_counts(self) -> dict:
        counts = defaultdict(int)
        for v in self.vehicles.values():
            counts[v["vehicle_class"]] += 1
        return dict(counts)
