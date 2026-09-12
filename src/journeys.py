"""Reconstruct the route one vehicle took across the camera fleet.

Given that the same car resolved to one `vehicle_id` at several cameras, emit
the ordered sequence of cameras it passed with the time and distance between
each.

WHAT THIS ASSERTS, AND WHAT IT DOES NOT

Only OBSERVED HOPS. The stretch between two cameras is reported as a gap, never
as an inferred route: the system saw a car at A and later at D, it did not see
what happened in between, and it does not claim to. There is deliberately no
map-matching onto a road network and no adjacency graph - a topology graph would
let this assert intermediate positions nothing witnessed, which is exactly the
claim being avoided.

OFFLINE AND READ-ONLY

Nothing here touches the pipeline loop, which knows about one camera and must
keep knowing about one camera. This module opens its own read connection and
runs after the fact.

THE HOP SET IS SPARSE AND NOISY

Cross-camera identity is strictly harder than the same-camera case: different
angle, lighting and plate resolution. With `reid.fuzzy_distance` at its default
of 0, two cameras must produce CHARACTER-IDENTICAL voted plate strings to link
at all. So every stage below is built to degrade into "we know less than you'd
like" rather than into a confident wrong answer:

  - a sighting with no comparable clock is set aside and counted, not dropped;
  - a camera with no location yields a distance-unknown hop, not a 0 km one;
  - an impossible hop is FLAGGED, not deleted.

That last point is the load-bearing one. A hop implying 2400 km/h is an OCR
collision or a cloned plate, and hiding it would hide the only evidence that
the identity is wrong. `query.py --path` prints it with its reason attached.
"""

import math
import sqlite3

from .timebase import absolute_time, describe_base

EARTH_RADIUS_KM = 6371.0088

# Defaults mirror DEFAULTS["journey"] in config.py. Duplicated as module
# constants so this stays usable from a script that never loaded a config.
MAX_GAP_S = 1800.0
MAX_SPEED_KMH = 150.0

# Reasons a hop or a journey is not to be trusted. Strings rather than an enum
# so they survive a round trip through JSON to an HTTP client unchanged.
SUSPECT_SPEED = "implied speed above the ceiling"
SUSPECT_OVERLAP = "sightings overlap in time"


def haversine_km(lat1, lon1, lat2, lon2) -> float | None:
    """Great-circle distance in km, or None if either point is unknown.

    Straight-line, not driven distance, and that matters for interpretation:
    the implied speed computed from it is a LOWER BOUND on the real one, since
    any road between two points is at least as long as the line. A hop that
    looks impossible on this measure is therefore impossible in reality too,
    which is the direction that makes the check safe to act on.
    """
    if None in (lat1, lon1, lat2, lon2):
        return None
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = p2 - p1
    dl = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


class Sighting:
    """One `objects` row, with its times resolved to absolute where possible."""

    __slots__ = ("object_id", "run_id", "camera", "cls_name", "plate",
                 "lane_id", "lane_flag", "crop_path", "frames_seen",
                 "lat", "lon", "start", "end", "time_note")

    def __init__(self, row):
        self.object_id = row["id"]
        self.run_id = row["run_id"]
        self.camera = row["camera"] or f"run {row['run_id']}"
        self.cls_name = row["cls_name"]
        self.plate = row["plate"] if "plate" in row.keys() else None
        self.lane_id = row["lane_id"]
        self.lane_flag = row["lane_flag"]
        self.crop_path = row["crop_path"]
        self.frames_seen = row["frames_seen"] or 0
        self.lat = row["cam_lat"]
        self.lon = row["cam_lon"]
        self.start = absolute_time(row, row["first_seen_s"])
        self.end = absolute_time(row, row["last_seen_s"])
        self.time_note = describe_base(row)

    @property
    def orderable(self) -> bool:
        return self.start is not None

    def __repr__(self):
        return f"<Sighting {self.object_id} {self.camera} @ {self.start}>"


class Visit:
    """Consecutive sightings at ONE camera, collapsed into one stay.

    A car lingering in A's view produces several `objects` rows (the tracker
    mints a new id whenever it loses and reacquires the car), and reporting
    those as N separate journey steps would read as N visits to the same place.
    One visit with a dwell time is what actually happened.
    """

    def __init__(self, sighting: Sighting):
        self.camera = sighting.camera
        self.lat = sighting.lat
        self.lon = sighting.lon
        self.start = sighting.start
        self.end = sighting.end if sighting.end is not None else sighting.start
        self.sightings = [sighting]

    def absorb(self, sighting: Sighting):
        self.sightings.append(sighting)
        if sighting.end is not None and sighting.end > self.end:
            self.end = sighting.end
        # Location can be absent on one run and present on another (the yaml
        # gained a location between runs); take whichever we have.
        if self.lat is None and sighting.lat is not None:
            self.lat, self.lon = sighting.lat, sighting.lon

    @property
    def dwell_s(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    @property
    def object_ids(self) -> list:
        return [s.object_id for s in self.sightings]

    def as_dict(self) -> dict:
        return {"camera": self.camera, "lat": self.lat, "lon": self.lon,
                "start": self.start, "end": self.end,
                "dwell_s": round(self.dwell_s, 2),
                "sightings": len(self.sightings),
                "object_ids": self.object_ids,
                "cls_name": self.sightings[0].cls_name}


class Hop:
    """The unobserved stretch between two consecutive visits.

    `distance_km` is None when either camera has no location, and that is
    reported as unknown rather than as zero - a 0 km hop would silently pass
    the implied-speed check that exists to catch plate collisions.
    """

    def __init__(self, frm: Visit, to: Visit, max_speed_kmh: float,
                 overlapping: bool = False):
        self.frm = frm
        self.to = to
        self.distance_km = haversine_km(frm.lat, frm.lon, to.lat, to.lon)
        # Gap between LEAVING the first camera and ARRIVING at the second, not
        # between the two arrivals: dwelling in A's view for a minute is not
        # travel time, and charging it as such deflates the implied speed.
        self.elapsed_s = float(to.start) - float(frm.end)
        self.suspect_reasons: list[str] = []
        if overlapping:
            self.suspect_reasons.append(SUSPECT_OVERLAP)
        speed = self.speed_kmh
        if speed is not None and speed > float(max_speed_kmh):
            self.suspect_reasons.append(SUSPECT_SPEED)

    @property
    def speed_kmh(self) -> float | None:
        """Implied speed, or None if it cannot be computed.

        Requires a known distance AND forward-moving time. A non-positive
        elapsed means the sightings overlap, which is reported as a conflict in
        its own right; dividing by it would produce an infinity that reads like
        a measurement.
        """
        if self.distance_km is None or self.elapsed_s <= 0:
            return None
        return self.distance_km / (self.elapsed_s / 3600.0)

    @property
    def suspect(self) -> bool:
        return bool(self.suspect_reasons)

    def as_dict(self) -> dict:
        speed = self.speed_kmh
        return {"from": self.frm.camera, "to": self.to.camera,
                "distance_km": (None if self.distance_km is None
                                else round(self.distance_km, 3)),
                "elapsed_s": round(self.elapsed_s, 2),
                "speed_kmh": None if speed is None else round(speed, 1),
                "suspect": self.suspect,
                "suspect_reasons": list(self.suspect_reasons),
                "observed": False}


class Journey:
    """One trip: ordered visits, and the gaps between them."""

    def __init__(self, visits: list, max_speed_kmh: float, overlaps: set):
        self.visits = visits
        self.hops = []
        for frm, to in zip(visits, visits[1:]):
            pair = frozenset((id(frm), id(to)))
            self.hops.append(Hop(frm, to, max_speed_kmh,
                                 overlapping=pair in overlaps))

    @property
    def start(self):
        return self.visits[0].start

    @property
    def end(self):
        return self.visits[-1].end

    @property
    def cameras(self) -> list:
        return [v.camera for v in self.visits]

    @property
    def distance_km(self) -> float | None:
        """Summed hop distance, or None if ANY hop is unmeasurable.

        Deliberately not a partial sum: adding up the hops we can measure and
        presenting the result as the journey's length understates it by exactly
        the hops that were skipped, with nothing in the number saying so.
        """
        known = [h.distance_km for h in self.hops]
        if any(d is None for d in known):
            return None
        return sum(known)

    @property
    def suspect(self) -> bool:
        return any(h.suspect for h in self.hops)

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end,
                "duration_s": (None if self.start is None or self.end is None
                               else round(float(self.end) - float(self.start), 2)),
                "cameras": self.cameras,
                "distance_km": (None if self.distance_km is None
                                else round(self.distance_km, 3)),
                "suspect": self.suspect,
                "visits": [v.as_dict() for v in self.visits],
                "hops": [h.as_dict() for h in self.hops]}


class Conflict:
    """Two sightings of one plate that overlap in time.

    Physically impossible: one car cannot be at two cameras at once. So either
    the plate is cloned, or OCR read two different cars as the same string. The
    journey is still built (with the affected hops flagged) because refusing to
    emit one would hide the route as well as the problem - but the conflict is
    reported alongside it, because an operator who cannot see it has no way to
    know the identity is untrustworthy.
    """

    def __init__(self, a: Visit, b: Visit):
        self.a = a
        self.b = b
        self.overlap_s = (min(float(a.end), float(b.end))
                          - max(float(a.start), float(b.start)))

    def as_dict(self) -> dict:
        return {"cameras": [self.a.camera, self.b.camera],
                "windows": [[self.a.start, self.a.end],
                            [self.b.start, self.b.end]],
                "overlap_s": round(self.overlap_s, 2),
                "object_ids": [self.a.object_ids, self.b.object_ids]}


class VehiclePath:
    """Everything known about where one vehicle went."""

    def __init__(self, vehicle_id, plate, journeys, unorderable, conflicts,
                 total_sightings):
        self.vehicle_id = vehicle_id
        self.plate = plate
        self.journeys = journeys
        self.unorderable = unorderable
        self.conflicts = conflicts
        self.total_sightings = total_sightings

    @property
    def cameras(self) -> list:
        """Distinct cameras, in first-seen order."""
        out = []
        for j in self.journeys:
            for cam in j.cameras:
                if cam not in out:
                    out.append(cam)
        return out

    @property
    def multi_camera(self) -> bool:
        return len(self.cameras) > 1

    def as_dict(self) -> dict:
        return {"vehicle_id": self.vehicle_id, "plate": self.plate,
                "total_sightings": self.total_sightings,
                "cameras": self.cameras,
                "journeys": [j.as_dict() for j in self.journeys],
                "unorderable": [{"object_id": s.object_id, "camera": s.camera,
                                 "reason": s.time_note} for s in self.unorderable],
                "conflicts": [c.as_dict() for c in self.conflicts]}


# The join query.py's own _SELECT does not do: objects carries no camera name,
# only run_id, so every camera-aware question needs runs alongside it.
_SIGHTINGS_SQL = """
SELECT o.id, o.run_id, o.cls_name, o.first_seen_s, o.last_seen_s,
       o.frames_seen, o.lane_id, o.lane_flag, o.crop_path,
       r.camera, r.time_base, r.recorded_at, r.cam_lat, r.cam_lon,
       (SELECT value FROM attributes a
         WHERE a.object_id = o.id AND a.key = 'plate_number') plate
FROM objects o
JOIN runs r ON r.id = o.run_id
WHERE o.vehicle_id = ?
"""


def build_path(conn, vehicle_id: int, max_gap_s: float = MAX_GAP_S,
               max_speed_kmh: float = MAX_SPEED_KMH) -> VehiclePath | None:
    """Assemble one vehicle's path. None if the vehicle does not exist.

    The five stages named in the plan, in order: gather, collapse, split,
    validate, emit. Each is a small function below so a caller can reuse one
    without the rest.
    """
    conn.row_factory = sqlite3.Row
    veh = conn.execute("SELECT * FROM vehicles WHERE id=?",
                       (int(vehicle_id),)).fetchone()
    if veh is None:
        return None
    rows = conn.execute(_SIGHTINGS_SQL, (int(vehicle_id),)).fetchall()
    sightings = [Sighting(r) for r in rows]

    # 1. Gather. Set aside anything with no comparable clock, and keep it: a
    #    reported count of unorderable sightings is the difference between
    #    "this car was seen twice" and "this car was seen twice that we can
    #    place in time, plus four we cannot".
    orderable = sorted((s for s in sightings if s.orderable),
                       key=lambda s: s.start)
    unorderable = [s for s in sightings if not s.orderable]

    visits = collapse(orderable)
    conflicts, overlaps = find_conflicts(visits)
    journeys = [Journey(group, max_speed_kmh, overlaps)
                for group in split(visits, max_gap_s)]
    return VehiclePath(vehicle_id=veh["id"], plate=veh["plate"],
                       journeys=journeys, unorderable=unorderable,
                       conflicts=conflicts, total_sightings=len(sightings))


def collapse(sightings: list) -> list:
    """2. Consecutive sightings at the same camera become one Visit."""
    visits: list = []
    for s in sightings:
        if visits and visits[-1].camera == s.camera:
            visits[-1].absorb(s)
        else:
            visits.append(Visit(s))
    return visits


def split(visits: list, max_gap_s: float) -> list:
    """3. Break the visit sequence into trips on an idle gap.

    Without this, a car seen at A in the morning and at A again at night is one
    journey that loops - which reads as a route it never drove. The gap is
    measured from LEAVING one camera to ARRIVING at the next, so a long dwell
    inside one camera's view does not split a trip in half.
    """
    if not visits:
        return []
    groups = [[visits[0]]]
    for prev, cur in zip(visits, visits[1:]):
        if (float(cur.start) - float(prev.end)) > float(max_gap_s):
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return groups


def find_conflicts(visits: list):
    """Visits whose time windows overlap, plus the hop pairs they affect.

    Returns (conflicts, overlapping_pairs). The pair set is keyed by object
    identity rather than camera name because one camera can legitimately appear
    twice in a journey, and only the specific hop between two overlapping
    visits should be flagged.

    Only ADJACENT visits are compared. Anything further apart in an ordered
    sequence cannot overlap without its neighbours overlapping too, so the
    quadratic scan would report the same physical conflict several times.
    """
    conflicts, pairs = [], set()
    for a, b in zip(visits, visits[1:]):
        if a.camera == b.camera:
            continue          # collapse() already merged same-camera runs
        if float(b.start) < float(a.end):
            conflicts.append(Conflict(a, b))
            pairs.add(frozenset((id(a), id(b))))
    return conflicts, pairs


def multi_camera_vehicles(conn, min_cameras: int = 2) -> list:
    """LAYER 0: the go/no-go measurement, as a callable.

    Counts vehicles seen by more than one camera. Zero rows means every layer
    above is scaffolding around an empty set and the real work is plate
    accuracy instead, so this number is worth publishing BEFORE trusting any
    journey output - and it is also the honest headline for whatever this
    feature reports.
    """
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT o.vehicle_id, COUNT(DISTINCT r.camera) cams, COUNT(*) sightings,"
        "       (SELECT plate FROM vehicles v WHERE v.id = o.vehicle_id) plate"
        " FROM objects o JOIN runs r ON r.id = o.run_id"
        " WHERE o.vehicle_id IS NOT NULL"
        " GROUP BY o.vehicle_id HAVING cams >= ?"
        " ORDER BY cams DESC, sightings DESC", (int(min_cameras),)).fetchall()


def yield_summary(conn) -> dict:
    """Layer 0 as one dict: the fleet's cross-camera hop yield.

    `unanchored_runs` is part of the answer, not a footnote. A file run with no
    recorded_at contributes sightings that no journey can ever order, so a low
    hop count with a high unanchored count is a configuration problem, whereas
    a low hop count with none is a plate-accuracy problem. The two need
    different work.
    """
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) n FROM vehicles").fetchone()["n"]
    multi = multi_camera_vehicles(conn)
    cams = conn.execute("SELECT COUNT(DISTINCT camera) n FROM runs").fetchone()["n"]
    unanchored = conn.execute(
        "SELECT COUNT(*) n FROM runs"
        " WHERE time_base = 'clip' AND recorded_at IS NULL").fetchone()["n"]
    no_location = conn.execute(
        "SELECT COUNT(*) n FROM runs WHERE cam_lat IS NULL").fetchone()["n"]
    return {"cameras": cams, "vehicles": total,
            "multi_camera_vehicles": len(multi),
            "unanchored_runs": unanchored, "runs_without_location": no_location,
            "rows": [dict(r) for r in multi]}
