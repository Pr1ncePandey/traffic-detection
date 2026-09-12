"""Look up objects in the SQLite record.

  python query.py --list
  python query.py --id 42
  python query.py --class truck
  python query.py --plate HR26AF7196 [--fuzzy]
  python query.py --vehicle 7
  python query.py --path 7
  python query.py --yield
  python query.py --group person
  python query.py --flagged

Identity note, and it has three levels:

  --id       ONE SIGHTING. An objects row, scoped to one run. A track id is
             only meaningful inside a run: ByteTrack mints a new one for
             anything that leaves and re-enters.
  --vehicle  ONE CAR, across runs and cameras. A vehicles row, keyed on the
             plate, with every sighting that resolved to it. This is what
             --plate now reports as well.
  --path     ONE CAR'S ROUTE. The same sightings as --vehicle, but ordered on
             a comparable clock, collapsed per camera and split into trips.
             Reports only observed hops: the stretch between two cameras is a
             gap, never an inferred route. See src/journeys.py.

A vehicle whose plate was never read confidently has no vehicles row at all -
its sightings stay unlinked, which is honest rather than guessed.

--yield answers the question to ask BEFORE trusting --path: how many vehicles
were actually seen by more than one camera. Cross-camera identity needs two
cameras to produce character-identical voted plates, so the honest expectation
is a sparse and noisy hop set. If that count is zero, the work to do is plate
accuracy, not journeys.
"""

import argparse
import datetime
import os
import re
import sqlite3

from src.journeys import build_path, yield_summary

DB = "outputs/traffic.db"


def parse_args():
    p = argparse.ArgumentParser(description="Query the traffic record")
    p.add_argument("--db", default=DB)
    p.add_argument("--id", type=int, default=None, help="object id")
    p.add_argument("--track", type=int, default=None, help="ByteTrack track id")
    p.add_argument("--class", dest="cls", default=None, help="e.g. car, truck, person")
    p.add_argument("--group", default=None,
                   help="vehicle | person | animal | obstacle | infrastructure")
    p.add_argument("--plate", default=None)
    p.add_argument("--vehicle", type=int, default=None,
                   help="vehicle id - every sighting of one car")
    p.add_argument("--path", type=int, default=None, metavar="V",
                   help="vehicle id - the route it took across the camera "
                        "fleet, as observed hops with gaps marked")
    p.add_argument("--yield", dest="yield_", action="store_true",
                   help="how many vehicles were seen by more than one camera "
                        "(the measurement that makes --path worth reading)")
    p.add_argument("--max-gap-s", type=float, default=None,
                   help="--path: idle seconds that end one trip (default 1800)")
    p.add_argument("--max-speed-kmh", type=float, default=None,
                   help="--path: implied-speed ceiling above which a hop is "
                        "flagged as a suspect identity (default 150)")
    p.add_argument("--fuzzy", action="store_true", help="tolerate up to 2 OCR typos")
    p.add_argument("--flagged", action="store_true", help="wrong-way / wrong-lane only")
    p.add_argument("--list", action="store_true")
    p.add_argument("--limit", type=int, default=50)
    return p.parse_args()


def _norm(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def _hamming(a: str, b: str) -> int:
    return len(a) if len(a) != len(b) else sum(x != y for x, y in zip(a, b))


_SELECT = """
SELECT o.*, (SELECT value FROM attributes a WHERE a.object_id=o.id
              AND a.key='plate_number') plate,
            (SELECT conf FROM attributes a WHERE a.object_id=o.id
              AND a.key='plate_number') plate_conf
FROM objects o
"""


def _show(conn, row):
    print(f"\nObject #{row['id']}  (track {row['track_id']}, run {row['run_id']})")
    vid = row["vehicle_id"] if "vehicle_id" in row.keys() else None
    if vid is not None:
        veh = conn.execute("SELECT plate FROM vehicles WHERE id=?", (vid,)).fetchone()
        n = conn.execute("SELECT COUNT(*) n FROM objects WHERE vehicle_id=?",
                         (vid,)).fetchone()["n"]
        plate = veh["plate"] if veh else "?"
        print(f"  Vehicle   : V{vid} ({plate}) - "
              f"{n} sighting{'s' if n != 1 else ''} on record")
    print(f"  Class     : {row['cls_name']}  [{row['cls_group']}]")
    print(f"  Seen      : {row['frames_seen']} frames, "
          f"{(row['first_seen_s'] or 0):.2f}s -> {(row['last_seen_s'] or 0):.2f}s")
    print(f"  Best conf : {(row['best_conf'] or 0):.3f}")
    print(f"  Lane      : {row['lane_id'] or '-'}  flag={row['lane_flag'] or 'ok'}")
    if row["plate"]:
        print(f"  Plate     : {row['plate']}  (conf {(row['plate_conf'] or 0):.3f})")
    if row["crop_path"]:
        mark = "" if os.path.exists(row["crop_path"]) else "  [file missing]"
        print(f"  Image     : {row['crop_path']}{mark}")
    attrs = conn.execute("SELECT key,value,conf FROM attributes WHERE object_id=?"
                         " AND key<>'plate_number'", (row["id"],)).fetchall()
    if attrs:
        print("  Attributes: " + ", ".join(
            f"{a['key']}={a['value']}({(a['conf'] or 0):.2f})" for a in attrs))
    for e in conn.execute("SELECT kind, ts, detail_json FROM events"
                          " WHERE object_id=? ORDER BY ts", (row["id"],)):
        print(f"  Event     : {e['kind']} @ {(e['ts'] or 0):.2f}s {e['detail_json'] or ''}")
    for b in conn.execute(
            "SELECT d.x1,d.y1,d.x2,d.y2,d.conf,f.frame_no,f.ts"
            " FROM detections d JOIN frames f ON f.id=d.frame_id"
            " WHERE d.object_id=? ORDER BY f.frame_no LIMIT 3", (row["id"],)):
        print(f"  Sighting  : frame {b['frame_no']} @ {(b['ts'] or 0):.2f}s "
              f"box=({b['x1']},{b['y1']},{b['x2']},{b['y2']}) conf={b['conf']:.2f}")


def _clock(ts) -> str:
    """Absolute epoch -> local wall-clock. Times in a path are real times."""
    if ts is None:
        return "?"
    return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _duration(secs) -> str:
    if secs is None:
        return "?"
    secs = float(secs)
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{int(secs // 60)}m{int(secs % 60):02d}s"
    return f"{int(secs // 3600)}h{int((secs % 3600) // 60):02d}m"


def _show_path(path):
    """Render one vehicle's route.

    The gap line between visits is printed with a `~` and the word "unobserved"
    on purpose: it is the one place a reader might otherwise assume the system
    knows the car drove a particular way between two cameras. It does not.
    """
    plate = path.plate or "?"
    print(f"\nVehicle V{path.vehicle_id}  plate {plate}")
    print(f"  {path.total_sightings} sighting(s) on record across "
          f"{len(path.cameras)} camera(s): {', '.join(path.cameras) or '-'}")

    if path.conflicts:
        # Impossible data, shown rather than silently resolved. One car cannot
        # be at two cameras at once, so this plate is cloned or two different
        # cars OCR'd to the same string - either way the identity below is not
        # to be trusted, and the journey is still printed so the evidence is
        # visible.
        print(f"\n  CONFLICT: {len(path.conflicts)} pair(s) of sightings "
              f"overlap in time - one vehicle cannot be at two cameras at "
              f"once, so this plate is cloned or two cars read the same:")
        for c in path.conflicts:
            print(f"    {c.a.camera} {_clock(c.a.start)}..{_clock(c.a.end)}  "
                  f"overlaps  {c.b.camera} {_clock(c.b.start)}..{_clock(c.b.end)}"
                  f"  by {_duration(c.overlap_s)}")

    if not path.journeys:
        print("\n  No journey could be built: no sighting of this vehicle has "
              "a clock that can be compared with another's.")
    for n, j in enumerate(path.journeys, 1):
        suffix = "  [contains suspect hops]" if j.suspect else ""
        # No hops means one camera and no observed travel, so there is no
        # distance to report - printing "0.00 km" would read as a measurement
        # rather than as the absence of one.
        if not j.hops:
            dist = ", single camera"
        elif j.distance_km is None:
            dist = ", distance unknown"
        else:
            dist = f", {j.distance_km:.2f} km"
        print(f"\n  Journey {n}/{len(path.journeys)}: {' -> '.join(j.cameras)}"
              f"{suffix}")
        span = None if j.start is None else float(j.end) - float(j.start)
        print(f"    {_clock(j.start)} -> {_clock(j.end)}  "
              f"({_duration(span)}{dist})")
        for i, visit in enumerate(j.visits):
            loc = ("" if visit.lat is None
                   else f"  ({visit.lat:.5f},{visit.lon:.5f})")
            n_sight = len(visit.sightings)
            print(f"      {visit.camera:<16} {_clock(visit.start)}  "
                  f"dwell {_duration(visit.dwell_s):<7} "
                  f"{n_sight} sighting{'s' if n_sight != 1 else ' '}{loc}")
            if i < len(j.hops):
                _show_hop(j.hops[i])

    if path.unorderable:
        # Counted and named, never dropped: the difference between "seen twice"
        # and "seen twice that we can place in time, plus four we cannot" is
        # the whole difference between a result and a guess.
        print(f"\n  {len(path.unorderable)} sighting(s) set aside as "
              f"unorderable - no comparable clock:")
        for s in path.unorderable:
            print(f"    object #{s.object_id} at {s.camera}: {s.time_note}")
        print("    Fix: re-run those file sources with --recorded-at so their "
              "timestamps can be anchored.")


def _show_hop(hop):
    dist = "distance unknown" if hop.distance_km is None else f"{hop.distance_km:.2f} km"
    speed = hop.speed_kmh
    implied = "" if speed is None else f" -> {speed:.0f} km/h"
    mark = ""
    if hop.suspect:
        mark = "   [SUSPECT: " + "; ".join(hop.suspect_reasons) + "]"
    print(f"      {'~':<16} unobserved: {dist} in {_duration(hop.elapsed_s)}"
          f"{implied}{mark}")


def _show_yield(summary):
    """Layer 0's number, with the two reasons it might be low kept apart."""
    print("Cross-camera yield")
    print(f"  cameras on record          : {summary['cameras']}")
    print(f"  vehicles with a plate      : {summary['vehicles']}")
    print(f"  seen by >1 camera          : {summary['multi_camera_vehicles']}")
    if summary["unanchored_runs"]:
        print(f"\n  {summary['unanchored_runs']} run(s) are file sources with no "
              f"recorded_at, so their sightings can never be ordered against "
              f"another camera. Re-run them with --recorded-at.")
    if summary["runs_without_location"]:
        print(f"  {summary['runs_without_location']} run(s) have no "
              f"camera.location, so hops to or from them report no distance "
              f"and cannot be speed-checked.")
    if not summary["rows"]:
        print("\n  No vehicle has been seen by more than one camera. Journeys "
              "would be scaffolding around an empty set: the work to do is "
              "plate accuracy (two cameras must produce character-identical "
              "voted plates to link at all), not route assembly.")
        return
    print(f"\n{'vehicle':>8} {'cams':>5} {'sightings':>10}  plate")
    for r in summary["rows"]:
        print(f"{('V' + str(r['vehicle_id'])):>8} {r['cams']:>5} "
              f"{r['sightings']:>10}  {r['plate'] or ''}")
    print(f"\n{len(summary['rows'])} vehicle(s) with a cross-camera hop. "
          f"Inspect one: python query.py --path <id>")


def _table(rows):
    if not rows:
        print("No matches.")
        return
    print(f"{'id':>5} {'track':>6} {'veh':>5} {'class':<12} {'group':<14} "
          f"{'frames':>6} {'conf':>5} {'flag':<10} plate")
    for r in rows:
        vid = r["vehicle_id"] if "vehicle_id" in r.keys() else None
        print(f"{r['id']:>5} {r['track_id']:>6} "
              f"{('V' + str(vid)) if vid is not None else '-':>5} "
              f"{str(r['cls_name'] or ''):<12} "
              f"{str(r['cls_group'] or ''):<14} {r['frames_seen']:>6} "
              f"{(r['best_conf'] or 0):>5.2f} {str(r['lane_flag'] or 'ok'):<10} "
              f"{r['plate'] or ''}")
    print(f"\n{len(rows)} row(s).")


def main():
    args = parse_args()
    if not os.path.exists(args.db):
        print(f"{args.db} not found. Run: python main.py")
        return
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    if args.yield_:
        _show_yield(yield_summary(conn))
    elif args.path is not None:
        kwargs = {}
        if args.max_gap_s is not None:
            kwargs["max_gap_s"] = args.max_gap_s
        if args.max_speed_kmh is not None:
            kwargs["max_speed_kmh"] = args.max_speed_kmh
        path = build_path(conn, args.path, **kwargs)
        if path is None:
            print(f"No vehicle V{args.path}. "
                  f"Only a vehicle whose plate was read confidently has a row "
                  f"at all; try: python query.py --yield")
        else:
            _show_path(path)
    elif args.id is not None:
        row = conn.execute(_SELECT + " WHERE o.id=?", (args.id,)).fetchone()
        _show(conn, row) if row else print(f"No object #{args.id}.")
    elif args.track is not None:
        rows = conn.execute(_SELECT + " WHERE o.track_id=? ORDER BY o.run_id DESC",
                            (args.track,)).fetchall()
        if not rows:
            print(f"No track {args.track}.")
        for r in rows:
            _show(conn, r)
    elif args.vehicle is not None:
        veh = conn.execute("SELECT * FROM vehicles WHERE id=?",
                           (args.vehicle,)).fetchone()
        if veh is None:
            print(f"No vehicle V{args.vehicle}.")
        else:
            print(f"Vehicle V{veh['id']}  plate {veh['plate']}")
            print(f"  First seen: {(veh['first_seen_at'] or 0):.2f}"
                  f"   Last seen: {(veh['last_seen_at'] or 0):.2f}")
            rows = conn.execute(_SELECT + " WHERE o.vehicle_id=?"
                                " ORDER BY o.run_id, o.first_seen_s",
                                (args.vehicle,)).fetchall()
            print(f"  {len(rows)} sighting(s):")
            for r in rows:
                _show(conn, r)
    elif args.plate:
        want = _norm(args.plate)
        # Exact hits go through the vehicles table, which is the indexed,
        # cross-run answer. The attributes scan below still runs, so a sighting
        # whose plate was read but never confident enough to bind is not lost.
        veh = conn.execute("SELECT * FROM vehicles WHERE plate=?", (want,)).fetchone()
        if veh is not None:
            print(f"Vehicle V{veh['id']}  plate {veh['plate']}  "
                  f"({(veh['first_seen_at'] or 0):.2f}s -> "
                  f"{(veh['last_seen_at'] or 0):.2f}s)")
        rows = conn.execute(_SELECT + " WHERE o.id IN (SELECT object_id FROM"
                            " attributes WHERE key='plate_number')").fetchall()
        hits = [r for r in rows
                if _norm(r["plate"]) == want
                or (args.fuzzy and _hamming(_norm(r["plate"]), want) <= 2)]
        if not hits:
            print(f"Plate {args.plate} not found"
                  f"{'' if args.fuzzy else ' (try --fuzzy for OCR typos)'}.")
        for r in hits:
            _show(conn, r)
    else:
        where, params = [], []
        if args.cls:
            where.append("LOWER(o.cls_name)=?")
            params.append(args.cls.lower())
        if args.group:
            where.append("LOWER(o.cls_group)=?")
            params.append(args.group.lower())
        if args.flagged:
            where.append("o.lane_flag LIKE '%wrong%'")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        rows = conn.execute(_SELECT + clause + " ORDER BY o.id LIMIT ?",
                            (*params, max(1, args.limit))).fetchall()
        total = conn.execute("SELECT COUNT(*) n FROM objects o" + clause,
                             params).fetchone()["n"]
        _table(rows)
        if len(rows) < total:
            print(f"(showing {len(rows)} of {total}; raise --limit)")
    conn.close()


if __name__ == "__main__":
    main()
