"""Look up objects in the SQLite record.

  python query.py --list
  python query.py --id 42
  python query.py --class truck
  python query.py --plate HR26AF7196 [--fuzzy]
  python query.py --vehicle 7
  python query.py --group person
  python query.py --flagged

Identity note, and it has two levels:

  --id       ONE SIGHTING. An objects row, scoped to one run. A track id is
             only meaningful inside a run: ByteTrack mints a new one for
             anything that leaves and re-enters.
  --vehicle  ONE CAR, across runs and cameras. A vehicles row, keyed on the
             plate, with every sighting that resolved to it. This is what
             --plate now reports as well.

A vehicle whose plate was never read confidently has no vehicles row at all -
its sightings stay unlinked, which is honest rather than guessed.
"""

import argparse
import os
import re
import sqlite3

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

    if args.id is not None:
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
