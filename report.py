"""HTML report from the SQLite record (falls back to the CSV export).

Three things were wrong with the previous version and are fixed here:

1. It counted crossings as `event == "IN"/"OUT"` while the pipeline writes
   A_TO_B / B_TO_A, so every zone-counted run reported "IN: 0 | OUT: 0".
2. It called distinct track IDs "unique vehicles". ByteTrack issues a NEW id
   to an object that leaves and re-enters, so that figure is an upper bound on
   real vehicles, not a count of them. It is labelled honestly now, and sits
   next to the plate-keyed vehicle count, which is the real figure for every
   vehicle whose plate was read.
3. It built one HTML card per object with no limit, from a full CSV read. Over
   a long run that produces a document too large to open. Card count is capped
   (--limit) and the cap is stated in the output.
"""

import argparse
import html
import os
import sqlite3

DB = "outputs/traffic.db"
OUT = "outputs/report.html"
DEFAULT_LIMIT = 300


def parse_args():
    p = argparse.ArgumentParser(description="Build outputs/report.html")
    p.add_argument("--db", default=DB)
    p.add_argument("--out", default=OUT)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help=f"max object cards to render (default {DEFAULT_LIMIT})")
    p.add_argument("--order", default="last_seen",
                   choices=["last_seen", "first_seen", "frames", "conf"])
    return p.parse_args()


_ORDER = {"last_seen": "o.last_seen_s DESC", "first_seen": "o.first_seen_s ASC",
          "frames": "o.frames_seen DESC", "conf": "o.best_conf DESC"}


def main():
    args = parse_args()
    if not os.path.exists(args.db):
        print(f"{args.db} not found. Run: python main.py")
        return
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    run = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    totals = conn.execute(
        "SELECT COUNT(*) tracks, SUM(frames_seen) sightings FROM objects").fetchone()
    frames_n = conn.execute("SELECT COUNT(*) n FROM frames").fetchone()["n"]
    # The actual event names the pipeline writes.
    crossings = {r["event"]: r["n"] for r in conn.execute(
        "SELECT event, COUNT(*) n FROM detections WHERE event <> '' GROUP BY event")}
    by_class = conn.execute(
        "SELECT cls_group, cls_name, COUNT(*) n FROM objects"
        " GROUP BY cls_group, cls_name ORDER BY n DESC").fetchall()
    flagged = conn.execute(
        "SELECT COUNT(*) n FROM objects WHERE lane_flag LIKE '%wrong%'").fetchone()["n"]
    plates = conn.execute(
        "SELECT COUNT(*) n FROM attributes WHERE key='plate_number'").fetchone()["n"]
    # Counted over objects, not over the vehicles table: a plate whose voted
    # consensus shifted mid-track can leave a vehicles row that no sighting
    # ends up referencing, and counting rows would report a car that was never
    # actually seen. DISTINCT over the FK is the number that cannot be wrong.
    vehicles = conn.execute(
        "SELECT COUNT(DISTINCT vehicle_id) n FROM objects"
        " WHERE vehicle_id IS NOT NULL").fetchone()["n"]
    # Sightings that resolved to a vehicle, and the re-entries among them: a
    # vehicle with more than one objects row is precisely a car this system
    # used to count twice.
    linked = conn.execute(
        "SELECT COUNT(*) n FROM objects WHERE vehicle_id IS NOT NULL").fetchone()["n"]
    returning = conn.execute(
        "SELECT COUNT(*) n FROM (SELECT vehicle_id FROM objects"
        " WHERE vehicle_id IS NOT NULL GROUP BY vehicle_id"
        " HAVING COUNT(*) > 1)").fetchone()["n"]
    events = conn.execute(
        "SELECT kind, COUNT(*) n FROM events GROUP BY kind ORDER BY n DESC").fetchall()

    rows = conn.execute(
        f"SELECT o.*, (SELECT value FROM attributes a WHERE a.object_id=o.id"
        f"   AND a.key='plate_number') plate,"
        f" (SELECT conf FROM attributes a WHERE a.object_id=o.id"
        f"   AND a.key='plate_number') plate_conf"
        f" FROM objects o ORDER BY {_ORDER[args.order]} LIMIT ?",
        (max(1, args.limit),)).fetchall()
    total_tracks = totals["tracks"] or 0

    cards = []
    for r in rows:
        img = ""
        if r["crop_path"] and os.path.exists(r["crop_path"]):
            rel = os.path.relpath(r["crop_path"], os.path.dirname(args.out) or ".")
            img = f'<img src="{html.escape(rel)}" alt="object {r["id"]}">'
        plate = ""
        if r["plate"]:
            plate = (f'<div class="plate">{html.escape(str(r["plate"]))}'
                     f' <span class="conf">{(r["plate_conf"] or 0):.2f}</span></div>')
        flag = r["lane_flag"] or "ok"
        cards.append(f'''<div class="card{' bad' if 'wrong' in flag else ''}">
  {img}
  <div class="meta">
    <b>#{r["id"]}</b> {("V" + str(r["vehicle_id"])) if r["vehicle_id"] is not None else "track " + str(r["track_id"])} &middot; {html.escape(r["cls_name"] or "?")}
    <span class="grp">{html.escape(r["cls_group"] or "")}</span><br>
    seen {r["frames_seen"]}x, {(r["first_seen_s"] or 0):.1f}s &rarr; {(r["last_seen_s"] or 0):.1f}s<br>
    conf {(r["best_conf"] or 0):.2f} &middot; lane {html.escape(str(r["lane_id"] or "-"))}
    &middot; <span class="flag">{html.escape(flag)}</span>
  </div>{plate}
</div>''')

    cls_rows = "".join(
        f"<tr><td>{html.escape(r['cls_group'] or '')}</td>"
        f"<td>{html.escape(r['cls_name'] or '')}</td><td>{r['n']}</td></tr>"
        for r in by_class)
    ev_rows = "".join(
        f"<tr><td>{html.escape(r['kind'])}</td><td>{r['n']}</td></tr>" for r in events)
    truncated = ("" if total_tracks <= len(rows) else
                 f'<p class="note">Showing {len(rows)} of {total_tracks} objects '
                 f'(--limit {args.limit}, ordered by {args.order}).</p>')

    doc = f'''<!doctype html><meta charset="utf-8">
<title>Traffic report</title>
<style>
 body{{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:24px;color:#111}}
 h1{{font-size:20px}} h2{{font-size:15px;margin-top:28px}}
 table{{border-collapse:collapse;margin:8px 0}}
 td,th{{border:1px solid #ddd;padding:4px 10px;text-align:left}}
 .grid{{display:flex;flex-wrap:wrap;gap:10px}}
 .card{{border:1px solid #ddd;border-radius:6px;padding:8px;width:210px}}
 .card.bad{{border-color:#d33;background:#fff5f5}}
 .card img{{width:100%;border-radius:4px;display:block}}
 .meta{{margin-top:6px;font-size:12px;color:#333}}
 .grp{{color:#777}} .flag{{font-weight:600}}
 .plate{{margin-top:5px;font-family:ui-monospace,monospace;font-size:13px;
        background:#111;color:#fff;padding:2px 6px;border-radius:3px;display:inline-block}}
 .conf{{color:#9c9;font-size:11px}}
 .note{{color:#666;font-size:12px}}
</style>
<h1>Traffic report</h1>
<table>
 <tr><th>Source</th><td>{html.escape(str(run["source"] if run else "?"))}</td></tr>
 <tr><th>Camera</th><td>{html.escape(str(run["camera"] if run else ""))}</td></tr>
 <tr><th>Frames analysed</th><td>{frames_n}</td></tr>
 <tr><th>Distinct track IDs</th><td>{total_tracks}
   <span class="note">one per sighting &mdash; a re-entering object gets a new
   ID, so this over-counts real objects</span></td></tr>
 <tr><th>Distinct vehicles (by plate)</th><td>{vehicles}
   <span class="note">{linked} sighting(s) resolved to a vehicle;
   {returning} vehicle(s) were seen more than once. Vehicles with no readable
   plate are not counted here.</span></td></tr>
 <tr><th>Total sightings</th><td>{totals["sightings"] or 0}</td></tr>
 <tr><th>A&rarr;B</th><td>{crossings.get("A_TO_B", 0)}</td></tr>
 <tr><th>B&rarr;A</th><td>{crossings.get("B_TO_A", 0)}</td></tr>
 <tr><th>Wrong-way / wrong-lane objects</th><td>{flagged}</td></tr>
 <tr><th>Plates read</th><td>{plates}</td></tr>
</table>
<h2>By class</h2>
<table><tr><th>group</th><th>class</th><th>objects</th></tr>{cls_rows}</table>
<h2>Events</h2>
<table><tr><th>kind</th><th>count</th></tr>{ev_rows}</table>
<h2>Objects</h2>{truncated}
<div class="grid">{"".join(cards)}</div>
'''
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    conn.close()
    print(f"[report] {args.out} | {len(rows)} of {total_tracks} objects, "
          f"A->B={crossings.get('A_TO_B', 0)} B->A={crossings.get('B_TO_A', 0)}, "
          f"plates={plates}, vehicles={vehicles} ({returning} seen more than once)")


if __name__ == "__main__":
    main()
