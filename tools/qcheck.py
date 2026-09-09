import sqlite3
c = sqlite3.connect("outputs/q_clothes.db")
print("persons:", c.execute("SELECT COUNT(*) FROM objects WHERE cls_group='person'").fetchone()[0])
print("attr keys:", c.execute("SELECT key, COUNT(*) FROM attributes GROUP BY key").fetchall())
print("clothes rows:")
for r in c.execute("SELECT o.id, o.cls_name, a.key, a.value, a.conf FROM objects o "
                   "JOIN attributes a ON a.object_id=o.id "
                   "WHERE a.key='upper_color' OR a.key='lower_color' LIMIT 12"):
    print(r)
print("events:", c.execute("SELECT kind, COUNT(*) FROM events GROUP BY kind").fetchall())
for r in c.execute("SELECT frame_id, kind, detail_json FROM events WHERE kind='congestion' LIMIT 6"):
    print(r)
