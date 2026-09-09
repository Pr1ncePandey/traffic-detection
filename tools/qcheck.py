import sqlite3
c = sqlite3.connect("outputs/q_color.db")
print("color rows:", c.execute("SELECT COUNT(*) FROM attributes WHERE key='color'").fetchone()[0])
for r in c.execute("SELECT object_id, value, conf FROM attributes WHERE key='color' ORDER BY object_id"):
    print(r)
print("keys:", [r[0] for r in c.execute("SELECT DISTINCT key FROM attributes")])
