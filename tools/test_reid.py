"""Self-test for plate-keyed vehicle re-identification.

    python tools/test_reid.py

Covers the two halves separately, because they fail in different ways:

  PlateIdentity   the resolver. Driven with plain dicts, no database, so a
                  failure here is unambiguously the identity logic.
  SqliteStore     the persistence: that a plate resolved in one run comes back
                  with the same id in the next, which is the whole claim of
                  "across runs and cameras".

Why any of this needs a test: the failure mode of re-identification is silent.
Bind too loosely and two vehicles' histories merge, which no later frame
undoes and no exception reports. The gate tests below are the ones that matter.
"""

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attributes.plate_format import fits_template
from src.storage.sqlite_store import SqliteStore
from src.trackers.identity import PlateIdentity, from_config
from src.trackers.store import TrackStore

_passed, _failed = 0, []


def check(label, condition, detail=""):
    global _passed
    if condition:
        _passed += 1
        print(f"  ok   {label}")
    else:
        _failed.append(label)
        print(f"  FAIL {label}{('  <- ' + detail) if detail else ''}")


def section(title):
    print(f"\n{title}")


def fake_backend():
    """A dict standing in for the identities table. Returns (lookup, create, rows).

    PlateIdentity is storage-agnostic and takes already-plate-bound callables,
    so these stay single-argument: identities.kind is bound in from_config.
    """
    rows: dict = {}
    counter = {"n": 0}

    def lookup(plate):
        return rows.get(plate)

    def create(plate, ts):
        counter["n"] += 1
        rows[plate] = counter["n"]
        return counter["n"]

    return lookup, create, rows


# ---------------------------------------------------------------------------
section("1. the same plate always resolves to the same vehicle id")
lookup, create, rows = fake_backend()
ident = PlateIdentity(lookup=lookup, create=create)
first = ident.resolve("HR26DK8337", 1.0)
again = ident.resolve("HR26DK8337", 90.0)      # re-entered the frame later
check("a re-entering plate gets its original id back", first == again,
      f"{first} != {again}")
check("and is counted as a resolve hit, not a creation",
      ident.created == 1 and ident.resolve_hits == 1, str(ident.stats()))

section("2. different plates are different vehicles")
other = ident.resolve("MH12AB1234", 2.0)
check("a second plate gets its own id", other != first, f"{other} == {first}")
check("two vehicles created in total", ident.created == 2, str(ident.created))

section("3. formatting differences are not different vehicles")
check("lowercase resolves to the same id",
      ident.resolve("hr26dk8337", 3.0) == first)
check("spaces and dashes resolve to the same id",
      ident.resolve("HR 26 DK-8337", 4.0) == first)
check("no spurious vehicles were created", ident.created == 2,
      str(ident.created))

section("4. an unusable plate is no identity at all")
check("empty string -> None", ident.resolve("", 1.0) is None)
check("None -> None", ident.resolve(None, 1.0) is None)
check("punctuation only -> None", ident.resolve("---", 1.0) is None)
check("still no extra vehicles", ident.created == 2, str(ident.created))

section("5. the cache is bounded, and a miss still resolves correctly")
lookup, create, rows = fake_backend()
small = PlateIdentity(lookup=lookup, create=create, cache_size=2)
a = small.resolve("HR26DK8337", 1.0)
small.resolve("MH12AB1234", 2.0)
small.resolve("DL12C0705", 3.0)                # evicts the first
check("cache never exceeds its cap", small.state_size() <= 2,
      str(small.state_size()))
check("an evicted plate still resolves to its original id via lookup",
      small.resolve("HR26DK8337", 4.0) == a)
check("and did NOT mint a duplicate vehicle", small.created == 3,
      str(small.created))

section("6. a storage failure costs identity, not the run")
def boom(*_a, **_k):
    raise RuntimeError("db down")

broken = PlateIdentity(lookup=boom, create=boom)
check("a failing lookup returns None instead of raising",
      broken.resolve("HR26DK8337", 1.0) is None)
check("and is counted", broken.errors == 1, str(broken.errors))

section("7. fuzzy matching is off by default")
lookup, create, rows = fake_backend()
strict = PlateIdentity(lookup=lookup, create=create)
p1 = strict.resolve("HR26DK8337", 1.0)
p2 = strict.resolve("HR26DK8338", 2.0)         # one character apart: a real car
check("two plates differing by one character stay two vehicles", p1 != p2,
      f"{p1} == {p2} - distinct vehicles were merged")

lookup, create, rows = fake_backend()
loose = PlateIdentity(lookup=lookup, create=create, fuzzy_distance=1)
q1 = loose.resolve("HR26DK8337", 1.0)
check("with fuzzy_distance=1 a one-character slip does merge (opt-in)",
      loose.resolve("HR26DK8338", 2.0) == q1)
q_len = loose.resolve("HR26DK833", 3.0)
check("but a length difference never merges", q_len != q1)

section("8. the binding gate is stricter than the display gate")
# Mirrors pipeline._plate_is_bindable without importing cv2 through pipeline.
def bindable(plate, conf, min_conf=0.7, require_format=True):
    if not plate or float(conf or 0.0) < min_conf:
        return False
    return fits_template(plate) if require_format else True

check("a confident, template-valid plate binds",
      bindable("HR26DK8337", 0.91))
check("a plate below reid.min_conf does not bind",
      not bindable("HR26DK8337", 0.55))
check("an HSRP band fragment does not bind even when confident",
      not bindable("IND22", 0.99))
check("a plate with an illegal series letter does not bind",
      not bindable("MH12IO1234", 0.99))
check("the same fragment DOES bind when require_format is off",
      bindable("IND22", 0.99, require_format=False))

section("9. identity survives across runs against one database")
db = os.path.join(tempfile.mkdtemp(), "reid.db")
meta = {"source": "s", "camera": "c", "fps": 30, "width": 8, "height": 8}

store1 = SqliteStore(db, batch_rows=1, commit_interval=0.05)
store1.start_run(meta)
run1 = from_config({"enabled": True}, store1)
v_run1 = run1.resolve("HR26DK8337", 1.0)
store1.flush()
store1.close()

store2 = SqliteStore(db, batch_rows=1, commit_interval=0.05)
store2.start_run(meta)
run2 = from_config({"enabled": True}, store2)          # cold cache, new process
v_run2 = run2.resolve("HR26DK8337", 100.0)
check("a plate seen in an earlier run keeps its id", v_run1 == v_run2,
      f"{v_run1} != {v_run2}")
check("the second run created no new vehicle", run2.created == 0,
      str(run2.stats()))
# One objects row per visit, both pointing at the one vehicle: this is the
# shape the whole design rests on, so assert it rather than trust it.
for oid_ts, run_track in ((1.0, 4), (100.0, 31)):
    oid = store2.next_object_id()
    store2.upsert_object({"id": oid, "track_id": run_track, "cls_name": "car",
                          "cls_group": "vehicle", "first_seen_s": oid_ts,
                          "last_seen_s": oid_ts + 4, "frames_seen": 10,
                          "best_conf": 0.9, "crop_path": None, "lane_id": None,
                          "lane_flag": None, "identity_id": v_run2})
store2.flush()
conn = store2.connect_ro()
vrows = conn.execute("SELECT id, kind, key, first_seen_at, last_seen_at"
                     " FROM identities").fetchall()
check("exactly one identities row exists", len(vrows) == 1,
      str([tuple(r) for r in vrows]))
check("first_seen_at kept the earliest reading", vrows[0]["first_seen_at"] == 1.0,
      str(vrows[0]["first_seen_at"]))
check("last_seen_at advanced to the later reading",
      vrows[0]["last_seen_at"] == 100.0, str(vrows[0]["last_seen_at"]))
seen = conn.execute("SELECT COUNT(*) n FROM objects WHERE identity_id=?",
                    (v_run2,)).fetchone()["n"]
check("sightings are derivable from objects.identity_id", seen == 2, str(seen))
check("the identity row is on the plate axis", vrows[0]["kind"] == "plate",
      str(vrows[0]["kind"]))
check("and its key is the normalised plate", vrows[0]["key"] == "HR26DK8337",
      str(vrows[0]["key"]))
store2.close()

section("10. a database predating identity_id fails loudly, not silently")
legacy = os.path.join(tempfile.mkdtemp(), "legacy.db")
_c = sqlite3.connect(legacy)
_c.executescript("""CREATE TABLE objects(
  id INTEGER PRIMARY KEY, run_id INT, track_id INT, cls_name TEXT, cls_group TEXT,
  first_seen_s REAL, last_seen_s REAL, frames_seen INT, best_conf REAL,
  crop_path TEXT, lane_id TEXT, lane_flag TEXT, UNIQUE(run_id, track_id));
INSERT INTO objects(id,run_id,track_id,cls_name) VALUES(1,1,7,'car');""")
_c.commit(); _c.close()
# Why this matters: _SQL supplies 13 values for objects, and _commit()
# catches per BATCH - so without the guard an old file loses up to batch_rows
# object rows per commit while the run looks healthy.
try:
    SqliteStore(legacy)
    check("opening a pre-identity_id database raises", False, "no error raised")
except RuntimeError as e:
    check("opening a pre-identity_id database raises RuntimeError", True)
    check("and the message says how to fix it",
          "identity_id" in str(e) and ("Delete" in str(e) or "--db" in str(e)),
          str(e)[:120])
except Exception as e:
    # An OperationalError here means the guard runs too late: SCHEMA's
    # CREATE INDEX ON objects(identity_id) fired first.
    check("the failure is the guard, not a raw sqlite error", False,
          f"{type(e).__name__}: {e}")
check("the legacy rows are left untouched",
      sqlite3.connect(legacy).execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 1)

fresh_db = os.path.join(tempfile.mkdtemp(), "fresh.db")
_fs = SqliteStore(fresh_db)
_tables = {r[0] for r in _fs.connect_ro().execute(
    "SELECT name FROM sqlite_master WHERE type='table'")}
check("a fresh database still gets the identities table", "identities" in _tables,
      str(sorted(_tables)))
_fs.close()

section("11. re-id can be turned off")
check("enabled:false yields no resolver at all",
      from_config({"enabled": False}, None) is None)

section("12. identity state does not leak across evictions")
ts = TrackStore()
ts.touch(1, "car", 0.0, group="vehicle", conf=0.9)
ts.identity_of[1], ts.plate_of[1] = 7, "HR26DK8337"
check("bound state is counted by state_size()", ts.state_size() >= 3,
      str(ts.state_size()))
ts.evict_stale(now_s=9999.0, ttl_s=5.0)
check("and is gone once the track is retired", ts.state_size() == 0,
      f"leaked: identity_of={ts.identity_of} plate_of={ts.plate_of}")

print(f"\n{'=' * 62}")
print(f"{_passed} passed, {len(_failed)} failed")
for name in _failed:
    print(f"  FAILED: {name}")
sys.exit(1 if _failed else 0)
