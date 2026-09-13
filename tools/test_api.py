"""Self-test for the HTTP API surface (src/server/app.py).

    python tools/test_api.py
    python tools/test_api.py --db outputs/api_test.db

Drives the app in-process with FastAPI's TestClient: no uvicorn, no free port,
no YOLO worker. `autostart=False` is what keeps this fast - the routes are
tested against a database, not against a running pipeline.

Needs a database written by the CURRENT schema. SqliteStore deliberately
refuses to open one that predates comparable cross-camera timestamps, so a DB
from before the timebase change will be reported here rather than silently
producing empty responses. Make one with:

    python main.py --camera indian_road --max-frames 80 --no-frames \
        --recorded-at 2026-09-13T08:30:00 --db outputs/api_test.db

Skips cleanly if fastapi is not installed or no usable database exists.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from fastapi.testclient import TestClient
    _fastapi_error = None
except Exception as e:
    _fastapi_error = e

if _fastapi_error is not None:
    print(f"SKIPPED: fastapi not installed ({_fastapi_error})\n"
          f"  pip install -r requirements.txt")
    sys.exit(0)

import sqlite3

import src.server.app as appmod

_passed, _failed = 0, []


def check(label, condition, detail=""):
    global _passed
    if condition:
        _passed += 1
        print(f"  ok   {label}")
    else:
        _failed.append(label)
        print(f"  FAIL {label}{('  <- ' + detail) if detail else ''}")


def usable(path: str) -> bool:
    """Current-schema and non-empty? The store rejects old DBs on open."""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cols = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
        n = con.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
        con.close()
        return "time_base" in cols and n > 0
    except Exception:
        return False


def find_db() -> str | None:
    for path in ["outputs/api_test.db"] + sorted(glob.glob("outputs/*.db")):
        if os.path.exists(path) and usable(path):
            return path
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--db", default=None)
args = ap.parse_args()

DB = args.db or find_db()
if DB is None:
    print("SKIPPED: no current-schema database with data in outputs/.\n"
          "  python main.py --camera indian_road --max-frames 80 --no-frames \\\n"
          "      --recorded-at 2026-09-13T08:30:00 --db outputs/api_test.db")
    sys.exit(0)
print(f"database: {DB}")

# Point the app at that DB without editing config.yaml.
_orig_load = appmod.load_for_camera


def _patched(camera, path="config.yaml"):
    cfg = _orig_load(camera, path)
    cfg.setdefault("storage", {})["path"] = DB
    return cfg


appmod.load_for_camera = _patched

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
CAM = (con.execute("SELECT camera FROM runs WHERE camera<>'' LIMIT 1").fetchone()
       or ["indian_road"])[0]
row = con.execute("SELECT id FROM vehicles LIMIT 1").fetchone()
VEHICLE = row[0] if row else None
row = con.execute("SELECT object_id FROM objects o JOIN attributes a"
                  " ON a.object_id=o.id LIMIT 1").fetchone()
OBJECT = row[0] if row else None
print(f"camera={CAM!r} vehicle={VEHICLE} object={OBJECT}")

app = appmod.create_app([CAM], config_path="config.yaml", autostart=False)

with TestClient(app) as client:

    print("\nservice state")
    r = client.get("/health")
    check("GET /health is 200", r.status_code == 200, str(r.status_code))
    health = r.json()
    for key in ("uptime_s", "cameras", "storage", "hub", "incidents",
                "webhooks", "retention"):
        check(f"/health reports {key}", key in health, str(list(health)))
    check("storage block reports write failures",
          "write_failures" in health.get("storage", {}))
    check("webhooks are off with no subscription configured",
          health["webhooks"]["enabled"] is False,
          "an empty subscriptions list means nothing is delivered")

    print("\ncameras")
    r = client.get("/cameras")
    check("GET /cameras is 200", r.status_code == 200)
    check("it lists the configured camera",
          any(c.get("camera") == CAM for c in r.json().get("cameras", [])),
          str(r.json())[:120])
    r = client.get(f"/cameras/{CAM}")
    check(f"GET /cameras/{CAM} is 200", r.status_code == 200)
    detail = r.json()
    for key in ("camera", "state", "healthy", "source"):
        check(f"camera detail reports {key}", key in detail, str(list(detail)))
    check("state is idle when autostart is off", detail.get("state") == "idle",
          str(detail.get("state")))
    check("GET an unknown camera is 404",
          client.get("/cameras/definitely-not-a-camera").status_code == 404)

    print("\nthe dashboard and the generated docs")
    r = client.get("/")
    check("GET / serves HTML", r.status_code == 200
          and "text/html" in r.headers.get("content-type", ""),
          r.headers.get("content-type"))
    check("the page is not a stub", len(r.content) > 4000, f"{len(r.content)}B")
    check("GET /docs is 200", client.get("/docs").status_code == 200)
    spec = client.get("/openapi.json")
    check("GET /openapi.json is 200", spec.status_code == 200)
    paths = spec.json().get("paths", {})
    for route in ("/health", "/cameras", "/incidents", "/deliveries",
                  "/events", "/yield"):
        check(f"{route} is in the OpenAPI spec", route in paths)

    print("\nreading the record")
    for route in ("/incidents", "/deliveries", "/events", "/yield"):
        check(f"GET {route} is 200",
              client.get(route).status_code == 200, route)
    check("/incidents returns a list",
          isinstance(client.get("/incidents").json().get("incidents"), list))
    check("/deliveries carries a summary for the dashboard",
          "summary" in client.get("/deliveries").json())
    events = client.get("/events?limit=5").json().get("events", [])
    check("/events honours limit", len(events) <= 5, str(len(events)))
    check("/events returned something from this run", len(events) > 0,
          "the run wrote no events")
    kinds = {e.get("kind") for e in client.get("/events?limit=200")
             .json().get("events", [])}
    if kinds:
        one = sorted(k for k in kinds if k)[0]
        filtered = client.get(f"/events?kind={one}&limit=50").json()["events"]
        check(f"/events?kind={one} filters", all(e["kind"] == one for e in filtered),
              str({e["kind"] for e in filtered}))

    print("\nyield: the honest-gaps report")
    y = client.get("/yield").json()
    for key in ("cameras", "vehicles", "unanchored_runs",
                "runs_without_location"):
        check(f"/yield reports {key}", key in y, str(list(y)))
    check("it names runs with no lat/lon rather than silently mapping nothing",
          isinstance(y.get("runs_without_location"), int),
          str(y.get("runs_without_location")))

    print("\nvehicles and journeys")
    if VEHICLE is None:
        check("no vehicles in this DB, so 404 is correct",
              client.get("/vehicles/1").status_code == 404,
              "run with plate reading on to exercise the populated path")
    else:
        r = client.get(f"/vehicles/{VEHICLE}")
        check(f"GET /vehicles/{VEHICLE} is 200", r.status_code == 200,
              str(r.status_code))
        check("it returns the plate", "plate" in str(r.json()))
        p = client.get(f"/vehicles/{VEHICLE}/path")
        check("GET .../path is 200", p.status_code == 200, str(p.status_code))
    check("an unknown vehicle is 404",
          client.get("/vehicles/99999999").status_code == 404)
    check("an unknown vehicle path is 404",
          client.get("/vehicles/99999999/path").status_code == 404)

    print("\ncrop imagery")
    check("a missing crop is 404, not a traceback",
          client.get("/crops/99999999.jpg").status_code == 404)

    print("\ncontrol endpoints")
    r = client.post(f"/cameras/{CAM}/stop")
    check("POST .../stop is 200", r.status_code == 200, str(r.status_code))
    check("it reports the resulting state", "state" in r.json(), str(r.json()))
    check("POST .../start on an unknown camera is 404",
          client.post("/cameras/nope/start").status_code == 404)

    print("\nbad input is rejected, not crashed")
    check("a non-numeric vehicle id is a 422",
          client.get("/vehicles/abc").status_code == 422,
          str(client.get("/vehicles/abc").status_code))
    check("a negative limit does not 500",
          client.get("/events?limit=-5").status_code in (200, 422),
          str(client.get("/events?limit=-5").status_code))
    check("an absurd limit does not 500",
          client.get("/events?limit=999999").status_code in (200, 422))
    check("an unknown route is 404", client.get("/no/such/route").status_code == 404)

con.close()

print(f"\n{'=' * 62}")
print(f"{_passed} passed, {len(_failed)} failed")
for name in _failed:
    print(f"  FAILED: {name}")
sys.exit(1 if _failed else 0)
