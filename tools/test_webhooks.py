"""Self-test for incident DELIVERY - the outbox dispatcher, end to end.

    python tools/test_webhooks.py

Runs a real HTTP receiver on 127.0.0.1 and lets the real dispatcher POST to it,
so this exercises httpx, the headers, the signature and the retry ladder rather
than a mock of them. Needs httpx (and, until src/server/__init__ stops
importing the app eagerly, fastapi too); skips cleanly if they are absent.

Policy - which events become incidents - is tested separately and without a
network in tools/test_incidents.py.

The three properties worth protecting:
  persistence before delivery   a crash between raising and sending loses nothing
  at-least-once                 a timeout is retried, so consumers must dedupe
                                on incident_id, which is therefore stable
  one bad consumer is isolated  its deliveries dead-letter without delaying
                                anyone else's
"""

import hashlib
import hmac
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import httpx
except ImportError:
    httpx = None

try:
    from src.server.webhooks import (BACKOFF_MAX_S, WebhookDispatcher,
                                     backoff_for)
    _import_error = None
except Exception as e:                       # fastapi missing, most likely
    _import_error = e

if httpx is None or _import_error is not None:
    why = "httpx not installed" if httpx is None else f"{_import_error}"
    print(f"SKIPPED: {why}\n  pip install -r requirements.txt")
    sys.exit(0)

from src.incidents import IncidentPolicy
from src.storage.sqlite_store import SqliteStore

PORT = 9099
SECRET = "test-secret-not-a-real-one"
SECRET_ENV = "_TRAFFIC_TEST_WEBHOOK_SECRET"
DB = os.path.join("outputs", "_test_webhooks.db")

_passed, _failed = 0, []


def check(label, condition, detail=""):
    global _passed
    if condition:
        _passed += 1
        print(f"  ok   {label}")
    else:
        _failed.append(label)
        print(f"  FAIL {label}{('  <- ' + detail) if detail else ''}")


# --- a receiver that records what it is sent, and can be told to fail ----
RECEIVED = []
REPLY = [200]
DELAY = [0.0]


class Receiver(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n)
        RECEIVED.append({"path": self.path, "raw": raw,
                         "headers": {k.lower(): v for k, v in self.headers.items()}})
        if DELAY[0]:
            time.sleep(DELAY[0])
        payload = b'{"ok":true}' if REPLY[0] < 300 else b"nope"
        self.send_response(REPLY[0])
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def fresh_store():
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(DB + suffix)
        except OSError:
            pass
    os.makedirs("outputs", exist_ok=True)
    store = SqliteStore(DB)
    store.start_run({"source": "test", "camera": "north", "fps": 30,
                     "width": 1920, "height": 1080, "analyse_fps": 30,
                     "config": {}})
    return store


RETIRE = {"object_id": 137, "vehicle_id": 9,
          "lane_id": "carriageway", "lane_flag": "wrong_way",
          "plate": "UP16PT9304", "plate_conf": 0.997,
          "cls_name": "car", "colour": "white",
          "first_seen_s": 1200.0, "last_seen_s": 1234.5, "frames_seen": 64,
          "crop_path": "outputs/north/crops/object_137.jpg"}


def raise_one(policy, store, object_id):
    detail = {**RETIRE, "object_id": object_id}
    policy._fired.clear()
    policy.handle({"kind": "track_retired", "track_id": object_id,
                   "ts": float(object_id), "detail": detail},
                  {"camera": "north"})
    store.flush()
    time.sleep(0.25)


def status_of(store, delivery_id):
    return store._rconn.execute(
        "SELECT status, attempts, last_error FROM deliveries WHERE id=?",
        (delivery_id,)).fetchone()


# --- backoff, before any network ----------------------------------------
print("\nbackoff ladder")
check("first retry waits 2s", backoff_for(1) == 2.0, str(backoff_for(1)))
check("it doubles", [backoff_for(n) for n in (1, 2, 3, 4)] == [2, 4, 8, 16],
      str([backoff_for(n) for n in (1, 2, 3, 4)]))
check("and is capped", backoff_for(50) == BACKOFF_MAX_S, str(backoff_for(50)))
check("cap keeps a recovered consumer inside 5 minutes",
      BACKOFF_MAX_S <= 300.0, str(BACKOFF_MAX_S))

# --- bring up the receiver ----------------------------------------------
os.environ[SECRET_ENV] = SECRET
server = HTTPServer(("127.0.0.1", PORT), Receiver)
threading.Thread(target=server.serve_forever, daemon=True).start()
ENDPOINT = f"http://127.0.0.1:{PORT}/hook"

store = fresh_store()
cfg = {"incidents": {
    "enabled": True, "base_url": "http://localhost:8000",
    "kinds": {"wrong_way": True, "congestion": True},
    "subscriptions": [{"endpoint": ENDPOINT, "secret_env": SECRET_ENV,
                       "timeout_s": 5}]}}
policy = IncidentPolicy(cfg, store, {"north": {"name": "North Gate",
                                               "lat": 28.6139, "lon": 77.2090}})
disp = WebhookDispatcher(store, policy)
# Driven synchronously instead of via start(), so every assertion below is
# deterministic rather than racing a 1s poll loop.
disp._client = httpx.Client(follow_redirects=False)

print("\nthe dispatcher is only live when someone is subscribed")
check("enabled with a subscription", disp.enabled)
check("disabled with none",
      not WebhookDispatcher(store, IncidentPolicy({}, store, {})).enabled,
      "an empty subscriptions list is why nothing sends out of the box")

# --- persistence before delivery ----------------------------------------
print("\npersistence before delivery")
raise_one(policy, store, 137)
rows = store.due_deliveries()
check("the incident row exists before anything is sent",
      store._rconn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] == 1)
check("a delivery row is queued per subscriber", len(rows) == 1, str(len(rows)))
check("it starts pending with 0 attempts",
      rows and rows[0]["attempts"] == 0)

# --- a successful POST --------------------------------------------------
print("\na successful delivery")
REPLY[0] = 200
RECEIVED.clear()
attempted = disp.drain_once()
store.flush()
time.sleep(0.3)
check("one delivery attempted", attempted == 1, str(attempted))
check("the receiver actually got a POST", len(RECEIVED) == 1, str(len(RECEIVED)))
check("dispatcher counted it sent", disp.stats()["sent"] == 1,
      str(disp.stats()))
check("marked sent in the database", status_of(store, rows[0]["id"])[0] == "sent",
      str(status_of(store, rows[0]["id"])))
check("nothing is due any more", not store.due_deliveries())

got = RECEIVED[0]
print("\nwhat the consumer receives")
check("content-type is json", got["headers"].get("content-type") == "application/json")
check("x-incident-id header matches the payload",
      got["headers"].get("x-incident-id") == "north-137-wrong_way",
      got["headers"].get("x-incident-id"))
check("x-incident-kind header is set",
      got["headers"].get("x-incident-kind") == "wrong_way")
check("x-timestamp header is set", bool(got["headers"].get("x-timestamp")))
check("user-agent identifies this system",
      "traffic-detection" in got["headers"].get("user-agent", ""))
body = json.loads(got["raw"])
check("body is the policy's payload", body["incident_id"] == "north-137-wrong_way")
check("body carries the plate", body["vehicle"]["plate"] == "UP16PT9304")

# --- signature, verified independently of the sending code --------------
print("\nsignature, recomputed here rather than trusted")
stamp = got["headers"].get("x-timestamp", "")
mine = hmac.new(SECRET.encode(), stamp.encode() + b"." + got["raw"],
                hashlib.sha256).hexdigest()
check("x-signature verifies against the raw bytes",
      hmac.compare_digest("sha256=" + mine, got["headers"].get("x-signature", "")),
      got["headers"].get("x-signature"))
tampered = hmac.new(SECRET.encode(), stamp.encode() + b"." + got["raw"] + b"x",
                    hashlib.sha256).hexdigest()
check("a tampered body does NOT verify",
      not hmac.compare_digest("sha256=" + tampered,
                              got["headers"].get("x-signature", "")))
replayed = hmac.new(SECRET.encode(), b"9999999999." + got["raw"],
                    hashlib.sha256).hexdigest()
check("a replayed timestamp does NOT verify",
      not hmac.compare_digest("sha256=" + replayed,
                              got["headers"].get("x-signature", "")),
      "the stamp is inside the signed material")
check("signed bytes are byte-identical to sent bytes",
      got["raw"] == json.dumps(body, sort_keys=True,
                               separators=(",", ":")).encode(),
      "a consumer re-serialising must get the same digest")

# --- a server error retries ---------------------------------------------
print("\na 5xx is retried, not discarded")
REPLY[0] = 500
raise_one(policy, store, 500)
row = [r for r in store.due_deliveries() if "500" in r["incident_id"]][0]
disp._attempt(row)
store.flush()
time.sleep(0.3)
st, att, err = status_of(store, row["id"])
check("still pending after a 500", st == "pending", st)
check("attempt was counted", att == 1, str(att))
check("the error is recorded for the dashboard", "500" in (err or ""), str(err))
nxt = store._rconn.execute(
    "SELECT next_attempt_at FROM deliveries WHERE id=?", (row["id"],)).fetchone()[0]
check("the retry is scheduled in the future", nxt > time.time(), str(nxt))
check("so it is not due again immediately",
      row["id"] not in [r["id"] for r in store.due_deliveries()])

# --- a client error dead-letters immediately ----------------------------
print("\na 4xx is permanent - retrying cannot fix a malformed request")
REPLY[0] = 400
raise_one(policy, store, 400)
row = [r for r in store.due_deliveries() if "400" in r["incident_id"]][0]
disp._attempt(row)
store.flush()
time.sleep(0.3)
st, att, err = status_of(store, row["id"])
check("dead after a single attempt", st == "dead", st)
check("not eight attempts", att == 1, str(att))
check("dead is a state, not a deletion - the payload survives",
      store._rconn.execute("SELECT COUNT(*) FROM incidents WHERE id LIKE '%-400-%'"
                           ).fetchone()[0] == 1)

print("\n429 and 408 are the 4xx exceptions (rate limit / timeout)")
for code in (408, 429):
    REPLY[0] = code
    raise_one(policy, store, code)
    row = [r for r in store.due_deliveries() if str(code) in r["incident_id"]][0]
    disp._attempt(row)
    store.flush()
    time.sleep(0.3)
    check(f"HTTP {code} is retried, not dead-lettered",
          status_of(store, row["id"])[0] == "pending",
          status_of(store, row["id"])[0])

# --- exhausting the ladder ----------------------------------------------
print("\ngiving up after max_attempts")
REPLY[0] = 500
short = WebhookDispatcher(store, policy, max_attempts=3)
short._client = disp._client
raise_one(policy, store, 777)
row = [r for r in store.due_deliveries() if "777" in r["incident_id"]][0]
for n in range(3):
    row["attempts"] = n
    short._attempt(row)
store.flush()
time.sleep(0.3)
check("dead once max_attempts is reached",
      status_of(store, row["id"])[0] == "dead", str(status_of(store, row["id"])))
check("counted as dead", short.stats()["dead"] >= 1, str(short.stats()))

# --- an unreachable endpoint --------------------------------------------
print("\nan endpoint that is not listening")
bad_cfg = {"incidents": {"kinds": {"wrong_way": True},
                         "subscriptions": [{"endpoint":
                                            "http://127.0.0.1:9098/nope"}]}}
bad_policy = IncidentPolicy(bad_cfg, store, {})
bad_disp = WebhookDispatcher(store, bad_policy)
bad_disp._client = httpx.Client(follow_redirects=False, timeout=1.0)
raise_one(bad_policy, store, 999)
row = [r for r in store.due_deliveries() if "999" in r["incident_id"]][0]
bad_disp._attempt(row)
store.flush()
time.sleep(0.3)
st, att, err = status_of(store, row["id"])
check("a connection refusal is retried, not crashed", st == "pending", st)
check("the exception type is recorded", bool(err), str(err))
check("the dispatcher survived it", bad_disp.stats()["failed_attempts"] == 1)

# --- a removed subscription ---------------------------------------------
print("\na subscription deleted from config while a delivery was pending")
raise_one(policy, store, 888)
row = [r for r in store.due_deliveries() if "888" in r["incident_id"]][0]
orphan = WebhookDispatcher(store, IncidentPolicy(
    {"incidents": {"kinds": {"wrong_way": True},
                   "subscriptions": [{"endpoint": "http://elsewhere/hook"}]}},
    store, {}))
orphan._client = disp._client
orphan._attempt(row)
store.flush()
time.sleep(0.3)
check("dead-lettered rather than retried forever",
      status_of(store, row["id"])[0] == "dead", str(status_of(store, row["id"])))
check("and says why", "no such subscription" in (status_of(store, row["id"])[2] or ""),
      str(status_of(store, row["id"])[2]))

# --- at-least-once means the consumer must dedupe ------------------------
print("\nthe duplicate contract")
store2 = None
before_inc = store._rconn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
before_del = store._rconn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
raise_one(policy, store, 137)          # the very first id, again
raise_one(policy, store, 137)
check("re-raising the same incident adds no incident row",
      store._rconn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
      == before_inc, "natural key must dedupe")
check("...and no second delivery row",
      store._rconn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
      == before_del,
      "UNIQUE(incident_id, endpoint) is what makes fire-once true in practice")

# --- a slow consumer cannot stall the loop ------------------------------
print("\na slow consumer costs a delay and nothing else")
REPLY[0] = 200
DELAY[0] = 1.2
raise_one(policy, store, 555)
row = [r for r in store.due_deliveries() if "555" in r["incident_id"]][0]
t0 = time.time()
disp._attempt(row)
elapsed = time.time() - t0
DELAY[0] = 0.0
store.flush()
time.sleep(0.3)
check("the slow POST still succeeded", status_of(store, row["id"])[0] == "sent",
      str(status_of(store, row["id"])))
check("it blocked only the dispatcher thread, and only briefly",
      1.0 <= elapsed < 5.0, f"{elapsed:.2f}s")

# --- the record ----------------------------------------------------------
print("\nwhat the dashboard can show")
summary = dict(store._rconn.execute(
    "SELECT status, COUNT(*) FROM deliveries GROUP BY 1").fetchall())
check("every terminal state is represented",
      {"sent", "pending", "dead"} <= set(summary), str(summary))
check("stats() reports sent/failed/dead",
      set(disp.stats()) >= {"sent", "failed_attempts", "dead", "last_error"},
      str(disp.stats()))

disp.stop()
bad_disp.stop()
store.close()
server.shutdown()
for suffix in ("", "-wal", "-shm"):
    try:
        os.remove(DB + suffix)
    except OSError:
        pass

print(f"\n{'=' * 62}")
print(f"{_passed} passed, {len(_failed)} failed")
for name in _failed:
    print(f"  FAILED: {name}")
sys.exit(1 if _failed else 0)
