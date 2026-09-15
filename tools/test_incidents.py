"""Self-test for incident POLICY - what fires, when, and what the payload says.

    python tools/test_incidents.py

Policy is tested separately from delivery (tools/test_webhooks.py) because
they fail in different ways and for different reasons. This half needs no
network, no database and no pipeline: it drives IncidentPolicy with dicts.

Why it needs a test at all: every failure here is silent. A kind wired to fire
on the wrong event floods a customer's endpoint; a dwell that does not hold
emits hundreds of webhooks a minute; evidence that never reaches the payload
produces an alert a consumer cannot act on. None of those raise an exception.
"""

import hashlib
import hmac
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.incidents import (CONGESTION_ORDER, NEVER, IncidentPolicy,
                           incident_id, serialise, sign)

_passed, _failed = 0, []


def check(label, condition, detail=""):
    global _passed
    if condition:
        _passed += 1
        print(f"  ok   {label}")
    else:
        _failed.append(label)
        print(f"  FAIL {label}{('  <- ' + detail) if detail else ''}")


def policy(**overrides):
    """A policy with everything on, wired to no store."""
    inc = {"enabled": True, "base_url": "http://localhost:8000",
           "congestion_dwell_s": 5, "max_dwell_s": 120,
           "kinds": {"wrong_way": True, "congestion": True,
                     "wrong_lane": False, "crossing": False}}
    inc.update(overrides)
    return IncidentPolicy({"incidents": inc}, None,
                          {"north": {"name": "North Gate",
                                     "lat": 28.6139, "lon": 77.2090}})


RETIRE = {"object_id": 137, "identity_id": 9, "identity_kind": "plate",
          "lane_id": "carriageway", "lane_flag": "wrong_way",
          "plate": "UP16PT9304", "plate_conf": 0.997,
          "cls_name": "car", "colour": "white",
          "first_seen_s": 1200.0, "last_seen_s": 1234.5, "frames_seen": 64,
          "crop_path": "outputs/north/crops/object_137.jpg"}


def retire(p, tid=42, detail=None, ts=1234.5):
    return p.handle({"kind": "track_retired", "track_id": tid, "ts": ts,
                     "detail": dict(detail or RETIRE)}, {"camera": "north"})


# --- which events are incidents at all ----------------------------------
print("\nwhat counts as an incident")
p = policy()
check("wrong_way fires", p.is_incident("wrong_way"))
check("congestion fires", p.is_incident("congestion"))
check("wrong_lane is off by default in this config", not p.is_incident("wrong_lane"))
for kind, why in NEVER.items():
    check(f"{kind} can never be an incident ({why[:34]}...)",
          not p.is_incident(kind))
check("crossing stays off even if config enables it",
      not policy(kinds={"crossing": True}).is_incident("crossing"),
      "NEVER must win over config")

# --- the normal path: flagged, then retires ------------------------------
print("\nthe normal path")
p = policy()
raised = retire(p)
check("a retiring wrong_way track raises exactly one incident", len(raised) == 1,
      str(len(raised)))
body = raised[0]["payload"]
check("payload carries the plate", body["vehicle"]["plate"] == "UP16PT9304")
check("payload carries the identity id", body["vehicle"]["identity_id"] == 9)
check("and names the identity axis",
      body["vehicle"]["identity_kind"] == "plate",
      str(body["vehicle"].get("identity_kind")))
check("payload carries camera lat/lon", body["camera"]["lat"] == 28.6139)
check("state is 'closed' for a retired track", body["state"] == "closed")
check("image_url is built from base_url, not the local crop path",
      body["image_url"] == "http://localhost:8000/crops/137.jpg", body["image_url"])
check("crop_path is NOT leaked to the consumer",
      "crop_path" not in json.dumps(body))

# --- absence is explicit ------------------------------------------------
print("\nabsence is explicit, not omitted")
p = policy()
bare = dict(RETIRE)
for k in ("plate", "plate_conf", "identity_id", "identity_kind", "crop_path"):
    bare.pop(k, None)
body = retire(p, detail=bare)[0]["payload"]
check("plate key present even with no plate read", "plate" in body["vehicle"])
check("...and its value is null", body["vehicle"]["plate"] is None)
check("identity_id key present and null",
      body["vehicle"]["identity_id"] is None)
check("image_url null when there is no crop", body["image_url"] is None)

# --- evidence arrives before identity does -------------------------------
print("\nevidence hand-off (the analyzer flags early, identity settles late)")
p = policy()
for ts, align, observed in ((1210.0, -0.61, "up (+12deg)"),
                            (1215.0, -0.79, "up-right (+31deg)"),
                            (1220.0, -0.86, "up-right (+40deg)")):
    fired = p.handle({"kind": "wrong_way", "track_id": 42, "ts": ts,
                      "detail": {"lane_id": "carriageway",
                                 "lane_flag": "wrong_way",
                                 "alignment": align,
                                 "observed_heading": observed,
                                 "expected_heading": "down (+166deg)",
                                 "travel_px": 38.2}}, {"camera": "north"})
    check(f"flagging at t={ts} fires nothing yet", not fired,
          "must wait for retirement")
detail = retire(p)[0]["payload"]["detail"]
check("evidence reached the payload", "alignment" in detail, str(detail))
check("the LATEST (sharpest) verdict won", detail["alignment"] == -0.86,
      str(detail.get("alignment")))
check("expected vs observed heading both present",
      detail.get("expected_heading") and detail.get("observed_heading"))

# --- a track that never retires ------------------------------------------
print("\nstuck track force-fires (the parked-obstruction case)")
p = policy(max_dwell_s=10)
flag = {"kind": "wrong_way", "track_id": 7,
        "detail": {"lane_id": "c", "lane_flag": "wrong_way"}}
check("nothing at t=0", not p.handle({**flag, "ts": 0.0}, {"camera": "north"}))
check("nothing before the dwell elapses",
      not p.handle({**flag, "ts": 9.0}, {"camera": "north"}))
out = p.handle({**flag, "ts": 11.0}, {"camera": "north"})
check("fires once the dwell is exceeded", len(out) == 1, str(len(out)))
check("state is 'ongoing', not 'closed'",
      out and out[0]["payload"]["state"] == "ongoing")
check("carries dwell_s so a consumer knows how long",
      out and "dwell_s" in out[0]["payload"])
check("does NOT fire a second time while still stuck",
      not p.handle({**flag, "ts": 30.0}, {"camera": "north"}))
check("and does not fire again when it finally retires",
      not retire(p, tid=7) and p.stats()["suppressed_duplicates"] == 1,
      f"suppressed={p.stats()['suppressed_duplicates']}")

# --- congestion: both edges, each after a dwell --------------------------
print("\ncongestion dwell and both edges")
p = policy(congestion_dwell_s=5)


def cong(p, level, ts):
    return p.handle({"kind": "congestion", "ts": ts,
                     "detail": {"level": level, "occupancy": 0.31}},
                    {"camera": "north"})


check("heavy at t=0 does not fire immediately", not cong(p, "heavy", 0.0))
check("still nothing at t=2 (dwell not met)", not cong(p, "heavy", 2.0))
check("fires once held past the dwell", len(cong(p, "heavy", 6.0)) == 1)
check("state is now congested", p.congestion_state("north") == "congested")
check("a repeat heavy does not re-fire", not cong(p, "heavy", 20.0))
check("free at t=21 does not fire immediately", not cong(p, "free", 21.0))
check("the clearing edge fires too", len(cong(p, "free", 27.0)) == 1,
      "onset-only cannot give a consumer a duration")
check("state is back to clear", p.congestion_state("north") == "clear")
p2 = policy(congestion_dwell_s=5)
check("'busy' does not count as congested at the default threshold",
      not cong(p2, "busy", 0.0) and not cong(p2, "busy", 99.0),
      "busy is ordinary rush hour")
p3 = policy(congestion_dwell_s=0)
cong(p3, "jammed", 1.0)          # first call only arms the candidate
jam = cong(p3, "jammed", 2.0)
check("a congestion payload has NO vehicle block",
      jam and "vehicle" not in jam[0]["payload"],
      "congestion is a property of the road, not of a car")
check("...and no sighting block either",
      jam and "sighting" not in jam[0]["payload"])
check("congestion evidence (level, occupancy) is passed through",
      jam and jam[0]["payload"]["detail"].get("level") == "jammed")

# --- flapping ------------------------------------------------------------
print("\nflapping cannot emit a storm")
p = policy(congestion_dwell_s=5)
fires = 0
for i in range(40):
    fires += len(cong(p, "heavy" if i % 2 else "free", float(i)))
check("40 alternating frames emit nothing", fires == 0, f"{fires} fired")

# --- ids are stable ------------------------------------------------------
print("\nidempotency key")
check("id is derived from the sighting, not a clock",
      incident_id("north", "wrong_way", 137) ==
      incident_id("north", "wrong_way", 137))
check("different object -> different id",
      incident_id("north", "wrong_way", 137) !=
      incident_id("north", "wrong_way", 138))
check("different camera -> different id",
      incident_id("north", "wrong_way", 137) !=
      incident_id("south", "wrong_way", 137))
check("congestion falls back to its transition time",
      incident_id("north", "congestion", None, 2008.0) == "north-t2008-congestion")
p = policy()
first = retire(p)[0]["id"]
second = retire(p)[0]["id"]
check("the same track retiring twice yields the SAME id", first == second,
      f"{first} vs {second}")

# --- signature -----------------------------------------------------------
print("\nsignature")
body = serialise({"b": 2, "a": 1})
check("serialise sorts keys so the digest is reproducible",
      body == b'{"a":1,"b":2}', body.decode())
stamp = "1700000000"
sig = sign(body, "s3cret", stamp)
expect = hmac.new(b"s3cret", stamp.encode() + b"." + body, hashlib.sha256)
check("signature is HMAC-SHA256 over timestamp + body",
      sig == f"sha256={expect.hexdigest()}")
check("a different timestamp changes the signature",
      sign(body, "s3cret", "1700000001") != sig,
      "the stamp must be INSIDE the signed material or a POST replays forever")
check("a different body changes the signature",
      sign(body + b" ", "s3cret", stamp) != sig)
check("a different secret changes the signature",
      sign(body, "other", stamp) != sig)

# --- routing -------------------------------------------------------------
print("\nsubscription filters")
p = IncidentPolicy({"incidents": {
    "kinds": {"wrong_way": True, "congestion": True},
    "subscriptions": [
        {"endpoint": "http://a/hook"},
        {"endpoint": "http://b/hook", "kinds": ["congestion"]},
        {"endpoint": "http://c/hook", "cameras": ["south"]},
        {"endpoint": ""},
    ]}}, None, {})
check("an endpoint-less subscription is dropped", len(p.subscriptions) == 3,
      str(len(p.subscriptions)))
check("no filters means everything",
      "http://a/hook" in p.endpoints_for("wrong_way", "north"))
check("a kind filter excludes other kinds",
      "http://b/hook" not in p.endpoints_for("wrong_way", "north"))
check("...and includes its own kind",
      "http://b/hook" in p.endpoints_for("congestion", "north"))
check("a camera filter excludes other cameras",
      "http://c/hook" not in p.endpoints_for("wrong_way", "north"))
check("secret comes from the environment, never the config file",
      p.subscriptions[0].secret == "",
      "no secret_env set, so no secret")
os.environ["_TRAFFIC_TEST_SECRET"] = "from-env"
p2 = IncidentPolicy({"incidents": {"subscriptions": [
    {"endpoint": "http://a/hook", "secret_env": "_TRAFFIC_TEST_SECRET"}]}},
    None, {})
check("...and is read when secret_env names one",
      p2.subscriptions[0].secret == "from-env")
check("describe() says UNSIGNED when there is no secret",
      "UNSIGNED" in p.subscriptions[0].describe())

# --- disabled ------------------------------------------------------------
print("\nkill switch and memory")
off = IncidentPolicy({"incidents": {"enabled": False,
                                    "kinds": {"wrong_way": True}}}, None, {})
check("enabled:false raises nothing at all", not retire(off))
p = policy()
for tid in range(200):
    p.handle({"kind": "wrong_way", "track_id": tid, "ts": 0.0,
              "detail": {"lane_flag": "wrong_way"}}, {"camera": "north"})
check("200 flagged tracks are being tracked", p.stats()["tracked_flags"] == 200,
      str(p.stats()["tracked_flags"]))
for tid in range(200):
    retire(p, tid=tid, detail={**RETIRE, "object_id": tid})
check("retirement frees every one of them (no leak on a 24/7 feed)",
      p.stats()["tracked_flags"] == 0, str(p.stats()["tracked_flags"]))

# --- bad config ----------------------------------------------------------
print("\nbad config degrades rather than crashes")
p = policy(congested_at="nonsense")
check("an unknown congested_at falls back to the default",
      p.congested_at_name == "heavy", p.congested_at_name)
check("...and the level order is still usable",
      CONGESTION_ORDER["jammed"] > CONGESTION_ORDER["free"])

print(f"\n{'=' * 62}")
print(f"{_passed} passed, {len(_failed)} failed")
for name in _failed:
    print(f"  FAILED: {name}")
sys.exit(1 if _failed else 0)
