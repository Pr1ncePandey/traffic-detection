"""Incident policy: what is worth telling someone about, and when to say it.

This module decides. `src/server/webhooks.py` delivers. Keeping those apart is
what lets the firing rules be tested without a network.

WHAT COUNTS

    wrong_way            yes            the motivating case
    congestion           on state change  own rule, see below
    wrong_lane           configurable   noisier; depends on lane confidence
    crossing             NO             fires for every vehicle. It is a
                                        counter, not an incident
    vehicle_identified   no             internal bookkeeping; fires on every
                                        mid-track rebind

The enable flags live in config (`incidents.kinds`), so this is policy an
operator can change rather than code.

WHEN IT FIRES

Per-vehicle incidents fire once, at TRACK RETIREMENT. That is the only moment
the payload is complete: the plate vote has settled, the sharpest crop has been
chosen, and the sighting's real bounds are known. The cost is latency - roughly
time-in-frame + 5 s, because `ttl_for()` floors at 5 s - and that is accepted
deliberately, because the dashboard already serves anyone who needs to know
sooner over the WebSocket. Two consumers, two different requirements.

THREE RULES THAT EXIST BECAUSE THE SIMPLE VERSION IS WRONG

1. Stuck tracks. A parked car, or a track the tracker keeps alive indefinitely,
   never reaches eviction - so "fire at retirement" would never fire for
   exactly the stationary-obstruction case most worth alerting on. After
   `max_dwell_s` of continuous flagging it force-fires with whatever data
   exists and `state: "ongoing"`, then suppresses that track PERMANENTLY. No
   second webhook when it eventually retires: "fire once" has to stay true or
   consumers cannot trust the id.

2. Congestion has no track. `congestion.py` emits a frame-scoped event with no
   track id, so retirement cannot trigger it. It fires on `clear -> congested`
   and back, but only once the new state has held for `congestion_dwell_s`. The
   dwell is the whole point: a metric oscillating around its threshold would
   otherwise emit hundreds of webhooks a minute. BOTH edges are emitted so a
   consumer can show a current state and compute a jam's duration from the
   pair - onset-only can do neither.

3. Absence is explicit. `plate` and `vehicle_id` may be null, because a vehicle
   whose plate never read confidently has no `vehicles` row at all. The payload
   carries the keys with null values rather than omitting them, so a consumer
   can tell "no plate" from "this version does not send plates".
"""

import hashlib
import hmac
import json
import os
import time

# Events that are never incidents, whatever config says. Named here rather
# than merely left out of the defaults so the reason survives.
NEVER = {
    "crossing": "fires for every vehicle - a counter, not an incident",
    "vehicle_identified": "internal bookkeeping; fires on every mid-track rebind",
    "track_retired": "the trigger for other incidents, not one itself",
    "color_read": "an attribute enricher reporting a fact",
    "clothes_read": "an attribute enricher reporting a fact",
}

DEFAULT_KINDS = {"wrong_way": True, "congestion": True,
                 "wrong_lane": False, "crossing": False}

# Kept out of the payload's `detail` block: either projected into a dedicated
# block already, or a local path a remote consumer cannot use.
_DETAIL_STRIP = frozenset({
    "object_id", "vehicle_id", "plate", "plate_conf", "cls_name", "colour",
    "first_seen_s", "last_seen_s", "frames_seen", "crop_path",
})

CONGESTED = "congested"
CLEAR = "clear"

# congestion.py reports a four-level scale, not a boolean:
#   free -> busy -> heavy -> jammed
# The webhook contract is two-state (a consumer wants "is it jammed or not"),
# so one threshold collapses the scale. `incidents.congested_at` names the
# lowest level that counts as congested; "heavy" by default because "busy" is
# ordinary rush-hour traffic on most roads and alerting on it would cry wolf.
CONGESTION_ORDER = {"free": 0, "busy": 1, "heavy": 2, "jammed": 3}
DEFAULT_CONGESTED_AT = "heavy"


def incident_id(camera: str, kind: str, object_id=None, stamp=None) -> str:
    """The idempotency key a consumer deduplicates on.

    Derived from the sighting, not from a counter or a clock, so the same
    physical incident produces the same id on a retry or a restart. Congestion
    has no object, so it falls back to its transition time - which is stable
    for one transition and different for the next.
    """
    tail = object_id if object_id is not None else f"t{int(stamp or time.time())}"
    return f"{camera}-{tail}-{kind}"


class Subscription:
    """One webhook consumer: an endpoint, a secret, and optional filters.

    Absent or empty filters mean "everything", so the simple single-endpoint
    case stays a three-line config block.
    """

    def __init__(self, spec: dict):
        self.endpoint = str(spec.get("endpoint") or "").strip()
        self.kinds = {str(k) for k in (spec.get("kinds") or [])}
        self.cameras = {str(c) for c in (spec.get("cameras") or [])}
        self.secret_env = spec.get("secret_env") or ""
        # Read from the environment, never from the config file, so a secret is
        # not committed to a repo alongside the endpoint it protects.
        self.secret = os.environ.get(self.secret_env, "") if self.secret_env else ""
        self.timeout_s = float(spec.get("timeout_s", 10.0))

    @property
    def valid(self) -> bool:
        return bool(self.endpoint)

    def matches(self, kind: str, camera: str) -> bool:
        if self.kinds and kind not in self.kinds:
            return False
        if self.cameras and camera not in self.cameras:
            return False
        return True

    def describe(self) -> str:
        kinds = ", ".join(sorted(self.kinds)) if self.kinds else "all kinds"
        cams = ", ".join(sorted(self.cameras)) if self.cameras else "all cameras"
        secret = "signed" if self.secret else "UNSIGNED"
        return f"{self.endpoint} [{kinds} | {cams} | {secret}]"


def sign(body: bytes, secret: str, timestamp: str) -> str:
    """HMAC-SHA256 over timestamp + body.

    The timestamp is INSIDE the signed material, not merely alongside it:
    signing the body alone lets a captured POST be replayed verbatim at any
    later time and still verify. A consumer should reject a timestamp outside
    its tolerance window as well as a bad signature.
    """
    mac = hmac.new(secret.encode("utf-8"),
                   timestamp.encode("utf-8") + b"." + body,
                   hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


class IncidentPolicy:
    """Turns pipeline events into incidents, and routes them to subscribers.

    One instance per server, shared by every camera worker. The per-track state
    it keeps (`_fired`, `_flagged_since`) is bounded by forgetting a track as
    soon as it retires - the same discipline the rest of the codebase applies,
    because a 24/7 feed makes any unbounded dict a leak.
    """

    def __init__(self, cfg: dict, store, cameras: dict | None = None):
        inc = (cfg or {}).get("incidents", {}) or {}
        self.enabled = bool(inc.get("enabled", True))
        self.kinds = {**DEFAULT_KINDS, **(inc.get("kinds") or {})}
        self.max_dwell_s = float(inc.get("max_dwell_s", 120))
        self.congestion_dwell_s = float(inc.get("congestion_dwell_s", 30))
        at = str(inc.get("congested_at", DEFAULT_CONGESTED_AT)).lower()
        if at not in CONGESTION_ORDER:
            print(f"[incidents] congested_at={at!r} is not one of "
                  f"{sorted(CONGESTION_ORDER)}; using {DEFAULT_CONGESTED_AT}")
            at = DEFAULT_CONGESTED_AT
        self.congested_at = CONGESTION_ORDER[at]
        self.congested_at_name = at
        self.base_url = str(inc.get("base_url", "") or "").rstrip("/")
        self.subscriptions = [s for s in
                              (Subscription(x) for x in (inc.get("subscriptions") or []))
                              if s.valid]
        self.store = store
        # camera id -> {"name","lat","lon"}; used to stamp location on payloads.
        self.cameras = dict(cameras or {})
        self.raised = 0
        self.suppressed = 0

        # Per-track: which (track, kind) pairs have already fired, when a
        # track's flag was first seen (for the force-fire dwell), and the
        # EVIDENCE the analyzer gave when it flagged.
        #
        # That last one exists because the evidence and the identity arrive at
        # different times. lanes.py knows the lane, the expected heading and
        # the observed heading at the moment it flags; the plate vote and the
        # crop are not settled until the track retires. A payload wants both,
        # so the evidence is held here until the identity catches up.
        self._fired: set = set()
        self._flagged_since: dict = {}
        self._evidence: dict = {}
        # Per-camera congestion state machine.
        self._congestion: dict = {}

    # --- policy questions --------------------------------------------------
    def is_incident(self, kind: str) -> bool:
        if kind in NEVER:
            return False
        return bool(self.kinds.get(kind, False))

    def endpoints_for(self, kind: str, camera: str) -> list:
        return [s.endpoint for s in self.subscriptions if s.matches(kind, camera)]

    def subscription_for(self, endpoint: str):
        for s in self.subscriptions:
            if s.endpoint == endpoint:
                return s
        return None

    # --- the event hook ----------------------------------------------------
    def handle(self, event: dict, ctx: dict) -> list:
        """Called for every pipeline event. Returns the incidents raised.

        Deliberately returns a list rather than firing straight into storage,
        so the caller (and a test) can see what a given event produced.
        """
        if not self.enabled:
            return []
        kind = event.get("kind")
        camera = ctx.get("camera") or ""
        if kind == "congestion":
            return self._on_congestion(event, ctx)
        if kind == "track_retired":
            return self._on_retire(event, ctx)
        if self.is_incident(kind):
            return self._on_flag(event, ctx, kind, camera)
        return []

    def _on_flag(self, event, ctx, kind, camera) -> list:
        """A per-vehicle flag was raised this frame.

        Nothing fires here in the normal case: the payload is not complete
        until the track retires. All this does is start the dwell clock, so a
        track that never retires can still be force-fired.
        """
        tid = event.get("track_id")
        if tid is None:
            return []
        key = (camera, tid, kind)
        if key in self._fired:
            return []
        now = float(event.get("ts") or ctx.get("timestamp") or 0.0)
        first = self._flagged_since.setdefault(key, now)
        # Latest evidence wins: a verdict refined over several frames (the
        # observed heading sharpens as the track accumulates travel) should
        # reach the consumer in its best form.
        self._evidence[key] = dict(event.get("detail") or {})
        if (now - first) < self.max_dwell_s:
            return []
        # Stuck track: fire with whatever exists, then never again for this
        # track. A payload with a null plate beats silence for a stationary
        # obstruction, which is the case this branch exists for.
        self._fired.add(key)
        payload = self._payload(kind, camera, ctx.get("object_id"), None,
                                evidence=self._evidence.get(key) or {},
                                sighting={}, detected_at=now, state="ongoing",
                                dwell_s=now - first)
        return [self._raise(kind, camera, ctx.get("object_id"), None, payload,
                            created_at=now)]

    def _on_retire(self, event, ctx) -> list:
        """A track retired: fire anything it was flagged for, then forget it.

        This is the normal firing path, and the forgetting is not incidental -
        it is what bounds this object's memory on a feed that never ends.
        """
        detail = event.get("detail") or {}
        camera = ctx.get("camera") or ""
        tid = event.get("track_id")
        flag = str(detail.get("lane_flag") or "")
        out = []
        for kind in ("wrong_way", "wrong_lane"):
            if kind not in flag or not self.is_incident(kind):
                continue
            key = (camera, tid, kind)
            if key in self._fired:
                # Already force-fired while stuck. Suppression is permanent so
                # the "fire once" contract survives.
                self.suppressed += 1
                continue
            self._fired.add(key)
            payload = self._payload(
                kind, camera, detail.get("object_id"), detail.get("vehicle_id"),
                evidence=self._evidence.get(key) or {"lane_id": detail.get("lane_id"),
                                                     "lane_flag": flag},
                sighting=detail,
                detected_at=float(detail.get("last_seen_s") or 0.0),
                state="closed")
            out.append(self._raise(kind, camera, detail.get("object_id"),
                                   detail.get("vehicle_id"), payload,
                                   created_at=time.time()))
        self._forget(camera, tid)
        return out

    def _forget(self, camera, tid):
        """Drop this track's state. Bounded memory on a feed that never ends.

        Called on every retirement, including one that fired nothing, because
        a track that was flagged and then unflagged still left an entry in
        _flagged_since and _evidence.
        """
        for kind in ("wrong_way", "wrong_lane"):
            key = (camera, tid, kind)
            self._flagged_since.pop(key, None)
            self._evidence.pop(key, None)
            self._fired.discard(key)

    def _on_congestion(self, event, ctx) -> list:
        """Congestion: emit both edges, each after a minimum dwell.

        The dwell is what makes this usable. A metric sitting on its threshold
        flips state every few frames, and firing on each flip would emit
        hundreds of webhooks a minute - so a transition is only published once
        the new state has HELD.
        """
        if not self.is_incident("congestion"):
            return []
        camera = ctx.get("camera") or ""
        detail = event.get("detail") or {}
        level = str(detail.get("level") or "").lower()
        observed = (CONGESTED
                    if CONGESTION_ORDER.get(level, 0) >= self.congested_at
                    else CLEAR)
        now = float(event.get("ts") or ctx.get("timestamp") or 0.0)

        state = self._congestion.setdefault(
            camera, {"published": CLEAR, "candidate": CLEAR, "since": now})
        if observed != state["candidate"]:
            # The state changed: restart the dwell clock rather than firing.
            state["candidate"] = observed
            state["since"] = now
            return []
        if observed == state["published"]:
            return []
        if (now - state["since"]) < self.congestion_dwell_s:
            return []
        state["published"] = observed
        payload = self._payload("congestion", camera, None, None,
                                evidence=detail, sighting={}, detected_at=now,
                                state=observed, dwell_s=now - state["since"])
        return [self._raise("congestion", camera, None, None, payload,
                            created_at=now,
                            id_stamp=state["since"])]

    def congestion_state(self, camera: str) -> str:
        return (self._congestion.get(camera) or {}).get("published", CLEAR)

    # --- payload -----------------------------------------------------------
    def _payload(self, kind, camera, object_id, vehicle_id, evidence, sighting,
                 detected_at, state="closed", dwell_s=None) -> dict:
        """The JSON body. Keys are present even when their value is null.

        `evidence` is the ANALYZER's own detail - the lane, the expected and
        observed headings, the occupancy - and goes out as `detail`. `sighting`
        is the settled track record, and is projected into the `vehicle` and
        `sighting` blocks. They are separate arguments because they come from
        different moments: see the note on `self._evidence`.

        A congestion payload carries no `vehicle` block at all. Congestion is a
        property of the road rather than of a car, and an all-null vehicle
        block would invite a consumer to go looking for one.
        """
        cam = self.cameras.get(camera) or {}
        body = {
            "incident_id": incident_id(camera, kind, object_id, detected_at),
            "kind": kind,
            "state": state,
            "camera": {"id": camera, "name": cam.get("name") or camera,
                       "lat": cam.get("lat"), "lon": cam.get("lon")},
            "detected_at": round(float(detected_at or 0.0), 3),
            # Internal bookkeeping is stripped: crop_path is a local path the
            # consumer cannot read (it gets image_url instead), and the vehicle
            # fields are already projected into their own blocks below.
            "detail": {k: v for k, v in (evidence or {}).items()
                       if k not in _DETAIL_STRIP},
        }
        if dwell_s is not None:
            body["dwell_s"] = round(float(dwell_s), 2)
        if kind != "congestion":
            sighting = sighting or {}
            body["vehicle"] = {
                "vehicle_id": vehicle_id,
                "plate": sighting.get("plate"),
                "plate_conf": sighting.get("plate_conf"),
                "cls": sighting.get("cls_name") or None,
                "colour": sighting.get("colour"),
            }
            body["sighting"] = {
                "object_id": object_id,
                "first_seen": sighting.get("first_seen_s"),
                "last_seen": sighting.get("last_seen_s"),
                "frames_seen": sighting.get("frames_seen"),
            }
            # A local path under outputs/ is useless to a remote consumer, so
            # the payload carries a URL served by /crops/{id}.jpg instead. With
            # no base_url configured the key is null rather than a broken
            # relative path that looks like it might work.
            body["image_url"] = (
                f"{self.base_url}/crops/{object_id}.jpg"
                if self.base_url and object_id is not None
                and sighting.get("crop_path")
                else None)
        return body

    def _raise(self, kind, camera, object_id, vehicle_id, payload,
               created_at, id_stamp=None) -> dict:
        incident = {"id": payload["incident_id"], "camera": camera, "kind": kind,
                    "object_id": object_id, "vehicle_id": vehicle_id,
                    "payload": payload, "created_at": created_at}
        endpoints = self.endpoints_for(kind, camera)
        if self.store is not None:
            self.store.put_incident(incident, endpoints)
        self.raised += 1
        return incident

    def stats(self) -> dict:
        return {"enabled": self.enabled, "raised": self.raised,
                "suppressed_duplicates": self.suppressed,
                "kinds_on": sorted(k for k, v in self.kinds.items() if v),
                "subscriptions": [s.describe() for s in self.subscriptions],
                "tracked_flags": len(self._flagged_since),
                "congestion": {c: s["published"]
                               for c, s in self._congestion.items()}}


def serialise(payload: dict) -> bytes:
    """Canonical body bytes. Signed and sent must be byte-identical.

    sort_keys is what makes the signature reproducible: a consumer that
    re-serialises before verifying would otherwise compute a different digest
    for the same data.
    """
    return json.dumps(payload, default=str, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
