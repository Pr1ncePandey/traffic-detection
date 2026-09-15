# Webserver, live dashboard, and incident webhooks

**Status:** planned, not started. Layer A is a prerequisite for everything else
and is worth doing on its own merits.

Turn the system from N independent one-shot processes into one long-running
service that runs every camera, serves a live dashboard, and POSTs incidents to
an external endpoint with a complete payload.

---

## Decisions taken

| Question | Decision |
|---|---|
| Process model | One server, N camera worker threads, one DB writer |
| Live view | Metadata over WebSocket, drawn client-side over MJPEG video (revised — see Layer C) |
| Webhook delivery | Persisted outbox, retry with backoff |
| Payload timing | Fire once, on track retirement |
| Congestion trigger | State transition both ways, after a minimum dwell |
| Stuck tracks | Force-fire at max dwell, then suppress |
| Routing | Subscription list with kind + camera filters |
| Auth | None at the application layer — bind to loopback/VPN |
| Crop retention | Keep crops an incident references; reap the rest |
| Row retention | Unlimited by default, same knobs as `frames` |

The last two combine into a specific split worth naming, because it resolves
what looks like a contradiction: **the dashboard serves the real-time need and
the webhook serves the durable-record need.** An operator watching the
dashboard sees a wrong-way flag the moment it is detected, over the WebSocket.
The webhook fires later, once, carrying the plate and the image. Two consumers
with different requirements, which is why a slow webhook is acceptable here and
would not be if it were the only output.

---

## What already exists

**Incidents are already first-class events.** `lanes.py:208` emits
`wrong_way` with its verdict detail, `counting.py:41` emits `crossing`,
`congestion.py:119` emits `congestion`, and the pipeline emits
`identity_bound` (`pipeline.py:396`). All land in
`events(run_id, frame_id, object_id, kind, detail_json, ts)`. The webhook layer
hooks the drain, not the analyzers — no detector needs touching.

**Alert dedup is already written.** `lanes.py:204` fires at most once per track
id via `_alerted_way`, and `lanes.py:151` holds all alerts back through a
warmup window while the geometry is checked against real motion. That is the
discipline a webhook needs, already in place.

**Storage already matches the chosen process model.** `SqliteStore` is built
around exactly one writer: WAL mode (`sqlite_store.py:29`), a single writer
thread (line 187), `check_same_thread=False` with the comment "written only by
the writer thread" (line 159), and a separate read connection so readers never
block the writer (line 173). N camera threads sharing one store preserves that
invariant exactly — which is the main reason this model was chosen over
separate processes.

**Track retirement already has a hook.** `TrackStore.evict_stale`
(`store.py:133`) calls `on_evict(tid, vehicle)` before dropping state, and
`pipeline.py:289` already uses it to run `_finalize`, which is where the plate
consensus settles and the best crop is chosen. That is the webhook's firing
point.

---

## Architecture

One process. Inside it:

- **N camera workers**, one thread each, running the existing pipeline loop.
- **One `SqliteStore`**, shared, with its single writer thread as today.
- **An event bus**, in-process, fed from each worker's `ctx.events` drain.
- **A WebSocket hub**, fanning frame metadata out to dashboard subscribers.
- **A webhook dispatcher**, its own thread, draining the outbox table.
- **An HTTP API** for queries and camera control.

**On the GIL.** N camera threads only parallelise if the heavy work releases
the GIL. OpenCV and onnxruntime do for their compute kernels, so detection and
decode genuinely overlap. Python-level work — the analysis stages, the drawing
pass, event handling — serialises. So per-camera throughput will degrade
non-linearly with camera count. **Measure it before promising a camera count**;
if it does not hold up, the fallback is worker *processes* plus one writer
process fed over a queue, which keeps the single-writer invariant but adds IPC.

**On isolation.** The cost of this model is that one camera thread raising an
unhandled exception must not take the server with it. Each worker needs a
supervisor wrapper: catch, log, mark the camera unhealthy, surface it on the
dashboard, and restart on a backoff. A camera that is silently dead is the
failure mode to design against.

---

## Layer A — storage hardening (prerequisite)

Three existing behaviours become dangerous in a long-running multi-camera
service. None are caused by this feature; all are exposed by it.

**1. The write queue blocks the frame loop.** `self._q.put(...)`
(`sqlite_store.py:251`) is a blocking put on a 20 000-slot queue. If the writer
falls behind, inference stalls. On a file that is merely slow; on a live camera
it means dropping real-world time that can never be recovered. Live sources
need a drop policy with a counter, not backpressure — and that counter belongs
on the dashboard, because a silently degrading feed is worse than an obviously
broken one.

**2. `_commit()` swallows write failures per batch.** The code documents this
itself at line 197: a schema mismatch "would appear to work while silently
discarding up to `batch_rows` object rows at a time" — 500 rows per collision.
For a one-shot run that is a contained bug. For a service running for weeks it
is unbounded silent data loss. A *persistent* write failure must raise a
visible, counted alarm, not only print.

**3. No `busy_timeout` is set.** The chosen process model means one writer, so
contention is largely designed out. But the frame reaper, external readers,
backups, and anyone opening the DB with a CLI can still collide. It is one
pragma and removes a whole class of intermittent failure.

---

## Layer B — server skeleton

**Stack:** FastAPI + uvicorn. Chosen for first-class WebSocket support, which
Layer C depends on.

**Refactoring the loop.** `pipeline.py` is currently a single blocking function
that owns the whole run. To be driven by a worker thread it needs three things:

- a **stop event** it checks per iteration, so a camera can be stopped cleanly;
- a **per-frame callback** invoked after the stages run, so the event bus and WS
  hub receive metadata without the pipeline importing either;
- **error containment** — failures currently print and continue, which is right
  for a one-shot but hides a permanently broken camera in a service.

**Endpoints:**

```
GET  /cameras                     configured cameras + live status
GET  /cameras/{id}                detail: fps, queue depth, drops, counts
POST /cameras/{id}/start|stop     control plane
GET  /incidents                   recent incidents, filterable by kind
GET  /identities/{id}             sightings across cameras
GET  /identities/{id}/path        journeys (see multi-camera-paths.md)
GET  /crops/{object_id}.jpg       serves objects.crop_path
WS   /live/{camera_id}            frame metadata stream
```

`/crops/...` is load-bearing beyond the dashboard: `objects.crop_path` is a
local path under `outputs/{camera_id}/crops`, which a webhook consumer cannot
read. This endpoint is what makes an image URL possible in the payload.

---

## Layer C — live dashboard

**This channel** is metadata only, and stays that way: the pixels travel
separately (see the video note below), so that the boxes remain data.
Per-message payload:

```json
{"camera": "demo", "frame_no": 1423, "ts": 1757664000.12,
 "boxes": [{"track_id": 88, "object_id": 1841, "identity_id": 7,
            "cls": "car", "group": "vehicle", "conf": 0.82,
            "xyxy": [420, 300, 512, 388],
            "lane_id": "right_going", "lane_flag": "wrong_way",
            "plate": "MH12AB1234",
            "attrs": {"color": "white"}}],
 "counts": {"top (exit)": 214, "bottom (entry)": 198},
 "crossings": {"a_to_b": 214, "b_to_a": 198},
 "flagged": {"wrong_way": 1, "wrong_lane": 0},
 "congestion": "clear",
 "health": {"queue_depth": 12, "rows_dropped": 0, "rows_failed": 0,
            "write_alarm": false, "frames_dropped": 0, "tracked": 18,
            "state_size": 212}}
```

`attrs` carries whatever the enrichers read for that object — colour for a
vehicle, and for a person the garment and appearance keys. It is built by
EXCLUSION (everything in `det.extra` that is not a `_conf` twin or internal
bookkeeping), so enabling a new enricher shows up on the dashboard with no
change to the pipeline or the UI. A person box is ~260 bytes against ~120 for
a vehicle, which is the whole reason the payload figures below have a range.

The dashboard filters *absent* attribute values (`bag: none`, `hat: no`) out of
its badges and box labels, while the object drawer shows them all: "no hat" is
worth knowing about one object and pure noise across ninety.

**Throttle independently of `analyse_fps`.** Push at a fixed rate (default
~8 Hz) regardless of how fast the pipeline runs. Nobody can read 30 updates a
second, and a file replaying at 3× real time would otherwise flood the socket.

**Slow subscribers drop frames, never apply backpressure.** A stalled browser
tab must not be able to slow a camera. This is the same rule as Layer A item 1,
one level up.

**Cost.** ~30 boxes at ~120 bytes is roughly 4 KB per message; at 8 Hz that is
~32 KB/s per viewer. Negligible, which is the point of this option.

**Video was deliberately out of scope here — that decision has been revised.**

The original reasoning was sound on cost: ~4 KB of JSON beats a video stream
when the boxes are the payload. What it got wrong was the purpose. Boxes over a
blank schematic answer *"is the pipeline running"*, and an operator's actual
question is *"is it right"* — which needs the road visible underneath, because
a correct box and a box on nothing look identical without it. The first thing
anyone said about the shipped dashboard was that the video was missing.

So `src/server/video.py` now fans MJPEG out per camera, `GET
/stream/{id}.mjpg` serves it, and the metadata channel above is unchanged and
still carries the boxes. Three things were kept from the original decision and
are worth not undoing:

- **The frames are clean, not `ctx.annotated`.** Serving the already-drawn
  frame would have been fewer lines and is the wrong frame: the dashboard
  draws its own boxes from the metadata, so burnt-in ones double up, and a
  burnt-in label cannot be toggled, filtered or clicked.
- **Encoding is skipped when nobody is watching**, falling back to 1 Hz so
  `/snapshot.jpg` stays fresh. A camera with no viewers pays ~1 ms a second.
- **The same drop-don't-block rule** as the hub and the write queue: one-slot
  newest-wins queues, so a stalled tab cannot slow a camera thread.

Measured on `samples/short/indian_road.mp4` (1080p, busy scene): 960 px at quality
70 is ~83 KB a frame, so ~4.3 Mbit/s per viewer at 8 fps — more than the
1–3 Mbit/s first estimated, because JPEG size follows scene detail.
`server.video.enabled: false` restores the metadata-only behaviour exactly.

**Health tiles.** Per camera: fps, write-queue depth, dropped frames, writer
failures, wrong-way total, congestion state. `identity.stats()`
(`identity.py:196`) already returns the re-id counters, and `state_size()`
exists on both `TrackStore` and `PlateIdentity` specifically so unbounded
growth can be asserted in a long run — exactly what a service needs surfaced.

---

## Layer D — incident webhooks

### What counts as an incident

| Event | Incident? | Why |
|---|---|---|
| `wrong_way` | yes | The motivating case |
| `congestion` | yes, on state change | Own rule — see below |
| `wrong_lane` | configurable | Noisier; depends on lane confidence |
| `crossing` | **no** | Every vehicle. It is a counter, not an incident |
| `identity_bound` | no | Internal bookkeeping, fires on every rebind |

Per-kind enable flags in config, so this is policy rather than code.

### Firing rule

Fire from the `on_evict` / `_finalize` path (`pipeline.py:289`), where the
voted plate and the chosen crop already exist.

**Latency, concretely.** `ttl_for()` (`store.py:16`) is
`max(5.0, (track_buffer / fps) * 3.0)`. At 30 fps with the default 30-frame
buffer that is 3.0 s, so the 5.0 s floor dominates. A webhook therefore lands
roughly **time-in-frame + 5 s** after the incident — about 9 s for a vehicle
that crosses in 4. That is the price of a complete payload, and the dashboard
already covers anyone who needs to know sooner.

**Stuck tracks: force-fire at max dwell, then suppress.** A parked car or a
track the tracker keeps alive indefinitely never reaches `evict_stale`, so the
rule above would never fire for it — exactly the stationary-obstruction case
most worth alerting on. So: after `incidents.max_dwell_s` of continuous
flagging, fire with whatever data exists, `"state": "ongoing"`, and mark the
track so it never fires again. The payload may carry a null plate; that is
better than silence. When the track eventually retires, **no second webhook is
sent** — suppression is permanent per track, which keeps the "fire once"
contract intact.

**Congestion: state transition, both ways, after a minimum dwell.**
`congestion.py:119` emits a frame-scoped event with no track id, so track
retirement cannot trigger it. Instead, fire on `clear -> congested` and
`congested -> clear`, but only once the new state has held for
`incidents.congestion_dwell_s`. The dwell is the whole point: a metric
oscillating around its threshold would otherwise emit hundreds of webhooks a
minute. Emitting both edges lets a consumer show a current state and compute
the jam's duration from the pair; onset-only cannot do either. These payloads
carry no `vehicle` block — congestion is a property of the road, not a car.

### Payload

```json
{"incident_id": "demo-1841-wrong_way",
 "kind": "wrong_way",
 "camera": {"id": "demo", "name": "demo", "lat": 12.9716, "lon": 77.5946},
 "detected_at": 1757664000.12,
 "vehicle": {"identity_id": 7, "identity_kind": "plate",
             "plate": "MH12AB1234", "plate_conf": 0.83,
             "cls": "car", "colour": "white"},
 "sighting": {"object_id": 1841, "first_seen": 1757663996.0,
              "last_seen": 1757664001.4, "frames_seen": 162},
 "detail": {"lane_id": "right_going", "expected_heading": [-0.076, -0.997],
            "observed_heading": [0.02, 0.998]},
 "image_url": "https://host/crops/1841.jpg"}
```

`plate` and `identity_id` may still be `null` — a vehicle whose plate never read
confidently has no `identities` row at all (`query.py:20` says so). The payload
must be explicit about absence rather than omitting the keys.

### Outbox

Two tables:

```sql
incidents(id, camera, kind, object_id, identity_id, payload_json, created_at)
deliveries(id, incident_id, endpoint, attempts, next_attempt_at,
           status, last_error)
```

A dispatcher thread drains `deliveries`, retries with exponential backoff, and
moves rows to a dead-letter state after a maximum attempt count — surfaced on
the dashboard, because a webhook endpoint that has been failing for a day
should be visible without reading logs.

**At-least-once.** Consumers must be idempotent on `incident_id`. Document that
in whatever consumer-facing note ships with this.

**Routing: a subscription list.** Config declares subscribers, each with its
own endpoint, secret and optional filters:

```python
"incidents": {
    "max_dwell_s": 120, "congestion_dwell_s": 30,
    "kinds": {"wrong_way": True, "congestion": True,
              "wrong_lane": False, "crossing": False},
    "subscriptions": [
        {"endpoint": "https://ops.example/hooks/traffic",
         "secret_env": "TRAFFIC_HOOK_SECRET",
         "kinds": ["wrong_way"], "cameras": []},      # [] = all cameras
    ],
},
```

One incident fans out to every matching subscription, each getting its own
`deliveries` row — which is why `endpoint` lives on that table rather than in
config alone. Retries and dead-lettering are therefore per subscriber: one
broken consumer cannot delay another. Absent or empty filters mean "everything",
so the simple single-endpoint case stays a three-line config.

**Signing.** HMAC-SHA256 over the raw body in an `X-Signature` header, plus a
timestamp header included in the signed material so a captured POST cannot be
replayed later. The secret comes from the environment, not the config file.

**Never block the frame loop.** The dispatcher owns its own thread and queue. A
customer endpoint taking 30 s to answer must cost nothing but a delayed
delivery.

---

## Layer E — retention

**Crops are never reaped today.** The `Reaper` (`frames.py:144`) only ever
deletes from the `frames` table — `reap_once` applies its age and byte budgets
to frame rows and nothing else (`frames.py:190-205`), and the module contains no
reference to crops at all. So `outputs/{camera_id}/crops/object_N.jpg`
accumulates one JPEG per object, indefinitely. Correct for a finite clip; an
unbounded disk leak in a service, and the same bug the frames reaper exists to
prevent.

**Policy: keep crops an incident references, reap the rest.** Ordinary crops
age out on a normal age/disk budget. A crop referenced by a retained incident
survives until that incident is itself reaped, so `image_url` is valid for
exactly as long as the incident it belongs to. The alternative — copying the
image into an incident-owned directory at fire time — stores the bytes twice,
and a stale 404 in a delivered payload is a worse failure than a reaper that
has to consult one more table.

This does mean the reaper gains a join it does not currently do. Two
consequences worth building for deliberately:

- A crop's reapability is no longer a function of its own age alone, so the
  byte-targeted path (`_delete_bytes`, `frames.py:208`) cannot assume the oldest
  file is always eligible. It must skip pinned crops and keep going rather than
  stopping at the first one.
- With `incidents` retention unlimited by default (below), a long deployment
  pins crops for every incident ever raised. That is the intended behaviour, but
  it means **incident volume sets a disk floor** the crop budget cannot reclaim.
  The dashboard should show pinned-crop bytes separately from reclaimable ones,
  so the floor is visible before it becomes a full disk.

**Row retention: unlimited by default, same knobs as frames.** `events`,
`incidents` and `deliveries` each get
`{max_age_hours: 0, max_disk_gb: 0, check_interval: 60}`, with `0` meaning
unlimited — matching `frames.retention` (`config.py:159`) exactly, so there is
one retention idiom in the codebase rather than two. Nothing is deleted that an
operator did not ask to have deleted; a long-running deployment opts in to a
cap. Delete order matters: `deliveries` rows reference an incident, and crops
are pinned by one, so an incident must be the last thing to go.

`events` is the high-volume table here — `crossing` fires per vehicle per line —
so it is the one most likely to need a real cap in practice. Worth revisiting
with a measured row count once a camera has run for a week.

---

## Deferred deliberately

**Application-level auth: none.** The API, WebSocket and crop endpoints are
protected by network placement only. Two requirements follow, and they are not
optional given `/crops/{object_id}.jpg` serves number-plate imagery and the
WebSocket streams plate strings:

- The listen address **defaults to `127.0.0.1`**, never `0.0.0.0`. Binding
  wider must be an explicit, documented act.
- That assumption gets written down next to the deployment instructions, because
  the difference between safe and exposed here is one config value and no code
  will complain.

Revisit if this is ever operated by more than one person, handed to a client, or
reachable from outside a VPN — at which point the shared-token option is a small
change, since every endpoint already routes through one place.

---

## Dependency order

Layer A is a prerequisite and stands alone — it fixes real defects in the
current storage layer whether or not a server is ever built. Layer B depends on
A. Layers C and D both depend on B and are independent of each other, so they
can proceed in parallel. Layer E depends on D only for the crop-pinning join;
its row-retention half is independent and, like Layer A, is worth doing for the
existing system regardless.

Layer D's payload additionally wants `camera.location` from Layer 2 of
`multi-camera-paths.md` and absolute timestamps from Layer 1 of the same, though
it degrades gracefully without either.
