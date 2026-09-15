# Traffic Detection

Analyses road traffic video from a **file, an RTSP/HTTP stream, or a webcam**.
Detects anything on the road, tracks it, reads number plates, and writes an
object-level record to SQLite alongside every stored frame.

## What it answers

- What passed through the view, of which type, and when?
- Where did a specific object go, by track id or by number plate?
- Is this the same vehicle that was here earlier, or yesterday, or on another
  camera? (Plate-keyed, see Identity below.)
- How many crossed in each direction, on which road, and was any going the wrong way?
- How congested was the road, over time?

## How it works

```
source (file | rtsp:// | webcam)
  -> FpsSampler        analyse at a target FPS, independent of source rate
  -> FrameReader       decode on its own thread, per-camera backpressure
  -> Detector          YOLO + ByteTrack, all 80 COCO classes, grouped
  -> Analyzers         counting | lanes | anpr | congestion   (config-driven)
  -> Storage           SQLite record + raw & annotated frames + object images
```

1. **Detection (YOLO).** Every frame is analysed for **all 80 COCO classes**, not
   just vehicles. Each detection is tagged with a *group* — `vehicle`, `person`,
   `animal`, `obstacle`, `infrastructure`, `other` — so an analyzer can ask "is
   this a vehicle?" without knowing class ids.
2. **Tracking (ByteTrack).** Detections are linked across frames and each object
   keeps one id while continuously visible. **See the identity caveat below — this
   is the single most important thing to understand about the output.**
3. **Analyzers.** Each use case is a module registered by name and enabled in
   config. Adding one requires no change to the pipeline.
4. **Storage.** One row per object (not per frame), an extensible attribute
   table, every analysed frame stored raw *and* annotated, and one image per
   detected object.

## Identity: sightings vs vehicles

There are **two** levels of identity, and the difference is the single most
important thing to understand about the output.

| | what it is | scope | where |
|---|---|---|---|
| **track id** | one *sighting* | inside one run | `objects.track_id` |
| **identity** | one *entity* | across runs **and cameras** | `identities.id` (`kind='plate'` for a car), referenced by `objects.identity_id` |

**Why two.** ByteTrack cannot re-assign a previous id to a reappearing object.
It matches on Kalman-predicted motion and IoU overlap only (`match_thresh:
0.8`) with no appearance model. A lost track is held for `track_buffer` frames
(30 by default, ~1s at 30fps) and then deleted; anything that returns after
that gets a **fresh, higher id**. Ids are never recycled.

So the track id alone over-counts real vehicles. The number plate is the only
identity this system reads that survives an absence, and it is now used as the
key: `reid:` in `config.yaml` resolves a plate to a stable **vehicle id**, held
in the `identities` table. A car that leaves and comes back gets its original
vehicle id back — on the annotated video it is labelled `V7`, not a new `#31` —
and so does the same car tomorrow, or on a different camera writing to the same
database.

**Each sighting still gets its own `objects` row.** That is deliberate: a
vehicle that genuinely passes twice really did pass twice, so throughput
counting (`A->B` / `B->A`) is unaffected by re-identification. `identity_id` is
what ties the sightings together. How many times a car was seen is
`SELECT COUNT(*) FROM objects WHERE identity_id = ?`.

```bash
python query.py --vehicle 7              # every sighting of one car
python query.py --plate HR26DK8337       # same, found by plate
```

### The honest limits

- **No readable plate means no durable identity.** Too far, occluded, night,
  or a two-wheeler whose plate never resolves: `identity_id` stays `NULL` and
  those sightings are not linked to each other. They are not guessed at. The
  report counts them in "Distinct track IDs" and not in "Distinct vehicles".
- **Binding is deliberately conservative.** A plate must clear
  `reid.min_conf` (0.7, higher than the 0.5 needed merely to *display* a
  plate) and, by default, fit a real registration template. A wrong caption
  costs one bad frame; a wrong merge fuses two cars' histories permanently.
- **Near-match merging is off** (`reid.fuzzy_distance: 0`). Real plates differ
  by one character — `...8337` and `...8338` are two different cars — so
  fuzzy matching fuses strangers rather than repairing OCR noise.
- **Identity arrives a little after the vehicle does.** The plate is not
  readable on a track's first frame; it takes a few reads plus a vote
  (`plate.read_every`, `plate.max_reads`). The id is bound as soon as the
  plate is confident and re-checked against the final voted plate when the
  track retires, so the value stored in the database is the settled one.
- For short-occlusion recovery *within* a run, `tracker.name: botsort.yaml`
  with `with_reid: True` adds appearance re-association. That is a config
  change, complementary to this — it keeps one track id alive across a brief
  occlusion, where plate identity links separate sightings.

## Detecting anything on the road — and the honest limit

`model.classes: null` detects all 80 COCO classes. It also accepts a group name
(`vehicle`, `on_road`), a list of names (`["car","truck"]`), or the legacy id
list (`[2,3,5,7]`).

**COCO has no class for pothole, debris, traffic cone, barrier or fallen branch.**
Those are not misclassified — they are invisible to this model. The `obstacle`
group is a pragmatic reuse of COCO classes that have no business on a
carriageway (a suitcase, a chair, a bottle); it is a useful signal, not a real
obstacle detector. Naming arbitrary road objects needs an open-vocabulary model
(YOLO-World / YOLOE), which drops in behind the same `Detector` interface.

## Input sources and analysis rate

```bash
python main.py --source samples/short/input.mp4    # file
python main.py --source rtsp://cam/stream          # live stream
python main.py --source 0                          # webcam index
python main.py --source rtsp://cam/stream --analyse-fps 5
```

`--analyse-fps` sets frames analysed **per second**, independent of the source
rate, and replaces the old integer `frame_skip` (still accepted, translated on
load). Files sample on frame index so runs are reproducible; live sources sample
on wall clock, because RTSP frequently reports its fps as 0 and its true rate
drifts.

**Backpressure** is per camera, because the right answer differs:

| mode | behaviour | default for |
|---|---|---|
| `drop_oldest` | keep only the newest frames; latency stays flat, frames are lost | live sources |
| `buffer_all` | block the reader; nothing is lost, latency may grow | files |

Streams reconnect with capped exponential backoff (0.5s → 30s).

## Storage

SQLite (`outputs/traffic.db`, WAL) written by a background thread that commits
every 500 rows or 2 seconds — so a hard kill loses at most ~2s of rows, and
nothing depends on the process exiting cleanly.

| table | one row per | notes |
|---|---|---|
| `runs` | run | source, geometry, analyse_fps, full config JSON |
| `frames` | analysed frame | `raw_path` + `annotated_path`, or segment + offset |
| `objects` | **one sighting** | class, group, first/last seen, best conf, crop path, lane, `identity_id` |
| `identities` | **one entity** | `kind` + `key` (UNIQUE together), first/last seen across every run — the durable, class-agnostic identity |
| `detections` | object per frame | box, conf, crossing event, lane flag |
| `attributes` | object + key | **tall**: `('plate_number', 'HR26AF7196', 0.94)` |
| `events` | notable moment | crossing, wrong_way, plate_read, identity_bound, congestion |

`attributes` is deliberately tall rather than wide columns: adding colour, brand
or speed later needs no migration, and the `(key, value)` index keeps plate
lookup fast.

`objects.identity_id` is what makes a re-entering car one car — see
[Identity](#identity-sightings-vs-vehicles). It is `NULL` when no plate was
read confidently. There is deliberately **no** sightings counter on `identities`:
the count is `COUNT(*)` over `objects.identity_id`, which cannot drift from the
rows it claims to count.

```sql
-- what was detected, by group
SELECT cls_group, cls_name, COUNT(*) FROM objects GROUP BY 1,2 ORDER BY 3 DESC;
-- plates, best read per object
SELECT object_id, value, conf FROM attributes WHERE key='plate_number';
-- cars that left and came back: one vehicle, several sightings
SELECT v.id, v.plate, COUNT(o.id) sightings
  FROM identities v JOIN objects o ON o.identity_id=v.id
  GROUP BY v.id HAVING COUNT(o.id) > 1 ORDER BY sightings DESC;
-- an object with its frames
SELECT f.frame_no, f.raw_path, d.conf FROM detections d
  JOIN frames f ON f.id=d.frame_id WHERE d.object_id=42 ORDER BY f.frame_no;
```

### Stored frames and object images

Every analysed frame is stored **twice** — untouched original and annotated —
plus **one image per detected object** (the sharpest crop seen, not the first).

`frames.format` picks how:

| format | disk (1080p, 30fps, raw+annotated) | when |
|---|---|---|
| `jpeg` (default) | **~2 TB/day** | finite clips; ~614 MB for the 812-frame sample |
| `segments` | **~130 GB/day** | permanent streams; rolling H.264, seek by (segment, offset) |

Those figures are **measured**, not estimated: 1080p Indian traffic footage
encodes to 378 KB/frame at q85. Switch to `segments`
before pointing this at a 24/7 camera. Retention (`max_age_hours`,
`max_disk_gb`) deletes oldest-first and keeps rows and files consistent.

## Use cases (analyzers)

```yaml
analyzers: ["counting", "lanes", "anpr", "congestion"]
```

| analyzer | what it does |
|---|---|
| `counting` | A→B / B→A crossings over two zones |
| `lanes` | lane membership, `wrong_way` (against the measured flow), `wrong_lane` (disallowed type) |
| `anpr` | number plate, vehicles only |
| `congestion` | free / busy / heavy / jammed from density, occupancy and motion |

Each analysis runs on the objects detection, classification and tracking have
already produced, and is independent of the others. **Adding a use case** means
one file and one config entry, with no edit to `pipeline.py`.

There are two shapes. The simple one is a single `process(ctx)`:

```python
class MyAnalyzer:
    name = "mine"
    def setup(self, source, cfg): ...
    def process(self, ctx):      # ctx.detections, ctx.raw, ctx.annotated, ctx.store
        ctx.emit("something", {"detail": 1}, track_id=...)
    def summary(self): return {}

register("mine", MyAnalyzer)
```

The **staged** shape splits that into three phases, which is what lets
independent analyses run side by side. `process()` did three different kinds of
work at once — reading detections (safe to parallelise), mutating the shared
`TrackStore` (not safe), and drawing on the one shared canvas (not safe) — so
only the first is separated out and scheduled:

```python
class MyAnalyzer:
    name = "mine"
    concurrent = True                  # my compute() touches nothing shared

    def setup(self, source, cfg): ...

    def compute(self, view) -> Findings:
        """PURE. Reads an immutable AnalysisView + my own state. Any thread."""
        found = Findings()
        for box in view.tracked():     # box.bbox, box.ground_point, box.group
            found.set(box.track_id, my_field="value")
        found.event("something", {"detail": 1}, track_id=...)
        return found

    def apply(self, ctx, findings):    # SERIAL, config order: shared writes
        ...
    def draw(self, ctx, findings):     # SERIAL, config order: overlay only
        ...
    def summary(self): return {}
```

`src/analysis/lanes.py` is the worked example. `compute()` gets an
`AnalysisView` whose boxes are frozen tuples, so a compute phase *cannot* write
to shared state even by accident; `apply` and `draw` are replayed in config
order, so ordering stays load-bearing (counting can still read the lane a
vehicle was in). `src/analysis/stage.py` owns the scheduling, and legacy
single-phase analyzers keep working unchanged — migration is per-analyzer, not
a flag day.

```yaml
analysis:
  parallel: false    # run concurrent-safe compute phases on worker threads
  workers: null      # null = one per concurrent analyzer
```

`parallel` is off by default, and honestly: the wrong-side rule is a few dozen
float operations per object (measured 0.19 ms/frame) — GIL-bound, and dispatch
costs more than it saves. It is worth turning on once a *heavy* analysis is
staged. Measured on a 400-frame run of Indian test footage: with
`counting,lanes` 24 fps; adding `anpr` drops it to 5.4 fps. ANPR is the
analysis with something to gain, because plate detection and OCR release the
GIL. Lanes is structured this way so it can sit next to that without being
serialised behind it.

If a new analyzer ever forces a pipeline change, that seam is wrong and should
be fixed rather than worked around.

## Wrong-side detection

This runs after classification and tracking and asks two separate questions per
object: which lane is it in, and is its heading aligned with the way that lane
flows.

```yaml
lanes_mode: "auto"        # measure the geometry from motion (default)
                          # "explicit" = use `lanes:` below, and CHECK it
                          # "off"      = no lane analysis
```

**The geometry is measured or checked, never just asserted.** It used to be a
pair of polygons hand-tuned on `samples/short/input.mp4` and inherited unchecked by
every other source. On Indian test footage that divider cut diagonally
through the middle of a single one-way carriageway: 2 of 90 vehicle tracks were
reported wrong-way, **both driving correctly, 0 real offenders**. So during a
warmup window (`lanes_rules.warmup_frames`, default 300) no alert can fire and
the flow of each lane is measured instead:

```
[verify] right_going: configured up (-4deg), observed up (+4deg) over 27 tracks
         -> agrees (alignment +0.99)
[verify] left_coming: configured down (+176deg), observed up-right (+40deg)
         over 9 tracks -> OPPOSES (alignment -0.71)
[verify] MISMATCH (OPPOSES): lane 'left_coming' is configured to flow
         down (+176deg) but its 9 observed track(s) flow up-right (+40deg).
         Correctly-driving vehicles in it would be reported wrong-way.
[lanes]  on_mismatch=suppress: wrong-way checking disabled for left_coming;
         every other lane keeps working.
```

`lanes_rules.on_mismatch` decides what happens to *that lane*: `suppress`
(default — stop checking it, keep everything else running), `flip` (trust the
measurement and correct its flow), `warn` (say so but alert anyway), `off`
(disable lane analysis for the run).

In `auto` mode nothing is invented. If only one direction of travel is
observed — true of **both** sample clips — you get one carriageway with a
measured flow, not a made-up oncoming lane. A vehicle heading the other way
through it is still flagged.

What the rule itself does differently, all of it measured on the two clips:

| before | now |
|---|---|
| sign of `dy` against one of 4 cardinal directions | cosine between travel and a per-lane flow **vector** — a road running diagonally in frame has no cardinal direction |
| flat 4 px/frame gate (96% of far-object frames and 68% of near ones never reached it) | travel over a window, gated on `max(6 px, 0.15 × the object's box height)` |
| bbox **centroid** — drifts upward for an approaching vehicle whose box is clipped by the bottom edge | **ground point** (bottom-centre), and no verdict at all while the box touches a frame edge |
| 5 **consecutive** opposing frames, reset by any frame under the gate | 5 opposing samples out of the last 8 |
| flat 14 px keep-out from the lane edge — metres of road near the top of frame | keep-out scaled by the object's own height, plus a band around the divider |
| the divider was the shared edge of two polygons; a point on it fell into whichever was listed first | the divider is a **line**: side is a signed distance, and it is drawn across the whole frame so a wrong one is obvious |
| position history was differenced straight across a detection gap | history older than `max_gap_frames` (or a step longer than 1.5 box-heights) is discarded — a track the detector lost and re-found is not one observation |
| a false positive latched permanently into the database with no evidence | every verdict carries its alignment, travel and heading; `summary` reports why verdicts were withheld (`too_slow`, `edge_clipped`, `near_divider`, `track_discontinuity`, `unverified_geometry`) |

Check any camera's geometry against its own footage without running the
pipeline:

```bash
python tools/calibrate.py --verify --camera demo          # does the config match?
python tools/calibrate.py --suggest-lanes --out cameras/new.yaml
python tools/draw_lanes.py --source other.mp4             # click it by hand
python tools/test_lanes.py                                # 47 rule checks, no video
python tools/test_stage.py                                # 30 scheduler checks
```

## Number plates

CPU pipeline: a YOLO plate-box detector (`models/lp_detector.pt`, 6MB) finds the
plate, then a **selectable OCR backend** reads it, followed by normalisation,
Indian-format correction and a plate-ish gate. Swap the reader with
`plate.ocr_backend` (or `--ocr-backend`), no code change:

| backend | install | CER | ms/plate | what it is |
|---|---|---|---|---|
| `fast_plate` **(default)** | in `requirements.txt` | 0.54 | 8 | Plate-specialised CCT model, ~5MB ONNX. |
| `paddle_anpr` | see below | **0.19** | 50 | PP-OCRv5 finetuned on 558k Indian plates; reads two-row (bike) plates. More accurate, but `pip` cannot install it. |
| `rapidocr` | in `requirements.txt` | 0.88 | 370 | Generic scene-text OCR, **not** plate-specialised. Fallback only. |

CER = character error rate, lower is better, measured on `samples/plate_gt.csv`
(5 hand-labelled plates; the crops in `samples/plates/` were cut from the
earlier Indian clip, since replaced by `samples/indian_road.mp4`). Only this
column is comparable — the vendors' published figures come from different datasets. n=5 is
thin: treat the ordering as solid and the values as indicative.

**The default is deliberately not the most accurate one.** `paddle_anpr` was
the default and cannot be installed by `pip install -r requirements.txt`, so out
of the box it failed to load and no plates were read at all. Via `main.py` that
at least printed a reason; on the dashboard it surfaced as an empty plate column
with no explanation, which is the worst of both. So the default is now the
backend that works everywhere, and `paddle_anpr` is the upgrade. `/cameras`
reports which backend actually loaded, and the dashboard shows a banner naming
the failure when one did not.

`paddle_anpr` needs weights and a PaddleOCR checkout:

```
pip install paddlepaddle safetensors scikit-image
mkdir -p models/anpr_ocr && cd models/anpr_ocr
curl -LO https://huggingface.co/Awiros/anpr-ocr/resolve/main/model.safetensors
curl -LO https://huggingface.co/Awiros/anpr-ocr/resolve/main/en_dict.txt
cd ../.. && git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git models/PaddleOCR
```

**When to read matters as much as which model.** The plate sits low on the
vehicle, so as a car enters from the bottom of frame its roof appears first and
the plate is still clipped off-screen. The plate is *widest* the moment it clears
the edge, but that is also its most slanted and motion-blurred view: confidence
peaks 9–16 frames later on a ~10% smaller plate. Selecting the biggest, sharpest
crop therefore picks a systematically poor frame.

So reading samples several frames per vehicle (`read_every`, `max_reads`) and
combines them character-by-character among the confident reads (`vote`,
`vote_min_conf`). One car produced 48 distinct readings over 64 frames and the
errors move around, so the majority is usually right — but only after gating on
confidence: voting over *every* frame is worse than not voting, because distant
views are junk (0 of 9 reads below conf 0.5 were correct, vs 9 of 13 above 0.9).

Accuracy here is gated by capture, not by the model: in this clip the plates are
21–115 px wide and most are not human-readable. A camera angle that puts more
pixels on a front-on plate is the single biggest remaining win.

## Outputs

| Path | Contents |
|---|---|
| `outputs/traffic.db` | **the record** — objects, vehicles, detections, attributes, events, frames |
| `outputs/frames/raw/` | every analysed frame, untouched |
| `outputs/frames/annotated/` | every analysed frame, with boxes and overlays |
| `outputs/crops/object_<id>.jpg` | one image per detected object |
| `outputs/annotated.mp4` | annotated video |
| `outputs/tracks.csv` | frame-level CSV export (kept for compatibility) |
| `outputs/summary.txt` | text summary |
| `outputs/report.html` | HTML report, one card per object |

## Multi-camera journeys

Where did one car go? `objects.identity_id` is already fleet-global — one
database serves every camera and `vehicles.plate` is `NOT NULL UNIQUE` — so a
journey is structurally `objects WHERE identity_id = ?` ordered by time and
joined to `runs`. Two things had to be fixed first.

**Time had to become comparable.** The pipeline writes two incompatible kinds of
number into `first_seen_s`: a wall-clock epoch for a live source, and
seconds-from-clip-start for a file. Both landed in one column with nothing
distinguishing them, so a fleet-wide database holding a live run and a file run
made any `ORDER BY` time return silent nonsense. `runs.time_base` now records
which clock a run used, and `--recorded-at` anchors a file's clip-seconds to
when the footage was actually shot:

```bash
python main.py --camera junction_7 --recorded-at 2026-09-12T08:30:00
```

Storing the *processing* wall-clock instead does not work, and it is the obvious
thing to reach for: process camera A's clip today and B's tomorrow and every A
sighting precedes every B sighting; and within one run the gaps between
detections are a function of throughput, so the ordering would encode GPU speed.
See `src/timebase.py`.

A file run with no `--recorded-at` is genuinely **unorderable**, and is reported
as such rather than guessed at.

**Cameras needed locations.** Set `camera.location` in `cameras/<id>.yaml`:

```yaml
camera:
  location: {lat: 12.9716, lon: 77.5946}
```

It buys one thing worth more than it looks: haversine distance over elapsed time
gives an implied speed per hop, and anything above `journey.max_speed_kmh` is an
OCR collision or a cloned plate rather than a journey. That recovers most of
what a road-topology graph would be wanted for with no topology file to
maintain. The value is snapshotted onto the run row, so editing the yaml later
cannot retroactively move a historical sighting.

**Read the yield before trusting the routes.** Cross-camera identity needs two
cameras to produce *character-identical* voted plates (`reid.fuzzy_distance` is
0, deliberately), so the honest expectation is a sparse, noisy hop set:

```bash
python query.py --yield      # how many vehicles crossed more than one camera
python query.py --path 7     # one vehicle's route
```

If the yield is zero, journeys would be scaffolding around an empty set and the
work to do is plate accuracy, not route assembly. With one camera on the sample
clips it *is* zero, for the reason in "The honest limits" above.

### Reproducing it without a camera fleet

`cameras/_twocam_north.yaml` and `_twocam_south.yaml` point at the **same**
clip, anchored five minutes apart and placed 3.4 km apart, so every vehicle in
the footage is seen by both cameras and there are real hops to assemble:

```bash
python main.py --camera _twocam_north --ocr-backend fast_plate
python main.py --camera _twocam_south --ocr-backend fast_plate
python query.py --yield        # 3 of 5 vehicles crossed both cameras
python query.py --path 2
```

That yields genuine journeys at ~41 km/h from real detections. The failure
modes come from re-anchoring one camera's clock: 20 s apart makes the hop
813 km/h and it is flagged `[SUSPECT]`; identical anchors make it a `CONFLICT`;
two hours apart splits it into two trips; removing `recorded_at` moves that
sighting to `unorderable`.

**What this does not prove.** It links because identical footage makes the OCR
produce identical strings — right or wrong — and exact match is all re-id asks
for. Two real cameras see different angles, lighting and plate sizes, and will
not agree that readily. Read this rig as "the assembly works", never as
"cross-camera re-id works"; only `--yield` on real footage answers the latter.
The `_` prefix keeps these two out of `serve.py`'s camera discovery.

Both rig files set `reid.min_conf: 0.4`, and that is the one thing in them not
to copy to a real camera. The best plate read on this clip scores 0.53, under
the 0.7 fleet default, so at the default exactly **one** vehicle binds and you
see one hop rather than three. The default is higher than `plate.min_conf` on
purpose — a read good enough to print on a box is not good enough to merge two
vehicles' histories on, and a wrong merge is permanent.

```
Vehicle V1  plate KA01AB1234
  3 sighting(s) on record across 2 camera(s): cam_a, cam_b

  Journey 1/1: cam_a -> cam_b
    2026-09-12 13:30:00 -> 2026-09-12 13:36:00  (6m00s, 3.11 km)
      cam_a            2026-09-12 13:30:00  dwell 40s     2 sightings
      ~                unobserved: 3.11 km in 5m00s -> 37 km/h
      cam_b            2026-09-12 13:35:40  dwell 20s     1 sighting
```

What it asserts is **only observed hops**. The stretch between two cameras is
printed as a gap — the system saw a car at A and later at B, it did not see what
happened in between and does not claim to. No intermediate camera is ever
inferred. Consecutive sightings at one camera collapse into a single visit with
a dwell time; an idle gap over `journey.max_gap_s` splits one trip from the
next, because a car at A in the morning and at A again at night made two trips
rather than a loop.

Bad data is made visible rather than hidden. A hop implying 2 400 km/h is
flagged `[SUSPECT]` and still printed, because a bad plate merge is worth
seeing; deleting it would leave a plausible route with no sign its identity is
wrong. Two sightings that overlap in time are reported as a `CONFLICT` — one car
cannot be at two cameras at once, so that plate is cloned or two cars read the
same — and the journey is still built, with the affected hops marked.

## Running as a service

`main.py` runs one camera once. `serve.py` runs every camera continuously,
serves a live dashboard, and POSTs incidents to an external endpoint:

```bash
python serve.py                    # every cameras/*.yaml
python serve.py --camera demo      # just one
```

Then open <http://127.0.0.1:8000/>. `/docs` has the API.

One process holds N camera worker threads, **one** `SqliteStore` with its single
writer thread, an in-process event bus, a WebSocket hub, a webhook dispatcher
and the HTTP API. One store rather than N is not tidiness: `SqliteStore` is
built around exactly one writer on one connection, and N stores would put N
writer threads on one SQLite file.

| Endpoint | |
|---|---|
| `GET /cameras` | configured cameras plus live status |
| `GET /cameras/{id}` | detail: fps, queue depth, drops, counts |
| `POST /cameras/{id}/start`, `/stop` | control plane |
| `GET /incidents`, `/deliveries` | the outbox and its delivery state |
| `GET /events` | the raw event log, filterable by kind |
| `GET /search?q=` | open-vocabulary search over embedded crops |
| `GET /objects`, `/objects/{id}` | browse the object record, with attributes |
| `GET /identities` | durable identities plus sighting counts (`?kind=plate`) |
| `GET /identities/{id}`, `/identities/{id}/path` | sightings, and journeys |
| `GET /vocabulary` | cameras / classes / groups / attribute values |
| `GET /yield` | the cross-camera measurement above |
| `GET /crops/{object_id}.jpg` | serves `objects.crop_path` |
| `GET /stream/{camera_id}.mjpg` | live MJPEG video |
| `GET /snapshot/{camera_id}.jpg` | the latest frame, once |
| `WS /live/{camera_id}` | frame metadata stream |

**There is no application-level authentication.** The service is protected by
network placement only, so it binds to `127.0.0.1` by default and `--host
0.0.0.0` prints a warning. That default matters, and it matters more now the
dashboard shows video: `/crops` serves number-plate imagery, `/live` streams
plate strings, and `/stream` serves live road footage, so exposing this on an
untrusted network publishes all three. Put it behind a VPN or an authenticating
proxy first. `server.video.enabled: false` turns off the stream alone and
leaves the rest of the dashboard working, if that is the trade you want.

**On the GIL, honestly.** N camera threads only parallelise where the heavy work
releases it. OpenCV and onnxruntime do for their compute kernels, so decode and
inference genuinely overlap; the analysis stages, drawing and event handling are
Python and serialise. Per-camera throughput therefore degrades non-linearly with
camera count — `/cameras` reports each worker's achieved fps so you can measure
it on your host rather than trust a promised number.

### Live dashboard

Six views at <http://127.0.0.1:8000/> — live, search, objects, vehicles,
incidents, system. Press `1`–`6` to switch, `c` to cycle camera, `f` for
fullscreen, `/` to search.

**Two channels, stacked.** The picture is an MJPEG stream of the *clean* frame;
the boxes are drawn on a canvas over it from the metadata WebSocket. Keeping
boxes as data rather than letting OpenCV burn them into the JPEG is what makes
them toggleable, filterable and clickable — the toolbar can hide labels, show
attributes, draw trails, or show only flagged objects, none of which is
possible once a box is pixels.

That the two line up is not luck. The canvas is sized to the camera's **source**
resolution while both elements are stretched to the same box by CSS, so the
encoder has to preserve aspect ratio when it downscales; a fixed output size
would put every box a few pixels off its object and look like a tracking bug.

| | cost per viewer | knob |
|---|---|---|
| metadata (boxes, health) | ~4 KB/msg for vehicles, ~8 KB with person attributes → ~32–64 KB/s at 8 Hz | `server.push_hz` |
| video | ~83 KB/frame → **~4.3 Mbit/s** at 8 fps, measured on `samples/short/indian_road.mp4` (1080p, busy) | `server.video.{fps,quality,max_width}` |

A quiet scene costs far less — JPEG size follows detail. **Encoding is skipped
when nobody is watching**, dropping to `snapshot_fps` (1 Hz) so `/snapshot.jpg`
stays useful for thumbnails.

**Watching a camera does cost it something, measured.** On one 1080p camera on
this host: ~10.1 fps with no viewers (1 encode/s) against ~9.2 fps with three
(~5 encodes/s) — roughly 10%. Encoding itself is only 1.1 ms a frame, so most
of that is the GIL contention of three concurrent HTTP streams rather than the
JPEG work, and it is the same non-linearity that makes per-camera throughput
fall as camera count rises. If you need the frame rate, turn video off or drop
`fps`; don't assume viewers are free.

Slow clients on either channel **drop frames rather than applying
backpressure**: a stalled browser tab must not be able to slow a camera.

Set `server.video.enabled: false` for the original metadata-only dashboard.
Everything else keeps working and `/stream` returns 503.

The health tiles are as much the point as the boxes: fps, write-queue depth,
shed rows, lost rows, writer alarm, dropped frames, wrong-way and congestion
totals. A camera that is silently dead is the failure mode this layer exists to
make visible.

**The index maintains itself.** Embeddings are not built by the pipeline - no
real-time decision needs one, and keeping CLIP out of `pipeline.py` makes a
model swap a script re-run. That used to mean search silently returned nothing
on a fresh database, and silently covered part of the corpus on an old one,
until somebody remembered `tools/embed_crops.py`. So the service embeds its own
backlog on a thread, alongside the webhook dispatcher and the retention reaper.
The frame loop is untouched.

It runs on a **duty cycle**, because L/14 inference is ~317 ms/crop and the
cameras want that CPU: `batch` crops, then a deliberate `pause_s`. The defaults
(8, 3.0 s) come to ~45%, about 1.4 crops/s against the CLI's 3.1 - slower on
purpose, since a backlog nobody is waiting on must not cost the live feeds
frame rate. `server.search.pause_s: 0` runs flat out;
`embed_in_background: false` goes back to embedding by hand.

Coverage is on screen and in the log rather than something to discover by
searching. Startup prints it:

```
[server] search index: 434 of 1980 crops embedded in clip-vit-l14,
         1546 without vectors - the background embedder is working through them
```

and the search view carries `indexed / coverage % / no vectors / space /
embedder state` tiles, so a **partial** index — the genuinely dangerous state,
where results look fine and 60% of the corpus was never indexed — says so.

`server.search.model` is the one place the embedding space is named. It used to
be a constant in the server while `tools/embed_crops.py` defaulted to a
different space, so running the embedder with no flags filled an index nothing
queried and search stayed empty with no error. Cosine is only meaningful within
one space, so two indexes can never cover for each other. The eligibility rules
(`min_px`, `min_conf`, `max_area`) live in `src/query/embed.py` and are shared
by the CLI and the service, so the two cannot build to different standards.

**Search, and what it does not claim.** The search view ranks embedded crops by
similarity to free text. It reports "the nearest N of M candidates" and never
"found X", because measurement found no similarity floor separating present
concepts from absent ones — so the top hit is the closest crop whether or not
the thing you asked for is in the footage. The UI carries that framing, the
candidate count and the score spread on every result, and surfaces the
router's advisory when an exact SQL answer would be cheaper. Embed first with
`python tools/embed_crops.py`; without it the view says so.

The rest of the honest limits are on screen too, for the same reason they are
in this README: journeys draw unobserved stretches as gaps and list sightings
they cannot order rather than silently sequencing them, an empty flagged-object
list explains that `no-data` lanes are unverified rather than clean, and a crop
that retention reaped renders differently from one that never existed.

**Empty is not the same as broken, and the dashboard has to say which.** Two
cases earned dedicated messages because both first appeared as a blank panel
with no explanation:

- **An empty plate column.** Ambiguous between "no plate was readable in this
  frame" — normal, constantly — and "the OCR backend never loaded", a setup
  problem that will not fix itself. `/cameras` reports `plate_ocr`
  (`{backend, ok, error}`), the column reads `reader off` rather than `–` when
  a backend failed, and the live view banners the reason with the fix. Note it
  stays silent until a backend is actually attempted: the engine initialises on
  the first frame carrying a plate box, so there is legitimately nothing to
  report in the first seconds of a run.
- **An empty search.** Almost always an unindexed model rather than a query
  with no matches, so the response carries `indexed_models` and the view
  distinguishes "nothing is embedded" from "*this* space is empty while the
  other has 1,450 vectors" — the second is a dropdown away from working, and
  calling it unindexed would be false. Cosine is only meaningful within one
  embedding space, so the two indexes can never be merged to paper over it.

### Incident webhooks

| Event | Incident? | Why |
|---|---|---|
| `wrong_way` | yes | the motivating case |
| `congestion` | yes, on state change | see below |
| `wrong_lane` | configurable | noisier; depends on lane confidence |
| `crossing` | **no** | fires for every vehicle. A counter, not an incident |
| `identity_bound` | no | internal bookkeeping; fires on every rebind |

Per-vehicle incidents fire **once, at track retirement** — the only moment the
payload is complete, because that is when the plate vote has settled and the
sharpest crop has been chosen. The cost is latency: `ttl_for()` floors at 5 s,
so a webhook lands roughly time-in-frame + 5 s after the event. That is accepted
deliberately, because the dashboard already serves anyone who needs to know
sooner. Two consumers, two different requirements.

Three rules exist because the simple version is wrong:

- **Stuck tracks.** A parked car never reaches eviction, so "fire at
  retirement" would never fire for exactly the stationary-obstruction case most
  worth alerting on. After `incidents.max_dwell_s` of continuous flagging it
  force-fires with `"state": "ongoing"` and whatever data exists, then is
  suppressed permanently — no second webhook when it eventually retires, so
  "fire once" stays true.
- **Congestion has no track**, so retirement cannot trigger it. It fires on
  `clear -> congested` and back, but only once the new state has *held* for
  `congestion_dwell_s`; a metric sitting on its threshold would otherwise emit
  hundreds of webhooks a minute. Both edges are emitted so a consumer can show
  current state and compute a jam's duration from the pair.
- **Absence is explicit.** `plate` and `identity_id` may be `null`, because a
  vehicle whose plate never read confidently has no `vehicles` row at all. The
  keys are present with null values rather than omitted.

Delivery is a persisted **outbox**: `incidents` holds the payload, `deliveries`
holds one row per subscriber. The incident is committed before any POST, so a
crash between raising and sending loses nothing. Retries are exponential with a
cap, then `status='dead'` — visible on the dashboard, because an endpoint that
has been failing for a day should not require reading logs. Because `endpoint`
lives on the delivery row, retries and dead-lettering are **per subscriber**:
one broken consumer cannot delay another.

Delivery is **at-least-once**, so consumers must be idempotent on
`incident_id` — which is derived from the sighting rather than a clock
precisely so it is stable across retries.

```yaml
incidents:
  base_url: "https://traffic.internal"     # for image_url in the payload
  kinds: {wrong_way: true, congestion: true, wrong_lane: false}
  subscriptions:
    - endpoint: "https://ops.example/hooks/traffic"
      secret_env: "TRAFFIC_HOOK_SECRET"     # from the ENV, never this file
      kinds: ["wrong_way"]
      cameras: []                           # [] = every camera
```

Signed with HMAC-SHA256 over `timestamp + "." + body` in `X-Signature`. The
timestamp is *inside* the signed material, not merely alongside it, so a
captured POST cannot be replayed later.

### Retention

Two things were previously unbounded and only became dangerous once something
ran for a week. Crops were **never reaped at all** — the frame `Reaper` only
ever touched the `frames` table — so one JPEG per object accumulated forever.
And `events`, the highest-volume table, had no cap.

Both now have budgets, `0` meaning unlimited exactly as `frames.retention`
already did, so there is one retention idiom rather than two. Nothing is deleted
that an operator did not ask to have deleted.

The crop policy is **keep what an incident references, reap the rest**. A crop
pinned by a retained incident survives until that incident is itself reaped, so
a delivered `image_url` stays valid for exactly as long as its incident. Two
consequences are built for deliberately: the byte-targeted pass *skips* pinned
crops and keeps going rather than stopping at the first one; and because
incident retention is unlimited by default, incident volume sets a disk floor
the crop budget cannot reclaim — so the dashboard reports pinned bytes
separately from reclaimable ones, to make that floor visible before it is a full
disk.

## Setup and usage

Python 3.10+. A virtual environment is required on macOS with Homebrew Python (PEP 668):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
python report.py
python query.py --list
```

That gives you detection, tracking, counting, lanes, colour, congestion **and
plates** — `plate.ocr_backend` defaults to `fast_plate`, which
`requirements.txt` installs, so plate reading and plate-keyed re-identification
work on a clean clone. On `samples/short/indian_road.mp4` that reads 8 plates
across 195 tracks and binds one vehicle.

For better characters (CER 0.19 vs 0.54), install `paddle_anpr` per the section
above and switch to it:

```bash
python main.py --ocr-backend paddle_anpr
```

Whichever you pick, the run prints which backend loaded — `[plate] OCR backend:
fast_plate` — or why it did not.

Windows (PowerShell), same idea — keep the venv **outside** the project folder
so it never gets zipped or pushed (venvs are machine-specific and gigabytes):

```powershell
py -3.12 -m venv C:\Users\Prince\venvs\traffic
C:\Users\Prince\venvs\traffic\Scripts\python.exe -m pip install -r requirements.txt
C:\Users\Prince\venvs\traffic\Scripts\python.exe main.py --max-frames 60
```

Python version rule: anything 3.10+ runs detection, tracking, lanes, color
and congestion. The `paddle_anpr` plate backend needs **3.11 or 3.12**
(`paddlepaddle` publishes no wheels for newer versions) — on 3.13+ plates
just stay empty with a warning, nothing crashes. Paddle setup (opt-in):
`pip install paddlepaddle safetensors scikit-image`, clone PaddleOCR to
`models/PaddleOCR`, weights auto-download from HuggingFace — full commands in
the ANPR section above.

All settings live in `config.yaml`; CLI flags override them.

```bash
python main.py --conf 0.4 --max-frames 200
python main.py --analyse-fps 5 --analyzers counting,lanes,anpr,congestion
python main.py --classes vehicle              # only vehicles
python main.py --classes all                  # everything (default)
python main.py --frames-format segments --max-disk-gb 200
python main.py --lanes auto                   # measure lanes from motion
python main.py --lanes explicit               # use + verify configured lanes
python main.py --lanes off                    # skip lane logic
python main.py --no-frames                    # do not store frame images
```

Lookups:

```bash
python query.py --list
python query.py --id 42
python query.py --group person
python query.py --flagged                     # wrong-way / wrong-lane only
python query.py --plate HR26AF7196 --fuzzy    # tolerates 2 OCR typos
```

Per-camera setup lives in `cameras/<name>.yaml` (zones, lane polygons, the
divider, per-camera rule tuning, backpressure), used with
`python main.py --camera shop1`. Only the keys a camera file actually declares
are applied, so a partial file cannot silently reset the rest of the config.
New clip? `python tools/calibrate.py --suggest-lanes --source other.mp4`
measures polygons and flow vectors; `python tools/draw_lanes.py` clicks exact
corners and the divider.

## Code layout

```
src/runtime/     frame sources, FPS sampling, reader thread, FrameContext
src/detectors/   YOLO wrapper + COCO class grouping
src/trackers/    per-object memory with TTL eviction:
                   store.py            per-track state, counting, eviction
                   identity.py         plate -> stable vehicle id (re-id)
src/attributes/  plate OCR (and colour/brand stubs)
src/analysis/    use cases + the registry and scheduler:
                   base.py             the Analyzer/StagedAnalyzer contract
                   stage.py            compute -> apply -> draw, optional threads
                   geometry.py         points, rings, lines. No domain.
                   lane_model.py       the wrong-side rule. No I/O.
                   lane_calibration.py learn the divider; verify a config
                   lanes.py            wrong-side detection (staged)
                   counting.py anpr.py congestion.py
src/storage/     SQLite backend, frame writers, retention:
                   sqlite_store.py     one writer thread, the outbox tables
                   frames.py           frame writers + the frame reaper
                   retention.py        crop pinning and row caps
src/query/       open-vocabulary search:
                   clip_onnx.py        both CLIP towers, loaded lazily
                   embed.py            crop eligibility + batching, SHARED by
                                       tools/embed_crops.py and the server
                   router.py           scope-vs-subject split; never rewrites
                                       the subject
                   search.py           SQL prefilter -> numpy dot-product rerank
src/timebase.py  epoch vs clip-seconds - the only reader of runs.time_base
src/journeys.py  multi-camera route assembly (offline, read-only)
src/incidents.py incident POLICY: what fires, when, and the payload
src/server/      the long-running service:
                   app.py              FastAPI endpoints + WebSocket
                   workers.py          one supervised thread per camera
                   hub.py              metadata fan-out, throttled, drops slow
                   video.py            MJPEG fan-out, same rules, skips idle
                   embedder.py         keeps the search index current, on a
                                       duty cycle so cameras keep their CPU
                   queries.py          the read side: search, browse, vocabulary
                   webhooks.py         outbox DELIVERY: retry, dead-letter
                   dashboard.py        where static/ lives, and why
                   static/             index.html + app.css + app.js, no build
src/pipeline.py  orchestrator (owns the loop, knows no individual use case)
main.py          one camera, one shot
serve.py         every camera, continuously
```

## Scope and next steps

In scope now: file / RTSP / webcam input at a configurable analysis rate; all-COCO
detection with grouping; tracking; two-zone counting; lane direction and
wrong-way flags; number plates; plate-keyed vehicle re-identification across
re-entries, runs and cameras; congestion levels; object-level SQLite storage
with every frame retained; multi-camera journey reconstruction; a long-running
service with a live dashboard, incident webhooks and disk/row retention.

Known limits, stated plainly:
- COCO cannot name potholes, debris, cones or barriers (see above).
- Track ids count sightings and over-count real objects; `identity_id`
  (plate-keyed) is the durable identity, and is NULL when no plate was read.
- Re-identification needs an EXACT plate match, so it is capped by OCR
  accuracy, not by the matching logic. On the sample clips `paddle_anpr`
  reaches CER 0.19 but exact-match 0/5, because the plates are only 21-115 px
  wide - a one-character slip is a different vehicle by design. Bigger plates
  in frame is the lever; `reid.fuzzy_distance` is not (it would merge
  `...8337` with `...8338`, two real cars).
- Wrong-side detection needs to SEE both directions to confirm a divider.
  Neither sample clip has oncoming traffic, so the two-way path is exercised
  only by `tools/test_lanes.py`, not by real footage.
- A lane whose traffic is never observed during warmup reports `no-data`, and
  its configured direction stays unverified — an absence of alerts from it
  means nothing either way.
- Multi-camera journeys inherit that exact-match cap: two cameras must produce
  character-identical voted plates to link at all, so the hop set is sparse and
  noisy rather than a dense trail. `python query.py --yield` is the measurement
  to take before reading any route. The `_twocam_*` rig above returns 3 of 5,
  but only because identical footage makes the OCR err identically; a single
  camera on the sample clips returns zero, and no real two-camera yield has
  been measured here.
- A journey asserts only observed hops. It cannot tell you that an A→D hop
  skipped B and C, because it never claims intermediate positions at all.
- The service has no application-level auth and defaults to loopback; that is a
  deployment assumption, not a feature.
- N camera threads share a GIL, so throughput per camera falls non-linearly
  with camera count. Measure it on your host; the fallback if it does not hold
  up is worker processes plus one writer process fed over a queue.
- In `auto` mode the lane is the convex hull of where warmup traffic actually
drove, so an object in a part of the road nothing used during warmup reads
as off-lane and gets no verdict. On the full Indian-clip run that is
  about half of all detection rows — most of them pedestrians and the far
  carriageway, but not all. Draw the lane by hand if you need full coverage.
- `jpeg` frame storage is ~2 TB/day at 1080p30 — use `segments` for permanent streams.
- Plate accuracy is capture-limited on the sample footage; exact match is 0/5.

Planned:
- Open-vocabulary detector behind the `Detector` interface, for real obstacles.
- Postgres `StorageBackend` (the protocol is already in `src/storage/base.py`).
- Camera GPS registry + journey-on-map across cameras, keyed on plate.
