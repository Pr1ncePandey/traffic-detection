# Traffic Detection

Analyses road traffic video from a **file, an RTSP/HTTP stream, or a webcam**.
Detects anything on the road, tracks it, reads number plates, and writes an
object-level record to SQLite alongside every stored frame.

## What it answers

- What passed through the view, of which type, and when?
- Where did a specific object go, by track id or by number plate?
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

## Identity: what a track id does and does not mean

**ByteTrack cannot re-assign a previous id to a reappearing object.** It matches
on Kalman-predicted motion and IoU overlap only (`match_thresh: 0.8`) with no
appearance model. A lost track is held for `track_buffer` frames (30 by default,
so ~1s at 30fps) and then deleted; anything that returns after that — or that was
occluded and moved meanwhile — gets a **fresh, higher id**. Ids are never recycled.

Consequences, and they are not cosmetic:

- **"Distinct track IDs" is an upper bound on real objects, not a count of them.**
  A car that leaves and re-enters is counted twice. The reports label this
  honestly; earlier versions called it "unique vehicles", which was wrong.
- **The number plate is the only durable identity.** It is stored once per object
  in the `attributes` table, which is why `query.py --plate` searches across runs
  and cameras while `--id` is scoped to one object row.
- If short-occlusion recovery matters, `tracker.name: botsort.yaml` with
  `with_reid: True` adds appearance re-association. It is a config change, not a
  code change — but it still will not give identity across long absences.

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
python main.py --source samples/input.mp4          # file
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
| `objects` | **tracked object** | class, group, first/last seen, best conf, crop path, lane |
| `detections` | object per frame | box, conf, crossing event, lane flag |
| `attributes` | object + key | **tall**: `('plate_number', 'HR26AF7196', 0.94)` |
| `events` | notable moment | crossing, wrong_way, plate_read, congestion |

`attributes` is deliberately tall rather than wide columns: adding colour, brand
or speed later needs no migration, and the `(key, value)` index keeps plate
lookup fast.

```sql
-- what was detected, by group
SELECT cls_group, cls_name, COUNT(*) FROM objects GROUP BY 1,2 ORDER BY 3 DESC;
-- plates, best read per object
SELECT object_id, value, conf FROM attributes WHERE key='plate_number';
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
pair of polygons hand-tuned on `samples/input.mp4` and inherited unchecked by
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
| `paddle_anpr` **(default)** | see below | **0.19** | 50 | PP-OCRv5 finetuned on 558k Indian plates; reads two-row (bike) plates. |
| `fast_plate` | in `requirements.txt` | 0.54 | 8 | Plate-specialised CCT model, ~5MB ONNX. |
| `rapidocr` | in `requirements.txt` | 0.88 | 370 | Generic scene-text OCR, **not** plate-specialised. Fallback only. |

CER = character error rate, lower is better, measured on `samples/plate_gt.csv`
(5 hand-labelled plates; the crops in `samples/plates/` were cut from the
earlier Indian clip, since replaced by `samples/indian_road.mp4`). Only this
column is comparable — the vendors' published figures come from different datasets. n=5 is
thin: treat the ordering as solid and the values as indicative.

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
| `outputs/traffic.db` | **the record** — objects, detections, attributes, events, frames |
| `outputs/frames/raw/` | every analysed frame, untouched |
| `outputs/frames/annotated/` | every analysed frame, with boxes and overlays |
| `outputs/crops/object_<id>.jpg` | one image per detected object |
| `outputs/annotated.mp4` | annotated video |
| `outputs/tracks.csv` | frame-level CSV export (kept for compatibility) |
| `outputs/summary.txt` | text summary |
| `outputs/report.html` | HTML report, one card per object |

## Setup and usage

Python 3.10+. A virtual environment is required on macOS with Homebrew Python (PEP 668):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
python report.py
python query.py --list
```

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
src/trackers/    per-object memory with TTL eviction
src/attributes/  plate OCR (and colour/brand stubs)
src/analysis/    use cases + the registry and scheduler:
                   base.py             the Analyzer/StagedAnalyzer contract
                   stage.py            compute -> apply -> draw, optional threads
                   geometry.py         points, rings, lines. No domain.
                   lane_model.py       the wrong-side rule. No I/O.
                   lane_calibration.py learn the divider; verify a config
                   lanes.py            wrong-side detection (staged)
                   counting.py anpr.py congestion.py
src/storage/     SQLite backend, frame writers, retention reaper
src/pipeline.py  orchestrator (owns the loop, knows no individual use case)
```

## Scope and next steps

In scope now: file / RTSP / webcam input at a configurable analysis rate; all-COCO
detection with grouping; tracking; two-zone counting; lane direction and
wrong-way flags; number plates; congestion levels; object-level SQLite storage
with every frame retained.

Known limits, stated plainly:
- COCO cannot name potholes, debris, cones or barriers (see above).
- Track ids over-count real objects; the plate is the durable identity.
- Wrong-side detection needs to SEE both directions to confirm a divider.
  Neither sample clip has oncoming traffic, so the two-way path is exercised
  only by `tools/test_lanes.py`, not by real footage.
- A lane whose traffic is never observed during warmup reports `no-data`, and
  its configured direction stays unverified — an absence of alerts from it
  means nothing either way.
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
