# Heatmaps — analysis and decision log, 2026-09-15

The brother asked for a "heat imaging" feature. This file records what that
can mean, how each option works, where it is useful, and what was built first
(the MVP) and why. Branch `heatmap`.

## Two different things called "heat imaging"

| | Heatmap (activity map) | Thermal (infrared) imaging |
|---|---|---|
| What it is | a coloured map of WHERE objects are most, built from detections | a camera that senses heat radiation per pixel |
| Hardware | none - any existing camera | a thermal / bi-spectrum camera (~Rs 40k to several lakh) |
| Model | none | a detector retrained on thermal footage (FLIR ADAS, LLVIP, KAIST) |
| Fits the project | reuses tracks, ground points, the drawing stage | video input works (RTSP); YOLO, plates, face recognition, colour do not |

**Decision: heatmaps first.** No hardware, no model, cheap, and it uses what
the pipeline already produces. Thermal is a separate project once a thermal
camera exists. Note: ordinary RGB video cannot be turned into real heat data -
"RGB to thermal" models invent the temperatures.

### Thermal, for later

- LWIR cameras see in darkness, smoke, fog and headlight glare; people,
  animals, engines and tyres stand out. Radiometric models also give a
  temperature per pixel (through the vendor SDK, not plain RTSP).
- Works on thermal: detection (after retraining), counting, lanes, wrong-way,
  congestion, heatmaps. Does not: number plates, face recognition, colour.
- Best setup: a bi-spectrum camera (RGB + thermal) - plates and faces from RGB,
  night/fog detection from thermal.
- Uses: people and animals on highways at night or in fog, all-weather counting,
  vehicle fire / overheating early warning (radiometric), tunnels, toll plazas.

## How a heatmap works

1. Split the frame into a grid (e.g. 96 x 54 cells).
2. For every detected person / vehicle, add to the cell under its GROUND POINT
   (bottom centre of the box - where it stands, which is what "where people
   gather" means; the box centre would put a standing person's heat at chest
   height).
3. Over many frames busy cells grow, quiet ones stay near zero. A place where
   people stand for a long time gets hot because it is counted every frame.
4. Smooth the grid, scale it to the busiest area, colour it (blue -> red) and
   blend it over the video.

Kinds of heatmap (same data, counted differently) - only the first is in the MVP:

| Kind | Counts | Answers |
|---|---|---|
| **Presence** (MVP) | detections per frame | where is it busiest / where do people gather |
| Unique visits | each track once per cell | how many different people passed here |
| Dwell | seconds spent per cell | where do things stop - queues, crowding, parking |
| Speed | mean movement per cell | where does traffic crawl |
| Events | wrong-way, congestion, face matches | where do incidents cluster |

## Use cases by sector (analysis features, NOT in the MVP)

**Roads and traffic:** busiest stretches, frequent stopping (illegal parking),
queue build-up at junctions, pedestrians crossing outside crossings, incident
hot spots, rush hour vs off-peak, before/after a road change.

**Retail and malls:** busy vs dead aisles and displays (product placement);
dwell vs walk-past (display interest, promotion effect); passers-by vs
entrants (conversion); billing queues (open another counter); floor,
corridor and entrance footfall (rent pricing, kiosk and ad placement);
common customer paths; trial-room use.

**Restaurants, cafes, food courts:** table use (tables that fill first / stay
empty); time until a customer is attended; counter and takeaway queues;
kitchen and service path collisions (layout); popular stalls and seating;
peak hours per zone (staff planning).

**Factories and warehouses:** workstation / machine presence and idle
stations; restricted and danger zones (presses, robots, forklift lanes) as
alerts; people-forklift near-miss spots; congested aisles and loading bays;
most-visited racks (move fast movers near dispatch); dock waiting; PPE per zone
(needs a PPE model); layout planning.

**Hospitals and clinics:** OPD, emergency and pharmacy waiting crowding;
patient flow bottlenecks (registration -> waiting -> doctor -> pharmacy);
staff movement and ward visits; restricted areas (ICU, OT, drug store) after
hours; isolation-area crowding and hand-wash station use; ambulance bay kept
clear; visiting-hour crowding.

**Offices / co-working:** desk and meeting-room use (booked but empty);
pantry and common-area use; lobby and lift morning peaks; energy saving by
zone presence.

**Banks, government offices, railway / metro stations, airports:** counter and
token queues; platform crowding near doors and stairs with an overcrowding
alert; security and check-in queues; gates filling early; ticket-window peaks.

**Education:** corridor and canteen crowding between classes; library seat
use; out-of-bounds areas after hours; gate crowding at dismissal.

**Temples, events, stadiums, public places:** crowd density above a safe
people-per-area limit (stampede prevention at melas, temples, concerts);
entry/exit gate load; popular zones in parks and tourist spots.

**Hotels and gyms:** lobby, check-in, pool and restaurant use by hour; machine
and area use, rush hours, unused equipment.

**Parking lots and fuel stations:** occupied vs free slots, long stays, wrong
parking; nozzle queues, waiting time, idle pumps.

Sectors to target first when the analysis layer comes: retail/malls (clearest
money value), factories/warehouses (safety zones sell), hospitals (queues,
anonymous by design), restaurants (simple and quick to show).

## What the analysis layer will need later (not built)

| Needed | Status |
|---|---|
| Named zones per camera (counter, aisle, machine) | lanes are already per-camera polygons; zones can reuse that |
| Dwell / wait time per person per zone | tracking already gives ids with first/last seen |
| Entry/exit counts through a door | line counting exists (vehicles-only today; needs people) |
| Excluding staff | uniform colour attribute, a staff zone, or enrolled-staff face recognition with consent |
| Zone alerts (restricted area, overcrowding) | incident + webhook system exists; new kinds needed |
| Hour / day / week comparisons | detections are timestamped; needs report/API views |
| Floor-plan view across cameras | new: per-camera homography (4 floor points) |

Limits to design for: ceiling/fisheye cameras (YOLO trained on street views
misses top-down people - fine-tune or pick an overhead model); dense crowds
(boxes occlude, counts run low - density estimation is better there);
perspective (a far cell covers more floor than a near one - weight by box size
or use a homography); privacy (heatmaps can be fully anonymous - counts per cell,
no faces - and should stay separate from face recognition; DPDP Act 2023).

## MVP - what is built

Scope agreed with the user: **just the heatmap feature**, like congestion - an
analysis switched on when wanted, shown on the video, marking where people and
vehicles gather. No zones, no "where do people go" analysis, no dashboard view
yet.

**Design**
- `src/analysis/heatmap.py`, an analysis plugin (same contract as congestion):
  `analyses.heatmap.enabled: false` fleet-wide; a camera file turns it on, or
  `python main.py --enable heatmap` for one run.
- Presence counting at each box's ground point, one grid per group (people,
  vehicles by default) so the video can show people, vehicles or both.
- Grid resolution `grid_w` (96 columns; rows follow the aspect ratio). Cheap:
  one small array addition per frame, whatever the video resolution.
- Rendering: Gaussian smoothing on the small grid, scaled to its 99th
  percentile, JET colours, blended only where the heat is above `min_level` -
  cold areas keep the original picture.
- **Log scale by default** (`scale: log`). Found while testing: counts are
  heavy-tailed - a parked car or a queue is counted every frame, a passer-by a
  few times. On a linear scale a spot seen once next to one seen 50 times is
  2% of the colour range and does not show at all; on log it is ~18%.
  `scale: linear` is kept for when the busiest spot really is all that matters. Re-rendered every `redraw_every` frames; blending the
  cached layer each frame is a single uint8 blend.
- `half_life_s`: 0 = accumulate over the whole run (right for a clip);
  e.g. 120 = old activity fades, showing the last few minutes (right for a
  live camera).
- Drawn in the analysis draw phase and declared first under `analyses:`, so it
  sits UNDER the counting lines, lane overlays and object boxes.
- At the end of a run, `outputs/<camera>/heatmap/` gets `heatmap_all.png`,
  one PNG per group (over a recent frame of the scene) and `heatmap_grid.npz`
  (the raw counts, for the later analysis layer).
- No storage table, no pipeline change, no new dependency.

**Files:** `src/analysis/heatmap.py`, `tools/test_heatmap.py` (32 checks),
`analyses.heatmap` in `src/config.py` and `config.yaml` (+ `dir` expanded per
camera), registered in `src/analysis/base.py`, a README section.

## How to use

```bash
python main.py --camera demo --enable heatmap
python main.py --camera person_test --source samples/short/person_test.mp4 --enable heatmap
```

Or per camera: `analyses: {heatmap: {enabled: true}}` in `cameras/<id>.yaml`.
Useful knobs: `show: person` / `show: vehicle`, `half_life_s: 120` for live
cameras, `alpha`, `min_level`, `groups: [person, vehicle, animal]`.
Output video: `video.target`; maps: `outputs/<camera>/heatmap/`.

## Tests and field check

- `tools/test_heatmap.py` 32/32: heat at the ground point, groups kept apart,
  unlisted groups ignored, edge clamping, half-life fading, cold areas left
  untouched, layer reuse between redraws, show/colormap/scale fallbacks, log
  vs linear, saving (files, raw counts, once only, nothing for an empty run),
  wiring (off by default, declared first, per-camera dir, builds when enabled).
- All other suites unchanged: stage 30, lanes 52, congestion 12, incidents 72,
  faces 96, reid 39, webhooks 48, api 57. (`heatmap` is declared first under
  `analyses`, so the order change was checked against the stage tests.)
- CLIP background embeddings stayed OFF (`embed_in_background: false`) for
  every run.

**Real clips, 600 frames each (`--no-frames`, heatmap on):**

| Clip | Counted | Speed | Heatmap cost |
|---|---|---|---|
| `samples/short/input.mp4` (road, 1280x720) | 3,102 vehicle, 234 person detections | 9.7 fps | analysis compute 0.95 ms/frame incl. lanes |
| `samples/short/person_test.mp4` (street, 1920x1080) | 3,565 person, 1,897 vehicle | 6.8 fps | analysis compute 1.58 ms/frame |

Compute plus drawing, measured alone on 1080p with 40 boxes: ~19 ms per
frame, almost all of it the colour blend onto the full frame. Small next to
YOLO (~100 ms/frame on this CPU).

**First look found a rendering bug.** The first saved maps were almost
entirely red. The log was taken AFTER smoothing, relative to the smallest
smoothed value - a blur tail of ~1e-4 - so every visited cell looked like a
hot spot. Fixed by taking the log of the raw counts first, then smoothing and
scaling to the 99th percentile. Checked again by eye on both clips:

- road: the used lanes are coloured, red where cars queue at the far end;
  the empty far carriageway stays clear. People: the person at the bus stop is
  the hot spot, plus a faint streak along the bus lane - people detected
  through the bus windows, a detector effect, not a heatmap one.
- street: people are hottest along the main walking diagonal and near the
  shop fronts; vehicles are hottest where the taxi and cars stood parked; the
  quiet paving keeps its normal picture.
- the heat is drawn under the boxes, lines and labels in the output video.

## Not done yet (next steps, in order)

1. Dashboard toggle: the dashboard shows the clean stream and draws boxes in
   the browser, so the burnt-in heatmap does not appear there yet. Needs a
   `/heatmap/{camera}.png` endpoint (the live grid) and a canvas layer.
2. Periodic snapshots to the database for hour/day comparisons.
3. Then the analysis layer: zones, dwell, unique visits, paths - see the
   sector use cases above.
4. Perspective weighting (far cells cover more ground) and a person-detection
   confidence floor to cut false people (e.g. through bus windows).
