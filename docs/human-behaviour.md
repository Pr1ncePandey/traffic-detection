# Human behaviour — tier 1: zones, intrusion, loitering, crowd, running

Analysis log for `src/analysis/behaviour.py`. What was built, why each rule is
shaped the way it is, how it was tested, and what it cannot do.

## Scope decision

Three tiers were proposed:

| Tier | Behaviours | Needs |
|---|---|---|
| **1. Rules on tracks** | zones, intrusion, loitering, crowding, running | nothing new |
| 2. Pose rules | falls, lying down, hands up | a keypoint model |
| 3. Action recognition | fights, snatching, smoking, phone use | clip/skeleton models, heavy on CPU |

**Only tier 1 is wanted.** So this work adds no model, no dependency and no
measurable CPU: every rule is a few float operations per person per frame over
the tracks the pipeline already produces. It matters on the brother's PC,
where CPU is the constraint (the same reason search embedding stays off).

## How it fits the code

Nothing in `pipeline.py` changed. That is the analysis seam doing its job
(`src/analysis/base.py`).

| Piece | Where | What |
|---|---|---|
| Analysis | `src/analysis/behaviour.py` | zones + four rules, staged (`compute`/`apply`/`draw`), `concurrent = True` |
| Registration | `src/analysis/base.py` | one import in `_load_builtins` |
| Defaults | `src/config.py` → `analyses.behaviour` | OFF fleet-wide; a camera turns it on |
| Incidents | `src/incidents.py` | new kinds fire at detection; `zone_exit` is never an incident |
| Dashboard | `src/server/static/app.js` | new kinds in the incident filter |
| Demo camera | `cameras/behaviour_demo.yaml` | three zones on `samples/short/person_test.mp4` |
| Tests | `tools/test_behaviour.py` | 80 checks, fabricated boxes, no video |
| Tuning | `tools/replay_behaviour.py` | re-runs the rules over a finished run's stored boxes in seconds, with contact sheets |

### Configuration

```yaml
analyses:
  behaviour:
    enabled: true
    exclude:                      # a sign pole the detector calls "person"
      - [[0.02, 0.32], [0.08, 0.32], [0.08, 0.39], [0.02, 0.39]]
    zones:
      - name: "road"
        polygon: [[0.0, 0.22], [0.72, 0.16], [0.72, 0.24], [0.0, 0.56]]
        restricted: true          # intrusion
      - name: "shopfront"
        polygon: [[0.74, 0.28], [0.93, 0.28], [0.93, 0.75], [0.80, 0.75]]
        loitering_s: 120          # loitering
      - name: "pavement"
        polygon: [...]
        crowd_people: 8           # crowd
      - name: "anywhere"          # no polygon = the whole frame
        loitering_s: 600
```

A zone with no rule still counts entries, occupancy and dwell time. Polygons
are ratios (0–1) or pixels, the same convention and helpers as `lanes`.
Fleet-wide knobs (`intrusion_s`, `exit_grace_s`, `crowd_hold_s`, riders,
`running.*`) are in `config.py` with a comment each.

## The rules, and why each is shaped like this

### Position is the ground point

"Inside the zone" means **the feet are inside**: the bottom-centre of the box,
the same point the heatmap and lanes use. A person on the pavement whose tall
box overlaps the road is on the pavement. Tested: box middle over the zone with
feet outside does not fire.

### Two clocks against edge wobble

A box's bottom edge moves a few pixels every frame even for a person standing
still. A person walking along a zone edge is therefore in, out, in, out. Two
separate mechanisms stop that from producing garbage:

- **Intrusion counts time actually inside** (`inside_s`, summed per frame), not
  time since first entry. Ten wobbles in and out do not add up to an intrusion
  until they total `intrusion_s` (default 1 s) inside.
- **Leaving needs `exit_grace_s` outside** (default 2 s). A wobble, or a
  one-frame detector miss, does not end a dwell, so the loitering clock does
  not restart and no false `zone_exit` is written.

Both are tested with a person alternating 10 px either side of an edge every
frame.

### Intrusion

A tracked person inside a `restricted` zone for `intrusion_s`. Fires **once per
track per zone**. A person who steps out and back in on the same track does
not fire twice. The use cases are jaywalking on a carriageway, a railway track,
a closed site, or a staff-only door.

### Loitering

A tracked person inside a zone for that zone's `loitering_s`, measured from
entry, with the grace rule above. Once per track per zone. The label on the
video keeps counting (`LOITERING atm 73s`).

**Honest limit.** Loitering is limited by the tracker, not by this rule. If
ByteTrack loses a person (occlusion by a bus, an umbrella, a crowd) for longer
than its buffer, they come back as a new track and the clock starts again. So
loitering is **under**-reported in busy scenes, never invented. Joining tracks
back together is appearance re-id, which is not part of tier 1.

### Crowd

A zone is crowded when `crowd_people` or more people are inside it in
`crowd_fraction` (0.8) of the frames of the last `crowd_hold_s` (5 s). It is
reported on **both edges**, `crowded` and then `clear`, and clear uses the same
test the other way (80% of the window under the threshold). As with
congestion, a count sitting on its threshold must not flip every frame.

**Why a fraction and not an unbroken hold.** The first version needed the count
over the threshold on *every* frame for 5 s, copied from congestion. On the
demo clip it **never fired**. In 4-second spans averaging 5.0 people, the
longest unbroken run at 5+ was **2.0 s**: the detector drops one person
(behind an umbrella, two boxes merged) for a frame every second or so.

Tested both ways:
- a crowd with a detector miss every 6th frame still counts;
- a count over the threshold only half the time does not;
- one missed frame does not clear a crowd.

Occupancy counts every person box in the zone, **including untracked ones**
(ByteTrack withholds ids from low-confidence boxes, and they are still
people). Dwell and the per-person rules need a track id.

### Running

Speed over a `window_s` (1 s) window, in **body heights per second**:

    speed = ground-point displacement / median box height / elapsed time

Why body heights and not pixels: pixels per second means nothing without
calibrating the floor, because a walker near the camera covers more pixels
than a sprinter far away. A person is ~1.65 m tall, so dividing by their own
box height makes the number roughly independent of depth with **no
calibration**. Reference points:

| Gait | m/s | body heights/s |
|---|---|---|
| walking | 1.2–1.5 | 0.7–0.9 |
| brisk walk | 1.8 | ~1.1 |
| jogging | 2.5–3 | 1.5–1.8 |
| running | 3.5+ | 2.1+ |

The threshold `min_speed_hps` is 1.6, held for `min_s` (1 s) after a full
window. A person must keep it up for about 2 s before one `running` event.

Guards, each tested:

- **Tracker swap**: a step faster than `max_jump_hps` (6 h/s, about 10 m/s)
  between samples clears the history. Two people swapping ids across the frame
  is not a sprint.
- **Clipped box**: a box touching the top or bottom frame edge has the wrong
  height and the wrong feet, so the sample is skipped.
- **Tiny box**: under `min_height_px` (40 px), a few pixels of jitter is a big
  speed. Skipped.

**Honest limit.** It is a 2-D projection. Running straight towards or away from
the camera barely moves the ground point, so that case is **missed**. The
error is always on the side of silence.

`running` incidents are **off by default** (children, joggers, someone running
for a bus). The events are still stored and shown on the video.

### Exclude masks

`exclude:` is a list of polygons. A detected "person" whose feet are inside
one is ignored by **every** rule, including occupancy.

They exist because on the demo clip the **detector**, not the rules, produced
6 of the first 37 intrusions. A no-parking sign pole (4 tracks) and a blue
arrow bollard (2) were boxed as `person` at up to **0.61** confidence. Real
people in the same clip go as low as 0.44, so no confidence floor can separate
them. A mask on the spot can, and that is how CCTV products handle it.

- **Size a mask to the object's measured feet jitter.** The bollard's feet
  moved over y 0.231–0.239 across 102 frames. The first mask stopped at 0.235,
  and the bollard still fired once when its box drifted out.
- **Cost.** A real person standing exactly on a mask is ignored there too. One
  real crossing (track 62, walking past the bollard) was lost to the widened
  mask.

### Riders and passengers

On Indian roads most fast "people" are on two-wheelers, and YOLO boxes the
rider as `person`. Without a guard every motorcyclist would be a runner and a
jaywalker. There are two tests, because two kinds of vehicle hide a person
differently:

| Vehicle | Person ignored when | Why |
|---|---|---|
| two-wheeler (`bicycle`, `motorcycle`) | feet inside its box **and** ≥ `rider_overlap` (0.3) of the person covered | a bike covers about the lower half of its rider (a fixture rider measured 0.73) |
| enclosed (`car`, `bus`, `truck`, `train`) | ≥ `passenger_overlap` (0.9) of the person inside its box | a passenger seen through a window is almost wholly inside |

The feet test is what keeps a pedestrian walking beside a bike: their feet are
not inside the bike's box.

**Found on real footage.** The first version applied the 0.3 rule to **every**
vehicle. On the demo clip it removed 23 tracks, and the contact sheet showed
that **all 23 were pedestrians** walking behind the parked taxi and cars. Their
legs were hidden, so their boxes ended inside the car's box. There is not one
motorbike in the clip.

The enclosed-vehicle test is now 90%, because a pedestrian behind a car shows
head and shoulders above the roof. Tested at 45% and at 73% covered; both still
count. After the fix, 2 tracks (9 boxes) remain: a person partly behind a
parked bicycle at the shop, which is ambiguous.

## Events and incidents

Events written to the `events` table (`detail_json`):

| kind | scope | detail |
|---|---|---|
| `intrusion` | track | zone, dwell_s, group, cls_name, bbox |
| `loitering` | track | zone, dwell_s, loitering_s, … |
| `running` | track | speed_hps, zone (first), zones, … |
| `crowd` | frame | zone, state, people, crowd_people, over_fraction, since, held_s |
| `zone_exit` | track | zone, dwell_s, intrusion, loitering |

`zone_exit` is the dwell-time record: one row per person per zone. It is in
`NEVER`, so no config can make it an incident.

**Incidents fire at detection, not at track retirement.** The wrong-way
incident waits for retirement because its payload wants the voted plate. A
behaviour alert is worthless once the person has left, and there is no plate
to wait for. There is also no second dwell in the policy: the analysis already
held each rule and emits it once, so another gate would only add latency.

Payload differences from a vehicle incident:

```json
{"incident_id": "behaviour_demo-412-road-intrusion",
 "kind": "intrusion", "state": "ongoing", "zone": "road",
 "subject": {"group": "person", "cls": "person"},
 "sighting": {"object_id": 412},
 "detail": {"zone": "road", "dwell_s": 1.0, "cls_name": "person", "bbox": [..]},
 "image_url": "https://host/crops/412.jpg"}
```

- The id carries the zone, so one person in two restricted zones is two
  incidents and a retry of either is deduplicated by storage (`INSERT OR
  IGNORE`).
- `crowd` ids are `camera-zone-t<onset>-crowd`, with no subject block and no
  image, like congestion.
- **No identity.** The payload says what someone did, never who they are. Face
  recognition is a separate, consented feature and stays separate.
- `image_url` limitation: the `objects` row, and with it `/crops/{id}.jpg`, is
  written when the track retires, so the URL can 404 while the person is still
  in view. A consumer should retry. A per-incident snapshot endpoint would
  remove this; not done.

Defaults in `incidents.kinds`: `intrusion`, `loitering` and `crowd` on,
`running` off.

## Testing

### Unit tests: `python tools/test_behaviour.py`, 80 passed

Fabricated boxes on a 1280×720 frame at 10 fps. It covers:
- **Zones**: ratio→pixel scaling, whole-frame zones, bad and duplicate zones.
- **Intrusion**: the hold time, firing once, feet versus box middle, edge
  wobble, unrestricted zones.
- **Leaving**: grace, disappearance, dwell stats, no re-fire on return,
  `exit_events: false`.
- **Loitering**: the threshold, a short gap not restarting the clock, the label.
- **Crowd**: hold, a one-frame miss, the clear hold, summary seconds.
- **Riders**: motorcycle rider, pedestrian beside the bike, bus passenger,
  `ignore_riders: false`.
- **Running**: walking, running, short bursts, far/small people, tracker
  swaps, clipped boxes, tiny boxes, disabled, speed percentiles.
- **Lifecycle**: clip restart, `forget`, stale sweep.
- **Drawing**: zone outline, red intruder box, untouched areas, `draw: false`.
- **Incidents**: immediate firing, ids, payload blocks, running off by
  default, crowd, `zone_exit` never.
- **Wiring**: off by default, the demo camera, `embed_in_background` still
  false.

Later sections added masks, the passenger rule, and the crowd fraction, each
with tests. Two bugs were caught by these tests before any video run:

1. `Findings.set(tid, extra=...)` **replaces** the field. Tagging a person with
   both their zones and their behaviour in two calls kept only the second.
   Fixed by building one `extra` dict.
2. The analyzer's own fallback for `riders_in` was an empty list, so any config
   not built through `config.py` (a test, a tool) silently stopped ignoring
   riders. The fallback is now the same list as the config default.

### Regression: existing suites

All still pass: incidents 73, webhooks 48, heatmap 32, congestion 12, stage
30, lanes 52, faces 96, api 57.

`test_api` first failed with `no such table: identities`. It picked up the
stale `outputs/api_test.db`, written on 13 Sep before the v3 rename. That is
not caused by this change: rebuilt with the command in the test's docstring
(into the scratchpad), it passes 57/57. The stale file is still there and
should be regenerated.

### Real footage: `samples/short/person_test.mp4`

The clip is 63 s at 1920×1080, 25 fps: a rainy pedestrian street with a road,
market stalls and shopfronts. It is a downloaded public clip, so this looks only
at behaviour and identifies nobody.

```bash
python main.py --camera behaviour_demo --db outputs/behaviour_demo/behaviour.db --no-frames
```

The run took 432 s on CPU (3.7 fps; YOLO is the cost) and produced 441 tracks,
230 of them people. **The behaviour analysis itself cost 0.56 ms per frame.**

**Every event was checked by eye** on a contact sheet (a crop of the box at the
event time), not just counted.

#### The tuning loop

`tools/replay_behaviour.py` feeds the boxes a run stored back through the rules
with the current camera config. It was checked first: replaying with the run's
own config reproduced the **identical 37 intrusions on the identical track
ids**. Each try then took seconds, not a 7-minute YOLO run.

| Step | Change | Intrusions | By eye |
|---|---|---|---|
| 1 | zones drawn by eye on a 10% grid | 37 | 25 real, 6 detector false positives (sign pole ×4, bollard ×2), 6 shoppers at market stalls (road polygon too big) |
| 2 | + exclude masks on the pole and bollard | 32 | pole gone; the bollard fired once more (mask too tight) |
| 3 | + bollard mask sized to its measured jitter; passenger rule | 30 | bollard gone; lost real track 62 walking past it |
| 4 | road edge moved to the stall kerb | **23** | all 6 stall shoppers gone; lost real track 305 on the new edge |

**Final: 23 intrusions, all 23 real by eye (0 false). 2 of the 25 real
crossings were lost** to the mask and the new edge. Track 305 walked alongside
track 310, which still fired.

**Caveat, stated plainly:** the zones were tuned on the same clip they are
scored on. So this shows the rules do what they claim once the zones are right.
It does not show how a new camera behaves before a review. The practical lesson
is that **zone placement and masks are most of the accuracy**. Plan one
review-and-adjust pass per camera: run, look at the contact sheets, replay.

#### The other rules on the same clip

- **Loitering (20 s demo threshold): 1 event, correct.** It is track 156, a man
  standing at the shopfront. His 23.2 s is the longest dwell in that zone, and
  nobody else reached 20 s.
- **Running: 0 events, correct.** Nobody runs in the clip. Speeds over 8,234
  samples were p50 0.4, p90 0.8 and **p99 1.3 body heights/s**. The 1.6
  threshold sits above the 99th percentile of a normal street, as the gait
  table predicts.
- **Riders: 2 tracks (9 boxes) after the fix**, down from 23 wrongly ignored
  pedestrians (see Riders and passengers).
- **Crowd (demo threshold 4 on the pavement): 3 events after the fraction fix,
  none before.**
  - crowded at 15.9 s: the pavement averaged 4.1–4.2 people in 4–16 s;
  - clear at 20.2 s: it averaged 0.3 in 20–24 s;
  - crowded again at 49.6 s until the end: it averaged 3.8–5.0, peaking at 8.
  
  At the first threshold of 5, no 5 s window had more than 67% of its frames
  at 5+, so a threshold of 5 correctly never fires on this pavement.

Measured pavement occupancy (the input to the crowd rule):

| Seconds | 0–4 | 4–8 | 8–12 | 12–16 | 16–20 | 20–24 | 24–44 | 44–48 | 48–52 | 52–56 | 56–60 | 60–63 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mean people | 2.1 | 4.2 | 2.9 | 4.1 | 2.3 | 0.3 | 1.4–3.4 | 3.8 | 5.0 | 4.3 | 3.9 | 5.0 |
| Max | 4 | 6 | 4 | 5 | 4 | 1 | 3–5 | 6 | 6 | 6 | 6 | 8 |

**The replay agrees with a real run.** The tuned camera was also run end to end
through `main.py` (video, YOLO, tracker; `behaviour_v2.db`). It produced the
same result: 23 intrusions and 1 loitering, with identical zone statistics
(road 73 entries, mean dwell 1.4 s, max 8.0 s; shopfront 49 entries, max dwell
23.2 s), at 0.59 ms per frame for the analysis. That run started just before
the crowd-fraction fix, so its crowd figures come from the old rule; the crowd
results above are from the replay.

**Overlay bug found in the video.** In the annotated video the `INTRUSION` and
`LOITERING` labels were barely visible. The pipeline draws each object's own
box and label *after* the analyses, on the same box and at the same spot, and
painted over them. The flag box is now drawn 4 px outside the object box and
its label above the pipeline's, so no pipeline change was needed. Checked on a
60-frame real run: track 7's red box sits outside its cyan box and
`INTRUSION road` reads clearly above the pipeline label.

**Incidents on these runs.** `main.py` stores events but does not run the
incident policy; `serve.py` does. So incident firing (ids, payloads, defaults)
is covered by the unit tests, not by these runs.

## Not done (next steps)

1. **A snapshot per incident**, so `image_url` works while the person is still
   in view. Today it points at the crop, which exists only after retirement.
2. **Zones on the dashboard.** The dashboard draws boxes in the browser, so zone
   outlines appear only in the annotated video. A zone editor (click the
   corners on a frame) would replace drawing by eye on a grid, which was the
   slowest part of this work.
3. **Time windows per zone**, for example restricted only 22:00–06:00. Live
   sources are already on wall-clock time; file sources need `recorded_at`.
4. **Loitering across occlusions.** Join a person's broken tracks back together
   (appearance re-id), so hiding behind a bus does not restart the clock.
5. **Real metres per second** from a floor homography (4 clicked ground points),
   if running ever needs more than body heights.
6. **Housekeeping.** `outputs/api_test.db` predates the v3 rename and makes
   `tools/test_api.py` crash. Regenerate it with the command in that test's
   docstring.
