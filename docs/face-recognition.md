# Face recognition — merge log and decision record, 2026-09-15

The standalone prototype `face-lab/` (YuNet + AdaFace IR101; its full model
research, benchmark and refactor log are in `face-lab/ANALYSIS.md`) merged into
the main project, following the plan agreed with the brother:

- cameras in config, each camera choosing its analyses — face recognition is
  one of them;
- a database of people: a name and the path to their face photo(s);
- a confirmed match triggers the incident webhook;
- any number of people, in real time.

Branch `face-recognition`, local only — **not pushed**. Background CLIP
embeddings (`server.search.embed_in_background`) stayed **off** for every run
here, as the brother asked; `/health` showed the embedder `disabled`.

## What was added

| Where | What |
|---|---|
| `src/faces/` | the library: `detector.py`, `align.py`, `embedder.py` (copied from face-lab unchanged apart from messages), `errors.py`, `gallery.py`, `voting.py`, `people.py` |
| `src/attributes/enrichers/face.py` | the per-camera `face` enricher |
| `src/incidents.py` | `face_match` incident kind: fires at confirmation, `person` block, per-person cooldown |
| `src/server/app.py` | `GET/POST /people`, `POST /people/{name}/enabled`, `DELETE /people/{name}`, `GET /faces/{object_id}.jpg` |
| `src/config.py`, `config.yaml` | `perception.attributes.face` block (off by default), `incidents.kinds.face_match`, `incidents.face_match_cooldown_s` |
| `src/pipeline.py` | box caption shows `name score` |
| `src/runtime/plugin.py` | `PluginStage.close()` also closes plugins that own a thread |
| `src/attributes/registry.py` | `face` is an OPTIONAL enricher: imported only when a camera enables it |
| `cameras/face_demo.yaml` | a test camera with face on and road analyses off |
| `tools/people.py` | `list / add / import / enable / disable / remove / enrol` |
| `tools/fetch_face_models.py` | YuNet download + AdaFace ONNX export into `models/faces/` |
| `tools/test_faces.py` | 96 checks, no models or video needed |
| `tools/dummy_receiver.py` | accepts the `face_match` payload shape (`person` block instead of `vehicle`) |
| `.gitignore` | `models/faces/`, `people/`, `samples/faces/` |

Not merged, on purpose: face-lab's own IoU face tracker, its offline
`run.py`/report writer and its benchmark scripts. The main pipeline already
tracks, stores, writes a labelled video and a tracks CSV; a second tracker and
a second output format would be two sources of truth. face-lab stays as the
prototype and benchmark.

## Design decisions and why

**1. Faces ride on the main pipeline's ByteTrack person tracks** (instead of
face-lab's face tracker). A name then lands on the same object id as that
person's crop, clothing attributes and events — one person, one record — and
query/API/dashboard get it for free (the dashboard shows any scalar attribute,
so `face_name` appears with no frontend change). A face is given to the person
box whose upper 60% contains the face centre. Two consequences, both measured
below: faces with no person box (faces in the framed photos on the wall) are
ignored, which is right; and ByteTrack's id swaps when people cross had to be
handled (decision 5).

**2. People live in the project database, with the embedding stored.** Two
tables in the same SQLite file as everything else (`people`, `person_photos`).
The brother asked for name + photo path; the AdaFace vector is stored next to
the path with the model name, file size and mtime, because AdaFace is ~1 s per
photo and would otherwise run for every photo at every camera start. A changed
photo or model is re-embedded automatically; a photo with no usable face stores
the reason and is not retried until the file changes. Paths inside the project
are stored relative, so the database survives copying the project to the
brother's laptop. The tables are created by `src/faces/people.py` on first use
through its own short connection, not through `SqliteStore`'s writer queue:
enrolment is rare, tiny, and the caller wants its error immediately.

**3. Any number of people, re-read live.** Matching is one matrix multiply
against every enrolled photo, so cost follows faces in view, not people
enrolled. A running camera polls a cheap signature of the tables every
`reload_s` (10 s) and reloads when it changes — a person added through the API
is searched for without a restart. `people: [names]` in a camera file limits
that camera to a subset.

**4. CPU budget.** AdaFace is the cost (1.6–2.8 s per face measured in these
runs, inline). So: face detection runs only while some visible person still
needs a check; a check needs eyes ≥ 28 px apart and a sharpness ≥ 40 (the
blur gate recommended in face-lab: blurred frames of the right person scored
0.07–0.40); at most 6 checks per unnamed person, 0.3 s apart; a named person
is re-checked only every 3 s. On a live source AdaFace runs on a background
thread with a bounded queue — a full queue skips a check instead of stalling
the camera. On a file it runs inline: slower, but deterministic, and the last
checks of a clip are never lost.

**5. Tracker swaps.** When two people cross, ByteTrack can swap their ids —
and the name follows the id. Handled three ways: a named track re-checked
that now confidently reads as another enrolled person twice is re-decided;
two person boxes overlapping ≥ 30% (of the smaller box) mark a crossing, and
once apart both get quick re-checks instead of waiting 3 s; a face inside two
person boxes is ambiguous and skipped. When a track loses its name, re-checks
that already named someone else are kept as evidence. One person cannot hold
a name on two boxes in the same frame — the better score keeps it.

**6. One webhook per sighting, at confirmation.** Other incident kinds fire at
track retirement (when the plate vote is settled). A face match is already
settled when it is confirmed (2 agreeing reads), and "who just walked in" is
worth little 5+ s later, so `face_match` fires immediately. Payload: a `person`
block (name, score, votes, reads — not the vehicle block relabelled), the
reason, and `image_url` → `/faces/{object_id}.jpg` (the clearest face of that
person on that track). Id `camera-object-person-face_match`, so a swap that
puts a second name on one object is its own incident. The same person on the
same camera within 60 s (a track split by an occlusion) is stored as an event
but not re-sent; a clock going backwards (a file re-run) counts as new. A
withdrawn match (`face_unmatched`) is an event, never an incident.

**7. Off by default, per camera on.** `perception.attributes.face.enabled:
false` fleet-wide; `cameras/face_demo.yaml` turns it on, or
`main.py --enable face` for one run. A machine without the face models never
imports the code.

## Tests

`tools/test_faces.py` — 96 checks, fake detector/embedder driving the real
code: voting rules, gallery, folder import, the people database (case-
insensitive names, duplicate photos, embedding cache and invalidation, bad
photos, missing files, subsets, enable/disable), face-to-person assignment,
blur/size gates, one-place rule, swap detection, crossing re-checks, reload
without restart, background worker, snapshots, the incident policy and payload,
config wiring (face off by default, CLIP embeddings off in `config.yaml`).

Existing suites still pass: test_lanes 52, test_stage 30, test_reid 39,
test_congestion 12, test_incidents 72, test_webhooks 48, test_api 57.
API routes checked with TestClient on a copy of the database: list, add,
missing photo → 400 with the message, disable, delete, unknown → 404.

Enrolling the two consented selfies with the real models: me vs mom similarity
**0.18** — identical to face-lab, so the copied models and alignment behave the
same.

## Real footage (consented clips, local only)

### Run 1 — `main.py --camera face_demo`, before the crossing fix

`switch_test.mp4` (25 s, the user and their mom walk and swap places):
760 frames in 281 s (2.7 fps on CPU, inline AdaFace 2.8 s/face, 24 checks).
Named: me at 0.7 s, mom at 5.2 s, me at 6.7 s, me at 21.7 s — all correct.
Faces skipped: 12 blurred, 123 with no person box (faces in the wall photos
and people half out of frame).

**Found by looking at the labelled frames:** at ~14 s, as they crossed,
ByteTrack swapped the two ids. From ~14.0 to 15.8 s mom's box read "me" and the
user's "mom". At 15.8 s the 3-second re-check caught it (`now reads as mom:
tracker swapped people`) and both were renamed correctly (frames at 17 s and
22 s verified). Correct in the end, but 1.8 s of wrong names — which led to
decision 5's crossing re-checks, the ambiguity rule and keeping reads on
un-naming.

`room_entry.mp4`: named "me" at 2.3 s, score 0.54; the second 2-frame track
stayed unknown. Same as face-lab.

`main.py` has no incident layer (webhooks run in the service), so run 2 used
`serve.py` with a subscription pointed at `tools/dummy_receiver.py`.

### Run 2 — `serve.py` + dummy receiver, final code

Same `switch_test.mp4`, fresh database, both selfies enrolled, webhooks
subscribed to `tools/dummy_receiver.py`, CLIP background embeddings disabled
(confirmed on `/health`).

| Time | Object | Event | Score |
|---|---|---|---|
| 0.67 s | 1 | face_match **me** | 0.55 |
| 5.28 s | 4 | face_match **mom** | 0.62 |
| 6.65 s | 5 | face_match **me** | 0.56 |
| 14.67 s | 5 | swap caught: "now reads as mom" → face_match **mom** (after_swap) | 0.72 |
| 14.67 s | 4 | lost "mom" (object 5 matched mom better) | — |
| 15.00 s | 4 | face_match **me** | 0.71 |
| 22.12 s | 14 | face_match **me** | 0.66 |

**The crossing fix, checked on full-resolution frames.** ByteTrack swapped
the ids at ~14.0 s again. Labels were wrong from ~14.0 to 14.67 s (**0.7 s**,
down from 1.8 s in run 1); the user's box was unnamed for 0.33 s; from 15.0 s
both names were right and stayed right (frames at 15.5, 17 and 22 s checked).
14 crossings were detected, 20 faces were skipped as ambiguous (one face inside
two person boxes), and AdaFace ran 35 times (24 in run 1: the extra runs are
the quick re-checks after crossings).

**Webhooks.** 6 `face_match` events → **2 incidents**, `me` at 0.67 s and `mom`
at 5.28 s. The other four were the same person on the same camera within the
60 s cooldown (the clip is 25 s) — stored as events, not re-sent, as designed.
Both deliveries: HTTP 200, signature **valid**, no duplicates.
`image_url` → `/faces/1.jpg` and `/faces/4.jpg` both served the face snapshot
(HTTP 200, JPEG).

The dummy receiver listed both as "missing keys: vehicle" — its check assumed
every non-congestion incident is about a vehicle. `tools/dummy_receiver.py`
now expects `person` / `sighting` / `image_url` for `face_match`; re-checking
the stored payload reports nothing missing.

**Speed.** 760 frames in 936 s (0.8 fps), against 281 s for the same clip under
`main.py`. The service also stores every frame as JPEG and serves the video
stream, and inline AdaFace averaged 11.1 s per face under that load (2.8 s in
run 1). Correct, but a warning: on this laptop, recognition inside the service
needs `frames.enabled: false` or a lower `processing.analyse_fps` to stay near
real time.

## Limits to know

- **Scores are cosine similarity, not percentages.** Same person in video
  0.51–0.70; different people at most 0.27. A match means "please check",
  with the snapshot.
- **CPU.** One or two people at a door is real time; a crowded street is not
  without a GPU (each new face needs ~1–3 s of AdaFace). Keep CLIP background
  embeddings off while this runs.
- **A name follows the tracker.** Between a swap and its correction (now
  about one to two checks after the people separate) a label can be wrong.
- **Snapshots** under `outputs/<camera>/faces/` are not reaped by retention yet.
- **Consent.** Enrol only people who agreed; `people/`, `samples/faces/`,
  `models/faces/` are gitignored and must stay local. The website's "no facial
  recognition" copy must be reconciled before this ships.
