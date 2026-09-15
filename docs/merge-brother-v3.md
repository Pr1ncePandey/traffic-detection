# Merge log — brother's v3 (vector search + new dashboard), 2026-09-15

What arrived, what was taken, what was deliberately left out, what had to be
fixed on the way in, and how the result was checked. Branch:
`merge-brother-v3`, off `main` at `9a05dbd`.

## What arrived

`traffic-detection-main (1).zip` — 4.6 GB compressed, 7.05 GB, 89,200 zip
entries. The project code itself is ~2 MB. Almost everything else is data or
tool state that must not enter the repository:

| Part | Size | Decision |
|---|---|---|
| `models/clip_l14`, `siglip`, `clip`, `clip_b16` | 4.4 GB | **copied locally, not committed.** Search encoders (ONNX + tokenizer). Already gitignored by his `.gitignore`; reproducible with `tools/fetch_clip.py`. |
| `models/person_attr` (incl. `.convert-venv/`) | 394 MB | **not copied.** The ONNX is byte-identical to ours; the conversion venv is a macOS Python environment. |
| `.venv/` | 1.3 GB | **not copied.** A macOS virtualenv, useless on Windows. |
| `samples/long/*.mp4` | 585 MB | **copied locally, gitignored.** 265 MB and 320 MB each: over GitHub's 100 MB per-file limit, and `person_test_long` is built from a downloaded YouTube clip we decided not to republish. |
| `__MACOSX/`, `.DS_Store` | 64 MB, 44,596 files | **not copied.** macOS Finder archive metadata. |
| `.code-review-graph/graph.db` | 139 KB | **not copied, now ignored.** An editor plugin's code-graph cache with absolute paths (its own `.gitignore` says "do not commit"). |
| `.claude/settings.local.json` | 1 KB | **not copied, now ignored.** His personal Claude Code permissions. |
| `__pycache__/`, `yolov8n.pt` | — | not copied (already ignored; we have our own `yolov8n.pt`). |

Symlinks in `.venv/bin/` and `.convert-venv/bin/` failed to extract on Windows
("symlink error") — irrelevant, since neither folder was taken.

## Which version the zip was based on

Compared every code/config/doc file (CRLF-normalised git blob hashes, models,
samples and outputs excluded) against our commits. The zip matches best at
`12f05af` — the last commit pushed to GitHub, which includes the person
attributes, the shared self-tests and `dummy_receiver.py`. So it was built on
top of our pushed work, not an older copy: taking his files does not undo it.

It did NOT include our one local, unpushed commit `9a05dbd`, so three things
in that commit had to be carried over by hand (see "Fixed on the way in").

## What changed (his work)

**21 new code/doc files, 40 changed.**

**Open-vocabulary search** — find objects by a text description the attribute
table cannot express ("person carrying a box"):
- `src/query/clip_onnx.py`, `embed.py`, `router.py`, `search.py`; `query.py --find`
- `src/server/embedder.py` — indexes crops on a background thread with a duty
  cycle so it does not starve the cameras; `src/server/queries.py`; `/search`,
  `/vocabulary` routes
- `tools/embed_crops.py`, `tools/fetch_clip.py`, `tools/eval_retrieval.py`
- `docs/vector-search.md` (1,430 lines): the design and its measurements
- Storage: a new `embeddings` table **inside the existing SQLite database**
  (`object_id, model, dim, vec, crop_*`). The "vector database" is this table,
  not a separate server — no new service to run, no new pip package.

**New dashboard** — `src/server/static/index.html`, `app.js` (1,123 lines),
`app.css`; `src/server/video.py` serves an MJPEG stream the dashboard draws
boxes over; `dashboard.py` shrank by ~275 lines as the page moved to static
files. Camera switching as described in the integration plan.

**Identity rename** — durable identity is now class-agnostic:
`vehicles` table -> `identities`, `vehicle_id` -> `identity_id`,
`store.vehicles` -> `store.tracks`, API `/vehicles` -> `/identities`. Touches
`sqlite_store.py` (+214/-91), `pipeline.py` (+120/-60), `trackers/`, the
person-attribute enrichers (one line each), tests and tools.

**Defaults**
- `plate.ocr_backend` default `paddle_anpr` -> `fast_plate`, because
  `paddle_anpr` cannot be installed by `pip install -r requirements.txt` and a
  clean install silently read no plates.
- New `server.video` (stream) and `server.search` config blocks.

**Samples reorganised** into `samples/short/` (feature testing) and
`samples/long/` (dashboard soak testing), with a `samples/README.md`;
`samples/retrieval_gt.csv` added; `person_test.mp4` trimmed of its title cards.
New `cameras/person_test.yaml` pointing at the long clip. `cameras/indian_road.yaml`
now points at `samples/long/indian_road_long.mp4`.

`requirements.txt`: comment changes only — nothing new to install. Every
third-party module his code imports was already present in the `traffic` venv.

## Fixed on the way in

1. **`--max-people` restored** in `tools/eval_person_models.py`. His copy
   predates our local commit, so taking it wholesale would have deleted the
   option. Re-applied on top of his version (which renamed `vehicle_class` ->
   `cls_name`).
2. **Consented footage re-protected.** His `.gitignore` also predates that
   commit and dropped `samples/model_test.mp4`. Re-added, together with the
   new sample paths, `samples/long/`, `__MACOSX/`, `.code-review-graph/` and
   `.claude/settings.local.json`. Verified with `git check-ignore` that every
   large or personal file is ignored and the small tracked samples are not.
3. **`cameras/demo.yaml` kept and repaired.** It is missing from his zip, but
   his own `README.md`, `config.yaml`, `main.py`, `serve.py`,
   `cameras/README.md` and `_dash_demo.yaml` still document
   `--camera demo` — so its absence looks accidental, not a deletion. Kept, and
   its `source` updated from `samples/input.mp4` to `samples/short/input.mp4`:
   the service listed the camera with the old path, so starting it from the
   dashboard would have failed to open the video.
4. **Background embeddings switched off** (`server.search.embed_in_background:
   false`) as the brother asked: CLIP L/14 is ~317 ms per crop and competes for
   the CPU with recognition, plates and attributes. `/health` confirms the
   embedder reports `state: disabled`. One line to turn back on.
5. **Tracked sample videos moved with `git mv`** (`samples/indian_road.mp4`,
   `samples/input.mp4` -> `samples/short/`), after confirming they are
   byte-identical to his copies, so git records renames rather than 20 MB of
   deletes and re-adds. `samples/plate_gt.csv` and `samples/plates/` kept: not
   in his zip, but `tools/bench_ocr.py`, `README.md` and `config.yaml` still use
   them.
6. **Old database archived, not deleted.** `outputs/traffic.db` predates the
   `identity_id` schema and the new store refuses to open it (by design, so
   rows are not dropped silently) — `serve.py` would not start. Moved to
   `outputs/_old_schema/traffic_pre_identity_2026-09-15.db`.
7. `docs/person-attributes-analysis.md` (ours, not in his zip) kept, with a
   note that the sample clips it names now live in `samples/short/`.

## How it was checked (Windows, `traffic` venv)

| Check | Result |
|---|---|
| Every tracked `.py` file compiles | pass |
| `tools/test_lanes.py` | 52 passed |
| `tools/test_stage.py` | 30 passed |
| `tools/test_reid.py` | 39 passed |
| `tools/test_congestion.py` | 12 passed |
| `tools/test_incidents.py` | 70 passed |
| `tools/test_webhooks.py` | 48 passed |
| `tools/test_api.py` against a fresh new-schema DB | 57 passed |
| `main.py --camera indian_road --source samples/short/indian_road.mp4 --max-frames 90` | ran clean: 58 tracks, 3 crossings, 1,763 rows; perception `color, plate, person` |
| New schema in that DB | `identities` table, `objects.identity_id`, `embeddings` table (0 rows — embedder off) |
| Person attributes after the `store.tracks` rename | stored: facing, sleeves, lower, bag, hat, glasses on 11 people; `person_read` events |
| `serve.py --no-autostart` | starts; `/health`, `/cameras`, `/`, `/static/app.js`, `/static/app.css`, `/docs`, `/openapi.json` all 200; cameras `demo`, `indian_road`, `person_test` |
| `tools/bench_ocr.py` on the 5 labelled plates | CER `paddle_anpr` 0.19, `fast_plate` 0.54, `rapidocr` 0.88 — identical to before the merge |

Total: 308 self-test checks passing.

## Things for the user to know

- **Plates on this machine.** `paddle_anpr` IS installed here and is almost
  3x more accurate (CER 0.19 vs 0.54). The new default `fast_plate` is right
  for a clean install but worse here; setting `ocr_backend: "paddle_anpr"` in
  `config.yaml` restores the old behaviour. Left as his default, not changed.
  The 90-frame check run read no plates — expected for so short a run on the
  weaker reader; reading itself is verified by the benchmark above.
- **Dashboard cameras use the long clips** (10 minutes each), which now exist
  locally in `samples/long/`.
- **Search needs embeddings.** With background embedding off, `/search` has an
  empty index until `tools/embed_crops.py` is run or the embedder is turned on.
  The CLIP/SigLIP weights are already in `models/`.

## Face recognition integration

Not started in this merge. Next step, per `face-lab/ANALYSIS.md` ("Merging
into the main project later - the agreed plan"). Relevant to it from this
merge: identities are now class-agnostic (a person can hold an
`identity_id`), crops and the `embeddings` table are the natural home for
face vectors, and each camera's config already selects its analyses.

**Update (2026-09-15):** integrated on branch `face-recognition` - see
`docs/face-recognition.md`.
