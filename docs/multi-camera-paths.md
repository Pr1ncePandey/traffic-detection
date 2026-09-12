# Multi-camera journey reconstruction

**Status:** planned, not started. Layer 0 is a go/no-go gate — see below.

Reconstruct the route a vehicle took across the camera fleet: given that the
same car was identified at several cameras, emit the ordered sequence of
cameras it passed, with the time and distance between each.

The output asserts *only observed hops*. Stretches between two cameras are
reported as gaps, never as an inferred route. The system saw a car at A and
later at D; it did not see what happened in between and does not claim to.

---

## What already exists

Three things are already in place, which is why this is a feature and not a
rewrite.

**The vehicle id is fleet-global.** `storage.path` defaults to
`outputs/traffic.db` (`src/config.py:150`) and is in the locked set that a
`cameras/<id>.yaml` may not override (`src/config.py:26`). One database serves
every camera, so `vehicles.plate NOT NULL UNIQUE`
(`src/storage/sqlite_store.py:63`) is a global key: camera A and camera B
resolving the same plate arrive at the same `vehicle_id` through
`PlateIdentity.resolve()`, with no cross-camera code involved.

**Sightings are already queryable across cameras.** `query.py --vehicle N`
returns every `objects` row for one car "across runs and cameras" (its own
docstring, line 16).

**Camera attribution exists**, via `objects.run_id -> runs.camera`
(`src/storage/sqlite_store.py:34`).

So a journey is, structurally, `objects WHERE vehicle_id = ?` ordered by time
and joined to `runs`. The data model does not have to change shape — but two
of those three ingredients have a defect that this feature exposes.

---

## Decisions taken

| Question | Decision |
|---|---|
| Absolute time origin | Wall-clock at detection (already true for live); `recorded_at` for files |
| Camera location | `lat`/`lon` only — no adjacency graph |
| Path semantics | Observed hops, gaps marked explicitly |

Rejected, deliberately: **map-matching onto a road network**. Snapping a
journey to street geometry needs OSM data plus a matching stage, and is larger
than every layer below combined. Also rejected: **inferring intermediate
cameras** from a topology graph, which conflicts with "gaps marked" — it would
assert positions nothing witnessed.

---

## Concern 1: `first_seen_s` holds two incompatible kinds of number

`src/pipeline.py:199`:

```python
timestamp = (time.time() if info.is_live
             else (index / info.fps if info.fps else index))
```

For a **live** source this is a wall-clock epoch, and it flows unchanged into
`objects.first_seen_s`. Cross-camera ordering therefore already works for RTSP
cameras, with no new column.

For a **recorded file** it is seconds-from-the-start-of-that-clip. Every clip
starts at 0, so A@12.4 and B@8.1 carry no information about which came first.

Storing processing wall-clock instead does not fix the file case, and is worth
writing down because it is the obvious thing to reach for:

- Process camera A's clip today and camera B's tomorrow, and every A sighting
  precedes every B sighting.
- Within one run, the spacing between detections is a function of throughput.
  A clip decoded at 3x real time and one decoded at 0.8x produce different
  wall-clock gaps for the same traffic. The ordering would encode GPU speed.

**The latent bug.** `runs` records no `is_live` flag, so nothing in the
database distinguishes an epoch (~1.7e9) from clip-seconds (~12.4) in the same
`first_seen_s` column. The intended deployment is one fleet-wide DB, so a live
run and a file run *will* coexist there, and any query that sorts by time then
returns silent nonsense. This is worth fixing on its own merits, independent of
journeys.

## Concern 2: expected hop yield is low

Cross-camera identity is strictly harder than the same-camera case: different
angle, lighting, and plate resolution. `reid.fuzzy_distance` defaults to `0`
(`src/config.py:168`), so two cameras must produce **character-identical**
voted plate strings to be linked. Measured on one sample, same-camera
exact-match was 0/5 (CER ~0.19) — across cameras it can only be worse.

This does not invalidate the design, but it fixes an assumption: the hop set
is **sparse and noisy**, not a dense trail. Every layer below is built to
degrade into "we know less than you'd like" rather than into a confident wrong
answer. It is also why Layer 0 comes first.

Raising `fuzzy_distance` is the available lever and is *not* recommended
blind: `src/trackers/identity.py:62` notes that real plates genuinely differ by
one character, so a Hamming-1 merge fuses strangers rather than repairing OCR
noise. If it is raised, measure the false-merge rate on real footage first.

---

## Layer 0 — measure the yield (go/no-go)

Before building anything. Run two cameras over footage containing genuinely
shared vehicles, then count vehicles seen by more than one camera:

```sql
SELECT o.vehicle_id, COUNT(DISTINCT r.camera) cams, COUNT(*) sightings
FROM objects o JOIN runs r ON r.id = o.run_id
WHERE o.vehicle_id IS NOT NULL
GROUP BY o.vehicle_id HAVING cams > 1;
```

Zero rows means the layers below would be scaffolding around an empty set, and
the real work is plate accuracy instead. Publish the number before proceeding —
it is also the honest headline for whatever this feature later reports.

## Layer 1 — make time comparable

Schema (`src/storage/sqlite_store.py`, the `runs` table):

- `time_base TEXT` — `'epoch'` or `'clip'`. Set from `info.is_live`.
- `recorded_at REAL` — the footage's real start time, for file sources.

`main.py` gains `--recorded-at`. Live runs need neither flag nor conversion.

One helper is the single point of truth:

```
absolute_time(run_row, secs) -> float | None
    time_base == 'epoch'                  -> secs        (already wall-clock)
    time_base == 'clip' and recorded_at   -> recorded_at + secs
    otherwise                             -> None
```

`None` is load-bearing. A file run with no `recorded_at` is *unorderable*, and
the journey query must set those sightings aside and say so, not guess at them.

**Migration.** `_check_schema` (`src/storage/sqlite_store.py:204`) already
hard-errors on a database predating `vehicle_id`, because `CREATE TABLE IF NOT
EXISTS` cannot add a column and the mismatch would otherwise discard whole
batches of rows silently. New columns need the same guard, for the same reason.

## Layer 2 — camera location

Add to `DEFAULTS["camera"]` (`src/config.py:65`):

```yaml
camera:
  location: {lat: 12.9716, lon: 77.5946}
```

Adding it to `DEFAULTS` is what makes it a *known* key. Without that,
`_warn_unknown` (`src/config.py:232`) prints "no such setting; ignoring" and
the value is dropped — a typo'd or absent default fails quietly.

Snapshot `cam_lat` / `cam_lon` onto the `runs` row at run time. Editing a yaml
months later must not retroactively rewrite historical journeys.

**What lat/lon buys, concretely: implied-speed gating.** Haversine distance
divided by elapsed time gives km/h for each hop. Anything above
`journey.max_speed_kmh` is an OCR collision or a cloned plate, not a journey.
This recovers most of what an adjacency graph was wanted for, with no topology
file to maintain.

**What it cannot do:** tell you that an A->D hop skipped B and C. Given that
the output marks gaps rather than claiming intermediate positions, this costs
nothing that is being promised.

## Layer 3 — journey assembly

New module `src/journeys.py`. Offline and read-only: it must not couple to the
pipeline loop, which knows nothing about other cameras.

1. **Gather.** Sightings for one `vehicle_id`, joined to `runs`, each converted
   through `absolute_time()`. Unorderable sightings are collected separately
   and reported, not dropped.
2. **Collapse.** Consecutive sightings at the same camera become one *visit*
   with a dwell time. A car lingering in A's view is one visit, not N.
3. **Split.** Break into separate journeys on an idle gap exceeding
   `journey.max_gap_s`. A car at A in the morning and at A again at night made
   two trips; it did not loop.
4. **Validate.** Per hop compute distance, elapsed, implied speed. Hops above
   `max_speed_kmh` are **flagged as suspect identity, not discarded** — a bad
   plate merge should be visible rather than invisible.
5. **Emit.** Ordered visits and hops, with every unobserved stretch labelled.

New config block:

```python
"journey": {"max_gap_s": 1800, "max_speed_kmh": 150},
```

## Layer 4 — output

`query.py --path V7`. Per visit: camera, local time, dwell. Per hop: distance,
elapsed, implied speed, and a suspect marker where Layer 3 set one.

Note that `query.py`'s `_SELECT` (line 58) does not join `runs` at all today,
so the camera name never appears in its output. That join is new work.

---

## Open question

**Overlapping time windows.** If one plate is present at two cameras
simultaneously, that is physically impossible — a cloned plate, or an OCR
collision between two different cars. Proposed: report it as a conflict on the
vehicle and refuse to build a journey through it, rather than silently picking
a winner. Confirm before implementing; the alternative is to keep it quiet,
which hides a data-quality signal worth seeing.

---

## Dependency order

Layer 0 gates everything. Layer 1 gates Layers 3 and 4 — without a comparable
clock there is nothing to sort. Layer 2 is independent of Layer 1 and can be
done in parallel. Layers 1 and 2 are worth having on their own: Layer 1 fixes a
real correctness bug in the existing time column, and Layer 2 is the
prerequisite for any geographic reporting at all, journeys or not.
