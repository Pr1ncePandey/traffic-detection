# Open-vocabulary search over object crops

**Status:** Layer 0 measured and **passed** (2026-09-13 — results below, and they
overturn Concern 4). Layers 1 and 4 were then restructured: most of the eval set
turns out to be derivable from the pipeline's own attribute rows, so the fatal
gate is now free and runs *before* the only hand work. Layer 4a is the decisive
go/no-go; this feature may still be measured and abandoned, and that is an
acceptable outcome.

**Update 2026-09-14: a Layer 4a preview has now been run** on two clips with
three encoders, ahead of Layers 2 and 3, because the ONNX weights turned out to
be cheap to obtain. It **splits the gate**: retrieval precision passes
(p@5 0.80), the negative-control gap fails on every encoder tried, and the
gap metric as this document specifies it is itself unsound — it returns a false
PASS. See *Layer 4a preview* below.

**Built 2026-09-14, and working end to end:**

| layer | state |
|---|---|
| 2 — schema | `embeddings` table, `put_embedding()`, **and the Concern 6 reaper fix** |
| 3 — embedder | `tools/embed_crops.py` + `tools/fetch_clip.py` + `src/query/clip_onnx.py` |
| 5a — router | `src/query/router.py`, deterministic path **and** the `claude` CLI path |
| 5b — query path | `src/query/search.py`, wired to `query.py --find` |

300 of 645 crops are embedded under `clip-vit-b16` (345 skipped by the three
filters, including both `giraffe` boxes), at 65.5 ms/crop. Not built: Layer 1b
(hand labels), Layer 4b, anything in Layer 6.

Answer questions the attribute vocabulary cannot express — "a person carrying a
cardboard box", "a car with a roof rack", "a scooter with three riders" — by
embedding the per-object crop that is already written to disk and searching it
by text.

This is **additive**. Queries that the attribute table already answers exactly
keep being answered exactly:

| Question | Path | Why |
|---|---|---|
| "man in a red shirt" | SQL on `attributes` | `upper_color` is already measured, with a confidence |
| "plate HR26DK8337" | SQL on `vehicles.plate` | exact string, `NOT NULL UNIQUE` |
| "person carrying a box" | vector | nothing extracts this |

Re-deriving `upper_color='red'` from an embedding would discard a measured
value and its confidence in favour of an uncalibrated similarity score. The
vector path must never shadow a predicate the schema can already state.

---

## What already exists

Four things are in place, which is why this is a feature and not a rewrite.

**One crop per object, already selected and written.**
`_keep_best_crop` (`src/pipeline.py:564`) writes `raw[y1:y2, x1:x2]` to
`objects.crop_path`, keeping the highest-scoring crop across the track's
frames rather than the first. Enabled by default (`config.yaml:289`). So the
input to an embedder already exists on disk and needs no pipeline change.

**A tall attribute table with the right index.** `attributes`
(`src/storage/sqlite_store.py:88`) plus `idx_attr_kv ON attributes(key, value)`
(`:91`) is what makes the SQL half of the hybrid cheap. Adding an
`embeddings` table follows the same "new capability, no migration" idiom.

**An ONNX runtime is already a dependency**, via `fast-plate-ocr[onnx]` in
`requirements.txt`. A CLIP image/text encoder exported to ONNX drops in behind
the same pattern as `models/lp_detector.pt` — no new framework, no torch.

**Crops are already addressable over HTTP.** `GET /crops/{object_id}.jpg`
(`src/server/app.py:286`) serves `objects.crop_path`, so a retrieval hit is
viewable by a human without new plumbing. This matters more than it looks: a
retrieval result nobody can eyeball is unverifiable.

---

## Decisions taken

| Question | Decision |
|---|---|
| What gets embedded | The existing per-object crop. One vector per object. |
| Where vectors live | A new `embeddings` table in the **same** SQLite file |
| Index | None. Brute-force cosine in numpy over the SQL-prefiltered subset |
| When inference runs | Offline script over crops on disk. **Never** inline in the pipeline |
| Search shape | Hybrid: SQL prefilter (camera, time, class, pixel floor) → vector rerank → threshold |
| Model | **CLIP ViT-B/16** via `onnxruntime` (measured 2026-09-14: best quality-per-ms of five variants; see Layer 4a preview § Throughput). 512-dim, `model` string `clip-vit-b16` |
| Query routing | **Everything semantic goes to the vector path** (decided 2026-09-14). SQL is used only for scoping that cannot be wrong — camera, time, `crop_conf`, `min_px`, box area. No attribute or class predicates are derived from the query. See Layer 5a |
| Router LLM | The **local `claude` CLI** in print mode (decided 2026-09-14), not an HTTP API client. No key handling in this project; deterministic fallback when absent |
| Surface | `query.py --find "..."` first. No HTTP endpoint in v1 |
| Answering "absent" | **Descoped for v1** (decided 2026-09-14). No calibrated similarity floor — `--find` ranks, it does not adjudicate presence. See Concern 2 |

**No ANN index and no external vector service in v1.** At test-phase volume
brute force is faster than any index and has no recall loss, no tuning and no
second process. Introducing Qdrant or pgvector before the object count is known
would be committing infrastructure to an unmeasured problem — and it reopens
Concern 6 below, which the single-file choice closes for free.

That holds for v1 and **not much beyond it**: Layer 0 measured ~8.9 sightings
per second of footage, so the runway before brute force stops being adequate is
shorter than "test phase" suggests. The decision stands for v1; the trigger to
revisit it is a volume measurement over a representative window, not a
milestone.

**Offline rather than inline** is not just a performance hedge. The embedding is
needed by no real-time decision, and keeping it out of the pipeline means a
model swap is a re-run of a script rather than a change to `pipeline.py`. It
also keeps the single-writer invariant intact: the embedder writes through the
existing `_put`/`prune` queue on the one `SqliteStore` connection.

**Rejected deliberately:** embedding every *frame* rather than every object
(multiplies volume by frames-per-track for no query the object crop cannot
serve); replacing attribute queries with semantic ones (see the table above);
and fuzzy natural-language matching over plates (the same reasoning that keeps
`reid.fuzzy_distance` at 0 — `...8337` and `...8338` are two cars).

---

## Concern 1: the model is weak at exactly this query shape

The motivating query is *attribute binding* — "a person wearing a **red**
shirt". CLIP-family encoders behave substantially like a bag of concepts: they
register "person", "red" and "shirt" as present without reliably binding them
to one another. A query for "man in a red shirt and blue trousers" will rank a
man in a blue shirt and red trousers highly.

This is the single largest risk, because compositional attribute queries are the
use case. Two consequences for the design:

- Any query expressible as attribute predicates **must** be routed to SQL, where
  binding is exact by construction. The router is not an optimisation; it is
  the correctness boundary.
- Because routing is LLM-inferred rather than rule-based (see Decisions), that
  boundary is **probabilistic**. An LLM router can send "person in a red shirt"
  to the vector path when `upper_color='red'` would have answered it exactly,
  silently trading an exact answer for a ranked guess. The router is therefore
  itself under test: Layer 4 measures **route accuracy** alongside retrieval
  quality, and the chosen route is printed with every answer so a wrong routing
  decision is visible rather than inferred from bad results.
- Vector queries should be pitched at *object-level* concepts ("carrying
  something", "roof rack", "open truck bed") rather than attribute
  conjunctions, and the eval set in Layer 1 must include a binding-specific
  probe so the weakness is measured rather than assumed.

### MEASURED 2026-09-14 — and it is the largest effect found so far

Layer 1a's `class × colour` rows are themselves a binding probe, so this cost
nothing to measure. 14 compositional queries per clip on ViT-B/16, scoring each
top-5 hit into four buckets:

| clip | queries | positives | p@5 **both** terms | p@5 **≥1** term | binding gap |
|---|---|---|---|---|---|
| indian_road | 14 | 99 | **0.31** | 0.73 | **+0.41** |
| person_test | 14 | 129 | **0.26** | 0.89 | **+0.63** |

Against class-only p@5 of 0.70–0.85 on the same corpus and encoder, adding one
colour term costs roughly **0.70 → 0.28**. The model is not failing to see; it
is failing to *combine*. Of the near-misses, 53 ignored the **colour** and 20
ignored the **class** — so the class term is the reliable one and the attribute
term is what gets dropped.

Individual rows are unambiguous:

| query | p@5 (both) | what the top 5 actually were |
|---|---|---|
| `blue car` | 0.0 | 5 cars, none blue |
| `blue person` | 0.0 | 5 people, none in blue |
| `white bus` | 0.0 | 4 white non-buses |
| `grey car` | 0.0 | 4 cars of other colours |
| `black car` | 0.0 | 4 cars of other colours |

**Consequence for the design — and the mitigation was declined, deliberately.**
This is no longer a predicted weakness, it is a measured one. The obvious
response is the one this document has argued for throughout: route every
attribute-expressible term to SQL, where `cls_name='truck' AND color='white'`
cannot confuse which term applies to what.

**That was decided against on 2026-09-14.** Everything semantic goes to the
vector path (see Decisions, and Layer 5a). The reason is a counter-measurement:
a SQL attribute predicate has perfect binding but **capped recall**, because
the colour enricher writes nothing when no hue wins 40% of band pixels
(`MIN_FRAC`, `src/attributes/color.py:26`) or when the body has under 400
pixels (`MIN_COUNT`):

| clip | predicate | objects it can see | blind to |
|---|---|---|---|
| indian_road | `color=` | 107/138 vehicles | **22%** |
| person_test | `color=` | 80/123 vehicles | **35%** |
| indian_road | `upper_color=` | 48/55 persons | 13% |
| person_test | `upper_color=` | 205/237 persons | 14% |

So `color='white'` cannot return a white truck the histogram failed to call —
and used as a *prefilter* for the vector step, it would permanently hide that
truck from a query like "a white truck with a roof rack". Neither path
dominates: **SQL is exact but blind to 22–35%; vectors see everything but bind
at 0.28.**

The accepted position is therefore: **Concern 1 is not mitigated, it is
absorbed.** Compositional queries run at the measured p@5 ≈ 0.28 rather than
being handed to SQL. That is a deliberate trade of precision for recall,
consistent with Concern 2's descope — `--find` is a recall-oriented browse
surface whose results a human judges, not a precision oracle.

Two things follow, and both are obligations rather than options:

- **p@5 ≈ 0.28 is the honest headline for compositional queries**, not the
  0.775 measured on class-only ones. Users will type "a white truck". Any
  published figure must say which kind of query it describes.
- **The output must advertise the weakness.** The vocabulary is already read
  from the database for the router, so detecting that a query contains a known
  attribute value is free. When it does, say so — *advise*, do not reroute:

  ```
  note  "white" is a known color value. The vector path binds colour
        unreliably (measured p@5 0.28 on class x colour). For an exact
        answer: query.py --class truck  and filter on the color attribute.
  ```

## Concern 2: cosine similarity cannot express "no match"

"Which camera saw X" needs a yes/no. Top-k retrieval always returns k ranked
rows, including when the footage contains nothing of the kind — it cannot
return empty. That is the inverse of this codebase's disposition everywhere
else: `lanes` withholds a verdict and reports why (`too_slow`, `edge_clipped`),
`identity_id` stays NULL rather than guessed, `absolute_time()` returns `None`
rather than `0.0`.

So a bare top-k search would be the first component here that answers
confidently when it does not know. The mitigation is a **similarity floor
calibrated from negative controls** — queries for things verifiably absent from
the footage — and reporting the score and the floor with every hit. Layer 4
measures whether such a floor exists at all; if the score distributions for
present and absent concepts overlap, there is no threshold to pick and the
feature cannot honestly ship.

### DESCOPED for v1 — decided 2026-09-14

The Layer 4a preview measured the distributions and **they do overlap**: no
floor exists for any encoder tried. Rather than let that kill the feature, the
capability is being dropped instead of faked. `--find` **ranks candidates; it
does not answer "was X here"**. Concretely:

- The query class "which camera saw X" / "was X ever seen" is **out of scope**.
  It is the only class that needs a floor.
- No automated consumer may read `--find` output. That rules out Layer 6's
  incident enrichment until this is revisited.
- The output must carry the uncertainty rather than hide it: report the score,
  the candidate-set size, and an explicit statement that **no similarity floor
  is calibrated** — these are the nearest vectors, not confirmed matches. Not
  withholding a verdict, but not claiming one either.

**Why this is defensible here and not a general licence.** v1 is a CLI a human
reads, and `/crops/{object_id}.jpg` makes every hit eyeballable — so the human
supplies the threshold. That defence is strongest for exactly the queries this
feature exists for: whether a crop shows someone carrying a box is obvious on
sight. It is weakest where verification is hard.

**The limit that must be revisited, because it degrades silently.** Top-1 for
an absent concept is the *corpus maximum*, so the larger the index the more
convincing the garbage: 5 mediocre cars out of 118 is obviously nothing; the 5
most refrigerator-like objects out of 10⁶ will look compelling. **Reopen this
before `--find` is wired to anything automated, or before the index passes
~10⁵ objects.** Both are reasoning from the measurement rather than measured.

**Consequence for the gate: descoping this retires stop condition 1.** Layer 4a
criterion 1 exists only to serve this concern. With the concern descoped, the
measured gap failure is no longer a stop — but it was *descoped consciously,
not passed*. Anyone reading the preview's `FAIL` column needs to know the
feature proceeded by narrowing its promise, not by clearing the bar.

## Concern 3: there is no ground truth — PARTLY RESOLVED

`samples/plate_gt.csv` holds 5 hand-labelled plates, and the README is careful
that n=5 means "ordering solid, values indicative". Nothing equivalent existed
for "person carrying a box", so retrieval quality was not merely unknown but
**unmeasurable**, and a regression would be invisible.

**Mostly solved without labelling.** The pipeline's own attribute rows are
labels: 104 positives across 13 `class × colour` queries, 53 across clothing
colours, plus free negative controls from COCO classes that cannot be present
(Layer 1a). That covers precision@5, route accuracy and — critically — the
negative-control gap that Concern 2 hangs on.

**What remains irreducible** is the open-vocabulary set: a query whose label an
extractor could produce is a query that needs no vector search, so those
~10–15 rows must be labelled by hand (Layer 1b). They must also be labelled
*before* seeing retrieval output for them, or the ground truth gets tuned to the
model.

Derived labels inherit the enricher's error rate and so cap measured precision;
that is fine for a gate and must be stated in any published number.

## Concern 4: crop resolution — RESOLVED, and the original claim was wrong

**The concern as first written was:** `_keep_best_crop` saves crops at native
size with no upscaling (`src/pipeline.py:574`), the colour enricher accepts
person crops as small as 20×60 px (`src/attributes/enrichers/color.py:22`),
CLIP's native input is 224×224 — therefore embeddings inherit the same capture
ceiling that holds plate exact-match at 0/5 on this footage, and retrieval
would decay to noise for most of the frame.

**Measured on `samples/indian_road.mp4`, that reasoning does not hold.** Actual
crop files on disk, 195 objects over 659 frames at 1920×1080:

| group | crops | h p10 | h p50 | h p90 | h max | short side p50 | ≥64 px tall | ≥224 px tall |
|---|---|---|---|---|---|---|---|---|
| vehicle | 138 | 121 | **229** | 486 | 892 | 196 | 99% | 54% |
| person | 55 | 81 | **128** | 258 | 288 | 77 | 95% | 22% |

**Why the inference was wrong:** a plate is a ~10% linear sub-region of the
vehicle that contains it. Vehicle crops here have a median height of 229 px
while the plates inside them are 21–115 px wide — so the plate ceiling is a
statement about plate-sized regions and does **not** transfer to whole-object
crops. Reasoning from one to the other was an error, not a conservative
estimate.

Median person crops at 128 px tall (77 px on the short side) clear the 64 px
gate comfortably, so **person-side search is viable on this footage** and v1
need not be restricted to vehicles on resolution grounds. Two residual limits
worth keeping:

- Only 22% of person crops reach 224 px, so most are upscaled into CLIP. Usable,
  not ideal — this is a quality gradient, not a wall.
- Person short side p10 is 38 px, and 24% fall below 48 px. A `--min-px` floor
  therefore still earns its place, and `crop_w`/`crop_h` in Layer 2 make it
  enforceable per query rather than a global guess.

This footage is 1080p with traffic close to the camera. A camera further back,
or 720p, would move these numbers, so **Layer 0 should be re-run per camera**
rather than treated as settled fleet-wide.

## Concern 5: the stored crop is chosen by a rule known to pick badly

`crop_score` (`src/attributes/plate.py:98`) is:

```python
w * h * (1.0 + _sharpness(gray) / 1000.0)
```

Biggest × sharpest. The README already documents this selection as
systematically wrong for plates — plate confidence peaks 9–16 frames *after*
the widest view. For an embedding the bias is different but real: biggest and
sharpest favours the instant the object is nearest the camera, which is
frequently when its box is **clipped by the frame edge**. `lanes` refuses to
issue a verdict on an edge-clipped box for closely related reasons; the crop
writer has no such rule.

Second-order: `_keep_best_crop` deliberately saves a crop even *below*
`crop_min_conf` (0.5) "so every object has one", scoring it `0.0`
(`src/pipeline.py:582`). Embedding those seeds the index with low-confidence
detections that still surface in top-k. Layer 3 records `crop_conf` per vector
so they can be excluded at query time instead of silently ranked.

**Measured, and larger than expected:** 40% of vehicle crops (55/138) and 25%
of person crops (14/55) came from detections below conf 0.5 — 69 of 195 objects
overall. So this is not a tail case; a `crop_conf` filter is load-bearing rather
than defensive.

The same run also produced **two `giraffe` detections at 1070 px tall** on
Indian road footage — full-frame-height false positives from a COCO class that
cannot be present. Those are exactly the rows that would become confident,
high-detail garbage vectors.

**What they actually are (inspected 2026-09-14).** There is no giraffe in the
footage. Both boxes are ~810×1070 — nearly the entire frame — anchored at
`y1=0` and spanning `x1≈50` to `x2≈870`, and both are dominated by the
**yellow-and-grey diagonally striped median kerb running alongside the green
hedge**. That is giraffe texture to a detector: a tall object with tan/yellow
blotches separated by dark bands, against foliage. Object 113 (5 frames,
t=8.07–8.20 s, conf peak 0.545) and object 143 (2 frames, t=11.37–11.40 s,
conf peak 0.506) are the same kerb at the same place in frame, found twice.

Two things follow. The obvious one is that `yolov8n.pt` on an 80-class COCO
head will occasionally emit a class that is physically impossible for the
scene. The one that matters here is that **a near-full-frame box is not an
object crop at all** — it is a picture of the whole road. Embedding it produces
a vector that is genuinely similar to a great many text queries, which is
precisely the "confident garbage" failure.

**So reject on geometry, not on class.** A class denylist would need per-camera
"cannot be present here" config that does not exist, and would not have caught
these anyway — the problem is the box, not the label. Box area as a fraction of
frame separates them cleanly across both runs (645 objects):

| class | objects | max box area |
|---|---|---|
| **giraffe** (indian_road) | 2 | **41.8%** |
| truck (indian_road) | 46 | 24.8% |
| car (indian_road) | 59 | 22.9% |
| person (person_test) | 237 | 14.7% |
| everything else | — | ≤ 10.1% |

There is a clean gap between the largest genuine object (24.8%) and the false
positives (41.8%). A cap anywhere in 25–40% excludes **exactly the two
giraffes and nothing else**:

```sql
-- at embed time, alongside the crop_conf and min_px floors
AND (1.0*(d.x2-d.x1)*(d.y2-d.y1))/(frame_w*frame_h) <= 0.33
```

Take the middle of the gap (~0.33) rather than the edge. **This threshold is
fitted to n=2 false positives on two clips and must be re-checked per camera**,
exactly as Layer 0 must be: a camera mounted closer to the road will have
legitimate vehicles filling much more of the frame, and the gap could close or
invert. Report the exclusion count, the way Layer 3 reports skipped-small,
so the cap cannot quietly start eating real objects.

**Corrected 2026-09-14:** this originally concluded that they "confirm the
confidence floor has to be applied at embed time as well as query time". They
do not. Their detection confidence is **0.55 and 0.51 — both above the 0.5
floor**, so no `crop_conf` threshold excludes them. A full-frame-height false
positive from an impossible class has to be excluded **by class**, not by
confidence. The `crop_conf` column still earns its place for the 35% of crops
that are genuinely sub-threshold; it just does not solve this.

One vector per object is also a real limit: a truck whose load is only visible
from behind, or a person who removes a jacket, gets one snapshot. Multi-view
embedding is deferred, not dismissed — see below.

## Concern 6: the reaper would orphan vectors

`CropReaper.reap_once()` deletes crop files and then nulls the column
(`src/storage/retention.py:134`):

```python
# Null the column so a query does not advertise a file that is gone.
"UPDATE objects SET crop_path=NULL WHERE id IN (...)"
```

That comment states the invariant this feature would break. Retention is
already careful to keep rows and files consistent, with a second axis:
incident-pinned crops survive until their incident is reaped
(`src/storage/retention.py:76`). A vector store adds a **third** thing to keep
consistent, and the reaper knows nothing about it.

The failure is quiet: vector search returns a hit whose image is gone,
`/crops/{id}.jpg` 404s, and the match cannot be viewed or verified. The index
rots over days with no error anywhere.

**Currently inert** — crop retention defaults to `0` (unlimited), so nothing is
reaped in the test phase. This is therefore a **hard prerequisite before crop
or object retention is ever enabled**, not a v1 blocker. Keeping vectors in the
same database makes the fix a single added statement in the same `prune()`
batch; an external store would make it a distributed-consistency problem.

> Adjacent, unrelated to this feature but found while tracing it:
> `_AGE_COLUMN` maps `events` to `ts` (`src/storage/retention.py:56`) and
> compares it against `time.time() - hours*3600`, but `events.ts` is on the
> pipeline's dual clock (`src/pipeline.py:366`) — clip-seconds for a file run.
> Every event row from a file run sits below that cutoff regardless of age, so
> the first retention pass would delete all of them. Inert only because
> `max_age_hours` defaults to `0`. Worth fixing independently.

## Concern 7: a model swap invalidates every vector

Cosine similarity is meaningful only *within* one embedding space. Swapping
CLIP ViT-B/32 for SigLIP invalidates every stored vector, and if crops have
been reaped the pixels are gone and backfill is impossible.

In the test phase there is no history to protect, so this costs nothing **today**
— but only if the `model` column exists from the first insert. Layer 2 includes
it, and query time filters on it, so two embedding spaces can never be ranked
against each other. Adding that column later, after an index exists, is the
expensive version of this.

## Concern 8: this turns a traffic counter into a person search engine

Not a technical concern, and the reason it is written down. The service has no
application-level authentication and binds loopback precisely because `/crops`
serves plate imagery. "Find the person in the red shirt across every camera" is
a materially different capability from counting vehicles — natural-language
person search, keyed alongside plates, times and camera locations.

That is a purpose-limitation question in most jurisdictions, and the moment to
decide it is before the capability exists. Concrete consequence for the plan:
**v1 stays behind `query.py`**, with no HTTP surface, and the Open Questions
below ask whether the target is person search at all.

---

## Layer 0 — measure the ceiling (go/no-go) — DONE, PASSED

Run used, 2026-09-13:

```bash
python main.py --camera indian_road --disable anpr --no-frames
```

Two notes on that command. `--disable anpr` was a **no-op** — the enricher is
registered as `plate`, not `anpr`, and the run said so
(`unknown analysis/attribute 'anpr'`), so plate reads happened anyway. Harmless
here: analyzer selection does not affect crop geometry. And `--no-frames` skips
raw/annotated frame images only; crops are governed separately by
`objects.save_crops` (`config.yaml:289`) and were still written.

The queries used:

```sql
-- crop pixel size by group, from the detection boxes
SELECT o.cls_group,
       COUNT(*)                          AS objects,
       ROUND(AVG(d.x2-d.x1))             AS avg_w,
       ROUND(AVG(d.y2-d.y1))             AS avg_h,
       MIN(d.y2-d.y1)                    AS min_h,
       MAX(d.y2-d.y1)                    AS max_h
FROM objects o JOIN detections d ON d.object_id = o.id
GROUP BY o.cls_group ORDER BY objects DESC;

-- how many objects even have a crop, and how many are sub-threshold
SELECT COUNT(*) total,
       SUM(crop_path IS NOT NULL) with_crop,
       SUM(best_conf < 0.5)       low_conf
FROM objects;
```

**Gate was:** if the median person crop is under ~64 px tall, person-side
semantic search is not viable on this footage and v1 should be restricted to
vehicles; under ~32 px, nothing downstream is worth building until camera
placement changes.

**Result: PASSED.** Median person crop 128 px tall / 77 px short side; median
vehicle crop 229 px. Full distribution in Concern 4 above, which this
measurement overturned.

What the run produced (195 objects over 659 frames, 22 s of 1080p footage):

| | |
|---|---|
| objects with a crop | 195 / 195 (100%) |
| by group | vehicle 138 (car 59, truck 46, motorcycle 22, bus 9, bicycle 2), person 55, animal 2 |
| below `crop_min_conf` 0.5 | 69 / 195 (35%) — see Concern 5 |
| attributes written | `color` 110, `lower_color` 58, `upper_color` 54 |
| `upper_color` values | white 29, black 15, **red 6**, blue 3, grey 1 |

Two findings the gate did not ask for but that change later layers:

**Only 6 objects in this clip have `upper_color='red'`.** So the motivating
query has 6 ground-truth positives in the only footage available. That is
enough to test *routing* and to sanity-check retrieval, but it is far too thin
to report a precision figure on — Layer 1 must either label across both sample
clips or state n explicitly, the way `samples/plate_gt.csv` does at n=5.

**Object volume is higher than the "test phase" framing assumes.** 195
sightings in 22 s is ~8.9/s of footage. Even discounting heavily for duty cycle
and for this being dense urban traffic, one camera plausibly generates 10⁵–10⁶
objects/day, i.e. hundreds of MB/day of vectors at 2 KB each, and a fleet
multiplies it. Brute-force cosine stops being adequate well before "production"
— so the ANN deferral in Layer 6 needs a measurement over a longer, more
representative window before it is relied on. Treat this extrapolation as an
order of magnitude from 22 seconds, not a capacity plan.

## Layer 1 — ground truth

**Revised 2026-09-13.** This layer was originally "hand-label 30–50 queries,
and it blocks the gate". Measurement showed most of it is free and that the
ordering was backwards — the *fatal* gate is the cheap one, so it now runs
first and hand-labelling happens only if it passes. See Layer 4a/4b.

### 1a — derived labels (free, no human effort)

The pipeline already writes labels. From the `indian_road` run:

| source | queries | labelled positives |
|---|---|---|
| `cls_name` × `color` | 13 | 104 |
| `upper_color` (clothing) | 4 | 53 |
| `cls_name` alone | 7 | 195 |
| COCO classes verifiably absent (zebra, boat, airplane, train, surfboard…) | unlimited | negative controls |

So `"a white truck"` arrives with 22 known positives, and the negative controls
need no labelling at all because absence is decidable from the COCO class list
plus the footage's context.

**Qualified 2026-09-14:** absence must be decided **from the database, per
clip**, not from the class list plus judgement about the scene. `person_test.mp4`
produced 7 `boat` and 4 `surfboard` detections — two of the classes this
section offers as examples of guaranteed absence. Derive the negative set with
`SELECT DISTINCT cls_name FROM objects` and subtract; a hand-written list is
how an invalid negative gets in unnoticed.

Generated by SQL, not by hand:

```sql
SELECT o.cls_name, a.value colour, COUNT(*) n
FROM objects o JOIN attributes a ON a.object_id = o.id
WHERE a.key = 'color' GROUP BY 1,2 HAVING n >= 2 ORDER BY n DESC;
```

**These are noisy labels, not truth.** They come from the HSV enricher with its
40% pixel-share floor (`src/attributes/color.py:26`), tuned for daylight — so
"22 white trucks" means "22 objects the histogram called white", and measured
precision is capped by the enricher's own error rate. Acceptable for a go/no-go
gate, which asks "does this work at all" rather than "is precision 0.62 or
0.71". Not acceptable as a published figure without saying so.

They also, by construction, only cover queries SQL already answers — which makes
them ideal for scoring the **router** and useless for testing the
open-vocabulary case that justifies the feature.

### 1b — hand-labelled open-vocabulary set (irreducible, ~10–15 queries)

Only runs if Layer 4a passes. These have no derived label **by definition**: if
an extractor produced the label, the vector path would not be needed.
"carrying a cardboard box", "a roof rack", "three riders on one scooter", "an
open truck bed". `samples/retrieval_gt.csv`, labelled by looking at
`outputs/<camera>/crops/` and the annotated video:

```csv
query,relevant_object_ids,kind,expect_route,notes
"a white truck",7|33,positive,vector,
"a person carrying a bag",12|48,positive,vector,
"a person in a red shirt",91,binding,sql,upper_color='red' is exact - router must prefer SQL
"a person in a red shirt and blue trousers",,binding,sql,Concern 1 probe
"a yellow school bus",,negative,vector,verified absent from this clip
"a person on a horse",,negative,vector,verified absent from this clip
```

Three row kinds carry the measurements: `positive` for precision/recall,
`binding` to size Concern 1 against the SQL answer for the same question, and
`negative` to calibrate the floor in Concern 2. `expect_route` is the fourth,
added once routing became LLM-inferred: it is the label the router is scored
against. Target ~30–50 rows across both sample clips.
`relevant_object_ids` empty means "nothing should clear the threshold", which is
a *testable* assertion, not a missing label.

Layer 0 measured only **6** objects with `upper_color='red'` in
`indian_road.mp4`, so the red-shirt family of queries cannot carry a precision
figure on its own. Label across both clips and state n, as
`samples/plate_gt.csv` does.

**Why ground truth is needed at all**, since the derived labels make it cheap
rather than optional: query `"a white truck"`, look at the top 5 crops, and they
will look plausible whether or not retrieval works — at 128–229 px most vehicles
read as vaguely truck-shaped, and white is the modal colour in this clip (29
cars + 22 trucks). Eyeballing cannot separate working retrieval from
confident-looking noise. This is the trap the plate work already documented: one
car produced 48 distinct readings, each individually plausible, and only labelled
data revealed CER 0.19. The gate is also numeric (precision@5, gap > 0) — drop
the labels and the gate silently becomes judgement.

## Layer 2 — schema

Additive, no migration, in the existing `SCHEMA` block:

```sql
CREATE TABLE IF NOT EXISTS embeddings(
  object_id  INTEGER PRIMARY KEY,
  model      TEXT NOT NULL,          -- 'clip-vit-b32'; two spaces must never rank together
  dim        INTEGER NOT NULL,
  vec        BLOB NOT NULL,          -- float32 LE, L2-normalised on write so cosine == dot
  crop_w     INTEGER,                -- Concern 4, per row and queryable
  crop_h     INTEGER,
  crop_conf  REAL,                   -- Concern 5: det conf of the crop's frame
  created_at REAL);
CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model);
```

Every non-obvious column is one of the concerns above made queryable:
`model` closes Concern 7, `crop_w`/`crop_h` make Concern 4 a `WHERE` clause
instead of an anecdote, `crop_conf` does the same for Concern 5.

**Where `crop_conf` actually comes from (measured 2026-09-14).** Nothing stores
it: `_keep_best_crop` never records *which frame* the kept crop came from, and
`objects.best_conf` is the track MAX, not that frame's confidence. It is
nonetheless recoverable exactly, with no pipeline change, because
`src/detectors/yolo.py:78`–`:79` clips boxes *before* they are stored — so a
saved crop's on-disk `(w, h)` equals some `detections` row's
`(x2-x1, y2-y1)` for that object:

```sql
SELECT MAX(conf) FROM detections
WHERE object_id=? AND (x2-x1)=? AND (y2-y1)=?
```

Over all 195 `indian_road` crops: **0 failed to match**, 172 (88%) matched
exactly one distinct confidence, 23 (12%) matched several detections sharing
that box size (take the max). Recovered values run 0.31–0.94, median 0.53, with
69/195 below 0.5 — reproducing Concern 5's figure exactly. The cheaper
`objects.best_conf` proxy happens to agree with this on the 0.5 filter for all
195 rows, but carries no real value, so a *tunable* floor needs the geometry
read.
`PRIMARY KEY(object_id)` matches `attributes`' one-row-per-thing shape and makes
the reaper fix in Concern 6 a plain `DELETE ... WHERE object_id IN (...)`.

Normalising on write means search is a dot product, so the query path needs no
division and cannot disagree with itself about metric.

Also add the writer template to `_SQL` and the column tuple to the ordering
map (`src/storage/sqlite_store.py:161`–`:186`), so vectors go through the one
writer thread like every other table.

## Layer 3 — offline embedder

`tools/embed_crops.py`:

```bash
python tools/embed_crops.py --db outputs/traffic.db --min-px 48
python tools/embed_crops.py --model clip-vit-b32 --limit 500   # smoke test
```

- select `objects` with a non-null `crop_path` and no row in `embeddings` for
  this `model` — so it is resumable and idempotent
- skip crops below `--min-px` on the short side, and **report the skip count**;
  a silent skip would hide Concern 4 rather than measure it
- batch through the CLIP image encoder, L2-normalise, write via the store queue
- print a one-line summary: embedded, skipped-small, skipped-missing-file,
  ms/crop

Text queries use the same model's text encoder, loaded lazily by the query
path. No new dependency beyond the ONNX weights.

## Layer 4 — evaluation, and THE gate

Split in two, because Layer 1a made the fatal measurement free. **4a runs
first** — it needs no hand labelling, so a failure costs nothing. 4b runs only
if 4a passes.

**Layer 4a — free gate.** `tools/eval_retrieval.py --derived` builds its own
ground truth from the SQL in Layer 1a and reports the negative-control gap,
precision@5 over attribute-expressible queries, and route accuracy. If the gap
is ≤ 0, stop here: nothing has been hand-labelled and nothing is wasted.

**Layer 4b — real measurement.** The same tool over
`samples/retrieval_gt.csv` (Layer 1b), which is the only part that tests
open-vocabulary retrieval — the feature's actual justification.

Both report:

| Metric | From | Meaning |
|---|---|---|
| precision@5, recall@10 | `positive` rows | does it retrieve the right objects |
| **route accuracy** | `expect_route` column | does the LLM router pick SQL when SQL was exact (Concern 1) |
| binding delta | `binding` rows | vector precision vs the SQL answer for the same question (Concern 1) |
| **negative-control gap** | `negative` vs `positive` | min positive score − max negative score (Concern 2) — **but see the preview below: this must be per-prompt centred, or it returns a false PASS** |

**Gate, in order of severity:**

1. ~~If the **negative-control gap is ≤ 0**, there is no threshold that separates
   present from absent. Stop — Concern 2 is unfixable here, and shipping would
   mean answering confidently at random.~~ **RETIRED 2026-09-14.** Measured
   ≤ 0 on every encoder tried, and Concern 2 was then descoped rather than
   solved: v1 does not answer "absent" at all. Still measure and report the
   gap — it is the evidence for the descope, and the trigger for reopening it —
   but it is no longer a stop condition. See Concern 2 § DESCOPED.
2. If **precision@5 < ~0.6** on positive rows, retrieval is not useful enough to
   build a query layer over. Report the number and stop.
3. If the **binding delta** shows vector losing badly to SQL on `binding` rows,
   that is expected (Concern 1) and fine — it confirms the routing boundary
   rather than failing the gate. Record it so the boundary is justified by
   measurement.

Whatever the numbers are, they go in the README next to the OCR CER table, in
the same format and with the same honesty about n.

## Layer 4a preview — measured 2026-09-14

Run **ahead of Layers 2 and 3**, because the ONNX weights turned out to be a
download rather than a conversion, so the fatal measurement could be taken
before any schema or embedder was written. It is a *preview*, not Layer 4a:
n is small and no vectors were persisted. It still hits both stop conditions.

### What was actually run

**Environment.** Neither system interpreter can run this project (no `cv2`,
`ultralytics` or `torch`), and `outputs/` is gitignored, so the Layer 0
database was gone. Rebuilt on **python 3.12** — 3.14 has no torch/opencv
wheels:

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install ultralytics opencv-python pandas tqdm numpy pyyaml
.venv/bin/pip install "fast-plate-ocr[onnx]" tokenizers
```

**Footage.** Two clips, both 1920×1080. These numbers are **as measured**, and
one of them no longer matches the file:

| clip | resolution | fps | frames | duration | scene |
|---|---|---|---|---|---|
| `samples/short/indian_road.mp4` | 1920×1080 | 30 | 659 | 22.0 s | dense vehicle traffic, close to camera |
| `samples/short/person_test.mp4` | 1920×1080 | 25 | 1935 | 77.4 s | pedestrian scene, people further back |

`person_test.mp4` has since had a "THE CCTV PEOPLE" title card trimmed off both
ends (4.1 s at the front, 10.1 s at the back), so it is now **63.2 s / 1,581
frames**. The 1935/77.4 figures above are kept because they are what the
retrieval measurements in this document were actually taken on — re-running
against the trimmed file will produce fewer objects and different ids, so it
will not reproduce the object numbers quoted later. The footage is otherwise
pixel-identical; see `samples/README.md`.

Both clips also moved under `samples/short/`, and the two camera configs now
point at the concatenated `samples/long/` versions for dashboard testing, so
these one-off runs need an explicit `--source`:

```bash
.venv/bin/python main.py --camera indian_road --source samples/short/indian_road.mp4 \
    --ocr-backend fast_plate --no-frames
.venv/bin/python main.py --camera person_test --source samples/short/person_test.mp4 \
    --disable plate --no-frames
```

`cameras/person_test.yaml` was added for the second clip. It disables `lanes`,
`counting` and `congestion`: no geometry is calibrated for a pedestrian scene,
and a verdict from uncalibrated geometry would be invented. Neither analysis
touches crops or attributes, so the two runs stay comparable for this purpose.
`--ocr-backend fast_plate` was used because the default `paddle_anpr` needs a
heavy gitignored setup; OCR backend does not affect crop geometry.

**The `indian_road` run reproduced every published Layer 0 number exactly** —
195 objects, 195 crops, 69 sub-0.5, the full Concern 4 percentile table,
`upper_color` red 6, 13 class×colour queries / 104 positives. The pipeline is
deterministic here, so this database is disposable rather than precious.

**Encoders.** Three, all ONNX, all via `onnxruntime` with no torch at
inference. Note this is **~0.6–1.6 GB per model, not a 6 MB drop-in** like
`models/lp_detector.pt`, and `tokenizers` **is** a new dependency — the text
tower needs a BPE/SentencePiece tokenizer, so "no new dependency beyond the
ONNX weights" was wrong.

| dir | model | vision | text | dim |
|---|---|---|---|---|
| `models/clip` | `Qdrant/clip-ViT-B-32-{vision,text}` | 335 MB | 242 MB | 512 |
| `models/siglip` | `Xenova/siglip-base-patch16-224` | 355 MB | 421 MB | 768 |
| `models/clip_l14` | `Xenova/clip-vit-large-patch14` | 1160 MB | 472 MB | 768 |

Preprocessing differs by family and is not interchangeable: CLIP resizes the
shortest edge to 224 then centre-crops, with its own mean/std; SigLIP squashes
straight to 224×224 with mean/std 0.5 and pads text to a fixed 64 tokens.

**Corpus.** The plan's own filters, with `crop_conf` recovered by the geometry
match described in Layer 2. Positives are `cls_name` classes with n ≥ 10 —
Layer 1a derived labels. Queries use 7-template prompt ensembling, which is how
CLIP zero-shot is normally run.

| | indian_road | person_test |
|---|---|---|
| objects | 195 | 450 |
| dropped `crop_conf` < 0.5 | 69 (35%) | 229 (51%) |
| dropped short side < 48 px | 8 | 37 |
| **corpus** | **118** | **184** |
| positives (n ≥ 10) | car 38, person 33, truck 22, motorcycle 16 | person 90, car 45, umbrella 18, truck 12 |

### Negative controls must be checked per clip, not assumed

Layer 1a proposes "COCO classes verifiably absent" as free negative controls.
`person_test.mp4` produced **7 `boat` and 4 `surfboard` detections** — two of
the exact classes this document lists as examples of guaranteed absence. Both
would have been silently invalid negatives, deflating the gap for a reason
having nothing to do with the encoder.

So the negative set has to be derived from the database
(`SELECT DISTINCT cls_name FROM objects`), not from a hand-written list. After
rejecting `boat` and `surfboard`, the common set used for both clips was:
zebra, train, pizza, refrigerator, elephant, airplane, traffic light, fire
hydrant, toilet, bed, laptop, microwave, teddy bear, sheep.

### Results

`GAP` is Layer 4's negative-control gap: min positive top-1 − max negative
top-1. `raw` is plain cosine, as this document specifies it; `centered`
subtracts each prompt's own mean over the corpus; `z` also divides by its std.

| clip | encoder | mean p@5 | raw GAP | centered GAP | z GAP |
|---|---|---|---|---|---|
| indian_road | CLIP ViT-B/32 | 0.50 | +0.004 "PASS" | −0.042 FAIL | −2.449 FAIL |
| indian_road | SigLIP base/16 | 0.33 | −0.023 FAIL | −0.035 FAIL | −1.964 FAIL |
| indian_road | **CLIP ViT-L/14** | **0.80** | +0.020 "PASS" | **−0.024 FAIL** | −1.579 FAIL |
| person_test | CLIP ViT-B/32 | 0.60 | +0.012 "PASS" | −0.036 FAIL | −2.155 FAIL |
| person_test | **CLIP ViT-L/14** | **0.80** | +0.032 "PASS" | **−0.014 FAIL** | −0.763 FAIL |

Per-class p@5 on ViT-L/14 — indian_road: car 1.0, person 0.8, truck 0.6,
motorcycle 0.8. person_test: person 1.0, car 0.8, umbrella 0.8, truck 0.6.

### What this says

**Criterion 2 (p@5 ≥ ~0.6) passes, on both clips, on ViT-L/14.** Scale is the
lever: B/32 → L/14 takes indian_road from 0.50 to 0.80 and fixes `truck` from
0.0 to 0.6. **SigLIP base is *worse* than CLIP B/32**, so a newer family is not
the lever — parameter count is. Retrieval *ranking* is more viable than
Concern 4 feared.

**Criterion 1 (negative-control gap > 0) fails on every encoder and both
clips.** Once per-prompt scale is removed, absent concepts outrank present
ones: `refrigerator` and `train` score higher against car crops than any
genuine positive does against its own class. This is Concern 2 exactly, and
this document ranks it as the most severe stop condition.

**Footage quality was not the differentiator.** `person_test.mp4` is the same
1080p but has *smaller* person crops (short side p50 **47 px** vs 77 px, and
51% of its detections below `crop_min_conf` vs 35%) — and it scores the **same
p@5 (0.80)** and a **better gap** (centered −0.014 vs −0.024, z −0.763 vs
−1.579). Whatever is capping this, it is not capture quality on these two
clips. The plausible cause of the gap difference is scene diversity: 9 detected
classes vs 7, so negative prompts have less of a single generic "street object"
to latch onto.

### Two corrections to this document's own method

**The gap metric is unsound as specified.** Layer 4 defines the gap on *raw
cosine across different prompts*. Raw cosine is not comparable across prompts —
every score here compresses into ~0.23–0.35, each prompt sitting at its own
scale. That metric returns **PASS (+0.004 to +0.032) on data where no global
threshold exists**, i.e. a false green light on the one measurement designed to
kill the feature. Layer 4a must centre per prompt against a background corpus
before its number means anything. As written it would wave this through into
Layer 1b's hand-labelling.

**mean p@5 is inflated by class imbalance** — the modal class is 32% of the
indian_road corpus and 49% of person_test, and in both cases it is the class
scoring 1.0. **Checked, and the signal survives it.** Against each class's own
prior (`n / corpus`), ViT-L/14's lift is 2.0×–9.2×, mean 3.8× on indian_road
and 5.7× on person_test:

| clip | class | n | prior | p@5 | lift |
|---|---|---|---|---|---|
| indian_road | car | 38 | 0.322 | 1.0 | 3.1× |
| indian_road | person | 33 | 0.280 | 0.8 | 2.9× |
| indian_road | truck | 22 | 0.186 | 0.6 | 3.2× |
| indian_road | motorcycle | 16 | 0.136 | 0.8 | 5.9× |
| person_test | person | 90 | 0.489 | 1.0 | 2.0× |
| person_test | car | 45 | 0.245 | 0.8 | 3.3× |
| person_test | umbrella | 18 | 0.098 | 0.8 | 8.2× |
| person_test | truck | 12 | 0.065 | 0.6 | 9.2× |

Every class beats its prior by at least 2×, and the rarest classes gain the
most — the opposite of what a prior artefact would produce. So p@5 0.80 is real
retrieval, not corpus composition. Report lift alongside p@5 in Layer 4a; the
bare number is misleading on an imbalanced corpus.

### Throughput: the model that passes is too slow to use

Measured on this hardware (Apple M3 Pro, 12 cores, `onnxruntime` 1.30), vision
tower only, batch 32, warm:

| encoder | provider | ms/crop | crops/s | to embed 10⁶ objects | bytes/vector |
|---|---|---|---|---|---|
| CLIP B/32 | CPU | 19.3 | 51.8 | 5.4 h | 2 KB (512×f32) |
| CLIP B/32 | CoreML | 17.1 | 58.4 | 4.8 h | 2 KB |
| **CLIP L/14** | CPU | **298.9** | **3.3** | **83.0 h** | 3 KB (768×f32) |
| CLIP L/14 | CoreML | 302.2 | 3.3 | 83.9 h | 3 KB |

**CoreML gives no speedup** — it falls back for most of the graph, so this is
not fixable by flipping a provider on Apple silicon.

At first this looked like a hard tension — the only encoder passing p@5 (L/14)
was 18× too slow to keep up with one camera, while the one that kept up (B/32)
measured 0.50, below the bar. "Offline rather than inline" does not rescue
that: offline still has to clear the arrival rate or the backlog grows without
bound.

### Resolved: ViT-B/16 — measured 2026-09-14

Five variants, both clips, identical corpora and metric. One model in memory at
a time, batch 16:

| variant | ms/crop | crops/s | 10⁶ | dim | p@5 road | p@5 person | mean p@5 |
|---|---|---|---|---|---|---|---|
| B/32 fp32 | 16.7 | 59.9 | 4.6 h | 512 | 0.50 | 0.60 | 0.55 |
| **B/16 fp32** | **64.6** | **15.5** | **18.0 h** | **512** | **0.70** | **0.85** | **0.775** |
| L/14 fp32 | 312 | 3.2 | 86.8 h | 768 | 0.80 | 0.80 | 0.80 |
| L/14 fp16 | 304 | 3.3 | 84.4 h | 768 | 0.80 | 0.80 | 0.80 |
| L/14 uint8 | 102 | 9.8 | 28.5 h | 768 | 0.70 | 0.80 | 0.75 |

**ViT-B/16 is the pick**, and it is not a compromise:

- **p@5 0.775 mean, clearing the ~0.6 bar on both clips** — and on
  `person_test` it *beats* L/14 outright (0.85 vs 0.80), at 4.5× the speed.
  Lift over prior 3.5×–5.9×.
- **15.5 crops/s ≈ 1.3×10⁶/day capacity**, against Layer 0's extrapolated
  7.7×10⁵/day arrival at full duty cycle — so it keeps up with one camera with
  headroom, and more once the `crop_conf`/`min_px` filters have dropped 35–51%.
- **512-dim = 2 KB/vector**, restoring the storage estimate this document
  started with instead of L/14's 3 KB.
- 571 MB of weights, the smallest of the passing options.

Two negative results worth not repeating:

- **fp16 buys nothing on CPU** (304 vs 312 ms). ONNX Runtime has no fast
  half-precision CPU kernels and effectively upcasts.
- **uint8 quantisation works but is dominated.** It does deliver 3× (102 ms vs
  312), but costs the quality it was bought for — 0.75 mean, i.e. the same as
  B/16 while being 1.6× slower and 768-dim.

**The remaining limit is fleet scale, not one camera.** 1.3×10⁶/day is one
embedder process on one CPU. Ten cameras at full duty is ~7.7×10⁶/day, which
needs roughly 6× more — so a fleet still implies a GPU or parallel embedders.
Record that as the trigger, and note `--min-px`/`crop_conf` are the cheapest
dials before either.

Because Concern 7 makes the `model` string permanent for every vector written
under it, Layer 3 should write **`clip-vit-b16`** from the first insert.

**Answered along the way:** open question 3 ("one threshold or one per query
kind?") — one global number does not hold on this evidence.

### Not tested

Open-vocabulary queries — "carrying a cardboard box", "a roof rack", "three
riders on one scooter" — which are the feature's actual justification. Only
class-level queries were measured, and those are precisely the ones SQL already
answers. Layer 1b remains the only way to test the real case, and this preview
does not substitute for it.

Also untested: the LLM router (no LLM infrastructure exists in this repo at
all — no client, no key handling, no failure path), and therefore route
accuracy. Since route accuracy is not one of Layer 4a's stop conditions, it
should move to Layer 4b so the free gate is not blocked on the most speculative
component.

## Layer 4b — measured 2026-09-14. The open-vocabulary case is the weakest.

This is the measurement the whole plan was built to obtain, and the one every
earlier number was a proxy for. `tools/eval_retrieval.py --csv
samples/retrieval_gt.csv`, ViT-B/16, 300 embedded crops, 13 relational
positives.

### Labels came free after all — but not from where this document expected

Layer 1b assumed ~10–15 rows of irreducible hand labelling. Two attempts to
avoid it:

**The `person` enricher — tried, and it failed.** It decodes `HandBag`,
`ShoulderBag`, `Backpack` and `HoldObjectsInFront`, was fetched and enabled
for exactly this purpose (`cameras/person_test.yaml`), and on this footage it
produced **5 carrying positives out of 143 persons**. Cross-checked against
independent proposals it confirmed **1 of 16**, and it labels object 238
`bag=none` when the crop plainly shows a shoulder bag. At a person-crop short
side of ~47 px it is below its useful range. Its own docstring says `bag` is
"low-trust, right only in good conditions"; this footage is not those
conditions.

**Box containment — worked.** The detector already emits
`handbag`/`backpack`/`suitcase`/`umbrella` as their own objects, so a person
box containing an accessory box ≥60% in the same frame is a *proposed*
positive. That yields 16 proposals (13 in the index), turning labelling from
searching 300 crops into confirming 13. Eye-verified 3/3 correct: 238
(shoulder bag), 477 (bag at side), 584 (blue plastic bag in both hands).

So Layer 1b cost minutes rather than hours — and the negative result about the
enricher is worth as much as the labels.

### The result

| query | n | p@5 | r@10 | best rank | median rank | random median |
|---|---|---|---|---|---|---|
| a person carrying a bag | 10 | **0.00** | 0.00 | 13 | 38 | 150 |
| a person holding an umbrella | 5 | **0.00** | 0.00 | 28 | 87 | 150 |
| a person carrying something | 13 | **0.00** | 0.08 | 6 | 41 | 150 |

**Nothing relevant reaches the top 5 for any relational query.**

But `p@5 = 0.00` alone would be misleading, which is why the rank columns are
there: median rank 38–87 of 300 against a random median of 150 means the
ranking is **roughly 2–4× better than chance**. This is weak signal, not
noise. "Carrying something" put a true positive at rank 6 — just outside the
window.

### All three query classes, on the same index and encoder

| query class | p@5 | answerable by SQL? |
|---|---|---|
| class only — "a truck" | 0.70–0.85 | **yes, exactly** |
| class × colour — "a white truck" | 0.23–0.30 | **yes, exactly** |
| **relational — "a person carrying a bag"** | **0.00** | **no — this is the justification** |

Retrieval quality is **inversely ordered against need**. It is best on the
queries SQL already answers exactly, and worst on the only class that
justifies building any of this.

### Why, and it revives Concern 4 in a narrower form

Concern 4's original argument — crops are too small, so the plate ceiling
transfers — was refuted for *whole objects*: a median 128 px person crop
retrieves "a person" at p@5 0.60–1.00. That refutation stands.

It does **not** extend to sub-object concepts. A bag is a ~15–20 px region
inside a 47 × 116 px person crop, which is the same relationship a plate has
to the vehicle containing it — and plate exact-match on this footage is 0/5.
So the corrected form of Concern 4 is:

> The capture ceiling does not bind for whole-object queries. It binds hard
> for queries about a **part of** an object, because the part inherits the
> ratio that defeated plate OCR.

That yields a testable prediction this measurement does not settle: relational
concepts on *large* objects should fare better. Vehicle crops have a median
height of 229 px, so "a roof rack" or "an open truck bed with cargo" occupies
far more pixels than a handbag does. No labels exist for those yet, and they
are now the most informative thing left to label.

### CORRECTION, same day: p@5 0.00 was a LABEL artifact, not a result

The table above is accurate and its interpretation was wrong. The labels are
**5% complete**, and precision@k against incomplete labels is a lower bound so
loose as to be meaningless.

`person_test.mp4` has 237 person objects. `yolov8n.pt` detected 78 accessory
objects in total (umbrella 49, handbag 17, suitcase 8, backpack 4), which
yielded **13 labelled carrying positives — 5% of persons.** In 77 seconds of
pedestrian street footage the real rate of someone carrying a bag or umbrella
is plausibly 40–60%, i.e. 100–140 people. **Every person the nano detector did
not happen to find an accessory box for was scored as a miss, including people
visibly carrying things.**

Checked by eye, which is what the whole design says to do. Top 5 for "a person
carrying a bag" on ViT-L/14:

| rank | object | class | actually carrying? | in my labels? |
|---|---|---|---|---|
| 1 | 380 | person | **yes** — under an umbrella, white bundle in both arms | **no** |
| 2 | 536 | handbag | it *is* a bag, not a person carrying one | no |
| 3 | 439 | handbag | ditto | no |
| 4 | 446 | person | **yes** — umbrella and a bag at the side | **no** |
| 5 | 635 | handbag | ditto | no |

So the true p@5 is **at least 0.40**, not 0.00 — two of five are people
genuinely carrying something, and the label set called both wrong. The other
three are bag objects, which is arguably a reasonable retrieval for the query
even though it is not what was asked for.

**Two lessons, and the second is the one that generalises.**

*Box containment cannot exceed detector recall.* It proposes a positive only
where the detector found the accessory. On `yolov8n` that is 5% coverage, so
the method is sound but its yield is capped by the weakest model in the
pipeline.

*Precision@k needs POOLED relevance judgement when ground truth is
incomplete.* The standard fix, and the one this should have used: judge only
the **retrieved** set by eye, rather than scoring against a label set built
independently. 3 queries × top-10 = 30 crops, which is an hour at most, and it
makes precision sound. It says nothing about recall — for that, the label set
has to be more complete, which means a better detector.

### What is still genuinely unresolved

Relational retrieval is **not** measured at 0.00. It is unmeasured, because
the measurement was against 5%-complete labels. The honest state is: eyeballed
p@5 ≈ 0.40 on one query with n=5 judged, which is neither a pass nor a fail.

What the earlier numbers do still support, because they used the pipeline's own
attribute rows rather than detector-derived proposals: class-only p@5 0.70–0.85
and class × colour 0.23–0.30. The binding finding stands.

### Caveats, stated plainly

n is 10, 5 and 13 labelled, of which 3 eye-verified. One clip. Two encoders.
The hardest relational shape — a small carried object on a small person.
Nothing here belongs in a README yet.

### What it is, given the routing decision

With everything semantic going to the vector path, the router is **not** a
SQL/vector arbiter. Its job shrinks to one honest task: **separate scope from
subject.**

```
query  ->  scope: {camera, time window}   +   subject: "free text, verbatim"
```

- **scope** becomes SQL `WHERE` clauses, over fields that cannot be wrong:
  `camera` and a time window. The always-on floors (`crop_conf`, `min_px`, box
  area) come from config, not from the query.
- **subject** goes to the vector path **unchanged**. No term is extracted from
  it, rewritten, or turned into a predicate.

| query | scope | subject |
|---|---|---|
| `a white truck with a roof rack` | — | `a white truck with a roof rack` |
| `a person carrying a box on the north camera yesterday` | `camera=north`, `t in yesterday` | `a person carrying a box` |
| `trucks in the last hour` | `t > now-1h` | `trucks` |

Note what is deliberately absent: `a white truck` produces **no** predicates.
`cls_name='truck'` and `color='white'` are available and exact, and go unused,
for the recall reason measured in Concern 1.

This is a far smaller component than a routing arbiter, and that should be said
plainly: **this decision retires most of the earlier argument for an LLM
router.** What is left does still benefit from a model — "yesterday between 2
and 4pm on the north camera" is real natural-language parsing — but *route
accuracy* is no longer a meaningful metric, because there is only one route.

### The vocabulary is still read from the database — for advice, not routing

```sql
SELECT key, value, COUNT(*) FROM attributes GROUP BY 1, 2;
SELECT DISTINCT cls_name, cls_group FROM objects;
```

No longer used to build predicates. Used to **warn**: when the subject contains
a known attribute value, the output says an exact SQL answer exists and that
the vector path binds it unreliably (Concern 1). One path, honest output. As
enrichers are added the advisory sharpens automatically, with no router change.

### Validation

Scope is the only thing the model produces, so validation is correspondingly
narrow:

1. **`camera` must exist** in `runs.camera`. Unknown → drop it, warn, and
   search all cameras rather than returning nothing.
2. **Time bounds must parse**, and must respect the dual clock: a file run is
   on clip-seconds, so a wall-clock window over file-run rows is **reported as
   excluded**, never silently applied — the `unorderable` discipline from
   `src/journeys.py`.
3. **Never invent scope.** No camera or time stated → none applied. A
   hallucinated window silently hides results.
4. **The subject is never validated or rewritten.** It is user text, passed
   through. This is what makes "everything to vector" hold in code rather than
   only in the doc.

### The LLM is the local `claude` CLI, not an HTTP client

Decided 2026-09-14. The router shells out to the `claude` binary in print mode,
so **this project holds no API key and takes on no HTTP client dependency** —
it reuses the operator's existing Claude Code login.

```
claude -p --bare --restricted --no-session-persistence \
       --output-format json --model haiku "<prompt>"
```

Every flag is load-bearing:

| flag | why |
|---|---|
| `-p` | non-interactive: print and exit |
| `--output-format json` | one JSON result carrying `result`, `is_error`, `duration_ms` |
| `--restricted` | drops the tools that run commands or code. A router needs none, so this removes the possibility of one running |
| `--bare` | skips hooks, LSP and plugins — startup cost, and it avoids firing the operator's own hook chain on every query |
| `--no-session-persistence` | one query is not a session worth keeping on disk |
| `--model haiku` | smallest model sufficient for a parse this narrow |

**Two implementation traps, both found by actually running it:**

- **Close stdin.** Without `stdin=DEVNULL` the CLI waits for piped input and
  logs `Warning: no stdin data received in 3s` — three seconds lost per call.
- **Check `is_error`, not the exit code.** A failed call exits **0** carrying
  `{"is_error": true, "result": "Not logged in · Please run /login"}`. A naive
  `returncode == 0` check feeds that sentence to the JSON parser as though it
  were a completion.

**This is a local *client*, not local inference.** The prompt still goes to
Anthropic, so Concern 8's egress point is unchanged. What leaves the machine is
the **query text and the vocabulary listing — never a crop, an image, a plate
or a row.** What the decision buys is credential handling and dependency count,
not privacy.

**Unmeasured: latency.** Process spawn plus a model call, so plausibly seconds
rather than milliseconds. That makes the cache below matter more than it would
for an HTTP client. Measure before relying on it.

### Failure behaviour, and the deterministic baseline

| condition | behaviour |
|---|---|
| `claude` not on PATH | **deterministic scope extraction** |
| not logged in (`is_error`) | deterministic |
| malformed JSON | one retry, then deterministic |
| timeout (default 20 s) | deterministic |
| never | block, hang, or invent scope |

**Deterministic extraction** regex-matches camera names from `runs.camera` plus
a small set of time phrases (`last N minutes/hours/days`, ISO dates), and
treats everything else as the subject.

Crude, and adequate — which is a consequence of the routing decision worth
noticing: because the subject is passed through verbatim anyway, the
deterministic path differs from the LLM path **only** in how well it parses
scope. So **`--find` works with no `claude` binary, no login and no network**,
degrading only in natural-language date handling. That also makes the
deterministic path the baseline the LLM has to beat.

Cache on (query, vocabulary hash, camera set). Query strings repeat, so most
invocations should never spawn a process.

### Interface

`src/query/router.py`, called by `query.py --find`. No HTTP surface (Concern 8).

```python
@dataclass
class Scope:
    camera: str | None
    t_from: float | None
    t_to: float | None
    excluded_runs: list[int]        # file-run rows a wall-clock window cannot order

@dataclass
class Route:
    scope: Scope
    subject: str                     # user text, verbatim, never rewritten
    source: str                      # 'claude-cli' | 'deterministic'
    warnings: list[str]              # vocabulary advisories, dropped scope, ...

def load_vocabulary(conn) -> Vocabulary
def route(query: str, vocab: Vocabulary, llm=None) -> Route
```

### Output contract

```
query    "a white truck with a roof rack"
route    vector (claude-cli)
  scope   all cameras, all time, crop_conf>=0.5, short>=48px, area<=33%
  subject "a white truck with a roof rack"  -> 116 candidates, reranked
  note    "white" is a known color value; the vector path binds colour
          unreliably (p@5 0.28 measured on class x colour). For an exact
          answer: query.py --class truck, filtered on the color attribute.
  no similarity floor is calibrated - nearest of 116, not confirmed matches
```

### How it is scored

Route accuracy is retired — there is one route. What stays testable:

- **scope-parsing accuracy** — camera and time window against a small labelled
  set, `claude-cli` versus `deterministic`
- **zero invented scope** — the gate. A hallucinated camera or window silently
  hides results, and it is the one failure this component can still cause.
- **subject fidelity** — the subject is byte-identical to the user's text minus
  the scope phrases

## Layer 5b — query path

`query.py --find "..."` only. Order is load-bearing:

1. **SQL prefilter** — camera, time window, `cls_group`, `crop_w >= floor`,
   `crop_conf >= floor`. Narrows to a candidate id set, exactly, cheaply.
2. **Vector rerank** — fetch those rows' vectors, dot product in numpy, sort.
3. **Threshold** — drop everything below the Layer 4 floor. Zero results is a
   valid, reportable answer: *"no match above 0.31"*.
4. **Report** — object id, camera, time, similarity, **the threshold**, and
   crop pixel size. A weak match must look weak in the output.

Time filtering reuses the existing clock discipline: for live runs
`objects.first_seen_s` is already a wall-clock epoch (`src/pipeline.py:366`), so
`first_seen_s > unixepoch() - N` works directly; file runs are on clip-seconds
and must be reported as excluded rather than silently dropped, the way
`src/journeys.py` handles `unorderable`.

## Layer 6 — deferred

Not in v1, listed so the boundary is deliberate:

- **`GET /search`** — waits on Concern 8 and on Layer 4's numbers.
- **Retention integration** — a `DELETE FROM embeddings` in the `CropReaper`
  batch. **Prerequisite before any crop retention is enabled** (Concern 6).
- **Multi-view embedding** — k crops per object, max-pooled at query. Waits for
  evidence that one view is the limiting factor, which Layer 4 will show.
- **ANN index / external store** — Layer 0's ~8.9 sightings/s makes this sooner
  than first assumed. Gated on a volume measurement over a longer window, not on
  v1 shipping.
- **Incident enrichment** — semantic tags on the webhook payload.

---

## Deferred deliberately

**Embedding frames rather than objects.** Multiplies volume by frames-per-track
and answers nothing the object crop cannot, because every query here is about a
*thing*, not a scene.

**Replacing the colour enricher with embeddings.** `upper_color` is measured
with a documented method, a 30% pixel-share floor
(`src/attributes/color.py:28`) and a confidence. A similarity score has none of
those properties.

**Fine-tuning the encoder on this footage.** Possibly the real answer to
Concerns 1 and 4, and far larger than every layer above combined. Revisit only
if Layer 4 says base CLIP is close but not sufficient.

---

## Open questions

**Is the target person search or vehicle search?** This shapes Layers 0 and 5
and decides how much of Concern 8 applies. Vehicles are larger in frame, so the
resolution ceiling bites far less, and the privacy question largely goes away.
Person search is the harder half on Concerns 1, 4 and 8 simultaneously. Confirm
before Layer 0's gate is interpreted.

**~~Should the query router be explicit or inferred?~~ SUPERSEDED 2026-09-14.**
The 2026-09-13 answer was "LLM-inferred", with route accuracy as a Layer 4
metric to police the probabilistic SQL/vector boundary. **There is no longer a
boundary to police:** everything semantic goes to the vector path, so there is
one route, `expect_route` is retired, and the LLM's remaining job is scope
extraction (Layer 5a). What survives from the old answer is the
output-transparency requirement — the chosen path, the scope, and the
vocabulary advisory are printed with every answer.

**One threshold or one per query kind?** A floor calibrated on "a yellow school
bus" may not transfer to "a person carrying a bag". Layer 4 will show whether
one number holds; if not, the honest option is per-kind thresholds with the
kind stated in the output, not an average that is wrong for both.

---

## Dependency order

Layer 0 gated everything and is **done**. Layer 1a is free and independent of
everything, so it can be generated at any point. Layer 4a needs 1a, 2 and 3,
and is the gate that can retire the feature. Layer 1b — the only hand work —
runs *after* that gate, and must not be written after seeing retrieval output
for the queries it labels. Layer 6 waits on Layer 4b's numbers and, for the HTTP
surface, on the privacy decision in Concern 8.

```
Layer 0 (measure)          Layer 1a (derived labels, free, SQL only)
  DONE / PASSED                        │
        │                              v
        └─> Layer 2 (schema) ─> Layer 3 (embedder) ─> Layer 4a (FREE GATE)
                                                             │
                                            gap <= 0 ? ──> STOP (nothing hand-labelled)
                                                             │
                                                    Layer 1b (hand-label ~10-15)
                                                             │
                                                    Layer 4b ─> Layer 5 ─> Layer 6
```

The ordering is the point. The gate that can kill the feature (Concern 2, the
negative-control gap) is now the *cheapest* step, so it runs before the only
expensive one. The original plan had the irreducible hand-labelling ahead of the
fatal gate, which was backwards.

Layer 1b is worth having regardless of whether this feature ships: it would be
the first labelled evaluation set this project has for anything other than OCR.
Layer 0 was worth it for a different reason than expected — it did not confirm
the capture limit, it **refuted** the assumption that the plate ceiling applies
to whole-object crops (Concern 4), which is a claim that would otherwise have
been inherited unexamined into the design.
