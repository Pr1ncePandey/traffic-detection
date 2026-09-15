# Person attributes — analysis and decision log

> **Paths note (2026-09-15):** the brother-v3 merge moved sample footage into
> `samples/short/` and `samples/long/`. The clips named below as
> `samples/indian_road.mp4`, `samples/person_test.mp4` and
> `samples/person_test2.mp4` now live in `samples/short/`. `person_test.mp4`
> there is the trimmed version (title cards removed); the field-test results in
> this document were measured on the untrimmed file.

This records why the `person`, `garments` and `age_gender` enrichers exist,
what was tried, what worked, and what didn't. It exists so the reasoning
survives even if the person who asked for it isn't in the room — read this
before changing any of `src/attributes/enrichers/{person,garments,age_gender}.py`,
`src/attributes/mivolo_loader.py`, or the `tools/fetch_person_*.py` /
`tools/eval_person_*.py` tools.

## The ask

The project tracked vehicles with plates and colour; the brother's side owns
storage/search/GPS. Vehicles had a `color` attribute already
([`src/attributes/enrichers/color.py`](../src/attributes/enrichers/color.py)).
People had none. The brother sent a list of candidate models (Sapiens,
Roboflow clothing, FashionNet/DeepFashion, Anny, farazBhatti body
measurements, body-measure, open-age-detection, OpenCV age/gender) and asked
for the best one(s), ideally one model covering many attributes so we
wouldn't need to install five different things — and it had to work on
**Indian** people specifically, since none of these were built or tested for
that.

## Round 1 — picking a starting model

Compared on accuracy, size/speed, licence, and whether they'd even resolve
on CCTV-sized people:

| Candidate | Verdict |
|---|---|
| **PP-Human PPLCNet_x1_0** (PaddleDetection) | **Chosen.** 26 attributes in one 7 MB model, Apache-2.0, ~9 ms/person on CPU (PA-100K claims 92.2% mA). Trained on real outdoor surveillance images (PA-100K), so the input distribution is closer to ours than a fashion dataset. |
| Sapiens (Meta) | CC-BY-NC — not usable commercially. 0.3–2B params, far too heavy for CPU anyway. |
| Roboflow clothing | Only localises garments (bounding boxes), doesn't classify attributes; Western clothing. |
| DeepFashion / FashionNet | Dataset is non-commercial research-only; built from shop photos, not CCTV. |
| Anny | A 3D parametric body *model*, not a measurement tool — needs a separate fitting method. |
| farazBhatti body measurements | Requires the user to type in the subject's real height as input — useless for a camera that doesn't know it. |
| body-measure | Needs a phone AR camera and two posed photos (front + side) — not compatible with a fixed CCTV feed. |
| open-age-detection | Apache-2.0 but needs a visible *face*; our people are ~18 px of face at typical CCTV distance. |
| OpenCV age/gender CNN | Old, face-only, weak on non-Western faces. |

Height: no open model measures height from a single 2D CCTV frame directly.
It needs pose keypoints (YOLOv8-pose, already in `ultralytics`) plus a
per-camera ground-plane calibration, à la `tools/calibrate.py` for lanes.
Deferred — not built yet.

**Getting the weights running was the hard part**, not picking them:
- `paddle2onnx` only converts cleanly with `paddlepaddle==2.6.2` +
  `paddle2onnx==1.3.1`, on Python 3.12 specifically — every other combination
  (a newer paddle2onnx demanding a newer paddle, or the project's own
  `paddlepaddle==3.3.1`) failed with a Windows DLL load error or a version
  rejection. `tools/fetch_person_attr.py` builds a throwaway venv just for
  the conversion step so the main project's paddle stays untouched.
- Verified the ONNX export against the original paddle inference engine:
  max difference `6.6e-07` — bit-identical for practical purposes.

## Round 1 field test — three CCTV clips, no ground truth

Built `tools/eval_person_attr.py` to run the model on a saved tracks CSV and
produce a contact sheet (crop + predicted labels) for a human to eyeball,
because **no labelled dataset of these specific clips exists** — accuracy
claims below are "I looked at N people and judged", not a measured metric.
Three clips, deliberately different:

1. `samples/indian_road.mp4` — the original clip. Traffic camera, people are
   almost all helmeted two-wheeler riders, ~136 px tall.
2. `samples/person_test.mp4` — downloaded CCTV footage (Turkish street,
   rainy, umbrellas, headscarves, elderly). Chosen because it's a *fixed
   CCTV camera*, unlike a handheld walking-tour video, and has people the
   original clip didn't (pedestrians, not just riders).
3. `samples/person_test2.mp4` — downloaded stock footage of a sunny
   lakeside walkway, people large and side-on (~330 px tall), summer
   clothes (shorts, skirts, caps, sunglasses).

**Result: the model reads *what the clothes look like* well (facing,
sleeves) and *who the person is* badly (gender, age).** Specifically, across
all three clips:

- **Facing direction, sleeve length** — consistently right.
- **Gender** — systematically wrong on: women in headscarves (read male at up
  to 0.97 confidence — Turkish and Indian footage both), and helmeted
  riders (helmet visor reads as short hair → male, even for women).
- **Age group** — essentially always "18–60"; never predicted "over 60" even
  for two visibly elderly men with canes. It has learned one answer.
- **Long coat** — 49/56 people on one clip flagged "yes" including men in
  short jackets and even bare-armed T-shirts; the model's idea of "coat"
  doesn't match a real coat on this footage.
- **Hat** — helmets and headscarves both trigger it.
- **Shorts/skirt** — only worked once people were large enough (lakeside
  clip); never predicted on the other two, plausibly because nobody wore
  them there, so it's untested rather than proven wrong.

**Why:** the model was trained on East Asian pedestrian datasets (PA-100K,
RAPv2, PETA) — few headscarves, riders, or elderly people. A confidence
threshold can't fix a confidently *wrong* answer; only better training data
or a different model can.

## Code fixes applied (in `person.py`)

None of these fix the model's judgement — they only stop *bad inputs* from
producing confident garbage, and stop us from *storing* the outputs known to
be wrong:

1. **Don't store what's proven wrong.** `gender`, `age_group`, `long_coat`,
   `holding`, upper/lower pattern, `boots` are still *decoded* (so a future
   caller can ask for them) but excluded from `DEFAULT_STORE`. Storing a
   confidently wrong value is worse than storing nothing — a downstream
   analysis can't tell a bad read from a good one.
2. **Reject bad crops before paying for inference:**
   - `edge_margin` — skip a person whose box touches the frame edge (a
     half-body silhouette is a guess, not a read).
   - `min_det_conf` — skip weak detections (a road sign and a camera
     watermark were both detected as "person" on the rainy clip and read
     garbage attributes).
   - `max_umbrella_cover` — skip a person more than 25% covered by a
     detected umbrella box (an umbrella read as a checked/patterned shirt on
     the rainy clip).
3. **`min_reads` before storing anything** — a single glimpse was the least
   reliable read in every clip; require 3 usable crops averaged before a
   track gets any stored value.
4. **Glasses threshold raised 0.3 → 0.6** — PP-Human's own pipeline code
   ships 0.3 for glasses, which on our footage let through 29/56 "yes"
   results that were almost all weak guesses (0.30–0.53 confidence).

Verified with a synthetic-boxes unit test (a real person crop, a box at the
edge, a weak-confidence box, a box under a synthetic umbrella, a too-small
box) exercising the enricher's `compute()`/`apply()` directly against a fake
`AnalysisView` — confirmed each rejection reason fires, and that only the
`DEFAULT_STORE` keys land in `store.vehicles[tid]["attrs"]`.

## Round 2 — researching replacements after the field test

The user asked: is this a code problem or a model problem? Answer: **model**.
Researched what could specifically fix gender/age and garment
classification, keeping the "doesn't need to be one model" freedom the user
gave.

| Job | Chosen | Why |
|---|---|---|
| Gender + age | **MiVOLO v2** (`iitolstykh/mivolo_v2`, Apache-2.0, 28.8M params) | Body-only inference (face optional, deliberately unused — see below), trained on ~800K photos across the LAGENDA/UTKFace/IMDB-clean family, reports usable body-only accuracy in its own paper. |
| Headwear / lower garment / bag / sunglasses | **SegFormer-B2 clothes** (`mattmdjaga/segformer_b2_clothes`, MIT) | Per-pixel semantic segmentation (18 ATR classes including a dedicated **Scarf** class, which is exactly what PP-Human conflates with Hat) rather than a single whole-crop yes/no vote — lets code derive an answer from pixel counts instead of trusting one opaque classifier output. |
| Facing / sleeves | Kept **PP-Human** | Already proven right in round 1. |

Rejected on this pass: LVFace/TopoFR-style face-recognition models (wrong
task — see face-lab below), OpenPAR/CLIP-prompt PAR variants (heavier than
justified, no clear accuracy win over AdaFace-class models for our images),
DINOv3-backed options (adds a second, more restrictive licence on top of
whatever model uses it).

**Why body-only, no face, for MiVOLO:** the product's own website copy says
"no facial recognition"; the footage rarely resolves a usable face anyway
(~18 px at typical distance); using the face input would contradict a public
claim for a gain that's usually unavailable. `faces_input` is deliberately
passed as `None`-derived zero tensors — see
[`src/attributes/mivolo_loader.py`](../src/attributes/mivolo_loader.py).

### Getting MiVOLO to load — the annoying part

`iitolstykh/mivolo_v2`'s HuggingFace `trust_remote_code` model file imports
the `mivolo` PyPI package, which pins `timm==0.8.13.dev0` (unreleased) and
`pip install`s from GitHub — which fails on Windows because the repo uses
symlinks. Solution, in `mivolo_loader.py`:
- Clone the repo directly (`core.symlinks=false`) instead of pip-installing
  it, and `sys.path.insert` it.
- Run on `timm==0.9.2` (a real release) with **three narrow compatibility
  shims**, none of which touch the cloned code:
  1. `timm.models._helpers.remap_checkpoint` — removed in later timm;
     stubbed to identity since we load safetensors, not the old checkpoint
     format that function was for.
  2. `timm.models._pretrained.split_model_name_tag` — removed; a 5-line
     re-implementation.
  3. `timm.models.volo.VOLO.__init__` — timm added a `pos_drop_rate`
     parameter *between* two existing positional arguments after MiVOLO's
     code was written; MiVOLO calls `super().__init__()` with all 21
     arguments positionally, so everything after the insertion point lands
     one slot late (`drop_path_rate` literally receives the norm-layer
     *class* as its value). The shim detects the 21-arg legacy call and
     re-inserts a `0.0` at the right position before delegating to the real
     `__init__`.

### SegFormer → ONNX

Also PyTorch/`transformers`-only on HuggingFace; exported once via
`tools/fetch_person_models.py` (traced through a thin wrapper that returns
`.logits` directly) so the runtime dependency is just `onnxruntime`, same as
the other two models. `onnxruntime` (batched) benchmarked ~6× faster than
eager PyTorch for this model at the sizes we use.

### Speed reality check

Both new models are far heavier than PP-Human. Measured on this machine (no
GPU), per person crop:

| Model | Speed | Notes |
|---|---|---|
| PP-Human | 8–20 ms, 1 thread | Trivial. |
| SegFormer garments | 400–520 ms, 4 threads | |
| MiVOLO age/gender | 1.3–1.7 s, 4 threads | |

**Decision: both ship `enabled: false` by default.** They cannot run on
every frame of a live feed; at best they run a handful of times per tracked
person, in a background queue, or need a GPU. `config.yaml` documents this
explicitly next to each block.

## Round 2 field test

Ran all three models side by side on the same crops via
`tools/eval_person_models.py`, on both downloaded clips plus, after consent
was obtained, on `samples/model_test.mp4` (a handheld Indian street video
the user shot themselves with people's consent — see
[`../outputs/person_models_model_test/notes.md`](../outputs/person_models_model_test/notes.md)
for the raw per-clip notes).

- **MiVOLO gender**: clearly better than PP-Human. ~21/24 correct per clip by
  eye vs. PP-Human's near-coinflip on women. Headscarf cases specifically
  improved (one case went from "male 0.82" to "female 0.99") but not fully
  solved — two headscarf cases were still wrong.
- **MiVOLO age**: now gives *years*, not a fixed 3-band guess. Several
  clearly-elderly people that PP-Human always called "18–60" now landed at
  46–56. Still under-reads some elderly people (one came out at 26, another
  at 34) and never predicted above 60 on the samples checked.
- **SegFormer**: real skirts correctly separated from trousers; caps read as
  hats; the **Scarf** class caught a genuine headscarf that PP-Human called
  "hat" on 1–2 of 4 checked cases. Weaknesses: shorts routinely
  under-detected as trousers (the "how much bare leg counts as shorts"
  threshold needs tuning against labelled data, not against the same clips
  used to judge it — deliberately left untuned); long jackets/loose shirts
  sometimes read as "dress"; in crowds it can bleed pixels from the person
  standing behind onto the target.

**No model has been validated against Indian ground-truth labels** — every
number above is "looked right/wrong by eye" on a handful of dozens of
people, not a measured accuracy on a labelled set. Explicitly flagged to the
user as the next real step (hand-label ~50 people, score for real) — not yet
done as of this writing.

## Config surface

Everything is switched through `perception.attributes.{person,garments,age_gender}`
in `config.yaml`, each block block-commented with what it does, its cost,
and its setup command. `person` is `enabled: true` by default (cheap, and
the crop-quality gates mean a bad read mostly gets *skipped* rather than
stored); `garments` and `age_gender` are `enabled: false` (expensive,
unvalidated on Indian faces). See
[`../src/attributes/registry.py`](../src/attributes/registry.py) —
`OPTIONAL` / `_load_optional` — for how the two heavy models are imported
only when their config block is switched on, so a machine without
torch/transformers installed never pays the import cost for a model nobody
enabled.

## Open items (not done, flagged for later)

- No Indian-labelled ground truth exists yet for any of the three models —
  the entire "works / doesn't work" judgement above is by-eye.
- SegFormer's shorts-vs-trousers bare-leg-fraction threshold (`0.35` in
  `garments.py`) is a guess, not fit to data.
- Height estimation (pose keypoints + per-camera calibration) — not started.
- `requirements.txt` documents but does not install the heavy-model
  dependencies (`transformers==4.51.0`, `timm==0.9.2`, `accelerate==1.8.1`)
  — left commented out on purpose so a default `pip install` stays light.
