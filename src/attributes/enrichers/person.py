"""Person attributes (facing, sleeves, lower garment, bag, hat, glasses) as an enricher.

One small model, 26 outputs: PP-Human's pedestrian attribute classifier
(PaddleDetection, PPLCNet_x1_0 backbone, StrongBaseline head, Apache-2.0;
trained on PA-100K + RAPv2 + PETA). Run through onnxruntime from a one-off
ONNX conversion, so it works on any interpreter that runs the rest of the
pipeline - paddle itself is not needed at runtime.

Fetch + convert once:  python tools/fetch_person_attr.py

WHAT IS STORED, AND WHY NOT EVERYTHING

Field-tested on three clips (Indian road, rainy street, lakeside walkway).
The model decodes 13 attributes, but only some deserve a database row:

    trusted     facing, sleeves                 right on all three clips
    low-trust   lower, bag, hat, glasses        right only in good conditions;
                                                filter on <key>_conf
    NOT stored  gender, age_group, long_coat,   confidently wrong: every woman
                holding, patterns, boots        in a headscarf read male at up
                                                to 0.97, elderly people read
                                                18-60, T-shirts read as coats

`perception.attributes.person.keys` changes the stored set. Storing a value
the model gets confidently wrong is worse than storing nothing: an analysis
downstream cannot tell it apart from a good one.

WHICH CROPS ARE READ

A read is only as good as its crop. Skipped before paying for inference:
  - smaller than min_height x min_width
  - touching the frame edge (a half-body is a guess, not a read)
  - detector confidence below min_det_conf
  - more than max_umbrella_cover of the person hidden by an umbrella box;
    COCO detects umbrellas, and an umbrella over the torso read as a "hat"
    and as a patterned shirt
And a track stores nothing until it has min_reads usable crops, because a
single look at a person was the least reliable read in every clip.

The model reads the body, never the face.
"""

import os

import numpy as np

from ...detectors.classes import PERSON
from ...runtime.plugin import Findings
from ..registry import block, register

# The model's output order, from the infer_cfg.yml shipped with the weights.
# NOTE: PaddleDetection's attr_infer.py names res[19:22] Less18/18-60/Over60,
# the reverse of this list. The shipped label_list matches PA-100K's own order
# (AgeOver60 first) and is what is used here.
LABELS = ("Hat", "Glasses", "ShortSleeve", "LongSleeve", "UpperStride",
          "UpperLogo", "UpperPlaid", "UpperSplice", "LowerStripe",
          "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress",
          "boots", "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
          "AgeOver60", "Age18-60", "AgeLess18", "Female", "Front", "Side",
          "Back")
IDX = {name: i for i, name in enumerate(LABELS)}

INPUT_H, INPUT_W = 256, 192
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Everything decode() can produce. Values are short lowercase strings so the
# EAV table stays queryable: WHERE key='bag' AND value='backpack'.
KEYS = ("gender", "age_group", "facing", "sleeves", "lower", "long_coat",
        "upper_pattern", "lower_pattern", "hat", "glasses", "bag", "holding",
        "boots")
# ...and what is stored by default. See the module docstring.
DEFAULT_STORE = ("facing", "sleeves", "lower", "bag", "hat", "glasses")

# Per-label yes/no thresholds. PP-Human ships glasses at 0.3; in the field
# test that let through 29 weak "yes" reads on one clip, nearly all wrong.
DEFAULT_THRESH = {"Glasses": 0.6, "HoldObjectsInFront": 0.6}

UMBRELLA = "umbrella"


def preprocess(crop_bgr) -> np.ndarray:
    """BGR crop -> 1x3x256x192 float32, exactly as infer_cfg.yml prescribes."""
    import cv2
    img = cv2.resize(crop_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0      # to RGB, scaled
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[None]


def decode(p, thresholds: dict | None = None) -> dict:
    """26 probabilities -> {key: (value, conf)}.

    conf is the probability of the value chosen, so a 50/50 call reads 0.5
    whichever way it went.
    """
    p = np.asarray(p, dtype=np.float32)
    thresh = {**DEFAULT_THRESH, **(thresholds or {})}

    def pick(options):                       # one-of-N: argmax over a group
        scores = [p[IDX[label]] for label, _ in options]
        k = int(np.argmax(scores))
        total = float(sum(scores)) or 1.0
        return options[k][1], float(scores[k]) / total

    def flag(label):                          # yes/no against its threshold
        s = float(p[IDX[label]])
        yes = s > thresh.get(label, 0.5)
        return ("yes" if yes else "no"), (s if yes else 1.0 - s)

    def pattern(options):                     # at most one, or "plain"
        best = max(options, key=lambda o: p[IDX[o[0]]])
        s = float(p[IDX[best[0]]])
        return (best[1], s) if s > 0.5 else ("plain", 1.0 - s)

    female = float(p[IDX["Female"]])
    out = {
        "gender": ("female", female) if female > 0.5 else ("male", 1.0 - female),
        "age_group": pick([("AgeLess18", "under_18"), ("Age18-60", "18_60"),
                           ("AgeOver60", "over_60")]),
        "facing": pick([("Front", "front"), ("Side", "side"), ("Back", "back")]),
        "sleeves": pick([("LongSleeve", "long"), ("ShortSleeve", "short")]),
        "lower": pick([("Trousers", "trousers"), ("Shorts", "shorts"),
                       ("Skirt&Dress", "skirt_dress")]),
        "long_coat": flag("LongCoat"),
        "upper_pattern": pattern([("UpperStride", "stripe"), ("UpperLogo", "logo"),
                                  ("UpperPlaid", "plaid"), ("UpperSplice", "splice")]),
        "lower_pattern": pattern([("LowerStripe", "stripe"),
                                  ("LowerPattern", "pattern")]),
        "hat": flag("Hat"),
        "glasses": flag("Glasses"),
        "holding": flag("HoldObjectsInFront"),
        "boots": flag("boots"),
    }
    bag_label = max(("HandBag", "ShoulderBag", "Backpack"), key=lambda b: p[IDX[b]])
    s = float(p[IDX[bag_label]])
    out["bag"] = ({"HandBag": "handbag", "ShoulderBag": "shoulder_bag",
                   "Backpack": "backpack"}[bag_label], s) if s > 0.5 else ("none", 1.0 - s)
    return out


def covered_by(bbox, others) -> float:
    """Largest fraction of `bbox`'s area hidden by any one box in `others`."""
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    worst = 0.0
    for ox1, oy1, ox2, oy2 in others:
        iw = min(x2, ox2) - max(x1, ox1)
        ih = min(y2, oy2) - max(y1, oy1)
        if iw > 0 and ih > 0:
            worst = max(worst, iw * ih / area)
    return worst


class CropRules:
    """Which person boxes are worth reading. Shared by the enricher and the
    evaluation tool, so what is measured is exactly what runs."""

    def __init__(self, conf: dict):
        self.min_h = int(conf.get("min_height", 80))
        self.min_w = int(conf.get("min_width", 30))
        self.edge_margin = int(conf.get("edge_margin", 4))
        self.min_det_conf = float(conf.get("min_det_conf", 0.35))
        self.max_umbrella_cover = float(conf.get("max_umbrella_cover", 0.25))

    def reject(self, bbox, det_conf, frame_w, frame_h, umbrellas=()) -> str:
        """'' when the crop is usable, else the reason it is not."""
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < self.min_w or (y2 - y1) < self.min_h:
            return "small"
        m = self.edge_margin
        if x1 <= m or y1 <= m or x2 >= frame_w - m or y2 >= frame_h - m:
            return "edge"
        if det_conf < self.min_det_conf:
            return "low_conf"
        if umbrellas and covered_by(bbox, umbrellas) > self.max_umbrella_cover:
            return "umbrella"
        return ""


class PersonAttrModel:
    """The ONNX session. Separate from the enricher so a tool can use it alone."""

    def __init__(self, path: str, threads: int = 1):
        import onnxruntime as ort
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found - run: python tools/fetch_person_attr.py")
        opts = ort.SessionOptions()
        # One thread per call: the stage already runs plugins in a pool, and
        # onnxruntime's own pool on top of it oversubscribes a laptop CPU.
        opts.intra_op_num_threads = max(1, int(threads))
        self.session = ort.InferenceSession(path, opts,
                                            providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def probs(self, crop_bgr) -> np.ndarray:
        y = self.session.run(None, {self.input: preprocess(crop_bgr)})[0][0]
        if y.min() < 0.0 or y.max() > 1.0:             # logits, not probabilities
            y = 1.0 / (1.0 + np.exp(-y))
        return y


class PersonEnricher:
    name = "person"
    concurrent = True

    def __init__(self, cfg: dict):
        conf = block(cfg, self.name)
        self.model_path = str(conf.get("model", "models/person_attr/person_attr.onnx"))
        self.rules = CropRules(conf)
        self.read_every = max(1, int(conf.get("read_every", 3)))
        self.max_reads = max(1, int(conf.get("max_reads", 8)))
        self.min_reads = max(1, min(self.max_reads, int(conf.get("min_reads", 3))))
        self.threads = int(conf.get("threads", 1))
        self.thresholds = dict(conf.get("thresholds") or {})
        keys = conf.get("keys")
        self.keys = tuple(k for k in (keys if keys else DEFAULT_STORE) if k in KEYS)
        unknown = sorted(set(keys or ()) - set(KEYS))
        if unknown:
            print(f"[person] unknown keys ignored: {', '.join(unknown)}; "
                  f"available: {', '.join(KEYS)}")
        self.model = None
        self._sum: dict = {}       # track_id -> summed probabilities
        self._n: dict = {}         # track_id -> reads so far
        self._seen: dict = {}      # track_id -> usable frames seen
        self._last: dict = {}      # track_id -> last reported labels
        self.reads = 0
        self.people = 0
        self.skipped: dict = {}    # reason -> count, for the run summary

    def setup(self, source, cfg: dict):
        self.model = PersonAttrModel(self.model_path, self.threads)

    def compute(self, view) -> Findings:
        found = Findings(analyzer=self.name)
        if view.raw is None or self.model is None:
            return found
        frame_h, frame_w = view.raw.shape[:2]
        umbrellas = [b.bbox for b in view.boxes if b.cls_name == UMBRELLA]
        for box in view.of_group(PERSON):
            tid = box.track_id
            if tid is None or self._n.get(tid, 0) >= self.max_reads:
                continue
            why = self.rules.reject(box.bbox, box.conf, frame_w, frame_h, umbrellas)
            if why:
                self.skipped[why] = self.skipped.get(why, 0) + 1
                continue
            seen = self._seen[tid] = self._seen.get(tid, 0) + 1
            # Spread the reads along the track rather than spending them all on
            # its first, often most distant, frames.
            if (seen - 1) % self.read_every:
                continue
            x1, y1, x2, y2 = box.bbox
            crop = view.raw[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                p = self.model.probs(crop)
            except Exception:
                continue                      # a bad crop must not stop the frame
            self.reads += 1
            self._sum[tid] = self._sum.get(tid, 0.0) + p
            n = self._n[tid] = self._n.get(tid, 0) + 1
            if n < self.min_reads:
                continue                      # not enough looks to say anything yet
            read = decode(self._sum[tid] / n, self.thresholds)
            extra = {}
            for key in self.keys:
                value, conf = read[key]
                extra[key] = value
                extra[key + "_conf"] = round(conf, 3)
            found.set(tid, extra=extra)
            labels = {k: read[k][0] for k in self.keys}
            if labels != self._last.get(tid):
                if tid not in self._last:
                    self.people += 1
                self._last[tid] = labels
                found.event("person_read", labels, track_id=tid)
        return found

    def apply(self, ctx, findings: Findings):
        """The one shared write: the object's durable attribute record."""
        store = ctx.store
        for tid, fields in findings.per_track.items():
            obj = store.tracks.get(tid)
            if obj is None:
                continue
            attrs = obj.setdefault("attrs", {})
            for key, value in (fields.get("extra") or {}).items():
                attrs[key] = value

    def forget(self, tid):
        for d in (self._sum, self._n, self._seen, self._last):
            d.pop(tid, None)

    def summary(self) -> dict:
        return {"person_attr_reads": self.reads, "people_described": self.people,
                "person_crops_skipped": dict(self.skipped)}


register("person", PersonEnricher)
