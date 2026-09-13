"""Garments by pixel: headwear, lower garment, sleeves, bag, sunglasses. EXPERIMENTAL.

SegFormer-B2 fine-tuned for clothes parsing on ATR (mattmdjaga/segformer_b2_clothes,
MIT). Unlike a whole-crop classifier it labels every pixel of the person with one
of 18 classes, so the questions the PP-Human model kept getting wrong become
arithmetic over pixel counts:

    headwear   Hat vs Scarf pixels in the head region. ATR has a SCARF class, so
               a headscarf or dupatta is no longer reported as a hat
    lower      Dress / Skirt / Pants, and shorts as Pants with much bare leg
               below them - ATR has no shorts class
    sleeves    bare-arm pixels vs upper-clothes pixels
    bag        Bag pixels as a share of the person
    sunglasses Sunglasses pixels vs face pixels (unreliable on small faces)

Cost is the catch: ~1.4 s per crop on one CPU thread, ~0.6 s on four, at
384x192. So it reads a person at most `max_reads` times (default 3) and is
disabled by default. Export the ONNX once:  python tools/fetch_person_models.py

Not trained on Indian clothing: a saree will usually parse as Dress or
Upper-clothes + Skirt. Test on Indian footage before trusting it there.
"""

import os

import numpy as np

from ...detectors.classes import PERSON
from ...runtime.plugin import Findings
from ..registry import block, register
from .person import UMBRELLA, CropRules

LABELS = ("background", "hat", "hair", "sunglasses", "upper", "skirt", "pants",
          "dress", "belt", "left_shoe", "right_shoe", "face", "left_leg",
          "right_leg", "left_arm", "right_arm", "bag", "scarf")
C = {name: i for i, name in enumerate(LABELS)}

INPUT_H, INPUT_W = 384, 192
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

KEYS = ("headwear", "lower", "sleeves", "bag", "sunglasses")
HEAD_ROWS = 0.30          # top share of the crop searched for headwear


def preprocess(crop_bgr) -> np.ndarray:
    import cv2
    img = cv2.resize(crop_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    return ((img - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def measure(label_map: np.ndarray) -> np.ndarray:
    """Per-read evidence vector: pixel counts per class, overall and head region.

    Kept as raw counts so several reads average cleanly before any decision.
    """
    n = len(LABELS)
    head = label_map[: max(1, int(label_map.shape[0] * HEAD_ROWS))]
    return np.concatenate([np.bincount(label_map.ravel(), minlength=n),
                           np.bincount(head.ravel(), minlength=n)]).astype(np.float32)


def decode(ev: np.ndarray) -> dict:
    """Averaged evidence -> {key: (value, conf)}. conf is the deciding share."""
    n = len(LABELS)
    body, head = ev[:n], ev[n:]
    person = max(1.0, body.sum() - body[C["background"]])

    head_px = max(1.0, head[[C["hat"], C["scarf"], C["hair"], C["face"],
                             C["sunglasses"]]].sum())
    hat, scarf = head[C["hat"]] / head_px, head[C["scarf"]] / head_px
    if max(hat, scarf) >= 0.20:
        headwear = ("scarf", scarf) if scarf >= hat else ("hat", hat)
    else:
        headwear = ("none", 1.0 - max(hat, scarf))

    dress, skirt, pants = body[C["dress"]], body[C["skirt"]], body[C["pants"]]
    legs = body[C["left_leg"]] + body[C["right_leg"]]
    garment = dress + skirt + pants
    if garment < 0.03 * person:
        lower = ("unknown", 0.0)
    elif dress >= max(skirt, pants):
        lower = ("dress", dress / garment)
    elif skirt >= pants:
        lower = ("skirt", skirt / garment)
    else:
        bare = legs / max(1.0, legs + pants)
        lower = ("shorts", bare) if bare >= 0.35 else ("trousers", 1.0 - bare)

    arms = body[C["left_arm"]] + body[C["right_arm"]]
    upper = body[C["upper"]] + dress
    bare_arm = arms / max(1.0, arms + upper)
    sleeves = ("short", bare_arm) if bare_arm >= 0.12 else ("long", 1.0 - bare_arm)

    bag_share = body[C["bag"]] / person
    bag = ("yes", min(1.0, bag_share / 0.04)) if bag_share >= 0.02 \
        else ("no", 1.0 - bag_share / 0.02)

    face = body[C["face"]] + body[C["sunglasses"]]
    sg = body[C["sunglasses"]] / max(1.0, face)
    if face < 40:
        sunglasses = ("unknown", 0.0)             # face too small to judge
    else:
        sunglasses = ("yes", sg) if sg >= 0.10 else ("no", 1.0 - sg)

    return {"headwear": headwear, "lower": lower, "sleeves": sleeves,
            "bag": bag, "sunglasses": sunglasses}


class GarmentSegModel:
    def __init__(self, path: str, threads: int = 4):
        import onnxruntime as ort
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found - run: python tools/fetch_person_models.py")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        self.session = ort.InferenceSession(path, opts,
                                            providers=["CPUExecutionProvider"])

    def label_map(self, crop_bgr) -> np.ndarray:
        logits = self.session.run(None, {"pixel_values": preprocess(crop_bgr)})[0]
        return logits[0].argmax(0).astype(np.int64)      # (96, 48) at 1/4 scale

    def evidence(self, crop_bgr) -> np.ndarray:
        return measure(self.label_map(crop_bgr))


class GarmentEnricher:
    name = "garments"
    concurrent = True

    def __init__(self, cfg: dict):
        conf = block(cfg, self.name)
        self.model_path = str(conf.get("model",
                                       "models/clothes_seg/segformer_b2_clothes.onnx"))
        self.rules = CropRules({"min_height": 120, "min_width": 40, **conf})
        self.read_every = max(1, int(conf.get("read_every", 6)))
        self.max_reads = max(1, int(conf.get("max_reads", 3)))
        self.min_reads = max(1, min(self.max_reads, int(conf.get("min_reads", 2))))
        self.threads = int(conf.get("threads", 4))
        self.model = None
        self._sum, self._n, self._seen, self._last = {}, {}, {}, {}
        self.reads = 0

    def setup(self, source, cfg: dict):
        self.model = GarmentSegModel(self.model_path, self.threads)

    def compute(self, view) -> Findings:
        found = Findings(analyzer=self.name)
        if view.raw is None or self.model is None:
            return found
        fh, fw = view.raw.shape[:2]
        umbrellas = [b.bbox for b in view.boxes if b.cls_name == UMBRELLA]
        for box in view.of_group(PERSON):
            tid = box.track_id
            if tid is None or self._n.get(tid, 0) >= self.max_reads:
                continue
            if self.rules.reject(box.bbox, box.conf, fw, fh, umbrellas):
                continue
            seen = self._seen[tid] = self._seen.get(tid, 0) + 1
            if (seen - 1) % self.read_every:
                continue
            x1, y1, x2, y2 = box.bbox
            crop = view.raw[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                ev = self.model.evidence(crop)
            except Exception:
                continue
            self.reads += 1
            self._sum[tid] = self._sum.get(tid, 0.0) + ev
            n = self._n[tid] = self._n.get(tid, 0) + 1
            if n < self.min_reads:
                continue
            read = decode(self._sum[tid] / n)
            extra = {}
            for key, (value, conf) in read.items():
                extra["garment_" + key] = value
                extra["garment_" + key + "_conf"] = round(float(conf), 3)
            found.set(tid, extra=extra)
            labels = {k: v for k, (v, _) in read.items()}
            if labels != self._last.get(tid):
                self._last[tid] = labels
                found.event("garments_read", labels, track_id=tid)
        return found

    def apply(self, ctx, findings: Findings):
        for tid, fields in findings.per_track.items():
            obj = ctx.store.vehicles.get(tid)
            if obj is not None:
                obj.setdefault("attrs", {}).update(fields.get("extra") or {})

    def forget(self, tid):
        for d in (self._sum, self._n, self._seen, self._last):
            d.pop(tid, None)

    def summary(self) -> dict:
        return {"garment_seg_reads": self.reads}


register("garments", GarmentEnricher)
