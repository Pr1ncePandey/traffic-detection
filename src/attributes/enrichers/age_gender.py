"""Apparent age (years) and gender from the body. EXPERIMENTAL.

MiVOLO v2 (iitolstykh/mivolo_v2, Apache-2.0), body-only: the face input is left
empty, because the footage rarely resolves a face and the product does not do
facial analysis. MiVOLO was trained with body crops and reports usable
body-only accuracy; it replaces PP-Human's gender/age, which read every woman
in a headscarf as male and every elderly person as 18-60.

Stored per person: age_years (mean over reads), age_band, gender, gender_conf.
Apparent, estimated values - not identity, and not verified on Indian footage.

Heavy: ~29M parameters, a few hundred ms per read on CPU, so at most
`max_reads` (default 3) reads per person and disabled by default.
Setup: python tools/fetch_person_models.py
"""

import numpy as np

from ...detectors.classes import PERSON
from ...runtime.plugin import Findings
from ..registry import block, register
from .person import UMBRELLA, CropRules

BANDS = ((13, "child"), (18, "teen"), (30, "18_29"), (45, "30_44"),
         (60, "45_59"), (200, "60_plus"))


def age_band(age: float) -> str:
    for upper, name in BANDS:
        if age < upper:
            return name
    return BANDS[-1][1]


def summarise(reads) -> dict:
    """[(age, p_female), ...] -> stored fields."""
    ages = np.array([a for a, _ in reads], dtype=np.float32)
    female = float(np.mean([f for _, f in reads]))
    age = float(ages.mean())
    return {"age_years": int(round(age)),
            "age_spread": round(float(ages.std()), 1),
            "age_band": age_band(age),
            "gender": "female" if female >= 0.5 else "male",
            "gender_conf": round(female if female >= 0.5 else 1.0 - female, 3)}


class AgeGenderEnricher:
    name = "age_gender"
    concurrent = True

    def __init__(self, cfg: dict):
        conf = block(cfg, self.name)
        self.rules = CropRules({"min_height": 100, "min_width": 35, **conf})
        self.read_every = max(1, int(conf.get("read_every", 6)))
        self.max_reads = max(1, int(conf.get("max_reads", 3)))
        self.min_reads = max(1, min(self.max_reads, int(conf.get("min_reads", 2))))
        self.threads = int(conf.get("threads", 4))
        self.model = self.processor = self.config = None
        self._reads, self._seen, self._last = {}, {}, {}
        self.calls = 0

    def setup(self, source, cfg: dict):
        from ..mivolo_loader import load_mivolo
        self.model, self.processor, self.config = load_mivolo(self.threads)

    def compute(self, view) -> Findings:
        from ..mivolo_loader import predict
        found = Findings(analyzer=self.name)
        if view.raw is None or self.model is None:
            return found
        fh, fw = view.raw.shape[:2]
        umbrellas = [b.bbox for b in view.boxes if b.cls_name == UMBRELLA]
        batch = []
        for box in view.of_group(PERSON):
            tid = box.track_id
            if tid is None or len(self._reads.get(tid, ())) >= self.max_reads:
                continue
            if self.rules.reject(box.bbox, box.conf, fw, fh, umbrellas):
                continue
            seen = self._seen[tid] = self._seen.get(tid, 0) + 1
            if (seen - 1) % self.read_every:
                continue
            x1, y1, x2, y2 = box.bbox
            crop = view.raw[y1:y2, x1:x2]
            if crop.size:
                batch.append((tid, crop))
        if not batch:
            return found
        try:
            results = predict(self.model, self.processor, self.config,
                              [c for _, c in batch])     # one batched call per frame
        except Exception:
            return found
        self.calls += 1
        for (tid, _), res in zip(batch, results):
            reads = self._reads.setdefault(tid, [])
            reads.append(res)
            if len(reads) < self.min_reads:
                continue
            fields = summarise(reads)
            found.set(tid, extra=fields)
            labels = (fields["gender"], fields["age_band"])
            if labels != self._last.get(tid):
                self._last[tid] = labels
                found.event("age_gender_read", {"gender": labels[0],
                                                "age_band": labels[1]}, track_id=tid)
        return found

    def apply(self, ctx, findings: Findings):
        for tid, fields in findings.per_track.items():
            obj = ctx.store.tracks.get(tid)
            if obj is not None:
                obj.setdefault("attrs", {}).update(fields.get("extra") or {})

    def forget(self, tid):
        for d in (self._reads, self._seen, self._last):
            d.pop(tid, None)

    def summary(self) -> dict:
        return {"age_gender_batches": self.calls}


register("age_gender", AgeGenderEnricher)
