"""Number-plate reading as an analyzer.

Gated on group == "vehicle", which is the point of the class taxonomy: with
all 80 COCO classes now detected, running OCR on every detection would put a
plate detector over people, dogs and traffic lights for nothing.

Read strategy (which frames, how many, voting) stays in attributes/plate.py so
it remains driven by the `plate:` config block.
"""

from ..attributes.base import run_attributes
from ..attributes.plate import configure as configure_plate
from ..detectors.classes import VEHICLE
from .base import register

MIN_CROP_W = 80        # below this OCR cannot succeed; skip before paying for it
MIN_CROP_H = 20


class PlateAnalyzer:
    name = "anpr"

    def __init__(self, cfg: dict):
        self._cfg = cfg or {}
        self.reads = 0
        self.plates = 0

    def setup(self, source, cfg: dict):
        # configure() owns its own defaults and coercion, so a new plate knob
        # in config.yaml never means editing this file.
        configure_plate(**(cfg.get("plate", {}) or {}))

    def process(self, ctx):
        store = ctx.store
        for det in ctx.detections:
            if det.group != VEHICLE or det.track_id is None:
                continue
            x1, y1, x2, y2 = det.bbox
            if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
                continue
            vehicle = store.vehicles.get(det.track_id)
            if vehicle is None:
                continue
            crop = ctx.raw[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            before = vehicle.get("attrs", {}).get("plate_number", "")
            self.reads += 1
            attrs = run_attributes(["plate"], crop, vehicle)
            plate = attrs.get("plate_number", "")
            if plate:
                det.extra["plate_number"] = plate
                det.extra["plate_conf"] = attrs.get("plate_conf", 0.0)
                if plate != before:
                    self.plates += 1
                    ctx.emit("plate_read",
                             {"plate": plate, "conf": attrs.get("plate_conf", 0.0)},
                             track_id=det.track_id)

    def summary(self) -> dict:
        return {"ocr_attempts": self.reads, "plates_read": self.plates}


register("anpr", PlateAnalyzer)
