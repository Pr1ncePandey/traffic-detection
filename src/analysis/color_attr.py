"""Vehicle color as an analyzer (mirrors PlateAnalyzer).

Gated on group == "vehicle": paint color is only meaningful for vehicles.
Cheap (HSV histogram, no model), so it runs on every detection with a big
enough crop; best-conf wins across frames (see attributes/base.py).
Emits "color_read" once per track when the color first resolves.
"""

from ..attributes.base import run_attributes
from ..detectors.classes import VEHICLE
from .base import register

MIN_CROP_W = 40
MIN_CROP_H = 40


class ColorAnalyzer:
    name = "color"

    def __init__(self, cfg: dict):
        self._cfg = cfg or {}
        self.reads = 0
        self.colors = 0

    def setup(self, source, cfg: dict):
        pass

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
            before = vehicle.get("attrs", {}).get("color", "")
            self.reads += 1
            attrs = run_attributes(["color"], crop, vehicle)
            color = attrs.get("color", "")
            if color:
                det.extra["color"] = color
                det.extra["color_conf"] = attrs.get("color_conf", 0.0)
                if color != before:
                    self.colors += 1
                    ctx.emit("color_read",
                             {"color": color, "conf": attrs.get("color_conf", 0.0)},
                             track_id=det.track_id)

    def summary(self) -> dict:
        return {"color_attempts": self.reads, "colors_read": self.colors}


register("color", ColorAnalyzer)
