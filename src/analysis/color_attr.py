"""Vehicle paint + person clothes as one analyzer (mirrors PlateAnalyzer).

Vehicles (group == "vehicle"): HSV paint color, best-conf wins across frames.
People (group == "person"): shirt (upper) + trousers (lower) colors, same
best-conf rule. One analyzer because the input (a crop) and the output
(attribute rows) are identical; only the key set differs per class.
Emits "color_read" / "clothes_read" once per track when first resolved.
"""

from ..attributes.base import run_attributes
from ..detectors.classes import PERSON, VEHICLE
from .base import register

MIN_VEHICLE_W = 40
MIN_VEHICLE_H = 40
MIN_PERSON_W = 20
MIN_PERSON_H = 60


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
            if det.group == VEHICLE:
                keys, need = ["color"], (MIN_VEHICLE_W, MIN_VEHICLE_H)
            elif det.group == PERSON:
                keys, need = ["clothes"], (MIN_PERSON_W, MIN_PERSON_H)
            else:
                continue
            if det.track_id is None:
                continue
            x1, y1, x2, y2 = det.bbox
            if (x2 - x1) < need[0] or (y2 - y1) < need[1]:
                continue
            obj = store.vehicles.get(det.track_id)
            if obj is None:
                continue
            crop = ctx.raw[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            before = tuple(obj.get("attrs", {}).get(k, "") for k in
                           (["color"] if keys == ["color"]
                            else ["upper_color", "lower_color"]))
            self.reads += 1
            attrs = run_attributes(keys, crop, obj)
            if keys == ["color"]:
                got, vals = "color", (attrs.get("color", ""),)
                evt = "color_read"
            else:
                got = "clothes"
                vals = (attrs.get("upper_color", ""), attrs.get("lower_color", ""))
                evt = "clothes_read"
            if any(vals):
                for k in (["color", "color_conf"] if keys == ["color"] else
                          ["upper_color", "upper_color_conf",
                           "lower_color", "lower_color_conf"]):
                    if attrs.get(k, "") != "":
                        det.extra[k] = attrs[k]
                if vals != before:
                    self.colors += 1
                    ctx.emit(evt, {k: attrs.get(k, "") for k in
                                   (["color"] if keys == ["color"] else
                                    ["upper_color", "lower_color"])},
                             track_id=det.track_id)

    def summary(self) -> dict:
        return {"color_attempts": self.reads, "colors_read": self.colors}


register("color", ColorAnalyzer)
