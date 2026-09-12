"""Zone counting as an analyzer. Wraps the existing ZoneCounter drawing and
TrackStore.zone_crossing logic - the counting rules are unchanged, only their
packaging.
"""

from ..detectors.classes import VEHICLE
from .base import block, register
from .line_counter import ZoneCounter


class CountingAnalyzer:
    name = "counting"

    def __init__(self, cfg: dict):
        self._cfg = cfg or {}
        conf = block(cfg, self.name)
        self.zone = None
        self.vehicles_only = bool(conf.get("vehicles_only", True))
        self.a_to_b = 0
        self.b_to_a = 0

    def setup(self, source, cfg: dict):
        # The zone lines are a property of THIS analysis, not of the camera's
        # identity, so they live in analyses.counting.zones rather than in the
        # camera block they used to share with the camera's name.
        self.zone = ZoneCounter(block(cfg, self.name).get("zones", {}),
                                source.height)

    def process(self, ctx):
        store = ctx.store
        for det in ctx.detections:
            if det.track_id is None:
                continue
            if self.vehicles_only and det.group != VEHICLE:
                continue
            _, cy = det.centroid
            event = store.zone_crossing(det.track_id, cy, self.zone.line_a_y,
                                        self.zone.line_b_y, self.zone.enabled)
            if event:
                det.event = event
                ctx.emit("crossing", {"direction": event, "cls": det.cls_name},
                         track_id=det.track_id)
        self.a_to_b, self.b_to_a = store.a_to_b, store.b_to_a
        self.zone.draw(ctx.annotated, store.a_to_b, store.b_to_a)

    def summary(self) -> dict:
        return {"zones_enabled": bool(self.zone and self.zone.enabled),
                "a_to_b": self.a_to_b, "b_to_a": self.b_to_a}


register("counting", CountingAnalyzer)
