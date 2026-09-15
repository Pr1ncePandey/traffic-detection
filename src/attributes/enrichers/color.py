"""Vehicle paint + person clothing as one attribute enricher. Three-phase.

Was src/analysis/color_attr.py. One enricher because the input (a crop) and the
output (attribute rows) are identical; only the key set differs per class.

Vehicles (group == "vehicle"): HSV paint colour.
People (group == "person"): shirt (upper) + trousers (lower).

Both keep the BEST-CONFIDENCE reading across the track's frames, not the first
or the last: a vehicle's colour reads badly while it is small and distant. That
running best used to be kept in the TrackStore's vehicle dict, which made the
extraction a shared write; it lives in self._best now, so compute() is pure.
"""

from ...detectors.classes import PERSON, VEHICLE
from ...runtime.plugin import Findings
from ..color import extract_color_conf, garment_colors
from ..registry import register

MIN_VEHICLE_W = 40
MIN_VEHICLE_H = 40
MIN_PERSON_W = 20
MIN_PERSON_H = 60

# group -> (attribute keys, minimum crop size, event name)
_VEHICLE_KEYS = ("color",)
_PERSON_KEYS = ("upper_color", "lower_color")


class ColorEnricher:
    name = "color"
    concurrent = True

    def __init__(self, cfg: dict):
        self._cfg = cfg or {}
        self._best: dict = {}       # track_id -> {key: (value, conf)}
        self.reads = 0
        self.colors = 0

    def setup(self, source, cfg: dict):
        pass

    def _read(self, group, crop) -> dict:
        """key -> (value, conf) for this crop. Pure, no state."""
        if group == VEHICLE:
            name, conf = extract_color_conf(crop)
            return {"color": (name, float(conf))} if name else {}
        g = garment_colors(crop)
        return {k: (g[k], float(g[k + "_conf"])) for k in _PERSON_KEYS if g[k]}

    def compute(self, view) -> Findings:
        found = Findings(analyzer=self.name)
        if view.raw is None:
            return found
        for box in view.boxes:
            if box.track_id is None:
                continue
            if box.group == VEHICLE:
                keys, need, evt = _VEHICLE_KEYS, (MIN_VEHICLE_W, MIN_VEHICLE_H), "color_read"
            elif box.group == PERSON:
                keys, need, evt = _PERSON_KEYS, (MIN_PERSON_W, MIN_PERSON_H), "clothes_read"
            else:
                continue
            x1, y1, x2, y2 = box.bbox
            if (x2 - x1) < need[0] or (y2 - y1) < need[1]:
                continue
            crop = view.raw[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            self.reads += 1
            try:
                read = self._read(box.group, crop)
            except Exception:
                continue                     # a bad crop must not stop the frame
            if not read:
                continue
            best = self._best.setdefault(box.track_id, {})
            before = tuple(best.get(k, ("", 0.0))[0] for k in keys)
            for key, (value, conf) in read.items():
                # >= so a later, equally confident read still refreshes - the
                # original best-conf rule, unchanged.
                if value and conf >= best.get(key, ("", 0.0))[1]:
                    best[key] = (value, round(conf, 3))
            now = tuple(best.get(k, ("", 0.0))[0] for k in keys)
            if not any(now):
                continue
            extra = {}
            for key in keys:
                value, conf = best.get(key, ("", 0.0))
                if value:
                    extra[key] = value
                    extra[key + "_conf"] = conf
            found.set(box.track_id, extra=extra)
            if now != before:
                self.colors += 1
                found.event(evt, {k: best.get(k, ("", 0.0))[0] for k in keys},
                            track_id=box.track_id)
        return found

    def apply(self, ctx, findings: Findings):
        """The one shared write: the object's durable attribute record."""
        store = ctx.store
        for tid, fields in findings.per_track.items():
            track = store.tracks.get(tid)
            if track is None:
                continue
            attrs = track.setdefault("attrs", {})
            for key, value in (fields.get("extra") or {}).items():
                attrs[key] = value

    def forget(self, tid):
        self._best.pop(tid, None)

    def summary(self) -> dict:
        return {"color_attempts": self.reads, "colors_read": self.colors}


register("color", ColorEnricher)
