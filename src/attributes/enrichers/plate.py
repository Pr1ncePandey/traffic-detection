"""Number-plate reading as an attribute enricher. Three-phase, concurrent.

Was src/analysis/anpr.py, where it sat in the same list as counting and lanes
and therefore ran AFTER them. It belongs in perception: a plate is a fact about
an object, not a conclusion about traffic.

This is the one plugin where parallelism actually pays - plate detection plus
OCR is tens of milliseconds of ONNX per frame, and it releases the GIL. What
used to prevent that was not the phase split but WHERE THE STATE LIVED: reads
accumulated in `store.vehicles[tid]`, so OCR mutated shared tracker state. The
reads now live here, in `self._reads`, and `apply()` does the one shared write.

Gated on group == "vehicle", which is the point of the class taxonomy: with all
80 COCO classes detected, running a plate detector over people, dogs and
traffic lights would cost real time for nothing.
"""

from ...detectors.classes import VEHICLE
from ...runtime.plugin import Findings
from ..plate import configure as configure_plate
from ..plate import new_read_state, read_plate_tracked
from ..registry import block, register

MIN_CROP_W = 80        # below this OCR cannot succeed; skip before paying for it
MIN_CROP_H = 20


class PlateEnricher:
    name = "plate"
    # compute() reads the immutable view, writes only to self._reads/_last, and
    # touches neither the TrackStore nor the canvas.
    concurrent = True

    def __init__(self, cfg: dict):
        self._cfg = cfg or {}
        self._reads: dict = {}      # track_id -> caller-owned read state
        self._last: dict = {}       # track_id -> last consensus reported
        self.reads = 0
        self.plates = 0

    def setup(self, source, cfg: dict):
        # configure() owns its own defaults and coercion, so a new plate knob in
        # config never means editing this file.
        conf = dict(block(cfg, self.name))
        conf.pop("enabled", None)      # a stage concern, not a reader knob
        configure_plate(**conf)

    def compute(self, view) -> Findings:
        found = Findings(analyzer=self.name)
        if view.raw is None:
            return found
        for box in view.of_group(VEHICLE):
            tid = box.track_id
            if tid is None:
                continue
            x1, y1, x2, y2 = box.bbox
            if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
                continue
            crop = view.raw[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            state = self._reads.setdefault(tid, new_read_state())
            self.reads += 1
            plate, conf = read_plate_tracked(crop, state)
            if not plate:
                continue
            found.set(tid, extra={"plate_number": plate, "plate_conf": conf})
            # The consensus is still being voted as more reads arrive, so the
            # string can legitimately change; report each change once.
            if plate != self._last.get(tid):
                self._last[tid] = plate
                self.plates += 1
                found.event("plate_read", {"plate": plate, "conf": conf},
                            track_id=tid)
        return found

    def apply(self, ctx, findings: Findings):
        """The one shared write: the object's durable attribute record.

        pipeline._finalize persists from store.vehicles[tid]["attrs"], so this
        is what makes a plate outlive the track that read it.
        """
        store = ctx.store
        for tid, fields in findings.per_track.items():
            extra = fields.get("extra") or {}
            vehicle = store.vehicles.get(tid)
            if vehicle is None or "plate_number" not in extra:
                continue
            attrs = vehicle.setdefault("attrs", {})
            attrs["plate_number"] = extra["plate_number"]
            attrs["plate_conf"] = round(float(extra.get("plate_conf", 0.0)), 3)

    def forget(self, tid):
        """Free the reads for a retired track - unbounded growth on a 24/7 feed."""
        self._reads.pop(tid, None)
        self._last.pop(tid, None)

    def summary(self) -> dict:
        return {"ocr_attempts": self.reads, "plates_read": self.plates}


register("plate", PlateEnricher)
