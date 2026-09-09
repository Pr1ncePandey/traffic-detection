"""YOLO detector + tracker wrapper, returning detector-agnostic Detections.

Two changes from the original:

1. Class names come from `model.names` instead of a hardcoded 4-entry dict, so
   all 80 COCO classes are named correctly and swapping to a custom model with
   its own labels needs no code change.
2. detect() returns a list of runtime.Detection rather than an ultralytics
   Results object, so analyzers and storage never import ultralytics. That is
   the seam that lets an open-vocabulary detector drop in later.

Detection and tracking stay fused in model.track() because that is how
ultralytics wires ByteTrack; `persist=True` is what carries ids across calls.
"""

from ultralytics import YOLO

from ..runtime.context import Detection
from .classes import group_of, resolve_classes


class Detector:
    def __init__(self, model_cfg: dict):
        self.cfg = dict(model_cfg or {})
        self.model = YOLO(self.cfg.get("name", "yolov8n.pt"))
        self.names = dict(self.model.names)
        # None here means "every class", which is what detecting anything on
        # the road requires. The old default of [2,3,5,7] is still accepted.
        self.classes = resolve_classes(self.cfg.get("classes"), self.names)
        self._warned = set()

    def class_name(self, cls_id: int) -> str:
        cid = int(cls_id)
        name = self.names.get(cid)
        if name is None:
            if cid not in self._warned:
                self._warned.add(cid)
                print(f"[detector] model reported unknown class id {cid}")
            return str(cid)
        return str(name)

    def track_raw(self, frame, tracker_cfg: dict):
        """Raw ultralytics Results[0]. Used by tools that need boxes directly."""
        m = self.cfg
        kwargs = dict(
            persist=(tracker_cfg or {}).get("persist", True),
            tracker=(tracker_cfg or {}).get("name", "bytetrack.yaml"),
            conf=m.get("conf", 0.3), iou=m.get("iou", 0.5),
            imgsz=m.get("imgsz", 640), verbose=False,
            device=m.get("device", "cpu"),
        )
        if self.classes is not None:
            kwargs["classes"] = self.classes
        return self.model.track(frame, **kwargs)[0]

    def detect(self, frame, tracker_cfg: dict) -> list:
        """Detect + track, normalized to Detections.

        A detection with no track id is still returned, with track_id=None:
        ByteTrack withholds ids for low-confidence boxes, and a frame-level
        analyzer (congestion, occupancy) still wants to count those.
        """
        result = self.track_raw(frame, tracker_cfg)
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = (boxes.id.cpu().numpy().astype(int)
               if boxes.id is not None else [None] * len(xyxy))
        out = []
        h, w = frame.shape[:2]
        for box, cls_id, conf, tid in zip(xyxy, clss, confs, ids):
            x1, y1, x2, y2 = (int(v) for v in box)
            # Clamp: ultralytics can return boxes a pixel or two outside the
            # frame, and a negative index silently produces an empty crop.
            x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
            x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
            name = self.class_name(cls_id)
            out.append(Detection(cls_id=int(cls_id), cls_name=name,
                                 group=group_of(name), conf=float(conf),
                                 bbox=(x1, y1, x2, y2),
                                 track_id=None if tid is None else int(tid)))
        return out

    def describe(self) -> str:
        scope = "all classes" if self.classes is None else f"{len(self.classes)} classes"
        return (f"{self.cfg.get('name')} conf={self.cfg.get('conf')} "
                f"imgsz={self.cfg.get('imgsz')} device={self.cfg.get('device', 'cpu')} "
                f"({scope})")


# The old name, kept so existing scripts and tools keep importing successfully.
VehicleDetector = Detector
