"""YOLO detector wrapper. Swap models via config with no code change."""

from ultralytics import YOLO

CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetector:
    def __init__(self, model_cfg: dict):
        self.cfg = model_cfg
        self.model = YOLO(model_cfg["name"])

    def track(self, frame, tracker_cfg: dict):
        """Detect + ByteTrack in one call. Returns ultralytics Results[0]."""
        m = self.cfg
        results = self.model.track(
            frame,
            persist=tracker_cfg.get("persist", True),
            tracker=tracker_cfg.get("name", "bytetrack.yaml"),
            conf=m["conf"], iou=m["iou"], classes=m["classes"],
            imgsz=m["imgsz"], verbose=False,
            device=m.get("device", "cpu"),
        )
        return results[0]

    @staticmethod
    def class_name(cls_id: int) -> str:
        return CLASS_NAMES.get(int(cls_id), str(cls_id))
