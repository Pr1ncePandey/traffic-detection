"""Face detection with YuNet (OpenCV Zoo, MIT).

YuNet returns a box and five landmarks (both eyes, nose tip, both mouth
corners) per face. It finds faces up to roughly 300 px across in its input,
so video frames are shrunk to a working width first, and photos - where a
selfie's face can be 1,000+ px - are searched at several shrinking widths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .errors import ModelLoadError, ModelNotFound

MODEL_FILE = Path("yunet") / "face_detection_yunet_2023mar.onnx"
PHOTO_WIDTHS = (1600, 960, 640, 480, 320)


@dataclass(frozen=True)
class Face:
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2 in full-frame pixels
    landmarks: np.ndarray                # (5, 2): image-left eye, image-right eye, nose,
                                         # image-left mouth corner, image-right mouth corner
    score: float

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def area(self) -> int:
        return self.width * (self.bbox[3] - self.bbox[1])


class FaceDetector:
    def __init__(self, models_dir: Path, width: int = 1280, min_score: float = 0.8):
        path = Path(models_dir) / MODEL_FILE
        if not path.exists():
            raise ModelNotFound(path)
        try:
            self._net = cv2.FaceDetectorYN.create(str(path), "", (320, 320),
                                                  score_threshold=float(min_score),
                                                  nms_threshold=0.3, top_k=5000)
        except cv2.error as exc:
            raise ModelLoadError(f"could not load face detector {path}: {exc}") from exc
        self.width = int(width)
        self._input_size = None

    def detect(self, frame: np.ndarray, width: int | None = None) -> list[Face]:
        """Faces in a BGR image, coordinates in the image's own pixels."""
        if frame is None or frame.ndim != 3 or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        scale = min(1.0, (width or self.width) / w)
        work = frame if scale >= 1.0 else cv2.resize(
            frame, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA)
        size = (work.shape[1], work.shape[0])
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size
        _, rows = self._net.detect(work)
        if rows is None:
            return []
        faces = []
        for row in rows:
            row = row.astype(np.float32)
            row[:14] /= scale
            x, y, bw, bh = row[:4]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append(Face((x1, y1, x2, y2), row[4:14].reshape(5, 2).copy(), float(row[14])))
        return faces

    def detect_in_photo(self, image: np.ndarray) -> list[Face]:
        """Search a photo at shrinking widths until a face is found."""
        w = image.shape[1]
        tried = set()
        for width in PHOTO_WIDTHS:
            effective = min(width, w)
            if effective in tried:
                continue
            tried.add(effective)
            faces = self.detect(image, effective)
            if faces:
                return faces
        return []
