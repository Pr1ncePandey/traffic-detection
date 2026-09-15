"""Alignment to the standard 112x112 face crop, and cheap quality signals.

Every face is warped so its five landmarks land on the reference positions
AdaFace was trained with; that is what makes two crops of one person from
different frames comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Face

CROP_SIZE = 112
REFERENCE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                          [56.0252, 71.7366], [41.5493, 92.3655],
                          [70.7299, 92.2041]], dtype=np.float32)


@dataclass(frozen=True)
class FaceQuality:
    eye_px: float   # distance between the eyes in pixels: the face-size measure that matters
    yaw: float      # nose offset from the eye midpoint in eye-distances: ~0 frontal, >0.35 turned
    sharp: float    # variance of the Laplacian inside the face box: low = blurred


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
    """112x112 BGR crop, or None if the landmarks are degenerate."""
    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32),
                                            REFERENCE_112, method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(image, matrix, (CROP_SIZE, CROP_SIZE), borderValue=0)


def measure_quality(image: np.ndarray, face: Face) -> FaceQuality:
    lm = face.landmarks
    eye_px = float(np.linalg.norm(lm[1] - lm[0]))
    mid = (lm[0] + lm[1]) / 2.0
    yaw = float((lm[2][0] - mid[0]) / max(eye_px, 1.0))
    x1, y1, x2, y2 = face.bbox
    patch = image[y1:y2, x1:x2]
    sharp = 0.0
    if patch.size:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return FaceQuality(round(eye_px, 1), round(yaw, 3), round(sharp, 1))


def face_snapshot(image: np.ndarray, face: Face, max_side: int = 220) -> np.ndarray:
    """The face box plus some context around it, shrunk for saving."""
    x1, y1, x2, y2 = face.bbox
    h, w = image.shape[:2]
    pad_x, pad_y = int((x2 - x1) * 0.35), int((y2 - y1) * 0.35)
    crop = image[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)]
    scale = min(1.0, max_side / max(crop.shape[:2]))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                                 max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return crop.copy()
