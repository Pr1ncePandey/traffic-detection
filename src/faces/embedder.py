"""AdaFace IR101 (WebFace12M) on onnxruntime: 112x112 BGR crop -> 512-d unit vector.

Why AdaFace: compared with OpenCV SFace and InsightFace ArcFace on
model_test.mp4 it had the lowest error rate and was clearly the best on small
faces - see docs/face-recognition.md. Weights: code MIT, trained on a research-only dataset.

Cosine similarity between two embeddings (their dot product) is ~1 for the
same face and ~0 for unrelated faces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .align import CROP_SIZE
from .errors import DependencyError, ModelLoadError, ModelNotFound

MODEL_NAME = "adaface_ir101_webface12m"
MODEL_FILE = Path("adaface_ir101") / "adaface_ir101.onnx"
EMBEDDING_DIM = 512


class FaceEmbedder:
    def __init__(self, models_dir: Path, threads: int = 4, batch: int = 16):
        path = Path(models_dir) / MODEL_FILE
        if not path.exists():
            raise ModelNotFound(path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DependencyError("onnxruntime is needed: pip install onnxruntime") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        try:
            self._session = ort.InferenceSession(str(path), options,
                                                 providers=["CPUExecutionProvider"])
        except Exception as exc:  # onnxruntime raises several unrelated types
            raise ModelLoadError(f"could not load {path}: {exc}\n"
                                 f"  fix: delete it and run python tools/fetch_face_models.py") from exc
        self._input = self._session.get_inputs()[0].name
        self.batch = max(1, int(batch))

    @staticmethod
    def _prepare(crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.shape != (CROP_SIZE, CROP_SIZE, 3):
            raise ValueError(f"expected an aligned {CROP_SIZE}x{CROP_SIZE} BGR crop, "
                             f"got {None if crop_bgr is None else crop_bgr.shape}")
        rgb = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0      # model expects RGB
        return ((rgb - 0.5) / 0.5).transpose(2, 0, 1)

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """(n, 512) float32, each row unit length."""
        if not len(crops):
            return np.zeros((0, EMBEDDING_DIM), np.float32)
        out = []
        for start in range(0, len(crops), self.batch):
            batch = np.stack([self._prepare(c) for c in crops[start:start + self.batch]])
            vectors = self._session.run(None, {self._input: batch.astype(np.float32)})[0]
            if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
                raise ModelLoadError(f"model returned shape {vectors.shape}, expected (n, {EMBEDDING_DIM})")
            out.append(vectors)
        vectors = np.vstack(out).astype(np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
