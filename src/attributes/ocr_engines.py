"""Selectable OCR backends. Swap models via `plate.ocr_backend` in config.yaml.

Every backend returns ONE already-assembled string plus a confidence, because
the engines disagree about what they hand back:

  rapidocr     generic detector + recogniser. Returns N text regions, so this
               module stitches them top-to-bottom. A two-row plate arrives as
               two regions and keeping only the best one drops a row - that
               was the old bug.
  fast_plate   plate-specialised recogniser. One string, plus per-character
               confidences. Nothing to stitch.
  paddle_anpr  PP-OCRv5 finetuned on Indian plates. One string, and it handles
               two-row layouts inside the model. Nothing to stitch.

Normalisation and format correction happen afterwards in plate_format, so they
apply identically whichever engine is selected.

Backends are built lazily and any missing dependency degrades to ("", 0.0)
after printing one line, so the pipeline still runs for vehicle counting with
no OCR extra installed.
"""

import os

import cv2
import numpy as np

from . import plate_format

# Where paddle_anpr looks for its weights and for a PaddleOCR checkout.
ANPR_DIR = os.path.join("models", "anpr_ocr")
PADDLEOCR_DIR = os.path.join("models", "PaddleOCR")


class OcrEngine:
    """What an OCR backend must implement.

    Same shape as DbStoreInterface in src/storage/csv_store.py: a documented
    contract, not an enforced ABC.
    """

    name = "base"

    # Only a backend running its own text detector benefits from a quiet
    # margin around the plate; a pure recogniser resizes internally, so
    # padding just distorts the aspect ratio it was trained on.
    wants_padding = False

    def read(self, plate_bgr):
        """Return (raw_text, conf). Raw = un-normalised, rows already joined."""
        raise NotImplementedError


def _rapidocr_regions(engine, img):
    """Normalise both rapidocr APIs to [(box, text, conf), ...].

    `rapidocr-onnxruntime` 1.x returns (regions, elapse); the 2.x/3.x
    `rapidocr` rewrite returns an object exposing .boxes/.txts/.scores.
    Someone who pip installs the newer package would otherwise land on an
    index error here, so accept both shapes.
    """
    out = engine(img)
    if isinstance(out, tuple):
        return [(r[0], r[1], float(r[2])) for r in (out[0] or [])]
    boxes = getattr(out, "boxes", None)
    if boxes is None:
        return []
    return list(zip(boxes, out.txts, [float(s) for s in out.scores]))


class RapidOcrEngine(OcrEngine):
    """Generic scene-text OCR. Baseline: not plate-specialised."""

    name = "rapidocr"
    wants_padding = True

    def __init__(self, model: str = "", join_rows: bool = True):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            from rapidocr import RapidOCR  # 2.x/3.x renamed the package
        self._ocr = RapidOCR()
        self._join_rows = join_rows

    def read(self, plate_bgr):
        out = _rapidocr_regions(self._ocr, plate_bgr)
        if not out:
            return ("", 0.0)
        if not self._join_rows:
            best = max(out, key=lambda r: float(r[2]))
            return (best[1], float(best[2]))
        # Each region is (box, text, conf) with box as 4 points. Hand the
        # centre and height to join_rows so it can band rows by y and order
        # fragments by x inside each row.
        parts, confs = [], []
        for box, text, conf in out:
            try:
                ys = [float(p[1]) for p in box]
                xs = [float(p[0]) for p in box]
                cy, cx = sum(ys) / len(ys), sum(xs) / len(xs)
                height = max(ys) - min(ys)
            except Exception:
                cy, cx, height = 0.0, 0.0, 1.0
            parts.append((cy, cx, height, text))
            confs.append(float(conf))
        # Weakest region governs: one unreadable row makes the whole plate
        # untrustworthy, which is what min_conf is meant to reject.
        return (plate_format.join_rows(parts), min(confs) if confs else 0.0)


class FastPlateEngine(OcrEngine):
    """Plate-specialised CCT recogniser (ONNX, CPU). Expects a cropped plate."""

    name = "fast_plate"
    default_model = "cct-s-v2-global-model"

    def __init__(self, model: str = "", join_rows: bool = True):
        from fast_plate_ocr import LicensePlateRecognizer
        self._model_name = model or self.default_model
        self._ocr = LicensePlateRecognizer(self._model_name, device="cpu")

    def read(self, plate_bgr):
        # This model's plate config declares image_color_mode="rgb". Only the
        # file-path route inside fast_plate_ocr does the BGR->RGB conversion;
        # the numpy route just resizes, so an unconverted cv2 crop would feed
        # swapped channels to the model and quietly lose accuracy.
        rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
        pred = self._ocr.run_one(rgb, return_confidence=True)
        text = pred.plate or ""
        conf = 1.0
        if pred.char_probs is not None:
            probs = np.asarray(pred.char_probs, dtype=float).ravel()
            # Per-character confidences: take the weakest character, since one
            # wrong character makes the whole plate wrong.
            if probs.size:
                conf = float(probs.min())
        return (text, conf)


class PaddleAnprEngine(OcrEngine):
    """PP-OCRv5 (PPHGNetV2_B4 + MultiHead) finetuned on Indian plates.

    Mirrors the model's own test.py, which is an argparse main() with no
    reusable class, so the ~40 lines that matter are reproduced here instead of
    imported. Needs paddlepaddle, safetensors, and a PaddleOCR checkout for
    model construction only.
    """

    name = "paddle_anpr"
    image_shape = (3, 48, 320)

    MODEL_CONFIG = {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",
        "Transform": None,
        "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
        "Head": {
            "name": "MultiHead",
            "out_channels_list": {
                "CTCLabelDecode": 64,     # 63 dict chars + blank
                "NRTRLabelDecode": 67,    # NRTRHead adds +1 internally
            },
            "head_list": [
                {"CTCHead": {"Neck": {"name": "svtr", "dims": 120, "depth": 2,
                                       "hidden_dims": 120, "kernel_size": [1, 3],
                                       "use_guide": True},
                             "Head": {"fc_decay": 1e-05}}},
                {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
            ],
        },
    }

    def __init__(self, model: str = "", join_rows: bool = True):
        weights = os.path.join(ANPR_DIR, "model.safetensors")
        char_dict = os.path.join(ANPR_DIR, "en_dict.txt")
        missing = [p for p in (weights, char_dict) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "missing " + ", ".join(missing) + ". Fetch with:\n"
                f"    mkdir -p {ANPR_DIR} && cd {ANPR_DIR} && \\\n"
                "    curl -LO https://huggingface.co/Awiros/anpr-ocr/resolve/main/model.safetensors && \\\n"
                "    curl -LO https://huggingface.co/Awiros/anpr-ocr/resolve/main/en_dict.txt"
            )
        self._ensure_ppocr()

        import paddle
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process
        from safetensors.numpy import load_file

        paddle.set_device("cpu")  # paddle on macOS is CPU-only anyway
        self._paddle = paddle
        self._post = build_post_process({"name": "CTCLabelDecode",
                                         "character_dict_path": char_dict,
                                         "use_space_char": True})
        self._model = build_model(dict(self.MODEL_CONFIG))
        self._model.eval()
        self._model.set_state_dict(
            {k: paddle.to_tensor(v) for k, v in load_file(weights).items()})

    @staticmethod
    def _ensure_ppocr():
        """Put a PaddleOCR checkout on sys.path; it builds the architecture."""
        import sys
        for candidate in (PADDLEOCR_DIR, "PaddleOCR", os.getcwd()):
            if os.path.isfile(os.path.join(candidate, "ppocr", "__init__.py")):
                if candidate not in sys.path:
                    sys.path.insert(0, candidate)
                return
        raise FileNotFoundError(
            f"PaddleOCR checkout not found. Clone it with:\n"
            f"    git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git {PADDLEOCR_DIR}")

    def _preprocess(self, img_bgr):
        import cv2
        _, target_h, target_w = self.image_shape
        img_h, img_w = img_bgr.shape[:2]
        new_w = min(int(img_w * (target_h / img_h)), target_w)
        resized = cv2.resize(img_bgr, (max(1, new_w), target_h))
        if new_w < target_w:
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            padded[:, :new_w, :] = resized
            resized = padded
        img = resized.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img.transpose((2, 0, 1))

    def read(self, plate_bgr):
        tensor = self._paddle.to_tensor(
            np.expand_dims(self._preprocess(plate_bgr), axis=0))
        with self._paddle.no_grad():
            preds = self._model(tensor)
        if isinstance(preds, dict):
            pred = preds.get("ctc", next(iter(preds.values())))
        elif isinstance(preds, (list, tuple)):
            pred = preds[0]
        else:
            pred = preds
        result = self._post(pred.numpy())
        if isinstance(result, (list, tuple)) and result:
            text, conf = result[0]
            return (str(text).strip(), float(conf))
        return ("", 0.0)


BACKENDS = {
    RapidOcrEngine.name: RapidOcrEngine,
    FastPlateEngine.name: FastPlateEngine,
    PaddleAnprEngine.name: PaddleAnprEngine,
}


# Why plate reading is or is not working, as STATE rather than only a print.
#
# A print is enough for `main.py`, where a human is watching the terminal. It
# is not enough for `serve.py`: stdout is redirected, the message scrolls past
# at startup, and the operator meets the consequence hours later as an empty
# plate column on the dashboard with nothing saying why. A silently degraded
# feed is exactly the failure this project surfaces health tiles for, and an
# OCR backend that failed to load is the same class of problem.
#
# Module-level because the engine itself is a module-level singleton (see
# plate.py). With N cameras on one backend - the normal case - this describes
# all of them. Two cameras on DIFFERENT backends would leave this reporting
# whichever initialised last, which is worth knowing before relying on it.
STATUS: dict = {"backend": None, "ok": False, "error": None}


def status() -> dict:
    """Snapshot of plate-OCR availability, for /cameras and the dashboard."""
    return dict(STATUS)


def get_engine(backend: str, model: str = "", join_rows: bool = True):
    """Build a backend, or return None after explaining why it is unavailable.

    None (never an exception) keeps plate reading optional: a missing OCR
    dependency must not stop vehicle counting.
    """
    STATUS.update(backend=backend, ok=False, error=None)
    cls = BACKENDS.get(backend)
    if cls is None:
        msg = f"unknown ocr_backend {backend!r}; choose one of {sorted(BACKENDS)}"
        print(f"[plate] {msg}")
        STATUS["error"] = msg
        return None
    try:
        engine = cls(model=model, join_rows=join_rows)
    except Exception as e:
        # The message carries fetch instructions for the paddle backend, so it
        # is kept whole rather than summarised - it is the actionable part.
        print(f"[plate] {backend} unavailable: {e}")
        STATUS["error"] = f"{backend} unavailable: {e}"
        return None
    print(f"[plate] OCR backend: {backend}"
          + (f" ({model})" if model else ""))
    STATUS["ok"] = True
    return engine
