"""CLIP image/text encoders over onnxruntime. No torch at inference.

Shared by the offline embedder (tools/embed_crops.py, image side) and the
query path (text side), so the two cannot drift apart on preprocessing - which
would silently put queries and crops in different spaces and quietly ruin
every similarity.

MODEL CHOICE IS MEASURED, NOT ASSUMED. ViT-B/16 is the default because on this
project's own footage it was the best quality-per-millisecond of five
variants (docs/vector-search.md, Layer 4a preview):

    B/32       17 ms/crop   p@5 0.55    too weak
    B/16       65 ms/crop   p@5 0.775   <- default
    L/14      312 ms/crop   p@5 0.80    18x slower for +0.025
    L/14 fp16 304 ms/crop   p@5 0.80    fp16 buys nothing on CPU
    L/14 uint8 102 ms/crop  p@5 0.75    dominated by B/16

Preprocessing differs by FAMILY and is not interchangeable: CLIP resizes the
shortest edge then centre-crops, SigLIP squashes to a square. Getting this
wrong does not raise - it just degrades silently, which is why each spec
carries its own resize mode rather than sharing one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# CLIP's published normalisation constants.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Averaged rather than used singly: prompt ensembling is how CLIP zero-shot is
# normally run, and it costs one extra text pass over ~7 short strings.
TEMPLATES = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a cctv photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of the {}.",
    "a cropped photo of a {}.",
    "a street photo of a {}.",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str                       # the `model` string stored in embeddings
    dir: str
    dim: int
    mean: tuple = _CLIP_MEAN
    std: tuple = _CLIP_STD
    resize: str = "shortest_crop"   # 'shortest_crop' (CLIP) | 'squash' (SigLIP)
    text_needs_mask: bool = False
    text_pad_to: int | None = None  # SigLIP pads to a fixed 64
    vis_out: str = "image_embeds"
    txt_out: str = "text_embeds"


SPECS: dict[str, ModelSpec] = {
    "clip-vit-b16": ModelSpec("clip-vit-b16", "models/clip_b16", 512),
    "clip-vit-b32": ModelSpec("clip-vit-b32", "models/clip", 512,
                              text_needs_mask=True),
    "clip-vit-l14": ModelSpec("clip-vit-l14", "models/clip_l14", 768),
    "siglip-b16": ModelSpec("siglip-b16", "models/siglip", 768,
                            mean=(0.5,) * 3, std=(0.5,) * 3, resize="squash",
                            text_pad_to=64, vis_out="pooler_output",
                            txt_out="pooler_output"),
}
DEFAULT_MODEL = "clip-vit-b16"


def l2_normalise(a: np.ndarray) -> np.ndarray:
    """Unit-length rows, with a guard so an all-zero row cannot produce NaN
    and poison every later dot product."""
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n == 0, 1.0, n)


def preprocess(bgr: np.ndarray, spec: ModelSpec) -> np.ndarray | None:
    """One decoded BGR image -> CHW float32, ready for the vision tower."""
    import cv2
    if bgr is None or bgr.size == 0:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if spec.resize == "squash":
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
    else:
        h, w = rgb.shape[:2]
        s = 224 / min(h, w)
        rgb = cv2.resize(rgb, (max(224, round(w * s)), max(224, round(h * s))),
                         interpolation=cv2.INTER_CUBIC)
        h, w = rgb.shape[:2]
        top, left = (h - 224) // 2, (w - 224) // 2
        rgb = rgb[top:top + 224, left:left + 224]
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.asarray(spec.mean, np.float32)) / np.asarray(spec.std, np.float32)
    return x.transpose(2, 0, 1)


class ClipOnnx:
    """Lazily-loaded encoders. The towers are ~330 MB and ~250 MB, so neither
    is loaded until something actually asks for that modality - the embedder
    never needs the text tower, the query path never needs the vision one."""

    def __init__(self, model: str = DEFAULT_MODEL, providers=None):
        if model not in SPECS:
            raise ValueError(f"unknown model {model!r}; "
                             f"known: {', '.join(sorted(SPECS))}")
        self.spec = SPECS[model]
        self.model = model
        self._providers = providers or ["CPUExecutionProvider"]
        self._vis = self._txt = self._tok = None

    # --- availability -----------------------------------------------------
    def missing(self) -> list[str]:
        d = self.spec.dir
        return [p for p in (os.path.join(d, "vision.onnx"),
                            os.path.join(d, "text.onnx"),
                            os.path.join(d, "tokenizer.json"))
                if not os.path.exists(p)]

    def _session(self, filename):
        import onnxruntime as ort
        path = os.path.join(self.spec.dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Fetch the weights first:\n"
                f"  python tools/fetch_clip.py --model {self.model}")
        return ort.InferenceSession(path, providers=self._providers)

    # --- image side -------------------------------------------------------
    def encode_images(self, batch: np.ndarray) -> np.ndarray:
        """Pre-processed NCHW float32 -> L2-normalised embeddings."""
        if self._vis is None:
            self._vis = self._session("vision.onnx")
        if self._vis.get_inputs()[0].type == "tensor(float16)":
            batch = batch.astype(np.float16)
        out = self._vis.run([self.spec.vis_out], {"pixel_values": batch})[0]
        return l2_normalise(out.astype(np.float32))

    # --- text side --------------------------------------------------------
    def _tokenize(self, texts):
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(
                os.path.join(self.spec.dir, "tokenizer.json"))
        enc = [self._tok.encode(t) for t in texts]
        # CLIP's context length is 77; truncate rather than let the session
        # reject a long query.
        cap = self.spec.text_pad_to or min(77, max(len(e.ids) for e in enc))
        ids = np.zeros((len(enc), cap), np.int64)
        mask = np.zeros((len(enc), cap), np.int64)
        for i, e in enumerate(enc):
            k = min(len(e.ids), cap)
            ids[i, :k] = e.ids[:k]
            mask[i, :k] = e.attention_mask[:k]
        return ids, mask

    def encode_text(self, texts) -> np.ndarray:
        if self._txt is None:
            self._txt = self._session("text.onnx")
        ids, mask = self._tokenize(list(texts))
        feed = {"input_ids": ids}
        if self.spec.text_needs_mask:
            feed["attention_mask"] = mask
        out = self._txt.run([self.spec.txt_out], feed)[0]
        return l2_normalise(out.astype(np.float32))

    def encode_query(self, text: str, ensemble: bool = True) -> np.ndarray:
        """One search string -> one unit vector.

        With `ensemble`, the templates are averaged and renormalised, which is
        the standard zero-shot recipe. A query is used verbatim inside the
        template, never rewritten - the router guarantees the same thing at
        its level.
        """
        if not ensemble:
            return self.encode_text([text])[0]
        return l2_normalise(
            self.encode_text([t.format(text) for t in TEMPLATES]).mean(axis=0))
