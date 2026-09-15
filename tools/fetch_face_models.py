"""Download the face models into models/faces and export AdaFace to ONNX. Run once.

    python tools/fetch_face_models.py

Skips anything already present.

  models/faces/yunet/face_detection_yunet_2023mar.onnx   OpenCV Zoo, MIT, 0.2 MB
  models/faces/adaface_ir101/adaface_ir101.onnx          exported from
                                                         minchul/cvlface_adaface_ir101_webface12m
                                                         (code MIT; trained on WebFace12M,
                                                         a research-only dataset)

The export needs torch, safetensors and huggingface_hub; running recognition
afterwards needs only onnxruntime. AdaFace's model file imports fvcore just to
count FLOPs, so fvcore is stubbed rather than installed. The export is checked
against PyTorch before it replaces anything.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "faces"
YUNET = MODELS / "yunet" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
ADAFACE_DIR = MODELS / "adaface_ir101"
ADAFACE_ONNX = ADAFACE_DIR / "adaface_ir101.onnx"
ADAFACE_REPO = "minchul/cvlface_adaface_ir101_webface12m"


def fetch_yunet():
    if YUNET.exists():
        print(f"ok   {YUNET.relative_to(ROOT)}")
        return
    YUNET.parent.mkdir(parents=True, exist_ok=True)
    print(f"get  {YUNET_URL}")
    tmp = YUNET.with_suffix(".part")
    urllib.request.urlretrieve(YUNET_URL, tmp)
    tmp.replace(YUNET)


def fetch_adaface():
    if ADAFACE_ONNX.exists():
        print(f"ok   {ADAFACE_ONNX.relative_to(ROOT)}")
        return
    try:
        import numpy as np
        import onnxruntime as ort
        import safetensors.torch
        import torch
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        sys.exit(f"exporting AdaFace needs: pip install torch safetensors huggingface_hub onnx "
                 f"onnxruntime  ({exc})")

    if not (ADAFACE_DIR / "model.safetensors").exists():
        print(f"get  {ADAFACE_REPO}")
        snapshot_download(ADAFACE_REPO, local_dir=str(ADAFACE_DIR))

    stub = types.ModuleType("fvcore")
    stub_nn = types.ModuleType("fvcore.nn")
    stub_nn.flop_count = lambda *a, **k: ({}, {})
    stub.nn = stub_nn
    sys.modules.setdefault("fvcore", stub)
    sys.modules.setdefault("fvcore.nn", stub_nn)

    spec = importlib.util.spec_from_file_location("adaface_iresnet",
                                                  ADAFACE_DIR / "models" / "iresnet" / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.IR_101(input_size=(112, 112), output_dim=512)

    state = safetensors.torch.load_file(str(ADAFACE_DIR / "model.safetensors"))
    state = {k.split("net.", 1)[1] if "net." in k else k: v for k, v in state.items()}
    missing, _ = net.load_state_dict(state, strict=False)
    if missing:
        sys.exit(f"AdaFace weights did not load: {len(missing)} missing keys, e.g. {missing[:3]}")
    net.eval()

    sample = torch.randn(2, 3, 112, 112)
    with torch.inference_mode():
        expected = net(sample).numpy()
    tmp = ADAFACE_ONNX.with_suffix(".part.onnx")
    torch.onnx.export(net, sample, str(tmp), input_names=["input"], output_names=["embedding"],
                      dynamic_axes={"input": {0: "n"}, "embedding": {0: "n"}},
                      opset_version=17, dynamo=False)
    got = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"]).run(
        None, {"input": sample.numpy()})[0]
    difference = float(np.abs(got - expected).max())
    if difference > 1e-3:
        tmp.unlink(missing_ok=True)
        sys.exit(f"exported AdaFace does not match PyTorch (max difference {difference:.2e})")
    tmp.replace(ADAFACE_ONNX)
    print(f"ok   {ADAFACE_ONNX.relative_to(ROOT)} (matches PyTorch to {difference:.1e})")


if __name__ == "__main__":
    fetch_yunet()
    fetch_adaface()
    print("models ready")
