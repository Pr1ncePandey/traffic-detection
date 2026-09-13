"""Fetch the experimental person models: SegFormer garments + MiVOLO age/gender.

    python tools/fetch_person_models.py

Run with the project interpreter (the one with torch). Downloads from
HuggingFace into its cache, then:

  SegFormer-B2 clothes (MIT)  -> exported to models/clothes_seg/segformer_b2_clothes.onnx
                                 (onnxruntime is ~6x faster than torch eager here)
  MiVOLO v2 (Apache-2.0)      -> source checkout at models/MiVOLO (its model code
                                 is imported from there; `pip install` of it fails
                                 on Windows because the repo uses symlinks), and
                                 weights warmed into the HF cache

Needs: transformers==4.51.0, timm==0.9.2 (see src/attributes/mivolo_loader.py
for the two shims that make MiVOLO load on a released timm).
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEG_REPO = "mattmdjaga/segformer_b2_clothes"
SEG_ONNX = os.path.join("models", "clothes_seg", "segformer_b2_clothes.onnx")
MIVOLO_GIT = "https://github.com/WildChlamydia/MiVOLO.git"
MIVOLO_DIR = os.path.join("models", "MiVOLO")


def export_segformer():
    if os.path.exists(SEG_ONNX):
        print(f"{SEG_ONNX} exists")
        return
    import torch
    from transformers import AutoModelForSemanticSegmentation
    model = AutoModelForSemanticSegmentation.from_pretrained(SEG_REPO).eval()

    class Logits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(pixel_values=x).logits

    os.makedirs(os.path.dirname(SEG_ONNX), exist_ok=True)
    torch.onnx.export(Logits(model), torch.randn(1, 3, 384, 192), SEG_ONNX,
                      input_names=["pixel_values"], output_names=["logits"],
                      dynamic_axes={"pixel_values": {0: "n", 2: "h", 3: "w"},
                                    "logits": {0: "n", 2: "h4", 3: "w4"}},
                      opset_version=17, dynamo=False)
    print(f"wrote {SEG_ONNX}")


def fetch_mivolo():
    if not os.path.exists(os.path.join(MIVOLO_DIR, "mivolo")):
        subprocess.check_call(["git", "-c", "core.symlinks=false", "clone", "--depth",
                               "1", MIVOLO_GIT, MIVOLO_DIR])
    from src.attributes.mivolo_loader import load_mivolo
    load_mivolo()
    print("MiVOLO v2 loads")


if __name__ == "__main__":
    export_segformer()
    fetch_mivolo()
