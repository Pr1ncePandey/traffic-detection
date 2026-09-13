"""Load MiVOLO v2 (iitolstykh/mivolo_v2, Apache-2.0) on a released timm.

MiVOLO's HuggingFace model code imports the `mivolo` package, which pins an
unreleased timm build (0.8.13.dev0). `pip install` of that package fails on
Windows (the repo uses symlinks), so it is imported from a source checkout at
models/MiVOLO instead, on timm 0.9.2, with three small shims:

  remap_checkpoint      removed from timm; only used when loading a raw .pth,
                        not the safetensors weights used here -> identity
  split_model_name_tag  removed from timm; re-implemented (5 lines)
  VOLO.__init__         timm added `pos_drop_rate` after `drop_rate`, and
                        MiVOLO passes all 21 arguments positionally, so every
                        argument from there on landed one slot late
                        (drop_path_rate received the norm layer CLASS)

Nothing in the checkout is edited.
"""

import os
import sys

REPO = "iitolstykh/mivolo_v2"
CHECKOUT = os.path.join(os.path.dirname(__file__), "..", "..", "models", "MiVOLO")


def _shim_timm():
    import timm.models._helpers as helpers
    import timm.models._pretrained as pretrained
    from timm.models import volo

    if not hasattr(helpers, "remap_checkpoint"):
        helpers.remap_checkpoint = lambda model, state_dict, **kw: state_dict
    if not hasattr(pretrained, "split_model_name_tag"):
        def split_model_name_tag(model_name, no_tag=""):
            name, *tag = model_name.split(".", 1)
            return name, (tag[0] if tag else no_tag)
        pretrained.split_model_name_tag = split_model_name_tag

    init = volo.VOLO.__init__
    if not getattr(init, "_mivolo_compat", False):
        import inspect
        if "pos_drop_rate" in inspect.signature(init).parameters:
            def compat(self, *args, **kwargs):
                if len(args) == 21:                   # MiVOLO's old positional order
                    args = args[:14] + (0.0,) + args[14:]
                init(self, *args, **kwargs)
            compat._mivolo_compat = True
            volo.VOLO.__init__ = compat


def load_mivolo(threads: int = 4):
    """-> (model, processor, config). Raises with an install hint on failure."""
    checkout = os.path.abspath(CHECKOUT)
    if not os.path.isdir(os.path.join(checkout, "mivolo")):
        raise FileNotFoundError(f"{checkout} missing - run: python tools/fetch_person_models.py")
    if checkout not in sys.path:
        sys.path.insert(0, checkout)
    import torch
    _shim_timm()
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    torch.set_num_threads(max(1, int(threads)))
    config = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(
        REPO, trust_remote_code=True, torch_dtype=torch.float32).eval()
    processor = AutoImageProcessor.from_pretrained(REPO, trust_remote_code=True)
    return model, processor, config


def predict(model, processor, config, body_crops_bgr) -> list:
    """Body-only inference. -> [(age_years, p_female), ...], one per crop."""
    import torch
    if not body_crops_bgr:
        return []
    bodies = processor(images=list(body_crops_bgr))["pixel_values"]
    faces = processor(images=[None] * len(body_crops_bgr))["pixel_values"]
    with torch.inference_mode():
        out = model(faces_input=faces, body_input=bodies)
    female = [k for k, v in config.gender_id2label.items() if v == "female"]
    female_idx = int(female[0]) if female else 1
    results = []
    for i in range(len(body_crops_bgr)):
        prob = float(out.gender_probs[i].item())
        idx = int(out.gender_class_idx[i].item())
        results.append((float(out.age_output[i].item()),
                        prob if idx == female_idx else 1.0 - prob))
    return results
