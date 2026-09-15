"""Fetch a CLIP/SigLIP ONNX encoder pair for open-vocabulary search. Run once.

    python tools/fetch_clip.py                      # the default, ViT-B/16
    python tools/fetch_clip.py --model clip-vit-l14
    python tools/fetch_clip.py --list

Result: models/<dir>/{vision.onnx,text.onnx,tokenizer.json}, which is all
src/query/clip_onnx.py needs at runtime - onnxruntime and tokenizers, no torch.

WHY PRE-EXPORTED RATHER THAN CONVERTED HERE

tools/fetch_person_attr.py has to build a throwaway venv because paddle2onnx
is particular about which paddle sits beside it. Nothing like that is needed
here: these are already-exported ONNX graphs, so this script is a download and
a checksum-free sanity check on the signatures. Exporting from torch ourselves
would reintroduce exactly the fragility that script documents.

WHICH MODEL, AND WHY B/16 IS THE DEFAULT

Measured on this project's own footage, five variants, same corpus and metric
(docs/vector-search.md, Layer 4a preview):

    B/32        17 ms/crop   p@5 0.55   fast, too weak
    B/16        65 ms/crop   p@5 0.775  DEFAULT - best quality per ms
    L/14       312 ms/crop   p@5 0.80   18x slower for +0.025
    L/14 fp16  304 ms/crop   p@5 0.80   fp16 buys nothing on ORT CPU
    L/14 uint8 102 ms/crop   p@5 0.75   dominated by B/16

B/16 at ~15 crops/s is ~1.3e6 crops/day, against ~7.7e5 objects/day from one
camera at full duty cycle - so it keeps up with one camera, and a fleet needs
a GPU or parallel embedders regardless of which of these you pick.

LICENCES. The ONNX exports are Xenova's and Qdrant's ports of the upstream
OpenAI CLIP / Google SigLIP weights. Check the upstream model cards before a
commercial deployment; this script does not vendor or relicense anything.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request

HF = "https://huggingface.co"

# model key -> (dest dir, {local filename: repo path})
SOURCES = {
    "clip-vit-b16": ("models/clip_b16", "Xenova/clip-vit-base-patch16", {
        "vision.onnx": "onnx/vision_model.onnx",
        "text.onnx": "onnx/text_model.onnx",
        "tokenizer.json": "tokenizer.json",
    }),
    "clip-vit-b32": ("models/clip", "Qdrant/clip-ViT-B-32-vision", {
        "vision.onnx": "model.onnx",
        "preprocessor_config.json": "preprocessor_config.json",
    }),
    "clip-vit-l14": ("models/clip_l14", "Xenova/clip-vit-large-patch14", {
        "vision.onnx": "onnx/vision_model.onnx",
        "text.onnx": "onnx/text_model.onnx",
        "tokenizer.json": "tokenizer.json",
    }),
    "siglip-b16": ("models/siglip", "Xenova/siglip-base-patch16-224", {
        "vision.onnx": "onnx/vision_model.onnx",
        "text.onnx": "onnx/text_model.onnx",
        "tokenizer.json": "tokenizer.json",
        "preprocessor_config.json": "preprocessor_config.json",
    }),
}
# B/32's vision and text towers live in two separate Qdrant repos.
EXTRA = {
    "clip-vit-b32": ("Qdrant/clip-ViT-B-32-text", {
        "text.onnx": "model.onnx",
        "tokenizer.json": "tokenizer.json",
    }),
}


def _progress(done, total, name):
    if not total:
        return
    pct = 100 * done / total
    sys.stdout.write(f"\r  {name:22} {pct:5.1f}%  "
                     f"{done / 1e6:6.1f}/{total / 1e6:.1f} MB")
    sys.stdout.flush()


def fetch(repo: str, remote: str, dest: str, force: bool) -> bool:
    if os.path.exists(dest) and not force:
        print(f"  {os.path.basename(dest):22} present, skipping "
              f"({os.path.getsize(dest) / 1e6:.1f} MB)")
        return True
    url = f"{HF}/{repo}/resolve/main/{remote}"
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            total = int(r.headers.get("content-length") or 0)
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    _progress(done, total, os.path.basename(dest))
        print()
        # Rename only after a complete read, so an interrupted download can
        # never leave a truncated .onnx that onnxruntime fails on cryptically.
        os.replace(tmp, dest)
        return True
    except (urllib.error.URLError, OSError) as e:
        print(f"\n  FAILED {os.path.basename(dest)}: {e}")
        if os.path.exists(tmp):
            os.unlink(tmp)
        return False


def verify(model: str, directory: str) -> bool:
    """Load the graphs and check the signatures agree with the spec.

    Worth doing here rather than at first query: a truncated or wrong-repo
    file otherwise surfaces as an opaque onnxruntime error in the middle of a
    search, and a dimension mismatch between the two towers would silently
    produce meaningless similarities.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed - skipping verification)")
        return True
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.query.clip_onnx import SPECS

    spec = SPECS[model]
    dims = {}
    for tower, out in (("vision", spec.vis_out), ("text", spec.txt_out)):
        path = os.path.join(directory, f"{tower}.onnx")
        if not os.path.exists(path):
            print(f"  {tower}.onnx missing")
            return False
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        names = [o.name for o in s.get_outputs()]
        if out not in names:
            print(f"  {tower}.onnx has outputs {names}, expected {out!r}")
            return False
        shape = next(o.shape for o in s.get_outputs() if o.name == out)
        dims[tower] = shape[-1] if isinstance(shape[-1], int) else None
    if dims["vision"] != dims["text"]:
        print(f"  MISMATCH: vision is {dims['vision']}-d, text is "
              f"{dims['text']}-d. These cannot be compared.")
        return False
    if dims["vision"] and dims["vision"] != spec.dim:
        print(f"  MISMATCH: files are {dims['vision']}-d but the spec says "
              f"{spec.dim}. Update SPECS in src/query/clip_onnx.py.")
        return False
    print(f"  verified: both towers output {dims['vision']}-d embeddings")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="clip-vit-b16", choices=sorted(SOURCES))
    p.add_argument("--force", action="store_true", help="re-download")
    p.add_argument("--list", action="store_true", help="show known models")
    a = p.parse_args()

    if a.list:
        print("known models (default: clip-vit-b16):")
        for k, (d, repo, files) in sorted(SOURCES.items()):
            print(f"  {k:14} {repo:38} -> {d}")
        return 0

    directory, repo, files = SOURCES[a.model]
    os.makedirs(directory, exist_ok=True)
    print(f"[clip] {a.model} from {repo} -> {directory}/")
    ok = all(fetch(repo, remote, os.path.join(directory, local), a.force)
             for local, remote in files.items())
    if a.model in EXTRA:
        repo2, files2 = EXTRA[a.model]
        ok = ok and all(
            fetch(repo2, remote, os.path.join(directory, local), a.force)
            for local, remote in files2.items())
    if not ok:
        print("[clip] some files failed; re-run to retry (completed files are "
              "skipped)")
        return 1
    if not verify(a.model, directory):
        return 1
    print(f"[clip] ready. Embed with:\n"
          f"  python tools/embed_crops.py --model {a.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
