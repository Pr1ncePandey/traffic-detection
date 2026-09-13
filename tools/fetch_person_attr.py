"""Fetch the person-attribute model and convert it to ONNX. Run once.

    python tools/fetch_person_attr.py

Result: models/person_attr/person_attr.onnx (~7 MB), which is all the
`person` enricher needs at runtime - onnxruntime, no paddle.

WHY A THROWAWAY VENV FOR THE CONVERSION

paddle2onnx is picky about which paddle sits next to it, and none of the
combinations that match this project's own interpreters work:

    paddle2onnx 2.1.0 + paddle 3.3.1 (the project venv)  DLL load fails
    paddle2onnx 2.1.0 + paddle 3.0.0                      rejected as too old
    paddle2onnx 1.3.1 + paddle 2.6.2                      WORKS

So conversion runs in its own venv under models/person_attr/.convert-venv,
created from a Python 3.12 (paddle 2.6.2 has no wheel for 3.13+). Delete that
folder afterwards if disk matters; it is only needed to convert again.

Model: PP-Human attribute, PPLCNet_x1_0, mA 94.5 (PaddleDetection, Apache-2.0).
Training data: PA-100K (CC-BY 4.0), RAPv2, PETA - check their terms before a
commercial deployment.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

URL = ("https://bj.bcebos.com/v1/paddledet/models/pipeline/"
       "PPLCNet_x1_0_person_attribute_945_infer.tar")
NAME = "PPLCNet_x1_0_person_attribute_945_infer"
DEST = os.path.join("models", "person_attr")
ONNX = os.path.join(DEST, "person_attr.onnx")


def find_py312(explicit: str | None) -> str:
    if explicit:
        return explicit
    for cand in ([r"C:\Users\Prince\venvs\traffic\Scripts\python.exe"]
                 + [shutil.which(n) or "" for n in ("python3.12", "python")]):
        if not cand or not os.path.exists(cand):
            continue
        out = subprocess.run([cand, "-c", "import sys;print(sys.version_info[:2])"],
                             capture_output=True, text=True).stdout.strip()
        if out == "(3, 12)":
            # A venv's interpreter makes a poor venv base; use what it was built on.
            base = subprocess.run(
                [cand, "-c", "import sys,os;print(os.path.join(sys.base_prefix,"
                             "'python.exe' if os.name=='nt' else 'bin/python3'))"],
                capture_output=True, text=True).stdout.strip()
            return base if os.path.exists(base) else cand
    raise SystemExit("Need a Python 3.12 to convert (paddle 2.6.2 has no newer "
                     "wheel). Pass one with --python.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", help="a Python 3.12 interpreter for the converter")
    ap.add_argument("--force", action="store_true", help="re-download and convert")
    args = ap.parse_args()

    if os.path.exists(ONNX) and not args.force:
        print(f"{ONNX} already exists (use --force to redo)")
        return
    os.makedirs(DEST, exist_ok=True)

    model_dir = os.path.join(DEST, NAME)
    if not os.path.exists(os.path.join(model_dir, "inference.pdmodel")) or args.force:
        tar_path = os.path.join(DEST, NAME + ".tar")
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(DEST, filter="data") if sys.version_info >= (3, 12) \
                else tf.extractall(DEST)
        os.remove(tar_path)

    venv = os.path.join(DEST, ".convert-venv")
    vpy = os.path.join(venv, "Scripts" if os.name == "nt" else "bin",
                       "python.exe" if os.name == "nt" else "python")
    if not os.path.exists(vpy):
        base = find_py312(args.python)
        print(f"creating converter venv from {base}")
        subprocess.check_call([base, "-m", "venv", venv])
    print("installing paddlepaddle 2.6.2 + paddle2onnx 1.3.1 into it")
    subprocess.check_call([vpy, "-m", "pip", "install", "-q", "paddlepaddle==2.6.2",
                           "paddle2onnx==1.3.1", "setuptools", "packaging"])

    print("converting")
    subprocess.check_call([vpy, "-m", "paddle2onnx.command",
                           "--model_dir", model_dir,
                           "--model_filename", "inference.pdmodel",
                           "--params_filename", "inference.pdiparams",
                           "--save_file", ONNX, "--opset_version", "14"])
    print(f"\nwrote {ONNX}. The converter venv at {venv} can be deleted.")


if __name__ == "__main__":
    main()
