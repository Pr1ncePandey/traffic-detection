"""Compare OCR preprocessing variants on crops where LP boxes exist."""
import glob
import os

import cv2
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR

crops = sorted(glob.glob("outputs/crops/vehicle_*.jpg"),
               key=os.path.getmtime, reverse=True)[:40]
lp = YOLO("models/lp_detector.pt")
ocr = RapidOCR()


def variants(plate):
    h, w = plate.shape[:2]
    out = {}
    out["raw"] = plate
    b = cv2.copyMakeBorder(plate, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    out["raw+pad"] = b
    u2 = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    out["2x+pad"] = cv2.copyMakeBorder(u2, 10, 10, 10, 10,
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return out


n = 0
for path in crops:
    if n >= 8:
        break
    img = cv2.imread(path)
    res = lp.predict(img, conf=0.25, imgsz=480, verbose=False)[0]
    if len(res.boxes) == 0:
        continue
    n += 1
    b = max(res.boxes, key=lambda x: float(x.conf[0]))
    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
    plate = img[max(0, y1-2):y2+2, max(0, x1-2):x2+2]
    print("=== %s plate %dx%d conf %.2f" % (os.path.basename(path), x2-x1, y2-y1, float(b.conf[0])))
    for name, v in variants(plate).items():
        out, _ = ocr(v)
        txt = [(r[1], round(float(r[2]), 2)) for r in out] if out else []
        print("  %-8s %s" % (name, txt))
