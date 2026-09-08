"""Number-plate reader: YOLO plate-box detector + RapidOCR (both CPU).

Flow: vehicle crop -> LP detector finds plate box -> RapidOCR on the raw box
(padded copy as fallback; upscaling tested worse, not used) -> normalize
(uppercase, alnum only) -> plate-ish gate -> (text, conf).
No plate / unreadable / deps missing -> ("", 0.0). Never raises.

Identity rule: ByteTrack ID = same car WITHIN one video; plate_number = same
car ACROSS videos/cameras (brother's journey key). Best-crop voting (max 3
tries/vehicle) lives in base.py.
"""

import cv2

_PLATE_RE_MIN = 4  # min alnum chars to accept

_lp_model = None
_ocr_engine = None
_cfg = {"det_conf": 0.3, "min_conf": 0.5, "det_imgsz": 480}


def configure(det_conf=0.3, min_conf=0.5, det_imgsz=480):
    _cfg.update(det_conf=det_conf, min_conf=min_conf, det_imgsz=det_imgsz)


def _engines():
    """Lazy singletons. (None, None) if deps/weights missing."""
    global _lp_model, _ocr_engine
    if _lp_model is None:
        try:
            from ultralytics import YOLO
            _lp_model = YOLO("models/lp_detector.pt")
        except Exception as e:
            print(f"[plate] LP detector unavailable: {e}")
            _lp_model = False
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception as e:
            print(f"[plate] RapidOCR unavailable: {e}")
            _ocr_engine = False
    return (_lp_model or None, _ocr_engine or None)


def _sharpness(gray) -> float:
    try:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def crop_score(crop_bgr) -> float:
    """Bigger + sharper crop = better OCR chance. Pipeline keeps the best."""
    try:
        h, w = crop_bgr.shape[:2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        return w * h * (1.0 + _sharpness(gray) / 1000.0)
    except Exception:
        return 0.0


def _normalize(text: str) -> str:
    return "".join(c for c in (text or "").upper() if c.isalnum())


def _plate_like(text: str) -> bool:
    return (len(text) >= _PLATE_RE_MIN
            and any(c.isalpha() for c in text)
            and any(c.isdigit() for c in text))


def read_plate(crop_bgr):
    """Real OCR. Returns (text, conf); ("", 0.0) when nothing readable."""
    try:
        h, w = crop_bgr.shape[:2]
        if w < 60 or h < 15:
            return ("", 0.0)
        lp_model, ocr = _engines()
        if lp_model is None or ocr is None:
            return ("", 0.0)
        res = lp_model.predict(crop_bgr, conf=_cfg["det_conf"],
                               imgsz=_cfg["det_imgsz"], verbose=False)[0]
        if len(res.boxes) == 0:
            return ("", 0.0)
        b = max(res.boxes, key=lambda x: float(x.conf[0]))
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1, y1 = max(0, x1 - 2), max(0, y1 - 2)
        x2, y2 = min(w, x2 + 2), min(h, y2 + 2)
        plate = crop_bgr[y1:y2, x1:x2]
        if plate.size == 0:
            return ("", 0.0)
        # Raw box first (RapidOCR's own detector handles small text best);
        # padded copy as fallback. Upscaling tested worse — not used.
        cands = [plate,
                 cv2.copyMakeBorder(plate, 8, 8, 8, 8,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))]
        for cand in cands:
            out, _ = ocr(cand)
            if not out:
                continue
            best = max(out, key=lambda r: float(r[2]))
            text = _normalize(best[1])
            conf = float(best[2])
            if _plate_like(text) and conf >= _cfg["min_conf"]:
                return (text, round(conf, 3))
        return ("", 0.0)
    except Exception:
        return ("", 0.0)
