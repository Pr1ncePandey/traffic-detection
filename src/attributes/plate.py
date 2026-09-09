"""Number-plate reader: YOLO plate-box detector + a selectable OCR backend.

Flow: vehicle crop -> LP detector finds the plate box -> the configured OCR
backend reads it (ocr_engines.py) -> normalize + Indian-format correction
(plate_format.py) -> plate-ish gate -> (text, conf).
No plate / unreadable / deps missing -> ("", 0.0). Never raises.

Swap the model with `plate.ocr_backend` in config.yaml - no code change:
  rapidocr     generic scene-text OCR. Small and CPU-only, but NOT
               plate-specialised, so it is the weakest of the three.
  fast_plate   plate-specialised CCT model (~5MB ONNX). First upgrade to try.
  paddle_anpr  PP-OCRv5 finetuned on Indian plates; best on two-row plates,
               heaviest to install.
Compare them on your own footage with tools/bench_ocr.py.

Identity rule: ByteTrack ID = same car WITHIN one video; plate_number = same
car ACROSS videos/cameras (the journey key). Best-crop voting (max 3
tries/vehicle) lives in base.py.
"""

from collections import Counter

import cv2

from . import ocr_engines, plate_format

# Each repaired character costs this much confidence, so a plate that needed
# fixing ranks below a cleanly-read one and the existing min_conf gate does the
# rejecting. One threshold, not two.
REPAIR_CONF_DECAY = 0.9

_lp_model = None
_ocr_engine = None
_cfg = {"det_conf": 0.3, "min_conf": 0.5, "det_imgsz": 480,
        "ocr_backend": "paddle_anpr", "ocr_model": "",
        "format_correction": True, "join_rows": True,
        "read_every": 3, "max_reads": 8, "vote": True, "vote_min_conf": 0.8}


def configure(**kwargs):
    """Set thresholds and pick the OCR backend. Called once from the pipeline.

    Takes the whole `plate:` config block, so adding a knob in config.yaml
    never means editing pipeline.py again. Owns its own type coercion because
    this is the module that knows the types.
    """
    global _ocr_engine
    unknown = sorted(k for k in kwargs if k not in _cfg)
    if unknown:
        print(f"[plate] ignoring unknown plate config keys: {unknown}")

    def pick(key, cast):
        return cast(kwargs[key]) if kwargs.get(key) is not None else _cfg[key]

    backend = pick("ocr_backend", str)
    model = pick("ocr_model", str)
    # Switching either invalidates the cached engine.
    if (backend, model) != (_cfg["ocr_backend"], _cfg["ocr_model"]):
        _ocr_engine = None
    _cfg.update(det_conf=pick("det_conf", float),
                min_conf=pick("min_conf", float),
                det_imgsz=pick("det_imgsz", int),
                ocr_backend=backend, ocr_model=model,
                format_correction=pick("format_correction", bool),
                join_rows=pick("join_rows", bool),
                read_every=max(1, pick("read_every", int)),
                max_reads=max(1, pick("max_reads", int)),
                vote=pick("vote", bool),
                vote_min_conf=pick("vote_min_conf", float))


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
        _ocr_engine = ocr_engines.get_engine(
            _cfg["ocr_backend"], _cfg["ocr_model"], _cfg["join_rows"]) or False
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


def detect_plate_box(crop_bgr):
    """Find the plate inside a vehicle crop. Returns the plate image or None.

    Public so tools/bench_ocr.py can reuse it instead of duplicating it.
    """
    h, w = crop_bgr.shape[:2]
    if w < 60 or h < 15:
        return None
    lp_model, _ = _engines()
    if lp_model is None:
        return None
    res = lp_model.predict(crop_bgr, conf=_cfg["det_conf"],
                           imgsz=_cfg["det_imgsz"], verbose=False)[0]
    if len(res.boxes) == 0:
        return None
    b = max(res.boxes, key=lambda x: float(x.conf[0]))
    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
    x1, y1 = max(0, x1 - 2), max(0, y1 - 2)
    x2, y2 = min(w, x2 + 2), min(h, y2 + 2)
    plate = crop_bgr[y1:y2, x1:x2]
    return plate if plate.size else None


def _candidates(plate, engine):
    """Crops to try, in order. Upscaling tested worse, so it is not used."""
    if not getattr(engine, "wants_padding", False):
        return [plate]
    # A backend running its own text detector wants a quiet margin to find the
    # text in: unpadded box first, padded copy as fallback.
    return [plate,
            cv2.copyMakeBorder(plate, 8, 8, 8, 8,
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))]


def finalize(raw_text, conf):
    """Normalize, correct, and price in the repairs. Returns (text, conf)."""
    text = plate_format.normalize(raw_text)
    if _cfg["format_correction"]:
        text, _matched, repairs = plate_format.correct(text)
        conf *= REPAIR_CONF_DECAY ** repairs
    return (text, conf)


def read_plate_image(plate_bgr):
    """OCR an already-cropped plate. Returns (text, conf), WITHOUT the
    min_conf gate.

    Split out from read_plate so tools/bench_ocr.py exercises exactly this
    path - crop variants included - instead of a parallel copy of it, and so
    the min_conf gate stays one policy decision applied in one place. A
    benchmark needs to see a weak-but-correct read; the pipeline does not.
    """
    _lp, engine = _engines()
    if engine is None:
        return ("", 0.0)
    best = ("", 0.0)
    for cand in _candidates(plate_bgr, engine):
        text, conf = finalize(*engine.read(cand))
        if plate_format.looks_like_plate(text) and conf > best[1]:
            best = (text, conf)
            if conf >= _cfg["min_conf"]:
                break  # already good enough, no need for the other variant
    return best


def read_plate(crop_bgr):
    """Real OCR. Returns (text, conf); ("", 0.0) when nothing readable."""
    try:
        plate = detect_plate_box(crop_bgr)
        if plate is None:
            return ("", 0.0)
        text, conf = read_plate_image(plate)
        if text and conf >= _cfg["min_conf"]:
            return (text, round(conf, 3))
        return ("", 0.0)
    except Exception:
        return ("", 0.0)


def _consensus(reads):
    """Pick one answer from a vehicle's reads. Returns (text, conf).

    Per-character vote across the confident reads, because the same plate
    reads differently frame to frame (one car here produced 48 distinct
    strings over 64 frames) and the errors are rarely in the same position
    twice. Gating on confidence first is essential: voting over ALL frames
    is worse than not voting, since the distant views contribute garbage.
    Below a few confident reads there is nothing to vote on, so the single
    best read wins.
    """
    if not reads:
        return ("", 0.0)
    best = max(reads, key=lambda r: r[1])
    if not _cfg["vote"]:
        return best
    strong = [t for t, c in reads if c >= _cfg["vote_min_conf"]]
    if len(strong) < 3:
        return best
    length = Counter(len(t) for t in strong).most_common(1)[0][0]
    aligned = [t for t in strong if len(t) == length]
    if len(aligned) < 3:
        return best
    voted = "".join(Counter(chars).most_common(1)[0][0]
                    for chars in zip(*aligned))
    if not plate_format.looks_like_plate(voted):
        return best
    # Report the best observed confidence for the string we actually return.
    same = [c for t, c in reads if t == voted]
    return (voted, max(same) if same else best[1])


def read_plate_tracked(crop_bgr, vehicle: dict):
    """Accumulate reads across a vehicle's frames and return the consensus.

    Why not just OCR the biggest, sharpest crop: measured on this clip, the
    plate is widest the moment it clears the bottom of the frame, but that is
    also its most oblique, most motion-blurred view. Confidence peaks 9-16
    frames LATER on a ~10% smaller plate, as the plate turns more front-on.
    Selecting on crop size therefore picks a systematically bad frame - and
    the old `score > best` gate made it worse, because crop score only falls
    for a receding vehicle, so no frame after the first ever got read.

    So: sample every Nth frame in which a plate is actually readable, spread
    over the vehicle's life, up to a budget. A frame whose plate box is not
    found yet (the vehicle has entered but its plate is still clipped by the
    bottom edge - true for 3 of 5 vehicles here) costs no budget.
    """
    reads = vehicle.setdefault("_plate_reads", [])
    if len(reads) >= _cfg["max_reads"]:
        return vehicle["attrs"].get("plate_number", ""), \
            vehicle["attrs"].get("plate_conf", 0.0)
    seen = vehicle.get("_plate_seen", 0) + 1
    vehicle["_plate_seen"] = seen
    if (seen - 1) % _cfg["read_every"]:
        return _consensus(reads)
    text, conf = read_plate(crop_bgr)
    if text:
        reads.append((text, conf))
    return _consensus(reads)
