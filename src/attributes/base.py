"""Attribute plug-ins. Each takes (crop_image, vehicle_dict) and fills one field.

Live now: 'type' (free from YOLO). Stubs with exact signatures: 'color',
'plate', 'brand'. To activate a model: implement the function, add its name
to attributes.enabled in config.yaml. No refactor needed.
"""
from .color import extract_color  # noqa: F401  (stub, returns "" for now)
from .plate import configure as configure_plate  # noqa: F401
from .plate import crop_score, read_plate

_configured = False

MIN_PLATE_W = 80        # skip tiny crops (CPU saving, OCR would fail anyway)
PLATE_MAX_TRIES = 3     # OCR attempts per vehicle, best crop wins


def run_attributes(enabled: list, crop, vehicle: dict) -> dict:
    """Run enabled attribute extractors, return vehicle['attrs']."""
    attrs = vehicle.setdefault("attrs", {})
    if "color" in enabled:
        try:
            attrs["color"] = extract_color(crop) or ""
        except Exception:
            attrs["color"] = ""
    if "plate" in enabled:
        try:
            h, w = crop.shape[:2]
            if w >= MIN_PLATE_W and h >= 20:
                tries = vehicle.setdefault("_plate_tries", 0)
                best = vehicle.setdefault("_plate_best", 0.0)
                score = crop_score(crop)
                if tries < PLATE_MAX_TRIES and score > best:
                    vehicle["_plate_best"] = score
                    vehicle["_plate_tries"] = tries + 1
                    text, conf = read_plate(crop)
                    if text and conf >= float(attrs.get("plate_conf", 0.0)):
                        attrs["plate_number"] = text
                        attrs["plate_conf"] = round(float(conf), 3)
        except Exception:
            pass
    attrs.setdefault("plate_number", "")
    attrs.setdefault("plate_conf", 0.0)
    attrs.setdefault("color", attrs.get("color", ""))
    attrs.setdefault("brand", attrs.get("brand", ""))
    return attrs
