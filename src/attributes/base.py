"""Attribute plug-ins. Each takes (crop_image, vehicle_dict) and fills one field.

Live now: 'type' (free from YOLO). Stubs with exact signatures: 'color',
'plate', 'brand'. To activate a model: implement the function, add its name
to attributes.enabled in config.yaml. No refactor needed.
"""
from .color import extract_color  # noqa: F401  (kept contract -> str)
from .color import extract_color_conf, garment_colors
from .plate import configure as configure_plate  # noqa: F401
from .plate import crop_score, read_plate, read_plate_tracked  # noqa: F401

MIN_PLATE_W = 80        # skip tiny crops (CPU saving, OCR would fail anyway)


def run_attributes(enabled: list, crop, vehicle: dict) -> dict:
    """Run enabled attribute extractors, return vehicle['attrs']."""
    attrs = vehicle.setdefault("attrs", {})
    if "color" in enabled:
        try:
            cname, cconf = extract_color_conf(crop)
            if cname and float(cconf) >= float(attrs.get("color_conf", 0.0)):
                attrs["color"] = cname
                attrs["color_conf"] = round(float(cconf), 3)
        except Exception:
            pass
    if "clothes" in enabled:
        try:
            g = garment_colors(crop)
            for key in ("upper_color", "lower_color"):
                conf_key = key + "_conf"
                if g[key] and float(g[conf_key]) >= float(attrs.get(conf_key, 0.0)):
                    attrs[key] = g[key]
                    attrs[conf_key] = round(float(g[conf_key]), 3)
        except Exception:
            pass
    if "plate" in enabled:
        try:
            h, w = crop.shape[:2]
            if w >= MIN_PLATE_W and h >= 20:
                # Read strategy (which frames, how many, voting) lives in
                # plate.py so it stays driven by the plate: config block.
                text, conf = read_plate_tracked(crop, vehicle)
                if text:
                    attrs["plate_number"] = text
                    attrs["plate_conf"] = round(float(conf), 3)
        except Exception:
            pass
    attrs.setdefault("plate_number", "")
    attrs.setdefault("plate_conf", 0.0)
    attrs.setdefault("color", attrs.get("color", ""))
    attrs.setdefault("color_conf", attrs.get("color_conf", 0.0))
    attrs.setdefault("upper_color", attrs.get("upper_color", ""))
    attrs.setdefault("upper_color_conf", attrs.get("upper_color_conf", 0.0))
    attrs.setdefault("lower_color", attrs.get("lower_color", ""))
    attrs.setdefault("lower_color_conf", attrs.get("lower_color_conf", 0.0))
    attrs.setdefault("brand", attrs.get("brand", ""))
    return attrs
