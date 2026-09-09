"""Vehicle color via HSV histogram. CPU-only, no model weights.

Contract (kept): extract_color(crop_bgr) -> str like "white"|"black"|"" .
New: extract_color_conf(crop_bgr) -> (str, float conf 0-1).

Method (simple English): car paint is the most common HSV color in the
body band (middle of the crop; roof glass and road edges excluded).
White/black/grey decided by saturation+brightness, the rest by hue.
Conf = share of counted pixels wearing the winning color.
Daylight tuned; night/streetlight shifts hue (documented limit).
Person clothes reuse: dominant_hsv(region) on upper/lower halves.
"""

import cv2
import numpy as np

# Hue ranges on OpenCV 0-179 wheel. Red wraps around 0.
HUE_NAMES = [
    ("red", ((0, 10), (160, 179))),
    ("yellow", ((20, 32),)),
    ("green", ((35, 85),)),
    ("blue", ((90, 130),)),
]
MIN_COUNT = 400  # fewer body pixels than this = unknown (too small/far)
MIN_FRAC = 0.40   # winner needs 40%+ of band pixels, else ambiguous (glass vs
                  # paint, e.g. black SUV rear where windshield dominates) -> ""


def dominant_hsv(region_bgr):
    """Most common HSV bin in region. Returns (kind, name, frac)."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total = h.size
    if total == 0:
        return ("none", "", 0.0)
    # Near-achromatic first: white / black / grey. Black uses v<100 (not 60)
    # because sunlit black paint sits at v 60-100; s<60 keeps asphalt out.
    white = ((s < 40) & (v > 180)).sum()
    black = ((v < 100) & (s < 60)).sum()
    grey = ((s < 40) & (v >= 100) & (v <= 180)).sum()
    # Chromatic: saturation + brightness gate, then hue vote.
    chroma = (s >= 40) & (v >= 60)
    votes = {}
    if chroma.sum() > 0:
        hh = h[chroma]
        for name, ranges in HUE_NAMES:
            m = np.zeros_like(hh, dtype=bool)
            for lo, hi in ranges:
                m |= (hh >= lo) & (hh <= hi)
            votes[name] = m.sum()
    cands = {"white": white, "black": black, "grey": grey}
    cands.update(votes)
    name = max(cands, key=lambda k: cands[k])
    n = int(cands[name])
    if n < MIN_COUNT:
        return ("none", "", 0.0)
    kind = "achromatic" if name in ("white", "black", "grey") else "hue"
    return (kind, name, n / total)


def body_band(crop_bgr):
    """Lower-middle band: trunk/bumper/doors. Skips roof + windshield top
    (sky reflections vote white/blue and drown the paint)."""
    h, w = crop_bgr.shape[:2]
    return crop_bgr[int(h * 0.40):int(h * 0.85), int(w * 0.10):int(w * 0.90)]


def extract_color_conf(crop_bgr):
    try:
        h, w = crop_bgr.shape[:2]
        if w < 40 or h < 40:
            return ("", 0.0)
        _, name, frac = dominant_hsv(body_band(crop_bgr))
        if not name or frac < MIN_FRAC:
            return ("", 0.0)
        return (name, round(float(frac), 3))
    except Exception:
        return ("", 0.0)


def extract_color(crop_bgr) -> str:
    return extract_color_conf(crop_bgr)[0]
