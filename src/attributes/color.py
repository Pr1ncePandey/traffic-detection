"""Color attribute stub. Real version: HSV histogram / small classifier on crop.

Contract: extract_color(crop_bgr) -> str like "red"|"white"|"black"|"" (unknown).
Returning "" keeps pipeline working until the real model lands.
"""


def extract_color(crop_bgr) -> str:
    return ""
