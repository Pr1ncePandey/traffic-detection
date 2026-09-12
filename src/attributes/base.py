"""Attribute helpers shared by the enrichers in enrichers/.

`run_attributes()` used to live here: one function that ran every extractor and
accumulated its results into the TRACKER's vehicle dict. That shared write is
what stopped attribute reading from running off the main thread, so each
enricher now owns its own accumulation state and does the one write to the
store in its `apply` phase. See attributes/enrichers/plate.py.

What remains is the re-export surface the tools import.
"""

from .color import extract_color, extract_color_conf, garment_colors  # noqa: F401
from .plate import configure as configure_plate  # noqa: F401
from .plate import (crop_score, new_read_state, read_plate,  # noqa: F401
                    read_plate_tracked)

MIN_PLATE_W = 80        # skip tiny crops (CPU saving, OCR would fail anyway)
