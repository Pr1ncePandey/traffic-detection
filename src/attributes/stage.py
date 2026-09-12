"""The PERCEPTION stage: runs the attribute enrichers.

Same scheduler as the analyses (runtime/plugin.py), separate instance, so it
gets its own thread pool. That separation is the point: plate OCR is the
expensive work in the whole loop - tens of milliseconds of ONNX per frame,
which releases the GIL - while the analyses are a few dozen float operations
each. This is the stage where `parallel: true` actually buys something.
"""

from ..runtime.plugin import PluginStage


class AttributeStage(PluginStage):
    """Attribute enrichers. Run before any analysis, on every analysed frame."""

    label = "perception"
