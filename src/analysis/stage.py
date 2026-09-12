"""The ANALYSIS stage: runs the analyses that consume tracked, enriched objects.

The scheduling machinery is shared with the perception stage and lives in
runtime/plugin.py - this is only the analysis-flavoured instance of it, so the
two stages cannot drift apart in how they order or thread their phases.

WHAT IS AND IS NOT PARALLEL

Only `compute` phases of analyses that declare `concurrent = True` leave the
main thread. `apply` and `draw` are always serial and always in declaration
order, because they write to shared state (the TrackStore, the detections, the
one annotated canvas) and because order is load-bearing.

Is it worth it, honestly, per analysis:

  lanes       not by itself. The wrong-side rule is a few dozen float
              operations per object - pure Python, GIL-bound, and running it on
              a worker costs more in dispatch than it saves.
  congestion  marginally: numpy area/motion arithmetic over a few dozen boxes.
  counting    no. It is a handful of comparisons.

The real win is in the PERCEPTION stage, where plate OCR is tens of
milliseconds of ONNX per frame. So `analysis.parallel` still defaults to false.
Analyses are structured this way not to make them individually faster but so
that one CAN sit next to a heavy one without being serialised behind it.
"""

from ..runtime.plugin import Findings, PluginStage, is_concurrent, is_staged  # noqa: F401


class AnalysisStage(PluginStage):
    """The analyses: counting, wrong-way, congestion. Independent by contract."""

    label = "analysis"
