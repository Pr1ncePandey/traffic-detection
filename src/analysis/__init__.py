"""Frame-level analysis stubs. Vehicle writer + frame analytics run in parallel.

- LineCounter: IN/OUT crossing + drawing (live now).
- SpeedEstimator / LaneMonitor / CongestionAnalyzer: stubs with fixed interfaces.
  Next build fills them; pipeline already calls them every frame.
"""
from .line_counter import LineCounter  # noqa: F401
