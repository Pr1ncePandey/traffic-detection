"""Analysis use cases. Each is an Analyzer registered by name and enabled in
config; the pipeline knows none of them individually.

Shipped: counting (zone A/B), lanes (wrong-way, wrong-lane), congestion
(density/occupancy/motion). Attribute readers (plate, colour) are NOT analyses
- they run earlier, in the perception stage; see attributes/registry.py.
"""

from .base import (ANALYZERS, Analyzer, available, block, build,  # noqa: F401
                   enabled_names, register)
