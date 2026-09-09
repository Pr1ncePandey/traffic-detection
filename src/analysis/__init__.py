"""Analysis use cases. Each is an Analyzer registered by name and enabled in
config; the pipeline knows none of them individually.

Shipped: counting (zone A/B), lanes (wrong-way, wrong-lane), anpr (number
plate), congestion (density/occupancy/motion).
"""

from .base import ANALYZERS, Analyzer, available, build, register  # noqa: F401
