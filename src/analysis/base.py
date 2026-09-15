"""The use-case seam for ANALYSES: one registry, one config block each.

This is the answer to "I might add traffic congestion or other use cases
later". An analysis sees a frame and nothing else - not the video source, not
the storage backend, not the config of another analysis. Adding a use case
means writing one file and giving it a block under `analyses:` in config, with
NO edit to pipeline.py. If a new analysis ever forces a pipeline change, this
seam is wrong and should be fixed rather than worked around.

The PLUGIN CONTRACT itself (Findings, the three phases, the concurrency
declaration) lives in runtime/plugin.py, because the perception stage runs the
same contract for attribute enrichers. It is re-exported here so that
`from .base import Findings` keeps working for the analyses.

Attribute readers are NOT analyses and must not be registered here: an
enricher adds facts to an object, an analysis draws conclusions from them, and
the enrichers have to run first. See attributes/registry.py.
"""

from ..runtime.plugin import (Analyzer, Findings, StagedAnalyzer,  # noqa: F401
                              is_concurrent, is_staged)

# name -> factory(cfg) -> Analyzer. Populated by register() at import time.
ANALYZERS: dict = {}


def register(name: str, factory) -> None:
    if name in ANALYZERS:
        raise ValueError(f"analyzer {name!r} already registered")
    ANALYZERS[name] = factory


def available() -> list:
    # Load first: reporting an empty list before the builtins are imported
    # would be a lie, and this is what error messages print.
    _load_builtins()
    return sorted(ANALYZERS)


def block(cfg: dict, name: str) -> dict:
    """One analysis's own config block, from `analyses.<name>`.

    Every analyzer reads its settings through here, so a per-camera override is
    always the same mechanical path - analyses.congestion.busy_count - and no
    analysis owns top-level keys any more. `lanes` used to own five of them
    (lanes, lanes_mode, lanes_units, lanes_rules, divider), which is why the
    per-camera whitelist could half-apply a camera file.
    """
    return (cfg or {}).get("analyses", {}).get(str(name), {}) or {}


def enabled_names(cfg: dict) -> list:
    """Which analyses to run, in declaration order.

    Order comes from the `analyses:` mapping in config, not from a separate
    list, so enabling one per camera is `enabled: true` in that camera's file
    rather than restating the whole list and silently reordering the rest.
    """
    out = []
    for name, conf in ((cfg or {}).get("analyses", {}) or {}).items():
        if isinstance(conf, dict) and not conf.get("enabled", True):
            continue
        out.append(str(name))
    return out


def build(cfg: dict, source) -> list:
    """Instantiate the enabled analyses, in declaration order.

    An unknown name is a loud warning rather than a crash: a typo in config
    should not take down a running camera, but it must not pass silently
    either.
    """
    _load_builtins()
    built = []
    for name in enabled_names(cfg):
        factory = ANALYZERS.get(name)
        if factory is None:
            print(f"[analysis] unknown analysis {name!r}; "
                  f"available: {', '.join(available())}")
            continue
        try:
            analyzer = factory(cfg)
            analyzer.setup(source, cfg)
            built.append(analyzer)
        except Exception as e:
            print(f"[analysis] analysis {name!r} failed to start, skipping: {e}")
    return built


_loaded = False


def _load_builtins():
    """Import the shipped analyzers so their register() calls run.

    Imported lazily and defensively: ANPR pulls in OCR dependencies that are
    optional, and a missing OCR backend must not stop vehicle counting.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    # Attribute readers are NOT here any more: `anpr` and `color` moved to
    # attributes/enrichers/ and run in the perception stage, ahead of these.
    from . import congestion, counting, heatmap, lanes  # noqa: F401
