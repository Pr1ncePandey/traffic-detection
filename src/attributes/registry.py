"""The perception seam: attribute enrichers, registered by name.

An ENRICHER adds facts to an already-detected, already-tracked object - its
paint colour, its number plate. It is NOT an analysis:

    enricher   object in  -> attributes out.     Runs FIRST, for everything.
    analysis   objects in -> conclusions out.    Reads what enrichers produced.

Keeping them apart is the point. They used to share one flat `analyzers:` list
where `anpr` and `color` were listed after `counting` and `lanes`, so the
enrichers ran last and no analysis could ever read an attribute - "congestion
for trucks only" or "the wrong-way vehicle's plate" were not expressible. They
are now a stage of their own, ahead of the analyses, and every analysis sees
their output on the same frame.

Enrichers share the plugin contract in runtime/plugin.py, so one scheduler runs
both stages. They have no `draw` phase: an enricher states a fact, and how that
fact is shown belongs to whoever draws (pipeline._label puts the plate on the
box caption).
"""

# name -> factory(cfg) -> enricher. Populated by register() at import time.
ENRICHERS: dict = {}


def register(name: str, factory) -> None:
    if name in ENRICHERS:
        raise ValueError(f"enricher {name!r} already registered")
    ENRICHERS[name] = factory


def available() -> list:
    _load_builtins()
    return sorted(ENRICHERS)


def block(cfg: dict, name: str) -> dict:
    """One enricher's config block, from `perception.attributes.<name>`."""
    return ((cfg or {}).get("perception", {})
            .get("attributes", {}).get(str(name), {}) or {})


def enabled_names(cfg: dict) -> list:
    """Which enrichers to run, in declaration order.

    Order is the order of the `perception.attributes:` mapping in config. A
    camera switches one off with `enabled: false` rather than by restating a
    list, which is what makes a per-camera override a one-line change.
    """
    out = []
    attrs = ((cfg or {}).get("perception", {}).get("attributes", {}) or {})
    for name, conf in attrs.items():
        if isinstance(conf, dict) and not conf.get("enabled", True):
            continue
        out.append(str(name))
    return out


def build(cfg: dict, source) -> list:
    """Instantiate the enabled enrichers, in declaration order.

    An unknown name is a loud warning rather than a crash, and a failure to
    start is skipped: a missing OCR backend must not stop vehicle counting.
    """
    _load_builtins()
    _load_optional(enabled_names(cfg))
    built = []
    for name in enabled_names(cfg):
        factory = ENRICHERS.get(name)
        if factory is None:
            print(f"[perception] unknown attribute {name!r}; "
                  f"available: {', '.join(available())}")
            continue
        try:
            enricher = factory(cfg)
            enricher.setup(source, cfg)
            built.append(enricher)
        except Exception as e:
            print(f"[perception] attribute {name!r} failed to start, "
                  f"skipping: {e}")
    return built


_loaded = False


def _load_builtins():
    """Import the shipped enrichers so their register() calls run.

    Imported lazily and defensively: the plate reader pulls in optional OCR
    dependencies, and a missing OCR backend must leave everything else running.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    from .enrichers import color  # noqa: F401
    try:
        from .enrichers import plate  # noqa: F401
    except Exception as e:
        print(f"[perception] plate reader unavailable ({e}); "
              f"plate reading disabled")
    try:
        from .enrichers import person  # noqa: F401
    except Exception as e:
        print(f"[perception] person attributes unavailable ({e}); "
              f"person attribute reading disabled")


# Experimental or heavier person models. Imported only when a config switches
# them on, so a machine without torch/transformers (or without the face
# models) never pays for the import.
OPTIONAL = ("garments", "age_gender", "face")


def _load_optional(names):
    import importlib
    for name in OPTIONAL:
        if name not in names or name in ENRICHERS:
            continue
        try:
            importlib.import_module(f"{__package__}.enrichers.{name}")
        except Exception as e:
            print(f"[perception] {name} unavailable ({e}); disabled")
