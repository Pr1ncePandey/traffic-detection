"""The use-case seam: one Analyzer protocol, one registry.

This is the answer to "I might add traffic congestion or other use cases
later". An analyzer sees a frame and nothing else - not the video source, not
the storage backend, not the config of other analyzers. Adding a use case
means writing one file and naming it in config, with NO edit to pipeline.py.
If a new analyzer ever forces a pipeline change, this seam is wrong and should
be fixed rather than worked around.

THREE PHASES, SO ANALYSES CAN RUN SIDE BY SIDE

Wrong-side detection, congestion, counting and ANPR are independent analyses
of the same already-classified, already-tracked objects. They should be able
to run concurrently. What stopped that was not the analyzer list - it was that
`process(ctx)` did three different kinds of work at once:

    reading detections      - concurrency-safe
    mutating the TrackStore  - shared state, not safe
    drawing on ctx.annotated - one shared canvas, not safe

So the work is split, and only the safe part is parallelisable:

    compute(view) -> Findings   PURE. Reads an immutable AnalysisView, keeps
                                its own state, returns a value. No shared
                                mutation, no drawing. Runs on any thread.
    apply(ctx, findings)        SERIAL, in config order. Writes findings onto
                                detections and the store, emits events.
    draw(ctx, findings)         SERIAL, in config order. Overlay only.

An analyzer that implements `compute` is staged and can be parallelised. One
that implements only `process` still works exactly as before, serially - see
stage.AnalysisStage. That is deliberate: the migration is per-analyzer, not a
flag day.

Ordering still matters and is still the config list's order. Parallelism
applies to the compute phase only, whose results are order-independent by
construction; apply and draw are replayed in the order given.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Findings:
    """What one analyzer concluded about one frame.

    A value object, so a compute phase can hand its conclusions back from a
    worker thread with nothing shared but this.
    """

    analyzer: str = ""
    # track_id -> fields to write onto that Detection (lane_id, lane_flag, ...)
    per_track: dict = field(default_factory=dict)
    # frame-level conclusions (congestion level, occupancy, ...)
    frame: dict = field(default_factory=dict)
    # (kind, detail, track_id) tuples, replayed through ctx.emit in apply
    events: list = field(default_factory=list)
    # anything the draw phase needs that is not worth recomputing
    overlay: Any = None
    # set when compute raised; apply/draw are skipped and the run continues
    error: str | None = None

    def set(self, track_id, **fields):
        if track_id is None:
            return
        self.per_track.setdefault(track_id, {}).update(fields)

    def event(self, kind: str, detail: dict | None = None, track_id=None):
        self.events.append((str(kind), dict(detail or {}), track_id))

    def is_empty(self) -> bool:
        return not (self.per_track or self.frame or self.events
                    or self.overlay is not None)


@runtime_checkable
class Analyzer(Protocol):
    """The legacy single-phase shape. Still supported, always serial."""

    name: str

    def setup(self, source, cfg: dict) -> None:
        """Called once, after frame geometry is known."""

    def process(self, ctx) -> None:
        """Called per analysed frame. May mutate ctx.annotated and detections."""

    def summary(self) -> dict:
        """End-of-run figures for the report."""


@runtime_checkable
class StagedAnalyzer(Protocol):
    """The three-phase shape. `compute` is what makes an analyzer schedulable.

    `concurrent` is the analyzer's own declaration that its compute phase
    touches nothing shared. Default it to False in a new analyzer and flip it
    to True once that is actually true - a wrong True is a data race, and the
    scheduler cannot check the claim for you.
    """

    name: str
    concurrent: bool

    def setup(self, source, cfg: dict) -> None: ...

    def compute(self, view) -> Findings:
        """Pure. AnalysisView in, Findings out. No shared writes, no drawing."""

    def apply(self, ctx, findings: Findings) -> None:
        """Serial. Write findings to detections/store, emit events."""

    def draw(self, ctx, findings: Findings) -> None:
        """Serial. Overlay only."""

    def summary(self) -> dict: ...


def is_staged(analyzer) -> bool:
    return callable(getattr(analyzer, "compute", None))


def is_concurrent(analyzer) -> bool:
    """Only a staged analyzer that opts in may leave the main thread."""
    return is_staged(analyzer) and bool(getattr(analyzer, "concurrent", False))


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


def build(enabled, cfg: dict, source) -> list:
    """Instantiate the enabled analyzers, in the order given.

    An unknown name is a loud warning rather than a crash: a typo in config
    should not take down a running camera, but it must not pass silently
    either.
    """
    _load_builtins()
    built = []
    for name in (enabled or []):
        factory = ANALYZERS.get(str(name))
        if factory is None:
            print(f"[analysis] unknown analyzer {name!r}; "
                  f"available: {', '.join(available())}")
            continue
        try:
            analyzer = factory(cfg)
            analyzer.setup(source, cfg)
            built.append(analyzer)
        except Exception as e:
            print(f"[analysis] analyzer {name!r} failed to start, skipping: {e}")
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
    from . import congestion, counting, lanes  # noqa: F401
    try:
        from . import anpr  # noqa: F401
    except Exception as e:
        print(f"[analysis] anpr unavailable ({e}); plate reading disabled")
