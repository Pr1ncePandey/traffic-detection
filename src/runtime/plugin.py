"""The plugin contract and the scheduler that runs a set of plugins.

There are TWO stages of plugins over the same tracked objects, and they are
different kinds of thing - which is the distinction this module exists to make
possible:

  PERCEPTION (attributes)  enrichers. They ADD facts to an object: its paint
                           colour, its number plate. They run first, so their
                           output is available to everything after them.
  ANALYSIS                 consumers. They draw CONCLUSIONS about objects or
                           about the frame: wrong-way, congestion, counts.

Both shapes are scheduled identically, so the machinery lives here once and
each stage subclasses it. Previously `anpr` and `color` were listed as peers of
`counting` and `lanes` in one flat, hand-ordered list, which meant the
enrichers ran AFTER the consumers and no analysis could ever read an attribute.

THREE PHASES, SO WORK CAN RUN SIDE BY SIDE

A single `process(ctx)` did three kinds of work at once:

    reading detections       - concurrency-safe
    mutating the TrackStore  - shared state, not safe
    drawing on ctx.annotated - one shared canvas, not safe

So the work is split, and only the safe part is parallelisable:

    compute(view) -> Findings   PURE. Reads an immutable AnalysisView, keeps
                                its own state, returns a value. No shared
                                mutation, no drawing. Runs on any thread.
    apply(ctx, findings)        SERIAL, in declaration order. Writes findings
                                onto detections and the store, emits events.
    draw(ctx, findings)         SERIAL, in declaration order. Overlay only.

A plugin implementing `compute` is staged and can be parallelised. One
implementing only `process` still works, serially. That is deliberate: the
migration is per-plugin, not a flag day.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Findings:
    """What one plugin concluded about one frame.

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
    """The three-phase shape. `compute` is what makes a plugin schedulable.

    `concurrent` is the plugin's own declaration that its compute phase touches
    nothing shared. Default it to False in a new plugin and flip it to True
    once that is actually true - a wrong True is a data race, and the scheduler
    cannot check the claim for you.
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


def is_staged(plugin) -> bool:
    return callable(getattr(plugin, "compute", None))


def is_concurrent(plugin) -> bool:
    """Only a staged plugin that opts in may leave the main thread."""
    return is_staged(plugin) and bool(getattr(plugin, "concurrent", False))


class PluginStage:
    """Owns the three-phase run over one frame for one set of plugins.

    Each stage instance holds its OWN thread pool. That matters: the expensive
    work in the whole loop is plate OCR in the perception stage (tens of ms of
    ONNX per frame, which releases the GIL), while the analyses are a few dozen
    float operations each. Sharing one pool would let the cheap stage queue
    behind the expensive one for no gain.
    """

    label = "plugins"

    def __init__(self, plugins, parallel: bool = False,
                 workers: int | None = None):
        self.analyzers = list(plugins or [])
        self.staged = [a for a in self.analyzers if is_staged(a)]
        self.legacy = [a for a in self.analyzers if not is_staged(a)]
        self.concurrent = [a for a in self.analyzers if is_concurrent(a)]
        # One worker per concurrent plugin is the most that can ever be used:
        # the compute phases of a single frame are the unit of work.
        want = int(workers or len(self.concurrent) or 1)
        # Parallelism pays only when there is something to OVERLAP WITH: at
        # least one concurrent plugin, and more than one staged unit of compute
        # work. One concurrent plugin on its own would be submitted to the pool
        # and immediately joined - pure dispatch overhead.
        #
        # The old gate was `len(self.concurrent) > 1`, which declined a
        # legitimate case: one concurrent plugin CAN run on a worker while the
        # main thread computes the staged-but-not-concurrent ones. Counting
        # staged units rather than concurrent ones captures both.
        self.parallel = (bool(parallel) and bool(self.concurrent)
                         and len(self.staged) > 1)
        self._pool = (ThreadPoolExecutor(max_workers=max(2, want),
                                         thread_name_prefix=self.label)
                      if self.parallel else None)
        self.compute_seconds = 0.0
        self.frames = 0
        self.errors: dict = {}

    def describe(self) -> str:
        names = [a.name for a in self.analyzers] or ["none"]
        if not self.staged:
            return f"{', '.join(names)} (all single-phase, serial)"
        mode = (f"compute in parallel x{self._pool._max_workers}"
                if self.parallel else "compute serial")
        return (f"{', '.join(names)} | staged: "
                f"{', '.join(a.name for a in self.staged)} | {mode}")

    # --- phases ------------------------------------------------------------

    def _compute_one(self, plugin, view) -> Findings:
        try:
            found = plugin.compute(view)
            if found is None:
                return Findings(analyzer=plugin.name)
            found.analyzer = found.analyzer or plugin.name
            return found
        except Exception as e:
            return Findings(analyzer=plugin.name, error=repr(e))

    def compute(self, view) -> dict:
        """name -> Findings for every staged plugin."""
        if not self.staged:
            return {}
        start = time.perf_counter()
        results: dict = {}
        if self.parallel:
            futures = {a.name: self._pool.submit(self._compute_one, a, view)
                       for a in self.concurrent}
            # Anything staged but not concurrent-safe runs here on the main
            # thread while the pool works, rather than being skipped.
            for a in self.staged:
                if a.name not in futures:
                    results[a.name] = self._compute_one(a, view)
            for name, fut in futures.items():
                results[name] = fut.result()
        else:
            for a in self.staged:
                results[a.name] = self._compute_one(a, view)
        self.compute_seconds += time.perf_counter() - start
        return results

    def apply(self, ctx, results: dict):
        """Serial, declaration order. Legacy plugins run their process() here."""
        by_track = ctx.by_track()
        for plugin in self.analyzers:
            if not is_staged(plugin):
                self._guard(plugin, "process", lambda p=plugin: p.process(ctx))
                continue
            found = results.get(plugin.name)
            if found is None:
                continue
            if found.error:
                self._note(plugin.name, found.error)
                continue
            for tid, fields in found.per_track.items():
                det = by_track.get(tid)
                if det is None:
                    continue
                for key, value in fields.items():
                    if key == "extra" and isinstance(value, dict):
                        det.extra.update(value)
                    elif hasattr(det, key):
                        setattr(det, key, value)
                    else:
                        det.extra[key] = value
            for kind, detail, tid in found.events:
                ctx.emit(kind, detail, track_id=tid)
            self._guard(plugin, "apply",
                        lambda p=plugin, f=found: p.apply(ctx, f))

    def draw(self, ctx, results: dict):
        """Serial, declaration order. Staged overlays only."""
        for plugin in self.analyzers:
            if not is_staged(plugin):
                continue
            found = results.get(plugin.name)
            if found is None or found.error:
                continue
            drawer = getattr(plugin, "draw", None)
            if drawer is None:
                continue
            self._guard(plugin, "draw", lambda d=drawer, f=found: d(ctx, f))

    def run(self, ctx) -> dict:
        """compute -> apply -> draw for one frame. Returns the findings."""
        results = self.compute(ctx.view())
        self.apply(ctx, results)
        self.draw(ctx, results)
        self.frames += 1
        return results

    # --- housekeeping ------------------------------------------------------

    def forget(self, tid):
        """Tell every plugin a track is retired, so per-track state is freed."""
        for plugin in self.analyzers:
            forget = getattr(plugin, "forget", None)
            if forget is None:
                continue
            try:
                forget(tid)
            except Exception:
                pass

    def summaries(self) -> dict:
        out = {}
        for plugin in self.analyzers:
            try:
                out[plugin.name] = plugin.summary()
            except Exception as e:
                out[plugin.name] = {"error": str(e)}
        if self.errors:
            out[f"_{self.label}_errors"] = dict(self.errors)
        if self.frames:
            out[f"_{self.label}_timing"] = {
                "compute_ms_per_frame": round(
                    1000.0 * self.compute_seconds / self.frames, 2),
                "parallel": self.parallel}
        return out

    def close(self):
        # A plugin may own a thread of its own (the face enricher's AdaFace
        # worker on a live source); it gets the chance to stop it here.
        for plugin in self.analyzers:
            closer = getattr(plugin, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as e:
                    print(f"[{self.label}] {plugin.name} failed to close: {e}")
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _note(self, name, message):
        """Count failures per plugin instead of printing one per frame.

        A broken plugin on a 24/7 feed printing every frame is its own outage;
        the count lands in the summary instead.
        """
        entry = self.errors.setdefault(name, {"count": 0, "last": ""})
        entry["count"] += 1
        entry["last"] = str(message)[:200]
        if entry["count"] in (1, 10, 100) or entry["count"] % 1000 == 0:
            print(f"[{self.label}] {name} failed ({entry['count']}x): {message}")

    def _guard(self, plugin, phase, call):
        try:
            call()
        except Exception as e:
            self._note(f"{plugin.name}.{phase}", repr(e))
