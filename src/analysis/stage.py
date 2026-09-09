"""Runs the analyzers: compute (optionally concurrent) -> apply -> draw.

The pipeline owns the loop; this owns the order and the threading, so neither
the pipeline nor any individual analyzer has to know about the other's
scheduling. Adding an analysis is still one file plus a name in config.

WHAT IS AND IS NOT PARALLEL

Only `compute` phases of analyzers that declare `concurrent = True` leave the
main thread. `apply` and `draw` are always serial and always in config order,
because they write to shared state (the TrackStore, the detections, the one
annotated canvas) and because order is load-bearing - counting may want the
lane context that lanes wrote.

Is it worth it? Honest answer, per analysis:

  anpr        yes, clearly. Plate detection plus OCR is tens of milliseconds
              of numpy/ONNX work that releases the GIL.
  congestion  yes, once migrated: it is numpy area/motion arithmetic.
  lanes       not by itself. The wrong-side rule is a few dozen float
              operations per object - pure Python, GIL-bound, and running it
              on a worker costs more in dispatch than it saves.

So `analysis.parallel` defaults to false. Lanes is structured this way not to
make lanes faster but so that it CAN sit next to a heavy analysis without
being serialised behind it, and so no future analysis has to untangle shared
state to get there.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from .base import Findings, is_concurrent, is_staged


class AnalysisStage:
    """Owns the three-phase run over one frame."""

    def __init__(self, analyzers, parallel: bool = False, workers: int | None = None):
        self.analyzers = list(analyzers or [])
        self.staged = [a for a in self.analyzers if is_staged(a)]
        self.legacy = [a for a in self.analyzers if not is_staged(a)]
        self.concurrent = [a for a in self.analyzers if is_concurrent(a)]
        # One worker per concurrent analyzer is the most that can ever be
        # used: the compute phases of a single frame are the unit of work.
        want = int(workers or len(self.concurrent) or 1)
        self.parallel = bool(parallel) and len(self.concurrent) > 1
        self._pool = (ThreadPoolExecutor(max_workers=max(2, want),
                                         thread_name_prefix="analysis")
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

    def _compute_one(self, analyzer, view) -> Findings:
        try:
            found = analyzer.compute(view)
            if found is None:
                return Findings(analyzer=analyzer.name)
            found.analyzer = found.analyzer or analyzer.name
            return found
        except Exception as e:
            return Findings(analyzer=analyzer.name, error=repr(e))

    def compute(self, view) -> dict:
        """name -> Findings for every staged analyzer."""
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
        """Serial, config order. Legacy analyzers run their process() here."""
        by_track = ctx.by_track()
        for analyzer in self.analyzers:
            if not is_staged(analyzer):
                self._guard(analyzer, "process", lambda: analyzer.process(ctx))
                continue
            found = results.get(analyzer.name)
            if found is None:
                continue
            if found.error:
                self._note(analyzer.name, found.error)
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
            self._guard(analyzer, "apply",
                        lambda a=analyzer, f=found: a.apply(ctx, f))

    def draw(self, ctx, results: dict):
        """Serial, config order. Staged overlays only - legacy drew in apply."""
        for analyzer in self.analyzers:
            if not is_staged(analyzer):
                continue
            found = results.get(analyzer.name)
            if found is None or found.error:
                continue
            drawer = getattr(analyzer, "draw", None)
            if drawer is None:
                continue
            self._guard(analyzer, "draw", lambda d=drawer, f=found: d(ctx, f))

    def run(self, ctx) -> dict:
        """compute -> apply -> draw for one frame. Returns the findings."""
        results = self.compute(ctx.view())
        self.apply(ctx, results)
        self.draw(ctx, results)
        self.frames += 1
        return results

    # --- housekeeping ------------------------------------------------------

    def forget(self, tid):
        """Tell every analyzer a track is retired, so per-track state is freed."""
        for analyzer in self.analyzers:
            forget = getattr(analyzer, "forget", None)
            if forget is None:
                continue
            try:
                forget(tid)
            except Exception:
                pass

    def summaries(self) -> dict:
        out = {}
        for analyzer in self.analyzers:
            try:
                out[analyzer.name] = analyzer.summary()
            except Exception as e:
                out[analyzer.name] = {"error": str(e)}
        if self.errors:
            out["_analysis_errors"] = dict(self.errors)
        if self.frames:
            out["_analysis_timing"] = {
                "compute_ms_per_frame": round(
                    1000.0 * self.compute_seconds / self.frames, 2),
                "parallel": self.parallel}
        return out

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _note(self, name, message):
        """Count failures per analyzer instead of printing one per frame.

        A broken analyzer on a 24/7 feed printing every frame is its own
        outage; the count lands in the summary instead.
        """
        entry = self.errors.setdefault(name, {"count": 0, "last": ""})
        entry["count"] += 1
        entry["last"] = str(message)[:200]
        if entry["count"] in (1, 10, 100) or entry["count"] % 1000 == 0:
            print(f"[analysis] {name} failed ({entry['count']}x): {message}")

    def _guard(self, analyzer, phase, call):
        try:
            call()
        except Exception as e:
            self._note(f"{analyzer.name}.{phase}", repr(e))
