"""Decide which frames to analyse, at a configurable target FPS.

Replaces the old integer `frame_skip` modulo, which could only express
1/1, 1/2, 1/3... of the source rate and said nothing about actual frames
per second.

Two modes, because the right answer genuinely differs:

  file mode  - sample on FRAME INDEX. A given file always yields the same
               frames, so runs are reproducible and comparable.
  live mode  - sample on WALL CLOCK. A stream's reported fps is often 0 or
               NaN (see source.DEFAULT_FPS) and its true rate drifts with
               network conditions, so indexing on frame number would drift
               with it. Asking for 5 fps must mean 5 frames a second.
"""

import time


class FpsSampler:
    def __init__(self, source_fps: float, target_fps: float | None, is_live: bool = False):
        self.source_fps = float(source_fps) if source_fps and source_fps > 0 else 30.0
        # None / <=0 / >= source means "every frame".
        self.target_fps = float(target_fps) if target_fps and target_fps > 0 else None
        self.is_live = bool(is_live)
        self.passthrough = self.target_fps is None or self.target_fps >= self.source_fps
        self._step = 1.0 if self.passthrough else self.source_fps / self.target_fps
        self._next_index = 0.0
        self._interval = 0.0 if self.passthrough else 1.0 / self.target_fps
        self._last_t = None

    @property
    def effective_fps(self) -> float:
        """The rate we actually expect to analyse at."""
        return self.source_fps if self.passthrough else self.target_fps

    def should_process(self, frame_index: int, now: float | None = None) -> bool:
        """frame_index is 0-based over frames READ from the source."""
        if self.passthrough:
            return True
        if self.is_live:
            t = time.monotonic() if now is None else now
            if self._last_t is None or (t - self._last_t) >= self._interval:
                self._last_t = t
                return True
            return False
        if frame_index >= self._next_index:
            self._next_index += self._step
            return True
        return False

    def describe(self) -> str:
        if self.passthrough:
            return f"every frame ({self.source_fps:.1f} fps)"
        mode = "wall-clock" if self.is_live else "frame-index"
        return (f"{self.target_fps:.1f} fps target from {self.source_fps:.1f} fps source "
                f"({mode}, ~1 in {self._step:.2f})")


def from_config(cfg: dict, source_fps: float, is_live: bool) -> FpsSampler:
    """Build from the `processing` block.

    analyse_fps is the only spelling. The old integer `frame_skip` is gone: it
    could express only 1/1, 1/2, 1/3... of the source rate and said nothing
    about frames per second, and carrying both meant two ways to say one thing.
    """
    proc = cfg or {}
    target = proc.get("analyse_fps")
    return FpsSampler(source_fps, target if target not in ("", 0) else None,
                      is_live=is_live)
