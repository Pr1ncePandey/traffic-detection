"""Decode frames on a background thread so inference never waits on I/O.

Backpressure is the interesting part, and it is per camera because the right
answer differs by source:

  drop_oldest  (default for live) - the queue holds only the newest frames; if
                analysis falls behind, stale frames are discarded so latency
                stays flat. You lose frames, which makes tracking choppier, but
                a monitoring feed that runs 40s behind is worse than useless.
  buffer_all   (default for files) - the reader blocks when the queue is full,
                so no frame is ever lost. Correct for a file, where there is no
                realtime to keep up with; on a live stream it would grow lag
                without bound.
"""

import queue
import threading

# drop_oldest wants a SHALLOW queue - its whole purpose is to not accumulate.
# buffer_all wants a deep one to absorb inference jitter without stalling decode.
QUEUE_DROP = 4
QUEUE_BUFFER = 256

DROP_OLDEST = "drop_oldest"
BUFFER_ALL = "buffer_all"


class FrameReader:
    """Wraps a FrameSource in a reader thread. Iterate with .frames()."""

    def __init__(self, source, backpressure: str | None = None, queue_size: int | None = None):
        self.source = source
        if backpressure not in (DROP_OLDEST, BUFFER_ALL):
            # Default by source kind, which is the sane thing for each.
            backpressure = DROP_OLDEST if source.info.is_live else BUFFER_ALL
        self.backpressure = backpressure
        if queue_size is None:
            queue_size = QUEUE_DROP if backpressure == DROP_OLDEST else QUEUE_BUFFER
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_read = 0
        self.frames_dropped = 0

    def start(self):
        self._thread = threading.Thread(target=self._pump, name="frame-reader", daemon=True)
        self._thread.start()
        return self

    def _pump(self):
        index = 0
        try:
            while not self._stop.is_set():
                ok, frame = self.source.read()
                if not ok:
                    break
                item = (index, frame)
                index += 1
                self.frames_read = index
                if self.backpressure == BUFFER_ALL:
                    # Block, but wake periodically so stop() is honoured promptly.
                    while not self._stop.is_set():
                        try:
                            self._q.put(item, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                else:
                    self._put_newest(item)
        finally:
            # Sentinel: unblocks a consumer waiting on an exhausted source.
            try:
                self._q.put(None, timeout=1.0)
            except queue.Full:
                pass

    def _put_newest(self, item):
        """Keep the newest frame, evicting the oldest when full."""
        while True:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass  # a consumer beat us to it; retry the put

    def frames(self):
        """Yield (frame_index, frame) until the source ends or stop() is called."""
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Drain so a blocked reader thread can reach its finally clause.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def describe(self) -> str:
        return f"{self.backpressure} (queue={self._q.maxsize})"
