"""Fan frame metadata out to dashboard subscribers.

TWO RULES, BOTH ABOUT NOT LETTING A VIEWER HURT A CAMERA

**Throttled independently of analyse_fps.** Pushed at a fixed rate (~8 Hz by
default) however fast the pipeline runs. Nobody can read 30 updates a second,
and a file replaying at 3x real time would otherwise flood the socket with
frames no human will see.

**Slow subscribers drop frames; they never apply backpressure.** Each
subscriber holds a small bounded queue and the newest frame evicts the oldest
when it is full. A stalled browser tab must not be able to slow a camera - this
is the same rule as the write queue's drop policy, one level up.

The hub is deliberately ignorant of WebSockets: it hands out queues and the
endpoint in app.py does the sending. That keeps the pipeline-facing side
synchronous and testable without an event loop.
"""

import asyncio
import threading
import time

PUSH_HZ = 8.0
SUBSCRIBER_QUEUE = 4          # newest-wins; a viewer 4 frames behind is stale
                              # anyway, so a deeper buffer only adds latency


class Subscriber:
    """One dashboard connection's queue.

    The queue is asyncio's, but `offer()` is called from a camera THREAD, so
    every touch of it goes through call_soon_threadsafe on the loop that owns
    it. Doing otherwise works until it does not, under load, intermittently.
    """

    def __init__(self, camera: str, loop, maxsize: int = SUBSCRIBER_QUEUE):
        self.camera = camera
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.sent = 0

    def offer(self, message: dict) -> None:
        """Called from a camera thread. Never blocks, never raises."""
        try:
            self.loop.call_soon_threadsafe(self._push, message)
        except RuntimeError:
            pass          # loop already closed: this subscriber is gone

    def _push(self, message: dict) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()      # evict the oldest
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            self.dropped += 1


class LiveHub:
    """Per-camera subscriber registry plus the push-rate throttle."""

    def __init__(self, push_hz: float = PUSH_HZ):
        self.min_interval = 1.0 / float(push_hz or PUSH_HZ)
        self._subs: dict[str, set] = {}
        self._last_push: dict[str, float] = {}
        self._latest: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.published = 0
        self.throttled = 0

    # --- subscriber side (async, one per connection) -----------------------
    def subscribe(self, camera: str, loop) -> Subscriber:
        sub = Subscriber(camera, loop)
        with self._lock:
            self._subs.setdefault(camera, set()).add(sub)
            # Hand over the last frame immediately so a tab that just opened
            # renders boxes now rather than after the next push tick.
            latest = self._latest.get(camera)
        if latest is not None:
            sub.offer(latest)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            peers = self._subs.get(sub.camera)
            if peers:
                peers.discard(sub)

    def viewers(self, camera: str | None = None) -> int:
        with self._lock:
            if camera is not None:
                return len(self._subs.get(camera, ()))
            return sum(len(s) for s in self._subs.values())

    # --- pipeline side (called from a camera thread, per frame) ------------
    def publish(self, message: dict) -> bool:
        """Offer one frame's metadata. Returns whether it was pushed.

        The latest message is always retained even when throttled, so a new
        subscriber and the /cameras/{id} endpoint see current state rather than
        whatever happened to fall on a push tick.
        """
        camera = message.get("camera") or ""
        now = time.monotonic()
        with self._lock:
            self._latest[camera] = message
            if (now - self._last_push.get(camera, 0.0)) < self.min_interval:
                self.throttled += 1
                return False
            self._last_push[camera] = now
            targets = list(self._subs.get(camera, ()))
        for sub in targets:
            sub.offer(message)
        self.published += 1
        return True

    def latest(self, camera: str) -> dict | None:
        with self._lock:
            return self._latest.get(camera)

    def forget(self, camera: str) -> None:
        """Drop a stopped camera's retained frame so the dashboard does not
        keep showing boxes from a feed that is no longer running."""
        with self._lock:
            self._latest.pop(camera, None)
            self._last_push.pop(camera, None)

    def stats(self) -> dict:
        with self._lock:
            per_camera = {c: len(s) for c, s in self._subs.items()}
            dropped = sum(sub.dropped for subs in self._subs.values()
                          for sub in subs)
        return {"push_hz": round(1.0 / self.min_interval, 1),
                "viewers": per_camera, "published": self.published,
                "throttled": self.throttled, "viewer_drops": dropped}
