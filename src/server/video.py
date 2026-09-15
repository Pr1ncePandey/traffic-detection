"""Fan live JPEG frames out to dashboard viewers.

This is `hub.py` for pixels, and deliberately the same shape: per-camera
subscriber sets, bounded newest-wins queues, `call_soon_threadsafe` from the
camera thread, a fixed push rate independent of `analyse_fps`. Two modules
doing the same fan-out two different ways would be the worse outcome, so where
this looks copied from the hub, it is.

WHY THERE IS VIDEO HERE AT ALL

The original decision was metadata only (docs/webserver-and-incidents.md,
Layer C): boxes drawn client-side over nothing, because ~4 KB of JSON per
frame beats a video stream when the boxes are the payload. What that misses is
that an operator cannot tell a correct box from a wrong one without seeing the
road, so the schematic answers "is the pipeline running" and not "is it right".
The metadata path is unchanged and still carries the boxes; this adds the
picture underneath it.

FRAMES ARE CLEAN, NOT ANNOTATED

`ctx.annotated` already holds a frame with boxes burnt in by OpenCV, and
serving that would have been fewer lines. It is the wrong frame: the dashboard
draws its own boxes from the metadata stream, so burnt-in ones would double up,
and burnt-in labels cannot be toggled, filtered or clicked. The pipeline hands
this sink `raw`.

THREE RULES, TWO INHERITED

**Encoding costs nothing when nobody is watching.** This is the rule the hub
does not need: its payload is already built by the time it is offered one, so
throttling only saves a fan-out. Ours costs a `cv2.imencode`, so a camera with
no viewers must not pay for it. With no stream subscribers this falls back to
`snapshot_fps` (1 Hz) rather than to zero, so `/snapshot` stays useful for
thumbnails - one encode a second, against eight.

**Slow viewers drop frames; they never apply backpressure.** A one-slot
newest-wins queue per subscriber. A stalled browser tab must not be able to
slow a camera thread - the same rule as the hub's, and as the write queue's.

**Throttled independently of the pipeline.** A file replaying at 3x real time
would otherwise encode 90 frames a second for a viewer who can see eight.
"""

import asyncio
import threading
import time

import cv2

STREAM_FPS = 8.0          # matches the hub's default; they are independent
SNAPSHOT_FPS = 1.0        # cadence when nobody is streaming, for /snapshot
JPEG_QUALITY = 70         # visually fine for road footage at 960 px
MAX_WIDTH = 960
SUBSCRIBER_QUEUE = 1      # newest-wins: a stale video frame is worthless


class VideoSubscriber:
    """One MJPEG connection's slot.

    The queue is asyncio's but `offer()` is called from a camera THREAD, so
    every touch goes through `call_soon_threadsafe` on the loop that owns it -
    identical to hub.Subscriber, for the identical reason.
    """

    def __init__(self, camera: str, loop, maxsize: int = SUBSCRIBER_QUEUE):
        self.camera = camera
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.sent = 0

    def offer(self, jpeg: bytes) -> None:
        """Called from a camera thread. Never blocks, never raises."""
        try:
            self.loop.call_soon_threadsafe(self._push, jpeg)
        except RuntimeError:
            pass          # loop already closed: this subscriber is gone

    def _push(self, jpeg: bytes) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()       # evict the stale frame
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self.queue.put_nowait(jpeg)
        except asyncio.QueueFull:
            self.dropped += 1


class VideoSink:
    """Per-camera JPEG encoder, throttle and fan-out."""

    def __init__(self, enabled: bool = True, fps: float = STREAM_FPS,
                 quality: int = JPEG_QUALITY, max_width: int = MAX_WIDTH,
                 snapshot_fps: float = SNAPSHOT_FPS):
        self.enabled = bool(enabled)
        self.fps = float(fps or STREAM_FPS)
        self.snapshot_fps = float(snapshot_fps or SNAPSHOT_FPS)
        self.quality = int(quality)
        self.max_width = int(max_width or 0)
        self._subs: dict[str, set] = {}
        self._latest: dict[str, bytes] = {}
        self._latest_at: dict[str, float] = {}
        self._last_encode: dict[str, float] = {}
        self._lock = threading.Lock()
        self.encoded = 0
        self.throttled = 0
        self.encode_ms = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "VideoSink":
        v = ((cfg.get("server", {}) or {}).get("video", {}) or {})
        return cls(enabled=v.get("enabled", True),
                   fps=v.get("fps", STREAM_FPS),
                   quality=v.get("quality", JPEG_QUALITY),
                   max_width=v.get("max_width", MAX_WIDTH),
                   snapshot_fps=v.get("snapshot_fps", SNAPSHOT_FPS))

    # --- subscriber side (async, one per connection) ----------------------
    def subscribe(self, camera: str, loop) -> VideoSubscriber:
        sub = VideoSubscriber(camera, loop)
        with self._lock:
            self._subs.setdefault(camera, set()).add(sub)
            latest = self._latest.get(camera)
        # Hand over the last frame immediately so the <img> paints now rather
        # than staying blank until the next encode tick.
        if latest is not None:
            sub.offer(latest)
        return sub

    def unsubscribe(self, sub: VideoSubscriber) -> None:
        with self._lock:
            peers = self._subs.get(sub.camera)
            if peers:
                peers.discard(sub)

    def viewers(self, camera: str | None = None) -> int:
        with self._lock:
            if camera is not None:
                return len(self._subs.get(camera, ()))
            return sum(len(s) for s in self._subs.values())

    # --- pipeline side (called from a camera thread, per frame) -----------
    def offer(self, camera: str, frame) -> bool:
        """Encode and fan out one frame. Returns whether it was encoded.

        Cheap to call and safe to call at full frame rate: the throttle and
        the viewer check both happen before any pixels are touched.
        """
        if not self.enabled or frame is None:
            return False
        now = time.monotonic()
        with self._lock:
            targets = list(self._subs.get(camera, ()))
            # No viewers still encodes, but at the snapshot cadence. Zero would
            # leave /snapshot serving a frame from whenever the last tab closed.
            interval = 1.0 / (self.fps if targets else self.snapshot_fps)
            if (now - self._last_encode.get(camera, 0.0)) < interval:
                self.throttled += 1
                return False
            self._last_encode[camera] = now

        jpeg = self._encode(frame)
        if jpeg is None:
            return False
        with self._lock:
            self._latest[camera] = jpeg
            self._latest_at[camera] = time.time()
            self.encoded += 1
        for sub in targets:
            sub.offer(jpeg)
        return True

    def _encode(self, frame) -> bytes | None:
        """Downscale (aspect preserved) and JPEG-encode. Never raises.

        ASPECT RATIO IS LOAD-BEARING. The dashboard overlays a canvas sized to
        the camera's SOURCE resolution on top of this image and lets CSS
        stretch both to the same box. That only lines up while this scales
        height by the same factor as width - a fixed output size would put
        every box a few pixels off its object, which looks like a tracking bug
        and is not one.
        """
        t0 = time.monotonic()
        try:
            h, w = frame.shape[:2]
            if self.max_width and w > self.max_width:
                scale = self.max_width / float(w)
                frame = cv2.resize(frame, (self.max_width, max(1, round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                return None
            self.encode_ms += (time.monotonic() - t0) * 1000.0
            return buf.tobytes()
        except Exception as e:                       # pragma: no cover
            # A broken frame must not kill the camera thread that handed it in.
            print(f"[video] encode failed: {type(e).__name__}: {e}")
            return None

    # --- readers ----------------------------------------------------------
    def latest(self, camera: str) -> bytes | None:
        with self._lock:
            return self._latest.get(camera)

    def age(self, camera: str) -> float | None:
        with self._lock:
            at = self._latest_at.get(camera)
        return None if at is None else round(time.time() - at, 2)

    def forget(self, camera: str) -> None:
        """Drop a stopped camera's last frame.

        Same reason as hub.forget: a stopped feed showing its final frame
        forever is indistinguishable from a running one that has frozen.
        """
        with self._lock:
            self._latest.pop(camera, None)
            self._latest_at.pop(camera, None)
            self._last_encode.pop(camera, None)

    def stats(self) -> dict:
        with self._lock:
            per_camera = {c: len(s) for c, s in self._subs.items() if s}
            dropped = sum(sub.dropped for subs in self._subs.values()
                          for sub in subs)
            encoded = self.encoded
        return {"enabled": self.enabled, "fps": self.fps,
                "quality": self.quality, "max_width": self.max_width,
                "viewers": per_camera, "encoded": encoded,
                "throttled": self.throttled, "viewer_drops": dropped,
                "avg_encode_ms": (round(self.encode_ms / encoded, 2)
                                  if encoded else None)}
