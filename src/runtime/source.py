"""Frame sources: video file, RTSP/HTTP stream, or webcam index.

One reason this module exists: the old pipeline did os.path.exists(source),
which rejected every URL before OpenCV ever saw it. The existence check now
lives in FileSource, where it belongs - files still get a clear error, and a
stream is no longer required to be a path on disk.

A stream differs from a file in ways callers must not guess at:
  is_live      - there is no end; max_frames becomes a stop condition, not a total
  is_seekable  - CAP_PROP_POS_FRAMES is a no-op, so tools must not seek
  total_frames - None, so a progress bar must run unbounded
  fps          - frequently reported as 0 or NaN over RTSP; never trust it blindly
"""

import os
import re
import time

import cv2

from dataclasses import dataclass

# RTSP over UDP loses packets and OpenCV then stalls; TCP is slower to start but
# does not wedge. Set before VideoCapture construction - FFMPEG reads it at open.
_FFMPEG_RTSP_OPTS = "rtsp_transport;tcp|stimeout;5000000"

RECONNECT_BACKOFF_START = 0.5
RECONNECT_BACKOFF_MAX = 30.0

DEFAULT_FPS = 30.0          # only used when the source reports nothing usable
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


@dataclass
class SourceInfo:
    width: int
    height: int
    fps: float
    total_frames: int | None
    is_live: bool
    is_seekable: bool
    spec: str = ""

    @property
    def label(self) -> str:
        kind = "live" if self.is_live else "file"
        n = self.total_frames if self.total_frames else "?"
        return f"{self.spec} [{kind}] {self.width}x{self.height} @ {self.fps:.1f}fps, {n} frames"


def _probe(cap, spec: str, is_live: bool) -> SourceInfo:
    fps = cap.get(cv2.CAP_PROP_FPS)
    # RTSP commonly reports 0 or NaN. Anything absurd is also a lie.
    if not fps or fps != fps or fps <= 0 or fps > 240:
        fps = DEFAULT_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return SourceInfo(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(fps),
        total_frames=None if (is_live or total <= 0) else total,
        is_live=is_live,
        is_seekable=not is_live,
        spec=str(spec),
    )


class FileSource:
    """A finite video file. Seekable, has a known length."""

    is_live = False

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input video not found: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.info = _probe(self._cap, path, is_live=False)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


class WebcamSource:
    """A local camera by index. Live, so no length and no seeking."""

    is_live = True

    def __init__(self, index: int):
        self._index = index
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {index}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.info = _probe(self._cap, str(index), is_live=True)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


class StreamSource:
    """An RTSP/HTTP stream. Reconnects on drop, because RTSP drops routinely.

    read() returns (False, None) only when the retry budget is exhausted; a
    transient drop is absorbed here so the pipeline is not written twice.
    """

    is_live = True

    def __init__(self, url: str, reconnect: bool = True, max_retries: int = 0):
        self._url = url
        self._reconnect = reconnect
        self._max_retries = max_retries          # 0 = retry forever
        self._retries = 0
        self._cap = self._open()
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open stream: {url}")
        self.info = _probe(self._cap, url, is_live=True)

    def _open(self):
        # FFMPEG picks this up at construction; harmless for non-RTSP URLs.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _FFMPEG_RTSP_OPTS)
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        # Keep the driver buffer at one frame: we want the newest frame, not a
        # backlog. Without this, falling behind shows up as growing latency.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def read(self):
        ok, frame = self._cap.read()
        if ok:
            self._retries = 0
            return True, frame
        if not self._reconnect:
            return False, None
        return self._reconnect_and_read()

    def _reconnect_and_read(self):
        delay = RECONNECT_BACKOFF_START
        while self._max_retries == 0 or self._retries < self._max_retries:
            self._retries += 1
            print(f"[source] stream dropped, reconnecting in {delay:.1f}s "
                  f"(attempt {self._retries})")
            time.sleep(delay)
            delay = min(delay * 2, RECONNECT_BACKOFF_MAX)
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = self._open()
            if self._cap.isOpened():
                ok, frame = self._cap.read()
                if ok:
                    print("[source] stream reconnected")
                    return True, frame
        print(f"[source] giving up after {self._retries} reconnect attempts")
        return False, None

    def release(self):
        self._cap.release()


def open_source(spec, cfg: dict | None = None):
    """Build the right FrameSource for `spec`.

    int or int-like string -> webcam index
    something://...        -> stream
    anything else          -> file on disk
    """
    cfg = cfg or {}
    if isinstance(spec, int):
        return WebcamSource(spec)
    text = str(spec).strip()
    if text.isdigit():
        return WebcamSource(int(text))
    if _URL_RE.match(text):
        return StreamSource(text,
                            reconnect=bool(cfg.get("reconnect", True)),
                            max_retries=int(cfg.get("max_retries", 0)))
    return FileSource(text)
