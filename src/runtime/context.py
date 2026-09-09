"""Per-frame data passed to analyzers.

There are two shapes, and the difference is the whole point:

  AnalysisView   READ-ONLY. What the compute phase of an analyzer sees. No
                 frame to draw on, no TrackStore to mutate, no other
                 analyzer's output. A compute phase that reads only this can
                 be moved to a worker thread without a single lock.

  FrameContext   the mutable frame. Only the apply and draw phases get it,
                 and those run serially in config order.

Detection is the normalized detector output, so swapping YOLO for anything
else touches one adapter instead of every analyzer.
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np


@dataclass
class Detection:
    """One detected object in one frame, detector-agnostic.

    track_id is None when the tracker did not assign an id this frame (low
    confidence, or tracking disabled) - analyzers must handle that rather than
    assume an id, because ByteTrack legitimately withholds ids.
    """

    cls_id: int
    cls_name: str
    group: str                       # vehicle|person|animal|other, see detectors/classes.py
    conf: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    track_id: int | None = None
    # Written by analyzers, persisted by the pipeline. First-class rather than
    # a side dict because the detections table has columns for them.
    event: str = ""                  # zone crossing, e.g. A_TO_B
    lane_id: str = ""
    lane_flag: str = ""              # ok | wrong_way | wrong_lane
    extra: dict = field(default_factory=dict)   # anything an analyzer wants to carry

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def ground_point(self) -> tuple[int, int]:
        """Bottom-centre: where the object meets the road.

        Preferred over the centroid for anything positional. The centroid of a
        box clipped by the bottom edge drifts UPWARD as the object approaches
        - the box stops growing downward while its top keeps rising - which
        made approaching vehicles read as travelling away.
        """
        x1, _y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, y2

    @property
    def height(self) -> int:
        """Box height in px. The cheapest available proxy for distance."""
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class TrackedBox(NamedTuple):
    """An immutable snapshot of one Detection, for the compute phase.

    A tuple rather than the Detection itself so that a compute phase CANNOT
    write to shared state even by accident - the "pure function" contract is
    enforced by the type instead of by a comment nobody reads.
    """

    track_id: int | None
    cls_name: str
    group: str
    conf: float
    bbox: tuple[int, int, int, int]

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def ground_point(self) -> tuple[int, int]:
        x1, _y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, y2

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @classmethod
    def of(cls, det: Detection) -> "TrackedBox":
        return cls(det.track_id, det.cls_name, det.group, det.conf, det.bbox)


@dataclass(frozen=True)
class AnalysisView:
    """Read-only snapshot of one analysed frame."""

    frame_no: int
    timestamp: float
    boxes: tuple[TrackedBox, ...]
    width: int
    height: int
    raw: np.ndarray | None = None    # pixels, for analyzers that need them
    source: Any = None

    def tracked(self):
        """Boxes the tracker gave an id to - the ones with usable history."""
        return tuple(b for b in self.boxes if b.track_id is not None)

    def of_group(self, *groups):
        return tuple(b for b in self.boxes if b.group in groups)


@dataclass
class FrameContext:
    """One analysed frame, mutable.

    raw is never mutated - analyzers that draw must draw on `annotated`, so the
    stored original stays a true original.
    """

    frame_no: int
    timestamp: float
    raw: np.ndarray
    annotated: np.ndarray
    detections: list[Detection]
    store: Any                       # TrackStore; Any avoids a circular import
    source: Any                      # SourceInfo
    frame_id: int | None = None      # storage row id, set once the frame is persisted
    events: list[dict] = field(default_factory=list)

    def emit(self, kind: str, detail: dict | None = None, track_id: int | None = None):
        """Record something worth storing. Drained by the pipeline into storage."""
        self.events.append({"kind": kind, "detail": detail or {},
                            "track_id": track_id, "ts": self.timestamp})

    def vehicles(self):
        """Detections that are vehicles - the ANPR/congestion subset."""
        return [d for d in self.detections if d.group == "vehicle"]

    def view(self) -> AnalysisView:
        """The read-only projection handed to compute phases."""
        if self.raw is not None:
            h, w = self.raw.shape[:2]
        else:
            h = int(getattr(self.source, "height", 0) or 0)
            w = int(getattr(self.source, "width", 0) or 0)
        return AnalysisView(frame_no=self.frame_no, timestamp=self.timestamp,
                            boxes=tuple(TrackedBox.of(d) for d in self.detections),
                            width=int(w), height=int(h),
                            raw=self.raw, source=self.source)

    def by_track(self) -> dict:
        """track_id -> Detection, so an apply phase can write findings back."""
        return {d.track_id: d for d in self.detections if d.track_id is not None}
