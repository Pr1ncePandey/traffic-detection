"""Face recognition as an enricher: names tracked people from the people database.

Switched on per camera (perception.attributes.face.enabled). People to look
for live in the database (src/faces/people.py; enrol with tools/people.py or
POST /people). A confirmed match puts the name on the person's object
(attributes face_name / face_name_conf), on the box caption, on the live
dashboard, and emits ONE `face_match` event per sighting - which the incident
layer turns into a webhook.

HOW A FRAME IS HANDLED

Faces ride on the main pipeline's ByteTrack PERSON tracks rather than on a
tracker of their own, so a name lands on the same object id as the person's
crop and clothing attributes. The cost model drives everything else - YuNet
is tens of ms per frame, AdaFace ~1 s per face on a laptop CPU:

  1. only if some tracked person still needs a read: run YuNet (every
     `detect_every` analysed frames), give each face to the person box it sits
     in the upper part of (smallest box wins, one face per person)
  2. AdaFace only on faces worth it: eyes >= min_eye_px apart, sharp enough
     (motion blur scored 0.07-0.40 against the right person in testing), at
     most `max_reads` per track, at least `read_gap_s` apart
  3. vote (src/faces/voting.py): a name needs `min_votes` reads that clear
     `threshold` and beat the next person by `margin`
  4. one person, one place: if the name is already on another person visible
     in the SAME frame, the better score keeps it
  5. a named track is re-checked every `recheck_s`. If the tracker has swapped
     two people (they crossed), reads start naming someone else confidently
     and the track is re-decided - the swap test in docs/face-recognition.md
     is exactly this case. Because a crossing is WHEN ByteTrack swaps ids,
     two person boxes overlapping by `crossing_overlap` mark both tracks, and
     the moment they separate each gets `min_votes` quick re-checks
     (read_gap_s apart) instead of waiting out recheck_s. In the switch test
     the wrong names stood for 1.8 s with the 3 s recheck alone.

A face inside the face region of TWO person boxes (one person in front of the
other) is ambiguous and is skipped: giving it to the wrong box would put one
person's read on the other's track.

Once no visible person needs a read, no face detection runs at all: a room
with one recognised person costs one AdaFace run every `recheck_s`.

LIVE vs FILE. On a live source AdaFace runs on a background thread
(`embed_async: auto`), so a slow face check never stalls the frame loop - a
full queue skips the read instead. On a file it runs inline, which is slower
but deterministic and never loses the last reads when the clip ends.
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from collections import Counter

import numpy as np

from ...detectors.classes import PERSON
from ...runtime.plugin import Findings
from ..registry import block, register

try:
    import cv2
except ImportError:          # pragma: no cover - opencv is a core dependency
    cv2 = None

NAME_KEY = "face_name"
CONF_KEY = "face_name_conf"


def _overlap(a, b) -> float:
    """Intersection over the SMALLER box: 1.0 when one box is inside the other."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / max(1, smaller)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "person"


class _Track:
    """What face recognition knows about one tracked person."""

    __slots__ = ("reads", "queued", "last_read_ts", "name", "score", "votes",
                 "since_confirm", "reason", "best", "crossing", "urgent")

    def __init__(self):
        self.crossing = False        # box overlaps another person box right now
        self.urgent = 0              # quick re-checks owed after a crossing
        self.reads = []              # voting.Read, since the last (re)decision
        self.queued = 0              # AdaFace runs requested for this track
        self.last_read_ts = -1e18
        self.name = None             # confirmed name, or None
        self.score = 0.0
        self.votes = 0
        self.since_confirm = []      # recheck reads after confirmation
        self.reason = "not checked yet"
        self.best = {}               # name -> (similarity, snapshot) best view


class _EmbedWorker:
    """AdaFace on its own thread, for live sources. Bounded in, unbounded out.

    Only vectors cross back; matching against the gallery happens in compute,
    so a gallery reload never races a worker mid-match.
    """

    def __init__(self, embedder, max_pending: int, batch: int):
        self.embedder = embedder
        self.batch = max(1, int(batch))
        self.inbox = queue.Queue(maxsize=max(1, int(max_pending)))
        self.outbox = queue.Queue()
        self.errors = 0
        self.last_error = ""
        self.seconds = 0.0
        self.done = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="face-embed", daemon=True)
        self._thread.start()

    def submit(self, item) -> bool:
        try:
            self.inbox.put_nowait(item)
            return True
        except queue.Full:
            return False

    def results(self) -> list:
        out = []
        while True:
            try:
                out.append(self.outbox.get_nowait())
            except queue.Empty:
                return out

    def _run(self):
        while not self._stop.is_set():
            try:
                items = [self.inbox.get(timeout=0.5)]
            except queue.Empty:
                continue
            while len(items) < self.batch:
                try:
                    items.append(self.inbox.get_nowait())
                except queue.Empty:
                    break
            started = time.perf_counter()
            try:
                vectors = self.embedder.embed([it[2] for it in items])
            except Exception as e:           # a bad batch must not kill the thread
                self.errors += 1
                self.last_error = repr(e)[:200]
                continue
            self.seconds += time.perf_counter() - started
            self.done += len(items)
            for item, vec in zip(items, vectors):
                self.outbox.put((item, vec))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)


class FaceEnricher:
    name = "face"
    # compute touches only this enricher's own state (and its worker's
    # queues), so it may run on the perception pool.
    concurrent = True

    def __init__(self, cfg: dict):
        conf = block(cfg, self.name)
        self.cfg_path = str(((cfg or {}).get("storage") or {}).get("path", "outputs/traffic.db"))
        self.models_dir = str(conf.get("models_dir", "models/faces"))
        self.only = [str(n) for n in (conf.get("people") or [])]
        self.detect_every = max(1, int(conf.get("detect_every", 2)))
        self.detect_width = int(conf.get("detect_width", 1280))
        self.min_det_score = float(conf.get("min_det_score", 0.8))
        self.min_eye_px = float(conf.get("min_eye_px", 28))
        self.min_sharpness = float(conf.get("min_sharpness", 40))
        self.face_top = float(conf.get("face_top", 0.6))
        self.threshold = float(conf.get("threshold", 0.45))
        self.margin = float(conf.get("margin", 0.05))
        self.min_votes = max(1, int(conf.get("min_votes", 2)))
        self.max_reads = max(self.min_votes, int(conf.get("max_reads", 6)))
        self.read_gap_s = max(0.0, float(conf.get("read_gap_s", 0.3)))
        self.recheck_s = max(0.0, float(conf.get("recheck_s", 3.0)))
        self.lost_below = float(conf.get("lost_below", 0.25))
        self.crossing_overlap = float(conf.get("crossing_overlap", 0.3))
        self.threads = int(conf.get("threads", 4))
        self.batch = max(1, int(conf.get("batch", 8)))
        self.max_pending = max(1, int(conf.get("max_pending", 16)))
        self.embed_async = str(conf.get("embed_async", "auto")).lower()
        self.reload_s = max(1.0, float(conf.get("reload_s", 10.0)))
        self.snapshot_dir = conf.get("snapshot_dir") or ""

        self.detector = None
        self.embedder = None
        self.people_db = None
        self.gallery = None
        self.worker = None
        self._sig = None
        self._next_reload = 0.0
        self._warned: set = set()
        self._tracks: dict = {}
        self._frames_wanting = 0
        self._last_results = {}      # track id -> extra dict set this frame
        self.stats = Counter()
        self.matches = Counter()
        self.embed_seconds = 0.0
        self.embedded = 0

    # --- lifecycle -------------------------------------------------------
    def setup(self, source, cfg: dict):
        from ...faces.detector import FaceDetector
        from ...faces.embedder import FaceEmbedder
        from ...faces.people import PeopleDB
        self.detector = FaceDetector(self.models_dir, self.detect_width, self.min_det_score)
        self.embedder = FaceEmbedder(self.models_dir, self.threads, self.batch)
        self.people_db = PeopleDB(self.cfg_path)
        live = bool(getattr(source, "is_live", False))
        use_async = self.embed_async == "on" or (self.embed_async == "auto" and live)
        if use_async:
            self.worker = _EmbedWorker(self.embedder, self.max_pending, self.batch)
        self.reload(force=True)
        who = ", ".join(self.gallery.names) if len(self.gallery) else "NOBODY"
        print(f"[face] AdaFace {'background thread' if self.worker else 'inline'} | "
              f"looking for {len(self.gallery)} people: {who}"
              + ("" if len(self.gallery) else
                 " - add some: python tools/people.py add NAME PHOTO"))

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def reload(self, force: bool = False):
        """Re-read the people database if it changed. Never fatal on a running camera."""
        now = time.monotonic()
        if not force and now < self._next_reload:
            return
        self._next_reload = now + self.reload_s
        try:
            sig = self.people_db.signature()
            if not force and sig == self._sig:
                return
            gallery, warnings = self.people_db.load_gallery(self.detector, self.embedder, self.only)
        except Exception as e:
            self.stats["reload_errors"] += 1
            if force:
                raise
            print(f"[face] could not reload the people database ({e}); keeping the old list")
            return
        for w in warnings:
            if w not in self._warned:
                self._warned.add(w)
                print(f"[face] {w}")
        if self.gallery is not None and gallery.names != self.gallery.names:
            print(f"[face] people database changed: now looking for "
                  f"{', '.join(gallery.names) or 'nobody'}")
        self._sig = sig
        self.gallery = gallery

    # --- per frame -------------------------------------------------------
    def compute(self, view) -> Findings:
        found = Findings(analyzer=self.name)
        if view.raw is None or self.detector is None:
            return found
        self.reload()
        ts = float(view.timestamp)
        people = {b.track_id: b for b in view.of_group(PERSON) if b.track_id is not None}
        for tid in people:
            self._tracks.setdefault(tid, _Track())
        self._mark_crossings(people)

        if self.worker is not None:
            for (tid, read_ts, _crop, snapshot), vec in self.worker.results():
                self._on_vector(tid, read_ts, vec, snapshot, people, found)

        wanting = {tid for tid in people if self._wants_read(self._tracks[tid], ts)}
        if wanting and len(self.gallery):
            self._frames_wanting += 1
            if (self._frames_wanting - 1) % self.detect_every == 0:
                pending = self._collect_faces(view, people, wanting, ts)
                if pending and self.worker is None:
                    self._embed_inline(pending, people, found)

        # Keep the name on the box every frame, not just the frame it was decided.
        for tid in people:
            track = self._tracks[tid]
            if track.name is not None:
                found.set(tid, extra={NAME_KEY: track.name, CONF_KEY: round(track.score, 3)})
        return found

    def _mark_crossings(self, people: dict):
        """Flag person boxes that overlap; owe named ones quick re-checks once apart."""
        items = list(people.items())
        overlapping = set()
        for i, (ta, a) in enumerate(items):
            for tb, b in items[i + 1:]:
                if _overlap(a.bbox, b.bbox) >= self.crossing_overlap:
                    overlapping.update((ta, tb))
        for tid in people:
            track = self._tracks[tid]
            if tid in overlapping:
                if track.name is not None and not track.crossing:
                    self.stats["crossings"] += 1
                track.crossing = True
            elif track.crossing:
                track.crossing = False
                if track.name is not None:
                    track.urgent = self.min_votes

    def _wants_read(self, track: _Track, ts: float) -> bool:
        if track.name is not None:
            if track.urgent > 0 and not track.crossing:
                return ts - track.last_read_ts >= self.read_gap_s
            return self.recheck_s > 0 and ts - track.last_read_ts >= self.recheck_s
        return track.queued < self.max_reads and ts - track.last_read_ts >= self.read_gap_s

    def _collect_faces(self, view, people: dict, wanting: set, ts: float) -> list:
        from ...faces.align import align_face, face_snapshot, measure_quality
        faces = [f for f in self.detector.detect(view.raw) if f.score >= self.min_det_score]
        self.stats["faces_detected"] += len(faces)
        chosen: dict = {}            # track id -> (eye_px, face, quality)
        for face in faces:
            quality = measure_quality(view.raw, face)
            owners = self._owners(face, people)
            if not owners:
                self.stats["skipped_no_person_box"] += 1
                continue
            if len(owners) > 1:
                self.stats["skipped_ambiguous"] += 1
                continue
            tid = owners[0]
            if tid not in chosen or quality.eye_px > chosen[tid][0]:
                chosen[tid] = (quality.eye_px, face, quality)
        pending = []
        for tid, (_eye, face, quality) in chosen.items():
            if tid not in wanting:
                continue
            if quality.eye_px < self.min_eye_px:
                self.stats["skipped_small"] += 1
                continue
            if quality.sharp < self.min_sharpness:
                self.stats["skipped_blurred"] += 1
                continue
            crop = align_face(view.raw, face.landmarks)
            if crop is None:
                self.stats["skipped_unalignable"] += 1
                continue
            item = (tid, ts, crop, face_snapshot(view.raw, face))
            if self.worker is not None and not self.worker.submit(item):
                self.stats["skipped_queue_full"] += 1
                continue
            track = self._tracks[tid]
            track.queued += 1
            track.last_read_ts = ts
            if track.name is not None and track.urgent > 0:
                track.urgent -= 1
            pending.append(item)
        return pending

    def _owners(self, face, people: dict) -> list:
        """Person boxes whose upper part contains this face's centre."""
        x1, y1, x2, y2 = face.bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return [tid for tid, box in people.items()
                if box.bbox[0] <= cx <= box.bbox[2]
                and box.bbox[1] <= cy <= box.bbox[1] + self.face_top * (box.bbox[3] - box.bbox[1])]

    def _embed_inline(self, pending: list, people: dict, found: Findings):
        started = time.perf_counter()
        vectors = self.embedder.embed([crop for _tid, _ts, crop, _snap in pending])
        self.embed_seconds += time.perf_counter() - started
        self.embedded += len(pending)
        for (tid, read_ts, _crop, snapshot), vec in zip(pending, vectors):
            self._on_vector(tid, read_ts, vec, snapshot, people, found)

    # --- decisions -------------------------------------------------------
    def _on_vector(self, tid, read_ts, vec, snapshot, people: dict, found: Findings):
        from ...faces.voting import Read, decide, votes_for
        track = self._tracks.get(tid)
        if track is None or self.gallery is None:
            return                   # retired while its read was queued
        best, sim, runner = self.gallery.match(np.asarray(vec)[None])
        name = self.gallery.names[int(best[0])] if int(best[0]) >= 0 else None
        read = Read(float(read_ts), name, float(sim[0]), float(runner[0]))
        self.stats["reads"] += 1
        if name is not None and (name not in track.best or read.similarity > track.best[name][0]):
            track.best[name] = (read.similarity, snapshot)

        if track.name is None:
            track.reads.append(read)
            decision = decide(track.reads, self.threshold, self.margin, self.min_votes)
            track.reason = decision.reason
            if decision.name is not None:
                self._confirm(tid, track, decision, people, found)
            return

        # A confirmed track: is it still the same person?
        track.since_confirm.append(read)
        recent = track.since_confirm[-self.min_votes:]
        if (len(recent) == self.min_votes
                and all(votes_for(r, self.threshold, self.margin) for r in recent)
                and len({r.name for r in recent}) == 1 and recent[0].name != track.name):
            self._unname(tid, track, found, f"now reads as {recent[0].name}: tracker swapped people")
            track.reads = list(recent)
            decision = decide(track.reads, self.threshold, self.margin, self.min_votes)
            if decision.name is not None:
                self._confirm(tid, track, decision, people, found, after_swap=True)
            return
        lost = track.since_confirm[-3:]
        if len(lost) == 3 and all(r.similarity < self.lost_below for r in lost):
            self._unname(tid, track, found, "last 3 checks look like nobody enrolled")

    def _confirm(self, tid, track: _Track, decision, people: dict, found: Findings,
                 after_swap: bool = False):
        # One person, one place: another person box in THIS frame already has the name.
        for other_tid in people:
            other = self._tracks.get(other_tid)
            if other_tid == tid or other is None or other.name != decision.name:
                continue
            if decision.score <= other.score:
                track.reason = (f"also looked like {decision.name}, but track {other_tid} "
                                f"in the same frame matched better")
                track.reads = []
                track.queued = min(track.queued, self.max_reads - 1)
                self.stats["conflicts"] += 1
                return
            self._unname(other_tid, other, found,
                         f"track {tid} in the same frame matched {decision.name} better")
            self.stats["conflicts"] += 1
        track.name, track.score, track.votes = decision.name, decision.score, decision.votes
        track.reason = decision.reason
        track.since_confirm = []
        self.matches[decision.name] += 1
        detail = {"person": decision.name, "score": decision.score,
                  "votes": decision.votes, "reads": decision.reads,
                  "reason": decision.reason, "threshold": self.threshold,
                  "after_swap": after_swap,
                  "snapshot_path": self._save_snapshot(tid, decision.name, track)}
        found.set(tid, extra={NAME_KEY: decision.name, CONF_KEY: round(decision.score, 3)})
        found.event("face_match", detail, track_id=tid)

    def _unname(self, tid, track: _Track, found: Findings, why: str):
        previous = track.name
        track.name, track.score, track.votes = None, 0.0, 0
        track.reason = why
        # Re-checks that already named someone ELSE are evidence for the new
        # decision: after a swap the other track's first re-check is usually
        # of the right person, and throwing it away costs one more ~1 s read.
        track.reads = [r for r in track.since_confirm if r.name != previous]
        track.since_confirm = []
        track.queued = 0
        found.set(tid, extra={NAME_KEY: "", CONF_KEY: 0.0})
        found.event("face_unmatched", {"person": previous, "reason": why}, track_id=tid)
        self.stats["unnamed"] += 1

    def _save_snapshot(self, tid, name, track: _Track):
        """The clearest face this track showed of that person, for the webhook's image_url."""
        if not self.snapshot_dir or cv2 is None or name not in track.best:
            return None
        snapshot = track.best[name][1]
        if snapshot is None or getattr(snapshot, "size", 0) == 0:
            return None
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            path = os.path.join(self.snapshot_dir,
                                f"{_slug(name)}_t{tid}_{int(time.time() * 1000)}.jpg")
            return path if cv2.imwrite(path, snapshot) else None
        except Exception as e:
            self.stats["snapshot_errors"] += 1
            print(f"[face] could not save snapshot: {e}")
            return None

    # --- serial phases ---------------------------------------------------
    def apply(self, ctx, findings: Findings):
        """Write the name onto the object's durable attribute record."""
        store = ctx.store
        for tid, fields in findings.per_track.items():
            obj = store.tracks.get(tid)
            if obj is None:
                continue
            attrs = obj.setdefault("attrs", {})
            for key, value in (fields.get("extra") or {}).items():
                attrs[key] = value

    def forget(self, tid):
        track = self._tracks.pop(tid, None)
        if track is None:
            return
        if track.name is None and track.queued:
            self.stats["tracks_unknown"] += 1
        elif track.name is not None:
            self.stats["tracks_named"] += 1

    def summary(self) -> dict:
        seconds = self.embed_seconds + (self.worker.seconds if self.worker else 0.0)
        runs = self.embedded + (self.worker.done if self.worker else 0)
        out = {"people_enrolled": len(self.gallery) if self.gallery is not None else 0,
               "matches": dict(self.matches),
               "adaface_runs": runs,
               "adaface_ms_per_face": round(1000.0 * seconds / runs, 1) if runs else None,
               **{k: v for k, v in sorted(self.stats.items())}}
        if self.worker is not None and self.worker.errors:
            out["worker_errors"] = self.worker.errors
            out["worker_last_error"] = self.worker.last_error
        return out


register("face", FaceEnricher)
