"""One supervised thread per camera.

THE FAILURE THIS IS BUILT AGAINST

A camera that is silently dead. The one-shot pipeline could let an exception
propagate and end the process, because ending was what a finished run did
anyway. In a service, an unhandled exception in one camera thread must not take
the server down with it - and equally must not leave that camera looking fine
while its feed has stopped. So every worker is wrapped: catch, log, mark
unhealthy, surface it, restart on a backoff.

`state` is the whole point of this class:

    idle      -> configured, not started
    starting  -> thread up, pipeline opening the source
    running   -> frames flowing (on_ready fired)
    retrying  -> crashed, waiting out a backoff before another attempt
    failed    -> gave up after max_restarts
    stopping  -> stop requested, finishing the frame in flight
    stopped   -> clean exit

ON THE GIL, HONESTLY

N camera threads only parallelise where the heavy work releases the GIL. OpenCV
and onnxruntime do release it for their compute kernels, so decode and
inference genuinely overlap. Everything Python-level - the analysis stages, the
drawing pass, event handling - serialises. Per-camera throughput therefore
degrades non-linearly with camera count, and the honest thing to do is MEASURE
it for a given host rather than promise a number: `/cameras` reports each
worker's achieved fps for exactly that purpose. If it does not hold up, the
fallback is worker processes plus one writer process fed over a queue, which
keeps the single-writer invariant at the cost of IPC.
"""

import copy
import threading
import time

from ..config import load_for_camera
from ..pipeline import run_pipeline

RESTART_BACKOFF_START_S = 2.0
RESTART_BACKOFF_MAX_S = 60.0
MAX_RESTARTS = 0            # 0 = keep trying forever, like the RTSP reconnect

IDLE, STARTING, RUNNING, RETRYING, FAILED, STOPPING, STOPPED = (
    "idle", "starting", "running", "retrying", "failed", "stopping", "stopped")


def _plate_ocr_status() -> dict:
    """Whether plate reading actually loaded. Never raises.

    Imported inside the function because the attributes package pulls in
    onnxruntime and friends, and a status call must not be the thing that
    triggers that import.
    """
    try:
        from ..attributes import ocr_engines
        return ocr_engines.status()
    except Exception:
        return {"backend": None, "ok": False, "error": None}


class CameraWorker:
    """Runs one camera's pipeline on a thread, and keeps it running."""

    def __init__(self, camera_id: str, cfg: dict, hub=None, policy=None,
                 config_path: str = "config.yaml",
                 max_restarts: int = MAX_RESTARTS, storage=None, video=None):
        self.camera_id = camera_id
        self.cfg = cfg
        self.config_path = config_path
        self.hub = hub
        self.policy = policy
        # The JPEG fan-out. Separate from `hub` because the two channels are
        # separate on purpose: metadata stays data, pixels stay pixels.
        self.video = video
        # The server's single shared SqliteStore. See run_pipeline's docstring
        # for why this is passed in rather than constructed per camera.
        self.storage = storage
        self.max_restarts = int(max_restarts)
        self.state = IDLE
        self.error: str | None = None
        self.restarts = 0
        self.started_at: float | None = None
        self.run_id: int | None = None
        self.info: dict = {}
        self.frames = 0
        self.last_frame_at: float | None = None
        self._fps = 0.0
        self._fps_window_start = 0.0
        self._fps_window_frames = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self.error = None
        self.restarts = 0
        self.state = STARTING
        self._thread = threading.Thread(target=self._supervise,
                                        name=f"camera-{self.camera_id}",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 30.0) -> bool:
        """Ask the pipeline to finish the current frame and exit.

        The generous timeout is not laziness: shutdown finalises every track
        still in flight, which writes their object rows and settles their plate
        votes. Killing that early loses the last vehicles of the run.
        """
        if self._thread is None:
            return False
        self.state = STOPPING
        self._stop.set()
        self._thread.join(timeout=timeout)
        alive = self._thread.is_alive()
        if not alive:
            self.state = STOPPED
            # Both channels, or the dashboard keeps showing a stopped camera's
            # final frame with its final boxes on top - indistinguishable from
            # a running feed that has frozen, which is the failure this whole
            # class exists to make visible.
            if self.hub is not None:
                self.hub.forget(self.camera_id)
            if self.video is not None:
                self.video.forget(self.camera_id)
        return not alive

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- the supervised loop ----------------------------------------------
    def _supervise(self):
        delay = RESTART_BACKOFF_START_S
        while not self._stop.is_set():
            self.state = STARTING
            self.started_at = time.time()
            try:
                self._run_once()
                # A clean return means the source ended (a file, or a live
                # source that exhausted its reconnect budget). Not an error, so
                # no restart: restarting a finished file would loop forever.
                if not self._stop.is_set():
                    print(f"[camera {self.camera_id}] source ended; worker idle")
                self.state = STOPPED
                return
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                self.restarts += 1
                print(f"[camera {self.camera_id}] CRASHED "
                      f"(restart {self.restarts}): {self.error}")
                if self._stop.is_set():
                    break
                if self.max_restarts and self.restarts > self.max_restarts:
                    self.state = FAILED
                    print(f"[camera {self.camera_id}] giving up after "
                          f"{self.restarts} restarts")
                    return
                self.state = RETRYING
                if self._stop.wait(delay):
                    break
                delay = min(delay * 2, RESTART_BACKOFF_MAX_S)
                # Re-read config on restart, so fixing a broken camera file is
                # enough to recover a crashed camera without a server bounce.
                try:
                    self.cfg = copy.deepcopy(
                        load_for_camera(self.camera_id, self.config_path))
                except Exception as reload_error:
                    print(f"[camera {self.camera_id}] config reload failed, "
                          f"keeping the previous one: {reload_error}")
        self.state = STOPPED

    def _run_once(self):
        cfg = copy.deepcopy(self.cfg)
        if self.storage is not None:
            # The server owns retention: one reaper over the shared database,
            # not N reapers racing to enforce the same byte budget from N
            # different usage estimates.
            cfg.setdefault("frames", {}).setdefault("retention", {})
            cfg["frames"]["retention"] = {**cfg["frames"]["retention"],
                                          "max_age_hours": 0, "max_disk_gb": 0}
        run_pipeline(cfg, stop=self._stop, on_frame=self._on_frame,
                     on_event=self._on_event, on_ready=self._on_ready,
                     # Passed as None when there is no sink, so the frame loop
                     # does not even make the call - `main.py` pays nothing for
                     # a feature only the server uses.
                     on_video=(self._on_video if self.video is not None
                               else None),
                     storage=self.storage,
                     # N cameras redrawing a tqdm bar into one log is
                     # unreadable, and a restarted camera would start a fresh
                     # one each time. The dashboard reports fps instead.
                     progress=False)

    # --- pipeline callbacks -----------------------------------------------
    def _on_ready(self, info: dict):
        self.info = info
        self.run_id = info.get("run_id")
        self.state = RUNNING
        self.error = None
        self._fps_window_start = time.monotonic()
        self._fps_window_frames = 0

    def _on_frame(self, meta: dict):
        self.frames += 1
        self.last_frame_at = time.time()
        self._tick_fps()
        meta["viewers"] = self.hub.viewers(self.camera_id) if self.hub else 0
        if self.policy is not None:
            meta["congestion"] = self.policy.congestion_state(self.camera_id)
        if self.hub is not None:
            self.hub.publish(meta)

    def _on_video(self, frame):
        """Hand the clean frame to the sink, which decides whether to encode.

        Deliberately thin: the throttle and the is-anyone-watching check live
        in the sink, so this runs at full frame rate and costs a dict lookup
        when nobody has the dashboard open.
        """
        self.video.offer(self.camera_id, frame)

    def _tick_fps(self):
        """Achieved fps over a rolling ~2 s window.

        Rolling rather than cumulative: a cumulative average hides a camera
        that was fast for an hour and has been crawling for the last minute,
        which is precisely the degradation worth seeing.
        """
        self._fps_window_frames += 1
        now = time.monotonic()
        span = now - self._fps_window_start
        if span >= 2.0:
            self._fps = self._fps_window_frames / span
            self._fps_window_start = now
            self._fps_window_frames = 0

    def _on_event(self, event: dict, ctx: dict):
        if self.policy is not None:
            self.policy.handle(event, ctx)

    # --- reporting ---------------------------------------------------------
    def status(self) -> dict:
        meta = self.hub.latest(self.camera_id) if self.hub else None
        health = (meta or {}).get("health", {})
        stale = (None if self.last_frame_at is None
                 else round(time.time() - self.last_frame_at, 1))
        return {
            "camera": self.camera_id,
            "name": (self.cfg.get("camera", {}) or {}).get("name", self.camera_id),
            "state": self.state,
            "healthy": self.state == RUNNING and not health.get("write_alarm"),
            "error": self.error,
            "restarts": self.restarts,
            "run_id": self.run_id,
            "source": (self.cfg.get("video", {}) or {}).get("source"),
            "is_live": self.info.get("is_live"),
            "resolution": (None if not self.info else
                           f"{self.info.get('width')}x{self.info.get('height')}"),
            "frames": self.frames,
            "fps": round(self._fps, 1),
            "seconds_since_frame": stale,
            "location": (self.cfg.get("camera", {}) or {}).get("location"),
            "viewers": self.hub.viewers(self.camera_id) if self.hub else 0,
            # Plate OCR availability, so an empty plate column on the
            # dashboard explains itself instead of looking like a bug. Read
            # lazily rather than cached: the engine initialises on the first
            # frame that has a plate box, not at startup.
            "plate_ocr": _plate_ocr_status(),
            "video_viewers": (self.video.viewers(self.camera_id)
                              if self.video else 0),
            "video_age_s": self.video.age(self.camera_id) if self.video else None,
            "counts": (meta or {}).get("counts", {}),
            "crossings": (meta or {}).get("crossings", {}),
            "flagged": (meta or {}).get("flagged", {}),
            "congestion": (meta or {}).get("congestion"),
            "health": health,
        }


class WorkerPool:
    """Every configured camera, and the control plane over them."""

    def __init__(self, camera_ids, config_path: str = "config.yaml",
                 hub=None, policy=None, storage=None, video=None):
        self.config_path = config_path
        self.hub = hub
        self.policy = policy
        self.storage = storage
        self.video = video
        self.workers: dict[str, CameraWorker] = {}
        for camera_id in camera_ids:
            try:
                cfg = copy.deepcopy(load_for_camera(camera_id, config_path))
            except Exception as e:
                print(f"[camera {camera_id}] config failed to load, skipping: {e}")
                continue
            self.workers[camera_id] = CameraWorker(
                camera_id, cfg, hub=hub, policy=policy, config_path=config_path,
                storage=storage, video=video)

    def locations(self) -> dict:
        """camera id -> {name, lat, lon}, for stamping incident payloads."""
        out = {}
        for cid, w in self.workers.items():
            cam = (w.cfg.get("camera", {}) or {})
            loc = cam.get("location") or {}
            out[cid] = {"name": cam.get("name") or cid,
                        "lat": loc.get("lat"), "lon": loc.get("lon")}
        return out

    def get(self, camera_id: str) -> CameraWorker | None:
        return self.workers.get(camera_id)

    def start_all(self):
        for w in self.workers.values():
            w.start()

    def stop_all(self):
        # Signal every camera first, THEN wait. Stopping serially would make
        # shutdown take N x the per-camera finalise time for no reason.
        for w in self.workers.values():
            w.state = STOPPING
            w._stop.set()
        for w in self.workers.values():
            w.stop()

    def status(self) -> list:
        return [w.status() for w in self.workers.values()]

    def unhealthy(self) -> list:
        return [s["camera"] for s in self.status() if not s["healthy"]]
