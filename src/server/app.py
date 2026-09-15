"""The HTTP/WebSocket surface: one server, N camera workers, one DB writer.

ON AUTH, WHICH THERE IS NONE OF

By decision, the API, WebSocket, crop and VIDEO endpoints are protected by
NETWORK PLACEMENT ONLY. Two things follow, and they are not optional given
that `/crops/{id}.jpg` serves number-plate imagery, the WebSocket streams
plate strings, and `/stream/{id}.mjpg` serves LIVE ROAD FOOTAGE:

  - the listen address DEFAULTS TO 127.0.0.1, never 0.0.0.0. Binding wider is
    an explicit act (`--host`), and the server says so out loud when you do.
  - that assumption is written down here and next to the deployment
    instructions, because the difference between safe and exposed is one config
    value and no code will complain about it.

Revisit if this is ever operated by more than one person, handed to a client,
or reachable from outside a VPN. A shared token would be a small change since
every endpoint already routes through this one module.

ASSEMBLY ORDER, AND WHY

    store      one SqliteStore, one writer thread - the invariant the storage
               layer is built around
    policy     incident rules, needs the store to raise into
    hub        WebSocket fan-out (metadata), needs nothing
    video      MJPEG fan-out (pixels), needs nothing
    workers    N camera threads, need store + hub + video + policy
    dispatcher webhook delivery, needs store + policy
    retention  crop and row reaping, needs the store

The frame loop never blocks on any of the last three: both fan-outs drop for
slow viewers, the dispatcher owns its thread, and retention owns another.

TWO CHANNELS, ON PURPOSE. `hub` carries boxes as DATA and `video` carries the
picture. The dashboard stacks them - a canvas of live boxes over an <img> of
the stream - which is why the pipeline hands the video sink `raw` and not the
frame OpenCV already drew on. Boxes that are part of the JPEG cannot be
toggled, filtered or clicked, and would double up against the canvas.
"""

import asyncio
import json
import os
import threading
import time

from fastapi import (Body, FastAPI, HTTPException, Query, WebSocket,
                     WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

from ..config import load_for_camera
from ..incidents import IncidentPolicy
from ..journeys import build_path, yield_summary
from ..query.clip_onnx import SPECS
from ..storage.retention import from_config as retention_from_config
from ..storage.sqlite_store import SqliteStore
from . import queries
from .dashboard import INDEX, STATIC_DIR, missing as missing_static
from .embedder import BackgroundEmbedder
from .hub import LiveHub
from .video import VideoSink
from .webhooks import WebhookDispatcher
from .workers import WorkerPool

# Fallback embedding space for /search when config says nothing. The CLI
# defaults to b16 for speed; the service defaults to l14 because it is the one
# that measured p@5 0.80, and a text encode is ~40 ms warm either way.
#
# ONE place decides this now. It used to be a constant here while
# tools/embed_crops.py defaulted to b16, so the obvious `embed_crops.py` with
# no flags populated a space nothing queried and search stayed empty with no
# error. server.search.model is now the single source, and the background
# embedder reads the same value.
SEARCH_MODEL = "clip-vit-l14"

# Boundary for the multipart video stream. Arbitrary but must match the
# Content-Type; browsers key their frame splitting on it.
BOUNDARY = "frame"


class Server:
    """Owns every long-lived object, so shutdown has one place to happen."""

    def __init__(self, cameras: list, cfg: dict, config_path: str = "config.yaml",
                 autostart: bool = True):
        self.cfg = cfg
        self.config_path = config_path
        self.autostart = autostart
        st = cfg.get("storage", {}) or {}
        # ONE store for every camera. See run_pipeline's docstring: N stores
        # would mean N writer threads on one SQLite file, which is precisely
        # what the single-writer design exists to avoid.
        self.store = SqliteStore(st.get("path", "outputs/traffic.db"),
                                 batch_rows=st.get("batch_rows"),
                                 commit_interval=st.get("commit_interval"),
                                 shed_when_full=True)
        srv = cfg.get("server", {}) or {}
        self.hub = LiveHub(push_hz=float(srv.get("push_hz", 8.0)))
        self.video = VideoSink.from_config(cfg)
        self.policy = IncidentPolicy(cfg, self.store)
        self.workers = WorkerPool(cameras, config_path=config_path, hub=self.hub,
                                  policy=self.policy, storage=self.store,
                                  video=self.video)
        # One encoder per embedding space, built on first use and then reused.
        # ClipOnnx loads its towers lazily and caches them on the instance, so
        # this is what makes the second search 40 ms instead of 600.
        self._encoders: dict = {}
        self._encoder_lock = threading.Lock()
        self.search_model = (srv.get("search", {}) or {}).get(
            "model", SEARCH_MODEL)
        self.embedder = BackgroundEmbedder.from_config(cfg, self.store)
        # Locations are read from the loaded camera configs rather than from
        # the run rows, because a payload is built while the run is live.
        self.policy.cameras = self.workers.locations()
        self.dispatcher = WebhookDispatcher(self.store, self.policy)
        self.retention = retention_from_config(cfg, self.store)
        self.started_at = time.time()

    def start(self):
        self.dispatcher.start()
        self.retention.start()
        self.embedder.start()
        if self.autostart:
            self.workers.start_all()

    def stop(self):
        # Cameras first: they are the only thing still producing work, and
        # stopping them lets the dispatcher and retention drain what exists.
        self.workers.stop_all()
        self.dispatcher.stop()
        self.retention.stop()
        self.embedder.stop()
        self.store.flush()
        # end_run=False: each worker already closed its own run row by id. The
        # store's own run_id points at whichever camera started last, and
        # stamping that one twice would be wrong.
        self.store.close(end_run=False)

    def search(self, q: str, camera=None, top: int = 24,
               model: str = SEARCH_MODEL, min_px: int = 0,
               min_conf: float = 0.0) -> dict:
        """Run one semantic search. SYNCHRONOUS - call it in a threadpool.

        Opens its own connection because `connect_ro()` builds one with
        sqlite3's default `check_same_thread=True`, so a connection made on
        the event loop cannot be used here.

        The per-model lock serialises searches in one space. That is
        deliberate: without it, two first-queries arriving together would each
        load the text tower, briefly doubling a 472 MB allocation for no gain.
        Queries are interactive and ~40 ms warm, so serialising them costs
        nothing anybody can perceive.
        """
        with self._encoder_lock:
            entry = self._encoders.get(model)
            if entry is None:
                from ..query.clip_onnx import ClipOnnx
                entry = self._encoders[model] = (ClipOnnx(model),
                                                 threading.Lock())
        encoder, lock = entry
        conn = self.store.connect_ro()
        try:
            with lock:
                return queries.run_search(conn, encoder, q, camera=camera,
                                          top=top, min_px=min_px,
                                          min_conf=min_conf)
        finally:
            conn.close()

    def health(self) -> dict:
        return {"uptime_s": round(time.time() - self.started_at, 1),
                "cameras": len(self.workers.workers),
                "unhealthy": self.workers.unhealthy(),
                "storage": self.store.health(),
                "hub": self.hub.stats(),
                "video": self.video.stats(),
                "incidents": self.policy.stats(),
                "webhooks": self.dispatcher.stats(),
                "retention": self.retention.stats(),
                "search": {"default_model": self.search_model,
                           "models": sorted(SPECS),
                           "loaded": sorted(self._encoders),
                           "index": self.index_status(),
                           "embedder": self.embedder.stats()}}

    def index_status(self) -> dict:
        conn = self.store.connect_ro()
        try:
            return queries.index_status(conn, self.search_model)
        finally:
            conn.close()


def create_app(cameras: list, config_path: str = "config.yaml",
               autostart: bool = True) -> FastAPI:
    cfg = load_for_camera(None, config_path)
    server = Server(cameras, cfg, config_path=config_path, autostart=autostart)
    app = FastAPI(title="traffic-detection", version="1.0",
                  docs_url="/docs", redoc_url=None)
    app.state.server = server

    @app.on_event("startup")
    def _startup():
        gone = missing_static()
        if gone:
            # Said at startup, not on first request: a dashboard that 404s
            # because a file did not ship should be visible in the log.
            print(f"[server] WARNING: dashboard files missing: "
                  f"{', '.join(os.path.basename(g) for g in gone)}")
        # SEARCH COVERAGE, OUT LOUD. Embeddings are not built by the pipeline,
        # so a fresh database means search returns nothing and a database that
        # has merely run a while means it covers part of the corpus. Neither
        # announced itself; search just looked like it worked.
        idx = server.index_status()
        if idx["unembedded"]:
            note = ("the background embedder is working through them"
                    if server.embedder.enabled
                    else f"run: python tools/embed_crops.py --model {idx['model']}")
            print(f"[server] search index: {idx['embedded']} of {idx['crops']} "
                  f"crops embedded in {idx['model']}, "
                  f"{idx['unembedded']} without vectors - {note}")
        elif idx["crops"]:
            print(f"[server] search index: all {idx['crops']} crops embedded "
                  f"in {idx['model']}")
        server.start()

    @app.on_event("shutdown")
    def _shutdown():
        server.stop()

    # --- dashboard --------------------------------------------------------
    # Plain files, no build step. See src/server/dashboard.py for why these
    # stopped being a Python string literal.
    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard():
        if not os.path.exists(INDEX):
            raise HTTPException(500, f"dashboard not installed: {INDEX} missing")
        # no-store so a redeployed dashboard is not served from a stale cache
        # while the API underneath it has already changed shape.
        return FileResponse(INDEX, media_type="text/html",
                            headers={"Cache-Control": "no-store"})

    # --- control plane ----------------------------------------------------
    @app.get("/health")
    def health():
        return server.health()

    @app.get("/cameras")
    def list_cameras():
        return {"cameras": server.workers.status()}

    @app.get("/cameras/{camera_id}")
    def camera_detail(camera_id: str):
        worker = server.workers.get(camera_id)
        if worker is None:
            raise HTTPException(404, f"no camera {camera_id!r}")
        status = worker.status()
        status["latest_frame"] = server.hub.latest(camera_id)
        return status

    @app.post("/cameras/{camera_id}/start")
    def start_camera(camera_id: str):
        worker = server.workers.get(camera_id)
        if worker is None:
            raise HTTPException(404, f"no camera {camera_id!r}")
        started = worker.start()
        return {"camera": camera_id, "started": started,
                "state": worker.state,
                "note": None if started else "already running"}

    @app.post("/cameras/{camera_id}/stop")
    def stop_camera(camera_id: str):
        worker = server.workers.get(camera_id)
        if worker is None:
            raise HTTPException(404, f"no camera {camera_id!r}")
        # Not instant by design: the pipeline finishes the frame in flight and
        # finalises every live track, which is what writes their object rows.
        stopped = worker.stop()
        return {"camera": camera_id, "stopped": stopped, "state": worker.state}

    # --- the record -------------------------------------------------------
    @app.get("/incidents")
    def incidents(kind: str | None = None, camera: str | None = None,
                  limit: int = Query(50, ge=1, le=500)):
        where, params = [], []
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if camera:
            where.append("camera = ?")
            params.append(camera)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        conn = server.store.connect_ro()
        try:
            rows = conn.execute(
                "SELECT i.*,"
                " (SELECT COUNT(*) FROM deliveries d WHERE d.incident_id=i.id"
                "   AND d.status='sent') sent,"
                " (SELECT COUNT(*) FROM deliveries d WHERE d.incident_id=i.id"
                "   AND d.status='pending') pending,"
                " (SELECT COUNT(*) FROM deliveries d WHERE d.incident_id=i.id"
                "   AND d.status='dead') dead"
                " FROM incidents i" + clause +
                " ORDER BY i.created_at DESC LIMIT ?",
                (*params, limit)).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            row = dict(r)
            row["payload"] = _loads(row.pop("payload_json", None))
            out.append(row)
        return {"incidents": out}

    @app.get("/deliveries")
    def deliveries(status: str | None = None,
                   limit: int = Query(50, ge=1, le=500)):
        """Delivery state per subscriber. A dead-lettered endpoint should be
        visible here without anyone reading logs."""
        clause = " WHERE status = ?" if status else ""
        params = (status,) if status else ()
        conn = server.store.connect_ro()
        try:
            rows = conn.execute(
                "SELECT * FROM deliveries" + clause +
                " ORDER BY COALESCE(sent_at, next_attempt_at) DESC LIMIT ?",
                (*params, limit)).fetchall()
            summary = {r[0]: r[1] for r in conn.execute(
                "SELECT status, COUNT(*) FROM deliveries GROUP BY status")}
        finally:
            conn.close()
        return {"summary": summary, "deliveries": [dict(r) for r in rows]}

    @app.get("/events")
    def events(kind: str | None = None, limit: int = Query(100, ge=1, le=1000)):
        clause = " WHERE kind = ?" if kind else ""
        params = (kind,) if kind else ()
        conn = server.store.connect_ro()
        try:
            rows = conn.execute(
                "SELECT e.*, r.camera FROM events e"
                " LEFT JOIN runs r ON r.id = e.run_id" + clause +
                " ORDER BY e.id DESC LIMIT ?", (*params, limit)).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            row = dict(r)
            row["detail"] = _loads(row.pop("detail_json", None))
            out.append(row)
        return {"events": out}

    @app.get("/identities/{identity_id}")
    def identity(identity_id: int):
        conn = server.store.connect_ro()
        try:
            # Any axis, not just plates: `plate` is aliased only where it
            # means something, so a person_reid row returns key without
            # pretending to have a registration.
            veh = conn.execute(
                "SELECT id, kind, key,"
                "       CASE WHEN kind='plate' THEN key END AS plate,"
                "       first_seen_at, last_seen_at"
                "  FROM identities WHERE id=?", (identity_id,)).fetchone()
            if veh is None:
                raise HTTPException(404, f"no identity {identity_id}")
            rows = conn.execute(
                "SELECT o.*, r.camera FROM objects o"
                " JOIN runs r ON r.id = o.run_id"
                " WHERE o.identity_id=? ORDER BY o.id", (identity_id,)).fetchall()
        finally:
            conn.close()
        return {"identity": dict(veh), "sightings": [dict(r) for r in rows]}

    @app.get("/identities/{identity_id}/path")
    def identity_path(identity_id: int):
        """Cross-camera journey. See src/journeys.py and docs/multi-camera-paths.md.

        Reports observed hops only; every unobserved stretch is a gap, and
        sightings with no comparable clock are returned in `unorderable`
        rather than being silently ordered.
        """
        j = server.cfg.get("journey", {}) or {}
        conn = server.store.connect_ro()
        try:
            path = build_path(conn, identity_id,
                              max_gap_s=float(j.get("max_gap_s", 1800)),
                              max_speed_kmh=float(j.get("max_speed_kmh", 150)))
        finally:
            conn.close()
        # build_path is plate-scoped: the implied-speed check is calibrated in
        # km/h for vehicles, so a person_reid identity is a 404 here rather
        # than a route validated against the wrong physics.
        if path is None:
            raise HTTPException(404, f"no vehicle identity {identity_id}")
        return path.as_dict()

    @app.get("/yield")
    def cross_camera_yield():
        """The go/no-go measurement: how many vehicles crossed cameras."""
        conn = server.store.connect_ro()
        try:
            return yield_summary(conn)
        finally:
            conn.close()

    @app.get("/crops/{object_id}.jpg")
    def crop(object_id: int):
        """Serve one object's crop.

        Load-bearing beyond the dashboard: `objects.crop_path` is a local path
        under outputs/, which a webhook consumer cannot read. This endpoint is
        what makes `image_url` in a payload possible at all.
        """
        conn = server.store.connect_ro()
        try:
            row = conn.execute("SELECT crop_path FROM objects WHERE id=?",
                               (object_id,)).fetchone()
        finally:
            conn.close()
        path = row["crop_path"] if row else None
        if not path:
            raise HTTPException(404, f"no crop for object {object_id}")
        if not os.path.exists(path):
            # Distinguished from "never had one": a retention pass deleting a
            # crop is a different problem from an object that was never
            # photographed, and conflating them hides a reaper misconfigured
            # to unpin incident crops.
            raise HTTPException(410, f"crop for object {object_id} has been reaped")
        return FileResponse(path, media_type="image/jpeg")


    # --- people (face recognition) ----------------------------------------
    # The database of who to look for: a name plus reference photo paths.
    # Cameras with `face` enabled re-read it every reload_s seconds, so a
    # person added here is searched for without restarting anything. Photo
    # paths are paths ON THIS MACHINE; the server reads them, never serves them.
    def _people():
        from ..faces.people import PeopleDB
        return PeopleDB(server.store.path)

    @app.get("/people")
    def list_people():
        return {"people": _people().list()}

    @app.post("/people")
    def add_person(body: dict = Body(..., examples=[
            {"name": "Prince", "photos": ["people/me/selfie.jpg"]}])):
        """Add a person, or more photos to an existing person."""
        from ..faces.errors import FaceError
        photos = body.get("photos") or ([body["photo"]] if body.get("photo") else [])
        if isinstance(photos, str):
            photos = [photos]
        try:
            return _people().add(body.get("name"), photos)
        except FaceError as e:
            raise HTTPException(400, str(e))

    @app.post("/people/{name}/enabled")
    def set_person_enabled(name: str, on: bool = True):
        from ..faces.errors import FaceError
        try:
            return _people().set_enabled(name, on)
        except FaceError as e:
            raise HTTPException(404, str(e))

    @app.delete("/people/{name}")
    def remove_person(name: str):
        from ..faces.errors import FaceError
        try:
            removed = _people().remove(name)
        except FaceError as e:
            raise HTTPException(400, str(e))
        if not removed:
            raise HTTPException(404, f"no person named {name!r}")
        return {"removed": name}

    @app.get("/faces/{object_id}.jpg")
    def face_snapshot(object_id: int):
        """The face snapshot of an object's latest face match: a webhook's image_url."""
        conn = server.store.connect_ro()
        try:
            row = conn.execute("SELECT detail_json FROM events WHERE kind='face_match'"
                               " AND object_id=? ORDER BY id DESC LIMIT 1",
                               (object_id,)).fetchone()
        finally:
            conn.close()
        detail = _loads(row["detail_json"]) if row else None
        path = detail.get("snapshot_path") if isinstance(detail, dict) else None
        if not path:
            raise HTTPException(404, f"no face snapshot for object {object_id}")
        if not os.path.exists(path):
            raise HTTPException(410, f"face snapshot for object {object_id} was deleted")
        return FileResponse(path, media_type="image/jpeg")

    # --- video ------------------------------------------------------------
    def _video_or_404(camera_id: str):
        """Shared guard. Distinguishes the three ways this can have no video."""
        if server.workers.get(camera_id) is None:
            raise HTTPException(404, f"no camera {camera_id!r}")
        if not server.video.enabled:
            raise HTTPException(503, "video is disabled; set "
                                     "server.video.enabled: true to serve it")

    @app.get("/stream/{camera_id}.mjpg", include_in_schema=False)
    async def stream(camera_id: str):
        """Live MJPEG. The picture the dashboard draws its boxes over.

        SERVES REAL FOOTAGE. See this module's docstring: there is no
        application auth, so whoever can reach this port can watch the road.

        A slow client drops frames rather than applying backpressure - its
        queue is one slot and newest-wins, so a stalled tab cannot slow the
        camera thread. Same rule as the metadata socket below.
        """
        _video_or_404(camera_id)
        sub = server.video.subscribe(camera_id, asyncio.get_running_loop())

        async def frames():
            try:
                while True:
                    jpeg = await sub.queue.get()
                    sub.sent += 1
                    yield (b"--" + BOUNDARY.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode()
                           + b"\r\n\r\n" + jpeg + b"\r\n")
            finally:
                # Runs on client disconnect too: the generator is closed, which
                # raises GeneratorExit here. Without this the subscriber set
                # would grow by one per reload and every one would be offered
                # frames forever.
                server.video.unsubscribe(sub)

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate",
                     "Pragma": "no-cache",
                     # Tells an nginx in front of this not to buffer, which
                     # would otherwise hold frames back until its buffer filled.
                     "X-Accel-Buffering": "no"})

    @app.get("/snapshot/{camera_id}.jpg", include_in_schema=False)
    def snapshot(camera_id: str):
        """The latest frame, once. For thumbnails and for a still fallback.

        Kept fresh at `server.video.snapshot_fps` (1 Hz) even with nobody
        streaming, so this is not stuck on whenever the last tab closed.
        """
        _video_or_404(camera_id)
        jpeg = server.video.latest(camera_id)
        if jpeg is None:
            raise HTTPException(404, f"no frame yet for {camera_id!r}; is the "
                                     f"camera running?")
        return Response(jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    # --- search and browse ------------------------------------------------
    @app.get("/search")
    async def search(q: str = Query(..., min_length=1,
                                    description="free text, e.g. "
                                                "'a person carrying a bag'"),
                     camera: str | None = None,
                     top: int = Query(24, ge=1, le=200),
                     model: str | None = None,
                     min_px: int = Query(0, ge=0, le=4096),
                     min_conf: float = Query(0.0, ge=0.0, le=1.0)):
        """Open-vocabulary search over embedded crops.

        THIS RANKS, IT DOES NOT DETECT. Every response carries `ranker_note`,
        `candidates` and `score_spread`, because measurement found no
        similarity threshold separating present concepts from absent ones -
        so a top hit is the nearest crop whether or not the thing asked for is
        in the footage. See src/query/search.py. Render the note.

        Runs in a threadpool: a cold text tower is ~600 ms and would otherwise
        stall the event loop that is streaming video to every open tab.
        """
        model = model or server.search_model
        if model not in SPECS:
            raise HTTPException(400, f"unknown model {model!r}; "
                                     f"known: {', '.join(sorted(SPECS))}")
        # `camera` is NOT validated against the running workers on purpose:
        # scoping to a camera whose run is historical is a legitimate search.
        return await run_in_threadpool(server.search, q, camera, top, model,
                                       min_px, min_conf)

    @app.get("/objects")
    def objects(cls: str | None = None, group: str | None = None,
                camera: str | None = None, flagged: bool = False,
                identity_id: int | None = None, has_crop: bool | None = None,
                attr_key: str | None = None, attr_value: str | None = None,
                limit: int = Query(60, ge=1, le=500),
                offset: int = Query(0, ge=0)):
        """Browse the object record with the filters the dashboard offers."""
        conn = server.store.connect_ro()
        try:
            return queries.list_objects(
                conn, cls=cls, group=group, camera=camera, flagged=flagged,
                identity_id=identity_id, has_crop=has_crop, attr_key=attr_key,
                attr_value=attr_value, limit=limit, offset=offset)
        finally:
            conn.close()

    @app.get("/objects/{object_id}")
    def object_detail(object_id: int):
        conn = server.store.connect_ro()
        try:
            obj = queries.get_object(conn, object_id)
        finally:
            conn.close()
        if obj is None:
            raise HTTPException(404, f"no object {object_id}")
        return obj

    @app.get("/identities")
    def identities(kind: str = "plate", key: str | None = None,
                   limit: int = Query(100, ge=1, le=500),
                   offset: int = Query(0, ge=0)):
        """Durable identities on one axis. `kind` defaults to 'plate'.

        Unbound sightings are not here by design: an object whose key never
        read confidently has no `identities` row."""
        conn = server.store.connect_ro()
        try:
            return queries.list_identities(conn, kind=kind, key=key,
                                           limit=limit, offset=offset)
        finally:
            conn.close()

    @app.get("/vocabulary")
    def vocabulary():
        """What the filters can offer, read from the data rather than declared."""
        conn = server.store.connect_ro()
        try:
            return queries.vocabulary(conn)
        finally:
            conn.close()

    # --- live view --------------------------------------------------------
    @app.websocket("/live/{camera_id}")
    async def live(ws: WebSocket, camera_id: str):
        """Frame metadata at the hub's fixed rate. No pixels.

        A slow client drops frames instead of applying backpressure: its queue
        is bounded and newest-wins, so a stalled browser tab cannot slow the
        camera thread feeding it.
        """
        await ws.accept()
        if server.workers.get(camera_id) is None:
            await ws.send_text(json.dumps({"error": f"no camera {camera_id!r}"}))
            await ws.close()
            return
        sub = server.hub.subscribe(camera_id, asyncio.get_running_loop())
        try:
            while True:
                message = await sub.queue.get()
                await ws.send_text(json.dumps(message, default=str))
                sub.sent += 1
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            server.hub.unsubscribe(sub)

    @app.exception_handler(Exception)
    async def unhandled(request, exc):       # pragma: no cover
        # A request handler raising must not look like a dead server.
        print(f"[server] unhandled error on {request.url.path}: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)

    return app


def _loads(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text
