"""The HTTP/WebSocket surface: one server, N camera workers, one DB writer.

ON AUTH, WHICH THERE IS NONE OF

By decision, the API, WebSocket and crop endpoints are protected by NETWORK
PLACEMENT ONLY. Two things follow, and they are not optional given that
`/crops/{id}.jpg` serves number-plate imagery and the WebSocket streams plate
strings:

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
    hub        WebSocket fan-out, needs nothing
    workers    N camera threads, need store + hub + policy
    dispatcher webhook delivery, needs store + policy
    retention  crop and row reaping, needs the store

The frame loop never blocks on any of the last three: the hub drops for slow
viewers, the dispatcher owns its thread, and retention owns another.
"""

import asyncio
import json
import os
import time

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..config import load_for_camera
from ..incidents import IncidentPolicy
from ..journeys import build_path, yield_summary
from ..storage.retention import from_config as retention_from_config
from ..storage.sqlite_store import SqliteStore
from .dashboard import DASHBOARD_HTML
from .hub import LiveHub
from .webhooks import WebhookDispatcher
from .workers import WorkerPool


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
        self.policy = IncidentPolicy(cfg, self.store)
        self.workers = WorkerPool(cameras, config_path=config_path, hub=self.hub,
                                  policy=self.policy, storage=self.store)
        # Locations are read from the loaded camera configs rather than from
        # the run rows, because a payload is built while the run is live.
        self.policy.cameras = self.workers.locations()
        self.dispatcher = WebhookDispatcher(self.store, self.policy)
        self.retention = retention_from_config(cfg, self.store)
        self.started_at = time.time()

    def start(self):
        self.dispatcher.start()
        self.retention.start()
        if self.autostart:
            self.workers.start_all()

    def stop(self):
        # Cameras first: they are the only thing still producing work, and
        # stopping them lets the dispatcher and retention drain what exists.
        self.workers.stop_all()
        self.dispatcher.stop()
        self.retention.stop()
        self.store.flush()
        # end_run=False: each worker already closed its own run row by id. The
        # store's own run_id points at whichever camera started last, and
        # stamping that one twice would be wrong.
        self.store.close(end_run=False)

    def health(self) -> dict:
        return {"uptime_s": round(time.time() - self.started_at, 1),
                "cameras": len(self.workers.workers),
                "unhealthy": self.workers.unhealthy(),
                "storage": self.store.health(),
                "hub": self.hub.stats(),
                "incidents": self.policy.stats(),
                "webhooks": self.dispatcher.stats(),
                "retention": self.retention.stats()}


def create_app(cameras: list, config_path: str = "config.yaml",
               autostart: bool = True) -> FastAPI:
    cfg = load_for_camera(None, config_path)
    server = Server(cameras, cfg, config_path=config_path, autostart=autostart)
    app = FastAPI(title="traffic-detection", version="1.0",
                  docs_url="/docs", redoc_url=None)
    app.state.server = server

    @app.on_event("startup")
    def _startup():
        server.start()

    @app.on_event("shutdown")
    def _shutdown():
        server.stop()

    # --- dashboard --------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD_HTML

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

    @app.get("/vehicles/{vehicle_id}")
    def vehicle(vehicle_id: int):
        conn = server.store.connect_ro()
        try:
            veh = conn.execute("SELECT * FROM vehicles WHERE id=?",
                               (vehicle_id,)).fetchone()
            if veh is None:
                raise HTTPException(404, f"no vehicle {vehicle_id}")
            rows = conn.execute(
                "SELECT o.*, r.camera FROM objects o"
                " JOIN runs r ON r.id = o.run_id"
                " WHERE o.vehicle_id=? ORDER BY o.id", (vehicle_id,)).fetchall()
        finally:
            conn.close()
        return {"vehicle": dict(veh), "sightings": [dict(r) for r in rows]}

    @app.get("/vehicles/{vehicle_id}/path")
    def vehicle_path(vehicle_id: int):
        """Cross-camera journey. See src/journeys.py and docs/multi-camera-paths.md.

        Reports observed hops only; every unobserved stretch is a gap, and
        sightings with no comparable clock are returned in `unorderable`
        rather than being silently ordered.
        """
        j = server.cfg.get("journey", {}) or {}
        conn = server.store.connect_ro()
        try:
            path = build_path(conn, vehicle_id,
                              max_gap_s=float(j.get("max_gap_s", 1800)),
                              max_speed_kmh=float(j.get("max_speed_kmh", 150)))
        finally:
            conn.close()
        if path is None:
            raise HTTPException(404, f"no vehicle {vehicle_id}")
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
