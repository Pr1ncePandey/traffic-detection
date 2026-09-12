"""Run every camera as one long-lived service with a dashboard and webhooks.

  python serve.py                          every cameras/*.yaml, autostarted
  python serve.py --camera demo             just one
  python serve.py --no-autostart            come up idle; start from the UI
  python serve.py --port 9000

Then open http://127.0.0.1:8000/ for the dashboard, or /docs for the API.

SECURITY, STATED PLAINLY

There is NO application-level authentication. By decision, this service is
protected by network placement only, so it binds to 127.0.0.1 by default and
`--host 0.0.0.0` is an explicit act that prints a warning.

That default matters more than it looks: `/crops/{id}.jpg` serves number-plate
imagery and the WebSocket streams plate strings. Exposing this on an untrusted
network publishes both. Put it behind a VPN or an authenticating reverse proxy
before binding wider, and revisit the decision if this is ever operated by more
than one person or handed to a client.
"""

import argparse
import glob
import os

DEFAULT_HOST = "127.0.0.1"       # never 0.0.0.0; see the module docstring
DEFAULT_PORT = 8000


def discover_cameras() -> list:
    """Every cameras/<id>.yaml, by stem. Files starting with _ are skipped."""
    ids = []
    for path in sorted(glob.glob(os.path.join("cameras", "*.yaml"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if not stem.startswith("_"):
            ids.append(stem)
    return ids


def parse_args():
    p = argparse.ArgumentParser(description="Traffic detection service")
    p.add_argument("--config", default="config.yaml", help="fleet-wide defaults")
    p.add_argument("--camera", action="append", default=None, metavar="ID",
                   help="camera to run; repeatable. Default: every cameras/*.yaml")
    p.add_argument("--host", default=None,
                   help=f"listen address (default {DEFAULT_HOST}). There is no "
                        f"application auth, so widening this publishes plate "
                        f"imagery and plate strings to whoever can reach it.")
    p.add_argument("--port", type=int, default=None,
                   help=f"listen port (default {DEFAULT_PORT})")
    p.add_argument("--no-autostart", action="store_true",
                   help="come up with every camera idle; start them from the "
                        "dashboard or POST /cameras/{id}/start")
    p.add_argument("--log-level", default="warning",
                   choices=["critical", "error", "warning", "info", "debug"],
                   help="uvicorn's log level. Default warning, so the "
                        "pipeline's own output is not buried in access logs.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "serve.py needs fastapi and uvicorn:\n"
            "    pip install 'fastapi' 'uvicorn[standard]'")

    from src.config import load_for_camera
    from src.server.app import create_app

    cameras = args.camera or discover_cameras()
    if not cameras:
        raise SystemExit(
            "No cameras found. Create cameras/<id>.yaml (copy cameras/demo.yaml) "
            "or name one with --camera.")

    srv = (load_for_camera(None, args.config).get("server", {}) or {})
    host = args.host or srv.get("host") or DEFAULT_HOST
    port = int(args.port or srv.get("port") or DEFAULT_PORT)

    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n  WARNING: binding to {host}, not loopback.\n"
              f"  This service has NO application-level authentication. Anyone\n"
              f"  who can reach {host}:{port} can read number-plate imagery\n"
              f"  (/crops/*.jpg) and live plate strings (/live/*). Put it behind\n"
              f"  a VPN or an authenticating proxy.\n")

    print(f"[serve] cameras: {', '.join(cameras)}")
    print(f"[serve] dashboard http://{host}:{port}/  |  API docs /docs")
    if args.no_autostart:
        print("[serve] --no-autostart: cameras are idle until started")

    app = create_app(cameras, config_path=args.config,
                     autostart=not args.no_autostart)
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)


if __name__ == "__main__":
    main()
