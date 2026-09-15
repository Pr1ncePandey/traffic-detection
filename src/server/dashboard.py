"""Where the dashboard's files live, and why they are files now.

THIS USED TO BE A PYTHON STRING

The dashboard was one `DASHBOARD_HTML = r"..."` literal in this module, on the
reasoning that it was "one file with no build step, and a StaticFiles mount
would be more moving parts than the thing it serves". That held at 311 lines.

It stopped holding when the page grew a video layer, six views, semantic
search and filter UI: a ~1,400-line HTML/CSS/JS document embedded in a Python
string has no syntax highlighting, no linting, and one stray backslash or
quote away from a runtime surprise no test would catch. The "no build step"
half of that decision is what mattered, and it is intact - these are plain
files served as they are, with nothing compiled, bundled or transpiled.

So: three static files, one mount, no build. `app.py` serves index.html at `/`
and mounts this directory at `/static`.
"""

import os

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
INDEX = os.path.join(STATIC_DIR, "index.html")


def missing() -> list:
    """Static files that should exist and do not.

    Checked at startup rather than on first request: a dashboard that 404s
    because a file did not ship should say so in the log when the server comes
    up, not when someone finally opens a browser.
    """
    return [p for p in (INDEX,
                        os.path.join(STATIC_DIR, "app.css"),
                        os.path.join(STATIC_DIR, "app.js"))
            if not os.path.exists(p)]
