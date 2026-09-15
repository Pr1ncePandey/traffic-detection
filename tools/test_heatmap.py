"""Self-test for the heatmap analysis (src/analysis/heatmap.py).

    python tools/test_heatmap.py

Fabricated boxes, no video, no models. A heatmap fails quietly - heat in the
wrong place, one group leaking into another, a map that never fades, a video
tinted everywhere - so each of those is checked.
"""

import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.base import build, enabled_names  # noqa: E402
from src.analysis.heatmap import HeatmapAnalyzer  # noqa: E402
from src.config import load_for_camera  # noqa: E402
from src.runtime.context import AnalysisView, Detection, TrackedBox  # noqa: E402

_passed, _failed = 0, []


def check(label, condition, detail=""):
    global _passed
    if condition:
        _passed += 1
        print(f"  ok   {label}")
    else:
        _failed.append(label)
        print(f"  FAIL {label}{('  <- ' + str(detail)) if detail else ''}")


class Source:
    fps, width, height = 30.0, 1280, 720


class Ctx:
    def __init__(self):
        self.annotated = np.full((720, 1280, 3), 100, np.uint8)


def box(x1, y1, x2, y2, group="person", tid=1):
    name = "person" if group == "person" else "car"
    return TrackedBox.of(Detection(0, name, group, 0.9, (x1, y1, x2, y2), track_id=tid))


def view(boxes, ts=0.0, raw=True):
    return AnalysisView(frame_no=0, timestamp=ts, boxes=tuple(boxes), width=1280, height=720,
                        raw=np.full((720, 1280, 3), 60, np.uint8) if raw else None)


def fresh(**kw):
    cfg = {"analyses": {"heatmap": {"enabled": True, "redraw_every": 1, "save": False, **kw}}}
    h = HeatmapAnalyzer(cfg)
    h.setup(Source(), cfg)
    return h


print("\ncounting")
h = fresh()
check("grid follows the frame's aspect ratio", h.grids["person"].shape == (54, 96), h.grids["person"].shape)
# person standing with feet at (640, 700): cell (x=48, y=52)
for _ in range(10):
    h.compute(view([box(600, 500, 680, 700)]))
g = h.grids["person"]
check("heat lands where the person STANDS (ground point), not at the box centre",
      g[52, 48] == 10 and g[33, 48] == 0, (g[52, 48], g[33, 48]))
check("a person standing still keeps adding heat every frame", h.counts["person"] == 10)
check("people do not heat the vehicle map", h.grids["vehicle"].sum() == 0)
h.compute(view([box(100, 100, 300, 250, group="vehicle", tid=2)]))
check("vehicles go to their own map", h.grids["vehicle"].sum() == 1 and h.counts["vehicle"] == 1)
h.compute(view([box(10, 10, 50, 50, group="animal", tid=3)]))
check("groups not listed are ignored", sum(h.counts.values()) == 11, h.counts)
h.compute(view([box(1250, 700, 1400, 900, tid=4)]))
check("a box past the frame edge is clamped, not a crash", h.grids["person"][53, 95] == 1)
h2 = fresh(groups=["vehicle"])
h2.compute(view([box(600, 500, 680, 700)]))
check("groups: [vehicle] ignores people", "person" not in h2.grids and h2.frames == 1)

print("\nfading")
h = fresh()
h.compute(view([box(600, 500, 680, 700)], ts=0.0))
h.compute(view([], ts=100.0))
check("half_life_s 0 keeps the whole run", h.grids["person"].sum() == 1.0)
h = fresh(half_life_s=10)
h.compute(view([box(600, 500, 680, 700)], ts=0.0))
h.compute(view([], ts=10.0))
check("after one half-life the heat halves", abs(h.grids["person"].sum() - 0.5) < 1e-4, h.grids["person"].sum())
h.compute(view([], ts=30.0))
check("...and keeps fading", abs(h.grids["person"].sum() - 0.125) < 1e-4, h.grids["person"].sum())

print("\nrendering")
h = fresh()
f = h.compute(view([]))
check("nothing seen yet: no overlay", f.overlay is None)
ctx = Ctx()
h.draw(ctx, f)
check("...and the video is untouched", (ctx.annotated == 100).all())
for _ in range(20):
    f = h.compute(view([box(600, 500, 680, 700)]))
ctx = Ctx()
h.draw(ctx, f)
changed = (ctx.annotated != 100).any(axis=2)
check("hot area is coloured", changed[690:720, 620:660].any())
check("cold areas keep the original picture", not changed[0:300, 0:300].any())
colour, mask = f.overlay
check("the layer is at video size", colour.shape == (720, 1280, 3) and mask.shape == (720, 1280))
h = fresh()
for _ in range(50):
    h.compute(view([box(100, 500, 180, 700, tid=1)]))
f = h.compute(view([box(1000, 500, 1080, 700, tid=2)]))
check("log scale: a spot seen once still shows next to one seen 50 times",
      f.overlay[1][700, 1040])
lin = fresh(scale="linear")
for _ in range(50):
    lin.compute(view([box(100, 500, 180, 700, tid=1)]))
f = lin.compute(view([box(1000, 500, 1080, 700, tid=2)]))
check("linear scale: the same spot is washed out (why log is the default)",
      f.overlay is not None and not f.overlay[1][700, 1040])
h = fresh(show="vehicle")
f = h.compute(view([box(600, 500, 680, 700)]))
check("show: vehicle hides the people map", f.overlay is None)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    bad = fresh(show="cats", colormap="rainbow")
check("bad show / colormap fall back with a message",
      bad.show == "all" and "show" in buf.getvalue() and "colormap" in buf.getvalue())
h = fresh(redraw_every=10)
f1 = h.compute(view([box(600, 500, 680, 700)]))
f2 = h.compute(view([box(100, 500, 180, 700)]))
check("layer is reused between redraws (not re-rendered every frame)", f1.overlay is f2.overlay)
h = fresh(draw=False)
f = h.compute(view([box(600, 500, 680, 700)]))
ctx = Ctx()
h.draw(ctx, f)
check("draw: false leaves the video alone", (ctx.annotated == 100).all())

print("\nsaving")
with tempfile.TemporaryDirectory() as tmp:
    h = fresh(save=True, dir=tmp)
    for _ in range(5):
        h.compute(view([box(600, 500, 680, 700), box(100, 300, 300, 450, group="vehicle", tid=2)]))
    h.close()
    names = sorted(os.path.basename(p) for p in h.saved)
    check("end of run saves all, per-group maps and raw grids",
          names == ["heatmap_all.png", "heatmap_grid.npz", "heatmap_person.png", "heatmap_vehicle.png"], names)
    with np.load(os.path.join(tmp, "heatmap_grid.npz")) as data:
        check("raw grids keep the counts", data["person"].sum() == 5 and int(data["frames"]) == 5)
    h.close()
    check("closing twice does not save twice", len(h.saved) == 4)
    check("summary lists what was saved", h.summary()["saved"] == h.saved)
    empty = fresh(save=True, dir=os.path.join(tmp, "empty"))
    empty.close()
    check("a run with no frames saves nothing", not empty.saved and not os.path.exists(os.path.join(tmp, "empty")))

print("\nwiring")
cfg = load_for_camera(None)
check("heatmap is OFF by default", "heatmap" not in enabled_names(cfg))
check("heatmap is declared first, so it draws under lines and boxes",
      list(cfg["analyses"])[0] == "heatmap", list(cfg["analyses"]))
check("output folder is per camera", cfg["analyses"]["heatmap"]["dir"] == "outputs/demo/heatmap",
      cfg["analyses"]["heatmap"]["dir"])
cfg["analyses"]["heatmap"]["enabled"] = True
cfg["analyses"]["heatmap"]["save"] = False
with contextlib.redirect_stdout(io.StringIO()):
    built = build(cfg, Source())
check("enabled, it is built like any analysis", any(a.name == "heatmap" for a in built))
check("background search embeddings stay OFF in config.yaml",
      cfg["server"]["search"]["embed_in_background"] is False)

print(f"\n{'=' * 60}\n{_passed} passed, {len(_failed)} failed")
for name in _failed:
    print(f"  FAILED: {name}")
sys.exit(1 if _failed else 0)
