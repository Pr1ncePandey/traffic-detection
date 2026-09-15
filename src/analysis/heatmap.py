"""Heatmap - colour the places where people and vehicles are most.

An analysis like congestion: it reads the tracked boxes, keeps its own
counters, and draws. Off unless a camera (or `--enable heatmap`) switches it
on. docs/heatmap.md has the reasoning and the planned analysis features that
build on it (zones, dwell, paths); this is only the map.

HOW

Every analysed frame, each box of a selected group (people, vehicles) adds 1
to the grid cell under its GROUND POINT - where it stands, which is what
"where people gather" means. A person standing still is counted every frame,
so gathering places get hot on their own. One grid per group, so the video can
show people, vehicles or both.

The grid is small (grid_w columns, rows by aspect ratio), so counting costs
the same at 720p and 4K. Rendering smooths the grid, scales it to its 99th
percentile, colours it, and marks only cells above `min_level` - cold areas
keep the original picture. Scaling is LOG by default: counts are heavy-tailed
(a parked car or a queue is counted every frame, a passer-by a few times), and
on a linear scale a spot visited once next to one visited 50 times sits at 2%
of the colour range - invisible. On log it is ~18%. The
coloured layer is rebuilt every `redraw_every` frames; each frame only blends
the cached layer onto the video.

half_life_s = 0 accumulates the whole run (a clip); > 0 lets old activity fade
so a live camera shows the last few minutes.

At the end of a run the maps are saved under `dir`: heatmap_all.png, one PNG
per group over a recent frame of the scene, and heatmap_grid.npz with the raw
counts for later analysis.
"""

import os

import cv2
import numpy as np

from ..runtime.plugin import Findings
from .base import block, register

COLORMAPS = {"jet": cv2.COLORMAP_JET, "turbo": cv2.COLORMAP_TURBO,
             "hot": cv2.COLORMAP_HOT, "inferno": cv2.COLORMAP_INFERNO}
ALL = "all"


class HeatmapAnalyzer:
    name = "heatmap"
    # compute() touches only this plugin's grids and cached layer; draw() only
    # blends that layer onto ctx.annotated.
    concurrent = True

    def __init__(self, cfg: dict):
        c = block(cfg, self.name)
        self.groups = [str(g) for g in (c.get("groups") or ["person", "vehicle"])]
        self.show = str(c.get("show", ALL))
        if self.show != ALL and self.show not in self.groups:
            print(f"[heatmap] show={self.show!r} is not one of {self.groups}; showing all")
            self.show = ALL
        self.grid_w = max(8, int(c.get("grid_w", 96)))
        self.blur = max(0.0, float(c.get("blur", 1.5)))
        self.half_life_s = max(0.0, float(c.get("half_life_s", 0)))
        self.alpha = min(1.0, max(0.0, float(c.get("alpha", 0.45))))
        self.min_level = min(0.99, max(0.0, float(c.get("min_level", 0.08))))
        self.scale = str(c.get("scale", "log")).lower()
        if self.scale not in ("log", "linear"):
            print(f"[heatmap] scale={self.scale!r} is not log or linear; using log")
            self.scale = "log"
        name = str(c.get("colormap", "jet")).lower()
        if name not in COLORMAPS:
            print(f"[heatmap] unknown colormap {name!r}; using jet "
                  f"({', '.join(COLORMAPS)})")
            name = "jet"
        self.colormap = COLORMAPS[name]
        self.redraw_every = max(1, int(c.get("redraw_every", 5)))
        self.draw_overlay = bool(c.get("draw", True))
        self.save = bool(c.get("save", True))
        self.out_dir = str(c.get("dir") or "outputs/heatmap")
        self.background_every = max(1, int(c.get("background_every", 150)))

        self.width = self.height = 0
        self.grid_h = 0
        self.grids: dict = {}
        self.counts = {g: 0 for g in self.groups}
        self.frames = 0
        self._last_ts = None
        self._layer = None           # (BGR colour image, bool mask) at video size
        self._since_render = 0
        self._background = None
        self.saved: list = []

    def setup(self, source, cfg: dict):
        self.width = max(1, int(getattr(source, "width", 0) or 1))
        self.height = max(1, int(getattr(source, "height", 0) or 1))
        self.grid_h = max(1, round(self.grid_w * self.height / self.width))
        self.grids = {g: np.zeros((self.grid_h, self.grid_w), np.float32)
                      for g in self.groups}

    # --- per frame ---------------------------------------------------------
    def compute(self, view) -> Findings:
        found = Findings(analyzer=self.name)
        if not self.grids:
            return found
        self._decay(float(view.timestamp))
        sx, sy = self.grid_w / self.width, self.grid_h / self.height
        for box in view.boxes:
            grid = self.grids.get(box.group)
            if grid is None:
                continue
            x, y = box.ground_point
            gx = min(self.grid_w - 1, max(0, int(x * sx)))
            gy = min(self.grid_h - 1, max(0, int(y * sy)))
            grid[gy, gx] += 1.0
            self.counts[box.group] += 1
        self.frames += 1
        if view.raw is not None and (self._background is None
                                     or self.frames % self.background_every == 0):
            self._background = view.raw.copy()
        self._since_render += 1
        if self._layer is None or self._since_render >= self.redraw_every:
            self._layer = self.render(self._selected(self.show))
            self._since_render = 0
        found.overlay = self._layer
        return found

    def _decay(self, ts: float):
        if self.half_life_s > 0 and self._last_ts is not None and ts > self._last_ts:
            factor = 0.5 ** ((ts - self._last_ts) / self.half_life_s)
            for grid in self.grids.values():
                grid *= factor
        self._last_ts = ts

    def _selected(self, which: str) -> np.ndarray:
        if which == ALL:
            return sum(self.grids.values())
        return self.grids[which]

    def render(self, grid):
        """(colour image, mask) at video size, or None when nothing is hot yet."""
        if grid is None or not np.any(grid > 0):
            return None
        # Log on the RAW counts, then smooth. Taking the log after smoothing was
        # tried first and painted almost everything red: the blur leaves tiny
        # tails around every spot, and measured against those tails every
        # visited cell looked like a hot spot (see docs/heatmap.md).
        g = np.log1p(grid) if self.scale == "log" else grid
        if self.blur > 0:
            g = cv2.GaussianBlur(g, (0, 0), self.blur)
        positive = g[g > 0]
        scale = float(np.percentile(positive, 99)) if positive.size else 0.0
        if scale <= 0:
            return None
        level = np.clip(g / scale, 0.0, 1.0)
        big = cv2.resize((level * 255).astype(np.uint8), (self.width, self.height),
                         interpolation=cv2.INTER_LINEAR)
        mask = big >= int(round(self.min_level * 255))
        if not mask.any():
            return None
        return cv2.applyColorMap(big, self.colormap), mask

    def blend(self, image, layer):
        """Blend a rendered layer onto `image` in place, only where it is hot."""
        if layer is None or image is None:
            return image
        colour, mask = layer
        if colour.shape[:2] != image.shape[:2]:
            return image
        mixed = cv2.addWeighted(image, 1.0 - self.alpha, colour, self.alpha, 0)
        np.copyto(image, mixed, where=mask[..., None])
        return image

    def apply(self, ctx, findings: Findings):
        """Nothing shared to write: a heatmap is drawn, not stored per object."""

    def draw(self, ctx, findings: Findings):
        if not self.draw_overlay or findings.overlay is None:
            return
        self.blend(ctx.annotated, findings.overlay)
        label = "heatmap: " + (" + ".join(self.groups) if self.show == ALL else self.show)
        cv2.putText(ctx.annotated, label, (10, ctx.annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- end of run --------------------------------------------------------
    def close(self):
        """Save the final maps once. Called by the stage when the run ends."""
        if not self.save or not self.frames or self.saved:
            return
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            base = (self._background if self._background is not None
                    else np.zeros((self.height, self.width, 3), np.uint8))
            for which in [ALL] + self.groups:
                layer = self.render(self._selected(which))
                if layer is None and which != ALL:
                    continue
                image = self.blend(base.copy(), layer)
                path = os.path.join(self.out_dir, f"heatmap_{which}.png")
                if cv2.imwrite(path, image):
                    self.saved.append(path)
            grid_path = os.path.join(self.out_dir, "heatmap_grid.npz")
            np.savez_compressed(grid_path, **self.grids,
                                frame_size=np.array([self.width, self.height]),
                                frames=np.array(self.frames))
            self.saved.append(grid_path)
        except Exception as e:
            print(f"[heatmap] could not save heatmaps to {self.out_dir}: {e}")

    def summary(self) -> dict:
        return {"frames": self.frames, "detections_counted": dict(self.counts),
                "grid": f"{self.grid_w}x{self.grid_h}",
                "half_life_s": self.half_life_s, "show": self.show,
                "saved": list(self.saved)}


register("heatmap", HeatmapAnalyzer)
