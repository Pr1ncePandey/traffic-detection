"""Pure 2-D geometry. No analyzer, config or OpenCV concerns.

Everything here is a function of numbers, which is why the lane rules can be
tested without a video, a model or a frame (tools/test_lanes.py).

Two things in here exist because of real bugs:

`ensure_simple` - tools/draw_lanes.py stored clicked corners in CLICK order and
fed them straight to a point-in-polygon test, which needs PERIMETER order.
Clicking the four corners of a road in reading order (top-left, top-right,
bottom-left, bottom-right) therefore produced a self-intersecting "bowtie":
the drawn outline looked wrong and half the interior tested as OUTSIDE. Rather
than trust every caller and every already-saved yaml to get the order right,
polygons are repaired at load time.

`scale_points` - the previous version guessed ratio-vs-pixel per POINT with
`x <= 1 and y <= 1`, so a pixel polygon that happened to contain the point
(1, 1) had that one corner exploded to (width, height). The unit is a property
of the polygon, not of a point, so it is decided once for the whole ring.
"""

import math

# Cardinal flow vectors, in image coordinates: y grows DOWNWARD, so "up the
# frame" (away from a camera looking along the road) is dy = -1.
DIR_VEC = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
}


def unit(dx: float, dy: float) -> tuple[float, float]:
    """Normalize a vector. A zero vector stays zero rather than dividing by 0."""
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def direction_vector(spec) -> tuple[float, float]:
    """Accept "up"/"down"/"left"/"right", or an explicit [dx, dy]. Unit vector.

    Explicit vectors are what a diagonal road needs: a camera on a pole sees
    the carriageway running across the frame, and no cardinal direction
    describes it.
    """
    if spec is None:
        return (0.0, 0.0)
    if isinstance(spec, str):
        return DIR_VEC.get(spec.strip().lower(), (0.0, 0.0))
    try:
        dx, dy = float(spec[0]), float(spec[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return (0.0, 0.0)
    return unit(dx, dy)


def describe_vector(v) -> str:
    """Human-readable flow, e.g. "up-left (-25deg)". For logs and overlays."""
    dx, dy = v
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return "none"
    vertical = "up" if dy < -0.35 else ("down" if dy > 0.35 else "")
    horizontal = "left" if dx < -0.35 else ("right" if dx > 0.35 else "")
    name = "-".join(p for p in (vertical, horizontal) if p) or "sideways"
    # Screen angle, measured from "straight up the frame", positive clockwise.
    deg = math.degrees(math.atan2(dx, -dy))
    return f"{name} ({deg:+.0f}deg)"


# --- unit handling ---------------------------------------------------------

def looks_like_ratios(points) -> bool:
    """True when every coordinate is within [0, 1] - i.e. resolution-free.

    Decided for the ring as a whole. A polygon mixing the two conventions is
    a config error, not something to silently interpret per point.
    """
    for p in points:
        for v in (p[0], p[1]):
            if not (-0.001 <= float(v) <= 1.001):
                return False
    return True


def scale_points(points, width: int, height: int, units: str = "auto") -> list:
    """Ratio (0-1) or pixel coordinates -> pixels. `units`: auto|ratio|pixel."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if not pts:
        return []
    mode = str(units or "auto").lower()
    if mode == "auto":
        mode = "ratio" if looks_like_ratios(pts) else "pixel"
    if mode == "ratio":
        return [[int(round(x * width)), int(round(y * height))] for x, y in pts]
    return [[int(round(x)), int(round(y))] for x, y in pts]


def to_ratios(points, width: int, height: int, ndigits: int = 4) -> list:
    """Pixels -> ratios, for writing a resolution-independent config."""
    w = max(1, int(width))
    h = max(1, int(height))
    return [[round(float(x) / w, ndigits), round(float(y) / h, ndigits)]
            for x, y in points]


# --- polygons --------------------------------------------------------------

def point_in_polygon(px: float, py: float, ring) -> bool:
    """Standard crossing-number test. `ring` must be in perimeter order."""
    inside = False
    n = len(ring)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        if (y1 > py) != (y2 > py):
            x_at_py = (x2 - x1) * (py - y1) / ((y2 - y1) or 1e-9) + x1
            if px < x_at_py:
                inside = not inside
    return inside


def _point_segment_distance(px, py, x1, y1, x2, y2) -> float:
    dx, dy = x2 - x1, y2 - y1
    span = dx * dx + dy * dy
    t = 0.0 if span == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / span))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def distance_to_edge(px: float, py: float, ring) -> float:
    """Shortest distance from a point to the polygon boundary, in px."""
    best = float("inf")
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        best = min(best, _point_segment_distance(px, py, x1, y1, x2, y2))
    return best


def polygon_area(ring) -> float:
    """Signed shoelace area. Sign gives winding; magnitude gives area."""
    total = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _segments_cross(a, b, c, d) -> bool:
    """Do open segments ab and cd properly cross? Touching does not count."""
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1, d2 = cross(c, d, a), cross(c, d, b)
    d3, d4 = cross(a, b, c), cross(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def is_simple(ring) -> bool:
    """True when no two non-adjacent edges cross - i.e. not a bowtie."""
    n = len(ring)
    if n < 4:
        return True
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue  # shares a vertex
            if _segments_cross(a, b, ring[j], ring[(j + 1) % n]):
                return False
    return True


def order_ring(points) -> list:
    """Sort points into perimeter order by angle about their centroid.

    Correct for convex rings, which is what a road quad clicked on a frame is.
    Deliberately NOT applied unconditionally - see ensure_simple.
    """
    pts = [[float(p[0]), float(p[1])] for p in points]
    if len(pts) < 3:
        return pts
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def ensure_simple(points, name: str = "", verbose: bool = True) -> tuple[list, bool]:
    """Return (ring, repaired). Reorders ONLY a self-intersecting ring.

    Angle-sorting unconditionally would silently rewrite a legitimate concave
    polygon, so a ring that is already simple is returned untouched and stays
    identical to what the config said.
    """
    ring = [[int(round(p[0])), int(round(p[1]))] for p in points]
    if len(ring) < 3 or is_simple(ring):
        return ring, False
    fixed = [[int(round(p[0])), int(round(p[1]))] for p in order_ring(ring)]
    if not is_simple(fixed):
        if verbose:
            print(f"[geometry] polygon {name!r} is self-intersecting and could "
                  f"not be repaired by reordering; check its corners")
        return fixed, False
    if verbose:
        print(f"[geometry] polygon {name!r} corners were out of order "
              f"(self-intersecting); reordered to {fixed}")
    return fixed, True


# --- lines (the coming/going divider) --------------------------------------

class Line:
    """An infinite line through two points, used to answer "which side?".

    This is what a road divider actually is. The previous code answered
    "which side" by testing polygon containment, so a point on the shared
    boundary landed in whichever polygon happened to be listed first, and the
    margin meant to protect against that was a flat 14 px - several metres of
    road near the top of a frame, a few centimetres at the bottom.
    """

    __slots__ = ("p", "q", "_dx", "_dy", "_len")

    def __init__(self, p, q):
        self.p = (float(p[0]), float(p[1]))
        self.q = (float(q[0]), float(q[1]))
        self._dx = self.q[0] - self.p[0]
        self._dy = self.q[1] - self.p[1]
        self._len = math.hypot(self._dx, self._dy)

    @property
    def valid(self) -> bool:
        return self._len > 1e-6

    def cross(self, px: float, py: float) -> float:
        """Signed cross product. Sign = side, magnitude = distance * length."""
        return self._dx * (py - self.p[1]) - self._dy * (px - self.p[0])

    def side(self, px: float, py: float, margin: float = 0.0) -> int:
        """+1 / -1 for the two sides, 0 when within `margin` px of the line."""
        if not self.valid:
            return 0
        signed = self.cross(px, py) / self._len
        if abs(signed) <= margin:
            return 0
        return 1 if signed > 0 else -1

    def distance(self, px: float, py: float) -> float:
        if not self.valid:
            return float("inf")
        return abs(self.cross(px, py)) / self._len

    def x_at_y(self, y: float):
        """Where the line sits at a given row. None for a horizontal line."""
        if abs(self._dy) < 1e-9:
            return None
        t = (y - self.p[1]) / self._dy
        return self.p[0] + t * self._dx

    def clipped_to_frame(self, width: int, height: int):
        """Two endpoints spanning the frame, for drawing the full divider.

        A divider given as two clicked points usually stops short of the frame
        edges; drawing only that stub is one reason an operator concludes the
        line is in the wrong place.
        """
        if not self.valid:
            return None
        pts = []
        if abs(self._dy) > 1e-9:
            for y in (0.0, float(height - 1)):
                x = self.x_at_y(y)
                if x is not None and -1e6 < x < 1e6:
                    pts.append((x, y))
        if abs(self._dx) > 1e-9:
            for x in (0.0, float(width - 1)):
                t = (x - self.p[0]) / self._dx
                pts.append((x, self.p[1] + t * self._dy))
        inside = [(int(round(x)), int(round(y))) for x, y in pts
                  if -1 <= x <= width and -1 <= y <= height]
        if len(inside) < 2:
            return None
        # Farthest-apart pair, so a corner-clipping line still draws fully.
        best, best_d = None, -1.0
        for i in range(len(inside)):
            for j in range(i + 1, len(inside)):
                d = math.hypot(inside[i][0] - inside[j][0],
                               inside[i][1] - inside[j][1])
                if d > best_d:
                    best, best_d = (inside[i], inside[j]), d
        return best

    def as_points(self) -> list:
        return [[self.p[0], self.p[1]], [self.q[0], self.q[1]]]

    def __repr__(self):
        return f"Line({self.p} -> {self.q})"


def fit_line_through(points):
    """Least-squares divider through (x, y) samples, as a Line.

    Fits x = a*y + b, NOT y = f(x): a road divider seen from a pole is close
    to vertical in the image, and fitting y against x would blow up exactly
    there.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        return None
    n = len(pts)
    mean_y = sum(y for _, y in pts) / n
    mean_x = sum(x for x, _ in pts) / n
    var_y = sum((y - mean_y) ** 2 for _, y in pts)
    if var_y < 1e-6:
        # All samples on one row: only a horizontal divider is defensible.
        return Line((mean_x - 100.0, mean_y), (mean_x + 100.0, mean_y))
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pts)
    a = cov / var_y
    b = mean_x - a * mean_y
    y0 = min(y for _, y in pts)
    y1 = max(y for _, y in pts)
    span = max(1.0, y1 - y0)
    # Extend well past the sampled rows so the line spans the whole frame.
    y0, y1 = y0 - span, y1 + span
    return Line((a * y0 + b, y0), (a * y1 + b, y1))


def convex_hull(points) -> list:
    """Andrew monotone chain. Used to bound where traffic actually drives.

    An auto-built lane that is simply the whole frame includes the footpath and
    the trees; the hull of observed ground points is the road as evidenced.
    """
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) < 3:
        return [[x, y] for x, y in pts]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                o, q = out[-2], out[-1]
                if (q[0] - o[0]) * (p[1] - o[1]) - (q[1] - o[1]) * (p[0] - o[0]) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return [[x, y] for x, y in (lower[:-1] + upper[:-1])]


def dilate_ring(ring, px: float) -> list:
    """Push every vertex `px` outward from the centroid.

    A hull drawn tight around observed ground points clips the vehicles that
    drive at the very edge of it, so the auto lane is grown slightly.
    """
    if not ring:
        return []
    n = len(ring)
    cx = sum(p[0] for p in ring) / n
    cy = sum(p[1] for p in ring) / n
    out = []
    for x, y in ring:
        dx, dy = float(x) - cx, float(y) - cy
        length = math.hypot(dx, dy)
        if length < 1e-9:
            out.append([int(round(x)), int(round(y))])
            continue
        out.append([int(round(x + dx / length * px)),
                    int(round(y + dy / length * px))])
    return out


def clip_to_halfplane(ring, line: "Line", side: int) -> list:
    """Sutherland-Hodgman clip of a polygon to one side of a line.

    Splitting the frame rectangle by the divider gives two lane polygons that
    tile it exactly, with no gap along the divider for an object to fall into
    - which is what the two hand-written demo polygons had.
    """
    if not ring or line is None or not line.valid or side == 0:
        return [[int(round(p[0])), int(round(p[1]))] for p in ring]
    sign = 1.0 if side > 0 else -1.0

    def inside(p):
        return sign * line.cross(p[0], p[1]) >= 0.0

    out = []
    n = len(ring)
    for i in range(n):
        a = (float(ring[i][0]), float(ring[i][1]))
        b = (float(ring[(i + 1) % n][0]), float(ring[(i + 1) % n][1]))
        ina, inb = inside(a), inside(b)
        if ina:
            out.append(a)
        if ina != inb:
            ca, cb = line.cross(*a), line.cross(*b)
            denom = ca - cb
            t = 0.5 if abs(denom) < 1e-12 else ca / denom
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return [[int(round(x)), int(round(y))] for x, y in out]


def frame_ring(width: int, height: int) -> list:
    return [[0, 0], [int(width), 0], [int(width), int(height)], [0, int(height)]]


def clamp_ring(ring, width: int, height: int) -> list:
    """Pull vertices back inside the frame.

    A dilated hull pokes a few pixels outside the image, which then round-trips
    into a config with ratios like 1.031 - harmless to the maths, but it makes
    a saved lane block look wrong to whoever reads it next.
    """
    w, h = int(width), int(height)
    return [[max(0, min(w, int(round(p[0])))),
             max(0, min(h, int(round(p[1]))))] for p in ring]
