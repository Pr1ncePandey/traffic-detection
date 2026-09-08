"""Lane bifurcation: which road is the vehicle on, and is it going the right way?

Two flags (simple English):
  wrong_way  = you are INSIDE Lane-A (arrow up) but MOVING down. Safety alert.
  wrong_lane = you are the wrong TYPE in a restricted lane (truck in bus-only),
               even if moving the right way. Rule alert.

Polygons are resolution-independent ratios (0-1). Example demo clip:
  left road  x 0-0.48, arrow DOWN (coming, top->bottom, mostly empty).
  right road x 0.52-1.0, arrow UP (going, bottom->top, busy).
Unknown clip: lanes_mode=auto builds these two halves as a starting guess;
tools/calibrate.py --suggest-lanes refines the split from real motion.
"""

import cv2
import numpy as np

DIR_VEC = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
MIN_MOVE = 4        # px per frame below this = jitter, not a direction
EDGE_MARGIN = 14    # px near a lane edge = too close to call -> ok (kills fence wobble)
WRONG_CONFIRM = 5   # consecutive opposite frames before wrong_way is confirmed


def _scale(poly, W, H):
    out = []
    for p in poly:
        x, y = p[0], p[1]
        if x <= 1.0 and y <= 1.0:  # ratio -> pixels
            out.append([int(x * W), int(y * H)])
        else:
            out.append([int(x), int(y)])
    return out


def point_in_polygon(cx, cy, poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > cy) != (y2 > cy) and cx < (x2 - x1) * (cy - y1) / (y2 - y1 + 1e-9) + x1:
            inside = not inside
    return inside


def dist_to_edge(cx, cy, poly) -> float:
    """Min distance from point to any polygon edge (px)."""
    best = float("inf")
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / L2))
        px, py = x1 + t * dx, y1 + t * dy
        best = min(best, ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
    return best


class LaneChecker:
    def __init__(self, lanes_cfg, W, H, mode="explicit"):
        self.W, self.H = W, H
        self.mode = mode
        if mode == "auto":
            # Unknown clip: ignore yaml, guess vertical halves (left=down, right=up).
            # calibrate --suggest-lanes or draw_lanes.py refines it into saved yaml.
            lanes_cfg = [
                {"name": "left", "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]],
                 "direction": "down", "allowed": []},
                {"name": "right", "polygon": [[0.5, 0], [1.0, 0], [1.0, 1], [0.5, 1]],
                 "direction": "up", "allowed": []},
            ]
        self.lanes = []
        for L in (lanes_cfg or []):
            poly = _scale(L.get("polygon", []), W, H)
            if len(poly) < 3:
                continue
            self.lanes.append({"name": str(L.get("name", "lane")),
                               "polygon": poly,
                               "direction": str(L.get("direction", "up")).lower(),
                               "allowed": list(L.get("allowed", []) or [])})
        self.enabled = bool(self.lanes) and mode != "off"
        self._streak = {}  # tid -> consecutive opposite-motion frames (wobble filter)

    def check(self, cx, cy, prev_cx, prev_cy, vclass="", tid=None):
        """Returns (lane_id, flag). flag in ok|wrong_way|wrong_lane|wrong_way+wrong_lane.

        wrong_way needs WRONG_CONFIRM consecutive opposite frames + the point
        must be EDGE_MARGIN px inside the lane (fence-straddlers read ok).
        """
        lane = None
        for L in self.lanes:
            if point_in_polygon(cx, cy, L["polygon"]):
                lane = L
                break
        if lane is None or not self.enabled:
            if tid is not None:
                self._streak.pop(tid, None)
            return ("", "ok")
        opposite = False
        if prev_cx is not None and prev_cy is not None:
            dx, dy = cx - prev_cx, cy - prev_cy
            d = lane["direction"]
            if d == "up" and dy > MIN_MOVE:
                opposite = True
            elif d == "down" and dy < -MIN_MOVE:
                opposite = True
            elif d == "left" and dx > MIN_MOVE:
                opposite = True
            elif d == "right" and dx < -MIN_MOVE:
                opposite = True
        flag = "ok"
        if opposite and dist_to_edge(cx, cy, lane["polygon"]) >= EDGE_MARGIN:
            n = self._streak.get(tid, 0) + 1 if tid is not None else WRONG_CONFIRM
            if tid is not None:
                self._streak[tid] = n
            if n >= WRONG_CONFIRM:
                flag = "wrong_way"
        else:
            if tid is not None:
                self._streak.pop(tid, None)
        if lane["allowed"] and vclass and vclass not in lane["allowed"]:
            flag = "wrong_lane" if flag == "ok" else "wrong_way+wrong_lane"
        return (lane["name"], flag)

    def draw(self, frame, lane_counts=None):
        if not self.enabled:
            return frame
        lane_counts = lane_counts or {}
        for L in self.lanes:
            pts = np.array(L["polygon"], np.int32)
            cv2.polylines(frame, [pts], True, (255, 255, 0), 2)
            x0 = min(p[0] for p in L["polygon"])
            n = lane_counts.get(L["name"], 0)
            arrow = {"up": "^", "down": "v", "left": "<", "right": ">"}.get(L["direction"], "?")
            cv2.putText(frame, f"{L['name']} {arrow} n={n}", (x0 + 5, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        return frame
