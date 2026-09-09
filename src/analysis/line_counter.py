"""Counting logic + drawing. ZoneCounter (two lines A/B) is primary.

Direction-neutral: code counts A->B and B->A only. Human labels
("bottom entry" -> "top exit") live in config camera block, one file per CCTV.
Single-line LineCounter kept for backward compat / --no-line tests.
"""

import cv2


class ZoneCounter:
    """Two lines: A (lower, entry side) and B (upper, exit side)."""

    def __init__(self, cfg: dict, frame_h: int, legacy: dict | None = None):
        cfg = cfg or {}
        # Fallback to legacy single line: split it into a band around it.
        if not cfg and legacy:
            y = float(legacy.get("y_ratio", 0.6))
            cfg = {"line_a_ratio": max(0.05, y - 0.15), "line_b_ratio": min(0.95, y + 0.15)}
        self.enabled = bool(cfg.get("enabled", True) if "enabled" in cfg
                            else (legacy.get("enabled", True) if legacy else True))
        self.a_ratio = float(cfg.get("line_a_ratio", 0.35))
        self.b_ratio = float(cfg.get("line_b_ratio", 0.65))
        self.label_a = str(cfg.get("label_a", "bottom (entry)"))
        self.label_b = str(cfg.get("label_b", "top (exit)"))
        self.color_a = tuple(cfg.get("color_a", [255, 0, 0]))
        self.color_b = tuple(cfg.get("color_b", [0, 255, 255]))
        self.line_a_y = int(frame_h * self.a_ratio)
        self.line_b_y = int(frame_h * self.b_ratio)

    def draw(self, frame, a_to_b: int, b_to_a: int):
        if not self.enabled:
            return frame
        h, w = frame.shape[:2]
        cv2.line(frame, (0, self.line_a_y), (w, self.line_a_y), self.color_a, 2)
        cv2.line(frame, (0, self.line_b_y), (w, self.line_b_y), self.color_b, 2)
        cv2.putText(frame, f"A->B:{a_to_b} B->A:{b_to_a}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"A:{self.label_a}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_a, 1)
        cv2.putText(frame, f"B:{self.label_b}", (10, self.line_a_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_b, 1)
        return frame


class LineCounter:
    def __init__(self, cfg: dict, frame_h: int):
        self.enabled = bool(cfg.get("enabled", True))
        self.y = int(frame_h * float(cfg.get("y_ratio", 0.6)))
        self.color = tuple(cfg.get("color", [0, 255, 255]))

    def draw(self, frame, in_count: int, out_count: int):
        if not self.enabled:
            return frame
        h, w = frame.shape[:2]
        cv2.line(frame, (0, self.y), (w, self.y), self.color, 2)
        cv2.putText(frame, f"IN:{in_count} OUT:{out_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, 2)
        return frame
