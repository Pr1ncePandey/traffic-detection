"""Draw lane polygons by clicking on one frozen frame. One-time per CCTV.

Usage:
  python tools/draw_lanes.py --source samples/input.mp4 --out cameras/shop1.yaml
  python tools/draw_lanes.py --source samples/input.mp4 --frame 100

Clicks (simple English):
  - Left-click 4 corners of road 1 (any order) -> type name + direction in console.
  - Then 4 corners of road 2, etc. Works for 2 separate roads AND 1 road/2 ways
    (adjacent polygons sharing the divider edge are fine).
  - Press U = undo last point, ENTER/S = save yaml snippet, Q/ESC = quit.

Output: lanes block in ratio coords (0-1, any resolution) ready to paste into
cameras/<name>.yaml. Then: python main.py --camera <name>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

_points = []
_polys = []


def _on_mouse(event, x, y, flags, img):
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append((x, y))


def main():
    p = argparse.ArgumentParser(description="Click lane polygons, save cameras yaml block")
    p.add_argument("--source", default="samples/input.mp4")
    p.add_argument("--frame", type=int, default=60)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read frame. Try another --frame number.")
    H, W = img.shape[:2]

    cv2.namedWindow("click 4 corners per road | U undo | ENTER save | Q quit")
    cv2.setMouseCallback("click 4 corners per road | U undo | ENTER save | Q quit",
                         _on_mouse, img)
    print("Click 4 corners per road. U=undo, ENTER=finish road, S=save+quit, Q=quit.")
    while True:
        view = img.copy()
        for (x, y) in _points:
            cv2.circle(view, (x, y), 5, (0, 255, 255), -1)
        for poly in _polys:
            for i, (x, y) in enumerate(poly["px"]):
                cv2.circle(view, (x, y), 4, (255, 255, 0), -1)
                cv2.putText(view, f"{poly['name']}", (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("click 4 corners per road | U undo | ENTER save | Q quit", view)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("u") and _points:
            _points.pop()
        elif k in (13, ord("s"), ord("S")) and len(_points) >= 3:
            name = input("Road name (e.g. left_coming): ").strip() or f"road{len(_polys) + 1}"
            direction = input("Direction (up/down/left/right): ").strip().lower() or "up"
            _polys.append({"name": name, "direction": direction, "px": list(_points)})
            _points.clear()
            print(f"Saved {name} ({direction}). Click next road or S to finish.")
            if k in (ord("s"), ord("S")):
                break
    cv2.destroyAllWindows()
    if not _polys:
        print("No polygons. Nothing saved.")
        return
    lines = ["lanes_mode: \"explicit\"", "lanes:"]
    for poly in _polys:
        ratios = [[round(x / W, 3), round(y / H, 3)] for (x, y) in poly["px"]]
        lines.append(f"  - name: \"{poly['name']}\"")
        lines.append(f"    polygon: {ratios}")
        lines.append(f"    direction: \"{poly['direction']}\"")
        lines.append("    allowed: []")
    snippet = "\n".join(lines) + "\n"
    print("--- paste into cameras/<name>.yaml ---\n" + snippet)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Lane polygons from tools/draw_lanes.py - confirm once per CCTV.\n" + snippet)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
