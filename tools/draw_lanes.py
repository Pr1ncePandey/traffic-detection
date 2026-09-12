"""Draw lane polygons and the divider by clicking on one frozen frame.

Usage:
  python tools/draw_lanes.py --source samples/input.mp4 --out cameras/shop1.yaml
  python tools/draw_lanes.py --source samples/input.mp4 --frame 100

Clicks:
  - Left-click the corners of road 1, then ENTER, then type its name and the
    direction traffic flows in it.
  - Repeat for road 2, etc. Adjacent roads sharing a divider edge are fine.
  - D    = the two clicked points are the DIVIDER (the coming/going boundary),
           not a road. Click exactly 2 points first.
  - U    = undo the last point
  - ENTER= finish this road
  - S    = save and quit,  Q/ESC = quit without saving

THE BUG THIS FILE USED TO HAVE

The docstring said "click 4 corners (any order)". It was not true. The points
were stored in CLICK order and written straight into `polygon:`, but a
point-in-polygon test needs PERIMETER order. Clicking the four corners in
natural reading order - top-left, top-right, bottom-left, bottom-right -
produced a self-intersecting bowtie:

    polygon [[100,100],[500,100],[100,400],[500,400]]
      point (150,150) inside? False     <- clearly inside the intended box
      point (450,350) inside? False

So the outline drawn on the video looked wrong AND half the road tested as
outside the lane. Corners are now sorted into perimeter order before saving,
and the ring is verified to be non-self-intersecting; the preview draws the
closed outline so what you are about to save is what you see.

Output: a lanes block in ratio coords (0-1, resolution-independent), ready to
paste into cameras/<name>.yaml. Then: python main.py --camera <name>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from src.analysis.geometry import (Line, describe_vector, direction_vector,
                                   ensure_simple, is_simple, order_ring,
                                   to_ratios)

_points = []
_polys = []
_divider = None

WIN = "lanes: click corners | D divider | U undo | ENTER road | S save | Q quit"


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append((x, y))


def _read_frame(source, index):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    # Read forward rather than seeking: CAP_PROP_POS_FRAMES is a no-op on a
    # live stream, so seeking here made this tool file-only.
    ret, img = False, None
    for _ in range(max(1, index)):
        ret, img = cap.read()
        if not ret:
            break
    cap.release()
    if not ret:
        raise RuntimeError(f"Source ended before frame {index}. "
                           f"Try a smaller --frame.")
    return img


def _preview(img):
    view = img.copy()
    h, w = view.shape[:2]
    for (x, y) in _points:
        cv2.circle(view, (x, y), 5, (0, 255, 255), -1)
    if len(_points) >= 2:
        # Show the ring you would actually get, closed, in perimeter order -
        # so a bowtie is visible before it is saved rather than after.
        ring = order_ring(_points) if len(_points) >= 3 else list(_points)
        pts = [(int(p[0]), int(p[1])) for p in ring]
        for i in range(len(pts)):
            cv2.line(view, pts[i], pts[(i + 1) % len(pts)], (0, 255, 255), 1)
    for poly in _polys:
        pts = [(int(x), int(y)) for x, y in poly["ring"]]
        for i in range(len(pts)):
            cv2.line(view, pts[i], pts[(i + 1) % len(pts)], (255, 255, 0), 2)
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        cv2.putText(view, poly["name"], (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        flow = poly["flow"]
        tip = (int(cx + flow[0] * 60), int(cy + flow[1] * 60))
        cv2.arrowedLine(view, (cx, cy), tip, (0, 255, 255), 3, tipLength=0.35)
    if _divider is not None:
        ends = _divider.clipped_to_frame(w, h)
        if ends:
            cv2.line(view, ends[0], ends[1], (0, 0, 255), 3)
            cv2.putText(view, "divider", (ends[0][0] + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(view, f"points:{len(_points)} roads:{len(_polys)} "
                      f"divider:{'yes' if _divider else 'no'}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return view


def _finish_road(img_shape):
    global _points
    if len(_points) < 3:
        print(f"A road needs 3+ corners, got {len(_points)}. Keep clicking.")
        return
    ring, repaired = ensure_simple(order_ring(_points), "new road", verbose=False)
    if repaired or not is_simple([[int(p[0]), int(p[1])] for p in _points]):
        print("  (corners reordered into perimeter order - clicking them in "
              "reading order would otherwise have made a self-crossing shape)")
    if not is_simple(ring):
        print("  WARNING: these corners still cross themselves. Undo (U) and "
              "click them going round the road, not across it.")
    name = input("Road name (e.g. right_going): ").strip() or f"road{len(_polys) + 1}"
    raw = input("Flow - up/down/left/right, or 'dx,dy' (e.g. -0.08,-1): ").strip()
    flow = direction_vector([float(v) for v in raw.split(",")]
                            if "," in raw else (raw or "up"))
    if flow == (0.0, 0.0):
        print(f"  could not read {raw!r} as a direction; defaulting to up")
        flow = direction_vector("up")
    _polys.append({"name": name, "ring": ring, "flow": flow})
    print(f"  saved {name}: {len(ring)} corners, flow {describe_vector(flow)}")
    _points = []


def _snippet(width, height):
    lines = ['analyses:', '  lanes:', '    mode: "explicit"']
    if _divider is not None:
        pts = to_ratios(_divider.as_points(), width, height, 4)
        lines += ["divider:", f"  points: {pts}"]
    lines.append("lanes:")
    for poly in _polys:
        lines.append(f'  - name: "{poly["name"]}"')
        lines.append(f"    polygon: {to_ratios(poly['ring'], width, height, 4)}")
        lines.append(f"    flow: [{poly['flow'][0]:.4f}, {poly['flow'][1]:.4f}]")
        lines.append("    allowed: []")
    return "\n".join(lines) + "\n"


def main():
    global _divider, _points
    p = argparse.ArgumentParser(
        description="Click lane polygons + divider, save a cameras yaml block")
    p.add_argument("--source", default="samples/input.mp4")
    p.add_argument("--frame", type=int, default=60)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    img = _read_frame(args.source, args.frame)
    height, width = img.shape[:2]
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, _on_mouse)
    print(f"{width}x{height}. Click corners of a road, ENTER to name it. "
          f"D = the 2 clicked points are the divider. S = save, Q = quit.")
    while True:
        cv2.imshow(WIN, _preview(img))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print("quit without saving")
            cv2.destroyAllWindows()
            return
        if key == ord("u") and _points:
            _points.pop()
        elif key in (ord("d"), ord("D")):
            if len(_points) != 2:
                print(f"The divider is exactly 2 points; you have {len(_points)}.")
                continue
            _divider = Line(_points[0], _points[1])
            print(f"  divider set: {_points[0]} -> {_points[1]}")
            _points = []
        elif key == 13:
            _finish_road(img.shape)
        elif key in (ord("s"), ord("S")):
            if len(_points) >= 3:
                _finish_road(img.shape)
            break
    cv2.destroyAllWindows()
    if not _polys:
        print("No roads clicked. Nothing saved.")
        return
    snippet = _snippet(width, height)
    print("--- paste into cameras/<name>.yaml ---\n" + snippet)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Lane polygons from tools/draw_lanes.py - confirm once "
                    "per CCTV.\n"
                    "# Corners are stored in perimeter order; the flow vector "
                    "is what the\n# wrong-way check compares each vehicle's "
                    "heading against.\n" + snippet)
        print(f"Saved to {args.out}. Run: python main.py --camera "
              f"{os.path.splitext(os.path.basename(args.out))[0]}")
    if _divider is None:
        print("No divider clicked. Lane membership still works; press D with "
              "2 points next time to also get the coming/going boundary drawn "
              "and used for side checks.")


if __name__ == "__main__":
    main()
