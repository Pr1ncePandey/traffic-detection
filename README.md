# Traffic Tracking Prototype

## Overview
This prototype analyses road traffic video from CCTV or recorded footage. It detects vehicles, assigns each vehicle a unique tracking ID, follows it across frames, counts movement across a reference line, and produces a structured output with video, data, and a summary report.

The system answers four basic questions:
- How many vehicles passed through the view, and of which type (car, motorcycle, bus, truck)?
- Where did a specific vehicle go, identified by its tracking ID and time of appearance?
- How many vehicles crossed in each direction, and on which road?
- Is any vehicle going the wrong way on its road?

## How It Works
1. **Detection (YOLOv8n).** Each video frame is analysed to locate vehicles. The model draws a box around each vehicle and reports a vehicle type with a confidence score.
2. **Tracking (ByteTrack).** Detections across consecutive frames are linked. Each vehicle keeps one tracking ID for as long as it remains visible. The ID is unique and is never reused for another vehicle. IDs live only inside one video — across cameras, the number plate is the identity.
3. **Lanes.** Each road is a polygon with an expected direction arrow (one file per camera in `cameras/`). A vehicle inside a lane but moving against its arrow is flagged `wrong_way`; a vehicle of a type not allowed in that lane is flagged `wrong_lane`. Edge-straddlers and single-frame wobbles are ignored (14 px margin + 5-frame confirm).
4. **Counting.** Two reference zones record direction-neutral A->B / B->A crossings (IN/OUT kept as legacy aliases).
5. **Reporting.** Results are saved as an annotated video, a frame-level data file, vehicle images, and a summary report.

## Key Terms
- **Object ID.** A unique number assigned to each vehicle. The same vehicle keeps the same ID in all frames where it is visible. IDs are unique but not always sequential, because some IDs belong to filtered or briefly lost detections.
- **Confidence Score (0 to 1).** A measure of how certain the detection model is. For example, 0.90 means high certainty and 0.30 means low certainty. Only detections above the configured threshold are kept.
- **Counting Line.** Two reference zones on the video. A crossing from entry side to exit side is A->B (legacy OUT), exit to entry is B->A (legacy IN).
- **Lane.** A road polygon with an expected direction. `lane_id` says which road the vehicle is on; `lane_flag` is `ok`, `wrong_way` (against the arrow — safety), or `wrong_lane` (disallowed type — rule).
- **Number plate.** CPU pipeline (`src/attributes/plate.py`): YOLO plate-box
(`models/lp_detector.pt`, 6MB) + RapidOCR on the raw box, uppercase/alnum
normalize, plate-ish gate (4+ chars, letter+digit, conf >= 0.5). Max 3 tries
per vehicle on the largest crops. Activate with `attributes.enabled: ["type",
"plate"]`. Within one video the ByteTrack ID identifies the vehicle; the plate
is the cross-camera identity (`python query.py --plate <NO> [--fuzzy] --csv
<file>`; fuzzy tolerates 2-char OCR typos).

## Outputs
| File | Description |
|------|-------------|
| `outputs/annotated.mp4` | Video with bounding boxes, tracking IDs, vehicle types, and the counting line. |
| `outputs/tracks.csv` | Frame-level data: frame number, time, object ID, vehicle type, confidence, box coordinates, crossing events, lane_id, lane_flag, plate_number. |
| `outputs/summary.txt` | Text summary: total vehicles, per-class counts, A->B/B->A totals, per-lane counts, wrong-way IDs. |
| `outputs/report.html` | Formatted report with summary tables and one card per vehicle with its image. |
| `outputs/crops/` | One saved image per tracked vehicle, used for review and future number-plate recognition. |
| `samples/input.mp4` | Input video file for analysis. |
| `samples/plate_test.mp4` | Indian traffic clip (27s, 1080p) used to validate plate reading. |
| `models/lp_detector.pt` | YOLOv8n plate-box detector (6MB, MIT). OCR engine: `rapidocr-onnxruntime`. |

The `plate_number` field is reserved in the data for a future number-plate recognition module. It is empty in this prototype.

## Setup and Usage
Requirements: Python 3.10 or above, Windows or Linux.

```
pip install -r requirements.txt
python main.py
python report.py
```

All settings live in `config.yaml` (model, thresholds, camera zones, outputs).
CLI flags override config, e.g. `python main.py --model yolov8s.pt --conf 0.4 --device cuda --no-line`.
Counting is direction-neutral: code counts A->B / B->A only; human labels
("bottom entry" -> "top exit") live in the `camera` block, one file per CCTV
in `cameras/`. Lanes work the same way: each camera yaml holds its road
polygons + arrows. New clip? `python main.py --lanes auto` guesses halves for a
quick test; `python tools/calibrate.py --suggest-lanes --source other.mp4`
proposes real polygons; `python tools/draw_lanes.py` clicks exact corners.
Then `python main.py --camera shop1` — no code change per video.
Code layout: `src/detectors/` (models), `src/trackers/` (ID memory),
`src/attributes/` (color/plate/brand plug-ins), `src/analysis/` (counting, lanes,
speed), `src/storage/` (CSV now, DB interface for Postgres).

- The annotated video is saved to `outputs/annotated.mp4`.
- The report is saved to `outputs/report.html` and can be opened in any browser.
- Vehicle lookup is available with `python query.py --id <ID>` or `python query.py --list`.

Useful options:
```
python main.py --conf 0.4 --max-frames 200
python main.py --lanes auto --max-frames 200   # unknown clip: guess lanes
python main.py --lanes off                     # skip lane logic entirely
```
- `--conf` sets the minimum confidence threshold (default 0.3).
- `--max-frames` limits processing to the first N frames, useful for a quick test.
- `--imgsz 640` sets the analysis resolution. A lower value runs faster on CPU.

## Scope and Next Steps
Scope of this prototype:
- Single recorded video file as input.
- Detection and tracking of four vehicle types: car, motorcycle, bus, truck.
- Two-road lane split with per-road direction arrows and wrong-way flags.
- Movement counting across two reference zones (A->B / B->A).
- Number-plate reading (CPU) with honest limits, measured on
  `samples/plate_test.mp4` (300 frames): 5 vehicles read, 4 of 5 human-readable
  plates matched at 7-8/10 chars (typical errors: single-char confusions C/L,
  4/6, 3/D; exact match rare on small plates). 1 small plate missed, roof-view
  crops correctly empty. Test: `python main.py --source samples/plate_test.mp4
  --lanes off --max-frames 300` with `attributes.enabled: ["type", "plate"]`
  (takes ~4 min CPU for 300 frames; OCR cost is bounded by max 3 tries/vehicle).

Planned extensions:
- Camera GPS registry + journey-on-map across cameras (plate = identity).
- Live CCTV stream input.
