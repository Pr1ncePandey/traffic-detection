# Sample footage

Split by how long a clip runs, because the two testing jobs want opposite
things and one folder cannot serve both.

```
short/   one-off feature testing - fast to iterate on
long/    dashboard and soak testing - long enough to leave running
```

## `short/`

| file | res | fps | duration | scene |
|---|---|---|---|---|
| `indian_road.mp4` | 1920×1080 | 30 | 22.0 s | dense vehicle traffic, close to camera |
| `person_test.mp4` | 1920×1080 | 25 | 63.2 s | pedestrians, rainy street, further back |
| `person_test2.mp4` | 1920×1080 | 23.98 | 45.7 s | second pedestrian scene |
| `input.mp4` | 1280×720 | 25 | 47.8 s | dual carriageway, the original demo clip |

These are what `main.py` and every `tools/*` script default to. A full run of
`indian_road.mp4` is ~78 s on CPU, which is what makes it usable in a loop.

## `long/`

| file | duration | built from |
|---|---|---|
| `indian_road_long.mp4` | 9 m 53 s | `short/indian_road.mp4` × 27 |
| `person_test_long.mp4` | 9 m 29 s | `short/person_test.mp4` × 9 |

Built by lossless concatenation (`ffmpeg -f concat -c copy`), so the pixels are
bit-identical to the source and no re-encode artefacts were introduced. The
scene repeats, which means **track ids and vehicle counts climb across loops
rather than resetting** — that is the point for dashboard testing, but do not
read a per-loop count as a distinct-object count.

`cameras/indian_road.yaml` and `cameras/person_test.yaml` point here, because
the dashboard is what needs the length. For a quick one-off against the same
camera geometry, override the source:

```bash
python main.py --camera indian_road --source samples/short/indian_road.mp4
python main.py --camera person_test --max-frames 400      # or just cap frames
```

## The removed title card

`person_test.mp4` shipped with a "THE CCTV PEOPLE" card at **both** ends — 4.1 s
at the front and 10.1 s at the back. Both are cut here. The front one was
obvious; the back one mattered more, because nine concatenated loops would have
spliced **81 seconds of black title card through the middle** of the long
video, flatlining detection counts in a way that looks like a pipeline bug.

Both cuts land on keyframes (4.08 s and 63.20 s in source time), so the trim is
a stream copy rather than a re-encode.

Note that `docs/vector-search.md` measured its retrieval numbers on the
**untrimmed** 77.4 s version, so re-running that work against this file yields
1,581 frames where the doc says 1,935. The footage is otherwise identical.
