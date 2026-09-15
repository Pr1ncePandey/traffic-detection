"""Self-test for face recognition. No models, no video, a few seconds.

    python tools/test_faces.py

Every failure here is a silent one in production: a wrong name on a person, a
stranger named, one person alerted twice, a webhook without the person's name,
a photo that quietly never gets searched for. Fake detector/embedder objects
drive the real enricher, voting, people database and incident policy.
The real models and real footage are exercised by running
`python main.py --camera face_demo` (docs/face-recognition.md).
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attributes.enrichers.face import FaceEnricher, _EmbedWorker  # noqa: E402
from src.attributes.registry import enabled_names  # noqa: E402
from src.config import load_for_camera  # noqa: E402
from src.faces.detector import Face  # noqa: E402
from src.faces.errors import PeopleError  # noqa: E402
from src.faces.gallery import Gallery, Person, discover_photos, person_name_for  # noqa: E402
from src.faces.people import PROJECT_ROOT, PeopleDB  # noqa: E402
from src.faces.voting import Read, decide  # noqa: E402
from src.incidents import NEVER, IncidentPolicy  # noqa: E402
from src.pipeline import _label  # noqa: E402
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


def unit(seed, dim=512):
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def toward(base, similarity, seed):
    """A unit vector whose cosine similarity to `base` is exactly `similarity`."""
    other = unit(seed)
    other = other - (other @ base) * base
    other /= np.linalg.norm(other)
    return (similarity * base + np.sqrt(1 - similarity ** 2) * other).astype(np.float32)


RAHUL, PRIYA = unit(1), unit(2)

# --- fakes ----------------------------------------------------------------


def face_at(x, y, eye=40, score=0.95):
    """A face whose eyes are `eye` px apart, top-left at (x, y)."""
    lm = np.array([[x + 10, y + 20], [x + 10 + eye, y + 20], [x + 10 + eye / 2, y + 35],
                   [x + 15, y + 50], [x + 5 + eye, y + 50]], np.float32)
    return Face((x, y, x + eye + 20, y + 70), lm, score)


class FakeDetector:
    def __init__(self):
        self.faces = []
        self.calls = 0

    def detect(self, frame, width=None):
        self.calls += 1
        return list(self.faces)

    def detect_in_photo(self, image):
        return [face_at(10, 10)] if image.mean() > 10 else []


class FakeEmbedder:
    """Returns queued vectors in order; each crop consumes one."""

    def __init__(self):
        self.vectors = []
        self.calls = 0

    def embed(self, crops):
        self.calls += len(crops)
        out = [self.vectors.pop(0) if self.vectors else unit(999) for _ in crops]
        return np.stack(out) if out else np.zeros((0, 512), np.float32)


class FakePeople:
    def __init__(self, gallery):
        self.gallery = gallery
        self.sig = 1

    def signature(self):
        return (self.sig,)

    def load_gallery(self, detector, embedder, only=None):
        return self.gallery, []


def enricher(gallery, **overrides):
    conf = {"enabled": True, "detect_every": 1, "min_eye_px": 28, "min_sharpness": 0,
            "read_gap_s": 0.3, "recheck_s": 3.0, "min_votes": 2, "max_reads": 6,
            "snapshot_dir": ""}
    conf.update(overrides)
    e = FaceEnricher({"perception": {"attributes": {"face": conf}}})
    e.detector, e.embedder, e.people_db = FakeDetector(), FakeEmbedder(), FakePeople(gallery)
    e.reload(force=True)
    return e


FRAME = np.zeros((720, 1280, 3), np.uint8)


def person(tid, x1, y1, x2, y2):
    return TrackedBox(tid, "person", "person", 0.9, (x1, y1, x2, y2))


def frame(e, ts, boxes, faces, vectors=()):
    e.detector.faces = faces
    e.embedder.vectors.extend(vectors)
    view = AnalysisView(frame_no=int(ts * 10), timestamp=ts, boxes=tuple(boxes),
                        width=1280, height=720, raw=FRAME)
    return e.compute(view)


def kinds(found):
    return [k for k, _d, _t in found.events]


two = Gallery([Person("Rahul", RAHUL[None]), Person("Priya", PRIYA[None])])

# --- voting ---------------------------------------------------------------
print("\nvoting")


def reads(*items):
    return [Read(i * 0.5, n, s, r) for i, (n, s, r) in enumerate(items)]


d = decide(reads(("Rahul", 0.62, 0.10), ("Rahul", 0.58, 0.12), ("Rahul", 0.30, 0.1)), 0.45, 0.05, 2)
check("two clear matching reads name the track", d.name == "Rahul" and d.votes == 2, d)
check("score is the mean of the winning reads", abs(d.score - 0.60) < 1e-6, d.score)
d = decide(reads(("Rahul", 0.62, 0.10)), 0.45, 0.05, 2)
check("one read is not enough with min_votes 2", d.name is None and "needs 2" in d.reason, d)
d = decide(reads(("Rahul", 0.40, 0.1), ("Rahul", 0.42, 0.1)), 0.45, 0.05, 2)
check("below threshold stays unknown, saying how close", d.name is None and "0.42" in d.reason, d)
d = decide(reads(("Rahul", 0.60, 0.58), ("Rahul", 0.61, 0.59)), 0.45, 0.05, 2)
check("too close between two people stays unknown", d.name is None and "not clearly ahead" in d.reason, d)
d = decide(reads(("Rahul", 0.6, 0.1), ("Priya", 0.61, 0.1), ("Rahul", 0.62, 0.1), ("Priya", 0.63, 0.1)),
           0.45, 0.05, 2)
check("a tie between two people stays unknown", d.name is None and "disagree" in d.reason, d)
check("no reads = unknown, saying so", "never clear" in decide([], 0.45, 0.05, 2).reason)
check("nobody enrolled = unknown", decide(reads((None, -1, -1)), 0.45, 0.05, 1).reason == "nobody enrolled")

# --- gallery --------------------------------------------------------------
print("\ngallery")
g = Gallery([Person("Rahul", np.stack([toward(RAHUL, 0.95, 3), toward(RAHUL, 0.9, 4)])),
             Person("Priya", PRIYA[None])])
best, sim, second = g.match(np.stack([toward(RAHUL, 0.9, 6), toward(PRIYA, 0.9, 7)]))
check("each face goes to the right person", best.tolist() == [0, 1], best)
check("a person's score is their best photo", sim[0] > 0.8 and second.max() < 0.3, (sim, second))
check("one person: runner-up is -1 so margin never blocks",
      Gallery([Person("Rahul", RAHUL[None])]).match(RAHUL[None])[2][0] == -1.0)
check("empty gallery matches nobody", Gallery().match(RAHUL[None])[0][0] == -1 and len(Gallery()) == 0)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for rel in ["Rahul/front.jpg", "Rahul/side.PNG", "Priya.jpg", "Priya_2.jpg", "priya-3.jpeg",
                "Amit Kumar 1.webp", "notes.txt", ".hidden.jpg", "deep/nested/too.jpg", "7.jpg"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")
    found = discover_photos(root)
    check("folder import: a folder per person", len(found.get("Rahul", [])) == 2, found)
    check("folder import: Name.jpg, numbers ignored, case merged", len(found.get("Priya", [])) == 3, found)
    check("folder import: spaces survive, junk skipped",
          "Amit Kumar" in found and not {"notes", ".hidden", "too"} & set(found), found)
    check("person_name_for strips a trailing -number", person_name_for(root / "priya-3.jpeg", root) == "priya")

# --- people database ------------------------------------------------------
print("\npeople database")
work = Path(tempfile.mkdtemp(dir=PROJECT_ROOT / "outputs" if (PROJECT_ROOT / "outputs").is_dir() else None))
try:
    bright = np.full((200, 200, 3), 200, np.uint8)
    dark = np.zeros((200, 200, 3), np.uint8)
    ok_photo, blank_photo = work / "rahul.jpg", work / "blank.jpg"
    cv2.imwrite(str(ok_photo), bright)
    cv2.imwrite(str(blank_photo), dark)
    db = PeopleDB(str(work / "t.db"))
    s0 = db.signature()
    person_row = db.add("  Rahul  ", [str(ok_photo)])
    check("add stores a person with a cleaned name", person_row["name"] == "Rahul" and len(person_row["photos"]) == 1)
    check("a photo inside the project is stored relative to it",
          not Path(person_row["photos"][0]["path"]).is_absolute(), person_row["photos"][0]["path"])
    db.add("rahul", [str(ok_photo)])
    check("names are case-insensitive and a photo is not added twice",
          len(db.list()) == 1 and len(db.get("RAHUL")["photos"]) == 1, db.list())
    s1 = db.signature()
    check("signature changes when a person is added", s1 != s0)
    for bad, text in ((lambda: db.add("X", [str(work / "nope.jpg")]), "not found"),
                      (lambda: db.add("", [str(ok_photo)]), "name"),
                      (lambda: db.add("X", []), "at least one"),
                      (lambda: db.add("X", [str(work / "t.db")]), "image type")):
        try:
            bad()
            check(f"rejected: {text}", False, "no error")
        except PeopleError as exc:
            check(f"rejected with a clear message: {text}", text in str(exc), exc)

    det, emb = FakeDetector(), FakeEmbedder()
    emb.vectors = [RAHUL]
    gal, warnings = db.load_gallery(det, emb)
    check("a new photo is embedded on load", gal.names == ["Rahul"] and emb.calls == 1, (gal.names, emb.calls))
    gal, warnings = db.load_gallery(det, emb)
    check("...and read from the database next time, not re-embedded", emb.calls == 1 and len(gal) == 1)
    check("the stored vector is the embedded one", float(gal.match(RAHUL[None])[1][0]) > 0.999)
    time.sleep(0.02)
    cv2.imwrite(str(ok_photo), np.full((210, 200, 3), 190, np.uint8))
    emb.vectors = [RAHUL]
    db.load_gallery(det, emb)
    check("a changed photo file is re-embedded", emb.calls == 2, emb.calls)

    db.add("Blank", [str(blank_photo)])
    gal, warnings = db.load_gallery(det, emb)
    check("a photo with no face is reported, not fatal",
          gal.names == ["Rahul"] and any("no face" in w for w in warnings), warnings)
    calls = emb.calls
    gal, warnings = db.load_gallery(det, emb)
    check("...and not retried while the file is unchanged",
          emb.calls == calls and any("no face" in w for w in warnings))
    check("list says why a photo is unusable",
          "no face" in (db.get("Blank")["photos"][0]["error"] or ""), db.get("Blank"))
    check("a camera can search for a subset", db.load_gallery(det, emb, only=["blank"])[0].names == [])
    db.set_enabled("Rahul", False)
    check("a disabled person is not searched for", db.load_gallery(det, emb)[0].names == [])
    check("disabling changes the signature", db.signature() != s1)
    db.set_enabled("Rahul", True)
    ok_photo.rename(work / "moved.jpg")
    gal, warnings = db.load_gallery(det, emb)
    check("a missing photo file is reported", any("missing" in w for w in warnings) and len(gal) == 0, warnings)
    check("remove deletes the person", db.remove("rahul") and db.get("Rahul") is None)
    check("removing an unknown name says so", db.remove("nobody") is False)
finally:
    shutil.rmtree(work, ignore_errors=True)

# --- enricher: faces on person tracks -------------------------------------
print("\nenricher")
e = enricher(two)
body = person(1, 100, 100, 300, 600)
f1 = frame(e, 0.0, [body], [face_at(160, 120)], [toward(RAHUL, 0.62, 10)])
check("one read is not a match yet", "face_match" not in kinds(f1) and not f1.per_track, f1.per_track)
f2 = frame(e, 0.1, [body], [face_at(160, 120)])
check("no second read before read_gap_s", e.embedder.calls == 1, e.embedder.calls)
f3 = frame(e, 0.4, [body], [face_at(160, 120)], [toward(RAHUL, 0.58, 11)])
check("second agreeing read confirms: one face_match event", kinds(f3) == ["face_match"], f3.events)
detail = f3.events[0][1] if f3.events else {}
check("the event carries name, score and votes",
      detail.get("person") == "Rahul" and detail.get("votes") == 2 and abs(detail.get("score") - 0.6) < 1e-3,
      detail)
check("the name goes on the track", f3.per_track.get(1, {}).get("extra", {}).get("face_name") == "Rahul")
calls = e.detector.calls
f4 = frame(e, 0.8, [body], [face_at(160, 120)])
check("a named person is not re-announced", "face_match" not in kinds(f4))
check("...keeps the name on later frames", f4.per_track.get(1, {}).get("extra", {}).get("face_name") == "Rahul")
check("...and costs no face detection until the recheck", e.detector.calls == calls, e.detector.calls)

e = enricher(two)
stranger = person(2, 100, 100, 300, 600)
for i in range(10):
    f = frame(e, i * 0.4, [stranger], [face_at(160, 120)], [toward(RAHUL, 0.2, 20 + i)])
    check_events = kinds(f)
    if check_events:
        break
check("a stranger (0.2) is never named", not check_events and e._tracks[2].name is None)
check("...reads stop at max_reads", e.embedder.calls == 6, e.embedder.calls)
check("...and the reason says how close", "below" in e._tracks[2].reason, e._tracks[2].reason)

e = enricher(two)
frame(e, 0.0, [person(3, 100, 100, 300, 600)], [face_at(500, 120)])
check("a face outside every person box is not used", e.stats["skipped_no_person_box"] == 1, dict(e.stats))
frame(e, 0.4, [person(3, 100, 100, 300, 600)], [face_at(160, 500)])
check("a 'face' low in the person box (a logo) is not used", e.stats["skipped_no_person_box"] == 2, dict(e.stats))
frame(e, 0.5, [person(3, 100, 100, 300, 600)], [face_at(160, 120, eye=15)])
check("a face too small to recognise is skipped", e.stats["skipped_small"] == 1 and e.embedder.calls == 0)
e = enricher(two, min_sharpness=40)
frame(e, 0.0, [person(3, 100, 100, 300, 600)], [face_at(160, 120)])
check("a blurred face is skipped (flat test frame = blur)", e.stats["skipped_blurred"] == 1 and e.embedder.calls == 0)

e = enricher(two)
big, small = person(4, 100, 50, 700, 700), person(5, 150, 100, 400, 600)
frame(e, 0.0, [big, small], [face_at(200, 130), face_at(560, 100)], [toward(RAHUL, 0.6, 30)])
check("a face inside two person boxes is ambiguous and skipped",
      e.stats["skipped_ambiguous"] == 1 and e._tracks[5].queued == 0, dict(e.stats))
check("...while a face only one box contains is still read", e._tracks[4].queued == 1)

print("\none person, one place")
e = enricher(two)
a, b = person(6, 100, 100, 300, 600), person(7, 700, 100, 900, 600)
fa, fb = face_at(160, 120), face_at(760, 120)
frame(e, 0.0, [a, b], [fa, fb], [toward(RAHUL, 0.55, 40), toward(RAHUL, 0.75, 41)])
f = frame(e, 0.4, [a, b], [fa, fb], [toward(RAHUL, 0.55, 42), toward(RAHUL, 0.75, 43)])
check("two people in one frame cannot both be Rahul", (e._tracks[6].name, e._tracks[7].name) == (None, "Rahul"),
      (e._tracks[6].name, e._tracks[7].name))
check("the better match keeps the name, the other is told why",
      "matched Rahul better" in e._tracks[6].reason, e._tracks[6].reason)
e = enricher(two)
frame(e, 0.0, [a], [fa], [toward(RAHUL, 0.6, 44)])
frame(e, 0.4, [a], [fa], [toward(RAHUL, 0.6, 45)])
e.forget(6)
frame(e, 1.0, [b], [fb], [toward(RAHUL, 0.6, 46)])
f = frame(e, 1.4, [b], [fb], [toward(RAHUL, 0.6, 47)])
check("the same person on a new track after the old one is gone IS named", kinds(f) == ["face_match"])

print("\ntracker swaps two people")
e = enricher(two)
frame(e, 0.0, [a], [fa], [toward(RAHUL, 0.6, 50)])
frame(e, 0.4, [a], [fa], [toward(RAHUL, 0.6, 51)])
f = frame(e, 3.5, [a], [fa], [toward(PRIYA, 0.62, 52)])
check("one read of someone else does not rename", e._tracks[6].name == "Rahul" and not kinds(f))
f = frame(e, 7.0, [a], [fa], [toward(PRIYA, 0.64, 53)])
check("two confident reads of someone else re-decide the track",
      e._tracks[6].name == "Priya" and kinds(f) == ["face_unmatched", "face_match"], (e._tracks[6].name, kinds(f)))
check("the new match says it followed a swap", f.events[-1][1].get("after_swap") is True)
print("\ncrossing triggers quick re-checks")
e = enricher(two)
left, right = person(20, 100, 100, 300, 600), person(21, 700, 100, 900, 600)
fl, fr = face_at(160, 120), face_at(760, 120)
frame(e, 0.0, [left, right], [fl, fr], [toward(RAHUL, 0.6, 90), toward(PRIYA, 0.6, 91)])
frame(e, 0.4, [left, right], [fl, fr], [toward(RAHUL, 0.6, 92), toward(PRIYA, 0.6, 93)])
check("both named before crossing", (e._tracks[20].name, e._tracks[21].name) == ("Rahul", "Priya"))
calls = e.embedder.calls
frame(e, 1.0, [person(20, 380, 100, 580, 600), person(21, 420, 100, 620, 600)], [])
check("overlapping boxes are marked as crossing, no reads while overlapped",
      e._tracks[20].crossing and e._tracks[21].crossing and e.embedder.calls == calls)
# ByteTrack swapped the ids while they overlapped: 20 is now Priya's box, 21 Rahul's.
f = frame(e, 1.2, [left, right], [fl, fr], [toward(PRIYA, 0.62, 94), toward(RAHUL, 0.62, 95)])
check("apart again: both get a re-check at once, not after recheck_s", e.embedder.calls == calls + 2,
      e.embedder.calls - calls)
f = frame(e, 1.6, [left, right], [fl, fr], [toward(PRIYA, 0.63, 96), toward(RAHUL, 0.63, 97)])
check("a swap is corrected within two quick checks (0.6 s, not 3 s)",
      (e._tracks[20].name, e._tracks[21].name) == ("Priya", "Rahul"), (e._tracks[20].name, e._tracks[21].name))
calls = e.embedder.calls
frame(e, 2.0, [left, right], [fl, fr])
check("quick re-checks stop after min_votes; back to recheck_s", e.embedder.calls == calls)

e2 = enricher(two)
frame(e2, 0.0, [a], [fa], [toward(RAHUL, 0.6, 60)])
frame(e2, 0.4, [a], [fa], [toward(RAHUL, 0.6, 61)])
for i, ts in enumerate((3.5, 7.0, 10.5)):
    f = frame(e2, ts, [a], [fa], [toward(RAHUL, 0.1, 62 + i)])
check("three checks that look like nobody enrolled drop the name",
      e2._tracks[6].name is None and "face_unmatched" in kinds(f))

print("\nlifecycle")
e = enricher(two)
frame(e, 0.0, [a], [fa], [toward(RAHUL, 0.6, 70)])
e.forget(6)
check("forget frees a retired track", 6 not in e._tracks)
e._on_vector(6, 0.5, RAHUL, None, {}, type(f)())
check("a late read for a retired track is ignored", 6 not in e._tracks)
e = enricher(Gallery())
frame(e, 0.0, [a], [fa])
check("nobody enrolled: no face detection runs at all", e.detector.calls == 0)
e.people_db.gallery, e.people_db.sig = two, 2
e._next_reload = 0
frame(e, 0.1, [a], [fa], [toward(RAHUL, 0.6, 71)])
check("a person added to the database is picked up without a restart",
      len(e.gallery) == 2 and e.detector.calls == 1)

emb = FakeEmbedder()
emb.vectors = [RAHUL, PRIYA]
worker = _EmbedWorker(emb, max_pending=1, batch=4)
accepted = worker.submit((1, 0.0, np.zeros((112, 112, 3), np.uint8), None))
deadline, got = time.time() + 3, []
while time.time() < deadline and not got:
    got = worker.results()
    time.sleep(0.02)
worker.close()
check("background worker embeds and hands the vector back", accepted and len(got) == 1 and got[0][0][0] == 1)
blocked = _EmbedWorker(FakeEmbedder(), max_pending=1, batch=1)
blocked._stop.set()
blocked._thread.join()
check("a full worker queue refuses instead of stalling the camera",
      blocked.submit((1, 0, None, None)) and not blocked.submit((2, 0, None, None)))

with tempfile.TemporaryDirectory() as tmp:
    e = enricher(two, snapshot_dir=tmp)
    f = frame(e, 0.0, [a], [fa], [toward(RAHUL, 0.6, 80)])
    f = frame(e, 0.4, [a], [fa], [toward(RAHUL, 0.6, 81)])
    path = f.events[0][1].get("snapshot_path") if f.events else None
    check("a confirmed match saves a face snapshot", bool(path) and os.path.exists(path), path)
    summary = e.summary()
    check("summary reports matches and AdaFace cost",
          summary["matches"] == {"Rahul": 1} and summary["adaface_runs"] == 2, summary)

# --- incidents --------------------------------------------------------------
print("\nincidents")


def policy(**overrides):
    inc = {"enabled": True, "base_url": "http://localhost:8000",
           "kinds": {"face_match": True}, "face_match_cooldown_s": 60}
    inc.update(overrides)
    return IncidentPolicy({"incidents": inc}, None, {"door": {"name": "Front door"}})


MATCH = {"person": "Rahul", "score": 0.61, "votes": 2, "reads": 3, "reason": "2 of 3 reads matched",
         "threshold": 0.45, "after_swap": False, "snapshot_path": "outputs/door/faces/Rahul_t4.jpg"}


def match(p, ts, person="Rahul", camera="door", oid=41):
    return p.handle({"kind": "face_match", "track_id": 4, "ts": ts, "detail": {**MATCH, "person": person}},
                    {"camera": camera, "object_id": oid, "timestamp": ts})


p = policy()
out = match(p, 100.0)
check("a face match fires at once (not at retirement)", len(out) == 1, out)
body = out[0]["payload"] if out else {}
check("payload has a person block with name, score, votes",
      body.get("person") == {"name": "Rahul", "score": 0.61, "votes": 2, "reads": 3}, body.get("person"))
check("payload has no vehicle block", "vehicle" not in body)
check("image_url points at /faces, never a local path",
      body.get("image_url") == "http://localhost:8000/faces/41.jpg"
      and "snapshot_path" not in body.get("detail", {}), body)
check("incident id names camera, object and person", body.get("incident_id") == "door-41-rahul-face_match",
      body.get("incident_id"))
check("the reason reaches the consumer", body.get("detail", {}).get("reason") == MATCH["reason"])
check("same person, same camera, inside the cooldown: no second webhook", match(p, 130.0, oid=42) == [])
check("a different person is not held back", len(match(p, 131.0, person="Priya", oid=43)) == 1)
check("the same person on another camera is not held back", len(match(p, 132.0, camera="gate", oid=44)) == 1)
check("after the cooldown the person fires again", len(match(p, 161.0, oid=45)) == 1)
check("clip time restarting (a new run) fires again", len(match(p, 5.0, oid=46)) == 1)
check("cooldown count reported", p.stats()["face_matches_in_cooldown"] == 1, p.stats())
check("face_match can be switched off", match(policy(kinds={"face_match": False}), 1.0) == [])
check("a withdrawn match is never an incident", "face_unmatched" in NEVER)
check("no base_url: image_url is null", match(policy(base_url=""), 1.0)[0]["payload"]["image_url"] is None)

# --- wiring -----------------------------------------------------------------
print("\nwiring")
cfg = load_for_camera(None)
check("face recognition is OFF fleet-wide by default", "face" not in enabled_names(cfg))
check("face_match incidents are on by default", cfg["incidents"]["kinds"].get("face_match") is True)
check("background CLIP embeddings stay OFF in config.yaml",
      cfg["server"]["search"]["embed_in_background"] is False)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    cam = load_for_camera("face_demo")
check("the face_demo camera switches face on", "face" in enabled_names(cam))
check("...with no unknown-key warnings", "unknown key" not in buf.getvalue(), buf.getvalue())
check("snapshot folder is per camera", cam["perception"]["attributes"]["face"]["snapshot_dir"]
      == "outputs/face_demo/faces", cam["perception"]["attributes"]["face"]["snapshot_dir"])
det = Detection(0, "person", "person", 0.91, (0, 0, 10, 10), track_id=3,
                extra={"face_name": "Rahul", "face_name_conf": 0.612})
check("the box caption shows the name and score", "Rahul 0.61" in _label(det), _label(det))

print(f"\n{'=' * 60}\n{_passed} passed, {len(_failed)} failed")
for name in _failed:
    print(f"  FAILED: {name}")
sys.exit(1 if _failed else 0)
