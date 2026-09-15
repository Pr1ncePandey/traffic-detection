"""Manage the people face recognition looks for (the `people` database tables).

    python tools/people.py list
    python tools/people.py add "Prince" people/me/selfie.jpg [more photos...]
    python tools/people.py import people/          # one folder per person, or Name.jpg
    python tools/people.py disable "Prince"        # keep, but stop searching for them
    python tools/people.py enable "Prince"
    python tools/people.py remove "Prince"
    python tools/people.py enrol                   # compute embeddings now (needs the models)

The database is storage.path from config.yaml (outputs/traffic.db) unless
--db is given - the same file the cameras write, so a running camera with
`face` enabled picks up changes within reload_s seconds.

`enrol` is optional: a camera computes any missing embedding itself when it
starts. Running it first moves that ~1 s per photo out of camera start-up and
shows any photo that has no usable face.

Only enrol people who have agreed to it.

Exit codes: 0 ok, 1 a problem to fix (message printed), 2 bad command line.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load  # noqa: E402
from src.faces.errors import FaceError  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", default=None, help="SQLite path (default: storage.path in config.yaml)")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="everyone in the database")
    add = sub.add_parser("add", help="add a person, or more photos to a person")
    add.add_argument("name")
    add.add_argument("photos", nargs="+")
    imp = sub.add_parser("import", help="add everyone from a folder")
    imp.add_argument("folder")
    for name in ("remove", "enable", "disable"):
        sub.add_parser(name).add_argument("name")
    sub.add_parser("enrol", help="compute missing embeddings now")
    return p.parse_args()


def show(people):
    if not people:
        print("nobody enrolled - add someone: python tools/people.py add NAME PHOTO")
        return
    for person in people:
        state = "" if person["enabled"] else "  (disabled)"
        print(f"{person['name']}{state}")
        for photo in person["photos"]:
            if not photo["exists"]:
                note = "MISSING FILE"
            elif photo["error"]:
                note = f"unusable: {photo['error']}"
            elif photo["embedded"]:
                note = "ready"
            else:
                note = "not embedded yet"
            print(f"    {photo['path']}  [{note}]")


def main() -> int:
    args = parse_args()
    cfg = load(args.config)
    db_path = args.db or (cfg.get("storage") or {}).get("path", "outputs/traffic.db")
    from src.faces.people import PeopleDB
    db = PeopleDB(db_path)

    if args.command == "list":
        show(db.list())
    elif args.command == "add":
        person = db.add(args.name, args.photos)
        print(f"added {person['name']} ({len(person['photos'])} photo(s))")
    elif args.command == "import":
        from src.faces.gallery import discover_photos
        found = discover_photos(Path(args.folder))
        if not found:
            print(f"no photos found in {args.folder} "
                  f"(expected {args.folder}/<Name>/photo.jpg or {args.folder}/<Name>.jpg)")
            return 1
        for name, photos in found.items():
            person = db.add(name, photos)
            print(f"added {person['name']} ({len(person['photos'])} photo(s))")
    elif args.command == "remove":
        if not db.remove(args.name):
            print(f"no person named {args.name!r}")
            return 1
        print(f"removed {args.name}")
    elif args.command in ("enable", "disable"):
        person = db.set_enabled(args.name, args.command == "enable")
        print(f"{person['name']}: {'enabled' if person['enabled'] else 'disabled'}")
    elif args.command == "enrol":
        from src.attributes.registry import block
        from src.faces.detector import FaceDetector
        from src.faces.embedder import FaceEmbedder
        face = block(cfg, "face")
        models = face.get("models_dir", "models/faces")
        detector = FaceDetector(models, int(face.get("detect_width", 1280)),
                                float(face.get("min_det_score", 0.8)))
        embedder = FaceEmbedder(models, int(face.get("threads", 4)))
        gallery, warnings = db.load_gallery(detector, embedder)
        for warning in warnings:
            print(f"warning: {warning}")
        print(f"ready: {len(gallery)} people ({', '.join(gallery.names) or 'nobody'})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FaceError as e:
        print(f"error: {e}")
        sys.exit(1)
