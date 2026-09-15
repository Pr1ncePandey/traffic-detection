"""The people database: who face recognition looks for.

Two tables in the SAME SQLite file as everything else (storage.path), so there
is one database to back up and the API can join a match to its object:

    people          one row per person: name (unique, case-insensitive),
                    enabled flag
    person_photos   one row per reference photo: its path, plus the AdaFace
                    embedding cached from it

WHY THE EMBEDDING IS STORED. AdaFace takes ~1 s per photo on a laptop CPU. It
is computed once, stored with the model name, file size and modification
time, and recomputed only when the photo file or the model changes - so a
camera starting up does not re-process every photo.

WHY NOT THROUGH SqliteStore's WRITER THREAD. That queue exists for the
high-volume, per-frame rows. Enrolling a person is rare, tiny, and the caller
(a CLI, an API request) wants the error now, so this uses its own short-lived
connection; WAL mode and busy_timeout (set by the store) handle the overlap.

The tables are created here, on first use, rather than in the store's SCHEMA:
a database that never used face recognition never grows them.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from .embedder import EMBEDDING_DIM, MODEL_NAME
from .errors import PeopleError
from .gallery import IMAGE_EXTENSIONS, Gallery, Person, photo_crop, read_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = """
CREATE TABLE IF NOT EXISTS people(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  enabled INTEGER NOT NULL DEFAULT 1,
  added_at REAL, updated_at REAL);

-- vec is a little-endian float32 unit vector (dim values), valid only for
-- `model` and for the file as it was (file_size, file_mtime_ns). error holds
-- why a photo could not be used ("no face found"), so it is not retried on
-- every camera start while the file is unchanged.
CREATE TABLE IF NOT EXISTS person_photos(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id INTEGER NOT NULL,
  path TEXT NOT NULL,
  added_at REAL,
  model TEXT, dim INTEGER, vec BLOB,
  file_size INTEGER, file_mtime_ns INTEGER,
  embedded_at REAL, error TEXT,
  UNIQUE(person_id, path));
CREATE INDEX IF NOT EXISTS idx_person_photos_person ON person_photos(person_id);
"""


def clean_name(name) -> str:
    name = " ".join(str(name or "").split())
    if not name:
        raise PeopleError("a person needs a name")
    if len(name) > 100:
        raise PeopleError("name is too long (100 characters at most)")
    return name


def stored_path(photo) -> str:
    """Absolute path of an existing image, written relative to the project when inside it.

    Relative-to-project keeps the database valid if the whole project folder
    is moved or copied to another machine (the brother's laptop).
    """
    path = Path(photo).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise PeopleError(f"photo not found: {photo}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise PeopleError(f"not a supported image type: {photo} "
                          f"(use {', '.join(sorted(IMAGE_EXTENSIONS))})")
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(stored: str) -> Path:
    path = Path(stored)
    return path if path.is_absolute() else PROJECT_ROOT / path


class PeopleDB:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    # --- editing ---------------------------------------------------------
    def add(self, name, photos) -> dict:
        """Add a person (or more photos to an existing one). Returns the person."""
        name = clean_name(name)
        paths = [stored_path(p) for p in (photos or [])]
        if not paths:
            raise PeopleError(f"{name}: give at least one photo")
        now = time.time()
        conn = self._connect()
        try:
            with conn:
                row = conn.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()
                if row is None:
                    person_id = conn.execute(
                        "INSERT INTO people(name, enabled, added_at, updated_at) VALUES(?,1,?,?)",
                        (name, now, now)).lastrowid
                else:
                    person_id = row["id"]
                    conn.execute("UPDATE people SET updated_at=? WHERE id=?", (now, person_id))
                for path in paths:
                    conn.execute("INSERT OR IGNORE INTO person_photos(person_id, path, added_at)"
                                 " VALUES(?,?,?)", (person_id, path, now))
        finally:
            conn.close()
        return self.get(name)

    def remove(self, name) -> bool:
        name = clean_name(name)
        conn = self._connect()
        try:
            with conn:
                row = conn.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()
                if row is None:
                    return False
                conn.execute("DELETE FROM person_photos WHERE person_id=?", (row["id"],))
                conn.execute("DELETE FROM people WHERE id=?", (row["id"],))
        finally:
            conn.close()
        return True

    def set_enabled(self, name, enabled: bool) -> dict:
        name = clean_name(name)
        conn = self._connect()
        try:
            with conn:
                changed = conn.execute("UPDATE people SET enabled=?, updated_at=? WHERE name=?",
                                       (1 if enabled else 0, time.time(), name)).rowcount
        finally:
            conn.close()
        if not changed:
            raise PeopleError(f"no person named {name!r}")
        return self.get(name)

    # --- reading ---------------------------------------------------------
    def list(self) -> list[dict]:
        conn = self._connect()
        try:
            people = conn.execute("SELECT * FROM people ORDER BY name").fetchall()
            photos = conn.execute("SELECT id, person_id, path, model, error, embedded_at"
                                  " FROM person_photos ORDER BY id").fetchall()
        finally:
            conn.close()
        by_person: dict[int, list] = {}
        for ph in photos:
            by_person.setdefault(ph["person_id"], []).append({
                "id": ph["id"], "path": ph["path"],
                "exists": resolve_path(ph["path"]).is_file(),
                "embedded": ph["model"] == MODEL_NAME and not ph["error"],
                "error": ph["error"]})
        return [{"id": p["id"], "name": p["name"], "enabled": bool(p["enabled"]),
                 "added_at": p["added_at"], "photos": by_person.get(p["id"], [])}
                for p in people]

    def get(self, name) -> dict | None:
        key = clean_name(name).casefold()
        return next((p for p in self.list() if p["name"].casefold() == key), None)

    def signature(self) -> tuple:
        """Changes whenever the set of people or photos to search for changes.

        Cheap enough to poll every few seconds from a running camera, which is
        how a person added while cameras run gets picked up without a restart.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(ph.id), COALESCE(SUM(ph.id), 0), COALESCE(MAX(p.updated_at), 0),"
                " (SELECT COUNT(*) FROM people), (SELECT COALESCE(SUM(enabled), 0) FROM people)"
                " FROM people p JOIN person_photos ph ON ph.person_id = p.id").fetchone()
        finally:
            conn.close()
        return tuple(row)

    # --- embeddings ------------------------------------------------------
    def load_gallery(self, detector, embedder, only=None) -> tuple[Gallery, list[str]]:
        """The enabled people as a Gallery, embedding any photo that needs it.

        `only`: optional list of names (a camera searching for a subset).
        Returns human-readable warnings about photos that could not be used;
        a bad photo never stops the others from loading.
        """
        wanted = {clean_name(n).casefold() for n in (only or [])}
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT p.name, ph.* FROM people p JOIN person_photos ph ON ph.person_id = p.id"
                " WHERE p.enabled = 1 ORDER BY p.name, ph.id").fetchall()
        finally:
            conn.close()
        warnings: list[str] = []
        vectors: dict[str, list[np.ndarray]] = {}
        names: dict[str, str] = {}
        updates = []
        for row in rows:
            name = row["name"]
            if wanted and name.casefold() not in wanted:
                continue
            names.setdefault(name.casefold(), name)
            path = resolve_path(row["path"])
            try:
                stat = path.stat()
            except OSError:
                warnings.append(f"{name}: photo missing: {row['path']}")
                continue
            fresh = (row["model"] == MODEL_NAME and row["file_size"] == stat.st_size
                     and row["file_mtime_ns"] == stat.st_mtime_ns)
            if fresh and row["error"]:
                warnings.append(f"{name}: {row['path']}: {row['error']}")
                continue
            if fresh and row["vec"] is not None and row["dim"] == EMBEDDING_DIM:
                vectors.setdefault(name.casefold(), []).append(
                    np.frombuffer(row["vec"], dtype="<f4").astype(np.float32))
                continue
            vec, error = self._embed_photo(path, detector, embedder)
            if error.startswith("!"):
                error = error[1:]
                warnings.append(f"{name}: {row['path']}: {error}")
            elif error:
                warnings.append(f"{name}: {row['path']}: {error}")
                error = ""
            updates.append((MODEL_NAME, EMBEDDING_DIM,
                            None if vec is None else vec.astype("<f4").tobytes(),
                            stat.st_size, stat.st_mtime_ns, time.time(), error or None, row["id"]))
            if vec is not None:
                vectors.setdefault(name.casefold(), []).append(vec)
        if updates:
            conn = self._connect()
            try:
                with conn:
                    conn.executemany(
                        "UPDATE person_photos SET model=?, dim=?, vec=?, file_size=?,"
                        " file_mtime_ns=?, embedded_at=?, error=? WHERE id=?", updates)
            finally:
                conn.close()
        for key, name in names.items():
            if key not in vectors:
                warnings.append(f"{name}: no usable photo - this person will not be searched for")
        people = [Person(names[key], np.vstack(vecs)) for key, vecs in vectors.items()]
        return Gallery(people), warnings

    @staticmethod
    def _embed_photo(path: Path, detector, embedder) -> tuple[np.ndarray | None, str]:
        """(vector, note). A note starting with '!' is a failure to store."""
        image = read_image(path)
        if image is None:
            hint = (" (HEIC needs: pip install pillow-heif, or convert it to JPG)"
                    if path.suffix.lower() in {".heic", ".heif"} else "")
            return None, f"!cannot read this image{hint}"
        crop, note = photo_crop(image, detector)
        if crop is None:
            return None, "!" + note
        return embedder.embed([crop])[0], note
