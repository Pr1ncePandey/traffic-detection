"""Enrolled embeddings -> who a face most looks like; reference photo helpers.

A person may have several photos. A face's similarity to a person is that
person's BEST photo, so a side-view photo helps side views without dragging
down frontal ones. Cost is one matrix multiply against every enrolled photo:
1 or 1,000 people cost about the same - what costs CPU is faces in the video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .align import align_face
from .embedder import EMBEDDING_DIM

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
_TRAILING_NUMBER = re.compile(r"[\s_\-]+\d+$")


@dataclass
class Person:
    name: str
    embeddings: np.ndarray          # (usable photos, 512), unit rows


@dataclass
class Gallery:
    people: list[Person] = field(default_factory=list)

    def __post_init__(self):
        rows, owners = [], []
        for index, person in enumerate(self.people):
            rows.append(np.asarray(person.embeddings, np.float32).reshape(-1, EMBEDDING_DIM))
            owners += [index] * len(rows[-1])
        self._matrix = (np.vstack(rows) if rows
                        else np.zeros((0, EMBEDDING_DIM), np.float32))
        self._owners = np.asarray(owners, dtype=np.int64)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.people]

    def __len__(self) -> int:
        return len(self.people)

    def match(self, embeddings: np.ndarray):
        """Per embedding: (best person index, its similarity, runner-up similarity).

        With one person enrolled the runner-up is -1, so the margin rule never
        blocks a match. An empty gallery returns index -1 for every face.
        """
        embeddings = np.asarray(embeddings, np.float32).reshape(-1, EMBEDDING_DIM)
        m = len(embeddings)
        if not self.people or m == 0:
            return (np.full(m, -1, np.int64), np.full(m, -1.0, np.float32),
                    np.full(m, -1.0, np.float32))
        sims = embeddings @ self._matrix.T
        per_person = np.full((m, len(self.people)), -1.0, np.float32)
        for index in range(len(self.people)):
            per_person[:, index] = sims[:, self._owners == index].max(axis=1)
        order = np.argsort(-per_person, axis=1)
        rows = np.arange(m)
        best = order[:, 0]
        runner_up = (per_person[rows, order[:, 1]] if len(self.people) > 1
                     else np.full(m, -1.0, np.float32))
        return best, per_person[rows, best], runner_up


def person_name_for(path: Path, people_dir: Path) -> str:
    """people/Rahul/front.jpg -> Rahul;  people/Priya_2.jpg -> Priya."""
    relative = path.relative_to(people_dir)
    if len(relative.parts) > 1:
        return relative.parts[0].strip()
    stem = path.stem.strip()
    return _TRAILING_NUMBER.sub("", stem) or stem


def discover_photos(people_dir: Path) -> dict[str, list[Path]]:
    """name -> photo paths, for bulk import from a folder.

    Layouts (mixable): people/<Name>/<any>.jpg, or people/<Name>.jpg with an
    optional trailing _2 / -2 / " 2". Hidden files, non-images and deeper
    nesting are skipped. Spellings differing only in case are one person; the
    label is chosen deterministically (folder name first, then the spelling
    most photos use, then alphabetical) because filesystem order is not.
    """
    people_dir = Path(people_dir)
    if not people_dir.is_dir():
        return {}
    photos: dict[str, list[Path]] = {}
    spellings: dict[str, dict[str, list]] = {}
    for path in sorted(people_dir.rglob("*"), key=lambda p: str(p)):
        relative = path.relative_to(people_dir)
        if (not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS
                or any(part.startswith(".") for part in relative.parts)
                or len(relative.parts) > 2):
            continue
        name = person_name_for(path, people_dir)
        key = name.casefold()
        photos.setdefault(key, []).append(path)
        entry = spellings.setdefault(key, {}).setdefault(name, [False, 0])
        entry[0] = entry[0] or len(relative.parts) == 2
        entry[1] += 1
    result = {}
    for key, variants in spellings.items():
        label = sorted(variants, key=lambda s: (not variants[s][0], -variants[s][1], s))[0]
        result[label] = photos[key]
    return result


def read_image(path: Path) -> np.ndarray | None:
    """BGR image. Handles non-ASCII Windows paths; HEIC if pillow-heif is installed."""
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            from PIL import Image, ImageOps
            from pillow_heif import register_heif_opener
        except ImportError:
            return None
        register_heif_opener()
        with Image.open(path) as img:
            return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(img).convert("RGB")),
                                cv2.COLOR_RGB2BGR)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def photo_crop(image: np.ndarray, detector) -> tuple[np.ndarray | None, str]:
    """The aligned face of a reference photo, or (None, why not).

    The largest face is taken to be the person. A second, smaller face is
    reported rather than silently ignored, because a group photo enrols
    whoever happens to be biggest.
    """
    faces = detector.detect_in_photo(image)
    if not faces:
        return None, "no face found - use a clear, front-facing photo"
    note = f"{len(faces)} faces found, used the largest" if len(faces) > 1 else ""
    crop = align_face(image, max(faces, key=lambda f: f.area).landmarks)
    if crop is None:
        return None, "face found but could not be aligned"
    return crop, note
