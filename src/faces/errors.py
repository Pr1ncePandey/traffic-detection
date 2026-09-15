"""Exceptions with messages meant for the person running the system.

Everything raised on purpose derives from FaceError, so a command line or an
API route can show it as a plain message and keep real bugs as tracebacks.
"""

from pathlib import Path


class FaceError(Exception):
    """A problem the operator can fix: a missing model, a bad photo, a bad name."""


class DependencyError(FaceError):
    """A required Python package is not installed."""


class ModelNotFound(FaceError):
    def __init__(self, path: Path):
        super().__init__(f"model file not found: {path}\n"
                         f"  fix: python tools/fetch_face_models.py")
        self.path = path


class ModelLoadError(FaceError):
    """A model file exists but could not be loaded (corrupt or incompatible)."""


class PeopleError(FaceError):
    """A people-database request that cannot be done: unknown name, missing photo."""
