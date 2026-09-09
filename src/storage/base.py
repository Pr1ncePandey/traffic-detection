"""Storage contract. SQLite implements it now; Postgres can later.

Two rules the hot loop depends on:

1. ID ALLOCATION IS SYNCHRONOUS, ROW WRITES ARE NOT. next_frame_id() /
   next_object_id() return immediately from a client-side counter, so a
   detection row can reference its frame without waiting for the database.
   Using AUTOINCREMENT would force a round-trip per frame just to learn the
   id, which is exactly the stall we are avoiding.
2. put_*() MUST NOT BLOCK. Implementations queue and let a writer thread
   batch. Durability is therefore "at the next commit boundary", not "on
   return" - see SqliteStore.COMMIT_INTERVAL for the window.
"""

from typing import Protocol


class StorageBackend(Protocol):
    def start_run(self, meta: dict) -> int: ...

    def next_frame_id(self) -> int: ...

    def next_object_id(self) -> int: ...

    def put_frame(self, row: dict) -> None: ...

    def put_detection(self, row: dict) -> None: ...

    def upsert_object(self, row: dict) -> None: ...

    def put_attribute(self, object_id: int, key: str, value: str,
                      conf: float = 0.0, ts: float = 0.0) -> None: ...

    def put_event(self, row: dict) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...
