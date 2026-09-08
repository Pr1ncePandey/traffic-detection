"""Storage layer. CSV now; DB interface reserved for brother's Postgres work.

Vehicle rows are queued from the pipeline and flushed by a background thread
(= 'vehicles data stores in parallel' requirement). Frame analytics results
go through the same queue mechanism in pipeline.py.
"""

import os
import queue
import threading

import pandas as pd


class CsvStore:
    def __init__(self, path: str):
        self.path = path
        self.q: "queue.Queue[dict]" = queue.Queue()
        self._rows: list = []
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def put(self, row: dict):
        self.q.put(row)

    def _drain(self):
        while True:
            row = self.q.get()
            if row is None:  # shutdown sentinel
                self.q.task_done()
                break
            self._rows.append(row)
            self.q.task_done()

    def close(self) -> pd.DataFrame:
        self.q.put(None)
        self._t.join()
        df = pd.DataFrame(self._rows)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        df.to_csv(self.path, index=False)
        return df


class DbStoreInterface:
    """What brother's Postgres writer must implement. Same put()/close() shape."""

    def put(self, row: dict):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
