"""Keep the search index current, without anyone remembering to.

THE PROBLEM THIS SOLVES

Embeddings are built only by `tools/embed_crops.py` - the pipeline never
builds them, deliberately, because no real-time decision needs one and
keeping CLIP out of pipeline.py makes a model swap a script re-run. The
consequence was a trap: a fresh database means search returns nothing, a
database that has merely run for an hour means search silently covers only
part of the corpus, and neither says so. Search LOOKED like it was working.

So the service embeds its own backlog on a thread, the same way the webhook
dispatcher and the retention reaper own theirs. This is not inference in
pipeline.py - the frame loop is untouched and knows nothing about it.

IT MUST NOT STARVE THE CAMERAS, AND IT CAN

L/14 inference is ~317 ms/crop on CPU here, and the cameras want that CPU:
measured, two 1080p cameras plus three MJPEG viewers already cost ~10% of
achieved fps. So this runs on a DUTY CYCLE - one batch, then a deliberate
pause - rather than as fast as it can. `batch` and `pause_s` are the dial:

    duty = work / (work + pause_s),  work ~= batch * 0.32s for L/14

The defaults (batch 8, pause 3 s) come to roughly 45%, about 1.4 crops/s -
slower than the CLI's 3.1, which is the point. A backlog of 1,500 crops takes
~18 minutes to clear in the background instead of 8 minutes with the cameras
fighting for scraps. Set `pause_s: 0` to let it run flat out, or
`embed_in_background: false` to go back to embedding by hand.

WHAT IT DOES NOT DO

It never re-embeds. `candidates()` skips what the model already has, so this
converges on a complete index and then idles, polling cheaply. It also never
decides WHICH crops belong in an index - those rules live in
src/query/embed.py and are shared with the CLI, so the two cannot drift.

WHY THERE IS A CURSOR, AND A SKIP MEMO

"Not embedded" is not "embeddable". A crop under the size or confidence floor
is never written, so it stays un-embedded forever - and a batched caller that
always asks for the first N un-embedded rows gets the SAME ineligible batch
back every time. Measured before this was fixed: 37 passes over one batch of 8
permanently-skipped crops, zero vectors written, backlog growing. Two things
prevent it:

  cursor      each pass asks for rows with id > the last one it looked at, so
              progress is monotonic within a sweep. At the end of a sweep the
              cursor wraps to 0 and the next sweep re-checks everything.
  skip memo   ids that were considered and not written are remembered in
              memory, so a wrap does not re-read the same JPEGs. It is
              deliberately NOT persisted: eligibility is a function of the
              crop and the config, so a restart or a config change re-examines
              them exactly once, which is the cheap way to stay correct.
"""

import threading
import time

from ..query.embed import (BATCH, MAX_AREA, MIN_CONF, MIN_PX, CropEmbedder,
                           candidates, new_skips)

DEFAULT_MODEL = "clip-vit-l14"
BATCH_SIZE = 8            # crops per pass; one encode batch
PAUSE_S = 3.0             # deliberate idle after each pass - see the docstring
IDLE_POLL_S = 30.0        # how often to look for new crops once caught up
ERROR_BACKOFF_S = 60.0


class BackgroundEmbedder:
    """One thread, embedding whatever the cameras have produced."""

    def __init__(self, store, model: str = DEFAULT_MODEL,
                 enabled: bool = True, batch: int = BATCH_SIZE,
                 pause_s: float = PAUSE_S, idle_poll_s: float = IDLE_POLL_S,
                 min_px: int = MIN_PX, min_conf: float = MIN_CONF,
                 max_area: float = MAX_AREA):
        self.store = store
        self.model = model
        self.enabled = bool(enabled)
        self.batch = max(1, int(batch))
        self.pause_s = max(0.0, float(pause_s))
        self.idle_poll_s = max(1.0, float(idle_poll_s))
        self.filters = {"min_px": min_px, "min_conf": min_conf,
                        "max_area": max_area}
        self.state = "disabled" if not self.enabled else "idle"
        self.embedded = 0
        self.skipped = new_skips()
        self.passes = 0
        self.last_error: str | None = None
        self.last_active_at: float | None = None
        self._enc = None
        self._worker: CropEmbedder | None = None
        self._cursor = 0
        self._skipped_ids: set[int] = set()
        self.sweeps = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, cfg: dict, store) -> "BackgroundEmbedder":
        se = ((cfg.get("server", {}) or {}).get("search", {}) or {})
        return cls(store,
                   model=se.get("model", DEFAULT_MODEL),
                   enabled=se.get("embed_in_background", True),
                   batch=se.get("batch", BATCH_SIZE),
                   pause_s=se.get("pause_s", PAUSE_S),
                   idle_poll_s=se.get("idle_poll_s", IDLE_POLL_S),
                   min_px=se.get("min_px", MIN_PX),
                   min_conf=se.get("min_conf", MIN_CONF),
                   max_area=se.get("max_area", MAX_AREA))

    # --- lifecycle ---------------------------------------------------------
    def start(self):
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop,
                                        name="embedder", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0):
        self._stop.set()
        if self._thread is not None:
            # Interrupts between batches, not mid-encode: one batch is ~2.5 s,
            # so the wait is bounded without needing to kill anything.
            self._thread.join(timeout=timeout)

    # --- the loop ----------------------------------------------------------
    def _ensure_encoder(self) -> bool:
        """Build the encoder on the THREAD, not at startup.

        The L/14 vision tower is 1.16 GB; loading it inline would add that to
        server start-up for a feature nobody may use in this session.
        """
        if self._worker is not None:
            return True
        try:
            from ..query.clip_onnx import ClipOnnx
            enc = ClipOnnx(self.model)
            missing = enc.missing()
            if missing:
                self.state = "unavailable"
                self.last_error = (
                    f"{self.model} not fetched: {missing[0]} missing. Run: "
                    f"python tools/fetch_clip.py --model {self.model}")
                print(f"[embedder] disabled - {self.last_error}")
                return False
            self._enc = enc
            self._worker = CropEmbedder(enc, batch=self.batch, **self.filters)
            return True
        except Exception as e:
            self.state = "failed"
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"[embedder] could not start: {self.last_error}")
            return False

    def _loop(self):
        if not self._ensure_encoder():
            return                      # reason already recorded and printed
        print(f"[embedder] background indexing with {self.model} "
              f"(batch {self.batch}, pause {self.pause_s}s)")
        while not self._stop.is_set():
            try:
                before = self.embedded
                n = self._one_pass()
                wrote = self.embedded > before
            except Exception as e:
                # A bad crop or a transient read must not kill the thread and
                # silently stop the index growing.
                self.state = "error"
                self.last_error = f"{type(e).__name__}: {e}"
                print(f"[embedder] pass failed, backing off: {self.last_error}")
                self._stop.wait(ERROR_BACKOFF_S)
                continue
            if n == 0:
                self.state = "caught up"
                self._stop.wait(self.idle_poll_s)
                continue
            self.state = "embedding"
            self.last_active_at = time.time()
            # THE PAUSE IS PAID FOR BY INFERENCE, NOT BY LOOKING. A pass that
            # embedded nothing only read a few JPEGs and cost the cameras
            # almost nothing, so pausing after it would be pure delay - and
            # the index has a long ineligible prefix to page through before it
            # reaches new crops (~1000 rows here, which at one pause per 8
            # would be six minutes of waiting to do no work).
            if self.pause_s and wrote:
                self._stop.wait(self.pause_s)
        self.state = "stopped"

    def _one_pass(self) -> int:
        """Look at the next batch. Returns rows CONSIDERED, not written.

        Considered rather than written, because a batch of entirely ineligible
        crops writes nothing and must not be mistaken for "caught up" - that
        would send the loop to its long idle poll with a backlog outstanding.
        """
        conn = self.store.connect_ro()
        try:
            # Over-fetch: the memo may exclude most of a window, and a query
            # that returns only memoed rows would waste a whole pass.
            rows = candidates(conn, self.model, limit=self.batch * 8,
                              after_id=self._cursor)
            if not rows:
                # End of the sweep. Wrap so new crops (and any whose
                # eligibility changed) are picked up on the next one.
                if self._cursor:
                    self._cursor = 0
                    self.sweeps += 1
                return 0
            todo = [r for r in rows if r["id"] not in self._skipped_ids][:self.batch]
            if not todo:
                # The whole window is already memoed as ineligible: step over
                # it so the sweep still advances.
                self._cursor = max(r["id"] for r in rows)
                return len(rows)
            # ONLY past what this pass actually looked at. Advancing to the end
            # of the over-fetched window would abandon every row between
            # `todo` and that end - they are neither embedded nor memoed, so
            # nothing would ever come back for them.
            self._cursor = max(r["id"] for r in todo)
            result = self._worker.run(
                conn, todo,
                # The shared store's single writer thread, not a second one -
                # put_embedding enqueues, which is what makes this safe from
                # a thread that is not the writer.
                put=self.store.put_embedding,
                skips=self.skipped,
                should_stop=self._stop.is_set)
            self.passes += 1
            self.embedded += result["embedded"]
            wrote = set(result.get("embedded_ids") or ())
            for r in todo:
                if r["id"] not in wrote:
                    self._skipped_ids.add(r["id"])
            return len(todo)
        finally:
            conn.close()

    # --- reporting ---------------------------------------------------------
    def backlog(self) -> int:
        """Crops with no vector in this model. Upper bound; see queries.py."""
        conn = self.store.connect_ro()
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM objects o"
                " WHERE o.crop_path IS NOT NULL AND o.crop_path != ''"
                "   AND NOT EXISTS (SELECT 1 FROM embeddings e"
                "                    WHERE e.object_id = o.id AND e.model = ?)",
                (self.model,)).fetchone()[0]
        except Exception:
            return 0
        finally:
            conn.close()

    def stats(self) -> dict:
        return {"enabled": self.enabled, "state": self.state,
                "model": self.model, "embedded": self.embedded,
                "passes": self.passes, "sweeps": self.sweeps,
                "ineligible": len(self._skipped_ids),
                "skipped": {k: v for k, v in self.skipped.items() if v},
                "batch": self.batch, "pause_s": self.pause_s,
                "last_error": self.last_error,
                "seconds_since_active": (
                    None if self.last_active_at is None
                    else round(time.time() - self.last_active_at, 1))}
