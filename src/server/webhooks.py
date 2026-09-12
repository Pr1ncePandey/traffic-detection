"""The outbox dispatcher: POSTs incidents, retries, dead-letters.

Delivery is separated from policy (`src/incidents.py` decides WHAT fires) so
the firing rules can be tested without a network and this can be tested without
a pipeline.

THREE PROPERTIES THAT MATTER

**It never blocks the frame loop.** Its own thread, draining a table rather
than an in-memory queue. A customer endpoint taking 30 s to answer costs a
delayed delivery and nothing else - no dropped frames, no stalled inference.

**Persistence before delivery.** The incident row is committed by the writer
thread before this loop ever sees it, so a crash between raising and sending
loses nothing: the pending delivery is still there on restart. That is what
makes the outbox an outbox rather than a queue.

**At-least-once, never at-most-once.** A timeout is indistinguishable from a
slow success, so a delivery that may have landed is retried. Consumers must
therefore be idempotent on `incident_id`, which is derived from the sighting
rather than from a clock precisely so it is stable across retries. That
requirement belongs in whatever consumer-facing note ships with this.

BACKOFF AND DEAD-LETTERING

Exponential with a cap, then `status='dead'` after `max_attempts`. Dead is a
state, not a deletion: a webhook endpoint that has been failing for a day
should be visible on the dashboard without anyone reading logs, and the payload
is still in `incidents` to be replayed once the consumer is fixed.
"""

import threading
import time

try:
    import httpx
except ImportError:                      # pragma: no cover
    httpx = None

from ..incidents import serialise, sign

BACKOFF_START_S = 2.0
BACKOFF_MAX_S = 300.0
MAX_ATTEMPTS = 8                         # ~10 min of retries at the cap
POLL_INTERVAL_S = 1.0
BATCH = 20


def backoff_for(attempts: int) -> float:
    """Seconds to wait before attempt N+1. Exponential, capped.

    Capped rather than unbounded so a consumer that comes back after an hour is
    retried within five minutes instead of in another hour.
    """
    return min(BACKOFF_MAX_S, BACKOFF_START_S * (2 ** max(0, attempts - 1)))


class WebhookDispatcher:
    """Drains `deliveries` on its own thread."""

    def __init__(self, store, policy, max_attempts: int = MAX_ATTEMPTS,
                 poll_interval: float = POLL_INTERVAL_S):
        self.store = store
        self.policy = policy
        self.max_attempts = int(max_attempts)
        self.poll_interval = float(poll_interval)
        self.sent = 0
        self.failed = 0
        self.dead = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.policy and self.policy.subscriptions)

    def start(self):
        if not self.enabled:
            return self
        if httpx is None:
            print("[webhooks] httpx not installed; incidents will be recorded "
                  "but never delivered. pip install httpx")
            return self
        self._client = httpx.Client(follow_redirects=False)
        self._thread = threading.Thread(target=self._loop, name="webhooks",
                                        daemon=True)
        self._thread.start()
        print(f"[webhooks] dispatcher running for "
              f"{len(self.policy.subscriptions)} subscription(s)")
        return self

    def _loop(self):
        while not self._stop.wait(self.poll_interval):
            try:
                self.drain_once()
            except Exception as e:
                # The dispatcher outliving its own bugs matters more than any
                # single delivery: a dead dispatcher silently stops every
                # webhook, which is the failure this catch exists to prevent.
                self.last_error = f"dispatcher: {e}"
                print(f"[webhooks] dispatch pass failed: {e}")

    def drain_once(self) -> int:
        """One pass over the due deliveries. Returns how many were attempted."""
        rows = self.store.due_deliveries(limit=BATCH)
        for row in rows:
            if self._stop.is_set():
                break
            self._attempt(row)
        return len(rows)

    def _attempt(self, row: dict) -> bool:
        sub = self.policy.subscription_for(row["endpoint"])
        attempts = int(row["attempts"]) + 1
        if sub is None:
            # The subscription was removed from config while a delivery was
            # pending. Dead-lettered rather than retried forever against an
            # endpoint nobody has asked for any more.
            self.store.mark_delivery(row["id"], "dead", attempts, None,
                                     "no such subscription in config")
            self.dead += 1
            return False

        import json as _json
        try:
            payload = _json.loads(row["payload_json"])
        except Exception as e:
            self.store.mark_delivery(row["id"], "dead", attempts, None,
                                     f"unreadable payload: {e}")
            self.dead += 1
            return False

        body = serialise(payload)
        stamp = str(int(time.time()))
        headers = {"content-type": "application/json",
                   "x-incident-id": str(row["incident_id"]),
                   "x-incident-kind": str(row["kind"] or ""),
                   "x-timestamp": stamp,
                   "user-agent": "traffic-detection/incidents"}
        if sub.secret:
            headers["x-signature"] = sign(body, sub.secret, stamp)

        try:
            resp = self._client.post(sub.endpoint, content=body, headers=headers,
                                     timeout=sub.timeout_s)
            if 200 <= resp.status_code < 300:
                self.store.mark_delivery(row["id"], "sent", attempts, None, None)
                self.sent += 1
                return True
            error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            # 4xx other than 408/429 is the consumer saying "this request is
            # wrong", which retrying cannot fix. Dead-letter it immediately
            # rather than spending eight attempts to be told the same thing.
            permanent = (400 <= resp.status_code < 500
                         and resp.status_code not in (408, 429))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            permanent = False

        self.failed += 1
        self.last_error = error
        if permanent or attempts >= self.max_attempts:
            self.store.mark_delivery(row["id"], "dead", attempts, None, error)
            self.dead += 1
            print(f"[webhooks] DEAD {row['incident_id']} -> {sub.endpoint} "
                  f"after {attempts} attempt(s): {error}")
            return False
        delay = backoff_for(attempts)
        self.store.mark_delivery(row["id"], "pending", attempts,
                                 time.time() + delay, error)
        return False

    def stats(self) -> dict:
        return {"enabled": self.enabled, "sent": self.sent,
                "failed_attempts": self.failed, "dead": self.dead,
                "last_error": self.last_error}

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
