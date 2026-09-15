"""Split a search query into SCOPE (SQL) and SUBJECT (vector).

This is deliberately NOT a SQL-or-vector arbiter. Everything semantic goes to
the vector path, so the router's whole job is to pull out the scoping that
cannot be wrong - which camera, which time window - and hand the rest through
untouched.

    "a person carrying a box on indian_road in the last 2 hours"
      scope   camera=indian_road, t > now-7200
      subject "a person carrying a box"

WHY NO ATTRIBUTE PREDICATES

`a white truck` could become `cls_name='truck' AND color='white'`, which is
exact and has perfect attribute binding. It deliberately does not, because a
SQL attribute predicate has capped RECALL: the colour enricher writes nothing
when no hue wins 40% of band pixels, so `color=` is blind to 22-35% of
vehicles (measured; see docs/vector-search.md Concern 1). Used as a prefilter
it would permanently hide those objects from the vector step.

The trade is explicit: SQL is exact but partially blind; vectors see every
object but bind attributes at p@5 ~0.28. This module takes recall, and pays
for it by WARNING when a query contains a term SQL could have answered
exactly. It advises; it never reroutes.

THE SUBJECT IS NEVER REWRITTEN. Only scope phrases are removed. That is the
one invariant here, and `subject_fidelity()` checks it.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field

# Model kept small on purpose: this is a narrow parse, not a reasoning task.
CLAUDE_MODEL = "haiku"
CLAUDE_TIMEOUT_S = 20.0

_REL = re.compile(
    r"\b(?:in|over|during|within|from)?\s*"
    r"(?:the\s+)?(?:last|past|previous)\s+"
    r"(\d+)?\s*(min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|week|weeks)\b",
    re.I)
_UNIT_S = {"min": 60, "mins": 60, "minute": 60, "minutes": 60,
           "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
           "d": 86400, "day": 86400, "days": 86400,
           "w": 604800, "week": 604800, "weeks": 604800}
_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_TODAY = re.compile(r"\b(today|yesterday)\b", re.I)
# "on <camera>", "camera <x>", "from <camera>" - the preposition is consumed
# with the name so it does not survive into the subject as dangling grammar.
_CAM_PREFIX = re.compile(r"\b(?:on|at|from|camera|cam|in)\s+$", re.I)


@dataclass
class Scope:
    """SQL-side narrowing. Every field is optional, and absent means absent -
    an unstated camera searches all cameras rather than guessing one."""
    camera: str | None = None
    t_from: float | None = None
    t_to: float | None = None
    # Runs whose clock cannot be compared to a wall-clock window. A file run
    # timestamps on clip-seconds, so a "last 2 hours" filter is meaningless
    # against it: report it as excluded rather than silently dropping the rows
    # (the `unorderable` discipline in src/journeys.py).
    excluded_runs: list[int] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (self.camera is None and self.t_from is None
                and self.t_to is None)

    def describe(self) -> str:
        bits = [f"camera={self.camera}" if self.camera else "all cameras"]
        if self.t_from is not None or self.t_to is not None:
            a = _clock(self.t_from) if self.t_from else "-inf"
            b = _clock(self.t_to) if self.t_to else "now"
            bits.append(f"t in [{a}, {b}]")
        else:
            bits.append("all time")
        if self.excluded_runs:
            bits.append(f"excluding {len(self.excluded_runs)} clip-clock run(s)")
        return ", ".join(bits)


@dataclass
class Route:
    scope: Scope
    subject: str                  # user text, verbatim minus scope phrases
    source: str                   # 'claude-cli' | 'deterministic'
    warnings: list[str] = field(default_factory=list)


@dataclass
class Vocabulary:
    """What SQL could express, read from the database at runtime.

    Not used to build predicates (see the module docstring). Used to warn the
    user that an exact answer exists. Because it is read rather than
    hardcoded, adding an enricher sharpens the advisory with no change here.
    """
    cameras: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    attributes: dict[str, list[str]] = field(default_factory=dict)

    def attribute_hits(self, text: str) -> list[tuple[str, list[str]]]:
        """(value, [keys]) for each vocabulary value appearing in `text`.

        Grouped by VALUE, not by key: 'white' is a legal value of `color`,
        `upper_color` and `lower_color`, and naming it once with its three
        keys is the useful form. Listing it three times is noise.
        """
        words = set(re.findall(r"[a-z_]+", text.lower()))
        hits: dict[str, list[str]] = {}
        for key, values in sorted(self.attributes.items()):
            for v in values:
                if v and v.lower() in words:
                    hits.setdefault(v, []).append(key)
        return sorted(hits.items())

    def class_hits(self, text: str) -> list[str]:
        words = set(re.findall(r"[a-z_]+", text.lower()))
        # crude singularisation: 'trucks' -> 'truck'. Enough for a warning.
        words |= {w[:-1] for w in words if w.endswith("s")}
        return [c for c in self.classes if c.lower() in words]


def load_vocabulary(conn) -> Vocabulary:
    def col(sql):
        try:
            return [r[0] for r in conn.execute(sql) if r[0]]
        except Exception:
            return []

    attrs: dict[str, list[str]] = {}
    try:
        for key, value in conn.execute(
                "SELECT DISTINCT key, value FROM attributes"
                " WHERE value IS NOT NULL AND value != ''"):
            attrs.setdefault(key, []).append(value)
    except Exception:
        pass
    # plate_number is a free-text identifier, not a closed vocabulary; warning
    # on it would fire for every plate-shaped token.
    attrs.pop("plate_number", None)
    return Vocabulary(
        cameras=col("SELECT DISTINCT camera FROM runs"),
        classes=col("SELECT DISTINCT cls_name FROM objects"),
        groups=col("SELECT DISTINCT cls_group FROM objects"),
        attributes={k: sorted(set(v)) for k, v in attrs.items()})


def _clock(ts) -> str:
    try:
        return _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _strip(text: str, span: tuple[int, int]) -> str:
    """Remove a span and any preposition immediately before it, then tidy
    whitespace. Nothing else about the text is touched."""
    a, b = span
    head = text[:a]
    m = _CAM_PREFIX.search(head)
    if m:
        head = head[:m.start()]
    return re.sub(r"\s{2,}", " ", (head + " " + text[b:])).strip(" ,")


# --------------------------------------------------------------------------
# deterministic path - works with no binary, no login, no network
# --------------------------------------------------------------------------
def route_deterministic(query: str, vocab: Vocabulary,
                        now: float | None = None) -> Route:
    now = time.time() if now is None else now
    subject, scope, warns = query.strip(), Scope(), []

    for cam in sorted(vocab.cameras, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(cam)}\b", subject, re.I)
        if m:
            scope.camera = cam
            subject = _strip(subject, m.span())
            break

    m = _REL.search(subject)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        scope.t_from = now - n * _UNIT_S[m.group(2).lower()]
        subject = _strip(subject, m.span())
    else:
        m = _TODAY.search(subject)
        if m:
            midnight = _dt.datetime.fromtimestamp(now).replace(
                hour=0, minute=0, second=0, microsecond=0).timestamp()
            if m.group(1).lower() == "today":
                scope.t_from = midnight
            else:
                scope.t_from, scope.t_to = midnight - 86400, midnight
            subject = _strip(subject, m.span())
        else:
            m = _ISO.search(subject)
            if m:
                d = _dt.datetime.strptime(m.group(1), "%Y-%m-%d")
                scope.t_from = d.timestamp()
                scope.t_to = d.timestamp() + 86400
                subject = _strip(subject, m.span())

    if not subject:
        warns.append("query was only scope - no subject left to search for")
    return Route(scope=scope, subject=subject, source="deterministic",
                 warnings=warns)


# --------------------------------------------------------------------------
# claude CLI path
# --------------------------------------------------------------------------
_PROMPT = """Extract search SCOPE from a query over a traffic-camera database.

Cameras that exist (use one of these verbatim, or null): {cameras}
Current unix time: {now}

Return ONLY a JSON object, no prose:
{{"camera": <one of the cameras above, or null>,
  "t_from": <unix seconds, or null>,
  "t_to": <unix seconds, or null>,
  "subject": "<the query with camera and time phrases removed>"}}

Rules:
- Never invent a camera. If none is named, camera is null.
- Never invent a time window. If no time is named, t_from and t_to are null.
- "subject" must be the original text with ONLY the camera and time phrases
  removed. Do not rephrase, translate, correct or summarise it. Do not remove
  colours, object types, or anything describing WHAT to look for.

QUERY: {query}"""


def _claude_cli(prompt: str, model: str = CLAUDE_MODEL,
                timeout: float = CLAUDE_TIMEOUT_S,
                binary: str = "claude") -> tuple[str | None, str | None]:
    """Run one non-interactive claude turn. Returns (text, error).

    A local CLIENT, not local inference: the prompt still goes to Anthropic.
    Only the query text and the camera list are sent - never a crop, an image,
    a plate or a row.

    Two non-obvious requirements, both learned the hard way:

    stdin=DEVNULL - without it the CLI waits for piped input and burns ~3s per
    call logging "no stdin data received in 3s".

    is_error, not returncode - a failed call exits 0 carrying
    {"is_error": true, "result": "Not logged in - Please run /login"}. Checking
    the exit code would feed that sentence to json.loads as a completion.
    """
    exe = shutil.which(binary)
    if not exe:
        return None, f"{binary} not on PATH"
    cmd = [exe, "-p", "--bare", "--restricted", "--no-session-persistence",
           "--output-format", "json", "--model", model, prompt]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout:g}s"
    except OSError as e:
        return None, f"could not run {binary}: {e}"
    try:
        env = json.loads(p.stdout or "{}")
    except json.JSONDecodeError:
        return None, "CLI did not return JSON"
    if env.get("is_error") or env.get("subtype") not in (None, "success"):
        return None, str(env.get("result") or "CLI reported an error").strip()
    return (env.get("result") or "").strip(), None


def _extract_json(text: str):
    """Pull the first JSON object out of a reply that may be fenced."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(),
                  flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth = start = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def route_claude(query: str, vocab: Vocabulary, now: float | None = None,
                 model: str = CLAUDE_MODEL, timeout: float = CLAUDE_TIMEOUT_S,
                 binary: str = "claude") -> Route | None:
    """Scope extraction via the local claude CLI. None on any failure, so the
    caller falls back rather than getting a half-parsed route."""
    now = time.time() if now is None else now
    prompt = _PROMPT.format(cameras=", ".join(vocab.cameras) or "(none)",
                            now=int(now), query=query)
    for attempt in (1, 2):
        text, err = _claude_cli(prompt, model, timeout, binary)
        if err:
            return None
        data = _extract_json(text or "")
        if data is not None:
            break
        if attempt == 2:
            return None
    warns: list[str] = []
    scope = Scope()

    cam = data.get("camera")
    if cam:
        match = next((c for c in vocab.cameras if c.lower() == str(cam).lower()),
                     None)
        if match:
            scope.camera = match
        else:
            # Dropped, not honoured: an invented camera returns zero rows and
            # looks like "nothing matched".
            warns.append(f"ignored unknown camera {cam!r} from the router; "
                         f"searching all cameras")

    for fld in ("t_from", "t_to"):
        v = data.get(fld)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            warns.append(f"ignored unparseable {fld}={v!r}")
            continue
        # Bound it: a hallucinated epoch silently hides everything.
        if not (0 < v < now + 86400):
            warns.append(f"ignored out-of-range {fld}={v!r}")
            continue
        setattr(scope, fld, v)
    if (scope.t_from is not None and scope.t_to is not None
            and scope.t_from > scope.t_to):
        warns.append("router returned t_from after t_to; dropped the window")
        scope.t_from = scope.t_to = None

    subject = str(data.get("subject") or "").strip()
    if not subject:
        return None      # a router that ate the subject is not usable
    return Route(scope=scope, subject=subject, source="claude-cli",
                 warnings=warns)


# --------------------------------------------------------------------------
def subject_fidelity(query: str, subject: str) -> bool:
    """Is `subject` the query minus removed words, with nothing invented?

    The router may only DELETE scope phrases. This catches a model that
    rephrased, translated or 'corrected' the subject - which would silently
    change what the user searched for.
    """
    def bag(s):
        return sorted(re.findall(r"[a-z0-9]+", s.lower()))
    q, s = bag(query), bag(subject)
    qi = iter(q)
    return all(tok in qi for tok in s)      # subject is a subsequence of query


def route(query: str, vocab: Vocabulary, use_llm: bool = True,
          now: float | None = None, model: str = CLAUDE_MODEL,
          timeout: float = CLAUDE_TIMEOUT_S, binary: str = "claude") -> Route:
    """Route one query. Never raises; always returns something searchable."""
    r = None
    if use_llm:
        r = route_claude(query, vocab, now, model, timeout, binary)
        if r is not None and not subject_fidelity(query, r.subject):
            # The one thing the LLM is not allowed to do.
            r = None
    if r is None:
        r = route_deterministic(query, vocab, now)

    # Advisory, never a reroute: name the exact alternative and the cost.
    hits = vocab.attribute_hits(r.subject)
    if hits:
        pretty = "; ".join(f"{v!r} ({'/'.join(keys)})" for v, keys in hits)
        r.warnings.append(
            f"{pretty} {'is a' if len(hits) == 1 else 'are'} known attribute "
            f"value{'' if len(hits) == 1 else 's'}. The vector path binds "
            f"attributes unreliably - p@5 ~0.28 measured on class x colour, "
            f"against ~0.78 for the object alone - so an exact answer may be "
            f"cheaper: query.py --class/--group plus a filter on the "
            f"attribute.")
    return r
