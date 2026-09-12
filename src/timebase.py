"""Turning a run-local timestamp into an absolute one.

THE PROBLEM THIS EXISTS FOR

`objects.first_seen_s` holds two incompatible kinds of number, decided by one
line in the pipeline (`src/pipeline.py`):

    timestamp = time.time() if info.is_live else index / info.fps

For a LIVE source that is a wall-clock epoch (~1.7e9), so cross-camera ordering
already works with no conversion. For a RECORDED FILE it is
seconds-from-the-start-of-that-clip (~12.4). Every clip starts at 0, so
A@12.4 and B@8.1 say nothing about which happened first.

The intended deployment is one fleet-wide database, so a live run and a file run
WILL coexist in it, and any query that sorts by time then returns silent
nonsense. `runs.time_base` is what distinguishes the two, and this module is the
only place that reads it.

WHY NOT JUST STORE PROCESSING WALL-CLOCK

It is the obvious thing to reach for and it does not work:

  - Process camera A's clip today and B's tomorrow and every A sighting
    precedes every B sighting, whatever the footage shows.
  - Within one run the spacing between detections is a function of throughput.
    A clip decoded at 3x real time and one at 0.8x produce different wall-clock
    gaps for the same traffic, so the ordering would encode GPU speed.

The footage's own start time is the only honest origin, which is why it has to
be supplied (`main.py --recorded-at`) rather than inferred.

WHY None IS A RETURN VALUE AND NOT AN ERROR

A file run with no `recorded_at` is genuinely *unorderable*. Callers must set
those sightings aside and say so - `src/journeys.py` collects them into
`unorderable` and reports the count. Substituting 0.0, or the processing time,
would turn "we do not know when this was" into a confident wrong answer.
"""

EPOCH = "epoch"
CLIP = "clip"

# Below this, a number cannot be a wall-clock epoch (2001-09-09). Used only to
# sanity-check a run row whose time_base is missing entirely, i.e. one written
# before that column existed.
_EPOCH_FLOOR = 1_000_000_000.0


def time_base_for(is_live: bool) -> str:
    """What to record in `runs.time_base` for a source."""
    return EPOCH if is_live else CLIP


def absolute_time(run, secs) -> float | None:
    """Absolute wall-clock seconds for `secs` as recorded against `run`.

    `run` is anything indexable by column name - an sqlite3.Row or a dict.

        time_base == 'epoch'                 -> secs (already wall-clock)
        time_base == 'clip' and recorded_at  -> recorded_at + secs
        otherwise                            -> None (unorderable, say so)

    A legacy row with no time_base at all falls back to guessing from the
    MAGNITUDE of secs, which is reliable only in the epoch direction: a value
    above _EPOCH_FLOOR cannot be clip-seconds. A small value is genuinely
    ambiguous, so it returns None rather than assuming.
    """
    if secs is None:
        return None
    try:
        value = float(secs)
    except (TypeError, ValueError):
        return None

    base = _get(run, "time_base")
    if base == EPOCH:
        return value
    if base == CLIP:
        origin = _get(run, "recorded_at")
        if origin is None:
            return None
        try:
            return float(origin) + value
        except (TypeError, ValueError):
            return None
    # No time_base: a pre-migration row. Only the epoch case is decidable.
    return value if value >= _EPOCH_FLOOR else None


def describe_base(run) -> str:
    """One human-readable phrase for why a run is or is not orderable."""
    base = _get(run, "time_base")
    if base == EPOCH:
        return "live (wall-clock)"
    if base == CLIP:
        origin = _get(run, "recorded_at")
        if origin is None:
            return "file (no --recorded-at: unorderable)"
        return "file (anchored)"
    return "unknown time base (pre-migration run)"


def _get(run, key):
    """Column access that works for sqlite3.Row, dict, and neither."""
    if run is None:
        return None
    if isinstance(run, dict):
        return run.get(key)
    try:
        return run[key]
    except (KeyError, IndexError, TypeError):
        return None
