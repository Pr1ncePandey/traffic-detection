"""Per-track voting: several AdaFace reads of one person -> a name, or unknown.

One read is one AdaFace run on one frame of a tracked person. A read VOTES for
the enrolled person it is closest to only when it clears `threshold` AND beats
the next-closest enrolled person by `margin`. A track is named once one person
has >= `min_votes` votes and more votes than everyone else combined.

The scores are cosine similarities, not probabilities. Measured on this
project's footage (docs/face-recognition.md): same person vs their own selfie
0.51-0.70 in video; different people at most 0.27. 0.45 sits in that gap.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Read:
    ts: float
    name: str | None         # closest enrolled person (None if nobody enrolled)
    similarity: float
    runner_up: float         # next-closest person's similarity (-1 with one person)


@dataclass(frozen=True)
class Decision:
    name: str | None         # None = unknown
    score: float             # mean similarity of the winning votes (0 if unknown)
    votes: int
    reads: int
    reason: str


def votes_for(read: Read, threshold: float, margin: float) -> bool:
    return (read.name is not None and read.similarity >= threshold
            and read.similarity - read.runner_up >= margin)


def decide(reads: list[Read], threshold: float, margin: float, min_votes: int) -> Decision:
    """Name a track from its reads, or say exactly why it stays unknown."""
    if not reads:
        return Decision(None, 0.0, 0, 0, "face never clear enough to check")
    if all(r.name is None for r in reads):
        return Decision(None, 0.0, 0, len(reads), "nobody enrolled")

    votes: dict[str, list[float]] = defaultdict(list)
    for read in reads:
        if votes_for(read, threshold, margin):
            votes[read.name].append(read.similarity)

    if not votes:
        best = max((r for r in reads if r.name is not None), key=lambda r: r.similarity)
        if best.similarity < threshold:
            why = f"closest was {best.name} at {best.similarity:.2f}, below {threshold:.2f}"
        else:
            why = f"{best.name} not clearly ahead of the next person (margin {margin:.2f})"
        return Decision(None, 0.0, 0, len(reads), why)

    winner = max(votes, key=lambda n: (len(votes[n]), sum(votes[n]) / len(votes[n])))
    count = len(votes[winner])
    others = sum(len(v) for n, v in votes.items() if n != winner)
    if count < min_votes:
        return Decision(None, 0.0, count, len(reads),
                        f"{winner} matched {count} of {len(reads)} reads, needs {min_votes}")
    if count <= others:
        return Decision(None, 0.0, count, len(reads),
                        f"reads disagree: {winner} {count} vs others {others}")
    score = round(sum(votes[winner]) / count, 4)
    return Decision(winner, score, count, len(reads), f"{count} of {len(reads)} reads matched")
