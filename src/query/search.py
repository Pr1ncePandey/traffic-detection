"""Open-vocabulary search: SQL prefilter -> vector rerank -> report.

The order is load-bearing, and for two separate reasons:

CORRECTNESS. The SQL step narrows on things that cannot be wrong - camera,
time window, and the floors already baked into the stored rows. It derives no
attribute or class predicate from the query text; see src/query/router.py for
why (capped recall on extracted attributes).

COST. Ranking is brute-force dot product in numpy, so every candidate vector
is read and multiplied. At 2 KB a vector that is fine for thousands and not
for millions, which makes the prefilter a performance component and not only a
semantic one. There is no ANN index: at this volume brute force is faster than
one, with no recall loss and nothing to tune.

THERE IS NO SIMILARITY FLOOR, DELIBERATELY. Measurement found no threshold
separating present concepts from absent ones - absent concepts outranked
present ones once per-prompt scale was removed - so rather than invent one,
this answers as a RANKER: "the nearest N of M candidates", never "X is here"
or "X is absent". Every result says so. Reopen that decision before wiring
this to anything automated, or past ~1e5 objects: top-1 for an absent concept
is the corpus maximum, so the bigger the index the more convincing the noise.
"""

from __future__ import annotations

import numpy as np

from .clip_onnx import DEFAULT_MODEL, ClipOnnx


def candidate_sql(model: str, scope, min_px: int = 0, min_conf: float = 0.0):
    """Build the prefilter. Returns (sql, params).

    Filters on the columns the embedder recorded, so a floor can be tightened
    at QUERY time without re-embedding - which is the whole reason crop_w/
    crop_h/crop_conf/crop_area are columns rather than a passing thought.
    """
    sql = ["SELECT e.object_id, e.vec, e.dim, e.crop_w, e.crop_h,",
           "       e.crop_conf, o.cls_name, o.first_seen_s, r.camera, r.id AS run_id",
           "FROM embeddings e",
           "JOIN objects o ON o.id = e.object_id",
           "JOIN runs r ON r.id = o.run_id",
           "WHERE e.model = ?"]
    params: list = [model]
    if scope.camera:
        sql.append("AND r.camera = ?")
        params.append(scope.camera)
    if min_px:
        sql.append("AND MIN(e.crop_w, e.crop_h) >= ?")
        params.append(int(min_px))
    if min_conf:
        sql.append("AND e.crop_conf >= ?")
        params.append(float(min_conf))
    if scope.t_from is not None:
        sql.append("AND o.first_seen_s >= ?")
        params.append(float(scope.t_from))
    if scope.t_to is not None:
        sql.append("AND o.first_seen_s <= ?")
        params.append(float(scope.t_to))
    return " ".join(sql), params


def unorderable_runs(conn, scope) -> list[int]:
    """Runs a wall-clock window cannot honestly be applied to.

    A file run timestamps on clip-seconds (0..duration), so comparing it to a
    unix epoch is meaningless - every row would fall below the cutoff and be
    dropped for the wrong reason. `runs.time_base` records which clock a run
    used. These are REPORTED, never silently filtered, matching how
    src/journeys.py handles `unorderable`.
    """
    if scope.t_from is None and scope.t_to is None:
        return []
    try:
        return [r[0] for r in conn.execute(
            "SELECT id FROM runs WHERE time_base IS NULL OR time_base != 'wall'")]
    except Exception:
        return []


def search(conn, subject: str, scope, model: str = DEFAULT_MODEL,
           top_k: int = 10, min_px: int = 0, min_conf: float = 0.0,
           encoder: ClipOnnx | None = None) -> dict:
    """Rank candidates against one query string. Never raises on empty."""
    enc = encoder or ClipOnnx(model)
    missing = enc.missing()
    if missing:
        return {"error": f"model {model} not fetched: {missing[0]} missing. "
                         f"Run: python tools/fetch_clip.py --model {model}"}

    sql, params = candidate_sql(model, scope, min_px, min_conf)
    rows = conn.execute(sql, params).fetchall()
    excluded = unorderable_runs(conn, scope)
    if not rows:
        return {"hits": [], "candidates": 0, "model": model,
                "excluded_runs": excluded,
                # --model is NOT optional in this hint. The embedder defaults
                # to b16 while a caller may be searching l14, and cosine is
                # only meaningful within one space - so the command without it
                # can appear to succeed and still leave this query with an
                # empty index.
                "note": f"no embedded crop matched the scope. Embed first: "
                        f"python tools/embed_crops.py --model {model}"}

    dim = int(rows[0]["dim"])
    # One contiguous matrix, then a single matmul. Vectors were normalised on
    # write, so cosine IS the dot product and no division happens here - which
    # is what keeps the query path from disagreeing with the writer about the
    # metric.
    mat = np.empty((len(rows), dim), dtype=np.float32)
    keep = []
    for i, r in enumerate(rows):
        v = np.frombuffer(r["vec"], dtype="<f4")
        if v.size != dim:
            continue          # a row from another space; skip rather than crash
        mat[len(keep)] = v
        keep.append(r)
    mat = mat[:len(keep)]

    qv = enc.encode_query(subject)
    if qv.size != dim:
        return {"error": f"query vector is {qv.size}-d but stored vectors are "
                         f"{dim}-d - the model string does not match the index"}

    sims = mat @ qv
    order = np.argsort(-sims)[:max(1, top_k)]
    hits = []
    for j in order:
        r = keep[j]
        hits.append({"object_id": r["object_id"], "score": float(sims[j]),
                     "cls_name": r["cls_name"], "camera": r["camera"],
                     "run_id": r["run_id"], "first_seen_s": r["first_seen_s"],
                     "crop_w": r["crop_w"], "crop_h": r["crop_h"],
                     "crop_conf": r["crop_conf"]})
    return {"hits": hits, "candidates": len(keep), "model": model,
            "excluded_runs": excluded,
            "score_spread": (float(sims.min()), float(sims.max()))}
