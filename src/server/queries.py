"""Read-side queries the dashboard needs and the API did not have.

Separate from app.py for the same reason hub.py and webhooks.py are: app.py
should read as a list of endpoints, not as a SQL file. Every function here
takes a connection and returns plain dicts, so each is testable without a
server and without an event loop.

ON RUNNING THESE OFF THE EVENT LOOP

`SqliteStore.connect_ro()` builds a fresh connection with sqlite3's default
`check_same_thread=True`, so a connection opened on the event loop cannot be
used from a worker thread. Anything slow therefore opens its OWN connection
inside the thread that uses it - which `run_search` does, because a CLIP text
encode plus a brute-force matmul is hundreds of milliseconds and must not
block the loop that is streaming video.

WHY SEARCH DOES NOT CALL router.route()

`route()` prefers an LLM and shells out to the `claude` CLI. From an HTTP
handler that is wrong twice over: it spawns a subprocess per request, and
inside a Claude Code session it is unauthenticated, so it burns ~0.9 s failing
before falling back. `route_deterministic` is the same interface without
either problem, and the scope it extracts is returned to the caller so the UI
can show what it understood.
"""

from __future__ import annotations

from ..query.router import load_vocabulary, route_deterministic
from ..query.search import search as vector_search

# The framing the measurement forces, carried in every search response so a
# consumer cannot accidentally present a ranking as a detection. See the module
# docstring of src/query/search.py: no similarity threshold separated present
# concepts from absent ones, so this answers "the nearest N of M candidates"
# and never "X is present". The UI renders this verbatim.
RANKER_NOTE = ("Ranked by visual similarity, not detected. These are the "
               "nearest {shown} of {candidates} embedded crops in scope - a "
               "top result is the closest match whether or not the thing you "
               "asked for is in the footage at all.")

_OBJECT_COLUMNS = """
    SELECT o.id, o.run_id, o.track_id, o.cls_name, o.cls_group,
           o.first_seen_s, o.last_seen_s, o.frames_seen, o.best_conf,
           o.lane_id, o.lane_flag, o.identity_id, v.kind AS identity_kind,
           (o.crop_path IS NOT NULL AND o.crop_path != '') AS has_crop,
           r.camera, r.time_base,
           CASE WHEN v.kind = 'plate' THEN v.key END AS plate
      FROM objects o
      JOIN runs r ON r.id = o.run_id
 LEFT JOIN identities v ON v.id = o.identity_id
"""


# --- filter vocabulary -----------------------------------------------------
def vocabulary(conn) -> dict:
    """Cameras, classes, groups and attribute values actually present.

    A thin pass-through over router.load_vocabulary, which already reads all
    four from the database rather than hardcoding them - so enabling a new
    enricher populates the dashboard's filters with no change here.
    """
    v = load_vocabulary(conn)
    return {"cameras": sorted(v.cameras), "classes": sorted(v.classes),
            "groups": sorted(v.groups),
            "attributes": {k: sorted(vals) for k, vals in
                           sorted(v.attributes.items())}}


# --- objects ---------------------------------------------------------------
def list_objects(conn, cls: str | None = None, group: str | None = None,
                 camera: str | None = None, flagged: bool = False,
                 identity_id: int | None = None, has_crop: bool | None = None,
                 attr_key: str | None = None, attr_value: str | None = None,
                 limit: int = 60, offset: int = 0) -> dict:
    """One page of objects, newest first, with their attributes attached."""
    where, params = [], []
    if cls:
        where.append("o.cls_name = ?")
        params.append(cls)
    if group:
        where.append("o.cls_group = ?")
        params.append(group)
    if camera:
        where.append("r.camera = ?")
        params.append(camera)
    if flagged:
        # 'ok' is a verdict, not a flag; '' and NULL are "never assessed".
        where.append("o.lane_flag IS NOT NULL AND o.lane_flag NOT IN ('', 'ok')")
    if identity_id is not None:
        where.append("o.identity_id = ?")
        params.append(int(identity_id))
    if has_crop is True:
        where.append("o.crop_path IS NOT NULL AND o.crop_path != ''")
    elif has_crop is False:
        where.append("(o.crop_path IS NULL OR o.crop_path = '')")
    if attr_key:
        # EXISTS rather than a join: idx_attr_kv covers (key, value), and a
        # join would multiply rows when an object matches more than one value.
        if attr_value:
            where.append("EXISTS (SELECT 1 FROM attributes a WHERE"
                         " a.object_id = o.id AND a.key = ? AND a.value = ?)")
            params += [attr_key, attr_value]
        else:
            where.append("EXISTS (SELECT 1 FROM attributes a WHERE"
                         " a.object_id = o.id AND a.key = ?)")
            params.append(attr_key)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    total = conn.execute(
        "SELECT COUNT(*) FROM objects o JOIN runs r ON r.id = o.run_id"
        + clause, params).fetchone()[0]
    rows = conn.execute(
        _OBJECT_COLUMNS + clause + " ORDER BY o.id DESC LIMIT ? OFFSET ?",
        (*params, int(limit), int(offset))).fetchall()

    out = [dict(r) for r in rows]
    _attach_attributes(conn, out)
    return {"objects": out, "total": total, "limit": limit, "offset": offset}


def get_object(conn, object_id: int) -> dict | None:
    row = conn.execute(_OBJECT_COLUMNS + " WHERE o.id = ?",
                       (int(object_id),)).fetchone()
    if row is None:
        return None
    obj = dict(row)
    _attach_attributes(conn, [obj])
    # Deliberately NOT joining `events` here. The per-object log is dominated
    # by `clothes_read`/`person_read`/`color_read` bookkeeping - one row per
    # enricher per object - and the facts worth reading off it are already on
    # this row: the attributes those events announced, and `lane_flag` for the
    # verdict ones. `GET /events` still serves the raw log for anyone who
    # wants it.
    return obj


def _attach_attributes(conn, objects: list) -> None:
    """One `IN` query for the whole page rather than one per row."""
    ids = [o["id"] for o in objects]
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    found: dict[int, dict] = {}
    for oid, key, value, conf in conn.execute(
            f"SELECT object_id, key, value, conf FROM attributes"
            f" WHERE object_id IN ({marks})", ids):
        if value not in (None, ""):
            found.setdefault(oid, {})[key] = {"value": value, "conf": conf}
    for o in objects:
        attrs = found.get(o["id"], {})
        # plate_number duplicates identities.key for objects that got a
        # durable identity, and is the only reading for those that did not.
        plate_attr = attrs.pop("plate_number", None)
        if not o.get("plate") and plate_attr:
            o["plate"] = plate_attr["value"]
            o["plate_unbound"] = True     # read, but no identities row
        o["attributes"] = attrs


# --- identities ------------------------------------------------------------
def list_identities(conn, kind: str = "plate", key: str | None = None,
                    limit: int = 100, offset: int = 0) -> dict:
    """Durable identities with their sighting counts.

    Sightings are COUNT(*) over objects.identity_id, not a stored counter - the
    schema comment in sqlite_store.py says why: a counter is bumped by every
    mid-track rebind as well as by finalize, so it drifts from the rows it
    claims to count.

    `kind` scopes the axis; the plate default reproduces the old vehicles-only
    behaviour. Rows carry both `key` (generic) and `plate` (an alias, so a
    plate-axis caller reads naturally).
    """
    where, params = ["v.kind = ?"], [str(kind)]
    if key:
        where.append("v.key LIKE ?")
        params.append(f"%{key.upper()}%")
    clause = " WHERE " + " AND ".join(where)
    total = conn.execute("SELECT COUNT(*) FROM identities v" + clause,
                         params).fetchone()[0]
    rows = conn.execute(
        "SELECT v.id, v.kind, v.key, v.key AS plate,"
        "       v.first_seen_at, v.last_seen_at,"
        "       COUNT(o.id) AS sightings,"
        "       COUNT(DISTINCT r.camera) AS camera_count,"
        "       GROUP_CONCAT(DISTINCT r.camera) AS cameras"
        "  FROM identities v"
        "  LEFT JOIN objects o ON o.identity_id = v.id"
        "  LEFT JOIN runs r ON r.id = o.run_id" + clause +
        " GROUP BY v.id ORDER BY COALESCE(v.last_seen_at, 0) DESC"
        " LIMIT ? OFFSET ?", (*params, int(limit), int(offset))).fetchall()
    out = []
    for r in rows:
        row = dict(r)
        row["cameras"] = sorted((row.get("cameras") or "").split(",")) if \
            row.get("cameras") else []
        out.append(row)
    return {"identities": out, "total": total, "limit": limit, "offset": offset}


# --- index freshness -------------------------------------------------------
def index_status(conn, model: str) -> dict:
    """How much of the crop corpus has vectors in one embedding space.

    WHY THIS EXISTS. Embeddings are built only by tools/embed_crops.py, never
    by the pipeline (a deliberate decision - no real-time decision needs an
    embedding). So a fresh database, or one that has simply run for a while
    since the last embed, has crops that search cannot see. Nothing announced
    that: search returned fewer results than the operator expected and looked
    like it was working. This is what makes a stale index say so.

    `unembedded` is an UPPER BOUND, not a promise. The embedder additionally
    skips crops that are too small, too low-confidence, or nearly frame-sized,
    and deciding that requires opening each JPEG - far too expensive for a
    status call. So this counts what has no vector, and the caller must not
    present it as "this many will be added".
    """
    out = {"model": model, "embedded": 0, "crops": 0, "unembedded": 0,
           "by_model": {}}
    try:
        out["crops"] = conn.execute(
            "SELECT COUNT(*) FROM objects"
            " WHERE crop_path IS NOT NULL AND crop_path != ''").fetchone()[0]
        out["by_model"] = {m: n for m, n in conn.execute(
            "SELECT model, COUNT(*) FROM embeddings GROUP BY model")}
        out["embedded"] = out["by_model"].get(model, 0)
        # Mirrors the embedder's own resumability clause (embed_crops.py:85):
        # a crop counts as missing when THIS model has no row for it.
        out["unembedded"] = conn.execute(
            "SELECT COUNT(*) FROM objects o"
            " WHERE o.crop_path IS NOT NULL AND o.crop_path != ''"
            "   AND NOT EXISTS (SELECT 1 FROM embeddings e"
            "                    WHERE e.object_id = o.id AND e.model = ?)",
            (model,)).fetchone()[0]
    except Exception:
        pass          # a pre-embeddings database is not an error
    return out


# --- semantic search -------------------------------------------------------
def run_search(conn, encoder, q: str, camera: str | None = None,
               top: int = 24, min_px: int = 0, min_conf: float = 0.0) -> dict:
    """Route, rank, and annotate one query. Synchronous; run it in a thread.

    Never raises on an empty or unfetched index - `vector_search` returns its
    `error`/`note` string instead, and both are passed through so the UI can
    say what is actually wrong (usually "nothing embedded yet").
    """
    vocab = load_vocabulary(conn)
    route = route_deterministic(q, vocab)
    if camera:
        # An explicit filter beats one parsed out of the text: the user picked
        # it in the UI after seeing the parse.
        route.scope.camera = camera

    result = vector_search(conn, route.subject, route.scope,
                           model=encoder.model, top_k=top, min_px=min_px,
                           min_conf=min_conf, encoder=encoder)

    hits = result.get("hits") or []
    if hits:
        _annotate_hits(conn, hits)
    else:
        # WHICH spaces have vectors, not just whether THIS one does. Embeddings
        # are per-model and cosine is only meaningful within one space, so a
        # model with an empty index is the single most common reason for an
        # empty result - and "nothing is indexed" would be a lie when the
        # other model is fully embedded. Cheap: one grouped count.
        try:
            result["indexed_models"] = {
                m: n for m, n in conn.execute(
                    "SELECT model, COUNT(*) FROM embeddings GROUP BY model")}
        except Exception:
            result["indexed_models"] = {}
    result.update({
        "query": q,
        "subject": route.subject,
        "scope": {"camera": route.scope.camera,
                  "t_from": route.scope.t_from, "t_to": route.scope.t_to},
        "warnings": list(route.warnings),
        "ranker_note": RANKER_NOTE.format(shown=len(hits),
                                          candidates=result.get("candidates", 0)),
    })
    return result


def _annotate_hits(conn, hits: list) -> None:
    """Attach plate and attributes so a result card needs no second request."""
    ids = [h["object_id"] for h in hits]
    marks = ",".join("?" * len(ids))
    rows = {r["id"]: r for r in conn.execute(
        _OBJECT_COLUMNS + f" WHERE o.id IN ({marks})", ids)}
    enriched = [dict(rows[i]) for i in ids if i in rows]
    _attach_attributes(conn, enriched)
    by_id = {o["id"]: o for o in enriched}
    for h in hits:
        o = by_id.get(h["object_id"], {})
        h["plate"] = o.get("plate")
        h["attributes"] = o.get("attributes", {})
        h["has_crop"] = bool(o.get("has_crop"))
        h["lane_flag"] = o.get("lane_flag") or None
        h["frames_seen"] = o.get("frames_seen")
        h["time_base"] = o.get("time_base")
