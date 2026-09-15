"""Score open-vocabulary retrieval. Derived labels are free; hand labels are not.

    python tools/eval_retrieval.py --derived                  # free, no labelling
    python tools/eval_retrieval.py --derived --camera indian_road
    python tools/eval_retrieval.py --csv samples/retrieval_gt.csv

Scores through the REAL query path (src/query/search.py), not a copy of it, so
a change that breaks search breaks this too. A parallel implementation would
happily report good numbers for code nobody runs.

THREE LABEL SOURCES, AND THEY MEASURE DIFFERENT THINGS

  class      `cls_name` alone - "a truck". SQL answers these exactly, so they
             measure the ceiling, not the use case.
  class+col  `cls_name` x `color` - "a white truck". Compositional, so these
             are the ATTRIBUTE BINDING probe, scored with a decomposition that
             separates "found the class" from "found both terms".
  negative   COCO classes with ZERO detections in the database. Derived by
             query, never from a hand-written list: person_test.mp4 contains
             `boat` and `surfboard` detections, two classes that look
             obviously absent from a pedestrian scene and are not.

Derived labels come from the HSV colour enricher and inherit its error rate,
so they CAP measured precision. Fine for a gate that asks "does this work at
all"; not fine as a published figure without saying so.

THE NEGATIVE-CONTROL GAP IS REPORTED TWO WAYS, AND ONE OF THEM IS WRONG

`raw` is the gap on plain cosine, which is how docs/vector-search.md Layer 4
originally specified it. It is unsound: cosine is not comparable ACROSS
prompts - every score lands in ~0.23-0.35, each prompt at its own scale - so
it returns a PASS on data where no global threshold exists.

`centred` subtracts each prompt's own mean over the corpus, which makes the
comparison meaningful. Trust that one. It is reported as EVIDENCE, not as a
gate: answering "absent" was descoped for v1, so a negative gap is the
justification for that decision rather than a failure.
"""

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.query.clip_onnx import DEFAULT_MODEL, SPECS, ClipOnnx  # noqa: E402
from src.query.router import Scope  # noqa: E402
from src.query.search import search  # noqa: E402

# Candidates for negative controls. Any that the detector actually emitted are
# dropped, by query - see the module docstring.
NEG_POOL = ("zebra", "train", "pizza", "refrigerator", "elephant", "airplane",
            "traffic light", "fire hydrant", "toilet", "bed", "laptop",
            "microwave", "teddy bear", "sheep", "boat", "surfboard",
            "hair drier", "parking meter")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default="outputs/traffic.db")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SPECS))
    p.add_argument("--derived", action="store_true",
                   help="build ground truth from the database (default)")
    p.add_argument("--csv", default=None,
                   help="hand-labelled set, e.g. samples/retrieval_gt.csv")
    p.add_argument("--camera", default=None, help="scope to one camera")
    p.add_argument("--min-n", type=int, default=2,
                   help="derived: minimum positives for a query to be scored")
    p.add_argument("--k", type=int, default=5, help="precision@k (default 5)")
    p.add_argument("--recall-k", type=int, default=10)
    p.add_argument("--verbose", action="store_true", help="per-query rows")
    return p.parse_args()


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------
def embedded_labels(conn, model, camera):
    """(object_id -> (cls_name, colour)) over rows that actually have a vector.

    Scoped to the index rather than to `objects`: scoring against a label
    whose crop was filtered out at embed time would count a miss the search
    could never have hit.
    """
    sql = ["SELECT o.id, o.cls_name, o.cls_group FROM embeddings e",
           "JOIN objects o ON o.id = e.object_id",
           "JOIN runs r ON r.id = o.run_id",
           "WHERE e.model = ?"]
    params = [model]
    if camera:
        sql.append("AND r.camera = ?")
        params.append(camera)
    out = {}
    for oid, cls, grp in conn.execute(" ".join(sql), params):
        key = "upper_color" if grp == "person" else "color"
        row = conn.execute("SELECT value FROM attributes"
                           " WHERE object_id=? AND key=? AND value!=''",
                           (oid, key)).fetchone()
        out[oid] = (cls, row[0] if row else None)
    return out


def derived_queries(labels, min_n):
    """Build class, class+colour and negative queries from the labels."""
    by_cls: dict[str, set] = {}
    by_pair: dict[tuple, set] = {}
    for oid, (cls, col) in labels.items():
        by_cls.setdefault(cls, set()).add(oid)
        if col:
            by_pair.setdefault((cls, col), set()).add(oid)

    qs = []
    for cls, ids in sorted(by_cls.items(), key=lambda x: -len(x[1])):
        if len(ids) >= min_n:
            qs.append({"query": cls, "relevant": ids, "kind": "class",
                       "cls": cls, "col": None})
    for (cls, col), ids in sorted(by_pair.items(), key=lambda x: -len(x[1])):
        if len(ids) >= min_n:
            qs.append({"query": f"{col} {cls}", "relevant": ids,
                       "kind": "class+col", "cls": cls, "col": col})
    return qs


def derived_negatives(conn):
    detected = {r[0] for r in conn.execute("SELECT DISTINCT cls_name FROM objects")}
    rejected = [c for c in NEG_POOL if c in detected]
    keep = [c for c in NEG_POOL if c not in detected]
    return [{"query": c, "relevant": set(), "kind": "negative",
             "cls": None, "col": None} for c in keep], rejected


def csv_queries(path):
    """Hand-labelled set. `expect_route` is accepted and ignored - routing is
    no longer a decision, so there is nothing to score it against."""
    import csv
    out = []
    with open(path, newline="") as f:
        # Strip comment lines before the reader sees them. The label file
        # carries its provenance inline - which rows were proposed, which were
        # eye-verified - and that is worth more than a tidy parser.
        body = [ln for ln in f if not ln.lstrip().startswith("#")]
    for row in csv.DictReader(body):
        q = (row.get("query") or "").strip()
        if q:
            ids = {int(x) for x in (row.get("relevant_object_ids") or "").split("|")
                   if x.strip().isdigit()}
            kind = (row.get("kind") or "positive").strip() or "positive"
            out.append({"query": q, "relevant": ids, "kind": kind,
                        "cls": None, "col": None})
    return out


# --------------------------------------------------------------------------
def score_one(conn, q, labels, scope, model, enc, k, rk):
    """Run one query through the real search path and score it."""
    res = search(conn, q["query"], scope, model=model,
                 top_k=10 ** 6, encoder=enc)
    if "error" in res:
        return None, res["error"]
    hits = res["hits"]
    if not hits:
        return None, "no candidates"

    ranked = [h["object_id"] for h in hits]
    scores = np.array([h["score"] for h in hits], dtype=np.float32)
    rel = q["relevant"]
    n_cand = len(ranked)

    out = {"query": q["query"], "kind": q["kind"], "n_relevant": len(rel),
           "candidates": n_cand, "top1_raw": float(scores[0]),
           # per-prompt centring: the same query's own mean over the corpus.
           "top1_centred": float(scores[0] - scores.mean())}

    if rel:
        got = sum(1 for o in ranked[:k] if o in rel)
        out["p_at_k"] = got / float(k)
        # p@k is unfairly capped when there are fewer than k relevant objects:
        # a class with 2 positives can score at most 2/5 = 0.40 on p@5 no
        # matter how perfect the ranking. Averaging raw p@k over classes of
        # mixed size therefore measures the class-size distribution as much as
        # the model. Dividing by the ACHIEVABLE maximum separates the two.
        out["p_at_k_capped"] = got / float(min(k, len(rel)))
        out["ceiling"] = min(k, len(rel)) / float(k)
        out["r_at_k"] = (sum(1 for o in ranked[:rk] if o in rel)
                         / float(len(rel)))
        prior = len(rel) / float(n_cand)
        out["lift"] = (out["p_at_k"] / prior) if prior else float("nan")
        # Where the relevant objects ACTUALLY landed. A p@5 of 0.00 means
        # something very different at best_rank 6 (nearly working) than at
        # best_rank 150 of 300 (indistinguishable from random), and the raw
        # metric cannot tell those apart. `median_rank` vs n_cand/2 is the
        # random baseline.
        ranks = [i + 1 for i, o in enumerate(ranked) if o in rel]
        if ranks:
            out["best_rank"] = min(ranks)
            out["median_rank"] = int(np.median(ranks))
            out["random_median"] = n_cand // 2

    # binding decomposition, only meaningful when both terms are known
    if q["kind"] == "class+col" and q["cls"] and q["col"]:
        b = c_only = col_only = neither = 0
        for o in ranked[:k]:
            cls, col = labels.get(o, (None, None))
            cm, colm = cls == q["cls"], col == q["col"]
            if cm and colm:
                b += 1
            elif cm:
                c_only += 1
            elif colm:
                col_only += 1
            else:
                neither += 1
        out["bind"] = {"both": b, "cls_only": c_only, "col_only": col_only,
                       "neither": neither, "k": k}
    return out, None


def main():
    a = parse_args()
    if not os.path.exists(a.db):
        print(f"[eval] no database at {a.db}")
        return 2
    if a.csv and not os.path.exists(a.csv):
        print(f"[eval] no label file at {a.csv}. Layer 1b has not been "
              f"written yet; run with --derived for the free set.")
        return 2

    conn = sqlite3.connect(a.db)
    conn.row_factory = sqlite3.Row
    enc = ClipOnnx(a.model)
    if enc.missing():
        print(f"[eval] model {a.model} not fetched: {enc.missing()[0]} "
              f"missing. Run: python tools/fetch_clip.py --model {a.model}")
        return 2

    labels = embedded_labels(conn, a.model, a.camera)
    if not labels:
        print(f"[eval] no embedded crops for {a.model}"
              f"{' on ' + a.camera if a.camera else ''}. "
              f"Run: python tools/embed_crops.py --model {a.model}")
        return 2

    scope = Scope(camera=a.camera)
    rejected: list = []
    if a.csv:
        queries = csv_queries(a.csv)
        source = a.csv
    else:
        queries = derived_queries(labels, a.min_n)
        negs, rejected = derived_negatives(conn)
        queries += negs
        source = "derived from the database"

    print(f"[eval] {source}")
    print(f"[eval] model {a.model}, {len(labels)} embedded crops"
          f"{', camera ' + a.camera if a.camera else ', all cameras'}")
    if rejected:
        print(f"[eval] negative controls REJECTED as actually detected: "
              f"{', '.join(rejected)}")

    rows, errs = [], []
    for q in queries:
        r, err = score_one(conn, q, labels, scope, a.model, enc, a.k, a.recall_k)
        (rows.append(r) if r else errs.append((q["query"], err)))
    for name, err in errs:
        print(f"[eval] {name!r}: {err}")
    if not rows:
        return 1

    def group(kind):
        return [r for r in rows if r["kind"] == kind]

    if a.verbose:
        print(f"\n{'query':30} {'kind':9} {'n':>3} {'p@' + str(a.k):>5} "
              f"{'r@' + str(a.recall_k):>5} {'best':>5} {'med':>5} {'rand':>5}")
        for r in rows:
            print(f"{r['query'][:30]:30} {r['kind']:9} {r['n_relevant']:>3} "
                  f"{r.get('p_at_k', float('nan')):>5.2f} "
                  f"{r.get('r_at_k', float('nan')):>5.2f} "
                  f"{r.get('best_rank', 0) or '-':>5} "
                  f"{r.get('median_rank', 0) or '-':>5} "
                  f"{r.get('random_median', 0) or '-':>5}")

    print(f"\n{'label set':12} {'queries':>8} {'p@' + str(a.k):>6} "
          f"{'capped':>7} {'r@' + str(a.recall_k):>6} {'lift':>6}")
    for kind in ("class", "class+col", "positive", "binding"):
        g = [r for r in group(kind) if "p_at_k" in r]
        if not g:
            continue
        print(f"{kind:12} {len(g):>8} "
              f"{np.mean([r['p_at_k'] for r in g]):>6.2f} "
              f"{np.mean([r['p_at_k_capped'] for r in g]):>7.2f} "
              f"{np.mean([r['r_at_k'] for r in g]):>6.2f} "
              f"{np.mean([r['lift'] for r in g]):>6.1f}x")
    thin = [r for r in rows if r.get("ceiling", 1.0) < 1.0]
    if thin:
        print(f"\n  {len(thin)} quer{'y' if len(thin) == 1 else 'ies'} have "
              f"fewer than k={a.k} positives, so raw p@{a.k} cannot reach 1.0 "
              f"for them.\n  'capped' divides by the achievable maximum "
              f"instead - use it when averaging across\n  classes of very "
              f"different size, and say which one a published figure is.")

    # --- binding ---------------------------------------------------------
    binds = [r["bind"] for r in rows if "bind" in r]
    if binds:
        tot = sum(b["k"] for b in binds)
        both = sum(b["both"] for b in binds)
        conly = sum(b["cls_only"] for b in binds)
        colonly = sum(b["col_only"] for b in binds)
        one = both + conly + colonly
        print(f"\nATTRIBUTE BINDING over {len(binds)} compositional queries:")
        print(f"  p@{a.k} both terms bound  : {both / tot:.2f}")
        print(f"  p@{a.k} at least one term : {one / tot:.2f}")
        print(f"  binding gap              : {(one - both) / tot:+.2f}")
        print(f"    near-misses: {conly} ignored the COLOUR, "
              f"{colonly} ignored the CLASS")

    # --- negative controls ------------------------------------------------
    pos = [r for r in rows if r["n_relevant"] > 0]
    neg = [r for r in rows if r["kind"] == "negative"]
    if pos and neg:
        print(f"\nNEGATIVE CONTROLS ({len(neg)} absent concepts):")
        for tag, key in (("raw (UNSOUND - see the docstring)", "top1_raw"),
                         ("centred per prompt (trust this)", "top1_centred")):
            lo = min(r[key] for r in pos)
            hi = max(r[key] for r in neg)
            print(f"  {tag:34} min_pos {lo:+.3f}  max_neg {hi:+.3f}  "
                  f"gap {lo - hi:+.3f}")
        print("  A negative gap is the EVIDENCE for descoping \"absent\" in v1,\n"
              "  not a failure: --find ranks, it does not adjudicate presence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
