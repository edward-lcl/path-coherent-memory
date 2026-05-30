#!/usr/bin/env python3
"""
DOUBLE-DISJOINT MuSiQue tail — the true public analogue of the Talos hard subset.

The single embedding-disjoint filter (q<->terminal cos<0.3) still leaves MuSiQue
bridges that are LEXICALLY reachable (named entities BM25 can hit). That's why
path looked junior there. Talos's hard subset is links unreachable by BM25 AND
dense simultaneously. This applies BOTH filters to MuSiQue:

  keep question iff terminal is NOT in dense top-k AND NOT in BM25 top-k
  (scored from the FULL question, i.e. single-hop retrieval genuinely fails)

then runs the 3-way complementarity (dense / path / oracle-iter) on that
double-disjoint subset. Hypothesis: in this regime path's EXCLUSIVE share rises
toward co-equal with iterative, matching the Talos finding — i.e. the
path/iterative balance is governed by bridge type, and the concept-bridge regime
is where path stops being redundant. Oracle-iter kept as the fast ceiling; the
real-LLM loop can be layered on the resulting subset separately.
"""
import json, sys, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hotpotqa_hybrid_eval as H
from path_coherent_retriever import build_idf_table
from embedding_bridge_retriever import embed_texts
from musique_eval import hop_bucket, embed_corpus

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

BIO_CORPUS = ROOT / "wikipedia_bio_corpus.json"
PLACEHOLDER = re.compile(r"#(\d+)")


def run(scan, emb_thresh, k, n_bios, split):
    print(f"Scanning MuSiQue {split} 2-hop (emb<{emb_thresh} pre-filter)…")
    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    cand = []
    for ex in ds:
        if hop_bucket(ex) != 2 or not ex.get("answerable", True):
            continue
        steps = ex["question_decomposition"]
        paras = {p["idx"]: p for p in ex["paragraphs"]}
        si, s0 = steps[-1].get("paragraph_support_idx"), steps[0].get("paragraph_support_idx")
        if si not in paras or s0 not in paras:
            continue
        cand.append({"id": ex["id"], "q": ex["question"], "paras": ex["paragraphs"],
                     "gold_term_idx": si, "gold_bridge_idx": s0,
                     "subqs": [s["question"] for s in steps],
                     "answers": [s["answer"] for s in steps]})
        if len(cand) >= scan:
            break

    # Pre-filter by embedding disjointness (cheap, narrows the scan).
    qv = embed_texts([c["q"] for c in cand], batch_size=256)
    tv = embed_texts([{p["idx"]: p for p in c["paras"]}[c["gold_term_idx"]]["paragraph_text"][:512]
                      for c in cand], batch_size=256)
    for i, c in enumerate(cand):
        c["qt_cos"] = float(qv[i] @ tv[i])
    emb_disjoint = [c for c in cand if c["qt_cos"] < emb_thresh]
    print(f"  embedding-disjoint pre-filter: {len(emb_disjoint)}/{len(cand)}")

    # Build the shared corpus (same as the single-disjoint run) so caches/scores align.
    notes_by_id, qrows = {}, []
    for c in emb_disjoint:
        pid_by_idx = {}
        for p in c["paras"]:
            pid = f"{c['id']}::p{p['idx']}"
            pid_by_idx[p["idx"]] = pid
            notes_by_id.setdefault(pid, {"id": pid, "source": p["title"],
                                         "content": p["paragraph_text"]})
        qrows.append({"q": c["q"], "gold": pid_by_idx[c["gold_term_idx"]],
                      "subqs": c["subqs"], "answers": c["answers"]})
    for b in (json.loads(BIO_CORPUS.read_text())[:n_bios] if n_bios else []):
        notes_by_id.setdefault("bio::" + b["source"],
                               {"id": "bio::" + b["source"], "source": b["source"],
                                "content": b["content"]})
    notes = list(notes_by_id.values())
    corpus = H.build_index(notes)
    N = len(notes)
    idf_table = build_idf_table(corpus["df"], N)
    bm25_score_all, bm25_ranked = H.make_bm25_proper(corpus)
    path_scores = H.make_path_scores(corpus, idf_table, bm25_ranked)
    id_list = [n["id"] for n in notes]
    # Reuse the single-disjoint run's full corpus embedding cache (identical notes).
    doc_vecs = embed_corpus(notes, f"disjoint_{split}_{len(qrows)}_{n_bios}")
    q_vecs = embed_texts([q["q"] for q in qrows], batch_size=256)

    # SECOND filter: terminal also NOT in BM25 top-k from the full question.
    double = []
    for qi, q in enumerate(qrows):
        bm = bm25_score_all(q["q"])
        bm_top = set(sorted(bm, key=lambda x: bm[x], reverse=True)[:k])
        sims = doc_vecs @ q_vecs[qi]
        dense_top = set(id_list[j] for j in np.argsort(-sims)[:k])
        if q["gold"] not in bm_top and q["gold"] not in dense_top:
            double.append((qi, q))
    print(f"  DOUBLE-disjoint (terminal not in BM25 top{k} AND not in dense top{k}): "
          f"{len(double)}/{len(qrows)}")
    if not double:
        print("  (empty subset — try larger --scan or higher --emb-thresh)")
        return

    # 3-way on the double-disjoint subset.
    res_subqs, sub_idx = [], []
    for qi, q in double:
        gold_bridge = q["answers"][0]
        res_subqs.append(PLACEHOLDER.sub(lambda m: gold_bridge, q["subqs"][1]))
    sq_vecs = embed_texts(res_subqs, batch_size=256)

    hitmaps = []
    for i, (qi, q) in enumerate(double):
        gold = q["gold"]
        sims = doc_vecs @ q_vecs[qi]
        dense_hit = gold in set(id_list[j] for j in np.argsort(-sims)[:k])  # ~0 by construction
        ps = path_scores(q["q"])
        path_hit = gold in set(sorted(ps, key=lambda x: ps[x], reverse=True)[:k])
        ss = doc_vecs @ sq_vecs[i]
        oracle_hit = gold in set(id_list[j] for j in np.argsort(-ss)[:k])
        hitmaps.append({"dense": dense_hit, "path": path_hit, "oracle-iter": oracle_hit})

    n = len(hitmaps)
    MODES = ["dense", "path", "oracle-iter"]
    rec = {m: sum(h[m] for h in hitmaps) / n for m in MODES}
    ens = ["path", "oracle-iter"]
    union = sum(any(h[m] for m in ens) for h in hitmaps) / n
    best = max(rec[m] for m in ens)
    excl = {m: sum(h[m] and not any(h[o] for o in ens if o != m) for h in hitmaps) / n for m in ens}
    inter = sum(h["path"] and h["oracle-iter"] for h in hitmaps)
    uni = sum(h["path"] or h["oracle-iter"] for h in hitmaps)
    jac = inter / uni if uni else 0.0

    print(f"\nMuSiQue DOUBLE-disjoint subset — n={n}, corpus {N:,}, k={k}")
    print(f"  dense={100*rec['dense']:.1f}%  path={100*rec['path']:.1f}%  "
          f"oracle-iter={100*rec['oracle-iter']:.1f}%")
    print(f"  PATH+ITER union={100*union:.1f}%  best-single={100*best:.1f}%  "
          f"lift=+{100*(union-best):.1f}pp")
    print(f"  exclusive: path={100*excl['path']:.1f}%  oracle-iter={100*excl['oracle-iter']:.1f}%")
    print(f"  Jaccard(path,iter)={jac:.2f}")
    print(f"  >> path share of union = {100*rec['path']/union if union else 0:.0f}% "
          f"(co-equal regime if path-exclusive is non-trivial)")

    out = ROOT / "musique_double_disjoint_results.json"
    out.write_text(json.dumps({"n": n, "corpus": N, "k": k, "emb_thresh": emb_thresh,
                               "recall": rec, "union": union, "best": best,
                               "exclusive": excl, "jaccard": jac}, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=int, default=2500)
    p.add_argument("--emb-thresh", type=float, default=0.35)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--bios", type=int, default=5000)
    p.add_argument("--split", default="validation")
    a = p.parse_args()
    run(a.scan, a.emb_thresh, a.k, a.bios, a.split)
