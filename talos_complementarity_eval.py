#!/usr/bin/env python3
"""
Complementarity quantification — the structural spine of the paper.

Question: do path, dense, and oracle-iterative recover the SAME links or
DIFFERENT links? If union >> best-single, no single retrieval mode suffices and
personal-memory retrieval provably needs >=2 orthogonal traversal modes. If
union ~= best-single, they are redundant and the claim is empty.

Reuses talos_clean_eval machinery. For each chain records WHICH methods hit the
terminal, then computes:
  - per-method recall
  - union recall (any method hits)
  - best-single recall
  - pairwise overlap (Jaccard of hit-sets)
  - exclusive contribution (chains ONLY method M recovers)
On ALL chains and on the embedding-disjoint HARD subset.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hotpotqa_hybrid_eval as H
from path_coherent_retriever import build_idf_table
from embedding_bridge_retriever import embed_texts
from talos_clean_eval import load_chains


def run(k: int, thresh: float, real_only: bool):
    chains = load_chains(real_only)
    print(f"Loaded {len(chains)} chains (real_only={real_only})")

    notes_by_id = {}
    for ch in chains:
        for tag, idk in [("a", "a_id"), ("b", "b_id"), ("c", "c_id")]:
            notes_by_id.setdefault(ch[idk], {"id": ch[idk], "source": ch[idk],
                                             "content": ch[tag]})
    notes = list(notes_by_id.values())
    corpus = H.build_index(notes)
    N = len(notes)
    idf_table = build_idf_table(corpus["df"], N)
    bm25_score_all, bm25_ranked = H.make_bm25_proper(corpus)
    path_scores = H.make_path_scores(corpus, idf_table, bm25_ranked)

    id_list = [n["id"] for n in notes]
    id_pos = {d: i for i, d in enumerate(id_list)}
    doc_vecs = embed_texts([n["content"][:512] for n in notes], batch_size=256)
    for ch in chains:
        ch["ac_cos"] = float(doc_vecs[id_pos[ch["a_id"]]] @ doc_vecs[id_pos[ch["c_id"]]])
    print(f"  corpus {N:,} nodes")

    MODES = ["dense", "path", "oracle-iter"]

    def hits_for(ch):
        gold = ch["c_id"]
        a_vec = doc_vecs[id_pos[ch["a_id"]]]
        query = ch["a"]
        sims = doc_vecs @ a_vec
        dense_top = [id_list[j] for j in np.argsort(-sims)[:k]]

        path_s = path_scores(query)
        path_top = sorted(path_s, key=lambda x: path_s[x], reverse=True)[:k]

        b_vec = doc_vecs[id_pos[ch["b_id"]]]
        bsims = doc_vecs @ b_vec
        iter_top = list(dict.fromkeys([ch["b_id"]] +
                        [id_list[j] for j in np.argsort(-bsims)[:k]]))[:k]

        return {
            "dense": gold in set(dense_top),
            "path": gold in set(path_top),
            "oracle-iter": gold in set(iter_top),
        }

    def analyze(subset, tag):
        n = len(subset)
        if not n:
            return {}
        hitmaps = [hits_for(ch) for ch in subset]
        rec = {m: sum(h[m] for h in hitmaps) / n for m in MODES}
        union = sum(any(h[m] for m in MODES) for h in hitmaps) / n
        best = max(rec.values())
        # exclusive: only this mode hit
        excl = {m: sum(h[m] and not any(h[o] for o in MODES if o != m)
                       for h in hitmaps) / n for m in MODES}
        # pairwise jaccard of hit-sets
        jac = {}
        for i, m1 in enumerate(MODES):
            for m2 in MODES[i+1:]:
                inter = sum(h[m1] and h[m2] for h in hitmaps)
                uni = sum(h[m1] or h[m2] for h in hitmaps)
                jac[f"{m1}&{m2}"] = inter / uni if uni else 0.0
        print(f"\n{tag} — n={n}")
        print(f"  per-method: " + "  ".join(f"{m}={100*rec[m]:.1f}%" for m in MODES))
        print(f"  best-single={100*best:.1f}%   UNION={100*union:.1f}%   "
              f"lift=+{100*(union-best):.1f}pp")
        print(f"  exclusive (only that mode recovers): " +
              "  ".join(f"{m}={100*excl[m]:.1f}%" for m in MODES))
        print(f"  pairwise Jaccard: " + "  ".join(f"{kk}={v:.2f}" for kk, v in jac.items()))
        return {"n": n, "recall": rec, "union": union, "best": best,
                "exclusive": excl, "jaccard": jac}

    res_all = analyze(chains, "ALL chains")
    hard = [ch for ch in chains if ch["ac_cos"] < thresh]
    res_hard = analyze(hard, f"HARD subset (cos<{thresh})")

    out = ROOT / "talos_complementarity_results.json"
    out.write_text(json.dumps({"k": k, "thresh": thresh, "real_only": real_only,
                               "all": res_all, "hard": res_hard}, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--thresh", type=float, default=0.3)
    p.add_argument("--all-labels", action="store_true")
    args = p.parse_args()
    run(args.k, args.thresh, real_only=not args.all_labels)
