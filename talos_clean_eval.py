#!/usr/bin/env python3
"""
De-artifacted Talos benchmark.

Three artifacts in the original 72.7%/0% result:
  1. Chains MINED with the same token-bridge rule the retriever follows (circular).
  2. Judge labeled polysemy collisions ("corrections") as real_semantic.
  3. Query was the bare START TOKEN, not a realistic query — BM25's 0% terminal
     is guaranteed because the terminal doc never contains a single rare token.

This rebuilds the test fairly:
  - QUERY = full start-document text (realistic "find memories related to THIS note").
  - Corpus = all chain nodes (A/B/C across all chains) as the retrieval pool,
    deduped by id, optionally padded with other Talos notes.
  - GOLD = the terminal node C (the hop we claim only topology can bridge).
  - HARD SUBSET = chains whose start<->terminal embedding cosine < THRESH
    (genuinely embedding-disjoint; objective, no judge subjectivity).
  - Methods: bm25, dense, bm25+dense, bm25+path, oracle-iterative
    (retrieve start-doc, then bridge-doc B, union — the "reading" ceiling).

Reports terminal-recall@k on ALL chains vs the embedding-disjoint HARD subset.
"""
import json, sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hotpotqa_hybrid_eval as H
from path_coherent_retriever import build_idf_table
from embedding_bridge_retriever import embed_texts

CANDS = ROOT / "talos_semantic_chain_candidates_v2.jsonl"
JUDGE = ROOT / "talos_semantic_chain_omlx_judge_v2.jsonl"


def load_chains(real_only: bool):
    cands = {json.loads(l)["idx"]: json.loads(l) for l in open(CANDS)}
    judge = {json.loads(l)["idx"]: json.loads(l) for l in open(JUDGE)}
    chains = []
    for idx, c in cands.items():
        lab = judge.get(idx, {}).get("label", "unknown")
        if real_only and lab != "real_semantic":
            continue
        ex = c["excerpts"]
        if not all(kk in ex for kk in ("a", "b", "c")):
            continue
        ids = c["required_ids"]
        if len(ids) < 3:
            continue
        chains.append({
            "idx": idx, "label": lab,
            "a_id": ids[0], "b_id": ids[1], "c_id": ids[2],
            "a": ex["a"], "b": ex["b"], "c": ex["c"],
        })
    return chains


def run(k: int, thresh: float, real_only: bool):
    chains = load_chains(real_only)
    print(f"Loaded {len(chains)} chains (real_only={real_only})")

    # Build corpus = all distinct nodes.
    notes_by_id = {}
    for ch in chains:
        for tag, idk in [("a", "a_id"), ("b", "b_id"), ("c", "c_id")]:
            nid = ch[idk]
            notes_by_id.setdefault(nid, {"id": nid, "source": nid, "content": ch[tag]})
    notes = list(notes_by_id.values())
    corpus = H.build_index(notes)
    N = len(notes)
    idf_table = build_idf_table(corpus["df"], N)
    bm25_score_all, bm25_ranked = H.make_bm25_proper(corpus)
    path_scores = H.make_path_scores(corpus, idf_table, bm25_ranked)
    print(f"  corpus {N:,} distinct nodes")

    # Embed corpus + queries (start docs) + bridge docs.
    id_list = [n["id"] for n in notes]
    id_pos = {d: i for i, d in enumerate(id_list)}
    doc_vecs = embed_texts([n["content"][:512] for n in notes], batch_size=256)

    # start<->terminal cosine for hard-subset filtering
    for ch in chains:
        a, c = doc_vecs[id_pos[ch["a_id"]]], doc_vecs[id_pos[ch["c_id"]]]
        ch["ac_cos"] = float(a @ c)

    def eval_set(subset, tag):
        methods = ["bm25", "dense", "bm25+dense", "bm25+path", "oracle-iter"]
        rec = {m: 0 for m in methods}
        n = 0
        for ch in subset:
            n += 1
            gold = {ch["c_id"]}
            a_vec = doc_vecs[id_pos[ch["a_id"]]]
            query = ch["a"]

            bm25_s = bm25_score_all(query)
            bm25_top = sorted(bm25_s, key=lambda x: bm25_s[x], reverse=True)[:k]

            sims = doc_vecs @ a_vec
            order = np.argsort(-sims)[:max(k, 50)]
            dense_s = {id_list[j]: float(sims[j]) for j in order}
            dense_top = [id_list[j] for j in order[:k]]

            path_s = path_scores(query)

            def norm(d):
                if not d:
                    return {}
                mx = max(d.values()) or 1.0
                return {kk: v / mx for kk, v in d.items()}
            bn, dn, pn = norm(bm25_s), norm(dense_s), norm(path_s)

            def fuse(*w):
                agg = {}
                for sc, wt in w:
                    for kk, v in sc.items():
                        agg[kk] = agg.get(kk, 0.0) + wt * v
                return sorted(agg, key=lambda x: agg[x], reverse=True)[:k]

            bd = fuse((bn, 1.0), (dn, 1.0))
            bp = fuse((bn, 1.0), (pn, 0.6))

            # oracle-iterative: hop to bridge doc B, then dense-retrieve from B
            b_vec = doc_vecs[id_pos[ch["b_id"]]]
            bsims = doc_vecs @ b_vec
            border = np.argsort(-bsims)[:k]
            iter_top = list(dict.fromkeys([ch["b_id"]] +
                            [id_list[j] for j in border]))[:k]

            tops = {"bm25": bm25_top, "dense": dense_top, "bm25+dense": bd,
                    "bm25+path": bp, "oracle-iter": iter_top}
            for m, top in tops.items():
                rec[m] += int(bool(gold & set(top)))
        print(f"\n{tag} — n={n}, terminal-recall@{k}:")
        print(f"  {'method':<14}{'recall':>8}")
        for m in methods:
            print(f"  {m:<14}{100*rec[m]/max(n,1):>7.1f}%")
        return {m: rec[m] / max(n, 1) for m in methods}, n

    all_res, n_all = eval_set(chains, "ALL chains")
    hard = [ch for ch in chains if ch["ac_cos"] < thresh]
    easy = [ch for ch in chains if ch["ac_cos"] >= thresh]
    print(f"\n  embedding-disjoint hard subset (cos<{thresh}): {len(hard)}/{len(chains)}")
    hard_res, n_hard = eval_set(hard, f"HARD subset (cos<{thresh})") if hard else ({}, 0)
    easy_res, n_easy = eval_set(easy, f"EASY subset (cos>={thresh})") if easy else ({}, 0)

    out = ROOT / "talos_clean_results.json"
    out.write_text(json.dumps({
        "k": k, "thresh": thresh, "real_only": real_only,
        "n_all": n_all, "n_hard": n_hard, "n_easy": n_easy,
        "all": all_res, "hard": hard_res, "easy": easy_res,
        "ac_cos_mean": float(np.mean([c["ac_cos"] for c in chains])),
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--thresh", type=float, default=0.3)
    p.add_argument("--all-labels", action="store_true",
                   help="include weak/artifact chains, not just real_semantic")
    args = p.parse_args()
    run(args.k, args.thresh, real_only=not args.all_labels)
