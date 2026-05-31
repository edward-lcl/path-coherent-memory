#!/usr/bin/env python3
"""
3-way complementarity on agent runtime session chains.

Same analysis as talos_complementarity_eval.py but on the cross-session
concept chains mined from OCPlatform main-agent sessions — the "multiple
threads pulling on shared context that should compound but don't" regime.

Gold = terminal chunk C (cross-session, concept-disjoint from A).
Corpus = all 867 session chunks (the full retrieval pool).
Modes:
  dense   — embed query A, retrieve top-k
  path    — token-path traversal (bridge tokens)
  oracle-iter — embed the bridge B directly (ceiling: the reader knew to look there)
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hotpotqa_hybrid_eval as H
from path_coherent_retriever import build_idf_table
from embedding_bridge_retriever import embed_texts

CHAINS_FILE = ROOT / "agent_session_chains.jsonl"
NODES_FILE  = ROOT / "agent_session_nodes.json"
VECS_FILE   = ROOT / "agent_session_embeddings.npy"


def run(k: int, thresh: float, real_only: bool):
    chains = [json.loads(l) for l in CHAINS_FILE.read_text().splitlines() if l.strip()]
    nodes  = json.loads(NODES_FILE.read_text())
    vecs   = np.load(VECS_FILE)
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    V = vecs / norms

    id2idx = {n["id"]: i for i, n in enumerate(nodes)}
    print(f"Loaded {len(chains)} chains, {len(nodes)} nodes, k={k}")

    # Hard subset: chains where sim_ac < thresh (embedding-disjoint)
    hard = [c for c in chains if c["sim_ac"] < thresh and c["cross_session"]]
    all_cs = [c for c in chains if c["cross_session"]]
    print(f"All cross-session: {len(all_cs)}")
    print(f"Hard subset (sim_ac<{thresh}): {len(hard)}")

    for label, subset in [("ALL cross-session", all_cs), (f"HARD (sim_ac<{thresh})", hard)]:
        if not subset:
            continue
        corpus = H.build_index(nodes)
        N = len(nodes)
        idf_table = build_idf_table(corpus["df"], N)
        _, bm25_ranked = H.make_bm25_proper(corpus)
        path_scores = H.make_path_scores(corpus, idf_table, bm25_ranked)
        id_list = [n["id"] for n in nodes]

        # Embed queries (A chunks) and bridge docs (B)
        a_texts  = [nodes[id2idx[c["a_id"]]]["content"][:512] for c in subset if c["a_id"] in id2idx and c["c_id"] in id2idx]
        b_texts  = [nodes[id2idx[c["b_id"]]]["content"][:512] for c in subset if c["a_id"] in id2idx and c["c_id"] in id2idx]
        valid    = [c for c in subset if c["a_id"] in id2idx and c["c_id"] in id2idx]

        a_vecs = embed_texts(a_texts, batch_size=256)
        b_vecs = embed_texts(b_texts, batch_size=256)

        MODES = ["dense", "path", "oracle-iter"]
        hitmaps = []
        for i, ch in enumerate(valid):
            gold = ch["c_id"]
            # dense: retrieve from query A
            sims = V @ a_vecs[i]
            dense_hit = gold in set(id_list[j] for j in np.argsort(-sims)[:k])
            # path: token-path from query A text
            ps = path_scores(nodes[id2idx[ch["a_id"]]]["content"])
            path_hit = gold in set(sorted(ps, key=lambda x: ps[x], reverse=True)[:k])
            # oracle-iter: retrieve from bridge B (the reader who found B can now get C)
            ssims = V @ b_vecs[i]
            oracle_hit = gold in set(id_list[j] for j in np.argsort(-ssims)[:k])
            hitmaps.append({"dense": dense_hit, "path": path_hit, "oracle-iter": oracle_hit})

        n = len(hitmaps)
        rec  = {m: sum(h[m] for h in hitmaps) / n for m in MODES}
        union = sum(any(h[m] for m in MODES) for h in hitmaps) / n
        best  = max(rec.values())
        excl  = {m: sum(h[m] and not any(h[o] for o in MODES if o!=m) for h in hitmaps)/n for m in MODES}
        jacs  = {}
        for i2, m1 in enumerate(MODES):
            for m2 in MODES[i2+1:]:
                inter = sum(h[m1] and h[m2] for h in hitmaps)
                uni   = sum(h[m1] or  h[m2] for h in hitmaps)
                jacs[f"{m1}&{m2}"] = inter/uni if uni else 0.0

        print(f"\n=== {label} (n={n}) ===")
        print("  per-method: " + "  ".join(f"{m}={100*rec[m]:.1f}%" for m in MODES))
        print(f"  UNION={100*union:.1f}%  best-single={100*best:.1f}%  lift=+{100*(union-best):.1f}pp")
        print("  exclusive: " + "  ".join(f"{m}={100*excl[m]:.1f}%" for m in MODES))
        print("  Jaccard: " + "  ".join(f"{k2}={v:.2f}" for k2,v in jacs.items()))

        out = ROOT / f"agent_session_results_{'hard' if 'HARD' in label else 'all'}.json"
        out.write_text(json.dumps({"label": label, "n": n, "k": k, "thresh": thresh,
                                   "recall": rec, "union": union, "best": best,
                                   "exclusive": excl, "jaccard": jacs}, indent=2))
        print(f"  Saved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--thresh", type=float, default=0.35)
    a = p.parse_args()
    run(a.k, a.thresh, False)
