#!/usr/bin/env python3
"""
HotpotQA fullwiki with a REAL dense retriever baseline — the reviewer-critical
fact-check. Everything so far beats only BM25; a reviewer's first question is
"did dense retrieval already beat BM25 here?" This answers it.

Adds to the hybrid harness:
  - dense:        Qwen3-Embedding-0.6B cosine retrieval (in-process via mlx_embeddings)
  - bm25+dense:   weighted fusion of BM25 ⊕ dense
  - +path:        BM25 ⊕ dense ⊕ path-coherent (does topology add over dense?)

Corpus embeddings are cached to disk so reruns are fast.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hotpotqa_hybrid_eval as H
from path_coherent_retriever import build_idf_table
from embedding_bridge_retriever import embed_texts

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")


def corpus_fingerprint(notes) -> str:
    h = hashlib.sha1()
    h.update(str(len(notes)).encode())
    for n in notes[:200]:
        h.update(n["id"].encode())
    return h.hexdigest()[:12]


def embed_corpus(notes) -> np.ndarray:
    fp = corpus_fingerprint(notes)
    cache = ROOT / f"hotpotqa_dense_cache_{fp}.npy"
    if cache.exists():
        print(f"  loaded cached embeddings {cache.name}")
        return np.load(cache)
    print(f"  embedding {len(notes):,} docs (first run, will cache)…")
    texts = [n["content"][:512] for n in notes]
    vecs = embed_texts(texts, batch_size=32)
    np.save(cache, vecs)
    print(f"  cached → {cache.name}")
    return vecs


def norm_scores(d):
    if not d:
        return {}
    mx = max(d.values()) or 1.0
    return {k: v / mx for k, v in d.items()}


def run(n_examples: int, n_bios: int, k: int, split: str, alpha: float, beta: float):
    print(f"Loading HotpotQA {split} — {n_examples} questions…")
    ds = load_dataset("hotpot_qa", "distractor", split=split).select(range(n_examples))

    notes_by_id, questions = {}, []
    for ex in ds:
        gold = set(ex["supporting_facts"]["title"])
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            notes_by_id.setdefault(title, {"id": title, "source": title,
                                           "content": " ".join(sents)})
        if gold.issubset(notes_by_id.keys()):
            questions.append({"q": ex["question"], "gold": gold,
                              "type": ex.get("type", "bridge")})
    bios = json.loads(H.BIO_CORPUS.read_text())[:n_bios]
    for b in bios:
        notes_by_id.setdefault(b["source"], {"id": b["source"], "source": b["source"],
                                             "content": b["content"]})
    notes = list(notes_by_id.values())
    corpus = H.build_index(notes)
    N = len(notes)
    idf_table = build_idf_table(corpus["df"], N)
    bm25_score_all, bm25_ranked = H.make_bm25_proper(corpus)
    path_scores = H.make_path_scores(corpus, idf_table, bm25_ranked)
    print(f"  corpus {N:,} docs, {len(questions)} questions")

    # Dense embeddings for the corpus.
    doc_vecs = embed_corpus(notes)           # (N, d), L2-normalized
    doc_ids = [n["id"] for n in notes]
    q_vecs = embed_texts([q["q"] for q in questions], batch_size=32)  # (Q, d)

    methods = ["bm25", "dense", "bm25+dense", "bm25+path", "bm25+dense+path", "routed_all"]
    res = {m: {"hits2": 0, "hits1": 0} for m in methods}
    total = 0

    for qi, item in enumerate(questions):
        q, gold, qt = item["q"], item["gold"], item["type"]
        total += 1

        bm25_s = bm25_score_all(q)
        bm25_top = sorted(bm25_s, key=lambda x: bm25_s[x], reverse=True)[:k]

        # dense cosine over whole corpus
        sims = doc_vecs @ q_vecs[qi]
        dense_order = np.argsort(-sims)[:max(k, 50)]
        dense_s = {doc_ids[j]: float(sims[j]) for j in dense_order}
        dense_top = [doc_ids[j] for j in dense_order[:k]]

        path_s = path_scores(q)

        bn, dn, pn = norm_scores(bm25_s), norm_scores(dense_s), norm_scores(path_s)

        def fuse(*weighted):
            agg = {}
            for scores, w in weighted:
                for kk, v in scores.items():
                    agg[kk] = agg.get(kk, 0.0) + w * v
            return sorted(agg, key=lambda x: agg[x], reverse=True)[:k]

        bd_top = fuse((bn, 1.0), (dn, beta))
        bp_top = fuse((bn, 1.0), (pn, alpha))
        bdp_top = fuse((bn, 1.0), (dn, beta), (pn, alpha))
        # routed: path only on bridge questions, dense always on
        pred = H.classify_qtype(q)
        routed_top = (fuse((bn, 1.0), (dn, beta), (pn, alpha)) if pred == "bridge"
                      else fuse((bn, 1.0), (dn, beta)))

        for m, top in [("bm25", bm25_top), ("dense", dense_top),
                       ("bm25+dense", bd_top), ("bm25+path", bp_top),
                       ("bm25+dense+path", bdp_top), ("routed_all", routed_top)]:
            res[m]["hits2"] += int(gold.issubset(set(top)))
            res[m]["hits1"] += int(bool(gold & set(top[:1])))

        if (qi + 1) % 25 == 0:
            print(f"  {qi+1}/{len(questions)} scored…")

    print(f"\nResults — {total} q, corpus {N:,} docs, k={k}, alpha={alpha}, beta={beta}\n")
    print(f"{'method':<18} {'both-gold@'+str(k):>14} {'any-gold@1':>12}")
    print("-" * 48)
    for m in methods:
        print(f"{m:<18} {100*res[m]['hits2']/total:>12.1f}%  {100*res[m]['hits1']/total:>10.1f}%")

    out = ROOT / "hotpotqa_dense_results.json"
    out.write_text(json.dumps({"corpus_size": N, "n": total, "k": k,
                               "alpha": alpha, "beta": beta, "results": res}, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--bios", type=int, default=30000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.6, help="path weight")
    p.add_argument("--beta", type=float, default=1.0, help="dense weight")
    p.add_argument("--split", default="validation")
    args = p.parse_args()
    run(args.n, args.bios, args.k, args.split, args.alpha, args.beta)
