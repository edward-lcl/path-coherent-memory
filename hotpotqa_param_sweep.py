#!/usr/bin/env python3
"""
Sweep path-coherent params (branch_k, bridge_k) on HotpotQA fullwiki, measuring
the FUSED both-gold@10 (weighted alpha) — the metric that matters for the hybrid.

Builds the 33K-doc corpus once, then evaluates each param combo. path defaults
(branch_k=10, bridge_k=3) came from the Levi personal corpus; this finds the
HotpotQA optimum.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hotpotqa_hybrid_eval as H
from path_coherent_retriever import build_idf_table

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")


def main(n_examples: int, n_bios: int, k: int, alpha: float, split: str) -> None:
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
    corpus = H.build_index(list(notes_by_id.values()))
    N = len(corpus["notes"])
    idf_table = build_idf_table(corpus["df"], N)
    bm25_score_all, bm25_ranked = H.make_bm25_proper(corpus)
    print(f"  corpus {N:,} docs, {len(questions)} questions\n")

    # Pre-compute BM25 scores per question (param-independent).
    bm25_cache = [(q["q"], q["gold"], q["type"], bm25_score_all(q["q"])) for q in questions]

    print(f"{'branch_k':>8} {'bridge_k':>8} {'both@10(fused)':>14} {'routed_clf':>11}")
    print("-" * 46)
    grid = [(b, br) for b in (6, 10, 15, 20) for br in (2, 3, 4, 5)]
    best = None
    results = []
    for branch_k, bridge_k in grid:
        path_scores = H.make_path_scores(corpus, idf_table, bm25_ranked)
        hits_fused = hits_routed = 0
        for q, gold, qt, bm25_s in bm25_cache:
            ps = path_scores(q, anchor_k=10, branch_k=branch_k, bridge_k=bridge_k)
            wt = H.weighted_fuse(bm25_s, ps, alpha, k)
            hits_fused += int(gold.issubset(set(wt)))
            bm25_top = sorted(bm25_s, key=lambda x: bm25_s[x], reverse=True)[:k]
            pred = H.classify_qtype(q)
            routed = wt if pred == "bridge" else bm25_top
            hits_routed += int(gold.issubset(set(routed)))
        n = len(bm25_cache)
        f_pct, r_pct = 100*hits_fused/n, 100*hits_routed/n
        results.append({"branch_k": branch_k, "bridge_k": bridge_k,
                        "fused": f_pct, "routed_clf": r_pct})
        print(f"{branch_k:>8} {bridge_k:>8} {f_pct:>13.1f}% {r_pct:>10.1f}%")
        if best is None or r_pct > best["routed_clf"]:
            best = results[-1]

    print(f"\nBest: branch_k={best['branch_k']} bridge_k={best['bridge_k']} "
          f"routed_clf={best['routed_clf']:.1f}%")
    out = ROOT / "hotpotqa_param_sweep_results.json"
    out.write_text(json.dumps({"corpus_size": N, "alpha": alpha,
                               "grid": results, "best": best}, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--bios", type=int, default=30000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--split", default="validation")
    args = p.parse_args()
    main(args.n, args.bios, args.k, args.alpha, args.split)
