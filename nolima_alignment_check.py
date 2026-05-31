#!/usr/bin/env python3
"""
E12 — NoLiMa Alignment Check

NoLiMa (2025) shows long-context models collapse when the needle has
*no lexical overlap* with the query. Our concept-bridge condition is
exactly this: start and terminal share no vocabulary and are embedding-disjoint.

This script verifies our Talos embedding-disjoint chains satisfy NoLiMa's
condition (near-zero lexical overlap between query and terminal), connecting
our result to their framework as corroboration.

Measures:
- Unigram overlap (Jaccard) between anchor and terminal texts
- BM25 score of terminal given anchor as query
- Confirms chains are in the same "no lexical overlap" regime as NoLiMa
"""
from __future__ import annotations
import json, re, math
from collections import Counter
from pathlib import Path
import numpy as np

RFM = Path(__file__).parent

STOP = set("""
a an the and or but in on at to of for with by from is are was were be been being
have has had do does did will would could should may might must can shall this that
these those it its i me my we our you your he she his her they them their what which
who how when where why not no nor so yet both either also just only even though
although because since while until unless if as such like
""".split())


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'[a-z]{3,}', text.lower())
            if w.lower() not in STOP]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 0.0
    return len(sa & sb) / len(sa | sb)


def bm25_score(query_toks: list[str], doc_toks: list[str],
               corpus_df: dict[str, int], N: int,
               k1: float = 1.2, b: float = 0.75, avgdl: float = 50.0) -> float:
    score = 0.0
    dl = len(doc_toks)
    for t in set(query_toks):
        tf = doc_toks.count(t)
        if tf == 0: continue
        df = corpus_df.get(t, 0)
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return score


def main():
    # Load embedding-disjoint chains
    valid_idxs: set[int] = set()
    jf = RFM / "talos_semantic_chain_omlx_judge_v2.jsonl"
    if jf.exists():
        with open(jf) as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try:
                    j = json.loads(line)
                    if j.get("label") == "real_semantic":
                        valid_idxs.add(j["idx"])
                except Exception:
                    pass

    chains = []
    with open(RFM / "talos_semantic_chain_candidates_v2.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try:
                c = json.loads(line)
                if valid_idxs and c.get("idx") not in valid_idxs: continue
                ex = c.get("excerpts", {})
                if ex.get("a") and ex.get("c"):
                    chains.append({"a": ex["a"], "b": ex.get("b",""), "c": ex["c"],
                                   "idx": c.get("idx"), "sim_ac": c.get("sim_ac", 1.0)})
            except Exception:
                pass

    # Compute embedding-disjoint filter
    print(f"Loaded {len(chains)} chains")
    sys.path.insert(0, str(RFM))
    from embedding_bridge_retriever import embed_texts
    import numpy as np

    a_vecs = np.array(embed_texts([c["a"][:512] for c in chains], batch_size=64), dtype=np.float32)
    c_vecs = np.array(embed_texts([c["c"][:512] for c in chains], batch_size=64), dtype=np.float32)
    an = np.linalg.norm(a_vecs,axis=1,keepdims=True); an[an==0]=1; a_vecs/=an
    cn = np.linalg.norm(c_vecs,axis=1,keepdims=True); cn[cn==0]=1; c_vecs/=cn
    ac_cos = (a_vecs * c_vecs).sum(axis=1)
    for i,c in enumerate(chains):
        c["sim_ac"] = float(ac_cos[i])

    disjoint = [c for c in chains if c["sim_ac"] < 0.3]
    easy     = [c for c in chains if c["sim_ac"] >= 0.3]
    print(f"Embedding-disjoint (cos<0.3): {len(disjoint)}  |  Easy (cos>=0.3): {len(easy)}")

    # Build corpus df for BM25
    all_toks = []
    for c in chains:
        all_toks.extend(tokenize(c["a"]) + tokenize(c["b"]) + tokenize(c["c"]))
    corpus_df = Counter()
    for c in chains:
        for tok in set(tokenize(c["a"]) + tokenize(c["b"]) + tokenize(c["c"])):
            corpus_df[tok] += 1
    N_docs = len(chains) * 3
    avgdl = sum(len(tokenize(c["a"]) + tokenize(c["c"])) for c in chains) / max(len(chains), 1)

    def analyze(subset: list[dict], label: str) -> dict:
        jaccards, bm25s, overlaps = [], [], []
        for c in subset:
            a_toks = tokenize(c["a"])
            c_toks = tokenize(c["c"])
            j = jaccard(a_toks, c_toks)
            bm = bm25_score(a_toks, c_toks, corpus_df, N_docs, avgdl=avgdl)
            ov = len(set(a_toks) & set(c_toks))
            jaccards.append(j); bm25s.append(bm); overlaps.append(ov)
        import statistics
        return {
            "n": len(subset),
            "jaccard_mean": round(statistics.mean(jaccards), 4) if jaccards else 0,
            "jaccard_median": round(statistics.median(jaccards), 4) if jaccards else 0,
            "jaccard_zero_pct": round(100*sum(j==0 for j in jaccards)/max(len(jaccards),1), 1),
            "bm25_mean": round(statistics.mean(bm25s), 3) if bm25s else 0,
            "bm25_zero_pct": round(100*sum(b==0 for b in bm25s)/max(len(bm25s),1), 1),
            "shared_tokens_mean": round(statistics.mean(overlaps), 2) if overlaps else 0,
            "shared_tokens_zero_pct": round(100*sum(o==0 for o in overlaps)/max(len(overlaps),1), 1),
        }

    print("\n=== LEXICAL OVERLAP ANALYSIS ===")
    print("(Connecting to NoLiMa: long-context fails when query↔needle have no lexical overlap)")

    d_stats = analyze(disjoint, "Embedding-disjoint (cos<0.3)")
    e_stats = analyze(easy, "Easy (cos>=0.3)")

    for label, stats, subset in [
        ("EMBEDDING-DISJOINT chains (cos<0.3)", d_stats, disjoint),
        ("EASY chains (cos>=0.3)", e_stats, easy),
    ]:
        print(f"\n  {label} (n={stats['n']}):")
        print(f"    Jaccard(anchor,terminal): mean={stats['jaccard_mean']} "
              f"median={stats['jaccard_median']} zero={stats['jaccard_zero_pct']}%")
        print(f"    BM25(anchor→terminal):    mean={stats['bm25_mean']} "
              f"zero={stats['bm25_zero_pct']}%")
        print(f"    Shared tokens:            mean={stats['shared_tokens_mean']} "
              f"zero={stats['shared_tokens_zero_pct']}%")

    print("\n  NoLiMa alignment verdict:")
    pct_zero_jac = d_stats["jaccard_zero_pct"]
    pct_zero_bm25 = d_stats["bm25_zero_pct"]
    if pct_zero_jac > 50 or pct_zero_bm25 > 50:
        print(f"  ✓ Embedding-disjoint chains ARE in NoLiMa 'no-lexical-overlap' regime")
        print(f"    ({pct_zero_jac}% have Jaccard=0, {pct_zero_bm25}% have BM25=0)")
        print(f"    → cite NoLiMa as corroborating evidence for E1 results")
    else:
        print(f"  ✗ Chains have partial lexical overlap — NoLiMa framing is approximate, not exact")
        print(f"    ({pct_zero_jac}% Jaccard=0, {pct_zero_bm25}% BM25=0)")

    out = {"disjoint": d_stats, "easy": e_stats,
           "nolima_aligned": pct_zero_jac > 50 or pct_zero_bm25 > 50}
    (RFM / "nolima_alignment_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> nolima_alignment_results.json")

    # Sample 3 disjoint chains to show the zero-overlap condition concretely
    if disjoint:
        print("\n  Sample concept-bridge chains (no lexical overlap):")
        for c in disjoint[:3]:
            a_toks = set(tokenize(c["a"]))
            c_toks = set(tokenize(c["c"]))
            shared = a_toks & c_toks
            print(f"    sim_ac={c['sim_ac']:.3f} shared={shared or '{}'}")
            print(f"    A: {c['a'][:90]}")
            print(f"    C: {c['c'][:90]}")


import sys
if __name__ == "__main__":
    main()
