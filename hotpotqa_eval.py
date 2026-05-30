#!/usr/bin/env python3
"""
HotpotQA distractor-setting evaluation for path-coherent retriever v6.

Task: given a question + 10 paragraphs (2 gold + 8 distractors), rank paragraphs
so that both supporting ones land in top-k. Measures supporting-doc recall@k.

Compares:
  - BM25 baseline
  - path-coherent v5 (legacy, n_cross <= 8 hard cutoff)
  - path-coherent v6 (IDF-weighted, max_hops=2)
"""
import json, math, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from path_coherent_retriever import candidate_bridges, build_idf_table, retrieve

# ── tokenizer ─────────────────────────────────────────────────────────────────
STOP = set(
    "able about above after again against all also although always among and any are "
    "because been before being below between both but can did does doing done down "
    "during each few for from further had has have having here how into its itself "
    "just like more most must need not now off only other our out over own same she "
    "should since some such than that the their them then there these they this those "
    "through too under until very was were what when where which while who will with "
    "would you your".split()
)

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return [t for t in tokens if t not in STOP and len(t) <= 20]


# ── corpus builder ─────────────────────────────────────────────────────────────
def build_corpus(paragraphs: list[tuple[str, list[str]]]) -> dict[str, Any]:
    """Convert HotpotQA context paragraphs into the retriever's note format."""
    notes = []
    for title, sentences in paragraphs:
        content = " ".join(sentences)
        nid = title  # use title as ID within one example
        notes.append({"id": nid, "source": title, "content": content})

    note_tokens: dict[str, set[str]] = {
        n["id"]: set(tokenize(n["content"])) for n in notes
    }
    df: dict[str, int] = defaultdict(int)
    postings: dict[str, list[str]] = defaultdict(list)
    token_sources: dict[str, set[str]] = defaultdict(set)

    for n in notes:
        seen = set()
        for t in note_tokens[n["id"]]:
            df[t] += 1
            if t not in seen:
                postings[t].append(n["id"])
                seen.add(t)
            token_sources[t].add(n["source"])

    return {
        "notes": notes,
        "note_by_id": {n["id"]: n for n in notes},
        "note_tokens": note_tokens,
        "df": dict(df),
        "postings": dict(postings),
        "token_sources": dict(token_sources),
    }


def bm25_fn(query: str, corpus: dict, top_k: int) -> list[str]:
    q_toks = tokenize(query)
    N = len(corpus["notes"])
    df = corpus["df"]
    postings = corpus["postings"]
    scores: dict[str, float] = {}
    for t in set(q_toks):
        if t not in postings:
            continue
        idf = math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)
        for nid in postings[t]:
            scores[nid] = scores.get(nid, 0) + idf
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]


# ── evaluation loop ────────────────────────────────────────────────────────────
def run_eval(split: str = "validation", n_examples: int = 500) -> None:
    print(f"Loading HotpotQA {split} (distractor)…")
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    if n_examples:
        ds = ds.select(range(min(n_examples, len(ds))))

    results: dict[str, dict[str, int]] = {
        "bm25": {"hits2": 0, "hits1": 0, "total": 0},
        "v5":   {"hits2": 0, "hits1": 0, "total": 0},
        "v6":   {"hits2": 0, "hits1": 0, "total": 0},
    }

    for ex in ds:
        question = ex["question"]
        gold_titles: set[str] = set(ex["supporting_facts"]["title"])
        paragraphs: list[tuple[str, list[str]]] = list(
            zip(ex["context"]["title"], ex["context"]["sentences"])
        )

        corpus = build_corpus(paragraphs)
        N = len(corpus["notes"])

        idf_table = build_idf_table(corpus["df"], N)
        note_by_id = corpus["note_by_id"]
        note_tokens = corpus["note_tokens"]
        postings = corpus["postings"]
        df = corpus["df"]
        token_sources = corpus["token_sources"]

        def _bm25(q: str, k: int) -> list[str]:
            return bm25_fn(q, corpus, k)

        # BM25 baseline
        bm25_top = bm25_fn(question, corpus, 10)
        hit2_bm25 = gold_titles.issubset(set(bm25_top[:2]))
        hit1_bm25 = bool(gold_titles & set(bm25_top[:1]))

        # path v5 (legacy: no idf_table, max_hops=3 but only 10 docs so 2-hop effectively)
        v5_top = retrieve(
            question, tokenize, _bm25,
            note_by_id, note_tokens, postings, df, token_sources,
            top_k=10, anchor_k=10, branch_k=10, bridge_k=3,
            idf_table=None, max_hops=2,
        )
        hit2_v5 = gold_titles.issubset(set(v5_top[:2]))
        hit1_v5 = bool(gold_titles & set(v5_top[:1]))

        # path v6 (IDF-weighted, max_hops=2)
        v6_top = retrieve(
            question, tokenize, _bm25,
            note_by_id, note_tokens, postings, df, token_sources,
            top_k=10, anchor_k=10, branch_k=10, bridge_k=3,
            idf_table=idf_table, max_hops=2,
        )
        hit2_v6 = gold_titles.issubset(set(v6_top[:2]))
        hit1_v6 = bool(gold_titles & set(v6_top[:1]))

        for key, h2, h1 in [
            ("bm25", hit2_bm25, hit1_bm25),
            ("v5",   hit2_v5,   hit1_v5),
            ("v6",   hit2_v6,   hit1_v6),
        ]:
            results[key]["total"] += 1
            results[key]["hits2"] += int(h2)
            results[key]["hits1"] += int(h1)

    print(f"\nResults on {results['bm25']['total']} HotpotQA distractor examples\n")
    print(f"{'method':<12} {'both-gold@2':>12} {'any-gold@1':>12}")
    print("-" * 40)
    for method in ("bm25", "v5", "v6"):
        r = results[method]
        t = r["total"]
        print(
            f"{method:<12} {100*r['hits2']/t:>10.1f}%  {100*r['hits1']/t:>10.1f}%"
        )

    out = ROOT / "hotpotqa_eval_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="examples to evaluate (0=all)")
    p.add_argument("--split", default="validation")
    args = p.parse_args()
    run_eval(args.split, args.n)
