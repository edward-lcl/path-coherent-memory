#!/usr/bin/env python3
"""
HotpotQA large-shared-corpus ("fullwiki-style") evaluation.

Unlike hotpotqa_eval.py (per-example 10-paragraph distractor setting, where BM25
wins because the corpus is tiny), this pools every gold + distractor paragraph
from N eval questions into ONE shared corpus, padded with 30K Wikipedia bios as
extra distractors. Each question is retrieved against the whole corpus.

This is the regime where the path-coherent hypothesis should pay off: BM25's
per-document term scoring degrades as the corpus grows and lexical collisions
multiply, while bridge topology can still connect the two gold hops through a
rare shared entity.

Metric: both-gold@k recall and any-gold@1.
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
from path_coherent_retriever import build_idf_table, retrieve

BIO_CORPUS = ROOT / "wikipedia_bio_corpus.json"

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


def build_index(notes: list[dict]) -> dict[str, Any]:
    note_tokens = {n["id"]: set(tokenize(n["content"])) for n in notes}
    df: dict[str, int] = defaultdict(int)
    postings: dict[str, list[str]] = defaultdict(list)
    token_sources: dict[str, set[str]] = defaultdict(set)
    for n in notes:
        for t in note_tokens[n["id"]]:
            df[t] += 1
            postings[t].append(n["id"])
            token_sources[t].add(n["source"])
    return {
        "notes": notes,
        "note_by_id": {n["id"]: n for n in notes},
        "note_tokens": note_tokens,
        "df": dict(df),
        "postings": dict(postings),
        "token_sources": dict(token_sources),
    }


def make_bm25(corpus: dict):
    N = len(corpus["notes"])
    df = corpus["df"]
    postings = corpus["postings"]

    def bm25(query: str, top_k: int) -> list[str]:
        q_toks = tokenize(query)
        scores: dict[str, float] = {}
        for t in set(q_toks):
            if t not in postings:
                continue
            idf = math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)
            if idf <= 0:
                continue
            for nid in postings[t]:
                scores[nid] = scores.get(nid, 0) + idf
        return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    return bm25


def run(n_examples: int, n_bios: int, k: int, split: str) -> None:
    print(f"Loading HotpotQA {split} (distractor) — {n_examples} questions…")
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    ds = ds.select(range(min(n_examples, len(ds))))

    # Pool all gold+distractor paragraphs into one shared corpus, deduped by title.
    notes_by_id: dict[str, dict] = {}
    questions: list[dict] = []
    for ex in ds:
        gold = set(ex["supporting_facts"]["title"])
        for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title not in notes_by_id:
                notes_by_id[title] = {
                    "id": title,
                    "source": title,
                    "content": " ".join(sentences),
                }
        # only keep questions whose gold docs are actually present
        if gold.issubset(notes_by_id.keys()):
            questions.append({"q": ex["question"], "gold": gold})

    print(f"  pooled {len(notes_by_id):,} unique HotpotQA paragraphs")

    # Pad with Wikipedia bio distractors.
    bios = json.loads(BIO_CORPUS.read_text())[:n_bios]
    added = 0
    for b in bios:
        if b["source"] not in notes_by_id:
            notes_by_id[b["source"]] = {
                "id": b["source"],
                "source": b["source"],
                "content": b["content"],
            }
            added += 1
    print(f"  added {added:,} bio distractors → corpus size {len(notes_by_id):,}")

    corpus = build_index(list(notes_by_id.values()))
    N = len(corpus["notes"])
    idf_table = build_idf_table(corpus["df"], N)
    bm25 = make_bm25(corpus)

    args = dict(
        note_by_id=corpus["note_by_id"],
        note_tokens=corpus["note_tokens"],
        postings=corpus["postings"],
        df=corpus["df"],
        token_sources=corpus["token_sources"],
        top_k=k, anchor_k=10, branch_k=10, bridge_k=3, max_hops=2,
    )

    res = {m: {"hits2": 0, "hits1": 0} for m in ("bm25", "v6")}
    total = 0
    for qi, item in enumerate(questions):
        q, gold = item["q"], item["gold"]
        total += 1

        bm25_top = bm25(q, k)
        res["bm25"]["hits2"] += int(gold.issubset(set(bm25_top[:k])))
        res["bm25"]["hits1"] += int(bool(gold & set(bm25_top[:1])))

        v6_top = retrieve(q, tokenize, bm25, idf_table=idf_table, **args)
        res["v6"]["hits2"] += int(gold.issubset(set(v6_top[:k])))
        res["v6"]["hits1"] += int(bool(gold & set(v6_top[:1])))

        if (qi + 1) % 25 == 0:
            print(f"  {qi+1}/{len(questions)} done…")

    print(f"\nLarge-shared-corpus results — {total} questions, corpus {N:,} docs, k={k}\n")
    print(f"{'method':<8} {'both-gold@'+str(k):>14} {'any-gold@1':>12}")
    print("-" * 38)
    for m in ("bm25", "v6"):
        print(f"{m:<8} {100*res[m]['hits2']/total:>12.1f}%  {100*res[m]['hits1']/total:>10.1f}%")

    out = ROOT / "hotpotqa_fullwiki_results.json"
    out.write_text(json.dumps(
        {"corpus_size": N, "n_questions": total, "k": k, "results": res}, indent=2
    ))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="HotpotQA questions")
    p.add_argument("--bios", type=int, default=30000, help="bio distractors to add")
    p.add_argument("--k", type=int, default=10, help="recall@k")
    p.add_argument("--split", default="validation")
    args = p.parse_args()
    run(args.n, args.bios, args.k, args.split)
