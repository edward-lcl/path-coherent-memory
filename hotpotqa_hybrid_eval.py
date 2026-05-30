#!/usr/bin/env python3
"""
HotpotQA large-shared-corpus iteration harness.

Goes beyond hotpotqa_fullwiki_eval.py (which ran stock v6 and tied BM25). Adds:

  1. Proper BM25 (k1, b) — fairer + stronger baseline than the bare IDF-sum.
  2. path_scores() — exposes raw per-doc path-coherent scores (not the merged
     slot-split list), so we can FUSE instead of replace.
  3. RRF hybrid — reciprocal-rank fusion of BM25 ⊕ path. Keeps BM25's anchor
     high, lets path only rescue the buried second hop.
  4. Weighted-add hybrid — normalized BM25 + alpha * normalized path.
  5. Failure-mode diagnostic — per question type (bridge/comparison), and for
     each missed gold doc, whether path-only would have recovered it.

Run: python3 hotpotqa_hybrid_eval.py --n 200 --bios 30000 --k 10
"""
import json, math, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from path_coherent_retriever import build_idf_table, candidate_bridges

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
    note_tokens = {n["id"]: tokenize(n["content"]) for n in notes}  # list, keep TF
    note_tokset = {nid: set(toks) for nid, toks in note_tokens.items()}
    df: dict[str, int] = defaultdict(int)
    postings: dict[str, list[str]] = defaultdict(list)
    token_sources: dict[str, set[str]] = defaultdict(set)
    doc_len: dict[str, int] = {}
    for n in notes:
        nid = n["id"]
        toks = note_tokens[nid]
        doc_len[nid] = len(toks)
        for t in note_tokset[nid]:
            df[t] += 1
            postings[t].append(nid)
            token_sources[t].add(n["source"])
    # term frequencies per doc
    tf: dict[str, dict[str, int]] = {}
    for nid, toks in note_tokens.items():
        d: dict[str, int] = defaultdict(int)
        for t in toks:
            d[t] += 1
        tf[nid] = dict(d)
    avgdl = sum(doc_len.values()) / max(len(doc_len), 1)
    return {
        "notes": notes,
        "note_by_id": {n["id"]: n for n in notes},
        "note_tokens": note_tokset,
        "tf": tf,
        "df": dict(df),
        "postings": dict(postings),
        "token_sources": dict(token_sources),
        "doc_len": doc_len,
        "avgdl": avgdl,
    }


def make_bm25_proper(corpus: dict, k1: float = 1.5, b: float = 0.75):
    """Okapi BM25 with TF saturation + length normalization."""
    N = len(corpus["notes"])
    df = corpus["df"]
    postings = corpus["postings"]
    tf = corpus["tf"]
    doc_len = corpus["doc_len"]
    avgdl = corpus["avgdl"]
    idf = {t: math.log((N - d + 0.5) / (d + 0.5) + 1) for t, d in df.items()}

    def score_all(query: str) -> dict[str, float]:
        q_toks = set(tokenize(query))
        scores: dict[str, float] = {}
        for t in q_toks:
            if t not in postings:
                continue
            t_idf = idf[t]
            for nid in postings[t]:
                f = tf[nid].get(t, 0)
                denom = f + k1 * (1 - b + b * doc_len[nid] / avgdl)
                scores[nid] = scores.get(nid, 0.0) + t_idf * (f * (k1 + 1)) / denom
        return scores

    def ranked(query: str, top_k: int) -> list[str]:
        s = score_all(query)
        return sorted(s, key=lambda x: s[x], reverse=True)[:top_k]

    return score_all, ranked


def make_path_scores(corpus: dict, idf_table: dict[str, float], bm25_ranked: Callable):
    """
    Path-coherent scoring that returns {doc_id: score} for ALL reached docs.
    2-hop: anchors (BM25 top) + bridge terminals. Mirrors v6 retrieve() math
    but exposes raw scores for fusion instead of slot-splitting.
    """
    note_by_id = corpus["note_by_id"]
    note_tokens = corpus["note_tokens"]
    postings = corpus["postings"]
    df = corpus["df"]
    token_sources = corpus["token_sources"]

    def path_scores(query: str, anchor_k: int = 10, branch_k: int = 6,
                    bridge_k: int = 5) -> dict[str, float]:
        query_toks = set(tokenize(query))
        scores: dict[str, float] = {}
        for rank, a_id in enumerate(bm25_ranked(query, anchor_k)):
            a_src = note_by_id[a_id]["source"]
            a_toks = note_tokens[a_id]
            scores[a_id] = max(scores.get(a_id, 0), 1.0 + 0.05 * (anchor_k - rank))
            bridges = candidate_bridges(
                a_id, a_src, a_toks, query_toks, token_sources, df, bridge_k, idf_table
            )
            for t1 in bridges:
                for b_id in postings.get(t1, [])[:branch_k]:
                    if b_id == a_id:
                        continue
                    b_score = 0.5 + 0.05 * (anchor_k - rank) + 0.1 * idf_table.get(t1, 0.0)
                    scores[b_id] = max(scores.get(b_id, 0), b_score)
        return scores

    return path_scores


def rrf_fuse(*rankings: list[str], k: int = 60, top_k: int = 10) -> list[str]:
    """Reciprocal-rank fusion of multiple ranked lists."""
    agg: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            agg[doc] += 1.0 / (k + rank + 1)
    return sorted(agg, key=lambda x: agg[x], reverse=True)[:top_k]


def weighted_fuse(bm25_scores: dict[str, float], path_scores: dict[str, float],
                  alpha: float, top_k: int) -> list[str]:
    """Normalized BM25 + alpha * normalized path."""
    def norm(d):
        if not d:
            return {}
        mx = max(d.values()) or 1.0
        return {k: v / mx for k, v in d.items()}
    bn, pn = norm(bm25_scores), norm(path_scores)
    agg: dict[str, float] = defaultdict(float)
    for k_, v in bn.items():
        agg[k_] += v
    for k_, v in pn.items():
        agg[k_] += alpha * v
    return sorted(agg, key=lambda x: agg[x], reverse=True)[:top_k]


def qtype(ex) -> str:
    return ex.get("type", "unknown")  # 'bridge' or 'comparison' in HotpotQA


# Lexical giveaways for comparison questions. Bridge is the default.
# Tuned for PRECISION on the comparison class: a misroute that skips fusion on a
# bridge question costs the full +8pp lane, while applying fusion to a comparison
# costs only ~1pp. So only fire 'comparison' on strong, unambiguous signals.
_CMP_PATTERNS = [
    # comparative "who/which is older/larger/..." — needs the comparative cue word
    re.compile(r"\b(?:older|younger|larger|smaller|bigger|taller|shorter|longer|"
               r"oldest|youngest|largest|smallest|earliest|latest)\b", re.I),
    re.compile(r"\bborn first\b|\bcame first\b|\bwho was born\b", re.I),
    # "both" anywhere with a conjunction — strong comparison marker
    re.compile(r"\bboth\b", re.I),
    re.compile(r"\bdo(?:es)? both\b", re.I),
    re.compile(r"\bare (?:both|either|neither)\b", re.I),
    # shared-attribute comparisons
    re.compile(r"\b(?:share|shared)\b", re.I),
    re.compile(r"\bin common\b", re.I),
    re.compile(r"\bsame\b", re.I),
]


def classify_qtype(question: str) -> str:
    """Heuristic bridge/comparison classifier. No gold label used."""
    q = question.strip()
    for p in _CMP_PATTERNS:
        if p.search(q):
            return "comparison"
    return "bridge"


def run(n_examples: int, n_bios: int, k: int, split: str, alpha: float) -> None:
    print(f"Loading HotpotQA {split} (distractor) — {n_examples} questions…")
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    ds = ds.select(range(min(n_examples, len(ds))))

    notes_by_id: dict[str, dict] = {}
    questions: list[dict] = []
    for ex in ds:
        gold = set(ex["supporting_facts"]["title"])
        for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title not in notes_by_id:
                notes_by_id[title] = {"id": title, "source": title,
                                      "content": " ".join(sentences)}
        if gold.issubset(notes_by_id.keys()):
            questions.append({"q": ex["question"], "gold": gold, "type": qtype(ex)})
    print(f"  pooled {len(notes_by_id):,} unique HotpotQA paragraphs")

    bios = json.loads(BIO_CORPUS.read_text())[:n_bios]
    added = 0
    for bdoc in bios:
        if bdoc["source"] not in notes_by_id:
            notes_by_id[bdoc["source"]] = {"id": bdoc["source"], "source": bdoc["source"],
                                           "content": bdoc["content"]}
            added += 1
    print(f"  added {added:,} bio distractors → corpus size {len(notes_by_id):,}")

    corpus = build_index(list(notes_by_id.values()))
    N = len(corpus["notes"])
    idf_table = build_idf_table(corpus["df"], N)
    bm25_score_all, bm25_ranked = make_bm25_proper(corpus)
    path_scores = make_path_scores(corpus, idf_table, bm25_ranked)

    methods = ["bm25", "path_only", "rrf", "weighted", "routed", "routed_clf"]
    clf_correct = 0  # classifier vs gold type
    res = {m: {"hits2": 0, "hits1": 0} for m in methods}
    # diagnostics
    by_type = {"bridge": defaultdict(lambda: {"hits2": 0, "n": 0}),
               "comparison": defaultdict(lambda: {"hits2": 0, "n": 0})}
    bm25_missed_recoverable = 0   # BM25 missed a gold doc, path found it
    bm25_missed_total = 0
    total = 0

    for qi, item in enumerate(questions):
        q, gold = item["q"], item["gold"]
        qt = item["type"] if item["type"] in by_type else "bridge"
        total += 1

        bm25_s = bm25_score_all(q)
        bm25_rank = sorted(bm25_s, key=lambda x: bm25_s[x], reverse=True)
        bm25_top = bm25_rank[:k]

        path_s = path_scores(q)
        path_rank = sorted(path_s, key=lambda x: path_s[x], reverse=True)
        path_top = path_rank[:k]

        rrf_top = rrf_fuse(bm25_rank[:50], path_rank[:50], top_k=k)
        wt_top = weighted_fuse(bm25_s, path_s, alpha, k)
        # routed: fuse only on bridge questions; pure BM25 on comparison (gold label)
        routed_top = wt_top if qt == "bridge" else bm25_top
        # routed_clf: same, but route by the heuristic classifier (no gold label)
        pred_type = classify_qtype(q)
        clf_correct += int(pred_type == qt)
        routed_clf_top = wt_top if pred_type == "bridge" else bm25_top

        for m, top in [("bm25", bm25_top), ("path_only", path_top),
                       ("rrf", rrf_top), ("weighted", wt_top),
                       ("routed", routed_top), ("routed_clf", routed_clf_top)]:
            res[m]["hits2"] += int(gold.issubset(set(top)))
            res[m]["hits1"] += int(bool(gold & set(top[:1])))

        by_type[qt]["bm25"]["hits2"] += int(gold.issubset(set(bm25_top)))
        by_type[qt]["bm25"]["n"] += 1
        by_type[qt]["rrf"]["hits2"] += int(gold.issubset(set(rrf_top)))
        by_type[qt]["rrf"]["n"] += 1

        # failure diagnostic: which gold docs did BM25 miss in top-k?
        missed = gold - set(bm25_top)
        if missed:
            bm25_missed_total += 1
            path_found = set(path_top)
            if missed.issubset(path_found):
                bm25_missed_recoverable += 1

        if (qi + 1) % 25 == 0:
            print(f"  {qi+1}/{len(questions)} done…")

    print(f"\nResults — {total} questions, corpus {N:,} docs, k={k}, alpha={alpha}\n")
    print(f"{'method':<12} {'both-gold@'+str(k):>14} {'any-gold@1':>12}")
    print("-" * 42)
    for m in methods:
        print(f"{m:<12} {100*res[m]['hits2']/total:>12.1f}%  {100*res[m]['hits1']/total:>10.1f}%")

    print("\nBy question type (both-gold@k):")
    for qt in ("bridge", "comparison"):
        b = by_type[qt]["bm25"]; r = by_type[qt]["rrf"]
        if b["n"]:
            print(f"  {qt:<11} n={b['n']:<4} BM25 {100*b['hits2']/b['n']:>5.1f}%  "
                  f"RRF {100*r['hits2']/r['n']:>5.1f}%")

    print(f"\nClassifier accuracy vs gold type: {100*clf_correct/total:.1f}% ({clf_correct}/{total})")

    print(f"\nFailure diagnostic:")
    print(f"  BM25 missed >=1 gold doc in top-{k}: {bm25_missed_total}/{total} questions")
    if bm25_missed_total:
        print(f"  …of those, path-only recovered ALL missed gold: "
              f"{bm25_missed_recoverable} ({100*bm25_missed_recoverable/bm25_missed_total:.1f}%)")
        print(f"  → headroom ceiling for hybrid: +{bm25_missed_recoverable} questions "
              f"({100*bm25_missed_recoverable/total:.1f}pp)")

    out = ROOT / "hotpotqa_hybrid_results.json"
    out.write_text(json.dumps({
        "corpus_size": N, "n_questions": total, "k": k, "alpha": alpha,
        "results": res,
        "by_type": {qt: {m: dict(v) for m, v in d.items()} for qt, d in by_type.items()},
        "bm25_missed_total": bm25_missed_total,
        "bm25_missed_recoverable": bm25_missed_recoverable,
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--bios", type=int, default=30000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--split", default="validation")
    args = p.parse_args()
    run(args.n, args.bios, args.k, args.split, args.alpha)
