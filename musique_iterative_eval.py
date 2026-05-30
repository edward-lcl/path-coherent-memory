#!/usr/bin/env python3
"""
Oracle-iterative retrieval on MuSiQue — the missing mechanism.

Every method tested so far is SINGLE-SHOT: embed the full question once, retrieve
top-k. But MuSiQue is the canonical benchmark single-shot retrieval cannot solve,
because the bridge entity is absent from the question and only appears in the
hop-1 document ("who does the voice of Stan on #1" — #1=South Park is unknown
until hop-1 is read). The real solution is ITERATIVE retrieval: retrieve hop-1,
extract the bridge, reformulate, retrieve hop-2.

This builds the ORACLE-iterative ceiling using MuSiQue's gold question_decomposition:
each sub-question, with #k placeholders resolved by gold intermediate answers, is
retrieved SEPARATELY; the per-hop top-k are unioned. No LLM needed — this isolates
"is the bottleneck decomposition+reading, or retrieval itself?"

Compares per-support recall@k by hop:
  - bm25 / dense / bm25+dense        (single-shot, from musique_eval)
  - dense-iter (oracle)              (retrieve each resolved sub-question, union)
  - bm25+path single-shot            (our method, for contrast)

If dense-iter solves MuSiQue → the missing mechanism is READING/REFORMULATION,
not corpus topology. The decisive follow-up is whether the same oracle-iterative
trick also cracks Talos (where the bridge is NOT a re-queryable named entity).
"""
import json, re, sys, hashlib
from collections import defaultdict
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


PLACEHOLDER = re.compile(r"#(\d+)")


def resolve_subq(subq: str, answers: list[str]) -> str:
    """Replace #k placeholders with the k-th prior gold answer."""
    def sub(m):
        idx = int(m.group(1)) - 1
        return answers[idx] if 0 <= idx < len(answers) else m.group(0)
    return PLACEHOLDER.sub(sub, subq)


def run(n_examples: int, k: int, split: str, beta: float):
    print(f"Loading MuSiQue {split} — stratified {n_examples}…")
    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    per = max(n_examples // 3, 1)
    buckets = {2: [], 3: [], 4: []}
    for ex in ds:
        b = hop_bucket(ex)
        if b in buckets and len(buckets[b]) < per:
            buckets[b].append(ex)
        if all(len(v) >= per for v in buckets.values()):
            break
    examples = buckets[2] + buckets[3] + buckets[4]
    print(f"  stratified 2/3/4-hop = {len(buckets[2])}/{len(buckets[3])}/{len(buckets[4])}")

    notes_by_id, questions = {}, []
    for ex in examples:
        if not ex.get("answerable", True):
            continue
        qid = ex["id"]
        pid_by_idx = {}
        for p in ex["paragraphs"]:
            pid = f"{qid}::p{p['idx']}"
            pid_by_idx[p["idx"]] = pid
            notes_by_id.setdefault(pid, {"id": pid, "source": p["title"],
                                         "content": p["paragraph_text"]})
        gold, subqs, answers = set(), [], []
        ok = True
        for step in ex["question_decomposition"]:
            si = step.get("paragraph_support_idx")
            if si is None or si not in pid_by_idx:
                ok = False
                break
            gold.add(pid_by_idx[si])
            subqs.append(step["question"])
            answers.append(step["answer"])
        if not ok or not gold:
            continue
        questions.append({"q": ex["question"], "gold": gold, "hops": hop_bucket(ex),
                          "subqs": subqs, "answers": answers})

    notes = list(notes_by_id.values())
    corpus = H.build_index(notes)
    N = len(notes)
    idf_table = build_idf_table(corpus["df"], N)
    bm25_score_all, bm25_ranked = H.make_bm25_proper(corpus)
    path_scores = H.make_path_scores(corpus, idf_table, bm25_ranked)
    print(f"  corpus {N:,} docs, {len(questions)} questions")

    doc_vecs = embed_corpus(notes, f"iter_{split}_{n_examples}")
    doc_ids = [n["id"] for n in notes]
    id_pos = {d: i for i, d in enumerate(doc_ids)}

    # Embed full questions + every resolved sub-question in one batch.
    flat_texts, q_spans = [], []
    for item in questions:
        start = len(flat_texts)
        flat_texts.append(item["q"])
        resolved = []
        for hop_i, sq in enumerate(item["subqs"]):
            resolved.append(resolve_subq(sq, item["answers"][:hop_i]))
        flat_texts.extend(resolved)
        q_spans.append((start, len(resolved)))
    all_vecs = embed_texts(flat_texts, batch_size=256)

    def norm(d):
        if not d:
            return {}
        mx = max(d.values()) or 1.0
        return {kk: v / mx for kk, v in d.items()}

    methods = ["bm25", "dense", "bm25+dense", "bm25+path", "dense-iter", "bm25+dense+path-iter"]
    res = {m: {"recall": 0.0, "h1": 0} for m in methods}
    by_hop = defaultdict(lambda: {m: 0.0 for m in methods} | {"n": 0})
    total = 0

    for qi, item in enumerate(questions):
        q, gold, hops = item["q"], item["gold"], item["hops"]
        total += 1
        qstart, nhops = q_spans[qi]
        qvec = all_vecs[qstart]

        bm25_s = bm25_score_all(q)
        bm25_top = sorted(bm25_s, key=lambda x: bm25_s[x], reverse=True)[:k]

        sims = doc_vecs @ qvec
        order = np.argsort(-sims)[:max(k, 50)]
        dense_s = {doc_ids[j]: float(sims[j]) for j in order}
        dense_top = [doc_ids[j] for j in order[:k]]

        path_s = path_scores(q)
        bn, dn, pn = norm(bm25_s), norm(dense_s), norm(path_s)

        def fuse(*w):
            agg = {}
            for sc, wt in w:
                for kk, v in sc.items():
                    agg[kk] = agg.get(kk, 0.0) + wt * v
            return sorted(agg, key=lambda x: agg[x], reverse=True)[:k]

        bd_top = fuse((bn, 1.0), (dn, beta))

        # ORACLE-ITERATIVE: retrieve each resolved sub-question separately, union.
        # Budget k total: take top ceil(k/nhops) per hop, then fill.
        per_hop = max(k // nhops, 1)
        iter_pool, iter_seen = [], set()
        hop_vecs = all_vecs[qstart + 1: qstart + 1 + nhops]
        for hv in hop_vecs:
            hs = doc_vecs @ hv
            ho = np.argsort(-hs)[:per_hop + 3]
            taken = 0
            for j in ho:
                did = doc_ids[j]
                if did not in iter_seen:
                    iter_pool.append(did)
                    iter_seen.add(did)
                    taken += 1
                if taken >= per_hop:
                    break
        # fill remaining budget from single-shot dense
        for did in dense_top:
            if len(iter_pool) >= k:
                break
            if did not in iter_seen:
                iter_pool.append(did)
                iter_seen.add(did)
        dense_iter_top = iter_pool[:k]

        # combined: union of single-shot fusion and iterative, capped at k
        combo = list(dict.fromkeys(bd_top + dense_iter_top))[:k]

        tops = {
            "bm25": bm25_top, "dense": dense_top, "bm25+dense": bd_top,
            "bm25+path": fuse((bn, 1.0), (pn, 0.6)),
            "dense-iter": dense_iter_top,
            "bm25+dense+path-iter": combo,
        }
        by_hop[hops]["n"] += 1
        for m, top in tops.items():
            tset = set(top)
            frac = len(gold & tset) / len(gold)
            res[m]["recall"] += frac
            res[m]["h1"] += int(bool(gold & set(top[:1])))
            by_hop[hops][m] += frac

        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(questions)} scored…")

    print(f"\nMuSiQue oracle-iterative — {total} q, corpus {N:,} docs, k={k}\n")
    print(f"{'method':<24} {'recall@'+str(k):>10} {'any@1':>8}")
    print("-" * 44)
    for m in methods:
        print(f"{m:<24} {100*res[m]['recall']/total:>8.1f}%  {100*res[m]['h1']/total:>6.1f}%")

    print(f"\nBy hop (per-support recall@{k}):")
    print(f"{'hops':<6}{'n':>5}  " + "".join(f"{m[:11]:>13}" for m in methods))
    for hops in sorted(by_hop):
        d = by_hop[hops]; n = d["n"]
        cells = "".join(f"{100*d[m]/n:>12.1f}%" for m in methods)
        print(f"{hops:<6}{n:>5}  {cells}")

    out = ROOT / "musique_iterative_results.json"
    out.write_text(json.dumps({"corpus_size": N, "n": total, "k": k, "beta": beta,
                               "results": res,
                               "by_hop": {h: dict(v) for h, v in by_hop.items()}},
                              indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--split", default="validation")
    args = p.parse_args()
    run(args.n, args.k, args.split, args.beta)
