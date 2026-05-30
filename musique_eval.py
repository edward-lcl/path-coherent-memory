#!/usr/bin/env python3
"""
MuSiQue evaluation — the reproducible vocabulary-gap test.

MuSiQue questions are built by COMPOSING single-hop questions, so the bridge
entity is frequently absent from the question text (e.g. "Who is the spouse of
the Green performer?" — the performer's name is not in the question). This is
the public, pip-installable analogue of the private Talos zero-vocabulary-gap
failure mode. The decisive experiment:

  - If dense retrieval ALSO craters here (like it allegedly does on Talos),
    we have a bulletproof reproducible version of the paper's headline.
  - If dense gets ~90% like HotpotQA, the failure mode may be private-corpus
    specific and we need to understand why.

Pools all support+distractor paragraphs across N questions into one shared
corpus (+ optional bio distractors). Gold = paragraphs flagged
paragraph_support_idx in question_decomposition.

Methods: bm25, dense, bm25+dense, bm25+path, bm25+dense+path.
Metric: all-gold@k (ALL supporting paragraphs in top-k) and any-gold@1.
Also breaks out by hop count (2/3/4) since deeper = bigger vocab gap.
"""
import json, sys, hashlib
from collections import defaultdict
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

BIO_CORPUS = ROOT / "wikipedia_bio_corpus.json"


def hop_count(ex) -> int:
    return len(ex["question_decomposition"])


def hop_bucket(ex) -> int:
    """MuSiQue encodes hop count in the id prefix (2hop__, 3hop1__, 4hop2__)."""
    pre = ex["id"].split("__")[0]
    if pre.startswith("2"):
        return 2
    if pre.startswith("3"):
        return 3
    if pre.startswith("4"):
        return 4
    return len(ex["question_decomposition"])


def embed_corpus(notes, tag: str) -> np.ndarray:
    h = hashlib.sha1((tag + str(len(notes))).encode())
    for n in notes[:200]:
        h.update(n["id"].encode())
    cache = ROOT / f"musique_dense_cache_{h.hexdigest()[:12]}.npy"
    if cache.exists():
        print(f"  loaded cached embeddings {cache.name}")
        return np.load(cache)
    print(f"  embedding {len(notes):,} docs (checkpointed)…")
    texts = [n["content"][:512] for n in notes]
    ckpt = cache.with_suffix(".partial.npy")
    chunk = 2000
    parts = []
    start = 0
    if ckpt.exists():
        prev = np.load(ckpt)
        parts.append(prev)
        start = len(prev)
        print(f"  resuming from checkpoint at {start:,}")
    for i in range(start, len(texts), chunk):
        v = embed_texts(texts[i:i + chunk], batch_size=256)
        parts.append(v)
        np.save(ckpt, np.vstack(parts))
        print(f"  checkpoint {min(i + chunk, len(texts)):,}/{len(texts):,}")
    vecs = np.vstack(parts)
    np.save(cache, vecs)
    ckpt.unlink(missing_ok=True)
    return vecs


def norm(d):
    if not d:
        return {}
    mx = max(d.values()) or 1.0
    return {k: v / mx for k, v in d.items()}


def run(n_examples, n_bios, k, split, alpha, beta, stratify=True):
    print(f"Loading MuSiQue {split} — {n_examples} questions…")
    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    if stratify:
        # Balanced draw across 2/3/4-hop so deeper questions are represented.
        per = max(n_examples // 3, 1)
        buckets = {2: [], 3: [], 4: []}
        for ex in ds:
            b = hop_bucket(ex)
            if b in buckets and len(buckets[b]) < per:
                buckets[b].append(ex)
            if all(len(v) >= per for v in buckets.values()):
                break
        ds = buckets[2] + buckets[3] + buckets[4]
        print(f"  stratified: 2hop={len(buckets[2])} 3hop={len(buckets[3])} 4hop={len(buckets[4])}")
    else:
        ds = ds.select(range(min(n_examples, len(ds))))

    notes_by_id, questions = {}, []
    for ex in ds:
        if not ex.get("answerable", True):
            continue
        # Build a globally-unique paragraph id per (question, idx).
        qid = ex["id"]
        para_id_by_idx = {}
        for p in ex["paragraphs"]:
            pid = f"{qid}::p{p['idx']}"
            para_id_by_idx[p["idx"]] = pid
            notes_by_id.setdefault(pid, {
                "id": pid,
                "source": p["title"],
                "content": p["paragraph_text"],
            })
        gold = set()
        for step in ex["question_decomposition"]:
            si = step.get("paragraph_support_idx")
            if si is not None and si in para_id_by_idx:
                gold.add(para_id_by_idx[si])
        if not gold:
            continue
        questions.append({"q": ex["question"], "gold": gold, "hops": hop_bucket(ex)})

    print(f"  pooled {len(notes_by_id):,} MuSiQue paragraphs")
    bios = json.loads(BIO_CORPUS.read_text())[:n_bios] if n_bios else []
    for b in bios:
        notes_by_id.setdefault("bio::" + b["source"],
                               {"id": "bio::" + b["source"], "source": b["source"],
                                "content": b["content"]})
    notes = list(notes_by_id.values())
    corpus = H.build_index(notes)
    N = len(notes)
    idf_table = build_idf_table(corpus["df"], N)
    bm25_score_all, bm25_ranked = H.make_bm25_proper(corpus)
    path_scores = H.make_path_scores(corpus, idf_table, bm25_ranked)
    print(f"  corpus {N:,} docs, {len(questions)} answerable questions")

    doc_vecs = embed_corpus(notes, f"{split}_{n_examples}_{n_bios}_strat{int(stratify)}")
    doc_ids = [n["id"] for n in notes]
    q_vecs = embed_texts([q["q"] for q in questions], batch_size=256)

    methods = ["bm25", "dense", "bm25+dense", "bm25+path", "bm25+dense+path"]
    res = {m: {"hits": 0, "h1": 0, "recall": 0.0} for m in methods}
    by_hop = defaultdict(lambda: {m: 0.0 for m in methods} | {"n": 0})
    total = 0

    for qi, item in enumerate(questions):
        q, gold, hops = item["q"], item["gold"], item["hops"]
        total += 1

        bm25_s = bm25_score_all(q)
        bm25_top = sorted(bm25_s, key=lambda x: bm25_s[x], reverse=True)[:k]

        sims = doc_vecs @ q_vecs[qi]
        order = np.argsort(-sims)[:max(k, 50)]
        dense_s = {doc_ids[j]: float(sims[j]) for j in order}
        dense_top = [doc_ids[j] for j in order[:k]]

        path_s = path_scores(q)
        bn, dn, pn = norm(bm25_s), norm(dense_s), norm(path_s)

        def fuse(*weighted):
            agg = {}
            for sc, w in weighted:
                for kk, v in sc.items():
                    agg[kk] = agg.get(kk, 0.0) + w * v
            return sorted(agg, key=lambda x: agg[x], reverse=True)[:k]

        tops = {
            "bm25": bm25_top,
            "dense": dense_top,
            "bm25+dense": fuse((bn, 1.0), (dn, beta)),
            "bm25+path": fuse((bn, 1.0), (pn, alpha)),
            "bm25+dense+path": fuse((bn, 1.0), (dn, beta), (pn, alpha)),
        }
        by_hop[hops]["n"] += 1
        for m, top in tops.items():
            tset = set(top)
            res[m]["hits"] += int(gold.issubset(tset))
            res[m]["h1"] += int(bool(gold & set(top[:1])))
            frac = len(gold & tset) / len(gold)
            res[m]["recall"] += frac
            by_hop[hops][m] += frac

        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(questions)} scored…")

    print(f"\nMuSiQue {split} — {total} q, corpus {N:,} docs, k={k}, "
          f"alpha={alpha}, beta={beta}\n")
    allg = "all-gold@" + str(k)
    rec = "recall@" + str(k)
    print(f"{'method':<18} {allg:>13} {rec:>10} {'any@1':>8}")
    print("-" * 52)
    for m in methods:
        a = 100 * res[m]["hits"] / total
        r = 100 * res[m]["recall"] / total
        h = 100 * res[m]["h1"] / total
        print(f"{m:<18} {a:>11.1f}%  {r:>8.1f}%  {h:>6.1f}%")

    print(f"\nBy hop count (per-support recall@{k} — fraction of gold docs found):")
    print(f"{'hops':<6} {'n':>5} " + " ".join(f"{m.split('+')[-1][:6]:>7}" for m in methods))
    for hops in sorted(by_hop):
        d = by_hop[hops]; n = d["n"]
        cells = " ".join(f"{100*d[m]/n:>6.1f}%" for m in methods)
        print(f"{hops:<6} {n:>5} {cells}")

    out = ROOT / "musique_results.json"
    out.write_text(json.dumps({"corpus_size": N, "n": total, "k": k,
                               "alpha": alpha, "beta": beta, "results": res,
                               "by_hop": {h: dict(v) for h, v in by_hop.items()}},
                              indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--bios", type=int, default=0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--split", default="validation")
    p.add_argument("--no-stratify", action="store_true")
    args = p.parse_args()
    run(args.n, args.bios, args.k, args.split, args.alpha, args.beta,
        stratify=not args.no_stratify)
