#!/usr/bin/env python3
"""
MuSiQue embedding-disjoint tail — the PUBLIC reproduction of the structural claim.

The Talos complementarity result lives on a private corpus. This applies the SAME
objective filter (question<->terminal cosine < thresh) to public MuSiQue, isolating
the tail where dense retrieval has no semantic signal, then runs the 3-way
complementarity (dense / path / oracle-iterative) to test whether the
"no single mode suffices" pattern reproduces on data a reviewer can run.

Focus on 2-hop (cleanest single-bridge structure). Query = full question.
Gold = terminal supporting paragraph. Oracle-iter resolves the bridge from gold
intermediate answer. Corpus padded with bio distractors for realistic scale.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hotpotqa_hybrid_eval as H
from path_coherent_retriever import build_idf_table
from embedding_bridge_retriever import embed_texts
from musique_eval import hop_bucket, embed_corpus
from musique_iterative_eval import resolve_subq

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

BIO_CORPUS = ROOT / "wikipedia_bio_corpus.json"


def run(n_scan: int, thresh: float, k: int, n_bios: int, split: str):
    print(f"Scanning MuSiQue {split} 2-hop for embedding-disjoint tail (cos<{thresh})…")
    ds = load_dataset("dgslibisey/MuSiQue", split=split)

    cand = []
    for ex in ds:
        if hop_bucket(ex) != 2 or not ex.get("answerable", True):
            continue
        steps = ex["question_decomposition"]
        paras = {p["idx"]: p for p in ex["paragraphs"]}
        si = steps[-1].get("paragraph_support_idx")
        s0 = steps[0].get("paragraph_support_idx")
        if si not in paras or s0 not in paras:
            continue
        cand.append({
            "id": ex["id"], "q": ex["question"],
            "paras": ex["paragraphs"],
            "gold_term_idx": si, "gold_bridge_idx": s0,
            "subqs": [s["question"] for s in steps],
            "answers": [s["answer"] for s in steps],
        })
        if len(cand) >= n_scan:
            break

    # Score question<->terminal cosine to find the disjoint tail.
    qv = embed_texts([c["q"] for c in cand], batch_size=256)
    tv = embed_texts([{p["idx"]: p for p in c["paras"]}[c["gold_term_idx"]]["paragraph_text"][:512]
                      for c in cand], batch_size=256)
    for i, c in enumerate(cand):
        c["qt_cos"] = float(qv[i] @ tv[i])
    disjoint = [c for c in cand if c["qt_cos"] < thresh]
    print(f"  {len(disjoint)}/{len(cand)} in disjoint tail")

    # Build shared corpus from disjoint questions' paragraphs + bio distractors.
    notes_by_id, questions = {}, []
    for c in disjoint:
        pid_by_idx = {}
        for p in c["paras"]:
            pid = f"{c['id']}::p{p['idx']}"
            pid_by_idx[p["idx"]] = pid
            notes_by_id.setdefault(pid, {"id": pid, "source": p["title"],
                                         "content": p["paragraph_text"]})
        questions.append({
            "q": c["q"],
            "gold": pid_by_idx[c["gold_term_idx"]],
            "bridge": pid_by_idx[c["gold_bridge_idx"]],
            "subqs": c["subqs"], "answers": c["answers"],
        })
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
    print(f"  corpus {N:,} docs, {len(questions)} disjoint questions")

    id_list = [n["id"] for n in notes]
    id_pos = {d: i for i, d in enumerate(id_list)}
    doc_vecs = embed_corpus(notes, f"disjoint_{split}_{len(questions)}_{n_bios}")
    # resolved hop-2 sub-question (bridge filled from gold answer)
    res_subqs = [resolve_subq(q["subqs"][1], q["answers"][:1]) for q in questions]
    sq_vecs = embed_texts(res_subqs, batch_size=256)
    q_vecs = embed_texts([q["q"] for q in questions], batch_size=256)

    MODES = ["dense", "path", "oracle-iter"]
    hitmaps = []
    for qi, q in enumerate(questions):
        gold = q["gold"]
        sims = doc_vecs @ q_vecs[qi]
        dense_top = set(id_list[j] for j in np.argsort(-sims)[:k])

        ps = path_scores(q["q"])
        path_top = set(sorted(ps, key=lambda x: ps[x], reverse=True)[:k])

        # oracle-iter: retrieve resolved hop-2 sub-question (bridge known)
        ssims = doc_vecs @ sq_vecs[qi]
        iter_top = set(id_list[j] for j in np.argsort(-ssims)[:k])

        hitmaps.append({"dense": gold in dense_top, "path": gold in path_top,
                        "oracle-iter": gold in iter_top})

    n = len(hitmaps)
    rec = {m: sum(h[m] for h in hitmaps) / n for m in MODES}
    union = sum(any(h[m] for m in MODES) for h in hitmaps) / n
    best = max(rec.values())
    excl = {m: sum(h[m] and not any(h[o] for o in MODES if o != m) for h in hitmaps) / n
            for m in MODES}
    jac = {}
    for i, m1 in enumerate(MODES):
        for m2 in MODES[i+1:]:
            inter = sum(h[m1] and h[m2] for h in hitmaps)
            uni = sum(h[m1] or h[m2] for h in hitmaps)
            jac[f"{m1}&{m2}"] = inter / uni if uni else 0.0

    print(f"\nMuSiQue embedding-disjoint tail — n={n}, corpus {N:,}, k={k}")
    print(f"  per-method: " + "  ".join(f"{m}={100*rec[m]:.1f}%" for m in MODES))
    print(f"  best-single={100*best:.1f}%   UNION={100*union:.1f}%   lift=+{100*(union-best):.1f}pp")
    print(f"  exclusive: " + "  ".join(f"{m}={100*excl[m]:.1f}%" for m in MODES))
    print(f"  Jaccard: " + "  ".join(f"{kk}={v:.2f}" for kk, v in jac.items()))

    out = ROOT / "musique_disjoint_results.json"
    out.write_text(json.dumps({"n": n, "corpus": N, "thresh": thresh, "k": k,
                               "recall": rec, "union": union, "best": best,
                               "exclusive": excl, "jaccard": jac}, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=int, default=1500)
    p.add_argument("--thresh", type=float, default=0.3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--bios", type=int, default=5000)
    p.add_argument("--split", default="validation")
    args = p.parse_args()
    run(args.scan, args.thresh, args.k, args.bios, args.split)
