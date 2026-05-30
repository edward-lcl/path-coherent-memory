#!/usr/bin/env python3
"""
REAL self-ask iterative baseline on the MuSiQue embedding-disjoint tail.

This replaces the oracle (gold-intermediate) ceiling with an honest loop:
  hop-1 sub-question -> retrieve top-k -> LLM READS retrieved docs and PREDICTS
  the intermediate answer -> fill predicted answer into hop-2 sub-question ->
  retrieve -> terminal hit@k.

The oracle version measured potential; this measures what a real IRCoT/self-ask
reader actually achieves when its bridge guess can be wrong. The gap between this
and the oracle is the honest cost of iterative reading. Compared head-to-head
against token-path (free, read-free) on the SAME disjoint tail.

LLM = local oMLX. Retrieval for the reader uses dense (the strong retriever);
this is the friendliest possible setup for iterative, so any residual path
contribution is conservative.
"""
import json, sys, re, urllib.request, urllib.error
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

OMLX_URL = "http://127.0.0.1:8000/v1/chat/completions"
import os as _os
def _load_omlx_key():
    k = _os.environ.get("OMLX_API_KEY")
    if k: return k
    f = ROOT / ".omlx_key"
    if f.exists(): return f.read_text().strip()
    raise RuntimeError("Set OMLX_API_KEY or create .omlx_key")
OMLX_KEY = _load_omlx_key()
DEFAULT_MODEL = "Qwen3-4B-Instruct-2507-MLX-8bit"
BIO_CORPUS = ROOT / "wikipedia_bio_corpus.json"
PLACEHOLDER = re.compile(r"#(\d+)")


def call_omlx(model, prompt, max_tokens=64, retries=3):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OMLX_KEY}"}
    for a in range(retries):
        try:
            req = urllib.request.Request(OMLX_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
                return d["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if a == retries - 1:
                return ""
    return ""


def predict_intermediate(model, subq, doc_texts):
    ctx = "\n\n".join(f"[{i+1}] {t[:500]}" for i, t in enumerate(doc_texts[:5]))
    prompt = (f"Answer the question using ONLY the passages. Reply with just the "
              f"answer entity, no explanation.\n\nPassages:\n{ctx}\n\n"
              f"Question: {subq}\nAnswer:")
    ans = call_omlx(model, prompt, max_tokens=32)
    ans = ans.split("\n")[0].strip().strip('".')
    return ans[:80]


def run(scan, thresh, k, n_bios, split, model):
    print(f"Scanning MuSiQue {split} 2-hop for disjoint tail (cos<{thresh})…")
    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    cand = []
    for ex in ds:
        if hop_bucket(ex) != 2 or not ex.get("answerable", True):
            continue
        steps = ex["question_decomposition"]
        paras = {p["idx"]: p for p in ex["paragraphs"]}
        si, s0 = steps[-1].get("paragraph_support_idx"), steps[0].get("paragraph_support_idx")
        if si not in paras or s0 not in paras:
            continue
        cand.append({"id": ex["id"], "q": ex["question"], "paras": ex["paragraphs"],
                     "gold_term_idx": si, "gold_bridge_idx": s0,
                     "subqs": [s["question"] for s in steps],
                     "answers": [s["answer"] for s in steps]})
        if len(cand) >= scan:
            break
    qv = embed_texts([c["q"] for c in cand], batch_size=256)
    tv = embed_texts([{p["idx"]: p for p in c["paras"]}[c["gold_term_idx"]]["paragraph_text"][:512]
                      for c in cand], batch_size=256)
    for i, c in enumerate(cand):
        c["qt_cos"] = float(qv[i] @ tv[i])
    disjoint = [c for c in cand if c["qt_cos"] < thresh]
    print(f"  {len(disjoint)}/{len(cand)} in disjoint tail")

    notes_by_id, questions = {}, []
    for c in disjoint:
        pid_by_idx = {}
        for p in c["paras"]:
            pid = f"{c['id']}::p{p['idx']}"
            pid_by_idx[p["idx"]] = pid
            notes_by_id.setdefault(pid, {"id": pid, "source": p["title"],
                                         "content": p["paragraph_text"]})
        questions.append({"q": c["q"], "gold": pid_by_idx[c["gold_term_idx"]],
                          "subqs": c["subqs"], "answers": c["answers"]})
    for b in (json.loads(BIO_CORPUS.read_text())[:n_bios] if n_bios else []):
        notes_by_id.setdefault("bio::" + b["source"],
                               {"id": "bio::" + b["source"], "source": b["source"],
                                "content": b["content"]})
    notes = list(notes_by_id.values())
    corpus = H.build_index(notes)
    N = len(notes)
    idf_table = build_idf_table(corpus["df"], N)
    _, bm25_ranked = H.make_bm25_proper(corpus)
    path_scores = H.make_path_scores(corpus, idf_table, bm25_ranked)
    id_list = [n["id"] for n in notes]
    doc_vecs = embed_corpus(notes, f"disjoint_{split}_{len(questions)}_{n_bios}")
    q_vecs = embed_texts([q["q"] for q in questions], batch_size=256)
    print(f"  corpus {N:,} docs, {len(questions)} disjoint questions, model={model}")

    hitmaps, oracle_hits = [], []
    bridge_correct = 0
    for qi, q in enumerate(questions):
        gold = q["gold"]
        # ---- token-path (free, read-free) ----
        ps = path_scores(q["q"])
        path_hit = gold in set(sorted(ps, key=lambda x: ps[x], reverse=True)[:k])

        # ---- REAL self-ask: hop-1 retrieve -> LLM predict bridge -> hop-2 retrieve ----
        sq1 = q["subqs"][0]
        sq1_vec = embed_texts([sq1], batch_size=8)[0]
        s1 = doc_vecs @ sq1_vec
        top1_idx = np.argsort(-s1)[:5]
        pred_bridge = predict_intermediate(model, sq1, [notes[j]["content"] for j in top1_idx])
        gold_bridge = q["answers"][0]
        if gold_bridge.lower() in pred_bridge.lower() or pred_bridge.lower() in gold_bridge.lower():
            bridge_correct += 1
        sq2 = PLACEHOLDER.sub(lambda m: pred_bridge, q["subqs"][1])
        sq2_vec = embed_texts([sq2], batch_size=8)[0]
        s2 = doc_vecs @ sq2_vec
        real_hit = gold in set(id_list[j] for j in np.argsort(-s2)[:k])

        # ---- oracle (gold bridge) for the gap ----
        osq2 = PLACEHOLDER.sub(lambda m: gold_bridge, q["subqs"][1])
        osq2_vec = embed_texts([osq2], batch_size=8)[0]
        os2 = doc_vecs @ osq2_vec
        oracle_hit = gold in set(id_list[j] for j in np.argsort(-os2)[:k])

        hitmaps.append({"path": path_hit, "real-iter": real_hit, "oracle-iter": oracle_hit})
        if (qi + 1) % 25 == 0:
            print(f"    {qi+1}/{len(questions)} processed")

    n = len(hitmaps)
    MODES = ["path", "real-iter", "oracle-iter"]
    rec = {m: sum(h[m] for h in hitmaps) / n for m in MODES}
    ens = ["path", "real-iter"]
    union = sum(any(h[m] for m in ens) for h in hitmaps) / n
    best = max(rec[m] for m in ens)
    excl = {m: sum(h[m] and not any(h[o] for o in ens if o != m) for h in hitmaps) / n for m in ens}
    inter = sum(h["path"] and h["real-iter"] for h in hitmaps)
    uni = sum(h["path"] or h["real-iter"] for h in hitmaps)
    jac = inter / uni if uni else 0.0

    print(f"\nMuSiQue disjoint tail — REAL self-ask loop, n={n}, corpus {N:,}, k={k}")
    print(f"  bridge prediction accuracy (hop-1): {100*bridge_correct/n:.1f}%")
    print(f"  path={100*rec['path']:.1f}%  REAL-iter={100*rec['real-iter']:.1f}%  "
          f"oracle-iter(ceiling)={100*rec['oracle-iter']:.1f}%")
    print(f"  real-iter cost of imperfect bridge: {100*(rec['oracle-iter']-rec['real-iter']):.1f}pp below ceiling")
    print(f"  PATH+REAL-iter union={100*union:.1f}%  best-single={100*best:.1f}%  lift=+{100*(union-best):.1f}pp")
    print(f"  exclusive: path={100*excl['path']:.1f}%  real-iter={100*excl['real-iter']:.1f}%")
    print(f"  Jaccard(path,real-iter)={jac:.2f}")

    out = ROOT / "musique_real_iterative_results.json"
    out.write_text(json.dumps({"n": n, "corpus": N, "k": k, "model": model,
                               "bridge_acc": bridge_correct / n, "recall": rec,
                               "union_path_real": union, "best_single": best,
                               "exclusive": excl, "jaccard_path_real": jac}, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=int, default=1500)
    p.add_argument("--thresh", type=float, default=0.3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--bios", type=int, default=5000)
    p.add_argument("--split", default="validation")
    p.add_argument("--model", default=DEFAULT_MODEL)
    a = p.parse_args()
    run(a.scan, a.thresh, a.k, a.bios, a.split, a.model)
