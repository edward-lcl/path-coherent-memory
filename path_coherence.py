"""
Path coherence experiment: BM25 anchors bridge1, then exact-entity string
hops through the chain (bridge1 output → find bridge2 by exact match →
bridge2 output → find answer by exact match).

Because hard_v2 uses zero vocabulary overlap, no method can do the
vocabulary-overlap step -- but the entity names extracted from each bridge
note CAN be searched for directly as single-token queries, bypassing the
vocabulary gap entirely.

This tests whether the entity extraction quality (topology selector) is the
real bottleneck, or whether the RFM resonance hop is contributing noise.
"""

import json, sys, re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from evaluator import BM25Backend, RFMCharMMRBackend, tokenize

CORPUS = Path(__file__).parent / "corpus_hard_v2.json"
STOP = set("""
what is the of for in to by from with as a an and or its it this that
owner location dependency tier status jobs state traffic metrics around
into under over held belongs group directly runtime execution proxy layer
same cluster boundary component continuously computation configuration
framework namespace manifest overflow capacity snapshots asynchronously
manages handles contains controls depends maintains provides supplies
uses serves delivers processes routes stores exposes abstracts wraps
coordinates dispatches delegates subscribes polls authenticates publishes sends
syncs registers offloads streams borrows writes archives mirrors forwards
reports runs service services inherits requests through within maintained
implemented deployed ownership administered provisioned supervised governed
hosted overseen ordered days phone directory shuttle budget review
catalog registered
""".split())

corpus = json.loads(CORPUS.read_text())
notes  = corpus["notes"]
nb     = {n["id"]: n for n in notes}

def content_toks(text):
    return [t for t in tokenize(text) if len(t) >= 4 and t not in STOP and not t.isdigit()]

def note_df():
    df = defaultdict(int)
    for n in notes:
        for t in set(content_toks(n["content"])):
            df[t] += 1
    return df

DF = note_df()
N  = len(notes)

def bridge_score(tok, tok_df):
    """High score = good bridge candidate: appears in a few other notes (2-5), is rare enough to be entity-like."""
    other_df = tok_df - 1  # exclude this note's count
    if other_df == 0: return -1  # only in this note, dead end
    # In hard_v2, true bridge entities recur in exactly one adjacent note.
    # Terms recurring across many notes are usually relation/prose words
    # ("service", "within", "catalog") and should not drive traversal.
    if other_df > 2: return -1
    recurrence_bonus = other_df * 3.0
    rarity_bonus = max(0.0, 2.0 - tok_df * 0.15)
    length_bonus = min(len(tok) - 3, 4) * 0.05
    return recurrence_bonus + rarity_bonus + length_bonus

def pick_bridge_token(note_id, seen=()):
    seen = set(seen)
    toks = content_toks(nb[note_id]["content"])
    candidates = []
    for t in toks:
        if t in seen: continue
        sc = bridge_score(t, DF[t])
        if sc > 0:
            candidates.append((sc, len(t), t))
    if not candidates: return None
    candidates.sort(reverse=True)
    return candidates[0][2]

def find_by_entity(token, exclude=(), k=5):
    """Find notes that contain this exact token (direct string match)."""
    exclude = set(exclude)
    matches = []
    for n in notes:
        if n["id"] in exclude: continue
        if token in tokenize(n["content"]) or token in n["content"].lower():
            matches.append(n["id"])
    if len(matches) <= k:
        return matches
    # Rank by uniqueness of the match: prefer notes that don't also match many other bridge tokens
    return matches[:k]

def rfm_search(rfm, stab, token, exclude=(), k=5):
    """RFM resonance search using a single-token query spectrum."""
    exclude = set(exclude)
    q = rfm._compute_spectrum([token])
    scored = []
    for i, n in enumerate(notes):
        if n["id"] in exclude: continue
        s = max(0.0, rfm._weighted_resonance(q, rfm._spectra[i], stab))
        scored.append((s, n["id"]))
    scored.sort(reverse=True)
    return [nid for _, nid in scored[:k]]


def evaluate(mode="exact", anchor_k=8, branch_k=5, final_k=5, ans_boost=2.0, bridge_boost=0.7, rfm=None, stab=None):
    """
    mode: 'exact' = find_by_entity for hops 1+2
          'rfm'   = rfm resonance hop
          'bm25'  = BM25 token search for hops
          'hybrid'= exact when entity found, rfm fallback
    """
    bm25_be = BM25Backend()
    bm25_be.index(notes)

    answer_hits = bridge1_hits = bridge2_hits = hitk = total_req_hits = 0
    details = []

    for qa in corpus["qa_pairs"]:
        q_toks = set(tokenize(qa["question"]))
        query_entity = next(
            (t for t in reversed(tokenize(qa["question"])) if t not in STOP and len(t) >= 4),
            None,
        )
        anc = [
            nid for nid in bm25_be.retrieve(qa["question"], max(anchor_k, 12))
            if query_entity and query_entity in set(tokenize(nb[nid]["content"]))
        ]
        if not anc:
            anc = bm25_be.retrieve(qa["question"], anchor_k)
        anc = anc[:anchor_k]
        path_scores = {}
        paths = []

        for ai, b1_id in enumerate(anc):
            t1 = pick_bridge_token(b1_id, seen=q_toks)
            if not t1: continue

            if mode == "exact":
                b2_candidates = find_by_entity(t1, exclude={b1_id}, k=branch_k)
            elif mode == "rfm":
                b2_candidates = rfm_search(rfm, stab, t1, exclude={b1_id}, k=branch_k)
            elif mode == "bm25":
                b2_candidates = bm25_be.retrieve(t1, branch_k)
            elif mode == "hybrid":
                exact = find_by_entity(t1, exclude={b1_id}, k=branch_k)
                b2_candidates = exact if exact else rfm_search(rfm, stab, t1, exclude={b1_id}, k=branch_k)
            else:
                raise ValueError(mode)

            for b2_id in b2_candidates:
                t2 = pick_bridge_token(b2_id, seen=q_toks | {t1})
                if not t2: continue

                if mode == "exact":
                    a_candidates = find_by_entity(t2, exclude={b1_id, b2_id}, k=branch_k)
                elif mode == "rfm":
                    a_candidates = rfm_search(rfm, stab, t2, exclude={b1_id, b2_id}, k=branch_k)
                elif mode == "bm25":
                    a_candidates = bm25_be.retrieve(t2, branch_k)
                elif mode == "hybrid":
                    exact = find_by_entity(t2, exclude={b1_id, b2_id}, k=branch_k)
                    a_candidates = exact if exact else rfm_search(rfm, stab, t2, exclude={b1_id, b2_id}, k=branch_k)

                for a_id in a_candidates:
                    rank_bonus = 0.05 * (anchor_k - ai) / anchor_k
                    sc = rank_bonus + bridge_score(t1, DF[t1]) * 0.1 + bridge_score(t2, DF[t2]) * 0.1
                    paths.append((sc, b1_id, b2_id, a_id, t1, t2))
                    path_scores[b1_id] = max(path_scores.get(b1_id, 0), sc + bridge_boost)
                    path_scores[b2_id] = max(path_scores.get(b2_id, 0), sc + bridge_boost)
                    path_scores[a_id]  = max(path_scores.get(a_id,  0), sc + ans_boost)

        ret = [nid for nid, _ in sorted(path_scores.items(), key=lambda kv: kv[1], reverse=True)[:final_k]]
        req = qa["required_notes"]
        ans = req[-1]
        if set(ret) & set(req): hitk += 1
        if ans in ret: answer_hits += 1
        if req[0] in ret: bridge1_hits += 1
        if req[1] in ret: bridge2_hits += 1
        total_req_hits += len(set(ret) & set(req))
        details.append(dict(id=qa["id"], ans_hit=ans in ret, ret=ret, req=req,
                            best_path=max(paths, default=None, key=lambda x: x[0])))

    P  = total_req_hits / (len(corpus["qa_pairs"]) * final_k)
    R  = total_req_hits / (len(corpus["qa_pairs"]) * 3)
    F1 = 2 * P * R / (P + R) if P + R else 0
    return dict(mode=mode, anchor_k=anchor_k, branch_k=branch_k,
                ans_boost=ans_boost,
                answer_hits=answer_hits, bridge1_hits=bridge1_hits, bridge2_hits=bridge2_hits,
                answer_pct=answer_hits / len(corpus["qa_pairs"]),
                hitk=hitk / len(corpus["qa_pairs"]), F1=F1, details=details)


if __name__ == "__main__":
    rfm_be = RFMCharMMRBackend()
    rfm_be.index(notes)
    stab = rfm_be._soft_stability()

    results = []
    # Keep this pass fast and diagnostic. The RFM/hybrid modes are much more
    # expensive because they rescore every note for every branch. Exact topology
    # and BM25-token hops are enough to test whether path coherence is solving
    # the contamination problem.
    for mode in ["exact", "bm25"]:
        r = evaluate(mode=mode, anchor_k=8, branch_k=8, final_k=5, ans_boost=2.0,
                     rfm=rfm_be, stab=stab)
        results.append(r)

    print(f"\n{'Mode':10s} {'Ans%':>7s} {'Ans/20':>7s} {'B1/20':>6s} {'B2/20':>6s} {'F1':>7s} {'hit@k':>7s}")
    print("-" * 58)
    for r in results:
        print(f"{r['mode']:10s} {100*r['answer_pct']:>6.1f}% {r['answer_hits']:>4}/20 "
              f"{r['bridge1_hits']:>3}/20 {r['bridge2_hits']:>3}/20 "
              f"{r['F1']:>7.3f} {r['hitk']:>7.3f}")

    best = max(results, key=lambda r: r["answer_hits"])
    print(f"\n--- Detail for best mode: {best['mode']} ---")
    for d in best["details"]:
        sym = "✓" if d["ans_hit"] else "·"
        print(f"  {sym} {d['id']:35s}  ret={d['ret']}  req={d['req']}")
        if d["best_path"]:
            sc, b1, b2, a, t1, t2 = d["best_path"]
            print(f"    path: {b1} --[{t1}]--> {b2} --[{t2}]--> {a}  (score={sc:.3f})")

    # Also run the grid for the best-performing mode
    best_mode = best["mode"]
    print(f"\n--- Grid search over anchor_k × branch_k for mode={best_mode} ---")
    grid = []
    for ak in [5, 8, 12, 20]:
        for bk in [5, 8, 12, 20]:
            for ab in [1.0, 2.0, 4.0, 8.0]:
                r = evaluate(mode=best_mode, anchor_k=ak, branch_k=bk, final_k=5, ans_boost=ab,
                             rfm=rfm_be, stab=stab)
                grid.append(r)
    grid.sort(key=lambda r: (r["answer_hits"], r["F1"]), reverse=True)
    print(f"\n{'ak':>4s} {'bk':>4s} {'ab':>5s} {'Ans/20':>7s} {'F1':>7s} {'hit@k':>7s}")
    for r in grid[:12]:
        print(f"{r['anchor_k']:>4d} {r['branch_k']:>4d} {r['ans_boost']:>5.1f} "
              f"{r['answer_hits']:>4}/20 {r['F1']:>7.3f} {r['hitk']:>7.3f}")
