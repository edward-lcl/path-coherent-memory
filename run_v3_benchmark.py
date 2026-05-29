#!/usr/bin/env python3
"""
V3 Semantic Chain Benchmark

Tests BM25, token-path v12, and embedding-bridge retrieval against
the semantically-mined v3 chains (A→B→C via embedding similarity gaps).

This is the native benchmark for the embedding-bridge retriever — chains
where the connections are semantic, not lexical coincidence.

Key question: does embedding-bridge outperform token-topology on chains
that were designed around semantic proximity rather than rare token overlap?
"""
from __future__ import annotations

import json, math, pickle, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp")

CANDIDATES = ROOT / "levi_semantic_chain_candidates_v3.jsonl"
JUDGE      = ROOT / "levi_semantic_chain_omlx_judge_v3.jsonl"
OUT        = ROOT / "v3_benchmark_results.json"

CACHE_FILE = ROOT / "embedding_cache_levi.pkl"

# ── stopwords / tokenizer ────────────────────────────────────────────────────
STOP = set("""
able accepted active actual additional adjacent ahead almost already another
available basic better broader careful clean clear common concrete correct
critical current different direct earlier enough exact explicit external final
first fresh full general good great hard high human important initial internal
known large latest likely live local long main major meaningful messy minimal
native new next obvious old ongoing only operational other personal possible
primary prior raw real recent related relevant same second separate simple
small specific stable strong sure technical top true useful weak whole
system memory file data user note output text block token result type name
status created updated source boundary tags evidence confidence tier person
organization entity service endpoint project work time week day month year
back going make makes made used using uses been have will would could should
also just like much very some more most than from this that with they them
their about into over under when where which while
""".split())

def toks(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'[a-z]{4,}', text.lower())]

def tok_ok(t: str) -> bool:
    return t not in STOP and 5 <= len(t) <= 20 and not t[:4].isdigit()

# ── corpus + index ────────────────────────────────────────────────────────────
def load_corpus():
    import rfm_substrate_path_test as substrate
    return substrate.load_notes()

class Index:
    def __init__(self, notes):
        self.nb = {n["id"]: n for n in notes}
        self.nt = {n["id"]: {t for t in toks(n["content"]) if tok_ok(t)} for n in notes}
        self.df: dict[str, int] = defaultdict(int)
        self.postings: dict[str, list[str]] = defaultdict(list)
        self.token_sources: dict[str, set[str]] = defaultdict(set)
        for nid, ts in self.nt.items():
            src = self.nb[nid]["source"]
            for t in ts:
                self.df[t] += 1
                self.postings[t].append(nid)
                self.token_sources[t].add(src)
        # IDF
        N = len(notes)
        self.idf = {t: math.log((N - d + 0.5) / (d + 0.5) + 1)
                    for t, d in self.df.items()}

    def bm25(self, query: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75):
        qtoks = [t for t in toks(query) if tok_ok(t) and t in self.df]
        if not qtoks:
            return []
        avgdl = sum(len(ts) for ts in self.nt.values()) / max(len(self.nt), 1)
        scores: dict[str, float] = defaultdict(float)
        for t in qtoks:
            idf = self.idf.get(t, 0)
            for nid in self.postings.get(t, []):
                dl = len(self.nt[nid])
                tf = sum(1 for tok in toks(self.nb[nid]["content"]) if tok == t)
                scores[nid] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        return sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]

# ── token-path v12 (inlined) ─────────────────────────────────────────────────
def _bridges(node_toks, exclude, token_sources, df, node_src, k=2):
    scored = []
    for t in node_toks:
        if t in exclude: continue
        cross = token_sources.get(t, set()) - {node_src}
        n_cross = len(cross)
        if n_cross == 0 or n_cross > 10: continue
        s = (1.0 / max(df.get(t, 1), 1) ** 0.5) * (1.0 / n_cross) * min(len(t), 12) * 0.1
        scored.append((s, t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:k]]

def path_v12(query, idx: Index, top_k=10, anchor_k=10, branch_k=8):
    qq = {t for t in toks(query) if tok_ok(t)}
    anchor_scores, terminal_scores = {}, {}
    for rank, a_id in enumerate(idx.bm25(query, anchor_k)):
        if a_id not in idx.nb: continue
        a_src = idx.nb[a_id]["source"]
        a_toks = idx.nt.get(a_id, set())
        anchor_scores[a_id] = max(anchor_scores.get(a_id, 0), 1.0 + 0.05 * (anchor_k - rank))
        for t1 in _bridges(a_toks, qq, idx.token_sources, idx.df, a_src):
            for b_id in idx.postings.get(t1, [])[:branch_k]:
                if b_id == a_id or b_id not in idx.nb: continue
                b_src = idx.nb[b_id]["source"]
                if b_src == a_src: continue
                anchor_scores[b_id] = max(anchor_scores.get(b_id, 0), 0.8)
                for t2 in _bridges(idx.nt.get(b_id, set()), qq | {t1},
                                   idx.token_sources, idx.df, b_src):
                    for c_id in idx.postings.get(t2, [])[:branch_k]:
                        if c_id in (a_id, b_id) or c_id not in idx.nb: continue
                        if idx.nb[c_id]["source"] in (a_src, b_src): continue
                        if len(idx.nt.get(c_id, set()) & a_toks) > 0: continue
                        terminal_scores[c_id] = max(terminal_scores.get(c_id, 0),
                                                     1.0 + 0.05 * (anchor_k - rank))
    tpt = sorted(terminal_scores, key=terminal_scores.__getitem__, reverse=True)
    tpa = sorted(anchor_scores, key=anchor_scores.__getitem__, reverse=True)
    n_t = min(len(tpt), max(top_k // 2, 3))
    res, seen = [], set()
    for nid in tpt[:n_t]:
        res.append(nid); seen.add(nid)
    for nid in tpa:
        if len(res) >= top_k: break
        if nid not in seen:
            res.append(nid); seen.add(nid)
    return res[:top_k]

# ── embedding-bridge retriever ────────────────────────────────────────────────
class EmbIndex:
    def __init__(self, notes, emb_ids, embs):
        self.nb = {n["id"]: n for n in notes}
        self.nt = {n["id"]: {t for t in toks(n["content"]) if tok_ok(t)} for n in notes}
        # Only notes that have embeddings
        emb_set = set(emb_ids)
        self.note_ids = [nid for nid in emb_ids if nid in self.nb]
        idx_map = {nid: i for i, nid in enumerate(emb_ids)}
        self.embs = embs[[idx_map[nid] for nid in self.note_ids]]
        self.id_to_local = {nid: i for i, nid in enumerate(self.note_ids)}
        # BM25 index
        self.df: dict[str, int] = defaultdict(int)
        self.postings: dict[str, list[str]] = defaultdict(list)
        for nid, ts in self.nt.items():
            for t in ts:
                self.df[t] += 1
                self.postings[t].append(nid)
        N = len(notes)
        self.idf = {t: math.log((N - d + 0.5) / (d + 0.5) + 1)
                    for t, d in self.df.items()}
        avgdl = sum(len(ts) for ts in self.nt.values()) / max(len(self.nt), 1)
        self._avgdl = avgdl

    def bm25(self, query: str, top_k: int = 10):
        qtoks = [t for t in toks(query) if tok_ok(t) and t in self.df]
        if not qtoks: return []
        scores: dict[str, float] = defaultdict(float)
        k1, b = 1.5, 0.75
        for t in qtoks:
            idf = self.idf.get(t, 0)
            for nid in self.postings.get(t, []):
                dl = len(self.nt[nid])
                tf = sum(1 for tok in toks(self.nb[nid]["content"]) if tok == t)
                scores[nid] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._avgdl))
        return sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]

    def sim_vec(self, nid: str):
        """Returns similarity of nid against all emb-indexed notes."""
        if nid not in self.id_to_local: return None
        return self.embs @ self.embs[self.id_to_local[nid]]


def emb_bridge(query, eidx: EmbIndex, top_k=10, anchor_k=10, branch_k=8,
               sim_bridge_min=0.45, sim_terminal_min=0.40):
    """Embedding-bridge retriever: BM25 anchor → embedding-similar bridge → embedding-similar terminal."""
    anchors = eidx.bm25(query, anchor_k)
    terminal_scores = {}
    anchor_scores = {}

    for rank, a_id in enumerate(anchors):
        if a_id not in eidx.nb: continue
        a_src = eidx.nb[a_id]["source"]
        anchor_scores[a_id] = max(anchor_scores.get(a_id, 0), 1.0 + 0.05 * (anchor_k - rank))

        # Find embedding-similar bridges (different source from A)
        sims_a = eidx.sim_vec(a_id)
        if sims_a is None: continue
        bridge_order = np.argsort(sims_a)[::-1]

        bridges_found = 0
        for b_local in bridge_order:
            if bridges_found >= branch_k: break
            b_id = eidx.note_ids[b_local]
            if b_id == a_id: continue
            if sims_a[b_local] < sim_bridge_min: break
            b_src = eidx.nb[b_id]["source"]
            if b_src == a_src: continue
            bridges_found += 1
            anchor_scores[b_id] = max(anchor_scores.get(b_id, 0), 0.8)

            # Find embedding-similar terminals (different source from A and B)
            sims_b = eidx.sim_vec(b_id)
            if sims_b is None: continue
            terminal_order = np.argsort(sims_b)[::-1]

            terminals_found = 0
            for c_local in terminal_order:
                if terminals_found >= branch_k: break
                c_id = eidx.note_ids[c_local]
                if c_id in (a_id, b_id): continue
                if sims_b[c_local] < sim_terminal_min: break
                c_src = eidx.nb[c_id]["source"]
                if c_src in (a_src, b_src): continue
                # Key filter: terminal should NOT be directly similar to anchor
                # (same logic as zero-overlap filter in token path)
                if sims_a[c_local] >= 0.50: continue
                terminals_found += 1
                score = sims_a[b_local] * sims_b[c_local] * (1.0 + 0.05 * (anchor_k - rank))
                terminal_scores[c_id] = max(terminal_scores.get(c_id, 0), score)

    tpt = sorted(terminal_scores, key=terminal_scores.__getitem__, reverse=True)
    tpa = sorted(anchor_scores, key=anchor_scores.__getitem__, reverse=True)
    n_t = min(len(tpt), max(top_k // 2, 3))
    res, seen = [], set()
    for nid in tpt[:n_t]:
        res.append(nid); seen.add(nid)
    for nid in tpa:
        if len(res) >= top_k: break
        if nid not in seen:
            res.append(nid); seen.add(nid)
    return res[:top_k]

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading corpus...")
    notes = load_corpus()
    print(f"  {len(notes)} notes")

    print("Building token index...")
    t0 = time.time()
    idx = Index(notes)
    print(f"  {len(idx.nb)} notes, {len(idx.df)} tokens, {time.time()-t0:.1f}s")

    print("Loading embeddings...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    emb_ids = cache["note_ids"]
    embs = cache["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-9)
    print(f"  {len(emb_ids)} embeddings, dim={embs.shape[1]}")

    print("Building embedding index...")
    eidx = EmbIndex(notes, emb_ids, embs)
    print(f"  {len(eidx.note_ids)} emb-indexed notes")

    # Load chains + labels
    cands = [json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip()]
    judge = {}
    for l in JUDGE.read_text().splitlines():
        try:
            d = json.loads(l)
            judge[int(d["idx"])] = d["label"]
        except: pass
    print(f"\nChains: {len(cands)}, judged: {len(judge)}")

    # Resolve required_ids to note ids
    resolved = []
    for c in cands:
        req = c["required_ids"]
        # v3 ids may already be in note format
        A, B, C = req[0], req[1], req[2]
        # verify they exist in the index
        if A not in idx.nb or B not in idx.nb or C not in idx.nb:
            continue
        lab = judge.get(int(c["idx"]), "unknown")
        resolved.append({"idx": c["idx"], "query": c["start_token"],
                         "A": A, "B": B, "C": C, "label": lab,
                         "sim_ab": c.get("sim_ab"), "sim_bc": c.get("sim_bc"),
                         "sim_ac": c.get("sim_ac")})

    print(f"Resolvable chains: {len(resolved)}")

    print("\nRunning retrievers...")
    rows = []
    for i, ch in enumerate(resolved):
        if i % 50 == 0:
            print(f"  {i}/{len(resolved)}")
        q = ch["query"]; A, B, C = ch["A"], ch["B"], ch["C"]

        t = time.time(); bm25_ids = idx.bm25(q, 10); bm25_ms = (time.time()-t)*1000
        t = time.time(); tok_ids  = path_v12(q, idx, 10); tok_ms = (time.time()-t)*1000
        t = time.time(); emb_ids_ = emb_bridge(q, eidx, 10); emb_ms = (time.time()-t)*1000

        def stats(ids):
            s = set(ids)
            return {"term_hit": C in s, "full_path": A in s and B in s and C in s, "anchor_hit": A in s}

        rows.append({
            "idx": ch["idx"], "label": ch["label"], "query": q,
            "sim_ab": ch["sim_ab"], "sim_bc": ch["sim_bc"], "sim_ac": ch["sim_ac"],
            "bm25": stats(bm25_ids), "bm25_ms": bm25_ms,
            "tok":  stats(tok_ids),  "tok_ms":  tok_ms,
            "emb":  stats(emb_ids_), "emb_ms":  emb_ms,
        })

    # ── Results ────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("V3 SEMANTIC CHAIN BENCHMARK RESULTS")
    print("(chains mined via embedding similarity, not token coincidence)")
    print("="*80)

    for slice_name, fn in [
        ("real_semantic",   lambda r: r["label"] == "real_semantic"),
        ("weak_semantic",   lambda r: r["label"] == "weak_semantic"),
        ("real+weak",       lambda r: r["label"] in ("real_semantic", "weak_semantic")),
        ("all",             lambda r: True),
    ]:
        sl = [r for r in rows if fn(r)]
        n = len(sl)
        if n == 0: continue
        print(f"\n----- slice={slice_name} (n={n}) -----")
        print(f"  {'metric':<30} {'BM25':>10} {'token-v12':>12} {'emb-bridge':>12}")
        for metric, get in [
            ("terminal hit@10",      lambda r,k: r[k]["term_hit"]),
            ("full-path recovery",   lambda r,k: r[k]["full_path"]),
            ("anchor hit@10",        lambda r,k: r[k]["anchor_hit"]),
        ]:
            b = sum(1 for r in sl if get(r, "bm25")) / n * 100
            t = sum(1 for r in sl if get(r, "tok"))  / n * 100
            e = sum(1 for r in sl if get(r, "emb"))  / n * 100
            print(f"  {metric:<30} {b:>9.1f}% {t:>11.1f}% {e:>11.1f}%")

    def med(xs): xs=sorted(xs); return xs[len(xs)//2]
    print(f"\n----- timing (median ms) -----")
    print(f"  BM25:       {med([r['bm25_ms'] for r in rows]):.1f}ms")
    print(f"  token-v12:  {med([r['tok_ms']  for r in rows]):.1f}ms")
    print(f"  emb-bridge: {med([r['emb_ms']  for r in rows]):.1f}ms")

    OUT.write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2))
    print(f"\nRaw results saved to: {OUT}")

if __name__ == "__main__":
    main()
