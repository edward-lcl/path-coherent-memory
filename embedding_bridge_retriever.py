"""
Embedding bridge retriever — replaces token bridge selection with embedding similarity.

Instead of picking bridge tokens (shared rare tokens across source files), we find
bridge notes by embedding similarity: notes that are semantically adjacent to the anchor
but come from different source files. Then hop to terminal via the same mechanism.

This tests whether semantic proximity captures cross-document connections that token
overlap misses — and directly tests the RFM hypothesis (field-like representation
over sequences).
"""
import json, re, math, time, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/Users/edward/.ocplatform/workspace/research/rfm")
CACHE_FILE = ROOT / "embedding_cache_levi.pkl"

# ── tokenizer helpers (same as path_coherent) ────────────────────────────────
STOP = set("able accepted active actual additional basic better clean clear common correct "
    "current different direct enough exact external final first full general good great hard "
    "high human important initial known large latest likely live local long main major "
    "meaningful messy minimal native new next obvious old ongoing only operational other "
    "personal possible primary prior raw real recent related relevant same second separate "
    "simple small specific stable strong sure technical top true useful weak whole system "
    "memory file data user note output text block token result type name status created "
    "updated source boundary tags evidence confidence tier person organization entity".split())

def toks(text):
    return [w.lower() for w in re.findall(r'[a-z]{4,}', text.lower())]

def tok_ok(t):
    return t not in STOP and 5 <= len(t) <= 20 and not t[:4].isdigit()

# ── embedding layer ───────────────────────────────────────────────────────────
_model = None
_tokenizer = None

EMB_MODEL_PATH = "/Users/edward/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

def get_model():
    global _model, _tokenizer
    if _model is None:
        print("Loading embedding model...")
        import mlx_embeddings
        _model, _tokenizer = mlx_embeddings.load(EMB_MODEL_PATH)
        print("Model loaded.")
    return _model, _tokenizer

def embed_texts(texts, batch_size=16):
    import mlx_embeddings
    import mlx.core as mx
    model, tokenizer = get_model()
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = mlx_embeddings.generate(model, tokenizer, batch)
        vecs = np.array(out.text_embeds.tolist(), dtype=np.float32)
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-9)
        all_vecs.append(vecs)
        if (i // batch_size) % 5 == 0:
            print(f"  embedded {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.vstack(all_vecs)

# ── corpus loading ────────────────────────────────────────────────────────────
def load_corpus():
    import sys
    sys.path.insert(0, "/tmp")
    import rfm_substrate_path_test as substrate
    notes_raw = substrate.load_notes()
    notes = []
    for n in notes_raw:
        t = [w for w in toks(n["content"]) if tok_ok(w)]
        if len(t) >= 6:
            notes.append({**n, "_toks": set(t)})
    return notes

def build_bm25_index(notes):
    df = defaultdict(int)
    postings = defaultdict(list)
    for n in notes:
        for t in n["_toks"]: df[t] += 1
    for n in notes:
        for t in n["_toks"]: postings[t].append(n["id"])
    return df, postings

def bm25_retrieve(query_toks, postings, df, N, top_k=10):
    scores = {}
    for t in set(query_toks):
        if t not in postings: continue
        idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
        for nid in postings[t]:
            scores[nid] = scores.get(nid, 0) + idf
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

# ── embedding bridge retriever ────────────────────────────────────────────────
def embedding_bridge_retrieve(query, notes, nb, embeddings, df, postings, N,
                               anchor_k=10, bridge_top_k=5, branch_k=8, top_k=10):
    """
    1. BM25 anchor retrieval (same as before)
    2. For each anchor, find semantically similar notes from DIFFERENT source files
       (embedding bridge instead of token bridge)
    3. For each bridge note, find semantically similar terminals from DIFFERENT source files
    4. Zero-overlap filter on terminals vs anchor
    """
    qt = [t for t in toks(query) if tok_ok(t)]
    anchor_ids = bm25_retrieve(qt, postings, df, N, top_k=anchor_k)
    note_ids = [n["id"] for n in notes]
    id_to_idx = {nid: i for i, nid in enumerate(note_ids)}

    terminal_scores = {}
    anchor_scores = {}

    for rank, a_id in enumerate(anchor_ids):
        if a_id not in id_to_idx: continue
        a_idx = id_to_idx[a_id]
        a_src = nb[a_id]["source"]
        a_toks = nb[a_id]["_toks"]
        anchor_scores[a_id] = max(anchor_scores.get(a_id, 0), 1.0 + 0.05*(anchor_k - rank))

        # embedding similarity to all notes
        a_vec = embeddings[a_idx]
        sims = embeddings @ a_vec  # dot product (already L2 normalized)

        # bridge candidates: top similar notes from different source files
        sim_ranked = np.argsort(-sims)
        bridges = []
        for idx in sim_ranked:
            if len(bridges) >= bridge_top_k: break
            nid = note_ids[idx]
            if nid == a_id: continue
            if nb[nid]["source"] == a_src: continue  # must be different source
            if sims[idx] < 0.3: break  # similarity floor
            bridges.append((nid, sims[idx]))

        for b_id, b_sim in bridges:
            if b_id == a_id: continue
            anchor_scores[b_id] = max(anchor_scores.get(b_id, 0), 0.8)
            b_idx = id_to_idx[b_id]
            b_src = nb[b_id]["source"]
            b_vec = embeddings[b_idx]

            # second hop: similar to bridge, different source
            b_sims = embeddings @ b_vec
            b_sim_ranked = np.argsort(-b_sims)
            hop2_count = 0
            for idx2 in b_sim_ranked:
                if hop2_count >= branch_k: break
                c_id = note_ids[idx2]
                if c_id in {a_id, b_id}: continue
                if nb[c_id]["source"] in {a_src, b_src}: continue
                if b_sims[idx2] < 0.3: break
                # zero-overlap filter
                if len(nb[c_id]["_toks"] & a_toks) > 0: continue
                terminal_scores[c_id] = max(
                    terminal_scores.get(c_id, 0),
                    (1.0 + 0.05*(anchor_k - rank)) * b_sim * b_sims[idx2]
                )
                hop2_count += 1

    top_t = sorted(terminal_scores, key=lambda x: terminal_scores[x], reverse=True)
    top_a = sorted(anchor_scores, key=lambda x: anchor_scores[x], reverse=True)
    n_t = min(len(top_t), max(top_k//2, 3))
    res, seen = [], set()
    for nid in top_t[:n_t]: res.append(nid); seen.add(nid)
    for nid in top_a:
        if len(res) >= top_k: break
        if nid not in seen: res.append(nid); seen.add(nid)
    return res[:top_k]

# ── main benchmark ────────────────────────────────────────────────────────────
def main():
    print("Loading corpus...")
    notes = load_corpus()
    nb = {n["id"]: n for n in notes}
    df, postings = build_bm25_index(notes)
    N = len(notes)
    print(f"Corpus: {N} notes")

    # Load or compute embeddings
    if CACHE_FILE.exists():
        print("Loading cached embeddings...")
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        embeddings = cached["embeddings"]
        note_ids_cached = cached["note_ids"]
        if note_ids_cached != [n["id"] for n in notes]:
            print("Cache mismatch — recomputing...")
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        print(f"Computing embeddings for {N} notes...")
        texts = [n["content"][:512] for n in notes]
        t0 = time.time()
        embeddings = embed_texts(texts)
        print(f"Done in {time.time()-t0:.0f}s. Shape: {embeddings.shape}")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"embeddings": embeddings, "note_ids": [n["id"] for n in notes]}, f)
        print(f"Cached to {CACHE_FILE}")

    # Load judged chains
    judge_file = ROOT / "levi_semantic_chain_omlx_judge_v1.jsonl"
    cand_file = ROOT / "levi_semantic_chain_candidates_v1.jsonl"
    judge = {int(json.loads(l)["idx"]): json.loads(l)["label"]
             for l in judge_file.read_text().splitlines() if l}
    candidates = [json.loads(l) for l in cand_file.read_text().splitlines() if l]

    def bench_emb(label_filter, top_k=10):
        chains = [c for c in candidates if judge.get(int(c["idx"])) in label_filter]
        t = f = 0
        for c in chains:
            req = c["required_ids"]
            ret = embedding_bridge_retrieve(
                c["start_token"], notes, nb, embeddings, df, postings, N, top_k=top_k
            )
            t += int(req[2] in ret)
            f += int(all(x in ret for x in req))
        return len(chains), t, f

    def bench_bm25(label_filter, top_k=10):
        chains = [c for c in candidates if judge.get(int(c["idx"])) in label_filter]
        t = f = 0
        for c in chains:
            req = c["required_ids"]
            qt = [w for w in toks(c["start_token"]) if tok_ok(w)]
            ret = bm25_retrieve(qt, postings, df, N, top_k=top_k)
            t += int(req[2] in ret)
            f += int(all(x in ret for x in req))
        return len(chains), t, f

    print("\n=== Embedding Bridge Retriever vs BM25 (Levi corpus) ===\n")
    for label_set, name in [({"real_semantic"}, "real_semantic"), ({"real_semantic","weak_semantic"}, "real+weak")]:
        n2, bt, bf = bench_bm25(label_set)
        _, et, ef = bench_emb(label_set)
        print(f"{name} (n={n2})")
        print(f"  BM25         terminal {100*bt/n2:5.1f}%  full {100*bf/n2:4.1f}%")
        print(f"  emb_bridge   terminal {100*et/n2:5.1f}%  full {100*ef/n2:4.1f}%")
        print(f"  delta: {100*(et-bt)/n2:+.1f}pp terminal")
        print()

if __name__ == "__main__":
    main()
