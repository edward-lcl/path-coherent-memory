#!/usr/bin/env python3
"""
Cross-session concept chain miner for agent runtime memory.

Parses the stable OCPlatform main-agent session jsonls into topic chunks
(one chunk = a coherent topic block within a session), then mines A→B→C chains
where:
  - A and C are in DIFFERENT sessions (the cross-thread "should compound" case)
  - B is a bridge chunk (in any session) with sim(A,B) > SIM_AB_MIN and
    sim(B,C) > SIM_BC_MIN
  - sim(A,C) < SIM_AC_MAX  (genuinely concept-disjoint, not lexically linked)
  - tok_overlap(A,C) <= TOK_AC_MAX  (not lexically bridgeable)

This is exactly the "threads that should compound but don't" regime Edward
described — the retrieval system sees A and C as unrelated, but B is the
conceptual bridge a human (or path-coherent traversal) would follow.

Output: agent_session_chains.jsonl — same format as talos_semantic_chain_candidates_v2.jsonl
so the existing talos_complementarity_eval.py / talos_clean_eval.py harness
can be reused almost as-is.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
SESSION_DIR = Path("/Users/edward/.ocplatform/agents/main/sessions")
OUT = ROOT / "agent_session_chains.jsonl"
CACHE = ROOT / "agent_session_embeddings.npy"
NODES_CACHE = ROOT / "agent_session_nodes.json"

# Chain quality gates (match v4 miner)
SIM_AB_MIN = 0.48
SIM_BC_MIN = 0.48
SIM_AC_MAX = 0.38
TOK_AC_MAX = 1
HUB_MAX = 35
SIM_HUB = 0.50
TARGET = 300
CHUNK_WORDS = 120   # target words per topic chunk
CHUNK_OVERLAP = 20  # overlap words between chunks

STOP = set("""
able accepted active actual additional also about basic better clean clear common correct
current different direct enough exact external final first full general good great hard
high human important initial known large latest likely live local long main major
meaningful minimal native new next obvious old ongoing only operational other personal
possible primary prior raw real recent related relevant same second separate simple
small specific stable strong sure technical top true useful weak whole system memory
file data user note output text block token result type name status created updated
source boundary tags evidence confidence tier person organization entity service
endpoint project work time week day month year back going make makes made used using
been have will would could should just like much very some more most than from this
that with they them their into over under when where which while
""".split())


def toks(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'[a-z]{4,}', text.lower())
            if w.lower() not in STOP]


def extract_text(message: dict) -> str:
    """Pull plain text from a message record."""
    content = message.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def chunk_text(text: str, session_id: str, chunk_idx: int) -> list[dict]:
    """Split a long text into overlapping topic chunks."""
    words = text.split()
    if len(words) < 30:
        return []
    chunks = []
    i = 0
    ci = chunk_idx
    while i < len(words):
        w = words[i:i + CHUNK_WORDS]
        if len(w) < 20:
            break
        chunk_text = " ".join(w)
        chunks.append({
            "id": f"{session_id}::c{ci}",
            "session_id": session_id,
            "source": f"session/{session_id[:8]}",
            "content": chunk_text,
            "word_count": len(w),
        })
        ci += 1
        i += CHUNK_WORDS - CHUNK_OVERLAP
    return chunks, ci


def load_nodes() -> list[dict]:
    if NODES_CACHE.exists():
        nodes = json.loads(NODES_CACHE.read_text())
        print(f"  loaded {len(nodes)} cached nodes")
        return nodes

    print("Parsing session jsonls into topic chunks…")
    files = sorted(
        [f for f in SESSION_DIR.glob("*.jsonl")
         if ".bak" not in f.name and "trajectory" not in f.name],
        key=lambda f: f.stat().st_mtime
    )
    print(f"  {len(files)} session files")
    nodes = []
    for f in files:
        sid = f.stem
        try:
            records = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        except Exception:
            continue
        # Group consecutive assistant+user turns into topic blocks by role
        # then chunk each block independently.
        block_text = []
        ci = 0
        for rec in records:
            if rec.get("type") != "message":
                continue
            role = rec.get("message", {}).get("role", "")
            if role not in ("user", "assistant"):
                continue
            txt = extract_text(rec).strip()
            if len(txt.split()) < 15:
                continue
            block_text.append(txt)
            # flush to chunks every ~4 turns to keep blocks topically coherent
            if len(block_text) >= 4:
                full = " ".join(block_text)
                result = chunk_text(full, sid, ci)
                if result:
                    chunks, ci = result
                    nodes.extend(chunks)
                block_text = []
        if block_text:
            full = " ".join(block_text)
            result = chunk_text(full, sid, ci)
            if result:
                chunks, ci = result
                nodes.extend(chunks)

    print(f"  {len(nodes)} chunks from {len(files)} sessions")
    NODES_CACHE.write_text(json.dumps(nodes))
    return nodes


def embed_nodes(nodes: list[dict]) -> np.ndarray:
    sys.path.insert(0, str(ROOT))
    from embedding_bridge_retriever import embed_texts

    if CACHE.exists():
        vecs = np.load(CACHE)
        if vecs.shape[0] == len(nodes):
            print(f"  loaded cached embeddings ({vecs.shape})")
            return vecs
        print(f"  cache shape mismatch ({vecs.shape[0]} vs {len(nodes)}), re-embedding")

    print(f"  embedding {len(nodes)} nodes…")
    texts = [n["content"][:512] for n in nodes]
    CKPT = CACHE.with_suffix(".partial.npy")
    batch = 256
    done = []
    if CKPT.exists():
        done_arr = np.load(CKPT)
        done = list(done_arr)
        print(f"  resuming from checkpoint at {len(done)}")
    for i in range(len(done), len(texts), batch):
        chunk = texts[i:i + batch]
        vecs = embed_texts(chunk, batch_size=256)
        done.extend(vecs)
        np.save(CKPT, np.array(done, dtype=np.float32))
        print(f"  checkpoint {len(done)}/{len(texts)}")
    vecs = np.array(done[:len(nodes)], dtype=np.float32)
    np.save(CACHE, vecs)
    return vecs


def mine(nodes: list[dict], vecs: np.ndarray) -> list[dict]:
    print(f"\nMining cross-session chains (target={TARGET})…")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    V = vecs / norms

    # Precompute hub scores (how many neighbors sim > SIM_HUB)
    print("  computing hub scores…")
    hub_counts = np.zeros(len(nodes), dtype=np.int32)
    batch = 512
    for i in range(0, len(nodes), batch):
        sims = V[i:i+batch] @ V.T
        hub_counts[i:i+batch] = (sims > SIM_HUB).sum(axis=1) - 1  # exclude self
    eligible_b = np.where(hub_counts <= HUB_MAX)[0]
    print(f"  eligible bridges (hub≤{HUB_MAX}): {len(eligible_b)}")

    # Index sessions
    by_session = defaultdict(list)
    for idx, n in enumerate(nodes):
        by_session[n["session_id"]].append(idx)
    sessions = list(by_session.keys())

    chains = []
    seen_ac = set()

    # For each node A, find good bridges B, then terminals C in a DIFFERENT session
    import random
    random.seed(42)
    a_idxs = list(range(len(nodes)))
    random.shuffle(a_idxs)

    for a_idx in a_idxs:
        if len(chains) >= TARGET:
            break
        a = nodes[a_idx]
        a_sess = a["session_id"]
        a_vec = V[a_idx]
        a_toks = set(toks(a["content"]))

        # Candidate bridges: top sim to A, excluding hubs, any session
        sims_ab = V @ a_vec
        top_b = np.argsort(-sims_ab)
        good_b = [i for i in top_b[1:50]
                  if i in set(eligible_b) and sims_ab[i] >= SIM_AB_MIN
                  and i != a_idx][:8]

        for b_idx in good_b:
            if len(chains) >= TARGET:
                break
            b_vec = V[b_idx]
            sims_bc = V @ b_vec

            # Terminal C: different session from A, embedding-disjoint with A
            sims_ac = V @ a_vec
            top_c = np.argsort(-sims_bc)
            for c_idx in top_c[1:80]:
                c = nodes[c_idx]
                if c["session_id"] == a_sess:
                    continue  # must be cross-session
                if sims_bc[c_idx] < SIM_BC_MIN:
                    break
                if sims_ac[c_idx] >= SIM_AC_MAX:
                    continue  # not disjoint enough
                # Lexical overlap gate
                c_toks_set = set(toks(c["content"]))
                overlap = len(a_toks & c_toks_set)
                if overlap > TOK_AC_MAX:
                    continue
                pair = (a_idx, c_idx)
                if pair in seen_ac:
                    continue
                seen_ac.add(pair)
                chains.append({
                    "idx": len(chains),
                    "a_id": a["id"], "b_id": nodes[b_idx]["id"], "c_id": c["id"],
                    "a_sess": a_sess[:8], "b_sess": nodes[b_idx]["session_id"][:8],
                    "c_sess": c["session_id"][:8],
                    "sim_ab": float(sims_ab[b_idx]),
                    "sim_bc": float(sims_bc[c_idx]),
                    "sim_ac": float(sims_ac[c_idx]),
                    "tok_overlap": overlap,
                    "cross_session": a_sess != c["session_id"],
                    "a_excerpt": a["content"][:200],
                    "b_excerpt": nodes[b_idx]["content"][:200],
                    "c_excerpt": c["content"][:200],
                })
                break  # one C per B

    print(f"  mined {len(chains)} chains ({sum(c['cross_session'] for c in chains)} cross-session)")
    return chains


def main():
    nodes = load_nodes()
    if not nodes:
        print("No nodes extracted — check SESSION_DIR")
        return
    vecs = embed_nodes(nodes)
    chains = mine(nodes, vecs)
    OUT.write_text("\n".join(json.dumps(c) for c in chains) + "\n")
    # Stats
    sims_ac = [c["sim_ac"] for c in chains]
    import statistics
    print(f"\nChain stats:")
    print(f"  total={len(chains)}, cross-session={sum(c['cross_session'] for c in chains)}")
    print(f"  sim_ac: mean={statistics.mean(sims_ac):.3f} max={max(sims_ac):.3f}")
    print(f"  tok_overlap=0: {sum(c['tok_overlap']==0 for c in chains)}")
    print(f"\nSaved → {OUT}")
    # Show a sample chain
    if chains:
        ch = chains[0]
        print(f"\nSample chain (sess {ch['a_sess']}→{ch['b_sess']}→{ch['c_sess']}):")
        print(f"  A: {ch['a_excerpt'][:120]}")
        print(f"  B: {ch['b_excerpt'][:120]}")
        print(f"  C: {ch['c_excerpt'][:120]}")
        print(f"  sim_ac={ch['sim_ac']:.3f} tok_overlap={ch['tok_overlap']}")


if __name__ == "__main__":
    main()
