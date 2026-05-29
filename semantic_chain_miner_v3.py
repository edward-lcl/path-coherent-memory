#!/usr/bin/env python3
"""
Semantic Chain Miner v3

Mines chains where A→B→C connections are *semantic* (embedding similarity),
not lexical (rare token co-occurrence). This gives the embedding-bridge retriever
its native benchmark family.

Design:
  - A and C share zero/low direct embedding similarity (< SIM_AC_MAX)
  - A→B are embedding-similar (> SIM_AB_MIN) but from different sources
  - B→C are embedding-similar (> SIM_BC_MIN) but from different sources
  - A has at least one distinctive token (unique or rare) for BM25 anchoring
  - B is the semantic bridge that makes the path traversable by embedding-bridge retrieval

Architecture:
  - Loads the existing 9788-note Levi embedding cache (Qwen3-Embedding-0.6B)
  - Groups notes by source file to enforce cross-source hops
  - Mines A→B pairs first, then extends to A→B→C by checking B-neighborhood
  - Deduplicates chains by (A, C) pair to avoid redundant paths
  - Outputs JSONL in the same schema as mine_candidates_v2.py for judge compatibility

This tests the RFM embedding hypothesis on its native problem — semantic gap
traversal where the vocabulary gap is *conceptual* rather than lexical.
"""
from __future__ import annotations

import json, pickle, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / "embedding_cache_levi.pkl"
OUT_CANDIDATES = ROOT / "levi_semantic_chain_candidates_v3.jsonl"

# ── thresholds ────────────────────────────────────────────────────────────────
SIM_AB_MIN = 0.55      # A and B must be at least this similar (semantic adjacency)
SIM_BC_MIN = 0.55      # B and C must be at least this similar
SIM_AC_MAX = 0.40      # A and C must NOT be this similar (vocabulary gap required)
TOK_AC_MAX = 1         # A and C share at most this many non-stopword tokens (lexical isolation)
TARGET = 300

# ── stopwords / token helpers ─────────────────────────────────────────────────
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
organization entity service endpoint building project work time week day month
year back going going make makes making made used using uses been have will
would could should also just like much very some more most than from this
that with they them their about into over under when where which while
""".split())

def toks(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'[a-z]{4,}', text.lower())]

def tok_ok(t: str) -> bool:
    return t not in STOP and 5 <= len(t) <= 20 and not t[:4].isdigit()

def note_toks(content: str) -> set[str]:
    return {t for t in toks(content) if tok_ok(t)}

def source_tier(src: str) -> int:
    """Higher tier = more semantically dense source."""
    if src in {"USER.md", "MEMORY.md", "SOUL.md"}:
        return 4
    if src.startswith("memory/topics/"):
        return 4
    if src.startswith("memory/summaries/"):
        return 3
    if re.match(r"memory/\d{4}-\d{2}-\d{2}.*\.md$", src):
        return 2
    if src.startswith("memory/"):
        return 1
    return 0

def clean_for_mining(note: dict) -> bool:
    """Filter out noisy notes unsuitable for semantic chains."""
    text = note["content"]
    if source_tier(note["source"]) < 2:
        return False
    # Skip code-heavy chunks
    if text.count("`") > 8:
        return False
    code_hits = len(re.findall(
        r"/Users/|/home/|\.py\b|\.ts\b|\.json\b|commit [0-9a-f]{6}", text, re.I))
    if code_hits > 3:
        return False
    # Skip tool/log artifacts
    noisy = [r"<<<EXTERNAL_UNTRUSTED_CONTENT", r"tool call:", r"Traceback \(",
             r"\[Audio\]", r"<media:", r"^[0-9a-f]{8,40}$"]
    if any(re.search(p, text) for p in noisy):
        return False
    if len(toks(text)) < 10:
        return False
    return True

def excerpt(note: dict, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", note["content"]).strip()
    snip = text[:max_chars - 1] + ("..." if len(text) > max_chars else "")
    return f"[{note['source']}] {snip}"

# ── corpus loading ────────────────────────────────────────────────────────────
def load_substrate() -> list[dict]:
    sys.path.insert(0, "/tmp")
    import rfm_substrate_path_test as substrate
    return substrate.load_notes()

# ── embedding cache ───────────────────────────────────────────────────────────
def load_embeddings() -> tuple[list[str], np.ndarray]:
    print("Loading embedding cache...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    ids = cache["note_ids"]
    embs = cache["embeddings"].astype(np.float32)
    # Ensure L2-normalized
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-9)
    print(f"  {len(ids)} notes, dim={embs.shape[1]}")
    return ids, embs

# ── mining ────────────────────────────────────────────────────────────────────
def cosine_row(embs: np.ndarray, idx: int) -> np.ndarray:
    """Cosine similarity of note[idx] against all notes (already L2-normalized)."""
    return embs @ embs[idx]

def mine(notes_all: list[dict], emb_ids: list[str], embs: np.ndarray,
         target: int = TARGET) -> list[dict]:
    """
    Core mining loop.

    Strategy:
      1. Filter notes to mining-eligible set.
      2. Build reverse map: note_id → position in emb matrix.
      3. Build df (token document frequency) over eligible notes for anchor scoring.
      4. For each eligible note B (potential bridge):
           - find A candidates: emb_sim > SIM_AB_MIN, different source
           - find C candidates: emb_sim > SIM_BC_MIN, different source, different from A's source
           - for each (A, C) pair: check A↔C sim < SIM_AC_MAX and tok overlap ≤ TOK_AC_MAX
           - score and emit chain
      5. Deduplicate by (A_id, C_id), sort by score, take top `target`.
    """
    # Index: note_id → note
    nb = {n["id"]: n for n in notes_all}

    # Filter to embedding-indexed notes only
    emb_id_set = set(emb_ids)
    emb_pos = {nid: i for i, nid in enumerate(emb_ids)}

    eligible = [
        n for n in notes_all
        if n["id"] in emb_id_set and clean_for_mining(n)
    ]
    print(f"eligible notes for mining: {len(eligible)}")

    eligible_ids = [n["id"] for n in eligible]
    eligible_set = set(eligible_ids)
    eligible_pos = {nid: emb_pos[nid] for nid in eligible_ids}

    # Eligible embedding matrix slice (for fast sim computation)
    elig_indices = np.array([emb_pos[nid] for nid in eligible_ids], dtype=np.int32)
    elig_embs = embs[elig_indices]  # shape: (N_elig, dim)

    # Token index for eligible notes
    nt: dict[str, set[str]] = {n["id"]: note_toks(n["content"]) for n in eligible}

    # Document frequency (over eligible set only)
    df: dict[str, int] = defaultdict(int)
    for toks_set in nt.values():
        for t in toks_set:
            df[t] += 1

    # Source grouping
    by_source: dict[str, list[str]] = defaultdict(list)
    for n in eligible:
        by_source[n["source"]].append(n["id"])

    # Pre-compute full eligible sim matrix (N_elig × N_elig) — feasible at ~5K notes
    # For larger corpora, switch to a batched approach
    print(f"Computing {len(eligible_ids)}×{len(eligible_ids)} similarity matrix...")
    sim_matrix = elig_embs @ elig_embs.T  # shape: (N_elig, N_elig)
    print("  done.")

    # local index: eligible position
    elig_local_pos = {nid: i for i, nid in enumerate(eligible_ids)}

    chains: list[tuple] = []
    seen_ac: set[tuple[str, str]] = set()

    n_elig = len(eligible_ids)
    print(f"Mining chains from {n_elig} eligible notes...")

    for b_idx, b_id in enumerate(eligible_ids):
        if b_idx % 1000 == 0:
            print(f"  processed {b_idx}/{n_elig} bridge candidates, chains so far: {len(chains)}")

        b_src = nb[b_id]["source"]
        b_tier = source_tier(b_src)
        if b_tier < 2:
            continue

        sims_b = sim_matrix[b_idx]  # similarities to all eligible notes

        # A candidates: similar to B, different source
        a_mask = (sims_b >= SIM_AB_MIN)
        a_candidates = [eligible_ids[i] for i in range(n_elig)
                        if a_mask[i] and eligible_ids[i] != b_id
                        and nb[eligible_ids[i]]["source"] != b_src]

        if not a_candidates:
            continue

        # C candidates: similar to B, different source, must differ from all A sources
        # We'll check per (A,C) pair below — just gather all C != B and != A-source
        c_candidates_all = [eligible_ids[i] for i in range(n_elig)
                            if sims_b[i] >= SIM_BC_MIN
                            and eligible_ids[i] != b_id]

        if not c_candidates_all:
            continue

        for a_id in a_candidates:
            a_src = nb[a_id]["source"]
            a_tier = source_tier(a_src)
            if a_tier < 2:
                continue

            # Need a distinctive anchor token from A for BM25 anchoring
            a_toks = nt[a_id]
            anchor_toks = [t for t in a_toks if df.get(t, 0) <= 4 and len(t) >= 5]
            if not anchor_toks:
                continue
            # Pick the rarest anchor token (best BM25 anchor)
            anchor_tok = min(anchor_toks, key=lambda t: df.get(t, 0))

            a_local = elig_local_pos[a_id]
            sims_a = sim_matrix[a_local]  # A vs all eligible

            for c_id in c_candidates_all:
                if c_id == a_id:
                    continue
                c_src = nb[c_id]["source"]
                if c_src == a_src or c_src == b_src:
                    continue
                c_tier = source_tier(c_src)
                if c_tier < 2:
                    continue

                # Key constraint: A and C must NOT be directly similar
                c_local = elig_local_pos[c_id]
                sim_ac = sims_a[c_local]
                if sim_ac >= SIM_AC_MAX:
                    continue

                # Lexical isolation: A and C share ≤ TOK_AC_MAX tokens
                tok_overlap = len(nt[a_id] & nt[c_id])
                if tok_overlap > TOK_AC_MAX:
                    continue

                # Dedup: only one chain per (A, C) pair
                ac_key = (a_id, c_id)
                if ac_key in seen_ac:
                    continue
                seen_ac.add(ac_key)

                # Score: reward high A-B and B-C similarity, penalize A-C similarity
                # Also reward source tier and anchor token rarity
                sim_ab = sims_b[a_local]  # sims_b[i] = sim(B, eligible[i])
                sim_bc = sims_b[c_local]
                tier_score = a_tier + b_tier + c_tier
                anchor_rarity = 1.0 / max(df.get(anchor_tok, 1), 1)

                score = (
                    sim_ab * 4.0
                    + sim_bc * 4.0
                    - sim_ac * 6.0       # strongly penalize direct A-C similarity
                    - tok_overlap * 2.0
                    + tier_score * 0.3
                    + anchor_rarity * 2.0
                )

                chains.append((score, anchor_tok, sim_ab, sim_bc, sim_ac,
                               a_id, b_id, c_id))

        if len(chains) >= target * 20:
            # Have plenty, stop early to avoid memory explosion
            break

    print(f"Raw chain candidates: {len(chains)}")
    chains.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by A start token to ensure query diversity
    seen_starts: set[str] = set()
    seen_ac_final: set[tuple[str, str]] = set()
    results = []

    for score, anchor, sim_ab, sim_bc, sim_ac, a_id, b_id, c_id in chains:
        if len(results) >= target:
            break
        ac_key = (a_id, c_id)
        if anchor in seen_starts or ac_key in seen_ac_final:
            continue
        seen_starts.add(anchor)
        seen_ac_final.add(ac_key)

        a_note = nb[a_id]; b_note = nb[b_id]; c_note = nb[c_id]
        results.append({
            "idx": len(results) + 1,
            "score": round(float(score), 4),
            "miner": "semantic_v3",
            "start_token": anchor,
            "sim_ab": round(float(sim_ab), 4),
            "sim_bc": round(float(sim_bc), 4),
            "sim_ac": round(float(sim_ac), 4),
            "bridge_type": "embedding",
            "sources": " | ".join([a_note["source"], b_note["source"], c_note["source"]]),
            "required_ids": [a_id, b_id, c_id],
            "excerpts": {
                "a": excerpt(a_note),
                "b": excerpt(b_note),
                "c": excerpt(c_note),
            },
        })

    return results


def main() -> None:
    notes_all = load_substrate()
    emb_ids, embs = load_embeddings()

    results = mine(notes_all, emb_ids, embs, TARGET)
    print(f"\nMined {len(results)} semantic chains (target {TARGET})")

    with OUT_CANDIDATES.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_CANDIDATES}")

    print("\nSample (first 10):")
    for r in results[:10]:
        print(f"  {r['idx']:03d}  anchor={r['start_token']:<18}  "
              f"sim_ab={r['sim_ab']:.3f}  sim_bc={r['sim_bc']:.3f}  sim_ac={r['sim_ac']:.3f}  "
              f"sources={r['sources'][:80]}")

    # Distribution stats
    sims_ab = [r['sim_ab'] for r in results]
    sims_bc = [r['sim_bc'] for r in results]
    sims_ac = [r['sim_ac'] for r in results]
    print(f"\nsim_ab: mean={np.mean(sims_ab):.3f}  min={np.min(sims_ab):.3f}  max={np.max(sims_ab):.3f}")
    print(f"sim_bc: mean={np.mean(sims_bc):.3f}  min={np.min(sims_bc):.3f}  max={np.max(sims_bc):.3f}")
    print(f"sim_ac: mean={np.mean(sims_ac):.3f}  min={np.min(sims_ac):.3f}  max={np.max(sims_ac):.3f}")

    # Source distribution
    from collections import Counter
    src_counts = Counter()
    for r in results:
        for s in r["sources"].split(" | "):
            src_counts[s] += 1
    print("\nTop source files in chains:")
    for s, c in src_counts.most_common(10):
        print(f"  {c:3d}  {s}")


if __name__ == "__main__":
    main()
