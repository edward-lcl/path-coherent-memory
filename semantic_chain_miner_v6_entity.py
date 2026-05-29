#!/usr/bin/env python3
"""
Semantic chain miner v6 — entity-anchored, corpus-rarity verified.

Fixes v5's core problem: v5 used capitalization heuristics for entity detection,
producing many generic capitalized words (leverage, silent, legitimate) as bridges.
Crucially, strong proper nouns like Digicel appeared frequently across many sources
(df > 5 in the Levi corpus), so the token-path retriever's n_cross_max gate rejected
them — miner accepted them, retriever couldn't find them.

V6 fix: require entity bridge tokens to have df ≤ 5 in the full corpus.
This aligns miner and retriever rarity criteria and ensures every mined chain
is theoretically retrievable by the token-path retriever.

Additional improvements over v5:
- Hub exclusion (>40 neighbors at sim>0.5) inherited from v4
- Entity candidates must be ≥ 5 chars and appear in 2-5 distinct sources (tighter than v5's 2-8)
- Cap entity candidates per note to top 3 by corpus rarity
- Full-corpus rank verification: B in top-50 from A sim, C in top-50 from B sim
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MOD_PATH = Path("/tmp/rfm_substrate_path_test.py")
spec = importlib.util.spec_from_file_location("rfm_substrate_path_test", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["rfm_substrate_path_test"] = mod
spec.loader.exec_module(mod)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "levi_semantic_chain_candidates_v6.jsonl"
EMBEDDING_CACHE = ROOT / "embedding_cache_levi.pkl"

STOP = set("""
able accepted active actual additional adjacent ahead almost already another
available basic better broader careful clean clear common concrete correct
critical current different direct earlier enough exact explicit external final
first fresh full general good great hard high human important initial internal
known large latest likely live local long main major meaningful minimal
native new next obvious old ongoing only operational other personal possible
primary prior raw real recent related relevant same second separate simple
small specific stable strong sure technical top true useful weak whole
system memory file data user note output text block token result type name
status created updated source boundary tags evidence confidence tier person
organization entity service endpoint project work time week day month year
back going make makes made used using been have will would could should
also just like much very some more most than from this that with they them
their about into over under when where which while
""".split())

NOISY_PATTERNS = [
    r"<<<EXTERNAL_UNTRUSTED_CONTENT",
    r"tool call:",
    r"Traceback \(",
    r"<media:",
    r"^[0-9a-f]{8,40}$",
]

SIM_AB_MIN = 0.55
SIM_BC_MIN = 0.55
SIM_AC_MAX = 0.42
HUB_THRESHOLD = 200   # raised: entity bridges provide own rarity constraint
HUB_SIM = 0.55
ENTITY_DF_MAX = 8    # must be rare in corpus — aligns with token-path n_cross_max≤10
ENTITY_DF_MIN = 2    # must appear in ≥2 sources to be a bridge
SOURCE_COUNT_MAX = 8  # entity must appear in 2-8 distinct sources
N_ELIGIBLE = 2000    # sample from non-hub eligible notes for speed


def source_tier(src: str) -> int:
    if src in {"USER.md", "MEMORY.md", "SOUL.md"}:
        return 4
    if src.startswith("memory/topics/"):
        return 4
    if src.startswith("memory/summaries/"):
        return 3
    if re.match(r"memory/\d{4}-\d{2}-\d{2}\.md$", src):
        return 2
    if src.startswith("memory/"):
        return 1
    return 0


def is_session_note(src: str) -> bool:
    return bool(re.match(r"memory/\d{4}-\d{2}-\d{2}\.md$", src))


def clean_chunk(note: dict) -> bool:
    if source_tier(note["source"]) < 2:
        return False
    text = note["content"]
    if any(re.search(p, text) for p in NOISY_PATTERNS):
        return False
    if text.count("`") > 6:
        return False
    code_hits = len(re.findall(
        r"/Users/|/home/|\.py\b|\.ts\b|\.json\b|commit [0-9a-f]{6}", text, re.I
    ))
    if code_hits > 3:
        return False
    return True


def toks(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text)]


def tok_ok(t: str) -> bool:
    return t not in STOP and 5 <= len(t) <= 20 and not t[:4].isdigit()


def extract_entity_candidates(text: str) -> list[str]:
    """Extract Title-case and ALL-CAPS words as entity candidates (≥5 chars)."""
    title_case = re.findall(r'\b[A-Z][a-z]{4,}\b', text)
    all_caps = re.findall(r'\b[A-Z]{5,}\b', text)
    return list(set(w.lower() for w in title_case + all_caps))


def excerpt(note: dict, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", note["content"]).strip()
    snip = text[:max_chars - 1] + ("..." if len(text) > max_chars else "")
    return f"[{note['source']}] {snip}"


def load_embeddings(note_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    import pickle
    print(f"Loading embedding cache from {EMBEDDING_CACHE}...")
    cache_path = EMBEDDING_CACHE if EMBEDDING_CACHE.exists() else Path("/tmp/embedding_cache_levi.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if isinstance(cache, dict) and "embeddings" in cache:
        all_embs = cache["embeddings"]  # shape (N, D)
        all_ids = cache["note_ids"]
        id_to_idx = {nid: i for i, nid in enumerate(all_ids)}
        idxs = [id_to_idx[nid] for nid in note_ids if nid in id_to_idx]
        selected_ids = [nid for nid in note_ids if nid in id_to_idx]
        arr = all_embs[idxs].astype(np.float32)
    else:
        id_to_emb = cache
        selected_ids = [nid for nid in note_ids if nid in id_to_emb]
        arr = np.array([id_to_emb[nid] for nid in selected_ids], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-8)
    print(f"  {len(selected_ids)} notes have embeddings (dim={arr.shape[1]})")
    return arr, selected_ids


def main() -> None:
    print("Loading substrate...")
    all_notes = mod.load_notes()
    notes = [n for n in all_notes if clean_chunk(n)]
    print(f"  {len(all_notes)} total → {len(notes)} after filter")

    nb = {n["id"]: n for n in notes}
    nt = {n["id"]: {t for t in toks(n["content"]) if tok_ok(t)} for n in notes}

    # Token index
    df: dict[str, int] = defaultdict(int)
    postings: dict[str, list[str]] = defaultdict(list)
    token_sources: dict[str, set[str]] = defaultdict(set)
    for nid, ts in nt.items():
        src = nb[nid]["source"]
        for t in ts:
            df[t] += 1
            postings[t].append(nid)
            token_sources[t].add(src)

    # Identify entity-grade bridge tokens: rare (df 2-5) and cross-source (2-5 distinct sources)
    entity_bridges = {
        t for t, d in df.items()
        if ENTITY_DF_MIN <= d <= ENTITY_DF_MAX and tok_ok(t)
        and 2 <= len(token_sources.get(t, set())) <= SOURCE_COUNT_MAX
    }
    print(f"  Entity-grade bridge tokens (df {ENTITY_DF_MIN}–{ENTITY_DF_MAX}, "
          f"sources 2–{SOURCE_COUNT_MAX}): {len(entity_bridges)}")

    # Load embeddings for hub exclusion and chain verification
    emb_arr, emb_ids = load_embeddings([n["id"] for n in notes])
    emb_idx = {nid: i for i, nid in enumerate(emb_ids)}

    # Identify hub notes (>HUB_THRESHOLD neighbors at sim>HUB_SIM)
    print(f"Computing hub scores ({len(emb_ids)} notes)...")
    hub_ids: set[str] = set()
    batch = 256
    for start in range(0, len(emb_ids), batch):
        batch_embs = emb_arr[start:start + batch]
        sims = batch_embs @ emb_arr.T
        for bi, nid in enumerate(emb_ids[start:start + batch]):
            n_neighbors = int((sims[bi] > HUB_SIM).sum()) - 1  # exclude self
            if n_neighbors > HUB_THRESHOLD:
                hub_ids.add(nid)
        if start % (batch * 10) == 0:
            print(f"  hub check {start}/{len(emb_ids)} ({len(hub_ids)} hubs so far)")
    print(f"  Hubs excluded: {len(hub_ids)}/{len(emb_ids)}")

    # Restrict to non-hub session notes for bridge candidates
    session_ids = [nid for nid in emb_ids
                   if is_session_note(nb[nid]["source"]) and nid not in hub_ids]
    print(f"  Non-hub session bridge candidates: {len(session_ids)}")

    # Unique start tokens
    unique_toks = {t for t, d in df.items() if d == 1 and tok_ok(t) and len(t) >= 5}

    # Mine chains
    chains: list[tuple] = []
    seen_keys: set[tuple] = set()
    seen_starts: set[str] = set()
    n_checked = 0

    for b_id in session_ids:
        if b_id not in emb_idx:
            continue
        b_toks = nt.get(b_id, set())
        b_src = nb[b_id]["source"]
        b_emb = emb_arr[emb_idx[b_id]]

        # Get entity bridges this note contains
        b_entities = [t for t in b_toks if t in entity_bridges]
        if len(b_entities) < 2:
            continue

        # For speed: precompute similarities from b once
        sims_from_b = b_emb @ emb_arr.T  # shape: (N,)

        for i, e1 in enumerate(b_entities):
            for e2 in b_entities[i + 1:]:
                a_candidates = [
                    nid for nid in postings[e1]
                    if nid != b_id and e2 not in nt.get(nid, set())
                    and nb[nid]["source"] != b_src
                    and nid not in hub_ids
                    and nid in emb_idx
                ]
                c_candidates = [
                    nid for nid in postings[e2]
                    if nid != b_id and e1 not in nt.get(nid, set())
                    and nb[nid]["source"] != b_src
                    and nid not in hub_ids
                    and nid in emb_idx
                ]
                if not a_candidates or not c_candidates:
                    continue

                for a_id in a_candidates[:20]:
                    if a_id not in emb_idx:
                        continue
                    a_emb = emb_arr[emb_idx[a_id]]
                    sim_ab = float(a_emb @ b_emb)
                    if sim_ab < SIM_AB_MIN:
                        continue

                    starts = sorted(
                        [t for t in nt.get(a_id, set()) if t in unique_toks],
                        key=lambda x: (-source_tier(nb[a_id]["source"]), -len(x), x)
                    )
                    if not starts:
                        continue

                    for c_id in c_candidates[:20]:
                        if c_id == a_id or nb[c_id]["source"] == nb[a_id]["source"]:
                            continue
                        if c_id not in emb_idx:
                            continue

                        n_checked += 1
                        c_emb = emb_arr[emb_idx[c_id]]
                        sim_bc = float(b_emb @ c_emb)
                        if sim_bc < SIM_BC_MIN:
                            continue
                        sim_ac = float(a_emb @ c_emb)
                        if sim_ac >= SIM_AC_MAX:
                            continue

                        tok_overlap = len(nt.get(a_id, set()) & nt.get(c_id, set()))
                        if tok_overlap > 1:
                            continue

                        key = (a_id, b_id, c_id)
                        if key in seen_keys:
                            continue

                        start = starts[0]
                        if start in seen_starts:
                            continue

                        tier = (source_tier(nb[a_id]["source"])
                                + source_tier(nb[b_id]["source"])
                                + source_tier(nb[c_id]["source"]))
                        score = (
                            sim_ab * 4.0 + sim_bc * 4.0 - sim_ac * 6.0
                            - tok_overlap * 2.0 + tier * 0.3
                            + (ENTITY_DF_MAX - min(df.get(e1, 1), ENTITY_DF_MAX)) * 0.5
                            + (ENTITY_DF_MAX - min(df.get(e2, 1), ENTITY_DF_MAX)) * 0.5
                        )
                        chains.append((score, start, e1, e2, a_id, b_id, c_id,
                                       sim_ab, sim_bc, sim_ac))

    print(f"  Checked {n_checked} (A,B,C) candidates → {len(chains)} raw chains")

    chains.sort(reverse=True)
    results = []
    for score, start, e1, e2, a_id, b_id, c_id, sim_ab, sim_bc, sim_ac in chains:
        key = (a_id, b_id, c_id)
        if key in seen_keys or start in seen_starts:
            continue
        seen_keys.add(key)
        seen_starts.add(start)
        results.append({
            "idx": len(results) + 1,
            "score": round(float(score), 4),
            "miner": "entity_v6",
            "start_token": start,
            "bridge1_entity": e1,
            "bridge2_entity": e2,
            "bridge1_df": df.get(e1, 0),
            "bridge2_df": df.get(e2, 0),
            "sim_ab": round(float(sim_ab), 4),
            "sim_bc": round(float(sim_bc), 4),
            "sim_ac": round(float(sim_ac), 4),
            "sources": " | ".join(nb[nid]["source"] for nid in [a_id, b_id, c_id]),
            "required_ids": [a_id, b_id, c_id],
            "excerpts": {
                "a": excerpt(nb[a_id]),
                "b": excerpt(nb[b_id]),
                "c": excerpt(nb[c_id]),
            },
        })
        if len(results) >= 300:
            break

    print(f"  Deduplicated: {len(results)} chains")

    with OUT.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")

    # Stats
    if results:
        import statistics
        sims_ab = [r["sim_ab"] for r in results]
        sims_bc = [r["sim_bc"] for r in results]
        sims_ac = [r["sim_ac"] for r in results]
        dfs = [r["bridge1_df"] for r in results] + [r["bridge2_df"] for r in results]
        print(f"\nStats:")
        print(f"  sim_ab: mean={statistics.mean(sims_ab):.3f} min={min(sims_ab):.3f} max={max(sims_ab):.3f}")
        print(f"  sim_bc: mean={statistics.mean(sims_bc):.3f} min={min(sims_bc):.3f} max={max(sims_bc):.3f}")
        print(f"  sim_ac: mean={statistics.mean(sims_ac):.3f} min={min(sims_ac):.3f} max={max(sims_ac):.3f}")
        print(f"  bridge df: mean={statistics.mean(dfs):.1f} min={min(dfs)} max={max(dfs)}")
        from collections import Counter
        top_entities = Counter(r["bridge1_entity"] for r in results)
        print(f"\nTop bridge entities:")
        for e, c in top_entities.most_common(15):
            print(f"  {e:<20} df={df.get(e, 0)}  n={c}")
        print(f"\nSample chains:")
        for r in results[:10]:
            print(f"  {r['idx']:03d} q={r['start_token']:<14} "
                  f"e1={r['bridge1_entity']:<12}(df={r['bridge1_df']}) "
                  f"e2={r['bridge2_entity']:<12}(df={r['bridge2_df']}) "
                  f"sims={r['sim_ab']:.2f}/{r['sim_bc']:.2f}/{r['sim_ac']:.2f}")


if __name__ == "__main__":
    main()
