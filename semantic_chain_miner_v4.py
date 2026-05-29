#!/usr/bin/env python3
"""
Semantic Chain Miner v4

Fixes two bugs in v3:
  1. Hub collapse — v3 routed ~100% of chains through USER.md because it
     computed similarities in a 7,623-note eligible subset. In the full corpus
     USER.md has 742+ neighbors above sim>0.45; branch_k<742 misses the chain.
  2. Subset/full-corpus rank divergence — ranks change when non-eligible notes
     are included. A note at rank #2 in the 7k subset can be rank #709 in full.

Fixes:
  - Full-corpus embedding matrix for all sim and rank computations.
  - Hub exclusion: bridges with >HUB_MAX full-corpus neighbors at sim>SIM_HUB
    are disqualified.
  - Rank gates: B must rank ≤ RANK_B_MAX from A in full corpus; C must rank
    ≤ RANK_C_MAX from B. Chains outside these gates are unretrievable.
"""
from __future__ import annotations
import json, pickle, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/Users/edward/.ocplatform/workspace/research/rfm")
CACHE_FILE = ROOT / "embedding_cache_levi.pkl"
OUT = ROOT / "levi_semantic_chain_candidates_v4.jsonl"

SIM_AB_MIN = 0.50
SIM_BC_MIN = 0.50
SIM_AC_MAX = 0.42
TOK_AC_MAX = 1
HUB_MAX    = 40
SIM_HUB    = 0.50
RANK_B_MAX = 20
RANK_C_MAX = 20
TARGET     = 200

STOP = set("""
able accepted active actual additional adjacent basic better clean clear common correct
current different direct enough exact external final first full general good great hard
high human important initial known large latest likely live local long main major
meaningful minimal native new next obvious old ongoing only operational other personal
possible primary prior raw real recent related relevant same second separate simple
small specific stable strong sure technical top true useful weak whole system memory
file data user note output text block token result type name status created updated
source boundary tags evidence confidence tier person organization entity service
endpoint project work time week day month year back going make makes made used using
been have will would could should also just like much very some more most than from
this that with they them their about into over under when where which while
""".split())

def toks(text):
    return [w.lower() for w in re.findall(r'[a-z]{4,}', text.lower())]

def tok_ok(t):
    return t not in STOP and 5 <= len(t) <= 20 and not t[:4].isdigit()

def note_toks(content):
    return {t for t in toks(content) if tok_ok(t)}

def source_tier(src):
    if src in {"USER.md", "MEMORY.md", "SOUL.md"}: return 4
    if src.startswith("memory/topics/"): return 4
    if src.startswith("memory/summaries/"): return 3
    if re.match(r"memory/\d{4}-\d{2}-\d{2}.*\.md$", src): return 2
    if src.startswith("memory/"): return 1
    return 0

def clean(note):
    t = note["content"]
    if source_tier(note["source"]) < 2: return False
    if t.count("`") > 8: return False
    if len(re.findall(r"/Users/|/home/|\.py\b|\.ts\b|\.json\b|commit [0-9a-f]{6}", t, re.I)) > 3: return False
    for pat in [r"<<<EXTERNAL_UNTRUSTED_CONTENT", r"tool call:", r"Traceback \(", r"\[Audio\]", r"<media:"]:
        if re.search(pat, t): return False
    return len(toks(t)) >= 10

def excerpt(note, max_chars=240):
    text = re.sub(r"\s+", " ", note["content"]).strip()
    return f"[{note['source']}] {text[:max_chars-1]}{'...' if len(text) > max_chars else ''}"

def main():
    print("Loading substrate...")
    sys.path.insert(0, "/tmp")
    import rfm_substrate_path_test as substrate
    notes_all = substrate.load_notes()
    nb = {n["id"]: n for n in notes_all}
    print(f"  {len(notes_all)} notes")

    print("Loading embeddings...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    emb_ids = cache["note_ids"]
    embs = cache["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-9)
    emb_set = set(emb_ids)
    all_pos = {nid: i for i, nid in enumerate(emb_ids)}
    print(f"  {len(emb_ids)} embeddings, dim={embs.shape[1]}")

    eligible = [n for n in notes_all if n["id"] in emb_set and clean(n)]
    eligible_set = {n["id"] for n in eligible}
    print(f"  eligible (tier≥2, clean, in emb cache): {len(eligible)}")

    nt = {n["id"]: note_toks(n["content"]) for n in eligible}
    df: dict[str, int] = defaultdict(int)
    for ts in nt.values():
        for t in ts: df[t] += 1

    elig_ids  = [n["id"] for n in eligible]
    elig_idx  = np.array([all_pos[nid] for nid in elig_ids])
    elig_embs = embs[elig_idx]

    print(f"\nComputing {len(elig_ids)}×{len(emb_ids)} sim matrix...")
    sim_elig_all = elig_embs @ embs.T  # shape: (n_elig, N_all)
    print("  done.")

    elig_to_row = {nid: i for i, nid in enumerate(elig_ids)}

    # Hub scores
    hub_counts = (sim_elig_all >= SIM_HUB).sum(axis=1)
    hub_scores = {nid: int(hub_counts[i]) for i, nid in enumerate(elig_ids)}
    n_hubs = sum(1 for v in hub_scores.values() if v > HUB_MAX)
    print(f"  hub nodes (>{HUB_MAX} neighbors@{SIM_HUB}): {n_hubs}/{len(eligible)}")
    non_hub = {nid for nid, v in hub_scores.items() if v <= HUB_MAX}
    print(f"  non-hub eligible bridge candidates: {len(non_hub)}")

    chains = []
    seen_ac: set[tuple] = set()
    seen_starts: set[str] = set()

    print(f"\nMining from {len(non_hub)} non-hub bridge candidates...")
    processed = 0
    for b_id in elig_ids:
        if b_id not in non_hub: continue
        processed += 1
        if processed % 1000 == 0:
            print(f"  {processed} bridges, {len(chains)} raw chains so far")

        b_src  = nb[b_id]["source"]
        b_tier = source_tier(b_src)
        if b_tier < 2: continue

        row = sim_elig_all[elig_to_row[b_id]]  # shape: (N_all,)

        # Top-(RANK_B_MAX+RANK_C_MAX+10) neighbors in full corpus
        k = RANK_B_MAX + RANK_C_MAX + 10
        top_local = np.argpartition(row, -k)[-k:]
        top_sorted = top_local[np.argsort(row[top_local])[::-1]]

        a_cands, c_cands = [], []
        rank = 0
        for local_i in top_sorted:
            nid = emb_ids[local_i]
            sim = float(row[local_i])
            if sim < SIM_AB_MIN: break
            rank += 1
            if nid == b_id or nid not in eligible_set: continue
            if nb[nid]["source"] == b_src: continue
            if source_tier(nb[nid]["source"]) < 2: continue
            if len(a_cands) < RANK_B_MAX:
                a_cands.append((rank, nid, sim))
            if len(c_cands) < RANK_C_MAX:
                c_cands.append((rank, nid, sim))

        if not a_cands or not c_cands: continue

        for rank_a, a_id, sim_ab in a_cands:
            a_src  = nb[a_id]["source"]
            a_toks = nt.get(a_id, set())
            anchors = [t for t in a_toks if df.get(t, 0) <= 4 and len(t) >= 5]
            if not anchors: continue
            anchor = min(anchors, key=lambda t: (df.get(t, 0), -len(t)))

            a_pos = all_pos[a_id]
            sims_a = embs @ embs[a_pos]

            for rank_c, c_id, sim_bc in c_cands:
                if c_id == a_id: continue
                c_src = nb[c_id]["source"]
                if c_src == a_src: continue
                if source_tier(c_src) < 2: continue

                c_pos = all_pos[c_id]
                sim_ac = float(sims_a[c_pos])
                if sim_ac >= SIM_AC_MAX: continue

                tok_ov = len(nt.get(a_id, set()) & nt.get(c_id, set()))
                if tok_ov > TOK_AC_MAX: continue

                ac_key = (a_id, c_id)
                if ac_key in seen_ac or anchor in seen_starts: continue
                seen_ac.add(ac_key)
                seen_starts.add(anchor)

                score = (
                    sim_ab * 3.0 + sim_bc * 3.0
                    - sim_ac * 5.0 - tok_ov * 2.0
                    + (source_tier(a_src) + b_tier + source_tier(c_src)) * 0.2
                    - (rank_a + rank_c) * 0.05
                    + 1.0 / max(df.get(anchor, 1), 1)
                )
                chains.append((score, anchor, sim_ab, sim_bc, sim_ac,
                               rank_a, rank_c, a_id, b_id, c_id))
                if len(chains) >= TARGET * 30: break
            if len(chains) >= TARGET * 30: break
        if len(chains) >= TARGET * 30: break

    print(f"Raw chain candidates: {len(chains)}")
    chains.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen_ac2: set[tuple] = set()
    seen_s2:  set[str]   = set()
    for score, anchor, sim_ab, sim_bc, sim_ac, rank_a, rank_c, a_id, b_id, c_id in chains:
        if len(results) >= TARGET: break
        ac_key = (a_id, c_id)
        if ac_key in seen_ac2 or anchor in seen_s2: continue
        seen_ac2.add(ac_key); seen_s2.add(anchor)
        results.append({
            "idx": len(results) + 1,
            "score": round(float(score), 4),
            "miner": "semantic_v4",
            "start_token": anchor,
            "sim_ab": round(float(sim_ab), 4),
            "sim_bc": round(float(sim_bc), 4),
            "sim_ac": round(float(sim_ac), 4),
            "rank_b_from_a": int(rank_a),
            "rank_c_from_b": int(rank_c),
            "sources": " | ".join(nb[x]["source"] for x in [a_id, b_id, c_id]),
            "required_ids": [a_id, b_id, c_id],
            "excerpts": {"a": excerpt(nb[a_id]), "b": excerpt(nb[b_id]), "c": excerpt(nb[c_id])},
        })

    print(f"\nMined {len(results)} rank-verified chains (target {TARGET})")
    with OUT.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")

    if results:
        sab = [r["sim_ab"] for r in results]; sbc = [r["sim_bc"] for r in results]
        sac = [r["sim_ac"] for r in results]; rb  = [r["rank_b_from_a"] for r in results]
        rc  = [r["rank_c_from_b"] for r in results]
        print(f"\nsim_ab mean={np.mean(sab):.3f} min={np.min(sab):.3f}")
        print(f"sim_bc mean={np.mean(sbc):.3f} min={np.min(sbc):.3f}")
        print(f"sim_ac mean={np.mean(sac):.3f} max={np.max(sac):.3f}")
        print(f"rank_B mean={np.mean(rb):.1f}  max={np.max(rb)}")
        print(f"rank_C mean={np.mean(rc):.1f}  max={np.max(rc)}")
        from collections import Counter
        bsrcs = Counter(r["sources"].split(" | ")[1] for r in results)
        print("\nTop bridge sources:")
        for s, c in bsrcs.most_common(8): print(f"  {c:3d}  {s}")
        print("\nFirst 10:")
        for r in results[:10]:
            print(f"  {r['idx']:03d} {r['start_token']:<18} rB={r['rank_b_from_a']:2d} rC={r['rank_c_from_b']:2d}"
                  f" sAB={r['sim_ab']:.3f} sBC={r['sim_bc']:.3f} | {r['sources'][:70]}")
        print(f"\n→ Use branch_k ≥ {max(max(rb), max(rc)) + 5} in benchmark")

if __name__ == "__main__":
    main()
