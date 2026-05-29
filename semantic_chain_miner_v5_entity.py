#!/usr/bin/env python3
"""
Semantic Chain Miner v5 — Entity-Anchored Session Chains

Mines session→session chains where the bridge token is a named entity
(proper noun: person, project, company, product) rather than a generic
concept word or temporal co-occurrence signal.

Why this works:
  - Named entities appear as rare tokens by definition (specific names)
  - Token-path retrieval fires on rare tokens → entity-bridged chains are
    token-retrievable, unlike generic embedding similarity chains
  - The chains are semantically meaningful by construction (A and B connect
    through a specific named entity, not just temporal proximity)
  - Creates a "hybrid chain family": testable by BOTH token-path AND
    embedding-bridge, letting us compare directly on the same chain set

Design:
  1. Extract named entity candidates from session notes:
     - Capitalized words (4+ chars, not at sentence start) appearing in 2-6
       session sources — rare enough to be bridging, common enough to connect
  2. For each entity E appearing in sources S_A and S_B:
     - A = note from S_A containing E (with a distinctive anchor token)
     - B = note from S_B containing E (the bridge, different source)
     - C = note from a third source S_C where B's entity connects via a
       second entity E2 (same chain construction as v4 but both hops are
       entity-anchored)
  3. Verify A and C have low direct embedding similarity (semantic gap exists)
  4. Verify A has a BM25-findable anchor token (unique or rare)

This creates chains testable by both:
  - Token-path: finds chains via rare entity token co-occurrence
  - Embedding-bridge: finds chains via semantic similarity gaps
"""
from __future__ import annotations

import json, pickle, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('/Users/edward/.ocplatform/workspace/research/rfm')
CACHE_FILE = ROOT / 'embedding_cache_levi.pkl'
OUT = ROOT / 'levi_semantic_chain_candidates_v5.jsonl'

# Thresholds
ENTITY_MIN_SOURCES = 2   # entity must appear in at least this many session sources
ENTITY_MAX_SOURCES = 8   # but not too many (otherwise it's a stopword-like term)
ENTITY_MIN_LEN = 4
SIM_AC_MAX = 0.42         # A and C must be semantically distant
TOK_AC_MAX = 1            # A and C share at most 1 token
ANCHOR_DF_MAX = 4         # anchor token for A must be distinctive
TARGET = 200

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

def toks_lower(text): return [w.lower() for w in re.findall(r'[a-z]{4,}', text.lower())]
def tok_ok(t): return t not in STOP and 5<=len(t)<=20 and not t[:4].isdigit()
def is_session(src): return bool(re.match(r'memory/\d{4}-\d{2}', src))

def excerpt(note, max_chars=240):
    text = re.sub(r'\s+', ' ', note['content']).strip()
    return f"[{note['source']}] {text[:max_chars-1]}{'...' if len(text)>max_chars else ''}"

def main():
    sys.path.insert(0, '/tmp')
    import rfm_substrate_path_test as substrate
    notes_all = substrate.load_notes()
    nb = {n['id']: n for n in notes_all}
    print(f'corpus: {len(notes_all)} notes')

    # Load embeddings
    print('loading embeddings...')
    with open(str(CACHE_FILE),'rb') as f: cache = pickle.load(f)
    emb_ids = cache['note_ids']
    embs = cache['embeddings'].astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= np.maximum(norms, 1e-9)
    id_to_pos = {nid: i for i, nid in enumerate(emb_ids)}
    emb_set = set(emb_ids)
    print(f'embeddings: {len(emb_ids)}')

    # Build token index for anchor token selection
    nt = {n['id']: {t for t in toks_lower(n['content']) if tok_ok(t)} for n in notes_all}
    df: dict[str, int] = defaultdict(int)
    postings: dict[str, list] = defaultdict(list)
    for nid, ts in nt.items():
        for t in ts: df[t]+=1; postings[t].append(nid)

    # ── Step 1: Extract named entity candidates from session notes ──────────
    # Named entity = capitalized word not at sentence start, 4+ chars,
    # appearing in 2-8 session source files
    session_notes = [n for n in notes_all if is_session(n['source'])]
    print(f'session notes: {len(session_notes)}')

    # For each session note, extract capitalized words (proper noun candidates)
    # that aren't at the very start of a sentence
    entity_to_sources: dict[str, set] = defaultdict(set)
    entity_to_notes: dict[str, list] = defaultdict(list)  # entity → [(note_id, source)]

    for n in session_notes:
        text = n['content']
        # Find capitalized words that appear mid-sentence (not after ., ?, !, \n)
        # Extract: word is capitalized, 4+ chars, not a stopword
        words = re.findall(r'(?<![.?!\n])\s([A-Z][a-z]{3,})', text)
        # Also get ALL-CAPS abbreviations/names 4+ chars
        words += re.findall(r'\b[A-Z]{4,}\b', text)
        for w in set(words):
            wl = w.lower()
            if wl in STOP: continue
            if len(wl) < ENTITY_MIN_LEN: continue
            # Filter out common sentence-start words even if mid-text
            if wl in {'this','that','what','when','where','which','there','these',
                      'those','with','from','they','them','their','have','will',
                      'would','could','should','about','into','over','under'}:
                continue
            entity_to_sources[wl].add(n['source'])
            entity_to_notes[wl].append((n['id'], n['source']))

    # Filter to bridging entities
    bridging = {e: srcs for e, srcs in entity_to_sources.items()
                if ENTITY_MIN_SOURCES <= len(srcs) <= ENTITY_MAX_SOURCES}
    print(f'bridging entity candidates: {len(bridging)}')

    # ── Step 2: Mine A→B→C chains using entity bridges ──────────────────────
    chains = []
    seen_ac: set[tuple] = set()
    seen_starts: set[str] = set()

    for e1, e1_srcs in list(bridging.items()):
        e1_notes = entity_to_notes[e1]
        # Group notes by source
        by_src: dict[str, list] = defaultdict(list)
        for nid, src in e1_notes:
            if is_session(src): by_src[src].append(nid)
        if len(by_src) < 2: continue

        src_list = list(by_src.keys())

        # For each pair of sources connected by e1:
        for i, src_a in enumerate(src_list):
            for src_b in src_list[i+1:]:
                # A comes from src_a, B comes from src_b (or vice versa)
                for a_src, b_src in [(src_a, src_b), (src_b, src_a)]:
                    a_candidates = by_src[a_src]
                    b_candidates = by_src[b_src]

                    for a_id in a_candidates:
                        if a_id not in emb_set: continue
                        # Need a distinctive anchor token for A
                        a_toks = nt.get(a_id, set())
                        anchor_cands = [t for t in a_toks if df.get(t,0) <= ANCHOR_DF_MAX and len(t)>=5]
                        if not anchor_cands: continue
                        anchor = min(anchor_cands, key=lambda t: (df.get(t,0), -len(t)))

                        for b_id in b_candidates:
                            if b_id == a_id or b_id not in emb_set: continue

                            # Now find a second entity e2 in B that connects to a third source
                            b_toks_raw = re.findall(r'(?<![.?!\n])\s([A-Z][a-z]{3,})', nb[b_id]['content'])
                            b_entities = {w.lower() for w in b_toks_raw
                                          if w.lower() not in STOP
                                          and len(w) >= ENTITY_MIN_LEN
                                          and w.lower() in bridging
                                          and w.lower() != e1}

                            for e2 in b_entities:
                                e2_notes = entity_to_notes[e2]
                                e2_srcs = entity_to_sources[e2]
                                # C must come from a source different from both a_src and b_src
                                c_srcs = {src for src in e2_srcs
                                          if src not in (a_src, b_src) and is_session(src)}
                                if not c_srcs: continue

                                for c_nid, c_src in entity_to_notes[e2]:
                                    if c_src not in c_srcs: continue
                                    if c_nid == b_id or c_nid not in emb_set: continue

                                    # Verify embedding gap: A and C must not be directly similar
                                    a_pos = id_to_pos[a_id]
                                    c_pos = id_to_pos[c_nid]
                                    sims_a = embs @ embs[a_pos]
                                    sim_ac = float(sims_a[c_pos])
                                    if sim_ac >= SIM_AC_MAX: continue

                                    # Lexical isolation
                                    tok_ov = len(nt.get(a_id,set()) & nt.get(c_nid,set()))
                                    if tok_ov > TOK_AC_MAX: continue

                                    ac_key = (a_id, c_nid)
                                    if ac_key in seen_ac or anchor in seen_starts: continue
                                    seen_ac.add(ac_key)
                                    seen_starts.add(anchor)

                                    # Score: entity rarity × anchor rarity × sim gap
                                    score = (
                                        (ENTITY_MAX_SOURCES - len(entity_to_sources[e1])) * 0.5
                                        + (ENTITY_MAX_SOURCES - len(entity_to_sources[e2])) * 0.5
                                        + 1.0 / max(df.get(anchor, 1), 1)
                                        + (SIM_AC_MAX - sim_ac) * 2.0  # reward larger gaps
                                        - tok_ov
                                    )
                                    chains.append((score, anchor, e1, e2, sim_ac,
                                                  a_id, b_id, c_nid))
                                    if len(chains) >= TARGET * 30: break
                                if len(chains) >= TARGET * 30: break
                            if len(chains) >= TARGET * 30: break
                        if len(chains) >= TARGET * 30: break
                    if len(chains) >= TARGET * 30: break

    print(f'raw chain candidates: {len(chains)}')
    chains.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen_ac2: set[tuple] = set()
    seen_s2:  set[str]   = set()
    for score, anchor, e1, e2, sim_ac, a_id, b_id, c_id in chains:
        if len(results) >= TARGET: break
        ac_key = (a_id, c_id)
        if ac_key in seen_ac2 or anchor in seen_s2: continue
        seen_ac2.add(ac_key); seen_s2.add(anchor)
        results.append({
            'idx': len(results)+1,
            'score': round(float(score),4),
            'miner': 'entity_v5',
            'start_token': anchor,
            'bridge1_entity': e1,
            'bridge2_entity': e2,
            'sim_ac': round(float(sim_ac),4),
            'sources': ' | '.join(nb[x]['source'] for x in [a_id, b_id, c_id]),
            'required_ids': [a_id, b_id, c_id],
            'excerpts': {'a': excerpt(nb[a_id]),'b': excerpt(nb[b_id]),'c': excerpt(nb[c_id])},
        })

    print(f'\nMined {len(results)} entity-anchored chains (target {TARGET})')
    with OUT.open('w') as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print(f'Wrote {OUT}')

    if results:
        from collections import Counter
        e1s = Counter(r['bridge1_entity'] for r in results)
        e2s = Counter(r['bridge2_entity'] for r in results)
        print(f'\nTop bridge1 entities: {e1s.most_common(8)}')
        print(f'Top bridge2 entities: {e2s.most_common(8)}')
        print('\nFirst 10:')
        for r in results[:10]:
            print(f"  {r['idx']:03d} anchor={r['start_token']:<16} "
                  f"e1={r['bridge1_entity']:<14} e2={r['bridge2_entity']:<14} "
                  f"sim_ac={r['sim_ac']:.3f} | {r['sources'][:60]}")

if __name__ == '__main__':
    main()
