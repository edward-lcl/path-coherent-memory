#!/usr/bin/env python3
"""
Held-out calibration set miner.

Mines ~100 fresh token-topology chains from the Levi substrate for use as
a calibration set to tune bridge_min_len and BRIDGE_DENY parameters without
contaminating the frozen v2 Talos 200-chain benchmark.

Design constraints:
  - Uses the same mining algorithm as mine_candidates_v2.py (token-topology)
  - Loads the Levi substrate (not Talos) to keep corpora isolated
  - Excludes any chain whose (A_id, B_id, C_id) triple appears in the existing
    Levi v2 candidate set (levi_semantic_chain_candidates_v2.jsonl)
  - Targets 120 chains so post-judge yield reaches ~100 real+weak

Output: levi_calibration_candidates_v1.jsonl (same schema as v2 candidates)

Workflow after mining:
  1. Run oMLX judge on the 120 candidates (same prompt as v2)
  2. Freeze judged results as levi_calibration_judged_v1.jsonl
  3. Use this set to tune token-gated retriever parameters:
       - bridge_min_len (currently 7)
       - BRIDGE_DENY list
       - n_cross_max (currently 4)
  4. Target: close the 12pp gap between token-gated (60.6%) and v12 (72.7%)
     on real_semantic chains without touching the frozen Talos v2 benchmark.
"""
from __future__ import annotations

import json, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_CANDIDATES = ROOT / "levi_calibration_candidates_v1.jsonl"
# Exclude these to avoid contaminating the calibration set
EXISTING_V2 = ROOT / "levi_semantic_chain_candidates_v2.jsonl"

TARGET = 120

GENERIC = set("""
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
organization entity
""".split())

NOISY_PATTERNS = [
    r"<<<EXTERNAL_UNTRUSTED_CONTENT",
    r"Discord Guild",
    r"tool call:",
    r"Traceback \(",
    r"\[Audio\]",
    r"<media:",
    r"^\s*```",
    r"^[0-9a-f]{8,40}$",
]

def source_tier(src: str) -> int:
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

def clean_chunk(note: dict) -> bool:
    text = note["content"]
    if source_tier(note["source"]) < 2:
        return False
    if any(re.search(p, text) for p in NOISY_PATTERNS):
        return False
    if text.count("`") > 6:
        return False
    code_hits = len(re.findall(
        r"/Users/|/home/|\.py\b|\.ts\b|\.json\b|commit [0-9a-f]{6}", text, re.I))
    if code_hits > 3:
        return False
    return True

def toks(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'[a-z]{4,}', text.lower())]

def token_ok(t: str) -> bool:
    if t in GENERIC:
        return False
    if len(t) < 5 or len(t) > 18:
        return False
    if t[:4].isdigit():
        return False
    if re.search(r"(ing|tion|ment|ness|less|ful|ible|ous|ive|ary|ory|ize|ise|ify)$", t) and len(t) > 9:
        return False
    if re.search(r"(command|service|warning|endpoint|backend|frontend|buildout|start|finish|"
                 r"function|artifact|parameter|argument|variable|config|setting|option|feature)$", t):
        return False
    return True

def note_tokens(notes: list[dict]) -> dict[str, set[str]]:
    base = {}
    for n in notes:
        base[n["id"]] = {t for t in toks(n["content"]) if token_ok(t)}
    return base

def excerpt(note: dict, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", note["content"]).strip()
    snip = text[:max_chars - 1] + ("..." if len(text) > max_chars else "")
    return f"[{note['source']}] {snip}"

def load_existing_triples() -> set[tuple]:
    """Load existing v2 candidate triples to avoid dedup collisions."""
    if not EXISTING_V2.exists():
        return set()
    triples = set()
    with EXISTING_V2.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                req = d.get("required_ids", [])
                if len(req) == 3:
                    triples.add(tuple(req))
            except Exception:
                continue
    print(f"Loaded {len(triples)} existing v2 triples to exclude")
    return triples

def mine(notes: list[dict], existing_triples: set[tuple],
         limit: int = TARGET) -> list[dict]:
    nb = {n["id"]: n for n in notes}
    nt = note_tokens(notes)

    # Build token index
    df: dict[str, int] = defaultdict(int)
    postings: dict[str, list[str]] = defaultdict(list)
    for nid, toks_set in nt.items():
        for t in toks_set:
            df[t] += 1
            postings[t].append(nid)

    rare = {t for t, d in df.items() if 2 <= d <= 5 and token_ok(t)}
    unique = {t for t, d in df.items() if d == 1 and token_ok(t) and len(t) >= 5}

    chains: list[tuple] = []
    seen_keys: set[tuple] = set()
    seen_starts: set[str] = set()

    for b_id, b_toks in nt.items():
        b_tier = source_tier(nb[b_id]["source"])
        if b_tier < 2:
            continue
        bridges = [t for t in b_toks if t in rare]
        if len(bridges) < 2:
            continue
        for i, t1 in enumerate(bridges):
            for t2 in bridges[i + 1:]:
                a_candidates = [nid for nid in postings[t1]
                                if nid != b_id and t2 not in nt[nid]]
                c_candidates = [nid for nid in postings[t2]
                                if nid != b_id and t1 not in nt[nid]]
                for a_id in a_candidates:
                    a_tier = source_tier(nb[a_id]["source"])
                    if a_tier < 2:
                        continue
                    starts = sorted(
                        [t for t in nt[a_id] if t in unique],
                        key=lambda x: (-a_tier, -len(x), x)
                    )
                    if not starts:
                        continue
                    for c_id in c_candidates:
                        if c_id == a_id:
                            continue
                        c_tier = source_tier(nb[c_id]["source"])
                        if c_tier < 2:
                            continue
                        if nb[a_id]["source"] == nb[c_id]["source"]:
                            continue
                        sources = {nb[a_id]["source"], nb[b_id]["source"], nb[c_id]["source"]}
                        if len(sources) < 2:
                            continue
                        direct_overlap = len(nt[a_id] & nt[c_id])
                        if direct_overlap > 1:
                            continue
                        triple = (a_id, b_id, c_id)
                        if triple in existing_triples:
                            continue  # exclude already-used chains
                        tier_score = a_tier + b_tier + c_tier
                        start = starts[0]
                        key = triple
                        if key in seen_keys or start in seen_starts:
                            continue
                        score = (
                            10.0
                            + tier_score * 2.0
                            + (5 - min(df[t1], 5)) + (5 - min(df[t2], 5))
                            + min(len(t1), 12) * 0.1 + min(len(t2), 12) * 0.1
                            - direct_overlap * 4
                        )
                        chains.append((score, start, t1, t2, a_id, b_id, c_id))

    chains.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, start, t1, t2, a_id, b_id, c_id in chains:
        key = (a_id, b_id, c_id)
        if key in seen_keys or start in seen_starts:
            continue
        seen_keys.add(key)
        seen_starts.add(start)
        results.append({
            "idx": len(results) + 1,
            "score": round(score, 3),
            "miner": "token_v2_calibration",
            "start_token": start,
            "bridge1": t1,
            "bridge2": t2,
            "sources": " | ".join(nb[nid]["source"] for nid in [a_id, b_id, c_id]),
            "required_ids": [a_id, b_id, c_id],
            "excerpts": {
                "a": excerpt(nb[a_id]),
                "b": excerpt(nb[b_id]),
                "c": excerpt(nb[c_id]),
            },
        })
        if len(results) >= limit:
            break

    return results


def main() -> None:
    sys.path.insert(0, "/tmp")
    import rfm_substrate_path_test as substrate

    all_notes = substrate.load_notes()
    notes = [n for n in all_notes if clean_chunk(n)]
    print(f"Eligible notes: {len(notes)}")

    existing_triples = load_existing_triples()
    candidates = mine(notes, existing_triples, TARGET)
    print(f"Mined {len(candidates)} calibration candidates (target {TARGET})")

    with OUT_CANDIDATES.open("w", encoding="utf-8") as f:
        for r in candidates:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_CANDIDATES}")

    print("\nSample (first 10):")
    for r in candidates[:10]:
        print(f"  {r['idx']:03d}  {r['start_token']:<18} -> {r['bridge1']} -> {r['bridge2']}"
              f"  sources={r['sources'][:80]}")

    print("\nNext steps:")
    print("  1. Run oMLX judge:  python3 run_omlx_semantic_judge.py --input levi_calibration_candidates_v1.jsonl --output levi_calibration_judged_v1.jsonl")
    print("  2. Review judged output, freeze levi_calibration_judged_v1.jsonl")
    print("  3. Use for parameter tuning: bridge_min_len, BRIDGE_DENY, n_cross_max")
    print("  4. Target: close the 12pp gap (token-gated 60.6% → v12 72.7%)")


if __name__ == "__main__":
    main()
