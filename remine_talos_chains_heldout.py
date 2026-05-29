#!/usr/bin/env python3
"""
Mine 100 held-out Talos token-topology chains for paper-integrity validation.

Purpose: provide a clean held-out set to validate the bridge_min_len=5 fix
without tuning on the frozen v2 200-chain benchmark.

Methodology:
- Mines fresh chains from the Talos substrate
- Explicitly excludes all (A, B, C) triples already in the frozen v2 200
- Outputs 100 candidates for judging and parameter validation
- Results go to levi_calibration_talos_heldout_candidates_v1.jsonl

Run on the Talos M1 machine in the research/rfm directory:
    python3 remine_talos_chains_heldout.py

Then judge with:
    python3 run_omlx_semantic_judge.py \
        --input levi_calibration_talos_heldout_candidates_v1.jsonl \
        --out levi_calibration_talos_heldout_judged_v1.jsonl

Then run the gated benchmark with different bridge_min_len values on the
judged held-out set, pick the best, and re-test on frozen v2 200 to report
the clean improvement.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── substrate loader ─────────────────────────────────────────────────────────
# Talos substrate path — adjust if substrate_ui.py is elsewhere
SUBSTRATE_PATHS = [
    Path("/Users/edward/talos/research/rfm"),
    Path("/Users/edward/.ocplatform/workspace/research/rfm"),
    Path(Path(__file__).resolve().parent),
]

def load_substrate():
    import importlib.util
    for base in SUBSTRATE_PATHS:
        mod_path = base / "rfm_substrate_path_test.py"
        if mod_path.exists():
            spec = importlib.util.spec_from_file_location("rfm_substrate_path_test", mod_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["rfm_substrate_path_test"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "rfm_substrate_path_test.py not found. Run from the rfm directory "
        "or adjust SUBSTRATE_PATHS in this script."
    )

# ── token filters (match production retriever at bridge_min_len=5) ────────────
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
organization entity service endpoint project work time week day month year
back going make makes made used using been have will would could should
also just like much very some more most than from this that with they them
their about into over under when where which while
""".split())

BRIDGE_DENY = set("""
module function variable argument parameter option setting feature
command service endpoint backend frontend buildout artifact return request
response process system output result value error warning message debug
""".split())

NOISY_PATTERNS = [
    r"<<<EXTERNAL_UNTRUSTED_CONTENT",
    r"tool call:",
    r"Traceback \(",
    r"<media:",
    r"^[0-9a-f]{8,40}$",
]


def token_ok(t: str, min_len: int = 5) -> bool:
    if t in GENERIC or t in BRIDGE_DENY:
        return False
    if len(t) < min_len or len(t) > 20:
        return False
    if t[:4].isdigit():
        return False
    if re.search(r"(ing|tion|ment|ness|less|ful|ible|ous|ive|ary|ory|ize|ise|ify)$", t) and len(t) > 9:
        return False
    return True


def clean_chunk(note: dict, source_tier_fn) -> bool:
    if source_tier_fn(note["source"]) < 2:
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


def source_tier_talos(src: str) -> int:
    """Talos substrate source tiers — adjust to match actual Talos source structure."""
    if src.startswith("organizations:"):
        return 4
    if src.startswith("claim:"):
        return 3
    if src.startswith("messages:"):
        return 3
    if src.startswith("events:"):
        return 2
    if src.startswith("reports:"):
        return 2
    return 1


def excerpt(note: dict, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", note["content"]).strip()
    snip = text[:max_chars - 1] + ("..." if len(text) > max_chars else "")
    return f"[{note['source']}] {snip}"


def mine_heldout(
    notes: list[dict],
    excluded_triples: set[tuple],
    limit: int = 100,
    min_bridge_len: int = 5,
    max_n_cross: int = 10,
) -> list[dict]:
    """Mine chains that don't overlap with excluded_triples (the frozen v2 200)."""
    from collections import defaultdict

    nb = {n["id"]: n for n in notes}

    def toks(text):
        return [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text)]

    nt = {n["id"]: {t for t in toks(n["content"]) if token_ok(t, min_bridge_len)} for n in notes}

    df: dict[str, int] = defaultdict(int)
    postings: dict[str, list[str]] = defaultdict(list)
    token_sources: dict[str, set[str]] = defaultdict(set)
    for nid, ts in nt.items():
        src = nb[nid]["source"]
        for t in ts:
            df[t] += 1
            postings[t].append(nid)
            token_sources[t].add(src)

    rare = {t for t, d in df.items() if 2 <= d <= 5 and token_ok(t, min_bridge_len)}
    unique = {t for t, d in df.items() if d == 1 and token_ok(t, 5) and len(t) >= 5}

    chains = []
    seen_keys: set[tuple] = set()
    seen_starts: set[str] = set()

    for b_id, b_toks in nt.items():
        b_tier = source_tier_talos(nb[b_id]["source"])
        if b_tier < 2:
            continue
        bridges = [t for t in b_toks if t in rare]
        if len(bridges) < 2:
            continue
        for i, t1 in enumerate(bridges):
            n_cross1 = len(token_sources.get(t1, set()) - {nb[b_id]["source"]})
            if n_cross1 == 0 or n_cross1 > max_n_cross:
                continue
            for t2 in bridges[i + 1:]:
                n_cross2 = len(token_sources.get(t2, set()) - {nb[b_id]["source"]})
                if n_cross2 == 0 or n_cross2 > max_n_cross:
                    continue
                a_candidates = [n for n in postings[t1] if n != b_id and t2 not in nt[n]]
                c_candidates = [n for n in postings[t2] if n != b_id and t1 not in nt[n]]
                for a_id in a_candidates:
                    a_tier = source_tier_talos(nb[a_id]["source"])
                    if a_tier < 2:
                        continue
                    starts = sorted(
                        [t for t in nt[a_id] if t in unique],
                        key=lambda x: (-a_tier, -len(x), x),
                    )
                    if not starts:
                        continue
                    for c_id in c_candidates:
                        if c_id == a_id:
                            continue
                        c_tier = source_tier_talos(nb[c_id]["source"])
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
                        key = (a_id, b_id, c_id)
                        if key in seen_keys or key in excluded_triples:
                            continue
                        start = starts[0]
                        if start in seen_starts:
                            continue
                        tier_score = a_tier + b_tier + c_tier
                        score = (
                            10.0
                            + tier_score * 2.0
                            + (5 - min(df[t1], 5)) + (5 - min(df[t2], 5))
                            + min(len(t1), 12) * 0.1 + min(len(t2), 12) * 0.1
                            - direct_overlap * 4.0
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
            "score": round(float(score), 3),
            "miner": "heldout_v1",
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
    ROOT = Path(__file__).resolve().parent
    FROZEN_V2 = ROOT / "talos_semantic_chain_candidates_v2.jsonl"
    OUT = ROOT / "levi_calibration_talos_heldout_candidates_v1.jsonl"

    if not FROZEN_V2.exists():
        print(f"ERROR: frozen v2 candidates not found at {FROZEN_V2}")
        print("Run from the rfm directory where talos_semantic_chain_candidates_v2.jsonl lives.")
        sys.exit(1)

    print("Loading frozen v2 chain triples to exclude...")
    frozen = [json.loads(l) for l in FROZEN_V2.read_text().splitlines() if l.strip()]
    excluded = {tuple(c["required_ids"]) for c in frozen}
    print(f"  Excluding {len(excluded)} triples from frozen v2")

    print("Loading substrate...")
    mod = load_substrate()
    all_notes = mod.load_notes()
    notes = [n for n in all_notes if clean_chunk(n, source_tier_talos)]
    print(f"  {len(all_notes)} total notes → {len(notes)} after tier/noise filter")

    print("Mining held-out chains...")
    candidates = mine_heldout(notes, excluded, limit=100)
    print(f"  Mined {len(candidates)} held-out chains")

    with OUT.open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")

    print("\nSample (first 10):")
    for r in candidates[:10]:
        print(f"  {r['idx']:03d} {r['start_token']:<16} {r['bridge1']:<14} {r['bridge2']:<14}  {r['sources']}")

    print("\nNext step:")
    print("  python3 run_omlx_semantic_judge.py \\")
    print(f"    --input {OUT.name} \\")
    print("    --out levi_calibration_talos_heldout_judged_v1.jsonl")


if __name__ == "__main__":
    main()
