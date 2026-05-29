#!/usr/bin/env python3
"""
Mine 300 semantic chain candidates from the Levi substrate.

Improvements over v1 miner:
- higher limit (300 candidates)
- stricter tier requirement: all 3 notes must be tier>=2 (topic files or summaries)
- no pure-date or config-style tokens as bridge tokens
- bridge tokens must be noun-like (≥5 chars, not verb-inflected)
- A↔C must be from different source files (not just different sources)
- excerpt includes source file tag for human review
"""
from __future__ import annotations

import importlib.util, json, re, sys
from pathlib import Path

MOD_PATH = Path("/tmp/rfm_substrate_path_test.py")
spec = importlib.util.spec_from_file_location("rfm_substrate_path_test", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["rfm_substrate_path_test"] = mod
spec.loader.exec_module(mod)

ROOT = Path(__file__).resolve().parent
OUT_CANDIDATES = ROOT / "levi_semantic_chain_candidates_v2.jsonl"

GENERIC = set("""
able accepted active actual additional adjacent ahead almost already another
available basic better broader careful clean clear common concrete correct
critical current different direct earlier enough exact explicit external final
first fresh full general good great hard high human important initial internal
known large latest likely live local long main major meaningful messy minimal
native new next obvious old ongoing only operational other personal possible
primary prior raw real recent related relevant same second separate simple
small specific stable strong sure technical top true useful weak whole
system memory file data user note output text block token result
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
    if re.match(r"memory/\d{4}-\d{2}-\d{2}\.md$", src):
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
    code_hits = len(re.findall(r"/Users/|/home/|\.py\b|\.ts\b|\.json\b|commit [0-9a-f]{6}", text, flags=re.I))
    if code_hits > 3:
        return False
    if len(mod.toks(text)) < 10:
        return False
    return True


def token_ok(t: str) -> bool:
    if t in GENERIC:
        return False
    if len(t) < 5 or len(t) > 18:
        return False
    if t[:4].isdigit():
        return False
    if re.search(r"(ing|tion|ment|ness|less|ness|ful|ible|ous|ive|ary|ory|ize|ise|ify)$", t) and len(t) > 9:
        return False
    if re.search(r"(command|service|warning|endpoint|backend|frontend|buildout|start|finish|function|artifact|parameter|argument|variable|config|setting|option|feature)$", t):
        return False
    return True


def note_tokens(notes: list[dict]) -> dict[str, set[str]]:
    base = {n["id"]: set(mod.toks(n["content"])) for n in notes}
    return {nid: {t for t in toks if token_ok(t)} for nid, toks in base.items()}


def excerpt(note: dict, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", note["content"]).strip()
    snip = text[:max_chars - 1] + ("..." if len(text) > max_chars else "")
    return f"[{note['source']}] {snip}"


def mine(notes: list[dict], limit: int = 300) -> list[dict]:
    nb = {n["id"]: n for n in notes}
    nt = note_tokens(notes)
    df, postings = mod.build_token_index(nt)

    rare = {t for t, d in df.items() if 2 <= d <= 5 and token_ok(t)}
    unique = {t for t, d in df.items() if d == 1 and token_ok(t) and len(t) >= 5}

    chains = []
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
                a_candidates = [nid for nid in postings[t1] if nid != b_id and t2 not in nt[nid]]
                c_candidates = [nid for nid in postings[t2] if nid != b_id and t1 not in nt[nid]]
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
                        # A and C must be from different files
                        if nb[a_id]["source"] == nb[c_id]["source"]:
                            continue
                        # require at least 2 distinct sources across A,B,C
                        sources = {nb[a_id]["source"], nb[b_id]["source"], nb[c_id]["source"]}
                        if len(sources) < 2:
                            continue
                        # A and C must not directly share meaningful tokens
                        direct_overlap = len(nt[a_id] & nt[c_id])
                        if direct_overlap > 1:
                            continue
                        tier_score = a_tier + b_tier + c_tier
                        start = starts[0]
                        key = (a_id, b_id, c_id)
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
    all_notes = mod.load_notes()
    notes = [n for n in all_notes if clean_chunk(n)]
    print(f"filtered notes: {len(notes)}")
    candidates = mine(notes, 300)
    print(f"mined candidates: {len(candidates)}")
    with OUT_CANDIDATES.open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_CANDIDATES}")
    print("\nSample (first 10):")
    for r in candidates[:10]:
        print(f"  {r['idx']:03d} {r['start_token']} -> {r['bridge1']} -> {r['bridge2']}  sources={r['sources']}")


if __name__ == "__main__":
    main()
