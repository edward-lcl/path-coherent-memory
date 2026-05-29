"""
Load real Levi memory files into RFM note format.

Sources:
  - memory/topics/*.md       — structured topic files
  - memory/YYYY-MM-DD.md     — session notes
  - MEMORY.md                — index lines
  - USER.md, IDENTITY.md     — identity/preference docs (paragraph-chunked)
"""

import re
import json
import os
from pathlib import Path

WORKSPACE = Path('/Users/edward/.openclaw/workspace')
OUTPUT    = str(WORKSPACE / 'research/rfm/levi_corpus.json')


def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def chunk_markdown(text: str, min_len: int = 40) -> list[str]:
    text = re.sub(r'^---.*?---\s*', '', text, flags=re.DOTALL)
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if not p or re.match(r'^#+\s', p) or re.match(r'^[-=]{3,}$', p):
            continue
        if len(p) < min_len:
            continue
        p = re.sub(r'^\s*[-*]\s+', '', p, flags=re.MULTILINE)
        p = clean(p)
        if len(p) >= min_len:
            chunks.append(p)
    return chunks


def make_note(nid: str, content: str, source: str, tags: list[str]) -> dict:
    return {"id": nid, "content": content, "category": "fact",
            "tags": tags, "source": source}


def read_file(path) -> str | None:
    try:
        return Path(path).read_text(encoding='utf-8', errors='replace')
    except Exception:
        return None


def load_corpus() -> dict:
    notes = []
    counter = [0]

    def next_id(prefix: str = "levi") -> str:
        counter[0] += 1
        return f"{prefix}_{counter[0]:04d}"

    # ── memory/topics/*.md ────────────────────────────────────────────────
    topic_dir = WORKSPACE / 'memory/topics'
    for path in sorted(topic_dir.glob('*.md')):
        text = read_file(path)
        if not text:
            continue
        slug = path.stem.replace('-', '_')[:12]
        for chunk in chunk_markdown(text):
            notes.append(make_note(next_id(slug), chunk,
                                   f"memory/topics/{path.name}",
                                   ["memory", "topic", slug]))

    # ── memory/YYYY-MM-DD.md session files ────────────────────────────────
    for path in sorted((WORKSPACE / 'memory').glob('????-??-??.md')):
        text = read_file(path)
        if not text:
            continue
        date = path.stem
        for chunk in chunk_markdown(text):
            notes.append(make_note(next_id("session"), chunk,
                                   f"memory/{path.name}",
                                   ["memory", "session", date]))

    # ── MEMORY.md index lines ─────────────────────────────────────────────
    mem_text = read_file(WORKSPACE / 'MEMORY.md')
    if mem_text:
        for line in mem_text.splitlines():
            line = line.strip()
            if line.startswith('- [') and len(line) > 50:
                plain = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', line)
                plain = clean(plain.lstrip('- '))
                if len(plain) >= 40:
                    notes.append(make_note(next_id("memidx"), plain,
                                           "MEMORY.md", ["memory", "index"]))

    # ── Identity files ────────────────────────────────────────────────────
    for fname in ['USER.md', 'IDENTITY.md']:
        text = read_file(WORKSPACE / fname)
        if not text:
            continue
        for chunk in chunk_markdown(text, min_len=60):
            notes.append(make_note(next_id(fname.replace('.md','').lower()),
                                   chunk, fname,
                                   ["identity", fname.lower()]))

    metadata = {
        "size": "real",
        "source": "levi_memory",
        "total_notes": len(notes),
        "sources": sorted({n["source"] for n in notes}),
    }

    return {
        "metadata": metadata,
        "notes": notes,
        "contradictions": [],
        "alias_groups": [],
        "multihop_chains": [],
        "qa_pairs": [],
    }


if __name__ == '__main__':
    corpus = load_corpus()
    with open(OUTPUT, 'w') as f:
        json.dump(corpus, f, indent=2)
    m = corpus["metadata"]
    print(f"Loaded {m['total_notes']} notes from {len(m['sources'])} sources")
    for s in m["sources"]:
        count = sum(1 for n in corpus["notes"] if n["source"] == s)
        print(f"  {count:>4}  {s}")
    print(f"\nOutput: {OUTPUT}")
