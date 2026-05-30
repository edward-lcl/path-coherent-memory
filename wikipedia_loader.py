#!/usr/bin/env python3
"""
Wikipedia biography corpus loader for path-coherent generalization experiments.

Uses HuggingFace wikimedia/wikipedia (20231101.en snapshot).
Filters to biography articles by checking for person-category signals.

Output: wikipedia_bio_corpus.json — list of {id, source, content} notes,
same schema as talos_corpus_notes.json, ready for the benchmark harness.
"""
import json, re, sys
from pathlib import Path
from typing import Iterator

try:
    from datasets import load_dataset, DownloadConfig
except ImportError:
    sys.exit("pip install datasets")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "wikipedia_bio_corpus.json"

# Signals that a Wikipedia article is a biography.
# These appear in the article text or categories for person articles.
BIO_PATTERNS = [
    re.compile(r"\bborn\s+\d{1,2}\s+\w+\s+\d{4}\b", re.I),   # "born 12 March 1954"
    re.compile(r"\bborn\s+\w+\s+\d{1,2},?\s+\d{4}\b", re.I), # "born March 12, 1954"
    re.compile(r"\b(?:is|was)\s+an?\s+\w+\s+(?:actor|musician|politician|scientist|"
               r"author|writer|director|athlete|footballer|tennis|basketball|"
               r"professor|researcher|journalist|artist|painter|composer|"
               r"philosopher|mathematician|physicist|biologist|chemist|engineer|"
               r"architect|judge|lawyer|general|admiral|president|prime minister|"
               r"senator|governor|chancellor|diplomat|explorer|inventor)\b", re.I),
]

def is_biography(text: str) -> bool:
    first_500 = text[:500]
    return any(p.search(first_500) for p in BIO_PATTERNS)


def stream_bios(max_articles: int = 50_000) -> Iterator[dict]:
    """Yield biography notes from Wikipedia until max_articles found."""
    print(f"Streaming wikimedia/wikipedia 20231101.en (target: {max_articles} bios)…")
    print("First run will download ~21GB; subsequent runs use HF cache.\n")

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    found = 0
    checked = 0
    for article in ds:
        checked += 1
        if checked % 50_000 == 0:
            print(f"  checked {checked:,} articles, found {found:,} bios…")

        text: str = article["text"]
        title: str = article["title"]

        if len(text) < 200:
            continue
        if not is_biography(text):
            continue

        # Truncate to first 1500 chars — the lead section contains bridge-worthy entities
        content = text[:1500].replace("\n", " ").strip()
        yield {
            "id": f"wiki:{title.replace(' ', '_')}",
            "source": title,
            "content": content,
        }
        found += 1
        if found >= max_articles:
            break

    print(f"\nDone. Found {found:,} biography articles from {checked:,} checked.")


def build_corpus(max_articles: int = 50_000, force: bool = False) -> list[dict]:
    if OUT.exists() and not force:
        notes = json.loads(OUT.read_text())
        print(f"Loaded cached corpus: {len(notes):,} articles from {OUT}")
        return notes

    notes = list(stream_bios(max_articles))
    OUT.write_text(json.dumps(notes, ensure_ascii=False))
    print(f"Saved {len(notes):,} biography notes → {OUT}")
    return notes


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=50_000, help="max bio articles to collect")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.add_argument("--stats", action="store_true", help="print corpus stats and exit")
    args = p.parse_args()

    notes = build_corpus(args.max, args.force)

    if args.stats or True:
        total_chars = sum(len(n["content"]) for n in notes)
        print(f"\nCorpus stats:")
        print(f"  Articles : {len(notes):,}")
        print(f"  Avg chars: {total_chars // max(len(notes), 1):,}")
        sample = notes[:3]
        for n in sample:
            print(f"\n  [{n['source']}]")
            print(f"  {n['content'][:200]}…")
