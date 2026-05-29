#!/usr/bin/env python3
"""Build judge-ready artifacts for Levi substrate semantic-chain evaluation.

This script takes the mined substrate audit pack and the current human labels,
then emits:

- a masked JSONL candidate pack for judging
- an answer-key JSONL with existing labels and rationales
- a calibration summary with representative examples per label

It deliberately does not run a judge. The point is to freeze the input surface
so humans or LLMs can judge the same candidates without moving the goalposts.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIT_PACK = Path("/tmp/rfm_substrate_audit_pack.tsv")
LABELS = ROOT / "levi_substrate_labels_v1.tsv"

MASKED_JSONL = ROOT / "levi_semantic_chain_candidates_v1.jsonl"
ANSWER_KEY_JSONL = ROOT / "levi_semantic_chain_answer_key_v1.jsonl"
CALIBRATION_JSON = ROOT / "levi_semantic_chain_calibration_v1.json"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def compact_candidate(row: dict[str, str]) -> dict:
    return {
        "idx": int(row["idx"]),
        "start_token": row["start_token"],
        "bridge1": row["bridge1"],
        "bridge2": row["bridge2"],
        "required_ids": [x.strip() for x in row["required_ids"].split("|")],
        "sources": [x.strip() for x in row["sources"].split("|")],
        "retrieval_flags": {
            "path_terminal_hit": row["path_terminal_hit"] == "True",
            "path_full_hit": row["path_full_hit"] == "True",
            "bm25_terminal_hit": row["bm25_terminal_hit"] == "True",
            "cosine_terminal_hit": row["cosine_terminal_hit"] == "True",
        },
        "path_ret": [x.strip() for x in row["path_ret"].split("|")],
        "excerpts": {
            "a": row["a_excerpt"],
            "b": row["b_excerpt"],
            "c": row["c_excerpt"],
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not AUDIT_PACK.exists():
        raise SystemExit(f"missing audit pack: {AUDIT_PACK}")
    if not LABELS.exists():
        raise SystemExit(f"missing labels: {LABELS}")

    audit_rows = read_tsv(AUDIT_PACK)
    label_rows = read_tsv(LABELS)
    labels = {int(row["idx"]): row for row in label_rows}

    candidates = [compact_candidate(row) for row in audit_rows]
    answer_key = []
    by_label: dict[str, list[dict]] = defaultdict(list)
    for cand in candidates:
        label = labels.get(cand["idx"])
        if not label:
            continue
        keyed = {
            "idx": cand["idx"],
            "label": label["label"],
            "rationale": label["rationale"],
            "start_token": cand["start_token"],
            "bridge1": cand["bridge1"],
            "bridge2": cand["bridge2"],
            "sources": cand["sources"],
        }
        answer_key.append(keyed)
        by_label[label["label"]].append({**keyed, "excerpts": cand["excerpts"]})

    write_jsonl(MASKED_JSONL, candidates)
    write_jsonl(ANSWER_KEY_JSONL, answer_key)

    counts = Counter(row["label"] for row in answer_key)
    calibration = {
        "candidate_count": len(candidates),
        "labeled_count": len(answer_key),
        "label_counts": dict(sorted(counts.items())),
        "label_yield": {
            label: count / max(len(answer_key), 1)
            for label, count in sorted(counts.items())
        },
        "representative_examples": {
            label: rows[:3]
            for label, rows in sorted(by_label.items())
        },
        "files": {
            "prompt": str(ROOT / "semantic_chain_judge_prompt.md"),
            "masked_candidates": str(MASKED_JSONL),
            "answer_key": str(ANSWER_KEY_JSONL),
            "calibration": str(CALIBRATION_JSON),
        },
    }
    CALIBRATION_JSON.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"candidates={len(candidates)} -> {MASKED_JSONL}")
    print(f"answer_key={len(answer_key)} -> {ANSWER_KEY_JSONL}")
    print("labels=" + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    print(f"calibration={CALIBRATION_JSON}")


if __name__ == "__main__":
    main()
