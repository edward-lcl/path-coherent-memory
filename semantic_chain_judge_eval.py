#!/usr/bin/env python3
"""Evaluate semantic-chain judge outputs against the v1 human labels.

Usage:

    python3 semantic_chain_judge_eval.py
    python3 semantic_chain_judge_eval.py judge_results.jsonl

If no judge result file is provided, the script uses the human answer key as a
sanity check. Judge result rows must contain at least `idx` and `label`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANSWER_KEY = ROOT / "levi_semantic_chain_answer_key_v1.jsonl"
DEFAULT_RESULTS = ANSWER_KEY


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    result_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    gold_rows = read_jsonl(ANSWER_KEY)
    pred_rows = read_jsonl(result_path)

    gold = {int(row["idx"]): row["label"] for row in gold_rows}
    pred = {int(row["idx"]): row["label"] for row in pred_rows}

    labels = sorted(set(gold.values()) | set(pred.values()))
    matrix: dict[str, Counter] = defaultdict(Counter)
    missing = []
    for idx, gold_label in gold.items():
        pred_label = pred.get(idx)
        if pred_label is None:
            missing.append(idx)
            continue
        matrix[gold_label][pred_label] += 1

    correct = sum(matrix[label][label] for label in labels)
    scored = sum(sum(row.values()) for row in matrix.values())
    accuracy = correct / scored if scored else 0.0

    def binary_stats(positive: set[str]) -> dict[str, float | int]:
        tp = fp = fn = 0
        for idx, gold_label in gold.items():
            if idx not in pred:
                continue
            pred_label = pred[idx]
            g = gold_label in positive
            p = pred_label in positive
            tp += int(g and p)
            fp += int((not g) and p)
            fn += int(g and (not p))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    report = {
        "result_file": str(result_path),
        "gold_count": len(gold),
        "scored_count": scored,
        "missing_count": len(missing),
        "accuracy": accuracy,
        "gold_distribution": dict(Counter(gold.values())),
        "pred_distribution": dict(Counter(pred.values())),
        "confusion": {g: dict(matrix[g]) for g in labels},
        "accept_real_only": binary_stats({"real_semantic"}),
        "accept_real_plus_weak": binary_stats({"real_semantic", "weak_semantic"}),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
