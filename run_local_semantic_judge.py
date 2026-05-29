#!/usr/bin/env python3
"""Run a local MLX model as a pilot semantic-chain judge.

This is a calibration probe, not the source of truth. It keeps all memory
content local and writes raw model outputs for inspection.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlx_lm import generate, load


ROOT = Path(__file__).resolve().parent
CANDIDATES_V1 = ROOT / "levi_semantic_chain_candidates_v1.jsonl"
CANDIDATES = ROOT / "levi_semantic_chain_candidates_v2.jsonl"
ANSWER_KEY = ROOT / "levi_semantic_chain_answer_key_v1.jsonl"
DEFAULT_OUT = ROOT / "levi_semantic_chain_local_judge_8b_v2.jsonl"
DEFAULT_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
LABELS = {"real_semantic", "weak_semantic", "artifact"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short(text: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def build_prompt(cand: dict, examples: list[dict] | None = None) -> str:
    ex = cand["excerpts"]
    example_block = ""
    if examples:
        chunks = []
        for eg in examples:
            chunks.append(
                f"""Example idx {eg['idx']}:
start_token={eg['start_token']} bridge1={eg['bridge1']} bridge2={eg['bridge2']}
A: {short(eg['excerpts']['a'], 260)}
B: {short(eg['excerpts']['b'], 260)}
C: {short(eg['excerpts']['c'], 260)}
Correct label: {eg['label']}
Why: {eg['rationale']}"""
            )
        example_block = "\nCalibration examples:\n" + "\n\n".join(chunks) + "\n"

    return f"""Judge this candidate three-hop memory chain.

Labels:
- real_semantic: A, B, and C form a defensible human-semantic memory chain.
- weak_semantic: at least one relation is meaningful, but the full chain drifts.
- artifact: mostly code/log/boilerplate/homonym/token coincidence.

Choose the stricter label when uncertain.
{example_block}

Candidate:
idx: {cand['idx']}
start_token: {cand['start_token']}
bridge1: {cand['bridge1']}
bridge2: {cand['bridge2']}
sources: {cand['sources']}

A: {short(ex['a'])}
B: {short(ex['b'])}
C: {short(ex['c'])}

Return only JSON:
{{"idx": {cand['idx']}, "label": "...", "rationale": "one sentence"}}
"""


def parse_output(idx: int, raw: str) -> dict:
    label = None
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        obj = json.loads(raw[start:end])
        label = obj.get("label")
        rationale = obj.get("rationale", "")
    except Exception:
        rationale = raw.strip().splitlines()[0][:300] if raw.strip() else ""
    if label not in LABELS:
        m = re.search(r"\b(real_semantic|weak_semantic|artifact)\b", raw)
        label = m.group(1) if m else "artifact"
    return {"idx": idx, "label": label, "rationale": rationale, "raw": raw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--few-shot",
        action="store_true",
        help="include held-out calibration examples from idx 69, 73, and 76",
    )
    args = ap.parse_args()

    all_candidates = read_jsonl(CANDIDATES)
    candidates = all_candidates[args.offset : args.offset + args.limit]
    examples = None
    if args.few_shot:
        answer = {int(row["idx"]): row for row in read_jsonl(ANSWER_KEY)}
        # load few-shot examples from v1 candidates (idx 2=real, 3=weak, 1=artifact)
        v1_candidates = read_jsonl(CANDIDATES_V1)
        by_idx_v1 = {int(row["idx"]): row for row in v1_candidates}
        examples = []
        for idx in [2, 3, 1]:
            if idx in answer and idx in by_idx_v1:
                row = {**by_idx_v1[idx], **answer[idx]}
                examples.append(row)
    print(f"loading model={args.model}")
    model, tokenizer = load(args.model)
    print(f"judging={len(candidates)} offset={args.offset}")

    with args.out.open("w", encoding="utf-8") as f:
        for i, cand in enumerate(candidates, 1):
            prompt = build_prompt(cand, examples=examples)
            raw = generate(
                model,
                tokenizer,
                prompt=prompt,
                verbose=False,
                max_tokens=180,
            )
            row = parse_output(cand["idx"], raw)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{i:03d}/{len(candidates):03d} idx={row['idx']} label={row['label']}")

    print(f"out={args.out}")


if __name__ == "__main__":
    main()
