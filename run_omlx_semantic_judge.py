#!/usr/bin/env python3
"""
Semantic chain judge using oMLX HTTP API.
Judges all candidates in levi_semantic_chain_candidates_v2.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / "levi_semantic_chain_candidates_v2.jsonl"
ANSWER_KEY_V1 = ROOT / "levi_semantic_chain_answer_key_v1.jsonl"
CANDIDATES_V1 = ROOT / "levi_semantic_chain_candidates_v1.jsonl"
DEFAULT_OUT = ROOT / "levi_semantic_chain_omlx_judge_v1.jsonl"
OMLX_URL = "http://127.0.0.1:8000/v1/chat/completions"
OMLX_KEY = "babablacksheep"
DEFAULT_MODEL = "gemma-4-E4B-it-MLX-4bit"
LABELS = {"real_semantic", "weak_semantic", "artifact"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def short(text: str, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def build_prompt(cand: dict, examples: list[dict] | None = None) -> str:
    ex = cand["excerpts"]
    example_block = ""
    if examples:
        chunks = []
        for eg in examples:
            chunks.append(
                f"Example (idx {eg['idx']}):\n"
                f"  start={eg['start_token']} bridge1={eg['bridge1']} bridge2={eg['bridge2']}\n"
                f"  A: {short(eg['excerpts']['a'], 200)}\n"
                f"  B: {short(eg['excerpts']['b'], 200)}\n"
                f"  C: {short(eg['excerpts']['c'], 200)}\n"
                f"  Label: {eg['label']}\n"
                f"  Why: {eg.get('rationale','')}"
            )
        example_block = "\nCalibration examples:\n" + "\n\n".join(chunks) + "\n"

    # bridge info differs between token-mined (bridge1/bridge2) and semantic v3 (sim_ab/sim_bc)
    if "bridge1" in cand:
        bridge_line = f"bridge1: {cand['bridge1']}\nbridge2: {cand['bridge2']}"
    else:
        bridge_line = f"sim_AB: {cand.get('sim_ab','?')}  sim_BC: {cand.get('sim_bc','?')}  sim_AC: {cand.get('sim_ac','?')}  (embedding-similarity chain)"

    return f"""You are a rigorous judge for a memory-retrieval research benchmark.

A "semantic memory chain" is a 3-hop path A→B→C through human memory notes where:
- A is the anchor note (start_token is its distinctive token)
- B is a bridge note connecting A to C
- C is the terminal note
- The relations A→B and B→C are meaningful to a human reader (not just token coincidence)

Labels (choose the strictest that fits):
- real_semantic: All three hops are defensible human-semantic memory connections.
- weak_semantic: At least one hop is meaningful, but the full chain drifts or one connection is loose.
- artifact: Mostly token coincidence, code/log noise, or structural repetition with no real semantic chain.
{example_block}
Now judge this candidate:
idx: {cand['idx']}
start_token: {cand['start_token']}
{bridge_line}
sources: {cand['sources']}

A: {short(ex['a'])}
B: {short(ex['b'])}
C: {short(ex['c'])}

Return only valid JSON on one line:
{{"idx": {cand['idx']}, "label": "real_semantic|weak_semantic|artifact", "rationale": "one sentence"}}"""


def call_omlx(model: str, prompt: str, max_tokens: int = 200, retries: int = 3) -> str:
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    headers = {
        "Authorization": f"Bearer {OMLX_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OMLX_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def parse_output(idx: int, raw: str) -> dict:
    label = None
    rationale = ""
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
    return {"idx": idx, "label": label, "rationale": rationale, "raw": raw[:500]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--few-shot", action="store_true")
    args = ap.parse_args()

    input_path = args.input if args.input else CANDIDATES
    candidates = read_jsonl(input_path)
    if args.limit:
        candidates = candidates[args.offset: args.offset + args.limit]
    else:
        candidates = candidates[args.offset:]

    # resume: skip already-judged
    done = set()
    if args.out.exists():
        for row in read_jsonl(args.out):
            done.add(int(row["idx"]))
        candidates = [c for c in candidates if int(c["idx"]) not in done]
        print(f"resuming: {len(done)} already done, {len(candidates)} remaining")

    examples = None
    if args.few_shot:
        answer = {int(r["idx"]): r for r in read_jsonl(ANSWER_KEY_V1)}
        v1 = {int(r["idx"]): r for r in read_jsonl(CANDIDATES_V1)}
        examples = []
        for idx in [2, 3, 1]:  # real, weak, artifact
            if idx in answer and idx in v1:
                examples.append({**v1[idx], **answer[idx]})

    print(f"model={args.model}  judging={len(candidates)}  few_shot={args.few_shot}")
    t0 = time.time()

    with args.out.open("a", encoding="utf-8") as f:
        for i, cand in enumerate(candidates, 1):
            prompt = build_prompt(cand, examples=examples)
            raw = call_omlx(args.model, prompt)
            row = parse_output(int(cand["idx"]), raw)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining = (len(candidates) - i) / rate if rate > 0 else 0
            print(f"{i:03d}/{len(candidates):03d} idx={row['idx']:03d} label={row['label']:<14} eta={remaining/60:.1f}m")

    # summary
    results = read_jsonl(args.out)
    from collections import Counter
    counts = Counter(r["label"] for r in results)
    total = len(results)
    print(f"\nDone: {total} judged")
    for lab in ["real_semantic", "weak_semantic", "artifact"]:
        n = counts.get(lab, 0)
        print(f"  {lab:<16} {n:3d}  ({100*n/total:.1f}%)")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
