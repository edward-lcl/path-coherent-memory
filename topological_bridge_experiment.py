#!/usr/bin/env python3
"""Topology-guided bridge traversal experiment for the hard RFM corpus.

This tests a simple non-parametric bridge-token selector:
the bridge token is the non-query token in a note that also appears elsewhere
in the corpus. Structural verbs tend to be local; bridge entities recur across
adjacent notes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluator import BM25Backend, RFMCharMMRBackend, tokenize


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus_hard_v2.json"

STOP = set(
    "what is the of for in to by from with as a an and or its it this that "
    "owner location dependency tier status jobs state traffic metrics around "
    "into under over held belongs group directly runtime execution proxy layer "
    "same cluster boundary component continuously computation configuration "
    "framework namespace manifest overflow capacity snapshots asynchronously"
    .split()
)


def main() -> None:
    corpus = json.loads(CORPUS.read_text())
    notes = corpus["notes"]
    note_by_id = {n["id"]: n for n in notes}

    bm25 = BM25Backend()
    bm25.index(notes)

    rfm = RFMCharMMRBackend()
    rfm.index(notes)
    stability = rfm._soft_stability()

    note_tokens = {
        n["id"]: {
            t for t in tokenize(n["content"])
            if len(t) >= 4 and not t.isdigit() and t not in STOP
        }
        for n in notes
    }

    df: dict[str, int] = {}
    for toks in note_tokens.values():
        for t in toks:
            df[t] = df.get(t, 0) + 1

    def bridge_token(note_id: str, seen: set[str]) -> str | None:
        candidates = []
        for token in note_tokens[note_id]:
            if token in seen:
                continue
            other_df = df.get(token, 0) - 1
            # Entity bridge tokens should recur in at least one other note.
            # Prefer recurring but still rare terms; generic repeated words are
            # penalised by total document frequency.
            score = 10.0 * min(other_df, 2) - 0.2 * df.get(token, 0) + 0.01 * len(token)
            candidates.append((score, token, other_df, df.get(token, 0)))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def resonance(spec_a: list[float], spec_b: list[float]) -> float:
        return max(0.0, rfm._weighted_resonance(spec_a, spec_b, stability))

    def token_search(token: str | None, exclude: set[str], k: int) -> list[tuple[float, str]]:
        if not token:
            return []
        q_spec = rfm._compute_spectrum([token])
        scored = []
        for i, note in enumerate(notes):
            if note["id"] in exclude:
                continue
            scored.append((resonance(q_spec, rfm._spectra[i]), note["id"]))
        scored.sort(reverse=True)
        return scored[:k]

    answer_hits = 0
    hitk = 0
    total_required_hits = 0
    total_retrieved = 0
    details = []

    for qa in corpus["qa_pairs"]:
        query_tokens = set(tokenize(qa["question"]))
        path_scores: dict[str, float] = {}
        paths = []

        query_entity = next(
            (t for t in reversed(tokenize(qa["question"])) if t not in STOP and len(t) >= 4),
            None,
        )
        anchors = [
            nid for nid in bm25.retrieve(qa["question"], 12)
            if query_entity and query_entity in note_tokens[nid]
        ]
        if not anchors:
            anchors = bm25.retrieve(qa["question"], 3)

        for anchor_rank, bridge1_id in enumerate(anchors[:5]):
            token1 = bridge_token(bridge1_id, seen=query_tokens)
            for score12, bridge2_id in token_search(token1, {bridge1_id}, 5):
                token2 = bridge_token(bridge2_id, seen=query_tokens | {token1 or ""})
                for score23, answer_id in token_search(token2, {bridge1_id, bridge2_id}, 5):
                    path_score = score12 + score23 + 0.1 * (8 - anchor_rank) / 8
                    paths.append((path_score, bridge1_id, bridge2_id, answer_id, token1, token2))
                    path_scores[bridge1_id] = max(path_scores.get(bridge1_id, 0), path_score + 0.7)
                    path_scores[bridge2_id] = max(path_scores.get(bridge2_id, 0), path_score + 0.7)
                    path_scores[answer_id] = max(path_scores.get(answer_id, 0), path_score + 2.0)

        retrieved = [
            nid for nid, _ in sorted(path_scores.items(), key=lambda item: item[1], reverse=True)[:5]
        ]
        required = set(qa["required_notes"])
        answer_id = qa["required_notes"][-1]
        if required & set(retrieved):
            hitk += 1
        if answer_id in retrieved:
            answer_hits += 1
        total_required_hits += len(required & set(retrieved))
        total_retrieved += len(retrieved)
        details.append((qa["id"], answer_id in retrieved, retrieved, qa["required_notes"], sorted(paths, reverse=True)[:1]))

    precision = total_required_hits / total_retrieved if total_retrieved else 0.0
    recall = total_required_hits / (len(corpus["qa_pairs"]) * 3)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print("RFM + topological bridge-token selector")
    print(f"Answer nodes: {answer_hits}/20 = {answer_hits / 20:.1%}")
    print(f"F1: {f1:.3f}  hit@k: {hitk / 20:.3f}")
    print()
    for row in details:
        print(row)


if __name__ == "__main__":
    main()
