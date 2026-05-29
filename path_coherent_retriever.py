"""
Path-Coherent Retriever v5 — reference implementation.

Key insight: standard BM25/cosine cannot cross zero-vocabulary gaps in multi-hop
memory chains. Path-coherent topology bridges these gaps by following cross-document
bridge tokens — tokens that recur across different source files but are rare enough
to be semantically specific.

Architecture:
  1. Anchor: BM25 retrieves top anchor_k notes for the query.
  2. Multi-bridge expansion: from each anchor, score top bridge_k cross-doc bridge
     tokens and follow their postings lists to bridge nodes.
  3. Second hop: repeat bridge selection from each bridge node.
  4. Separate pools: terminal nodes (step 3) and anchor/bridge nodes (steps 1-2)
     score independently so terminals are not displaced by accumulating anchors.
  5. Merge: top half of results from terminal pool, remainder from anchor pool.

Benchmark result (Levi substrate, Gemma-E4B judged, 2026-05-27):
  real_semantic chains (n=107, judge-validated)
    BM25:         terminal  6.5%   full  0.0%
    cosine:       terminal  6.5%   full  0.0%
    path v5:      terminal 15.0%   full  0.9%   hit@1  2.8%
"""
from __future__ import annotations
from typing import Callable


def build_token_sources(notes: list[dict], note_tokens: dict[str, set[str]]) -> dict[str, set[str]]:
    """token -> set of source files containing it."""
    token_sources: dict[str, set[str]] = {}
    for n in notes:
        for t in note_tokens[n["id"]]:
            token_sources.setdefault(t, set()).add(n["source"])
    return token_sources


def candidate_bridges(
    node_id: str,
    node_src: str,
    node_toks: set[str],
    exclude: set[str],
    token_sources: dict[str, set[str]],
    df: dict[str, int],
    top_k: int = 2,
) -> list[str]:
    """
    Return top-k bridge token candidates from a node.
    Scored by cross-document specificity: prefer tokens that appear in
    1-8 other source files (rare enough to be meaningful, common enough to bridge).
    """
    scored: list[tuple[float, str]] = []
    for t in node_toks:
        if t in exclude:
            continue
        cross = token_sources.get(t, set()) - {node_src}
        n_cross = len(cross)
        if n_cross == 0 or n_cross > 8:
            continue
        idf = 1.0 / (df.get(t, 1) ** 0.5)
        score = idf * (1.0 / n_cross) * min(len(t), 12) * 0.1
        scored.append((score, t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:top_k]]


def retrieve(
    query: str,
    tokenize: Callable[[str], list[str]],
    bm25_retrieve: Callable[[str, int], list[str]],
    note_by_id: dict[str, dict],
    note_tokens: dict[str, set[str]],
    postings: dict[str, list[str]],
    df: dict[str, int],
    token_sources: dict[str, set[str]],
    top_k: int = 10,
    anchor_k: int = 10,
    branch_k: int = 8,
    bridge_k: int = 2,
) -> list[str]:
    """
    Path-coherent multi-hop retrieval.

    Returns up to top_k note IDs, with terminal (3rd-hop) nodes prioritized.
    """
    query_toks = set(tokenize(query))
    anchor_scores: dict[str, float] = {}
    terminal_scores: dict[str, float] = {}

    for rank, a_id in enumerate(bm25_retrieve(query, anchor_k)):
        a_src = note_by_id[a_id]["source"]
        a_toks = note_tokens[a_id]
        anchor_scores[a_id] = max(anchor_scores.get(a_id, 0), 1.0 + 0.05 * (anchor_k - rank))

        bridges1 = candidate_bridges(a_id, a_src, a_toks, query_toks, token_sources, df, bridge_k)
        for t1 in bridges1:
            for b_id in postings.get(t1, [])[:branch_k]:
                if b_id == a_id:
                    continue
                b_src = note_by_id[b_id]["source"]
                b_toks = note_tokens[b_id]
                anchor_scores[b_id] = max(anchor_scores.get(b_id, 0), 0.8 + 0.02 * (anchor_k - rank))

                bridges2 = candidate_bridges(b_id, b_src, b_toks, query_toks | {t1}, token_sources, df, bridge_k)
                for t2 in bridges2:
                    for c_id in postings.get(t2, [])[:branch_k]:
                        if c_id in {a_id, b_id}:
                            continue
                        c_score = (
                            1.0
                            + 0.05 * (anchor_k - rank)
                            + 0.1 * (1.0 / max(df.get(t2, 1), 1) ** 0.5)
                        )
                        terminal_scores[c_id] = max(terminal_scores.get(c_id, 0), c_score)

    top_terminals = sorted(terminal_scores, key=lambda x: terminal_scores[x], reverse=True)
    top_anchors = sorted(anchor_scores, key=lambda x: anchor_scores[x], reverse=True)

    n_terminal = min(len(top_terminals), max(top_k // 2, 3))
    result: list[str] = []
    seen: set[str] = set()
    for nid in top_terminals[:n_terminal]:
        result.append(nid)
        seen.add(nid)
    for nid in top_anchors:
        if len(result) >= top_k:
            break
        if nid not in seen:
            result.append(nid)
            seen.add(nid)
    return result[:top_k]
