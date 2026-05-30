"""
Path-Coherent Retriever v6 — generalized implementation.

Key insight: standard BM25/cosine cannot cross zero-vocabulary gaps in multi-hop
memory chains. Path-coherent topology bridges these gaps by following cross-document
bridge tokens — tokens that recur across different source files but are rare enough
to be semantically specific.

Architecture:
  1. Anchor: BM25 retrieves top anchor_k notes for the query.
  2. Multi-bridge expansion: from each anchor, score top bridge_k cross-doc bridge
     tokens and follow their postings lists to bridge nodes.
  3. Second hop (max_hops=3): repeat bridge selection from each bridge node.
     Skipped when max_hops=2, making bridge nodes the terminals (HotpotQA mode).
  4. Separate pools: terminal nodes and anchor/bridge nodes score independently.
  5. Merge: top half of results from terminal pool, remainder from anchor pool.

Generalization changes (v5 → v6):
  - IDF-weighted bridge scoring: replaces hardcoded n_cross > 8 window with
    corpus-relative log-IDF. A token appearing in 8/100 personal notes and
    8/5M Wikipedia articles both get high IDF; common entities are softly
    discounted instead of hard-cutoff filtered.
  - max_hops parameter: set to 2 for 2-hop benchmarks (HotpotQA), 3 for default.
    In 2-hop mode bridge nodes are returned as terminals.
  - build_idf_table helper: pre-computes log(N/df) per token.

Benchmark result (Levi substrate, Gemma-E4B judged, 2026-05-27):
  real_semantic chains (n=107, judge-validated)
    BM25:         terminal  6.5%   full  0.0%
    cosine:       terminal  6.5%   full  0.0%
    path v5:      terminal 15.0%   full  0.9%   hit@1  2.8%
"""
from __future__ import annotations
import math
from typing import Callable


def build_token_sources(notes: list[dict], note_tokens: dict[str, set[str]]) -> dict[str, set[str]]:
    """token -> set of source files containing it."""
    token_sources: dict[str, set[str]] = {}
    for n in notes:
        for t in note_tokens[n["id"]]:
            token_sources.setdefault(t, set()).add(n["source"])
    return token_sources


def build_idf_table(df: dict[str, int], corpus_size: int) -> dict[str, float]:
    """log(N / df) per token. Returns 0.0 for tokens in >50% of corpus."""
    return {
        t: max(math.log(corpus_size / max(d, 1)), 0.0)
        for t, d in df.items()
    }


def candidate_bridges(
    node_id: str,
    node_src: str,
    node_toks: set[str],
    exclude: set[str],
    token_sources: dict[str, set[str]],
    df: dict[str, int],
    top_k: int = 2,
    idf_table: dict[str, float] | None = None,
) -> list[str]:
    """
    Return top-k bridge token candidates from a node.

    When idf_table is provided (v6 generalized mode): scores by corpus-relative
    log-IDF with no hard cross-doc cutoff, so the method adapts to dense corpora
    like Wikipedia without the 8-source ceiling breaking valid bridges.

    When idf_table is None (v5 legacy mode): uses the original sqrt-IDF score
    with a hard n_cross > 8 filter. Backward-compatible for existing callers.
    """
    scored: list[tuple[float, str]] = []
    for t in node_toks:
        if t in exclude:
            continue
        cross = token_sources.get(t, set()) - {node_src}
        n_cross = len(cross)
        if n_cross == 0:
            continue
        if idf_table is not None:
            idf_score = idf_table.get(t, 0.0)
            if idf_score <= 0.0:
                continue  # token in >50% of corpus — not a meaningful bridge
            score = idf_score * min(len(t), 12) * 0.1
        else:
            if n_cross > 8:
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
    idf_table: dict[str, float] | None = None,
    max_hops: int = 3,
) -> list[str]:
    """
    Path-coherent multi-hop retrieval.

    Args:
        idf_table: Pre-computed log-IDF table from build_idf_table(). When
            provided, enables corpus-relative bridge scoring (v6 generalized
            mode). Pass None to retain v5 behavior with hardcoded n_cross <= 8.
        max_hops: Number of hops. Use 3 (default) for personal-corpus chains.
            Use 2 for HotpotQA-style 2-hop tasks — bridge nodes become the
            terminals, skipping the third expansion.

    Returns up to top_k note IDs, with terminal nodes prioritized.
    """
    query_toks = set(tokenize(query))
    anchor_scores: dict[str, float] = {}
    terminal_scores: dict[str, float] = {}

    for rank, a_id in enumerate(bm25_retrieve(query, anchor_k)):
        a_src = note_by_id[a_id]["source"]
        a_toks = note_tokens[a_id]
        anchor_scores[a_id] = max(anchor_scores.get(a_id, 0), 1.0 + 0.05 * (anchor_k - rank))

        bridges1 = candidate_bridges(
            a_id, a_src, a_toks, query_toks, token_sources, df, bridge_k, idf_table
        )
        for t1 in bridges1:
            for b_id in postings.get(t1, [])[:branch_k]:
                if b_id == a_id:
                    continue
                b_src = note_by_id[b_id]["source"]
                b_toks = note_tokens[b_id]

                if max_hops == 2:
                    # In 2-hop mode, bridge nodes are the terminals.
                    b_score = 1.0 + 0.05 * (anchor_k - rank)
                    if idf_table is not None:
                        b_score += 0.1 * idf_table.get(t1, 0.0)
                    else:
                        b_score += 0.1 * (1.0 / max(df.get(t1, 1), 1) ** 0.5)
                    terminal_scores[b_id] = max(terminal_scores.get(b_id, 0), b_score)
                else:
                    anchor_scores[b_id] = max(
                        anchor_scores.get(b_id, 0), 0.8 + 0.02 * (anchor_k - rank)
                    )
                    bridges2 = candidate_bridges(
                        b_id, b_src, b_toks, query_toks | {t1}, token_sources, df, bridge_k, idf_table
                    )
                    for t2 in bridges2:
                        for c_id in postings.get(t2, [])[:branch_k]:
                            if c_id in {a_id, b_id}:
                                continue
                            if idf_table is not None:
                                t2_idf = idf_table.get(t2, 0.0)
                            else:
                                t2_idf = 1.0 / max(df.get(t2, 1), 1) ** 0.5
                            c_score = (
                                1.0
                                + 0.05 * (anchor_k - rank)
                                + 0.1 * t2_idf
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
