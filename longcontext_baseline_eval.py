#!/usr/bin/env python3
"""
E1 — Long-Context Baseline Experiment

Tests whether a long-context LLM can identify the terminal document
when given dense-retrieved candidate sets of varying sizes.

The frontier-lab objection: "Just use long context — shove everything in."
The counter: if the bridge document isn't in the candidate set, no context
window length saves you. Retrieval failure happens at candidate selection.

Protocol:
- For each embedding-disjoint chain (Talos cos<0.3, n=131):
    Condition A: top-K dense candidates (bridge typically absent)
    Condition B: top-K + bridge injected (oracle condition)
    Condition C: top-K + bridge + terminal injected (sanity check)
- Feed candidates to a long-context model with the question
- Measure: does it identify the terminal document in each condition?

Prediction:
    Cond A (dense top-K, no bridge): ~0% — same as dense retrieval
    Cond B (bridge injected): >50% — model can reason once bridge is visible
    Gap A→B = the retrieval failure, not a reasoning failure

This separates retrieval failure from reasoning failure, which is the
key argument against "just use a smarter model."
"""
from __future__ import annotations
import json, os, sys, urllib.request, urllib.error, random
from pathlib import Path
import numpy as np

RFM = Path(__file__).parent
sys.path.insert(0, str(RFM))
from embedding_bridge_retriever import embed_texts

# ── config ──────────────────────────────────────────────────────────────────
OMLX_URL   = "http://127.0.0.1:8000/v1/chat/completions"
OMLX_MODEL = "Qwen3-27B"          # long-context capable; swap to any oMLX model
K_VALUES   = [10, 20, 50]         # candidate set sizes to test
N_CHAINS   = 131                  # Talos embedding-disjoint subset size
TALOS_CHAINS_FILE = RFM / "talos_heldout_judged_v1.jsonl"   # adjust if needed
OUT_FILE   = RFM / "longcontext_baseline_results.json"

PROMPT_TEMPLATE = """\
You are a retrieval assistant. Below are {k} candidate documents retrieved for \
a query. Your task: identify which document (by its index, 0-based) is the \
best answer to the query.

Query: {query}

Candidates:
{candidates}

Reply with ONLY the index number of the best matching document. If none match, \
reply with -1."""


def _load_key() -> str:
    key_file = RFM / ".omlx_key"
    if key_file.exists():
        return key_file.read_text().strip()
    return os.environ.get("OMLX_API_KEY", "babablacksheep")


def call_llm(prompt: str, model: str = OMLX_MODEL, max_tokens: int = 16) -> str:
    key = _load_key()
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        OMLX_URL,
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR:{e}"


def parse_index(response: str, k: int) -> int:
    """Extract integer index from LLM response."""
    import re
    m = re.search(r"-?\d+", response)
    if m:
        idx = int(m.group())
        if -1 <= idx < k:
            return idx
    return -1


def load_talos_chains() -> list[dict]:
    """Load Talos chains with text content. Returns normalized chain dicts."""
    # Primary: candidates file has full excerpt text (a=anchor, b=bridge, c=terminal)
    # Cross-reference with judge file to filter to real_semantic only
    candidate_files = [
        RFM / "talos_semantic_chain_candidates_v2.jsonl",
        RFM / "talos_semantic_chain_candidates_v1.jsonl",
    ]
    judge_files = [
        RFM / "talos_semantic_chain_omlx_judge_v2.jsonl",
        RFM / "talos_semantic_chain_omlx_judge_v1.jsonl",
    ]

    # Load judge labels
    valid_idxs: set[int] = set()
    for jf in judge_files:
        if jf.exists():
            with open(jf) as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    try:
                        j = json.loads(line)
                        if j.get("label") in ("real_semantic", "yes", "valid"):
                            valid_idxs.add(j["idx"])
                    except Exception:
                        pass
            if valid_idxs:
                print(f"Loaded {len(valid_idxs)} valid judge labels from {jf.name}")
                break

    # Load candidates and normalize to {query, bridge, terminal, sim_ac}
    for cf in candidate_files:
        if not cf.exists():
            continue
        chains = []
        with open(cf) as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try:
                    c = json.loads(line)
                    # Filter to valid chains if we have labels
                    if valid_idxs and c.get("idx") not in valid_idxs:
                        continue
                    excerpts = c.get("excerpts", {})
                    a_text = excerpts.get("a", "")
                    b_text = excerpts.get("b", "")
                    c_text = excerpts.get("c", "")
                    if not a_text or not c_text:
                        continue
                    chains.append({
                        "idx": c.get("idx"),
                        "query_text": a_text,
                        "intermediate_text": b_text,
                        "terminal_text": c_text,
                        "sim_ac": c.get("sim_ac", 1.0),
                        "required_ids": c.get("required_ids", []),
                        "sources": c.get("sources", []),
                    })
                except Exception:
                    pass
        if chains:
            print(f"Loaded {len(chains)} chains from {cf.name}")
            return chains

    print("WARNING: No Talos chain file found. Run talos_benchmark.py first.")
    return []


def build_corpus_from_chains(chains: list[dict]) -> tuple[list[str], list[str]]:
    """Build a flat corpus of unique document texts + IDs from chain data."""
    seen, texts, ids = set(), [], []
    for c in chains:
        for field in ["query_text", "intermediate_text", "terminal_text"]:
            txt = c.get(field, "")
            if txt and txt not in seen:
                seen.add(txt)
                texts.append(txt)
                ids.append(f"doc_{len(ids)}")
    return texts, ids


def run_experiment(chains: list[dict], corpus_texts: list[str],
                   corpus_vecs: np.ndarray, k: int) -> dict:
    """
    Run one condition (top-K dense candidates) on all chains.
    Returns per-chain results with hit/miss for:
      - dense_only: top-K by dense sim
      - bridge_injected: top-(K-1) dense + bridge doc
      - terminal_injected: top-(K-2) dense + bridge + terminal (sanity check)
    """
    norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    V = corpus_vecs / norms

    results = {"k": k, "n": 0, "dense_only": 0, "bridge_injected": 0,
               "terminal_injected": 0, "chains": []}

    for i, chain in enumerate(chains):
        query = chain.get("query_text") or chain.get("anchor_text") or chain.get("a_text", "")
        bridge_text = chain.get("intermediate_text") or chain.get("b_text", "")
        terminal_text = chain.get("terminal_text") or chain.get("c_text", "")
        if not query or not terminal_text:
            continue

        # Find terminal index in corpus
        try:
            term_idx = corpus_texts.index(terminal_text)
        except ValueError:
            continue

        # Dense retrieval: top-K by cosine sim to query embedding
        q_vec = np.array(embed_texts([query[:512]])[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec); q_vec = q_vec / max(q_norm, 1e-9)
        sims = V @ q_vec
        top_k_idx = list(np.argsort(-sims)[:k])

        def run_condition(cand_indices: list[int], label: str) -> bool:
            cand_texts = [corpus_texts[j] for j in cand_indices]
            if terminal_text not in cand_texts:
                return False  # terminal not in candidates at all
            gold_pos = cand_texts.index(terminal_text)
            formatted = "\n\n".join(
                f"[{n}] {t[:600]}" for n, t in enumerate(cand_texts)
            )
            prompt = PROMPT_TEMPLATE.format(
                k=len(cand_texts), query=query[:400], candidates=formatted
            )
            response = call_llm(prompt)
            predicted = parse_index(response, len(cand_texts))
            hit = predicted == gold_pos
            return hit

        # Condition A: dense top-K only
        dense_hit = run_condition(top_k_idx, "dense_only")

        # Condition B: replace last slot with bridge (if not already in top-K)
        bridge_idx = None
        if bridge_text:
            try:
                bridge_idx = corpus_texts.index(bridge_text)
            except ValueError:
                pass
        if bridge_idx is not None and bridge_idx not in top_k_idx:
            bridge_cands = top_k_idx[:k-1] + [bridge_idx]
        else:
            bridge_cands = top_k_idx
        bridge_hit = run_condition(bridge_cands, "bridge_injected")

        # Condition C: inject both bridge and terminal
        if bridge_idx is not None and bridge_idx not in top_k_idx:
            oracle_cands = top_k_idx[:k-2] + [bridge_idx, term_idx]
        else:
            oracle_cands = top_k_idx[:k-1] + [term_idx]
        terminal_hit = run_condition(oracle_cands, "terminal_injected")

        results["n"] += 1
        results["dense_only"]        += int(dense_hit)
        results["bridge_injected"]   += int(bridge_hit)
        results["terminal_injected"] += int(terminal_hit)
        results["chains"].append({
            "query": query[:100],
            "dense_hit": dense_hit,
            "bridge_hit": bridge_hit,
            "terminal_hit": terminal_hit,
        })

        if (i + 1) % 10 == 0:
            n = results["n"]
            print(f"  [{i+1}] n={n} "
                  f"dense={100*results['dense_only']/max(n,1):.1f}% "
                  f"bridge={100*results['bridge_injected']/max(n,1):.1f}% "
                  f"terminal={100*results['terminal_injected']/max(n,1):.1f}%")

    return results


def main():
    print(f"E1 — Long-Context Baseline (model: {OMLX_MODEL})")
    print("=" * 60)

    # Check oMLX is reachable
    try:
        req = urllib.request.Request(
            OMLX_URL.replace("/chat/completions", "/models"),
            headers={"Authorization": f"Bearer {_load_key()}"}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            models = json.loads(r.read())
            available = [m["id"] for m in models.get("data", [])]
            print(f"oMLX available models: {available[:5]}...")
            if OMLX_MODEL not in available:
                print(f"WARNING: {OMLX_MODEL} not found. Available: {available}")
                print("Set OMLX_MODEL at top of script to an available model.")
    except Exception as e:
        print(f"WARNING: oMLX check failed ({e}) — continuing anyway")

    # Load chains
    chains = load_talos_chains()
    if not chains:
        print("No chains found. Exiting.")
        return

    # Compute anchor↔terminal cosine to filter to embedding-disjoint subset
    print("Computing anchor↔terminal cosine similarity for disjoint filtering...")
    a_texts = [c["query_text"][:512] for c in chains]
    c_texts = [c["terminal_text"][:512] for c in chains]
    a_vecs = np.array(embed_texts(a_texts, batch_size=64), dtype=np.float32)
    c_vecs = np.array(embed_texts(c_texts, batch_size=64), dtype=np.float32)
    an = np.linalg.norm(a_vecs, axis=1, keepdims=True); an[an==0]=1; a_vecs /= an
    cn = np.linalg.norm(c_vecs, axis=1, keepdims=True); cn[cn==0]=1; c_vecs /= cn
    ac_cos = (a_vecs * c_vecs).sum(axis=1)
    for i, c in enumerate(chains):
        c["sim_ac"] = float(ac_cos[i])
    disjoint = [c for c in chains if c["sim_ac"] < 0.3]
    print(f"Embedding-disjoint subset (cos<0.3): {len(disjoint)}/{len(chains)} chains")
    chains = disjoint if disjoint else chains

    random.seed(42)
    random.shuffle(chains)
    chains = chains[:N_CHAINS]

    # Build corpus
    corpus_texts, corpus_ids = build_corpus_from_chains(chains)
    print(f"Corpus: {len(corpus_texts)} unique documents")

    if not corpus_texts:
        print("No corpus documents found. Check chain data structure.")
        return

    # Embed corpus
    print("Embedding corpus...")
    corpus_vecs = np.array(embed_texts(corpus_texts, batch_size=64), dtype=np.float32)
    print(f"Embeddings: {corpus_vecs.shape}")

    all_results = {"model": OMLX_MODEL, "experiment": "E1-longcontext-baseline",
                   "conditions": []}

    for k in K_VALUES:
        print(f"\n--- k={k} ---")
        res = run_experiment(chains, corpus_texts, corpus_vecs, k)
        n = max(res["n"], 1)
        print(f"  RESULTS k={k} (n={res['n']}):")
        print(f"    dense_only:        {100*res['dense_only']/n:.1f}%")
        print(f"    bridge_injected:   {100*res['bridge_injected']/n:.1f}%")
        print(f"    terminal_injected: {100*res['terminal_injected']/n:.1f}%")
        print(f"    gap (A→B):         +{100*(res['bridge_injected']-res['dense_only'])/n:.1f}pp")
        all_results["conditions"].append({
            "k": k,
            "n": res["n"],
            "dense_only_pct":      round(100*res["dense_only"]/n, 1),
            "bridge_injected_pct": round(100*res["bridge_injected"]/n, 1),
            "terminal_injected_pct": round(100*res["terminal_injected"]/n, 1),
            "gap_pp": round(100*(res["bridge_injected"]-res["dense_only"])/n, 1),
        })

    OUT_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved -> {OUT_FILE}")

    print("\n=== SUMMARY ===")
    print("The gap (bridge_injected - dense_only) measures retrieval failure:")
    print("If gap is large: the model CAN reason once bridge is visible,")
    print("  but retrieval never gives it the bridge. This is a retrieval failure.")
    print("If gap is small: the model fails even with bridge — reasoning failure.")
    print("Prediction: large gap (~0% → >50%), proving retrieval failure, not reasoning.")


if __name__ == "__main__":
    main()
