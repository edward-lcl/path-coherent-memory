# Path-Coherent Memory Retrieval

**Personal AI memory is not a retrieval problem over a flat corpus. It is a graph traversal problem over a latent knowledge graph.**

BM25 achieves 0.0% terminal recovery on personal AI memory chains. Dense retrieval achieves 0.0%. Recent graph-augmented methods (HippoRAG, GraphRAG, RAPTOR) do not address this failure mode. This repository introduces three complementary retrieval modes that do.

---

## The Core Finding

Personal AI memory contains multi-hop chains where the terminal node shares **zero vocabulary** with the query anchor and is not adjacent to it in any embedding space. This is not a performance difference — it is a structural failure. No amount of parameter tuning fixes BM25 or dense retrieval on this problem.

```
Query: "ultrapure inductors enrolment"
   ↓ BM25: 0.0% terminal recovery
   ↓ Dense: 0.0% terminal recovery
   ↓ Token-path: 72.7% terminal recovery  ← this paper
```

The gap is structural: token-path follows bridge entities across source boundaries; BM25 requires shared vocabulary that the benchmark chains deliberately exclude.

---

## Three Retrieval Modes

### Mode 1 — Token-Path (Lexical Bridges)
Follows locally-recurring bridge tokens across document boundaries. Bridge entities are tokens with df 2–5 — common enough to appear in multiple sources, rare enough to be specific.

```
Query → BM25 anchor → [bridge token] → intermediate → [bridge token] → terminal
```

**Result on Talos corpus:** 72.7% terminal hit@10 vs. BM25 0.0%

### Mode 2 — Embedding-Bridge (Semantic Gaps)
Traverses semantic proximity gaps across document-type boundaries (topic files ↔ session logs). Finds chains that are semantically connected but lexically disjoint.

**Result on Levi corpus:** 8–33% by chain type vs. BM25/token-path 0.0%

Zero overlap with Mode 1. They retrieve non-overlapping chain families — a structural partition, not a performance difference.

### Mode 3 — Relationship-Walk (Entity Graph)
Traverses the typed entity graph that structured personal-AI substrates maintain. Anchors on entity nodes via search, walks typed relationship edges (for_client, involves_supplier, employed_by), surfaces path-traced results.

```
"WASA water tender"
  → anchors on org-email-wasa (sim=0.999)
  → walks for_client → CTL
  → walks for_client → Heritage, WINDALCO, NGC
  → returns entity network in 6ms
```

Token-path returns none of this. Mode 3 covers what the other two structurally cannot.

---

## Architecture

```
                        Query
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    Token-path      Embedding-bridge   Relationship-walk
    (lexical        (semantic gaps,    (entity graph,
    bridges,        cross-type)        typed edges)
    zero-vocab)
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                   Merge + Re-rank
                          │
                     Final result
                     (with path trace)

Architecture prerequisite:
A structured ontology with typed relationships enables Mode 3.
Without the ontology, the relationship graph doesn't exist to traverse.
The algorithms are a consequence of the architecture, not a replacement for it.
```

---

## Benchmarks

### Talos Operational Knowledge Base (Benchmark Family 1 — Lexical Gaps)
- 19,422 notes, 3,811 organizations, 37,000+ typed relationship edges
- 200 LLM-judged chains (Gemma-4-E4B-it), post-dedup
- 132 real_semantic, 63 weak_semantic, 5 artifact

| Method | Terminal@10 (real) | Full-path@10 |
|---|---|---|
| BM25 | 0.0% | 0.0% |
| Cosine | 0.0% | 0.0% |
| **Token-path v12** | **72.7%** | **72.7%** |

### Levi Personal Memory Corpus (Benchmark Family 2 — Semantic Gaps)
- 10,153 notes, personal session logs + topic files
- 70 real_semantic chains (hub-excluded, full-corpus rank-verified)

| Chain type | BM25 | Token-path | Embedding-bridge |
|---|---|---|---|
| session→session | 0.0% | 0.0% | 0.0% |
| session→topic | 0.0% | 0.0% | 8.3% |
| topic→session | 0.0% | 0.0% | 10.0% |
| topic→topic | 0.0% | 0.0% | 33.3% |

### Held-Out Validation
100 fresh Talos chains (excluding frozen benchmark), independently judged:

| Method | Terminal@10 (real, n=28) |
|---|---|
| BM25 | 0.0% |
| Token-path min=7 (baseline) | 50.0% |
| **Token-path min=5 (deployed)** | **53.6%** |

---

## Repository Structure

```
path-coherent-memory/
├── paper_draft_v1.md           # Full paper (1011 lines)
├── FINDINGS.md                 # Complete research log
│
├── path_coherent_retriever.py  # Mode 1: token-path (production)
├── embedding_bridge_retriever.py # Mode 2: embedding-bridge
│
├── mine_candidates_v2.py       # Levi chain miner
├── remine_talos_chains_heldout.py  # Held-out Talos chain miner
├── mine_email_chains.py        # Email corpus chain miner (new)
│
├── semantic_chain_miner_v4.py  # Hub-excluded semantic miner
├── semantic_chain_miner_v5_entity.py  # Entity-anchored miner
│
├── run_omlx_semantic_judge.py  # LLM judge (Gemma-4-E4B-it)
├── run_talos_omlx_judge.py     # Talos-specific judge runner
├── semantic_chain_judge_prompt.md  # Judge prompt (frozen)
├── semantic_chain_judge_eval.py    # Judge calibration eval
│
├── run_gated_benchmark.py      # (on Talos) Benchmark harness
├── learned_reranker_v2.py      # Terminal re-ranker (experimental)
│
├── calibration_set_mine.py     # Parameter calibration miner
├── miner_judge_roadmap.md      # Research roadmap
└── algoverse_main_conference_brief.md  # Algoverse submission brief
```

---

## Reproducing the Results

### Prerequisites
- Python 3.10+
- A personal AI memory corpus (see below)
- Access to a local LLM for judging (or OpenAI API)

### Corpus options

**Option 1: Build your own substrate**
Any structured personal memory corpus works: notes apps (Obsidian, Notion exports), email archives, session logs.

**Option 2: Personal email corpus**
Export your Gmail via Google Takeout or use Apple Mail's local `.emlx` files:

```bash
# Auto-discover Apple Mail corpus
python3 mine_email_chains.py

# Or point at a specific mailbox
python3 mine_email_chains.py --mail-dir ~/Library/Mail/V10/account/All\ Mail.mbox

# Or standard mbox format
python3 mine_email_chains.py --mbox ~/Downloads/mail-export.mbox
```

### Running the benchmark

```bash
# 1. Mine chains
python3 mine_candidates_v2.py   # or mine_email_chains.py for email corpus

# 2. Judge with LLM
python3 run_omlx_semantic_judge.py --input <candidates.jsonl> --out <judged.jsonl>

# 3. Run retrieval benchmark
# (on Talos) python3 run_gated_benchmark.py
```

---

## The Architecture Prerequisite

The paper makes a claim beyond retrieval algorithms: **the quality of multi-hop retrieval is bounded by the structure of the memory representation.**

A flat document corpus limits retrieval to token overlap and semantic similarity. A typed knowledge graph enables relationship-walk traversal that flat retrieval structurally cannot perform. The ontology is the prerequisite; the algorithms follow from it.

This positions personal AI memory as an infrastructure problem, not an algorithm problem. Enterprise systems (Palantir Gotham, Neo4j) understood this at scale. This work demonstrates it at personal-AI scale.

---

## Citation

```bibtex
@misc{luecheelip2026pathcoherent,
  title={Path-Coherent Topology for Personal AI Memory Retrieval:
         Three Structurally Distinct Traversal Modes Where BM25 Achieves 0\%},
  author={Lue Chee Lip, Edward},
  year={2026},
  note={Preprint. \url{https://github.com/edward-lcl/path-coherent-memory}}
}
```

---

## Status

- [x] Token-path retrieval (Mode 1) — production deployed on Talos
- [x] Embedding-bridge retrieval (Mode 2) — benchmarked on Levi corpus
- [x] Relationship-walk retrieval (Mode 3) — live on Talos port 8401
- [x] Self-maintaining substrate monitor — autonomous edge inference running
- [x] Held-out validation — 100 chains, BM25 0%, tok-path 53.6%
- [ ] Email corpus benchmark (Benchmark Family 3) — in progress
- [ ] HippoRAG baseline comparison — planned
- [ ] Public arxiv preprint — planned
