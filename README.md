# Path-Coherent Memory Retrieval

**Personal-memory multi-hop retrieval requires multiple _orthogonal_ traversal modes. No single retrieval method suffices.**

On personal-memory links that are neither lexically nor semantically adjacent
(BM25 0%, dense 0% by an objective embedding criterion), corpus-topology
traversal and iterative LLM-reading each recover ~38% of links — but their
hit-sets are nearly disjoint (Jaccard 0.15), and their union reaches 67%. Dense
retrieval is _orthogonally useless_ in this regime. The contribution of this
repository is not a state-of-the-art retriever; it is a structural result about
what retrieval over personal memory actually requires, plus a cheap, read-free,
LLM-free traversal mode (token-path) that is a necessary member of the ensemble.

> **Note (2026-05-30):** An earlier version of this README led with a
> "72.7% vs 0.0%" headline. That number was inflated by three artifacts —
> (1) the benchmark chains were mined with the same token-bridge rule the
> retriever follows (circular), (2) the LLM judge labeled polysemy collisions as
> genuine, and (3) the query was a single rare token rather than a realistic
> document. After de-artifacting (realistic queries, an objective
> embedding-disjoint subset, judge bypassed), the honest numbers below are
> roughly half the original claim — and the surviving result is, in our view,
> stronger because it is defensible. See `FINDINGS.md` for the full red-team.

---

## The Core Finding (de-artifacted)

Personal memory contains multi-hop chains where the terminal node shares **zero
vocabulary** with the query anchor **and** is not adjacent to it in embedding
space. On a subset defined by an objective embedding criterion (start↔terminal
cosine < 0.3), BM25 and dense both score exactly 0.0% terminal recall. This is
not a tuning gap; it is a structural blind spot of lexical and dense retrieval.

Two mechanisms can see into that blind spot, and they see **different** parts of
it:

```
embedding-disjoint hard subset (BM25 0%, dense 0%), terminal-recall@10:
   token-path (read-free, LLM-free)   20.0%
   oracle-iterative (LLM reading)      38.3%
   UNION                               66.7%   ← +28pp over best single mode
   path & iterative Jaccard            0.15    ← nearly disjoint hit-sets
```

Reading is the stronger single mode, but it does **not** dominate: ~29% of these
hard links are recovered _only_ by token-path and missed entirely by reading.
The modes are complementary, not redundant.

---

## Three Retrieval Modes

### Mode 1 — Token-Path (Lexical Bridges)
Follows locally-recurring bridge tokens across document boundaries. Bridge entities are tokens with df 2–5 — common enough to appear in multiple sources, rare enough to be specific.

```
Query → BM25 anchor → [bridge token] → intermediate → [bridge token] → terminal
```

**Result on Talos corpus (de-artifacted):** 20.0% terminal hit@10 on the
embedding-disjoint hard subset vs. BM25 0.0% / dense 0.0% (29.5% over all
chains). Read-free and LLM-free — a cheap structural prior, not a SOTA retriever.

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
- 200 mined chains; analyzed with realistic full-document queries
- Hard subset = objective embedding-disjoint split (start↔terminal cos < 0.3)

**De-artifacted terminal-recall@10** (`talos_clean_eval.py`,
`talos_complementarity_eval.py`, all 200 chains, judge bypassed):

| Method | ALL chains | HARD (embedding-disjoint) |
|---|---|---|
| BM25 | 5.0% | **0.0%** |
| Dense (Qwen3-0.6B) | 13.5% | **0.0%** |
| BM25 + Dense | 11.0% | **0.0%** |
| Token-path (read-free) | 29.5% | **20.0%** |
| Oracle-iterative (LLM reading) | 50.5% | **38.3%** |
| **Union (path ∪ dense ∪ iterative)** | **70.5%** | **66.7%** |

On the hard subset, path and iterative each recover ~38% but with Jaccard 0.15
(near-disjoint), each exclusively recovering ~29% the other misses. Dense scores
0.0% and Jaccard 0.00 with every mode — orthogonally useless where lexical and
semantic signal both vanish.

> The original `72.7% / 0.0%` table (token-path v12) is retained in git history
> and `FINDINGS.md`. It used a bare-token query, a chain set mined by the
> retriever's own rule, and a permissive judge; it is superseded by the
> de-artifacted numbers above.

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

---

## Direction (2026-05-30): Public Reproducibility & the Vocabulary-Gap Spectrum

The Talos/Levi benchmarks above are private corpora (self-mined, self-judged). A
reviewer cannot run them and cannot rule out a mining artifact. The current
research thrust is to reproduce the failure mode on **public, pip-installable**
datasets and to position the contribution against a **real dense retriever**
(not just BM25).

### Key correction: BM25 is the wrong baseline to beat

On standard multi-hop (HotpotQA), an off-the-shelf dense retriever
(Qwen3-Embedding-0.6B) beats BM25 outright and path-coherence adds **nothing**
on top of dense — because HotpotQA's bridge entity is usually present in the
question, so embeddings already connect the hops. Any "win over BM25" there is
illusory. The honest baseline is BM25 ⊕ dense.

### The vocabulary-gap spectrum (the paper's real spine)

The contribution is not "we beat dense on multi-hop." It is that there is a
**failure-mode spectrum** indexed by how hidden the bridge entity is, and that
topology adds value monotonically as the gap widens:

| Regime | Dataset | Bridge entity | Dense alone | Path adds over dense |
|---|---|---|---|---|
| Standard multi-hop | HotpotQA | usually in question | strong (~90%) | ~0 (dead weight) |
| Compositional multi-hop | **MuSiQue** (public) | often hidden | wins | ~0 (does NOT transfer) |
| Personal memory chains | Talos (private) | zero shared vocab | 0.0% | the only thing that works |

MuSiQue is the bridge between the private headline and a result a reviewer can
run: it composes single-hop questions, so the linking entity is frequently
absent from the query text ("Who is the spouse of the Green performer?" — the
performer is never named). This is the public analogue of the Talos zero-vocab
condition.

**Status (honest, NEGATIVE RESULT):** path-coherence does NOT transfer to
MuSiQue. Hop-stratified per-support recall@10 (200 each of 2/3/4-hop) shows
dense alone winning at every depth and path adding nothing on top of dense
(bm25+dense+path ≤ dense at 2-, 3-, and 4-hop). The n=60 smoke's +13.3pp was
small-sample noise. The likely reason: MuSiQue's bridges are Wikipedia-style and
remain embedding-reachable across hops (dense gets 58% at 2-hop), so it is NOT
the public analogue of the Talos zero-vocab condition we hoped for. The
generalization claim for a single token-path mode did not survive contact with a
real dense baseline on public data. The defensible spine is now **complementarity**
(token-path ⊕ embedding-bridge retrieve non-overlapping families), not
single-method generalization. Finding a public corpus with genuinely
embedding-disjoint bridges is the open problem.

### Reproducing the public results

```bash
# Build the bio-distractor corpus (one-time, ~30K Wikipedia biographies)
python3 wikipedia_loader.py --max 30000

# HotpotQA large-shared-corpus with proper BM25 + dense + path fusion
python3 hotpotqa_dense_eval.py --n 300 --bios 30000 --k 10

# MuSiQue — the vocabulary-gap test (gold = paragraph_support_idx)
python3 musique_eval.py --n 800 --k 10        # breaks out by hop count 2/3/4
```

Public harnesses: `hotpotqa_hybrid_eval.py` (proper Okapi BM25 + weighted/RRF
fusion + bridge/comparison routing), `hotpotqa_dense_eval.py` (adds in-process
Qwen3 dense retrieval), `musique_eval.py` (the failure-mode reproduction).
Embeddings run in-process via `embedding_bridge_retriever.embed_texts` and are
checkpointed to `.npy` every 2K docs.

### Open threads

- Does path's lift over dense **grow** at 3- and 4-hop MuSiQue? (gap compounds
  with depth — this is the figure that sells the paper)
- Complementarity as the spine: quantify union(BM25, dense, token-path,
  embedding-bridge) vs. best single mode — "personal memory retrieval needs ≥2
  orthogonal traversal modes."
- 2WikiMultiHopQA as a second public failure-mode corpus.
