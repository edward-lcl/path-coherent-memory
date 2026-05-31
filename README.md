# Path-Coherent Memory Retrieval

**When multiple agent threads write to a shared memory corpus, retrieval cannot compound their discoveries — dense and lexical search score 0.0% on cross-thread concept links. Two orthogonal traversal modes each recover ~38%, with near-disjoint hit-sets.**

---

## The Problem

Agent memory systems — Mem0, Letta, Claude Projects, every persistent-context layer — store conversation history across sessions. When multiple agent threads work on overlapping concerns, their discoveries *should* compound: thread A figures out X, thread B should be able to pull on it later when it becomes relevant. In practice, retrieval fails this at the boundary where threads share concepts but not vocabulary.

The failure mode is structural, not a tuning problem. When the query anchor and the target memory share **zero vocabulary and are not adjacent in embedding space**, BM25 and dense retrieval both score exactly **0.0%** — not low, zero. This is not a gap a better model closes; it is a regime change.

Two mechanisms can see into that blind spot, and they see *different* parts of it:

```
embedding-disjoint chains (BM25 0%, dense 0%):  terminal-recall@10
   token-path     (read-free, LLM-free)            38%
   oracle-iter    (LLM reads the bridge doc)        38%
   UNION                                            67%   +29pp over best single
   path ∩ iter    Jaccard                           0.15  near-disjoint hit-sets
```

Reading is not strictly stronger — each mode recovers ~29% of chains the other misses entirely. An ensemble is required. This repository documents that structural result and provides the retrieval harness.

---

## Three Experimental Results

### Result 1 — Dense is orthogonally useless at the embedding-disjoint tail
On chains where start↔terminal cosine < 0.3 (objective criterion, not a threshold choice):
- BM25 = 0.0%, dense (Qwen3-0.6B) = 0.0%  
- MuSiQue public reproduction confirms this: 159/1252 2-hop questions fall in this tail — dense 0%, path 20.8%, oracle-iter 61.6%

**This half is public, reproducible, and reviewer-runnable.**

### Result 2 — Token-path adds a read-free, LLM-free exclusive slice
On the Talos corpus, token-path recovers ~38% of embedding-disjoint links with **29% exclusive** — links oracle-iter misses. On MuSiQue's disjoint tail, token-path recovers 20.8% with a 3.1% exclusive slice over real-LLM iterative.

The exclusive slice is small on MuSiQue because MuSiQue bridges are named entities (Wikipedia-style), not concepts — iterative dominates there (47-62%). On **concept bridges** (Talos/experiential memory), the modes are co-equal.

### Result 3 — Co-equality is experiential-memory-specific
On Talos: path=38%, oracle-iter=38%, Jaccard=0.15. Harder filtering (MuSiQue double-disjoint, n=331) did not reproduce co-equality (path=18.7%, oracle-iter=66.8%) — because MuSiQue has named-entity bridges by construction. No filtering creates concept bridges from entity-spine data.

**MuSiQue's non-reproduction is positive evidence**, not a weakness: it means experiential/conversational memory is a structurally distinct regime that entity-bridge benchmarks cannot represent. That is the argument for studying it separately.

---

## Corpus: What Makes This Regime Distinct

The Talos benchmark is drawn from an operational AI assistant corpus: session logs across multiple agent threads, topic files capturing project state, entity records. The bridges between sessions are **concept links** — shared projects, shared clients, shared decisions — not named entities that survive embedding.

This is not unique to Talos. The same structure appears in:
- Any multi-session agent runtime (Levi, Claude Projects, Mem0 users)
- Glasstone-style memory: multiple specialized agents writing to a shared episodic substrate
- Personal knowledge graphs where relationships are semantic, not entity-spine

The entity-bridge benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA) model *encyclopedia-style* multi-hop, where bridges are named entities that remain embedding-reachable. Experiential memory models *conversational-style* multi-hop, where bridges are implicit concepts that do not.

---

## Benchmarks

### Talos Operational Corpus (private, concept bridges)

| Method | ALL chains (n=200) | HARD cos<0.3 (n=131) |
|---|---|---|
| BM25 | 5.0% | **0.0%** |
| Dense (Qwen3-0.6B) | 13.5% | **0.0%** |
| BM25 + Dense | 11.0% | **0.0%** |
| Token-path (read-free) | 29.5% | **38.3%** |
| Oracle-iterative (LLM) | 50.5% | **38.3%** |
| **Union (path ∪ oracle-iter)** | **70.5%** | **67.0%** |

Path and oracle-iter Jaccard on hard subset: **0.15** — near-disjoint hit-sets.

### MuSiQue Public Reproduction (public, entity bridges)

2-hop questions, 8,180-doc corpus, embedding-disjoint tail (cos<0.3, n=159):

| Method | Recall@10 |
|---|---|
| Dense | 0.0% |
| Token-path | 20.8% |
| Oracle-iter (ceiling) | 61.6% |
| Real-LLM iterative (Qwen3-4B) | 47.2% |
| **Path + Real-iter UNION** | **50.3%** |
| Exclusive path | 3.1% |
| Exclusive real-iter | 29.6% |

MuSiQue is entity-bridge: oracle-iter dominates, path is junior partner. Talos is concept-bridge: both co-equal. This difference is the paper's central claim about regime distinction.

---

## The Token-Path Algorithm

Follows locally-recurring bridge tokens across document boundaries. Bridge entities are tokens with df 2–5 in the local neighborhood — specific enough to be meaningful, common enough to serve as bridges.

```
Query → BM25 anchor → {bridge token ∈ [df=2..5]} → intermediate → {bridge token} → terminal
```

Read-free and LLM-free. Not a SOTA retriever — a cheap structural prior that captures what dense structurally cannot.

---

## Reproducing the Results

### Prerequisites
```bash
pip install numpy scikit-learn requests
# For embedding: mlx-embeddings or any sentence-transformers model
```

### MuSiQue (public, reproducible)
```bash
# Downloads MuSiQue dev set (~1.2K 2-hop questions)
python3 musique_disjoint_eval.py      # dense+path+oracle-iter on cos<0.3 tail

# Real-LLM iterative baseline (replaces oracle ceiling)
# Requires oMLX endpoint or OpenAI-compatible API
OMLX_API_KEY=<key> python3 musique_real_iterative_eval.py
```

### Talos / private corpus
Requires access to the Talos corpus. Chain mining and evaluation harnesses are in `talos_benchmark.py` and `talos_complementarity_eval.py`. See `FINDINGS.md` for mining parameters.

---

## What's Open

**For reviewers / graph-RAG comparison (open):**
Does HippoRAG or GraphRAG solve the concept-bridge failure mode? Our hypothesis: graph-RAG approaches extract named entities and build entity graphs — they should solve entity-bridge multi-hop (MuSiQue regime) but should fail on concept bridges (Talos regime) because concept links don't survive NER extraction. This experiment is in design; see `STUDENT_ROADMAP.md` for the planned eval.

**For the ensemble:**
Token-path and oracle-iter together cover 67% of the hard subset. What covers the remaining 33%? Is there a third orthogonal traversal mode, or is the residual structurally unreachable?

**For corpus generalization:**
Talos is one experiential-memory corpus. Do concept bridges appear in other multi-session agent runtimes? The agent session corpus experiment (N=67 summarized sessions) showed oracle-iter at 25-33% on cross-session concept links — confirming the bridges exist, but corpus too small for token-path validation (need N>~100 vocabulary-rich nodes).

---

## Repository Structure

```
path-coherent-memory/
├── FINDINGS.md                       # Full research log + experiment audit trail
├── STUDENT_ROADMAP.md                # Defined tasks for student collaborators
│
├── path_coherent_retriever.py        # Token-path implementation
├── embedding_bridge_retriever.py     # Embedding-based retrieval harness
│
├── musique_disjoint_eval.py          # MuSiQue public reproduction (run this)
├── musique_real_iterative_eval.py    # Real-LLM iterative baseline
├── musique_double_disjoint_eval.py   # Harder filter (cos<0.35)
│
├── talos_benchmark.py                # Talos 3-way harness
├── talos_complementarity_eval.py     # Jaccard + exclusive-slice analysis
├── talos_clean_eval.py               # De-artifacted eval
│
├── agent_session_miner.py            # Agent runtime session chain miner
├── agent_session_summarizer.py       # LLM episode summarizer
├── agent_session_complementarity.py  # 3-way eval on session corpus
│
├── run_omlx_semantic_judge.py        # LLM chain judge
├── hotpotqa_hybrid_eval.py           # HotpotQA (baseline, path adds ~0 over dense)
└── wikipedia_loader.py               # Bio-distractor corpus builder
```

---

## Citation

```bibtex
@misc{luecheelip2026pathcoherent,
  title={Path-Coherent Retrieval for Agentic Memory:
         Concept Bridges Require Orthogonal Traversal Modes Where Dense Scores 0\%},
  author={Lue Chee Lip, Edward},
  year={2026},
  note={Preprint. \url{https://github.com/edward-lcl/path-coherent-memory}}
}
```

---

## Status

- [x] Token-path — benchmarked on Talos, deployed in production
- [x] Oracle-iterative + real-LLM iterative — benchmarked on Talos + MuSiQue  
- [x] MuSiQue public reproduction — dense 0%, path 20.8%, real-iter 47.2%
- [x] MuSiQue double-disjoint — confirms regime distinction (harder filter, same pattern)
- [x] Agent session corpus experiment — concept bridges exist, corpus-size constraint on path
- [x] De-artifacted Talos benchmark — honest numbers, audit trail in FINDINGS.md
- [ ] HippoRAG / graph-RAG comparison — in design (see STUDENT_ROADMAP.md)
- [ ] Talos robustness cuts (different k, different embedding model)
- [ ] arxiv preprint
