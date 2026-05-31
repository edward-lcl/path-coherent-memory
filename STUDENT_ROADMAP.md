# Contributor Roadmap — Path-Coherent Memory Retrieval

Welcome. This project has a finished core result and a clear path to a conference
submission. The hard structural work is done; what remains is **breadth,
robustness, and presentation** — well-scoped tasks where you own a piece end to
end and your name is on the contribution.

## The one-paragraph thesis (read this first)

Agent memory systems — Mem0, Letta, Claude Projects, any persistent-context
layer — store conversation history across multiple agent threads. When threads
work on overlapping concerns, their discoveries *should* compound on retrieval.
In practice, cross-thread concept links are neither lexically (BM25=0%) nor
semanticaly (dense=0%) reachable — a structural blind spot, not a tuning gap.
Recovering these links requires **two complementary traversal modes**: a free,
read-free token-path prior plus iterative LLM reading. Their hit-sets are
near-disjoint (Jaccard 0.15); neither alone suffices; their union reaches 67%.
The contribution is a structural result about what retrieval over agentic/
conversational memory requires. See `README.md` and `FINDINGS.md` for the evidence.

## What's already locked (don't redo)

- The 3-way complementarity result on the private Talos corpus (path/iterative
  co-equal, dense=0%, union >> best-single).
- Public reproduction of dense's orthogonal failure on the MuSiQue
  embedding-disjoint tail (`musique_disjoint_eval.py`).
- The honest real-LLM self-ask baseline vs. token-path
  (`musique_real_iterative_eval.py`): real-iter 47.2%, path contributes 3.1%
  exclusive that the LLM loop never reaches.

## Workstreams (pick one to own)

Each is self-contained, has a clear "done" bar, and feeds a specific paper
section. Difficulty is marked. Start anywhere — they're independent.

### A. Scale the public reproduction  ·  difficulty: low-medium  ·  owns: §Results table
The disjoint-tail result is on n=159, one corpus size, one threshold. Make it a
*curve*, not a point.
- Sweep the embedding-disjoint threshold (cos < 0.2 / 0.25 / 0.3 / 0.35) and plot
  dense/path/real-iter recall vs. threshold. Hypothesis: dense stays ~0 across
  the whole disjoint band; path degrades gracefully.
- Sweep corpus size (1k / 4k / 8k / 16k bio distractors) and confirm the
  complementarity gap is stable, not a small-corpus artifact.
- Add a second public multi-hop set (2WikiMultiHopQA) filtered by the same
  objective criterion. Does the pattern hold on a different bridge distribution?
- **Done when:** a threshold-sweep figure + a corpus-size table + 2Wiki
  confirmation, all reproducible from one script.

### B. Strengthen the iterative baseline  ·  difficulty: medium  ·  owns: §Baselines
Reviewers will attack the iterative comparison. Make it bulletproof.
- Try 2-3 different LLMs in the loop (we have several on oMLX) and report bridge
  accuracy vs. terminal recall — does a better reader close the path gap?
- Add a real multi-iteration IRCoT loop (not just 1 hop-1 prediction) so the
  comparison matches the published self-ask method, not a simplification.
- Measure and report the actual cost: LLM calls per question, wall-clock,
  tokens. The "path is free" claim needs a real number next to it.
- **Done when:** a cost-vs-recovery table (path vs. each LLM-iter config) that
  makes the efficiency argument quantitative.

### C. Graph-RAG comparison  ·  difficulty: medium-high  ·  owns: §Related Work table
The #1 reviewer question is "why doesn't HippoRAG / GraphRAG solve this?"

**Read `HIPPOGRAPH_EXPERIMENT_BRIEF.md` first — the experiment design,
hypothesis, protocol, and deliverables are fully specified there.** Short version:
- Run HippoRAG on the same 159-chain MuSiQue embedding-disjoint eval we already have
- Report recall@10 and exclusive Jaccard vs. token-path and real-LLM-iter
- Our hypothesis: HippoRAG solves entity bridges (MuSiQue) but not concept bridges
  (Talos) — because concept links don't survive NER extraction
- **Decision point before starting:** HippoRAG defaults to OpenAI NER (~$5-20
  for the 8K-doc corpus). Check `HIPPOGRAPH_EXPERIMENT_BRIEF.md` §Decision Point
  for the local alternative (spaCy + local LLM, no API cost).
- **Done when:** one HippoRAG row in the main table + one paragraph in FINDINGS.md
  (Experiment 6) explaining why it does/doesn't close the gap.

### D. Figures, writing, reproducibility  ·  difficulty: low  ·  owns: §Method figures + repo
- Turn the complementarity result into a clean Venn / bar figure (path-exclusive,
  iter-exclusive, overlap, union) for both Talos and MuSiQue.
- Write the Method section walkthrough of token-path (the bridge-token traversal)
  with a worked example from the corpus.
- Add a `make reproduce` that runs the full public pipeline from scratch and
  regenerates every number in the paper.
- **Done when:** a reviewer can clone, run one command, and get our table.

## How to start (day 1)

1. Read `README.md` then `FINDINGS.md` (skim — it's a lab notebook, newest at the
   bottom is the current state).
2. Clone, set up the venv, and run `python musique_disjoint_eval.py` — confirm you
   reproduce dense=0.0% / path=20.8% on your machine. That's your "hello world."
3. Pick a workstream above, open an issue claiming it, and ship the smallest
   version first (one threshold, one model, one figure). Iterate from there.

## Ground rules

- Every claim is an experiment with a script that regenerates it. No hand-waved
  numbers — if it's in the paper, there's a `*_eval.py` that produces it.
- We red-team our own results (see the de-artifacting commit history). If a
  number looks too good, find the artifact before celebrating.
- Private corpora (Talos) stay private — they're gitignored. Public work
  (MuSiQue/HotpotQA/2Wiki) is what goes in the repo and the paper's reproducible
  core.
- "Personal memory" framing is deprecated — use "agentic/conversational memory"
  or "experiential memory" in any writing you do for this project.

---

## Additional Workstreams (added 2026-05-30)

### E. LOCOMO Benchmark  ·  difficulty: medium  ·  owns: §Baselines table
Reviewers familiar with multi-session memory will ask: "why not LOCOMO?" (Maharana et al., ACL 2024 — the standard multi-session conversational memory benchmark).

- Download LOCOMO dataset, extract multi-session QA pairs
- Filter to embedding-disjoint subset (same cos<0.3 criterion)
- Run 3-way: dense / token-path / oracle-iter
- Report recall@10 alongside Talos and MuSiQue in the main table
- Prediction: same pattern — concept-bridge links score ~0% on dense; path and oracle-iter recover different slices
- **Done when:** one LOCOMO row in the main results table + one paragraph in FINDINGS.md

### F. NoLiMa Alignment Check  ·  difficulty: low  ·  owns: 1 paragraph in related work
NoLiMa (2025) shows long-context models fail when the needle has *zero lexical overlap* with the query — which is our exact concept-bridge condition. Connecting our result to theirs strengthens both the E1 motivation and the related work.

- On the Talos embedding-disjoint subset (n=131): measure query↔terminal BM25 score (should be ~0) and unigram overlap
- Confirm these chains satisfy NoLiMa's "no lexical overlap" criterion
- Write one paragraph in FINDINGS.md connecting to NoLiMa
- **Done when:** a table or stat confirming lexical disjointness + related-work paragraph citing NoLiMa arXiv

### Key papers to read first (all workstreams)
- HippoRAG (NeurIPS 2024, arXiv 2405.14831) — main threat, understand it well
- HippoRAG 2 (ICML 2025, arXiv 2502.14802) — their own admission that graph-RAG degrades on factual memory
- NoLiMa (2025) — your ally; long-context fails on no-lexical-overlap conditions
- LOCOMO (ACL 2024) — the multi-session benchmark
