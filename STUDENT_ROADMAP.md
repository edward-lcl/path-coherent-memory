# Contributor Roadmap — Path-Coherent Memory Retrieval

Welcome. This project has a finished core result and a clear path to a conference
submission. The hard structural work is done; what remains is **breadth,
robustness, and presentation** — well-scoped tasks where you own a piece end to
end and your name is on the contribution.

## The one-paragraph thesis (read this first)

Personal-memory multi-hop retrieval has links that are neither lexically
(BM25=0%) nor semantically (dense=0%) reachable. We show dense retrieval is
*orthogonally useless* in this regime, and that recovering these links requires
**multiple complementary traversal modes** — a free, read-free corpus-topology
prior (token-path) plus iterative LLM reading. Neither alone suffices; their
union does. The contribution is a *structural result about what retrieval over
personal memory requires*, plus a cheap traversal mode that is a necessary
ensemble member. See `README.md` and `FINDINGS.md` for the evidence.

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
- Stand up at least ONE graph-RAG method (HippoRAG is the closest analogue) and
  run it on the disjoint tail.
- Report where it lands relative to dense/path/iter. Our hypothesis: it helps but
  still misses the concept-bridge links, because its graph is entity-centric.
- **Done when:** one graph-RAG row exists in the main table with a one-paragraph
  analysis of why it does/doesn't close the gap.

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
