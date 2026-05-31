# Research Direction — Path-Coherent Retrieval for Agentic Memory

_Last updated: 2026-05-30_

## The Bet

This paper's claim is architectural, not algorithmic. The argument:

> **Retrieval quality is bounded by the expressivity of the memory schema.**
> Flat text → only lexical + dense → concept bridges score 0%.
> Typed ontology → relationship-walk → recovers what neither can reach.
> The paper measures that gap empirically and shows it is structural, not tunable.

If this holds across multiple independent corpora and survives the graph-RAG
objection, it is not just a retrieval paper — it is an argument for why
ontology-centric AI memory architectures are necessary. That is the highest-level
version of this work and the target.

---

## What's Locked (Do Not Redo)

| # | Result | Evidence | Status |
|---|--------|----------|--------|
| 1 | Dense = 0% on concept-bridge embedding-disjoint tail | Talos (n=131), MuSiQue (n=159, n=331) | ✅ |
| 2 | Token-path recovers ~38% on Talos concept bridges, path-exclusive ~29% | talos_complementarity_eval.py | ✅ |
| 3 | Path + oracle-iter co-equal on concept bridges (Jaccard 0.15) | Talos cos<0.3 subset | ✅ |
| 4 | MuSiQue entity bridges: oracle-iter dominates, path junior (20.8% vs 47.2%) | musique_real_iterative_eval.py | ✅ |
| 5 | MuSiQue non-reproduction = positive evidence for regime distinction | musique_double_disjoint_eval.py | ✅ |
| 6 | Real-LLM iterative replaces oracle ceiling | Qwen3-4B, bridge acc 73.6% | ✅ |
| 7 | Agent session corpus: oracle-iter 25-33%, path=0% (corpus-size constraint N=29) | agent_session_complementarity.py | ✅ |

---

## Open Experiments — Full List

### Tier 1 — Biggest Fish (set research direction, hardest to refute)

These are the experiments that turn this from "interesting" into "undeniable."
Edward should own the design and interpret the results; execution can be shared.

---

**E1 — Long-Context Baseline**
_Kills the frontier-lab "just use long context" objection._

The objection: "GPT-4o / Gemini / Claude with 100K+ context doesn't need retrieval — shove everything in the window."
The counter: retrieval failure happens at candidate selection. If the bridge document isn't in the top-50 candidates, no context window length saves you.

**Design:**
- On the Talos embedding-disjoint subset (n=131): retrieve top-50 by dense, feed to a 100K+ context model with the question, ask it to identify the terminal document
- Measure: does it find the terminal when the bridge is absent from the candidate set?
- Prediction: ~0% when bridge excluded; >50% when bridge included (oracle condition)
- This proves: it's a retrieval failure, not a reasoning failure

**Deliverable:** one table row "long-context (dense top-50)" = ~0% on hard subset.
**Lane:** Edward — needs interpretation, shapes the whole paper's framing.
**Effort:** 1 day. Uses existing chains. Needs a 100K+ context model call (oMLX or API).

---

**E2 — HippoRAG / Graph-RAG Comparison**
_Kills the "just use graph-RAG" objection._

See `HIPPOGRAPH_EXPERIMENT_BRIEF.md` for full design.

Short version: HippoRAG builds entity graphs via NER + PPR traversal. Our hypothesis: it solves MuSiQue entity bridges (where it should outperform path) but fails on Talos concept bridges (where NER extraction misses the links). If confirmed, the paper gains a clean typology: **three methods, three regimes.**

**Lane:** Student (workstream C in STUDENT_ROADMAP.md).
**Effort:** 1-2 days setup + compute. Decision point: local NER (spaCy) vs OpenAI NER (~$5-20).

---

**E3 — Second Public Experiential-Memory Corpus (Email)**
_Eliminates "one-corpus artifact" objection. Makes Talos result corpus-agnostic._

`mine_email_chains.py` already exists. Email archives are the most common form of
personal/agentic memory outside of Talos. If the same pattern (dense=0%, path>0%,
oracle-iter>0%) holds on a Gmail/mbox export with no special structure, it's a
structural finding, not a Talos property.

**Design:**
- Export: Google Takeout (any user's Gmail) or Apple Mail mbox
- Run mine_email_chains.py → get embedding-disjoint chains
- Run 3-way: dense / path / oracle-iter
- Report alongside Talos and MuSiQue in the main table

**Lane:** Student (most mechanical) or Edward if you want to run on your own archive fast.
**Effort:** 2-3 days. Self-contained, existing harness.

---

**E4 — Ontology Architecture Section**
_Elevates the paper from retrieval technique to architecture argument._

One section (not an experiment, but the most important writing task):

> "The retrieval ceiling is architectural. A flat corpus caps retrieval at token
> overlap + semantic proximity. A typed knowledge graph enables traversal modes
> that are structurally impossible over flat text. This paper measures that gap
> empirically. The implication for system design: memory schema expressivity
> directly determines multi-hop retrieval capability."

Connect to: Palantir Gotham (enterprise scale), Neo4j knowledge graphs, Talos
(personal/agentic scale). The claim is that the ontology IS the retrieval
infrastructure — not a layer on top of it.

**Lane:** Edward — this is the thesis, not a student task.
**Effort:** 1-2 hours of focused writing. Goes in README "Architecture Argument" section + paper.

---

### Tier 2 — Strengthen the Core (student-runnable, clear specs)

**E5 — Talos Robustness Cuts**
Verify 38/38 co-equality holds across parameters.
- Sweep k=5, 10, 20 — does the recall ratio hold?
- Swap embedding model (e.g., Qwen3-0.6B → a different family)
- **Done when:** a 3×3 table (k × model) showing co-equality is stable.
- **Lane:** Student (workstream A extension). Effort: 1 day.

**E6 — Scale the Public Reproduction**
Turn MuSiQue n=159 from a point to a curve.
- Threshold sweep: cos < 0.2 / 0.25 / 0.3 / 0.35 — dense stays ~0 across band?
- Corpus size sweep: 1K / 4K / 8K / 16K docs — gap stable at smaller corpus?
- Add 2WikiMultiHopQA as second public entity-bridge benchmark
- **Done when:** threshold-sweep figure + corpus-size table + 2Wiki confirmation.
- **Lane:** Student (workstream A). Effort: 2-3 days.

**E7 — Stronger Iterative Baseline**
Make the iterative comparison bulletproof for reviewers.
- Try 2-3 LLMs in the loop (oMLX has several: Qwen3-27B, Dolphin, Gemma)
- Add real IRCoT multi-iteration loop (not just 1 hop)
- Measure and report cost: LLM calls per question, tokens, wall-clock
- **Done when:** cost-vs-recovery table makes "path is free" argument quantitative.
- **Lane:** Student (workstream B). Effort: 2-3 days.

---

### Tier 3 — Figures, Writing, Reproducibility (student, lower priority)

**E8 — Complementarity Venn/Bar Figures**
Clean visual: path-exclusive, iter-exclusive, overlap, union — for both Talos and MuSiQue.
One figure that tells the whole story at a glance.

**E9 — `make reproduce` Pipeline**
Single command that downloads MuSiQue, runs dense+path+iter, and regenerates the
table. Reviewers should be able to clone and reproduce in under 30 minutes.

**E10 — Method Section Walkthrough**
Worked example of token-path traversal from the corpus — show a real chain,
the bridge tokens, the traversal path. Makes the method intuitive.

---

## Priority Order for Edward's Attention

```
1. E1 — Long-context baseline        ← biggest single experiment, 1 day, kills frontier objection
2. E4 — Architecture section writing ← the thesis, sets paper's ceiling
3. E2 — HippoRAG (design locked,     ← supervise student, interpret result
         student runs it)
4. E3 — Email corpus                 ← second public corpus, corpus-agnostic claim
```

Students own E5–E10 from STUDENT_ROADMAP.md with the workstream specs there.

---

## Target Venue

**Near-term:** NeurIPS 2026 workshop on agent memory / LLM agents — gets feedback fast,
establishes priority, workshop track is appropriate for a result this specific.

**Main conference target:** NeurIPS or ICLR 2027 — needs E1 + E2 + E3 completed,
architecture section written, and robustness cuts (E5) done. The story needs to be:
"dense=0% on concept bridges, graph-RAG doesn't solve it, long-context doesn't solve
it, our ensemble does, and here's what that means for how you should build AI memory."

That's a complete, hard-to-dismiss paper. Everything above is building toward it.

---

## The Undeniable Version

The paper is undeniable when it has:
- [ ] Dense=0% on concept bridges: **public, reproducible** (MuSiQue, done) + email corpus
- [ ] Graph-RAG doesn't solve it: HippoRAG comparison (E2)
- [ ] Long context doesn't solve it: long-context baseline (E1)
- [ ] Two orthogonal modes are necessary: Talos complementarity (done) + robustness cuts (E5)
- [ ] Architecture argument explicit: ontology section (E4)
- [ ] One-command reproduce: E9

Five checkboxes. Two are Edward-lane. Three are student-lane.

---

## Grok Frontier Analysis — Key Updates (2026-05-30)

### Papers to cite (newly identified)

**Allies:**
- **NoLiMa (2025)** — long-context models collapse when needle has *no lexical overlap* with query. Directly confirms the concept-bridge failure mode on the reading side. Cite prominently in E1 and E4. This is your strongest external ally.
- **HippoRAG 2 (ICML 2025, arXiv 2502.14802)** — the authors themselves admit graph-RAG *"drops considerably below standard RAG on basic factual memory."* Cite this against reviewers who claim HippoRAG solves everything. Frame HippoRAG as a *data point in your argument* (it had to construct a graph = evidence for schema-expressivity thesis), not a competitor.

**Threats to neutralize:**
- **HippoRAG (NeurIPS 2024, arXiv 2405.14831)** — PPR over extracted KG, beats IRCoT 10–30×. Top dismissal risk. Counter: only works on entity-string bridges; concept bridges produce no KG edges.
- **Mem0g** — Mem0's graph variant adds only ~2% over flat Mem0. Reviewers will cite this against "graphs are necessary." Counter: Mem0g is entity-graph over flat text — same structural limitation, no concept-bridge edges possible.
- **GraphRAG (Microsoft 2024, arXiv 2404.16130)** — community-detection for global summarization, not session-pair traversal. Different regime.

**Benchmark gap:**
- **LOCOMO (ACL 2024)** — the multi-session conversational memory benchmark. Reviewers will ask why you didn't run on it. Add to student roadmap as E11. Not fatal but worth addressing.

### Reframes to work into writing

1. **"Eval substrate blind spot"** — every standard benchmark (MuSiQue, HotpotQA, 2Wiki, LOCOMO) uses entity bridges by construction. HippoRAG wins on those because entity strings survive KG extraction. Our MuSiQue non-reproduction is *empirical proof the field's eval suite has a blind spot* — not a weakness. Lead with this in the abstract.

2. **Orthogonality over leaderboard delta** — the field reports deltas; we report set-overlap (Jaccard 0.15, UNION=67% vs best-single=38%). This is a structurally different and harder-to-dismiss claim: modes are orthogonal bases, not a quality continuum.

3. **Impossibility framing (E4)** — "the relation needed to walk the concept bridge does not exist as a retrievable object in a flat store — no retriever over flat text can recover it regardless of model scale." This is what HippoRAG implicitly proved by having to construct a graph. State it explicitly.

### Updated experiment list additions

**E11 — LOCOMO Benchmark (student, medium)**
Run 3-way (dense/path/oracle-iter) on LOCOMO multi-session eval. Preempts the
"why not LOCOMO?" reviewer question. Prediction: similar pattern — concept-bridge
links score ~0% on dense, path and oracle-iter split the recovery.

**E12 — NoLiMa-aligned analysis**
On the embedding-disjoint Talos subset, verify that query↔terminal lexical
overlap is zero (consistent with NoLiMa's "no lexical overlap" condition). This
connects our result to NoLiMa's framework and lets us cite it as corroboration.
One script, one paragraph in FINDINGS.md. 30 min effort.
