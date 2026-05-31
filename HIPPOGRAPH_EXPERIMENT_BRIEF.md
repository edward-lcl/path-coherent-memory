# Experiment Brief: HippoRAG Comparison

**Status:** In design — assigned to student collaborators  
**Priority:** High — this is the most common reviewer question  
**Estimated effort:** 1-2 days of compute + 2-3 days analysis

---

## The Question

HippoRAG (Gutiérrez et al., 2024) builds a knowledge graph from retrieved passages using named entity extraction and Personalized PageRank traversal. It targets multi-hop retrieval and outperforms dense baselines on HotpotQA and MuSiQue.

**Our hypothesis:** HippoRAG solves entity-bridge multi-hop (same regime as MuSiQue — named entities survive NER extraction and graph edges) but does **not** solve concept-bridge multi-hop (our experiential-memory regime — bridges are implicit concepts that don't survive NER). If this holds, it's a clean result for the paper: graph-RAG and token-path solve *different* retrieval failure modes.

**If our hypothesis is wrong** (HippoRAG solves our problem) — that is also a finding worth reporting. It means graph-RAG on text is sufficient, and the contribution of token-path is redundant. We need to know either way.

---

## Experiment Design

### Setup
1. Install HippoRAG per the official repo: `pip install hippoprag` (or clone + install)
2. Use the same MuSiQue embedding-disjoint evaluation corpus we already have:
   - `musique_disjoint_results.json` has the 159 disjoint chains + document IDs
   - The 8,180-doc corpus is in `musique_dense_cache_*.npy` (or rebuild from `musique_disjoint_eval.py`)

### Eval protocol (match our existing eval exactly)

For each of the 159 embedding-disjoint MuSiQue chains:
- **Query:** the question text (same as our current eval)
- **Corpus:** the same 8,180 Wikipedia paragraphs
- **Target:** the gold terminal paragraph (paragraph_support_idx from the dataset)
- **Metric:** recall@10 — is the gold paragraph in the top 10 retrieved?

Run HippoRAG in standard mode (entity graph + PPR retrieval) and report:
- HippoRAG recall@10 on the disjoint tail (n=159)
- Jaccard(HippoRAG, token-path) and Jaccard(HippoRAG, real-iter) — exclusive overlap
- For a qualitative check: sample 5 chains where HippoRAG hits and we miss, and 5 where we hit and HippoRAG misses — what type of bridge does each use?

### If the corpus is too large for HippoRAG indexing
HippoRAG needs to build a knowledge graph over the 8,180-doc corpus, which requires LLM-based NER (often GPT-4 or similar). If API costs are a constraint, use a smaller subset:
- Take the 159 disjoint chains, and for each chain include only the gold answer document + 50 random distractors (total ~8,000 unique docs but index only per-query)
- Or use the HippoRAG offline mode if available

### What to deliver

1. A script `hippograph_eval.py` that:
   - Loads the 159 disjoint MuSiQue chains
   - Runs HippoRAG retrieval on each
   - Computes recall@10, exclusive slices, Jaccard vs. token-path and real-iter
   - Saves results to `hippograph_results.json` (gitignored)

2. A short written analysis (add to FINDINGS.md under Experiment 6):
   - The recall numbers
   - Which chains HippoRAG recovers that we miss (qualitative: is it entity-bridge?)
   - Which chains we recover that HippoRAG misses (qualitative: is it concept-bridge?)
   - One-paragraph interpretation: what does this tell us about the method's regime?

---

## Hypothesis & Expected Results

**Our prediction:**
- HippoRAG recall on MuSiQue disjoint tail: similar to or higher than real-LLM iterative (47.2%) — MuSiQue is entity-spine and graph-RAG should handle it
- HippoRAG on Talos concept-bridge chains: lower than token-path — concept links don't survive NER, PPR finds no edges to traverse

**If HippoRAG < token-path on MuSiQue disjoint:** unexpected — would suggest graph-RAG is also blind to the embedding-disjoint regime, which would strengthen our claim but needs interpretation.

**If HippoRAG >> token-path on MuSiQue disjoint:** expected — confirms entity-bridge vs. concept-bridge distinction. Write: "HippoRAG solves entity-bridge multi-hop but does not address concept-bridge retrieval. Token-path and oracle-iter are complementary to, not redundant with, graph-RAG approaches."

---

## References

- HippoRAG paper: Gutiérrez et al., 2024 — "HippoRAG: Neurologically Inspired Long-Term Memory for Large Language Models"
- HippoRAG repo: https://github.com/OSU-NLP-Group/HippoRAG
- Our MuSiQue eval harness: `musique_disjoint_eval.py`
- Our existing results to compare against: `FINDINGS.md` Experiment 3 (MuSiQue cos<0.3 disjoint)

---

## Decision Point (for Edward)

Before running: check whether HippoRAG requires GPT-4-level NER or works with a local model. The OSU-NLP-Group repo defaults to OpenAI — if API cost is significant (~$5-20 for 8,180 docs at GPT-4 rates), confirm budget or swap NER to a local model (spaCy's NER + a local LLM for relation extraction).

Local alternative: run spaCy NER over the 8,180 docs, build the entity graph manually, then use PPR. Equivalent methodology, no API cost, ~30 min to implement.
