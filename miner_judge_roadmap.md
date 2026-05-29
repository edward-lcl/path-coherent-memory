# Miner and Judge Roadmap

Current status: path-coherent topology is ahead of the benchmark. The method
can retrieve terminal memory nodes across lexical gaps, but the miner produces
too many weak or artifact chains. The judge layer is now the main experimental
bottleneck.

## Working Thesis

RFM should be evaluated as a path-coherent memory traversal layer:

1. lexical retrieval anchors the first relevant chunk;
2. topology selects locally recurring bridge entities;
3. path coherence preserves a chain across zero-vocabulary gaps;
4. a semantic judge decides whether the mined path is a human-meaningful
   memory chain.

The retriever should not get credit for artifact paths. The benchmark must
separate "topology exists" from "semantic memory exists."

## Current Evidence

- hard_v2 synthetic benchmark: path-coherent topology retrieves 20/20 terminal
  answer nodes; BM25 and cosine retrieve 0/20.
- Levi substrate self-supervised benchmark: path coherence retrieves about 40%
  of terminal nodes; BM25 and cosine retrieve about 1%.
- Levi labeled subset: real_semantic chains show 8/9 terminal retrieval for
  path coherence vs 1/9 for BM25/cosine.
- Artifact chains still score above baseline, which proves the retriever can
  follow nonsemantic structure.

## Miner Improvements

The miner should optimize semantic-chain yield, not raw path-retrieval score.

### Candidate generation

- Generate a larger candidate pool, at least 300-500 chains.
- Preserve source diversity but avoid over-weighting noisy session chunks.
- Keep hard filters for code/hash artifacts:
  - reject long hex/hash-like tokens;
  - reject non-alphabetic bridge tokens;
  - reject obvious implementation tokens when both adjacent chunks are code;
  - reject high-frequency structural/prose tokens.
- Add context-window evidence around bridge terms instead of using whole-chunk
  token overlap alone.

### Candidate scoring

Rank candidates before judging with features that should correlate with
semantic chain quality:

- adjacent bridge locality: bridge term appears near the relevant sentence in
  both chunks;
- source coherence: sources are different enough to be nontrivial but not so
  different that the path is likely drift;
- endpoint specificity: terminal chunk contains a concrete project, person,
  decision, artifact, or durable state;
- bridge entity quality: bridge tokens look like named concepts/entities rather
  than generic adjectives or verbs;
- artifact risk: code symbols, CLI flags, markdown boilerplate, log fragments,
  IDs, cache keys, and repeated status text.

Cheap feature scoring already failed as a final separator. Use it only to
improve candidate ordering before semantic judgment.

## Judge Improvements

The judge should be calibrated against `levi_semantic_chain_answer_key_v1.jsonl`.

### Local prefilter

`mlx-community/Llama-3.2-3B-Instruct-4bit` is usable as a prefilter but not as
the final arbiter:

- zero-shot: 55% accuracy on first 20 chains;
- few-shot: 75% accuracy on first 20 chains;
- still too lenient on artifacts.

### Strong judge target

Use a stronger judge only after privacy boundaries are explicit:

- local larger model if available and stable;
- human adjudication for the strict real_semantic slice;
- external/cloud judge only with explicit approval, because substrate excerpts
  can contain private memory content.

### Acceptance criteria

For a judge to become benchmark-grade:

- real_semantic selector precision >= 80%;
- real_semantic recall >= 70%;
- artifact false-positive rate <= 10%;
- disagreements saved for human review;
- retrieval metrics reported only on judge-accepted chains.

## Next Benchmark

Build `levi_substrate_v1` as a frozen judged benchmark:

1. Mine 300-500 candidate chains.
2. Apply feature prefilter to pick the top 120-200 for judging.
3. Judge candidates with the rubric.
4. Human-review borderline and positive cases.
5. Freeze 30-50 high-confidence semantic chains if enough exist.
6. Rerun BM25, cosine, path coherence, and negative controls only on the frozen
   accepted set.

Success condition: path coherence keeps a large terminal-retrieval lift over
BM25/cosine on at least 30 judged semantic chains.

