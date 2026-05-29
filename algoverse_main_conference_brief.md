# Algoverse Main-Conference Brief: Path-Coherent Memory Retrieval

## One-Sentence Pitch

We may have isolated a retrieval layer that lexical and dense methods often
miss: path-coherent topology over memory graphs, where sparse lexical search
anchors the first node and topology retrieves multi-hop endpoints across
zero-vocabulary gaps.

## Why It Might Matter

Modern retrieval systems are strong at direct similarity. They are weaker when
the answer is several memory hops away and shares no terms with the query.
In controlled zero-overlap tests, BM25 and TF-IDF cosine retrieve no terminal
answer nodes, while path-coherent topology retrieves all of them. Early tests on
Levi's real memory substrate show the same shape, but the benchmark still needs
stronger semantic judgment.

## Current Results

Synthetic hard_v2:

- BM25/cosine terminal answer retrieval: 0/20
- path-coherent topology: 20/20
- key condition: answer nodes have zero query-vocabulary overlap

Levi substrate, self-supervised:

- BM25/cosine terminal retrieval: about 1%
- path coherence terminal retrieval: about 40%
- rotated-terminal negative control: 0%

Levi substrate, manually labeled:

- real_semantic chains: path coherence 8/9, BM25/cosine 1/9
- real+weak chains: path coherence 24/53, BM25/cosine 1/53
- artifact chains also score above baseline, proving the judge/miner must be
  improved before publication claims.

## Honest Claim Boundary

Do not claim "RFM beats BM25" generally.

The credible claim is narrower and stronger:

> Path-coherent topology can retrieve multi-hop memory endpoints across lexical
> gaps where lexical similarity methods cannot directly reach the answer.

The current bottleneck is benchmark construction: mining and judging real
semantic chains from messy memory data.

## Research Questions

1. Can path-coherent topology beat BM25/cosine on judged semantic multi-hop
   chains from real memory logs?
2. Can the effect reproduce on an independent corpus, such as Talos/M1, while
   preserving project boundaries?
3. How much of the gain comes from lexical anchoring, bridge selection,
   topology, and path-coherence constraints?
4. Can a semantic-chain judge reliably distinguish meaningful memory paths from
   code/log/topology artifacts?

## Work Packages

### Benchmark

- Generate 300-500 candidate memory chains.
- Build a high-precision semantic-chain judge.
- Freeze a 30-50 chain `levi_substrate_v1` benchmark.
- Run negative controls and ablations.

### Retrieval Method

- Formalize path-coherent topology.
- Compare against BM25, TF-IDF cosine, random, oracle, and dense retrieval if
  available.
- Run ablations: anchor only, topology only, path coherence only, combined.

### External Validation

- Repeat the judged protocol on a separate corpus only after boundaries are
  clean.
- Talos/M1 is a candidate validation corpus, not the first proving ground.

### Paper

- Frame as memory retrieval across lexical gaps.
- Include synthetic mechanism proof, real-substrate judged benchmark, ablations,
  negative controls, and failure analysis.

## Why Algoverse Could Help

This is a good fit for a main-conference-focused research group because it needs
rigor more than product polish:

- benchmark design;
- independent adjudication;
- related-work review;
- ablation discipline;
- stronger baselines;
- writing and submission strategy.

The useful collaborator profile is someone willing to stress-test the claim,
not someone looking to hype it. The result becomes interesting only if it
survives stricter benchmarks.

## Immediate Ask

If Algoverse wants a main-conference acceptance track, this project could be a
candidate under a "retrieval and memory systems" section. The first milestone
would be a reproducible judged benchmark packet, not a paper draft.

