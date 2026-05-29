# Semantic Chain Judge Prompt

You are judging candidate three-hop memory chains mined from Levi's local
memory substrate. The retrieval algorithm is not being judged here. Your job is
to decide whether the mined path is a meaningful semantic memory chain or just
a token/topology coincidence.

## Input fields

- `idx`: candidate number
- `start_token`: sparse query token used to anchor the first chunk
- `bridge1`: token shared by chunk A and chunk B
- `bridge2`: token shared by chunk B and chunk C
- `sources`: source files for chunks A, B, and C
- `a_excerpt`: anchor chunk
- `b_excerpt`: bridge chunk
- `c_excerpt`: terminal chunk

## Labels

Use exactly one label:

- `real_semantic`: A, B, and C form a defensible human-semantic memory chain.
  The bridge terms carry actual meaning, and a person could explain why the
  terminal chunk belongs downstream of the anchor chunk.
- `weak_semantic`: At least one adjacent relation is meaningful, but the full
  chain drifts, overgeneralizes, or depends on a broad topical association.
- `artifact`: The path is mostly code symbols, logs, hashes, boilerplate,
  repeated prose, homonyms, formatting, or accidental token overlap.

## Scoring dimensions

Score each dimension from 0 to 2:

- `ab_coherence`: Does A connect meaningfully to B through bridge1?
- `bc_coherence`: Does B connect meaningfully to C through bridge2?
- `terminal_relevance`: Is C a meaningful endpoint for the chain, not merely
  a chunk that happens to share bridge2?
- `bridge_meaning`: Are bridge1 and bridge2 domain concepts/entities rather
  than generic prose, implementation tokens, or formatting artifacts?
- `artifact_penalty`: 0 means no artifact smell; 1 means mixed; 2 means the
  chain is dominated by code/log/boilerplate/homonym artifacts.

## Decision rules

- Label `real_semantic` only when both adjacent hops are meaningful and the
  terminal chunk adds a coherent downstream memory.
- Label `weak_semantic` when the chain has a real relation but the path drifts
  or the terminal is only loosely related.
- Label `artifact` when bridge terms are code/API symbols, generic adjectives,
  status words, markdown boilerplate, repeated UI text, or unrelated homonyms.
- Do not reward a chain just because the retrieval method found the terminal.
  The label is about human semantic validity.
- When uncertain between `real_semantic` and `weak_semantic`, choose
  `weak_semantic`.
- When uncertain between `weak_semantic` and `artifact`, choose `artifact`
  if either bridge is mainly implementation/log/formatting noise.

## Output schema

Return one JSON object per candidate:

```json
{
  "idx": 1,
  "label": "real_semantic",
  "scores": {
    "ab_coherence": 2,
    "bc_coherence": 2,
    "terminal_relevance": 2,
    "bridge_meaning": 2,
    "artifact_penalty": 0
  },
  "rationale": "One concise sentence explaining the decision."
}
```

