"""
Hard multi-hop corpus generator.

Design principle: bridge nodes have ZERO vocabulary overlap with the query.
The query asks about X. X connects to Y via a bridge note that doesn't
mention X at all. Y connects to Z via another bridge note that doesn't
mention Y. The answer Z is only reachable by crossing the bridges.

This is the genuine test of field propagation — direct resonance cannot
find the answer; propagation is the only path.

Structure:
  - Query vocabulary: {A}
  - Bridge note 1: "{A} → {B}" (mentions A and B)
  - Bridge note 2: "{B} → {C}" (mentions B and C, NOT A)
  - Answer note:   "{C} has property {Z}" (mentions C and Z, NOT A or B)

To retrieve Z, the system must: find bridge1 (via A), propagate to bridge2
(via B), propagate to answer (via C). Direct resonance on the query
vocabulary {A} cannot reach the answer note.
"""

import json
import random
from dataclasses import dataclass, asdict, field
from typing import Literal


@dataclass
class HardNote:
    id: str
    content: str
    vocab: list[str]          # key vocabulary tokens in this note
    hop: int                  # 0=query-reachable, 1=bridge1, 2=bridge2, 3=answer


@dataclass
class HardChain:
    query: str
    query_vocab: list[str]    # tokens in the query
    note_ids: list[str]       # [bridge1_id, bridge2_id, answer_id]
    answer: str


@dataclass
class HardQAPair:
    id: str
    question: str
    answer: str
    required_notes: list[str]
    difficulty: str = "hard_multihop"
    ground_truth_type: str = "chain_reasoning"


# ── Vocabulary pools — disjoint sets so notes can be made truly non-overlapping ──

# Each "concept" is a cluster of tokens around a fictional entity
CONCEPTS = [
    ("zephyr",  ["zephyr", "project", "zephyr-core", "zcore"]),
    ("vantage", ["vantage", "platform", "vantage-sys", "vsys"]),
    ("cobalt",  ["cobalt", "module", "cobalt-net", "cnet"]),
    ("fenwick",  ["fenwick", "service", "fenwick-api", "fapi"]),
    ("stratos", ["stratos", "layer", "stratos-io", "sio"]),
    ("luminos", ["luminos", "engine", "luminos-ml", "lml"]),
    ("orenda",  ["orenda", "core", "orenda-hub", "ohub"]),
    ("kestrel", ["kestrel", "runner", "kestrel-rt", "krt"]),
    ("pallada", ["pallada", "node", "pallada-net", "pnet"]),
    ("solace",  ["solace", "bridge", "solace-link", "slink"]),
    ("nereid",  ["nereid", "agent", "nereid-ai", "nai"]),
    ("thalweg", ["thalweg", "stream", "thalweg-io", "twio"]),
]

PROPERTIES = {
    "owner":      ["owned by {val}", "ownership held by {val}", "lead is {val}"],
    "location":   ["deployed in {val}", "hosted in {val}", "runs in {val}"],
    "dependency": ["depends on {val}", "requires {val}", "built on {val}"],
    "successor":  ["replaced by {val}", "succeeded by {val}", "deprecated in favor of {val}"],
    "team":       ["maintained by {val}", "operated by {val}", "team: {val}"],
}

TEAMS = ["platform-alpha", "core-infra", "ml-ops", "reliability-eng",
         "data-services", "edge-compute", "security-ops", "api-gateway"]
LOCATIONS = ["us-east-1", "eu-west-2", "ap-south-1", "us-central",
             "london-dc", "frankfurt-zone", "singapore-hub", "toronto-edge"]


def _fill(template: str, val: str) -> str:
    return template.replace("{val}", val)


class HardCorpusBuilder:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self._ctr = 0
        self._qa_ctr = 0

    def _nid(self) -> str:
        self._ctr += 1
        return f"hard_{self._ctr:04d}"

    def _qid(self) -> str:
        self._qa_ctr += 1
        return f"hqa_{self._qa_ctr:04d}"

    def _build_chain(
        self,
        concepts: list[tuple],  # 4 distinct concept clusters: query, b1, b2, answer
        prop: str,
        val: str,
        chain_idx: int = 0,
    ) -> tuple[list[HardNote], HardChain, HardQAPair]:
        """
        Build a 3-hop chain:
          Query mentions c0. Bridge1 links c0→c1. Bridge2 links c1→c2.
          Answer note: c2 has property=val.
          Query: "What is the {prop} of {c0_name}?"
          Answer: val
          Path: query→bridge1(c0,c1)→bridge2(c1,c2)→answer(c2,val)
        """
        c0, c1, c2, c3 = concepts
        c0_name, c0_toks = c0
        c1_name, c1_toks = c1
        c2_name, c2_toks = c2
        c3_name, c3_toks = c3

        # Bridge note 1: c0 → c1 (both vocabularies present)
        # Unique relation verbs per chain_idx to prevent cross-chain char-ngram bleed
        rel_verbs_01 = [
            (f"{c0_name} coordinates with {c1_name} directly.", f"{c0_name} links through {c1_name}."),
            (f"{c0_name} forwards traffic to {c1_name}.", f"{c0_name} syncs data to {c1_name}."),
            (f"{c0_name} wraps around {c1_name}.", f"{c0_name} proxies through {c1_name}."),
        ]
        # Pick based on position in all_notes to get a unique verb set
        verb_idx = chain_idx % len(rel_verbs_01)
        rel_phrase_01 = self.rng.choice(rel_verbs_01[verb_idx])
        b1 = HardNote(
            id=self._nid(),
            content=rel_phrase_01,
            vocab=c0_toks + c1_toks,
            hop=1,
        )

        # Bridge note 2: c1 → c2 (ONLY c1 and c2 vocab — no c0 tokens)
        rel_verbs_12 = [
            (f"{c1_name} belongs under {c2_name}.", f"{c1_name} operates within {c2_name}."),
            (f"{c1_name} reports status to {c2_name}.", f"{c1_name} is supervised by {c2_name}."),
            (f"{c1_name} runs inside {c2_name}.", f"{c1_name} is embedded within {c2_name}."),
        ]
        verb_idx_12 = chain_idx % len(rel_verbs_12)
        rel_phrase_12 = self.rng.choice(rel_verbs_12[verb_idx_12])
        b2 = HardNote(
            id=self._nid(),
            content=rel_phrase_12,
            vocab=c1_toks + c2_toks,
            hop=2,
        )

        # Answer note: c2 has the property (ONLY c2 vocab and val — no c0/c1 tokens)
        prop_template = self.rng.choice(PROPERTIES[prop])
        answer_content = f"{c2_name} {_fill(prop_template, val)}."
        a = HardNote(
            id=self._nid(),
            content=answer_content,
            vocab=c2_toks + val.split("-"),
            hop=3,
        )

        query = f"What is the {prop} of {c0_name}?"
        chain = HardChain(
            query=query,
            query_vocab=c0_toks,
            note_ids=[b1.id, b2.id, a.id],
            answer=val,
        )
        qa = HardQAPair(
            id=self._qid(),
            question=query,
            answer=f"{val} (via {c1_name} and {c2_name})",
            required_notes=[b1.id, b2.id, a.id],
        )
        return [b1, b2, a], chain, qa

    def build(self, n_chains: int = 8, n_distractors: int = 40) -> dict:
        all_notes: list[HardNote] = []
        chains: list[HardChain] = []
        qa_pairs: list[HardQAPair] = []

        # Shuffle concepts so each chain gets a unique set of 4
        concepts = list(CONCEPTS)
        self.rng.shuffle(concepts)

        prop_cycle = list(PROPERTIES.keys())
        val_pools = {"owner": TEAMS, "location": LOCATIONS, "dependency": TEAMS,
                     "successor": [c[0] for c in CONCEPTS], "team": TEAMS}

        for i in range(min(n_chains, len(concepts) // 4)):
            c_slice = concepts[i*4:(i+1)*4]
            prop = prop_cycle[i % len(prop_cycle)]
            val = self.rng.choice(val_pools[prop])
            notes, chain, qa = self._build_chain(c_slice, prop, val, chain_idx=i)
            all_notes.extend(notes)
            chains.append(chain)
            qa_pairs.append(qa)

        # Distractor notes: real-looking but irrelevant to any chain
        distractor_templates = [
            # Non-technical distractors — minimal char-ngram overlap with chain concepts
            "Quarterly budget review scheduled for the third week of next month.",
            "Building access cards must be renewed before the end of the fiscal year.",
            "The annual staff survey results will be published on the intranet portal.",
            "Travel reimbursements should be submitted within thirty days of the trip.",
            "The library catalog has been updated with new research journals this season.",
            "Parking permits for the north lot expire at the end of this quarter.",
            "Emergency exit procedures have been updated on the posted floor plan.",
            "Cafeteria hours have been extended to accommodate the evening shift.",
            "All visitors must sign in at the reception desk before proceeding.",
            "The recycling program now accepts batteries and small electronics.",
            "Room bookings for the conference center require two weeks advance notice.",
            "Health insurance open enrollment closes at the end of this month.",
            "The new phone directory has been distributed to all department heads.",
            "Volunteers needed for the community garden project this weekend.",
            "Printer supplies can be ordered through the facilities request portal.",
            "Onboarding documents for new hires are available in the shared folder.",
            "Security badges must be worn visibly at all times in the building.",
            "The annual fire safety inspection is scheduled for the coming Friday.",
            "Feedback forms for the workshop are available at the registration table.",
            "IT support tickets should be submitted via the help desk portal.",
        ]
        all_concept_names = [c[0] for c in CONCEPTS]
        for j in range(n_distractors):
            cname = self.rng.choice(all_concept_names)
            template = self.rng.choice(distractor_templates)
            content = template.replace("{c}", cname)
            all_notes.append(HardNote(
                id=self._nid(),
                content=content,
                vocab=[cname],
                hop=-1,
            ))

        # Shuffle note order
        self.rng.shuffle(all_notes)

        notes_dicts = [
            {"id": n.id, "content": n.content, "category": "fact",
             "tags": ["hard_multihop", f"hop{n.hop}"], "hop": n.hop}
            for n in all_notes
        ]
        qa_dicts = [asdict(q) for q in qa_pairs]

        metadata = {
            "size": "hard_multihop",
            "seed": seed,
            "total_notes": len(all_notes),
            "total_qa_pairs": len(qa_pairs),
            "n_chains": len(chains),
            "n_distractors": n_distractors,
            "design": "bridge nodes have zero query-vocabulary overlap; propagation required",
        }

        return {
            "metadata": metadata,
            "notes": notes_dicts,
            "contradictions": [],
            "alias_groups": [],
            "multihop_chains": [
                {"steps": c.note_ids, "question": c.query, "answer": c.answer,
                 "query_vocab": c.query_vocab}
                for c in chains
            ],
            "qa_pairs": qa_dicts,
        }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/Users/edward/.ocplatform/workspace/research/rfm/corpus_hard.json")
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--distractors", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    seed = args.seed

    builder = HardCorpusBuilder(seed=seed)
    corpus = builder.build(n_chains=args.chains, n_distractors=args.distractors)

    with open(args.output, "w") as f:
        json.dump(corpus, f, indent=2)

    m = corpus["metadata"]
    print(f"Hard corpus: {m['total_notes']} notes, {m['total_qa_pairs']} QA pairs, "
          f"{m['n_chains']} chains, {m['n_distractors']} distractors")
    print(f"Design: {m['design']}")
    print(f"Output: {args.output}")

    # Quick sanity check: verify bridge nodes really don't overlap with queries
    note_map = {n["id"]: n for n in corpus["notes"]}
    import re
    def toks(t): return set(re.findall(r"[a-z]+", t.lower()))
    print("\nVocabulary isolation check (answer notes vs queries):")
    for chain in corpus["multihop_chains"]:
        q_toks = toks(chain["question"])
        ans_note = note_map[chain["steps"][-1]]
        a_toks = toks(ans_note["content"])
        overlap = q_toks & a_toks - {"is", "the", "of", "a", "what"}
        print(f"  query: '{chain['question'][:50]}...'")
        print(f"  answer note overlap with query vocab: {sorted(overlap)} "
              f"{'✓ ISOLATED' if not overlap else '✗ HAS OVERLAP'}")
