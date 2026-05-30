"""
Synthetic corpus generator for Resonance Field Memory (RFM) benchmarking.

Generates a controlled dataset with known properties:
  - Contradictions: facts that mutually conflict
  - Aliases:        different names for the same entity
  - Multi-hop chains: A→B, B→C, therefore A→C

Each generated item has full ground-truth labels so ablation experiments
can measure recall/precision against known answers.

Usage:
    python3 synthetic_corpus.py --output corpus.json --size medium
    python3 synthetic_corpus.py --output corpus.json --size large --seed 42
"""

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from itertools import combinations

# ────────────────────────────────────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryNote:
    id: str
    content: str
    category: Literal["fact", "alias", "event", "relationship", "preference"]
    tags: list[str] = field(default_factory=list)

@dataclass
class Contradiction:
    note_a: str          # note id
    note_b: str          # note id
    field: str           # which field contradicts
    value_a: str
    value_b: str
    canonical: str       # which is the ground-truth value ("a" | "b" | "unknown")

@dataclass
class AliasGroup:
    canonical_name: str
    aliases: list[str]
    note_ids: list[str]

@dataclass
class MultiHopChain:
    steps: list[str]     # note ids in hop order
    question: str        # natural-language query that requires all hops
    answer: str          # ground-truth answer

@dataclass
class QAPair:
    id: str
    question: str
    answer: str
    required_notes: list[str]   # note ids needed to answer
    difficulty: Literal["direct", "contradiction", "alias", "multihop"]
    ground_truth_type: str      # "recall", "contradiction_detect", "alias_resolve", "chain_reasoning"

@dataclass
class Corpus:
    notes: list[MemoryNote]
    contradictions: list[Contradiction]
    alias_groups: list[AliasGroup]
    multihop_chains: list[MultiHopChain]
    qa_pairs: list[QAPair]
    metadata: dict

# ────────────────────────────────────────────────────────────────────────────
# Templates
# ────────────────────────────────────────────────────────────────────────────

PERSON_TEMPLATES = [
    ("Alice Chen",    "engineer",     "San Francisco", "Python",    "hiking"),
    ("Bob Martinez",  "designer",     "Austin",        "Figma",     "cooking"),
    ("Carol Singh",   "PM",           "New York",      "Notion",    "cycling"),
    ("David Kim",     "researcher",   "Boston",        "R",         "chess"),
    ("Elena Petrov",  "data scientist","Seattle",      "PyTorch",   "swimming"),
    ("Fiona Walsh",   "CTO",          "Dublin",        "Go",        "running"),
    ("George Nkosi",  "architect",    "Johannesburg",  "Rust",      "photography"),
    ("Hannah Müller",  "analyst",     "Berlin",        "SQL",       "reading"),
    ("Ivan Torres",   "devops",       "Mexico City",   "Kubernetes","football"),
    ("Julia Yamamoto","UX lead",      "Tokyo",         "Sketch",    "yoga"),
]

PROJECT_TEMPLATES = [
    ("Atlas",    "distributed cache",  "Q3 2026",  "internal",  "Redis-compatible"),
    ("Helios",   "ML pipeline",        "Q4 2026",  "external",  "Python-first"),
    ("Nimbus",   "edge CDN",           "Q2 2027",  "internal",  "WebAssembly"),
    ("Orion",    "billing system",     "Q1 2027",  "external",  "event-sourced"),
    ("Solaris",  "auth service",       "Q3 2026",  "internal",  "zero-trust"),
    ("Vortex",   "realtime analytics", "Q2 2026",  "external",  "ClickHouse"),
    ("Horizon",  "mobile platform",    "Q4 2026",  "internal",  "cross-platform"),
    ("Zenith",   "recommendation engine","Q1 2027","external",  "graph-based"),
]

ALIAS_SEEDS = [
    ("Project Atlas",   ["Atlas", "the cache project", "atlas-cache", "PROJ-100"]),
    ("Alice Chen",      ["Alice", "a.chen", "the SF engineer", "AChen"]),
    ("ML pipeline",     ["Helios", "the pipeline", "helios-ml", "the ML project"]),
    ("Bob Martinez",    ["Bob", "b.martinez", "the Austin designer", "BobM"]),
    ("auth service",    ["Solaris", "the auth layer", "solaris-auth", "IAM service"]),
]

# ────────────────────────────────────────────────────────────────────────────
# Builder
# ────────────────────────────────────────────────────────────────────────────

class CorpusBuilder:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._note_counter = 0
        self._qa_counter = 0

    def _note_id(self) -> str:
        self._note_counter += 1
        return f"note_{self._note_counter:04d}"

    def _qa_id(self) -> str:
        self._qa_counter += 1
        return f"qa_{self._qa_counter:04d}"

    # ── Person notes ──────────────────────────────────────────────────────

    def _person_notes(self, name, role, city, tool, hobby) -> list[MemoryNote]:
        nid = self._note_id()
        return [MemoryNote(
            id=nid,
            content=(
                f"{name} is a {role} based in {city}. "
                f"Their primary tool is {tool} and they enjoy {hobby} outside work."
            ),
            category="fact",
            tags=["person", role.split()[0].lower(), city.lower().replace(" ", "_")],
        )]

    # ── Project notes ─────────────────────────────────────────────────────

    def _project_notes(self, pname, ptype, deadline, scope, tech) -> list[MemoryNote]:
        nid = self._note_id()
        return [MemoryNote(
            id=nid,
            content=(
                f"{pname} is an {scope} {ptype} project targeting {deadline}. "
                f"The stack is {tech}."
            ),
            category="fact",
            tags=["project", scope, tech.split("-")[0].lower()],
        )]

    # ── Contradiction notes ───────────────────────────────────────────────

    def _contradiction_pair(
        self,
        subject: str,
        field: str,
        value_a: str,
        value_b: str,
        canonical: str = "a",
    ) -> tuple[list[MemoryNote], Contradiction]:
        nid_a = self._note_id()
        nid_b = self._note_id()
        note_a = MemoryNote(
            id=nid_a,
            content=f"{subject}'s {field} is {value_a}.",
            category="fact",
            tags=["fact", subject.lower().replace(" ", "_"), field.lower()],
        )
        note_b = MemoryNote(
            id=nid_b,
            content=f"{subject}'s {field} is {value_b}.",
            category="fact",
            tags=["fact", subject.lower().replace(" ", "_"), field.lower()],
        )
        contradiction = Contradiction(
            note_a=nid_a,
            note_b=nid_b,
            field=field,
            value_a=value_a,
            value_b=value_b,
            canonical=canonical,
        )
        return [note_a, note_b], contradiction

    # ── Alias notes ───────────────────────────────────────────────────────

    def _alias_group(
        self, canonical: str, aliases: list[str], note_ids: list[str]
    ) -> tuple[list[MemoryNote], AliasGroup]:
        alias_notes = []
        all_ids = list(note_ids)
        for alias in aliases[1:]:  # first alias is usually the canonical name itself
            nid = self._note_id()
            alias_notes.append(MemoryNote(
                id=nid,
                content=f'"{alias}" refers to {canonical}.',
                category="alias",
                tags=["alias", canonical.lower().replace(" ", "_")],
            ))
            all_ids.append(nid)
        group = AliasGroup(canonical_name=canonical, aliases=aliases, note_ids=all_ids)
        return alias_notes, group

    # ── Multi-hop chains ──────────────────────────────────────────────────

    def _multihop_chain(
        self,
        steps: list[tuple[str, str]],  # [(content, answer_fragment), ...]
        question: str,
        answer: str,
    ) -> tuple[list[MemoryNote], MultiHopChain]:
        notes = []
        nids = []
        for content, _ in steps:
            nid = self._note_id()
            notes.append(MemoryNote(
                id=nid,
                content=content,
                category="relationship",
                tags=["chain", "relationship"],
            ))
            nids.append(nid)
        chain = MultiHopChain(steps=nids, question=question, answer=answer)
        return notes, chain

    # ── QA generation ─────────────────────────────────────────────────────

    def _qa_direct(self, note: MemoryNote, question: str, answer: str) -> QAPair:
        return QAPair(
            id=self._qa_id(),
            question=question,
            answer=answer,
            required_notes=[note.id],
            difficulty="direct",
            ground_truth_type="recall",
        )

    def _qa_contradiction(
        self, c: Contradiction, question: str, answer: str
    ) -> QAPair:
        return QAPair(
            id=self._qa_id(),
            question=question,
            answer=answer,
            required_notes=[c.note_a, c.note_b],
            difficulty="contradiction",
            ground_truth_type="contradiction_detect",
        )

    def _qa_alias(
        self, alias_group: AliasGroup, question: str, answer: str
    ) -> QAPair:
        return QAPair(
            id=self._qa_id(),
            question=question,
            answer=answer,
            required_notes=alias_group.note_ids,
            difficulty="alias",
            ground_truth_type="alias_resolve",
        )

    def _qa_multihop(self, chain: MultiHopChain) -> QAPair:
        return QAPair(
            id=self._qa_id(),
            question=chain.question,
            answer=chain.answer,
            required_notes=chain.steps,
            difficulty="multihop",
            ground_truth_type="chain_reasoning",
        )

    # ── Main build ────────────────────────────────────────────────────────

    def build(self, size: Literal["small", "medium", "large"] = "medium") -> Corpus:
        n_persons  = {"small": 4,  "medium": 8,  "large": 10}[size]
        n_projects = {"small": 3,  "medium": 5,  "large": 8}[size]
        n_contras  = {"small": 3,  "medium": 6,  "large": 10}[size]
        n_aliases  = {"small": 2,  "medium": 4,  "large": 5}[size]
        n_chains   = {"small": 2,  "medium": 4,  "large": 6}[size]

        all_notes: list[MemoryNote] = []
        contradictions: list[Contradiction] = []
        alias_groups: list[AliasGroup] = []
        chains: list[MultiHopChain] = []
        qa_pairs: list[QAPair] = []

        # ── Persons ───────────────────────────────────────────────────────
        person_note_map: dict[str, MemoryNote] = {}
        persons = self.rng.sample(PERSON_TEMPLATES, n_persons)
        for name, role, city, tool, hobby in persons:
            notes = self._person_notes(name, role, city, tool, hobby)
            all_notes.extend(notes)
            person_note_map[name] = notes[0]
            qa_pairs.append(self._qa_direct(
                notes[0],
                f"Where is {name} based?",
                city,
            ))
            qa_pairs.append(self._qa_direct(
                notes[0],
                f"What is {name}'s primary tool?",
                tool,
            ))

        # ── Projects ──────────────────────────────────────────────────────
        project_note_map: dict[str, MemoryNote] = {}
        projects = self.rng.sample(PROJECT_TEMPLATES, n_projects)
        for pname, ptype, deadline, scope, tech in projects:
            notes = self._project_notes(pname, ptype, deadline, scope, tech)
            all_notes.extend(notes)
            project_note_map[pname] = notes[0]
            qa_pairs.append(self._qa_direct(
                notes[0],
                f"What is the target deadline for {pname}?",
                deadline,
            ))

        # ── Contradictions ────────────────────────────────────────────────
        contra_seeds = [
            ("Alice Chen",  "location",      "San Francisco", "Austin"),
            ("Bob Martinez","seniority",      "senior",        "junior"),
            ("Project Atlas","deadline",      "Q3 2026",       "Q4 2026"),
            ("Carol Singh",  "role",          "PM",            "Engineering Manager"),
            ("Project Helios","scope",        "external",      "internal"),
            ("David Kim",    "primary tool",  "R",             "Python"),
            ("Elena Petrov", "city",          "Seattle",       "Portland"),
            ("Fiona Walsh",  "company size",  "startup",       "enterprise"),
            ("Project Nimbus","tech",         "WebAssembly",   "V8-native"),
            ("George Nkosi", "timezone",      "SAST",          "UTC"),
        ]
        chosen_contras = self.rng.sample(contra_seeds, min(n_contras, len(contra_seeds)))
        for subject, field, val_a, val_b in chosen_contras:
            notes, contra = self._contradiction_pair(subject, field, val_a, val_b)
            all_notes.extend(notes)
            contradictions.append(contra)
            qa_pairs.append(self._qa_contradiction(
                contra,
                f"There are two conflicting records about {subject}'s {field}. "
                f"What are they, and which is more likely correct?",
                f"Conflict: '{val_a}' vs '{val_b}'. Canonical value: {val_a}.",
            ))

        # ── Aliases ───────────────────────────────────────────────────────
        chosen_alias_seeds = self.rng.sample(ALIAS_SEEDS, min(n_aliases, len(ALIAS_SEEDS)))
        for canonical, aliases in chosen_alias_seeds:
            # tie alias group to existing notes where possible
            base_ids = [
                n.id for n in all_notes
                if canonical.lower() in n.content.lower()
            ][:2]
            alias_notes, group = self._alias_group(canonical, aliases, base_ids)
            all_notes.extend(alias_notes)
            alias_groups.append(group)
            # pick a random alias and ask about the canonical
            alias_query = self.rng.choice(aliases[1:]) if len(aliases) > 1 else aliases[0]
            qa_pairs.append(self._qa_alias(
                group,
                f'What entity is "{alias_query}" referring to?',
                canonical,
            ))

        # ── Multi-hop chains ──────────────────────────────────────────────
        chain_templates = [
            (
                [
                    ("Alice Chen reports to Fiona Walsh.",      "Alice → Fiona"),
                    ("Fiona Walsh is the CTO.",                  "Fiona → CTO"),
                    ("The CTO owns the Solaris project.",        "CTO → Solaris"),
                ],
                "Who owns the Solaris project, and what is their relationship to Alice Chen?",
                "Fiona Walsh owns Solaris. Alice Chen reports to Fiona Walsh, who is CTO.",
            ),
            (
                [
                    ("Project Atlas is owned by the Platform team.",     "Atlas → Platform"),
                    ("The Platform team lead is George Nkosi.",          "Platform → George"),
                    ("George Nkosi is based in Johannesburg.",           "George → Johannesburg"),
                ],
                "In which city is the owner of Project Atlas based?",
                "Johannesburg. Project Atlas is owned by the Platform team, led by George Nkosi, who is based there.",
            ),
            (
                [
                    ("Helios uses a Python-first stack.",                "Helios → Python"),
                    ("Elena Petrov is the lead data scientist on Helios.","Helios → Elena"),
                    ("Elena Petrov's preferred tool is PyTorch.",         "Elena → PyTorch"),
                ],
                "What ML library does the lead of the Helios project prefer?",
                "PyTorch. Elena Petrov leads Helios and PyTorch is her primary tool.",
            ),
            (
                [
                    ("Bob Martinez is the designer on Project Nimbus.",    "Bob → Nimbus"),
                    ("Project Nimbus targets edge CDN use cases.",         "Nimbus → CDN"),
                    ("Edge CDN work requires latency benchmarks below 5ms.","CDN → 5ms"),
                ],
                "What latency requirement applies to the project Bob Martinez is working on?",
                "Below 5ms. Bob is on Project Nimbus (edge CDN), which requires sub-5ms latency.",
            ),
            (
                [
                    ("Carol Singh manages the Orion project.",             "Carol → Orion"),
                    ("Orion is an external billing system.",               "Orion → billing"),
                    ("External billing systems must comply with PCI-DSS.", "billing → PCI-DSS"),
                ],
                "What compliance standard applies to the project Carol Singh manages?",
                "PCI-DSS. Carol manages Orion, which is an external billing system subject to PCI-DSS.",
            ),
            (
                [
                    ("Hannah Müller is an analyst on the Vortex project.", "Hannah → Vortex"),
                    ("Vortex is built on ClickHouse.",                     "Vortex → ClickHouse"),
                    ("ClickHouse clusters require columnar storage planning.","ClickHouse → columnar"),
                ],
                "What storage planning consideration applies to Hannah Müller's project?",
                "Columnar storage planning. Hannah works on Vortex, which is ClickHouse-based.",
            ),
        ]
        chosen_chains = self.rng.sample(chain_templates, min(n_chains, len(chain_templates)))
        for steps, question, answer in chosen_chains:
            notes, chain = self._multihop_chain(steps, question, answer)
            all_notes.extend(notes)
            chains.append(chain)
            qa_pairs.append(self._qa_multihop(chain))

        # ── Metadata ──────────────────────────────────────────────────────
        difficulty_dist = {}
        for qa in qa_pairs:
            difficulty_dist[qa.difficulty] = difficulty_dist.get(qa.difficulty, 0) + 1

        metadata = {
            "size": size,
            "seed": self.rng.getstate()[1][0],
            "total_notes": len(all_notes),
            "total_qa_pairs": len(qa_pairs),
            "total_contradictions": len(contradictions),
            "total_alias_groups": len(alias_groups),
            "total_multihop_chains": len(chains),
            "difficulty_distribution": difficulty_dist,
        }

        return Corpus(
            notes=all_notes,
            contradictions=contradictions,
            alias_groups=alias_groups,
            multihop_chains=chains,
            qa_pairs=qa_pairs,
            metadata=metadata,
        )


# ────────────────────────────────────────────────────────────────────────────
# Serialization helpers
# ────────────────────────────────────────────────────────────────────────────

def corpus_to_dict(corpus: Corpus) -> dict:
    return {
        "metadata": corpus.metadata,
        "notes": [asdict(n) for n in corpus.notes],
        "contradictions": [asdict(c) for c in corpus.contradictions],
        "alias_groups": [asdict(a) for a in corpus.alias_groups],
        "multihop_chains": [asdict(c) for c in corpus.multihop_chains],
        "qa_pairs": [asdict(q) for q in corpus.qa_pairs],
    }


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate RFM synthetic benchmark corpus")
    parser.add_argument("--output", default="corpus.json", help="Output file path")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    builder = CorpusBuilder(seed=args.seed)
    corpus = builder.build(size=args.size)
    data = corpus_to_dict(corpus)

    indent = 2 if args.pretty else None
    with open(args.output, "w") as f:
        json.dump(data, f, indent=indent)

    m = data["metadata"]
    print(f"Generated {m['size']} corpus → {args.output}")
    print(f"  Notes:          {m['total_notes']}")
    print(f"  QA pairs:       {m['total_qa_pairs']}")
    print(f"  Contradictions: {m['total_contradictions']}")
    print(f"  Alias groups:   {m['total_alias_groups']}")
    print(f"  Multi-hop:      {m['total_multihop_chains']}")
    print(f"  Difficulty:     {m['difficulty_distribution']}")


if __name__ == "__main__":
    main()
