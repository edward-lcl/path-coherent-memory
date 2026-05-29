"""
RFM Experiment Suite

Exp 1 — Ablation: RFMv3 with propagation disabled vs enabled
Exp 2 — Scaling: generate large synthetic corpora (200/500/1000 notes), test all backends
Exp 3 — Noise injection: add k random irrelevant notes, measure degradation curve

Usage:
    python3 experiments.py              # run all
    python3 experiments.py --exp 1      # ablation only
    python3 experiments.py --exp 2      # scaling only
    python3 experiments.py --exp 3      # noise only
"""

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass

RFM_DIR = Path('/Users/edward/.ocplatform/workspace/research/rfm')
sys.path.insert(0, str(RFM_DIR))

import evaluator as ev
from synthetic_corpus import CorpusBuilder, corpus_to_dict

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def run_backend(corpus: dict, backend_name: str, top_k: int = 5) -> dict:
    return ev.run_eval(corpus, backend_name, top_k)

def f1(report: dict, difficulty: str | None = None) -> float:
    if difficulty:
        return report["by_difficulty"].get(difficulty, {}).get("f1", 0.0)
    return report["overall"]["f1"]

def hitk(report: dict, difficulty: str | None = None) -> float:
    if difficulty:
        return report["by_difficulty"].get(difficulty, {}).get("hit@k", 0.0)
    return report["overall"]["hit@k"]

def header(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

# ────────────────────────────────────────────────────────────────────────────
# Exp 1 — Ablation
# ────────────────────────────────────────────────────────────────────────────

class RFMNoPropagate(ev.RFMv3Backend):
    """RFMv3 with propagation zeroed out — isolates direct resonance only."""
    PROP_W = 0.0

def exp1_ablation():
    header("EXP 1 — Ablation: propagation ON vs OFF")

    # Register the no-prop backend temporarily
    ev._ABLATION_BACKEND = RFMNoPropagate

    sizes = ["small", "medium", "large"]
    print(f"\n{'size':>8}  {'rfmv3_noprop F1':>16}  {'rfmv3 F1':>10}  "
          f"{'delta':>7}  {'mhop noprop':>12}  {'mhop rfmv3':>11}  {'mhop delta':>11}")
    print("-" * 90)

    for size in sizes:
        corpus_path = RFM_DIR / f"corpus_{size}.json"
        with open(corpus_path) as f:
            corpus = json.load(f)

        # No-prop: just direct resonance
        noprop = RFMNoPropagate()
        noprop.index(corpus["notes"])
        noprop_results = ev.aggregate([
            ev.evaluate_qa(qa, noprop.retrieve(qa["question"], 5))
            for qa in corpus["qa_pairs"]
        ])

        # Full RFMv3
        full_results = run_backend(corpus, "rfmv3", 5)

        delta_f1  = full_results["overall"]["f1"] - noprop_results["overall"]["f1"]
        mhop_no   = hitk(noprop_results, "multihop")
        mhop_full = hitk(full_results,   "multihop")
        mhop_d    = mhop_full - mhop_no

        print(f"{size:>8}  {noprop_results['overall']['f1']:>16.3f}  "
              f"{full_results['overall']['f1']:>10.3f}  "
              f"{delta_f1:>+7.3f}  {mhop_no:>12.3f}  {mhop_full:>11.3f}  {mhop_d:>+11.3f}")

    print("\nInterpretation: positive delta = propagation helps; "
          "multihop delta shows chain-reasoning benefit specifically.")

# ────────────────────────────────────────────────────────────────────────────
# Exp 2 — Scaling
# ────────────────────────────────────────────────────────────────────────────

def make_scaled_corpus(n_persons: int, seed: int = 42) -> dict:
    """
    Generate a corpus with approximately n_persons * 5 notes
    (person facts + project facts + contradictions + aliases + chains).
    We patch the builder to support arbitrary sizes.
    """
    builder = CorpusBuilder(seed=seed)

    # Manually set builder parameters to match desired scale
    n_projects   = max(3, n_persons // 2)
    n_contras    = max(2, n_persons // 3)
    n_aliases    = max(2, n_persons // 4)
    n_chains     = max(2, n_persons // 4)

    from synthetic_corpus import (
        PERSON_TEMPLATES, PROJECT_TEMPLATES, ALIAS_SEEDS,
        MemoryNote, Contradiction, AliasGroup, MultiHopChain, QAPair, Corpus
    )
    from dataclasses import asdict

    # Extend person templates by cycling
    persons_extended = []
    base = list(PERSON_TEMPLATES)
    for i in range(n_persons):
        p = base[i % len(base)]
        # Make unique by appending index suffix
        name, role, city, tool, hobby = p
        if i >= len(base):
            name = f"{name.split()[0]}{i} {name.split()[1]}"
        persons_extended.append((name, role, city, tool, hobby))

    all_notes = []
    contradictions = []
    alias_groups = []
    chains = []
    qa_pairs = []

    person_note_map = {}
    for name, role, city, tool, hobby in persons_extended:
        notes = builder._person_notes(name, role, city, tool, hobby)
        all_notes.extend(notes)
        person_note_map[name] = notes[0]
        qa_pairs.append(builder._qa_direct(notes[0], f"Where is {name} based?", city))
        qa_pairs.append(builder._qa_direct(notes[0], f"What is {name}'s primary tool?", tool))

    projects = builder.rng.sample(PROJECT_TEMPLATES, min(n_projects, len(PROJECT_TEMPLATES)))
    for pname, ptype, deadline, scope, tech in projects:
        notes = builder._project_notes(pname, ptype, deadline, scope, tech)
        all_notes.extend(notes)
        qa_pairs.append(builder._qa_direct(notes[0], f"What is the target deadline for {pname}?", deadline))

    contra_seeds = [
        ("Alice Chen",  "location", "San Francisco", "Austin"),
        ("Bob Martinez","seniority", "senior", "junior"),
        ("Project Atlas","deadline", "Q3 2026", "Q4 2026"),
        ("Carol Singh",  "role",     "PM", "Engineering Manager"),
        ("Project Helios","scope",   "external", "internal"),
        ("David Kim",    "primary tool", "R", "Python"),
        ("Elena Petrov", "city",     "Seattle", "Portland"),
        ("Fiona Walsh",  "company size", "startup", "enterprise"),
        ("Project Nimbus","tech",    "WebAssembly", "V8-native"),
        ("George Nkosi", "timezone", "SAST", "UTC"),
    ]
    chosen = builder.rng.sample(contra_seeds, min(n_contras, len(contra_seeds)))
    for subj, field, va, vb in chosen:
        notes, contra = builder._contradiction_pair(subj, field, va, vb)
        all_notes.extend(notes)
        contradictions.append(contra)
        qa_pairs.append(builder._qa_contradiction(
            contra,
            f"There are two conflicting records about {subj}'s {field}. What are they?",
            f"Conflict: '{va}' vs '{vb}'. Canonical: {va}.",
        ))

    chosen_aliases = builder.rng.sample(ALIAS_SEEDS, min(n_aliases, len(ALIAS_SEEDS)))
    for canonical, aliases in chosen_aliases:
        base_ids = [n.id for n in all_notes if canonical.lower() in n.content.lower()][:2]
        alias_notes, group = builder._alias_group(canonical, aliases, base_ids)
        all_notes.extend(alias_notes)
        alias_groups.append(group)
        alias_query = builder.rng.choice(aliases[1:]) if len(aliases) > 1 else aliases[0]
        qa_pairs.append(builder._qa_alias(group, f'What entity is "{alias_query}" referring to?', canonical))

    chain_templates = [
        ([("Alice Chen reports to Fiona Walsh.", ""), ("Fiona Walsh is the CTO.", ""), ("The CTO owns the Solaris project.", "")],
         "Who owns Solaris and what is their relationship to Alice?",
         "Fiona Walsh owns Solaris. Alice reports to Fiona (CTO)."),
        ([("Project Atlas is owned by the Platform team.", ""), ("The Platform team lead is George Nkosi.", ""), ("George Nkosi is based in Johannesburg.", "")],
         "In which city is the owner of Project Atlas based?",
         "Johannesburg. Atlas → Platform team → George Nkosi."),
        ([("Helios uses a Python-first stack.", ""), ("Elena Petrov leads Helios.", ""), ("Elena Petrov's preferred tool is PyTorch.", "")],
         "What ML library does the Helios lead prefer?",
         "PyTorch. Elena leads Helios; PyTorch is her tool."),
        ([("Bob Martinez is the designer on Nimbus.", ""), ("Project Nimbus targets edge CDN.", ""), ("Edge CDN requires latency below 5ms.", "")],
         "What latency requirement applies to Bob's project?",
         "Below 5ms. Bob → Nimbus → edge CDN → 5ms."),
        ([("Carol Singh manages Orion.", ""), ("Orion is an external billing system.", ""), ("External billing must comply with PCI-DSS.", "")],
         "What compliance standard applies to Carol's project?",
         "PCI-DSS. Carol → Orion → external billing → PCI-DSS."),
        ([("Hannah Müller works on Vortex.", ""), ("Vortex is built on ClickHouse.", ""), ("ClickHouse clusters require columnar storage planning.", "")],
         "What storage consideration applies to Hannah's project?",
         "Columnar storage. Hannah → Vortex → ClickHouse."),
    ]
    chosen_chains = builder.rng.sample(chain_templates, min(n_chains, len(chain_templates)))
    for steps, question, answer in chosen_chains:
        notes, chain = builder._multihop_chain(steps, question, answer)
        all_notes.extend(notes)
        chains.append(chain)
        qa_pairs.append(builder._qa_multihop(chain))

    diff_dist = {}
    for qa in qa_pairs:
        diff_dist[qa.difficulty] = diff_dist.get(qa.difficulty, 0) + 1

    metadata = {
        "size": f"scaled_{len(all_notes)}",
        "seed": seed,
        "total_notes": len(all_notes),
        "total_qa_pairs": len(qa_pairs),
        "total_contradictions": len(contradictions),
        "total_alias_groups": len(alias_groups),
        "total_multihop_chains": len(chains),
        "difficulty_distribution": diff_dist,
    }

    from dataclasses import asdict as _asdict
    return {
        "metadata": metadata,
        "notes": [_asdict(n) for n in all_notes],
        "contradictions": [_asdict(c) for c in contradictions],
        "alias_groups": [_asdict(a) for a in alias_groups],
        "multihop_chains": [_asdict(c) for c in chains],
        "qa_pairs": [_asdict(q) for q in qa_pairs],
    }


def exp2_scaling():
    header("EXP 2 — Scaling: how do backends perform as corpus grows?")

    sizes = [("small",10), ("medium",18), ("large",25), ("xl",50), ("xxl",80)]

    print(f"\n{'notes':>6}  {'qa':>4}  {'bm25 F1':>8}  {'cos F1':>8}  {'rfm F1':>8}  "
          f"{'mhop bm25':>10}  {'mhop cos':>9}  {'mhop rfm':>9}")
    print("-" * 80)

    for label, n_persons in sizes:
        corpus = make_scaled_corpus(n_persons, seed=42)
        n = corpus["metadata"]["total_notes"]
        qa = corpus["metadata"]["total_qa_pairs"]

        r_bm25   = run_backend(corpus, "bm25",   5)
        r_cosine = run_backend(corpus, "cosine", 5)
        r_rfm    = run_backend(corpus, "rfmv3",  5)

        print(f"{n:>6}  {qa:>4}  "
              f"{f1(r_bm25):>8.3f}  {f1(r_cosine):>8.3f}  {f1(r_rfm):>8.3f}  "
              f"{hitk(r_bm25,'multihop'):>10.3f}  "
              f"{hitk(r_cosine,'multihop'):>9.3f}  "
              f"{hitk(r_rfm,'multihop'):>9.3f}")

    print("\nHypothesis: RFM multihop hit@k improves relative to cosine as N grows.")


# ────────────────────────────────────────────────────────────────────────────
# Exp 3 — Noise injection
# ────────────────────────────────────────────────────────────────────────────

NOISE_TEMPLATES = [
    "The quarterly review meeting is scheduled for next Tuesday.",
    "Please remember to submit your expense reports by end of month.",
    "The office kitchen will be closed for cleaning on Friday afternoon.",
    "New parking permits are available from the facilities desk.",
    "The fire drill has been rescheduled to Wednesday at 2pm.",
    "All staff are reminded to update their emergency contact information.",
    "The printer on floor 3 is currently out of service.",
    "Free coffee and bagels in the main conference room this morning.",
    "Please ensure all laptops are encrypted before the security audit.",
    "The annual company picnic will be held at Riverside Park in August.",
    "Network maintenance is scheduled for Saturday night from midnight to 4am.",
    "The new expense policy takes effect on the first of next month.",
    "Hot desking is now available on floors 2 through 5.",
    "Team offsite planning documents are due by Thursday.",
    "The library has updated its reference collection with new journals.",
    "Please keep the server room door closed at all times.",
    "Visitor badges must be returned to reception at the end of each day.",
    "The cafeteria will be serving a special menu for the holiday this week.",
    "All software license renewals should go through IT procurement.",
    "Meeting rooms can now be booked up to three weeks in advance.",
]

def inject_noise(corpus: dict, n_noise: int, seed: int = 99) -> dict:
    """Return a copy of corpus with n_noise irrelevant notes added."""
    import copy
    rng = random.Random(seed)
    corpus = copy.deepcopy(corpus)
    ctr = 9000
    for template in rng.sample(NOISE_TEMPLATES * (n_noise // len(NOISE_TEMPLATES) + 1), n_noise):
        ctr += 1
        corpus["notes"].append({
            "id": f"noise_{ctr:04d}",
            "content": template,
            "category": "fact",
            "tags": ["noise"],
        })
    corpus["metadata"]["total_notes"] = len(corpus["notes"])
    return corpus


def exp3_noise():
    header("EXP 3 — Noise injection: degradation as irrelevant notes are added")

    with open(RFM_DIR / "corpus_medium.json") as f:
        base_corpus = json.load(f)

    noise_levels = [0, 10, 25, 50, 100, 200]

    print(f"\n{'noise':>6}  {'total':>6}  {'bm25 F1':>8}  {'cos F1':>8}  {'rfm F1':>8}  "
          f"{'bm25 mhop':>10}  {'cos mhop':>9}  {'rfm mhop':>9}")
    print("-" * 80)

    for n_noise in noise_levels:
        corpus = inject_noise(base_corpus, n_noise)
        total = corpus["metadata"]["total_notes"]

        r_bm25   = run_backend(corpus, "bm25",   5)
        r_cosine = run_backend(corpus, "cosine", 5)
        r_rfm    = run_backend(corpus, "rfmv3",  5)

        print(f"{n_noise:>6}  {total:>6}  "
              f"{f1(r_bm25):>8.3f}  {f1(r_cosine):>8.3f}  {f1(r_rfm):>8.3f}  "
              f"{hitk(r_bm25,'multihop'):>10.3f}  "
              f"{hitk(r_cosine,'multihop'):>9.3f}  "
              f"{hitk(r_rfm,'multihop'):>9.3f}")

    print("\nHypothesis: RFM stability weighting degrades more gracefully than BM25 TF-IDF under noise.")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3],
                        help="Run specific experiment (default: all)")
    args = parser.parse_args()

    import os
    os.environ.setdefault("PYTHONHASHSEED", "0")

    run_all = args.exp is None
    if run_all or args.exp == 1:
        exp1_ablation()
    if run_all or args.exp == 2:
        exp2_scaling()
    if run_all or args.exp == 3:
        exp3_noise()


if __name__ == "__main__":
    main()
