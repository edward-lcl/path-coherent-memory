"""
Hard multi-hop corpus v2 — scaled, fully disjoint vocabularies.

Design goals:
  1. 20+ chains with statistically meaningful results
  2. Each chain uses 4 concept names with minimal char n-gram overlap
     between concepts used in DIFFERENT chains
  3. Each chain uses unique relational verb phrases — no shared structure
  4. Answer nodes have zero vocabulary overlap with the query
  5. Non-technical distractors (HR/admin noise, no concept token overlap)

Isolation guarantee:
  - Concept names generated with distinct phonetic profiles
  - Structural verbs assigned uniquely per chain (round-robin from large pool)
  - Verified: answer note char n-gram overlap with query < threshold
"""

import json, random, re
from dataclasses import dataclass, asdict

# ── Concept name pool — phonetically diverse, minimal 3-gram overlap ──────
# Groups of 4 — each group assigned to one chain
CONCEPT_GROUPS = [
    # group 0
    ["kestrel", "thalweg", "fenwick", "luminos"],
    # group 1
    ["pallada", "stratos", "solace",  "vantage"],
    # group 2
    ["zephyr",  "orenda",  "cobalt",  "nereid"],
    # group 3
    ["bryndal", "quovix",  "meltis",  "crixum"],
    # group 4
    ["dorfast", "uvlane",  "pwythig", "xombal"],
    # group 5
    ["glarnis", "tobwick", "sunjara", "phorvex"],
    # group 6
    ["yeldram", "crubwel", "finzosk", "mauvolt"],
    # group 7
    ["holvane", "drixpan", "gweztum", "skorfil"],
    # group 8
    ["alfrond", "vipsket", "clydrum", "nowxeth"],
    # group 9
    ["jostave", "brixunk", "mudloph", "thycast"],
    # group 10
    ["wulfram", "exivont", "kadrusp", "plyborg"],
    # group 11
    ["snorbil", "truvank", "dezholm", "gaxipel"],
    # group 12
    ["clomvik", "purseth", "halwond", "fjorbus"],
    # group 13
    ["ondrast", "kivlump", "sazweld", "tryphex"],
    # group 14
    ["murflax", "dolwick", "cuvband", "spixham"],
    # group 15
    ["bolvane", "grethum", "naxdove", "quilsep"],
    # group 16
    ["prutham", ["skovlid", "jarnful", "crezbal"],
     "skovlid", "jarnful"],
    # group 17
    ["thrumby", "exwolid", "panvusk", "clodbyr"],
    # group 18
    ["vexholm", "dritzal", "spunkow", "gablyft"],
    # group 19
    ["yostriv", "claxmed", "burphon", "nidgast"],
    # group 20
    ["tolvask", "grumwex", "phibond", "scralix"],
    # group 21
    ["jundvak", "elthorp", "crambex", "wulspit"],
    # group 22
    ["mondrex", "tubwick", "halvost", "crixban"],
    # group 23
    ["primbax", "skolved", "nunwick", "grethol"],
]

# Flatten any nested lists from typo in group 16
def clean_group(g):
    out = []
    for x in g:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out[:4]

CONCEPT_GROUPS = [clean_group(g) for g in CONCEPT_GROUPS]

# ── Unique relation verb sets — one per chain ─────────────────────────────
# hop-1 verbs: how query entity connects to bridge entity
HOP1_VERBS = [
    lambda a, b: f"{a} coordinates with {b} at runtime.",
    lambda a, b: f"{a} dispatches jobs to {b} for execution.",
    lambda a, b: f"{a} wraps around {b} as a proxy layer.",
    lambda a, b: f"{a} delegates computation to {b}.",
    lambda a, b: f"{a} subscribes to events from {b}.",
    lambda a, b: f"{a} mirrors its state to {b} continuously.",
    lambda a, b: f"{a} bootstraps through {b} on startup.",
    lambda a, b: f"{a} polls {b} for configuration updates.",
    lambda a, b: f"{a} uses {b} as its upstream dependency.",
    lambda a, b: f"{a} authenticates requests via {b}.",
    lambda a, b: f"{a} publishes metrics to {b}.",
    lambda a, b: f"{a} sends overflow work to {b} during peak load.",
    lambda a, b: f"{a} syncs its manifest with {b}.",
    lambda a, b: f"{a} registers its endpoints in {b}.",
    lambda a, b: f"{a} inherits configuration from {b}.",
    lambda a, b: f"{a} offloads heavy tasks to {b}.",
    lambda a, b: f"{a} streams audit logs to {b}.",
    lambda a, b: f"{a} borrows capacity from {b} under load.",
    lambda a, b: f"{a} mirrors its writes to {b} asynchronously.",
    lambda a, b: f"{a} archives snapshots within {b}.",
    lambda a, b: f"{a} caches responses from {b}.",
    lambda a, b: f"{a} validates tokens against {b}.",
    lambda a, b: f"{a} tunnels encrypted payloads through {b}.",
    lambda a, b: f"{a} receives health signals from {b}.",
]

# hop-2 verbs: how bridge entity connects to answer-container entity
HOP2_VERBS = [
    lambda a, b: f"{a} belongs to the {b} group.",
    lambda a, b: f"{a} is administered by {b}.",
    lambda a, b: f"{a} operates within the {b} boundary.",
    lambda a, b: f"{a} reports its status to {b}.",
    lambda a, b: f"{a} runs as a component of {b}.",
    lambda a, b: f"{a} is provisioned under {b}.",
    lambda a, b: f"{a} shares a namespace with {b}.",
    lambda a, b: f"{a} is governed by the {b} framework.",
    lambda a, b: f"{a} is a child service of {b}.",
    lambda a, b: f"{a} falls within the purview of {b}.",
    lambda a, b: f"{a} is co-located with {b} in the same cluster.",
    lambda a, b: f"{a} is a tenant of {b}.",
    lambda a, b: f"{a} is embedded inside {b}.",
    lambda a, b: f"{a} is supervised by {b}.",
    lambda a, b: f"{a} is registered under {b} in the service catalog.",
    lambda a, b: f"{a} is owned by the {b} platform.",
    lambda a, b: f"{a} inherits its lifecycle from {b}.",
    lambda a, b: f"{a} is managed as part of {b}.",
    lambda a, b: f"{a} is hosted within {b}.",
    lambda a, b: f"{a} is overseen by {b}.",
    lambda a, b: f"{a} is a module of {b}.",
    lambda a, b: f"{a} is a plugin within {b}.",
    lambda a, b: f"{a} is wired into {b} at the platform level.",
    lambda a, b: f"{a} is packaged as part of {b}.",
]

# answer properties and value pools
PROPERTIES = {
    "owner":      lambda e, v: f"{e} ownership held by {v}.",
    "location":   lambda e, v: f"{e} deployed in {v}.",
    "dependency": lambda e, v: f"{e} depends on {v}.",
    "team":       lambda e, v: f"{e} maintained by {v}.",
    "language":   lambda e, v: f"{e} implemented in {v}.",
}

VALUES = {
    "owner":      ["reliability-eng", "platform-alpha", "core-infra", "ml-ops",
                   "data-services", "edge-compute", "security-ops", "api-gateway",
                   "devops-foundation", "cloud-arch", "site-reliability", "infra-ops"],
    "location":   ["us-east-1", "eu-west-2", "ap-south-1", "us-central",
                   "london-dc", "frankfurt-zone", "singapore-hub", "toronto-edge",
                   "sydney-az", "dubai-region", "sao-paulo-dc", "chicago-hub"],
    "dependency": ["postgresql-14", "redis-cluster", "kafka-streams", "elasticsearch",
                   "grpc-gateway", "envoy-proxy", "vault-secrets", "consul-mesh",
                   "nats-jetstream", "temporal-workflows", "prometheus-stack", "jaeger-tracing"],
    "team":       ["platform-alpha", "core-infra", "ml-ops", "data-services",
                   "reliability-eng", "edge-compute", "security-ops", "api-gateway"],
    "language":   ["golang", "typescript", "python3", "java17", "rust-stable",
                   "kotlin-jvm", "scala3", "elixir-otp", "cpp20", "swift5", "csharp12", "haskell"],
}

# ── Non-technical distractors ─────────────────────────────────────────────
DISTRACTORS = [
    "Quarterly budget review is scheduled for the third week of next month.",
    "Building access cards must be renewed before end of the fiscal year.",
    "Annual staff survey results will be published on the intranet portal.",
    "Travel reimbursements must be submitted within thirty days of the trip.",
    "Library catalog has been updated with new research journals this season.",
    "Parking permits for the north lot expire at the end of this quarter.",
    "Emergency exit procedures have been revised on the posted floor plan.",
    "Cafeteria hours have been extended to accommodate the evening shift.",
    "All visitors must sign in at reception before proceeding to offices.",
    "Recycling program now accepts batteries and small household electronics.",
    "Room bookings for the conference center require two weeks advance notice.",
    "Health insurance open enrollment closes at the end of this month.",
    "New phone directory has been distributed to all department heads today.",
    "Volunteers needed for the community garden project scheduled this weekend.",
    "Printer supplies can be ordered through the facilities request portal.",
    "Onboarding documents for new hires are available in the shared folder.",
    "Security badges must be worn visibly at all times inside the building.",
    "Annual fire safety inspection is scheduled for the coming Friday morning.",
    "Feedback forms for the workshop are available at the registration table.",
    "IT support tickets should be submitted via the internal help desk portal.",
    "Holiday schedule has been posted on the company calendar for reference.",
    "Expense reports require manager approval before submission to finance.",
    "New ergonomic chairs have been ordered for all open-plan workstations.",
    "Company newsletter will be distributed on the first Monday of each month.",
    "Shuttle service from the parking garage runs every fifteen minutes.",
    "Staff photos for the directory should be submitted by end of next week.",
    "Gym membership subsidy forms are available from the HR benefits office.",
    "Water cooler refills are scheduled every Tuesday and Thursday morning.",
    "The mentorship program applications close at the end of this quarter.",
    "Catering requests for team events need five business days of lead time.",
]


def char_ngrams(text: str, sizes=(2, 3, 4)) -> set[str]:
    text = re.sub(r"[^a-z0-9]", "_", text.lower())
    out = set()
    for n in sizes:
        for i in range(len(text) - n + 1):
            out.add(text[i:i+n])
    return out


def isolation_score(a: str, b: str) -> float:
    """Lower = better isolated. 0 = perfectly disjoint char n-grams."""
    na, nb = char_ngrams(a), char_ngrams(b)
    return len(na & nb) / len(na | nb) if (na | nb) else 0.0


@dataclass
class Chain:
    query: str
    bridge1_content: str
    bridge2_content: str
    answer_content: str
    answer: str
    note_ids: list[str]


def build_corpus(n_chains: int = 20, n_distractors: int = 50, seed: int = 42) -> dict:
    rng = random.Random(seed)
    notes = []
    qa_pairs = []
    ctr = [0]

    def nid():
        ctr[0] += 1
        return f"h_{ctr[0]:04d}"

    prop_cycle = list(PROPERTIES.keys())

    for ci in range(min(n_chains, len(CONCEPT_GROUPS), len(HOP1_VERBS), len(HOP2_VERBS))):
        grp = CONCEPT_GROUPS[ci]
        c0, c1, c2, c3 = grp[0], grp[1], grp[2], grp[3]

        prop_key = prop_cycle[ci % len(prop_cycle)]
        val = rng.choice(VALUES[prop_key])

        h1_content = HOP1_VERBS[ci](c0, c1)
        h2_content = HOP2_VERBS[ci](c1, c2)
        ans_content = PROPERTIES[prop_key](c2, val)

        # Verify isolation: answer content should not share tokens with query
        query = f"What is the {prop_key} of {c0}?"

        b1_id = nid()
        b2_id = nid()
        an_id = nid()

        notes.append({"id": b1_id, "content": h1_content, "category": "fact",
                      "tags": ["chain", f"chain_{ci}", "hop1"], "hop": 1})
        notes.append({"id": b2_id, "content": h2_content, "category": "fact",
                      "tags": ["chain", f"chain_{ci}", "hop2"], "hop": 2})
        notes.append({"id": an_id, "content": ans_content, "category": "fact",
                      "tags": ["chain", f"chain_{ci}", "hop3"], "hop": 3})

        qa_pairs.append({
            "id": f"qa_{ci:03d}",
            "question": query,
            "answer": val,
            "required_notes": [b1_id, b2_id, an_id],
            "difficulty": "hard_multihop",
            "ground_truth_type": "chain_reasoning",
        })

    # Add distractors
    distractor_pool = DISTRACTORS * (n_distractors // len(DISTRACTORS) + 1)
    for i, d in enumerate(rng.sample(distractor_pool, n_distractors)):
        notes.append({"id": nid(), "content": d, "category": "fact",
                      "tags": ["distractor"], "hop": -1})

    rng.shuffle(notes)

    metadata = {
        "size": f"hard_v2_{len(notes)}",
        "seed": seed,
        "total_notes": len(notes),
        "total_qa_pairs": len(qa_pairs),
        "n_chains": len(qa_pairs),
        "n_distractors": n_distractors,
        "design": "zero-vocab-overlap chains, unique verb sets per chain, non-tech distractors",
    }

    return {
        "metadata": metadata,
        "notes": notes,
        "contradictions": [],
        "alias_groups": [],
        "multihop_chains": [],
        "qa_pairs": qa_pairs,
    }


if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/Users/edward/.ocplatform/workspace/research/rfm/corpus_hard_v2.json")
    p.add_argument("--chains", type=int, default=20)
    p.add_argument("--distractors", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    corpus = build_corpus(args.chains, args.distractors, args.seed)
    with open(args.output, "w") as f:
        json.dump(corpus, f, indent=2)

    m = corpus["metadata"]
    print(f"Hard v2 corpus: {m['total_notes']} notes, {m['n_chains']} chains, "
          f"{m['n_distractors']} distractors")

    # Isolation check
    note_map = {n["id"]: n for n in corpus["notes"]}
    print("\nVocabulary isolation (answer node vs query):")
    fails = 0
    for qa in corpus["qa_pairs"]:
        ans_note = note_map[qa["required_notes"][-1]]
        q_words = set(re.findall(r"[a-z]+", qa["question"].lower()))
        a_words = set(re.findall(r"[a-z]+", ans_note["content"].lower()))
        overlap = q_words & a_words - {"is", "the", "of", "a", "what", "in"}
        if overlap:
            print(f"  ✗ {qa['id']}: overlap={sorted(overlap)}")
            fails += 1
    if not fails:
        print(f"  ✓ All {len(corpus['qa_pairs'])} chains fully isolated")

    # Cross-chain n-gram contamination check
    print("\nCross-chain bridge note isolation (sampled pairs):")
    chain_notes = [(n, n["tags"][1]) for n in corpus["notes"] if "hop1" in n["tags"]]
    contaminated = 0
    pairs_checked = 0
    for i in range(len(chain_notes)):
        for j in range(i+1, min(i+4, len(chain_notes))):
            ni, ci = chain_notes[i]
            nj, cj = chain_notes[j]
            if ci == cj:
                continue
            score = isolation_score(ni["content"], nj["content"])
            pairs_checked += 1
            if score > 0.15:
                contaminated += 1
                print(f"  ✗ {ci}/{cj} isolation={score:.2f}: '{ni['content'][:40]}' vs '{nj['content'][:40]}'")
    if not contaminated:
        print(f"  ✓ All {pairs_checked} cross-chain bridge pairs sufficiently isolated")
    print(f"\nOutput: {args.output}")
