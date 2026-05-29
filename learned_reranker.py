"""
Learned terminal reranker — structural features only.

For each chain, path_v12 produces a terminal candidate pool.
We extract structural features of each (anchor, path, terminal) triple
and train a binary classifier: is this candidate the gold terminal?

Features (all structural, no content):
  - overlap_with_anchor: token overlap between terminal and anchor (0.0 for good terminals)
  - n_paths: number of independent bridge routes that reach this terminal
  - min_bridge_df: minimum df of any bridge token used to reach this terminal (rarity)
  - avg_bridge_df: average df of bridge tokens
  - min_bridge_cross_sources: min number of source files sharing any bridge token
  - terminal_n_toks: token count of terminal note
  - terminal_source_tier: tier of terminal source (4=topic, 3=summary, 2=dated, 1=other)
  - anchor_best_rank: BM25 rank of best anchor that produced a path to this terminal
  - n_distinct_anchors: how many distinct anchors produced paths to this terminal
  - n_distinct_bridge1: how many distinct bridge1 tokens reached this terminal
"""
import json, re, math, sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path("/Users/edward/.ocplatform/workspace/research/rfm")
sys.path.insert(0, "/tmp")
import rfm_substrate_path_test as substrate

# ── tokenization ──────────────────────────────────────────────────────────────
STOP = set("able accepted active actual additional basic better clean clear common correct "
    "current different direct enough exact external final first full general good great hard "
    "high human important initial known large latest likely live local long main major "
    "meaningful messy minimal native new next obvious old ongoing only operational other "
    "personal possible primary prior raw real recent related relevant same second separate "
    "simple small specific stable strong sure technical top true useful weak whole system "
    "memory file data user note output text block token result type name status created "
    "updated source boundary tags evidence confidence tier person organization entity".split())

def toks(text): return [w.lower() for w in re.findall(r'[a-z]{4,}', text.lower())]
def tok_ok(t): return t not in STOP and 5 <= len(t) <= 20 and not t[:4].isdigit()

# ── source tier ───────────────────────────────────────────────────────────────
def source_tier(src):
    if "topics/" in src: return 4
    if "summaries/" in src or "summary" in src.lower(): return 3
    if re.search(r'\d{4}-\d{2}-\d{2}', src): return 2
    return 1

# ── corpus + index ────────────────────────────────────────────────────────────
print("Loading substrate...")
notes_raw = substrate.load_notes()
notes = [{**n, "_toks": set(t for t in toks(n["content"]) if tok_ok(t))}
         for n in notes_raw if len([t for t in toks(n["content"]) if tok_ok(t)]) >= 6]
nb = {n["id"]: n for n in notes}
N = len(notes)
print(f"  {N} notes")

df = defaultdict(int)
postings = defaultdict(list)
token_sources = defaultdict(set)
for n in notes:
    for t in n["_toks"]: df[t] += 1
for n in notes:
    for t in n["_toks"]:
        postings[t].append(n["id"])
        token_sources[t].add(n["source"])

def bm25_retrieve(qt, top_k=10):
    scores = {}
    for t in set(qt):
        if t not in postings: continue
        idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
        for nid in postings[t]: scores[nid] = scores.get(nid, 0) + idf
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

def candidate_bridges(node_id, exclude, top_k=2):
    node_src = nb[node_id]["source"]
    scored = []
    for t in nb[node_id]["_toks"]:
        if t in exclude: continue
        cross = token_sources.get(t, set()) - {node_src}
        n_cross = len(cross)
        if n_cross == 0 or n_cross > 10: continue
        score = (1.0/df.get(t,1)**0.5) * (1.0/n_cross) * min(len(t),12)*0.1
        scored.append((score, t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:top_k]]

# ── path traversal with feature collection ────────────────────────────────────
def collect_terminal_features(query, anchor_k=10, bridge_k=2, branch_k=8):
    """
    Returns dict: terminal_id -> feature_dict
    Also returns anchor_toks for overlap computation.
    """
    qt = set(t for t in toks(query) if tok_ok(t))
    anchor_ids = bm25_retrieve(list(qt), top_k=anchor_k)

    # terminal_id -> accumulated feature info across all paths
    terminal_info = defaultdict(lambda: {
        "n_paths": 0,
        "bridge_dfs": [],
        "bridge_cross_sources": [],
        "anchor_ranks": [],
        "distinct_anchors": set(),
        "distinct_bridge1": set(),
    })
    anchor_toks = set()
    for a_id in anchor_ids:
        if a_id in nb: anchor_toks |= nb[a_id]["_toks"]

    for rank, a_id in enumerate(anchor_ids):
        if a_id not in nb: continue
        a_toks = nb[a_id]["_toks"]

        for t1 in candidate_bridges(a_id, qt, bridge_k):
            for b_id in postings.get(t1, [])[:branch_k]:
                if b_id == a_id or b_id not in nb: continue
                for t2 in candidate_bridges(b_id, qt | {t1}, bridge_k):
                    for c_id in postings.get(t2, [])[:branch_k]:
                        if c_id in {a_id, b_id} or c_id not in nb: continue
                        if len(nb[c_id]["_toks"] & a_toks) > 0: continue  # zero-overlap filter

                        info = terminal_info[c_id]
                        info["n_paths"] += 1
                        info["bridge_dfs"].append(df.get(t1, 1))
                        info["bridge_dfs"].append(df.get(t2, 1))
                        info["bridge_cross_sources"].append(len(token_sources.get(t1, set())))
                        info["bridge_cross_sources"].append(len(token_sources.get(t2, set())))
                        info["anchor_ranks"].append(rank)
                        info["distinct_anchors"].add(a_id)
                        info["distinct_bridge1"].add(t1)

    # Build feature vectors
    features = {}
    anchor_toks_set = anchor_toks  # union of all anchor token sets
    for c_id, info in terminal_info.items():
        if c_id not in nb: continue
        c_toks = nb[c_id]["_toks"]
        overlap = len(c_toks & anchor_toks_set) / max(len(c_toks), 1)
        bdfs = info["bridge_dfs"] or [N]
        bcs = info["bridge_cross_sources"] or [1]
        features[c_id] = {
            "overlap_with_anchor": overlap,
            "n_paths": info["n_paths"],
            "min_bridge_df": min(bdfs),
            "avg_bridge_df": sum(bdfs) / len(bdfs),
            "min_bridge_cross_sources": min(bcs),
            "terminal_n_toks": len(c_toks),
            "terminal_source_tier": source_tier(nb[c_id]["source"]),
            "anchor_best_rank": min(info["anchor_ranks"]),
            "n_distinct_anchors": len(info["distinct_anchors"]),
            "n_distinct_bridge1": len(info["distinct_bridge1"]),
        }
    return features

FEATURE_NAMES = [
    "overlap_with_anchor", "n_paths", "min_bridge_df", "avg_bridge_df",
    "min_bridge_cross_sources", "terminal_n_toks", "terminal_source_tier",
    "anchor_best_rank", "n_distinct_anchors", "n_distinct_bridge1",
]

def feats_to_vec(f):
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)

# ── build training dataset ────────────────────────────────────────────────────
judge_file = ROOT / "levi_semantic_chain_omlx_judge_v1.jsonl"
cand_file = ROOT / "levi_semantic_chain_candidates_v1.jsonl"
judge = {int(json.loads(l)["idx"]): json.loads(l)["label"]
         for l in judge_file.read_text().splitlines() if l}
candidates = [json.loads(l) for l in cand_file.read_text().splitlines() if l]
real_chains = [c for c in candidates if judge.get(int(c["idx"])) == "real_semantic"]
print(f"\nBuilding features for {len(real_chains)} real_semantic chains...")

chain_data = []  # list of (chain_idx, gold_terminal_id, features_dict)
for i, c in enumerate(real_chains):
    gold_c = c["required_ids"][2]
    feats = collect_terminal_features(c["start_token"])
    if feats:
        chain_data.append((int(c["idx"]), gold_c, feats))
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(real_chains)} chains processed")

print(f"  Done. {len(chain_data)} chains with non-empty pools")
gold_in_pool = sum(1 for _, gold, feats in chain_data if gold in feats)
print(f"  Gold terminal in pool: {gold_in_pool}/{len(chain_data)} ({100*gold_in_pool/len(chain_data):.1f}%)")

# ── leave-one-chain-out cross-validation ─────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

print("\n=== Leave-one-chain-out CV ===")

results = {"v12_baseline": [], "lr_reranker": [], "gb_reranker": []}

for test_idx, (chain_idx, gold_c, test_feats) in enumerate(chain_data):
    if gold_c not in test_feats:
        results["v12_baseline"].append(0)
        results["lr_reranker"].append(0)
        results["gb_reranker"].append(0)
        continue

    # Build train set from all other chains
    X_train, y_train = [], []
    for j, (_, gold_j, feats_j) in enumerate(chain_data):
        if j == test_idx: continue
        for cid, fv in feats_j.items():
            X_train.append(feats_to_vec(fv))
            y_train.append(1 if cid == gold_j else 0)

    if sum(y_train) == 0:
        results["v12_baseline"].append(0)
        results["lr_reranker"].append(0)
        results["gb_reranker"].append(0)
        continue

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Logistic regression
    lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
    lr.fit(X_train_s, y_train)

    # Gradient boosting
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)

    # Score test pool
    cids = list(test_feats.keys())
    X_test = np.array([feats_to_vec(test_feats[c]) for c in cids])
    X_test_s = scaler.transform(X_test)

    lr_scores = lr.predict_proba(X_test_s)[:, 1]
    gb_scores = gb.predict_proba(X_test)[:, 1]

    # v12 baseline: order by n_paths desc (original ranking proxy)
    v12_scores = np.array([test_feats[c]["n_paths"] + (1.0 - test_feats[c]["anchor_best_rank"] * 0.05) for c in cids])

    def hit_at_k(scores, gold, k=10):
        ranked = [cids[i] for i in np.argsort(-scores)][:k]
        return int(gold in ranked)

    results["v12_baseline"].append(hit_at_k(v12_scores, gold_c))
    results["lr_reranker"].append(hit_at_k(lr_scores, gold_c))
    results["gb_reranker"].append(hit_at_k(gb_scores, gold_c))

    if (test_idx + 1) % 20 == 0:
        lr_so_far = sum(results["lr_reranker"]) / len(results["lr_reranker"])
        print(f"  {test_idx+1}/{len(chain_data)} — LR hit@10 so far: {lr_so_far:.3f}")

n = len(chain_data)
print(f"\n=== Results (top_k=10, n={n} chains with non-empty pools) ===")
for name, hits in results.items():
    print(f"  {name:<20} hit@10 = {sum(hits)}/{n} = {100*sum(hits)/n:.1f}%")

# ── feature importance (from GB on full dataset) ──────────────────────────────
print("\n=== Feature importance (GB, full dataset) ===")
X_all, y_all = [], []
for _, gold_j, feats_j in chain_data:
    for cid, fv in feats_j.items():
        X_all.append(feats_to_vec(fv))
        y_all.append(1 if cid == gold_j else 0)
X_all = np.array(X_all); y_all = np.array(y_all)
gb_full = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
gb_full.fit(X_all, y_all)
importances = sorted(zip(FEATURE_NAMES, gb_full.feature_importances_), key=lambda x: -x[1])
for feat, imp in importances:
    print(f"  {feat:<35} {imp:.4f}")
