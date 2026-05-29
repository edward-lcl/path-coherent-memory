#!/usr/bin/env python3
"""
Learned terminal reranker v2 — trained on Levi calibration + v4 judged chains.

Uses 47 real_semantic calibration chains + 70 real_semantic v4 chains = 117 chains.
Evaluated with leave-one-chain-out cross-validation.

Goal: close the gap between path-v12 (72.7% on Talos) and oracle ceiling (88.5%).
This runs on the Levi corpus as a proxy; Talos performance will differ due to
corpus structure but the re-ranker architecture is portable.
"""
import json, re, math, sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path("/Users/edward/.ocplatform/workspace/research/rfm")
sys.path.insert(0, "/tmp")
import rfm_substrate_path_test as substrate

STOP = set("""able accepted active actual additional adjacent basic better clean clear common
correct current different direct enough exact external final first full general good great hard
high human important initial known large latest likely live local long main major meaningful
minimal native new next obvious old ongoing only operational other personal possible primary
prior raw real recent related relevant same second separate simple small specific stable strong
sure technical top true useful weak whole system memory file data user note output text block
token result type name status created updated source boundary tags evidence confidence tier
person organization entity service endpoint project work time week day month year back going
make makes made used using been have will would could should also just like much very some
more most than from this that with they them their about into over under when where which
while""".split())

def toks(text): return [w.lower() for w in re.findall(r'[a-z]{4,}', text.lower())]
def tok_ok(t): return t not in STOP and 5 <= len(t) <= 20 and not t[:4].isdigit()

def source_tier(src):
    if src in {"USER.md", "MEMORY.md", "SOUL.md"}: return 4
    if src.startswith("memory/topics/"): return 4
    if src.startswith("memory/summaries/"): return 3
    if re.match(r'memory/\d{4}-\d{2}-\d{2}\.md$', src): return 2
    if src.startswith("memory/"): return 1
    return 0

def is_session(src): return bool(re.match(r'memory/\d{4}-\d{2}-\d{2}\.md$', src))
def is_topic(src): return src.startswith("memory/topics/")

print("Loading substrate...")
notes_raw = substrate.load_notes()
notes = []
for n in notes_raw:
    ts = set(t for t in toks(n["content"]) if tok_ok(t))
    if len(ts) >= 6:
        notes.append({**n, "_toks": ts})
nb = {n["id"]: n for n in notes}
N = len(notes)
print(f"  {N} notes loaded")

df = defaultdict(int)
postings = defaultdict(list)
token_sources = defaultdict(set)
idf_cache = {}

for n in notes:
    for t in n["_toks"]:
        df[t] += 1
        postings[t].append(n["id"])
        token_sources[t].add(n["source"])

def get_idf(t):
    if t not in idf_cache:
        idf_cache[t] = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
    return idf_cache[t]

k1, bv = 1.5, 0.75
avgdl = sum(len(n["_toks"]) for n in notes) / N

def bm25_retrieve(qt, top_k=10):
    scores = {}
    for t in set(qt):
        if t not in postings: continue
        for nid in postings[t]:
            dl = len(nb[nid]["_toks"])
            tf_num = (k1 + 1)
            tf_den = 1 + k1 * (1 - bv + bv * dl / avgdl)
            scores[nid] = scores.get(nid, 0) + get_idf(t) * tf_num / tf_den
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

def bridge_score(t, node_src, qt):
    if t in qt or len(t) < 5: return 0.0
    cross = token_sources.get(t, set()) - {node_src}
    n_cross = len(cross)
    if n_cross == 0 or n_cross > 10: return 0.0
    return (1.0 / df.get(t, 1) ** 0.5) * (1.0 / n_cross) * min(len(t), 12) * 0.1

def candidate_bridges(node_id, exclude, top_k=2):
    src = nb[node_id]["source"]
    scored = [(bridge_score(t, src, exclude), t)
              for t in nb[node_id]["_toks"] if t not in exclude]
    scored.sort(reverse=True)
    return [t for sc, t in scored[:top_k] if sc > 0]


def collect_terminal_features(query, anchor_k=10, bridge_k=2, branch_k=8):
    qt = set(t for t in toks(query) if tok_ok(t))
    anchor_ids = bm25_retrieve(list(qt), top_k=anchor_k)
    anchor_toks_union = set()
    for a_id in anchor_ids:
        if a_id in nb:
            anchor_toks_union |= nb[a_id]["_toks"]

    terminal_info = defaultdict(lambda: {
        "n_paths": 0,
        "bridge_dfs": [],
        "bridge_cross_sources": [],
        "anchor_ranks": [],
        "distinct_anchors": set(),
        "distinct_bridge1": set(),
        "distinct_bridge2": set(),
        "min_bridge_score": [],
        "path_anchor_tier": [],
    })

    for rank, a_id in enumerate(anchor_ids):
        if a_id not in nb: continue
        a_toks = nb[a_id]["_toks"]
        a_tier = source_tier(nb[a_id]["source"])

        for t1 in candidate_bridges(a_id, qt, bridge_k):
            bs1 = bridge_score(t1, nb[a_id]["source"], qt)
            for b_id in postings.get(t1, [])[:branch_k]:
                if b_id == a_id or b_id not in nb: continue
                if nb[b_id]["source"] == nb[a_id]["source"]: continue

                for t2 in candidate_bridges(b_id, qt | {t1}, bridge_k):
                    bs2 = bridge_score(t2, nb[b_id]["source"], qt | {t1})
                    for c_id in postings.get(t2, [])[:branch_k]:
                        if c_id in {a_id, b_id} or c_id not in nb: continue
                        if nb[c_id]["source"] in {nb[a_id]["source"], nb[b_id]["source"]}: continue
                        if len(nb[c_id]["_toks"] & a_toks) > 0: continue

                        info = terminal_info[c_id]
                        info["n_paths"] += 1
                        info["bridge_dfs"] += [df.get(t1, 1), df.get(t2, 1)]
                        info["bridge_cross_sources"] += [
                            len(token_sources.get(t1, set())),
                            len(token_sources.get(t2, set()))
                        ]
                        info["anchor_ranks"].append(rank)
                        info["distinct_anchors"].add(a_id)
                        info["distinct_bridge1"].add(t1)
                        info["distinct_bridge2"].add(t2)
                        info["min_bridge_score"].append(min(bs1, bs2))
                        info["path_anchor_tier"].append(a_tier)

    features = {}
    for c_id, info in terminal_info.items():
        if c_id not in nb: continue
        c_toks = nb[c_id]["_toks"]
        c_src = nb[c_id]["source"]
        overlap = len(c_toks & anchor_toks_union) / max(len(c_toks), 1)
        bdfs = info["bridge_dfs"] or [N]
        bcs = info["bridge_cross_sources"] or [1]
        bsc = info["min_bridge_score"] or [0.0]
        at = info["path_anchor_tier"] or [1]
        features[c_id] = {
            "overlap_with_anchor": overlap,
            "n_paths": info["n_paths"],
            "log_n_paths": math.log1p(info["n_paths"]),
            "min_bridge_df": min(bdfs),
            "avg_bridge_df": sum(bdfs) / len(bdfs),
            "max_bridge_df": max(bdfs),
            "min_bridge_cross_sources": min(bcs),
            "avg_bridge_cross_sources": sum(bcs) / len(bcs),
            "avg_bridge_score": sum(bsc) / len(bsc),
            "terminal_n_toks": len(c_toks),
            "terminal_source_tier": source_tier(c_src),
            "terminal_is_session": int(is_session(c_src)),
            "terminal_is_topic": int(is_topic(c_src)),
            "anchor_best_rank": min(info["anchor_ranks"]),
            "anchor_avg_rank": sum(info["anchor_ranks"]) / len(info["anchor_ranks"]),
            "n_distinct_anchors": len(info["distinct_anchors"]),
            "n_distinct_bridge1": len(info["distinct_bridge1"]),
            "n_distinct_bridge2": len(info["distinct_bridge2"]),
            "avg_path_anchor_tier": sum(at) / len(at),
        }
    return features


FEATURE_NAMES = [
    "overlap_with_anchor", "n_paths", "log_n_paths",
    "min_bridge_df", "avg_bridge_df", "max_bridge_df",
    "min_bridge_cross_sources", "avg_bridge_cross_sources", "avg_bridge_score",
    "terminal_n_toks", "terminal_source_tier", "terminal_is_session", "terminal_is_topic",
    "anchor_best_rank", "anchor_avg_rank", "n_distinct_anchors",
    "n_distinct_bridge1", "n_distinct_bridge2", "avg_path_anchor_tier",
]

def feats_to_vec(f):
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)


# ── load all training chains ──────────────────────────────────────────────────
def load_judged_chains(candidates_path, judge_path, label="real_semantic"):
    candidates = [json.loads(l) for l in Path(candidates_path).read_text().splitlines() if l.strip()]
    judge = {int(json.loads(l)["idx"]): json.loads(l)["label"]
             for l in Path(judge_path).read_text().splitlines() if l.strip()}
    return [c for c in candidates if judge.get(int(c["idx"])) == label]

print("\nLoading training chains...")
calib_chains = load_judged_chains(
    ROOT / "levi_calibration_candidates_v1.jsonl",
    ROOT / "levi_calibration_judged_v1.jsonl",
)
v4_chains = load_judged_chains(
    ROOT / "levi_semantic_chain_candidates_v4.jsonl",
    ROOT / "levi_semantic_chain_omlx_judge_v4.jsonl",
)
all_chains = calib_chains + v4_chains
print(f"  Calibration real_semantic: {len(calib_chains)}")
print(f"  V4 real_semantic: {len(v4_chains)}")
print(f"  Total: {len(all_chains)} chains")

print("\nBuilding feature pools...")
chain_data = []
for i, c in enumerate(all_chains):
    gold_c = c["required_ids"][2]
    if gold_c not in nb:
        continue
    feats = collect_terminal_features(c["start_token"])
    if feats:
        chain_data.append({
            "idx": int(c["idx"]),
            "query": c["start_token"],
            "gold": gold_c,
            "feats": feats,
            "source": c.get("sources", ""),
        })
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(all_chains)}")

print(f"  Done: {len(chain_data)} chains with non-empty feature pools")
gold_in_pool = sum(1 for d in chain_data if d["gold"] in d["feats"])
print(f"  Gold in pool: {gold_in_pool}/{len(chain_data)} ({100*gold_in_pool/max(len(chain_data),1):.1f}%)")


# ── leave-one-chain-out cross-validation ─────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn not available — skipping ML models, running baseline only")

print("\n=== Leave-one-chain-out cross-validation ===")

hits_v12 = []
hits_lr = []
hits_gb = []
hits_rf = []
hits_v12_raw = []

def v12_rank_score(f):
    # Approximate path-v12 scoring: prioritise terminals reached via many paths
    # from high-ranked anchors, with rare bridge tokens
    return (f["n_paths"] * 2.0
            + (10 - f["anchor_best_rank"]) * 0.5
            + (5 - min(f["min_bridge_df"], 5)) * 0.3)

for test_i, test_d in enumerate(chain_data):
    gold = test_d["gold"]
    test_feats = test_d["feats"]

    # v12 baseline
    cids = list(test_feats.keys())
    v12_scores = np.array([v12_rank_score(test_feats[c]) for c in cids])
    ranked_v12 = [cids[i] for i in np.argsort(-v12_scores)][:10]
    hits_v12.append(int(gold in ranked_v12))

    if not HAS_SKLEARN:
        continue

    # Build training set from all other chains
    X_pos, X_neg = [], []
    for j, train_d in enumerate(chain_data):
        if j == test_i: continue
        gold_j = train_d["gold"]
        for cid, fv in train_d["feats"].items():
            vec = feats_to_vec(fv)
            if cid == gold_j:
                X_pos.append(vec)
            else:
                X_neg.append(vec)

    if not X_pos:
        hits_lr.append(0); hits_gb.append(0); hits_rf.append(0)
        continue

    # Balance negatives (4:1 ratio to keep training tractable)
    rng = np.random.default_rng(42 + test_i)
    n_neg = min(len(X_neg), len(X_pos) * 4)
    neg_idx = rng.choice(len(X_neg), n_neg, replace=False)
    X_neg_s = [X_neg[i] for i in neg_idx]

    X_train = np.array(X_pos + X_neg_s)
    y_train = np.array([1] * len(X_pos) + [0] * len(X_neg_s))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    X_test = np.array([feats_to_vec(test_feats[c]) for c in cids])
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
    lr.fit(X_train_s, y_train)
    lr_scores = lr.predict_proba(X_test_s)[:, 1]

    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    gb_scores = gb.predict_proba(X_test)[:, 1]

    rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42,
                                class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_scores = rf.predict_proba(X_test)[:, 1]

    def hit_at_k(scores, gold, k=10):
        ranked = [cids[i] for i in np.argsort(-scores)][:k]
        return int(gold in ranked)

    hits_lr.append(hit_at_k(lr_scores, gold))
    hits_gb.append(hit_at_k(gb_scores, gold))
    hits_rf.append(hit_at_k(rf_scores, gold))

    if (test_i + 1) % 20 == 0:
        n_done = test_i + 1
        print(f"  {n_done}/{len(chain_data)} done | "
              f"v12={sum(hits_v12)/n_done:.3f} "
              f"LR={sum(hits_lr)/n_done:.3f} "
              f"GB={sum(hits_gb)/n_done:.3f} "
              f"RF={sum(hits_rf)/n_done:.3f}")

n = len(chain_data)
print(f"\n{'='*55}")
print(f"RESULTS (hit@10, n={n} chains with non-empty pools)")
print(f"{'='*55}")
print(f"  v12_baseline         {sum(hits_v12):3d}/{n} = {100*sum(hits_v12)/n:.1f}%")
if HAS_SKLEARN:
    print(f"  logistic_regression  {sum(hits_lr):3d}/{n} = {100*sum(hits_lr)/n:.1f}%")
    print(f"  gradient_boosting    {sum(hits_gb):3d}/{n} = {100*sum(hits_gb)/n:.1f}%")
    print(f"  random_forest        {sum(hits_rf):3d}/{n} = {100*sum(hits_rf)/n:.1f}%")

# ── feature importance (GB, full dataset) ────────────────────────────────────
if HAS_SKLEARN and chain_data:
    print(f"\n{'='*55}")
    print("FEATURE IMPORTANCE (GB, full dataset)")
    print(f"{'='*55}")
    X_all, y_all = [], []
    for d in chain_data:
        for cid, fv in d["feats"].items():
            X_all.append(feats_to_vec(fv))
            y_all.append(1 if cid == d["gold"] else 0)
    X_all = np.array(X_all); y_all = np.array(y_all)
    gb_full = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    gb_full.fit(X_all, y_all)
    imps = sorted(zip(FEATURE_NAMES, gb_full.feature_importances_), key=lambda x: -x[1])
    for feat, imp in imps:
        bar = '█' * int(imp * 80)
        print(f"  {feat:<38} {imp:.4f} {bar}")

    # Save model
    import pickle
    model_path = ROOT / "reranker_gb_v2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": gb_full, "scaler": None, "features": FEATURE_NAMES}, f)
    print(f"\nModel saved to {model_path}")
