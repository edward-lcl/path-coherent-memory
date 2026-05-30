#!/usr/bin/env python3
"""RFM benchmark on Talos corpus."""
import json, re, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
notes_raw = json.loads((ROOT / "talos_corpus_notes.json").read_text())

GENERIC = set("able accepted active actual additional basic better clean clear common correct current different direct enough exact external final first full general good great hard high human important initial known large latest likely live local long main major meaningful messy minimal native new next obvious old ongoing only operational other personal possible primary prior raw real recent related relevant same second separate simple small specific stable strong sure technical top true useful weak whole system memory file data user note output text block token result".split())
STOP_EXTRA = set("type name status created updated source boundary tags evidence confidence tier person organization entity".split())
ALL_STOP = GENERIC | STOP_EXTRA

def toks(text):
    return [w.lower() for w in re.findall(r'[a-z]{4,}', text.lower())]

def tok_ok(t):
    return t not in ALL_STOP and 5 <= len(t) <= 20 and not t[:4].isdigit()

notes = []
for n in notes_raw:
    content = n["content"]
    tokens = [t for t in toks(content) if tok_ok(t)]
    if len(tokens) >= 6:
        notes.append({**n, "_toks": set(tokens)})

print(f"Talos corpus: {len(notes)} usable notes")
src_dist = defaultdict(int)
for n in notes: src_dist[n["source"]] += 1
print("Sources:", dict(sorted(src_dist.items(), key=lambda x: x[1], reverse=True)))

df = defaultdict(int)
postings = defaultdict(list)
for n in notes:
    for t in n["_toks"]: df[t] += 1
for n in notes:
    for t in n["_toks"]: postings[t].append(n["id"])

token_sources = defaultdict(set)
for n in notes:
    for t in n["_toks"]: token_sources[t].add(n["source"])

nb = {n["id"]: n for n in notes}
nt = {n["id"]: n["_toks"] for n in notes}
N = len(notes)

def candidate_bridges(node_id, exclude, top_k=None):
    node_src = nb[node_id]["source"]
    scored = []
    for t in nt.get(node_id, set()):
        if t in exclude: continue
        cross = token_sources.get(t, set()) - {node_src}
        n_cross = len(cross)
        if n_cross == 0 or n_cross > 10: continue
        score = (1.0/df.get(t,1)**0.5) * (1.0/n_cross) * min(len(t),12)*0.1
        scored.append((score,t))
    scored.sort(reverse=True)
    return [t for _,t in (scored[:top_k] if top_k else scored)]

def bm25_retrieve(query_toks, top_k=10):
    scores = {}
    for t in set(query_toks):
        if t not in postings: continue
        idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
        for nid in postings[t]: scores[nid] = scores.get(nid, 0) + idf
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

# Mine chains
print("\nMining chains...")
rare = {t for t, d in df.items() if 2 <= d <= 6 and tok_ok(t)}
unique = {t for t, d in df.items() if d == 1 and tok_ok(t) and len(t) >= 5}

chains = []; seen_k=set(); seen_s=set()
for b_id, b_toks in nt.items():
    b_src = nb[b_id]["source"]
    bridges = [t for t in b_toks if t in rare]
    if len(bridges) < 2: continue
    for i, t1 in enumerate(bridges):
        for t2 in bridges[i+1:]:
            for a_id in postings[t1]:
                if a_id == b_id or t2 in nt[a_id]: continue
                starts = [t for t in nt[a_id] if t in unique]
                if not starts: continue
                for c_id in postings[t2]:
                    if c_id in {a_id, b_id}: continue
                    if nb[a_id]["source"] == nb[c_id]["source"]: continue
                    if len(nt[a_id] & nt[c_id]) > 1: continue
                    if len({nb[a_id]["source"], b_src, nb[c_id]["source"]}) < 2: continue
                    k=(a_id,b_id,c_id); st=starts[0]
                    if k in seen_k or st in seen_s: continue
                    seen_k.add(k); seen_s.add(st)
                    score = 10.0+(5-min(df[t1],5))+(5-min(df[t2],5))
                    chains.append((score,st,t1,t2,a_id,b_id,c_id))
                    if len(chains) >= 5000: break
                if len(chains) >= 5000: break
            if len(chains) >= 5000: break
        if len(chains) >= 5000: break
    if len(chains) >= 5000: break

chains.sort(key=lambda x: x[0], reverse=True)
selected=[]; sk=set(); ss=set()
for sc,st,t1,t2,a,b,c in chains:
    k=(a,b,c)
    if k in sk or st in ss: continue
    sk.add(k); ss.add(st)
    selected.append((sc,st,t1,t2,a,b,c))
    if len(selected)>=200: break
print(f"  Mined {len(selected)} chains")

# Oracle ceiling
print("Computing oracle ceiling...")
reachable=0
for _,q,_,_,a_id,b_id,c_id in selected:
    q_toks=[t for t in toks(q) if tok_ok(t)]
    found=False
    for a in bm25_retrieve(q_toks,20):
        for t1 in candidate_bridges(a,set(q_toks),8):
            for b in postings.get(t1,[])[:8]:
                if b==a: continue
                for t2 in candidate_bridges(b,set(q_toks)|{t1},8):
                    if c_id in postings.get(t2,[]):
                        found=True; break
                if found: break
            if found: break
        if found: break
    reachable+=int(found)
print(f"Oracle ceiling: {reachable}/{len(selected)} = {100*reachable/max(len(selected),1):.1f}%")

# Source connectivity
src_list=sorted(src_dist.keys())
connected=set()
for t,srcs in token_sources.items():
    if 2<=len(srcs)<=10:
        sl=sorted(srcs)
        for i in range(len(sl)):
            for j in range(i+1,len(sl)): connected.add((sl[i],sl[j]))
total=len(src_list)*(len(src_list)-1)//2
print(f"Source connectivity: {len(connected)}/{total} = {100*len(connected)/max(total,1):.1f}%")

def path_v12(q, top_k=10):
    q_toks=[t for t in toks(q) if tok_ok(t)]; qq=set(q_toks)
    as_,ts_={},{}
    for rank,a_id in enumerate(bm25_retrieve(q_toks,10)):
        a_toks=nt.get(a_id,set())
        as_[a_id]=max(as_.get(a_id,0),1.0+0.05*(10-rank))
        for t1 in candidate_bridges(a_id,qq,2):
            for b_id in postings.get(t1,[])[:8]:
                if b_id==a_id: continue
                as_[b_id]=max(as_.get(b_id,0),0.8)
                for t2 in candidate_bridges(b_id,qq|{t1},2):
                    for c_id in postings.get(t2,[])[:8]:
                        if c_id in {a_id,b_id}: continue
                        if len(nt.get(c_id,set())&a_toks)>0: continue
                        ts_[c_id]=max(ts_.get(c_id,0),1.0+0.05*(10-rank))
    tpt=sorted(ts_,key=lambda x:ts_[x],reverse=True)
    tpa=sorted(as_,key=lambda x:as_[x],reverse=True)
    n_t=min(len(tpt),max(top_k//2,3)); res,seen=[],set()
    for nid in tpt[:n_t]: res.append(nid); seen.add(nid)
    for nid in tpa:
        if len(res)>=top_k: break
        if nid not in seen: res.append(nid); seen.add(nid)
    return res[:top_k]

print("\nRunning benchmark...")
bm25_t=bm25_f=path_t=path_f=0
n_ch=len(selected)
for _,q,t1,t2,a,b,c in selected:
    qt=[t for t in toks(q) if tok_ok(t)]
    br=set(bm25_retrieve(qt,10)); pr=set(path_v12(q,10))
    bm25_t+=int(c in br); bm25_f+=int(all(x in br for x in [a,b,c]))
    path_t+=int(c in pr); path_f+=int(all(x in pr for x in [a,b,c]))

print(f"\nTalos benchmark (n={n_ch}):")
print(f"  BM25:         terminal {100*bm25_t/n_ch:5.1f}%  full {100*bm25_f/n_ch:4.1f}%")
print(f"  path-coherent terminal {100*path_t/n_ch:5.1f}%  full {100*path_f/n_ch:4.1f}%")
print(f"  gain: +{100*(path_t-bm25_t)/n_ch:.1f}pp terminal")

results={
    "corpus":"talos","notes":len(notes),"sources":len(src_dist),"chains":n_ch,
    "oracle_ceiling":f"{reachable}/{n_ch}={100*reachable/max(n_ch,1):.1f}%",
    "source_connectivity":f"{len(connected)}/{total}={100*len(connected)/max(total,1):.1f}%",
    "bm25_terminal":f"{100*bm25_t/n_ch:.1f}%","path_terminal":f"{100*path_t/n_ch:.1f}%",
    "bm25_full":f"{100*bm25_f/n_ch:.1f}%","path_full":f"{100*path_f/n_ch:.1f}%",
}
(ROOT/"talos_benchmark_results.json").write_text(json.dumps(results,indent=2))
print("Saved talos_benchmark_results.json")
