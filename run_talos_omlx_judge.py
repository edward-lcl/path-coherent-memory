#!/usr/bin/env python3
"""Judge Talos semantic chain candidates via oMLX."""
import json, re, time, urllib.request, subprocess
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / "talos_semantic_chain_candidates_v1.jsonl"
ANSWER_KEY_V1 = ROOT / "levi_semantic_chain_answer_key_v1.jsonl"
CANDIDATES_V1 = ROOT / "levi_semantic_chain_candidates_v1.jsonl"
OUT = ROOT / "talos_semantic_chain_omlx_judge_v1.jsonl"
OMLX_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "gemma-4-E4B-it-MLX-4bit"
LABELS = {"real_semantic","weak_semantic","artifact"}

def get_key():
    settings = json.loads(Path("/Users/edward/.omlx/settings.json").read_text())
    return settings["auth"]["api_key"]

def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

def short(text, limit=280):
    return re.sub(r'\s+',' ',text).strip()[:limit]

def build_prompt(cand, examples=None):
    ex=cand["excerpts"]; eb=""
    if examples:
        chunks=[]
        for e in examples:
            chunks.append(f"Example (idx {e['idx']}):\n  start={e['start_token']} bridge1={e['bridge1']} bridge2={e['bridge2']}\n  A: {short(e['excerpts']['a'],200)}\n  B: {short(e['excerpts']['b'],200)}\n  C: {short(e['excerpts']['c'],200)}\n  Label: {e['label']}\n  Why: {e.get('rationale','')}")
        eb="\nCalibration examples:\n"+"\n\n".join(chunks)+"\n"
    return f"""Judge this 3-hop memory chain.

Labels:
- real_semantic: A→B→C form a defensible human-semantic chain.
- weak_semantic: At least one hop is meaningful but the full chain drifts.
- artifact: Token coincidence, boilerplate, structural noise.
{eb}
Candidate:
idx: {cand['idx']}
start_token: {cand['start_token']}
bridge1: {cand['bridge1']}
bridge2: {cand['bridge2']}
sources: {cand['sources']}

A: {short(ex['a'])}
B: {short(ex['b'])}
C: {short(ex['c'])}

Return only JSON: {{"idx":{cand['idx']},"label":"real_semantic|weak_semantic|artifact","rationale":"one sentence"}}"""

def call_omlx(prompt, key, retries=3):
    payload=json.dumps({"model":MODEL,"max_tokens":200,"temperature":0.1,"messages":[{"role":"user","content":prompt}]}).encode()
    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}
    for attempt in range(retries):
        try:
            req=urllib.request.Request(OMLX_URL,data=payload,headers=headers,method="POST")
            with urllib.request.urlopen(req,timeout=120) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt<retries-1: time.sleep(2**attempt)
            else: raise

def parse_output(idx, raw):
    label=None; rationale=""
    try:
        start=raw.index("{"); end=raw.rindex("}")+1
        obj=json.loads(raw[start:end]); label=obj.get("label"); rationale=obj.get("rationale","")
    except: rationale=raw.strip().splitlines()[0][:300] if raw.strip() else ""
    if label not in LABELS:
        m=re.search(r'\b(real_semantic|weak_semantic|artifact)\b',raw)
        label=m.group(1) if m else "artifact"
    return {"idx":idx,"label":label,"rationale":rationale,"raw":raw[:500]}

key = get_key()
candidates = read_jsonl(CANDIDATES)
answer = {int(r["idx"]): r for r in read_jsonl(ANSWER_KEY_V1)}
v1 = {int(r["idx"]): r for r in read_jsonl(CANDIDATES_V1)}
examples = [{**v1[i],**answer[i]} for i in [2,3,1] if i in answer and i in v1]

done = set()
if OUT.exists():
    for r in read_jsonl(OUT): done.add(int(r["idx"]))
    candidates = [c for c in candidates if int(c["idx"]) not in done]
    print(f"Resuming: {len(done)} done, {len(candidates)} remaining")

print(f"Model={MODEL}  judging={len(candidates)}  few_shot=True")
t0 = time.time()
with OUT.open("a") as f:
    for i, cand in enumerate(candidates, 1):
        prompt = build_prompt(cand, examples=examples)
        raw = call_omlx(prompt, key)
        row = parse_output(int(cand["idx"]), raw)
        f.write(json.dumps(row, ensure_ascii=False)+"\n")
        elapsed = time.time()-t0
        rate = i/elapsed; eta = (len(candidates)-i)/rate if rate>0 else 0
        print(f"{i:03d}/{len(candidates):03d} idx={row['idx']:03d} label={row['label']:<14} eta={eta/60:.1f}m")

results = read_jsonl(OUT)
counts = Counter(r["label"] for r in results)
total = len(results)
print(f"\nDone: {total} judged")
for lab in ["real_semantic","weak_semantic","artifact"]:
    n = counts.get(lab,0)
    print(f"  {lab:<16} {n:3d}  ({100*n/total:.1f}%)")
