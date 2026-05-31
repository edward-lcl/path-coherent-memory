#!/usr/bin/env python3
"""
Summarize each OCPlatform main-agent session into a structured episode summary.

Each session -> one summary node: "This session worked on X. Key decision/action: Y.
Outcome / blocker: Z." These sparse, distinctive nodes give token-path the kind of
concept-bridge vocabulary it needs (project names, specific decisions, outcomes)
rather than high-frequency agent-conversation words (running, output, files, check).

Output: agent_session_summaries.json — list of {id, session_id, source, content}
Same schema as the raw chunk nodes so the miner/harness can ingest directly.
"""
import json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def _load_omlx_key():
    import os
    k=os.environ.get('OMLX_API_KEY')
    if k: return k
    f=ROOT/'.omlx_key'
    if f.exists(): return f.read_text().strip()
    raise RuntimeError('no OMLX key')
OMLX_KEY = (ROOT / '.omlx_key').read_text().strip()
SESSION_DIR = Path("/Users/edward/.ocplatform/agents/main/sessions")
OUT = ROOT / "agent_session_summaries.json"
PROGRESS = ROOT / "agent_session_summaries.progress.json"

# Reuse the working oMLX key loader
OMLX_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen3-4B-Instruct-2507-MLX-8bit"

import urllib.request

SUMMARY_PROMPT = """You are summarizing an AI agent work session. Extract the key facts in 3-4 sentences:
1. What specific project/task was this session focused on? (name it precisely)
2. What was the key action, decision, or problem encountered?
3. What was the outcome, result, or blocker?

Be concrete and specific — use actual project names, tool names, error messages. Avoid generic phrases like "the agent worked on" or "various tasks". Include distinctive terminology that would only appear in this session's context.

Session excerpt (first ~1200 words):
{excerpt}

Summary (3-4 sentences, concrete and specific):"""


def extract_text(rec: dict) -> str:
    content = rec.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text","") for b in content if isinstance(b,dict) and b.get("type")=="text"]
        return " ".join(parts)
    return ""


def call_omlx(prompt: str, retries: int = 3) -> str:
    import urllib.error
    payload = json.dumps({"model": MODEL, "messages": [{"role":"user","content": prompt}],
                          "max_tokens": 200, "temperature": 0.2}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OMLX_KEY}"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OMLX_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
                return d["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries - 1:
                return ""
            time.sleep(2)
    return ""


def summarize_session(path: Path) -> dict | None:
    sid = path.stem
    try:
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except Exception:
        return None

    # Extract first ~1200 words of user+assistant text (enough to identify the session)
    words = []
    for rec in records:
        if rec.get("type") != "message":
            continue
        role = rec.get("message", {}).get("role", "")
        if role not in ("user", "assistant"):
            continue
        txt = extract_text(rec).strip()
        if not txt or len(txt.split()) < 5:
            continue
        words.extend(txt.split()[:300])
        if len(words) >= 1200:
            break

    if len(words) < 50:
        return None

    excerpt = " ".join(words[:1200])
    prompt = SUMMARY_PROMPT.format(excerpt=excerpt)
    summary = call_omlx(prompt)
    if not summary or len(summary.split()) < 10:
        return None

    return {
        "id": f"session_summary::{sid}",
        "session_id": sid,
        "source": f"session/{sid[:8]}",
        "content": summary,
        "word_count": len(summary.split()),
    }


def main():
    files = sorted(
        [f for f in SESSION_DIR.glob("*.jsonl")
         if ".bak" not in f.name and "trajectory" not in f.name],
        key=lambda f: f.stat().st_mtime
    )
    print(f"Summarizing {len(files)} sessions with {MODEL}…")

    # Load progress
    done = {}
    if PROGRESS.exists():
        done = {d["session_id"]: d for d in json.loads(PROGRESS.read_text())}
        print(f"  resuming: {len(done)} already done")

    summaries = list(done.values())
    for i, f in enumerate(files):
        sid = f.stem
        if sid in done:
            continue
        node = summarize_session(f)
        if node:
            summaries.append(node)
            done[sid] = node
            if (i+1) % 5 == 0 or i < 3:
                print(f"  [{i+1}/{len(files)}] {sid[:8]}: {node['content'][:100]}")
        else:
            print(f"  [{i+1}/{len(files)}] {sid[:8]}: skipped (short/empty)")
        # Save progress every 10
        if (i+1) % 10 == 0:
            PROGRESS.write_text(json.dumps(summaries))

    PROGRESS.write_text(json.dumps(summaries))
    OUT.write_text(json.dumps(summaries))
    wc = [s["word_count"] for s in summaries]
    import statistics
    print(f"\nDone: {len(summaries)} summaries, avg {statistics.mean(wc):.0f} words/summary")
    print(f"Saved → {OUT}")
    if summaries:
        print(f"\nSample summaries:")
        for s in summaries[:3]:
            print(f"  [{s['session_id'][:8]}] {s['content'][:160]}")


if __name__ == "__main__":
    main()
