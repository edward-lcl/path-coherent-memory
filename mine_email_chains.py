#!/usr/bin/env python3
"""
Email chain miner — third independent benchmark corpus.

Reads Apple Mail .emlx files, extracts structured email notes,
then runs token-path chain mining identical to the Levi/Talos miners.

This is the reproducibility corpus: any personal email archive works.
Demonstrates that path-coherent topology generalizes across corpus types.

Usage:
    python3 mine_email_chains.py --mail-dir /path/to/Mail/V10/account/mailbox.mbox
    python3 mine_email_chains.py  # auto-discovers Apple Mail on macOS
    python3 mine_email_chains.py --mbox /path/to/export.mbox  # standard mbox format
"""
import argparse, email, email.policy, json, mailbox, os, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "email_chain_candidates_v1.jsonl"

# ── Apple Mail emlx parser ────────────────────────────────────────────────���───
def parse_emlx(path: Path) -> dict | None:
    """Parse a single Apple Mail .emlx file into a note dict."""
    try:
        with open(path, "rb") as f:
            # emlx format: first line is byte count, then raw email, then plist
            first = f.readline().decode("ascii", errors="ignore").strip()
            try:
                byte_count = int(first)
            except ValueError:
                return None
            raw = f.read(byte_count)
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        sender = str(msg.get("From", ""))
        subject = str(msg.get("Subject", ""))
        date = str(msg.get("Date", ""))
        to = str(msg.get("To", ""))
        sender_domain = ""
        m = re.search(r"@([\w.-]+)", sender)
        if m:
            sender_domain = m.group(1).lower()

        # Extract plain text body
        body = ""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    ct = part.get_content_type()
                    if ct == "text/plain":
                        try:
                            body = part.get_content()
                            break
                        except Exception:
                            pass
            else:
                if msg.get_content_type() == "text/plain":
                    body = msg.get_content()
        except Exception:
            pass

        # Build note content: subject + sender + cleaned body
        body_clean = re.sub(r"\s+", " ", body or "").strip()[:1000]
        content = f"{subject}\n{sender}\n{to}\n{body_clean}"

        return {
            "id": f"email-{path.stem}",
            "source": f"email:{sender_domain}:{path.stem}",
            "sender_domain": sender_domain,
            "sender": sender[:100],
            "subject": subject[:200],
            "date": date,
            "content": content,
        }
    except Exception:
        return None


def parse_mbox(mbox_path: Path) -> list[dict]:
    """Parse a standard .mbox file."""
    notes = []
    try:
        mb = mailbox.mbox(str(mbox_path))
        for i, msg in enumerate(mb):
            sender = str(msg.get("From", ""))
            subject = str(msg.get("Subject", ""))
            date = str(msg.get("Date", ""))
            to = str(msg.get("To", ""))
            sender_domain = ""
            m = re.search(r"@([\w.-]+)", sender)
            if m:
                sender_domain = m.group(1).lower()

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                            break
                        except Exception:
                            pass
            else:
                try:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                except Exception:
                    pass

            body_clean = re.sub(r"\s+", " ", body or "").strip()[:1000]
            content = f"{subject}\n{sender}\n{to}\n{body_clean}"
            notes.append({
                "id": f"email-{i:06d}",
                "source": f"email:{sender_domain}:{i}",
                "sender_domain": sender_domain,
                "sender": sender[:100],
                "subject": subject[:200],
                "date": date,
                "content": content,
            })
    except Exception as e:
        print(f"mbox parse error: {e}")
    return notes


def discover_apple_mail() -> list[Path]:
    """Find all .emlx files in Apple Mail's local store."""
    mail_root = Path.home() / "Library" / "Mail"
    if not mail_root.exists():
        return []
    emlx_files = list(mail_root.rglob("*.emlx"))
    # Exclude partial downloads
    emlx_files = [p for p in emlx_files if ".partial." not in p.name]
    return emlx_files


# ── Token index + chain mining (same algorithm as mine_candidates_v2.py) ──────
GENERIC = set("""able accepted active actual additional adjacent ahead almost already another
available basic better broader careful clean clear common concrete correct critical current
different direct earlier enough exact explicit external final first fresh full general good
great hard high human important initial internal known large latest likely live local long
main major meaningful minimal native new next obvious old ongoing only operational other
personal possible primary prior raw real recent related relevant same second separate simple
small specific stable strong sure technical top true useful weak whole system memory file
data user note output text block token result type name status created updated source
boundary tags evidence confidence tier person organization entity service endpoint project
work time week day month year back going make makes made used using been have will would
could should also just like much very some more most than from this that with they them
their about into over under when where which while unsubscribe click here view browser
email account manage preferences update notification alert welcome thank dear hello""".split())


def toks(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text)]


def tok_ok(t: str) -> bool:
    if t in GENERIC: return False
    if len(t) < 5 or len(t) > 20: return False
    if t[:4].isdigit(): return False
    if re.search(r"(ing|tion|ment|ness|less|ful|ible|ous|ive|ary|ory|ize|ise|ify)$", t) and len(t) > 9:
        return False
    return True


def excerpt(note: dict, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", note["content"]).strip()
    snip = text[:max_chars - 1] + ("..." if len(text) > max_chars else "")
    return f"[{note['sender_domain']}] {note['subject'][:60]}: {snip}"


def mine(notes: list[dict], limit: int = 300) -> list[dict]:
    nb = {n["id"]: n for n in notes}
    nt = {n["id"]: {t for t in toks(n["content"]) if tok_ok(t)} for n in notes}

    df: dict[str, int] = defaultdict(int)
    postings: dict[str, list] = defaultdict(list)
    token_sources: dict[str, set] = defaultdict(set)
    for nid, ts in nt.items():
        src = nb[nid]["source"]
        for t in ts:
            df[t] += 1
            postings[t].append(nid)
            token_sources[t].add(src)

    rare = {t for t, d in df.items() if 2 <= d <= 5 and tok_ok(t)}
    unique = {t for t, d in df.items() if d == 1 and tok_ok(t) and len(t) >= 5}
    print(f"  rare bridge tokens: {len(rare)}")
    print(f"  unique start tokens: {len(unique)}")

    chains = []
    seen_keys: set[tuple] = set()
    seen_starts: set[str] = set()

    for b_id, b_toks in nt.items():
        b_src = nb[b_id]["source"]
        bridges = [t for t in b_toks if t in rare]
        if len(bridges) < 2:
            continue
        for i, t1 in enumerate(bridges):
            nc1 = len(token_sources.get(t1, set()) - {b_src})
            if nc1 == 0 or nc1 > 10:
                continue
            for t2 in bridges[i + 1:]:
                nc2 = len(token_sources.get(t2, set()) - {b_src})
                if nc2 == 0 or nc2 > 10:
                    continue
                a_cands = [n for n in postings[t1]
                           if n != b_id and t2 not in nt.get(n, set())]
                c_cands = [n for n in postings[t2]
                           if n != b_id and t1 not in nt.get(n, set())]
                for a_id in a_cands:
                    a_src = nb[a_id]["source"]
                    starts = sorted(
                        [t for t in nt.get(a_id, set()) if t in unique],
                        key=lambda x: -len(x),
                    )
                    if not starts:
                        continue
                    for c_id in c_cands:
                        if c_id == a_id:
                            continue
                        c_src = nb[c_id]["source"]
                        if a_src == c_src:
                            continue
                        srcs = {a_src, b_src, c_src}
                        if len(srcs) < 2:
                            continue
                        direct_overlap = len(nt.get(a_id, set()) & nt.get(c_id, set()))
                        if direct_overlap > 1:
                            continue
                        key = (a_id, b_id, c_id)
                        if key in seen_keys:
                            continue
                        start = starts[0]
                        if start in seen_starts:
                            continue
                        score = (
                            10.0
                            + (5 - min(df[t1], 5)) + (5 - min(df[t2], 5))
                            + min(len(t1), 12) * 0.1 + min(len(t2), 12) * 0.1
                            - direct_overlap * 4.0
                        )
                        chains.append((score, start, t1, t2, a_id, b_id, c_id))

    chains.sort(reverse=True)
    results = []
    for score, start, t1, t2, a_id, b_id, c_id in chains:
        key = (a_id, b_id, c_id)
        if key in seen_keys or start in seen_starts:
            continue
        seen_keys.add(key)
        seen_starts.add(start)
        results.append({
            "idx": len(results) + 1,
            "score": round(float(score), 3),
            "miner": "email_v1",
            "start_token": start,
            "bridge1": t1,
            "bridge2": t2,
            "sources": " | ".join(nb[nid]["source"] for nid in [a_id, b_id, c_id]),
            "required_ids": [a_id, b_id, c_id],
            "excerpts": {
                "a": excerpt(nb[a_id]),
                "b": excerpt(nb[b_id]),
                "c": excerpt(nb[c_id]),
            },
        })
        if len(results) >= limit:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Email chain miner for RFM benchmark")
    parser.add_argument("--mail-dir", type=Path,
                        help="Apple Mail account directory (contains *.mbox subdirs)")
    parser.add_argument("--mbox", type=Path,
                        help="Standard .mbox file")
    parser.add_argument("--limit", type=int, default=300,
                        help="Max chains to mine (default 300)")
    parser.add_argument("--out", type=Path, default=OUT,
                        help=f"Output file (default {OUT})")
    args = parser.parse_args()

    notes = []

    if args.mbox:
        print(f"Parsing mbox: {args.mbox}")
        notes = parse_mbox(args.mbox)
    elif args.mail_dir:
        print(f"Scanning Apple Mail dir: {args.mail_dir}")
        emlx_files = list(args.mail_dir.rglob("*.emlx"))
        emlx_files = [p for p in emlx_files if ".partial." not in p.name]
        print(f"  Found {len(emlx_files)} .emlx files")
        for p in emlx_files:
            n = parse_emlx(p)
            if n and len(n["content"].strip()) > 50:
                notes.append(n)
    else:
        print("Auto-discovering Apple Mail...")
        emlx_files = discover_apple_mail()
        print(f"  Found {len(emlx_files)} .emlx files across all accounts")
        for p in emlx_files:
            n = parse_emlx(p)
            if n and len(n["content"].strip()) > 50:
                notes.append(n)

    print(f"\nLoaded {len(notes)} email notes")
    if len(notes) < 10:
        print("Not enough emails to mine chains. Try --mail-dir or --mbox.")
        return

    # Deduplicate by subject+sender
    seen_sigs = set()
    deduped = []
    for n in notes:
        sig = (n["sender_domain"], re.sub(r"\s+", "", n["subject"].lower())[:50])
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            deduped.append(n)
    print(f"After dedup: {len(deduped)} notes")
    print(f"Unique sender domains: {len(set(n['sender_domain'] for n in deduped))}")

    print("\nMining chains...")
    chains = mine(deduped, limit=args.limit)
    print(f"Mined {len(chains)} chains")

    args.out.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chains) + "\n"
    )
    print(f"\nWrote {args.out}")
    print("\nSample:")
    for c in chains[:10]:
        print(f"  {c['idx']:03d} q={c['start_token']:<14} b1={c['bridge1']:<14} b2={c['bridge2']:<14}")
        print(f"       {c['sources'][:80]}")


if __name__ == "__main__":
    main()
