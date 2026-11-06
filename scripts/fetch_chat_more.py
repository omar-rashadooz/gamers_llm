"""Bigger chat dataset -> data/raw/chat.txt   (replaces the oasst1-only file)

Sources (all free):
    oasst1   OpenAssistant human-written conversations     (Apache-2.0)
             -> the data/raw/oasst1.jsonl.gz you ALREADY have, nothing to download
    hh       Anthropic hh-rlhf "helpful-base" conversations (MIT)      ~16 MB
    dolly    Databricks Dolly 15k questions with answers    (CC BY-SA 3.0) ~13 MB
    oasst2   (optional, not used by default) oasst1 + ~3,500 newer trees

Output: multi-turn conversations, one per block, same <|endoftext|> style as wiki.txt:

    User: How do I get better at chess?
    Bot: Play often and review your lost games.
    User: How often?
    Bot: A few games a week is enough to start.
    <|endoftext|>

Only English, no code, no links, short turns, and every conversation starts at its
FIRST message (so the model never sees an answer whose question is missing).

    python scripts/fetch_chat_more.py
"""
import argparse
import gzip
import json
import random
import re
from pathlib import Path

import requests

from textclean import clean_line

UA = "GameLM-learning-project/0.1 (personal learning project)"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
EOT = "<|endoftext|>"

HF = "https://huggingface.co/datasets"
SOURCES = {
    "oasst1": (f"{HF}/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz",
               "oasst1.jsonl.gz"),
    "oasst2": (f"{HF}/OpenAssistant/oasst2/resolve/main/2023-11-05_oasst2_ready.trees.jsonl.gz",
               "oasst2.jsonl.gz"),
    "hh": (f"{HF}/Anthropic/hh-rlhf/resolve/main/helpful-base/train.jsonl.gz",
           "hh_helpful_base.jsonl.gz"),
    "dolly": (f"{HF}/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
              "dolly.jsonl"),
}

# anything that looks like code or a link is dropped (we want plain conversation)
BAD = ["```", "{", "}", "def ", "#include", "import ", "</", "/>", "();", "http", "www."]


# ----------------------------------------------------------------- download
def download(name):
    url, fname = SOURCES[name]
    dest = RAW / fname
    if dest.exists():
        print(f"[{name}] using existing {fname}")
        return dest
    tmp = dest.with_name(dest.name + ".part")
    print(f"[{name}] downloading ...")
    try:
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": UA}) as r:
            r.raise_for_status()
            total, got = int(r.headers.get("Content-Length", 0)), 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    print(f"\r  {got / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
        tmp.rename(dest)
        return dest
    except requests.RequestException as e:
        print(f"\n[{name}] download failed: {e}")
        print(f"  manual fix: download it in your browser and save it as\n  {dest}\n  {url}")
        return None


# ----------------------------------------------------------------- cleaning
def clean_turn(text, role, a):
    text = clean_line(" ".join(str(text).split()))       # also drops non-English text
    if text is None:
        return None
    lo, hi = (2, a.max_user) if role == "User" else (8, a.max_bot)
    if not lo <= len(text) <= hi or any(b in text for b in BAD):
        return None
    return text


def build_conv(turns, a):
    """turns: [(role, text), ...] starting with User and alternating.
    Cleans every turn, stops at the first bad one, keeps as many turns as fit in
    --max-chars, and always ends on a Bot turn. Returns the text or None."""
    lines, used, last_bot = [], 0, 0
    for i, (role, text) in enumerate(turns):
        if role != ("User" if i % 2 == 0 else "Bot"):
            break
        text = clean_turn(text, role, a)
        if text is None:
            break
        line = f"{role}: {text}"
        if used + len(line) + 1 > a.max_chars:
            break
        lines.append(line)
        used += len(line) + 1
        if role == "Bot":
            last_bot = len(lines)
    return "\n".join(lines[:last_bot]) if last_bot else None


# ------------------------------------------------------------------ sources
def oasst_paths(node, prefix, out, limit):
    """All root-to-leaf paths through the good English replies, best-ranked first."""
    if len(out) >= limit:
        return
    prefix = prefix + [node]
    kids = [r for r in node.get("replies", [])
            if r.get("lang") == "en" and not r.get("deleted")
            and r.get("review_result") is not False]
    kids.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
    if not kids:
        out.append(prefix)
        return
    for k in kids:
        oasst_paths(k, prefix, out, limit)


def read_oasst(path, a):          # same tree format for oasst1 and oasst2
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            root = json.loads(line)["prompt"]
            if root.get("lang") != "en" or root.get("deleted"):
                continue
            paths = []
            oasst_paths(root, [], paths, a.paths_per_tree)
            for p in paths:
                yield [("User" if m["role"] == "prompter" else "Bot", m["text"]) for m in p]


def read_hh(path, a):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["chosen"]
            parts = re.split(r"\n\n(Human|Assistant):", text)   # ['', 'Human', msg, 'Assistant', msg ...]
            yield [("User" if r == "Human" else "Bot", m) for r, m in zip(parts[1::2], parts[2::2])]


def read_dolly(path, a):
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("context"):          # needs a reference text we do not have
                continue
            yield [("User", r["instruction"]), ("Bot", r["response"])]


READERS = {"oasst1": read_oasst, "oasst2": read_oasst, "hh": read_hh, "dolly": read_dolly}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["oasst1", "hh", "dolly"],
                    choices=list(SOURCES))
    ap.add_argument("--max-user", type=int, default=300, help="max chars in a User turn")
    ap.add_argument("--max-bot", type=int, default=400, help="max chars in a Bot turn")
    ap.add_argument("--max-chars", type=int, default=800, help="max chars in one conversation")
    ap.add_argument("--paths-per-tree", type=int, default=3, help="oasst: conversations per tree")
    ap.add_argument("--hh-max", type=int, default=15000,
                    help="cap for hh so one source does not dominate (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    seen, convs, per_source = set(), [], {}

    for name in a.sources:
        path = download(name)
        if path is None:
            continue
        got = []
        for turns in READERS[name](path, a):
            conv = build_conv(turns, a)
            if conv is not None and conv not in seen:
                seen.add(conv)
                got.append(conv)
        if name == "hh" and a.hh_max and len(got) > a.hh_max:
            rng.shuffle(got)
            got = got[:a.hh_max]
        per_source[name] = len(got)
        convs += got
        print(f"[{name}] kept {len(got):,} conversations")

    if not convs:
        raise SystemExit("Nothing was collected.")
    rng.shuffle(convs)

    out = RAW / "chat.txt"
    old = RAW / "chat_oasst1.txt"
    if out.exists() and not old.exists():
        out.rename(old)
        print(f"old chat.txt saved as {old.name}")
    with open(out, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(f"{c}\n{EOT}\n")

    chars = sum(len(c) for c in convs)
    exchanges = sum(c.count("\nBot:") + c.startswith("Bot:") for c in convs)
    print(f"\n{len(convs):,} conversations | {chars:,} characters (~{chars // 4:,} tokens) "
          f"| avg {chars // len(convs)} chars | {exchanges:,} Bot turns")
    print(f"per source: {per_source}\nsaved to {out}")
    for c in rng.sample(convs, min(3, len(convs))):
        print("\n" + c)


if __name__ == "__main__":
    main()
