"""Build the chat part (the 10%) of the dataset from OpenAssistant oasst1.

Output format (one conversation per block, same <|endoftext|> style as wiki.txt):

    User: <question>
    Bot: <answer>
    <|endoftext|>
"""
import argparse, gzip, json, random
from pathlib import Path
import requests

URL = ("https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
       "2023-04-12_oasst_ready.trees.jsonl.gz")
UA = "GameLM-learning-project/0.1 (personal learning project)"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
EOT = "<|endoftext|>"

# Anything that looks like code or a link is dropped (we want plain chat).
BAD = ["```", "{", "}", "def ", "#include", "import ", "</", "/>", "();",
       "http", "www."]


def download(dest):
    if dest.exists():
        print(f"Using existing file: {dest.name}")
        return True
    tmp = dest.with_name(dest.name + ".part")
    print("Downloading oasst1...")
    try:
        with requests.get(URL, stream=True, timeout=60,
                          headers={"User-Agent": UA}) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            got = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    print(f"\r  {got / 1e6:.1f} / {total / 1e6:.1f} MB",
                          end="", flush=True)
        print()
        tmp.rename(dest)
        return True
    except requests.RequestException as e:
        print(f"\nDownload failed: {e}")
        print("Manual fix: download this file in your browser and save it as")
        print(f"  {dest}\n  {URL}")
        return False


def norm(text):
    """Collapse all whitespace/newlines so each turn stays on one line."""
    return " ".join(text.split())


def usable(msg):
    return (not msg.get("deleted")
            and msg.get("review_result") is not False
            and msg.get("lang") == "en")


def pairs_from_tree(root):
    """Yield (question, answer) for every prompter -> assistant reply."""
    if not usable(root):
        return
    stack = [root]
    while stack:
        node = stack.pop()
        for reply in node.get("replies", []):
            if not usable(reply):
                continue
            if node.get("role") == "prompter" and reply.get("role") == "assistant":
                yield norm(node["text"]), norm(reply["text"])
            stack.append(reply)


def good(text, max_len):
    return 10 <= len(text) <= max_len and not any(b in text for b in BAD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-prompt", type=int, default=300)
    ap.add_argument("--max-reply", type=int, default=400)
    a = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    src = RAW / "oasst1.jsonl.gz"
    if not download(src):
        return

    seen, kept, total = set(), [], 0
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            tree = json.loads(line)
            for q, ans in pairs_from_tree(tree["prompt"]):
                total += 1
                if good(q, a.max_prompt) and good(ans, a.max_reply) \
                        and (q, ans) not in seen:
                    seen.add((q, ans))
                    kept.append((q, ans))

    out_path = RAW / "chat.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        for q, ans in kept:
            out.write(f"User: {q}\nBot: {ans}\n{EOT}\n")

    chars = sum(len(q) + len(ans) for q, ans in kept)
    print(f"English pairs found: {total:,} | kept after filtering: {len(kept):,}")
    print(f"Chat text: {chars:,} characters -> {out_path}")
    random.seed(0)
    for q, ans in random.sample(kept, min(3, len(kept))):
        print(f"\nUser: {q}\nBot: {ans}")


if __name__ == "__main__":
    main()
