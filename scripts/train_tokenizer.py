import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tokenizer.bpe import BPETokenizer  # noqa: E402

RAW = ROOT / "data" / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--input", nargs="+", default=None,
                    help="text files to train on (default: wiki.txt and chat.txt)")
    a = ap.parse_args()

    paths = [Path(p) for p in a.input] if a.input else [RAW / "wiki.txt", RAW / "chat.txt"]
    paths = [p for p in paths if p.exists()]
    if not paths:
        sys.exit("No input files found in data/raw/")
    print("Training on:", ", ".join(p.name for p in paths))
    text = "\n".join(p.read_text(encoding="utf-8") for p in paths)
    print(f"{len(text):,} characters")

    t0 = time.time()
    tok = BPETokenizer.train(text, a.vocab_size)
    print(f"Trained in {time.time() - t0:.1f}s, vocab size = {tok.vocab_size}")

    out = ROOT / "tokenizer" / "tokenizer.json"
    tok.save(out)
    print(f"Saved to {out}")

    # ---- sanity checks ----
    sample = text[:20000]
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, "round-trip failed!"
    print(f"Round-trip OK. {len(sample) / len(ids):.2f} characters per token")

    for demo in ["The player unlocked a new level in the RPG game.",
                 "User: How do I beat the boss?\nBot: Dodge his attacks first."]:
        ids = tok.encode(demo)
        print("\nDemo:", demo)
        print("pieces:", [tok.decode([i]) for i in ids])


if __name__ == "__main__":
    main()
