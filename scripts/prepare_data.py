"""Turn the raw text files into token files the model can train on.

wiki.txt + chat.txt  --(tokenizer)-->  train.bin / val.bin  (uint16 tokens)

Steps:
  1. split every file into documents (separated by <|endoftext|>)
  2. tokenize each document and append the end-of-text token
  3. hold out a small validation set (from wiki AND chat)
  4. repeat the chat documents enough that chat is ~10% of the training tokens
  5. shuffle documents, glue everything into one long token stream, save it
"""
import argparse
import json
import random
import sys
import time
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tokenizer.bpe import BPETokenizer, EOT  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def load_docs(path):
    text = path.read_text(encoding="utf-8")
    return [d.strip() for d in text.split(EOT) if d.strip()]


def encode_docs(tok, docs, name):
    out, t0 = [], time.time()
    for i, d in enumerate(docs, 1):
        out.append(array("H", tok.encode(d) + [tok.eot_id]))  # 2 bytes/token
        if i % 500 == 0:
            print(f"  {name}: {i}/{len(docs)} docs ({time.time() - t0:.0f}s)",
                  flush=True)
    return out


def split(docs, frac, rng):
    docs = docs[:]
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * frac)) if docs else 0
    return docs[n_val:], docs[:n_val]


def n_tokens(docs):
    return sum(len(d) for d in docs)


def write_bin(docs, path):
    stream = array("H")
    for d in docs:
        stream.extend(d)
    with open(path, "wb") as f:
        stream.tofile(f)
    return len(stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-share", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    tok = BPETokenizer.load(ROOT / "tokenizer" / "tokenizer.json")
    assert tok.vocab_size < 65536, "ids must fit in uint16"
    print(f"Tokenizer loaded: vocab size {tok.vocab_size}")

    wiki = encode_docs(tok, load_docs(RAW / "wiki.txt"), "wiki")
    chat_path = RAW / "chat.txt"
    chat = encode_docs(tok, load_docs(chat_path), "chat") if chat_path.exists() else []

    w_train, w_val = split(wiki, a.val_frac, rng)
    c_train, c_val = split(chat, a.val_frac, rng)
    W, C = n_tokens(w_train), n_tokens(c_train)

    # How many times must chat be repeated so that it is chat_share of all
    # training tokens?   C*r / (W + C*r) = share   ->   r = share*W / ((1-share)*C)
    repeat = 0.0
    mixed_chat = []
    if C > 0 and a.chat_share > 0:
        repeat = a.chat_share * W / ((1 - a.chat_share) * C)
        whole, frac = int(repeat), repeat - int(repeat)
        mixed_chat = c_train * whole + rng.sample(c_train, int(len(c_train) * frac))

    train_docs = w_train + mixed_chat
    val_docs = w_val + c_val
    rng.shuffle(train_docs)
    rng.shuffle(val_docs)

    OUT.mkdir(parents=True, exist_ok=True)
    n_train = write_bin(train_docs, OUT / "train.bin")
    n_val = write_bin(val_docs, OUT / "val.bin")

    chat_tokens = n_tokens(mixed_chat)
    meta = {
        "vocab_size": tok.vocab_size,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "chat_repeat": round(repeat, 3),
        "chat_share_actual": round(chat_tokens / max(n_train, 1), 4),
        "dtype": "uint16",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== Summary ===")
    print(f"wiki docs: {len(wiki):,} | chat docs: {len(chat):,}")
    print(f"train tokens: {n_train:,} | val tokens: {n_val:,}")
    print(f"chat repeated x{repeat:.2f} -> {meta['chat_share_actual'] * 100:.1f}% of train tokens")
    print(f"tokens per parameter (6M model, 1 epoch): {n_train / 6e6:.1f}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
