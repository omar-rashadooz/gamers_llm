"""Talk to your trained model in the terminal (CPU is fine, works on Termux).

    python scripts/chat.py                          # newest checkpoints/*/best.pt (prefers 10m)
    python scripts/chat.py --ckpt checkpoints/10m/best.pt
    python scripts/chat.py --prompt "How do I beat a boss?"     # one answer, then exit

Inside the chat:  /help  shows the commands (/temp, /rep, /len, /mode text ...).

Important: the model was trained on SINGLE question -> answer pairs
("User: ...\nBot: ...<|endoftext|>"), so every message is answered on its own;
it does not remember earlier messages.
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import GPT, GPTConfig  # noqa: E402
from tokenizer.bpe import BPETokenizer  # noqa: E402

STOPS = ["\nUser:"]          # if the model starts inventing the next question, cut it


# --------------------------------------------------------------- streaming text
class Streamer:
    """Turns generated token ids into text that can be printed piece by piece.

    Byte-level BPE can split one character (an accent, an emoji) over several tokens,
    and a stop string like "\\nUser:" can arrive in pieces. So we hold back any tail
    that might still turn into either of those, and print it only when it is safe."""

    def __init__(self, tok, stops):
        self.tok, self.stops = tok, stops
        self.ids, self.printed, self.done = [], 0, False

    def _text(self):
        return self.tok.decode(self.ids)

    def push(self, token_id):
        self.ids.append(token_id)
        text = self._text()
        cut = min((text.find(s) for s in self.stops if s in text), default=-1)
        if cut != -1:
            self.done = True
            new = text[self.printed:cut]
            self.printed = cut
            return new
        hold = 1 if text.endswith("�") else 0        # half of a character
        for s in self.stops:                               # half of a stop string
            for k in range(min(len(s) - 1, len(text)), 0, -1):
                if text.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        end = len(text) - hold
        new = text[self.printed:end] if end > self.printed else ""
        self.printed += len(new)
        return new

    def finish(self):
        if self.done:
            return ""
        text = self._text()
        cut = min((text.find(s) for s in self.stops if s in text), default=len(text))
        new = text[self.printed:cut]
        self.printed = cut
        return new


# ----------------------------------------------------- generation with a KV-cache
@torch.no_grad()
def step_logits(model, ids_new, cache, pos0):
    """Run only the NEW tokens through the model and return next-token logits.

    Without a cache, every generated word recomputes attention over the whole text
    again. Here each layer remembers the keys/values (k, v) of everything it has
    already seen, so a step only processes the one new token. The maths is identical
    to GPT.forward; check_cache() proves it on your device before we rely on it."""
    cfg = model.cfg
    B, n = ids_new.shape
    nh, hd = cfg.n_head, cfg.n_embd // cfg.n_head
    pos = torch.arange(pos0, pos0 + n)
    x = model.tok_emb(ids_new) + model.pos_emb(pos)
    for i, blk in enumerate(model.blocks):
        q, k, v = blk.attn.qkv(blk.ln1(x)).split(cfg.n_embd, dim=2)
        q = q.view(B, n, nh, hd).transpose(1, 2)
        k = k.view(B, n, nh, hd).transpose(1, 2)
        v = v.view(B, n, nh, hd).transpose(1, 2)
        if cache[i] is not None:
            k = torch.cat([cache[i][0], k], dim=2)
            v = torch.cat([cache[i][1], v], dim=2)
        cache[i] = (k, v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)           # (B, nh, n, total)
        if n > 1:   # new token j sits at absolute position pos0+j: it may see keys <= that
            qpos = torch.arange(pos0, pos0 + n).unsqueeze(1)
            kpos = torch.arange(k.size(2)).unsqueeze(0)
            att = att.masked_fill(kpos > qpos, float("-inf"))
        y = torch.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(B, n, cfg.n_embd)
        x = x + blk.attn.proj(y)
        x = x + blk.mlp(blk.ln2(x))
    x = model.ln_f(x[:, -1:, :])
    return model.lm_head(x)[:, -1, :]


@torch.no_grad()
def check_cache(model):
    """True if cached generation gives the same logits as the normal forward pass."""
    cfg = model.cfg
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    ref = model(ids)[0][:, -1, :]
    cache = [None] * cfg.n_layer
    step_logits(model, ids[:, :11], cache, 0)            # "prompt" of 11 tokens
    out = step_logits(model, ids[:, 11:], cache, 11)     # then one more token
    return torch.allclose(ref, out, atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------- sampling
def sample_next(logits, generated, s):
    logits = logits.clone().squeeze(0)                    # (vocab,)
    if s["rep"] != 1.0 and generated:
        ids = torch.tensor(sorted(set(generated[-s["rep_window"]:])))
        vals = logits[ids]
        # repeated words become less likely (works for positive and negative logits)
        logits[ids] = torch.where(vals > 0, vals / s["rep"], vals * s["rep"])
    if s["temp"] <= 0:
        return int(logits.argmax())
    logits = logits / s["temp"]
    if s["topk"] > 0:
        kth = torch.topk(logits, min(s["topk"], logits.numel())).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if s["topp"] < 1.0:                                   # keep the smallest set with mass >= topp
        sp, si = torch.sort(probs, descending=True)
        sp[sp.cumsum(0) - sp > s["topp"]] = 0.0
        probs = torch.zeros_like(probs)
        probs[si] = sp
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))


def generate_reply(model, tok, prompt, s, use_cache):
    cfg = model.cfg
    ids = tok.encode(prompt)[-(cfg.block_size - 32):]     # always leave room to answer
    max_new = min(s["len"], cfg.block_size - len(ids))
    streamer, gen = Streamer(tok, STOPS), []
    cache = [None] * cfg.n_layer
    x, pos0, t0 = torch.tensor([ids], dtype=torch.long), 0, time.time()
    with torch.no_grad():
        for _ in range(max_new):
            if use_cache:
                logits = step_logits(model, x, cache, pos0)
                pos0 += x.size(1)
            else:
                logits = model(torch.tensor([ids + gen], dtype=torch.long))[0][:, -1, :]
            nxt = sample_next(logits, gen, s)
            if nxt == tok.eot_id:
                break
            gen.append(nxt)
            print(streamer.push(nxt), end="", flush=True)
            if streamer.done:
                break
            x = torch.tensor([[nxt]], dtype=torch.long)
    print(streamer.finish())
    dt = max(time.time() - t0, 1e-6)
    print(f"  [{len(gen)} tokens, {len(gen) / dt:.1f} tok/s]")


# ------------------------------------------------------------------------ setup
def find_ckpt(arg):
    if arg:
        return Path(arg)
    preferred = ROOT / "checkpoints" / "10m" / "best.pt"
    if preferred.exists():
        return preferred
    found = sorted((ROOT / "checkpoints").glob("*/best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if found:
        return found[0]
    sys.exit("No best.pt found. Put it in checkpoints/10m/ or pass --ckpt PATH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default=None, help="answer this one message and exit")
    ap.add_argument("--raw", action="store_true", help="with --prompt: plain text continuation")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--topp", type=float, default=0.95)
    ap.add_argument("--rep", type=float, default=1.15, help="repetition penalty (1 = off)")
    ap.add_argument("--len", type=int, default=150, help="max new tokens")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    if a.seed is not None:
        torch.manual_seed(a.seed)
        random.seed(a.seed)

    path = find_ckpt(a.ckpt)
    ck = torch.load(path, map_location="cpu")
    cfg = GPTConfig(**ck["cfg"])
    cfg.use_sdpa = False
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    # the tokenizer MUST be the one the model was trained with
    if a.tokenizer:
        tok_path = Path(a.tokenizer)
    elif (path.parent / "tokenizer.json").exists():
        tok_path = path.parent / "tokenizer.json"
    else:
        tok_path = ROOT / "tokenizer" / "tokenizer.json"
    tok = BPETokenizer.load(tok_path)
    if tok.vocab_size != cfg.vocab_size:
        sys.exit(f"Tokenizer has {tok.vocab_size} tokens but the model expects {cfg.vocab_size}.")

    use_cache = not a.no_cache
    if use_cache and not check_cache(model):
        print("WARNING: cached generation did not match the normal forward pass on this "
              "device -> using the slow but safe mode.")
        use_cache = False

    print(f"loaded {path}")
    print(f"  {model.num_params():,} parameters | trained {ck.get('iter', '?')} iters | "
          f"best val loss {ck.get('best_val', float('nan')):.3f} | tokenizer: {tok_path}")
    print(f"  generation: {'KV-cache (fast)' if use_cache else 'no cache (slow)'}")

    s = {"temp": a.temp, "topk": a.topk, "topp": a.topp, "rep": a.rep,
         "rep_window": 64, "len": a.len}

    def build_prompt(text, mode):
        text = " ".join(text.split())                    # training data had single-line turns
        return f"User: {text}\nBot:" if mode == "chat" else text

    if a.prompt is not None:
        mode = "text" if a.raw else "chat"
        if mode == "chat":
            print(f"User: {a.prompt}\nBot:", end="")
            generate_reply(model, tok, build_prompt(a.prompt, mode), s, use_cache)
        else:
            print(a.prompt, end="")
            generate_reply(model, tok, a.prompt, s, use_cache)
        return

    help_text = (
        "commands:  /mode chat|text   chat = answer a question, text = continue your text\n"
        "           /temp X   randomness (0 = always the most likely word, 0.7 default, 1.0 wild)\n"
        "           /topk N   /topp X   limit the choice of next words\n"
        "           /rep X    repetition penalty (1.0 = off, 1.15 default, 1.3 strong)\n"
        "           /len N    max reply length in tokens\n"
        "           /settings /help /quit      (Ctrl+C stops a reply)")
    print("\nType a message. /help for commands, /quit to exit.\n")
    mode = "chat"
    names = {"temp": float, "topk": int, "topp": float, "rep": float, "len": int}
    while True:
        try:
            line = input(f"You{' (text)' if mode == 'text' else ''}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            cmd, arg = cmd.lower(), arg.strip()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print(help_text)
            elif cmd == "settings":
                print(f"mode={mode}  " + "  ".join(f"{k}={v}" for k, v in s.items()))
            elif cmd == "mode" and arg in ("chat", "text"):
                mode = arg
                print(f"mode: {mode}")
            elif cmd in names:
                try:
                    s[cmd] = names[cmd](arg)
                    print(f"{cmd} = {s[cmd]}")
                except ValueError:
                    print(f"usage: /{cmd} <{names[cmd].__name__}>")
            else:
                print("unknown command, try /help")
            continue
        prompt = build_prompt(line, mode)
        print("Bot:" if mode == "chat" else "...", end="", flush=True)
        try:
            generate_reply(model, tok, prompt, s, use_cache)
        except KeyboardInterrupt:
            print("\n  [stopped]")


if __name__ == "__main__":
    main()
