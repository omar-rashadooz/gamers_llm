"""Training loop.

    python scripts/train.py --bench 5                 # measure speed, then exit
    python scripts/train.py                           # train the default (3.5m) model
    python scripts/train.py --preset 3.0m             # compare another size
    python scripts/train.py --resume                  # continue after a stop

What happens in every step (this is the whole of "training"):
    1. take a random batch of text windows  x  and the same windows shifted by one  y
    2. forward pass: the model predicts, for every position, the next token
    3. loss = how surprised the model was by the real next tokens
    4. backward pass: how should each weight change to reduce the loss?
    5. clip the gradient, then let AdamW update the weights (learning rate follows a schedule)
Every --eval-interval steps we measure the loss on text the model has NEVER
trained on (val.bin). If train loss keeps falling but val loss rises, the
model has started to memorise: that is overfitting.
"""
import argparse
import contextlib
import csv
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import GPT, GPTConfig, PRESETS, DEFAULT_PRESET  # noqa: E402
from tokenizer.bpe import BPETokenizer  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS))
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="windows per micro-step (use 8 on the phone)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-steps per update; effective batch = batch-size * grad-accum")
    ap.add_argument("--max-iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6,
                    help="stop after this many evals without a better val loss")
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--out", default=None, help="checkpoint folder (default checkpoints/<preset>)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    ap.add_argument("--sdpa", action="store_true",
                    help="use PyTorch's fused attention (NOT on Termux: it was unreliable there)")
    ap.add_argument("--bench", type=int, default=0,
                    help="run N steps, print speed + projected total time, then exit")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def get_lr(it, a):
    """Linear warm-up, then cosine decay from --lr down to --min-lr."""
    if it < a.warmup:
        return a.lr * (it + 1) / a.warmup
    if it >= a.max_iters:
        return a.min_lr
    ratio = (it - a.warmup) / max(1, a.max_iters - a.warmup)
    return a.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (a.lr - a.min_lr)


def fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # ------------------------------------------------------------ device
    if a.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(a.device)
    on_cuda = device.type == "cuda"

    # Mixed precision (GPU only): do the heavy maths in 16-bit = faster, less memory.
    amp_dtype = None
    if on_cuda and not a.no_amp:
        # bf16 needs a GPU of the Ampere generation or newer (compute capability >= 8).
        # Colab's free T4 is older, so it gets fp16 (is_bf16_supported() can say "yes"
        # there because it counts slow emulation).
        major = torch.cuda.get_device_capability()[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
           if amp_dtype else contextlib.nullcontext())
    # fp16 can underflow tiny gradients; the scaler multiplies the loss to avoid that
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # -------------------------------------------------------------- data
    proc = ROOT / "data" / "processed"
    train_data = np.memmap(proc / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(proc / "val.bin", dtype=np.uint16, mode="r")

    # ------------------------------------------------- model (or resume)
    out_dir = Path(a.out) if a.out else ROOT / "checkpoints" / a.preset
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"

    ckpt = None
    if a.resume:
        if last_path.exists():
            ckpt = torch.load(last_path, map_location=device)
            cfg = GPTConfig(**ckpt["cfg"])
            cfg.use_sdpa = a.sdpa
        else:
            print(f"--resume given but {last_path} not found: starting fresh")
    if ckpt is None:
        over = dict(block_size=a.block_size, use_sdpa=a.sdpa)
        if a.dropout is not None:
            over["dropout"] = a.dropout
        cfg = GPTConfig.from_preset(a.preset, **over)

    meta_path = proc / "meta.json"
    if meta_path.exists():
        import json
        meta_vocab = json.loads(meta_path.read_text())["vocab_size"]
        assert meta_vocab == cfg.vocab_size, \
            f"data was tokenized with vocab {meta_vocab}, model expects {cfg.vocab_size}"
    if a.sdpa and not on_cuda:
        print("WARNING: --sdpa on CPU. This gave intermittent wrong results on Termux.")

    block, bs = cfg.block_size, a.batch_size
    assert len(val_data) > block + 1, "val.bin is smaller than one context window"

    model = GPT(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]      # weight matrices
    no_decay = [p for p in model.parameters() if p.dim() < 2]    # LayerNorm params
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95))

    it, best_val, bad_evals = 0, float("inf"), 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        it, best_val = ckpt["iter"], ckpt["best_val"]
        bad_evals = ckpt.get("bad_evals", 0)
        print(f"Resumed from {last_path} at iter {it} (best val {best_val:.3f})")

    tokens_per_iter = bs * a.grad_accum * block
    iters_per_epoch = len(train_data) / tokens_per_iter
    print(f"device: {device} | amp: {amp_dtype} | parameters: {model.num_params():,}")
    print(f"train tokens: {len(train_data):,} | tokens/iter: {tokens_per_iter:,} "
          f"| {iters_per_epoch:.0f} iters = 1 epoch")
    print(f"max-iters {a.max_iters} = {a.max_iters / iters_per_epoch:.1f} epochs")

    tok = None
    tok_path = ROOT / "tokenizer" / "tokenizer.json"
    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
        # keep a copy of the tokenizer next to the checkpoints: a model is useless
        # (it prints gibberish) if it is later paired with a different tokenizer
        shutil.copy2(tok_path, out_dir / "tokenizer.json")

    # ----------------------------------------------------------- helpers
    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block - 1, (bs,)).tolist()
        x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()                      # dropout off
        result = {}
        for split in ("train", "val"):
            losses = torch.zeros(a.eval_iters)
            for k in range(a.eval_iters):
                x, y = get_batch(split)
                with ctx:
                    _, loss = model(x, y)
                losses[k] = loss.item()
            result[split] = losses.mean().item()
        model.train()
        return result

    def save(path):
        tmp = str(path) + ".tmp"          # write fully, then rename: safe if killed mid-save
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "cfg": cfg.to_dict(), "iter": it, "best_val": best_val,
                    "bad_evals": bad_evals}, tmp)
        os.replace(tmp, path)

    def sample(prompt, n=60):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(ids, n, temperature=0.8, top_k=40, eot_id=tok.eot_id)
        return tok.decode(out[0].tolist()).replace("\n", " / ")

    def train_step(lr):
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(a.grad_accum):
            x, y = get_batch("train")
            with ctx:
                _, loss = model(x, y)
                loss = loss / a.grad_accum
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)              # so clipping sees the real gradient size
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        scaler.step(opt)
        scaler.update()
        return total, float(gnorm)

    # ------------------------------------------------------------- bench
    if a.bench:
        model.train()
        train_step(get_lr(0, a))                      # warm-up step (not timed)
        t0 = time.time()
        for i in range(a.bench):
            train_step(get_lr(i + 1, a))
        per_iter = (time.time() - t0) / a.bench
        print(f"\n{per_iter:.2f} s/iter | {tokens_per_iter / per_iter:,.0f} tokens/s")
        print(f"projected for {a.max_iters} iters: {fmt_time(per_iter * a.max_iters)} "
              f"(+ evaluation time)")
        return

    # --------------------------------------------------------------- eval
    log_path = out_dir / "log.csv"
    if not log_path.exists():
        log_path.write_text("iter,train_loss,val_loss,lr,epochs\n")
    last_eval_it = it if ckpt is not None else -1
    stop = False

    def do_eval():
        nonlocal best_val, bad_evals, last_eval_it, stop
        last_eval_it = it
        r = estimate_loss()
        gap = r["val"] - r["train"]
        mark = ""
        if r["val"] < best_val:
            best_val, bad_evals, mark = r["val"], 0, "  * new best"
        else:
            bad_evals += 1
        print(f"\n== iter {it} | epochs {it / iters_per_epoch:.2f} | train {r['train']:.3f} "
              f"| val {r['val']:.3f} (ppl {math.exp(r['val']):.0f}) | gap {gap:+.3f}{mark}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([it, f"{r['train']:.4f}", f"{r['val']:.4f}",
                                    f"{get_lr(it, a):.6f}", f"{it / iters_per_epoch:.3f}"])
        if mark:
            save(best_path)
        save(last_path)
        if tok is not None:
            for prompt in ("A video game", "User: How do I"):
                print(f"   [{prompt}] {sample(prompt)}")
        print()
        if bad_evals >= a.patience:
            print(f"Val loss has not improved for {a.patience} evals -> stopping "
                  f"(best val {best_val:.3f}, saved in {best_path.name})")
            stop = True

    # ------------------------------------------------------------- train
    model.train()
    t_log, run_start, it_start = time.time(), time.time(), it
    try:
        while it < a.max_iters and not stop:
            if it % a.eval_interval == 0 and it != last_eval_it:
                do_eval()
                if stop:
                    break
                t_log = time.time()
            lr = get_lr(it, a)
            loss, gnorm = train_step(lr)
            it += 1
            if it % a.log_interval == 0:
                now = time.time()
                per_iter = (now - t_log) / a.log_interval
                t_log = now
                left = per_iter * (a.max_iters - it)
                print(f"iter {it:5d}/{a.max_iters} | loss {loss:.3f} | lr {lr:.2e} | gnorm {gnorm:.2f} "
                      f"| {tokens_per_iter / per_iter:,.0f} tok/s | left ~{fmt_time(left)}", flush=True)
        if it != last_eval_it and not stop:
            do_eval()
    except KeyboardInterrupt:
        print("\nInterrupted: saving last.pt ...")
        save(last_path)
        print(f"Saved at iter {it}. Continue later with:  python scripts/train.py --resume"
              + (f" --out {a.out}" if a.out else ""))
        return

    print(f"Done in {fmt_time(time.time() - run_start)} | {it - it_start} iters "
          f"| best val loss {best_val:.3f} (ppl {math.exp(best_val):.0f}) -> {best_path}")


if __name__ == "__main__":
    main()
