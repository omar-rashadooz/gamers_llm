"""Sanity checks for the model. Run this BEFORE training anything.

    python scripts/check_model.py                 # default preset (3.5m)
    python scripts/check_model.py --preset 3.0m
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import GPT, GPTConfig, PRESETS, DEFAULT_PRESET  # noqa: E402

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS))
    a = ap.parse_args()

    torch.manual_seed(0)
    cfg = GPTConfig.from_preset(a.preset)
    model = GPT(cfg)
    n = model.num_params()
    print(f"preset {a.preset}: {n:,} parameters | {cfg}\n")

    # 1) parameter count matches the formula in config.py
    report("parameter count == formula", n == cfg.estimate_params(),
           f"{n:,} vs {cfg.estimate_params():,}")

    # 2) a model that knows nothing guesses uniformly: loss should be ~ln(vocab)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (8, 64))
    y = torch.randint(0, cfg.vocab_size, (8, 64))
    logits, loss = model(x, y)
    expected = math.log(cfg.vocab_size)
    report("output shape", tuple(logits.shape) == (8, 64, cfg.vocab_size),
           str(tuple(logits.shape)))
    report("initial loss ~ ln(vocab)", abs(loss.item() - expected) < 0.3,
           f"{loss.item():.3f} vs {expected:.3f}")

    # 3) causality: changing a FUTURE token must not change earlier outputs
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % cfg.vocab_size
    l1, _ = model(x)
    l2, _ = model(x2)
    report("causal (future cannot leak into the past)",
           torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5)
           and not torch.allclose(l1[:, -1], l2[:, -1], atol=1e-5))

    # 4) our hand-written attention == PyTorch's fused attention
    #    (same model, same weights; we only flip the switch inside attention)
    def set_sdpa(flag):
        for m in model.modules():
            if hasattr(m, "use_sdpa"):
                m.use_sdpa = flag

    set_sdpa(True)
    l3, _ = model(x)
    set_sdpa(False)
    diff = (l1 - l3).abs().max().item()
    report("manual attention == F.scaled_dot_product_attention",
           diff < 1e-4, f"max diff {diff:.2e}")
    if diff >= 1e-4:
        print("      -> not blocking: training uses the manual attention by default."
              "\n         run: python scripts/debug_sdpa.py   and send me the output")

    # 5) can it learn? overfit one tiny batch; the loss must fall a lot.
    #    (stops early; prints progress so you can see the speed of your phone)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    xb = torch.randint(0, cfg.vocab_size, (2, 32))
    yb = torch.randint(0, cfg.vocab_size, (2, 32))
    t0 = time.time()
    first = last = None
    for step in range(100):
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
        if (step + 1) % 10 == 0:
            print(f"      step {step + 1:3d}  loss {last:.3f}  "
                  f"({(time.time() - t0) / (step + 1):.2f}s/step)", flush=True)
        if last < first * 0.5:
            break
    report("overfits one batch", last < first * 0.5,
           f"loss {first:.2f} -> {last:.2f} in {time.time() - t0:.1f}s")

    # 6) generation works and respects the context window
    model.eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    report("generate", out.shape == (1, 21), str(tuple(out.shape)))
    long_in = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 10))
    out = model.generate(long_in, max_new_tokens=3)
    report("generate with input longer than block_size",
           out.shape == (1, cfg.block_size + 13))

    # 7) real data (if prepare_data.py was run)
    train_bin = ROOT / "data" / "processed" / "train.bin"
    if train_bin.exists():
        import numpy as np
        data = np.memmap(train_bin, dtype=np.uint16, mode="r")
        i = 1000
        chunk = torch.from_numpy(data[i:i + cfg.block_size + 1].astype(np.int64))
        report("real data ids fit the vocab", int(chunk.max()) < cfg.vocab_size,
               f"max id {int(chunk.max())}")
        _, loss = model(chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0))
        print(f"      loss on real text before training: {loss.item():.2f}")
    else:
        print("[skip] real data check (data/processed/train.bin not found)")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
