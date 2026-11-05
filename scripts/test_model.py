"""Sanity checks for the model. Run:  python scripts/test_model.py [--preset small]

1. parameter count
2. initial loss is about ln(vocab)            (the model starts "clueless")
3. fused attention == hand-written attention  (same math, two implementations)
4. causality: changing FUTURE tokens must not change earlier predictions
5. overfit one real batch: the loss must fall (proves learning works at all)
6. sample some text, and measure speed
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.gpt import GPT, PRESETS, make_config  # noqa: E402
from tokenizer.bpe import BPETokenizer  # noqa: E402


def get_batch(cfg, batch_size, T):
    path = ROOT / "data" / "processed" / "train.bin"
    if path.exists():
        import numpy as np
        data = np.memmap(path, dtype=np.uint16, mode="r")
        g = torch.Generator().manual_seed(0)
        ix = torch.randint(0, len(data) - T - 1, (batch_size,), generator=g)
        x = torch.stack([torch.from_numpy(data[i:i + T].astype("int64")) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + T + 1].astype("int64")) for i in ix])
        return x, y, True
    x = torch.randint(0, cfg.vocab_size, (batch_size, T))
    return x, torch.roll(x, -1, dims=1), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()

    torch.manual_seed(0)
    cfg = make_config(a.preset, dropout=0.0)  # no dropout: cleaner tests
    model = GPT(cfg)
    n = model.num_params()
    print(f"[1] preset '{a.preset}': {n:,} parameters ({n / 1e6:.2f}M)")

    x, y, real = get_batch(cfg, 8, 128)
    print(f"    test batch: {'real data from train.bin' if real else 'random tokens (train.bin not found)'}")

    # 2) initial loss
    model.eval()
    _, loss = model(x, y)
    print(f"[2] initial loss {loss.item():.3f} (expected about {math.log(cfg.vocab_size):.3f})")

    # 3) fused vs manual attention
    with torch.no_grad():
        for b in model.blocks:
            b.attn.flash = True
        l_fast, _ = model(x, y)
        for b in model.blocks:
            b.attn.flash = False
        l_slow, _ = model(x, y)
        for b in model.blocks:
            b.attn.flash = cfg.flash
    diff = (l_fast - l_slow).abs().max().item()
    print(f"[3] fused vs manual attention, max difference {diff:.2e} -> {'OK' if diff < 1e-3 else 'MISMATCH!'}")

    # 4) causality
    with torch.no_grad():
        l1, _ = model(x, y)
        x2 = x.clone()
        x2[:, 64:] = torch.randint(0, cfg.vocab_size, x2[:, 64:].shape)
        l2, _ = model(x2, y)
    d_past = (l1[:, :64] - l2[:, :64]).abs().max().item()
    d_future = (l1[:, 64:] - l2[:, 64:]).abs().max().item()
    ok = d_past < 1e-4 and d_future > 1e-4
    print(f"[4] causality: past changed by {d_past:.1e} (must be ~0), "
          f"future by {d_future:.1e} -> {'OK' if ok else 'LEAK!'}")

    # 5) overfit one batch
    model.train()
    opt = model.configure_optimizers(weight_decay=0.0, lr=3e-3)
    first = None
    t0 = time.time()
    for step in range(a.steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first is None:
            first = loss.item()
        if step % max(1, a.steps // 5) == 0 or step == a.steps - 1:
            print(f"    step {step:4d}  loss {loss.item():.3f}")
    dt = (time.time() - t0) / a.steps
    tok_per_step = x.numel()
    verdict = "OK" if loss.item() < 0.6 * first else "did not learn enough"
    print(f"[5] loss {first:.3f} -> {loss.item():.3f}: {verdict}")
    print(f"    speed: {dt:.2f} s/step = {tok_per_step / dt:,.0f} tokens/s on this device")

    # 6) generate (the model just memorized one batch, so expect odd text)
    tok_path = ROOT / "tokenizer" / "tokenizer.json"
    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
        prompt = torch.tensor([tok.encode("User: How do I")])
        out = model.generate(prompt, max_new_tokens=30, temperature=0.8, top_k=20)
        print("[6] sample:", repr(tok.decode(out[0].tolist())))
    else:
        out = model.generate(torch.zeros((1, 1), dtype=torch.long), 10)
        print("[6] generate works, output shape", tuple(out.shape))


if __name__ == "__main__":
    main()
