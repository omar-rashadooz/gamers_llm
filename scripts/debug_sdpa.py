"""Find out WHY PyTorch's fused attention differs from ours on this device.
Every variant is compared with a float64 reference (the 'true' answer)."""
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, nh, T, hd = 2, 6, 64, 32
sdpa = F.scaled_dot_product_attention


def manual(q, k, v):
    att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    mask = torch.tril(torch.ones(T, T, device=q.device))
    att = att.masked_fill(mask == 0, float("-inf"))
    return F.softmax(att, dim=-1) @ v


def noncontig(t):
    # same values, but laid out in memory as (B, T, nh, hd) like inside the model
    return t.transpose(1, 2).contiguous().transpose(1, 2)


q, k, v = (torch.randn(B, nh, T, hd) for _ in range(3))
ref = manual(q.double(), k.double(), v.double())
bool_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))


def show(name, y):
    print(f"{name:42s} max diff vs float64: {(y.double() - ref).abs().max().item():.2e}")


print("torch", torch.__version__, "| threads", torch.get_num_threads())
show("manual (float32)", manual(q, k, v))
show("sdpa, contiguous, is_causal", sdpa(q, k, v, is_causal=True))
show("sdpa, model layout, is_causal", sdpa(noncontig(q), noncontig(k), noncontig(v), is_causal=True))
show("sdpa, contiguous, explicit mask", sdpa(q, k, v, attn_mask=bool_mask))
show("sdpa, model layout, explicit mask",
     sdpa(noncontig(q), noncontig(k), noncontig(v), attn_mask=bool_mask))
