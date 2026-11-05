"""Causal self-attention, written by hand.

Every token creates three vectors:
    query (q): "what am I looking for?"
    key   (k): "what do I contain?"
    value (v): "what will I give you if you pick me?"
The score between token i and token j is  q_i . k_j.  A softmax turns the
scores into weights, and the output of token i is the weighted sum of the
values. "Causal" means token i may only look at tokens j <= i (the past),
otherwise the model could cheat by reading the answer it must predict.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_sdpa = cfg.use_sdpa
        self.dropout = cfg.dropout

        # one matrix produces q, k and v for ALL heads at once
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        # mixes the heads' outputs back together
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # lower-triangular matrix of ones: 1 = "may look", 0 = "future, hidden"
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer(
            "mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False)

    def forward(self, x):
        B, T, C = x.shape                       # batch, time (tokens), channels

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)       # each (B, T, C)

        # split C into n_head pieces; move heads next to batch so each head
        # works independently:  (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            # same maths, done by one optimised PyTorch kernel
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True)
        else:
            # scores[b, h, i, j] = how much token i wants to look at token j
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # hide the future: -inf becomes 0 after softmax
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)        # each row sums to 1
            att = self.attn_drop(att)
            y = att @ v                         # weighted sum of the values

        # put the heads side by side again: (B, n_head, T, hd) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
