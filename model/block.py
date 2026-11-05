"""One Transformer block = attention (mix information BETWEEN tokens)
                          + MLP       (think about each token ON ITS OWN).
"""
import torch.nn as nn

from .attention import CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)    # widen
        self.act = nn.GELU()                                          # non-linearity
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)  # narrow
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # "x + something" is the residual connection: the block only has to
        # learn a small CORRECTION, and gradients flow straight through the +.
        # LayerNorm is applied BEFORE each part ("pre-norm"): more stable.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
