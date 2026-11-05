"""The full GPT: tokens -> vectors -> N blocks -> scores for the next token."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .config import GPTConfig


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # id -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # position -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: the layer that turns a vector back into word scores
        # reuses the embedding table. Saves V*d parameters and works better
        # for small models.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the layers that write into the residual stream,
        # so the sum of many blocks does not blow up at the start.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weights count once

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids.  targets: (B, T) the ids that should come next.
        Returns logits (B, T, vocab) and, if targets are given, the loss."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                               # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, eot_id=None):
        """Continue the token sequence idx (B, T), one token at a time.
        temperature: <1 = safer/more repetitive, >1 = more random, 0 = greedy.
        top_k: only sample among the k most likely tokens.
        eot_id: stop early when every sequence just produced this token."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]       # keep the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                      # only the last position matters

            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, nxt], dim=1)
            if eot_id is not None and bool((nxt == eot_id).all()):
                break
        self.train(was_training)
        return idx
