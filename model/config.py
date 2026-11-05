"""Model configuration. Change the model size HERE (or with --preset).

Parameter count (weights of the output layer are shared with the token
embedding, so they are counted once):

    V*d  +  T*d  +  L * (12*d^2 + 4*d)  +  2*d
    ^        ^              ^                ^
    tokens   positions      transformer      final LayerNorm
    table    table          layers
"""
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int = 4096   # must match the tokenizer
    block_size: int = 256    # context length (tokens the model can see)
    n_layer: int = 6         # number of Transformer blocks
    n_head: int = 6          # attention heads (n_embd must divide by it)
    n_embd: int = 192        # width of every token vector
    dropout: float = 0.1     # regularisation; helps because our data is small
    # True = PyTorch's fused attention (faster on GPU). Keep False on Termux:
    # there it gave intermittently WRONG results (see scripts/debug_sdpa.py).
    use_sdpa: bool = False

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must divide by n_head"

    def estimate_params(self):
        V, T, d, L = self.vocab_size, self.block_size, self.n_embd, self.n_layer
        return V * d + T * d + L * (12 * d * d + 4 * d) + 2 * d

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_preset(cls, name, **overrides):
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})


# name -> only the fields that differ from the defaults above
PRESETS = {
    "1.3m": dict(n_layer=4, n_head=4, n_embd=128),
    "2.6m": dict(n_layer=4, n_head=6, n_embd=192),
    "3.0m": dict(n_layer=5, n_head=6, n_embd=192),
    "3.5m": dict(n_layer=6, n_head=6, n_embd=192),   # default
    "5.8m": dict(n_layer=6, n_head=4, n_embd=256),
    "10m": dict(n_layer=7, n_head=5, n_embd=320),
}
DEFAULT_PRESET = "10m"


if __name__ == "__main__":
    for name in PRESETS:
        c = GPTConfig.from_preset(name)
        print(f"{name:>5}: {c.estimate_params():>9,} parameters")
