"""Option-scoring heads (design §4.4), shared with Tiny-Jev."""

import torch
from torch import nn

INIT_STD = 0.02   # ModernBERT's initializer_range


class PairScorer(nn.Module):
    """logit(anchor, option) = MLP(Wa·LN(a) + Wo·LN(o) + (Wa·LN(a)) ⊙ (Wo·LN(o))).

    The last layer starts small (std INIT_STD), so the model starts near uniform probabilities
    (NLL ~ log K). Not exactly zero: a zero last layer passes zero gradient to everything below
    it, and on real data training then sat at log K for hundreds of steps (notes, 2026-10-01).
    """

    def __init__(self, d: int, h: int | None = None, p: float = 0.1):
        super().__init__()
        h = h or d
        # ModernBERT's last hidden states have a few outlier dims (~50x the median magnitude);
        # without input norms they dominate wa/wo and the head barely learned (notes, 2026-10-01).
        self.na, self.no = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wa, self.wo = nn.Linear(d, h), nn.Linear(d, h)
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, 1))
        nn.init.normal_(self.mlp[-1].weight, std=INIT_STD)
        nn.init.zeros_(self.mlp[-1].bias)
        # Direct linear path on the option token. Without it the MLP alone sat at log K on real
        # data; with it, global decisions learn like a linear head (notes, 2026-10-01).
        self.lin = nn.Linear(d, 1)
        nn.init.normal_(self.lin.weight, std=INIT_STD)
        nn.init.zeros_(self.lin.bias)

    def forward(self, a: torch.Tensor, o: torch.Tensor) -> torch.Tensor:  # a [G, d], o [G, K, d]
        no = self.no(o)
        ha, ho = self.wa(self.na(a))[:, None], self.wo(no)
        return (self.lin(no) + self.mlp(ha + ho + ha * ho)).squeeze(-1)


class LinearScorer(nn.Module):
    """Ablation A5: Nano-style linear(o); the anchor is ignored."""

    def __init__(self, d: int, p: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.lin = nn.Linear(d, 1)
        nn.init.normal_(self.lin.weight, std=INIT_STD)
        nn.init.zeros_(self.lin.bias)

    def forward(self, a: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        return self.lin(self.drop(o)).squeeze(-1)


def build_head(kind: str, d: int, p: float = 0.1) -> nn.Module:
    if kind == "pair":
        return PairScorer(d, p=p)
    if kind == "linear":
        return LinearScorer(d, p=p)
    raise ValueError(f"unknown head {kind!r}")
