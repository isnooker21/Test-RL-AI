"""
Physics Specialist Model — v7 Transformer encoder
================================================

PyTorch model for the AI Physics Specialist.

Architecture (v7):
    Input:  X  ∈ R^(B, W, F)   physics-feature window (F = len(FEATURE_COLUMNS); grows with FE list)
    │
    ├──► LayerNorm per feature
    ├──► Linear(F → d_model) + learnable positional embeddings
    ├──► Causal TransformerEncoder (multi-head self-attention)
    │        └── last-token state h_T
    ├──► LayerNorm + Dropout
    ├──► Classification head (3-way softmax)  [P_down, P_noisy, P_up]
    └──► Regression head (μ, log σ)

Loss v7 (see losses.py):
    CE + asymmetric directional penalties + regression (SmoothL1 or NLL).
"""

from __future__ import annotations
from dataclasses import asdict, dataclass

import math
import torch
import torch.nn as nn

from losses import LossConfig, SpecialistLoss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpecialistConfig:
    # Runtime n_features = len(FEATURE_COLUMNS); default only for smoke tests.
    n_features: int = 25
    window: int = 64
    hidden: int = 96  # d_model
    n_heads: int = 8
    ff_mult: int = 4
    n_layers: int = 2
    dropout: float = 0.20
    use_layernorm: bool = True
    n_classes: int = 3
    encoder: str = "transformer"
    # legacy JSON fields from older checkpoints (ignored by forward)
    bidirectional: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive mask: upper triangle = -inf (position i may not attend to j > i)."""
    t = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    u = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    t = t.masked_fill(u, float("-inf"))
    return t


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PhysicsSpecialist(nn.Module):
    def __init__(self, cfg: SpecialistConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.hidden % cfg.n_heads != 0:
            raise ValueError(f"hidden={cfg.hidden} must be divisible by n_heads={cfg.n_heads}")

        self.input_norm = nn.LayerNorm(cfg.n_features)
        self.input_proj = nn.Linear(cfg.n_features, cfg.hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.window, cfg.hidden))
        nn.init.normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        out_dim = cfg.hidden
        self.post_norm = nn.LayerNorm(out_dim) if cfg.use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(cfg.dropout)

        self.cls_head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(out_dim // 2, cfg.n_classes),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(out_dim // 2, 2),
        )

        self._init_weights()
        with torch.no_grad():
            last_lin = self.reg_head[-1]
            last_lin.bias[1].fill_(math.log(1.5))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, W, F)
        """
        cfg = self.cfg
        x = self.input_norm(x)
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        x = x * math.sqrt(cfg.hidden)

        mask = _causal_mask(seq_len, x.device, x.dtype)
        x = self.encoder(x, mask=mask, is_causal=False)
        h = x[:, -1, :]
        h = self.post_norm(h)
        h = self.dropout(h)

        reg = self.reg_head(h)
        mu = reg[:, 0]
        log_sigma = reg[:, 1].clamp(min=-3.0, max=3.0)
        logits = self.cls_head(h)
        return {"logits": logits, "mu": mu, "log_sigma": log_sigma, "embedding": h}


# Re-export for callers using `from physics_specialist import LossConfig, SpecialistLoss`
__all__ = ["SpecialistConfig", "PhysicsSpecialist", "LossConfig", "SpecialistLoss"]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = SpecialistConfig()
    m = PhysicsSpecialist(cfg)
    x = torch.randn(4, cfg.window, cfg.n_features)
    out = m(x)
    print("model parameters: {:,}".format(sum(p.numel() for p in m.parameters())))
    print("logits   :", tuple(out["logits"].shape))
    print("mu       :", tuple(out["mu"].shape))
    print("log_sigma:", tuple(out["log_sigma"].shape))
    print("embedding:", tuple(out["embedding"].shape))

    loss_fn = SpecialistLoss(LossConfig())
    y_cls = torch.tensor([0, 1, 2, 1])
    y_ret = torch.tensor([-1.5, 0.1, 1.7, -0.2])
    losses = loss_fn(out, y_cls, y_ret)
    print("loss     :", float(losses["loss"]))
    print("  ce     :", float(losses["ce"]))
    print("  nll    :", float(losses["nll"]))
