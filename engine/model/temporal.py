"""
Causal temporal transformer over the round's last `model.temporal_window_s` seconds with
relative-time encodings (P6.2.3). Counterfactual edits never touch the history.

Input: the snapshot embeddings h of one window per row, in time order, left-aligned and
padded ([B,L,d_pool]), their times t [B,L] (seconds from freeze end) and a mask.
Query q may attend to key k only if k <= q (index causality: windows are time ordered),
k is present, and t_q - t_k <= window_s; every query may attend to itself, so padded
rows never produce NaN. The relative-time encoding is a per-head additive bias computed
from the actual gap t_q - t_k in seconds (an MLP on [gap/window, log1p(gap)] and a few
sinusoids), so non-uniform spacing from boundary samples is represented exactly.

`L_time` pre-norm blocks (the same Block as the set encoder), final LayerNorm; output
z [B,L,d_pool]. The model reads z at each window's last position.
"""
import math

import torch
import torch.nn as nn

from engine.model.set_encoder import Block

N_FREQ = 4


class RelTimeBias(nn.Module):
    def __init__(self, n_heads: int, window_s: float, hidden: int = 32):
        super().__init__()
        self.window_s = float(window_s)
        freqs = torch.tensor([2 * math.pi / (self.window_s * 2 ** -k) for k in range(N_FREQ)])
        self.register_buffer("freqs", freqs, persistent=False)
        self.net = nn.Sequential(nn.Linear(2 + 2 * N_FREQ, hidden), nn.GELU(), nn.Linear(hidden, n_heads))

    def forward(self, gap: torch.Tensor) -> torch.Tensor:
        """gap [B,L,L] seconds (>= 0) -> [B,H,L,L]."""
        g = gap.clamp(min=0.0).unsqueeze(-1)
        ang = g * self.freqs
        feats = torch.cat([g / self.window_s, torch.log1p(g), torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.net(feats).permute(0, 3, 1, 2)


class TemporalEncoder(nn.Module):
    def __init__(self, d: int, n_layers: int, n_heads: int, window_s: float):
        super().__init__()
        self.window_s = float(window_s)
        self.blocks   = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.rel      = RelTimeBias(n_heads, window_s)
        self.norm     = nn.LayerNorm(d)

    def allowed(self, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        L   = t.shape[1]
        ar  = torch.arange(L, device=t.device)
        causal = ar[None, :] <= ar[:, None]                         # [Lq, Lk]: k <= q
        gap = t[:, :, None] - t[:, None, :]                         # t_q - t_k
        ok  = causal[None] & mask[:, None, :] & (gap <= self.window_s + 1e-6)
        ok  = ok | torch.eye(L, dtype=torch.bool, device=t.device)[None]
        return ok

    def forward(self, h: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t32     = t.float()
        allowed = self.allowed(t32, mask).unsqueeze(1)              # [B,1,L,L]
        bias    = self.rel(t32[:, :, None] - t32[:, None, :])
        x = h
        for blk in self.blocks:
            x = blk(x, allowed, bias)
        return self.norm(x)
