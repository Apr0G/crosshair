"""
Set transformer over [ctx, p0..p9, u0..u15] with pairwise attention bias and attention
pooling (P6.2.2). Permutation-invariant over players.

- `L_set` pre-norm blocks (LN -> attention -> residual, LN -> FFN -> residual).
- Key-padding mask: absent tokens (player_mask / util_mask 0) are never attended to.
  The ctx token is always present, so no query row is ever fully masked.
- Pairwise bias: the 10x10x5 table is z-scored (NormStats `pair` buffers), NaN -> 0
  with a presence channel, entries touching a dead player treated as missing (a dead
  player's features cannot move the output, matching tokens.py), then
  `Linear(10 -> n_heads)` gives one logit per head added to player-player attention
  logits; zeros elsewhere.
- Attention pooling with one learned query -> `Linear(d_model -> d_pool)` = h_t.

Attention is written out (matmul + masked softmax in fp32) instead of
`scaled_dot_product_attention`, so no fused ROCm kernel is ever selected and the
per-head additive bias needs no backend support.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.data.dataset import N_PLAYERS, PAIR_CHANNELS
from engine.model.tokens import MLP

N_PAIR = len(PAIR_CHANNELS)


# ── attention ─────────────────────────────────────────────────────────────────

def masked_softmax_attention(q, k, v, allowed: torch.Tensor, bias: torch.Tensor | None = None):
    """q,k,v [B,H,S,dh]; allowed [B,1|H,S|1,S] bool; bias [B,H,S,S] or None."""
    logits = (q @ k.transpose(-2, -1)).float() / math.sqrt(q.shape[-1])
    if bias is not None:
        logits = logits + bias.float()
    logits = logits.masked_fill(~allowed, float("-inf"))
    attn = torch.softmax(logits, dim=-1).to(v.dtype)
    return attn @ v


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h   = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor, bias: torch.Tensor | None = None):
        B, S, D = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        o = masked_softmax_attention(q, k, v, allowed, bias)
        return self.out(o.transpose(1, 2).reshape(B, S, D))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.n1   = nn.LayerNorm(d)
        self.attn = Attention(d, n_heads)
        self.n2   = nn.LayerNorm(d)
        self.ffn  = MLP(d, ffn_mult * d, d)

    def forward(self, x, allowed, bias=None):
        x = x + self.attn(self.n1(x), allowed, bias)
        return x + self.ffn(self.n2(x))


class AttentionPool(nn.Module):
    """One learned query attends over the valid tokens."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h     = n_heads
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.kv    = nn.Linear(d, 2 * d)
        self.out   = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        k, v = self.kv(x).view(B, S, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q = self.query.to(k.dtype).view(1, self.h, 1, D // self.h).expand(B, -1, -1, -1)
        o = masked_softmax_attention(q, k, v, key_mask[:, None, None, :])
        return self.out(o.reshape(B, D))


# ── pairwise bias ─────────────────────────────────────────────────────────────

class PairBias(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_PAIR))
        self.register_buffer("std", torch.ones(N_PAIR))
        self.lin = nn.Linear(2 * N_PAIR, n_heads)

    def set_norm(self, mean, std) -> None:
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, pair: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        """pair [B,10,10,5] world units with NaN; alive [B,10] bool -> [B,H,10,10]."""
        both    = (alive[:, :, None] & alive[:, None, :]).unsqueeze(-1)
        present = ~torch.isnan(pair) & both
        z       = torch.where(present, (pair - self.mean) / self.std, torch.zeros_like(pair))
        feats   = torch.cat([z, present.to(z.dtype)], dim=-1)
        return self.lin(feats).permute(0, 3, 1, 2)


# ── set encoder ───────────────────────────────────────────────────────────────

class SetEncoder(nn.Module):
    def __init__(self, d_model: int, d_pool: int, n_layers: int, n_heads: int, pairwise: bool = True):
        super().__init__()
        self.n_heads   = n_heads
        self.blocks    = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm      = nn.LayerNorm(d_model)
        self.pool      = AttentionPool(d_model, n_heads)
        self.proj      = nn.Linear(d_model, d_pool)
        self.pair_bias = PairBias(n_heads) if pairwise else None

    def set_norm(self, norm) -> None:
        if self.pair_bias is not None:
            self.pair_bias.set_norm(*norm.arrays("pair"))

    def forward(self, ctx, players, util, player_mask, util_mask, pair, alive):
        """ctx [B,1,D], players [B,10,D], util [B,16,D] -> (h [B,d_pool], players_out [B,10,D])."""
        B = ctx.shape[0]
        x = torch.cat([ctx, players, util], dim=1)
        S = x.shape[1]
        key_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device),
                              player_mask, util_mask], dim=1)
        allowed = key_mask[:, None, None, :]
        bias = None
        if self.pair_bias is not None:
            pb   = self.pair_bias(pair, alive)                           # [B,H,10,10]
            bias = F.pad(pb, (1, S - 1 - N_PLAYERS, 1, S - 1 - N_PLAYERS))
        for blk in self.blocks:
            x = blk(x, allowed, bias)
        x = self.norm(x)
        h = self.proj(self.pool(x, key_mask))
        return h, x[:, 1:1 + N_PLAYERS]
