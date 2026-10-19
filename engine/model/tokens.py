"""
Token embeddings to d_model (P6.2.1): numeric -> linear, categoricals -> summed
embeddings, None -> learned missing vector + presence bit.

Inputs are the world-unit arrays of engine/data/dataset.py. The z-scoring happens here,
from buffers set by `set_norm(NormStats)` (saved with the checkpoint), so the
model sees normalized values while shards, augmentation and QA keep real units.

Per numeric block: z = (x - mean) / std on continuous columns, values whose presence
bit is 0 are forced to 0, then `Linear(values + presence bits)` plus
`sum_f (1 - present_f) * missing_f` (one learned d_model vector per nullable field),
so a None input differs from a 0 input in two independent ways.

Dead players: their token keeps only `side`, `tier` and `alive = 0`; every other
numeric value is zeroed with its presence bit (-> missing vectors) and every other
categorical becomes NULL. A dead player's features therefore cannot move the output
(P6.2.2 verify), and "dead token present" vs "absent" (player_mask, P6.3.3) are the two
cases the model sees.

Embedding tables are shared by vocabulary name (`player` serves player_id and
utility owner_id, `side` serves player side and utility owner side, `tier` serves
player and context tier).
"""
import torch
import torch.nn as nn

from engine.data.dataset import (CAT_VOCABS, CTX_LAYOUT, NULL, PLAYER_CAT, PLAYER_LAYOUT, UTIL_LAYOUT,
                                 BlockLayout)

ALIVE_COL      = PLAYER_LAYOUT.col("alive")
DEAD_KEEP_CATS = ("side", "tier")


# ── building blocks ───────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NumericEmbed(nn.Module):
    """One numeric block (values + presence bits) -> d_model."""

    def __init__(self, layout: BlockLayout, d_model: int):
        super().__init__()
        n = len(layout)
        self.n_cols = n
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("std", torch.ones(n))
        # for each column: the column holding its presence bit, or n (an appended ones column)
        gidx = [p if p >= 0 else n for p in layout.presence_of] + [n] * (n - layout.n_values)
        self.register_buffer("gate_idx", torch.tensor(gidx, dtype=torch.long), persistent=False)
        pres = [layout.col(f"present_{s}") for s in layout.presence_sources]
        self.register_buffer("pres_idx", torch.tensor(pres, dtype=torch.long), persistent=False)
        self.lin     = nn.Linear(n, d_model)
        self.missing = nn.Parameter(torch.randn(len(pres), d_model) * 0.02)

    def set_norm(self, mean, std) -> None:
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z    = (x - self.mean) / self.std
        ones = torch.ones_like(x[..., :1])
        gate = torch.cat([x, ones], dim=-1).index_select(-1, self.gate_idx)
        z    = z * gate
        out  = self.lin(z)
        if self.pres_idx.numel():
            absent = 1.0 - x.index_select(-1, self.pres_idx)
            out = out + absent @ self.missing
        return out


class TokenMLP(nn.Module):
    """Residual 2-layer MLP on a token embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp  = MLP(d_model, 2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


# ── token embedder ────────────────────────────────────────────────────────────

class TokenEmbedder(nn.Module):
    """batch -> (ctx [B,1,D], players [B,10,D], utility [B,16,D])."""

    def __init__(self, vocab_sizes: dict[str, int], d_model: int):
        super().__init__()
        used = sorted({v for vs in CAT_VOCABS.values() for v in vs})
        missing = [v for v in used if v not in vocab_sizes]
        if missing:
            raise ValueError(f"vocab_sizes lacks {missing}")
        self.emb = nn.ModuleDict({v: nn.Embedding(int(vocab_sizes[v]), d_model) for v in used})
        for e in self.emb.values():
            nn.init.normal_(e.weight, std=0.02)
        self.player_num = NumericEmbed(PLAYER_LAYOUT, d_model)
        self.util_num   = NumericEmbed(UTIL_LAYOUT, d_model)
        self.ctx_num    = NumericEmbed(CTX_LAYOUT, d_model)
        self.player_mlp = TokenMLP(d_model)
        self.util_mlp   = TokenMLP(d_model)
        self.ctx_mlp    = TokenMLP(d_model)
        keep = torch.zeros(len(PLAYER_LAYOUT))
        keep[ALIVE_COL] = 1.0
        self.register_buffer("dead_keep_num", keep, persistent=False)
        keep_cat = torch.tensor([c in DEAD_KEEP_CATS for c, _, _ in PLAYER_CAT])
        self.register_buffer("dead_keep_cat", keep_cat, persistent=False)

    def set_norm(self, norm) -> None:
        for block, mod in (("player_num", self.player_num), ("util_num", self.util_num),
                           ("ctx_num", self.ctx_num)):
            mod.set_norm(*norm.arrays(block))

    def _cats(self, cat: torch.Tensor, block: str) -> torch.Tensor:
        out = 0
        for c, voc in enumerate(CAT_VOCABS[block]):
            out = out + self.emb[voc](cat[..., c])
        return out

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pn, pc = batch["player_num"], batch["player_cat"]
        dead   = (pn[..., ALIVE_COL] < 0.5).unsqueeze(-1)
        pn = torch.where(dead, pn * self.dead_keep_num, pn)
        pc = torch.where(dead & ~self.dead_keep_cat, torch.full_like(pc, NULL), pc)
        players = self.player_mlp(self.player_num(pn) + self._cats(pc, "player_cat"))
        util    = self.util_mlp(self.util_num(batch["util_num"]) + self._cats(batch["util_cat"], "util_cat"))
        ctx     = self.ctx_mlp(self.ctx_num(batch["ctx_num"]) + self._cats(batch["ctx_cat"], "ctx_cat"))
        return ctx.unsqueeze(1), players, util
