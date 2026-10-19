"""
Assembles the network (P6.2.5): `WPModel(cfg, vocab_sizes)`, `forward(batch) -> dict`.

    model = WPModel(cfg, vocab.sizes())               # cfg: engine Config, config.Model or dict
    model.set_norm(norm)                              # NormStats fitted on train
    out = model(batch)                                # win [B], hazard [B,10], plant15, defuse15, tte (logits), alive [B,10]
    h   = model.encode(batch)                         # h_t [U, d_pool], the snapshot embedding (P9)
    model.n_params

A batch holds the per-snapshot tensors of dataset.SNAPSHOT_KEYS for U unique snapshots,
and optionally windows over them (`win_idx [B,L]`, `win_mask`, `win_t`, `win_last`, see
batching.gather). Without windows every snapshot is its own single-snapshot window.
Outputs are logits for the window's last snapshot.

Ablation switches (P6.7): `temporal=False` (z_t = h_t, no temporal encoder),
`pairwise=False` (no pairwise attention bias), `aux_heads=False` (win head only).
"""
import torch
import torch.nn as nn

from engine.model.heads import Heads
from engine.model.set_encoder import SetEncoder
from engine.model.temporal import TemporalEncoder
from engine.model.tokens import ALIVE_COL, TokenEmbedder

ARCH_KEYS = ("d_model", "d_pool", "L_set", "L_time", "n_heads", "temporal_window_s")


def arch_dict(cfg) -> dict:
    """The architecture keys from an engine Config, a config.Model, or a dict."""
    m = getattr(cfg, "model", cfg)
    if isinstance(m, dict):
        return {k: m[k] for k in ARCH_KEYS}
    return {k: getattr(m, k) for k in ARCH_KEYS}


class WPModel(nn.Module):
    def __init__(self, cfg, vocab_sizes: dict[str, int], *, temporal: bool = True,
                 pairwise: bool = True, aux_heads: bool = True):
        super().__init__()
        a = arch_dict(cfg)
        self.arch        = a
        self.vocab_sizes = dict(vocab_sizes)
        self.flags       = {"temporal": bool(temporal), "pairwise": bool(pairwise),
                            "aux_heads": bool(aux_heads)}
        d, dp = int(a["d_model"]), int(a["d_pool"])
        if d % int(a["n_heads"]) or dp % int(a["n_heads"]):
            raise ValueError("n_heads must divide d_model and d_pool")
        self.window_s = float(a["temporal_window_s"])
        self.tokens   = TokenEmbedder(vocab_sizes, d)
        self.set_enc  = SetEncoder(d, dp, int(a["L_set"]), int(a["n_heads"]), pairwise=pairwise)
        self.temporal = TemporalEncoder(dp, int(a["L_time"]), int(a["n_heads"]), self.window_s) if temporal else None
        self.heads    = Heads(dp, d, aux=aux_heads)

    # ── info ──

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_norm(self, norm) -> None:
        self.tokens.set_norm(norm)
        self.set_enc.set_norm(norm)

    # ── pieces (also used by predict.Predictor with cached embeddings) ──

    @staticmethod
    def alive(batch: dict) -> torch.Tensor:
        return (batch["player_num"][..., ALIVE_COL] > 0.5) & batch["player_mask"]

    def encode_full(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(h_t [U,d_pool], players_out [U,10,d_model]) for every snapshot in the batch."""
        ctx, players, util = self.tokens(batch)
        alive = self.alive(batch)
        return self.set_enc(ctx, players, util, batch["player_mask"], batch["util_mask"],
                            batch["pair"], alive)

    def encode(self, batch: dict) -> torch.Tensor:
        """h_t, the snapshot embedding exposed for P9 (neighbours, Q/B) and P7.2.7 (OOD)."""
        return self.encode_full(batch)[0]

    def temporal_z(self, hw: torch.Tensor, t: torch.Tensor, mask: torch.Tensor,
                   last: torch.Tensor) -> torch.Tensor:
        """hw [B,L,d_pool] window embeddings -> z_t [B,d_pool] at each window's last slot."""
        rows = torch.arange(hw.shape[0], device=hw.device)
        if self.temporal is None:
            return hw[rows, last]
        return self.temporal(hw, t, mask)[rows, last]

    def head(self, z: torch.Tensor, players_out: torch.Tensor, alive: torch.Tensor) -> dict:
        return self.heads(z, players_out, alive)

    # ── forward ──

    def forward(self, batch: dict) -> dict:
        h, players_out = self.encode_full(batch)
        alive = self.alive(batch)
        if "win_idx" in batch:
            idx, mask, t, last = batch["win_idx"], batch["win_mask"], batch["win_t"], batch["win_last"]
            target = idx[torch.arange(idx.shape[0], device=idx.device), last]
            z = self.temporal_z(h[idx], t, mask, last)
        else:
            U = h.shape[0]
            target = torch.arange(U, device=h.device)
            zeros  = torch.zeros(U, 1, device=h.device)
            z = self.temporal_z(h[:, None], zeros, torch.ones(U, 1, dtype=torch.bool, device=h.device),
                                torch.zeros(U, dtype=torch.long, device=h.device))
        out = self.head(z, players_out[target], alive[target])
        out["alive"] = alive[target]          # [B,10] bool, read by the hazard loss
        return out
