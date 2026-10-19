"""
Output heads (P6.2.4): win, hazard (10, dead masked), plant15, defuse15, tte.

- win: 1 logit (CT round win), from z_t.
- hazard: 10 logits, one per player, from [player token output of the set encoder,
  Linear(z_t)] (the temporal state broadcast back to the players). Dead or absent
  players get -inf, so sigmoid gives exactly 0. `hazard_raw` (unmasked) is what the
  loss reads, restricted to alive players.
- plant15, defuse15: 1 logit each. tte: regression on log1p(seconds to the next
  engagement).

`aux=False` (ablation `--no-aux-heads`) builds only the win head; the other outputs are
then absent from the dict.
"""
import torch
import torch.nn as nn

from engine.model.tokens import MLP


class Heads(nn.Module):
    def __init__(self, d_pool: int, d_model: int, aux: bool = True):
        super().__init__()
        self.aux = aux
        self.win = MLP(d_pool, d_pool, 1)
        if aux:
            self.plant  = MLP(d_pool, d_pool // 2, 1)
            self.defuse = MLP(d_pool, d_pool // 2, 1)
            self.tte    = MLP(d_pool, d_pool // 2, 1)
            self.z_to_player = nn.Linear(d_pool, d_model)
            self.hazard = MLP(2 * d_model, d_model, 1)

    def forward(self, z: torch.Tensor, players_out: torch.Tensor, alive: torch.Tensor) -> dict:
        """z [B,d_pool], players_out [B,10,d_model], alive [B,10] bool."""
        out = {"win": self.win(z).squeeze(-1)}
        if self.aux:
            zp  = self.z_to_player(z).unsqueeze(1).expand(-1, players_out.shape[1], -1)
            raw = self.hazard(torch.cat([players_out, zp], dim=-1)).squeeze(-1)
            out["hazard_raw"] = raw
            out["hazard"]     = raw.masked_fill(~alive, float("-inf"))
            out["plant15"]    = self.plant(z).squeeze(-1)
            out["defuse15"]   = self.defuse(z).squeeze(-1)
            out["tte"]        = self.tte(z).squeeze(-1)
        return out
