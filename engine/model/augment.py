"""
Augmentation for counterfactual robustness (P6.3.3). Training only: `augment(...,
training=False)` returns the batch untouched, and train.py passes `model.training`.

Operates on the torch batch of unique snapshots (world units, see dataset.py), each
augmentation touching only the fields it names:

| name            | p / size (config `augment.*`) | touches                                  |
|-----------------|-------------------------------|------------------------------------------|
| util_drop       | 0.1 per utility token         | util_mask                                |
| pos_jitter      | N(0, 16 u) on x, y            | player_num x, y (alive, present); pair dist_units |
| unflash         | 0.05 per blind alive player   | player_num flash_duration_remaining      |
| los_flip        | 0.02 per known enemy LOS pair | pair los (symmetric)                     |
| dead_token_drop | 0.5 per dead player           | player_mask                              |

Jitter is horizontal (x, y); z is left alone so nobody is pushed through a floor.
dist_units is recomputed from the jittered positions for pairs where it was known;
LOS, callout, grid cell, nav distance and in_smoke are unchanged (P6.3.3: "LOS
unchanged"). Randomness comes from an explicit torch.Generator so a resumed run replays
the same augmentations.
"""
import torch

from engine.data.dataset import PAIR_CHANNELS, PLAYER_LAYOUT

X, Y, Z   = PLAYER_LAYOUT.col("x"), PLAYER_LAYOUT.col("y"), PLAYER_LAYOUT.col("z")
PX        = PLAYER_LAYOUT.col("present_x")
ALIVE     = PLAYER_LAYOUT.col("alive")
FLASH     = PLAYER_LAYOUT.col("flash_duration_remaining")
LOS, DIST = PAIR_CHANNELS.index("los"), PAIR_CHANNELS.index("dist_units")
NAMES     = ("util_drop", "pos_jitter", "unflash", "los_flip", "dead_token_drop")


def _rand(shape, gen, device) -> torch.Tensor:
    return torch.rand(shape, generator=gen, device=gen.device).to(device)


def util_drop(b: dict, p: float, gen) -> None:
    keep = _rand(b["util_mask"].shape, gen, b["util_mask"].device) >= p
    b["util_mask"] = b["util_mask"] & keep


def pos_jitter(b: dict, sigma: float, gen) -> None:
    pn = b["player_num"].clone()
    ok = (pn[..., ALIVE] > 0.5) & (pn[..., PX] > 0.5) & b["player_mask"]
    noise = torch.randn(pn.shape[:-1] + (2,), generator=gen, device=gen.device).to(pn.device) * sigma
    noise = noise * ok.unsqueeze(-1)
    pn[..., X] += noise[..., 0]
    pn[..., Y] += noise[..., 1]
    b["player_num"] = pn
    pos  = pn[..., [X, Y, Z]]
    d    = torch.linalg.vector_norm(pos[:, :, None, :] - pos[:, None, :, :], dim=-1)
    pair = b["pair"].clone()
    known = ~torch.isnan(pair[..., DIST])
    moved = ok[:, :, None] | ok[:, None, :]
    pair[..., DIST] = torch.where(known & moved, d, pair[..., DIST])
    b["pair"] = pair


def unflash(b: dict, p: float, gen) -> None:
    pn = b["player_num"].clone()
    blind = (pn[..., ALIVE] > 0.5) & (pn[..., FLASH] > 0)
    hit = blind & (_rand(blind.shape, gen, pn.device) < p)
    pn[..., FLASH] = torch.where(hit, torch.zeros_like(pn[..., FLASH]), pn[..., FLASH])
    b["player_num"] = pn


def los_flip(b: dict, p: float, gen) -> None:
    pair = b["pair"].clone()
    los  = pair[..., LOS]
    n    = los.shape[-1]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=los.device), diagonal=1)
    hit  = (_rand(los.shape, gen, los.device) < p) & upper & ~torch.isnan(los)
    hit  = hit | hit.transpose(-1, -2)
    pair[..., LOS] = torch.where(hit, 1.0 - los, los)
    b["pair"] = pair


def dead_token_drop(b: dict, p: float, gen) -> None:
    dead = b["player_num"][..., ALIVE] < 0.5
    hit  = dead & (_rand(dead.shape, gen, dead.device) < p)
    b["player_mask"] = b["player_mask"] & ~hit


def augment(batch: dict, aug_cfg, gen: torch.Generator, training: bool,
            enabled: tuple[str, ...] = NAMES) -> dict:
    """Return an augmented shallow copy of `batch` (the input is never modified)."""
    if not training:
        return batch
    b = dict(batch)
    if "util_drop" in enabled and aug_cfg.util_drop > 0:
        util_drop(b, aug_cfg.util_drop, gen)
    if "pos_jitter" in enabled and aug_cfg.pos_jitter_units > 0:
        pos_jitter(b, aug_cfg.pos_jitter_units, gen)
    if "unflash" in enabled and aug_cfg.unflash > 0:
        unflash(b, aug_cfg.unflash, gen)
    if "los_flip" in enabled and aug_cfg.los_flip > 0:
        los_flip(b, aug_cfg.los_flip, gen)
    if "dead_token_drop" in enabled and aug_cfg.dead_token_drop > 0:
        dead_token_drop(b, aug_cfg.dead_token_drop, gen)
    return b
