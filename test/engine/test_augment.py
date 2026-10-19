"""
Augmentation (P6.3.3): each augmentation changes only the fields it claims (forced
with p = 1 so the change is visible), dist_units is recomputed from the jittered
positions, LOS flips stay symmetric and never touch unknown pairs, everything is off
at eval, defaults are on in training, and a seeded generator replays exactly.

    .venv\\Scripts\\python.exe test\\engine\\test_augment.py [--cpu]
"""
import dataclasses

from _harness import check, require, section, summary
torch = require("torch", "needs the platform torch build (docs/ENGINE_ENV.md)")   # LOW-61
from _fake_states import fake_arrays, finish, test_device  # noqa: E402

from engine import config  # noqa: E402
from engine.data.dataset import PAIR_CHANNELS, PLAYER_LAYOUT, SNAPSHOT_KEYS  # noqa: E402
from engine.model import augment as aug  # noqa: E402
from engine.model import batching  # noqa: E402

DEV = test_device()
print(f"device: {DEV}")
cfg = config.load()
snaps, vocab, arrs, norm = fake_arrays(6, seed=21)
base = batching.to_device({k: arrs[k] for k in SNAPSHOT_KEYS}, DEV)
forced = dataclasses.replace(cfg.augment, util_drop=1.0, pos_jitter_units=16.0, unflash=1.0,
                             los_flip=1.0, dead_token_drop=1.0)
P = PLAYER_LAYOUT
ALIVE = base["player_num"][..., P.col("alive")] > 0.5
LOS, DIST = PAIR_CHANNELS.index("los"), PAIR_CHANNELS.index("dist_units")


def run(name: str, c=forced, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return aug.augment(base, c, g, training=True, enabled=(name,))


def changed_keys(out: dict) -> set[str]:
    return {k for k in SNAPSHOT_KEYS if not torch.equal(torch.nan_to_num(out[k].float(), nan=-7.0),
                                                        torch.nan_to_num(base[k].float(), nan=-7.0))}


def changed_cols(a: torch.Tensor, b: torch.Tensor) -> set[int]:
    diff = (torch.nan_to_num(a, nan=-7.0) != torch.nan_to_num(b, nan=-7.0))
    return set(torch.nonzero(diff.reshape(-1, a.shape[-1]).any(0)).flatten().tolist())


section("util_drop")
o = run("util_drop")
check("touches only util_mask", changed_keys(o) == {"util_mask"}, str(changed_keys(o)))
check("p = 1 drops every utility token", not bool(o["util_mask"].any()))

section("pos_jitter")
o = run("pos_jitter")
check("touches only player_num and pair", changed_keys(o) == {"player_num", "pair"}, str(changed_keys(o)))
check("player_num: only x and y columns", changed_cols(o["player_num"], base["player_num"]) == {P.col("x"), P.col("y")})
check("pair: only dist_units", changed_cols(o["pair"], base["pair"]) == {DIST})
moved = (o["player_num"][..., P.col("x")] != base["player_num"][..., P.col("x")])
check("only alive players move", not bool((moved & ~ALIVE).any()) and bool(moved.any()))
pos = o["player_num"][..., [P.col("x"), P.col("y"), P.col("z")]]
d = torch.linalg.vector_norm(pos[:, :, None] - pos[:, None], dim=-1)
known = ~torch.isnan(base["pair"][..., DIST])
check("dist_units recomputed from the jittered positions", bool(torch.allclose(o["pair"][..., DIST][known], d[known], atol=1e-2)))
std = float((o["player_num"][..., P.col("x")] - base["player_num"][..., P.col("x")])[ALIVE].std())
check("jitter std ~ 16 units", 12 < std < 20, f"{std:.2f}")
check("LOS unchanged by jitter", torch.equal(torch.nan_to_num(o["pair"][..., LOS], nan=-1), torch.nan_to_num(base["pair"][..., LOS], nan=-1)))

section("unflash")
o = run("unflash")
blind = ALIVE & (base["player_num"][..., P.col("flash_duration_remaining")] > 0)
check("fixture has blind players", bool(blind.any()))
check("touches only player_num flash_duration_remaining", changed_keys(o) <= {"player_num"}
      and changed_cols(o["player_num"], base["player_num"]) == {P.col("flash_duration_remaining")})
check("p = 1 zeroes every blind player's flash", bool((o["player_num"][..., P.col("flash_duration_remaining")][blind] == 0).all()))

section("los_flip")
o = run("los_flip")
check("touches only pair los", changed_keys(o) == {"pair"} and changed_cols(o["pair"], base["pair"]) == {LOS})
lo, lb = o["pair"][..., LOS], base["pair"][..., LOS]
check("unknown (NaN) LOS never touched", torch.equal(torch.isnan(lo), torch.isnan(lb)))
fin = ~torch.isnan(lb)
check("p = 1 flips every known bit", bool((lo[fin] == 1 - lb[fin]).all()))
check("flips stay symmetric", torch.equal(torch.nan_to_num(lo, nan=-1), torch.nan_to_num(lo.transpose(1, 2), nan=-1)))
half = dataclasses.replace(forced, los_flip=0.3)
oh = run("los_flip", half, seed=3)
lh = oh["pair"][..., LOS]
check("partial flips stay symmetric", torch.equal(torch.nan_to_num(lh, nan=-1), torch.nan_to_num(lh.transpose(1, 2), nan=-1)))

section("dead_token_drop")
o = run("dead_token_drop")
check("touches only player_mask", changed_keys(o) == {"player_mask"})
dropped = base["player_mask"] & ~o["player_mask"]
check("only dead players dropped", not bool((dropped & ALIVE).any()))
check("p = 1 drops every dead token", torch.equal(dropped, base["player_mask"] & ~ALIVE))

section("eval / defaults / determinism")
g = torch.Generator().manual_seed(0)
off = aug.augment(base, forced, g, training=False)
check("training=False returns the batch untouched", off is base)
check("defaults on in training (every probability > 0)",
      all(getattr(cfg.augment, k) > 0 for k in ("util_drop", "pos_jitter_units", "unflash", "los_flip", "dead_token_drop")))
a1 = aug.augment(base, cfg.augment, torch.Generator().manual_seed(9), training=True)
a2 = aug.augment(base, cfg.augment, torch.Generator().manual_seed(9), training=True)
check("same generator seed -> identical augmentation",
      all(torch.equal(torch.nan_to_num(a1[k].float(), nan=-7), torch.nan_to_num(a2[k].float(), nan=-7)) for k in SNAPSHOT_KEYS))
check("input batch never modified in place", torch.equal(base["util_mask"], batching.to_device({"m": arrs["util_mask"]}, DEV)["m"]))

finish(summary())
