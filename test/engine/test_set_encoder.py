"""
Set encoder (P6.2.2): permuting players (and the pairwise table accordingly) changes
h_t by < 1e-5; masking a dead player's features (keeping the alive bit) changes
nothing; absent utility slots change nothing; one sample's output does not depend on
the batch; the pairwise bias is live (and absent with pairwise=False).

    .venv\\Scripts\\python.exe test\\engine\\test_set_encoder.py [--cpu]
"""

import numpy as np
import torch

from _harness import check, section, summary
from _fake_states import fake_arrays, finish, test_device

from engine import config
from engine.data.dataset import FEATURE_LAYOUT, PAIR_CHANNELS, PLAYER_LAYOUT, SNAPSHOT_KEYS, UTIL_LAYOUT
from engine.model import batching
from engine.model.wp_model import WPModel

DEV = test_device()
print(f"device: {DEV}")
cfg = config.load()
snaps, vocab, arrs, norm = fake_arrays(8, seed=2)
torch.manual_seed(0)
model = WPModel(cfg, vocab.sizes()).to(DEV).eval()
model.set_norm(norm)


def tb(a: dict, idx) -> dict:
    return batching.to_device({k: a[k][idx] for k in SNAPSHOT_KEYS}, DEV)


@torch.no_grad()
def enc(a, idx):
    h, po = model.encode_full(tb(a, idx))
    return h.double().cpu().numpy(), po.double().cpu().numpy()


idx = np.arange(0, 64)
h0, po0 = enc(arrs, idx)
check("h_t shape [B, d_pool]", h0.shape == (64, cfg.model.d_pool))

section("player permutation invariance")
rng = np.random.default_rng(0)
worst = 0.0
worst_po = 0.0
for _ in range(5):
    perm = rng.permutation(10)
    a2 = dict(arrs)
    for k in ("player_num", "player_cat", "player_mask"):
        a2[k] = arrs[k][:, perm]
    a2["pair"] = arrs["pair"][:, perm][:, :, perm]
    h1, po1 = enc(a2, idx)
    worst = max(worst, float(np.abs(h1 - h0).max()))
    worst_po = max(worst_po, float(np.abs(po1 - po0[:, perm]).max()))
check("permuting players + pair changes h_t by < 1e-5", worst < 1e-5, f"max |diff| {worst:.2e}")
check("player outputs permute with the players", worst_po < 1e-5, f"max |diff| {worst_po:.2e}")
a3 = dict(arrs)
perm = rng.permutation(10)
for k in ("player_num", "player_cat", "player_mask"):
    a3[k] = arrs[k][:, perm]                      # pair NOT permuted -> should differ
h3, _ = enc(a3, idx)
check("permuting players without the pair table does change h_t (bias is live)",
      float(np.abs(h3 - h0).max()) > 1e-4)

section("dead player features are masked")
ALIVE = PLAYER_LAYOUT.col("alive")
dead = np.argwhere(arrs["player_num"][:64, :, ALIVE] < 0.5)
check("fixture has dead players", len(dead) > 0)
a4 = {k: v.copy() for k, v in arrs.items()}
for b, j in dead:
    row = a4["player_num"][b, j]
    keep = row[ALIVE]
    row[:] = np.random.default_rng(int(b * 10 + j)).normal(0, 50, row.shape)
    row[ALIVE] = keep
    a4["player_cat"][b, j, FEATURE_LAYOUT["player_cat"].index("callout")] = 3
    a4["player_cat"][b, j, FEATURE_LAYOUT["player_cat"].index("weapon_class")] = 4
    a4["player_cat"][b, j, FEATURE_LAYOUT["player_cat"].index("player_id")] = 5
    a4["pair"][b, j, :, PAIR_CHANNELS.index("dist_units")] = 9.0
    a4["pair"][b, :, j, PAIR_CHANNELS.index("dist_units")] = 9.0
h4, _ = enc(a4, idx)
check("dead players' features (alive bit kept) change nothing", float(np.abs(h4 - h0).max()) < 1e-6,
      f"max |diff| {float(np.abs(h4 - h0).max()):.2e}")
a5 = {k: v.copy() for k, v in arrs.items()}
b, j = dead[0]
a5["player_mask"][b, j] = False
h5, _ = enc(a5, idx)
check("dropping a dead token entirely does change h_t (present vs absent are distinct)",
      float(np.abs(h5[b] - h0[b]).max()) > 1e-6)

section("absent utility slots are masked")
a6 = {k: v.copy() for k, v in arrs.items()}
um = a6["util_mask"]
a6["util_num"][~um] = 123.0
a6["util_cat"][~um] = 4
h6, _ = enc(a6, idx)
check("garbage in absent utility slots changes nothing", float(np.abs(h6 - h0).max()) < 1e-6)
has_u = np.flatnonzero(um[:64].any(1))
if len(has_u):
    a7 = {k: v.copy() for k, v in arrs.items()}
    bu = has_u[0]
    ku = np.flatnonzero(um[bu])[0]
    a7["util_num"][bu, ku, UTIL_LAYOUT.col("x")] += 500
    h7, _ = enc(a7, idx)
    check("a present utility token does matter", float(np.abs(h7[bu] - h0[bu]).max()) > 1e-6)

section("batch independence")
h_single = np.concatenate([enc(arrs, [i])[0] for i in (0, 17, 63)])
d = float(np.abs(h_single - h0[[0, 17, 63]]).max())
check("one sample's h_t does not depend on the batch (< 1e-5)", d < 1e-5, f"max |diff| {d:.2e}")
a8 = {k: v.copy() for k, v in arrs.items()}
a8["player_num"][1:64] *= 3.0
h8, _ = enc(a8, idx)
check("changing other samples leaves sample 0 unchanged", float(np.abs(h8[0] - h0[0]).max()) < 1e-6)

section("pairwise switch")
torch.manual_seed(0)
nopair = WPModel(cfg, vocab.sizes(), pairwise=False).to(DEV).eval()
nopair.set_norm(norm)
with torch.no_grad():
    a9 = dict(arrs)
    a9["pair"] = arrs["pair"].copy()
    a9["pair"][..., PAIR_CHANNELS.index("los")] = np.where(np.isnan(arrs["pair"][..., 0]), np.nan,
                                                         1.0 - arrs["pair"][..., 0])
    hp0 = nopair.encode(tb(arrs, idx))
    hp1 = nopair.encode(tb(a9, idx))
    hq1 = model.encode(tb(a9, idx)).double().cpu().numpy()
check("pairwise=False ignores the pair table", float((hp0 - hp1).abs().max()) == 0.0)
check("pairwise=True reads LOS", float(np.abs(hq1 - h0).max()) > 1e-5)

finish(summary())
