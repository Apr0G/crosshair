"""
Temporal encoder (P6.2.3): causality (changing any snapshot after t leaves z_t
unchanged), single-snapshot windows, non-uniform dt (relative time is used, absolute
time is not), the window_s cutoff, and no NaN from padded rows.

    .venv\\Scripts\\python.exe test\\engine\\test_temporal.py [--cpu]
"""

import numpy as np
import torch

from _harness import check, section, summary
from _fake_states import fake_arrays, finish, test_device

from engine import config
from engine.model import batching
from engine.model.temporal import TemporalEncoder
from engine.model.wp_model import WPModel

DEV = test_device()
print(f"device: {DEV}")
torch.manual_seed(0)
D, L, B = 64, 12, 3
enc = TemporalEncoder(D, 2, 4, window_s=10.0).to(DEV).eval()
h = torch.randn(B, L, D, device=DEV)
t = torch.tensor(np.cumsum(np.random.default_rng(0).uniform(0.01, 0.9, (B, L)), axis=1), dtype=torch.float32, device=DEV)
mask = torch.ones(B, L, dtype=torch.bool, device=DEV)

with torch.no_grad():
    z0 = enc(h, t, mask)

section("causality")
worst = 0.0
for q in (0, 4, 8):
    h2 = h.clone()
    h2[:, q + 1:] = torch.randn_like(h2[:, q + 1:]) * 5
    t2 = t.clone()
    with torch.no_grad():
        z2 = enc(h2, t2, mask)
    worst = max(worst, float((z2[:, :q + 1] - z0[:, :q + 1]).abs().max()))
check("future snapshots do not change z_t (any t)", worst < 1e-6, f"max |diff| {worst:.2e}")
h3 = h.clone()
h3[:, 2] += torch.randn_like(h3[:, 2])   # not a constant: LayerNorm removes constant shifts
with torch.no_grad():
    z3 = enc(h3, t, mask)
check("a past snapshot inside the window does change z_t", float((z3[:, 5] - z0[:, 5]).abs().max()) > 1e-5)

section("single-snapshot windows")
with torch.no_grad():
    z1 = enc(h[:, :1], t[:, :1], mask[:, :1])
check("L = 1 works and is finite", tuple(z1.shape) == (B, 1, D) and bool(torch.isfinite(z1).all()))
check("L = 1 equals the first position of a longer window", float((z1[:, 0] - z0[:, 0]).abs().max()) < 1e-5)

section("non-uniform dt, relative time")
with torch.no_grad():
    z_shift = enc(h, t + 37.5, mask)
    t_warp = t.clone()
    t_warp[:, 3:] += 0.4                       # same order, different gaps
    z_warp = enc(h, t_warp, mask)
check("shifting every time by a constant changes nothing (relative encoding only)",
      float((z_shift - z0).abs().max()) < 1e-4, f"{float((z_shift - z0).abs().max()):.2e}")
check("changing the gaps changes z (actual dt is used)", float((z_warp[:, -1] - z0[:, -1]).abs().max()) > 1e-5)
check("non-uniform dt output finite", bool(torch.isfinite(z_warp).all()))

section("window cutoff and padding")
t_old = t.clone()
t_old[:, 0] = t[:, -1] - 25.0                 # first snapshot far outside the 10 s window
h_old = h.clone()
h_old[:, 0] += torch.randn_like(h_old[:, 0]) * 3
with torch.no_grad():
    za = enc(h, t_old, mask)
    zb = enc(h_old, t_old, mask)
check("a key older than window_s does not affect the last z", float((za[:, -1] - zb[:, -1]).abs().max()) < 1e-6)
pm = mask.clone()
pm[:, 7:] = False
hp = h.clone()
hp[:, 7:] = 1e3
with torch.no_grad():
    zp = enc(hp, t, pm)
    zq = enc(h, t, pm)
check("padded rows give no NaN", bool(torch.isfinite(zp).all()))
check("padded keys do not affect valid rows", float((zp[:, :7] - zq[:, :7]).abs().max()) < 1e-6)

section("through WPModel windows")
cfg = config.load()
snaps, vocab, arrs, norm = fake_arrays(4, seed=5, n_snapshots=40)
torch.manual_seed(1)
model = WPModel(cfg, vocab.sizes()).to(DEV).eval()
model.set_norm(norm)
wi = batching.WindowIndex(arrs["round_key"], arrs["t_s"], cfg.model.temporal_window_s)
i = int(np.argmax(wi.lengths()))
win = wi.window(i)
nxt = win[-1] + 1                                   # next snapshot of the same round, if any
same_round = nxt < len(snaps) and arrs["round_key"][nxt] == arrs["round_key"][i]
with torch.no_grad():
    o1 = model(batching.to_device(batching.gather(arrs, [win]), DEV))
    a2 = {k: v.copy() for k, v in arrs.items()}
    if same_round:
        a2["player_num"][nxt] += 7.0
    o2 = model(batching.to_device(batching.gather(a2, [win]), DEV))
    o_single = model(batching.to_device(batching.gather(arrs, [win[-1:]]), DEV))
check("editing the next snapshot (outside the window) leaves the prediction unchanged",
      same_round and float((o1["win"] - o2["win"]).abs().max()) == 0.0)
check("1-snapshot window through the model is finite", bool(torch.isfinite(o_single["win"]).all()))
check("history matters: full window != 1-snapshot window", float((o1["win"] - o_single["win"]).abs().max()) > 1e-6)

finish(summary())
