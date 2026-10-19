"""
Over-fit test (P6.2.6): the full default-size model memorizes one shard of 512 fake
snapshots: total train loss (all five heads, weighted) below 0.05 within 300 steps.

Needs a GPU: without one (or with `--cpu`) it exits 2, unless `--allow-cpu`, which
runs the same check on 64 snapshots.

    .venv\\Scripts\\python.exe test\\engine\\test_overfit_tiny.py
    .venv\\Scripts\\python.exe test\\engine\\test_overfit_tiny.py --cpu --allow-cpu
"""
import sys
import time

import numpy as np
import torch

from _harness import check, section, summary
from _fake_states import fake_arrays, finish, test_device

from engine import config
from engine.data import dataset
from engine.model import batching, train
from engine.model.wp_model import WPModel

DEV = test_device()
if DEV != "cuda" and "--allow-cpu" not in sys.argv:
    print("test_overfit_tiny: no GPU (or --cpu) and no --allow-cpu; prerequisite missing")
    sys.exit(2)
N_SNAP = 512 if DEV == "cuda" else 64
print(f"device: {DEV}, {N_SNAP} snapshots")

cfg = config.load()
snaps, vocab, arrs, norm = fake_arrays(16 if DEV == "cuda" else 4, seed=31, n_snapshots=32 if DEV == "cuda" else 16)
arrs = dataset.take(arrs, np.arange(N_SNAP))
check(f"{N_SNAP} snapshots in the shard", len(arrs["t_s"]) == N_SNAP)

section("<= 300 steps")
# The real training pieces (WindowIndex, TierSampler, gather, compute_loss, AdamW + warm-up
# cosine, grad clip, bf16 autocast on GPU) in a loop that stops as soon as the mean of the
# last 10 losses is below 0.05: on CPU each step is slow (see the report), so the test
# does not run the remaining steps once the target is met. fit() itself: test_train_loop.
train.set_seed(cfg.train.seed)
model = WPModel(cfg, vocab.sizes()).to(DEV)
model.set_norm(norm)
wi   = batching.WindowIndex(arrs["round_key"], arrs["t_s"], model.window_s)
smp  = batching.TierSampler(arrs["tier"], 64, cfg.train.pro_oversample, cfg.train.seed)
opt  = train.make_optimizer(model, 1e-3, cfg.train.wd)
sch  = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: train.lr_factor(k, 30, 300))
bf16 = train.use_bf16(DEV)
losses: list[float] = []
t0 = time.perf_counter()
model.train()
for targets in smp.epoch(0, n_batches=300):
    b = batching.to_device(batching.gather(arrs, [wi.window(i) for i in targets]), DEV)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        out = model(b)
    loss, _ = train.compute_loss(out, b, cfg.model.loss_weights, cfg.model.terminal_weight)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sch.step()
    losses.append(float(loss.detach()))
    if len(losses) >= 10 and np.mean(losses[-10:]) < 0.05:
        break
secs = time.perf_counter() - t0
losses = np.array(losses)
final = float(losses[-10:].mean())
print(f"first loss {losses[0]:.4f}, final (mean of last 10) {final:.4f} after {len(losses)} steps, {secs:.1f} s")
check("ran <= 300 steps", len(losses) <= 300)
check("all losses finite", bool(np.isfinite(losses).all()))
check("train loss < 0.05 within 300 steps (mean of the last 10 steps)", final < 0.05, f"{final:.4f} at step {len(losses)}")

finish(summary())
