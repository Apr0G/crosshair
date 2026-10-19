"""
Temporal windows, collation and the tier-aware sampler (P5.3.1, P6.2.3, P6.3.2):
window membership (same round, last `window_s` seconds, time order, ends at the
snapshot), lengths, masks, relative dt, batch gathering, and the observed pro share in
1,000 batches within +-2% of the oversampling target.

    .venv\\Scripts\\python.exe test\\engine\\test_batching.py [--cpu]
"""
import sys

import numpy as np

from _harness import check, section, summary
from _fake_states import fake_arrays

from engine.model import batching

W = 10.0
snaps, vocab, arrs, norm = fake_arrays(12, seed=3, n_snapshots=40)
N = len(snaps)
rng = np.random.default_rng(0)
perm = rng.permutation(N)                      # windows must not rely on input order
sh = {k: v[perm] for k, v in arrs.items()}

# ── windows ──
section("windows")
wi = batching.WindowIndex(sh["round_key"], sh["t_s"], W)
ok_member = ok_order = ok_end = True
for i in range(0, N, 7):
    w = wi.window(i)
    rk, t = sh["round_key"][i], sh["t_s"][i]
    want = {j for j in range(N) if sh["round_key"][j] == rk and t - W - 1e-6 <= sh["t_s"][j] <= t}
    ok_member &= set(w.tolist()) == want
    ok_order &= bool(np.all(np.diff(sh["t_s"][w]) > 0))
    ok_end &= w[-1] == i
check("window = same round, t_i - W <= t_j <= t_i (brute force)", ok_member)
check("window in time order", ok_order)
check("window ends at its own snapshot", ok_end)
lens = wi.lengths()
check("lengths() equals len(window(i))", all(lens[i] == len(wi.window(i)) for i in range(0, N, 5)))
first = [i for i in range(N) if sh["t_s"][i] == 0.0]
check("first snapshot of a round has a 1-snapshot window", all(lens[i] == 1 for i in first) and len(first) == 12)
check("windows vary in length (boundary samples)", len(set(lens.tolist())) > 5)
check("round count", wi.n_rounds == 12)
wi_short = batching.WindowIndex(sh["round_key"], sh["t_s"], 1.0)
check("shorter window_s gives shorter windows", int(wi_short.lengths().max()) < int(lens.max()))

# ── collate ──
section("collate")
targets = [3, 50, 100, 7]
win = [wi.window(i) for i in targets]
c = batching.collate(win, sh["t_s"])
L = max(len(w) for w in win)
check("padded to the longest window", c["idx"].shape == (4, L) and c["mask"].shape == (4, L))
check("mask counts = window lengths", c["mask"].sum(1).tolist() == [len(w) for w in win])
check("last = length - 1", c["last"].tolist() == [len(w) - 1 for w in win])
check("idx matches the windows", all(np.array_equal(c["idx"][b, :len(w)], w) for b, w in enumerate(win)))
dt_ok = all(np.allclose(c["dt"][b, :len(w)], sh["t_s"][w[-1]] - sh["t_s"][w]) for b, w in enumerate(win))
check("dt = t_last - t_j (>= 0, 0 at the snapshot itself)", dt_ok and bool((c["dt"] >= 0).all())
      and all(c["dt"][b, len(w) - 1] == 0 for b, w in enumerate(win)))
check("padding: dt 0, idx 0", bool((c["dt"][~c["mask"]] == 0).all() and (c["idx"][~c["mask"]] == 0).all()))
nonuni = np.diff(sh["t_s"][wi.window(int(np.argmax(lens)))])
check("dt spacing is non-uniform in the fake corpus", nonuni.min() < 0.1 < nonuni.max())

# ── gather ──
section("gather")
g = batching.gather(sh, win)
U = np.unique(np.concatenate(win))
check("each touched snapshot gathered once", g["player_num"].shape[0] == len(U))
remap_ok = all(np.array_equal(U[g["win_idx"][b, :len(w)]], w) for b, w in enumerate(win))
check("win_idx maps back to the original indices", remap_ok)
check("targets and their labels", g["target"].tolist() == targets
      and np.array_equal(g["y_win"], sh["y_win"][targets], equal_nan=True))
check("snapshot tensors gathered from the right rows", np.array_equal(g["ctx_num"], sh["ctx_num"][U]))

# ── sampler ──
section("tier sampler")
tiers = np.array(["pro"] * 200 + ["faceit_10"] * 800)
smp = batching.TierSampler(tiers, batch_size=256, pro_oversample=4, seed=1337)
target = 4 * 0.2 / (4 * 0.2 + 0.8)
check("target share = k*s / (k*s + 1 - s)", abs(smp.target_share - target) < 1e-12, f"{smp.target_share:.4f}")
batches = list(smp.epoch(0, n_batches=1000))
share = float(np.mean([np.mean(tiers[b] == "pro") for b in batches]))
check("observed pro share in 1,000 batches within +-2% of target", abs(share - target) <= 0.02,
      f"observed {share:.4f}, target {target:.4f}")
check("batch size", all(len(b) == 256 for b in batches))
again = list(smp.epoch(0, n_batches=5))
check("same (seed, epoch) -> same batches", all(np.array_equal(a, b) for a, b in zip(again, batches[:5])))
other = list(smp.epoch(1, n_batches=5))
check("different epoch -> different batches", not all(np.array_equal(a, b) for a, b in zip(other, batches[:5])))
cover = np.unique(np.concatenate(list(smp.epoch(2))))
check("an epoch draws from every stratum", np.isin(np.arange(200), cover).mean() > 0.9)
only_pro = batching.TierSampler(np.array(["pro"] * 10), 4, 4, 0)
no_pro   = batching.TierSampler(np.array(["faceit_10"] * 10), 4, 4, 0)
check("all-pro corpus -> share 1", only_pro.target_share == 1.0 and len(next(only_pro.epoch(0))) == 4)
check("no-pro corpus -> share 0", no_pro.target_share == 0.0 and len(next(no_pro.epoch(0))) == 4)
cand = np.arange(0, 1000, 2)
smp_c = batching.TierSampler(tiers, 64, 4, 0, candidates=cand)
check("candidates restrict the draw", set(np.concatenate(list(smp_c.epoch(0))).tolist()) <= set(cand.tolist()))

sys.exit(summary())
