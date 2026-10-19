"""
Temporal windows, collation and tier-aware sampling (P5.3.1 "windows are formed at load
time", P6.2.3, P6.3.2). numpy only; `to_device` is the one place torch appears.

    from engine.model import batching
    wi  = batching.WindowIndex(arrs["round_key"], arrs["t_s"], window_s=10)
    win = [wi.window(i) for i in targets]            # original indices, time order, ending at i
    b   = batching.collate(win, arrs["t_s"])         # idx/mask/t/dt/last, left-aligned, padded
    mb  = batching.gather(arrs, win)                 # unique snapshots + remapped windows
    smp = batching.TierSampler(arrs["tier"], batch_size=256, pro_oversample=4, seed=1337)
    for targets in smp.epoch(0): ...

A window for snapshot i holds every snapshot of the same round with
`t_i - window_s <= t_j <= t_i`, in time order, i last. Variable length (boundary
samples make the count vary); `collate` pads with a mask and gives `dt = t_i - t_j`.
"""
import math
from typing import Iterator, Sequence

import numpy as np

from engine.data.dataset import LABEL_KEYS, SNAPSHOT_KEYS

EPS_S = 1e-6


# ── windows ───────────────────────────────────────────────────────────────────

class WindowIndex:
    """Per-snapshot temporal windows over arrays that may be in any order."""

    def __init__(self, round_key: np.ndarray, t_s: np.ndarray, window_s: float):
        rk = np.asarray(round_key)
        ts = np.asarray(t_s, dtype=np.float64)
        if np.isnan(ts).any():
            raise ValueError("t_s has NaN; every snapshot needs its time from freeze end")
        self.window_s = float(window_s)
        self.order    = np.lexsort((ts, rk))                  # by round, then time
        n             = len(self.order)
        self.pos      = np.empty(n, np.int64)
        self.pos[self.order] = np.arange(n)
        rks, tss      = rk[self.order], ts[self.order]
        self.start    = np.empty(n, np.int64)                 # sorted position where i's window starts
        bounds = np.flatnonzero(np.r_[True, rks[1:] != rks[:-1], True]) if n else np.array([0])
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = tss[a:b]
            self.start[a:b] = a + np.searchsorted(seg, seg - self.window_s - EPS_S, side="left")
        self.n_rounds = max(0, len(bounds) - 1)

    def __len__(self) -> int:
        return len(self.order)

    def window(self, i: int) -> np.ndarray:
        p = self.pos[i]
        return self.order[self.start[p]:p + 1]

    def lengths(self) -> np.ndarray:
        """Window length of every snapshot, in original order."""
        return (self.pos - self.start[self.pos] + 1).astype(np.int64)


def collate(windows: Sequence[np.ndarray], t_s: np.ndarray) -> dict[str, np.ndarray]:
    """Pad windows (left-aligned) to the longest one.

    idx [B,L] original indices (0 where padded), mask [B,L], t [B,L] seconds,
    dt [B,L] = t_last - t_j (0 where padded), last [B] position of the window's own
    snapshot."""
    B = len(windows)
    L = max((len(w) for w in windows), default=1)
    idx  = np.zeros((B, L), np.int64)
    mask = np.zeros((B, L), bool)
    t    = np.zeros((B, L), np.float64)
    last = np.zeros(B, np.int64)
    for b, w in enumerate(windows):
        n = len(w)
        idx[b, :n]  = w
        mask[b, :n] = True
        t[b, :n]    = t_s[w]
        t[b, n:]    = t_s[w[-1]] if n else 0.0
        last[b]     = n - 1
    dt = np.where(mask, t[np.arange(B), last][:, None] - t, 0.0)
    return {"idx": idx, "mask": mask, "t": t, "dt": dt, "last": last}


def gather(arrays: dict[str, np.ndarray], windows: Sequence[np.ndarray],
           with_labels: bool = True) -> dict[str, np.ndarray]:
    """A model batch: the unique snapshots the windows touch (each encoded once) plus
    the windows remapped onto them (`win_idx`, `win_mask`, `win_t`, `win_last`) and
    the targets' labels (`y_*`, `is_terminal`, in window order) and indices (`target`)."""
    flat = np.concatenate([np.asarray(w, np.int64) for w in windows]) if windows else np.zeros(0, np.int64)
    uniq = np.unique(flat)
    col  = collate(windows, arrays["t_s"])
    out  = {k: arrays[k][uniq] for k in SNAPSHOT_KEYS}
    out["win_idx"]  = np.where(col["mask"], np.searchsorted(uniq, col["idx"]), 0)
    out["win_mask"] = col["mask"]
    out["win_t"]    = col["t"].astype(np.float32)
    out["win_last"] = col["last"]
    target = np.array([w[-1] for w in windows], np.int64)
    out["target"] = target
    if with_labels:
        for k in LABEL_KEYS:
            out[k] = arrays[k][target]
    return out


def to_device(batch: dict[str, np.ndarray], device) -> dict:
    """numpy batch -> torch tensors on `device` (bool/int/float kept; strings dropped)."""
    import torch
    out = {}
    for k, v in batch.items():
        if v.dtype.kind in "USO":
            continue
        t = torch.from_numpy(np.ascontiguousarray(v))
        if t.dtype == torch.float64:
            t = t.float()
        out[k] = t.to(device, non_blocking=True)
    return out


# ── tier-aware sampling (P6.3.2) ──────────────────────────────────────────────

def target_pro_share(natural_share: float, pro_oversample: float) -> float:
    """Pro snapshots weighted `pro_oversample` x relative to their natural share."""
    s, k = natural_share, pro_oversample
    if s <= 0.0:
        return 0.0
    if s >= 1.0:
        return 1.0
    return k * s / (k * s + (1.0 - s))


class TierSampler:
    """Batches of target indices with the pro share at `target_share` in expectation.

    Each batch draws n_pro ~ Binomial(batch, share) from a shuffled cycle of pro
    indices and the rest from a shuffled cycle of the others; an epoch is
    ceil(n / batch) batches. Seeded by (seed, epoch, stage), so a resumed run replays
    exactly the batches it would have seen."""

    def __init__(self, tiers: np.ndarray, batch_size: int, pro_oversample: float, seed: int,
                 candidates: np.ndarray | None = None, pro_tier: str = "pro"):
        idx = np.arange(len(tiers)) if candidates is None else np.asarray(candidates, np.int64)
        if len(idx) == 0:
            raise ValueError("TierSampler: no candidate snapshots")
        is_pro = np.asarray(tiers)[idx] == pro_tier
        self.pro, self.other = idx[is_pro], idx[~is_pro]
        self.batch_size      = int(batch_size)
        self.seed            = int(seed)
        self.natural_share   = len(self.pro) / len(idx)
        self.target_share    = target_pro_share(self.natural_share, pro_oversample)
        self.n               = len(idx)

    def n_batches(self) -> int:
        return math.ceil(self.n / self.batch_size)

    def epoch(self, epoch: int, n_batches: int | None = None, stage: int = 0) -> Iterator[np.ndarray]:
        rng = np.random.default_rng([self.seed, int(epoch), int(stage)])
        pools = {"pro": [self.pro, rng.permutation(self.pro), 0],
                 "other": [self.other, rng.permutation(self.other), 0]}

        def draw(name: str, k: int) -> np.ndarray:
            src, perm, at = pools[name]
            out = []
            while k > 0:
                if at >= len(perm):
                    perm, at = rng.permutation(src), 0
                take = min(k, len(perm) - at)
                out.append(perm[at:at + take])
                at += take
                k  -= take
            pools[name][1], pools[name][2] = perm, at
            return np.concatenate(out) if out else np.zeros(0, np.int64)

        for _ in range(n_batches if n_batches is not None else self.n_batches()):
            n_pro = int(rng.binomial(self.batch_size, self.target_share))
            batch = np.concatenate([draw("pro", n_pro), draw("other", self.batch_size - n_pro)])
            yield rng.permutation(batch)
