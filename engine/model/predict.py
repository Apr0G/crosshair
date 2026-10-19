"""
Batched inference (P6.4.1, Amendment 1 contract):
`Predictor(checkpoint, device).predict(snapshots, histories) -> dict(win, hazard,
plant15, defuse15)`, caching history encodings per (match, round, tick, edit_hash).
Tests use `test/engine/_stubs.py::StubPredictor`, which has the same interface.

    pred = Predictor("data/engine/models/wp_v1", "cuda")
    out  = pred.predict([s_pre, s_post], [hist, hist + [s_pre]])
    out["win"]        # (N,) calibrated CT win probability
    out["hazard"]     # (N, 10) P(dies within 5 s), exactly 0 for dead / absent players
    out["plant15"], out["defuse15"]                     # (N,)
    h = pred.encode([s])                                # (N, d_pool) snapshot embedding (P9, P7.2.7)

`histories[i]` is the factual past of `snapshots[i]` (any order, any length): the
window is every history snapshot with `0 < t_s(present) - t_s(h) <= temporal_window_s`,
in time order, then the present snapshot. Counterfactual edits touch only the present
snapshot, so the history encodings are reused from the cache across calls; the cache
key carries `edit_hash` (a hash of the snapshot's features, labels excluded), so an
edited snapshot never collides with its factual version at the same tick.

Win probabilities are calibrated with the model directory's `calibration.json` (single
T, or per (map_version, phase) temperatures, P6.5.1) when it exists. The model runs in
fp32 (or bf16 autocast with `bf16=True`, GPU only), in eval mode. Heads the model lacks (ablation `--no-aux-heads`) come back as NaN.
"""
import argparse
import hashlib
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from engine.data import dataset
from engine.model import batching, calibrate, checkpoints

HEADS = ("win", "hazard", "plant15", "defuse15")


def edit_hash(s) -> str:
    """Content hash of a snapshot's features (players, utility, ctx, pair); labels and
    meta are excluded (meta's match/round/tick are already in the cache key). Field
    values are pickled in dataclass field order, so equal content gives equal hashes."""
    h = hashlib.sha1()
    h.update(pickle.dumps([tuple(p.__dict__.values()) for p in s.players], protocol=5))
    h.update(pickle.dumps([tuple(u.__dict__.values()) for u in s.utility], protocol=5))
    h.update(pickle.dumps(tuple(s.ctx.__dict__.values()), protocol=5))
    h.update(np.ascontiguousarray(s.pair, dtype=np.float32).tobytes())
    return h.hexdigest()


def cache_key(s) -> tuple:
    m = s.meta
    if m is None:
        return ("", s.ctx.round_num, None, edit_hash(s))
    return (m.match_key, m.round_num, m.tick, edit_hash(s))


def _t(s) -> float:
    if s.meta is None or s.meta.t_s is None:
        raise ValueError("Predictor needs snapshot.meta.t_s (seconds from freeze end)")
    return float(s.meta.t_s)


class Predictor:
    def __init__(self, checkpoint, device=None, *, cache: bool = True, batch_size: int = 512,
                 max_cache: int = 100_000, checkpoint_name: str | None = None, bf16: bool = False,
                 require_calibration: bool = False):
        """`require_calibration=True` (what `analyze` passes) refuses a model directory
        without calibration.json: a plain sigmoid is never served as a win probability
        in production (R6; MED-20, 2026-10-19). Tests and training tools may load an
        uncalibrated model explicitly."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bf16 autocast (GPU only): ~2x throughput, logits within ~1e-2 of fp32 (test_wp_model);
        # off by default because attribution differences W(s') - W(s) can be that small
        self.bf16   = bool(bf16) and self.device == "cuda" and torch.cuda.is_bf16_supported()
        (self.model, self.vocab, self.norm, self.calibration,
         self.checkpoint_path) = checkpoints.load_model_dir(checkpoint, self.device, checkpoint_name)
        if require_calibration and self.calibration is None:
            raise FileNotFoundError(str(Path(checkpoint) / "calibration.json"))
        self.model.eval()
        self.window_s   = self.model.window_s
        self.use_cache  = cache
        self.batch_size = int(batch_size)
        self.max_cache  = int(max_cache)
        self._cache: OrderedDict = OrderedDict()
        self.hits = self.misses = 0

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ── encoding with cache ──

    @torch.no_grad()
    def _encode_new(self, snaps: list) -> list[tuple]:
        out = []
        for a in range(0, len(snaps), self.batch_size):
            part = snaps[a:a + self.batch_size]
            arr  = dataset.tensorize(part, self.vocab)
            b    = batching.to_device({k: arr[k] for k in dataset.SNAPSHOT_KEYS}, self.device)
            with self._autocast():
                h, po = self.model.encode_full(b)
            h, po = h.float(), po.float()
            alive = self.model.alive(b)
            out += [(h[i], po[i], alive[i]) for i in range(len(part))]
        return out

    def _lookup(self, snaps: list, keys: list) -> dict:
        """key -> (h, players_out, alive) for every snapshot, encoding the missing ones."""
        found: dict = {}
        todo, todo_keys = [], []
        for s, k in zip(snaps, keys):
            if k in found:
                continue
            if self.use_cache and k in self._cache:
                self._cache.move_to_end(k)
                found[k] = self._cache[k]
                self.hits += 1
            else:
                found[k] = None
                todo.append(s)
                todo_keys.append(k)
        for k, enc in zip(todo_keys, self._encode_new(todo)):
            found[k] = enc
            self.misses += 1
            if self.use_cache:
                self._cache[k] = enc
                if len(self._cache) > self.max_cache:
                    self._cache.popitem(last=False)
        return found

    def window(self, snapshot, history: Sequence) -> list:
        t = _t(snapshot)
        past = [h for h in history if 0.0 < t - _t(h) <= self.window_s + 1e-6]
        return sorted(past, key=_t) + [snapshot]

    # ── public API ──

    @torch.no_grad()
    def predict(self, snapshots: Sequence, histories: Sequence) -> dict[str, np.ndarray]:
        if len(snapshots) != len(histories):
            raise ValueError(f"{len(snapshots)} snapshots but {len(histories)} histories")
        n = len(snapshots)
        out = {"win": np.full(n, np.nan), "hazard": np.full((n, dataset.N_PLAYERS), np.nan),
               "plant15": np.full(n, np.nan), "defuse15": np.full(n, np.nan)}
        if n == 0:
            return out
        windows = [self.window(s, h) for s, h in zip(snapshots, histories)]
        memo: dict[int, tuple] = {}       # id -> key, valid within this call only
        wkeys   = [[memo[id(x)] if id(x) in memo else memo.setdefault(id(x), cache_key(x)) for x in w]
                   for w in windows]
        flat    = [x for w in windows for x in w]
        enc     = self._lookup(flat, [k for ks in wkeys for k in ks])
        logits  = np.full(n, np.nan)
        order = {k: i for i, k in enumerate(enc)}                 # unique snapshot -> row
        H  = torch.stack([e[0] for e in enc.values()])
        PO = torch.stack([e[1] for e in enc.values()])
        AL = torch.stack([e[2] for e in enc.values()])
        times = [[_t(x) for x in w] for w in windows]
        for a in range(0, n, self.batch_size):
            ks, ts = wkeys[a:a + self.batch_size], times[a:a + self.batch_size]
            B, L = len(ks), max(len(k) for k in ks)
            idx  = np.zeros((B, L), np.int64)
            t    = np.zeros((B, L), np.float32)
            mask = np.zeros((B, L), bool)
            for b, (kk, tt) in enumerate(zip(ks, ts)):
                m = len(kk)
                idx[b, :m]  = [order[k] for k in kk]
                t[b, :m]    = tt
                t[b, m:]    = tt[-1]
                mask[b, :m] = True
            last  = torch.from_numpy(mask.sum(1) - 1).to(self.device)
            cur   = torch.from_numpy(idx[np.arange(B), mask.sum(1) - 1]).to(self.device)
            hw    = H[torch.from_numpy(idx).to(self.device)]
            with self._autocast():
                z = self.model.temporal_z(hw, torch.from_numpy(t).to(self.device),
                                          torch.from_numpy(mask).to(self.device), last)
                o = self.model.head(z, PO[cur], AL[cur])
            logits[a:a + B] = o["win"].double().cpu().numpy()
            for name in ("hazard", "plant15", "defuse15"):
                if name in o:
                    out[name][a:a + B] = torch.sigmoid(o[name].double()).cpu().numpy()
        mv = calibrate.map_versions_through_vocab([s.ctx.map_version for s in snapshots], self.vocab)   # LOW-29
        ph = calibrate.phase_of([s.ctx.bomb_planted for s in snapshots],
                                [s.ctx.time_remaining_s for s in snapshots],
                                [s.ctx.time_into_round_s for s in snapshots])
        out["win"] = calibrate.calibrated_probs(logits, self.calibration, mv, ph)
        return out

    @torch.no_grad()
    def encode(self, snapshots: Sequence) -> np.ndarray:
        """h_t of each snapshot (the set-encoder embedding, no history), (N, d_pool)."""
        if not len(snapshots):
            return np.zeros((0, self.model.arch["d_pool"]))
        keys = [cache_key(s) for s in snapshots]
        enc  = self._lookup(list(snapshots), keys)
        return torch.stack([enc[k][0] for k in keys]).double().cpu().numpy()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import polars as pl
    from engine.state.schema import frame_to_snapshots
    ap = argparse.ArgumentParser(description="Predict W for every snapshot of a states.parquet (P6.4).")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    snaps = frame_to_snapshots(pl.read_parquet(Path(a.states)))
    pred  = Predictor(a.model_dir, "cpu" if a.cpu else None)
    by_round: dict = {}
    hist = []
    for s in snaps:
        rk = (s.meta.match_key, s.meta.round_num)
        hist.append(list(by_round.get(rk, [])))
        by_round.setdefault(rk, []).append(s)
    out = pred.predict(snaps, hist)
    for s, w in zip(snaps, out["win"]):
        print(f"{s.meta.match_key} r{s.meta.round_num} t={s.meta.t_s:7.2f} W={w:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
