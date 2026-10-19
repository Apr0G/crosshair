"""
QNet (expected V of this action here) and BNet (expected V of the usual action here)
on frozen encoder embeddings (P9.2).

    from engine.value import q_model
    q, b, metrics = q_model.train_q(frame, h, vocab, cfg, split_fn, "cuda", out_dir=d)
    q, b, vocab, metrics = q_model.load_q(d, device="cpu")
    Q = q_model.predict_q(q, h, desc_ids, tier, side)      # float64[n]
    B = q_model.predict_b(b, h, tier, side)
    res = q_model.evaluate_q(q, b, test_frame, h_test, vocab)

Inputs (B.7): `h` = float32[n, d_h], the frozen `h_pre` from the adopted win model
(never computed here: the caller passes it, `h_matrix(frame)` reads the `h_pre` list
column); `desc_ids` = int64[n, len(descriptors.FIELDS)]; `tier` = "pro" / "faceit"
(anything else -> the UNK tier row); `side` = actor side "ct" (bit 1) / "t" (bit 0).
QNet input = LayerNorm(h) ⊕ one embedding per descriptor field ⊕ tier embedding ⊕
side bit → MLP hidden → 1. BNet is the same without the descriptor embeddings.
Target: actor-signed `V` (column `V`). Loss: Huber, δ = 0.05.

Frame columns read by train_q / evaluate_q: `V`, `actor_side`, `tier`, `type`, and
`desc_<field>` for every descriptor field (descriptors.frame_columns).

Checkpoint dir (`paths.MODELS_DIR / q_v<N>/` unless `out_dir`): `q.pt`, `b.pt`
({"args", "state"}), `vocab.json`, `metrics.json`, `manifest.json`.

Filled by: P9.2.1-P9.2.4.
"""
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from engine import manifest, paths
from engine.value.descriptors import FIELDS, PAD, DescriptorVocab, ids_from_frame

# ── constants ─────────────────────────────────────────────────────────────────

TIERS        = ("pro", "faceit")
TIER_UNK     = len(TIERS)
HUBER_DELTA  = 0.05
D_TIER       = 4
CALIB_BINS   = 20
CALIB_MIN_N  = 200
CALIB_TOL    = 0.01
SPEARMAN_TARGET = 0.3
Q_FILE, B_FILE, VOCAB_FILE, METRICS_FILE = "q.pt", "b.pt", "vocab.json", "metrics.json"


@dataclass(frozen=True)
class QTrainParams:
    lr:          float = 1e-3
    wd:          float = 1e-4
    batch_size:  int   = 1024
    max_epochs:  int   = 50
    patience:    int   = 5
    d_emb:       int   = 16
    hidden:      tuple[int, ...] = (512, 256)
    huber_delta: float = HUBER_DELTA


# ── input encoding ────────────────────────────────────────────────────────────

def tier_ids(tier) -> np.ndarray:
    """int64[n]: index in TIERS, TIER_UNK for anything else; ints pass through."""
    arr = tier.to_list() if isinstance(tier, pl.Series) else list(np.atleast_1d(np.asarray(tier, dtype=object)))
    out = np.empty(len(arr), dtype=np.int64)
    for i, t in enumerate(arr):
        if isinstance(t, (int, np.integer)) and not isinstance(t, bool):
            out[i] = int(t) if 0 <= int(t) <= TIER_UNK else TIER_UNK
        else:
            out[i] = TIERS.index(t) if t in TIERS else TIER_UNK
    return out


def side_bits(side) -> np.ndarray:
    """float32[n]: 1 for "ct", 0 for "t"; numeric 0/1 pass through. Anything else raises
    (an unknown actor side is an error, never silently T)."""
    arr = side.to_list() if isinstance(side, pl.Series) else list(np.atleast_1d(np.asarray(side, dtype=object)))
    out = np.empty(len(arr), dtype=np.float32)
    for i, s in enumerate(arr):
        if s == "ct" or (isinstance(s, (int, float, np.number)) and not isinstance(s, str) and s == 1):
            out[i] = 1.0
        elif s == "t" or (isinstance(s, (int, float, np.number)) and not isinstance(s, str) and s == 0):
            out[i] = 0.0
        else:
            raise ValueError(f"actor side {s!r} must be 'ct' or 't'")
    return out


def h_matrix(frame: pl.DataFrame, column: str = "h_pre") -> np.ndarray:
    """float32[n, d_h] from the frame's `h_pre` list column; raises on nulls or ragged rows."""
    s = frame[column]
    if s.null_count():
        raise ValueError(f"{column}: {s.null_count()} null embedding(s); fill h_pre before P9")
    rows = s.to_list()
    d = {len(r) for r in rows}
    if len(d) > 1:
        raise ValueError(f"{column}: ragged embeddings, lengths {sorted(d)}")
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), d.pop() if d else 0)


# ── networks ──────────────────────────────────────────────────────────────────

def _mlp(d_in: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [nn.Linear(d_in, width), nn.GELU()]
        d_in = width
    layers.append(nn.Linear(d_in, 1))
    return nn.Sequential(*layers)


def _sizes_list(vocab_sizes) -> list[int]:
    if isinstance(vocab_sizes, dict):
        return [int(vocab_sizes[f]) for f in FIELDS]
    sizes = [int(v) for v in vocab_sizes]
    if len(sizes) != len(FIELDS):
        raise ValueError(f"need {len(FIELDS)} vocab sizes, got {len(sizes)}")
    return sizes


class QNet(nn.Module):
    """Q̂(s, a) from (h_pre, descriptor ids, tier, side)."""

    def __init__(self, d_h: int, vocab_sizes, d_emb: int = 16, hidden: tuple[int, ...] = (512, 256),
                 d_tier: int = D_TIER):
        super().__init__()
        self.d_h, self.d_emb, self.hidden, self.d_tier = int(d_h), int(d_emb), tuple(hidden), int(d_tier)
        self.vocab_sizes = _sizes_list(vocab_sizes)
        self.norm = nn.LayerNorm(self.d_h)
        self.emb  = nn.ModuleList(nn.Embedding(n, self.d_emb, padding_idx=PAD) for n in self.vocab_sizes)
        self.tier = nn.Embedding(TIER_UNK + 1, self.d_tier)
        self.mlp  = _mlp(self.d_h + self.d_emb * len(self.vocab_sizes) + self.d_tier + 1, self.hidden)

    def args(self) -> dict:
        return {"d_h": self.d_h, "vocab_sizes": list(self.vocab_sizes), "d_emb": self.d_emb,
                "hidden": list(self.hidden), "d_tier": self.d_tier}

    def forward(self, h, desc_ids, tier, side) -> torch.Tensor:
        parts = [self.norm(h)] + [e(desc_ids[:, j]) for j, e in enumerate(self.emb)]
        parts += [self.tier(tier), side.unsqueeze(-1)]
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)


class BNet(nn.Module):
    """B̂(s) from (h_pre, tier, side): expected V of whatever is done here."""

    def __init__(self, d_h: int, hidden: tuple[int, ...] = (512, 256), d_tier: int = D_TIER):
        super().__init__()
        self.d_h, self.hidden, self.d_tier = int(d_h), tuple(hidden), int(d_tier)
        self.norm = nn.LayerNorm(self.d_h)
        self.tier = nn.Embedding(TIER_UNK + 1, self.d_tier)
        self.mlp  = _mlp(self.d_h + self.d_tier + 1, self.hidden)

    def args(self) -> dict:
        return {"d_h": self.d_h, "hidden": list(self.hidden), "d_tier": self.d_tier}

    def forward(self, h, tier, side) -> torch.Tensor:
        return self.mlp(torch.cat([self.norm(h), self.tier(tier), side.unsqueeze(-1)], dim=-1)).squeeze(-1)


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)


# ── tensors ───────────────────────────────────────────────────────────────────

def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _tensors(h, desc_ids, tier, side, device) -> tuple:
    th = torch.as_tensor(np.ascontiguousarray(h, dtype=np.float32), device=device)
    td = None if desc_ids is None else torch.as_tensor(np.asarray(desc_ids, dtype=np.int64), device=device)
    tt = torch.as_tensor(tier_ids(tier), device=device)
    ts = torch.as_tensor(side_bits(side), device=device)
    return th, td, tt, ts


@torch.no_grad()
def predict_q(model: QNet, h, desc_ids, tier, side, *, batch_size: int = 65536) -> np.ndarray:
    """Batched Q̂, float64[n]."""
    model.eval()
    th, td, tt, ts = _tensors(h, desc_ids, tier, side, _device_of(model))
    out = [model(th[i:i + batch_size], td[i:i + batch_size], tt[i:i + batch_size], ts[i:i + batch_size])
           for i in range(0, th.shape[0], batch_size)]
    return torch.cat(out).double().cpu().numpy() if out else np.zeros(0)


@torch.no_grad()
def predict_b(model: BNet, h, tier, side, *, batch_size: int = 65536) -> np.ndarray:
    """Batched B̂, float64[n]."""
    model.eval()
    th, _, tt, ts = _tensors(h, None, tier, side, _device_of(model))
    out = [model(th[i:i + batch_size], tt[i:i + batch_size], ts[i:i + batch_size])
           for i in range(0, th.shape[0], batch_size)]
    return torch.cat(out).double().cpu().numpy() if out else np.zeros(0)


# ── training ──────────────────────────────────────────────────────────────────

def next_version_dir(parent: Path, prefix: str) -> Path:
    parent = Path(parent)
    have = [int(p.name[len(prefix):]) for p in parent.glob(f"{prefix}*") if p.name[len(prefix):].isdigit()]
    return parent / f"{prefix}{max(have, default=0) + 1}"


def _seed_all(seed: int) -> None:
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def train_q(frame: pl.DataFrame, h: np.ndarray, vocab: DescriptorVocab, cfg,
            split_fn: Callable[[pl.DataFrame], object], device="cpu", *,
            params: QTrainParams | None = None, out_dir: Path | None = None,
            save: bool = True) -> tuple[QNet, BNet, dict]:
    """Train QNet and BNet on the rows `split_fn(frame) == "train"`, early-stopping each
    on its own Huber loss over the "val" rows (train loss when there is no val row).
    `split_fn(frame)` returns one split name per row. Seeded from `cfg.train.seed`.
    Saves to `out_dir` (default the next `paths.MODELS_DIR / q_v<N>`) when `save`."""
    if cfg.value.q_finetune_encoder:
        raise NotImplementedError("value.q_finetune_encoder: true needs the encoder; P9 trains on frozen h_pre")
    p = params or QTrainParams()
    h = np.asarray(h, dtype=np.float32)
    if h.ndim != 2 or h.shape[0] != frame.height:
        raise ValueError(f"h shape {h.shape} does not match {frame.height} frame rows")
    split = np.asarray(list(split_fn(frame)), dtype=object)
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    if tr.size == 0:
        raise ValueError("split_fn gave no 'train' rows")
    y = np.array(frame["V"].cast(pl.Float64).to_numpy(), dtype=np.float64)     # writable copy
    if not np.isfinite(y).all():
        raise ValueError("V has null / non-finite values")

    _seed_all(cfg.train.seed)
    device = torch.device(device)
    th, td, tt, ts = _tensors(h, ids_from_frame(frame, vocab), frame["tier"], frame["actor_side"], device)
    ty = torch.as_tensor(y, dtype=torch.float32, device=device)
    q = QNet(h.shape[1], vocab.sizes(), d_emb=p.d_emb, hidden=p.hidden).to(device)
    b = BNet(h.shape[1], hidden=p.hidden).to(device)
    opt_q = torch.optim.AdamW(q.parameters(), lr=p.lr, weight_decay=p.wd)
    opt_b = torch.optim.AdamW(b.parameters(), lr=p.lr, weight_decay=p.wd)
    gen = torch.Generator().manual_seed(cfg.train.seed)
    tr_t = torch.as_tensor(tr, device=device)
    va_t = torch.as_tensor(va if va.size else tr, device=device)

    best = {"q": (math.inf, None, -1), "b": (math.inf, None, -1)}
    stale = {"q": 0, "b": 0}
    history = []
    for epoch in range(p.max_epochs):
        q.train()
        b.train()
        perm = tr_t[torch.randperm(tr_t.numel(), generator=gen).to(device)]
        sums = {"q": 0.0, "b": 0.0}
        for i in range(0, perm.numel(), p.batch_size):
            ix = perm[i:i + p.batch_size]
            for name, model, opt in (("q", q, opt_q), ("b", b, opt_b)):
                if stale[name] > p.patience:
                    continue
                pred = model(th[ix], td[ix], tt[ix], ts[ix]) if name == "q" else model(th[ix], tt[ix], ts[ix])
                loss = huber(pred, ty[ix], p.huber_delta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sums[name] += loss.detach() * ix.numel()
        row = {"epoch": epoch}
        for name, model in (("q", q), ("b", b)):
            if stale[name] > p.patience:
                continue
            vl = val_loss_on(model, th, td, tt, ts, ty, va_t, name == "q", p.huber_delta)
            row[f"train_{name}"] = float(sums[name]) / tr.size
            row[f"val_{name}"] = vl
            if vl < best[name][0]:
                best[name] = (vl, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), epoch)
                stale[name] = 0
            else:
                stale[name] += 1
        history.append(row)
        if stale["q"] > p.patience and stale["b"] > p.patience:
            break

    q.load_state_dict(best["q"][1])
    b.load_state_dict(best["b"][1])
    q.eval()
    b.eval()
    metrics = {
        "n_train": int(tr.size), "n_val": int(va.size), "epochs": len(history),
        "best_epoch_q": best["q"][2], "best_epoch_b": best["b"][2],
        "val_loss_q": best["q"][0], "val_loss_b": best["b"][0],
        "train_loss_q": val_loss_on(q, th, td, tt, ts, ty, tr_t, True, p.huber_delta),
        "train_loss_b": val_loss_on(b, th, td, tt, ts, ty, tr_t, False, p.huber_delta),
        "seed": cfg.train.seed, "params": asdict(p), "d_h": int(h.shape[1]), "history": history,
    }
    if save:
        out = Path(out_dir) if out_dir is not None else next_version_dir(paths.MODELS_DIR, "q_v")
        save_q(out, q, b, vocab, metrics, cfg)
        metrics["dir"] = str(out)
    return q, b, metrics


@torch.no_grad()
def val_loss_on(model, th, td, tt, ts, ty, idx, is_q: bool, delta: float = HUBER_DELTA) -> float:
    """Mean Huber loss of `model` on the rows `idx` of already-built tensors."""
    model.eval()
    tot = 0.0
    for i in range(0, idx.numel(), 65536):
        ix = idx[i:i + 65536]
        pred = model(th[ix], td[ix], tt[ix], ts[ix]) if is_q else model(th[ix], tt[ix], ts[ix])
        tot += float(F.huber_loss(pred, ty[ix], delta=delta, reduction="sum"))
    return tot / max(1, idx.numel())


# ── checkpoints ───────────────────────────────────────────────────────────────

def save_q(out_dir: Path, q: QNet, b: BNet, vocab: DescriptorVocab, metrics: dict, cfg=None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"args": q.args(), "state": {k: v.detach().cpu() for k, v in q.state_dict().items()}}, out / Q_FILE)
    torch.save({"args": b.args(), "state": {k: v.detach().cpu() for k, v in b.state_dict().items()}}, out / B_FILE)
    vocab.save(out / VOCAB_FILE)
    (out / METRICS_FILE).write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
    manifest.write(out, "q_model", outputs=[Q_FILE, B_FILE, VOCAB_FILE, METRICS_FILE],
                   extra={"n_train": metrics.get("n_train"), "d_h": metrics.get("d_h")},
                   **({"cfg": cfg} if cfg is not None else {}))
    return out


def load_q(model_dir: Path, device="cpu") -> tuple[QNet, BNet, DescriptorVocab, dict]:
    d = Path(model_dir)
    cq = torch.load(d / Q_FILE, map_location="cpu", weights_only=True)
    cb = torch.load(d / B_FILE, map_location="cpu", weights_only=True)
    aq, ab = dict(cq["args"]), dict(cb["args"])
    q = QNet(aq["d_h"], aq["vocab_sizes"], d_emb=aq["d_emb"], hidden=tuple(aq["hidden"]), d_tier=aq["d_tier"])
    b = BNet(ab["d_h"], hidden=tuple(ab["hidden"]), d_tier=ab["d_tier"])
    q.load_state_dict(cq["state"])
    b.load_state_dict(cb["state"])
    vocab = DescriptorVocab.load(d / VOCAB_FILE)
    mp = d / METRICS_FILE
    metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    return q.to(device).eval(), b.to(device).eval(), vocab, metrics


# ── evaluation (P9.2.3, P9.2.4) ───────────────────────────────────────────────

def spearman(a, b) -> float:
    """Spearman rank correlation (average ranks for ties); NaN when undefined."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    ra = pl.Series(a).rank("average").to_numpy().astype(np.float64)
    rb = pl.Series(b).rank("average").to_numpy().astype(np.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


CALIB_SCHEMA = {"bin": pl.Int64, "n": pl.Int64, "pred_lo": pl.Float64, "pred_hi": pl.Float64,
                "mean_pred": pl.Float64, "mean_V": pl.Float64, "abs_err": pl.Float64, "ok": pl.Boolean}


def calibration_table(pred, v, n_bins: int = CALIB_BINS, min_bin_n: int = CALIB_MIN_N,
                      tol: float = CALIB_TOL) -> pl.DataFrame:
    """Equal-count bins of `pred` (sorted, stable): per bin n, range, mean prediction,
    mean V, |difference|, and `ok` (|diff| <= tol; null when n < min_bin_n)."""
    pred, v = np.asarray(pred, dtype=np.float64), np.asarray(v, dtype=np.float64)
    order = np.argsort(pred, kind="stable")
    rows = []
    for i, ix in enumerate(np.array_split(order, min(n_bins, max(1, pred.size)))):
        if ix.size == 0:
            continue
        mp, mv = float(pred[ix].mean()), float(v[ix].mean())
        err = abs(mp - mv)
        rows.append({"bin": i, "n": int(ix.size), "pred_lo": float(pred[ix].min()),
                     "pred_hi": float(pred[ix].max()), "mean_pred": mp, "mean_V": mv, "abs_err": err,
                     "ok": (err <= tol) if ix.size >= min_bin_n else None})
    return pl.DataFrame(rows, schema=CALIB_SCHEMA)


def _calib_ok(table: pl.DataFrame) -> bool | None:
    vals = [x for x in table["ok"].to_list() if x is not None]
    return all(vals) if vals else None


SANITY_CHECKS = (
    # (name, type, side or None, sign)
    ("kill_mean_Q_positive",          "kill",            None, +1),
    ("team_kill_mean_Q_negative",     "team_kill",       None, -1),
    ("defuse_complete_ct_Q_positive", "defuse_complete", "ct", +1),
    ("plant_complete_t_Q_positive",   "plant_complete",  "t",  +1),
)


def evaluate_q(q: QNet, b: BNet, frame: pl.DataFrame, h: np.ndarray, vocab: DescriptorVocab, *,
               n_bins: int = CALIB_BINS, min_bin_n: int = CALIB_MIN_N, tol: float = CALIB_TOL) -> dict:
    """P9.2.3 / P9.2.4 on a (test, pro) frame: calibration tables for Q̂ and B̂ against V,
    Spearman, per-type means, the four sign checks, and `passed` (the calibration target:
    every bin with n >= min_bin_n within ±tol for Q̂ and B̂; None when no bin qualifies;
    the sign checks must not be False)."""
    ids = ids_from_frame(frame, vocab)
    qh = predict_q(q, h, ids, frame["tier"], frame["actor_side"])
    bh = predict_b(b, h, frame["tier"], frame["actor_side"])
    v = frame["V"].cast(pl.Float64).to_numpy()
    q_tab, b_tab = calibration_table(qh, v, n_bins, min_bin_n, tol), calibration_table(bh, v, n_bins, min_bin_n, tol)
    per_type = (frame.select("type", "actor_side").with_columns(Q=pl.Series(qh), B=pl.Series(bh), V=pl.Series(v))
                .group_by("type", "actor_side").agg(n=pl.len(), mean_Q=pl.col("Q").mean(),
                                                    mean_B=pl.col("B").mean(), mean_V=pl.col("V").mean())
                .sort("type", "actor_side"))
    sanity = {}
    for name, t, side, sign in SANITY_CHECKS:
        m = (frame["type"] == t).to_numpy()
        if side is not None:
            m &= (frame["actor_side"] == side).to_numpy()
        val = float(qh[m].mean()) if m.any() else None
        sanity[name] = {"n": int(m.sum()), "mean_Q": val, "ok": None if val is None else bool(sign * val > 0)}
    cq, cb = _calib_ok(q_tab), _calib_ok(b_tab)
    calib = None if cq is None and cb is None else bool((cq is not False) and (cb is not False))
    sanity_ok = all(s["ok"] is not False for s in sanity.values())
    return {"n": frame.height, "q_bins": q_tab, "b_bins": b_tab, "calibration_q": cq, "calibration_b": cb,
            "spearman_q": spearman(qh, v), "spearman_b": spearman(bh, v), "spearman_target": SPEARMAN_TARGET,
            "per_type": per_type, "sanity": sanity, "sanity_ok": sanity_ok,
            "passed": None if calib is None else bool(calib and sanity_ok)}


def eval_markdown(res: dict, title: str = "Q / B evaluation") -> str:
    """`eval/q_v<N>.md` body for an `evaluate_q` result."""
    def table(df: pl.DataFrame) -> list[str]:
        cols = df.columns
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in df.iter_rows():
            out.append("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |")
        return out
    lines = [f"# {title}", "", f"rows: {res['n']}  passed: {res['passed']}", "",
             f"Spearman Q̂ vs V: {res['spearman_q']:.3f}  B̂ vs V: {res['spearman_b']:.3f} "
             f"(target ≥ {res['spearman_target']}, informational)", "", "## Q̂ calibration", ""]
    lines += table(res["q_bins"]) + ["", "## B̂ calibration", ""] + table(res["b_bins"])
    lines += ["", "## Per-type means", ""] + table(res["per_type"]) + ["", "## Sign checks", ""]
    lines += [f"- {k}: n={s['n']} mean_Q={s['mean_Q']} ok={s['ok']}" for k, s in res["sanity"].items()]
    return "\n".join(lines) + "\n"
