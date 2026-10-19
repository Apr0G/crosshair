"""
Temperature scaling on val pro snapshots, reliability/ECE, terminal pin check (P6.5),
and the martingale drift check (P6.6.2). Formulas: Appendix B.9.

    from engine.model import calibrate
    cal = calibrate.fit_calibration(logits, y, map_versions, phases, n_bins=20)
    p   = calibrate.calibrated_probs(logits, cal, map_versions, phases)
    m   = calibrate.metrics(p, y)                    # logloss, brier, auc, ece, reliability
    calibrate.martingale_check(p, round_key, t_s, (1, 5, 10))
    calibrate.terminal_pin(p, y, is_terminal, round_key)

Everything here is numpy on arrays of logits / probabilities, so it is shared by the
NN (train.evaluate), the LightGBM baseline and the CLI. `main()` fits calibration for
a trained model directory on val pro and reports on test pro.

Phases (P6.1.2 lists 0-30 s, 30-75 s, post-plant, last 15 s; the pre-plant gap from 75 s
to the last 15 s is its own phase here so every snapshot has one):
`post_plant` if the bomb is planted, else `last_15` if time_remaining_s <= 15, else
`0_30` / `30_75` / `75_plus` by time_into_round_s.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PHASES       = ("0_30", "30_75", "75_plus", "last_15", "post_plant")
EPS          = 1e-7
MIN_IMPROVE  = 0.002          # P6.5.1: keep per-group temperatures only if val ECE improves this much


# ── phases ────────────────────────────────────────────────────────────────────

def phase_of(bomb_planted, time_remaining_s, time_into_round_s) -> np.ndarray:
    bp = np.asarray(bomb_planted, dtype=np.float64)
    tr = np.asarray(time_remaining_s, dtype=np.float64)
    ti = np.asarray(time_into_round_s, dtype=np.float64)
    out = np.where(ti < 30, "0_30", np.where(ti < 75, "30_75", "75_plus")).astype("<U10")
    out = np.where(tr <= 15, "last_15", out)
    return np.where(bp > 0.5, "post_plant", out)


def phases_from_arrays(arrays: dict) -> np.ndarray:
    """Phase of every snapshot from world-unit dataset arrays."""
    from engine.data.dataset import CTX_LAYOUT
    c = arrays["ctx_num"]
    return phase_of(c[:, CTX_LAYOUT.col("bomb_planted")], c[:, CTX_LAYOUT.col("time_remaining_s")],
                    c[:, CTX_LAYOUT.col("time_into_round_s")])


def map_versions_from_arrays(arrays: dict, vocab) -> np.ndarray:
    from engine.data.dataset import FEATURE_LAYOUT
    inv = np.array(vocab.inverse("map_version"))
    return inv[arrays["ctx_cat"][:, FEATURE_LAYOUT["ctx_cat"].index("map_version")]]


# ── metrics ───────────────────────────────────────────────────────────────────

def sigmoid(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    y = np.asarray(y, np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, np.float64) - np.asarray(y, np.float64)) ** 2))


def auc(p, y) -> float | None:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def reliability(p, y, n_bins: int = 20) -> list[dict]:
    """Equal-width bins on [0, 1]; empty bins are listed with n = 0 and None means."""
    p = np.asarray(p, np.float64)
    y = np.asarray(y, np.float64)
    b = np.clip((p * n_bins).astype(np.int64), 0, n_bins - 1)
    out = []
    for k in range(n_bins):
        m = b == k
        n = int(m.sum())
        out.append({"lo": k / n_bins, "hi": (k + 1) / n_bins, "n": n,
                    "mean_pred": float(p[m].mean()) if n else None,
                    "observed": float(y[m].mean()) if n else None})
    return out


def ece(p, y, n_bins: int = 20) -> float:
    """ECE = sum_b (n_b / N) |mean_pred_b - obs_b| (Appendix B.9)."""
    N = len(p)
    if N == 0:
        return float("nan")
    return float(sum(r["n"] / N * abs(r["mean_pred"] - r["observed"])
                     for r in reliability(p, y, n_bins) if r["n"]))


def metrics(p, y, n_bins: int = 20) -> dict:
    p = np.asarray(p, np.float64)
    y = np.asarray(y, np.float64)
    ok = ~np.isnan(y) & ~np.isnan(p)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {"n": 0, "logloss": None, "brier": None, "auc": None, "ece": None, "reliability": []}
    return {"n": int(len(p)), "logloss": logloss(p, y), "brier": brier(p, y), "auc": auc(p, y),
            "ece": ece(p, y, n_bins), "reliability": reliability(p, y, n_bins)}


def bin_violations(rel: list[dict], bin_tol: float, min_bin_n: int) -> list[dict]:
    """P6.5.2: bins with n >= min_bin_n and |mean_pred - observed| > bin_tol."""
    return [r for r in rel if r["n"] >= min_bin_n and abs(r["mean_pred"] - r["observed"]) > bin_tol]


# ── temperature ───────────────────────────────────────────────────────────────

def nll_at(logits, y, T: float) -> float:
    return logloss(sigmoid(np.asarray(logits, np.float64) / T), y)


def fit_temperature(logits, y, lo: float = -4.0, hi: float = 4.0, iters: int = 80) -> float:
    """T = argmin NLL(sigmoid(logit / T)) by golden-section search over log T
    (NLL is unimodal in log T for a single temperature)."""
    logits = np.asarray(logits, np.float64)
    y = np.asarray(y, np.float64)
    if len(logits) == 0 or len(np.unique(y)) < 2:
        return 1.0
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll_at(logits, y, math.exp(c)), nll_at(logits, y, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll_at(logits, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll_at(logits, y, math.exp(d))
    return float(math.exp((a + b) / 2))


def group_keys(map_versions, phases) -> np.ndarray:
    return np.char.add(np.char.add(np.asarray(map_versions, dtype=str), "|"), np.asarray(phases, dtype=str))


def temperatures_for(cal: dict, map_versions, phases, n: int) -> np.ndarray:
    T = np.full(n, float(cal["T"]))
    groups = cal.get("groups") if cal.get("mode") == "group" else None
    if groups and n:
        keys = group_keys(map_versions, phases)
        for k, t in groups.items():
            T[keys == k] = t
    return T


def calibrated_probs(logits, cal: dict | None, map_versions=None, phases=None) -> np.ndarray:
    logits = np.asarray(logits, np.float64)
    if cal is None:
        return sigmoid(logits)
    T = temperatures_for(cal, map_versions, phases, len(logits)) if cal.get("mode") == "group" \
        else float(cal["T"])
    return sigmoid(logits / T)


def fit_calibration(logits, y, map_versions, phases, *, n_bins: int = 20,
                    min_group_n: int = 200, min_improve: float = MIN_IMPROVE) -> dict:
    """Single T on val; per-(map_version, phase) temperatures (groups with fewer than
    `min_group_n` snapshots keep the single T) kept only if val ECE improves by at least
    `min_improve` (P6.5.1)."""
    logits = np.asarray(logits, np.float64)
    y = np.asarray(y, np.float64)
    ok = ~np.isnan(y)
    logits, y = logits[ok], y[ok]
    mv, ph = np.asarray(map_versions)[ok], np.asarray(phases)[ok]
    T = fit_temperature(logits, y)
    ece_single = ece(sigmoid(logits / T), y, n_bins)
    keys = group_keys(mv, ph)
    groups: dict[str, float] = {}
    for k in sorted(set(keys.tolist())):
        m = keys == k
        if m.sum() >= min_group_n:
            groups[k] = fit_temperature(logits[m], y[m])
    trial = {"T": T, "mode": "group", "groups": groups}
    ece_group = ece(calibrated_probs(logits, trial, mv, ph), y, n_bins) if groups else ece_single
    keep = bool(groups) and (ece_single - ece_group) >= min_improve
    return {"T": T, "mode": "group" if keep else "single", "groups": groups if keep else {},
            "candidate_groups": groups, "ece_val_single": ece_single, "ece_val_group": ece_group,
            "n_val": int(len(y)), "min_improve": min_improve, "n_bins": n_bins}


def save_calibration(run_dir: Path, cal: dict) -> Path:
    path = Path(run_dir) / "calibration.json"
    path.write_text(json.dumps(cal, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ── terminal pin (P6.5.3) and martingale (P6.6.2) ─────────────────────────────

def terminal_pin(p, y, is_terminal, round_key, hi: float = 0.99, lo: float = 0.01,
                 min_rate: float = 0.99) -> dict:
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    term = np.asarray(is_terminal).astype(bool) & ~np.isnan(y)
    if not term.any():
        return {"n_rounds": 0, "rate": None, "ok": None}
    good = np.where(y[term] > 0.5, p[term] >= hi, p[term] <= lo)
    rate = float(good.mean())
    return {"n_rounds": int(len(np.unique(np.asarray(round_key)[term]))), "rate": rate,
            "ok": rate >= min_rate}


def martingale_check(p, round_key, t_s, horizons=(1, 5, 10), n_bins: int = 20,
                     min_n: int = 500, tol: float = 0.01) -> dict:
    """Mean of W(t+k) - W(t) per W(t) bin and horizon k; W(t+k) is the first snapshot of
    the same round at or after t + k (pairs without one are skipped). Bins with
    n >= min_n must have |mean drift| <= tol."""
    p  = np.asarray(p, np.float64)
    rk = np.asarray(round_key)
    ts = np.asarray(t_s, np.float64)
    order = np.lexsort((ts, rk))
    rks, tss, ps = rk[order], ts[order], p[order]
    bounds = np.flatnonzero(np.r_[True, rks[1:] != rks[:-1], True]) if len(p) else np.array([0])
    rows, bad = [], []
    for k in horizons:
        cur, fut = [], []
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = tss[a:b]
            j = np.searchsorted(seg, seg + k - 1e-9, side="left")
            ok = j < (b - a)
            cur.append(ps[a:b][ok])
            fut.append(ps[a:b][j[ok]])
        cur = np.concatenate(cur) if cur else np.zeros(0)
        fut = np.concatenate(fut) if fut else np.zeros(0)
        bins = np.clip((cur * n_bins).astype(np.int64), 0, n_bins - 1)
        for q in range(n_bins):
            m = bins == q
            n = int(m.sum())
            drift = float((fut[m] - cur[m]).mean()) if n else None
            row = {"k": k, "lo": q / n_bins, "hi": (q + 1) / n_bins, "n": n, "mean_drift": drift,
                   "judged": n >= min_n}
            rows.append(row)
            if n >= min_n and abs(drift) > tol:
                bad.append(row)
    return {"rows": rows, "violations": bad, "ok": not bad, "tol": tol, "min_n": min_n}


# ── report ────────────────────────────────────────────────────────────────────

def reliability_md(rel: list[dict]) -> str:
    lines = ["| bin | n | mean pred | observed | gap |", "|---|---:|---:|---:|---:|"]
    for r in rel:
        if r["n"]:
            lines.append(f"| {r['lo']:.2f}-{r['hi']:.2f} | {r['n']} | {r['mean_pred']:.4f} | "
                         f"{r['observed']:.4f} | {r['mean_pred'] - r['observed']:+.4f} |")
    return "\n".join(lines)


def metrics_md(title: str, m: dict) -> str:
    def f(v):
        return "n/a" if v is None else f"{v:.4f}"
    return (f"### {title}\n\nn={m['n']}  logloss={f(m['logloss'])}  brier={f(m['brier'])}  "
            f"auc={f(m['auc'])}  ece={f(m['ece'])}\n\n{reliability_md(m['reliability'])}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    from datetime import date

    from engine import config, paths
    from engine.data import dataset
    from engine.model import checkpoints, train

    ap = argparse.ArgumentParser(description="Fit calibration on val pro, report on test pro (P6.5).")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--val", default="val", help="val split name or shard directory")
    ap.add_argument("--test", default="test", help="test split name or shard directory")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    help="exit 1 when ECE / bin / terminal-pin / martingale targets are missed")
    a = ap.parse_args(argv)

    cfg    = config.load()
    device = train.pick_device(a.cpu)
    try:
        model, vocab, _, _, ckpt = checkpoints.load_model_dir(a.model_dir, device)
        for split in (a.val, a.test):                   # the model's vocab must be the shards' vocab (MED-18)
            dataset.check_artifacts(split, Path(a.model_dir) / "vocab.json", Path(a.model_dir) / "norm.json")
        val    = train.only_tier(dataset.read_shards(a.val), "pro")
        test   = train.only_tier(dataset.read_shards(a.test), "pro")
        for what, arr in (("--val", val), ("--test", test)):
            if dataset.n_labelled(arr) == 0:            # before anything is fitted or written (MED-19)
                raise ValueError(f"{what} has no labelled pro rows; nothing to calibrate on")
    except (FileNotFoundError, ValueError) as exc:
        print(f"calibrate: {exc}", file=sys.stderr)
        return 2
    lv     = train.predict_logits(model, val, device)["win"]
    cal    = fit_calibration(lv, val["y_win"], map_versions_from_arrays(val, vocab), phases_from_arrays(val),
                             n_bins=cfg.calib.n_bins, min_group_n=cfg.calib.min_bin_n)
    cal["checkpoint"], cal["state_hash"] = ckpt.name, checkpoints.state_hash(model.state_dict())   # MED-20
    save_calibration(a.model_dir, cal)
    res = train.evaluate(model, test, device, cfg=cfg, calibration=cal, vocab=vocab)
    name = Path(a.model_dir).name
    out = paths.EVAL_DIR / f"{name}_calibration_{date.today().isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(train.eval_markdown(f"{name} calibration on test pro ({ckpt.name})", res, cal),
                   encoding="utf-8")
    print(f"T={cal['T']:.4f} mode={cal['mode']}  test ece={res['ece']}  -> {out}")
    if a.do_assert:
        bad = train.target_failures(res, cfg)
        for line in bad:
            print(f"target missed: {line}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
