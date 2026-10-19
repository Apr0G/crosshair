"""
Empirical-Bayes shrinkage of per-player execution / decision aggregates (P9.4.2, B.8).

    from engine.value import shrinkage
    fit = shrinkage.fit_k(train_scored)                  # {"X": {cls: {...}}, "D": {...}}
    shrinkage.save(fit, model_dir / "shrinkage.json"); fit = shrinkage.load(path)
    agg = shrinkage.player_aggregates(scored, fit)       # X_tilde, D_tilde, intervals, n

Per type class c and score column (X, D), from players with at least 2 actions in c:

    σ²_noise = pooled within-player variance  Σ_p Σ_i (x − x̄_p)² / Σ_p (n_p − 1)
    n̄        = harmonic mean of n_p (makes E[Var_p(x̄_p)] = σ²_skill + σ²_noise / n̄
               exact for unequal n; equals the plain mean when all n_p are equal)
    σ²_skill = max(ε, Var_p(x̄_p) − σ²_noise / n̄)        (Var with ddof = 1)
    k        = σ²_noise / σ²_skill  (> 0)

    x̃ = m_c + (x̄ − m_c) · n / (n + k)    90% half-width = 1.645 · sqrt(σ²_noise / (n + k))

Shrinkage is toward the class mean m_c (Σ n_p x̄_p / Σ n_p over the fitted players),
not toward 0: D = Q − B has a non-zero class mean, so pulling it to 0 would reward
volume (B.8 amended; Appendix H 8, decided 2026-10-19).

`fit_from_models(frame, h, q, b, vocab)` computes X and D for a valued frame from the
trained Q / B nets and fits; `train-q` calls it on the train rows and writes
`shrinkage.json` next to the Q model (MED-34).

Frame columns read: `actor_id_hash` (player), `type`, `X`, `D` (null rows dropped per
column).

Filled by: P9.4.2.
"""
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

# ── constants ─────────────────────────────────────────────────────────────────

Z90       = 1.645
EPS       = 1e-8
VALUES    = ("X", "D")
PLAYER    = "actor_id_hash"
FILE_NAME = "shrinkage.json"

TYPE_CLASS: dict[str, str] = {
    "kill": "kill", "team_kill": "team_kill", "damage": "damage",
    "flash": "utility", "he": "utility", "smoke": "utility", "molotov": "utility", "decoy": "utility",
    "smoke_pop": "utility", "smoke_fade": "utility", "fire_start": "utility", "fire_end": "utility",
    "decoy_start": "utility", "decoy_end": "utility",
    "peek": "movement", "rotation": "movement", "hold": "movement", "save": "movement",
    "positioning": "movement",
    "plant_start": "bomb", "plant_complete": "bomb", "defuse_start": "bomb", "defuse_complete": "bomb",
    "defuse_abort": "bomb", "bomb_drop": "bomb", "bomb_pickup": "bomb",
    "buy": "economy", "team_buy": "economy", "drop": "economy",
    "trade_setup": "support",
}
OTHER = "other"


def type_class(t: str | None) -> str:
    return TYPE_CLASS.get(t, OTHER)


def with_class(frame: pl.DataFrame) -> pl.DataFrame:
    """Add `type_class` from `type` when absent."""
    if "type_class" in frame.columns:
        return frame
    return frame.with_columns(pl.col("type").replace_strict(TYPE_CLASS, default=OTHER,
                                                            return_dtype=pl.String).alias("type_class"))


# ── formulas ──────────────────────────────────────────────────────────────────

def shrink(mean: float, n, k: float, centre: float = 0.0):
    """centre + (x̄ − centre) · n / (n + k) (works on scalars and numpy arrays)."""
    return centre + (mean - centre) * n / (n + k)


def factor(n, k: float):
    return n / (n + k)


def interval(n, k: float, sigma2_noise: float):
    """90% half-width 1.645 · sqrt(σ²_noise / (n + k))."""
    return Z90 * np.sqrt(sigma2_noise / (np.asarray(n, dtype=np.float64) + k))


def _fit_one(players: pl.DataFrame, eps: float) -> dict | None:
    """`players`: one row per player with n, mean, ss (within sum of squares)."""
    p = players.filter(pl.col("n") >= 2)
    if p.height < 2:
        return None
    n = p["n"].to_numpy().astype(np.float64)
    dof = float((n - 1).sum())
    s2_noise = float(p["ss"].sum()) / dof if dof > 0 else math.nan
    var_means = float(np.var(p["mean"].to_numpy(), ddof=1))
    n_bar = float(n.size / (1.0 / n).sum())
    s2_skill = max(eps, var_means - s2_noise / n_bar)
    k = s2_noise / s2_skill
    if not math.isfinite(k) or k <= 0:
        k = eps                          # no noise at all: trust the raw mean
    mean_all = float((p["mean"].to_numpy() * n).sum() / n.sum())        # the class mean: the shrink target
    return {"k": float(k), "sigma2_noise": s2_noise, "sigma2_skill": s2_skill, "mean_all": mean_all,
            "var_means": var_means, "n_bar": n_bar, "n_players": int(p.height), "n_actions": int(n.sum())}


def _player_stats(frame: pl.DataFrame, by: str, value: str) -> pl.DataFrame:
    return (frame.filter(pl.col(value).is_not_null() & pl.col(value).is_not_nan() & pl.col(PLAYER).is_not_null())
            .group_by(by, PLAYER)
            .agg(n=pl.len(), mean=pl.col(value).mean(),
                 ss=((pl.col(value) - pl.col(value).mean()) ** 2).sum()))


def fit_k(frame: pl.DataFrame, by: str = "type_class", *, values=VALUES, eps: float = EPS) -> dict:
    """Method-of-moments fit per class of `by` for each score column in `values`:
    `{value: {class: {k, sigma2_noise, sigma2_skill, var_means, n_bar, n_players,
    n_actions}}}`. Classes with fewer than 2 players of n >= 2 are left out."""
    f = with_class(frame) if by == "type_class" else frame
    out: dict = {}
    for v in values:
        stats = _player_stats(f, by, v)
        out[v] = {}
        for cls in sorted(stats[by].unique().to_list(), key=str):
            fit = _fit_one(stats.filter(pl.col(by) == cls), eps)
            if fit is not None:
                out[v][str(cls)] = fit
    out["by"] = by
    return out


# ── aggregates ────────────────────────────────────────────────────────────────

def player_aggregates(frame: pl.DataFrame, k_map: dict, *, values=VALUES) -> pl.DataFrame:
    """Per (player, class): n and, per score v in `values`, `v_mean`, `v_tilde`
    (shrunk), `v_factor` n/(n+k), `v_ci` (90% half-width), `v_lo`, `v_hi`. Classes
    without a fit get null shrunk values."""
    by = k_map.get("by", "type_class")
    f = with_class(frame) if by == "type_class" else frame
    base = f.group_by(PLAYER, by).agg(n=pl.len()).sort(PLAYER, by)
    for v in values:
        st = _player_stats(f, by, v).select(PLAYER, by, pl.col("n").alias(f"{v}_n"), pl.col("mean").alias(f"{v}_mean"))
        fits = k_map.get(v, {})
        st = st.with_columns(
            pl.col(by).cast(pl.String).replace_strict({c: fit["k"] for c, fit in fits.items()}, default=None,
                                                      return_dtype=pl.Float64).alias("_k"),
            pl.col(by).cast(pl.String).replace_strict({c: fit["sigma2_noise"] for c, fit in fits.items()},
                                                      default=None, return_dtype=pl.Float64).alias("_s2"),
            pl.col(by).cast(pl.String).replace_strict({c: fit.get("mean_all", 0.0) for c, fit in fits.items()},
                                                      default=None, return_dtype=pl.Float64).alias("_m"))
        nn = pl.col(f"{v}_n").cast(pl.Float64)
        st = st.with_columns(
            (nn / (nn + pl.col("_k"))).alias(f"{v}_factor"),
            (pl.col("_m") + (pl.col(f"{v}_mean") - pl.col("_m")) * nn / (nn + pl.col("_k"))).alias(f"{v}_tilde"),
            (Z90 * (pl.col("_s2") / (nn + pl.col("_k"))).sqrt()).alias(f"{v}_ci"))
        st = st.with_columns((pl.col(f"{v}_tilde") - pl.col(f"{v}_ci")).alias(f"{v}_lo"),
                             (pl.col(f"{v}_tilde") + pl.col(f"{v}_ci")).alias(f"{v}_hi")).drop("_k", "_s2", "_m")
        base = base.join(st, on=[PLAYER, by], how="left")
    return base


def fit_from_models(frame: pl.DataFrame, h, q, b, vocab, *, by: str = "type_class") -> dict:
    """X and D of every row of a valued frame from the trained Q / B nets, then `fit_k`.
    `frame` needs `actor_id_hash`, `type`, `V`, `tier`, `actor_side` and the `desc_*`
    columns; `h` is its h_pre matrix (MED-34, 2026-10-19)."""
    from engine.value import descriptors, q_model
    ids = descriptors.ids_from_frame(frame, vocab)
    Q = q_model.predict_q(q, h, ids, frame["tier"].to_list(), frame["actor_side"].to_list())
    B = q_model.predict_b(b, h, frame["tier"].to_list(), frame["actor_side"].to_list())
    V = frame["V"].to_numpy().astype(np.float64)
    scored = frame.select(PLAYER, "type").with_columns(pl.Series("D", Q - B), pl.Series("X", V - Q))
    return fit_k(scored, by=by)


# ── files ─────────────────────────────────────────────────────────────────────

def save(k_map: dict, path: Path) -> Path:
    path = Path(path)
    if path.suffix != ".json":
        path = path / FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(k_map, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def load(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / FILE_NAME
    return json.loads(path.read_text(encoding="utf-8"))
