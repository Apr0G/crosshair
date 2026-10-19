"""
P9.4.2 empirical-Bayes shrinkage: planted true skills recovered within the 90% interval
for 90% ± 3% of players; shrinkage factor monotone in n; k > 0 (also with no skill
spread); k per class near the truth; formulas; save / load.

    .venv\\Scripts\\python.exe test\\engine\\test_shrinkage.py
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from _harness import check, section, summary
from engine.value import shrinkage as sh


def players_frame(n_players: int, tau: float, sigma: float, seed: int, type_: str = "kill",
                  n_lo: int = 5, n_hi: int = 200, offset: int = 0) -> tuple[pl.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    skill = {offset + p: float(rng.normal(0, tau)) for p in range(n_players)}
    pid, xs, ds = [], [], []
    for p, th in skill.items():
        n = int(rng.integers(n_lo, n_hi + 1))
        pid += [p] * n
        xs += list(th + sigma * rng.normal(size=n))
        ds += list(0.5 * th + sigma * rng.normal(size=n))
    return pl.DataFrame({"actor_id_hash": pid, "type": [type_] * len(pid), "X": xs, "D": ds}), skill


section("formulas")
check("shrink(0.1, 50, 50) = 0.05", math.isclose(sh.shrink(0.1, 50, 50), 0.05))
check("interval = 1.645 · sqrt(σ² / (n + k))", math.isclose(float(sh.interval(50, 50, 0.01)), 1.645 * math.sqrt(0.01 / 100)))
ns = np.arange(1, 2001)
for k in (0.5, 10.0, 300.0):
    f = sh.factor(ns, k)
    check(f"factor n/(n+k) strictly increasing in n (k={k})", bool(np.all(np.diff(f) > 0)) and f[-1] < 1)
check("type classes: kill / utility / movement", (sh.type_class("kill"), sh.type_class("flash"), sh.type_class("rotation"))
      == ("kill", "utility", "movement"))
check("unknown type -> other", sh.type_class("teabag") == "other")

section("planted skills recovered (2000 players, τ = 0.01, σ = 0.1)")
TAU, SIGMA = 0.01, 0.1
frame, skill = players_frame(2000, TAU, SIGMA, seed=1)
fit = sh.fit_k(frame)
fk = fit["X"]["kill"]
true_k = SIGMA ** 2 / TAU ** 2
check("k > 0", fk["k"] > 0)
check("k near σ²/τ² = 100 (±30%)", abs(fk["k"] - true_k) / true_k < 0.3, f"k = {fk['k']:.1f}")
check("σ²_noise near 0.01 (±3%)", abs(fk["sigma2_noise"] - SIGMA ** 2) / SIGMA ** 2 < 0.03, f"{fk['sigma2_noise']:.5f}")
agg = sh.player_aggregates(frame, fit)
truth = np.array([skill[p] for p in agg["actor_id_hash"].to_list()])
inside = np.abs(agg["X_tilde"].to_numpy() - truth) <= agg["X_ci"].to_numpy()
cov = float(inside.mean())
check("true skill inside the 90% interval for 90% ± 3% of players", abs(cov - 0.90) <= 0.03, f"{cov:.1%}")
check("aggregates: one row per player and class with n", agg.height == 2000 and int(agg["n"].sum()) == frame.height)
check("X_tilde = X_mean · n / (n + k)",
      np.allclose(agg["X_tilde"].to_numpy(), agg["X_mean"].to_numpy() * agg["n"].to_numpy() / (agg["n"].to_numpy() + fk["k"])))
check("shrunk estimate beats the raw mean (MSE)",
      float(((agg["X_tilde"].to_numpy() - truth) ** 2).mean()) < float(((agg["X_mean"].to_numpy() - truth) ** 2).mean()))
check("D fitted and shrunk too", "kill" in fit["D"] and agg["D_tilde"].null_count() == 0)
check("lo / hi = tilde ∓ ci", np.allclose(agg["X_hi"].to_numpy() - agg["X_lo"].to_numpy(), 2 * agg["X_ci"].to_numpy()))
facs = agg.sort("n")["X_factor"].to_numpy()
check("per-player factor monotone in n", bool(np.all(np.diff(facs) >= -1e-15)))

section("per class, degenerate cases")
f2, _ = players_frame(400, 0.03, 0.1, seed=2, type_="flash", offset=10_000)
both = pl.concat([frame, f2])
fb = sh.fit_k(both)
check("classes fitted separately", set(fb["X"]) == {"kill", "utility"})
check("utility k near 0.01 / 0.0009 ≈ 11 (±40%)", abs(fb["X"]["utility"]["k"] - 0.01 / 0.0009) / (0.01 / 0.0009) < 0.4,
      f"k = {fb['X']['utility']['k']:.1f}")
f0, _ = players_frame(300, 0.0, 0.1, seed=3)
k0 = sh.fit_k(f0)["X"]["kill"]
check("no planted spread: k positive and finite", 0 < k0["k"] < math.inf, f"k = {k0['k']:.3g}")
flat = pl.DataFrame({"actor_id_hash": [p for p in range(50) for _ in range(10)], "type": ["kill"] * 500,
                     "X": [0.1, -0.1] * 250, "D": [0.0] * 500})
kf = sh.fit_k(flat)
check("identical player means: σ²_skill floored at ε, k = σ²_noise / ε",
      kf["X"]["kill"]["sigma2_skill"] == sh.EPS and math.isclose(kf["X"]["kill"]["k"], kf["X"]["kill"]["sigma2_noise"] / sh.EPS))
check("zero noise: k still > 0", kf["D"]["kill"]["k"] > 0)
withnull = frame.with_columns(pl.when(pl.int_range(pl.len()) % 10 == 0).then(None).otherwise(pl.col("X")).alias("X"))
check("null X rows ignored", sh.fit_k(withnull)["X"]["kill"]["n_actions"] < fk["n_actions"])
check("class with a single player is left out", "kill" not in sh.fit_k(frame.filter(pl.col("actor_id_hash") == 0))["X"])

section("save / load")
with tempfile.TemporaryDirectory() as tmp:
    p = sh.save(fb, Path(tmp))
    check("written as shrinkage.json", p.name == "shrinkage.json")
    back = sh.load(Path(tmp))
    check("round trip equal", back == fb)
    check("aggregates identical from the reloaded fit", sh.player_aggregates(both, back).equals(sh.player_aggregates(both, fb)))

sys.exit(summary())
