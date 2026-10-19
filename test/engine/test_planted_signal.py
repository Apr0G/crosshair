"""
P9.4.3 planted signal: in a synthetic corpus one player's V is shifted by +0.02 on kills.
Pipeline: train Q / B on the corpus -> scores.compute per action -> to_frame ->
shrinkage.fit_k -> player_aggregates. The planted player must come out with
X̃(kill) >= 0.01 and every other player with |X̃(kill)| < 0.005.

    .venv\\Scripts\\python.exe test\\engine\\test_planted_signal.py [--cpu]
"""
import sys

import numpy as np
import polars as pl

from _harness import check, section, summary
from _value_states import device, split_by_match, synthetic_corpus
from engine import config
from engine.value import q_model as qm
from engine.value import scores as sc
from engine.value import shrinkage as sh
from engine.value.descriptors import DescriptorVocab, ids_from_frame

DEV = device()
cfg = config.load()
PLANTED, SHIFT = 1007, 0.02
TYPES = {"kill": (0.5, None), "peek": (0.25, None), "rotation": (0.25, None)}
print(f"device: {DEV}")

section("corpus and Q / B")
frame, h = synthetic_corpus(48_000, seed=31, n_matches=80, n_players=40, noise=0.03, type_p=TYPES,
                            planted={(PLANTED, "kill"): SHIFT})
vocab = DescriptorVocab.fit_frame(frame)
q, b, m = qm.train_q(frame, h, vocab, cfg, split_by_match, DEV,
                     params=qm.QTrainParams(max_epochs=30, patience=4, batch_size=1024), save=False)
ids = ids_from_frame(frame, vocab)
Q = qm.predict_q(q, h, ids, frame["tier"], frame["actor_side"])
B = qm.predict_b(b, h, frame["tier"], frame["actor_side"])
bias = float((Q - frame["Q_true"].to_numpy()).mean())
check("Q̂ unbiased against the true expected value (|mean err| < 0.003)", abs(bias) < 0.003, f"{bias:+.4f}")

section("scores -> shrinkage")
valued = sc.valued_from_frame(frame)
scored = [sc.compute(sa, Q[i], B[i], None, cfg) for i, sa in enumerate(valued)]
out = sc.to_frame(scored, run_id="planted")
fit = sh.fit_k(out)
agg = sh.player_aggregates(out, fit).filter(pl.col("type_class") == "kill")
print(f"  k(kill) = {fit['X']['kill']['k']:.1f}  σ²_noise = {fit['X']['kill']['sigma2_noise']:.2e}")
pl_row = agg.filter(pl.col("actor_id_hash") == PLANTED)
others = agg.filter(pl.col("actor_id_hash") != PLANTED)
xt = float(pl_row["X_tilde"][0])
worst = float(others["X_tilde"].abs().max())
check("planted player recovered: X̃(kill) >= 0.01", xt >= 0.01, f"X̃ = {xt:.4f} (n = {int(pl_row['n'][0])})")
check("every other player |X̃(kill)| < 0.005", worst < 0.005, f"max |X̃| = {worst:.4f} over {others.height}")
check("planted player is the top X̃", int(agg.sort("X_tilde", descending=True)["actor_id_hash"][0]) == PLANTED)
d_gap = float(pl_row["D_tilde"][0]) - float(others["D_tilde"].median())
check("the shift shows up in X, not D (planted D̃ within 0.005 of the others' median)",
      abs(d_gap) < 0.005, f"gap = {d_gap:+.4f}")
check("40 players aggregated", agg.height == 40 and np.isfinite(agg["X_tilde"].to_numpy()).all())

sys.exit(summary())
