"""
P9.2 QNet / BNet: shapes; overfit tiny (Huber loss < 0.01); checkpoint round trip;
GPU and CPU agree; `evaluate_q` table shape and sign checks; input encoding.

    .venv\\Scripts\\python.exe test\\engine\\test_q_model.py [--cpu]
"""
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from _harness import check, require, section, skip, summary
torch = require("torch", "needs the platform torch build (docs/ENGINE_ENV.md)")   # LOW-61
from _value_states import D_TOY, device, split_by_match, synthetic_corpus  # noqa: E402
from engine import config  # noqa: E402
from engine.value import q_model as qm  # noqa: E402
from engine.value.descriptors import FIELDS, DescriptorVocab, ids_from_frame  # noqa: E402

DEV = device()
cfg = config.load()
print(f"device: {DEV}")

# ── shapes and encoding ───────────────────────────────────────────────────────

section("shapes and input encoding")
frame, h = synthetic_corpus(64, seed=3, noise=0.0)
vocab = DescriptorVocab.fit_frame(frame)
ids = ids_from_frame(frame, vocab)
q = qm.QNet(D_TOY, vocab.sizes()).to(DEV)
b = qm.BNet(D_TOY).to(DEV)
th, td, tt, ts = qm._tensors(h, ids, frame["tier"], frame["actor_side"], DEV)
check("QNet output (n,)", tuple(q(th, td, tt, ts).shape) == (64,))
check("BNet output (n,)", tuple(b(th, tt, ts).shape) == (64,))
check("QNet: one embedding per descriptor field", len(q.emb) == len(FIELDS))
check("hidden widths (512, 256) by default",
      [m.out_features for m in q.mlp if isinstance(m, torch.nn.Linear)] == [512, 256, 1])
check("predict_q -> float64[n]", qm.predict_q(q, h, ids, frame["tier"], frame["actor_side"]).shape == (64,))
check("predict on zero rows", qm.predict_b(b, np.zeros((0, D_TOY), np.float32), [], []).shape == (0,))
check("tier ids: pro 0, faceit 1, other -> UNK", list(qm.tier_ids(["pro", "faceit", "lan", None])) == [0, 1, 2, 2])
check("side bits: ct 1, t 0", list(qm.side_bits(["ct", "t"])) == [1.0, 0.0])
try:
    qm.side_bits(["?"])
    check("unknown side raises (never silently T)", False)
except ValueError:
    check("unknown side raises (never silently T)", True)
hf = frame.with_columns(pl.Series("h_pre", h.tolist(), dtype=pl.List(pl.Float32)))
check("h_matrix reads the h_pre column", np.allclose(qm.h_matrix(hf), h))
try:
    qm.h_matrix(frame.with_columns(pl.lit(None, dtype=pl.List(pl.Float32)).alias("h_pre")))
    check("h_matrix refuses null embeddings", False)
except ValueError:
    check("h_matrix refuses null embeddings", True)
check("Huber δ = 0.05", qm.HUBER_DELTA == 0.05)
r = torch.tensor([0.0, 0.0])
check("huber: quadratic inside δ, linear outside",
      math.isclose(float(qm.huber(torch.tensor([0.02, 0.0]), r)), 0.5 * 0.02 ** 2 / 2, rel_tol=1e-5)
      and math.isclose(float(qm.huber(torch.tensor([0.25, 0.0]), r)), 0.05 * (0.25 - 0.025) / 2, rel_tol=1e-5))

# ── overfit tiny ──────────────────────────────────────────────────────────────

section("overfit tiny")
rng = np.random.default_rng(0)
tiny = frame.with_columns(pl.Series("V", rng.uniform(-0.5, 0.5, size=64)))
all_train = lambda f: ["train"] * f.height                           # noqa: E731
p = qm.QTrainParams(lr=3e-3, wd=0.0, batch_size=64, max_epochs=400, patience=400)
q1, b1, m1 = qm.train_q(tiny, h, vocab, cfg, all_train, DEV, params=p, save=False)
first = m1["history"][0]["train_q"]
check("Huber loss < 0.01 after overfitting 64 rows", m1["train_loss_q"] < 0.01, f"{m1['train_loss_q']:.2e}")
check("Huber loss fell by > 20x", m1["train_loss_q"] < first / 20, f"{first:.2e} -> {m1['train_loss_q']:.2e}")
check("BNet trains too (finite, decreased)", math.isfinite(m1["train_loss_b"]) and m1["train_loss_b"] < m1["history"][0]["train_b"])
try:
    bad_cfg = replace(cfg, value=replace(cfg.value, q_finetune_encoder=True))
    qm.train_q(tiny, h, vocab, bad_cfg, all_train, DEV, params=p, save=False)
    check("q_finetune_encoder: true is refused", False)
except NotImplementedError:
    check("q_finetune_encoder: true is refused", True)

# ── corpus training, checkpoint, devices ──────────────────────────────────────

section("train on a corpus, checkpoint round trip")
corpus, hc = synthetic_corpus(6000, seed=11, n_matches=60)
vc = DescriptorVocab.fit_frame(corpus)
pc = qm.QTrainParams(max_epochs=25, patience=4, batch_size=512)
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "q_v1"
    q2, b2, m2 = qm.train_q(corpus, hc, vc, cfg, split_by_match, DEV, params=pc, out_dir=out)
    check("files written", all((out / f).is_file() for f in ("q.pt", "b.pt", "vocab.json", "metrics.json", "manifest.json")))
    check("early stopping used the val split", m2["n_val"] > 0 and m2["best_epoch_q"] >= 0, f"epochs={m2['epochs']}")
    check("seed recorded", m2["seed"] == cfg.train.seed)
    lq, lb, lv, lm = qm.load_q(out, device=DEV)
    idc = ids_from_frame(corpus, vc)
    a1 = qm.predict_q(q2, hc, idc, corpus["tier"], corpus["actor_side"])
    a2 = qm.predict_q(lq, hc, ids_from_frame(corpus, lv), corpus["tier"], corpus["actor_side"])
    check("reloaded QNet predicts identically", np.allclose(a1, a2, atol=1e-6))
    check("reloaded BNet predicts identically",
          np.allclose(qm.predict_b(b2, hc, corpus["tier"], corpus["actor_side"]),
                      qm.predict_b(lb, hc, corpus["tier"], corpus["actor_side"]), atol=1e-6))
    check("vocab and metrics reloaded", lv == vc and lm.get("n_train") == m2["n_train"])
    if DEV != "cpu":
        cq, _, _, _ = qm.load_q(out, device="cpu")
        a3 = qm.predict_q(cq, hc, idc, corpus["tier"], corpus["actor_side"])
        check("GPU and CPU predictions agree (1e-4)", np.allclose(a1, a3, atol=1e-4), f"max {np.abs(a1 - a3).max():.2e}")
    else:
        skip("GPU and CPU predictions agree", "running on CPU")

section("evaluate_q")
test_rows = np.array(split_by_match(corpus)) != "train"
tf, ht = corpus.filter(pl.Series(test_rows)), hc[test_rows]
res = qm.evaluate_q(q2, b2, tf, ht, vc, min_bin_n=20)
check("Q̂ calibration table: 20 bins, documented columns",
      res["q_bins"].height == 20 and res["q_bins"].schema == pl.Schema(qm.CALIB_SCHEMA))
check("B̂ calibration table: 20 bins", res["b_bins"].height == 20)
check("bins cover every row", int(res["q_bins"]["n"].sum()) == tf.height)
check("Spearman Q̂ vs V finite and > B̂'s (descriptor carries signal)",
      math.isfinite(res["spearman_q"]) and res["spearman_q"] > res["spearman_b"],
      f"q={res['spearman_q']:.3f} b={res['spearman_b']:.3f}")
check("per-type table has every type", set(res["per_type"]["type"].to_list()) == set(tf["type"].to_list()))
check("four sign checks reported", set(res["sanity"]) == {c[0] for c in qm.SANITY_CHECKS})
check("sign checks pass on the planted corpus", res["sanity_ok"], str({k: v["mean_Q"] for k, v in res["sanity"].items()}))
check("eval markdown renders", "Q̂ calibration" in qm.eval_markdown(res))
perfect = qm.calibration_table(np.linspace(-1, 1, 4000), np.linspace(-1, 1, 4000))
check("calibration_table: perfect predictions -> every bin ok", perfect["ok"].to_list() == [True] * 20)
small = qm.calibration_table(np.zeros(100), np.ones(100))
check("calibration_table: bins with n < 200 -> ok null", small["ok"].null_count() == small.height)
check("calibration_table: off by 1 -> not ok", qm.calibration_table(np.zeros(4000), np.ones(4000))["ok"].to_list() == [False] * 20)

sys.exit(summary())
