"""
LightGBM yardstick (P6.1): flattening is permutation-invariant within a team and uses
NaN (never 0) for unknown; trains with early stopping on the fake corpus, predicts
probabilities, the metrics dict is complete, and save() writes the model directory
(model.txt, metrics.json, manifest.json) and the eval markdown.

    .venv\\Scripts\\python.exe test\\engine\\test_baseline.py [--cpu]
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from _harness import check, section, summary
from _fake_states import fake_arrays, make_corpus

from engine import config, manifest
from engine.data import dataset
from engine.model import baseline_lgbm as bl

cfg = config.load()
_, vocab, tr, _ = fake_arrays(24, seed=61, n_snapshots=25)
va = dataset.tensorize(make_corpus(8, seed=62, n_snapshots=25, prefix="val"), vocab)
te = dataset.tensorize(make_corpus(8, seed=63, n_snapshots=25, prefix="test"), vocab)

section("flatten")
X, names = bl.flatten(tr)
check("one row per snapshot", X.shape == (len(tr["t_s"]), len(names)))
check("feature names unique", len(set(names)) == len(names), f"{len(names)} features")
check("categorical features listed", set(bl.categorical(names)) >= {"ctx_cat_map_version", "ctx_cat_tier"})
check("unknown is NaN, not 0 (time_since_plant pre-plant)",
      bool(np.isnan(X[:, names.index("ctx_time_since_plant_s")]).any()))
rng = np.random.default_rng(0)
perm = np.r_[rng.permutation(5), 5 + rng.permutation(5)]
tp = dict(tr)
for k in ("player_num", "player_cat", "player_mask"):
    tp[k] = tr[k][:, perm]
tp["pair"] = tr["pair"][:, perm][:, :, perm]
Xp, _ = bl.flatten(tp)
check("permuting players within a team leaves the vector unchanged (up to float32 summation order)",
      np.allclose(X, Xp, rtol=1e-6, atol=1e-4, equal_nan=True))
check("team alive counts match ctx", np.array_equal(X[:, names.index("ct_alive")], X[:, names.index("ctx_alive_ct")]))

section("train / predict / evaluate")
booster = bl.train(tr, va, seed=cfg.train.seed, num_boost_round=200, early_stopping=20)
p = bl.predict(booster, te)
check("one probability per test snapshot", p.shape == (len(te["t_s"]),))
check("probabilities in (0, 1)", bool(((p > 0) & (p < 1)).all()))
res = bl.evaluate(booster, te, cfg, vocab)
check("metrics dict complete", {"n", "logloss", "brier", "auc", "ece", "reliability", "per_phase",
                                "per_map", "best_iteration"} <= set(res), str(sorted(res)))
check("metrics finite", all(np.isfinite(res[k]) for k in ("logloss", "brier", "ece")))
check("20 reliability bins", len(res["reliability"]) == cfg.calib.n_bins)
check("per-phase breakdown covers every test snapshot", sum(m["n"] for m in res["per_phase"].values()) == res["n"])
check("per-map uses map version names", set(res["per_map"]) <= {"de_fake@1", "de_other@2"})
check("early stopping recorded", res["best_iteration"] >= 1)

section("save")
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "models" / "baseline_v1"
    bl.save(out, booster, {"val": bl.evaluate(booster, va, cfg, vocab), "test_pro": res,
                           "best_iteration": res["best_iteration"]}, cfg=cfg, eval_dir=Path(d) / "eval")
    for name in ("model.txt", "metrics.json", "manifest.json"):
        check(f"{name} written", (out / name).is_file())
    check("manifest kind baseline_lgbm", manifest.read(out)["kind"] == "baseline_lgbm")
    md = (Path(d) / "eval" / "baseline_v1.md")
    check("eval/baseline_v1.md has per-phase tables", md.is_file() and "by phase" in md.read_text(encoding="utf-8"))
    import lightgbm as lgb
    re = lgb.Booster(model_file=str(out / "model.txt"))
    check("saved model reloads and predicts the same", np.allclose(re.predict(bl.flatten(te)[0], num_iteration=booster.best_iteration), p))

sys.exit(summary())
