"""
LightGBM yardstick (P6.1): flattening is permutation-invariant within a team and uses
NaN (never 0) for unknown; trains with early stopping on the fake corpus, predicts
probabilities, the metrics dict is complete, and save() writes the model directory
(model.txt, metrics.json, manifest.json) and the eval markdown.

    .venv\\Scripts\\python.exe test\\engine\\test_baseline.py [--cpu]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from _harness import ROOT, check, require, section, summary
require("lightgbm", "the yardstick needs lightgbm (requirements-engine.txt)")
from _fake_states import fake_arrays, make_corpus  # noqa: E402

from engine import config, manifest  # noqa: E402
from engine.data import dataset  # noqa: E402
from engine.data.dataset import PLAYER_LAYOUT  # noqa: E402
from engine.model import baseline_lgbm as bl  # noqa: E402
from engine.model import calibrate  # noqa: E402

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
# LOW-31: an alive player's unknown equipment / helmet / defuser makes the team sum NaN,
# a dead player's unknown value does not
P_ = PLAYER_LAYOUT
alive0 = tr["player_num"][:, :5, P_.col("alive")] > 0.5
r_all = int(np.flatnonzero(alive0.all(1))[0])
r_dead = int(np.flatnonzero(~alive0.all(1) & alive0.any(1))[0])
k_dead = int(np.flatnonzero(~alive0[r_dead])[0])
tu = {k: v.copy() for k, v in tr.items()}
for field in ("equipment_value", "has_helmet", "has_defuser"):
    pres = P_.presence_of[P_.col(field)]
    tu["player_num"][r_all, 0, pres] = 0.0
    tu["player_num"][r_dead, k_dead, pres] = 0.0
Xu, _ = bl.flatten(tu)
sums = ("ct_equip_sum", "ct_helmets", "ct_defusers")
check("alive player's unknown equipment / helmet / defuser -> team sum NaN (LOW-31)",
      all(np.isnan(Xu[r_all, names.index(c)]) for c in sums))
check("dead player's unknown value leaves the team sum known",
      all(np.isfinite(Xu[r_dead, names.index(c)]) for c in sums))
check("known rows unchanged", np.allclose(np.delete(Xu, [r_all, r_dead], 0), np.delete(X, [r_all, r_dead], 0),
                                          equal_nan=True))

section("train / predict / evaluate")
booster = bl.train(tr, va, seed=cfg.train.seed, num_boost_round=200, early_stopping=20)
p = bl.predict(booster, te)
check("one probability per test snapshot", p.shape == (len(te["t_s"]),))
res = bl.evaluate(booster, te, cfg, vocab)
y_tr, y_te = tr["y_win"][~np.isnan(tr["y_win"])], te["y_win"][~np.isnan(te["y_win"])]
ll_const = calibrate.logloss(np.full(len(y_te), y_tr.mean()), y_te)
check("beats the constant (train base rate) predictor on test logloss (LOW-70)", res["logloss"] < ll_const,
      f"{res['logloss']:.4f} vs {ll_const:.4f}")
check("metrics dict complete", {"n", "logloss", "brier", "auc", "ece", "reliability", "per_phase",
                                "per_map", "best_iteration"} <= set(res), str(sorted(res)))
check("metrics finite", all(np.isfinite(res[k]) for k in ("logloss", "brier", "ece")))
check("20 reliability bins", len(res["reliability"]) == cfg.calib.n_bins)
check("per-phase breakdown covers every test snapshot", sum(m["n"] for m in res["per_phase"].values()) == res["n"])
check("per-map uses map version names", set(res["per_map"]) <= {"de_fake@1", "de_other@2"})
check("early stopping recorded", res["best_iteration"] >= 1)

section("early stopping on val pro (LOW-30)")
es = bl.early_stop_rows(va)
check("val has pro and non-pro rows (fixture)", "pro" in set(va["tier"]) and len(set(va["tier"])) >= 2, str(np.unique(va["tier"])))
check("early-stop rows are the labelled pro rows",
      set(va["tier"][es]) == {"pro"} and len(es) == int(((va["tier"] == "pro") & ~np.isnan(va["y_win"])).sum()))
va_fc = dataset.take(va, np.flatnonzero(va["tier"] != "pro"))
check("no pro rows -> every labelled val row", np.array_equal(bl.early_stop_rows(va_fc),
                                                              np.flatnonzero(~np.isnan(va_fc["y_win"]))))
va_pro = dataset.take(va, es)
p_vp = booster.predict(bl.flatten(va_pro)[0], num_iteration=booster.best_iteration)
ll_vp = calibrate.logloss(p_vp, va_pro["y_win"])
check("booster's best val score is the val-pro logloss",
      abs(booster.best_score["val"]["binary_logloss"] - ll_vp) < 1e-4,
      f"{booster.best_score['val']['binary_logloss']:.6f} vs {ll_vp:.6f}")

section("torch-free (LOW-32)")
probe = ("import sys\n"
         "from pathlib import Path\n"
         "import polars as pl\n"
         "from engine.model import baseline_lgbm as bl\n"
         "rc = bl.main(['--train', sys.argv[1], '--val', sys.argv[1], '--test', sys.argv[1]])\n"
         "from engine.value import neighbours as nb\n"
         "nb.save_index(nb.NeighbourIndex(parts={}, table=pl.DataFrame({'a': [1]}), d=4), Path(sys.argv[2]))\n"
         "print(rc, 'torch' in sys.modules)\n")
with tempfile.TemporaryDirectory() as d:
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", probe, str(Path(d) / "no_split"),
                        str(Path(d) / "neighbours_v1")], cwd=ROOT, capture_output=True, text=True, timeout=120)
check("baseline main (exit 2 on a missing split) and neighbours.save_index never import torch",
      r.stdout.strip().splitlines()[-1:] == ["2 False"], (r.stdout.strip() or r.stderr.strip())[-200:])

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
