"""
Calibration (P6.5, Appendix B.9) and drift checks (P6.6.2): a known temperature is
recovered within 5% from synthetic logits; ECE on a hand-computed table; the
per-(map_version, phase) rule (kept only if val ECE improves by >= 0.002); terminal
pin; martingale check flags drift and passes a driftless process; phases; metrics.

    .venv\\Scripts\\python.exe test\\engine\\test_calibrate.py [--cpu]
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from _harness import check, close, section, summary

from engine.model import calibrate as C

rng = np.random.default_rng(0)

section("temperature recovery")
for T_true in (2.0, 0.5):
    z = rng.normal(0, 1.5, 200_000)
    y = (rng.random(z.size) < C.sigmoid(z)).astype(float)
    T = C.fit_temperature(z * T_true, y)
    check(f"T={T_true} recovered within 5%", abs(T / T_true - 1) < 0.05, f"fitted {T:.4f}")
z = rng.normal(0, 1, 50_000)
y = (rng.random(z.size) < C.sigmoid(z)).astype(float)
check("calibrated logits -> T ~ 1", abs(C.fit_temperature(z, y) - 1) < 0.05)
check("one-class labels -> T = 1 (no fit)", C.fit_temperature(z, np.ones_like(z)) == 1.0)

section("ECE on a known table")
p = np.array([0.1] * 10 + [0.9] * 10)
yk = np.array([1, 1] + [0] * 8 + [1] * 8 + [0, 0], float)
# bin 2 (0.10-0.15): mean 0.1, observed 0.2; bin 18 (0.90-0.95): mean 0.9, observed 0.8
check("ECE = 0.5*|0.1-0.2| + 0.5*|0.9-0.8| = 0.1", close(C.ece(p, yk, 20), 0.1, 1e-12), f"{C.ece(p, yk, 20)}")
rel = C.reliability(p, yk, 20)
check("20 bins", len(rel) == 20)
check("occupied bins and counts", [r["n"] for r in rel if r["n"]] == [10, 10])
check("perfect calibration -> ECE 0", C.ece(np.array([0.25] * 4), np.array([1, 0, 0, 0.0])) == 0.0)
viol = C.bin_violations(rel, 0.03, 5)
check("bin violations: both bins off by 0.1", len(viol) == 2)
check("bins below min_bin_n not judged", C.bin_violations(rel, 0.03, 11) == [])

section("per-group rule")
n = 40_000
mv = np.where(rng.random(n) < 0.5, "de_a@1", "de_b@1")
ph = np.where(rng.random(n) < 0.5, "0_30", "post_plant")
z = rng.normal(0, 1.5, n)
y = (rng.random(n) < C.sigmoid(z)).astype(float)
T_g = np.where(mv == "de_a@1", 0.4, 3.0)                    # opposite miscalibration per map
cal = C.fit_calibration(z * T_g, y, mv, ph, n_bins=20, min_group_n=200)
check("different miscalibration per group -> groups kept", cal["mode"] == "group",
      f"ece single {cal['ece_val_single']:.4f} group {cal['ece_val_group']:.4f}")
check("group temperatures recovered (within 10%)",
      abs(cal["groups"]["de_a@1|0_30"] / 0.4 - 1) < 0.1 and abs(cal["groups"]["de_b@1|post_plant"] / 3.0 - 1) < 0.1)
pg = C.calibrated_probs(z * T_g, cal, mv, ph)
check("calibrated_probs uses the group temperatures", C.ece(pg, y) < C.ece(C.sigmoid(z * T_g / cal["T"]), y))
cal_same = C.fit_calibration(z * 2.0, y, mv, ph, n_bins=20, min_group_n=200)
check("same miscalibration everywhere -> single T kept", cal_same["mode"] == "single" and cal_same["groups"] == {},
      f"improvement {cal_same['ece_val_single'] - cal_same['ece_val_group']:.4f}")
check("single T recovered", abs(cal_same["T"] / 2.0 - 1) < 0.05)
check("small groups fall back to the single T",
      C.fit_calibration(z[:300] * 2, y[:300], mv[:300], ph[:300], min_group_n=1000)["candidate_groups"] == {})
with tempfile.TemporaryDirectory() as d:
    path = C.save_calibration(Path(d), cal)
    check("calibration.json written", path.name == "calibration.json" and path.is_file())

section("terminal pin")
rk = np.array([f"m#{i}" for i in range(100)])
yt = (np.arange(100) % 2).astype(float)
pt = np.where(yt > 0, 0.995, 0.004)
check("pinned terminals pass", C.terminal_pin(pt, yt, np.ones(100), rk)["ok"] is True)
pt[0] = 0.5
pt[1] = 0.5
check("2% unpinned -> fails the 99% rule", C.terminal_pin(pt, yt, np.ones(100), rk)["ok"] is False)

section("martingale check")
rounds, steps = 400, 60
p_list, rk_list, t_list = [], [], []
for r in range(rounds):
    w = 0.5 + np.cumsum(rng.choice([-0.004, 0.004], steps))
    p_list.append(np.clip(w, 0.01, 0.99))
    rk_list += [f"r{r}"] * steps
    t_list.append(np.arange(steps) * 0.5)
pm, rkm, tm = np.concatenate(p_list), np.array(rk_list), np.concatenate(t_list)
res = C.martingale_check(pm, rkm, tm, (1, 5, 10), n_bins=20, min_n=500, tol=0.01)
check("driftless process passes", res["ok"], f"{len(res['violations'])} violations")
check("some bins judged", any(r["judged"] for r in res["rows"]))
drift = pm + np.concatenate([np.arange(steps) * 0.002 for _ in range(rounds)])
res2 = C.martingale_check(np.clip(drift, 0, 1), rkm, tm, (1, 5, 10), n_bins=20, min_n=500, tol=0.01)
check("+0.004/s drift is flagged", not res2["ok"] and any(v["k"] in (5, 10) for v in res2["violations"]))
perm = rng.permutation(len(pm))
res3 = C.martingale_check(pm[perm], rkm[perm], tm[perm], (1, 5, 10), n_bins=20, min_n=500, tol=0.01)
check("input order does not matter", [r["mean_drift"] for r in res3["rows"]] == [r["mean_drift"] for r in res["rows"]])

section("phases and metrics")
check("phases", list(C.phase_of([0, 0, 0, 0, 1], [100, 60, 30, 10, 30], [10, 50, 85, 105, 60]))
      == ["0_30", "30_75", "75_plus", "last_15", "post_plant"])
m = C.metrics(np.array([0.9, 0.2, 0.7, 0.4]), np.array([1, 0, 1, 0.0]))
check("metrics keys", set(m) == {"n", "logloss", "brier", "auc", "ece", "reliability"})
check("auc 1 on a separable set", m["auc"] == 1.0)
check("brier", close(m["brier"], (0.01 + 0.04 + 0.09 + 0.16) / 4, 1e-12))
check("logloss", close(m["logloss"], -np.mean(np.log([0.9, 0.8, 0.7, 0.6])), 1e-9))
check("NaN labels dropped", C.metrics(np.array([0.5, 0.5]), np.array([1.0, np.nan]))["n"] == 1)
check("one-class auc is None", C.auc(np.array([0.2, 0.3]), np.array([1, 1])) is None)

sys.exit(summary())
