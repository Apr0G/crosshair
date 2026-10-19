"""
Counterfactual sanity gate (P7.2) with rule stubs: a sensible W passes every test; a
backwards W fails un-kill and un-plant; thresholds come from `config.sanity` (changing
one flips the verdict); the markdown report is written; missing OOD inputs, missing
sample kinds and intervention errors are counted against the gate; `main()` exits 2
without a model and states.

    .venv\\Scripts\\python.exe test\\engine\\test_sanity_gate.py
"""
import contextlib
import dataclasses
import io
import itertools
import random
import sys
import tempfile
from pathlib import Path


from _harness import check, section, summary
from _cf_states import (RuleStub, broken_win, embed, gate_samples, good_win, make_kill_pair,
                        make_snapshot)

from engine import config
from engine.counterfactual import sanity
from engine.counterfactual.ood import OODDetector

cfg = config.load()
samples = gate_samples(n=30, seed=0)

# a temporary OOD index over factual embeddings (P7.2.7 allows one "built here")
factual = [s for k, a, b, _ in samples for s in (a, b)]
rng = random.Random(7)
extra = [make_snapshot(rng.randint(2, 5), rng.randint(2, 5), seed=10_000 + i) for i in range(400)]
det = OODDetector(k=cfg.sanity.ood_k, percentile=cfg.sanity.ood_percentile).fit(embed(factual + extra))
held = [s for k, a, b, _ in gate_samples(n=30, seed=1) for s in (a, b)]
det.calibrate(embed(held))

# ── good stub ─────────────────────────────────────────────────────────────────
section("sensible W passes")

good = RuleStub(good_win)
rep = sanity.run_gate(good, samples, cfg, ood=det, embed=embed, model="rule:good")
for name, t in rep.tests.items():
    check(f"good stub: {name} passes", t.passed, f"{t.metrics} {t.note}")
check("good stub: gate passed", rep.passed)
check("one batched Predictor call for a small gate", good.calls == 1, f"{good.calls} calls")
check("every sample counted", rep.n_samples == len(samples))
check("un-kill median |dW| is the stub's 0.1", abs(rep.tests["unkill"].metrics["median_abs_dw"] - 0.1) < 1e-9)
check("order test: 5 members -> empty + singletons + leave-one-out (11) per window",
      rep.tests["order"].metrics["n_coalitions"] == 10 * 11)

# ── broken stub ───────────────────────────────────────────────────────────────
section("backwards W fails")

bad = sanity.run_gate(RuleStub(broken_win), samples, cfg, ood=det, embed=embed, model="rule:broken")
check("broken stub: gate failed", not bad.passed)
check("broken stub: un-kill fails", not bad.tests["unkill"].passed, f"{bad.tests['unkill'].metrics}")
check("broken stub: un-plant fails", not bad.tests["unplant"].passed, f"{bad.tests['unplant'].metrics}")
check("broken stub: failed() names them", {"unkill", "unplant"} <= set(bad.failed()))
w = bad.tests["unkill"].worst
check("worst cases: <= 20, with match/round/tick/before/after",
      0 < len(w) <= 20 and {"match", "round", "tick", "w_before", "w_after"} <= set(w[0]))
check("worst cases sorted worst first", all(a["signed"] <= b["signed"] for a, b in zip(w, w[1:])))


def jumpy_win(s, h=None) -> float:
    """Swings wildly with smoke count: fails the smoke max-abs test."""
    return good_win(s) + 0.3 * s.ctx.n_active_smokes - 0.15


smk = sanity.run_gate(RuleStub(jumpy_win), samples, cfg, ood=det, embed=embed)
check("smoke-sensitive W fails the smoke test", not smk.tests["smoke"].passed, f"{smk.tests['smoke'].metrics}")

# ── thresholds from config ────────────────────────────────────────────────────
section("thresholds read from config.sanity")

t = rep.tests["unkill"].thresholds
check("unkill thresholds are cfg.sanity values",
      t == {"toward_rate >=": cfg.sanity.unkill_toward_rate, "median_abs_dw >=": cfg.sanity.unkill_median_abs})
check("ood threshold rate is cfg.sanity.ood_max_rate", rep.tests["ood"].thresholds["flag_rate <="] == cfg.sanity.ood_max_rate)
strict = dataclasses.replace(cfg, sanity=dataclasses.replace(cfg.sanity, unkill_median_abs=0.5))
rs = sanity.run_gate(RuleStub(good_win), samples, strict, ood=det, embed=embed)
check("raising unkill_median_abs to 0.5 fails the good stub", not rs.tests["unkill"].passed and not rs.passed)
loose = dataclasses.replace(cfg, sanity=dataclasses.replace(cfg.sanity, unplant_down_rate=0.0, unkill_toward_rate=0.0,
                                                            unkill_median_abs=0.0))
rl = sanity.run_gate(RuleStub(broken_win), samples, loose, ood=det, embed=embed)
check("zero thresholds let the broken stub through un-kill and un-plant",
      rl.tests["unkill"].passed and rl.tests["unplant"].passed)
tight_ood = dataclasses.replace(cfg, sanity=dataclasses.replace(cfg.sanity, ood_max_rate=-1.0))
check("ood_max_rate < 0 fails the OOD test",
      not sanity.run_gate(RuleStub(good_win), samples, tight_ood, ood=det, embed=embed).tests["ood"].passed)

# ── report ────────────────────────────────────────────────────────────────────
section("report")

with tempfile.TemporaryDirectory() as tmp:
    p = sanity.write_report(bad, Path(tmp) / "eval" / "sanity_cf_wp_vtest.md")
    text = p.read_text(encoding="utf-8")
    check("report written", p.is_file() and text.startswith("# Counterfactual sanity gate"))
    check("report states the verdict and failed tests", "**FAIL**" in text and "unkill" in text and "failed tests" in text)
    check("report lists worst cases per test", text.count("worst ") >= 6)
    pg = sanity.write_report(rep, Path(tmp) / "good.md")
    check("passing report says PASS", "**PASS**" in pg.read_text(encoding="utf-8"))

# ── what the gate refuses to certify ──────────────────────────────────────────
section("missing inputs and errors")

no_ood = sanity.run_gate(RuleStub(good_win), samples, cfg)
check("without an OOD detector the gate does not pass", not no_ood.passed and not no_ood.tests["ood"].passed
      and "not run" in no_ood.tests["ood"].note)
only_kills = [s for s in samples if s[0] == "kill"]
ok_kills = sanity.run_gate(RuleStub(good_win), only_kills, cfg, ood=det, embed=embed)
check("missing sample kinds fail the gate", not ok_kills.passed and ok_kills.tests["unkill"].passed
      and "no samples" in ok_kills.tests["unplant"].note)
s_pre, s_post, kill = make_kill_pair()
broken_sample = ("kill", s_post, s_post, kill)          # victim alive in "pre": undo precondition fails
with_err = sanity.run_gate(RuleStub(good_win), samples + [broken_sample], cfg, ood=det, embed=embed)
u = with_err.tests["unkill"]
check("an InterventionError is counted and listed", u.n_errors == 1 and len(u.errors) == 1)
check("the error counts in the rate denominator", abs(u.metrics["toward_rate"] - 30 / 31) < 1e-9)
hist = sanity.run_gate(RuleStub(good_win), [(*smp, []) for smp in samples], cfg, ood=det, embed=embed)
check("5-tuples with a history are accepted", hist.passed)
try:
    sanity.run_gate(RuleStub(good_win), [("nonsense", s_pre, s_post, kill)], cfg)
    check("unknown kind -> ValueError", False)
except ValueError:
    check("unknown kind -> ValueError", True)
tick = itertools.count()
noisy = sanity.run_gate(RuleStub(lambda s, h: good_win(s) + 1e-9 * next(tick)), samples, cfg, ood=det, embed=embed)
check("order test catches a W that differs between the two orders",
      not noisy.tests["order"].passed and noisy.tests["order"].metrics["identical_rate"] == 0.0)

# ── CLI ───────────────────────────────────────────────────────────────────────
section("main()")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    code = sanity.main([])
check("main() exits 2 without a model and states", code == 2, buf.getvalue().splitlines()[0][:80])
with contextlib.redirect_stdout(io.StringIO()):
    code2 = sanity.main(["--model", "nope.ckpt", "--states", "nope.parquet"])
check("main() exits 2 for missing paths", code2 == 2)

sys.exit(summary())
