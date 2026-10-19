"""
Training loop (P6.3.1-P6.3.4): 20 steps give finite, decreasing loss; the model
directory has every file; a checkpoint saved and reloaded reproduces outputs within
1e-6; an interrupted run resumed from last.pt matches the uninterrupted run; the pro
fine-tune starts from best.pt, not last.pt; loss masking, terminal weight and the
warm-up / cosine schedule behave.

    .venv\\Scripts\\python.exe test\\engine\\test_train_loop.py [--cpu]
"""
import csv
import tempfile
from pathlib import Path

import numpy as np

from _harness import check, require, section, summary
torch = require("torch", "the training loop needs the platform torch build (docs/ENGINE_ENV.md)")   # LOW-61
from _fake_states import fake_arrays, finish, test_device  # noqa: E402

from engine import config, manifest  # noqa: E402
from engine.data import dataset  # noqa: E402
from engine.model import checkpoints, train  # noqa: E402

DEV = test_device()
print(f"device: {DEV}")
cfg = config.load()
_, vocab, tr, norm = fake_arrays(8, seed=41, n_snapshots=10)
va = dataset.tensorize(__import__("_fake_states").make_corpus(4, seed=42, n_snapshots=10, prefix="val"), vocab)
KW = dict(vocab=vocab, norm=norm, device=DEV, batch_size=16, warmup_steps=5, lr=2e-3, quiet=True)
# Resume equality tolerance. Not bit-exact: tiny gradient differences (parallel
# reductions; atomics in the GPU embedding backward) can flip Adam's update sign for
# parameters whose gradient is ~0, which moves them by up to ~lr per step.
TOL_LOSS, TOL_PARAM = 1e-3, 1e-3

with tempfile.TemporaryDirectory() as d:
    d = Path(d)

    # ── 20 steps ──
    section("20 steps")
    run = d / "wp_v1"
    s = train.fit(cfg, tr, va, run, max_epochs=2, steps_per_epoch=10, finetune_epochs=1,
                  eval_dir=d / "eval", **KW)
    L = np.array(s["step_losses"])
    check("20 steps ran", len(L) == 20, str(len(L)))
    check("loss finite", bool(np.isfinite(L).all()))
    check("loss decreasing (mean of last 5 < mean of first 5)", L[-5:].mean() < L[:5].mean(),
          f"{L[:5].mean():.3f} -> {L[-5:].mean():.3f}")
    for name in ("best.pt", "last.pt", "final.pt", "ft_last.pt", "manifest.json", "curves.csv",
                 "config.yaml", "vocab.json", "norm.json"):
        check(f"{name} written", (run / name).is_file())
    check("eval/wp_v1_train.md written", (d / "eval" / "wp_v1_train.md").is_file())
    m = manifest.read(run)
    check("manifest kind train_wp with n_params and seed", m["kind"] == "train_wp"
          and m["extra"]["n_params"] == s["n_params"] and m["extra"]["seed"] == cfg.train.seed)
    rows = list(csv.DictReader(open(run / "curves.csv", encoding="utf-8")))
    check("curves: 2 main epochs + 1 fine-tune epoch", [r["stage"] for r in rows] == ["main", "main", "finetune"])
    check("fine-tune started from best.pt", s["finetune"]["init_from"] == "best.pt")

    # ── checkpoint round trip ──
    section("checkpoint save / reload")
    m1 = checkpoints.build_model(checkpoints.load(run / "last.pt", DEV), DEV).eval()
    o1 = train.predict_logits(m1, va, DEV)
    checkpoints.save(d / "copy.pt", m1)
    m2 = checkpoints.build_model(checkpoints.load(d / "copy.pt", DEV), DEV).eval()
    o2 = train.predict_logits(m2, va, DEV)
    fin = {k: np.isfinite(o1[k]) for k in o1}
    worst = max(float(np.abs(o1[k][fin[k]] - o2[k][fin[k]]).max()) for k in o1)
    check("same -inf (dead) pattern after reload", all(np.array_equal(fin[k], np.isfinite(o2[k])) for k in o1))
    check("reloaded checkpoint reproduces outputs within 1e-6", worst <= 1e-6, f"max |diff| {worst:.2e}")
    check("norm buffers travel with the checkpoint", torch.equal(m1.tokens.player_num.mean.cpu(),
                                                                 m2.tokens.player_num.mean.cpu()))
    model, v2, n2, cal, path = checkpoints.load_model_dir(run, DEV)
    check("load_model_dir prefers final.pt", path.name == "final.pt" and cal is None and v2.tables == vocab.tables)

    # ── evaluate + calibration on the trained model ──
    section("evaluate() and calibration")
    res = train.evaluate(model, va, DEV, cfg=cfg, vocab=vocab)
    want = {"n", "logloss", "brier", "auc", "ece", "reliability", "per_phase", "per_map", "per_alive_diff",
            "bin_violations", "hazard_auc", "terminal_pin", "martingale"}
    check("evaluate() metrics dict complete", want <= set(res), str(sorted(want - set(res))))
    check("per-phase covers every labelled snapshot", sum(m["n"] for m in res["per_phase"].values()) == res["n"])
    check("martingale rows for k in 1, 5, 10", {r["k"] for r in res["martingale"]["rows"]} == {1, 5, 10})
    # LOW-79: the ECE target (R6: <= 0.02) is actually judged
    ok_r, bad_r = train.target_failures({"ece": 0.019}, cfg), train.target_failures({"ece": 0.021}, cfg)
    check("target_failures: ECE 0.019 passes, 0.021 fails", ok_r == [] and len(bad_r) == 1
          and bad_r[0].startswith("ECE 0.0210"), f"{ok_r} / {bad_r}")
    md = train.eval_markdown("t", res)
    check("eval markdown has the breakdowns", "By phase" in md and "By map" in md and "Martingale" in md)
    from engine.model import calibrate
    lv = train.predict_logits(model, va, DEV)["win"]
    cal = calibrate.fit_calibration(lv, va["y_win"], calibrate.map_versions_from_arrays(va, vocab),
                                    calibrate.phases_from_arrays(va), min_group_n=50)
    calibrate.save_calibration(run, cal)
    res_c = train.evaluate(model, va, DEV, cfg=cfg, vocab=vocab, calibration=cal)
    check("calibration lowers (or keeps) val log loss", res_c["logloss"] <= res["logloss"] + 1e-9,
          f"{res['logloss']:.4f} -> {res_c['logloss']:.4f} (T={cal['T']:.3f}, {cal['mode']})")
    check("load_model_dir picks up calibration.json", checkpoints.load_model_dir(run, DEV)[3]["T"] == cal["T"])

    # ── resume ──
    section("resume from last.pt")
    rk = dict(max_epochs=2, steps_per_epoch=5, finetune=False, eval_dir=d / "eval", **KW)
    sx = train.fit(cfg, tr, va, d / "x", **rk)
    sy1 = train.fit(cfg, tr, va, d / "y", interrupt_after_epochs=1, **rk)
    check("interrupted run stopped after 1 epoch", sy1.get("interrupted") and sy1["epochs"] == 1)
    sy2 = train.fit(cfg, tr, va, d / "y", resume=True, **rk)
    check("resumed run continued to epoch 2 with the same step count", sy2["epochs"] == 2 and sy2["steps"] == sx["steps"])
    check("resumed run trained only the missing epoch", len(sy2["step_losses"]) == 5)
    check("resumed step losses == uninterrupted epoch-2 losses",
          np.allclose(sy2["step_losses"], sx["step_losses"][5:], atol=TOL_LOSS),
          f"{np.max(np.abs(np.array(sy2['step_losses']) - sx['step_losses'][5:])):.2e}")
    px = checkpoints.load(d / "x" / "last.pt")["model"]
    py = checkpoints.load(d / "y" / "last.pt")["model"]
    pd = max(float((px[k].float() - py[k].float()).abs().max()) for k in px)
    check("resumed parameters == uninterrupted parameters (within 1e-3)", pd <= TOL_PARAM, f"max |diff| {pd:.2e}")
    ry = list(csv.DictReader(open(d / "y" / "curves.csv", encoding="utf-8")))
    check("resumed curves.csv has both epochs", [int(r["epoch"]) for r in ry] == [0, 1])
    keep0 = train.STEP_LOSS_KEEP
    train.STEP_LOSS_KEEP = 3                                  # LOW-36: only the last N step losses kept
    try:
        sc = train.fit(cfg, tr, va, d / "cap", **rk)
    finally:
        train.STEP_LOSS_KEEP = keep0
    check("step losses capped at STEP_LOSS_KEEP, the last ones kept (LOW-36)",
          sc["steps"] == 10 and len(sc["step_losses"]) == 3
          and np.allclose(sc["step_losses"], sx["step_losses"][-3:], atol=TOL_LOSS), str(len(sc["step_losses"])))

    # ── fine-tune loads best, not last ──
    section("pro fine-tune loads best.pt, not last.pt")
    last = checkpoints.load(d / "x" / "last.pt")
    perturbed = checkpoints.build_model(last)
    with torch.no_grad():
        for p in perturbed.parameters():
            p.add_(0.01)
    checkpoints.save(d / "x" / "last.pt", perturbed, state=last["state"])
    h_best = checkpoints.state_hash(checkpoints.build_model(checkpoints.load(d / "x" / "best.pt")).state_dict())
    h_last = checkpoints.state_hash(perturbed.state_dict())
    pro_tr, pro_va = train.only_tier(tr, "pro"), train.only_tier(va, "pro")
    info = train.finetune_stage(cfg, d / "x", pro_tr, pro_va, device=DEV, epochs=1, batch_size=16)
    check("best.pt and last.pt differ for this check", h_best != h_last)
    check("fine-tune initial weights == best.pt", info["init_hash"] == h_best)
    check("fine-tune initial weights != last.pt", info["init_hash"] != h_last)
    check("final.pt written by the fine-tune", (d / "x" / "final.pt").is_file())

# ── loss and schedule ──
section("loss masking, terminal weight, schedule")
lw = cfg.model.loss_weights
out = {"win": torch.tensor([0.3, -1.0, 2.0])}
b = {"y_win": torch.tensor([1.0, float("nan"), 0.0]), "is_terminal": torch.tensor([0, 0, 0])}
b2 = {"y_win": torch.tensor([1.0, 0.0]), "is_terminal": torch.tensor([0, 0])}
l_nan, _ = train.compute_loss(out, b, lw, 2.0)
l_drop, _ = train.compute_loss({"win": torch.tensor([0.3, 2.0])}, b2, lw, 2.0)
check("NaN labels are excluded (not treated as 0)", abs(float(l_nan) - float(l_drop)) < 1e-6)
bt = {"y_win": torch.tensor([1.0, 0.0]), "is_terminal": torch.tensor([1, 0])}
l_t, _ = train.compute_loss({"win": torch.tensor([0.3, 2.0])}, bt, lw, 2.0)
import torch.nn.functional as F  # noqa: E402
e = F.binary_cross_entropy_with_logits(torch.tensor([0.3, 2.0]), torch.tensor([1.0, 0.0]), reduction="none")
check("terminal snapshots weighted x terminal_weight", abs(float(l_t) - float((2 * e[0] + e[1]) / 3)) < 1e-6)
check("warm-up is linear", abs(train.lr_factor(4, 10, 100) - 0.5) < 1e-9 and train.lr_factor(9, 10, 100) == 1.0)
check("cosine decays to 0 at the end", abs(train.lr_factor(100, 10, 100)) < 1e-9
      and abs(train.lr_factor(55, 10, 100) - 0.5) < 1e-9)

finish(summary())
