"""
Training loop (P6.3): AdamW + cosine warm-up, tier oversampling, pro fine-tune stage,
augmentation, checkpoints with manifests, early stopping on val win log loss.
Evaluation helpers used by calibrate.py and the CLI also live here.

    from engine.model import train
    summary = train.fit(cfg, train_arrs, val_arrs, out_dir, vocab=vocab, norm=norm)
    res     = train.evaluate(model, test_arrs, device, cfg=cfg, calibration=cal, vocab=vocab)

One training sample is one round-window: a target snapshot plus the snapshots of the
same round in the `temporal_window_s` seconds before it (batching.WindowIndex). A batch
is `train.batch_rounds` windows; the unique snapshots they touch are encoded once.
Loss (P6.3.1): `w.win*BCE(win) + w.hazard*BCE(hazard, alive and labelled only) +
w.plant*BCE(plant15) + w.defuse*BCE(defuse15) + w.tte*Huber(tte on log1p seconds)`,
each a per-sample weighted mean with terminal snapshots weighted `terminal_weight`.
Unlabelled entries (NaN, e.g. an event that already happened) are excluded, never 0.

Stages: main (early stopping on val pro win log loss, patience) then, when the train set
has pro snapshots, a pro-only fine-tune that starts from best.pt (never last.pt) at
`pro_finetune_lr`; final.pt is whichever of best.pt and the fine-tune epochs has the
lowest val pro log loss. Resume (`resume=True`) continues from last.pt at the next
epoch; batches and augmentations are seeded by (seed, stage, epoch/step), so a resumed
run sees exactly the batches an uninterrupted run would have.

Ablations (P6.7): temporal / pairwise / aux_heads flags, `tiers=("pro",)`.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from engine import config as engine_config
from engine import manifest, paths
from engine.data import dataset
from engine.data.dataset import CTX_LAYOUT, NormStats, Vocab
from engine.model import augment as aug
from engine.model import batching, calibrate, checkpoints
from engine.model.wp_model import WPModel

CURVE_FIELDS = ("stage", "epoch", "step", "lr", "train_loss", "train_win", "train_hazard",
                "train_plant", "train_defuse", "train_tte", "val_logloss", "val_brier", "val_auc",
                "val_ece", "seconds")
PARTS = ("win", "hazard", "plant", "defuse", "tte")


# ── small helpers ─────────────────────────────────────────────────────────────

def pick_device(force_cpu: bool = False) -> str:
    return "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def use_bf16(device: str) -> bool:
    return device == "cuda" and torch.cuda.is_bf16_supported()


def only_tier(arrays: dict, *tiers: str) -> dict:
    if not tiers:
        return arrays
    keep = np.isin(arrays["tier"], list(tiers))
    return dataset.take(arrays, np.flatnonzero(keep))


def labelled(arrays: dict) -> np.ndarray:
    return np.flatnonzero(~np.isnan(arrays["y_win"]))


def aug_seed(seed: int, stage: int, step: int) -> int:
    return (seed * 1_000_003 + stage * 7_919 + step) % (2**63 - 1)


# ── loss ──────────────────────────────────────────────────────────────────────

def _wmean(loss: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    wt = weight * mask.to(weight.dtype)
    return (torch.where(mask, loss, torch.zeros_like(loss)) * wt).sum() / wt.sum().clamp_min(1e-8)


def compute_loss(out: dict, b: dict, loss_weights, terminal_weight: float) -> tuple[torch.Tensor, dict]:
    """Total loss and its parts (floats) for the targets of a batch."""
    w = 1.0 + (terminal_weight - 1.0) * b["is_terminal"].float()
    parts: dict[str, torch.Tensor] = {}
    y = b["y_win"].float()
    m = ~torch.isnan(y)
    parts["win"] = _wmean(F.binary_cross_entropy_with_logits(out["win"].float(), torch.nan_to_num(y),
                                                             reduction="none"), m, w)
    if "hazard_raw" in out:
        yh = b["y_hazard"].float()
        mh = ~torch.isnan(yh) & out["alive"]
        lh = F.binary_cross_entropy_with_logits(out["hazard_raw"].float(), torch.nan_to_num(yh), reduction="none")
        parts["hazard"] = _wmean(lh, mh, w[:, None].expand_as(lh))
        for name, key, head in (("plant", "y_plant15", "plant15"), ("defuse", "y_defuse15", "defuse15")):
            yy = b[key].float()
            mm = ~torch.isnan(yy)
            parts[name] = _wmean(F.binary_cross_entropy_with_logits(out[head].float(), torch.nan_to_num(yy),
                                                                    reduction="none"), mm, w)
        yt = b["y_tte"].float()
        mt = ~torch.isnan(yt)
        lt = F.huber_loss(out["tte"].float(), torch.log1p(torch.nan_to_num(yt).clamp_min(0)), reduction="none")
        parts["tte"] = _wmean(lt, mt, w)
    total = sum(getattr(loss_weights, k) * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warm-up to 1 over `warmup` steps, then cosine decay to 0 at `total`."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def make_optimizer(model: WPModel, lr: float, wd: float) -> torch.optim.Optimizer:
    decay    = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


# ── inference over arrays ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_logits(model: WPModel, arrays: dict, device, chunk: int = 512) -> dict[str, np.ndarray]:
    """Logits for every snapshot (each with its own causal window), fp32, eval mode.
    hazard is -inf for dead/absent players; heads the model lacks are absent."""
    was = model.training
    model.eval()
    n  = len(arrays["t_s"])
    wi = batching.WindowIndex(arrays["round_key"], arrays["t_s"], model.window_s)
    out: dict[str, np.ndarray] = {}
    for a in range(0, n, chunk):
        targets = wi.order[a:a + chunk]                   # consecutive in (round, time): windows overlap
        b = batching.to_device(batching.gather(arrays, [wi.window(i) for i in targets], with_labels=False), device)
        o = model(b)
        for k, v in o.items():
            if k in ("hazard_raw", "alive"):
                continue
            if k not in out:
                out[k] = np.zeros((n,) + tuple(v.shape[1:]), np.float32)
            out[k][targets] = v.float().cpu().numpy()
    model.train(was)
    return out


def _group_metrics(p, y, groups, n_bins) -> dict:
    return {str(g): calibrate.metrics(p[groups == g], y[groups == g], n_bins) for g in np.unique(groups)}


def evaluate(model: WPModel, arrays: dict, device, *, cfg=None, calibration: dict | None = None,
             vocab: Vocab | None = None, logits: dict | None = None) -> dict:
    """logloss, brier, auc, ece, reliability bins, per-phase / per-map / per-alive-diff
    breakdowns, hazard AUC, terminal pin (P6.5.3) and martingale drift (P6.6.2)."""
    cfg = cfg if cfg is not None else engine_config.load()
    c   = cfg.calib
    lg  = logits if logits is not None else predict_logits(model, arrays, device)
    ph  = calibrate.phases_from_arrays(arrays)
    mv  = calibrate.map_versions_from_arrays(arrays, vocab) if vocab is not None else None
    p   = calibrate.calibrated_probs(lg["win"], calibration, mv, ph)
    y   = arrays["y_win"].astype(np.float64)
    ok  = ~np.isnan(y)
    res = calibrate.metrics(p[ok], y[ok], c.n_bins)
    res["per_phase"] = _group_metrics(p[ok], y[ok], ph[ok], c.n_bins)
    if mv is not None:
        res["per_map"] = _group_metrics(p[ok], y[ok], mv[ok], c.n_bins)
    cn = arrays["ctx_num"]
    diff = (cn[:, CTX_LAYOUT.col("alive_ct")] - cn[:, CTX_LAYOUT.col("alive_t")]).astype(np.int64)
    res["per_alive_diff"] = _group_metrics(p[ok], y[ok], diff[ok], c.n_bins)
    res["bin_violations"] = calibrate.bin_violations(res["reliability"], c.bin_tol, c.min_bin_n)
    if "hazard" in lg:
        alive = np.isfinite(lg["hazard"])
        yh = arrays["y_hazard"]
        mh = alive & ~np.isnan(yh)
        res["hazard_auc"] = calibrate.auc(calibrate.sigmoid(lg["hazard"][mh]), yh[mh]) if mh.any() else None
    else:
        res["hazard_auc"] = None
    res["terminal_pin"] = calibrate.terminal_pin(p, y, arrays["is_terminal"], arrays["round_key"])
    res["martingale"]   = calibrate.martingale_check(p, arrays["round_key"], arrays["t_s"],
                                                     tuple(c.martingale_horizons_s), c.n_bins,
                                                     c.martingale_min_n, c.martingale_tol)
    return res


def target_failures(res: dict, cfg) -> list[str]:
    """P6.5.2 / P6.5.3 / P6.6.2 targets missed by an evaluate() result."""
    c, bad = cfg.calib, []
    if res.get("ece") is not None and res["ece"] > c.ece_target:
        bad.append(f"ECE {res['ece']:.4f} > {c.ece_target}")
    for r in res.get("bin_violations", []):
        bad.append(f"bin {r['lo']:.2f}-{r['hi']:.2f} (n={r['n']}): |{r['mean_pred']:.3f} - {r['observed']:.3f}| > {c.bin_tol}")
    tp = res.get("terminal_pin", {})
    if tp.get("ok") is False:
        bad.append(f"terminal pin rate {tp['rate']:.4f} < 0.99")
    mg = res.get("martingale", {})
    for r in mg.get("violations", []):
        bad.append(f"martingale k={r['k']} bin {r['lo']:.2f}: drift {r['mean_drift']:+.4f} (n={r['n']})")
    return bad


def eval_markdown(title: str, res: dict, cal: dict | None = None) -> str:
    def f(v):
        return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    lines = [f"# {title}", ""]
    if cal is not None:
        lines += [f"Calibration: T={cal['T']:.4f}, mode={cal['mode']}, val ECE single="
                  f"{f(cal.get('ece_val_single'))}, group={f(cal.get('ece_val_group'))}", ""]
    lines.append(calibrate.metrics_md("All", res))
    lines.append(f"hazard AUC: {f(res.get('hazard_auc'))}  terminal pin: {res.get('terminal_pin')}\n")
    for key, name in (("per_phase", "phase"), ("per_map", "map"), ("per_alive_diff", "alive_ct - alive_t")):
        if key in res:
            lines += [f"## By {name}", "", "| group | n | logloss | brier | auc | ece |", "|---|---:|---:|---:|---:|---:|"]
            for g, m in res[key].items():
                lines.append(f"| {g} | {m['n']} | {f(m['logloss'])} | {f(m['brier'])} | {f(m['auc'])} | {f(m['ece'])} |")
            lines.append("")
    mg = res.get("martingale")
    if mg:
        lines += ["## Martingale drift (judged bins)", "", "| k | bin | n | mean drift |", "|---:|---|---:|---:|"]
        for r in mg["rows"]:
            if r["judged"]:
                lines.append(f"| {r['k']} | {r['lo']:.2f}-{r['hi']:.2f} | {r['n']} | {r['mean_drift']:+.4f} |")
        lines.append(f"\nok: {mg['ok']} (tol {mg['tol']}, min n {mg['min_n']})\n")
    return "\n".join(lines)


# ── training ──────────────────────────────────────────────────────────────────

class _Logger:
    def __init__(self, out_dir: Path, quiet: bool):
        from engine import log
        self.lg = log.get_logger(kind="train_wp", log_dir=out_dir, console=not quiet)

    def __call__(self, msg: str, **fields) -> None:
        self.lg.info(msg, **fields)

    def close(self) -> None:
        self.lg.close()


def _val_metrics(model, val: dict, device, n_bins: int) -> dict:
    lg = predict_logits(model, val, device)
    y  = val["y_win"].astype(np.float64)
    ok = ~np.isnan(y)
    return calibrate.metrics(calibrate.sigmoid(lg["win"][ok]), y[ok], n_bins)


def _append_curve(path: Path, row: dict) -> None:
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CURVE_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in CURVE_FIELDS})


def _train_epoch(model, opt, sched, arrays, wi, sampler, cfg, *, device, epoch, stage, state,
                 n_batches, augment_on, max_steps) -> dict:
    model.train()
    bf16 = use_bf16(device)
    gen  = torch.Generator()
    sums = {k: 0.0 for k in ("loss", *PARTS)}
    n    = 0
    for targets in sampler.epoch(epoch, n_batches, stage=stage):
        if max_steps is not None and state["step"] >= max_steps:
            break
        nb = batching.gather(arrays, [wi.window(i) for i in targets])
        b  = batching.to_device(nb, device)
        gen.manual_seed(aug_seed(cfg.train.seed, stage, state["step"]))
        b  = aug.augment(b, cfg.augment, gen, training=augment_on and model.training)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model(b)
        loss, parts = compute_loss(out, b, cfg.model.loss_weights, cfg.model.terminal_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        state["step"] += 1
        sums["loss"] += float(loss.detach())
        for k, v in parts.items():
            sums[k] += v
        n += 1
        state.setdefault("step_losses", []).append(float(loss.detach()))
    return {k: (v / n if n else None) for k, v in sums.items()} | {"n_steps": n}


def fit(cfg, train: dict, val: dict, out_dir: Path, *, vocab: Vocab, norm: NormStats,
        temporal: bool = True, pairwise: bool = True, aux_heads: bool = True,
        tiers: tuple[str, ...] | None = None, device: str | None = None,
        max_epochs: int | None = None, batch_size: int | None = None,
        steps_per_epoch: int | None = None, warmup_steps: int | None = None, lr: float | None = None,
        augment_on: bool = True, finetune: bool = True, finetune_epochs: int | None = None,
        resume: bool = False, max_steps: int | None = None, eval_dir: Path | None = None,
        quiet: bool = False, interrupt_after_epochs: int | None = None) -> dict:
    """Train a WPModel into `out_dir` (see module docstring). Returns a summary dict.

    `max_steps` ends the main stage early (tests); `interrupt_after_epochs` returns
    right after that many epochs of this call as if the process had died (no
    fine-tune, nothing marked done), so tests can exercise `resume=True`."""
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tc      = cfg.train
    device  = device or pick_device()
    max_epochs   = tc.max_epochs if max_epochs is None else max_epochs
    batch_size   = tc.batch_rounds if batch_size is None else batch_size
    warmup_steps = tc.warmup_steps if warmup_steps is None else warmup_steps
    lr           = tc.lr if lr is None else lr
    ft_epochs    = tc.pro_finetune_epochs if finetune_epochs is None else finetune_epochs
    if tiers:
        train, val = only_tier(train, *tiers), only_tier(val, *tiers)
    val_pro = only_tier(val, "pro") if (val["tier"] == "pro").any() else val
    for what, arr in (("train", train), ("val", val_pro)):
        if dataset.n_labelled(arr) == 0:                   # fail now, not after the first epoch (MED-19)
            raise ValueError(f"{what} split has no labelled rows" + (f" for tiers {tiers}" if tiers else "")
                             + "; check the split name and the tier filter")
    log = _Logger(out_dir, quiet)

    set_seed(tc.seed)
    model = WPModel(cfg, vocab.sizes(), temporal=temporal, pairwise=pairwise, aux_heads=aux_heads)
    model.set_norm(norm)
    model.to(device)
    checkpoints.write_model_files(out_dir, vocab, norm, cfg)

    cand    = labelled(train)
    wi      = batching.WindowIndex(train["round_key"], train["t_s"], model.window_s)
    sampler = batching.TierSampler(train["tier"], batch_size, tc.pro_oversample, tc.seed, candidates=cand)
    n_batches = steps_per_epoch or sampler.n_batches()
    total     = max_epochs * n_batches
    opt   = make_optimizer(model, lr, tc.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warmup_steps, total))
    state = {"epoch": 0, "step": 0, "best_val": None, "best_epoch": None, "bad_epochs": 0,
             "main_done": False, "seed": tc.seed}
    curves = out_dir / "curves.csv"
    last_p, best_p = out_dir / "last.pt", out_dir / "best.pt"
    if resume and last_p.is_file():
        ck = checkpoints.load(last_p, device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        state.update(ck["state"])
        log("resumed", epoch=state["epoch"], step=state["step"])
    log("train start", device=device, n_params=model.n_params, n_train=len(cand), n_val=len(val["t_s"]),
        pro_share=round(sampler.natural_share, 4), target_pro_share=round(sampler.target_share, 4),
        batches_per_epoch=n_batches, flags=model.flags)

    # ── main stage ──
    epochs_this_call = 0
    while not state["main_done"] and state["epoch"] < max_epochs:
        t0 = time.perf_counter()
        ep = state["epoch"]
        tr = _train_epoch(model, opt, sched, train, wi, sampler, cfg, device=device, epoch=ep, stage=0,
                          state=state, n_batches=n_batches, augment_on=augment_on, max_steps=max_steps)
        vm = _val_metrics(model, val_pro, device, cfg.calib.n_bins)
        state["epoch"] = ep + 1
        improved = vm["logloss"] is not None and (state["best_val"] is None or vm["logloss"] < state["best_val"])
        if improved:
            state["best_val"], state["best_epoch"], state["bad_epochs"] = vm["logloss"], ep, 0
        else:
            state["bad_epochs"] += 1
        # done = early stopping (or the test step cap); reaching max_epochs ends the loop
        # without marking done, so a resume with a larger max_epochs can continue
        state["main_done"] = bool(state["bad_epochs"] >= tc.patience
                                  or (max_steps is not None and state["step"] >= max_steps))
        ck_state = {k: v for k, v in state.items() if k != "step_losses"}
        checkpoints.save(last_p, model, optimizer=opt, scheduler=sched, state=ck_state)
        if improved:
            checkpoints.save(best_p, model, state=ck_state)
        row = {"stage": "main", "epoch": ep, "step": state["step"], "lr": opt.param_groups[0]["lr"],
               "train_loss": tr["loss"], **{f"train_{k}": tr.get(k) for k in PARTS},
               "val_logloss": vm["logloss"], "val_brier": vm["brier"], "val_auc": vm["auc"],
               "val_ece": vm["ece"], "seconds": round(time.perf_counter() - t0, 3)}
        _append_curve(curves, row)
        _write_manifest(out_dir, cfg, model, state, device, len(cand), t_start)
        log("epoch", **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        epochs_this_call += 1
        if interrupt_after_epochs is not None and epochs_this_call >= interrupt_after_epochs:
            log("interrupted (simulated)", epoch=state["epoch"])
            log.close()
            return {"interrupted": True, "epochs": state["epoch"], "steps": state["step"],
                    "step_losses": state.get("step_losses", [])}

    # ── pro fine-tune stage (P6.3.2): starts from best.pt, never last.pt ──
    ft_info = None
    pro_train = only_tier(train, "pro")
    if finetune and ft_epochs > 0 and len(labelled(pro_train)) and best_p.is_file():
        ft_info = finetune_stage(cfg, out_dir, pro_train, val_pro, device=device, epochs=ft_epochs,
                                 batch_size=batch_size, augment_on=augment_on, log=log,
                                 step0=state["step"])
        model = checkpoints.build_model(checkpoints.load(out_dir / "final.pt", device), device)
    elif best_p.is_file():
        checkpoints.save(out_dir / "final.pt", checkpoints.build_model(checkpoints.load(best_p)),
                         state={**checkpoints.load(best_p)["state"], "final_from": "best.pt"})

    summary = {"out_dir": str(out_dir), "n_params": model.n_params, "best_val": state["best_val"],
               "best_epoch": state["best_epoch"], "epochs": state["epoch"], "steps": state["step"],
               "finetune": ft_info, "device": device, "flags": model.flags,
               "step_losses": state.get("step_losses", []), "seconds": time.perf_counter() - t_start}
    _write_manifest(out_dir, cfg, model, state, device, len(cand), t_start, ft_info)
    _write_train_md(out_dir, eval_dir, summary)
    log("train done", best_val=state["best_val"], best_epoch=state["best_epoch"], seconds=round(summary["seconds"], 1))
    log.close()
    return summary


def finetune_stage(cfg, out_dir: Path, pro_train: dict, val_pro: dict, *, device: str, epochs: int,
                   batch_size: int, augment_on: bool = True, log=None, step0: int = 0) -> dict:
    """P6.3.2 pro-only fine-tune. Always starts from `<out_dir>/best.pt` (never
    last.pt), constant lr `train.pro_finetune_lr`, early-stopping style selection:
    final.pt = the lowest val pro log loss among best.pt and the fine-tune epochs.
    Writes ft_last.pt each epoch and appends `finetune` rows to curves.csv."""
    out_dir = Path(out_dir)
    log    = log or (lambda msg, **kw: None)
    best_p = out_dir / "best.pt"
    ck     = checkpoints.load(best_p, device)
    model  = checkpoints.build_model(ck, device)
    init_hash = checkpoints.state_hash(model.state_dict())
    tc   = cfg.train
    opt  = make_optimizer(model, tc.pro_finetune_lr, tc.wd)
    wi   = batching.WindowIndex(pro_train["round_key"], pro_train["t_s"], model.window_s)
    smp  = batching.TierSampler(pro_train["tier"], batch_size, 1.0, tc.seed, candidates=labelled(pro_train))
    best_val, best_from = ck["state"].get("best_val"), "best.pt"
    ft_state = {"step": step0}
    for e in range(epochs):
        t0 = time.perf_counter()
        tr = _train_epoch(model, opt, None, pro_train, wi, smp, cfg, device=device, epoch=e, stage=1,
                          state=ft_state, n_batches=smp.n_batches(), augment_on=augment_on, max_steps=None)
        vm = _val_metrics(model, val_pro, device, cfg.calib.n_bins)
        checkpoints.save(out_dir / "ft_last.pt", model, state={"ft_epoch": e, "init_from": "best.pt"})
        if vm["logloss"] is not None and (best_val is None or vm["logloss"] < best_val):
            best_val, best_from = vm["logloss"], f"finetune epoch {e}"
            checkpoints.save(out_dir / "final.pt", model,
                             state={"final_from": best_from, "best_val": best_val, "init_from": "best.pt"})
        row = {"stage": "finetune", "epoch": e, "step": ft_state["step"], "lr": tc.pro_finetune_lr,
               "train_loss": tr["loss"], **{f"train_{k}": tr.get(k) for k in PARTS},
               "val_logloss": vm["logloss"], "val_brier": vm["brier"], "val_auc": vm["auc"],
               "val_ece": vm["ece"], "seconds": round(time.perf_counter() - t0, 3)}
        _append_curve(out_dir / "curves.csv", row)
        log("finetune epoch", epoch=e, val_logloss=vm["logloss"])
    if best_from == "best.pt":
        checkpoints.save(out_dir / "final.pt", checkpoints.build_model(ck),
                         state={"final_from": "best.pt", "best_val": best_val, "init_from": "best.pt"})
    return {"init_from": "best.pt", "init_epoch": ck["state"].get("best_epoch"), "init_hash": init_hash,
            "epochs": epochs, "final_from": best_from, "best_val": best_val}


def _write_manifest(out_dir, cfg, model, state, device, n_train, t_start, ft_info=None) -> None:
    outs = [p.name for p in sorted(out_dir.iterdir()) if p.suffix in (".pt", ".json", ".csv", ".yaml")
            and p.name != manifest.MANIFEST_NAME]
    manifest.write(out_dir, "train_wp", outputs=outs, cfg=cfg, timings={"train": time.perf_counter() - t_start},
                   extra={"n_params": model.n_params, "flags": model.flags, "arch": model.arch,
                          "vocab_sizes": model.vocab_sizes, "layout_hash": dataset.layout_hash(),
                          "device": device, "n_train": n_train, "seed": cfg.train.seed,
                          "state": {k: v for k, v in state.items() if k != "step_losses"},
                          "finetune": ft_info})


def _write_train_md(out_dir: Path, eval_dir: Path | None, summary: dict) -> Path:
    eval_dir = Path(eval_dir) if eval_dir is not None else paths.EVAL_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(out_dir / "curves.csv", encoding="utf-8"))) if (out_dir / "curves.csv").is_file() else []
    lines = [f"# {out_dir.name} training", "",
             f"params {summary['n_params']:,}; device {summary['device']}; flags {summary['flags']}; "
             f"best val pro win logloss {summary['best_val']} at epoch {summary['best_epoch']}; "
             f"fine-tune {summary['finetune']}; {summary['seconds']:.1f} s", "",
             "| stage | epoch | step | lr | train loss | val logloss | val brier | val auc | val ece |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['stage']} | {r['epoch']} | {r['step']} | {float(r['lr']):.2e} | {r['train_loss']} | "
                     f"{r['val_logloss']} | {r['val_brier']} | {r['val_auc']} | {r['val_ece']} |")
    path = eval_dir / f"{out_dir.name}_train.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the win-probability model (P6.3).")
    ap.add_argument("--train", default="train", help="train split name or shard directory")
    ap.add_argument("--val", default="val", help="val split name or shard directory")
    ap.add_argument("--vocab", required=True, help="datasets/vocab_v<N>.json")
    ap.add_argument("--norm", required=True, help="datasets/norm_v<N>.json")
    ap.add_argument("--out", default=None, help="model directory (default models/wp_v<N+1>)")
    ap.add_argument("--no-temporal", action="store_true")
    ap.add_argument("--no-pairwise", action="store_true")
    ap.add_argument("--no-aux-heads", action="store_true")
    ap.add_argument("--tiers", nargs="*", default=None, help="e.g. --tiers pro (ablation b)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-finetune", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    cfg = engine_config.load()
    out = Path(a.out) if a.out else checkpoints.next_version_dir(paths.MODELS_DIR, "wp_v")
    try:
        dataset.check_artifacts(a.train, a.vocab, a.norm)
        tr, va = dataset.read_shards(a.train), dataset.read_shards(a.val)
        for what, arr in (("--train", tr), ("--val", va)):
            if dataset.n_labelled(arr, tuple(a.tiers) if a.tiers else ()) == 0:
                raise ValueError(f"{what} split has no labelled rows" + (f" for --tiers {a.tiers}" if a.tiers else ""))
    except (FileNotFoundError, ValueError) as exc:
        print(f"train-wp: {exc}", file=sys.stderr)
        return 2
    s = fit(cfg, tr, va, out,
            vocab=Vocab.load(a.vocab), norm=NormStats.load(a.norm),
            temporal=not a.no_temporal, pairwise=not a.no_pairwise, aux_heads=not a.no_aux_heads,
            tiers=tuple(a.tiers) if a.tiers else None, device=pick_device(a.cpu),
            max_epochs=a.max_epochs, batch_size=a.batch, steps_per_epoch=a.steps_per_epoch,
            augment_on=not a.no_augment, finetune=not a.no_finetune, resume=a.resume)
    print(json.dumps({k: v for k, v in s.items() if k != "step_losses"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
