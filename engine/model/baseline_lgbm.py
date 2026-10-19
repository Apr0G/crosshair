"""
LightGBM yardstick over flattened, permutation-invariant snapshot vectors (P6.1).
The numbers the NN must beat on pro held-out months.

    from engine.model import baseline_lgbm as bl
    X, names = bl.flatten(arrs)                       # from the same shards as the NN
    booster  = bl.train(train_arrs, val_arrs)         # early stopping on val pro (all val if no pro)
    p        = bl.predict(booster, test_arrs)
    res      = bl.evaluate(booster, test_arrs, cfg)   # logloss, brier, auc, ece, reliability, per phase/map
    bl.save(out_dir, booster, res, ...)               # models/baseline_v<N>/

Flattening (P6.1.1), from world-unit dataset arrays (missing = NaN, LightGBM's native
missing handling):
- every context value (live clock, bomb state, alive, equipment, money, score, ...);
- per team: alive, hp sum/mean, armor sum, equipment sum, grenade count, blind count,
  helmets, defusers, mean speed, spread (mean distance between alive teammates);
- per player, sorted by health within the team (descending; ties by armor, equipment,
  position, so the vector does not depend on slot order): alive, health, armor, equipment, weapon class, flash time,
  speed, x, y, z, seconds since last seen, spotted-by count, grenades, defuser, c4;
- pairwise: enemy pairs with LOS, min / mean enemy distance, enemy LOS pairs through
  smoke;
- tier, map_version, bomb_site and weapon classes as categorical features.
Not included: nav distance to the bomb (the schema has no bomb position).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from engine.data.dataset import CTX_LAYOUT, FEATURE_LAYOUT, PAIR_CHANNELS, PLAYER_LAYOUT
from engine.model import calibrate

CAT_PREFIXES = ("ctx_cat_", "weapon_")
_P = PLAYER_LAYOUT
_NADES = ("n_smoke", "n_flash", "n_he", "n_molly", "n_decoy")
PLAYER_COLS = ("alive", "health", "armor", "equipment_value", "flash_duration_remaining", "speed",
               "x", "y", "z", "seconds_since_last_seen_by_enemy", "spotted_by_count", "has_defuser", "has_c4")


def _values(arr: np.ndarray, layout, name: str) -> np.ndarray:
    """A value column with NaN where its presence bit is 0."""
    c = layout.col(name)
    v = arr[..., c].astype(np.float32)
    p = layout.presence_of[c]
    return np.where(arr[..., p] > 0.5, v, np.nan) if p >= 0 else v


def _nanmean(v: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    """Mean over `axis` ignoring NaN; NaN (not 0) when nothing is known."""
    ok  = ~np.isnan(v)
    cnt = ok.sum(axis=axis)
    return np.where(cnt > 0, np.where(ok, v, 0.0).sum(axis=axis) / np.maximum(cnt, 1), np.nan)


def flatten(arrays: dict) -> tuple[np.ndarray, list[str]]:
    pn, cn, pair = arrays["player_num"], arrays["ctx_num"], arrays["pair"]
    pmask = arrays["player_mask"]
    N = len(cn)
    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, v) -> None:
        names.append(name)
        cols.append(np.asarray(v, np.float32).reshape(N))

    for name in CTX_LAYOUT.names[:CTX_LAYOUT.n_values]:
        add(f"ctx_{name}", _values(cn, CTX_LAYOUT, name))
    for k, name in enumerate(FEATURE_LAYOUT["ctx_cat"]):
        add(f"ctx_cat_{name}", arrays["ctx_cat"][:, k])

    alive = (pn[..., _P.col("alive")] > 0.5) & pmask
    nades = sum(np.nan_to_num(_values(pn, _P, f)) for f in _NADES)
    wcol  = FEATURE_LAYOUT["player_cat"].index("weapon_class")
    pv    = {c: _values(pn, _P, c) for c in PLAYER_COLS}
    dist  = pair[..., PAIR_CHANNELS.index("dist_units")]
    for t, sl in (("ct", slice(0, 5)), ("t", slice(5, 10))):
        a = alive[:, sl]
        n_alive = a.sum(1)
        hp = np.where(a, pv["health"][:, sl], 0.0)
        add(f"{t}_alive", n_alive)
        add(f"{t}_hp_sum", hp.sum(1))
        add(f"{t}_hp_mean", np.where(n_alive > 0, hp.sum(1) / np.maximum(n_alive, 1), np.nan))
        add(f"{t}_armor_sum", np.where(a, pv["armor"][:, sl], 0.0).sum(1))
        # team sums: NaN when any alive player's value is unknown, never 0 for it (R7, LOW-31)
        add(f"{t}_equip_sum", np.where(a, pv["equipment_value"][:, sl], 0.0).sum(1))
        add(f"{t}_nades", np.where(a, nades[:, sl], 0.0).sum(1))
        add(f"{t}_blind", (a & (np.nan_to_num(pv["flash_duration_remaining"][:, sl]) > 0)).sum(1))
        add(f"{t}_helmets", np.where(a, _values(pn, _P, "has_helmet")[:, sl], 0.0).sum(1))
        add(f"{t}_defusers", np.where(a, pv["has_defuser"][:, sl], 0.0).sum(1))
        add(f"{t}_speed_mean", _nanmean(np.where(a, pv["speed"][:, sl], np.nan), (1,)))
        d  = dist[:, sl, sl]
        aa = a[:, :, None] & a[:, None, :] & ~np.eye(5, dtype=bool)[None]
        add(f"{t}_spread", _nanmean(np.where(aa, d, np.nan), (1, 2)))
        # players sorted by health within the team (dead -> -1, last); ties broken by
        # armor, equipment, then position, so the order never depends on slot order
        key = lambda c: -np.nan_to_num(pv[c][:, sl], nan=-1e9)   # noqa: E731
        order = np.lexsort((key("z"), key("y"), key("x"), key("equipment_value"), key("armor"),
                            -np.where(a, np.nan_to_num(pv["health"][:, sl]), -1.0)), axis=-1)
        rows = np.arange(N)[:, None]
        for c in PLAYER_COLS:
            v = pv[c][:, sl][rows, order]
            for k in range(5):
                add(f"{t}{k}_{c}", v[:, k])
        nd = nades[:, sl][rows, order]
        wc = arrays["player_cat"][:, sl, wcol][rows, order]
        for k in range(5):
            add(f"{t}{k}_nades", nd[:, k])
            add(f"weapon_{t}{k}", wc[:, k])

    los = pair[..., PAIR_CHANNELS.index("los")]
    smk = pair[..., PAIR_CHANNELS.index("smoke_between")]
    enemy = np.zeros((10, 10), bool)
    enemy[:5, 5:] = True
    ea = enemy[None] & alive[:, :, None] & alive[:, None, :]
    add("pair_enemy_los", (ea & (los == 1)).sum((1, 2)))
    add("pair_enemy_los_smoke", (ea & (los == 1) & (smk == 1)).sum((1, 2)))
    ed = np.where(ea, dist, np.nan)
    mn = np.where(np.isnan(ed), np.inf, ed).min(axis=(1, 2))
    add("pair_enemy_min_dist", np.where(np.isinf(mn), np.nan, mn))
    add("pair_enemy_mean_dist", _nanmean(ed, (1, 2)))
    return np.stack(cols, axis=1), names


def categorical(names: list[str]) -> list[str]:
    return [n for n in names if n.startswith(CAT_PREFIXES)]


def _labelled(arrays: dict) -> np.ndarray:
    return np.flatnonzero(~np.isnan(arrays["y_win"]))


def early_stop_rows(val_arrays: dict) -> np.ndarray:
    """Labelled val rows the booster early-stops on: the pro ones when any exist, as the
    NN selects on val pro (train.fit), else every labelled row (LOW-30, 2026-10-19)."""
    lab = _labelled(val_arrays)
    tier = val_arrays.get("tier")
    if tier is None:
        return lab
    pro = lab[np.asarray(tier)[lab] == "pro"]
    return pro if len(pro) else lab


def default_params(n_train: int, seed: int) -> dict:
    return {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
            "num_leaves": 63, "min_data_in_leaf": int(max(5, min(100, n_train // 50))),
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "seed": seed, "deterministic": True, "num_threads": 0, "verbose": -1}


def train(train_arrays: dict, val_arrays: dict, *, seed: int = 1337, num_boost_round: int = 2000,
          early_stopping: int = 50, params: dict | None = None):
    import lightgbm as lgb
    tr, va = _labelled(train_arrays), early_stop_rows(val_arrays)
    Xtr, names = flatten({k: v[tr] for k, v in train_arrays.items()})
    Xva, _     = flatten({k: v[va] for k, v in val_arrays.items()})
    cats = categorical(names)
    dtr = lgb.Dataset(Xtr, train_arrays["y_win"][tr], feature_name=names, categorical_feature=cats,
                      free_raw_data=False)
    dva = lgb.Dataset(Xva, val_arrays["y_win"][va], reference=dtr, feature_name=names,
                      categorical_feature=cats, free_raw_data=False)
    p = params or default_params(len(tr), seed)
    return lgb.train(p, dtr, num_boost_round=num_boost_round, valid_sets=[dva], valid_names=["val"],
                     callbacks=[lgb.early_stopping(early_stopping, verbose=False)])


def predict(booster, arrays: dict) -> np.ndarray:
    X, _ = flatten(arrays)
    return booster.predict(X, num_iteration=booster.best_iteration or None)


def evaluate(booster, arrays: dict, cfg=None, vocab=None) -> dict:
    """Same metric set as train.evaluate: logloss, brier, auc, ece, reliability, per
    phase, per map (with a vocab)."""
    n_bins = cfg.calib.n_bins if cfg is not None else 20
    p  = predict(booster, arrays)
    y  = arrays["y_win"].astype(np.float64)
    ok = ~np.isnan(y)
    res = calibrate.metrics(p[ok], y[ok], n_bins)
    ph  = calibrate.phases_from_arrays(arrays)[ok]
    res["per_phase"] = {g: calibrate.metrics(p[ok][ph == g], y[ok][ph == g], n_bins) for g in np.unique(ph)}
    if vocab is not None:
        mv = calibrate.map_versions_from_arrays(arrays, vocab)[ok]
        res["per_map"] = {g: calibrate.metrics(p[ok][mv == g], y[ok][mv == g], n_bins) for g in np.unique(mv)}
    res["best_iteration"] = int(booster.best_iteration or booster.current_iteration())
    return res


def save(out_dir: Path, booster, results: dict, *, cfg=None, extra: dict | None = None,
         eval_dir: Path | None = None) -> Path:
    from engine import manifest, paths
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.txt"))
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    manifest.write(out_dir, "baseline_lgbm", outputs=["model.txt", "metrics.json"], cfg=cfg,
                   extra={"best_iteration": results.get("best_iteration"), **(extra or {})})
    ev = Path(eval_dir) if eval_dir is not None else paths.EVAL_DIR
    ev.mkdir(parents=True, exist_ok=True)
    lines = [f"# {out_dir.name} (LightGBM baseline)", ""]
    for split, res in results.items():
        if isinstance(res, dict) and "logloss" in res:
            lines.append(calibrate.metrics_md(split, res))
            for key in ("per_phase", "per_map"):
                if key in res:
                    lines += [f"#### {split} by {key[4:]}", "", "| group | n | logloss | brier | auc | ece |",
                              "|---|---:|---:|---:|---:|---:|"]
                    for g, m in res[key].items():
                        f = lambda v: "n/a" if v is None else f"{v:.4f}"   # noqa: E731
                        lines.append(f"| {g} | {m['n']} | {f(m['logloss'])} | {f(m['brier'])} | {f(m['auc'])} | {f(m['ece'])} |")
                    lines.append("")
    (ev / f"{out_dir.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    from engine import config, paths
    from engine.data import dataset
    from engine.value._paths import next_version_dir      # torch-free (LOW-32)
    ap = argparse.ArgumentParser(description="Train and evaluate the LightGBM yardstick (P6.1).")
    ap.add_argument("--train", default="train")
    ap.add_argument("--val", default="val")
    ap.add_argument("--test", default="test")
    ap.add_argument("--vocab", default=None, help="vocab json, for the per-map breakdown")
    ap.add_argument("--out", default=None, help="default models/baseline_v<N+1>")
    a = ap.parse_args(argv)
    cfg   = config.load()
    try:
        if a.vocab:
            for split in (a.train, a.val, a.test):
                dataset.check_artifacts(split, a.vocab)
        tr, va = dataset.read_shards(a.train), dataset.read_shards(a.val)
        te    = dataset.read_shards(a.test)
        for what, arr in (("--train", tr), ("--val", va), ("--test", te)):
            if dataset.n_labelled(arr) == 0:
                raise ValueError(f"{what} split has no labelled rows")
    except (FileNotFoundError, ValueError) as exc:
        print(f"baseline: {exc}", file=sys.stderr)
        return 2
    te    = dataset.take(te, np.flatnonzero(te["tier"] == "pro")) if (te["tier"] == "pro").any() else te
    vocab = dataset.Vocab.load(a.vocab) if a.vocab else None
    b     = train(tr, va, seed=cfg.train.seed)          # early-stops on val pro when present
    res   = {"val": evaluate(b, va, cfg, vocab), "test_pro": evaluate(b, te, cfg, vocab)}
    res["best_iteration"] = res["val"]["best_iteration"]
    out = Path(a.out) if a.out else next_version_dir(paths.MODELS_DIR, "baseline_v")
    save(out, b, res, cfg=cfg)
    print(f"{out}: test pro logloss={res['test_pro']['logloss']} brier={res['test_pro']['brier']} "
          f"auc={res['test_pro']['auc']} ece={res['test_pro']['ece']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
