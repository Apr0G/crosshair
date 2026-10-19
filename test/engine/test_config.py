"""
Config loading, validation and hashing (P0.4).

    .venv\\Scripts\\python.exe test\\engine\\test_config.py
"""
import copy
import sys
import tempfile
from pathlib import Path

import yaml

from _harness import ROOT, check, section, summary

from engine import config

# Appendix D, group -> keys. Extra keys in the YAML are allowed (the plan names more
# thresholds than the appendix lists); missing ones are not.
APPENDIX_D = {
    "sampling":    ["sample_hz", "boundary_offset_ticks", "max_utility_tokens", "los_cache"],
    "clock":       ["round_time_s", "bomb_timer_s", "freeze_time_s"],
    "actions":     ["simultaneity_ticks", "damage_gap_s", "movement_min_units", "rotation_min_units",
                    "hold_min_s", "hold_max_speed", "save_min_units", "save_window_s",
                    "trade_window_s", "trade_max_units", "enabler_recent_s", "buy_bands"],
    "model":       ["d_model", "d_pool", "L_set", "L_time", "n_heads", "temporal_window_s",
                    "hazard_horizon_s", "plant_horizon_s", "defuse_horizon_s", "loss_weights",
                    "terminal_weight"],
    "train":       ["lr", "wd", "warmup_steps", "batch_rounds", "max_epochs", "patience",
                    "pro_oversample", "pro_finetune_epochs", "pro_finetune_lr", "seed"],
    "augment":     ["util_drop", "pos_jitter_units", "unflash", "los_flip", "dead_token_drop"],
    "calib":       ["ece_target", "bin_tol", "min_bin_n", "martingale_tol"],
    "sanity":      ["ood_percentile", "ood_max_rate", "revive_ct_up_rate", "revive_t_down_rate",
                    "plant_down_rate", "time_left_down_rate", "unkill_toward_rate",
                    "unkill_median_abs", "smoke_floor", "unflash_floor", "unplant_down_rate",
                    "freeze_max_abs", "freeze_mean_abs"],
    "attribution": ["shapley_exact_max", "shapley_samples", "drift_threshold_wp", "enabler_mode",
                    "enabler_total_cap", "trade_share_band", "other_share_band", "min_hazard"],
    "value":       ["k_neighbours", "n_min_regret", "delta_D", "delta_X", "q_finetune_encoder"],
    "bench":       ["budgets", "hard_ceiling_gpu", "hard_ceiling_cpu"],
    "data":        ["awpy_data_release", "targets"],
    "qa":          ["max_null_pct", "max_outliers"],
}
NESTED = {
    ("actions", "buy_bands"):   ["pistol", "eco", "half", "force"],
    ("model", "loss_weights"):  ["win", "hazard", "plant", "defuse", "tte"],
    ("data", "targets"):        ["pro_maps", "pro_per_map", "faceit_maps"],
}
# P0.4.1 starting values.
P041 = {
    ("sampling", "sample_hz"): 2, ("sampling", "boundary_offset_ticks"): 1,
    ("model", "hazard_horizon_s"): 5, ("model", "plant_horizon_s"): 15,
    ("model", "defuse_horizon_s"): 15, ("model", "temporal_window_s"): 10,
    ("attribution", "shapley_exact_max"): 6, ("attribution", "shapley_samples"): 200,
    ("attribution", "enabler_total_cap"): 0.5, ("attribution", "trade_share_band"): [0.05, 0.40],
    ("value", "n_min_regret"): 30, ("value", "k_neighbours"): 200,
    ("attribution", "drift_threshold_wp"): 0.02, ("actions", "movement_min_units"): 32,
}


def raw() -> dict:
    with open(config.DEFAULT_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def rejects(data: dict) -> tuple[bool, str]:
    try:
        config.from_dict(data)
    except config.ConfigError as e:
        return True, str(e)
    return False, "accepted"


def with_value(data: dict, path: list, value) -> dict:
    d = copy.deepcopy(data)
    node = d
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return d


def leaves(d: dict, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v, prefix + (k,))
        else:
            yield list(prefix + (k,)), v


def candidates(v) -> list:
    """Small edits to a leaf, at least one of which should still validate."""
    if isinstance(v, bool):
        return [not v]
    if isinstance(v, int):
        return [v + 1, v * 2, v - 1]
    if isinstance(v, float):
        return [v * 1.001 if v else 0.001, v * 0.999, v - 0.001]
    if isinstance(v, str):
        return [v + "x", "fixed"]
    if v is None:
        return [0.5, 1, "x"]
    if isinstance(v, list):
        return [[*v[:-1], x] for x in candidates(v[-1])] + [[x, *v[1:]] for x in candidates(v[0])]
    return []


def test_default() -> None:
    section("default config")
    cfg = config.load()
    check("default config loads", isinstance(cfg, config.Config))
    check("load(DEFAULT_PATH) == load()", config.load(config.DEFAULT_PATH) == cfg)
    try:
        cfg.sampling.sample_hz = 4
        check("config is frozen", False)
    except Exception:
        check("config is frozen", True)
    data = raw()
    missing = [f"{g}.{k}" for g, ks in APPENDIX_D.items() for k in ks if k not in data.get(g, {})]
    missing += [f"{g}.{k}.{n}" for (g, k), ns in NESTED.items() for n in ns if n not in data[g][k]]
    check("every Appendix D key present", not missing, f"missing {missing}")
    wrong = [f"{g}.{k}" for (g, k), want in P041.items() if data[g][k] != want]
    check("P0.4.1 starting values", not wrong, f"differ: {wrong}")
    check("trade band parsed as a tuple", cfg.attribution.trade_share_band == (0.05, 0.40))
    check("lr parsed as a float (3.0e-4, not a string)", cfg.train.lr == 3e-4)


def test_validation() -> None:
    section("validation")
    data = raw()
    cases = [
        ("rejects sample_hz: 3",          ["sampling", "sample_hz"], 3),
        ("rejects band [0.4, 0.05]",      ["attribution", "trade_share_band"], [0.4, 0.05]),
        ("rejects band above 1",          ["attribution", "other_share_band"], [0.0, 1.5]),
        ("rejects probability 1.2",       ["augment", "util_drop"], 1.2),
        ("rejects negative probability",  ["attribution", "enabler_total_cap"], -0.1),
        ("rejects zero count",            ["value", "k_neighbours"], 0),
        ("rejects float for an int",      ["train", "batch_rounds"], 256.5),
        ("rejects bool for an int",       ["sampling", "max_utility_tokens"], True),
        ("rejects string number",         ["train", "lr"], "3e-4"),
        ("rejects unordered buy bands",   ["actions", "buy_bands", "eco"], 900),
        ("rejects unknown enabler_mode",  ["attribution", "enabler_mode"], "magic"),
        ("rejects n_heads not dividing d_model", ["model", "n_heads"], 3),
        ("rejects stage budgets over total",     ["bench", "budgets", "attribution", "gpu"], 50),
        ("rejects null for a required number",   ["clock", "round_time_s"], None),
    ]
    for desc, path, value in cases:
        ok, why = rejects(with_value(data, path, value))
        check(desc, ok, why)
    ok, why = rejects({**data, "typo_group": {}})
    check("rejects unknown group", ok, why)
    d = copy.deepcopy(data)
    del d["sampling"]["los_cache"]
    ok, why = rejects(d)
    check("rejects missing key", ok, why)
    check("accepts sample_hz 4", not rejects(with_value(data, ["sampling", "sample_hz"], 4))[0])
    for desc, path, value in (("accepts qa.max_outliers 0 (LOW-05)", ["qa", "max_outliers"], 0),
                              ("accepts train.seed 0 (LOW-05)", ["train", "seed"], 0)):
        ok, why = rejects(with_value(data, path, value))
        check(desc, not ok, why)
    for desc, path, value in (("rejects qa.max_outliers -1", ["qa", "max_outliers"], -1),
                              ("rejects a 0 s martingale horizon (LOW-05)", ["calib", "martingale_horizons_s"], [0, 5, 10]),
                              ("rejects duplicate martingale horizons (LOW-05)", ["calib", "martingale_horizons_s"], [5, 5, 10])):
        ok, why = rejects(with_value(data, path, value))
        check(desc, ok, why)


def test_load_errors() -> None:
    section("load(): malformed and duplicate-key YAML (LOW-06, LOW-75)")
    text = config.DEFAULT_PATH.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        dup = Path(tmp) / "dup.yaml"
        dup.write_text(text.replace("sampling:\n", "sampling:\n  sample_hz: 4\n", 1), encoding="utf-8")
        try:
            config.load(dup)
            check("duplicate key refused with file name and key", False, "accepted (last one won)")
        except config.ConfigError as e:
            check("duplicate key refused with file name and key", "dup.yaml" in str(e) and "sample_hz" in str(e)
                  and "line" in str(e), str(e))
        bad = Path(tmp) / "bad.yaml"
        bad.write_text("sampling:\n  sample_hz: [1, 2\n", encoding="utf-8")
        try:
            config.load(bad)
            check("malformed YAML -> ConfigError naming the file and line", False, "accepted")
        except config.ConfigError as e:
            check("malformed YAML -> ConfigError naming the file and line", "bad.yaml" in str(e) and "line" in str(e),
                  str(e))
        same = Path(tmp) / "same.yaml"
        same.write_text(text, encoding="utf-8")
        check("the default file still loads through the strict loader", config.load(same) == config.load())


def test_hash() -> None:
    section("config_hash")
    data = raw()
    base = config.config_hash(config.from_dict(data))
    check("hash is sha256 hex", len(base) == 64 and int(base, 16) >= 0)
    check("hash is deterministic", config.config_hash(config.load()) == base)

    reordered = {g: dict(reversed(list(v.items()))) for g, v in reversed(list(data.items()))}
    check("hash stable across key order", config.config_hash(config.from_dict(reordered)) == base)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text(yaml.safe_dump(reordered, sort_keys=False), encoding="utf-8")
        check("hash stable across file formatting/comments", config.config_hash(config.load(p)) == base)

    unchanged, untestable = [], []
    for path, value in leaves(data):
        for cand in candidates(value):
            try:
                cfg = config.from_dict(with_value(data, path, cand))
            except config.ConfigError:
                continue
            if config.config_hash(cfg) == base:
                unchanged.append(".".join(path))
            break
        else:
            untestable.append(".".join(path))
    n = sum(1 for _ in leaves(data))
    check(f"hash changes when any of the {n} values changes", not unchanged, f"unchanged: {unchanged}")
    check("every value had a valid edit to test", not untestable, f"no valid edit: {untestable}")


def test_budget_mirror() -> None:
    section("bench budget mirror")
    with open(ROOT / "bench" / "budget.yaml", encoding="utf-8") as fh:
        budget = yaml.safe_load(fh)
    cfg = config.load()
    b = config.as_dict(cfg)["bench"]
    same = (b["budgets"] == {**budget["stages"], "total": budget["total"]}
            and b["hard_ceiling_gpu"] == budget["hard_ceiling"]["gpu"]
            and b["hard_ceiling_cpu"] == budget["hard_ceiling"]["cpu"])
    check("config.yaml bench == bench/budget.yaml", same)


def main() -> int:
    test_default()
    test_validation()
    test_load_errors()
    test_hash()
    test_budget_mirror()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
