"""
Load and validate `engine/config.yaml` (P0.4).

    from engine import config
    cfg = config.load()                  # frozen Config, one dataclass per YAML group
    cfg.sampling.sample_hz               # 2
    config.config_hash(cfg)              # sha256 hex of the canonical YAML

Every key is required and unknown keys are rejected, so a typo in the YAML fails at
load time instead of silently falling back to a default; a key given twice in one
mapping is rejected too (LOW-06: YAML would otherwise keep the last one silently). Types come from the dataclass
annotations; ranges and cross-field rules are in the "validation" section below.
"""
import dataclasses
import hashlib
import math
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent / "config.yaml"


class ConfigError(ValueError):
    """The YAML is missing a key, has an unknown key, a wrong type, or a bad value."""


# ── nested value groups ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class BuyBands:
    pistol: int
    eco:    int
    half:   int
    force:  int


@dataclass(frozen=True)
class LossWeights:
    win:    float
    hazard: float
    plant:  float
    defuse: float
    tte:    float


@dataclass(frozen=True)
class StageBudget:
    gpu: float
    cpu: float


@dataclass(frozen=True)
class Budgets:
    parse:       StageBudget
    states:      StageBudget
    actions:     StageBudget
    predict:     StageBudget
    attribution: StageBudget
    value:       StageBudget
    report:      StageBudget
    total:       StageBudget


@dataclass(frozen=True)
class Targets:
    pro_maps:       int
    pro_per_map:    int
    pro_per_month:  int
    faceit_maps:    int
    faceit_per_map: int


@dataclass(frozen=True)
class FixedShares:
    flash:         float
    smoke_cover:   float
    trade_setup:   float
    he_damage:     float
    fire_pressure: float


@dataclass(frozen=True)
class NullPct:
    player:   float | None
    utility:  float | None
    context:  float | None
    pairwise: float | None


# ── top-level groups (one per YAML section) ───────────────────────────────────

@dataclass(frozen=True)
class Sampling:
    sample_hz:             int
    boundary_offset_ticks: int
    max_utility_tokens:    int
    los_cache:             bool
    heard_window_s:        float


@dataclass(frozen=True)
class Clock:
    round_time_s:  float
    bomb_timer_s:  float
    freeze_time_s: float


@dataclass(frozen=True)
class Actions:
    simultaneity_ticks: int
    damage_gap_s:       float
    movement_min_units: float
    rotation_min_units: float
    hold_min_s:         float
    hold_max_speed:     float
    save_min_units:     float
    save_window_s:      float
    trade_window_s:     float
    trade_max_units:    float
    enabler_recent_s:   float
    buy_bands:          BuyBands


@dataclass(frozen=True)
class Model:
    d_model:           int
    d_pool:            int
    L_set:             int
    L_time:            int
    n_heads:           int
    temporal_window_s: float
    hazard_horizon_s:  float
    plant_horizon_s:   float
    defuse_horizon_s:  float
    loss_weights:      LossWeights
    terminal_weight:   float


@dataclass(frozen=True)
class Train:
    lr:                  float
    wd:                  float
    warmup_steps:        int
    batch_rounds:        int
    max_epochs:          int
    patience:            int
    pro_oversample:      float
    pro_finetune_epochs: int
    pro_finetune_lr:     float
    seed:                int


@dataclass(frozen=True)
class Augment:
    util_drop:        float
    pos_jitter_units: float
    unflash:          float
    los_flip:         float
    dead_token_drop:  float


@dataclass(frozen=True)
class Calib:
    ece_target:            float
    n_bins:                int
    bin_tol:               float
    min_bin_n:             int
    martingale_tol:        float
    martingale_min_n:      int
    martingale_horizons_s: tuple[int, ...]


@dataclass(frozen=True)
class Sanity:
    model_n_snapshots:   int
    revive_ct_up_rate:   float
    revive_t_down_rate:  float
    plant_down_rate:     float
    time_left_down_rate: float
    hazard_auc_target:   float
    hazard_auc_floor:    float
    cf_n_events:         int
    cf_n_drift:          int
    unkill_toward_rate:  float
    unkill_median_abs:   float
    smoke_floor:         float
    smoke_floor_rate:    float
    smoke_max_abs:       float
    smoke_max_abs_rate:  float
    unflash_floor:       float
    unflash_rate:        float
    unplant_down_rate:   float
    freeze_max_abs:      float
    freeze_max_abs_rate: float
    freeze_mean_abs:     float
    ood_k:               int
    ood_percentile:      float
    ood_max_rate:        float


@dataclass(frozen=True)
class Attribution:
    shapley_exact_max:  int
    shapley_samples:    int
    drift_threshold_wp: float
    enabler_mode:       str
    enabler_total_cap:  float
    trade_share_band:   tuple[float, float]
    other_share_band:   tuple[float, float]
    min_hazard:         float
    fixed_shares:       FixedShares


@dataclass(frozen=True)
class Value:
    k_neighbours:       int
    n_min_regret:       int
    delta_D:            float
    delta_X:            float
    q_finetune_encoder: bool


@dataclass(frozen=True)
class Bench:
    budgets:          Budgets
    hard_ceiling_gpu: float
    hard_ceiling_cpu: float


@dataclass(frozen=True)
class Data:
    awpy_data_release: str | None
    targets:           Targets


@dataclass(frozen=True)
class QA:
    max_null_pct:       NullPct
    max_outliers:       int | None
    outlier_max_speed:  float
    outlier_max_health: int
    outlier_max_alive:  int


@dataclass(frozen=True)
class Config:
    sampling:    Sampling
    clock:       Clock
    actions:     Actions
    model:       Model
    train:       Train
    augment:     Augment
    calib:       Calib
    sanity:      Sanity
    attribution: Attribution
    value:       Value
    bench:       Bench
    data:        Data
    qa:          QA


# ── typed construction ────────────────────────────────────────────────────────

def _coerce(value, tp, where: str):
    """Check `value` against annotation `tp`; return it in the dataclass's form."""
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0], where)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        if not math.isfinite(value):
            raise ConfigError(f"{where}: must be finite, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported annotation {tp!r}")


def _build(cls, data, where: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{where or 'config'}: expected a mapping, got {type(data).__name__}")
    hints  = typing.get_type_hints(cls)
    names  = [f.name for f in dataclasses.fields(cls)]
    prefix = f"{where}." if where else ""
    unknown = sorted(set(data) - set(names))
    missing = [n for n in names if n not in data]
    if unknown:
        raise ConfigError(f"unknown key(s): {', '.join(prefix + k for k in unknown)}")
    if missing:
        raise ConfigError(f"missing key(s): {', '.join(prefix + k for k in missing)}")
    return cls(**{n: _coerce(data[n], hints[n], prefix + n) for n in names})


# ── validation ────────────────────────────────────────────────────────────────

SAMPLE_HZ_ALLOWED = (1, 2, 4, 8)
ENABLER_MODES     = ("hazard", "fixed")

# Fields that must lie in [0, 1] (probabilities, rates, win-probability amounts).
_UNIT_INTERVAL = (
    "augment.util_drop", "augment.unflash", "augment.los_flip", "augment.dead_token_drop",
    "calib.ece_target", "calib.bin_tol", "calib.martingale_tol",
    "sanity.revive_ct_up_rate", "sanity.revive_t_down_rate", "sanity.plant_down_rate",
    "sanity.time_left_down_rate", "sanity.hazard_auc_target", "sanity.hazard_auc_floor",
    "sanity.unkill_toward_rate", "sanity.unkill_median_abs", "sanity.smoke_floor_rate",
    "sanity.smoke_max_abs", "sanity.smoke_max_abs_rate", "sanity.unflash_rate",
    "sanity.unplant_down_rate", "sanity.freeze_max_abs", "sanity.freeze_max_abs_rate",
    "sanity.freeze_mean_abs", "sanity.ood_max_rate",
    "attribution.drift_threshold_wp", "attribution.enabler_total_cap", "attribution.min_hazard",
    "attribution.fixed_shares.flash", "attribution.fixed_shares.smoke_cover",
    "attribution.fixed_shares.trade_setup", "attribution.fixed_shares.he_damage",
    "attribution.fixed_shares.fire_pressure",
    "value.delta_D", "value.delta_X",
)
# The only fields allowed below zero: floors on a W change, in [-1, 0].
_NEGATIVE_OK = ("sanity.smoke_floor", "sanity.unflash_floor")
# Integers for which 0 is a legitimate value (LOW-05); every other integer must be >= 1.
_ZERO_OK = ("qa.max_outliers", "train.seed")
# Floats that must be strictly positive (a zero here would divide or never fire).
_STRICTLY_POSITIVE = (
    "clock.round_time_s", "clock.bomb_timer_s",
    "actions.damage_gap_s", "actions.hold_min_s", "actions.hold_max_speed",
    "actions.save_window_s", "actions.trade_window_s", "actions.enabler_recent_s",
    "model.temporal_window_s", "model.hazard_horizon_s", "model.plant_horizon_s",
    "model.defuse_horizon_s", "model.terminal_weight",
    "train.lr", "train.pro_finetune_lr", "train.pro_oversample",
    "attribution.drift_threshold_wp",
)


def _leaves(obj, prefix: str = ""):
    """Yield (dotted_path, value) for every scalar/tuple leaf of a config dataclass."""
    for f in dataclasses.fields(obj):
        v, path = getattr(obj, f.name), f"{prefix}{f.name}"
        if dataclasses.is_dataclass(v):
            yield from _leaves(v, path + ".")
        else:
            yield path, v


def _check_generic(cfg: Config) -> None:
    for path, v in _leaves(cfg):
        if v is None or isinstance(v, (bool, str)):
            continue
        if isinstance(v, tuple):
            if any(x < 0 for x in v):
                raise ConfigError(f"{path}: values must be >= 0, got {list(v)}")
            continue
        if isinstance(v, int) and v < (0 if path in _ZERO_OK else 1):
            raise ConfigError(f"{path}: must be {'>= 0' if path in _ZERO_OK else 'a positive integer'}, got {v}")
        if path in _NEGATIVE_OK:
            if not -1.0 <= v <= 0.0:
                raise ConfigError(f"{path}: must be in [-1, 0], got {v}")
        elif v < 0:
            raise ConfigError(f"{path}: must be >= 0, got {v}")
        if path in _UNIT_INTERVAL and v > 1:
            raise ConfigError(f"{path}: must be in [0, 1], got {v}")
        if path in _STRICTLY_POSITIVE and v <= 0:
            raise ConfigError(f"{path}: must be > 0, got {v}")


def _check_band(path: str, band: tuple[float, float]) -> None:
    lo, hi = band
    if not 0.0 <= lo <= hi <= 1.0:
        raise ConfigError(f"{path}: need 0 <= low <= high <= 1, got [{lo}, {hi}]")


def _check_budgets(cfg: Config) -> None:
    b = cfg.bench.budgets
    stages = [getattr(b, f.name) for f in dataclasses.fields(b) if f.name != "total"]
    for dev, ceiling in (("gpu", cfg.bench.hard_ceiling_gpu), ("cpu", cfg.bench.hard_ceiling_cpu)):
        parts, total = sum(getattr(s, dev) for s in stages), getattr(b.total, dev)
        if not parts <= total <= ceiling:
            raise ConfigError(f"bench.budgets ({dev}): need sum of stages ({parts}) <= "
                              f"total ({total}) <= hard ceiling ({ceiling})")


def validate(cfg: Config) -> Config:
    """Raise ConfigError on the first violated rule; return cfg unchanged."""
    _check_generic(cfg)
    if cfg.sampling.sample_hz not in SAMPLE_HZ_ALLOWED:
        raise ConfigError(f"sampling.sample_hz: must be one of {SAMPLE_HZ_ALLOWED}, got {cfg.sampling.sample_hz}")
    if cfg.model.d_model % cfg.model.n_heads:
        raise ConfigError(f"model.n_heads ({cfg.model.n_heads}) must divide model.d_model ({cfg.model.d_model})")
    bands = cfg.actions.buy_bands
    order = [bands.pistol, bands.eco, bands.half, bands.force]
    if order != sorted(set(order)):
        raise ConfigError(f"actions.buy_bands: need pistol < eco < half < force, got {order}")
    _check_band("attribution.trade_share_band", cfg.attribution.trade_share_band)
    _check_band("attribution.other_share_band", cfg.attribution.other_share_band)
    if cfg.attribution.enabler_mode not in ENABLER_MODES:
        raise ConfigError(f"attribution.enabler_mode: must be one of {ENABLER_MODES}, got {cfg.attribution.enabler_mode!r}")
    if cfg.sanity.hazard_auc_floor > cfg.sanity.hazard_auc_target:
        raise ConfigError("sanity.hazard_auc_floor must be <= sanity.hazard_auc_target")
    if not 0 < cfg.sanity.ood_percentile <= 100:
        raise ConfigError(f"sanity.ood_percentile: must be in (0, 100], got {cfg.sanity.ood_percentile}")
    for path, v in _leaves(cfg.qa.max_null_pct, "qa.max_null_pct."):
        if v is not None and v > 100:
            raise ConfigError(f"{path}: a percentage, must be <= 100, got {v}")
    hz = cfg.calib.martingale_horizons_s
    if not hz:
        raise ConfigError("calib.martingale_horizons_s: must not be empty")
    if any(h <= 0 for h in hz) or len(set(hz)) != len(hz):       # LOW-05: a 0 s look-ahead is vacuous
        raise ConfigError(f"calib.martingale_horizons_s: need distinct values > 0, got {list(hz)}")
    _check_budgets(cfg)
    return cfg


# ── public API ────────────────────────────────────────────────────────────────

class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping holding the same key twice (LOW-06)."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node, deep: bool = False):
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            dup = key in seen
        except TypeError:                             # unhashable key: SafeLoader reports it below
            continue
        if dup:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def from_dict(data: dict) -> Config:
    """Build and validate a Config from an already-parsed YAML mapping."""
    return validate(_build(Config, data, ""))


def load(path: Path | str | None = None) -> Config:
    """Load, type-check and validate the YAML at `path` (default DEFAULT_PATH)."""
    p = Path(path) if path is not None else DEFAULT_PATH
    with open(p, encoding="utf-8") as fh:
        try:
            data = yaml.load(fh, Loader=_UniqueKeyLoader)    # safe_load + duplicate-key refusal
        except yaml.YAMLError as exc:                 # a tab indent, an unclosed bracket, a duplicate key: exit 2, not 1
            mark = getattr(exc, "problem_mark", None)
            where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
            raise ConfigError(f"{p.name}: YAML syntax error{where}: {getattr(exc, 'problem', None) or exc}") from None
    return from_dict(data)


def as_dict(cfg: Config) -> dict:
    """Plain nested dict (tuples become lists), the form that is hashed and dumped."""
    def plain(v):
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v
    return plain(dataclasses.asdict(cfg))


def config_hash(cfg: Config) -> str:
    """sha256 of the canonical YAML dump (sorted keys), so key order and comments in
    the file never change the hash but any value does."""
    canonical = yaml.safe_dump(as_dict(cfg), sort_keys=True, default_flow_style=False,
                               allow_unicode=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
