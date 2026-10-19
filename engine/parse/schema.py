"""
`RawTables` (Amendment 1 contract): one polars DataFrame per Appendix A.1 table plus
`meta`. `validate()` checks columns and dtypes; `slice_until(tick)` returns the causal
slice (what was knowable at tick t) the state builder passes to feature code (P3.6.3).

Column semantics the dtype alone does not say:
- `ticks.flash_duration` is the REMAINING blind time in seconds at that tick (0.0 when
  not blind; 0 is a legitimate value here). awpy exposes the onset duration, so the
  parser derives the remaining time from the blind events.
- `ticks` rows for dead players continue (is_alive false, health 0, last position,
  active_weapon null, inventory empty, equipment_value 0).
- `kills.distance` and every x/y/z are Hammer units; `yaw`/`pitch` raw degrees.
- `grenades.expire_tick` is null for flash and HE (instantaneous).
- `kills.is_trade` (awpy 3 exposes it) is optional: validated when present,
  filled with null by `from_parsed()` when absent.

Filled by: P2.5.1.
"""
from dataclasses import dataclass, field, fields

import polars as pl

# ── dtypes ────────────────────────────────────────────────────────────────────

I16, I32, I64, U64 = pl.Int16, pl.Int32, pl.Int64, pl.UInt64
F32, STR, BOOL     = pl.Float32, pl.String, pl.Boolean

_XYZ = {"x": F32, "y": F32, "z": F32}

# ── Appendix A.1 ──────────────────────────────────────────────────────────────

TABLE_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "rounds": {
        "round_num": I32, "start_tick": I64, "freeze_end_tick": I64, "end_tick": I64,
        "official_end_tick": I64, "winner_side": STR, "end_reason": STR,
        "bomb_planted": BOOL, "ct_score_before": I32, "t_score_before": I32,
    },
    "ticks": {
        "tick": I64, "round_num": I32, "steamid": U64, "name": STR, "side": STR,
        **_XYZ, "yaw": F32, "pitch": F32, "vx": F32, "vy": F32, "vz": F32,
        "health": I16, "armor": I16, "has_helmet": BOOL, "has_defuser": BOOL,
        "has_c4": BOOL, "is_alive": BOOL, "active_weapon": STR, "inventory": pl.List(STR),
        "money": I32, "equipment_value": I32, "is_scoped": BOOL, "is_ducking": BOOL,
        "is_walking": BOOL, "is_reloading": BOOL, "is_defusing": BOOL, "is_planting": BOOL,
        "flash_duration": F32, "spotted_by": pl.List(U64), "kills_round": I32,
        "damage_round": I32,
    },
    "kills": {
        "tick": I64, "round_num": I32, "attacker_steamid": U64, "victim_steamid": U64,
        "assister_steamid": U64, "flash_assister_steamid": U64, "weapon": STR,
        "headshot": BOOL, "penetrated": I16, "through_smoke": BOOL, "attacker_blind": BOOL,
        "victim_blind": BOOL, "noscope": BOOL, "distance": F32,
        "attacker_x": F32, "attacker_y": F32, "attacker_z": F32,
        "victim_x": F32, "victim_y": F32, "victim_z": F32, "team_kill": BOOL,
    },
    "damages": {
        "tick": I64, "round_num": I32, "attacker_steamid": U64, "victim_steamid": U64,
        "weapon": STR, "dmg_health": I16, "dmg_armor": I16, "hitgroup": STR,
    },
    "grenades": {
        "entity_id": I64, "thrower_steamid": U64, "type": STR, "throw_tick": I64,
        "detonate_tick": I64, "expire_tick": I64, **_XYZ,
    },
    "smokes": {
        "entity_id": I64, "start_tick": I64, "end_tick": I64, **_XYZ, "thrower_steamid": U64,
    },
    "infernos": {
        "entity_id": I64, "start_tick": I64, "end_tick": I64, **_XYZ,
        "extent_radius": F32, "thrower_steamid": U64,
    },
    "flashes": {
        "entity_id": I64, "detonate_tick": I64, "player_steamid": U64, "blind_duration": F32,
    },
    "bomb": {
        "tick": I64, "round_num": I32, "event": STR, "steamid": U64, "site": STR,
        "has_kit": BOOL, **_XYZ,
    },
    "weapon_fire": {"tick": I64, "steamid": U64, "weapon": STR},
    "footsteps":   {"tick": I64, "steamid": U64, **_XYZ},
    "items":       {"tick": I64, "steamid": U64, "event": STR, "item": STR},
}

TABLES          = tuple(TABLE_SCHEMAS)
OPTIONAL_TABLES = ("weapon_fire", "footsteps", "items")
META_KEYS       = ("map", "tick_rate", "game_version", "server", "demo_sha256")

# columns not in A.1 that are checked when present (never required)
OPTIONAL_COLUMNS: dict[str, dict[str, pl.DataType]] = {
    "kills": {"is_trade": BOOL},
}

# the few columns that may never be null (everything else follows the A.1 null policy)
NOT_NULL: dict[str, tuple[str, ...]] = {
    "rounds": ("round_num", "start_tick", "freeze_end_tick"),
    "ticks":  ("tick", "round_num", "steamid", "side", "is_alive"),
}

BOMB_EVENTS    = ("plant_start", "plant_complete", "defuse_start", "defuse_complete",
                  "defuse_abort", "drop", "pickup", "explode")
GRENADE_TYPES  = ("smoke", "flash", "he", "molotov", "incendiary", "decoy")
ITEM_EVENTS    = ("drop", "pickup")

# causal slicing: which column says when a row became known
_TICK_COL = {
    "ticks": "tick", "kills": "tick", "damages": "tick", "bomb": "tick",
    "weapon_fire": "tick", "footsteps": "tick", "items": "tick",
    "flashes": "detonate_tick",
}
# columns of a round that are only known once the round has ended
_ROUND_END_COLS = ("end_tick", "official_end_tick", "winner_side", "end_reason", "bomb_planted")


class SchemaError(ValueError):
    """Raised by `RawTables.validate(strict=True)` and `from_parsed()`."""


# ── dtype compatibility ───────────────────────────────────────────────────────

def _family(dt: pl.DataType) -> str:
    if dt == pl.Null:
        return "null"
    if dt == pl.Boolean:
        return "bool"
    if dt.is_integer():
        return "int"
    if dt.is_float():
        return "float"
    if dt == pl.String or dt == pl.Categorical or dt == pl.Enum:
        return "str"
    if isinstance(dt, pl.List):
        return "list"
    return str(dt)


def _castable(series: pl.Series, target: pl.DataType) -> bool:
    """True when `series` can become `target` without changing meaning: exact match,
    an all-null column, int -> wider/other int with every value in range, int -> float,
    float -> float, str-like -> str, list -> list with a castable inner type. Float ->
    int and str -> number are rejected (polars would truncate / parse silently)."""
    src = series.dtype
    if src == target:
        return True
    fs, ft = _family(src), _family(target)
    if fs == "null" or series.null_count() == series.len():
        return True
    ok_pairs = {("int", "int"), ("int", "float"), ("float", "float"), ("str", "str"),
                ("bool", "bool"), ("list", "list")}
    if (fs, ft) not in ok_pairs:
        return False
    if fs == "list":
        return _castable(series.explode(), target.inner)
    try:
        series.cast(target, strict=True)
    except Exception:
        return False
    return True


def _empty(name: str) -> pl.DataFrame:
    return pl.DataFrame(schema=TABLE_SCHEMAS[name])


# ── RawTables ─────────────────────────────────────────────────────────────────

@dataclass
class RawTables:
    """One parsed map. Every frame carries at least its A.1 columns; extras are kept."""
    rounds:      pl.DataFrame
    ticks:       pl.DataFrame
    kills:       pl.DataFrame
    damages:     pl.DataFrame
    grenades:    pl.DataFrame
    smokes:      pl.DataFrame
    infernos:    pl.DataFrame
    flashes:     pl.DataFrame
    bomb:        pl.DataFrame
    weapon_fire: pl.DataFrame
    footsteps:   pl.DataFrame
    items:       pl.DataFrame
    meta:        dict = field(default_factory=dict)

    # ── construction ──

    @classmethod
    def empty(cls, meta: dict | None = None) -> "RawTables":
        """Every table empty with the A.1 schema."""
        return cls(**{t: _empty(t) for t in TABLES}, meta=dict(meta or {}))

    @classmethod
    def from_parsed(cls, tables: dict[str, pl.DataFrame], meta: dict) -> "RawTables":
        """Adapter for the parser's output. Missing optional tables become empty frames;
        a missing required table raises SchemaError (the parser must emit an empty
        frame for, e.g., a demo with no fires). Known columns are cast to their A.1
        dtype when that is lossless (see `_castable`); the rest is left for
        `validate()` to report. Missing optional columns are added as null."""
        missing = [t for t in TABLES if t not in tables and t not in OPTIONAL_TABLES]
        if missing:
            raise SchemaError(f"parser output lacks required tables: {missing}")
        out = {}
        for name in TABLES:
            df = tables.get(name)
            if df is None:
                out[name] = _empty(name)
                continue
            casts = []
            for col, dt in {**TABLE_SCHEMAS[name], **OPTIONAL_COLUMNS.get(name, {})}.items():
                if col in df.columns and df[col].dtype != dt and _castable(df[col], dt):
                    casts.append(pl.col(col).cast(dt))
            if casts:
                df = df.with_columns(casts)
            for col, dt in OPTIONAL_COLUMNS.get(name, {}).items():
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None, dtype=dt).alias(col))
            out[name] = df
        return cls(**out, meta=dict(meta))

    def frames(self) -> dict[str, pl.DataFrame]:
        return {t: getattr(self, t) for t in TABLES}

    # ── checks ──

    def validate(self, strict: bool = True) -> list[str]:
        """Return every problem found (required columns, dtypes, NOT_NULL, meta). With
        `strict`, raise SchemaError listing them instead of returning a non-empty list."""
        problems: list[str] = []
        for key in META_KEYS:
            if key not in self.meta:
                problems.append(f"meta: missing key {key!r}")
        tr = self.meta.get("tick_rate")
        if not isinstance(tr, int) or isinstance(tr, bool) or tr <= 0:
            problems.append(f"meta: tick_rate must be a positive int, got {tr!r}")
        if not self.meta.get("map"):
            problems.append("meta: map is empty")

        for name in TABLES:
            df = getattr(self, name)
            if not isinstance(df, pl.DataFrame):
                problems.append(f"{name}: not a polars DataFrame ({type(df).__name__})")
                continue
            for col, dt in TABLE_SCHEMAS[name].items():
                if col not in df.columns:
                    problems.append(f"{name}: missing column {col!r}")
                elif not _castable(df[col], dt):
                    problems.append(f"{name}.{col}: dtype {df[col].dtype} not castable to {dt}")
            for col, dt in OPTIONAL_COLUMNS.get(name, {}).items():
                if col in df.columns and not _castable(df[col], dt):
                    problems.append(f"{name}.{col}: dtype {df[col].dtype} not castable to {dt}")
            for col in NOT_NULL.get(name, ()):
                if col in df.columns and df[col].null_count():
                    problems.append(f"{name}.{col}: {df[col].null_count()} nulls in a never-null column")
        if strict and problems:
            raise SchemaError("RawTables invalid:\n  " + "\n  ".join(problems))
        return problems

    # ── causal slice ──

    def slice_until(self, tick: int) -> "RawTables":
        """What was knowable at `tick` (P3.6.3): rows with their event tick <= tick.
        Smokes / infernos / grenades are kept from their start / throw tick, with every
        end-of-life value that lies in the future nulled (when a smoke fades, when and
        where a thrown grenade will detonate). Rounds that have not started are
        dropped (their score columns would reveal the current round's winner); for the
        round in progress the end-of-round columns are nulled. Meta is copied."""
        t = int(tick)
        out = {}
        for name, col in _TICK_COL.items():
            out[name] = getattr(self, name).filter(pl.col(col) <= t)

        for name in ("smokes", "infernos"):
            df = getattr(self, name).filter(pl.col("start_tick") <= t)
            out[name] = df.with_columns(
                pl.when(pl.col("end_tick") > t).then(None).otherwise(pl.col("end_tick"))
                  .alias("end_tick"))

        g = self.grenades.filter(pl.col("throw_tick") <= t)
        future_det = pl.col("detonate_tick") > t
        out["grenades"] = g.with_columns(
            [pl.when(future_det).then(None).otherwise(pl.col(c)).alias(c)
             for c in ("x", "y", "z")]
            + [pl.when(future_det).then(None).otherwise(pl.col("detonate_tick")).alias("detonate_tick"),
               pl.when(pl.col("expire_tick") > t).then(None).otherwise(pl.col("expire_tick"))
                 .alias("expire_tick")])

        r = self.rounds.filter(pl.col("start_tick") <= t)
        not_over = pl.col("end_tick").is_null() | (pl.col("end_tick") > t)
        r = r.with_columns(
            [pl.when(not_over).then(None).otherwise(pl.col(c)).alias(c) for c in _ROUND_END_COLS])
        r = r.with_columns(
            pl.when(pl.col("official_end_tick") > t).then(None)
              .otherwise(pl.col("official_end_tick")).alias("official_end_tick"))
        out["rounds"] = r
        return RawTables(**out, meta=dict(self.meta))


assert [f.name for f in fields(RawTables)] == [*TABLES, "meta"], "RawTables fields drifted from A.1"
