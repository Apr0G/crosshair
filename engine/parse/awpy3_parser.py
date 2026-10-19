"""
Demo -> raw polars tables with awpy 3 (P2.5.1). Tick rate, map name and game version
come from the demo header, never a literal (FACEIT/pro demos are 128-tick).

    from engine.parse import awpy3_parser
    tables, meta = awpy3_parser.parse("match.dem")      # Appendix A.1 tables + meta dict
    raw = RawTables.from_parsed(tables, meta)           # engine/parse/schema.py

    info   = awpy3_parser.header_info("match.dem")      # cheap: header only
    reason = awpy3_parser.quarantine_check(info)        # None, or a P1.4.1 reason

`parse` returns exactly the Appendix A.1 tables (`TABLES`), every column present with
the dtype in `A1_SCHEMAS`; a field awpy does not expose is a typed all-null column,
never 0 or "?". The awpy -> A.1 mapping, and every field that could not be obtained,
is tabled in the "Parser notes" section of docs/ENGINE_ENV.md.

`demo=` takes any awpy-shaped object (tests pass a fake, so no real demo is needed):
attributes `header`, `rounds`, `kills`, `damages`, `bomb`, `grenades`, `fires`,
`smokes`, `blinds`, `shots`, `item_events`, `players`, `events` and a
`snapshots(start_tick=, end_tick=)` method.
"""
import hashlib
import math
import statistics
from pathlib import Path

import polars as pl

# ── contract (Appendix A.1) ───────────────────────────────────────────────────
# A.1 names some dtypes and leaves others implied; the implied ones follow the
# nearest named sibling (every *_tick is i64, every steamid u64, positions f32).

TICK  = pl.Int64
ROUND = pl.Int32
SID   = pl.UInt64
POS   = pl.Float32
STR   = pl.String
BOOL  = pl.Boolean

A1_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "rounds": {
        "round_num": ROUND, "start_tick": TICK, "freeze_end_tick": TICK, "end_tick": TICK,
        "official_end_tick": TICK, "winner_side": STR, "end_reason": STR,
        "bomb_planted": BOOL, "ct_score_before": pl.Int32, "t_score_before": pl.Int32,
    },
    "ticks": {
        "tick": TICK, "round_num": ROUND, "steamid": SID, "name": STR, "side": STR,
        "x": POS, "y": POS, "z": POS, "yaw": POS, "pitch": POS,
        "vx": POS, "vy": POS, "vz": POS, "health": pl.Int16, "armor": pl.Int16,
        "has_helmet": BOOL, "has_defuser": BOOL, "has_c4": BOOL, "is_alive": BOOL,
        "active_weapon": STR, "inventory": pl.List(STR), "money": pl.Int32,
        "equipment_value": pl.Int32, "is_scoped": BOOL, "is_ducking": BOOL,
        "is_walking": BOOL, "is_reloading": BOOL, "is_defusing": BOOL, "is_planting": BOOL,
        "flash_duration": POS, "spotted_by": pl.List(SID), "kills_round": pl.Int32,
        "damage_round": pl.Int32,
    },
    "kills": {
        "tick": TICK, "round_num": ROUND, "attacker_steamid": SID, "victim_steamid": SID,
        "assister_steamid": SID, "flash_assister_steamid": SID, "weapon": STR,
        "headshot": BOOL, "penetrated": pl.Int16, "through_smoke": BOOL,
        "attacker_blind": BOOL, "victim_blind": BOOL, "noscope": BOOL, "distance": POS,
        "attacker_x": POS, "attacker_y": POS, "attacker_z": POS,
        "victim_x": POS, "victim_y": POS, "victim_z": POS, "team_kill": BOOL,
    },
    "damages": {
        "tick": TICK, "round_num": ROUND, "attacker_steamid": SID, "victim_steamid": SID,
        "weapon": STR, "dmg_health": pl.Int16, "dmg_armor": pl.Int16, "hitgroup": STR,
    },
    "grenades": {
        "entity_id": pl.Int64, "thrower_steamid": SID, "type": STR, "throw_tick": TICK,
        "detonate_tick": TICK, "expire_tick": TICK, "x": POS, "y": POS, "z": POS,
    },
    "smokes": {
        "entity_id": pl.Int64, "start_tick": TICK, "end_tick": TICK,
        "x": POS, "y": POS, "z": POS, "thrower_steamid": SID,
    },
    "infernos": {
        "entity_id": pl.Int64, "start_tick": TICK, "end_tick": TICK,
        "x": POS, "y": POS, "z": POS, "extent_radius": POS, "thrower_steamid": SID,
    },
    "flashes": {
        "entity_id": pl.Int64, "detonate_tick": TICK, "player_steamid": SID,
        "blind_duration": POS,
    },
    "bomb": {
        "tick": TICK, "round_num": ROUND, "event": STR, "steamid": SID, "site": STR,
        "has_kit": BOOL, "x": POS, "y": POS, "z": POS,
    },
    "weapon_fire": {"tick": TICK, "steamid": SID, "weapon": STR},
    "footsteps":   {"tick": TICK, "steamid": SID, "x": POS, "y": POS, "z": POS},
    "items":       {"tick": TICK, "steamid": SID, "event": STR, "item": STR},
}
TABLES = tuple(A1_SCHEMAS)

# Not in A.1 but accepted by engine/parse/schema.py (OPTIONAL_COLUMNS): emitted only
# when awpy provides them.
A1_OPTIONAL: dict[str, dict[str, pl.DataType]] = {"kills": {"is_trade": BOOL}}

META_SCHEMA = {"map": STR, "tick_rate": pl.Int32, "game_version": STR, "server": STR,
               "demo_sha256": STR}

GRENADE_TYPES = ("smoke", "flash", "he", "molotov", "incendiary", "decoy")
BOMB_EVENTS   = ("plant_start", "plant_complete", "defuse_start", "defuse_complete",
                 "defuse_abort", "drop", "pickup", "explode")
ITEM_EVENTS   = ("drop", "pickup")

# awpy name -> A.1 name. A value of None means "no A.1 name": the row is dropped and
# counted in meta["dropped_rows"] (see docs/ENGINE_ENV.md, parser notes).
_GRENADE_TYPE_MAP = {
    "smoke": "smoke", "smokegrenade": "smoke", "flashbang": "flash", "flash": "flash",
    "he": "he", "hegrenade": "he", "molotov": "molotov", "incendiary": "incendiary",
    "incgrenade": "incendiary", "inferno": "molotov", "decoy": "decoy",
}
_BOMB_EVENT_MAP = {
    "start_plant": "plant_start", "finish_plant": "plant_complete", "defuse": "defuse_complete",
    "pickup": "pickup", "drop": "drop", "interrupt_plant": None,
}
_EXPLODE_REASONS = ("target_bombed", "bomb_exploded")

# ── quarantine (P1.4.1) ───────────────────────────────────────────────────────
# config.yaml has no quarantine keys yet (Appendix D lists none), so the thresholds
# live here. Move them to config when Appendix D gains them.

EXPECTED_TICK_RATE = 128
TICK_RATE_TOL      = 0.5           # ticks/s; playback_ticks / playback_time is not exact
MIN_ROUNDS         = 13            # a completed MR12 map: 13-0 .. 13-2 are legitimate (decided 2026-10-19)
PLAYERS_PER_SIDE   = 5
MIN_DURATION_S     = 12 * 60       # 13 rounds cannot finish faster
MAX_DURATION_S     = 5 * 3600      # longer than any plausible MR12 + overtime

QUARANTINE_REASONS = ("tick_rate_not_128", "too_few_rounds", "not_5v5", "unknown_map",
                      "duration_implausible")

# Linking tolerances in seconds (converted with the demo's tick rate).
_LINK_TOL_S   = 20.0               # grenade throw -> smoke / fire / flash detonation
_SEGMENT_GAP  = 8                  # ticks; a gap this long in one entity slot = new grenade
ID_SEGMENTS   = 16                 # grenade entity_id = awpy key * 16 + segment ordinal (< 16)


# ── small helpers ─────────────────────────────────────────────────────────────

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _frame(obj) -> pl.DataFrame | None:
    """A polars frame from an awpy attribute, or None when it is missing/empty-typed."""
    if obj is None:
        return None
    return obj if isinstance(obj, pl.DataFrame) else pl.DataFrame(obj)


def _attr(demo, name: str) -> pl.DataFrame | None:
    try:
        return _frame(getattr(demo, name))
    except AttributeError:
        return None


def _col(df: pl.DataFrame, *names: str, dtype=None) -> pl.Expr:
    """First existing column among `names`, else a typed null literal."""
    for n in names:
        if n in df.columns:
            return pl.col(n) if dtype is None else pl.col(n).cast(dtype, strict=False)
    return pl.lit(None, dtype=dtype if dtype is not None else STR)


def _side(df: pl.DataFrame, name: str) -> pl.Expr:
    """awpy side (2/3, 'terrorist'/'counter-terrorist', 'T'/'CT') -> 't' / 'ct' / null."""
    if name not in df.columns:
        return pl.lit(None, dtype=STR)
    dt = df.schema[name]
    if dt.is_integer():
        return (pl.when(pl.col(name) == 2).then(pl.lit("t"))
                .when(pl.col(name) == 3).then(pl.lit("ct"))
                .otherwise(pl.lit(None, dtype=STR)))
    s = pl.col(name).cast(STR).str.to_lowercase()
    return (pl.when(s.is_in(["t", "terrorist", "terrorists", "2"])).then(pl.lit("t"))
            .when(s.is_in(["ct", "counter-terrorist", "counterterrorist", "cts", "3"])).then(pl.lit("ct"))
            .otherwise(pl.lit(None, dtype=STR)))


def _map_values(expr: pl.Expr, mapping: dict) -> pl.Expr:
    """Lower-cased string mapping; values absent from `mapping` become null."""
    keys = [k for k, v in mapping.items() if v is not None]
    return expr.cast(STR).str.to_lowercase().replace_strict(
        keys, [mapping[k] for k in keys], default=None, return_dtype=STR)


def empty(table: str) -> pl.DataFrame:
    """A zero-row frame with the A.1 columns and dtypes of `table`."""
    return pl.DataFrame(schema=A1_SCHEMAS[table])


def conform(df: pl.DataFrame | None, table: str) -> pl.DataFrame:
    """Select exactly the A.1 columns of `table` in order, cast to their dtypes;
    a missing column becomes a typed all-null column. `A1_OPTIONAL` columns are kept
    (cast) when `df` has them."""
    schema = A1_SCHEMAS[table]
    if df is None or not df.width:
        return empty(table)
    exprs = [(pl.col(c).cast(dt, strict=False) if c in df.columns else pl.lit(None, dtype=dt)).alias(c)
             for c, dt in schema.items()]
    exprs += [pl.col(c).cast(dt, strict=False) for c, dt in A1_OPTIONAL.get(table, {}).items() if c in df.columns]
    return df.select(exprs)


def _ticks_per_s(tick_rate: int | None) -> int | None:
    return int(tick_rate) if tick_rate else None


# ── header / meta ─────────────────────────────────────────────────────────────

def _tick_rate_from_header(header: dict) -> tuple[int | None, float | None]:
    """(rounded, raw) ticks per second from playback_ticks / playback_time.
    `Demo.tick_rate` silently returns 64.0 when playback timing is missing, so it is
    never used: an unknown rate is None (and quarantined as tick_rate_not_128)."""
    ticks, secs = header.get("playback_ticks"), header.get("playback_time")
    try:
        raw = float(ticks) / float(secs)
    except (TypeError, ValueError, ZeroDivisionError):
        return None, None
    if raw <= 0:
        return None, None
    return int(round(raw)), raw


def _header_dict(demo) -> dict:
    h = getattr(demo, "header", None)
    return dict(h) if h else {}


def _str_or_none(v) -> str | None:
    return None if v is None or v == "" else str(v)


def _basic_meta(header: dict) -> dict:
    rate, raw = _tick_rate_from_header(header)
    secs = header.get("playback_time")
    return {
        "map":               _str_or_none(header.get("map_name")),
        "tick_rate":         rate,
        "tick_rate_raw":     raw,
        "game_version":      _str_or_none(header.get("build_num")),
        "server":            _str_or_none(header.get("server_name")),
        "demo_version_name": _str_or_none(header.get("demo_version_name")),
        "client_name":       _str_or_none(header.get("client_name")),
        "duration_s":        float(secs) if isinstance(secs, (int, float)) and secs > 0 else None,
    }


def _roster(demo) -> list[dict]:
    players = _attr(demo, "players")
    if players is None or players.height == 0:
        return []
    df = players.select(
        _col(players, "steamid", dtype=SID).alias("steamid"),
        _col(players, "name").cast(STR).alias("name"),
        _side(players, "side").alias("side"),
        _col(players, "team_clan_name").cast(STR).alias("team"),
    )
    return df.to_dicts()


def _roster_side_counts(roster: list[dict]) -> dict[str, int]:
    """Distinct human players per side. Bots (steamid 0) are not counted."""
    out = {"ct": 0, "t": 0}
    for side in out:
        out[side] = len({p["steamid"] for p in roster if p["side"] == side and p["steamid"]})
    return out


def header_info(path, *, demo=None, with_roster: bool = False) -> dict:
    """Cheap metadata for the quarantine smoke test (P1.4.1).

    Header only by default: map, tick_rate (None if the header has no playback
    timing), game_version, server, duration_s. `with_roster=True` also reads
    `demo.rounds` and `demo.players` (one decode pass, still far cheaper than ticks) so
    `n_rounds` and `side_counts` can be checked. Raises FileNotFoundError for a
    missing file and lets awpy's InvalidDemoError through for a corrupt one."""
    p = Path(path)
    if demo is None:
        if not p.is_file():
            raise FileNotFoundError(f"demo not found: {p}")
        import awpy                      # imported lazily: tests never need the backend
        demo = awpy.Demo(p)
    info = _basic_meta(_header_dict(demo))
    if with_roster:
        rounds = _attr(demo, "rounds")
        info["n_rounds"] = int(rounds.height) if rounds is not None else None
        roster = _roster(demo)
        info["players"]     = roster
        info["side_counts"] = _roster_side_counts(roster) if roster else None
    return info


def _first_finite(*vals):
    """First value that is a finite number (not None / NaN / inf), else None. The R8
    rule: `or` is not a null-coalesce over numerics (0.0 is falsy, NaN is truthy)."""
    for v in vals:
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return None


def quarantine_check(meta: dict, known_maps=None) -> str | None:
    """First failing P1.4.1 rule as a reason string, or None when the demo passes.
    Keys absent from `meta` are not judged (a header-only dict has no n_rounds), but a
    present-and-None tick rate or map fails: it could not be verified."""
    if known_maps is None:
        from engine.maps.assets import ACTIVE_POOL
        known_maps = ACTIVE_POOL
    if "tick_rate" in meta:
        rate = _first_finite(meta.get("tick_rate_raw"), meta.get("tick_rate"))
        if rate is None or abs(rate - EXPECTED_TICK_RATE) > TICK_RATE_TOL:
            return "tick_rate_not_128"
    if "n_rounds" in meta and (meta["n_rounds"] is None or meta["n_rounds"] < MIN_ROUNDS):
        return "too_few_rounds"
    if "side_counts" in meta:
        sc = meta["side_counts"] or {}
        if sc.get("ct") != PLAYERS_PER_SIDE or sc.get("t") != PLAYERS_PER_SIDE:
            return "not_5v5"
    if "map" in meta and meta["map"] not in known_maps:
        return "unknown_map"
    if "duration_s" in meta:
        d = meta["duration_s"]
        if d is None or not MIN_DURATION_S <= d <= MAX_DURATION_S:
            return "duration_implausible"
    return None


# ── rounds and round windows ──────────────────────────────────────────────────

def _rounds_base(demo) -> pl.DataFrame:
    """awpy rounds -> round_num, ticks, winner_side, end_reason, knife flag, window."""
    r = _attr(demo, "rounds")
    if r is None or r.height == 0:
        return pl.DataFrame(schema={"round_num": ROUND, "start_tick": TICK, "freeze_end_tick": TICK,
                                    "end_tick": TICK, "official_end_tick": TICK, "winner_side": STR,
                                    "end_reason": STR, "is_knife": BOOL, "_lo": TICK, "_hi": TICK})
    winner = _side(r, "winner_side")
    if "winner" in r.columns:
        winner = pl.coalesce(winner, _side(r, "winner"))
    df = r.select(
        _col(r, "round_num", dtype=ROUND).alias("round_num"),
        _col(r, "start_tick", dtype=TICK).alias("start_tick"),
        _col(r, "freeze_end_tick", dtype=TICK).alias("freeze_end_tick"),
        _col(r, "end_tick", dtype=TICK).alias("end_tick"),
        _col(r, "official_end_tick", dtype=TICK).alias("official_end_tick"),
        winner.alias("winner_side"),
        _col(r, "reason_name", "reason").cast(STR).alias("end_reason"),
        _col(r, "is_knife_round", dtype=BOOL).alias("is_knife"),
    ).sort("round_num")
    return df.with_columns(
        pl.coalesce("start_tick", "freeze_end_tick", "end_tick").alias("_lo"),
        pl.coalesce("official_end_tick", "end_tick").alias("_hi"),
    )


def _assign_round(df: pl.DataFrame, windows: pl.DataFrame, tick_col: str = "tick",
                  keep_round: bool = True) -> pl.DataFrame:
    """Attach round_num by tick; drop rows outside every round window
    [_lo, _hi] (warmup, the gap after official end). Later round wins on a tie."""
    src = df.drop("round_num") if "round_num" in df.columns else df
    if windows.height == 0:
        src = src.head(0)
        return src.with_columns(pl.lit(None, dtype=ROUND).alias("round_num")) if keep_round else src
    w = windows.select("_lo", "_hi", pl.col("round_num").alias("_rn")).drop_nulls("_lo").sort("_lo")
    out = (src.with_columns(pl.col(tick_col).cast(TICK))
              .sort(tick_col)
              .join_asof(w, left_on=tick_col, right_on="_lo", strategy="backward")
              .filter(pl.col("_rn").is_not_null() & (pl.col(tick_col) <= pl.col("_hi"))))
    out = out.rename({"_rn": "round_num"}) if keep_round else out.drop("_rn")
    return out.drop([c for c in ("_lo", "_hi") if c in out.columns])


# ── ticks ─────────────────────────────────────────────────────────────────────

def _snapshots(demo, windows: pl.DataFrame) -> pl.DataFrame:
    """Every player at every tick from the first round start to the last round end,
    in one call (awpy decodes keyframe segments in parallel), then cut to rounds."""
    if windows.height == 0:
        return pl.DataFrame()
    lo, hi = windows["_lo"].min(), windows["_hi"].max()
    snap = _frame(demo.snapshots(start_tick=int(lo), end_tick=int(hi)))
    if snap is None or snap.height == 0:
        return pl.DataFrame()
    snap = snap.with_columns(pl.col("tick").cast(TICK), pl.col("steamid").cast(SID, strict=False))
    return _assign_round(snap, windows)


def _cumulative(events: pl.DataFrame, value: str | None, name: str) -> pl.DataFrame:
    """Per (round_num, steamid): running count (value None) or sum of `value` at each
    event tick, collapsed to one row per tick so an as-of join is unambiguous."""
    if events.height == 0:
        return pl.DataFrame(schema={"tick": TICK, "round_num": ROUND, "steamid": SID, name: pl.Int32})
    agg = pl.len() if value is None else pl.col(value).fill_null(0).sum()
    per_tick = (events.group_by(["round_num", "steamid", "tick"]).agg(agg.alias("_v"))
                      .sort("tick"))
    return per_tick.with_columns(
        pl.col("_v").cum_sum().over(["round_num", "steamid"]).cast(pl.Int32).alias(name)
    ).select("tick", pl.col("round_num").cast(ROUND), pl.col("steamid").cast(SID), name)


def _asof_by_player(left: pl.DataFrame, right: pl.DataFrame, **kw) -> pl.DataFrame:
    # Globally sorted by tick, hence sorted within every group (polars cannot check that).
    return left.sort("tick").join_asof(right.sort("tick"), on="tick", by=["round_num", "steamid"],
                                       strategy="backward", check_sortedness=False, **kw)


def _plant_intervals(bomb_raw: pl.DataFrame | None, windows: pl.DataFrame) -> pl.DataFrame | None:
    """[start_plant, finish/interrupt) per player; None when the bomb frame is unusable."""
    if bomb_raw is None or "event" not in bomb_raw.columns:
        return None
    ev = pl.col("event").cast(STR).str.to_lowercase()
    starts = bomb_raw.filter(ev == "start_plant").select("tick", "round_num", "steamid")
    ends   = bomb_raw.filter(ev.is_in(["finish_plant", "interrupt_plant"])).select(
        pl.col("tick").alias("_pe"), "round_num", "steamid")
    schema = {"tick": TICK, "round_num": ROUND, "steamid": SID, "_pe": TICK}
    if starts.height == 0:
        return pl.DataFrame(schema=schema)
    iv = (starts.sort("tick")
                .join_asof(ends.sort("_pe"), left_on="tick", right_on="_pe",
                           by=["round_num", "steamid"], strategy="forward", check_sortedness=False)
                .join(windows.select("round_num", "_hi"), on="round_num", how="left")
                .with_columns(pl.coalesce("_pe", "_hi").alias("_pe"))
                .select(pl.col("tick").alias("_ps"), "round_num", "steamid", "_pe"))
    return iv.with_columns(pl.col("_ps").alias("tick"))


def _remaining_flash(t: pl.DataFrame, flashes: pl.DataFrame | None, tick_rate: int | None) -> pl.DataFrame:
    """flash_duration = REMAINING blind seconds at each tick (schema.py contract):
    latest blind of that player, duration minus elapsed time, floored at 0.0. awpy's
    own `flash_duration` is the onset duration, so it is not used. Null when the
    blinds frame or the tick rate is unavailable."""
    if flashes is None or not tick_rate:
        return t.with_columns(pl.lit(None, dtype=POS).alias("flash_duration"))
    b = (flashes.select(pl.col("detonate_tick").alias("_bt"), pl.col("player_steamid").alias("steamid"),
                        pl.col("blind_duration").alias("_bd"))
                .drop_nulls(["_bt", "steamid"]).sort("_bt"))
    t = t.sort("tick").join_asof(b, left_on="tick", right_on="_bt", by="steamid",
                                 strategy="backward", check_sortedness=False)
    rem = (pl.col("_bd") - (pl.col("tick") - pl.col("_bt")) / tick_rate).clip(lower_bound=0.0)
    return t.with_columns(rem.fill_null(0.0).cast(POS).alias("flash_duration")).drop("_bt", "_bd")


def _dead_rows(t: pl.DataFrame) -> pl.DataFrame:
    """Dead players keep their rows (schema.py contract): health 0, last alive
    position, no active weapon, empty inventory, equipment 0, not blind.
    A snapshot table whose `health` is null on every row has no health column at all
    (an awpy rename): refused here rather than parsed as everyone dead (MED-13)."""
    if t.height and t["health"].null_count() == t.height:
        raise ValueError("ticks.health is null on every row: the snapshot has no health column; "
                         "refusing to parse every player as dead")
    alive = pl.col("is_alive")
    grp = ["round_num", "steamid"]
    t = t.sort("tick")
    return t.with_columns(
        *[pl.when(alive).then(pl.col(c)).otherwise(None).forward_fill().over(grp).alias(c) for c in ("x", "y", "z")],
        pl.when(alive).then(pl.col("health")).otherwise(0).alias("health"),
        pl.when(alive).then(pl.col("active_weapon")).otherwise(None).alias("active_weapon"),
        pl.when(alive).then(pl.col("inventory")).otherwise(pl.col("inventory").list.head(0)).alias("inventory"),
        pl.when(alive).then(pl.col("equipment_value")).otherwise(0).alias("equipment_value"),
        pl.when(alive).then(pl.col("flash_duration")).otherwise(0.0).alias("flash_duration"),
    )


def _build_ticks(snap: pl.DataFrame, kills: pl.DataFrame, damages: pl.DataFrame,
                 plants: pl.DataFrame | None, flashes: pl.DataFrame | None,
                 tick_rate: int | None) -> pl.DataFrame:
    if snap.height == 0:
        return empty("ticks")
    # Rows without a ct/t side (spectators, casters) are not players of the match.
    s = snap.filter(_side(snap, "side").is_not_null())
    health = _col(s, "health", dtype=pl.Int32)
    t = s.select(
        "tick", "round_num", "steamid",
        _col(s, "name").cast(STR).alias("name"),
        _side(s, "side").alias("side"),
        _col(s, "x", dtype=POS).alias("x"), _col(s, "y", dtype=POS).alias("y"),
        _col(s, "z", dtype=POS).alias("z"),
        _col(s, "yaw", dtype=POS).alias("yaw"), _col(s, "pitch", dtype=POS).alias("pitch"),
        _col(s, "velocity_x", dtype=POS).alias("vx"), _col(s, "velocity_y", dtype=POS).alias("vy"),
        _col(s, "velocity_z", dtype=POS).alias("vz"),
        health.alias("health"), _col(s, "armor").alias("armor"),
        _col(s, "has_helmet", dtype=BOOL).alias("has_helmet"),
        _col(s, "has_defuser", dtype=BOOL).alias("has_defuser"),
        _col(s, "has_bomb", dtype=BOOL).alias("has_c4"),
        (health > 0).fill_null(False).alias("is_alive"),
        _col(s, "active_weapon").cast(STR).alias("active_weapon"),
        _col(s, "inventory").cast(pl.List(STR), strict=False).alias("inventory"),
        _col(s, "money").alias("money"),
        _col(s, "equipment_value").alias("equipment_value"),
        _col(s, "is_scoped", dtype=BOOL).alias("is_scoped"),
        _col(s, "is_crouched", dtype=BOOL).alias("is_ducking"),
        _col(s, "is_walking", dtype=BOOL).alias("is_walking"),
        _col(s, "is_reloading", dtype=BOOL).alias("is_reloading"),
        _col(s, "is_defusing", dtype=BOOL).alias("is_defusing"),
        # spotted_by: not obtainable from awpy 3.1.0 (see parser notes) -> null.
        pl.lit(None, dtype=pl.List(SID)).alias("spotted_by"),
    )

    # Running kills / damage of each player within the round, enemies only.
    enemy_k = kills.filter(pl.col("attacker_steamid").is_not_null()
                           & (pl.col("team_kill") == False)       # noqa: E712  (null = unknown, excluded)
                           & (pl.col("attacker_steamid") != pl.col("victim_steamid")))
    k_cum = _cumulative(enemy_k.select("tick", "round_num", pl.col("attacker_steamid").alias("steamid")),
                        None, "kills_round")
    enemy_d = damages.filter(pl.col("attacker_steamid").is_not_null()
                             & (pl.col("_team_damage") == False)  # noqa: E712
                             & (pl.col("attacker_steamid") != pl.col("victim_steamid")))
    d_cum = _cumulative(enemy_d.select("tick", "round_num", pl.col("attacker_steamid").alias("steamid"),
                                       pl.col("dmg_health").cast(pl.Int64)),
                        "dmg_health", "damage_round")
    t = _asof_by_player(t, k_cum)
    t = _asof_by_player(t, d_cum)
    t = t.with_columns(pl.col("kills_round").fill_null(0), pl.col("damage_round").fill_null(0))

    if plants is None:
        t = t.with_columns(pl.lit(None, dtype=BOOL).alias("is_planting"))
    else:
        t = _asof_by_player(t, plants.select("tick", pl.col("round_num").cast(ROUND),
                                             pl.col("steamid").cast(SID), "_ps", "_pe"))
        t = t.with_columns((pl.col("_ps").is_not_null() & (pl.col("tick") < pl.col("_pe")))
                           .alias("is_planting")).drop("_ps", "_pe")
    t = _dead_rows(_remaining_flash(t, flashes, tick_rate))
    return conform(t.sort(["tick", "steamid"]), "ticks")


# ── events ────────────────────────────────────────────────────────────────────

def _blind_lookup(snap: pl.DataFrame) -> pl.DataFrame | None:
    if snap.height == 0 or "is_blinded" not in snap.columns:
        return None
    return snap.select("tick", "steamid", pl.col("is_blinded").cast(BOOL))


def _build_kills(demo, windows: pl.DataFrame, snap: pl.DataFrame) -> pl.DataFrame:
    k = _attr(demo, "kills")
    if k is None or k.height == 0:
        return empty("kills")
    k = _assign_round(k.with_columns(pl.col("tick").cast(TICK)), windows)
    att, vic, ass = (_col(k, f"{p}_steamid", dtype=SID) for p in ("attacker", "victim", "assister"))
    if "assistedflash" in k.columns:
        flash = pl.col("assistedflash").cast(BOOL).fill_null(False)
        assister       = pl.when(flash).then(pl.lit(None, dtype=SID)).otherwise(ass)
        flash_assister = pl.when(flash).then(ass).otherwise(pl.lit(None, dtype=SID))
    else:
        assister, flash_assister = ass, pl.lit(None, dtype=SID)
    a_side, v_side = _side(k, "attacker_side"), _side(k, "victim_side")
    pos = {f"{p}_{c}": _col(k, f"{p}_{c}", dtype=POS) for p in ("attacker", "victim") for c in "xyz"}
    out = k.select(
        "tick", "round_num",
        att.alias("attacker_steamid"), vic.alias("victim_steamid"),
        assister.alias("assister_steamid"), flash_assister.alias("flash_assister_steamid"),
        _col(k, "weapon").cast(STR).alias("weapon"),
        _col(k, "headshot", dtype=BOOL).alias("headshot"),
        _col(k, "penetrated", dtype=pl.Int16).alias("penetrated"),
        _col(k, "thrusmoke", "through_smoke", dtype=BOOL).alias("through_smoke"),
        _col(k, "attackerblind", "attacker_blind", dtype=BOOL).alias("_att_blind_ev"),
        _col(k, "noscope", dtype=BOOL).alias("noscope"),
        *([pl.col("is_trade").cast(BOOL)] if "is_trade" in k.columns else []),
        *[e.alias(n) for n, e in pos.items()],
        pl.when(att.is_null()).then(pl.lit(False))
          .when(a_side.is_null() | v_side.is_null()).then(pl.lit(None, dtype=BOOL))
          .otherwise((a_side == v_side) & (att != vic)).alias("team_kill"),
    )
    # Distance in Hammer units from the two positions (CS2's own `distance` key is in
    # metres and not always present, so it is never used).
    out = out.with_columns(
        ((pl.col("attacker_x") - pl.col("victim_x")) ** 2
         + (pl.col("attacker_y") - pl.col("victim_y")) ** 2
         + (pl.col("attacker_z") - pl.col("victim_z")) ** 2).sqrt().cast(POS).alias("distance"))
    blind = _blind_lookup(snap)
    if blind is None:
        out = out.with_columns(pl.col("_att_blind_ev").alias("attacker_blind"),
                               pl.lit(None, dtype=BOOL).alias("victim_blind"))
    else:
        out = (out.join(blind.rename({"steamid": "attacker_steamid", "is_blinded": "_ab"}),
                        on=["tick", "attacker_steamid"], how="left")
                  .join(blind.rename({"steamid": "victim_steamid", "is_blinded": "victim_blind"}),
                        on=["tick", "victim_steamid"], how="left")
                  .with_columns(pl.coalesce("_att_blind_ev", "_ab").alias("attacker_blind")))
    return conform(out.sort("tick"), "kills")


def _build_damages(demo, windows: pl.DataFrame) -> pl.DataFrame:
    """A.1 damages plus a private `_team_damage` column (dropped by conform)."""
    d = _attr(demo, "damages")
    if d is None or d.height == 0:
        return empty("damages").with_columns(pl.lit(None, dtype=BOOL).alias("_team_damage"))
    d = _assign_round(d.with_columns(pl.col("tick").cast(TICK)), windows)
    a_side, v_side = _side(d, "attacker_side"), _side(d, "victim_side")
    return d.select(
        "tick", "round_num",
        _col(d, "attacker_steamid", dtype=SID).alias("attacker_steamid"),
        _col(d, "victim_steamid", dtype=SID).alias("victim_steamid"),
        _col(d, "weapon").cast(STR).alias("weapon"),
        _col(d, "dmg_health", dtype=pl.Int16).alias("dmg_health"),
        _col(d, "dmg_armor", dtype=pl.Int16).alias("dmg_armor"),
        _col(d, "hitgroup_name", "hitgroup").cast(STR).alias("hitgroup"),
        (a_side == v_side).alias("_team_damage"),
    ).sort("tick")


def _entity_key(df: pl.DataFrame) -> pl.Expr:
    """One integer per entity: slot + generation (awpy: use id and serial together)."""
    eid = _col(df, "entity_id", dtype=pl.Int64)
    if "entity_serial" not in df.columns:
        return eid
    ser = pl.col("entity_serial").cast(pl.Int64, strict=False)
    return pl.when(ser.is_null()).then(eid).otherwise(ser * 65536 + eid)


def _collapse_grenades(demo, windows: pl.DataFrame) -> pl.DataFrame:
    """Trajectory rows -> one row per thrown grenade (first/last tick, last position)."""
    schema = {"entity_id": pl.Int64, "thrower_steamid": SID, "type": STR, "throw_tick": TICK,
              "last_tick": TICK, "rest_tick": TICK, "lx": POS, "ly": POS, "lz": POS}
    g = _attr(demo, "grenades")
    if g is None or g.height == 0:
        return pl.DataFrame(schema=schema)
    g = g.with_columns(pl.col("tick").cast(TICK), _entity_key(g).alias("_key"),
                       _map_values(_col(g, "type", "grenade_type"), _GRENADE_TYPE_MAP).alias("_type"),
                       _col(g, "thrower_steamid", dtype=SID).alias("_thrower"))
    new = ((pl.col("_key") != pl.col("_key").shift(1))
           | ((pl.col("tick") - pl.col("tick").shift(1)) > _SEGMENT_GAP)).fill_null(True)
    g = g.sort(["_key", "tick"]).with_columns(new.cum_sum().alias("_seg"),
                                              (new.cast(pl.Int64).cum_sum().over("_key") - 1).alias("_ord"))
    x, y, z = (_col(g, c, dtype=POS) for c in ("x", "y", "z"))
    at_rest = (x == x.last()) & (y == y.last()) & (z == z.last())
    out = g.group_by("_seg", maintain_order=True).agg(
        # one id per SEGMENT: key * 16 + ordinal, so a slot reused without a serial (or a
        # trajectory with a hole) never shares an id with its predecessor (MED-12, 2026-10-19)
        (pl.col("_key").first() * ID_SEGMENTS + pl.col("_ord").first().clip(upper_bound=ID_SEGMENTS - 1))
        .alias("entity_id"),
        pl.col("_thrower").drop_nulls().first().alias("thrower_steamid"),
        pl.col("_type").drop_nulls().first().alias("type"),
        pl.col("tick").min().alias("throw_tick"),
        pl.col("tick").max().alias("last_tick"),
        # first tick at the final position: when a decoy came to rest (MED-08)
        pl.col("tick").filter(at_rest).min().alias("rest_tick"),
        x.last().alias("lx"), y.last().alias("ly"), z.last().alias("lz"),
    ).drop("_seg")
    out = _assign_round(out, windows, "throw_tick", keep_round=False)
    return out.select(list(schema)).sort("throw_tick")


def _link(events: pl.DataFrame, gren: pl.DataFrame, types: list[str], tick_col: str,
          thrower_col: str, tol: int | None) -> pl.DataFrame:
    """For each event, the entity_id of the thrower's latest grenade of `types` thrown
    at or before the event tick (within `tol` ticks) -> column `_gkey`."""
    cand = (gren.filter(pl.col("type").is_in(types))
                .select(pl.col("throw_tick").alias("_tt"), pl.col("thrower_steamid").alias(thrower_col),
                        pl.col("entity_id").alias("_gkey"))
                .drop_nulls([thrower_col]).sort("_tt"))
    ev = events.with_columns(pl.col(tick_col).cast(TICK), pl.col(thrower_col).cast(SID, strict=False))
    if ev.height == 0:
        return ev.with_columns(pl.lit(None, dtype=pl.Int64).alias("_gkey"))
    kw = {"tolerance": tol} if tol else {}
    return (ev.sort(tick_col)
              .join_asof(cand, left_on=tick_col, right_on="_tt", by=thrower_col,
                         strategy="backward", check_sortedness=False, **kw)
              .drop("_tt"))


def _area_frame(demo, name: str, windows: pl.DataFrame, gren: pl.DataFrame, types: list[str],
                tol: int | None) -> pl.DataFrame:
    """smokes / fires -> entity_id (linked grenade id when found), ticks, position."""
    f = _attr(demo, name)
    cols = {"_own": pl.Int64, "start_tick": TICK, "end_tick": TICK, "x": POS, "y": POS, "z": POS,
            "thrower_steamid": SID, "_ftype": STR, "_gkey": pl.Int64}
    if f is None or f.height == 0:
        return pl.DataFrame(schema=cols)
    f = f.select(
        _entity_key(f).alias("_own"),
        _col(f, "start_tick", dtype=TICK).alias("start_tick"),
        _col(f, "end_tick", dtype=TICK).alias("end_tick"),
        _col(f, "x", dtype=POS).alias("x"), _col(f, "y", dtype=POS).alias("y"),
        _col(f, "z", dtype=POS).alias("z"),
        _col(f, "thrower_steamid", dtype=SID).alias("thrower_steamid"),
        _map_values(_col(f, "type"), _GRENADE_TYPE_MAP).alias("_ftype"),
    )
    f = _assign_round(f, windows, "start_tick", keep_round=False)
    return _link(f, gren, types, "start_tick", "thrower_steamid", tol).select(list(cols))


def _build_utility(demo, windows: pl.DataFrame, tick_rate: int | None):
    """grenades, smokes, infernos, flashes, all sharing grenade entity ids."""
    tol  = int(_LINK_TOL_S * tick_rate) if tick_rate else None
    gren = _collapse_grenades(demo, windows)
    smk  = _area_frame(demo, "smokes", windows, gren, ["smoke"], tol)
    fir  = _area_frame(demo, "fires", windows, gren, ["molotov", "incendiary"], tol)

    smokes = smk.with_columns(pl.coalesce("_gkey", "_own").alias("entity_id"))
    infern = fir.with_columns(pl.coalesce("_gkey", "_own").alias("entity_id"),
                              # awpy 3.1.0 gives one row per fire, no flame extents -> null.
                              pl.lit(None, dtype=POS).alias("extent_radius"))

    # Detonation / expiry of smokes and molotovs from their linked cloud / fire.
    link = pl.concat([
        smk.drop_nulls("_gkey").select("_gkey", "start_tick", "end_tick", "x", "y", "z", "_ftype"),
        fir.drop_nulls("_gkey").select("_gkey", "start_tick", "end_tick", "x", "y", "z", "_ftype"),
    ]).unique("_gkey", keep="first")
    g = gren.join(link.rename({"_gkey": "entity_id"}), on="entity_id", how="left")
    g = g.with_columns(
        pl.when(pl.col("type").is_in(["molotov", "incendiary"]) & pl.col("_ftype").is_in(["molotov", "incendiary"]))
          .then(pl.col("_ftype")).otherwise(pl.col("type")).alias("type"),
        # smoke / molotov: linked cloud; decoy: at rest -> removed (MED-08); flash / HE: last tick
        pl.coalesce("start_tick",
                    pl.when(pl.col("type") == "decoy").then(pl.coalesce("rest_tick", "last_tick"))
                      .otherwise(pl.col("last_tick"))).alias("detonate_tick"),
        pl.when(pl.col("type") == "decoy").then(pl.col("last_tick")).otherwise(pl.col("end_tick")).alias("expire_tick"),
        pl.coalesce("x", "lx").alias("x"), pl.coalesce("y", "ly").alias("y"),
        pl.coalesce("z", "lz").alias("z"),
    )

    b = _attr(demo, "blinds")
    if b is None or b.height == 0:
        flashes = empty("flashes")
    else:
        bl = b.select(pl.col("tick").cast(TICK).alias("detonate_tick"),
                      _col(b, "attacker_steamid", dtype=SID).alias("attacker_steamid"),
                      _col(b, "victim_steamid", dtype=SID).alias("player_steamid"),
                      _col(b, "duration", dtype=POS).alias("blind_duration"))
        bl = _assign_round(bl, windows, "detonate_tick", keep_round=False)
        bl = _link(bl, gren, ["flash"], "detonate_tick", "attacker_steamid", tol)
        flashes = conform(bl.rename({"_gkey": "entity_id"}).sort("detonate_tick"), "flashes")
    return (conform(g.sort("throw_tick"), "grenades"), conform(smokes.sort("start_tick"), "smokes"),
            conform(infern.sort("start_tick"), "infernos"), flashes)


def _bomb_raw(demo, windows: pl.DataFrame) -> pl.DataFrame | None:
    b = _attr(demo, "bomb")
    if b is None:
        return None
    if b.height == 0:
        return pl.DataFrame(schema={"tick": TICK, "round_num": ROUND, "event": STR, "steamid": SID,
                                    "site": STR, "x": POS, "y": POS, "z": POS})
    b = b.select(pl.col("tick").cast(TICK), _col(b, "event", "action").cast(STR).alias("event"),
                 _col(b, "steamid", dtype=SID).alias("steamid"),
                 _col(b, "bombsite", "site").cast(STR).alias("site"),
                 _col(b, "x", dtype=POS).alias("x"), _col(b, "y", dtype=POS).alias("y"),
                 _col(b, "z", dtype=POS).alias("z"))
    return _assign_round(b, windows)


def _defuse_events(snap: pl.DataFrame, completes: pl.DataFrame) -> pl.DataFrame:
    """defuse_start / defuse_abort from the per-tick is_defusing flag: a rising edge
    starts a defuse; a falling edge with no defuse_complete within 4 ticks is an abort."""
    cols = ["tick", "round_num", "event", "steamid", "has_kit", "x", "y", "z"]
    if snap.height == 0 or "is_defusing" not in snap.columns:
        return pl.DataFrame(schema={c: A1_SCHEMAS["bomb"][c] for c in cols})
    grp = ["round_num", "steamid"]
    d = (snap.select("tick", "round_num", "steamid", pl.col("is_defusing").cast(BOOL).fill_null(False).alias("_d"),
                     _col(snap, "has_defuser", dtype=BOOL).alias("has_kit"),
                     _col(snap, "x", dtype=POS).alias("x"), _col(snap, "y", dtype=POS).alias("y"),
                     _col(snap, "z", dtype=POS).alias("z"))
             .sort(["round_num", "steamid", "tick"])
             .with_columns(pl.col("_d").shift(1).over(grp).fill_null(False).alias("_prev"),
                           pl.col("_d").shift(-1).over(grp).fill_null(False).alias("_next")))
    starts = d.filter(pl.col("_d") & ~pl.col("_prev")).with_columns(pl.lit("defuse_start").alias("event"))
    lasts  = d.filter(pl.col("_d") & ~pl.col("_next"))
    if completes.height:
        c = completes.select("round_num", pl.col("tick").alias("_ct"))
        lasts = (lasts.join(c, on="round_num", how="left")
                      .with_columns(((pl.col("_ct") - pl.col("tick")).abs() <= 4).fill_null(False).alias("_done"))
                      .group_by(["round_num", "steamid", "tick"], maintain_order=True)
                      .agg(pl.all().exclude("_done", "_ct").first(), pl.col("_done").any())
                      .filter(~pl.col("_done")))
    aborts = lasts.with_columns((pl.col("tick") + 1).alias("tick"), pl.lit("defuse_abort").alias("event"),
                                pl.lit(None, dtype=BOOL).alias("has_kit"))
    return pl.concat([starts.select(cols), aborts.select(cols)], how="vertical_relaxed")


def _build_bomb(bomb_raw: pl.DataFrame | None, rounds: pl.DataFrame, snap: pl.DataFrame,
                dropped: dict) -> pl.DataFrame:
    if bomb_raw is None:
        return empty("bomb")
    mapped = bomb_raw.with_columns(_map_values(pl.col("event"), _BOMB_EVENT_MAP).alias("_ev"))
    n_drop = mapped.filter(pl.col("_ev").is_null()).height
    if n_drop:
        dropped["bomb"] = dropped.get("bomb", 0) + n_drop
    base = mapped.filter(pl.col("_ev").is_not_null()).with_columns(pl.col("_ev").alias("event")).drop("_ev")
    base = base.with_columns(pl.lit(None, dtype=BOOL).alias("has_kit"))
    completes = base.filter(pl.col("event") == "defuse_complete")
    if snap.height and "has_defuser" in snap.columns and completes.height:
        kit = snap.select("tick", "steamid", pl.col("has_defuser").cast(BOOL).alias("_kit"))
        base = (base.join(kit, on=["tick", "steamid"], how="left")
                    .with_columns(pl.when(pl.col("event") == "defuse_complete").then(pl.col("_kit"))
                                    .otherwise(pl.col("has_kit")).alias("has_kit")).drop("_kit"))
    defuses = _defuse_events(snap, completes)
    defuses = defuses.join(base.filter(pl.col("event") == "plant_complete")
                               .group_by("round_num").agg(pl.col("site").last()),
                           on="round_num", how="left")
    plants = (base.filter(pl.col("event") == "plant_complete").group_by("round_num")
                  .agg(pl.col("site").last(), pl.col("x").last(), pl.col("y").last(), pl.col("z").last()))
    explode = (rounds.filter(pl.col("end_reason").str.to_lowercase().is_in(list(_EXPLODE_REASONS)))
                     .select(pl.col("end_tick").alias("tick"), "round_num")
                     .join(plants, on="round_num", how="left")
                     .with_columns(pl.lit("explode").alias("event"), pl.lit(None, dtype=SID).alias("steamid"),
                                   pl.lit(None, dtype=BOOL).alias("has_kit")))
    cols = list(A1_SCHEMAS["bomb"])
    parts = [conform(p, "bomb").select(cols) for p in (base, defuses, explode)]
    return pl.concat(parts).sort(["tick", "event"])


def _simple(f: pl.DataFrame | None, windows: pl.DataFrame, table: str, select) -> pl.DataFrame:
    """Tick-keyed event frame -> A.1 `table`: cut to round windows, `select(f)` columns."""
    if f is None or f.height == 0:
        return empty(table)
    f = _assign_round(f.with_columns(pl.col("tick").cast(TICK)), windows, keep_round=False)
    return conform(f.select(select(f)).sort("tick"), table)


def _build_items(demo, windows: pl.DataFrame, dropped: dict) -> pl.DataFrame:
    f = _attr(demo, "item_events")
    if f is None or f.height == 0:
        return empty("items")
    f = f.with_columns(_col(f, "action", "event").cast(STR).str.to_lowercase().alias("_act"))
    keep = pl.col("_act").is_in(list(ITEM_EVENTS)).fill_null(False)
    n_drop = f.filter(~keep).height
    if n_drop:
        dropped["items"] = dropped.get("items", 0) + n_drop        # purchases: no A.1 name
    return _simple(f.filter(keep), windows, "items",
                   lambda d: ["tick", _col(d, "steamid", dtype=SID).alias("steamid"),
                              pl.col("_act").alias("event"), _col(d, "item").cast(STR).alias("item")])


_FOOTSTEP_SID = ("user_steamid", "userid_steamid", "player_steamid", "steamid")


def _build_footsteps(demo, windows: pl.DataFrame, snap: pl.DataFrame) -> pl.DataFrame:
    """From the raw `player_footstep` event when awpy exposes it with a resolved steamid;
    position comes from the same tick's snapshot. Otherwise an empty A.1 frame."""
    events = getattr(demo, "events", None)
    try:
        if events is None or "player_footstep" not in events:
            return empty("footsteps")
        fs = _frame(events["player_footstep"])
    except (KeyError, AttributeError):
        return empty("footsteps")
    sid = next((c for c in _FOOTSTEP_SID if c in fs.columns), None)
    if fs is None or fs.height == 0 or sid is None:
        return empty("footsteps")
    fs = fs.select(pl.col("tick").cast(TICK), pl.col(sid).cast(SID, strict=False).alias("steamid"))
    fs = _assign_round(fs.drop_nulls("steamid"), windows, keep_round=False)
    if snap.height:
        fs = fs.join(snap.select("tick", "steamid", "x", "y", "z"), on=["tick", "steamid"], how="left")
    return conform(fs.sort("tick"), "footsteps")


# ── rounds (derived columns) ──────────────────────────────────────────────────

def _side_rosters(snap: pl.DataFrame, rounds: pl.DataFrame) -> pl.DataFrame:
    """Per round: CT and T steamid lists at the round's first tick at/after freeze end."""
    if snap.height == 0:
        return pl.DataFrame(schema={"round_num": ROUND, "ct": pl.List(SID), "t": pl.List(SID)})
    ref = rounds.select("round_num", pl.coalesce("freeze_end_tick", "_lo").alias("_ref"))
    s = (snap.select("tick", "round_num", "steamid", _side(snap, "side").alias("_side"))
             .join(ref, on="round_num", how="left")
             .filter(pl.col("tick") >= pl.col("_ref")))
    first = s.group_by("round_num").agg(pl.col("tick").min().alias("_t0"))
    s = s.join(first, on="round_num").filter(pl.col("tick") == pl.col("_t0"))
    s = s.filter(pl.col("steamid").is_not_null() & (pl.col("steamid") != 0))
    return s.group_by("round_num").agg(
        pl.col("steamid").filter(pl.col("_side") == "ct").unique().alias("ct"),
        pl.col("steamid").filter(pl.col("_side") == "t").unique().alias("t"),
    )


def _scores(rounds: pl.DataFrame, rosters: pl.DataFrame) -> tuple[list, list]:
    """Score of the team on each side before each round. Teams are tracked by roster
    overlap with the first round's CT side, so the halftime swap is handled; knife
    rounds score nothing. Unknown rosters -> None."""
    by_round = {r["round_num"]: (set(r["ct"] or []), set(r["t"] or [])) for r in rosters.to_dicts()}
    team_a: set | None = None
    wins = {"a": 0, "b": 0}
    ct_before, t_before = [], []
    for r in rounds.to_dicts():
        ct, t = by_round.get(r["round_num"], (set(), set()))
        if team_a is None and ct:
            team_a = ct
        if team_a is None or not (ct or t):
            ct_before.append(None)
            t_before.append(None)
            continue
        a_is_ct = len(ct & team_a) >= len(t & team_a)
        ct_team, t_team = ("a", "b") if a_is_ct else ("b", "a")
        ct_before.append(wins[ct_team])
        t_before.append(wins[t_team])
        if not r.get("is_knife") and r["winner_side"] in ("ct", "t"):
            wins[ct_team if r["winner_side"] == "ct" else t_team] += 1
    return ct_before, t_before


def _is_knife_item(name) -> bool:
    from engine.state.players import weapon_class
    return weapon_class(name) == "knife"


def knife_round_nums(ticks: pl.DataFrame, rounds: pl.DataFrame) -> list[int]:
    """Rounds where, at the first tick at or after freeze end, every alive player holds
    nothing but knives (or an empty inventory). awpy 3.1 exposes no `is_knife_round`,
    so the knife round is detected from the data (MED-35, 2026-10-19). Rounds without
    any alive player at that tick are not judged."""
    if ticks.height == 0 or rounds.height == 0 or "inventory" not in ticks.columns:
        return []
    out = []
    for r in rounds.select("round_num", "freeze_end_tick").to_dicts():
        fe = r["freeze_end_tick"]
        if fe is None:
            continue
        rows = ticks.filter((pl.col("round_num") == r["round_num"]) & (pl.col("tick") >= fe) & pl.col("is_alive"))
        if rows.height == 0:
            continue
        first = rows.filter(pl.col("tick") == rows["tick"].min())
        inventories = first["inventory"].to_list()
        if inventories and all(all(_is_knife_item(x) for x in (inv or [])) for inv in inventories):
            out.append(int(r["round_num"]))
    return out


def drop_rounds(tables: dict, nums: list[int]) -> dict:
    """Remove the given rounds from every table that has `round_num` and renumber the
    rest consecutively, so halves and overtime are counted from the first real round."""
    if not nums:
        return tables
    drop = set(int(n) for n in nums)
    keep = sorted(int(n) for n in tables["rounds"]["round_num"].to_list() if int(n) not in drop)
    remap = {old: new for new, old in enumerate(keep, 1)}
    out = {}
    for name, df in tables.items():
        if "round_num" in df.columns and df.height:
            df = df.filter(~pl.col("round_num").is_in(list(drop)))
            df = df.with_columns(pl.col("round_num").replace_strict(remap, default=None,
                                                                    return_dtype=df.schema["round_num"]))
        out[name] = df
    return out


def _build_rounds(rounds: pl.DataFrame, bomb: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    if rounds.height == 0:
        return empty("rounds")
    planted = bomb.filter(pl.col("event") == "plant_complete").select("round_num").unique()
    ct_b, t_b = _scores(rounds, rosters)
    out = rounds.with_columns(
        # schema.py: start_tick is never null; awpy leaves it null when freeze start was
        # not observed, so fall back to the previous round's end, then freeze end.
        pl.coalesce("start_tick", pl.col("_hi").shift(1), "freeze_end_tick").alias("start_tick"),
        pl.col("round_num").is_in(planted["round_num"].to_list()).alias("bomb_planted"),
        pl.Series("ct_score_before", ct_b, dtype=pl.Int32),
        pl.Series("t_score_before", t_b, dtype=pl.Int32),
    )
    return conform(out, "rounds")


def _side_counts(rosters: pl.DataFrame) -> dict[str, int] | None:
    if rosters.height == 0:
        return None
    rows = rosters.to_dicts()
    return {side: int(statistics.median(len(r[side] or []) for r in rows)) for side in ("ct", "t")}


# ── public API ────────────────────────────────────────────────────────────────

_DATASETS = ("rounds", "kills", "damages", "bomb", "grenades", "fires", "smokes", "shots",
             "players", "blinds", "item_events")


def parse(path, *, demo=None, footsteps: bool = True) -> tuple[dict[str, pl.DataFrame], dict]:
    """Parse one demo into (tables, meta) for `RawTables.from_parsed`.

    tables: exactly `TABLES`, each with the A.1 columns and dtypes (`A1_SCHEMAS`).
    meta:   map, tick_rate, game_version, server, demo_sha256 (A.1 meta) plus n_rounds,
            players (roster), side_counts, duration_s, knife_rounds, dropped_rows,
            unavailable (A.1 fields emitted as null), awpy_version."""
    p = Path(path)
    if demo is None:
        if not p.is_file():
            raise FileNotFoundError(f"demo not found: {p}")
        import awpy
        demo = awpy.Demo(p)
    if hasattr(demo, "load"):
        # One call so awpy can share decode passes between compatible datasets.
        demo.load(*_DATASETS)
    meta = _basic_meta(_header_dict(demo))
    tick_rate = meta["tick_rate"]
    dropped: dict[str, int] = {}

    rounds  = _rounds_base(demo)
    snap    = _snapshots(demo, rounds)
    damages = _build_damages(demo, rounds)
    kills   = _build_kills(demo, rounds, snap)
    braw    = _bomb_raw(demo, rounds)
    grenades, smokes, infernos, flashes = _build_utility(demo, rounds, _ticks_per_s(tick_rate))
    blinds  = flashes if _attr(demo, "blinds") is not None else None
    ticks   = _build_ticks(snap, kills, damages, _plant_intervals(braw, rounds), blinds, tick_rate)
    bomb    = _build_bomb(braw, rounds, snap, dropped)
    rosters = _side_rosters(snap, rounds)
    knife   = knife_round_nums(ticks, rounds)
    if knife:
        rounds = rounds.with_columns(pl.when(pl.col("round_num").is_in(knife)).then(True)
                                       .otherwise(pl.col("is_knife")).alias("is_knife"))

    tables = {
        "rounds":      _build_rounds(rounds, bomb, rosters),
        "ticks":       ticks,
        "kills":       kills,
        "damages":     conform(damages, "damages"),
        "grenades":    grenades,
        "smokes":      smokes,
        "infernos":    infernos,
        "flashes":     flashes,
        "bomb":        bomb,
        "weapon_fire": _simple(_attr(demo, "shots"), rounds, "weapon_fire",
                               lambda d: ["tick", _col(d, "steamid", dtype=SID).alias("steamid"),
                                          _col(d, "weapon").cast(STR).alias("weapon")]),
        "footsteps":   _build_footsteps(demo, rounds, snap) if footsteps else empty("footsteps"),
        "items":       _build_items(demo, rounds, dropped),
    }

    knife_nums = rounds.filter(pl.col("is_knife").fill_null(False))["round_num"].to_list()
    tables = drop_rounds(tables, knife_nums)             # MED-35: knife rounds never sampled or scored

    if meta["duration_s"] is None and rounds.height and tick_rate:
        span = rounds["_hi"].max() - rounds["_lo"].min()
        meta["duration_s"] = float(span) / tick_rate if span is not None else None
    try:
        import awpy
        awpy_version = awpy.__version__
    except ImportError:
        awpy_version = None
    meta.update({
        "demo_sha256":  sha256_file(p) if p.is_file() else None,
        "n_rounds":     int(tables["rounds"].height),
        "players":      _roster(demo),
        "side_counts":  _side_counts(rosters),
        "knife_rounds": [int(n) for n in knife_nums],   # original numbers of the rounds dropped
        "dropped_rows": dropped,
        "unavailable":  unavailable_fields(tables),
        "awpy_version": awpy_version,
    })
    return tables, meta


def meta_frame(meta: dict) -> pl.DataFrame:
    """The one-row A.1 `meta` table from a parse meta dict."""
    return pl.DataFrame([{k: meta.get(k) for k in META_SCHEMA}], schema=META_SCHEMA)


def unavailable_fields(tables: dict[str, pl.DataFrame]) -> list[str]:
    """`table.column` for every non-empty table whose column is entirely null."""
    out = []
    for name, df in tables.items():
        if df.height == 0:
            continue
        out += [f"{name}.{c}" for c in df.columns if df[c].null_count() == df.height]
    return out
