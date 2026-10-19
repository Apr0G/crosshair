"""
Automated leakage audit (P3.6): build twice with the outcome flipped and official_end
moved +5 s; every non-terminal feature must be identical. Plus the poisoned-future-row
causality check (P3.6.3) and the LIVE_SAFE coverage check (P3.6.2).

    from engine.state import leakage
    build_fn = lambda raw: builder.build(raw, geo, cfg, match_key="m", tier="pro", labels=False)
    problems = leakage.audit(build_fn, raw)                  # [] = clean
    problems = leakage.poison_check(build_fn, raw, round_num=1, tick=3000)

The outcome flip is done one round at a time (later rounds legitimately depend on an
earlier round's winner through scores and loss streaks), and only that round's
snapshots are compared, at ticks present in both builds that are terminal in neither.
Labels are excluded: they read the future by design.
"""
import numpy as np
import polars as pl

from engine.parse.schema import RawTables
from engine.state.schema import CTX_FIELDS, PLAYER_FIELDS, UTILITY_FIELDS, missing_live_safe

# ── outcome flip ──────────────────────────────────────────────────────────────

_FLIP_SIDE = {"ct": "t", "t": "ct"}


def flip_outcome(raw: RawTables, round_num: int, shift_s: float = 5.0) -> RawTables:
    """Copy of `raw` with round `round_num`'s winner flipped, its end_reason swapped to
    the other side's, and official_end_tick moved by +shift_s."""
    tr = int(raw.meta["tick_rate"])
    is_r = pl.col("round_num") == round_num
    reason = pl.col("end_reason")
    flipped_reason = (pl.when(reason.str.starts_with("ct_")).then(pl.lit("t_") + reason.str.slice(3))
                        .when(reason.str.starts_with("t_")).then(pl.lit("ct_") + reason.str.slice(2))
                        .otherwise(reason))
    rounds = raw.rounds.with_columns(
        pl.when(is_r).then(pl.col("winner_side").replace_strict(_FLIP_SIDE, default=None))
          .otherwise(pl.col("winner_side")).alias("winner_side"),
        pl.when(is_r).then(flipped_reason).otherwise(reason).alias("end_reason"),
        pl.when(is_r).then(pl.col("official_end_tick") + int(round(shift_s * tr)))
          .otherwise(pl.col("official_end_tick")).alias("official_end_tick"))
    frames = raw.frames()
    frames["rounds"] = rounds
    return RawTables(**frames, meta=dict(raw.meta))


# ── comparison ────────────────────────────────────────────────────────────────

def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


def diff_features(a, b) -> list[str]:
    """Field-level differences between two snapshots' features (labels ignored)."""
    out = []
    for i, (pa, pb) in enumerate(zip(a.players, b.players)):
        out += [f"players[{i}].{f}: {getattr(pa, f)!r} != {getattr(pb, f)!r}"
                for f in PLAYER_FIELDS if not _same(getattr(pa, f), getattr(pb, f))]
    if len(a.players) != len(b.players):
        out.append(f"players: {len(a.players)} != {len(b.players)}")
    if len(a.utility) != len(b.utility):
        out.append(f"utility: {len(a.utility)} tokens != {len(b.utility)}")
    for j, (ua, ub) in enumerate(zip(a.utility, b.utility)):
        out += [f"utility[{j}].{f}: {getattr(ua, f)!r} != {getattr(ub, f)!r}"
                for f in UTILITY_FIELDS if not _same(getattr(ua, f), getattr(ub, f))]
    out += [f"ctx.{f}: {getattr(a.ctx, f)!r} != {getattr(b.ctx, f)!r}"
            for f in CTX_FIELDS if not _same(getattr(a.ctx, f), getattr(b.ctx, f))]
    if a.pair.shape != b.pair.shape or not np.array_equal(a.pair, b.pair, equal_nan=True):
        out.append("pair: tables differ")
    if a.meta.kinds != b.meta.kinds and "terminal" not in a.meta.kinds + b.meta.kinds:
        out.append(f"kinds: {a.meta.kinds} != {b.meta.kinds}")
    return out


def _by_tick(snaps, round_num: int) -> dict:
    return {s.meta.tick: s for s in snaps if s.meta.round_num == round_num}


def compare(base, other, round_num: int, *, before: int | None = None, label: str = "") -> list[str]:
    """Problems for every common, non-terminal snapshot of `round_num` (and tick <
    `before` if given) whose features differ."""
    a, b = _by_tick(base, round_num), _by_tick(other, round_num)
    problems = []
    for t in sorted(set(a) & set(b)):
        if before is not None and t >= before:
            continue
        if "terminal" in a[t].meta.kinds or "terminal" in b[t].meta.kinds:
            continue
        d = diff_features(a[t], b[t])
        if d:
            problems.append(f"{label}round {round_num} tick {t}: " + "; ".join(d[:5])
                            + (f" (+{len(d) - 5} more)" if len(d) > 5 else ""))
    return problems


# ── P3.6.1 ────────────────────────────────────────────────────────────────────

def audit(build_fn, raw: RawTables, shift_s: float = 5.0) -> list[str]:
    """Outcome-flip audit. `build_fn(raw) -> list[Snapshot]`. Empty list = clean."""
    problems = [f"LIVE_SAFE: {m}" for m in missing_live_safe()]
    base = build_fn(raw)
    for r in sorted(raw.rounds["round_num"].to_list()):
        flipped = build_fn(flip_outcome(raw, r, shift_s))
        n_common = len(set(_by_tick(base, r)) & set(_by_tick(flipped, r)))
        if n_common == 0:
            problems.append(f"round {r}: no common snapshots to compare")
        problems += compare(base, flipped, r, label="outcome flip: ")
    return problems


# ── P3.6.3 ────────────────────────────────────────────────────────────────────

def poison(raw: RawTables, round_num: int, tick: int) -> RawTables:
    """Copy of `raw` with impossible rows at `tick` in every event table: a ticks row
    per player with health 999 and position 1e6, a team-less kill and a 999 damage
    by the first player on the second, a 999 s flash on every player, a smoke and a
    fire at 1e6, a weapon_fire and a footstep at 1e6, a bomb plant at site Z."""
    t = int(tick)
    f = raw.frames()
    ticks = raw.ticks.filter(pl.col("round_num") == round_num)
    last = ticks.filter(pl.col("tick") == ticks["tick"].max())
    ids = last["steamid"].to_list()
    big = 1.0e6
    f["ticks"] = pl.concat([raw.ticks, last.with_columns(
        pl.lit(t, dtype=pl.Int64).alias("tick"), pl.lit(999, dtype=pl.Int16).alias("health"),
        pl.lit(big, dtype=pl.Float32).alias("x"), pl.lit(True).alias("is_alive"))])

    def add(name, row):
        df = getattr(raw, name)
        new = pl.DataFrame([{c: row.get(c) for c in df.columns}], schema=df.schema)
        f[name] = pl.concat([df, new])

    a, v = ids[0], ids[-1]
    add("kills", dict(tick=t, round_num=round_num, attacker_steamid=a, victim_steamid=v,
                      weapon="ak47", headshot=True, team_kill=False, distance=big))
    add("damages", dict(tick=t, round_num=round_num, attacker_steamid=a, victim_steamid=v,
                        weapon="ak47", dmg_health=999, dmg_armor=999, hitgroup="head"))
    for sid in ids:
        add("flashes", dict(entity_id=999_000 + len(f["flashes"]), detonate_tick=t,
                            player_steamid=sid, blind_duration=999.0))
    add("smokes", dict(entity_id=999_100, start_tick=t, end_tick=None, x=big, y=big, z=0.0,
                       thrower_steamid=a))
    add("infernos", dict(entity_id=999_101, start_tick=t, end_tick=None, x=big, y=big, z=0.0,
                         extent_radius=999.0, thrower_steamid=a))
    add("weapon_fire", dict(tick=t, steamid=v, weapon="ak47"))
    add("footsteps", dict(tick=t, steamid=v, x=0.0, y=0.0, z=0.0))
    add("bomb", dict(tick=t, round_num=round_num, event="plant_complete", steamid=v, site="Z",
                     has_kit=None, x=big, y=big, z=0.0))
    return RawTables(**f, meta=dict(raw.meta))


def poison_check(build_fn, raw: RawTables, round_num: int, tick: int) -> list[str]:
    """Snapshots strictly before `tick` must not change when future rows are poisoned."""
    base = build_fn(raw)
    bad  = build_fn(poison(raw, round_num, tick))
    if not any(s.meta.round_num == round_num and s.meta.tick < tick for s in base):
        return [f"round {round_num}: no snapshot before tick {tick}"]
    return compare(base, bad, round_num, before=tick, label="poison: ")
