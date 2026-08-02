"""
Round-state sampler.

Walks every round and emits a snapshot every SAMPLE_INTERVAL ticks (~1 Hz).
Each snapshot captures team-level state: alive counts, HP, armor, equipment value,
utility on hand, active smokes/molotovs, post-plant flags, audio/visual info.

Each row is labelled with `round_won_ct` so the snapshots double as training data
for the win-probability model.
"""
import math
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Durations in SECONDS. Everything tick-valued is derived from the demo's real tick
# rate at run time — a fixed tick count silently halves the real duration on a
# 128-tick FACEIT demo, which is what these constants used to do.
SAMPLE_INTERVAL_S = 1.0    # seconds between training samples
SNAP_HALF_S       = 0.5    # seconds either side of a sample tick to build a snapshot
HEARD_WINDOW_S    = 10.0   # how long a sound is "remembered"

SAMPLE_INTERVAL   = 64     # legacy 64-tick defaults, kept for callers without a rate
SNAP_HALF_WINDOW  = 32
C4_TIMER          = 40.0   # seconds bomb burns after plant
ROUND_TIME_S      = 115.0  # seconds of round clock after freeze time (CS2 competitive)
SITE_BLOCK_RADIUS = 300.0  # units — smoke/molotov covers the bombsite within this
HEARD_WINDOW      = 640    # legacy default; prefer ctx.heard_window
HEARD_RANGE       = 1500.0 # units — max distance to hear footsteps/shots


def _nan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except (TypeError, ValueError):
        return False


def _first_valid(*vals):
    """First value that is neither None nor NaN. `or` can't be used here — NaN is
    truthy, so `official_end or end` never falls through, and 0 is legitimate."""
    for v in vals:
        if not _nan(v):
            return v
    return None


_UTIL_KEYWORDS = {
    "smokes":   ["smokegrenade", "smoke"],
    "flashes":  ["flashbang", "flash"],
    "he":       ["hegrenade", "he_grenade"],
    "molotovs": ["molotov", "incgrenade", "incendiary"],
}


def _count_util(inventory) -> dict:
    counts = {k: 0 for k in _UTIL_KEYWORDS}
    if not isinstance(inventory, list):
        return counts
    for item in inventory:
        s = str(item).lower()
        for key, keywords in _UTIL_KEYWORDS.items():
            if any(kw in s for kw in keywords):
                counts[key] += 1
                break
    return counts


def _team_util(snap) -> dict:
    totals = {k: 0 for k in _UTIL_KEYWORDS}
    if "inventory" not in snap.columns:
        return totals
    for inv in snap["inventory"]:
        for k, v in _count_util(inv).items():
            totals[k] += v
    return totals


def _team_spread(snap) -> float | None:
    """Mean pairwise distance between alive teammates — proxy for team cohesion.

    Returns None (not 0.0) when fewer than two players contribute: 0.0 means
    "stacked tightly together", which is the opposite of a 1vX clutch.
    """
    if snap.empty or not {"X", "Y"}.issubset(snap.columns) or len(snap) < 2:
        return None
    xs = snap["X"].to_numpy(dtype=float, na_value=np.nan)
    ys = snap["Y"].to_numpy(dtype=float, na_value=np.nan)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 2:
        return None
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d  = np.sqrt(dx * dx + dy * dy)
    n  = len(xs)
    return float(d.sum() / (n * (n - 1)))


def _min_dist_to_xy(snap, xy) -> float | None:
    if xy is None or snap.empty or not {"X", "Y"}.issubset(snap.columns):
        return None
    xs = snap["X"].to_numpy(dtype=float, na_value=np.nan)
    ys = snap["Y"].to_numpy(dtype=float, na_value=np.nan)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if not mask.any():
        return None
    return float(np.sqrt((xs[mask] - xy[0]) ** 2 + (ys[mask] - xy[1]) ** 2).min())


def _active_xy(df, tick, start_col="start_tick", end_col="end_tick") -> list[tuple]:
    if df is None or df.empty or not {start_col, end_col, "X", "Y"}.issubset(df.columns):
        return []
    active = df[(df[start_col] <= tick) & (df[end_col] >= tick)]
    if active.empty:
        return []
    return list(zip(active["X"].astype(float), active["Y"].astype(float)))


def _site_blocked(bomb_xy, positions, radius=SITE_BLOCK_RADIUS) -> bool:
    if bomb_xy is None or not positions:
        return False
    bx, by = bomb_xy
    return any(math.sqrt((sx - bx) ** 2 + (sy - by) ** 2) <= radius for sx, sy in positions)


def _team_heard_enemy(snap_team, enemy_side, tick, footsteps_df, shots_df, reloads_df,
                      enemy_names: set | None = None, heard_window: int = HEARD_WINDOW) -> int:
    """Count how many players on snap_team heard an enemy in the last HEARD_WINDOW ticks."""
    if snap_team.empty or "X" not in snap_team.columns:
        return 0
    t_start = tick - heard_window
    heard = 0
    sound_sources = []
    for df in [footsteps_df, shots_df, reloads_df]:
        if df is None or df.empty or "X" not in df.columns:
            continue
        recent = df[(df["tick"] >= t_start) & (df["tick"] <= tick)] if "tick" in df.columns else pd.DataFrame()
        if recent.empty:
            continue
        if "side" in recent.columns:
            enemy = recent[recent["side"] == enemy_side]
        elif enemy_names is not None and "name" in recent.columns:
            enemy = recent[recent["name"].isin(enemy_names)]
        elif enemy_names is not None and "user_name" in recent.columns:
            enemy = recent[recent["user_name"].isin(enemy_names)]
        else:
            continue
        if enemy.empty:
            continue
        xs = enemy["X"].to_numpy(dtype=float, na_value=np.nan)
        ys = enemy["Y"].to_numpy(dtype=float, na_value=np.nan)
        sound_sources.extend(zip(xs, ys))

    if not sound_sources:
        return 0

    src_arr = np.array(sound_sources)
    for _, row in snap_team.iterrows():
        try:
            px, py = float(row["X"]), float(row["Y"])
        except (ValueError, TypeError):
            continue
        dists = np.sqrt((src_arr[:, 0] - px) ** 2 + (src_arr[:, 1] - py) ** 2)
        if np.any(dists <= HEARD_RANGE):
            heard += 1
    return heard


def _info_state(snap_ct, snap_t, vis_cache, tick, window_ticks: int = HEARD_WINDOW) -> dict:
    empty = {"ct_spotted_count": 0, "t_spotted_count": 0}
    if not vis_cache:
        return empty

    # Only look BACKWARDS, and only within the window. _precompute_visibility skips
    # ticks where nobody was spotted, so an unbounded nearest-match would reach
    # forward to the round's first contact and tag every earlier state with it.
    past = [k for k in vis_cache if tick - window_ticks <= k <= tick]
    if not past:
        return empty
    nearest = max(past)

    tick_entry = vis_cache.get(nearest, {})
    ct_names = set(snap_ct["name"].tolist()) if "name" in snap_ct.columns else set()
    t_names  = set(snap_t["name"].tolist())  if "name" in snap_t.columns  else set()

    ct_spots = set()
    t_spots  = set()
    for pname, spotted in tick_entry.items():
        if pname in ct_names:
            ct_spots.update(e["name"] for e in spotted if e.get("side") == "t")
        elif pname in t_names:
            t_spots.update(e["name"] for e in spotted if e.get("side") == "ct")

    return {
        "ct_spotted_count": len(ct_spots),
        "t_spotted_count":  len(t_spots),
    }


@dataclass
class RoundContext:
    """Everything `build_state` needs for one round, precomputed once.

    Splitting this out is what lets a state vector be built at an ARBITRARY tick
    rather than only on the SAMPLE_INTERVAL grid. Impact attribution needs the state
    immediately before and after each action; bracketing events between two ~1 Hz
    samples hands every event inside that second the same win-probability delta.
    """
    match_id:     str
    map_name:     str
    round_num:    int
    r_start:      float
    r_end:        float
    r_ticks:      "pd.DataFrame"
    r_smokes:     "pd.DataFrame"
    r_infernos:   "pd.DataFrame"
    footsteps_df: object
    shots_df:     object
    reloads_df:   object
    vis_cache:    dict
    plant_tick:   int | None
    bomb_xy:      tuple | None
    round_won_ct: int
    tick_rate:    int = 64
    snap_half:    int = SNAP_HALF_WINDOW
    sample_interval: int = SAMPLE_INTERVAL
    heard_window:    int = HEARD_WINDOW


def build_state(tick: int, ctx: RoundContext) -> dict | None:
    """State vector at `tick`. None when no usable snapshot exists there.

    Must stay a pure function of (tick, ctx): the attribution work evaluates it at
    tick-1 and tick+1 around an action, so it cannot depend on iteration order or on
    anything cached across calls.
    """
    tick = int(tick)
    # BACKWARD-looking window. A symmetric one reads up to tick+snap_half, so
    # build_state(t-1) saw ticks after t — meaning the state "before" a kill already
    # had the victim dead, the state after was identical, and the action's own jump
    # was exactly zero. That is what made 95% of instantaneous impacts vanish even
    # with boundary states present. A state at time t must never contain t's future.
    snap = ctx.r_ticks[(ctx.r_ticks["tick"] >  tick - ctx.snap_half) &
                       (ctx.r_ticks["tick"] <= tick)]
    if snap.empty or "name" not in snap.columns:
        return None
    # drop_duplicates keeps a genuine single row per player. groupby().last()
    # takes the last NON-NULL value per column independently, which can pair
    # one tick's position with another tick's health.
    snap = snap.sort_values("tick").drop_duplicates(subset="name", keep="last")

    alive   = snap[snap["is_alive"] == True] if "is_alive" in snap.columns else snap
    snap_ct = alive[alive["side"] == "ct"] if "side" in alive.columns else pd.DataFrame()
    snap_t  = alive[alive["side"] == "t"]  if "side" in alive.columns else pd.DataFrame()

    alive_ct = len(snap_ct)
    alive_t  = len(snap_t)
    if alive_ct == 0 and alive_t == 0:
        return None

    tick_rate  = ctx.tick_rate
    plant_tick = ctx.plant_tick
    bomb_xy    = ctx.bomb_xy

    post_plant        = plant_tick is not None and tick >= plant_tick
    time_into_round_s = round((tick - ctx.r_start) / tick_rate, 2)

    # Clock the state can actually know. Deriving this from r_end (the round's
    # ACTUAL end tick) leaks the outcome: short rounds are decisive rounds, so
    # the model learns "little time left => whoever leads now wins".
    if post_plant:
        time_remaining_s = round(max(0.0, C4_TIMER - (tick - plant_tick) / tick_rate), 2)
    else:
        time_remaining_s = round(max(0.0, ROUND_TIME_S - time_into_round_s), 2)

    hp_ct    = float(snap_ct["health"].sum()) if "health" in snap_ct.columns else 0.0
    hp_t     = float(snap_t["health"].sum())  if "health" in snap_t.columns  else 0.0
    equip_ct = float(snap_ct["current_equip_value"].sum()) if "current_equip_value" in snap_ct.columns else 0.0
    equip_t  = float(snap_t["current_equip_value"].sum())  if "current_equip_value" in snap_t.columns  else 0.0

    armor_ct   = float(snap_ct["armor_value"].sum()) if "armor_value" in snap_ct.columns else 0.0
    armor_t    = float(snap_t["armor_value"].sum())  if "armor_value" in snap_t.columns  else 0.0
    helmets_ct = int(snap_ct["has_helmet"].sum())    if "has_helmet"  in snap_ct.columns else 0
    helmets_t  = int(snap_t["has_helmet"].sum())     if "has_helmet"  in snap_t.columns  else 0

    ct_spread = _team_spread(snap_ct)
    t_spread  = _team_spread(snap_t)

    has_defuser = bool(snap_ct["has_defuser"].any()) if "has_defuser" in snap_ct.columns else False

    util_ct = _team_util(snap_ct)
    util_t  = _team_util(snap_t)

    active_smokes_xy   = _active_xy(ctx.r_smokes, tick)
    active_infernos_xy = _active_xy(ctx.r_infernos, tick)

    site_smoked  = _site_blocked(bomb_xy, active_smokes_xy)   if post_plant else False
    site_on_fire = _site_blocked(bomb_xy, active_infernos_xy) if post_plant else False

    min_dist_ct = _min_dist_to_xy(snap_ct, bomb_xy) if post_plant and bomb_xy else None
    min_dist_t  = _min_dist_to_xy(snap_t,  bomb_xy) if post_plant and bomb_xy else None

    info = _info_state(snap_ct, snap_t, ctx.vis_cache, tick, window_ticks=ctx.heard_window)

    ct_names = set(snap_ct["name"].tolist()) if "name" in snap_ct.columns else set()
    t_names  = set(snap_t["name"].tolist())  if "name" in snap_t.columns  else set()
    heard_ct = _team_heard_enemy(snap_ct, "t",  tick, ctx.footsteps_df, ctx.shots_df, ctx.reloads_df,
                                enemy_names=t_names, heard_window=ctx.heard_window)
    heard_t  = _team_heard_enemy(snap_t,  "ct", tick, ctx.footsteps_df, ctx.shots_df, ctx.reloads_df,
                                enemy_names=ct_names, heard_window=ctx.heard_window)

    return {
        "match_id":            ctx.match_id,
        "map":                 ctx.map_name,
        "round_num":           ctx.round_num,
        "tick":                tick,
        "time_into_round_s":   time_into_round_s,
        "time_remaining_s":    time_remaining_s,
        "post_plant":          int(post_plant),
        "alive_ct":            alive_ct,
        "alive_t":             alive_t,
        "total_hp_ct":         round(hp_ct, 1),
        "total_hp_t":          round(hp_t, 1),
        "total_armor_ct":      round(armor_ct, 1),
        "total_armor_t":       round(armor_t, 1),
        "helmets_ct":          helmets_ct,
        "helmets_t":           helmets_t,
        "ct_spread":           None if ct_spread is None else round(ct_spread, 1),
        "t_spread":            None if t_spread  is None else round(t_spread, 1),
        "has_defuser":         int(has_defuser),
        "equip_value_ct":      round(equip_ct),
        "equip_value_t":       round(equip_t),
        "smokes_ct":           util_ct["smokes"],
        "smokes_t":            util_t["smokes"],
        "flashes_ct":          util_ct["flashes"],
        "flashes_t":           util_t["flashes"],
        "he_ct":               util_ct["he"],
        "he_t":                util_t["he"],
        "molotovs_ct":         util_ct["molotovs"],
        "molotovs_t":          util_t["molotovs"],
        "active_smokes":       len(active_smokes_xy),
        "active_infernos":     len(active_infernos_xy),
        "active_smokes_xy":    active_smokes_xy,
        "active_infernos_xy":  active_infernos_xy,
        "site_smoked":         int(site_smoked),
        "site_on_fire":        int(site_on_fire),
        "min_dist_ct_to_bomb": min_dist_ct,
        "min_dist_t_to_bomb":  min_dist_t,
        "ct_spotted_count":    info["ct_spotted_count"],
        "t_spotted_count":     info["t_spotted_count"],
        "ct_heard_enemy":      heard_ct,
        "t_heard_enemy":       heard_t,
        "round_won_ct":        ctx.round_won_ct,
    }


def iter_round_contexts(
    tables: dict,
    match_id: str,
    map_name: str = "unknown",
    vis_checker=None,
):
    """Yield a RoundContext per playable round. Shared by the sampler and by any
    consumer that needs state at arbitrary ticks (impact attribution)."""
    from feature_extractor import _precompute_visibility

    rounds_df       = tables.get("rounds",               pd.DataFrame())
    ticks_df        = tables.get("ticks",                pd.DataFrame())
    smokes_df       = tables.get("smokes",               pd.DataFrame())
    infernos_df     = tables.get("infernos",             pd.DataFrame())
    bomb_planted_df = tables.get("event_bomb_planted",   pd.DataFrame())
    footsteps_df    = tables.get("footsteps",            None)
    shots_df        = tables.get("shots",                None)
    reloads_df      = tables.get("event_weapon_reload",  None)

    if rounds_df.empty or ticks_df.empty:
        return

    tick_rate = int(tables.get("tick_rate") or 64)

    # match bomb plants to rounds by tick range since bomb_planted has no round_num
    bomb_info: dict[int, dict] = {}
    if not bomb_planted_df.empty and "tick" in bomb_planted_df.columns:
        for _, r in rounds_df.iterrows():
            rn       = int(r["round_num"])
            r_freeze = r.get("freeze_end")
            r_offend = _first_valid(r.get("official_end"), r.get("end"))
            if _nan(r_freeze) or _nan(r_offend):
                continue
            in_round = bomb_planted_df[
                (bomb_planted_df["tick"] >= float(r_freeze)) &
                (bomb_planted_df["tick"] <= float(r_offend))
            ]
            if not in_round.empty:
                row = in_round.sort_values("tick").iloc[0]
                bomb_info[rn] = {
                    "tick": int(row["tick"]),
                    "xy":   (float(row["user_X"]), float(row["user_Y"]))
                            if {"user_X", "user_Y"}.issubset(row.index)
                            and pd.notna(row["user_X"]) and pd.notna(row["user_Y"])
                            else None,
                }

    for _, r in rounds_df.iterrows():
        try:
            rn = int(r["round_num"])
        except (ValueError, TypeError):
            continue
        r_freeze  = r.get("freeze_end")
        r_offend  = _first_valid(r.get("official_end"), r.get("end"))
        round_won = str(_first_valid(r.get("winner")) or "").strip().lower()

        if _nan(r_freeze) or _nan(r_offend):
            continue

        # A round with no resolvable winner must be skipped, not labelled "T won" —
        # `!= "ct"` would silently fabricate a label for every such round.
        if round_won not in ("ct", "t"):
            print(f"  [sampler] round {rn}: unknown winner {round_won!r}, skipping", flush=True)
            continue

        try:
            r_start = float(r_freeze)
            r_end   = float(r_offend)
        except (ValueError, TypeError):
            continue
        if math.isnan(r_start) or math.isnan(r_end):
            continue
        round_won_ct = 1 if round_won == "ct" else 0

        r_ticks = ticks_df[(ticks_df["tick"] >= r_start) & (ticks_df["tick"] <= r_end)]
        r_ticks = r_ticks.sort_values("tick").reset_index(drop=True)
        if r_ticks.empty:
            continue

        r_smokes = smokes_df[
            (smokes_df["start_tick"] >= r_start) & (smokes_df["start_tick"] <= r_end)
        ] if not smokes_df.empty and "start_tick" in smokes_df.columns else pd.DataFrame()

        r_infernos = infernos_df[
            (infernos_df["start_tick"] >= r_start) & (infernos_df["start_tick"] <= r_end)
        ] if not infernos_df.empty and "start_tick" in infernos_df.columns else pd.DataFrame()

        bomb       = bomb_info.get(rn, {})
        plant_tick = bomb.get("tick")
        bomb_xy    = bomb.get("xy")

        vis_cache = _precompute_visibility(r_ticks, vis_checker, smokes_df=r_smokes)

        yield RoundContext(
            match_id     = match_id,
            map_name     = map_name,
            round_num    = rn,
            r_start      = r_start,
            r_end        = r_end,
            r_ticks      = r_ticks,
            r_smokes     = r_smokes,
            r_infernos   = r_infernos,
            footsteps_df = footsteps_df,
            shots_df     = shots_df,
            reloads_df   = reloads_df,
            vis_cache    = vis_cache,
            plant_tick   = plant_tick,
            bomb_xy      = bomb_xy,
            round_won_ct = round_won_ct,
            tick_rate    = tick_rate,
            snap_half       = max(1, int(round(SNAP_HALF_S       * tick_rate))),
            sample_interval = max(1, int(round(SAMPLE_INTERVAL_S * tick_rate))),
            heard_window    = max(1, int(round(HEARD_WINDOW_S    * tick_rate))),
        )


def sample_round_states(
    tables: dict,
    match_id: str,
    map_name: str = "unknown",
    vis_checker=None,
) -> list[dict]:
    """~1 Hz snapshots across every round — the win-probability training rows.

    Now a thin loop over build_state. Behaviour is unchanged: same grid, same fields.
    """
    states: list[dict] = []
    for ctx in iter_round_contexts(tables, match_id, map_name, vis_checker):
        n = 0
        for tick in range(int(ctx.r_start), int(ctx.r_end), ctx.sample_interval):
            st = build_state(tick, ctx)
            if st is not None:
                st["state_kind"] = "grid"
                states.append(st)
                n += 1
        print(f"  [sampler] round {ctx.round_num}: {n} samples", flush=True)
    return states


def sample_boundary_states(
    tables: dict,
    match_id: str,
    events: list[dict],
    map_name: str = "unknown",
    vis_checker=None,
) -> list[dict]:
    """States immediately before and after each instantaneous action.

    Impact attribution values an action as WP(t+1) - WP(t-1). On the 1 Hz grid alone
    those two ticks land on the same sample for 25% of actions, so the action's own
    jump is invisible and gets smeared across the whole second. These rows make the
    boundaries addressable.

    Tagged `boundary` so training can exclude them — they cluster around kills, and
    training on them would over-sample exactly the moments being predicted.
    """
    import json as _json

    # resolve ticks wanted per round, from the classification feature_extractor emits
    wanted: dict[int, set] = {}
    for e in events:
        if e.get("match_id") != match_id:
            continue
        act = e.get("action")
        if isinstance(act, str):
            try:
                act = _json.loads(act)
            except Exception:
                continue
        if not isinstance(act, dict) or act.get("impact_kind") != "instantaneous":
            continue
        rt, rn = act.get("resolve_tick"), e.get("round_num")
        if rt is None or rn is None:
            continue
        wanted.setdefault(int(rn), set()).update((int(rt) - 1, int(rt) + 1))

    if not wanted:
        return []

    out: list[dict] = []
    for ctx in iter_round_contexts(tables, match_id, map_name, vis_checker):
        ticks = wanted.get(ctx.round_num)
        if not ticks:
            continue
        # Skip anything already on the grid — those rows exist and duplicating them
        # would double-weight that tick.
        grid = set(range(int(ctx.r_start), int(ctx.r_end), ctx.sample_interval))
        for tick in sorted(t for t in ticks if t not in grid):
            if not (ctx.r_start <= tick <= ctx.r_end):
                continue
            st = build_state(tick, ctx)
            if st is not None:
                st["state_kind"] = "boundary"
                out.append(st)
    return out
