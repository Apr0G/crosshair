"""
Round-state sampler.

Walks every round and emits a snapshot every SAMPLE_INTERVAL ticks (~1 Hz).
Each snapshot captures team-level state: alive counts, HP, armor, equipment value,
utility on hand, active smokes/molotovs, post-plant flags, audio/visual info.

Each row is labelled with `round_won_ct` so the snapshots double as training data
for the win-probability model.
"""
import math
import numpy as np
import pandas as pd

SAMPLE_INTERVAL   = 64     # ticks between samples (~1 s at 64 tick)
C4_TIMER          = 40.0   # seconds bomb burns after plant
SITE_BLOCK_RADIUS = 300.0  # units — smoke/molotov covers the bombsite within this
HEARD_WINDOW      = 640    # ticks (~10 s) — how long sound is "remembered"
HEARD_RANGE       = 1500.0 # units — max distance to hear footsteps/shots


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


def _team_spread(snap) -> float:
    """Mean pairwise distance between alive teammates — proxy for team cohesion."""
    if snap.empty or "X" not in snap.columns or len(snap) < 2:
        return 0.0
    xs = snap["X"].to_numpy(dtype=float, na_value=np.nan)
    ys = snap["Y"].to_numpy(dtype=float, na_value=np.nan)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 2:
        return 0.0
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d  = np.sqrt(dx * dx + dy * dy)
    n  = len(xs)
    return float(d.sum() / (n * (n - 1)))


def _min_dist_to_xy(snap, xy) -> float | None:
    if xy is None or snap.empty or "X" not in snap.columns:
        return None
    xs = snap["X"].to_numpy(dtype=float, na_value=np.nan)
    ys = snap["Y"].to_numpy(dtype=float, na_value=np.nan)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if not mask.any():
        return None
    return float(np.sqrt((xs[mask] - xy[0]) ** 2 + (ys[mask] - xy[1]) ** 2).min())


def _active_xy(df, tick, start_col="start_tick", end_col="end_tick") -> list[tuple]:
    if df is None or df.empty:
        return []
    active = df[(df[start_col] <= tick) & (df[end_col] >= tick)]
    if active.empty or "X" not in active.columns:
        return []
    return list(zip(active["X"].astype(float), active["Y"].astype(float)))


def _site_blocked(bomb_xy, positions, radius=SITE_BLOCK_RADIUS) -> bool:
    if bomb_xy is None or not positions:
        return False
    bx, by = bomb_xy
    return any(math.sqrt((sx - bx) ** 2 + (sy - by) ** 2) <= radius for sx, sy in positions)


def _team_heard_enemy(snap_team, enemy_side, tick, footsteps_df, shots_df, reloads_df,
                      enemy_names: set | None = None) -> int:
    """Count how many players on snap_team heard an enemy in the last HEARD_WINDOW ticks."""
    if snap_team.empty or "X" not in snap_team.columns:
        return 0
    t_start = tick - HEARD_WINDOW
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


def _info_state(snap_ct, snap_t, vis_cache, tick) -> dict:
    if not vis_cache:
        return {"ct_spotted_count": 0, "t_spotted_count": 0}

    cache_ticks = sorted(vis_cache.keys())
    if not cache_ticks:
        return {"ct_spotted_count": 0, "t_spotted_count": 0}
    nearest = min(cache_ticks, key=lambda t: abs(t - tick))

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


def sample_round_states(
    tables: dict,
    match_id: str,
    map_name: str = "unknown",
    vis_checker=None,
) -> list[dict]:
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
        return []

    tick_rate = 64
    states    = []

    # match bomb plants to rounds by tick range since bomb_planted has no round_num
    bomb_info: dict[int, dict] = {}
    if not bomb_planted_df.empty and "tick" in bomb_planted_df.columns:
        for _, r in rounds_df.iterrows():
            rn       = int(r["round_num"])
            r_freeze = r.get("freeze_end")
            r_offend = r.get("official_end") or r.get("end")
            if r_freeze is None or r_offend is None:
                continue
            in_round = bomb_planted_df[
                (bomb_planted_df["tick"] >= float(r_freeze)) &
                (bomb_planted_df["tick"] <= float(r_offend))
            ]
            if not in_round.empty:
                row = in_round.iloc[0]
                bomb_info[rn] = {
                    "tick": int(row["tick"]),
                    "xy":   (float(row["user_X"]), float(row["user_Y"])) if "user_X" in row and pd.notna(row["user_X"]) else None,
                }

    for _, r in rounds_df.iterrows():
        try:
            rn = int(r["round_num"])
        except (ValueError, TypeError):
            continue
        r_freeze  = r.get("freeze_end")
        r_offend  = r.get("official_end") or r.get("end")
        round_won = r.get("winner", "?")

        if r_freeze is None or r_offend is None:
            continue

        try:
            r_start = float(r_freeze)
            r_end   = float(r_offend)
        except (ValueError, TypeError):
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

        for tick in range(int(r_start), int(r_end), SAMPLE_INTERVAL):
            snap = r_ticks[(r_ticks["tick"] >= tick - 32) & (r_ticks["tick"] <= tick + 32)]
            if snap.empty:
                continue
            snap = snap.sort_values("tick").groupby("name").last().reset_index()

            alive   = snap[snap["is_alive"] == True] if "is_alive" in snap.columns else snap
            snap_ct = alive[alive["side"] == "ct"] if "side" in alive.columns else pd.DataFrame()
            snap_t  = alive[alive["side"] == "t"]  if "side" in alive.columns else pd.DataFrame()

            alive_ct = len(snap_ct)
            alive_t  = len(snap_t)
            if alive_ct == 0 and alive_t == 0:
                continue

            post_plant        = plant_tick is not None and tick >= plant_tick
            time_remaining_s  = round((r_end - tick) / tick_rate, 2)
            time_into_round_s = round((tick - r_start) / tick_rate, 2)

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

            active_smokes_xy   = _active_xy(r_smokes, tick)
            active_infernos_xy = _active_xy(r_infernos, tick)

            site_smoked  = _site_blocked(bomb_xy, active_smokes_xy)  if post_plant else False
            site_on_fire = _site_blocked(bomb_xy, active_infernos_xy) if post_plant else False

            min_dist_ct = _min_dist_to_xy(snap_ct, bomb_xy) if post_plant and bomb_xy else None
            min_dist_t  = _min_dist_to_xy(snap_t,  bomb_xy) if post_plant and bomb_xy else None

            info = _info_state(snap_ct, snap_t, vis_cache, tick)

            ct_names = set(snap_ct["name"].tolist()) if "name" in snap_ct.columns else set()
            t_names  = set(snap_t["name"].tolist())  if "name" in snap_t.columns  else set()
            heard_ct = _team_heard_enemy(snap_ct, "t",  tick, footsteps_df, shots_df, reloads_df, enemy_names=t_names)
            heard_t  = _team_heard_enemy(snap_t,  "ct", tick, footsteps_df, shots_df, reloads_df, enemy_names=ct_names)

            states.append({
                "match_id":            match_id,
                "map":                 map_name,
                "round_num":           rn,
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
                "ct_spread":           round(ct_spread, 1),
                "t_spread":            round(t_spread, 1),
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
                "round_won_ct":        round_won_ct,
            })

        print(f"  [sampler] round {rn}: {len([s for s in states if s['round_num'] == rn])} samples", flush=True)

    return states
