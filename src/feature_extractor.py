"""
Extracts player-decision events from parsed demo tables.

For every meaningful moment a player makes a choice, this module records:
  - situation (what the player saw and knew)
  - action    (what they did)
  - outcome   (immediate result + whether their round was won)

Event types:
  buy        — equipment purchase at freeze-end (situation: cash, economy)
  engagement — fired a shot or was shot (situation: spotted enemies, sound, HP)
  utility    — grenade thrown (situation: targets, action: landing pos, outcome: blinds/kills/blocked area)
  rotation   — significant position change (sampled ~1 Hz)
  bomb       — plant / defuse / explode

Output rows are stored in the `events` table; `score_impact.py` later fills
p_before, p_after, impact using the win-probability model.
"""

import math


def _nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _safe(v, default=None):
    if v is None or (isinstance(v, float) and _nan(v)):
        return default
    if hasattr(v, "__iter__") and not isinstance(v, str):
        try:
            return [str(x) for x in v if x is not None]
        except Exception:
            pass
    return v


def _pos(row, prefix="") -> list[float]:
    x = _safe(row.get(f"{prefix}X"))
    y = _safe(row.get(f"{prefix}Y"))
    z = _safe(row.get(f"{prefix}Z"))
    try:
        return [round(float(x), 1), round(float(y), 1), round(float(z), 1)]
    except (TypeError, ValueError):
        return [None, None, None]


# A failed distance must not read as "zero units away" — 0.0 passes every proximity
# and audibility test maximally, fabricating a "close" cue out of a missing coordinate.
def _dist3(a, b) -> float:
    try:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
    except Exception:
        return float("inf")


def _dist2(a, b) -> float:
    try:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    except Exception:
        return float("inf")


def _first_valid(*vals):
    """First value that is neither None nor NaN. `or` can't be used on these — NaN is
    truthy, so `official_end or end` never falls through, and 0 is legitimate."""
    for v in vals:
        if v is not None and not _nan(v):
            return v
    return None


def _enemy_of(side: str) -> str:
    """Opposing side, or "?" when the side is unknown — never guess "ct"."""
    return {"ct": "t", "t": "ct"}.get(side, "?")


# ── Impact-attribution classification ─────────────────────────────────────────
# How an action resolves determines how its win-probability contribution must be
# measured. Taking a 2-tick delta around a smoke or a rotation yields ~0 and makes
# 77% of the corpus meaningless, so the three classes are valued differently:
#
#   INSTANT   the jump across the action:  WP(t+1) - WP(t-1)
#   DURATIVE  the RESIDUAL drift across the action's window — the window's total
#             change minus the instantaneous jumps inside it
#   ROUND     not an in-round action; valued once per team per round
#
# Consumed by score_impact's event-boundary evaluator (design step 3). Emitted into
# the action JSON blob rather than new columns, so this stays schema-compatible.
KIND_INSTANT  = "instantaneous"
KIND_DURATIVE = "durative"
KIND_ROUND    = "round"

# Seconds, converted to ticks against the demo's real tick rate. These were
# previously hardcoded tick counts that silently assumed 64 tick — on a 128-tick
# FACEIT demo every one of them covered half the intended real time.
SMOKE_POP_S    = 2.0    # throw -> deployed
SMOKE_LIFE_S   = 18.0   # deployed -> faded
MOLOTOV_LIFE_S = 7.0    # ignition -> burnt out
FLASH_EFFECT_S = 0.5    # detonation is effectively instant; blindness is valued
                        # through the kills it enables, which are scored separately


def _ticks(seconds: float, tick_rate: int) -> int:
    return int(round(seconds * tick_rate))


def attribution_window(gtype: str, throw_tick, tick_rate: int,
                       start_tick=None, end_tick=None) -> dict:
    """Classification + window for one utility throw.

    Prefers the parser's real start/end ticks when the smokes/infernos tables carry
    them; falls back to nominal durations. Returns the fields merged into `action`.
    """
    if throw_tick is None or _nan(throw_tick):
        return {"impact_kind": KIND_INSTANT, "resolve_tick": None}
    t0 = int(throw_tick)

    if gtype == "Smoke":
        s = int(start_tick) if start_tick is not None and not _nan(start_tick) else t0 + _ticks(SMOKE_POP_S, tick_rate)
        e = int(end_tick)   if end_tick   is not None and not _nan(end_tick)   else s  + _ticks(SMOKE_LIFE_S, tick_rate)
        return {"impact_kind": KIND_DURATIVE, "window_start_tick": s, "window_end_tick": e}

    if gtype in ("Molotov", "Incendiary"):
        s = int(start_tick) if start_tick is not None and not _nan(start_tick) else t0
        e = int(end_tick)   if end_tick   is not None and not _nan(end_tick)   else s + _ticks(MOLOTOV_LIFE_S, tick_rate)
        return {"impact_kind": KIND_DURATIVE, "window_start_tick": s, "window_end_tick": e}

    # HE, Flash, Decoy — the effect lands at detonation.
    return {"impact_kind": KIND_INSTANT, "resolve_tick": t0}


_DAMAGE_COLS = ["dmg_health", "dmg_health_real", "damage"]


def _damage_after(r_damages, tick, pname):
    """Total damage `pname` dealt after `tick`. None if no damage column exists."""
    if r_damages is None or r_damages.empty or tick is None:
        return 0
    if "attacker_name" not in r_damages.columns or "tick" not in r_damages.columns:
        return 0
    col = next((c for c in _DAMAGE_COLS if c in r_damages.columns), None)
    if col is None:
        return None
    sel = r_damages[(r_damages["tick"] > tick) & (r_damages["attacker_name"] == pname)]
    try:
        return int(sel[col].sum())
    except (TypeError, ValueError):
        return None


def _vel_mag(row) -> float:
    try:
        vx = float(row.get("velocity_X") or 0)
        vy = float(row.get("velocity_Y") or 0)
        vz = float(row.get("velocity_Z") or 0)
        return math.sqrt(vx**2 + vy**2 + vz**2)
    except Exception:
        return 0.0


def _relative_direction(player_pos, player_yaw, target_pos) -> str:
    """
    Returns 'front', 'back', 'left', or 'right' based on
    where target_pos is relative to player's facing direction.
    """
    try:
        dx = target_pos[0] - player_pos[0]
        dy = target_pos[1] - player_pos[1]
        angle_to_target = math.degrees(math.atan2(dy, dx))
        yaw = float(player_yaw)
        # CS2 yaw: 0=east, 90=south (clockwise). Convert to standard math angle.
        facing = -yaw  # flip to counter-clockwise
        relative = (angle_to_target - facing) % 360
        if relative < 45 or relative >= 315:
            return "front"
        elif 45 <= relative < 135:
            return "left"
        elif 135 <= relative < 225:
            return "back"
        else:
            return "right"
    except Exception:
        return "unknown"


_GRENADE_MAP = {
    "CHEGrenade": "HE", "CHEGrenadeProjectile": "HE",
    "CFlashbang": "Flash", "CFlashProjectile": "Flash", "CFlashbangProjectile": "Flash",
    "CSmokeGrenadeProjectile": "Smoke", "CSmokeGrenade": "Smoke",
    "CDecoyProjectile": "Decoy", "CMolotovProjectile": "Molotov",
    "CMolotovGrenade": "Molotov", "CIncendiaryGrenade": "Incendiary",
}

def _round_type(equip_value: float, side: str = "t") -> str:
    v = equip_value
    if v < 500:
        return "full_eco"
    if v < 1500:
        return "eco"
    if v < 2500:
        return "force"
    if side == "ct":
        return "half_buy" if v < 5000 else "full_buy"
    return "half_buy" if v < 4500 else "full_buy"


def _tick_slice(df, t_start, t_end):
    """Binary search on a tick-sorted DataFrame. O(log n) instead of O(n) boolean mask."""
    import numpy as np
    if df is None or df.empty or "tick" not in df.columns:
        return df
    ticks = df["tick"].to_numpy()
    lo = np.searchsorted(ticks, t_start, side="left")
    hi = np.searchsorted(ticks, t_end,   side="right")
    return df.iloc[lo:hi]


def _get_tick_snap(ticks_df, tick, window=None, tick_rate: int = 64):
    import pandas as pd
    window = tick_rate if window is None else window
    if ticks_df.empty or tick is None:
        return pd.DataFrame()
    snap = _tick_slice(ticks_df, int(tick), int(tick))
    if snap.empty:
        snap = _tick_slice(ticks_df, int(tick) - window, int(tick))
        snap = snap.sort_values("tick").groupby("name").last().reset_index()
    return snap


def _player_row(snap, name: str):
    if snap.empty:
        return None
    rows = snap[snap["name"] == name]
    return rows.iloc[0] if not rows.empty else None


def _alive_counts_from_snap(snap, player_name, player_side):
    """Returns (alive_teammates, alive_enemies) relative to player."""
    if snap.empty:
        return None, None
    try:
        alive_ct = int(snap[(snap["side"] == "ct") & (snap["is_alive"] == True)].shape[0])
        alive_t  = int(snap[(snap["side"] == "t")  & (snap["is_alive"] == True)].shape[0])
        p_row = _player_row(snap, player_name)
        p_alive = bool(p_row.get("is_alive", True)) if p_row is not None else True
        if player_side == "ct":
            teammates = alive_ct - (1 if p_alive else 0)
            enemies   = alive_t
        else:
            teammates = alive_t - (1 if p_alive else 0)
            enemies   = alive_ct
        return max(0, teammates), enemies
    except Exception:
        return None, None


_SMOKE_RADIUS = 144.0

def _smoke_blocks_los(p1, p2, active_smoke_xys) -> bool:
    """Returns True if any active smoke's center is within SMOKE_RADIUS of the p1→p2 line segment (2D XY)."""
    if not active_smoke_xys:
        return False
    x1, y1 = p1[0], p1[1]
    dx, dy = p2[0] - x1, p2[1] - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return False
    for sx, sy in active_smoke_xys:
        t = max(0.0, min(1.0, ((sx - x1) * dx + (sy - y1) * dy) / len_sq))
        cx, cy = x1 + t * dx, y1 + t * dy
        if (sx - cx) ** 2 + (sy - cy) ** 2 <= _SMOKE_RADIUS ** 2:
            return True
    return False


def _precompute_visibility(r_ticks_df, vis_checker, smokes_df=None, sample_interval: int = 16) -> dict:
    """
    Pre-computes LOS visibility for all player pairs at every sample_interval ticks.
    Returns: {tick: {player_name: [{"name", "position", "side"}, ...]}}
    Called once per round; _round_state does a fast dict lookup.
    """
    if vis_checker is None or r_ticks_df is None or r_ticks_df.empty:
        return {}

    needed = {"tick", "side", "name", "X", "Y", "Z", "is_alive"}
    if not needed.issubset(r_ticks_df.columns):
        return {}

    # Convert Arrow-backed df to numpy-backed once — avoids per-row Arrow overhead
    df = r_ticks_df[list(needed)].copy()
    df["tick"] = df["tick"].astype(int)
    df = df[
        (df["tick"] % sample_interval == 0) &
        (df["is_alive"] == True) &
        df["X"].notna() & df["Y"].notna() & df["Z"].notna()
    ]
    if df.empty:
        return {}

    # Pre-index smokes by tick range for fast lookup: list of (start, end, sx, sy)
    smoke_list = []
    if smokes_df is not None and not smokes_df.empty:
        for _, sr in smokes_df[["start_tick", "end_tick", "X", "Y"]].dropna().iterrows():
            smoke_list.append((int(sr["start_tick"]), int(sr["end_tick"]), float(sr["X"]), float(sr["Y"])))

    result = {}

    for sample_tick, snap in df.groupby("tick"):
        ct = snap[snap["side"] == "ct"]
        t  = snap[snap["side"] == "t"]
        if ct.empty or t.empty:
            continue

        ct_names = ct["name"].to_numpy()
        ct_xyz   = ct[["X", "Y", "Z"]].to_numpy(dtype=float)
        t_names  = t["name"].to_numpy()
        t_xyz    = t[["X", "Y", "Z"]].to_numpy(dtype=float)

        active_smokes = [(sx, sy) for s, e, sx, sy in smoke_list if s <= sample_tick <= e]

        tick_entry = {}

        for side_names, side_xyz, enemy_names, enemy_xyz, enemy_side in [
            (ct_names, ct_xyz, t_names, t_xyz, "t"),
            (t_names, t_xyz, ct_names, ct_xyz, "ct"),
        ]:
            for i, pname in enumerate(side_names):
                pp = tuple(side_xyz[i])
                spotted = []
                for j, ename in enumerate(enemy_names):
                    ep = tuple(enemy_xyz[j])
                    if _smoke_blocks_los(pp, ep, active_smokes):
                        continue
                    try:
                        if vis_checker.is_visible(pp, ep):
                            spotted.append({"name": ename, "position": list(ep), "side": enemy_side})
                    except Exception:
                        continue
                if spotted:
                    existing = tick_entry.get(pname, [])
                    existing.extend(spotted)
                    tick_entry[pname] = existing

        if tick_entry:
            result[int(sample_tick)] = tick_entry

    return result


def _lookup_spotted(vis_cache: dict, player_name: str, tick, window_ticks: int = None,
                    tick_rate: int = 64) -> list[dict]:
    """Enemies this player spotted within the last window_ticks (default ~10 s)."""
    window_ticks = _ticks(10.0, tick_rate) if window_ticks is None else window_ticks
    if not vis_cache or tick is None:
        return []
    t = int(tick)
    seen_names = {}
    # Walk backwards through cache ticks in window; keep most recent position per enemy
    for cache_tick in sorted((k for k in vis_cache if t - window_ticks <= k <= t), reverse=True):
        for entry in vis_cache[cache_tick].get(player_name, []):
            name = entry["name"]
            if name not in seen_names:
                seen_names[name] = entry
    return list(seen_names.values())


_SILENCED_WEAPONS = {"usp_silencer", "m4a1_silencer", "mp5sd"}

_SOUND_RANGES = {
    "footstep":  800,
    "landing":   800,
    "gunshot":   1500,
    "gunshot_silenced": 800,
    "reload":    500,
}


def _get_enemy_names(ticks_df, enemy_side: str) -> set:
    if ticks_df.empty or "name" not in ticks_df.columns or "side" not in ticks_df.columns:
        return set()
    return set(ticks_df[ticks_df["side"] == enemy_side]["name"].unique())


def _smoke_impact(land_pos, throw_tick, r_ticks_df, r_kills, enemy_side, tick_rate: int = 64) -> dict:
    """
    Computes smoke impactfulness metrics.
    All metrics are from the perspective of the throwing team (enemy = opposing side).
    """
    import numpy as np

    if land_pos is None or None in land_pos or r_ticks_df is None or r_ticks_df.empty:
        return {}

    lx, ly     = float(land_pos[0]), float(land_pos[1])
    throw_tick = int(throw_tick)
    # Durations in SECONDS against the demo's real tick rate. These were hardcoded
    # tick counts assuming 64 tick; on a 128-tick FACEIT demo they covered half the
    # intended real time, so "enemies at pop" was sampled 1 s after the throw rather
    # than at the smoke actually deploying.
    POP_DELAY  = _ticks(SMOKE_POP_S,  tick_rate)
    SMOKE_LIFE = _ticks(SMOKE_LIFE_S, tick_rate)
    HALF_SEC   = max(1, tick_rate // 2)
    NEAR_R     = 150    # "nearby" radius
    KILL_R     = 300    # "in area" radius for kills
    APPROACH_W = _ticks(3.0, tick_rate)   # 3s approach window before throw

    pop_tick = throw_tick + POP_DELAY
    end_tick = pop_tick + SMOKE_LIFE
    mid_tick = pop_tick + SMOKE_LIFE // 2

    def _near_count(at_tick, radius):
        sl = _tick_slice(r_ticks_df, at_tick - HALF_SEC, at_tick + HALF_SEC)
        if sl.empty or "side" not in sl.columns or "X" not in sl.columns:
            return 0
        e = sl[sl["side"] == enemy_side]
        if e.empty:
            return 0
        # r_ticks_df is one row PER PLAYER PER TICK. Without collapsing to one row
        # per player, a single enemy standing here for the 1 s window counts ~65×.
        if "name" in e.columns:
            e = e.sort_values("tick").drop_duplicates(subset="name", keep="last")
        xs = e["X"].to_numpy(dtype=float, na_value=0.0)
        ys = e["Y"].to_numpy(dtype=float, na_value=0.0)
        return int((np.sqrt((xs - lx) ** 2 + (ys - ly) ** 2) <= radius).sum())

    def _last_xy_by_name(sl):
        if sl.empty or "side" not in sl.columns or "X" not in sl.columns:
            return {}
        e = sl[sl["side"] == enemy_side]
        if e.empty:
            return {}
        grouped = e.groupby("name")[["X", "Y"]].last()
        return {name: (float(r["X"]), float(r["Y"])) for name, r in grouped.iterrows()}

    enemies_at_pop    = _near_count(pop_tick, NEAR_R)
    enemies_mid_smoke = _near_count(mid_tick, NEAR_R)

    # Enemy kills made from within the smoke area during smoke lifetime
    enemy_kills_in_area = 0
    if r_kills is not None and not r_kills.empty and "attacker_side" in r_kills.columns:
        area_kills = r_kills[
            (r_kills["tick"] >= pop_tick) &
            (r_kills["tick"] <= end_tick) &
            (r_kills["attacker_side"] == enemy_side)
        ]
        if not area_kills.empty and "attacker_X" in area_kills.columns:
            xs    = area_kills["attacker_X"].to_numpy(dtype=float, na_value=0.0)
            ys    = area_kills["attacker_Y"].to_numpy(dtype=float, na_value=0.0)
            dists = np.sqrt((xs - lx) ** 2 + (ys - ly) ** 2)
            enemy_kills_in_area = int((dists <= KILL_R).sum())

    # Approach before throw and stop/reverse after pop
    early_pos = _last_xy_by_name(_tick_slice(r_ticks_df, throw_tick - APPROACH_W, throw_tick - APPROACH_W + HALF_SEC))
    late_pos  = _last_xy_by_name(_tick_slice(r_ticks_df, throw_tick - HALF_SEC, throw_tick))
    after_pos = _last_xy_by_name(_tick_slice(r_ticks_df, pop_tick, pop_tick + _ticks(2.0, tick_rate)))

    enemies_approaching        = 0
    enemies_stopped_or_reversed = 0

    for name in set(early_pos) & set(late_pos):
        ex, ey   = early_pos[name]
        lxe, lye = late_pos[name]
        d_early  = math.sqrt((ex  - lx) ** 2 + (ey  - ly) ** 2)
        d_late   = math.sqrt((lxe - lx) ** 2 + (lye - ly) ** 2)
        if d_late < d_early:
            enemies_approaching += 1
            if name in after_pos:
                axe, aye = after_pos[name]
                d_after  = math.sqrt((axe - lx) ** 2 + (aye - ly) ** 2)
                if d_after >= d_late:
                    enemies_stopped_or_reversed += 1

    return {
        "enemies_at_pop":              enemies_at_pop,
        "enemies_mid_smoke":           enemies_mid_smoke,
        "enemy_kills_in_area":         enemy_kills_in_area,
        "enemies_approaching_before":  enemies_approaching,
        "enemies_stopped_or_reversed": enemies_stopped_or_reversed,
    }


def _flash_impact(throw_tick, r_ticks_df, thrower_side, enemy_side, tick_rate: int = 64) -> dict:
    """Counts enemies and teammates flashed, and avg enemy flash duration."""
    # 640 ticks (~10s): flash travels up to ~2s, effect lasts up to ~3.5s
    window = _tick_slice(r_ticks_df, int(throw_tick), int(throw_tick) + _ticks(10.0, tick_rate))
    if window.empty or "flash_duration" not in window.columns:
        return {"enemies_flashed": 0, "teammates_flashed": 0, "avg_enemy_flash_duration": 0.0}
    flashed = window[window["flash_duration"] > 0]
    if flashed.empty:
        return {"enemies_flashed": 0, "teammates_flashed": 0, "avg_enemy_flash_duration": 0.0}
    peak = flashed.groupby(["name", "side"])["flash_duration"].max().reset_index()
    enemy_rows    = peak[peak["side"] == enemy_side]
    teammate_rows = peak[peak["side"] == thrower_side]
    return {
        "enemies_flashed":          int(len(enemy_rows)),
        "teammates_flashed":        int(len(teammate_rows)),
        "avg_enemy_flash_duration": round(float(enemy_rows["flash_duration"].mean()), 2) if not enemy_rows.empty else 0.0,
    }


def _grenade_damage(throw_tick, thrower, gtype, r_damages, thrower_side=None, name_to_side=None,
                    tick_rate: int = 64) -> dict:
    """Sums damage and unique players hit by HE or Molotov/Incendiary, split by team."""
    if r_damages is None or r_damages.empty or "weapon" not in r_damages.columns:
        return {"damage_dealt": 0, "enemies_damaged": 0, "teammates_damaged": 0}
    if gtype == "HE":
        weapons = ["hegrenade", "he_grenade", "he grenade"]
        ticks   = _ticks(2.0, tick_rate)
    elif gtype in ("Molotov", "Incendiary"):
        weapons = ["inferno", "molotov", "incgrenade", "firebomb", "fire"]
        ticks   = _ticks(MOLOTOV_LIFE_S, tick_rate)
    else:
        return {}
    dmg = r_damages[
        (r_damages["tick"] >= int(throw_tick)) &
        (r_damages["tick"] <= int(throw_tick) + ticks) &
        (r_damages["attacker_name"] == thrower) &
        (r_damages["weapon"].astype(str).str.lower().isin(weapons))
    ]
    if dmg.empty:
        return {"damage_dealt": 0, "enemies_damaged": 0, "teammates_damaged": 0}
    dmg_col = next((c for c in ["dmg_health", "dmg_health_real", "damage"] if c in dmg.columns), None)
    damage_dealt = int(round(float(dmg[dmg_col].sum()))) if dmg_col else 0
    if thrower_side in ("ct", "t") and name_to_side and "victim_name" in dmg.columns:
        enemy_side = _enemy_of(thrower_side)
        victims = dmg["victim_name"].unique()
        enemies_damaged   = sum(1 for v in victims if name_to_side.get(v) == enemy_side)
        teammates_damaged = sum(1 for v in victims if name_to_side.get(v) == thrower_side)
    else:
        enemies_damaged   = int(dmg["victim_name"].nunique()) if "victim_name" in dmg.columns else 0
        teammates_damaged = 0
    return {
        "damage_dealt":      damage_dealt,
        "enemies_damaged":   enemies_damaged,
        "teammates_damaged": teammates_damaged,
    }


def _toward_bomb(from_pos, to_pos, bomb_pos) -> bool | None:
    if bomb_pos is None or None in bomb_pos or None in from_pos or None in to_pos:
        return None
    try:
        mx = to_pos[0] - from_pos[0]
        my = to_pos[1] - from_pos[1]
        bx = bomb_pos[0] - from_pos[0]
        by = bomb_pos[1] - from_pos[1]
        return (mx * bx + my * by) > 0
    except Exception:
        return None


def _precompute_landings(ticks_df, enemy_side: str) -> list[tuple]:
    """Pre-computes enemy landing events for a round. Returns sorted list of (tick, x, y, z)."""
    if ticks_df is None or ticks_df.empty:
        return []
    needed = {"velocity_Z", "X", "Y", "Z", "side", "name", "tick"}
    if not needed.issubset(ticks_df.columns):
        return []
    enemy_ticks = ticks_df[ticks_df["side"] == enemy_side].sort_values("tick")
    landings = []
    for _, e_df in enemy_ticks.groupby("name"):
        vz    = e_df["velocity_Z"].to_numpy(dtype=float, na_value=0.0)
        ticks = e_df["tick"].to_numpy()
        xs    = e_df["X"].to_numpy(dtype=float, na_value=0.0)
        ys    = e_df["Y"].to_numpy(dtype=float, na_value=0.0)
        zs    = e_df["Z"].to_numpy(dtype=float, na_value=0.0)
        for i in range(1, len(vz)):
            if vz[i - 1] < -150 and abs(vz[i]) < 50:
                landings.append((int(ticks[i]), float(xs[i]), float(ys[i]), float(zs[i])))
    landings.sort(key=lambda x: x[0])
    return landings


def _heard_enemy(
    tick, player_name, player_side, player_pos, player_yaw,
    footsteps_df, shots_df, reloads_df, ticks_df,
    precomputed_landings: dict = None,
    precomputed_enemy_names: dict = None,
    window_ticks: int = None,
    tick_rate: int = 64,
) -> dict:
    """
    Checks all sound types (footsteps, gunshots, landings, reloads) from enemies.
    Returns heard_enemy bool and list of heard_sounds with type/direction/distance.
    """
    import bisect

    result = {"heard_enemy": False, "heard_sounds": []}
    if player_pos is None or None in player_pos or tick is None:
        return result

    enemy_side   = "t" if player_side == "ct" else "ct"
    enemy_names  = (precomputed_enemy_names or {}).get(enemy_side) or _get_enemy_names(ticks_df, enemy_side)
    window_ticks = _ticks(10.0, tick_rate) if window_ticks is None else window_ticks
    t_start      = tick - window_ticks
    heard_sounds = []

    def _check_source(source_df, sound_type: str, range_override=None):
        import numpy as np
        if source_df is None or source_df.empty:
            return
        recent = _tick_slice(source_df, int(t_start), int(tick))
        if recent.empty:
            return

        if "side" in recent.columns:
            enemy_rows = recent[recent["side"] == enemy_side]
        elif "name" in recent.columns and enemy_names:
            enemy_rows = recent[recent["name"].isin(enemy_names)]
        else:
            return
        if enemy_rows.empty or "X" not in enemy_rows.columns or "Y" not in enemy_rows.columns:
            return

        px, py = float(player_pos[0]), float(player_pos[1])
        xs = enemy_rows["X"].to_numpy(dtype=float, na_value=0.0)
        ys = enemy_rows["Y"].to_numpy(dtype=float, na_value=0.0)
        dists = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)

        if sound_type == "gunshot" and "weapon" in enemy_rows.columns:
            weps = enemy_rows["weapon"].fillna("").astype(str).str.lower().to_numpy()
            silenced = np.array([any(s in w for s in _SILENCED_WEAPONS) for w in weps])
            hearing_ranges = np.where(silenced, _SOUND_RANGES["gunshot_silenced"], _SOUND_RANGES["gunshot"])
            actual_types   = np.where(silenced, "gunshot_silenced", "gunshot")
        else:
            hr = range_override or _SOUND_RANGES.get(sound_type, 600)
            hearing_ranges = np.full(len(dists), hr)
            actual_types   = None

        in_range_idx = np.where(dists <= hearing_ranges)[0]
        if len(in_range_idx) == 0:
            return

        weapons_col = enemy_rows["weapon"].to_numpy() if sound_type == "gunshot" and "weapon" in enemy_rows.columns else None

        for i in in_range_idx:
            dx, dy = xs[i] - px, ys[i] - py
            dist = float(dists[i])
            angle_to_target = math.degrees(math.atan2(dy, dx))
            facing = -float(player_yaw)
            relative = (angle_to_target - facing) % 360
            if relative < 45 or relative >= 315:
                direction = "front"
            elif relative < 135:
                direction = "left"
            elif relative < 225:
                direction = "back"
            else:
                direction = "right"

            bucket      = "close" if dist < 200 else "medium" if dist < 500 else "far"
            actual_type = actual_types[i] if actual_types is not None else sound_type
            entry = {"type": actual_type, "direction": direction, "distance": bucket}
            if sound_type == "gunshot" and weapons_col is not None:
                entry["weapon"] = str(weapons_col[i] or "unknown")
            if entry not in heard_sounds:
                heard_sounds.append(entry)

    _check_source(footsteps_df, "footstep")
    _check_source(shots_df, "gunshot")
    _check_source(reloads_df, "reload")

    # Landings: use pre-computed list with bisect for O(log n) lookup
    if precomputed_landings is not None:
        land_events = precomputed_landings.get(enemy_side, [])
        if land_events:
            tick_keys = [l[0] for l in land_events]
            lo = bisect.bisect_left(tick_keys, t_start)
            hi = bisect.bisect_right(tick_keys, tick)
            for land in land_events[lo:hi]:
                land_pos = [land[1], land[2], land[3]]
                dist = _dist2(player_pos[:2], land_pos[:2])
                if dist <= _SOUND_RANGES["landing"]:
                    direction = _relative_direction(player_pos, player_yaw, land_pos)
                    bucket    = "close" if dist < 200 else "medium" if dist < 400 else "far"
                    entry = {"type": "landing", "direction": direction, "distance": bucket}
                    if entry not in heard_sounds:
                        heard_sounds.append(entry)

    result["heard_enemy"]  = len(heard_sounds) > 0
    result["heard_sounds"] = heard_sounds
    return result


def _round_state(
    tick, player_name, player_side, player_pos, player_yaw,
    ticks_df, damages_df, footsteps_df, shots_df, reloads_df,
    bomb_planted_ticks: set, r_freeze, r_offend,
    precomputed_landings: dict = None,
    precomputed_enemy_names: dict = None,
    snap=None,
    vis_cache: dict = None,
    tick_rate: int = 64,
):
    """
    Full round state context at a given tick.
    """
    if snap is None:
        snap = _get_tick_snap(ticks_df, tick)

    alive_teammates, alive_enemies = _alive_counts_from_snap(snap, player_name, player_side)

    # Clutch
    clutch = isinstance(alive_teammates, int) and alive_teammates == 0

    # Post-plant
    post_plant = bool(bomb_planted_ticks and tick is not None and
                      any(t <= tick for t in bomb_planted_ticks))

    # Time remaining (rough: ticks from now to round end)
    time_remaining = None
    if r_offend is not None and tick is not None:
        try:
            time_remaining = round((float(r_offend) - float(tick)) / tick_rate, 1)
        except Exception:
            pass

    # Has enemy info from damage (seen or taken damage in last 15s)
    has_damage_info = False
    if not damages_df.empty and tick is not None:
        window_start = tick - _ticks(15.0, tick_rate)
        recent = damages_df[damages_df["tick"] >= window_start]
        if not recent.empty and "attacker_name" in recent.columns:
            has_damage_info = (
                recent["attacker_name"].eq(player_name).any() or
                recent["victim_name"].eq(player_name).any()
            )

    # Spotted enemies — fast lookup into pre-computed visibility cache
    spotted = _lookup_spotted(vis_cache, player_name, tick) if vis_cache else []

    # Heard enemy
    p_yaw = player_yaw if player_yaw is not None else 0.0
    heard = _heard_enemy(tick, player_name, player_side, player_pos, p_yaw,
                         footsteps_df, shots_df, reloads_df, ticks_df,
                         precomputed_landings=precomputed_landings,
                         precomputed_enemy_names=precomputed_enemy_names)

    # Bomb position: check C4 carrier in snap only (snap is ≤10 rows)
    bomb_pos = None
    if snap is not None and not snap.empty and "inventory" in snap.columns:
        for _, row in snap.iterrows():
            inv = _safe(row.get("inventory"), [])
            if isinstance(inv, list) and any("c4" in str(w).lower() for w in inv):
                bomb_pos = _pos(row)
                break

    return {
        "alive_teammates":  alive_teammates,
        "alive_enemies":    alive_enemies,
        "clutch":           clutch,
        "post_plant":       post_plant,
        "time_remaining":   time_remaining,
        "has_damage_info":  has_damage_info,
        "enemies_spotted":  spotted,
        "radar_enemies":    [s["position"] for s in spotted],
        "heard_enemy":      heard["heard_enemy"],
        "heard_sounds":     heard["heard_sounds"],
        "bomb_position":    bomb_pos,
    }


def _utility_in_inventory(row) -> list[str]:
    inv = _safe(row.get("inventory"), [])
    if not isinstance(inv, list):
        return []
    grenade_keywords = ["grenade", "flash", "smoke", "molotov", "decoy", "incendiary", "he"]
    return [w for w in inv if any(k in w.lower() for k in grenade_keywords)]


def _build_round_history(rounds_df, current_round_num) -> list[dict]:
    """Returns list of {round, winner} for all rounds before current."""
    history = []
    for _, r in rounds_df.iterrows():
        rn = int(r.get("round_num", 0) or 0)
        if rn >= current_round_num:
            continue
        winner = _safe(r.get("winner"), "?")
        history.append({"round": rn, "winner": winner})
    return sorted(history, key=lambda x: x["round"])


def build_vis_checker(map_name: str):
    try:
        from awpy.data import TRIS_DIR
        from awpy.visibility import VisibilityChecker
        tri_path = TRIS_DIR / f"{map_name}.tri"
        if tri_path.exists():
            vc = VisibilityChecker(path=tri_path)
            print(f"  visibility checker loaded for {map_name}", flush=True)
            return vc
        else:
            print(f"  no .tri file for {map_name}, skipping LOS checks", flush=True)
    except Exception as e:
        print(f"  VisibilityChecker unavailable: {e}", flush=True)
    return None


def extract_events(tables: dict, match_id: str, map_name: str = "unknown", vis_checker=None) -> list[dict]:
    import pandas as pd

    if vis_checker is None:
        vis_checker = build_vis_checker(map_name)

    rounds_df    = tables.get("rounds",     pd.DataFrame())
    ticks_df     = tables.get("ticks",      pd.DataFrame())
    kills_df     = tables.get("kills",      pd.DataFrame())
    damages_df   = tables.get("damages",    pd.DataFrame())
    grenades_df  = tables.get("grenades",   pd.DataFrame())
    infernos_df  = tables.get("infernos",   pd.DataFrame())
    smokes_df    = tables.get("smokes",     pd.DataFrame())
    footsteps_df    = tables.get("footsteps",  None)
    shots_df        = tables.get("shots",      None)
    reloads_df      = tables.get("event_weapon_reload", None)

    if ticks_df.empty or "tick" not in ticks_df.columns:
        return []
    bomb_planted_df = tables.get("event_bomb_planted",  pd.DataFrame())
    bomb_defused_df = tables.get("event_bomb_defused",  pd.DataFrame())
    bomb_explode_df = tables.get("event_bomb_exploded", pd.DataFrame())

    # Collect bomb plant ticks per round (vectorized — no iterrows)
    bomb_planted_by_round: dict[int, set] = {}
    for table_key in ["event_bomb_planted", "bomb"]:
        tbl = tables.get(table_key, pd.DataFrame())
        if tbl.empty or "tick" not in tbl.columns or "round_num" not in tbl.columns:
            continue
        # If table has an event/type/action column, filter to plant rows only
        filtered = tbl
        for c in ["event", "type", "action"]:
            if c in tbl.columns:
                filtered = tbl[tbl[c].astype(str).str.contains("plant", case=False, na=False)]
                break
        if filtered.empty:
            continue
        for rn, tick_val in zip(filtered["round_num"].astype(int), filtered["tick"].astype(int)):
            bomb_planted_by_round.setdefault(int(rn), set()).add(int(tick_val))

    if rounds_df.empty:
        return []

    tick_rate = int(tables.get("tick_rate") or 64)
    events: list[dict] = []

    for _, r in rounds_df.iterrows():
        try:
            rn = int(r["round_num"])
        except (ValueError, TypeError):
            continue
        r_freeze  = r.get("freeze_end")
        r_offend  = _first_valid(r.get("official_end"), r.get("end"))
        # Normalise once. A casing change in the parser would otherwise silently make
        # every round_won 0 rather than raising.
        round_won = str(_safe(r.get("winner"), "?") or "?").strip().lower()
        bomb_ticks = bomb_planted_by_round.get(rn, set())
        round_history = _build_round_history(rounds_df, rn)

        def secs(tick) -> float:
            if r_freeze is None or tick is None:
                return 0.0
            return round((float(tick) - float(r_freeze)) / tick_rate, 2)

        if _nan(r_freeze) or r_freeze is None or _nan(r_offend) or r_offend is None:
            continue
        if round_won not in ("ct", "t"):
            print(f"  [extractor] round {rn}: unknown winner {round_won!r}, skipping")
            continue

        r_kills   = kills_df[kills_df["round_num"] == rn] if not kills_df.empty else pd.DataFrame()
        r_damages = damages_df[damages_df["round_num"] == rn] if not damages_df.empty else pd.DataFrame()

        # Pre-filter all tick-indexed DataFrames to the current round's tick range
        try:
            _r_start = float(r_freeze)
            _r_end   = float(r_offend)
        except (ValueError, TypeError):
            continue

        def _rf(df):
            if df is None or (hasattr(df, "empty") and df.empty) or "tick" not in df.columns:
                return df
            filtered = df[(df["tick"] >= _r_start) & (df["tick"] <= _r_end)]
            return filtered.sort_values("tick").reset_index(drop=True)

        r_ticks_df     = _rf(ticks_df)
        r_footsteps_df = _rf(footsteps_df)
        r_shots_df     = _rf(shots_df)
        r_reloads_df   = _rf(reloads_df)

        # Pre-compute enemy landing events and enemy name sets once per round per side
        precomputed_landings = {
            "ct": _precompute_landings(r_ticks_df, "ct"),
            "t":  _precompute_landings(r_ticks_df, "t"),
        }
        precomputed_enemy_names = {
            "ct": _get_enemy_names(r_ticks_df, "ct"),
            "t":  _get_enemy_names(r_ticks_df, "t"),
        }
        r_smokes = smokes_df[smokes_df["round_num"] == rn] if not smokes_df.empty and "round_num" in smokes_df.columns else pd.DataFrame()
        vis_cache = _precompute_visibility(r_ticks_df, vis_checker, smokes_df=r_smokes)
        name_to_side: dict[str, str] = {}
        for _s in ("ct", "t"):
            for _n in precomputed_enemy_names.get(_s, set()):
                name_to_side[_n] = _s

        # Only use Projectile rows - they have valid X/Y/Z and represent actual throws.
        # Base types (CSmokeGrenade, CHEGrenade, etc.) fire on aim-start and have null positions.
        if not grenades_df.empty:
            rg = grenades_df[grenades_df["round_num"] == rn]
            if not rg.empty and "entity_id" in rg.columns:
                _pos_cols = [c for c in ["X", "Y", "Z"] if c in rg.columns]
                rg_valid  = rg.dropna(subset=_pos_cols) if _pos_cols else rg
                if rg_valid.empty:
                    # Keep the columns — a bare pd.DataFrame() has none, and the
                    # .sort_values("tick") below would raise KeyError and abort the
                    # whole match after the expensive parse.
                    r_grens = rg.iloc[0:0]
                else:
                    rg_sorted     = rg_valid.sort_values("tick")
                    r_grens_first = rg_sorted.groupby("entity_id").first().reset_index()
                    r_grens_last  = rg_sorted.groupby("entity_id").last().reset_index()
                    r_grens_last  = r_grens_last.rename(columns={"X": "land_X", "Y": "land_Y", "Z": "land_Z"})
                    r_grens = r_grens_first.merge(
                        r_grens_last[["entity_id", "land_X", "land_Y", "land_Z"]],
                        on="entity_id", how="left"
                    ).reset_index(drop=True)
            else:
                r_grens = rg
        else:
            r_grens = grenades_df.iloc[0:0]

        # Incendiary only: no Projectile type exists, use infernos table (CT side).
        # Molotov uses CMolotovProjectile from grenades_df (already included above).
        if not infernos_df.empty and "round_num" in infernos_df.columns and "thrower_side" in infernos_df.columns:
            r_inf = infernos_df[(infernos_df["round_num"] == rn) & (infernos_df["thrower_side"] == "ct")]
            if not r_inf.empty:
                incend_rows = pd.DataFrame({
                    "entity_id":    r_inf["entity_id"].values if "entity_id" in r_inf.columns else range(len(r_inf)),
                    "tick":         r_inf["start_tick"].values,
                    "thrower":      r_inf["thrower_name"].values,
                    "grenade_type": "CIncendiaryGrenade",
                    "X":            r_inf["thrower_X"].values,
                    "Y":            r_inf["thrower_Y"].values,
                    "Z":            r_inf["thrower_Z"].values,
                    "land_X":       r_inf["X"].values,
                    "land_Y":       r_inf["Y"].values,
                    "land_Z":       r_inf["Z"].values,
                    "round_num":    rn,
                })
                r_grens = pd.concat([r_grens, incend_rows], ignore_index=True) if not r_grens.empty else incend_rows

        def _base(player_side: str) -> dict:
            return {
                "match_id":    match_id,
                "map":         map_name,
                "round_num":   rn,
                "player_side": player_side,
                "round_won":   1 if round_won == player_side else 0,
            }

        # ── Economy (buy decision) ───────────────────────────────────────────
        if not r_ticks_df.empty and r_freeze is not None:
            snap = r_ticks_df[r_ticks_df["tick"] == r_freeze]
            if snap.empty:
                snap = r_ticks_df[
                    (r_ticks_df["tick"] >= r_freeze) & (r_ticks_df["tick"] <= r_freeze + tick_rate)
                ].sort_values("tick").groupby("name").first().reset_index()

            for _, row in snap.iterrows():
                side       = _safe(row.get("side"), "?")
                equip      = float(_safe(row.get("current_equip_value"), 0) or 0)
                armor      = _safe(row.get("armor"), 0)
                helmet     = bool(row.get("has_helmet", False))
                defuser    = bool(row.get("has_defuser", False))
                inv        = _safe(row.get("inventory"), [])
                cash       = float(_safe(row.get("cash"), 0) or 0)
                cash_spent = float(_safe(row.get("cash_spent_this_round"), 0) or 0)
                pos        = _pos(row)
                yaw        = _safe(row.get("yaw"), 0.0)

                team_snap       = snap[snap["side"] == side]
                enemy_snap      = snap[snap["side"] != side]
                team_equip_avg  = team_snap["current_equip_value"].mean() if "current_equip_value" in team_snap.columns else 0
                enemy_equip_avg = enemy_snap["current_equip_value"].mean() if "current_equip_value" in enemy_snap.columns else 0

                weapons = [w for w in (inv if isinstance(inv, list) else [])
                           if w and "knife" not in w.lower() and "taser" not in w.lower()]

                pname = _safe(row.get("name"), "?")

                ev = _base(side)
                ev.update({
                    "event_type":      "buy",
                    # 0.0 is correct: a buy happens at freeze end, which is where the
                    # round clock starts. It collides with the round's first state
                    # sample, which is exactly why a buy must NOT be bracketed like an
                    # in-round action — see impact_kind below.
                    "time_into_round": 0.0,
                    "situation": {
                        "cash":             round(cash),
                        "own_equip_value":  round(equip),
                        # `if x` doesn't catch NaN (NaN is truthy) and round(nan) raises.
                        "team_equip_avg":   0 if _nan(team_equip_avg)  else round(float(team_equip_avg or 0)),
                        "enemy_equip_avg":  0 if _nan(enemy_equip_avg) else round(float(enemy_equip_avg or 0)),
                        "round_history":    round_history,
                        "round_num":        rn,
                        "round_type":       _round_type(equip, side),
                    },
                    "action": {
                        "equip_value": round(equip),
                        "cash_spent":  round(cash_spent),
                        "armor":       _safe(armor, 0),
                        "impact_kind": KIND_ROUND,
                        "helmet":      helmet,
                        "defuser":     defuser,
                        "weapons":     weapons,
                        "inventory":   inv if isinstance(inv, list) else [],
                        "round_type":  _round_type(equip, side),
                    },
                    "outcome": {
                        "round_won": 1 if round_won == side else 0,
                    },
                })
                events.append(ev)

        # ── Engagements (kills) ──────────────────────────────────────────────
        for _, row in r_kills.iterrows():
            tick = row.get("tick")
            t    = secs(tick)
            snap = _get_tick_snap(r_ticks_df, tick)

            for role, name_col, side_col, pos_prefix, other_prefix in [
                ("attacker", "attacker_name", "attacker_side", "attacker_", "victim_"),
                ("victim",   "victim_name",   "victim_side",   "victim_",   "attacker_"),
            ]:
                pname     = _safe(row.get(name_col), "?")
                side      = _safe(row.get(side_col), "?")
                pos       = _pos(row, pos_prefix)
                enemy_pos = _pos(row, other_prefix)
                place     = _safe(row.get(f"{pos_prefix}place"), "?")
                hp        = _safe(row.get(f"{pos_prefix}health"), 100)
                armor     = _safe(row.get(f"{pos_prefix}armor"), 0)
                weapon    = _safe(row.get("weapon"), "?")
                distance  = round(float(row.get("distance", 0) or 0), 1)

                p_row      = _player_row(snap, pname)
                velocity   = round(_vel_mag(p_row), 1) if p_row is not None else 0.0
                pitch      = round(float(p_row.get("pitch", 0) or 0), 2) if p_row is not None else None
                yaw        = round(float(p_row.get("yaw", 0) or 0), 2) if p_row is not None else None
                flash_dur  = round(float(p_row.get("flash_duration", 0) or 0), 2) if p_row is not None else 0.0
                zoom_lvl   = int(_safe(p_row.get("zoom_lvl"), 0) or 0) if p_row is not None else 0
                acc_pen    = round(float(p_row.get("accuracy_penalty", 0) or 0), 3) if p_row is not None else 0.0
                utility    = _utility_in_inventory(p_row) if p_row is not None else []

                # Trade kill: did this player avenge a recently killed teammate?
                trade_kill = None
                if not r_kills.empty and tick is not None and "victim_side" in r_kills.columns:
                    recent_deaths = r_kills[
                        (r_kills["tick"] >= tick - _ticks(3.0, tick_rate)) &
                        (r_kills["tick"] < tick) &
                        (r_kills["victim_side"] == side)
                    ]
                    # Only the attacker can trade; for a victim the answer is N/A, and
                    # returning False there would be indistinguishable from "not a trade".
                    if role == "attacker" and not recent_deaths.empty:
                        trade_kill = False
                        killed = _safe(row.get("victim_name"), "")
                        # Scan ALL nearby recent deaths — breaking on the first (which
                        # is the OLDEST, since r_kills is tick-ordered) misses the trade
                        # whenever two teammates died close together.
                        for _, td in recent_deaths.iterrows():
                            td_pos = _pos(td, "victim_")
                            if None not in pos and None not in td_pos and _dist3(pos, td_pos) <= 500:
                                if killed and killed == _safe(td.get("attacker_name"), ""):
                                    trade_kill = True
                                    break

                e_name         = _safe(row.get("victim_name" if role == "attacker" else "attacker_name"), "?")
                e_row          = _player_row(snap, e_name)
                enemy_velocity = round(_vel_mag(e_row), 1) if e_row is not None else 0.0
                enemy_flash    = round(float(e_row.get("flash_duration", 0) or 0), 2) if e_row is not None else 0.0

                if velocity > 100 and velocity > enemy_velocity:
                    peek_role = "peeked"
                elif enemy_velocity > 100 and enemy_velocity > velocity:
                    peek_role = "got_peeked"
                else:
                    peek_role = "stationary_duel"

                rstate = _round_state(
                    tick, pname, side, pos, yaw,
                    r_ticks_df, r_damages, r_footsteps_df, r_shots_df, r_reloads_df,
                    bomb_ticks, r_freeze, r_offend,
                    precomputed_landings=precomputed_landings,
                    precomputed_enemy_names=precomputed_enemy_names, snap=snap, vis_cache=vis_cache,
                    tick_rate=tick_rate,
                )

                ev = _base(side)
                ev.update({
                    "event_type":      "engagement",
                    "time_into_round": t,
                    "situation": {
                        "position":          pos,
                        "area":              place,
                        "hp":                _safe(hp, 100),
                        "armor":             _safe(armor, 0),
                        "weapon":            weapon,
                        "velocity":          velocity,
                        "flash_duration":    flash_dur,
                        "utility_available": utility,
                        "time_into_round":   t,
                        **rstate,
                    },
                    "action": {
                        "impact_kind":     KIND_INSTANT,
                        "resolve_tick":    int(tick) if tick is not None else None,
                        "role":            role,
                        "weapon":          weapon,
                        "headshot":        bool(row.get("headshot", False)),
                        "through_smoke":   bool(row.get("thrusmoke", False)),
                        "attacker_blind":  bool(row.get("attackerblind", False)),
                        "noscope":         bool(row.get("noscope", False)),
                        "distance":        distance,
                        "peek_role":       peek_role,
                        "pitch":           pitch,
                        "yaw":             yaw,
                        "attacker_place":  _safe(row.get("attacker_place"), "?"),
                        "victim_place":    _safe(row.get("victim_place"),   "?"),
                        "enemy_position":  enemy_pos,
                        "enemy_velocity":  enemy_velocity,
                        "enemy_flashed":   enemy_flash > 0.5,
                        "zoom_lvl":        zoom_lvl,
                        "accuracy_penalty": acc_pen,
                        "trade_kill":      trade_kill,
                        "was_peeking":     peek_role == "peeked",
                    },
                    "outcome": {
                        "result":       "kill" if role == "attacker" else "death",
                        "round_won":    1 if round_won == side else 0,
                        "kills_after":  int(r_kills[
                            (r_kills["tick"] > tick) & (r_kills["attacker_name"] == pname)
                        ].shape[0]) if not r_kills.empty and tick is not None and "attacker_name" in r_kills.columns else 0,
                        # awpy's column is dmg_health, not damage — see the same
                        # resolution in _grenade_damage. None (not 0) when absent, so
                        # "no damage column" stays distinguishable from "did no damage".
                        "damage_after": _damage_after(r_damages, tick, pname),
                        "died_after":   not r_kills[
                            (r_kills["tick"] > tick) & (r_kills["victim_name"] == pname)
                        ].empty if not r_kills.empty and tick is not None and "victim_name" in r_kills.columns else False,
                    },
                })
                events.append(ev)

        # ── Utility throws ───────────────────────────────────────────────────
        for _, row in r_grens.sort_values("tick").iterrows():
            tick    = row.get("tick")
            t       = secs(tick)
            thrower = _safe(row.get("thrower"), "?")
            gtype   = _GRENADE_MAP.get(_safe(row.get("grenade_type"), "?"), _safe(row.get("grenade_type"), "?"))

            if r_freeze is not None and tick is not None and float(tick) < float(r_freeze):
                continue

            side = "?"
            pos  = [None, None, None]
            yaw  = 0.0
            if not r_ticks_df.empty and thrower != "?":
                p_snap = r_ticks_df[r_ticks_df["name"] == thrower]
                if not p_snap.empty:
                    if "side" in p_snap.columns:
                        side = str(p_snap.iloc[0]["side"])
                    # Use row closest to throw tick, not just first row
                    closest_idx = (p_snap["tick"] - float(tick)).abs().argmin()
                    closest_row = p_snap.iloc[closest_idx]
                    pos = _pos(closest_row)
                    yaw = float(closest_row.get("yaw", 0) or 0)
            # Fall back to the per-round name→side map before giving up: an unresolved
            # side would make every `"t" if side == "ct" else "ct"` below resolve to
            # "ct", silently treating this thrower's own team as the enemy.
            if side == "?" and thrower != "?":
                side = name_to_side.get(thrower, "?")
            # Fallback: use grenade's own X/Y/Z (Projectile types have valid positions)
            if None in pos:
                gren_pos = _pos(row)
                if None not in gren_pos:
                    pos = gren_pos

            kill_after = False
            if not r_kills.empty and tick is not None:
                window = r_kills[
                    (r_kills["tick"] >= tick) &
                    (r_kills["tick"] <= tick + _ticks(5.0, tick_rate))
                ]
                kill_after = not window.empty

            throw_pos = _pos(row)
            land_pos  = [_safe(row.get("land_X")), _safe(row.get("land_Y")), _safe(row.get("land_Z"))]
            try:
                land_pos = [round(float(v), 1) if v is not None else None for v in land_pos]
            except Exception:
                land_pos = [None, None, None]

            snap   = _get_tick_snap(r_ticks_df, tick)
            rstate = _round_state(
                tick, thrower, side, pos, yaw,
                r_ticks_df, r_damages, r_footsteps_df, r_shots_df, r_reloads_df,
                bomb_ticks, r_freeze, r_offend,
                precomputed_landings=precomputed_landings,
                precomputed_enemy_names=precomputed_enemy_names, snap=snap, vis_cache=vis_cache,
                    tick_rate=tick_rate,
                )

            ev = _base(side)
            ev.update({
                "event_type":      "utility",
                "time_into_round": t,
                "situation": {
                    "time_into_round": t,
                    "grenade_type":    gtype,
                    **rstate,
                },
                "action": {
                    "grenade_type":   gtype,
                    "throw_position": throw_pos,
                    "land_position":  land_pos,
                    # A smoke's value accrues over its lifetime, not at the throw —
                    # so it is scored on residual drift across this window.
                    **attribution_window(gtype, tick, tick_rate,
                                         start_tick=row.get("start_tick"),
                                         end_tick=row.get("end_tick")),
                },
                "outcome": {
                    "kill_within_5s": kill_after,
                    "round_won":      1 if round_won == side else 0,
                    # Impact blocks need a known thrower side to identify enemies.
                    # With side unresolved they would count this thrower's own team.
                    **(
                        _smoke_impact(land_pos, tick, r_ticks_df, r_kills, _enemy_of(side), tick_rate)
                        if gtype == "Smoke" and side in ("ct", "t") else {}
                    ),
                    **(
                        _flash_impact(tick, r_ticks_df, side, _enemy_of(side), tick_rate)
                        if gtype == "Flash" and side in ("ct", "t") else {}
                    ),
                    **(
                        _grenade_damage(tick, thrower, gtype, r_damages, thrower_side=side,
                                        name_to_side=name_to_side, tick_rate=tick_rate)
                        if gtype in ("HE", "Molotov", "Incendiary") and side in ("ct", "t") else {}
                    ),
                },
            })
            events.append(ev)

        # ── Rotations (sampled every 1s, only if moved >200 units) ───────────
        ROTATION_TICK_INTERVAL = tick_rate          # ~1 s
        MOVE_SAMPLE_INTERVAL   = max(1, tick_rate // 2)   # ~0.5 s
        ROTATION_DIST_THRESHOLD = 200.0

        if not r_ticks_df.empty:
            round_ticks = r_ticks_df[r_ticks_df["tick"] % ROTATION_TICK_INTERVAL == 0].copy()
            round_ticks = round_ticks.sort_values(["name", "tick"])

            all_round_ticks = r_ticks_df.sort_values(["name", "tick"])

            for pname, p_df in round_ticks.groupby("name"):
                side = str(p_df.iloc[0].get("side", "?")) if "side" in p_df.columns else "?"
                prev_pos  = None
                prev_area = None
                prev_tick = None

                # Full tick data for this player (for movement window analysis)
                p_all = all_round_ticks[all_round_ticks["name"] == pname] if "name" in all_round_ticks.columns else p_df.copy()

                for _, row in p_df.iterrows():
                    alive = bool(row.get("is_alive", True))
                    if not alive:
                        prev_pos  = None
                        prev_area = None
                        prev_tick = None
                        continue

                    cur_pos  = _pos(row)
                    cur_area = _safe(row.get("place"), "?")
                    cur_tick = row.get("tick")
                    yaw      = float(row.get("yaw", 0) or 0)

                    if prev_pos is not None and None not in cur_pos and None not in prev_pos:
                        dist = _dist3(cur_pos, prev_pos)
                        if dist >= ROTATION_DIST_THRESHOLD:
                            tick = cur_tick
                            t    = secs(tick)
                            hp   = _safe(row.get("health"), 100)

                            # Analyze movement over the whole window (every 32 ticks = 0.5s)
                            # Build run-length encoded intervals: [{type, from, to}, ...]
                            movement_types = []
                            made_sound = False
                            if prev_tick is not None and tick is not None:
                                window_samples = p_all[
                                    (p_all["tick"] >= prev_tick) &
                                    (p_all["tick"] <= tick) &
                                    (p_all["tick"] % MOVE_SAMPLE_INTERVAL == 0)
                                ].sort_values("tick")

                                cur_seg_type  = None
                                cur_seg_start = None

                                for _, s in window_samples.iterrows():
                                    vel    = _vel_mag(s)
                                    vz     = float(s.get("velocity_Z") or 0)
                                    s_tick = s.get("tick")
                                    s_time = secs(s_tick)

                                    if abs(vz) > 200:
                                        s_type = "jumping"
                                        made_sound = True
                                    elif vel > 170:
                                        s_type = "running"
                                        made_sound = True
                                    elif vel > 90:
                                        s_type = "walking"
                                    elif vel > 0:
                                        s_type = "crouching"
                                    else:
                                        s_type = "stationary"

                                    if s_type != cur_seg_type:
                                        if cur_seg_type is not None:
                                            movement_types[-1]["to"] = round(s_time, 2)
                                        cur_seg_type  = s_type
                                        cur_seg_start = s_time
                                        movement_types.append({"type": s_type, "from": round(s_time, 2), "to": None})

                                # Close last segment
                                if movement_types:
                                    movement_types[-1]["to"] = round(secs(tick), 2)

                            if not movement_types:
                                vel = _vel_mag(row)
                                t   = secs(tick)
                                s_type = "running" if vel > 170 else "walking" if vel > 90 else "crouching" if vel > 0 else "stationary"
                                made_sound = vel > 250
                                movement_types = [{"type": s_type, "from": round(t, 2), "to": round(t, 2)}]

                            snap = _get_tick_snap(r_ticks_df, tick)
                            rstate = _round_state(
                                tick, pname, side, cur_pos, yaw,
                                r_ticks_df, r_damages, r_footsteps_df, r_shots_df, r_reloads_df,
                                bomb_ticks, r_freeze, r_offend,
                                precomputed_landings=precomputed_landings,
                                precomputed_enemy_names=precomputed_enemy_names, snap=snap, vis_cache=vis_cache,
                    tick_rate=tick_rate,
                )

                            ev = _base(side)
                            ev.update({
                                "event_type":      "rotation",
                                "time_into_round": t,
                                "situation": {
                                    "from_area":       prev_area,
                                    "from_position":   prev_pos,
                                    "time_into_round": t,
                                    "hp":              _safe(hp, 100),
                                    **rstate,
                                },
                                "action": {
                                    # Movement has no instant: its value is whatever
                                    # win probability accrued over the traversal and
                                    # that no discrete event explains.
                                    "impact_kind":       KIND_DURATIVE,
                                    "window_start_tick": int(prev_tick) if prev_tick is not None else None,
                                    "window_end_tick":   int(tick),
                                    "moved_to_area":     cur_area,
                                    "moved_to_position": cur_pos,
                                    "distance_moved":    round(dist, 1),
                                    "movement_type":     movement_types,
                                    "made_sound":        made_sound,
                                    "toward_bomb":       _toward_bomb(prev_pos, cur_pos, rstate.get("bomb_position")) if rstate.get("post_plant") else None,
                                },
                                "outcome": {
                                    "round_won": 1 if round_won == side else 0,
                                },
                            })
                            events.append(ev)

                    prev_pos  = cur_pos
                    prev_area = cur_area
                    prev_tick = cur_tick

        # ── Bomb events (plant / defuse / explode) ──────────────────────────
        C4_TIMER = 40.0  # seconds

        def _bomb_rows(df):
            if df is None or df.empty or "tick" not in df.columns:
                return df if df is not None else pd.DataFrame()
            return df[(df["tick"] >= _r_start) & (df["tick"] <= _r_end)]

        def _bomb_situation(tick, pname, pside, pos, yaw, snap):
            rstate = _round_state(
                tick, pname, pside, pos, yaw,
                r_ticks_df, r_damages, r_footsteps_df, r_shots_df, r_reloads_df,
                bomb_ticks, r_freeze, r_offend,
                precomputed_landings=precomputed_landings,
                precomputed_enemy_names=precomputed_enemy_names, snap=snap, vis_cache=vis_cache,
                    tick_rate=tick_rate,
                )
            return {"time_into_round": secs(tick), **rstate}

        # Plant
        for _, row in _bomb_rows(bomb_planted_df).iterrows():
            tick   = row.get("tick")
            pname  = _safe(row.get("user_name"), "?")
            pside  = _safe(row.get("user_side"), "t")
            pos    = [_safe(row.get("user_X")), _safe(row.get("user_Y")), _safe(row.get("user_Z"))]
            yaw    = float(row.get("user_yaw") or 0)
            site   = _safe(row.get("site"), "?")
            snap   = _get_tick_snap(r_ticks_df, tick)
            ev = _base(pside)
            ev.update({
                "event_type":      "bomb",
                "time_into_round": secs(tick),
                "situation":       _bomb_situation(tick, pname, pside, pos, yaw, snap),
                "action": {
                    "impact_kind":   KIND_INSTANT,
                    "resolve_tick":  int(tick),
                    "bomb_action":   "plant",
                    "site":          site,
                    "position":      pos,
                    "area":          _safe(row.get("user_place"), "?"),
                    "hp":            _safe(row.get("user_health"), 100),
                    "has_defuser":   bool(row.get("user_has_defuser", False)),
                },
                "outcome": {"round_won": 1 if round_won == pside else 0},
            })
            events.append(ev)

        # Defuse
        plant_tick = min(bomb_ticks) if bomb_ticks else None
        for _, row in _bomb_rows(bomb_defused_df).iterrows():
            tick        = row.get("tick")
            pname       = _safe(row.get("user_name"), "?")
            pside       = _safe(row.get("user_side"), "ct")
            pos         = [_safe(row.get("user_X")), _safe(row.get("user_Y")), _safe(row.get("user_Z"))]
            yaw         = float(row.get("user_yaw") or 0)
            had_kit     = bool(row.get("user_has_defuser", False))
            site        = _safe(row.get("site"), "?")
            snap        = _get_tick_snap(r_ticks_df, tick)
            defuse_dur  = 5.0 if had_kit else 10.0
            bomb_live_s = round((float(tick) - float(plant_tick)) / tick_rate, 2) if plant_tick else None
            time_left   = round(C4_TIMER - bomb_live_s, 2) if bomb_live_s is not None else None
            ev = _base(pside)
            ev.update({
                "event_type":      "bomb",
                "time_into_round": secs(tick),
                "situation":       _bomb_situation(tick, pname, pside, pos, yaw, snap),
                "action": {
                    "impact_kind":          KIND_INSTANT,
                    "resolve_tick":         int(tick),
                    "bomb_action":          "defuse",
                    "site":                 site,
                    "position":             pos,
                    "area":                 _safe(row.get("user_place"), "?"),
                    "hp":                   _safe(row.get("user_health"), 100),
                    "had_defuser":          had_kit,
                    "defuse_duration_s":    defuse_dur,
                    "bomb_live_seconds":    bomb_live_s,
                    "seconds_left_on_bomb": time_left,
                },
                "outcome": {"round_won": 1 if round_won == pside else 0},
            })
            events.append(ev)

        # Explode
        for _, row in _bomb_rows(bomb_explode_df).iterrows():
            tick        = row.get("tick")
            site        = _safe(row.get("site"), "?")
            snap        = _get_tick_snap(r_ticks_df, tick)
            bomb_live_s = round((float(tick) - float(plant_tick)) / tick_rate, 2) if plant_tick else None
            alive_t, alive_ct = 0, 0
            if snap is not None and not snap.empty:
                alive_t  = int(snap[(snap["side"] == "t")  & (snap["is_alive"] == True)].shape[0])
                alive_ct = int(snap[(snap["side"] == "ct") & (snap["is_alive"] == True)].shape[0])
            ev = _base("t")
            ev.update({
                "event_type":      "bomb",
                "time_into_round": secs(tick),
                "situation": {
                    "time_into_round": secs(tick),
                    "alive_t":         alive_t,
                    "alive_ct":        alive_ct,
                    "bomb_live_seconds": bomb_live_s,
                },
                "action":  {"impact_kind": KIND_INSTANT, "resolve_tick": int(tick),
                            "bomb_action": "explode", "site": site},
                "outcome": {"round_won": 1 if round_won == "t" else 0},
            })
            events.append(ev)

        print(f"  Round {rn}", flush=True)

    return events
