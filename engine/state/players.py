"""
Player token (P3.2): exactly the Appendix A.2 fields, in order. Unknown -> None; dead
players keep their last known position with alive = 0, health = 0, armor = 0 and every
DEAD_NULL_FIELDS entry None. Legitimate zeros (documented): health / armor of a dead
player, flash_duration_remaining when not blind, kills_round / damage_round before the
first kill / hit.

    from engine.state import players
    tok = players.token(sl, steamid, "ct", geometry, tier="pro", utility=utility_tokens)

`sl` is a causal slice from `engine.state.builder` (`RoundView.until(tick)`): every
table in it holds only rows with event tick <= sl.tick of the current round.

Derived-field rules:
- flash_duration_remaining comes from the `flashes` table (blind_duration minus the
  time since detonation, floored at 0), never from `ticks.flash_duration` (the parser
  contract and the task spec disagree on whether that column is onset or remaining).
- speed is the horizontal speed sqrt(vx^2 + vy^2) in u/s.
- heard_enemy: 1 when an enemy fired (weapon_fire) or stepped (footsteps) within the
  last `heard_window_s` seconds within HEARD_RADIUS units of the player's current
  position, else 0; None when dead. The enemy's position for a shot is their tick row
  at the shot tick.
- kills_round excludes team kills; damage_round sums dmg_health dealt to enemies.
"""
import hashlib
import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from engine.state.schema import PlayerToken

log = logging.getLogger("engine.state.players")

# ── constants ─────────────────────────────────────────────────────────────────

VOCAB_PATH      = Path(__file__).resolve().parent / "vocab_weapons.yaml"
WEAPON_CLASSES  = ("rifle", "sniper", "smg", "shotgun", "lmg", "pistol", "knife",
                   "grenade", "zeus", "c4", "unknown")
UNKNOWN_CLASS   = "unknown"
HEARD_RADIUS    = 1500.0     # u, sound cue range for heard_enemy
HEARD_WINDOW_S  = 2.0        # s, default when config has no sampling.heard_window_s
PAD_PLAYER_ID   = -1         # player_id of an empty slot (hashes are >= 0, so no clash)

NADE_FIELDS = {"smokegrenade": "n_smoke", "flashbang": "n_flash", "hegrenade": "n_he",
               "molotov": "n_molly", "incgrenade": "n_molly", "decoy": "n_decoy"}
FLAG_COLS   = ("is_scoped", "is_ducking", "is_walking", "is_reloading", "is_defusing", "is_planting")


# ── null helpers (`or` is not a null-coalesce: NaN is truthy, 0 is legitimate) ─

def _nan(v):
    """None for None / NaN, else the value unchanged (numpy scalars -> Python)."""
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _first_valid(*vals):
    """First value that is neither None nor NaN, else None."""
    for v in vals:
        v = _nan(v)
        if v is not None:
            return v
    return None


def _flag(v) -> int | None:
    v = _nan(v)
    return None if v is None else int(bool(v))


def _float(v) -> float | None:
    v = _nan(v)
    return None if v is None else float(v)


# ── identity ──────────────────────────────────────────────────────────────────

def player_id_hash(steamid: int) -> int:
    """sha1 of the decimal SteamID, first 8 bytes, truncated to 63 bits (P3.2.6)."""
    digest = hashlib.sha1(str(int(steamid)).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


# ── weapon vocabulary ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _vocab() -> tuple[dict[str, str], dict[str, str]]:
    """(name -> short name, short name -> class) from vocab_weapons.yaml."""
    doc = yaml.safe_load(VOCAB_PATH.read_text(encoding="utf-8"))
    short_class: dict[str, str] = {}
    for cls, names in doc["classes"].items():
        if cls not in WEAPON_CLASSES:
            raise ValueError(f"{VOCAB_PATH.name}: class {cls!r} not in {WEAPON_CLASSES}")
        for n in names:
            short_class[n.lower()] = cls
    names: dict[str, str] = {}
    for n in short_class:
        names[n] = n
        names[f"weapon_{n}"] = n
    for alias, short in (doc.get("aliases") or {}).items():
        names[str(alias).lower()] = str(short).lower()
    return names, short_class


_UNMAPPED_LOGGED: set[str] = set()


def short_name(name: str | None) -> str | None:
    """Canonical short name (`ak47`) of any emitted form, or None if unmapped/None."""
    if name is None:
        return None
    key = str(name).strip().lower()
    names, _ = _vocab()
    if key in names:
        return names[key]
    bare = key[7:] if key.startswith("weapon_") else key
    if bare.startswith(("knife", "bayonet")):
        return "knife"
    return None


def weapon_class(name: str | None) -> str | None:
    """Class of a weapon name; None for None; `unknown` (logged once) when unmapped."""
    if _nan(name) is None:
        return None
    short = short_name(name)
    if short is None:
        if name not in _UNMAPPED_LOGGED:
            _UNMAPPED_LOGGED.add(name)
            log.warning(f"unmapped weapon name {name!r} -> {UNKNOWN_CLASS}")
        return UNKNOWN_CLASS
    return _vocab()[1].get(short, UNKNOWN_CLASS)


def grenade_counts(inventory) -> dict[str, int] | None:
    """n_smoke .. n_decoy from an inventory list (None if the inventory is unknown)."""
    if inventory is None:
        return None
    out = {f: 0 for f in ("n_smoke", "n_flash", "n_he", "n_molly", "n_decoy")}
    for item in inventory:
        f = NADE_FIELDS.get(short_name(item) or "")
        if f:
            out[f] += 1
    return out


# ── tokens ────────────────────────────────────────────────────────────────────

def pad_token(side: str, tier: str) -> PlayerToken:
    """An empty slot (fewer than 5 players on a side): dead-shaped, id PAD_PLAYER_ID."""
    return PlayerToken(
        side=side, alive=0, x=None, y=None, z=None, nav_area=None, callout=None, grid_cell=None,
        yaw=None, pitch=None, speed=None, health=0.0, armor=0.0, has_helmet=None,
        has_defuser=None, has_c4=None, weapon_class=None, n_smoke=None, n_flash=None, n_he=None,
        n_molly=None, n_decoy=None, is_scoped=None, is_ducking=None, is_walking=None,
        is_reloading=None, is_defusing=None, is_planting=None, flash_duration_remaining=0.0,
        in_smoke=None, on_fire=None, money=None, equipment_value=None,
        seconds_since_last_seen_by_enemy=None, seconds_since_last_shot=None,
        spotted_by_count=None, heard_enemy=None, kills_round=0.0, damage_round=0.0,
        player_id=PAD_PLAYER_ID, tier=tier)


def flash_remaining(sl, steamid: int) -> float:
    """Blind seconds left at sl.tick from flashes that already popped; 0.0 if none."""
    fl = sl.flashes
    if fl.n == 0:
        return 0.0
    who = fl.col("player_steamid")
    mask = who == steamid
    if not mask.any():
        return 0.0
    det   = fl.col("detonate_tick")[mask].astype(np.float64)
    blind = fl.col("blind_duration")[mask].astype(np.float64)
    left  = blind - (sl.tick - det) / sl.tick_rate
    left  = left[~np.isnan(left)]
    return float(max(0.0, left.max())) if left.size else 0.0


def _since(sl, last_tick) -> float | None:
    last_tick = _nan(last_tick)
    if last_tick is None or last_tick < 0:
        return None
    return (sl.tick - int(last_tick)) / sl.tick_rate


def _position_at(table, tick: int) -> tuple[float, float, float] | None:
    """Position of a player-ticks table at its last row <= tick."""
    k = int(np.searchsorted(table.col("tick"), tick, side="right")) - 1
    if k < 0:
        return None
    p = tuple(_float(table.cols[c][k]) for c in ("x", "y", "z"))
    return None if any(v is None for v in p) else p


def heard_enemy(sl, steamid: int, pos, enemies, window_s: float) -> int | None:
    if pos is None:
        return None
    lo = sl.tick - window_s * sl.tick_rate
    for e in enemies:
        fire = sl.fire.get(e)
        if fire is not None and fire.n:
            ft = fire.col("tick")
            for t in ft[ft >= lo]:
                ep = sl.players.get(e)
                q = _position_at(ep, int(t)) if ep is not None else None
                if q is not None and math.dist(pos, q) <= HEARD_RADIUS:
                    return 1
        steps = sl.steps.get(e)
        if steps is not None and steps.n:
            st = steps.col("tick")
            m  = st >= lo
            if m.any():
                d = np.sqrt((steps.col("x")[m] - pos[0]) ** 2 + (steps.col("y")[m] - pos[1]) ** 2
                            + (steps.col("z")[m] - pos[2]) ** 2)
                if (d <= HEARD_RADIUS).any():
                    return 1
    return 0


def round_kills_damage(sl, steamid: int, enemies) -> tuple[float, float]:
    kills = 0.0
    if sl.kills.n:
        att = sl.kills.col("attacker_steamid")
        tk  = sl.kills.col("team_kill")
        kills = float(sum(1 for a, t in zip(att, tk) if _nan(a) == steamid and _nan(t) is not True))
    dmg = 0.0
    if sl.damages.n:
        att = sl.damages.col("attacker_steamid")
        vic = sl.damages.col("victim_steamid")
        hp  = sl.damages.col("dmg_health")
        dmg = float(sum(_first_valid(h, 0) for a, v, h in zip(att, vic, hp)
                        if _nan(a) == steamid and _nan(v) in enemies))
    return kills, dmg


def token(sl, steamid: int, side: str, geometry, tier: str, utility=(),
          heard_window_s: float = HEARD_WINDOW_S) -> PlayerToken:
    """The A.2 token of one roster player at sl.tick."""
    from engine.state.utility import inside_flag   # local: utility imports nothing from here

    enemies = {s for s, sd in sl.sides.items() if sd != side}
    kills, dmg = round_kills_damage(sl, steamid, enemies)
    pid = player_id_hash(steamid)
    pt  = sl.players.get(steamid)
    if pt is None or pt.n == 0:
        tok = pad_token(side, tier)
        tok.player_id, tok.kills_round, tok.damage_round = pid, kills, dmg
        return tok

    i     = pt.n - 1
    row   = lambda c: pt.cols[c][i] if c in pt.cols else None   # noqa: E731
    alive = bool(_nan(row("is_alive")))

    # position: dead players keep the last position they had while alive
    src = i
    if not alive:
        la = int(pt.cols["_last_alive"][i])
        src = la if la >= 0 else i
    x, y, z = (_float(pt.cols[c][src]) for c in ("x", "y", "z"))
    pos = None if None in (x, y, z) else (x, y, z)
    area = geometry.find_area(pos) if pos is not None else None

    fire  = sl.fire.get(steamid)
    shot  = int(fire.col("tick")[-1]) if fire is not None and fire.n else None
    spot_n = _nan(row("_spot_n"))

    if alive:
        vx, vy = _float(row("vx")), _float(row("vy"))
        counts = grenade_counts(row("inventory"))
        in_smoke = on_fire = None
        if pos is not None:
            in_smoke = inside_flag(utility, "smoke", *pos)
            on_fire  = inside_flag(utility, "fire", *pos)      # None while a fire of unknown extent burns
        live = dict(
            yaw=_float(row("yaw")), pitch=_float(row("pitch")),
            speed=None if vx is None or vy is None else math.hypot(vx, vy),
            health=_float(row("health")), armor=_float(row("armor")),
            has_helmet=_flag(row("has_helmet")), has_defuser=_flag(row("has_defuser")),
            has_c4=_flag(row("has_c4")), weapon_class=weapon_class(row("active_weapon")),
            **(counts or {f: None for f in ("n_smoke", "n_flash", "n_he", "n_molly", "n_decoy")}),
            flash_duration_remaining=flash_remaining(sl, steamid),
            in_smoke=in_smoke, on_fire=on_fire,
            money=_float(row("money")), equipment_value=_float(row("equipment_value")),
            spotted_by_count=None if spot_n is None else int(spot_n),
            heard_enemy=heard_enemy(sl, steamid, pos, enemies, heard_window_s))
    else:
        live = dict(
            yaw=None, pitch=None, speed=None, health=0.0, armor=0.0, has_helmet=None,
            has_defuser=None, has_c4=None, weapon_class=None, n_smoke=None, n_flash=None,
            n_he=None, n_molly=None, n_decoy=None, flash_duration_remaining=0.0,
            in_smoke=None, on_fire=None, money=None, equipment_value=None,
            spotted_by_count=None, heard_enemy=None)

    return PlayerToken(
        side=side, alive=int(alive), x=x, y=y, z=z, nav_area=area,
        callout=geometry.callout(x, y, z) if pos is not None else None,
        grid_cell=geometry.grid_cell(x, y) if pos is not None else None,
        yaw=live["yaw"], pitch=live["pitch"], speed=live["speed"],
        health=live["health"], armor=live["armor"],
        has_helmet=live["has_helmet"], has_defuser=live["has_defuser"], has_c4=live["has_c4"],
        weapon_class=live["weapon_class"],
        n_smoke=live["n_smoke"], n_flash=live["n_flash"], n_he=live["n_he"],
        n_molly=live["n_molly"], n_decoy=live["n_decoy"],
        **{c: _flag(row(c)) for c in FLAG_COLS},
        flash_duration_remaining=live["flash_duration_remaining"],
        in_smoke=live["in_smoke"], on_fire=live["on_fire"],
        money=live["money"], equipment_value=live["equipment_value"],
        seconds_since_last_seen_by_enemy=_since(sl, row("_last_spot")),
        seconds_since_last_shot=_since(sl, shot),
        spotted_by_count=live["spotted_by_count"], heard_enemy=live["heard_enemy"],
        kills_round=kills, damage_round=dmg, player_id=pid, tier=tier)
