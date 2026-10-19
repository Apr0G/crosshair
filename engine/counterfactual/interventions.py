"""
State-editing interventions (P7.1). Each is a pure function on a `Snapshot` (never on
tensors): it deep-copies the input, edits the copy, then calls `recompute_derived` on
it. The input is never mutated.

    from engine.counterfactual.interventions import undo_kill, INTERVENTION_FOR_TYPE
    s_cf = undo_kill(s_post, kill, s_pre)              # victim alive again
    undo, redo = INTERVENTION_FOR_TYPE[action.type]    # uniform: fn(state, action, ref, *, geometry=None)

Uniform signature: every entry of `INTERVENTION_FOR_TYPE` is
`fn(state, action, ref, *, geometry=None) -> Snapshot`, where `ref` is the snapshot that
holds the values to restore: `s_pre` for an undo, `s_post` for a redo. `apply.py` builds
coalition states from that table.

Action params read here (Appendix A.5 names; extra keys are marked "optional"):
- kill / team_kill: `victim_id` (required), `victim_health_before` (optional; else the
  victim's health in `s_pre` / the current state).
- damage: `victim_id`, `damage_total` (required), `armor_damage` (optional, 0).
- he: `enemies_damaged` / `teammates_damaged` = [{"id", "dmg", "armor" (optional)}].
- flash: `blinded_enemies` / `blinded_teammates` = [{"id", "duration"}], `blinded_self`
  (bool or seconds).
- plant_complete: `site`. defuse_*: `kit` (optional).
- smoke / molotov / decoy and the pseudo types: optional `x`, `y`, `z` (land position),
  only needed to pick the token when one owner has two of that kind active.
- drop: `receiver_id` (or `receiver_id_hash`).

Documented approximations:
- Information is not undone: `spotted_by_count`, `seconds_since_last_seen_by_enemy`,
  `heard_enemy` of players other than a revived one stay as they are (P7.1.7).
- A revived player takes every field from `s_pre` except `kills_round` and
  `damage_round`, which belong to that player's own actions.
- `team_kill` leaves the attacker's counters alone (whether the parser counts team
  kills/damage is unknown).
- Flash / smoke / plant clocks copied from a reference state are not shifted by the
  1-2 ticks between `s_pre` and `s_post`, except the round clock on `unplant`/`replant`.
- `meta` and `labels` are never edited: an edited state is not a factual state, and the
  caller must not pin it as terminal (a defuse_complete undo has no "defused" field to
  flip; the round-end is carried only by `meta.kinds`).
- Hazard masks: the Snapshot has none; the model derives them from `alive`, which
  `recompute_derived` keeps consistent.
- `los` is computed for alive *enemy* pairs only (teammate pairs NaN), matching P3.5
  "LOS computed only for alive enemy pairs". `dist_units` is 3D Euclidean.

Filled by: P7.1.1-P7.1.8.
"""
import copy as _copy
import math
from functools import lru_cache
from typing import Callable

import numpy as np

from engine.actions.types import Action
from engine.state.pairwise import eye_point
from engine.state.schema import (DEAD_NULL_FIELDS, DEAD_ZERO_FIELDS, MAX_UTILITY, N_PLAYERS,
                                 PAIR_CHANNELS, PlayerToken, Snapshot, UtilityToken)

# ── constants ─────────────────────────────────────────────────────────────────

LOS      = PAIR_CHANNELS.index("los")
DIST     = PAIR_CHANNELS.index("dist_units")
NAV      = PAIR_CHANNELS.index("nav_dist_units")
SMOKE    = PAIR_CHANNELS.index("smoke_between")
SAME     = PAIR_CHANNELS.index("same_team")

MOVE_FIELDS    = ("x", "y", "z", "nav_area", "callout", "grid_cell", "speed")
LOADOUT_FIELDS = ("armor", "has_helmet", "has_defuser", "weapon_class", "n_smoke", "n_flash",
                  "n_he", "n_molly", "n_decoy", "money", "equipment_value")
OWN_COUNTERS   = ("kills_round", "damage_round")   # kept on revive: owned by the player's actions
FIRE_Z_TOL     = 64.0     # u, mirrors engine.state.utility.FIRE_Z_TOL
TOKEN_POS_TOL  = 8.0      # u, land position vs token position when params carry one
MAX_HEALTH     = 100.0
LOS_ENEMIES_ONLY = True

UTILITY_KIND_OF = {
    "smoke": "smoke", "smoke_pop": "smoke", "smoke_fade": "smoke",
    "molotov": "fire", "fire_start": "fire", "fire_end": "fire",
    "decoy": "decoy", "decoy_start": "decoy", "decoy_end": "decoy",
}


class InterventionError(ValueError):
    """An intervention's precondition does not hold for this state / action."""


@lru_cache(maxsize=1)
def _clock() -> tuple[float, float]:
    """(round_time_s, bomb_timer_s) from config.clock, read once."""
    from engine import config
    c = config.load().clock
    return float(c.round_time_s), float(c.bomb_timer_s)


# ── small helpers ─────────────────────────────────────────────────────────────

def _bad(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _pos(p: PlayerToken) -> tuple[float, float, float] | None:
    if any(_bad(v) for v in (p.x, p.y, p.z)):
        return None
    return (float(p.x), float(p.y), float(p.z))


def _slot(state: Snapshot, player_id, what: str) -> int:
    if player_id is None:
        raise InterventionError(f"{what}: player id is missing")
    i = state.index_of(player_id)
    if i is None:
        raise InterventionError(f"{what}: player {player_id} is not in the snapshot")
    return i


def _require_type(action: Action, types: tuple[str, ...], what: str) -> None:
    if action.type not in types:
        raise InterventionError(f"{what}: action type {action.type!r} is not one of {types}")


def _param(action: Action, *keys, required: bool = True):
    for k in keys:
        v = action.params.get(k)
        if not _bad(v):
            return v
    if required:
        raise InterventionError(f"{action.type}: params need one of {keys}")
    return None


def _ref_player(ref: Snapshot | None, player_id) -> PlayerToken | None:
    if ref is None:
        return None
    j = ref.index_of(player_id)
    return None if j is None else ref.players[j]


def _same_pos(a: PlayerToken, b: PlayerToken) -> bool:
    pa, pb = _pos(a), _pos(b)
    return pa is not None and pa == pb


def _eye(p: PlayerToken) -> tuple[float, float, float] | None:
    """The builder's eye point (pairwise.eye_point); None for an unknown position."""
    return None if _pos(p) is None else eye_point(p)


def _same_eye(a: PlayerToken, b: PlayerToken) -> bool:
    return _same_pos(a, b) and a.is_ducking == b.is_ducking


def _in_token(u: UtilityToken, pos) -> bool:
    """Mirrors engine.state.utility.inside: smoke = 3D sphere, fire = 2D disc with
    FIRE_Z_TOL vertical slack, decoy / unknown radius = never."""
    if pos is None or _bad(u.radius_units) or any(_bad(v) for v in (u.x, u.y, u.z)):
        return False
    x, y, z = pos
    if u.kind == "smoke":
        return math.dist((x, y, z), (u.x, u.y, u.z)) <= u.radius_units
    if u.kind == "fire":
        return math.hypot(x - u.x, y - u.y) <= u.radius_units and abs(z - u.z) <= FIRE_Z_TOL
    return False


def segment_hits_sphere(a, b, centre, radius: float) -> bool:
    """Does the segment a-b pass within `radius` of `centre` (P3.5 smoke_between)?"""
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = bx - ax, by - ay, bz - az
    cx, cy, cz = centre
    seg2 = dx * dx + dy * dy + dz * dz
    t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy + (cz - az) * dz) / seg2))
    px, py, pz = ax + t * dx, ay + t * dy, az + t * dz
    return math.dist((px, py, pz), (cx, cy, cz)) <= radius


def _smoke_blocks(u: UtilityToken, a, b) -> bool:
    if u.kind != "smoke" or _bad(u.radius_units) or any(_bad(v) for v in (u.x, u.y, u.z)):
        return False
    return segment_hits_sphere(a, b, (u.x, u.y, u.z), u.radius_units)


def _token_key(u: UtilityToken) -> tuple:
    return (u.kind, u.owner_id, u.x, u.y, u.z)


def _match_token(tokens: list[UtilityToken], action: Action) -> int | None:
    """Index of the token this utility action owns: same kind and owner, and (when the
    params carry a land position) within TOKEN_POS_TOL of it. None if absent; raises
    when two tokens match and nothing tells them apart."""
    kind = UTILITY_KIND_OF.get(action.type)
    if kind is None:
        raise InterventionError(f"{action.type!r} is not a utility action")
    cands = [i for i, u in enumerate(tokens) if u.kind == kind and u.owner_id == action.actor_id_hash]
    land = [action.params.get(k) for k in ("x", "y", "z")]
    if all(not _bad(v) for v in land):
        cands = [i for i in cands
                 if math.dist(land, (tokens[i].x, tokens[i].y, tokens[i].z)) <= TOKEN_POS_TOL]
    if len(cands) > 1:
        raise InterventionError(f"{action.type}: {len(cands)} {kind} tokens of owner "
                                f"{action.actor_id_hash}; pass params x/y/z to pick one")
    return cands[0] if cands else None


# ── recompute_derived ─────────────────────────────────────────────────────────

def _los_value(state: Snapshot, i: int, j: int, ref: Snapshot | None, geometry) -> float:
    a, b = state.players[i], state.players[j]
    if a.alive != 1 or b.alive != 1:
        return np.nan
    if LOS_ENEMIES_ONLY and a.side == b.side:
        return np.nan
    ra, rb = _ref_player(ref, a.player_id), _ref_player(ref, b.player_id)
    if (ra is not None and rb is not None and ra.alive == 1 and rb.alive == 1
            and _same_eye(a, ra) and _same_eye(b, rb)):
        v = ref.pair[ref.index_of(a.player_id), ref.index_of(b.player_id), LOS]
        if not np.isnan(v):
            return float(v)
    ea, eb = _eye(a), _eye(b)
    if geometry is not None and ea is not None and eb is not None:
        vis = geometry.is_visible(ea, eb)        # eye to eye, as pairwise.table
        return np.nan if vis is None else float(bool(vis))
    return np.nan


def _nav_value(state: Snapshot, i: int, j: int, ref: Snapshot | None, geometry) -> float:
    a, b = state.players[i], state.players[j]
    ra, rb = _ref_player(ref, a.player_id), _ref_player(ref, b.player_id)
    if (ra is not None and rb is not None and a.nav_area is not None and b.nav_area is not None
            and ra.nav_area == a.nav_area and rb.nav_area == b.nav_area):
        return float(ref.pair[ref.index_of(a.player_id), ref.index_of(b.player_id), NAV])
    if geometry is not None and a.nav_area is not None and b.nav_area is not None:
        d = geometry.nav_distance(a.nav_area, b.nav_area)
        return np.nan if d is None else float(d)
    return np.nan


def _smoke_value(state: Snapshot, i: int, j: int) -> float:
    """As pairwise.table: eye-to-eye segment against every smoke sphere; NaN unless both
    players are alive with a known position."""
    a, b = state.players[i], state.players[j]
    if a.alive != 1 or b.alive != 1:
        return np.nan
    ea, eb = _eye(a), _eye(b)
    if ea is None or eb is None:
        return np.nan
    return float(any(_smoke_blocks(u, ea, eb) for u in state.utility))


def _set_pair(state: Snapshot, i: int, j: int, k: int, v: float) -> None:
    state.pair[i, j, k] = v
    state.pair[j, i, k] = v


def _recompute_rows(state: Snapshot, i: int, ref: Snapshot | None, geometry) -> None:
    """Every pair channel of player i's row/column (moved or revived player)."""
    for j in range(N_PLAYERS):
        if j == i:
            continue
        pa, pb = _pos(state.players[i]), _pos(state.players[j])
        _set_pair(state, i, j, DIST, np.nan if pa is None or pb is None else math.dist(pa, pb))
        _set_pair(state, i, j, LOS, _los_value(state, i, j, ref, geometry))
        _set_pair(state, i, j, NAV, _nav_value(state, i, j, ref, geometry))
        _set_pair(state, i, j, SMOKE, _smoke_value(state, i, j))
        _set_pair(state, i, j, SAME, float(state.players[i].side == state.players[j].side))


def _inside_flag(utility, kind: str, pos) -> int | None:
    """Mirrors engine.state.utility.inside_flag: 1 inside a known token, 0 when every
    token of the kind is known and misses, None when one has an unknown radius."""
    unknown = False
    for u in utility:
        if u.kind != kind:
            continue
        if _in_token(u, pos):
            return 1
        if _bad(u.radius_units):
            unknown = True
    return None if unknown else 0


def _set_flags(state: Snapshot, i: int) -> None:
    p = state.players[i]
    pos = _pos(p)
    if p.alive != 1 or pos is None:
        return
    p.in_smoke = _inside_flag(state.utility, "smoke", pos)
    p.on_fire  = _inside_flag(state.utility, "fire", pos)


def _sum_alive(players: list[PlayerToken], side: str, field: str) -> float | None:
    vals = [getattr(p, field) for p in players if p.side == side and p.alive == 1]
    if any(v is None for v in vals):
        return None
    return float(sum(vals))


def recompute_derived(state: Snapshot, *, players=(), tokens=(), ref: Snapshot | None = None,
                      geometry=None) -> Snapshot:
    """Fix every field that depends on what an intervention edited. Mutates `state` in
    place (interventions call it on their own copy) and returns it.

    Always: `alive_ct/t` and `equip_ct/t` (sums over alive players, None if any is
    unknown), `n_active_smokes/fires` (token counts; at the token cap only raised, never
    lowered below the stored count), `los` NaN for every pair with a dead player.
    Never `money_ct/t`: the builder sums every roster account (dead included, LIVE_SAFE
    "current sum of accounts"), dead tokens carry `money = None`, and undoing a kill
    does not change anyone's account; recomputing it alive-only made v(full) and
    v(full minus a null member) differ by the dead players' money and broke C1
    (2026-10-19 audit, HIGH-05).
    `players`: slots that moved or changed alive state; their pair rows are recomputed
    (`dist_units` from positions; `los`/`nav_dist_units` copied from `ref` when both
    players stand where they stood in `ref`, else from `geometry`, else NaN;
    `smoke_between` from the tokens) and their `in_smoke`/`on_fire` flags re-derived.
    `tokens`: utility tokens that were added or removed; when one is a smoke, every
    pair's `smoke_between` is recomputed (eye-to-eye, as the builder), and every player
    inside one (every player, for a token of unknown extent) gets its flags re-derived."""
    ps, c = state.players, state.ctx
    for i in players:
        _recompute_rows(state, i, ref, geometry)
        _set_flags(state, i)
    for u in tokens:
        for i in range(N_PLAYERS):
            # a token of unknown extent bears on every player's flag (None <-> 0 / 1)
            if _bad(u.radius_units) or _in_token(u, _pos(ps[i])):
                _set_flags(state, i)
    if any(u.kind == "smoke" for u in tokens):
        for i in range(N_PLAYERS):
            for j in range(i + 1, N_PLAYERS):
                _set_pair(state, i, j, SMOKE, _smoke_value(state, i, j))

    c.alive_ct = sum(1 for p in ps if p.side == "ct" and p.alive == 1)
    c.alive_t  = sum(1 for p in ps if p.side == "t" and p.alive == 1)
    c.equip_ct = _sum_alive(ps, "ct", "equipment_value")
    c.equip_t  = _sum_alive(ps, "t", "equipment_value")
    n_smokes = sum(u.kind == "smoke" for u in state.utility)
    n_fires  = sum(u.kind == "fire" for u in state.utility)
    if len(state.utility) < MAX_UTILITY:
        c.n_active_smokes, c.n_active_fires = n_smokes, n_fires
    else:
        c.n_active_smokes = max(c.n_active_smokes, n_smokes)
        c.n_active_fires  = max(c.n_active_fires, n_fires)

    for i in range(N_PLAYERS):
        if i < len(ps) and ps[i].alive != 1:
            state.pair[i, :, LOS] = np.nan
            state.pair[:, i, LOS] = np.nan
    return state


# ── P7.1.1 kill ───────────────────────────────────────────────────────────────

def _lethal(kill: Action, victim: PlayerToken) -> float:
    v = kill.params.get("victim_health_before")
    return float(victim.health if _bad(v) else v)


def undo_kill(state: Snapshot, kill: Action, state_pre: Snapshot, *, geometry=None) -> Snapshot:
    """Undo a kill: the victim is alive again with every field from `state_pre`
    (position, health, armor, equipment, flags) except its own `kills_round` /
    `damage_round`; the attacker's `kills_round` drops by one and its `damage_round` by
    the lethal damage (`kill` only, not `team_kill`); a revived bomb carrier restores
    `bomb_carrier_alive`; alive counts, sums and the victim's pair rows recomputed.
    Preconditions: type kill/team_kill with `victim_id`; victim dead in `state` and
    alive in `state_pre`; attacker's `kills_round` >= 1."""
    _require_type(kill, ("kill", "team_kill"), "undo_kill")
    if state_pre is None:
        raise InterventionError("undo_kill needs state_pre")
    vid = _param(kill, "victim_id")
    s = state.copy()
    vi = _slot(s, vid, "undo_kill victim")
    if s.players[vi].alive != 0:
        raise InterventionError(f"undo_kill: victim {vid} is alive in the state")
    pre_v = _ref_player(state_pre, vid)
    if pre_v is None or pre_v.alive != 1:
        raise InterventionError(f"undo_kill: victim {vid} is not alive in state_pre")
    restored = _copy.deepcopy(pre_v)
    for f in OWN_COUNTERS:
        setattr(restored, f, getattr(s.players[vi], f))
    s.players[vi] = restored
    if kill.type == "kill" and kill.actor_id_hash is not None and kill.actor_id_hash != vid:
        a = s.players[_slot(s, kill.actor_id_hash, "undo_kill attacker")]
        if a.kills_round < 1:
            raise InterventionError(f"undo_kill: attacker {kill.actor_id_hash} has kills_round {a.kills_round}")
        a.kills_round  = a.kills_round - 1
        a.damage_round = max(0.0, a.damage_round - _lethal(kill, pre_v))
    if pre_v.has_c4 == 1 and s.ctx.bomb_planted == 0:
        s.ctx.bomb_carrier_alive = state_pre.ctx.bomb_carrier_alive
    return recompute_derived(s, players=[vi], ref=state_pre, geometry=geometry)


def _dead_token(p: PlayerToken) -> PlayerToken:
    d = _copy.deepcopy(p)
    d.alive = 0
    for f in DEAD_NULL_FIELDS:
        setattr(d, f, None)
    for f in DEAD_ZERO_FIELDS:
        setattr(d, f, 0.0)
    d.flash_duration_remaining = 0.0
    return d


def redo_kill(state: Snapshot, kill: Action, state_post: Snapshot | None, *, geometry=None) -> Snapshot:
    """Inverse of `undo_kill`: the victim dies (token from `state_post` when given, else
    the dead form of the current token at the same position), the attacker's counters
    rise again. Preconditions: victim alive in `state` (and dead in `state_post`)."""
    _require_type(kill, ("kill", "team_kill"), "redo_kill")
    vid = _param(kill, "victim_id")
    s = state.copy()
    vi = _slot(s, vid, "redo_kill victim")
    cur = s.players[vi]
    if cur.alive != 1:
        raise InterventionError(f"redo_kill: victim {vid} is already dead in the state")
    post_v = _ref_player(state_post, vid)
    if state_post is not None and (post_v is None or post_v.alive != 0):
        raise InterventionError(f"redo_kill: victim {vid} is not dead in state_post")
    dead = _copy.deepcopy(post_v) if post_v is not None else _dead_token(cur)
    for f in OWN_COUNTERS:
        setattr(dead, f, getattr(cur, f))
    s.players[vi] = dead
    if kill.type == "kill" and kill.actor_id_hash is not None and kill.actor_id_hash != vid:
        a = s.players[_slot(s, kill.actor_id_hash, "redo_kill attacker")]
        a.kills_round  = a.kills_round + 1
        a.damage_round = a.damage_round + _lethal(kill, cur)
    if cur.has_c4 == 1 and s.ctx.bomb_planted == 0:
        s.ctx.bomb_carrier_alive = state_post.ctx.bomb_carrier_alive if state_post is not None else 0
    return recompute_derived(s, players=[vi], ref=state_post, geometry=geometry)


# ── P7.1.2 damage (and HE) ────────────────────────────────────────────────────

def _restore_hp(s: Snapshot, attacker_id, victim_id, dmg: float, armor: float,
                ref: Snapshot | None, what: str) -> None:
    """Give `victim_id` back up to `dmg` health / `armor` armor, capped at its `ref`
    values (100 without one). A dead victim's health is left alone (the kill owns it).
    The attacker's `damage_round` (owned by this damage) drops by `dmg`, floored at the
    attacker's `ref` value, so the roll-back also happens when a same-window kill undo
    already revived the victim at full `ref` health, or the victim is still dead, in any
    undo order (LOW-28). Without a usable `ref` it drops by the health restored."""
    if dmg < 0 or armor < 0:
        raise InterventionError(f"{what}: negative damage {dmg}/{armor}")
    v = s.players[_slot(s, victim_id, f"{what} victim")]
    got_h = 0.0
    if v.alive == 1:
        rv = _ref_player(ref, victim_id)
        cap_h = rv.health if rv is not None and rv.alive == 1 else MAX_HEALTH
        cap_a = rv.armor if rv is not None and rv.alive == 1 else MAX_HEALTH
        got_h = min(dmg, max(0.0, cap_h - v.health))
        got_a = min(armor, max(0.0, cap_a - v.armor))
        v.health = v.health + got_h
        v.armor  = v.armor + got_a
    if attacker_id is None or attacker_id == victim_id:
        return
    a  = s.players[_slot(s, attacker_id, f"{what} attacker")]
    ra = _ref_player(ref, attacker_id)
    if ra is not None and not _bad(ra.damage_round) and not _bad(a.damage_round):
        a.damage_round = max(min(a.damage_round, ra.damage_round), a.damage_round - dmg)
    elif got_h > 0:
        a.damage_round = max(0.0, a.damage_round - got_h)


def _take_hp(s: Snapshot, attacker_id, victim_id, dmg: float, armor: float,
             ref: Snapshot | None, what: str) -> None:
    """Inverse of `_restore_hp`: health/armor from `ref` when given, else lowered by the
    amounts (non-lethal: floored at 1 health); the attacker's `damage_round` rises by the
    health taken."""
    v = s.players[_slot(s, victim_id, f"{what} victim")]
    if v.alive != 1:
        raise InterventionError(f"{what}: victim {victim_id} is dead in the state")
    rv = _ref_player(ref, victim_id)
    if rv is not None and rv.alive == 1:
        new_h, new_a = min(v.health, rv.health), min(v.armor, rv.armor)
    else:
        new_h, new_a = max(1.0, v.health - dmg), max(0.0, v.armor - armor)
    taken = v.health - new_h
    v.health, v.armor = new_h, new_a
    if attacker_id is not None and attacker_id != victim_id and taken > 0:
        a = s.players[_slot(s, attacker_id, f"{what} attacker")]
        a.damage_round = a.damage_round + taken


def undo_damage(state: Snapshot, damage: Action, state_pre: Snapshot | None = None, *,
                geometry=None) -> Snapshot:
    """Undo non-lethal damage: the victim's health/armor rise by `damage_total` /
    `armor_damage`, capped at `state_pre` values (100 without it), and the attacker's
    `damage_round` drops by `damage_total`, floored at its `state_pre` value (by the
    health restored without `state_pre`; see `_restore_hp`). Idempotent once capped. A
    dead victim's health is untouched (the kill owns the victim). Preconditions: type
    damage with `victim_id` and a non-negative `damage_total`."""
    _require_type(damage, ("damage",), "undo_damage")
    vid = _param(damage, "victim_id")
    dmg = float(_param(damage, "damage_total"))
    arm = float(_param(damage, "armor_damage", required=False) or 0.0)
    s = state.copy()
    _restore_hp(s, damage.actor_id_hash, vid, dmg, arm, state_pre, "undo_damage")
    return recompute_derived(s, ref=state_pre, geometry=geometry)


def redo_damage(state: Snapshot, damage: Action, state_post: Snapshot | None = None, *,
                geometry=None) -> Snapshot:
    """Inverse of `undo_damage`: health/armor from `state_post` (else lowered by the
    amounts, never below 1). Precondition: victim alive in `state`."""
    _require_type(damage, ("damage",), "redo_damage")
    vid = _param(damage, "victim_id")
    dmg = float(_param(damage, "damage_total"))
    arm = float(_param(damage, "armor_damage", required=False) or 0.0)
    s = state.copy()
    _take_hp(s, damage.actor_id_hash, vid, dmg, arm, state_post, "redo_damage")
    return recompute_derived(s, ref=state_post, geometry=geometry)


def _he_hits(he: Action) -> list[tuple[int, float, float]]:
    out = []
    for key in ("enemies_damaged", "teammates_damaged"):
        for e in he.params.get(key) or []:
            if not isinstance(e, dict) or _bad(e.get("id")) or _bad(e.get("dmg")):
                raise InterventionError(f"he: {key} entry {e!r} needs id and dmg")
            out.append((e["id"], float(e["dmg"]), float(e.get("armor") or 0.0)))
    return out


def undo_he(state: Snapshot, he: Action, state_pre: Snapshot | None = None, *, geometry=None) -> Snapshot:
    """Undo an HE detonation's damage: like `undo_damage` for every damaged player
    (capped at `state_pre`, so undoing the HE and its damage actions together restores
    each victim once). Preconditions: type he with well-formed damaged lists."""
    _require_type(he, ("he",), "undo_he")
    s = state.copy()
    for vid, dmg, arm in _he_hits(he):
        _restore_hp(s, he.actor_id_hash, vid, dmg, arm, state_pre, "undo_he")
    return recompute_derived(s, ref=state_pre, geometry=geometry)


def redo_he(state: Snapshot, he: Action, state_post: Snapshot | None = None, *, geometry=None) -> Snapshot:
    """Inverse of `undo_he`. Preconditions: damaged players alive in `state`."""
    _require_type(he, ("he",), "redo_he")
    s = state.copy()
    for vid, dmg, arm in _he_hits(he):
        _take_hp(s, he.actor_id_hash, vid, dmg, arm, state_post, "redo_he")
    return recompute_derived(s, ref=state_post, geometry=geometry)


# ── P7.1.3 utility tokens ─────────────────────────────────────────────────────

def remove_utility(state: Snapshot, utility_action: Action, state_ref: Snapshot | None = None, *,
                   geometry=None) -> Snapshot:
    """Drop the smoke / fire / decoy token the action owns (also the pop / fade / start /
    end pseudo-actions); recompute `smoke_between`, `in_smoke`, `on_fire`, active counts.
    `state_ref` is unused (uniform signature). Preconditions: a utility type; exactly one
    matching token in `state`."""
    s = state.copy()
    k = _match_token(s.utility, utility_action)
    if k is None:
        raise InterventionError(f"remove_utility: no {UTILITY_KIND_OF[utility_action.type]} token "
                                f"of owner {utility_action.actor_id_hash} in the state")
    gone = s.utility.pop(k)
    if len(state.utility) >= MAX_UTILITY:      # capped list: the stored count included this token
        if gone.kind == "smoke":
            s.ctx.n_active_smokes = max(0, s.ctx.n_active_smokes - 1)
        elif gone.kind == "fire":
            s.ctx.n_active_fires = max(0, s.ctx.n_active_fires - 1)
    return recompute_derived(s, tokens=[gone], geometry=geometry)


def add_utility(state: Snapshot, utility_action: Action, state_ref: Snapshot, *,
                geometry=None) -> Snapshot:
    """Inverse of `remove_utility`: copy the action's token from `state_ref` into the
    state, at the position it has in `state_ref`'s oldest-first order (at the cap the
    oldest token is dropped, as the builder does). Preconditions: a utility type; the
    token exists in `state_ref` and not in `state`."""
    if state_ref is None:
        raise InterventionError("add_utility needs state_ref (the state holding the token)")
    k = _match_token(state_ref.utility, utility_action)
    if k is None:
        raise InterventionError(f"add_utility: no matching token in state_ref for {utility_action.type}")
    s = state.copy()
    if _match_token(s.utility, utility_action) is not None:
        raise InterventionError(f"add_utility: the {utility_action.type} token is already in the state")
    tok = _copy.deepcopy(state_ref.utility[k])
    before = {_token_key(u) for u in state_ref.utility[:k]}
    pos = 0
    for idx, u in enumerate(s.utility):
        if _token_key(u) in before:
            pos = idx + 1
    s.utility.insert(pos, tok)
    if len(s.utility) > MAX_UTILITY:
        s.utility.pop(0)
        if tok.kind == "smoke":
            s.ctx.n_active_smokes += 1
        elif tok.kind == "fire":
            s.ctx.n_active_fires += 1
    return recompute_derived(s, tokens=[tok], geometry=geometry)


# ── P7.1.4 flash ──────────────────────────────────────────────────────────────

def blinded_players(flash: Action) -> list[tuple[int, float | None]]:
    out = []
    for key in ("blinded_enemies", "blinded_teammates"):
        for e in flash.params.get(key) or []:
            if not isinstance(e, dict) or _bad(e.get("id")):
                raise InterventionError(f"flash: {key} entry {e!r} needs an id")
            out.append((e["id"], None if _bad(e.get("duration")) else float(e["duration"])))
    me = flash.params.get("blinded_self")
    if me and flash.actor_id_hash is not None:
        out.append((flash.actor_id_hash, float(me) if not isinstance(me, bool) else None))
    return out


def unflash(state: Snapshot, flash_action: Action, state_pre: Snapshot | None = None, *,
            geometry=None) -> Snapshot:
    """Undo a flash: every alive player it blinded gets `flash_duration_remaining` from
    `state_pre` (0 without it, as P8.5 `remove_i` uses it on `s_pre`). Idempotent; a
    flash that blinded nobody is a no-op. Preconditions: type flash; blinded ids are in
    the snapshot."""
    _require_type(flash_action, ("flash",), "unflash")
    s = state.copy()
    for pid, _dur in blinded_players(flash_action):
        p = s.players[_slot(s, pid, "unflash")]
        if p.alive != 1:
            continue
        rp = _ref_player(state_pre, pid)
        p.flash_duration_remaining = (float(rp.flash_duration_remaining or 0.0)
                                      if rp is not None and rp.alive == 1 else 0.0)
    return recompute_derived(s, geometry=geometry)


def reflash(state: Snapshot, flash_action: Action, state_post: Snapshot | None = None, *,
            geometry=None) -> Snapshot:
    """Inverse of `unflash`: blinded alive players get their duration from `state_post`
    (else the params duration). Preconditions: type flash; a duration is available for
    every blinded alive player."""
    _require_type(flash_action, ("flash",), "reflash")
    s = state.copy()
    for pid, dur in blinded_players(flash_action):
        p = s.players[_slot(s, pid, "reflash")]
        if p.alive != 1:
            continue
        rp = _ref_player(state_post, pid)
        if rp is not None and rp.alive == 1:
            p.flash_duration_remaining = float(rp.flash_duration_remaining or 0.0)
        elif dur is not None:
            p.flash_duration_remaining = dur
        else:
            raise InterventionError(f"reflash: no duration for player {pid}")
    return recompute_derived(s, geometry=geometry)


# ── P7.1.5 plant ──────────────────────────────────────────────────────────────

def _set_prop(p: PlayerToken, field: str, ref_p: PlayerToken | None, default: int) -> None:
    """Set a per-tick 0/1 prop from the reference player, else `default`; a prop the
    parser lacks (None) stays None, and dead players are not touched."""
    if p.alive != 1 or getattr(p, field) is None:
        return
    v = getattr(ref_p, field) if ref_p is not None and ref_p.alive == 1 else None
    setattr(p, field, default if v is None else v)


def unplant_start(state: Snapshot, plant_start: Action, state_pre: Snapshot | None = None, *,
                  geometry=None) -> Snapshot:
    """Undo a plant start: the planter's `is_planting` from `state_pre` (else 0).
    Preconditions: type plant_start; bomb not planted."""
    _require_type(plant_start, ("plant_start",), "unplant_start")
    s = state.copy()
    if s.ctx.bomb_planted == 1:
        raise InterventionError("unplant_start: bomb already planted (undo plant_complete first)")
    pid = plant_start.actor_id_hash
    _set_prop(s.players[_slot(s, pid, "unplant_start")], "is_planting", _ref_player(state_pre, pid), 0)
    return recompute_derived(s, geometry=geometry)


def replant_start(state: Snapshot, plant_start: Action, state_post: Snapshot | None = None, *,
                  geometry=None) -> Snapshot:
    """Inverse of `unplant_start`: `is_planting` from `state_post` (else 1)."""
    _require_type(plant_start, ("plant_start",), "replant_start")
    s = state.copy()
    pid = plant_start.actor_id_hash
    _set_prop(s.players[_slot(s, pid, "replant_start")], "is_planting", _ref_player(state_post, pid), 1)
    return recompute_derived(s, geometry=geometry)


def unplant(state: Snapshot, plant_action: Action, state_pre: Snapshot | None = None, *,
            geometry=None) -> Snapshot:
    """Undo a plant: `bomb_planted = 0`, `bomb_site`/`time_since_plant_s` None, the clock
    switches back to the round timer (`state_pre`'s clock advanced by the elapsed
    `time_into_round_s`, else `clock.round_time_s - time_into_round_s`), the planter gets
    the bomb back where it stands (`has_c4 = 1`, `is_planting` from `state_pre`, else 0)
    and `bomb_carrier_alive` comes from `state_pre` (else the planter's alive bit).
    Preconditions: type plant_complete; bomb planted; no defuse in progress."""
    _require_type(plant_action, ("plant_complete",), "unplant")
    s = state.copy()
    c = s.ctx
    if c.bomb_planted != 1:
        raise InterventionError("unplant: bomb is not planted in the state")
    if c.defuse_in_progress == 1:
        raise InterventionError("unplant: a defuse is in progress (undo it first)")
    pid = plant_action.actor_id_hash
    planter = s.players[_slot(s, pid, "unplant planter")]
    c.bomb_planted, c.bomb_site, c.time_since_plant_s = 0, None, None
    if state_pre is not None and state_pre.ctx.bomb_planted == 0:
        elapsed = c.time_into_round_s - state_pre.ctx.time_into_round_s
        c.time_remaining_s = max(0.0, state_pre.ctx.time_remaining_s - elapsed)
        c.bomb_carrier_alive = state_pre.ctx.bomb_carrier_alive
    else:
        c.time_remaining_s = max(0.0, _clock()[0] - c.time_into_round_s)
        c.bomb_carrier_alive = int(planter.alive == 1)
    if planter.alive == 1:
        planter.has_c4 = 1
    _set_prop(planter, "is_planting", _ref_player(state_pre, pid), 0)
    return recompute_derived(s, geometry=geometry)


def replant(state: Snapshot, plant_action: Action, state_post: Snapshot | None = None, *,
            geometry=None) -> Snapshot:
    """Inverse of `unplant`: bomb planted at `params.site` (or `state_post`'s site), the
    bomb clock from `state_post` advanced to this state's time (else
    `time_since_plant_s = 0`, `time_remaining_s = clock.bomb_timer_s`),
    `bomb_carrier_alive` None, the planter's `has_c4 = 0` and `is_planting` from
    `state_post` (else 0). Preconditions: type plant_complete; bomb not planted; a site."""
    _require_type(plant_action, ("plant_complete",), "replant")
    s = state.copy()
    c = s.ctx
    if c.bomb_planted == 1:
        raise InterventionError("replant: bomb is already planted in the state")
    post_planted = state_post is not None and state_post.ctx.bomb_planted == 1
    site = _param(plant_action, "site", required=False) or (state_post.ctx.bomb_site if post_planted else None)
    if site is None:
        raise InterventionError("replant: no site in params or state_post")
    c.bomb_planted, c.bomb_site, c.bomb_carrier_alive = 1, site, None
    if post_planted:
        elapsed = c.time_into_round_s - state_post.ctx.time_into_round_s
        c.time_since_plant_s = max(0.0, state_post.ctx.time_since_plant_s + elapsed)
        c.time_remaining_s   = max(0.0, state_post.ctx.time_remaining_s - elapsed)
    else:
        c.time_since_plant_s, c.time_remaining_s = 0.0, _clock()[1]
    pid = plant_action.actor_id_hash
    planter = s.players[_slot(s, pid, "replant planter")]
    if planter.alive == 1:
        planter.has_c4 = 0
    _set_prop(planter, "is_planting", _ref_player(state_post, pid), 0)
    return recompute_derived(s, geometry=geometry)


# ── P7.1.6 defuse ─────────────────────────────────────────────────────────────

def _set_defuse(state: Snapshot, action: Action, ref: Snapshot | None, *, now: int, to: int,
                what: str, geometry) -> Snapshot:
    """Shared body of the defuse edits: requires a planted bomb and
    `defuse_in_progress == now`, then sets it to `to`; `defuse_with_kit` follows (from
    `ref` when its in-progress bit matches, else params `kit`, else the defuser's
    `has_defuser`; None when no defuse) and the defuser's `is_defusing` comes from
    `ref` (else `to`)."""
    s = state.copy()
    c = s.ctx
    if c.bomb_planted != 1:
        raise InterventionError(f"{what}: bomb is not planted")
    if c.defuse_in_progress != now:
        raise InterventionError(f"{what}: defuse_in_progress is {c.defuse_in_progress}, expected {now}")
    pid = action.actor_id_hash
    p = s.players[_slot(s, pid, what)]
    if to == 1 and p.alive != 1:
        raise InterventionError(f"{what}: defuser {pid} is dead")
    c.defuse_in_progress = to
    if to == 0:
        c.defuse_with_kit = None
    elif ref is not None and ref.ctx.defuse_in_progress == 1:
        c.defuse_with_kit = ref.ctx.defuse_with_kit
    else:
        kit = action.params.get("kit")
        c.defuse_with_kit = int(bool(kit)) if not _bad(kit) else p.has_defuser
    _set_prop(p, "is_defusing", _ref_player(ref, pid), to)
    return recompute_derived(s, geometry=geometry)


def undefuse_start(state: Snapshot, defuse_action: Action, state_pre: Snapshot | None = None, *,
                   geometry=None) -> Snapshot:
    """Undo a defuse start: `defuse_in_progress = 0`, `defuse_with_kit = None`, the
    defuser's `is_defusing` from `state_pre` (else 0). Preconditions: type defuse_start;
    bomb planted; a defuse in progress."""
    _require_type(defuse_action, ("defuse_start",), "undefuse_start")
    return _set_defuse(state, defuse_action, state_pre, now=1, to=0, what="undefuse_start", geometry=geometry)


def redefuse_start(state: Snapshot, defuse_action: Action, state_post: Snapshot | None = None, *,
                   geometry=None) -> Snapshot:
    """Inverse of `undefuse_start`. Preconditions: bomb planted; no defuse in progress;
    defuser alive."""
    _require_type(defuse_action, ("defuse_start",), "redefuse_start")
    return _set_defuse(state, defuse_action, state_post, now=0, to=1, what="redefuse_start", geometry=geometry)


def undefuse_complete(state: Snapshot, defuse_action: Action, state_pre: Snapshot | None = None, *,
                      geometry=None) -> Snapshot:
    """Undo a defuse completion (or an abort, `undefuse_abort`): the defuse is back in
    progress with `defuse_with_kit` / `is_defusing` from `state_pre`. The Snapshot has
    no "defused" field; the round end lives only in `meta.kinds`, which is not edited.
    Preconditions: bomb planted; no defuse in progress in `state`; defuser alive."""
    _require_type(defuse_action, ("defuse_complete", "defuse_abort"), "undefuse_complete")
    return _set_defuse(state, defuse_action, state_pre, now=0, to=1, what="undefuse_complete", geometry=geometry)


def redefuse_complete(state: Snapshot, defuse_action: Action, state_post: Snapshot | None = None, *,
                      geometry=None) -> Snapshot:
    """Inverse of `undefuse_complete`: the defuse ends (`defuse_in_progress = 0`).
    Preconditions: bomb planted; a defuse in progress."""
    _require_type(defuse_action, ("defuse_complete", "defuse_abort"), "redefuse_complete")
    return _set_defuse(state, defuse_action, state_post, now=1, to=0, what="redefuse_complete", geometry=geometry)


undefuse_abort  = undefuse_complete
redefuse_abort  = redefuse_complete


# ── bomb carry and weapon drops ───────────────────────────────────────────────

def _restore_c4(state: Snapshot, action: Action, ref: Snapshot | None, *, now: int, default: int,
                what: str, geometry) -> Snapshot:
    s = state.copy()
    if s.ctx.bomb_planted == 1:
        raise InterventionError(f"{what}: bomb is planted")
    pid = action.actor_id_hash
    p = s.players[_slot(s, pid, what)]
    if p.alive != 1:
        raise InterventionError(f"{what}: player {pid} is dead")
    if p.has_c4 != now:
        raise InterventionError(f"{what}: has_c4 is {p.has_c4}, expected {now}")
    rp = _ref_player(ref, pid)
    p.has_c4 = rp.has_c4 if rp is not None and rp.alive == 1 and rp.has_c4 is not None else default
    if ref is not None and ref.ctx.bomb_planted == 0:
        s.ctx.bomb_carrier_alive = ref.ctx.bomb_carrier_alive
    return recompute_derived(s, geometry=geometry)


def undo_bomb_drop(state, action, state_pre=None, *, geometry=None) -> Snapshot:
    """Undo a bomb drop: the dropper carries the bomb again (`has_c4` and
    `bomb_carrier_alive` from `state_pre`, else 1). Preconditions: bomb not planted;
    dropper alive without the bomb."""
    _require_type(action, ("bomb_drop",), "undo_bomb_drop")
    return _restore_c4(state, action, state_pre, now=0, default=1, what="undo_bomb_drop", geometry=geometry)


def redo_bomb_drop(state, action, state_post=None, *, geometry=None) -> Snapshot:
    """Inverse of `undo_bomb_drop`."""
    _require_type(action, ("bomb_drop",), "redo_bomb_drop")
    return _restore_c4(state, action, state_post, now=1, default=0, what="redo_bomb_drop", geometry=geometry)


def undo_bomb_pickup(state, action, state_pre=None, *, geometry=None) -> Snapshot:
    """Undo a bomb pickup: the picker no longer carries it (`has_c4` and
    `bomb_carrier_alive` from `state_pre`, else 0). Preconditions: bomb not planted;
    picker alive with the bomb."""
    _require_type(action, ("bomb_pickup",), "undo_bomb_pickup")
    return _restore_c4(state, action, state_pre, now=1, default=0, what="undo_bomb_pickup", geometry=geometry)


def redo_bomb_pickup(state, action, state_post=None, *, geometry=None) -> Snapshot:
    """Inverse of `undo_bomb_pickup`."""
    _require_type(action, ("bomb_pickup",), "redo_bomb_pickup")
    return _restore_c4(state, action, state_post, now=0, default=1, what="redo_bomb_pickup", geometry=geometry)


def _restore_loadouts(state: Snapshot, ids: list, ref: Snapshot, what: str, geometry) -> Snapshot:
    if ref is None:
        raise InterventionError(f"{what} needs a reference state")
    s = state.copy()
    for pid in ids:
        p = s.players[_slot(s, pid, what)]
        rp = _ref_player(ref, pid)
        if p.alive != 1 or rp is None or rp.alive != 1:
            continue
        for f in LOADOUT_FIELDS:
            setattr(p, f, getattr(rp, f))
    return recompute_derived(s, geometry=geometry)


def undo_drop(state, action, state_pre, *, geometry=None) -> Snapshot:
    """Undo a weapon drop/give: dropper's and receiver's loadouts from `state_pre`.
    Preconditions: type drop with `receiver_id`; both in the snapshot."""
    _require_type(action, ("drop",), "undo_drop")
    rid = _param(action, "receiver_id", "receiver_id_hash")
    return _restore_loadouts(state, [action.actor_id_hash, rid], state_pre, "undo_drop", geometry)


def redo_drop(state, action, state_post, *, geometry=None) -> Snapshot:
    """Inverse of `undo_drop`: both loadouts from `state_post`."""
    _require_type(action, ("drop",), "redo_drop")
    rid = _param(action, "receiver_id", "receiver_id_hash")
    return _restore_loadouts(state, [action.actor_id_hash, rid], state_post, "redo_drop", geometry)


# ── P7.1.7 movement ───────────────────────────────────────────────────────────

def freeze_position(state: Snapshot, player_id: int, state_ref: Snapshot, *, geometry=None) -> Snapshot:
    """Undo a player's movement: position, nav area, callout, grid cell and speed from
    `state_ref` (the interval's start); pair rows/cols and smoke/fire flags for that
    player recomputed. `spotted` fields are left as they are (information is not undone,
    only movement). Already at the reference position -> unchanged copy (idempotent).
    Preconditions: player alive in `state` and with a known position in `state_ref`."""
    if state_ref is None:
        raise InterventionError("freeze_position needs state_ref")
    s = state.copy()
    i = _slot(s, player_id, "freeze_position")
    p = s.players[i]
    if p.alive != 1:
        raise InterventionError(f"freeze_position: player {player_id} is dead")
    rp = _ref_player(state_ref, player_id)
    if rp is None or _pos(rp) is None:
        raise InterventionError(f"freeze_position: player {player_id} has no position in state_ref")
    if all(getattr(p, f) == getattr(rp, f) for f in MOVE_FIELDS):
        return s
    for f in MOVE_FIELDS:
        setattr(p, f, getattr(rp, f))
    return recompute_derived(s, players=[i], ref=state_ref, geometry=geometry)


# ── P7.1.8 buy ────────────────────────────────────────────────────────────────

def loadout_of(state: Snapshot, player_id: int) -> dict:
    """The player's LOADOUT_FIELDS as a dict (the form `revert_buy` takes)."""
    p = state.players[_slot(state, player_id, "loadout_of")]
    return {f: getattr(p, f) for f in LOADOUT_FIELDS}


def revert_buy(state: Snapshot, player_id: int, prior_loadout: dict, *, geometry=None) -> Snapshot:
    """Undo a player's buy: the loadout fields (armor, helmet, kit, weapon class,
    grenades, money, equipment value) set to `prior_loadout`, the economy prior of
    Appendix B.6 (median loadout for money bucket x side x round type, supplied by the
    caller); team equip recomputed, team money (the all-roster sum) shifted by the
    change to this one account. Preconditions: player alive; keys a subset of
    LOADOUT_FIELDS including `equipment_value`."""
    unknown = set(prior_loadout) - set(LOADOUT_FIELDS)
    if unknown:
        raise InterventionError(f"revert_buy: unknown loadout keys {sorted(unknown)}")
    if _bad(prior_loadout.get("equipment_value")):
        raise InterventionError("revert_buy: prior_loadout needs equipment_value")
    s = state.copy()
    p = s.players[_slot(s, player_id, "revert_buy")]
    if p.alive != 1:
        raise InterventionError(f"revert_buy: player {player_id} is dead")
    old_money = p.money
    for f, v in prior_loadout.items():
        setattr(p, f, v)
    if "money" in prior_loadout:
        attr = f"money_{p.side}"
        team = getattr(s.ctx, attr)
        if _bad(team) or _bad(old_money) or _bad(p.money):
            setattr(s.ctx, attr, None)                     # unknown stays unknown, never 0
        else:
            setattr(s.ctx, attr, float(team) - float(old_money) + float(p.money))
    return recompute_derived(s, geometry=geometry)


# ── uniform adapters for the table ────────────────────────────────────────────

def undo_move(state, action, state_ref, *, geometry=None) -> Snapshot:
    """`freeze_position` of the actor to `state_ref` (movement types and `peek`)."""
    return freeze_position(state, action.actor_id_hash, state_ref, geometry=geometry)


redo_move = undo_move       # same edit, the reference is s_post


def undo_buy(state, action, state_prior, *, geometry=None) -> Snapshot:
    """`revert_buy` of the actor to its loadout in `state_prior` (the B.6 prior state,
    which P4.4.1 stores as the buy's `state_pre`)."""
    return revert_buy(state, action.actor_id_hash, loadout_of(state_prior, action.actor_id_hash),
                      geometry=geometry)


redo_buy = undo_buy         # the reference is the real freeze-end state


def undo_team_buy(state, action, state_prior, *, geometry=None) -> Snapshot:
    """`revert_buy` for every alive player of the action's side."""
    s = state
    for p in state.players:
        if p.side == action.actor_side and p.alive == 1:
            s = revert_buy(s, p.player_id, loadout_of(state_prior, p.player_id), geometry=geometry)
    return s if s is not state else state.copy()


redo_team_buy = undo_team_buy


def _redo_utility(state, action, state_post, *, geometry=None) -> Snapshot:
    return add_utility(state, action, state_post, geometry=geometry)


def _undo_fade(state, action, state_pre, *, geometry=None) -> Snapshot:
    return add_utility(state, action, state_pre, geometry=geometry)


Intervention = Callable[..., Snapshot]

INTERVENTION_FOR_TYPE: dict[str, tuple[Intervention, Intervention]] = {
    "kill":            (undo_kill, redo_kill),
    "team_kill":       (undo_kill, redo_kill),
    "damage":          (undo_damage, redo_damage),
    "flash":           (unflash, reflash),
    "he":              (undo_he, redo_he),
    "peek":            (undo_move, redo_move),
    "plant_start":     (unplant_start, replant_start),
    "plant_complete":  (unplant, replant),
    "defuse_start":    (undefuse_start, redefuse_start),
    "defuse_complete": (undefuse_complete, redefuse_complete),
    "defuse_abort":    (undefuse_abort, redefuse_abort),
    "bomb_drop":       (undo_bomb_drop, redo_bomb_drop),
    "bomb_pickup":     (undo_bomb_pickup, redo_bomb_pickup),
    "drop":            (undo_drop, redo_drop),
    "smoke":           (remove_utility, _redo_utility),
    "molotov":         (remove_utility, _redo_utility),
    "decoy":           (remove_utility, _redo_utility),
    "rotation":        (undo_move, redo_move),
    "hold":            (undo_move, redo_move),
    "save":            (undo_move, redo_move),
    "buy":             (undo_buy, redo_buy),
    "team_buy":        (undo_team_buy, redo_team_buy),
    "smoke_pop":       (remove_utility, _redo_utility),
    "smoke_fade":      (_undo_fade, remove_utility),
    "fire_start":      (remove_utility, _redo_utility),
    "fire_end":        (_undo_fade, remove_utility),
    "decoy_start":     (remove_utility, _redo_utility),
    "decoy_end":       (_undo_fade, remove_utility),
}
