"""
Flash and HE detonations (P4.2.3, P4.2.4); smoke, molotov and decoy windows with their
pop/fade pseudo-actions (P4.3.1, P4.3.2, P4.3.6).

    from engine.actions import utility
    acts = utility.extract_smokes(raw, None, ctx)     # smoke + smoke_pop + smoke_fade
    utility.fill_blocks(acts, ctx)                    # blocks_pairs_at_pop, needs snapshots

Sources match the state builder's utility tokens, so every pop/fade is a tick where a
token appears/disappears: smokes from `smokes`, fires from `infernos`, decoys from
`grenades` (type decoy, detonate -> expire). The matching `grenades` row (same
entity_id, else same thrower with a detonation within MATCH_TOL_S of the start) gives
the throw tick / position and, for fires, molotov vs incendiary (`subtype`).

Timing:
- A null end tick takes the nominal duration (`engine.state.utility.NOMINAL_DURATION_S`:
  smoke 18 s, fire 7 s, decoy 15 s — the one definition). A null start with a known
  throw tick takes `throw + NOMINAL_POP_DELAY_S`.
- The window is clipped to the round's terminal tick; the fade pseudo-action is emitted
  only when the end lies before the terminal tick (a smoke still up when the round is
  over has no fade inside the round). The window start is never clipped.

Params (Appendix A.5; `x`, `y`, `z` land position, `entity_id` and `throw_tick` added
for the interventions / descriptors):
- flash: blinded_enemies / blinded_teammates [{id, duration}] sorted by id (from the
  `flashes` rows of that entity and detonation), blinded_self (seconds; 0.0 when the
  thrower was not blinded — a legitimate value), land_callout, throw_callout,
  pop_delay_s.
- he: enemies_damaged / teammates_damaged [{id, dmg, armor}] from `damages` rows with
  weapon hegrenade by the thrower within `simultaneity_ticks` of the detonation (the
  thrower's self-damage is left out), damage_total = health damage over both lists,
  land_callout.
- smoke: land_callout, throw_callout, throw_tick, pop_tick, expire_tick (the end used,
  before clipping), blocks_pairs_at_pop (alive enemy pairs whose `smoke_between` is 1
  at pop + 1 and was not 1 at pop - 1; None until `fill_blocks` sees snapshots).
- molotov: land_callout, enemies_damaged [{id, dmg}] (fire damage by the thrower to
  enemies during the window), enemies_displaced (ids of enemies inside the extent at
  the start who left it within DISPLACE_S; None when the radius is unknown),
  radius_units.
- decoy: land_callout.
- molotov / decoy also carry `end_tick` (the end used, before clipping), like smoke's
  expire_tick.
- pseudo: parent_action_id (+ x, y, z).
"""
import math

import numpy as np
import polars as pl

from engine.actions.types import Action, register
from engine.state import pairwise
from engine.state.players import _nan, short_name
from engine.state.utility import FIRE_Z_TOL, NOMINAL_DURATION_S, SMOKE_RADIUS

# ── constants ─────────────────────────────────────────────────────────────────

NOMINAL_POP_DELAY_S = 2.0      # s, start = throw + this when the start tick is missing
MATCH_TOL_S         = 1.0      # s, grenade row <-> smoke/inferno row match by thrower + time
DISPLACE_S          = 2.0      # s, "left the fire within 2 s" (P4.3.2)
FIRE_WEAPONS        = ("molotov", "incgrenade", "inferno")
SMOKE_BETWEEN       = 3        # pair channel index of smoke_between

NOMINAL = {"smoke": NOMINAL_DURATION_S["smoke"], "molotov": NOMINAL_DURATION_S["fire"],
           "decoy": NOMINAL_DURATION_S["decoy"]}
PSEUDO  = {"smoke": ("smoke_pop", "smoke_fade"), "molotov": ("fire_start", "fire_end"),
           "decoy": ("decoy_start", "decoy_end")}


# ── helpers ───────────────────────────────────────────────────────────────────

def _int(v) -> int | None:
    v = _nan(v)
    return None if v is None else int(v)


def _f(v, nd: int = 4) -> float | None:
    v = _nan(v)
    return None if v is None else round(float(v), nd)


def _xyz(row: dict):
    p = tuple(_nan(row.get(c)) for c in ("x", "y", "z"))
    return None if any(v is None for v in p) else tuple(float(v) for v in p)


def _land(pos) -> dict:
    return {"x": None, "y": None, "z": None} if pos is None else \
           {"x": round(pos[0], 3), "y": round(pos[1], 3), "z": round(pos[2], 3)}


def _grenades(raw, types: tuple[str, ...]) -> list[dict]:
    g = raw.grenades.filter(pl.col("type").is_in(list(types)))
    return g.sort(["throw_tick", "entity_id"], nulls_last=True, maintain_order=True).to_dicts()


def _match(row: dict, start, grenades: list[dict], used: set, tol: int) -> dict | None:
    """The grenade row of a smoke/inferno row: same entity_id first, else same thrower
    and a detonation within tol of the start. Each grenade row matches once."""
    eid = _nan(row.get("entity_id"))
    for i, g in enumerate(grenades):
        if i not in used and eid is not None and _nan(g["entity_id"]) == eid:
            det = _nan(g["detonate_tick"])
            if start is None or det is None or abs(det - start) <= tol:
                used.add(i)
                return g
    best, bi = None, None
    for i, g in enumerate(grenades):
        if i in used or _nan(g["thrower_steamid"]) != _nan(row.get("thrower_steamid")):
            continue
        det = _nan(g["detonate_tick"])
        if start is None or det is None or abs(det - start) > tol:
            continue
        if best is None or abs(det - start) < abs(_nan(best["detonate_tick"]) - start):
            best, bi = g, i
    if bi is not None:
        used.add(bi)
    return best


def _thrower_info(ctx, r: int, g: dict | None, thrower) -> tuple[int | None, str | None]:
    if g is None:
        return None, None
    tt = _int(g.get("throw_tick"))
    return tt, (ctx.callout(ctx.pos(r, thrower, tt)) if tt is not None else None)


def _victims(ctx, r: int, thrower, rows: list[tuple]) -> tuple[list, list]:
    """rows (victim_sid, value dict) -> (enemies, teammates) lists sorted by id; the
    thrower's own entry is dropped."""
    side = ctx.side(r, thrower)
    en, tm = [], []
    for sid, val in rows:
        if sid == thrower:
            continue
        entry = {"id": ctx.pid(sid), **val}
        (tm if side is not None and ctx.side(r, sid) == side else en).append(entry)
    return sorted(en, key=lambda e: e["id"]), sorted(tm, key=lambda e: e["id"])


# ── flash (P4.2.3) ────────────────────────────────────────────────────────────

@register("flash")
def extract_flashes(raw, states, ctx) -> list[Action]:
    sim = int(ctx.cfg.actions.simultaneity_ticks)
    fl  = raw.flashes.to_dicts()
    out: list[Action] = []
    for g in _grenades(raw, ("flash",)):
        det, thrower = _int(g["detonate_tick"]), _nan(g["thrower_steamid"])
        r = ctx.round_of(det)
        if det is None or thrower is None or r is None:
            continue
        rows = [f for f in fl if _nan(f["entity_id"]) == _nan(g["entity_id"])
                and _nan(f["detonate_tick"]) is not None and abs(f["detonate_tick"] - det) <= sim]
        if not rows:   # entity ids that disagree: fall back to the detonation tick when unambiguous
            same = [h for h in _grenades(raw, ("flash",)) if _int(h["detonate_tick"]) is not None
                    and abs(_int(h["detonate_tick"]) - det) <= sim]
            if len(same) == 1:
                rows = [f for f in fl if _nan(f["detonate_tick"]) is not None
                        and abs(f["detonate_tick"] - det) <= sim]
        dur: dict[int, float] = {}
        for f in rows:
            p, d = _nan(f["player_steamid"]), _nan(f["blind_duration"])
            if p is not None and d is not None:
                dur[int(p)] = max(dur.get(int(p), 0.0), float(d))
        en, tm = _victims(ctx, r, thrower, [(p, {"duration": _f(d)}) for p, d in sorted(dur.items())])
        tt, throw_callout = _thrower_info(ctx, r, g, thrower)
        land = _xyz(g)
        params = {
            "blinded_enemies":   en,
            "blinded_teammates": tm,
            "blinded_self":      _f(dur.get(int(thrower), 0.0)),
            "land_callout":      ctx.callout(land),
            "throw_callout":     throw_callout,
            "pop_delay_s":       None if tt is None else _f((det - tt) / ctx.tick_rate),
            "throw_tick":        tt,
            "entity_id":         _int(g["entity_id"]),
            **_land(land),
        }
        a = ctx.instant(r, "flash", thrower, det, params)
        if a is not None:
            out.append(a)
    return ctx.unique(out, lambda a: a.params["entity_id"])


# ── HE (P4.2.4) ───────────────────────────────────────────────────────────────

@register("he")
def extract_he(raw, states, ctx) -> list[Action]:
    sim = int(ctx.cfg.actions.simultaneity_ticks)
    dmg = raw.damages.to_dicts()
    out: list[Action] = []
    for g in _grenades(raw, ("he",)):
        det, thrower = _int(g["detonate_tick"]), _nan(g["thrower_steamid"])
        r = ctx.round_of(det)
        if det is None or thrower is None or r is None:
            continue
        hits: dict[int, list[int]] = {}
        for d in dmg:
            if (_nan(d["attacker_steamid"]) == thrower and short_name(_nan(d["weapon"])) == "hegrenade"
                    and _nan(d["tick"]) is not None and abs(d["tick"] - det) <= sim
                    and _nan(d["victim_steamid"]) is not None):
                h = hits.setdefault(int(d["victim_steamid"]), [0, 0])
                h[0] += int(_nan(d["dmg_health"]) if _nan(d["dmg_health"]) is not None else 0)
                h[1] += int(_nan(d["dmg_armor"]) if _nan(d["dmg_armor"]) is not None else 0)
        en, tm = _victims(ctx, r, thrower, [(v, {"dmg": h[0], "armor": h[1]}) for v, h in sorted(hits.items())])
        tt, _ = _thrower_info(ctx, r, g, thrower)
        land = _xyz(g)
        params = {
            "enemies_damaged":   en,
            "teammates_damaged": tm,
            "damage_total":      int(sum(e["dmg"] for e in en + tm)),
            "land_callout":      ctx.callout(land),
            "throw_tick":        tt,
            "entity_id":         _int(g["entity_id"]),
            **_land(land),
        }
        a = ctx.instant(r, "he", thrower, det, params)
        if a is not None:
            out.append(a)
    return ctx.unique(out, lambda a: a.params["entity_id"])


# ── durative utility (P4.3.1, P4.3.2, P4.3.6) ─────────────────────────────────

def _durative(ctx, type_: str, r: int, thrower, start: int, end_raw: int | None,
              params: dict, subtype: str | None = None) -> list[Action]:
    """The durative action plus its start/end pseudo-actions."""
    tr   = ctx.tick_rate
    end  = end_raw if end_raw is not None else start + int(round(NOMINAL[type_] * tr))
    term = ctx.terminal[r]
    a = ctx.durative(r, type_, thrower, start, max(start, min(end, term)), params, subtype=subtype)
    if a is None:
        return []
    a.params["expire_tick" if type_ == "smoke" else "end_tick"] = int(end)
    pop_t, fade_t = PSEUDO[type_]
    land = {k: params.get(k) for k in ("x", "y", "z")}
    out = [a]
    p = ctx.instant(r, pop_t, thrower, start, {"parent_action_id": a.action_id, **land}, side=a.actor_side)
    out.append(p)
    if end < term:
        out.append(ctx.instant(r, fade_t, thrower, end, {"parent_action_id": a.action_id, **land},
                               side=a.actor_side))
    for child in out[1:]:
        if child is not None:
            child._parent = a                     # resolved again after ctx.unique() (MED-14)
    return out


def _unique(ctx, out: list, salt_of) -> list[Action]:
    """`ctx.unique` plus the MED-14 fix: a parent salted by `unique()` has its pseudo-
    actions' `parent_action_id` rewritten to the final id."""
    acts = ctx.unique([a for a in out if a is not None], salt_of)
    for a in acts:
        parent = getattr(a, "_parent", None)
        if parent is not None:
            a.params["parent_action_id"] = parent.action_id
            del a._parent
    return acts


def _window_rows(rows: list[dict], grenades: list[dict], ctx, start_col: str) -> list[tuple]:
    """(round, thrower, start, end_raw, row, grenade) for every row with a known start."""
    tol, used, out = int(round(MATCH_TOL_S * ctx.tick_rate)), set(), []
    for row in rows:
        start = _int(row.get(start_col))
        g = _match(row, start, grenades, used, tol)
        thrower = _nan(row.get("thrower_steamid"))
        if thrower is None and g is not None:
            thrower = _nan(g["thrower_steamid"])
        if start is None and g is not None and _int(g.get("throw_tick")) is not None:
            start = int(g["throw_tick"]) + int(round(NOMINAL_POP_DELAY_S * ctx.tick_rate))
        r = ctx.round_of(start)
        if start is None or thrower is None or r is None:
            continue
        out.append((r, thrower, start, _int(row.get("end_tick")), row, g))
    return out


@register("smoke")
def extract_smokes(raw, states, ctx) -> list[Action]:
    rows = raw.smokes.sort(["start_tick", "entity_id"], nulls_last=True, maintain_order=True).to_dicts()
    out: list[Action] = []
    for r, thrower, start, end, row, g in _window_rows(rows, _grenades(raw, ("smoke",)), ctx, "start_tick"):
        land = _xyz(row) or (_xyz(g) if g is not None else None)
        tt, throw_callout = _thrower_info(ctx, r, g, thrower)
        params = {
            "land_callout":        ctx.callout(land),
            "throw_callout":       throw_callout,
            "throw_tick":          tt,
            "pop_tick":            int(start),
            "blocks_pairs_at_pop": None,
            "entity_id":           _int(row.get("entity_id")),
            **_land(land),
        }
        out += _durative(ctx, "smoke", r, thrower, start, end, params)
    return _unique(ctx, out, lambda a: a.params.get("entity_id"))


def _fire_damage(raw, ctx, r: int, thrower, start: int, end: int) -> list[dict]:
    d = raw.damages.filter((pl.col("round_num") == r) & (pl.col("attacker_steamid") == thrower)
                           & (pl.col("tick") >= start) & (pl.col("tick") <= end))
    tot: dict[int, int] = {}
    side = ctx.side(r, thrower)
    for v, w, h in zip(d["victim_steamid"].to_list(), d["weapon"].to_list(), d["dmg_health"].to_list()):
        if v is None or short_name(w) not in FIRE_WEAPONS or ctx.side(r, v) in (None, side):
            continue
        tot[int(v)] = tot.get(int(v), 0) + int(h if h is not None else 0)
    return sorted(({"id": ctx.pid(v), "dmg": n} for v, n in tot.items()), key=lambda e: e["id"])


def _inside(pos, centre, radius: float) -> bool:
    return (math.hypot(pos[0] - centre[0], pos[1] - centre[1]) <= radius
            and abs(pos[2] - centre[2]) <= FIRE_Z_TOL)


def _displaced(ctx, r: int, thrower, start: int, centre, radius) -> list[int] | None:
    if radius is None or centre is None:
        return None
    side, hi, out = ctx.side(r, thrower), start + int(round(DISPLACE_S * ctx.tick_rate)), []
    for sid in sorted(ctx.sides.get(r, {})):
        if ctx.side(r, sid) == side:
            continue
        tr = ctx.track(r, sid)
        p0 = tr.pos(start) if tr is not None and tr.alive(start) else None
        if p0 is None or not _inside(p0, centre, radius):
            continue
        sl = tr.span(start + 1, hi)
        xs, ys, zs = tr.col("x")[sl], tr.col("y")[sl], tr.col("z")[sl]
        alive = np.asarray(tr.col("is_alive")[sl], dtype=bool)
        out_now = (np.hypot(xs - centre[0], ys - centre[1]) > radius) | (np.abs(zs - centre[2]) > FIRE_Z_TOL)
        if bool((out_now & alive).any()):
            out.append(ctx.pid(sid))
    return sorted(out)


@register("molotov")
def extract_molotovs(raw, states, ctx) -> list[Action]:
    rows = raw.infernos.sort(["start_tick", "entity_id"], nulls_last=True, maintain_order=True).to_dicts()
    out: list[Action] = []
    for r, thrower, start, end, row, g in _window_rows(rows, _grenades(raw, ("molotov", "incendiary")),
                                                         ctx, "start_tick"):
        land = _xyz(row) or (_xyz(g) if g is not None else None)
        radius = _f(row.get("extent_radius"), 3)
        end_used = end if end is not None else start + int(round(NOMINAL["molotov"] * ctx.tick_rate))
        params = {
            "land_callout":      ctx.callout(land),
            "enemies_damaged":   _fire_damage(raw, ctx, r, thrower, start, end_used),
            "enemies_displaced": _displaced(ctx, r, thrower, start, land, radius),
            "radius_units":      radius,
            "throw_tick":        _int(g.get("throw_tick")) if g is not None else None,
            "entity_id":         _int(row.get("entity_id")),
            **_land(land),
        }
        sub = g["type"] if g is not None else None
        out += _durative(ctx, "molotov", r, thrower, start, end, params, subtype=sub)
    return _unique(ctx, out, lambda a: a.params.get("entity_id"))


@register("decoy")
def extract_decoys(raw, states, ctx) -> list[Action]:
    out: list[Action] = []
    for g in _grenades(raw, ("decoy",)):
        start, thrower = _int(g["detonate_tick"]), _nan(g["thrower_steamid"])
        r = ctx.round_of(start)
        if start is None or thrower is None or r is None:
            continue
        land = _xyz(g)
        params = {"land_callout": ctx.callout(land), "throw_tick": _int(g.get("throw_tick")),
                  "entity_id": _int(g["entity_id"]), **_land(land)}
        out += _durative(ctx, "decoy", r, thrower, start, _int(g.get("expire_tick")), params)
    return _unique(ctx, out, lambda a: a.params.get("entity_id"))


# ── blocks_pairs_at_pop (needs snapshots) ─────────────────────────────────────

def blocks_at_pop(ctx, r: int, pop: int) -> int | None:
    post = ctx.snap_at(r, pop + ctx.off, exact=True)
    if post is None:
        return None
    pre = ctx.snap_at(r, pop - ctx.off, exact=True)
    n, ps = 0, post.players
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if ps[i].side == ps[j].side or ps[i].alive != 1 or ps[j].alive != 1:
                continue
            if post.pair[i, j, SMOKE_BETWEEN] == 1.0 and (pre is None or pre.pair[i, j, SMOKE_BETWEEN] != 1.0):
                n += 1
    return n


def fill_blocks(actions: list[Action], ctx) -> list[Action]:
    """Set `blocks_pairs_at_pop` on smoke actions from the snapshots at pop -/+ 1."""
    for a in actions:
        if a.type == "smoke":
            a.params["blocks_pairs_at_pop"] = blocks_at_pop(ctx, a.round_num, a.window_start_tick)
    return actions


def smoke_blocks_segment(action: Action, a_pos, b_pos) -> bool:
    """Does this smoke action's sphere cut the segment between two eye points?"""
    c = tuple(action.params.get(k) for k in ("x", "y", "z"))
    if any(v is None for v in c) or a_pos is None or b_pos is None:
        return False
    return pairwise.segment_hits_sphere(a_pos, b_pos, c, SMOKE_RADIUS)
