"""
Enablers active at tick(e) - 1 for kill, damage, plant_complete and defuse_complete
(P4.5.1). Entries are `{action_id, kind, beneficiary_id}`, deduplicated, sorted by
(kind, action_id) with a null action_id first (P4.5.2). `beneficiary_id` is who gets the
enabler's share: the owner of the linked action (thrower, holder/peeker), the dead
teammate for `trade_setup`, None for team-level info.

    from engine.actions import enablers
    enablers.attach(actions, ctx)          # ctx.set_snapshots(snaps) first

`s` is the snapshot at tick - 1 (the builder's boundary_pre sample; else the latest
before it). `recent` = `actions.enabler_recent_s`.
- flash: the victim's `flash_duration_remaining` > 0 in `s`; linked to the most recent
  flash action at or before tick - 1 that lists the victim (enemies, teammates or the
  thrower's own blinded_self).
- smoke_cover: `smoke_between` = 1 in `s` between the actor and an alive enemy other
  than the victim; linked to every active smoke action (popped at or before tick - 1,
  not yet faded) whose sphere cuts that eye-to-eye segment.
- trade_setup: a teammate of the actor died in [tick - trade_window_s, tick) and the
  victim at tick - 1 is within `trade_max_units` of that death — measured to the nearer
  of the teammate's last alive position and the position of the teammate's killer at
  the death (so the classic trade, killing the player who just killed your teammate,
  qualifies even when the two deaths are far apart); linked to the kill action whose
  victim is the teammate; beneficiary = the dead teammate.
- he_damage: an HE action listing the victim, detonated in [tick - recent, tick - 1].
- fire_pressure: a molotov action active during [tick - recent, tick - 1] whose extent
  (radius_units, 2D + FIRE_Z_TOL) contained the victim's position at some tick there.
- info: a teammate of the actor (not the actor) spotted the victim (`spotted_by`) or
  heard them (a victim shot or footstep within HEARD_RADIUS of that teammate, both
  alive) in [tick - recent, tick - 1]; linked to that teammate's latest peek at the
  victim in the window, else the teammate's latest hold covering an info tick; with no
  such action one team-level entry {action_id: None, beneficiary_id: None}.
Victim-based kinds need a victim, so plants and defuses only get smoke_cover.
"""
import math

import numpy as np

from engine.actions import utility
from engine.actions.types import Action
from engine.state.pairwise import EYE_DUCK, EYE_STAND
from engine.state.players import HEARD_RADIUS
from engine.state.utility import FIRE_Z_TOL

# ── constants ─────────────────────────────────────────────────────────────────

ENABLED_TYPES = ("kill", "damage", "plant_complete", "defuse_complete")
KINDS         = ("flash", "smoke_cover", "trade_setup", "he_damage", "fire_pressure", "info")
SMOKE_BETWEEN = 3


def _entry(action_id, kind: str, beneficiary) -> dict:
    return {"action_id": action_id, "kind": kind, "beneficiary_id": beneficiary}


def _eye(p):
    if None in (p.x, p.y, p.z):
        return None
    return (p.x, p.y, p.z + (EYE_DUCK if p.is_ducking == 1 else EYE_STAND))


def sort_entries(entries: list[dict]) -> list[dict]:
    seen, out = set(), []
    for e in entries:
        key = (e["kind"], e["action_id"], e["beneficiary_id"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return sorted(out, key=lambda e: (e["kind"], e["action_id"] or "",
                                      -1 if e["beneficiary_id"] is None else e["beneficiary_id"]))


# ── index of linkable actions ─────────────────────────────────────────────────

class _Index:
    def __init__(self, actions: list[Action], raw=None):
        self.sounds: dict[int, tuple] = {}     # steamid -> (fire ticks, step ticks, step xyz)
        if raw is not None:
            fire: dict[int, list] = {}
            for t, sid in zip(raw.weapon_fire["tick"].to_list(), raw.weapon_fire["steamid"].to_list()):
                if t is not None and sid is not None:
                    fire.setdefault(int(sid), []).append(int(t))
            steps: dict[int, list] = {}
            fs = raw.footsteps
            for t, sid, x, y, z in zip(fs["tick"].to_list(), fs["steamid"].to_list(), fs["x"].to_list(),
                                       fs["y"].to_list(), fs["z"].to_list()):
                if t is not None and sid is not None and None not in (x, y, z):
                    steps.setdefault(int(sid), []).append((int(t), float(x), float(y), float(z)))
            for sid in set(fire) | set(steps):
                st = sorted(steps.get(sid, []))
                self.sounds[sid] = (np.array(sorted(fire.get(sid, [])), dtype=np.int64),
                                    np.array([q[0] for q in st], dtype=np.int64),
                                    np.array([q[1:] for q in st], dtype=np.float64).reshape(-1, 3))
        self.flash, self.he, self.smokes, self.fires = {}, {}, [], []
        self.kills_of, self.peeks, self.holds = {}, {}, {}
        for a in actions:
            r = a.round_num
            if a.type == "flash":
                ids = [e["id"] for e in a.params.get("blinded_enemies", []) + a.params.get("blinded_teammates", [])]
                if a.params.get("blinded_self"):
                    ids.append(a.actor_id_hash)
                for v in ids:
                    self.flash.setdefault((r, v), []).append(a)
            elif a.type == "he":
                for e in a.params.get("enemies_damaged", []) + a.params.get("teammates_damaged", []):
                    self.he.setdefault((r, e["id"]), []).append(a)
            elif a.type == "smoke":
                self.smokes.append(a)
            elif a.type == "molotov":
                self.fires.append(a)
            elif a.type in ("kill", "team_kill"):
                self.kills_of.setdefault((r, a.params.get("victim_id")), []).append(a)
            elif a.type == "peek":
                self.peeks.setdefault((r, a.actor_id_hash), []).append(a)
            elif a.type == "hold":
                self.holds.setdefault((r, a.actor_id_hash), []).append(a)


# ── kinds ─────────────────────────────────────────────────────────────────────

def _flash(ix, s, vk, e, r, vid) -> list[dict]:
    left = s.players[vk].flash_duration_remaining if vk is not None else None
    if left is None or not left > 0:
        return []
    cands = [a for a in ix.flash.get((r, vid), []) if a.tick <= e.tick - 1]
    if not cands:
        return []
    a = max(cands, key=lambda a: (a.tick, a.action_id))
    return [_entry(a.action_id, "flash", a.actor_id_hash)]


def _smoke_cover(ix, s, ak, vk, e, r) -> list[dict]:
    if ak is None:
        return []
    t = e.tick - 1
    active = [a for a in ix.smokes if a.round_num == r and a.window_start_tick <= t
              and a.params.get("expire_tick", a.window_end_tick) > t]
    out, me = [], s.players[ak]
    for j, q in enumerate(s.players):
        if j in (ak, vk) or q.side == me.side or q.alive != 1 or me.alive != 1:
            continue
        if s.pair[ak, j, SMOKE_BETWEEN] != 1.0:
            continue
        for a in active:
            if utility.smoke_blocks_segment(a, _eye(me), _eye(q)):
                out.append(_entry(a.action_id, "smoke_cover", a.actor_id_hash))
    return out


def _trade(ix, ctx, e, r, vid, cfg) -> list[dict]:
    tr = ctx.tick_rate
    lo = e.tick - int(round(float(cfg.actions.trade_window_s) * tr))
    max_u = float(cfg.actions.trade_max_units)
    v_sid = ctx.sid_of_pid.get(vid)
    vpos = ctx.pos(r, v_sid, e.tick - 1) if v_sid is not None else None
    if vpos is None:
        return []
    out = []
    for (rr, dead), ks in sorted(ix.kills_of.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
        if rr != r or dead is None or dead == e.actor_id_hash:
            continue
        d_sid = ctx.sid_of_pid.get(dead)
        if d_sid is None or ctx.side(r, d_sid) != e.actor_side:
            continue
        for k in ks:
            if not (lo <= k.tick < e.tick):
                continue
            spots = [ctx.pos(r, d_sid, k.tick - 1), ctx.pos(r, ctx.sid_of_pid.get(k.actor_id_hash), k.tick)]
            dists = [math.dist(vpos, p) for p in spots if p is not None]
            if dists and min(dists) <= max_u:
                out.append(_entry(k.action_id, "trade_setup", dead))
    return out


def _he(ix, e, r, vid, recent: int) -> list[dict]:
    return [_entry(a.action_id, "he_damage", a.actor_id_hash) for a in ix.he.get((r, vid), [])
            if e.tick - recent <= a.tick <= e.tick - 1]


def _fire(ix, ctx, e, r, vid, recent: int) -> list[dict]:
    v_sid = ctx.sid_of_pid.get(vid)
    tr = ctx.track(r, v_sid) if v_sid is not None else None
    if tr is None:
        return []
    lo, hi, out = e.tick - recent, e.tick - 1, []
    for a in ix.fires:
        rad = a.params.get("radius_units")
        c = tuple(a.params.get(k) for k in ("x", "y", "z"))
        if a.round_num != r or rad is None or None in c:
            continue
        a_lo, a_hi = max(lo, a.window_start_tick), min(hi, a.window_end_tick)
        if a_lo > a_hi:
            continue
        sl = tr.span(a_lo, a_hi)
        xs, ys, zs = tr.col("x")[sl], tr.col("y")[sl], tr.col("z")[sl]
        inside = (np.hypot(xs - c[0], ys - c[1]) <= rad) & (np.abs(zs - c[2]) <= FIRE_Z_TOL)
        if bool(inside.any()):
            out.append(_entry(a.action_id, "fire_pressure", a.actor_id_hash))
    return out


def _info(ix, ctx, e, r, vid, recent: int) -> list[dict]:
    v_sid = ctx.sid_of_pid.get(vid)
    vt = ctx.track(r, v_sid) if v_sid is not None else None
    if vt is None:
        return []
    lo, hi = e.tick - recent, e.tick - 1
    mates = sorted(s for s, sd in ctx.sides.get(r, {}).items()
                   if sd == e.actor_side and ctx.pid(s) != e.actor_id_hash)
    sounds = []      # (tick, victim position)
    fire_t, step_t, step_p = ix.sounds.get(v_sid, (np.zeros(0, np.int64),) * 2 + (np.zeros((0, 3)),))
    for t in fire_t[(fire_t >= lo) & (fire_t <= hi)].tolist():
        p = vt.pos(int(t))
        if p is not None:
            sounds.append((int(t), p))
    m = (step_t >= lo) & (step_t <= hi)
    for t, p in zip(step_t[m].tolist(), step_p[m].tolist()):
        sounds.append((int(t), tuple(p)))
    out, unlinked = [], False
    for m in mates:
        mt = ctx.track(r, m)
        if mt is None:
            continue
        ticks = []
        sl = vt.span(lo, hi)
        lists = vt.col("spotted_by")
        for k in range(sl.start, sl.stop):
            lst = lists[k]
            if lst is not None and any(x is not None and int(x) == m for x in lst):
                ticks.append(int(vt.tick[k]))
        for t, p in sounds:
            q = mt.pos(t)
            if mt.alive(t) and q is not None and math.dist(p, q) <= HEARD_RADIUS:
                ticks.append(t)
        if not ticks:
            continue
        mp = ctx.pid(m)
        peeks = [a for a in ix.peeks.get((r, mp), []) if a.params.get("target_id") == vid and lo <= a.tick <= hi]
        holds = [a for a in ix.holds.get((r, mp), [])
                 if any(a.window_start_tick <= t <= a.window_end_tick for t in ticks)]
        link = max(peeks, key=lambda a: (a.tick, a.action_id)) if peeks else \
            (max(holds, key=lambda a: (a.window_start_tick, a.action_id)) if holds else None)
        if link is not None:
            out.append(_entry(link.action_id, "info", mp))
        else:
            unlinked = True
    if unlinked:
        out.append(_entry(None, "info", None))
    return out


# ── attach ────────────────────────────────────────────────────────────────────

def attach(actions: list[Action], ctx) -> list[Action]:
    cfg = ctx.cfg
    recent = int(round(float(cfg.actions.enabler_recent_s) * ctx.tick_rate))
    ix = _Index(actions, ctx.raw)
    for e in actions:
        if e.type not in ENABLED_TYPES or e.tick is None:
            continue
        r = e.round_num
        s = ctx.snap_at(r, e.tick - 1)
        entries: list[dict] = []
        vid = e.params.get("victim_id")
        ak = s.index_of(e.actor_id_hash) if s is not None else None
        vk = s.index_of(vid) if s is not None and vid is not None else None
        if s is not None:
            entries += _smoke_cover(ix, s, ak, vk, e, r)
        if vid is not None:
            if s is not None:
                entries += _flash(ix, s, vk, e, r, vid)
            entries += _trade(ix, ctx, e, r, vid, cfg)
            entries += _he(ix, e, r, vid, recent)
            entries += _fire(ix, ctx, e, r, vid, recent)
            entries += _info(ix, ctx, e, r, vid, recent)
        e.enablers = sort_entries([x for x in entries if x["action_id"] != e.action_id])
    return actions
