"""
Movement actions (heuristic, config thresholds): peek (P4.2.5), rotation (P4.3.3),
hold (P4.3.4), save/retreat (P4.3.5).

    from engine.actions import movement
    acts = movement.extract_peeks(raw, snaps, ctx)    # ctx.set_snapshots(snaps) first

Only GRID snapshots between freeze_end and the round's end tick are read, so adding
boundary samples (for instance around the peeks found here) never changes the result,
and nothing is detected after the round is decided. Every actor is alive in every
sample its action is built from (a death ends the player's sequence).

- peek: an alive enemy pair whose effective visibility (`los` = 1 and `smoke_between` =
  0, so a smoke that blocks the view counts as no sight) goes 0 -> 1 between
  consecutive grid samples, where at least one of the two moved
  >= `movement_min_units` over the last PEEK_LOOKBACK_S (tick tracks). The actor is
  the mover (the larger displacement when both moved, lower id on a tie), the other is
  the target; tick = the sample where sight begins. Params: from_callout (actor, the
  sample before), target_id, target_callout, target_spotted_me_first (True / False from
  the spotted_by masks between the two samples, None when neither spotted the other).
- rotation: per player, consecutive samples with one callout form a stay; a stay is
  settled when its first-to-last sample span is >= ROTATION_DWELL_S (the first stay of
  the round always is). A settled stay in a different callout from the previous
  settled one, with nav distance between the two callouts' centroids
  >= `rotation_min_units`, is a rotation; every settled stay becomes the new anchor.
  Window = [last sample in the old callout, first sample in the new one]. Params:
  from_callout, to_callout, nav_dist_units, duration_s, toward_bomb (nav distance to the
  bomb lower at the end than at the start; None if the bomb position is unknown).
- hold: a run of samples with speed < `hold_max_speed` in one callout spanning
  >= `hold_min_s`. Window = the run. Params: callout, duration_s, facing_callout
  (callout FACING_UNITS along the view direction at the start, None if none).
- save: from plant_complete, else from the last LATE_ROUND_S of the round clock, a span
  over which the player's distance to the bomb (planted) or to the nearest alive enemy
  (not planted) grows by >= `save_min_units` within `save_window_s` with no shot fired.
  Overlapping spans merge; each merged span is trimmed to [the sample at the smallest
  distance, the first sample at the largest]. Distances are nav distances, falling back
  to straight-line when the geometry has none. Params: from_callout, to_callout,
  teammates_alive (alive teammates at the start).
"""
import math

import numpy as np

from engine.actions.types import Action, register

# ── constants ─────────────────────────────────────────────────────────────────

PEEK_LOOKBACK_S  = 0.5      # s, displacement window for a peek (P4.2.5)
ROTATION_DWELL_S = 2.0      # s, a new callout must be held this long (P4.3.3)
FACING_UNITS     = 500.0    # u, facing_callout probe distance (P4.3.4)
LATE_ROUND_S     = 20.0     # s, "last 20 s of the round" for saves (P4.3.5)
LOS, NAV, SMOKE  = 0, 2, 3  # pair channels used here


def _sid(ctx, pid):
    return ctx.sid_of_pid.get(pid)


def _alive_seq(seq, k: int) -> list:
    """Samples of slot k up to (not including) the first sample where it is dead."""
    out = []
    for s in seq:
        if s.players[k].alive != 1:
            break
        out.append(s)
    return out


def _slots(seq) -> list[int]:
    if not seq:
        return []
    return [k for k, p in enumerate(seq[0].players) if p.player_id is not None and p.player_id >= 0]


def _visible(s, i: int, j: int) -> int | None:
    los, sm = s.pair[i, j, LOS], s.pair[i, j, SMOKE]
    if np.isnan(los) or np.isnan(sm):
        return None
    return int(los == 1.0 and sm == 0.0)


def _displacement(ctx, r: int, s0, s1, k: int) -> float | None:
    """How far slot k moved over the PEEK_LOOKBACK_S before s1 (tick track), falling
    back to the two samples' positions."""
    pid = s1.players[k].player_id
    tr = ctx.track(r, _sid(ctx, pid))
    t1 = s1.meta.tick
    t0 = t1 - int(round(PEEK_LOOKBACK_S * ctx.tick_rate))
    if tr is not None:
        a, b = tr.pos(t0), tr.pos(t1)
        if a is not None and b is not None:
            return math.dist(a, b)
    p0, p1 = s0.players[k], s1.players[k]
    if None in (p0.x, p0.y, p0.z, p1.x, p1.y, p1.z):
        return None
    return math.dist((p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z))


# ── peek (P4.2.5) ─────────────────────────────────────────────────────────────

@register("peek")
def extract_peeks(raw, states, ctx) -> list[Action]:
    min_u = float(ctx.cfg.actions.movement_min_units)
    out: list[Action] = []
    for r in sorted(ctx.snaps):
        seq = ctx.grid(r)
        for s0, s1 in zip(seq, seq[1:]):
            ps0, ps1 = s0.players, s1.players
            for i in range(len(ps1)):
                for j in range(i + 1, len(ps1)):
                    if ps1[i].side == ps1[j].side:
                        continue
                    if not (ps0[i].alive == ps0[j].alive == ps1[i].alive == ps1[j].alive == 1):
                        continue
                    if _visible(s0, i, j) != 0 or _visible(s1, i, j) != 1:
                        continue
                    di, dj = _displacement(ctx, r, s0, s1, i), _displacement(ctx, r, s0, s1, j)
                    movers = [(d, ps1[k].player_id, k) for d, k in ((di, i), (dj, j))
                              if d is not None and d >= min_u]
                    if not movers:
                        continue
                    _, _, a = max(movers, key=lambda m: (m[0], -m[1]))
                    b = j if a == i else i
                    out.append(_peek(ctx, r, s0, s1, a, b))
    out = [a for a in out if a is not None]
    out.sort(key=lambda a: (a.round_num, a.tick, a.actor_id_hash, a.params["target_id"]))
    return ctx.unique(out, lambda a: a.params["target_id"])


def _peek(ctx, r: int, s0, s1, a: int, b: int) -> Action | None:
    me, tg = s1.players[a], s1.players[b]
    sa, sb = _sid(ctx, me.player_id), _sid(ctx, tg.player_id)
    first = None
    if sa is not None and sb is not None:
        lo, hi = s0.meta.tick + 1, s1.meta.tick
        t_me = ctx.first_spotted(r, sa, sb, lo, hi)      # target spotted me
        t_tg = ctx.first_spotted(r, sb, sa, lo, hi)      # I spotted the target
        if t_me is not None or t_tg is not None:
            first = t_me is not None and (t_tg is None or t_me < t_tg)
    params = {"from_callout": s0.players[a].callout, "target_id": tg.player_id,
              "target_callout": tg.callout, "target_spotted_me_first": first}
    return ctx.instant(r, "peek", sa, s1.meta.tick, params, side=me.side)


# ── rotation (P4.3.3) ─────────────────────────────────────────────────────────

def stays(samples: list[tuple[int, str | None]]) -> list[list]:
    """[(tick, callout)] -> [[callout, first_tick, last_tick, first_index, last_index]]."""
    out: list[list] = []
    for n, (t, c) in enumerate(samples):
        if out and out[-1][0] == c:
            out[-1][2], out[-1][4] = t, n
        else:
            out.append([c, t, t, n, n])
    return out


def rotation_spans(samples: list[tuple[int, str | None]], dwell_ticks: int, far) -> list[tuple]:
    """(from_stay, to_stay) pairs per the module rules; `far(a, b)` says whether two
    callouts are at least rotation_min_units apart."""
    st = stays(samples)
    if not st:
        return []
    anchor, out = st[0], []
    for s in st[1:]:
        if s[0] is None or s[2] - s[1] < dwell_ticks:
            continue
        if s[0] != anchor[0] and anchor[0] is not None and far(anchor[0], s[0]):
            out.append((anchor, s))
        anchor = s
    return out


@register("rotation")
def extract_rotations(raw, states, ctx) -> list[Action]:
    tr_ = ctx.tick_rate
    dwell = int(round(ROTATION_DWELL_S * tr_))
    min_u = float(ctx.cfg.actions.rotation_min_units)
    nav_cache: dict[tuple[str, str], float | None] = {}

    def nav(a: str, b: str) -> float | None:
        key = (min(a, b), max(a, b))
        if key not in nav_cache:
            ca, cb = ctx.callout_centroid(a), ctx.callout_centroid(b)
            nav_cache[key] = None if ca is None or cb is None else \
                ctx.nav_or_dist(ca[:3], cb[:3], ca[3], cb[3])
        return nav_cache[key]

    def far(a: str, b: str) -> bool:
        d = nav(a, b)
        return d is not None and d >= min_u

    out: list[Action] = []
    for r in sorted(ctx.snaps):
        seq = ctx.grid(r)
        for k in _slots(seq):
            alive = _alive_seq(seq, k)
            samples = [(s.meta.tick, s.players[k].callout) for s in alive]
            for frm, to in rotation_spans(samples, dwell, far):
                s_a, s_b = alive[frm[4]], alive[to[3]]
                pa, pb = s_a.players[k], s_b.players[k]
                ba, bb = ctx.bomb_pos(r, s_a.meta.tick), ctx.bomb_pos(r, s_b.meta.tick)
                da = ctx.nav_or_dist((pa.x, pa.y, pa.z), ba, pa.nav_area) if ba is not None else None
                db = ctx.nav_or_dist((pb.x, pb.y, pb.z), bb, pb.nav_area) if bb is not None else None
                params = {"from_callout": frm[0], "to_callout": to[0],
                          "nav_dist_units": round(nav(frm[0], to[0]), 3),
                          "duration_s": round((s_b.meta.tick - s_a.meta.tick) / tr_, 4),
                          "toward_bomb": None if da is None or db is None else bool(db < da)}
                a = ctx.durative(r, "rotation", _sid(ctx, pa.player_id), s_a.meta.tick, s_b.meta.tick,
                                 params, side=pa.side)
                if a is not None:
                    out.append(a)
    out.sort(key=lambda a: (a.round_num, a.window_start_tick, a.actor_id_hash))
    return ctx.unique(out, lambda a: a.window_end_tick)


# ── hold (P4.3.4) ─────────────────────────────────────────────────────────────

def hold_runs(samples: list[tuple[int, str | None, float | None]], max_speed: float,
              min_ticks: int) -> list[tuple[int, int]]:
    """[(tick, callout, speed)] -> [(first_index, last_index)] of stationary runs in one
    callout spanning >= min_ticks."""
    runs, cur = [], None
    for n, (t, c, v) in enumerate(samples):
        ok = v is not None and v < max_speed and c is not None
        if ok and cur is not None and samples[cur[0]][1] == c:
            cur[1] = n
            continue
        if cur is not None and samples[cur[1]][0] - samples[cur[0]][0] >= min_ticks:
            runs.append(tuple(cur))
        cur = [n, n] if ok else None
    if cur is not None and samples[cur[1]][0] - samples[cur[0]][0] >= min_ticks:
        runs.append(tuple(cur))
    return runs


@register("hold")
def extract_holds(raw, states, ctx) -> list[Action]:
    tr_ = ctx.tick_rate
    max_v = float(ctx.cfg.actions.hold_max_speed)
    min_t = int(round(float(ctx.cfg.actions.hold_min_s) * tr_))
    out: list[Action] = []
    for r in sorted(ctx.snaps):
        seq = ctx.grid(r)
        for k in _slots(seq):
            alive = _alive_seq(seq, k)
            samples = [(s.meta.tick, s.players[k].callout, s.players[k].speed) for s in alive]
            for i0, i1 in hold_runs(samples, max_v, min_t):
                p = alive[i0].players[k]
                facing = None
                if p.yaw is not None and None not in (p.x, p.y, p.z):
                    yaw = math.radians(p.yaw)
                    facing = ctx.callout((p.x + FACING_UNITS * math.cos(yaw),
                                          p.y + FACING_UNITS * math.sin(yaw), p.z))
                t0, t1 = alive[i0].meta.tick, alive[i1].meta.tick
                params = {"callout": p.callout, "duration_s": round((t1 - t0) / tr_, 4),
                          "facing_callout": facing}
                a = ctx.durative(r, "hold", _sid(ctx, p.player_id), t0, t1, params, side=p.side)
                if a is not None:
                    out.append(a)
    out.sort(key=lambda a: (a.round_num, a.window_start_tick, a.actor_id_hash))
    return ctx.unique(out, lambda a: a.window_end_tick)


# ── save / retreat (P4.3.5) ───────────────────────────────────────────────────

def retreat_spans(ticks: list[int], dist: list[float | None], refs: list, window_ticks: int,
                  min_units: float, fired) -> list[tuple[int, int]]:
    """Index spans (start, end) per the module rules. `refs[n]` tags what dist[n] is
    measured to (bomb / enemy); only samples with the same tag are compared.
    `fired(i, j)` -> True when a shot was fired in (ticks[i], ticks[j]]."""
    n, marked = len(ticks), []
    for i in range(n):
        if dist[i] is None:
            continue
        j = i
        while j + 1 < n and ticks[j + 1] <= ticks[i] + window_ticks:
            j += 1
        if j == i or dist[j] is None or refs[j] != refs[i]:
            continue
        if dist[j] - dist[i] >= min_units and not fired(i, j):
            marked.append((i, j))
    merged: list[list[int]] = []
    for i, j in marked:
        if merged and i <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], j)
        else:
            merged.append([i, j])
    out = []
    for lo, hi in merged:
        idx = [m for m in range(lo, hi + 1) if dist[m] is not None and refs[m] == refs[lo]]
        top = max(dist[m] for m in idx)
        end = next(m for m in idx if dist[m] == top)
        start = min((m for m in idx if m <= end), key=lambda m: (dist[m], -m))
        if start < end:
            out.append((start, end))
    return out


@register("save")
def extract_saves(raw, states, ctx) -> list[Action]:
    tr_ = ctx.tick_rate
    win = int(round(float(ctx.cfg.actions.save_window_s) * tr_))
    min_u = float(ctx.cfg.actions.save_min_units)
    late = int(round((float(ctx.cfg.clock.round_time_s) - LATE_ROUND_S) * tr_))
    out: list[Action] = []
    for r in sorted(ctx.snaps):
        seq = ctx.grid(r)
        if not seq:
            continue
        b = ctx.raw.bomb
        plants = sorted(int(t) for t, e, rn in zip(b["tick"].to_list(), b["event"].to_list(),
                                                    b["round_num"].to_list())
                        if rn == r and e == "plant_complete" and t is not None)
        plant = plants[0] if plants else None
        start = min(plant if plant is not None else 1 << 62, ctx.freeze_end(r) + late)
        phase = [s for s in seq if s.meta.tick >= start]
        bomb_at = {}
        for k in _slots(phase):
            alive = _alive_seq(phase, k)
            if len(alive) < 2:
                continue
            p_sid = _sid(ctx, alive[0].players[k].player_id)
            ticks, dist, refs = [], [], []
            for s in alive:
                p = s.players[k]
                t = s.meta.tick
                if plant is not None and t >= plant:
                    if t not in bomb_at:
                        bomb_at[t] = ctx.bomb_pos(r, t)
                    d = ctx.nav_or_dist((p.x, p.y, p.z), bomb_at[t], p.nav_area) if bomb_at[t] else None
                    ref = "bomb"
                else:
                    ds = []
                    for j, q in enumerate(s.players):
                        if q.side != p.side and q.alive == 1:
                            v = s.pair[k, j, NAV]
                            if np.isnan(v):
                                v = s.pair[k, j, 1]
                            if not np.isnan(v):
                                ds.append(float(v))
                    d, ref = (min(ds) if ds else None), "enemy"
                ticks.append(t)
                dist.append(d)
                refs.append(ref)

            def fired(i, j, _sid_=p_sid, _t=ticks):
                return _sid_ is not None and ctx.fired(_sid_, _t[i], _t[j])

            for i0, i1 in retreat_spans(ticks, dist, refs, win, min_u, fired):
                s0, s1 = alive[i0], alive[i1]
                p0, p1 = s0.players[k], s1.players[k]
                mates = sum(1 for j, q in enumerate(s0.players) if j != k and q.side == p0.side and q.alive == 1)
                params = {"from_callout": p0.callout, "to_callout": p1.callout, "teammates_alive": mates}
                a = ctx.durative(r, "save", p_sid, s0.meta.tick, s1.meta.tick, params, side=p0.side)
                if a is not None:
                    out.append(a)
    out.sort(key=lambda a: (a.round_num, a.window_start_tick, a.actor_id_hash))
    return ctx.unique(out, lambda a: a.window_end_tick)
