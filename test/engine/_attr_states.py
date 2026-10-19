"""
Hand-built rounds for the P8 attribution tests and bench, on top of `_cf_states`.

    from _attr_states import make_round, round_win, round_hazard, ROUND_TICKS
    snaps, actions, windows, ids = make_round()        # the story below
    stub = StubPredictor(win=round_win, hazard=round_hazard)

Story (128 tick, freeze_end 4600, terminal 5560, grid every 64 ticks, round label CT win):
- 4600-4920  CT 103 walks x 0 -> 600 (y 200) inside a `rotation` action [4600, 4920].
- 4664-4792  T 204 walks x 1500 -> 1200 (y 300); no movement action covers it, so its
             credit lands on a synthetic `positioning` row.
- 5000       one window, three members: 101 kills 202, 203 kills 102 (same tick), and
             105's `decoy_start` far from everyone (parent: a `decoy` durative action).
- 5200       104's `smoke_pop` (parent: a `smoke` durative action), a CT smoke.
- 5250       104's flash blinds T 204 for 2 s.
- 5300       101 kills the blinded 204; enablers: the flash, a trade_setup linking
             203's kill of 102 (beneficiary 102), and an unlinked `info`.

Boundary samples exist at every action tick +/- 1 and at the rotation's ends, as the
sampler would make them. Windows are dicts with the `Window` field names (P4.6.1:
`start_tick = tick - 1`, `end_tick = tick + 1`).

`round_win` (CT win probability): `_cf_states.good_win`'s terms unclipped (+0.1 per man,
blind side worse) + 0.05 while a CT smoke is up + 0.04 per 600 u that CT 103 advanced
− 0.06 per 300 u that T 204 advanced (while alive), clipped to [0.02, 0.98]. It ignores
decoys and the clock, so a decoy is worth exactly 0 and window clock gaps are 0.
`round_hazard`: 0.05 per alive enemy + 0.25 · min(blind, 3) / 3, 0 when dead.
"""
from _cf_states import (TICK_RATE, action, make_snapshot)

# ── constants ─────────────────────────────────────────────────────────────────

FREEZE_END = 4600
TERMINAL   = 5560
STEP       = 64
ACTION_TICKS = (5000, 5200, 5250, 5300)
EXTRA_TICKS  = (4601, 4919, 4921)            # rotation boundary samples (4599 clips to 4600)
DECOY_AT   = (-3000.0, -3000.0, 0.0)
SMOKE_AT   = (1450.0, 400.0, 0.0)


def round_ticks() -> list[int]:
    grid = list(range(FREEZE_END, TERMINAL + 1, STEP))
    bounds = [t + d for t in ACTION_TICKS for d in (-1, 1)]
    return sorted(set(grid) | set(bounds) | set(EXTRA_TICKS))


ROUND_TICKS = round_ticks()


def _lerp(t: int, t0: int, t1: int, a: float, b: float) -> float:
    if t <= t0:
        return a
    if t >= t1:
        return b
    return a + (b - a) * (t - t0) / (t1 - t0)


def x103(t: int) -> float:
    return _lerp(t, 4600, 4920, 0.0, 600.0)


def x204(t: int) -> float:
    return _lerp(t, 4664, 4792, 1500.0, 1200.0)


def _kinds(t: int) -> tuple:
    k = []
    if (t - FREEZE_END) % STEP == 0:
        k.append("grid")
    if any(t == a - 1 for a in ACTION_TICKS) or t == 4919:
        k.append("boundary_pre")
    if any(t == a + 1 for a in ACTION_TICKS) or t in (4601, 4921):
        k.append("boundary_post")
    if t == TERMINAL:
        k.append("terminal")
    return tuple(k)


def snapshot_at(t: int):
    dead, kills = set(), {}
    if t > 5000:
        dead |= {202, 102}
        kills = {101: 1, 203: 1}
    if t > 5300:
        dead.add(204)
        kills[101] = 2
    damage = {k: 100.0 * v for k, v in kills.items()}
    blind = {}
    if 5250 < t <= 5300:
        blind[204] = max(0.0, 2.0 - (t - 5250) / TICK_RATE)
    smokes = [{"x": SMOKE_AT[0], "y": SMOKE_AT[1], "z": 0.0, "owner": 104, "start_tick": 5200}] if t > 5200 else []
    decoys = [{"x": DECOY_AT[0], "y": DECOY_AT[1], "z": 0.0, "owner": 105, "start_tick": 5000}] if t > 5000 else []
    moving = {}
    if 4600 <= t < 4920:
        moving[103] = 250.0
    if 4664 <= t < 4792:
        moving[204] = 250.0
    return make_snapshot(tick=t, dead=dead, kills=kills, damage=damage, blind=blind, smokes=smokes,
                         decoys=decoys, positions={103: (x103(t), 200.0, 0.0), 204: (x204(t), 300.0, 0.0)},
                         speeds=moving, kinds=_kinds(t))


# ── actions and windows ───────────────────────────────────────────────────────

def make_actions() -> dict:
    """{name: Action} for the story."""
    rot = action("rotation", 103, tick=4600, kind="durative")
    rot.window_end_tick = 4920
    decoy = action("decoy", 105, tick=5000, kind="durative", x=DECOY_AT[0], y=DECOY_AT[1], z=0.0)
    smoke = action("smoke", 104, tick=5200, kind="durative", x=SMOKE_AT[0], y=SMOKE_AT[1], z=0.0)
    k1 = action("kill", 101, tick=5000, victim_id=202, victim_health_before=100.0)
    k2 = action("kill", 203, tick=5000, victim_id=102, victim_health_before=100.0)
    dstart = action("decoy_start", 105, tick=5000, parent_action_id=decoy.action_id)
    pop = action("smoke_pop", 104, tick=5200, parent_action_id=smoke.action_id)
    fl = action("flash", 104, tick=5250, blinded_enemies=[{"id": 204, "duration": 2.0}],
                blinded_teammates=[], blinded_self=False)
    k3 = action("kill", 101, tick=5300, victim_id=204, victim_health_before=100.0)
    k3.enablers = sorted([{"action_id": fl.action_id, "kind": "flash"},
                          {"action_id": k2.action_id, "kind": "trade_setup", "beneficiary_id": 102},
                          {"action_id": None, "kind": "info"}],
                         key=lambda e: (e["kind"], e["action_id"] or ""))
    return {"rotation": rot, "decoy": decoy, "smoke": smoke, "kill_101_202": k1, "kill_203_102": k2,
            "decoy_start": dstart, "smoke_pop": pop, "flash": fl, "kill_101_204": k3}


def window(wid: str, tick: int, members: list, round_num: int = 7, span: int = 1) -> dict:
    return {"window_id": wid, "round_num": round_num, "start_tick": tick - span,
            "end_tick": tick + span, "member_action_ids": [m.action_id for m in members]}


def make_round():
    """(snapshots, actions, windows, actions-by-name) for the story."""
    a = make_actions()
    snaps = [snapshot_at(t) for t in ROUND_TICKS]
    windows = [
        window("w5000", 5000, [a["kill_101_202"], a["kill_203_102"], a["decoy_start"]]),
        window("w5200", 5200, [a["smoke_pop"]]),
        window("w5250", 5250, [a["flash"]]),
        window("w5300", 5300, [a["kill_101_204"]]),
    ]
    return snaps, list(a.values()), windows, a


# ── predictor rules ───────────────────────────────────────────────────────────

def _blind(s, side: str) -> float:
    return sum(min(p.flash_duration_remaining or 0.0, 3.0) for p in s.players if p.side == side and p.alive)


def round_win(s, h=None) -> float:
    w = 0.5 + 0.1 * (s.ctx.alive_ct - s.ctx.alive_t) - 0.15 * s.ctx.bomb_planted
    w += 0.01 * (_blind(s, "t") - _blind(s, "ct"))
    if any(u.kind == "smoke" and u.owner_side == "ct" for u in s.utility):
        w += 0.05
    i, j = s.index_of(103), s.index_of(204)
    if s.players[i].alive:
        w += 0.04 * s.players[i].x / 600.0
    if s.players[j].alive:
        w -= 0.06 * (1500.0 - s.players[j].x) / 300.0
    return min(0.98, max(0.02, w))


def round_hazard(s, h=None) -> list[float]:
    out = []
    for p in s.players:
        if not p.alive:
            out.append(0.0)
            continue
        enemies = s.ctx.alive_t if p.side == "ct" else s.ctx.alive_ct
        out.append(0.05 * enemies + 0.25 * min(p.flash_duration_remaining or 0.0, 3.0) / 3.0)
    return out
