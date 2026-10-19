"""
Hand-built, internally consistent snapshots for the P7 counterfactual tests.

Built straight from the schema dataclasses (no builder, no `_fake_states.py`), with
their own pair-table / flag code, so the interventions' `recompute_derived` is checked
against an independent implementation:

    from _cf_states import make_snapshot, make_kill_pair, RuleStub, good_win
    s = make_snapshot(4, 5, smokes=[(750, 200, 0, 103)])     # 4v5, one CT smoke
    s_pre, s_post, kill = make_kill_pair()                   # 101 (CT) kills 202 (T) at 5000

Layout: CT ids 101-105 stand at x = 0, T ids 201-205 at x = 1500, both at y = 100 * k
(k = 0..4), z = 0; `seed` jitters x/y by up to 40 u. Clock: tick 5000 is 30 s into the
round at 128 tick. A pair is (pre at tick - 1, post at tick + 1) around an action at 5000.
Dead players keep their position; dead fields follow schema.DEAD_NULL/ZERO_FIELDS.

`RuleStub` is a small Predictor double (`predict(snapshots, histories) -> dict(win,
hazard, plant15, defuse15)`), so these tests do not depend on `_stubs.py` changing.
"""
import math
import random

import numpy as np

from engine.actions.types import Action
from engine.state.schema import (N_PLAYERS, PAIR_CHANNELS, ContextToken, PlayerToken, Snapshot,
                                 SnapshotMeta, UtilityToken)

# ── constants ─────────────────────────────────────────────────────────────────

CT_IDS      = (101, 102, 103, 104, 105)
T_IDS       = (201, 202, 203, 204, 205)
TICK_RATE   = 128
BASE_TICK   = 5000
BASE_INTO_S = 30.0
ROUND_S     = 115.0
BOMB_S      = 40.0
MATCH       = "cf_m1"
ROUND       = 7
SMOKE_R     = 144.0
FIRE_R      = 120.0
CELL        = 512.0
NOMINAL     = {"smoke": 18.0, "fire": 7.0, "decoy": 15.0}


def home(pid: int) -> tuple[float, float, float]:
    k = (pid % 100) - 1
    return (0.0 if pid < 200 else 1500.0, 100.0 * k, 0.0)


def side_of(pid: int) -> str:
    return "ct" if pid < 200 else "t"


# ── open-ground geometry (Geometry protocol, test-local) ──────────────────────

def _cell(x: float, y: float) -> int:
    return (math.floor(x / CELL) + 64) * 128 + (math.floor(y / CELL) + 64)


def _centre(c: int) -> tuple[float, float]:
    ix, iy = divmod(c, 128)
    return ((ix - 64 + 0.5) * CELL, (iy - 64 + 0.5) * CELL)


class OpenGeometry:
    """No walls: everything visible; nav areas are 512 u cells."""
    map_version = "de_test@cf"

    def is_visible(self, a, b):
        return True

    def find_area(self, p):
        return _cell(p[0], p[1])

    def nav_distance(self, a, b):
        return math.dist(_centre(a), _centre(b))

    def callout(self, x, y, z):
        return f"cell_{_cell(x, y)}"

    def grid_cell(self, x, y):
        return _cell(x, y)


class BlindGeometry(OpenGeometry):
    """Nothing visible: lets a test tell geometry-recomputed LOS from copied LOS."""

    def is_visible(self, a, b):
        return False


# ── independent derived-field code (the tests compare interventions against it) ─

def _eye(p) -> tuple[float, float, float]:
    return (p.x, p.y, p.z + (46.0 if p.is_ducking == 1 else 64.0))


def _seg_sphere(a, b, c, r) -> bool:
    ab = np.subtract(b, a)
    t = 0.0 if not ab.any() else float(np.clip(np.dot(np.subtract(c, a), ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(np.add(a, t * ab) - np.asarray(c))) <= r


def _inside(u: UtilityToken, p) -> bool:
    if u.radius_units is None:
        return False
    if u.kind == "smoke":
        return math.dist(p, (u.x, u.y, u.z)) <= u.radius_units
    if u.kind == "fire":
        return math.hypot(p[0] - u.x, p[1] - u.y) <= u.radius_units and abs(p[2] - u.z) <= 64.0
    return False


def _inside_flag(utility, kind: str, p) -> int | None:
    """As engine.state.utility.inside_flag: 1 / 0 / None (a token of unknown radius)."""
    unknown = False
    for u in utility:
        if u.kind != kind:
            continue
        if _inside(u, p):
            return 1
        if u.radius_units is None:
            unknown = True
    return None if unknown else 0


def build_pair(players: list[PlayerToken], utility: list[UtilityToken]) -> np.ndarray:
    pr = np.full((N_PLAYERS, N_PLAYERS, len(PAIR_CHANNELS)), np.nan, dtype=np.float32)
    for i in range(N_PLAYERS):
        for j in range(N_PLAYERS):
            if i == j:
                continue
            a, b = players[i], players[j]
            pa, pb = (a.x, a.y, a.z), (b.x, b.y, b.z)
            ea, eb = _eye(a), _eye(b)                  # LOS and smoke are cast from eye points
            pr[i, j, 0] = 1.0 if a.side != b.side and a.alive and b.alive else np.nan
            pr[i, j, 1] = math.dist(pa, pb)
            pr[i, j, 2] = math.dist(_centre(a.nav_area), _centre(b.nav_area))
            pr[i, j, 3] = (float(any(u.kind == "smoke" and _seg_sphere(ea, eb, (u.x, u.y, u.z), u.radius_units)
                                     for u in utility)) if a.alive and b.alive else np.nan)
            pr[i, j, 4] = float(a.side == b.side)
    return pr


# ── snapshots ─────────────────────────────────────────────────────────────────

def _util(kind: str, spec, tick: int) -> tuple[int, UtilityToken]:
    if isinstance(spec, dict):
        x, y, z, owner = spec["x"], spec["y"], spec.get("z", 0.0), spec["owner"]
        start, radius = spec.get("start_tick", 4500), spec.get("radius")
    else:
        x, y, z, owner = spec
        start, radius = 4500, None
    if radius == "unknown":                       # a fire with no extent (awpy's real-demo shape)
        radius = None
    elif radius is None:
        radius = {"smoke": SMOKE_R, "fire": FIRE_R}.get(kind)
    tok = UtilityToken(kind=kind, x=float(x), y=float(y), z=float(z),
                       time_left_s=max(0.0, NOMINAL[kind] - (tick - start) / TICK_RATE),
                       owner_side=side_of(owner), owner_id=owner,
                       radius_units=None if radius is None else float(radius))
    return start, tok


def make_snapshot(alive_ct: int = 5, alive_t: int = 5, *, dead=(), tick: int = BASE_TICK,
                  smokes=(), fires=(), decoys=(), blind=None, planted: bool = False,
                  site: str = "A", plant_tick: int = BASE_TICK, planting=None, defusing=None,
                  defuse_kit=None, c4=201, kills=None, damage=None, health=None, armor=None,
                  positions=None, speeds=None, equipment=None, seed=None,
                  kinds=("grid",)) -> Snapshot:
    """A Snapshot that passes `validate()`. Dead players: the last `5 - alive_*` of each
    side plus every id in `dead`. Utility specs are `(x, y, z, owner)` tuples or dicts
    with `x, y, z, owner, start_tick, radius` (`radius: "unknown"` = no extent, token
    radius None)."""
    rng = random.Random(seed)
    dead = set(dead) | set(CT_IDS[alive_ct:]) | set(T_IDS[alive_t:])
    blind, kills, damage = blind or {}, kills or {}, damage or {}
    health, armor, positions = health or {}, armor or {}, positions or {}
    speeds, equipment = speeds or {}, equipment or {}

    found = []
    for kind, specs in (("smoke", smokes), ("fire", fires), ("decoy", decoys)):
        for n, spec in enumerate(specs):
            start, tok = _util(kind, spec, tick)
            found.append((start, kind, n, tok))
    found.sort(key=lambda f: f[:3])
    utility = [f[3] for f in found]

    players = []
    for pid in CT_IDS + T_IDS:
        x, y, z = positions.get(pid, home(pid))
        jx, jy = rng.uniform(-40, 40), rng.uniform(-40, 40)     # drawn always: same stream per seed
        if seed is not None and pid not in positions:
            x, y = x + jx, y + jy
        cell = _cell(x, y)
        common = dict(side=side_of(pid), x=float(x), y=float(y), z=float(z), nav_area=cell,
                      callout=f"cell_{cell}", grid_cell=cell, is_scoped=0, is_ducking=0,
                      is_walking=0, is_reloading=0, is_defusing=int(pid == defusing),
                      is_planting=int(pid == planting), seconds_since_last_seen_by_enemy=None,
                      seconds_since_last_shot=None, kills_round=float(kills.get(pid, 0)),
                      damage_round=float(damage.get(pid, 0)), player_id=pid, tier="pro")
        if pid in dead:
            players.append(PlayerToken(**common, alive=0, yaw=None, pitch=None, speed=None,
                                       health=0.0, armor=0.0, has_helmet=None, has_defuser=None,
                                       has_c4=None, weapon_class=None, n_smoke=None, n_flash=None,
                                       n_he=None, n_molly=None, n_decoy=None,
                                       flash_duration_remaining=0.0, in_smoke=None, on_fire=None,
                                       money=None, equipment_value=None, spotted_by_count=None,
                                       heard_enemy=None))
            continue
        pos = (float(x), float(y), float(z))
        players.append(PlayerToken(
            **common, alive=1, yaw=0.0 if pid < 200 else 180.0, pitch=0.0,
            speed=float(speeds.get(pid, 0.0)), health=float(health.get(pid, 100.0)),
            armor=float(armor.get(pid, 100.0)), has_helmet=1,
            has_defuser=int(pid in (101, 102)), has_c4=int(pid == c4 and not planted),
            weapon_class="rifle", n_smoke=1, n_flash=1, n_he=0, n_molly=0, n_decoy=0,
            flash_duration_remaining=float(blind.get(pid, 0.0)),
            in_smoke=_inside_flag(utility, "smoke", pos),
            on_fire=_inside_flag(utility, "fire", pos),
            money=800.0 + 10 * (pid % 100), equipment_value=float(equipment.get(pid, 4700.0 if pid < 200 else 4500.0)),
            spotted_by_count=0, heard_enemy=0))

    def tot(side, field):
        return float(sum(getattr(p, field) for p in players if p.side == side and p.alive))

    def roster_money(side):           # the builder's definition: every account, dead included
        return float(sum(800.0 + 10 * (pid % 100) for pid in (CT_IDS if side == "ct" else T_IDS)))

    into = BASE_INTO_S + (tick - BASE_TICK) / TICK_RATE
    since = (tick - plant_tick) / TICK_RATE if planted else None
    carrier = next((p for p in players if p.player_id == c4), None)
    ctx = ContextToken(
        map_version="de_test@cf", tier="pro",
        time_remaining_s=(BOMB_S - since) if planted else ROUND_S - into, time_into_round_s=into,
        bomb_planted=int(planted), bomb_site=site if planted else None, time_since_plant_s=since,
        defuse_in_progress=int(defusing is not None),
        defuse_with_kit=defuse_kit if defusing is not None else None,
        bomb_carrier_alive=None if planted else int(carrier is not None and carrier.alive == 1),
        alive_ct=sum(p.alive for p in players if p.side == "ct"),
        alive_t=sum(p.alive for p in players if p.side == "t"),
        equip_ct=tot("ct", "equipment_value"), equip_t=tot("t", "equipment_value"),
        money_ct=roster_money("ct"), money_t=roster_money("t"),
        loss_streak_ct=0, loss_streak_t=1, score_ct=3, score_t=3, round_num=ROUND, half=1,
        is_overtime=0, n_active_smokes=sum(u.kind == "smoke" for u in utility),
        n_active_fires=sum(u.kind == "fire" for u in utility))
    meta = SnapshotMeta(match_key=MATCH, round_num=ROUND, tick=tick, t_s=tick / TICK_RATE, kinds=kinds)
    return Snapshot(players=players, utility=utility, ctx=ctx, pair=build_pair(players, utility),
                    meta=meta, labels={"round_won_ct": 1})


# ── actions ───────────────────────────────────────────────────────────────────

def action(type_: str, actor: int, tick: int = BASE_TICK, kind: str = "instant", **params) -> Action:
    a = Action(match_key=MATCH, round_num=ROUND, actor_id_hash=actor, actor_side=side_of(actor),
               kind=kind, type=type_, tick=tick, params=params,
               state_pre_tick=tick - 1, state_post_tick=tick + 1)
    if kind == "durative":
        a.window_start_tick, a.window_end_tick, a.tick = tick, tick + 18 * TICK_RATE, None
    return a


def _merge(common: dict, own: dict) -> dict:
    out = dict(common)
    for k, v in own.items():
        if k == "dead":
            out[k] = set(out.get(k, ())) | set(v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def pair_states(pre: dict, post: dict, **common) -> tuple[Snapshot, Snapshot]:
    """(pre at tick 4999, post at tick 5001) from shared + per-side make_snapshot kwargs."""
    s_pre  = make_snapshot(**_merge({**common, "tick": BASE_TICK - 1, "kinds": ("boundary_pre",)}, pre))
    s_post = make_snapshot(**_merge({**common, "tick": BASE_TICK + 1, "kinds": ("boundary_post",)}, post))
    return s_pre, s_post


def make_kill_pair(attacker: int = 101, victim: int = 202, **common):
    """`attacker` kills `victim` at 5000 (full-health victim, 100 lethal damage)."""
    s_pre, s_post = pair_states({}, {"dead": {victim}, "kills": {attacker: 1}, "damage": {attacker: 100}},
                                **common)
    kill = action("kill", attacker, victim_id=victim, victim_health_before=100.0, weapon="ak47")
    return s_pre, s_post, kill


def make_smoke_pair(owner: int = 103, at=(1450.0, 400.0, 0.0), fade: bool = False, **common):
    """A smoke pops (or, with `fade`, fades) at 5000. The default spot covers T 205."""
    tok = {"x": at[0], "y": at[1], "z": at[2], "owner": owner,
           "start_tick": 5000 - 18 * TICK_RATE if fade else 5000}
    extra = list(common.pop("smokes", ()))
    with_tok, without = {"smokes": extra + [tok]}, {"smokes": extra}
    if fade:
        s_pre, s_post = pair_states(with_tok, without, **common)
    else:
        s_pre, s_post = pair_states(without, with_tok, **common)
    return s_pre, s_post, action("smoke_fade" if fade else "smoke_pop", owner, parent_action_id="p" * 16)


def make_flash_pair(thrower: int = 101, enemies=None, teammates=None, **common):
    """`thrower`'s flash at 5000 blinds `enemies` / `teammates` ({id: seconds})."""
    enemies   = {203: 2.5, 204: 1.0} if enemies is None else enemies
    teammates = {102: 0.5} if teammates is None else teammates
    s_pre, s_post = pair_states({}, {"blind": {**enemies, **teammates}}, **common)
    fl = action("flash", thrower,
                blinded_enemies=[{"id": k, "duration": v} for k, v in enemies.items()],
                blinded_teammates=[{"id": k, "duration": v} for k, v in teammates.items()],
                blinded_self=False)
    return s_pre, s_post, fl


def make_plant_pair(planter: int = 201, site: str = "A", **common):
    """`planter` (carrying the bomb, planting) completes the plant at 5000."""
    s_pre, s_post = pair_states({"planting": planter, "c4": planter},
                                {"planted": True, "site": site, "plant_tick": BASE_TICK, "c4": None},
                                **common)
    return s_pre, s_post, action("plant_complete", planter, site=site)


def make_move_pair(mover: int = 101, to=(400.0, 0.0, 0.0), dt_ticks: int = 64, **common):
    """`mover` walks from home to `to` over `dt_ticks` ending at 5000 (a peek)."""
    base = dict(common)
    pos = base.pop("positions", {})
    s0 = make_snapshot(**_merge({**base, "tick": BASE_TICK - dt_ticks, "positions": pos}, {}))
    s1 = make_snapshot(**_merge({**base, "tick": BASE_TICK, "positions": {**pos, mover: to},
                                 "speeds": {mover: 250.0}}, {}))
    return s0, s1, action("peek", mover)


def make_window(**common):
    """One event window at 5000 with five members: 101 kills 202; 104's flash blinds
    203, 204 and the victim 202; 103's smoke pops over T 205; 105's decoy starts far
    from everyone; 102 deals 30 non-lethal damage to 205. Returns (s_pre, s_post, members)."""
    smoke = {"x": 1450.0, "y": 400.0, "z": 0.0, "owner": 103, "start_tick": 5000}
    decoy = {"x": -3000.0, "y": -3000.0, "z": 0.0, "owner": 105, "start_tick": 5000}
    s_pre, s_post = pair_states(
        {},
        {"dead": {202}, "kills": {101: 1}, "damage": {101: 100, 102: 30}, "blind": {203: 2.0, 204: 1.5},
         "smokes": [smoke], "decoys": [decoy], "health": {205: 70.0}},
        **common)
    members = [
        action("kill", 101, victim_id=202, victim_health_before=100.0),
        action("flash", 104, blinded_enemies=[{"id": 203, "duration": 2.0}, {"id": 204, "duration": 1.5},
                                              {"id": 202, "duration": 3.0}], blinded_teammates=[]),
        action("smoke_pop", 103),
        action("decoy_start", 105),
        action("damage", 102, victim_id=205, damage_total=30.0, armor_damage=0.0),
    ]
    return s_pre, s_post, members


# ── predictor doubles ─────────────────────────────────────────────────────────

def _blind_term(s) -> float:
    ct = sum(min(p.flash_duration_remaining, 3.0) for p in s.players if p.side == "ct" and p.alive)
    t  = sum(min(p.flash_duration_remaining, 3.0) for p in s.players if p.side == "t" and p.alive)
    return 0.01 * (t - ct)


def good_win(s, h=None) -> float:
    """Sensible W: +0.1 per man advantage, -0.15 when planted, blind side worse."""
    w = 0.5 + 0.1 * (s.ctx.alive_ct - s.ctx.alive_t) - 0.15 * s.ctx.bomb_planted + _blind_term(s)
    return min(0.98, max(0.02, w))


def broken_win(s, h=None) -> float:
    """Backwards W: reviving a player helps the other side, planting helps CT."""
    w = 0.5 - 0.1 * (s.ctx.alive_ct - s.ctx.alive_t) + 0.15 * s.ctx.bomb_planted
    return min(0.98, max(0.02, w))


class RuleStub:
    """Predictor double: `win` from a rule, the other heads NaN."""

    def __init__(self, win):
        self.win   = win
        self.calls = 0

    def predict(self, snapshots, histories) -> dict:
        if len(snapshots) != len(histories):
            raise ValueError("snapshots and histories differ in length")
        self.calls += 1
        n = len(snapshots)
        return {"win": np.array([self.win(s, h) for s, h in zip(snapshots, histories)], dtype=np.float64),
                "hazard": np.full((n, N_PLAYERS), np.nan), "plant15": np.full(n, np.nan),
                "defuse15": np.full(n, np.nan)}


def embed(snapshots) -> np.ndarray:
    """A toy embedding for OOD tests: positions (per 1000 u), alive bits, bomb state."""
    rows = []
    for s in snapshots:
        r = []
        for p in s.players:
            r += [p.x / 1000.0, p.y / 1000.0, float(p.alive)]
        r += [float(s.ctx.bomb_planted), s.ctx.n_active_smokes / 4.0]
        rows.append(r)
    return np.asarray(rows, dtype=np.float32)


# ── gate samples ──────────────────────────────────────────────────────────────

def gate_samples(n: int = 30, seed: int = 0) -> list[tuple]:
    """`(kind, s_pre, s_post, action)` tuples for every P7.2 test, varied by seed:
    kills both ways, smokes of both sides, flashes of both sides, plants, peeks, and
    multi-member windows (kind "order", action = list of members)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        sd = seed * 1000 + i
        att, vic = (rng.choice(CT_IDS), rng.choice(T_IDS))
        if i % 2:
            att, vic = vic, att
        out.append(("kill",) + make_kill_pair(att, vic, seed=sd))
        owner = rng.choice(CT_IDS + T_IDS)
        at = (rng.uniform(300, 1200), rng.uniform(0, 400), 0.0)
        out.append(("smoke",) + make_smoke_pair(owner, at=at, seed=sd))
        thrower = rng.choice(CT_IDS + T_IDS)
        foes = T_IDS if thrower < 200 else CT_IDS
        enemies = {f: round(rng.uniform(0.5, 3.0), 2) for f in rng.sample(foes, 2)}
        out.append(("flash",) + make_flash_pair(thrower, enemies=enemies, teammates={}, seed=sd))
        out.append(("plant",) + make_plant_pair(rng.choice(T_IDS), site=rng.choice("AB"), seed=sd))
        mover = rng.choice(CT_IDS + T_IDS)
        hx, hy, _ = home(mover)
        out.append(("freeze",) + make_move_pair(mover, to=(hx + rng.uniform(-300, 300), hy + rng.uniform(-200, 200), 0.0),
                                                seed=sd))
        if i % 3 == 0:
            out.append(("order",) + make_window(seed=sd))
    return out
