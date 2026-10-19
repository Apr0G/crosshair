"""
Synthetic awpy-3-shaped `RawTables` for engine tests (P3.9). Stdlib + numpy + polars.

    from _synthetic import make_raw_tables, STORIES, STORY_FACTS, PLAYERS, scale_tick
    raw = make_raw_tables("flash", tick_rate=64)
    raw = make_raw_tables("base", spotted_by=False)      # ticks.spotted_by all null, as the parser emits

The 2v2 skeleton every story shares (ticks written for 128 tick; at another tick rate
every tick is scaled by tick_rate/128, so a 64-tick story has half the ticks and the
same durations in seconds):
- players A (1001) and B (1002) on CT, C (2001) and D (2002) on T; map "de_test".
- round 1: start 900, freeze_end 1000; a tick row for every tick from start_tick to
  official_end_tick (inclusive) for all 4 players, dead players included.
- positions (Hammer units, z = 0): A stands at (0, 300) looking +x (yaw 0); B walks at
  250 u/s from (0, 0) at 1000 to (2000, 0) at 2024; C stands at (3000, 300) and D at
  (3000, -300), both yaw 180. After plant_complete D walks to (3400, -300) at 250 u/s.
- A's smoke pops at 1500 at (1000, 0, 0) and fades at 1500 + 18 s; A's HE detonates at
  1800 at (1500, 0, 0) and damages C 30 and D 20. D plants at site "A" (3000, -300):
  plant_start 2900, plant_complete 3000.
- round 2 starts 5 s after round 1's official end (5000 in `base`); freeze_end +100
  ticks; D plants at fe+5 s..fe+8 s; A walks to the bomb and defuses with a kit from
  fe+15 s to fe+20 s; round ends ct_win_defuse with no kills. official_end = end+2.5 s.

Stories (STORIES; per-story counts in STORY_FACTS):
- base:  A kills C at 2000 (ak47, headshot); B kills D at 4000 (m4a1); round ends 4040
  ct_win_elimination, official_end 4360.
- flash: base + A's flash pops at 1900 at (1400, 0, 0) blinding D 2.5 s and C 1.0 s,
  so C is blind at 2000 (kills.victim_blind true).
- double_kill_same_tick: at 2000 A kills C AND C kills B; B is dead, so A (not B)
  kills D at 4000 (ak47).
- trade: C kills A at 2000, B kills C at 2100 (kills.is_trade true); D survives; the
  bomb explodes at 3000 + 40 s, round ends t_win_bomb.
- retreat: C survives (no kill at 2000); after the plant C runs straight away from
  the bomb (+y) at 250 u/s from 3000 to 3640; B walks to the bomb, starts defusing with
  a kit at 3900, kills D at 4000, defuse completes at 4200 (ct_win_defuse). The defuse
  is shorter than a real kit defuse and overlaps the kill: the plan fixed those ticks.
- drop: base, except A also carries a spare m4a1 (not possible in game, needed for the
  story) and B starts without a primary; A drops weapon_m4a1 at 1100, B picks it up at
  1150 and kills D with it.

Naming: inventory / active_weapon / items use item class names (`weapon_ak47`);
kills / damages / weapon_fire use the short weapon name (`ak47`, `hegrenade`).
"""
import hashlib
import math

import numpy as np
import polars as pl

from _harness import ROOT  # importing _harness puts the repo root on sys.path
from engine.parse.schema import TABLE_SCHEMAS, RawTables

# ── constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = ROOT
REF_TR  = 128
SPEED   = 250.0
MAP     = "de_test"
STORIES = ["base", "flash", "double_kill_same_tick", "trade", "retreat", "drop"]

PLAYERS: dict[str, tuple[int, str]] = {"A": (1001, "ct"), "B": (1002, "ct"),
                                       "C": (2001, "t"), "D": (2002, "t")}
START_POS = {"A": (0.0, 300.0), "B": (0.0, 0.0), "C": (3000.0, 300.0), "D": (3000.0, -300.0)}
YAW       = {"A": 0.0, "B": 0.0, "C": 180.0, "D": 180.0}
BOMB_POS  = (3000.0, -300.0)
SMOKE_RADIUS = 144.0

PRICES = {"weapon_knife": 0, "weapon_usp_silencer": 200, "weapon_glock": 200,
          "weapon_ak47": 2700, "weapon_m4a1": 2900, "weapon_smokegrenade": 300,
          "weapon_hegrenade": 300, "weapon_flashbang": 200, "weapon_c4": 0}
ARMOR_HELMET_VALUE = 1000
DEFUSER_VALUE      = 400
PRIMARIES = ("weapon_ak47", "weapon_m4a1")
PISTOLS   = ("weapon_usp_silencer", "weapon_glock")
NADE_ITEM = {"smoke": "weapon_smokegrenade", "he": "weapon_hegrenade", "flash": "weapon_flashbang"}
NADE_WEAPON = {"smoke": "smokegrenade", "he": "hegrenade", "flash": "flashbang"}

MONEY = {1: {"A": 650, "B": 800, "C": 400, "D": 300},
         2: {"A": 3100, "B": 2900, "C": 1900, "D": 2400}}
HAS_KIT = {"A": True, "B": True, "C": False, "D": False}


def scale_tick(t128: int, tick_rate: int) -> int:
    """A tick written for 128 tick, at `tick_rate`."""
    return int(round(t128 * tick_rate / REF_TR))


# ── per-story facts (tick-rate independent; ticks are 128-tick reference ticks) ──

def _facts(kills_r1, grenades, flashes, bomb_r1, damages, items, winner, reason, r1_end, r2_start):
    weapon_fire = len(kills_r1) + grenades
    return {
        "rounds": 2, "kills": len(kills_r1), "kills_round": {1: len(kills_r1), 2: 0},
        "kill_events": kills_r1, "smokes": 1, "infernos": 0, "grenades": grenades,
        "flashes": flashes, "bomb_events": len(bomb_r1) + 4,
        "bomb_round": {1: bomb_r1, 2: ["plant_start", "plant_complete", "defuse_start", "defuse_complete"]},
        "damages": damages, "items": items, "weapon_fire": weapon_fire,
        "winner": {1: winner, 2: "ct"}, "end_reason": {1: reason, 2: "ct_win_defuse"},
        "r1_end": r1_end, "r1_official_end": r1_end + 320, "r2_start": r2_start,
    }


_PLANT = ["plant_start", "plant_complete"]
STORY_FACTS: dict[str, dict] = {
    "base":  _facts([(2000, "A", "C"), (4000, "B", "D")], 2, 0, _PLANT, 4, 0,
                    "ct", "ct_win_elimination", 4040, 5000),
    "flash": _facts([(2000, "A", "C"), (4000, "B", "D")], 3, 2, _PLANT, 4, 0,
                    "ct", "ct_win_elimination", 4040, 5000),
    "double_kill_same_tick": _facts([(2000, "A", "C"), (2000, "C", "B"), (4000, "A", "D")], 2, 0,
                                    _PLANT, 5, 0, "ct", "ct_win_elimination", 4040, 5000),
    "trade": _facts([(2000, "C", "A"), (2100, "B", "C")], 2, 0, _PLANT + ["explode"], 4, 0,
                    "t", "t_win_bomb", 3000 + 40 * 128, 3000 + 40 * 128 + 320 + 640),
    "retreat": _facts([(4000, "B", "D")], 2, 0, _PLANT + ["defuse_start", "defuse_complete"], 3, 0,
                      "ct", "ct_win_defuse", 4200, 4200 + 320 + 640),
    "drop":  _facts([(2000, "A", "C"), (4000, "B", "D")], 2, 0, _PLANT, 4, 2,
                    "ct", "ct_win_elimination", 4040, 5000),
}
STORY_FACTS["trade"]["trade_kills"] = [(2100, "B", "C")]
STORY_FACTS["flash"]["blinded"] = {"D": 2.5, "C": 1.0}
STORY_FACTS["drop"]["item_events"] = [(1100, "A", "drop", "weapon_m4a1"), (1150, "B", "pickup", "weapon_m4a1")]
STORY_FACTS["retreat"]["retreat"] = {"player": "C", "from_tick": 3000, "to_tick": 3640, "units": 1250.0}


# ── story description ─────────────────────────────────────────────────────────

class _Story:
    """Events of one story at one tick rate, in real ticks."""

    def __init__(self, tr: int):
        self.tr       = tr
        self.rounds   = []      # dicts
        self.paths    = {}      # (round, player) -> [(tick, x, y)]
        self.loadout  = {}      # (round, player) -> [items]
        self.kills    = []
        self.damages  = []
        self.grenades = []
        self.flashes  = []
        self.bomb     = []
        self.items    = []

    def T(self, t128: float) -> int:
        return int(round(t128 * self.tr / REF_TR))

    def S(self, seconds: float) -> int:
        return int(round(seconds * self.tr))

    def walk(self, rnd: int, p: str, depart: int | None, dest: tuple[float, float],
             arrive: int | None = None) -> None:
        """Append a straight walk at SPEED from the path's last point. Give `depart`
        or `arrive`; the other end follows from the distance."""
        pts = self.paths[(rnd, p)]
        _, x0, y0 = pts[-1]
        dur = self.S(math.dist((x0, y0), dest) / SPEED)
        if depart is None:
            depart = arrive - dur
        pts.append((depart, x0, y0))
        pts.append((depart + dur, *dest))

    def kill(self, tick, rnd, att, vic, weapon, headshot, hitgroup, is_trade=False):
        self.kills.append(dict(tick=tick, round_num=rnd, att=att, vic=vic, weapon=weapon,
                               headshot=headshot, hitgroup=hitgroup, is_trade=is_trade))

    def damage(self, tick, rnd, att, vic, weapon, dmg, armor, hitgroup):
        self.damages.append(dict(tick=tick, round_num=rnd, att=att, vic=vic, weapon=weapon,
                                 dmg_health=dmg, dmg_armor=armor, hitgroup=hitgroup))

    def grenade(self, eid, thrower, kind, throw, det, expire, pos, rnd):
        self.grenades.append(dict(entity_id=eid, thrower=thrower, type=kind, throw_tick=throw,
                                  detonate_tick=det, expire_tick=expire, pos=pos, round_num=rnd))

    def bomb_event(self, tick, rnd, event, p, site="A", has_kit=None, pos=BOMB_POS):
        self.bomb.append(dict(tick=tick, round_num=rnd, event=event, p=p, site=site,
                              has_kit=has_kit, pos=pos))


def _base_loadout(p: str) -> list[str]:
    return {"A": ["weapon_knife", "weapon_usp_silencer", "weapon_ak47",
                  "weapon_smokegrenade", "weapon_hegrenade"],
            "B": ["weapon_knife", "weapon_usp_silencer", "weapon_m4a1"],
            "C": ["weapon_knife", "weapon_glock", "weapon_ak47"],
            "D": ["weapon_knife", "weapon_glock", "weapon_ak47", "weapon_c4"]}[p]


def _describe(story: str, tr: int) -> _Story:
    s = _Story(tr)
    T, S = s.T, s.S

    # ── round 1 ──
    for p in PLAYERS:
        s.paths[(1, p)]   = [(T(900), *START_POS[p])]
        s.loadout[(1, p)] = _base_loadout(p)
    s.paths[(1, "B")] = [(T(1000), 0.0, 0.0), (T(2024), 2000.0, 0.0)]

    s.grenade(101, "A", "smoke", T(1400), T(1500), T(1500) + S(18), (1000.0, 0.0, 0.0), 1)
    s.grenade(102, "A", "he", T(1700), T(1800), None, (1500.0, 0.0, 0.0), 1)
    s.damage(T(1800), 1, "A", "C", "hegrenade", 30, 5, "generic")
    s.damage(T(1800), 1, "A", "D", "hegrenade", 20, 5, "generic")
    s.bomb_event(T(2900), 1, "plant_start", "D")
    s.bomb_event(T(3000), 1, "plant_complete", "D")
    s.walk(1, "D", T(3000), (3400.0, -300.0))

    end, reason, winner = T(4040), "ct_win_elimination", "ct"
    if story in ("base", "flash", "drop"):
        s.kill(T(2000), 1, "A", "C", "ak47", True, "head")
        s.kill(T(4000), 1, "B", "D", "m4a1", False, "chest")
    if story == "flash":
        s.loadout[(1, "A")] = _base_loadout("A") + ["weapon_flashbang"]
        s.grenade(103, "A", "flash", T(1850), T(1900), None, (1400.0, 0.0, 0.0), 1)
        s.flashes += [dict(entity_id=103, detonate_tick=T(1900), p="D", duration=2.5),
                      dict(entity_id=103, detonate_tick=T(1900), p="C", duration=1.0)]
    if story == "drop":
        s.loadout[(1, "A")] = _base_loadout("A") + ["weapon_m4a1"]
        s.loadout[(1, "B")] = ["weapon_knife", "weapon_usp_silencer"]
        s.items += [dict(tick=T(1100), p="A", event="drop", item="weapon_m4a1"),
                    dict(tick=T(1150), p="B", event="pickup", item="weapon_m4a1")]
    if story == "double_kill_same_tick":
        s.kill(T(2000), 1, "A", "C", "ak47", True, "head")
        s.kill(T(2000), 1, "C", "B", "ak47", True, "head")
        s.kill(T(4000), 1, "A", "D", "ak47", False, "chest")
    if story == "trade":
        s.kill(T(2000), 1, "C", "A", "ak47", True, "head")
        s.kill(T(2100), 1, "B", "C", "m4a1", False, "chest", is_trade=True)
        end, reason, winner = T(3000) + S(40), "t_win_bomb", "t"
        s.bomb_event(end, 1, "explode", None)
    if story == "retreat":
        s.walk(1, "C", T(3000), (3000.0, 300.0 + 1250.0))
        s.walk(1, "B", None, BOMB_POS, arrive=T(3900))
        s.bomb_event(T(3900), 1, "defuse_start", "B", has_kit=True)
        s.kill(T(4000), 1, "B", "D", "m4a1", False, "chest")
        end, reason, winner = T(4200), "ct_win_defuse", "ct"
        s.bomb_event(end, 1, "defuse_complete", "B", has_kit=True)
    official = end + S(2.5)
    s.rounds.append(dict(round_num=1, start=T(900), freeze_end=T(1000), end=end,
                         official_end=official, winner=winner, reason=reason,
                         bomb_planted=True, ct_before=0, t_before=0))

    # ── round 2 ──
    start = official + S(5)
    fe    = start + (T(1000) - T(900))
    for p in PLAYERS:
        s.paths[(2, p)]   = [(start, *START_POS[p])]
        s.loadout[(2, p)] = _base_loadout(p)
    s.bomb_event(fe + S(5), 2, "plant_start", "D")
    s.bomb_event(fe + S(8), 2, "plant_complete", "D")
    s.walk(2, "D", fe + S(8), (3400.0, -300.0))
    s.walk(2, "A", None, BOMB_POS, arrive=fe + S(15))
    s.bomb_event(fe + S(15), 2, "defuse_start", "A", has_kit=True)
    end2 = fe + S(20)
    s.bomb_event(end2, 2, "defuse_complete", "A", has_kit=True)
    s.rounds.append(dict(round_num=2, start=start, freeze_end=fe, end=end2,
                         official_end=end2 + S(2.5), winner="ct", reason="ct_win_defuse",
                         bomb_planted=True, ct_before=int(winner == "ct"), t_before=int(winner == "t")))
    return s


# ── geometry helpers ──────────────────────────────────────────────────────────

def _pos(s: _Story, rnd: int, p: str, ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = s.paths[(rnd, p)]
    pt  = np.array([q[0] for q in pts], dtype=np.float64)
    return (np.interp(ticks, pt, [q[1] for q in pts]),
            np.interp(ticks, pt, [q[2] for q in pts]))


def _seg_point_dist(a, b, c) -> float:
    a, b, c = (np.asarray(v, dtype=np.float64) for v in (a, b, c))
    ab = b - a
    t  = 0.0 if not ab.any() else float(np.clip(np.dot(c - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - c))


def _flash_left(s: _Story, p: str, tick: int) -> float:
    left = 0.0
    for f in s.flashes:
        if f["p"] == p and tick >= f["detonate_tick"]:
            left = max(left, f["duration"] - (tick - f["detonate_tick"]) / s.tr)
    return left


# ── table builders ────────────────────────────────────────────────────────────

def _death_tick(s: _Story, rnd: int, p: str) -> int | None:
    ts = [k["tick"] for k in s.kills if k["round_num"] == rnd and k["vic"] == p]
    return min(ts) if ts else None


def _lethal_damages(s: _Story) -> list[dict]:
    """Kill hits as damage rows: dmg_health = victim's health at the kill tick."""
    out = []
    for k in sorted(s.kills, key=lambda k: k["tick"]):
        taken = sum(d["dmg_health"] for d in s.damages
                    if d["round_num"] == k["round_num"] and d["vic"] == k["vic"] and d["tick"] < k["tick"])
        out.append(dict(tick=k["tick"], round_num=k["round_num"], att=k["att"], vic=k["vic"],
                        weapon=k["weapon"], dmg_health=100 - taken, dmg_armor=3 if k["headshot"] else 5,
                        hitgroup=k["hitgroup"]))
    return out


def _inventory_changes(s: _Story, rnd: int, p: str) -> list[tuple[int, list[str]]]:
    """(tick, inventory from that tick on), starting at the round start."""
    events = []
    for g in s.grenades:
        if g["round_num"] == rnd and g["thrower"] == p:
            events.append((g["throw_tick"], "remove", NADE_ITEM[g["type"]]))
    for it in s.items:
        r = next(r for r in s.rounds if r["start"] <= it["tick"] <= r["official_end"])
        if r["round_num"] == rnd and it["p"] == p:
            events.append((it["tick"], "remove" if it["event"] == "drop" else "add", it["item"]))
    for b in s.bomb:
        if b["round_num"] == rnd and b["p"] == p and b["event"] == "plant_complete":
            events.append((b["tick"], "remove", "weapon_c4"))
    inv = list(s.loadout[(rnd, p)])
    rstart = next(r["start"] for r in s.rounds if r["round_num"] == rnd)
    out = [(rstart, list(inv))]
    for tick, op, item in sorted(events):
        if op == "remove":
            inv.remove(item)
        else:
            inv.append(item)
        out.append((tick, list(inv)))
    return out


def _windows(s: _Story, rnd: int, p: str, start_ev: str, end_evs: tuple[str, ...]) -> list[tuple[int, int]]:
    """[start, end) spans between a start bomb event by `p` and the next end event."""
    evs = sorted((b for b in s.bomb if b["round_num"] == rnd), key=lambda b: b["tick"])
    spans = []
    for b in evs:
        if b["event"] == start_ev and b["p"] == p:
            nxt = next((e["tick"] for e in evs if e["tick"] > b["tick"] and e["event"] in end_evs), None)
            spans.append((b["tick"], nxt))
    return spans


def _player_round_frame(s: _Story, rnd: int, p: str) -> pl.DataFrame:
    r     = next(r for r in s.rounds if r["round_num"] == rnd)
    steam, side = PLAYERS[p]
    ticks = np.arange(r["start"], r["official_end"] + 1, dtype=np.int64)
    death = _death_tick(s, rnd, p)
    alive = ticks < death if death is not None else np.ones(len(ticks), dtype=bool)
    clamp = np.minimum(ticks, death) if death is not None else ticks
    x, y  = _pos(s, rnd, p, clamp)
    xp, yp = _pos(s, rnd, p, np.maximum(clamp - 1, r["start"]))
    vx = np.where(alive, (x - xp) * s.tr, 0.0)
    vy = np.where(alive, (y - yp) * s.tr, 0.0)

    all_dmg = s.damages + _lethal_damages(s)
    dmg_in  = [d for d in all_dmg if d["round_num"] == rnd and d["vic"] == p]
    dmg_out = [d for d in all_dmg if d["round_num"] == rnd and d["att"] == p]
    # cumulative sums are evaluated at the ticks where they change, then broadcast
    ticks_change = sorted({r["start"], *(d["tick"] for d in dmg_in + dmg_out),
                           *(k["tick"] for k in s.kills if k["round_num"] == rnd)})
    idx = np.searchsorted(np.array(ticks_change), ticks, side="right") - 1

    def cum(rows, key):
        return np.array([sum(d[key] for d in rows if d["tick"] <= t) for t in ticks_change], dtype=np.int64)

    health = np.where(alive, np.clip(100 - cum(dmg_in, "dmg_health")[idx], 0, 100), 0)
    armor  = np.where(alive, np.clip(100 - cum(dmg_in, "dmg_armor")[idx], 0, 100), 0)
    dealt  = cum(dmg_out, "dmg_health")[idx]
    kc_t   = [k["tick"] for k in s.kills if k["round_num"] == rnd and k["att"] == p]
    kills_round = np.array([sum(kt <= t for kt in kc_t) for t in ticks_change], dtype=np.int64)[idx]

    changes  = _inventory_changes(s, rnd, p)
    inv_idx  = np.searchsorted(np.array([c[0] for c in changes]), ticks, side="right") - 1
    plants   = _windows(s, rnd, p, "plant_start", ("plant_complete",))
    defuses  = _windows(s, rnd, p, "defuse_start", ("defuse_complete", "defuse_abort", "explode"))

    def in_spans(t, spans):
        return any(a <= t < (b if b is not None else t + 1) for a, b in spans)

    spot_pairs = []   # (from, to, spotter) for this player
    for k in s.kills:
        if k["round_num"] != rnd:
            continue
        for me, other in ((k["att"], k["vic"]), (k["vic"], k["att"])):
            if me == p:
                spot_pairs.append((k["tick"] - s.tr // 2, k["tick"], other))

    cols = {c: [] for c in TABLE_SCHEMAS["ticks"]}
    for i, t in enumerate(ticks.tolist()):
        a   = bool(alive[i])
        inv = changes[inv_idx[i]][1] if a else []
        planting = a and in_spans(t, plants)
        if not a:
            active = None
        elif planting:
            active = "weapon_c4"
        else:
            active = next((w for w in PRIMARIES if w in inv),
                          next((w for w in PISTOLS if w in inv), "weapon_knife"))
        equip = 0
        if a:
            equip = sum(PRICES[w] for w in inv) + (ARMOR_HELMET_VALUE if armor[i] > 0 else 0) \
                    + (DEFUSER_VALUE if HAS_KIT[p] else 0)
        spotters = []
        for lo, hi, other in spot_pairs:
            o_death = _death_tick(s, rnd, other)
            if a and lo <= t <= hi and (o_death is None or t < o_death):
                spotters.append(PLAYERS[other][0])
        row = dict(
            tick=t, round_num=rnd, steamid=steam, name=p, side=side,
            x=float(x[i]), y=float(y[i]), z=0.0, yaw=YAW[p], pitch=0.0,
            vx=float(vx[i]), vy=float(vy[i]), vz=0.0,
            health=int(health[i]), armor=int(armor[i]),
            has_helmet=a, has_defuser=a and HAS_KIT[p], has_c4=a and "weapon_c4" in inv,
            is_alive=a, active_weapon=active, inventory=list(inv), money=MONEY[rnd][p],
            equipment_value=int(equip), is_scoped=False, is_ducking=False, is_walking=False,
            is_reloading=False, is_defusing=a and in_spans(t, defuses), is_planting=planting,
            flash_duration=float(_flash_left(s, p, t)) if a else 0.0,
            spotted_by=sorted(spotters), kills_round=int(kills_round[i]), damage_round=int(dealt[i]),
        )
        for c in cols:
            cols[c].append(row[c])
    return pl.DataFrame(cols, schema=TABLE_SCHEMAS["ticks"])


def _ticks(s: _Story) -> pl.DataFrame:
    frames = [_player_round_frame(s, r["round_num"], p) for r in s.rounds for p in PLAYERS]
    return pl.concat(frames).sort(["tick", "steamid"])


def _at(s: _Story, ticks: pl.DataFrame, p: str, tick: int) -> tuple[float, float, float]:
    row = ticks.filter((pl.col("tick") == tick) & (pl.col("steamid") == PLAYERS[p][0]))
    return float(row["x"][0]), float(row["y"][0]), float(row["z"][0])


def _kills(s: _Story, ticks: pl.DataFrame) -> pl.DataFrame:
    rows = []
    smokes = [g for g in s.grenades if g["type"] == "smoke"]
    for k in s.kills:
        ap, vp = _at(s, ticks, k["att"], k["tick"]), _at(s, ticks, k["vic"], k["tick"])
        through = any(g["detonate_tick"] <= k["tick"] < g["expire_tick"]
                      and _seg_point_dist(ap, vp, g["pos"]) <= SMOKE_RADIUS for g in smokes)
        rows.append(dict(
            tick=k["tick"], round_num=k["round_num"], attacker_steamid=PLAYERS[k["att"]][0],
            victim_steamid=PLAYERS[k["vic"]][0], assister_steamid=None, flash_assister_steamid=None,
            weapon=k["weapon"], headshot=k["headshot"], penetrated=0, through_smoke=through,
            attacker_blind=_flash_left(s, k["att"], k["tick"]) > 0,
            victim_blind=_flash_left(s, k["vic"], k["tick"]) > 0, noscope=False,
            distance=float(math.dist(ap, vp)),
            attacker_x=ap[0], attacker_y=ap[1], attacker_z=ap[2],
            victim_x=vp[0], victim_y=vp[1], victim_z=vp[2], team_kill=False,
            is_trade=k["is_trade"]))
    schema = {**TABLE_SCHEMAS["kills"], "is_trade": pl.Boolean}
    return pl.DataFrame(rows, schema=schema).sort("tick") if rows else pl.DataFrame(schema=schema)


def _frame(rows: list[dict], table: str) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=TABLE_SCHEMAS[table]) if rows else pl.DataFrame(schema=TABLE_SCHEMAS[table])


def _weapon_fire(s: _Story) -> pl.DataFrame:
    rows  = [dict(tick=k["tick"], steamid=PLAYERS[k["att"]][0], weapon=k["weapon"]) for k in s.kills]
    rows += [dict(tick=g["throw_tick"], steamid=PLAYERS[g["thrower"]][0], weapon=NADE_WEAPON[g["type"]])
             for g in s.grenades]
    return _frame(sorted(rows, key=lambda r: (r["tick"], r["steamid"])), "weapon_fire")


def _footsteps(s: _Story, ticks: pl.DataFrame) -> pl.DataFrame:
    """One footstep every 0.3 s while a live player runs faster than 200 u/s."""
    step = max(1, s.S(0.3))
    f = ticks.filter(pl.col("is_alive") & ((pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt() > 200)
                     & (pl.col("tick") % step == 0))
    return f.select(TABLE_SCHEMAS["footsteps"].keys())


# ── public ────────────────────────────────────────────────────────────────────

def make_raw_tables(story: str = "base", tick_rate: int = 128, *, spotted_by: bool = True) -> RawTables:
    """The story as awpy-3-shaped RawTables at `tick_rate`. Passes validate(strict=True).
    `spotted_by=False`: `ticks.spotted_by` is null on every row, which is what the awpy 3
    parser emits (the field is not obtainable; LOW-63)."""
    if story not in STORIES:
        raise ValueError(f"unknown story {story!r}; one of {STORIES}")
    s     = _describe(story, tick_rate)
    ticks = _ticks(s)
    if not spotted_by:
        ticks = ticks.with_columns(pl.lit(None, dtype=TABLE_SCHEMAS["ticks"]["spotted_by"]).alias("spotted_by"))
    rounds = _frame([dict(round_num=r["round_num"], start_tick=r["start"], freeze_end_tick=r["freeze_end"],
                          end_tick=r["end"], official_end_tick=r["official_end"], winner_side=r["winner"],
                          end_reason=r["reason"], bomb_planted=r["bomb_planted"],
                          ct_score_before=r["ct_before"], t_score_before=r["t_before"])
                     for r in s.rounds], "rounds")
    damages = _frame(sorted([dict(tick=d["tick"], round_num=d["round_num"],
                                  attacker_steamid=PLAYERS[d["att"]][0], victim_steamid=PLAYERS[d["vic"]][0],
                                  weapon=d["weapon"], dmg_health=d["dmg_health"], dmg_armor=d["dmg_armor"],
                                  hitgroup=d["hitgroup"]) for d in s.damages + _lethal_damages(s)],
                            key=lambda r: (r["tick"], r["victim_steamid"])), "damages")
    grenades = _frame([dict(entity_id=g["entity_id"], thrower_steamid=PLAYERS[g["thrower"]][0], type=g["type"],
                            throw_tick=g["throw_tick"], detonate_tick=g["detonate_tick"],
                            expire_tick=g["expire_tick"], x=g["pos"][0], y=g["pos"][1], z=g["pos"][2])
                       for g in s.grenades], "grenades")
    smokes = _frame([dict(entity_id=g["entity_id"], start_tick=g["detonate_tick"], end_tick=g["expire_tick"],
                          x=g["pos"][0], y=g["pos"][1], z=g["pos"][2], thrower_steamid=PLAYERS[g["thrower"]][0])
                     for g in s.grenades if g["type"] == "smoke"], "smokes")
    flashes = _frame([dict(entity_id=f["entity_id"], detonate_tick=f["detonate_tick"],
                           player_steamid=PLAYERS[f["p"]][0], blind_duration=f["duration"])
                      for f in s.flashes], "flashes")
    bomb = _frame(sorted([dict(tick=b["tick"], round_num=b["round_num"], event=b["event"],
                               steamid=PLAYERS[b["p"]][0] if b["p"] else None, site=b["site"],
                               has_kit=b["has_kit"], x=b["pos"][0], y=b["pos"][1], z=0.0)
                          for b in s.bomb], key=lambda r: r["tick"]), "bomb")
    items = _frame([dict(tick=i["tick"], steamid=PLAYERS[i["p"]][0], event=i["event"], item=i["item"])
                    for i in s.items], "items")
    meta = {"map": MAP, "tick_rate": tick_rate, "game_version": "synthetic", "server": "synthetic",
            "demo_sha256": hashlib.sha256(f"synthetic:{story}:{tick_rate}".encode()).hexdigest()}
    raw = RawTables(rounds=rounds, ticks=ticks, kills=_kills(s, ticks), damages=damages,
                    grenades=grenades, smokes=smokes, infernos=_frame([], "infernos"), flashes=flashes,
                    bomb=bomb, weapon_fire=_weapon_fire(s), footsteps=_footsteps(s, ticks), items=items,
                    meta=meta)
    raw.validate(strict=True)
    return raw
