"""
Round-level actions (P4.4.1, P4.4.2): one buy per player at freeze_end with loadout
classes from `actions.buy_bands` (exclusive upper bounds, edges tested), team_buy with
the majority class; drop/give from `items` within 5 s, with death drops, self pickups and
late pickups rejected.

    .venv\\Scripts\\python.exe test\\engine\\test_actions_economy.py
"""
import dataclasses
import sys

import polars as pl
from _harness import check, section, summary
from _synthetic import PLAYERS, STORIES, make_raw_tables, scale_tick

from engine import config
from engine.actions import economy, extractor
from engine.state.geometry import FlatGeometry
from engine.state.players import player_id_hash

CFG = config.load()
GEO = FlatGeometry()
PID = {n: player_id_hash(s) for n, (s, _) in PLAYERS.items()}
SID = {n: s for n, (s, _) in PLAYERS.items()}
A5_BUY = {"loadout_class", "equip_value", "money_before", "money_after", "has_armor", "has_helmet",
          "has_kit", "utility_count", "primary_class"}


def events(raw):
    return extractor.extract_event_actions(raw, CFG, "m1", geometry=GEO)


# ── bands ──
section("loadout_class edges (bands are exclusive upper bounds)")
b = CFG.actions.buy_bands
for v, want in ((0, "pistol"), (b.pistol - 1, "pistol"), (b.pistol, "eco"), (b.eco - 1, "eco"), (b.eco, "half"),
                (b.half - 1, "half"), (b.half, "force"), (b.force - 1, "force"), (b.force, "full"), (20000, "full")):
    check(f"{v} -> {want}", economy.loadout_class(v, b) == want, str(economy.loadout_class(v, b)))
check("unknown value -> None", economy.loadout_class(None, b) is None)

# ── buys ──
section("buys (base)")
for tr in (128, 64):
    acts = events(make_raw_tables("base", tr))
    for r in (1, 2):
        buys = [a for a in acts if a.type == "buy" and a.round_num == r]
        tbs = [a for a in acts if a.type == "team_buy" and a.round_num == r]
        fe = scale_tick(1000, tr) if r == 1 else buys[0].tick if buys else None
        check(f"@{tr} round {r}: 4 buys + 2 team_buy", len(buys) == 4 and len(tbs) == 2)
        check(f"@{tr} round {r}: kind round, tick = freeze_end, pre = post = freeze_end",
              all(a.kind == "round" and a.tick == fe and a.state_pre_tick == a.state_post_tick == fe for a in buys + tbs))
acts = events(make_raw_tables("base"))
buys = {a.actor_id_hash: a for a in acts if a.type == "buy" and a.round_num == 1}
cls = {n: (buys[PID[n]].params["equip_value"], buys[PID[n]].params["loadout_class"]) for n in PLAYERS}
check("A 4900 full, B 4500 full (4500 = force edge -> full), C/D 3900 force",
      cls == {"A": (4900, "full"), "B": (4500, "full"), "C": (3900, "force"), "D": (3900, "force")}, str(cls))
pa = buys[PID["A"]].params
check("A.5 keys", set(pa) == A5_BUY)
check("A: money 650 -> 650, armor+helmet+kit, 2 grenades, rifle",
      (pa["money_before"], pa["money_after"], pa["has_armor"], pa["has_helmet"], pa["has_kit"], pa["utility_count"],
       pa["primary_class"]) == (650, 650, True, True, True, 2, "rifle"))
check("C has no kit", buys[PID["C"]].params["has_kit"] is False)
tbs = {a.actor_side: a for a in acts if a.type == "team_buy" and a.round_num == 1}
check("team_buy ct: full, 9400", tbs["ct"].params == {"team_class": "full", "total_equip": 9400})
check("team_buy t: force, 7800", tbs["t"].params == {"team_class": "force", "total_equip": 7800})
check("team_buy has no actor and distinct ids", tbs["ct"].actor_id_hash is None and tbs["ct"].action_id != tbs["t"].action_id)
check("all valid", all(a.validate() == [] for a in acts if a.kind == "round"))
db = {a.actor_id_hash: a for a in events(make_raw_tables("drop")) if a.type == "buy" and a.round_num == 1}
check("drop story: B without a primary is eco (1600), primary_class pistol",
      (db[PID["B"]].params["equip_value"], db[PID["B"]].params["loadout_class"], db[PID["B"]].params["primary_class"])
      == (1600, "eco", "pistol"))
check("tie on team class goes to the higher band", max(economy.BAND_ORDER, key=lambda c: (["eco", "full"].count(c),
      economy.BAND_ORDER.index(c))) == "full")

# ── drops ──
section("drop / give (drop story)")
for tr in (128, 64):
    dr = [a for a in events(make_raw_tables("drop", tr)) if a.type == "drop"]
    check(f"@{tr}: one drop A -> B", len(dr) == 1 and dr[0].actor_id_hash == PID["A"]
          and dr[0].params["receiver_id"] == PID["B"])
    check(f"@{tr}: at the drop tick, pickup tick in params, post state after the pickup",
          dr[0].tick == scale_tick(1100, tr) and dr[0].params["pickup_tick"] == scale_tick(1150, tr)
          and dr[0].state_post_tick == scale_tick(1150, tr) + 1 and dr[0].state_pre_tick == scale_tick(1100, tr) - 1)
    check(f"@{tr}: weapon_class rifle", dr[0].params["weapon_class"] == "rifle")
for story in STORIES:
    if story != "drop":
        check(f"{story}: no drop", not [a for a in events(make_raw_tables(story)) if a.type == "drop"])
check("drop ticks reach the builder (drop and pickup)",
      {1100, 1150} <= extractor.event_ticks(events(make_raw_tables("drop")))[1])

raw = make_raw_tables("drop")


def with_items(rows):
    it = pl.DataFrame(rows, schema=raw.items.schema)
    return [a for a in events(dataclasses.replace(raw, items=it)) if a.type == "drop"]


check("pickup after 5 s -> no give", not with_items([
    dict(tick=1100, steamid=SID["A"], event="drop", item="weapon_m4a1"),
    dict(tick=1100 + 5 * 128 + 1, steamid=SID["B"], event="pickup", item="weapon_m4a1")]))
check("pickup exactly 5 s later -> give", len(with_items([
    dict(tick=1100, steamid=SID["A"], event="drop", item="weapon_m4a1"),
    dict(tick=1100 + 5 * 128, steamid=SID["B"], event="pickup", item="weapon_m4a1")])) == 1)
check("picked up by an enemy -> no give", not with_items([
    dict(tick=1100, steamid=SID["A"], event="drop", item="weapon_m4a1"),
    dict(tick=1150, steamid=SID["C"], event="pickup", item="weapon_m4a1")]))
check("dropper picks it back up -> no give", not with_items([
    dict(tick=1100, steamid=SID["A"], event="drop", item="weapon_m4a1"),
    dict(tick=1150, steamid=SID["A"], event="pickup", item="weapon_m4a1")]))
check("death drop (C killed at 2000) picked up by D -> no give", not with_items([
    dict(tick=2000, steamid=SID["C"], event="drop", item="weapon_ak47"),
    dict(tick=2050, steamid=SID["D"], event="pickup", item="weapon_ak47")]))
check("c4 is bomb_drop's business", not with_items([
    dict(tick=1100, steamid=SID["D"], event="drop", item="weapon_c4"),
    dict(tick=1150, steamid=SID["C"], event="pickup", item="weapon_c4")]))
check("different item picked up -> no give", not with_items([
    dict(tick=1100, steamid=SID["A"], event="drop", item="weapon_m4a1"),
    dict(tick=1150, steamid=SID["B"], event="pickup", item="weapon_ak47")]))

sys.exit(summary())
