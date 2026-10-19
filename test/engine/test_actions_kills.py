"""
Kill, team_kill and damage actions (P4.2.1, P4.2.2) on the synthetic stories through the
real state builder with FlatGeometry: exact kills per story, A.5 params, state ticks,
team kill / suicide handling, damage engagements and the damage_gap_s split, 64 tick.

    .venv\\Scripts\\python.exe test\\engine\\test_actions_kills.py
"""
import dataclasses
import sys

import polars as pl
from _harness import check, section, summary
from _synthetic import PLAYERS, STORIES, STORY_FACTS, make_raw_tables, scale_tick

from engine import config
from engine.actions import extractor, kills
from engine.actions.types import make_action_id
from engine.state.geometry import FlatGeometry
from engine.state.players import player_id_hash

CFG = config.load()
GEO = FlatGeometry()
PID = {n: player_id_hash(s) for n, (s, _) in PLAYERS.items()}
A5_KILL = {"weapon", "weapon_class", "headshot", "through_smoke", "attacker_blind", "victim_blind",
           "wallbang", "noscope", "distance_units", "attacker_callout", "victim_callout",
           "victim_health_before", "assister_id", "flash_assister_id", "victim_id"}
A5_DAMAGE = {"damage_total", "hits", "weapon_class", "armor_damage", "victim_id"}


def event_actions(raw):
    return extractor.extract_event_actions(raw, CFG, "m1", geometry=GEO)


def of(acts, type_):
    return [a for a in acts if a.type == type_]


# ── kills per story ──
section("one kill action per kill row, every story, 64 and 128 tick")
for tr in (64, 128):
    for story in STORIES:
        raw = make_raw_tables(story, tr)
        ks = of(event_actions(raw), "kill")
        want = sorted((scale_tick(t, tr), PID[a], PID[v]) for t, a, v in STORY_FACTS[story]["kill_events"])
        got = sorted((k.tick, k.actor_id_hash, k.params["victim_id"]) for k in ks)
        check(f"{story}@{tr}: kills = STORY_FACTS", got == want, f"{got} vs {want}")
        check(f"{story}@{tr}: state_pre = tick-1, state_post = tick+1",
              all(k.state_pre_tick == k.tick - 1 and k.state_post_tick == k.tick + 1 for k in ks))
        check(f"{story}@{tr}: no team kills", not of(event_actions(raw), "team_kill"))

# ── params ──
section("kill params (base)")
raw = make_raw_tables("base")
ks = {k.tick: k for k in of(event_actions(raw), "kill")}
k1, k2 = ks[2000], ks[4000]
check("A.5 keys exactly", set(k1.params) == A5_KILL, str(set(k1.params) ^ A5_KILL))
check("actor A, side ct", k1.actor_id_hash == PID["A"] and k1.actor_side == "ct")
check("weapon ak47 / rifle / headshot", (k1.params["weapon"], k1.params["weapon_class"], k1.params["headshot"])
      == ("ak47", "rifle", True))
check("victim C", k1.params["victim_id"] == PID["C"])
check("victim_health_before = 70 (HE took 30 at 1800)", k1.params["victim_health_before"] == 70)
check("D had 80 before B's kill", k2.params["victim_health_before"] == 80)
check("wallbang False (penetrated 0)", k1.params["wallbang"] is False)
check("distance 3000", k1.params["distance_units"] == 3000.0)
check("callouts from geometry", k1.params["attacker_callout"] == GEO.callout(0, 300, 0)
      and k1.params["victim_callout"] == GEO.callout(3000, 300, 0))
check("no assister -> None (not 0)", k1.params["assister_id"] is None and k1.params["flash_assister_id"] is None)
check("default id = sha1(match|round|kill|actor|tick)", k1.action_id == make_action_id("m1", 1, "kill", PID["A"], 2000))
check("JSON-safe and valid", k1.validate() == [] and k2.validate() == [], str(k1.validate()))
check("flash story: victim_blind True", of(event_actions(make_raw_tables("flash")), "kill")[0].params["victim_blind"] is True)
nogeo = of(extractor.extract_event_actions(raw, CFG, "m1"), "kill")[0]
check("without geometry callouts are None", nogeo.params["attacker_callout"] is None)

# ── team kill / suicide ──
section("team_kill and suicide")
k = raw.kills.with_columns(pl.when(pl.col("tick") == 4000).then(True).otherwise(pl.col("team_kill")).alias("team_kill"))
tk_acts = event_actions(dataclasses.replace(raw, kills=k))
check("team_kill flag -> type team_kill", len(of(tk_acts, "team_kill")) == 1 and len(of(tk_acts, "kill")) == 1)
check("team_kill keeps the kill params", set(of(tk_acts, "team_kill")[0].params) == A5_KILL)
k = raw.kills.with_columns(pl.when(pl.col("tick") == 4000).then(None).otherwise(pl.col("team_kill")).alias("team_kill"))
check("null flag, enemies -> kill", len(of(event_actions(dataclasses.replace(raw, kills=k)), "kill")) == 2)
sui = raw.kills.filter(pl.col("tick") == 2000).with_columns(
    pl.col("attacker_steamid").alias("victim_steamid"), pl.lit(2500, dtype=pl.Int64).alias("tick"))
acts = event_actions(dataclasses.replace(raw, kills=pl.concat([raw.kills, sui]).sort("tick")))
check("suicide row skipped", len(of(acts, "kill")) == 2 and not of(acts, "team_kill"))
world = sui.with_columns(pl.lit(None, dtype=pl.UInt64).alias("attacker_steamid"))
acts = event_actions(dataclasses.replace(raw, kills=pl.concat([raw.kills, world]).sort("tick")))
check("world kill (no attacker) skipped", len(of(acts, "kill")) == 2)

# ── damage ──
section("damage engagements (P4.2.2)")
dm = of(event_actions(raw), "damage")
check("HE at 1800 -> two damage actions for A", sorted((d.tick, d.actor_id_hash, d.params["victim_id"]) for d in dm)
      == sorted([(1800, PID["A"], PID["C"]), (1800, PID["A"], PID["D"])]))
check("lethal hits are not damage actions", all(d.tick != 2000 and d.tick != 4000 for d in dm))
check("A.5 keys", all(set(d.params) == A5_DAMAGE for d in dm))
dC = next(d for d in dm if d.params["victim_id"] == PID["C"])
check("C: 30 health, 5 armor, 1 hit, grenade", (dC.params["damage_total"], dC.params["armor_damage"],
      dC.params["hits"], dC.params["weapon_class"]) == (30, 5, 1, "grenade"))
check("window metadata [tick, last hit]", dC.window_start_tick == 1800 and dC.window_end_tick == 1800)
check("ids unique (salted by victim)", len({d.action_id for d in dm}) == 2)
check("valid", all(d.validate() == [] for d in dm))

gap = int(CFG.actions.damage_gap_s * 128)
t1, t2 = 1800 + gap - 1, 1800 + 2 * gap - 2     # each gap one tick short of damage_gap_s
t3 = t2 + gap                                   # exactly damage_gap_s later: a new engagement
extra = pl.DataFrame([
    dict(tick=t1, round_num=1, attacker_steamid=PLAYERS["A"][0], victim_steamid=PLAYERS["D"][0],
         weapon="ak47", dmg_health=10, dmg_armor=2, hitgroup="chest"),
    dict(tick=t2, round_num=1, attacker_steamid=PLAYERS["A"][0], victim_steamid=PLAYERS["D"][0],
         weapon="ak47", dmg_health=12, dmg_armor=3, hitgroup="chest"),
    dict(tick=t3, round_num=1, attacker_steamid=PLAYERS["A"][0], victim_steamid=PLAYERS["D"][0],
         weapon="ak47", dmg_health=7, dmg_armor=1, hitgroup="arm"),
], schema=raw.damages.schema)
raw2 = dataclasses.replace(raw, damages=pl.concat([raw.damages, extra]).sort("tick"))
dD = sorted((d for d in kills.extract_damage(raw2, None, extractor.make_context(raw2, CFG, "m1", GEO))
             if d.params["victim_id"] == PID["D"]), key=lambda d: d.tick)
check("hits < gap apart chain into one engagement, gap >= damage_gap_s splits",
      [(d.tick, d.window_end_tick, d.params["hits"]) for d in dD]
      == [(1800, t2, 3), (t3, t3, 1)],
      str([(d.tick, d.window_end_tick, d.params["hits"]) for d in dD]))
check("engagement totals summed", dD[0].params["damage_total"] == 42 and dD[0].params["armor_damage"] == 10)
check("most frequent weapon class wins", dD[0].params["weapon_class"] == "rifle")
team = extra.head(1).with_columns(pl.lit(PLAYERS["B"][0], dtype=pl.UInt64).alias("victim_steamid"))
raw3 = dataclasses.replace(raw, damages=pl.concat([raw.damages, team]).sort("tick"))
td = [d for d in of(event_actions(raw3), "damage") if d.params["victim_id"] == PID["B"]]
check("team damage kept with subtype 'team'", len(td) == 1 and td[0].subtype == "team")
check("enemy damage has no subtype", all(d.subtype is None for d in dm))

section("64 tick")
dm64 = of(event_actions(make_raw_tables("base", 64)), "damage")
check("same damage actions at 64 tick, ticks scaled", sorted(d.tick for d in dm64) == [900, 900])

sys.exit(summary())
