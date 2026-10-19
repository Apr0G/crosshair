"""
Utility actions (P4.2.3, P4.2.4, P4.3.1, P4.3.2, P4.3.6): flash blinded lists, HE damaged
lists, smoke window + pop/fade pseudo-actions + blocks_pairs_at_pop (real builder,
FlatGeometry), nominal durations when the end is missing, windows scale with the tick
rate, molotov (enemies_damaged / enemies_displaced) and decoy on hand-added rows.

    .venv\\Scripts\\python.exe test\\engine\\test_actions_utility.py
"""
import dataclasses
import sys

import polars as pl
from _harness import check, section, summary
from _synthetic import PLAYERS, make_raw_tables, scale_tick

from engine import config
from engine.actions import extractor, utility
from engine.state.geometry import FlatGeometry
from engine.state.players import player_id_hash
from engine.state.utility import NOMINAL_DURATION_S

CFG = config.load()
GEO = FlatGeometry()
PID = {n: player_id_hash(s) for n, (s, _) in PLAYERS.items()}
SID = {n: s for n, (s, _) in PLAYERS.items()}


def of(acts, type_):
    return [a for a in acts if a.type == type_]


def events(raw):
    return extractor.extract_event_actions(raw, CFG, "m1", geometry=GEO)


# ── flash ──
section("flash detonation (flash story)")
fa = of(events(make_raw_tables("flash")), "flash")
check("one flash action", len(fa) == 1)
f = fa[0]
check("thrower A at 1900", f.actor_id_hash == PID["A"] and f.tick == 1900)
want = sorted([{"id": PID["D"], "duration": 2.5}, {"id": PID["C"], "duration": 1.0}], key=lambda e: e["id"])
check("blinded_enemies D 2.5 s, C 1.0 s", f.params["blinded_enemies"] == want, str(f.params["blinded_enemies"]))
check("no teammates blinded, self 0.0", f.params["blinded_teammates"] == [] and f.params["blinded_self"] == 0.0)
check("pop_delay_s = 50 ticks", f.params["pop_delay_s"] == round(50 / 128, 4))
check("callouts", f.params["land_callout"] == GEO.callout(1400, 0, 0) and f.params["throw_callout"] == GEO.callout(0, 300, 0))
check("A.5 keys present", {"blinded_enemies", "blinded_teammates", "blinded_self", "land_callout",
                            "throw_callout", "pop_delay_s"} <= set(f.params))
check("valid", f.validate() == [])
check("base story: no flash action", not of(events(make_raw_tables("base")), "flash"))

# ── HE ──
section("HE detonation (base)")
he = of(events(make_raw_tables("base")), "he")
check("one HE action by A at 1800", len(he) == 1 and he[0].actor_id_hash == PID["A"] and he[0].tick == 1800)
h = he[0].params
want = sorted([{"id": PID["C"], "dmg": 30, "armor": 5}, {"id": PID["D"], "dmg": 20, "armor": 5}], key=lambda e: e["id"])
check("enemies_damaged C 30, D 20", h["enemies_damaged"] == want, str(h["enemies_damaged"]))
check("teammates_damaged empty, damage_total 50", h["teammates_damaged"] == [] and h["damage_total"] == 50)
check("land_callout", h["land_callout"] == GEO.callout(1500, 0, 0))

# ── smoke ──
section("smoke window and pseudo-actions (base, real builder)")
for tr in (128, 64):
    raw = make_raw_tables("base", tr)
    snaps, acts, wins = extractor.run(raw, CFG, match_key="m1", geometry=GEO)
    sm, pop, fade = of(acts, "smoke"), of(acts, "smoke_pop"), of(acts, "smoke_fade")
    check(f"@{tr}: one smoke, one pop, one fade", (len(sm), len(pop), len(fade)) == (1, 1, 1))
    s = sm[0]
    t0, t1 = scale_tick(1500, tr), scale_tick(1500, tr) + 18 * tr
    check(f"@{tr}: window [pop, pop + 18 s] scales with tick rate", (s.window_start_tick, s.window_end_tick) == (t0, t1),
          f"{s.window_start_tick}, {s.window_end_tick}")
    check(f"@{tr}: kind durative, tick null", s.kind == "durative" and s.tick is None)
    check(f"@{tr}: pop/fade at the window ends with parent_action_id",
          pop[0].tick == t0 and fade[0].tick == t1 and pop[0].params["parent_action_id"] == s.action_id
          and fade[0].params["parent_action_id"] == s.action_id)
    check(f"@{tr}: pseudo-actions belong to the thrower", pop[0].actor_id_hash == fade[0].actor_id_hash == PID["A"])
    check(f"@{tr}: params", (s.params["throw_tick"], s.params["pop_tick"], s.params["expire_tick"])
          == (scale_tick(1400, tr), t0, t1) and s.params["land_callout"] == GEO.callout(1000, 0, 0))
    # A-D (line passes ~98 u from the centre) and B (inside the cloud) to C and D: 3 pairs
    check(f"@{tr}: blocks_pairs_at_pop = 3 (A-D, B-C, B-D)", s.params["blocks_pairs_at_pop"] == 3,
          str(s.params["blocks_pairs_at_pop"]))
    check(f"@{tr}: pop and fade are window members",
          {pop[0].action_id, fade[0].action_id} <= {m for w in wins for m in w.member_action_ids})
ev = of(events(make_raw_tables("base")), "smoke")
check("event stage (no snapshots): blocks_pairs_at_pop is None", ev[0].params["blocks_pairs_at_pop"] is None)

section("nominal durations when the end tick is null")
for tr in (128, 64):
    raw = make_raw_tables("base", tr)
    raw = dataclasses.replace(raw, smokes=raw.smokes.with_columns(pl.lit(None, dtype=pl.Int64).alias("end_tick")))
    s = of(events(raw), "smoke")[0]
    check(f"@{tr}: smoke end = pop + {NOMINAL_DURATION_S['smoke']} s",
          s.window_end_tick == scale_tick(1500, tr) + int(NOMINAL_DURATION_S["smoke"] * tr))
    check(f"@{tr}: nominal fade emitted", len(of(events(raw), "smoke_fade")) == 1)
raw = make_raw_tables("base")
late = raw.smokes.with_columns(pl.lit(4300, dtype=pl.Int64).alias("start_tick"), pl.lit(None, dtype=pl.Int64).alias("end_tick"))
acts = events(dataclasses.replace(raw, smokes=late))
s = of(acts, "smoke")[0]
check("smoke outliving the round: window clipped to terminal, no fade",
      s.window_end_tick == 4360 and not of(acts, "smoke_fade") and len(of(acts, "smoke_pop")) == 1)

# ── molotov and decoy on added rows ──
section("molotov (added inferno: C's fire on B's path at 1300)")
raw = make_raw_tables("base")
g = pl.DataFrame([dict(entity_id=201, thrower_steamid=SID["C"], type="molotov", throw_tick=1200,
                       detonate_tick=1300, expire_tick=None, x=600.0, y=0.0, z=0.0),
                  dict(entity_id=202, thrower_steamid=SID["D"], type="decoy", throw_tick=1250,
                       detonate_tick=1350, expire_tick=None, x=2500.0, y=0.0, z=0.0)], schema=raw.grenades.schema)
inf = pl.DataFrame([dict(entity_id=201, start_tick=1300, end_tick=1300 + 7 * 128, x=600.0, y=0.0, z=0.0,
                         extent_radius=150.0, thrower_steamid=SID["C"])], schema=raw.infernos.schema)
fire = pl.DataFrame([dict(tick=1320, round_num=1, attacker_steamid=SID["C"], victim_steamid=SID["B"],
                          weapon="inferno", dmg_health=8, dmg_armor=0, hitgroup="generic"),
                     dict(tick=1340, round_num=1, attacker_steamid=SID["C"], victim_steamid=SID["B"],
                          weapon="inferno", dmg_health=6, dmg_armor=0, hitgroup="generic")], schema=raw.damages.schema)
raw_m = dataclasses.replace(raw, grenades=pl.concat([raw.grenades, g]), infernos=inf,
                            damages=pl.concat([raw.damages, fire]).sort("tick"))
acts = events(raw_m)
mo = of(acts, "molotov")
check("one molotov action", len(mo) == 1)
m = mo[0]
check("thrower C, T side, subtype molotov", m.actor_id_hash == PID["C"] and m.actor_side == "t" and m.subtype == "molotov")
check("window [1300, 1300 + 7 s]", (m.window_start_tick, m.window_end_tick) == (1300, 1300 + 7 * 128))
check("enemies_damaged B 14", m.params["enemies_damaged"] == [{"id": PID["B"], "dmg": 14}], str(m.params["enemies_damaged"]))
check("enemies_displaced = [B] (inside at start, out within 2 s)", m.params["enemies_displaced"] == [PID["B"]],
      str(m.params["enemies_displaced"]))
check("fire_start / fire_end with parent", [a.tick for a in of(acts, "fire_start")] == [1300]
      and [a.tick for a in of(acts, "fire_end")] == [1300 + 7 * 128]
      and of(acts, "fire_start")[0].params["parent_action_id"] == m.action_id)
check("land_callout", m.params["land_callout"] == GEO.callout(600, 0, 0))
far = inf.with_columns(pl.lit(-2000.0, dtype=pl.Float32).alias("x"))
m2 = of(events(dataclasses.replace(raw_m, infernos=far)), "molotov")[0]
check("nobody inside at start -> enemies_displaced []", m2.params["enemies_displaced"] == [])
nor = inf.with_columns(pl.lit(None, dtype=pl.Float32).alias("extent_radius"), pl.lit(None, dtype=pl.Int64).alias("end_tick"))
m3 = of(events(dataclasses.replace(raw_m, infernos=nor)), "molotov")[0]
check("unknown radius -> enemies_displaced None", m3.params["enemies_displaced"] is None)
check(f"null end -> nominal {NOMINAL_DURATION_S['fire']} s", m3.window_end_tick == 1300 + int(NOMINAL_DURATION_S["fire"] * 128))
check("base story: no molotov", not of(events(raw), "molotov"))

section("decoy")
de = of(acts, "decoy")
check("one decoy by D", len(de) == 1 and de[0].actor_id_hash == PID["D"])
check(f"null expire -> nominal {NOMINAL_DURATION_S['decoy']} s window",
      (de[0].window_start_tick, de[0].window_end_tick) == (1350, 1350 + int(NOMINAL_DURATION_S["decoy"] * 128)))
check("decoy_start / decoy_end", [a.tick for a in of(acts, "decoy_start")] == [1350]
      and [a.tick for a in of(acts, "decoy_end")] == [1350 + 15 * 128])
check("land_callout", de[0].params["land_callout"] == GEO.callout(2500, 0, 0))
check("every utility action validates", all(a.validate() == [] for a in acts))

section("smoke_blocks_segment helper")
s = of(events(make_raw_tables("base")), "smoke")[0]
check("A-D eye segment cut", utility.smoke_blocks_segment(s, (0, 300, 64), (3000, -300, 64)))
check("A-C eye segment clear", not utility.smoke_blocks_segment(s, (0, 300, 64), (3000, 300, 64)))

sys.exit(summary())
