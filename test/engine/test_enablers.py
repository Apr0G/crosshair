"""
Enablers (P4.5.1, P4.5.2) on the synthetic stories through the real builder: flash
(flash story), smoke_cover (base), trade_setup with the dead teammate as beneficiary
(trade story), he_damage, fire_pressure (an added inferno on B's path), info (linked to
a hold, and team-level when no action can carry it), negatives, sorted entries and
determinism.

    .venv\\Scripts\\python.exe test\\engine\\test_enablers.py
"""
import dataclasses
import sys

import polars as pl
from _harness import check, section, summary
from _synthetic import PLAYERS, make_raw_tables

from engine import config
from engine.actions import extractor
from engine.state.geometry import FlatGeometry
from engine.state.players import player_id_hash

CFG = config.load()
GEO = FlatGeometry()
PID = {n: player_id_hash(s) for n, (s, _) in PLAYERS.items()}
SID = {n: s for n, (s, _) in PLAYERS.items()}


def run(raw):
    return extractor.run(raw, CFG, match_key="m1", geometry=GEO)


def kill_at(acts, tick, actor):
    return next(a for a in acts if a.type == "kill" and a.tick == tick and a.actor_id_hash == PID[actor])


def kinds(a):
    return [e["kind"] for e in a.enablers]


def by_id(acts):
    return {a.action_id: a for a in acts}


# ── flash ──
section("flash (flash story): kill at 2000 has a flash enabler")
_, acts, _ = run(make_raw_tables("flash"))
k = kill_at(acts, 2000, "A")
fl = [e for e in k.enablers if e["kind"] == "flash"]
flash = next(a for a in acts if a.type == "flash")
check("one flash enabler linked to the flash action", len(fl) == 1 and fl[0]["action_id"] == flash.action_id)
check("beneficiary = the thrower A", fl[0]["beneficiary_id"] == PID["A"])
check("entry keys {action_id, kind, beneficiary_id}", all(set(e) == {"action_id", "kind", "beneficiary_id"} for e in k.enablers))
check("B's kill at 4000 (D no longer blind): no flash enabler", "flash" not in kinds(kill_at(acts, 4000, "B")))
_, base, _ = run(make_raw_tables("base"))
check("base: kill at 2000 has no flash enabler", "flash" not in kinds(kill_at(base, 2000, "A")))

# ── smoke_cover / he_damage ──
section("smoke_cover and he_damage (base)")
k = kill_at(base, 2000, "A")
smoke = next(a for a in base if a.type == "smoke")
he = next(a for a in base if a.type == "he")
sc = [e for e in k.enablers if e["kind"] == "smoke_cover"]
check("kill at 2000: smoke_cover (A-D blocked) linked to A's smoke", [e["action_id"] for e in sc] == [smoke.action_id])
check("kill at 2000: he_damage linked to the HE (C hit 1.56 s before)",
      [e["action_id"] for e in k.enablers if e["kind"] == "he_damage"] == [he.action_id])
check("B's kill at 4000: no enablers at all (smoke faded, HE too old)", kill_at(base, 4000, "B").enablers == [])
pc = next(a for a in base if a.type == "plant_complete" and a.round_num == 1)
check("plant_complete: smoke_cover only (no victim)", kinds(pc) == ["smoke_cover"])
dmg = [a for a in base if a.type == "damage"]
check("HE damage actions at 1800 do not link the HE itself (not before tick-1)",
      all("he_damage" not in kinds(d) for d in dmg))
check("no action enables itself", all(e["action_id"] != a.action_id for a in base for e in a.enablers))

# ── trade_setup ──
section("trade_setup (trade story): B's kill at 2100 traded A")
_, trade, _ = run(make_raw_tables("trade"))
k = kill_at(trade, 2100, "B")
ts = [e for e in k.enablers if e["kind"] == "trade_setup"]
c_kill = kill_at(trade, 2000, "C")
check("one trade_setup linked to C's kill of A", len(ts) == 1 and ts[0]["action_id"] == c_kill.action_id)
check("beneficiary = the dead teammate A", ts[0]["beneficiary_id"] == PID["A"])
check("C's kill of A has no trade_setup", "trade_setup" not in kinds(c_kill))
check("base: no trade_setup anywhere", not [e for a in base for e in a.enablers if e["kind"] == "trade_setup"])
_, dk, _ = run(make_raw_tables("double_kill_same_tick"))
check("double kill: teammate dying the same tick is not 'before' -> no trade_setup",
      not [e for a in dk for e in a.enablers if e["kind"] == "trade_setup"])
check("double kill: A's kill of D 15 s after B died -> outside trade_window_s",
      "trade_setup" not in kinds(kill_at(dk, 4000, "A")))


def rerun_with(cfg):
    raw_t = make_raw_tables("trade")
    snaps, acts, _ = run(raw_t)
    for a in acts:
        a.enablers = []
    extractor.attach_enablers(acts, snaps, raw_t, cfg, ctx=extractor.make_context(raw_t, cfg, "m1", GEO))
    return "trade_setup" in kinds(kill_at(acts, 2100, "B"))


short = dataclasses.replace(CFG, actions=dataclasses.replace(CFG.actions, trade_window_s=0.5))
check("trade_window_s respected: 0.5 s < 0.78 s between the deaths -> no trade_setup", not rerun_with(short))
none = dataclasses.replace(CFG, actions=dataclasses.replace(CFG.actions, trade_max_units=-1.0))
check("trade_max_units respected: below every distance -> no trade_setup", not rerun_with(none))
check("defaults -> trade_setup", rerun_with(CFG))

# ── info ──
section("info")
inf = [e for e in k.enablers if e["kind"] == "info"]
hold_a = next(a for a in trade if a.type == "hold" and a.actor_id_hash == PID["A"])
check("trade: A spotted C before dying -> info linked to A's hold, beneficiary A",
      inf == [{"action_id": hold_a.action_id, "kind": "info", "beneficiary_id": PID["A"]}], str(inf))
snaps, acts3, _ = run(make_raw_tables("trade"))
no_holds = [a for a in acts3 if a.type != "hold"]
for a in no_holds:
    a.enablers = []
extractor.attach_enablers(no_holds, snaps, make_raw_tables("trade"), CFG,
                          ctx=extractor.make_context(make_raw_tables("trade"), CFG, "m1", GEO))
check("without a linkable hold/peek -> one team-level info entry (no action, no beneficiary)",
      [e for e in kill_at(no_holds, 2100, "B").enablers if e["kind"] == "info"]
      == [{"action_id": None, "kind": "info", "beneficiary_id": None}])
check("info never comes from the actor alone (base kill at 2000: B saw nothing)", "info" not in kinds(kill_at(base, 2000, "A")))

# ── fire_pressure ──
section("fire_pressure (double kill + D's fire on B's path at 1800)")
raw = make_raw_tables("double_kill_same_tick")
inf_rows = pl.DataFrame([dict(entity_id=301, start_tick=1800, end_tick=2000, x=1600.0, y=0.0, z=0.0,
                              extent_radius=150.0, thrower_steamid=SID["D"])], schema=raw.infernos.schema)
_, fa, _ = run(dataclasses.replace(raw, infernos=inf_rows))
mol = next(a for a in fa if a.type == "molotov")
kb = kill_at(fa, 2000, "C")
check("C's kill of B: fire_pressure linked to D's molotov, beneficiary D",
      [e for e in kb.enablers if e["kind"] == "fire_pressure"]
      == [{"action_id": mol.action_id, "kind": "fire_pressure", "beneficiary_id": PID["D"]}])
check("A's kill of C (C never in the fire): no fire_pressure", "fire_pressure" not in kinds(kill_at(fa, 2000, "A")))

# ── order and determinism ──
section("sorted by (kind, action_id), deterministic")
for acts in (base, trade, dk, fa):
    check("entries sorted", all(a.enablers == sorted(a.enablers, key=lambda e: (e["kind"], e["action_id"] or ""))
                                for a in acts))
_, again, _ = run(make_raw_tables("trade"))
check("two runs -> identical enablers", [a.enablers for a in again] == [a.enablers for a in trade])
check("only kill / damage / plant_complete / defuse_complete carry enablers",
      all(not a.enablers for a in base + trade if a.type not in ("kill", "damage", "plant_complete", "defuse_complete")))
check("every linked action exists", all(e["action_id"] is None or e["action_id"] in by_id(acts)
                                        for acts in (base, trade, dk, fa) for a in acts for e in a.enablers))

sys.exit(summary())
