"""
Simultaneity windows (P4.6.1, P4.6.2): same tick -> one window, 3 ticks apart -> two,
a smoke pop on a kill tick -> one window [kill, smoke_pop], single-linkage chaining,
round clipping, non-overlap and full coverage of every round on every story.

    .venv\\Scripts\\python.exe test\\engine\\test_windows.py
"""
import dataclasses
import sys

from _harness import check, section, summary
from _synthetic import PLAYERS, STORIES, make_raw_tables, scale_tick

from engine import config
from engine.actions import extractor, windows
from engine.actions.types import Action
from engine.state.geometry import FlatGeometry
from engine.state.players import player_id_hash

CFG = config.load()
GEO = FlatGeometry()
PID = {n: player_id_hash(s) for n, (s, _) in PLAYERS.items()}
SIM = CFG.actions.simultaneity_ticks


def inst(type_, tick, actor=1, **params):
    return Action(match_key="m1", round_num=1, actor_id_hash=actor, actor_side="ct", kind="instant",
                  type=type_, tick=tick, params=params)


# ── unit ──
section("grouping rules")
k1, k2 = inst("kill", 2000, 1, victim_id=9), inst("kill", 2000, 2, victim_id=8)
ws = windows.group([k1, k2], CFG, 128)
check("two kills on one tick -> one window, two members", len(ws) == 1 and set(ws[0].member_action_ids)
      == {k1.action_id, k2.action_id})
check("bounds [min - 1, max + 1]", (ws[0].start_tick, ws[0].end_tick) == (1999, 2001))
k3 = inst("kill", 2003, 2, victim_id=8)
ws = windows.group([k1, k3], CFG, 128)
check("kills 3 ticks apart -> two windows", len(ws) == 2)
check("... which share no tick", ws[0].end_tick < ws[1].start_tick and not windows.overlaps(ws))
ws = windows.group([k1, inst("kill", 2000 + SIM, 2)], CFG, 128)
check(f"{SIM} ticks apart (= simultaneity_ticks) -> one window", len(ws) == 1 and (ws[0].start_tick, ws[0].end_tick)
      == (1999, 2001 + SIM))
pop = inst("smoke_pop", 2000, 3, parent_action_id="p" * 16)
ws = windows.group([k1, pop], CFG, 128)
check("smoke pop on the kill tick -> one window [kill, smoke_pop]", len(ws) == 1
      and set(ws[0].member_action_ids) == {k1.action_id, pop.action_id})
give = inst("drop", 900, 4, receiver_id=7, pickup_tick=1300)
give.state_post_tick = 1301
ws = windows.group([give, inst("kill", 1200, 1), inst("kill", 1900, 2)], CFG, 128)
check("a drop's window reaches its pickup and absorbs a kill inside the span (MED-15)",
      len(ws) == 2 and (ws[0].start_tick, ws[0].end_tick) == (899, 1301) and len(ws[0].member_action_ids) == 2
      and (ws[1].start_tick, ws[1].end_tick) == (1899, 1901), str([(w.start_tick, w.end_tick) for w in ws]))
chain = [inst("damage", t, 5) for t in (100, 102, 104, 106)]
for i, a in enumerate(chain):
    a.window_start_tick = a.window_end_tick = a.tick
    a.action_id = f"{i:016x}"
ws = windows.group(chain, CFG, 128)
check("single-linkage: 100,102,104,106 chain into one window", len(ws) == 1 and (ws[0].start_tick, ws[0].end_tick)
      == (99, 107))
smoke = Action(match_key="m1", round_num=1, actor_id_hash=3, actor_side="ct", kind="durative", type="smoke",
               window_start_tick=2000, window_end_tick=4000)
buy = Action(match_key="m1", round_num=1, actor_id_hash=3, actor_side="ct", kind="round", type="buy", tick=1000)
check("durative and round actions are never members", windows.group([smoke, buy], CFG, 128) == [])
ws = windows.group([inst("drop", 900), inst("kill", 1000), inst("kill", 5000)], CFG, 128, rounds={1: (1000, 4360)})
check("rounds: actions outside [freeze_end, terminal] skipped, bounds clipped",
      len(ws) == 1 and (ws[0].start_tick, ws[0].end_tick) == (1000, 1001))
check("deterministic window_id", windows.group([k1, k2], CFG, 128)[0].window_id
      == windows.group([k2, k1], CFG, 128)[0].window_id)
try:
    windows.group([k1], CFG, 0)
    check("tick_rate 0 rejected", False)
except ValueError:
    check("tick_rate 0 rejected", True)

section("bounds follow sampling.boundary_offset_ticks (LOW-24)")
CFG2 = dataclasses.replace(CFG, sampling=dataclasses.replace(CFG.sampling, boundary_offset_ticks=2))
ws = windows.group([k1, k2], CFG2, 128)
check("offset 2: bounds [min - 2, max + 2]", len(ws) == 1 and (ws[0].start_tick, ws[0].end_tick) == (1998, 2002),
      str([(w.start_tick, w.end_tick) for w in ws]))
ws = windows.group([k1, k3], CFG2, 128)
check("offset 2: kills 3 ticks apart would share ticks, so they form one window [1998, 2005]",
      len(ws) == 1 and (ws[0].start_tick, ws[0].end_tick) == (1998, 2005) and not windows.overlaps(ws),
      str([(w.start_tick, w.end_tick) for w in ws]))
ws = windows.group([k1, inst("kill", 2005, 2)], CFG2, 128)
check("offset 2: kills 5 ticks apart -> two windows sharing no tick",
      len(ws) == 2 and ws[0].end_tick < ws[1].start_tick and not windows.overlaps(ws))
ws = windows.group([give, inst("kill", 1302, 1)], CFG, 128)
check("a kill right after a drop's span joins it instead of sharing the span's end tick",
      len(ws) == 1 and (ws[0].start_tick, ws[0].end_tick) == (899, 1303) and not windows.overlaps(ws),
      str([(w.start_tick, w.end_tick) for w in ws]))
for story in ("double_kill_same_tick", STORIES[-1]):
    snaps2, acts2, ws2 = extractor.run(make_raw_tables(story, 128), CFG2, match_key="m1", geometry=GEO)
    snap_ticks2 = {(s.meta.round_num, s.meta.tick) for s in snaps2}
    check(f"offset 2, {story}: every window has its pre and post sample, no overlaps",
          bool(ws2) and not windows.overlaps(ws2)
          and all((w.round_num, w.start_tick) in snap_ticks2 and (w.round_num, w.end_tick) in snap_ticks2 for w in ws2))

section("coverage")
row = {"round_num": 1, "freeze_end_tick": 1000, "end_tick": 4040, "official_end_tick": 4360}
ws = windows.group([k1, k3, inst("kill", 3000)], CFG, 128, rounds={1: (1000, 4360)})
cov = windows.coverage(ws, row)
check("starts at freeze_end, ends at terminal", cov[0][1] == 1000 and cov[-1][2] == 4360)
check("consecutive intervals share end points (no gap, no overlap)", all(a[2] == b[1] for a, b in zip(cov, cov[1:])))
check("every window is exactly one event interval", [c[3] for c in cov if c[0] == "event"] == [w.window_id for w in ws])
check("drift and event alternate", [c[0] for c in cov] == ["drift", "event", "drift", "event", "drift", "event", "drift"])
bad = [windows.Window("a" * 16, 1, 1999, 2005, ("x",)), windows.Window("b" * 16, 1, 2003, 2010, ("y",))]
check("overlaps() finds overlapping windows", windows.overlaps(bad) == [("a" * 16, "b" * 16)])
try:
    windows.coverage(bad, row)
    check("coverage rejects overlapping windows", False)
except ValueError:
    check("coverage rejects overlapping windows", True)

# ── stories ──
section("double_kill_same_tick: one window with two kill members")
for tr in (128, 64):
    snaps, acts, ws = extractor.run(make_raw_tables("double_kill_same_tick", tr), CFG, match_key="m1", geometry=GEO)
    t = scale_tick(2000, tr)
    kills = {a.action_id for a in acts if a.type == "kill" and a.tick == t}
    w = [w for w in ws if w.start_tick <= t <= w.end_tick]
    check(f"@{tr}: one window holds both kills at {t}", len(w) == 1 and kills <= set(w[0].member_action_ids)
          and len(kills) == 2)

section("every story: ordered, non-overlapping, full coverage, every instant action in exactly one window")
for tr in (128, 64):
    for story in STORIES:
        raw = make_raw_tables(story, tr)
        snaps, acts, ws = extractor.run(raw, CFG, match_key="m1", geometry=GEO)
        rows = {r["round_num"]: r for r in raw.rounds.to_dicts()}
        ok_cov = True
        for r, rr in rows.items():
            try:
                cov = windows.coverage([w for w in ws if w.round_num == r], rr)
                ok_cov &= all(a[2] == b[1] for a, b in zip(cov, cov[1:]))
            except ValueError:
                ok_cov = False
        check(f"{story}@{tr}: ordered", ws == sorted(ws, key=lambda w: (w.round_num, w.start_tick)))
        check(f"{story}@{tr}: no overlaps", not windows.overlaps(ws))
        check(f"{story}@{tr}: coverage tiles every round", ok_cov)
        members = [m for w in ws for m in w.member_action_ids]
        inst_ids = [a.action_id for a in acts if a.kind == "instant"]
        check(f"{story}@{tr}: each instant action in exactly one window",
              sorted(members) == sorted(inst_ids) and len(set(members)) == len(members))
        snap_ticks = {(s.meta.round_num, s.meta.tick) for s in snaps}
        check(f"{story}@{tr}: every window has its pre and post sample",
              all((w.round_num, w.start_tick) in snap_ticks and (w.round_num, w.end_tick) in snap_ticks for w in ws))

sys.exit(summary())
