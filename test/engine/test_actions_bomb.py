"""
Bomb events (P4.2.6): plant_start / plant_complete with site, callout and alive counts;
defuse_start / defuse_complete with kit; defuse_abort, bomb_drop, bomb_pickup from added
parser rows; explode and actor-less rows produce nothing; duplicates collapse.

    .venv\\Scripts\\python.exe test\\engine\\test_actions_bomb.py
"""
import dataclasses
import sys

import polars as pl
from _harness import check, section, summary
from _synthetic import PLAYERS, STORIES, STORY_FACTS, make_raw_tables, scale_tick

from engine import config
from engine.actions import extractor
from engine.state.geometry import FlatGeometry
from engine.state.players import player_id_hash

CFG = config.load()
GEO = FlatGeometry()
PID = {n: player_id_hash(s) for n, (s, _) in PLAYERS.items()}
SID = {n: s for n, (s, _) in PLAYERS.items()}
BOMB_TYPES = ("plant_start", "plant_complete", "defuse_start", "defuse_complete", "defuse_abort",
              "bomb_drop", "bomb_pickup")


def bomb_actions(raw):
    return [a for a in extractor.extract_event_actions(raw, CFG, "m1", geometry=GEO) if a.type in BOMB_TYPES]


section("bomb events per story match STORY_FACTS (explode has no action)")
for tr in (128, 64):
    for story in STORIES:
        acts = bomb_actions(make_raw_tables(story, tr))
        for r in (1, 2):
            want = [e for e in STORY_FACTS[story]["bomb_round"][r] if e != "explode"]
            got = [a.type for a in acts if a.round_num == r]
            check(f"{story}@{tr} round {r}: {want}", got == want, str(got))

section("plant params (base)")
acts = bomb_actions(make_raw_tables("base"))
ps, pc = [a for a in acts if a.round_num == 1]
check("plant_start 2900, plant_complete 3000 by D", (ps.tick, pc.tick) == (2900, 3000)
      and ps.actor_id_hash == pc.actor_id_hash == PID["D"] and pc.actor_side == "t")
check("site A, planter_callout", pc.params["site"] == "A" and pc.params["planter_callout"] == GEO.callout(3000, -300, 0))
check("alive counts at tick-1: 2 CT, 1 T (C died at 2000)", (pc.params["ct_alive"], pc.params["t_alive"]) == (2, 1))
check("A.5 keys", set(pc.params) == {"site", "planter_callout", "ct_alive", "t_alive"})

section("defuse params")
ds, dc = [a for a in acts if a.round_num == 2 and a.type.startswith("defuse")]
check("round 2: A defuses with a kit", ds.actor_id_hash == dc.actor_id_hash == PID["A"]
      and ds.params["kit"] is True and dc.params["kit"] is True)
check("A.5 keys", set(dc.params) == {"kit", "ct_alive", "t_alive"})
rt = [a for a in bomb_actions(make_raw_tables("retreat")) if a.round_num == 1 and a.type == "defuse_complete"]
check("retreat: B's defuse completes at 4200", len(rt) == 1 and rt[0].tick == 4200 and rt[0].actor_id_hash == PID["B"])

section("defuse_abort / bomb_drop / bomb_pickup from parser rows")
raw = make_raw_tables("base")
rows = pl.DataFrame([
    dict(tick=2500, round_num=1, event="drop", steamid=SID["D"], site=None, has_kit=None, x=3000.0, y=-300.0, z=0.0),
    dict(tick=2600, round_num=1, event="pickup", steamid=SID["D"], site=None, has_kit=None, x=3000.0, y=-300.0, z=0.0),
    dict(tick=3500, round_num=1, event="defuse_start", steamid=SID["B"], site="A", has_kit=None, x=None, y=None, z=None),
    dict(tick=3600, round_num=1, event="defuse_abort", steamid=SID["B"], site="A", has_kit=None, x=None, y=None, z=None),
    dict(tick=3600, round_num=1, event="defuse_abort", steamid=SID["B"], site="A", has_kit=None, x=None, y=None, z=None),
    dict(tick=3700, round_num=1, event="defuse_start", steamid=None, site="A", has_kit=None, x=None, y=None, z=None),
], schema=raw.bomb.schema)
acts = bomb_actions(dataclasses.replace(raw, bomb=pl.concat([raw.bomb, rows]).sort("tick")))
by = {(a.type, a.tick): a for a in acts}
check("bomb_drop and bomb_pickup with callout", ("bomb_drop", 2500) in by and ("bomb_pickup", 2600) in by
      and by[("bomb_drop", 2500)].params == {"callout": GEO.callout(3000, -300, 0)})
check("defuse_abort straight from the parser event", ("defuse_abort", 3600) in by)
check("duplicate parser row collapses to one action", sum(a.type == "defuse_abort" for a in acts) == 1)
check("row without a steamid -> no action", ("defuse_start", 3700) not in by)
check("kit falls back to the defuser's has_defuser", by[("defuse_start", 3500)].params["kit"] is True)
check("every bomb action validates", all(a.validate() == [] for a in acts))
check("64 tick: ticks scale", [a.tick for a in bomb_actions(make_raw_tables("base", 64)) if a.round_num == 1]
      == [scale_tick(2900, 64), scale_tick(3000, 64)])

sys.exit(summary())
