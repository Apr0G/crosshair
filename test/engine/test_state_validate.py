"""
The P7.1 consistency checker `Snapshot.validate()` catches each class of inconsistency an
intervention could introduce (alive count, equip sum, pair symmetry, dead with
equipment, smoke counts, LOS with a dead player, bomb fields, same_team, alive with null
flags), and `recompute_derived` repairs the derived ones. Run on the `_cf_states`
stories so the checker is exercised on the same states the interventions edit.

    .venv\\Scripts\\python.exe test\\engine\\test_state_validate.py
"""
import sys

import numpy as np

from _harness import check, section, summary
from _cf_states import make_kill_pair, make_plant_pair, make_smoke_pair, make_snapshot

from engine.counterfactual.interventions import recompute_derived
from engine.state.schema import PAIR_CHANNELS

LOS, DIST, SMOKE, SAME = (PAIR_CHANNELS.index(c) for c in ("los", "dist_units", "smoke_between", "same_team"))


def caught(name: str, s, needle: str) -> None:
    probs = s.validate()
    check(f"caught: {name}", any(needle in p for p in probs), "; ".join(probs[:2]) or "no problems reported")


def pl(s, pid):
    return s.players[s.index_of(pid)]


# ── baseline ──────────────────────────────────────────────────────────────────
section("baseline states are clean")

base = make_snapshot(4, 3, smokes=[(750, 200, 0, 103)], blind={203: 1.0})
stories = [base, *make_kill_pair()[:2], *make_smoke_pair()[:2], *make_plant_pair()[:2]]
check("every _cf_states story validates", all(not s.validate() for s in stories))

# ── each inconsistency class ──────────────────────────────────────────────────
section("inconsistencies an intervention could introduce")

s = base.copy()
pl(s, 104).alive = 0                         # killed without fixing the rest
caught("alive bit flipped, ctx.alive_ct stale", s, "ctx.alive_ct")
caught("alive bit flipped, player keeps equipment", s, "dead but has")

s = base.copy()
s.ctx.alive_t += 1
caught("alive count off by one", s, "ctx.alive_t")

s = base.copy()
pl(s, 101).equipment_value += 500.0
caught("equip sum stale after a loadout edit", s, "ctx.equip_ct")

s = base.copy()
s.pair[0, 6, DIST] += 10.0
caught("pair not symmetric", s, "not symmetric")

s = base.copy()
d = pl(s, 105)                               # 105 is dead in a 4v3
d.has_helmet, d.weapon_class, d.health = 1, "rifle", 50.0
caught("dead player with equipment/health", s, "dead but has")

s = base.copy()
s.ctx.n_active_smokes = 0
caught("smoke count disagrees with tokens", s, "n_active_smokes")

s = base.copy()
s.utility.pop()
caught("token removed, counts stale", s, "n_active_smokes")

s = base.copy()
i, j = s.index_of(101), s.index_of(205)      # 205 is dead
s.pair[i, j, LOS] = s.pair[j, i, LOS] = 1.0
caught("LOS set for a pair with a dead player", s, "pair.los")

s = base.copy()
s.pair[0, 1, SAME] = s.pair[1, 0, SAME] = 0.0
caught("same_team disagrees with sides", s, "same_team")

s = base.copy()
s.pair[0, 6, SMOKE] = s.pair[6, 0, SMOKE] = 0.5
caught("smoke_between not 0/1", s, "smoke_between")

s = base.copy()
s.ctx.bomb_site = "A"
caught("bomb_site without a plant", s, "bomb_site")

s = make_plant_pair()[1].copy()
s.ctx.time_since_plant_s = None
caught("planted without time_since_plant_s", s, "time_since_plant_s")

s = make_plant_pair()[1].copy()
s.ctx.bomb_carrier_alive = 1
caught("bomb_carrier_alive set post-plant", s, "bomb_carrier_alive")

s = base.copy()
s.ctx.defuse_with_kit = 1
caught("defuse_with_kit without a defuse", s, "defuse_with_kit")

s = base.copy()
pl(s, 101).in_smoke = None
caught("alive player with null in_smoke", s, "alive but")

s = base.copy()
s.ctx.time_remaining_s = -1.0
caught("negative clock", s, "time_remaining_s")

s = base.copy()
s.players[0], s.players[5] = s.players[5], s.players[0]
caught("player order broken (CT first, then T)", s, "order")

# ── recompute_derived repairs the derived ones ────────────────────────────────
section("recompute_derived repairs derived fields")

s = base.copy()
s.ctx.alive_ct, s.ctx.alive_t = 0, 0
s.ctx.equip_ct = s.ctx.equip_t = 1.0
s.ctx.n_active_smokes, s.ctx.n_active_fires = 5, 3
i, j = s.index_of(101), s.index_of(205)
s.pair[i, j, LOS] = s.pair[j, i, LOS] = 1.0
check("corrupted copy is caught", len(s.validate()) >= 3)
recompute_derived(s)
check("recompute_derived restores counts, sums, dead LOS", not s.validate() and s == base, "; ".join(s.validate()[:2]))
s = base.copy()
s.ctx.money_ct = 123.0
recompute_derived(s)
check("recompute_derived leaves money alone (all-roster accounts are not derivable from tokens; HIGH-05)",
      s.ctx.money_ct == 123.0)

s = base.copy()
pl(s, 104).alive = 0
for f in ("yaw", "pitch", "speed", "has_helmet", "has_defuser", "has_c4", "weapon_class", "n_smoke",
          "n_flash", "n_he", "n_molly", "n_decoy", "money", "equipment_value", "spotted_by_count", "heard_enemy"):
    setattr(pl(s, 104), f, None)
pl(s, 104).health = pl(s, 104).armor = 0.0
recompute_derived(s, players=[s.index_of(104)])
check("a hand-killed player plus recompute_derived validates", not s.validate(), "; ".join(s.validate()[:2]))
check("its LOS row is NaN", np.isnan(s.pair[s.index_of(104), :, LOS]).all())

sys.exit(summary())
