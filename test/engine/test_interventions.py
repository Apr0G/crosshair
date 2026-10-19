"""
Intervention library (P7.1.1-P7.1.8): every intervention gets a positive case, a no-op /
idempotent / round-trip case, and an invalid precondition -> InterventionError; each
call must leave its input untouched and return a state whose `validate()` is empty.
Also: `INTERVENTION_FOR_TYPE` covers every instant and pseudo type of TYPE_KIND, and
the LOS/nav geometry path for moved players.

    .venv\\Scripts\\python.exe test\\engine\\test_interventions.py
"""
import copy
import sys

import numpy as np

from _harness import check, section, summary
from _cf_states import (BlindGeometry, OpenGeometry, action, make_flash_pair, make_kill_pair,
                        make_move_pair, make_plant_pair, make_smoke_pair, make_snapshot, pair_states)

from engine.actions.types import TYPE_KIND
from engine.counterfactual import interventions as iv
from engine.counterfactual.interventions import INTERVENTION_FOR_TYPE, InterventionError
from engine.state.schema import PAIR_CHANNELS

LOS, DIST, NAV, SMOKE = (PAIR_CHANNELS.index(c) for c in ("los", "dist_units", "nav_dist_units", "smoke_between"))


def run(name: str, fn, *args, **kw):
    """Call an intervention; check purity and consistency; return the result."""
    before = [copy.deepcopy(a) for a in args]
    out = fn(*args, **kw)
    pure = all((a == b) if hasattr(a, "validate") else True for a, b in zip(args, before))
    problems = out.validate()
    check(f"{name}: input untouched, validate() empty", pure and not problems, "; ".join(problems[:3]))
    return out


def raises(name: str, fn, *args, **kw) -> None:
    try:
        fn(*args, **kw)
    except InterventionError as e:
        check(f"{name}: InterventionError", True, str(e)[:70])
        return
    except Exception as e:      # noqa: BLE001  (any other exception is a failure)
        check(f"{name}: InterventionError", False, f"raised {type(e).__name__}: {e}")
        return
    check(f"{name}: InterventionError", False, "did not raise")


def pl(s, pid):
    return s.players[s.index_of(pid)]


def pair_eq(a, b) -> bool:
    return bool(np.array_equal(a.pair, b.pair, equal_nan=True))


# ── table ─────────────────────────────────────────────────────────────────────
section("INTERVENTION_FOR_TYPE")

missing = [t for t, k in TYPE_KIND.items() if k == "instant" and t not in INTERVENTION_FOR_TYPE]
check("every instant and pseudo type has (undo, redo)", not missing, f"missing {missing}")
check("every TYPE_KIND type is covered", set(TYPE_KIND) <= set(INTERVENTION_FOR_TYPE),
      f"missing {sorted(set(TYPE_KIND) - set(INTERVENTION_FOR_TYPE))}")
check("entries are callable pairs", all(len(v) == 2 and all(map(callable, v)) for v in INTERVENTION_FOR_TYPE.values()))
check("movement types and peek map to freeze_position",
      all(INTERVENTION_FOR_TYPE[t][0] is iv.undo_move for t in ("rotation", "hold", "save", "peek")))
check("buy maps to revert_buy (undo_buy)", INTERVENTION_FOR_TYPE["buy"][0] is iv.undo_buy)

# ── P7.1.1 kill ───────────────────────────────────────────────────────────────
section("undo_kill / redo_kill")

s_pre, s_post, kill = make_kill_pair()
u = run("undo_kill positive", iv.undo_kill, s_post, kill, s_pre)
check("victim alive with s_pre token", pl(u, 202) == pl(s_pre, 202))
check("attacker kills_round and damage_round back to s_pre", pl(u, 101) == pl(s_pre, 101))
check("alive_t, equip_t, money_t restored", (u.ctx.alive_t, u.ctx.equip_t, u.ctx.money_t)
      == (s_pre.ctx.alive_t, s_pre.ctx.equip_t, s_pre.ctx.money_t))
check("victim pair rows equal s_pre (LOS copied: nobody moved)", pair_eq(u, s_pre))
r = run("redo_kill (round trip)", iv.redo_kill, u, kill, s_post)
check("undo then redo == s_post", r == s_post)
raises("undo_kill on an alive victim", iv.undo_kill, s_pre, kill, s_pre)
raises("undo_kill without victim_id", iv.undo_kill, s_post, action("kill", 101), s_pre)
raises("undo_kill of a non-kill action", iv.undo_kill, s_post, action("flash", 101), s_pre)
raises("redo_kill on a dead victim", iv.redo_kill, s_post, kill, s_post)

# the bomb carrier dies and comes back
c_pre, c_post, ckill = make_kill_pair(101, 201)
check("carrier kill: s_post bomb_carrier_alive 0", c_post.ctx.bomb_carrier_alive == 0)
cu = run("undo_kill of the carrier", iv.undo_kill, c_post, ckill, c_pre)
check("carrier revived with the bomb, bomb_carrier_alive 1",
      pl(cu, 201).has_c4 == 1 and cu.ctx.bomb_carrier_alive == 1)

# team kill: attacker counters untouched
tk_pre, tk_post = pair_states({}, {"dead": {102}})
tk = action("team_kill", 101, victim_id=102, victim_health_before=100.0)
tu = run("undo_kill of a team_kill", iv.undo_kill, tk_post, tk, tk_pre)
check("team_kill: victim back, attacker counters unchanged", tu.players == tk_pre.players and pair_eq(tu, tk_pre))

# ── P7.1.2 damage and HE ──────────────────────────────────────────────────────
section("undo_damage / redo_damage / undo_he / redo_he")

d_pre, d_post = pair_states({}, {"health": {205: 70.0}, "armor": {205: 90.0}, "damage": {102: 30}})
dmg = action("damage", 102, victim_id=205, damage_total=30.0, armor_damage=10.0)
dmg.window_start_tick, dmg.window_end_tick = dmg.tick, dmg.tick + 200
du = run("undo_damage positive", iv.undo_damage, d_post, dmg, d_pre)
check("victim health/armor and attacker damage_round back to s_pre",
      pl(du, 205) == pl(d_pre, 205) and pl(du, 102) == pl(d_pre, 102))
du2 = run("undo_damage twice", iv.undo_damage, du, dmg, d_pre)
check("undo_damage is idempotent once capped", du2 == du)
big = action("damage", 102, victim_id=205, damage_total=500.0)
check("engagement total larger than the in-window hit is capped at s_pre",
      pl(run("undo_damage capped", iv.undo_damage, d_post, big, d_pre), 205).health == 100.0)
dr = run("redo_damage (round trip)", iv.redo_damage, du, dmg, d_post)
check("undo then redo == s_post", dr == d_post)
k_pre, k_post, _ = make_kill_pair(101, 205)
check("undo_damage on a dead victim is a no-op",
      run("undo_damage dead victim", iv.undo_damage, k_post, dmg, k_pre) == k_post)
raises("undo_damage without damage_total", iv.undo_damage, d_post, action("damage", 102, victim_id=205), d_pre)
raises("undo_damage with negative damage", iv.undo_damage, d_post,
       action("damage", 102, victim_id=205, damage_total=-5.0), d_pre)
raises("redo_damage on a dead victim", iv.redo_damage, k_post, dmg, d_post)

he = action("he", 102, enemies_damaged=[{"id": 205, "dmg": 30.0, "armor": 10.0}], teammates_damaged=[])
hu = run("undo_he positive", iv.undo_he, d_post, he, d_pre)
check("undo_he restores like undo_damage", hu == du)
both = iv.undo_damage(iv.undo_he(d_post, he, d_pre), dmg, d_pre)
check("undo_he + undo_damage on one victim restores once (no double damage_round drop)",
      pl(both, 102).damage_round == 0.0 and pl(both, 205).health == 100.0)
check("redo_he round trip", run("redo_he", iv.redo_he, hu, he, d_post) == d_post)
check("undo_he with no victims is a no-op",
      run("undo_he empty", iv.undo_he, d_post, action("he", 102, enemies_damaged=[]), d_pre) == d_post)
raises("undo_he with a malformed entry", iv.undo_he, d_post, action("he", 102, enemies_damaged=[{"id": 205}]), d_pre)

# ── P7.1.3 utility ────────────────────────────────────────────────────────────
section("remove_utility / add_utility")

sm_pre, sm_post, pop = make_smoke_pair()
check("smoke story: T 205 in smoke and some pairs blocked in s_post",
      pl(sm_post, 205).in_smoke == 1 and np.nansum(sm_post.pair[:, :, SMOKE]) > 0)
ru = run("remove_utility positive", iv.remove_utility, sm_post, pop)
check("token gone, flags, smoke_between and counts equal s_pre",
      ru.utility == sm_pre.utility and ru.players == sm_pre.players and pair_eq(ru, sm_pre)
      and ru.ctx.n_active_smokes == 0)
au = run("add_utility (round trip)", iv.add_utility, ru, pop, sm_post)
check("remove then add == s_post", au == sm_post)
raises("remove_utility twice", iv.remove_utility, ru, pop)
raises("add_utility when the token is already there", iv.add_utility, sm_post, pop, sm_post)
raises("add_utility without a reference", iv.add_utility, ru, pop, None)
raises("remove_utility of a non-utility action", iv.remove_utility, sm_post, action("kill", 103))

dec = {"x": -3000.0, "y": -3000.0, "z": 0.0, "owner": 105}
dsnap = make_snapshot(decoys=[dec])
dr_ = run("remove_utility of a far decoy", iv.remove_utility, dsnap, action("decoy_start", 105))
check("far decoy: only the token changes (no-op on derived fields)",
      dr_.players == dsnap.players and pair_eq(dr_, dsnap) and dr_.utility == [])

two = make_snapshot(smokes=[(700, 0, 0, 103), (700, 400, 0, 103)])
raises("two smokes of one owner without a land position", iv.remove_utility, two, action("smoke_pop", 103))
pick = run("land position picks the token", iv.remove_utility, two, action("smoke_pop", 103, x=700.0, y=400.0, z=0.0))
check("the other smoke stays", [(u.x, u.y) for u in pick.utility] == [(700.0, 0.0)])

f_pre, f_post, fade = make_smoke_pair(fade=True)
fu = run("undo smoke_fade (add from s_pre)", INTERVENTION_FOR_TYPE["smoke_fade"][0], f_post, fade, f_pre)
check("faded smoke back, in s_pre's order and flags", fu.utility == f_pre.utility and fu.players == f_pre.players)

order_pre = make_snapshot(smokes=[{"x": 700, "y": 0, "owner": 103, "start_tick": 4000},
                                  {"x": 700, "y": 400, "owner": 104, "start_tick": 4600}])
order_post = make_snapshot(smokes=[{"x": 700, "y": 400, "owner": 104, "start_tick": 4600}])
ou = iv.add_utility(order_post, action("smoke_fade", 103), order_pre)
check("add_utility keeps oldest-first order", [u.owner_id for u in ou.utility] == [103, 104])

fire_pre = make_snapshot()
fire_post = make_snapshot(fires=[{"x": 0.0, "y": 0.0, "z": 0.0, "owner": 203}])
check("fire story: CT 101 on fire in s_post", pl(fire_post, 101).on_fire == 1)
fr = run("remove fire", iv.remove_utility, fire_post, action("fire_start", 203))
check("on_fire cleared, n_active_fires 0", pl(fr, 101).on_fire == 0 and fr.ctx.n_active_fires == 0 and fr.players == fire_pre.players)
unk_post = make_snapshot(fires=[{"x": 0.0, "y": 0.0, "z": 0.0, "owner": 203, "radius": "unknown"}])
check("fire of unknown extent: on_fire None for every alive player (HIGH-04)",
      all(p.on_fire is None for p in unk_post.players if p.alive == 1) and unk_post.validate() == [])
ur = run("remove unknown-extent fire", iv.remove_utility, unk_post, action("fire_start", 203))
check("removing it resolves on_fire to 0 (a real change, no longer inert)",
      all(p.on_fire == 0 for p in ur.players if p.alive == 1) and ur.ctx.n_active_fires == 0)

# ── P7.1.4 flash ──────────────────────────────────────────────────────────────
section("unflash / reflash")

fl_pre, fl_post, fl = make_flash_pair()
uf = run("unflash positive", iv.unflash, fl_post, fl, fl_pre)
check("blinded players back to s_pre durations", uf.players == fl_pre.players)
check("unflash without state_pre sets 0", all(p.flash_duration_remaining == 0.0
                                                for p in run("unflash no ref", iv.unflash, fl_post, fl).players))
check("unflash is idempotent", run("unflash twice", iv.unflash, uf, fl, fl_pre) == uf)
check("flash that blinded nobody is a no-op",
      run("unflash empty", iv.unflash, fl_post, action("flash", 101, blinded_enemies=[]), fl_pre) == fl_post)
check("reflash round trip", run("reflash", iv.reflash, uf, fl, fl_post) == fl_post)
check("reflash from params durations", pl(run("reflash params", iv.reflash, uf, fl), 203).flash_duration_remaining == 2.5)
raises("unflash with an unknown player", iv.unflash, fl_post, action("flash", 101, blinded_enemies=[{"id": 999, "duration": 1.0}]))
raises("unflash of a non-flash action", iv.unflash, fl_post, action("he", 101))
raises("reflash without any duration", iv.reflash, uf, action("flash", 101, blinded_enemies=[{"id": 203}]))

# ── P7.1.5 plant ──────────────────────────────────────────────────────────────
section("unplant / replant / unplant_start / replant_start")

p_pre, p_post, plant = make_plant_pair()
up = run("unplant positive", iv.unplant, p_post, plant, p_pre)
check("bomb fields back to pre-plant", (up.ctx.bomb_planted, up.ctx.bomb_site, up.ctx.time_since_plant_s,
                                        up.ctx.bomb_carrier_alive) == (0, None, None, 1))
dt = p_post.ctx.time_into_round_s - p_pre.ctx.time_into_round_s
check("clock mode switches back: s_pre round clock advanced to s_post time",
      abs(up.ctx.time_remaining_s - (p_pre.ctx.time_remaining_s - dt)) < 1e-9)
check("planter has the bomb and is planting again (s_pre)", pl(up, 201) == pl(p_pre, 201))
nr = iv.unplant(p_post, plant)
check("unplant without s_pre: clock = round_time_s - time_into_round_s",
      abs(nr.ctx.time_remaining_s - (115.0 - p_post.ctx.time_into_round_s)) < 1e-9 and pl(nr, 201).is_planting == 0)
check("replant round trip", run("replant", iv.replant, up, plant, p_post) == p_post)
rp = run("replant without s_post", iv.replant, up, plant)
check("replant without s_post: bomb clock 40 s, site from params",
      rp.ctx.time_remaining_s == 40.0 and rp.ctx.time_since_plant_s == 0.0 and rp.ctx.bomb_site == "A")
raises("unplant when not planted", iv.unplant, p_pre, plant, p_pre)
raises("replant when planted", iv.replant, p_post, plant, p_post)
raises("replant without any site", iv.replant, up, action("plant_complete", 201))
dfs = make_snapshot(planted=True, plant_tick=4000, c4=None, defusing=101, defuse_kit=1)
raises("unplant during a defuse", iv.unplant, dfs, plant)

ps_pre = make_snapshot(tick=4999)
ps_post = make_snapshot(tick=5001, planting=201)
pstart = action("plant_start", 201, site="A")
check("unplant_start clears is_planting", run("unplant_start", iv.unplant_start, ps_post, pstart, ps_pre).players == ps_pre.players)
check("replant_start round trip", run("replant_start", iv.replant_start, ps_pre, pstart, ps_post).players == ps_post.players)
check("unplant_start on a non-planting state is a no-op",
      run("unplant_start no-op", iv.unplant_start, ps_pre, pstart, ps_pre).players == ps_pre.players)
raises("unplant_start after plant_complete", iv.unplant_start, p_post, pstart)

# ── P7.1.6 defuse ─────────────────────────────────────────────────────────────
section("undefuse_start / redefuse_start / undefuse_complete / redefuse_complete")

kw = dict(planted=True, plant_tick=4000, c4=None)
ds_pre, ds_post = pair_states({}, {"defusing": 101, "defuse_kit": 1}, **kw)
dstart = action("defuse_start", 101, kit=True)
uds = run("undefuse_start positive", iv.undefuse_start, ds_post, dstart, ds_pre)
check("defuse fields and is_defusing back to s_pre",
      uds.players == ds_pre.players and (uds.ctx.defuse_in_progress, uds.ctx.defuse_with_kit) == (0, None))
check("redefuse_start round trip", run("redefuse_start", iv.redefuse_start, uds, dstart, ds_post) == ds_post)
check("redefuse_start without s_post takes params kit",
      run("redefuse_start params", iv.redefuse_start, uds, dstart).ctx.defuse_with_kit == 1)
raises("undefuse_start with no defuse in progress", iv.undefuse_start, ds_pre, dstart, ds_pre)
raises("undefuse_start pre-plant", iv.undefuse_start, make_snapshot(), dstart)

dc_pre, dc_post = pair_states({"defusing": 101, "defuse_kit": 1}, {}, **kw)
dcomp = action("defuse_complete", 101, kit=True)
udc = run("undefuse_complete positive", iv.undefuse_complete, dc_post, dcomp, dc_pre)
check("defuse back in progress with s_pre kit and is_defusing",
      udc.players == dc_pre.players and (udc.ctx.defuse_in_progress, udc.ctx.defuse_with_kit) == (1, 1))
check("redefuse_complete round trip", run("redefuse_complete", iv.redefuse_complete, udc, dcomp, dc_post) == dc_post)
abort = action("defuse_abort", 101)
check("undefuse_abort behaves like undefuse_complete",
      run("undefuse_abort", iv.undefuse_abort, dc_post, abort, dc_pre) == udc)
raises("undefuse_complete while still defusing", iv.undefuse_complete, dc_pre, dcomp, dc_pre)
dead_def = make_snapshot(**kw, dead={101})
raises("undefuse_complete with a dead defuser", iv.undefuse_complete, dead_def, dcomp)

# ── bomb carry and weapon drops ───────────────────────────────────────────────
section("bomb_drop / bomb_pickup / drop")

bd_pre, bd_post = pair_states({}, {"c4": None})
bdrop = action("bomb_drop", 201)
ubd = run("undo_bomb_drop positive", iv.undo_bomb_drop, bd_post, bdrop, bd_pre)
check("dropper carries the bomb again", ubd.players == bd_pre.players and ubd.ctx.bomb_carrier_alive == 1)
check("redo_bomb_drop round trip", run("redo_bomb_drop", iv.redo_bomb_drop, ubd, bdrop, bd_post) == bd_post)
raises("undo_bomb_drop when already carrying", iv.undo_bomb_drop, bd_pre, bdrop, bd_pre)
bp_pre, bp_post = pair_states({"c4": None}, {"c4": 203})
bpick = action("bomb_pickup", 203)
ubp = run("undo_bomb_pickup positive", iv.undo_bomb_pickup, bp_post, bpick, bp_pre)
check("picker no longer carries", ubp.players == bp_pre.players and ubp.ctx.bomb_carrier_alive == 0)
check("redo_bomb_pickup round trip", run("redo_bomb_pickup", iv.redo_bomb_pickup, ubp, bpick, bp_post) == bp_post)
raises("undo_bomb_pickup when not carrying", iv.undo_bomb_pickup, bp_pre, bpick, bp_pre)

dr_pre = make_snapshot(tick=4999)
dr_post = make_snapshot(tick=5001)
for pid, (wc, ev) in ((102, ("knife", 1000.0)), (103, ("rifle", 7400.0))):
    setattr(pl(dr_post, pid), "weapon_class", wc)
    setattr(pl(dr_post, pid), "equipment_value", ev)
dr_post.ctx.equip_ct = float(sum(p.equipment_value for p in dr_post.players if p.side == "ct"))
check("drop story is consistent", not dr_post.validate())
give = action("drop", 102, receiver_id=103, weapon_class="rifle")
udr = run("undo_drop positive", iv.undo_drop, dr_post, give, dr_pre)
check("both loadouts and equip_ct back to s_pre", udr.players == dr_pre.players and udr.ctx.equip_ct == dr_pre.ctx.equip_ct)
check("redo_drop round trip", run("redo_drop", iv.redo_drop, udr, give, dr_post).players == dr_post.players)
raises("undo_drop without receiver", iv.undo_drop, dr_post, action("drop", 102), dr_pre)

# ── P7.1.7 freeze_position ────────────────────────────────────────────────────
section("freeze_position")

s0, s1, peek = make_move_pair()
fz = run("freeze_position positive", iv.freeze_position, s1, 101, s0)
check("mover back at s0 position/area/callout/cell/speed", pl(fz, 101) == pl(s0, 101))
check("mover pair rows equal s0 (dist recomputed, LOS copied)", pair_eq(fz, s0))
check("freeze to the current position is an unchanged copy", run("freeze no-op", iv.freeze_position, s0, 101, s0) == s0)
check("freeze is idempotent", run("freeze twice", iv.freeze_position, fz, 101, s0) == fz)
check("undo_move (peek) == freeze_position of the actor", INTERVENTION_FOR_TYPE["peek"][0](s1, peek, s0) == fz)
raises("freeze a dead player", iv.freeze_position, make_snapshot(dead={101}), 101, s0)
raises("freeze without a reference", iv.freeze_position, s1, 101, None)
raises("freeze an unknown player", iv.freeze_position, s1, 999, s0)

# two movers: freezing one leaves its pair with the other mover unknown unless geometry is given
m0 = make_snapshot(tick=4936)
m1 = make_snapshot(tick=5000, positions={101: (400.0, 0.0, 0.0), 201: (900.0, 0.0, 0.0)})
i101, i201, i202 = m1.index_of(101), m1.index_of(201), m1.index_of(202)
g0 = run("freeze one of two movers (no geometry)", iv.freeze_position, m1, 101, m0)
check("pair with the other mover: LOS and nav NaN, dist recomputed",
      np.isnan(g0.pair[i101, i201, LOS]) and np.isnan(g0.pair[i101, i201, NAV])
      and abs(g0.pair[i101, i201, DIST] - 900.0) < 1e-3)
check("pair with a static player: LOS copied from s0", g0.pair[i101, i202, LOS] == m0.pair[i101, i202, LOS])
gb = run("freeze with BlindGeometry", iv.freeze_position, m1, 101, m0, geometry=BlindGeometry())
check("geometry recomputes LOS for the moved pair (0 when blind)", gb.pair[i101, i201, LOS] == 0.0)
go = run("freeze with OpenGeometry", iv.freeze_position, m1, 101, m0, geometry=OpenGeometry())
check("geometry recomputes nav_dist_units", not np.isnan(go.pair[i101, i201, NAV]))

# ── P7.1.8 revert_buy ─────────────────────────────────────────────────────────
section("revert_buy / undo_buy / undo_team_buy")

fe = make_snapshot()
prior = {"equipment_value": 1000.0, "weapon_class": "pistol", "armor": 0.0, "has_helmet": 0, "money": 3500.0}
rb = run("revert_buy positive", iv.revert_buy, fe, 101, prior)
check("loadout set, equip_ct and money_ct recomputed",
      pl(rb, 101).equipment_value == 1000.0 and rb.ctx.equip_ct == fe.ctx.equip_ct - 3700.0
      and rb.ctx.money_ct == fe.ctx.money_ct - pl(fe, 101).money + 3500.0)
check("revert_buy to the current loadout is a no-op", run("revert_buy no-op", iv.revert_buy, fe, 101, iv.loadout_of(fe, 101)) == fe)
raises("revert_buy with an unknown key", iv.revert_buy, fe, 101, {"equipment_value": 1.0, "health": 5.0})
raises("revert_buy without equipment_value", iv.revert_buy, fe, 101, {"armor": 0.0})
raises("revert_buy of a dead player", iv.revert_buy, make_snapshot(dead={101}), 101, prior)
prior_state = make_snapshot(equipment={p: 800.0 for p in (101, 102, 103, 104, 105)})
ub = run("undo_buy from the prior state", iv.undo_buy, fe, action("buy", 103, kind="round"), prior_state)
check("undo_buy takes the actor's loadout from the prior state", pl(ub, 103).equipment_value == 800.0)
tb = run("undo_team_buy", iv.undo_team_buy, fe, action("team_buy", 101, kind="round"), prior_state)
check("undo_team_buy reverts the whole side", tb.ctx.equip_ct == 4000.0 and tb.ctx.equip_t == fe.ctx.equip_t)

# ── recompute_derived ─────────────────────────────────────────────────────────
section("recompute_derived")

base = make_snapshot(4, 3, smokes=[(750, 200, 0, 103)])
again = base.copy()
iv.recompute_derived(again)
check("recompute_derived on a consistent state changes nothing", again == base)

sys.exit(summary())
