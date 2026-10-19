"""
Tests for state_sampler.py on synthetic tables — no awpy, no demo, no model, no DB.

Run from repo root:
    .venv/bin/python test/test_state_sampler.py

test_build_state.py checks the same properties against a REAL demo when one is on
disk; this file is the always-runnable version, with hand-built inputs whose right
answers are known exactly.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from _harness import check, close, section, summary
import state_sampler as ss
from _synthetic import (make_tables, AlwaysVisible, MAP_NAME,
                        FREEZE_END, KILL_1, KILL_2, PLANT, SMOKE)


def _snap(*rows):
    cols = ["name", "side", "X", "Y", "is_alive", "inventory"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame(columns=cols)


def main() -> int:
    section("helpers")
    check("_team_spread: one player -> None, not 0.0", ss._team_spread(_snap(("a", "ct", 0, 0, True, []))) is None)
    check("_team_spread: two players 300 apart -> 300",
          close(ss._team_spread(_snap(("a", "ct", 0, 0, True, []), ("b", "ct", 300, 0, True, []))), 300.0))
    check("_team_spread: NaN positions are ignored",
          ss._team_spread(_snap(("a", "ct", 0, 0, True, []), ("b", "ct", np.nan, 0, True, []))) is None)
    check("_team_spread: empty frame -> None", ss._team_spread(_snap()) is None)

    check("_count_util classifies inventory strings",
          ss._count_util(["weapon_smokegrenade", "weapon_flashbang", "weapon_flashbang",
                          "weapon_hegrenade", "weapon_incgrenade", "weapon_molotov", "weapon_ak47"])
          == {"smokes": 1, "flashes": 2, "he": 1, "molotovs": 2})
    check("_count_util tolerates a non-list", ss._count_util(None) == {"smokes": 0, "flashes": 0, "he": 0, "molotovs": 0})
    check("_team_util sums over the team",
          ss._team_util(_snap(("a", "ct", 0, 0, True, ["weapon_smokegrenade"]),
                              ("b", "ct", 0, 0, True, ["weapon_smokegrenade", "weapon_hegrenade"])))
          == {"smokes": 2, "flashes": 0, "he": 1, "molotovs": 0})

    check("_min_dist_to_xy picks the closest player",
          close(ss._min_dist_to_xy(_snap(("a", "ct", 0, 0, True, []), ("b", "ct", 30, 40, True, [])), (0, 0)), 0.0))
    check("_min_dist_to_xy: no target -> None", ss._min_dist_to_xy(_snap(("a", "ct", 0, 0, True, [])), None) is None)

    smokes = pd.DataFrame([{"start_tick": 100, "end_tick": 200, "X": 1.0, "Y": 2.0},
                           {"start_tick": 300, "end_tick": 400, "X": 3.0, "Y": 4.0}])
    check("_active_xy returns only smokes live at the tick", ss._active_xy(smokes, 150) == [(1.0, 2.0)])
    check("_active_xy is inclusive at both ends", ss._active_xy(smokes, 200) == [(1.0, 2.0)] and ss._active_xy(smokes, 300) == [(3.0, 4.0)])
    check("_active_xy: nothing live -> []", ss._active_xy(smokes, 250) == [])
    check("_active_xy: missing columns -> []", ss._active_xy(pd.DataFrame({"X": [1]}), 1) == [])

    check("_site_blocked: a smoke within the radius covers the site", ss._site_blocked((0, 0), [(100, 0)]))
    check("_site_blocked: beyond the radius does not", not ss._site_blocked((0, 0), [(ss.SITE_BLOCK_RADIUS + 1, 0)]))
    check("_site_blocked: no bomb / no smokes -> False", not ss._site_blocked(None, [(0, 0)]) and not ss._site_blocked((0, 0), []))

    section("_info_state only looks BACKWARDS within the window (CRIT-02 regression)")
    ct = _snap(("a", "ct", 0, 0, True, []))
    t  = _snap(("c", "t", 0, 0, True, []))
    cache = {500:  {"a": [{"name": "c", "side": "t", "position": [0, 0, 0]}]},
             1500: {"a": [{"name": "c", "side": "t", "position": [0, 0, 0]},
                          {"name": "d", "side": "t", "position": [0, 0, 0]}],
                    "c": [{"name": "a", "side": "ct", "position": [0, 0, 0]}]}}
    check("before any sighting: zeros, even though a later tick is cached",
          ss._info_state(ct, t, cache, 400, 640) == {"ct_spotted_count": 0, "t_spotted_count": 0})
    check("inside the window: the most recent PAST entry is used",
          ss._info_state(ct, t, cache, 1000, 640) == {"ct_spotted_count": 1, "t_spotted_count": 0})
    check("a sighting older than the window is forgotten",
          ss._info_state(ct, t, cache, 1200, 640) == {"ct_spotted_count": 0, "t_spotted_count": 0})
    check("exactly at a cached tick counts it",
          ss._info_state(ct, t, cache, 1500, 640) == {"ct_spotted_count": 2, "t_spotted_count": 1})
    check("empty cache -> zeros", ss._info_state(ct, t, {}, 1500, 640) == {"ct_spotted_count": 0, "t_spotted_count": 0})

    section("_team_heard_enemy")
    foot = pd.DataFrame([{"tick": 90, "side": "t", "X": 500.0, "Y": 0.0, "name": "c"}])
    team = _snap(("a", "ct", 0, 0, True, []), ("b", "ct", 5000, 0, True, []))
    check("players within HEARD_RANGE of a recent enemy sound hear it",
          ss._team_heard_enemy(team, "t", 100, foot, None, None, heard_window=64) == 1)
    check("a sound older than the window is not heard",
          ss._team_heard_enemy(team, "t", 200, foot, None, None, heard_window=64) == 0)
    check("own-team sounds are ignored",
          ss._team_heard_enemy(team, "ct", 100, foot, None, None, heard_window=64) == 0)
    noside = foot.drop(columns=["side"])
    check("without a side column, enemy names are used",
          ss._team_heard_enemy(team, "t", 100, noside, None, None, heard_window=64, enemy_names={"c"}) == 1)

    section("iter_round_contexts on the synthetic demo")
    tables = make_tables(64)
    ctxs = list(ss.iter_round_contexts(tables, "syn", MAP_NAME, None))
    check("the round with no winner is skipped, the playable one kept",
          [c.round_num for c in ctxs] == [1], str([c.round_num for c in ctxs]))
    ctx = ctxs[0]
    check("round bounds come from freeze_end / official_end", (ctx.r_start, ctx.r_end) == (FREEZE_END, tables["rounds"].loc[0, "official_end"]))
    check("plant tick and site are picked up from event_bomb_planted",
          ctx.plant_tick == PLANT and ctx.bomb_xy == (1200.0, 100.0))
    check("label is CT win", ctx.round_won_ct == 1)
    check("64-tick windows: 64 / 32 / 640",
          (ctx.sample_interval, ctx.snap_half, ctx.heard_window) == (64, 32, 640))
    ctx128 = next(ss.iter_round_contexts(make_tables(128), "syn", MAP_NAME, None))
    check("128-tick windows double in ticks (same seconds)",
          (ctx128.sample_interval, ctx128.snap_half, ctx128.heard_window) == (128, 64, 1280))

    noend = make_tables(64)
    noend["rounds"].loc[0, "official_end"] = np.nan
    c2 = next(ss.iter_round_contexts(noend, "syn", MAP_NAME, None))
    check("NaN official_end falls back to end (NaN is truthy — `or` would not)", c2.r_end == tables["rounds"].loc[0, "end"])

    section("build_state on the synthetic round")
    st0 = ss.build_state(FREEZE_END, ctx)
    check("state exists at freeze end", st0 is not None)
    check("5v5-style counts: 2 CT, 2 T alive at the start", (st0["alive_ct"], st0["alive_t"]) == (2, 2))
    check("clock starts at 0 with the full round time left",
          st0["time_into_round_s"] == 0.0 and st0["time_remaining_s"] == ss.ROUND_TIME_S)
    check("utility on hand is counted from inventories",
          (st0["smokes_ct"], st0["flashes_ct"], st0["he_t"]) == (1, 1, 1), str((st0["smokes_ct"], st0["flashes_ct"], st0["he_t"])))
    check("defuser is detected on the CT side", st0["has_defuser"] == 1)
    check("team totals: hp / armor / helmets / equipment",
          (st0["total_hp_ct"], st0["total_armor_t"], st0["helmets_ct"], st0["equip_value_t"]) == (200.0, 200.0, 2, 8000))
    check("spreads are real for 2-player teams", st0["ct_spread"] == 300.0 and st0["t_spread"] is not None)
    check("no bomb down: distances None, post_plant 0",
          st0["post_plant"] == 0 and st0["min_dist_ct_to_bomb"] is None and st0["min_dist_t_to_bomb"] is None)
    check("round label is carried through", st0["round_won_ct"] == 1)
    check("tick is echoed back", st0["tick"] == FREEZE_END)

    mid = ss.build_state(KILL_1 + 64, ctx)       # one second after A killed C
    check("after the first kill: 2v1", (mid["alive_ct"], mid["alive_t"]) == (2, 1))
    check("one T left: t_spread is None, not 0.0", mid["t_spread"] is None)
    check("pre-plant clock counts down from the round time",
          close(mid["time_remaining_s"], ss.ROUND_TIME_S - mid["time_into_round_s"]))
    check("the enemy footsteps 600 u away were heard by A within the window",
          ss.build_state(KILL_1, ctx)["ct_heard_enemy"] >= 1)

    post = ss.build_state(PLANT + 128, ctx)      # two seconds after the plant
    # D is standing on the bomb; A at the origin is the closest CT (B has walked far off).
    check("post-plant flag and bomb distances",
          post["post_plant"] == 1 and post["min_dist_t_to_bomb"] == 0.0
          and close(post["min_dist_ct_to_bomb"], np.hypot(1200, 100), 0.01),
          str((post["post_plant"], post["min_dist_t_to_bomb"], post["min_dist_ct_to_bomb"])))
    check("post-plant clock is the C4 timer", close(post["time_remaining_s"], ss.C4_TIMER - 2.0), str(post["time_remaining_s"]))

    smoked = ss.build_state(SMOKE + 128 + 64, ctx)   # smoke deployed and live
    check("active smokes are counted with their positions",
          smoked["active_smokes"] == 1 and smoked["active_smokes_xy"] == [(500.0, 0.0)])
    check("a smoke far from the site does not flag site_smoked", smoked["site_smoked"] == 0)

    check("a tick before any player row -> None", ss.build_state(FREEZE_END - 1000, ctx) is None)
    decided = ss.build_state(KILL_2 + 64, ctx)
    check("post-decision state still exists (score_impact needs it) with alive_t == 0",
          decided is not None and decided["alive_t"] == 0)

    section("build_state is pure and never looks ahead")
    a, b = ss.build_state(KILL_1 + 5, ctx), ss.build_state(KILL_1 + 5, ctx)
    check("same input, same output", a == b)
    before, after = ss.build_state(KILL_1 - 1, ctx), ss.build_state(KILL_1 + 1, ctx)
    check("tick-1 before a kill still has the victim alive; tick+1 does not",
          before["alive_t"] == 2 and after["alive_t"] == 1, f"{before['alive_t']} -> {after['alive_t']}")
    probe = KILL_1 + 40
    trunc = copy.copy(ctx)
    trunc.r_ticks = ctx.r_ticks[ctx.r_ticks["tick"] <= probe]
    check("removing every tick after T leaves the state at T unchanged",
          ss.build_state(probe, trunc) == ss.build_state(probe, ctx))

    section("visibility cache feeds the spotted counts")
    vis_ctx = next(ss.iter_round_contexts(tables, "syn", MAP_NAME, AlwaysVisible()))
    seen = ss.build_state(FREEZE_END + 64, vis_ctx)
    check("with line of sight everywhere, each side sees every alive enemy",
          (seen["ct_spotted_count"], seen["t_spotted_count"]) == (2, 2), str((seen["ct_spotted_count"], seen["t_spotted_count"])))
    # The smoke lands at (500, 0): dead on the A(0,0) <-> C(1000,0) line.
    in_smoke = ss.build_state(SMOKE + 128 + 64, vis_ctx)
    check("a live smoke on the line of sight hides players behind it",
          in_smoke["ct_spotted_count"] < 2 or in_smoke["t_spotted_count"] < 2,
          str((in_smoke["ct_spotted_count"], in_smoke["t_spotted_count"])))

    section("sample_round_states: the training grid")
    grid = ss.sample_round_states(tables, "syn", MAP_NAME, None)
    ticks = [s["tick"] for s in grid]
    check("one row per second from freeze end", ticks[:3] == [FREEZE_END, FREEZE_END + 64, FREEZE_END + 128])
    check("rows are tagged grid", all(s["state_kind"] == "grid" for s in grid))
    check("rows stop before official end", max(ticks) < ctx.r_end)
    check("every row carries match_id and map", all(s["match_id"] == "syn" and s["map"] == MAP_NAME for s in grid))
    grid128 = ss.sample_round_states(make_tables(128), "syn", MAP_NAME, None)
    check("128 tick yields the same number of 1 s samples", len(grid128) == len(grid), f"{len(grid128)} vs {len(grid)}")
    check("128 tick: same clocks in seconds", [s["time_into_round_s"] for s in grid128][:5] == [s["time_into_round_s"] for s in grid][:5])

    section("sample_boundary_states: tick±1 around instantaneous actions")
    events = [
        {"match_id": "syn", "round_num": 1, "action": {"impact_kind": "instantaneous", "resolve_tick": KILL_1}},
        {"match_id": "syn", "round_num": 1, "action": '{"impact_kind": "instantaneous", "resolve_tick": %d}' % KILL_1},  # dup, as JSON
        {"match_id": "syn", "round_num": 1, "action": {"impact_kind": "instantaneous", "resolve_tick": FREEZE_END + 64 * 10 + 1}},  # +1 side lands ON the grid
        {"match_id": "syn", "round_num": 1, "action": {"impact_kind": "durative", "window_start_tick": 1, "window_end_tick": 2}},
        {"match_id": "syn", "round_num": 1, "action": {"impact_kind": "instantaneous", "resolve_tick": None}},
        {"match_id": "other", "round_num": 1, "action": {"impact_kind": "instantaneous", "resolve_tick": KILL_1}},
        {"match_id": "syn", "round_num": 7, "action": {"impact_kind": "instantaneous", "resolve_tick": KILL_1}},
    ]
    boundary = ss.sample_boundary_states(tables, "syn", events, MAP_NAME, None)
    bticks = sorted(s["tick"] for s in boundary)
    g10 = FREEZE_END + 64 * 10          # the grid tick the third event's -1 side lands on
    check("boundary ticks are resolve_tick ± 1, deduped, minus anything already on the grid",
          bticks == [g10 + 2, KILL_1 - 1, KILL_1 + 1], str(bticks))
    check("rows are tagged boundary", all(s["state_kind"] == "boundary" for s in boundary))
    check("no events -> no boundary rows", ss.sample_boundary_states(tables, "syn", [], MAP_NAME, None) == [])

    return summary()


if __name__ == "__main__":
    sys.exit(main())
