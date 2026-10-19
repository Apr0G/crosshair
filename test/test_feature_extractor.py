"""
Tests for feature_extractor.py on synthetic tables — no awpy, no demo, no model.

Run from repo root:
    .venv/bin/python test/test_feature_extractor.py

Half pure-helper checks, half a full extract_events() run over the synthetic
round in _synthetic.py, whose right answers are known exactly. The contract
checks in test_attribution_windows.py cover the same emitted fields against a
REAL demo when one is on disk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from _harness import check, close, section, summary
import feature_extractor as fx
from _synthetic import make_tables, AlwaysVisible, MAP_NAME, FREEZE_END, SMOKE, HE, KILL_1, KILL_2, PLANT


def _by_type(events):
    out = {}
    for e in events:
        out.setdefault(e["event_type"], []).append(e)
    return out


def main() -> int:
    section("scalar helpers")
    check("_nan: None and NaN are 'missing'; 0 and strings are not",
          fx._nan(None) and fx._nan(float("nan")) and fx._nan(np.nan) and not fx._nan(0) and not fx._nan("x"))
    check("_first_valid skips None/NaN and keeps a legitimate 0", fx._first_valid(None, float("nan"), 0, 5) == 0)
    check("_first_valid with nothing valid -> None", fx._first_valid(None, np.nan) is None)
    check("_safe maps NaN to the default", fx._safe(np.nan, "d") == "d" and fx._safe(None) is None)
    check("_safe stringifies iterables and drops None entries", fx._safe([1, None, "x"]) == ["1", "x"])
    check("_safe leaves strings alone", fx._safe("abc") == "abc")
    check("_pos rounds to 0.1 and tolerates missing coordinates",
          fx._pos({"X": 1.26, "Y": 2, "Z": 3}) == [1.3, 2.0, 3.0] and fx._pos({"X": 1}) == [None, None, None])
    check("_dist2/_dist3 return inf (never 0.0) on bad input",
          fx._dist2([None, 0], [0, 0]) == float("inf") and fx._dist3([0, 0], [0, 0, 0]) == float("inf"))
    check("_dist3 is euclidean", close(fx._dist3([0, 0, 0], [3, 4, 12]), 13.0))
    check("_enemy_of never guesses", fx._enemy_of("ct") == "t" and fx._enemy_of("t") == "ct" and fx._enemy_of("?") == "?")
    check("_vel_mag", close(fx._vel_mag({"velocity_X": 3, "velocity_Y": 4, "velocity_Z": 0}), 5.0))
    check("_round_type thresholds",
          [fx._round_type(v, s) for v, s in ((400, "t"), (1000, "t"), (2000, "t"), (4000, "ct"), (5000, "ct"), (4400, "t"), (4500, "t"))]
          == ["full_eco", "eco", "force", "half_buy", "full_buy", "half_buy", "full_buy"])
    check("_ticks scales seconds by the tick rate", fx._ticks(2.0, 64) == 128 and fx._ticks(2.0, 128) == 256)

    section("geometry helpers")
    p = (0, 0, 0)
    check("_relative_direction: yaw 0 faces +X",
          [fx._relative_direction(p, 0, t) for t in ((10, 0, 0), (0, 10, 0), (-10, 0, 0), (0, -10, 0))]
          == ["front", "left", "back", "right"])
    check("_relative_direction: yaw 90 rotates the frame", fx._relative_direction(p, 90, (0, -10, 0)) == "front")
    check("_relative_direction: bad input -> 'unknown'", fx._relative_direction(p, None, (1, 1)) == "unknown")
    check("_smoke_blocks_los: a smoke on the segment blocks", fx._smoke_blocks_los((0, 0), (1000, 0), [(500, 0)]))
    check("_smoke_blocks_los: a smoke beyond its radius from the segment does not",
          not fx._smoke_blocks_los((0, 0), (1000, 0), [(500, fx._SMOKE_RADIUS + 1)]))
    check("_smoke_blocks_los: a smoke past the far end does not block",
          not fx._smoke_blocks_los((0, 0), (1000, 0), [(1300, 0)]))
    check("_smoke_blocks_los: degenerate segment / no smokes -> False",
          not fx._smoke_blocks_los((0, 0), (0, 0), [(0, 0)]) and not fx._smoke_blocks_los((0, 0), (1, 1), []))
    check("_toward_bomb uses the dot product of movement and bomb direction",
          fx._toward_bomb([0, 0, 0], [10, 0, 0], [100, 0, 0]) is True
          and fx._toward_bomb([0, 0, 0], [-10, 0, 0], [100, 0, 0]) is False
          and fx._toward_bomb([0, 0, 0], [10, 0, 0], None) is None)

    section("frame helpers")
    df = pd.DataFrame({"tick": [0, 10, 20, 30], "name": ["a", "a", "b", "a"], "v": [1, 2, 3, 4]})
    check("_tick_slice is inclusive on both ends", list(fx._tick_slice(df, 10, 20)["v"]) == [2, 3])
    check("_tick_slice on a missing tick column returns the frame", fx._tick_slice(df.drop(columns=["tick"]), 0, 1).shape == (4, 2))
    snap = fx._get_tick_snap(df, 20, tick_rate=64)
    check("_get_tick_snap: exact tick rows are returned as-is", list(snap["name"]) == ["b"])
    snap = fx._get_tick_snap(df, 15, tick_rate=64)
    check("_get_tick_snap: otherwise the last row per player within one second",
          sorted(snap["name"]) == ["a"] and int(snap["v"].iloc[0]) == 2, str(snap.to_dict("records")))
    check("_get_tick_snap: the fallback window scales with tick rate",
          fx._get_tick_snap(df, 35, tick_rate=4).empty and len(fx._get_tick_snap(df, 35, tick_rate=10)) == 1)
    check("_get_tick_snap: empty frame / None tick -> empty", fx._get_tick_snap(df.iloc[0:0], 1).empty and fx._get_tick_snap(df, None).empty)
    check("_player_row finds by name, None otherwise",
          fx._player_row(df, "b")["v"] == 3 and fx._player_row(df, "zz") is None and fx._player_row(df.iloc[0:0], "a") is None)

    snap = pd.DataFrame([{"name": "A", "side": "ct", "is_alive": True}, {"name": "B", "side": "ct", "is_alive": False},
                         {"name": "C", "side": "t", "is_alive": True},  {"name": "D", "side": "t", "is_alive": True}])
    check("_alive_counts_from_snap: (teammates excluding self, enemies)",
          fx._alive_counts_from_snap(snap, "A", "ct") == (0, 2) and fx._alive_counts_from_snap(snap, "C", "t") == (1, 1))
    check("_alive_counts_from_snap: empty -> (None, None)", fx._alive_counts_from_snap(snap.iloc[0:0], "A", "ct") == (None, None))
    check("_get_enemy_names", fx._get_enemy_names(snap, "t") == {"C", "D"} and fx._get_enemy_names(pd.DataFrame(), "t") == set())
    check("_utility_in_inventory keeps grenades only",
          fx._utility_in_inventory({"inventory": ["weapon_ak47", "weapon_flashbang", "weapon_smokegrenade", "weapon_knife"]})
          == ["weapon_flashbang", "weapon_smokegrenade"])
    rounds = pd.DataFrame([{"round_num": 2, "winner": "t"}, {"round_num": 1, "winner": "ct"}, {"round_num": 3, "winner": "ct"}])
    check("_build_round_history: earlier rounds, sorted",
          fx._build_round_history(rounds, 3) == [{"round": 1, "winner": "ct"}, {"round": 2, "winner": "t"}])

    dmg = pd.DataFrame([{"tick": 100, "attacker_name": "A", "victim_name": "C", "weapon": "ak47", "dmg_health": 30},
                        {"tick": 200, "attacker_name": "A", "victim_name": "C", "weapon": "hegrenade", "dmg_health": 45},
                        {"tick": 300, "attacker_name": "B", "victim_name": "D", "weapon": "ak47", "dmg_health": 10}])
    check("_damage_after sums the attacker's damage strictly after the tick", fx._damage_after(dmg, 100, "A") == 45)
    check("_damage_after: no damage column -> None (not 0)", fx._damage_after(dmg.drop(columns=["dmg_health"]), 0, "A") is None)
    check("_damage_after: empty -> 0", fx._damage_after(dmg.iloc[0:0], 0, "A") == 0)
    g = fx._grenade_damage(150, "A", "HE", dmg, thrower_side="ct", name_to_side={"C": "t", "D": "t"}, tick_rate=64)
    check("_grenade_damage: HE damage in the 2 s window, split by team",
          g == {"damage_dealt": 45, "enemies_damaged": 1, "teammates_damaged": 0}, str(g))
    check("_grenade_damage: not a damaging grenade -> {}", fx._grenade_damage(0, "A", "Smoke", dmg) == {})

    section("sound and sight helpers")
    foot = pd.DataFrame([{"tick": 90, "name": "C", "side": "t", "X": 600.0, "Y": 0.0}])
    shots = pd.DataFrame([{"tick": 95, "name": "C", "side": "t", "X": 1000.0, "Y": 0.0, "weapon": "usp_silencer"},
                          {"tick": 96, "name": "D", "side": "t", "X": -1000.0, "Y": 0.0, "weapon": "ak47"}])
    heard = fx._heard_enemy(100, "A", "ct", [0.0, 0.0, 0.0], 0.0, foot, shots, None, pd.DataFrame(), tick_rate=64)
    kinds = {(h["type"], h["direction"]) for h in heard["heard_sounds"]}
    check("footstep within range is heard, to the front, far",
          {"type": "footstep", "direction": "front", "distance": "far"} in heard["heard_sounds"], str(heard["heard_sounds"]))
    check("a silenced shot at 1000 u is NOT heard; an unsilenced one behind is",
          ("gunshot_silenced", "front") not in kinds and ("gunshot", "back") in kinds, str(kinds))
    check("heard_enemy flag set", heard["heard_enemy"] is True)
    check("nothing recent -> nothing heard",
          fx._heard_enemy(10_000, "A", "ct", [0.0, 0.0, 0.0], 0.0, foot, shots, None, pd.DataFrame(), tick_rate=64)["heard_enemy"] is False)
    check("missing position -> nothing heard", fx._heard_enemy(100, "A", "ct", None, 0.0, foot, shots, None, pd.DataFrame())["heard_enemy"] is False)

    landings = pd.DataFrame([{"tick": 0, "name": "C", "side": "t", "velocity_Z": -300.0, "X": 1, "Y": 2, "Z": 3},
                             {"tick": 8, "name": "C", "side": "t", "velocity_Z": 0.0, "X": 4, "Y": 5, "Z": 6}])
    check("_precompute_landings detects a fall ending", fx._precompute_landings(landings, "t") == [(8, 4.0, 5.0, 6.0)])
    check("_precompute_landings: missing columns -> []", fx._precompute_landings(landings.drop(columns=["velocity_Z"]), "t") == [])

    cache = {100: {"A": [{"name": "C", "side": "t", "position": [1, 1, 1]}]},
             150: {"A": [{"name": "C", "side": "t", "position": [2, 2, 2]}, {"name": "D", "side": "t", "position": [3, 3, 3]}]}}
    seen = fx._lookup_spotted(cache, "A", 160, tick_rate=64)
    check("_lookup_spotted: most recent position per enemy within the window",
          {s["name"]: s["position"] for s in seen} == {"C": [2, 2, 2], "D": [3, 3, 3]})
    check("_lookup_spotted never looks forward", fx._lookup_spotted(cache, "A", 90) == [])
    check("_lookup_spotted: window scales with tick rate (640 ticks at 64, 1280 at 128)",
          fx._lookup_spotted(cache, "A", 100 + 700, tick_rate=64) == []
          and len(fx._lookup_spotted(cache, "A", 100 + 700, tick_rate=128)) == 2)

    tables = make_tables(64)
    vis = fx._precompute_visibility(tables["ticks"], AlwaysVisible(), smokes_df=tables["smokes"])
    check("_precompute_visibility samples every 16 ticks", vis and all(t % 16 == 0 for t in vis))
    pre = vis[max(t for t in vis if t < SMOKE)]
    check("before the smoke, A sees both Ts", {e["name"] for e in pre["A"]} == {"C", "D"}, str(pre.get("A")))
    live = vis[min(t for t in vis if t >= SMOKE + 128 + 16)]
    check("with the smoke up on the A<->T line, A sees nobody", "A" not in live, str(live.get("A")))
    check("None checker -> empty cache", fx._precompute_visibility(tables["ticks"], None) == {})

    flash_ticks = pd.DataFrame([{"tick": 10, "name": "C", "side": "t", "flash_duration": 2.0},
                                {"tick": 12, "name": "C", "side": "t", "flash_duration": 1.0},
                                {"tick": 11, "name": "B", "side": "ct", "flash_duration": 0.5},
                                {"tick": 11, "name": "D", "side": "t", "flash_duration": 0.0}])
    fl = fx._flash_impact(0, flash_ticks, "ct", "t", tick_rate=64)
    check("_flash_impact counts unique flashed players per side with peak durations",
          fl == {"enemies_flashed": 1, "teammates_flashed": 1, "avg_enemy_flash_duration": 2.0}, str(fl))

    section("attribution_window")
    check("smoke window honours parser ticks", fx.attribution_window("Smoke", 10, 64, 100, 200)
          == {"impact_kind": fx.KIND_DURATIVE, "window_start_tick": 100, "window_end_tick": 200})
    check("HE is instantaneous at the throw tick", fx.attribution_window("HE", 10, 64) == {"impact_kind": fx.KIND_INSTANT, "resolve_tick": 10})

    section("extract_events over the synthetic round")
    events = fx.extract_events(tables, "syn", MAP_NAME, vis_checker=None)
    by = _by_type(events)
    check("every event type is produced", set(by) == {"buy", "engagement", "utility", "rotation", "bomb"}, str(sorted(by)))
    check("every event names the match, map and round",
          all(e["match_id"] == "syn" and e["map"] == MAP_NAME and e["round_num"] == 1 for e in events))
    check("round 2 (no winner) produced nothing", all(e["round_num"] == 1 for e in events))
    check("round_won follows the actor's side",
          all(e["round_won"] == (1 if e["player_side"] == "ct" else 0) for e in events if e["player_side"] in ("ct", "t")))
    check("every event is classified for attribution", all(e["action"].get("impact_kind") for e in events))

    buys = by["buy"]
    check("one buy per player at t=0, round-level", len(buys) == 4 and {b["time_into_round"] for b in buys} == {0.0}
          and all(b["action"]["impact_kind"] == fx.KIND_ROUND for b in buys))
    check("buy carries equipment and round type", buys[0]["action"]["equip_value"] == 4000 and buys[0]["situation"]["round_type"] == "half_buy")
    check("buy weapons exclude the knife", all("weapon_knife" not in b["action"]["weapons"] for b in buys))

    eng = by["engagement"]
    check("two kills -> attacker + victim rows each", len(eng) == 4)
    k1 = [e for e in eng if e["action"]["resolve_tick"] == KILL_1]
    att = next(e for e in k1 if e["action"]["role"] == "attacker")
    vic = next(e for e in k1 if e["action"]["role"] == "victim")
    check("kill time in seconds from freeze end (2 dp)", close(att["time_into_round"], (KILL_1 - FREEZE_END) / 64, 0.01))
    check("sides and results", (att["player_side"], vic["player_side"], att["outcome"]["result"], vic["outcome"]["result"]) == ("ct", "t", "kill", "death"))
    check("victim trade_kill is N/A (None), attacker with nobody to trade is None too",
          vic["action"]["trade_kill"] is None and att["action"]["trade_kill"] is None)
    check("weapon, distance and headshot propagate", (att["action"]["weapon"], att["action"]["distance"], att["action"]["headshot"]) == ("ak47", 1000.0, True))
    check("attacker heard the enemy footsteps just before the kill",
          att["situation"]["heard_enemy"] is True and any(s["type"] == "footstep" for s in att["situation"]["heard_sounds"]))
    # The snapshot is taken AT the kill tick, where the victim is already down.
    check("situation counts at the first kill: 1 teammate, 1 enemy still up",
          (att["situation"]["alive_teammates"], att["situation"]["alive_enemies"]) == (1, 1), str((att["situation"]["alive_teammates"], att["situation"]["alive_enemies"])))
    check("pre-plant at the first kill, post-plant at the second",
          att["situation"]["post_plant"] is False
          and all(e["situation"]["post_plant"] is True for e in eng if e["action"]["resolve_tick"] == KILL_2))
    check("damage_after is a number (the column was resolved), not None", att["outcome"]["damage_after"] == 0)
    k2_att = next(e for e in eng if e["action"]["resolve_tick"] == KILL_2 and e["action"]["role"] == "attacker")
    check("the round-ending kill: no enemies left at that tick, and not a clutch (a teammate is alive)",
          k2_att["situation"]["alive_enemies"] == 0 and k2_att["situation"]["clutch"] is False)

    util = by["utility"]
    smoke = next(e for e in util if e["action"]["grenade_type"] == "Smoke")
    he    = next(e for e in util if e["action"]["grenade_type"] == "HE")
    check("exactly one smoke and one HE", len(util) == 2)
    check("smoke is durative with a nominal window from the throw tick",
          smoke["action"]["impact_kind"] == fx.KIND_DURATIVE
          and smoke["action"]["window_start_tick"] == SMOKE + fx._ticks(fx.SMOKE_POP_S, 64)
          and smoke["action"]["window_end_tick"] == smoke["action"]["window_start_tick"] + fx._ticks(fx.SMOKE_LIFE_S, 64))
    check("smoke thrower side resolved from the tick table", smoke["player_side"] == "ct")
    check("smoke landing position comes from the projectile's last row", smoke["action"]["land_position"] == [500.0, 0.0, 0.0])
    check("smoke impact block present with no enemies near the landing",
          smoke["outcome"]["enemies_at_pop"] == 0 and "enemies_stopped_or_reversed" in smoke["outcome"])
    check("HE is instantaneous at its throw tick", he["action"] ["impact_kind"] == fx.KIND_INSTANT and he["action"]["resolve_tick"] == HE)
    check("HE damage attributed to the thrower against the right team",
          he["outcome"]["damage_dealt"] == 30 and he["outcome"]["enemies_damaged"] == 1 and he["outcome"]["teammates_damaged"] == 0, str(he["outcome"]))
    check("HE followed by a kill within 5 s", he["outcome"]["kill_within_5s"] is True and smoke["outcome"]["kill_within_5s"] is False)

    rot = by["rotation"]
    check("B's constant movement produces a rotation roughly every second", 40 <= len(rot) <= 60, str(len(rot)))
    check("rotations are durative over [prev sample, this sample]",
          all(r["action"]["impact_kind"] == fx.KIND_DURATIVE and r["action"]["window_start_tick"] < r["action"]["window_end_tick"] for r in rot))
    check("running is audible", all(r["action"]["made_sound"] for r in rot) and rot[5]["action"]["movement_type"][0]["type"] == "running")
    check("rotation distance ~250 u per second", all(240 <= r["action"]["distance_moved"] <= 260 for r in rot))
    check("only CT rotations (the Ts stood still)", {r["player_side"] for r in rot} == {"ct"})

    bomb = by["bomb"]
    check("one plant, by T, at the plant tick",
          len(bomb) == 1 and bomb[0]["player_side"] == "t" and bomb[0]["action"]["bomb_action"] == "plant"
          and bomb[0]["action"]["resolve_tick"] == PLANT and bomb[0]["action"]["site"] == "A")

    section("tick rate: same story at 128 tick")
    ev128 = fx.extract_events(make_tables(128), "syn", MAP_NAME, vis_checker=None)
    by128 = _by_type(ev128)
    check("same event counts per type at 128 tick",
          {k: len(v) for k, v in by128.items()} == {k: len(v) for k, v in by.items()},
          f"{ {k: len(v) for k, v in by128.items()} } vs { {k: len(v) for k, v in by.items()} }")
    a128 = next(e for e in by128["engagement"] if e["action"]["role"] == "attacker" and e["action"]["resolve_tick"] == 2 * KILL_1)
    check("kill time in SECONDS is unchanged", close(a128["time_into_round"], att["time_into_round"]))
    s128 = next(e for e in by128["utility"] if e["action"]["grenade_type"] == "Smoke")
    check("smoke window spans twice the ticks (same seconds)",
          (s128["action"]["window_end_tick"] - s128["action"]["window_start_tick"])
          == 2 * (smoke["action"]["window_end_tick"] - smoke["action"]["window_start_tick"]))
    check("the kill's footsteps are still heard (10 s window scaled to ticks)", a128["situation"]["heard_enemy"] is True)

    section("degenerate inputs")
    check("no ticks -> no events", fx.extract_events({"rounds": tables["rounds"], "ticks": pd.DataFrame()}, "x") == [])
    check("no rounds -> no events", fx.extract_events({"rounds": pd.DataFrame(), "ticks": tables["ticks"]}, "x") == [])

    return summary()


if __name__ == "__main__":
    sys.exit(main())
