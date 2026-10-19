"""
A tiny synthetic "parsed demo" — the dict of tables extract.extract() returns —
so the extractor, the sampler and the seam can be tested without awpy or a .dem.

One playable round on a fictional map, 2v2:

    tick (64-tick units)   what
    1000  FREEZE_END       round clock starts; everyone in spawn
    1500  SMOKE            A (ct) throws a smoke that lands at (500, 0)
    1800  HE               C (t) throws an HE that lands at (600, 0); hurts A for 30
    2000  KILL_1           A (ct) kills C (t)            -> 2v1
    3000  PLANT            D (t) plants at site A
    4000  KILL_2           B (ct) kills D (t)            -> round decided, CT win
    4040  END / 4360 OFFICIAL_END

B walks the whole round at 250 u/s, so rotation events fire every second.
A second round with no resolvable winner is included so skip paths are exercised.

make_tables(tick_rate) scales every tick by tick_rate / 64, so the same real-time
story can be told at 128 tick and the tick-rate handling checked against it.
"""
import numpy as np
import pandas as pd

PLAYERS = {"A": "ct", "B": "ct", "C": "t", "D": "t"}

# 64-tick reference timeline
FREEZE_END = 1000
SMOKE      = 1500
HE         = 1800
KILL_1     = 2000
PLANT      = 3000
KILL_2     = 4000
END        = KILL_2 + 40
OFFICIAL   = END + 320
STEP       = 4             # tick-table resolution (divides every event tick above)

MAP_NAME = "de_test"


def make_tables(tick_rate: int = 64) -> dict:
    k = tick_rate / 64
    T = lambda t: int(round(t * k))

    # ── ticks: one row per player per STEP ticks ───────────────────────────────
    rows = []
    for tick in range(T(FREEZE_END), T(OFFICIAL) + 1, T(STEP)):
        s = (tick - T(FREEZE_END)) / tick_rate          # seconds into round
        for name, side in PLAYERS.items():
            alive = not ((name == "C" and tick >= T(KILL_1)) or (name == "D" and tick >= T(KILL_2)))
            if name == "A":
                pos, vel = (0.0, 0.0, 0.0), 0.0
                inv = ["weapon_ak47", "weapon_knife"] + (["weapon_smokegrenade"] if tick < T(SMOKE) else [])
            elif name == "B":
                pos, vel = (300.0 + 250.0 * s, 0.0, 0.0), 250.0
                inv = ["weapon_m4a1", "weapon_flashbang", "weapon_knife"]
            elif name == "C":
                pos, vel = (1000.0, 0.0, 0.0), 0.0
                inv = ["weapon_ak47", "weapon_knife"] + (["weapon_hegrenade"] if tick < T(HE) else [])
            else:
                pos, vel = (1200.0, 100.0, 0.0), 0.0
                inv = ["weapon_ak47", "weapon_knife"] + (["weapon_c4"] if tick < T(PLANT) else [])
            rows.append({
                "tick": tick, "name": name, "side": side, "is_alive": alive,
                "health": 100 if alive else 0, "armor_value": 100 if alive else 0,
                "has_helmet": True, "has_defuser": name == "B",
                "inventory": inv if alive else [],
                "current_equip_value": 4000 if alive else 0,
                "cash": 1000, "cash_spent_this_round": 3000,
                "X": pos[0], "Y": pos[1], "Z": pos[2],
                "velocity_X": vel if alive else 0.0, "velocity_Y": 0.0, "velocity_Z": 0.0,
                "pitch": 0.0, "yaw": 0.0, "flash_duration": 0.0,
                "zoom_lvl": 0, "accuracy_penalty": 0.0, "place": "Mid",
            })
    ticks = pd.DataFrame(rows)

    rounds = pd.DataFrame([
        {"round_num": 1, "freeze_end": T(FREEZE_END), "end": T(END),
         "official_end": T(OFFICIAL), "winner": "ct"},
        # no resolvable winner: every consumer must skip this round, never label it
        {"round_num": 2, "freeze_end": T(OFFICIAL) + 1000, "end": T(OFFICIAL) + 2000,
         "official_end": np.nan, "winner": None},
    ])

    def kill(tick, att, vic, weapon, att_pos, vic_pos):
        return {
            "tick": tick, "round_num": 1,
            "attacker_name": att, "attacker_side": PLAYERS[att],
            "attacker_X": att_pos[0], "attacker_Y": att_pos[1], "attacker_Z": att_pos[2],
            "attacker_place": "Mid", "attacker_health": 100, "attacker_armor": 100,
            "victim_name": vic, "victim_side": PLAYERS[vic],
            "victim_X": vic_pos[0], "victim_Y": vic_pos[1], "victim_Z": vic_pos[2],
            "victim_place": "Mid", "victim_health": 100, "victim_armor": 100,
            "weapon": weapon, "distance": float(abs(vic_pos[0] - att_pos[0])),
            "headshot": True, "thrusmoke": False, "attackerblind": False, "noscope": False,
        }
    kills = pd.DataFrame([
        kill(T(KILL_1), "A", "C", "ak47", (0, 0, 0), (1000, 0, 0)),
        kill(T(KILL_2), "B", "D", "m4a1", (300 + 250 * 46.875, 0, 0), (1200, 100, 0)),
    ])

    damages = pd.DataFrame([
        {"tick": T(HE) + T(30), "round_num": 1, "attacker_name": "C", "victim_name": "A",
         "weapon": "hegrenade", "dmg_health": 30},
        {"tick": T(KILL_1), "round_num": 1, "attacker_name": "A", "victim_name": "C",
         "weapon": "ak47", "dmg_health": 100},
    ])

    grenades = pd.DataFrame([
        {"tick": T(SMOKE),       "round_num": 1, "entity_id": 1, "thrower": "A",
         "grenade_type": "CSmokeGrenadeProjectile", "X": 0.0,    "Y": 0.0, "Z": 0.0},
        {"tick": T(SMOKE) + T(100), "round_num": 1, "entity_id": 1, "thrower": "A",
         "grenade_type": "CSmokeGrenadeProjectile", "X": 500.0,  "Y": 0.0, "Z": 0.0},
        {"tick": T(HE),          "round_num": 1, "entity_id": 2, "thrower": "C",
         "grenade_type": "CHEGrenadeProjectile",    "X": 1000.0, "Y": 0.0, "Z": 0.0},
        {"tick": T(HE) + T(40),  "round_num": 1, "entity_id": 2, "thrower": "C",
         "grenade_type": "CHEGrenadeProjectile",    "X": 600.0,  "Y": 0.0, "Z": 0.0},
    ])

    smokes = pd.DataFrame([{
        "round_num": 1, "entity_id": 1, "thrower_name": "A", "thrower_side": "ct",
        "start_tick": T(SMOKE) + T(128), "end_tick": T(SMOKE) + T(128) + T(18 * 64),
        "X": 500.0, "Y": 0.0, "Z": 0.0,
    }])
    infernos = pd.DataFrame(columns=["round_num", "entity_id", "thrower_name", "thrower_side",
                                     "thrower_X", "thrower_Y", "thrower_Z",
                                     "start_tick", "end_tick", "X", "Y", "Z"])

    bomb_planted = pd.DataFrame([{
        "tick": T(PLANT), "round_num": 1, "user_name": "D", "user_side": "t",
        "user_X": 1200.0, "user_Y": 100.0, "user_Z": 0.0, "user_yaw": 0.0,
        "site": "A", "user_place": "BombsiteA", "user_health": 100, "user_has_defuser": False,
    }])
    bomb_cols = list(bomb_planted.columns)

    # C walks around 600 units from A just before the kill: audible footsteps.
    footsteps = pd.DataFrame([
        {"tick": t, "name": "C", "side": "t", "X": 600.0, "Y": 0.0}
        for t in range(T(KILL_1) - T(96), T(KILL_1), T(STEP))
    ])
    shots = pd.DataFrame([
        {"tick": T(KILL_1), "name": "A", "side": "ct", "X": 0.0, "Y": 0.0, "weapon": "ak47"},
    ])

    return {
        "map_name":  MAP_NAME,
        "tick_rate": tick_rate,
        "rounds":    rounds,
        "ticks":     ticks,
        "kills":     kills,
        "damages":   damages,
        "grenades":  grenades,
        "smokes":    smokes,
        "infernos":  infernos,
        "footsteps": footsteps,
        "shots":     shots,
        "event_bomb_planted":  bomb_planted,
        "event_bomb_defused":  pd.DataFrame(columns=bomb_cols),
        "event_bomb_exploded": pd.DataFrame(columns=["tick", "site"]),
    }


class AlwaysVisible:
    """Stand-in for awpy's VisibilityChecker: every pair has line of sight, so the
    only thing that can hide a player is a smoke."""
    def is_visible(self, p, q) -> bool:
        return True
