"""
Demo parsing wrapper around awpy.

Loads a .dem file and returns a dict of pandas DataFrames:
  rounds, ticks, kills, damages, shots, grenades, smokes, infernos, bomb,
  footsteps, plus one DataFrame per game event (event_<name>),
and two scalars: map_name and tick_rate.

To eyeball what a demo yields: `.venv/bin/python test/test_local_demo.py <demo>`.
"""
from pathlib import Path

import pandas as pd
import polars as pl
from awpy.demo import Demo

# awpy 2.0.2's default event list asks for `player_sound`, which CS2 renamed to
# `player_footstep`. Leaving it to the default makes demo.parse() fail on every
# modern demo — as a KeyError on demoparser2 0.41.2/0.41.3, an EntityNotFound on
# 0.40.x/0.41.0/0.41.1, and a hard crash on 0.41.4 (an unsigned-integer underflow
# in the Rust parser: it tries to allocate 2**64-1 bytes). Passing the list
# explicitly, with the current name, sidesteps all of it.
EVENTS = [
    "bomb_defused", "bomb_dropped", "bomb_exploded", "bomb_pickup", "bomb_planted",
    "flashbang_detonate", "hegrenade_detonate", "inferno_expire", "inferno_startburn",
    "item_pickup", "player_death", "player_given_c4", "player_hurt", "player_footstep",
    "player_spawn", "round_freeze_end", "round_officially_ended",
    "smokegrenade_detonate", "smokegrenade_expired", "weapon_fire",
]

PLAYER_PROPS = [
    "cash", "cash_spent_this_round", "total_cash_spent",
    "armor_value", "has_helmet", "has_defuser", "inventory", "current_equip_value",
    "health", "is_alive",
    "velocity_X", "velocity_Y", "velocity_Z", "pitch", "yaw",
    "flash_duration", "zoom_lvl", "accuracy_penalty", "ping",
]


def extract(demo_path: str) -> dict[str, object]:
    path = Path(demo_path)
    if not path.exists():
        raise FileNotFoundError(f"Demo not found: {demo_path}")

    print(f"Parsing {path.name} ...")
    demo = Demo(path=path, verbose=False)
    demo.parse(events=EVENTS, player_props=PLAYER_PROPS)
    print("Done.\n")

    def to_pd(df: pl.DataFrame) -> pd.DataFrame:
        return df.to_pandas()

    header = getattr(demo, "header", None) or {}
    tick_rate = getattr(demo, "tickrate", None) or header.get("tickrate") or 64

    tables: dict[str, object] = {
        "map_name":  getattr(demo, "map_name", None) or header.get("map_name", "unknown"),
        "tick_rate": int(tick_rate),
        "rounds":    to_pd(demo.rounds),
        "ticks":     to_pd(demo.ticks),
        "kills":     to_pd(demo.kills),
        "damages":   to_pd(demo.damages),
        "shots":     to_pd(demo.shots),
        "grenades":  to_pd(demo.grenades),
    }

    if int(tick_rate) != 64:
        # Informational, not a warning: every tick-valued window downstream is
        # derived from this number rather than assuming 64.
        print(f"  tickrate: {tick_rate} (all time windows scale to it)")

    # demo.footsteps also reads the old `player_sound` key, so build it from the
    # renamed event directly rather than going through the property.
    try:
        tables["footsteps"] = to_pd(demo.events["player_footstep"])
    except (KeyError, AttributeError) as e:
        print(f"  [warn] table 'footsteps' unavailable: {e}")

    for name in ("smokes", "infernos", "bomb"):
        try:
            tables[name] = to_pd(getattr(demo, name))
        except AttributeError as e:
            # A genuinely absent optional table. Anything else (a parser error, a
            # renamed column) must NOT be swallowed — it silently degrades six-plus
            # model features to constants while the run still prints "done".
            print(f"  [warn] table '{name}' unavailable: {e}")

    for event_name, event_df in demo.events.items():
        tables[f"event_{event_name}"] = to_pd(event_df)

    return tables
