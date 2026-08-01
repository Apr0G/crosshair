"""
Demo parsing wrapper around awpy.

Loads a .dem file and returns a dict of pandas DataFrames:
  rounds, ticks, kills, damages, shots, grenades, smokes, infernos, bomb,
  footsteps, plus one DataFrame per game event (event_<name>).

Usage:
    python src/extract.py path/to/match.dem
"""

import sys
from pathlib import Path

import polars as pl
from awpy.demo import Demo


def extract(demo_path: str) -> dict[str, "pd.DataFrame"]:
    import pandas as pd

    path = Path(demo_path)
    if not path.exists():
        raise FileNotFoundError(f"Demo not found: {demo_path}")

    print(f"Parsing {path.name} ...")
    demo = Demo(path=path, verbose=False)
    demo.parse(
        player_props=[
            "cash",
            "cash_spent_this_round",
            "total_cash_spent",
            "armor_value",
            "has_helmet",
            "has_defuser",
            "inventory",
            "current_equip_value",
            "health",
            "is_alive",
            "velocity_X",
            "velocity_Y",
            "velocity_Z",
            "pitch",
            "yaw",
            "flash_duration",
            "zoom_lvl",
            "accuracy_penalty",
            "ping",
        ]
    )
    print("Done.\n")

    def to_pd(df: pl.DataFrame) -> "pd.DataFrame":
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
        print(f"  WARNING: demo tickrate is {tick_rate}, not 64. Time-based features "
              f"assume 64 and will be wrong by a factor of {tick_rate / 64:.2f}.")

    for name, attr in [("smokes", "smokes"), ("infernos", "infernos"),
                       ("bomb", "bomb"), ("footsteps", "footsteps")]:
        try:
            tables[name] = to_pd(getattr(demo, attr))
        except AttributeError as e:
            # A genuinely absent optional table. Anything else (a parser error, a
            # renamed column) must NOT be swallowed — it silently degrades six-plus
            # model features to constants while the run still prints "done".
            print(f"  [warn] table '{name}' unavailable: {e}")

    for event_name, event_df in demo.events.items():
        tables[f"event_{event_name}"] = to_pd(event_df)

    return tables


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract.py path/to/match.dem")
        sys.exit(1)

    tables = extract(sys.argv[1])
    for name, df in tables.items():
        # map_name and tick_rate are scalars, not frames.
        if not hasattr(df, "shape"):
            print(f"{name}: {df}\n")
            continue
        print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(df.head(3))
        print()
