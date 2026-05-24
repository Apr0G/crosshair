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

    tables: dict[str, "pd.DataFrame"] = {
        "map_name": getattr(demo, "map_name", None) or getattr(demo, "header", {}).get("map_name", "unknown"),
        "rounds":   to_pd(demo.rounds),
        "ticks":    to_pd(demo.ticks),
        "kills":    to_pd(demo.kills),
        "damages":  to_pd(demo.damages),
        "shots":    to_pd(demo.shots),
        "grenades": to_pd(demo.grenades),
    }

    for name, attr in [("smokes", "smokes"), ("infernos", "infernos"),
                       ("bomb", "bomb"), ("footsteps", "footsteps")]:
        try:
            tables[name] = to_pd(getattr(demo, attr))
        except Exception:
            pass

    for event_name, event_df in demo.events.items():
        tables[f"event_{event_name}"] = to_pd(event_df)

    return tables


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract.py path/to/match.dem")
        sys.exit(1)

    tables = extract(sys.argv[1])
    for name, df in tables.items():
        print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(df.head(3))
        print()
