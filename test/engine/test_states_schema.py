"""
states.parquet schema (P3.8.1) on builder output: every column name and dtype against
`engine/state/schema.py`, label columns, `rows_to_frame` and parquet round trips, and
the SCHEMA_CHANGELOG entry for the current schema_version.

    .venv\\Scripts\\python.exe test\\engine\\test_states_schema.py
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from _harness import check, section, summary
from _synthetic import make_raw_tables

from engine import config
from engine.state import builder
from engine.state.geometry import FlatGeometry
from engine.state.schema import (CTX_DTYPES, LABEL_PREFIX, N_PLAYERS, PAIR_CHANNELS, PLAYER_DTYPES,
                                 SCHEMA_VERSION, STATE_COLUMNS, UTILITY_DTYPES, frame_to_snapshots,
                                 rows_to_frame)

CFG = config.load()
CHANGELOG = Path(builder.__file__).resolve().parent / "SCHEMA_CHANGELOG.md"
LABELS = {"y_round_won_ct", "y_round_end_reason", "y_dies_within_5s", "y_plant_within_15s",
          "y_defuse_within_15s", "y_time_to_next_engagement_s", "y_is_terminal"}


def close_snap(a, b, tol=1e-3) -> bool:
    """Equal up to float32 rounding of float fields."""
    def eq(x, y):
        if isinstance(x, float) and isinstance(y, float):
            return (math.isnan(x) and math.isnan(y)) or abs(x - y) <= tol * max(1.0, abs(x))
        return x == y
    from dataclasses import asdict
    for pa, pb in zip(a.players + a.utility + [a.ctx], b.players + b.utility + [b.ctx]):
        da, db = asdict(pa), asdict(pb)
        if da.keys() != db.keys() or not all(eq(da[k], db[k]) for k in da):
            return False
    return (len(a.utility) == len(b.utility) and a.meta == b.meta
            and np.allclose(a.pair, b.pair, rtol=1e-6, equal_nan=True) and a.labels == b.labels)


raw   = make_raw_tables("flash", 128)
snaps = builder.build(raw, FlatGeometry(), CFG, match_key="schema", tier="faceit_10", debug=True)
df    = builder.to_frame(snaps)

# ── columns and dtypes ──
section("columns and dtypes")
want = list(STATE_COLUMNS)
check("every STATE_COLUMNS column present, in order", df.columns[:len(want)] == want,
      str(set(want) ^ set(df.columns[:len(want)])))
bad = {c: (df.schema[c], dt) for c, dt in STATE_COLUMNS.items() if df.schema[c] != dt}
check("every column has its schema dtype", not bad, str(bad))
check("only label columns beyond STATE_COLUMNS, all y_*", set(df.columns[len(want):]) == LABELS,
      str(df.columns[len(want):]))
check("players: list of structs with the A.2 fields and dtypes",
      df.schema["players"] == pl.List(pl.Struct(PLAYER_DTYPES)))
check("utility: list of structs with the A.3 fields", df.schema["utility"] == pl.List(pl.Struct(UTILITY_DTYPES)))
check("ctx_* flat with A.4 dtypes", all(df.schema["ctx_" + k] == v for k, v in CTX_DTYPES.items()))
check("pair: fixed 10x10x5 float32", df.schema["pair"] == pl.Array(pl.Float32, N_PLAYERS * N_PLAYERS * len(PAIR_CHANNELS)))
check("one row per snapshot", df.height == len(snaps))
check("schema_version column = SCHEMA_VERSION", df["schema_version"].unique().to_list() == [SCHEMA_VERSION])
check("10 players per row", df["players"].list.len().unique().to_list() == [N_PLAYERS])
check("kinds is a non-empty list per row", df["kinds"].list.len().min() >= 1)
check("label dtypes", df.schema["y_dies_within_5s"] == pl.List(pl.Int64) and df.schema["y_round_end_reason"] == pl.String
      and df.schema["y_time_to_next_engagement_s"] == pl.Float64)
check("labels dropped at inference by prefix leave exactly STATE_COLUMNS",
      [c for c in df.columns if not c.startswith(LABEL_PREFIX)] == want)

# ── round trips ──
section("round trips")
back = frame_to_snapshots(df)
check("frame -> snapshots: same count, equal up to float32 storage",
      len(back) == len(snaps) and all(close_snap(a, b) for a, b in zip(snaps, back)))
again = frame_to_snapshots(rows_to_frame([s.to_row() for s in back]))
check("second trip is exact (idempotent)", again == back)
with tempfile.TemporaryDirectory() as d:
    p = builder.write(snaps, Path(d) / "states.parquet")
    disk = pl.read_parquet(p)
    check("parquet write/read keeps schema and rows", disk.schema == df.schema and disk.equals(df))
    check("parquet -> snapshots exact vs the in-memory frame", frame_to_snapshots(disk) == back)
check("empty build writes an empty frame with the schema", builder.to_frame([]).columns == want)

# ── changelog ──
section("SCHEMA_CHANGELOG")
text = CHANGELOG.read_text(encoding="utf-8") if CHANGELOG.is_file() else ""
check("SCHEMA_CHANGELOG.md exists", bool(text))
check(f"has an entry for schema_version {SCHEMA_VERSION}", f"## v{SCHEMA_VERSION}" in text)

sys.exit(summary())
