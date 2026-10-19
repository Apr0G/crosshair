"""
RawTables contract (P2.5.1, Amendment 1): A.1 columns and dtypes, `validate()`,
`from_parsed()`, and the causal `slice_until()`. Runs on the synthetic stories. The
golden-demo half of P2.5.1 runs inside the golden gates: test_states_golden.py and
test_actions_golden.py parse the pinned golden demo through `RawTables.from_parsed`
(both exit 2 until a golden demo is pinned); test_awpy3_parser.py <demo> runs the same
contract checks on any real demo.

    .venv\\Scripts\\python.exe test\\engine\\test_raw_schema.py
"""
import sys

import polars as pl

from _harness import check, section, summary
from _synthetic import STORIES, make_raw_tables

from engine.parse.schema import (META_KEYS, OPTIONAL_TABLES, TABLE_SCHEMAS, TABLES,
                                 RawTables, SchemaError)


def raises_schema_error(fn) -> bool:
    try:
        fn()
    except SchemaError:
        return True
    return False


# ── contract shape ──
section("contract shape")
check("12 A.1 tables", len(TABLES) == 12 and set(TABLES) == {
    "rounds", "ticks", "kills", "damages", "grenades", "smokes", "infernos", "flashes",
    "bomb", "weapon_fire", "footsteps", "items"})
check("META_KEYS", META_KEYS == ("map", "tick_rate", "game_version", "server", "demo_sha256"))
check("every dtype is a polars DataType",
      all(isinstance(dt, pl.DataType) or isinstance(dt(), pl.DataType)
          for sch in TABLE_SCHEMAS.values() for dt in sch.values()))
meta = {"map": "de_test", "tick_rate": 128, "game_version": None, "server": None, "demo_sha256": None}
check("empty() validates", RawTables.empty(meta).validate() == [])

# ── every story validates ──
section("stories validate (64 and 128 tick)")
raws = {}
for story in STORIES:
    for tr in (64, 128):
        raw = make_raw_tables(story, tr)
        raws[(story, tr)] = raw
        check(f"{story}@{tr} validates strict", raw.validate(strict=True) == [])
        exact = all(raw.frames()[t][c].dtype == dt
                    for t, sch in TABLE_SCHEMAS.items() for c, dt in sch.items())
        check(f"{story}@{tr} dtypes exactly A.1", exact)

base = raws[("base", 128)]

# ── problems are caught ──
section("validate catches problems")
broken = RawTables(**{**base.frames(), "kills": base.kills.drop("headshot")}, meta=base.meta)
probs = broken.validate(strict=False)
check("dropped column reported", any("kills: missing column 'headshot'" in p for p in probs), str(probs))
check("dropped column raises when strict", raises_schema_error(broken.validate))

broken = RawTables(**{**base.frames(), "ticks": base.ticks.with_columns(pl.col("health").cast(pl.String))},
                   meta=base.meta)
probs = broken.validate(strict=False)
check("wrong dtype (str health) reported", any(p.startswith("ticks.health") for p in probs), str(probs))
broken = RawTables(**{**base.frames(), "kills": base.kills.with_columns(pl.col("distance").cast(pl.Int64))},
                   meta=base.meta)
check("int distance is castable (int -> f32)", broken.validate(strict=False) == [])
broken = RawTables(**{**base.frames(), "rounds": base.rounds.with_columns(pl.col("end_tick") + 0.5)},
                   meta=base.meta)
check("float tick not castable to i64", any("rounds.end_tick" in p for p in broken.validate(strict=False)))
neg = base.ticks.with_columns(pl.col("steamid").cast(pl.Int64) * -1)
broken = RawTables(**{**base.frames(), "ticks": neg}, meta=base.meta)
check("negative steamid not castable to u64", any("ticks.steamid" in p for p in broken.validate(strict=False)))
broken = RawTables(**base.frames(), meta={k: v for k, v in base.meta.items() if k != "tick_rate"})
check("missing meta tick_rate reported", any("tick_rate" in p for p in broken.validate(strict=False)))
extra = RawTables(**{**base.frames(), "kills": base.kills.with_columns(pl.lit(1).alias("whatever"))},
                  meta=base.meta)
check("extra columns allowed", extra.validate() == [])

# ── from_parsed ──
section("from_parsed")
parsed = {t: df for t, df in base.frames().items() if t not in OPTIONAL_TABLES}
parsed["kills"] = parsed["kills"].drop("is_trade").with_columns(pl.col("round_num").cast(pl.Int64))
r = RawTables.from_parsed(parsed, base.meta)
check("missing optional tables become empty frames",
      all(getattr(r, t).height == 0 and dict(getattr(r, t).schema) == TABLE_SCHEMAS[t] for t in OPTIONAL_TABLES))
check("castable column cast to A.1 dtype", r.kills["round_num"].dtype == pl.Int32)
check("missing optional column added as null", r.kills["is_trade"].null_count() == r.kills.height)
check("from_parsed result validates", r.validate() == [])
check("missing required table raises",
      raises_schema_error(lambda: RawTables.from_parsed({"ticks": base.ticks}, base.meta)))

# ── slice_until ──
section("slice_until (causal)")
s = base.slice_until(1700)
check("1700: validates", s.validate() == [])
check("1700: no kills", s.kills.height == 0)
check("1700: no ticks after 1700", s.ticks["tick"].max() == 1700)
check("1700: smoke present", s.smokes.height == 1)
check("1700: smoke end_tick nulled", s.smokes["end_tick"].null_count() == 1)
check("1700: round 2 not started -> dropped", s.rounds["round_num"].to_list() == [1])
r1 = s.rounds.row(0, named=True)
check("1700: round 1 winner/end nulled",
      all(r1[c] is None for c in ("end_tick", "official_end_tick", "winner_side", "end_reason", "bomb_planted")))
check("1700: round 1 start/freeze_end kept", (r1["start_tick"], r1["freeze_end_tick"]) == (900, 1000))
g = {row["entity_id"]: row for row in s.grenades.iter_rows(named=True)}
check("1700: HE thrown at 1700 kept, detonation nulled",
      102 in g and g[102]["detonate_tick"] is None and g[102]["x"] is None)
check("1700: smoke grenade expire nulled", g[101]["expire_tick"] is None and g[101]["detonate_tick"] == 1500)
check("1700: HE damage at 1800 absent", s.damages.height == 0)
check("1700: meta kept", s.meta == base.meta)

s = base.slice_until(4100)
check("4100: both kills", s.kills.height == 2)
check("4100: smoke has its real end", s.smokes["end_tick"].to_list() == [1500 + 18 * 128])
r1 = s.rounds.row(0, named=True)
check("4100: round 1 ended -> winner kept", r1["winner_side"] == "ct" and r1["end_tick"] == 4040)
check("4100: official end 4360 still in the future -> nulled", r1["official_end_tick"] is None)
check("4100: round 1 only", s.rounds.height == 1)

s = base.slice_until(5000)
check("5000: round 2 started, its end nulled",
      s.rounds.height == 2 and s.rounds["winner_side"].to_list() == ["ct", None])
check("5000: round 1 official end known", s.rounds["official_end_tick"].to_list()[0] == 4360)

fire = pl.DataFrame([{"entity_id": 9, "start_tick": 1000, "end_tick": 2000, "x": 0.0, "y": 0.0, "z": 0.0,
                      "extent_radius": 120.0, "thrower_steamid": 2001}], schema=TABLE_SCHEMAS["infernos"])
withfire = RawTables(**{**base.frames(), "infernos": fire}, meta=base.meta)
check("inferno burning at 1500: kept, end nulled", withfire.slice_until(1500).infernos["end_tick"].to_list() == [None])
check("inferno not started at 900: absent", withfire.slice_until(900).infernos.height == 0)
check("inferno over at 2500: end kept", withfire.slice_until(2500).infernos["end_tick"].to_list() == [2000])

fl = raws[("flash", 128)]
check("flash rows appear at their detonate tick",
      (fl.slice_until(1899).flashes.height, fl.slice_until(1900).flashes.height) == (0, 2))

sys.exit(summary())
