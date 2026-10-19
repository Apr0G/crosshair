"""
actions.parquet / windows.parquet (P4.7.1, P4.5.2): columns and dtypes, params/enablers as
JSON strings, write -> read round trip, empty output keeps the schema, two runs give
byte-identical files, every params key set per Appendix A.5.

    .venv\\Scripts\\python.exe test\\engine\\test_actions_schema.py
"""
import json
import sys
import tempfile
from pathlib import Path

import polars as pl
from _harness import check, section, summary
from _synthetic import make_raw_tables

from engine import config
from engine.actions import extractor
from engine.actions.types import Action
from engine.state.geometry import FlatGeometry

CFG = config.load()
GEO = FlatGeometry()

# Appendix A.5 keys every action of the type must carry (extra keys are allowed)
A5 = {
    "kill":            {"weapon", "weapon_class", "headshot", "through_smoke", "attacker_blind", "victim_blind",
                        "wallbang", "noscope", "distance_units", "attacker_callout", "victim_callout",
                        "victim_health_before", "assister_id", "flash_assister_id", "victim_id"},
    "damage":          {"damage_total", "hits", "weapon_class", "armor_damage", "victim_id"},
    "flash":           {"blinded_enemies", "blinded_teammates", "blinded_self", "land_callout", "throw_callout",
                        "pop_delay_s"},
    "he":              {"enemies_damaged", "teammates_damaged", "damage_total", "land_callout"},
    "peek":            {"from_callout", "target_id", "target_callout", "target_spotted_me_first"},
    "plant_start":     {"site", "planter_callout", "ct_alive", "t_alive"},
    "plant_complete":  {"site", "planter_callout", "ct_alive", "t_alive"},
    "defuse_start":    {"kit", "ct_alive", "t_alive"},
    "defuse_complete": {"kit", "ct_alive", "t_alive"},
    "smoke":           {"land_callout", "throw_callout", "throw_tick", "pop_tick", "expire_tick", "blocks_pairs_at_pop"},
    "rotation":        {"from_callout", "to_callout", "nav_dist_units", "duration_s", "toward_bomb"},
    "hold":            {"callout", "duration_s", "facing_callout"},
    "save":            {"from_callout", "to_callout", "teammates_alive"},
    "buy":             {"loadout_class", "equip_value", "money_before", "money_after", "has_armor", "has_helmet",
                        "has_kit", "utility_count", "primary_class"},
    "team_buy":        {"team_class", "total_equip"},
    "drop":            {"weapon_class", "receiver_id"},
    "smoke_pop":       {"parent_action_id"},
    "smoke_fade":      {"parent_action_id"},
}

section("columns and dtypes")
check("actions columns = Action fields", list(extractor.ACTION_SCHEMA) == list(Action.__dataclass_fields__))
check("windows columns", list(extractor.WINDOW_SCHEMA) == ["window_id", "round_num", "start_tick", "end_tick",
                                                           "member_action_ids"])

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    all_acts = []
    for story in ("base", "flash", "trade", "retreat", "drop"):
        _, acts, ws = extractor.run(make_raw_tables(story), CFG, match_key=f"m-{story}", geometry=GEO)
        all_acts += acts
        pa, pw = extractor.write(acts, ws, d / story)
        df, dw = pl.read_parquet(pa), pl.read_parquet(pw)
        check(f"{story}: actions dtypes", dict(df.schema) == extractor.ACTION_SCHEMA, str(df.schema))
        check(f"{story}: windows dtypes", dict(dw.schema) == extractor.WINDOW_SCHEMA, str(dw.schema))
        check(f"{story}: one row per action / window", (df.height, dw.height) == (len(acts), len(ws)))
        check(f"{story}: params / enablers are JSON strings",
              all(isinstance(json.loads(p), dict) for p in df["params"]) and
              all(isinstance(json.loads(e), list) for e in df["enablers"]))
        back_a, back_w = extractor.read(d / story)
        check(f"{story}: read(write(x)) == x (sorted)", back_a == sorted(acts, key=extractor.sort_key)
              and back_w == sorted(ws, key=lambda w: (w.round_num, w.start_tick)))
        check(f"{story}: action_id unique column", df["action_id"].n_unique() == df.height)

    section("Appendix A.5 params keys")
    for t, keys in A5.items():
        got = [a for a in all_acts if a.type == t]
        if not got:
            check(f"{t}: present in the stories", False)
            continue
        missing = [sorted(keys - set(a.params)) for a in got if keys - set(a.params)]
        check(f"{t}: every action has {sorted(keys)}", not missing, str(missing[:1]))
    check("pseudo-actions carry parent ids of real smoke actions", all(
        a.params["parent_action_id"] in {b.action_id for b in all_acts if b.type == "smoke"}
        for a in all_acts if a.type in ("smoke_pop", "smoke_fade")))
    check("callout params are str or None (never '?' or 0)", all(
        v is None or (isinstance(v, str) and v not in ("", "?"))
        for a in all_acts for k, v in a.params.items() if k.endswith("callout")))

    section("empty output")
    pa, pw = extractor.write([], [], d / "empty")
    check("empty actions keep the schema", dict(pl.read_parquet(pa).schema) == extractor.ACTION_SCHEMA)
    check("empty windows keep the schema", dict(pl.read_parquet(pw).schema) == extractor.WINDOW_SCHEMA)
    check("read of empty", extractor.read(d / "empty") == ([], []))

    section("deterministic bytes (P4.5.2)")
    for story in ("base", "trade"):
        outs = []
        for n in range(2):
            raw = make_raw_tables(story)
            _, acts, ws = extractor.run(raw, CFG, match_key="m1", geometry=GEO)
            outs.append(extractor.write(acts, ws, d / f"det-{story}-{n}"))
        check(f"{story}: actions.parquet identical bytes", outs[0][0].read_bytes() == outs[1][0].read_bytes())
        check(f"{story}: windows.parquet identical bytes", outs[0][1].read_bytes() == outs[1][1].read_bytes())

sys.exit(summary())
