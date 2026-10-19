"""
State schema contract (P3.8.1, P3.6.2, P7.1): token fields exactly Appendix A.2-A.4,
`Snapshot.to_row()/from_row()` round trip, frame round trip, `validate()` (the
derived-field consistency checker), and `LIVE_SAFE` coverage.

The Snapshot here is hand-made on purpose: this tests the container, not the builder
(builder tests build states from `_synthetic.py` through the real builder, P3.9.2).

    .venv\\Scripts\\python.exe test\\engine\\test_state_schema.py
"""
import math
import sys

import numpy as np
import polars as pl

from _fake_states import make_round
from _harness import check, section, summary

from engine.state import pairwise
from engine.state.geometry import FlatGeometry
from engine.state.schema import (CTX_FIELDS, LIVE_SAFE, MAX_UTILITY, N_PLAYERS, PAIR_CHANNELS,
                                 PLAYER_FIELDS, SCHEMA_VERSION, STATE_COLUMNS, UTILITY_FIELDS,
                                 ContextToken, PlayerToken, Snapshot, SnapshotMeta, UtilityToken,
                                 frame_to_snapshots, missing_live_safe, rows_to_frame)

A2 = ("side alive x y z nav_area callout grid_cell yaw pitch speed health armor has_helmet "
      "has_defuser has_c4 weapon_class n_smoke n_flash n_he n_molly n_decoy is_scoped is_ducking "
      "is_walking is_reloading is_defusing is_planting flash_duration_remaining in_smoke on_fire "
      "money equipment_value seconds_since_last_seen_by_enemy seconds_since_last_shot "
      "spotted_by_count heard_enemy kills_round damage_round player_id tier").split()
A3 = "kind x y z time_left_s owner_side owner_id radius_units".split()
A4 = ("map_version tier time_remaining_s time_into_round_s bomb_planted bomb_site "
      "time_since_plant_s defuse_in_progress defuse_with_kit bomb_carrier_alive alive_ct alive_t "
      "equip_ct equip_t money_ct money_t loss_streak_ct loss_streak_t score_ct score_t round_num "
      "half is_overtime n_active_smokes n_active_fires").split()


# ── a hand-made consistent snapshot: 10 players, CT slot 4 and T slot 9 dead ──

def player(i: int) -> PlayerToken:
    side  = "ct" if i < 5 else "t"
    alive = 0 if i in (4, 9) else 1
    x, y  = 100.0 * i, 50.0 * (i % 5)
    common = dict(side=side, alive=alive, x=x, y=y, z=0.0, nav_area=10 + i, callout=f"area{i}",
                  grid_cell=200 + i, is_scoped=0, is_ducking=0, is_walking=0, is_reloading=0,
                  is_defusing=0, is_planting=0, in_smoke=0, on_fire=0,
                  seconds_since_last_seen_by_enemy=None if i % 3 else 1.5,
                  seconds_since_last_shot=None, kills_round=float(i == 0), damage_round=0.0,
                  player_id=1_000_000 + i, tier="pro")
    if alive:
        return PlayerToken(**common, yaw=90.0, pitch=0.0, speed=0.0, health=100.0, armor=100.0,
                           has_helmet=1, has_defuser=int(side == "ct"), has_c4=int(i == 5),
                           weapon_class="rifle", n_smoke=1, n_flash=0, n_he=0, n_molly=0, n_decoy=0,
                           flash_duration_remaining=0.0, money=800.0, equipment_value=4000.0 + i,
                           spotted_by_count=0, heard_enemy=0)
    return PlayerToken(**common, yaw=None, pitch=None, speed=None, health=0.0, armor=0.0,
                       has_helmet=None, has_defuser=None, has_c4=None, weapon_class=None,
                       n_smoke=None, n_flash=None, n_he=None, n_molly=None, n_decoy=None,
                       flash_duration_remaining=0.0, money=None, equipment_value=None,
                       spotted_by_count=None, heard_enemy=None)


def make_pair(players: list[PlayerToken]) -> np.ndarray:
    pr = np.full((N_PLAYERS, N_PLAYERS, len(PAIR_CHANNELS)), np.nan, dtype=np.float32)
    for i in range(N_PLAYERS):
        for j in range(N_PLAYERS):
            if i == j:
                continue
            a, b = players[i], players[j]
            enemies = a.side != b.side
            pr[i, j, 0] = (1.0 if (i + j) % 2 else 0.0) if enemies and a.alive and b.alive else np.nan
            pr[i, j, 1] = math.dist((a.x, a.y), (b.x, b.y))
            pr[i, j, 2] = np.nan if 9 in (i, j) else pr[i, j, 1] * 1.2
            pr[i, j, 3] = 0.0 if a.alive and b.alive else np.nan     # as pairwise.py: alive pairs only
            pr[i, j, 4] = 0.0 if enemies else 1.0
    return pr


def make_snapshot() -> Snapshot:
    ps = [player(i) for i in range(N_PLAYERS)]
    ctx = ContextToken(map_version="de_test@1", tier="pro", time_remaining_s=80.0,
                       time_into_round_s=35.0, bomb_planted=0, bomb_site=None,
                       time_since_plant_s=None, defuse_in_progress=0, defuse_with_kit=None,
                       bomb_carrier_alive=1, alive_ct=4, alive_t=4,
                       equip_ct=float(sum(4000 + i for i in range(4))),
                       equip_t=float(sum(4000 + i for i in range(5, 9))),
                       money_ct=4000.0, money_t=4000.0,     # 5 accounts of 800, dead ones included
                       loss_streak_ct=0, loss_streak_t=1,
                       score_ct=3, score_t=2, round_num=6, half=1, is_overtime=0,
                       n_active_smokes=1, n_active_fires=0)
    util = [UtilityToken(kind="smoke", x=1000.0, y=0.0, z=0.0, time_left_s=12.5, owner_side="ct",
                         owner_id=1_000_000, radius_units=144.0)]
    meta = SnapshotMeta(match_key="synthetic-1", round_num=6, tick=2000, t_s=7.8125,
                        kinds=("grid", "boundary_pre"))
    labels = {"round_won_ct": 1, "dies_within_5s": [0] * 9 + [1], "time_to_next_engagement_s": None}
    return Snapshot(players=ps, utility=util, ctx=ctx, pair=make_pair(ps), meta=meta, labels=labels)


# ── fields exactly Appendix A ──
section("fields exactly Appendix A.2 - A.4")
check("PlayerToken fields == A.2 in order", list(PLAYER_FIELDS) == A2,
      str(set(PLAYER_FIELDS) ^ set(A2)))
check("UtilityToken fields == A.3", list(UTILITY_FIELDS) == A3)
check("ContextToken fields == A.4", list(CTX_FIELDS) == A4, str(set(CTX_FIELDS) ^ set(A4)))
check("constants", (SCHEMA_VERSION, N_PLAYERS, MAX_UTILITY) == (1, 10, 16)
      and PAIR_CHANNELS == ("los", "dist_units", "nav_dist_units", "smoke_between", "same_team"))

# ── validate on the consistent snapshot ──
section("validate()")
snap = make_snapshot()
check("consistent snapshot: no problems", snap.validate() == [], str(snap.validate()))

s = snap.copy()
s.ctx.alive_ct = 5
check("broken alive count -> exactly one problem", len(s.validate()) == 1, str(s.validate()))
check("copy() is deep (original untouched)", snap.ctx.alive_ct == 4 and snap.validate() == [])

s = snap.copy()
s.pair[0, 6, 1] += 10.0
probs = s.validate()
check("broken symmetry -> exactly one problem", len(probs) == 1 and "dist_units" in probs[0], str(probs))

s = snap.copy()
s.players[4].equipment_value = 800.0
probs = s.validate()
check("dead player with equipment -> problem", len(probs) == 1 and "dead" in probs[0], str(probs))

s = snap.copy()
s.players[9].health = 35.0
check("dead player with health -> problem", any("dead" in p for p in s.validate()))

s = snap.copy()
s.ctx.equip_t += 100.0
check("equip sum mismatch -> problem", any("equip_t" in p for p in s.validate()))

s = snap.copy()
s.players[0].equipment_value = None
s.ctx.equip_ct = None
check("unknown equipment skips the equip check", s.validate() == [], str(s.validate()))

s = snap.copy()
s.pair[2, 2, 0] = 1.0
check("diagonal not NaN -> problem", any("diagonal" in p for p in s.validate()))

s = snap.copy()
s.ctx.bomb_site = "A"
check("bomb_site without plant -> problem", any("bomb_site" in p for p in s.validate()))

s = snap.copy()
s.ctx.time_remaining_s = -0.5
check("negative clock -> problem", any("time_remaining_s" in p for p in s.validate()))

s = snap.copy()
s.utility = s.utility * (MAX_UTILITY + 1)
check("17 utility tokens -> problem", any("utility" in p and "> 16" in p for p in s.validate()))

s = snap.copy()
s.players[0], s.players[5] = s.players[5], s.players[0]
check("T before CT -> problem", any("order" in p for p in s.validate()))

s = snap.copy()
s.players = s.players[:9]
check("9 players -> problem", any("9 tokens" in p for p in s.validate()))

s = snap.copy()
s.players[0].side = "t"
s.pair = make_pair(s.players)
check("6 T / 4 CT -> problem", any("on side t" in p for p in s.validate()))

s = snap.copy()
s.pair[0, 4, 0] = s.pair[4, 0, 0] = 1.0
check("los with a dead player -> problem", any("pair.los" in p for p in s.validate()))

# ── fixtures agree with the builder (LOW-64) ──
section("hand-made and _fake_states fixtures follow the builder's pair / money definitions")
SMOKE_CH = PAIR_CHANNELS.index("smoke_between")
built = pairwise.table(snap.players, FlatGeometry(), snap.utility)
check("hand-made pair: smoke_between NaN exactly where pairwise.table writes NaN (dead pairs)",
      np.array_equal(np.isnan(snap.pair[:, :, SMOKE_CH]), np.isnan(built[:, :, SMOKE_CH])))
fr = make_round(40, seed=3, ct_wins=True)
dead_pair_ok = all(np.isnan(s.pair[i, j, SMOKE_CH]) == (s.players[i].alive != 1 or s.players[j].alive != 1)
                   for s in fr for i in range(N_PLAYERS) for j in range(N_PLAYERS) if i != j)
check("_fake_states: smoke_between NaN for every pair with a dead player, 0/1 otherwise", dead_pair_ok)
deaths = fr[0].ctx.alive_ct + fr[0].ctx.alive_t > fr[-1].ctx.alive_ct + fr[-1].ctx.alive_t
check("_fake_states: money_ct/t is every roster account, so a death does not lower it",
      deaths and len({(s.ctx.money_ct, s.ctx.money_t) for s in fr}) == 1)

# ── rows and frames ──
section("to_row / from_row / frames")
row = snap.to_row()
check("row has schema_version", row["schema_version"] == SCHEMA_VERSION)
check("pair flattened to 500 values, NaN -> None",
      len(row["pair"]) == 500 and row["pair"][0] is None and isinstance(row["pair"][6], float))  # [0,1,dist]
check("ctx inlined with ctx_ prefix", all(f"ctx_{k}" in row for k in CTX_FIELDS))
check("labels as y_ keys", {"y_round_won_ct", "y_dies_within_5s", "y_time_to_next_engagement_s"} <= set(row))
check("players / utility as lists of dicts",
      len(row["players"]) == 10 and isinstance(row["players"][0], dict) and len(row["utility"]) == 1)
back = Snapshot.from_row(row)
check("from_row(to_row(s)) == s", back == snap)
check("round trip keeps NaN pattern",
      np.array_equal(np.isnan(back.pair), np.isnan(snap.pair)) and back.pair.dtype == np.float32)
check("y_ prefix stripped in labels", set(back.labels) == {"round_won_ct", "dies_within_5s",
                                                          "time_to_next_engagement_s"})

nolab = snap.copy()
nolab.labels = None
nolab.meta.tick = 2001
df = rows_to_frame([snap.to_row(), nolab.to_row()])
check("frame columns carry STATE_COLUMNS dtypes",
      all(df.schema[c] == dt for c, dt in STATE_COLUMNS.items()), str(df.schema))
check("pair is a 500-wide float32 array", df.schema["pair"] == pl.Array(pl.Float32, 500))
snaps = frame_to_snapshots(df)
check("frame round trip (labelled)", snaps[0] == snap)
check("frame round trip (unlabelled -> labels None)", snaps[1] == nolab and snaps[1].labels is None)
check("empty rows -> empty frame with schema", rows_to_frame([]).columns == list(STATE_COLUMNS))
check("prefixed label keys normalise", Snapshot(snap.players, [], snap.ctx, labels={"y_a": 1}).labels == {"a": 1})

bad = dict(row, schema_version=SCHEMA_VERSION + 1)
try:
    Snapshot.from_row(bad)
    check("wrong schema_version rejected", False)
except ValueError:
    check("wrong schema_version rejected", True)

# ── helpers ──
section("helpers")
check("index_of finds a player", snap.index_of(1_000_007) == 7 and snap.index_of(42) is None)
check("alive_ct / alive_t read the context", (snap.alive_ct, snap.alive_t) == (4, 4))
other = snap.copy()
other.labels["round_won_ct"] = 0
check("equality sees label changes", other != snap)

# ── LIVE_SAFE ──
section("LIVE_SAFE (P3.6.2)")
check("missing_live_safe() is empty", missing_live_safe() == [], str(missing_live_safe()))
check("covers every token field and pair channel",
      len(LIVE_SAFE) == len(A2) + len(A3) + len(A4) + len(PAIR_CHANNELS))
saved = LIVE_SAFE.pop("ctx.alive_ct")
check("a removed justification is reported", missing_live_safe() == ["no justification: ctx.alive_ct"])
LIVE_SAFE["ctx.alive_ct"] = saved
LIVE_SAFE["ctx.round_duration_s"] = "leaky"
check("a stale key is reported", missing_live_safe() == ["stale key: ctx.round_duration_s"])
del LIVE_SAFE["ctx.round_duration_s"]

sys.exit(summary())
