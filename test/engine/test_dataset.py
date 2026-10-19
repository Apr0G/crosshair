"""
Dataset builder (P5.3): tensor keys exactly P5.3.1, shapes / masks / labels match the
snapshots, None -> 0 + presence bit, UNK for unseen categories (per map version for
callouts), vocab and norm stats deterministic and round-trip, shards round-trip, and
`build()` from a catalog + split file never puts a val round in a train shard.

    .venv\\Scripts\\python.exe test\\engine\\test_dataset.py [--cpu]
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from _harness import check, section, summary
from _fake_states import make_corpus, make_round

from engine.data import dataset
from engine.data.dataset import (BATCH_KEYS, CTX_LAYOUT, FEATURE_LAYOUT, NULL, PAD, PLAYER_LAYOUT, UNK,
                                 UTIL_LAYOUT, NormStats, Vocab)

train_snaps = make_corpus(12, seed=0)
val_snaps   = make_corpus(4, seed=1, prefix="val")
vocab = Vocab().fit(train_snaps)
arrs  = dataset.tensorize(train_snaps, vocab)
N = len(train_snaps)

# ── keys, shapes, dtypes ──
section("keys and shapes")
check("keys are exactly P5.3.1", tuple(arrs) == BATCH_KEYS, str(list(arrs)))
shapes = {"player_num": (N, 10, len(PLAYER_LAYOUT)), "player_cat": (N, 10, 6), "player_mask": (N, 10),
          "util_num": (N, 16, len(UTIL_LAYOUT)), "util_cat": (N, 16, 3), "util_mask": (N, 16),
          "ctx_num": (N, len(CTX_LAYOUT)), "ctx_cat": (N, 3), "pair": (N, 10, 10, 5), "y_win": (N,),
          "y_hazard": (N, 10), "y_plant15": (N,), "y_defuse15": (N,), "y_tte": (N,), "is_terminal": (N,),
          "round_key": (N,), "t_s": (N,), "tier": (N,), "tick_kinds": (N,)}
for k, shp in shapes.items():
    check(f"{k} shape {shp}", arrs[k].shape == shp, str(arrs[k].shape))
check("round_key is a string array", arrs["round_key"].dtype.kind == "U")
for block in ("player_num", "player_cat", "util_num", "util_cat", "ctx_num", "ctx_cat", "pair"):
    check(f"FEATURE_LAYOUT[{block}] names every column", len(FEATURE_LAYOUT[block]) == arrs[block].shape[-1])

# ── values match the snapshots ──
section("values and masks match the snapshots")
P = PLAYER_LAYOUT
ok_vals = ok_pres = ok_dead = ok_angle = True
for i in range(0, N, 37):
    s = train_snaps[i]
    for j, p in enumerate(s.players):
        row = arrs["player_num"][i, j]
        if p.alive:
            ok_vals &= math.isclose(row[P.col("x")], p.x, rel_tol=1e-6, abs_tol=1e-3)
            ok_vals &= math.isclose(row[P.col("health")], p.health, abs_tol=1e-4)
            ok_angle &= math.isclose(row[P.col("yaw_sin")], math.sin(math.radians(p.yaw)), abs_tol=1e-5)
            ok_angle &= math.isclose(row[P.col("yaw_cos")], math.cos(math.radians(p.yaw)), abs_tol=1e-5)
            ok_pres &= row[P.col("present_yaw")] == 1.0
        else:
            ok_dead &= row[P.col("present_yaw")] == 0.0 and row[P.col("yaw_sin")] == 0.0
            ok_dead &= row[P.col("present_money")] == 0.0 and row[P.col("money")] == 0.0
        sl = p.seconds_since_last_shot
        pres = row[P.col("present_seconds_since_last_shot")]
        ok_pres &= (pres == 0.0 and row[P.col("seconds_since_last_shot")] == 0.0) if sl is None else \
                   (pres == 1.0 and math.isclose(row[P.col("seconds_since_last_shot")], sl, abs_tol=1e-4))
check("x / health values equal the token fields", ok_vals)
check("yaw encoded as sin/cos", ok_angle)
check("None -> value 0 with presence bit 0; known -> presence 1", ok_pres)
check("dead players: nulled fields are 0 with presence 0", ok_dead)

s0 = train_snaps[5]
check("player_mask all True for 10 real players", bool(arrs["player_mask"][5].all()))
check("util_mask counts the utility tokens", int(arrs["util_mask"][5].sum()) == len(s0.utility))
check("absent utility slots are PAD", bool((arrs["util_cat"][5][~arrs["util_mask"][5]] == PAD).all()))
check("pair copied with NaN preserved", np.array_equal(arrs["pair"][5], s0.pair, equal_nan=True))
check("ctx alive_ct value", arrs["ctx_num"][5, CTX_LAYOUT.col("alive_ct")] == s0.ctx.alive_ct)
pre = [s for s in train_snaps if not s.ctx.bomb_planted][0]
ip = train_snaps.index(pre)
check("time_since_plant_s None pre-plant -> presence 0",
      arrs["ctx_num"][ip, CTX_LAYOUT.col("present_time_since_plant_s")] == 0.0)
check("bomb_site None pre-plant -> NULL", arrs["ctx_cat"][ip, FEATURE_LAYOUT["ctx_cat"].index("bomb_site")] == NULL)

section("labels and meta")
check("y_win = round_won_ct", all(arrs["y_win"][i] == s.labels["round_won_ct"] for i, s in enumerate(train_snaps)))
haz_ok = True
for i, s in enumerate(train_snaps[:200]):
    for j, v in enumerate(s.labels["dies_within_5s"]):
        haz_ok &= (np.isnan(arrs["y_hazard"][i, j]) if v is None else arrs["y_hazard"][i, j] == v)
check("y_hazard = dies_within_5s with None -> NaN (never 0)", haz_ok)
pl = [(i, s.labels["plant_within_15s"]) for i, s in enumerate(train_snaps)]
check("y_plant15 None -> NaN", all(np.isnan(arrs["y_plant15"][i]) if v is None else arrs["y_plant15"][i] == v for i, v in pl))
check("y_tte equals the label", np.allclose(arrs["y_tte"], [s.labels["time_to_next_engagement_s"] for s in train_snaps]))
check("is_terminal: one per round", int(arrs["is_terminal"].sum()) == 12)
check("round_key = match#round", arrs["round_key"][0] == f"{train_snaps[0].meta.match_key}#{train_snaps[0].meta.round_num}")
check("t_s copied", np.allclose(arrs["t_s"], [s.meta.t_s for s in train_snaps]))
check("tier copied", list(arrs["tier"][:3]) == [s.ctx.tier for s in train_snaps[:3]])
term_bits = arrs["tick_kinds"][arrs["is_terminal"] == 1]
check("tick_kinds terminal bit set on terminal snapshots", bool((term_bits & 8).all()))

# ── padding slot, UNK, per-map vocab ──
section("pad slot, UNK and per-map vocabularies")
pad = train_snaps[0].copy()
pad.players[9].player_id = -1
a_pad = dataset.tensorize([pad], vocab)
check("player_id -1 (empty slot) -> player_mask False", not a_pad["player_mask"][0, 9] and a_pad["player_mask"][0, :9].all())

odd = val_snaps[0].copy()
alive_j = next(j for j, p in enumerate(odd.players) if p.alive)
odd.players[alive_j].callout = "never_seen_in_train"
odd.players[alive_j].player_id = 999_999_999
a_odd = dataset.tensorize([odd], vocab)
ccol = FEATURE_LAYOUT["player_cat"].index("callout")
icol = FEATURE_LAYOUT["player_cat"].index("player_id")
check("unseen callout -> UNK", a_odd["player_cat"][0, alive_j, ccol] == UNK)
check("unseen player id -> UNK", a_odd["player_cat"][0, alive_j, icol] == UNK)
dead_j = next((j for j, p in enumerate(odd.players) if not p.alive and p.weapon_class is None), None)
if dead_j is not None:
    wcol = FEATURE_LAYOUT["player_cat"].index("weapon_class")
    check("None weapon -> NULL (not UNK)", a_odd["player_cat"][0, dead_j, wcol] == NULL)
some_callout = train_snaps[0].players[0].callout
k1 = vocab.index("callout", f"de_fake@1|{some_callout}")
k2 = vocab.index("callout", f"de_other@2|{some_callout}")
check("callouts keyed per map version", k1 != k2 or k1 == UNK, f"{k1} vs {k2}")
check("every train category is in the vocab (no UNK in train arrays)",
      int((arrs["player_cat"][arrs["player_mask"]] == UNK).sum()) == 0 and int((arrs["ctx_cat"] == UNK).sum()) == 0)
check("reserved ids UNK=0 PAD=1", UNK == 0 and PAD == 1)

with tempfile.TemporaryDirectory() as d:
    vp = vocab.save(Path(d) / "vocab_v1.json")
    v2 = Vocab.load(vp)
    check("vocab save/load round trip", v2.tables == vocab.tables)
check("vocab refit is identical", Vocab().fit(train_snaps).tables == vocab.tables)

# ── norm stats ──
section("normalization stats")
n1 = NormStats.fit([arrs])
n2 = NormStats.fit([arrs])
half = N // 2
n3 = NormStats.fit([dataset.take(arrs, np.arange(half)), dataset.take(arrs, np.arange(half, N))])
check("norm fit deterministic", n1.stats == n2.stats)
check("norm fit streaming over parts == one pass",
      all(np.allclose(n1.stats[b][k], n3.stats[b][k], rtol=1e-9, atol=1e-9) for b in n1.stats for k in n1.stats[b]))
alive_x = np.concatenate([[p.x for p in s.players if p.player_id != -1 and p.x is not None] for s in train_snaps])
check("x mean matches the known values", math.isclose(n1.stats["player_num"]["x"][0], float(alive_x.mean()), rel_tol=1e-6))
check("flags are not z-scored", "alive" not in n1.stats["player_num"] and "has_c4" not in n1.stats["player_num"])
check("pair distance channels have stats", set(n1.stats["pair"]) == {"dist_units", "nav_dist_units"})
nz = n1.normalize(arrs)
hc = P.col("health")
hv = nz["player_num"][..., hc][arrs["player_mask"]]
check("normalized health has mean ~0, std ~1", abs(hv.mean()) < 1e-3 and abs(hv.std() - 1) < 1e-3)
miss = arrs["player_num"][..., P.col("present_money")] == 0
check("missing values stay 0 after normalization", bool((nz["player_num"][..., P.col("money")][miss] == 0).all()))
check("tensorize(norm=...) == normalize(tensorize())",
      np.allclose(dataset.tensorize(train_snaps[:20], vocab, n1)["player_num"], nz["player_num"][:20]))
with tempfile.TemporaryDirectory() as d:
    p = n1.save(Path(d) / "norm_v1.json")
    check("norm save/load round trip", NormStats.load(p).stats == n1.stats)

# ── shards ──
section("shards")
with tempfile.TemporaryDirectory() as d:
    base = Path(d)
    paths_ = dataset.write_shards("train", iter(train_snaps), 100, vocab=vocab, base=base)
    check("shard count = ceil(n / shard_size)", len(paths_) == math.ceil(N / 100), str(len(paths_)))
    back = dataset.read_shards("train", base)
    check("read_shards returns every key", tuple(back) == BATCH_KEYS)
    check("round trip equal", all(np.array_equal(back[k], arrs[k], equal_nan=back[k].dtype.kind == "f") for k in BATCH_KEYS))
    idx = json.loads((base / "train" / "index.json").read_text())
    check("index.json records n and layout hash", idx["n"] == N and idx["layout_hash"] == dataset.layout_hash())
    try:
        dataset.write_shards("train", iter(train_snaps), 100, vocab=vocab, base=base)
        check("existing shards are never overwritten", False)
    except FileExistsError:
        check("existing shards are never overwritten", True)

# ── build from catalog + split file ──
section("build() from catalog + split file")
with tempfile.TemporaryDirectory() as d:
    from engine.data import layout
    from engine.state.schema import rows_to_frame
    root = Path(d) / "engine"
    matches = {}
    for s in train_snaps + val_snaps:
        matches.setdefault(s.meta.match_key, []).append(s)
    extra_val = make_round(20, seed=99, ct_wins=True, match_key="valonly-0", map_version="de_valonly@1")
    matches["valonly-0"] = extra_val
    catalog = []
    for mk, ss in matches.items():
        mv = ss[0].ctx.map_version
        path = layout.states_path(mv, "2026-09", mk, base=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows_to_frame([s.to_row() for s in ss]).write_parquet(path)
        catalog.append({"match_key": mk, "tier": ss[0].ctx.tier, "map_version": mv,
                        "states_path": path.relative_to(root).as_posix()})
    train_keys = sorted({s.meta.match_key for s in train_snaps})
    val_keys   = sorted({s.meta.match_key for s in val_snaps}) + ["valonly-0"]
    split_file = Path(d) / "splits_v1.json"
    split_file.write_text(json.dumps({"splits": {"train": train_keys, "val": val_keys, "test": [], "holdout": []}}))
    res = dataset.build(catalog, split_file, tiers=None, maps=None, sample_hz=None,
                        base=Path(d) / "datasets", engine_dir=root, shard_size=150)
    tr = dataset.read_shards("train", Path(d) / "datasets")
    va = dataset.read_shards("val", Path(d) / "datasets")
    check("train shards hold every train snapshot", len(tr["t_s"]) == len(train_snaps))
    train_matches = {r.split("#")[0] for r in tr["round_key"]}
    val_matches   = {r.split("#")[0] for r in va["round_key"]}
    check("no val round in a train shard", not (set(tr["round_key"]) & set(va["round_key"]))
          and train_matches <= set(train_keys) and not (train_matches & set(val_keys)))
    check("val shards hold the val matches", val_matches == set(val_keys))
    v_built = Vocab.load(res["vocab"])
    check("vocab fitted on train only (val-only map is not in it)",
          "de_valonly@1" not in v_built.tables["map_version"])
    mvcol = FEATURE_LAYOUT["ctx_cat"].index("map_version")
    vo = np.array([r.startswith("valonly") for r in va["round_key"]])
    check("val-only map version maps to UNK in val shards", bool((va["ctx_cat"][vo, mvcol] == UNK).all()))
    check("norm stats written", Path(res["norm"]).is_file())
    res_p = dataset.build(catalog, split_file, tiers=("pro",), maps=("de_fake",), sample_hz=None,
                          base=Path(d) / "datasets", engine_dir=root, shard_size=150, suffix="_pro")
    trp = dataset.read_shards("train_pro", Path(d) / "datasets")
    check("tier + map filters applied", set(trp["tier"]) == {"pro"} and len(trp["t_s"]) > 0)
    check("version numbers increment", Path(res_p["vocab"]).name == "vocab_v2.json")

# ── rate filter (P5.3.3) ──
section("sample-rate filter")
r4 = make_round(40, seed=4, ct_wins=False)
kept = [s for s in r4 if dataset.keep_for_rate(s, 1)]
grid = [s for s in r4 if s.meta.kinds == ("grid",)]
grid_kept = [s for s in kept if s.meta.kinds == ("grid",)]
check("1 Hz keeps only whole-second grid samples", all(abs(s.meta.t_s - round(s.meta.t_s)) < 1e-6 for s in grid_kept)
      and len(grid_kept) < len(grid))
check("boundary and terminal samples always kept",
      all(dataset.keep_for_rate(s, 1) for s in r4 if set(s.meta.kinds) - {"grid"}))

sys.exit(summary())
