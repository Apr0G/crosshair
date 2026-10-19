"""
Data QA (P5.4): planted outliers (speed, health, negative time left, > 5 alive) are
each reported with their round and time; null %, ranges and 10-bin histograms per
feature and per map; per-tier counts; UNK counts for a val-only callout; thresholds
(`qa.max_null_pct.*`, `qa.max_outliers`) produce violations and `assert_thresholds`
raises; the markdown report is written.

    .venv\\Scripts\\python.exe test\\engine\\test_qa.py [--cpu]
"""
import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

from _harness import check, section, summary
from _fake_states import fake_arrays, make_corpus

from engine import config
from engine.data import dataset, qa
from engine.data.dataset import CTX_LAYOUT, PLAYER_LAYOUT, UTIL_LAYOUT

cfg = config.load()
snaps, vocab, arrs, _ = fake_arrays(8, seed=71)
P = PLAYER_LAYOUT

with tempfile.TemporaryDirectory() as d:
    out = Path(d)
    clean = qa.report("clean", arrays=arrs, vocab=vocab, cfg=cfg, out_dir=out, date_str="2026-10-19")

    section("clean data")
    check("no outliers in clean fake data", all(o["count"] == 0 for o in clean["outliers"].values()),
          str({k: v["count"] for k, v in clean["outliers"].items()}))
    check("report written", Path(clean["path"]).name == "qa_clean_2026-10-19.md" and Path(clean["path"]).is_file())
    check("counts", clean["n_snapshots"] == len(snaps) and clean["n_rounds"] == 8)
    check("per-tier snapshots sum to n", sum(t["snapshots"] for t in clean["per_tier"].values()) == len(snaps))
    check("tiers reported", set(clean["per_tier"]) == {"pro", "faceit_10"})
    hx = clean["features"]["player.health"]
    check("feature stats: null %, min, max, mean, 10-bin histogram",
          hx["min"] == 0.0 and hx["max"] <= 100 and len(hx["hist"]["counts"]) == 10 and hx["null_pct"] == 0.0)
    check("nullable feature has a null %", clean["features"]["player.money"]["null_pct"] > 0)
    check("pairwise LOS null % (dead pairs) reported", clean["features"]["pairwise.los"]["null_pct"] > 0)
    check("per map version breakdown with names", set(clean["per_map"]) == {"de_fake@1", "de_other@2"})
    check("histogram counts = known values", sum(hx["hist"]["counts"]) == round(hx["n"] * (1 - hx["null_pct"] / 100)))

    section("planted outliers")
    bad = {k: v.copy() for k, v in arrs.items()}
    alive = bad["player_num"][..., P.col("alive")] > 0.5
    i_s, j_s = map(int, np.argwhere(alive)[3])
    bad["player_num"][i_s, j_s, P.col("speed")] = 500.0
    i_h, j_h = map(int, np.argwhere(alive)[40])
    bad["player_num"][i_h, j_h, P.col("health")] = 150.0
    i_t = 77
    bad["ctx_num"][i_t, CTX_LAYOUT.col("time_remaining_s")] = -1.0
    i_a = 120
    bad["ctx_num"][i_a, CTX_LAYOUT.col("alive_ct")] = 6
    i_u = int(np.flatnonzero(bad["util_mask"].any(1))[0])
    k_u = int(np.flatnonzero(bad["util_mask"][i_u])[0])
    bad["util_num"][i_u, k_u, UTIL_LAYOUT.col("time_left_s")] = -2.0
    r = qa.report("planted", arrays=bad, vocab=vocab, cfg=cfg, out_dir=out)
    o = r["outliers"]
    check("speed > 320 reported once", o["speed"]["count"] == 1)
    check("speed example names the round and time",
          o["speed"]["examples"][0]["round_key"] == arrs["round_key"][i_s]
          and abs(o["speed"]["examples"][0]["t_s"] - arrs["t_s"][i_s]) < 1e-9)
    check("health > 100 reported once", o["health"]["count"] == 1)
    check("negative time left (clock and utility) reported", o["negative_time_left"]["count"] == (1 if i_u == i_t else 2))
    check("> 5 alive per side reported once", o["alive_per_side"]["count"] == 1)
    md = Path(r["path"]).read_text(encoding="utf-8")
    check("markdown lists the outlier rules", all(k in md for k in ("speed", "health", "negative_time_left", "alive_per_side")))
    dead_fast = {k: v.copy() for k, v in arrs.items()}
    dj = np.argwhere(~alive)[0]
    dead_fast["player_num"][dj[0], dj[1], P.col("speed")] = 999.0      # dead: speed is not judged
    check("dead players' speed is not judged", qa.report("df", arrays=dead_fast, cfg=cfg, write=False)["outliers"]["speed"]["count"] == 0)

    section("thresholds")
    def with_qa(**kw):
        q = cfg.qa
        null = dataclasses.replace(q.max_null_pct, **{k: kw.pop(k) for k in list(kw) if k in ("player", "utility", "context", "pairwise")})
        return dataclasses.replace(cfg, qa=dataclasses.replace(q, max_null_pct=null, **kw))
    check("config thresholds unset -> nothing violated", qa.violations(r, cfg) == [])
    c0 = with_qa(max_outliers=0)
    v0 = qa.violations(r, c0)
    check("max_outliers 0 with planted outliers -> violation", len(v0) == 1 and "outliers" in v0[0])
    check("max_outliers 10 -> ok", qa.violations(r, with_qa(max_outliers=10)) == [])
    vp = qa.violations(clean, with_qa(player=5.0))
    check("player null limit 5% -> violations name the features", any(v.startswith("player.money") for v in vp))
    check("player null limit 100% -> ok", qa.violations(clean, with_qa(player=100.0)) == [])
    try:
        qa.assert_thresholds(r, c0)
        check("assert_thresholds raises QAError", False)
    except qa.QAError as e:
        check("assert_thresholds raises QAError", e.violations == v0)
    qa.assert_thresholds(clean, cfg)
    check("assert_thresholds passes on clean data with unset thresholds", True)

    section("UNK counted")
    val = make_corpus(2, seed=72, prefix="val")
    for s in val:
        for p in s.players:
            if p.callout is not None:
                p.callout = "val_only_" + p.callout
    va = dataset.tensorize(val, vocab)
    rv = qa.report("val", arrays=va, vocab=vocab, cfg=cfg, write=False)
    check("val-only callouts counted as UNK", rv["unk"]["player_cat.callout"] > 0)
    check("no UNK in train categories", clean["unk"]["player_cat.callout"] == 0)

sys.exit(summary())
