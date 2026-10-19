"""
Context token (P3.4): live clock (115 s from freeze_end, 40 s bomb timer, never the end
tick), bomb state, defuse, carrier, alive / equipment / money sums, scores, loss
streaks, halves and overtime.

    .venv\\Scripts\\python.exe test\\engine\\test_context.py
"""
import sys

from _harness import check, close, section, summary
from _synthetic import make_raw_tables

from engine import config
from engine.state import builder, context
from engine.state.geometry import FlatGeometry

CFG = config.load()
GEO = FlatGeometry()


def ctx_at(raw, rnd, tick):
    sl = builder.RoundView.build(raw, rnd).until(tick)
    return builder.snapshot_at(sl, GEO, CFG, match_key="c", tier="pro").ctx


for tr in (64, 128):
    T = lambda t128: int(round(t128 * tr / 128))  # noqa: E731
    raw = make_raw_tables("base", tr)
    fe, plant = T(1000), T(3000)

    # ── clock ──
    section(f"clock @ {tr} tick")
    check("115.0 at freeze_end", ctx_at(raw, 1, fe).time_remaining_s == 115.0)
    check("time_into_round 0.0 at freeze_end", ctx_at(raw, 1, fe).time_into_round_s == 0.0)
    check("110.0 five seconds in", close(ctx_at(raw, 1, fe + 5 * tr).time_remaining_s, 110.0))
    c = ctx_at(raw, 1, plant)
    check("40.0 at the plant tick", c.time_remaining_s == 40.0, str(c.time_remaining_s))
    check("30.0 ten seconds after the plant", close(ctx_at(raw, 1, plant + 10 * tr).time_remaining_s, 30.0))
    check("time_since_plant 10.0", close(ctx_at(raw, 1, plant + 10 * tr).time_since_plant_s, 10.0))
    check("clock unaffected by the round having ended (terminal tick)",
          close(ctx_at(raw, 1, T(4360)).time_remaining_s, 40.0 - (T(4360) - plant) / tr))
    check("never below 0", ctx_at(make_raw_tables("trade", tr), 1, T(3000) + 40 * tr + int(2.5 * tr)).time_remaining_s == 0.0)

    # ── bomb ──
    section(f"bomb @ {tr} tick")
    pre = ctx_at(raw, 1, plant - 1)
    check("pre-plant: bomb_planted 0, bomb_site None (not '?')", pre.bomb_planted == 0 and pre.bomb_site is None)
    check("pre-plant: time_since_plant None", pre.time_since_plant_s is None)
    check("pre-plant: carrier alive 1 (D has the bomb)", pre.bomb_carrier_alive == 1)
    check("post-plant: planted 1, site 'A', carrier None", (c.bomb_planted, c.bomb_site, c.bomb_carrier_alive) == (1, "A", None))
    check("no defuse in base: in_progress 0, kit None", c.defuse_in_progress == 0 and c.defuse_with_kit is None)
    rt = make_raw_tables("retreat", tr)
    d = ctx_at(rt, 1, T(3950))
    check("retreat: defuse in progress with kit at 3950", (d.defuse_in_progress, d.defuse_with_kit) == (1, 1))
    check("retreat: defuse not in progress before 3900 and after completion",
          ctx_at(rt, 1, T(3900) - 1).defuse_in_progress == 0 and ctx_at(rt, 1, T(4200)).defuse_in_progress == 0)
    tr_ = make_raw_tables("trade", tr)
    sl = builder.RoundView.build(raw, 1).until(T(2500))
    snap = builder.snapshot_at(sl, GEO, CFG, match_key="c", tier="pro")
    for p in snap.players:
        if p.side == "t" and p.alive == 1:
            p.has_c4 = 0
    check("no alive T has the bomb -> carrier 0",
          context.token(sl, snap.players, 1, 0, clock=CFG.clock, map_version="m", tier="pro").bomb_carrier_alive == 0)
    for p in snap.players:
        if p.side == "t" and p.alive == 1:
            p.has_c4 = None
    check("carrier flags unknown -> None",
          context.token(sl, snap.players, 1, 0, clock=CFG.clock, map_version="m", tier="pro").bomb_carrier_alive is None)

    # ── counts and sums ──
    section(f"counts @ {tr} tick")
    c1 = ctx_at(raw, 1, T(1512))
    check("alive 2v2 early, 2v1 after 2000", (c1.alive_ct, c1.alive_t) == (2, 2)
          and (ctx_at(raw, 1, T(2001)).alive_ct, ctx_at(raw, 1, T(2001)).alive_t) == (2, 1))
    check("equip sums alive players' equipment", c1.equip_ct == 4600.0 + 4500.0 and c1.equip_t == 3900.0 + 3900.0)
    check("money sums every account (dead included)", ctx_at(raw, 1, T(2001)).money_t == 400.0 + 300.0)
    check("active smokes / fires", (c1.n_active_smokes, c1.n_active_fires) == (1, 0))
    check("map_version and tier", c1.map_version == GEO.map_version and c1.tier == "pro")

    # ── score, streaks, half ──
    section(f"score and streaks @ {tr} tick")
    r1, r2 = ctx_at(raw, 1, fe), ctx_at(raw, 2, raw.rounds["freeze_end_tick"][1])
    check("round 1: 0-0, no streaks", (r1.score_ct, r1.score_t, r1.loss_streak_ct, r1.loss_streak_t) == (0, 0, 0, 0))
    check("round 2 after a CT win: 1-0, T streak 1", (r2.score_ct, r2.score_t, r2.loss_streak_ct, r2.loss_streak_t)
          == (1, 0, 0, 1))
    t2 = ctx_at(tr_, 2, tr_.rounds["freeze_end_tick"][1])
    check("round 2 after a T win: 0-1, CT streak 1", (t2.score_ct, t2.score_t, t2.loss_streak_ct, t2.loss_streak_t)
          == (0, 1, 1, 0))
    check("round_num / half / overtime", (r2.round_num, r2.half, r2.is_overtime) == (2, 1, 0))

section("loss streaks and halves (pure)")
W = lambda *ws: [{"round_num": i + 1, "winner_side": w} for i, w in enumerate(ws)]  # noqa: E731
check("three T wins -> CT streak 3", context.loss_streaks(W("ct", "t", "t", "t"), 5) == (3, 0))
check("streak broken by a win", context.loss_streaks(W("t", "t", "ct"), 4) == (0, 1))
check("streak resets at half time (round 13)", context.loss_streaks(W(*["t"] * 12), 13) == (0, 0))
check("streak counts only same-half rounds", context.loss_streaks(W(*(["t"] * 12 + ["ct", "ct"])), 15) == (0, 2))
check("unknown winner stops the count", context.loss_streaks(W("t", None, "t"), 4) == (1, 0))
check("future rounds ignored", context.loss_streaks(W("t", "t", "t", "t"), 3) == (2, 0))
check("half_of: MR12 + MR3 overtime", [context.half_of(r) for r in (1, 12, 13, 24, 25, 27, 28, 31)]
      == [1, 1, 2, 2, 3, 3, 4, 5])
check("site names", [context.site_name(s) for s in ("A", "b", "BombsiteA", "bombsite_b", None)]
      == ["A", "B", "A", "B", None])

sys.exit(summary())
