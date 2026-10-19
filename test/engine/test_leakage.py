"""
Leakage audit (P3.6): outcome-flip invariance on every story at 64 and 128 tick, a
deliberately leaky feature (round duration / winner) is caught, the poisoned-future-row
check passes on the real builder and catches a builder that skips the causal cut, and
LIVE_SAFE justifies every feature.

    .venv\\Scripts\\python.exe test\\engine\\test_leakage.py
"""
import sys

from _harness import check, section, summary
from _synthetic import STORIES, make_raw_tables

from engine import config
from engine.state import builder, leakage
from engine.state.geometry import FlatGeometry
from engine.state.schema import missing_live_safe

CFG = config.load()
GEO = FlatGeometry()
_CURRENT = {}


def build_fn(raw):
    _CURRENT["raw"] = raw
    return builder.build(raw, GEO, CFG, match_key="leak", tier="pro", labels=False)


# ── P3.6.2 ──
section("LIVE_SAFE")
check("every feature has a live-safety justification", missing_live_safe() == [], str(missing_live_safe()))

# ── P3.6.1 on every story ──
section("outcome flip audit")
for tr in (64, 128):
    for story in STORIES:
        probs = leakage.audit(build_fn, make_raw_tables(story, tr))
        check(f"{story} @ {tr}: no feature changes when the outcome flips", probs == [], "; ".join(probs[:3]))

raw = make_raw_tables("base", 128)
flipped = leakage.flip_outcome(raw, 1)
r0, r1 = raw.rounds.row(0, named=True), flipped.rounds.row(0, named=True)
check("flip_outcome flips winner and reason, moves official_end AND end_tick +5 s, leaves round 2",
      (r1["winner_side"], r1["end_reason"], r1["official_end_tick"] - r0["official_end_tick"],
       r1["end_tick"] - r0["end_tick"]) == ("t", "t_win_elimination", 5 * 128, 5 * 128)
      and flipped.rounds.row(1, named=True) == raw.rounds.row(1, named=True))

# ── a leaky feature is caught ──
section("deliberately leaky features")
orig = builder.snapshot_at


def leaky_duration(sl, *a, **k):
    s = orig(sl, *a, **k)
    full = _CURRENT["raw"].rounds.filter(_CURRENT["raw"].rounds["round_num"] == s.meta.round_num).row(0, named=True)
    s.ctx.time_into_round_s = (full["official_end_tick"] - full["freeze_end_tick"]) / sl.tick_rate
    return s


def leaky_winner(sl, *a, **k):
    s = orig(sl, *a, **k)
    full = _CURRENT["raw"].rounds.filter(_CURRENT["raw"].rounds["round_num"] == s.meta.round_num).row(0, named=True)
    s.players[0].kills_round = float(full["winner_side"] == "ct")
    return s


def leaky_end_tick(sl, *a, **k):
    """The 2026-08-01 bug class: a feature derived from rounds.end_tick (MED-31)."""
    s = orig(sl, *a, **k)
    full = _CURRENT["raw"].rounds.filter(_CURRENT["raw"].rounds["round_num"] == s.meta.round_num).row(0, named=True)
    s.ctx.time_remaining_s = (full["end_tick"] - sl.tick) / sl.tick_rate
    return s


for name, fn in (("round duration", leaky_duration), ("round winner", leaky_winner), ("end_tick", leaky_end_tick)):
    builder.snapshot_at = fn
    try:
        probs = leakage.audit(build_fn, raw)
    finally:
        builder.snapshot_at = orig
    check(f"leaky feature ({name}) is caught", len(probs) > 0, f"{len(probs)} problems")
check("monkeypatch undone: audit clean again", leakage.audit(build_fn, raw) == [])

# ── P3.6.3 poisoned future rows ──
section("poisoned future rows")
for story in STORIES:
    for rnd, tick in ((1, 2500), (1, 3500)):
        r = make_raw_tables(story, 128)
        probs = leakage.poison_check(build_fn, r, rnd, tick)
        check(f"{story}: poison at {tick} changes nothing before it", probs == [], "; ".join(probs[:2]))
r = make_raw_tables("base", 128)
pois = leakage.poison(r, 1, 2500)
check("poisoned raw still validates (dtypes kept)", pois.validate(strict=False) == [], str(pois.validate(strict=False)[:3]))
after = [s for s in build_fn(pois) if s.meta.round_num == 1 and s.meta.tick >= 2500]
check("the poison is not inert: snapshots at/after the poison tick see it",
      any(p.flash_duration_remaining > 100 for s in after for p in s.players if p.alive))

orig_until = builder.Table.until
builder.Table.until = lambda self, tick: self      # a builder that forgets the causal cut
try:
    probs = leakage.poison_check(build_fn, r, 1, 2500)
finally:
    builder.Table.until = orig_until
check("a builder that skips the causal cut is caught by the poison check", len(probs) > 0)

sys.exit(summary())
