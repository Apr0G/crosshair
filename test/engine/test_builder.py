"""
State builder (P3.8.1) end to end on every synthetic story at 64 and 128 tick: builds,
every snapshot validates, snapshot counts match an independent count derived from
STORY_FACTS and the story constants, slot order, the fast causal view equals the slow
`slice_until` path, action_ticks overrides, the double-kill boundary pair, C's retreat.

    .venv\\Scripts\\python.exe test\\engine\\test_builder.py
"""
import math
import sys

from _harness import check, section, summary
from _synthetic import BOMB_POS, PLAYERS, STORIES, STORY_FACTS, make_raw_tables

from engine import config
from engine.state import builder, players
from engine.state.geometry import FlatGeometry

CFG = config.load()
GEO = FlatGeometry()
HZ  = CFG.sampling.sample_hz
OFF = CFG.sampling.boundary_offset_ticks


def expected_ticks(story: str, tr: int) -> dict[int, set[int]]:
    """Independent count: per round, grid ticks every tr/HZ from freeze_end plus the
    terminal, and event ticks +/- OFF, from STORY_FACTS + the _synthetic.py docstring."""
    f = STORY_FACTS[story]
    T = lambda t128: int(round(t128 * tr / 128))  # noqa: E731
    S = lambda s: int(round(s * tr))               # noqa: E731
    out = {}
    # round 1
    fe, term = T(1000), T(f["r1_official_end"])
    ev = {T(t) for t, _, _ in f["kill_events"]} | {T(2900), T(3000), T(1800)}
    ev |= {T(1500), T(1500) + S(18)}                       # smoke start / end
    if story == "flash":
        ev.add(T(1900))
    if "explode" in f["bomb_round"][1]:
        ev.add(T(f["r1_end"]))
    if "defuse_start" in f["bomb_round"][1]:
        ev |= {T(3900), T(4200)}
    out[1] = (fe, term, ev)
    # round 2
    start = T(f["r1_official_end"]) + S(5)
    fe2 = start + (T(1000) - T(900))
    end2 = fe2 + S(20)
    out[2] = (fe2, end2 + S(2.5), {fe2 + S(5), fe2 + S(8), fe2 + S(15), end2})
    ticks = {}
    step = tr // HZ
    for r, (fe, term, ev) in out.items():
        grid = set(range(fe, term + 1, step)) | {term}
        bound = {min(term, max(fe, e + d)) for e in ev for d in (-OFF, OFF)}
        ticks[r] = grid | bound
    return ticks


# ── every story builds and validates ──
section("every story at 64 and 128 tick")
for tr in (64, 128):
    for story in STORIES:
        raw = make_raw_tables(story, tr)
        try:
            snaps = builder.build(raw, GEO, CFG, match_key=f"{story}-{tr}", tier="pro", debug=True)
            ok, err = True, ""
        except AssertionError as e:
            snaps, ok, err = [], False, str(e)[:200]
        check(f"{story} @ {tr}: builds with debug validation", ok, err)
        if not snaps:
            continue
        want = expected_ticks(story, tr)
        got = {r: {s.meta.tick for s in snaps if s.meta.round_num == r} for r in (1, 2)}
        check(f"{story} @ {tr}: snapshot ticks match the independent count "
              f"({len(want[1])} + {len(want[2])})", got == want,
              f"r1 extra {sorted(got[1] - want[1])[:4]} missing {sorted(want[1] - got[1])[:4]}; "
              f"r2 extra {sorted(got[2] - want[2])[:4]} missing {sorted(want[2] - got[2])[:4]}")
        check(f"{story} @ {tr}: no duplicate (round, tick)",
              len({(s.meta.round_num, s.meta.tick) for s in snaps}) == len(snaps))
        order = all([p.side for p in s.players] == ["ct"] * 5 + ["t"] * 5 for s in snaps)
        ids = [players.player_id_hash(PLAYERS[w][0]) for w in "ABCD"]
        first = snaps[0].players
        check(f"{story} @ {tr}: slots CT then T, by steamid (A, B, pads, C, D, pads)",
              order and [first[i].player_id for i in (0, 1, 5, 6)] == ids
              and all(first[i].player_id == players.PAD_PLAYER_ID for i in (2, 3, 4, 7, 8, 9)))
        check(f"{story} @ {tr}: meta (match_key, t_s from freeze_end)",
              snaps[0].meta.match_key == f"{story}-{tr}" and snaps[0].meta.t_s == 0.0)

# ── fast view == slow slice_until ──
section("fast causal view equals slice_until")
for tr in (64, 128):
    raw = make_raw_tables("base", tr)
    fast = builder.build(raw, GEO, CFG, match_key="eq", tier="pro")
    slow = builder.build(raw, GEO, CFG, match_key="eq", tier="pro", causal="slow")
    diff = [s.meta.tick for s, t in zip(fast, slow) if s != t]
    check(f"base @ {tr}: {len(fast)} snapshots identical", len(fast) == len(slow) and not diff, str(diff[:5]))
raw = make_raw_tables("flash", 128)
fast = builder.build(raw, GEO, CFG, match_key="eq", tier="pro")
slow = builder.build(raw, GEO, CFG, match_key="eq", tier="pro", causal="slow")
check("flash @ 128: identical too", fast == slow)

# ── action_ticks override ──
section("action_ticks / durative_windows")
raw = make_raw_tables("base", 128)
only = builder.build(raw, GEO, CFG, match_key="a", tier="pro", action_ticks={1: {2500}}, durative_windows={})
b1 = sorted(s.meta.tick for s in only if s.meta.round_num == 1 and s.meta.kinds != ("grid",)
            and "terminal" not in s.meta.kinds)
check("given action_ticks replace the raw-event defaults", b1 == [2499, 2501], str(b1))
check("a round missing from action_ticks gets no boundary samples",
      all(s.meta.kinds in (("grid",), ("grid", "terminal")) for s in only if s.meta.round_num == 2))

# ── double kill on one tick ──
section("double kill on the same tick")
raw = make_raw_tables("double_kill_same_tick", 128)
snaps = builder.build(raw, GEO, CFG, match_key="dk", tier="pro")
near = [(s.meta.tick, s.meta.kinds) for s in snaps if s.meta.round_num == 1 and 1990 <= s.meta.tick <= 2010]
check("one boundary pair at 2000 (1999 pre, 2001 post)",
      [(t, k) for t, k in near if k != ("grid",)] == [(1999, ("boundary_pre",)), (2001, ("boundary_post",))], str(near))
s2001 = next(s for s in snaps if s.meta.tick == 2001)
s1999 = next(s for s in snaps if s.meta.tick == 1999)
check("both victims alive at 1999, both dead at 2001",
      (s1999.ctx.alive_ct, s1999.ctx.alive_t) == (2, 2) and (s2001.ctx.alive_ct, s2001.ctx.alive_t) == (1, 1))

# ── retreat ──
section("retreat: C moves away from the bomb")
for tr in (64, 128):
    raw = make_raw_tables("retreat", tr)
    snaps = builder.build(raw, GEO, CFG, match_key="rt", tier="pro")
    T = lambda t128: int(round(t128 * tr / 128))  # noqa: E731
    c_slot = 5
    seg = [s for s in snaps if s.meta.round_num == 1 and T(3000) <= s.meta.tick <= T(3640)]
    dist = [math.dist((s.players[c_slot].x, s.players[c_slot].y), BOMB_POS) for s in seg]
    check(f"@ {tr}: C's distance to the bomb never decreases over {len(seg)} snapshots",
          len(seg) > 5 and all(b >= a for a, b in zip(dist, dist[1:])))
    r1 = [s for s in snaps if s.meta.round_num == 1]
    before = max((s for s in r1 if s.meta.tick <= T(3000)), key=lambda s: s.meta.tick).players[c_slot]
    after  = min((s for s in r1 if s.meta.tick >= T(3640)), key=lambda s: s.meta.tick).players[c_slot]
    moved  = math.dist((before.x, before.y), (after.x, after.y))
    check(f"@ {tr}: C moves {STORY_FACTS['retreat']['retreat']['units']:.0f} u straight up +y",
          abs(moved - STORY_FACTS["retreat"]["retreat"]["units"]) < 1e-3 and after.y > before.y
          and after.x == before.x, f"{moved:.1f}")
    check(f"@ {tr}: C survives (alive throughout round 1)", all(s.players[c_slot].alive == 1 for s in snaps if s.meta.round_num == 1))

# ── debug flag ──
section("debug flag")
raw = make_raw_tables("base", 64)
orig = builder.snapshot_at


def broken(*a, **k):
    s = orig(*a, **k)
    s.ctx.alive_ct = 9
    return s


builder.snapshot_at = broken
try:
    try:
        builder.build(raw, GEO, CFG, match_key="d", tier="pro", debug=True)
        raised = False
    except AssertionError:
        raised = True
    quiet = builder.build(raw, GEO, CFG, match_key="d", tier="pro")
finally:
    builder.snapshot_at = orig
check("debug=True asserts on an invalid snapshot", raised)
check("debug=False does not (returns the snapshots)", len(quiet) > 0)
try:
    builder.build(raw, GEO, CFG, match_key="d", tier="pro", causal="maybe")
    bad_arg = False
except ValueError:
    bad_arg = True
check("unknown causal mode rejected", bad_arg)
try:
    builder.build(raw, GEO, CFG, match_key="d", tier="silver")
    bad_tier = False
except ValueError:
    bad_tier = True
check("tier is the P3.2.7 enum", bad_tier and builder.TIERS == ("pro", "faceit_10", "faceit_8_9", "other")
      and {p.tier for s in quiet for p in s.players} == {"pro"} and {s.ctx.tier for s in quiet} == {"pro"})

sys.exit(summary())
