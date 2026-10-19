"""
Tick selection (P3.1): grid step and count, boundary ticks around actions, clipping,
kinds, monotone unique ticks.

    .venv\\Scripts\\python.exe test\\engine\\test_sampler.py
"""
import math
import sys

from _harness import check, section, summary
from _synthetic import STORY_FACTS, make_raw_tables, scale_tick

from engine.state import sampler

ROUND = {"round_num": 1, "start_tick": 900, "freeze_end_tick": 1000, "end_tick": 4040,
         "official_end_tick": 4360}

# ── P3.1.1 grid ──
section("grid ticks")
check("step 64 at 128 tick / 2 Hz", sampler.step_ticks(128, 2) == 64)
check("step 32 at 64 tick / 2 Hz", sampler.step_ticks(64, 2) == 32)
for tr in (64, 128):
    for hz in (1, 2, 4, 8):
        step = sampler.step_ticks(tr, hz)
        even = {"round_num": 1, "freeze_end_tick": 1000, "official_end_tick": 1000 + 40 * tr}
        g = sampler.grid_ticks(even, tr, hz)
        dur = 40.0
        check(f"{tr} tick {hz} Hz: count = floor(dur*hz)+1 when the duration divides",
              len(g) == math.floor(dur * hz) + 1, f"{len(g)}")
        check(f"{tr} tick {hz} Hz: first = freeze_end, last = terminal, step {step}",
              g[0] == 1000 and g[-1] == 1000 + 40 * tr
              and all(b - a == step for a, b in zip(g, g[1:])))
g = sampler.grid_ticks(ROUND, 128, 2)
check("uneven duration: floor(dur*hz)+1 regular ticks plus the terminal tick",
      len(g) == math.floor((4360 - 1000) / 128 * 2) + 2 and g[-1] == 4360 and g[-2] == 1000 + 52 * 64,
      f"{len(g)} {g[-3:]}")
check("grid unique and monotone", g == sorted(set(g)))
check("terminal falls back to end_tick when official_end is null",
      sampler.terminal_tick({**ROUND, "official_end_tick": None}) == 4040
      and sampler.terminal_tick({"official_end_tick": None, "end_tick": None}) is None)

# ── P3.1.2 boundary ──
section("boundary ticks")
b = sampler.boundary_ticks([2000, 4000], offset=1, lo=1000, hi=4360)
check("kills at 2000/4000 -> {1999, 2001, 3999, 4001}", set(b) == {1999, 2001, 3999, 4001}, str(sorted(b)))
check("pre/post kinds", b[1999] == {"boundary_pre"} and b[2001] == {"boundary_post"})
b = sampler.boundary_ticks([4360], offset=1, lo=1000, hi=4360)
check("kill on the last tick: post clips to the terminal tick", set(b) == {4359, 4360} and "boundary_post" in b[4360])
b = sampler.boundary_ticks([1000], offset=1, lo=1000, hi=4360)
check("event at freeze_end: pre clips to freeze_end", set(b) == {1000, 1001})
b = sampler.boundary_ticks([], [(1500, 3804), (2500, None)], offset=1, lo=1000, hi=4360)
check("durative windows: both ends, unknown end skipped", set(b) == {1499, 1501, 3803, 3805, 2499, 2501})
b = sampler.boundary_ticks([2000], offset=3)
check("offset honoured", set(b) == {1997, 2003})
b = sampler.boundary_ticks([2000, 2000, 2002], offset=1)
check("dedupe; a tick can be both pre and post", sorted(b) == [1999, 2001, 2003] and b[2001] == {"boundary_pre", "boundary_post"})

# ── P3.1.3 union + kinds ──
section("all_ticks")
at = sampler.all_ticks(ROUND, 128, 2, action_ticks={2000, 4000}, offset=1)
ticks = [t for t, _ in at]
kinds = dict(at)
check("no duplicate ticks, monotone", ticks == sorted(set(ticks)))
check("grid tick kinds", kinds[1000] == ("grid",) and kinds[1064] == ("grid",))
check("boundary kinds", kinds[1999] == ("boundary_pre",) and kinds[2001] == ("boundary_post",)
      and kinds[3999] == ("boundary_pre",) and kinds[4001] == ("boundary_post",))
check("terminal tick carries grid + terminal", kinds[4360] == ("grid", "terminal"))
check("exactly one terminal", sum("terminal" in k for _, k in at) == 1)
check("count = grid + 4 boundary", len(at) == len(sampler.grid_ticks(ROUND, 128, 2)) + 4)
at2 = sampler.all_ticks(ROUND, 128, 2, action_ticks={1064}, offset=0)
check("offset 0: boundary on a grid tick merges kinds",
      dict(at2)[1064] == ("grid", "boundary_pre", "boundary_post"))

# ── synthetic stories at both rates ──
section("synthetic rounds")
for tr in (64, 128):
    raw = make_raw_tables("base", tr)
    rr = raw.rounds.to_dicts()[0]
    kills = [scale_tick(t, tr) for t, _, _ in STORY_FACTS["base"]["kill_events"]]
    at = sampler.all_ticks(rr, tr, 2, action_ticks=set(kills))
    check(f"{tr} tick: first is freeze_end, last is official_end",
          at[0][0] == rr["freeze_end_tick"] and at[-1][0] == rr["official_end_tick"])
    check(f"{tr} tick: boundary samples at kill +/- 1",
          all(k - 1 in dict(at) and k + 1 in dict(at) for k in kills))
    check(f"{tr} tick: end tick never sampled for its own sake",
          rr["end_tick"] not in dict(at) or "grid" in dict(at)[rr["end_tick"]])

sys.exit(summary())
