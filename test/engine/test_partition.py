"""
Round partition (P8.1): intervals tile [freeze_end, terminal] with no gaps or overlaps;
every window is exactly one `event` interval; grid samples inside a window are dropped;
synthetic tick lists give the expected interval list; W at breakpoints comes from one
batched call with factual histories (count = breakpoints); ΔW telescopes.

    .venv\\Scripts\\python.exe test\\engine\\test_partition.py
"""
import sys
from types import SimpleNamespace

from _harness import check, section, summary
from _attr_states import FREEZE_END, TERMINAL, make_round, round_win
from _stubs import StubPredictor

from engine.attribution import partition
from engine.attribution.partition import DRIFT, EVENT, Interval


def fake(ticks):
    return [SimpleNamespace(meta=SimpleNamespace(tick=t, kinds=("grid",))) for t in ticks]


def win(wid, st, en):
    return {"window_id": wid, "round_num": 1, "start_tick": st, "end_tick": en, "member_action_ids": ["x"]}


def raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


# ── synthetic lists ───────────────────────────────────────────────────────────

section("synthetic tick lists")
snaps = fake([0, 64, 99, 101, 128, 192, 256])
ivs = partition.breakpoints(snaps, [win("w1", 99, 101)])
want = [Interval(0, 64, DRIFT), Interval(64, 99, DRIFT), Interval(99, 101, EVENT, "w1"),
        Interval(101, 128, DRIFT), Interval(128, 192, DRIFT), Interval(192, 256, DRIFT)]
check("one window between grid samples -> expected interval list", ivs == want, str(ivs))

snaps = fake([0, 63, 64, 65, 128])
ivs = partition.breakpoints(snaps, [win("w2", 63, 65)])
want = [Interval(0, 63, DRIFT), Interval(63, 65, EVENT, "w2"), Interval(65, 128, DRIFT)]
check("grid sample inside a window is dropped as a breakpoint", ivs == want, str(ivs))

snaps = fake([0, 64, 127, 128, 129, 130, 131, 192])
ivs = partition.breakpoints(snaps, [win("a", 127, 129), win("b", 129, 131)])
want = [Interval(0, 64, DRIFT), Interval(64, 127, DRIFT), Interval(127, 129, EVENT, "a"),
        Interval(129, 131, EVENT, "b"), Interval(131, 192, DRIFT)]
check("adjacent windows sharing a boundary sample", ivs == want, str(ivs))

snaps = fake([0, 1, 64, 128])
ivs = partition.breakpoints(snaps, [win("f", -1, 1)])
check("window at freeze_end: pre clamped to the first sample",
      ivs[0] == Interval(0, 1, EVENT, "f") and not partition.check_tiling(ivs, 0, 128), str(ivs))

snaps = fake([0, 64, 127, 128])
ivs = partition.breakpoints(snaps, [win("t", 127, 129)])
check("window at the terminal tick: post clamped to the last sample",
      ivs[-1] == Interval(127, 128, EVENT, "t"), str(ivs))

check("no windows -> all drift", all(iv.kind == DRIFT for iv in partition.breakpoints(fake([0, 64, 128]), [])))
check("overlapping windows raise",
      raises(lambda: partition.breakpoints(fake([0, 10, 12, 14, 16, 64]), [win("a", 10, 14), win("b", 12, 16)])))
check("duplicate ticks raise", raises(lambda: partition.breakpoints(fake([0, 64, 64]), [])))
check("window object with attributes works like a dict",
      partition.breakpoints(fake([0, 99, 101, 128]), [SimpleNamespace(**win("o", 99, 101))])[1].window_id == "o")

# ── story round ───────────────────────────────────────────────────────────────

section("story round tiling")
snaps, actions, windows, _ = make_round()
ivs = partition.breakpoints(snaps, windows)
check("tiles [freeze_end, terminal] with no gaps / overlaps",
      partition.check_tiling(ivs, FREEZE_END, TERMINAL) == [], str(partition.check_tiling(ivs, FREEZE_END, TERMINAL)))
ev = [iv for iv in ivs if iv.kind == EVENT]
check("every window is exactly one event interval", sorted(iv.window_id for iv in ev) == sorted(w["window_id"] for w in windows))
check("event interval = (tick - 1, tick + 1) boundary samples",
      all((iv.start_tick, iv.end_tick) == (w["start_tick"], w["end_tick"])
          for iv in ev for w in windows if w["window_id"] == iv.window_id))
check("check_tiling detects a gap", partition.check_tiling([Interval(0, 5, DRIFT), Interval(6, 9, DRIFT)], 0, 9) != [])
check("check_tiling detects an overlap", partition.check_tiling([Interval(0, 6, DRIFT), Interval(5, 9, DRIFT)], 0, 9) != [])

# ── W at breakpoints ──────────────────────────────────────────────────────────

section("W at breakpoints (P8.1.2) and telescoping (P8.1.3)")
seen = []


def rec_win(s, h):
    seen.append((s.meta.tick, [x.meta.tick for x in h]))
    return round_win(s, h)


stub = StubPredictor(win=rec_win)
W = partition.wp_at_breakpoints(stub, snaps, ivs)
bts = partition.breakpoint_ticks(ivs)
check("one batched predictor call", stub.calls == 1, f"calls={stub.calls}")
check("W count equals breakpoints", len(W) == len(bts) == stub.n_snapshots, f"{len(W)} vs {len(bts)}")
check("histories are the factual past only (every earlier snapshot, nothing later)",
      all(all(x < t for x in h) and len(h) == sum(1 for s in snaps if s.meta.tick < t) for t, h in seen))
res = partition.telescoping_residual(ivs, W)
check("Σ ΔW_i = W(terminal) − W(freeze_end) to floating error", abs(res) <= 1e-12, f"residual={res:.3e}")
check("interval deltas count", len(partition.interval_deltas(ivs, W)) == len(ivs))

sys.exit(summary())
