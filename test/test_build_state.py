"""
Tests for state_sampler.build_state — the arbitrary-tick state constructor.

Run from repo root:
    .venv/bin/python test/test_build_state.py [path/to/demo.dem]

Defaults to demos/faceit2.dem. Needs awpy (parses a real demo); no model, no DB.

build_state is the foundation of event-boundary impact attribution, so the
properties that matter are:
  - it is a PURE function of (tick, ctx): same input, same output, no order effects
  - it reproduces the 1 Hz grid exactly (sample_round_states is now a caller)
  - it works OFF the grid, which the sampler never exercised
  - it never looks into the future (the CRIT-02 class of bug)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_results: list[tuple[bool, str]] = []


def check(desc: str, ok: bool, detail: str = ""):
    _results.append((bool(ok), desc))
    print(f"  {PASS if ok else FAIL}  {desc}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    import extract
    import feature_extractor
    import state_sampler as ss

    demo = sys.argv[1] if len(sys.argv) > 1 else "demos/faceit2.dem"
    if not Path(demo).exists():
        print(f"no demo at {demo} — pass one as argv[1]")
        return 2

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        tables = extract.extract(demo)
        map_name = tables.get("map_name", "unknown")
        vis = feature_extractor.build_vis_checker(map_name)
        ctxs = list(ss.iter_round_contexts(tables, "t_bs", map_name, vis))
        grid = ss.sample_round_states(tables, "t_bs", map_name, vis)

    print(f"\n{demo}  ({map_name}, tick_rate={tables.get('tick_rate')})")
    print(f"  {len(ctxs)} rounds, {len(grid)} grid states\n")

    if not ctxs:
        check("demo yields at least one round context", False)
        return _summary()

    ctx = max(ctxs, key=lambda c: c.r_end - c.r_start)

    print("── purity ───────────────────────────────────────────────────────────────")
    mid = int((ctx.r_start + ctx.r_end) // 2)
    a, b = ss.build_state(mid, ctx), ss.build_state(mid, ctx)
    check("same (tick, ctx) gives an identical dict", a == b)

    # Evaluating out of order must not change anything — attribution jumps around.
    ticks = [int(ctx.r_start) + k for k in (500, 100, 900, 300)]
    fwd = {t: ss.build_state(t, ctx) for t in ticks}
    rev = {t: ss.build_state(t, ctx) for t in reversed(ticks)}
    check("evaluation order does not affect results", fwd == rev)

    print("\n── reproduces the 1 Hz grid exactly ─────────────────────────────────────")
    rebuilt = []
    for c in ctxs:
        for t in range(int(c.r_start), int(c.r_end), c.sample_interval):
            st = ss.build_state(t, c)
            if st is not None:
                rebuilt.append(st)
    check("build_state on the grid == sample_round_states",
          rebuilt == grid, f"{len(rebuilt)} vs {len(grid)}")

    print("\n── works OFF the grid (what the sampler never did) ───────────────────────")
    off = [ss.build_state(int(ctx.r_start) + o, ctx) for o in (37, 91, 143, 205, 511)]
    got = [s for s in off if s is not None]
    check("off-grid ticks produce states", len(got) >= 4, f"{len(got)}/5")
    check("off-grid ticks are NOT all on the sample grid",
          any((s["tick"] - int(ctx.r_start)) % ctx.sample_interval != 0 for s in got))

    print("\n── adjacent ticks: the attribution use case ─────────────────────────────")
    # An action at tick T is valued as WP(T+1) - WP(T-1). Those must be buildable
    # and must differ only slightly in a quiet moment.
    before, after = ss.build_state(mid - 1, ctx), ss.build_state(mid + 1, ctx)
    check("state exists at tick-1 and tick+1", before is not None and after is not None)
    if before and after:
        check("adjacent states carry distinct ticks", before["tick"] != after["tick"])
        check("adjacent states agree on alive counts in a quiet window",
              before["alive_ct"] == after["alive_ct"] and before["alive_t"] == after["alive_t"])

    print("\n── field contract ───────────────────────────────────────────────────────")
    from win_probability import FEATURES
    st = ss.build_state(mid, ctx)
    missing = [f for f in FEATURES if f not in st]
    check("state covers every model FEATURE", not missing, f"missing: {missing}")
    check("round_won_ct is the context's label, not re-derived",
          st["round_won_ct"] == ctx.round_won_ct)
    check("tick is echoed back exactly", st["tick"] == mid)

    print("\n── clocks stay inside legal bounds ──────────────────────────────────────")
    bad_pre  = [s for s in grid if not s["post_plant"] and s["time_remaining_s"] > ss.ROUND_TIME_S]
    bad_post = [s for s in grid if s["post_plant"] and s["time_remaining_s"] > ss.C4_TIMER]
    check(f"pre-plant time_remaining_s <= {ss.ROUND_TIME_S}", not bad_pre, f"{len(bad_pre)} violations")
    check(f"post-plant time_remaining_s <= {ss.C4_TIMER}", not bad_post, f"{len(bad_post)} violations")
    check("time_into_round_s never negative",
          all(s["time_into_round_s"] >= 0 for s in grid))

    print("\n── no lookahead (CRIT-02 class) ─────────────────────────────────────────")
    # Truncating the round's tick data after T must not change the state at T.
    probe = int(ctx.r_start) + 400
    full = ss.build_state(probe, ctx)
    if full is not None:
        import copy
        trunc = copy.copy(ctx)
        trunc.r_ticks = ctx.r_ticks[ctx.r_ticks["tick"] <= probe + ctx.snap_half]
        trunc.vis_cache = {k: v for k, v in ctx.vis_cache.items() if k <= probe}
        cut = ss.build_state(probe, trunc)
        same = cut is not None and all(
            cut[k] == full[k] for k in ("alive_ct", "alive_t", "total_hp_ct", "total_hp_t",
                                        "ct_spotted_count", "t_spotted_count"))
        check("state at T is unchanged when future ticks are removed", same)
    else:
        check("state at T is unchanged when future ticks are removed", False, "no state at probe")

    print("\n── every window scales to the demo's real tick rate ─────────────────────")
    tr = ctx.tick_rate
    for name, ticks, want_s in (("sample_interval", ctx.sample_interval, ss.SAMPLE_INTERVAL_S),
                                ("snap_half",       ctx.snap_half,       ss.SNAP_HALF_S),
                                ("heard_window",    ctx.heard_window,    ss.HEARD_WINDOW_S)):
        got_s = ticks / tr
        check(f"{name} is {want_s}s at {tr} tick", abs(got_s - want_s) < 0.01, f"{got_s:.2f}s")
    grid_ticks = sorted({s["tick"] for s in grid if s["round_num"] == ctx.round_num})
    gaps = {round((b - a) / tr, 3) for a, b in zip(grid_ticks, grid_ticks[1:])}
    check("consecutive samples are 1s apart in REAL time", gaps <= {ss.SAMPLE_INTERVAL_S}, str(sorted(gaps)[:4]))

    print("\n── sentinels stay out of the data ───────────────────────────────────────")
    check("ct_spread is None (not 0.0) whenever fewer than 2 CT alive",
          all(s["ct_spread"] is None for s in grid if s["alive_ct"] < 2))
    check("no state has both sides eliminated", all(s["alive_ct"] or s["alive_t"] for s in grid))

    return _summary()


def _summary() -> int:
    passed = sum(1 for ok, _ in _results if ok)
    failed = [d for ok, d in _results if not ok]
    print(f"\n{'─'*60}\n  {passed}/{len(_results)} passed")
    if failed:
        print("  Failed:")
        for d in failed:
            print(f"    - {d}")
    print()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
