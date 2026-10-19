"""
Action extractor timing (P4.7.4): time per stage on a synthetic story (default `base` at
128 tick). Prints one row per stage; no budget assert for synthetic input (the <= 3 s
target applies to the golden demo, which does not exist yet).

    .venv\\Scripts\\python.exe bench\\bench_actions.py [--story base] [--tick-rate 128]
                                                     [--repeat 3] [--out result.json]

Stages: fixture (synthetic RawTables) and states (one state build with the event ticks)
are context, not part of the actions budget; then context (tick tracks), events
(raw-table actions), movement, enablers, windows, write (both parquet files to a temp
dir). Seconds are the best of `--repeat` runs.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "engine"))

from _bench import write                          # noqa: E402
from _synthetic import STORIES, make_raw_tables   # noqa: E402
from engine import config                         # noqa: E402
from engine.actions import enablers, extractor, utility, windows   # noqa: E402
from engine.state import builder                  # noqa: E402
from engine.state.geometry import FlatGeometry    # noqa: E402

STAGES  = ("fixture", "states", "context", "events", "movement", "enablers", "windows", "write")
CONTEXT = ("fixture", "states")


def run_once(story: str, tr: int) -> tuple[dict, int, int]:
    cfg, geo, t = config.load(), FlatGeometry(), {}

    def lap(stage, fn):
        t0 = time.perf_counter()
        out = fn()
        t[stage] = t.get(stage, 0.0) + time.perf_counter() - t0
        return out

    raw = lap("fixture", lambda: make_raw_tables(story, tr))
    ev0 = extractor.extract_event_actions(raw, cfg, "bench", geometry=geo)
    snaps = lap("states", lambda: builder.build(raw, geo, cfg, match_key="bench", tier="pro",
                                                action_ticks=extractor.event_ticks(ev0),
                                                durative_windows=extractor.durative_windows(ev0)))
    ctx = lap("context", lambda: extractor.make_context(raw, cfg, "bench", geo, snaps))
    acts = lap("events", lambda: extractor.extract_event_actions(raw, cfg, "bench", geometry=geo, ctx=ctx))
    lap("events", lambda: utility.fill_blocks(acts, ctx))
    acts += lap("movement", lambda: extractor.extract_movement_actions(snaps, raw, cfg, "bench", geo, ctx=ctx))
    acts.sort(key=extractor.sort_key)
    lap("enablers", lambda: enablers.attach(acts, ctx))
    ws = lap("windows", lambda: windows.group(acts, cfg, ctx.tick_rate, rounds=ctx.round_bounds()))
    with tempfile.TemporaryDirectory() as d:
        lap("write", lambda: extractor.write(acts, ws, Path(d)))
    return t, len(acts), len(ws)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--story", default="base", choices=STORIES)
    ap.add_argument("--tick-rate", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None, help="write a bench result JSON here")
    args = ap.parse_args()

    best, n, nw = None, 0, 0
    for _ in range(max(1, args.repeat)):
        t, n, nw = run_once(args.story, args.tick_rate)
        best = t if best is None else {k: min(best.get(k, v), v) for k, v in t.items()}

    total = sum(v for k, v in best.items() if k not in CONTEXT)
    print(f"bench_actions: story={args.story} tick_rate={args.tick_rate} actions={n} windows={nw} "
          f"(best of {args.repeat})")
    print(f"  {'stage':<10} {'seconds':>9}")
    for k in STAGES:
        note = "   (context, not in total)" if k in CONTEXT else ""
        print(f"  {k:<10} {best.get(k, 0.0):9.4f}{note}")
    print(f"  {'total':<10} {total:9.4f}   (actions stage: context .. write)")
    print("  budget: not asserted for synthetic input (golden-demo target 3 s, P4.7.4)")
    if args.out:
        write(args.out, {k: best.get(k, 0.0) for k in STAGES if k not in CONTEXT},
              extra={"story": args.story, "tick_rate": args.tick_rate, "actions": n, "windows": nw})
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
