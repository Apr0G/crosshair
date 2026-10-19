"""
State builder timing (P3.8.3): time per stage on a synthetic story (default `base` at
128 tick). Prints one row per stage; no budget assert for synthetic input (the
<= 8 s target applies to the golden demo, which does not exist yet).

    .venv\\Scripts\\python.exe bench\\bench_states.py [--story base] [--tick-rate 128]
                                                    [--repeat 3] [--no-cache] [--out result.json]

Stages: fixture (building the synthetic RawTables, not part of the budget), sampling,
slicing (causal view build + per-snapshot cut), utility, players, context, pairwise,
labels, write (parquet to a temp dir). Seconds are the best of `--repeat` runs.
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

from _bench import write                     # noqa: E402
from _synthetic import STORIES, make_raw_tables   # noqa: E402
from engine import config                    # noqa: E402
from engine.state import builder             # noqa: E402
from engine.state.geometry import FlatGeometry   # noqa: E402

STAGES = ("fixture", "sampling", "slicing", "utility", "players", "context", "pairwise", "labels", "write")


def run_once(story: str, tr: int, los_cache: bool) -> tuple[dict, int]:
    cfg = config.load()
    t = {}
    t0 = time.perf_counter()
    raw = make_raw_tables(story, tr)
    t["fixture"] = time.perf_counter() - t0
    snaps = builder.build(raw, FlatGeometry(), cfg, match_key=f"bench-{story}", tier="pro",
                          los_cache=los_cache, timings=t)
    with tempfile.TemporaryDirectory() as d:
        t0 = time.perf_counter()
        builder.write(snaps, Path(d) / "states.parquet")
        t["write"] = time.perf_counter() - t0
    return t, len(snaps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--story", default="base", choices=STORIES)
    ap.add_argument("--tick-rate", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--no-cache", action="store_true", help="exact LOS (no per-area cache)")
    ap.add_argument("--out", type=Path, default=None, help="write a bench result JSON here")
    args = ap.parse_args()

    best, n = None, 0
    for _ in range(max(1, args.repeat)):
        t, n = run_once(args.story, args.tick_rate, not args.no_cache)
        best = t if best is None else {k: min(best.get(k, v), v) for k, v in t.items()}

    total = sum(v for k, v in best.items() if k != "fixture")
    print(f"bench_states: story={args.story} tick_rate={args.tick_rate} snapshots={n} "
          f"los_cache={not args.no_cache} (best of {args.repeat})")
    print(f"  {'stage':<10} {'seconds':>9} {'ms/snapshot':>12}")
    for k in STAGES:
        v = best.get(k, 0.0)
        print(f"  {k:<10} {v:9.4f} {1000 * v / max(n, 1):12.3f}")
    print(f"  {'total':<10} {total:9.4f} {1000 * total / max(n, 1):12.3f}   (excludes fixture)")
    print("  budget: not asserted for synthetic input (golden-demo target 8 s, P3.8.3)")
    if args.out:
        write(args.out, {k: best.get(k, 0.0) for k in STAGES if k != "fixture"},
              extra={"story": args.story, "tick_rate": args.tick_rate, "snapshots": n})
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
