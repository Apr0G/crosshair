"""
Parse benchmark (P2.6.2): single-process parse time of one demo (target <= 15 s) and
multi-process throughput (target >= 0.4 demos/s with 6 processes on 24 demos).

    .venv\\Scripts\\python.exe bench\\bench_parse.py <demo.dem> [--dir DEMO_DIR] [--procs 6] [--n 24]

Without a demo path it prints why and exits 2. Throughput runs the real worker
(parse + parquet + manifest) on the first `--n` demos of `--dir` (default: the demo's
own folder) into a temp directory, so `data/engine/parsed/` is never touched.
Results go to `data/engine/bench/parse_<date>.json`.
"""
import argparse
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from _bench import timed, write          # noqa: E402
from engine import paths                 # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

TARGET_SINGLE_S   = 15.0
TARGET_THROUGHPUT = 0.4


def single(demo: Path, stages: dict) -> dict:
    from engine.parse import awpy3_parser
    with timed("parse_single", stages):
        tables, meta = awpy3_parser.parse(demo)
    rows = {k: v.height for k, v in tables.items()}
    print(f"single-process parse: {stages['parse_single']:.2f}s (target <= {TARGET_SINGLE_S:.0f}s)  "
          f"map={meta['map']} tick_rate={meta['tick_rate']} rounds={meta['n_rounds']}")
    print("  rows: " + ", ".join(f"{k}={v}" for k, v in rows.items()))
    return {"rows": rows, "map": meta["map"], "n_rounds": meta["n_rounds"]}


def throughput(demos: list[Path], procs: int) -> dict:
    from engine.parse import worker
    with tempfile.TemporaryDirectory() as d:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=procs) as pool:
            results = list(pool.map(worker.parse_one, demos, [Path(d)] * len(demos),
                                    [None] * len(demos), [True] * len(demos)))
        secs = time.perf_counter() - t0
    ok = sum(r["status"] == "parsed" for r in results)
    rate = ok / secs if secs > 0 else None
    print(f"throughput: {ok}/{len(demos)} demos in {secs:.1f}s with {procs} processes = "
          f"{rate:.3f} demos/s (target >= {TARGET_THROUGHPUT})")
    for r in results:
        if r["status"] != "parsed":
            print(f"  failed {r['match_key']}: {r.get('exc_type')}: {r.get('error')}")
    return {"n": len(demos), "parsed": ok, "seconds": round(secs, 3), "procs": procs,
            "demos_per_s": None if rate is None else round(rate, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("demo", nargs="?", type=Path)
    ap.add_argument("--dir", type=Path, default=None, help="demo folder for the throughput run")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    if a.demo is None or not a.demo.is_file():
        print("bench_parse: no demo given (or not a file); pass a .dem path. Nothing measured.")
        return 2
    stages: dict[str, float] = {}
    extra = {"demo": a.demo.name, "single": single(a.demo, stages)}
    folder = a.dir or a.demo.parent
    demos = sorted(folder.glob("*.dem"))[: a.n]
    if len(demos) >= 2:
        extra["throughput"] = throughput(demos, a.procs)
    else:
        print(f"throughput: skipped (only {len(demos)} demo in {folder})")
    out = write(paths.BENCH_DIR / f"parse_{date.today().isoformat()}.json", stages, extra)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
