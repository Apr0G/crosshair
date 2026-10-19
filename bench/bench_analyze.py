"""
End-to-end time budget on the golden demo (P0.6, P10.3; condition C3).

    .venv\\Scripts\\python.exe bench\\bench_analyze.py [--device gpu|cpu] [--assert]
                                                  [--model-dir DIR | --stub] [--synthetic]

Runs `engine.analyze` stage by stage (parse, states, actions, predict, attribution,
value, report: the stage names of `bench/budget.yaml`), times each with
`bench/_bench.timed`, and prints one row per stage: budget, measured seconds, ratio,
status.

Modes:
- golden (default): `analyze.analyze_demo` on the pinned golden demo with the model in
  `--model-dir` (any model, adopted or not), or with the rule stub (`--stub`, times
  every stage except a real `predict`). Exit 2 when the golden demo is not pinned or
  missing, or when neither a model nor `--stub` is given. The result JSON goes to
  `data/engine/bench/analyze_<date>.json`.
- `--synthetic`: `analyze.analyze_tables` on the synthetic `base` story with
  `RuleStub(good_win)` and FlatGeometry into a temp dir (no DB rows, no result file),
  so the harness itself is exercised without a demo or a model. `parse` is n/a there.

n/a behaviour: a stage that did not run (no parse in --synthetic, value stage skipped
without Q / neighbour models) is printed as `n/a` and never judged. With `--assert`
the script exits 1 only when a measured stage exceeds its budget by more than
`over_budget_factor` (50%), or the measured total exceeds the hard ceiling for the
device.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from _bench import write                 # noqa: E402  (stage timing: engine.analyze uses _bench.timed)
from engine import paths                 # noqa: E402

BUDGET_PATH = HERE / "budget.yaml"
GOLDEN_JSON = ROOT / "test" / "engine" / "golden" / "GOLDEN.json"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

STAGES = ("parse", "states", "actions", "predict", "attribution", "value", "report")


def load_budget(path: Path = BUDGET_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def golden_demo() -> Path | None:
    """The pinned golden demo, or None when it is not pinned or not on disk."""
    try:
        meta = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not meta.get("demo_sha256"):
        return None
    if meta.get("demo_file"):
        p = paths.DEMOS_GOLDEN_DIR / meta["demo_file"]
        return p if p.is_file() else None
    dems = sorted(paths.DEMOS_GOLDEN_DIR.glob("*.dem"))
    return dems[0] if len(dems) == 1 else None


def _stage_seconds(timings: dict, value_stage: str) -> dict[str, float | None]:
    out = {st: timings.get(st) for st in STAGES}
    if value_stage != "done":
        out["value"] = None                     # skipped stage: n/a, never judged
    return out


def measure_golden(demo: Path, device: str, model_dir: Path | None, stub: bool,
                   value_dirs: tuple | None = None) -> dict[str, float | None]:
    import tempfile
    from engine import analyze, config
    pred = analyze.stub_predictor() if stub else None
    with tempfile.TemporaryDirectory() as tmp:
        res = analyze.analyze_demo(demo, tier="pro", out_dir=Path(tmp) / "report", model_dir=model_dir,
                                   device="cuda" if device == "gpu" else "cpu", cfg=config.load(),
                                   allow_unadopted=True, predictor=pred, write_db=False, value_dirs=value_dirs)
    return _stage_seconds(res.timings, res.value_stage)


def measure_synthetic() -> dict[str, float | None]:
    import tempfile
    sys.path.insert(0, str(ROOT / "test" / "engine"))
    from _cf_states import RuleStub, good_win
    from _synthetic import make_raw_tables
    from engine import analyze, config
    from engine.state.geometry import FlatGeometry
    with tempfile.TemporaryDirectory() as tmp:
        res = analyze.analyze_tables(make_raw_tables("base"), match_key="bench_synthetic", tier="pro",
                                     geometry=FlatGeometry(), predictor=RuleStub(good_win), cfg=config.load(),
                                     out_dir=Path(tmp) / "report", write_db=False)
    return _stage_seconds(res.timings, res.value_stage)


def evaluate(measured: dict[str, float | None], budget: dict, device: str) -> tuple[list, list]:
    """Rows (stage, budget_s, measured_s, ratio, status) and the list of failures."""
    factor = float(budget.get("over_budget_factor", 1.5))
    rows, failures = [], []
    for stage, limits in budget["stages"].items():
        b, m = float(limits[device]), measured.get(stage)
        if m is None:
            rows.append((stage, b, None, None, "n/a"))
            continue
        ratio = m / b
        status = "FAIL" if ratio > factor else "over" if ratio > 1 else "ok"
        if status == "FAIL":
            failures.append(f"{stage}: {m:.1f}s > {factor:.1f} x budget {b:.0f}s")
        rows.append((stage, b, m, ratio, status))

    done    = [m for m in measured.values() if m is not None]
    total_b = float(budget["total"][device])
    ceiling = float(budget["hard_ceiling"][device])
    partial = any(measured.get(st) is None for st in STAGES)   # a stage not timed: the total is partial (MED-28)
    if not done:
        rows.append(("total", total_b, None, None, "n/a"))
    else:
        total = sum(done)
        status = "FAIL" if total > ceiling else "over" if total > total_b else "ok"
        if status == "FAIL":
            failures.append(f"total: {total:.1f}s > hard ceiling {ceiling:.0f}s ({device})")
        if partial:
            status += " (partial: " + ", ".join(st for st in STAGES if measured.get(st) is None) + " not timed)"
        rows.append(("total", total_b, total, total / total_b, status))
    return rows, failures


def print_table(rows: list, device: str, ceiling: float) -> None:
    print(f"\n  bench analyze  device={device}  hard ceiling={ceiling:.0f}s\n")
    print(f"  {'stage':<12} {'budget s':>9} {'measured s':>11} {'ratio':>7}  status")
    print(f"  {'-' * 12} {'-' * 9} {'-' * 11} {'-' * 7}  {'-' * 6}")
    for stage, b, m, ratio, status in rows:
        ms = f"{m:.2f}" if m is not None else "n/a"
        rs = f"{ratio:.2f}" if ratio is not None else "n/a"
        print(f"  {stage:<12} {b:>9.0f} {ms:>11} {rs:>7}  {status}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Time the pipeline on the golden demo against bench/budget.yaml.")
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--assert", dest="assert_", action="store_true",
                    help="exit 1 if a measured stage is >50%% over budget or the total exceeds the hard ceiling")
    ap.add_argument("--synthetic", action="store_true",
                    help="time the synthetic base story with a rule stub (no demo, no model)")
    ap.add_argument("--model-dir", type=Path, default=None, help="win model directory for the golden run")
    ap.add_argument("--stub", action="store_true", help="golden run with the rule stub instead of a model")
    ap.add_argument("--q-model", default=None, help="Q/B model name or directory: times the value stage too")
    ap.add_argument("--neighbours", default=None, help="neighbour index name or directory (with --q-model)")
    args = ap.parse_args(argv)
    if (args.q_model is None) != (args.neighbours is None):
        print("the value stage needs both --q-model and --neighbours")
        return 2
    value_dirs = None
    if args.q_model:
        def _dir(x):
            return Path(x) if Path(x).is_dir() else paths.MODELS_DIR / x
        value_dirs = (_dir(args.q_model), _dir(args.neighbours))

    budget = load_budget()
    if args.synthetic:
        measured = measure_synthetic()
    else:
        demo = golden_demo()
        if demo is None:
            print("golden demo not pinned or missing (see test/engine/golden/GOLDEN.json); cannot bench "
                  "(use --synthetic to exercise the harness)")
            return 2
        if args.model_dir is None and not args.stub:
            print("no model: pass --model-dir DIR (or --stub to time everything but a real predict)")
            return 2
        measured = measure_golden(demo, args.device, args.model_dir, args.stub, value_dirs)

    rows, failures = evaluate(measured, budget, args.device)
    print_table(rows, args.device, float(budget["hard_ceiling"][args.device]))
    if args.synthetic:
        print("\n  synthetic run (base story, rule stub): harness check only, no result file written")
    else:
        out = paths.BENCH_DIR / f"analyze_{date.today().isoformat()}.json"
        write(out, {s: m for s, m in measured.items() if m is not None},
              extra={"device": args.device, "failures": failures, "stub": bool(args.stub),
                     "value_timed": measured.get("value") is not None,
                     "partial": [st for st in STAGES if measured.get(st) is None]})
        print(f"\n  wrote {out}")
    if failures:
        print("\n  over budget:\n    " + "\n    ".join(failures))
    return 1 if (args.assert_ and failures) else 0


if __name__ == "__main__":
    sys.exit(main())
