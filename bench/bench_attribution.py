"""
Attribution timing (P8.6.5) on synthetic rounds built from `test/engine/_cf_states.py`
(via `_attr_states.py`) with a stub predictor, so the numbers are the engine's own cost
(interventions, coalition building, Shapley) without model inference. Counts predictor
calls and coalition evaluations; prints seconds (best of `--repeat`). No budget assert:
the <= 15 s GPU / <= 90 s CPU per-map target applies to the real model on a real map.

    .venv\\Scripts\\python.exe bench\\bench_attribution.py [--repeat 3] [--rounds 24] [--out result.json]

Cases: `story` (the 4-window `_attr_states` round, exact), `story_sampled` (same with
shapley_exact_max 2, so the 3-member window is sampled with shapley_samples
permutations), `window5` (the 5-member `make_window` alone, exact) and
`window5_sampled` (forced sampled). `per_map_s` multiplies the story time by
`--rounds` (an extrapolation, not a measurement).
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "engine"))

from _bench import write                                         # noqa: E402
from _attr_states import make_round, round_hazard, round_win     # noqa: E402
from _cf_states import action, good_win, make_window             # noqa: E402
from _stubs import StubPredictor                                 # noqa: E402
from engine import config                                        # noqa: E402
from engine.attribution import engine                            # noqa: E402


def window5():
    s_pre, s_post, members = make_window()
    extra = []
    for m in members:
        if m.type in ("smoke_pop", "decoy_start"):
            par = action("smoke" if m.type == "smoke_pop" else "decoy", m.actor_id_hash, kind="durative")
            m.params["parent_action_id"] = par.action_id
            extra.append(par)
    w = {"window_id": "w5", "round_num": 7, "start_tick": 4999, "end_tick": 5001,
         "member_action_ids": [m.action_id for m in members]}
    return [s_pre, s_post], members + extra, [w]


def run_case(build, win, hazard, cfg, repeat: int) -> dict:
    best, ra, stub = None, None, None
    for _ in range(repeat):
        snaps, acts, wins = build()
        stub = StubPredictor(win=win, hazard=hazard)
        t0 = time.perf_counter()
        ra = engine.attribute_round(snaps, acts, wins, stub, cfg)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return {"seconds": best, "predictor_calls": stub.calls, "snapshots_predicted": stub.n_snapshots,
            "coalition_evals": ra.stats["coalition_evals"], "windows": ra.stats["n_windows"],
            "drift": ra.stats["n_drift"], "residual": ra.conservation_residual}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = config.load()
    sampled = replace(cfg, attribution=replace(cfg.attribution, shapley_exact_max=2))
    story = lambda: make_round()[:3]                                   # noqa: E731
    cases = {
        "story":           run_case(story, round_win, round_hazard, cfg, args.repeat),
        "story_sampled":   run_case(story, round_win, round_hazard, sampled, args.repeat),
        "window5":         run_case(window5, good_win, None, cfg, args.repeat),
        "window5_sampled": run_case(window5, good_win, None, sampled, args.repeat),
    }
    print(f"{'case':16s} {'seconds':>9s} {'calls':>6s} {'windows':>8s} {'drift':>6s} "
          f"{'coal_evals':>11s} {'snaps_pred':>11s} {'residual':>10s}")
    for name, r in cases.items():
        print(f"{name:16s} {r['seconds']:9.4f} {r['predictor_calls']:6d} {r['windows']:8d} {r['drift']:6d} "
              f"{r['coalition_evals']:11d} {r['snapshots_predicted']:11d} {r['residual']:10.1e}")
    per_map = cases["story"]["seconds"] * args.rounds
    print(f"\nper_map_s (story x {args.rounds} rounds, extrapolated, stub predictor): {per_map:.3f}")
    if args.out:
        write(args.out, {k: v["seconds"] for k, v in cases.items()},
              extra={"cases": cases, "per_map_s_extrapolated": per_map, "rounds": args.rounds})
    return 0


if __name__ == "__main__":
    sys.exit(main())
