"""
Predictor benchmark (P6.4.2): snapshots/s at batch 512 on GPU and CPU, on fake states
(test/engine/_fake_states.py) with an untrained default-size model (speed does not
depend on the weights). Targets: >= 5,000/s GPU, >= 300/s CPU.

Devices: GPU fp32 (the default Predictor), GPU bf16 (`Predictor(bf16=True)`), CPU fp32.
Two cases per run:
- cold: empty cache, every window snapshot is tensorized and encoded;
- warm history: the factual histories are already cached and every present snapshot
  is an edit (the attribution pattern), so only the present is encoded.

    .venv\\Scripts\\python.exe bench\\bench_predict.py [--n 2048] [--no-cpu]

Writes `data/engine/bench/predict_<date>.json`.
"""
import argparse
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "engine"))

import torch                                          # noqa: E402

from _bench import write                              # noqa: E402
from _fake_states import fake_arrays, finish, histories, make_model_dir   # noqa: E402
from engine import paths                              # noqa: E402
from engine.model.predict import Predictor            # noqa: E402

TARGET = {"cuda": 5000.0, "cpu": 300.0}


def bench(model_dir: Path, device: str, snaps: list, hist: list, bf16: bool = False) -> dict:
    pred = Predictor(model_dir, device, batch_size=512, bf16=bf16)
    pred.predict(snaps[:512], hist[:512])                  # warm-up
    pred.clear_cache()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for a in range(0, len(snaps), 512):
        pred.predict(snaps[a:a + 512], hist[a:a + 512])
    if device == "cuda":
        torch.cuda.synchronize()
    cold = len(snaps) / (time.perf_counter() - t0)

    edited = []
    for s in snaps:
        e = s.copy()
        j = next((k for k, p in enumerate(e.players) if p.alive), 0)
        e.players[j].health = max(1.0, (e.players[j].health or 1.0) - 1.0)
        edited.append(e)
    t0 = time.perf_counter()
    for a in range(0, len(snaps), 512):
        pred.predict(edited[a:a + 512], hist[a:a + 512])
    if device == "cuda":
        torch.cuda.synchronize()
    warm = len(snaps) / (time.perf_counter() - t0)
    res = {"device": device, "precision": "bf16" if pred.bf16 else "fp32", "n": len(snaps), "batch": 512,
           "cold_snapshots_per_s": round(cold, 1),
           "warm_history_snapshots_per_s": round(warm, 1), "target": TARGET[device],
           "meets_target_cold": cold >= TARGET[device], "meets_target_warm": warm >= TARGET[device]}
    print(f"{device:4s} {res['precision']}: cold {cold:9.1f}/s   warm-history {warm:9.1f}/s   (target >= {TARGET[device]:.0f}/s)")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Predictor throughput at batch 512 (P6.4.2).")
    ap.add_argument("--n", type=int, default=2048, help="snapshots per measurement")
    ap.add_argument("--no-cpu", action="store_true")
    a = ap.parse_args(argv)
    n_rounds = max(1, a.n // 40 + 1)
    snaps, vocab, _, norm = fake_arrays(n_rounds, seed=0, n_snapshots=40)
    snaps, hist = snaps[:a.n], histories(snaps)[:a.n]
    results, stages = [], {}
    with tempfile.TemporaryDirectory() as d:
        mdir = Path(d) / "wp_bench"
        model = make_model_dir(mdir, vocab, norm)
        print(f"model parameters {model.n_params:,}; {len(snaps)} fake snapshots, mean history "
              f"{sum(len(h) for h in hist) / len(hist):.1f}")
        devices = (["cuda"] if torch.cuda.is_available() else []) + ([] if a.no_cpu else ["cpu"])
        runs = [(dev, False) for dev in devices] + ([("cuda", True)] if "cuda" in devices else [])
        for dev, bf16 in runs:
            t0 = time.perf_counter()
            r = bench(mdir, dev, snaps, hist, bf16)
            results.append(r)
            stages[f"predict_{dev}_{r['precision']}"] = time.perf_counter() - t0
    out = write(paths.BENCH_DIR / f"predict_{date.today().isoformat()}.json", stages,
                extra={"results": results, "torch": torch.__version__})
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    finish(main())          # os._exit after GPU use: see _fake_states.finish
