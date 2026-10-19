"""
Value-stage timing (P9.5.2) on a synthetic frame: Q / B inference and neighbour panels
for `--n` actions (default 5,000, about one map) against a FlatIP index of `--index-n`
stored pro actions, plus the full `scores.score_frame` (alternatives + compute). The
QNet / BNet are untrained (inference cost does not depend on the weights) and `h_pre`
is the toy embedding lifted to `--d-h` dims by a fixed random map. Prints seconds (best
of `--repeat`). Target from the plan: <= 5 s GPU, <= 20 s CPU per map (informational
here: the real index and h come from the corpus).

    .venv\\Scripts\\python.exe bench\\bench_value.py [--cpu] [--n 5000] [--index-n 100000] [--d-h 256]
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "engine"))

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from _bench import write                                         # noqa: E402
from _value_states import side_table, synthetic_corpus           # noqa: E402
from engine import config                                        # noqa: E402
from engine.value import neighbours as nb                        # noqa: E402
from engine.value import q_model as qm                           # noqa: E402
from engine.value import scores as sc                            # noqa: E402
from engine.value.descriptors import DescriptorVocab, ids_from_frame   # noqa: E402

MAPS = ("de_test@a", "de_test@b")


def best_of(fn, repeat: int) -> tuple[float, object]:
    best, out = None, None
    for _ in range(repeat):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--index-n", type=int, default=100_000)
    ap.add_argument("--d-h", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    cfg = config.load()

    t0 = time.perf_counter()
    store, hs = synthetic_corpus(args.index_n, seed=101, n_matches=400, maps=MAPS)
    query, hq = synthetic_corpus(args.n, seed=102, n_matches=400, maps=MAPS)
    lift = np.random.default_rng(7).normal(size=(hs.shape[1], args.d_h)).astype(np.float32)
    hs, hq = hs @ lift, hq @ lift
    vocab = DescriptorVocab.fit_frame(store)
    t_data = time.perf_counter() - t0

    t0 = time.perf_counter()
    index = nb.build_index(hs, side_table(store, vocab))
    t_index = time.perf_counter() - t0

    torch.manual_seed(cfg.train.seed)
    q = qm.QNet(args.d_h, vocab.sizes()).to(dev).eval()
    b = qm.BNet(args.d_h).to(dev).eval()
    ids = ids_from_frame(query, vocab)
    qm.predict_q(q, hq[:8], ids[:8], query["tier"][:8], query["actor_side"][:8])     # warm-up (kernels)

    t_qb, _ = best_of(lambda: (qm.predict_q(q, hq, ids, query["tier"], query["actor_side"]),
                               qm.predict_b(b, hq, query["tier"], query["actor_side"])), args.repeat)
    t_pan, pans = best_of(lambda: nb.panels(index, hq, query["actor_side"].to_list(), query["map_version"].to_list(),
                                            cfg.value.k_neighbours, exclude_match_keys=query["match_key"].to_list()),
                          args.repeat)
    t_all, scored = best_of(lambda: sc.score_frame(query, hq, q, b, vocab, index, cfg), args.repeat)
    t_frame, _ = best_of(lambda: sc.to_frame(scored, "bench"), args.repeat)

    stages = {"q_b_inference": t_qb, "panels": t_pan, "score_frame_total": t_all, "to_frame": t_frame}
    print(f"device {dev}  actions {args.n}  index {index.n} ({index.kind})  d_h {args.d_h}  k {cfg.value.k_neighbours}")
    print(f"  synthetic data      {t_data:8.3f} s   (not timed against the budget)")
    print(f"  build index         {t_index:8.3f} s   (offline, once per corpus)")
    print(f"  Q/B inference       {t_qb:8.3f} s")
    print(f"  panels              {t_pan:8.3f} s")
    print(f"  score_frame total   {t_all:8.3f} s   (Q/B + panels + alternatives + compute)")
    print(f"  to_frame            {t_frame:8.3f} s")
    budget = cfg.bench.budgets.value.gpu if dev == "cuda" else cfg.bench.budgets.value.cpu
    print(f"  per-map value total {t_all + t_frame:8.3f} s   (budget {budget} s on {dev})")
    if args.out:
        write(args.out, stages, extra={"device": dev, "n": args.n, "index_n": index.n, "d_h": args.d_h,
                                       "panels_with_entries": sum(bool(p.entries) for p in pans)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
