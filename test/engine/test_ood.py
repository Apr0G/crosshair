"""
Off-distribution detector (P7.2.7): k-th neighbour distance via faiss IndexFlatL2; a
planted far point is flagged; the flag rate on fresh factual data is about 1% at the
99th-percentile threshold; save/load round trip; input checks.

    .venv\\Scripts\\python.exe test\\engine\\test_ood.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from _harness import check, section, summary

from engine.counterfactual.ood import OODDetector

rng = np.random.default_rng(1337)
D = 16
train   = rng.normal(size=(5000, D))
factual = rng.normal(size=(5000, D))
fresh   = rng.normal(size=(20000, D))

# ── fit / score ───────────────────────────────────────────────────────────────
section("fit and score")

det = OODDetector(k=10, percentile=99).fit(train)
check("fit sets a provisional threshold", det.threshold is not None and det.threshold > 0)
s_train = det.score(train[:5])
check("score is the k-th neighbour distance (brute-force agrees)",
      np.allclose(s_train[0], np.sort(np.linalg.norm(train - train[0], axis=1))[9], atol=1e-4))
thr = det.calibrate(factual)
check("calibrate: threshold = 99th percentile of factual scores",
      abs(thr - np.percentile(det.score(factual), 99)) < 1e-9)
rate = det.flag_rate(det.score(fresh))
check("flag rate on fresh factual data is about 1%", 0.004 <= rate <= 0.02, f"{rate:.4f}")
far = np.full((1, D), 10.0)
check("a planted far point is flagged", bool(det.flags(det.score(far))[0]))
mixed = np.vstack([fresh[:990], np.full((10, D), 8.0)])
check("planted points raise the flag rate", det.flag_rate(det.score(mixed)) >= 0.01)
check("1-D input is treated as one row", det.score(train[0]).shape == (1,))
check("k is respected", OODDetector(k=1).fit(train).score(train[:3]).max() < 1e-3)

# ── persistence ───────────────────────────────────────────────────────────────
section("save / load")

with tempfile.TemporaryDirectory() as tmp:
    npz, meta = det.save(Path(tmp) / "ood_test")
    check("save writes .npz and .json", npz.is_file() and meta.is_file() and npz.suffix == ".npz")
    back = OODDetector.load(Path(tmp) / "ood_test")
    check("load keeps k, percentile, threshold", (back.k, back.percentile, back.threshold) == (det.k, det.percentile, det.threshold))
    check("load gives identical scores", np.array_equal(back.score(fresh[:100]), det.score(fresh[:100])))

# ── input checks ──────────────────────────────────────────────────────────────
section("input checks")


def raises(name: str, fn, exc=ValueError) -> None:
    try:
        fn()
    except exc:
        check(name, True)
        return
    check(name, False, "did not raise")


raises("fit with <= k points", lambda: OODDetector(k=10).fit(train[:10]))
raises("NaN embeddings", lambda: OODDetector().fit(np.full((50, 3), np.nan)))
raises("dimension mismatch on score", lambda: det.score(np.zeros((2, D + 1))))
raises("score before fit", lambda: OODDetector().score(train[:2]), RuntimeError)
raises("bad k", lambda: OODDetector(k=0))
raises("bad percentile", lambda: OODDetector(percentile=0))

sys.exit(summary())
