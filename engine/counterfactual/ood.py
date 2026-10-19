"""
Off-distribution detector (P7.2.7): distance from an edited state's embedding to its
10th nearest training embedding vs the 99th percentile of factual test states.

    from engine.counterfactual.ood import OODDetector
    det = OODDetector(k=10, percentile=99).fit(train_h)      # faiss IndexFlatL2
    det.calibrate(factual_test_h)                            # threshold from held-out factual states
    det.flag_rate(det.score(edited_h))                       # share above the threshold
    det.save("eval/ood_wp_v1"); OODDetector.load("eval/ood_wp_v1")   # .npz + .json

`score` is the L2 distance (not squared) to the k-th nearest training embedding. `fit`
sets a provisional threshold from the training set itself (each point's k-th neighbour
excluding itself); `calibrate` replaces it with the percentile over factual held-out
states, which is the threshold P7.2.7 specifies.

Filled by: P7.2.7.
"""
import json
from pathlib import Path

import faiss
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_K          = 10
DEFAULT_PERCENTILE = 99.0
SEARCH_BATCH       = 65536


def _as_matrix(x) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, d), got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("embeddings contain NaN/inf")
    return a


class OODDetector:
    """k-NN distance detector over a fixed set of training embeddings."""

    def __init__(self, k: int = DEFAULT_K, percentile: float = DEFAULT_PERCENTILE):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not 0 < percentile <= 100:
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")
        self.k          = int(k)
        self.percentile = float(percentile)
        self.threshold: float | None = None
        self._train: np.ndarray | None = None
        self._index = None

    # ── fit / score ──

    def fit(self, embeddings, k: int | None = None) -> "OODDetector":
        """Index the training embeddings (faiss IndexFlatL2) and set a provisional
        threshold from their own leave-self-out scores."""
        if k is not None:
            self.k = int(k)
        x = _as_matrix(embeddings)
        if len(x) <= self.k:
            raise ValueError(f"need more than k={self.k} training embeddings, got {len(x)}")
        self._train = x
        self._index = faiss.IndexFlatL2(x.shape[1])
        self._index.add(x)
        self.threshold = float(np.percentile(self._kth(x, self.k + 1), self.percentile))
        return self

    def _kth(self, x: np.ndarray, kk: int) -> np.ndarray:
        out = []
        for i in range(0, len(x), SEARCH_BATCH):
            d, _ = self._index.search(x[i:i + SEARCH_BATCH], kk)
            out.append(np.sqrt(np.maximum(d[:, kk - 1], 0.0)))
        return np.concatenate(out).astype(np.float64) if out else np.empty(0)

    def score(self, x) -> np.ndarray:
        """Distance from each row to its k-th nearest training embedding."""
        if self._index is None:
            raise RuntimeError("OODDetector.score before fit/load")
        x = _as_matrix(x)
        if x.shape[1] != self._train.shape[1]:
            raise ValueError(f"embedding dim {x.shape[1]} != trained dim {self._train.shape[1]}")
        return self._kth(x, self.k)

    def calibrate(self, factual) -> float:
        """Threshold = `percentile` of the scores of factual held-out embeddings."""
        self.threshold = float(np.percentile(self.score(factual), self.percentile))
        return self.threshold

    def flags(self, scores) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("OODDetector has no threshold")
        return np.asarray(scores, dtype=np.float64) > self.threshold

    def flag_rate(self, scores) -> float:
        """Share of scores above the threshold (NaN for no scores)."""
        f = self.flags(scores)
        return float(f.mean()) if f.size else float("nan")

    # ── persistence ──

    def save(self, path) -> tuple[Path, Path]:
        """Write `<path>.npz` (training embeddings) and `<path>.json` (k, percentile,
        threshold, shape); the faiss index is rebuilt on load."""
        if self._train is None:
            raise RuntimeError("OODDetector.save before fit")
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        npz, meta = base.with_suffix(".npz"), base.with_suffix(".json")
        np.savez_compressed(npz, embeddings=self._train)
        meta.write_text(json.dumps({"k": self.k, "percentile": self.percentile,
                                    "threshold": self.threshold, "n": int(len(self._train)),
                                    "dim": int(self._train.shape[1])}, indent=2), encoding="utf-8")
        return npz, meta

    @classmethod
    def load(cls, path) -> "OODDetector":
        base = Path(path)
        meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(base.with_suffix(".npz")) as z:
            x = z["embeddings"]
        det = cls(k=meta["k"], percentile=meta["percentile"]).fit(x)
        det.threshold = meta["threshold"]
        return det
