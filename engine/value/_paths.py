"""
Torch-free helpers shared by the value stage and the LightGBM baseline (LOW-32,
2026-10-19): `next_version_dir` and `h_matrix` used to live in q_model.py, so
`baseline_lgbm.main` and `neighbours.save_index` imported torch only to number an
output directory. q_model re-exports both names.

    from engine.value._paths import h_matrix, next_version_dir
    out = next_version_dir(paths.MODELS_DIR, "neighbours_v")     # models/neighbours_v<N+1>
    h   = h_matrix(frame)                                          # float32[n, d_h] from h_pre

Filled by: P9.2 (moved out of q_model.py).
"""
from pathlib import Path

import numpy as np
import polars as pl


def h_matrix(frame: pl.DataFrame, column: str = "h_pre") -> np.ndarray:
    """float32[n, d_h] from the frame's `h_pre` list column; raises on nulls or ragged rows."""
    s = frame[column]
    if s.null_count():
        raise ValueError(f"{column}: {s.null_count()} null embedding(s); fill h_pre before P9")
    rows = s.to_list()
    d = {len(r) for r in rows}
    if len(d) > 1:
        raise ValueError(f"{column}: ragged embeddings, lengths {sorted(d)}")
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), d.pop() if d else 0)


def next_version_dir(parent: Path, prefix: str) -> Path:
    parent = Path(parent)
    have = [int(p.name[len(prefix):]) for p in parent.glob(f"{prefix}*") if p.name[len(prefix):].isdigit()]
    return parent / f"{prefix}{max(have, default=0) + 1}"
