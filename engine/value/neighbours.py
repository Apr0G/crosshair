"""
FAISS index over pro train embeddings and the neighbour panel (P9.3).

    from engine.value import neighbours
    idx = neighbours.build_index(h_train, side_table)          # FlatIP below 2 M rows
    neighbours.save_index(idx, out_dir); idx = neighbours.load_index(out_dir)
    p   = neighbours.panel(idx, h_q, "ct", "de_mirage@1", k=200, exclude_match_key="m42")
    p.entries[0].key, p.entries[0].n, p.entries[0].mean_V
    ps  = neighbours.panels(idx, H, sides, map_versions, 200, exclude_match_keys=mks)   # batched

Vectors are L2-normalized and searched by inner product (= cosine). The stored rows are
partitioned by (side, map_version), one FAISS index per partition, so the same-side and
same-map filters are exact before the search instead of discarding most of the
candidates afterwards (with 9 maps × 2 sides a global index would keep ~1/18 of them).
Each partition's kind is `index_kind(n_partition)`: `IndexHNSWFlat` (inner product)
when it holds >= 2,000,000 vectors, else exact `IndexFlatIP`. A query with
`map_version=None` searches every partition of its side.

Side table (one row per stored vector, same order): `match_key, round_num, action_id,
descriptor_key, V, side, map_version, tier`, optionally `desc_ids` (List[Int64], the
action's descriptor ids, needed to score alternatives for the regret). Extra columns
(e.g. a precomputed `Q`) are kept.

Panel (P9.3.2): retrieve `4·k` nearest candidates from the query's partition(s), drop
the query's own match, take the first `k` (ordered by similarity desc, then stored row
asc). If fewer than `k` survive and the partition has more vectors, the search is
widened (×2) until `k` survive or every vector was seen. Kept neighbours are grouped by
`descriptor_key`; groups ordered by (n desc, key asc); the top 20 become `entries` and
the rest are counted in `n_other`, so `sum(e.n) + n_other == n_used` (== k when enough
candidates exist). Each entry carries the nearest member's `desc_ids` as its
representative descriptor. `mean_Q` is filled only when a `q_fn(row_idx) -> Q̂ per
stored row` is passed.

Filled by: P9.3.1-P9.3.3.
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import faiss
import numpy as np
import polars as pl

from engine import manifest, paths

# ── constants ─────────────────────────────────────────────────────────────────

SIDE_TABLE_SCHEMA = {
    "match_key":      pl.String,
    "round_num":      pl.Int64,
    "action_id":      pl.String,
    "descriptor_key": pl.String,
    "V":              pl.Float64,
    "side":           pl.String,
    "map_version":    pl.String,
    "tier":           pl.String,
}
HNSW_MIN_N      = 2_000_000
HNSW_M          = 32
EF_CONSTRUCTION = 200
CANDIDATE_MULT  = 4
TOP_N           = 20
TABLE_FILE, META_FILE = "side_table.parquet", "meta.json"


def index_kind(n: int, threshold: int = HNSW_MIN_N) -> str:
    """"hnsw" when the searched set has at least `threshold` vectors, else "flat"."""
    return "hnsw" if n >= threshold else "flat"


def normalize(h) -> np.ndarray:
    """Row-wise L2-normalized float32 copy (an all-zero row stays zero)."""
    x = np.array(h, dtype=np.float32, copy=True, order="C")
    if x.ndim == 1:
        x = x[None, :]
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    np.divide(x, nrm, out=x, where=nrm > 0)
    return x


# ── index ─────────────────────────────────────────────────────────────────────

@dataclass
class Partition:
    side:        str
    map_version: str | None
    index:       faiss.Index
    rows:        np.ndarray            # global side-table row of each FAISS id
    kind:        str

    @property
    def n(self) -> int:
        return int(self.rows.size)


@dataclass
class NeighbourIndex:
    parts: dict[tuple, Partition]
    table: pl.DataFrame
    d:     int
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return self.table.height

    @property
    def kind(self) -> str:
        """"hnsw" if any partition is HNSW, else "flat"."""
        return "hnsw" if any(p.kind == "hnsw" for p in self.parts.values()) else "flat"

    def arrays(self) -> dict:
        """Per-row numpy views used by the panel (built once)."""
        if not self._cache:
            t = self.table
            keys = t["descriptor_key"].fill_null("").to_numpy()
            uniq, codes = np.unique(keys, return_inverse=True)        # sorted keys -> code order = key order
            self._cache.update({
                "match": t["match_key"].to_numpy(), "side": t["side"].to_numpy(),
                "map": t["map_version"].to_numpy(), "V": t["V"].cast(pl.Float64).to_numpy(),
                "key_codes": codes.astype(np.int64), "keys": uniq, "action_id": t["action_id"].to_numpy(),
                "desc_ids": (np.asarray(t["desc_ids"].to_list(), dtype=np.int64)
                             if "desc_ids" in t.columns and t.height else None),
            })
        return self._cache

    def partitions_for(self, side: str, map_version) -> list[Partition]:
        if map_version is not None:
            p = self.parts.get((side, map_version))
            return [p] if p is not None else []
        return [p for (s, _), p in sorted(self.parts.items(), key=lambda kv: str(kv[0])) if s == side]


def _check_table(h: np.ndarray, table: pl.DataFrame) -> None:
    missing = [c for c in SIDE_TABLE_SCHEMA if c not in table.columns]
    if missing:
        raise ValueError(f"side table missing columns {missing}")
    if h.shape[0] != table.height:
        raise ValueError(f"{h.shape[0]} vectors but {table.height} side-table rows")
    bad = table.filter(~pl.col("side").is_in(["ct", "t"])).height
    if bad:
        raise ValueError(f"side table: {bad} row(s) with side not in ('ct', 't')")


def _new_index(d: int, n: int, threshold: int, hnsw_m: int, ef_construction: int) -> tuple[faiss.Index, str]:
    kind = index_kind(n, threshold)
    if kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
    else:
        index = faiss.IndexFlatIP(d)
    return index, kind


def build_index(h, side_table: pl.DataFrame, *, hnsw_threshold: int = HNSW_MIN_N,
                hnsw_m: int = HNSW_M, ef_construction: int = EF_CONSTRUCTION) -> NeighbourIndex:
    """Partitioned FAISS index over L2-normalized `h` (float[n, d]) with its side table."""
    x = normalize(h)
    _check_table(x, side_table)
    casts = [pl.col(c).cast(t) for c, t in SIDE_TABLE_SCHEMA.items()]
    if "desc_ids" in side_table.columns:
        casts.append(pl.col("desc_ids").cast(pl.List(pl.Int64)))
    table = side_table.with_columns(casts)            # extra columns (e.g. a precomputed Q) are kept
    d = int(x.shape[1])
    sides, maps = table["side"].to_numpy(), table["map_version"].to_numpy()
    parts: dict[tuple, Partition] = {}
    for key in sorted({(s, m) for s, m in zip(sides, maps)}, key=str):
        rows = np.flatnonzero((sides == key[0]) & (maps == key[1]) if key[1] is not None
                              else (sides == key[0]) & np.array([m is None for m in maps]))
        index, kind = _new_index(d, rows.size, hnsw_threshold, hnsw_m, ef_construction)
        index.add(x[rows])
        parts[key] = Partition(key[0], key[1], index, rows.astype(np.int64), kind)
    return NeighbourIndex(parts=parts, table=table, d=d)


def save_index(nidx: NeighbourIndex, out_dir: Path | None = None, *, cfg=None) -> Path:
    """Write partition indices + side table + meta (+ manifest) to `out_dir` (default
    the next `paths.MODELS_DIR / neighbours_v<N>`)."""
    from engine.value.q_model import next_version_dir
    out = Path(out_dir) if out_dir is not None else next_version_dir(paths.MODELS_DIR, "neighbours_v")
    out.mkdir(parents=True, exist_ok=True)
    meta_parts, outputs = [], [TABLE_FILE, META_FILE]
    for i, (key, p) in enumerate(nidx.parts.items()):
        name = f"part_{i:03d}.faiss"
        faiss.write_index(p.index, str(out / name))
        np.save(out / f"part_{i:03d}_rows.npy", p.rows)
        meta_parts.append({"side": p.side, "map_version": p.map_version, "kind": p.kind, "n": p.n, "file": name,
                           "rows": f"part_{i:03d}_rows.npy"})
        outputs += [name, f"part_{i:03d}_rows.npy"]
    nidx.table.write_parquet(out / TABLE_FILE)
    (out / META_FILE).write_text(json.dumps({"d": nidx.d, "n": nidx.n, "parts": meta_parts}, indent=2),
                                 encoding="utf-8")
    manifest.write(out, "neighbours", outputs=outputs, extra={"kind": nidx.kind, "n": nidx.n, "d": nidx.d,
                                                              "n_parts": len(meta_parts)},
                   **({"cfg": cfg} if cfg is not None else {}))
    return out


def load_index(index_dir: Path) -> NeighbourIndex:
    d = Path(index_dir)
    meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
    table = pl.read_parquet(d / TABLE_FILE)
    parts = {}
    for mp in meta["parts"]:
        index = faiss.read_index(str(d / mp["file"]))
        rows = np.load(d / mp["rows"])
        if index.ntotal != rows.size:
            raise ValueError(f"{d}: partition {mp['file']} has {index.ntotal} vectors, {rows.size} rows")
        parts[(mp["side"], mp["map_version"])] = Partition(mp["side"], mp["map_version"], index, rows, mp["kind"])
    if sum(p.n for p in parts.values()) != table.height:
        raise ValueError(f"{d}: partitions hold {sum(p.n for p in parts.values())} vectors, side table {table.height}")
    return NeighbourIndex(parts=parts, table=table, d=int(meta["d"]))


# ── panel ─────────────────────────────────────────────────────────────────────

@dataclass
class PanelEntry:
    key:      str
    n:        int
    freq:     float
    mean_V:   float
    mean_Q:   float | None = None
    desc_ids: list[int] | None = None


@dataclass
class Panel:
    entries:       list[PanelEntry]
    k:             int
    n_used:        int
    n_other:       int
    side:          str
    map_version:   str | None
    neighbour_ids: list[str] = field(default_factory=list, repr=False)
    counts:        dict[str, int] = field(default_factory=dict, repr=False)

    def total(self) -> int:
        return sum(e.n for e in self.entries) + self.n_other

    def support(self, key: str) -> int:
        """How many kept neighbours did `key` (0 when absent)."""
        return self.counts.get(key, 0)

    @property
    def ref(self) -> str:
        """Stable reference: sha1 of side, map version and the kept neighbour action ids."""
        s = f"{self.side}|{self.map_version}|" + ",".join(self.neighbour_ids)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"ref": self.ref, "k": self.k, "n_used": self.n_used, "n_other": self.n_other,
                "side": self.side, "map_version": self.map_version,
                "entries": [{"key": e.key, "n": e.n, "freq": e.freq, "mean_V": e.mean_V, "mean_Q": e.mean_Q}
                            for e in self.entries]}


def _search(p: Partition, x: np.ndarray, k_search: int) -> tuple[np.ndarray, np.ndarray]:
    """Global row ids and scores of the `k_search` nearest in partition `p`."""
    if p.kind == "hnsw":
        p.index.hnsw.efSearch = max(64, k_search)
    scores, ids = p.index.search(x, k_search)
    rows = np.where(ids >= 0, p.rows[np.clip(ids, 0, None)], -1)
    return rows, scores


def _keep(nidx: NeighbourIndex, rows: np.ndarray, scores: np.ndarray, exclude_match_key, k: int) -> np.ndarray:
    ok = rows >= 0
    rows, scores = rows[ok], scores[ok]
    rows = rows[np.lexsort((rows, -scores))]           # similarity desc, then row asc
    if exclude_match_key is not None:
        rows = rows[nidx.arrays()["match"][rows] != exclude_match_key]
    return rows[:k]


def _query_one(nidx: NeighbourIndex, parts: list[Partition], xi: np.ndarray, exclude, k: int,
               first: list[tuple] | None) -> np.ndarray:
    """Kept rows for one query over its partitions (`first`: precomputed batch results)."""
    n_avail = sum(p.n for p in parts)
    k_search = min(n_avail, CANDIDATE_MULT * k)
    if first is None:
        first = [_search(p, xi, min(p.n, k_search)) for p in parts]
    def merged(res):                                   # [(rows[1, m], scores[1, m])] -> 1-D arrays
        return np.concatenate([r[0] for r, _ in res]), np.concatenate([sc[0] for _, sc in res])

    kept = _keep(nidx, *merged(first), exclude, k)
    ks = k_search
    while kept.size < k and ks < n_avail:              # widen until k survive or all seen
        ks = min(n_avail, ks * 2)
        kept = _keep(nidx, *merged([_search(p, xi, min(p.n, ks)) for p in parts]), exclude, k)
    return kept


def _make_panel(nidx: NeighbourIndex, kept: np.ndarray, k: int, side: str, map_version,
                q_vals: np.ndarray | None, top: int) -> Panel:
    a = nidx.arrays()
    n_used = int(kept.size)
    if n_used == 0:
        return Panel([], k, 0, 0, side, map_version, [], {})
    codes = a["key_codes"][kept]
    uniq, first, cnt = np.unique(codes, return_index=True, return_counts=True)
    inv = np.searchsorted(uniq, codes)
    v_sum = np.bincount(inv, weights=a["V"][kept], minlength=uniq.size)
    q_sum = np.bincount(inv, weights=q_vals, minlength=uniq.size) if q_vals is not None else None
    order = np.lexsort((uniq, -cnt))                   # n desc, key asc (codes follow key order)
    entries = []
    for j in order[:top]:
        rep = kept[first[j]]                           # nearest member (kept is similarity-ordered)
        entries.append(PanelEntry(
            key=str(a["keys"][uniq[j]]), n=int(cnt[j]), freq=float(cnt[j]) / n_used,
            mean_V=float(v_sum[j] / cnt[j]), mean_Q=None if q_sum is None else float(q_sum[j] / cnt[j]),
            desc_ids=None if a["desc_ids"] is None else a["desc_ids"][rep].tolist()))
    n_other = n_used - sum(e.n for e in entries)
    counts = {str(a["keys"][u]): int(c) for u, c in zip(uniq, cnt)}
    return Panel(entries, k, n_used, n_other, side, map_version, a["action_id"][kept].tolist(), counts)


def panels(nidx: NeighbourIndex, h_query, sides, map_versions, k: int, *, exclude_match_keys,
           q_fn: Callable[[np.ndarray], np.ndarray] | None = None, top: int = TOP_N) -> list[Panel]:
    """One Panel per query row: queries are grouped by partition and searched in one
    FAISS call per partition, then filtered per query (own match) and grouped."""
    x = normalize(h_query)
    nq = x.shape[0]
    sides, maps, excl = list(sides), list(map_versions), list(exclude_match_keys)
    if not (len(sides) == len(maps) == len(excl) == nq):
        raise ValueError("h_query, sides, map_versions and exclude_match_keys must have one entry per query")
    if nq and x.shape[1] != nidx.d:
        raise ValueError(f"query dim {x.shape[1]} != index dim {nidx.d}")

    groups: dict[tuple, list[int]] = {}
    for i, (s, m) in enumerate(zip(sides, maps)):
        groups.setdefault((s, m), []).append(i)
    kept_all: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * nq
    for (s, m), qi in groups.items():
        parts = nidx.partitions_for(s, m)
        if not parts:
            continue
        k_search = min(sum(p.n for p in parts), CANDIDATE_MULT * k)
        batch = [_search(p, x[qi], min(p.n, k_search)) for p in parts]
        for r, i in enumerate(qi):
            first = [(b[0][r:r + 1], b[1][r:r + 1]) for b in batch]
            kept_all[i] = _query_one(nidx, parts, x[i:i + 1], excl[i], k, first)

    q_split: list = [None] * nq
    if q_fn is not None:
        flat = np.concatenate(kept_all) if nq else np.zeros(0, dtype=np.int64)
        qv = np.asarray(q_fn(flat), dtype=np.float64)
        if qv.shape != flat.shape:
            raise ValueError(f"q_fn returned shape {qv.shape}, expected {flat.shape}")
        q_split = np.split(qv, np.cumsum([kk.size for kk in kept_all])[:-1])
    return [_make_panel(nidx, kept_all[i], k, sides[i], maps[i], q_split[i], top) for i in range(nq)]


def panel(nidx: NeighbourIndex, h_query, side: str, map_version, k: int, *, exclude_match_key,
          q_fn: Callable[[np.ndarray], np.ndarray] | None = None, top: int = TOP_N) -> Panel:
    """The P9.3.2 panel of one query vector (see module docstring)."""
    return panels(nidx, np.atleast_2d(h_query), [side], [map_version], k,
                  exclude_match_keys=[exclude_match_key], q_fn=q_fn, top=top)[0]


def q_from_column(nidx: NeighbourIndex, column: str = "Q") -> Callable[[np.ndarray], np.ndarray]:
    """A `q_fn` reading precomputed Q̂ of the stored actions from a side-table column."""
    vals = nidx.table[column].cast(pl.Float64).to_numpy()
    return lambda rows: vals[rows]
