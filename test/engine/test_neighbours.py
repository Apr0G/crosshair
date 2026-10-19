"""
P9.3 neighbours: a stored vector returns itself first; the query's own match is
excluded; side / map filters; the panel sums to k (widening the search when filters
bite); deterministic tie order; mean_V / mean_Q; HNSW vs flat choice; save / load.

    .venv\\Scripts\\python.exe test\\engine\\test_neighbours.py [--cpu]
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from _harness import check, require, section, summary
require("faiss", "the neighbour index needs faiss-cpu (requirements-engine.txt)")   # LOW-61
from _value_states import side_table, synthetic_corpus  # noqa: E402
from engine.value import neighbours as nb  # noqa: E402
from engine.value.descriptors import DescriptorVocab  # noqa: E402

K = 200
frame, h = synthetic_corpus(4000, seed=5, n_matches=40, maps=("de_test@a", "de_test@b"))
vocab = DescriptorVocab.fit_frame(frame)
table = side_table(frame, vocab).with_columns(Q=pl.col("V") * 0.5)
idx = nb.build_index(h, table)
A = idx.arrays()

section("index kind")
check("flat below 2 M", nb.index_kind(1_999_999) == "flat" and idx.kind == "flat")
check("hnsw at 2 M and above", nb.index_kind(2_000_000) == "hnsw" and nb.index_kind(5_000_000) == "hnsw")
check("threshold is a parameter", nb.index_kind(100, threshold=50) == "hnsw")
check("one partition per (side, map_version)", sorted(idx.parts) == [(s, m) for s in ("ct", "t") for m in ("de_test@a", "de_test@b")]
      and sum(pp.n for pp in idx.parts.values()) == 4000)
check("side table kept with extra columns", idx.n == 4000 and "Q" in idx.table.columns and "desc_ids" in idx.table.columns)
try:
    nb.build_index(h[:10], table)
    check("vector / side-table length mismatch refused", False)
except ValueError:
    check("vector / side-table length mismatch refused", True)

section("self query, exclusions, filters")
r = 123
row = table.row(r, named=True)
p_self = nb.panel(idx, h[r], row["side"], row["map_version"], K, exclude_match_key=None)
check("query equal to a stored vector returns it first", p_self.neighbour_ids[0] == row["action_id"])
p = nb.panel(idx, h[r], row["side"], row["map_version"], K, exclude_match_key=row["match_key"])
pos = {a: i for i, a in enumerate(A["action_id"])}
kept = np.array([pos[a] for a in p.neighbour_ids])
check("own match never in the panel", not (A["match"][kept] == row["match_key"]).any())
check("all neighbours on the query's side", (A["side"][kept] == row["side"]).all())
check("all neighbours on the query's map version", (A["map"][kept] == row["map_version"]).all())
check("panel sums to k", p.total() == K and p.n_used == K, f"total={p.total()}")
check("at most 20 entries, the rest in n_other", len(p.entries) <= 20 and p.n_other == K - sum(e.n for e in p.entries))
check("entries ordered n desc, key asc",
      [(-e.n, e.key) for e in p.entries] == sorted((-e.n, e.key) for e in p.entries))
e0 = p.entries[0]
members = kept[np.array([k == e0.key for k in table["descriptor_key"].to_numpy()[kept]])]
check("mean_V of an entry = mean V of its members", abs(e0.mean_V - A["V"][members].mean()) < 1e-12)
check("freq = n / n_used", abs(e0.freq - e0.n / K) < 1e-12)
check("mean_Q None without a Q predictor", all(e.mean_Q is None for e in p.entries))
pq = nb.panel(idx, h[r], row["side"], row["map_version"], K, exclude_match_key=row["match_key"],
              q_fn=nb.q_from_column(idx, "Q"))
check("mean_Q from q_fn = mean of members' Q", abs(pq.entries[0].mean_Q - 0.5 * A["V"][members].mean()) < 1e-12)
check("representative desc_ids = nearest member's", e0.desc_ids == A["desc_ids"][members[0]].tolist())
check("support(key) counts kept neighbours", p.support(e0.key) == e0.n and p.support("nope||") == 0)
check("ref is stable", p.ref == nb.panel(idx, h[r], row["side"], row["map_version"], K,
                                         exclude_match_key=row["match_key"]).ref)

section("batched panels == single panels")
qs = [3, 77, 1500, 3999]
batch = nb.panels(idx, h[qs], table["side"].to_numpy()[qs], table["map_version"].to_numpy()[qs], K,
                  exclude_match_keys=table["match_key"].to_numpy()[qs])
single = [nb.panel(idx, h[i], table["side"][i], table["map_version"][i], K, exclude_match_key=table["match_key"][i])
          for i in qs]
check("same neighbours and entries", all(b.neighbour_ids == s.neighbour_ids and b.entries == s.entries
                                         for b, s in zip(batch, single)))

section("search widens when the own-match filter removes the 4k candidates")
rng = np.random.default_rng(0)
base = rng.normal(size=16).astype(np.float32)
near = base + 0.01 * rng.normal(size=(1000, 16)).astype(np.float32)
far = -base + 0.5 * rng.normal(size=(300, 16)).astype(np.float32)
hv = np.vstack([near, far])
tv = pl.DataFrame({"match_key": ["own"] * 1000 + [f"w{i % 7}" for i in range(300)], "round_num": [1] * 1300,
                   "action_id": [f"a{i}" for i in range(1300)], "descriptor_key": ["kill||Mid"] * 1300,
                   "V": [0.0] * 1300, "side": ["ct"] * 1300,
                   "map_version": ["de_test@a"] * 1300, "tier": ["pro"] * 1300})
iv = nb.build_index(hv, tv)
pw = nb.panel(iv, base, "ct", "de_test@a", K, exclude_match_key="own")
check("k neighbours found beyond the first 4k candidates", pw.n_used == K and pw.total() == K)
check("none of them from the excluded match", all(int(a[1:]) >= 1000 for a in pw.neighbour_ids))
pf = nb.panel(iv, base, "ct", "de_test@b", K, exclude_match_key=None)
check("no candidate on that map -> empty panel", pf.n_used == 0 and pf.entries == [])
pt_ = nb.panel(iv, base, "t", "de_test@a", K, exclude_match_key=None)
check("no candidate on that side -> empty panel", pt_.n_used == 0)
pfew = nb.panel(iv, base, "ct", "de_test@a", 1000, exclude_match_key="own")
check("fewer than k available -> all of them", pfew.n_used == 300)
pall = nb.panel(iv, base, "ct", None, K, exclude_match_key=None)
check("map_version None searches every partition of the side", pall.n_used == K)

section("deterministic ties")
same = np.tile(base, (40, 1))
keys = ["smoke||B", "kill||A", "flash||C", "he||D"]


def tie_table(order):
    return pl.DataFrame({"match_key": ["t1"] * 40, "round_num": [1] * 40,
                         "action_id": [f"id{i}" for i in order], "descriptor_key": [keys[i % 4] for i in order],
                         "V": [float(i) for i in order], "side": ["ct"] * 40,
                         "map_version": ["m"] * 40, "tier": ["pro"] * 40})


order1 = list(range(40))
order2 = list(rng.permutation(40))
pt1 = nb.panel(nb.build_index(same, tie_table(order1)), base, "ct", "m", 40, exclude_match_key=None)
pt2 = nb.panel(nb.build_index(same, tie_table(order2)), base, "ct", "m", 40, exclude_match_key=None)
check("equal counts ordered by key", [e.key for e in pt1.entries] == sorted(keys))
check("same entries whatever the insertion order",
      [(e.key, e.n, e.mean_V) for e in pt1.entries] == [(e.key, e.n, e.mean_V) for e in pt2.entries])
check("equal similarity: stored row order", pt1.neighbour_ids == [f"id{i}" for i in range(40)])
check("repeat call identical", pt1 == nb.panel(nb.build_index(same, tie_table(order1)), base, "ct", "m", 40,
                                               exclude_match_key=None))

section("HNSW path and save / load")
ih = nb.build_index(h, table, hnsw_threshold=500)
check("hnsw chosen per partition above the threshold", all(pp.kind == "hnsw" for pp in ih.parts.values()))
ph = nb.panel(ih, h[r], row["side"], row["map_version"], K, exclude_match_key=None)
check("hnsw: self query first", ph.neighbour_ids[0] == row["action_id"])
check("hnsw: panel sums to k", ph.total() == K)
with tempfile.TemporaryDirectory() as tmp:
    for name, ix in (("flat", idx), ("hnsw", ih)):
        out = nb.save_index(ix, Path(tmp) / f"neighbours_{name}")
        back = nb.load_index(out)
        p1 = nb.panel(ix, h[r], row["side"], row["map_version"], K, exclude_match_key=row["match_key"])
        p2 = nb.panel(back, h[r], row["side"], row["map_version"], K, exclude_match_key=row["match_key"])
        check(f"{name}: save / load gives the same panel", p1 == p2 and back.kind == ix.kind and back.n == ix.n)
        check(f"{name}: manifest written", (out / "manifest.json").is_file())

sys.exit(summary())
