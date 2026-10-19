"""
P9.4.1 / condition C2: every valued action gets V, Q, B, D, R, X, verdicts and a panel;
`V = B + D + X` to 1e-9 on 10,000 synthetic actions; verdict thresholds inclusive as
documented (a score exactly on ±δ is neutral); `R <= 0` always; an alternative with
n < n_min never sets the max; a panel for >= 95% of a synthetic pro set; `to_frame`
schema, parquet round trip, and DB rows matching `db.SCORE_COLUMNS`.

    .venv\\Scripts\\python.exe test\\engine\\test_decision_split.py [--cpu]
"""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from _harness import check, require, section, summary
require("torch", "the Q model needs the platform torch build (docs/ENGINE_ENV.md)")      # LOW-61
require("faiss", "the neighbour index needs faiss-cpu (requirements-engine.txt)")
from _value_states import device, side_table, split_by_match, synthetic_corpus  # noqa: E402
from engine import config, db  # noqa: E402
from engine.attribution.types import ScoredAction  # noqa: E402
from engine.value import neighbours as nb  # noqa: E402
from engine.value import q_model as qm  # noqa: E402
from engine.value import scores as sc  # noqa: E402
from engine.value.descriptors import STATE_FIELDS, FIELDS, DescriptorVocab, ids_from_frame  # noqa: E402

DEV = device()
cfg = config.load()
VC = cfg.value
MAPS = ("de_test@a", "de_test@b")
COARSE = ("A Site", "B Site")
print(f"device: {DEV}  k={VC.k_neighbours} n_min={VC.n_min_regret} δ_D={VC.delta_D} δ_X={VC.delta_X}")

# ── pieces ────────────────────────────────────────────────────────────────────

section("verdict thresholds (neutral band inclusive)")
dD, dX = VC.delta_D, VC.delta_X
check("D = +δ_D -> neutral", sc.verdict(dD, dD) == "neutral")
check("D = −δ_D -> neutral", sc.verdict(-dD, dD) == "neutral")
check("D just above +δ_D -> good", sc.verdict(dD + 1e-12, dD) == "good")
check("D just below −δ_D -> bad", sc.verdict(-dD - 1e-12, dD) == "bad")
check("unknown -> None", sc.verdict(None, dD) is None and sc.verdict(float("nan"), dD) is None)
sa = ScoredAction(action_id="a" * 16, match_key="m1", round_num=1, actor_id_hash=1, actor_side="ct", type="kill",
                  V=dX)                                 # X = V - Q = Î´_X whatever the config says (LOW-60)
s_edge = sc.compute(sa, Q=0.0, B=0.0, panel=None, cfg=cfg)
check("compute: X = V − Q = δ_X exactly -> neutral execution", s_edge.X == dX and s_edge.verdict_execution == "neutral")
s_d = sc.compute(ScoredAction(**{**sa.__dict__, "V": 0.0}), Q=dD, B=0.0, panel=None, cfg=cfg)
check("compute: D = δ_D exactly -> neutral decision", s_d.D == dD and s_d.verdict_decision == "neutral")
s_g = sc.compute(ScoredAction(**{**sa.__dict__, "V": 0.5}), Q=0.3, B=0.1, panel=None, cfg=cfg)
check("compute: D = 0.2 good, X = 0.2 good", (s_g.verdict_decision, s_g.verdict_execution) == ("good", "good"))
s_b = sc.compute(ScoredAction(**{**sa.__dict__, "V": -0.5}), Q=-0.3, B=0.1, panel=None, cfg=cfg)
check("compute: D = −0.4 bad, X = −0.2 bad", (s_b.verdict_decision, s_b.verdict_execution) == ("bad", "bad"))
check("no panel: R = 0, flag no_panel, panel_ref None",
      s_g.R == 0.0 and "no_panel" in s_g.flags and s_g.panel_ref is None)

section("regret: n_min, own Q in the max")


def mk_panel(ns):
    es = [nb.PanelEntry(key=f"k{i}", n=n, freq=n / sum(ns), mean_V=0.0) for i, n in enumerate(ns)]
    return nb.Panel(es, k=sum(ns), n_used=sum(ns), n_other=0, side="ct", map_version="m",
                    counts={e.key: e.n for e in es})


p = mk_panel([29, 30, 100])
check("alternative with n = 29 and the highest Q is ignored", sc.regret(0.1, p, [0.9, 0.2, 0.15], 30) == 0.1 - 0.2)
check("n = 30 counts (>= n_min)", sc.regret(0.1, mk_panel([30]), [0.4], 30) == 0.1 - 0.4)
check("only unsupported alternatives -> R = 0", sc.regret(0.1, mk_panel([5, 29]), [0.9, 0.8], 30) == 0.0)
check("own Q above every alternative -> R = 0", sc.regret(0.5, p, [0.9, 0.2, 0.15], 30) == 0.0)
check("NaN alternative ignored", sc.regret(0.1, mk_panel([50]), [float("nan")], 30) == 0.0)
rng = np.random.default_rng(7)
ok_r, ok_nmin = True, True
for _ in range(10_000):
    m = int(rng.integers(1, 21))
    ns = rng.integers(1, 80, size=m)
    qa = rng.normal(0, 0.1, size=m)
    qs = float(rng.normal(0, 0.1))
    R = sc.regret(qs, mk_panel([int(x) for x in ns]), qa, 30)
    want = qs - max([qs] + [q for q, n in zip(qa, ns) if n >= 30])
    ok_r &= R <= 0.0
    ok_nmin &= R == want
check("R <= 0 on 10,000 random panels", ok_r)
check("R uses only n >= n_min alternatives on 10,000 random panels", ok_nmin)
try:
    sc.compute(sa, 0.0, 0.0, mk_panel([40]), cfg)
    check("supported panel without q_alternatives refused", False)
except ValueError:
    check("supported panel without q_alternatives refused", True)
alt = sc.alternative_ids(np.arange(len(FIELDS)), np.full(len(FIELDS), 99))
check("alternative_ids: state fields from the current action, choice fields from a'",
      all(alt[FIELDS.index(f)] == 99 for f in STATE_FIELDS)
      and all(alt[i] == i for i, f in enumerate(FIELDS) if f not in STATE_FIELDS))

# ── 10,000 actions end to end ─────────────────────────────────────────────────

section("C2 on 10,000 synthetic pro actions")
corpus, hc = synthetic_corpus(30_000, seed=21, n_matches=60, maps=MAPS, callouts=COARSE)
vocab = DescriptorVocab.fit_frame(corpus)
q, b, _ = qm.train_q(corpus, hc, vocab, cfg, split_by_match, DEV,
                     params=qm.QTrainParams(max_epochs=6, patience=2, batch_size=1024), save=False)
train_rows = np.array(split_by_match(corpus)) == "train"
index = nb.build_index(hc[train_rows], side_table(corpus.filter(pl.Series(train_rows)), vocab))
query, hq = synthetic_corpus(10_000, seed=22, n_matches=60, maps=MAPS, callouts=COARSE)
scored = sc.score_frame(query, hq, q, b, vocab, index, cfg)
check("one scored record per action", len(scored) == query.height)
fields_ok = all(all(getattr(s, f) is not None for f in ("V", "Q", "B", "D", "R", "X", "verdict_decision",
                                                         "verdict_execution", "descriptor_key")) for s in scored)
check("every action has V, Q, B, D, R, X, both verdicts", fields_ok)
err = max(abs(s.B + s.D + s.X - s.V) for s in scored)
check("identity V = B + D + X to 1e-9 (plumbing guard: definitional, cannot fail on its own)", err <= 1e-9,
      f"max |err| = {err:.1e}")
rho_qv = qm.spearman([s.Q for s in scored], [s.V for s in scored])
check("Q tracks realized V on held-out actions (Spearman > 0.2; the check that CAN fail, MED-33)",
      rho_qv > 0.2, f"rho = {rho_qv:.3f}")
check("D = Q − B and X = V − Q", all(s.D == s.Q - s.B and s.X == s.V - s.Q for s in scored))
check("R <= 0 on every action", all(s.R <= 0.0 for s in scored), f"min R = {min(s.R for s in scored):.4f}")
from engine.value import shrinkage as sh_mod                                      # noqa: E402
fit_sh = sh_mod.fit_from_models(corpus.filter(pl.Series(train_rows)), hc[train_rows], q, b, vocab)
check("shrinkage fitted from the trained Q / B nets on the train rows (MED-34)",
      all(len(fit_sh.get(v, {})) >= 1 for v in sh_mod.VALUES) and all(f["k"] > 0 for v in sh_mod.VALUES for f in fit_sh[v].values()),
      str({v: list(fit_sh.get(v, {})) for v in sh_mod.VALUES}))
with_panel = sum(s.panel is not None and len(s.panel.entries) >= 1 for s in scored) / len(scored)
check("a panel with >= 1 entry for >= 95% of actions", with_panel >= 0.95, f"{with_panel:.1%}")
check("panels are full (k neighbours) when present",
      all(s.panel.n_used == VC.k_neighbours for s in scored if s.panel is not None))
check("panel_ref set exactly when a panel exists", all((s.panel_ref is None) == (s.panel is None) for s in scored))
pos = {a: i for i, a in enumerate(index.arrays()["action_id"])}
own_ok = all(not (index.arrays()["match"][[pos[a] for a in s.panel.neighbour_ids]] == s.match_key).any()
             for s in scored[:300] if s.panel is not None)
check("own match excluded (300 sampled panels)", own_ok)
n_neg = sum(s.R < 0 for s in scored)
check("regret path exercised (some R < 0)", n_neg > 100, f"{n_neg} actions with R < 0")

# independent recomputation of R for a sample
ids_q = ids_from_frame(query, vocab)
recomp_ok = True
for i in range(0, 10_000, 97):
    s = scored[i]
    best = s.Q
    for e in (s.panel.entries if s.panel is not None else []):
        if e.n >= VC.n_min_regret:
            a = sc.alternative_ids(e.desc_ids, ids_q[i])[None, :]
            best = max(best, float(qm.predict_q(q, hq[i:i + 1], a, [query["tier"][i]], [query["actor_side"][i]])[0]))
    recomp_ok &= abs(s.R - (s.Q - best)) < 1e-5
check("R matches an independent per-action recomputation (104 sampled)", recomp_ok)
low = sum("q_low_support" in s.flags for s in scored)
check("q_low_support flags set where own key has n < n_min", low > 0, f"{low} flagged")

section("outputs: actions_scored.parquet and DB rows")
df = sc.to_frame(scored, run_id="run_test")
check("to_frame schema = SCORED_SCHEMA", df.schema == pl.Schema(sc.SCORED_SCHEMA))
check("to_frame keeps every row", df.height == len(scored))
check("no null Q/B/D/R/X in the frame", all(df[c].null_count() == 0 for c in ("Q", "B", "D", "R", "X")))
rows = sc.db_rows(df, "run_test")
check("DB row keys == db.SCORE_COLUMNS", all(tuple(r) == db.SCORE_COLUMNS for r in rows))
with tempfile.TemporaryDirectory() as tmp:
    pq = Path(tmp) / "actions_scored.parquet"
    df.write_parquet(pq)
    back = pl.read_parquet(pq)
    check("parquet round trip equal", back.equals(df))
    old = os.environ.get(db.ENV_DB)
    os.environ[db.ENV_DB] = str(Path(tmp) / "engine.db")
    try:
        n = db.insert_action_scores(rows[:500])
        c = sqlite3.connect(Path(tmp) / "engine.db")
        got = c.execute("SELECT COUNT(*), SUM(ABS(B + D + X - V) > 1e-9), MAX(R) FROM action_scores").fetchone()
        c.close()
        check("db.insert_action_scores accepts the rows", n == 500 and got[0] == 500)
        check("identity holds in the DB, R <= 0", got[1] == 0 and got[2] <= 0.0)
    finally:
        if old is None:
            os.environ.pop(db.ENV_DB, None)
        else:
            os.environ[db.ENV_DB] = old

sys.exit(summary())
