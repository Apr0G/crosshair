"""
P10 end to end: `analyze.analyze_tables` on EVERY synthetic story with FlatGeometry and
RuleStub(good_win).

    .venv\\Scripts\\python.exe test\\engine\\test_analyze_e2e.py

Checks per story: report.json validates against engine/report/schema.json; the
conservation residual of every round < 1e-6; every action row has V and value fields
(None while the value stage is skipped); every chart exists with size > 0;
actions_scored.parquet reads back with one row per scored action; action_scores rows
in the (temporary) engine DB. Then: two runs give byte-identical report.json except
meta.run_id / created_at / timings; the staged states+actions equal `extractor.run`;
the local SCORED_SCHEMA mirrors `engine.value.scores`; and the value stage runs with
tiny untrained Q/B nets and a FAISS index (skipped when torch / faiss / the value
modules cannot be imported).
"""
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from _harness import check, section, skip, summary

TMP = Path(tempfile.mkdtemp(prefix="crosshair_e2e_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")       # never the real DB

import numpy as np                                               # noqa: E402
import polars as pl                                              # noqa: E402

from _cf_states import RuleStub, good_win                        # noqa: E402
from _synthetic import STORIES, make_raw_tables                  # noqa: E402
from engine import analyze, config, db                           # noqa: E402
from engine.report import writer                                 # noqa: E402
from engine.state.geometry import FlatGeometry                   # noqa: E402

CFG = config.load()
VALUE_KEYS = ("Q", "B", "D", "R", "X")


def run(story: str, out: Path, **kw):
    return analyze.analyze_tables(make_raw_tables(story), match_key=f"syn_{story}", tier="pro",
                                  geometry=FlatGeometry(), predictor=RuleStub(good_win), cfg=CFG,
                                  out_dir=out, **kw)


# ── every story ───────────────────────────────────────────────────────────────

for story in STORIES:
    section(f"story {story}")
    out = TMP / story
    res = run(story, out)
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    check("report.json validates against schema.json", not writer.validate(rep), "; ".join(writer.validate(rep)[:3]))
    check("report_schema_version is 1", rep["report_schema_version"] == 1)
    check("report.json is in canonical form", writer.dumps(rep) == (out / "report.json").read_text(encoding="utf-8"))
    resid = [abs(r["conservation_residual"]) for r in rep["rounds"]]
    check("conservation residual < 1e-6 in every round", resid and max(resid) < 1e-6, f"max {max(resid):.1e}")
    check("residual matches the attribution", all(abs(r.conservation_residual) < 1e-6 for r in res.round_attributions))
    rows = [a for r in rep["rounds"] for a in r["actions"]]
    check("every action row has a finite V", rows and all(isinstance(a["V"], float) for a in rows))
    check("value fields present and None (stage skipped)",
          all(all(k in a and a[k] is None for k in VALUE_KEYS) for a in rows))
    check("meta.value_stage is skipped + a warning says so",
          rep["meta"]["value_stage"] == "skipped" and any(w["code"] == "value_stage_skipped" for w in rep["warnings"]))
    check("V = V_direct + V_enabler + V_movement on every row",
          all(abs(a["V"] - a["V_direct"] - a["V_enabler"] - a["V_movement"]) < 1e-9 for a in rows))
    check("wp_curve per round, W in [0, 1]",
          all(r["wp_curve"] and all(0 <= p["W"] <= 1 for p in r["wp_curve"]) for r in rep["rounds"]))
    check("wp_curve ends at the round's W pins",
          all(abs(r["wp_curve"][0]["W"] - r["w_start"]) < 1e-12 and abs(r["wp_curve"][-1]["W"] - r["w_end"]) < 1e-12
              for r in rep["rounds"]))
    expected = ([f"round_{r['round_num']}_wp.png" for r in rep["rounds"]]
                + [f"player_{p['player']}_summary.png" for p in rep["players"]] + ["team_clock.png"])
    sizes = {n: (out / "charts" / n).stat().st_size if (out / "charts" / n).is_file() else 0 for n in expected}
    check("every chart exists with size > 0", all(v > 0 for v in sizes.values()),
          ", ".join(n for n, v in sizes.items() if not v))
    for name in ("report.md", "manifest.json", "timings.json"):
        check(f"{name} written", (out / name).is_file() and (out / name).stat().st_size > 0)
    frame = pl.read_parquet(out / "actions_scored.parquet")
    check("actions_scored.parquet readable, one row per scored action", frame.height == len(rows),
          f"{frame.height} vs {len(rows)}")
    check("actions_scored columns = SCORED_SCHEMA", list(frame.columns) == list(analyze.SCORED_SCHEMA))
    check("actions_scored run_id is the run's", frame["run_id"].unique().to_list() == [res.run_id])
    check("db action_scores rows written", res.n_db_rows == frame.height and db.stats()["action_scores"] > 0)
    tim = json.loads((out / "timings.json").read_text(encoding="utf-8"))
    check("timings.json has states/actions/predict/attribution/value/report/total",
          all(k in tim for k in ("states", "actions", "predict", "attribution", "value", "report", "total")))
    check("players present with hashed ids", rep["players"] and all(p["player"].startswith("p_") for p in rep["players"]))
    check("teams: ct and t", [t["side"] for t in rep["teams"]] == ["ct", "t"])


# ── reproducibility ───────────────────────────────────────────────────────────

section("reproducibility (base twice)")
r1, r2 = run("base", TMP / "rep1"), run("base", TMP / "rep2")
b1 = (TMP / "rep1" / "report.json").read_text(encoding="utf-8")
b2 = (TMP / "rep2" / "report.json").read_text(encoding="utf-8")
check("run ids differ", r1.run_id != r2.run_id)
check("report.json identical except run_id / created_at / timings",
      writer.dumps(writer.strip_dynamic(json.loads(b1))) == writer.dumps(writer.strip_dynamic(json.loads(b2))))
diff = [(x, y) for x, y in zip(b1.splitlines(), b2.splitlines()) if x != y]
allowed = ('"run_id"', '"created_at"', '"states"', '"actions"', '"predict"', '"attribution"', '"value"', '"report"')
check("only dynamic lines differ in the bytes", all(x.strip().startswith(allowed) for x, _ in diff),
      str(diff[:2]))
f1 = pl.read_parquet(TMP / "rep1" / "actions_scored.parquet").drop("run_id")
f2 = pl.read_parquet(TMP / "rep2" / "actions_scored.parquet").drop("run_id")
check("actions_scored identical apart from run_id", f1.equals(f2))


# ── usage log (P12.3) ─────────────────────────────────────────────────────────

section("usage log")
ulog = analyze.usage_log_path()
check("usage log sits next to the test DB", ulog == TMP / "usage.jsonl")
recs = [json.loads(line) for line in ulog.read_text(encoding="utf-8").splitlines()]
check("one usage record per DB-writing run", {r1.run_id, r2.run_id} <= {r["run_id"] for r in recs})
rec = next(r for r in recs if r["run_id"] == r1.run_id)
check("record has match_key, map_version, tier, timings, warnings, residual",
      rec["match_key"] == "syn_base" and rec["map_version"] == "de_test@flat" and rec["tier"] == "pro"
      and "total" in rec["timings"] and isinstance(rec["warnings"], int) and rec["conservation_residual"] < 1e-6)
n_before = len(recs)
run("base", TMP / "nodb", write_db=False)
check("--no-db run writes no usage record", len(ulog.read_text(encoding="utf-8").splitlines()) == n_before)
from engine.evaluation import usage as _usage                    # noqa: E402
real_append = _usage.append_usage


def _fail(*a, **k):
    raise OSError("disk full")


_usage.append_usage = _fail
try:
    r3 = run("base", TMP / "usage_fail")
    ok = r3.files["report.json"].is_file()
except Exception:
    ok = False
_usage.append_usage = real_append
check("a failing usage append does not fail the run", ok)
check("append_usage_record returns False on failure", analyze.append_usage_record({}, None) is False)


# ── staged extractor == extractor.run ─────────────────────────────────────────

section("staged states/actions equal extractor.run")
from engine.actions import extractor                             # noqa: E402
raw = make_raw_tables("retreat")
s_run, a_run, w_run = extractor.run(raw, CFG, match_key="m", geometry=FlatGeometry(), tier="pro")
s_st, a_st, w_st = analyze.build_states_actions(raw, CFG, match_key="m", tier="pro", geometry=FlatGeometry(),
                                                timings={})
check("same snapshot ticks", [(s.meta.round_num, s.meta.tick) for s in s_run] == [(s.meta.round_num, s.meta.tick) for s in s_st])
check("same action ids", [a.action_id for a in a_run] == [a.action_id for a in a_st])
check("same windows", [w.window_id for w in w_run] == [w.window_id for w in w_st])


# ── SCORED_SCHEMA mirror ──────────────────────────────────────────────────────

section("SCORED_SCHEMA mirrors engine.value.scores")
try:
    from engine.value import scores as _scores
except Exception as exc:                                     # torch / value package unavailable
    _scores = None
    skip("SCORED_SCHEMA mirror", f"engine.value.scores not importable: {type(exc).__name__}")
if _scores is not None:
    check("same columns and dtypes", dict(_scores.SCORED_SCHEMA) == dict(analyze.SCORED_SCHEMA))


# ── value stage with tiny models ──────────────────────────────────────────────

section("value stage (tiny untrained Q/B + FAISS index)")


class ToyEncoder:
    """Deterministic 8-d h_pre from the context token."""

    def encode(self, snaps):
        rows = [[s.ctx.alive_ct, s.ctx.alive_t, s.ctx.bomb_planted, s.ctx.time_into_round_s / 100.0,
                 s.ctx.time_remaining_s / 100.0, 1.0, (s.meta.tick % 97) / 97.0, 0.5] for s in snaps]
        return np.asarray(rows, dtype=np.float32)


def tiny_value_models(raw) -> analyze.ValueModels:
    import torch
    from engine.value import descriptors, neighbours, q_model
    snaps, actions, _ = analyze.build_states_actions(raw, CFG, match_key="train_m", tier="pro",
                                                     geometry=FlatGeometry(), timings={})
    by_tick = sorted(snaps, key=lambda s: (s.meta.round_num, s.meta.tick))
    descs, pre = [], []
    for a in actions:
        cand = [s for s in by_tick if s.meta.round_num == a.round_num and s.meta.tick <= (a.state_pre_tick or a.id_tick)]
        if cand:
            pre.append(cand[-1])
            descs.append(descriptors.encode(a, cand[-1]))
    vocab = descriptors.DescriptorVocab.fit(descs)
    torch.manual_seed(0)
    q, b = q_model.QNet(8, vocab.sizes(), hidden=(16,)), q_model.BNet(8, hidden=(16,))
    h = ToyEncoder().encode(pre)
    reps = 3                                              # every stored action thrice, so some keys reach n_min
    table = pl.DataFrame({
        "match_key": ["train_m"] * (len(descs) * reps),
        "round_num": [s.meta.round_num for s in pre] * reps,
        "action_id": [f"{i}" for i in range(len(descs) * reps)],
        "descriptor_key": [descriptors.key(d) for d in descs] * reps,
        "V": [0.01 * ((i % 7) - 3) for i in range(len(descs) * reps)],
        "side": [a.actor_side for a in actions if True][:len(descs)] * reps,
        "map_version": ["de_test@flat"] * (len(descs) * reps),
        "tier": ["pro"] * (len(descs) * reps),
        "desc_ids": [descriptors.to_ids(d, vocab).tolist() for d in descs] * reps,
    })
    idx = neighbours.build_index(np.vstack([h] * reps), table)
    k_map = {"by": "type_class",
             "X": {c: {"k": 2.0, "sigma2_noise": 0.01} for c in ("kill", "utility", "movement", "bomb", "economy", "damage")},
             "D": {c: {"k": 2.0, "sigma2_noise": 0.01} for c in ("kill", "utility", "movement", "bomb", "economy", "damage")}}
    return analyze.ValueModels(q, b, vocab, idx, k_map, ToyEncoder(), {"q": "q_toy", "neighbours": "nb_toy"})


try:
    vm = tiny_value_models(make_raw_tables("flash"))
except Exception as exc:
    vm = None
    skip("value stage", f"value models could not be built: {type(exc).__name__}: {exc}")
if vm is not None:
    cfg_v = replace(CFG, value=replace(CFG.value, k_neighbours=20, n_min_regret=3))
    out = TMP / "value"
    res = analyze.analyze_tables(make_raw_tables("base"), match_key="syn_value", tier="pro", geometry=FlatGeometry(),
                                 predictor=RuleStub(good_win), cfg=cfg_v, value_models=vm, out_dir=out)
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    rows = [a for r in rep["rounds"] for a in r["actions"]]
    real = [a for a in rows if not a["synthetic"]]
    check("value report validates", not writer.validate(rep), "; ".join(writer.validate(rep)[:3]))
    check("meta.value_stage done, model versions named",
          rep["meta"]["value_stage"] == "done" and rep["meta"]["model_versions"]["q"] == "q_toy")
    check("every non-synthetic action has Q, B, D, R, X", real and all(a[k] is not None for a in real for k in VALUE_KEYS))
    check("synthetic rows keep None value fields", all(a[k] is None for a in rows if a["synthetic"] for k in VALUE_KEYS))
    check("identity V = B + D + X", all(abs(a["B"] + a["D"] + a["X"] - a["V"]) < 1e-9 for a in real))
    check("R <= 0", all(a["R"] <= 1e-12 for a in real))
    check("panels with at most 20 entries", any(a["panel"] for a in real)
          and all(len(a["panel"]["top"]) <= 20 for a in real if a["panel"]))
    check("player by_class aggregates present", any(p.get("by_class") for p in rep["players"]))
    check("player charts written", all((out / "charts" / f"player_{p['player']}_summary.png").stat().st_size > 0
                                       for p in rep["players"]))
    frame = pl.read_parquet(out / "actions_scored.parquet")
    check("actions_scored Q filled for real actions", frame.filter(pl.col("Q").is_not_null()).height == len(real))

sys.exit(summary())
