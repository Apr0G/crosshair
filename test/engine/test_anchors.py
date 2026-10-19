"""
P11.1 external anchors on synthetic scored frames with planted structure: every anchor
returns the planted value and the right in-band / out-of-band / no-data flag; `enrich`
joins actions and states causally; the report is written and never fails on an
out-of-band result; `main` without data exits 2.

    .venv\\Scripts\\python.exe test\\engine\\test_anchors.py
"""
import json
import tempfile
from pathlib import Path

import polars as pl

from _harness import ROOT, check, close, section, summary
from engine import config
from engine.evaluation import anchors as an
from engine.state.players import player_id_hash
from engine.value.scores import SCORED_SCHEMA

SCHEMA = dict(an.enrich(pl.DataFrame(schema=SCORED_SCHEMA)).schema)
_n = [0]


def act(match: str = "m1", rnd: int = 1, type_: str = "kill", side: str = "ct", tick: int = 100,
        V: float | None = None, V_direct: float | None = None, **kw) -> dict:
    _n[0] += 1
    r = {c: None for c in SCHEMA}
    r.update(match_key=match, round_num=rnd, action_id=f"a{_n[0]:06d}", actor_side=side, type=type_,
             tick=tick, V_direct=V_direct, V=V if V is not None else V_direct, flags=[], enabler_kinds=[])
    r.update(kw)
    return r


def frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMA)


BAL = dict(alive_ct_pre=5, alive_t_pre=5, equip_ct_pre=4000.0, equip_t_pre=4200.0)


# ── helpers ───────────────────────────────────────────────────────────────────
section("rank / spearman")
check("ties get the average rank", list(an.rank([1, 2, 2, 3])) == [1.0, 2.5, 2.5, 4.0])
check("monotone -> rho 1", close(an.spearman([1, 2, 3, 4], [10, 20, 25, 90]), 1.0))
check("reversed -> rho -1", close(an.spearman([1, 2, 3], [3, 2, 1]), -1.0))
check("constant side -> None", an.spearman([1, 2, 3], [5, 5, 5]) is None)
check("< 3 pairs -> None", an.spearman([1, 2], [1, 2]) is None)
check("None / NaN pairs dropped", close(an.spearman([1, None, 2, 3, float("nan")], [1, 5, 2, 3, 4]), 1.0))


# ── P11.1.1 opening kill ──────────────────────────────────────────────────────
section("opening kill band (P11.1.1)")
rows = []
for i in range(10):
    side = "ct" if i % 2 else "t"
    rows.append(act("m1", i + 1, tick=200, side=side, V_direct=0.20, map_version="de_mirage@unknown", **BAL))
    rows.append(act("m1", i + 1, tick=300, V_direct=0.90, map_version="de_mirage@unknown", **BAL))   # not first
    rows.append(act("m2", i + 1, tick=150, side=side, V_direct=0.16, map_version="de_inferno@unknown", **BAL))
rows.append(act("m3", 1, tick=100, V_direct=0.95, alive_ct_pre=5, alive_t_pre=5, equip_ct_pre=8000.0,
                equip_t_pre=2000.0))                                                 # unbalanced (ratio 4)
rows.append(act("m3", 2, tick=100, V_direct=0.95, alive_ct_pre=4, alive_t_pre=5, equip_ct_pre=4000.0,
                equip_t_pre=4000.0))                                                 # not a 5v5
rows.append(act("m3", 3, tick=100, V_direct=0.95, alive_ct_pre=5, alive_t_pre=5))   # equipment unknown
rows.append(act("m3", 4, type_="peek", tick=50, V_direct=0.95, **BAL))              # not a kill
ok_frame = frame(rows)
r = an.opening_kill_band(ok_frame)
check("value = mean |V_direct| of balanced first kills (0.18)", close(r["value"], 0.18, 1e-12), f"{r.get('value')}")
check("n = 20 balanced opening kills", r["n"] == 20, f"{r['n']}")
check("in band [0.12, 0.30]", r["status"] == an.IN_BAND)
check("excluded: unbalanced + not 5v5 = 2; unknown equipment = 1",
      r["n_excluded"] == 2 and r["n_unknown_equipment"] == 1, f"{r['n_excluded']} {r['n_unknown_equipment']}")
check("by map: mirage 0.20, inferno 0.16",
      close(r["by_map"]["de_mirage@unknown"]["mean"], 0.20) and close(r["by_map"]["de_inferno@unknown"]["mean"], 0.16))
check("first kill picked by tick, not by row order",
      an.first_kills(ok_frame).filter(pl.col("match_key") == "m1")["V_direct"].to_list() == [0.20] * 10)
hot = frame([act("h", i, tick=10, V_direct=0.9, **BAL) for i in range(1, 6)])
check("0.9 -> out of band", an.opening_kill_band(hot)["status"] == an.OUT_OF_BAND)
check("custom band respected", an.opening_kill_band(hot, band=(0.8, 1.0))["status"] == an.IN_BAND)
check("missing equipment columns -> no data",
      an.opening_kill_band(ok_frame.drop("equip_ct_pre"))["status"] == an.NO_DATA)
check("no balanced kills -> no data (not a failure)",
      an.opening_kill_band(frame([act("x", 1, V_direct=0.2, alive_ct_pre=5, alive_t_pre=5)]))["status"] == an.NO_DATA)


# ── P11.1.2 duel difficulty ───────────────────────────────────────────────────
section("duel difficulty (P11.1.2)")
plan = {("rifle", "pistol"): 0.05, ("rifle", "rifle"): 0.10, ("smg", "rifle"): 0.15, ("pistol", "rifle"): 0.20}


def duel_rows(values: dict) -> list[dict]:
    out = []
    for (kw, vw), v in values.items():
        for j in range(6):
            out.append(act("d", j + 1, V=v + 0.001 * j, weapon_class=kw, victim_weapon_class=vw,
                           alive_ct_pre=5, alive_t_pre=5))
    return out


dd = an.duel_difficulty(frame(duel_rows(plan)))
check("disadvantage levels -2, 0, +1, +2", sorted(dd["levels"]) == [-2, 0, 1, 2], f"{sorted(dd['levels'])}")
check("Spearman of mean V over levels = +1", close(dd["value"], 1.0))
check("rifle-vs-rifle > rifle-vs-pistol in bucket 0",
      len(dd["comparisons"]) == 1 and dd["comparisons"][0]["ok"] and dd["comparisons"][0]["alive_diff"] == 0)
check("in band", dd["status"] == an.IN_BAND)
rev = an.duel_difficulty(frame(duel_rows({k: 0.3 - v for k, v in plan.items()})))
check("reversed values -> out of band, rho -1", rev["status"] == an.OUT_OF_BAND and close(rev["value"], -1.0))
check("disadvantage(): pistol killer vs rifle victim = +2, unknown class -> None",
      an.disadvantage("pistol", "rifle") == 2 and an.disadvantage("grenade", "rifle") is None)
check("missing victim weapon column -> no data",
      an.duel_difficulty(frame(duel_rows(plan)).drop("victim_weapon_class"))["status"] == an.NO_DATA)


# ── P11.1.3 man advantage ─────────────────────────────────────────────────────
section("man advantage ordering (P11.1.3)")


def states_rows(w: dict) -> pl.DataFrame:
    rows = []
    for bucket_t, bomb in ((55.0, 0), (25.0, 1)):
        for sit, (ct, t) in an.SITUATIONS.items():
            rows += [{"W": w[sit], "ctx_alive_ct": ct, "ctx_alive_t": t, "ctx_time_remaining_s": bucket_t + k,
                      "ctx_bomb_planted": bomb} for k in range(6)]
    rows += [{"W": 0.01, "ctx_alive_ct": 3, "ctx_alive_t": 2, "ctx_time_remaining_s": 95.0,
              "ctx_bomb_planted": 0} for _ in range(6)]              # unmatched stratum: only 3v2
    return pl.DataFrame(rows)


ma = an.man_advantage_ordering(states_rows({"3v2": 0.70, "2v3": 0.35, "1v2": 0.20}))
check("matched means 0.70 > 0.35 > 0.20",
      close(ma["value"]["3v2"], 0.70) and close(ma["value"]["2v3"], 0.35) and close(ma["value"]["1v2"], 0.20),
      f"{ma.get('value')}")
check("2 matched strata; unmatched 3v2 rows only in raw means",
      ma["n_strata"] == 2 and ma["raw_means"]["3v2"] < 0.70)
check("in band", ma["status"] == an.IN_BAND)
check("2v3 above 3v2 -> out of band",
      an.man_advantage_ordering(states_rows({"3v2": 0.30, "2v3": 0.35, "1v2": 0.20}))["status"] == an.OUT_OF_BAND)
check("no W column -> no data", an.man_advantage_ordering(states_rows(
    {"3v2": 0.7, "2v3": 0.3, "1v2": 0.2}).drop("W"))["status"] == an.NO_DATA)
ks_rows = ([act("k", i, side="t", V_direct=0.30, alive_ct_pre=2, alive_t_pre=1) for i in range(5)]
           + [act("k", 10 + i, side="ct", V_direct=-0.05, alive_ct_pre=4, alive_t_pre=5) for i in range(5)])
ks = an.kill_swing_ordering(frame(ks_rows))
check("kill swing: 2v1->1v1 (0.30) > 5v4->4v4 (0.05), in band",
      ks["status"] == an.IN_BAND and close(ks["value"]["2v1->1v1"], 0.30) and close(ks["value"]["5v4->4v4"], 0.05))


# ── P11.1.4 clutch defuse ─────────────────────────────────────────────────────
section("clutch defuse sign (P11.1.4)")
clutch = dict(alive_ct_pre=1, alive_t_pre=2, time_remaining_s_pre=6.0)
cd_rows = ([act("c", i, type_="defuse_start", V=0.15, **clutch) for i in range(3)]
           + [act("c", 9, type_="defuse_start", V=-0.50, alive_ct_pre=1, alive_t_pre=2, time_remaining_s_pre=20.0)]
           + [act("c", 10, type_="defuse_start", V=-0.50, alive_ct_pre=2, alive_t_pre=2, time_remaining_s_pre=6.0)]
           + [act("c", 11 + i, type_="defuse_abort", V=-0.20, **clutch) for i in range(2)])
cd = an.clutch_defuse_sign(frame(cd_rows))
check("start mean +0.15 (time >= 10 s and 2v2 excluded), abort mean -0.20",
      close(cd["value"]["defuse_start"], 0.15) and close(cd["value"]["defuse_abort"], -0.20), f"{cd.get('value')}")
check("in band; n_start 3, n_abort 2", cd["status"] == an.IN_BAND and cd["n_start"] == 3 and cd["n_abort"] == 2)
bad_cd = an.clutch_defuse_sign(frame([act("c", 1, type_="defuse_start", V=-0.1, **clutch)]))
check("negative start -> out of band, abort sign noted as unchecked",
      bad_cd["status"] == an.OUT_OF_BAND and bad_cd["note"] is not None)
check("no clutch defuse -> no data", an.clutch_defuse_sign(frame(cd_rows[3:5]))["status"] == an.NO_DATA)


# ── P11.1.5 player rating ─────────────────────────────────────────────────────
section("player rating correlation (P11.1.5)")
fixture = ROOT / "test" / "engine" / "fixtures" / "hltv_ratings.csv"
check("fixture exists with the documented header",
      fixture.is_file() and fixture.read_text(encoding="utf-8").splitlines()[0].split(",") == list(an.HLTV_COLUMNS))
sids = [76561198000000000 + i for i in range(1, 7)]
pr_rows = []
for i, sid in enumerate(sids):
    for j in range(i + 1):
        pr_rows.append(act("pm1", 1 + j, V=0.05, actor_id_hash=player_id_hash(sid)))
pr_frame = frame(pr_rows + [act("pm1", 1, type_="team_buy", V=0.4)])     # team row (no actor) ignored
with tempfile.TemporaryDirectory() as td:
    csv_path = Path(td) / "ratings.csv"
    lines = [",".join(an.HLTV_COLUMNS)]
    lines += [f"pm1,de_mirage,{sid},player{i},{0.6 + 0.1 * i:.2f}" for i, sid in enumerate(sids)]
    lines += ["other,de_mirage,76561198999999999,ghost,1.30", ""]
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    pr = an.player_rating_correlation(pr_frame, csv_path)
    check("6 player-maps matched, rho = +1, in band",
          pr["n"] == 6 and close(pr["value"], 1.0) and pr["status"] == an.IN_BAND, f"{pr.get('value')} n={pr['n']}")
    check("fewer than 50 matches noted", pr["note"] is not None and "50" in pr["note"])
    check("disagreements listed (<= 10)", 0 < len(pr["disagreements"]) <= an.TOP_DISAGREEMENTS)
    flipped = Path(td) / "flipped.csv"
    flipped.write_text("\n".join([lines[0]] + [f"pm1,de_mirage,{sid},p{i},{1.6 - 0.1 * i:.2f}"
                                                for i, sid in enumerate(sids)]), encoding="utf-8")
    check("anti-correlated ratings -> out of band",
          an.player_rating_correlation(pr_frame, flipped)["status"] == an.OUT_OF_BAND)
    check("header-only fixture -> no data", an.player_rating_correlation(pr_frame, fixture)["status"] == an.NO_DATA)
    check("no CSV -> no data", an.player_rating_correlation(pr_frame, None)["status"] == an.NO_DATA)
    bad_csv = Path(td) / "bad.csv"
    bad_csv.write_text("match_key,rating\nx,1\n", encoding="utf-8")
    try:
        an.player_rating_correlation(pr_frame, bad_csv)
        check("CSV with wrong header raises", False)
    except ValueError:
        check("CSV with wrong header raises", True)


# ── enrich ────────────────────────────────────────────────────────────────────
section("enrich (actions + states joined causally)")
scored = pl.DataFrame([
    {**{c: None for c in SCORED_SCHEMA}, "match_key": "e1", "round_num": 3, "action_id": "k1", "actor_side": "ct",
     "type": "kill", "V": 0.2, "V_direct": 0.2, "flags": []},
    {**{c: None for c in SCORED_SCHEMA}, "match_key": "e1", "round_num": 3, "action_id": "d1", "actor_side": "ct",
     "type": "defuse_start", "V": 0.1, "V_direct": 0.1, "flags": []},
    {**{c: None for c in SCORED_SCHEMA}, "match_key": "e1", "round_num": 4, "action_id": "zz", "actor_side": "t",
     "type": "peek", "V": 0.0, "V_direct": 0.0, "flags": []},
], schema=SCORED_SCHEMA)
actions = pl.DataFrame([
    {"action_id": "k1", "tick": 101, "state_pre_tick": 100, "subtype": None,
     "params": json.dumps({"weapon_class": "rifle", "victim_id": 7}),
     "enablers": json.dumps([{"action_id": "f1", "kind": "flash", "beneficiary_id": 3}])},
    {"action_id": "d1", "tick": 301, "state_pre_tick": 300, "subtype": None,
     "params": json.dumps({"kit": True, "ct_alive": 1, "t_alive": 2}), "enablers": "[]"},
])
states = pl.DataFrame([
    {"match_key": "e1", "round_num": 3, "tick": 64, "ctx_alive_ct": 5, "ctx_alive_t": 5, "ctx_equip_ct": 4000.0,
     "ctx_equip_t": 3900.0, "ctx_time_remaining_s": 100.0, "ctx_bomb_planted": 0, "ctx_map_version": "de_nuke@x",
     "players": [{"player_id": 7, "weapon_class": "pistol"}, {"player_id": 3, "weapon_class": "rifle"}],
     "y_round_won_ct": 1, "y_round_end_reason": "t_killed"},
    {"match_key": "e1", "round_num": 3, "tick": 99, "ctx_alive_ct": 4, "ctx_alive_t": 5, "ctx_equip_ct": 3500.0,
     "ctx_equip_t": 3900.0, "ctx_time_remaining_s": 80.0, "ctx_bomb_planted": 0, "ctx_map_version": "de_nuke@x",
     "players": [{"player_id": 7, "weapon_class": "smg"}], "y_round_won_ct": 1, "y_round_end_reason": "t_killed"},
    {"match_key": "e1", "round_num": 3, "tick": 400, "ctx_alive_ct": 1, "ctx_alive_t": 0, "ctx_equip_ct": 100.0,
     "ctx_equip_t": 0.0, "ctx_time_remaining_s": 3.0, "ctx_bomb_planted": 1, "ctx_map_version": "de_nuke@x",
     "players": [], "y_round_won_ct": 1, "y_round_end_reason": "t_killed"},
])
en = an.enrich(scored, actions, states)
k1 = en.filter(pl.col("action_id") == "k1").row(0, named=True)
d1 = en.filter(pl.col("action_id") == "d1").row(0, named=True)
zz = en.filter(pl.col("action_id") == "zz").row(0, named=True)
check("every ANCHOR_COLUMN present", all(c in en.columns for c in an.ANCHOR_COLUMNS))
check("state at pre tick 100 = latest at or before (tick 99), never the future",
      k1["alive_ct_pre"] == 4 and k1["equip_ct_pre"] == 3500.0)
check("victim weapon class from the pre-state's player tokens", k1["victim_weapon_class"] == "smg")
check("params: weapon_class rifle, victim 7; enabler kinds [flash]",
      k1["weapon_class"] == "rifle" and k1["victim_id"] == 7 and k1["enabler_kinds"] == ["flash"])
check("round labels joined (won_ct 1, end reason)", k1["round_won_ct"] == 1 and k1["round_end_reason"] == "t_killed")
check("defuse pre-state is tick 99 (not the tick-400 future state)", d1["time_remaining_s_pre"] == 80.0)
check("action without actions/states row -> nulls, not 0", zz["tick"] is None and zz["alive_ct_pre"] is None)
check("row count and V preserved", en.height == 3 and en["V"].to_list() == scored["V"].to_list())
no_states = an.enrich(scored, actions, None)
check("no states: alive counts fall back to defuse params",
      no_states.filter(pl.col("action_id") == "d1")["alive_ct_pre"][0] == 1)


# ── report + main ─────────────────────────────────────────────────────────────
section("run() report and main()")
cfg = config.load()
full = pl.concat([ok_frame, frame(duel_rows(plan)), frame(ks_rows), frame(cd_rows), hot])
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "eval" / "anchors_wp_test.md"
    res = an.run(full, states_rows({"3v2": 0.70, "2v3": 0.35, "1v2": 0.20}), cfg, out, hltv_csv=None,
                 model="wp_test")
    text = out.read_text(encoding="utf-8") if out.is_file() else ""
    check("report written under a created directory", out.is_file())
    check("six anchors, each with a status", len(res) == 6 and all(r["status"] in (an.IN_BAND, an.OUT_OF_BAND,
                                                                                    an.NO_DATA) for r in res))
    check("hot opening kills pull the mean out of band -> flagged, not raised",
          res[0]["status"] == an.OUT_OF_BAND and "OUT OF BAND" in text)
    check("in-band and no-data rows also listed", "in band" in text and "no data" in text)
    check("player rating without CSV is 'no data'", res[5]["status"] == an.NO_DATA)
    check("report names the model and the by-map table", "wp_test" in text and "de_mirage@unknown" in text)
    res2 = an.run(full, None, cfg, Path(td) / "b.md")
    check("no states frame -> man advantage no data", res2[2]["status"] == an.NO_DATA)
check("main() without scored data exits 2", an.main([]) == 2)
check("main() with --model-dir only exits 2", an.main(["--model-dir", "nowhere/wp_v0"]) == 2)
with tempfile.TemporaryDirectory() as td:
    sp, out = Path(td) / "actions_scored.parquet", Path(td) / "anchors.md"
    full.write_parquet(sp)
    rc = an.main(["--model-dir", "x/wp_test", "--scored", str(sp), "--out", str(out), "--hltv-csv", str(fixture)])
    check("main() on a scored parquet exits 0 and writes the report", rc == 0 and out.is_file())

raise SystemExit(summary())
