"""
P11.2 review-loop tooling: category detection per round, seeded selection covering every
category present (and reporting the ones absent), case directories with the top-3
actions, the review-log template round trip, strict judgement parsing, and every
"disagree" turned into a numbered item.

    .venv\\Scripts\\python.exe test\\engine\\test_review.py
"""
import json
import tempfile
from pathlib import Path

import polars as pl

from _harness import check, section, summary
from engine.evaluation import anchors as an
from engine.evaluation import review as rv
from engine.value.scores import SCORED_SCHEMA

SCHEMA = dict(an.enrich(pl.DataFrame(schema=SCORED_SCHEMA)).schema)
_n = [0]


def act(rnd: int, type_: str, side: str = "ct", tick: int = 100, V: float = 0.01, match: str = "m1", **kw) -> dict:
    _n[0] += 1
    r = {c: None for c in SCHEMA}
    r.update(match_key=match, round_num=rnd, action_id=f"{_n[0]:016x}", actor_side=side, type=type_, tick=tick,
             V=V, V_direct=V, D=V / 2, X=V / 4, panel_n=10, flags=[], enabler_kinds=[], actor_id_hash=1000 + _n[0])
    r.update(kw)
    return r


EXPECTED = {
    2: {"opening_kill"},
    3: {"opening_kill", "trade"},
    4: {"opening_kill", "flash_assisted_kill"},
    5: {"smoke_covered_plant"},
    6: {"clutch"},
    7: {"save"},
    8: {"opening_kill", "double_kill_same_tick"},
    9: {"clock_decided"},
    10: {"eco_win"},
    11: {"failed_defuse"},
}


def corpus(with_abort: bool = True) -> pl.DataFrame:
    rows = [act(2, "kill", V=0.20)]
    rows += [act(3, "kill", V=0.15, enabler_kinds=["trade_setup"]), act(3, "trade_setup", V=0.03)]
    rows += [act(4, "kill", V=0.12, enabler_kinds=["flash"]), act(4, "flash", V=0.02)]
    plant = act(5, "plant_complete", side="t", V=0.10)
    rows += [plant, act(5, "smoke_pop", side="t", V=0.01, components=json.dumps(
        [{"source": "enabler", "kind": "smoke_cover", "event_id": plant["action_id"], "phi": -0.01}]))]
    rows += [act(6, "peek", side="ct", alive_ct_pre=1, alive_t_pre=3, V=-0.02)]
    rows += [act(7, "save", side="t", V=0.0)]
    rows += [act(8, "kill", tick=500, V=0.1), act(8, "kill", tick=500, side="t", V=0.09)]
    rows += [act(9, "hold", V=0.0, round_end_reason="target_saved")]
    rows += [act(10, "team_buy", side="ct", V=0.0, team_class="eco", round_won_ct=1, actor_id_hash=None),
             act(10, "team_buy", side="t", V=0.0, team_class="full", round_won_ct=1, actor_id_hash=None)]
    if with_abort:
        rows += [act(11, "defuse_abort", V=-0.05)]
    # negative controls: pistol round with an "eco" buy, eco side that lost, a non-clutch 1v1
    rows += [act(1, "team_buy", side="ct", team_class="pistol", round_won_ct=1, actor_id_hash=None),
             act(1, "team_buy", side="t", team_class="pistol", round_won_ct=1, actor_id_hash=None)]
    rows += [act(12, "team_buy", side="t", team_class="eco", round_won_ct=1, actor_id_hash=None),
             act(12, "peek", alive_ct_pre=1, alive_t_pre=1, round_end_reason="t_killed")]
    rows += [act(r, "peek", V=0.001 * r) for r in range(13, 33)]          # filler rounds, no category
    return pl.DataFrame(rows, schema=SCHEMA)


frame = corpus()

# ── categories ────────────────────────────────────────────────────────────────
section("round categories (P11.2.1)")
cats = rv.round_categories(frame)
for rnd, want in EXPECTED.items():
    check(f"round {rnd}: {sorted(want)}", cats[("m1", rnd)] == want, f"{sorted(cats[('m1', rnd)])}")
check("pistol round with pistol buys is not an eco win", cats[("m1", 1)] == set())
check("eco side that lost / 1v1 / non-clock reason give nothing", cats[("m1", 12)] == set())
check("filler rounds have no category", all(not cats[("m1", r)] for r in range(13, 33)))
check("components path: smoke_cover found via event_id",
      "smoke_cover" in rv.enabler_kinds_by_event(frame)[frame.filter(pl.col("type") == "plant_complete")["action_id"][0]])

# ── selection ─────────────────────────────────────────────────────────────────
section("selection covers every category present")
sel = rv.select_review_rounds(frame, n=20, seed=7)
keys = [(s["match_key"], s["round_num"]) for s in sel]
check("20 rounds, all distinct", len(sel) == 20 and len(set(keys)) == 20)
check("every category covered", rv.missing_categories(sel) == [], f"{rv.missing_categories(sel)}")
check("all 10 category rounds chosen", all(("m1", r) in keys for r in EXPECTED))
check("same seed -> same selection", rv.select_review_rounds(frame, n=20, seed=7) == sel)
check("selection entries carry sorted categories", all(s["categories"] == sorted(s["categories"]) for s in sel))
small = rv.select_review_rounds(frame, n=4, seed=1)
check("n=4: 4 rounds, greedy picks the multi-category rounds first, rest reported missing",
      len(small) == 4 and len(rv.missing_categories(small)) == 10 - len(rv.covered(small)))
excl = rv.select_review_rounds(frame, n=20, seed=7, exclude=[("m1", 10)])
check("exclude: the only eco round skipped -> eco_win missing",
      ("m1", 10) not in [(s["match_key"], s["round_num"]) for s in excl] and rv.missing_categories(excl) == ["eco_win"])
no_abort = rv.select_review_rounds(corpus(with_abort=False), n=20, seed=7)
check("category absent from the data -> reported missing (failed_defuse)",
      rv.missing_categories(no_abort) == ["failed_defuse"])
check("fewer rounds than n -> every round once", len(rv.select_review_rounds(frame.filter(pl.col("round_num") < 5),
                                                                            n=20)) == 4)

# ── case directories ──────────────────────────────────────────────────────────
section("review set directories")
with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "eval"
    dirs = rv.write_review_set(frame, root, n=20, seed=7)
    check("20 review_round_<i> directories with case.json",
          len(dirs) == 20 and all((d / rv.CASE_NAME).is_file() for d in dirs)
          and [d.name for d in dirs] == [f"review_round_{i}" for i in range(1, 21)])
    cases = rv.read_cases(root)
    check("cases read back in index order", [c["index"] for c in cases] == list(range(1, 21)))
    c3 = next(c for c in cases if c["round_num"] == 3)
    check("top actions sorted by |V|, at most 3, ranks 1..k",
          [a["rank"] for a in c3["top_actions"]] == [1, 2] and c3["top_actions"][0]["V"] == 0.15)
    c8 = next(c for c in cases if c["round_num"] == 8)
    check("case carries categories, n_actions, V/D/X, verdict fields",
          c8["categories"] == ["double_kill_same_tick", "opening_kill"] and c8["n_actions"] == 2
          and {"V", "D", "X", "verdict_decision", "panel_n"} <= set(c8["top_actions"][0]))
    try:
        rv.write_review_set(frame, root, n=20, seed=7)
        check("second write refuses to overwrite existing cases", False)
    except FileExistsError:
        check("second write refuses to overwrite existing cases", True)

    # ── review log ────────────────────────────────────────────────────────────
    section("review log template round trip (P11.2.2)")
    log = rv.init_review_log(root / "review_log.md", cases)
    text = log.read_text(encoding="utf-8")
    n_rows = sum(len(c["top_actions"]) for c in cases)
    rows = rv.review_judgements(log)
    check("one row per top action, all todo", len(rows) == n_rows and
          all(set(r["judgements"].values()) == {"todo"} for r in rows))
    check("case markers round-trip match_key / round_num / action_id",
          {(r["round_index"], r["match_key"], r["round_num"]) for r in rows}
          == {(c["index"], c["match_key"], c["round_num"]) for c in cases if c["top_actions"]}
          and {r["action_id"] for r in rows} == {a["action_id"] for c in cases for a in c["top_actions"]})
    check("blank template -> no items", rv.parse_review_log(log) == [])
    s0 = rv.cycle_summary(log)
    check("summary: all todo, target not met", s0["todo"] == n_rows * 4 and not s0["target_met"])
    try:
        rv.init_review_log(log, cases)
        check("init refuses to overwrite an existing log", False)
    except FileExistsError:
        check("init refuses to overwrite an existing log", True)

    section("disagreements become numbered items (P11.2.3)")
    lines = text.splitlines()
    table_rows = [i for i, ln in enumerate(lines) if ln.startswith("| ") and ln.split("|")[1].strip().isdigit()]
    filled = []
    for k, i in enumerate(table_rows):
        cells = lines[i].split("|")
        if k == 0:
            cells[4:8] = [" agree ", " Disagree ", " unsure ", " agree "]
        elif k == 1:
            cells[4:8] = [" disagree: too big ", " agree ", " agree ", " disagree "]
        else:
            cells[4:8] = [" agree "] * 4
        lines[i] = "|".join(cells)
        filled.append(k)
    first_case = next(i for i, ln in enumerate(lines) if ln.startswith("Comment:"))
    lines[first_case] = "Comment: the flash share looks too small."
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    items = rv.parse_review_log(log)
    check("3 disagree cells -> items 1..3", [it["item"] for it in items] == [1, 2, 3], f"{len(items)}")
    check("aspects in log order: D, then V and panel",
          [it["aspect"] for it in items] == ["D", "V", "panel"])
    check("items carry round, action id and the round's comment",
          items[0]["action_id"] == rows[0]["action_id"] and items[0]["round_num"] == rows[0]["round_num"]
          and rows[0]["round_index"] == 1 and items[0]["comment"] == "the flash share looks too small.")
    s1 = rv.cycle_summary(log)
    check("summary: 2 disagreeing rows, nothing todo, target met (<= 2)",
          s1["disagree_rows"] == 2 and s1["todo"] == 0 and s1["target_met"], f"{s1}")
    bad = log.read_text(encoding="utf-8").replace("| agree |", "| maybe |", 1)
    log.write_text(bad, encoding="utf-8")
    try:
        rv.review_judgements(log)
        check("an unknown judgement word raises with the line number", False)
    except ValueError as exc:
        check("an unknown judgement word raises with the line number", "line" in str(exc))

check("main() without data exits 2", rv.main([]) == 2)

raise SystemExit(summary())
