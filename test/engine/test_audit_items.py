"""
Regression checks for the findings of the full-codebase audits (docs/reviews/).

One function per fixed finding, named after its ID in the dated report
(`crit_01`, `high_02`, `med_17`, ...). Each rebuilds the failing case without `data/`,
runs the stage that was wrong and asserts the corrected behaviour; the docstring names
the report and the finding title. Review-loop items (P11.2.3) live in
test_review_items.py, not here.

    .venv\\Scripts\\python.exe test\\engine\\test_audit_items.py
"""
import os
import subprocess
import tempfile
from pathlib import Path

from _harness import ROOT, check, section, skip, summary

TMP = Path(tempfile.mkdtemp(prefix="ch_audit_items_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")     # tests never touch data/

SOURCE_DIRS = ("engine", "test", "bench")


# ── 2026-10-19 ────────────────────────────────────────────────────────────────

def crit_01() -> None:
    """2026-10-19 CRIT-01: `.gitignore` `data/` also ignored the source package
    `engine/data/`, so the pushed tree had no dataset/splits/quarantine modules.
    Every source file under engine/, test/ and bench/ must be free of ignore rules."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True, cwd=ROOT)
    except (OSError, subprocess.CalledProcessError):
        skip("no source file is gitignored", "git not available")
        return
    files = [p for d in SOURCE_DIRS for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts]
    rel   = [str(p.relative_to(ROOT)).replace("\\", "/") for p in files]
    proc  = subprocess.run(["git", "check-ignore", "--no-index", *rel],
                           capture_output=True, text=True, cwd=ROOT)
    ignored = [l for l in proc.stdout.splitlines() if l.strip()]
    check(f"no source file under {'/'.join(SOURCE_DIRS)} is gitignored ({len(rel)} checked)",
          not ignored, ", ".join(ignored[:5]))
    pkg = sorted(p.name for p in (ROOT / "engine" / "data").glob("*.py"))
    check("engine/data package is present with its seven modules",
          pkg == ["__init__.py", "dataset.py", "inventory.py", "layout.py", "qa.py", "quarantine.py", "splits.py"],
          str(pkg))


def _header_64(path, with_roster=False):
    return {"map": "de_mirage", "tick_rate": 64, "tick_rate_raw": 64.0, "duration_s": 2700.0,
            "n_rounds": 24, "side_counts": {"ct": 5, "t": 5}}


def _parser_must_not_run(path):
    raise AssertionError("parser ran on a demo that fails quarantine")


def high_01() -> None:
    """2026-10-19 HIGH-01: the quarantine rules ran only in `analyze`; `parse` (worker)
    parsed any demo and `build-states --all` overwrote a `quarantined` status."""
    from engine import cli, db
    from engine.parse import worker
    from engine.data import quarantine as q
    demo_dir = TMP / "h01"; demo_dir.mkdir()
    demo = demo_dir / "m64.dem"; demo.write_bytes(b"PBDEMS2\x00" + b"x" * 64)
    r = worker.parse_one(demo, TMP / "h01_parsed", _parser_must_not_run, header_fn=_header_64)
    check("worker.parse_one refuses a 64-tick demo before parsing",
          r["status"] == "quarantined" and r["reason"] == "tick_rate_not_128", str(r))
    check("nothing written for a quarantined demo", not (TMP / "h01_parsed").exists())
    db.register_match({"match_key": "m64", "demo_sha256": "a" * 64, "tier": "pro", "status": "downloaded"})
    base = TMP / "h01_quarantine"
    dest = q.quarantine("m64", demo, r["reason"], base=base)
    row = db.get_match("m64")
    check("quarantine() uses the parser's reason vocabulary and sets the row",
          dest.is_file() and row["status"] == "quarantined" and row["quarantine_reason"] == "tick_rate_not_128")
    try:
        cli._build_one("m64", None, None, None, {}, "states")
        check("_build_one refuses a quarantined match", False)
    except cli.CLIError as exc:
        check("_build_one refuses a quarantined match", "quarantined" in str(exc), str(exc))
    check("status not overwritten by the refused build", db.get_match("m64")["status"] == "quarantined")


_HEADER = ("| date | model | kind | log loss | Brier | ECE | baseline log loss | hazard AUC "
           "| decision rule satisfied | status |\n|---|---|---|---|---|---|---|---|---|---|\n")


def _models_doc(path: Path, rows: str, prose: str = "") -> Path:
    path.write_text(f"# Models\n\n{prose}\n\n## Adoption table\n\n{_HEADER}{rows}", encoding="utf-8")
    return path


def high_02() -> None:
    """2026-10-19 HIGH-02: adoption was read by scanning ENGINE_MODELS.md for `<kind>_vN`
    tokens, so a version named in prose or in another row's rule text counted as adopted."""
    from engine import analyze
    from engine.evaluation import registry as rg
    d = TMP / "h02"; d.mkdir()
    doc = _models_doc(d / "ENGINE_MODELS.md",
                      "| 2026-11-01 | wp_v7 | wp | 0.4600 | 0.1500 | 0.0120 | 0.4900 | – | ok | adopted |\n"
                      "| 2026-11-02 | q_v2 | q | – | – | – | – | – | C2 identity ok | retired |\n",
                      prose="The candidate wp_v3 and neighbours_v1 are discussed here but not adopted.")
    check("prose tokens adopt nothing; retired row excluded", analyze.adopted_models(doc) == {"wp_v7"},
          str(sorted(analyze.adopted_models(doc))))
    for name in ("wp_v7", "wp_v8"):
        (d / name).mkdir(); (d / name / "manifest.json").write_text("{}", encoding="utf-8")
        (d / name / "calibration.json").write_text("{}", encoding="utf-8")       # R6 gate (MED-20) satisfied
        (d / name / "sanity_cf.json").write_text('{"passed": true}', encoding="utf-8")   # P7 gate (MED-16)
    try:
        rg.adopt("wp_v8", {"logloss": 0.46, "brier": 0.15, "ece": 0.012, "baseline_logloss": 0.49},
                 "beats wp_v3 on pro test", doc=doc, models_dir=d)
        check("adopt refuses a rule naming another model", False)
    except ValueError as exc:
        check("adopt refuses a rule naming another model", "wp_v3" in str(exc), str(exc))
    check("the refused adopt wrote nothing", analyze.adopted_models(doc) == {"wp_v7"})


def high_03() -> None:
    """2026-10-19 HIGH-03: --q-model and --neighbours were loaded without any adoption
    check; only the win model went through check_model."""
    from engine import analyze
    d = TMP / "h03"; d.mkdir()
    doc = _models_doc(d / "ENGINE_MODELS.md",
                      "| 2026-11-01 | q_v1 | q | – | – | – | – | – | C2 ok | adopted |\n")
    for name in ("q_v1", "neighbours_v1"):
        (d / name).mkdir()
    dirs = (d / "q_v1", d / "neighbours_v1")
    try:
        analyze.check_value_models(dirs, False, [], None, doc=doc, models_dir=d)
        check("unadopted neighbours index refused", False)
    except analyze.ModelUnavailable as exc:
        check("unadopted neighbours index refused", "neighbours_v1" in str(exc) and "not adopted" in str(exc), str(exc))
    try:
        analyze.check_value_models((d / "q_v1", d / "neighbours_v9"), True, [], None, doc=doc, models_dir=d)
        check("missing value directory refused even with --allow-unadopted", False)
    except analyze.ModelUnavailable as exc:
        check("missing value directory refused even with --allow-unadopted", "neighbours_v9" in str(exc), str(exc))
    warnings: list = []
    analyze.check_value_models(dirs, True, warnings, None, doc=doc, models_dir=d)
    check("--allow-unadopted: passes with one model_unadopted warning per unadopted value model",
          [w["code"] for w in warnings] == ["model_unadopted"] and "neighbours_v1" in warnings[0]["message"], str(warnings))
    src = (ROOT / "engine" / "analyze.py").read_text(encoding="utf-8") + (ROOT / "engine" / "cli.py").read_text(encoding="utf-8")
    check("every ValueModels.load call site is preceded by check_value_models",
          src.count("ValueModels.load(") == 2 and src.count("check_value_models(value_dirs,") == 2)


def high_04() -> None:
    """2026-10-19 HIGH-04: with awpy's null fire extent, every player's on_fire was 0 on
    real demos (unknown written as 0) and fire_pressure never fired. Now: on_fire is
    None while an unknown-extent fire burns, and the enabler links through burn damage."""
    import polars as pl
    from _synthetic import PLAYERS, make_raw_tables
    from engine.state import players, utility
    from engine.state.builder import RoundView
    from engine.state.geometry import FlatGeometry
    raw = make_raw_tables("base", 128)
    a_sid, a_side = PLAYERS["A"]
    row = dict(entity_id=950, start_tick=1400, end_tick=1400 + 7 * 128, x=0.0, y=300.0, z=0.0,
               extent_radius=None, thrower_steamid=PLAYERS["C"][0])
    nof = raw.__class__(**{**raw.__dict__, "infernos": pl.DataFrame([row], schema=raw.infernos.schema)})
    known = raw.__class__(**{**raw.__dict__, "infernos": pl.DataFrame([{**row, "extent_radius": 150.0}],
                                                                       schema=raw.infernos.schema)})
    def tok(r):
        sl = RoundView.build(r, 1).until(1512)
        util, _, _ = utility.tokens(sl)
        return players.token(sl, a_sid, a_side, FlatGeometry(), "pro", util)
    check("builder: on_fire None while a fire of unknown extent burns (production shape)", tok(nof).on_fire is None)
    check("builder: on_fire 1 inside a fire of known extent", tok(known).on_fire == 1)
    check("builder: on_fire 0 with no fire", tok(raw).on_fire == 0)
    from _cf_states import make_snapshot
    snap = make_snapshot(fires=[{"x": 0.0, "y": 0.0, "z": 0.0, "owner": 203, "radius": "unknown"}])
    check("schema: an alive player with on_fire None validates",
          snap.validate() == [] and any(p.alive == 1 and p.on_fire is None for p in snap.players))


def high_05() -> None:
    """2026-10-19 HIGH-05: recompute_derived rewrote ctx.money_ct/t as an alive-only sum
    while the builder sums every roster account, so an unedited snapshot changed under
    the counterfactual editor and a zero-effect action got a nonzero Shapley value."""
    from _synthetic import STORIES, make_raw_tables
    from engine import config
    from engine.counterfactual import apply as cf_apply
    from engine.counterfactual import interventions as iv
    from engine.state import builder
    from engine.state.geometry import FlatGeometry
    cfg, geo = config.load(), FlatGeometry()
    changed, dead_seen, n = [], 0, 0
    for story in STORIES:
        raw = make_raw_tables(story, 128)
        for s in builder.build(raw, geo, cfg, match_key=f"h05-{story}", tier="pro"):
            n += 1
            dead_seen += any(p.alive == 0 and p.player_id != -1 for p in s.players)
            r = iv.recompute_derived(s.copy())
            if r != s:
                changed.append((story, s.meta.round_num, s.meta.tick, r.ctx.money_ct, s.ctx.money_ct, r.ctx.money_t, s.ctx.money_t))
            a, _ = cf_apply.apply(s, s, [], set())
            if a != s:
                changed.append((story, s.meta.round_num, s.meta.tick, "apply"))
    check(f"recompute_derived and apply(s, s, [], set()) leave every builder snapshot unchanged "
          f"({n} snapshots, {dead_seen} with a dead player)", not changed and dead_seen > 0, str(changed[:3]))
    from _cf_states import make_snapshot
    snap = make_snapshot(alive_ct=3, alive_t=4)
    check("C1 fixture: money is the roster sum, not the alive sum (builder's definition)",
          snap.ctx.money_ct == sum(800.0 + 10 * (pid % 100) for pid in range(101, 106))
          and snap.ctx.money_ct > sum(p.money for p in snap.players if p.side == "ct" and p.alive == 1))


class _EyeLevelGeometry:
    """Visible only between points above 50 u: a feet-level ray (z = 0) is always
    blocked, an eye-level one (z = 64 / 46) never. Everything else as FlatGeometry."""
    def __init__(self):
        from engine.state.geometry import FlatGeometry
        self.flat = FlatGeometry()
        self.map_version = self.flat.map_version
    def is_visible(self, a, b):
        return a[2] > 50 and b[2] > 50
    def __getattr__(self, name):
        return getattr(self.flat, name)


def high_06() -> None:
    """2026-10-19 HIGH-06: counterfactual LOS and smoke_between were cast from feet
    positions while the builder casts eye to eye; a removed smoke only revisited pairs
    whose FEET segment crossed it, so eye-crossing pairs kept a stale 1."""
    import numpy as np
    from _cf_states import action, make_snapshot
    from _synthetic import STORIES, make_raw_tables
    from engine import config
    from engine.counterfactual import interventions as iv
    from engine.state import builder
    cfg, geo = config.load(), _EyeLevelGeometry()
    diffs, n_los = [], 0
    for story in STORIES:
        raw = make_raw_tables(story, 128)
        for s in builder.build(raw, geo, cfg, match_key=f"h06-{story}", tier="pro"):
            r = s.copy()
            for i, p in enumerate(r.players):
                if p.alive == 1:
                    iv._recompute_rows(r, i, None, geo)
            n_los += int((s.pair[:, :, 0] == 1.0).sum())
            if not np.array_equal(r.pair, s.pair, equal_nan=True):
                diffs.append((story, s.meta.round_num, s.meta.tick))
    check(f"_recompute_rows with the builder's geometry reproduces every builder pair table "
          f"({n_los} visible LOS cells, all eye-level)", not diffs and n_los > 0, str(diffs[:3]))
    # a smoke the eye segment crosses but the feet segment misses (centre 64 u up, radius 50)
    post = make_snapshot(positions={101: (0.0, 0.0, 0.0), 201: (1000.0, 0.0, 0.0)},
                         smokes=[{"x": 500.0, "y": 0.0, "z": 64.0, "owner": 103, "radius": 50.0}])
    i, j, SMOKE = post.index_of(101), post.index_of(201), 3
    check("fixture: eye-to-eye segment crosses the smoke (smoke_between 1)", post.pair[i, j, SMOKE] == 1.0)
    gone = iv.remove_utility(post, action("smoke", 103, x=500.0, y=0.0, z=64.0))
    check("remove_utility recomputes that pair (no stale smoke_between after the smoke is gone)",
          gone.pair[i, j, SMOKE] == 0.0 and not (np.nanmax(gone.pair[:, :, SMOKE]) > 0.0))


def med_02() -> None:
    """2026-10-19 MED-02: an adopted model NAME was accepted from any directory."""
    from engine import analyze
    d = TMP / "m02"; (d / "models" / "wp_v3").mkdir(parents=True); (d / "copy" / "wp_v3").mkdir(parents=True)
    doc = _models_doc(d / "ENGINE_MODELS.md", "| 2026-11-01 | wp_v3 | wp | 0.4600 | 0.1500 | 0.0120 | 0.4900 | – | ok | adopted |\n")
    analyze.check_model(d / "models" / "wp_v3", doc=doc, models_dir=d / "models")
    try:
        analyze.check_model(d / "copy" / "wp_v3", doc=doc, models_dir=d / "models")
        check("same-named directory outside models/ refused", False)
    except analyze.ModelUnavailable as exc:
        check("same-named directory outside models/ refused", "not the adopted one" in str(exc), str(exc))


def med_04() -> None:
    """2026-10-19 MED-04: `or` as a numeric null-coalesce on the tick-rate gates."""
    from engine import analyze
    from engine.parse import awpy3_parser as ap
    def refused(m):
        try:
            analyze.check_tick_rate(m)
            return False
        except analyze.Refused:
            return True
    zero, nan = {"tick_rate": 128, "tick_rate_raw": 0.0}, {"tick_rate": None, "tick_rate_raw": float("nan")}
    check("check_tick_rate: a 0.0 raw rate is a value (refused), not a missing one that falls through",
          refused(zero) and ap.quarantine_check(zero) == "tick_rate_not_128")
    check("check_tick_rate: NaN raw with no rounded rate is unknown (refused), with 128 it falls back (ok)",
          refused(nan) and not refused({"tick_rate": 128, "tick_rate_raw": float("nan")}))
    check("a finite raw rate still wins over the rounded one",
          ap.quarantine_check({"tick_rate": 128, "tick_rate_raw": 127.9}) is None)


def med_06() -> None:
    """2026-10-19 MED-06: re-registering an existing match_id with a different demo
    reported success and left a copy the DB did not describe."""
    from engine import db
    db.register_match({"match_key": "m06", "demo_sha256": "1" * 64, "tier": "pro"})
    try:
        db.register_match({"match_key": "m06", "demo_sha256": "2" * 64, "tier": "pro"})
        check("register_match raises MatchConflict", False)
    except db.MatchConflict:
        check("register_match raises MatchConflict", db.get_match("m06")["demo_sha256"] == "1" * 64)
    check("same sha under the same key is still idempotent",
          db.register_match({"match_key": "m06", "demo_sha256": "1" * 64}) == "m06")


def med_07() -> None:
    """2026-10-19 MED-07: manifest.json recorded the absolute demo path and the hostname."""
    import json, socket
    from engine import manifest
    d = TMP / "m07"; d.mkdir(); demo = d / "some match.dem"; demo.write_bytes(b"PBDEMS2\x00" * 8)
    man = manifest.write(d, "parse", inputs=[demo], outputs=[])
    text = (d / manifest.MANIFEST_NAME).read_text(encoding="utf-8")
    check("input recorded by file name + sha256, no directory", man["inputs"][0]["path"] == "some match.dem"
          and str(d) not in text and "Users" not in man["inputs"][0]["path"])
    check("hostname replaced by a 16-hex machine id", socket.gethostname() not in text
          and len(man["hostname"]) == 16 and all(c in "0123456789abcdef" for c in man["hostname"]))
    check("manifest still round-trips", json.loads(text)["inputs"][0]["sha256"] == manifest.sha256_file(demo))


def med_08() -> None:
    """2026-10-19 MED-08: decoys never expired in the state (null end -> active to round end)."""
    import polars as pl
    from _synthetic import PLAYERS, make_raw_tables
    from engine.state import utility
    from engine.state.builder import RoundView
    raw = make_raw_tables("base", 128)
    row = dict(entity_id=777, thrower_steamid=PLAYERS["C"][0], type="decoy", throw_tick=1300, detonate_tick=1400,
               expire_tick=None, x=50.0, y=50.0, z=0.0)
    gren = pl.concat([raw.grenades, pl.DataFrame([row], schema=raw.grenades.schema)])
    r2 = raw.__class__(**{**raw.__dict__, "grenades": gren})
    v = RoundView.build(r2, 1)
    life = int(utility.NOMINAL_DURATION_S["decoy"] * 128)
    check("builder gives a decoy with no end its nominal life", v.decoys[-1]["end_tick"] == 1400 + life)
    alive_at = lambda t: any(u.kind == "decoy" for u in utility.tokens(v.until(t))[0])  # noqa: E731
    check("decoy token present during its life, gone after it (not forever)",
          alive_at(1500) and not alive_at(1400 + life + 1))


def med_09() -> None:
    """2026-10-19 MED-09: kills_round counted self-kills and kills with unknown team_kill."""
    import polars as pl
    from _synthetic import PLAYERS, make_raw_tables
    from engine.state import players, utility
    from engine.state.builder import RoundView
    from engine.state.geometry import FlatGeometry
    raw = make_raw_tables("base", 128)
    a_sid, a_side = PLAYERS["A"]
    k = raw.kills.row(0, named=True)
    self_kill = {**k, "tick": 1600, "attacker_steamid": a_sid, "victim_steamid": a_sid, "team_kill": False}
    unknown   = {**k, "tick": 1601, "attacker_steamid": a_sid, "victim_steamid": PLAYERS["C"][0], "team_kill": None}
    kills = pl.concat([raw.kills, pl.DataFrame([self_kill, unknown], schema=raw.kills.schema)]).sort("tick")
    r2 = raw.__class__(**{**raw.__dict__, "kills": kills})
    sl = RoundView.build(r2, 1).until(1700)
    tok = players.token(sl, a_sid, a_side, FlatGeometry(), "pro", utility.tokens(sl)[0])
    base_sl = RoundView.build(raw, 1).until(1700)
    base_tok = players.token(base_sl, a_sid, a_side, FlatGeometry(), "pro", utility.tokens(base_sl)[0])
    check("a self-kill and a kill with unknown team_kill do not raise kills_round",
          tok.kills_round == base_tok.kills_round, f"{tok.kills_round} vs {base_tok.kills_round}")


def med_10() -> None:
    """2026-10-19 MED-10: knife skin display names mapped to weapon class unknown."""
    from engine.state.players import weapon_class
    names = ("Karambit", "Butterfly Knife", "M9 Bayonet", "Shadow Daggers", "weapon_knife_karambit", "Kukri Knife")
    check("knife skins map to class knife", all(weapon_class(n) == "knife" for n in names),
          str({n: weapon_class(n) for n in names}))
    check("a rifle is still a rifle", weapon_class("AK-47") == "rifle")


def med_14() -> None:
    """2026-10-19 MED-14: a second same-tick smoke's pop/fade pointed at the first parent."""
    import polars as pl
    from _synthetic import make_raw_tables
    from engine import config
    from engine.actions import extractor
    from engine.state.geometry import FlatGeometry
    raw = make_raw_tables("base", 128)
    s0 = raw.smokes.row(0, named=True)
    twin = {**s0, "entity_id": int(s0["entity_id"]) + 1, "x": float(s0["x"]) + 300.0}
    smokes = pl.concat([raw.smokes, pl.DataFrame([twin], schema=raw.smokes.schema)])
    r2 = raw.__class__(**{**raw.__dict__, "smokes": smokes})
    _, acts, _ = extractor.run(r2, config.load(), match_key="m14", geometry=FlatGeometry())
    parents = [a for a in acts if a.type == "smoke"]
    pops = [a for a in acts if a.type == "smoke_pop"]
    check("two smoke parents with distinct ids", len(parents) == 2 and parents[0].action_id != parents[1].action_id)
    check("each pop points at its own parent", sorted(p.params["parent_action_id"] for p in pops)
          == sorted(a.action_id for a in parents), str([p.params["parent_action_id"] for p in pops]))
    check("no stray _parent attribute survives", all(not hasattr(a, "_parent") for a in acts))


def med_17() -> None:
    """2026-10-19 MED-17: the Q model, the neighbour index and the WP encoder recorded no
    link to each other; ValueModels.load now refuses an index built with another vocab."""
    import json
    from engine import analyze, manifest
    d = TMP / "m17"; q = d / "q_v1"; n = d / "neighbours_v1"
    q.mkdir(parents=True); n.mkdir(parents=True)
    (q / "vocab.json").write_text(json.dumps({"fields": [], "unk": 0, "pad": 1, "maps": {}}), encoding="utf-8")
    manifest.write(q, "q_model", outputs=["vocab.json"], extra={"wp_model": "wp_v1"})
    manifest.write(n, "neighbours", outputs=[], extra={"q_model": "q_v0", "q_vocab_sha256": "0" * 64, "wp_model": "wp_v1"})
    try:
        analyze.ValueModels.load(q, n)
        check("index built with another Q vocab refused", False)
    except analyze.ModelUnavailable as exc:
        check("index built with another Q vocab refused", "descriptor vocab" in str(exc) and "q_v0" in str(exc), str(exc))
    class VM:
        versions = {"q": "q_v1", "neighbours": "neighbours_v1", "q_wp_model": "wp_v1",
                    "neighbours_wp_model": "wp_v2", "provenance": "checked"}
    w: list = []
    analyze.warn_value_provenance(VM(), "wp_v3", w)
    check("one value_provenance warning per artifact built from another WP model",
          [x["code"] for x in w] == ["value_provenance", "value_provenance"] and "wp_v2" in w[1]["message"], str(w))


def med_18_19() -> None:
    """2026-10-19 MED-18 / MED-19: shards never checked their vocab / norm; an empty or
    missing split crashed after a full epoch."""
    import json
    import numpy as np
    from engine.data import dataset
    d = TMP / "m18"; (d / "train").mkdir(parents=True)
    vocab = d / "vocab_v1.json"; vocab.write_text("{}", encoding="utf-8")
    other = d / "vocab_v2.json"; other.write_text("{ }", encoding="utf-8")
    from engine.manifest import sha256_file
    (d / "train" / dataset.SHARD_INDEX).write_text(json.dumps({"split": "train", "n": 0, "shards": [], "counts": [],
        "layout_hash": dataset.layout_hash(), "keys": list(dataset.BATCH_KEYS),
        "extra": {"vocab": vocab.name, "vocab_sha256": sha256_file(vocab)}}), encoding="utf-8")
    dataset.check_artifacts("train", vocab, base=d)
    try:
        dataset.check_artifacts("train", other, base=d)
        check("a vocab other than the one the shards were built with is refused", False)
    except ValueError as exc:
        check("a vocab other than the one the shards were built with is refused", "vocab_v1.json" in str(exc), str(exc))
    try:
        dataset.read_shards("no_such_split", base=d)
        check("a missing split raises instead of reading as zero rows", False)
    except FileNotFoundError:
        check("a missing split raises instead of reading as zero rows", True)
    arrays = dataset.empty_arrays(0)
    check("n_labelled of an empty split is 0", dataset.n_labelled(arrays) == 0 and dataset.n_labelled(arrays, ("pro",)) == 0)
    check("n_labelled counts win labels", dataset.n_labelled({"y_win": np.array([0.0, np.nan, 1.0]),
                                                              "tier": np.array(["pro", "pro", "faceit_10"])}, ("pro",)) == 1)


def med_20() -> None:
    """2026-10-19 MED-20: the Predictor served a plain sigmoid when calibration.json was
    missing; adoption did not require it; calibration never named its checkpoint."""
    import inspect
    from engine import analyze
    from engine.evaluation import registry as rg
    from engine.model import predict
    check("Predictor has require_calibration and analyze.load_predictor passes it",
          "require_calibration" in inspect.signature(predict.Predictor.__init__).parameters
          and "require_calibration=True" in inspect.getsource(analyze.load_predictor))
    d = TMP / "m20"; (d / "wp_v5").mkdir(parents=True)
    (d / "wp_v5" / "manifest.json").write_text("{}", encoding="utf-8")
    doc = _models_doc(d / "ENGINE_MODELS.md", "")
    try:
        rg.adopt("wp_v5", {"logloss": 0.46, "brier": 0.15, "ece": 0.012, "baseline_logloss": 0.49}, "ok",
                 doc=doc, models_dir=d)
        check("adopting a wp model without calibration.json refused", False)
    except ValueError as exc:
        check("adopting a wp model without calibration.json refused", "calibration.json" in str(exc), str(exc))


# ── runner ────────────────────────────────────────────────────────────────────

ITEMS = [crit_01, high_01, high_02, high_03, high_04, high_05, high_06, med_02, med_04, med_06,
         med_07, med_08, med_09, med_10, med_14, med_17, med_18_19, med_20]

section("audit items (docs/reviews)")
for fn in ITEMS:
    fn()

raise SystemExit(summary())
