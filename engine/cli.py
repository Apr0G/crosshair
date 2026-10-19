"""
Engine command line: `python -m engine.cli <command>` or the `crosshair` console script.

Commands (P10.1, Amendment 1): analyze | parse | build-states | build-actions |
build-dataset | train-wp | train-q | calibrate | build-neighbours | eval | sanity |
review | bench | inventory | status | scrape | ingest. `docs/ENGINE_CLI.md` documents each one.

Common to every command (P10.1.6): prints `run_id`, `config_hash`, device and stage
timings through the redacting run logger (`engine.log`); writes a manifest (analyze:
`<out>/manifest.json`; others: `<log-dir>/<run_id>/manifest.json`); `--help`
everywhere; startup free-space check (P0.3.2: warn below 100 GB, refuse batch parsing
and scraping below 20 GB).

Exit codes: 0 ok; 1 the command ran and failed; 2 usage error or missing prerequisite
(missing file, bad argument, no model, unfinished module, refused demo, no disk
space). Nothing else (LOW-04): Ctrl-C exits 1 ("interrupted"), and a delegate or bench
script returning any other code exits 1. Errors print one redacted line, no traceback
(`--debug` adds the redacted traceback to the console).

Commands whose implementation lives in another module (`train-wp`, `calibrate`,
`eval`, `sanity`, `review`, `build-dataset`) import it lazily and hand it every
argument this parser did not consume; a module that is missing, has no `main`, or
raises NotImplementedError exits 2 with a message. Use `--` to reach the module's own
help: `python -m engine.cli train-wp -- --help`. `train-q` and `build-neighbours` have
no module `main()`: small adapters here call `q_model.train_q` / `neighbours.build_index`
and exit 2 when their inputs (valued-actions parquet with `h_pre`, model dirs) are absent.

Test hooks of `analyze` (never needed for real use): `--raw-tables-story STORY`
replaces parsing with a synthetic story from `test/engine/_synthetic.py`, and the
environment variable `CROSSHAIR_STUB_PREDICTOR=1` replaces the model with a rule stub.

Filled by: P10.1.
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from engine import paths

# ── command table ─────────────────────────────────────────────────────────────

COMMANDS = ("analyze", "parse", "build-states", "build-actions", "build-dataset", "train-wp", "train-q",
            "calibrate", "build-neighbours", "eval", "sanity", "review", "bench", "inventory", "status",
            "scrape", "ingest")

# command -> (module, help line); main(argv) of the module receives the extra arguments
DELEGATES = {
    "build-dataset":    ("engine.data.dataset",         "tensor shards per split (P5.3); --split FILE"),
    "train-wp":         ("engine.model.train",          "train the win-probability model (P6.3)"),
    "calibrate":        ("engine.model.calibrate",      "temperature calibration of a WP model (P6.5)"),
    "review":           ("engine.evaluation.review",    "analyst review loop: review set, log, disagree items (P11.2)"),
}
# target -> (module, flag that carries --model; None = the module takes no model)
EVAL_TARGETS = {
    "calibration": ("engine.model.calibrate",        "--model-dir"),
    "value":       ("engine.value.q_model",          "--model-dir"),
    "attribution": ("engine.attribution.evaluate",   "--model-dir"),
    "anchors":     ("engine.evaluation.anchors",     "--model-dir"),
    "robustness":  ("engine.evaluation.robustness",  None),
}
SANITY_TARGETS = {
    "model":          ("engine.model.sanity",          "--model"),
    "counterfactual": ("engine.counterfactual.sanity", "--model"),
}
BENCH_SCRIPTS = ("analyze", "parse", "states", "actions", "predict", "attribution", "value", "gpu")
ANALYZE_TIERS = ("pro", "faceit_10", "faceit_8_9", "other")       # engine.state.builder.TIERS
INGEST_TIERS  = ("pro", "faceit_10", "golden")                    # engine.ingest.TIER_DIRS
SOURCES       = ("api", "playwright", "hltv", "hf")
BATCH         = ("parse", "scrape")          # refused below the free-space floor
STORIES       = ("base", "flash", "double_kill_same_tick", "trade", "retreat", "drop")


class CLIError(Exception):
    """One-line failure with an exit code (2 = usage / prerequisite, 1 = failed)."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


# ── output ────────────────────────────────────────────────────────────────────

def say(text: str) -> None:
    """stdout line, redacted (results meant for the user or a pipe)."""
    from engine.log import redact_text
    print(redact_text(text), flush=True)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(paths.ROOT))
    except ValueError:
        return str(p)


# ── parser ────────────────────────────────────────────────────────────────────

def _common() -> argparse.ArgumentParser:
    c = argparse.ArgumentParser(add_help=False)
    g = c.add_argument_group("common options")
    g.add_argument("--config", type=Path, default=None, help="engine config YAML (default engine/config.yaml)")
    g.add_argument("--log-dir", type=Path, default=None, help="run log + manifest directory (default data/engine/logs)")
    g.add_argument("--debug", action="store_true", help="print the traceback of an unexpected error")
    return c


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    ap = argparse.ArgumentParser(prog="python -m engine.cli",
                                 description="Crosshair engine v2: demo -> win probability -> per-action value.",
                                 epilog="docs/ENGINE_CLI.md documents every command, its outputs and exit codes.")
    sub = ap.add_subparsers(dest="command", metavar="<command>", required=True)
    ap.commands = sub.choices                      # name -> subparser (for error messages)

    def add(name: str, help_: str, **kw) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_, description=help_, parents=[common], **kw)

    p = add("analyze", "demo -> report (parse, states, actions, predict, attribute, value, report)")
    p.add_argument("demo", type=Path, help="the .dem file (with --raw-tables-story: only its name is used)")
    p.add_argument("--tier", choices=ANALYZE_TIERS, default="pro", help="tier token for the model (default pro)")
    p.add_argument("--out", type=Path, default=None, help="output directory (default data/engine/reports/<match>)")
    p.add_argument("--model", default=None, help="win model name under data/engine/models, e.g. wp_v1")
    p.add_argument("--model-dir", type=Path, default=None, help="win model directory (instead of --model)")
    p.add_argument("--q-model", default=None, help="Q/B model name (value stage), e.g. q_v1")
    p.add_argument("--neighbours", default=None, help="neighbour index name (value stage), e.g. neighbours_v1")
    p.add_argument("--cpu", action="store_true", help="run the model on the CPU")
    p.add_argument("--allow-unadopted", action="store_true", help="use a model not adopted in docs/ENGINE_MODELS.md")
    p.add_argument("--show-ids", action="store_true", help="show SteamIDs and names in the report (default hashed)")
    p.add_argument("--no-db", action="store_true", help="do not write action_scores rows")
    p.add_argument("--raw-tables-story", choices=STORIES, default=None,
                   help="TEST ONLY: analyse a synthetic story instead of parsing the demo")
    p.add_argument("--raw-tables-tick-rate", type=int, default=128, help="TEST ONLY: tick rate of the story")

    p = add("parse", "parse a demo or a directory of demos (P2.6 worker)")
    p.add_argument("target", type=Path, help="a .dem file or a directory of them")
    p.add_argument("--workers", type=int, default=None, help="processes (default cpu_count - 2)")
    p.add_argument("--force", action="store_true", help="re-parse matches already parsed")
    p.add_argument("--smoke", action="store_true", help="header + quarantine check only; writes nothing")
    p.add_argument("--out", type=Path, default=None, help="output root (default data/engine/parsed)")

    for name, what in (("build-states", "states.parquet"), ("build-actions", "actions + windows parquet")):
        p = add(name, f"parsed match -> {what} in the P5.1 layout")
        p.add_argument("match_key", nargs="?", default=None, help="a parsed match (data/engine/parsed/<key>)")
        p.add_argument("--all", action="store_true", help="every parsed match")
        p.add_argument("--tier", choices=ANALYZE_TIERS, default=None, help="tier when the DB has none")
        p.add_argument("--month", default=None, help="yyyy-mm when the DB has no date for the match")

    for name, (module, help_) in DELEGATES.items():
        add(name, f"{help_} [delegates to {module}.main; extra arguments are passed through]")

    p = add("train-q", "train the Q / B value networks on valued actions (P9.2; calls q_model.train_q)")
    p.add_argument("--valued", type=Path, default=None,
                   help="actions_valued parquet with h_pre, V, actor_side, tier, type and desc_* columns")
    p.add_argument("--split-file", type=Path, default=None, help="splits_vN.json (train / val by match_key)")
    p.add_argument("--out", type=Path, default=None, help="model directory (default data/engine/models/q_v<N+1>)")
    p.add_argument("--wp-model", default=None, help="WP model whose encoder filled h_pre (recorded in the manifest)")
    p.add_argument("--cpu", action="store_true", help="train on the CPU")

    p = add("build-neighbours", "FAISS neighbour index over pro train actions (P9.3; calls neighbours.build_index)")
    p.add_argument("--valued", type=Path, default=None,
                   help="actions_valued parquet with h_pre, V, actor_side, map_version, tier and desc_* columns")
    p.add_argument("--q-model", default=None, help="Q model name or directory (its descriptor vocab gives desc_ids)")
    p.add_argument("--split-file", type=Path, default=None, help="splits_vN.json: keep only train matches")
    p.add_argument("--tier", default="pro", help="keep rows of this tier (default pro)")
    p.add_argument("--wp-model", default=None, help="WP model whose encoder filled h_pre (recorded in the manifest)")
    p.add_argument("--out", type=Path, default=None, help="index directory (default data/engine/models/neighbours_v<N+1>)")

    p = add("eval", "evaluation reports (delegates per target)")
    p.add_argument("--model", required=False, default=None, help="model name or directory")
    for t in EVAL_TARGETS:
        p.add_argument(f"--{t}", action="store_true", help=f"run the {t} evaluation ({EVAL_TARGETS[t][0]})")
    p.add_argument("--all", action="store_true", help="every evaluation")

    p = add("sanity", "model sanity gates (P6.6, P7.2)")
    p.add_argument("--model", default=None, help="model name or directory")
    p.add_argument("--counterfactual", action="store_true", help="the P7.2 counterfactual gate")

    p = add("bench", "timing scripts under bench/")
    p.add_argument("which", nargs="?", choices=BENCH_SCRIPTS, default="analyze", help="which bench (default analyze)")

    p = add("inventory", "match counts, disk use, failures (P1.3)")
    p.add_argument("--assert-targets", action="store_true", help="exit 1 when a data target is missed")
    p.add_argument("--rebuild-catalog", action="store_true", help="rebuild data/engine/catalog.parquet first")
    p.add_argument("--models", action="store_true", help="list model directories and their manifests")

    add("status", "engine DB counts and free space")

    p = add("scrape", "discover + download + register demos from a source")
    p.add_argument("--source", choices=SOURCES, default="api", help="demo source (default api)")
    p.add_argument("--max", type=int, default=None, help="stop after this many matches")
    p.add_argument("--tier", choices=INGEST_TIERS, default=None, help="override the source's tier")
    p.add_argument("--min-level", type=int, default=None, help="FACEIT API: minimum skill level")
    p.add_argument("--min-elo", type=int, default=None, help="FACEIT: minimum elo of discovered players")
    p.add_argument("--region", default=None, help="FACEIT region (default EU)")
    p.add_argument("--repo", default=None, help="Hugging Face: archive repo 'owner/name' (required with --source hf)")

    p = add("ingest", "register one local demo (the seam: copy + matches row)")
    p.add_argument("demo", type=Path, help="the .dem file")
    p.add_argument("--match-id", default=None, help="match id (default: file name stem)")
    p.add_argument("--tier", choices=INGEST_TIERS, default="faceit_10", help="tier directory (default faceit_10)")
    p.add_argument("--source", default="local", help="source name stored in matches (default local)")
    return ap


# ── delegation ────────────────────────────────────────────────────────────────

def delegate(module: str, argv: list[str], lg) -> int:
    """Import `module` lazily and run its `main(argv)`; exit 2 when it does not exist yet."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and (module == exc.name or module.startswith(exc.name + ".")):
            raise CLIError(f"{module} does not exist yet; this command is not available") from None
        raise CLIError(f"{module} cannot be imported: missing dependency {exc.name}") from None
    except NotImplementedError as exc:
        raise CLIError(f"{module} is not implemented yet: {exc}") from None
    main_fn = getattr(mod, "main", None)
    if not callable(main_fn):
        raise CLIError(f"{module} has no main(); this command is not available yet")
    lg.info("delegating", module=module)
    try:
        rc = main_fn(list(argv))
    except NotImplementedError as exc:
        raise CLIError(f"{module} is not implemented yet: {exc}") from None
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
    return 0 if rc is None else int(rc)


def _passthrough(extra: list[str]) -> list[str]:
    return extra[1:] if extra[:1] == ["--"] else list(extra)


def _model_arg(model: str | None) -> str | None:
    if model is None:
        return None
    p = Path(model)
    return str(p if p.exists() or p.is_absolute() else paths.MODELS_DIR / model)


# ── commands ──────────────────────────────────────────────────────────────────

def _test_hook_guard(a, lg, write_db: bool) -> tuple[bool, list[dict]]:
    """MED-03 (2026-10-19): a test hook (CROSSHAIR_STUB_PREDICTOR, --raw-tables-story) may
    never write under the production report dir or into the production engine.db.
    Returns (write_db, warnings): `--out` is required; `action_scores` are written only
    when CROSSHAIR_ENGINE_DB points the DB elsewhere; the report carries a `test_hook`
    warning so stub / synthetic numbers are never mistaken for real ones."""
    from engine import db
    if a.out is None:
        raise CLIError("TEST HOOK active: pass --out; a stub or synthetic analysis never writes under "
                       "data/engine/reports")
    warnings = [{"code": "test_hook", "message": "stub predictor and/or synthetic tables: numbers are not "
                                                 "from a model on this demo"}]
    if write_db and os.environ.get(db.ENV_DB) is None:
        lg.warning("TEST HOOK: CROSSHAIR_ENGINE_DB is not set; action_scores are NOT written to the production DB")
        write_db = False
    return write_db, warnings


def cmd_analyze(a, cfg, lg, run_id: str, timings: dict) -> int:
    from engine import analyze
    from engine.config import config_hash
    stub = analyze.stub_enabled()
    if (a.q_model is None) != (a.neighbours is None):
        raise CLIError("the value stage needs both --q-model and --neighbours")
    hook, write_db, hook_warnings = stub or bool(a.raw_tables_story), not a.no_db, []
    value_dirs = None
    if a.q_model:
        value_dirs = (paths.MODELS_DIR / a.q_model, paths.MODELS_DIR / a.neighbours)
    device = "cpu" if a.cpu else None
    lg.info("analyze", run_id=run_id, config_hash=config_hash(cfg), device=device or "auto")
    predictor = None
    if stub:
        predictor = analyze.stub_predictor()
        lg.warning("TEST HOOK: CROSSHAIR_STUB_PREDICTOR is set; using a rule stub instead of the model")

    if a.raw_tables_story:
        from engine.state.geometry import FlatGeometry
        lg.warning(f"TEST HOOK: --raw-tables-story {a.raw_tables_story}; the demo file is not read")
        raw = analyze.synthetic_raw(a.raw_tables_story, a.raw_tables_tick_rate)
        analyze.check_tick_rate(raw.meta)
        model_versions = {"wp": "stub" if stub else None}
        warnings = []
        if predictor is None:
            md = analyze.resolve_model_dir(a.model, a.model_dir)
            analyze.check_model(md, a.allow_unadopted)
            predictor = analyze.load_predictor(md, device)
            model_versions["wp"] = md.name
            cfg = analyze.apply_hazard_rule(cfg, md, warnings, lg)
        geo = FlatGeometry()
        warnings += analyze.guard_warnings(raw.meta, a.tier, geo.map_version, lg)
        key = a.demo.stem
        vm = None
        if value_dirs:
            analyze.check_value_models(value_dirs, a.allow_unadopted, warnings, lg)
            vm = analyze.ValueModels.load(*value_dirs, device or "cpu")
            analyze.warn_value_provenance(vm, model_versions.get("wp"), warnings, lg)
        if hook:                                   # after the model gates, before anything is written
            write_db, hook_warnings = _test_hook_guard(a, lg, write_db)
            warnings = hook_warnings + warnings
        res = analyze.analyze_tables(raw, match_key=key, tier=a.tier, geometry=geo, predictor=predictor, cfg=cfg,
                                     value_models=vm, out_dir=a.out or paths.REPORTS_DIR / key, run_id=run_id,
                                     show_ids=a.show_ids, model_versions=model_versions, warnings=warnings,
                                     write_db=write_db, source_tier_=analyze.source_tier(raw.meta), logger=lg)
    else:
        if not a.demo.is_file():
            raise CLIError(f"demo not found: {a.demo.name}")
        model_dir = None
        if predictor is None:
            model_dir = analyze.resolve_model_dir(a.model, a.model_dir)
        if hook:
            write_db, hook_warnings = _test_hook_guard(a, lg, write_db)
        res = analyze.analyze_demo(a.demo, tier=a.tier, out_dir=a.out or paths.REPORTS_DIR / a.demo.stem,
                                   model_dir=model_dir, device=device, cfg=cfg,
                                   allow_unadopted=a.allow_unadopted, show_ids=a.show_ids, predictor=predictor,
                                   value_dirs=value_dirs, write_db=write_db, run_id=run_id, logger=lg,
                                   warnings=hook_warnings)
    timings.update(res.timings)
    dev = getattr(predictor, "device", None) or ("stub" if stub else device or "auto")
    resid = max((abs(r["conservation_residual"]) for r in res.report["rounds"]), default=0.0)
    lg.info("analyzed", run_id=res.run_id, device=dev, rounds=len(res.report["rounds"]),
            actions=sum(len(r["actions"]) for r in res.report["rounds"]), value_stage=res.value_stage,
            max_residual=f"{resid:.1e}", warnings=len(res.warnings), db_rows=res.n_db_rows)
    say(f"report: {_rel(res.files['report.json'])}")
    say(f"markdown: {_rel(res.files['report.md'])}")
    return 0


def cmd_parse(a, cfg, lg, run_id: str, timings: dict) -> int:
    from engine.parse import worker
    t = a.target
    if not t.exists():
        raise CLIError(f"not found: {t.name}")
    out = a.out or paths.PARSED_DIR
    t0 = time.perf_counter()
    if t.is_dir():
        res = worker.run(t, out_dir=out, procs=a.workers, force=a.force, smoke=a.smoke)
        results, counts = res["results"], res["counts"]
    else:
        r = worker.smoke_one(t) if a.smoke else worker.parse_one(t, out, force=a.force)
        results, counts = [r], {r["status"]: 1}
    timings["parse"] = time.perf_counter() - t0
    if not a.smoke:
        _quarantine_results(results, lg)
    for r in results:
        say(json.dumps(r, ensure_ascii=False, default=str))
    lg.info("parse done", **{k: v for k, v in sorted(counts.items())})
    return 1 if counts.get("failed") else 0


def _quarantine_results(results: list[dict], lg) -> None:
    """P1.4.1 follow-through for `parse` (not `--smoke`): a demo the worker reported as
    `quarantined` is moved under `demos/quarantine/<reason>/` and its `matches` row set
    to `quarantined`, when it has one. Without a DB row the file stays and the result
    says so (`moved: False`)."""
    from engine import db
    from engine.data import quarantine
    for r in results:
        if r.get("status") != "quarantined":
            continue
        row = db.get_match(r["match_key"]) if db.db_path().is_file() else None
        if row is None:
            r["moved"] = False
            continue
        try:
            dest = quarantine.quarantine(r["match_key"], Path(r["path"]), r["reason"])
            r["moved"], r["path"] = True, str(dest)
        except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
            r["moved"], r["error"] = False, f"{type(exc).__name__}: {exc}"
            lg.error("quarantine move failed", match_key=r["match_key"], reason=r["reason"],
                     error=r["error"])


def _parsed_raw(key: str):
    """RawTables back from data/engine/parsed/<key>/ (worker output) + its parse meta."""
    import polars as pl
    from engine import manifest
    from engine.parse.schema import TABLES, RawTables
    d = paths.PARSED_DIR / key
    if not (d / manifest.MANIFEST_NAME).is_file():
        raise CLIError(f"match {key} is not parsed (run `parse` first)")
    meta = manifest.read(d)["extra"].get("meta") or {}
    tables = {t: pl.read_parquet(d / f"{t}.parquet") for t in TABLES if (d / f"{t}.parquet").is_file()}
    return RawTables.from_parsed(tables, meta)


def _build_one(key: str, a, cfg, lg, timings: dict, write: str) -> None:
    from engine import analyze, db
    from engine.actions import extractor
    from engine.config import config_hash
    from engine.data import layout
    from engine.maps import assets, callouts, version
    from engine.state import builder
    from engine.state.geometry import MapGeometry
    from engine.state.schema import SCHEMA_VERSION
    row = db.get_match(key) if db.db_path().is_file() else None
    if row is not None and row.get("status") == "quarantined":
        raise CLIError(f"{key} is quarantined ({row.get('quarantine_reason')}); not building {write}")
    raw = _parsed_raw(key)
    tier = (row or {}).get("tier") or a.tier
    if tier not in ANALYZE_TIERS:
        raise CLIError(f"{key}: tier unknown ({tier!r}); pass --tier")
    month = ((row or {}).get("date") or "")[:7] or a.month
    if not month:
        raise CLIError(f"{key}: no match date in the DB; pass --month yyyy-mm")
    map_name = raw.meta.get("map")
    try:
        cos = callouts.load(map_name)
    except (FileNotFoundError, ValueError):
        cos = []
    mv = version.detect({"build_num": raw.meta.get("game_version")}, map_name, logger=lg)
    geo = MapGeometry.from_assets(assets.ensure(map_name), cos, mv)
    snaps, actions, windows = analyze.build_states_actions(raw, cfg, match_key=key, tier=tier, geometry=geo,
                                                           timings=timings)
    h = config_hash(cfg)
    if write == "states":
        p = builder.write(snaps, layout.states_path(mv, month, key))
        layout.write_dir_manifest(p.parent, schema_version=SCHEMA_VERSION, config_hash=h)
    else:
        pa, pw = layout.actions_path(mv, month, key), layout.windows_path(mv, month, key)
        pa.parent.mkdir(parents=True, exist_ok=True)
        pw.parent.mkdir(parents=True, exist_ok=True)
        extractor.actions_frame(actions).write_parquet(pa)
        extractor.windows_frame(windows).write_parquet(pw)
        layout.write_dir_manifest(pa.parent, schema_version=SCHEMA_VERSION, config_hash=h)
        layout.write_dir_manifest(pw.parent, schema_version=SCHEMA_VERSION, config_hash=h)
    if row is not None:
        db.set_status(key, write)
    lg.info(f"built {write}", match_key=key, snapshots=len(snaps), actions=len(actions), windows=len(windows))


def _cmd_build(write: str):
    def run(a, cfg, lg, run_id: str, timings: dict) -> int:
        from engine import manifest
        if a.all == (a.match_key is not None):
            raise CLIError("give exactly one of <match_key> or --all")
        keys = [a.match_key] if a.match_key else sorted(
            d.name for d in paths.PARSED_DIR.glob("*") if (d / manifest.MANIFEST_NAME).is_file())
        failed = 0
        for key in keys:
            try:
                _build_one(key, a, cfg, lg, timings, write)
            except CLIError as exc:
                if not a.all:
                    raise
                lg.error(f"{key}: {exc}")              # LOW-07: say why, not just how many
                failed += 1
            except Exception as exc:
                if not a.all:
                    raise
                lg.error(f"{key}: {type(exc).__name__}: {exc}")
                failed += 1
        lg.info(f"build-{write} done", matches=len(keys), failed=failed)
        return 1 if failed else 0
    return run


def _cmd_delegate(name: str):
    def run(a, cfg, lg, run_id: str, timings: dict) -> int:
        extra = _passthrough(a._extra)
        if name == "build-dataset":
            extra = ["--split-file" if x == "--split" else x for x in extra]
        t0 = time.perf_counter()
        rc = delegate(DELEGATES[name][0], extra, lg)
        timings[name] = time.perf_counter() - t0
        return rc
    return run


def _valued_frame(path: Path | None):
    """The valued-actions frame + its h_pre matrix; CLIError(2) when either is absent."""
    import polars as pl
    if path is None:
        raise CLIError("needs --valued <actions_valued.parquet> (valued actions with h_pre embeddings)")
    if not Path(path).is_file():
        raise CLIError(f"valued-actions parquet not found: {Path(path).name}")
    frame = pl.read_parquet(path)
    if frame.height == 0:
        raise CLIError(f"{Path(path).name} has no rows")
    if "h_pre" not in frame.columns or frame["h_pre"].null_count() == frame.height:
        raise CLIError(f"{Path(path).name} has no h_pre embeddings (fill them from the adopted win model first)")
    from engine.value import q_model
    try:
        h = q_model.h_matrix(frame)
    except ValueError as exc:
        raise CLIError(f"h_pre unusable: {exc}") from None
    return frame, h


def _split_of(split_file: Path | None):
    """match_key -> split name from a splits file (None: every row is train)."""
    if split_file is None:
        return None
    if not Path(split_file).is_file():
        raise CLIError(f"split file not found: {Path(split_file).name}")
    from engine.data import splits
    doc = splits.load(split_file)
    return {k: name for name, keys in (doc.get("splits") or {}).items() for k in keys}


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def cmd_train_q(a, cfg, lg, run_id: str, timings: dict) -> int:
    """Adapter: `q_model.train_q(frame, h, vocab, cfg, split_fn, device, out_dir=...)`."""
    if a.split_file is None:
        raise CLIError("train-q needs --split-file splits_v<N>.json: without it held-out months would be "
                       "trained on silently (R4, MED-38)")
    frame, h = _valued_frame(a.valued)
    from engine.value import descriptors, q_model
    missing = [c for c in ("V", "actor_side", "tier", "type", *(f"desc_{f}" for f in descriptors.FIELDS))
               if c not in frame.columns]
    if missing:
        raise CLIError(f"{a.valued.name} lacks columns {missing}")
    which = _split_of(a.split_file)
    if which is None:
        def split_fn(f):
            return ["train"] * f.height
    else:
        def split_fn(f):
            return [which.get(k, "none") for k in f["match_key"].to_list()]
    import polars as pl
    train_mask = pl.Series([s == "train" for s in split_fn(frame)], dtype=pl.Boolean)
    if not train_mask.any():
        raise CLIError("no train rows: the split file names none of the valued frame's matches")
    vocab = descriptors.DescriptorVocab.fit_frame(frame.filter(train_mask))      # train rows only (MED-38)
    t0 = time.perf_counter()
    try:
        _q, _b, metrics = q_model.train_q(frame, h, vocab, cfg, split_fn, "cpu" if a.cpu else _device(),
                                          out_dir=a.out, wp_model=a.wp_model)
    except NotImplementedError as exc:
        raise CLIError(f"train_q not available: {exc}") from None
    timings["train_q"] = time.perf_counter() - t0
    lg.info("trained Q/B", rows=frame.height,
            **{k: v for k, v in (metrics or {}).items() if isinstance(v, (int, float, str))})
    if metrics and metrics.get("dir"):                   # P9.4.2: shrinkage fitted on the train rows (MED-34)
        import polars as pl
        from engine.value import shrinkage
        is_train = pl.Series([s == "train" for s in split_fn(frame)], dtype=pl.Boolean)
        fit = shrinkage.fit_from_models(frame.filter(is_train), h[is_train.to_numpy()], _q, _b, vocab)
        p = shrinkage.save(fit, Path(metrics["dir"]) / shrinkage.FILE_NAME)
        lg.info("shrinkage fitted", path=_rel(p), classes={v: len(fit.get(v, {})) for v in shrinkage.VALUES})
    return 0


def cmd_build_neighbours(a, cfg, lg, run_id: str, timings: dict) -> int:
    """Adapter: `neighbours.build_index(h, side_table)` + `save_index`."""
    import polars as pl
    if a.split_file is None:
        raise CLIError("build-neighbours needs --split-file splits_v<N>.json: the panel must hold train matches "
                       "only (R4, MED-38)")
    frame, h = _valued_frame(a.valued)
    if a.q_model is None:
        raise CLIError("needs --q-model (its descriptor vocab gives the desc_ids the regret uses)")
    q_dir = Path(_model_arg(a.q_model))
    if not q_dir.is_dir():
        raise CLIError(f"Q model directory not found: {q_dir.name} (run train-q first)")
    vpath = q_dir / "vocab.json"
    if not vpath.is_file():
        raise CLIError(f"{q_dir.name} has no vocab.json")
    from engine.value import descriptors, neighbours
    vocab = descriptors.DescriptorVocab.load(vpath)
    need = ["match_key", "round_num", "action_id", "descriptor_key", "V", "actor_side", "map_version", "tier",
            *(f"desc_{f}" for f in descriptors.FIELDS)]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise CLIError(f"{a.valued.name} lacks columns {missing}")
    keep = [t == a.tier for t in frame["tier"].to_list()]
    which = _split_of(a.split_file)
    if which is not None:
        keep = [k and which.get(m) == "train" for k, m in zip(keep, frame["match_key"].to_list())]
    frame, h = frame.filter(pl.Series(keep, dtype=pl.Boolean)), h[keep]
    if frame.height == 0:
        raise CLIError(f"no {a.tier} rows left to index")
    table = frame.select("match_key", "round_num", "action_id", "descriptor_key", "V",
                         pl.col("actor_side").alias("side"), "map_version", "tier")
    table = table.with_columns(pl.Series("desc_ids", descriptors.ids_from_frame(frame, vocab).tolist()))
    t0 = time.perf_counter()
    idx = neighbours.build_index(h, table)
    out = neighbours.save_index(idx, a.out, cfg=cfg)
    timings["build_index"] = time.perf_counter() - t0
    lg.info("built neighbour index", rows=idx.n, kind=idx.kind, d=idx.d)
    say(f"index: {_rel(out)}")
    return 0


def _cmd_targets(table: dict, chosen: list[str], model: str | None, extra: list[str], lg, timings: dict) -> int:
    worst = 0
    for t in chosen:
        module, flag = table[t]
        argv = ([flag, _model_arg(model)] if model and flag else []) + extra
        t0 = time.perf_counter()
        try:
            rc = delegate(module, argv, lg)
        except CLIError as exc:
            if len(chosen) == 1:
                raise
            lg.error(f"{t}: {exc}")
            rc = exc.code
        timings[t] = time.perf_counter() - t0
        worst = max(worst, rc)
    return worst


def cmd_eval(a, cfg, lg, run_id: str, timings: dict) -> int:
    chosen = list(EVAL_TARGETS) if a.all else [t for t in EVAL_TARGETS if getattr(a, t)]
    if not chosen:
        raise CLIError("choose an evaluation: " + ", ".join(f"--{t}" for t in EVAL_TARGETS) + " or --all")
    return _cmd_targets(EVAL_TARGETS, chosen, a.model, _passthrough(a._extra), lg, timings)


def cmd_sanity(a, cfg, lg, run_id: str, timings: dict) -> int:
    t = "counterfactual" if a.counterfactual else "model"
    return _cmd_targets(SANITY_TARGETS, [t], a.model, _passthrough(a._extra), lg, timings)


def cmd_bench(a, cfg, lg, run_id: str, timings: dict) -> int:
    script = paths.ROOT / "bench" / f"bench_{a.which}.py"
    if not script.is_file():
        raise CLIError(f"bench/{script.name} does not exist yet")
    t0 = time.perf_counter()
    rc = subprocess.run([sys.executable, str(script), *_passthrough(a._extra)], cwd=paths.ROOT).returncode
    timings[f"bench_{a.which}"] = time.perf_counter() - t0
    return rc


def cmd_inventory(a, cfg, lg, run_id: str, timings: dict) -> int:
    from engine.data import inventory
    if a.rebuild_catalog:
        from engine.data import layout
        say(f"catalog: {_rel(layout.rebuild_catalog())}")
    if a.models:
        from engine import manifest
        dirs = sorted(d for d in paths.MODELS_DIR.glob("*") if d.is_dir()) if paths.MODELS_DIR.is_dir() else []
        from engine.analyze import adopted_models
        adopted = adopted_models()
        for d in dirs:
            try:
                m = manifest.read(d)
                when = m.get("created_at")
            except Exception:
                when = None
            say(f"{d.name}  adopted={'yes' if d.name in adopted else 'no'}  created={when}")
        if not dirs:
            say("no model directories")
    return inventory.run(cfg, check_targets=a.assert_targets)


def cmd_status(a, cfg, lg, run_id: str, timings: dict) -> int:
    from engine import db
    s = db.stats()
    say(f"engine db: {s['matches']} matches | {s['processed']} processed ids | {s['runs']} runs | "
        f"{s['action_scores']} action scores")
    say(f"by status: {s['by_status'] or '{}'}")
    say(f"free space: {paths.free_space_gb():.0f} GB")
    return 0


def _load_dotenv(path: Path) -> None:
    """KEY=VALUE lines of the repo's .env into os.environ (existing values win)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cmd_scrape(a, cfg, lg, run_id: str, timings: dict) -> int:
    from engine import ingest
    if a.source == "hf" and not a.repo:                  # LOW-54: the HF source has no default repo
        raise CLIError("scrape --source hf needs --repo owner/name (see docs/DATA_SOURCES.md)", 2)
    _load_dotenv(paths.ROOT / ".env")
    if a.source in ("api", "playwright") and not os.environ.get("FACEIT_API_KEY", "").strip():
        raise CLIError("FACEIT_API_KEY is not set (put it in .env, see .env.example); scrape needs it", 2)
    kw = {k: v for k, v in (("min_level", a.min_level), ("min_elo", a.min_elo), ("region", a.region),
                            ("repo", a.repo)) if v is not None}
    t0 = time.perf_counter()
    try:
        failed = ingest.run(a.source, a.max, tier=a.tier, **kw)
    except Exception as exc:
        if type(exc).__name__ == "RateLimited":          # engine.sources.scraper.RateLimited (MED-24)
            raise CLIError(f"scrape stopped: {exc}", 1) from None
        raise
    finally:
        timings["scrape"] = time.perf_counter() - t0
    return 1 if failed else 0


def cmd_ingest(a, cfg, lg, run_id: str, timings: dict) -> int:
    from engine import db, ingest
    if not a.demo.is_file():
        raise CLIError(f"demo not found: {a.demo.name}")
    try:
        key = ingest.register_demo(a.match_id or a.demo.stem, a.demo, tier=a.tier, source=a.source)
    except db.MatchConflict as exc:
        raise CLIError(f"refused: {exc}", 2) from None
    say(f"registered as {key}")
    return 0


HANDLERS = {
    "analyze": cmd_analyze, "parse": cmd_parse, "build-states": _cmd_build("states"),
    "build-actions": _cmd_build("actions"), "eval": cmd_eval, "sanity": cmd_sanity, "bench": cmd_bench,
    "inventory": cmd_inventory, "status": cmd_status, "scrape": cmd_scrape, "ingest": cmd_ingest,
    "train-q": cmd_train_q, "build-neighbours": cmd_build_neighbours,
    **{name: _cmd_delegate(name) for name in DELEGATES},
}
PASSTHROUGH = set(DELEGATES) | {"eval", "sanity", "bench"}


# ── main ──────────────────────────────────────────────────────────────────────

def _exit_code(rc, lg) -> int:
    """LOW-04: the CLI exits 0, 1 or 2 only; any other command / delegate / bench code is 1."""
    rc = 0 if rc is None else rc
    if isinstance(rc, int) and not isinstance(rc, bool) and rc in (0, 1, 2):
        return rc
    lg.warning(f"exit code {rc!r} from the command reported as 1 (failed)")
    return 1


def _space_check(cmd: str, a, lg) -> None:
    status = paths.check_space()
    if status is paths.SpaceStatus.OK:
        return
    free = paths.free_space_gb()
    batch = cmd in BATCH and not (cmd == "parse" and not Path(a.target).is_dir())
    if status is paths.SpaceStatus.REFUSE and batch:
        raise CLIError(f"refused: only {free:.0f} GB free; batch {cmd} needs at least 20 GB")
    lg.warning(f"low disk space: {free:.0f} GB free (warning below 100 GB)")


def _write_manifest(cmd: str, a, cfg, run_id: str, timings: dict, rc: int, log_dir: Path) -> None:
    from engine import manifest
    from engine.log import redact_obj
    argv = [str(x) for x in (sys.argv[1:] if a._argv is None else a._argv)]
    manifest.write(log_dir / run_id, cmd, cfg=cfg, timings=timings,
                   extra=redact_obj({"argv": argv, "exit_code": rc, "run_id": run_id}))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if not (sys.argv[1:] if argv is None else argv):
        parser.print_help()                        # no command: the overview, not an error
        return 0
    try:
        a, extra = parser.parse_known_args(argv)
    except SystemExit as exc:                      # --help (0) or a usage error (2)
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
    cmd = a.command
    if extra and cmd not in PASSTHROUGH:
        try:
            parser.commands[cmd].error(f"unrecognized arguments: {' '.join(extra)}")
        except SystemExit as exc:
            return int(exc.code or 2)
    a._extra, a._argv = extra, argv

    from engine import config, log
    run_id  = log.new_run_id(cmd)
    log_dir = a.log_dir or paths.LOGS_DIR
    lg = log.get_logger(run_id, kind=cmd, log_dir=log_dir)
    timings: dict = {}
    cfg, rc = None, 1
    try:
        try:
            cfg = config.load(a.config)
        except (OSError, config.ConfigError) as exc:
            raise CLIError(f"config: {exc}") from None
        if cmd != "analyze":
            lg.info(cmd, run_id=run_id, config_hash=config.config_hash(cfg), device="cpu")
        _space_check(cmd, a, lg)
        rc = _exit_code(HANDLERS[cmd](a, cfg, lg, run_id, timings), lg)
    except CLIError as exc:
        lg.error(str(exc))
        rc = exc.code
    except KeyboardInterrupt:
        lg.error("interrupted")
        rc = 1                                     # LOW-04: 0/1/2 only, not 130
    except Exception as exc:
        try:                                       # LOW-08: a broken analyze import must not raise again here
            from engine.analyze import AnalyzeError
        except Exception:
            AnalyzeError = None
        refusal = AnalyzeError is not None and isinstance(exc, AnalyzeError)
        code = getattr(exc, "exit_code", None) if refusal else None
        if code is None:
            code = 2 if isinstance(exc, FileNotFoundError) else 1
        if refusal:
            lg.error(str(exc))                     # an expected refusal: one line
        else:
            lg.error(f"{type(exc).__name__}: {exc}", exc=exc)    # traceback goes to the log file only
        if a.debug:                                # LOW-03: redacted like every other line
            from engine.log import redact_text
            print(redact_text("".join(traceback.format_exception(exc))), file=sys.stderr, flush=True)
        rc = code
    finally:
        if timings:
            lg.info("timings", **{k: f"{v:.2f}s" for k, v in timings.items()})
        if cfg is not None and (cmd != "analyze" or rc != 0):     # analyze writes <out>/manifest.json
            try:
                _write_manifest(cmd, a, cfg, run_id, timings, rc, log_dir)
            except Exception as exc:              # a manifest failure must not hide the result
                lg.warning(f"manifest not written: {type(exc).__name__}: {exc}")
        lg.info("exit", run_id=run_id, code=rc)
        lg.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
