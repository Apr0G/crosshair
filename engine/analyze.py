"""
End-to-end `engine analyze` (P10.1.1): demo -> report directory.

    from engine import analyze
    res = analyze.analyze_demo("match.dem", tier="pro", out_dir=d, model_dir=m, device="cuda", cfg=cfg)
    res = analyze.analyze_tables(raw, match_key="m1", tier="pro", geometry=geo,
                                 predictor=stub, cfg=cfg, out_dir=d)        # tests, bench
    res.report, res.files, res.timings, res.value_stage

Stages (each timed with `bench/_bench.timed` into `timings`, the names of
`bench/budget.yaml`): parse -> states -> actions -> predict -> attribution -> value ->
report. `states` / `actions` run the staged extractor functions exactly as
`extractor.run` does (event actions, states with their boundary samples, movement
actions, one rebuild when movement added ticks, then enablers + windows), so peeks
have boundary samples. `predict` is the time spent inside the predictor during
attribution; `attribution` is the rest of `attribute_match`.

Value stage: with `value_models` (Q/B nets, descriptor vocab, neighbour index,
optional shrinkage fit; `ValueModels.load`) every non-synthetic action gets Q, B, D, R,
X, verdicts and a panel via `engine.value` (imported lazily). Without it Q, B, D, R, X
are None, `meta.value_stage = "skipped"` and a warning says so.

Outputs in `out_dir`: report.json (schema-validated, redacted), report.md, charts/,
actions_scored.parquet, manifest.json, timings.json; rows in `db.action_scores` under
the run_id (unless `write_db=False`). report.json is byte-identical across runs except
`meta.run_id`, `meta.created_at`, `meta.timings`. `meta.timings` holds the stages up to
value; timings.json adds `report` and `total`.

Usage log (P12.3): after a successful run with `write_db`, one record (run_id,
match_key, map_version, tier, timings, warnings count, max conservation residual) goes
through `engine.evaluation.usage.append_usage` to `usage_log_path()`: next to the
engine DB when `CROSSHAIR_ENGINE_DB` points elsewhere (tests), else
`data/engine/eval/usage.jsonl`. A failure to append is logged, never raised.

Guards (P11.3.3, P12.1): quarantine rules before parsing (a 64-tick demo is refused
with the reason); a `--tier` that differs from the demo's source tier (its `matches`
row, else a FACEIT server name) is a warning; a map version with < 100 registered
matches is a warning (skipped with a note when there is no engine DB); a model without
an `adopted` row in the adoption table of docs/ENGINE_MODELS.md is refused unless
`allow_unadopted` (the win model AND the value models `--q-model` / `--neighbours`;
2026-10-19 audit HIGH-02, HIGH-03).

Test hooks (never used by a real analysis): `stub_predictor()` (env
`CROSSHAIR_STUB_PREDICTOR=1` in the CLI) and `synthetic_raw(story)` (CLI
`--raw-tables-story`) import from `test/engine/`.
"""
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from engine import config as config_mod
from engine import db, manifest, paths
from engine.report import charts, writer

# ── constants ─────────────────────────────────────────────────────────────────

STAGES            = ("parse", "states", "actions", "predict", "attribution", "value", "report")
TIERS             = ("pro", "faceit_10", "faceit_8_9", "other")       # engine.state.builder.TIERS
EXPECTED_TICKRATE = 128
TICKRATE_TOL      = 0.5
MIN_MAP_MATCHES   = 100                  # P11.3.1: warn below this many matches per map version
MODELS_DOC        = paths.ROOT / "docs" / "ENGINE_MODELS.md"
STUB_ENV          = "CROSSHAIR_STUB_PREDICTOR"
TEST_DIR          = paths.ROOT / "test" / "engine"
_WHERE_KEYS       = ("attacker_callout", "from_callout", "throw_callout", "planter_callout", "callout",
                     "land_callout", "to_callout")

# actions_scored.parquet columns: mirrors engine.value.scores.SCORED_SCHEMA (P9.5.1) so the
# skipped-value path writes the same file without importing the value package.
SCORED_SCHEMA = {
    "match_key": pl.String, "round_num": pl.Int64, "action_id": pl.String, "actor_id_hash": pl.Int64,
    "actor_side": pl.String, "type": pl.String, "V": pl.Float64, "V_raw": pl.Float64,
    "V_direct": pl.Float64, "V_enabler": pl.Float64, "V_movement": pl.Float64, "Q": pl.Float64,
    "B": pl.Float64, "D": pl.Float64, "R": pl.Float64, "X": pl.Float64, "verdict_decision": pl.String,
    "verdict_execution": pl.String, "descriptor_key": pl.String, "panel_ref": pl.String,
    "panel_n": pl.Int64, "panel": pl.String, "method": pl.String, "n_coalition": pl.Int64,
    "residual": pl.Float64, "flags": pl.List(pl.String), "components": pl.String, "run_id": pl.String,
}


# ── errors ────────────────────────────────────────────────────────────────────

class AnalyzeError(RuntimeError):
    """A failure the CLI reports without a traceback; `exit_code` is its exit status."""
    exit_code = 1


class Refused(AnalyzeError):
    """The input breaks a rule (quarantine, tick rate): nothing was analysed."""
    exit_code = 2


class ModelUnavailable(AnalyzeError):
    """No model directory, a model that is not adopted, or an unfinished model module."""
    exit_code = 2


# ── helpers ───────────────────────────────────────────────────────────────────

def _timed():
    """`bench/_bench.timed` (bench/ is a plain directory, so put the repo root on the path)."""
    try:
        from bench._bench import timed
    except ImportError:
        sys.path.insert(0, str(paths.ROOT))
        from bench._bench import timed
    return timed


def _finite(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _warn(warnings: list, code: str, message: str, logger=None, *, round_num=None, action_id=None) -> None:
    w = {"code": code, "message": message}
    if round_num is not None:
        w["round_num"] = int(round_num)
    if action_id is not None:
        w["action_id"] = str(action_id)
    warnings.append(w)
    if logger is not None:
        logger.warning(message, code=code)


def q_tier(tier: str) -> str:
    """Analyze tier -> the tier token the Q/B nets know ("pro" / "faceit" / other)."""
    return "faceit" if tier.startswith("faceit") else tier


# ── predictor proxy ───────────────────────────────────────────────────────────

class _Recorder:
    """Wraps the predictor: times every call (the `predict` stage) and keeps W of every
    snapshot object it saw, so the report's WP curve uses exactly the factual W the
    attribution used. Factual snapshots stay alive for the whole run, so their id()
    cannot be reused by an edited state."""

    def __init__(self, inner):
        self.inner   = inner
        self.seconds = 0.0
        self.w: dict[int, float] = {}

    def predict(self, snapshots, histories) -> dict:
        t0 = time.perf_counter()
        out = self.inner.predict(snapshots, histories)
        self.seconds += time.perf_counter() - t0
        for s, w in zip(snapshots, out["win"]):
            self.w[id(s)] = float(w)
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


# ── value models ──────────────────────────────────────────────────────────────

@dataclass
class ValueModels:
    """Everything the value stage needs. `encoder.encode(snapshots) -> (N, d)` gives
    h_pre (default: the predictor's own `encode`)."""
    qnet:     object
    bnet:     object
    vocab:    object
    index:    object
    k_map:    dict | None = None
    encoder:  object = None
    versions: dict = field(default_factory=dict)

    @classmethod
    def load(cls, q_dir: Path, neighbours_dir: Path, device: str = "cpu") -> "ValueModels":
        """Load the Q/B nets and the neighbour index. The index must have been built with
        this Q model's descriptor vocab (manifest `q_vocab_sha256`); a different vocab
        means its stored ids point at other tokens, so the pair is refused (MED-17).
        `versions` also carries the WP model each artifact was built from, when known,
        for the caller's mismatch warning."""
        q_dir, neighbours_dir = Path(q_dir), Path(neighbours_dir)
        for d in (q_dir, neighbours_dir):
            if not d.is_dir():
                raise ModelUnavailable(f"value model directory not found: {d.name}")
        try:
            from engine.value import neighbours, q_model, shrinkage
        except ImportError as exc:
            raise ModelUnavailable(f"value modules unavailable: {exc}") from exc
        q_extra, n_extra = _manifest_extra(q_dir), _manifest_extra(neighbours_dir)
        want = n_extra.get("q_vocab_sha256")
        if want is not None and (q_dir / "vocab.json").is_file() and want != q_model.vocab_sha256(q_dir):
            raise ModelUnavailable(f"neighbour index {neighbours_dir.name} was built with the descriptor vocab of "
                                   f"{n_extra.get('q_model') or 'another Q model'}, not {q_dir.name}; rebuild it "
                                   f"with --q-model {q_dir.name}")
        q, b, vocab, _metrics = q_model.load_q(q_dir, device=device)
        index = neighbours.load_index(neighbours_dir)
        k_path = q_dir / shrinkage.FILE_NAME
        k_map = shrinkage.load(k_path) if k_path.is_file() else None
        versions = {"q": q_dir.name, "neighbours": neighbours_dir.name,
                    "q_wp_model": q_extra.get("wp_model"), "neighbours_wp_model": n_extra.get("wp_model"),
                    "provenance": "checked" if want is not None else "unrecorded"}
        return cls(q, b, vocab, index, k_map, None, versions)


def _manifest_extra(d: Path) -> dict:
    try:
        return dict(manifest.read(d).get("extra") or {})
    except (OSError, ValueError, manifest.ManifestSchemaError):
        return {}


def warn_value_provenance(vm, wp_name, warnings: list, logger=None) -> None:
    """One `value_provenance` warning per value artifact built from another WP model
    than the one analysing, and one when the index's vocab link is unrecorded."""
    if vm is None:
        return
    v = vm.versions or {}
    for key in ("q_wp_model", "neighbours_wp_model"):
        rec = v.get(key)
        if rec is not None and wp_name is not None and rec != wp_name:
            _warn(warnings, "value_provenance", f"{v.get(key.split('_')[0])} was built from WP model {rec}, "
                  f"this analysis uses {wp_name}: Q / regret use another encoder's h_pre", logger)
    if v.get("provenance") == "unrecorded":
        _warn(warnings, "value_provenance", f"neighbour index {v.get('neighbours')} does not record which Q vocab "
              "built it (built before 2026-10-19); rebuild to make the link checkable", logger)


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class AnalyzeResult:
    run_id:             str
    out_dir:            Path
    report:             dict
    scored:             pl.DataFrame
    round_attributions: list
    snapshots:          list
    actions:            list
    windows:            list
    timings:            dict
    warnings:           list
    files:              dict
    value_stage:        str
    n_db_rows:          int = 0


# ── guards (P1.4.1, P11.3, P12.1) ─────────────────────────────────────────────

def check_tick_rate(meta: dict) -> None:
    """Refuse anything that is not a 128-tick demo (P1.4.1 rule `tick_rate_not_128`).
    `tick_rate_raw` is preferred over the rounded `tick_rate`; a None / NaN / inf
    value is unknown and refused (R8: no `or` over numerics; MED-04, 2026-10-19)."""
    from engine.parse.awpy3_parser import _first_finite
    rate = _first_finite(meta.get("tick_rate_raw"), meta.get("tick_rate"))
    if rate is None or abs(float(rate) - EXPECTED_TICKRATE) > TICKRATE_TOL:
        shown = "unknown" if rate is None else f"{float(rate):g}"
        raise Refused(f"refused: demo tick rate is {shown}, the engine analyses only {EXPECTED_TICKRATE}-tick "
                      f"demos (quarantine rule tick_rate_not_128)")


def _db_available() -> bool:
    return db.db_path().is_file()


def source_tier(meta: dict) -> str | None:
    """The demo's own tier: its `matches` row (by demo_sha256) when the engine DB exists,
    else `faceit_10` when the server name says FACEIT, else None (unknown)."""
    sha = meta.get("demo_sha256")
    if sha and _db_available():
        key = db.match_key_for_sha(sha)
        row = db.get_match(key) if key else None
        if row and row.get("tier"):
            return row["tier"]
    server = str(meta.get("server") or "")
    if "faceit" in server.lower():
        return "faceit_10"
    return None


def match_row(meta: dict) -> dict | None:
    sha = meta.get("demo_sha256")
    if not sha or not _db_available():
        return None
    key = db.match_key_for_sha(sha)
    return db.get_match(key) if key else None


def map_support(map_version: str | None) -> int | None:
    """Registered, non-quarantined matches on this map version; None without a DB."""
    if map_version is None or not _db_available():
        return None
    return sum(1 for m in db.list_matches()
               if m.get("map_version") == map_version and m.get("status") != "quarantined")


def guard_warnings(meta: dict, tier: str, map_version: str | None, logger=None) -> list[dict]:
    """Tier override and map-version support warnings (P11.3.1, P11.3.3)."""
    out: list[dict] = []
    src = source_tier(meta)
    if src is not None and src != tier and not (src == "golden"):
        _warn(out, "tier_overridden", f"tier token overridden: the demo's source tier is {src}, "
              f"analysed as {tier}", logger)
    n = map_support(map_version)
    if n is None:
        if logger is not None:
            logger.info("map-version support check skipped (no engine DB)", map_version=map_version)
    elif n < MIN_MAP_MATCHES:
        _warn(out, "map_version_low_support", f"map version {map_version} has {n} registered matches "
              f"(< {MIN_MAP_MATCHES}); the model may not have seen enough of it", logger)
    return out


def adopted_models(doc: Path | None = None) -> set[str]:
    """Model names with an `adopted` row in the adoption table of docs/ENGINE_MODELS.md
    (P12.1), read with `engine.evaluation.registry.adoption_rows`: the model cell and
    the status cell, nothing else. A version named in prose or in another row's rule
    text adopts nothing (2026-10-19 audit, HIGH-02)."""
    from engine.evaluation import registry
    doc = Path(doc) if doc is not None else MODELS_DOC
    return {r["model"] for r in registry.adoption_rows(doc) if r["status"].lower() == "adopted"}


def resolve_model_dir(model: str | None, model_dir: str | Path | None) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    if model:
        return paths.MODELS_DIR / model
    raise ModelUnavailable("no model given: pass --model wp_vN (under data/engine/models) or --model-dir")


def check_model(model_dir: Path, allow_unadopted: bool = False, doc: Path | None = None,
                hint: str = "train one with `train-wp`", models_dir: Path | None = None) -> None:
    """The adoption gate (P12.1). An adopted name counts only for the directory the
    registry adopted, `<models_dir>/<name>`; a same-named folder elsewhere needs
    `--allow-unadopted` (MED-02, 2026-10-19)."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ModelUnavailable(f"model directory not found: {model_dir.name} ({hint})")
    if allow_unadopted:
        return
    if model_dir.name not in adopted_models(doc):
        raise ModelUnavailable(f"model {model_dir.name} is not adopted in docs/ENGINE_MODELS.md; "
                               f"pass --allow-unadopted to use it anyway")
    home = Path(models_dir) if models_dir is not None else paths.MODELS_DIR
    if model_dir.resolve() != (home / model_dir.name).resolve():
        raise ModelUnavailable(f"model {model_dir.name} is adopted, but this directory is not the adopted one "
                               f"under {home.name}/; pass --allow-unadopted to analyse a copy")


VALUE_HINTS = ("train one with `train-q`", "build one with `build-neighbours`")


def model_hazard_auc(model_dir: Path | None) -> float | None:
    """The model's hazard-head AUC on test pro from `<model_dir>/eval.json` (written by
    `calibrate`), else None."""
    if model_dir is None:
        return None
    p = Path(model_dir) / "eval.json"
    if not p.is_file():
        return None
    try:
        v = json.loads(p.read_text(encoding="utf-8")).get("hazard_auc")
        return float(v) if v is not None else None
    except (OSError, ValueError, TypeError):
        return None


def apply_hazard_rule(cfg, model_dir: Path | None, warnings: list, logger=None):
    """P6.7.2's hazard-head rule (MED-39, 2026-10-19): AUC below `sanity.hazard_auc_floor`
    forces `enabler_mode: fixed` for this analysis and warns; between the floor and
    `sanity.hazard_auc_target` every report warns "hazard head below target"; an
    unknown AUC warns once. Returns the (possibly replaced) config."""
    from dataclasses import replace as _replace
    if cfg.attribution.enabler_mode != "hazard":
        return cfg
    auc = model_hazard_auc(model_dir)
    if auc is None:
        _warn(warnings, "hazard_auc_unknown", "the model records no hazard AUC (no eval.json from calibrate); "
              "enabler splits use the hazard head unchecked", logger)
        return cfg
    if auc < cfg.sanity.hazard_auc_floor:
        _warn(warnings, "fixed_share_fallback", f"hazard AUC {auc:.3f} is below the floor "
              f"{cfg.sanity.hazard_auc_floor}: enabler shares are the configured fixed shares (P8.5.3)", logger)
        return _replace(cfg, attribution=_replace(cfg.attribution, enabler_mode="fixed"))
    if auc < cfg.sanity.hazard_auc_target:
        _warn(warnings, "hazard_below_target", f"hazard head below target: AUC {auc:.3f} < "
              f"{cfg.sanity.hazard_auc_target}", logger)
    return cfg


def check_value_models(value_dirs: tuple, allow_unadopted: bool, warnings: list, logger=None,
                       doc: Path | None = None, models_dir: Path | None = None) -> None:
    """The adoption gate for the value stage (P12.1 applies to every model a report
    uses): `--q-model` and `--neighbours` directories must exist and be adopted, or
    `allow_unadopted` is set and each unadopted one is a `model_unadopted` warning."""
    for d, hint in zip(value_dirs, VALUE_HINTS):
        check_model(d, allow_unadopted, doc, hint, models_dir)
        if allow_unadopted and Path(d).name not in adopted_models(doc):
            _warn(warnings, "model_unadopted", f"model {Path(d).name} is not adopted (--allow-unadopted)", logger)


def load_predictor(model_dir: Path, device: str | None):
    try:
        from engine.model.predict import Predictor
    except ImportError as exc:
        raise ModelUnavailable(f"engine.model.predict unavailable: {exc}") from exc
    try:
        return Predictor(model_dir, device, require_calibration=True)    # R6: never uncalibrated (MED-20)
    except NotImplementedError as exc:
        raise ModelUnavailable(f"Predictor not implemented yet: {exc}") from exc
    except FileNotFoundError as exc:
        raise ModelUnavailable(f"model directory incomplete: {Path(str(exc)).name}") from exc


# ── test hooks ────────────────────────────────────────────────────────────────

def _test_path() -> None:
    if str(TEST_DIR) not in sys.path:
        sys.path.insert(0, str(TEST_DIR))


def stub_predictor():
    """TEST ONLY: `_cf_states.RuleStub(good_win)` (W from man advantage and the bomb)."""
    _test_path()
    from _cf_states import RuleStub, good_win
    return RuleStub(good_win)


def synthetic_raw(story: str, tick_rate: int = 128):
    """TEST ONLY: `_synthetic.make_raw_tables(story, tick_rate)`."""
    _test_path()
    from _synthetic import make_raw_tables
    return make_raw_tables(story, tick_rate=tick_rate)


# ── states + actions ──────────────────────────────────────────────────────────

def build_states_actions(raw, cfg, *, match_key: str, tier: str, geometry, timings: dict):
    """`extractor.run` split into timed `states` / `actions` stages; same result."""
    from engine.actions import extractor
    from engine.state import builder
    timed = _timed()
    with timed("actions", timings):
        ctx = extractor.Context(raw, cfg, match_key, geometry)
        ev = extractor.extract_event_actions(raw, cfg, match_key, geometry=geometry, ctx=ctx)
        at, dw = extractor.event_ticks(ev), extractor.durative_windows(ev)
    with timed("states", timings):
        snaps = builder.build(raw, geometry, cfg, match_key=match_key, tier=tier,
                              action_ticks=at, durative_windows=dw)
    with timed("actions", timings):
        mv = extractor.extract_movement_actions(snaps, raw, cfg, match_key, geometry, ctx=ctx)
        at2, dw2 = extractor.event_ticks(ev + mv), extractor.durative_windows(ev + mv)
    if at2 != at or dw2 != dw:
        with timed("states", timings):
            snaps = builder.build(raw, geometry, cfg, match_key=match_key, tier=tier,
                                  action_ticks=at2, durative_windows=dw2)
    with timed("actions", timings):
        actions, windows = extractor.extract(raw, snaps, cfg, match_key=match_key, geometry=geometry, ctx=ctx)
    return snaps, actions, windows


# ── value stage ───────────────────────────────────────────────────────────────

def _state_pre(a, snaps_sorted: list, ticks: list[int]):
    """The action's pre snapshot: exact `state_pre_tick`, else the latest before it; an
    action starting at freeze end (pre tick before the first sample) falls back to the
    latest snapshot at or before its own tick."""
    from bisect import bisect_right
    for t in (a.state_pre_tick, a.id_tick):
        if t is None:
            continue
        i = bisect_right(ticks, int(t)) - 1
        if i >= 0:
            return snaps_sorted[i]
    return None


def value_stage(ras, actions_by_id: dict, snaps: list, predictor, vm: ValueModels, cfg, *, tier: str,
                map_version: str | None):
    """({action_id: ScoredActionWithValue}, player aggregate rows or None)."""
    from engine.attribution import engine as att
    from engine.value import descriptors, scores
    by_round: dict = {}
    for s in sorted(snaps, key=lambda s: (s.meta.round_num, s.meta.tick)):
        by_round.setdefault(s.meta.round_num, []).append(s)
    ticks = {r: [s.meta.tick for s in v] for r, v in by_round.items()}
    chosen, pre = [], []
    for ra in ras:
        for sa in ra.actions:
            a = actions_by_id.get(sa.action_id)
            if a is None:
                continue                                    # synthetic rows have no descriptor
            s = _state_pre(a, by_round.get(a.round_num, []), ticks.get(a.round_num, []))
            if s is None:
                continue
            chosen.append(a)
            pre.append(s)
    if not chosen:
        return {}, None
    encoder = vm.encoder if vm.encoder is not None else predictor
    h = encoder.encode(pre)
    desc = descriptors.frame_columns(chosen, pre).with_row_index("_i")
    frame = (att.to_frame(ras).join(desc, on="action_id", how="inner").sort("_i").drop("_i")
             .with_columns(pl.lit(q_tier(tier)).alias("tier"), pl.lit(map_version, dtype=pl.String).alias("map_version")))
    svs = scores.score_frame(frame, h, vm.qnet, vm.bnet, vm.vocab, vm.index, cfg)
    out = {sv.action_id: sv for sv in svs}
    agg = None
    if vm.k_map is not None:
        from engine.value import shrinkage
        f = pl.DataFrame({"actor_id_hash": [sv.actor_id_hash for sv in svs], "type": [sv.type for sv in svs],
                          "X": [sv.X for sv in svs], "D": [sv.D for sv in svs]},
                         schema={"actor_id_hash": pl.Int64, "type": pl.String, "X": pl.Float64, "D": pl.Float64})
        agg = shrinkage.player_aggregates(f, vm.k_map).to_dicts()
    return out, agg


# ── report assembly ───────────────────────────────────────────────────────────

def _id_maps(raw, show_ids: bool) -> tuple[dict, dict]:
    """(actor_id_hash -> steamid str, actor_id_hash -> name) when show_ids; else empty."""
    if not show_ids:
        return {}, {}
    from engine.state.players import player_id_hash
    sids = [s for s in raw.ticks["steamid"].unique().to_list() if s]
    sid_of = {player_id_hash(int(s)): str(int(s)) for s in sids}
    names = {}
    for p in raw.meta.get("players") or []:
        if p.get("steamid"):
            names[player_id_hash(int(p["steamid"]))] = p.get("name")
    return sid_of, names


def _teams(raw) -> dict:
    out = {"ct_start": None, "t_start": None}
    for side in ("ct", "t"):
        c = Counter(p.get("team") for p in raw.meta.get("players") or [] if p.get("side") == side and p.get("team"))
        if c:
            out[f"{side}_start"] = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return out


def _where(a) -> str | None:
    if a is None:
        return None
    for k in _WHERE_KEYS:
        v = (a.params or {}).get(k)
        if v is not None:
            return str(v)
    return None


def _value_fields(sv) -> dict:
    if sv is None:
        return {"Q": None, "B": None, "D": None, "R": None, "X": None, "verdict_decision": None,
                "verdict_execution": None, "descriptor_key": None, "panel": None}
    panel = None
    if sv.panel is not None:
        d = sv.panel.to_dict()
        panel = {"ref": d["ref"], "k": int(d["k"]), "n_used": int(d["n_used"]), "n_other": int(d["n_other"]),
                 "top": [{"key": e["key"], "n": int(e["n"]), "freq": float(e["freq"]), "mean_V": float(e["mean_V"]),
                          "mean_Q": _finite(e.get("mean_Q"))} for e in d["entries"][:20]]}
    return {"Q": _finite(sv.Q), "B": _finite(sv.B), "D": _finite(sv.D), "R": _finite(sv.R), "X": _finite(sv.X),
            "verdict_decision": sv.verdict_decision, "verdict_execution": sv.verdict_execution,
            "descriptor_key": sv.descriptor_key, "panel": panel}


def _enabler_index(ras) -> dict:
    """event action_id -> [(beneficiary ScoredAction, component)] for enabler shares."""
    out: dict = {}
    for ra in ras:
        for sa in ra.actions:
            for c in sa.components:
                if c.get("source") == "enabler" and c.get("event_id"):
                    out.setdefault(c["event_id"], []).append((sa, c))
    return out


def build_report(*, raw, ras, actions, windows, rec: _Recorder, value: dict, agg, cfg, match_key: str,
                 tier: str, map_version: str | None, run_id: str, timings: dict, warnings: list,
                 model_versions: dict, value_stage_state: str, show_ids: bool, source_tier_: str | None,
                 date: str | None, snaps: list) -> dict:
    from engine.attribution.types import actor_sign
    tr = int(raw.meta["tick_rate"])
    rounds_meta = {int(r["round_num"]): r for r in raw.rounds.to_dicts()}
    acts = {a.action_id: a for a in actions}
    sid_of, name_of = _id_maps(raw, show_ids)
    enablers = _enabler_index(ras)
    snaps_by: dict = {}
    for s in snaps:
        snaps_by.setdefault(s.meta.round_num, []).append(s)

    def t_s(r: int, tick) -> float | None:
        fe = rounds_meta.get(r, {}).get("freeze_end_tick")
        return None if tick is None or fe is None else round((int(tick) - int(fe)) / tr, 6)

    def ident(h) -> dict:
        d = {"actor": writer.player_label(h)}
        if show_ids and h is not None:
            d["steamid"] = sid_of.get(h)
            d["name"] = name_of.get(h)
        return d

    rounds = []
    for ra in ras:
        r = ra.round_num
        st = ra.stats
        sampled, failed = set(st.get("sampled_windows") or []), {f["window_id"] for f in st.get("failed_windows") or []}
        curve = sorted(((s.meta.tick, rec.w[id(s)]) for s in snaps_by.get(r, []) if id(s) in rec.w))
        if curve:
            curve[0], curve[-1] = (curve[0][0], ra.w_start), (curve[-1][0], ra.w_end)
        wins = sorted((w for w in windows if int(w.round_num) == r), key=lambda w: (w.start_tick, w.window_id))
        rows = []
        for sa in ra.actions:
            a = acts.get(sa.action_id)
            tick = a.tick if a is not None else None
            if a is not None and tick is None:
                tick = a.window_start_tick
            en_rows = [{"kind": c.get("kind"), "action_id": ben.action_id, "actor": writer.player_label(ben.actor_id_hash),
                        "type": ben.type, "share": _finite(c.get("share")), "V": actor_sign(ben.actor_side) * float(c["phi"])}
                       for ben, c in enablers.get(sa.action_id, [])]
            en_rows.sort(key=lambda e: (e["kind"] or "", e["action_id"] or ""))
            row = {"action_id": sa.action_id, "round_num": r, "type": sa.type,
                   "subtype": a.subtype if a is not None else None, **ident(sa.actor_id_hash),
                   "actor_side": sa.actor_side, "tick": tick, "t_s": t_s(r, tick),
                   "window_start_tick": a.window_start_tick if a is not None else None,
                   "window_end_tick": a.window_end_tick if a is not None else None,
                   "where": _where(a), "synthetic": a is None,
                   "V": float(sa.V), "V_direct": float(sa.V_direct), "V_enabler": float(sa.V_enabler),
                   "V_movement": float(sa.V_movement), **_value_fields(value.get(sa.action_id)),
                   "method": sa.method, "n_coalition": sa.n_coalition, "residual": _finite(sa.residual),
                   "flags": sorted(set(sa.flags) | set(value[sa.action_id].flags if sa.action_id in value else ())),
                   "enablers": en_rows}
            rows.append(row)
        rounds.append({
            "round_num": r, "winner_side": rounds_meta.get(r, {}).get("winner_side"),
            "end_reason": rounds_meta.get(r, {}).get("end_reason"),
            "w_start": float(ra.w_start), "w_end": float(ra.w_end),
            "wp_curve": [{"tick": int(t), "t_s": t_s(r, t), "W": float(w)} for t, w in curve],
            "windows": [{"window_id": w.window_id, "start_tick": int(w.start_tick), "end_tick": int(w.end_tick),
                         "member_action_ids": list(w.member_action_ids), "sampled": w.window_id in sampled,
                         "failed": w.window_id in failed} for w in wins],
            "actions": rows, "clock_ct": float(ra.clock_ct), "clock_t": float(ra.clock_t),
            "conservation_residual": float(ra.conservation_residual),
            "stats": {k: st.get(k) for k in ("n_windows", "n_drift", "predictor_calls", "coalition_evals",
                                             "terminal_pinned", "conservation_ok", "conservation_tol")},
        })

    # ── warnings from the attribution and value outputs ──
    for rnd, ra in zip(rounds, ras):
        r = ra.round_num
        for wid in sorted(ra.stats.get("sampled_windows") or []):
            _warn(warnings, "sampled_window", f"window {wid} used sampled Shapley", round_num=r)
        for f in ra.stats.get("failed_windows") or []:
            _warn(warnings, "window_failed", f"window {f['window_id']}: intervention failed, its change went "
                  f"to the clock ({f['error']})", round_num=r)
        for f in ra.stats.get("enabler_flags") or []:
            _warn(warnings, "enabler_flag", str(f), round_num=r)
        if not ra.stats.get("conservation_ok", True):
            _warn(warnings, "conservation_residual", f"conservation residual {ra.conservation_residual:.2e}",
                  round_num=r)
        for row in rnd["actions"]:
            fl = row["flags"]
            if any("ood" in f for f in fl):
                _warn(warnings, "ood_edit", "an edited state was off-distribution", round_num=r, action_id=row["action_id"])
            if "q_low_support" in fl:
                _warn(warnings, "low_support_descriptor", "fewer than n_min_regret pro neighbours did this",
                      round_num=r, action_id=row["action_id"])
            if any(f in ("hazard_nan", "min_hazard") for f in fl):
                _warn(warnings, "enabler_band_floor", "hazard head unavailable or unstable: enabler shares at "
                      "their band floor", round_num=r, action_id=row["action_id"])
    if cfg.attribution.enabler_mode == "fixed":
        _warn(warnings, "fixed_share_fallback", "enabler_mode is fixed: enabler shares are the configured "
              "fixed shares, not hazard-based")

    # ── players ──
    agg_by: dict = {}
    for row in agg or []:
        agg_by.setdefault(row["actor_id_hash"], []).append(row)
    hashes = sorted({sa.actor_id_hash for ra in ras for sa in ra.actions if sa.actor_id_hash is not None},
                    key=lambda h: writer.player_label(h))
    players = []
    for h in hashes:
        mine = [sa for ra in ras for sa in ra.actions if sa.actor_id_hash == h]
        by_type: dict = {}
        for sa in mine:
            n, v = by_type.get(sa.type, (0, 0.0))
            by_type[sa.type] = (n + 1, v + float(sa.V))
        p = {"player": writer.player_label(h), "sides": sorted({sa.actor_side for sa in mine}),
             "n": len(mine), "V_total": math.fsum(float(sa.V) for sa in mine),
             "by_type": [{"type": t, "n": n, "V": v} for t, (n, v) in sorted(by_type.items())],
             "by_class": [{"class": str(c["type_class"]), "n": int(c["n"]),
                           **{f"{v}_{s}": _finite(c.get(f"{v}_{s}")) for v in ("D", "X")
                              for s in ("mean", "tilde", "lo", "hi")}}
                          for c in sorted(agg_by.get(h, []), key=lambda c: str(c["type_class"]))]}
        if show_ids:
            p["steamid"], p["name"] = sid_of.get(h), name_of.get(h)
        players.append(p)

    # ── teams (by side) ──
    teams = []
    for side in ("ct", "t"):
        rows = [row for rnd in rounds for row in rnd["actions"] if row["actor_side"] == side]
        buys = [{"round_num": row["round_num"], "action_id": row["action_id"], "type": row["type"],
                 "actor": row.get("actor"), "V": row["V"],
                 "loadout": (acts[row["action_id"]].params.get("loadout_class")
                             or acts[row["action_id"]].params.get("team_class")) if row["action_id"] in acts else None}
                for row in rows if row["type"] in ("buy", "team_buy")]
        clock = math.fsum(ra.clock_ct if side == "ct" else ra.clock_t for ra in ras)
        teams.append({"side": side, "clock_total": clock, "V_total": math.fsum(row["V"] for row in rows),
                      "buys": buys})

    uniq, seen = [], set()
    for w in warnings:
        key = json.dumps(w, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    uniq.sort(key=lambda w: (w.get("round_num") if w.get("round_num") is not None else -1, w["code"],
                             w.get("action_id") or "", w["message"]))

    meta = {
        "match_key": match_key, "map": raw.meta.get("map"), "map_version": map_version, "tier": tier,
        "source_tier": source_tier_, "teams": _teams(raw), "date": date,
        "tick_rate": tr, "n_rounds": len(rounds), "config_hash": config_mod.config_hash(cfg),
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timings": {k: round(float(v), 4) for k, v in timings.items()},
        "model_versions": dict(sorted(model_versions.items())), "value_stage": value_stage_state,
        "show_ids": bool(show_ids),
    }
    return {"report_schema_version": writer.REPORT_SCHEMA_VERSION, "meta": meta, "rounds": rounds,
            "players": players, "teams": teams, "warnings": uniq}


def scored_frame(ras, value: dict, run_id: str) -> pl.DataFrame:
    """actions_scored.parquet rows (SCORED_SCHEMA); value columns None when skipped."""
    rows = []
    for ra in ras:
        for sa in ra.actions:
            sv = value.get(sa.action_id)
            vf = _value_fields(sv)
            rows.append({
                "match_key": sa.match_key, "round_num": sa.round_num, "action_id": sa.action_id,
                "actor_id_hash": sa.actor_id_hash, "actor_side": sa.actor_side, "type": sa.type,
                "V": sa.V, "V_raw": sa.V_raw, "V_direct": sa.V_direct, "V_enabler": sa.V_enabler,
                "V_movement": sa.V_movement, "Q": vf["Q"], "B": vf["B"], "D": vf["D"], "R": vf["R"], "X": vf["X"],
                "verdict_decision": vf["verdict_decision"], "verdict_execution": vf["verdict_execution"],
                "descriptor_key": vf["descriptor_key"], "panel_ref": sv.panel_ref if sv is not None else None,
                "panel_n": sv.panel.n_used if sv is not None and sv.panel is not None else 0,
                "panel": (json.dumps(sv.panel.to_dict(), sort_keys=True, allow_nan=False)
                          if sv is not None and sv.panel is not None else None),
                "method": sa.method, "n_coalition": sa.n_coalition, "residual": _finite(sa.residual),
                "flags": list(sv.flags if sv is not None else sa.flags),
                "components": json.dumps(sa.components, sort_keys=True, allow_nan=False), "run_id": run_id,
            })
    return pl.DataFrame(rows, schema=SCORED_SCHEMA)


# ── usage log (P12.3) ─────────────────────────────────────────────────────────

def usage_log_path() -> Path:
    env = os.environ.get(db.ENV_DB)
    if env:
        return Path(env).parent / "usage.jsonl"
    from engine.evaluation import usage
    return usage.default_path()


def append_usage_record(record: dict, logger=None) -> bool:
    """`engine.evaluation.usage.append_usage(record, path)`; False (and a log line) on
    any failure: the usage log must never fail an analysis."""
    try:
        from engine.evaluation import usage
        usage.append_usage(record, usage_log_path())
        return True
    except Exception as exc:
        msg = f"usage record not written: {type(exc).__name__}: {exc}"
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg, file=sys.stderr)
        return False


# ── analyze ───────────────────────────────────────────────────────────────────

def analyze_tables(raw, *, match_key: str, tier: str, geometry, predictor, cfg=None, value_models=None,
                   out_dir, run_id: str | None = None, show_ids: bool = False, model_versions: dict | None = None,
                   warnings: list | None = None, timings: dict | None = None, write_db: bool = True,
                   source_tier_: str | None = None, date: str | None = None, inputs: list | None = None,
                   map_assets: dict | None = None, logger=None) -> AnalyzeResult:
    """Everything after parsing (see the module docstring). `predictor` has
    `predict(snapshots, histories)` (and `encode` for the value stage)."""
    from engine import log
    from engine.attribution import engine as att
    timed = _timed()
    cfg = cfg if cfg is not None else config_mod.load()
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    timings = timings if timings is not None else {}
    warnings = list(warnings or [])
    run_id = run_id or log.new_run_id("analyze")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_version = getattr(geometry, "map_version", None)
    model_versions = dict(model_versions or {"wp": None})
    model_versions.setdefault("q", None)
    model_versions.setdefault("neighbours", None)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    snaps, actions, windows = build_states_actions(raw, cfg, match_key=match_key, tier=tier,
                                                   geometry=geometry, timings=timings)

    rec = _Recorder(predictor)
    t0 = time.perf_counter()
    ras = att.attribute_match(snaps, actions, windows, rec, cfg, geometry=geometry)
    total = time.perf_counter() - t0
    timings["predict"] = timings.get("predict", 0.0) + rec.seconds
    timings["attribution"] = timings.get("attribution", 0.0) + max(0.0, total - rec.seconds)

    value, agg, state = {}, None, "skipped"
    with timed("value", timings):
        if value_models is not None:
            value, agg = value_stage(ras, {a.action_id: a for a in actions}, snaps, predictor, value_models, cfg,
                                     tier=tier, map_version=map_version)
            state = "done"
            model_versions.update({k: v for k, v in (value_models.versions or {}).items() if v})
    if state == "skipped":
        _warn(warnings, "value_stage_skipped", "value stage skipped (no Q / neighbour models): "
              "Q, B, D, R, X are null", logger)

    files: dict = {}
    with timed("report", timings):
        stage_timings = {k: timings[k] for k in STAGES if k in timings and k != "report"}
        rep = build_report(raw=raw, ras=ras, actions=actions, windows=windows, rec=rec, value=value, agg=agg,
                           cfg=cfg, match_key=match_key, tier=tier, map_version=map_version, run_id=run_id,
                           timings=stage_timings, warnings=warnings, model_versions=model_versions,
                           value_stage_state=state, show_ids=show_ids, source_tier_=source_tier_, date=date,
                           snaps=snaps)
        rep = writer.redact(rep, out_dir, show_ids)
        files["report.json"] = writer.write_json(rep, out_dir / "report.json")
        chart_names = charts.write_all(rep, out_dir / "charts")
        files["charts"] = [out_dir / c for c in chart_names]
        files["report.md"] = writer.write_markdown(rep, out_dir / "report.md", chart_names)
        frame = scored_frame(ras, value, run_id)
        files["actions_scored.parquet"] = out_dir / "actions_scored.parquet"
        frame.write_parquet(files["actions_scored.parquet"])

    timings_doc = {k: round(float(timings[k]), 4) for k in STAGES if k in timings}
    timings_doc["total"] = round(math.fsum(timings_doc.values()), 4)
    files["timings.json"] = out_dir / "timings.json"
    files["timings.json"].write_text(json.dumps(timings_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = ["report.json", "report.md", "actions_scored.parquet", "timings.json", *chart_names]
    m = manifest.write(out_dir, "analyze", inputs=inputs or [], outputs=outputs, cfg=cfg, timings=timings_doc,
                       map_assets=map_assets,
                       extra={"match_key": match_key, "tier": tier, "run_id": run_id, "value_stage": state,
                              "model_versions": model_versions, "map_version": map_version,
                              "n_warnings": len(rep["warnings"])})
    files["manifest.json"] = out_dir / manifest.MANIFEST_NAME

    n_rows = 0
    if write_db:
        db.upsert_run(run_id, "analyze", started_at=started, config_hash=m["config_hash"], git_sha=m["git_sha"],
                      manifest_path=str(files["manifest.json"]), status="running")
        rows = [{c: r.get(c) for c in db.SCORE_COLUMNS} for r in frame.iter_rows(named=True)]
        n_rows = db.insert_action_scores(rows)
        db.upsert_run(run_id, finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), status="ok")
        append_usage_record({
            "run_id": run_id, "match_key": match_key, "map_version": map_version, "tier": tier,
            "device": getattr(predictor, "device", None), "timings": timings_doc, "warnings": len(rep["warnings"]),
            "conservation_residual": max((abs(r["conservation_residual"]) for r in rep["rounds"]), default=0.0),
        }, logger)

    return AnalyzeResult(run_id=run_id, out_dir=out_dir, report=rep, scored=frame, round_attributions=ras,
                         snapshots=snaps, actions=actions, windows=windows, timings=timings_doc,
                         warnings=rep["warnings"], files=files, value_stage=state, n_db_rows=n_rows)


def analyze_demo(path, *, tier: str, out_dir, model_dir=None, device: str | None = None, cfg=None,
                 allow_unadopted: bool = False, show_ids: bool = False, predictor=None,
                 value_dirs: tuple | None = None, write_db: bool = True, run_id: str | None = None,
                 logger=None, warnings: list | None = None) -> AnalyzeResult:
    """Quarantine check -> parse -> RawTables -> map geometry -> Predictor -> analyze_tables.
    `predictor` (tests) replaces loading one from `model_dir`; `value_dirs` = (q_dir,
    neighbours_dir) enables the value stage; `warnings` seeds the report's warning list."""
    from engine.maps import assets as map_assets
    from engine.maps import callouts as map_callouts
    from engine.maps import version as map_version_mod
    from engine.parse import awpy3_parser
    from engine.parse.schema import RawTables
    from engine.state.geometry import MapGeometry
    timed = _timed()
    cfg = cfg if cfg is not None else config_mod.load()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"demo not found: {path.name}")
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    timings: dict = {}
    warnings = list(warnings or [])

    # ── quarantine before the expensive parse (P1.4.1) ──
    with timed("parse", timings):
        header = awpy3_parser.header_info(path, with_roster=True)
    check_tick_rate(header)
    reason = awpy3_parser.quarantine_check(header)
    if reason is not None:
        raise Refused(f"refused: demo fails the quarantine rule {reason}")

    # ── model before parse too: a missing model should not cost a parse ──
    model_versions = {"wp": None}
    if predictor is None:
        model_dir = Path(model_dir) if model_dir is not None else None
        if model_dir is None:
            raise ModelUnavailable("no model directory given (--model wp_vN or --model-dir)")
        check_model(model_dir, allow_unadopted)
        if allow_unadopted and model_dir.name not in adopted_models():
            _warn(warnings, "model_unadopted", f"model {model_dir.name} is not adopted (--allow-unadopted)", logger)
        predictor = load_predictor(model_dir, device)
        model_versions["wp"] = model_dir.name
        cfg = apply_hazard_rule(cfg, model_dir, warnings, logger)
    else:
        model_versions["wp"] = getattr(predictor, "name", None) or type(predictor).__name__
    vm = None
    if value_dirs is not None:
        check_value_models(value_dirs, allow_unadopted, warnings, logger)
        vm = ValueModels.load(value_dirs[0], value_dirs[1], device or "cpu")
        warn_value_provenance(vm, model_versions.get("wp"), warnings, logger)

    with timed("parse", timings):
        tables, meta = awpy3_parser.parse(path)
        raw = RawTables.from_parsed(tables, meta)
        raw.validate(strict=True)
    check_tick_rate(raw.meta)

    map_name = raw.meta["map"]
    try:
        cos = map_callouts.load(map_name)
    except (FileNotFoundError, ValueError) as exc:
        cos = []
        _warn(warnings, "callouts_missing", f"no usable callouts for {map_name} ({type(exc).__name__}); "
              "positions fall back to nav-area names", logger)
    mv = map_version_mod.detect({"build_num": raw.meta.get("game_version")}, map_name, logger=logger)
    geo = MapGeometry.from_assets(map_assets.ensure(map_name), cos, mv)
    row = match_row(raw.meta)
    src = source_tier(raw.meta)
    warnings += guard_warnings(raw.meta, tier, mv, logger)
    return analyze_tables(raw, match_key=(row or {}).get("match_key") or path.stem, tier=tier, geometry=geo,
                          predictor=predictor, cfg=cfg, value_models=vm, out_dir=out_dir, run_id=run_id,
                          show_ids=show_ids, model_versions=model_versions, warnings=warnings, timings=timings,
                          write_db=write_db, source_tier_=src, date=(row or {}).get("date"), inputs=[path],
                          map_assets=map_assets.asset_hashes(map_name), logger=logger)


def stub_enabled() -> bool:
    return os.environ.get(STUB_ENV, "") not in ("", "0")
