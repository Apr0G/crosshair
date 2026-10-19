"""
The counterfactual sanity gate `engine sanity --counterfactual` (P7.2): un-kill, enemy
smoke, un-flash, un-plant, freeze, order independence, OOD rate; thresholds in
`config.sanity`. P8 does not start until it is green.

    from engine.counterfactual import sanity
    report = sanity.run_gate(predictor, samples, cfg, ood=detector, embed=embed_fn)
    sanity.write_report(report, "eval/sanity_cf_wp_v1.md")
    report.passed                                   # False -> exit 1

`samples` yields `(kind, s_pre, s_post, action)` (optionally a fifth item, the factual
history passed to the Predictor; default `[]`). `samples_from(snapshots, actions,
windows, ...)` builds them from a match's stored states, actions and windows, and
`main()` does so for every match under `--states` with the map's real geometry, so the
gate certifies the same intervention path `analyze` runs (MED-16, 2026-10-19). A green
run writes `<model_dir>/sanity_cf.json`, which `registry.adopt` requires for a wp model.
Kinds and the edit each one makes:

    kill    undo_kill(s_post, action, s_pre)          P7.2.1  W moves toward the victim's side
    smoke   remove_utility(s_post, action)            P7.2.2  non-owner side not hurt, no wild swing
    flash   unflash(s_post, action, s_pre)            P7.2.3  blinded side not hurt
    plant   unplant(s_post, action, s_pre)            P7.2.4  T-side W falls
    freeze  freeze_position(s_post, actor, s_pre)     P7.2.5  small |dW|, no systematic sign
    order   action = list of members; every coalition (all 2^n up to 4 members, else
            empty + singletons + leave-one-out) built in reverse and forward order must
            give identical states and identical W                     P7.2.6
    OOD     with `ood` (an `OODDetector`) and `embed(snapshots) -> (N, d)`: per kind, the
            share of edited states scoring above the detector threshold <= ood_max_rate  P7.2.7

`W` is the Predictor's CT win probability; "before" is W(s_post) (factual), "after"
W(edited). Deltas are CT-positive unless a test says it orients them. A case whose
intervention raises `InterventionError` counts as a failed case (in every rate's
denominator) and is listed. A test with no samples cannot pass (the gate refuses to
certify what it did not see). The predictor is passed in, so this module never imports
torch.

Filled by: P7.2.1-P7.2.8.
"""
import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from engine.actions.types import Action
from engine.counterfactual.apply import INTERVENTION_FOR_TYPE, all_subsets, apply
from engine.counterfactual.interventions import (InterventionError, blinded_players, freeze_position,
                                                 remove_utility, undo_kill, unflash, unplant)
from engine.state.schema import Snapshot

GATE_FILE = "sanity_cf.json"       # written next to the model: the record adoption checks
# instant action type -> gate sample kind (a window member becomes one sample)
KIND_OF_TYPE = {"kill": "kill", "team_kill": "kill", "smoke_pop": "smoke", "flash": "flash",
                "plant_complete": "plant", "peek": "freeze"}

# ── constants ─────────────────────────────────────────────────────────────────

TESTS      = ("unkill", "smoke", "unflash", "unplant", "freeze", "order", "ood")
KIND_TEST  = {"kill": "unkill", "smoke": "smoke", "flash": "unflash", "plant": "unplant",
              "freeze": "freeze", "order": "order"}
N_WORST    = 20
BATCH      = 1024
ORDER_FULL_MAX = 4       # all 2^n coalitions up to this many members


@dataclass
class TestResult:
    name:       str
    n:          int = 0
    n_errors:   int = 0
    metrics:    dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    checks:     dict = field(default_factory=dict)
    passed:     bool = False
    worst:      list = field(default_factory=list)
    errors:     list = field(default_factory=list)
    note:       str = ""


@dataclass
class GateReport:
    tests:     dict
    passed:    bool
    n_samples: int
    created:   str
    model:     str | None = None

    def failed(self) -> list[str]:
        return [k for k, t in self.tests.items() if not t.passed]


# ── helpers ───────────────────────────────────────────────────────────────────

def _predict_win(predictor, snaps: list[Snapshot], hists: list) -> np.ndarray:
    out = []
    for i in range(0, len(snaps), BATCH):
        res = predictor.predict(snaps[i:i + BATCH], hists[i:i + BATCH])
        out.append(np.asarray(res["win"], dtype=np.float64).reshape(-1))
    return np.concatenate(out) if out else np.empty(0)


def _where(s: Snapshot) -> dict:
    m = s.meta
    return {"match": m.match_key if m else None, "round": m.round_num if m else s.ctx.round_num,
            "tick": m.tick if m else None}


def _side_sign(side: str) -> float:
    return 1.0 if side == "ct" else -1.0


def _rate(ok: list[bool], n_errors: int) -> float:
    total = len(ok) + n_errors
    return float(sum(ok)) / total if total else float("nan")


def _victim_side(s_post: Snapshot, kill: Action) -> str:
    i = s_post.index_of(kill.params.get("victim_id"))
    if i is None:
        raise InterventionError(f"victim {kill.params.get('victim_id')} not in s_post")
    return s_post.players[i].side


def _blinded_side(s_post: Snapshot, flash: Action) -> str:
    """Side of the blinded enemies (the thrower's opponents)."""
    return "t" if flash.actor_side == "ct" else "ct"


def _actor(action) -> int:
    return action if isinstance(action, (int, np.integer)) else action.actor_id_hash


def _edit(kind: str, s_pre: Snapshot, s_post: Snapshot, action, geometry=None) -> Snapshot:
    if kind == "kill":
        return undo_kill(s_post, action, s_pre, geometry=geometry)
    if kind == "smoke":
        return remove_utility(s_post, action, geometry=geometry)
    if kind == "flash":
        return unflash(s_post, action, s_pre, geometry=geometry)
    if kind == "plant":
        return unplant(s_post, action, s_pre, geometry=geometry)
    if kind == "freeze":
        return freeze_position(s_post, _actor(action), s_pre, geometry=geometry)
    raise ValueError(f"unknown sample kind {kind!r}")


def _order_subsets(members: list[Action]) -> list[frozenset]:
    if len(members) <= ORDER_FULL_MAX:
        return all_subsets(members)
    ids = [m.action_id for m in members]
    out = [frozenset()] + [frozenset([i]) for i in ids] + [frozenset(ids) - {i} for i in ids]
    return list(dict.fromkeys(out))


def _worst(cases: list[dict], key: Callable, reverse: bool = False) -> list[dict]:
    return sorted(cases, key=key, reverse=reverse)[:N_WORST]


# ── the gate ──────────────────────────────────────────────────────────────────

def run_gate(predictor, samples: Iterable, cfg, *, ood=None, embed: Callable | None = None,
             model: str | None = None, geometry=None) -> GateReport:
    """Run P7.2.1-P7.2.7 over `samples` with thresholds from `cfg.sanity`. One batched
    Predictor pass over every factual and edited state. `geometry` is passed to every
    intervention, as `analyze` does; without it edited LOS rows are NaN (MED-16)."""
    th = cfg.sanity
    results = {name: TestResult(name) for name in TESTS}
    cases: list[dict] = []           # one per evaluated edit: kind, refs into the batch
    snaps: list[Snapshot] = []
    hists: list = []
    edited_by_kind: dict[str, list[Snapshot]] = {}
    n_samples = 0

    def push(s: Snapshot, h) -> int:
        snaps.append(s)
        hists.append(h)
        return len(snaps) - 1

    for sample in samples:
        n_samples += 1
        kind, s_pre, s_post, action = sample[:4]
        hist = sample[4] if len(sample) > 4 else []
        test = KIND_TEST.get(kind)
        if test is None:
            raise ValueError(f"unknown sample kind {kind!r}; expected one of {sorted(KIND_TEST)}")
        res = results[test]
        res.n += 1
        try:
            if kind == "order":
                members = list(action)
                if len(members) < 2:
                    raise InterventionError("order sample needs >= 2 members")
                pairs = []
                for sub in _order_subsets(members):
                    a, _ = apply(s_pre, s_post, members, set(sub), order="reverse", geometry=geometry)
                    b, _ = apply(s_pre, s_post, members, set(sub), order="forward", geometry=geometry)
                    pairs.append((a == b, push(a, hist), push(b, hist)))
                cases.append({"kind": kind, "where": _where(s_post), "pairs": pairs})
                continue
            s_cf = _edit(kind, s_pre, s_post, action, geometry)
        except InterventionError as e:
            res.n_errors += 1
            if len(res.errors) < N_WORST:
                res.errors.append(f"{_where(s_post)}: {e}")
            continue
        edited_by_kind.setdefault(kind, []).append(s_cf)
        case = {"kind": kind, "where": _where(s_post), "i_before": push(s_post, hist),
                "i_after": push(s_cf, hist)}
        if kind == "kill":
            case["orient"] = _side_sign(_victim_side(s_post, action))
        elif kind == "smoke":
            case["orient"] = -_side_sign(action.actor_side)          # the non-owner side
        elif kind == "flash":
            case["orient"] = _side_sign(_blinded_side(s_post, action))
            case["n_blinded"] = len(blinded_players(action))
        elif kind == "plant":
            case["orient"] = -1.0                                       # T side
        cases.append(case)

    w = _predict_win(predictor, snaps, hists)

    by_kind: dict[str, list[dict]] = {}
    for c in cases:
        if c["kind"] == "order":
            c["state_ok"] = all(p[0] for p in c["pairs"])
            c["w_ok"] = all(w[p[1]] == w[p[2]] for p in c["pairs"])
            c["max_dw"] = max((abs(w[p[1]] - w[p[2]]) for p in c["pairs"]), default=0.0)
        else:
            c["w_before"], c["w_after"] = float(w[c["i_before"]]), float(w[c["i_after"]])
            c["delta"] = c["w_after"] - c["w_before"]
            c["signed"] = c["delta"] * c.get("orient", 1.0)
        by_kind.setdefault(c["kind"], []).append(c)

    _score_unkill(results["unkill"], by_kind.get("kill", []), th)
    _score_smoke(results["smoke"], by_kind.get("smoke", []), th)
    _score_unflash(results["unflash"], by_kind.get("flash", []), th)
    _score_unplant(results["unplant"], by_kind.get("plant", []), th)
    _score_freeze(results["freeze"], by_kind.get("freeze", []), th)
    _score_order(results["order"], by_kind.get("order", []))
    _score_ood(results["ood"], edited_by_kind, th, ood, embed)

    for r in results.values():
        if r.n == 0 and r.name != "ood":
            r.passed, r.note = False, "no samples of this kind: cannot certify"
    passed = all(r.passed for r in results.values())
    return GateReport(tests=results, passed=passed, n_samples=n_samples, model=model,
                      created=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _row(c: dict) -> dict:
    return {**c["where"], "w_before": round(c["w_before"], 6), "w_after": round(c["w_after"], 6),
            "delta": round(c["delta"], 6), "signed": round(c["signed"], 6)}


def _finish(r: TestResult, checks: dict, worst: list[dict]) -> None:
    r.checks = checks
    r.passed = bool(checks) and all(checks.values())
    r.worst = [_row(c) for c in worst]


def _score_unkill(r: TestResult, cs: list[dict], th) -> None:
    signed = [c["signed"] for c in cs]
    rate = _rate([s > 0 for s in signed], r.n_errors)
    med = float(np.median(np.abs([c["delta"] for c in cs]))) if cs else float("nan")
    r.metrics = {"toward_rate": rate, "median_abs_dw": med}
    r.thresholds = {"toward_rate >=": th.unkill_toward_rate, "median_abs_dw >=": th.unkill_median_abs}
    _finish(r, {"toward_rate": rate >= th.unkill_toward_rate, "median_abs_dw": med >= th.unkill_median_abs},
            _worst(cs, key=lambda c: c["signed"]))


def _score_smoke(r: TestResult, cs: list[dict], th) -> None:
    floor = _rate([c["signed"] >= th.smoke_floor for c in cs], r.n_errors)
    calm  = _rate([abs(c["delta"]) <= th.smoke_max_abs for c in cs], r.n_errors)
    r.metrics = {"floor_rate": floor, "max_abs_rate": calm}
    r.thresholds = {"floor": th.smoke_floor, "floor_rate >=": th.smoke_floor_rate,
                    "max_abs": th.smoke_max_abs, "max_abs_rate >=": th.smoke_max_abs_rate}
    _finish(r, {"floor_rate": floor >= th.smoke_floor_rate, "max_abs_rate": calm >= th.smoke_max_abs_rate},
            _worst(cs, key=lambda c: min(c["signed"] - th.smoke_floor, th.smoke_max_abs - abs(c["delta"]))))


def _score_unflash(r: TestResult, cs: list[dict], th) -> None:
    rate = _rate([c["signed"] >= th.unflash_floor for c in cs], r.n_errors)
    r.metrics = {"floor_rate": rate}
    r.thresholds = {"floor": th.unflash_floor, "floor_rate >=": th.unflash_rate}
    _finish(r, {"floor_rate": rate >= th.unflash_rate}, _worst(cs, key=lambda c: c["signed"]))


def _score_unplant(r: TestResult, cs: list[dict], th) -> None:
    rate = _rate([c["signed"] < 0 for c in cs], r.n_errors)
    r.metrics = {"t_down_rate": rate}
    r.thresholds = {"t_down_rate >=": th.unplant_down_rate}
    _finish(r, {"t_down_rate": rate >= th.unplant_down_rate}, _worst(cs, key=lambda c: c["signed"], reverse=True))


def _score_freeze(r: TestResult, cs: list[dict], th) -> None:
    deltas = [c["delta"] for c in cs]
    calm = _rate([abs(d) <= th.freeze_max_abs for d in deltas], r.n_errors)
    mean = float(np.mean(deltas)) if deltas else float("nan")
    r.metrics = {"max_abs_rate": calm, "mean_dw": mean}
    r.thresholds = {"max_abs": th.freeze_max_abs, "max_abs_rate >=": th.freeze_max_abs_rate,
                    "mean_dw abs <=": th.freeze_mean_abs}
    _finish(r, {"max_abs_rate": calm >= th.freeze_max_abs_rate, "mean_dw": abs(mean) <= th.freeze_mean_abs},
            _worst(cs, key=lambda c: abs(c["delta"]), reverse=True))


def _score_order(r: TestResult, cs: list[dict]) -> None:
    ok = [c["state_ok"] and c["w_ok"] for c in cs]
    rate = _rate(ok, r.n_errors)
    r.metrics = {"identical_rate": rate, "n_coalitions": sum(len(c["pairs"]) for c in cs)}
    r.thresholds = {"identical_rate ==": 1.0}
    r.checks = {"identical_rate": rate == 1.0}
    r.passed = bool(cs) and rate == 1.0
    bad = sorted(cs, key=lambda c: (c["state_ok"] and c["w_ok"], -c["max_dw"]))[:N_WORST]
    r.worst = [{**c["where"], "states_identical": c["state_ok"], "w_identical": c["w_ok"],
                "max_abs_dw": c["max_dw"]} for c in bad]


def _score_ood(r: TestResult, edited: dict, th, ood, embed) -> None:
    r.thresholds = {"flag_rate <=": th.ood_max_rate, "percentile": th.ood_percentile, "k": th.ood_k}
    if ood is None or embed is None:
        r.note = "no OOD detector / embedding function given: P7.2.7 not run"
        r.passed = False
        return
    for kind, states in sorted(edited.items()):
        scores = np.asarray(ood.score(embed(states)), dtype=np.float64)
        rate = float(ood.flag_rate(scores))
        r.n += len(states)
        r.metrics[f"{kind}_flag_rate"] = rate
        r.checks[kind] = rate <= th.ood_max_rate
        top = np.argsort(-scores)[:N_WORST]
        r.worst += [{**_where(states[i]), "kind": kind, "score": round(float(scores[i]), 6),
                     "threshold": round(float(ood.threshold), 6)} for i in top]
    r.worst = sorted(r.worst, key=lambda c: -c["score"])[:N_WORST]
    r.passed = bool(r.checks) and all(r.checks.values())


# ── report ────────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4f}"
    return str(v)


def write_report(report: GateReport, path) -> Path:
    """Markdown report (P7.2.8): verdict, per-test rates vs thresholds, the worst 20
    cases per test, intervention errors."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Counterfactual sanity gate (P7.2)", "",
             f"- verdict: **{'PASS' if report.passed else 'FAIL'}**",
             f"- model: {report.model or 'unspecified'}",
             f"- samples: {report.n_samples}",
             f"- created: {report.created}", ""]
    if report.failed():
        lines += [f"- failed tests: {', '.join(report.failed())}", ""]
    lines += ["| test | n | errors | metric | value | threshold | pass |", "|---|---|---|---|---|---|---|"]
    for t in report.tests.values():
        if not t.metrics:
            lines.append(f"| {t.name} | {t.n} | {t.n_errors} | - | - | - | {'yes' if t.passed else 'no'} |")
        for k, v in t.metrics.items():
            base = "flag_rate" if k.endswith("_flag_rate") else k
            thr = ", ".join(f"{a} {_fmt(b)}" for a, b in t.thresholds.items() if a.split()[0] == base) or "-"
            ok = t.checks.get(k, t.checks.get(k.replace("_flag_rate", ""), None))
            lines.append(f"| {t.name} | {t.n} | {t.n_errors} | {k} | {_fmt(v)} | {thr} | "
                         f"{'-' if ok is None else ('yes' if ok else 'no')} |")
    for t in report.tests.values():
        lines += ["", f"## {t.name}", ""]
        if t.note:
            lines += [f"_{t.note}_", ""]
        lines += [f"thresholds: {', '.join(f'{k} {_fmt(v)}' for k, v in t.thresholds.items())}", ""]
        if t.worst:
            cols = list(t.worst[0].keys())
            lines += [f"worst {len(t.worst)} cases:", "", "| " + " | ".join(cols) + " |",
                      "|" + "---|" * len(cols)]
            lines += ["| " + " | ".join(_fmt(c.get(k)) for k in cols) + " |" for c in t.worst]
        if t.errors:
            lines += ["", "intervention errors (first 20):", ""] + [f"- {e}" for e in t.errors]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ── CLI ───────────────────────────────────────────────────────────────────────

# ── sampling from stored states / actions / windows (P7.2.8) ─────────────────

def samples_from(snapshots: list, actions: list, windows: list, *, n_events: int | None = None,
                 n_drift: int | None = None, rng: random.Random | None = None) -> list[tuple]:
    """Gate samples of one match: every window member whose type has a gate kind gives
    `(kind, s_pre, s_post, action)` with s_pre / s_post the snapshots at the window's
    start and end; every window with >= 2 interventionable members gives an `order`
    sample; consecutive factual snapshots outside windows give `freeze` samples for a
    player who moved between them (drift). `n_events` / `n_drift` cap the two groups
    (uniform without replacement, seeded by `rng`)."""
    rng = rng or random.Random(0)
    by_id = {a.action_id: a for a in actions}
    rounds: dict[int, dict[int, Snapshot]] = {}
    for s in snapshots:
        rounds.setdefault(s.meta.round_num, {})[int(s.meta.tick)] = s
    events, drift = [], []
    covered: dict[int, set[int]] = {}
    for w in windows:
        by_tick = rounds.get(w.round_num, {})
        s_pre, s_post = by_tick.get(int(w.start_tick)), by_tick.get(int(w.end_tick))
        if s_pre is None or s_post is None:
            continue
        covered.setdefault(w.round_num, set()).update(range(int(w.start_tick), int(w.end_tick) + 1))
        members = [by_id[i] for i in w.member_action_ids if i in by_id]
        for a in members:
            kind = KIND_OF_TYPE.get(a.type)
            if kind is not None:
                events.append((kind, s_pre, s_post, a))
        order = [a for a in members if a.type in INTERVENTION_FOR_TYPE]
        if len(order) >= 2:
            events.append(("order", s_pre, s_post, order))
    for r, by_tick in rounds.items():
        ticks = sorted(by_tick)
        cov = covered.get(r, set())
        for t0, t1 in zip(ticks, ticks[1:]):
            if any(t in cov for t in (t0, t1)):
                continue
            a, b = by_tick[t0], by_tick[t1]
            pos_b = {p.player_id: (p.x, p.y, p.z) for p in b.players if p.alive == 1}
            movers = [p.player_id for p in a.players
                      if p.alive == 1 and p.player_id in pos_b and (p.x, p.y, p.z) != pos_b[p.player_id]]
            if movers:
                drift.append(("freeze", a, b, rng.choice(sorted(movers))))
    if n_events is not None and len(events) > n_events:
        events = rng.sample(events, n_events)
    if n_drift is not None and len(drift) > n_drift:
        drift = rng.sample(drift, n_drift)
    return events + drift


def _load_match(states_path: Path):
    """(snapshots, actions, windows, map_version) of one stored match (layout P5.1)."""
    import polars as pl

    from engine.actions.windows import Window
    from engine.state.schema import frame_to_snapshots
    sp = Path(states_path)
    snaps = frame_to_snapshots(pl.read_parquet(sp))
    kind_dir = sp.parents[2]                                     # <root>/states
    a_path = kind_dir.parent / "actions" / sp.parent.parent.name / sp.parent.name / sp.name
    w_path = kind_dir.parent / "windows" / sp.parent.parent.name / sp.parent.name / sp.name
    if not a_path.is_file() or not w_path.is_file():
        raise FileNotFoundError(f"{sp.stem}: actions / windows parquet missing next to the states")
    acts = [Action.from_dict(r) for r in pl.read_parquet(a_path).iter_rows(named=True)]
    wins = [Window(window_id=r["window_id"], round_num=r["round_num"], start_tick=r["start_tick"],
                   end_tick=r["end_tick"], member_action_ids=tuple(r["member_action_ids"] or ()))
            for r in pl.read_parquet(w_path).iter_rows(named=True)]
    mv = snaps[0].ctx.map_version if snaps else None
    return snaps, acts, wins, mv


def _geometry_for(map_version: str | None):
    from engine.maps import assets, callouts
    from engine.state.geometry import MapGeometry
    if not map_version:
        raise ValueError("stored states carry no map_version")
    map_name = map_version.split("@")[0]
    try:
        cos = callouts.load(map_name)
    except (FileNotFoundError, ValueError):
        cos = []
    return MapGeometry.from_assets(assets.ensure(map_name), cos, map_version)


def write_record(model_dir: Path, report: GateReport) -> Path:
    """`<model_dir>/sanity_cf.json`: the verdict the registry checks before adopting."""
    p = Path(model_dir) / GATE_FILE
    p.write_text(json.dumps({"passed": bool(report.passed), "n_samples": report.n_samples,
                             "created": report.created, "model": report.model,
                             "tests": {k: bool(t.passed) for k, t in report.tests.items()},
                             "failed": report.failed()}, indent=2, sort_keys=True), encoding="utf-8")
    return p


def gate_passed(model_dir: Path) -> bool | None:
    """True / False from `<model_dir>/sanity_cf.json`, None when the gate never ran."""
    p = Path(model_dir) / GATE_FILE
    if not p.is_file():
        return None
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("passed"))
    except (OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    """`engine sanity --counterfactual`. Exit 0 = gate green, 1 = a test failed,
    2 = prerequisites missing (a trained, calibrated model; stored states, actions and
    windows under --states; map assets)."""
    from engine import config, paths
    from engine.data import layout
    ap = argparse.ArgumentParser(prog="engine sanity --counterfactual", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="WP model name under data/engine/models, or a model directory")
    ap.add_argument("--states", default=None, help=f"engine data root holding states/ actions/ windows/ "
                                                   f"(default {paths.ENGINE_DIR})")
    ap.add_argument("--n-events", type=int, default=None, help="events to sample (default config sanity.cf_n_events)")
    ap.add_argument("--n-drift", type=int, default=None, help="drift intervals to sample (default sanity.cf_n_drift)")
    ap.add_argument("--max-matches", type=int, default=None, help="stop after this many stored matches")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=None, help="markdown report path (default data/engine/eval/sanity_cf_<model>.md)")
    args = ap.parse_args(argv)
    if not args.model:
        print("sanity gate needs --model <wp_vN or model dir>")
        return 2
    model_dir = Path(args.model) if Path(args.model).is_dir() else paths.MODELS_DIR / args.model
    if not model_dir.is_dir():
        print(f"model directory not found: {model_dir.name}")
        return 2
    root = Path(args.states) if args.states else paths.ENGINE_DIR
    states = layout.find("states", base=root) if (root / "states").is_dir() else []
    if not states:
        print(f"no stored states under {root / 'states'} (run build-states / build-actions first)")
        return 2
    cfg = config.load()
    try:
        from engine.model.predict import Predictor
        predictor = Predictor(model_dir, "cpu" if args.cpu else None, require_calibration=True)
    except FileNotFoundError as exc:
        print(f"model directory incomplete: {Path(str(exc)).name}")
        return 2
    rng = random.Random(args.seed)
    n_events = args.n_events if args.n_events is not None else cfg.sanity.cf_n_events
    n_drift  = args.n_drift if args.n_drift is not None else cfg.sanity.cf_n_drift
    samples: list[tuple] = []
    geometry = None
    for i, sp in enumerate(states):
        if args.max_matches is not None and i >= args.max_matches:
            break
        try:
            snaps, acts, wins, mv = _load_match(sp)
        except FileNotFoundError as exc:
            print(f"skipped: {exc}")
            continue
        if geometry is None or getattr(geometry, "map_version", None) != mv:
            try:
                geometry = _geometry_for(mv)
            except Exception as exc:              # MapAssetError, missing release: a prerequisite
                print(f"map geometry unavailable for {mv}: {type(exc).__name__}: {exc}")
                return 2
        samples += samples_from(snaps, acts, wins, rng=rng)
    if not samples:
        print("stored states yielded no gate samples (no windows with interventionable members)")
        return 2
    events = [s for s in samples if s[0] != "freeze"]
    drift  = [s for s in samples if s[0] == "freeze"]
    if len(events) > n_events:
        events = rng.sample(events, n_events)
    if len(drift) > n_drift:
        drift = rng.sample(drift, n_drift)
    samples = events + drift
    ood = embed = None
    if hasattr(predictor, "encode"):               # P7.2.7: a detector built here on the factual states
        from engine.counterfactual.ood import OODDetector
        factual = [s for smp in samples for s in (smp[1], smp[2])]
        rng.shuffle(factual)
        cut = max(1, len(factual) * 4 // 5)
        embed = lambda snaps: predictor.encode(snaps)          # noqa: E731
        ood = OODDetector(k=cfg.sanity.ood_k, percentile=cfg.sanity.ood_percentile).fit(embed(factual[:cut]))
        if len(factual) > cut:
            ood.calibrate(embed(factual[cut:]))
    report = run_gate(predictor, samples, cfg, ood=ood, embed=embed, model=model_dir.name, geometry=geometry)
    out = Path(args.out) if args.out else paths.EVAL_DIR / f"sanity_cf_{model_dir.name}.md"
    write_report(report, out)
    rec = write_record(model_dir, report)
    print(f"{'PASS' if report.passed else 'FAIL'}: {report.n_samples} samples; report {out}; record {rec}")
    if not report.passed:
        print("failed tests: " + ", ".join(report.failed()))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
