"""
Movement credit in drift intervals (P8.4.1): leave-one-out below
`drift_threshold_wp`, Shapley over movers above it, via `freeze_position`.

    from engine.attribution import movement
    plan = movement.plan_interval(s0, s1, dW, cfg, seed)       # movers, method, coalitions
    states = {S: movement.frozen_state(s1, s0, plan.movers, S) for S in plan.coalitions}
    ...one batched predictor call over every interval's states...
    res = movement.solve_interval(plan, v, dW)                 # φ per player + clock

`v(S)` is W of `s1` with every mover NOT in `S` frozen to its `s0` position, so
`v(M) = W(s1)` and `v(∅)` = W(s1 with every mover frozen). Both methods are efficient
over `v(M) − v(∅)`; the interval's clock share is the rest, `ΔW − Σ φ_p`
(= `v(∅) − W(s0)`: time passing and whatever else changed with nobody moving).

- `|ΔW| < drift_threshold_wp`: leave-one-out `φ_p = v(M) − v(M \\ {p})`, then
  efficiency-enforced (B.3) against `v(M) − v(∅)`; method `loo`.
- otherwise: exact Shapley over `M` when `|M| <= shapley_exact_max`, else sampled.

`movers` already enforces `freeze_position`'s preconditions (alive in both, positions
known), so a player that cannot be frozen is never a mover.

Filled by: P8.4.1, P8.4.3.
"""
import math
import statistics
from dataclasses import dataclass, field

from engine.attribution import shapley
from engine.counterfactual.interventions import freeze_position

# ── movers ────────────────────────────────────────────────────────────────────

def _pos(p):
    if p.x is None or p.y is None or p.z is None:
        return None
    v = (float(p.x), float(p.y), float(p.z))
    return None if any(math.isnan(c) for c in v) else v


def displacement(s0, s1, player_id) -> float | None:
    """3D distance a player moved between two snapshots; None when unknown."""
    i, j = s0.index_of(player_id), s1.index_of(player_id)
    if i is None or j is None:
        return None
    a, b = _pos(s0.players[i]), _pos(s1.players[j])
    if a is None or b is None:
        return None
    return math.dist(a, b)


def movers(s0, s1, min_units: float) -> list:
    """Alive (in both) players whose displacement is >= `min_units`, sorted by id."""
    out = []
    for p in s1.players:
        if p.alive != 1:
            continue
        i = s0.index_of(p.player_id)
        if i is None or s0.players[i].alive != 1:
            continue
        d = displacement(s0, s1, p.player_id)
        if d is not None and d >= min_units:
            out.append(p.player_id)
    return sorted(out)


def frozen_state(s1, s0, all_movers, moving: frozenset, *, geometry=None):
    """`s1` with every mover not in `moving` frozen to its `s0` position."""
    s = s1
    for pid in sorted(all_movers):
        if pid not in moving:
            s = freeze_position(s, pid, s0, geometry=geometry)
    return s if s is not s1 else s1.copy()


# ── plan / solve ──────────────────────────────────────────────────────────────

@dataclass
class MovementPlan:
    movers:     list
    method:     str | None                      # loo | exact | sampled | None (no movers)
    coalitions: list[frozenset]                 # every coalition to evaluate, v(M) included
    seed:       int = 0
    skipped:    list = field(default_factory=list)


@dataclass
class MovementResult:
    phi:      dict
    clock:    float
    method:   str | None
    residual: float
    n_movers: int


def plan_interval(s0, s1, dW: float, cfg, seed: int) -> MovementPlan:
    """Movers and the coalitions their attribution needs (decided by |ΔW|)."""
    a = cfg.attribution
    ok, skipped = movers(s0, s1, cfg.actions.movement_min_units), []
    if not ok:
        return MovementPlan([], None, [], seed, skipped)
    full = frozenset(ok)
    if abs(dW) < a.drift_threshold_wp:
        cs = [full, frozenset()] + [full - {p} for p in ok]
        seen, uniq = set(), []
        for c in cs:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return MovementPlan(ok, "loo", uniq, seed, skipped)
    method = shapley.method_for(len(ok), a)
    return MovementPlan(ok, method, shapley.required_coalitions(ok, a, seed), seed, skipped)


def loo(movers_: list, v) -> tuple[dict, float]:
    """Leave-one-out φ_p = v(M) − v(M \\ {p}), efficiency-enforced over v(M) − v(∅).
    Returns (φ, |residual| before enforcement)."""
    full = frozenset(movers_)
    phi = {p: v(full) - v(full - {p}) for p in sorted(movers_)}
    return shapley.enforce_efficiency(phi, v(full) - v(frozenset()))


def solve_interval(plan: MovementPlan, v, dW: float, cfg) -> MovementResult:
    """φ per mover and the clock remainder `ΔW − Σ φ` for one drift interval."""
    if not plan.movers:
        return MovementResult({}, float(dW), None, 0.0, 0)
    if plan.method == "loo":
        phi, resid = loo(plan.movers, v)
    else:
        phi, _m, resid = shapley.shapley(plan.movers, v, cfg.attribution, plan.seed)
    clock = float(dW) - math.fsum(phi.values())
    return MovementResult(phi, clock, plan.method, resid, len(plan.movers))


# ── P8.4.3 ablation metric ────────────────────────────────────────────────────

def movement_credit_variance(round_attributions) -> dict:
    """Standard deviation of per-player movement shares per round (P6.7 rate
    ablation): `{"per_round": {(match, round): std}, "mean": ..., "n_rounds": ...}`.
    A player's share is the sum of the raw movement components credited to them;
    rounds with fewer than two credited players are skipped (std undefined)."""
    per = {}
    for ra in round_attributions:
        by_player: dict = {}
        for sa in ra.actions:
            m = math.fsum(c["phi"] for c in sa.components if c.get("source") == "drift")
            if m != 0.0 or any(c.get("source") == "drift" for c in sa.components):
                by_player[sa.actor_id_hash] = by_player.get(sa.actor_id_hash, 0.0) + m
        if len(by_player) >= 2:
            per[(ra.match_key, ra.round_num)] = statistics.pstdev(by_player.values())
    vals = list(per.values())
    return {"per_round": per, "mean": statistics.fmean(vals) if vals else None, "n_rounds": len(vals)}
