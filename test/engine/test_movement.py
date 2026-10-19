"""
Drift intervals (P8.4): the mover gets the movement share, a stationary player none;
efficiency over v(M) − v(∅); the clock remainder identity per interval and per round;
leave-one-out below `drift_threshold_wp` and Shapley at / above it; sampled Shapley for
more than `shapley_exact_max` movers; movement shares go to the covering rotation
(walking story, P8.3.2) or to a synthetic `positioning` row; the P8.4.3 metric.

    .venv\\Scripts\\python.exe test\\engine\\test_movement.py
"""
import math
import statistics
import sys
from dataclasses import replace

from _harness import check, close, section, summary
from _cf_states import CT_IDS, T_IDS, action, home, make_snapshot
from _attr_states import make_round, round_hazard, round_win
from _stubs import StubPredictor

from engine import config
from engine.attribution import engine, movement
from engine.attribution.clock import ClockBuckets, interval_identity, round_identity

CFG = config.load()
THR = CFG.attribution.drift_threshold_wp


def snap(t, pos=None, dead=(), kinds=("grid",)):
    return make_snapshot(tick=t, positions=pos or {}, dead=dead, kinds=kinds)


def dx(s, pid) -> float:
    p = s.players[s.index_of(pid)]
    return (p.x - home(pid)[0]) / 100.0 if p.alive else 0.0


# ── movers ────────────────────────────────────────────────────────────────────

section("movers")
s0 = snap(4936, {103: (0.0, 200.0, 0.0)})
s1 = snap(5000, {102: (300.0, 100.0, 0.0), 103: (0.0, 200.0, 0.0), 104: (20.0, 300.0, 0.0),
                 105: (500.0, 400.0, 0.0)}, dead={105})
check("B (102, 300 u) is a mover", 102 in movement.movers(s0, s1, CFG.actions.movement_min_units))
mv = movement.movers(s0, s1, CFG.actions.movement_min_units)
check("stationary A (101) is not a mover", 101 not in mv)
check("20 u < movement_min_units is not a mover", 104 not in mv)
check("a dead player is not a mover", 105 not in mv)
check("displacement is 3D Euclidean", close(movement.displacement(s0, s1, 102), 300.0, 1e-9))

# ── one mover, engine level ───────────────────────────────────────────────────

section("walking story: B moves, A stands (engine)")


def walk_win(s, h=None) -> float:
    return 0.5 + 0.0001 * (s.players[s.index_of(102)].x) - 0.001 * s.ctx.time_into_round_s


snaps = [snap(4872, kinds=("grid",)), snap(4936, {102: (150.0, 100.0, 0.0)}),
         snap(5000, {102: (300.0, 100.0, 0.0)}, kinds=("grid", "terminal"))]
rot = action("rotation", 102, tick=4872, kind="durative")
rot.window_end_tick = 5000
ra = engine.attribute_round(snaps, [rot], [], StubPredictor(win=walk_win), CFG, pin_terminal=False)
rows = {sa.actor_id_hash: sa for sa in ra.actions}
drift = [c for c in rows[102].components if c["source"] == "drift"]
check("B's rotation gets the movement shares", rows[102].type == "rotation" and len(drift) == 2)
check("B's rotation value = Σ of its movement shares over its window (P8.3.2)",
      close(rows[102].V, math.fsum(c["phi"] for c in drift), 1e-15) and close(rows[102].V, 0.03, 1e-12),
      f"V={rows[102].V:+.6f}")
check("stationary A gets nothing (no row at all)", 101 not in rows)
check("no synthetic positioning row when a rotation covers the interval",
      not any("synthetic" in sa.flags for sa in ra.actions))
dW = ra.w_end - ra.w_start
check("clock gets the time term: clock_ct − clock_t = ΔW − Σ φ",
      close(ra.clock_ct - ra.clock_t, dW - rows[102].V_raw, 1e-12)
      and close(ra.clock_ct - ra.clock_t, -0.001 * (128 / 128), 1e-9), f"clock={ra.clock_ct - ra.clock_t:+.6f}")
check("conservation ≤ 1e-9", abs(ra.conservation_residual) <= 1e-9)
check("per-round clock identity", ra.stats["clock_identity_ok"])

ra2 = engine.attribute_round(snaps, [], [], StubPredictor(win=walk_win), CFG, pin_terminal=False)
pos = [sa for sa in ra2.actions if sa.type == "positioning"]
check("without a movement action: synthetic positioning row for B",
      len(pos) == 1 and pos[0].actor_id_hash == 102 and "synthetic" in pos[0].flags)
check("unassigned movement recorded in stats", close(ra2.stats["unassigned_movement"].get(102), 0.03, 1e-12))

# ── interval-level: LOO vs Shapley at the threshold ───────────────────────────

section("LOO below drift_threshold_wp, Shapley at / above it")
a, b, c = 0.004, 0.006, 0.003


def inter(s, h=None) -> float:
    d1, d2 = dx(s, 101), dx(s, 102)
    return 0.5 + a * d1 + b * d2 + c * d1 * d2


s0 = snap(4936)
s1 = snap(5000, {101: (200.0, 0.0, 0.0), 102: (300.0, 100.0, 0.0)})
d1, d2 = 2.0, 3.0


def solve(dW):
    plan = movement.plan_interval(s0, s1, dW, CFG, seed=1)
    table = {S: inter(movement.frozen_state(s1, s0, plan.movers, S)) for S in plan.coalitions}
    return plan, movement.solve_interval(plan, table.__getitem__, dW, CFG)


plan, res = solve(THR - 1e-6)
check("|ΔW| just below threshold -> loo", plan.method == "loo" and res.method == "loo")
total = a * d1 + b * d2 + c * d1 * d2
raw = {101: a * d1 + c * d1 * d2, 102: b * d2 + c * d1 * d2}
r = total - sum(raw.values())
want = {p: raw[p] + r * abs(raw[p]) / sum(abs(x) for x in raw.values()) for p in raw}
check("loo values = leave-one-out with the residual redistributed ∝ |φ|",
      all(close(res.phi[p], want[p], 1e-12) for p in raw), str(res.phi))
check("loo is efficient: Σ φ = v(M) − v(∅)", close(sum(res.phi.values()), total, 1e-12))
check("loo residual (before enforcement) = |c·d1·d2|", close(res.residual, c * d1 * d2, 1e-12))
plan, res = solve(THR)
check("|ΔW| at the threshold -> exact Shapley", plan.method == "exact" and len(plan.coalitions) == 4)
check("exact values split the interaction equally",
      close(res.phi[101], a * d1 + c * d1 * d2 / 2, 1e-12) and close(res.phi[102], b * d2 + c * d1 * d2 / 2, 1e-12))
check("clock remainder identity Σ φ + clock = ΔW (1e-9)", interval_identity(res.phi.values(), res.clock, THR))

section("LOO vs Shapley switch through the engine")
for k, want_m in ((0.0099, "loo"), (0.0101, "exact")):
    st = [snap(4936), snap(5000, {101: (100.0, 0.0, 0.0), 102: (100.0, 100.0, 0.0)}, kinds=("grid", "terminal"))]
    ra = engine.attribute_round(st, [], [], StubPredictor(win=lambda s, h, k=k: 0.5 + k * (dx(s, 101) + dx(s, 102))),
                                CFG, pin_terminal=False)
    meth = {cc["method"] for sa in ra.actions for cc in sa.components}
    check(f"ΔW = {2 * k:.4f} -> {want_m}", meth == {want_m} and ra.stats["drift_methods"] == {want_m: 1}, str(meth))
    check(f"ΔW = {2 * k:.4f}: each mover gets {k:.4f}", all(close(sa.V_raw, k, 1e-12) for sa in ra.actions))

section("more than shapley_exact_max movers -> sampled")
movers7 = list(CT_IDS) + list(T_IDS[:2])
pos7 = {p: (home(p)[0] + 100.0 * (i + 1), home(p)[1], 0.0) for i, p in enumerate(movers7)}
add7 = lambda s, h=None: 0.3 + 0.002 * sum(dx(s, p) for p in movers7)       # noqa: E731
st = [snap(4936), snap(5000, pos7, kinds=("grid", "terminal"))]
ra = engine.attribute_round(st, [], [], StubPredictor(win=add7), CFG, pin_terminal=False)
check("7 movers above threshold -> sampled", ra.stats["drift_methods"] == {"sampled": 1}, str(ra.stats["drift_methods"]))
check("sampled (antithetic, additive) gives each mover its exact share",
      all(close(sa.V_raw, 0.002 * dx(st[1], sa.actor_id_hash), 1e-12) for sa in ra.actions))
check("conservation within the sampled tolerance (1e-6)", abs(ra.conservation_residual) <= 1e-6)
fewer = replace(CFG, attribution=replace(CFG.attribution, shapley_exact_max=7))
ra = engine.attribute_round(st, [], [], StubPredictor(win=add7), fewer, pin_terminal=False)
check("same interval with shapley_exact_max 7 -> exact", ra.stats["drift_methods"] == {"exact": 1})

# ── clock helpers ─────────────────────────────────────────────────────────────

section("clock buckets")
cb = ClockBuckets()
for x in (0.03, -0.01, 0.0, -0.005):
    cb.add(x)
check("clock_ct accumulates positives, clock_t magnitudes of negatives",
      close(cb.clock_ct, 0.03, 1e-15) and close(cb.clock_t, 0.015, 1e-15))
check("per-round identity clock_ct − clock_t = Σ remainders", round_identity(cb))
try:
    cb.add(float("nan"))
    nan_ok = False
except ValueError:
    nan_ok = True
check("a NaN remainder is an error", nan_ok)

# ── P8.4.3 metric ─────────────────────────────────────────────────────────────

section("movement-credit variance (P8.4.3)")
snaps, actions, windows, _ = make_round()
ra = engine.attribute_round(snaps, actions, windows, StubPredictor(win=round_win, hazard=round_hazard), CFG)
m = movement.movement_credit_variance([ra])
shares = {}
for sa in ra.actions:
    for cc in sa.components:
        if cc["source"] == "drift":
            shares[sa.actor_id_hash] = shares.get(sa.actor_id_hash, 0.0) + cc["phi"]
check("metric = std of per-player movement shares in the round",
      m["n_rounds"] == 1 and close(m["mean"], statistics.pstdev(shares.values()), 1e-15), f"{m['mean']:.6f}")
check("story: 103 (rotation) and 204 (positioning) are the credited movers", set(shares) == {103, 204})

sys.exit(summary())
