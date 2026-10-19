"""
Condition C1 (P8.2.4) through the full attribution engine, plus pop / fade (P8.3.1) and
the actor sign (P8.2.5).

`null_action_same_tick` runs on a kill window (101 kills 202 at 5000) with an added
decoy pop 4,000+ units from everyone, in two stub variants:
  1. a stub W that values smokes and alive counts but ignores decoys entirely;
  2. `_cf_states.RuleStub(good_win)`, the alive-count / blind rule, which ignores all
     utility, so the decoy is null there too.
Both must give `φ_decoy == 0 ± 1e-9` and `φ_kill == ΔW ± 1e-9`. The same holds in a
5-member window (`_cf_states.make_window`) with exact and with forced sampled Shapley.

The production bound on the real model (`|φ_decoy| ≤ 0.002` and `|φ_kill − ΔW| ≤ 0.002`
over 200 real kill windows) needs a trained model and real windows. It is DEFERRED to
real data: `--real` runs only that variant, which prints why and exits 2; the default
run records it as a skip and exits 0 when the stub variants pass.

    .venv\\Scripts\\python.exe test\\engine\\test_attribution_shapley.py [--real]
"""
import math
import sys
from dataclasses import replace

from _harness import check, section, skip, summary
from _cf_states import RuleStub, action, good_win, make_snapshot, make_window, pair_states
from _stubs import StubPredictor

from engine import config
from engine.attribution import engine

CFG   = config.load()
DECOY = {"x": -3000.0, "y": -3000.0, "z": 0.0, "owner": 105, "start_tick": 5000}

if "--real" in sys.argv:
    print("null_action_same_tick (real model): DEFERRED to real data — needs a trained "
          "Predictor and 200 real kill windows (|φ_decoy| ≤ 0.002, |φ_kill − ΔW| ≤ 0.002).")
    sys.exit(2)


def comp_phi(ra, source=None, pseudo_type=None, action_id=None) -> float:
    out = []
    for sa in ra.actions:
        for c in sa.components:
            if source and c.get("source") != source:
                continue
            if pseudo_type and c.get("pseudo_type") != pseudo_type:
                continue
            if action_id and sa.action_id != action_id:
                continue
            out.append(c["phi"])
    return math.fsum(out)


def by_id(ra, aid):
    return next(sa for sa in ra.actions if sa.action_id == aid)


def ignores_decoys(s, h=None) -> float:
    w = good_win(s) + 0.05 * sum(u.kind == "smoke" and u.owner_side == "ct" for u in s.utility)
    w -= 0.03 * sum(u.kind == "fire" for u in s.utility)
    return min(0.98, max(0.02, w))


# ── C1: kill + far decoy, one window ──────────────────────────────────────────

def kill_decoy_round():
    s_pre, s_post = pair_states({}, {"dead": {202}, "kills": {101: 1}, "damage": {101: 100},
                                     "decoys": [DECOY]})
    decoy = action("decoy", 105, kind="durative", x=DECOY["x"], y=DECOY["y"], z=0.0)
    kill  = action("kill", 101, victim_id=202, victim_health_before=100.0)
    pop   = action("decoy_start", 105, parent_action_id=decoy.action_id)
    w = {"window_id": "c1", "round_num": 7, "start_tick": 4999, "end_tick": 5001,
         "member_action_ids": [kill.action_id, pop.action_id]}
    return [s_pre, s_post], [decoy, kill, pop], [w], kill, decoy


def null_action_same_tick(label: str, predictor) -> None:
    snaps, acts, wins, kill, decoy = kill_decoy_round()
    far = min(math.dist((p.x, p.y), (DECOY["x"], DECOY["y"])) for p in snaps[1].players)
    ra = engine.attribute_round(snaps, acts, wins, predictor, CFG, pin_terminal=False)
    dW = ra.w_end - ra.w_start
    f_decoy = comp_phi(ra, source="pop", action_id=decoy.action_id)
    f_kill  = comp_phi(ra, source="window", action_id=kill.action_id)
    check(f"[{label}] decoy is ≥ 2000 u from everyone", far >= 2000, f"{far:.0f} u")
    check(f"[{label}] window ΔW is non-zero (the kill moved W)", abs(dW) > 0.01, f"ΔW={dW:+.4f}")
    check(f"[{label}] φ_decoy == 0 ± 1e-9", abs(f_decoy) <= 1e-9, f"{f_decoy:.3e}")
    check(f"[{label}] φ_kill == ΔW ± 1e-9", abs(f_kill - dW) <= 1e-9, f"{f_kill:+.6f} vs {dW:+.6f}")
    check(f"[{label}] decoy action V == 0", abs(by_id(ra, decoy.action_id).V) <= 1e-9)
    check(f"[{label}] conservation residual ≤ 1e-9", abs(ra.conservation_residual) <= 1e-9)


section("C1 null_action_same_tick — variant 1: stub that ignores decoys")
null_action_same_tick("ignores-decoys", StubPredictor(win=ignores_decoys))
section("C1 null_action_same_tick — variant 2: RuleStub alive-count rule (ignores all utility)")
null_action_same_tick("RuleStub", RuleStub(good_win))
section("C1 null_action_same_tick — real model")
skip("real-model bound |φ_decoy| ≤ 0.002 over 200 windows", "deferred to real data (run with --real: exits 2)")

# ── C1 in a 5-member window, exact and sampled ────────────────────────────────

section("C1 in a 5-member window (kill, flash, smoke pop, decoy start, damage)")
s_pre, s_post, members = make_window()
parents = {}
for m in members:
    if m.type in ("smoke_pop", "decoy_start"):
        par = action("smoke" if m.type == "smoke_pop" else "decoy", m.actor_id_hash, kind="durative")
        m.params["parent_action_id"] = par.action_id
        parents[m.type] = par
wins = [{"window_id": "w5", "round_num": 7, "start_tick": 4999, "end_tick": 5001,
         "member_action_ids": [m.action_id for m in members]}]
acts = members + list(parents.values())
for label, cfg in (("exact", CFG), ("sampled", replace(CFG, attribution=replace(CFG.attribution, shapley_exact_max=2)))):
    for pname, pred in (("ignores-decoys", StubPredictor(win=ignores_decoys)), ("RuleStub", RuleStub(good_win))):
        ra = engine.attribute_round([s_pre, s_post], acts, wins, pred, cfg, pin_terminal=False)
        fd = comp_phi(ra, pseudo_type="decoy_start")
        total = math.fsum(c["phi"] for sa in ra.actions for c in sa.components)
        meth = {c["method"] for sa in ra.actions for c in sa.components}
        check(f"[{label}/{pname}] method is {label}", meth == {label}, str(meth))
        check(f"[{label}/{pname}] φ_decoy == 0 ± 1e-9", abs(fd) <= 1e-9, f"{fd:.3e}")
        check(f"[{label}/{pname}] Σ φ == ΔW ± 1e-9", abs(total - (ra.w_end - ra.w_start)) <= 1e-9)
        if pname == "ignores-decoys":
            fs = comp_phi(ra, pseudo_type="smoke_pop")
            check(f"[{label}/{pname}] the CT smoke pop is worth its +0.05", abs(fs - 0.05) <= 1e-9, f"{fs:+.6f}")

# ── P8.3.1 pop and fade; P8.2.5 sign ──────────────────────────────────────────

section("durative smoke: φ_pop = +0.05, φ_fade = −0.05, net 0; actor sign")


def smoke_story(owner: int):
    tok = {"x": 1450.0, "y": 400.0, "z": 0.0, "owner": owner, "start_tick": 5000}
    ticks = [(4936, False, ("grid",)), (4999, False, ("boundary_pre",)), (5001, True, ("boundary_post",)),
             (5064, True, ("grid",)), (5127, True, ("boundary_pre",)), (5129, False, ("boundary_post",)),
             (5192, False, ("grid", "terminal"))]
    snaps = [make_snapshot(tick=t, smokes=[tok] if on else [], kinds=k) for t, on, k in ticks]
    smoke = action("smoke", owner, kind="durative", x=1450.0, y=400.0, z=0.0)
    smoke.window_end_tick = 5128
    pop  = action("smoke_pop", owner, tick=5000, parent_action_id=smoke.action_id)
    fade = action("smoke_fade", owner, tick=5128, parent_action_id=smoke.action_id)
    wins = [{"window_id": "pop", "round_num": 7, "start_tick": 4999, "end_tick": 5001, "member_action_ids": [pop.action_id]},
            {"window_id": "fade", "round_num": 7, "start_tick": 5127, "end_tick": 5129, "member_action_ids": [fade.action_id]}]
    return snaps, [smoke, pop, fade], wins, smoke


def smoke_value(s, h=None) -> float:
    w = 0.5
    for u in s.utility:
        if u.kind == "smoke":
            w += 0.05 if u.owner_side == "ct" else -0.05
    return w


for owner, side in ((103, "ct"), (203, "t")):
    snaps, acts, wins, smoke = smoke_story(owner)
    ra = engine.attribute_round(snaps, acts, wins, StubPredictor(win=smoke_value), CFG, pin_terminal=False)
    sa = by_id(ra, smoke.action_id)
    pop, fade = comp_phi(ra, source="pop"), comp_phi(ra, source="fade")
    sgn = 1.0 if side == "ct" else -1.0
    check(f"[{side} smoke] φ_pop = {sgn * 0.05:+.2f} (CT-positive)", abs(pop - sgn * 0.05) <= 1e-12, f"{pop:+.6f}")
    check(f"[{side} smoke] φ_fade = {-sgn * 0.05:+.2f}", abs(fade + sgn * 0.05) <= 1e-12, f"{fade:+.6f}")
    check(f"[{side} smoke] V_smoke = φ_pop + φ_fade + enablers = 0 (nothing happened)", abs(sa.V) <= 1e-12)
    check(f"[{side} smoke] pseudo actions fold into the parent (no own rows)",
          {x.type for x in ra.actions} == {"smoke"})
    check(f"[{side} smoke] components keep pop and fade", sorted(c["source"] for c in sa.components) == ["fade", "pop"])
    pop_only = [w for w in wins if w["window_id"] == "pop"]
    ra2 = engine.attribute_round(snaps[:4], acts[:2], pop_only, StubPredictor(win=smoke_value), CFG, pin_terminal=False)
    v = by_id(ra2, smoke.action_id).V
    check(f"[{side} smoke] actor-signed V of the pop alone is +0.05 for its own side", abs(v - 0.05) <= 1e-12, f"{v:+.6f}")

sys.exit(summary())
