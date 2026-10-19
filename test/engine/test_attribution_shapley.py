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

REAL_BOUND = 0.002           # P8.2.4: the real-model C1 bound (decided 2026-10-19)
REAL_WINDOWS = 200


def real_model_check() -> int:
    """`--real`: the P8.2.4 variant on the adopted win model and the stored states. Exit
    2 only when no adopted wp model or no stored states exist (MED-41)."""
    from engine import paths
    from engine.actions.windows import Window
    from engine.counterfactual import sanity
    from engine.data import layout
    from engine.evaluation import registry
    from engine.state.schema import UtilityToken
    adopted = [r["model"] for r in registry.adoption_rows() if r["status"].lower() == "adopted" and r["kind"] == "wp"]
    if not adopted:
        print("null_action_same_tick (real model): no adopted wp model in docs/ENGINE_MODELS.md")
        return 2
    states = layout.find("states") if (paths.ENGINE_DIR / "states").is_dir() else []
    if not states:
        print("null_action_same_tick (real model): no stored states under data/engine/states")
        return 2
    from engine.model.predict import Predictor
    predictor = Predictor(paths.MODELS_DIR / adopted[-1], require_calibration=True)
    n, worst_decoy, worst_kill = 0, 0.0, 0.0
    for sp in states:
        snaps, acts, wins, _mv = sanity._load_match(sp)
        by_id = {a.action_id: a for a in acts}
        by_round: dict = {}
        for s in snaps:
            by_round.setdefault(s.meta.round_num, []).append(s)
        for w in wins:
            members = [by_id[i] for i in w.member_action_ids if i in by_id]
            if len(members) != 1 or members[0].type != "kill":
                continue
            kill = members[0]
            rs = sorted(by_round[w.round_num], key=lambda s: s.meta.tick)
            post = next((s for s in rs if s.meta.tick == w.end_tick), None)
            if post is None:
                continue
            far = (-3000.0 + min(p.x for p in post.players if p.x is not None) , -3000.0, 0.0)
            decoy = action("decoy", kill.actor_id_hash, kind="durative", x=far[0], y=far[1], z=far[2])
            pop   = action("decoy_start", kill.actor_id_hash, tick=kill.tick, parent_action_id=decoy.action_id)
            pop.match_key = decoy.match_key = kill.match_key
            pop.round_num = decoy.round_num = kill.round_num
            tok = UtilityToken(kind="decoy", x=far[0], y=far[1], z=far[2], time_left_s=15.0,
                               owner_side=kill.actor_side, owner_id=kill.actor_id_hash, radius_units=None)
            edited = [s.copy() if s.meta.tick >= w.end_tick else s for s in rs]
            for s in edited:
                if s.meta.tick >= w.end_tick:
                    s.utility.append(tok)
            w2 = Window(window_id=w.window_id, round_num=w.round_num, start_tick=w.start_tick, end_tick=w.end_tick,
                        member_action_ids=(kill.action_id, pop.action_id))
            others = [x for x in wins if x.round_num == w.round_num and x.window_id != w.window_id]
            ra = engine.attribute_round(edited, acts + [decoy, pop], others + [w2], predictor, CFG, pin_terminal=False)
            f_decoy = comp_phi(ra, source="pop", action_id=decoy.action_id)
            f_kill  = comp_phi(ra, source="window", action_id=kill.action_id)
            dW = by_id_w(ra, w2.window_id)
            worst_decoy = max(worst_decoy, abs(f_decoy))
            worst_kill  = max(worst_kill, abs(f_kill - dW)) if dW is not None else worst_kill
            n += 1
            if n >= REAL_WINDOWS:
                break
        if n >= REAL_WINDOWS:
            break
    if n == 0:
        print("null_action_same_tick (real model): the stored matches hold no single-kill window")
        return 2
    check(f"[real] |φ_decoy| ≤ {REAL_BOUND} over {n} real kill windows", worst_decoy <= REAL_BOUND, f"max {worst_decoy:.4f}")
    check(f"[real] |φ_kill − ΔW| ≤ {REAL_BOUND} over {n} windows", worst_kill <= REAL_BOUND, f"max {worst_kill:.4f}")
    return summary()


def by_id_w(ra, window_id):
    for iv in getattr(ra, "intervals", []) or []:
        if getattr(iv, "window_id", None) == window_id:
            return getattr(iv, "delta_w", None)
    return None


if "--real" in sys.argv:
    sys.exit(real_model_check())


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
skip(f"real-model bound |φ_decoy| ≤ {REAL_BOUND} over {REAL_WINDOWS} windows",
     "run with --real on a PC with an adopted model and stored states (exit 2 there until both exist)")

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
