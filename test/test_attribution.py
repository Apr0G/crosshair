"""
Tests for the event-boundary attribution engine.

Run from repo root:
    .venv/bin/python test/test_attribution.py

Pure unit tests: synthetic win-probability paths, no demo, no model, no database.
That is deliberate — the engine takes a WP *function*, so its correctness can be
established independently of whether the model is any good.

The load-bearing property is conservation:

    sum(impacts) + drift == wp(round_end) - wp(round_start)

Every scenario below asserts it, because credit that is double-counted, invented or
lost breaks that identity and nothing else would catch it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attribution import (Action, attribute_round, terminal_wp,
                         KIND_INSTANT, KIND_DURATIVE, KIND_ROUND, CT, T)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_results: list[tuple[bool, str]] = []
TOL = 1e-9


def check(desc: str, ok: bool, detail: str = ""):
    _results.append((bool(ok), desc))
    print(f"  {PASS if ok else FAIL}  {desc}" + (f"  ({detail})" if detail else ""))


def close(a, b, tol=1e-6) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def step_path(points: dict, default: float = 0.5):
    """WP function that holds each value until the next breakpoint."""
    keys = sorted(points)
    def wp(tick):
        val = default
        for k in keys:
            if tick >= k:
                val = points[k]
        return val
    return wp


def imp(res, eid):
    for a in res.attributions:
        if a.event_id == eid:
            return a.impact
    return None


def main() -> int:
    print("── a single instantaneous action takes its own jump ─────────────────────")
    wp = step_path({0: 0.50, 100: 0.80})
    res = attribute_round([Action(1, KIND_INSTANT, CT, resolve_tick=100)], wp, 0, 200)
    check("CT actor gets the positive jump", close(imp(res, 1), 0.30), f"{imp(res,1)}")
    check("conserves", res.conserves(TOL), f"residual {res.residual:.2e}")
    check("no drift when one action explains everything", close(res.drift, 0.0))

    res = attribute_round([Action(1, KIND_INSTANT, T, resolve_tick=100)], wp, 0, 200)
    check("T actor gets the negated jump", close(imp(res, 1), -0.30), f"{imp(res,1)}")
    check("still conserves with a T actor", res.conserves(TOL))

    print("\n── the bug this replaces: two actors, one bracket ───────────────────────")
    # A T kill at 100 then a CT trade at 140. Under grid bracketing both land in the
    # same ~1 Hz interval and each receives the FULL window delta.
    wp = step_path({0: 0.50, 100: 0.30, 140: 0.62})
    res = attribute_round([
        Action(1, KIND_INSTANT, T,  resolve_tick=100),
        Action(2, KIND_INSTANT, CT, resolve_tick=140),
    ], wp, 0, 200)
    check("the T kill is credited +0.20 to the T actor", close(imp(res, 1), 0.20), f"{imp(res,1)}")
    check("the CT trade is credited +0.32 to the CT actor", close(imp(res, 2), 0.32), f"{imp(res,2)}")
    check("the two impacts differ", not close(imp(res, 1), imp(res, 2)))
    check("conserves", res.conserves(TOL), f"residual {res.residual:.2e}")

    print("\n── simultaneity: equal split without state hooks ────────────────────────")
    wp = step_path({0: 0.50, 100: 0.60})
    res = attribute_round([
        Action(1, KIND_INSTANT, CT, resolve_tick=100),
        Action(2, KIND_INSTANT, CT, resolve_tick=100),
    ], wp, 0, 200)
    check("two same-side simultaneous actions split the jump", close(imp(res, 1), 0.05) and close(imp(res, 2), 0.05))
    check("conserves", res.conserves(TOL))

    res = attribute_round([
        Action(1, KIND_INSTANT, CT, resolve_tick=100),
        Action(2, KIND_INSTANT, T,  resolve_tick=100),
    ], wp, 0, 200)
    check("opposite-side simultaneous actions get opposite signs",
          close(imp(res, 1), 0.05) and close(imp(res, 2), -0.05))

    print("\n── simultaneity: exact Shapley with state hooks ─────────────────────────")
    # Non-linear value function so ordering genuinely matters; Shapley must average
    # over orderings rather than depend on an arbitrary one.
    def wp_of_state(s):
        return round(0.5 + 0.1 * s["a"] - 0.02 * s["a"] ** 2 + 0.05 * s["b"], 6)
    def apply_delta(s, d):
        out = dict(s)
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
        return out
    base = {"a": 0, "b": 0}
    states = {99: base}
    def wp2(tick):
        if tick <= 99:
            return wp_of_state(base)
        return wp_of_state(apply_delta(apply_delta(base, {"a": 2}), {"b": 1}))
    acts = [
        Action(1, KIND_INSTANT, CT, resolve_tick=100, state_delta={"a": 2}),
        Action(2, KIND_INSTANT, CT, resolve_tick=100, state_delta={"b": 1}),
    ]
    res = attribute_round(acts, wp2, 0, 200,
                          wp_of_state=wp_of_state, apply_delta=apply_delta,
                          state_at=lambda t: states.get(t, base))
    methods = {a.method for a in res.attributions}
    check("shapley path is taken when hooks are supplied",
          all("shapley" in m for m in methods), str(methods))
    check("shapley shares are unequal for unequal contributions",
          not close(imp(res, 1), imp(res, 2)), f"{imp(res,1):.4f} vs {imp(res,2):.4f}")
    check("shapley still conserves", res.conserves(1e-9), f"residual {res.residual:.2e}")

    # Symmetry: identical contributions must receive identical credit.
    acts_sym = [
        Action(1, KIND_INSTANT, CT, resolve_tick=100, state_delta={"b": 1}),
        Action(2, KIND_INSTANT, CT, resolve_tick=100, state_delta={"b": 1}),
    ]
    def wp3(tick):
        return wp_of_state(base) if tick <= 99 else wp_of_state(apply_delta(base, {"b": 2}))
    res = attribute_round(acts_sym, wp3, 0, 200,
                          wp_of_state=wp_of_state, apply_delta=apply_delta,
                          state_at=lambda t: base)
    check("shapley is symmetric for identical actions", close(imp(res, 1), imp(res, 2)))

    print("\n── durative: residual drift, not the raw window delta ───────────────────")
    # A smoke covering [50, 150]. A kill at 100 inside it moves WP by +0.20.
    # The smoke must get the window's change MINUS that kill, not the whole thing.
    wp = step_path({0: 0.50, 100: 0.70, 150: 0.75})
    res = attribute_round([
        Action(1, KIND_INSTANT,  CT, resolve_tick=100),
        Action(2, KIND_DURATIVE, CT, window=(50, 150)),
    ], wp, 0, 200)
    check("the kill keeps its own jump", close(imp(res, 1), 0.20), f"{imp(res,1)}")
    check("the smoke gets window delta minus the kill", close(imp(res, 2), 0.05), f"{imp(res,2)}")
    check("conserves", res.conserves(TOL), f"residual {res.residual:.2e}")

    print("\n── durative: overlapping windows do not double-count ────────────────────")
    wp = step_path({0: 0.50, 200: 0.70})
    res = attribute_round([
        Action(1, KIND_DURATIVE, CT, window=(0, 200)),
        Action(2, KIND_DURATIVE, CT, window=(0, 200)),
    ], wp, 0, 200)
    total = imp(res, 1) + imp(res, 2)
    check("two identical windows split the drift rather than each taking it",
          close(total, 0.20) and close(imp(res, 1), 0.10), f"each {imp(res,1)}, total {total}")
    check("conserves", res.conserves(TOL), f"residual {res.residual:.2e}")
    check("nothing is left as drift when windows tile the round",
          close(res.drift, 0.0), f"drift {res.drift}")

    print("\n── round-level actions carry no in-round credit ─────────────────────────")
    wp = step_path({0: 0.50, 100: 0.90})
    res = attribute_round([
        Action(1, KIND_ROUND,   CT),
        Action(2, KIND_INSTANT, CT, resolve_tick=100),
    ], wp, 0, 200)
    check("a buy is reported unvalued, not scored 0.0", imp(res, 1) is None)
    check("the in-round action keeps its jump", close(imp(res, 2), 0.40))
    check("conserves", res.conserves(TOL))

    print("\n── unattributed movement surfaces as drift, never as credit ─────────────")
    wp = step_path({0: 0.50, 100: 0.65})
    res = attribute_round([], wp, 0, 200)
    check("with no actions the whole change is drift", close(res.drift, 0.15), f"{res.drift}")
    check("nothing is attributed", res.total_attributed == 0.0)
    check("conserves", res.conserves(TOL))

    print("\n── unknown side is never silently signed as T ───────────────────────────")
    wp = step_path({0: 0.50, 100: 0.80})
    res = attribute_round([Action(1, KIND_INSTANT, "?", resolve_tick=100)], wp, 0, 200)
    check("'?' side yields impact None, not -delta", imp(res, 1) is None)
    check("its change falls into drift instead", close(res.drift, 0.30), f"{res.drift}")
    check("conserves", res.conserves(TOL))

    print("\n── degenerate input ─────────────────────────────────────────────────────")
    res = attribute_round([Action(1, KIND_INSTANT, CT, resolve_tick=100)],
                          lambda t: None, 0, 200)
    check("an unusable WP path yields nothing rather than raising", res.attributions == [])
    res = attribute_round([Action(1, KIND_DURATIVE, CT, window=(150, 150))],
                          step_path({0: 0.5}), 0, 200)
    check("a zero-width window is reported, not credited", imp(res, 1) is None)
    res = attribute_round([Action(1, KIND_DURATIVE, CT, window=(-500, 9999))],
                          step_path({0: 0.5, 200: 0.7}), 0, 200)
    check("a window overhanging the round is clamped to it", close(imp(res, 1), 0.20), f"{imp(res,1)}")
    check("conserves after clamping", res.conserves(TOL))

    print("\n── conservation holds on a busy, mixed round ────────────────────────────")
    wp = step_path({0: 0.50, 40: 0.44, 90: 0.58, 91: 0.58, 130: 0.35, 175: 0.83})
    actions = [
        Action(1, KIND_ROUND,    CT),
        Action(2, KIND_ROUND,    T),
        Action(3, KIND_INSTANT,  T,  resolve_tick=40),
        Action(4, KIND_INSTANT,  CT, resolve_tick=90),
        Action(5, KIND_INSTANT,  CT, resolve_tick=90),   # simultaneous
        Action(6, KIND_INSTANT,  T,  resolve_tick=130),
        Action(7, KIND_DURATIVE, CT, window=(20, 120)),
        Action(8, KIND_DURATIVE, T,  window=(100, 180)),
        Action(9, KIND_DURATIVE, CT, window=(60, 160)),  # overlaps both
    ]
    res = attribute_round(actions, wp, 0, 200)
    check("every action receives a verdict", len(res.attributions) == 9, str(len(res.attributions)))
    check("conserves on a mixed round", res.conserves(1e-9), f"residual {res.residual:.2e}")
    print(f"      wp {res.wp_start:.2f} -> {res.wp_end:.2f}   "
          f"attributed {res.total_attributed:+.4f}   drift {res.drift:+.4f}")

    print("\n── the guarantee is NOT vacuous ─────────────────────────────────────────")
    # An earlier version derived drift as (change - attributed), so ANY numbers
    # passed. drift is now accumulated from unclaimed intervals, so a partition that
    # loses or duplicates movement shows up as a non-zero residual.
    from attribution import RoundResult as RR, Attribution as At
    bogus = RR(wp_start=0.5, wp_end=0.6)
    bogus.attributions = [At(1, 99999.0, 0, 0, "x")]
    bogus.ct_attributed, bogus.drift = 99999.0, 0.0
    check("garbage attributions FAIL conservation", not bogus.conserves(TOL),
          f"residual {bogus.residual:.1f}")

    print("\n── partition invariance (splitting a window changes nothing) ────────────")
    wp = step_path({0: 0.50, 100: 0.62, 200: 0.70})
    whole = attribute_round([Action(1, KIND_DURATIVE, CT, window=(0, 200))], wp, 0, 200)
    halves = attribute_round([Action(1, KIND_DURATIVE, CT, window=(0, 100)),
                             Action(2, KIND_DURATIVE, CT, window=(100, 200))], wp, 0, 200)
    check("one window == two abutting halves",
          close(imp(whole, 1), imp(halves, 1) + imp(halves, 2)),
          f"{imp(whole,1):.4f} vs {imp(halves,1)+imp(halves,2):.4f}")

    print("\n── adjacent ticks do not double-count ──────────────────────────────────")
    wp = step_path({0: 0.50, 100: 0.60, 101: 0.70})
    res = attribute_round([Action(1, KIND_INSTANT, CT, resolve_tick=100),
                           Action(2, KIND_INSTANT, CT, resolve_tick=101)], wp, 0, 200)
    tot = imp(res, 1) + imp(res, 2)
    check("two kills one tick apart sum to the real movement", close(tot, 0.20), f"{tot:.4f}")
    check("conserves", res.conserves(TOL), f"residual {res.residual:.2e}")

    print("\n── a jump on a window boundary is paid once ────────────────────────────")
    wp = step_path({0: 0.50, 100: 0.90})
    res = attribute_round([Action(1, KIND_INSTANT,  CT, resolve_tick=100),
                           Action(2, KIND_DURATIVE, CT, window=(0, 100)),
                           Action(3, KIND_DURATIVE, CT, window=(100, 200))], wp, 0, 200)
    tot = sum(x for x in (imp(res,1), imp(res,2), imp(res,3)) if x is not None)
    check("kill + two touching windows sum to the real movement", close(tot, 0.40), f"{tot:.4f}")
    check("conserves", res.conserves(TOL), f"residual {res.residual:.2e}")

    print("\n── monotone path: no negative credit, none exceeding the total ─────────")
    wp = step_path({0: 0.30, 50: 0.45, 120: 0.62, 190: 0.80})
    res = attribute_round([Action(1, KIND_INSTANT,  CT, resolve_tick=50),
                           Action(2, KIND_INSTANT,  CT, resolve_tick=120),
                           Action(3, KIND_DURATIVE, CT, window=(0, 200))], wp, 0, 200)
    vals = [a.impact for a in res.attributions if a.impact is not None]
    check("no CT impact is negative on a rising path", all(v >= -1e-12 for v in vals), str([f"{v:.3f}" for v in vals]))
    check("no single impact exceeds the round's total change",
          all(v <= (res.wp_end - res.wp_start) + 1e-12 for v in vals))

    print("\n── Shapley gap is redistributed additively, never by rescaling ─────────")
    # A state-space jump that disagrees in DIRECTION with the timeline movement used
    # to invert every share and, when marginals nearly cancelled, produce 1e10 impacts.
    def wos(s): return 0.5 + 0.10 * s["a"] - 0.02 * s["b"]
    def app(s, d):
        o = dict(s)
        for k, v in d.items(): o[k] = o.get(k, 0) + v
        return o
    base = {"a": 0, "b": 0}
    wpx = step_path({0: 0.50, 100: 0.46})   # timeline says -0.04
    res = attribute_round([Action(1, KIND_INSTANT, CT, resolve_tick=100, state_delta={"a": 1}),
                           Action(2, KIND_INSTANT, CT, resolve_tick=100, state_delta={"b": 1})],
                          wpx, 0, 200, wp_of_state=wos, apply_delta=app, state_at=lambda t: base)
    i1, i2 = imp(res, 1), imp(res, 2)
    check("the helping action is not inverted into blame", i1 > i2, f"{i1:.4f} vs {i2:.4f}")
    check("shares stay bounded", all(abs(v) < 1.0 for v in (i1, i2)), f"{i1:.4f}, {i2:.4f}")
    check("conserves", res.conserves(TOL), f"residual {res.residual:.2e}")

    print("\n── malformed actions get a verdict row, never silent ───────────────────")
    wp = step_path({0: 0.5, 100: 0.7})
    res = attribute_round([Action(1, KIND_INSTANT,  CT, resolve_tick=None),
                           Action(2, KIND_DURATIVE, CT, window=None),
                           Action(3, "nonsense",    CT),
                           Action(4, KIND_INSTANT,  CT, resolve_tick=99999)], wp, 0, 200)
    check("every malformed action still produces an attribution row",
          len(res.attributions) == 4, str(len(res.attributions)))
    check("none of them is credited", all(a.impact is None for a in res.attributions))
    methods = {a.method for a in res.attributions}
    check("each carries a distinct reason", len(methods) == 4, str(sorted(methods)))

    print("\n── terminal states are facts, not predictions ───────────────────────────")
    check("alive_ct == 0 is a certain CT loss",  terminal_wp({"alive_ct": 0, "alive_t": 3}) == 0.0)
    check("alive_t == 0 pre-plant is a certain CT win",
          terminal_wp({"alive_ct": 2, "alive_t": 0, "post_plant": 0}) == 1.0)
    check("a MISSING post_plant is not treated as pre-plant",
          terminal_wp({"alive_ct": 2, "alive_t": 0}) is None)
    check("alive_t == 0 POST-plant is not terminal (bomb can still explode)",
          terminal_wp({"alive_ct": 2, "alive_t": 0, "post_plant": 1}) is None)
    check("a live state is left to the model", terminal_wp({"alive_ct": 3, "alive_t": 2}) is None)
    check("both sides eliminated is undefined", terminal_wp({"alive_ct": 0, "alive_t": 0}) is None)
    check("None state is handled", terminal_wp(None) is None)

    passed = sum(1 for ok, _ in _results if ok)
    failed = [d for ok, d in _results if not ok]
    print(f"\n{'─'*60}\n  {passed}/{len(_results)} passed")
    if failed:
        print("  Failed:")
        for d in failed:
            print(f"    - {d}")
    print()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
