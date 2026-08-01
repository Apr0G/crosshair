"""
Tests for the impact-attribution classification emitted by feature_extractor.

Run from repo root:
    .venv/bin/python test/test_attribution_windows.py [path/to/demo.dem]

Two halves:
  1. Pure unit checks on attribution_window() — no demo needed.
  2. Contract checks over a really-parsed demo: every event carries a class, the
     windows are ordered and in-round, and the tick-rate-derived durations come out
     as the intended number of SECONDS rather than a number of ticks that silently
     assumed 64.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_results: list[tuple[bool, str]] = []


def check(desc: str, ok: bool, detail: str = ""):
    _results.append((bool(ok), desc))
    print(f"  {PASS if ok else FAIL}  {desc}" + (f"  ({detail})" if detail else ""))


def unit_checks():
    from feature_extractor import (attribution_window, _ticks,
                                   KIND_INSTANT, KIND_DURATIVE,
                                   SMOKE_POP_S, SMOKE_LIFE_S, MOLOTOV_LIFE_S)

    print("── attribution_window: classification ───────────────────────────────────")
    for g in ("HE", "Flash", "Decoy"):
        w = attribution_window(g, 1000, 64)
        check(f"{g} is instantaneous", w["impact_kind"] == KIND_INSTANT and w["resolve_tick"] == 1000)
    for g in ("Smoke", "Molotov", "Incendiary"):
        w = attribution_window(g, 1000, 64)
        check(f"{g} is durative", w["impact_kind"] == KIND_DURATIVE)

    print("\n── windows scale with tick rate (the 64-vs-128 bug) ─────────────────────")
    for tr in (64, 128):
        w = attribution_window("Smoke", 0, tr)
        pop_s  = w["window_start_tick"] / tr
        life_s = (w["window_end_tick"] - w["window_start_tick"]) / tr
        check(f"smoke pop is {SMOKE_POP_S}s at {tr} tick", abs(pop_s - SMOKE_POP_S) < 0.01, f"{pop_s:.2f}s")
        check(f"smoke life is {SMOKE_LIFE_S}s at {tr} tick", abs(life_s - SMOKE_LIFE_S) < 0.01, f"{life_s:.2f}s")
        m = attribution_window("Molotov", 0, tr)
        mol_s = (m["window_end_tick"] - m["window_start_tick"]) / tr
        check(f"molotov burn is {MOLOTOV_LIFE_S}s at {tr} tick", abs(mol_s - MOLOTOV_LIFE_S) < 0.01, f"{mol_s:.2f}s")

    # The real regression: a window expressed in ticks must differ between rates.
    w64, w128 = attribution_window("Smoke", 0, 64), attribution_window("Smoke", 0, 128)
    check("128-tick smoke window spans twice the ticks of a 64-tick one",
          w128["window_end_tick"] == 2 * w64["window_end_tick"],
          f"{w64['window_end_tick']} vs {w128['window_end_tick']}")

    print("\n── parser-supplied ticks beat nominal durations ─────────────────────────")
    w = attribution_window("Smoke", 1000, 64, start_tick=1111, end_tick=2222)
    check("real start/end ticks are used when present",
          w["window_start_tick"] == 1111 and w["window_end_tick"] == 2222)
    w = attribution_window("Smoke", 1000, 64, start_tick=float("nan"), end_tick=None)
    check("NaN start falls back to the nominal pop delay",
          w["window_start_tick"] == 1000 + _ticks(SMOKE_POP_S, 64))

    print("\n── degenerate input ─────────────────────────────────────────────────────")
    check("None throw tick degrades to instantaneous with no resolve tick",
          attribution_window("Smoke", None, 64) == {"impact_kind": KIND_INSTANT, "resolve_tick": None})
    check("NaN throw tick degrades the same way",
          attribution_window("Smoke", float("nan"), 64)["resolve_tick"] is None)
    check("windows are strictly ordered", all(
        attribution_window(g, 500, 64)["window_start_tick"] <
        attribution_window(g, 500, 64)["window_end_tick"]
        for g in ("Smoke", "Molotov", "Incendiary")))


def demo_checks(demo: str) -> None:
    import io, contextlib
    import extract, feature_extractor as fx
    from feature_extractor import KIND_INSTANT, KIND_DURATIVE, KIND_ROUND

    with contextlib.redirect_stdout(io.StringIO()):
        tables = extract.extract(demo)
        mp = tables.get("map_name", "unknown")
        vis = fx.build_vis_checker(mp)
        events = fx.extract_events(tables, "t_aw", mp, vis)

    tr = int(tables.get("tick_rate") or 64)
    print(f"\n{demo}  ({mp}, tick_rate={tr}) — {len(events)} events\n")
    print("── every event is classified ────────────────────────────────────────────")

    by_type: dict[str, set] = {}
    for e in events:
        by_type.setdefault(e["event_type"], set()).add(
            (e.get("action") or {}).get("impact_kind"))
    for et in sorted(by_type):
        print(f"    {et:12} -> {sorted(str(k) for k in by_type[et])}")

    expected = {"engagement": KIND_INSTANT, "bomb": KIND_INSTANT, "rotation": KIND_DURATIVE}
    for et, want in expected.items():
        if et in by_type:
            check(f"{et} events are all {want}", by_type[et] == {want}, str(by_type[et]))
    if "utility" in by_type:
        check("utility events are instantaneous or durative, never unset",
              by_type["utility"] <= {KIND_INSTANT, KIND_DURATIVE}, str(by_type["utility"]))

    print("\n── durative windows are sane ────────────────────────────────────────────")
    dur = [e for e in events if (e.get("action") or {}).get("impact_kind") == KIND_DURATIVE]
    bad_order = [e for e in dur
                 if e["action"].get("window_start_tick") is not None
                 and e["action"].get("window_end_tick") is not None
                 and e["action"]["window_start_tick"] > e["action"]["window_end_tick"]]
    check("no durative window ends before it starts", not bad_order, f"{len(bad_order)} bad")
    check("durative events exist at all", len(dur) > 0, f"{len(dur)}")

    smokes = [e for e in dur if e["action"].get("grenade_type") == "Smoke"]
    if smokes:
        spans = [(e["action"]["window_end_tick"] - e["action"]["window_start_tick"]) / tr for e in smokes]
        worst = max(abs(s - fx.SMOKE_LIFE_S) for s in spans)
        check(f"every smoke window is ~{fx.SMOKE_LIFE_S}s in REAL time",
              worst < 0.05, f"worst deviation {worst:.3f}s over {len(smokes)} smokes")

    print("\n── buys are round-level, not in-round ───────────────────────────────────")
    buys = [e for e in events if e["event_type"] == "buy"]
    if buys:
        kinds = {(e.get("action") or {}).get("impact_kind") for e in buys}
        check("buy events are classified round-level", kinds == {KIND_ROUND}, str(kinds))
        # t=0.0 is CORRECT for a buy (freeze end == round clock zero). The old bug was
        # not the timestamp but that it was bracketed like an in-round action, which
        # collapsed every buy in a round onto the first state sample.
        check("buy timestamp stays at the round clock origin",
              {e["time_into_round"] for e in buys} == {0.0})
        check("no buy carries an in-round resolve window",
              all("window_start_tick" not in (e.get("action") or {}) for e in buys))

    print("\n── instantaneous events carry a resolve tick ────────────────────────────")
    inst = [e for e in events if (e.get("action") or {}).get("impact_kind") == KIND_INSTANT]
    missing = [e for e in inst if e["action"].get("resolve_tick") is None]
    check("every instantaneous event has a resolve_tick", not missing,
          f"{len(missing)} of {len(inst)} missing")


def main() -> int:
    unit_checks()
    demo = sys.argv[1] if len(sys.argv) > 1 else "demos/faceit2.dem"
    if Path(demo).exists():
        demo_checks(demo)
    else:
        print(f"\n(skipping demo checks — no {demo})")

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
