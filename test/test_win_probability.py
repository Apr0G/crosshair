"""
Tests for win_probability.py

Run from repo root:
    .venv/bin/python test/test_win_probability.py

Requires a trained model at data/win_prob.lgb.
Train first with:  .venv/bin/python src/main.py train --eval

Everything lives inside main() so the module stays importable — at module level,
a missing model would raise mid-import and sys.exit() would kill any collector.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from win_probability import predict, predict_batch, FEATURES, MODEL_PATH

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

_results: list[tuple[bool, str]] = []


def check(desc: str, condition: bool, detail: str = ""):
    ok = bool(condition)
    tag = PASS if ok else FAIL
    msg = f"  {tag}  {desc}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((ok, desc))


def between(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi


# ── base state (5v5, mid-round, no plant, balanced) ──────────────────────────

def _state(**overrides) -> dict:
    base = dict(
        time_into_round_s  = 45.0,
        time_remaining_s   = 70.0,
        post_plant         = 0,
        alive_ct           = 5,
        alive_t            = 5,
        total_hp_ct        = 500,
        total_hp_t         = 500,
        total_armor_ct     = 500,
        total_armor_t      = 500,
        helmets_ct         = 5,
        helmets_t          = 5,
        has_defuser        = 0,
        equip_value_ct     = 20000,
        equip_value_t      = 20000,
        smokes_ct          = 2,
        smokes_t           = 2,
        flashes_ct         = 1,
        flashes_t          = 1,
        he_ct              = 1,
        he_t               = 1,
        molotovs_ct        = 0,
        molotovs_t         = 0,
        active_smokes      = 0,
        active_infernos    = 0,
        min_dist_ct_to_bomb= None,
        min_dist_t_to_bomb = None,
        ct_spotted_count   = 0,
        t_spotted_count    = 0,
        ct_heard_enemy     = 0,
        t_heard_enemy      = 0,
        ct_spread          = 500.0,
        t_spread           = 500.0,
        map                = "de_mirage",
    )
    base.update(overrides)
    return base


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"no model at {MODEL_PATH}")
        print("train first:  .venv/bin/python src/main.py train --eval")
        return 2   # distinct from 1 (a real failure) so CI can tell them apart

    print("\n── Feature parity ───────────────────────────────────────────────────────")

    # _state() is a second hand-maintained copy of FEATURES. Without this check, a
    # feature added to FEATURES and forgotten here is silently NaN-filled on every
    # call — the exact "fails silently" outcome the P8 invariant warns about.
    check("_state() covers exactly FEATURES", set(_state()) == set(FEATURES),
          f"missing={sorted(set(FEATURES) - set(_state()))} "
          f"extra={sorted(set(_state()) - set(FEATURES))}")

    print("\n── Output range ─────────────────────────────────────────────────────────")

    p = predict(_state())
    check("predict() returns float",    isinstance(p, float))
    check("balanced state near 0.5",    between(p, 0.35, 0.65), f"{p:.4f}")

    print("\n── Player advantage ─────────────────────────────────────────────────────")

    p_5v1 = predict(_state(alive_ct=5, alive_t=1, total_hp_ct=500, total_hp_t=80))
    p_1v5 = predict(_state(alive_ct=1, alive_t=5, total_hp_ct=80,  total_hp_t=500))
    check("5v1 CT > 0.5",               p_5v1 > 0.5, f"{p_5v1:.4f}")
    check("1v5 CT < 0.5",               p_1v5 < 0.5, f"{p_1v5:.4f}")

    p_5v3 = predict(_state(alive_ct=5, alive_t=3))
    p_3v5 = predict(_state(alive_ct=3, alive_t=5))
    check("5v3 CT > 3v5 CT",            p_5v3 > p_3v5)

    print("\n── HP advantage ─────────────────────────────────────────────────────────")

    p_high_ct = predict(_state(total_hp_ct=500, total_hp_t=100))
    p_high_t  = predict(_state(total_hp_ct=100, total_hp_t=500))
    check("CT high HP > CT low HP",     p_high_ct > p_high_t)

    print("\n── Post-plant ───────────────────────────────────────────────────────────")

    # T should have advantage post-plant with bomb close and little time
    p_t_close = predict(_state(
        post_plant=1,
        min_dist_ct_to_bomb=1500.0,
        min_dist_t_to_bomb=50.0,
        has_defuser=0,
        time_remaining_s=10.0,
        alive_ct=1,
        alive_t=3,
        total_hp_ct=80,
        total_hp_t=300,
    ))
    check("T dominant post-plant < 0.5 for CT",    p_t_close < 0.5, f"{p_t_close:.4f}")

    # CT with defuser close should be better than without
    p_defuser     = predict(_state(post_plant=1, min_dist_ct_to_bomb=100.0, min_dist_t_to_bomb=800.0, has_defuser=1, time_remaining_s=25.0))
    p_no_defuser  = predict(_state(post_plant=1, min_dist_ct_to_bomb=100.0, min_dist_t_to_bomb=800.0, has_defuser=0, time_remaining_s=25.0))
    check("defuser > no defuser (post-plant, CT close)", p_defuser > p_no_defuser)

    print("\n── Economy advantage ────────────────────────────────────────────────────")

    p_rich_ct = predict(_state(equip_value_ct=40000, equip_value_t=4000))
    p_rich_t  = predict(_state(equip_value_ct=4000,  equip_value_t=40000))
    check("CT rich eco > T rich eco",   p_rich_ct > p_rich_t)

    print("\n── Edge cases ───────────────────────────────────────────────────────────")

    p_1v1 = predict(_state(alive_ct=1, alive_t=1, total_hp_ct=100, total_hp_t=100))
    check("1v1 balanced near 0.5",      between(p_1v1, 0.3, 0.7), f"{p_1v1:.4f}")

    p_zero = predict(_state(
        alive_ct=0, alive_t=5,
        total_hp_ct=0, total_hp_t=500,
        equip_value_ct=0, equip_value_t=20000,
    ))
    check("0 CT alive < 0.1",           p_zero < 0.1, f"{p_zero:.4f}")

    # Unknown map and null distances must not raise — a range check would pass
    # tautologically, so assert non-raising explicitly instead.
    try:
        predict(_state(map="de_unknown"))
        predict(_state(min_dist_ct_to_bomb=None, min_dist_t_to_bomb=None))
        predict(_state(ct_spread=None, t_spread=None))
        check("unknown map / null distances / null spread don't raise", True)
    except Exception as e:
        check("unknown map / null distances / null spread don't raise", False, repr(e))

    # The map feature must actually do something. A range check over all maps holds
    # identically whether the map is honoured or entirely inert.
    maps = ["de_mirage", "de_dust2", "de_inferno", "de_nuke", "de_ancient", "de_anubis", "de_vertigo"]
    map_preds = {m: predict(_state(map=m)) for m in maps}
    check("map feature is not inert (predictions differ by map)",
          len(set(round(v, 6) for v in map_preds.values())) > 1,
          " ".join(f"{m.replace('de_','')}={v:.3f}" for m, v in map_preds.items()))

    print("\n── predict_batch ────────────────────────────────────────────────────────")

    states = [_state(alive_ct=5, alive_t=i) for i in range(1, 6)]
    batch  = predict_batch(states)
    check("batch returns correct length",   len(batch) == 5)
    check("batch monotone: more T = lower CT WP", all(batch[i] > batch[i+1] for i in range(len(batch)-1)),
          " ".join(f"{v:.3f}" for v in batch))

    # A batch spanning several maps is the only case score-adjacent code hits in
    # anger, and it is where a per-row categorical would break.
    mixed = [_state(map=m) for m in ("de_mirage", "de_nuke", "de_inferno")]
    try:
        mixed_batch = predict_batch(mixed)
        singles = [predict(s) for s in mixed]
        check("mixed-map batch matches element-wise predict",
              len(mixed_batch) == 3 and all(abs(a - b) < 1e-9 for a, b in zip(mixed_batch, singles)))
    except Exception as e:
        check("mixed-map batch matches element-wise predict", False, repr(e))

    check("predict_batch([]) returns empty", len(predict_batch([])) == 0)

    print("\n── Monotonicity ─────────────────────────────────────────────────────────")

    ct_alive_preds = [predict(_state(alive_ct=n, alive_t=3, total_hp_ct=n*100, total_hp_t=300))
                      for n in range(1, 6)]
    check("CT alive 1→5 is monotone increasing",
          all(ct_alive_preds[i] < ct_alive_preds[i+1] for i in range(len(ct_alive_preds)-1)),
          " ".join(f"{v:.3f}" for v in ct_alive_preds))

    print("\n── Calibration ──────────────────────────────────────────────────────────")
    _check_calibration()

    # ── summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for ok, _ in _results if ok)
    total  = len(_results)
    failed = [desc for ok, desc in _results if not ok]

    print(f"\n{'─'*60}")
    print(f"  {passed}/{total} passed")
    if failed:
        print("  Failed:")
        for d in failed:
            print(f"    - {d}")
    print()
    return 0 if not failed else 1


def _check_calibration(n_bins: int = 10, tolerance: float = 0.10):
    """Impact scoring needs CALIBRATED probabilities, not just good ranking.

    Every other check here is a ranking property — they all survive any monotone
    distortion of the output (p -> 0.5 + 0.9*(p-0.5) passes all of them) while every
    impact delta silently changes. This bins real states by predicted probability and
    compares each bin's predicted mean against the observed CT win rate.
    """
    import sqlite3
    db_path = Path(__file__).parent.parent / "data" / "crosshair.db"
    if not db_path.exists():
        check("calibration: reliability within tolerance", True, "skipped — no database")
        return

    con = sqlite3.connect(db_path)
    try:
        cols = ", ".join(FEATURES)
        rows = con.execute(
            f"SELECT {cols}, round_won_ct FROM round_states "
            f"WHERE alive_ct > 0 AND alive_t > 0 "
            f"ORDER BY match_id LIMIT 20000"
        ).fetchall()
    except sqlite3.OperationalError as e:
        check("calibration: reliability within tolerance", True, f"skipped — {e}")
        return
    finally:
        con.close()

    if len(rows) < 1000:
        check("calibration: reliability within tolerance", True,
              f"skipped — only {len(rows)} states")
        return

    states = [dict(zip(FEATURES, r[:-1])) for r in rows]
    labels = [r[-1] for r in rows]
    preds  = predict_batch(states)

    bins: dict[int, list] = {}
    for p, y in zip(preds, labels):
        bins.setdefault(min(int(p * n_bins), n_bins - 1), []).append((p, y))

    worst_gap, worst_bin = 0.0, None
    for b in sorted(bins):
        pairs = bins[b]
        if len(pairs) < 50:
            continue
        mean_p = sum(p for p, _ in pairs) / len(pairs)
        obs    = sum(y for _, y in pairs) / len(pairs)
        gap    = abs(mean_p - obs)
        print(f"    bin {b/n_bins:.1f}-{(b+1)/n_bins:.1f}: predicted {mean_p:.3f}  "
              f"observed {obs:.3f}  gap {gap:+.3f}  (n={len(pairs)})")
        if gap > worst_gap:
            worst_gap, worst_bin = gap, b

    check(f"calibration: every bin within {tolerance:.2f} of observed",
          worst_gap <= tolerance,
          f"worst gap {worst_gap:.3f} in bin {worst_bin}")


if __name__ == "__main__":
    sys.exit(main())
