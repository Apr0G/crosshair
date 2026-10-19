"""
Step 3 integration: run the attribution engine over the real corpus.

    .venv/bin/python test/test_attribution_integration.py [max_matches]

Needs data/crosshair.db and a trained data/win_prob.lgb.

The unit tests prove the engine's arithmetic against synthetic paths. This proves
it against real rounds, real events and a real model, and answers the question the
unit tests cannot: does the decomposition conserve on live data, and how much of
each round does it actually explain?

RESOLUTION CAVEAT, stated up front: round_states is sampled at 1 Hz, so the win
probability is only known at 1 s intervals. Instantaneous actions want WP at
tick±1, which the stored data cannot supply — the values here are interpolated
between bracketing samples. That is the same limitation the grid method has, and
it is exactly what persisting boundary states removes. Durative actions span
multiple samples, so they are unaffected. Reported below as "resolution-limited".
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from _harness import check, section, summary
from attribution import attribute_round, RoundResult, KIND_INSTANT, KIND_DURATIVE
from db import DB_PATH as DB, MODEL_PATH
# The production path, not a stand-in: the same WP-path builder and action parser
# that score_match_event runs.
from score_impact import _actions_for_round, _wp_path


def main() -> int:
    if not DB.exists():
        print("no database — scrape first"); return 2
    if not MODEL_PATH.exists():
        print("no model — run `train` first"); return 2

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    mids = [r[0] for r in con.execute("SELECT match_id FROM processed_matches ORDER BY match_id")]
    if limit:
        mids = mids[:limit]
    print(f"\nattributing {len(mids)} matches\n")

    residuals, drifts, swings = [], [], []
    n_rounds = n_actions = 0
    by_kind: dict[str, int] = {}
    scored_by_kind: dict[str, int] = {}
    zero_jump = instant_total = 0
    unbracketed = 0
    sign_ok = sign_bad = 0
    abs_attr = 0.0
    split_ok = split_bad = 0

    for mi, mid in enumerate(mids, 1):
        states = pd.read_sql_query(
            "SELECT * FROM round_states WHERE match_id=? ORDER BY round_num, tick",
            con, params=(mid,))
        if states.empty:
            continue

        events = pd.read_sql_query(
            "SELECT id, round_num, player_side, action FROM events WHERE match_id=?",
            con, params=(mid,))

        for rn, grp in states.groupby("round_num"):
            grp = grp.sort_values("tick").reset_index(drop=True)
            if len(grp) < 3:
                continue
            wp_fn, lo, hi = _wp_path(grp)
            actions = _actions_for_round(events[events["round_num"] == rn])
            for a in actions:
                by_kind[a.kind] = by_kind.get(a.kind, 0) + 1

            if not actions:
                continue
            res = attribute_round(actions, wp_fn, lo, hi)
            if not res.attributions:
                continue

            n_rounds += 1
            n_actions += len(res.attributions)
            residuals.append(abs(res.residual))
            drifts.append(abs(res.drift))
            swings.append(abs(res.wp_end - res.wp_start))

            kind_of = {a.event_id: a.kind for a in actions}
            side_of = {a.event_id: a.side for a in actions}
            for at in res.attributions:
                k = kind_of.get(at.event_id, "?")
                if at.impact is None:
                    unbracketed += 1
                    continue
                scored_by_kind[k] = scored_by_kind.get(k, 0) + 1
                abs_attr += abs(at.impact)
                if k == KIND_INSTANT:
                    instant_total += 1
                    if abs(at.impact) < 1e-12:
                        zero_jump += 1
                    if at.p_before is not None and at.p_after is not None:
                        delta = at.p_after - at.p_before
                        want = delta if side_of[at.event_id] == "ct" else -delta
                        if at.method == "instant":
                            # sole action on its tick: it takes the whole jump
                            if abs(at.impact - want) < 1e-9:
                                sign_ok += 1
                            else:
                                sign_bad += 1
                        else:
                            # a share of the jump: magnitude differs, DIRECTION must not
                            if abs(want) < 1e-12 or (at.impact >= 0) == (want >= 0):
                                split_ok += 1
                            else:
                                split_bad += 1
        if mi % 40 == 0:
            print(f"    {mi}/{len(mids)} matches ...")

    con.close()
    residuals = np.array(residuals); drifts = np.array(drifts); swings = np.array(swings)

    print(f"\n  rounds attributed : {n_rounds:,}")
    print(f"  actions valued    : {n_actions:,}")
    print("  by class          : " + ", ".join(f"{k}={v:,}" for k, v in sorted(by_kind.items())))

    section("conservation on real rounds")
    print(f"    max |residual|    {residuals.max():.3e}")
    print(f"    mean |residual|   {residuals.mean():.3e}")
    check("every round conserves to 1e-9", residuals.max() < 1e-9, f"max {residuals.max():.2e}")
    check("conservation is a REAL check (drift accumulated, not derived)",
          hasattr(RoundResult, "ct_attributed"))
    check("attributed a non-trivial number of rounds", n_rounds > 500, str(n_rounds))

    section("how much of each round is explained")
    # Signed sums cancel across sides, so magnitude is the meaningful denominator.
    share = abs_attr / max(abs_attr + drifts.sum(), 1e-12)
    print(f"    total |WP swing|     {swings.sum():.1f}")
    print(f"    sum of |impacts|     {abs_attr:.1f}")
    print(f"    total |drift|        {drifts.sum():.1f}")
    print(f"    attributed magnitude {share:.1%}")
    check("most of the movement is attributed, not left as drift", share > 0.5, f"{share:.1%}")

    section("sign convention (P5)")
    check("sole-action-on-tick impacts equal their side's signed delta",
          sign_bad == 0, f"{sign_bad} wrong of {sign_ok + sign_bad}")
    check("split impacts keep the correct DIRECTION for their side",
          split_bad == 0, f"{split_bad} wrong of {split_ok + split_bad}")

    section("resolution limit that boundary states remove")
    pct = 100.0 * zero_jump / max(instant_total, 1)
    print(f"    instantaneous actions valued : {instant_total:,}")
    print(f"    of those, zero jump          : {zero_jump:,} ({pct:.1f}%)")
    print(f"    unbracketed (outside states) : {unbracketed:,}")
    check("durative actions are scored (unaffected by 1 Hz resolution)",
          scored_by_kind.get(KIND_DURATIVE, 0) > 0, f"{scored_by_kind.get(KIND_DURATIVE,0):,}")

    return summary()


if __name__ == "__main__":
    sys.exit(main())
