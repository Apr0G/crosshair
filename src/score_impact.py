"""
Score every event in the DB with WP-before / WP-after / impact.

For each event:
  - p_before = CT win probability at the round_state just before the event
  - p_after  = CT win probability at the round_state just after the event
  - impact   = signed delta from the actor's perspective
                 ct actor: p_after - p_before
                 t  actor: p_before - p_after   (i.e. negative of CT delta)

Usage:
    python src/score_impact.py                     # all matches
    python src/score_impact.py --match-id <id>     # single match
    python src/score_impact.py --limit 50          # first N matches
"""
import argparse
import sqlite3
import time
from pathlib import Path
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

from win_probability import FEATURES, load_model

DB_PATH = Path(__file__).parent.parent / "data" / "crosshair.db"


def _score_states(states_df: pd.DataFrame) -> np.ndarray:
    """Run the WP model on a batch of round_states rows."""
    X = states_df[FEATURES].copy()
    X["map"] = X["map"].astype("category")
    return load_model().predict(X)


def _impact_for_event(t_event: float, side: str, state_times: list[float], wp: np.ndarray
                      ) -> tuple[float | None, float | None, float | None]:
    """Find bracketing round_states and compute (p_before, p_after, impact)."""
    if not state_times:
        return None, None, None

    # p_before: latest state with time <= t_event
    i_before = bisect_right(state_times, t_event) - 1
    # p_after:  earliest state with time >  t_event
    i_after  = bisect_left(state_times, t_event)
    # if the event lands exactly on a sample, take the next one as "after"
    if i_after < len(state_times) and state_times[i_after] <= t_event:
        i_after += 1

    p_before = float(wp[i_before]) if 0 <= i_before < len(wp) else None
    p_after  = float(wp[i_after])  if 0 <= i_after  < len(wp) else None

    if p_before is None or p_after is None:
        return p_before, p_after, None

    delta = p_after - p_before
    impact = delta if side == "ct" else -delta
    return p_before, p_after, float(impact)


def score_match(con: sqlite3.Connection, match_id: str) -> int:
    """Score all events in one match. Returns # of events updated."""
    states = pd.read_sql_query(
        "SELECT * FROM round_states WHERE match_id = ? "
        "ORDER BY round_num, time_into_round_s",
        con, params=(match_id,),
    )
    if states.empty:
        return 0

    states["map"] = states["map"].astype("category")
    wp_all = _score_states(states)

    # Bucket states + WP by round
    by_round: dict[int, tuple[list[float], np.ndarray]] = {}
    for rn, group in states.groupby("round_num", sort=False):
        idx = group.index.to_numpy()
        by_round[int(rn)] = (
            group["time_into_round_s"].astype(float).tolist(),
            wp_all[idx],
        )

    # Pull events for this match
    events = pd.read_sql_query(
        "SELECT id, round_num, player_side, time_into_round FROM events WHERE match_id = ?",
        con, params=(match_id,),
    )
    if events.empty:
        return 0

    updates = []
    for row in events.itertuples(index=False):
        rn   = int(row.round_num) if pd.notna(row.round_num) else None
        if rn is None or rn not in by_round:
            continue
        t    = float(row.time_into_round) if pd.notna(row.time_into_round) else 0.0
        side = (row.player_side or "").lower()
        state_times, wp = by_round[rn]
        p_b, p_a, imp = _impact_for_event(t, side, state_times, wp)
        updates.append((p_b, p_a, imp, int(row.id)))

    if not updates:
        return 0

    con.executemany(
        "UPDATE events SET p_before=?, p_after=?, impact=? WHERE id=?",
        updates,
    )
    con.commit()
    return len(updates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", help="Score only this match")
    ap.add_argument("--limit",    type=int, help="Score first N matches")
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH, timeout=30)
    try:
        if args.match_id:
            match_ids = [args.match_id]
        else:
            q = "SELECT match_id FROM processed_matches ORDER BY match_id"
            if args.limit:
                q += f" LIMIT {int(args.limit)}"
            match_ids = [r[0] for r in con.execute(q).fetchall()]

        print(f"Scoring {len(match_ids)} match(es) ...")
        t0 = time.time()
        total = 0
        for i, mid in enumerate(match_ids, 1):
            n = score_match(con, mid)
            total += n
            if i % 10 == 0 or i == len(match_ids):
                rate = total / max(time.time() - t0, 1e-9)
                eta = (len(match_ids) - i) * ((time.time() - t0) / i)
                print(f"  [{i}/{len(match_ids)}] {mid[:36]} → {n:>5} events "
                      f"({total:,} total, {rate:.0f} ev/s, eta {eta:.0f}s)")
        print(f"\nDone. Updated {total:,} events in {time.time() - t0:.1f}s.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
