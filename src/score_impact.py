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
import sys
import time
from pathlib import Path
from bisect import bisect_right

import numpy as np
import pandas as pd

from win_probability import FEATURES, load_model

DB_PATH = Path(__file__).parent.parent / "data" / "crosshair.db"


def _score_states(states_df: pd.DataFrame) -> np.ndarray:
    """Run the WP model on a batch of round_states rows."""
    X = states_df[FEATURES].copy()
    X["map"] = X["map"].astype("category")
    # A nullable feature that is NULL for EVERY row of a match comes back from
    # sqlite as object dtype, and LightGBM rejects the frame outright. That is not
    # hypothetical: a match with no bomb plant in any round has min_dist_*_to_bomb
    # all-NULL. Coerce to float so NULL stays NaN, which LightGBM handles natively.
    for col in X.columns:
        if col != "map" and X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return load_model().predict(X)


def _impact_for_event(t_event: float, side: str, state_times: list[float], wp: np.ndarray
                      ) -> tuple[float | None, float | None, float | None]:
    """Find bracketing round_states and compute (p_before, p_after, impact)."""
    if not state_times:
        return None, None, None

    # p_before: latest state with time <= t_event
    i_before = bisect_right(state_times, t_event) - 1
    # p_after:  earliest state with time strictly >  t_event
    # (bisect_right already skips every sample equal to t_event, including duplicates)
    i_after  = bisect_right(state_times, t_event)

    p_before = float(wp[i_before]) if 0 <= i_before < len(wp) else None
    p_after  = float(wp[i_after])  if 0 <= i_after  < len(wp) else None

    if p_before is None or p_after is None:
        return p_before, p_after, None

    delta = p_after - p_before
    # Anything that is not explicitly ct or t must not silently take the T branch —
    # feature_extractor writes "?" whenever the parser's side column is missing.
    if side == "ct":
        impact = delta
    elif side == "t":
        impact = -delta
    else:
        return p_before, p_after, None
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

    # wp_all is positional; group.index is label-based. They coincide only because
    # read_sql_query hands back a RangeIndex — make that explicit so a future filter
    # or sort here can't silently score events against the wrong states.
    states = states.reset_index(drop=True)
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
        # 0.0 is a real sample timestamp (the round's first state), so defaulting a
        # missing time to it would confidently attribute the first second's swing.
        if pd.isna(row.time_into_round):
            continue
        t    = float(row.time_into_round)
        side = (row.player_side or "").lower()
        state_times, wp = by_round[rn]
        p_b, p_a, imp = _impact_for_event(t, side, state_times, wp)
        updates.append((p_b, p_a, imp, int(row.id)))

    if not updates:
        return 0

    # Clear first: an event this pass skips (no states for its round, NULL time)
    # would otherwise keep values from a previous model and silently mix scales.
    con.execute(
        "UPDATE events SET p_before=NULL, p_after=NULL, impact=NULL WHERE match_id=?",
        (match_id,),
    )
    con.executemany(
        "UPDATE events SET p_before=?, p_after=?, impact=? WHERE id=?",
        updates,
    )
    con.commit()
    return len(updates)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", help="Score only this match")
    ap.add_argument("--limit",    type=int, help="Score first N matches")
    args = ap.parse_args(argv)

    if not DB_PATH.exists():
        print(f"No database at {DB_PATH}. Run `scrape` or `demo` first.")
        return 1

    # Fail before the loop rather than mid-match with a raw traceback.
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"{e}\nTrain first:  .venv/bin/python src/main.py train --eval")
        return 1

    con = sqlite3.connect(f"file:{DB_PATH}?mode=rw", uri=True, timeout=30)
    try:
        if args.match_id:
            match_ids = [args.match_id]
        else:
            q = "SELECT match_id FROM processed_matches ORDER BY match_id"
            if args.limit is not None:   # `if args.limit:` made --limit 0 mean "all"
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
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
