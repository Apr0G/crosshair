"""
Tests for score_impact.py and the model-free half of win_probability.py.

Run from repo root:
    .venv/bin/python test/test_score_impact.py

Needs pandas/numpy. No trained model: the WP model is replaced by a stub that
returns whatever win probabilities the scenario needs, so what is under test is
the bracketing, the sign convention (P5), the terminal pinning, the clear-then-
write update, and the frame construction LightGBM is handed.
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from _harness import check, close, section, summary
import db
import score_impact as si
import win_probability as wp
from attribution import KIND_INSTANT, KIND_DURATIVE


def _mem_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    db._init_schema(con)
    return con


def _insert_states(con, match_id, rows):
    """rows: list of (round_num, tick, time_s, alive_ct, alive_t, post_plant)."""
    for rn, tick, t, act, at, pp in rows:
        con.execute(
            "INSERT INTO round_states (match_id, map, round_num, tick, time_into_round_s, "
            "alive_ct, alive_t, post_plant, round_won_ct) VALUES (?, 'de_test', ?, ?, ?, ?, ?, ?, 1)",
            (match_id, rn, tick, t, act, at, pp))


def _insert_event(con, match_id, rn, t, side, action: dict | None = None) -> int:
    cur = con.execute(
        "INSERT INTO events (match_id, round_num, player_side, event_type, time_into_round, action) "
        "VALUES (?, ?, ?, 'engagement', ?, ?)",
        (match_id, rn, side, t, json.dumps(action) if action is not None else None))
    return cur.lastrowid


def main() -> int:
    section("_impact_for_event: sign convention (P5) and bracketing")
    times = [0.0, 1.0, 2.0, 3.0]
    probs = np.array([0.50, 0.70, 0.20, 0.60])
    pb, pa, imp = si._impact_for_event(0.5, "ct", times, probs)
    check("CT actor: impact = p_after - p_before", close(imp, 0.20) and close(pb, 0.5) and close(pa, 0.7), f"{pb} {pa} {imp}")
    pb, pa, imp = si._impact_for_event(1.5, "t", times, probs)
    check("T actor: impact is negated", close(imp, 0.50), f"{imp}")
    pb, pa, imp = si._impact_for_event(1.5, "?", times, probs)
    check("unknown side: p_before/p_after kept, impact None", imp is None and pb is not None and pa is not None)
    pb, pa, imp = si._impact_for_event(1.0, "ct", times, probs)
    check("an event ON a sample uses that sample as 'before' and the next as 'after'",
          close(pb, 0.7) and close(pa, 0.2) and close(imp, -0.5), f"{pb} {pa} {imp}")
    pb, pa, imp = si._impact_for_event(1.0, "ct", [0.0, 1.0, 1.0, 2.0], np.array([0.5, 0.6, 0.65, 0.9]))
    check("duplicate sample timestamps: 'after' skips every equal sample", close(pa, 0.9), f"{pa}")
    pb, pa, imp = si._impact_for_event(-1.0, "ct", times, probs)
    check("before the first sample: p_before None, impact None", pb is None and imp is None)
    pb, pa, imp = si._impact_for_event(3.0, "ct", times, probs)
    check("on/after the last sample: p_after None, impact None", pa is None and imp is None)
    check("no states at all: all None", si._impact_for_event(1.0, "ct", [], np.array([])) == (None, None, None))

    section("_actions_for_round: parses the classification out of the action blob")
    ev = pd.DataFrame([
        {"id": 1, "player_side": "CT", "action": json.dumps({"impact_kind": KIND_INSTANT, "resolve_tick": 100})},
        {"id": 2, "player_side": "t",  "action": json.dumps({"impact_kind": KIND_DURATIVE, "window_start_tick": 10, "window_end_tick": 90})},
        {"id": 3, "player_side": "t",  "action": json.dumps({"impact_kind": KIND_DURATIVE, "window_start_tick": None, "window_end_tick": 90})},
        {"id": 4, "player_side": "t",  "action": json.dumps({"role": "victim"})},
        {"id": 5, "player_side": None, "action": "{not json"},
        {"id": 6, "player_side": None, "action": None},
    ])
    acts = si._actions_for_round(ev)
    check("only classified, well-formed actions survive", [a.event_id for a in acts] == [1, 2], str([a.event_id for a in acts]))
    check("side is lower-cased", acts[0].side == "ct")
    check("instantaneous carries its resolve tick", acts[0].kind == KIND_INSTANT and acts[0].resolve_tick == 100)
    check("durative carries an int window", acts[1].window == (10, 90) and acts[1].resolve_tick is None)

    section("_wp_path: interpolation + terminal pinning (model stubbed)")
    states = pd.DataFrame([
        {"tick": 0,   "alive_ct": 5, "alive_t": 5, "post_plant": 0},
        {"tick": 100, "alive_ct": 5, "alive_t": 5, "post_plant": 0},
        {"tick": 200, "alive_ct": 5, "alive_t": 0, "post_plant": 0},   # CT win: pinned 1.0
        {"tick": 300, "alive_ct": 0, "alive_t": 2, "post_plant": 0},   # (nonsense, but) pinned 0.0
        {"tick": 400, "alive_ct": 3, "alive_t": 0, "post_plant": 1},   # bomb down: NOT terminal
    ])
    original = si._score_states
    si._score_states = lambda df: np.array([0.5, 0.7, 0.3, 0.9, 0.42])
    try:
        at, lo, hi = si._wp_path(states)
        check("range is the first and last tick", (lo, hi) == (0, 400))
        check("exact ticks return the model value", close(at(100), 0.7))
        check("between ticks is linearly interpolated", close(at(50), 0.6), f"{at(50)}")
        check("alive_t == 0 pre-plant is pinned to 1.0 regardless of the model", close(at(200), 1.0), f"{at(200)}")
        check("alive_ct == 0 is pinned to 0.0", close(at(300), 0.0), f"{at(300)}")
        check("alive_t == 0 POST-plant is left to the model", close(at(400), 0.42), f"{at(400)}")
        check("outside the round returns None", at(-1) is None and at(401) is None and at(None) is None)
    finally:
        si._score_states = original

    section("_score_states: all-NULL feature columns do not break prediction")
    class _Booster:
        def predict(self, X):
            assert all(str(X[c].dtype) != "object" for c in X.columns if c != "map"), "object column reached the model"
            return np.full(len(X), 0.5)
    wp._model = _Booster()
    try:
        df = pd.DataFrame([{f: None for f in wp.FEATURES} for _ in range(3)])
        df["map"] = "de_test"
        df["alive_ct"] = 5
        out = si._score_states(df)
        check("object-typed NULL columns are coerced before the model sees them", len(out) == 3)
    except AssertionError as e:
        check("object-typed NULL columns are coerced before the model sees them", False, str(e))
    finally:
        wp._model = None

    section("score_match (grid) end to end on an in-memory DB")
    con = _mem_db()
    _insert_states(con, "m", [(1, 0, 0.0, 5, 5, 0), (1, 64, 1.0, 5, 5, 0), (1, 128, 2.0, 5, 4, 0),
                              (2, 1000, 0.0, 5, 5, 0), (2, 1064, 1.0, 4, 5, 0)])
    e_ct   = _insert_event(con, "m", 1, 0.5, "ct")
    e_t    = _insert_event(con, "m", 1, 1.5, "t")
    e_none = _insert_event(con, "m", 1, None, "ct")        # NULL time: must stay unscored
    e_r9   = _insert_event(con, "m", 9, 0.5, "ct")         # round with no states
    e_r2   = _insert_event(con, "m", 2, 0.5, "T")
    # stale values from a "previous model" that the pass must clear
    con.execute("UPDATE events SET impact=0.99, p_before=0.1, p_after=0.1 WHERE id IN (?, ?)", (e_none, e_r9))
    si._score_states = lambda df: np.array([0.5, 0.7, 0.2, 0.4, 0.3])[: len(df)]
    try:
        n = si.score_match(con, "m")
    finally:
        si._score_states = original
    got = {r[0]: r[1:] for r in con.execute("SELECT id, p_before, p_after, impact FROM events")}
    check("three events scored", n == 3, str(n))
    check("CT event in round 1 gets +0.20", close(got[e_ct][2], 0.20), str(got[e_ct]))
    check("T event in round 1 gets -( -0.50 ) = +0.50", close(got[e_t][2], 0.50), str(got[e_t]))
    check("round-2 event brackets round-2 states (not the global array)",
          close(got[e_r2][0], 0.4) and close(got[e_r2][1], 0.3) and close(got[e_r2][2], 0.1), str(got[e_r2]))
    check("NULL-time event is cleared, not scored as t=0", got[e_none] == (None, None, None), str(got[e_none]))
    check("event in a round with no states is cleared too", got[e_r9] == (None, None, None), str(got[e_r9]))
    check("a match with no states scores nothing", si.score_match(con, "absent") == 0)

    section("score_match_event (event-boundary) end to end")
    con2 = _mem_db()
    # ticks 0..400; the kill at 200 is bracketed by boundary rows at 199 / 201
    _insert_states(con2, "m", [(1, 0, 0.0, 5, 5, 0), (1, 100, 1.5, 5, 5, 0), (1, 199, 3.1, 5, 5, 0),
                               (1, 201, 3.2, 5, 4, 0), (1, 300, 4.7, 5, 4, 0), (1, 400, 6.2, 5, 4, 0)])
    kill  = _insert_event(con2, "m", 1, 3.1, "ct", {"impact_kind": KIND_INSTANT, "resolve_tick": 200})
    smoke = _insert_event(con2, "m", 1, 0.0, "t",  {"impact_kind": KIND_DURATIVE, "window_start_tick": 0, "window_end_tick": 400})
    buy   = _insert_event(con2, "m", 1, 0.0, "ct", {"impact_kind": "round"})
    plain = _insert_event(con2, "m", 1, 2.0, "ct", None)
    si._score_states = lambda df: np.array([0.50, 0.52, 0.55, 0.80, 0.82, 0.90])
    try:
        n, resid = si.score_match_event(con2, "m")
    finally:
        si._score_states = original
    got = {r[0]: r[1:] for r in con2.execute("SELECT id, p_before, p_after, impact FROM events")}
    check("rows updated = every classified action (incl. unvalued ones)", n == 3, str(n))
    check("conservation residual ~0", resid < 1e-9, f"{resid:.2e}")
    check("kill takes the jump across its boundary states", close(got[kill][2], 0.25), str(got[kill]))
    check("the T smoke gets the rest of the window, negated",
          close(got[smoke][2], -((0.90 - 0.50) - 0.25)), str(got[smoke]))
    check("buy is left unvalued (None), not scored", got[buy] == (None, None, None), str(got[buy]))
    check("an unclassified event is not touched by the event method", got[plain] == (None, None, None))

    section("run(): guards")
    saved = si.DB_PATH
    si.DB_PATH = Path("definitely/not/here.db")
    try:
        check("missing DB returns 1 without raising", si.run() == 1)
    finally:
        si.DB_PATH = saved

    section("win_probability._states_to_frame")
    frame = wp._states_to_frame([
        {f: 1 for f in wp.FEATURES} | {"map": "de_mirage"},
        {f: None for f in wp.FEATURES} | {"map": float("nan")},
        {"alive_ct": 3},
    ])
    check("columns are exactly FEATURES, in order", list(frame.columns) == wp.FEATURES)
    check("map is categorical", str(frame["map"].dtype) == "category")
    check("NaN / missing map becomes 'unknown', not 'nan'",
          list(frame["map"]) == ["de_mirage", "unknown", "unknown"], str(list(frame["map"])))
    check("None numerics become float NaN, never object dtype",
          all(frame[c].dtype != object for c in wp.FEATURES if c != "map")
          and np.isnan(frame.loc[1, "alive_ct"]) and np.isnan(frame.loc[2, "alive_t"]))
    check("_map_label handles None/NaN/str", [wp._map_label(None), wp._map_label(float("nan")), wp._map_label("de_x")]
          == ["unknown", "unknown", "de_x"])
    check("predict_batch([]) is empty without touching the model", len(wp.predict_batch([])) == 0)

    return summary()


if __name__ == "__main__":
    sys.exit(main())
