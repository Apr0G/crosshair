"""
Tests for db.py — schema creation, migrations and the one-transaction seam write.

Run from repo root:
    .venv/bin/python test/test_db.py

Stdlib only: db imports numpy lazily, so this runs on a bare interpreter. Every
check uses a throw-away database in a temp dir; data/crosshair.db is never touched.
"""
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _harness import check, section, summary
import db


def _state_keys() -> list[str]:
    """Every :param the round_states INSERT binds — the contract a state dict meets."""
    return re.findall(r":([a-z_]+)", db._STATES_SQL)


def _state(match_id="m1", round_num=1, tick=1000, **over) -> dict:
    s = {k: 0 for k in _state_keys()}
    s.update(match_id=match_id, map="de_test", round_num=round_num, tick=tick,
             time_into_round_s=float(tick - 1000) / 64, time_remaining_s=100.0,
             alive_ct=5, alive_t=5, ct_spread=None, t_spread=None,
             min_dist_ct_to_bomb=None, min_dist_t_to_bomb=None,
             active_smokes_xy=[], active_infernos_xy=[], round_won_ct=1)
    s.pop("state_kind")           # the writer defaults it; tests set it explicitly
    s.update(over)
    return s


def _event(match_id="m1", round_num=1, t=1.0, side="ct", etype="engagement") -> dict:
    return {"match_id": match_id, "map": "de_test", "round_num": round_num,
            "player_side": side, "event_type": etype, "time_into_round": t,
            "situation": {"hp": 100}, "action": {"impact_kind": "instantaneous"},
            "outcome": {"result": "kill"}, "round_won": 1}


def _cols(con, table) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _make_v0_db(path: Path, with_impact: bool = False) -> None:
    """A database as _init_schema wrote it before any migration existed."""
    con = sqlite3.connect(path)
    extra = ", p_before REAL, p_after REAL, impact REAL" if with_impact else ""
    con.executescript(f"""
        CREATE TABLE processed_matches (match_id TEXT PRIMARY KEY, map TEXT, demo_url TEXT,
                                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT NOT NULL,
                             map TEXT, round_num INTEGER, player_side TEXT, event_type TEXT,
                             time_into_round REAL, situation TEXT, action TEXT, outcome TEXT,
                             round_won INTEGER{extra});
        CREATE TABLE round_states (id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT NOT NULL,
                             map TEXT, round_num INTEGER, tick INTEGER, alive_ct INTEGER,
                             alive_t INTEGER, round_won_ct INTEGER);
        INSERT INTO processed_matches (match_id, map) VALUES ('old', 'de_old');
        INSERT INTO round_states (match_id, round_num, tick, alive_ct, alive_t, round_won_ct)
            VALUES ('old', 1, 100, 5, 5, 1);
    """)
    con.commit()
    con.close()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ch_db_test_"))
    db.DB_PATH = tmp / "fresh.db"

    section("fresh database")
    db.init_db()
    con = sqlite3.connect(db.DB_PATH)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check("init_db creates the three tables",
          {"processed_matches", "events", "round_states"} <= tables, str(sorted(tables)))
    check("fresh DB is stamped with the current schema version",
          con.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION)
    check("fresh DB is in WAL mode",
          con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal")
    check("every column store_match binds exists in a fresh round_states",
          set(_state_keys()) <= _cols(con, "round_states"),
          str(sorted(set(_state_keys()) - _cols(con, "round_states"))))
    check("events has the scoring columns", {"p_before", "p_after", "impact"} <= _cols(con, "events"))
    idx = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    check("fresh DB carries the state_kind index (not just via migration)", "idx_states_kind" in idx)
    con.close()
    check("re-running init_db on an existing DB is harmless", (db.init_db() or True))

    section("store_match: one transaction, idempotent")
    events = [_event(t=1.0, side="ct"), _event(t=2.0, side="t")]
    states = [_state(tick=1000), _state(tick=1064), _state(tick=1031, state_kind=db.STATE_BOUNDARY)]
    db.store_match("m1", events, states, "de_test", "https://x/y.dem?<redacted>")
    check("match is marked processed", db.is_processed("m1"))
    check("unknown match is not processed", not db.is_processed("nope"))
    s = db.stats()
    check("stats counts what was written",
          (s["matches_processed"], s["total_events"], s["total_states"]) == (1, 2, 3), str(s))
    check("events_by_type is tallied", s["events_by_type"] == {"engagement": 2}, str(s["events_by_type"]))

    con = sqlite3.connect(db.DB_PATH)
    kinds = [r[0] for r in con.execute("SELECT state_kind FROM round_states ORDER BY tick")]
    check("state_kind defaults to grid and keeps an explicit boundary",
          kinds == [db.STATE_GRID, db.STATE_BOUNDARY, db.STATE_GRID], str(kinds))
    sit = con.execute("SELECT situation, action FROM events ORDER BY id LIMIT 1").fetchone()
    check("json blobs are serialised as text", sit == ('{"hp": 100}', '{"impact_kind": "instantaneous"}'), str(sit))
    url = con.execute("SELECT demo_url, map FROM processed_matches").fetchone()
    check("demo_url and map land on the processed row", url == ("https://x/y.dem?<redacted>", "de_test"))
    con.close()

    # Store the same match again with different content: replaced, not appended.
    db.store_match("m1", events[:1], states[:1], "de_test", None)
    s = db.stats()
    check("re-storing a match REPLACES its rows (no double insert)",
          (s["matches_processed"], s["total_events"], s["total_states"]) == (1, 1, 1), str(s))

    section("store_match: a failure leaves nothing behind")
    bad_event = {k: v for k, v in _event().items() if k != "round_won"}   # missing bind
    bad_event["match_id"] = "m2"
    try:
        db.store_match("m2", [bad_event], [_state(match_id="m2")], "de_test", None)
        check("a malformed row raises", False, "no exception")
    except Exception as e:
        check("a malformed row raises", True, type(e).__name__)
    check("the failed match is NOT marked processed", not db.is_processed("m2"))
    con = sqlite3.connect(db.DB_PATH)
    n = con.execute("SELECT COUNT(*) FROM round_states WHERE match_id='m2'").fetchone()[0]
    check("no partial rows survive the rollback", n == 0, f"{n} rows")
    con.close()

    section("numpy values in a row are encoded")
    try:
        import numpy as np
    except ImportError:
        print("  – numpy not installed, skipping")
    else:
        ev = _event(match_id="m3")
        ev["situation"] = {"n": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True),
                           "arr": np.array([1, 2])}
        db.store_match("m3", [ev], [], "de_test", None)
        con = sqlite3.connect(db.DB_PATH)
        blob = con.execute("SELECT situation FROM events WHERE match_id='m3'").fetchone()[0]
        con.close()
        check("numpy scalars/arrays serialise to plain JSON",
              blob == '{"n": 3, "f": 0.5, "b": true, "arr": [1, 2]}', blob)

    section("migration of a pre-migration database")
    db.DB_PATH = tmp / "old.db"
    _make_v0_db(db.DB_PATH)
    db.init_db()
    con = sqlite3.connect(db.DB_PATH)
    check("user_version is bumped to the current schema",
          con.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION)
    check("v1 adds the scoring columns to events", {"p_before", "p_after", "impact"} <= _cols(con, "events"))
    check("v1 adds the smoke/fire columns to round_states",
          {"active_smokes_xy", "active_infernos_xy", "site_smoked", "site_on_fire"} <= _cols(con, "round_states"))
    check("v2 adds state_kind", "state_kind" in _cols(con, "round_states"))
    kind = con.execute("SELECT state_kind FROM round_states WHERE match_id='old'").fetchone()[0]
    check("pre-existing rows keep NULL state_kind (= grid)", kind is None)
    check("existing data survives the migration", db.is_processed("old"))
    con.close()

    db.DB_PATH = tmp / "old_partial.db"
    _make_v0_db(db.DB_PATH, with_impact=True)
    try:
        db.init_db()
        con = sqlite3.connect(db.DB_PATH)
        ok = con.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        con.close()
        check("a column added out of band does not break the migration", ok)
    except Exception as e:
        check("a column added out of band does not break the migration", False, repr(e))

    section("an empty file is initialised, not mistaken for a database")
    db.DB_PATH = tmp / "empty.db"
    db.DB_PATH.touch()
    db.init_db()
    con = sqlite3.connect(db.DB_PATH)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    check("schema is created inside a pre-existing empty file", "round_states" in tables)

    return summary()


if __name__ == "__main__":
    sys.exit(main())
