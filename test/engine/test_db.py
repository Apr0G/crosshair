"""
Engine database (Amendment 1): fresh schema, WAL, idempotent register_match, both id
forms, transactional action scores, migration mechanism, numpy-safe JSON.

    .venv\\Scripts\\python.exe test\\engine\\test_db.py

Runs on a temp file through CROSSHAIR_ENGINE_DB; data/ is never touched.
"""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="ch_db_test_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")

from engine import db  # noqa: E402

SECRET = "SUPERSECRETSIGNATURE123"


def _raw() -> sqlite3.Connection:
    return sqlite3.connect(db.db_path())


def test_fresh() -> None:
    section("fresh schema")
    db.init_db()
    con = _raw()
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check("four tables", {"matches", "runs", "action_scores", "processed"} <= tables, str(sorted(tables)))
    check("user_version = SCHEMA_VERSION = 1",
          con.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION == 1)
    check("journal_mode is WAL", con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal")
    cols = [r[1] for r in con.execute("PRAGMA table_info(matches)")]
    check("matches has the Appendix A.6 columns in order", tuple(cols) == db.MATCH_COLUMNS, str(cols))
    cols = [r[1] for r in con.execute("PRAGMA table_info(action_scores)")]
    check("action_scores columns", tuple(cols) == db.SCORE_COLUMNS, str(cols))
    cols = [r[1] for r in con.execute("PRAGMA table_info(processed)")]
    check("processed columns", cols == ["match_id", "map", "demo_url", "processed_at"], str(cols))
    idx = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    check("both action_scores indices", {"idx_eas_match", "idx_eas_run"} <= idx, str(sorted(idx)))
    plan = " ".join(str(r) for r in con.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM action_scores WHERE match_key='m' AND round_num=3"))
    check("report query uses idx_eas_match", "idx_eas_match" in plan, plan)
    con.close()
    s = db.stats()
    check("stats on an empty db", s["matches"] == 0 and s["processed"] == 0 and s["by_status"] == {}, str(s))


def test_register() -> None:
    section("register_match")
    meta = {"match_key": "1-abc", "source": "faceit_api", "tier": "faceit_10", "map": "de_mirage",
            "demo_sha256": "a" * 64, "file_bytes": 1234, "tick_rate": 128, "date": "2026-09-01"}
    url  = f"https://demos.faceit-cdn.net/cs2/m.dem.zst?sig={SECRET}"
    key  = db.register_match(meta, url)
    check("returns the key", key == "1-abc")
    row = db.get_match("1-abc")
    check("row stored with defaults", row and row["status"] == "downloaded" and row["created_at"]
          and row["quarantine_reason"] is None and row["event"] is None, str(row))
    con = _raw()
    stored = con.execute("SELECT demo_url, map FROM processed WHERE match_id='1-abc'").fetchone()
    check("processed row written with a redacted url",
          stored == ("https://demos.faceit-cdn.net/cs2/m.dem.zst?<redacted>", "de_mirage"), str(stored))
    con.close()
    check("idempotent: same key again", db.register_match(meta, url) == "1-abc"
          and db.stats()["matches"] == 1 and db.stats()["processed"] == 1)

    dup = {**meta, "match_key": "hltv-other", "source": "hltv"}
    key = db.register_match(dup)
    check("duplicate sha256 returns the existing key", key == "1-abc", key)
    check("duplicate keeps the earlier row only", db.get_match("hltv-other") is None
          and db.get_match("1-abc")["source"] == "faceit_api")
    check("duplicate id is marked processed (no re-download)", db.is_processed("hltv-other"))

    for bad, why in [({"source": "x"}, "no match_key"), ({**meta, "match_key": "z", "bogus": 1}, "unknown column"),
                     ({**meta, "match_key": "z", "demo_sha256": "b" * 64, "status": "weird"}, "bad status")]:
        try:
            db.register_match(bad)
            check(f"rejects {why}", False, "no exception")
        except ValueError:
            check(f"rejects {why}", True)
    check("nothing half-written by the rejects", db.get_match("z") is None and not db.is_processed("z"))


def test_id_forms() -> None:
    section("is_processed: both id forms")
    db.register_match({"match_key": "1-series_m1", "demo_sha256": "c" * 64})
    check("exact id", db.is_processed("1-series_m1"))
    check("bare id is not an exact hit", not db.is_processed("1-series"))
    check("is_processed_any(bare) sees <mid>_m1", db.is_processed_any("1-series"))
    check("is_processed_any(bare) sees <mid>", db.is_processed_any("1-abc"))
    check("a stored _m1 does not block _m2", not db.is_processed_any("1-series_m2"))
    db.register_match({"match_key": "1-bo1", "demo_sha256": "d" * 64})
    check("is_processed_any(<mid>_m1) sees the BO1 form <mid>", db.is_processed_any("1-bo1_m1"))
    check("unknown id is not processed", not db.is_processed_any("1-nothing"))
    check("prefix lookup", db.has_processed_prefix("1-ser") and not db.has_processed_prefix("1-ser%x"))
    check("LIKE wildcards are escaped", not db.has_processed_prefix("1_s"))


def test_status_runs() -> None:
    section("set_status / upsert_run")
    db.set_status("1-abc", "quarantined", "tick_rate")
    row = db.get_match("1-abc")
    check("status + reason set", row["status"] == "quarantined" and row["quarantine_reason"] == "tick_rate")
    db.set_status("1-abc", "parsed")
    check("reason cleared on a normal status", db.get_match("1-abc")["quarantine_reason"] is None)
    for args, exc, why in [(("nope", "parsed"), KeyError, "unknown match"),
                           (("1-abc", "bogus"), ValueError, "unknown status")]:
        try:
            db.set_status(*args)
            check(f"set_status refuses {why}", False)
        except exc:
            check(f"set_status refuses {why}", True)
    db.upsert_run("r1", "parse", started_at="2026-10-19T00:00:00+00:00", status="running")
    db.upsert_run("r1", finished_at="2026-10-19T00:01:00+00:00", status="ok", manifest_path=TMP / "m.json")
    run = db.list_runs()[0]
    check("upsert keeps earlier fields and updates later ones",
          run["kind"] == "parse" and run["started_at"].startswith("2026-10-19T00:00")
          and run["status"] == "ok" and run["manifest_path"].endswith("m.json"), str(run))


def test_scores() -> None:
    section("insert_action_scores")
    import numpy as np
    rows = [{"action_id": f"a{i}", "match_key": "1-abc", "round_num": np.int64(1), "run_id": "r1",
             "V": np.float32(0.25), "n_coalition": np.int32(2), "flags": {"ok": np.bool_(True), "k": np.int64(3),
                                                                        "arr": np.arange(2)}}
            for i in range(3)]
    check("writes all rows", db.insert_action_scores(rows) == 3)
    con = _raw()
    got = con.execute("SELECT round_num, V, n_coalition, flags FROM action_scores WHERE action_id='a0'").fetchone()
    check("numpy scalars stored as plain numbers", got[0] == 1 and abs(got[1] - 0.25) < 1e-6 and got[2] == 2, str(got))
    check("numpy flags JSON-encoded", got[3] == '{"arr": [0, 1], "k": 3, "ok": true}', got[3])
    db.insert_action_scores(rows[:2])
    n = con.execute("SELECT COUNT(*) FROM action_scores WHERE run_id='r1'").fetchone()[0]
    check("re-insert replaces the (match_key, run_id) group, not appends", n == 2, str(n))
    con.close()

    before = db.stats()["action_scores"]
    for bad, why in [([rows[0], {"match_key": "1-abc", "run_id": "r1"}], "row without action_id"),
                     ([rows[0], rows[0]], "duplicate primary key inside the batch"),
                     ([rows[0], {**rows[1], "nonsense": 1}], "unknown column")]:
        try:
            db.insert_action_scores(bad)
            check(f"bad batch raises ({why})", False)
        except (ValueError, sqlite3.IntegrityError):
            check(f"bad batch raises ({why})", True)
        check(f"rolled back, earlier rows intact ({why})", db.stats()["action_scores"] == before)
    check("_Encoder handles numpy", db._dumps({"x": np.float64(1.5)}) == '{"x": 1.5}')


def test_migrations() -> None:
    section("migration mechanism")
    old_v, old_m = db.SCHEMA_VERSION, dict(db._MIGRATIONS)
    try:
        db.SCHEMA_VERSION = 2
        db._MIGRATIONS[2] = ["ALTER TABLE matches ADD COLUMN note TEXT"]
        db.init_db()
        con  = _raw()
        cols = [r[1] for r in con.execute("PRAGMA table_info(matches)")]
        ver  = con.execute("PRAGMA user_version").fetchone()[0]
        con.close()
        check("v1 file migrated to v2 (column added, version bumped)", "note" in cols and ver == 2, f"v{ver}")
        check("rows survive the migration", db.get_match("1-abc") is not None)
        db.init_db()
        check("re-open at v2 is a no-op", True)
    finally:
        db.SCHEMA_VERSION = old_v
        db._MIGRATIONS.clear()
        db._MIGRATIONS.update(old_m)
    try:
        db.init_db()
        check("a newer file (user_version 2 > 1) is refused", False, "opened")
    except db.SchemaVersionError as e:
        check("a newer file (user_version 2 > 1) is refused", "newer" in str(e), str(e)[:90])

    os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "future.db")
    con = sqlite3.connect(db.db_path())
    con.execute("PRAGMA user_version = 99")
    con.commit()
    con.close()
    try:
        db.is_processed("x")
        check("an empty file stamped v99 is refused, not initialised", False)
    except db.SchemaVersionError:
        check("an empty file stamped v99 is refused, not initialised", True)
    os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")


def main() -> int:
    test_fresh()
    test_register()
    test_id_forms()
    test_status_runs()
    test_scores()
    test_migrations()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
