"""
Match metadata at download time (P1.1.4, P1.1.5): the `matches` table holds every
P1.1.4 column, `demo_sha256` is unique, and a duplicate keeps the earlier row.

    .venv\\Scripts\\python.exe test\\engine\\test_matches_meta.py
"""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="ch_meta_test_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")

from engine import db  # noqa: E402

P114 = ("match_key", "source", "event", "event_tier", "date", "team_ct_start", "team_t_start",
        "map", "demo_sha256", "file_bytes", "game_version", "tick_rate", "status")


def main() -> int:
    section("columns and round trip")
    db.init_db()
    check("every P1.1.4 column exists", set(P114) <= set(db.MATCH_COLUMNS))
    meta = {"match_key": "iem-cologne_2026-08-10_navi-faze_de_mirage", "source": "hltv",
            "event": "IEM Cologne", "event_tier": 1, "date": "2026-08-10", "team_ct_start": "NAVI",
            "team_t_start": "FaZe", "map": "de_mirage", "map_version": "de_mirage@unknown",
            "demo_sha256": "e" * 64, "file_bytes": 123456789, "game_version": "14090",
            "tick_rate": 128, "tier": "pro"}
    db.register_match(meta)
    row = db.get_match(meta["match_key"])
    check("all supplied values round-trip", all(row[k] == v for k, v in meta.items()),
          str({k: (row[k], v) for k, v in meta.items() if row[k] != v}))
    check("status defaults to downloaded", row["status"] == "downloaded")
    db.register_match({"match_key": "faceit-1", "source": "faceit_api", "tier": "faceit_10",
                       "demo_sha256": "f" * 64, "map": "de_nuke", "date": "2026-09-02"})
    con = sqlite3.connect(db.db_path())
    rows = con.execute("SELECT match_key FROM matches WHERE tier='pro' AND map='de_mirage'").fetchall()
    check("query by tier and map", rows == [(meta["match_key"],)], str(rows))
    n = con.execute("SELECT COUNT(*) FROM matches WHERE substr(date,1,7)='2026-09'").fetchone()[0]
    check("query by month", n == 1)
    unknown = con.execute("SELECT event, event_tier, tick_rate FROM matches WHERE match_key='faceit-1'").fetchone()
    check("unknown fields are NULL", unknown == (None, None, None), str(unknown))

    section("sha256 uniqueness and duplicates (P1.1.5)")
    try:
        con.execute("INSERT INTO matches (match_key, demo_sha256) VALUES ('raw-dup', ?)", ("e" * 64,))
        check("unique index rejects a raw duplicate sha256", False)
    except sqlite3.IntegrityError:
        check("unique index rejects a raw duplicate sha256", True)
    con.rollback()
    idx = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='matches'").fetchone()[0]
    check("demo_sha256 declared UNIQUE", "demo_sha256 TEXT UNIQUE" in " ".join(idx.split()))   # LOW-78: any spacing
    con.close()
    key = db.register_match({**meta, "match_key": "hf-same-file", "source": "hf_archive"})
    check("same file from another source keeps the earlier row", key == meta["match_key"]
          and db.get_match("hf-same-file") is None and db.get_match(key)["source"] == "hltv")
    check("NULL sha256 does not collide", db.register_match({"match_key": "n1"}) == "n1"
          and db.register_match({"match_key": "n2"}) == "n2")
    for st in db.STATUSES:
        db.set_status("n1", st, "tick_rate" if st == "quarantined" else None)
    check("every P1.1.4 status accepted", db.get_match("n1")["status"] == "quarantined")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
