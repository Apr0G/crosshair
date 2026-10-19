"""
Tests for the seam: pipeline.ingest_demo / process_one / run and the main.py
commands that sit on top of them.

Run from repo root:
    .venv/bin/python test/test_pipeline.py

No awpy, no network, no browser: the demo parser is replaced by the synthetic
tables and the download by a stub, so what is exercised is the plumbing —
dedup, the one-transaction write, boundary states reaching the DB, URL
redaction in stored rows and error output, and the CLI wiring.
"""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _harness import check, section, summary
from _synthetic import make_tables, MAP_NAME
import db
import extract
import feature_extractor
import main as cli
import pipeline

SECRET = "SUPERSECRETSIGNATURE123"


class FakeSource:
    """A source module: yields demos, 'downloads' them, and can be told to fail."""
    def __init__(self, items, fail_ids=()):
        self.items, self.fail_ids, self.calls = items, set(fail_ids), []

    def iter_unprocessed_demos(self, **kw):
        yield from self.items

    def download_demo(self, url, dest: Path) -> Path:
        self.calls.append(url)
        for bad in self.fail_ids:
            if bad in str(dest):
                # what a real failure looks like when something upstream forgot to redact
                raise IOError(f"HTTP 500 fetching {url}")
        dest.write_bytes(b"PBDEMS2" + b"\x00" * 2000)
        return dest


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **kw)
    return rc, buf.getvalue()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ch_pipe_test_"))
    db.DB_PATH = cli.DB_PATH = tmp / "crosshair.db"

    # The parser is the only thing that needs awpy; everything after it is real.
    tables = make_tables(64)
    extract.extract = lambda path: tables
    feature_extractor.build_vis_checker = lambda map_name: None

    section("ingest_demo: the seam")
    (n_events, n_states), out = _quiet(pipeline.ingest_demo, "syn", "whatever.dem", "file://whatever.dem")
    check("returns the counts it stored", n_events > 0 and n_states > 0, f"{n_events} events, {n_states} states")
    check("match is marked processed with the given demo_url", db.is_processed("syn"))
    import sqlite3
    con = sqlite3.connect(db.DB_PATH)
    kinds = dict(con.execute("SELECT state_kind, COUNT(*) FROM round_states GROUP BY state_kind").fetchall())
    check("grid AND boundary states reach the DB", kinds.get("grid", 0) > 0 and kinds.get("boundary", 0) > 0, str(kinds))
    check("state count = grid + boundary", sum(kinds.values()) == n_states)
    types = dict(con.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type").fetchall())
    check("all event types stored", set(types) == {"buy", "engagement", "utility", "rotation", "bomb"}, str(types))
    row = con.execute("SELECT map, demo_url FROM processed_matches WHERE match_id='syn'").fetchone()
    check("map and demo_url recorded", row == (MAP_NAME, "file://whatever.dem"), str(row))
    check("boundary rows sit at tick±1 around the kills",
          {r[0] for r in con.execute("SELECT tick FROM round_states WHERE state_kind='boundary'")} >= {1999, 2001, 3999, 4001})
    con.close()
    check("output names each stage", all(s in out for s in ("parsing demo", "extracting events", "sampling round states", "storing")))

    section("process_one: download -> ingest, redacted URL, dedup")
    src = FakeSource([])
    signed = f"https://demos.faceit-cdn.net/cs2/m.dem.zst?sig={SECRET}"
    n, out = _quiet(pipeline.process_one, "1-abc", signed, source=src)
    check("downloads then ingests", n == n_events and src.calls == [signed], f"{n} events")
    con = sqlite3.connect(db.DB_PATH)
    stored = con.execute("SELECT demo_url FROM processed_matches WHERE match_id='1-abc'").fetchone()[0]
    con.close()
    check("the stored demo_url has its signature stripped", stored == "https://demos.faceit-cdn.net/cs2/m.dem.zst?<redacted>", stored)
    check("the signature never reaches stdout", SECRET not in out)
    n2, out2 = _quiet(pipeline.process_one, "1-abc", signed, source=src)
    check("a processed match is skipped without downloading", n2 == 0 and len(src.calls) == 1 and "skipping" in out2)
    for bad in ("../escape", "a b", "x;rm", ""):
        try:
            pipeline.process_one(bad, signed, source=src)
            check(f"unsafe match_id {bad!r} is refused", False, "no exception")
        except ValueError:
            check(f"unsafe match_id {bad!r} is refused", True)

    section("run: counts failures and redacts error output")
    fake = FakeSource([("1-ok", signed), ("1-bad", signed + "&x=1"), ("1-abc", signed)], fail_ids=["1-bad"])
    saved = pipeline.scraper
    pipeline.scraper = fake
    try:
        n_failed, out = _quiet(pipeline.run, max_players=1, source="playwright")
    finally:
        pipeline.scraper = saved
    check("returns the number of failed matches", n_failed == 1, str(n_failed))
    check("the failing match is listed", "1-bad" in out and "failed matches:" in out)
    check("the raw signed URL from the exception is redacted in the log", SECRET not in out)
    check("the already-processed match is skipped, the new one stored", db.is_processed("1-ok") and not db.is_processed("1-bad"))
    # an already-processed match counts as ok (it was skipped cleanly), not as a failure
    check("summary line reports ok/failed/attempted", "done: 2 ok, 1 failed, of 3 attempted" in out,
          next((l for l in out.splitlines() if l.startswith("done:")), ""))

    section("redact_text")
    msg = f"HTTPSConnectionPool: Max retries exceeded with url: /cs2/m.dem.zst?sig={SECRET}&exp=9"
    check("query strings are scrubbed even when the URL shape differs", SECRET not in pipeline.redact_text(msg, signed))
    check("works with no URL hint", SECRET not in pipeline.redact_text(msg))
    check("a plain message passes through", pipeline.redact_text("plain error") == "plain error")

    section("main.py commands on the test database")
    rc, out = _quiet(cli.main, ["status"])
    check("status runs", rc == 0 and "Matches   : 3" in out, out.splitlines()[1] if out else "")
    rc, out = _quiet(cli.main, ["demo", "C:/fake/local_demo.dem"])
    check("demo ingests through the seam", rc == 0 and db.is_processed("local_local_demo"), out[-120:])
    rc, out = _quiet(cli.main, ["demo", "C:/fake/local_demo.dem"])
    check("demo is idempotent (dedup before parsing)", rc == 0 and "already in DB" in out)
    rc, out = _quiet(cli.main, ["demo", "x.dem", "--match-id", "1-custom"])
    check("--match-id overrides the derived id", rc == 0 and db.is_processed("1-custom"))
    rc, out = _quiet(cli.main, ["top", "-n", "3"])
    check("top runs on an unscored DB (prints header only)", rc == 0 and "Top 3 events" in out)
    rc, out = _quiet(cli.main, ["round", "syn", "1", "--engagements-only"])
    check("round shows the play-by-play with the winner", rc == 0 and "(CT won)" in out and "engagement" in out)
    rc, out = _quiet(cli.main, ["round", "syn", "99"])
    check("round with no events exits 1", rc == 1)
    rc, out = _quiet(cli.main, ["player", "A"])
    check("player search runs (nothing scored yet)", rc == 0 and "No scored engagements" in out)
    rc, out = _quiet(cli.main, ["score"])
    check("score without a model exits 1 with a hint", rc == 1 and "Train first" in out)

    return summary()


if __name__ == "__main__":
    sys.exit(main())
