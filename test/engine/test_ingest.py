"""
The seam (Amendment 1): ingest.register_demo / process_one / run.

    .venv\\Scripts\\python.exe test\\engine\\test_ingest.py

No awpy, no network: the header reader is a fake and the source is a FakeSource that
"downloads" by writing bytes. Exercised: the copy under demos/<tier>/, the matches +
processed rows, idempotency (same id, same sha under another id, existing file),
unsafe ids, both-id-forms dedup before download, failure counting, the summary line,
and that a signed URL never reaches stdout or the database.
"""
import contextlib
import io
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="ch_ingest_test_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")

from engine import db, ingest, paths  # noqa: E402

SECRET = "SUPERSECRETSIGNATURE123"
SIGNED = f"https://demos.faceit-cdn.net/cs2/m.dem.zst?sig={SECRET}"

# point every tier dir at the temp dir (ingest reads them at call time)
paths.DEMOS_PRO_DIR    = TMP / "demos" / "pro"
paths.DEMOS_FACEIT_DIR = TMP / "demos" / "faceit"
paths.DEMOS_GOLDEN_DIR = TMP / "demos" / "golden"


def fake_header(path: Path) -> dict:
    return {"map": "de_mirage", "tick_rate": 128, "game_version": "14090", "n_rounds": 24}


def demo_bytes(tag: str) -> bytes:
    return b"PBDEMS2\x00" + tag.encode() + b"\x00" * 2000


class FakeSource:
    """A source: yields demos, 'downloads' them, and can be told to fail."""
    SOURCE_NAME = "fake"

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
        dest.write_bytes(demo_bytes(dest.stem))
        return dest


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **kw)
    return rc, buf.getvalue()


def _db_dump() -> str:
    con = sqlite3.connect(db.db_path())
    out = " ".join(str(r) for t in ("matches", "processed") for r in con.execute(f"SELECT * FROM {t}"))
    con.close()
    return out


def test_register_demo() -> None:
    section("register_demo")
    src = TMP / "in" / "local.dem"
    src.parent.mkdir(parents=True)
    src.write_bytes(demo_bytes("local"))
    key, out = _quiet(ingest.register_demo, "1-local", src, SIGNED, tier="faceit_10",
                      source="faceit_api", header_reader=fake_header)
    dest = paths.DEMOS_FACEIT_DIR / "1-local.dem"
    check("returns the match_key", key == "1-local")
    check("copied under demos/<tier>/<match_id>.dem", dest.is_file() and dest.read_bytes() == src.read_bytes())
    check("source file left in place", src.is_file())
    row = db.get_match("1-local")
    check("matches row carries header + file meta",
          row and row["map"] == "de_mirage" and row["tick_rate"] == 128 and row["game_version"] == "14090"
          and row["tier"] == "faceit_10" and row["source"] == "faceit_api" and row["status"] == "downloaded"
          and row["file_bytes"] == src.stat().st_size and row["demo_sha256"] == ingest.sha256_file(src), str(row))
    check("unknown header fields are NULL, not 0", row["date"] is None and row["event_tier"] is None)
    check("processed row written", db.is_processed("1-local"))
    check("signed URL not in the db", SECRET not in _db_dump())

    key2, _ = _quiet(ingest.register_demo, "1-local", src, SIGNED, tier="faceit_10",
                     source="faceit_api", header_reader=fake_header)
    check("idempotent: same id again", key2 == "1-local" and db.stats()["matches"] == 1)
    key3, out = _quiet(ingest.register_demo, "hltv-copy", src, tier="pro", source="hltv",
                       header_reader=fake_header)
    check("same sha under another id returns the earlier key", key3 == "1-local", key3)
    check("no second copy written", not (paths.DEMOS_PRO_DIR / "hltv-copy.dem").exists())
    check("the duplicate id is marked processed", db.is_processed("hltv-copy"))

    other = TMP / "in" / "other.dem"
    other.write_bytes(demo_bytes("other"))
    (paths.DEMOS_PRO_DIR).mkdir(parents=True, exist_ok=True)
    (paths.DEMOS_PRO_DIR / "1-clash.dem").write_bytes(b"different content" * 100)
    try:
        _quiet(ingest.register_demo, "1-clash", other, tier="pro", source="x", header_reader=fake_header)
        check("an existing different file is never overwritten", False, "no exception")
    except FileExistsError:
        check("an existing different file is never overwritten",
              (paths.DEMOS_PRO_DIR / "1-clash.dem").read_bytes() == b"different content" * 100)

    same = TMP / "in" / "same.dem"
    same.write_bytes(demo_bytes("same"))
    (paths.DEMOS_PRO_DIR / "1-same.dem").write_bytes(same.read_bytes())
    key, _ = _quiet(ingest.register_demo, "1-same", same, tier="pro", source="x", header_reader=fake_header)
    check("an existing identical file is reused", key == "1-same" and db.get_match("1-same") is not None)

    def boom(path):
        raise RuntimeError("header unreadable")
    try:
        _quiet(ingest.register_demo, "1-boom", other, tier="pro", source="x", header_reader=boom)
        check("header failure raises", False)
    except RuntimeError:
        check("header failure raises; no row, no orphan copy",
              db.get_match("1-boom") is None and not (paths.DEMOS_PRO_DIR / "1-boom.dem").exists())

    for bad in ("../escape", "a b", "x;rm", "", "..", "a/b", "C:\\x"):
        try:
            ingest.register_demo(bad, src, tier="pro", source="x", header_reader=fake_header)
            check(f"unsafe match_id {bad!r} is refused", False, "no exception")
        except ValueError:
            check(f"unsafe match_id {bad!r} is refused", True)
    try:
        ingest.register_demo("1-tier", src, tier="nonsense", source="x", header_reader=fake_header)
        check("unknown tier refused", False)
    except ValueError:
        check("unknown tier refused", True)


def test_process_one() -> None:
    section("process_one: download -> register, redacted URL, dedup on both id forms")
    src = FakeSource([])
    key, out = _quiet(ingest.process_one, src, "1-abc", SIGNED, "faceit_10", header_reader=fake_header)
    check("downloads then registers", key == "1-abc" and src.calls == [SIGNED])
    con = sqlite3.connect(db.db_path())
    stored = con.execute("SELECT demo_url FROM processed WHERE match_id='1-abc'").fetchone()[0]
    source = con.execute("SELECT source FROM matches WHERE match_key='1-abc'").fetchone()[0]
    con.close()
    check("stored demo_url has its signature stripped",
          stored == "https://demos.faceit-cdn.net/cs2/m.dem.zst?<redacted>", stored)
    check("source name recorded", source == "fake", source)
    check("the signature never reaches stdout", SECRET not in out)
    k2, out2 = _quiet(ingest.process_one, src, "1-abc", SIGNED, "faceit_10", header_reader=fake_header)
    check("a processed match is skipped without downloading", k2 is None and len(src.calls) == 1 and "skipping" in out2)
    k3, _ = _quiet(ingest.process_one, src, "1-abc_m1", SIGNED, "faceit_10", header_reader=fake_header)
    check("<mid>_m1 skipped when the BO1 form <mid> is stored", k3 is None and len(src.calls) == 1)
    _quiet(ingest.process_one, src, "1-ser_m1", SIGNED, "faceit_10", header_reader=fake_header)
    k4, _ = _quiet(ingest.process_one, src, "1-ser", SIGNED, "faceit_10", header_reader=fake_header)
    check("bare <mid> skipped when <mid>_m1 is stored", k4 is None and len(src.calls) == 2)
    k5, _ = _quiet(ingest.process_one, src, "1-ser_m2", SIGNED, "faceit_10", header_reader=fake_header)
    check("a stored _m1 does not block _m2", k5 == "1-ser_m2" and len(src.calls) == 3)
    for bad in ("../escape", "a b", ""):
        try:
            ingest.process_one(src, bad, SIGNED, "faceit_10", header_reader=fake_header)
            check(f"process_one refuses unsafe id {bad!r}", False)
        except ValueError:
            check(f"process_one refuses unsafe id {bad!r}", True)


def test_run() -> None:
    section("run: counts failures and redacts error output")
    fake = FakeSource([("1-ok", SIGNED), ("1-bad", SIGNED + "&x=1"), ("1-abc", SIGNED)], fail_ids=["1-bad"])
    n_failed, out = _quiet(ingest.run, fake, tier="faceit_10", header_reader=fake_header)
    check("returns the number of failed matches", n_failed == 1, str(n_failed))
    check("the failing match is listed", "1-bad" in out and "failed matches:" in out)
    check("the raw signed URL from the exception is redacted", SECRET not in out)
    check("new match stored, failed one not", db.is_processed("1-ok") and not db.is_processed("1-bad"))
    check("summary line reports ok/failed/attempted", "done: 2 ok, 1 failed, of 3 attempted" in out,
          next((ln for ln in out.splitlines() if ln.startswith("done:")), ""))
    check("signed URL never in the db", SECRET not in _db_dump())

    fake2 = FakeSource([("2-a", SIGNED), ("2-b", SIGNED), ("2-c", SIGNED)])
    _, out = _quiet(ingest.run, fake2, max_matches=2, tier="faceit_10", header_reader=fake_header)
    check("max_matches stops after N attempts", "of 2 attempted" in out and not db.is_processed("2-c"))
    _, out = _quiet(ingest.run, FakeSource([]), tier="faceit_10")
    check("empty source says so", "no new demos found." in out and "of 0 attempted" in out)
    try:
        ingest.run("nonsense")
        check("unknown source name refused", False)
    except ValueError:
        check("unknown source name refused", True)
    check("source table names the moved modules",
          ingest.SOURCES["api"][0] == "faceit_api" and ingest.SOURCES["playwright"][0] == "scraper")
    check("named source resolves to the module", ingest._load_source("api").__name__ == "engine.sources.faceit_api")


def test_min_level() -> None:
    section("faceit_api.iter_unprocessed_demos(min_level=...) (P1.2.1)")
    from types import SimpleNamespace
    from engine.sources import faceit_api

    def roster(levels):
        return {"teams": {"faction1": {"roster": [{"game_skill_level": v} for v in levels[:5]]},
                          "faction2": {"roster": [{"game_skill_level": v} for v in levels[5:]]}}}
    payloads = {
        "3-hi":      {**roster([10] * 10), "demo_url": ["https://x/hi.dem.zst"]},
        "3-lo":      {**roster([10] * 9 + [9]), "demo_url": ["https://x/lo.dem.zst"]},
        "3-nine":    {**roster([10] * 9), "demo_url": ["https://x/nine.dem.zst"]},
        "3-missing": {**roster([10] * 9 + [None]), "demo_url": ["https://x/m.dem.zst"]},
        "3-bo3":     {**roster([10] * 10), "demo_url": ["https://x/a.dem.zst", "https://x/b.dem.zst"]},
    }
    saved = (faceit_api.scraper.get_leaderboard_players, faceit_api.scraper.get_player_match_ids,
             faceit_api.scraper._api_get, faceit_api.time)
    faceit_api.scraper.get_leaderboard_players = lambda **kw: [{"nickname": "p", "elo": 3000, "player_id": "x"}]
    faceit_api.scraper.get_player_match_ids = lambda nick, limit=20: list(payloads)
    faceit_api.scraper._api_get = lambda endpoint, params=None: payloads[endpoint.rsplit("/", 1)[-1]]
    faceit_api.time = SimpleNamespace(sleep=lambda s: None)
    try:
        got, _ = _quiet(lambda: list(faceit_api.iter_unprocessed_demos(min_level=10)))
        check("only all-ten-at-level matches pass; unknown/9-player rosters never qualify",
              [m for m, _ in got] == ["3-hi", "3-bo3_m1", "3-bo3_m2"], str(got))
        got, _ = _quiet(lambda: list(faceit_api.iter_unprocessed_demos()))
        check("min_level=None keeps the previous behaviour (no filter)", len(got) == 6, str([m for m, _ in got]))
        check("lowest level read from the roster", faceit_api.get_match_min_level("3-lo") == 9
              and faceit_api.get_match_min_level("3-nine") is None)
    finally:
        (faceit_api.scraper.get_leaderboard_players, faceit_api.scraper.get_player_match_ids,
         faceit_api.scraper._api_get, faceit_api.time) = saved


def main() -> int:
    test_register_demo()
    test_process_one()
    test_run()
    test_min_level()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
