"""
Parse worker (P2.6.1) with fake parsers: idempotent rerun, one retry, failures logged
with redacted errors, --force reparses, --smoke writes no parquet, multiprocess run,
quarantine on the header before every parse (HIGH-01, 2026-10-19 audit).

    .venv\\Scripts\\python.exe test\\engine\\test_parse_worker.py

Works in a temp dir; never touches data/engine/parsed or a real demo.
"""
import json
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

from engine import manifest
from engine.parse import awpy3_parser as ap
from engine.parse import worker

SECRET = "SUPERSECRETSIGNATURE123"
_FLAKY_CALLS: dict[str, int] = {}


# Fake parsers are module-level so the process pool can pickle them.

def fake_parse(path: Path):
    tables = {name: ap.empty(name) for name in ap.TABLES}
    meta = {"map": "de_mirage", "tick_rate": 128, "game_version": "10524", "server": "fake",
            "demo_sha256": ap.sha256_file(path), "n_rounds": 0, "players": []}
    return tables, meta


def failing_parse(path: Path):
    raise RuntimeError(f"download broke: https://demos.faceit-cdn.net/cs2/a.dem.zst?X-Amz-Signature={SECRET}")


def invalid_parse(path: Path):
    tables, meta = fake_parse(path)
    meta["tick_rate"] = None                      # RawTables.validate() must refuse this
    return tables, meta


def flaky_parse(path: Path):
    n = _FLAKY_CALLS[str(path)] = _FLAKY_CALLS.get(str(path), 0) + 1
    if n == 1:
        raise OSError("transient")
    return fake_parse(path)


def fake_header(path: Path, with_roster: bool = False):
    good = {"map": "de_mirage", "tick_rate": 128, "tick_rate_raw": 128.0, "duration_s": 2700.0,
            "n_rounds": 24, "side_counts": {"ct": 5, "t": 5}}
    if "short" in Path(path).name:
        good["n_rounds"] = 3
    if "t64" in Path(path).name:
        good["tick_rate"], good["tick_rate_raw"] = 64, 64.0
    if "broken" in Path(path).name:
        raise ValueError("not a demo")
    return good


def make_demos(d: Path, names: list[str]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"PBDEMS2\x00" + n.encode() * 10)
    return d


def statuses(out: dict) -> dict:
    return {r["match_key"]: r["status"] for r in out["results"]}


def test_parse(tmp: Path) -> None:
    section("parse, idempotent rerun, --force")
    demos = make_demos(tmp / "demos", ["m1.dem", "m2.dem", "notes.txt"])
    out_dir = tmp / "parsed"
    first = worker.run(demos, out_dir=out_dir, procs=1, parser=fake_parse, header_fn=fake_header)
    check("both demos parsed (non-.dem ignored)", statuses(first) == {"m1": "parsed", "m2": "parsed"},
          f"{statuses(first)}")
    m1 = out_dir / "m1"
    want = {f"{t}.parquet" for t in ap.TABLES} | {"meta.parquet", "manifest.json"}
    have = {p.name for p in m1.iterdir()}
    check("one parquet per table + meta + manifest", want <= have, f"missing {want - have}")
    man = manifest.read(m1)
    check("manifest kind parse", man["kind"] == "parse")
    check("manifest records match_key", man["extra"]["match_key"] == "m1")
    check("manifest input hash", man["inputs"][0]["sha256"] == manifest.sha256_file(demos / "m1.dem"))
    check("manifest map asset hashes or a recorded reason",
          bool(man["map_assets"]) or "map_assets_error" in man["extra"])
    created = man["created_at"]

    again = worker.run(demos, out_dir=out_dir, procs=1, parser=failing_parse, header_fn=fake_header)
    check("rerun skips already-parsed matches (parser never called)",
          statuses(again) == {"m1": "skipped", "m2": "skipped"})
    check("rerun leaves no failures", not (out_dir / worker.FAILURES_NAME).exists())

    forced = worker.run(demos, out_dir=out_dir, procs=1, parser=fake_parse, force=True, header_fn=fake_header)
    check("--force reparses", statuses(forced) == {"m1": "parsed", "m2": "parsed"})
    check("--force rewrote the manifest", manifest.read(m1)["created_at"] >= created)


def test_failures(tmp: Path) -> None:
    section("retry and failure log")
    demos = make_demos(tmp / "fdemos", ["bad.dem", "flaky.dem"])
    out_dir = tmp / "fparsed"
    out = worker.run(demos / "..", out_dir=out_dir, procs=1, parser=failing_parse)   # no .dem at top
    check("empty directory: nothing to do", out["results"] == [])
    r = worker.parse_one(demos / "bad.dem", out_dir, failing_parse, header_fn=fake_header)
    check("failed after one retry", r["status"] == "failed" and r["attempts"] == 2)
    worker._append_failures(out_dir, [r])
    r2 = worker.parse_one(demos / "flaky.dem", out_dir, flaky_parse, header_fn=fake_header)
    check("a transient failure succeeds on the retry", r2["status"] == "parsed" and r2["attempts"] == 2)
    log = (out_dir / worker.FAILURES_NAME).read_text(encoding="utf-8")
    rec = json.loads(log.splitlines()[0])
    check("failure logged with exception type", rec["exc_type"] == "RuntimeError" and rec["match_key"] == "bad")
    check("failure message redacted", SECRET not in log and "<redacted>" in log, rec["error"])
    check("failed match has no manifest", not worker.is_parsed(out_dir, "bad"))
    r3 = worker.parse_one(demos / "bad.dem", tmp / "fparsed3", invalid_parse, header_fn=fake_header)
    check("schema-invalid output is a failure, nothing written",
          r3["status"] == "failed" and r3["exc_type"] == "SchemaError" and not (tmp / "fparsed3" / "bad").exists())

    out = worker.run(demos, out_dir=tmp / "fparsed2", procs=1, parser=failing_parse, header_fn=fake_header)
    lines = (tmp / "fparsed2" / worker.FAILURES_NAME).read_text(encoding="utf-8").splitlines()
    check("run() appends one line per failed match", len(lines) == 2 and out["counts"] == {"failed": 2})


def test_quarantine(tmp: Path) -> None:
    section("quarantine before the parse (P1.4.1 on the batch path)")
    demos = make_demos(tmp / "qdemos", ["ok.dem", "t64.dem", "short.dem", "broken.dem"])
    out_dir = tmp / "qparsed"
    _FLAKY_CALLS.clear()
    out = worker.run(demos, out_dir=out_dir, procs=1, parser=fake_parse, header_fn=fake_header)
    by = {r["match_key"]: r for r in out["results"]}
    check("good demo parsed", by["ok"]["status"] == "parsed")
    check("64-tick demo quarantined, never parsed", by["t64"]["status"] == "quarantined"
          and by["t64"]["reason"] == "tick_rate_not_128" and not (out_dir / "t64").exists())
    check("3-round demo quarantined too_few_rounds", by["short"]["reason"] == "too_few_rounds")
    check("unreadable header quarantined header_unreadable",
          by["broken"]["status"] == "quarantined" and by["broken"]["reason"] == "header_unreadable")
    check("quarantined is not failed (no failure log line)", not (out_dir / worker.FAILURES_NAME).exists()
          and out["counts"] == {"parsed": 1, "quarantined": 3}, f"{out['counts']}")
    check("quarantined result carries the demo path", by["t64"]["path"] == str(demos / "t64.dem"))
    check("default header reader refuses a fake file (awpy cannot read it)",
          worker.parse_one(demos / "ok.dem", tmp / "qparsed2", fake_parse)["status"] == "quarantined")
    pooled = worker.run(demos, out_dir=tmp / "qparsed3", procs=2, parser=fake_parse, header_fn=fake_header)
    check("process pool applies the same gate", pooled["counts"] == {"parsed": 1, "quarantined": 3})


def test_smoke(tmp: Path) -> None:
    section("--smoke")
    demos = make_demos(tmp / "sdemos", ["good.dem", "short.dem", "broken.dem"])
    out_dir = tmp / "sparsed"
    out = worker.run(demos, out_dir=out_dir, procs=1, smoke=True, header_fn=fake_header,
                     parser=failing_parse)
    by = {r["match_key"]: r for r in out["results"]}
    check("good demo passes", by["good"]["status"] == "ok" and by["good"]["reason"] is None)
    check("short demo quarantined too_few_rounds", by["short"]["reason"] == "too_few_rounds")
    check("unreadable header quarantined", by["broken"]["reason"] == "header_unreadable")
    check("smoke writes no parquet", not out_dir.exists() or not list(out_dir.rglob("*.parquet")))


def test_pool(tmp: Path) -> None:
    section("process pool")
    demos = make_demos(tmp / "pdemos", ["p1.dem", "p2.dem", "p3.dem"])
    out = worker.run(demos, out_dir=tmp / "pparsed", procs=2, parser=fake_parse, header_fn=fake_header)
    check("2 processes parse 3 demos", out["counts"] == {"parsed": 3}, f"{out['counts']}")
    check("every match has a manifest", all(worker.is_parsed(tmp / "pparsed", k) for k in ("p1", "p2", "p3")))


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_parse(tmp)
        test_failures(tmp)
        test_smoke(tmp)
        test_quarantine(tmp)
        test_pool(tmp)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
