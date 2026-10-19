"""
P11.3.3 / P11.3.1 / P12.1 / P0.3.2 guards of the CLI.

    .venv\\Scripts\\python.exe test\\engine\\test_cli_guards.py

- a 64-tick demo is refused (exit 2, clear message), also an unknown tick rate;
- a demo registered as FACEIT analysed with `--tier pro` prints a "tier token
  overridden" warning and carries it in report.json; the same tier does not;
- a map version with < 100 registered matches warns; no DB -> no warning, a note;
- models: missing directory / not adopted refused, `--allow-unadopted` accepted,
  docs/ENGINE_MODELS.md parsing;
- free space: batch parsing refused below 20 GB, warning below 100 GB.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _harness import ROOT, check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="crosshair_guards_"))
DB_FILE = TMP / "engine.db"
os.environ["CROSSHAIR_ENGINE_DB"] = str(DB_FILE)
ENV = {**os.environ, "CROSSHAIR_STUB_PREDICTOR": "1"}
LOGS = ["--log-dir", str(TMP / "logs")]

from _synthetic import make_raw_tables      # noqa: E402
from engine import analyze, cli, db, paths  # noqa: E402


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "engine.cli", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=ENV)


section("tick rate")
p = run("analyze", "syn.dem", "--raw-tables-story", "base", "--raw-tables-tick-rate", "64",
        "--out", str(TMP / "o64"), *LOGS)
check("64-tick demo refused with exit 2", p.returncode == 2, p.stderr[-300:])
check("message names 64 and 128-tick", "64" in p.stderr and "128-tick" in p.stderr)
check("no traceback", "Traceback" not in p.stderr)
check("nothing written for the refused demo", not (TMP / "o64" / "report.json").exists())
for meta, ok in (({"tick_rate": 64}, False), ({"tick_rate": None}, False), ({"tick_rate": 128}, True),
                 ({"tick_rate": 128, "tick_rate_raw": 127.8}, True), ({"tick_rate": 128, "tick_rate_raw": 64.0}, False)):
    try:
        analyze.check_tick_rate(meta)
        got = True
    except analyze.Refused:
        got = False
    check(f"check_tick_rate({meta}) -> {'ok' if ok else 'refused'}", got == ok)

section("tier override")
raw = make_raw_tables("base")
check("no DB yet: map support check skipped (None)", analyze.map_support("de_test@flat") is None)
db.register_match({"match_key": "fc_1", "demo_sha256": raw.meta["demo_sha256"], "source": "faceit_api",
                   "tier": "faceit_10", "map": "de_test", "map_version": "de_test@flat", "status": "parsed"})
check("source_tier from the matches row", analyze.source_tier(raw.meta) == "faceit_10")
check("source_tier from a FACEIT server name", analyze.source_tier({"server": "FACEIT.com register"}) == "faceit_10")
check("source_tier unknown -> None", analyze.source_tier({"server": "Valve"}) is None)
p = run("analyze", "syn.dem", "--raw-tables-story", "base", "--tier", "pro", "--out", str(TMP / "pro"), *LOGS)
check("FACEIT demo with --tier pro runs (exit 0)", p.returncode == 0, p.stderr[-300:])
check("prints the tier-override warning", "tier token overridden" in p.stderr and "[WARNING" in p.stderr)
rep = json.loads((TMP / "pro" / "report.json").read_text(encoding="utf-8"))
check("report.json carries tier_overridden", any(w["code"] == "tier_overridden" for w in rep["warnings"]))
check("meta.source_tier is faceit_10", rep["meta"]["source_tier"] == "faceit_10")
p = run("analyze", "syn.dem", "--raw-tables-story", "base", "--tier", "faceit_10", "--out", str(TMP / "fc"), *LOGS)
check("same tier: no override warning", p.returncode == 0 and "tier token overridden" not in p.stderr)

section("map version support")
check("map_support counts registered matches", analyze.map_support("de_test@flat") == 1)
w = analyze.guard_warnings({"server": "x"}, "pro", "de_test@flat")
check("< 100 matches -> map_version_low_support warning", [x["code"] for x in w] == ["map_version_low_support"])
check("report.json carries map_version_low_support",
      any(x["code"] == "map_version_low_support" for x in rep["warnings"]))
real = db.db_path
db.db_path = lambda: TMP / "absent.db"
w = analyze.guard_warnings({"server": "x"}, "pro", "de_test@flat")
db.db_path = real
check("no DB -> no warning (check skipped)", w == [])

section("model adoption (P12.1)")
doc = TMP / "ENGINE_MODELS.md"
HEADER = "| date | model | kind | log loss | Brier | ECE | baseline log loss | hazard AUC | decision rule satisfied | status |"
doc.write_text("# Models\n\nProse may mention wp_v5 without adopting it.\n\n" + HEADER + "\n|---|---|---|---|---|---|---|---|---|---|\n"
               "| 2026-11-01 | wp_v3 | wp | 0.4600 | 0.1500 | 0.0150 | 0.4900 | – | P6.7.2: beats LightGBM | adopted |\n"
               "| 2026-11-02 | wp_v4 | wp | 0.4700 | 0.1500 | 0.0300 | 0.4900 | – | ECE too high | rejected |\n",
               encoding="utf-8")
check("adopted_models reads adopted rows, skips rejected and prose", analyze.adopted_models(doc) == {"wp_v3"})
check("missing ENGINE_MODELS.md -> nothing adopted", analyze.adopted_models(TMP / "none.md") == set())
for name in ("wp_v3", "wp_v4"):
    (TMP / name).mkdir()


def refused(fn) -> bool:
    try:
        fn()
        return False
    except analyze.ModelUnavailable:
        return True


check("missing model dir refused", refused(lambda: analyze.check_model(TMP / "wp_v9", doc=doc)))
check("adopted model accepted", not refused(lambda: analyze.check_model(TMP / "wp_v3", doc=doc)))
check("unadopted model refused", refused(lambda: analyze.check_model(TMP / "wp_v4", doc=doc)))
check("--allow-unadopted accepts it", not refused(lambda: analyze.check_model(TMP / "wp_v4", True, doc=doc)))

section("free space (P0.3.2)")
demo_dir = TMP / "demos"
demo_dir.mkdir()
real_free = paths.free_space_gb
for gb, want_rc, want in ((10.0, 2, "refused"), (50.0, 0, "low disk space")):
    paths.free_space_gb = lambda path=None, _gb=gb: _gb
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        rc = cli.main(["parse", str(demo_dir), "--out", str(TMP / "parsed"), *LOGS])
    check(f"{gb:.0f} GB free: parse <dir> exits {want_rc} and says '{want}'", rc == want_rc and want in err.getvalue(),
          err.getvalue()[-200:])
paths.free_space_gb = lambda path=None: 10.0
err = io.StringIO()
with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
    rc = cli.main(["status", *LOGS])
check("10 GB free: a non-batch command only warns", rc == 0 and "low disk space" in err.getvalue())
paths.free_space_gb = real_free

sys.exit(summary())
