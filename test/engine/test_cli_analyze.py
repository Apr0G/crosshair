"""
P10.1.1: `python -m engine.cli analyze` end to end through the CLI, with the test hooks
(no golden demo, no trained model yet):

    CROSSHAIR_STUB_PREDICTOR=1  -> rule stub instead of the model
    --raw-tables-story base     -> synthetic RawTables instead of parsing the demo

    .venv\\Scripts\\python.exe test\\engine\\test_cli_analyze.py

Checks: exit 0; run_id / config_hash / device / timings printed; every output exists
and report.json validates; action_scores rows under the printed run_id; a second run
gives the same report.json except run_id / created_at / timings; --no-db writes no
rows; --show-ids adds steamid fields, the default does not.
"""
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from _harness import ROOT, check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="crosshair_cli_analyze_"))
DB_FILE = TMP / "engine.db"
ENV = {**os.environ, "CROSSHAIR_ENGINE_DB": str(DB_FILE), "CROSSHAIR_STUB_PREDICTOR": "1"}
LOGS = ["--log-dir", str(TMP / "logs")]

os.environ["CROSSHAIR_ENGINE_DB"] = str(DB_FILE)
from engine import config, db        # noqa: E402
from engine.report import writer     # noqa: E402

db.init_db()      # the DB exists from the start, so both runs see the same map-support check


def analyze(out: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "engine.cli", "analyze", "synthetic_base.dem",
                           "--raw-tables-story", "base", "--out", str(TMP / out), *LOGS, *extra],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)


def db_rows(run_id: str) -> int:
    with sqlite3.connect(DB_FILE) as c:
        return c.execute("SELECT COUNT(*) FROM action_scores WHERE run_id = ?", (run_id,)).fetchone()[0]


section("first run")
p = analyze("a")
check("exit 0", p.returncode == 0, p.stderr[-400:])
out = TMP / "a"
for name in ("report.json", "report.md", "actions_scored.parquet", "manifest.json", "timings.json"):
    check(f"{name} exists", (out / name).is_file() and (out / name).stat().st_size > 0)
charts = sorted((out / "charts").glob("*.png"))
check("charts/ has round, player and team charts",
      any(c.name.startswith("round_") for c in charts) and any(c.name.startswith("player_") for c in charts)
      and (out / "charts" / "team_clock.png").is_file())
rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
check("report.json validates against the schema", not writer.validate(rep))
m = re.search(r"run_id=(\S+)", p.stderr)
run_id = m.group(1) if m else None
check("run_id printed and equal to meta.run_id", run_id == rep["meta"]["run_id"])
check("config_hash printed", f"config_hash={config.config_hash(config.load())}" in p.stderr)
check("device printed (stub)", "device=stub" in p.stderr)
check("stage timings printed", "[INFO   ] timings" in p.stderr and "attribution=" in p.stderr)
check("test hooks announced as warnings", p.stderr.count("TEST HOOK") == 2)
check("report path on stdout", "report.json" in p.stdout)
check("action_scores rows under the run_id", run_id is not None and db_rows(run_id) ==
      sum(len(r["actions"]) for r in rep["rounds"]))
man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
check("manifest names the run and the value stage", man["extra"]["run_id"] == run_id
      and man["extra"]["value_stage"] == "skipped" and man["kind"] == "analyze")
check("model version recorded as stub", rep["meta"]["model_versions"]["wp"] == "stub")

section("second run: reproducible")
p2 = analyze("b")
rep2 = json.loads((TMP / "b" / "report.json").read_text(encoding="utf-8"))
check("exit 0", p2.returncode == 0)
check("new run_id", rep2["meta"]["run_id"] != rep["meta"]["run_id"])
check("report.json identical except run_id / created_at / timings",
      writer.dumps(writer.strip_dynamic(rep)) == writer.dumps(writer.strip_dynamic(rep2)))

section("--no-db and --show-ids")
p3 = analyze("c", "--no-db", "--show-ids")
rep3 = json.loads((TMP / "c" / "report.json").read_text(encoding="utf-8"))
check("exit 0", p3.returncode == 0, p3.stderr[-300:])
check("--no-db: no rows for its run", db_rows(rep3["meta"]["run_id"]) == 0)
acts3 = [a for r in rep3["rounds"] for a in r["actions"] if a.get("actor")]
check("--show-ids: actions carry steamid", acts3 and all("steamid" in a for a in acts3))
check("--show-ids: steamids are the story's", {a["steamid"] for a in acts3} <= {"1001", "1002", "2001", "2002"})
check("meta.show_ids true", rep3["meta"]["show_ids"] is True)
acts = [a for r in rep["rounds"] for a in r["actions"]]
check("default: no steamid field anywhere in actions", all("steamid" not in a for a in acts))
check("default: meta.show_ids false", rep["meta"]["show_ids"] is False)

sys.exit(summary())
