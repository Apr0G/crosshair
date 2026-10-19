"""
Run every engine test script and tally the results.

    .venv\\Scripts\\python.exe test\\run_all.py [--suite engine|all] [path\\to\\demo.dem]

Discovers `test/engine/test_*.py`. Each test is a plain script (no pytest) run in its
own process with the project venv's interpreter (`.venv\\Scripts\\python.exe`, or
`.venv/bin/python`), falling back to the interpreter running this file. Exit codes:
0 pass, 1 fail, 2 skipped because a prerequisite is missing (demo, model, golden).
A run that dies with an uncaught exception (traceback on stderr) or any other exit
code is reported as ERROR, separately from a test that ran and failed a check.

A demo path, if given, is resolved against the caller's directory and appended to
every test's argv; tests that do not take one ignore extra arguments. When the caller
has no `CROSSHAIR_ENGINE_DB`, each test gets a fresh temp DB through it (LOW-42), so a
test that forgets to set its own can never open `data/engine/engine.db`.

`--suite engine` and `--suite all` are the same today (Amendment 1: the engine suite is
the only suite); the flag exists so later suites can be added without changing callers.

Not run here: test_scraper.py (opens a real Chrome window).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent
SUITES = {"engine": [HERE / "engine"], "all": [HERE / "engine"]}

EXCLUDE = {"test_scraper.py"}
TIMEOUT_S = 1800

sys.path.insert(0, str(HERE / "engine"))
from _harness import PASS_MARK, FAIL_MARK   # noqa: E402  (also switches the console to UTF-8)


def interpreter() -> str:
    for cand in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if cand.is_file():
            return str(cand)
    return sys.executable


def discover(suite: str) -> list[Path]:
    found = []
    for d in SUITES[suite]:
        found += [p for p in d.glob("test_*.py") if p.name not in EXCLUDE]
    return sorted(found)


def classify(returncode: int, stderr: str) -> str:
    if returncode == 0:
        return "PASS"
    if returncode == 2:
        return "SKIP"
    if returncode == 1 and "Traceback (most recent call last)" not in stderr:
        return "FAIL"
    return "ERROR"


def run_one(python: str, script: Path, demo: str | None) -> tuple[str, str, float, list[str], str]:
    cmd = [python, str(script)] + ([demo] if demo else [])
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="ch_run_all_", ignore_cleanup_errors=True) as tmp:
        env = dict(os.environ)
        env.setdefault("CROSSHAIR_ENGINE_DB", str(Path(tmp) / "engine.db"))    # LOW-42
        try:
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env,
                                  encoding="utf-8", errors="replace", timeout=TIMEOUT_S)
            out, err, rc = (proc.stdout or ""), (proc.stderr or ""), proc.returncode
        except subprocess.TimeoutExpired:
            out, err, rc = "", f"timed out after {TIMEOUT_S}s", -1
    dt = time.time() - t0
    lines  = out.strip().splitlines()
    status = classify(rc, err)
    if status == "ERROR":
        tally = (err.strip().splitlines() or [f"exit code {rc}"])[-1]
    elif status == "SKIP":       # the reason: a harness '(skipped: ...)' line, else the last non-banner line
        tally = (next((l.strip() for l in lines if "(skipped:" in l), "")
                 or next((l.strip() for l in reversed(lines) if l.strip() and not l.strip().startswith("─")), ""))
    else:
        tally = next((l.strip() for l in reversed(lines) if "passed" in l), "")
    return status, tally, dt, lines, err


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the plain-script test suite.")
    ap.add_argument("--suite", choices=sorted(SUITES), default="all")
    ap.add_argument("demo", nargs="?", default=None, help="optional demo path forwarded to every test")
    args = ap.parse_args()

    python  = interpreter()
    scripts = discover(args.suite)
    demo    = str(Path(args.demo).resolve()) if args.demo else None      # children run with cwd=ROOT (LOW-42)
    print(f"  interpreter: {python}")
    print(f"  suite: {args.suite}  ({len(scripts)} test scripts)\n")

    results = []
    for script in scripts:
        status, tally, dt, lines, err = run_one(python, script, demo)
        results.append((script.name, status, dt))
        mark = PASS_MARK if status == "PASS" else FAIL_MARK if status in ("FAIL", "ERROR") else " "
        print(f"  {mark} {status:<5} {script.name:<30} {tally}  ({dt:.1f}s)", flush=True)
        if status in ("FAIL", "ERROR"):
            if lines:
                print("        " + "\n        ".join(lines[-12:]))
            if err.strip():
                print("        " + "\n        ".join(err.strip().splitlines()[-6:]))

    n = {s: sum(1 for _, st, _ in results if st == s) for s in ("PASS", "FAIL", "ERROR", "SKIP")}
    total_s = sum(dt for _, _, dt in results)
    print(f"\n  {n['PASS']} passed, {n['FAIL']} failed, {n['ERROR']} errored, "
          f"{n['SKIP']} skipped (missing prerequisite)  [{total_s:.1f}s]")
    return 1 if (n["FAIL"] or n["ERROR"]) else 0


if __name__ == "__main__":
    sys.exit(main())
