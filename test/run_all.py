"""
Run every test script and tally the results.

    .venv/bin/python test/run_all.py [path/to/demo.dem]

Each test is a plain script (no pytest) run in its own process with this same
interpreter. Exit codes: 0 pass, 1 fail, 2 skipped because a prerequisite is
missing (demo, trained model, database). A demo path is forwarded to the tests
that can use one.

Not run here: test_scraper.py (opens a real Chrome window) and test_local_demo.py
(an inspection script, not a test).
"""
import subprocess
import sys
import time
from pathlib import Path

from _harness import PASS, FAIL   # importing also switches this console to UTF-8 (Windows)

HERE = Path(__file__).parent
TAKES_DEMO = {"test_build_state.py", "test_attribution_windows.py"}
EXCLUDE    = {"test_scraper.py", "test_local_demo.py", "run_all.py"}


def main() -> int:
    demo = sys.argv[1] if len(sys.argv) > 1 else None
    scripts = sorted(p for p in HERE.glob("test_*.py") if p.name not in EXCLUDE)
    results = []
    for script in scripts:
        cmd = [sys.executable, str(script)]
        if demo and script.name in TAKES_DEMO:
            cmd.append(demo)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=HERE.parent, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        dt = time.time() - t0
        out = (proc.stdout or "").strip().splitlines()
        tally = next((l.strip() for l in reversed(out) if "passed" in l), "")
        if proc.returncode == 0:
            status = "PASS"
        elif proc.returncode == 2:
            status = "SKIP"
            tally = next((l.strip() for l in out if l.strip()), "")
        elif proc.returncode == 1:
            status = "FAIL"
        else:
            status = "ERROR"
            tally = (proc.stderr or "").strip().splitlines()[-1:] or [""]
            tally = tally[0]
        results.append((script.name, status, tally, dt))
        mark = PASS if status == "PASS" else FAIL if status in ("FAIL", "ERROR") else " "
        print(f"  {mark} {status:<5} {script.name:<34} {tally}  ({dt:.1f}s)", flush=True)
        if status in ("FAIL", "ERROR"):
            print("        " + "\n        ".join(out[-12:]))
            if proc.stderr:
                print("        " + "\n        ".join(proc.stderr.strip().splitlines()[-6:]))

    n = {s: sum(1 for _, st, _, _ in results if st == s) for s in ("PASS", "FAIL", "ERROR", "SKIP")}
    print(f"\n  {n['PASS']} passed, {n['FAIL']} failed, {n['ERROR']} errored, {n['SKIP']} skipped (missing demo/model/db)")
    return 1 if (n["FAIL"] or n["ERROR"]) else 0


if __name__ == "__main__":
    sys.exit(main())
