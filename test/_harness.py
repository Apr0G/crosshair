"""
Shared bits for the plain-script tests. Stdlib only — test_invariants.py must keep
running on a bare interpreter.

    from _harness import check, skip, close, summary

Every test module records its checks through check()/skip() and ends with
`return summary()`, which prints the tally and gives the process exit code:
0 = all passed, 1 = something failed. Tests that cannot run at all (no demo, no
model, no DB) return 2 themselves so a runner can tell "skipped" from "broken".
"""
import sys

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m–\033[0m"

_results: list[tuple[bool, str]] = []

# Windows consoles default to a legacy code page that cannot print ✓/✗/─.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def check(desc: str, ok: bool, detail: str = "") -> bool:
    ok = bool(ok)
    _results.append((ok, desc))
    print(f"  {PASS if ok else FAIL}  {desc}" + (f"  ({detail})" if detail else ""))
    return ok


def skip(desc: str, why: str) -> None:
    print(f"  {SKIP}  {desc}  (skipped — {why})")


def close(a, b, tol: float = 1e-6) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 72 - len(title))}")


def summary() -> int:
    passed = sum(1 for ok, _ in _results if ok)
    failed = [d for ok, d in _results if not ok]
    print(f"\n{'─' * 60}\n  {passed}/{len(_results)} passed")
    if failed:
        print("  Failed:")
        for d in failed:
            print(f"    - {d}")
    print()
    return 0 if not failed else 1
