"""
The one test harness for the engine suite (Amendment 1). Stdlib only.

    from _harness import check, skip, close, section, summary
    require("torch", "the model tests need the platform torch build")   # exit 2 if absent

Importing it also puts the repo root on sys.path (so tests can `import engine`) and
switches stdout/stderr to UTF-8 with errors="replace" (Windows consoles default to a
legacy code page that cannot print the marks below).

Every test records results through check()/skip() and ends with
`sys.exit(summary())`. Exit codes: 0 = all passed, 1 = something failed,
2 = a prerequisite is missing (nothing ran, or only skips). A test that cannot start
at all (no demo, no model) may also `sys.exit(2)` itself after printing why.

Run directly (`python test/engine/_harness.py`) it self-checks and exits 0.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PASS_MARK = "\033[32m✓\033[0m"
FAIL_MARK = "\033[31m✗\033[0m"
SKIP_MARK = "\033[33m–\033[0m"

# kept for scripts that import the legacy names
PASS, FAIL, SKIP = PASS_MARK, FAIL_MARK, SKIP_MARK

_passed: list[str]             = []
_failed: list[str]             = []
_skipped: list[tuple[str, str]] = []


def check(desc: str, ok, detail: str = "") -> bool:
    ok = bool(ok)
    (_passed if ok else _failed).append(desc)
    print(f"  {PASS_MARK if ok else FAIL_MARK}  {desc}" + (f"  ({detail})" if detail else ""))
    return ok


def skip(desc: str, why: str) -> None:
    _skipped.append((desc, why))
    print(f"  {SKIP_MARK}  {desc}  (skipped: {why})")


def close(a, b, tol: float = 1e-6) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def require(module: str, why: str):
    """Import `module` and return it; on ImportError print why and exit 2 (a missing
    prerequisite, not an error). Call it before importing anything that needs the
    module, e.g. torch or faiss when the platform build is absent (LOW-61)."""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        print(f"prerequisite missing: cannot import {module} ({type(exc).__name__}: {exc}); {why}")
        sys.exit(2)


def section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 72 - len(title))}")


def counts() -> dict[str, int]:
    return {"pass": len(_passed), "fail": len(_failed), "skip": len(_skipped)}


def summary(exit_on_only_skips: bool = True) -> int:
    """Print the tally and return the exit code: 1 if anything failed; 2 if nothing
    ran at all, or only skips ran and `exit_on_only_skips`; else 0."""
    n_checks = len(_passed) + len(_failed)
    print(f"\n{'─' * 60}\n  {len(_passed)}/{n_checks} passed, {len(_skipped)} skipped")
    if _failed:
        print("  Failed:")
        for d in _failed:
            print(f"    - {d}")
    print()
    if _failed:
        return 1
    if n_checks == 0 and not _skipped:
        return 2
    if n_checks == 0 and exit_on_only_skips:
        return 2
    return 0


def _selftest() -> int:
    ok = close(0.1 + 0.2, 0.3) and not close(None, 0.3) and not close(1.0, 1.1, 0.05)
    print("harness self-check:", "ok" if ok else "BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
