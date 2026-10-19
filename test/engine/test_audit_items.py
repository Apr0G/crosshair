"""
Regression checks for the findings of the full-codebase audits (docs/reviews/).

One function per fixed finding, named after its ID in the dated report
(`crit_01`, `high_02`, `med_17`, ...). Each rebuilds the failing case without `data/`,
runs the stage that was wrong and asserts the corrected behaviour; the docstring names
the report and the finding title. Review-loop items (P11.2.3) live in
test_review_items.py, not here.

    .venv\\Scripts\\python.exe test\\engine\\test_audit_items.py
"""
import subprocess

from _harness import ROOT, check, section, skip, summary

SOURCE_DIRS = ("engine", "test", "bench")


# ── 2026-10-19 ────────────────────────────────────────────────────────────────

def crit_01() -> None:
    """2026-10-19 CRIT-01: `.gitignore` `data/` also ignored the source package
    `engine/data/`, so the pushed tree had no dataset/splits/quarantine modules.
    Every source file under engine/, test/ and bench/ must be free of ignore rules."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True, cwd=ROOT)
    except (OSError, subprocess.CalledProcessError):
        skip("no source file is gitignored", "git not available")
        return
    files = [p for d in SOURCE_DIRS for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts]
    rel   = [str(p.relative_to(ROOT)).replace("\\", "/") for p in files]
    proc  = subprocess.run(["git", "check-ignore", "--no-index", *rel],
                           capture_output=True, text=True, cwd=ROOT)
    ignored = [l for l in proc.stdout.splitlines() if l.strip()]
    check(f"no source file under {'/'.join(SOURCE_DIRS)} is gitignored ({len(rel)} checked)",
          not ignored, ", ".join(ignored[:5]))
    pkg = sorted(p.name for p in (ROOT / "engine" / "data").glob("*.py"))
    check("engine/data package is present with its seven modules",
          pkg == ["__init__.py", "dataset.py", "inventory.py", "layout.py", "qa.py", "quarantine.py", "splits.py"],
          str(pkg))


# ── runner ────────────────────────────────────────────────────────────────────

ITEMS = [crit_01]

section("audit items (docs/reviews)")
for fn in ITEMS:
    fn()

raise SystemExit(summary())
