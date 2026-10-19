"""
Trivial always-passing test so run_all.py has more than one script to discover, plus
a few cheap checks that the package and the harness are wired up.

    .venv\\Scripts\\python.exe test\\engine\\test_run_all_smoke.py
"""
import sys

from _harness import ROOT, check, section, summary

import engine
from engine import cli


def main() -> int:
    section("smoke")
    check("engine package importable from the repo root", engine.__file__.startswith(str(ROOT)))
    check("engine.cli.main returns 0", cli.main([]) == 0)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
