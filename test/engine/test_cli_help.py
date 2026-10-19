"""
P10.1.6: `--help` exits 0 for the top level and for every subcommand; every command
has a handler and a section in docs/ENGINE_CLI.md; the CLI's tier lists match the
modules that own them.

    .venv\\Scripts\\python.exe test\\engine\\test_cli_help.py
"""
import contextlib
import io
import subprocess
import sys

from _harness import ROOT, check, section, summary

from engine import analyze, cli, ingest
from engine.state import builder

section("top level --help (subprocess)")
proc = subprocess.run([sys.executable, "-m", "engine.cli", "--help"], cwd=ROOT, capture_output=True, text=True,
                      encoding="utf-8", errors="replace")
check("python -m engine.cli --help exits 0", proc.returncode == 0, proc.stderr[-200:])
check("lists every command", all(c in proc.stdout for c in cli.COMMANDS))

section("every subcommand --help (in process)")
for cmd in cli.COMMANDS:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main([cmd, "--help"])
    check(f"{cmd} --help exits 0 and prints usage", rc == 0 and "usage:" in out.getvalue(), err.getvalue()[-200:])

section("command table")
parser = cli.build_parser()
check("every command is a subparser", sorted(parser.commands) == sorted(cli.COMMANDS))
check("every command has a handler", sorted(cli.HANDLERS) == sorted(cli.COMMANDS))
doc = (ROOT / "docs" / "ENGINE_CLI.md").read_text(encoding="utf-8")
check("docs/ENGINE_CLI.md has a section per command", all(f"### `{c}`" in doc for c in cli.COMMANDS),
      ", ".join(c for c in cli.COMMANDS if f"### `{c}`" not in doc))
check("ANALYZE_TIERS == builder.TIERS == analyze.TIERS", cli.ANALYZE_TIERS == builder.TIERS == analyze.TIERS)
check("INGEST_TIERS == ingest.TIER_DIRS", set(cli.INGEST_TIERS) == set(ingest.TIER_DIRS))

section("no command -> overview")
out = io.StringIO()
with contextlib.redirect_stdout(out):
    rc = cli.main([])
check("no command prints the overview and exits 0", rc == 0 and "usage:" in out.getvalue())

sys.exit(summary())
