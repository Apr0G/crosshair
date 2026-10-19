"""
Callout plot over the nav mesh (P2.3.3), `engine/maps/plot.py`.

    .venv\\Scripts\\python.exe test\\engine\\test_map_plot.py

The CLI must exit 2 with one line (no traceback) when the map assets are missing
(LOW-19). With the pinned awpy-data release and the de_mirage callouts present, one
map is drawn to a temp png; without them that part is skipped.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _harness import ROOT, check, section, skip, summary

from engine.maps import assets
from engine.maps import callouts
from engine.maps import plot as plot_mod


def _run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "engine.maps.plot", *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=300,
                          env={**os.environ, **(env or {})})


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        section("CLI: missing assets exit 2 with one line")
        r = _run(["de_notamap", str(tmp / "x.png")])
        lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
        check("unknown map: exit 2", r.returncode == 2, f"exit {r.returncode}")
        check("unknown map: one line, no traceback", len(lines) == 1 and "Traceback" not in r.stderr,
              f"{lines[:3]}")
        check("unknown map: no png written", not (tmp / "x.png").exists())
        empty = tmp / "awpy_empty"
        empty.mkdir()
        r = _run(["de_mirage", str(tmp / "y.png")], env={"AWPY_DATA_DIR": str(empty)})
        check("release not cached: exit 2, no traceback", r.returncode == 2 and "Traceback" not in r.stderr,
              f"exit {r.returncode}: {r.stderr.strip()[-200:]}")
        r = _run([])
        check("no map argument: usage, exit 2", r.returncode == 2 and "usage" in r.stdout)

        section("plot one map")
        release = assets.pinned_release()
        if not assets.release_dir(release).is_dir() or "de_mirage" not in assets.cached_maps(release):
            skip("de_mirage drawn to a png", f"awpy-data release {release} with de_mirage not cached")
        elif not callouts.path_for("de_mirage").is_file():
            skip("de_mirage drawn to a png", "no de_mirage callouts file")
        else:
            out = plot_mod.plot("de_mirage", tmp / "sub" / "mirage.png")
            check("returns the requested path", out == tmp / "sub" / "mirage.png")
            check("png written (non-empty, PNG magic)",
                  out.is_file() and out.stat().st_size > 1000 and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
