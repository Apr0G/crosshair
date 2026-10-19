"""
Disk layout and free-space checks (P0.3).

    .venv\\Scripts\\python.exe test\\engine\\test_paths.py

ensure() is exercised in a temp dir; check_space() with a monkeypatched
shutil.disk_usage. Nothing under data/ is created or read.
"""
import shutil
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

from _harness import check, section, summary

from engine import paths

Usage = namedtuple("Usage", "total used free")
P031  = ("demos/pro", "demos/faceit", "parsed", "states", "actions", "datasets", "models",
         "eval", "reports", "bench", "logs", "cache/maps")


def test_layout() -> None:
    section("layout")
    check("everything under data/engine", all(paths.ENGINE_DIR in d.parents for d in paths.ALL_DIRS))
    rels = {d.relative_to(paths.ENGINE_DIR).as_posix() for d in paths.ALL_DIRS}
    missing = [r for r in P031 if r not in rels]
    check("one constant per P0.3.1 directory", not missing, f"missing {missing}")
    check("engine db under data/engine", paths.DB_PATH.parent == paths.ENGINE_DIR)
    with tempfile.TemporaryDirectory() as d:
        made = paths.ensure(Path(d))
        check("ensure() creates every dir", all(p.is_dir() for p in made) and len(made) == len(paths.ALL_DIRS))
        check("ensure() is idempotent", paths.ensure(Path(d)) == made)


def test_space() -> None:
    section("free space")
    real = shutil.disk_usage
    try:
        for free_gb, want in ((500, paths.SpaceStatus.OK), (100, paths.SpaceStatus.OK),
                              (99.9, paths.SpaceStatus.WARN), (20, paths.SpaceStatus.WARN),
                              (19.9, paths.SpaceStatus.REFUSE), (0, paths.SpaceStatus.REFUSE)):
            shutil.disk_usage = lambda p, f=free_gb: Usage(10**13, 0, int(f * 1e9))
            got = paths.check_space()
            check(f"{free_gb} GB free -> {want.name}", got is want, got.name)
        shutil.disk_usage = lambda p: Usage(10**13, 0, int(42.5e9))
        check("free_space_gb reports GB", abs(paths.free_space_gb() - 42.5) < 1e-6)
        check("custom thresholds honoured",
              paths.check_space(min_warn_gb=50, min_refuse_gb=45) is paths.SpaceStatus.REFUSE)
        try:
            paths.check_space(min_warn_gb=10, min_refuse_gb=20)
            check("refuse > warn rejected", False)
        except ValueError:
            check("refuse > warn rejected", True)
    finally:
        shutil.disk_usage = real
    check("real disk_usage works on a missing path (walks up)",
          paths.free_space_gb(paths.ENGINE_DIR / "no" / "such" / "dir") > 0)


def main() -> int:
    test_layout()
    test_space()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
