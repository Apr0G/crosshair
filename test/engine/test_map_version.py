"""
Map version detection (P2.4.2): two synthetic headers map to two versions via a
synthetic table; unmatched builds fall back to `<map>@unknown` and are logged once;
the shipped versions.yaml loads.

    .venv\\Scripts\\python.exe test\\engine\\test_map_version.py
"""
import sys

from _harness import check, section, summary

from engine.maps import version

TABLE = {"default": "unknown", "maps": {"de_mirage": [
    {"version": "original", "min_build": None, "max_build": 9999},
    {"version": "2026-rework", "min_build": 10000, "max_build": None},
]}}


class Recorder:
    def __init__(self):
        self.lines = []

    def warning(self, msg, **fields):
        self.lines.append((msg, fields))


def main() -> int:
    section("synthetic table")
    rec = Recorder()
    check("old build -> original", version.detect({"build_num": 9500}, "de_mirage", table=TABLE, logger=rec)
          == "de_mirage@original")
    check("new build -> rework", version.detect({"build_num": "10524"}, "de_mirage", table=TABLE, logger=rec)
          == "de_mirage@2026-rework")
    check("matches are not logged", rec.lines == [])
    check("unlisted map -> unknown", version.detect({"build_num": 10524}, "de_nuke", table=TABLE, logger=rec)
          == "de_nuke@unknown")
    check("missing build -> unknown", version.detect({}, "de_mirage", table=TABLE, logger=rec) == "de_mirage@unknown")
    version.detect({"build_num": 10524}, "de_nuke", table=TABLE, logger=rec)
    check("unknown logged once per (map, build)", len(rec.lines) == 2, f"{rec.lines}")

    section("shipped versions.yaml")
    t = version.load_table()
    check("loads with a default", t["default"] == "unknown" and isinstance(t["maps"], dict))
    check("real header -> <map>@unknown until a rework is recorded",
          version.detect({"build_num": 10524, "map_name": "de_inferno"}, "de_inferno", logger=Recorder())
          == "de_inferno@unknown")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
