"""
Map version detection (P2.4.2): `<map>@<version>` from the demo header's game build,
using the known-rework table in `engine/maps/versions.yaml`. Unmatched demos get
`<map>@unknown` (the table's `default`), logged once per map + build.

    from engine.maps import version
    version.detect(demo_header, "de_mirage")     # "de_mirage@unknown"

The demo header carries no map-file hash (awpy 3.1.0 exposes map_name, build_num,
server/client names, playback timing), so the game build is the only key.
"""
import sys
from pathlib import Path

import yaml

VERSIONS_PATH = Path(__file__).resolve().parent / "versions.yaml"

_logged: set[tuple[str, object]] = set()


def load_table(path: Path | None = None) -> dict:
    with open(path or VERSIONS_PATH, encoding="utf-8") as fh:
        table = yaml.safe_load(fh) or {}
    table.setdefault("default", "unknown")
    table["maps"] = table.get("maps") or {}
    return table


def _build(header: dict) -> int | None:
    try:
        return int(header.get("build_num"))
    except (TypeError, ValueError):
        return None


def _matches(entry: dict, build: int | None) -> bool:
    lo, hi = entry.get("min_build"), entry.get("max_build")
    if build is None:
        return False
    return (lo is None or build >= int(lo)) and (hi is None or build <= int(hi))


def detect(header: dict, map_name: str, *, table: dict | None = None, logger=None) -> str:
    """`<map>@<version>` for the first table entry whose build range contains the
    header's build_num; otherwise `<map>@<default>` and one log line per (map, build)."""
    table = table if table is not None else load_table()
    build = _build(header or {})
    for entry in table["maps"].get(map_name) or []:
        if _matches(entry, build):
            return f"{map_name}@{entry['version']}"
    key = (map_name, build)
    if key not in _logged:
        _logged.add(key)
        msg = f"map version unknown for {map_name} build {build}; using {map_name}@{table['default']}"
        if logger is not None:
            logger.warning(msg, map=map_name, build=build)
        else:
            print(f"[map-version] {msg}", file=sys.stderr)
    return f"{map_name}@{table['default']}"
