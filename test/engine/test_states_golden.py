"""
Golden states (P3.8.2): build the golden demo's states in exact-LOS mode and compare
row count, per-kind counts and per-column checksums with `golden/states.json`.

    .venv\\Scripts\\python.exe test\\engine\\test_states_golden.py [--write]

Exit 2 while no golden demo is pinned (GOLDEN.json demo_sha256 null), the demo file is
missing, the parser has no `parse(path) -> RawTables` yet, map assets are missing, or
`states.json` has not been written (`--write` creates it from the current builder,
after a human has checked the output).
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from _harness import check, section, summary

from engine import config
from engine.state import builder

HERE        = Path(__file__).resolve().parent
GOLDEN_JSON = HERE / "golden" / "GOLDEN.json"
STATES_JSON = HERE / "golden" / "states.json"


def checksums(snaps) -> dict:
    """Row count, per-kind counts and a sha256 per column of the states frame (floats
    rounded to 4 decimals so the checksum survives harmless float noise)."""
    df = builder.to_frame(snaps)
    sums = {}
    for c in df.columns:
        vals = df[c].to_list()
        if df.schema[c].is_float():
            vals = [None if v is None else round(v, 4) for v in vals]
        blob = json.dumps(vals, default=str, sort_keys=True, separators=(",", ":")).encode()
        sums[c] = hashlib.sha256(blob).hexdigest()
    kinds = Counter(k for s in snaps for k in s.meta.kinds)
    return {"rows": len(snaps), "kinds": dict(sorted(kinds.items())), "columns": sums}


def main() -> int:
    section("golden states")
    meta = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    if not meta.get("demo_sha256"):
        print(f"  no golden demo pinned yet (demo_sha256 is null in {GOLDEN_JSON.name}); "
              f"states golden needs one (P0.7.1)")
        return 2
    sys.path.insert(0, str(HERE))
    from test_golden_present import locate
    demo = locate(meta)
    if demo is None:
        print("  golden demo file missing; see test_golden_present.py")
        return 2
    try:
        from engine.parse.awpy3_parser import parse
    except ImportError:
        print("  engine.parse.awpy3_parser.parse not available yet (P2.5)")
        return 2
    try:
        from engine.maps import assets as map_assets
        from engine.maps import callouts as map_callouts
        from engine.maps import version as map_version
        from engine.state.geometry import MapGeometry
        raw = parse(demo)
        a   = map_assets.ensure(raw.meta["map"])
        geo = MapGeometry.from_assets(a, map_callouts.load(raw.meta["map"]),
                                      map_version.detect(raw.meta, raw.meta["map"]))
    except Exception as e:  # missing assets / callouts are prerequisites, not failures
        print(f"  prerequisite missing: {type(e).__name__}: {e}")
        return 2
    snaps = builder.build(raw, geo, config.load(), match_key="golden", tier="pro", los_cache=False, debug=True)
    got = checksums(snaps)
    if "--write" in sys.argv:
        STATES_JSON.write_text(json.dumps(got, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  wrote {STATES_JSON}")
        return 0
    if not STATES_JSON.is_file():
        print(f"  {STATES_JSON.name} not written yet; run with --write after checking the output")
        return 2
    want = json.loads(STATES_JSON.read_text(encoding="utf-8"))
    check("row count", got["rows"] == want["rows"], f"{got['rows']} vs {want['rows']}")
    check("per-kind counts", got["kinds"] == want["kinds"], f"{got['kinds']} vs {want['kinds']}")
    diff = sorted(c for c in set(got["columns"]) | set(want["columns"])
                  if got["columns"].get(c) != want["columns"].get(c))
    check("column checksums", not diff, str(diff))
    return summary()


if __name__ == "__main__":
    sys.exit(main())
