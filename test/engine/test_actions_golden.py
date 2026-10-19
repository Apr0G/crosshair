"""
Golden action counts (P4.7.2): extract the golden demo's actions (exact-LOS states) and
compare per-type counts, the window count and the kill count against `len(kills)` with
`golden/actions.json`.

    .venv\\Scripts\\python.exe test\\engine\\test_actions_golden.py [--write]

Exit 2 while no golden demo is pinned (GOLDEN.json demo_sha256 null), the demo file is
missing, the parser has no `parse(path) -> RawTables` yet, map assets are missing, or
`actions.json` has not been written (`--write` creates it from the current extractor,
after a human has checked the output).
"""
import json
import sys
from collections import Counter
from pathlib import Path

from _harness import check, section, summary

from engine import config
from engine.actions import extractor

HERE         = Path(__file__).resolve().parent
GOLDEN_JSON  = HERE / "golden" / "GOLDEN.json"
ACTIONS_JSON = HERE / "golden" / "actions.json"


def counts(actions, windows) -> dict:
    return {"types": dict(sorted(Counter(a.type for a in actions).items())), "windows": len(windows),
            "actions": len(actions)}


def main() -> int:
    section("golden actions")
    meta = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    if not meta.get("demo_sha256"):
        print(f"  no golden demo pinned yet (demo_sha256 is null in {GOLDEN_JSON.name}); "
              f"actions golden needs one (P0.7.1)")
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
    _, actions, windows = extractor.run(raw, config.load(), match_key="golden", geometry=geo,
                                        los_cache=False)
    got = counts(actions, windows)
    n_kill_rows = raw.kills.filter(raw.kills["attacker_steamid"] != raw.kills["victim_steamid"]).height
    check("kill + team_kill actions = kill rows (suicides excluded)",
          got["types"].get("kill", 0) + got["types"].get("team_kill", 0) == n_kill_rows)
    if "--write" in sys.argv:
        ACTIONS_JSON.write_text(json.dumps(got, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  wrote {ACTIONS_JSON}")
        return summary()
    if not ACTIONS_JSON.is_file():
        print(f"  {ACTIONS_JSON.name} not written yet; run with --write after checking the output")
        return 2
    want = json.loads(ACTIONS_JSON.read_text(encoding="utf-8"))
    check("per-type counts", got["types"] == want["types"], f"{got['types']} vs {want['types']}")
    check("window count", got["windows"] == want["windows"], f"{got['windows']} vs {want['windows']}")
    check("action count", got["actions"] == want["actions"], f"{got['actions']} vs {want['actions']}")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
