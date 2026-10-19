"""
Nav-mesh coverage (P2.2.3), demo-free version: for every active-pool map, >= 99% of
nav-area centroids must `find_area` back to their own area id.

    .venv\\Scripts\\python.exe test\\engine\\test_nav_coverage.py [path\\to\\demo.dem]

With a demo path it also checks the plan's original criterion: >= 99.5% of kill
positions (attacker and victim) in that demo resolve to some nav area.
Needs the pinned awpy-data release cached; exits 2 if it is missing.
"""
import sys
from pathlib import Path

from _harness import check, section, skip, summary

from engine.maps import assets

MIN_SELF_SHARE = 0.99
MIN_KILL_SHARE = 0.995


def centroid_coverage(map_name: str) -> None:
    a = assets.ensure(map_name)
    df = a.nav.areas
    rows = zip(df["area_id"], df["centroid_x"], df["centroid_y"], df["centroid_z"])
    hits = sum(a.nav.find_area((float(x), float(y), float(z))) == int(i) for i, x, y, z in rows)
    share = hits / df.height
    check(f"{map_name}: centroids find their own area >= {MIN_SELF_SHARE:.0%}", share >= MIN_SELF_SHARE,
          f"{hits}/{df.height} = {share:.4f}")


def kill_coverage(demo_path: Path) -> None:
    from engine.parse import awpy3_parser
    section(f"kill positions in {demo_path.name}")
    tables, meta = awpy3_parser.parse(demo_path)
    a = assets.ensure(meta["map"])
    k = tables["kills"]
    pts = [(x, y, z) for p in ("attacker", "victim")
           for x, y, z in zip(k[f"{p}_x"], k[f"{p}_y"], k[f"{p}_z"]) if x is not None]
    hits = sum(a.nav.find_area((float(x), float(y), float(z))) is not None for x, y, z in pts)
    share = hits / len(pts) if pts else 0.0
    check(f"kill positions on the nav mesh >= {MIN_KILL_SHARE:.1%}", share >= MIN_KILL_SHARE,
          f"{hits}/{len(pts)} = {share:.4f}")


def main() -> int:
    if not assets.release_dir().is_dir():
        print(f"awpy-data release {assets.pinned_release()} not cached under {assets.cache_root()}")
        return 2
    section("nav-area centroids")
    for m in assets.ACTIVE_POOL:
        centroid_coverage(m)
    if len(sys.argv) > 1:
        kill_coverage(Path(sys.argv[1]))
    else:
        skip("kill-position coverage on a real demo", "no demo path given")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
