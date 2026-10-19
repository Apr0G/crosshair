"""
Grid cells (P2.4.1): 128-unit cells over the nav-mesh extents, ids within
[0, n_cells), corner points, stability.

    .venv\\Scripts\\python.exe test\\engine\\test_grid.py

The synthetic-extent checks are pure; the per-map checks need the pinned awpy-data
release cached (skipped otherwise).
"""
import sys

from _harness import check, section, skip, summary

from engine.maps import assets, grid


def test_synthetic() -> None:
    section("synthetic extents")
    g = grid.from_extents(0, 0, 1000, 500, margin=0)
    check("snapped to 128 multiples", (g.x0, g.y0) == (0.0, 0.0) and g.n_cols == 8 and g.n_rows == 4,
          f"{g}")
    check("n_cells = cols * rows", g.n_cells == 32)
    check("lower-left corner is cell 0", g.cell(0, 0) == 0)
    check("x just inside the first cell", g.cell(127.9, 0) == 0 and g.cell(128, 0) == 1)
    check("next row", g.cell(0, 128) == g.n_cols)
    check("upper-right inside corner is the last cell", g.cell(1023.9, 511.9) == g.n_cells - 1)
    check("outside -> None", g.cell(-0.1, 0) is None and g.cell(1024, 0) is None and g.cell(0, 512) is None)
    check("NaN / None -> None", g.cell(float("nan"), 0) is None and g.cell(None, 0) is None)
    check("bounds round-trip", g.bounds(g.cell(300, 200)) == (256.0, 128.0, 384.0, 256.0))
    gm = grid.from_extents(-100, -100, 100, 100)
    check("margin and negative coordinates", gm.x0 == -384.0 and gm.cell(-100, -100) is not None)


def test_maps() -> None:
    section("nav-mesh grids")
    if not assets.release_dir().is_dir():
        skip("per-map grids", "awpy-data release not cached")
        return
    for m in assets.ACTIVE_POOL:
        a = assets.ensure(m)
        n = grid.n_cells(a)
        df = a.nav.areas
        ids = [grid.cell(a, x, y) for x, y in zip(df["centroid_x"], df["centroid_y"])]
        ok = all(i is not None and 0 <= i < n for i in ids)
        check(f"{m}: every centroid in [0, {n})", ok)
        check(f"{m}: grid is stable", grid.grid_for(a) == grid.from_extents(
            float(df["centroid_x"].min()), float(df["centroid_y"].min()),
            float(df["centroid_x"].max()), float(df["centroid_y"].max())))


def main() -> int:
    test_synthetic()
    test_maps()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
