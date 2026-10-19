"""
Callouts (P2.3): polygon validation, point-in-polygon lookup, nearest-within-128 u,
the `area_<id>` nav fallback, off-map None, and every active-pool map has a loadable
callouts file.

    .venv\\Scripts\\python.exe test\\engine\\test_callouts.py

The validation and lookup checks are pure. The fallback checks need the pinned
awpy-data release cached; without it they are skipped (exit 2 only if nothing ran).
"""
import json
import sys
import tempfile
from pathlib import Path

from _harness import check, section, skip, summary

from engine.maps import assets, callouts

SQUARE = callouts.Callout(name="Test Square", polygons=(((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),),
                          z_min=-50.0, z_max=50.0)


def test_validation() -> None:
    section("validate")
    check("a square is valid", callouts.validate([SQUARE]) == [])
    bowtie = callouts.Callout("Bowtie", (((0, 0), (10, 10), (10, 0), (0, 10)),), 0, 1)
    check("self-intersecting polygon rejected", any("self-intersects" in p for p in callouts.validate([bowtie])))
    two = callouts.Callout("Two", (((0, 0), (1, 1)),), 0, 1)
    check("fewer than 3 points rejected", any(">= 3" in p for p in callouts.validate([two])))
    flat = callouts.Callout("Flat", (((0, 0), (1, 1), (2, 2)),), 0, 1)
    check("zero-area polygon rejected", any("zero area" in p for p in callouts.validate([flat])))
    check("duplicate names rejected", any("duplicate" in p for p in callouts.validate([SQUARE, SQUARE])))
    upside = callouts.Callout("Upside", SQUARE.polygons, 10, -10)
    check("z_min > z_max rejected", any("z_min" in p for p in callouts.validate([upside])))
    reserved = callouts.Callout("area_12", SQUARE.polygons, 0, 1)
    check("fallback-shaped name rejected", any("reserved" in p for p in callouts.validate([reserved])))
    closed = callouts.Callout("Closed", (((0, 0), (5, 0), (5, 5), (0, 0)),), 0, 1)
    check("an explicitly closed ring is fine", callouts.validate([closed]) == [])

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "callouts.json"
        p.write_text(json.dumps({"note": "x", "callouts": [{"name": "Test Square", "z_min": -50, "z_max": 50,
                                                             "polygons": [[[0, 0], [100, 0], [100, 100], [0, 100]]]}]}))
        check("load accepts the object form", callouts.load("x", p) == [SQUARE])
        p.write_text(json.dumps([{"name": "A", "z_min": 0, "z_max": 1, "polygons": [[[0, 0], [1, 1]]]}]))
        try:
            callouts.load("x", p)
            check("load refuses an invalid file", False)
        except ValueError as exc:
            check("load refuses an invalid file", ">= 3" in str(exc))


def test_lookup() -> None:
    section("lookup (no nav mesh)")
    cs = [SQUARE]
    check("inside the square", callouts.lookup(None, cs, 50, 50, 0) == "Test Square")
    check("on the edge counts as inside", callouts.lookup(None, cs, 100, 50, 0) == "Test Square")
    check("within 128 u outside -> nearest", callouts.lookup(None, cs, 200, 50, 0) == "Test Square")
    check("beyond 128 u -> None without a nav mesh", callouts.lookup(None, cs, 300, 50, 0) is None)
    check("above z range (jump) resolves via nearest", callouts.lookup(None, cs, 50, 50, 120) == "Test Square")
    check("far above z range does not", callouts.lookup(None, cs, 50, 50, 400) is None)
    check("None coordinate -> None", callouts.lookup(None, cs, None, 50, 0) is None)
    near = callouts.Callout("Near", (((150, 0), (250, 0), (250, 100), (150, 100)),), -50, 50)
    check("the nearer of two callouts wins", callouts.lookup(None, [SQUARE, near], 140, 50, 0) == "Near")


def test_fallback() -> None:
    section("nav fallback")
    if not assets.release_dir().is_dir():
        skip("area_<id> fallback", "awpy-data release not cached")
        return
    a = assets.ensure("de_mirage")
    row = a.nav.areas.row(0, named=True)
    x, y, z = row["centroid_x"], row["centroid_y"], row["centroid_z"]
    got = callouts.lookup(a, [], x, y, z)
    check("on-mesh point with no callouts -> area_<id>", got == f"area_{row['area_id']}", f"{got}")
    check("is_fallback recognises it", callouts.is_fallback(got) and not callouts.is_fallback("A Site"))
    check("off-map point -> None", callouts.lookup(a, [], 1e6, 1e6, 0.0) is None)
    check("a real callout beats the fallback",
          callouts.lookup(a, [callouts.Callout("Spot", (((x - 10, y - 10), (x + 10, y - 10), (x + 10, y + 10),
                                                         (x - 10, y + 10)),), z - 100, z + 100)], x, y, z) == "Spot")


def test_files() -> None:
    section("callouts files")
    for m in assets.ACTIVE_POOL:
        p = callouts.path_for(m)
        try:
            cs = callouts.load(m)
            doc = json.loads(p.read_text(encoding="utf-8"))
            note = isinstance(doc, list) or bool(doc.get("note"))
            check(f"{m}: callouts.json loads and validates", note, f"{len(cs)} callouts")
        except (FileNotFoundError, ValueError) as exc:
            check(f"{m}: callouts.json loads and validates", False, str(exc))


def main() -> int:
    test_validation()
    test_lookup()
    test_fallback()
    test_files()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
