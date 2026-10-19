"""
Line-of-sight fixtures per map (P2.2.2), STRUCTURAL version.

    .venv\\Scripts\\python.exe test\\engine\\test_los_fixtures.py [--regen]

The plan asks for 8 visible + 8 blocked pairs at well-known named positions. No named
coordinates have been verified yet, so these fixtures are structural instead, chosen
at random (fixed seed) from the nav mesh WITHOUT looking at visibility:

- a point is visible from itself;
- of 8 adjacent nav-area centroid pairs (eye height above both), >= 6 are visible;
- of 8 area pairs >= 2,000 u apart with |dz| >= 300 u, >= 6 are occluded.

The pairs live in `test/engine/fixtures/los_<map>.json` (status field says they are
structural). `--regen` re-draws them; otherwise a missing fixture file fails.
Needs the pinned awpy-data release cached; exits 2 if it is missing.
"""
import json
import math
import random
import sys
from pathlib import Path

from _harness import check, section, summary

from engine.maps import assets

FIXTURES   = Path(__file__).resolve().parent / "fixtures"
STATUS     = "structural; named callout fixtures pending"
EYE        = 64.0          # u above the area centroid (standing eye height)
N_PAIRS    = 8
MIN_OK     = 6
FAR_MIN    = 2000.0
DZ_MIN     = 300.0


def fixture_path(map_name: str) -> Path:
    return FIXTURES / f"los_{map_name}.json"


def _eye(a, area_id: int) -> list[float]:
    x, y, z = a.centroid(area_id)
    return [round(x, 2), round(y, 2), round(z + EYE, 2)]


def generate(map_name: str) -> dict:
    a = assets.ensure(map_name)
    rng = random.Random(f"los-{map_name}")
    ids = sorted(int(i) for i in a.nav.areas["area_id"])
    adjacent, seen = [], set()
    while len(adjacent) < N_PAIRS:
        u = rng.choice(ids)
        nbrs = [n for n in a.nav.neighbors(u) if a.centroid(n) is not None]
        if not nbrs:
            continue
        v = rng.choice(nbrs)
        if (u, v) in seen:
            continue
        seen.add((u, v))
        adjacent.append({"areas": [u, v], "a": _eye(a, u), "b": _eye(a, v)})
    far, tries = [], 0
    while len(far) < N_PAIRS and tries < 200_000:
        tries += 1
        u, v = rng.sample(ids, 2)
        cu, cv = a.centroid(u), a.centroid(v)
        if math.dist(cu, cv) >= FAR_MIN and abs(cu[2] - cv[2]) >= DZ_MIN:
            far.append({"areas": [u, v], "a": _eye(a, u), "b": _eye(a, v)})
    s = rng.choice(ids)
    return {"map": map_name, "release": a.release, "status": STATUS, "eye_height": EYE,
            "seed": f"los-{map_name}", "self": {"area": s, "p": _eye(a, s)},
            "adjacent_visible": adjacent, "far_occluded": far}


def run_map(map_name: str) -> None:
    section(map_name)
    p = fixture_path(map_name)
    if not p.is_file():
        check(f"{map_name}: fixture file exists", False, f"run with --regen to create {p.name}")
        return
    fx = json.loads(p.read_text(encoding="utf-8"))
    a = assets.ensure(map_name)
    check(f"{map_name}: fixture status is structural", fx.get("status") == STATUS)
    check(f"{map_name}: fixture release matches the pin", fx.get("release") == a.release)
    vis = a.visibility
    pt = tuple(fx["self"]["p"])
    check(f"{map_name}: a point sees itself", vis.is_visible(pt, pt))
    n_vis = sum(vis.is_visible(tuple(q["a"]), tuple(q["b"])) for q in fx["adjacent_visible"])
    check(f"{map_name}: adjacent centroid pairs visible >= {MIN_OK}/{N_PAIRS}",
          len(fx["adjacent_visible"]) == N_PAIRS and n_vis >= MIN_OK, f"{n_vis}/{len(fx['adjacent_visible'])}")
    n_occ = sum(vis.is_occluded(tuple(q["a"]), tuple(q["b"])) for q in fx["far_occluded"])
    check(f"{map_name}: far pairs (>= {FAR_MIN:.0f} u, |dz| >= {DZ_MIN:.0f}) occluded >= {MIN_OK}/{N_PAIRS}",
          len(fx["far_occluded"]) == N_PAIRS and n_occ >= MIN_OK, f"{n_occ}/{len(fx['far_occluded'])}")


def test_geometry_errors() -> None:
    """LOW-22: MapGeometry reads only a lookup error as unknown (logged once); any other
    exception from awpy propagates instead of silently becoming NaN LOS."""
    import logging
    from engine.state import geometry as geo_mod

    section("MapGeometry error handling (stub assets, LOW-22)")

    class _Raiser:
        def __init__(self, exc):
            self.exc = exc

        def is_visible(self, a, b):
            raise self.exc

        def find_area(self, p):
            raise self.exc

    class _Stub:
        map_name, release = "de_stub", "0"

        def __init__(self, exc):
            self.visibility = self.nav = _Raiser(exc)

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    geo_mod.log.addHandler(handler)
    geo_mod._LOGGED_LOOKUP.clear()
    try:
        g = geo_mod.MapGeometry.from_assets(_Stub(KeyError("area")), [])
        p, q = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        out = [g.is_visible(p, q), g.is_visible(p, q), g.find_area(p), g.find_area(p)]
        check("a lookup error is unknown (None) for is_visible and find_area", out == [None] * 4, f"{out}")
        msgs = [r.getMessage() for r in records]
        check("logged once per method, with the exception type",
              len(msgs) == 2 and all("KeyError" in m for m in msgs), f"{msgs}")
        for exc in (TypeError("signature changed"), ValueError("expected a sequence of length 3")):
            g = geo_mod.MapGeometry.from_assets(_Stub(exc), [])
            for name, call in (("is_visible", lambda: g.is_visible(p, q)), ("find_area", lambda: g.find_area(p))):
                try:
                    call()
                    check(f"{name}: {type(exc).__name__} propagates", False, "swallowed")
                except type(exc):
                    check(f"{name}: {type(exc).__name__} propagates", True)
    finally:
        geo_mod.log.removeHandler(handler)


def main() -> int:
    test_geometry_errors()
    if not assets.release_dir().is_dir():
        print(f"awpy-data release {assets.pinned_release()} not cached under {assets.cache_root()}")
        return 2
    if "--regen" in sys.argv:
        FIXTURES.mkdir(parents=True, exist_ok=True)
        for m in assets.ACTIVE_POOL:
            fixture_path(m).write_text(json.dumps(generate(m), indent=2) + "\n", encoding="utf-8")
            print(f"wrote {fixture_path(m)}")
    for m in assets.ACTIVE_POOL:
        run_map(m)
    section("MapGeometry on real assets: an out-of-mesh point raises nothing (LOW-22)")
    from engine.state.geometry import MapGeometry
    g = MapGeometry.from_assets(assets.ensure("de_mirage"), [])
    far = (1e6, 1e6, 1e6)
    check("out-of-mesh find_area -> None", g.find_area(far) is None)
    check("out-of-mesh is_visible -> a bool, not an exception",
          isinstance(g.is_visible((0.0, 0.0, 0.0), far), bool))
    return summary()


if __name__ == "__main__":
    sys.exit(main())
