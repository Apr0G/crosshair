"""
Pairwise table (P3.5) and the geometry seam: shape, symmetry, NaN diagonal, LOS only for
alive enemy pairs, smoke_between (segment-sphere intersection), nav distance, same_team,
LOS cache agreement, MapGeometry adapter over a fake asset object.

    .venv\\Scripts\\python.exe test\\engine\\test_pairwise.py
"""
import math
import sys

import numpy as np

from _harness import check, section, summary
from _synthetic import STORIES, make_raw_tables

from engine import config
from engine.state import builder, pairwise, players
from engine.state.geometry import FlatGeometry, Geometry, MapGeometry
from engine.state.schema import PAIR_CHANNELS, Snapshot, UtilityToken

CFG = config.load()
GEO = FlatGeometry()
LOS, DIST, NAV, SMOKE, SAME = (PAIR_CHANNELS.index(c) for c in PAIR_CHANNELS)


def alive_token(side, x, y, z=0.0, pid=1):
    t = players.pad_token(side, "pro")
    t.alive, t.x, t.y, t.z, t.player_id = 1, x, y, z, pid
    t.health, t.armor, t.in_smoke, t.on_fire = 100.0, 100.0, 0, 0
    t.nav_area = GEO.find_area((x, y, z))
    return t


def smoke(x, y, z=64.0):
    return UtilityToken(kind="smoke", x=x, y=y, z=z, time_left_s=10.0, owner_side="ct", owner_id=1,
                        radius_units=144.0)


# ── geometry ──
section("segment / sphere")
check("segment through the centre hits", pairwise.segment_hits_sphere((0, 0, 0), (1000, 0, 0), (500, 0, 0), 144))
check("sphere 300 u off the segment misses", not pairwise.segment_hits_sphere((0, 0, 0), (1000, 0, 0), (500, 300, 0), 144))
check("tangent-ish inside radius hits", pairwise.segment_hits_sphere((0, 0, 0), (1000, 0, 0), (500, 143, 0), 144))
check("sphere beyond the segment end misses", not pairwise.segment_hits_sphere((0, 0, 0), (1000, 0, 0), (1300, 0, 0), 144))
check("endpoint inside counts", pairwise.segment_hits_sphere((0, 0, 0), (1000, 0, 0), (1100, 0, 0), 144))
check("degenerate segment", pairwise.segment_hits_sphere((5, 0, 0), (5, 0, 0), (0, 0, 0), 10))

# ── crafted table ──
section("crafted 10-slot table")
toks = [alive_token("ct", 0.0, 0.0, pid=1), alive_token("ct", 0.0, 600.0, pid=2)] + \
       [players.pad_token("ct", "pro") for _ in range(3)] + \
       [alive_token("t", 1000.0, 0.0, pid=3), alive_token("t", 1000.0, 600.0, pid=4)] + \
       [players.pad_token("t", "pro") for _ in range(3)]
dead = players.pad_token("t", "pro")
dead.player_id, dead.x, dead.y, dead.z = 5, 2000.0, 0.0, 0.0
toks[7] = dead
pr = pairwise.table(toks, GEO, [smoke(500.0, 0.0)])
check("shape (10,10,5) float32", pr.shape == (10, 10, 5) and pr.dtype == np.float32)
check("diagonal all NaN", np.isnan(pr[np.arange(10), np.arange(10), :]).all())
check("every channel symmetric", all(np.array_equal(pr[:, :, k], pr[:, :, k].T, equal_nan=True) for k in range(5)))
check("smoke centred on the 0-5 segment midpoint -> smoke_between 1", pr[0, 5, SMOKE] == 1.0)
check("smoke 300+ u off the 1-6 segment -> 0", pr[1, 6, SMOKE] == 0.0)
check("LOS 1 for alive enemies on flat ground", pr[0, 5, LOS] == 1.0 and pr[1, 6, LOS] == 1.0)
check("LOS NaN for teammates", np.isnan(pr[0, 1, LOS]) and np.isnan(pr[5, 6, LOS]))
check("LOS NaN when either is dead", np.isnan(pr[0, 7, LOS]) and np.isnan(pr[7, 0, LOS]))
check("smoke_between NaN when either is dead", np.isnan(pr[0, 7, SMOKE]))
check("dist 3D, dead keep their position", math.isclose(pr[0, 5, DIST], 1000.0) and math.isclose(pr[0, 7, DIST], 2000.0))
check("dist NaN for empty slots", np.isnan(pr[0, 2, DIST]))
check("nav dist from geometry areas", math.isclose(pr[0, 5, NAV], GEO.nav_distance(toks[0].nav_area, toks[5].nav_area), rel_tol=1e-6))
check("nav dist NaN when an area is unknown", np.isnan(pr[0, 7, NAV]))
check("same_team for every off-diagonal pair incl. empty slots",
      pr[0, 1, SAME] == 1.0 and pr[0, 5, SAME] == 0.0 and pr[2, 3, SAME] == 1.0 and pr[2, 8, SAME] == 0.0)
check("validate() accepts it (pair part)", Snapshot(players=toks, utility=[], ctx=None, pair=pr)._validate_pair() == [])


class _Walls(FlatGeometry):
    """Flat ground with a wall on x = 500 for y > 300."""

    def is_visible(self, a, b):
        return not (min(a[0], b[0]) < 500 < max(a[0], b[0]) and a[1] > 300 and b[1] > 300)


pw = pairwise.table(toks, _Walls(), [])
check("geometry LOS is used (wall blocks 1-6, not 0-5)", pw[1, 6, LOS] == 0.0 and pw[0, 5, LOS] == 1.0)


class _Unknown(FlatGeometry):
    def is_visible(self, a, b):
        return None


check("geometry None -> LOS NaN", np.isnan(pairwise.table(toks, _Unknown(), [])[0, 5, LOS]))

# ── cache ──
section("LOS cache")
calls = []


class _Counting(_Walls):
    def is_visible(self, a, b):
        calls.append((a, b))
        return super().is_visible(a, b)


cache = {}
p1 = pairwise.table(toks, _Counting(), [], los_cache=cache)
n1 = len(calls)
p2 = pairwise.table(toks, _Counting(), [], los_cache=cache)
check("second identical table served from the cache", len(calls) == n1 and np.array_equal(p1, p2, equal_nan=True))
pairwise.table(toks, _Counting(), [smoke(5000.0, 5000.0)], los_cache=cache)
check("a smoke change invalidates the key", len(calls) > n1)
agree = []
for story in STORIES:
    raw = make_raw_tables(story, 128)
    exact = builder.build(raw, GEO, CFG, match_key="p", tier="pro", los_cache=False, labels=False)
    cached = builder.build(raw, GEO, CFG, match_key="p", tier="pro", los_cache=True, labels=False)
    agree.append(all(np.array_equal(a.pair, b.pair, equal_nan=True) for a, b in zip(exact, cached)))
check("exact vs cached agree on every story (flat geometry)", all(agree))

# ── builder output ──
section("builder snapshots")
raw = make_raw_tables("base", 128)
snaps = builder.build(raw, GEO, CFG, match_key="p", tier="pro", labels=False)
bad = [s.meta.tick for s in snaps if s._validate_pair()]
check("every snapshot's pair passes validate", not bad, str(bad[:5]))
s = next(s for s in snaps if s.meta.tick == 1512)
# slots: 0 A, 1 B, 5 C, 6 D
check("B inside the smoke at 1512: smoke_between B-C 1, A-C 0", s.pair[1, 5, SMOKE] == 1.0 and s.pair[0, 5, SMOKE] == 0.0)
s = next(s for s in snaps if s.meta.tick == 2001)
check("C dead at 2001: LOS NaN for C's pairs", np.isnan(s.pair[5, :, LOS]).all())

# ── geometry seam ──
section("geometry seam")
check("FlatGeometry satisfies the Protocol", isinstance(GEO, Geometry))
check("FlatGeometry: 512-u cells, None off-map, None for unknown",
      GEO.find_area((0, 0, 0)) == GEO.find_area((511, 511, 0)) != GEO.find_area((512, 0, 0))
      and GEO.find_area((1e9, 0, 0)) is None and GEO.find_area((None, 0, 0)) is None
      and GEO.nav_distance(None, 3) is None and GEO.callout(None, 0, 0) is None)


class _Nav:
    def find_area(self, p):
        return 7 if p[0] >= 0 else None


class _Vis:
    def is_visible(self, a, b):
        return a[1] == b[1]


class _Assets:
    map_name, release = "de_fake", "123"
    nav, visibility = _Nav(), _Vis()

    def nav_distance(self, a, b):
        return 42.0 if a == b else None


mg = MapGeometry.from_assets(_Assets(), [])
check("MapGeometry: default map_version", mg.map_version == "de_fake@123")
check("MapGeometry: visibility / area / nav distance pass through",
      mg.is_visible((0, 0, 0), (5, 0, 0)) is True and mg.is_visible((0, 0, 0), (0, 5, 0)) is False
      and mg.find_area((1, 1, 1)) == 7 and mg.find_area((-1, 1, 1)) is None
      and mg.nav_distance(7, 7) == 42.0 and mg.nav_distance(None, 7) is None)
check("MapGeometry satisfies the Protocol", isinstance(mg, Geometry))
check("explicit map_version wins", MapGeometry.from_assets(_Assets(), [], "de_fake@v2").map_version == "de_fake@v2")

sys.exit(summary())
