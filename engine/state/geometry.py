"""
Map geometry seen by the state builder (P3.2 / P3.5): one small Protocol so feature code
never imports awpy or `engine.maps` directly.

    from engine.state.geometry import FlatGeometry, MapGeometry
    geo = FlatGeometry()                                     # synthetic maps (de_test)
    geo = MapGeometry.from_assets(assets, callouts, map_version="de_mirage@v1")

`FlatGeometry` is the synthetic stand-in: open ground, everything visible (smoke
blocking is the `smoke_between` channel, computed in `pairwise.py`), "nav areas" are
512-unit square cells and nav distance is the 2D distance between cell centres.
`MapGeometry` adapts the P2 map assets; its `engine.maps` imports are lazy, so this
module imports (and the synthetic tests run) even when map assets are absent.
Every method returns None for "unknown", never 0 or "?".
"""
import logging
import math
from typing import Protocol, runtime_checkable

Point = tuple[float, float, float]

log = logging.getLogger("engine.state.geometry")

# ── protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class Geometry(Protocol):
    map_version: str

    def is_visible(self, a: Point, b: Point) -> bool | None: ...
    def find_area(self, p: Point) -> int | None: ...
    def nav_distance(self, area_a: int, area_b: int) -> float | None: ...
    def callout(self, x: float, y: float, z: float) -> str | None: ...
    def grid_cell(self, x: float, y: float) -> int | None: ...


def _finite(*vals) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)


# ── synthetic: flat open ground ───────────────────────────────────────────────

FLAT_CELL   = 512.0      # u, edge of a FlatGeometry cell ("nav area" and grid cell)
_FLAT_OFF   = 64         # cell index offset, so ids are non-negative
_FLAT_SPAN  = 128        # cells per axis: covers +-32768 u, the Source map limit


class FlatGeometry:
    """Synthetic geometry for `de_test`: no walls, 512-unit cells."""

    def __init__(self, map_version: str = "de_test@flat"):
        self.map_version = map_version

    def _cell(self, x: float, y: float) -> int | None:
        if not _finite(x, y):
            return None
        ix = math.floor(x / FLAT_CELL) + _FLAT_OFF
        iy = math.floor(y / FLAT_CELL) + _FLAT_OFF
        if not (0 <= ix < _FLAT_SPAN and 0 <= iy < _FLAT_SPAN):
            return None
        return ix * _FLAT_SPAN + iy

    def cell_centre(self, cell_id: int) -> tuple[float, float]:
        ix, iy = divmod(int(cell_id), _FLAT_SPAN)
        return ((ix - _FLAT_OFF + 0.5) * FLAT_CELL, (iy - _FLAT_OFF + 0.5) * FLAT_CELL)

    def is_visible(self, a: Point, b: Point) -> bool | None:
        if not (_finite(*a) and _finite(*b)):
            return None
        return True

    def find_area(self, p: Point) -> int | None:
        return self._cell(p[0], p[1])

    def nav_distance(self, area_a: int, area_b: int) -> float | None:
        if area_a is None or area_b is None:
            return None
        return math.dist(self.cell_centre(area_a), self.cell_centre(area_b))

    def callout(self, x: float, y: float, z: float) -> str | None:
        c = self._cell(x, y)
        return None if c is None else f"cell_{c}"

    def grid_cell(self, x: float, y: float) -> int | None:
        return self._cell(x, y)


# ── real maps: adapter over engine.maps (P2) ──────────────────────────────────
# awpy 3.1 answers an out-of-mesh point with find_area -> None and is_visible -> a bool
# (checked on de_mirage, 2026-10-19); it raises nothing for it. Only a lookup error is
# read as "unknown" (logged once per process with its type); anything else, e.g. a
# ValueError / TypeError from an awpy signature change, propagates instead of turning
# every LOS value into NaN (LOW-22).
_LOOKUP_ERRORS            = (LookupError,)
_LOGGED_LOOKUP: set[str]  = set()


def _lookup_failed(what: str, exc: Exception) -> None:
    if what not in _LOGGED_LOOKUP:
        _LOGGED_LOOKUP.add(what)
        log.warning(f"{what}: {type(exc).__name__} from awpy read as unknown (logged once per process)")


class MapGeometry:
    """Adapter over `engine.maps.assets.MapAssets` + a callout list (P2.2-P2.4)."""

    def __init__(self, assets, callouts: list, map_version: str):
        self.assets      = assets
        self.callouts    = callouts
        self.map_version = map_version

    @classmethod
    def from_assets(cls, assets, callouts: list, map_version: str | None = None) -> "MapGeometry":
        """`map_version` should come from `engine.maps.version.detect`; without it the
        version falls back to `<map>@<awpy-data release>`."""
        if map_version is None:
            map_version = f"{assets.map_name}@{assets.release}"
        return cls(assets, callouts, map_version)

    def is_visible(self, a: Point, b: Point) -> bool | None:
        if not (_finite(*a) and _finite(*b)):
            return None
        try:
            return bool(self.assets.visibility.is_visible(tuple(map(float, a)), tuple(map(float, b))))
        except _LOOKUP_ERRORS as exc:
            _lookup_failed("is_visible", exc)
            return None

    def find_area(self, p: Point) -> int | None:
        if not _finite(*p):
            return None
        try:
            area = self.assets.nav.find_area(tuple(map(float, p)))
        except _LOOKUP_ERRORS as exc:
            _lookup_failed("find_area", exc)
            return None
        if area is None:
            return None
        return int(getattr(area, "area_id", area))

    def nav_distance(self, area_a: int, area_b: int) -> float | None:
        if area_a is None or area_b is None:
            return None
        return self.assets.nav_distance(int(area_a), int(area_b))

    def callout(self, x: float, y: float, z: float) -> str | None:
        if not _finite(x, y, z):
            return None
        from engine.maps import callouts as _callouts
        return _callouts.lookup(self.assets, self.callouts, x, y, z)

    def grid_cell(self, x: float, y: float) -> int | None:
        if not _finite(x, y):
            return None
        from engine.maps import grid as _grid
        return _grid.cell(self.assets, x, y)
