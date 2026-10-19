"""
Position -> coarse 128-unit grid cell over the nav-mesh extents (P2.4.1). Ids are
row-major, within [0, n_cells), and stable for a given map + awpy-data release
(the extents come only from the nav mesh, never from a demo).

    from engine.maps import assets, grid
    a = assets.ensure("de_mirage")
    grid.cell(a, x, y)          # int, or None outside the grid
    grid.n_cells(a)
"""
import math
from dataclasses import dataclass

CELL_SIZE = 128       # u, cell edge
MARGIN    = 256       # u, added around the nav-area centroid extents (areas have size)


@dataclass(frozen=True)
class Grid:
    x0:     float     # lower-left corner, a multiple of CELL_SIZE
    y0:     float
    n_cols: int
    n_rows: int
    size:   int = CELL_SIZE

    @property
    def n_cells(self) -> int:
        return self.n_cols * self.n_rows

    def cell(self, x: float, y: float) -> int | None:
        if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
            return None
        col = math.floor((x - self.x0) / self.size)
        row = math.floor((y - self.y0) / self.size)
        if not (0 <= col < self.n_cols and 0 <= row < self.n_rows):
            return None
        return row * self.n_cols + col

    def bounds(self, cell_id: int) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) of a cell."""
        row, col = divmod(int(cell_id), self.n_cols)
        x, y = self.x0 + col * self.size, self.y0 + row * self.size
        return x, y, x + self.size, y + self.size


def from_extents(x_min: float, y_min: float, x_max: float, y_max: float,
                 margin: float = MARGIN, size: int = CELL_SIZE) -> Grid:
    """Grid covering the box plus `margin`, snapped outward to multiples of `size`."""
    x0 = math.floor((x_min - margin) / size) * size
    y0 = math.floor((y_min - margin) / size) * size
    x1 = math.ceil((x_max + margin) / size) * size
    y1 = math.ceil((y_max + margin) / size) * size
    return Grid(x0=float(x0), y0=float(y0), n_cols=int((x1 - x0) // size),
                n_rows=int((y1 - y0) // size), size=size)


_GRIDS: dict[tuple[str, str], Grid] = {}


def grid_for(map_assets) -> Grid:
    """The grid of a MapAssets (cached per map + release)."""
    key = (map_assets.map_name, map_assets.release)
    if key not in _GRIDS:
        a = map_assets.nav.areas
        _GRIDS[key] = from_extents(float(a["centroid_x"].min()), float(a["centroid_y"].min()),
                                   float(a["centroid_x"].max()), float(a["centroid_y"].max()))
    return _GRIDS[key]


def cell(map_assets, x: float, y: float) -> int | None:
    return grid_for(map_assets).cell(x, y)


def n_cells(map_assets) -> int:
    return grid_for(map_assets).n_cells
