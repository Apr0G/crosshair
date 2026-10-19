"""
Map assets (P2.2.1): collision mesh + `.nav` per map from awpy-data, pinned to
`config.data.awpy_data_release`, with sha256 recorded. Also nav-graph distances
between areas (P2.2.4).

    from engine.maps import assets
    a = assets.ensure("de_mirage")          # MapAssets, cached per process
    a.visibility.is_visible(p, q)           # awpy VisibilityChecker
    a.nav.find_area((x, y, z))              # awpy NavMesh
    a.nav_distance(area_a, area_b)          # shortest nav path length (u) or None
    assets.asset_hashes("de_mirage")        # for run manifests

Files are read from awpy's own cache (`~/.awpy/<release>/geometry|navs/`, or
`AWPY_DATA_DIR`). This module never downloads: a map or release that is not already
cached is refused with MapAssetError (fetch it deliberately with `awpy get`).
The optional all-pairs table goes to `data/engine/cache/maps/<map>_navdist.npy`.
"""
import hashlib
import heapq
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from engine import paths

# The known map pool for quarantine (P1.4.1): the maps whose assets were cached and
# verified on 2026-10-19 (docs/ENGINE_ENV.md, "Map assets").
ACTIVE_POOL = ("de_ancient", "de_anubis", "de_dust2", "de_inferno", "de_mirage",
               "de_nuke", "de_overpass", "de_train", "de_vertigo")

NAVDIST_UNREACHABLE = 65535          # uint16 sentinel in the all-pairs table
NAVDIST_MAX         = 65534          # longer paths are clipped (none exist on real maps)


class MapAssetError(RuntimeError):
    """The map or the pinned release is not in the local awpy cache."""


# ── release and paths ─────────────────────────────────────────────────────────

def pinned_release(cfg=None) -> str:
    """The awpy-data release from config (`data.awpy_data_release`) as a string."""
    if cfg is None:
        from engine import config
        cfg = config.load()
    rel = cfg.data.awpy_data_release
    if rel is None:
        raise MapAssetError("config data.awpy_data_release is null: pin a release first")
    return str(rel)


def cache_root() -> Path:
    """awpy's cache root (`~/.awpy` or AWPY_DATA_DIR), read at call time."""
    import awpy.data
    return Path(awpy.data.AWPY_DATA_DIR)


def release_dir(release: str | None = None) -> Path:
    return cache_root() / (release or pinned_release())


def asset_paths(map_name: str, release: str | None = None) -> tuple[Path, Path]:
    d = release_dir(release)
    return d / "geometry" / f"{map_name}.mesh", d / "navs" / f"{map_name}.nav"


def cached_maps(release: str | None = None) -> list[str]:
    """Maps with both a mesh and a nav in the cached release (empty if not cached)."""
    d = release_dir(release)
    if not (d / "navs").is_dir() or not (d / "geometry").is_dir():
        return []
    navs = {p.stem for p in (d / "navs").glob("*.nav")}
    return sorted(navs & {p.stem for p in (d / "geometry").glob("*.mesh")})


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ── MapAssets ─────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class MapAssets:
    map_name:    str
    release:     str
    visibility:  object              # awpy.VisibilityChecker
    nav:         object              # awpy.NavMesh
    mesh_sha256: str
    nav_sha256:  str
    mesh_path:   Path
    nav_path:    Path
    _centroids:  dict = field(default_factory=dict, repr=False)

    def centroid(self, area_id: int) -> tuple[float, float, float] | None:
        if not self._centroids:
            a = self.nav.areas
            self._centroids.update({int(i): (float(x), float(y), float(z)) for i, x, y, z in
                                    zip(a["area_id"], a["centroid_x"], a["centroid_y"], a["centroid_z"])})
        return self._centroids.get(int(area_id))

    def nav_distance(self, area_a: int, area_b: int) -> float | None:
        return nav_distance(self.map_name, area_a, area_b, release=self.release)

    def hashes(self) -> dict[str, str]:
        return {"release": self.release, f"{self.map_name}.mesh": self.mesh_sha256,
                f"{self.map_name}.nav": self.nav_sha256}


_CACHE: dict[tuple[str, str], MapAssets] = {}


def ensure(map_name: str, *, release: str | None = None) -> MapAssets:
    """Load (once per process) the visibility checker and nav mesh of `map_name` from
    the pinned awpy-data release. Refuses a release or map that is not cached."""
    rel = str(release) if release is not None else pinned_release()
    key = (map_name, rel)
    if key in _CACHE:
        return _CACHE[key]
    d = release_dir(rel)
    if not d.is_dir():
        raise MapAssetError(f"awpy-data release {rel} is not cached under {cache_root()} "
                            f"(pin mismatch?); fetch it with `awpy get` first")
    mesh, nav = asset_paths(map_name, rel)
    missing = [p.name for p in (mesh, nav) if not p.is_file()]
    if missing:
        raise MapAssetError(f"map {map_name!r} is not in cached awpy-data release {rel} "
                            f"(missing {', '.join(missing)}); cached maps: {', '.join(cached_maps(rel))}")
    import awpy
    a = MapAssets(map_name=map_name, release=rel,
                  visibility=awpy.VisibilityChecker(mesh), nav=awpy.NavMesh(nav),
                  mesh_sha256=_sha256(mesh), nav_sha256=_sha256(nav), mesh_path=mesh, nav_path=nav)
    _CACHE[key] = a
    return a


def asset_hashes(map_name: str, *, release: str | None = None) -> dict[str, str]:
    """`{"release", "<map>.mesh", "<map>.nav"}` for `manifest.write(map_assets=...)`."""
    return ensure(map_name, release=release).hashes()


# ── nav distances (P2.2.4) ────────────────────────────────────────────────────

def _dist3(p: tuple, q: tuple) -> float:
    return math.dist(p, q)


@lru_cache(maxsize=200_000)
def _nav_distance(map_name: str, release: str, area_a: int, area_b: int) -> float | None:
    a = ensure(map_name, release=release)
    if area_a == area_b:
        return 0.0 if a.centroid(area_a) is not None else None
    path = a.nav.find_path(int(area_a), int(area_b), weight="distance")
    if not path:
        return None
    pts = [a.centroid(i) for i in path]
    return float(sum(_dist3(p, q) for p, q in zip(pts, pts[1:])))


def nav_distance(map_name: str, area_a: int, area_b: int, *, release: str | None = None) -> float | None:
    """Shortest path length in units from area_a to area_b over the nav graph, edge
    weight = 3D centroid distance (awpy's find_path, weight="distance"). None when
    either area is unknown or no path exists. Connections can be one-way (drops), so
    d(a, b) need not equal d(b, a). Lazy and LRU-cached: this is the default path."""
    rel = str(release) if release is not None else pinned_release()
    return _nav_distance(map_name, rel, int(area_a), int(area_b))


def _dijkstra(src: int, adj: list[list[tuple[int, float]]], n: int) -> list[float]:
    dist = [math.inf] * n
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def navdist_paths(map_name: str, out_dir: Path | None = None) -> tuple[Path, Path]:
    d = Path(out_dir) if out_dir is not None else paths.CACHE_MAPS_DIR
    return d / f"{map_name}_navdist.npy", d / f"{map_name}_navdist_areas.npy"


def precompute_all_pairs(map_name: str, *, release: str | None = None,
                         out_dir: Path | None = None) -> Path:
    """Optional: the full area x area distance table as uint16 (units, rounded;
    65535 = unreachable), by repeated Dijkstra over `neighbors()`. Row/column i is area
    `<map>_navdist_areas.npy[i]`. Minutes per map in pure Python; the lazy
    `nav_distance` is the default and needs none of this."""
    a = ensure(map_name, release=release)
    ids = [int(i) for i in a.nav.areas["area_id"]]
    index = {aid: i for i, aid in enumerate(ids)}
    adj: list[list[tuple[int, float]]] = []
    for aid in ids:
        c = a.centroid(aid)
        adj.append([(index[n], _dist3(c, a.centroid(n))) for n in a.nav.neighbors(aid) if n in index])
    n = len(ids)
    mat = np.full((n, n), NAVDIST_UNREACHABLE, dtype=np.uint16)
    for i in range(n):
        row = np.asarray(_dijkstra(i, adj, n))
        ok = np.isfinite(row)
        mat[i, ok] = np.minimum(np.rint(row[ok]), NAVDIST_MAX).astype(np.uint16)
    mat_path, ids_path = navdist_paths(map_name, out_dir)
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mat_path, mat)
    np.save(ids_path, np.asarray(ids, dtype=np.uint32))
    return mat_path


def load_all_pairs(map_name: str, out_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """(matrix, area_ids) written by precompute_all_pairs, or None if not computed."""
    mat_path, ids_path = navdist_paths(map_name, out_dir)
    if not (mat_path.is_file() and ids_path.is_file()):
        return None
    return np.load(mat_path), np.load(ids_path)
