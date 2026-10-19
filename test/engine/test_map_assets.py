"""
Map assets (P2.2.1) and nav distances (P2.2.4).

    .venv\\Scripts\\python.exe test\\engine\\test_map_assets.py

Needs the pinned awpy-data release in awpy's cache (`~/.awpy/<release>`); exits 2 if
it is missing. Never downloads: a cache miss must be refused.
"""
import math
import random
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

from engine.maps import assets


def main() -> int:
    section("map name validation (no assets needed)")
    for bad in ("../de_mirage", "de_mirage/../../x", "de_Mirage", "de mirage", "", "C:\\x", None):
        try:
            assets.ensure(bad, release="no-such-release")
            check(f"map name {bad!r} refused before any path is built", False)
        except assets.MapAssetError as exc:
            check(f"map name {bad!r} refused before any path is built", "invalid map name" in str(exc), str(exc))

    release = assets.pinned_release()
    if not assets.release_dir(release).is_dir():
        print(f"awpy-data release {release} not cached under {assets.cache_root()}; "
              f"run `awpy get --version {release}` first")
        return 2

    section("ensure")
    a = assets.ensure("de_mirage")
    check("visibility checker loaded", a.visibility.triangle_count > 0, f"{a.visibility.triangle_count} triangles")
    check("nav mesh loaded", a.nav.area_count > 0, f"{a.nav.area_count} areas")
    check("release is the config pin", a.release == release)
    check("cached per process", assets.ensure("de_mirage") is a)
    check("every active-pool map is cached", set(assets.ACTIVE_POOL) <= set(assets.cached_maps()),
          f"missing {set(assets.ACTIVE_POOL) - set(assets.cached_maps())}")

    section("hashes")
    h1 = assets.asset_hashes("de_mirage")
    assets._CACHE.pop(("de_mirage", release), None)       # recompute, not the cached object (LOW-65)
    h2 = assets.asset_hashes("de_mirage")
    check("hashes stable across calls (second call re-reads the files)",
          h1 == h2 and assets.ensure("de_mirage") is not a)
    check("mesh hash is the file's sha256", h1["de_mirage.mesh"] == assets._sha256(a.mesh_path))
    check("nav hash is the file's sha256", h1["de_mirage.nav"] == assets._sha256(a.nav_path))
    check("hashes record the release", h1["release"] == release)
    check("hashes differ between maps", assets.asset_hashes("de_nuke")["de_nuke.nav"] != h1["de_mirage.nav"])

    section("refusals")
    for desc, kw in (("unknown map refused", {"map_name": "de_notamap"}),
                     ("pin mismatch (release not cached) refused", {"map_name": "de_mirage", "release": "1"})):
        try:
            assets.ensure(**kw)
            check(desc, False)
        except assets.MapAssetError as exc:
            check(desc, True, str(exc)[:90])
    import awpy.data
    saved = awpy.data.AWPY_DATA_DIR
    with tempfile.TemporaryDirectory() as d:
        awpy.data.AWPY_DATA_DIR = Path(d)
        try:
            assets.ensure("de_inferno", release=release + "0")   # fresh cache key, empty cache
            check("empty cache refused without downloading", False)
        except assets.MapAssetError:
            check("empty cache refused without downloading", not any(Path(d).iterdir()))
        finally:
            awpy.data.AWPY_DATA_DIR = saved

    section("nav distances (lazy)")
    ids = [int(i) for i in a.nav.areas["area_id"]]
    rng = random.Random(1337)
    check("d(a, a) = 0", all(a.nav_distance(i, i) == 0.0 for i in rng.sample(ids, 20)))
    check("unknown area -> None", a.nav_distance(ids[0], 10**9) is None)
    bad, judged = [], 0
    for _ in range(200):
        x, y, z = rng.sample(ids, 3)
        dxz, dxy, dyz = a.nav_distance(x, z), a.nav_distance(x, y), a.nav_distance(y, z)
        if None in (dxz, dxy, dyz):
            continue
        judged += 1
        if dxz > dxy + dyz + 1.0:
            bad.append((x, y, z, dxz, dxy + dyz))
    check("triangle inequality on 200 random triples (within 1 u)", not bad and judged >= 150,
          f"{judged} judged, {len(bad)} violations {bad[:2]}")

    section("Dijkstra (precompute path) agrees with the lazy path")
    index = {aid: i for i, aid in enumerate(ids)}
    adj = [[(index[n], math.dist(a.centroid(aid), a.centroid(n))) for n in a.nav.neighbors(aid) if n in index]
           for aid in ids]
    worst = 0.0
    for src in rng.sample(ids, 3):
        row = assets._dijkstra(index[src], adj, len(ids))
        for dst in rng.sample(ids, 30):
            lazy = a.nav_distance(src, dst)
            full = row[index[dst]]
            if lazy is None or not math.isfinite(full):
                worst = max(worst, 0.0 if (lazy is None and not math.isfinite(full)) else math.inf)
            else:
                worst = max(worst, abs(lazy - full))
    check("lazy find_path length = Dijkstra length (within 1 u)", worst <= 1.0, f"max diff {worst:.3f}")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
