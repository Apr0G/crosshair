"""
Pairwise 10x10 table (P3.5): los, dist_units, nav_dist_units, smoke_between, same_team.
LOS only for alive enemy pairs; optional per-area LOS cache (`sampling.los_cache`).

    from engine.state import pairwise
    pair = pairwise.table(player_tokens, geometry, utility_tokens, los_cache={})

Shape (10, 10, 5) float32, channel order `schema.PAIR_CHANNELS`, NaN for None, the
diagonal all NaN, every channel symmetric.
- los: geometry visibility between eye points, alive enemy pairs only (walls only:
  smoke is the separate `smoke_between` channel).
- dist_units: 3D distance between positions (dead players use their last position).
- nav_dist_units: geometry nav distance between the players' areas; NaN if unknown.
- smoke_between: 1 if the eye-to-eye segment intersects any active smoke sphere;
  computed for pairs where both players are alive, NaN otherwise.
- same_team: 1/0 for every off-diagonal pair (empty slots have a side too).
The LOS cache is keyed by (area_a, area_b, smoke signature) and lives for one match;
it is approximate (positions move inside an area), so golden files use exact mode.
"""
import math

import numpy as np

from engine.state.schema import N_PLAYERS, PAIR_CHANNELS

# ── constants ─────────────────────────────────────────────────────────────────

EYE_STAND = 64.0     # u, eye height above the origin, standing
EYE_DUCK  = 46.0     # u, eye height above the origin, ducking
LOS, DIST, NAV, SMOKE, SAME = range(len(PAIR_CHANNELS))


def _eye(p) -> tuple[float, float, float] | None:
    if None in (p.x, p.y, p.z):
        return None
    return (p.x, p.y, p.z + (EYE_DUCK if p.is_ducking == 1 else EYE_STAND))


def segment_hits_sphere(a, b, centre, radius: float) -> bool:
    """Does the closed segment a-b intersect the sphere (centre, radius)?"""
    ax, ay, az = a
    dx, dy, dz = b[0] - ax, b[1] - ay, b[2] - az
    fx, fy, fz = ax - centre[0], ay - centre[1], az - centre[2]
    dd = dx * dx + dy * dy + dz * dz
    if dd == 0.0:
        return fx * fx + fy * fy + fz * fz <= radius * radius
    s = -(fx * dx + fy * dy + fz * dz) / dd
    s = min(1.0, max(0.0, s))
    cx, cy, cz = fx + s * dx, fy + s * dy, fz + s * dz
    return cx * cx + cy * cy + cz * cz <= radius * radius


def smoke_signature(utility) -> tuple:
    return tuple(sorted((u.x, u.y, u.z, u.radius_units) for u in utility if u.kind == "smoke"))


def table(players, geometry, utility=(), los_cache: dict | None = None) -> np.ndarray:
    n = len(players)
    if n != N_PLAYERS:
        raise ValueError(f"pairwise.table needs {N_PLAYERS} players, got {n}")
    pr     = np.full((n, n, len(PAIR_CHANNELS)), np.nan, dtype=np.float32)
    eyes   = [_eye(p) for p in players]
    smokes = [((u.x, u.y, u.z), u.radius_units) for u in utility
              if u.kind == "smoke" and u.radius_units is not None and None not in (u.x, u.y, u.z)]
    sig    = smoke_signature(utility) if los_cache is not None else None

    for i in range(n):
        a = players[i]
        for j in range(i + 1, n):
            b = players[j]
            vals = [np.nan] * len(PAIR_CHANNELS)
            vals[SAME] = 1.0 if a.side == b.side else 0.0
            if None not in (a.x, a.y, a.z, b.x, b.y, b.z):
                vals[DIST] = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
            if a.nav_area is not None and b.nav_area is not None:
                nd = geometry.nav_distance(a.nav_area, b.nav_area)
                vals[NAV] = np.nan if nd is None else float(nd)
            both_alive = a.alive == 1 and b.alive == 1
            if both_alive and eyes[i] is not None and eyes[j] is not None:
                vals[SMOKE] = 1.0 if any(segment_hits_sphere(eyes[i], eyes[j], c, r) for c, r in smokes) else 0.0
                if a.side != b.side:
                    vis = None
                    key = None
                    if los_cache is not None and a.nav_area is not None and b.nav_area is not None:
                        key = (min(a.nav_area, b.nav_area), max(a.nav_area, b.nav_area), sig)
                        vis = los_cache.get(key)
                    if vis is None:
                        vis = geometry.is_visible(eyes[i], eyes[j])
                        if key is not None and vis is not None:
                            los_cache[key] = vis
                    vals[LOS] = np.nan if vis is None else float(bool(vis))
            pr[i, j, :] = vals
            pr[j, i, :] = vals
    return pr
