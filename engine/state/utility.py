"""
Utility tokens (P3.3): active smokes, fires, decoys at a tick (Appendix A.3), capped at
`sampling.max_utility_tokens`, newest kept, cap hits logged.

    from engine.state import utility
    toks, n_smokes, n_fires = utility.tokens(sl, max_tokens=16)

Active means `start_tick <= tick` and the end is not yet known: the causal slice nulls
every end tick that lies in the future, so a token disappears exactly when
`end_tick <= tick` becomes known. `time_left_s` is the nominal lifetime minus the
elapsed time, floored at 0; it never reads the real end tick (that would leak when the
smoke is going to fade, e.g. because the round ends).
Flashes and HE grenades are instantaneous and produce no token.
"""
import logging
import math

from engine.state.players import _float, _nan, player_id_hash
from engine.state.schema import UtilityToken

log = logging.getLogger("engine.state.utility")

# ── constants ─────────────────────────────────────────────────────────────────

NOMINAL_DURATION_S = {"smoke": 18.0, "fire": 7.0, "decoy": 15.0}
SMOKE_RADIUS       = 144.0     # u, smoke sphere radius (P3.3.1)
FIRE_Z_TOL         = 64.0      # u, vertical slack for "standing in the fire"


def _active(rows: list[dict], start_col: str, tick: int) -> list[dict]:
    out = []
    for r in rows:
        start = _nan(r.get(start_col))
        if start is None or start > tick:
            continue
        end = _nan(r.get("end_tick"))
        if end is not None and end <= tick:
            continue
        out.append(r)
    return out


def _token(kind: str, r: dict, start: int, sl, radius) -> UtilityToken:
    thrower = _nan(r.get("thrower_steamid"))
    elapsed = (sl.tick - start) / sl.tick_rate
    return UtilityToken(
        kind=kind, x=_float(r["x"]), y=_float(r["y"]), z=_float(r["z"]),
        time_left_s=max(0.0, NOMINAL_DURATION_S[kind] - elapsed),
        owner_side=sl.sides.get(thrower) if thrower is not None else None,
        owner_id=player_id_hash(thrower) if thrower is not None else None,
        radius_units=_float(radius))


def tokens(sl, max_tokens: int = 16) -> tuple[list[UtilityToken], int, int]:
    """(tokens oldest-first after the cap, active smokes, active fires before the cap)."""
    found: list[tuple[int, int, UtilityToken]] = []
    for r in _active(sl.smokes, "start_tick", sl.tick):
        found.append((int(r["start_tick"]), int(r["entity_id"]),
                      _token("smoke", r, int(r["start_tick"]), sl, SMOKE_RADIUS)))
    for r in _active(sl.infernos, "start_tick", sl.tick):
        found.append((int(r["start_tick"]), int(r["entity_id"]),
                      _token("fire", r, int(r["start_tick"]), sl, r.get("extent_radius"))))
    for r in _active(sl.decoys, "start_tick", sl.tick):
        found.append((int(r["start_tick"]), int(r["entity_id"]),
                      _token("decoy", r, int(r["start_tick"]), sl, None)))
    found.sort(key=lambda f: (f[0], f[1]))
    n_smokes = sum(1 for f in found if f[2].kind == "smoke")
    n_fires  = sum(1 for f in found if f[2].kind == "fire")
    if len(found) > max_tokens:
        log.info(f"utility cap hit at tick {sl.tick}: {len(found)} active, keeping newest {max_tokens}")
        found = found[-max_tokens:]
    return [f[2] for f in found], n_smokes, n_fires


def inside(u: UtilityToken, x: float, y: float, z: float) -> bool:
    """Is the point inside the token's volume? Smoke: 3D sphere; fire: 2D disc with
    FIRE_Z_TOL vertical slack; decoy / unknown radius: never."""
    if u.radius_units is None or None in (u.x, u.y, u.z, x, y, z):
        return False
    if u.kind == "smoke":
        return math.dist((x, y, z), (u.x, u.y, u.z)) <= u.radius_units
    if u.kind == "fire":
        return math.hypot(x - u.x, y - u.y) <= u.radius_units and abs(z - u.z) <= FIRE_Z_TOL
    return False


def inside_flag(utility: list[UtilityToken], kind: str, x: float, y: float, z: float) -> int | None:
    """`in_smoke` / `on_fire` as a tri-state: 1 inside some active token of `kind` with
    a known radius, 0 when every token of that kind is known and misses, None when no
    known token contains the point but one of that kind has an unknown radius (it
    might). awpy 3.1 gives no flame extents, so on a real demo `on_fire` is None
    whenever a fire is burning: unknown is never written as 0 (2026-10-19 audit, HIGH-04)."""
    unknown = False
    for u in utility:
        if u.kind != kind:
            continue
        if inside(u, x, y, z):
            return 1
        if u.radius_units is None:
            unknown = True
    return None if unknown else 0
