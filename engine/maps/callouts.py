"""
Position -> callout (P2.3): per-map polygons in `engine/maps/<map>/callouts.json`.

    from engine.maps import assets, callouts
    a  = assets.ensure("de_mirage")
    cs = callouts.load("de_mirage")
    callouts.lookup(a, cs, x, y, z)     # "A Ramp" | "area_123" | None

`lookup` order: the callout polygon containing the point (z within [z_min, z_max]);
else the nearest callout polygon within 128 u (z within 128 u of its range); else
`area_<id>` of the nav area under the point, so every on-mesh position resolves even
while the callout files are empty; off the mesh -> None (never "?").

File format: either a bare list, or `{"note": ..., "callouts": [...]}`, where each
callout is `{"name": str, "polygons": [[[x, y], ...], ...], "z_min": num, "z_max": num}`.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

MAPS_DIR    = Path(__file__).resolve().parent
NEAREST_MAX = 128.0     # u, max 2D distance for the "nearest callout" fallback
Z_TOL       = 128.0     # u, z slack for the nearest fallback (jumps, stairs)
FALLBACK_PREFIX = "area_"

Point   = tuple[float, float]
Polygon = list[Point]


@dataclass(frozen=True)
class Callout:
    name:     str
    polygons: tuple[tuple[Point, ...], ...]
    z_min:    float
    z_max:    float


# ── loading and validation ────────────────────────────────────────────────────

def path_for(map_name: str) -> Path:
    return MAPS_DIR / map_name / "callouts.json"


def _raw_list(doc) -> list:
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict) and isinstance(doc.get("callouts"), list):
        return doc["callouts"]
    raise ValueError("callouts file must be a list or an object with a 'callouts' list")


def from_list(items: list) -> list[Callout]:
    out = []
    for i, c in enumerate(items):
        try:
            polys = tuple(tuple((float(p[0]), float(p[1])) for p in poly) for poly in c["polygons"])
            out.append(Callout(name=str(c["name"]), polygons=polys,
                               z_min=float(c["z_min"]), z_max=float(c["z_max"])))
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise ValueError(f"callout #{i}: malformed ({type(exc).__name__}: {exc})") from exc
    return out


def load(map_name: str, path: Path | None = None) -> list[Callout]:
    """Load and validate a map's callouts; raises ValueError listing every problem."""
    p = Path(path) if path is not None else path_for(map_name)
    if not p.is_file():
        raise FileNotFoundError(f"no callouts file for {map_name}: {p}")
    callouts = from_list(_raw_list(json.loads(p.read_text(encoding="utf-8"))))
    problems = validate(callouts)
    if problems:
        raise ValueError(f"{p}: " + "; ".join(problems))
    return callouts


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Proper or touching intersection of two closed segments."""
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0 not in (d1, d2, d3, d4):
        return True

    def on(a, b, c, d):
        return d == 0 and min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    return on(q1, q2, p1, d1) or on(q1, q2, p2, d2) or on(p1, p2, q1, d3) or on(p1, p2, q2, d4)


def polygon_problems(poly) -> list[str]:
    pts = list(poly)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]                      # an explicitly closed ring is fine
    if len(pts) < 3:
        return [f"has {len(pts)} points (need >= 3)"]
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in pts):
        return ["has a non-finite coordinate"]
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue                    # neighbours share a vertex
            if _segments_cross(*edges[i], *edges[j]):
                return [f"self-intersects (edges {i} and {j})"]
    area2 = sum(_cross((0.0, 0.0), pts[i], pts[(i + 1) % n]) for i in range(n))
    if abs(area2) < 1e-9:
        return ["has zero area"]
    return []


def validate(callouts: list[Callout]) -> list[str]:
    """Every problem found (empty list = valid): >= 3 points, non-self-intersecting,
    non-zero area, z_min <= z_max, names unique and not shaped like the fallback."""
    problems, seen = [], set()
    for c in callouts:
        if not c.name.strip():
            problems.append("a callout has an empty name")
        if c.name in seen:
            problems.append(f"duplicate name {c.name!r}")
        seen.add(c.name)
        if c.name.startswith(FALLBACK_PREFIX):
            problems.append(f"{c.name!r}: names starting with {FALLBACK_PREFIX!r} are reserved for the fallback")
        if not c.z_min <= c.z_max:
            problems.append(f"{c.name!r}: z_min {c.z_min} > z_max {c.z_max}")
        if not c.polygons:
            problems.append(f"{c.name!r}: no polygons")
        for k, poly in enumerate(c.polygons):
            problems += [f"{c.name!r} polygon {k} {msg}" for msg in polygon_problems(poly)]
    return problems


# ── lookup ────────────────────────────────────────────────────────────────────

def point_in_polygon(x: float, y: float, poly) -> bool:
    """Even-odd ray cast; points on an edge count as inside."""
    pts = list(poly)
    inside = False
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        if _segment_distance(x, y, x1, y1, x2, y2) == 0.0:
            return True
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def _segment_distance(px, py, x1, y1, x2, y2) -> float:
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def polygon_distance(x: float, y: float, poly) -> float:
    """0 inside, else the 2D distance to the nearest edge."""
    if point_in_polygon(x, y, poly):
        return 0.0
    pts = list(poly)
    return min(_segment_distance(x, y, *pts[i], *pts[(i + 1) % len(pts)]) for i in range(len(pts)))


def lookup(map_assets, callouts: list[Callout], x: float, y: float, z: float) -> str | None:
    if x is None or y is None or z is None:
        return None
    for c in callouts:
        if c.z_min <= z <= c.z_max and any(point_in_polygon(x, y, p) for p in c.polygons):
            return c.name
    best, best_d = None, NEAREST_MAX
    for c in callouts:
        if not c.z_min - Z_TOL <= z <= c.z_max + Z_TOL:
            continue
        d = min(polygon_distance(x, y, p) for p in c.polygons)
        if d <= best_d:
            best, best_d = c.name, d
    if best is not None:
        return best
    if map_assets is None:
        return None
    area = map_assets.nav.find_area((float(x), float(y), float(z)))
    return None if area is None else f"{FALLBACK_PREFIX}{int(area)}"


def is_fallback(name: str | None) -> bool:
    return name is not None and name.startswith(FALLBACK_PREFIX)
