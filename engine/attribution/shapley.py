"""
Shapley over coalitions of a window's members (P8.2): exact up to
`attribution.shapley_exact_max`, antithetic permutation sampling above it with
efficiency enforced and the residual recorded.

    from engine.attribution import shapley
    phi = shapley.exact(["a", "b", "c"], v)                     # v: frozenset -> float
    phi, resid = shapley.sampled(ids, v, n_perm=200, seed=shapley.seed_of("w17"))
    phi, method, resid = shapley.shapley(ids, v, cfg.attribution, seed)

`v` is any callable over frozensets of member ids. The engine evaluates every
coalition a call will need up front, in one batched predictor call
(`required_coalitions` gives that list for both methods), and passes a table lookup as
`v`. `exact` memoizes `v` by `key(S)` (the engine passes `edit_hash`), so a value is
never requested twice.

Determinism (P8.6.4): members are sorted before anything is enumerated or permuted,
and the permutations come from `random.Random(seed)` with the seed derived from the
window id / action id by `seed_of` (sha1, not Python's salted `hash`).

Filled by: P8.2.1, P8.2.2.
"""
import hashlib
import math
import random
from itertools import combinations
from typing import Callable, Hashable, Iterable

# ── helpers ───────────────────────────────────────────────────────────────────

def seed_of(key: str) -> int:
    """A stable 64-bit seed from a string (window_id, action_id, interval key)."""
    return int(hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:16], 16)


def _ids(members: Iterable[Hashable]) -> list:
    ids = sorted(members)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate member ids")
    return ids


def _memo(v: Callable, key: Callable | None) -> Callable:
    cache: dict = {}

    def get(S: frozenset) -> float:
        k = key(S) if key is not None else S
        if k not in cache:
            cache[k] = float(v(S))
        return cache[k]
    return get


def enforce_efficiency(phi: dict, total: float) -> tuple[dict, float]:
    """Distribute `total − Σ φ` over the members ∝ |φ| (equally when every φ is 0);
    the last member (sorted id) takes the floating remainder so Σ φ == total as
    closely as float addition allows. Returns (new φ, |residual| before)."""
    ids = sorted(phi)
    if not ids:
        return {}, abs(total)
    r = total - math.fsum(phi.values())
    mag = math.fsum(abs(phi[a]) for a in ids)
    out = {}
    for a in ids:
        w = abs(phi[a]) / mag if mag > 0 else 1.0 / len(ids)
        out[a] = phi[a] + r * w
    last = ids[-1]
    out[last] = total - math.fsum(out[a] for a in ids[:-1])
    return out, abs(r)


# ── P8.2.1 exact ──────────────────────────────────────────────────────────────

def all_coalitions(ids: list) -> list[frozenset]:
    return [frozenset(c) for r in range(len(ids) + 1) for c in combinations(ids, r)]


def exact(members, v: Callable, *, key: Callable | None = None) -> dict:
    """φ_a = Σ_{S ⊆ A\\{a}} |S|!(n−|S|−1)!/n! · [v(S ∪ {a}) − v(S)] for every member,
    each of the 2^n coalitions evaluated once."""
    ids = _ids(members)
    n = len(ids)
    if n == 0:
        return {}
    get = _memo(v, key)
    weight = [math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)]
    phi = {}
    for a in ids:
        rest = [b for b in ids if b != a]
        terms = []
        for r in range(n):
            for c in combinations(rest, r):
                S = frozenset(c)
                terms.append(weight[r] * (get(S | {a}) - get(S)))
        phi[a] = math.fsum(terms)
    return phi


# ── P8.2.2 sampled ────────────────────────────────────────────────────────────

def permutations(members, n_perm: int, seed: int) -> list[list]:
    """`n_perm` permutations (rounded up to even) as antithetic pairs: each random
    permutation followed by its reverse."""
    ids = _ids(members)
    rng = random.Random(seed)
    out = []
    for _ in range(max(1, (int(n_perm) + 1) // 2)):
        p = ids[:]
        rng.shuffle(p)
        out += [p, p[::-1]]
    return out


def sampled(members, v: Callable, n_perm: int, seed: int, *, enforce: bool = True,
            key: Callable | None = None) -> tuple[dict, float]:
    """Mean marginal contribution over antithetic permutation pairs; with `enforce`
    the residual `v(A) − v(∅) − Σ φ` is redistributed ∝ |φ| (B.2). Returns
    (φ, |residual| before enforcement)."""
    ids = _ids(members)
    if not ids:
        return {}, 0.0
    get = _memo(v, key)
    perms = permutations(ids, n_perm, seed)
    acc: dict = {a: [] for a in ids}
    for p in perms:
        S = frozenset()
        prev = get(S)
        for a in p:
            S = S | {a}
            cur = get(S)
            acc[a].append(cur - prev)
            prev = cur
    phi = {a: math.fsum(acc[a]) / len(perms) for a in ids}
    total = get(frozenset(ids)) - get(frozenset())
    resid = abs(total - math.fsum(phi.values()))
    if enforce:
        phi, resid = enforce_efficiency(phi, total)
    return phi, resid


# ── dispatch ──────────────────────────────────────────────────────────────────

def method_for(n_members: int, cfg_attr) -> str:
    return "exact" if n_members <= cfg_attr.shapley_exact_max else "sampled"


def required_coalitions(members, cfg_attr, seed: int) -> list[frozenset]:
    """Every coalition `shapley(members, v, cfg_attr, seed)` will evaluate, so the
    caller can predict them in one batch."""
    ids = _ids(members)
    if method_for(len(ids), cfg_attr) == "exact":
        return all_coalitions(ids)
    seen, out = set(), []
    for p in permutations(ids, cfg_attr.shapley_samples, seed):
        S = frozenset()
        for c in [S] + [frozenset(p[:k + 1]) for k in range(len(p))]:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def shapley(members, v: Callable, cfg_attr, seed: int, *,
            key: Callable | None = None) -> tuple[dict, str, float]:
    """Exact when `len(members) <= shapley_exact_max`, else sampled with
    `shapley_samples` permutations. Returns (φ, method, |residual|)."""
    ids = _ids(members)
    method = method_for(len(ids), cfg_attr)
    if method == "exact":
        return exact(ids, v, key=key), method, 0.0
    phi, resid = sampled(ids, v, cfg_attr.shapley_samples, seed, key=key)
    return phi, method, resid
