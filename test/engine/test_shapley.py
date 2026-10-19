"""
Shapley (P8.2.1, P8.2.2): textbook 3-member values, efficiency to 1e-12, symmetry, null
member 0; sampled within 1e-3 of exact on a 7-member additive v, residual < 1e-3 before
enforcement and 0 (to float addition) after; n_perm convergence; determinism.

Convergence note: Monte-Carlo error scales as 1/sqrt(n_perm), so doubling n_perm cuts
the error by ~0.71 and quadrupling halves it. The test checks both ratios on a random
(non-additive) 7-member game; antithetic pairs are exact on additive and quadratic games,
so those cannot show convergence.

Efficiency note: every permutation's marginals telescope to v(A) − v(∅), so the
permutation estimator is efficient by construction and its raw residual is float
rounding only; the B.2 enforcement step still runs and is tested for exactness.

    .venv\\Scripts\\python.exe test\\engine\\test_shapley.py
"""
import hashlib
import math
import random
import statistics
import sys
from dataclasses import replace
from itertools import combinations

from _harness import check, close, section, summary

from engine import config
from engine.attribution import shapley

CFG = config.load().attribution


def glove(S) -> float:
    """a holds the left glove, b and c a right one each: a pair is worth 1."""
    return 1.0 if "a" in S and ("b" in S or "c" in S) else 0.0


# ── P8.2.1 exact ──────────────────────────────────────────────────────────────

section("exact: textbook values and axioms")
phi = shapley.exact(["a", "b", "c"], glove)
check("glove game: a = 2/3", close(phi["a"], 2 / 3, 1e-12), f"{phi['a']}")
check("glove game: b = c = 1/6", close(phi["b"], 1 / 6, 1e-12) and close(phi["c"], 1 / 6, 1e-12))
check("efficiency Σ φ = v(A) − v(∅) to 1e-12", abs(sum(phi.values()) - 1.0) <= 1e-12)
check("symmetry: b and c have identical marginals -> equal φ", phi["b"] == phi["c"])

w = {"x": 0.3, "y": -0.1, "z": 0.5}
quad = lambda S: sum(w[i] for i in S) ** 2                       # noqa: E731
phi = shapley.exact(list(w), quad)
check("v = (Σ w)^2 -> φ_i = w_i · Σ w (closed form)",
      all(close(phi[i], w[i] * sum(w.values()), 1e-12) for i in w), str(phi))
check("efficiency on the quadratic game", abs(sum(phi.values()) - (quad(set(w)) - quad(set()))) <= 1e-12)

phi = shapley.exact(["a", "b", "c", "d"], glove)
check("null member d (v ignores it) gets exactly 0", phi["d"] == 0.0, f"{phi['d']}")
check("adding a null member leaves the others unchanged", close(phi["a"], 2 / 3, 1e-12))

calls = []
shapley.exact(["a", "b", "c"], lambda S: calls.append(S) or glove(S), key=lambda S: tuple(sorted(S)))
check("each of the 2^n coalitions evaluated once (cache by key)",
      len(calls) == 8 and len(set(calls)) == 8, f"{len(calls)} calls")
check("member order does not matter", shapley.exact(["c", "a", "b"], glove) == shapley.exact(["a", "b", "c"], glove))
check("no members -> {}", shapley.exact([], glove) == {})

# ── P8.2.2 sampled ────────────────────────────────────────────────────────────

section("sampled: 7-member additive game")
ids = [f"m{i}" for i in range(7)]
val = {m: 0.01 * (i + 1) * (-1) ** i for i, m in enumerate(ids)}
add = lambda S: 0.2 + sum(val[m] for m in S)                    # noqa: E731
ex = shapley.exact(ids, add)
raw, resid = shapley.sampled(ids, add, 200, seed=shapley.seed_of("w7"), enforce=False)
check("sampled within 1e-3 of exact", max(abs(raw[m] - ex[m]) for m in ids) <= 1e-3)
check("residual < 1e-3 before enforcement", resid < 1e-3, f"{resid:.2e}")
enf, resid2 = shapley.sampled(ids, add, 200, seed=shapley.seed_of("w7"))
total = add(set(ids)) - add(set())
check("residual after enforcement is 0 (to float addition, ≤ 1e-12)",
      abs(math.fsum(enf.values()) - total) <= 1e-12, f"{math.fsum(enf.values()) - total:.2e}")
check("returned residual is the pre-enforcement magnitude", close(resid, resid2, 1e-15))

section("sampled: enforcement on a non-additive game")
rng = random.Random(7)
table = {}
for r in range(8):
    for c in combinations(ids, r):
        table[frozenset(c)] = rng.uniform(-1, 1) if c else 0.0
game = lambda S: table[frozenset(S)]                             # noqa: E731
ex = shapley.exact(ids, game)
enf, resid = shapley.sampled(ids, game, 40, seed=11)
check("non-additive: permutation marginals telescope, so the raw residual is float noise (≤ 1e-12)",
      resid <= 1e-12, f"{resid:.3e}")
check("non-additive: estimate differs from exact at 40 permutations (it is an estimate)",
      max(abs(enf[m] - ex[m]) for m in ids) > 1e-6)
check("non-additive: Σ φ == v(A) − v(∅) after enforcement (≤ 1e-12)",
      abs(math.fsum(enf.values()) - (game(ids) - game([]))) <= 1e-12)
check("same seed -> identical values", shapley.sampled(ids, game, 40, seed=11) == (enf, resid))
check("member order does not matter (sorted before permuting)",
      shapley.sampled(list(reversed(ids)), game, 40, seed=11) == (enf, resid))
check("different seed -> different estimate", shapley.sampled(ids, game, 40, seed=12)[0] != enf)
perms = shapley.permutations(ids, 10, 3)
check("antithetic pairs: each permutation followed by its reverse",
      len(perms) == 10 and all(perms[k + 1] == perms[k][::-1] for k in range(0, 10, 2)))
z, r0 = shapley.enforce_efficiency({"a": 0.0, "b": 0.0}, 0.4)
check("enforcement with all-zero φ splits the residual equally", z == {"a": 0.2, "b": 0.2} and r0 == 0.4)

section("sampled: n_perm convergence (RMSE vs exact over 40 seeds)")


def rmse(n: int) -> float:
    errs = []
    for sd in range(40):
        est, _ = shapley.sampled(ids, game, n, seed=1000 + sd, enforce=False)
        errs += [(est[m] - ex[m]) ** 2 for m in ids]
    return math.sqrt(statistics.fmean(errs))


e = {n: rmse(n) for n in (50, 100, 200)}
r2, r4 = e[100] / e[50], e[200] / e[50]
print(f"     RMSE: n=50 {e[50]:.4f}  n=100 {e[100]:.4f}  n=200 {e[200]:.4f}  (2x ratio {r2:.3f}, 4x ratio {r4:.3f})")
check("doubling n_perm cuts the error by ~1/sqrt(2) (ratio in [0.55, 0.85])", 0.55 <= r2 <= 0.85, f"{r2:.3f}")
check("quadrupling n_perm halves the error (ratio in [0.35, 0.65])", 0.35 <= r4 <= 0.65, f"{r4:.3f}")

# ── dispatch ──────────────────────────────────────────────────────────────────

section("dispatch on shapley_exact_max")
_, m6, _ = shapley.shapley(ids[:6], game, CFG, 1)
_, m7, _ = shapley.shapley(ids, game, CFG, 1)
check("≤ shapley_exact_max members -> exact", m6 == "exact" and CFG.shapley_exact_max == 6)
check("> shapley_exact_max members -> sampled", m7 == "sampled")
asked = set(shapley.required_coalitions(ids, CFG, 5))
strict = lambda S: table[S] if S in asked else (_ for _ in ()).throw(KeyError(S))   # noqa: E731
try:
    shapley.shapley(ids, strict, CFG, 5)
    ok = True
except KeyError:
    ok = False
check("required_coalitions lists every coalition sampled Shapley evaluates", ok)
small = replace(CFG, shapley_exact_max=2)
check("required_coalitions = 2^n for exact", len(shapley.required_coalitions(ids[:2], small, 0)) == 4)
check("seed_of is stable across runs (sha1, not hash())",
      shapley.seed_of("w1") == int(hashlib.sha1(b"w1").hexdigest()[:16], 16))
check("seed_of differs per key", shapley.seed_of("w1") != shapley.seed_of("w2"))

sys.exit(summary())
