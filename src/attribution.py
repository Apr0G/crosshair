"""
Event-boundary impact attribution.

Replaces the sampling-grid bracketing in score_impact, which evaluates the
win-probability model on a fixed ~1 Hz grid and hands every event inside a
sampling interval the same delta. On the previous corpus that meant 49% of scored
events shared an |impact| value, and 115,022 brackets contained both a CT and a T
actor receiving equal-and-opposite full-window deltas.

The engine is deliberately pure: it takes a win-probability *function* rather than
a model or a database, so its correctness is testable with synthetic paths and no
demo, no trained model and no DB. Production supplies a lookup backed by real
states; tests supply an analytic path.

Three action classes, valued differently because they resolve differently:

  instantaneous   claims [t-1, t+1] — the jump across the action
  durative        claims its window, and receives only what no instantaneous action
                  inside that window already explains. This is what lets "was that
                  smoke useful?" have an answer, and why rotations (the bulk of the
                  corpus) do not collapse to zero.
  round           not an in-round action; currently unvalued (impact None)

Correctness rests on PARTITIONING the timeline. The round is cut at every claim
boundary and each elementary interval is handed to its claimants exactly once, so:

    sum(ct_movement) + drift == wp(round_end) - wp(round_start)

`drift` is accumulated from the intervals nobody claims, INDEPENDENTLY of the
attributed total — so the identity is a real check. An earlier version derived
drift as (total change - attributed), which made the check algebraically
vacuous: any set of numbers passed it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

CT, T = "ct", "t"

KIND_INSTANT  = "instantaneous"
KIND_DURATIVE = "durative"
KIND_ROUND    = "round"

# Beyond this many events on one tick, enumerating every ordering is not worth it;
# fall back to an equal split (which is the Shapley value under exchangeability).
MAX_EXACT_SHAPLEY = 4


@dataclass
class Action:
    """One attributable action. `side` signs the result; `state_delta` is optional
    and only needed for exact Shapley on simultaneous actions."""
    event_id:     int
    kind:         str
    side:         str
    resolve_tick: int | None = None
    window:       tuple[int, int] | None = None
    state_delta:  dict | None = None


@dataclass
class Attribution:
    event_id: int
    impact:   float | None
    p_before: float | None
    p_after:  float | None
    method:   str


@dataclass
class RoundResult:
    attributions: list[Attribution] = field(default_factory=list)
    wp_start:     float = 0.0
    wp_end:       float = 0.0
    drift:        float = 0.0
    # Signed movement accumulated per action BEFORE the CT/T sign is applied. The
    # conservation identity lives in this space: signs are a presentation choice and
    # a T actor's -0.2 still corresponds to +0.2 of CT-perspective movement.
    ct_attributed: float = 0.0

    @property
    def total_attributed(self) -> float:
        return sum(a.impact for a in self.attributions if a.impact is not None)

    @property
    def residual(self) -> float:
        """Conservation error.

        NOT a plug: `drift` is accumulated independently from the elementary
        intervals that no action claims, so this can and does go non-zero if the
        partition ever loses or duplicates movement.
        """
        return (self.wp_end - self.wp_start) - (self.ct_attributed + self.drift)

    def conserves(self, tol: float = 1e-9) -> bool:
        return abs(self.residual) <= tol


def _sign(side: str, delta: float) -> float:
    """CT actors gain when CT win probability rises; T actors are negated. Anything
    else must not be silently signed as T — that inverted every T event in the
    previous corpus."""
    if side == CT:
        return delta
    if side == T:
        return -delta
    return None


def attribute_round(
    actions: list[Action],
    wp_at,
    round_start: int,
    round_end: int,
    *,
    wp_of_state=None,
    apply_delta=None,
    state_at=None,
) -> RoundResult:
    """Decompose a round's win-probability change across its actions.

    Works by PARTITIONING the round's timeline, not by measuring each action in
    isolation. Every action claims an interval — an instantaneous action claims
    [t-1, t+1], a durative one claims its window — and the round is cut at every
    claim boundary. Each elementary interval's movement is then handed to whoever
    covers it, exactly once.

    That partition is what makes the guarantee real. Measuring actions independently
    lets two adjacent kills both claim the tick between them, lets a jump on a
    window's edge be paid to both the kill and the smoke, and makes overlapping
    windows share credit by a global overlap count that over-divides. All three are
    impossible here: an interval has one set of claimants and is distributed once.

    Priority within an interval: instantaneous claimants take it (they are the
    discrete cause); durative ones receive only what no instant explains; anything
    unclaimed becomes drift.

    wp_at(tick) -> float | None      value function on the round's timeline
    wp_of_state / apply_delta / state_at   optional; enable exact Shapley weighting
    """
    res = RoundResult()
    p0, p1 = wp_at(round_start), wp_at(round_end)
    if p0 is None or p1 is None:
        return res
    res.wp_start, res.wp_end = float(p0), float(p1)
    rs, re_ = int(round_start), int(round_end)

    claims: dict[int, tuple[int, int]] = {}   # index in `actions` -> clamped interval
    verdict: dict[int, str] = {}
    for i, a in enumerate(actions):
        if a.kind == KIND_INSTANT:
            if a.resolve_tick is None:
                verdict[i] = "no-resolve-tick"; continue
            tk = int(a.resolve_tick)
            if tk < rs or tk > re_:
                verdict[i] = "outside-round"; continue
            claims[i] = (max(rs, tk - 1), min(re_, tk + 1))
        elif a.kind == KIND_DURATIVE:
            if not a.window:
                verdict[i] = "no-window"; continue
            w0, w1 = max(rs, int(a.window[0])), min(re_, int(a.window[1]))
            if w1 <= w0:
                verdict[i] = "empty-window"; continue
            claims[i] = (w0, w1)
        elif a.kind == KIND_ROUND:
            verdict[i] = "round-level-unvalued"
        else:
            verdict[i] = "unknown-kind"

    # Cut the round at every claim boundary.
    cuts = {rs, re_}
    for lo, hi in claims.values():
        cuts.add(lo); cuts.add(hi)
    edges = sorted(c for c in cuts if rs <= c <= re_)

    ct_share = {i: 0.0 for i in claims}       # CT-perspective movement per action
    how: dict[int, set] = {i: set() for i in claims}   # weighting actually used
    drift = 0.0

    for lo, hi in zip(edges, edges[1:]):
        if hi <= lo:
            continue
        a_wp, b_wp = wp_at(lo), wp_at(hi)
        if a_wp is None or b_wp is None:
            continue
        delta = float(b_wp) - float(a_wp)

        inst = [i for i, (cl, ch) in claims.items()
                if actions[i].kind == KIND_INSTANT and cl <= lo and hi <= ch]
        dur  = [i for i, (cl, ch) in claims.items()
                if actions[i].kind == KIND_DURATIVE and cl <= lo and hi <= ch]
        holders = inst or dur
        if not holders:
            drift += delta
            continue

        weights, how_used = _weights(holders, actions, delta, lo,
                                     wp_of_state, apply_delta, state_at)
        for i, w in zip(holders, weights):
            how[i].add(how_used)
            # An action with no usable side cannot be credited; its movement is
            # drift rather than being silently signed as T.
            if _sign(actions[i].side, 1.0) is None:
                drift += w
            else:
                ct_share[i] += w

    for i, a in enumerate(actions):
        if i in verdict:
            res.attributions.append(Attribution(a.event_id, None, None, None, verdict[i]))
            continue
        lo, hi = claims[i]
        signed = _sign(a.side, ct_share[i])
        base = "instant" if a.kind == KIND_INSTANT else "durative"
        # Report the weighting that actually ran, not the one that was available.
        used = how[i] - {"sole"}
        method = base if not used else f"{base}/{'+'.join(sorted(used))}"
        res.attributions.append(Attribution(a.event_id, signed, wp_at(lo), wp_at(hi), method))

    res.ct_attributed = sum(ct_share[i] for i in claims
                            if _sign(actions[i].side, 1.0) is not None)
    res.drift = drift
    return res


def _weights(holders, actions, delta, lo, wp_of_state, apply_delta, state_at):
    """Split one interval's movement among the actions covering it."""
    n = len(holders)
    if n == 1:
        return [delta], "sole"
    group = [actions[i] for i in holders]
    can = (wp_of_state is not None and apply_delta is not None and state_at is not None
           and n <= MAX_EXACT_SHAPLEY and all(a.state_delta is not None for a in group))
    if not can:
        return [delta / n] * n, "equal-split"
    base = state_at(lo)
    if base is None:
        return [delta / n] * n, "equal-split"

    totals = [0.0] * n
    orders = list(permutations(range(n)))
    for order in orders:
        state, prev = base, wp_of_state(base)
        for idx in order:
            state = apply_delta(state, group[idx].state_delta)
            cur = wp_of_state(state)
            totals[idx] += cur - prev
            prev = cur
    shares = [x / len(orders) for x in totals]

    # Shapley marginals telescope exactly within every ordering, so `shares` already
    # sums to the STATE-space jump. That can differ from this interval's TIMELINE
    # movement, and the two are different estimators. Redistribute the gap ADDITIVELY:
    # multiplicative rescaling flips every sign when the two disagree in direction,
    # and explodes without bound when the marginals nearly cancel.
    gap = delta - sum(shares)
    return [x + gap / n for x in shares], "shapley"


def _overlaps(w1, w2) -> bool:
    return w1 is not None and w2 is not None and w1[0] < w2[1] and w2[0] < w1[1]


def _split_simultaneous(group, before, after, wp_of_state, apply_delta, state_at, tick):
    """Marginal contribution per action for actions resolving on the same tick.

    With state hooks available this is the exact Shapley value: the average marginal
    contribution over every ordering, which removes the arbitrariness of picking one.
    Without them, an equal split — which IS the Shapley value when the actions are
    exchangeable.
    """
    joint = after - before
    n = len(group)
    can_shapley = (wp_of_state is not None and apply_delta is not None
                   and state_at is not None and n <= MAX_EXACT_SHAPLEY
                   and all(a.state_delta is not None for a in group))
    if not can_shapley:
        return [joint / n] * n

    base = state_at(tick - 1)
    if base is None:
        return [joint / n] * n

    totals = {a.event_id: 0.0 for a in group}
    orders = list(permutations(range(n)))
    for order in orders:
        state = base
        prev  = wp_of_state(state)
        for idx in order:
            state = apply_delta(state, group[idx].state_delta)
            cur   = wp_of_state(state)
            totals[group[idx].event_id] += cur - prev
            prev = cur
    shares = [totals[a.event_id] / len(orders) for a in group]

    # Shapley is efficient by construction, but the model is not linear in the
    # state, so the sum of marginals need not equal the observed joint jump.
    # Rescale onto the real jump so the round still conserves.
    s = sum(shares)
    if abs(s) > 1e-12:
        shares = [x * joint / s for x in shares]
    else:
        shares = [joint / n] * n
    return shares


# ── terminal states ───────────────────────────────────────────────────────────

def terminal_wp(state: dict) -> float | None:
    """Win probability of a decided state, as a fact rather than a prediction.

    Post-decision rows are excluded from training (they are trivially separable and
    inflate AUC), so the model has never seen alive_ct == 0 — yet scoring feeds it
    exactly those states to get p_after for the round-deciding kill. Short-circuit
    them instead of asking a model about inputs it was trained never to encounter.
    """
    if state is None:
        return None
    alive_ct, alive_t = state.get("alive_ct"), state.get("alive_t")
    if alive_ct == 0 and alive_t == 0:
        return None
    if alive_ct == 0:
        return 0.0                      # no CT left: CT cannot win
    if alive_t == 0:
        # A MISSING post_plant must not take the same branch as post_plant=0.
        # `not state.get("post_plant")` was True for an absent key and returned a
        # certain CT win from no information at all.
        pp = state.get("post_plant")
        if pp is not None and not pp:
            return 1.0                  # no T left and no bomb down: CT wins
        return None                     # bomb may still be live, or we don't know
    return None                         # not terminal — ask the model
