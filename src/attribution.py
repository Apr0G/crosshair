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

  instantaneous   the jump across the action, WP(t+1) - WP(t-1)
  durative        the RESIDUAL drift across the action's window: the window's total
                  change minus the instantaneous jumps inside it. This is what lets
                  "was that smoke useful?" have an answer, and it is why rotations
                  (the bulk of the corpus) do not collapse to zero.
  round           not an in-round action; valued against the round's opening state

Correctness condition, checked per round:

    sum(impacts) + drift == wp(round_end) - wp(round_start)

Credit that is double-counted, invented or lost breaks that identity, so it fails a
test rather than passing silently.
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

    @property
    def total_attributed(self) -> float:
        return sum(a.impact for a in self.attributions if a.impact is not None)

    @property
    def residual(self) -> float:
        """Conservation error. Must be ~0 — see the module docstring."""
        return (self.wp_end - self.wp_start) - (self.total_attributed + self.drift)

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

    wp_at(tick) -> float | None      value function on the round's timeline
    wp_of_state(state) -> float      optional; enables exact Shapley
    apply_delta(state, delta)        optional; returns a new state with delta applied
    state_at(tick) -> state          optional; needed alongside the two above
    """
    res = RoundResult()
    p0, p1 = wp_at(round_start), wp_at(round_end)
    if p0 is None or p1 is None:
        return res
    res.wp_start, res.wp_end = float(p0), float(p1)

    instant  = [a for a in actions if a.kind == KIND_INSTANT and a.resolve_tick is not None]
    durative = [a for a in actions if a.kind == KIND_DURATIVE and a.window]
    roundlvl = [a for a in actions if a.kind == KIND_ROUND]

    # ── instantaneous: group by tick so simultaneity is handled explicitly ──────
    by_tick: dict[int, list[Action]] = {}
    for a in instant:
        by_tick.setdefault(int(a.resolve_tick), []).append(a)

    # tick -> total CT-perspective jump, used later to subtract from durative windows
    jump_at: dict[int, float] = {}

    for tick in sorted(by_tick):
        group = by_tick[tick]
        before, after = wp_at(tick - 1), wp_at(tick + 1)
        if before is None or after is None:
            for a in group:
                res.attributions.append(Attribution(a.event_id, None, before, after, "unbracketed"))
            continue
        joint = float(after) - float(before)
        jump_at[tick] = joint

        if len(group) == 1:
            a = group[0]
            res.attributions.append(
                Attribution(a.event_id, _sign(a.side, joint), float(before), float(after), "instant"))
            continue

        shares = _split_simultaneous(group, float(before), float(after),
                                     wp_of_state, apply_delta, state_at, tick)
        for a, share in zip(group, shares):
            res.attributions.append(
                Attribution(a.event_id, _sign(a.side, share), float(before), float(after),
                            "shapley" if len(group) <= MAX_EXACT_SHAPLEY and wp_of_state else "equal-split"))

    # ── durative: residual drift over the window ───────────────────────────────
    # A window's total change includes any instantaneous jumps inside it; those are
    # already credited to their own actions, so only what is left belongs here.
    # Overlapping windows share the same residual, so it is split between them —
    # otherwise the same drift is credited twice and conservation breaks.
    residual_by_window: list[tuple[Action, float]] = []
    for a in durative:
        w0, w1 = int(a.window[0]), int(a.window[1])
        w0, w1 = max(w0, round_start), min(w1, round_end)
        if w1 <= w0:
            res.attributions.append(Attribution(a.event_id, None, None, None, "empty-window"))
            continue
        pw0, pw1 = wp_at(w0), wp_at(w1)
        if pw0 is None or pw1 is None:
            res.attributions.append(Attribution(a.event_id, None, pw0, pw1, "unbracketed"))
            continue
        inner = sum(j for tk, j in jump_at.items() if w0 < tk < w1)
        residual_by_window.append((a, (float(pw1) - float(pw0)) - inner))

    # Split overlapping residuals by how many windows cover each contributing action.
    overlap_count: dict[int, int] = {}
    for a, _ in residual_by_window:
        for b, _ in residual_by_window:
            if _overlaps(a.window, b.window):
                overlap_count[a.event_id] = overlap_count.get(a.event_id, 0) + 1

    for a, resid in residual_by_window:
        n = max(1, overlap_count.get(a.event_id, 1))
        w0, w1 = max(int(a.window[0]), round_start), min(int(a.window[1]), round_end)
        res.attributions.append(
            Attribution(a.event_id, _sign(a.side, resid / n), wp_at(w0), wp_at(w1), "durative"))

    # ── round-level: valued against the round's opening state ──────────────────
    # A buy has no in-round instant; its effect is the state the team starts from.
    for a in roundlvl:
        res.attributions.append(
            Attribution(a.event_id, 0.0, res.wp_start, res.wp_start, "round-level"))

    # ── drift: everything no action explains ───────────────────────────────────
    res.drift = (res.wp_end - res.wp_start) - res.total_attributed
    return res


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
    if alive_t == 0 and not state.get("post_plant"):
        return 1.0                      # no T left and no bomb down: CT wins
    return None                         # not terminal — ask the model
