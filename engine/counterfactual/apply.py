"""
Coalition states for Shapley (P7.1.9, used by P8.2): `apply(s_pre, s_post, members, S)`
builds `s0 + S` by starting from `s_post` and undoing every window member NOT in `S`,
in reverse tick order, with the undo function of `INTERVENTION_FOR_TYPE`.

    from engine.counterfactual.apply import apply, coalition_states
    s, h = apply(s_pre, s_post, members, {kill.action_id})     # only the kill happened
    states = coalition_states(s_pre, s_post, members)          # all 2^n coalitions

Identities (test/engine/test_apply.py): `apply(all)` equals `s_post` field for field;
`apply(empty)` equals `s_pre` on every field the members can touch. The fields no member
can touch are listed in UNTOUCHABLE and stay at their `s_post` values: the clock
(`time_into_round_s`; `time_remaining_s` is `s_pre`'s clock advanced to `s_post`'s
time), `meta`, `labels`, and `time_left_s` of utility tokens present in both states.

After the undos, one reconciliation runs: a player revived by an undone kill gets the
blind of every kept flash member that blinded it (params duration), so `v({flash})` in a
flash+kill window sees a blinded, alive victim rather than an unflashed one.

Filled by: P7.1.9.
"""
import hashlib
import random
from itertools import combinations
from typing import Iterable

from engine.actions.types import Action
from engine.counterfactual.interventions import INTERVENTION_FOR_TYPE, InterventionError, blinded_players
from engine.state.schema import Snapshot

# ── constants ─────────────────────────────────────────────────────────────────

# fields no window member can edit: compared to s_post by the apply(empty) identity
UNTOUCHABLE = {
    "ctx":     ("time_into_round_s", "map_version", "tier", "loss_streak_ct", "loss_streak_t",
                "score_ct", "score_t", "round_num", "half", "is_overtime"),
    "utility": ("time_left_s",),
    "meta":    "all",
    "labels":  "all",
}
ORDERS = ("reverse", "forward")


def edit_hash(s_post: Snapshot, subset: Iterable[str]) -> str:
    """sha1 over the sorted subset ids and `s_post.meta` (match, round, tick)."""
    m = s_post.meta
    head = f"{m.match_key}|{m.round_num}|{m.tick}" if m is not None else "nometa"
    body = ",".join(sorted(subset))
    return hashlib.sha1(f"{head}|{body}".encode("utf-8")).hexdigest()


def _sort_key(a: Action) -> tuple:
    return (a.tick if a.tick is not None else (a.window_start_tick or 0), a.action_id)


def _ordered(to_undo: list[Action], order: str) -> list[Action]:
    if order == "reverse":
        return sorted(to_undo, key=_sort_key, reverse=True)
    if order == "forward":
        return sorted(to_undo, key=_sort_key)
    if order.startswith("shuffle:"):
        out = sorted(to_undo, key=_sort_key)
        random.Random(order).shuffle(out)
        return out
    raise ValueError(f"order {order!r} must be one of {ORDERS} or 'shuffle:<seed>'")


def _reconcile(state: Snapshot, s_post: Snapshot, kept: list[Action]) -> None:
    """Re-blind revived players with the kept flash members' durations (in place)."""
    for a in kept:
        if a.type != "flash":
            continue
        for pid, dur in blinded_players(a):
            i, j = state.index_of(pid), s_post.index_of(pid)
            if i is None or j is None or dur is None:
                continue
            if state.players[i].alive == 1 and s_post.players[j].alive == 0:
                state.players[i].flash_duration_remaining = dur


def apply(s_pre: Snapshot, s_post: Snapshot, members: list[Action], subset: set[str], *,
          geometry=None, order: str = "reverse") -> tuple[Snapshot, str]:
    """`s0 + S`: `s_post` with every member whose `action_id` is not in `subset` undone
    (reverse tick order; `order="forward"` / `"shuffle:<seed>"` exist for the P7.2.6
    order-independence check). Returns `(state, edit_hash)`; the inputs are never
    mutated. Raises ValueError for ids outside the members and InterventionError when
    an undo's precondition fails or a member type has no intervention."""
    ids = [m.action_id for m in members]
    if len(set(ids)) != len(ids):
        raise ValueError("members have duplicate action_ids")
    subset = set(subset)
    stray = subset - set(ids)
    if stray:
        raise ValueError(f"subset ids not among the members: {sorted(stray)}")
    h = edit_hash(s_post, subset)
    to_undo = [m for m in members if m.action_id not in subset]
    if not to_undo:
        return s_post.copy(), h
    state = s_post
    for m in _ordered(to_undo, order):
        if m.type not in INTERVENTION_FOR_TYPE:
            raise InterventionError(f"no intervention for action type {m.type!r}")
        undo, _redo = INTERVENTION_FOR_TYPE[m.type]
        state = undo(state, m, s_pre, geometry=geometry)
    _reconcile(state, s_post, [m for m in members if m.action_id in subset])
    return state, h


def all_subsets(members: list[Action]) -> list[frozenset]:
    """Every coalition of the members' ids (2^n), smallest first."""
    ids = [m.action_id for m in members]
    return [frozenset(c) for r in range(len(ids) + 1) for c in combinations(ids, r)]


def coalition_states(s_pre: Snapshot, s_post: Snapshot, members: list[Action],
                     subsets: Iterable | None = None, *, geometry=None) -> dict[frozenset, Snapshot]:
    """`{frozenset(S): apply(s_pre, s_post, members, S)}` for every requested subset
    (all 2^n when `subsets` is None), the batch Shapley feeds to one Predictor call."""
    want = all_subsets(members) if subsets is None else [frozenset(s) for s in subsets]
    return {s: apply(s_pre, s_post, members, set(s), geometry=geometry)[0] for s in want}
