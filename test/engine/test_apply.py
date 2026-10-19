"""
Coalition states (P7.1.9): `apply(all)` == s_post field for field; `apply(empty)` ==
s_pre on every field the members can touch (UNTOUCHABLE fields compared to s_post, the
round clock to s_pre's clock advanced to s_post's time); undo order is reverse tick
order; `edit_hash` is deterministic and distinct per subset; `coalition_states` gives
2^n states; reverse and forward undo orders agree; a stub W that ignores decoys gives a
decoy member zero marginal everywhere (the C1 property at the state level).

    .venv\\Scripts\\python.exe test\\engine\\test_apply.py
"""
import sys
from dataclasses import fields

import numpy as np

from _harness import check, section, summary
from _cf_states import (action, good_win, make_flash_pair, make_kill_pair, make_move_pair,
                        make_plant_pair, make_smoke_pair, make_window, pair_states)

from engine.counterfactual import apply as ap
from engine.counterfactual.apply import UNTOUCHABLE, apply, coalition_states, edit_hash
from engine.counterfactual.interventions import INTERVENTION_FOR_TYPE, InterventionError
from engine.state.schema import ContextToken


def pair_eq(a, b) -> bool:
    return bool(np.array_equal(a.pair, b.pair, equal_nan=True))


def tok_id(u) -> tuple:
    return (u.kind, u.x, u.y, u.z, u.owner_side, u.owner_id, u.radius_units)


def empty_identity(name: str, s_pre, s_post, members) -> None:
    """apply(empty) == s_pre on touchable fields; untouchables == s_post."""
    e, _ = apply(s_pre, s_post, members, set())
    check(f"{name}: apply(empty) validates", not e.validate(), "; ".join(e.validate()[:2]))
    check(f"{name}: players == s_pre", e.players == s_pre.players)
    check(f"{name}: pair == s_pre", pair_eq(e, s_pre))
    check(f"{name}: utility tokens == s_pre's (identity fields)",
          [tok_id(u) for u in e.utility] == [tok_id(u) for u in s_pre.utility])
    post_left = {tok_id(u): u.time_left_s for u in s_post.utility}
    pre_left  = {tok_id(u): u.time_left_s for u in s_pre.utility}
    check(f"{name}: time_left_s of tokens in both = s_post's, re-added = s_pre's",
          all(u.time_left_s == post_left.get(tok_id(u), pre_left[tok_id(u)]) for u in e.utility))
    dt = s_post.ctx.time_into_round_s - s_pre.ctx.time_into_round_s
    bad = []
    for f in (f.name for f in fields(ContextToken)):
        got = getattr(e.ctx, f)
        if f in UNTOUCHABLE["ctx"]:
            want = getattr(s_post.ctx, f)
        elif f == "time_remaining_s":
            want = s_pre.ctx.time_remaining_s - dt if s_pre.ctx.bomb_planted == 0 else s_post.ctx.time_remaining_s
        elif f == "time_since_plant_s" and s_pre.ctx.bomb_planted == 1:
            want = s_post.ctx.time_since_plant_s
        else:
            want = getattr(s_pre.ctx, f)
        if not (got == want or (isinstance(got, float) and isinstance(want, float) and abs(got - want) < 1e-9)):
            bad.append(f"{f}: {got!r} != {want!r}")
    check(f"{name}: ctx == s_pre (touchable), s_post (untouchable), clock advanced", not bad, "; ".join(bad[:3]))
    check(f"{name}: meta and labels == s_post", e.meta == s_post.meta and e.labels == s_post.labels)


def all_identity(name: str, s_pre, s_post, members) -> None:
    a, _ = apply(s_pre, s_post, members, {m.action_id for m in members})
    check(f"{name}: apply(all) == s_post field for field", a == s_post and a is not s_post)


# ── identities on every story ─────────────────────────────────────────────────
section("identities")

stories = {
    "kill":   make_kill_pair(),
    "smoke":  make_smoke_pair(),
    "fade":   make_smoke_pair(fade=True),
    "flash":  make_flash_pair(),
    "plant":  make_plant_pair(),
    "move":   make_move_pair(),
    "kill+carrier": make_kill_pair(101, 201),
}
for name, (s_pre, s_post, act) in stories.items():
    all_identity(name, s_pre, s_post, [act])
    empty_identity(name, s_pre, s_post, [act])

w_pre, w_post, members = make_window()
all_identity("window(5)", w_pre, w_post, members)
empty_identity("window(5)", w_pre, w_post, members)
before = (w_pre.copy(), w_post.copy())
coalition_states(w_pre, w_post, members)
check("apply never mutates s_pre / s_post", before[0] == w_pre and before[1] == w_post)

# ── reverse tick order ────────────────────────────────────────────────────────
section("reverse tick order")

seen: list[int] = []
saved = dict(INTERVENTION_FOR_TYPE)
try:
    for t in ("kill", "flash", "smoke_pop", "decoy_start", "damage"):
        undo, redo = saved[t]

        def spy(state, a, ref, *, geometry=None, _undo=undo):
            seen.append(a.tick)
            return _undo(state, a, ref, geometry=geometry)
        INTERVENTION_FOR_TYPE[t] = (spy, redo)
    ticked = []
    for m, tick in zip(members, (5001, 4999, 5000, 5000, 4999)):
        a = action(m.type, m.actor_id_hash, tick=tick, **m.params)
        ticked.append(a)
    apply(w_pre, w_post, ticked, set())
    check("members undone in reverse tick order", seen == sorted(seen, reverse=True) and len(seen) == 5, f"{seen}")
    seen.clear()
    apply(w_pre, w_post, ticked, set(), order="forward")
    check("order='forward' undoes in tick order", seen == sorted(seen), f"{seen}")
finally:
    INTERVENTION_FOR_TYPE.update(saved)
check("table restored after the spy", INTERVENTION_FOR_TYPE == saved)

rev, _ = apply(w_pre, w_post, members, set())
for order in ("forward", "shuffle:1", "shuffle:2"):
    other, _ = apply(w_pre, w_post, members, set(), order=order)
    check(f"empty coalition: reverse == {order}", other == rev)
for m in members:
    a, _ = apply(w_pre, w_post, members, {m.action_id})
    b, _ = apply(w_pre, w_post, members, {m.action_id}, order="forward")
    check(f"only {m.type} kept: reverse == forward", a == b)

# ── damage by A + kill by B on one victim (LOW-28) ────────────────────────────
section("damage by A, kill by B, one victim (LOW-28)")

dk_pre, dk_post = pair_states({}, {"dead": {202}, "kills": {101: 1}, "damage": {101: 60, 102: 40}})
dk = [action("damage", 102, victim_id=202, damage_total=40.0, armor_damage=0.0),
      action("kill", 101, victim_id=202, victim_health_before=60.0)]
dk[0].tick, dk[0].state_pre_tick = 4999, 4998           # the hit lands one tick before the kill
all_identity("damage+kill", dk_pre, dk_post, dk)
empty_identity("damage+kill", dk_pre, dk_post, dk)
for order in ("reverse", "forward", "shuffle:3"):
    e, _ = apply(dk_pre, dk_post, dk, set(), order=order)
    pa = e.players[e.index_of(102)]
    check(f"apply(empty, {order}): A's damage_round back to s_pre (not left at 40)",
          pa.damage_round == dk_pre.players[dk_pre.index_of(102)].damage_round, f"{pa.damage_round}")
ko, _ = apply(dk_pre, dk_post, dk, {dk[1].action_id})
check("only the kill kept: victim dead, A's damage_round back to s_pre, B keeps the kill",
      ko.players[ko.index_of(202)].alive == 0 and ko.players[ko.index_of(102)].damage_round == 0.0
      and ko.players[ko.index_of(101)].kills_round == 1)
do, _ = apply(dk_pre, dk_post, dk, {dk[0].action_id})
check("only the damage kept: A keeps 40, B back to 0 kills / 0 damage",
      do.players[do.index_of(102)].damage_round == 40.0
      and (do.players[do.index_of(101)].kills_round, do.players[do.index_of(101)].damage_round) == (0, 0.0))


# ── reconciliation and C1 at the state level ──────────────────────────────────
section("reconciliation, null member")

kill, flash, pop, decoy, dmg = members
kf, _ = apply(w_pre, w_post, members, {flash.action_id})
v = kf.players[kf.index_of(202)]
check("flash kept, kill undone: victim alive and blinded by the kept flash",
      v.alive == 1 and v.flash_duration_remaining == 3.0)
kn, _ = apply(w_pre, w_post, members, set())
check("nothing kept: victim alive and not blinded", kn.players[kn.index_of(202)].flash_duration_remaining == 0.0)

states = coalition_states(w_pre, w_post, members)
zero = all(good_win(states[s | {decoy.action_id}]) == good_win(states[s])
           for s in states if decoy.action_id not in s)
check("decoy member has zero marginal in every coalition (W ignores decoys)", zero)
same_state = all((states[s | {decoy.action_id}].players == states[s].players)
                 for s in states if decoy.action_id not in s)
check("decoy far from everyone changes no player field in any coalition", same_state)

# ── edit_hash and coalition_states ────────────────────────────────────────────
section("edit_hash, coalition_states")

ids = [m.action_id for m in members]
h1 = apply(w_pre, w_post, members, {ids[0], ids[2]})[1]
h2 = apply(w_pre, w_post, members, {ids[2], ids[0]})[1]
check("edit_hash deterministic (set order irrelevant)", h1 == h2 == edit_hash(w_post, [ids[2], ids[0]]))
hashes = {apply(w_pre, w_post, members, set(s))[1] for s in ap.all_subsets(members)}
check("edit_hash distinct across all 32 subsets", len(hashes) == 32)
other_post = w_post.copy()
other_post.meta.tick += 1
check("edit_hash depends on s_post.meta", edit_hash(other_post, [ids[0]]) != edit_hash(w_post, [ids[0]]))
check("coalition_states: 2^5 = 32 states", len(states) == 32)
check("coalition_states: every state validates", all(not s.validate() for s in states.values()))
check("coalition_states: keys are frozensets incl. empty and full",
      frozenset() in states and frozenset(ids) in states)
part = coalition_states(w_pre, w_post, members, [set(), {ids[1]}])
check("coalition_states with explicit subsets", list(part) == [frozenset(), frozenset({ids[1]})]
      and part[frozenset()] == states[frozenset()])
check("coalition state equals apply()", states[frozenset({ids[3]})] == apply(w_pre, w_post, members, {ids[3]})[0])
k_pre, k_post, k = make_kill_pair()
check("1 member: 2 coalition states", len(coalition_states(k_pre, k_post, [k])) == 2)

# ── errors ────────────────────────────────────────────────────────────────────
section("errors")

try:
    apply(w_pre, w_post, members, {"0" * 16})
    check("subset id outside the members -> ValueError", False)
except ValueError:
    check("subset id outside the members -> ValueError", True)
try:
    apply(w_pre, w_post, members + [members[0]], set())
    check("duplicate member ids -> ValueError", False)
except ValueError:
    check("duplicate member ids -> ValueError", True)
try:
    apply(k_post, k_post, [k], set())
    check("an undo whose precondition fails -> InterventionError", False)
except InterventionError:
    check("an undo whose precondition fails -> InterventionError", True)

sys.exit(summary())
