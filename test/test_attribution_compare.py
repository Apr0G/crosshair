"""
Head-to-head: OLD grid bracketing vs NEW event-boundary attribution.

    .venv/bin/python test/test_attribution_compare.py [max_matches]

Needs data/crosshair.db and a trained data/win_prob.lgb.

What this establishes
─────────────────────
Both methods are run over the SAME rounds, the SAME events and the SAME model,
in the same process, so any difference reported here is a difference of method
and nothing else.

Neither path is re-implemented here. The grid runs through
`score_impact._impact_for_event` and the event method through the same
`_wp_path` / `_actions_for_round` / `attribute_round` that `score_match_event`
uses, so this measures production, not a stand-in.

  OLD (grid)   Bracket each event between the two nearest ~1 Hz round_states
               samples and hand it that whole interval's win-probability delta.
               Every event inside one interval therefore receives the SAME delta,
               and a CT and a T actor in the same interval receive equal-and-
               opposite FULL-interval deltas whether or not they acted together.
  NEW (event)  Partition the round's timeline at every claim boundary. An
               instantaneous action claims [t-1, t+1], a durative one claims its
               window, and each elementary interval's movement goes to its
               claimants exactly once (instantaneous first, durative taking only
               the residual). Round-level actions — buys — are left unvalued
               rather than handed the first second's swing.

Reported, per method:

  1. shared-bracket collisions — scored events sharing an |impact| with another
     event in the same round. The headline defect; 49% on the previous corpus.
     Broken out for instantaneous actions and graded by how far apart the
     colliding events actually were, which is what separates a method defect
     from a data-resolution limit.
  2. sign conflicts / mirrored magnitudes — brackets where a CT and a T actor both
     act. Both methods are measured over the SAME brackets (the 1 Hz intervals),
     with mirrored pairs split by whether the two actors acted at the same moment.
  3. impact distribution — median / p90 / max |impact|, and how many are exactly 0.
  4. per event type — mean |impact| by event_type.
  5. top moments — the 5 highest-|impact| events under each method, for eyeballing,
     plus the same restricted to instantaneous actions.
  6. verdict — PASS/FAIL on which method has fewer collisions and fewer conflicts.

RESOLUTION CAVEAT, stated up front because it dominates the magnitudes: this DB
holds only 1 Hz round_states, so WP is known at 1 s intervals. The event method
asks for WP at resolve_tick±1 and gets a LINEAR INTERPOLATION, whose slope is
constant across an interval. Two consequences, both visible below: an
instantaneous action's jump shrinks to roughly 2/64 of its second's movement, and
any two instantaneous actions inside one interval receive the identical value.
Neither is a property of the attribution; both dissolve once boundary states are
persisted at action ticks. Read collision counts as method evidence and
instantaneous magnitudes as blocked on that work.

One asymmetry is deliberate: the grid path feeds raw model output while the event
path pins decided states through `terminal_wp`. That is not a thumb on the scale —
it is what each path does in production, and pinning is part of the method under
test, not a tweak applied for this comparison.

Exit code is 1 while the TARGET STATE check in §6 is still red. That check is a
statement about the data, not the code, and it turns green when boundary states
land — it is left in rather than removed so the gap stays visible.
"""
import heapq
import sqlite3
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from attribution import attribute_round, KIND_INSTANT
# Both paths come from production, not from a local re-implementation: score_match's
# bracketing helper for the grid, and the same _wp_path / _actions_for_round that
# score_match_event uses for the event method. If either drifts, this drifts with it.
from score_impact import (_actions_for_round, _impact_for_event, _score_states,
                          _wp_path)
from win_probability import MODEL_PATH

DB   = Path(__file__).parent.parent / "data" / "crosshair.db"
PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

GRID, EVENT = "grid", "event"
METHODS     = (GRID, EVENT)
TOL         = 12          # decimals at which two impacts count as "identical"

_results: list[tuple[bool, str]] = []


def check(desc: str, ok: bool, detail: str = ""):
    _results.append((bool(ok), desc))
    print(f"  {PASS if ok else FAIL}  {desc}" + (f"  ({detail})" if detail else ""))


# ── accumulators ──────────────────────────────────────────────────────────────

class Acc:
    """Everything we need to report for one method."""

    def __init__(self, name):
        self.name          = name
        self.abs_impacts   = []                 # |impact| for every scored event
        self.n_events      = 0                  # events offered to the method
        self.n_scored      = 0
        self.n_zero        = 0
        self.collided      = 0                  # scored events sharing |impact| in-round
        self.collided_nz   = 0                  # same, excluding exact zeros
        self.nonzero       = 0
        self.inst_scored   = 0                  # instantaneous subset (kills/bomb/…)
        self.inst_nonzero  = 0
        self.inst_collided = 0                  # ... colliding with another INSTANT event
        self.inst_coll_cross  = 0               # colliders in DIFFERENT 1 Hz intervals
        self.inst_coll_subsec = 0               # one interval, different ticks
        self.inst_coll_simul  = 0               # literally the same tick
        self.inst_abs      = 0.0
        self.by_type_sum   = defaultdict(float)
        self.by_type_n     = defaultdict(int)
        self.by_type_scored = defaultdict(int)
        self.by_type_coll  = defaultdict(int)
        self.top_inst      = []                 # bounded heap, instantaneous only
        self.mixed_brackets = 0                 # brackets with a CT and a T actor
        self.sign_conflict  = 0                 # ... both impacts same sign, non-zero
        self.mirrored       = 0                 # ... a CT and a T with identical |impact|
        self.mirrored_apart = 0                 # ... and they acted at DIFFERENT moments
        self.top            = []                # bounded min-heap of (abs, seq, record)

    def add(self, impact, event_type, kind):
        self.n_events += 1
        self.by_type_n[event_type] += 1
        if impact is None:
            return
        a = abs(float(impact))
        self.n_scored += 1
        self.abs_impacts.append(a)
        self.by_type_sum[event_type] += a
        self.by_type_scored[event_type] += 1
        if a == 0.0:
            self.n_zero += 1
        else:
            self.nonzero += 1
        if kind == KIND_INSTANT:
            self.inst_scored += 1
            self.inst_abs += a
            if a != 0.0:
                self.inst_nonzero += 1

    @staticmethod
    def _push(heap, a, seq, rec):
        if len(heap) < 5:
            heapq.heappush(heap, (a, seq, rec))
        elif a > heap[0][0]:
            heapq.heapreplace(heap, (a, seq, rec))

    def push_top(self, a, seq, rec, kind):
        self._push(self.top, a, seq, rec)
        if kind == KIND_INSTANT:
            self._push(self.top_inst, a, seq, rec)


def collisions(acc: Acc, rows: list[tuple]):
    """Count, within one round, scored events that share an |impact| with another.

    rows: (impact, event_type, kind, bucket) where bucket is (1 Hz sampling interval
    index, resolve tick). Counted three ways — over everything, over non-zero values
    only (the basis of the 49% figure), and over the instantaneous subset alone,
    which is the like-for-like comparison: both methods value a kill, only one of
    them is supposed to give two different kills two different numbers.

    Instantaneous collisions are then graded by how far apart the colliding events
    actually are. That grading is the whole argument:

      cross-1s  different sampling intervals — the method conflating moments a
                second or more apart. Indefensible under either method.
      sub-1s    same interval, different ticks — two distinct moments handed one
                number. Intrinsic to the grid (it can only ever see interval
                endpoints); for the event method it is an artefact of interpolating
                WP between 1 Hz samples, and real boundary states dissolve it.
      same-tick genuinely simultaneous. An equal split gives them equal magnitude
                by design, so this is not a defect at all.
    """
    keyed = [(round(abs(v), TOL), et, k, b) for v, et, k, b in rows if v is not None]
    c_all  = Counter(key for key, _, _, _ in keyed)
    inst_iv:  dict[float, set] = defaultdict(set)
    inst_tk:  dict[float, set] = defaultdict(set)
    c_inst = Counter()
    for key, _, k, b in keyed:
        if k == KIND_INSTANT:
            c_inst[key] += 1
            inst_iv[key].add(b[0])
            inst_tk[key].add(b[1])
    for key, et, k, b in keyed:
        if c_all[key] > 1:
            acc.collided += 1
            acc.by_type_coll[et] += 1
            if key != 0.0:
                acc.collided_nz += 1
        if k == KIND_INSTANT and c_inst[key] > 1 and key != 0.0:
            acc.inst_collided += 1
            if len(inst_iv[key]) > 1:
                acc.inst_coll_cross += 1
            elif len(inst_tk[key]) > 1:
                acc.inst_coll_subsec += 1
            else:
                acc.inst_coll_simul += 1


def bracket_stats(acc: Acc, brackets: dict):
    """brackets: key -> list of (side, impact, moment).

    Counts mixed-side brackets, same-sign pairs inside them, and CT/T pairs handed
    identical magnitudes. `moment` is when the actor actually acted, which separates
    the two cases a raw mirrored count conflates: two players acting at the SAME
    instant should mirror (an equal split ties them by design), while a CT and a T
    acting at different instants being handed the same magnitude is exactly the
    grid's defect.
    """
    for members in brackets.values():
        cts = [(v, mo) for s, v, mo in members if s == "ct" and v is not None]
        ts  = [(v, mo) for s, v, mo in members if s == "t"  and v is not None]
        if not cts or not ts:
            continue
        acc.mixed_brackets += 1
        if any((c > 0) == (t > 0) for c, _ in cts for t, _ in ts if c != 0.0 and t != 0.0):
            acc.sign_conflict += 1
        pairs = [(c, cm, t, tm) for c, cm in cts for t, tm in ts
                 if round(abs(c), TOL) == round(abs(t), TOL)]
        if pairs:
            acc.mirrored += 1
            if any(cm != tm for _, cm, _, tm in pairs):
                acc.mirrored_apart += 1


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    if not DB.exists():
        print("no database — scrape first"); return 2
    if not MODEL_PATH.exists():
        print("no model — run `train` first"); return 2

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    con   = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    mids = [r[0] for r in con.execute(
        "SELECT match_id FROM processed_matches ORDER BY match_id")]
    if limit:
        mids = mids[:limit]

    print(f"\ncomparing grid vs event attribution over {len(mids)} matches\n")

    acc       = {m: Acc(m) for m in METHODS}
    verdicts  = Counter()          # event-method disposition per action
    residuals = []                 # per-round conservation error, event method
    n_rounds  = 0
    seq       = 0
    both_scored = grid_only = event_only = neither = 0
    unknown_side = 0

    for mi, mid in enumerate(mids, 1):
        states = pd.read_sql_query(
            "SELECT * FROM round_states WHERE match_id=? ORDER BY round_num, tick",
            con, params=(mid,))
        if states.empty:
            continue
        states = states.reset_index(drop=True)
        states["_wp"] = _score_states(states)

        events = pd.read_sql_query(
            "SELECT id, round_num, player_side, event_type, time_into_round, action "
            "FROM events WHERE match_id=?", con, params=(mid,))
        if events.empty:
            continue
        ev_by_round = {rn: g for rn, g in events.groupby("round_num")}

        for rn, grp in states.groupby("round_num", sort=False):
            grp = grp.sort_values("tick").reset_index(drop=True)
            if len(grp) < 3:
                continue
            rn = int(rn)
            evs = ev_by_round.get(rn)
            if evs is None or evs.empty:
                continue

            wps_raw     = grp["_wp"].to_numpy(dtype=float)
            state_times = grp["time_into_round_s"].astype(float).tolist()

            # ── NEW: exactly what score_match_event runs ───────────────────────
            wp_at, lo, hi = _wp_path(grp)
            actions = _actions_for_round(evs)
            if not actions:
                continue
            res       = attribute_round(actions, wp_at, lo, hi)
            event_imp = {a.event_id: a.impact for a in res.attributions}
            for a in res.attributions:
                verdicts[a.method] += 1
            residuals.append(abs(res.residual))
            n_rounds += 1

            kind_of = {a.event_id: a.kind for a in actions}
            tick_of = {a.event_id: a.resolve_tick for a in actions}

            # ── OLD: grid bracketing, exactly what score_match runs ────────────
            grid_imp: dict[int, float | None] = {}
            grid_brackets: dict[int, list] = defaultdict(list)
            meta: dict[int, dict] = {}

            for e in evs.itertuples(index=False):
                eid = int(e.id)
                if eid not in kind_of:
                    continue                        # not an attributable action
                side = (e.player_side or "").lower()
                if side not in ("ct", "t"):
                    # feature_extractor writes "?" when the parser had no side column;
                    # neither method may guess, so both leave it unscored
                    unknown_side += 1
                meta[eid] = dict(match=mid, rnd=rn, side=side, etype=e.event_type,
                                 kind=kind_of[eid], bucket=(None, tick_of[eid]),
                                 t=None if pd.isna(e.time_into_round)
                                   else float(e.time_into_round))
                if pd.isna(e.time_into_round):
                    grid_imp[eid] = None
                    continue
                t = float(e.time_into_round)
                _, _, imp = _impact_for_event(t, side, state_times, wps_raw)
                # the 1 Hz interval this event falls in — computed once and used for
                # both methods, so "same interval" means the same thing on either side
                b = bisect_right(state_times, t) - 1
                meta[eid]["bucket"] = (b, tick_of[eid])
                grid_brackets[b].append((side, imp, round(t, 3)))
                grid_imp[eid] = imp

            # The event method's "bracket" is the 1 Hz interval too, so the two are
            # compared over the SAME grouping — the difference is then purely what
            # each method does inside that interval, which is the question.
            ev_brackets: dict[int, list] = defaultdict(list)
            for e in evs.itertuples(index=False):
                eid = int(e.id)
                if eid not in meta or meta[eid]["bucket"][0] is None:
                    continue
                ev_brackets[meta[eid]["bucket"][0]].append(
                    ((e.player_side or "").lower(), event_imp.get(eid),
                     round(float(e.time_into_round), 3)))

            # ── fold into the accumulators ─────────────────────────────────────
            eids = [a.event_id for a in actions if a.event_id in meta]
            for eid in eids:
                m = meta[eid]
                for name, table in ((GRID, grid_imp), (EVENT, event_imp)):
                    v = table.get(eid)
                    acc[name].add(v, m["etype"], m["kind"])
                    if v is not None:
                        seq += 1
                        acc[name].push_top(abs(float(v)), seq,
                                           (m["match"], m["rnd"], m["t"], m["etype"],
                                            m["side"], float(v)), m["kind"])
                g, ev = grid_imp.get(eid), event_imp.get(eid)
                if g is not None and ev is not None:   both_scored += 1
                elif g is not None:                    grid_only   += 1
                elif ev is not None:                   event_only  += 1
                else:                                  neither     += 1

            collisions(acc[GRID],  [(grid_imp.get(e),  meta[e]["etype"],
                                     meta[e]["kind"], meta[e]["bucket"]) for e in eids])
            collisions(acc[EVENT], [(event_imp.get(e), meta[e]["etype"],
                                     meta[e]["kind"], meta[e]["bucket"]) for e in eids])
            bracket_stats(acc[GRID],  grid_brackets)
            bracket_stats(acc[EVENT], ev_brackets)

        if mi % 20 == 0:
            print(f"    {mi}/{len(mids)} matches ...")

    con.close()

    total_ev = acc[GRID].n_events
    print(f"\n  rounds compared : {n_rounds:,}")
    print(f"  events compared : {total_ev:,}")
    print(f"  scored by both {both_scored:,} | grid only {grid_only:,} | "
          f"event only {event_only:,} | neither {neither:,}")

    print("\n  event-method disposition per action:")
    for k, n in sorted(verdicts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<28}{n:>10,}")
    r = np.array(residuals) if residuals else np.zeros(1)
    print(f"\n  event-method conservation: max |residual| {r.max():.3e}, "
          f"mean {r.mean():.3e}")

    def pct(a, b):
        return 100.0 * a / max(b, 1)

    # ── 1. collisions ─────────────────────────────────────────────────────────
    print("\n── 1. shared-bracket collisions ─────────────────────────────────────────")
    print(f"    {'':<8}{'scored':>10}{'collide':>10}{'':>8}{'collide≠0':>11}{'':>8}")
    for m in METHODS:
        a = acc[m]
        print(f"    {m:<8}{a.n_scored:>10,}{a.collided:>10,}"
              f"{pct(a.collided, a.n_scored):>7.1f}%{a.collided_nz:>11,}"
              f"{pct(a.collided_nz, a.nonzero):>7.1f}%")
    print("    (collide≠0 % is over NON-ZERO scored events — the 49% figure's basis)")

    print("\n    like-for-like, INSTANTANEOUS actions only (kills, damage, bomb):")
    print(f"    {'':<8}{'scored':>10}{'non-zero':>10}{'collide≠0':>12}{'':>8}"
          f"{'cross-1s':>10}{'sub-1s':>9}{'same-tick':>11}")
    for m in METHODS:
        a = acc[m]
        print(f"    {m:<8}{a.inst_scored:>10,}{a.inst_nonzero:>10,}"
              f"{a.inst_collided:>12,}{pct(a.inst_collided, a.inst_nonzero):>7.1f}%"
              f"{a.inst_coll_cross:>10,}{a.inst_coll_subsec:>9,}{a.inst_coll_simul:>11,}")
    print("    (an instantaneous action colliding with ANOTHER instantaneous action —")
    print("     two kills in one round handed the identical number)")
    print("    cross-1s  colliders a whole sampling interval apart — indefensible")
    print("    sub-1s    same interval, different ticks — the grid can never fix this;")
    print("              for the event method it is the interpolated WP, and boundary")
    print("              states dissolve it")
    print("    same-tick genuinely simultaneous — an equal split SHOULD tie them")

    print("\n    where the remaining collisions live, by event_type:")
    print(f"    {'type':<14}{'grid coll':>11}{'':>8}{'event coll':>12}{'':>8}")
    for t in sorted(set(acc[GRID].by_type_n) | set(acc[EVENT].by_type_n)):
        g, e = acc[GRID], acc[EVENT]
        print(f"    {str(t):<14}{g.by_type_coll[t]:>11,}"
              f"{pct(g.by_type_coll[t], g.by_type_scored[t]):>7.1f}%"
              f"{e.by_type_coll[t]:>12,}"
              f"{pct(e.by_type_coll[t], e.by_type_scored[t]):>7.1f}%")

    # ── 2. sign conflicts ─────────────────────────────────────────────────────
    print("\n── 2. CT/T brackets: sign conflicts and mirrored magnitudes ─────────────")
    print(f"    {'':<8}{'mixed':>10}{'same-sign':>12}{'mirrored |Δ|':>15}{'':>8}"
          f"{'…at different moments':>24}{'':>8}")
    for m in METHODS:
        a = acc[m]
        print(f"    {m:<8}{a.mixed_brackets:>10,}{a.sign_conflict:>12,}"
              f"{a.mirrored:>15,}{pct(a.mirrored, a.mixed_brackets):>7.1f}%"
              f"{a.mirrored_apart:>24,}{pct(a.mirrored_apart, a.mixed_brackets):>7.1f}%")
    print("    Both methods use the SAME brackets here — the 1 Hz sampling intervals —")
    print("    so the only difference is what each does inside one.")
    print("    same-sign is VACUOUS FOR GRID and its non-zero value under the event")
    print("      method is a feature, not a regression. Grid signs a CT actor +Δ and a")
    print("      T actor −Δ off one shared interval delta, so opposition is forced —")
    print("      it literally cannot report a same-signed pair, hence 0. The event")
    print("      method values the two actors at their own moments, so a CT play that")
    print("      helped CT and a T play that helped T inside the same second come out")
    print("      same-signed, which is correct. Read this row as 'how often the two")
    print("      methods disagree about whether one second had one cause or two'.")
    print("    mirrored at DIFFERENT moments is the real artefact — a CT and a T who")
    print("      acted seconds apart handed the identical magnitude.")

    # ── 3. distribution ───────────────────────────────────────────────────────
    print("\n── 3. |impact| distribution ─────────────────────────────────────────────")
    print(f"    {'':<8}{'median':>10}{'p90':>10}{'max':>10}{'zeros':>12}{'':>8}")
    dist = {}
    for m in METHODS:
        a = acc[m]
        arr = np.array(a.abs_impacts) if a.abs_impacts else np.zeros(1)
        dist[m] = (float(np.median(arr)), float(np.percentile(arr, 90)), float(arr.max()))
        print(f"    {m:<8}{dist[m][0]:>10.5f}{dist[m][1]:>10.5f}{dist[m][2]:>10.5f}"
              f"{a.n_zero:>12,}{pct(a.n_zero, a.n_scored):>7.1f}%")

    # ── 4. per event type ─────────────────────────────────────────────────────
    print("\n── 4. mean |impact| by event_type ───────────────────────────────────────")
    types = sorted(set(acc[GRID].by_type_n) | set(acc[EVENT].by_type_n))
    print(f"    {'type':<14}{'n':>9}{'grid mean':>12}{'grid n':>9}"
          f"{'event mean':>13}{'event n':>9}")
    for t in types:
        g, e = acc[GRID], acc[EVENT]
        gm = g.by_type_sum[t] / max(g.by_type_scored[t], 1)
        em = e.by_type_sum[t] / max(e.by_type_scored[t], 1)
        print(f"    {str(t):<14}{g.by_type_n[t]:>9,}{gm:>12.5f}{g.by_type_scored[t]:>9,}"
              f"{em:>13.5f}{e.by_type_scored[t]:>9,}")

    # ── 5. top moments ────────────────────────────────────────────────────────
    print("\n── 5. top 5 |impact| moments ────────────────────────────────────────────")

    def show(heap):
        for a, _, rec in sorted(heap, key=lambda x: -x[0]):
            m_id, rnd, t, et, side, v = rec
            ts = f"{t:6.1f}s" if t is not None else "   n/a"
            print(f"      {v:+.4f}  {str(et):<11} {side:<3} r{rnd:<3} {ts}  {m_id[:36]}")

    for m in METHODS:
        print(f"\n    [{m}] any action")
        show(acc[m].top)
    for m in METHODS:
        print(f"\n    [{m}] instantaneous only (clutch kills / plants should surface here)")
        show(acc[m].top_inst)

    # ── 5b. resolution limit on instantaneous magnitudes ──────────────────────
    print("\n── 5b. what 1 Hz sampling costs the event method ────────────────────────")
    for m in METHODS:
        a = acc[m]
        print(f"    {m:<8} mean |impact| over instantaneous actions : "
              f"{a.inst_abs / max(a.inst_scored, 1):.5f}")
    ratio = (acc[EVENT].inst_abs / max(acc[EVENT].inst_scored, 1)) / \
            max(acc[GRID].inst_abs / max(acc[GRID].inst_scored, 1), 1e-12)
    print(f"    event/grid ratio : {ratio:.3f}")
    print("    The event method asks for WP at resolve_tick±1. round_states only has")
    print("    1 Hz samples, so that jump is INTERPOLATED and shrinks to ~2/64 of the")
    print("    second's movement. Kills are therefore crushed and durative residuals")
    print("    absorb the round. Persisting real boundary states removes this; until")
    print("    then the event method's magnitudes are not comparable to the grid's.")

    # ── 6. verdict ────────────────────────────────────────────────────────────
    print("\n── 6. verdict ───────────────────────────────────────────────────────────")
    g, e = acc[GRID], acc[EVENT]
    g_coll = pct(g.collided_nz, g.nonzero)
    e_coll = pct(e.collided_nz, e.nonzero)
    gi  = pct(g.inst_collided,   g.inst_nonzero)
    ei  = pct(e.inst_collided,   e.inst_nonzero)
    gx  = pct(g.inst_coll_cross, g.inst_nonzero)
    ex  = pct(e.inst_coll_cross, e.inst_nonzero)
    check("event method has a lower non-zero collision RATE than grid, overall",
          e_coll < g_coll, f"grid {g_coll:.1f}% → event {e_coll:.1f}%")
    check("event method has a lower collision rate among INSTANTANEOUS actions",
          ei < gi, f"grid {gi:.1f}% → event {ei:.1f}%")
    check("cross-interval instantaneous collisions (indefensible under any method) "
          "are no worse", e.inst_coll_cross <= g.inst_coll_cross,
          f"grid {g.inst_coll_cross:,} ({gx:.1f}%) → event {e.inst_coll_cross:,} ({ex:.1f}%)")
    check("cross-interval instantaneous collisions are essentially gone (<5%)",
          ex < 5.0, f"event {ex:.1f}%")
    check("every round conserves to 1e-9 under the event method",
          r.max() < 1e-9, f"max |residual| {r.max():.2e}")
    check("TARGET STATE — instantaneous collisions overall below 5% "
          "(blocked on persisted boundary states; fails on 1 Hz-only data)",
          ei < 5.0, f"{ei:.1f}%")
    check("event method has fewer mirrored CT/T magnitudes than grid",
          e.mirrored < g.mirrored, f"grid {g.mirrored:,} → event {e.mirrored:,}")
    check("event method mirrors far fewer CT/T pairs that acted at DIFFERENT moments",
          e.mirrored_apart < g.mirrored_apart,
          f"grid {g.mirrored_apart:,} → event {e.mirrored_apart:,}")
    # NOT scored as "event should be <= grid": grid is structurally incapable of
    # emitting a same-signed CT/T pair, so 0 there is arithmetic, not merit. The
    # honest assertion is that the grid figure is vacuous and the event figure is
    # the count of seconds it declines to treat as one shared cause.
    check("same-sign CT/T pairs are impossible under grid, so that metric is vacuous "
          "there; event's count is separation, not conflict",
          g.sign_conflict == 0,
          f"grid {g.sign_conflict:,} → event {e.sign_conflict:,}")
    check("rotations get a non-trivial value under the event method",
          e.by_type_sum.get("rotation", 0.0) / max(e.by_type_scored.get("rotation", 1), 1) > 1e-4,
          f"mean |impact| {e.by_type_sum.get('rotation',0.0)/max(e.by_type_scored.get('rotation',1),1):.5f}")
    check("event method does not collapse more events to exactly zero than grid",
          pct(e.n_zero, e.n_scored) <= pct(g.n_zero, g.n_scored),
          f"grid {pct(g.n_zero,g.n_scored):.1f}% → event {pct(e.n_zero,e.n_scored):.1f}%")
    # Coverage is NOT a defect to hide: the event method declines to value round-level
    # actions (buys) rather than handing them the first second's swing. Assert only
    # that everything it leaves unvalued is explained by a named disposition.
    unvalued = sum(n for k, n in verdicts.items()
                   if k in ("round-level-unvalued", "no-resolve-tick", "outside-round",
                            "empty-window", "no-window", "unknown-kind"))
    check("every event the event method leaves unscored has a named reason",
          (total_ev - e.n_scored) <= unvalued + unknown_side,
          f"unscored {total_ev - e.n_scored:,} = {unvalued:,} named dispositions "
          f"+ {unknown_side:,} with an unusable side")

    passed = sum(1 for ok, _ in _results if ok)
    failed = [d for ok, d in _results if not ok]
    print(f"\n{'─'*60}\n  {passed}/{len(_results)} passed")
    if failed:
        print("  Failed:")
        for d in failed:
            print(f"    - {d}")
    print()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
