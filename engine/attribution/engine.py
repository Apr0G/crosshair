"""
Round assembly (P8.6.1): runs P8.1-P8.5, sums direct, enabler and movement shares into
V, and holds the conservation identity per round.

    from engine.attribution import engine
    ra  = engine.attribute_round(snapshots, actions, windows, predictor, cfg)
    ras = engine.attribute_match(all_snapshots, all_actions, all_windows, predictor, cfg)
    df  = engine.to_frame(ras)                       # actions_valued.parquet rows

Steps per round:
1. `partition.breakpoints` tiles [freeze_end, terminal] into event / drift intervals.
2. Predictor call 1: factual W at every breakpoint, plus every `remove_i(s_pre)` state
   the hazard enabler split needs (both are known before any W). The terminal W is
   pinned to the round label (`round_won_ct`) when `pin_terminal` and the last snapshot
   is a `terminal` sample.
3. Predictor call 2: every drift interval's movement coalition states, all intervals in
   one batch (|ΔW| from step 2 picks loo / exact / sampled per interval).
4. One predictor call per event window: every coalition `apply(s_pre, s_post, S)` the
   window's Shapley needs (`v(A)` is the factual `W(s_post)`, not re-predicted).
5. Window φ to members: pseudo pop / fade shares go to the parent durative action
   (`params["parent_action_id"]`); an event with enablers keeps `(1 − total) · φ_e`
   and the enablers take `share_i · φ_e` (trade_setup to the dead teammate, `info` to
   the clock); movement φ to the rotation / hold / save of that player whose window
   contains the interval midpoint, else to a synthetic per-player `positioning` row.
6. Clock: drift remainders, each window's `W(apply(∅)) − W(s_pre)` gap (v(∅) keeps
   s_post's clock: 1-2 ticks of time), info shares, and the ΔW of windows whose
   interventions failed (flagged in `stats["failed_windows"]`).
7. Conservation `Σ V_raw + clock_ct − clock_t = W(terminal) − W(freeze_end)`, residual
   stored; tolerance 1e-9 with only exact windows, 1e-6 with any sampled one.

Histories: a factual snapshot's history is every factual snapshot of the round with a
smaller tick; an edited snapshot reuses the history of the factual snapshot it was
derived from (`s_post` for coalitions, `s1` for movement, `s_pre` for enabler removals).

Window contract relied on (engine/actions/windows.py, P4.6.1): fields `window_id`,
`round_num`, `start_tick` (= min member tick − 1), `end_tick` (= max member tick + 1),
`member_action_ids`; read by attribute or dict key, so this module never imports it.
Pseudo-actions (`smoke_pop`, `smoke_fade`, `fire_start`, `fire_end`, `decoy_start`,
`decoy_end`) are in `actions` with `params["parent_action_id"]`.

Filled by: P8.6.1-P8.6.4.
"""
import json
import math
import time
from bisect import bisect_left

import polars as pl

from engine import config as config_mod
from engine.actions.types import PSEUDO_TYPES, make_action_id
from engine.attribution import enablers as en
from engine.attribution import movement, partition, shapley
from engine.attribution.clock import ClockBuckets, round_identity
from engine.attribution.partition import DRIFT, EVENT, tick_of, wfield
from engine.attribution.types import METHOD_RANK, RoundAttribution, ScoredAction, actor_sign
from engine.counterfactual.apply import apply, edit_hash

# ── constants ─────────────────────────────────────────────────────────────────

FADE_TYPES     = ("smoke_fade", "fire_end", "decoy_end")
MOVEMENT_TYPES = ("rotation", "hold", "save")
TOL_EXACT      = 1e-9
TOL_SAMPLED    = 1e-6


# ── helpers ───────────────────────────────────────────────────────────────────

class _Counted:
    """Wraps the injected predictor to count calls and predicted snapshots."""

    def __init__(self, predictor):
        self.inner       = predictor
        self.calls       = 0
        self.n_snapshots = 0

    def predict(self, snapshots, histories) -> dict:
        self.calls       += 1
        self.n_snapshots += len(snapshots)
        return self.inner.predict(snapshots, histories)


class _Acc:
    """Raw (CT-positive) components of one scored row while the round is assembled."""

    def __init__(self, action_id, actor, side, type_, flags=()):
        self.action_id, self.actor, self.side, self.type = action_id, actor, side, type_
        self.direct = self.enabler = self.movement = 0.0
        self.methods: set = set()
        self.n_coalition: int | None = None
        self.residual: float | None = None
        self.flags: list[str] = list(flags)
        self.components: list[dict] = []

    def add(self, bucket: str, phi: float, comp: dict, method: str | None = None,
            n: int | None = None, resid: float | None = None) -> None:
        setattr(self, bucket, getattr(self, bucket) + phi)
        self.components.append({**comp, "phi": phi})
        if method is not None:
            self.methods.add(method)
        if n is not None:
            self.n_coalition = n if self.n_coalition is None else max(self.n_coalition, n)
        if resid is not None:
            self.residual = resid if self.residual is None else max(self.residual, resid)

    def scored(self, match_key: str, round_num: int) -> ScoredAction:
        sg = actor_sign(self.side)
        vd, ve, vm = sg * self.direct + 0.0, sg * self.enabler + 0.0, sg * self.movement + 0.0   # no -0.0
        method = max(self.methods, key=lambda m: METHOD_RANK[m]) if self.methods else None
        return ScoredAction(action_id=self.action_id, match_key=match_key, round_num=round_num,
                            actor_id_hash=self.actor, actor_side=self.side, type=self.type,
                            V=vd + ve + vm, V_direct=vd, V_enabler=ve, V_movement=vm,
                            method=method, n_coalition=self.n_coalition, residual=self.residual,
                            flags=sorted(set(self.flags)), components=self.components)


def _snapshot_before(snaps: list, ticks: list[int], tick: int | None):
    """Latest factual snapshot strictly before `tick` (None when there is none)."""
    if tick is None:
        return None
    i = bisect_left(ticks, tick) - 1
    return snaps[i] if i >= 0 else None


def _side_of(snapshot, player_id) -> str | None:
    i = snapshot.index_of(player_id)
    return snapshot.players[i].side if i is not None else None


def _pinned(last) -> float | None:
    if last.meta is None or "terminal" not in last.meta.kinds or not last.labels:
        return None
    y = last.labels.get("round_won_ct")
    if y is None or (isinstance(y, float) and math.isnan(y)):
        return None
    return float(y)


# ── P8.6.1 round ──────────────────────────────────────────────────────────────

def attribute_round(snapshots, actions, windows, predictor, cfg=None, *, geometry=None,
                    pin_terminal: bool = True) -> RoundAttribution:
    """Realized value V of every action of one round (P8.1-P8.5) and the clock buckets;
    `predictor` has `predict(snapshots, histories)`. See the module docstring."""
    t_start = time.perf_counter()
    cfg   = cfg if cfg is not None else config_mod.load()
    acfg  = cfg.attribution
    snaps = partition.sorted_round(snapshots)
    ticks = [tick_of(s) for s in snaps]
    by_tick = {tick_of(s): s for s in snaps}
    hist  = partition.histories_for(snaps)
    match_key, round_num = snaps[0].meta.match_key, snaps[0].meta.round_num
    pred  = _Counted(predictor)
    clock = ClockBuckets()

    acts: dict = {}
    for a in actions:
        if a.action_id in acts:
            raise ValueError(f"duplicate action_id {a.action_id}")
        acts[a.action_id] = a
    windows = list(windows)
    win_by_id = {str(wfield(w, "window_id")): w for w in windows}

    # ── P8.1 partition ──
    intervals = partition.breakpoints(snaps, windows)
    bad = partition.check_tiling(intervals, ticks[0], ticks[-1])
    if bad:
        raise ValueError(f"partition does not tile the round: {bad}")
    events = [iv for iv in intervals if iv.kind == EVENT]
    drifts = [iv for iv in intervals if iv.kind == DRIFT]
    members_of = {}
    for iv in events:
        ids = [str(x) for x in wfield(win_by_id[iv.window_id], "member_action_ids")]
        missing = [x for x in ids if x not in acts]
        if missing or not ids:
            raise ValueError(f"window {iv.window_id}: member ids not among the actions: {missing or 'none'}")
        members_of[iv.window_id] = [acts[x] for x in ids]

    # ── call 1: factual W at breakpoints + enabler removal states ──
    bts = partition.breakpoint_ticks(intervals)
    states1, hist1 = [by_tick[t] for t in bts], [hist[t] for t in bts]
    row_of_tick = {t: k for k, t in enumerate(bts)}
    enabler_row: dict = {}                      # (event_id, enabler index) -> row in call 1
    enabler_flags: list[str] = []
    if acfg.enabler_mode == "hazard":
        for iv in events:
            s_pre = by_tick[iv.start_tick]
            for m in members_of[iv.window_id]:
                if m.type not in en.HEAD_FOR_TYPE:
                    continue
                for i, e in enumerate(m.enablers):
                    if e.get("kind") in en.CLOCK_KINDS:
                        continue
                    linked = acts.get(e.get("action_id"))
                    ref = _snapshot_before(snaps, ticks, linked.id_tick if linked is not None else None)
                    try:
                        s_rm = en.removal_state(e.get("kind"), linked, s_pre, ref, geometry=geometry)
                    except ValueError as err:
                        enabler_flags.append(f"{m.action_id}:{e.get('kind')}: {err}")
                        continue
                    enabler_row[(m.action_id, i)] = len(states1)
                    states1.append(s_rm)
                    hist1.append(hist[iv.start_tick])
    out1 = pred.predict(states1, hist1)
    W = {t: float(out1["win"][row_of_tick[t]]) for t in bts}
    pinned = _pinned(snaps[-1]) if pin_terminal else None
    if pinned is not None:
        W[ticks[-1]] = pinned
    tele = partition.telescoping_residual(intervals, W)

    accs: dict = {}

    def acc_for_action(a, flags=()) -> _Acc:
        if a.action_id not in accs:
            accs[a.action_id] = _Acc(a.action_id, a.actor_id_hash, a.actor_side, a.type, flags)
        return accs[a.action_id]

    def acc_synthetic(type_: str, pid) -> _Acc:
        aid = make_action_id(match_key, round_num, type_, pid, None)
        if aid not in accs:
            accs[aid] = _Acc(aid, pid, _side_of(snaps[0], pid), type_, ("synthetic",))
        return accs[aid]

    # ── call 2: every drift interval's movement coalitions ──
    plans, states2, hist2, where2 = [], [], [], []
    for iv in drifts:
        s0, s1 = by_tick[iv.start_tick], by_tick[iv.end_tick]
        dW = W[iv.end_tick] - W[iv.start_tick]
        seed = shapley.seed_of(f"{match_key}|{round_num}|drift|{iv.start_tick}|{iv.end_tick}")
        plan = movement.plan_interval(s0, s1, dW, cfg, seed)
        full = frozenset(plan.movers)
        for S in plan.coalitions:
            if S == full:
                continue
            where2.append((len(plans), S))
            states2.append(movement.frozen_state(s1, s0, plan.movers, S, geometry=geometry))
            hist2.append(hist[iv.end_tick])
        plans.append((iv, plan, dW, {full: W[iv.end_tick]}))
    if states2:
        out2 = pred.predict(states2, hist2)
        for (k, S), w in zip(where2, out2["win"]):
            plans[k][3][S] = float(w)

    by_actor_moves: dict = {}
    for a in actions:
        if a.type in MOVEMENT_TYPES and a.window_start_tick is not None and a.window_end_tick is not None:
            by_actor_moves.setdefault(a.actor_id_hash, []).append(a)
    for lst in by_actor_moves.values():
        lst.sort(key=lambda a: (a.window_start_tick, a.action_id))

    drift_methods: dict = {}
    unassigned: dict = {}
    for iv, plan, dW, table in plans:
        res = movement.solve_interval(plan, table.__getitem__, dW, cfg)
        clock.add(res.clock, "drift")
        drift_methods[res.method or "none"] = drift_methods.get(res.method or "none", 0) + 1
        mid = (iv.start_tick + iv.end_tick) / 2
        for pid in sorted(res.phi):
            phi = res.phi[pid]
            mv = next((a for a in by_actor_moves.get(pid, ())
                       if a.window_start_tick <= mid <= a.window_end_tick), None)
            if mv is not None:
                acc = acc_for_action(mv)
            else:
                acc = acc_synthetic("positioning", pid)
                unassigned[pid] = unassigned.get(pid, 0.0) + phi
            acc.add("movement", phi, {"source": "drift", "interval": [iv.start_tick, iv.end_tick],
                                      "method": res.method, "n_movers": res.n_movers},
                    res.method, res.n_movers, res.residual)

    # ── calls 3..: one per event window ──
    failed, sampled_windows, gap_total, info_total, max_wres = [], [], 0.0, 0.0, 0.0
    n_coalition_states = len(states2)
    for iv in events:
        wid = iv.window_id
        members = members_of[wid]
        s_pre, s_post = by_tick[iv.start_tick], by_tick[iv.end_tick]
        dW = W[iv.end_tick] - W[iv.start_tick]
        ids = sorted(m.action_id for m in members)
        full = frozenset(ids)
        seed = shapley.seed_of(wid)
        need = [S for S in shapley.required_coalitions(ids, acfg, seed) if S != full]
        try:
            built = [(edit_hash(s_post, S), apply(s_pre, s_post, members, set(S), geometry=geometry)[0])
                     for S in need]
        except ValueError as err:                # InterventionError is a ValueError
            failed.append({"window_id": wid, "error": str(err)})
            clock.add(dW, "failed_window")
            for m in members:
                acc_for_action(m).flags.append("window_failed")
            continue
        n_coalition_states += len(built)
        out = pred.predict([b[1] for b in built], [hist[iv.end_tick]] * len(built))
        values = {h: float(w) for (h, _), w in zip(built, out["win"])}
        values[edit_hash(s_post, full)] = W[iv.end_tick]

        def key(S, _s=s_post):
            return edit_hash(_s, S)

        phi, method, resid = shapley.shapley(ids, lambda S: values[key(S)], acfg, seed, key=key)
        if method == "sampled":
            sampled_windows.append(wid)
        max_wres = max(max_wres, resid)
        gap = values[key(frozenset())] - W[iv.start_tick]
        clock.add(gap, "window_gap")
        gap_total += gap
        n = len(ids)
        base = {"window_id": wid, "method": method}
        for m in members:
            p = phi[m.action_id]
            if m.type in PSEUDO_TYPES:
                parent = m.params.get("parent_action_id")
                src = "fade" if m.type in FADE_TYPES else "pop"
                if parent in acts:
                    acc = acc_for_action(acts[parent])
                else:
                    acc = acc_for_action(m, ("orphan_pseudo",))
                acc.add("direct", p, {**base, "source": src, "pseudo_id": m.action_id,
                                      "pseudo_type": m.type}, method, n, resid)
                continue
            acc = acc_for_action(m)
            if m.type not in en.HEAD_FOR_TYPE or not m.enablers:
                acc.add("direct", p, {**base, "source": "window"}, method, n, resid)
                continue
            # ── P8.5 enabler split of this event ──
            q_pre = en.head_value(out1, row_of_tick[iv.start_tick], m, s_pre)
            q_rm = {}
            for i, _e in enumerate(m.enablers):
                r = enabler_row.get((m.action_id, i))
                q_rm[i] = None if r is None else en.head_value(out1, r, m, states1[r])
            sp = en.split(m.enablers, q_pre, q_rm, acfg)
            actor_amt, parts = en.distribute(sp, p)
            acc.add("direct", actor_amt, {**base, "source": "window", "phi_event": p,
                                          "enabler_total": sp.total, "enabler_mode": sp.mode},
                    method, n, resid)
            acc.flags += sp.flags + [f for s in sp.shares for f in s.flags]
            for s, amt in parts:
                e = m.enablers[s.index]
                comp = {**base, "source": "enabler", "kind": s.kind, "event_id": m.action_id,
                        "share": s.share, "raw": s.raw}
                if s.to_clock:
                    clock.add(amt, "info")
                    info_total += amt
                    continue
                linked = acts.get(s.action_id)
                if s.kind == "trade_setup":
                    ben = e.get("beneficiary_id")
                    if ben is None and linked is not None:
                        ben = linked.params.get("victim_id")
                    if ben is None:
                        clock.add(amt, "enabler_unlinked")
                        acc.flags.append("enabler_unlinked:trade_setup")
                        continue
                    acc_synthetic("trade_setup", ben).add("enabler", amt, comp, method, n, resid)
                elif linked is not None:
                    acc_for_action(linked).add("enabler", amt, comp, method, n, resid)
                else:
                    clock.add(amt, "enabler_unlinked")
                    acc.flags.append(f"enabler_unlinked:{s.kind}")

    # ── P8.6.2 rows ──
    for a in actions:                            # every non-pseudo action gets a row
        if a.type not in PSEUDO_TYPES:
            acc_for_action(a)
    order = sorted(accs.values(), key=lambda c: ("synthetic" in c.flags, c.action_id))
    rows = [c.scored(match_key, round_num) for c in order]

    # ── P8.6.1 conservation ──
    w0, w1 = W[ticks[0]], W[ticks[-1]]
    v_sum = math.fsum(c.direct + c.enabler + c.movement for c in order)
    resid = math.fsum([v_sum, clock.clock_ct, -clock.clock_t, -(w1 - w0)])
    tol = TOL_SAMPLED if (sampled_windows or drift_methods.get("sampled")) else TOL_EXACT
    stats = {
        "predictor_calls":     pred.calls,
        "n_snapshots_predicted": pred.n_snapshots,
        "coalition_evals":     n_coalition_states,
        "n_breakpoints":       len(bts),
        "n_enabler_states":    len(enabler_row),
        "n_windows":           len(events),
        "n_drift":             len(drifts),
        "drift_methods":       drift_methods,
        "sampled_windows":     sampled_windows,
        "max_window_residual": max_wres,
        "failed_windows":      failed,
        "window_gap_to_clock": gap_total,
        "info_to_clock":       info_total,
        "clock_by_source":     dict(clock.by_source),
        "clock_identity_ok":   round_identity(clock),
        "unassigned_movement": unassigned,
        "enabler_flags":       enabler_flags,
        "terminal_pinned":     pinned is not None,
        "telescoping_residual": tele,
        "conservation_tol":    tol,
        "conservation_ok":     abs(resid) <= tol,
        "seconds":             time.perf_counter() - t_start,
    }
    return RoundAttribution(match_key=match_key, round_num=round_num, actions=rows,
                            clock_ct=clock.clock_ct, clock_t=clock.clock_t, w_start=w0, w_end=w1,
                            conservation_residual=resid, stats=stats)


# ── match ─────────────────────────────────────────────────────────────────────

def attribute_match(snapshots, actions, windows, predictor, cfg=None, *, geometry=None,
                    pin_terminal: bool = True) -> list[RoundAttribution]:
    """`attribute_round` for every round present in `snapshots` (grouped by
    `meta.round_num`; actions by `round_num`, windows by their `round_num` field)."""
    cfg = cfg if cfg is not None else config_mod.load()
    snaps_by, acts_by, wins_by = {}, {}, {}
    for s in snapshots:
        snaps_by.setdefault(s.meta.round_num, []).append(s)
    for a in actions:
        acts_by.setdefault(a.round_num, []).append(a)
    for w in windows:
        wins_by.setdefault(int(wfield(w, "round_num")), []).append(w)
    return [attribute_round(snaps_by[r], acts_by.get(r, []), wins_by.get(r, []), predictor, cfg,
                            geometry=geometry, pin_terminal=pin_terminal)
            for r in sorted(snaps_by)]


# ── output ────────────────────────────────────────────────────────────────────

FRAME_SCHEMA = {
    "match_key":     pl.String,
    "round_num":     pl.Int64,
    "action_id":     pl.String,
    "actor_id_hash": pl.Int64,
    "actor_side":    pl.String,
    "type":          pl.String,
    "V":             pl.Float64,
    "V_raw":         pl.Float64,
    "V_direct":      pl.Float64,
    "V_enabler":     pl.Float64,
    "V_movement":    pl.Float64,
    "method":        pl.String,
    "n_coalition":   pl.Int64,
    "residual":      pl.Float64,
    "flags":         pl.List(pl.String),
    "components":    pl.String,               # JSON list of component dicts
    "h_pre":         pl.List(pl.Float32),     # filled by P9; None here
}


def to_frame(round_attributions) -> pl.DataFrame:
    """One row per ScoredAction, columns FRAME_SCHEMA (`actions_valued.parquet`)."""
    rows = []
    for ra in round_attributions:
        for sa in ra.actions:
            rows.append({
                "match_key": sa.match_key, "round_num": sa.round_num, "action_id": sa.action_id,
                "actor_id_hash": sa.actor_id_hash, "actor_side": sa.actor_side, "type": sa.type,
                "V": sa.V, "V_raw": sa.V_raw, "V_direct": sa.V_direct, "V_enabler": sa.V_enabler,
                "V_movement": sa.V_movement, "method": sa.method, "n_coalition": sa.n_coalition,
                "residual": sa.residual, "flags": list(sa.flags),
                "components": json.dumps(sa.components, sort_keys=True, allow_nan=False),
                "h_pre": None,
            })
    return pl.DataFrame(rows, schema=FRAME_SCHEMA)
