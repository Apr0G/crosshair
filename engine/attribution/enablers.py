"""
Enabler splits (P8.5): hazard-based raw shares, clamped to bands, capped at
`attribution.enabler_total_cap`; `enabler_mode: fixed` uses `fixed_shares`.

    from engine.attribution import enablers as en
    s_rm = en.removal_state("flash", flash_action, s_pre, ref=None)        # remove_i(s_pre)
    q_pre, q_rm = en.head_value(out_pre, kill, s_pre), en.head_value(out_rm, kill, s_rm)
    sp = en.split(kill.enablers, q_pre, {0: q_rm}, cfg.attribution)
    actor_amt, parts = en.distribute(sp, phi_e)        # actor_amt + Σ parts == phi_e

Formula (B.4), for an event e with direct share φ_e and hazard head q_v at s_pre:

    raw_i   = max(0, q_v(s_pre) − q_v(remove_i(s_pre))) / q_v(s_pre)
    share_i = clamp(raw_i, band_i)          band_trade = trade_share_band, else other_share_band
    total   = min(Σ share_i, enabler_total_cap)   (shares rescaled ∝ when capped)
    V_i    += share_i · φ_e ;  V_actor += (1 − total) · φ_e

Head per event type: death hazard of the victim (its slot in s_pre) for kill / damage,
`plant15` for plant_complete, `defuse15` for defuse_complete. `remove_i` by enabler
kind: flash → unflash, smoke_cover / fire_pressure → remove_utility, trade_setup →
undo_kill of the linked kill (the teammate's death; reference = the factual snapshot
before that kill), he_damage → the linked action's own undo from
`INTERVENTION_FOR_TYPE` (`undo_he` for an `he` action, `undo_damage` for `damage`).

Edge rules:
- `q_v(s_pre) < min_hazard` (P8.5.2), or `q_v(s_pre) <= 0` (config allows
  `min_hazard = 0`; raw_i would divide by zero), or NaN (head missing): every enabler
  present gets its band's lower bound; flagged `min_hazard` / `hazard_nan`.
- an intervention that fails its precondition: that enabler gets its band's lower bound;
  flagged `remove_failed:<kind>`.
- `info`: no intervention exists, so its share is the other band's lower bound in
  hazard mode and 0 in fixed mode (no fixed share is defined); whatever it gets goes to
  the team clock, not to an action (P8.5.1; with the shipped config that is 0).
- `fixed` mode (P8.5.3): `share_i = fixed_shares[kind]`, then the cap.

Filled by: P8.5.1-P8.5.3.
"""
import math
from dataclasses import dataclass, field

from engine.counterfactual.interventions import (INTERVENTION_FOR_TYPE, remove_utility, undo_kill,
                                                 unflash)

# ── constants ─────────────────────────────────────────────────────────────────

HEAD_FOR_TYPE = {
    "kill":            "hazard",
    "damage":          "hazard",
    "plant_complete":  "plant15",
    "defuse_complete": "defuse15",
}
ENABLER_KINDS = ("flash", "smoke_cover", "trade_setup", "he_damage", "fire_pressure", "info")
CLOCK_KINDS   = ("info",)


# ── heads ─────────────────────────────────────────────────────────────────────

def head_value(out: dict, row: int, event, snapshot) -> float | None:
    """q_v for `event` from row `row` of a predictor output; None when unknown."""
    head = HEAD_FOR_TYPE.get(event.type)
    if head is None:
        return None
    if head == "hazard":
        vid = event.params.get("victim_id")
        i = snapshot.index_of(vid) if vid is not None else None
        if i is None:
            return None
        q = float(out["hazard"][row][i])
    else:
        q = float(out[head][row])
    return None if math.isnan(q) else q


# ── P8.5.1 remove_i ───────────────────────────────────────────────────────────

def removal_state(kind: str, linked, s_pre, ref=None, *, geometry=None):
    """`remove_i(s_pre)` for one enabler. `linked` is the enabler's action; `ref` the
    factual snapshot before it (needed by trade_setup and he_damage). Raises
    InterventionError (a ValueError) when the edit's precondition fails, ValueError for
    `info` (no intervention)."""
    if kind in CLOCK_KINDS:
        raise ValueError("info enablers have no intervention")
    if linked is None:
        raise ValueError(f"{kind} enabler has no linked action")
    if kind == "flash":
        return unflash(s_pre, linked, None, geometry=geometry)
    if kind in ("smoke_cover", "fire_pressure"):
        return remove_utility(s_pre, linked, None, geometry=geometry)
    if kind == "trade_setup":
        return undo_kill(s_pre, linked, ref, geometry=geometry)
    if linked.type not in INTERVENTION_FOR_TYPE:
        raise ValueError(f"{kind} enabler: no intervention for linked type {linked.type!r}")
    undo, _redo = INTERVENTION_FOR_TYPE[linked.type]
    return undo(s_pre, linked, ref, geometry=geometry)


def band_for(kind: str, cfg_attr) -> tuple[float, float]:
    return tuple(cfg_attr.trade_share_band if kind == "trade_setup" else cfg_attr.other_share_band)


# ── split ─────────────────────────────────────────────────────────────────────

@dataclass
class EnablerShare:
    index:          int                  # position in the event's enabler list
    action_id:      str | None
    kind:           str
    share:          float
    raw:            float | None
    to_clock:       bool
    flags:          list[str] = field(default_factory=list)


@dataclass
class EnablerSplit:
    shares:      list[EnablerShare]
    total:       float
    actor_share: float
    capped:      bool
    mode:        str
    flags:       list[str] = field(default_factory=list)


def _clamp(x: float, band) -> float:
    return min(max(x, band[0]), band[1])


def split(enablers: list[dict], q_pre: float | None, q_removed: dict, cfg_attr) -> EnablerSplit:
    """Shares for every enabler of one event. `q_removed[i]` is q_v(remove_i(s_pre))
    for enabler `i` (None when the removal failed or the kind has none)."""
    mode, flags, shares = cfg_attr.enabler_mode, [], []
    unstable = None
    if mode == "hazard":
        if q_pre is None or math.isnan(q_pre):
            unstable = "hazard_nan"
        elif q_pre <= 0 or q_pre < cfg_attr.min_hazard:     # q_pre <= 0: no division (LOW-27)
            unstable = "min_hazard"
        if unstable and enablers:
            flags.append(unstable)
    for i, e in enumerate(enablers):
        kind, aid = e.get("kind"), e.get("action_id")
        band, f = band_for(kind, cfg_attr), []
        raw = None
        if mode == "fixed":
            share = 0.0 if kind in CLOCK_KINDS else float(getattr(cfg_attr.fixed_shares, kind, 0.0))
        elif kind in CLOCK_KINDS:
            share = band[0]
        elif unstable:
            share = band[0]
            f.append(unstable)
        elif q_removed.get(i) is None:
            share = band[0]
            f.append(f"remove_failed:{kind}")
        else:
            raw = max(0.0, q_pre - q_removed[i]) / q_pre
            share = _clamp(raw, band)
        shares.append(EnablerShare(i, aid, kind, share, raw, kind in CLOCK_KINDS, f))
    tot = math.fsum(s.share for s in shares)
    cap, capped = cfg_attr.enabler_total_cap, False
    if tot > cap:
        capped = True
        for s in shares:
            s.share = s.share * cap / tot
        tot = math.fsum(s.share for s in shares)
    return EnablerSplit(shares, tot, 1.0 - tot, capped, mode, flags)


def distribute(sp: EnablerSplit, phi_e: float) -> tuple[float, list[tuple[EnablerShare, float]]]:
    """(actor amount, [(share, amount)]) with actor amount = φ_e − Σ amounts, so the
    parts add back to φ_e."""
    parts = [(s, s.share * phi_e) for s in sp.shares]
    return phi_e - math.fsum(a for _, a in parts), parts
