"""
D = Q - B, X = V - Q, regret R <= 0, verdicts (P9.4.1) and the P9.5.1 outputs.

    from engine.value import scores
    sv = scores.compute(scored_action, Q, B, panel, cfg, q_alternatives=qa, descriptor_key=k)
    sv.D, sv.X, sv.R, sv.verdict_decision
    svs = scores.score_frame(valued, h, qnet, bnet, vocab, index, cfg)   # whole match
    df  = scores.to_frame(svs, run_id)          # actions_scored.parquet
    db.insert_action_scores(scores.db_rows(df, run_id))

Formulas (B.7):

    D = Q̂(s,a) − B̂(s)          X = V − Q̂(s,a)          V = B̂ + D + X  (checked to 1e-9)
    R = Q̂(s,a) − max({Q̂(s,a)} ∪ {Q̂(s,a') : a' in panel, n(a') >= n_min_regret})  <= 0

The alternatives a' are scored at the CURRENT state: the current `h_pre`, tier and side,
with the alternative's representative descriptor in which the state fields
(`descriptors.STATE_FIELDS`: timing and alive-difference buckets) are replaced by the
current action's (`alternative_ids`). The action's own Q̂ is always in the max, so R <= 0
even when the action itself has support < n_min. An alternative with n < n_min never
enters the max.

Verdicts, inclusive neutral band (δ_D = value.delta_D, δ_X = value.delta_X):

    good    if score >  +δ
    bad     if score <  −δ
    neutral if −δ <= score <= +δ      (a score exactly on ±δ is neutral)

Flags added: `no_panel` (no neighbour survived the filters), `q_low_support` (the
action's own descriptor key has fewer than n_min neighbours in its panel).

Filled by: P9.4.1, P9.5.1.
"""
import json
import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from engine.attribution.types import ScoredAction
from engine.db import SCORE_COLUMNS
from engine.value import neighbours
from engine.value.descriptors import FIELDS, STATE_FIELDS, ids_from_frame
from engine.value.q_model import predict_b, predict_q

# ── constants ─────────────────────────────────────────────────────────────────

VERDICTS        = ("good", "neutral", "bad")
IDENTITY_TOL    = 1e-9
_STATE_IDX      = [FIELDS.index(f) for f in STATE_FIELDS]
# frame columns score_frame reads on top of attribution FRAME_SCHEMA
INPUT_COLUMNS   = ("match_key", "round_num", "action_id", "actor_id_hash", "actor_side", "type", "V",
                   "V_direct", "V_enabler", "V_movement", "method", "n_coalition", "residual", "flags",
                   "components", "tier", "map_version", "descriptor_key",
                   *(f"desc_{f}" for f in FIELDS))

SCORED_SCHEMA = {
    "match_key":         pl.String,
    "round_num":         pl.Int64,
    "action_id":         pl.String,
    "actor_id_hash":     pl.Int64,
    "actor_side":        pl.String,
    "type":              pl.String,
    "V":                 pl.Float64,
    "V_raw":             pl.Float64,
    "V_direct":          pl.Float64,
    "V_enabler":         pl.Float64,
    "V_movement":        pl.Float64,
    "Q":                 pl.Float64,
    "B":                 pl.Float64,
    "D":                 pl.Float64,
    "R":                 pl.Float64,
    "X":                 pl.Float64,
    "verdict_decision":  pl.String,
    "verdict_execution": pl.String,
    "descriptor_key":    pl.String,
    "panel_ref":         pl.String,
    "panel_n":           pl.Int64,
    "panel":             pl.String,           # JSON Panel.to_dict(), null without a panel
    "method":            pl.String,
    "n_coalition":       pl.Int64,
    "residual":          pl.Float64,
    "flags":             pl.List(pl.String),
    "components":        pl.String,
    "run_id":            pl.String,
}


# ── record ────────────────────────────────────────────────────────────────────

@dataclass
class ScoredActionWithValue(ScoredAction):
    Q:                 float | None = None
    B:                 float | None = None
    D:                 float | None = None
    R:                 float | None = None
    X:                 float | None = None
    verdict_decision:  str | None = None
    verdict_execution: str | None = None
    descriptor_key:    str | None = None
    panel_ref:         str | None = None
    panel:             neighbours.Panel | None = field(default=None, repr=False)


# ── pieces ────────────────────────────────────────────────────────────────────

def verdict(score: float | None, delta: float) -> str | None:
    """good / bad / neutral with the inclusive neutral band [−δ, +δ]; None if unknown."""
    if score is None or score != score:
        return None
    if score > delta:
        return "good"
    if score < -delta:
        return "bad"
    return "neutral"


def regret(q_self: float, panel: neighbours.Panel | None, q_alternatives, n_min: int) -> float:
    """R = q_self − max(q_self, Q̂ of panel alternatives with n >= n_min) <= 0.
    `q_alternatives` is aligned with `panel.entries`; NaN / None entries are ignored."""
    entries = panel.entries if panel is not None else []
    if q_alternatives is None:
        if any(e.n >= n_min for e in entries):
            raise ValueError("panel has supported alternatives but q_alternatives was not given")
        return 0.0
    q_alternatives = list(q_alternatives)
    if len(q_alternatives) != len(entries):
        raise ValueError(f"{len(q_alternatives)} alternative Q values for {len(entries)} panel entries")
    best = q_self
    for e, q in zip(entries, q_alternatives):
        if e.n >= n_min and q is not None and q == q and q > best:
            best = q
    return q_self - best


def alternative_ids(alt_ids, current_ids) -> np.ndarray:
    """The alternative's descriptor ids with the state fields of the current action."""
    out = np.array(alt_ids, dtype=np.int64, copy=True)
    cur = np.asarray(current_ids, dtype=np.int64)
    out[..., _STATE_IDX] = cur[..., _STATE_IDX]
    return out


def compute(scored_action: ScoredAction, Q: float, B: float, panel: neighbours.Panel | None, cfg, *,
            q_alternatives=None, descriptor_key: str | None = None) -> ScoredActionWithValue:
    """Add Q, B, D, R, X, verdicts and the panel reference to one valued action.
    `q_alternatives` (aligned with `panel.entries`) is required when any entry has
    n >= n_min_regret; without a panel R = 0 (only the action itself is in the max)."""
    vc = cfg.value
    Q, B, V = float(Q), float(B), float(scored_action.V)
    for name, x in (("Q", Q), ("B", B), ("V", V)):
        if not math.isfinite(x):
            raise ValueError(f"{name} = {x} for action {scored_action.action_id}")
    eligible = panel is not None and any(e.n >= vc.n_min_regret for e in panel.entries)
    if eligible and q_alternatives is None:
        raise ValueError("panel has supported alternatives but q_alternatives was not given")
    D = Q - B
    X = V - Q
    if abs(B + D + X - V) > IDENTITY_TOL:
        raise ArithmeticError(f"identity V = B + D + X off by {B + D + X - V}")
    R = regret(Q, panel, q_alternatives if eligible else None, vc.n_min_regret)

    flags = list(scored_action.flags)
    has_panel = panel is not None and panel.n_used > 0
    if not has_panel:
        flags.append("no_panel")
    elif descriptor_key is not None and panel.support(descriptor_key) < vc.n_min_regret:
        flags.append("q_low_support")
    base = {f: getattr(scored_action, f) for f in ScoredAction.__dataclass_fields__}
    base["flags"] = flags
    return ScoredActionWithValue(
        **base, Q=Q, B=B, D=D, R=R, X=X,
        verdict_decision=verdict(D, vc.delta_D), verdict_execution=verdict(X, vc.delta_X),
        descriptor_key=descriptor_key, panel_ref=panel.ref if has_panel else None,
        panel=panel if has_panel else None)


# ── whole frame ───────────────────────────────────────────────────────────────

def valued_from_frame(frame: pl.DataFrame) -> list[ScoredAction]:
    """ScoredAction records back from an `actions_valued` frame."""
    out = []
    for r in frame.iter_rows(named=True):
        out.append(ScoredAction(
            action_id=r["action_id"], match_key=r["match_key"], round_num=r["round_num"],
            actor_id_hash=r["actor_id_hash"], actor_side=r["actor_side"], type=r["type"], V=r["V"],
            V_direct=r["V_direct"], V_enabler=r["V_enabler"], V_movement=r["V_movement"],
            method=r["method"], n_coalition=r["n_coalition"], residual=r["residual"],
            flags=list(r["flags"] or []), components=json.loads(r["components"]) if r["components"] else []))
    return out


def score_frame(frame: pl.DataFrame, h: np.ndarray, qnet, bnet, vocab, index: neighbours.NeighbourIndex,
                cfg, *, exclude_own_match: bool = True, top: int = neighbours.TOP_N) -> list[ScoredActionWithValue]:
    """Score every row of a valued frame (columns INPUT_COLUMNS; `h` = its h_pre rows):
    Q̂ / B̂ batched, one panel per row (same side, same map version, own match
    excluded), alternatives scored at the current state, then `compute`."""
    missing = [c for c in INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"valued frame missing columns {missing}")
    h = np.asarray(h, dtype=np.float32)
    if h.shape[0] != frame.height:
        raise ValueError(f"h has {h.shape[0]} rows, frame {frame.height}")
    if frame.height == 0:
        return []
    vc = cfg.value
    ids = ids_from_frame(frame, vocab)
    tier, side = frame["tier"], frame["actor_side"]
    Q = predict_q(qnet, h, ids, tier, side)
    B = predict_b(bnet, h, tier, side)
    sides = side.to_list()
    pans = neighbours.panels(index, h, sides, frame["map_version"].to_list(), vc.k_neighbours,
                             exclude_match_keys=frame["match_key"].to_list() if exclude_own_match
                             else [None] * frame.height, top=top)

    # alternatives with support, scored in one batch at the current states
    rows_i, rows_j, alt = [], [], []
    for i, p in enumerate(pans):
        for j, e in enumerate(p.entries):
            if e.n >= vc.n_min_regret:
                if e.desc_ids is None:
                    raise ValueError("neighbour side table has no desc_ids; alternatives cannot be scored")
                rows_i.append(i)
                rows_j.append(j)
                alt.append(alternative_ids(e.desc_ids, ids[i]))
    q_alt = [[math.nan] * len(p.entries) for p in pans]
    if alt:
        ri = np.asarray(rows_i)
        tiers_np = np.asarray(tier.to_list(), dtype=object)
        qa = predict_q(qnet, h[ri], np.stack(alt), tiers_np[ri], np.asarray(sides, dtype=object)[ri])
        for i, j, v in zip(rows_i, rows_j, qa):
            q_alt[i][j] = float(v)

    valued = valued_from_frame(frame)
    keys = frame["descriptor_key"].to_list()
    return [compute(sa, Q[i], B[i], pans[i], cfg, q_alternatives=q_alt[i], descriptor_key=keys[i])
            for i, sa in enumerate(valued)]


# ── outputs (P9.5.1) ──────────────────────────────────────────────────────────

def _f(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def to_frame(scored: list[ScoredActionWithValue], run_id: str | None = None) -> pl.DataFrame:
    """`actions_scored.parquet`: one row per scored action, columns SCORED_SCHEMA."""
    rows = []
    for s in scored:
        rows.append({
            "match_key": s.match_key, "round_num": s.round_num, "action_id": s.action_id,
            "actor_id_hash": s.actor_id_hash, "actor_side": s.actor_side, "type": s.type,
            "V": s.V, "V_raw": s.V_raw, "V_direct": s.V_direct, "V_enabler": s.V_enabler,
            "V_movement": s.V_movement, "Q": _f(s.Q), "B": _f(s.B), "D": _f(s.D), "R": _f(s.R), "X": _f(s.X),
            "verdict_decision": s.verdict_decision, "verdict_execution": s.verdict_execution,
            "descriptor_key": s.descriptor_key, "panel_ref": s.panel_ref,
            "panel_n": s.panel.n_used if s.panel is not None else 0,
            "panel": json.dumps(s.panel.to_dict(), sort_keys=True, allow_nan=False) if s.panel is not None else None,
            "method": s.method, "n_coalition": s.n_coalition, "residual": _f(s.residual),
            "flags": list(s.flags), "components": json.dumps(s.components, sort_keys=True, allow_nan=False),
            "run_id": run_id,
        })
    return pl.DataFrame(rows, schema=SCORED_SCHEMA)


def db_rows(frame: pl.DataFrame, run_id: str | None = None) -> list[dict]:
    """Rows for `db.insert_action_scores`: exactly `db.SCORE_COLUMNS`, NaN -> None.
    `run_id` overrides the frame's column."""
    out = []
    for r in frame.iter_rows(named=True):
        row = {c: _f(r.get(c)) for c in SCORE_COLUMNS}
        if run_id is not None:
            row["run_id"] = run_id
        row["flags"] = list(r.get("flags") or [])
        out.append(row)
    return out
