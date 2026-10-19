"""
Result records of the attribution engine (P8.6): one `ScoredAction` per action (plus
synthetic `positioning` / `trade_setup` rows for credit that has no action of its own)
and one `RoundAttribution` per round.

    from engine.attribution.types import ScoredAction, RoundAttribution
    ra = engine.attribute_round(snapshots, actions, windows, predictor, cfg)
    ra.actions[0].V                     # actor-signed realized value
    ra.conservation_residual            # Σ V_raw + clock_ct − clock_t − ΔW_round

Sign convention (P8.2.5, B.5): every internal share `φ` is CT-positive. `V_raw` is the
CT-positive sum; `V`, `V_direct`, `V_enabler`, `V_movement` are actor-signed (`+V_raw`
for a CT actor, `−V_raw` for a T actor), so `V = V_direct + V_enabler + V_movement`.
Component dicts keep the raw CT-positive `phi` they contributed.

Filled by: P8.6.1, P8.6.2.
"""
from dataclasses import dataclass, field

# ── constants ─────────────────────────────────────────────────────────────────

METHODS          = ("exact", "sampled", "loo")
SYNTHETIC_TYPES  = ("positioning", "trade_setup")
# when an action got shares by several methods, `method` names the least exact one
METHOD_RANK      = {"exact": 0, "loo": 1, "sampled": 2}


def actor_sign(side: str | None) -> float:
    """+1 for a CT actor, −1 for a T actor (B.5). Unknown side is an error, not 0."""
    if side == "ct":
        return 1.0
    if side == "t":
        return -1.0
    raise ValueError(f"actor side {side!r} must be 'ct' or 't'")


# ── records ───────────────────────────────────────────────────────────────────

@dataclass
class ScoredAction:
    action_id:     str
    match_key:     str
    round_num:     int
    actor_id_hash: int | None
    actor_side:    str
    type:          str
    V:             float = 0.0
    V_direct:      float = 0.0
    V_enabler:     float = 0.0
    V_movement:    float = 0.0
    method:        str | None = None
    n_coalition:   int | None = None
    residual:      float | None = None
    flags:         list[str] = field(default_factory=list)
    components:    list[dict] = field(default_factory=list)

    @property
    def V_raw(self) -> float:
        """CT-positive realized value (the term in the conservation identity)."""
        return actor_sign(self.actor_side) * self.V


@dataclass
class RoundAttribution:
    match_key:             str
    round_num:             int
    actions:               list[ScoredAction]
    clock_ct:              float
    clock_t:               float
    w_start:               float
    w_end:                 float
    conservation_residual: float
    stats:                 dict = field(default_factory=dict)
