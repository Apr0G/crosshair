"""
Action descriptors for Q and the neighbour panel (P9.1.1).

    from engine.value import descriptors
    d     = descriptors.encode(action, state_pre)      # dict with exactly FIELDS
    vocab = descriptors.DescriptorVocab.fit([d, ...]); vocab.save(path)
    ids   = descriptors.to_ids(d, vocab)               # np.int64[len(FIELDS)], UNK = 0
    descriptors.key(d)                                 # "rotation||B Site" (panel grouping)

Fields (P9.1.1), every value a str / int or None (None = unknown, never "?"):

    type, subtype          the action's own (subtype: molotov/incendiary, team damage)
    weapon_class           params["weapon_class"]; buys: primary_class; else the actor's
                           active weapon class in `state_pre`
    from_callout           where the action starts (attacker / thrower / mover callout),
                           else the actor's callout in `state_pre`
    to_callout             where it goes or lands: to_callout (rotation, save),
                           land_callout (utility), victim_callout (kill), target_callout
                           (peek), facing_callout (hold). One field for both spellings.
    timing_bucket          pre-plant by `time_into_round_s`, half-open: "0-15" [0, 15),
                           "15-45" [15, 45), "45-115" [45, inf); post-plant by
                           `time_since_plant_s`: "pp_early" <= 20 s, "pp_late" > 20 s
    alive_diff_bucket      actor-side alive minus enemy alive, clipped to [-3, +3]
    has_los_to_enemy       peek / hold only: 1 if the actor sees any enemy in
                           `state_pre.pair` (los channel), 0 if none, None if unknown
    n_enemies_affected     flash / he / molotov: number of enemies blinded / damaged,
                           clipped to 5
    loadout_class          buy: loadout_class; team_buy: team_class
    site                   plant_*: params site; defuse_*: `ctx.bomb_site`

Vocabulary: one per field, ids `UNK = 0`, `PAD = 1`, tokens from 2 in sorted order.
None and tokens never seen at fit time both map to UNK.

Filled by: P9.1.1, P9.1.2.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

from engine.actions.types import TYPE_KIND
from engine.state.schema import PAIR_CHANNELS

# ── constants ─────────────────────────────────────────────────────────────────

FIELDS = ("type", "subtype", "weapon_class", "from_callout", "to_callout", "timing_bucket",
          "alive_diff_bucket", "has_los_to_enemy", "n_enemies_affected", "loadout_class", "site")
# fields that describe the state, not the choice: an alternative a' is scored at the
# current state with these taken from the current action (scores.alternative_ids)
STATE_FIELDS = ("timing_bucket", "alive_diff_bucket")
COLUMN_PREFIX = "desc_"

UNK, PAD = 0, 1
N_SPECIAL = 2

TIMING_BUCKETS = ("0-15", "15-45", "45-115", "pp_early", "pp_late")
PRE_EDGES      = (15.0, 45.0)        # s into the round, half-open [lo, hi)
POST_EARLY_S   = 20.0                # s since plant, early is <= this
ALIVE_CLIP     = 3
AFFECTED_CLIP  = 5

LOS_TYPES      = ("peek", "hold")
AFFECTED_TYPES = ("flash", "he", "molotov")
PLANT_TYPES    = ("plant_start", "plant_complete")
DEFUSE_TYPES   = ("defuse_start", "defuse_complete", "defuse_abort")

# params key holding (from, to) callouts per type; None = fall back / no value
_CALLOUT_KEYS: dict[str, tuple[str | None, str | None]] = {
    "kill":           ("attacker_callout", "victim_callout"),
    "team_kill":      ("attacker_callout", "victim_callout"),
    "peek":           ("from_callout", "target_callout"),
    "flash":          ("throw_callout", "land_callout"),
    "he":             ("throw_callout", "land_callout"),
    "smoke":          ("throw_callout", "land_callout"),
    "molotov":        ("throw_callout", "land_callout"),
    "decoy":          ("throw_callout", "land_callout"),
    "rotation":       ("from_callout", "to_callout"),
    "save":           ("from_callout", "to_callout"),
    "hold":           ("callout", "facing_callout"),
    "plant_start":    ("planter_callout", None),
    "plant_complete": ("planter_callout", None),
    "bomb_drop":      ("callout", None),
    "bomb_pickup":    ("callout", None),
}
_NO_ACTOR_STATE = ("buy", "team_buy")


# ── helpers ───────────────────────────────────────────────────────────────────

def _actor(action, state_pre):
    if state_pre is None or action.actor_id_hash is None:
        return None, None
    i = state_pre.index_of(action.actor_id_hash)
    return (i, state_pre.players[i]) if i is not None else (None, None)


def _count(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple, dict)):
        return len(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
        return int(v)
    return None


def timing_bucket(state_pre) -> str | None:
    """Bucket of the state's clock (edges in the module docstring)."""
    if state_pre is None:
        return None
    c = state_pre.ctx
    if c.bomb_planted == 1:
        if c.time_since_plant_s is None:
            return None
        return "pp_early" if c.time_since_plant_s <= POST_EARLY_S else "pp_late"
    t = c.time_into_round_s
    if t is None or t != t:
        return None
    if t < PRE_EDGES[0]:
        return "0-15"
    if t < PRE_EDGES[1]:
        return "15-45"
    return "45-115"


def alive_diff_bucket(side: str | None, state_pre) -> int | None:
    if state_pre is None or side not in ("ct", "t"):
        return None
    c = state_pre.ctx
    own, enemy = (c.alive_ct, c.alive_t) if side == "ct" else (c.alive_t, c.alive_ct)
    if own is None or enemy is None:
        return None
    return int(max(-ALIVE_CLIP, min(ALIVE_CLIP, own - enemy)))


def has_los_to_enemy(action, state_pre) -> int | None:
    i, p = _actor(action, state_pre)
    if p is None or p.alive != 1:
        return None
    los = state_pre.pair[i, :, PAIR_CHANNELS.index("los")]
    vals = [los[j] for j, q in enumerate(state_pre.players) if j != i and q.side != p.side]
    known = [v for v in vals if v == v]
    if not known:
        return None
    return int(any(v == 1.0 for v in known))


# ── encode ────────────────────────────────────────────────────────────────────

def encode(action, state_pre) -> dict:
    """The P9.1.1 descriptor of `action` at `state_pre` (may be None: state fields are
    then None). Raises on a type outside Appendix A.5."""
    t = action.type
    if t not in TYPE_KIND:
        raise ValueError(f"action type {t!r} not in Appendix A.5")
    pr = action.params or {}
    _, actor = _actor(action, state_pre)

    weapon = pr.get("weapon_class")
    if weapon is None and t == "buy":
        weapon = pr.get("primary_class")
    if weapon is None and actor is not None and t not in _NO_ACTOR_STATE:
        weapon = actor.weapon_class

    k_from, k_to = _CALLOUT_KEYS.get(t, (None, None))
    frm = pr.get(k_from) if k_from else None
    if frm is None and actor is not None and t not in _NO_ACTOR_STATE:
        frm = actor.callout
    to = pr.get(k_to) if k_to else None
    if to is None and t == "damage" and state_pre is not None and pr.get("victim_id") is not None:
        j = state_pre.index_of(pr["victim_id"])
        to = state_pre.players[j].callout if j is not None else None

    affected = None
    if t == "flash":
        affected = _count(pr.get("blinded_enemies"))
    elif t in ("he", "molotov"):
        affected = _count(pr.get("enemies_damaged"))
    if affected is not None:
        affected = max(0, min(AFFECTED_CLIP, affected))

    loadout = pr.get("loadout_class") if t == "buy" else pr.get("team_class") if t == "team_buy" else None
    site = None
    if t in PLANT_TYPES:
        site = pr.get("site")
    elif t in DEFUSE_TYPES and state_pre is not None:
        site = state_pre.ctx.bomb_site

    return {
        "type":               t,
        "subtype":            action.subtype,
        "weapon_class":       weapon,
        "from_callout":       frm,
        "to_callout":         to,
        "timing_bucket":      timing_bucket(state_pre),
        "alive_diff_bucket":  alive_diff_bucket(action.actor_side, state_pre),
        "has_los_to_enemy":   has_los_to_enemy(action, state_pre) if t in LOS_TYPES else None,
        "n_enemies_affected": affected,
        "loadout_class":      loadout,
        "site":               site,
    }


def location(desc: dict) -> str | None:
    """The 'location' of the panel key: destination, else origin, else bomb site."""
    for f in ("to_callout", "from_callout", "site"):
        if desc.get(f) is not None:
            return str(desc[f])
    return None


def key(desc: dict) -> str:
    """Panel grouping key `type|subtype|location` (empty string for an unknown part)."""
    return f"{desc['type']}|{desc.get('subtype') or ''}|{location(desc) or ''}"


def token(v) -> str | None:
    """Vocabulary token of a descriptor value (None stays None -> UNK)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    return str(v)


# ── vocabulary ────────────────────────────────────────────────────────────────

class DescriptorVocab:
    """One token -> id map per field; UNK = 0, PAD = 1, real tokens from 2."""

    def __init__(self, maps: dict[str, dict[str, int]]):
        missing = [f for f in FIELDS if f not in maps]
        if missing:
            raise ValueError(f"vocab missing fields {missing}")
        self.maps = {f: dict(maps[f]) for f in FIELDS}

    @classmethod
    def fit(cls, descs, min_count: int = 1) -> "DescriptorVocab":
        counts: dict[str, dict[str, int]] = {f: {} for f in FIELDS}
        for d in descs:
            for f in FIELDS:
                tok = token(d.get(f))
                if tok is not None:
                    counts[f][tok] = counts[f].get(tok, 0) + 1
        maps = {f: {t: N_SPECIAL + i for i, t in enumerate(sorted(t for t, n in c.items() if n >= min_count))}
                for f, c in counts.items()}
        return cls(maps)

    @classmethod
    def fit_frame(cls, frame: pl.DataFrame, min_count: int = 1) -> "DescriptorVocab":
        """Fit from `desc_*` columns (as written by `frame_columns`)."""
        cols = [COLUMN_PREFIX + f for f in FIELDS]
        return cls.fit((dict(zip(FIELDS, r)) for r in frame.select(cols).iter_rows()), min_count)

    def sizes(self) -> dict[str, int]:
        """Embedding table size per field (tokens + UNK + PAD)."""
        return {f: len(self.maps[f]) + N_SPECIAL for f in FIELDS}

    def id(self, field: str, value) -> int:
        tok = token(value)
        return UNK if tok is None else self.maps[field].get(tok, UNK)

    def to_dict(self) -> dict:
        return {"fields": list(FIELDS), "unk": UNK, "pad": PAD, "maps": self.maps}

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "DescriptorVocab":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(doc.get("fields", ())) != FIELDS or doc.get("unk") != UNK or doc.get("pad") != PAD:
            raise ValueError(f"{path}: descriptor vocab does not match FIELDS / UNK / PAD of this build")
        return cls(doc["maps"])

    def __eq__(self, other) -> bool:
        return isinstance(other, DescriptorVocab) and self.maps == other.maps


def to_ids(desc: dict, vocab: DescriptorVocab) -> np.ndarray:
    """int64[len(FIELDS)] ids in FIELDS order."""
    return np.array([vocab.id(f, desc.get(f)) for f in FIELDS], dtype=np.int64)


def ids_from_frame(frame: pl.DataFrame, vocab: DescriptorVocab) -> np.ndarray:
    """int64[n, len(FIELDS)] from the frame's `desc_*` columns."""
    out = np.empty((frame.height, len(FIELDS)), dtype=np.int64)
    for j, f in enumerate(FIELDS):
        m = vocab.maps[f]
        out[:, j] = [UNK if v is None else m.get(v, UNK) for v in frame[COLUMN_PREFIX + f].to_list()]
    return out


# ── frame columns (joined onto actions_valued by action_id) ───────────────────

def frame_columns(actions, states_pre) -> pl.DataFrame:
    """`action_id`, `descriptor_key` and one `desc_<field>` String column (tokens) per
    field for parallel lists of actions and their `state_pre` snapshots (None allowed)."""
    rows = []
    for a, s in zip(actions, states_pre):
        d = encode(a, s)
        row = {"action_id": a.action_id, "descriptor_key": key(d)}
        row.update({COLUMN_PREFIX + f: token(d[f]) for f in FIELDS})
        rows.append(row)
    schema = {"action_id": pl.String, "descriptor_key": pl.String,
              **{COLUMN_PREFIX + f: pl.String for f in FIELDS}}
    return pl.DataFrame(rows, schema=schema)
