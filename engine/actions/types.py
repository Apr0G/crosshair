"""
`Action` dataclass exactly as P4.1.1 (deterministic `action_id` = sha1 of
`match_key|round|type|actor|tick`) and the type -> extractor registry (P4.1.2).

    a = Action(match_key="m1", round_num=1, actor_id_hash=123, actor_side="ct",
               kind="instant", type="kill", tick=2000, params={"weapon": "ak47"})
    a.action_id              # 16 hex chars, same inputs -> same id on every run
    a.validate()             # [] when kind / type / timing agree

    @register("kill")
    def extract_kills(raw, states, ctx) -> list[Action]: ...

Two actions of one type by one actor on one tick (an HE damaging two victims) would
share the default id; the extractor passes `salt` (e.g. the victim id) to
`make_action_id` and sets `action_id` itself. `assert_unique_ids` catches misses.

Filled by: P4.1.1, P4.1.2.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from typing import Callable

# ── vocabulary (Appendix A.5) ─────────────────────────────────────────────────

KINDS = ("instant", "durative", "round")
SIDES = ("ct", "t")

TYPE_KIND: dict[str, str] = {
    "kill":            "instant",
    "team_kill":       "instant",
    "damage":          "instant",
    "flash":           "instant",
    "he":              "instant",
    "peek":            "instant",
    "plant_start":     "instant",
    "plant_complete":  "instant",
    "defuse_start":    "instant",
    "defuse_complete": "instant",
    "defuse_abort":    "instant",
    "bomb_drop":       "instant",
    "bomb_pickup":     "instant",
    "drop":            "instant",
    "smoke":           "durative",
    "molotov":         "durative",
    "decoy":           "durative",
    "rotation":        "durative",
    "hold":            "durative",
    "save":            "durative",
    "buy":             "round",
    "team_buy":        "round",
    # pseudo-actions emitted by extractor.group_windows (P4.6.1), not by extractors
    "smoke_pop":       "instant",
    "smoke_fade":      "instant",
    "fire_start":      "instant",
    "fire_end":        "instant",
    "decoy_start":     "instant",
    "decoy_end":       "instant",
}

PSEUDO_TYPES = ("smoke_pop", "smoke_fade", "fire_start", "fire_end", "decoy_start", "decoy_end")
# instant types that carry a window as metadata (P4.2.2: damage engagement span)
WINDOW_METADATA_TYPES = ("damage",)
# team-level actions with no single actor
TEAM_TYPES = ("team_buy",)

_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def make_action_id(match_key: str, round_num: int, type_: str, actor_id_hash: int | None,
                   tick: int | None, salt: str | None = None) -> str:
    """sha1 of `match_key|round_num|type|actor_id_hash|tick` (plus `|salt` when given),
    first 16 hex chars."""
    key = f"{match_key}|{round_num}|{type_}|{actor_id_hash}|{tick}"
    if salt is not None:
        key += f"|{salt}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ── Action ────────────────────────────────────────────────────────────────────

@dataclass(kw_only=True)
class Action:
    action_id:         str | None = None
    match_key:         str
    round_num:         int
    actor_id_hash:     int | None
    actor_side:        str
    kind:              str
    type:              str
    subtype:           str | None = None
    tick:              int | None = None
    window_start_tick: int | None = None
    window_end_tick:   int | None = None
    params:            dict = field(default_factory=dict)
    enablers:          list[dict] = field(default_factory=list)
    state_pre_tick:    int | None = None
    state_post_tick:   int | None = None

    def __post_init__(self):
        if self.action_id is None:
            self.action_id = make_action_id(self.match_key, self.round_num, self.type,
                                            self.actor_id_hash, self.id_tick)

    @property
    def id_tick(self) -> int | None:
        """The tick that goes into the id: `tick`, else the window start (durative)."""
        return self.tick if self.tick is not None else self.window_start_tick

    # ── checks ──

    def validate(self) -> list[str]:
        """Kind matches type, timing fields agree with the kind, params JSON-safe."""
        p: list[str] = []
        if not isinstance(self.action_id, str) or not _HEX16.match(self.action_id):
            p.append(f"action_id {self.action_id!r} is not 16 hex chars")
        if self.kind not in KINDS:
            p.append(f"kind {self.kind!r} not in {KINDS}")
        if self.type not in TYPE_KIND:
            p.append(f"type {self.type!r} not in Appendix A.5")
        elif self.kind in KINDS and TYPE_KIND[self.type] != self.kind:
            p.append(f"type {self.type!r} is {TYPE_KIND[self.type]}, got kind {self.kind!r}")
        if self.actor_side not in SIDES:
            p.append(f"actor_side {self.actor_side!r} not in {SIDES}")
        if self.actor_id_hash is None and self.type not in TEAM_TYPES:
            p.append(f"actor_id_hash is null for non-team type {self.type!r}")

        ws, we = self.window_start_tick, self.window_end_tick
        if self.kind == "instant":
            if self.tick is None:
                p.append("instant action without tick")
            if self.type in WINDOW_METADATA_TYPES:
                if (ws is None) != (we is None):
                    p.append("window metadata needs both window_start_tick and window_end_tick")
                elif ws is not None and not (ws == self.tick and we >= ws):
                    p.append(f"window metadata [{ws}, {we}] must start at tick {self.tick}")
            elif ws is not None or we is not None:
                p.append("instant action must not have a window")
        elif self.kind == "durative":
            if ws is None or we is None:
                p.append("durative action needs window_start_tick and window_end_tick")
            elif ws > we:
                p.append(f"window_start_tick {ws} > window_end_tick {we}")
            if self.tick is not None and self.tick != ws:
                p.append(f"durative tick {self.tick} must be null or equal window_start_tick {ws}")
        elif self.kind == "round":
            if self.tick is None:
                p.append("round action without tick")
            if ws is not None or we is not None:
                p.append("round action must not have a window")

        if (self.state_pre_tick is not None and self.state_post_tick is not None
                and self.state_pre_tick > self.state_post_tick):
            p.append(f"state_pre_tick {self.state_pre_tick} > state_post_tick {self.state_post_tick}")
        try:
            json.dumps(self.params, allow_nan=False)
        except (TypeError, ValueError) as e:
            p.append(f"params not JSON-safe: {e}")
        for e in self.enablers:
            if not isinstance(e, dict) or "action_id" not in e or "kind" not in e:
                p.append(f"enabler {e!r} must be a dict with action_id and kind")
        return p

    # ── serialization ──

    def to_dict(self, flatten: bool = False) -> dict:
        """JSON-safe dict. With `flatten`, `params` and `enablers` are JSON strings
        (sorted keys) so the row fits a flat parquet / SQLite table."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["params"]   = dict(self.params)
        d["enablers"] = [dict(e) for e in self.enablers]
        if flatten:
            d["params"]   = json.dumps(d["params"], sort_keys=True, allow_nan=False)
            d["enablers"] = json.dumps(d["enablers"], sort_keys=True, allow_nan=False)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        """Inverse of `to_dict` for either flatten mode."""
        d = {k: d[k] for k in (f.name for f in fields(cls)) if k in d}
        for k in ("params", "enablers"):
            if isinstance(d.get(k), str):
                d[k] = json.loads(d[k])
        if d.get("params") is None:
            d["params"] = {}
        if d.get("enablers") is None:
            d["enablers"] = []
        return cls(**d)


def assert_unique_ids(actions: list[Action]) -> None:
    seen: dict[str, Action] = {}
    for a in actions:
        if a.action_id in seen:
            raise ValueError(f"duplicate action_id {a.action_id}: {seen[a.action_id].type} and {a.type} "
                             f"at tick {a.id_tick}; pass a salt to make_action_id")
        seen[a.action_id] = a


# ── registry (P4.1.2) ─────────────────────────────────────────────────────────

REGISTRY: dict[str, Callable] = {}


def register(type_name: str) -> Callable[[Callable], Callable]:
    """Decorator: `@register("kill")` records the extractor for that type. Unknown
    types and a second, different extractor for one type are errors."""
    if type_name not in TYPE_KIND:
        raise ValueError(f"cannot register unknown action type {type_name!r}")
    if type_name in PSEUDO_TYPES:
        raise ValueError(f"{type_name!r} is a pseudo type made by group_windows, not an extractor")

    def deco(fn: Callable) -> Callable:
        old = REGISTRY.get(type_name)
        if old is not None and (old.__module__, old.__qualname__) != (fn.__module__, fn.__qualname__):
            raise ValueError(f"type {type_name!r} already registered to {old.__module__}.{old.__qualname__}")
        REGISTRY[type_name] = fn
        return fn
    return deco


def missing_extractors() -> list[str]:
    """Non-pseudo types in Appendix A.5 with no registered extractor."""
    return [t for t in TYPE_KIND if t not in PSEUDO_TYPES and t not in REGISTRY]
