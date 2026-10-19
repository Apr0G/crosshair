"""
State schema (Amendment 1 contract): `PlayerToken`, `UtilityToken`, `ContextToken`,
`Snapshot` dataclasses (fields exactly Appendix A.2-A.4; `pair: np.ndarray[10,10,5]`),
`Snapshot.to_row()/from_row()/validate()`, column dtypes, null policy, schema version,
and `LIVE_SAFE` (every feature with its live-safety justification, P3.6.2).

Conventions:
- Unknown is `None`, never 0 or "?" (CLAUDE.md). In `pair` unknown is NaN.
- Flags are ints 0/1 (or None), not bools, so tokens map 1:1 onto model inputs.
- Positions are raw Hammer units and `yaw`/`pitch` raw degrees; normalization to map
  bounds and sin/cos encoding happen in the model (engine/model/tokens.py), so the
  pairwise table and the interventions can work in world units.
- `players` is always 10 long: CT first, then T, each team sorted by SteamID.
- Labels live in `Snapshot.labels` with bare keys (`round_won_ct`); rows carry them
  as `y_<key>` columns, which are dropped at inference by that prefix.

Filled by: P3.8.1, P3.6.2, P7.1 (validate).
"""
import copy as _copy
import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import polars as pl

# ── constants ─────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 1
PAIR_CHANNELS  = ("los", "dist_units", "nav_dist_units", "smoke_between", "same_team")
N_PLAYERS      = 10
N_PER_SIDE     = 5
MAX_UTILITY    = 16
SIDES          = ("ct", "t")
UTILITY_KINDS  = ("smoke", "fire", "decoy")
LABEL_PREFIX   = "y_"
CTX_PREFIX     = "ctx_"

# player fields that must be None when the player is dead (A.2 "null when dead")
DEAD_NULL_FIELDS = ("yaw", "pitch", "speed", "has_helmet", "has_defuser", "has_c4",
                    "weapon_class", "n_smoke", "n_flash", "n_he", "n_molly", "n_decoy",
                    "money", "equipment_value", "spotted_by_count", "heard_enemy")
# player fields that are 0 (a legitimate value) when dead
DEAD_ZERO_FIELDS = ("health", "armor")


# ── tokens (fields and order exactly Appendix A.2 - A.4) ──────────────────────

@dataclass
class PlayerToken:
    side:                             str
    alive:                            int
    x:                                float | None
    y:                                float | None
    z:                                float | None
    nav_area:                         int | None
    callout:                          str | None
    grid_cell:                        int | None
    yaw:                              float | None
    pitch:                            float | None
    speed:                            float | None
    health:                           float
    armor:                            float
    has_helmet:                       int | None
    has_defuser:                      int | None
    has_c4:                           int | None
    weapon_class:                     str | None
    n_smoke:                          int | None
    n_flash:                          int | None
    n_he:                             int | None
    n_molly:                          int | None
    n_decoy:                          int | None
    is_scoped:                        int | None
    is_ducking:                       int | None
    is_walking:                       int | None
    is_reloading:                     int | None
    is_defusing:                      int | None
    is_planting:                      int | None
    flash_duration_remaining:         float
    in_smoke:                         int | None
    on_fire:                          int | None
    money:                            float | None
    equipment_value:                  float | None
    seconds_since_last_seen_by_enemy: float | None
    seconds_since_last_shot:          float | None
    spotted_by_count:                 int | None
    heard_enemy:                      int | None
    kills_round:                      float
    damage_round:                     float
    player_id:                        int
    tier:                             str


@dataclass
class UtilityToken:
    kind:         str
    x:            float
    y:            float
    z:            float
    time_left_s:  float | None
    owner_side:   str | None
    owner_id:     int | None
    radius_units: float | None


@dataclass
class ContextToken:
    map_version:        str | None
    tier:               str
    time_remaining_s:   float
    time_into_round_s:  float
    bomb_planted:       int
    bomb_site:          str | None
    time_since_plant_s: float | None
    defuse_in_progress: int
    defuse_with_kit:    int | None
    bomb_carrier_alive: int | None
    alive_ct:           int
    alive_t:            int
    equip_ct:           float | None
    equip_t:            float | None
    money_ct:           float | None
    money_t:            float | None
    loss_streak_ct:     int
    loss_streak_t:      int
    score_ct:           int
    score_t:            int
    round_num:          int
    half:               int
    is_overtime:        int
    n_active_smokes:    int
    n_active_fires:     int


@dataclass
class SnapshotMeta:
    match_key: str
    round_num: int
    tick:      int
    t_s:       float
    kinds:     tuple[str, ...] = ()

    def __post_init__(self):
        self.kinds = tuple(self.kinds)


PLAYER_FIELDS  = tuple(f.name for f in fields(PlayerToken))
UTILITY_FIELDS = tuple(f.name for f in fields(UtilityToken))
CTX_FIELDS     = tuple(f.name for f in fields(ContextToken))
META_FIELDS    = tuple(f.name for f in fields(SnapshotMeta))


# ── polars dtypes (P3.8.1) ────────────────────────────────────────────────────

def _pl_dtype(annotation) -> pl.DataType:
    text = str(annotation)
    if "str" in text:
        return pl.String
    if "float" in text:
        return pl.Float32
    return pl.Int64


PLAYER_DTYPES  = {f.name: _pl_dtype(f.type) for f in fields(PlayerToken)}
UTILITY_DTYPES = {f.name: _pl_dtype(f.type) for f in fields(UtilityToken)}
CTX_DTYPES     = {f.name: _pl_dtype(f.type) for f in fields(ContextToken)}

# every non-label column of states.parquet; label columns are `y_*` with inferred dtype
STATE_COLUMNS: dict[str, pl.DataType] = {
    "schema_version": pl.Int32,
    "match_key":      pl.String,
    "round_num":      pl.Int32,
    "tick":           pl.Int64,
    "t_s":            pl.Float64,
    "kinds":          pl.List(pl.String),
    "players":        pl.List(pl.Struct(PLAYER_DTYPES)),
    "utility":        pl.List(pl.Struct(UTILITY_DTYPES)),
    **{CTX_PREFIX + k: v for k, v in CTX_DTYPES.items()},
    "pair":           pl.Array(pl.Float32, N_PLAYERS * N_PLAYERS * len(PAIR_CHANNELS)),
}


# ── Snapshot ──────────────────────────────────────────────────────────────────

def _nan_pair() -> np.ndarray:
    return np.full((N_PLAYERS, N_PLAYERS, len(PAIR_CHANNELS)), np.nan, dtype=np.float32)


def _strip_label(key: str) -> str:
    return key[len(LABEL_PREFIX):] if key.startswith(LABEL_PREFIX) else key


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@dataclass(eq=False)
class Snapshot:
    players: list[PlayerToken]
    utility: list[UtilityToken]
    ctx:     ContextToken
    pair:    np.ndarray = field(default_factory=_nan_pair)
    meta:    SnapshotMeta | None = None
    labels:  dict | None = None

    def __post_init__(self):
        self.pair = np.asarray(self.pair, dtype=np.float32)
        if self.labels is not None:
            self.labels = {_strip_label(k): v for k, v in self.labels.items()}

    # convenience reads used by test rules (`StubPredictor(win=lambda s, h: s.alive_ct ...)`)
    @property
    def alive_ct(self) -> int:
        return self.ctx.alive_ct

    @property
    def alive_t(self) -> int:
        return self.ctx.alive_t

    def index_of(self, player_id: int) -> int | None:
        """Slot of the player with this `player_id`, or None."""
        for i, p in enumerate(self.players):
            if p.player_id == player_id:
                return i
        return None

    def copy(self) -> "Snapshot":
        """Deep copy: interventions edit copies, never the original."""
        return _copy.deepcopy(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Snapshot):
            return NotImplemented
        return (self.players == other.players and self.utility == other.utility
                and self.ctx == other.ctx and self.meta == other.meta
                and self.pair.shape == other.pair.shape
                and bool(np.array_equal(self.pair, other.pair, equal_nan=True))
                and _same(self.labels, other.labels))

    # ── rows (parquet) ──

    def to_row(self) -> dict:
        """Flat, parquet-friendly dict: meta fields, `players`/`utility` as lists of
        dicts, ctx fields as `ctx_*`, `pair` as 500 floats (NaN -> None), labels as
        `y_*`, and `schema_version`."""
        meta = self.meta or SnapshotMeta(match_key="", round_num=self.ctx.round_num, tick=-1, t_s=float("nan"))
        row = {
            "schema_version": SCHEMA_VERSION,
            "match_key":      meta.match_key,
            "round_num":      meta.round_num,
            "tick":           meta.tick,
            "t_s":            meta.t_s,
            "kinds":          list(meta.kinds),
            "players":        [asdict(p) for p in self.players],
            "utility":        [asdict(u) for u in self.utility],
        }
        for k, v in asdict(self.ctx).items():
            row[CTX_PREFIX + k] = v
        row["pair"] = [None if math.isnan(v) else float(v) for v in self.pair.reshape(-1).tolist()]
        for k, v in (self.labels or {}).items():
            row[LABEL_PREFIX + k] = v
        return row

    @classmethod
    def from_row(cls, row: dict) -> "Snapshot":
        version = row.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"row has schema_version {version}, this code reads {SCHEMA_VERSION}")
        pair = np.array([np.nan if v is None else v for v in row["pair"]], dtype=np.float32)
        pair = pair.reshape(N_PLAYERS, N_PLAYERS, len(PAIR_CHANNELS))
        meta = SnapshotMeta(match_key=row["match_key"], round_num=row["round_num"],
                            tick=row["tick"], t_s=row["t_s"], kinds=tuple(row.get("kinds") or ()))
        labels = {k[len(LABEL_PREFIX):]: v for k, v in row.items() if k.startswith(LABEL_PREFIX)}
        if all(v is None for v in labels.values()):
            labels = None
        return cls(
            players=[PlayerToken(**{k: p[k] for k in PLAYER_FIELDS}) for p in row["players"]],
            utility=[UtilityToken(**{k: u[k] for k in UTILITY_FIELDS}) for u in (row.get("utility") or [])],
            ctx=ContextToken(**{k: row[CTX_PREFIX + k] for k in CTX_FIELDS}),
            pair=pair, meta=meta, labels=labels)

    # ── P7.1 consistency checker ──

    def validate(self) -> list[str]:
        """Derived-field consistency (P7.1). Returns one line per problem; empty = ok."""
        problems: list[str] = []
        ps, c = self.players, self.ctx

        if len(ps) != N_PLAYERS:
            problems.append(f"players: {len(ps)} tokens, expected {N_PLAYERS}")
        sides = [p.side for p in ps]
        for s in SIDES:
            if sides.count(s) != N_PER_SIDE:
                problems.append(f"players: {sides.count(s)} on side {s}, expected {N_PER_SIDE}")
        counts_ok = all(sides.count(s) == N_PER_SIDE for s in SIDES)
        if counts_ok and sides != ["ct"] * N_PER_SIDE + ["t"] * N_PER_SIDE:
            problems.append("players: order must be CT first, then T")

        for i, p in enumerate(ps):
            if p.alive not in (0, 1):
                problems.append(f"player {i}: alive={p.alive!r}, expected 0/1")
            if p.player_id is None or p.tier is None:
                problems.append(f"player {i}: player_id and tier are never null")
            if p.kills_round is None or p.damage_round is None:
                problems.append(f"player {i}: kills_round/damage_round are never null")
            if p.alive == 0:
                bad = [f for f in DEAD_NULL_FIELDS if getattr(p, f) is not None]
                bad += [f for f in DEAD_ZERO_FIELDS if getattr(p, f) != 0]
                if bad:
                    problems.append(f"player {i}: dead but has {bad}")
            elif p.alive == 1:
                bad = [f for f in ("in_smoke", "on_fire", "flash_duration_remaining")
                       if getattr(p, f) is None]
                if bad:
                    problems.append(f"player {i}: alive but {bad} null")

        alive = {s: [p for p in ps if p.side == s and p.alive == 1] for s in SIDES}
        for s, got in (("ct", c.alive_ct), ("t", c.alive_t)):
            if got != len(alive[s]):
                problems.append(f"ctx.alive_{s}={got} but {len(alive[s])} alive {s} players")
        for s, got in (("ct", c.equip_ct), ("t", c.equip_t)):
            vals = [p.equipment_value for p in alive[s]]
            if all(v is not None for v in vals):
                want = float(sum(vals))
                if got is None or abs(got - want) > 1e-3:
                    problems.append(f"ctx.equip_{s}={got} but alive {s} players sum to {want}")

        if c.time_remaining_s is None or c.time_remaining_s < 0:
            problems.append(f"ctx.time_remaining_s={c.time_remaining_s} must be >= 0")
        planted = c.bomb_planted == 1
        if planted != (c.bomb_site is not None):
            problems.append(f"ctx.bomb_site={c.bomb_site!r} with bomb_planted={c.bomb_planted}")
        if planted != (c.time_since_plant_s is not None):
            problems.append(f"ctx.time_since_plant_s={c.time_since_plant_s} with bomb_planted={c.bomb_planted}")
        if planted and c.bomb_carrier_alive is not None:
            problems.append("ctx.bomb_carrier_alive must be null post-plant")
        if c.defuse_in_progress != 1 and c.defuse_with_kit is not None:
            problems.append("ctx.defuse_with_kit must be null when no defuse is in progress")

        if len(self.utility) > MAX_UTILITY:
            problems.append(f"utility: {len(self.utility)} tokens > {MAX_UTILITY}")
        for j, u in enumerate(self.utility):
            if u.kind not in UTILITY_KINDS:
                problems.append(f"utility {j}: kind {u.kind!r} not in {UTILITY_KINDS}")
        n_smokes = sum(u.kind == "smoke" for u in self.utility)
        n_fires  = sum(u.kind == "fire" for u in self.utility)
        if len(self.utility) < MAX_UTILITY and (c.n_active_smokes, c.n_active_fires) != (n_smokes, n_fires):
            problems.append(f"ctx.n_active_smokes/fires=({c.n_active_smokes},{c.n_active_fires}) "
                            f"but tokens give ({n_smokes},{n_fires})")

        problems += self._validate_pair()
        return problems

    def _validate_pair(self) -> list[str]:
        want = (N_PLAYERS, N_PLAYERS, len(PAIR_CHANNELS))
        if self.pair.shape != want:
            return [f"pair: shape {self.pair.shape}, expected {want}"]
        problems = []
        pr = self.pair
        for k, ch in enumerate(PAIR_CHANNELS):
            m = pr[:, :, k]
            asym = ~((m == m.T) | (np.isnan(m) & np.isnan(m.T)))
            if asym.any():
                problems.append(f"pair.{ch}: not symmetric at {int(asym.sum()) // 2} pair(s)")
        diag = pr[np.arange(N_PLAYERS), np.arange(N_PLAYERS), :]
        if not np.isnan(diag).all():
            problems.append("pair: diagonal must be NaN")
        if len(self.players) == N_PLAYERS:
            iu = np.triu_indices(N_PLAYERS, k=1)
            st  = pr[:, :, PAIR_CHANNELS.index("same_team")][iu]
            los = pr[:, :, PAIR_CHANNELS.index("los")][iu]
            same = np.array([self.players[i].side == self.players[j].side for i, j in zip(*iu)])
            if not np.array_equal(st, same.astype(np.float32)):
                problems.append("pair.same_team: disagrees with player sides")
            dead = np.array([self.players[i].alive == 0 or self.players[j].alive == 0 for i, j in zip(*iu)])
            if (~np.isnan(los[dead])).any():
                problems.append("pair.los: must be NaN when either player is dead")
        for ch in ("los", "smoke_between", "same_team"):
            m = pr[:, :, PAIR_CHANNELS.index(ch)]
            vals = m[~np.isnan(m)]
            if not np.isin(vals, (0.0, 1.0)).all():
                problems.append(f"pair.{ch}: values must be 0/1/NaN")
        return problems


# ── frames ────────────────────────────────────────────────────────────────────

def rows_to_frame(rows: list[dict]) -> pl.DataFrame:
    """Rows from `Snapshot.to_row()` -> one DataFrame with the STATE_COLUMNS dtypes."""
    if not rows:
        return pl.DataFrame(schema=STATE_COLUMNS)
    label_cols = sorted({k for r in rows for k in r if k.startswith(LABEL_PREFIX)})
    cols = [*STATE_COLUMNS, *label_cols]
    data = {k: [r.get(k) for r in rows] for k in cols}
    df = pl.DataFrame({k: data[k] for k in label_cols}) if label_cols else None
    base = pl.DataFrame({k: data[k] for k in STATE_COLUMNS}, schema=STATE_COLUMNS)
    return base.hstack(df) if df is not None else base


def frame_to_snapshots(df: pl.DataFrame) -> list[Snapshot]:
    return [Snapshot.from_row(r) for r in df.iter_rows(named=True)]


# ── P3.6.2 live-safety justifications ─────────────────────────────────────────
# Keys are "<group>.<field>" for group in player / utility / ctx / pair.

LIVE_SAFE: dict[str, str] = {
    # player (A.2)
    "player.side":                             "known: team assignment is public all round",
    "player.alive":                            "known: deaths are announced as they happen",
    "player.x":                                "current position at the tick",
    "player.y":                                "current position at the tick",
    "player.z":                                "current position at the tick",
    "player.nav_area":                         "current: nav area of the current position",
    "player.callout":                          "current: callout of the current position",
    "player.grid_cell":                        "current: grid cell of the current position",
    "player.yaw":                              "current view angle",
    "player.pitch":                            "current view angle",
    "player.speed":                            "current velocity magnitude",
    "player.health":                           "current health",
    "player.armor":                            "current armor",
    "player.has_helmet":                       "current loadout",
    "player.has_defuser":                      "current loadout",
    "player.has_c4":                           "current loadout",
    "player.weapon_class":                     "current active weapon",
    "player.n_smoke":                          "current inventory",
    "player.n_flash":                          "current inventory",
    "player.n_he":                             "current inventory",
    "player.n_molly":                          "current inventory",
    "player.n_decoy":                          "current inventory",
    "player.is_scoped":                        "current per-tick prop",
    "player.is_ducking":                       "current per-tick prop",
    "player.is_walking":                       "current per-tick prop",
    "player.is_reloading":                     "current per-tick prop",
    "player.is_defusing":                      "current per-tick prop",
    "player.is_planting":                      "current per-tick prop",
    "player.flash_duration_remaining":         "current: blind time left from a flash that already popped",
    "player.in_smoke":                         "derived from current position and active smoke tokens",
    "player.on_fire":                          "derived from current position and active fire tokens",
    "player.money":                            "current account balance",
    "player.equipment_value":                  "current loadout value",
    "player.seconds_since_last_seen_by_enemy": "past only: last spotted tick <= t",
    "player.seconds_since_last_shot":          "past only: last weapon_fire tick <= t",
    "player.spotted_by_count":                 "current spotted props",
    "player.heard_enemy":                      "past window only: sound cues with tick <= t",
    "player.kills_round":                      "past only: kills so far this round",
    "player.damage_round":                     "past only: damage dealt so far this round",
    "player.player_id":                        "identity: hashed SteamID",
    "player.tier":                             "metadata: match tier, known before the round",
    # utility (A.3)
    "utility.kind":                            "current: the grenade type of an active cloud/fire",
    "utility.x":                               "current: where the active utility is",
    "utility.y":                               "current: where the active utility is",
    "utility.z":                               "current: where the active utility is",
    "utility.time_left_s":                     "nominal lifetime minus elapsed time, not the real end tick",
    "utility.owner_side":                      "known: side of the thrower",
    "utility.owner_id":                        "known: identity of the thrower",
    "utility.radius_units":                    "current: nominal smoke radius / current fire extent",
    # context (A.4)
    "ctx.map_version":                         "metadata: known before the match",
    "ctx.tier":                                "metadata: known before the match",
    "ctx.time_remaining_s":                    "live round clock from freeze_end / plant tick, never the end tick",
    "ctx.time_into_round_s":                   "elapsed since freeze_end",
    "ctx.bomb_planted":                        "current: plant_complete tick <= t",
    "ctx.bomb_site":                           "current: site of a plant that already happened",
    "ctx.time_since_plant_s":                  "elapsed since a plant that already happened",
    "ctx.defuse_in_progress":                  "current per-tick prop",
    "ctx.defuse_with_kit":                     "current: the defuser's loadout",
    "ctx.bomb_carrier_alive":                  "current: carrier's alive state",
    "ctx.alive_ct":                            "current count of alive players",
    "ctx.alive_t":                             "current count of alive players",
    "ctx.equip_ct":                            "current sum of alive players' loadouts",
    "ctx.equip_t":                             "current sum of alive players' loadouts",
    "ctx.money_ct":                            "current sum of accounts",
    "ctx.money_t":                             "current sum of accounts",
    "ctx.loss_streak_ct":                      "past rounds only, entering this round",
    "ctx.loss_streak_t":                       "past rounds only, entering this round",
    "ctx.score_ct":                            "past rounds only, entering this round",
    "ctx.score_t":                             "past rounds only, entering this round",
    "ctx.round_num":                           "known: round counter",
    "ctx.half":                                "known: from the round counter",
    "ctx.is_overtime":                         "known: from the round counter",
    "ctx.n_active_smokes":                     "current count of active smoke tokens",
    "ctx.n_active_fires":                      "current count of active fire tokens",
    # pairwise (P3.5)
    "pair.los":                                "current positions and map geometry",
    "pair.dist_units":                         "current positions",
    "pair.nav_dist_units":                     "current nav areas",
    "pair.smoke_between":                      "current positions and active smoke tokens",
    "pair.same_team":                          "known: team assignment",
}


def live_safe_keys() -> list[str]:
    """Every key LIVE_SAFE must contain."""
    return ([f"player.{f}" for f in PLAYER_FIELDS] + [f"utility.{f}" for f in UTILITY_FIELDS]
            + [f"ctx.{f}" for f in CTX_FIELDS] + [f"pair.{c}" for c in PAIR_CHANNELS])


def missing_live_safe() -> list[str]:
    """Problems with LIVE_SAFE: features without a justification, empty
    justifications, and stale keys that name no feature. Empty = ok."""
    want = live_safe_keys()
    out  = [f"no justification: {k}" for k in want if not LIVE_SAFE.get(k, "").strip()]
    out += [f"stale key: {k}" for k in LIVE_SAFE if k not in want]
    return out
