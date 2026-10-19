"""
Synthetic valued-action corpora and a toy embedding for the P9 value tests and bench.

    from _value_states import synthetic_corpus, split_by_match, toy_embed, device
    frame, h = synthetic_corpus(10_000, seed=1)       # actions_valued-like frame + h_pre rows
    labels   = split_by_match(frame)                  # "train" / "val" / "test" per row
    h1       = toy_embed([snapshot])                  # same projection on real Snapshots

The toy embedding is a fixed random projection (seed TOY_SEED) of a few context
features (`ctx_features`) through tanh, so states with similar alive counts / clock /
bomb state have nearby vectors and the neighbour panels are meaningful. It never
touches the win model.

Corpus rows carry every column `scores.score_frame` reads (attribution FRAME_SCHEMA,
`tier`, `map_version`, `descriptor_key`, `desc_*`). True expected value:
`Q_true = TYPE_BASE[type] + 0.02 · alive_diff_bucket`; `V = Q_true + noise · N(0, 1)`
plus `planted[(player, type)]` for planted rows. `Q_true` is kept as a column.
"""
import sys

import numpy as np
import polars as pl

from engine.actions.types import make_action_id
from engine.value import descriptors

# ── constants ─────────────────────────────────────────────────────────────────

TOY_SEED = 1234
D_TOY    = 16
CTX_FEATURES = ("alive_ct", "alive_t", "time_into_round_s", "bomb_planted", "time_since_plant_s",
                "n_active_smokes")
_SCALE   = np.array([5.0, 5.0, 115.0, 1.0, 40.0, 4.0], dtype=np.float64)
_PROJ    = np.random.default_rng(TOY_SEED).normal(size=(len(CTX_FEATURES) + 1, D_TOY))

CALLOUTS = ("A Site", "B Site", "Mid", "Long", "Short", "Ramp", "Connector", "Palace")
WEAPONS  = ("rifle", "sniper", "smg", "pistol")
TYPE_P   = {   # type: (probability, side or None)
    "kill": (0.25, None), "peek": (0.14, None), "rotation": (0.14, None), "flash": (0.10, None),
    "smoke": (0.08, None), "hold": (0.07, None), "damage": (0.05, None), "he": (0.04, None),
    "molotov": (0.03, None), "buy": (0.03, None), "plant_complete": (0.025, "t"),
    "defuse_complete": (0.025, "ct"), "team_kill": (0.01, None),
}
TYPE_BASE = {"kill": 0.10, "team_kill": -0.10, "damage": 0.02, "flash": 0.01, "he": 0.005,
             "smoke": 0.0, "molotov": 0.005, "peek": -0.01, "rotation": 0.0, "hold": 0.0,
             "buy": 0.0, "plant_complete": 0.06, "defuse_complete": 0.08}
ALIVE_SLOPE = 0.02


def device() -> str:
    """"cuda" when available and `--cpu` is not on the command line, else "cpu"."""
    if "--cpu" in sys.argv:
        return "cpu"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── toy embedding ─────────────────────────────────────────────────────────────

def project(features: np.ndarray) -> np.ndarray:
    """float32[n, D_TOY] = tanh([features / scale, 1] @ PROJ)."""
    f = np.asarray(features, dtype=np.float64) / _SCALE
    f = np.hstack([f, np.ones((f.shape[0], 1))])
    return np.tanh(f @ _PROJ).astype(np.float32)


def ctx_features(s) -> list[float]:
    c = s.ctx
    return [c.alive_ct, c.alive_t, c.time_into_round_s, c.bomb_planted, c.time_since_plant_s or 0.0,
            c.n_active_smokes]


def toy_embed(snapshots, histories=None) -> np.ndarray:
    """`embed_fn(snapshots, histories)` for real Snapshots (histories ignored)."""
    return project(np.array([ctx_features(s) for s in snapshots], dtype=np.float64).reshape(-1, len(CTX_FEATURES)))


# ── corpus ────────────────────────────────────────────────────────────────────

def _timing(into: np.ndarray, planted: np.ndarray, since: np.ndarray) -> list[str]:
    out = []
    for t, p, s in zip(into, planted, since):
        if p:
            out.append("pp_early" if s <= descriptors.POST_EARLY_S else "pp_late")
        else:
            out.append("0-15" if t < 15 else "15-45" if t < 45 else "45-115")
    return out


def synthetic_corpus(n: int, *, seed: int = 0, n_matches: int = 50, n_players: int = 40,
                     maps=("de_test@a",), tier: str = "pro", noise: float = 0.03,
                     planted: dict | None = None, match_prefix: str = "m",
                     type_p: dict | None = None, callouts=CALLOUTS,
                     with_h_column: bool = False) -> tuple[pl.DataFrame, np.ndarray]:
    """`n` valued actions (see module docstring) and their h_pre rows."""
    rng = np.random.default_rng(seed)
    tp = type_p or TYPE_P
    names = list(tp)
    probs = np.array([tp[t][0] for t in names], dtype=np.float64)
    types = rng.choice(names, size=n, p=probs / probs.sum())
    side = np.where(rng.random(n) < 0.5, "ct", "t").astype(object)
    for t, (_, forced) in tp.items():
        if forced is not None:
            side[types == t] = forced
    players = rng.integers(0, n_players, size=n)
    actor = (1000 + players).astype(np.int64)            # a player acts on both sides
    match = rng.integers(0, n_matches, size=n)
    rnd = rng.integers(1, 25, size=n)
    alive_ct, alive_t = rng.integers(1, 6, size=n), rng.integers(1, 6, size=n)
    into = rng.uniform(0, 115, size=n)
    planted_b = (rng.random(n) < 0.25) | (types == "defuse_complete")
    planted_b &= types != "plant_complete"
    since = np.where(planted_b, rng.uniform(0, 40, size=n), 0.0)
    smokes = rng.integers(0, 4, size=n)
    feats = np.column_stack([alive_ct, alive_t, into, planted_b.astype(float), since, smokes])
    h = project(feats)

    own = np.where(side == "ct", alive_ct, alive_t)
    enemy = np.where(side == "ct", alive_t, alive_ct)
    adiff = np.clip(own - enemy, -3, 3)
    q_true = np.array([TYPE_BASE[t] for t in types]) + ALIVE_SLOPE * adiff
    v = q_true + noise * rng.normal(size=n)
    if planted:
        for (pid, t), shift in planted.items():
            v = v + shift * ((actor == pid) & (types == t))

    timing = _timing(into, planted_b, since)
    frm = rng.choice(list(callouts), size=n)
    to = rng.choice(list(callouts), size=n)
    weap = rng.choice(WEAPONS, size=n)
    site = rng.choice(["A", "B"], size=n)
    map_v = rng.choice(list(maps), size=n)
    rows = []
    for i in range(n):
        t = str(types[i])
        d = {"type": t, "subtype": None, "weapon_class": str(weap[i]), "from_callout": str(frm[i]),
             "to_callout": str(to[i]) if t not in ("buy", "plant_complete", "defuse_complete") else None,
             "timing_bucket": timing[i], "alive_diff_bucket": int(adiff[i]),
             "has_los_to_enemy": int(rng.random() < 0.5) if t in descriptors.LOS_TYPES else None,
             "n_enemies_affected": int(rng.integers(0, 3)) if t in descriptors.AFFECTED_TYPES else None,
             "loadout_class": "full" if t == "buy" else None,
             "site": str(site[i]) if t in ("plant_complete", "defuse_complete") else None}
        mk = f"{match_prefix}{match[i]}"
        sign = 1.0 if side[i] == "ct" else -1.0
        row = {"match_key": mk, "round_num": int(rnd[i]),
               "action_id": make_action_id(mk, int(rnd[i]), t, int(actor[i]), i, salt=f"{seed}"),
               "actor_id_hash": int(actor[i]), "actor_side": str(side[i]), "type": t,
               "V": float(v[i]), "V_raw": sign * float(v[i]), "V_direct": float(v[i]), "V_enabler": 0.0,
               "V_movement": 0.0, "method": "exact", "n_coalition": 1, "residual": 0.0, "flags": [],
               "components": "[]", "tier": tier, "map_version": str(map_v[i]),
               "descriptor_key": descriptors.key(d), "Q_true": float(q_true[i])}
        row.update({descriptors.COLUMN_PREFIX + f: descriptors.token(d[f]) for f in descriptors.FIELDS})
        rows.append(row)
    schema = {"match_key": pl.String, "round_num": pl.Int64, "action_id": pl.String, "actor_id_hash": pl.Int64,
              "actor_side": pl.String, "type": pl.String, "V": pl.Float64, "V_raw": pl.Float64,
              "V_direct": pl.Float64, "V_enabler": pl.Float64, "V_movement": pl.Float64, "method": pl.String,
              "n_coalition": pl.Int64, "residual": pl.Float64, "flags": pl.List(pl.String),
              "components": pl.String, "tier": pl.String, "map_version": pl.String,
              "descriptor_key": pl.String, "Q_true": pl.Float64,
              **{descriptors.COLUMN_PREFIX + f: pl.String for f in descriptors.FIELDS}}
    frame = pl.DataFrame(rows, schema=schema)
    if with_h_column:
        frame = frame.with_columns(pl.Series("h_pre", h.tolist(), dtype=pl.List(pl.Float32)))
    return frame, h


def split_by_match(frame: pl.DataFrame) -> list[str]:
    """Deterministic match-level split: match number % 10 -> 0-7 train, 8 val, 9 test."""
    out = []
    for mk in frame["match_key"].to_list():
        d = int("".join(ch for ch in mk if ch.isdigit()) or 0) % 10
        out.append("train" if d < 8 else "val" if d == 8 else "test")
    return out


def side_table(frame: pl.DataFrame, vocab) -> pl.DataFrame:
    """Neighbour side table (with desc_ids) for a corpus frame."""
    ids = descriptors.ids_from_frame(frame, vocab)
    return frame.select("match_key", "round_num", "action_id", "descriptor_key", "V",
                        pl.col("actor_side").alias("side"), "map_version", "tier").with_columns(
        pl.Series("desc_ids", ids.tolist(), dtype=pl.List(pl.Int64)))
