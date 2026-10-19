"""
States -> tensor shards per split (P5.3.1), normalization stats and vocabularies fitted
on train only (P5.3.2), optional 4 Hz ablation set (P5.3.3).

    from engine.data import dataset
    vocab = dataset.Vocab().fit(train_snaps)
    arrs  = dataset.tensorize(snaps, vocab)            # dict, keys exactly BATCH_KEYS
    norm  = dataset.NormStats.fit([arrs])              # train only
    dataset.write_shards("train", iter(snaps), 50_000, vocab=vocab)
    arrs  = dataset.read_shards("train")
    dataset.build(catalog, split_file, tiers=("pro",), maps=None, sample_hz=2)

Arrays are in world units. Numeric None/NaN becomes 0 plus a presence bit appended to
the numeric block (`FEATURE_LAYOUT[block]` names every column); `yaw`/`pitch` become
sin/cos. Normalization (`NormStats`) is applied inside the model (engine/model/tokens.py
buffers), not in the shards, so augmentation, QA and the LightGBM baseline read real
units (speed in u/s, health in hp, time in s). `tensorize(..., norm=...)` returns
normalized arrays for callers that want them.

Categoricals: index 0 = UNK (unseen in train), 1 = PAD (absent utility slot),
2 = NULL (the value is None: dead, off-map, no plant). Callouts and grid cells are keyed
per map version (`"<map_version>|<callout>"`). `nav_area` is not fed to the model: it
is redundant with `grid_cell` + `callout` + x/y/z, and a per-map-version embedding of
thousands of nav areas would dominate the parameter count (P6.2.5 estimate 2-4 M).

The pairwise table keeps NaN for unknown (the schema's convention); the model turns NaN
into 0 plus a presence channel.
"""
import argparse
import hashlib
import json
import math
import sys
from operator import attrgetter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from engine import paths
from engine.state.schema import (CTX_FIELDS, MAX_UTILITY, N_PLAYERS, PAIR_CHANNELS, ContextToken,
                                 PlayerToken, Snapshot, UtilityToken)

# ── constants ─────────────────────────────────────────────────────────────────

UNK, PAD, NULL = 0, 1, 2
RESERVED       = ("<unk>", "<pad>", "<null>")
PAD_PLAYER_ID  = -1           # player_id of an empty slot (engine/state/players.py), masked out
VOCAB_NAMES    = ("weapon", "callout", "grid", "player", "tier", "map_version", "side",
                  "bomb_site", "util_kind")

# P5.3.1, exactly; the model reads the first nine, the loss the labels.
SNAPSHOT_KEYS = ("player_num", "player_cat", "player_mask", "util_num", "util_cat", "util_mask",
                 "ctx_num", "ctx_cat", "pair")
LABEL_KEYS    = ("y_win", "y_hazard", "y_plant15", "y_defuse15", "y_tte", "is_terminal")
META_KEYS     = ("round_key", "t_s", "tier", "tick_kinds")
BATCH_KEYS    = SNAPSHOT_KEYS + LABEL_KEYS + META_KEYS

# Snapshot.labels keys written by engine/state/builder.py (P3.7.1)
SOURCE_LABELS = {"y_win": "round_won_ct", "y_hazard": "dies_within_5s",
                 "y_plant15": "plant_within_15s", "y_defuse15": "defuse_within_15s",
                 "y_tte": "time_to_next_engagement_s", "is_terminal": "is_terminal"}

KIND_BITS = {"grid": 1, "boundary_pre": 2, "boundary_post": 4, "terminal": 8}
OTHER_KIND_BIT = 16


# ── feature layout ────────────────────────────────────────────────────────────

def _nullable(cls, name: str) -> bool:
    return "None" in str(cls.__dataclass_fields__[name].type)


class BlockLayout:
    """One numeric block: value columns then one presence bit per nullable source field.

    `values` is a tuple of (column name, kind, source field); kind is `cont` (z-scored by
    NormStats), `flag` (0/1, raw) or `angle` (sin/cos, raw)."""

    def __init__(self, cls, values: tuple[tuple[str, str, str], ...]):
        self.values   = values
        sources       = []
        for _, _, src in values:
            if _nullable(cls, src) and src not in sources:
                sources.append(src)
        self.presence_sources = tuple(sources)
        self.names = tuple(n for n, _, _ in values) + tuple(f"present_{s}" for s in sources)
        self.kinds = tuple(k for _, k, _ in values) + ("presence",) * len(sources)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.n_values = len(values)
        # column of the presence bit for each value column (-1 = never null)
        self.presence_of = tuple(self.index[f"present_{src}"] if src in sources else -1
                                 for _, _, src in values)
        self.sources    = tuple(dict.fromkeys(src for _, _, src in values))
        self.src_index  = {src: i for i, src in enumerate(self.sources)}
        self.getter     = attrgetter(*self.sources)

    def __len__(self) -> int:
        return len(self.names)

    def col(self, name: str) -> int:
        return self.index[name]


_PLAYER_FLAGS = ("has_helmet", "has_defuser", "has_c4")
_PLAYER_STATE = ("is_scoped", "is_ducking", "is_walking", "is_reloading", "is_defusing", "is_planting")
_NADES        = ("n_smoke", "n_flash", "n_he", "n_molly", "n_decoy")

PLAYER_LAYOUT = BlockLayout(PlayerToken, (
    ("alive", "flag", "alive"),
    ("x", "cont", "x"), ("y", "cont", "y"), ("z", "cont", "z"),
    ("yaw_sin", "angle", "yaw"), ("yaw_cos", "angle", "yaw"),
    ("pitch_sin", "angle", "pitch"), ("pitch_cos", "angle", "pitch"),
    ("speed", "cont", "speed"), ("health", "cont", "health"), ("armor", "cont", "armor"),
    *((f, "flag", f) for f in _PLAYER_FLAGS),
    *((f, "cont", f) for f in _NADES),
    *((f, "flag", f) for f in _PLAYER_STATE),
    ("flash_duration_remaining", "cont", "flash_duration_remaining"),
    ("in_smoke", "flag", "in_smoke"), ("on_fire", "flag", "on_fire"),
    ("money", "cont", "money"), ("equipment_value", "cont", "equipment_value"),
    ("seconds_since_last_seen_by_enemy", "cont", "seconds_since_last_seen_by_enemy"),
    ("seconds_since_last_shot", "cont", "seconds_since_last_shot"),
    ("spotted_by_count", "cont", "spotted_by_count"), ("heard_enemy", "flag", "heard_enemy"),
    ("kills_round", "cont", "kills_round"), ("damage_round", "cont", "damage_round"),
))

UTIL_LAYOUT = BlockLayout(UtilityToken, (
    ("x", "cont", "x"), ("y", "cont", "y"), ("z", "cont", "z"),
    ("time_left_s", "cont", "time_left_s"), ("radius_units", "cont", "radius_units"),
))

_CTX_FLAGS = ("bomb_planted", "defuse_in_progress", "defuse_with_kit", "bomb_carrier_alive",
              "is_overtime")
CTX_LAYOUT = BlockLayout(ContextToken, tuple(
    (f, "flag" if f in _CTX_FLAGS else "cont", f)
    for f in CTX_FIELDS if f not in ("map_version", "tier", "bomb_site")))

# categorical columns: (column name, source field, vocab name)
PLAYER_CAT = (("side", "side", "side"), ("weapon_class", "weapon_class", "weapon"),
              ("callout", "callout", "callout"), ("grid_cell", "grid_cell", "grid"),
              ("player_id", "player_id", "player"), ("tier", "tier", "tier"))
UTIL_CAT   = (("kind", "kind", "util_kind"), ("owner_side", "owner_side", "side"),
              ("owner_id", "owner_id", "player"))
CTX_CAT    = (("map_version", "map_version", "map_version"), ("tier", "tier", "tier"),
              ("bomb_site", "bomb_site", "bomb_site"))
PER_MAP_VOCABS = ("callout", "grid")

PAIR_KINDS = {"los": "flag", "dist_units": "cont", "nav_dist_units": "cont",
              "smoke_between": "flag", "same_team": "flag"}

FEATURE_LAYOUT: dict[str, tuple[str, ...]] = {
    "player_num": PLAYER_LAYOUT.names,
    "player_cat": tuple(c for c, _, _ in PLAYER_CAT),
    "util_num":   UTIL_LAYOUT.names,
    "util_cat":   tuple(c for c, _, _ in UTIL_CAT),
    "ctx_num":    CTX_LAYOUT.names,
    "ctx_cat":    tuple(c for c, _, _ in CTX_CAT),
    "pair":       tuple(PAIR_CHANNELS),
}
NUM_LAYOUTS = {"player_num": PLAYER_LAYOUT, "util_num": UTIL_LAYOUT, "ctx_num": CTX_LAYOUT}
CAT_VOCABS  = {"player_cat": tuple(v for _, _, v in PLAYER_CAT),
               "util_cat":   tuple(v for _, _, v in UTIL_CAT),
               "ctx_cat":    tuple(v for _, _, v in CTX_CAT)}
# features not fed to the model, with the reason (reported, never silently dropped)
EXCLUDED = {"player.nav_area": "redundant with grid_cell/callout/x,y,z; vocab too large"}


def layout_hash() -> str:
    """Changes whenever a column is added, removed or reordered; stored in every shard
    index and checkpoint so a model refuses shards it was not trained on."""
    doc = json.dumps({"layout": FEATURE_LAYOUT, "kinds": {k: v.kinds for k, v in NUM_LAYOUTS.items()},
                      "cat_vocabs": CAT_VOCABS, "reserved": RESERVED}, sort_keys=True)
    return hashlib.sha256(doc.encode()).hexdigest()[:16]


def round_key(match_key: str, round_num: int) -> str:
    return f"{match_key}#{round_num}"


# ── vocabularies ──────────────────────────────────────────────────────────────

def _cat_key(vocab_name: str, value, map_version: str | None) -> str | None:
    if value is None:
        return None
    if vocab_name in PER_MAP_VOCABS:
        return f"{map_version}|{value}"
    return str(value)


class Vocab:
    """Categorical vocabularies, UNK=0, PAD=1, NULL=2, then sorted train categories."""

    def __init__(self, tables: dict[str, dict[str, int]] | None = None):
        tables = tables or {}
        self.tables = {n: dict(tables.get(n, {})) for n in VOCAB_NAMES}

    @staticmethod
    def _keys(s: Snapshot) -> Iterator[tuple[str, str | None]]:
        mv = s.ctx.map_version
        for p in s.players:
            if p.player_id == PAD_PLAYER_ID:
                continue
            for _, src, voc in PLAYER_CAT:
                yield voc, _cat_key(voc, getattr(p, src), mv)
        for u in s.utility:
            for _, src, voc in UTIL_CAT:
                yield voc, _cat_key(voc, getattr(u, src), mv)
        for _, src, voc in CTX_CAT:
            yield voc, _cat_key(voc, getattr(s.ctx, src), mv)

    def fit(self, snapshots: Iterable[Snapshot], min_count: int = 1) -> "Vocab":
        """Every category seen at least `min_count` times; indices in sorted order so a
        refit on the same data is identical."""
        counts: dict[str, dict[str, int]] = {n: {} for n in VOCAB_NAMES}
        for s in snapshots:
            for voc, key in self._keys(s):
                if key is not None:
                    counts[voc][key] = counts[voc].get(key, 0) + 1
        self.tables = {n: {k: i + len(RESERVED) for i, k in
                           enumerate(sorted(k for k, c in counts[n].items() if c >= min_count))}
                       for n in VOCAB_NAMES}
        return self

    def index(self, name: str, key: str | None) -> int:
        if key is None:
            return NULL
        return self.tables[name].get(key, UNK)

    def size(self, name: str) -> int:
        return len(RESERVED) + len(self.tables[name])

    def sizes(self) -> dict[str, int]:
        return {n: self.size(n) for n in VOCAB_NAMES}

    def inverse(self, name: str) -> list[str]:
        out = list(RESERVED) + [""] * len(self.tables[name])
        for k, i in self.tables[name].items():
            out[i] = k
        return out

    def to_dict(self) -> dict:
        return {"reserved": list(RESERVED), "tables": self.tables, "layout_hash": layout_hash()}

    @classmethod
    def from_dict(cls, doc: dict) -> "Vocab":
        if tuple(doc.get("reserved", ())) != RESERVED:
            raise ValueError(f"vocab reserved ids {doc.get('reserved')} != {RESERVED}")
        return cls(doc["tables"])

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ── tensorize ─────────────────────────────────────────────────────────────────

def _num(v) -> float | None:
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) else f


def block_matrix(layout: BlockLayout, objs: list) -> np.ndarray:
    """Tokens -> [len(objs), len(layout)] float32: values (None/NaN -> 0) then presence
    bits; angles as sin/cos. One list comprehension over the raw fields, then numpy."""
    n = len(objs)
    out = np.zeros((n, len(layout)), np.float32)
    if n == 0:
        return out
    raw = np.array([[np.nan if v is None else v for v in layout.getter(o)] for o in objs],
                   dtype=np.float64).reshape(n, len(layout.sources))
    present = ~np.isnan(raw)
    for c, (name, kind, src) in enumerate(layout.values):
        k = layout.src_index[src]
        v = raw[:, k]
        if kind == "angle":
            v = np.sin(np.radians(v)) if name.endswith("_sin") else np.cos(np.radians(v))
        out[:, c] = np.where(present[:, k], v, 0.0)
        if layout.presence_of[c] >= 0:
            out[:, layout.presence_of[c]] = present[:, k]
    return out


def _cat_matrix(cols, objs: list, map_versions: list, vocab: "Vocab") -> np.ndarray:
    out = np.empty((len(objs), len(cols)), np.int64)
    for c, (_, src, voc) in enumerate(cols):
        get = attrgetter(src)
        out[:, c] = [vocab.index(voc, _cat_key(voc, get(o), mv)) for o, mv in zip(objs, map_versions)]
    return out


def _label(labels: dict | None, key: str):
    return None if labels is None else labels.get(SOURCE_LABELS[key])


def _kind_bits(kinds) -> int:
    bits = 0
    for k in kinds or ():
        bits |= KIND_BITS.get(k, OTHER_KIND_BIT)
    return bits


def empty_arrays(n: int) -> dict[str, np.ndarray]:
    return {
        "player_num":  np.zeros((n, N_PLAYERS, len(PLAYER_LAYOUT)), np.float32),
        "player_cat":  np.full((n, N_PLAYERS, len(PLAYER_CAT)), PAD, np.int64),
        "player_mask": np.zeros((n, N_PLAYERS), bool),
        "util_num":    np.zeros((n, MAX_UTILITY, len(UTIL_LAYOUT)), np.float32),
        "util_cat":    np.full((n, MAX_UTILITY, len(UTIL_CAT)), PAD, np.int64),
        "util_mask":   np.zeros((n, MAX_UTILITY), bool),
        "ctx_num":     np.zeros((n, len(CTX_LAYOUT)), np.float32),
        "ctx_cat":     np.full((n, len(CTX_CAT)), PAD, np.int64),
        "pair":        np.full((n, N_PLAYERS, N_PLAYERS, len(PAIR_CHANNELS)), np.nan, np.float32),
        "y_win":       np.full(n, np.nan, np.float32),
        "y_hazard":    np.full((n, N_PLAYERS), np.nan, np.float32),
        "y_plant15":   np.full(n, np.nan, np.float32),
        "y_defuse15":  np.full(n, np.nan, np.float32),
        "y_tte":       np.full(n, np.nan, np.float32),
        "is_terminal": np.zeros(n, np.int8),
        "round_key":   np.array([""] * n, dtype=str),
        "t_s":         np.full(n, np.nan, np.float64),
        "tier":        np.array([""] * n, dtype=str),
        "tick_kinds":  np.zeros(n, np.int16),
    }


def tensorize(snapshots: list[Snapshot], vocab: Vocab, norm: "NormStats | None" = None) -> dict[str, np.ndarray]:
    """Snapshots -> dict with exactly BATCH_KEYS. World units unless `norm` is given."""
    n   = len(snapshots)
    out = empty_arrays(n)
    if n == 0:
        return out
    for i, s in enumerate(snapshots):
        if len(s.players) != N_PLAYERS:
            raise ValueError(f"snapshot {i}: {len(s.players)} players, expected {N_PLAYERS}")
    mvs = [s.ctx.map_version for s in snapshots]

    players = [p for s in snapshots for p in s.players]
    p_mv    = [mv for mv in mvs for _ in range(N_PLAYERS)]
    out["player_num"]  = block_matrix(PLAYER_LAYOUT, players).reshape(n, N_PLAYERS, -1)
    out["player_cat"]  = _cat_matrix(PLAYER_CAT, players, p_mv, vocab).reshape(n, N_PLAYERS, -1)
    out["player_mask"] = np.array([p.player_id != PAD_PLAYER_ID for p in players]).reshape(n, N_PLAYERS)

    slots = [(i, k) for i, s in enumerate(snapshots) for k in range(min(len(s.utility), MAX_UTILITY))]
    if slots:
        utils = [snapshots[i].utility[k] for i, k in slots]
        ii, kk = np.array(slots).T
        out["util_num"][ii, kk]  = block_matrix(UTIL_LAYOUT, utils)
        out["util_cat"][ii, kk]  = _cat_matrix(UTIL_CAT, utils, [mvs[i] for i, _ in slots], vocab)
        out["util_mask"][ii, kk] = True

    ctxs = [s.ctx for s in snapshots]
    out["ctx_num"] = block_matrix(CTX_LAYOUT, ctxs)
    out["ctx_cat"] = _cat_matrix(CTX_CAT, ctxs, mvs, vocab)
    out["pair"]    = np.stack([np.asarray(s.pair, np.float32) for s in snapshots])

    rks, tiers = [], []
    for i, s in enumerate(snapshots):
        lab = s.labels
        for key in ("y_win", "y_plant15", "y_defuse15", "y_tte"):
            v = _num(_label(lab, key))
            if v is not None:
                out[key][i] = v
        haz = _label(lab, "y_hazard")
        if haz is not None:
            out["y_hazard"][i] = [np.nan if _num(h) is None else float(h) for h in haz]
        meta  = s.meta
        kinds = meta.kinds if meta is not None else ()
        term  = _label(lab, "is_terminal")
        out["is_terminal"][i] = int(term) if term is not None else int("terminal" in kinds)
        rks.append(round_key(meta.match_key, meta.round_num) if meta is not None
                   else round_key("", s.ctx.round_num))
        out["t_s"][i] = meta.t_s if meta is not None and meta.t_s is not None else np.nan
        tiers.append(s.ctx.tier)
        out["tick_kinds"][i] = _kind_bits(kinds)
    out["round_key"] = np.array(rks, dtype=str)
    out["tier"]      = np.array(tiers, dtype=str)
    return norm.normalize(out) if norm is not None else out


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        return empty_arrays(0)
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def take(arrays: dict[str, np.ndarray], idx) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in arrays.items()}


# ── normalization stats ───────────────────────────────────────────────────────

class NormStats:
    """Mean/std per continuous feature (train only). Flags, angles and presence bits are
    identity (mean 0, std 1). Values whose presence bit is 0, and absent tokens, are
    excluded from the fit."""

    MIN_STD = 1e-6

    def __init__(self, stats: dict[str, dict[str, list[float]]] | None = None, n: int = 0):
        self.stats = stats or {}
        self.n     = n

    @classmethod
    def fit(cls, parts: Iterable[dict[str, np.ndarray]]) -> "NormStats":
        acc: dict[str, dict[str, list[float]]] = {}       # block -> name -> [count, sum, sumsq]
        n = 0

        def add(block: str, name: str, vals: np.ndarray) -> None:
            a = acc.setdefault(block, {}).setdefault(name, [0.0, 0.0, 0.0])
            v = vals.astype(np.float64)
            a[0] += v.size
            a[1] += float(v.sum())
            a[2] += float((v * v).sum())

        for arr in parts:
            n += len(arr["ctx_num"])
            for block, lay in NUM_LAYOUTS.items():
                x = arr[block]
                if block == "player_num":
                    tok = arr["player_mask"]
                elif block == "util_num":
                    tok = arr["util_mask"]
                else:
                    tok = np.ones(x.shape[0], bool)
                for c, (name, kind, _) in enumerate(lay.values):
                    if kind != "cont":
                        continue
                    ok = tok.copy()
                    if lay.presence_of[c] >= 0:
                        ok &= x[..., lay.presence_of[c]] > 0.5
                    add(block, name, x[..., c][ok])
            for k, ch in enumerate(PAIR_CHANNELS):
                if PAIR_KINDS[ch] == "cont":
                    v = arr["pair"][..., k]
                    add("pair", ch, v[~np.isnan(v)])

        stats: dict[str, dict[str, list[float]]] = {}
        for block, cols in acc.items():
            for name, (cnt, s, ss) in cols.items():
                if cnt == 0:
                    mean, std = 0.0, 1.0
                else:
                    mean = s / cnt
                    var  = max(0.0, ss / cnt - mean * mean)
                    std  = math.sqrt(var)
                    std  = std if std >= cls.MIN_STD else 1.0
                stats.setdefault(block, {})[name] = [mean, std]
        return cls(stats, n)

    def arrays(self, block: str) -> tuple[np.ndarray, np.ndarray]:
        """(mean, std) over every column of a numeric block (or the 5 pair channels)."""
        names = PAIR_CHANNELS if block == "pair" else NUM_LAYOUTS[block].names
        mean = np.zeros(len(names), np.float32)
        std  = np.ones(len(names), np.float32)
        for i, name in enumerate(names):
            ms = self.stats.get(block, {}).get(name)
            if ms is not None:
                mean[i], std[i] = ms
        return mean, std

    def normalize(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Copy with continuous columns z-scored; missing values stay 0."""
        out = dict(arrays)
        for block, lay in NUM_LAYOUTS.items():
            mean, std = self.arrays(block)
            x = (arrays[block] - mean) / std
            for c, p in enumerate(lay.presence_of):
                if p >= 0:
                    x[..., c] *= arrays[block][..., p]
            out[block] = x.astype(np.float32)
        mean, std = self.arrays("pair")
        out["pair"] = ((arrays["pair"] - mean) / std).astype(np.float32)
        return out

    def denormalize(self, block: str, name: str, values: np.ndarray) -> np.ndarray:
        mean, std = self.stats.get(block, {}).get(name, (0.0, 1.0))
        return values * std + mean

    def to_dict(self) -> dict:
        return {"stats": self.stats, "n": self.n, "layout_hash": layout_hash()}

    @classmethod
    def from_dict(cls, doc: dict) -> "NormStats":
        return cls(doc["stats"], doc.get("n", 0))

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ── shards ────────────────────────────────────────────────────────────────────

SHARD_INDEX = "index.json"


def split_dir(split_name: str, base: Path | None = None) -> Path:
    return Path(base if base is not None else paths.DATASETS_DIR) / split_name


def write_shards(split_name: str, snapshots_iter: Iterable[Snapshot], shard_size: int, *,
                 vocab: Vocab, base: Path | None = None, extra: dict | None = None) -> list[Path]:
    """Tensorize and write `<base>/<split>/shard_00000.npz ...` of `shard_size`
    snapshots each, plus `index.json`. Refuses a directory that already has shards
    (never overwrites: a new dataset gets a new name)."""
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    d = split_dir(split_name, base)
    if d.is_dir() and any(d.glob("shard_*.npz")):
        raise FileExistsError(f"{d} already has shards; write to a new split name")
    d.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    counts: list[int] = []
    buf: list[Snapshot] = []

    def flush() -> None:
        arr  = tensorize(buf, vocab)
        path = d / f"shard_{len(out):05d}.npz"
        np.savez(path, **arr)
        out.append(path)
        counts.append(len(buf))
        buf.clear()

    for s in snapshots_iter:
        buf.append(s)
        if len(buf) >= shard_size:
            flush()
    if buf:
        flush()
    index = {"split": split_name, "n": int(sum(counts)), "shards": [p.name for p in out],
             "counts": counts, "layout_hash": layout_hash(), "keys": list(BATCH_KEYS),
             "extra": extra or {}}
    (d / SHARD_INDEX).write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return out


def _resolve(split: str | Path, base: Path | None) -> Path:
    p = Path(split)
    return p if p.is_dir() else split_dir(str(split), base)


def iter_shards(split: str | Path, base: Path | None = None) -> Iterator[dict[str, np.ndarray]]:
    d = _resolve(split, base)
    idx = d / SHARD_INDEX
    if idx.is_file():
        doc = json.loads(idx.read_text(encoding="utf-8"))
        if doc.get("layout_hash") != layout_hash():
            raise ValueError(f"{d}: shards have layout {doc.get('layout_hash')}, this code "
                             f"expects {layout_hash()}; rebuild the dataset")
        names = doc["shards"]
    else:
        names = sorted(p.name for p in d.glob("shard_*.npz"))
    for name in names:
        with np.load(d / name, allow_pickle=False) as z:
            yield {k: z[k] for k in BATCH_KEYS}


def read_shards(split: str | Path, base: Path | None = None) -> dict[str, np.ndarray]:
    """Every shard of a split, concatenated (a split name under `base`, or a directory)."""
    return concat(list(iter_shards(split, base)))


# ── build (P5.3.1) ────────────────────────────────────────────────────────────

def keep_for_rate(s: Snapshot, sample_hz: float | None, tol_s: float = 1.0 / 256) -> bool:
    """Keep every non-grid snapshot; keep a grid one only if its t_s lies on the
    1/sample_hz grid (lets a 4 Hz states build serve the 2 Hz dataset and vice versa)."""
    if sample_hz is None or s.meta is None:
        return True
    kinds = set(s.meta.kinds)
    if kinds - {"grid"}:
        return True
    k = s.meta.t_s * sample_hz
    return abs(k - round(k)) * (1.0 / sample_hz) <= tol_s


def next_version(directory: Path, stem: str) -> int:
    d = Path(directory)
    have = [int(p.stem[len(stem):]) for p in d.glob(f"{stem}*.json") if p.stem[len(stem):].isdigit()]
    return max(have, default=0) + 1


def _map_ok(map_version: str, maps) -> bool:
    return not maps or map_version in maps or map_version.split("@")[0] in maps


def select_matches(catalog_rows: list[dict], split_doc: dict, split: str, tiers, maps) -> list[dict]:
    keys = set(split_doc["splits"][split])
    return [r for r in catalog_rows if r["match_key"] in keys
            and (not tiers or r.get("tier") in tiers) and _map_ok(r["map_version"], maps)]


def _iter_states(rows: list[dict], root: Path, sample_hz) -> Iterator[Snapshot]:
    import polars as pl
    from engine.state.schema import frame_to_snapshots
    for r in rows:
        df = pl.read_parquet(root / r["states_path"])
        for s in frame_to_snapshots(df):
            if keep_for_rate(s, sample_hz):
                yield s


def build(catalog, split_file, tiers=None, maps=None, sample_hz: float | None = None, *,
          base: Path | None = None, engine_dir: Path | None = None, shard_size: int = 50_000,
          suffix: str = "") -> dict:
    """Catalog + frozen split file -> shards per split, vocab and norm stats fitted on
    train only. `catalog` is a path to catalog.parquet or its rows; `suffix` names a
    variant (e.g. "_hz4" for the P5.3.3 rate ablation: datasets/train_hz4/...)."""
    from engine.data import layout, splits        # owned by P1/P5.1, imported lazily
    root     = Path(engine_dir) if engine_dir is not None else paths.ENGINE_DIR
    out_base = Path(base) if base is not None else paths.DATASETS_DIR
    if isinstance(catalog, (str, Path)):
        import pyarrow.parquet as pq
        rows = pq.read_table(catalog).to_pylist()
    elif catalog is None:
        rows = layout.read_catalog(root)
    else:
        rows = list(catalog)
    split_doc = splits.load(split_file)
    chosen = {name: select_matches(rows, split_doc, name, tiers, maps) for name in ("train", "val", "test")}

    vocab = Vocab().fit(_iter_states(chosen["train"], root, sample_hz))
    version = next_version(out_base, "vocab_v")
    vocab_path = vocab.save(out_base / f"vocab_v{version}.json")
    extra = {"split_file": str(split_file), "split_hash": splits.file_hash(split_file),
             "tiers": list(tiers) if tiers else None, "maps": list(maps) if maps else None,
             "sample_hz": sample_hz, "vocab": vocab_path.name}
    written = {}
    for name, sel in chosen.items():
        written[name] = write_shards(name + suffix, _iter_states(sel, root, sample_hz), shard_size,
                                     vocab=vocab, base=out_base,
                                     extra={**extra, "matches": [r["match_key"] for r in sel]})
    norm = NormStats.fit(iter_shards("train" + suffix, out_base))
    norm_path = norm.save(out_base / f"norm_v{version}.json")
    return {"vocab": vocab_path, "norm": norm_path, "shards": written,
            "n_matches": {k: len(v) for k, v in chosen.items()}}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build tensor shards per split (P5.3).")
    ap.add_argument("--catalog", default=None, help="catalog.parquet (default: data/engine/catalog.parquet)")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--tiers", nargs="*", default=None)
    ap.add_argument("--maps", nargs="*", default=None)
    ap.add_argument("--sample-hz", type=float, default=None)
    ap.add_argument("--shard-size", type=int, default=50_000)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--assert-qa", action="store_true", help="exit 1 when QA thresholds are violated")
    a = ap.parse_args(argv)
    res = build(a.catalog, a.split_file, a.tiers, a.maps, a.sample_hz,
                shard_size=a.shard_size, suffix=a.suffix)
    print(f"vocab {res['vocab']}  norm {res['norm']}  matches {res['n_matches']}")
    if a.assert_qa:
        from engine import config
        from engine.data import qa
        cfg = config.load()
        bad = []
        for name in ("train", "val", "test"):
            r = qa.report(name + a.suffix, vocab=Vocab.load(res["vocab"]), cfg=cfg)
            bad += qa.violations(r, cfg)
        for line in bad:
            print(f"QA violation: {line}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
