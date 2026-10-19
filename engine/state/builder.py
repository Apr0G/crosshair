"""
Orchestrates one match -> `states.parquet` (P3.8.1): slices raw tables causally,
builds tokens per sampled tick, attaches labels in a separate step (P3.7).

    from engine.state import builder
    from engine.state.geometry import FlatGeometry
    snaps = builder.build(raw, FlatGeometry(), cfg, match_key="m1", tier="pro")
    builder.write(snaps, path)

Causality (P3.6.3). Feature code only ever sees a `Slice`: the rows of the current round
with event tick <= t, end-of-life values in the future nulled, exactly the semantics of
`RawTables.slice_until(t)`. Two ways to get one, provably equal (test_builder.py
compares them snapshot by snapshot on the `base` story):
- slow: `RoundView.build(raw.slice_until(t), r).until(t)` (the contract, re-slices the
  whole match per snapshot);
- fast (default): `RoundView.build(raw, r)` once per round, then `.until(t)`, which
  bisects the tick-sorted per-round arrays. Every derived column in a RoundView
  (last-alive index, last-spotted tick) is a forward fill, so it is prefix-consistent:
  computing it on the full round then cutting at t equals computing it on the cut.

Round scope: tables with `round_num` are filtered by it; the others (weapon_fire,
footsteps, flashes, smokes, infernos, grenades) keep rows from the round's start_tick
on. The roster (who is in the round, on which side) is read from the round's tick rows
at ticks <= freeze_end (or the first tick of the round if none), so it is known by the
time the first snapshot is taken. Slots: CT first then T, each sorted by SteamID,
padded to 5 per side with empty slots (`players.pad_token`, player_id -1).

Boundary samples. Without `action_ticks` the builder takes the instantaneous events P4
will catalogue from the raw tables: kills, every bomb event, flash and HE detonations;
and as durative windows the smoke and fire lifetimes (start, end). P4 may add more
(movement, holds, rotations, buys) by passing `action_ticks` / `durative_windows`.
"""
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl

from engine.parse.schema import RawTables
from engine.state import context, pairwise, players, sampler, utility
from engine.state.schema import N_PER_SIDE, Snapshot, SnapshotMeta, rows_to_frame

log = logging.getLogger("engine.state.builder")

TIERS = ("pro", "faceit_10", "faceit_8_9", "other")     # P3.2.7

# ── causal view ───────────────────────────────────────────────────────────────


def _to_np(s: pl.Series) -> np.ndarray:
    """Column -> numpy. Floats: float64 with NaN for null. Ints / bools without nulls:
    native dtype. Everything else (strings, lists, nullable ints/bools): object array of
    Python values with None for null."""
    dt = s.dtype
    if dt.is_float():
        return s.cast(pl.Float64).to_numpy()
    if (dt.is_integer() or dt == pl.Boolean) and s.null_count() == 0:
        return s.to_numpy()
    out = np.empty(s.len(), dtype=object)
    out[:] = s.to_list()
    return out


class Table:
    """Columns of one table, sorted by `tick_col`; rows [:n] are visible."""
    __slots__ = ("cols", "tick_col", "n")

    def __init__(self, cols: dict[str, np.ndarray], tick_col: str, n: int | None = None):
        self.cols, self.tick_col = cols, tick_col
        self.n = len(cols[tick_col]) if n is None else n

    @classmethod
    def from_frame(cls, df: pl.DataFrame, tick_col: str) -> "Table":
        df = df.sort(tick_col, maintain_order=True)
        return cls({c: _to_np(df[c]) for c in df.columns}, tick_col)

    def col(self, name: str) -> np.ndarray:
        return self.cols[name][: self.n]

    def until(self, tick: int) -> "Table":
        k = int(np.searchsorted(self.cols[self.tick_col], tick, side="right"))
        return Table(self.cols, self.tick_col, min(k, self.n))


class Slice:
    """What feature code sees at one tick (see module doc)."""
    __slots__ = ("tick", "tick_rate", "round", "past_rounds", "roster", "sides", "players",
                 "fire", "steps", "kills", "damages", "bomb", "flashes", "smokes", "infernos",
                 "decoys")


_ROUND_END_COLS = ("end_tick", "official_end_tick", "winner_side", "end_reason", "bomb_planted")


def _null_after(rows: list[dict], tick: int, cols: tuple[str, ...], when: str) -> list[dict]:
    out = []
    for r in rows:
        w = r.get(when)
        if w is not None and w > tick:
            r = {**r, **{c: None for c in cols}}
        out.append(r)
    return out


class RoundView:
    """One round of a RawTables, pre-split and tick-sorted; `.until(t)` -> Slice."""

    def __init__(self):
        self.round_num = None

    @classmethod
    def build(cls, raw: RawTables, round_num: int) -> "RoundView":
        v = cls()
        v.round_num = int(round_num)
        v.tick_rate = int(raw.meta["tick_rate"])
        v.rounds = sorted(raw.rounds.to_dicts(), key=lambda r: r["round_num"])
        rr = next((r for r in v.rounds if r["round_num"] == v.round_num), None)
        if rr is None:
            raise ValueError(f"round {round_num} not in rounds table")
        start, fe = int(rr["start_tick"]), int(rr["freeze_end_tick"])
        v.start, v.freeze_end = start, fe

        ticks = raw.ticks.filter(pl.col("round_num") == v.round_num)
        v.roster, v.sides = _roster(ticks, fe)
        enemy_ids = {side: np.array([s for s, sd in v.sides.items() if sd != side], dtype=np.uint64)
                     for side in ("ct", "t")}
        v.players = {}
        for (sid,), df in ticks.partition_by("steamid", as_dict=True, maintain_order=True).items():
            sid = int(sid)
            if sid not in v.sides:
                continue
            tb = Table.from_frame(df, "tick")
            _derive_player(tb, enemy_ids[v.sides[sid]])
            v.players[sid] = tb

        in_round = pl.col("tick") >= start
        v.fire  = _split(raw.weapon_fire.filter(in_round), "tick")
        v.steps = _split(raw.footsteps.filter(in_round), "tick")
        v.kills   = Table.from_frame(raw.kills.filter(pl.col("round_num") == v.round_num), "tick")
        v.damages = Table.from_frame(raw.damages.filter(pl.col("round_num") == v.round_num), "tick")
        v.bomb    = Table.from_frame(raw.bomb.filter(pl.col("round_num") == v.round_num), "tick")
        v.flashes = Table.from_frame(raw.flashes.filter(pl.col("detonate_tick") >= start), "detonate_tick")
        v.smokes   = raw.smokes.filter(pl.col("start_tick") >= start).sort("start_tick", maintain_order=True).to_dicts()
        v.infernos = raw.infernos.filter(pl.col("start_tick") >= start).sort("start_tick", maintain_order=True).to_dicts()
        dec = raw.grenades.filter((pl.col("type") == "decoy") & (pl.col("throw_tick") >= start))
        v.decoys = [{"entity_id": g["entity_id"], "start_tick": g["detonate_tick"],
                     "end_tick": g["expire_tick"], "throw_tick": g["throw_tick"],
                     "x": g["x"], "y": g["y"], "z": g["z"], "thrower_steamid": g["thrower_steamid"]}
                    for g in dec.sort("throw_tick", maintain_order=True).to_dicts()]
        return v

    def until(self, tick: int) -> Slice:
        t = int(tick)
        sl = Slice()
        sl.tick, sl.tick_rate = t, self.tick_rate
        rounds = [r for r in self.rounds if r["start_tick"] <= t]
        rounds = [{**r, **{c: None for c in _ROUND_END_COLS}}
                  if r["end_tick"] is None or r["end_tick"] > t else r for r in rounds]
        rounds = [{**r, "official_end_tick": None}
                  if r["official_end_tick"] is not None and r["official_end_tick"] > t else r
                  for r in rounds]
        sl.round = next(r for r in rounds if r["round_num"] == self.round_num)
        sl.past_rounds = [r for r in rounds if r["round_num"] < self.round_num]
        sl.roster, sl.sides = self.roster, self.sides
        sl.players = {s: tb.until(t) for s, tb in self.players.items()}
        sl.fire    = {s: tb.until(t) for s, tb in self.fire.items()}
        sl.steps   = {s: tb.until(t) for s, tb in self.steps.items()}
        sl.kills, sl.damages = self.kills.until(t), self.damages.until(t)
        sl.bomb, sl.flashes  = self.bomb.until(t), self.flashes.until(t)
        sl.smokes   = _null_after([r for r in self.smokes if r["start_tick"] <= t], t, ("end_tick",), "end_tick")
        sl.infernos = _null_after([r for r in self.infernos if r["start_tick"] <= t], t, ("end_tick",), "end_tick")
        dec = [r for r in self.decoys if r["throw_tick"] <= t]
        dec = _null_after(dec, t, ("start_tick", "x", "y", "z"), "start_tick")
        sl.decoys = _null_after(dec, t, ("end_tick",), "end_tick")
        return sl


def _roster(ticks: pl.DataFrame, freeze_end: int) -> tuple[list[tuple[int, str]], dict[int, str]]:
    """[(steamid, side)] CT then T, sorted by steamid, at most 5 per side."""
    if ticks.height == 0:
        return [], {}
    early = ticks.filter(pl.col("tick") <= freeze_end)
    if early.height == 0:
        early = ticks.filter(pl.col("tick") == ticks["tick"].min())
    last = early.sort("tick", maintain_order=True).group_by("steamid", maintain_order=True).last()
    sides: dict[int, str] = {}
    for sid, side in zip(last["steamid"].to_list(), last["side"].to_list()):
        if side in ("ct", "t"):
            sides[int(sid)] = side
    roster = []
    for side in ("ct", "t"):
        ids = sorted(s for s, sd in sides.items() if sd == side)
        if len(ids) > N_PER_SIDE:
            log.warning(f"{len(ids)} players on {side}; keeping the first {N_PER_SIDE} by steamid")
            for s in ids[N_PER_SIDE:]:
                del sides[s]
            ids = ids[:N_PER_SIDE]
        roster += [(s, side) for s in ids]
    return roster, sides


def _derive_player(tb: Table, enemies: np.ndarray) -> None:
    """Forward-fill columns (prefix-consistent): `_last_alive` row index (-1 none),
    `_spot_n` enemies spotting now (NaN if spotted_by is null), `_last_spot` tick of
    the last row with _spot_n > 0 (-1 none)."""
    n = tb.n
    idx = np.arange(n, dtype=np.int64)
    alive = np.array([bool(a) if a is not None else False for a in tb.cols["is_alive"]], dtype=bool)
    tb.cols["_last_alive"] = np.maximum.accumulate(np.where(alive, idx, -1)) if n else idx
    spot = np.full(n, np.nan)
    if "spotted_by" in tb.cols:
        en = set(int(e) for e in enemies)
        for k, lst in enumerate(tb.cols["spotted_by"]):
            if lst is not None:
                spot[k] = sum(1 for s in lst if s is not None and int(s) in en)
    tb.cols["_spot_n"] = spot
    ticks = tb.cols["tick"].astype(np.int64)
    tb.cols["_last_spot"] = np.maximum.accumulate(np.where(spot > 0, ticks, -1)) if n else ticks


def _split(df: pl.DataFrame, tick_col: str) -> dict[int, Table]:
    out = {}
    if df.height == 0:
        return out
    for (sid,), part in df.partition_by("steamid", as_dict=True, maintain_order=True).items():
        if sid is not None:
            out[int(sid)] = Table.from_frame(part, tick_col)
    return out


def slow_slice(raw: RawTables, round_num: int, tick: int) -> Slice:
    """The contract path: slice the whole match, then view the round."""
    return RoundView.build(raw.slice_until(tick), round_num).until(tick)


# ── sampling ──────────────────────────────────────────────────────────────────

def raw_event_ticks(raw: RawTables, round_row: dict, terminal: int) -> tuple[set[int], list[tuple]]:
    """(instantaneous event ticks, durative windows) of one round from the raw tables."""
    r, start = int(round_row["round_num"]), int(round_row["start_tick"])
    win = (pl.col("tick") >= start) & (pl.col("tick") <= terminal)
    ticks: set[int] = set()
    ticks.update(raw.kills.filter(pl.col("round_num") == r)["tick"].to_list())
    ticks.update(raw.bomb.filter(pl.col("round_num") == r)["tick"].to_list())
    g = raw.grenades.filter(pl.col("type").is_in(["flash", "he"])).rename({"detonate_tick": "tick"})
    ticks.update(t for t in g.filter(win)["tick"].to_list() if t is not None)
    ticks.update(raw.flashes.rename({"detonate_tick": "tick"}).filter(win)["tick"].to_list())
    windows = []
    for df in (raw.smokes, raw.infernos):
        d = df.filter((pl.col("start_tick") >= start) & (pl.col("start_tick") <= terminal))
        windows += list(zip(d["start_tick"].to_list(), d["end_tick"].to_list()))
    return {int(t) for t in ticks if t is not None}, windows


# ── snapshot assembly ─────────────────────────────────────────────────────────

@contextmanager
def _timed(timings: dict | None, stage: str):
    if timings is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = timings.get(stage, 0.0) + time.perf_counter() - t0


def slots(sl: Slice) -> list[tuple[int | None, str]]:
    """10 (steamid | None, side) slots: CT then T, sorted by steamid, padded."""
    out = []
    for side in ("ct", "t"):
        ids = [s for s, sd in sl.roster if sd == side]
        out += [(s, side) for s in ids] + [(None, side)] * (N_PER_SIDE - len(ids))
    return out


def snapshot_at(sl: Slice, geometry, cfg, *, match_key: str, tier: str, kinds=(),
                los_cache: dict | None = None, heard_window_s: float | None = None,
                timings: dict | None = None) -> Snapshot:
    hw = heard_window_s if heard_window_s is not None else _heard_window(cfg)
    with _timed(timings, "utility"):
        util, n_smokes, n_fires = utility.tokens(sl, cfg.sampling.max_utility_tokens)
    with _timed(timings, "players"):
        toks = [players.pad_token(side, tier) if sid is None else
                players.token(sl, sid, side, geometry, tier, util, hw)
                for sid, side in slots(sl)]
    with _timed(timings, "context"):
        ctx = context.token(sl, toks, n_smokes, n_fires, clock=cfg.clock,
                            map_version=geometry.map_version, tier=tier)
    with _timed(timings, "pairwise"):
        pair = pairwise.table(toks, geometry, util, los_cache)
    fe = int(sl.round["freeze_end_tick"])
    meta = SnapshotMeta(match_key=match_key, round_num=int(sl.round["round_num"]), tick=sl.tick,
                        t_s=(sl.tick - fe) / sl.tick_rate, kinds=tuple(kinds))
    return Snapshot(players=toks, utility=util, ctx=ctx, pair=pair, meta=meta, labels=None)


def _heard_window(cfg) -> float:
    """sampling.heard_window_s if the config has it, else players.HEARD_WINDOW_S."""
    v = getattr(cfg.sampling, "heard_window_s", None)
    return float(v) if v is not None else players.HEARD_WINDOW_S


def _last_round_tick(raw: RawTables, r: int) -> int | None:
    t = raw.ticks.filter(pl.col("round_num") == r)["tick"]
    return None if t.len() == 0 else int(t.max())


def build(raw: RawTables, geometry, cfg, *, match_key: str, tier: str,
          action_ticks: dict | None = None, durative_windows: dict | None = None,
          los_cache: bool | None = None, causal: str = "fast", debug: bool = False,
          labels: bool = True, timings: dict | None = None) -> list[Snapshot]:
    """Every snapshot of the match, labelled (unless `labels=False`).
    `action_ticks` / `durative_windows`: {round_num: ticks / [(start, end)]}; None ->
    taken from the raw tables (see module doc). `los_cache`: None -> config.
    `causal`: "fast" (default) or "slow" (re-slice per snapshot, for tests).
    `debug`: assert Snapshot.validate() is empty for every snapshot."""
    if causal not in ("fast", "slow"):
        raise ValueError(f"causal must be 'fast' or 'slow', got {causal!r}")
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    tr  = int(raw.meta["tick_rate"])
    hz  = int(cfg.sampling.sample_hz)
    off = int(cfg.sampling.boundary_offset_ticks)
    use_cache = cfg.sampling.los_cache if los_cache is None else los_cache
    cache = {} if use_cache else None
    hw  = _heard_window(cfg)

    snaps: list[Snapshot] = []
    for rr in sorted(raw.rounds.to_dicts(), key=lambda r: r["round_num"]):
        r = int(rr["round_num"])
        with _timed(timings, "sampling"):
            term = sampler.terminal_tick(rr)
            if term is None:
                term = _last_round_tick(raw, r)
                log.warning(f"round {r}: no end tick; stopping at the last tick row ({term})")
            if term is None:
                continue
            ev, wins = raw_event_ticks(raw, rr, term)
            at = ev if action_ticks is None else set(action_ticks.get(r, ()))
            dw = wins if durative_windows is None else list(durative_windows.get(r, ()))
            ticks = sampler.all_ticks(rr, tr, hz, at, dw, off, terminal=term)
        with _timed(timings, "slicing"):
            view = RoundView.build(raw, r) if causal == "fast" else None
        for t, kinds in ticks:
            with _timed(timings, "slicing"):
                sl = view.until(t) if view is not None else slow_slice(raw, r, t)
            s = snapshot_at(sl, geometry, cfg, match_key=match_key, tier=tier, kinds=kinds,
                            los_cache=cache, heard_window_s=hw, timings=timings)
            snaps.append(s)
    if labels:
        with _timed(timings, "labels"):
            attach_labels(snaps, raw, cfg)
    if debug:
        for s in snaps:
            probs = s.validate()
            assert not probs, f"round {s.meta.round_num} tick {s.meta.tick}: {probs}"
    return snaps


# ── labels (P3.7): a separate pass that may read the future ───────────────────

def _horizon(cfg, name: str, default: float) -> float:
    m = getattr(cfg, "model", None)
    v = getattr(m, name, None) if m is not None else None
    return float(v) if v is not None else default


def attach_labels(snapshots: list[Snapshot], raw: RawTables, cfg=None) -> list[Snapshot]:
    """Set `labels` on every snapshot (in place; also returned). Event windows are
    `t <= e < t + H*tick_rate` (so C killed at 2000 is 1 on (2000 - 5 s, 2000]).
    A label whose event already happened before t is None (not at risk any more)."""
    tr = int(raw.meta["tick_rate"])
    h_die    = _horizon(cfg, "hazard_horizon_s", 5.0) * tr
    h_plant  = _horizon(cfg, "plant_horizon_s", 15.0) * tr
    h_defuse = _horizon(cfg, "defuse_horizon_s", 15.0) * tr
    tte_cap  = 30.0

    rounds = {int(r["round_num"]): r for r in raw.rounds.to_dicts()}
    cache: dict[int, dict] = {}

    def facts(r: int) -> dict:
        if r in cache:
            return cache[r]
        k = raw.kills.filter(pl.col("round_num") == r)
        deaths: dict[int, int] = {}
        for t, v in zip(k["tick"].to_list(), k["victim_steamid"].to_list()):
            if v is not None:
                deaths[int(v)] = min(int(t), deaths.get(int(v), int(t)))
        by_hash = {players.player_id_hash(s): s for s in
                   raw.ticks.filter(pl.col("round_num") == r)["steamid"].unique().to_list()}
        sides = dict(raw.ticks.filter(pl.col("round_num") == r)
                     .group_by("steamid").agg(pl.col("side").last()).iter_rows())
        b = raw.bomb.filter(pl.col("round_num") == r)
        plants  = sorted(b.filter(pl.col("event") == "plant_complete")["tick"].to_list())
        defuses = sorted(b.filter(pl.col("event") == "defuse_complete")["tick"].to_list())
        d = raw.damages.filter(pl.col("round_num") == r)
        eng = [int(t) for t, a, v in zip(d["tick"].to_list(), d["attacker_steamid"].to_list(),
                                         d["victim_steamid"].to_list())
               if a is not None and v is not None and a != v and sides.get(a) != sides.get(v)]
        eng = np.array(sorted(set(eng) | set(int(t) for t in k["tick"].to_list())), dtype=np.int64)
        cache[r] = dict(deaths=deaths, by_hash=by_hash, plant=plants[0] if plants else None,
                        defuse=defuses[0] if defuses else None, eng=eng)
        return cache[r]

    def window(e, t, h):
        if e is None:
            return 0
        if e < t:
            return None
        return int(e < t + h)

    for s in snapshots:
        r, t = s.meta.round_num, s.meta.tick
        rr, f = rounds[r], facts(r)
        w = rr.get("winner_side")
        dies = []
        for p in s.players:
            sid = f["by_hash"].get(p.player_id)
            if sid is None:
                dies.append(None)
                continue
            d = f["deaths"].get(int(sid))
            dies.append(0 if d is None else window(d, t, h_die))
        k = int(np.searchsorted(f["eng"], t, side="left"))
        tte = tte_cap if k >= len(f["eng"]) else min(tte_cap, (int(f["eng"][k]) - t) / tr)
        s.labels = {
            "round_won_ct":              None if w not in ("ct", "t") else int(w == "ct"),
            "round_end_reason":          rr.get("end_reason"),
            "dies_within_5s":            dies,
            "plant_within_15s":          window(f["plant"], t, h_plant),
            "defuse_within_15s":         window(f["defuse"], t, h_defuse),
            "time_to_next_engagement_s": float(tte),
            "is_terminal":               int(sampler.TERMINAL in s.meta.kinds),
        }
    return snapshots


# ── output ────────────────────────────────────────────────────────────────────

def to_frame(snapshots: list[Snapshot]) -> pl.DataFrame:
    return rows_to_frame([s.to_row() for s in snapshots])


def write(snapshots: list[Snapshot], path) -> Path:
    """states.parquet via schema.rows_to_frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_frame(snapshots).write_parquet(path)
    return path
