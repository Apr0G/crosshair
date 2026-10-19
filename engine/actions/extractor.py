"""
Orchestrates one match (P4): event actions from the raw tables, movement actions and
enablers from the state snapshots, simultaneity windows, and `actions.parquet` +
`windows.parquet` (P4.7.1).

    from engine.actions import extractor
    ev    = extractor.extract_event_actions(raw, cfg, "m1", geometry=geo)            # (1)
    snaps = builder.build(raw, geo, cfg, match_key="m1", tier="pro",                 # (2)
                          action_ticks=extractor.event_ticks(ev),
                          durative_windows=extractor.durative_windows(ev))
    actions, windows = extractor.extract(raw, snaps, cfg, match_key="m1", geometry=geo)  # (3)-(5)
    extractor.write(actions, windows, out_dir)

    snaps, actions, windows = extractor.run(raw, cfg, match_key="m1", geometry=geo)   # all of it

Flow (the builder needs action ticks for boundary samples; movement needs snapshots):
(1) `extract_event_actions` reads only the raw tables: kills, damage, flash, HE, bomb,
    smoke / molotov / decoy (+ their pop/fade pseudo-actions), buys, drops.
(2) the caller builds snapshots with `event_ticks` / `durative_windows` of (1).
(3) `extract_movement_actions` reads the GRID snapshots only (peek, rotation, hold,
    save), so its output does not change when boundary samples are added later.
(4) `attach_enablers` fills `Action.enablers` for kill / damage / plant_complete /
    defuse_complete from the snapshot at tick - 1 and the raw tables.
(5) `windows.group` makes the event windows.
`run()` does (1)-(5) and rebuilds the snapshots once when (3) added ticks (peek ticks
need their own boundary samples), so the returned snapshots cover every action.

Shared lookups live in `Context` (per-round rosters, sides, per-player tick tracks,
snapshots by round); every type module receives it as the `ctx` argument of its
registered extractor `fn(raw, states, ctx) -> list[Action]` (P4.1.2).

Determinism: actions are sorted by (round, tick or window start, kind, type, id);
enablers by (kind, action_id); duplicate ids within a type are salted (victim / target
/ receiver id) in a fixed order, so two runs write identical parquet bytes.

Filled by: P4.6.1, P4.6.2, P4.7.1.
"""
import logging
import math
from bisect import bisect_right
from pathlib import Path

import numpy as np
import polars as pl

from engine.actions import bomb, economy, enablers, kills, movement, utility
from engine.actions import windows as winmod
from engine.actions.types import (REGISTRY, TYPE_KIND, Action, assert_unique_ids,
                                  make_action_id, missing_extractors)
from engine.actions.windows import Window
from engine.state import sampler
from engine.state.players import _nan, player_id_hash

log = logging.getLogger("engine.actions.extractor")

# ── stages ────────────────────────────────────────────────────────────────────

TYPE_MODULES = (kills, utility, bomb, economy, movement)    # importing them registers every extractor
EVENT_TYPES = ("kill", "team_kill", "damage", "flash", "he", "smoke", "molotov", "decoy",
               "plant_start", "plant_complete", "defuse_start", "defuse_complete",
               "defuse_abort", "bomb_drop", "bomb_pickup", "buy", "team_buy", "drop")
MOVEMENT_TYPES = ("peek", "rotation", "hold", "save")
KIND_ORDER = {"round": 0, "instant": 1, "durative": 2}

ACTION_SCHEMA = {
    "action_id": pl.String, "match_key": pl.String, "round_num": pl.Int32,
    "actor_id_hash": pl.Int64, "actor_side": pl.String, "kind": pl.String, "type": pl.String,
    "subtype": pl.String, "tick": pl.Int64, "window_start_tick": pl.Int64,
    "window_end_tick": pl.Int64, "params": pl.String, "enablers": pl.String,
    "state_pre_tick": pl.Int64, "state_post_tick": pl.Int64,
}
WINDOW_SCHEMA = {
    "window_id": pl.String, "round_num": pl.Int32, "start_tick": pl.Int64,
    "end_tick": pl.Int64, "member_action_ids": pl.List(pl.String),
}


# ── per-player tick track ─────────────────────────────────────────────────────

class Track:
    """One player's tick rows in one round, sorted by tick; columns converted lazily."""

    def __init__(self, df: pl.DataFrame):
        self.df    = df.sort("tick", maintain_order=True)
        self.tick  = self.df["tick"].to_numpy().astype(np.int64)
        self._cols: dict[str, np.ndarray | list] = {}

    def col(self, name: str):
        """Floats -> float64 with NaN; ints/bools without nulls -> numpy; else a list."""
        c = self._cols.get(name)
        if c is None:
            s = self.df[name]
            if s.dtype.is_float():
                c = s.cast(pl.Float64).to_numpy()
            elif (s.dtype.is_integer() or s.dtype == pl.Boolean) and s.null_count() == 0:
                c = s.to_numpy()
            else:
                c = s.to_list()
            self._cols[name] = c
        return c

    def idx(self, tick: int) -> int | None:
        """Index of the last row with tick <= `tick`, or None."""
        k = int(np.searchsorted(self.tick, tick, side="right")) - 1
        return None if k < 0 else k

    def at(self, name: str, tick: int):
        k = self.idx(tick)
        return None if k is None else _nan(self.col(name)[k])

    def pos(self, tick: int) -> tuple[float, float, float] | None:
        k = self.idx(tick)
        if k is None:
            return None
        p = tuple(_nan(self.col(c)[k]) for c in ("x", "y", "z"))
        return None if any(v is None for v in p) else tuple(float(v) for v in p)

    def alive(self, tick: int) -> bool:
        return bool(self.at("is_alive", tick))

    def span(self, lo: int, hi: int) -> slice:
        """Row slice with lo <= tick <= hi."""
        return slice(int(np.searchsorted(self.tick, lo, side="left")),
                     int(np.searchsorted(self.tick, hi, side="right")))


# ── context ───────────────────────────────────────────────────────────────────

class Context:
    """Lookups shared by every extractor of one match."""

    def __init__(self, raw, cfg, match_key: str, geometry=None):
        self.raw, self.cfg, self.match_key, self.geometry = raw, cfg, match_key, geometry
        self.tick_rate = int(raw.meta["tick_rate"])
        self.off       = int(cfg.sampling.boundary_offset_ticks)
        self.rounds    = {int(r["round_num"]): r for r in raw.rounds.to_dicts()}
        self.terminal: dict[int, int] = {}
        self.end: dict[int, int]      = {}
        for r, row in self.rounds.items():
            term = sampler.terminal_tick(row)
            if term is None:
                t = raw.ticks.filter(pl.col("round_num") == r)["tick"]
                term = None if t.len() == 0 else int(t.max())
            if term is None:
                continue
            self.terminal[r] = int(term)
            end = _nan(row.get("end_tick"))
            self.end[r] = int(end) if end is not None else int(term)
        self._starts = sorted((int(self.rounds[r]["start_tick"]), r) for r in self.terminal)

        self.tracks: dict[int, dict[int, Track]] = {}
        self.sides: dict[int, dict[int, str]]    = {}
        if raw.ticks.height:
            parts = raw.ticks.partition_by(["round_num", "steamid"], as_dict=True, maintain_order=True)
            for (r, sid), df in parts.items():
                if r is None or sid is None or int(r) not in self.terminal:
                    continue
                r, sid = int(r), int(sid)
                tr = Track(df)
                fe = int(self.rounds[r]["freeze_end_tick"])
                k  = tr.idx(fe)
                side = tr.col("side")[k if k is not None else 0]
                if side in ("ct", "t"):
                    self.tracks.setdefault(r, {})[sid] = tr
                    self.sides.setdefault(r, {})[sid] = side
        self._pid: dict[int, int] = {}
        self.sid_of_pid: dict[int, int] = {}
        for r in self.sides:
            for sid in self.sides[r]:
                self.sid_of_pid[self.pid(sid)] = sid
        self._fire: dict[int, np.ndarray] | None = None
        self._centroids: dict[str, tuple] | None = None
        self.snaps: dict[int, list] = {}
        self._snap_ticks: dict[int, list[int]] = {}

    # ── rounds ──

    def round_of(self, tick) -> int | None:
        """Round whose [start_tick, terminal] contains `tick`."""
        tick = _nan(tick)
        if tick is None:
            return None
        k = bisect_right(self._starts, (int(tick), float("inf"))) - 1
        if k < 0:
            return None
        r = self._starts[k][1]
        return r if int(tick) <= self.terminal[r] else None

    def check_round(self, round_num, tick) -> int | None:
        """`round_num` if it exists and `tick` lies in it, else None (logged)."""
        r, tick = _nan(round_num), _nan(tick)
        if r is None or tick is None or int(r) not in self.terminal:
            return None
        r = int(r)
        if not int(self.rounds[r]["start_tick"]) <= int(tick) <= self.terminal[r]:
            log.warning(f"round {r}: row at tick {tick} lies outside the round; dropped")
            return None
        return r

    def freeze_end(self, r: int) -> int:
        return int(self.rounds[r]["freeze_end_tick"])

    def round_bounds(self) -> dict[int, tuple[int, int]]:
        """{round: (freeze_end, terminal)}: the timeline P8 partitions."""
        return {r: (self.freeze_end(r), t) for r, t in self.terminal.items()}

    # ── players ──

    def pid(self, sid) -> int | None:
        sid = _nan(sid)
        if sid is None:
            return None
        sid = int(sid)
        h = self._pid.get(sid)
        if h is None:
            h = self._pid[sid] = player_id_hash(sid)
        return h

    def side(self, r: int, sid) -> str | None:
        sid = _nan(sid)
        return None if sid is None else self.sides.get(r, {}).get(int(sid))

    def track(self, r: int, sid) -> Track | None:
        sid = _nan(sid)
        return None if sid is None else self.tracks.get(r, {}).get(int(sid))

    def pos(self, r: int, sid, tick: int):
        tr = self.track(r, sid)
        return None if tr is None else tr.pos(tick)

    def alive(self, r: int, sid, tick: int) -> bool:
        tr = self.track(r, sid)
        return tr is not None and tr.alive(tick)

    def alive_counts(self, r: int, tick: int) -> tuple[int, int]:
        n = {"ct": 0, "t": 0}
        for sid, side in self.sides.get(r, {}).items():
            if self.tracks[r][sid].alive(tick):
                n[side] += 1
        return n["ct"], n["t"]

    def callout(self, pos) -> str | None:
        if self.geometry is None or pos is None or any(_nan(v) is None for v in pos):
            return None
        return self.geometry.callout(float(pos[0]), float(pos[1]), float(pos[2]))

    def fired(self, sid, lo: int, hi: int) -> bool:
        """Did the player fire (weapon_fire) at a tick in (lo, hi]?"""
        if self._fire is None:
            self._fire = {}
            wf = self.raw.weapon_fire
            if wf.height:
                for (s,), df in wf.partition_by("steamid", as_dict=True, maintain_order=True).items():
                    if s is not None:
                        self._fire[int(s)] = np.sort(df["tick"].to_numpy().astype(np.int64))
        t = self._fire.get(int(sid))
        if t is None:
            return False
        return int(np.searchsorted(t, hi, side="right")) > int(np.searchsorted(t, lo, side="right"))

    def first_spotted(self, r: int, spotted_sid, by_sid, lo: int, hi: int) -> int | None:
        """First tick in [lo, hi] where `by_sid` is in `spotted_sid`'s spotted_by list."""
        tr = self.track(r, spotted_sid)
        if tr is None or "spotted_by" not in tr.df.columns:
            return None
        lists = tr.col("spotted_by")
        by = int(by_sid)
        sl = tr.span(lo, hi)
        for k in range(sl.start, sl.stop):
            lst = lists[k]
            if lst is not None and any(s is not None and int(s) == by for s in lst):
                return int(tr.tick[k])
        return None

    # ── bomb ──

    def bomb_pos(self, r: int, tick: int):
        """Where the bomb is at `tick`: the plant position once planted, else the
        carrier's position, else the last drop position; None if unknown."""
        b = self.raw.bomb.filter((pl.col("round_num") == r) & (pl.col("tick") <= tick)).sort("tick")
        planted = b.filter(pl.col("event") == "plant_complete")
        if planted.height:
            row = planted.row(0, named=True)
            p = tuple(_nan(row[c]) for c in ("x", "y", "z"))
            if all(v is not None for v in p):
                return p
            return self.pos(r, row["steamid"], int(row["tick"]))
        for sid in sorted(self.sides.get(r, {})):
            tr = self.tracks[r][sid]
            if tr.alive(tick) and tr.at("has_c4", tick):
                return tr.pos(tick)
        drops = b.filter(pl.col("event") == "drop")
        if drops.height:
            row = drops.row(drops.height - 1, named=True)
            p = tuple(_nan(row[c]) for c in ("x", "y", "z"))
            if all(v is not None for v in p):
                return p
        return None

    # ── snapshots ──

    def set_snapshots(self, snapshots) -> None:
        self.snaps, self._snap_ticks, self._centroids = {}, {}, None
        for s in sorted(snapshots or [], key=lambda s: (s.meta.round_num, s.meta.tick)):
            self.snaps.setdefault(s.meta.round_num, []).append(s)
        for r, lst in self.snaps.items():
            self._snap_ticks[r] = [s.meta.tick for s in lst]

    def snap_at(self, r: int, tick: int, exact: bool = False):
        """The snapshot at `tick` (exact) or the latest one at or before it."""
        ticks = self._snap_ticks.get(r)
        if not ticks:
            return None
        k = bisect_right(ticks, tick) - 1
        if k < 0:
            return None
        s = self.snaps[r][k]
        return s if not exact or s.meta.tick == tick else None

    def grid(self, r: int) -> list:
        """Grid snapshots of round r from freeze_end to the round's end tick."""
        end = self.end.get(r)
        return [s for s in self.snaps.get(r, ())
                if sampler.GRID in s.meta.kinds and (end is None or s.meta.tick <= end)]

    def callout_centroid(self, name: str):
        """(x, y, z, nav_area) for a callout: `geometry.callout_centroid` when the
        geometry has one, else the observed alive position (over this match's grid
        snapshots) closest to the mean of every position seen in that callout."""
        if name is None:
            return None
        g = getattr(self.geometry, "callout_centroid", None)
        if callable(g):
            c = g(name)
            if c is not None:
                area = self.geometry.find_area(tuple(c)) if self.geometry is not None else None
                return (*c, area)
        if self._centroids is None:
            seen: dict[str, list] = {}
            for r in sorted(self.snaps):
                for s in self.grid(r):
                    for p in s.players:
                        if p.alive == 1 and p.callout is not None and None not in (p.x, p.y, p.z):
                            seen.setdefault(p.callout, []).append((p.x, p.y, p.z, p.nav_area))
            self._centroids = {}
            for c, pts in seen.items():
                m = np.mean(np.array([q[:3] for q in pts], dtype=np.float64), axis=0)
                best = min(pts, key=lambda q: (math.dist(q[:3], m), q[:3]))
                self._centroids[c] = best
        return self._centroids.get(name)

    def nav_or_dist(self, a, b, area_a=None, area_b=None) -> float | None:
        """Nav distance between two points (areas looked up when not given), falling
        back to the 3D distance when the geometry cannot answer."""
        if a is None or b is None:
            return None
        g = self.geometry
        if g is not None:
            area_a = area_a if area_a is not None else g.find_area(tuple(a))
            area_b = area_b if area_b is not None else g.find_area(tuple(b))
            if area_a is not None and area_b is not None:
                d = g.nav_distance(area_a, area_b)
                if d is not None and not math.isnan(d):
                    return float(d)
        return math.dist(tuple(a), tuple(b))

    # ── action construction ──

    def instant(self, r: int, type_: str, sid, tick: int, params: dict, *,
                subtype: str | None = None, side: str | None = None) -> Action | None:
        side = side if side is not None else self.side(r, sid)
        pid = self.pid(sid)
        if side is None or pid is None:
            log.warning(f"round {r}: {type_} at {tick}: actor {sid} has no side; dropped")
            return None
        tick = int(tick)
        return Action(match_key=self.match_key, round_num=r, actor_id_hash=pid, actor_side=side,
                      kind=TYPE_KIND[type_], type=type_, subtype=subtype, tick=tick, params=params,
                      state_pre_tick=tick - self.off, state_post_tick=tick + self.off)

    def durative(self, r: int, type_: str, sid, start: int, end: int, params: dict, *,
                 subtype: str | None = None, side: str | None = None) -> Action | None:
        side = side if side is not None else self.side(r, sid)
        pid = self.pid(sid)
        if side is None or pid is None:
            log.warning(f"round {r}: {type_} at {start}: actor {sid} has no side; dropped")
            return None
        start, end = int(start), int(end)
        return Action(match_key=self.match_key, round_num=r, actor_id_hash=pid, actor_side=side,
                      kind="durative", type=type_, subtype=subtype, window_start_tick=start,
                      window_end_tick=end, params=params,
                      state_pre_tick=start - self.off, state_post_tick=end + self.off)

    @staticmethod
    def unique(actions: list[Action], salt_of) -> list[Action]:
        """Give every duplicate id (same match/round/type/actor/tick) a salted id, in
        list order (callers pass a deterministic order). Returns the list."""
        seen: set[str] = set()
        for a in actions:
            if a.action_id in seen:
                a.action_id = make_action_id(a.match_key, a.round_num, a.type, a.actor_id_hash,
                                             a.id_tick, salt=str(salt_of(a)))
            if a.action_id in seen:
                raise ValueError(f"cannot make {a.type} id unique at tick {a.id_tick}")
            seen.add(a.action_id)
        return actions


def make_context(raw, cfg, match_key: str, geometry=None, snapshots=None) -> Context:
    ctx = Context(raw, cfg, match_key, geometry)
    if snapshots is not None:
        ctx.set_snapshots(snapshots)
    return ctx


# ── stages ────────────────────────────────────────────────────────────────────

def _check_registry() -> None:
    missing = missing_extractors()
    if missing:
        raise RuntimeError(f"no extractor registered for {missing}")


def _run_types(types: tuple[str, ...], raw, states, ctx) -> list[Action]:
    _check_registry()
    done, out = [], []
    for t in types:
        fn = REGISTRY[t]
        if fn in done:
            continue
        done.append(fn)
        out += fn(raw, states, ctx)
    return out


def sort_key(a: Action) -> tuple:
    t = a.tick if a.tick is not None else a.window_start_tick
    return (a.round_num, t if t is not None else -1, KIND_ORDER.get(a.kind, 9), a.type, a.action_id)


def extract_event_actions(raw, cfg, match_key: str, *, geometry=None, ctx: Context | None = None) -> list[Action]:
    """Step (1): every action readable from the raw tables alone. `geometry` only adds
    callouts; without it every callout param is None."""
    ctx = ctx if ctx is not None else Context(raw, cfg, match_key, geometry)
    return sorted(_run_types(EVENT_TYPES, raw, None, ctx), key=sort_key)


def extract_movement_actions(snapshots, raw, cfg, match_key: str, geometry,
                             *, ctx: Context | None = None) -> list[Action]:
    """Step (3): peek / rotation / hold / save from the grid snapshots."""
    ctx = ctx if ctx is not None else Context(raw, cfg, match_key, geometry)
    ctx.set_snapshots(snapshots)
    return sorted(_run_types(MOVEMENT_TYPES, raw, snapshots, ctx), key=sort_key)


def attach_enablers(actions: list[Action], snapshots, raw, cfg, *, ctx: Context | None = None) -> list[Action]:
    """Step (4): fill `enablers` in place (also returned)."""
    if ctx is None:
        ctx = Context(raw, cfg, actions[0].match_key if actions else "", None)
    ctx.set_snapshots(snapshots)
    return enablers.attach(actions, ctx)


def group_windows(actions: list[Action], cfg, tick_rate: int, rounds=None) -> list[Window]:
    """Step (5), the P4.6.1 name: see `windows.group`."""
    return winmod.group(actions, cfg, tick_rate, rounds=rounds)


def extract(raw, snapshots, cfg, *, match_key: str, geometry,
            ctx: Context | None = None) -> tuple[list[Action], list[Window]]:
    """Steps (1)-(5) given the snapshots. Every action validates; ids are unique."""
    ctx = ctx if ctx is not None else Context(raw, cfg, match_key, geometry)
    ctx.set_snapshots(snapshots)
    actions = extract_event_actions(raw, cfg, match_key, geometry=geometry, ctx=ctx)
    utility.fill_blocks(actions, ctx)
    actions += extract_movement_actions(snapshots, raw, cfg, match_key, geometry, ctx=ctx)
    actions.sort(key=sort_key)
    enablers.attach(actions, ctx)
    problems = [f"{a.type}@{a.id_tick}: {p}" for a in actions for p in a.validate()]
    if problems:
        raise ValueError("invalid actions:\n  " + "\n  ".join(problems[:20]))
    assert_unique_ids(actions)
    windows = winmod.group(actions, cfg, ctx.tick_rate, rounds=ctx.round_bounds())
    return actions, windows


def run(raw, cfg, *, match_key: str, geometry, tier: str = "pro", **build_kw):
    """Steps (1)-(5) including the state build: (snapshots, actions, windows)."""
    from engine.state import builder

    ctx = Context(raw, cfg, match_key, geometry)
    ev = extract_event_actions(raw, cfg, match_key, geometry=geometry, ctx=ctx)
    at, dw = event_ticks(ev), durative_windows(ev)
    snaps = builder.build(raw, geometry, cfg, match_key=match_key, tier=tier,
                          action_ticks=at, durative_windows=dw, **build_kw)
    mv = extract_movement_actions(snaps, raw, cfg, match_key, geometry, ctx=ctx)
    at2, dw2 = event_ticks(ev + mv), durative_windows(ev + mv)
    if at2 != at or dw2 != dw:
        snaps = builder.build(raw, geometry, cfg, match_key=match_key, tier=tier,
                              action_ticks=at2, durative_windows=dw2, **build_kw)
    actions, windows = extract(raw, snaps, cfg, match_key=match_key, geometry=geometry, ctx=ctx)
    return snaps, actions, windows


# ── builder hand-off ──────────────────────────────────────────────────────────

def event_ticks(actions: list[Action]) -> dict[int, set[int]]:
    """{round: instantaneous ticks} for the builder's boundary samples: every instant
    action (pseudo-actions included) plus a drop's pickup tick."""
    out: dict[int, set[int]] = {}
    for a in actions:
        if a.kind != "instant" or a.tick is None:
            continue
        s = out.setdefault(a.round_num, set())
        s.add(int(a.tick))
        if a.type == "drop" and a.params.get("pickup_tick") is not None:
            s.add(int(a.params["pickup_tick"]))
    return out


def durative_windows(actions: list[Action]) -> dict[int, list[tuple[int, int]]]:
    """{round: sorted unique (start, end)} of every durative action."""
    out: dict[int, set[tuple[int, int]]] = {}
    for a in actions:
        if a.kind == "durative":
            out.setdefault(a.round_num, set()).add((int(a.window_start_tick), int(a.window_end_tick)))
    return {r: sorted(v) for r, v in out.items()}


# ── output (P4.7.1) ───────────────────────────────────────────────────────────

def actions_frame(actions: list[Action]) -> pl.DataFrame:
    rows = [a.to_dict(flatten=True) for a in sorted(actions, key=sort_key)]
    if not rows:
        return pl.DataFrame(schema=ACTION_SCHEMA)
    return pl.DataFrame({k: [r[k] for r in rows] for k in ACTION_SCHEMA}, schema=ACTION_SCHEMA)


def windows_frame(windows: list[Window]) -> pl.DataFrame:
    ws = sorted(windows, key=lambda w: (w.round_num, w.start_tick))
    return pl.DataFrame({"window_id": [w.window_id for w in ws], "round_num": [w.round_num for w in ws],
                         "start_tick": [w.start_tick for w in ws], "end_tick": [w.end_tick for w in ws],
                         "member_action_ids": [list(w.member_action_ids) for w in ws]},
                        schema=WINDOW_SCHEMA)


def write(actions: list[Action], windows: list[Window], out_dir) -> tuple[Path, Path]:
    """`<dir>/actions.parquet` (fields flattened, params/enablers as JSON strings) and
    `<dir>/windows.parquet`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pa, pw = out_dir / "actions.parquet", out_dir / "windows.parquet"
    actions_frame(actions).write_parquet(pa)
    windows_frame(windows).write_parquet(pw)
    return pa, pw


def read(out_dir) -> tuple[list[Action], list[Window]]:
    out_dir = Path(out_dir)
    acts = [Action.from_dict(r) for r in pl.read_parquet(out_dir / "actions.parquet").iter_rows(named=True)]
    wins = [Window(window_id=r["window_id"], round_num=r["round_num"], start_tick=r["start_tick"],
                   end_tick=r["end_tick"], member_action_ids=tuple(r["member_action_ids"] or ()))
            for r in pl.read_parquet(out_dir / "windows.parquet").iter_rows(named=True)]
    return acts, wins
