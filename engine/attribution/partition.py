"""
Round timeline partition (P8.1): breakpoints from grid samples and window bounds,
intervals labelled `event` or `drift`, tiling [freeze_end, terminal] exactly.

    from engine.attribution import partition
    ivs = partition.breakpoints(snapshots, windows)      # [Interval(t0, t1, kind, window_id)]
    W   = partition.wp_at_breakpoints(predictor, snapshots, ivs)   # {tick: W}, one call
    partition.telescoping_residual(ivs, W)               # Σ ΔW_i − (W(terminal) − W(freeze_end))

Inputs: `snapshots` are one round's factual snapshots (any order; ticks unique); the
first tick is freeze_end and the last is the terminal tick. A window is anything with
the fields of `engine.actions.windows.Window` (`window_id`, `round_num`, `start_tick`,
`end_tick`, `member_action_ids`) as attributes or dict keys. P4.6.1 sets
`start_tick = min member tick − 1` and `end_tick = max member tick + 1`, i.e. the
window's own boundary_pre / boundary_post sample ticks; the event interval runs from
the latest snapshot at or before `start_tick` to the earliest at or after `end_tick`
(clamped to the round when a boundary sample was clipped at freeze_end / terminal).
Snapshots strictly inside an event interval are dropped as breakpoints.

Filled by: P8.1.1-P8.1.3.
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

# ── constants ─────────────────────────────────────────────────────────────────

EVENT, DRIFT = "event", "drift"
WINDOW_FIELDS = ("window_id", "round_num", "start_tick", "end_tick", "member_action_ids")


@dataclass(frozen=True)
class Interval:
    start_tick: int
    end_tick:   int
    kind:       str
    window_id:  str | None = None


# ── small accessors ───────────────────────────────────────────────────────────

def wfield(window, name: str):
    """A window field from a `Window` object or a dict with the same names."""
    if isinstance(window, dict):
        return window[name]
    return getattr(window, name)


def tick_of(snapshot) -> int:
    return int(snapshot.meta.tick)


def sorted_round(snapshots) -> list:
    """Snapshots sorted by tick; duplicate ticks are an error (the sampler merges kinds)."""
    out = sorted(snapshots, key=tick_of)
    ticks = [tick_of(s) for s in out]
    if len(set(ticks)) != len(ticks):
        raise ValueError("snapshots of one round have duplicate ticks")
    if not out:
        raise ValueError("no snapshots")
    return out


def histories_for(snapshots_sorted: list) -> dict[int, list]:
    """{tick: factual snapshots of the round with a smaller tick} (shared list slices)."""
    return {tick_of(s): snapshots_sorted[:i] for i, s in enumerate(snapshots_sorted)}


# ── P8.1.1 breakpoints ────────────────────────────────────────────────────────

def window_span(ticks: list[int], window) -> tuple[int, int, list[str]]:
    """(pre_tick, post_tick, flags) of a window on the sorted snapshot ticks."""
    flags = []
    st, en = int(wfield(window, "start_tick")), int(wfield(window, "end_tick"))
    if st > en:
        raise ValueError(f"window {wfield(window, 'window_id')}: start_tick {st} > end_tick {en}")
    i = bisect_right(ticks, st) - 1
    if i < 0:
        i, flags = 0, flags + ["pre_clamped_to_freeze_end"]
    j = bisect_left(ticks, en)
    if j >= len(ticks):
        j, flags = len(ticks) - 1, flags + ["post_clamped_to_terminal"]
    if ticks[i] >= ticks[j]:
        raise ValueError(f"window {wfield(window, 'window_id')} [{st}, {en}] has no snapshot "
                         f"interval on this round (pre {ticks[i]}, post {ticks[j]})")
    return ticks[i], ticks[j], flags


def breakpoints(snapshots, windows) -> list[Interval]:
    """Intervals tiling [freeze_end, terminal]: one `event` interval per window (its
    pre/post samples), `drift` intervals between consecutive remaining samples.
    Windows outside the round's tick range, or overlapping each other, are errors."""
    snaps = sorted_round(snapshots)
    ticks = [tick_of(s) for s in snaps]
    spans = []
    for w in windows:
        pre, post, _flags = window_span(ticks, w)
        spans.append((pre, post, str(wfield(w, "window_id"))))
    spans.sort()
    for (a0, a1, aid), (b0, b1, bid) in zip(spans, spans[1:]):
        if b0 < a1:
            raise ValueError(f"windows {aid} [{a0}, {a1}] and {bid} [{b0}, {b1}] overlap")
    ids = [s[2] for s in spans]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate window_id")

    inside = set()
    for pre, post, _ in spans:
        lo, hi = bisect_right(ticks, pre), bisect_left(ticks, post)
        inside.update(ticks[lo:hi])
    bps = [t for t in ticks if t not in inside]
    event_of = {(pre, post): wid for pre, post, wid in spans}
    out = []
    for t0, t1 in zip(bps, bps[1:]):
        wid = event_of.get((t0, t1))
        out.append(Interval(t0, t1, EVENT if wid is not None else DRIFT, wid))
    return out


def check_tiling(intervals: list[Interval], freeze_end: int, terminal: int) -> list[str]:
    """Problems with the tiling: gaps, overlaps, ends not at [freeze_end, terminal]."""
    p = []
    if not intervals:
        return [] if freeze_end == terminal else ["no intervals"]
    if intervals[0].start_tick != freeze_end:
        p.append(f"first interval starts at {intervals[0].start_tick}, not freeze_end {freeze_end}")
    if intervals[-1].end_tick != terminal:
        p.append(f"last interval ends at {intervals[-1].end_tick}, not terminal {terminal}")
    for a, b in zip(intervals, intervals[1:]):
        if a.end_tick != b.start_tick:
            p.append(f"{'gap' if a.end_tick < b.start_tick else 'overlap'} between "
                     f"[{a.start_tick}, {a.end_tick}] and [{b.start_tick}, {b.end_tick}]")
    for iv in intervals:
        if iv.start_tick >= iv.end_tick:
            p.append(f"empty interval [{iv.start_tick}, {iv.end_tick}]")
    return p


def breakpoint_ticks(intervals: list[Interval]) -> list[int]:
    if not intervals:
        return []
    return [intervals[0].start_tick] + [iv.end_tick for iv in intervals]


# ── P8.1.2 W at breakpoints ───────────────────────────────────────────────────

def wp_at_breakpoints(predictor, snapshots, intervals: list[Interval]) -> dict[int, float]:
    """{tick: W} for every breakpoint, from one batched predictor call with factual
    histories (every earlier snapshot of the round)."""
    snaps = sorted_round(snapshots)
    by_tick = {tick_of(s): s for s in snaps}
    hist = histories_for(snaps)
    bts = breakpoint_ticks(intervals)
    out = predictor.predict([by_tick[t] for t in bts], [hist[t] for t in bts])
    return {t: float(w) for t, w in zip(bts, out["win"])}


# ── P8.1.3 telescoping ────────────────────────────────────────────────────────

def interval_deltas(intervals: list[Interval], W: dict[int, float]) -> list[float]:
    return [W[iv.end_tick] - W[iv.start_tick] for iv in intervals]


def telescoping_residual(intervals: list[Interval], W: dict[int, float]) -> float:
    """Σ_i ΔW_i − (W(terminal) − W(freeze_end)); zero up to floating error."""
    if not intervals:
        return 0.0
    total = W[intervals[-1].end_tick] - W[intervals[0].start_tick]
    return float(sum(interval_deltas(intervals, W)) - total)
