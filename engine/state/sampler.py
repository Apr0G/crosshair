"""
Tick selection (P3.1): grid ticks every `tick_rate / sample_hz` from freeze_end to the
terminal tick, boundary ticks at every action tick +/- offset, and their union with a
`kind` list per tick. The round end tick is used only to know when to stop (P3.1.4).

    from engine.state import sampler
    for tick, kinds in sampler.all_ticks(round_row, tick_rate, 2, action_ticks={2000}):
        ...

Grid count: `floor(duration_s * sample_hz) + 1` regular ticks (freeze_end, +step, ...);
when the duration is not a whole number of steps the terminal tick is appended as one
extra sample, so the last tick is always the terminal tick (both P3.1.1 statements
hold: the formula when the duration divides evenly, "last is terminal" always).
"""
import math

# ── kinds ─────────────────────────────────────────────────────────────────────

GRID, PRE, POST, TERMINAL = "grid", "boundary_pre", "boundary_post", "terminal"
KIND_ORDER = (GRID, PRE, POST, TERMINAL)


def terminal_tick(round_row: dict) -> int | None:
    """The tick sampling stops at: official_end_tick, else end_tick, else None."""
    for col in ("official_end_tick", "end_tick"):
        v = round_row.get(col)
        if v is not None:
            return int(v)
    return None


def step_ticks(tick_rate: int, sample_hz: int) -> int:
    """Grid step in ticks, rounded to an integer (64 at 128 tick / 2 Hz)."""
    if tick_rate <= 0 or sample_hz <= 0:
        raise ValueError(f"tick_rate and sample_hz must be > 0, got {tick_rate}, {sample_hz}")
    return max(1, int(round(tick_rate / sample_hz)))


# ── grid ──────────────────────────────────────────────────────────────────────

def grid_ticks(round_row: dict, tick_rate: int, sample_hz: int, terminal: int | None = None) -> list[int]:
    """Grid ticks from freeze_end to the terminal tick (inclusive), deduplicated."""
    start = int(round_row["freeze_end_tick"])
    end   = terminal if terminal is not None else terminal_tick(round_row)
    if end is None:
        raise ValueError(f"round {round_row.get('round_num')}: no terminal tick")
    if end < start:
        return [end]
    step  = tick_rate / sample_hz
    n     = math.floor((end - start) / tick_rate * sample_hz) + 1
    ticks = sorted({min(end, start + int(round(k * step))) for k in range(n)})
    if ticks[-1] != end:
        ticks.append(end)
    return ticks


# ── boundary ──────────────────────────────────────────────────────────────────

def boundary_ticks(action_ticks, durative_windows=(), offset: int = 1,
                   lo: int | None = None, hi: int | None = None) -> dict[int, set[str]]:
    """{tick: {boundary_pre | boundary_post}} for every instantaneous action tick and
    both ends of every durative window `(start, end)` (an end of None is skipped),
    each +/- `offset`, clipped into [lo, hi]."""
    def clip(t: int) -> int:
        if lo is not None:
            t = max(lo, t)
        if hi is not None:
            t = min(hi, t)
        return t

    taus = {int(t) for t in action_ticks}
    for w in durative_windows:
        taus.update(int(t) for t in w if t is not None)
    out: dict[int, set[str]] = {}
    for tau in taus:
        out.setdefault(clip(tau - offset), set()).add(PRE)
        out.setdefault(clip(tau + offset), set()).add(POST)
    return out


# ── union ─────────────────────────────────────────────────────────────────────

def all_ticks(round_row: dict, tick_rate: int, sample_hz: int, action_ticks=(),
              durative_windows=(), offset: int = 1,
              terminal: int | None = None) -> list[tuple[int, tuple[str, ...]]]:
    """Sorted, unique `(tick, kinds)`; kinds in KIND_ORDER."""
    end   = terminal if terminal is not None else terminal_tick(round_row)
    grid  = grid_ticks(round_row, tick_rate, sample_hz, terminal=end)
    kinds: dict[int, set[str]] = {t: {GRID} for t in grid}
    for t, ks in boundary_ticks(action_ticks, durative_windows, offset,
                                lo=int(round_row["freeze_end_tick"]), hi=end).items():
        kinds.setdefault(t, set()).update(ks)
    kinds.setdefault(end, set()).add(TERMINAL)
    return [(t, tuple(k for k in KIND_ORDER if k in kinds[t])) for t in sorted(kinds)]
