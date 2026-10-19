"""
Simultaneity grouping (P4.6): instantaneous actions (pseudo pop/fade/start/end actions
included) whose ticks chain within `actions.simultaneity_ticks` of each other form one
event window `[min_tick - 1, max_tick + 1]`; windows never overlap. `coverage` tiles a
round's timeline into event and drift intervals for P8.1.

    from engine.actions import windows
    ws = windows.group(actions, cfg, tick_rate, rounds={1: (freeze_end, terminal)})
    windows.coverage(ws_of_round_1, {"freeze_end_tick": 1000, "official_end_tick": 4360})
    # [("drift", 1000, 1499, None), ("event", 1499, 1501, "<id>"), ("drift", 1501, ...), ...]

Rules:
- Chaining is single-linkage on sorted ticks: a new window starts when the gap to the
  previous member tick is greater than `simultaneity_ticks`. With the default 2, kills
  at the same tick share a window and kills 3 ticks apart get two windows, which are
  then at least one tick apart (no shared tick).
- `simultaneity_ticks` is in ticks, as the config states; it is not rescaled by the
  tick rate (`tick_rate` is checked and kept for future second-based keys).
- With `rounds` ({round: (freeze_end, terminal)}), only actions inside that span are
  grouped (an item drop during freeze time has no WP jump of its own: no snapshot is
  taken before freeze_end) and window bounds are clipped into it.
- Durative and round actions are never members (their pop/fade pseudo-actions are).
- `window_id` = first 16 hex of sha1(`round|start|end|member ids`).
"""
import hashlib
from dataclasses import dataclass

from engine.state import sampler

# ── Window ────────────────────────────────────────────────────────────────────


@dataclass
class Window:
    window_id:         str
    round_num:         int
    start_tick:        int
    end_tick:          int
    member_action_ids: tuple[str, ...]

    def __post_init__(self):
        self.member_action_ids = tuple(self.member_action_ids)


def _window_id(round_num: int, start: int, end: int, members) -> str:
    key = f"{round_num}|{start}|{end}|{','.join(members)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ── grouping ──────────────────────────────────────────────────────────────────

def _span_end(a) -> int:
    """Last tick an instant action's window must reach: tick + 1, or its
    `state_post_tick` when the action set one further out (a drop's pickup: the
    receiver's jump then belongs to the drop, not to drift; MED-15, 2026-10-19)."""
    end = int(a.tick) + 1
    post = getattr(a, "state_post_tick", None)
    if post is not None and int(post) > end:
        end = int(post)
    return end


def group(actions, cfg, tick_rate: int, rounds: dict | None = None) -> list[Window]:
    """Event windows of every round, sorted by (round, start). Two actions share a
    window when their ticks are within `simultaneity_ticks`, or when one falls inside
    the other's span (see `_span_end`)."""
    if int(tick_rate) <= 0:
        raise ValueError(f"tick_rate must be > 0, got {tick_rate}")
    sim = int(cfg.actions.simultaneity_ticks)
    by_round: dict[int, list[tuple[int, int, str]]] = {}
    for a in actions:
        if a.kind != "instant" or a.tick is None:
            continue
        if rounds is not None:
            b = rounds.get(a.round_num)
            if b is None or not (b[0] <= a.tick <= b[1]):
                continue
        by_round.setdefault(a.round_num, []).append((int(a.tick), _span_end(a), a.action_id))

    out: list[Window] = []
    for r in sorted(by_round):
        items = sorted(by_round[r])
        clusters: list[list[tuple[int, int, str]]] = []
        for it in items:
            if clusters and (it[0] - clusters[-1][-1][0] <= sim or it[0] <= max(e for _, e, _ in clusters[-1])):
                clusters[-1].append(it)
            else:
                clusters.append([it])
        lo_b, hi_b = rounds[r] if rounds is not None else (None, None)
        for c in clusters:
            start, end = c[0][0] - 1, max(e for _, e, _ in c)
            if lo_b is not None:
                start, end = max(lo_b, start), min(hi_b, end)
            members = tuple(aid for _, _, aid in c)
            out.append(Window(window_id=_window_id(r, start, end, members), round_num=r,
                              start_tick=start, end_tick=end, member_action_ids=members))
    return out


# ── checks and coverage (P4.6.2, consumed by P8.1) ────────────────────────────

def overlaps(windows) -> list[tuple[str, str]]:
    """Pairs of windows of one round that share a tick (should be empty)."""
    bad = []
    by_round: dict[int, list[Window]] = {}
    for w in windows:
        by_round.setdefault(w.round_num, []).append(w)
    for ws in by_round.values():
        ws = sorted(ws, key=lambda w: (w.start_tick, w.end_tick))
        for a, b in zip(ws, ws[1:]):
            if b.start_tick <= a.end_tick:
                bad.append((a.window_id, b.window_id))
    return bad


def coverage(windows, round_row: dict) -> list[tuple[str, int, int, str | None]]:
    """Tile [freeze_end, terminal] of one round into consecutive intervals sharing
    their end points: ("event", start, end, window_id) for each window and ("drift",
    a, b, None) for each gap. Raises ValueError when windows overlap or fall outside."""
    fe   = int(round_row["freeze_end_tick"])
    term = sampler.terminal_tick(round_row)
    if term is None:
        raise ValueError("round has no terminal tick")
    r  = round_row.get("round_num")
    ws = sorted((w for w in windows if r is None or w.round_num == r), key=lambda w: w.start_tick)
    out: list[tuple[str, int, int, str | None]] = []
    cur = fe
    for w in ws:
        if w.start_tick < cur or w.end_tick > term or w.start_tick > w.end_tick:
            raise ValueError(f"window {w.window_id} [{w.start_tick}, {w.end_tick}] overlaps or "
                             f"leaves [{fe}, {term}] (cursor {cur})")
        if w.start_tick > cur:
            out.append(("drift", cur, w.start_tick, None))
        out.append(("event", w.start_tick, w.end_tick, w.window_id))
        cur = w.end_tick
    if cur < term:
        out.append(("drift", cur, term, None))
    return out
