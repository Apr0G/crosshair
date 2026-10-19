"""
Bomb events (P4.2.6): plant_start, plant_complete, defuse_start, defuse_complete,
defuse_abort, bomb_drop, bomb_pickup — one action per `bomb` row of those events, with
the actor from the row's steamid (rows without one, and `explode`, produce nothing).
`defuse_abort` comes straight from the parser's event.

Params (Appendix A.5):
- plant_start / plant_complete: site, planter_callout (bomb row position, else the
  planter's tick row), ct_alive, t_alive (alive counts at tick - 1).
- defuse_start / defuse_complete / defuse_abort: kit (the row's has_kit, else the
  defuser's has_defuser at tick - 1), ct_alive, t_alive.
- bomb_drop / bomb_pickup: callout.
"""
from engine.actions.types import Action, register
from engine.state.players import _nan

# ── constants ─────────────────────────────────────────────────────────────────

EVENT_TYPE = {"plant_start": "plant_start", "plant_complete": "plant_complete",
              "defuse_start": "defuse_start", "defuse_complete": "defuse_complete",
              "defuse_abort": "defuse_abort", "drop": "bomb_drop", "pickup": "bomb_pickup"}


def _pos(row: dict):
    p = tuple(_nan(row.get(c)) for c in ("x", "y", "z"))
    return None if any(v is None for v in p) else tuple(float(v) for v in p)


# ── extractor ─────────────────────────────────────────────────────────────────

@register("plant_start")
@register("plant_complete")
@register("defuse_start")
@register("defuse_complete")
@register("defuse_abort")
@register("bomb_drop")
@register("bomb_pickup")
def extract_bomb(raw, states, ctx) -> list[Action]:
    out: list[Action] = []
    seen: set[tuple] = set()
    df = raw.bomb.sort(["tick", "event"], nulls_last=True, maintain_order=True)
    for row in df.iter_rows(named=True):
        type_ = EVENT_TYPE.get(row["event"])
        sid = _nan(row["steamid"])
        r = ctx.check_round(row["round_num"], row["tick"])
        if type_ is None or sid is None or r is None:
            continue
        tick = int(row["tick"])
        if (type_, int(sid), tick) in seen:      # a parser duplicate of the same event
            continue
        seen.add((type_, int(sid), tick))
        pos  = _pos(row) or ctx.pos(r, sid, tick)
        if type_ in ("bomb_drop", "bomb_pickup"):
            params = {"callout": ctx.callout(pos)}
        else:
            ct, t = ctx.alive_counts(r, tick - 1)
            if type_.startswith("plant"):
                params = {"site": _nan(row["site"]), "planter_callout": ctx.callout(pos),
                          "ct_alive": ct, "t_alive": t}
            else:
                kit = _nan(row["has_kit"])
                if kit is None:
                    tr = ctx.track(r, sid)
                    kit = tr.at("has_defuser", tick - 1) if tr is not None else None
                params = {"kit": None if kit is None else bool(kit), "ct_alive": ct, "t_alive": t}
        a = ctx.instant(r, type_, sid, tick, params)
        if a is not None:
            out.append(a)
    return out
