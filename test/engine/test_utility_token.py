"""
Utility tokens (P3.3): active window, time_left_s from the nominal lifetime (never the
real end tick), fires and decoys, owner fields, the 16-token cap with one log line.

    .venv\\Scripts\\python.exe test\\engine\\test_utility_token.py
"""
import logging
import sys

import polars as pl

from _harness import check, close, section, summary
from _synthetic import PLAYERS, make_raw_tables

from engine import config
from engine.parse.schema import TABLE_SCHEMAS
from engine.state import builder, players, utility
from engine.state.builder import RoundView
from engine.state.geometry import FlatGeometry

CFG = config.load()


class _Catch(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def with_rows(raw, table: str, rows: list[dict]):
    f = raw.frames()
    f[table] = pl.concat([getattr(raw, table), pl.DataFrame(rows, schema=TABLE_SCHEMAS[table])])
    return type(raw)(**f, meta=dict(raw.meta))


A_ID, C_ID = PLAYERS["A"][0], PLAYERS["C"][0]

for tr in (64, 128):
    T = lambda t128: int(round(t128 * tr / 128))  # noqa: E731
    raw = make_raw_tables("base", tr)
    v = RoundView.build(raw, 1)
    s0, s_end = T(1500), T(1500) + 18 * tr

    # ── active window ──
    section(f"smoke window @ {tr} tick")
    toks = lambda t: utility.tokens(v.until(t))[0]  # noqa: E731
    check("no token before the smoke pops", toks(s0 - 1) == [])
    check("token at the pop tick and at 1600", len(toks(s0)) == 1 and len(toks(T(1600))) == 1)
    check("token one tick before the end", len(toks(s_end - 1)) == 1)
    check("gone at the end tick (end_tick <= tick is known)", toks(s_end) == [])
    check("gone at 1500 + 19 s", toks(s0 + 19 * tr) == [])
    u = toks(T(1600))[0]
    check("kind smoke, position (1000,0,0), radius 144", (u.kind, u.x, u.y, u.z, u.radius_units)
          == ("smoke", 1000.0, 0.0, 0.0, 144.0))
    check("owner side ct, owner id = hash(A)", u.owner_side == "ct" and u.owner_id == players.player_id_hash(A_ID))
    check("time_left_s 18.0 at the pop", close(toks(s0)[0].time_left_s, 18.0))
    check("time_left_s 1/tick_rate one tick before the end", close(toks(s_end - 1)[0].time_left_s, 1 / tr))

    # ── time_left decrements by 0.5 per grid step at 2 Hz ──
    snaps = builder.build(raw, FlatGeometry(), CFG, match_key="u", tier="pro", labels=False)
    grid = [(s.meta.tick, s.utility[0].time_left_s) for s in snaps
            if s.meta.round_num == 1 and s.meta.kinds == ("grid",) and s.utility]
    steps = [round(a[1] - b[1], 9) for a, b in zip(grid, grid[1:])]
    check(f"time_left_s decreases by 0.5 per grid step ({len(grid)} grid snapshots)",
          len(grid) >= 30 and all(close(x, 0.5, 1e-9) for x in steps), str(steps[:5]))
    check("ctx.n_active_smokes follows the token", all(s.ctx.n_active_smokes == len(s.utility) for s in snaps))

    # ── nominal, not real, lifetime ──
    section(f"nominal lifetime @ {tr} tick")
    short = with_rows(raw, "smokes", [dict(entity_id=900, start_tick=T(2500), end_tick=T(2500) + 5 * tr,
                                           x=0.0, y=0.0, z=0.0, thrower_steamid=C_ID)])
    vs = RoundView.build(short, 1)
    us = [u for u in utility.tokens(vs.until(T(2500) + tr))[0] if u.owner_side == "t"]
    check("a smoke that will fade early still shows 18 - elapsed (17.0), not its real end",
          len(us) == 1 and close(us[0].time_left_s, 17.0))
    check("and disappears once its end tick is in the past",
          not [u for u in utility.tokens(vs.until(T(2500) + 5 * tr))[0] if u.owner_side == "t"])
    check("time_left floors at 0 for an unknown end",
          close(utility.tokens(RoundView.build(with_rows(raw, "smokes", [dict(
              entity_id=901, start_tick=T(1000), end_tick=None, x=0.0, y=0.0, z=0.0, thrower_steamid=C_ID)]),
              1).until(T(1000) + 30 * tr))[0][0].time_left_s, 0.0))

    # ── fire and decoy ──
    section(f"fire and decoy @ {tr} tick")
    fire = with_rows(raw, "infernos", [dict(entity_id=950, start_tick=T(2200), end_tick=T(2200) + 7 * tr,
                                            x=3000.0, y=300.0, z=0.0, extent_radius=150.0, thrower_steamid=A_ID)])
    fire = with_rows(fire, "grenades", [dict(entity_id=960, thrower_steamid=C_ID, type="decoy",
                                             throw_tick=T(1100), detonate_tick=T(1200),
                                             expire_tick=T(1200) + 15 * tr, x=10.0, y=20.0, z=0.0)])
    vf = RoundView.build(fire, 1)
    ut, n_sm, n_fi = utility.tokens(vf.until(T(2200) + tr))
    kinds = sorted(u.kind for u in ut)
    check("smoke + fire + decoy active together", kinds == ["decoy", "fire", "smoke"], str(kinds))
    fz = next(u for u in ut if u.kind == "fire")
    check("fire radius from extent, time_left 7 - 1", fz.radius_units == 150.0 and close(fz.time_left_s, 6.0))
    check("inside_flag: 1 inside the known fire, 0 outside it, 0 with no fire",
          utility.inside_flag(ut, "fire", 3000.0, 300.0, 0.0) == 1
          and utility.inside_flag(ut, "fire", 0.0, 0.0, 0.0) == 0
          and utility.inside_flag([u for u in ut if u.kind != "fire"], "fire", 3000.0, 300.0, 0.0) == 0)
    nof = with_rows(raw, "infernos", [dict(entity_id=951, start_tick=T(2200), end_tick=T(2200) + 7 * tr,
                                           x=3000.0, y=300.0, z=0.0, extent_radius=None, thrower_steamid=A_ID)])
    un, _, _ = utility.tokens(RoundView.build(nof, 1).until(T(2200) + tr))
    check("inside_flag: None (not 0) while a fire of unknown extent burns (HIGH-04)",
          utility.inside_flag(un, "fire", 3000.0, 300.0, 0.0) is None
          and utility.inside_flag(un, "fire", 0.0, 0.0, 0.0) is None
          and utility.inside_flag(un, "smoke", 0.0, 0.0, 0.0) in (0, 1))
    dz = next(u for u in ut if u.kind == "decoy")
    check("decoy: radius None, owner t, time_left 15 - elapsed",
          dz.radius_units is None and dz.owner_side == "t" and close(dz.time_left_s, 15.0 - (T(2200) + tr - T(1200)) / tr))
    check("counts returned before the cap", (n_sm, n_fi) == (1, 1))
    check("decoy not active before it detonates", not [u for u in utility.tokens(vf.until(T(1150)))[0] if u.kind == "decoy"])
    check("inside(): fire disc", utility.inside(fz, 3000.0, 400.0, 0.0) and not utility.inside(fz, 3000.0, 500.0, 0.0))
    check("inside(): decoy never", not utility.inside(dz, 10.0, 20.0, 0.0))

# ── cap ──
section("cap at 16, newest kept, one log line")
raw = make_raw_tables("base", 128)
many = with_rows(raw, "smokes", [dict(entity_id=1000 + k, start_tick=1600 + k, end_tick=None,
                                      x=float(k), y=0.0, z=0.0, thrower_steamid=A_ID) for k in range(19)])
h = _Catch()
logging.getLogger("engine.state.utility").addHandler(h)
logging.getLogger("engine.state.utility").setLevel(logging.INFO)
ut, n_sm, _ = utility.tokens(RoundView.build(many, 1).until(1700), max_tokens=CFG.sampling.max_utility_tokens)
logging.getLogger("engine.state.utility").removeHandler(h)
check("20 active smokes -> 16 tokens", n_sm == 20 and len(ut) == 16, f"{n_sm} {len(ut)}")
check("newest kept (the first 4 by start tick dropped)", sorted(u.x for u in ut) == [float(k) for k in range(3, 19)])
check("exactly one log line", len(h.lines) == 1 and "cap" in h.lines[0], str(h.lines))

sys.exit(summary())
