"""
Context token (P3.4, Appendix A.4): live round clock (never from the round end tick),
score, economy, alive counts, bomb state, map_version, tier.

    from engine.state import context
    ctx = context.token(sl, player_tokens, n_smokes, n_fires, clock=cfg.clock,
                        map_version=geometry.map_version, tier="pro")

Rules:
- time_remaining_s = round_time_s - (tick - freeze_end)/tick_rate before the plant and
  bomb_timer_s - (tick - plant_tick)/tick_rate after it, floored at 0.
- halves assume MR12 (regulation halves of 12 rounds, overtime halves of 3): half is
  1, 2 in regulation and 3, 4, ... in overtime; is_overtime = round_num > 24.
- loss_streak_<side> counts the consecutive rounds the team now on <side> lost right
  before this round, within the same half (teams swap sides at half time and the loss
  bonus resets there), from past rounds' winners only.
- score_ct / score_t come from rounds.ct_score_before / t_score_before (known when the
  round starts); if null they are counted from past winners in the same way.
- money_<side> sums every roster player's current account (dead players still have
  one); equip_<side> sums alive players' equipment_value (None if one is unknown).
- defuse state comes from bomb events: a defuse_start with no later defuse_abort /
  defuse_complete / explode, by a defuser who is still alive.
"""
import logging

from engine.state.players import _first_valid, _flag, _float, _nan
from engine.state.schema import ContextToken

log = logging.getLogger("engine.state.context")

# ── constants ─────────────────────────────────────────────────────────────────

REG_HALF_ROUNDS = 12
REG_ROUNDS      = 2 * REG_HALF_ROUNDS
OT_HALF_ROUNDS  = 3
_DEFUSE_END     = ("defuse_abort", "defuse_complete", "explode")
_SITES_LOGGED: set[str] = set()


def half_of(round_num: int) -> int:
    r = int(round_num)
    if r <= REG_ROUNDS:
        return 1 if r <= REG_HALF_ROUNDS else 2
    return 3 + (r - REG_ROUNDS - 1) // OT_HALF_ROUNDS


def site_name(site) -> str | None:
    """'A' / 'B' from parser spellings (A, a, BombsiteA, bombsite_b); None (logged once
    per spelling) if unknown."""
    s = _nan(site)
    if s is None:
        return None
    s = str(s).strip().upper()
    if s.endswith("A"):
        return "A"
    if s.endswith("B"):
        return "B"
    if s not in _SITES_LOGGED:
        _SITES_LOGGED.add(s)
        log.warning(f"unrecognised bomb site {site!r} -> None")
    return None


def loss_streaks(past_rounds: list[dict], round_num: int) -> tuple[int, int]:
    """(loss_streak_ct, loss_streak_t) entering `round_num`, past rounds only."""
    h = half_of(round_num)
    streak = {"ct": 0, "t": 0}
    done   = {"ct": False, "t": False}
    for r in sorted(past_rounds, key=lambda r: r["round_num"], reverse=True):
        if r["round_num"] >= round_num:
            continue
        if half_of(r["round_num"]) != h:
            break
        w = _nan(r.get("winner_side"))
        if w not in ("ct", "t"):
            break
        for side in ("ct", "t"):
            if done[side]:
                continue
            if w == side:
                done[side] = True
            else:
                streak[side] += 1
    return streak["ct"], streak["t"]


def _past_score(past_rounds: list[dict], round_num: int) -> tuple[int, int]:
    """Rounds won by the team now on CT / T, counted from past winners (side-aware)."""
    h_now = half_of(round_num)
    ct = t = 0
    for r in past_rounds:
        if r["round_num"] >= round_num:
            continue
        w = _nan(r.get("winner_side"))
        if w not in ("ct", "t"):
            continue
        swapped = (half_of(r["round_num"]) - h_now) % 2 == 1
        same_team_as_ct = (w == "ct") != swapped
        if same_team_as_ct:
            ct += 1
        else:
            t += 1
    return ct, t


def bomb_state(sl, players) -> dict:
    """plant tick / site and defuse state from bomb events with tick <= sl.tick."""
    st = {"plant_tick": None, "site": None, "defuser": None, "kit": None}
    b = sl.bomb
    if b.n == 0:
        return st
    for ev, tick, sid, site, kit in zip(b.col("event"), b.col("tick"), b.col("steamid"),
                                        b.col("site"), b.col("has_kit")):
        if ev == "plant_complete" and st["plant_tick"] is None:
            st["plant_tick"], st["site"] = int(tick), site_name(site)
        elif ev == "defuse_start":
            st["defuser"], st["kit"] = _nan(sid), _flag(kit)
        elif ev in _DEFUSE_END:
            st["defuser"], st["kit"] = None, None
    return st


def token(sl, players, n_smokes: int, n_fires: int, *, clock, map_version: str | None,
          tier: str) -> ContextToken:
    """The A.4 context token at sl.tick. `players` are the 10 PlayerTokens in slot
    order (pads included); `clock` has round_time_s and bomb_timer_s."""
    tr, t  = sl.tick_rate, sl.tick
    rr     = sl.round
    rnum   = int(rr["round_num"])
    fe     = int(rr["freeze_end_tick"])
    bs     = bomb_state(sl, players)
    plant  = bs["plant_tick"]

    into = max(0.0, (t - fe) / tr)
    if plant is not None:
        since = (t - plant) / tr
        remaining = max(0.0, float(clock.bomb_timer_s) - since)
    else:
        since = None
        remaining = max(0.0, min(float(clock.round_time_s), float(clock.round_time_s) - (t - fe) / tr))

    alive = {s: [p for p in players if p.side == s and p.alive == 1] for s in ("ct", "t")}

    def equip(side):
        vals = [p.equipment_value for p in alive[side]]
        return None if any(v is None for v in vals) else float(sum(vals))

    def money(side):
        vals = []
        for sid, sd in sl.roster:
            if sd != side:
                continue
            pt = sl.players.get(sid)
            vals.append(_float(pt.cols["money"][pt.n - 1]) if pt is not None and pt.n else None)
        return None if any(v is None for v in vals) else float(sum(vals))

    # defuse: in progress only while the defuser is alive
    defuser = bs["defuser"]
    in_prog, kit = 0, None
    if defuser is not None and plant is not None:
        pt = sl.players.get(defuser)
        if pt is not None and pt.n and bool(_nan(pt.cols["is_alive"][pt.n - 1])):
            in_prog = 1
            kit = _first_valid(bs["kit"], _flag(pt.cols["has_defuser"][pt.n - 1]))

    carrier = None
    if plant is None:
        flags = [p.has_c4 for p in alive["t"]]
        carrier = 1 if 1 in flags else (0 if all(f is not None for f in flags) else None)

    ls_ct, ls_t = loss_streaks(sl.past_rounds, rnum)
    pc, pt_ = _past_score(sl.past_rounds, rnum)
    score_ct = _first_valid(rr.get("ct_score_before"), pc)
    score_t  = _first_valid(rr.get("t_score_before"), pt_)

    return ContextToken(
        map_version=map_version, tier=tier, time_remaining_s=remaining, time_into_round_s=into,
        bomb_planted=int(plant is not None), bomb_site=bs["site"] if plant is not None else None,
        time_since_plant_s=since, defuse_in_progress=in_prog, defuse_with_kit=kit,
        bomb_carrier_alive=carrier, alive_ct=len(alive["ct"]), alive_t=len(alive["t"]),
        equip_ct=equip("ct"), equip_t=equip("t"), money_ct=money("ct"), money_t=money("t"),
        loss_streak_ct=ls_ct, loss_streak_t=ls_t, score_ct=int(score_ct), score_t=int(score_t),
        round_num=rnum, half=half_of(rnum), is_overtime=int(rnum > REG_ROUNDS),
        n_active_smokes=n_smokes, n_active_fires=n_fires)
