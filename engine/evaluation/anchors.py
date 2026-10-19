"""
External anchors (P11.1): sanity bands from published work, checked on scored pro data.
Informational only: a result outside its band is flagged in the report and discussed
in the review loop (P11.2), never a failure.

    from engine.evaluation import anchors
    frame = anchors.enrich(scored, actions, states)        # adds the ANCHOR_COLUMNS
    anchors.opening_kill_band(frame)                       # {"status": "in_band", "value": 0.18, ...}
    res = anchors.run(frame, states_with_w, cfg, out_path, hltv_csv=path, model="wp_v<N>")

Inputs
- `scored`: an `actions_scored` frame (`engine.value.scores.SCORED_SCHEMA`).
- `actions`: the P4 actions frame (`engine.actions.extractor.ACTION_SCHEMA`: tick,
  state_pre_tick, params JSON, enablers JSON).
- `states`: states rows (`Snapshot.to_row`: `ctx_*` columns, `players` list, `y_*`
  labels). `man_advantage_ordering` additionally needs a `W` column (CT win
  probability of the evaluated model at that state), joined by the caller.

`enrich` adds (null when unknown, never 0 or "?"): tick, state_pre_tick, subtype,
weapon_class, victim_id, team_class, enabler_kinds, alive_ct_pre, alive_t_pre,
equip_ct_pre, equip_t_pre, time_remaining_s_pre, bomb_planted_pre, map_version,
victim_weapon_class, round_won_ct, round_end_reason. The pre-state is the state at
`state_pre_tick`, else the latest state of the round before it (causal).

Anchors (bands are module constants below; a config section `anchors` with the same
lower-case names overrides them when it exists; `engine/config.yaml` has none today):
- P11.1.1 opening kill: mean |V_direct| of the first kill of each round in a 5v5 with
  equipment ratio (CT / T, symmetric band so orientation does not matter) in
  [0.8, 1.25] lies in [0.12, 0.30]; distribution by map version.
- P11.1.2 duel difficulty: disadvantage = tier(victim weapon) − tier(killer weapon)
  (WEAPON_TIER); Spearman over disadvantage levels of mean V > 0, and per
  alive-difference bucket rifle-vs-rifle mean V > rifle-vs-pistol mean V.
- P11.1.3 man advantage: mean W (CT) in 3v2 > 2v3 > 1v2, strata matched on a 10 s
  time-remaining bucket and bomb state, weighted by the smallest of the three counts;
  plus `kill_swing_ordering`: mean |V_direct| of a kill that turns 2v1 into 1v1 is
  larger than one that turns 5v4 into 4v4.
- P11.1.4 clutch defuse: `defuse_start` with CT alone (1vX) and < 10 s left has mean
  V > 0; `defuse_abort` in the same situation mean V < 0.
- P11.1.5 player rating: Spearman between summed V per player-map and the HLTV rating
  in `test/engine/fixtures/hltv_ratings.csv` >= 0.6; biggest disagreements listed.

Filled by: P11.1.1 - P11.1.6.
"""
import argparse
import csv
import json
import math
from bisect import bisect_right
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

# ── constants (bands are informational, P11.1) ────────────────────────────────

IN_BAND, OUT_OF_BAND, NO_DATA = "in_band", "out_of_band", "no_data"

OPENING_BAND       = (0.12, 0.30)     # mean |V_direct| of the opening kill (P11.1.1)
EQUIP_RATIO_BAND   = (0.8, 1.25)      # CT / T equipment ratio counted as balanced (P11.1.1)
RATING_RHO_MIN     = 0.6              # Spearman vs HLTV rating, informational (P11.1.5)
RATING_MIN_MATCHES = 50               # matches the plan asks for in the CSV (P11.1.5)
CLUTCH_TIME_S      = 10.0             # s left for a clutch defuse (P11.1.4)
TIME_BUCKET_S      = 10.0             # s, time-remaining strata for man advantage (P11.1.3)
MIN_GROUP_N        = 5                # rows a group needs before it is compared
ALIVE_CLIP         = 3                # alive-difference buckets clipped to [-3, +3]
TOP_DISAGREEMENTS  = 10

# ordinal weapon strength for the duel-difficulty anchor; classes not listed
# (grenade, c4, unknown, None) are excluded rather than guessed
WEAPON_TIER = {"knife": 0, "zeus": 0, "pistol": 1, "shotgun": 2, "smg": 2,
               "lmg": 3, "rifle": 3, "sniper": 3}
KILL_TYPES  = ("kill",)
DUEL_PAIRS  = (("rifle", "rifle"), ("rifle", "pistol"))     # (killer, victim)

ANCHOR_COLUMNS = ("tick", "state_pre_tick", "subtype", "weapon_class", "victim_id", "team_class",
                  "enabler_kinds", "alive_ct_pre", "alive_t_pre", "equip_ct_pre", "equip_t_pre",
                  "time_remaining_s_pre", "bomb_planted_pre", "map_version", "victim_weapon_class",
                  "round_won_ct", "round_end_reason")
HLTV_COLUMNS = ("match_key", "map", "steamid64", "player_name", "hltv_rating")


# ── small helpers ─────────────────────────────────────────────────────────────

def _num(v) -> float | None:
    """float, or None for None / NaN / non-numbers (never 0 for unknown)."""
    if v is None or isinstance(v, bool):
        return None if v is None else float(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _has(frame: pl.DataFrame, *cols: str) -> list[str]:
    """Columns of `cols` missing from `frame`."""
    return [c for c in cols if c not in frame.columns]


def _first_col(frame: pl.DataFrame, *names: str) -> str | None:
    return next((n for n in names if n in frame.columns), None)


def _mean(xs) -> float | None:
    xs = [x for x in (_num(v) for v in xs) if x is not None]
    return float(np.mean(xs)) if xs else None


def rank(x) -> np.ndarray:
    """1-based ranks, ties get their average rank."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y) -> float | None:
    """Spearman rank correlation; None with < 3 pairs or a constant side."""
    pairs = [(a, b) for a, b in zip((_num(v) for v in x), (_num(v) for v in y)) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    rx, ry = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _band(cfg, name: str, default):
    """`cfg.anchors.<name>` when the config has an anchors section, else `default`."""
    section = getattr(cfg, "anchors", None) if cfg is not None else None
    return getattr(section, name, default) if section is not None else default


def _actor_enemy_alive(row: dict) -> tuple[float | None, float | None]:
    ct, t = _num(row.get("alive_ct_pre")), _num(row.get("alive_t_pre"))
    if row.get("actor_side") == "ct":
        return ct, t
    if row.get("actor_side") == "t":
        return t, ct
    return None, None


def _no_data(name: str, reason: str, **extra) -> dict:
    return {"name": name, "status": NO_DATA, "reason": reason, "n": 0, **extra}


# ── enrichment ────────────────────────────────────────────────────────────────

def _loads(s, default):
    if s is None:
        return default
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default


def _state_index(states: pl.DataFrame) -> dict:
    """(match_key, round_num) -> (sorted ticks, rows) for causal lookups."""
    idx: dict = {}
    for r in states.iter_rows(named=True):
        idx.setdefault((r.get("match_key"), r.get("round_num")), []).append(r)
    out = {}
    for k, rows in idx.items():
        rows.sort(key=lambda r: r["tick"])
        out[k] = ([r["tick"] for r in rows], rows)
    return out


def _state_at(index: dict, match_key, round_num, tick) -> dict | None:
    """The state at `tick`, else the latest one before it in the round."""
    if tick is None:
        return None
    entry = index.get((match_key, round_num))
    if entry is None:
        return None
    ticks, rows = entry
    i = bisect_right(ticks, tick) - 1
    return rows[i] if i >= 0 else None


def _round_label(entry, key: str):
    """First non-null `y_<key>` of a round's states (labels repeat on every row)."""
    if entry is None:
        return None
    for r in entry[1]:
        v = r.get(f"y_{key}")
        if v is not None:
            return v
    return None


def enrich(scored: pl.DataFrame, actions: pl.DataFrame | None = None,
           states: pl.DataFrame | None = None) -> pl.DataFrame:
    """The scored frame plus ANCHOR_COLUMNS joined from the actions and states frames.
    Columns that cannot be filled are null. Existing columns of `scored` win."""
    act = {}
    if actions is not None and actions.height:
        for r in actions.iter_rows(named=True):
            act[r["action_id"]] = r
    index = _state_index(states) if states is not None and states.height else {}

    rows = []
    for r in scored.iter_rows(named=True):
        a = act.get(r["action_id"], {})
        params = _loads(a.get("params"), {}) or {}
        enab = _loads(a.get("enablers"), []) or []
        pre_tick = a.get("state_pre_tick")
        tick = a.get("tick")
        s = _state_at(index, r["match_key"], r["round_num"], pre_tick if pre_tick is not None else
                      (None if tick is None else tick - 1))
        entry = index.get((r["match_key"], r["round_num"]))
        victim = params.get("victim_id")
        vwc = None
        if s is not None and victim is not None:
            for p in s.get("players") or []:
                if p.get("player_id") == victim:
                    vwc = p.get("weapon_class")
                    break
        alive_ct = s.get("ctx_alive_ct") if s is not None else params.get("ct_alive")
        alive_t = s.get("ctx_alive_t") if s is not None else params.get("t_alive")
        extra = {
            "tick": tick, "state_pre_tick": pre_tick, "subtype": a.get("subtype"),
            "weapon_class": params.get("weapon_class"), "victim_id": victim,
            "team_class": params.get("team_class"),
            "enabler_kinds": sorted({e.get("kind") for e in enab if isinstance(e, dict) and e.get("kind")}),
            "alive_ct_pre": alive_ct, "alive_t_pre": alive_t,
            "equip_ct_pre": s.get("ctx_equip_ct") if s is not None else None,
            "equip_t_pre": s.get("ctx_equip_t") if s is not None else None,
            "time_remaining_s_pre": s.get("ctx_time_remaining_s") if s is not None else None,
            "bomb_planted_pre": s.get("ctx_bomb_planted") if s is not None else None,
            "map_version": s.get("ctx_map_version") if s is not None else None,
            "victim_weapon_class": vwc,
            "round_won_ct": _round_label(entry, "round_won_ct"),
            "round_end_reason": _round_label(entry, "round_end_reason"),
        }
        rows.append({**extra, **{k: v for k, v in r.items()}})
    schema = dict(scored.schema)
    add = {"tick": pl.Int64, "state_pre_tick": pl.Int64, "subtype": pl.String, "weapon_class": pl.String,
           "victim_id": pl.Int64, "team_class": pl.String, "enabler_kinds": pl.List(pl.String),
           "alive_ct_pre": pl.Int64, "alive_t_pre": pl.Int64, "equip_ct_pre": pl.Float64,
           "equip_t_pre": pl.Float64, "time_remaining_s_pre": pl.Float64, "bomb_planted_pre": pl.Int64,
           "map_version": pl.String, "victim_weapon_class": pl.String, "round_won_ct": pl.Int64,
           "round_end_reason": pl.String}
    for k, t in add.items():
        schema.setdefault(k, t)
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


# ── P11.1.1 opening kill ──────────────────────────────────────────────────────

def first_kills(frame: pl.DataFrame) -> pl.DataFrame:
    """The first `kill` of every round (by tick, then action_id)."""
    k = frame.filter(pl.col("type").is_in(list(KILL_TYPES)) & pl.col("tick").is_not_null())
    return (k.sort(["match_key", "round_num", "tick", "action_id"])
             .group_by(["match_key", "round_num"], maintain_order=True).first())


def opening_kill_band(frame: pl.DataFrame, band: tuple[float, float] = OPENING_BAND,
                      ratio_band: tuple[float, float] = EQUIP_RATIO_BAND) -> dict:
    """Mean |V_direct| of balanced 5v5 opening kills; in/out of `band`, by map version.
    A kill with an unknown alive count is skipped and counted (`n_unknown_alive`), never
    taken for 5v5 (LOW-39)."""
    name = "opening_kill"
    miss = _has(frame, "type", "tick", "V_direct", "equip_ct_pre", "equip_t_pre")
    if miss:
        return _no_data(name, f"missing columns {miss}", band=list(band))
    fk = first_kills(frame)
    vals, by_map, n_unbalanced, n_unknown, n_unknown_alive = [], {}, 0, 0, 0
    for r in fk.iter_rows(named=True):
        ct, t = _num(r.get("equip_ct_pre")), _num(r.get("equip_t_pre"))
        if ct is None or t is None or t <= 0:
            n_unknown += 1
            continue
        a_ct, a_t = _num(r.get("alive_ct_pre")), _num(r.get("alive_t_pre"))
        if a_ct is None or a_t is None:
            n_unknown_alive += 1
            continue
        if a_ct != 5 or a_t != 5:
            n_unbalanced += 1
            continue
        if not ratio_band[0] <= ct / t <= ratio_band[1]:
            n_unbalanced += 1
            continue
        v = _num(r.get("V_direct"))
        if v is None:
            continue
        vals.append(abs(v))
        by_map.setdefault(r.get("map_version") or "unknown", []).append(abs(v))
    if not vals:
        return _no_data(name, "no balanced 5v5 opening kills", band=list(band),
                        n_rounds=fk.height, n_excluded=n_unbalanced, n_unknown_equipment=n_unknown,
                        n_unknown_alive=n_unknown_alive)
    value = float(np.mean(vals))
    return {"name": name, "status": IN_BAND if band[0] <= value <= band[1] else OUT_OF_BAND,
            "value": value, "band": list(band), "n": len(vals), "n_rounds": fk.height,
            "n_excluded": n_unbalanced, "n_unknown_equipment": n_unknown, "n_unknown_alive": n_unknown_alive,
            "median": float(np.median(vals)),
            "by_map": {m: {"n": len(v), "mean": float(np.mean(v))} for m, v in sorted(by_map.items())}}


# ── P11.1.2 duel difficulty ───────────────────────────────────────────────────

def disadvantage(killer_class: str | None, victim_class: str | None) -> int | None:
    """tier(victim) − tier(killer): positive = the killer had the weaker weapon."""
    a, b = WEAPON_TIER.get(killer_class), WEAPON_TIER.get(victim_class)
    return None if a is None or b is None else b - a


def duel_difficulty(frame: pl.DataFrame, min_n: int = MIN_GROUP_N) -> dict:
    """Mean V by weapon disadvantage (Spearman > 0) and rifle-vs-rifle > rifle-vs-pistol
    within every alive-difference bucket where both have >= min_n kills."""
    name = "duel_difficulty"
    miss = _has(frame, "type", "V", "weapon_class", "victim_weapon_class")
    if miss:
        return _no_data(name, f"missing columns {miss}")
    levels: dict[int, list[float]] = {}
    pairs: dict[tuple, list[float]] = {}
    for r in frame.filter(pl.col("type").is_in(list(KILL_TYPES))).iter_rows(named=True):
        v = _num(r.get("V"))
        d = disadvantage(r.get("weapon_class"), r.get("victim_weapon_class"))
        if v is None or d is None:
            continue
        levels.setdefault(d, []).append(v)
        mine, theirs = _actor_enemy_alive(r)
        bucket = "all" if mine is None or theirs is None else int(max(-ALIVE_CLIP, min(ALIVE_CLIP, mine - theirs)))
        pair = (r.get("weapon_class"), r.get("victim_weapon_class"))
        if pair in DUEL_PAIRS:
            pairs.setdefault((bucket, pair), []).append(v)
    table = {d: {"n": len(v), "mean_V": float(np.mean(v))} for d, v in sorted(levels.items())}
    usable = [d for d, t in table.items() if t["n"] >= min_n]
    rho = spearman(usable, [table[d]["mean_V"] for d in usable])
    comps = []
    for b in sorted({k[0] for k in pairs}, key=str):
        rr, rp = pairs.get((b, DUEL_PAIRS[0]), []), pairs.get((b, DUEL_PAIRS[1]), [])
        if len(rr) >= min_n and len(rp) >= min_n:
            comps.append({"alive_diff": b, "rifle_vs_rifle": float(np.mean(rr)), "n_rr": len(rr),
                          "rifle_vs_pistol": float(np.mean(rp)), "n_rp": len(rp),
                          "ok": float(np.mean(rr)) > float(np.mean(rp))})
    n = sum(len(v) for v in levels.values())
    if rho is None and not comps:
        return _no_data(name, f"fewer than 3 disadvantage levels and no rifle comparisons with n >= {min_n}",
                        levels=table)
    ok = (rho is None or rho > 0) and all(c["ok"] for c in comps)
    return {"name": name, "status": IN_BAND if ok else OUT_OF_BAND, "value": rho, "band": "rho > 0",
            "n": n, "levels": table, "comparisons": comps}


# ── P11.1.3 man advantage ─────────────────────────────────────────────────────

SITUATIONS = {"3v2": (3, 2), "2v3": (2, 3), "1v2": (1, 2)}      # (alive CT, alive T)


def man_advantage_ordering(states: pl.DataFrame, min_n: int = MIN_GROUP_N,
                           bucket_s: float = TIME_BUCKET_S) -> dict:
    """Mean W (CT) in 3v2 > 2v3 > 1v2 over strata matched on time bucket and bomb state."""
    name = "man_advantage"
    w = _first_col(states, "W", "w_ct")
    ct = _first_col(states, "ctx_alive_ct", "alive_ct")
    t = _first_col(states, "ctx_alive_t", "alive_t")
    tr = _first_col(states, "ctx_time_remaining_s", "time_remaining_s")
    bp = _first_col(states, "ctx_bomb_planted", "bomb_planted")
    if None in (w, ct, t, tr, bp):
        return _no_data(name, "states need W, alive_ct, alive_t, time_remaining_s, bomb_planted columns")
    cells: dict[tuple, dict[str, list[float]]] = {}
    for r in states.iter_rows(named=True):
        wv, rem = _num(r[w]), _num(r[tr])
        if wv is None or rem is None or r[bp] is None:
            continue
        sit = next((s for s, (a, b) in SITUATIONS.items() if r[ct] == a and r[t] == b), None)
        if sit is None:
            continue
        key = (int(rem // bucket_s), int(r[bp]))
        cells.setdefault(key, {s: [] for s in SITUATIONS})[sit].append(wv)
    num = {s: 0.0 for s in SITUATIONS}
    den = 0
    n_strata = 0
    for key, by in cells.items():
        m = min(len(by[s]) for s in SITUATIONS)
        if m < min_n:
            continue
        n_strata += 1
        den += m
        for s in SITUATIONS:
            num[s] += m * float(np.mean(by[s]))
    raw = {s: _mean(v for by in cells.values() for v in by[s]) for s in SITUATIONS}
    if den == 0:
        return _no_data(name, f"no stratum with >= {min_n} states in all of 3v2, 2v3, 1v2", raw_means=raw)
    means = {s: num[s] / den for s in SITUATIONS}
    ok = means["3v2"] > means["2v3"] > means["1v2"]
    return {"name": name, "status": IN_BAND if ok else OUT_OF_BAND, "value": means,
            "band": "W(3v2) > W(2v3) > W(1v2)", "n": den, "n_strata": n_strata, "raw_means": raw}


def kill_swing_ordering(frame: pl.DataFrame, min_n: int = MIN_GROUP_N) -> dict:
    """Mean |V_direct| of a kill turning 2v1 into 1v1 > one turning 5v4 into 4v4."""
    name = "kill_swing"
    miss = _has(frame, "type", "V_direct", "alive_ct_pre", "alive_t_pre", "actor_side")
    if miss:
        return _no_data(name, f"missing columns {miss}")
    groups = {"2v1->1v1": [], "5v4->4v4": []}
    for r in frame.filter(pl.col("type").is_in(list(KILL_TYPES))).iter_rows(named=True):
        mine, theirs = _actor_enemy_alive(r)
        v = _num(r.get("V_direct"))
        if v is None:
            continue
        if (mine, theirs) == (1, 2):
            groups["2v1->1v1"].append(abs(v))
        elif (mine, theirs) == (4, 5):
            groups["5v4->4v4"].append(abs(v))
    if any(len(g) < min_n for g in groups.values()):
        return _no_data(name, f"need >= {min_n} kills in both situations",
                        counts={k: len(g) for k, g in groups.items()})
    means = {k: float(np.mean(g)) for k, g in groups.items()}
    return {"name": name, "status": IN_BAND if means["2v1->1v1"] > means["5v4->4v4"] else OUT_OF_BAND,
            "value": means, "band": "|V| 2v1->1v1 > 5v4->4v4", "n": sum(len(g) for g in groups.values())}


# ── P11.1.4 clutch defuse ─────────────────────────────────────────────────────

def clutch_defuse_sign(frame: pl.DataFrame, time_s: float = CLUTCH_TIME_S) -> dict:
    """defuse_start in a 1vX with < time_s left: mean V > 0; defuse_abort: mean V < 0."""
    name = "clutch_defuse"
    miss = _has(frame, "type", "V", "alive_ct_pre", "alive_t_pre", "time_remaining_s_pre")
    if miss:
        return _no_data(name, f"missing columns {miss}")
    got: dict[str, list[float]] = {"defuse_start": [], "defuse_abort": []}
    for r in frame.filter(pl.col("type").is_in(list(got))).iter_rows(named=True):
        rem, v = _num(r.get("time_remaining_s_pre")), _num(r.get("V"))
        a_ct, a_t = _num(r.get("alive_ct_pre")), _num(r.get("alive_t_pre"))
        if None in (rem, v, a_ct, a_t) or a_ct != 1 or a_t < 1 or rem >= time_s:
            continue
        got[r["type"]].append(v)
    starts, aborts = got["defuse_start"], got["defuse_abort"]
    if not starts:
        return _no_data(name, f"no defuse_start in a 1vX with < {time_s:g} s left", n_abort=len(aborts))
    m_start = float(np.mean(starts))
    m_abort = float(np.mean(aborts)) if aborts else None
    ok = m_start > 0 and (m_abort is None or m_abort < 0)
    return {"name": name, "status": IN_BAND if ok else OUT_OF_BAND,
            "value": {"defuse_start": m_start, "defuse_abort": m_abort},
            "band": "start > 0, abort < 0", "n": len(starts) + len(aborts), "n_start": len(starts),
            "n_abort": len(aborts), "share_start_positive": float(np.mean([v > 0 for v in starts])),
            "note": None if aborts else "no aborts in this situation; only the start sign was checked"}


# ── P11.1.5 player rating ─────────────────────────────────────────────────────

def read_hltv_csv(path: Path) -> list[dict]:
    """Rows of the HLTV ratings CSV (header HLTV_COLUMNS); blank lines skipped."""
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        miss = [c for c in HLTV_COLUMNS if c not in (rd.fieldnames or [])]
        if miss:
            raise ValueError(f"{Path(path).name}: missing columns {miss}")
        return [r for r in rd if any((v or "").strip() for v in r.values())]


def player_rating_correlation(frame: pl.DataFrame, hltv_csv: Path | None,
                              rho_min: float = RATING_RHO_MIN) -> dict:
    """Spearman between summed V per (match, player) and the HLTV per-map rating."""
    from engine.state.players import player_id_hash
    name = "player_rating"
    if hltv_csv is None or not Path(hltv_csv).is_file():
        return _no_data(name, "no HLTV ratings CSV", band=f"rho >= {rho_min}")
    rows = read_hltv_csv(Path(hltv_csv))
    if not rows:
        return _no_data(name, f"{Path(hltv_csv).name} has no rows yet (collect >= {RATING_MIN_MATCHES} matches)",
                        band=f"rho >= {rho_min}")
    sums: dict[tuple, float] = {}
    for r in frame.filter(pl.col("actor_id_hash").is_not_null()).iter_rows(named=True):
        v = _num(r.get("V"))
        if v is not None:
            key = (r["match_key"], r["actor_id_hash"])
            sums[key] = sums.get(key, 0.0) + v
    pairs = []
    for r in rows:
        rating = _num(r.get("hltv_rating"))
        try:
            pid = player_id_hash(int(str(r["steamid64"]).strip()))
        except ValueError:
            continue
        key = (r["match_key"].strip(), pid)
        if rating is not None and key in sums:
            # pname, not name: `name` is the anchor's name returned below (shadowed until 2026-10-19)
            pname = (r.get("player_name") or "").strip().replace("|", "\\|")
            if not pname:
                from engine.report.writer import player_label
                pname = player_label(pid)                # never the raw SteamID64 (R14, MED-36)
            pairs.append({"match_key": key[0], "player": pname, "sum_V": sums[key], "rating": rating})
    if len(pairs) < 3:
        return _no_data(name, f"only {len(pairs)} player-maps matched the scored frame", band=f"rho >= {rho_min}")
    rho = spearman([p["sum_V"] for p in pairs], [p["rating"] for p in pairs])
    rv, rr = rank([p["sum_V"] for p in pairs]), rank([p["rating"] for p in pairs])
    for p, a, b in zip(pairs, rv, rr):
        p["rank_gap"] = float(a - b)
    worst = sorted(pairs, key=lambda p: -abs(p["rank_gap"]))[:TOP_DISAGREEMENTS]
    n_matches = len({p["match_key"] for p in pairs})
    status = NO_DATA if rho is None else (IN_BAND if rho >= rho_min else OUT_OF_BAND)
    return {"name": name, "status": status, "value": rho, "band": f"rho >= {rho_min}", "n": len(pairs),
            "n_matches": n_matches, "disagreements": worst,
            "note": None if n_matches >= RATING_MIN_MATCHES else
            f"only {n_matches} matches (plan: >= {RATING_MIN_MATCHES})"}


# ── report ────────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_fmt(x)}" for k, x in v.items())
    if isinstance(v, float):
        return f"{v:+.4f}"
    return str(v)


def _band_txt(b) -> str:
    return f"[{b[0]:g}, {b[1]:g}]" if isinstance(b, (list, tuple)) else str(b or "–")


def report_markdown(results: list[dict], model: str | None, n_actions: int) -> str:
    flag = {IN_BAND: "in band", OUT_OF_BAND: "**OUT OF BAND**", NO_DATA: "no data"}
    lines = [f"# External anchors — {model or 'unnamed model'}", "",
             f"Date {date.today().isoformat()} · {n_actions} scored actions · P11.1.", "",
             "Bands are informational: an out-of-band result is flagged here and discussed in the review "
             "loop (P11.2), never a failure.", "",
             "| anchor | result | band | status | n |", "|---|---|---|---|---:|"]
    for r in results:
        lines.append(f"| {r['name']} | {_fmt(r.get('value'))} | {_band_txt(r.get('band'))} | "
                     f"{flag[r['status']]} | {r.get('n', 0)} |")
    for r in results:
        lines += ["", f"## {r['name']}", ""]
        if r["status"] == NO_DATA:
            lines.append(f"No data: {r.get('reason')}.")
        if r.get("note"):
            lines.append(f"Note: {r['note']}.")
        if r.get("by_map"):
            lines += ["| map version | n | mean abs V_direct |", "|---|---:|---:|"]
            lines += [f"| {m} | {d['n']} | {d['mean']:.4f} |" for m, d in r["by_map"].items()]
        if r.get("levels"):
            lines += ["| disadvantage | n | mean V |", "|---:|---:|---:|"]
            lines += [f"| {d} | {t['n']} | {t['mean_V']:+.4f} |" for d, t in r["levels"].items()]
        if r.get("comparisons"):
            lines += ["", "| alive diff | rifle vs rifle | n | rifle vs pistol | n | ok |", "|---|---:|---:|---:|---:|---|"]
            lines += [f"| {c['alive_diff']} | {c['rifle_vs_rifle']:+.4f} | {c['n_rr']} | {c['rifle_vs_pistol']:+.4f} | "
                      f"{c['n_rp']} | {'yes' if c['ok'] else 'no'} |" for c in r["comparisons"]]
        if r.get("raw_means"):
            lines.append(f"Unmatched means: {_fmt(r['raw_means'])}; matched strata: {r.get('n_strata', 0)}.")
        if r.get("disagreements"):
            lines += ["Biggest disagreements (for the review loop):", "",
                      "| match | player | sum V | HLTV rating | rank gap |", "|---|---|---:|---:|---:|"]
            lines += [f"| {p['match_key']} | {p['player']} | {p['sum_V']:+.3f} | {p['rating']:.2f} | "
                      f"{p['rank_gap']:+.1f} |" for p in r["disagreements"]]
    return "\n".join(lines) + "\n"


def run(frame: pl.DataFrame, states: pl.DataFrame | None, cfg, out_path: Path, *,
        hltv_csv: Path | None = None, model: str | None = None) -> list[dict]:
    """Every anchor on an enriched scored frame (+ states with W); writes the report."""
    results = [
        opening_kill_band(frame, tuple(_band(cfg, "opening_band", OPENING_BAND)),
                          tuple(_band(cfg, "equip_ratio_band", EQUIP_RATIO_BAND))),
        duel_difficulty(frame),
        man_advantage_ordering(states) if states is not None else _no_data("man_advantage", "no states frame"),
        kill_swing_ordering(frame),
        clutch_defuse_sign(frame, float(_band(cfg, "clutch_time_s", CLUTCH_TIME_S))),
        player_rating_correlation(frame, hltv_csv, float(_band(cfg, "rating_rho_min", RATING_RHO_MIN))),
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_markdown(results, model, frame.height), encoding="utf-8")
    return results


# ── command line ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """`engine eval --anchors`: needs scored pro test data; exits 2 without it."""
    from engine import paths
    ap = argparse.ArgumentParser(prog="engine eval --anchors", description="External anchors report (P11.1).")
    ap.add_argument("--model-dir", default=None, help="model directory (names the report)")
    ap.add_argument("--scored", type=Path, default=None, help="actions_scored parquet (test pro matches)")
    ap.add_argument("--actions", type=Path, default=None, help="matching P4 actions parquet")
    ap.add_argument("--states", type=Path, default=None, help="states parquet; a W column enables man advantage")
    ap.add_argument("--hltv-csv", type=Path, default=paths.EVAL_DIR / "hltv_ratings.csv",
                    help="hand-collected HLTV ratings (gitignored under data/engine/eval; the header-only "
                         "template is test/engine/fixtures/hltv_ratings.csv) (MED-37)")
    ap.add_argument("--out", type=Path, default=None, help="report path (default data/engine/eval/anchors_<model>.md)")
    a = ap.parse_args(argv)
    if a.scored is None or not a.scored.is_file():
        print("anchors: needs scored pro test data (--scored actions_scored.parquet, plus --actions and "
              "--states for the context columns). None exists until a model is trained and test pro "
              "matches are analysed (P6, P10).")
        return 2
    from engine import config
    model = Path(a.model_dir).name if a.model_dir else None
    scored = pl.read_parquet(a.scored)
    actions = pl.read_parquet(a.actions) if a.actions and a.actions.is_file() else None
    states = pl.read_parquet(a.states) if a.states and a.states.is_file() else None
    frame = enrich(scored, actions, states)
    out = a.out or paths.EVAL_DIR / f"anchors_{model or 'unnamed'}.md"
    results = run(frame, states, config.load(), out, hltv_csv=a.hltv_csv, model=model)
    for r in results:
        print(f"{r['name']}: {r['status']} {_fmt(r.get('value'))}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
