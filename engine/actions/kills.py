"""
Kill, team_kill and grouped non-lethal damage actions (P4.2.1, P4.2.2).

    from engine.actions import kills
    acts = kills.extract_kills(raw, None, ctx) + kills.extract_damage(raw, None, ctx)

Kill (one per `kills` row): suicides and world kills (attacker null or equal to the
victim) are skipped; `team_kill` true (or, when the flag is null, attacker and victim
on one side) gives `type = team_kill` with the same params. Params (Appendix A.5):
weapon (short name), weapon_class, headshot, through_smoke, attacker_blind,
victim_blind, wallbang (`penetrated > 0`), noscope, distance_units, attacker_callout,
victim_callout, victim_health_before (victim's tick row at tick - 1), assister_id,
flash_assister_id, victim_id. Ids are player id hashes. `state_pre_tick = tick - 1`,
`state_post_tick = tick + 1` (sampling.boundary_offset_ticks).

Damage: every non-lethal damage row (attacker set, attacker != victim, and not the
lethal hit of a kill row with the same tick, attacker and victim) grouped per
(attacker, victim) into engagements; a gap of `actions.damage_gap_s` or more between
two hits of the pair starts a new engagement. One `damage` action per engagement at the
first hit, `window_end_tick` at the last hit (window metadata, kind stays instant).
Params: damage_total (health), hits, weapon_class (most frequent, first on ties),
armor_damage, victim_id. Team damage is kept with `subtype = "team"`.
"""
import math

from engine.actions.types import Action, register
from engine.state.players import _first_valid, _nan, short_name, weapon_class

# ── helpers ───────────────────────────────────────────────────────────────────


def _bool(v) -> bool | None:
    v = _nan(v)
    return None if v is None else bool(v)


def _round(v, nd: int = 3) -> float | None:
    v = _nan(v)
    return None if v is None else round(float(v), nd)


def _xyz(row: dict, prefix: str):
    p = tuple(_nan(row.get(f"{prefix}{c}")) for c in ("x", "y", "z"))
    return None if any(v is None for v in p) else tuple(float(v) for v in p)


# ── kills (P4.2.1) ────────────────────────────────────────────────────────────

@register("kill")
@register("team_kill")
def extract_kills(raw, states, ctx) -> list[Action]:
    out: list[Action] = []
    df = raw.kills.sort(["tick", "attacker_steamid", "victim_steamid"], nulls_last=True, maintain_order=True)
    for row in df.iter_rows(named=True):
        r = ctx.check_round(row["round_num"], row["tick"])
        if r is None:
            continue
        att, vic, tick = _nan(row["attacker_steamid"]), _nan(row["victim_steamid"]), int(row["tick"])
        if att is None or vic is None or att == vic:
            continue
        tk = _bool(row.get("team_kill"))
        if tk is None:
            sa, sv = ctx.side(r, att), ctx.side(r, vic)
            tk = sa is not None and sa == sv
        apos = _xyz(row, "attacker_") or ctx.pos(r, att, tick)
        vpos = _xyz(row, "victim_") or ctx.pos(r, vic, tick)
        dist = _nan(row.get("distance"))
        if dist is None and apos is not None and vpos is not None:
            dist = math.dist(apos, vpos)
        pen  = _nan(row.get("penetrated"))
        vt   = ctx.track(r, vic)
        hp   = vt.at("health", tick - 1) if vt is not None else None
        w    = _nan(row.get("weapon"))
        params = {
            "weapon":               short_name(w) or w,
            "weapon_class":         weapon_class(w),
            "headshot":             _bool(row.get("headshot")),
            "through_smoke":        _bool(row.get("through_smoke")),
            "attacker_blind":       _bool(row.get("attacker_blind")),
            "victim_blind":         _bool(row.get("victim_blind")),
            "wallbang":             None if pen is None else bool(pen > 0),
            "noscope":              _bool(row.get("noscope")),
            "distance_units":       _round(dist),
            "attacker_callout":     ctx.callout(apos),
            "victim_callout":       ctx.callout(vpos),
            "victim_health_before": None if hp is None else int(hp),
            "assister_id":          ctx.pid(row.get("assister_steamid")),
            "flash_assister_id":    ctx.pid(row.get("flash_assister_steamid")),
            "victim_id":            ctx.pid(vic),
        }
        a = ctx.instant(r, "team_kill" if tk else "kill", att, tick, params)
        if a is not None:
            out.append(a)
    return ctx.unique(out, lambda a: a.params["victim_id"])


# ── damage engagements (P4.2.2) ───────────────────────────────────────────────

def _lethal_keys(raw) -> set[tuple]:
    k = raw.kills
    return {(int(r), int(t), int(a), int(v)) for r, t, a, v in
            zip(k["round_num"].to_list(), k["tick"].to_list(), k["attacker_steamid"].to_list(),
                k["victim_steamid"].to_list())
            if None not in (r, t, a, v)}


def _mode(values: list):
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    return next(v for v in values if counts[v] == best)


@register("damage")
def extract_damage(raw, states, ctx) -> list[Action]:
    gap = float(ctx.cfg.actions.damage_gap_s) * ctx.tick_rate
    lethal = _lethal_keys(raw)
    pairs: dict[tuple[int, int, int], list[dict]] = {}
    df = raw.damages.sort(["tick", "attacker_steamid", "victim_steamid"], nulls_last=True, maintain_order=True)
    for row in df.iter_rows(named=True):
        r = ctx.check_round(row["round_num"], row["tick"])
        att, vic = _nan(row["attacker_steamid"]), _nan(row["victim_steamid"])
        if r is None or att is None or vic is None or att == vic:
            continue
        if (r, int(row["tick"]), int(att), int(vic)) in lethal:
            continue
        pairs.setdefault((r, int(att), int(vic)), []).append(row)

    out: list[Action] = []
    for (r, att, vic) in sorted(pairs):
        engagements: list[list[dict]] = []
        for row in pairs[(r, att, vic)]:
            if engagements and int(row["tick"]) - int(engagements[-1][-1]["tick"]) < gap:
                engagements[-1].append(row)
            else:
                engagements.append([row])
        sa, sv = ctx.side(r, att), ctx.side(r, vic)
        for eng in engagements:
            t0, t1 = int(eng[0]["tick"]), int(eng[-1]["tick"])
            params = {
                "damage_total": int(sum(_first_valid(e["dmg_health"], 0) for e in eng)),
                "hits":         len(eng),
                "weapon_class": _mode([weapon_class(_nan(e["weapon"])) for e in eng]),
                "armor_damage": int(sum(_first_valid(e["dmg_armor"], 0) for e in eng)),
                "victim_id":    ctx.pid(vic),
            }
            subtype = "team" if sa is not None and sa == sv else None
            a = ctx.instant(r, "damage", att, t0, params, subtype=subtype)
            if a is not None:
                a.window_start_tick, a.window_end_tick = t0, t1
                out.append(a)
    out.sort(key=lambda a: (a.round_num, a.tick, a.actor_id_hash, a.params["victim_id"]))
    return ctx.unique(out, lambda a: a.params["victim_id"])
