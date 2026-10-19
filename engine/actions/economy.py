"""
Round-level actions: per-player buy and team_buy (P4.4.1), drop/give to a teammate
(P4.4.2).

Buy: one per roster player at `freeze_end`, read from the player's tick row at
freeze_end (the last row at or before it). `loadout_class` from `actions.buy_bands`,
whose values are EXCLUSIVE upper bounds checked in order:

    equip_value <  pistol            -> "pistol"
    pistol <= v <  eco               -> "eco"
    eco    <= v <  half              -> "half"
    half   <= v <  force             -> "force"
    force  <= v                      -> "full"      (so 4500 is full with force = 4500)

Params: loadout_class, equip_value, money_before (money at the round's first tick
row: before buying), money_after (money at freeze_end), has_armor (armor > 0),
has_helmet, has_kit, utility_count (grenades in the inventory), primary_class (class
of the best weapon in the inventory, PRIMARY_ORDER; "pistol" / "knife" when that is all
the player has; None when the inventory is unknown). `team_buy` per side (no actor,
salted by side): team_class = the most common loadout_class, ties broken toward the
higher band; total_equip. Kind round, tick = freeze_end; `state_pre_tick` and
`state_post_tick` are both freeze_end: the post state is the freeze-end snapshot and
the pre state is the economy-only prior built FROM that snapshot (Appendix B.6), which
has no tick of its own.

Drop/give: an `items` drop by a player who is still alive one tick later (a death drop
is not a give), picked up by a teammate (not the dropper) within GIVE_WINDOW_S; the
first such pickup of the same item, each pickup used once. C4 is excluded (bomb_drop
covers it). One instant action at the drop tick, actor = dropper. Params: weapon_class,
receiver_id, plus pickup_tick (also handed to the builder as a boundary tick).
`state_pre_tick = drop - 1`, `state_post_tick = pickup + 1`.
"""
from engine.actions.types import Action, register
from engine.state.players import _nan, grenade_counts, short_name, weapon_class

# ── constants ─────────────────────────────────────────────────────────────────

BAND_ORDER    = ("pistol", "eco", "half", "force", "full")
PRIMARY_ORDER = ("sniper", "rifle", "lmg", "smg", "shotgun", "pistol", "knife")
GIVE_WINDOW_S = 5.0      # s, P4.4.2


def loadout_class(equip_value, bands) -> str | None:
    """Band of an equipment value (bands are exclusive upper bounds, see module doc)."""
    v = _nan(equip_value)
    if v is None:
        return None
    for name in BAND_ORDER[:-1]:
        if v < getattr(bands, name):
            return name
    return BAND_ORDER[-1]


def primary_class(inventory) -> str | None:
    if inventory is None:
        return None
    classes = {weapon_class(i) for i in inventory}
    return next((c for c in PRIMARY_ORDER if c in classes), None)


# ── buys (P4.4.1) ─────────────────────────────────────────────────────────────

@register("buy")
@register("team_buy")
def extract_buys(raw, states, ctx) -> list[Action]:
    bands = ctx.cfg.actions.buy_bands
    out: list[Action] = []
    for r in sorted(ctx.sides):
        fe = ctx.freeze_end(r)
        per_side: dict[str, list[Action]] = {"ct": [], "t": []}
        for side in ("ct", "t"):
            for sid in sorted(s for s, sd in ctx.sides[r].items() if sd == side):
                tr = ctx.track(r, sid)
                k = tr.idx(fe)
                if k is None or not tr.alive(fe):
                    continue
                inv = tr.col("inventory")[k] if "inventory" in tr.df.columns else None
                counts = grenade_counts(inv)
                equip = tr.at("equipment_value", fe)
                m0, m1 = tr.at("money", int(tr.tick[0])), tr.at("money", fe)
                armor = tr.at("armor", fe)
                helmet, kit = tr.at("has_helmet", fe), tr.at("has_defuser", fe)
                params = {
                    "loadout_class": loadout_class(equip, bands),
                    "equip_value":   None if equip is None else int(equip),
                    "money_before":  None if m0 is None else int(m0),
                    "money_after":   None if m1 is None else int(m1),
                    "has_armor":     None if armor is None else bool(armor > 0),
                    "has_helmet":    None if helmet is None else bool(helmet),
                    "has_kit":       None if kit is None else bool(kit),
                    "utility_count": None if counts is None else int(sum(counts.values())),
                    "primary_class": primary_class(inv),
                }
                a = ctx.instant(r, "buy", sid, fe, params, side=side)
                if a is not None:
                    a.state_pre_tick = a.state_post_tick = fe
                    per_side[side].append(a)
                    out.append(a)
        for side in ("ct", "t"):
            buys = per_side[side]
            if not buys:
                continue
            classes = [b.params["loadout_class"] for b in buys if b.params["loadout_class"] is not None]
            team_class = None
            if classes:
                team_class = max(BAND_ORDER, key=lambda c: (classes.count(c), BAND_ORDER.index(c)))
            equips = [b.params["equip_value"] for b in buys]
            tb = Action(match_key=ctx.match_key, round_num=r, actor_id_hash=None, actor_side=side,
                        kind="round", type="team_buy", tick=fe, state_pre_tick=fe, state_post_tick=fe,
                        params={"team_class": team_class,
                                "total_equip": None if None in equips else int(sum(equips))})
            out.append(tb)
    return ctx.unique(out, lambda a: a.actor_side)


# ── drop / give (P4.4.2) ──────────────────────────────────────────────────────

@register("drop")
def extract_drops(raw, states, ctx) -> list[Action]:
    win = int(round(GIVE_WINDOW_S * ctx.tick_rate))
    items = raw.items.sort(["tick", "steamid"], nulls_last=True, maintain_order=True).to_dicts()
    pickups = [(i, it) for i, it in enumerate(items) if it["event"] == "pickup"]
    used: set[int] = set()
    out: list[Action] = []
    for it in items:
        if it["event"] != "drop":
            continue
        tick, sid, item = _nan(it["tick"]), _nan(it["steamid"]), short_name(_nan(it["item"]))
        r = ctx.round_of(tick)
        if tick is None or sid is None or r is None or item in (None, "c4", "planted_c4"):
            continue
        side = ctx.side(r, sid)
        if side is None or not ctx.alive(r, sid, tick + 1):
            continue
        for i, p in pickups:
            q = _nan(p["steamid"])
            if (i in used or q is None or q == sid or not (tick < p["tick"] <= tick + win)
                    or short_name(_nan(p["item"])) != item or ctx.side(r, q) != side):
                continue
            used.add(i)
            a = ctx.instant(r, "drop", sid, tick, {"weapon_class": weapon_class(item),
                                                  "receiver_id": ctx.pid(q),
                                                  "pickup_tick": int(p["tick"])})
            if a is not None:
                a.state_post_tick = int(p["tick"]) + ctx.off
                out.append(a)
            break
    return ctx.unique(out, lambda a: a.params["receiver_id"])

